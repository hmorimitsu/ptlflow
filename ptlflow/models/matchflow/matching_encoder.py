import torch
import torch.nn as nn
from einops.einops import rearrange
import math
import copy
from timm.layers import DropPath, trunc_normal_
from .quadtree_attention import QuadtreeAttention
from .common import torch_init_model
from .resnet_fpn import ResNetFPN_8_2


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dwconv(x, H, W)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class QuadtreeBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        topks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        scale=1,
        attn_type="B",
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = QuadtreeAttention(
            dim,
            num_heads=num_heads,
            topks=topks,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale=scale,
            attn_type=attn_type,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, "init"):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, target, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(target), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, layer_names, topks=[16, 8, 8], d_model=256):
        super(LocalFeatureTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = 8
        self.layer_names = layer_names

        encoder_layer = QuadtreeBlock(256, 8, attn_type="B", topks=topks, scale=3)
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if "temp" in name or "radius_offset" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, H=0, W=0, pos0=0, pos1=0):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(
            2
        ), "the feature number of src and transformer must be equal"
        if len(feat0.shape) == 4:
            B, C, H, W = feat0.shape
            feat0 = rearrange(feat0, "b c h w -> b (h w) c")
            feat1 = rearrange(feat1, "b c h w -> b (h w) c")
        ii = -1
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, H, W)
                feat1 = layer(feat1, feat1, H, W)
            elif name == "cross":
                feat0, feat1 = layer(feat0, feat1, H, W), layer(feat1, feat0, H, W)
            else:
                raise KeyError
        return feat0, feat1


class PositionEncodingSineNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / (d_model // 2))
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        self.eval_reso = None

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x, train_reso=None, eval_reso=None):
        """
        Args:
            x: [N, C, H, W]
        """
        if self.pe.dtype != x.dtype:
            self.pe.to(dtype=x.dtype)

        if train_reso is None and eval_reso is None:
            return (
                x + self.pe[:, :, : x.size(2), : x.size(3)],
                self.pe[:, :, : x.size(2), : x.size(3)],
            )
        elif eval_reso != self.eval_reso:
            self.eval_reso = eval_reso
            pe = torch.zeros(
                (self.d_model, *self.max_shape), dtype=x.dtype, device=x.device
            )
            y_position = (
                torch.ones(self.max_shape, dtype=x.dtype, device=x.device)
                .cumsum(0)
                .unsqueeze(0)
                * train_reso[0]
                / eval_reso[0]
            )
            x_position = (
                torch.ones(self.max_shape, dtype=x.dtype, device=x.device)
                .cumsum(1)
                .unsqueeze(0)
                * train_reso[1]
                / eval_reso[1]
            )

            div_term = torch.exp(
                torch.arange(0, self.d_model // 2, 2, dtype=x.dtype, device=x.device)
                * (-math.log(10000.0) / (self.d_model // 2))
            )
            div_term = div_term[:, None, None]  # [C//4, 1, 1]
            pe[0::4, :, :] = torch.sin(x_position * div_term)
            pe[1::4, :, :] = torch.cos(x_position * div_term)
            pe[2::4, :, :] = torch.sin(y_position * div_term)
            pe[3::4, :, :] = torch.cos(y_position * div_term)
            pe = pe.unsqueeze(0).to(x.device)
            self.register_buffer("eval_pe", pe, persistent=False)  # [1, C, H, W]

            return (
                x + pe[:, :, : x.size(2), : x.size(3)],
                pe[:, :, : x.size(2), : x.size(3)],
            )
        else:
            if self.eval_pe.dtype != x.dtype:
                self.eval_pe.to(dtype=x.dtype)

            return (
                x + self.eval_pe[:, :, : x.size(2), : x.size(3)],
                self.eval_pe[:, :, : x.size(2), : x.size(3)],
            )


class MatchingModel(nn.Module):
    def __init__(self, cfg, train_size=(384, 512)):
        super().__init__()
        self.cfg = cfg
        self.image_size = train_size
        self.backbone = ResNetFPN_8_2(cfg)

        self.pos_encoding = PositionEncodingSineNorm(256)

        self.loftr_coarse = LocalFeatureTransformer(
            layer_names=[
                "self",
                "cross",
                "self",
                "cross",
                "self",
                "cross",
                "self",
                "cross",
            ],
            topks=[16, 8, 8],
        )

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        # input x in [-1, 1]
        x = (x + 1) / 2.0
        feats_c = self.backbone(x)
        B, _, H, W = feats_c.shape

        (feat_c0, feat_c1) = feats_c.split(B // 2)

        # 2. coarse-level loftr module
        if self.training:
            feat_c0, pos_encoding0 = self.pos_encoding(feat_c0, None, None)
            feat_c1, pos_encoding1 = self.pos_encoding(feat_c1, None, None)
        else:
            feat_c0, pos_encoding0 = self.pos_encoding(
                feat_c0, self.image_size, x.shape[2:4]
            )
            feat_c1, pos_encoding1 = self.pos_encoding(
                feat_c1, self.image_size, x.shape[2:4]
            )

        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None  # mask is useful in training
        feat_c0, feat_c1 = self.loftr_coarse(
            feat_c0,
            feat_c1,
            mask_c0,
            mask_c1,
            H=H,
            W=W,
            pos0=pos_encoding0,
            pos1=pos_encoding1,
        )

        feat_c0 = rearrange(feat_c0, "n (h w) c -> n c h w", h=H)
        feat_c1 = rearrange(feat_c1, "n (h w) c -> n c h w", h=H)

        output = torch.cat([feat_c0, feat_c1], dim=0)

        if is_list:
            output = torch.split(output, [batch_dim, batch_dim], dim=0)

        return output

    def load_state_dict(self, path=""):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                if "radius_offset" in k:
                    state_dict.pop(k)
                else:
                    state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)

        for k in list(state_dict.keys()):
            if (
                k.startswith("backbone.")
                or k.startswith("pos_encoding.")
                or k.startswith("loftr_coarse.")
            ):
                continue
            else:
                state_dict.pop(k)

        torch_init_model(self, state_dict, key="model")
        self.backbone.reset_model()
        return self

    def reset_model(self):
        self.backbone.reset_model()
        return self
