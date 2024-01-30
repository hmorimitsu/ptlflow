import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange

from ...utils import coords_grid
from .attention import (
    MultiHeadAttention,
    LinearPositionEmbeddingSine,
    ExpPositionEmbeddingSine,
)
from ..encoders import twins_svt_large, convnext_large
from .cnn import BasicEncoder

from ...timm0412.models.layers import DropPath


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, embed_dim=64, pe="linear", cfg=None):
        super().__init__()
        self.patch_size = patch_size
        self.dim = embed_dim
        self.pe = pe
        self.cfg = cfg

        if cfg.patch_embed == "no_relu":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 4, kernel_size=6, stride=2, padding=2),
                nn.Conv2d(
                    embed_dim // 4, embed_dim // 2, kernel_size=6, stride=2, padding=2
                ),
                nn.Conv2d(
                    embed_dim // 2, embed_dim, kernel_size=6, stride=2, padding=2
                ),
            )
        elif cfg.patch_embed == "single":
            # assert patch_size == 8
            if patch_size == 8:
                self.proj = nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_chans, embed_dim // 4, kernel_size=6, stride=2, padding=2
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            embed_dim // 4,
                            embed_dim // 2,
                            kernel_size=6,
                            stride=2,
                            padding=2,
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            embed_dim // 2,
                            embed_dim,
                            kernel_size=6,
                            stride=2,
                            padding=2,
                        ),
                    ]
                )
            elif patch_size == 4:
                self.proj = nn.Sequential(
                    nn.Conv2d(
                        in_chans, embed_dim // 4, kernel_size=6, stride=2, padding=2
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        embed_dim // 4, embed_dim, kernel_size=6, stride=2, padding=2
                    ),
                )
            else:
                print(f"patch size = {patch_size} is unacceptable.")

        self.ffn_with_coord = nn.Sequential(
            nn.Conv2d(embed_dim + 64, embed_dim + 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim + 64, embed_dim + 64, kernel_size=1),
        )
        self.norm = nn.LayerNorm(embed_dim + 64)

    def forward(
        self, x, mask_for_patch1=None, mask_for_patch2=None, mask_for_patch3=None
    ):
        B, C, H, W = x.shape  # C == 1

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        masks = [mask_for_patch1, mask_for_patch2, mask_for_patch3]
        for idx, layer in enumerate(self.proj):
            if idx % 2 == 0:
                if masks[idx // 2] is not None:
                    x = x * (1 - masks[idx // 2])
            x = layer(x)

        out_size = x.shape[2:]

        patch_coord = (
            coords_grid(B, out_size[0], out_size[1], dtype=x.dtype, device=x.device)
            * self.patch_size
            + self.patch_size / 2
        )  # in feature coordinate space
        if self.cfg.use_rpe:
            center_coord = coords_grid(1, H, W, dtype=x.dtype, device=x.device)
            center_coord = (
                center_coord.permute(2, 3, 1, 0)
                .reshape(H * W, 2, 1, 1)
                .repeat(B // (H * W), 1, 1, 1)
            )
            patch_coord = patch_coord - center_coord

        patch_coord = patch_coord.view(B, 2, -1).permute(0, 2, 1)
        if self.pe == "linear":
            patch_coord_enc = LinearPositionEmbeddingSine(patch_coord, dim=64)
        elif self.pe == "exp":
            patch_coord_enc = ExpPositionEmbeddingSine(patch_coord, dim=64)
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(
            B, -1, out_size[0], out_size[1]
        )

        x_pe = torch.cat([x, patch_coord_enc], dim=1)
        x = self.ffn_with_coord(x_pe)

        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, out_size


from .twins import Block, CrossBlock


class VerticalSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        cfg,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
    ):
        super(VerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = cfg.droppath
        drop_rate = dropout
        attn_drop_rate = 0.0

        self.local_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            sr_ratio=sr_ratio,
            ws=ws,
            with_rpe=True,
            vert_c_dim=cfg.vert_c_dim,
            encoder_latent_dim=cfg.encoder_latent_dim,
        )
        self.global_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            sr_ratio=sr_ratio,
            ws=1,
            with_rpe=True,
            vert_c_dim=cfg.vert_c_dim,
            encoder_latent_dim=cfg.encoder_latent_dim,
        )

    def forward(self, x, size, context=None):
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)

        return x


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        cfg,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
    ):
        super(SelfAttentionLayer, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        proj_drop = drop_path = cfg.droppath

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = (
            nn.Linear(dim, dim, bias=True),
            nn.Linear(dim, dim, bias=True),
            nn.Linear(dim, dim, bias=True),
        )

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class CostPerceiverEncoder(nn.Module):
    def __init__(self, cfg):
        super(CostPerceiverEncoder, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.patch_embed = PatchEmbed(
            in_chans=self.cfg.cost_heads_num,
            patch_size=self.patch_size,
            embed_dim=cfg.cost_latent_input_dim,
            pe=cfg.pe,
            cfg=self.cfg,
        )

        self.depth = cfg.encoder_depth

        self.latent_tokens = nn.Parameter(
            torch.randn(1, cfg.cost_latent_token_num, cfg.cost_latent_dim)
        )

        query_token_dim, tgt_token_dim = (
            cfg.cost_latent_dim,
            cfg.cost_latent_input_dim * 2,
        )
        qk_dim, v_dim = query_token_dim, query_token_dim

        if cfg.cross_attn == "all":
            from .crossattentionlayer import CrossAttentionLayer

            self.input_layer = CrossAttentionLayer(
                qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout
            )
        elif cfg.cross_attn == "part":
            from .crossattentionlayer import CrossAttentionLayer_two_level

            self.input_layer = CrossAttentionLayer_two_level(
                qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout
            )
        elif cfg.cross_attn == "rep":
            from .crossattentionlayer import CrossAttentionLayer_two_level_rep

            self.input_layer = CrossAttentionLayer_two_level_rep(
                qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout
            )
        elif cfg.cross_attn == "k3s2":
            from .crossattentionlayer import CrossAttentionLayer_convk3s2

            self.input_layer = CrossAttentionLayer_convk3s2(
                qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout
            )
        elif cfg.cross_attn == "34":
            print("[Using 34 crossattention layer]")
            from .crossattentionlayer import CrossAttentionLayer_34

            self.input_layer = CrossAttentionLayer_34(
                qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout
            )

        self.encoder_layers = nn.ModuleList(
            [
                SelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout)
                for idx in range(self.depth)
            ]
        )

        if self.cfg.vertical_encoder_attn == "twins":
            self.vertical_encoder_layers = nn.ModuleList(
                [
                    VerticalSelfAttentionLayer(
                        cfg.cost_latent_dim, cfg, dropout=cfg.dropout
                    )
                    for idx in range(self.depth)
                ]
            )
        elif self.cfg.vertical_encoder_attn == "NA":
            from .NA import selfattentionlayer_nat

            self.vertical_encoder_layers = nn.ModuleList(
                [selfattentionlayer_nat(cfg) for idx in range(self.depth)]
            )
        elif self.cfg.vertical_encoder_attn == "NA-twins":
            from .NA import NATwins

            assert cfg.vert_c_dim > 0, "Only support vert_c_dim>0 for NA-Twins"
            print("[Using NA-twins vertical attention layers]")
            self.vertical_encoder_layers = nn.ModuleList(
                [NATwins(cfg) for idx in range(self.depth)]
            )

    def random_masking(self, x, mask_ratio, mask=None):
        B, _, H, W = x.shape

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        H = H + pad_b
        W = W + pad_r

        # number of keys for crossattentionlayer
        H_down = H // 8
        W_down = W // 8

        L = H_down * W_down
        len_keep = int(L * (1 - mask_ratio))

        if mask is not None:
            noise = mask.reshape(B, L)
        else:
            # print("random mask")
            noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_for_keys = torch.gather(mask, dim=1, index=ids_restore)

        mask_for_patch1 = (
            mask_for_keys.reshape(-1, H_down, W_down)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 64)
            .reshape(-1, H_down, W_down, 8, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, H_down * 8, W_down * 8)
            .unsqueeze(1)
        )
        mask_for_patch2 = (
            mask_for_keys.reshape(-1, H_down, W_down)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 16)
            .reshape(-1, H_down, W_down, 4, 4)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, H_down * 4, W_down * 4)
            .unsqueeze(1)
        )
        mask_for_patch3 = (
            mask_for_keys.reshape(-1, H_down, W_down)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 4)
            .reshape(-1, H_down, W_down, 2, 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, H_down * 2, W_down * 2)
            .unsqueeze(1)
        )

        return (
            ids_keep,
            mask_for_keys,
            mask_for_patch1,
            mask_for_patch2,
            mask_for_patch3,
            ids_restore,
        )

    def forward(self, cost_volume, data, context=None):
        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = (
            cost_volume.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
        )
        data["cost_maps"] = cost_maps

        x, size = self.patch_embed(cost_maps)  # B*H1*W1, size[0]*size[1], C
        data["H3W3"] = size
        H3, W3 = size

        cost_patches = x

        x = self.input_layer(self.latent_tokens, x, size)

        short_cut = x

        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.cfg.vertical_encoder_attn is not None:
                x = (
                    x.view(B, H1 * W1, self.cfg.cost_latent_token_num, -1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * self.cfg.cost_latent_token_num, H1 * W1, -1)
                )
                x = self.vertical_encoder_layers[idx](x, (H1, W1), context)
                x = (
                    x.view(B, self.cfg.cost_latent_token_num, H1 * W1, -1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * H1 * W1, self.cfg.cost_latent_token_num, -1)
                )

        if self.cfg.cost_encoder_res is True:
            x = x + short_cut

        _B, _HW, _C = cost_patches.shape
        cost_patches = cost_patches.reshape(_B, H3, W3, _C).permute(0, 3, 1, 2)

        return x, cost_patches

    def pretrain_forward(
        self, cost_volume_outter, cost_volume, data, context=None, mask=None
    ):
        B, heads, H1, W1, H2, W2 = cost_volume_outter.shape
        cost_maps = (
            cost_volume_outter.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
        )
        data["cost_maps_outter"] = cost_maps

        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = (
            cost_volume.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
        )
        data["cost_maps"] = cost_maps

        (
            ids_keep,
            mask_for_keys,
            mask_for_patch1,
            mask_for_patch2,
            mask_for_patch3,
            ids_restore,
        ) = self.random_masking(cost_maps, self.cfg.mask_ratio, mask)
        # ids_keep = mask_for_keys = mask_for_patch1 = mask_for_patch2 = mask_for_patch3 = ids_restore = None

        x, size = self.patch_embed(
            cost_maps, mask_for_patch1, mask_for_patch2, mask_for_patch3
        )  # B*H1*W1, size[0]*size[1], C

        data["H3W3"] = size
        H3, W3 = size

        cost_patches = x

        x = self.input_layer(self.latent_tokens, x, size, ids_keep)

        short_cut = x

        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.cfg.vertical_encoder_attn is not None:
                x = (
                    x.view(B, H1 * W1, self.cfg.cost_latent_token_num, -1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * self.cfg.cost_latent_token_num, H1 * W1, -1)
                )
                x = self.vertical_encoder_layers[idx](x, (H1, W1), context)
                x = (
                    x.view(B, self.cfg.cost_latent_token_num, H1 * W1, -1)
                    .permute(0, 2, 1, 3)
                    .reshape(B * H1 * W1, self.cfg.cost_latent_token_num, -1)
                )

        if self.cfg.cost_encoder_res is True:
            x = x + short_cut

        _B, _HW, _C = cost_patches.shape
        cost_patches = cost_patches.reshape(_B, H3, W3, _C).permute(0, 3, 1, 2)

        return x, cost_patches


class MemoryEncoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg

        if cfg.fnet == "twins":
            self.feat_encoder = twins_svt_large(
                pretrained=self.cfg.pretrain, del_layers=cfg.del_layers
            )
        elif cfg.fnet == "basicencoder":
            self.feat_encoder = BasicEncoder(output_dim=256, norm_fn="instance")
        elif cfg.fnet == "convnext":
            self.feat_encoder = convnext_large(pretrained=self.cfg.pretrain)
        else:
            exit()

        if cfg.pretrain_mode:
            print("[In pretrain mode, freeze feature encoder]")
            for param in self.feat_encoder.parameters():
                param.requires_grad = False

        if cfg.use_convertor:
            self.channel_convertor = nn.Conv2d(
                cfg.encoder_latent_dim, 256, 1, padding=0, bias=False
            )

        self.cost_perceiver_encoder = CostPerceiverEncoder(cfg)

        if self.cfg.pretrain_mode and self.cfg.crop_cost_volume:
            print(
                "[H_offset is {}, W_offset is {}, and crop_cost_volume to get inner cost volume]".format(
                    self.cfg.H_offset, self.cfg.W_offset
                )
            )

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape

        fmap1 = rearrange(
            fmap1, "b (heads d) h w -> b heads (h w) d", heads=self.cfg.cost_heads_num
        )
        fmap2 = rearrange(
            fmap2, "b (heads d) h w -> b heads (h w) d", heads=self.cfg.cost_heads_num
        )
        corr = einsum("bhid, bhjd -> bhij", fmap1, fmap2)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht2, wd2)

        return corr

    def corr_16(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        _, _, ht2, wd2 = fmap2.shape

        # fmap1 = F.avg_pool2d(fmap1, kernel_size=2, stride=2, padding=0)
        fmap1 = fmap1[:, :, ::2, ::2]

        fmap1 = rearrange(
            fmap1, "b (heads d) h w -> b heads (h w) d", heads=self.cfg.cost_heads_num
        )
        fmap2 = rearrange(
            fmap2, "b (heads d) h w -> b heads (h w) d", heads=self.cfg.cost_heads_num
        )
        corr = einsum("bhid, bhjd -> bhij", fmap1, fmap2)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht // 2, wd // 2, ht2, wd2)

        return corr

    def forward(self, img1, img2, data, context=None):
        feat_s, _ = self.feat_encoder(img1)
        feat_t, _ = self.feat_encoder(img2)

        feat_s_16 = None
        feat_t_16 = None

        B, C, H, W = feat_s.shape
        size = (H, W)

        if self.cfg.use_convertor:
            feat_s = self.channel_convertor(feat_s)
            feat_t = self.channel_convertor(feat_t)

        cost_volume = self.corr(feat_s, feat_t)
        if self.cfg.r_16 > 0:
            cost_volume_16 = self.corr_16(feat_s_16, feat_t_16)
            B, heads, H1, W1, H2, W2 = cost_volume_16.shape
            cost_maps = (
                cost_volume_16.permute(0, 2, 3, 1, 4, 5)
                .contiguous()
                .view(B * H1 * W1, self.cfg.cost_heads_num, H2, W2)
            )
            data["cost_maps_16"] = cost_maps

        x, cost_patches = self.cost_perceiver_encoder(cost_volume, data, context)

        return x, cost_patches, feat_s_16, feat_t_16

    def pretrain_forward(
        self, img1, img2, img1_inner, img2_inner, data, context=None, mask=None
    ):
        feat_t, _ = self.feat_encoder(img2)

        feat_s_inner, _ = self.feat_encoder(img1_inner)
        feat_t_inner, _ = self.feat_encoder(img2_inner)

        cost_volume = self.corr(feat_s_inner, feat_t)
        if self.cfg.crop_cost_volume:
            H_border = self.cfg.H_offset // 8
            W_border = self.cfg.W_offset // 8
            cost_volume_inner = cost_volume[
                :, :, :, :, H_border:-H_border, W_border:-W_border
            ]
        else:
            cost_volume_inner = self.corr(feat_s_inner, feat_t_inner)
        x, cost_patches = self.cost_perceiver_encoder.pretrain_forward(
            cost_volume, cost_volume_inner, data, context, mask=mask
        )

        return x, cost_patches
