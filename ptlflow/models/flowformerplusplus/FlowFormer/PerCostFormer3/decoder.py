import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from ...utils import coords_grid, bilinear_sampler
from .attention import (
    MultiHeadAttention,
    LinearPositionEmbeddingSine,
    ExpPositionEmbeddingSine,
)

from timm.layers import DropPath

from .gru import BasicUpdateBlock, GMAUpdateBlock
from .gma import Attention
from .sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder


def initialize_flow(img):
    """Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W, dtype=img.dtype, device=img.device)
    mean_init = coords_grid(N, H, W, dtype=img.dtype, device=img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class CrossAttentionLayer(nn.Module):
    # def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
    def __init__(
        self,
        qk_dim,
        v_dim,
        query_token_dim,
        tgt_token_dim,
        flow_or_pe,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
        pe="linear",
        no_sc=False,
    ):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim**-0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = (
            nn.Linear(query_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, v_dim, bias=True),
        )

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout),
        )
        self.flow_or_pe = flow_or_pe
        print("[Decoder flow_or_pe setting is: {}]".format(self.flow_or_pe))
        self.no_sc = no_sc
        if self.no_sc:
            print("[No short cut in cost decoding]")
        self.dim = qk_dim

    def forward(self, query, key, value, memory, query_coord, patch_size, size_h3w3):
        """
        query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = (
            query_coord.view(B, 2, -1)
            .permute(0, 2, 1)[:, :, None, :]
            .contiguous()
            .view(B * H1 * W1, 1, 2)
        )
        if self.pe == "linear":
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == "exp":
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == "norm_linear":
            query_coord[:, :, 0:1] = query_coord[:, :, 0:1] / W1
            query_coord[:, :, 1:2] = query_coord[:, :, 1:2] / H1
            query_coord_enc = LinearPositionEmbeddingSine(
                query_coord, dim=self.dim, NORMALIZE_FACOR=2
            )

        short_cut = query
        if query is not None:
            query = self.norm1(query)

        if self.flow_or_pe == "and":
            q = self.q(query + query_coord_enc)
        elif self.flow_or_pe == "pe":
            q = self.q(query_coord_enc)
        elif self.flow_or_pe == "flow":
            q = self.q(query)
        else:
            print("[Wrong setting of flow_or_pe]")
            exit()
        k, v = key, value

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        # x = self.proj(torch.cat([x, short_cut],dim=2))
        if short_cut is not None and not self.no_sc:
            # print("short cut")
            x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v


class MemoryDecoderLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size  # for converting coords into H2', W2' space

        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim

        self.cross_attend = CrossAttentionLayer(
            qk_dim,
            v_dim,
            query_token_dim,
            tgt_token_dim,
            flow_or_pe=cfg.flow_or_pe,
            dropout=cfg.dropout,
            pe=cfg.pe,
            no_sc=cfg.no_sc,
        )

    def forward(self, query, key, value, memory, coords1, size, size_h3w3):
        """
        x:      [B*H1*W1, 1, C]
        memory: [B*H1*W1, H2'*W2', C]
        coords1 [B, 2, H2, W2]
        size: B, C, H1, W1
        1. Note that here coords0 and coords1 are in H2, W2 space.
           Should first convert it into H2', W2' space.
        2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(
            query, key, value, memory, coords1, self.patch_size, size_h3w3
        )
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v


class MemoryDecoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        if cfg.use_patch:
            print("[Using cost patch as local cost]")
            self.flow_token_encoder = nn.Conv2d(
                cfg.cost_latent_input_dim + 64, cfg.query_latent_dim, 1, 1
            )
        else:
            self.flow_token_encoder = nn.Sequential(
                nn.Conv2d(81 * cfg.cost_heads_num, dim, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1),
            )

        if self.cfg.fix_pe:
            print("[fix_pe: regress 8*8 block]")
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, 64, 1, 1),
            )
        elif self.cfg.gt_r > 0:
            print("[Using larger cost as gt, radius is {}]".format(self.cfg.gt_r))
            # self.pretrain_head = nn.Conv2d(dim, self.cfg.gt_r**2, 1, 1)
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                # nn.Conv2d(dim*2, dim*2, 1, 1),
                # nn.GELU(),
                nn.Conv2d(dim * 2, self.cfg.gt_r**2, 1, 1),
            )
        else:
            self.pretrain_head = nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim * 2, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim * 2, 81, 1, 1),
            )

        self.proj = nn.Conv2d(cfg.encoder_latent_dim, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(dim, cfg)

        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(
                args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128
            )
        elif self.cfg.gma == "GMA-SK":
            print("[Using GMA-SK]")
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128
            )
            self.att = Attention(
                args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128
            )
        else:
            print("[Not using GMA decoder]")
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)

        if self.cfg.r_16 > 0:
            print("[r_16 = {}]".format(self.cfg.r_16))

        if self.cfg.quater_refine:
            print("[Using Quater Refinement]")
            from .quater_upsampler import quater_upsampler

            self.quater_upsampler = quater_upsampler()

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def sample_feature_map(self, coords, feat_t_quater, r=1):
        H, W = feat_t_quater.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(feat_t_quater, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.to(dtype=coords.dtype)

        return img

    def encode_flow_token(self, cost_maps, coords, r=4):
        """
        cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
        coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        dx = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

        centroid = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)

        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def forward(
        self,
        cost_memory,
        context,
        context_quater,
        feat_s_quater,
        feat_t_quater,
        data={},
        prev_flow=None,
        cost_patches=None,
    ):
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_maps = data["cost_maps"]
        coords0, coords1 = initialize_flow(context)

        if prev_flow is not None:
            forward_flow = forward_interpolate_batch(prev_flow)
            coords1 = coords1 + forward_flow

        # flow = coords1

        flow_predictions = []

        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        if self.cfg.gma is not None:
            attention = self.att(inp)

        size = net.shape
        key, value = None, None

        for idx in range(self.depth):
            coords1 = coords1.detach()

            cost_forward = self.encode_flow_token(cost_maps, coords1)

            if self.cfg.use_patch:
                if self.cfg.detach_local:
                    _local_cost = self.encode_flow_token(
                        cost_patches, coords1 / 8.0, r=0
                    )
                    _local_cost = _local_cost.contiguous().detach()
                    query = self.flow_token_encoder(_local_cost)
                else:
                    query = self.flow_token_encoder(
                        self.encode_flow_token(cost_patches, coords1 / 8.0, r=0)
                    )
            else:
                if self.cfg.detach_local:
                    _local_cost = cost_forward.contiguous().detach()
                    query = self.flow_token_encoder(_local_cost)
                else:
                    query = self.flow_token_encoder(cost_forward)
            query = (
                query.permute(0, 2, 3, 1)
                .contiguous()
                .view(size[0] * size[2] * size[3], 1, self.dim)
            )

            if self.cfg.use_rpe:
                query_coord = coords1 - coords0
            else:
                query_coord = coords1
            cost_global, key, value = self.decoder_layer(
                query, key, value, cost_memory, query_coord, size, data["H3W3"]
            )

            if self.cfg.r_16 > 0:
                cost_forward_16 = self.encode_flow_token(
                    data["cost_maps_16"], coords1 * 2.0, r=(self.cfg.r_16 - 1) // 2
                )

                corr = torch.cat([cost_global, cost_forward, cost_forward_16], dim=1)
            else:
                corr = torch.cat([cost_global, cost_forward], dim=1)

            flow = coords1 - coords0

            if self.cfg.gma is not None:
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow, attention
                )
            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # flow = delta_flow
            coords1 = coords1 + delta_flow

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if self.cfg.quater_refine:
            coords1 = coords1.detach()
            new_size = context_quater.shape[-2:]
            flow = 2 * F.interpolate(
                coords1 - coords0, size=new_size, mode="bilinear", align_corners=True
            )
            flow_up = self.quater_upsampler(
                flow, context_quater, feat_s_quater, feat_t_quater, r=1
            )
            flow_predictions.append(flow_up)

        return flow_predictions, coords1 - coords0

    def pretrain_forward(
        self,
        cost_memory,
        context,
        data={},
        prev_flow=None,
        cost_patches=None,
        mask_for_patch1=None,
    ):
        cost_maps_outter = data["cost_maps_outter"].detach()
        cost_maps = data["cost_maps"].detach()

        _, _, H_outter, W_outter = cost_maps_outter.shape
        Bs, _, H_inner, W_inner = cost_maps.shape
        # print(H_inner,W_inner,H_outter,W_outter, self.cfg.H_offset, self.cfg.W_offset)
        # exit()

        net, inp = torch.split(context, [128, 128], dim=1)

        size = net.shape
        B = size[0]
        key, value = None, None

        loss = 0

        if self.cfg.fix_pe:
            pad_l = pad_t = 0
            pad_r = (8 - W_inner % 8) % 8
            pad_b = (8 - H_inner % 8) % 8
            _H_inner = H_inner + pad_b
            _W_inner = W_inner + pad_r

            # number of keys for crossattentionlayer
            H_down = _H_inner // 8
            W_down = _W_inner // 8

            cost_maps = F.pad(cost_maps, (pad_l, pad_r, pad_t, pad_b))
            cost_maps_patches = F.unfold(cost_maps, kernel_size=8, padding=0, stride=8)
            mean = cost_maps_patches.mean(dim=1, keepdim=True)
            var = cost_maps_patches.var(dim=1, keepdim=True)
            cost_maps_patches = (cost_maps_patches - mean) / (var + 1.0e-6) ** 0.5
            cost_maps_patches = cost_maps_patches.reshape(Bs, 64, H_down, W_down)

            for idx_h, idx_w in zip(range(H_down), range(W_down)):
                query_coord = torch.zeros(
                    B, 2, H_inner, W_inner, device=cost_memory.device
                )
                query_coord[:, 0, :, :] = idx_w
                query_coord[:, 1, :, :] = idx_h
                cost_global, key, value = self.decoder_layer(
                    None,
                    key,
                    value,
                    cost_memory,
                    query_coord.detach(),
                    size,
                    data["H3W3"],
                )
                cost_forward_pred = self.pretrain_head(cost_global)

                target = (
                    cost_maps_patches[:, :, idx_h, idx_w]
                    .reshape(B, H_inner, W_inner, 64)
                    .permute(0, 3, 1, 2)
                )
                loss += ((cost_forward_pred - target) ** 2).mean()
        elif self.cfg.gt_r > 0:
            for idx in range(self.cfg.query_num):
                coords_outter = torch.rand(
                    B, 2, H_inner, W_inner, device=cost_memory.device
                )
                radius = (self.cfg.gt_r - 1) // 2
                if self.cfg.no_border:
                    coords_outter = (
                        torch.cat(
                            [
                                coords_outter[:, 0:1, :, :]
                                * (W_outter - self.cfg.gt_r),
                                coords_outter[:, 1:, :, :] * (H_outter - self.cfg.gt_r),
                            ],
                            dim=1,
                        )
                        + radius
                    )
                else:
                    coords_outter = torch.cat(
                        [
                            coords_outter[:, 0:1, :, :] * W_outter,
                            coords_outter[:, 1:, :, :] * H_outter,
                        ],
                        dim=1,
                    )

                coords_inner = torch.cat(
                    [
                        coords_outter[:, 0:1, :, :] - self.cfg.W_offset // 8,
                        coords_outter[:, 1:, :, :] - self.cfg.H_offset // 8,
                    ],
                    dim=1,
                )

                cost_forward_outter = self.encode_flow_token(
                    cost_maps_outter.detach(), coords_outter.detach(), r=radius
                )

                query_coord = coords_inner
                cost_forward = self.encode_flow_token(cost_maps.detach(), coords_inner)
                query = self.flow_token_encoder(cost_forward)
                query = (
                    query.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(size[0] * size[2] * size[3], 1, self.dim)
                )
                cost_global, key, value = self.decoder_layer(
                    query,
                    key,
                    value,
                    cost_memory,
                    query_coord.detach(),
                    size,
                    data["H3W3"],
                )

                cost_forward_pred = self.pretrain_head(cost_global)
                mean = cost_forward_outter.mean(dim=1, keepdim=True)
                var = cost_forward_outter.var(dim=1, keepdim=True)
                cost_forward_outter = (cost_forward_outter - mean) / (
                    var + 1.0e-6
                ) ** 0.5

                loss += ((cost_forward_pred - cost_forward_outter) ** 2).mean()
        else:
            for idx in range(self.cfg.query_num):
                coords_outter = torch.rand(
                    B, 2, H_inner, W_inner, device=cost_memory.device
                )
                if self.cfg.no_border:
                    coords_outter = (
                        torch.cat(
                            [
                                coords_outter[:, 0:1, :, :] * (W_outter - 8),
                                coords_outter[:, 1:, :, :] * (H_outter - 8),
                            ],
                            dim=1,
                        )
                        + 4.0
                    )
                else:
                    coords_outter = torch.cat(
                        [
                            coords_outter[:, 0:1, :, :] * W_outter,
                            coords_outter[:, 1:, :, :] * H_outter,
                        ],
                        dim=1,
                    )

                coords_inner = torch.cat(
                    [
                        coords_outter[:, 0:1, :, :] - self.cfg.W_offset // 8,
                        coords_outter[:, 1:, :, :] - self.cfg.H_offset // 8,
                    ],
                    dim=1,
                )

                cost_forward_outter = self.encode_flow_token(
                    cost_maps_outter.detach(), coords_outter.detach()
                )

                query_coord = coords_inner
                cost_global, key, value = self.decoder_layer(
                    None,
                    key,
                    value,
                    cost_memory,
                    query_coord.detach(),
                    size,
                    data["H3W3"],
                )

                cost_forward_pred = self.pretrain_head(cost_global)
                mean = cost_forward_outter.mean(dim=1, keepdim=True)
                var = cost_forward_outter.var(dim=1, keepdim=True)
                cost_forward_outter = (cost_forward_outter - mean) / (
                    var + 1.0e-6
                ) ** 0.5

                loss += ((cost_forward_pred - cost_forward_outter) ** 2).mean()

        return loss
