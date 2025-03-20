import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import (
    Aggregate,
    SpatioTemporalAggregate,
    TemporalAggregate,
    TMMAggregate,
)

# from core.models.sk_decoder import vis_featmap

from torch import nn
from einops import rearrange

# from models.gaflow_modules.modules import GGAM


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKBlock_Temporal(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv1d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv1d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv1d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv1d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv1d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x, T=2):
        _, C, H, W = x.shape
        x = rearrange(x, "(b t) c h w -> (b h w) c t", t=T)
        # print(x.shape)
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        x = rearrange(x, "(b h w) c t -> (b t) c h w", h=H, w=W)

        return x


# class SKUpdate_GGAM(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
#         ratio = 16 if args.Encoder == 'UMT' else 8

#         self.gma = args.use_gma
#         embed_dim = args.decoder_dim // 2

#         if self.gma:
#             self.aggregator = GGAM(self.args, 128)
#             self.gru = PCBlock4_Deep_nopool_res(embed_dim*4, embed_dim, k_conv=args.PCUpdater_conv)
#         else:
#             self.gru = PCBlock4_Deep_nopool_res(embed_dim*4, embed_dim, k_conv=args.PCUpdater_conv)

#         self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

#         self.mask = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim*2, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim*2, ratio*ratio*9, 1, padding=0))

#     def forward(self, net, inp, corr, flow, iters):
#         motion_features = self.encoder(flow, corr)

#         if self.gma:
#             motion_features_global = self.aggregator(inp, motion_features, iters)
#             inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
#         else:
#             inp_cat = torch.cat([inp, motion_features], dim=1)
#         # Attentional update
#         net = self.gru(torch.cat([net, inp_cat], dim=1))

#         delta_flow = self.flow_head(net)

#         # scale mask to balence gradients
#         mask = .25 * self.mask(net)
#         return net, mask, delta_flow


class SKBlock_Dilated(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.dilated = nn.Conv2d(
            C_in,
            C_in,
            args.dilated_kernel,
            stride=1,
            padding=2 * (args.dilated_kernel - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + self.dilated(x))
        x = self.ffn2(x)
        return x


class SKBlock_Dilated_v2(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.dilated = nn.Conv2d(
            C_in,
            C_in,
            args.dilated_kernel,
            stride=1,
            padding=2 * (args.dilated_kernel - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        dil_x = F.gelu(self.dilated(x))
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + dil_x)
        x = self.ffn2(x)
        return x


class SKBlock_Dilated_v3(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.dilated = nn.Conv2d(
            C_in,
            C_in,
            args.dilated_kernel,
            stride=1,
            padding=2 * (args.dilated_kernel - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        dil_x = F.gelu(self.dilated(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + dil_x)
        x = self.ffn2(x)
        return x


class SKBlock_Dilated_v4(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.dilated1 = nn.Conv2d(
            C_in,
            C_in,
            9,
            stride=1,
            padding=2 * (9 - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.dilated2 = nn.Conv2d(
            C_in,
            C_in,
            5,
            stride=1,
            padding=2 * (5 - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        dil_x1 = F.gelu(self.dilated1(x))
        dil_x2 = F.gelu(self.dilated2(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + dil_x1 + dil_x2)
        x = self.ffn2(x)
        return x


class SKBlock_Dilated_multi(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.perceptual_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in args.perceptuals
            ]
        )

        self.dilated_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel,
                    stride=1,
                    padding=2 * (kernel - 1) // 2,
                    groups=C_in,
                    dilation=args.dilation_rate,
                )
                for kernel in args.dilated_kernels
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        res = x

        x = F.gelu(x + sum(conv(x) for conv in self.perceptual_list))
        x = F.gelu(x + sum(conv(x) for conv in self.conv_list))
        x = F.gelu(x + self.pw(x))

        x = F.gelu(x + sum(F.gelu(conv(res)) for conv in self.dilated_list))
        x = self.ffn2(x)
        return x


class SKBlock_Dilated_v5(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.aux = nn.Conv2d(C_in, C_in, 1, stride=1, padding=0)
        self.dilated1 = nn.Conv2d(
            C_in,
            C_in,
            9,
            stride=1,
            padding=2 * (9 - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.dilated2 = nn.Conv2d(
            C_in,
            C_in,
            5,
            stride=1,
            padding=2 * (5 - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        dil_x1 = F.gelu(self.dilated1(x))
        dil_x2 = F.gelu(self.dilated2(x))
        x3 = F.gelu(self.aux(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + dil_x1 + dil_x2 + x3)
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, decoder_dim, corr_levels, corr_radius, k_conv):
        super().__init__()
        out_dim = decoder_dim // 2
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2

        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64 + 192, out_dim - 2, k_conv)

    def forward(self, flow, corr, attention=None):
        cor = F.gelu(self.convc1(corr))
        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKMotionEncoder_CorrTemporal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        out_dim = args.decoder_dim // 2
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2

        self.convc1 = PCBlock4_Deep_nopool_res(
            cor_planes, 256 // (args.T - 1), args.k_conv
        )
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192 * (args.T - 1), args.k_conv)

        self.convf1 = nn.Conv2d(2 * (args.T - 1), 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64 * (args.T - 1), args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64 + 192, out_dim - 2, args.k_conv)

    def forward(self, flow, corr, attention=None):
        cor = F.gelu(self.convc1(corr))
        _, C, H, W = cor.shape
        cor = rearrange(cor, "(B T) C H W -> B (T C) H W", T=self.args.T - 1)
        cor = self.convc2(cor)

        cor = rearrange(cor, "B (T C) H W -> (B T) C H W", T=self.args.T - 1)

        flo = self.convf1(
            rearrange(flow, "(B T) C H W -> B (T C) H W", T=self.args.T - 1)
        )
        flo = self.convf2(flo)
        flo = rearrange(flo, "B (T C) H W -> (B T) C H W", T=self.args.T - 1)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKBlock_Dilated_v6(nn.Module):
    def __init__(self, C_in, C_out, k_conv, args=None):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.dilated = nn.Conv2d(
            C_in,
            C_in,
            args.dilated_kernel,
            stride=1,
            padding=2 * (args.dilated_kernel - 1) // 2,
            groups=C_in,
            dilation=args.dilation_rate,
        )
        self.aux = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        dil_x = F.gelu(self.dilated(x))
        aux_x = F.gelu(self.aux(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = F.gelu(x + dil_x + aux_x)
        x = self.ffn2(x)
        return x


class SKUpdateBlock6_Deep_nopoolres_AllDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, attention, ref_nets=None):
        motion_features = self.encoder(flow, corr, attention)

        if self.gma:
            motion_features_global = self.aggregator(attention, motion_features)
            inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp_cat = torch.cat([inp, motion_features], dim=1)
        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


from timm.models.vision_transformer import Attention as timm_attn
from timm.models.layers import DropPath, Mlp


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=2, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path1 = DropPath(drop_rate)
        self.drop_path2 = DropPath(drop_rate)
        self.attn = timm_attn(
            dim,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.GELU,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class TemporalLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = zero_module(TransformerBlock(dim))

    def forward(self, x):
        # input: B T C H W
        # output: (B T) C H W
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b h w) t c")
        x = self.transformer_block(x)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", b=B, h=H, w=W)
        return x


class TemporalLayer2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = zero_module(TransformerBlock(dim))

    def forward(self, x, HW):
        # input: (B) (T H W) C
        # output: (B T) C H W
        H, W = HW[0], HW[1]
        x = self.transformer_block(x)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", h=H, w=W)
        return x


class TemporalLayer_noinit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = TransformerBlock(dim)

    def forward(self, x, HW):
        # input: (B) (T H W) C
        # output: (B T) C H W
        H, W = HW[0], HW[1]
        x = self.transformer_block(x)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", h=H, w=W)
        return x


class TemporalFlowRegressor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
        )
        self.transformer_block = zero_module(TransformerBlock(dim))
        self.fc2 = nn.Linear(dim, 2)

    def forward(self, x, HW):
        # input: (B) (T H W) C
        # output: (B T) C H W
        H, W = HW[0], HW[1]
        x = self.fc1(x)
        x = self.transformer_block(x)
        x = self.fc2(x)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", h=H, w=W)
        return x


class TemporalLayer_Ablation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = zero_module(TransformerBlock(dim))

    def forward(self, x, HW):
        # input: (B T) (H W) C
        # output: (B T) C H W
        H, W = HW[0], HW[1]
        x = self.transformer_block(x)
        x = rearrange(x, "B (h w) c -> B c h w", h=H, w=W)
        return x


class TemporalFlowHead(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.temporal = TemporalLayer(C_in)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        # input: B, T, C, H, W
        # output: (B T), C, H, W
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)
        x = self.temporal(x)
        x = self.ffn2(x)
        return x


class TemporalFlowHead2(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.temporal = nn.Sequential(
            nn.Conv1d(C_in, C_in, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(C_in, C_in, 3, padding=1),
        )
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        # input: B, T, C, H, W
        # output: (B T), C, H, W
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        # print(rearrange(x, '(b t) c h w -> (b h w) c t', b=B, t=T).shape)
        x = self.temporal(rearrange(x, "(b t) c h w -> (b h w) c t", b=B, t=T))
        x = rearrange(x, "(b h w) c t -> (b t) c h w", h=H, w=W)
        x = self.ffn2(x)
        return x


class SKFlow_Temporal2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)  # b (t c) h w

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class SKUpdateBlock_TAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)  # b (t c) h w

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


# temporal flow head v1
class SKUpdateBlock_TAM_v3(nn.Module):
    def __init__(
        self,
        decoder_dim,
        num_heads,
        use_gma,
        pcupdater_conv,
        corr_levels,
        corr_radius,
        T,
        k_conv,
    ):
        super().__init__()
        self.encoder = SKMotionEncoder6_Deep_nopool_res(
            decoder_dim=decoder_dim,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            k_conv=k_conv,
        )
        ratio = 8

        self.gma = use_gma
        embed_dim = decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                dim=embed_dim, dim_head=embed_dim, heads=num_heads
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=pcupdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=pcupdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)
        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (T - 1), 2 * (T - 1), k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        if self.gma:
            motion_features_globals = self.aggregator(attentions, motion_features)
            inp_cats = torch.cat(
                [
                    inps,
                    motion_features,
                    motion_features_globals,
                    motion_features_temporal,
                ],
                dim=1,
            )
        else:
            inp_cats = torch.cat(
                [inps, motion_features, motion_features_temporal],
                dim=1,
            )
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", T=T)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W ", T=T)

        return nets, masks, delta_flows


# temporal flow head v1
class SKUpdateBlock_TAM_v3_noinit(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer_noinit(dim=embed_dim)
        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        # print(T, rearrange(nets, '(B T) C H W -> B (T C) H W', T=T).shape, nets.shape)
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", T=T)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W ", T=T)

        return nets, masks, delta_flows


# temporal flow head v1
class SKUpdateBlock_TAM_ParamAblation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer_Ablation(dim=embed_dim)
        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_additional = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B T) (H W) C", T=T), HW=[H, W]
        )
        inp_cats = torch.cat(
            [
                inps,
                motion_features,
                motion_features_globals,
                motion_features_additional,
            ],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W ", T=T)

        return nets, masks, delta_flows


# temporal flow head v1
class SKUpdateBlock_TAM_v7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)
        self.temporal_regressor = TemporalLayer2(dim=2)
        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> (B H W) T C ", T=T)
        delta_flows = self.temporal_regressor(delta_flows, HW=(H, W))
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W", B=B)

        return nets, masks, delta_flows


# temporal flow head v1
class SKUpdateBlock_TAM_v8(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)
        self.temporal_regressor = TemporalFlowRegressor(dim=embed_dim)
        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> (B H W) T C ", T=T)
        delta_flows = self.temporal_regressor(delta_flows, HW=(H, W))
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W", B=B)

        return nets, masks, delta_flows


# temporal flow head v2
class SKUpdateBlock_TAM_v4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.transformer_block2 = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        nets = self.transformer_block2(
            rearrange(nets, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)  # b (t c) h w

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


# temporal flow head v1 + mask
class SKUpdateBlock_TAM_v5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim * (args.T - 1), embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9 * (args.T - 1), 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(rearrange(nets, "(B T) C H W -> B (T C) H W", B=B))

        masks = rearrange(masks, "B (T C) H W -> B T C H W", T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W ", T=T)

        return nets, masks, delta_flows


# temporal flow head v1 + corr temporal
class SKUpdateBlock_TAM_v6(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder_CorrTemporal(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim * (args.T - 1), 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W ", T=T)

        return nets, masks, delta_flows


class Bi_SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.T = args.T - 1
        out_dim = args.decoder_dim // 2
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2

        self.convc1 = PCBlock4_Deep_nopool_res(
            cor_planes, int(256 / self.T), args.k_conv
        )
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, args.k_conv)

        self.convf1 = nn.Conv2d(2 * self.T, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(
            64 + 192, out_dim - 2 * self.T, args.k_conv
        )

    def forward(self, flow, corr, attention=None):
        sub_corrs = list(corr.chunk(self.T, dim=1))
        sub_corrs = [self.convc1(cor) for cor in sub_corrs]

        cor = F.gelu(torch.cat(sub_corrs, dim=1))
        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class Bi_SKUpdateBlock_TAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Bi_SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 5, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9 * (args.T - 1), 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)

        self.flow_head = PCBlock4_Deep_nopool_res(
            embed_dim, 2 * (args.T - 1), args.k_conv
        )

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )
        inp_cats = torch.cat(
            [inps, motion_features, motion_features_globals, motion_features_temporal],
            dim=1,
        )
        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class SKFlow_TMM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = TMMAggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(
        self, nets, inps, corrs, flows, attentions, temporal_attentions, T=None
    ):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(
            attentions, temporal_attentions, motion_features
        )
        # motion_features = self.transformer_block(rearrange(motion_features, '(B T) C H W -> (B H W) T C', T=T), HW=(H, W))
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


# TODO
class CrossTransformerBlock(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, flow, feature):
        b, c, h, w = feature.size()
        # f1为v; f2为q, k
        query = feature.view(b, c, h * w).permute(0, 2, 1)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]
        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c**0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)
        return out


class FlowRegressorv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 128, 3, padding=1)
        # self.conv2 = nn.Conv2d(128, 2, 3, padding=1)
        self.cross_attn_layer = CrossTransformerBlock()

    def forward(self, flo1, flo2):
        feat1 = self.conv1(flo1)
        feat2 = self.conv1(flo2)

        res1 = self.corss_attn_layer(flo1, feat2)
        res2 = self.corss_attn_layer(flo2, feat1)
        flo1 = flo1 + res1
        flo2 = flo2 + res2
        flows = torch.stack([flo1, flo2], dim=1)
        flows = rearrange(flows, "B T C H W -> (B T) C H W")
        # flows = self.conv2(flows)

        return flows


class SKFlow_TMM3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = TMMAggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)
        self.flow_regressor = FlowRegressorv1()

    def forward(
        self, nets, inps, corrs, flows, attentions, temporal_attentions, T=None
    ):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(
            attentions, temporal_attentions, motion_features
        )
        # motion_features = self.transformer_block(rearrange(motion_features, '(B T) C H W -> (B H W) T C', T=T), HW=(H, W))
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)  # [(B, T) C H W]

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = self.flow_regressor(
            delta_flows[:, 0, ...], delta_flows[:, 1, ...]
        )

        return nets, masks, delta_flows


# temporal attn layer
class SKFlow_TMM2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = TMMAggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.flow_head = TemporalFlowHead(embed_dim, 2, args.k_conv)
        # self.flow_head = TemporalFlowHead2(embed_dim, 2, args.k_conv)
        # self.flow_head = nn.Sequential(
        #     PCBlock4_Deep_nopool_res(embed_dim, embed_dim//2, args.k_conv),
        #     SKBlock_Temporal(embed_dim//2, 2, args.k_conv),
        # )

    def forward(
        self, nets, inps, corrs, flows, attentions, temporal_attentions, T=None
    ):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(
            attentions, temporal_attentions, motion_features
        )
        # motion_features = self.transformer_block(rearrange(motion_features, '(B T) C H W -> (B H W) T C', T=T), HW=(H, W))
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(rearrange(nets, "(B T) C H W -> B T C H W", T=T))
        # delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


# temporal sk


class SKFlow_TMM3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = TMMAggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        # self.flow_head = TemporalFlowHead(embed_dim, 2, args.k_conv)
        # self.flow_head = TemporalFlowHead2(embed_dim, 2, args.k_conv)
        self.flow_head = nn.Sequential(
            PCBlock4_Deep_nopool_res(embed_dim, embed_dim // 2, args.k_conv),
            SKBlock_Temporal(embed_dim // 2, 2, args.k_conv),
        )

    def forward(
        self, nets, inps, corrs, flows, attentions, temporal_attentions, T=None
    ):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(
            attentions, temporal_attentions, motion_features
        )
        # motion_features = self.transformer_block(rearrange(motion_features, '(B T) C H W -> (B H W) T C', T=T), HW=(H, W))
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        # delta_flows = self.flow_head(rearrange(nets, '(B T) C H W -> B T C H W', T=T))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class SKFlow_Temporal3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.skblock = PCBlock4_Deep_nopool_res(
            embed_dim, embed_dim, k_conv=args.k_conv
        )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features = self.skblock(
            rearrange(motion_features, "(B T) C H W -> B C H (T W)", T=T)
        )
        motion_features = rearrange(motion_features, "B C H (T W) -> (B T) C H W", T=T)
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)  # b (t c) h w

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class SKFlow_Temporal4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.skblock = PCBlock4_Deep_nopool_res(
            embed_dim, embed_dim, k_conv=args.k_conv
        )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_globals = self.aggregator(attentions, motion_features)
        motion_features = self.skblock(
            rearrange(motion_features, "(B T) C H W -> B C H (T W)", T=T)
        )
        motion_features = rearrange(motion_features, "B C H (T W) -> (B T) C H W", T=T)
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        delta_flows = self.flow_head(nets)

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)  # b (t c) h w

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class SKFlow_Temporal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.temporal_attention1 = TemporalLayer(dim=embed_dim * 3)
        self.temporal_attention2 = TemporalLayer(dim=embed_dim)
        self.temporal_flow_head = TemporalFlowHead(embed_dim, 2, args.k_conv)

    def forward(self, nets, inps, corrs, flows, attentions, ref_nets=None, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs, attentions)
        motion_features_globals = self.aggregator(attentions, motion_features)
        inp_cats = torch.cat([inps, motion_features, motion_features_globals], dim=1)

        # Temporal Attention
        inp_cats = rearrange(inp_cats, "(B T) C H W -> B T C H W", B=B, T=T)
        inp_cats = self.temporal_attention1(inp_cats)
        # Attentional update
        nets = self.gru(torch.cat([nets, inp_cats], dim=1))
        nets = rearrange(nets, "(B T) C H W -> B T C H W", B=B, T=T)
        nets = self.temporal_attention2(nets)

        nets = rearrange(nets, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = self.temporal_flow_head(nets)

        # scale mask to balence gradients
        nets = rearrange(nets, "B T C H W -> (B T) C H W")
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W ", B=B, T=T)
        return nets, masks, delta_flows


class TemporalFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fusion1 = args.fusion1
        self.proj = nn.Sequential(
            nn.Conv2d(
                args.decoder_dim // 2, args.decoder_dim // 2, kernel_size=3, padding=1
            ),
            nn.GELU(),
        )
        # init
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, temporal_refs):
        # x: [B, C, H, W]; temporal_refs: [B, T, C, H, W]; (x + temporal_refs) => (B, T, C, H, W) (B, C, H, W, T)
        if self.fusion1:
            ref = torch.mean(temporal_refs, dim=1, keepdim=False)
        else:
            res_x = x.unsqueeze(1).expand_as(temporal_refs)
            res_x = torch.abs(res_x - temporal_refs).sum(
                dim=(2, 3, 4)
            )  # 遮挡最严重的部分
            _, res_index = torch.max(res_x, dim=1)
            a = []
            for i in range(res_index.shape[0]):
                a.append(temporal_refs[i, res_index[i]])
            ref = torch.stack(a, dim=0)
            # print(ref.shape, res_index)
        x = x + self.gamma * self.proj(ref)

        return x


class SKFlowDecoder_MMBank(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        # self.temporal_fusion = TemporalFusion(args) # TODO

    def forward(self, net, inp, corr, flow, attention, ref_nets):
        motion_features = self.encoder(flow, corr, attention)

        if self.gma:
            motion_features_global = self.aggregator(attention, motion_features)
            inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp_cat = torch.cat([inp, motion_features], dim=1)
        # Attentional update

        # temporal Attention?
        # MemoryBank

        net = self.gru(torch.cat([net, inp_cat], dim=1))
        net = self.temporal_fusion(net, ref_nets)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class SKFlowDecoder_planB(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )
        else:
            self.gru = PCBlock4_Deep_nopool_res(
                embed_dim * 4, embed_dim, k_conv=args.PCUpdater_conv
            )

        self.flow_head = PCBlock4_Deep_nopool_res(embed_dim, 2, args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.temporal_fusion = TemporalFusion(args)

    def forward(self, net, inp, corr, flow, attention, ref_nets):
        motion_features = self.encoder(flow, corr, attention)

        if self.gma:
            motion_features_global = self.aggregator(attention, motion_features)
            inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp_cat = torch.cat([inp, motion_features], dim=1)
        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        net = self.temporal_fusion(net, ref_nets)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FlowHeadNew(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        hidden_dim = 128
        self.conv1 = nn.Conv2d(C_in, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, C_out, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        if "All" in args.model_name or args.cost_encoder_v1:
            cor_planes += 128
        elif "cost_encoder" in args.model_name or args.cost_encoder_v2:
            cor_planes = 512

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class RAFTDeepMotionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        if "All" in args.model_name or args.cost_encoder_v1:
            cor_planes += 128
        elif "cost_encoder" in args.model_name or args.cost_encoder_v2:
            cor_planes = 512

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convc3 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 128, 3, padding=1)
        self.convf3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 144, 3, padding=1)
        # self.conv2 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv2 = nn.Conv2d(144, 128 - 2, 3, padding=1)
        # self.conv3 = nn.Conv2d(126, 126, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc3(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        flo = F.relu(self.convf3(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        out = F.relu(self.conv2(out))
        # out = F.relu(self.conv3(out))
        return torch.cat([out, flow], dim=1)


class DeepMotionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        if "All" in args.model_name or args.cost_encoder_v1:
            cor_planes += 128
        elif "cost_encoder" in args.model_name or args.cost_encoder_v2:
            cor_planes = 512

        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convc3 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 128, 3, padding=1)
        self.convf3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 192, 3, padding=1)
        self.conv2 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv3 = nn.Conv2d(192, 128 - 2, 3, padding=1)
        # self.conv3 = nn.Conv2d(126, 126, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc3(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        flo = F.relu(self.convf3(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class RAFTDeepUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super().__init__()
        self.args = args
        self.encoder = RAFTDeepMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class GMADeepUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super().__init__()
        self.args = args
        self.encoder = DeepMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SKBlock_CBAM(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )
        self.ca = ChannelAttention(C_in)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))

        # CBAM
        in_feature = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = F.gelu(x + in_feature)

        x = self.ffn2(x)
        return x


class SKMotionEncoder_CBAM(nn.Module):
    def __init__(self, args):
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = SKBlock_CBAM(cor_planes, 256, args.k_conv)
        self.convc2 = SKBlock_CBAM(256, 192, args.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = SKBlock_CBAM(128, 64, args.k_conv)

        self.conv = SKBlock_CBAM(64 + 192, 128 - 2, args.k_conv)

    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKUpdateBlock_CBAM(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder_CBAM(args)

        self.gru = SKBlock_CBAM(
            128 + hidden_dim + hidden_dim + 128, 128, k_conv=args.PCUpdater_conv
        )
        self.flow_head = SKBlock_CBAM(128, 2, args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

        self.aggregator = Aggregate(
            args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
        )

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


# temporal flow head v1
class GMAUpdateBlock_TAM_v7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2
        hidden_dim = embed_dim

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = SepConvGRU(
                hidden_dim=hidden_dim,
                input_dim=128 + hidden_dim + hidden_dim + hidden_dim,
            )
        else:
            self.gru = SepConvGRU(
                hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer2(dim=embed_dim)
        self.temporal_regressor = TemporalLayer2(dim=2)
        self.flow_head = FlowHeadNew(embed_dim * (args.T - 1), 2 * (args.T - 1))

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B H W) T C", T=T), HW=(H, W)
        )

        if self.gma:
            motion_features_globals = self.aggregator(attentions, motion_features)
            inp_cats = torch.cat(
                [
                    inps,
                    motion_features,
                    motion_features_globals,
                    motion_features_temporal,
                ],
                dim=1,
            )
        else:
            inp_cats = torch.cat(
                [inps, motion_features, motion_features_temporal], dim=1
            )
        # Temporal Attention
        # Attentional update
        nets = self.gru(nets, inp_cats)
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> (B H W) T C ", T=T)
        delta_flows = self.temporal_regressor(delta_flows, HW=(H, W))
        delta_flows = rearrange(delta_flows, "(B T) C H W -> B T C H W", B=B)

        return nets, masks, delta_flows


# temporal flow head v1
class GMAUpdateBlock_TAM_ablation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        ratio = 16 if args.Encoder == "UMT" else 8

        self.gma = args.use_gma
        embed_dim = args.decoder_dim // 2
        hidden_dim = embed_dim

        if self.gma:
            self.aggregator = Aggregate(
                args=self.args,
                dim=embed_dim,
                dim_head=embed_dim,
                heads=self.args.num_heads,
            )
            self.gru = SepConvGRU(
                hidden_dim=hidden_dim,
                input_dim=128 + hidden_dim + hidden_dim + hidden_dim,
            )
        else:
            self.gru = SepConvGRU(
                hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
            )

        self.mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, ratio * ratio * 9, 1, padding=0),
        )

        self.transformer_block = TemporalLayer_Ablation(dim=embed_dim)
        self.flow_head = FlowHeadNew(embed_dim * (args.T - 1), 2 * (args.T - 1))

    def forward(self, nets, inps, corrs, flows, attentions, T=None):
        BT, _, H, W = nets.shape
        B = BT // T

        motion_features = self.encoder(flows, corrs)
        motion_features_temporal = self.transformer_block(
            rearrange(motion_features, "(B T) C H W -> (B T) (H W) C", T=T), HW=(H, W)
        )

        if self.gma:
            motion_features_globals = self.aggregator(attentions, motion_features)
            inp_cats = torch.cat(
                [
                    inps,
                    motion_features,
                    motion_features_globals,
                    motion_features_temporal,
                ],
                dim=1,
            )
        else:
            inp_cats = torch.cat(
                [inps, motion_features, motion_features_temporal], dim=1
            )
        # Temporal Attention
        # Attentional update
        nets = self.gru(nets, inp_cats)
        delta_flows = self.flow_head(
            rearrange(nets, "(B T) C H W -> B (T C) H W", B=B)
        )  # (b t) c h w => b (t c) h w

        # scale mask to balence gradients
        masks = 0.25 * self.mask(nets)

        masks = rearrange(masks, "(B T) C H W -> B T C H W", B=B, T=T)
        delta_flows = rearrange(delta_flows, "B (T C) H W -> B T C H W", T=T)

        return nets, masks, delta_flows


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
        )
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

        self.aggregator = Aggregate(
            args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
        )

    def forward(self, net, inp, corr, flow, attention, T=None):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        # corr关注的是非遮挡区域，所以和motion feature极其相似?。。。。。。。。。。。。。
        delta_flow = self.flow_head(net)

        # plot
        # plot_featmap(motion_features.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-motion_feature')
        # plot_featmap(motion_features_global.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-motion_feature_global')
        # plot_featmap(net.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-net')

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class RAFTUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

        self.aggregator = Aggregate(
            args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
        )

    def forward(self, net, inp, corr, flow, attention, T=None):
        motion_features = self.encoder(flow, corr)
        inp_cat = torch.cat([inp, motion_features], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        # corr关注的是非遮挡区域，所以和motion feature极其相似?。。。。。。。。。。。。。
        delta_flow = self.flow_head(net)

        # plot
        # plot_featmap(motion_features.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-motion_feature')
        # plot_featmap(motion_features_global.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-motion_feature_global')
        # plot_featmap(net.cpu().squeeze(0), '/mnt/cloud_disk/ssk/playground/GMA-main/nips-figs/RAFTGMA/487-net')

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class MFUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    # corr: 第i和i+1; corr_before: 第i-1和i
    def forward(self, net, inp, motion_features, attention=None):
        inp_cat = torch.cat([inp, motion_features], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class SKFlowMotionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2

        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, args.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64 + 192, 128 - 2, args.k_conv)

    def forward(self, flow, corr, attention=None):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class MFSKFlowUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        if args.use_temporal_decoder:
            self.aggregator = SpatioTemporalAggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
            )
        else:
            self.aggregator = Aggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
            )
        self.gru = PCBlock4_Deep_nopool_res(
            128 + hidden_dim + hidden_dim + 128, 128, k_conv=args.PCUpdater_conv
        )
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, motion_features, attention, temporal_attention=None):
        if self.args.use_temporal_decoder:
            motion_features_global = self.aggregator(
                attention, temporal_attention, motion_features
            )
        else:
            motion_features_global = self.aggregator(attention, motion_features)

        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class MFRAFTUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        if args.use_temporal_decoder:
            self.aggregator = TemporalAggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
            )
            if self.args.fuse2:
                self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
            else:
                self.temporal_gru = SepConvGRU(
                    hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
                )
        else:
            self.aggregator = None
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, motion_features, temporal_attention=None):
        if self.args.use_temporal_decoder:
            motion_features_global = self.aggregator(
                temporal_attention, motion_features
            )
            if self.args.fuse2:
                motion_features = motion_features + motion_features_global
                inp_cat = torch.cat([inp, motion_features], dim=1)
                net = self.gru(net, inp_cat)
            else:
                inp_cat = torch.cat(
                    [inp, motion_features, motion_features_global], dim=1
                )
                net = self.temporal_gru(net, inp_cat)
        else:
            inp_cat = torch.cat([inp, motion_features], dim=1)
            net = self.gru(net, inp_cat)
        # Attentional update
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class MFGMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        if args.use_temporal_decoder:
            self.aggregator = SpatioTemporalAggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
            )
        else:
            self.aggregator = Aggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
            )
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim
        )
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, motion_features, attention, temporal_attention=None):
        if self.args.use_temporal_decoder:
            motion_features_global = self.aggregator(
                attention, temporal_attention, motion_features
            )
        else:
            motion_features_global = self.aggregator(attention, motion_features)

        # Attentional update
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class ControlMFSKIIUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args
        self.aggregator = SpatioTemporalAggregate(
            args=self.args, dim=128, dim_head=128, heads=self.args.num_heads
        )
        self.gru = eval(args.SKII_Block)(
            128 + hidden_dim + hidden_dim + 128, 128, args.PCUpdater_conv, args
        )
        self.flow_head = eval(args.SKII_Block)(128, 2, args.k_conv, args)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

        self.zero_conv0 = nn.Conv2d(128, 128, 1)
        self.zero_conv1 = nn.Conv2d(2, 2, 1)
        self.zero_conv2 = nn.Conv2d(64 * 9, 64 * 9, 1)

        for p in self.zero_conv0.parameters():
            nn.init.zeros_(p)
        for p in self.zero_conv1.parameters():
            nn.init.zeros_(p)
        for p in self.zero_conv2.parameters():
            nn.init.zeros_(p)

    def forward(self, net, inp, motion_features, attention, temporal_attention=None):
        motion_features_global = self.aggregator(
            attention, temporal_attention, motion_features
        )
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        net = self.zero_conv0(net)

        delta_flow = self.flow_head(net)
        delta_flow = self.zero_conv1(delta_flow)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        mask = self.zero_conv2(mask)
        return net, mask, delta_flow
