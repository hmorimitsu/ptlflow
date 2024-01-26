# =============================================================================
# Copyright 2023 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code adapted from RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
# =============================================================================

from functools import partial

import torch
import torch.nn as nn

from .pkconv import PKConv2d
from .pkconv_slk import PKConvSLK
from .local_timm.norm import LayerNorm2d


class FlowHeadPartial(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, cache_pkconv_weights=False):
        super(FlowHeadPartial, self).__init__()
        self.conv1 = PKConv2d(
            input_dim, hidden_dim, 3, padding=1, cache_weights=cache_pkconv_weights
        )
        self.conv2 = PKConv2d(
            hidden_dim, 2, 3, padding=1, cache_weights=cache_pkconv_weights
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, out_ch=None):
        return self.conv2(self.act(self.conv1(x)), out_ch)


class ConvPartialGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, cache_pkconv_weights=False):
        super(ConvPartialGRU, self).__init__()
        self.convz = PKConv2d(
            hidden_dim + input_dim,
            hidden_dim,
            3,
            padding=1,
            cache_weights=cache_pkconv_weights,
        )
        self.convr = PKConv2d(
            hidden_dim + input_dim,
            hidden_dim,
            3,
            padding=1,
            cache_weights=cache_pkconv_weights,
        )
        self.convq = PKConv2d(
            hidden_dim + input_dim,
            hidden_dim,
            3,
            padding=1,
            cache_weights=cache_pkconv_weights,
        )

    def forward(self, h, x, out_ch):
        hx = torch.cat([h, x], dim=1)

        z = self.convz(hx, out_ch)
        z = torch.sigmoid(z)

        r = self.convr(hx, out_ch)
        r = torch.sigmoid(r)

        q = self.convq(torch.cat([r * h, x], dim=1), out_ch)
        q = torch.tanh(q)

        h = (1 - z) * h + z * q
        return h


class PKConvSLKGRU(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        use_norm_affine=False,
        num_groups=8,
        depth=2,
        mlp_ratio=4,
        cache_pkconv_weights=False,
    ):
        super(PKConvSLKGRU, self).__init__()
        norm_layer = partial(LayerNorm2d, affine=use_norm_affine)

        self.convz = PKConvSLK(
            hidden_dim + input_dim,
            hidden_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            stride=1,
            depth=depth,
            cache_pkconv_weights=cache_pkconv_weights,
        )
        self.convr = PKConvSLK(
            hidden_dim + input_dim,
            hidden_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            stride=1,
            depth=depth,
            cache_pkconv_weights=cache_pkconv_weights,
        )
        self.convq = PKConvSLK(
            hidden_dim + input_dim,
            hidden_dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            stride=1,
            depth=depth,
            cache_pkconv_weights=cache_pkconv_weights,
        )

    def forward(self, h, x, out_ch):
        hx = torch.cat([h, x], dim=1)

        z = self.convz(hx, out_ch)
        z = torch.sigmoid(z)

        r = self.convr(hx, out_ch)
        r = torch.sigmoid(r)

        q = self.convq(torch.cat([r * h, x], dim=1), out_ch)
        q = torch.tanh(q)

        h = (1 - z) * h + z * q
        return h


class ConvexMask(nn.Module):
    def __init__(self, net_chs, pred_stride, cache_pkconv_weights=False):
        super(ConvexMask, self).__init__()
        self.conv1 = PKConv2d(
            net_chs, net_chs * 2, 3, padding=1, cache_weights=cache_pkconv_weights
        )
        self.act = nn.ReLU(inplace=True)
        self.conv2 = PKConv2d(
            net_chs * 2,
            pred_stride**2 * 9,
            1,
            padding=0,
            cache_weights=cache_pkconv_weights,
        )

    def forward(self, x):
        in_chs = x.shape[1]
        x = self.conv1(x, 2 * in_chs)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MotionEncoderPartial(nn.Module):
    def __init__(self, args):
        super(MotionEncoderPartial, self).__init__()

        self.args = args

        c_hidden = 256
        c_out = 192
        f_hidden = 128
        f_out = 64

        cor_planes = args.corr_levels * (2 * args.corr_range + 1) ** 2
        self.convc1 = PKConv2d(
            cor_planes, c_hidden, 1, padding=0, cache_weights=args.cache_pkconv_weights
        )
        self.convc2 = PKConv2d(
            c_hidden, c_out, 3, padding=1, cache_weights=args.cache_pkconv_weights
        )

        self.convf1 = PKConv2d(
            2, f_hidden, 7, padding=3, cache_weights=args.cache_pkconv_weights
        )
        self.convf2 = PKConv2d(
            f_hidden, f_out, 3, padding=1, cache_weights=args.cache_pkconv_weights
        )

        in_ch = f_out + c_out
        out_ch = args.dec_motion_chs - 2
        self.conv = PKConv2d(
            in_ch, out_ch, 3, padding=1, cache_weights=args.cache_pkconv_weights
        )

        self.act = nn.ReLU()

    def forward(self, flow, corr):
        cor = self.act(self.convc1(corr))
        cor = self.act(self.convc2(cor))

        outs = [cor]

        flo = self.act(self.convf1(flow))
        flo = self.act(self.convf2(flo))
        outs.append(flo)

        outs = torch.cat(outs, dim=1)
        out = self.act(self.conv(outs))
        out_t = [out, flow]
        return torch.cat(out_t, dim=1)


class UpdatePartialBlock(nn.Module):
    def __init__(self, args):
        super(UpdatePartialBlock, self).__init__()
        self.args = args
        self.encoder = MotionEncoderPartial(args)
        self.gru_list = nn.ModuleList(
            [
                PKConvSLKGRU(
                    hidden_dim=args.net_chs_fixed,
                    input_dim=args.dec_motion_chs + args.inp_chs_fixed,
                    use_norm_affine=args.use_norm_affine,
                    num_groups=args.group_norm_num_groups,
                    depth=args.dec_gru_depth,
                    mlp_ratio=args.dec_gru_mlp_ratio,
                    cache_pkconv_weights=args.cache_pkconv_weights,
                )
                for _ in range(args.dec_gru_iters)
            ]
        )

        self.flow_head = FlowHeadPartial(
            args.net_chs_fixed,
            hidden_dim=256,
            cache_pkconv_weights=args.cache_pkconv_weights,
        )

        if self.args.use_upsample_mask:
            pred_stride = min(self.args.pyramid_ranges)
            self.mask = ConvexMask(
                args.net_chs_fixed,
                pred_stride,
                cache_pkconv_weights=args.cache_pkconv_weights,
            )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)

        for gru in self.gru_list:
            net = gru(net, inp, net.shape[1])

        delta_flow = self.flow_head(net)

        mask = None
        if self.args.use_upsample_mask:
            mask = self.args.upmask_gradient_scale * self.mask(net)

        return delta_flow, net, mask
