# =============================================================================
# Copyright 2024 Henrique Morimitsu
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
# Adapted from RAFT: https://github.com/princeton-vl/RAFT
#
# Original license BSD 3-clause: https://github.com/princeton-vl/RAFT/blob/master/LICENSE
#
# Modifications by Henrique Morimitsu:
# - Adapt model parameters
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .next1d import NeXt1DStage
from .local_timm.norm import LayerNorm2d


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class NeXT1DDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        ksize=7,
        depth=1,
        mlp_ratio=4.0,
        norm_layer=None,
        fuse_next1d_weights=False,
    ):
        super(NeXT1DDecoder, self).__init__()
        self.conv = NeXt1DStage(
            hidden_dim + input_dim,
            hidden_dim,
            kernel_size=ksize,
            stride=1,
            depth=depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            fuse_next1d_weights=fuse_next1d_weights,
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        h = self.conv(hx)
        h = torch.tanh(h)
        return h


class MotionEncoder(nn.Module):
    def __init__(self, args):
        super(MotionEncoder, self).__init__()

        c_hidden = 256
        c_out = 192
        f_hidden = 128
        f_out = 64

        cor_planes = args.corr_levels * (2 * args.corr_range + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, c_hidden, 1, padding=0)
        self.convc2 = nn.Conv2d(c_hidden, c_out, 3, padding=1)
        self.convf1 = nn.Conv2d(2, f_hidden, 7, padding=3)
        self.convf2 = nn.Conv2d(f_hidden, f_out, 3, padding=1)
        self.conv = nn.Conv2d(f_out + c_out, args.dec_motion_chs - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, args):
        super(UpdateBlock, self).__init__()
        self.args = args

        self.encoder = MotionEncoder(args)
        self.decoder = NeXT1DDecoder(
            hidden_dim=args.dec_net_chs,
            input_dim=args.dec_motion_chs + args.dec_inp_chs,
            ksize=7,
            depth=args.dec_depth,
            mlp_ratio=args.dec_mlp_ratio,
            norm_layer=LayerNorm2d,
            fuse_next1d_weights=args.fuse_next1d_weights,
        )

        self.flow_head = FlowHead(args.dec_net_chs, hidden_dim=256)

        pred_stride = (
            min(self.args.pyramid_ranges) if self.args.use_upsample_mask else 8
        )
        self.mask = nn.Sequential(
            nn.Conv2d(args.dec_net_chs, args.dec_net_chs * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dec_net_chs * 2, pred_stride**2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, get_mask=False):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.decoder(net, inp)
        delta_flow = self.flow_head(net)

        mask = None
        if self.args.use_upsample_mask and get_mask:
            mask = self.mask(net)

        return delta_flow, net, mask
