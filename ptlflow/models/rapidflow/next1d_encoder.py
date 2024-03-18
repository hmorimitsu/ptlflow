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
# =============================================================================

import math
from typing import Tuple

import torch.nn as nn

from .local_timm.norm import LayerNorm2d
from .local_timm.weight_init import trunc_normal_
from .next1d import NeXt1DStage


class NeXt1DEncoder(nn.Module):
    def __init__(
        self,
        max_pyr_range: Tuple[int, int],
        stem_stride: int,
        num_recurrent_layers: int,
        hidden_chs: int,
        out_chs: int,
        norm_layer=LayerNorm2d,
        mlp_ratio: float = 4,
        depth: int = 2,
        fuse_next1d_weights: bool = False,
    ):
        super(NeXt1DEncoder, self).__init__()
        self.max_pyr_range = max_pyr_range
        self.stem_stride = stem_stride
        self.num_recurrent_layers = num_recurrent_layers

        self.stem = self._make_stem(hidden_chs, stem_stride, norm_layer=norm_layer)
        self.rec_stage = self._make_stage(
            hidden_chs,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            depth=depth,
            fuse_next1d_weights=fuse_next1d_weights,
        )
        self.out_layer = self._make_out_layer(hidden_chs, out_chs)

    def _make_stem(self, hidden_chs: int, stem_stride: int, norm_layer):
        conv = nn.Conv2d(3, hidden_chs, kernel_size=7, stride=stem_stride, padding=3)
        return nn.Sequential(conv, norm_layer(hidden_chs))

    def _make_stage(
        self, hidden_chs: int, norm_layer, mlp_ratio, depth, fuse_next1d_weights
    ):
        return NeXt1DStage(
            hidden_chs,
            hidden_chs,
            stride=2,
            depth=depth,
            norm_layer=norm_layer,
            mlp_ratio=mlp_ratio,
            fuse_next1d_weights=fuse_next1d_weights,
        )

    def _make_out_layer(self, hidden_chs: int, out_chs: int):
        return nn.Conv2d(hidden_chs, out_chs, kernel_size=1)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x_pyramid = []
        curr_stride = 1
        iters = self.num_recurrent_layers + 2 - int(math.log2(self.stem_stride))
        for i in range(iters):
            if i == 0:
                x = self.stem(x)
                curr_stride *= self.stem_stride
            else:
                x = self.rec_stage(x).contiguous()
                curr_stride *= 2

            if curr_stride >= self.max_pyr_range[0]:
                x_pyramid.append(x)

        for i, x in enumerate(x_pyramid[::-1]):
            x = self.out_layer(x)
            x_pyramid[i] = x

        return x_pyramid
