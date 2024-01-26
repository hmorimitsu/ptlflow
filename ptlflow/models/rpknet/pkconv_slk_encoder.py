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
# =============================================================================

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .local_timm.norm import LayerNorm2d
from .local_timm.weight_init import trunc_normal_
from .local_timm.gelu import GELU
from .pkconv_slk import PKConvSLK
from .pkconv import PKConv2d
from .update_partial import ConvPartialGRU


class PKConvSLKEncoder(nn.Module):
    """Our proposed recurrent encoder with PKConv and SLK layers"""

    def __init__(
        self,
        pyr_range: Tuple[int, int],
        hidden_chs: int,
        out_1x1_abs_chs: int,
        out_1x1_factor: Optional[float],
        stem_stride: int = 2,
        norm_layer=LayerNorm2d,
        mlp_ratio: float = 4,
        depth: int = 2,
        cache_pkconv_weights: bool = False,
    ):
        super(PKConvSLKEncoder, self).__init__()

        self.pyr_range = pyr_range
        self.hidden_chs = hidden_chs
        self.out_1x1_abs_chs = out_1x1_abs_chs
        self.out_1x1_factor = out_1x1_factor
        self.stem_stride = stem_stride
        self.norm_layer = norm_layer
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        self.pyr_level_range = [int(math.log2(v)) for v in self.pyr_range]

        self.forward_gru = ConvPartialGRU(
            hidden_chs[-1], hidden_chs[-1], cache_pkconv_weights=cache_pkconv_weights
        )
        self.down_gru = PKConv2d(
            hidden_chs[-1],
            hidden_chs[-1],
            3,
            stride=2,
            padding=1,
            bias=True,
            cache_weights=cache_pkconv_weights,
        )

        self.stem = self._make_stem(
            hidden_chs[0],
            norm_layer=norm_layer,
            stem_stride=stem_stride,
            cache_pkconv_weights=cache_pkconv_weights,
        )

        self.rec_stage = self._make_stage(
            hidden_chs[-1],
            norm_layer=norm_layer,
            out_chs=hidden_chs[-1],
            depth=depth,
            mlp_ratio=mlp_ratio,
            cache_pkconv_weights=cache_pkconv_weights,
        )

        if self.out_1x1_abs_chs > 0:
            self.out_1x1 = self._make_out_1x1_layer(
                hidden_chs[-1],
                self.out_1x1_abs_chs,
                cache_pkconv_weights=cache_pkconv_weights,
            )

        self._init_weights()

    def _make_stem(
        self, hidden_chs: int, norm_layer, stem_stride, cache_pkconv_weights
    ):
        conv = PKConv2d(
            3,
            hidden_chs,
            kernel_size=7,
            stride=stem_stride,
            padding=3,
            cache_weights=cache_pkconv_weights,
        )
        return nn.Sequential(conv, norm_layer(num_channels=hidden_chs))

    def _make_stage(
        self,
        hidden_chs: int,
        norm_layer,
        out_chs=None,
        depth=2,
        mlp_ratio=4,
        cache_pkconv_weights=False,
    ):
        if out_chs is None:
            out_chs = hidden_chs
        return PKConvSLK(
            hidden_chs,
            out_chs,
            mlp_ratio=mlp_ratio,
            act_layer=GELU,
            norm_layer=norm_layer,
            stride=2,
            depth=depth,
            cache_pkconv_weights=cache_pkconv_weights,
        )

    def _make_out_1x1_layer(
        self, hidden_chs: int, out_chs: int, cache_pkconv_weights: bool
    ):
        return PKConv2d(
            hidden_chs, out_chs, kernel_size=1, cache_weights=cache_pkconv_weights
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def _compute_pyr_iters(self, x):
        pyr_iters = self.pyr_level_range[1]
        if self.stem_stride > 2:
            pyr_iters -= int(math.log2(self.stem_stride)) - 1
        return pyr_iters

    def forward(self, x):
        x_pyramid = []
        if self.pyr_level_range[0] == 0:
            x_pyramid.append(x)

        pyr_iters = self._compute_pyr_iters(x)
        offset = 1
        if self.stem_stride > 2:
            offset += int(math.log2(self.stem_stride)) - 1
        for i in range(pyr_iters):
            if i == 0:
                x = self.stem(x)
                h = torch.zeros_like(x)
            else:
                in_ch = self.hidden_chs[min(i - 1, len(self.hidden_chs) - 1)]
                out_ch = self.hidden_chs[min(i, len(self.hidden_chs) - 1)]
                h = self.forward_gru(h, x, in_ch)

                x = self.rec_stage(h, out_ch).contiguous()
                if i < (pyr_iters - 1):
                    h = self.down_gru(h, out_ch)
                    h = torch.tanh(h)

            if i >= (self.pyr_level_range[0] - offset):
                x_pyramid.append(x)

        for i, x in enumerate(x_pyramid):
            if self.out_1x1_abs_chs > 0:
                if self.out_1x1_factor is None:
                    x = self.out_1x1(x)
                else:
                    x = self.out_1x1(x, int(self.out_1x1_factor * x.shape[1]))
            x_pyramid[i] = x

        return x_pyramid[::-1]
