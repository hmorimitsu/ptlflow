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
# The following parts have been borrowed from other codes:
# ResidualBlock: https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py
# forward_interpolate and bilinear_sampler: https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
# =============================================================================

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pkconv import PKConv2d
from .local_timm.norm import GroupNorm, LayerNorm2d, BatchNorm2d


class SequentialPartial(nn.Module):
    def __init__(self, *ops) -> None:
        super().__init__()
        self.ops = nn.ModuleList(ops)

    def forward(self, x, out_ch=None):
        for op in self.ops:
            x = op(x, out_ch=out_ch)
        return x


class InterpolationTransition(nn.Module):
    """Change the input tensor shape using interpolation"""

    def __init__(
        self, use_zero_channel: bool = True, hw_scale_factor: float = 1.0
    ) -> None:
        super(InterpolationTransition, self).__init__()
        self.use_zero_channel = use_zero_channel
        self.hw_scale_factor = hw_scale_factor

    def forward(self, x, out_ch):
        b, c, h, w = x.shape
        if (self.hw_scale_factor < 0.999 or self.hw_scale_factor > 1.001) and (
            self.use_zero_channel or c == out_ch
        ):
            h, w = int(h * self.hw_scale_factor), int(w * self.hw_scale_factor)
            x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)

        if c != out_ch:
            if self.use_zero_channel:
                if c < out_ch:
                    x_zero = torch.zeros(
                        b, out_ch - c, h, w, dtype=x.dtype, device=x.device
                    )
                    x = torch.cat([x, x_zero], dim=1)
                elif c > out_ch:
                    x = x[:, :out_ch]
            else:
                h, w = int(h * self.hw_scale_factor), int(w * self.hw_scale_factor)
                x = F.interpolate(
                    x[None], (out_ch, h, w), mode="trilinear", align_corners=True
                )[0]
        return x


class ResidualPartialBlock(nn.Module):
    """ResNet block with PKConv layers"""

    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=None,
        stride=1,
        use_out_activation=True,
        cache_pkconv_weights=False,
    ):
        super(ResidualPartialBlock, self).__init__()

        self.use_out_activation = use_out_activation
        self.conv1 = PKConv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            cache_weights=cache_pkconv_weights,
        )
        self.norm1 = (
            norm_layer(num_channels=planes)
            if norm_layer is not None
            else nn.Sequential()
        )
        self.conv2 = PKConv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            cache_weights=cache_pkconv_weights,
        )
        self.norm2 = (
            norm_layer(num_channels=planes)
            if norm_layer is not None
            else nn.Sequential()
        )
        self.act = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = PKConv2d(
                in_planes,
                planes,
                kernel_size=1,
                stride=stride,
                cache_weights=cache_pkconv_weights,
            )
            self.norm3 = (
                norm_layer(num_channels=planes)
                if norm_layer is not None
                else nn.Sequential()
            )

    def forward(self, x, out_ch):
        y = x
        y = self.act(self.norm1(self.conv1(y, out_ch)))
        y = self.norm2(self.conv2(y, out_ch))
        if self.use_out_activation:
            y = self.act(y)

        if self.downsample is not None:
            x = self.norm3(self.downsample(x, out_ch))

        out = x + y
        if self.use_out_activation:
            out = self.act(out)

        return out


class ResidualBlock(nn.Module):
    """ResNet block"""

    def __init__(
        self, in_planes, planes, norm_fn="group", stride=1, use_out_activation=True
    ):
        super(ResidualBlock, self).__init__()

        self.use_out_activation = use_out_activation

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.act(self.norm1(self.conv1(y)))
        y = self.norm2(self.conv2(y))
        if self.use_out_activation:
            y = self.act(y)

        if self.downsample is not None:
            x = self.downsample(x)

        out = x + y
        if self.use_out_activation:
            out = self.act(out)

        return out


def bilinear_sampler(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask

    return img


def get_norm_layer(layer_name, affine=False, num_groups=8):
    if layer_name == "batch":
        norm_layer = BatchNorm2d
    elif layer_name == "group":
        norm_layer = partial(
            GroupNorm,
            affine=affine,
            num_groups=num_groups,
        )
    elif layer_name == "layer":
        norm_layer = partial(LayerNorm2d, affine=affine)
    elif layer_name == "none":
        norm_layer = None
    else:
        raise ValueError(f"Unknown norm layer {layer_name}")
    return norm_layer
