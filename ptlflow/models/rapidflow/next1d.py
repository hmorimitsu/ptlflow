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
# This code on based on the ConvNeXt implementation from timm:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
#
# Original Copyright below:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
# =============================================================================

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from .local_timm.create_conv2d import create_conv2d
from .local_timm.drop import DropPath
from .local_timm.gelu import GELU
from .local_timm.mlp import ConvMlp


class FusedConv1d(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding=0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        fuse_weights=False,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.fuse_weights = fuse_weights

        if fuse_weights:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros(
                        out_channels,
                        in_channels // groups,
                        kernel_size,
                        kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
        else:
            self.register_parameter(
                "weight_h",
                nn.Parameter(
                    torch.zeros(
                        out_channels,
                        in_channels // groups,
                        1,
                        kernel_size,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
            self.register_parameter(
                "weight_v",
                nn.Parameter(
                    torch.zeros(
                        out_channels,
                        in_channels // groups,
                        kernel_size,
                        1,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
            self.weight = None

        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(out_channels, device=device, dtype=dtype)),
            )

    def forward(self, input):
        if self.training and not self.fuse_weights:
            x = F.conv2d(
                input,
                self.weight_h,
                self.bias,
                self.stride,
                (0, self.padding),
                self.dilation,
                self.groups,
            )
            x = F.conv2d(
                x,
                self.weight_v,
                self.bias,
                self.stride,
                (self.padding, 0),
                self.dilation,
                self.groups,
            )
            self.weight = None
        else:
            if self.weight is None:
                self.weight = torch.einsum(
                    "cijk,cimj->cimk", self.weight_h, self.weight_v
                ).to(device=input.device)
            x = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return x


class NeXt1DBlock(pl.LightningModule):
    def __init__(
        self,
        in_chs,
        out_chs=None,
        kernel_size=7,
        stride=1,
        dilation=1,
        mlp_ratio=4,
        conv_bias=True,
        ls_init_value=1e-6,
        norm_layer=None,
        drop_path=0.0,
        fuse_next1d_weights=False,
    ):
        super().__init__()
        out_chs = out_chs or in_chs

        self.conv_dw = FusedConv1d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=in_chs,
            bias=conv_bias,
            fuse_weights=fuse_next1d_weights,
        )

        self.norm = norm_layer(out_chs)
        self.mlp = ConvMlp(out_chs, int(mlp_ratio * out_chs), act_layer=GELU)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class NeXt1DStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=7,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        conv_bias=True,
        norm_layer=None,
        mlp_ratio=4,
        fuse_next1d_weights=False,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = (
                "same" if dilation[1] > 1 else 0
            )  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                NeXt1DBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_bias=conv_bias,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                    fuse_next1d_weights=fuse_next1d_weights,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
