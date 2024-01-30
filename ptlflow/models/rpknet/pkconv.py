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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pkconv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    is_transpose: bool = False,
    out_ch: Optional[int] = None,
    return_weights: bool = False,
    skip_slicing: bool = False,
) -> torch.Tensor:
    bounded_groups = min(groups, x.shape[1])

    if skip_slicing:
        w = weight
        b = bias
    else:
        if out_ch is None:
            out_ch = weight.shape[0]

        b = None
        if groups == 1:
            if is_transpose:
                w = weight[: x.shape[1], :out_ch]
            else:
                w = weight[:out_ch, : x.shape[1]]

            if bias is not None:
                b = bias[:out_ch]
        else:
            int_size = out_ch // groups
            collect_group_sizes = [int_size] * groups
            remainder = out_ch - groups * int_size
            collect_group_sizes = [
                collect_group_sizes[i] + 1 if i < remainder else collect_group_sizes[i]
                for i in range(len(collect_group_sizes))
            ]

            total_out_ch = weight.shape[1] if is_transpose else weight.shape[0]
            group_size = total_out_ch // groups
            group_init_idx = np.arange(0, total_out_ch, group_size)

            slice_idx = [
                group_init_idx[i] + j
                for i in range(groups)
                for j in range(collect_group_sizes[i])
            ]

            if is_transpose:
                w = weight[: x.shape[1] // bounded_groups, slice_idx]
            else:
                w = weight[slice_idx, : x.shape[1] // bounded_groups]

            if bias is not None:
                b = bias[slice_idx]

    if is_transpose:
        x = F.conv_transpose2d(
            x,
            weight=w,
            bias=b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=bounded_groups,
        )
    else:
        x = F.conv2d(
            x,
            weight=w,
            bias=b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=bounded_groups,
        )

    if return_weights:
        return x, w, b
    else:
        return x


class PKConvBase(nn.Module):
    def __init__(
        self,
        is_transpose,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        cache_weights=False,
    ) -> None:
        super().__init__()
        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.cache_weights = cache_weights

        if self.is_transpose:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros(
                        in_channels // groups,
                        out_channels,
                        kernel_size[0],
                        kernel_size[1],
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
        else:
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.zeros(
                        out_channels,
                        in_channels // groups,
                        kernel_size[0],
                        kernel_size[1],
                        device=device,
                        dtype=dtype,
                    )
                ),
            )

        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(out_channels, device=device, dtype=dtype)),
            )
        else:
            self.bias = None

        self.reset_parameters()

        if self.cache_weights:
            self.cache = {}

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, out_ch: Optional[int] = None) -> torch.Tensor:
        cache_key = (x.shape[1], out_ch)
        if self.cache_weights and cache_key in self.cache:
            w, b = self.cache[cache_key]
            skip_slicing = True
            return_weights = False
        else:
            w = self.weight
            b = self.bias
            skip_slicing = False
            return_weights = self.cache_weights
        res = pkconv2d(
            x=x,
            weight=w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            is_transpose=self.is_transpose,
            out_ch=out_ch,
            skip_slicing=skip_slicing,
            return_weights=return_weights,
        )

        if return_weights:
            x = res[0]
            self.cache[cache_key] = res[1:3]
        else:
            x = res

        return x


class PKConv2d(PKConvBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        cache_weights=False,
    ) -> None:
        super().__init__(
            is_transpose=False,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            cache_weights=cache_weights,
        )


class PKConvTranspose2d(PKConvBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        cache_weights=False,
    ) -> None:
        super().__init__(
            is_transpose=True,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            cache_weights=cache_weights,
        )
