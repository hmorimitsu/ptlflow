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

import torch
import torch.nn as nn
import torch.nn.functional as F


def pklinear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    pre_act_fn=None,
    post_act_fn=None,
    out_ch: Optional[int] = None,
) -> torch.Tensor:
    if out_ch is None:
        out_ch = weight.shape[0]

    if pre_act_fn is not None:
        x = pre_act_fn(x)

    w = weight[:out_ch, : x.shape[-1]]
    b = None
    if bias is not None:
        b = bias[:out_ch]
    x = F.linear(x, weight=w, bias=b)

    if post_act_fn is not None:
        x = post_act_fn(x)

    return x


class PKLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        pre_act_fn=None,
        post_act_fn=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.pre_act_fn = pre_act_fn
        self.post_act_fn = post_act_fn

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.zeros(out_features, in_features, device=device, dtype=dtype)
            ),
        )

        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)),
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, out_ch: Optional[int] = None) -> torch.Tensor:
        return pklinear(
            x=x,
            weight=self.weight,
            bias=self.bias,
            pre_act_fn=self.pre_act_fn,
            post_act_fn=self.post_act_fn,
            out_ch=out_ch,
        )
