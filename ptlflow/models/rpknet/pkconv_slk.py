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
# Adapted from VAN: https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py
# =============================================================================

import math

import torch
import torch.nn as nn

from .local_timm.drop import DropPath
from .local_timm.layer_helpers import to_2tuple
from .local_timm.norm import GroupNorm
from .local_timm.weight_init import trunc_normal_
from .pkconv import PKConv2d


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        skip_dw=False,
        cache_pkconv_weights=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = PKConv2d(
            in_features, hidden_features, 1, cache_weights=cache_pkconv_weights
        )
        self.dwconv = None
        if not skip_dw:
            self.dwconv = DWConv(
                hidden_features, cache_pkconv_weights=cache_pkconv_weights
            )
        self.act = act_layer()
        self.fc2 = PKConv2d(
            hidden_features, out_features, 1, cache_weights=cache_pkconv_weights
        )
        self.drop = nn.Dropout(drop)

        self.in_hid_factor = float(hidden_features) / in_features
        self.hid_out_factor = float(out_features) / hidden_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):
        out_ch = int(self.in_hid_factor * x.shape[1])
        x = self.fc1(x, out_ch=out_ch)
        if self.dwconv is not None:
            x = self.dwconv(x, out_ch=out_ch)
        x = self.act(x)
        x = self.drop(x)
        out_ch = int(self.hid_out_factor * x.shape[1])
        x = self.fc2(x, out_ch=out_ch)
        x = self.drop(x)
        return x


class SLKUnitCore(nn.Module):
    def __init__(
        self,
        dim,
        ksize=23,
        cache_pkconv_weights=False,
    ):
        super().__init__()
        self.conv1_branches = nn.ModuleList()
        self.conv1_branches.append(
            PKConv2d(
                dim,
                dim,
                (ksize, 1),
                padding=(ksize // 2, 0),
                groups=dim,
                cache_weights=cache_pkconv_weights,
            )
        )
        self.conv2_branches = nn.ModuleList()
        self.conv2_branches.append(
            PKConv2d(
                dim,
                dim,
                (1, ksize),
                padding=(0, ksize // 2),
                groups=dim,
                cache_weights=cache_pkconv_weights,
            )
        )
        self.conv_out = PKConv2d(dim, dim, 1, cache_weights=cache_pkconv_weights)

    def forward(self, x, out_ch=None):
        y = x
        y = y + self.conv1_branches[0](y, out_ch=out_ch)
        y = y + self.conv2_branches[0](y, out_ch=out_ch)
        y = self.conv_out(y, out_ch=out_ch)
        y = y + x
        return y


class SLKUnit(nn.Module):
    def __init__(
        self,
        dim,
        cache_pkconv_weights=False,
    ):
        super().__init__()

        self.proj_1 = PKConv2d(dim, dim, 1, cache_weights=cache_pkconv_weights)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SLKUnitCore(
            dim,
            cache_pkconv_weights=cache_pkconv_weights,
        )
        self.proj_2 = PKConv2d(dim, dim, 1, cache_weights=cache_pkconv_weights)

    def forward(self, x):
        out_ch = x.shape[1]
        shorcut = x.clone()
        x = self.proj_1(x, out_ch=out_ch)
        x = self.activation(x)
        x = self.spatial_gating_unit(x, out_ch=out_ch)
        x = self.proj_2(x, out_ch=out_ch)
        x = x + shorcut
        return x


class SLK(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        cache_pkconv_weights=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(num_channels=dim)
        self.attn = SLKUnit(
            dim,
            cache_pkconv_weights=cache_pkconv_weights,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(num_channels=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            cache_pkconv_weights=cache_pkconv_weights,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, out_ch=None):
        x = x + self.drop_path(
            self.layer_scale_1[: x.shape[1]].unsqueeze(-1).unsqueeze(-1)
            * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2[: x.shape[1]].unsqueeze(-1).unsqueeze(-1)
            * self.mlp(self.norm2(x))
        )
        return x


class LayerTransition(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=3,
        stride=2,
        in_chans=64,
        embed_dim=64,
        norm_layer=GroupNorm,
        cache_pkconv_weights=False,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = PKConv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
            cache_weights=cache_pkconv_weights,
        )
        self.norm = norm_layer(num_channels=embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, out_ch=None):
        x = self.proj(x, out_ch=out_ch)
        x = self.norm(x)
        return x


class DWConv(nn.Module):
    def __init__(
        self,
        dim=768,
        cache_pkconv_weights=False,
    ):
        super(DWConv, self).__init__()
        self.dwconv = PKConv2d(
            dim, dim, 3, 1, 1, bias=True, groups=dim, cache_weights=cache_pkconv_weights
        )

    def forward(self, x, out_ch=None):
        x = self.dwconv(x, out_ch=out_ch)
        return x


class PKConvSLK(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        stride=1,
        depth=2,
        cache_pkconv_weights=False,
    ):
        super(PKConvSLK, self).__init__()
        self.down = None
        if stride > 1 or in_chs != out_chs:
            patch_size = 1
            if stride > 1:
                patch_size = 3
            self.down = LayerTransition(
                patch_size=patch_size,
                stride=stride,
                in_chans=in_chs,
                embed_dim=out_chs,
                norm_layer=norm_layer,
                cache_pkconv_weights=cache_pkconv_weights,
            )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                SLK(
                    dim=out_chs,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    cache_pkconv_weights=cache_pkconv_weights,
                )
            )
        self.norm = norm_layer(num_channels=out_chs)

    def forward(self, x, out_ch=None):
        if self.down is not None:
            x = self.down(x, out_ch=out_ch)
        for blk in self.blocks:
            x = blk(x, out_ch=out_ch)
        x = self.norm(x)
        return x
