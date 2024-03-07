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
# Adapted from IRR: https://github.com/visinf/irr/blob/master/models/pwc_modules.py
#
# Modifications by Henrique Morimitsu:
# - Remove operations not related to rescaling
# =============================================================================

import torch
import torch.nn.functional as F


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    if inputs.shape[-2] != h or inputs.shape[-1] != w:
        inputs = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return inputs


def rescale_flow(flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im)
        v_scale = float(flow.size(2) / height_im)
    else:
        u_scale = float(width_im / flow.size(3))
        v_scale = float(height_im / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u = u * u_scale
    v = v * v_scale

    return torch.cat([u, v], dim=1)
