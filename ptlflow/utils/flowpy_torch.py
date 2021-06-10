"""

This code is a port to PyTorch of the flow to RGB convertion from flowpy.

https://gitlab-research.centralesupelec.fr/2018seznecm/flowpy

"""

# =============================================================================
# Copyright 2021 Henrique Morimitsu
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
from collections import namedtuple
from typing import Optional, Tuple, Union

import torch

from .external.flowpy import make_colorwheel

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[Union[float, torch.Tensor]] = None,
    background: str = 'bright',
    custom_colorwheel: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Create a RGB representation of an optical flow.

    Parameters
    ----------
    flow : torch.Tensor
        Flow with at least 3 dimensions in the ...CHW (..., Channels, Height, Width) layout, where ... represents any number
        of dimensions.
        flow[..., 0, h, w] should be the x-displacement
        flow[..., 1, h, w] should be the y-displacement
    flow_max_radius : float or torch.Tensor, optional
        Set the radius that gives the maximum color intensity, useful for comparing different flows.
        Default: The normalization is based on the input flow maximum radius per batch element.
    background : str, default 'bright'
        States if zero-valued flow should look 'bright' or 'dark'.
    custom_colorwheel : torch.Tensor
        Use a custom colorwheel for specific hue transition lengths. By default, the default transition lengths are used.

    Returns
    -------
    torch.Tensor
        The RGB representation of the flow. RGB values are float in the [0, 1] interval. The output shape is (..., 3, H, W).

    Raises
    ------
    ValueError
        If the background choice is invalid.

    See Also
    --------
    ptlflow.utils.external.flowpy.make_colorwheel : How the colorwheel can be generated.
    """
    valid_backgrounds = ('bright', 'dark')
    if background not in valid_backgrounds:
        raise ValueError(f'background should be one the following: {valid_backgrounds}, not {background}')

    wheel = make_colorwheel() if custom_colorwheel is None else custom_colorwheel
    wheel = torch.from_numpy(wheel).to(dtype=flow.dtype, device=flow.device) / 255

    orig_shape = flow.shape
    if len(orig_shape) == 3:
        flow = flow[None]
    elif len(orig_shape) > 4:
        flow = flow.view(-1, 2, flow.shape[-2], flow.shape[-1])

    complex_flow = flow[:, 0] + 1j * flow[:, 1]
    complex_flow, nan_mask = _replace_nans(complex_flow)

    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)

    if flow_max_radius is None:
        flow_max_radius = radius.view(radius.shape[0], -1).max(dim=1)[0]
    else:
        flow_max_radius = torch.zeros(radius.shape[0]).to(dtype=flow.dtype, device=flow.device) + flow_max_radius

    flow_max_radius = torch.clamp(flow_max_radius[:, None, None], 1)
    radius /= flow_max_radius

    ncols = len(wheel)

    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * math.pi
    angle = angle * ((ncols - 1) / (2 * math.pi))

    # Make the wheel cyclic for interpolation
    wheel = torch.cat((wheel, wheel[:1]), dim=0)

    # Interpolate the hues
    angle_fractional = torch.frac(angle)
    angle_floor = torch.floor(angle)
    angle_ceil = torch.ceil(angle)
    angle_fractional = angle_fractional.view((angle_fractional.shape)+(1,))
    float_hue = (wheel[angle_floor.long()] * (1 - angle_fractional)
                 + wheel[angle_ceil.long()] * angle_fractional)

    ColorizationArgs = namedtuple('ColorizationArgs', [
        'move_hue_valid_radius',
        'move_hue_oversized_radius',
        'invalid_color'])

    def _move_hue_on_v_axis(hues: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return hues * torch.unsqueeze(factors, -1)

    def _move_hue_on_s_axis(hues: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return 1. - torch.unsqueeze(factors, -1) * (1. - hues)

    if background == 'dark':
        parameters = ColorizationArgs(_move_hue_on_v_axis, _move_hue_on_s_axis,
                                      torch.zeros(3).to(dtype=flow.dtype, device=flow.device)+1)
    else:
        parameters = ColorizationArgs(_move_hue_on_s_axis, _move_hue_on_v_axis,
                                      torch.zeros(3).to(dtype=flow.dtype, device=flow.device))

    colors = parameters.move_hue_valid_radius(float_hue, radius)

    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    colors[nan_mask] = parameters.invalid_color

    output_shape = tuple(3 if i == len(orig_shape)-3 else orig_shape[i] for i in range(len(orig_shape)))
    colors = colors.permute(0, 3, 1, 2).contiguous()
    colors = colors.view(output_shape)

    return colors


def _replace_nans(
    array: torch.Tensor,
    value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    nan_mask = torch.isnan(array)
    array[nan_mask] = value

    return array, nan_mask
