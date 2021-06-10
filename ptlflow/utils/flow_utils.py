"""Basic functions to handle optical flow."""

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

import pathlib
from typing import IO, Optional, Union

import numpy as np
import torch

from .external import flowpy, raft, selflow
from . import flowpy_torch


def flow_to_rgb(
    flow: Union[np.ndarray, torch.Tensor],
    flow_max_radius: Optional[Union[float, torch.Tensor]] = None,
    background: str = 'bright',
    custom_colorwheel: Optional[torch.Tensor] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Convert flows to RGB images.

    The input can be either numpy or torch tensors. This function is just a wrapper for flowpy and flowpy_torch.

    Parameters
    ----------
    flow : np.ndarray or torch.Tensor
        If flow is a numpy array, then it must have 3 dimensions HWC (Height, Width, Channels) - notice it is channels last.
        If it is a torch tensor, then it has at least 3 dimensions in the ...CHW (..., Channels, Height, Width) layout,
        where ... represents any number of dimensions.
        Channel 0 should be the x-displacement.
        Channel 1 should be the y-displacement.
    flow_max_radius : float or torch.Tensor, optional
        Set the radius that gives the maximum color intensity, useful for comparing different flows.
        Default: The normalization is based on the input flow maximum radius per batch element.
    background : str, default 'bright'
        States if zero-valued flow should look 'bright' or 'dark'.
    custom_colorwheel : np.ndarray or torch.Tensor
        Use a custom colorwheel for specific hue transition lengths. By default, the default transition lengths are used.

    Returns
    -------
    np.ndarray or torch.Tensor
        The RGB image representing the flow. It keeps the same dimensions and type as the input.

    See Also
    --------
    ptlflow.utils.external.flowpy.flow_to_rgb
    ptlflow.utils.flowpy_torch.flow_to_rgb
    """
    if isinstance(flow, np.ndarray):
        flow_rgb = flowpy.flow_to_rgb(flow, flow_max_radius, background, custom_colorwheel)
    else:
        flow_rgb = flowpy_torch.flow_to_rgb(flow, flow_max_radius, background, custom_colorwheel)
    return flow_rgb


def flow_read(
    input_file: Union[str, pathlib.Path, IO],
    format: str = None
) -> np.ndarray:
    """Read optical flow from file.

    This is just a wrapper for flowpy (for .flo and .png) or raft (for pfm), added for convenience.

    Parameters
    ----------
    input_file: str, pathlib.Path or IO
        Path of the file to read or file object.
    format: str, optional
        Specify in what format the flow is read, accepted formats: "png", "flo", or "pfm".
        If None, it is guessed on the file extension.

    Returns
    -------
    numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] is the x-displacement.
        flow[..., 1] is the y-displacement.

    See Also
    --------
    ptlflow.utils.external.flowpy.flow_read
    ptlflow.utils.external.raft.read_pfm
    write_pfm
    """
    if (format is not None and format == 'pfm') or str(input_file).endswith('pfm'):
        return raft.read_pfm(input_file)
    else:
        return flowpy.flow_read(input_file, format)


def flow_write(
    output_file: Union[str, pathlib.Path, IO],
    flow: np.ndarray,
    format: str = None
) -> None:
    """Write optical flow to file.

    This is just a wrapper for flowpy (for .flo and .png) or selflow (for pfm), added for convenience.

    Parameters
    ----------
    output_file: str, pathlib.Path or IO
        Path of the file to write or file object.
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    format: str, optional
        Specify in what format the flow is written, accepted formats: "png" or "flo"
        If None, it is guessed on the file extension

    See Also
    --------
    ptlflow.utils.external.flowpy.flow_write
    """
    if (format is not None and format == 'pfm') or str(output_file).endswith('pfm'):
        selflow.write_pfm(output_file, flow)
    else:
        flowpy.flow_write(output_file, flow, format)
