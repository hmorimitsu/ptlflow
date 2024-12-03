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

import json
from pathlib import Path
from typing import Any, IO, Optional, Sequence, Union

import cv2 as cv
import numpy as np
import torch

from .external import flowpy, raft, selflow, flow_IO
from . import flowpy_torch


def flow_to_rgb(
    flow: Union[np.ndarray, torch.Tensor],
    flow_max_radius: Optional[Union[float, torch.Tensor]] = None,
    background: str = "bright",
    custom_colorwheel: Optional[torch.Tensor] = None,
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
        flow_rgb = flowpy.flow_to_rgb(
            flow, flow_max_radius, background, custom_colorwheel
        )
    else:
        flow_rgb = flowpy_torch.flow_to_rgb(
            flow, flow_max_radius, background, custom_colorwheel
        )
    return flow_rgb


def flow_read(
    input_data: Union[Sequence[Any], str, Path, IO],
    format: Optional[str] = None,
) -> np.ndarray:
    """Read optical flow from file.

    This is just a wrapper for flowpy (for .flo and .png) or raft (for pfm), added for convenience.

    Parameters
    ----------
    input_data: Sequence[Any], str, Path or IO
        Path of the file to read or a sequence containing the path and extra information.
    format: str, optional
        Specify in what format the flow is read, accepted formats: "flo", "flo5", "kubric_png", "npz", "pfm", "png".
        If None, it is guessed from the file extension.

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
    ptlflow.utils.external.flow_IO.readFlo5Flow
    write_pfm
    """
    if (format is not None and format == "pfm") or str(input_data).endswith("pfm"):
        return raft.read_pfm(input_data)
    elif (format is not None and format == "flo5") or str(input_data).endswith("flo5"):
        return flow_IO.readFlo5Flow(input_data)
    elif (format is not None and format == "npy") or str(input_data).endswith("npy"):
        return np.load(input_data)
    elif format is not None and format == "kubric_png":
        return read_kubric_flow(input_data[0], input_data[1])
    elif format is not None and format == "viper_npz":
        return read_viper_flow(input_data)
    else:
        return flowpy.flow_read(input_data, format)


def flow_write(
    output_file: Union[str, Path, IO], flow: np.ndarray, format: str = None
) -> None:
    """Write optical flow to file.

    This is just a wrapper for flowpy (for .flo and .png) or selflow (for pfm), added for convenience.

    Parameters
    ----------
    output_file: str, Path or IO
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
    if (format is not None and format == "pfm") or str(output_file).endswith("pfm"):
        selflow.write_pfm(output_file, flow)
    elif (format is not None and format == "flo5") or str(output_file).endswith("flo5"):
        flow_IO.writeFlo5File(flow, output_file)
    elif (format is not None and format == "npy") or str(output_file).endswith("npy"):
        np.save(output_file, flow)
    elif format is not None and format == "viper_npz":
        return write_viper_flow(output_file, flow)
    else:
        flowpy.flow_write(output_file, flow, format)


def read_kubric_flow(
    input_file: Union[str, Path, IO],
    flow_direction: str,
) -> np.ndarray:
    """Read optical flow in Kubric PNG format from file.

    Parameters
    ----------
    input_file: str, Path or IO
        Path of the file to read or file object.
    flow_direction: str
        Either "backward_flow" or "forward_flow".

    Returns
    -------
    numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] is the x-displacement.
        flow[..., 1] is the y-displacement.
    """
    with open(Path(input_file).parent / "data_ranges.json", "r") as f:
        data_ranges = json.load(f)
    flow_min = data_ranges[flow_direction]["min"]
    flow_max = data_ranges[flow_direction]["max"]

    flow = cv.imread(str(input_file), cv.IMREAD_UNCHANGED)[..., 1:].astype(np.float32)
    flow = flow / 65535 * (flow_max - flow_min) + flow_min
    return flow


def read_viper_flow(input_file: Union[str, Path, IO]) -> np.ndarray:
    """Read optical flow in VIPER npz format from file.

    Parameters
    ----------
    input_file: str, Path or IO
        Path of the file to read or file object.

    Returns
    -------
    numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] is the x-displacement.
        flow[..., 1] is the y-displacement.
    """
    flow_npz = np.load(input_file)
    flow = np.stack([flow_npz["u"], flow_npz["v"]], 2).astype(np.float32)
    flow[np.abs(flow) > 512] = np.nan
    return flow


def fb_check(
    forward_flow: Union[np.ndarray, torch.Tensor],
    backward_flow: Union[np.ndarray, torch.Tensor],
    threshold: float = 1.0,
):
    is_np_input = False
    if isinstance(forward_flow, np.ndarray):
        assert len(forward_flow.shape) == 3
        forward_flow = torch.from_numpy(forward_flow).permute(2, 0, 1)[None]
        backward_flow = torch.from_numpy(backward_flow).permute(2, 0, 1)[None]
        is_np_input = True
    assert len(forward_flow.shape) == 4

    coords = torch.meshgrid(
        torch.arange(forward_flow.shape[-2], dtype=torch.float32),
        torch.arange(forward_flow.shape[-1], dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0)[None]

    coords = coords + forward_flow
    coords = coords.permute(0, 2, 3, 1)
    warped_backward_flow, in_mask = raft.bilinear_sampler(
        backward_flow, coords, mask=True
    )
    fb_diff = torch.norm(forward_flow + warped_backward_flow, p=2, dim=1)
    fb_mask = (fb_diff < threshold) & (in_mask[..., 0] > 0.5)

    if is_np_input:
        fb_mask = fb_mask[0].numpy()
    return fb_mask


def write_viper_flow(output_file: Union[str, Path], flow: np.ndarray):
    flow = flow.astype(np.float16)
    np.savez(output_file, u=flow[..., 0], v=flow[..., 1])
