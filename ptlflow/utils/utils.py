"""Various utility functions."""

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

import logging
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import ptlflow
from ptlflow.utils.external.raft import InputPadder as _InputPadder, forward_interpolate


class InputPadder(_InputPadder):
    """Pads images such that dimensions are divisible by stride.

    This is just a wrapper for ptlflow.utils.external.raft.InputPadder.
    """

    def __init__(
        self,
        dims: Sequence[int],
        stride: int,
        size: Optional[Tuple[int, int]] = None,
        two_side_pad: bool = True,
        pad_mode: str = "replicate",
        pad_value: float = 0.0,
    ) -> None:
        """Initialize InputPadder.

        Parameters
        ----------
        dims : Sequence[int]
            The shape of the original input. It must have at least two elements. It is assumed that the last two dimensions
            are (height, width).
        stride : int
            The number to compute the amount of padding. The padding will be applied so that the input size is divisible
            by stride.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        two_side_pad : bool, default True
            If True, half of the padding goes to left/top and the rest to right/bottom. Otherwise, all the padding goes to the bottom right.
        pad_mode : str, default "replicate"
            How to pad the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.pad.
        pad_value : float, default 0.0
            Used if pad_mode == "constant". The value to fill in the padded area.
        """
        super().__init__(
            dims,
            stride=stride,
            size=size,
            two_side_pad=two_side_pad,
            pad_mode=pad_mode,
            pad_value=pad_value,
        )
        if size is None:
            self.tgt_size = (
                int(math.ceil(float(dims[-2]) / stride)) * stride,
                int(math.ceil(float(dims[-1]) / stride)) * stride,
            )
        else:
            self.tgt_size = size

    def fill(self, x):
        return self.pad(x)

    def unfill(self, x):
        if x.shape[-2] == self.tgt_size[0] and x.shape[-1] == self.tgt_size[1]:
            x = self.unpad(x)
        return x


class InputScaler(object):
    """Scale 2D torch.Tensor input to a target size, and then rescale it back to the original size."""

    def __init__(
        self,
        orig_shape: Tuple[int, int],
        stride: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = False,
    ) -> None:
        """Initialize InputScaler.

        Parameters
        ----------
        orig_shape : Tuple[int, int]
            The shape of the input tensor before the scale. I.e., the shape to which it will be rescaled back.
        stride : Optional[int], optional
            If provided, the input will be resized to the closest larger multiple of stride.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        scale_factor : Optional[float], default 1.0
            This value is only used if stride and size are None. The multiplier that will be applied to the original shape to scale
            the input.
        interpolation_mode : str, default 'bilinear'
            How to perform the interpolation. It must be a value accepted by the 'mode' argument from
            torch.nn.functional.interpolate function.
        interpolation_align_corners : bool, default False
            Whether the interpolation keep the corners aligned. As defined in torch.nn.functional.interpolate.

        See Also
        --------
        torch.nn.functional.interpolate : The function used to scale the inputs.
        """
        super().__init__()
        self.orig_height, self.orig_width = orig_shape[-2:]
        if stride is not None:
            assert size is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height = int(math.ceil(float(self.orig_height) / stride)) * stride
            self.tgt_width = int(math.ceil(float(self.orig_width) / stride)) * stride
        elif size is not None:
            assert stride is None, "only stride OR size can be provided, NOT BOTH."
            self.tgt_height, self.tgt_width = size
        else:
            self.tgt_height = int(self.orig_height * scale_factor)
            self.tgt_width = int(self.orig_width * scale_factor)

        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def fill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to the target size specified during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be scaled. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The scaled input.
        """
        return self._scale_keep_dims(x, (self.tgt_height, self.tgt_width), is_flow)

    def unfill(self, x: torch.Tensor, is_flow: bool = False) -> torch.Tensor:
        """Scale the input to back to the original size defined during initialization.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        return self._scale_keep_dims(x, (self.orig_height, self.orig_width), is_flow)

    def _scale_keep_dims(
        self, x: torch.Tensor, size: Tuple[int, int], is_flow: bool
    ) -> torch.Tensor:
        """Scale the input to a given size while keeping the other dimensions intact.

        Parameters
        ----------
        x : torch.Tensor
            The input to be rescaled back. Its shape must be (..., C, H, W), where ... means any number of dimensions.
        size : Tuple[int, int]
            The target size to scale the input.
        is_flow : bool
            Whether the input is a flow field or not. If it is, then its values are multiplied by the rescale factor.

        Returns
        -------
        torch.Tensor
            The rescaled input.
        """
        x_shape = x.shape
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=size,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
        )

        if is_flow:
            x[:, 0] = x[:, 0] * (float(x.shape[-1]) / x_shape[-1])
            x[:, 1] = x[:, 1] * (float(x.shape[-2]) / x_shape[-2])

        new_shape = list(x_shape)
        new_shape[-2], new_shape[-1] = x.shape[-2], x.shape[-1]
        x = x.view(new_shape)
        return x


def add_datasets_to_parser(
    parser: ArgumentParser, dataset_config_path: str
) -> ArgumentParser:
    """Add dataset paths as parser arguments.

    The dataset names and default paths are loaded from a yaml file.

    Parameters
    ----------
    parser : ArgumentParser
        An initialized parser, this function will add more arguments to it.
    dataset_config_path : str
        The path to the yaml file containing the dataset paths.

    Returns
    -------
    ArgumentParser
        The updated parser.
    """
    with open(dataset_config_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    for name, path in dataset_paths.items():
        parser.add_argument(
            f"--{name}_root_dir",
            type=str,
            default=path,
            help=f"Path to the root of the {name} dataset",
        )
    return parser


def config_logging() -> None:
    """Initialize logging parameters."""
    log_dir = Path("ptlflow_logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / "log_run.txt"),
            logging.StreamHandler(),
        ],
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters of a model.

    Taken from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    Parameters
    ----------
    model : torch.nn.Module
        The model to count the parameters from.

    Returns
    -------
    int
        The number of trainable parameters of the given model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_list_of_available_models_list() -> List[str]:
    """Return a list of the names of the available models.

    Returns
    -------
    list[str]
        The list with the model names.
    """
    return sorted(ptlflow.models_dict.keys())


def make_divisible(v: int, div: int) -> int:
    """Decrease a number v until it is divisible by div.

    Parameters
    ----------
    v : int
        The number to be made divisible.
    div : int
        The divisor value.

    Returns
    -------
    int
        The new value of v which is divisible by div.
    """
    return max(div, v - (v % div))


def release_gpu(tensors_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Detach and move to cpu the tensors from the input dict.

    The non-tensor elements are kept intact.

    Parameters
    ----------
    tensors_dict : Dict[str, Any]
        A dictionary containing the tensors to move.

    Returns
    -------
    Dict[str, Any]
        The same dictionary, but with the tensors moved to cpu.
    """
    for k, v in tensors_dict.items():
        if isinstance(v, torch.Tensor):
            tensors_dict[k] = v.detach().cpu()
            del v
    return tensors_dict


def tensor_dict_to_numpy(
    tensor_dict: Dict[str, torch.Tensor], padder: Optional[InputPadder] = None
) -> Dict[str, np.ndarray]:
    """Convert all tensors into numpy format, changing the shape from CHW to HWC.

    If "flows" is available, then a color representation "flows_viz" is added to the outputs.

    Parameters
    ----------
    tensor_dict : dict[str, torch.Tensor]
        A dictionary with the torch.Tensor inputs/outputs of the model.
    padder: InputPadder
        Helper to unpad the images back to their original sizes.

    Returns
    -------
    dict[str, np.ndarray]
        The torch.Tensor entries from tensor_dict converted to numpy format.
    """
    npy_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if padder is not None:
                v = padder.unpad(v)

            while len(v.shape) > 3:
                v = v[0]
            v = v.permute(1, 2, 0).numpy()
        npy_dict[k] = v
    return npy_dict


def are_shapes_compatible(
    shape1: Sequence[int],
    shape2: Sequence[int],
) -> bool:
    """Check if two tensor shapes are compatible.

    Similar to PyTorch or Numpy, two shapes are considered "compatible" if either they have the same shape or if one shape can be broadcasted into the other.
    We consider two shapes compatible if, and only if:
    1. their shapes have the same length (same number of dimension), and
    2. each dimension size is either equal or at least one of them is one.

    Parameters
    ----------
    shape1 : Sequence[int]
        The dimensions of the first shape.
    shape2 : Sequence[int]
        The dimensions of the second shape.

    Returns
    -------
    bool
        Whether the two given shapes are compatible.
    """
    if len(shape1) != len(shape2):
        return False
    for v1, v2 in zip(shape1, shape2):
        if v1 != 1 and v2 != 1 and v1 != v2:
            return False
    return True


def bgr_val_as_tensor(
    bgr_val: Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor],
    reference_tensor: torch.Tensor,
    bgr_tensor_shape_position: int = -3,
) -> torch.Tensor:
    """Convert multiple types of BGR values given as input to a torch.Tensor where the BGR values are in the same position as a reference tensor.

    The bgr values can be:
    - a single number, in which case it will be repeated three times to represent BGR.
    - a tuple, list, np.ndarray, or torch.Tensor with three elements.

    The resulting tensor will have the BGR values in the same index position as the reference_tensor.
    For example, given a reference tensor with shape [B, 3, H, W] and setting bgr_tensor_shape_position == -3
    indicates that the BGR position in this reference_tensor is at shape index -3, which is equivalent to index 1.
    Given these inputs, the resulting BGR tensor will have shape [1, 3, 1, 1], and the BGR values will be at shape index 1.

    Parameters
    ----------
    bgr_val : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor]
        The BGR values to be converted into a tensor that will have a compatible shape with the input.
    reference_tensor: torch.Tensor
        The tensor with the reference shape to convert the bgr_val.
    bgr_tensor_shape_position : int, default -3
        Which position of the reference_tensor corresponds to the BGR values.
        Typical values are -1 (used in channels-last tensors) or -3 (..., CHW tensors)

    Returns
    -------
    torch.Tensor
        The bgr_val converted into a tensor with a shape compatible with reference_tensor.
    """
    is_compatible = False
    if isinstance(bgr_val, torch.Tensor):
        is_compatible = are_shapes_compatible(bgr_val.shape, reference_tensor.shape)
        assert is_compatible or (len(bgr_val.shape) == 1 and bgr_val.shape[0] == 3)
    elif isinstance(bgr_val, np.ndarray):
        is_compatible = are_shapes_compatible(bgr_val.shape, reference_tensor.shape)
        assert is_compatible or (len(bgr_val.shape) == 1 and bgr_val.shape[0] == 3)
        bgr_val = torch.from_numpy(bgr_val).to(
            dtype=reference_tensor.dtype, device=reference_tensor.device
        )
    elif isinstance(bgr_val, (tuple, list)):
        assert len(bgr_val) == 3
        bgr_val = torch.Tensor(bgr_val).to(
            dtype=reference_tensor.dtype, device=reference_tensor.device
        )
    elif isinstance(bgr_val, (int, float)):
        bgr_val = (
            torch.zeros(3, dtype=reference_tensor.dtype, device=reference_tensor.device)
            + bgr_val
        )

    if not is_compatible:
        bgr_dims = [1] * len(reference_tensor.shape)
        bgr_dims[bgr_tensor_shape_position] = 3
        bgr_val = bgr_val.reshape(bgr_dims)
    return bgr_val


def forward_interpolate_batch(prev_flow: torch.Tensor) -> torch.Tensor:
    """Apply RAFT's forward_interpolate in a batch of torch.Tensors.

    forward_interpolate in the warm start strategy where the previous flow estimation is forward projected
    and then used as initialization for the next estimation.

    Parameters
    ----------
    prev_flow : torch.Tensor
        A 4D tensor [B, 2, H, W] containing a batch of previous flow predictions.

    Returns
    -------
    torch.Tensor
        The previous flow predictions after being forward interpolated.
    """
    forward_flow = []
    for i in range(prev_flow.shape[0]):
        forward_flow.append(
            forward_interpolate(prev_flow[i]).to(
                dtype=prev_flow.dtype, device=prev_flow.device
            )
        )
    forward_flow = torch.stack(forward_flow, 0)
    return forward_flow
