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
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import ptlflow
from ptlflow.utils.external.raft import InputPadder as _InputPadder


class InputPadder(_InputPadder):
    """Pads images such that dimensions are divisible by stride.

    This is just a wrapper for ptlflow.utils.external.raft.InputPadder.
    """

    def __init__(
        self,
        dims: Sequence[int],
        stride: int
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
        """
        super().__init__(dims, stride=stride)


class InputScaler(object):
    """Scale 2D torch.Tensor input to a target size, and then rescale it back to the original size."""

    def __init__(
        self,
        orig_shape: Tuple[int, int],
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = 1.0,
        interpolation_mode: str = 'bilinear',
        interpolation_align_corners: bool = False
    ) -> None:
        """Initialize InputScaler.

        Parameters
        ----------
        orig_shape : Tuple[int, int]
            The shape of the input tensor before the scale. I.e., the shape to which it will be rescaled back.
        size : Optional[Tuple[int, int]], optional
            The desired size after scaling defined as (height, width). If not provided, then scale_factor will be used instead.
        scale_factor : Optional[float], default 1.0
            This value is only used if size is None. The multiplier that will be applied to the original shape to scale
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
        if size is None:
            self.tgt_height = int(self.orig_height * scale_factor)
            self.tgt_width = int(self.orig_width * scale_factor)
        else:
            self.tgt_height, self.tgt_width = size

        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def scale(
        self,
        x: torch.Tensor,
        is_flow: bool = False
    ) -> torch.Tensor:
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

    def unscale(
        self,
        x: torch.Tensor,
        is_flow: bool = False
    ) -> torch.Tensor:
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
        self,
        x: torch.Tensor,
        size: Tuple[int, int],
        is_flow: bool
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
            x, size=size, mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners)

        if is_flow:
            x[:, 0] = x[:, 0] * (float(x.shape[-1]) / x_shape[-1])
            x[:, 1] = x[:, 1] * (float(x.shape[-2]) / x_shape[-2])

        new_shape = list(x_shape)
        new_shape[-2], new_shape[-1] = x.shape[-2], x.shape[-1]
        x = x.view(new_shape)
        return x


def add_datasets_to_parser(
    parser: ArgumentParser,
    dataset_config_path: str
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
    with open(dataset_config_path, 'r') as f:
        dataset_paths = yaml.safe_load(f)
    for name, path in dataset_paths.items():
        parser.add_argument(
            f'--{name}_root_dir', type=str, default=path,
            help=f'Path to the root of the {name} dataset')
    return parser


def config_logging() -> None:
    """Initialize logging parameters."""
    log_dir = Path('ptlflow_logs')
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir / 'log_run.txt'),
            logging.StreamHandler()
        ]
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


def make_divisible(
    v: int,
    div: int
) -> int:
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


def release_gpu(
    tensors_dict: Dict[str, Any]
) -> Dict[str, Any]:
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
    tensor_dict: Dict[str, torch.Tensor],
    padder: Optional[InputPadder] = None
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
    npy_dict = {k: v.detach().cpu() for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)}
    if padder is not None:
        npy_dict = {k: padder.unpad(v) for k, v in npy_dict.items()}
    for k, v in npy_dict.items():
        while len(v.shape) > 3:
            v = v[0]
        npy_dict[k] = v
    npy_dict = {k: v.permute(1, 2, 0).numpy() for k, v in npy_dict.items()}
    return npy_dict
