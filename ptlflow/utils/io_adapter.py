"""Helper to facilitate the use of inputs and outputs of the models."""

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ptlflow.data.flow_transforms import ToTensor
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.utils import InputPadder, InputScaler


class IOAdapter(object):
    """Handle the inputs and outputs of optical flow models."""

    def __init__(
        self,
        model: BaseModel,
        input_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None,
        target_scale_factor: Optional[float] = None,
        interpolation_mode: str = 'bilinear',
        interpolation_align_corners: bool = False,
        cuda: bool = False
    ) -> None:
        """Initialize IOAdapter.

        Parameters
        ----------
        model : BaseModel
            An instance of optical flow mode that will be used for estimation.
        input_size : Tuple[int, int]
            The shape of the original inputs, must be a tuple with at least two elements. It is assumed that the last two
            elements are (height, with).
        target_size : Optional[Tuple[int, int]], optional
            If provided, the inputs will be resized to target_size. target_size is defined as a tuple (height, with).
        target_scale_factor : Optional[float], default 1.0
            This value is only used if size is None. The multiplier that will be applied to the original shape to scale
            the input.
        interpolation_mode : str, default 'bilinear'
            How to perform the interpolation. It must be a value accepted by the 'mode' argument from
            torch.nn.functional.interpolate function.
        interpolation_align_corners : bool, default False
            Whether the interpolation keep the corners aligned. As defined in torch.nn.functional.interpolate.
        cuda : bool
            If True, the input tensors are transferred to GPU (if a GPU is available).
        """
        self.output_stride = model.output_stride
        self.target_size = target_size
        self.target_scale_factor = target_scale_factor
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.cuda = cuda

        self.transform = ToTensor()
        padder_input_size = input_size
        self.scaler = None
        if (target_size is not None and min(target_size) > 0) or (target_scale_factor is not None and target_scale_factor > 0):
            self.scaler = InputScaler(
                orig_shape=input_size, size=target_size, scale_factor=target_scale_factor,
                interpolation_mode=interpolation_mode, interpolation_align_corners=interpolation_align_corners)

            if (target_size is not None and min(target_size) > 0):
                padder_input_size = target_size
            else:
                padder_input_size = (int(target_scale_factor*input_size[0]), int(target_scale_factor*input_size[1]))

        self.padder = InputPadder(padder_input_size, model.output_stride)

    def prepare_inputs(
        self,
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        flows: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Union[np.ndarray, List[np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """Transform numpy inputs into the input format of theoptical flow models.

        This basically consists on tranform the numpy arrays into torch tensors, and then putting them into a dict.

        Parameters
        ----------
        images : Union[np.ndarray, List[np.ndarray]]
            One or more images to use to estimate the optical flow. Typically, it will be a least with two images in the
            HWC format.
        flows : Optional[Union[np.ndarray, List[np.ndarray]]], optional
            One or more groundtruth optical flow, which can be used for validation. Typically it will be an array HWC.
        inputs : Optional[Dict[str, Any]]
            Dict containing input tensors or other metadata. Only the tensors will be transformed.
        kwargs : Union[np.ndarray, List[np.ndarray]]
            Any other array inputs can be provided as keyworded arguments. This function will create an entry in the input dict
            for each keyworded array given.

        Returns
        -------
        Dict[str, Any]
            The inputs converted and transformed to the input format of the optical flow models.
        """
        if inputs is None:
            inputs = {'images': images, 'flows': flows}
            inputs.update(kwargs)
            keys_to_remove = []
            for k, v in inputs.items():
                if v is None or len(v) == 0:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del inputs[k]
            inputs = self.transform(inputs)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                while len(v.shape) < 5:
                    v = v.unsqueeze(0)
                if self.scaler is not None:
                    v = self.scaler.scale(v, is_flow=k.startswith('flow'))
                v = self.padder.pad(v)
                inputs[k] = v

        inputs = self._to_cuda(inputs)

        return inputs

    def unpad_and_unscale(
        self,
        outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Remove padding and scaling to restore the original shapes.

        The inputs and outputs obtained from prepare_inputs() may have been padded and scaled.
        This function can be used to revert those operations.

        Typically, this function should be used on both the inputs and outputs of the model after the model generated the
        predictions.

        Parameters
        ----------
        outputs : Dict[str, Any]
            It can be either the inputs or the outputs of optical flow models. All tensors will have the extra padding and
            rescaling removed.

        Returns
        -------
        Dict[str, Any]
            The same tensors, with the padding and scaling removed.

        See Also
        --------
        prepare_inputs
        """
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                v = self.padder.unpad(v)
                if self.scaler is not None:
                    v = self.scaler.unscale(v, is_flow=k.startswith('flow'))
                outputs[k] = v

        return outputs

    def _to_cuda(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.cuda:
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                logging.warning(
                    'IOAdapter was asked to use cuda, but torch.cuda.is_available() == False. Tensors will remain on CPU.')
        return inputs
