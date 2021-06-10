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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ptlflow.data.flow_transforms import Compose, Resize, ToTensor
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.utils import InputPadder


class IOAdapter(object):
    """Handle the inputs and outputs of optical flow models."""

    def __init__(
        self,
        model: BaseModel,
        input_size: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None
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
        """
        self.output_stride = model.output_stride
        self.target_size = target_size
        self.transform = Compose([ToTensor(), Resize(target_size)])
        self.padder = InputPadder(input_size, model.output_stride)

    def prepare_inputs(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        flows: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
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
        kwargs : Union[np.ndarray, List[np.ndarray]]
            Any other array inputs can be provided as keyworded arguments. This function will create an entry in the input dict
            for each keyworded array given.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs converted to the input format of the optical flow models.
        """
        inputs = {'images': images, 'flows': flows}
        inputs.update(kwargs)
        keys_to_remove = []
        for k, v in inputs.items():
            if v is None or len(v) == 0:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del inputs[k]

        inputs = self.transform(inputs)
        inputs = {k: self.padder.pad(v) for k, v in inputs.items()}
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        return inputs

    def unpad(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove padding to restore the original shapes.

        The inputs and outputs obtained from prepare_inputs() will usually have some additional padding which was added to make
        the input shape divisible by the model output stride. This function can be used to remove that padding.

        Typically, this function should be used on both the inputs and outputs of the model after the model generated the
        predictions.

        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            It can be either the inputs or the outputs of optical flow models. All tensors will have the extra padding removed.

        Returns
        -------
        Dict[str, torch.Tensor]
            The same tensors, with the padding removed.

        See Also
        --------
        prepare_inputs
        """
        outputs = {k: self.padder.unpad(v) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        return outputs
