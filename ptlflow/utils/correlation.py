"""Perform spatial correlation to generate a cost volume.

This is meant to be a simplified version of the SpatialCorrelationSampler from
https://github.com/ClementPinard/Pytorch-Correlation-extension.

This version is implemented purely in PyTorch. However, it only supports correlation with 1x1 kernels.
It is also not as efficient as the original SpatialCorrelationSampler.
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

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def iter_spatial_correlation_sample(
    input1: torch.Tensor,
    input2: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 1,
    patch_size: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    dilation_patch: Union[int, Tuple[int, int]] = 1
) -> torch.Tensor:
    """Apply spatial correlation sampling from input1 to input2 using iteration in PyTorch.

    This docstring is taken and adapted from the original package.

    Every parameter except input1 and input2 can be either single int or a pair of int. For more information about
    Spatial Correlation Sampling, see this page. https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Parameters
    ----------
    input1 : torch.Tensor
        The origin feature map.
    input2 : torch.Tensor
        The target feature map.
    kernel_size : Union[int, Tuple[int, int]], default 1
        Total size of your correlation kernel, in pixels
    patch_size : Union[int, Tuple[int, int]], default 1
        Total size of your patch, determining how many different shifts will be applied.
    stride : Union[int, Tuple[int, int]], default 1
        Stride of the spatial sampler, will modify output height and width.
    padding : Union[int, Tuple[int, int]], default 0
        Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
    dilation : Union[int, Tuple[int, int]], default 1
        Similar to dilation in convolution.
    dilation_patch : Union[int, Tuple[int, int]], default 1
        Step for every shift in patch.

    Returns
    -------
    torch.Tensor
        Result of correlation sampling.

    Raises
    ------
    NotImplementedError
        If kernel_size != 1.
    NotImplementedError
        If dilation != 1.
    """
    # Make inputs be tuples
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (dilation_patch, dilation_patch) if isinstance(dilation_patch, int) else dilation_patch

    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError('Only kernel_size=1 is supported.')
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError('Only dilation=1 is supported.')
    if (patch_size[0] % 2) == 0 or (patch_size[1] % 2) == 0:
        raise NotImplementedError('Only odd patch sizes are supperted.')

    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))

    max_displacement = (dilation_patch[0] * (patch_size[0] - 1) // 2, dilation_patch[1] * (patch_size[1] - 1) // 2)
    input2 = F.pad(input2, (max_displacement[1], max_displacement[1], max_displacement[0], max_displacement[0]))

    b, _, h, w = input1.shape
    input1 = input1[:, :, ::stride[0], ::stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw).to(dtype=input1.dtype, device=input1.device)

    for i in range(0, 2*max_displacement[0]+1, dilation_patch[0]):
        for j in range(0, 2*max_displacement[1]+1, dilation_patch[1]):
            p2 = input2[:, :, i:i+h, j:j+w]
            p2 = p2[:, :, ::stride[0], ::stride[1]]
            corr[:, i//dilation_patch[0], j//dilation_patch[1]] = (input1 * p2).sum(dim=1)

    return corr


class IterSpatialCorrelationSampler(nn.Module):
    """Apply spatial correlation sampling from two inputs using iteration in PyTorch."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        patch_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        dilation_patch: Union[int, Tuple[int, int]] = 1
    ) -> None:
        """Initialize IterSpatialCorrelationSampler.

        Parameters
        ----------
        kernel_size : Union[int, Tuple[int, int]], default 1
            Total size of your correlation kernel, in pixels
        patch_size : Union[int, Tuple[int, int]], default 1
            Total size of your patch, determining how many different shifts will be applied.
        stride : Union[int, Tuple[int, int]], default 1
            Stride of the spatial sampler, will modify output height and width.
        padding : Union[int, Tuple[int, int]], default 0
            Padding applied to input1 and input2 before applying the correlation sampling, will modify output height and width.
        dilation : Union[int, Tuple[int, int]], default 1
            Similar to dilation in convolution.
        dilation_patch : Union[int, Tuple[int, int]], default 1
            Step for every shift in patch.
        """
        super(IterSpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the correlation sampling from input1 to input2.

        Parameters
        ----------
        input1 : torch.Tensor
            The origin feature map.
        input2 : torch.Tensor
            The target feature map.

        Returns
        -------
        torch.Tensor
            Result of correlation sampling.
        """
        return iter_spatial_correlation_sample(
            input1=input1, input2=input2, kernel_size=self.kernel_size, patch_size=self.patch_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation, dilation_patch=self.dilation_patch)
