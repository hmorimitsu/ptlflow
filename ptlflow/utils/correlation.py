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

import math
from typing import Optional, Tuple, Union

from einops import rearrange
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
    dilation_patch: Union[int, Tuple[int, int]] = 1,
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
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (
        (dilation_patch, dilation_patch)
        if isinstance(dilation_patch, int)
        else dilation_patch
    )

    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError("Only kernel_size=1 is supported.")
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError("Only dilation=1 is supported.")

    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))

    input2 = F.pad(
        input2,
        (
            dilation_patch[1] * ((patch_size[1] - 1) // 2),
            dilation_patch[1] * (patch_size[1] // 2),
            dilation_patch[0] * ((patch_size[0] - 1) // 2),
            dilation_patch[0] * (patch_size[0] // 2),
        ),
    )

    b, _, h, w = input1.shape
    input1 = input1[:, :, :: stride[0], :: stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw).to(
        dtype=input1.dtype, device=input1.device
    )

    for i in range(0, patch_size[0] * dilation_patch[0], dilation_patch[0]):
        for j in range(0, patch_size[1] * dilation_patch[1], dilation_patch[1]):
            p2 = input2[:, :, i : i + h, j : j + w]
            p2 = p2[:, :, :: stride[0], :: stride[1]]
            corr[:, i // dilation_patch[0], j // dilation_patch[1]] = (input1 * p2).sum(
                dim=1
            )

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
        dilation_patch: Union[int, Tuple[int, int]] = 1,
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

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
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
            input1=input1,
            input2=input2,
            kernel_size=self.kernel_size,
            patch_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dilation_patch=self.dilation_patch,
        )


def _init_coords_grid(flow: torch.Tensor) -> torch.Tensor:
    """Creates a grid of absolute 2D coordinates.

    Parameters
    ----------
    flow : torch.Tensor
        The optical flow field to translate the points from input1. The flow values should be represented in number of pixels
        (do not provide normalized values, e.g. between -1 and 1). It should be a 4D tensor (b, 2, h, w), where
        flow[:, 0] represent the horizontal flow and flow[:, 1] the vertical ones.

    Returns
    -------
    torch.Tensor
        The grid with the 2D coordinates of the pixels.
    """
    b, _, h, w = flow.shape
    coords_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords_grid = torch.stack(coords_grid[::-1], dim=0).to(
        dtype=flow.dtype, device=flow.device
    )
    coords_grid = coords_grid[None].repeat(b, 1, 1, 1)
    return coords_grid


def iter_translated_spatial_correlation_sample(
    input1: torch.Tensor,
    input2: torch.Tensor,
    flow: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    kernel_size: Union[int, Tuple[int, int]] = 1,
    patch_size: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    dilation_patch: Union[int, Tuple[int, int]] = 1,
    coords_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply spatial correlation sampling with translation from input1 to input2 using iteration in PyTorch.

    This operation is equivalent to first translating the points from input1 using the given flow, and then doing a local
    correlation sampling around the translated points.

    This allows us to do correlation sampling without warping the second input.

    Every parameter except input1, input2, and flow can be either single int or a pair of int. For more information about
    Spatial Correlation Sampling (without translation), see this page: https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Parameters
    ----------
    input1 : torch.Tensor
        The origin feature map.
    input2 : torch.Tensor
        The target feature map.
    flow : Optional[torch.Tensor]
        This argument and "coords" are mutually exclusive, only one of them can be not None.
        The optical flow field to translate the points from input1. The flow values should be represented in number of pixels
        (do not provide normalized values, e.g. between -1 and 1). It should be a 4D tensor (b, 2, h, w), where
        flow[:, 0] represent the horizontal flow and flow[:, 1] the vertical ones.
    coords : torch.Tensor
        This argument and "flow" are mutually exclusive, only one of them can be not None.
        This value should be equivalent to "flow" + "coords_grid".
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
    coords_grid : Optional[torch.Tensor], default None
        A tensor with the same shape as flow containing a grid of 2D coordinates of the pixels. This can be created using torch.meshgrid.
        This parameter is optional. If not provided, the grid will be created internally. Only useful if the grid can be buffered somewhere
        to avoid recreating it at every call.

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
    assert (flow is None and coords is not None) or (
        flow is not None and coords is None
    )
    # Make inputs be tuples
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dilation_patch = (
        (dilation_patch, dilation_patch)
        if isinstance(dilation_patch, int)
        else dilation_patch
    )

    if kernel_size[0] != 1 or kernel_size[1] != 1:
        raise NotImplementedError("Only kernel_size=1 is supported.")
    if dilation[0] != 1 or dilation[1] != 1:
        raise NotImplementedError("Only dilation=1 is supported.")

    if max(padding) > 0:
        input1 = F.pad(input1, (padding[1], padding[1], padding[0], padding[0]))
        input2 = F.pad(input2, (padding[1], padding[1], padding[0], padding[0]))

    b, _, h, w = input2.shape
    input1 = input1[:, :, :: stride[0], :: stride[1]]
    sh, sw = input1.shape[2:4]
    corr = torch.zeros(b, patch_size[0], patch_size[1], sh, sw).to(
        dtype=input1.dtype, device=input1.device
    )

    if coords is None:
        if coords_grid is None:
            coords_grid = _init_coords_grid(flow)
        coords = coords_grid + flow

    cx = 2 * coords[:, 0] / (w - 1) - 1
    cy = 2 * coords[:, 1] / (h - 1) - 1

    offset = (
        dilation_patch[0] * ((patch_size[0] - 1) // 2),
        dilation_patch[1] * ((patch_size[1] - 1) // 2),
    )

    for i in range(0, patch_size[0] * dilation_patch[0], dilation_patch[0]):
        for j in range(0, patch_size[1] * dilation_patch[1], dilation_patch[1]):
            grid = torch.stack(
                [
                    cx + 2 * (j - offset[1]) / float(w - 1),
                    cy + 2 * (i - offset[0]) / float(h - 1),
                ],
                dim=-1,
            )
            p2 = F.grid_sample(input2, grid, mode="bilinear", align_corners=True)
            p2 = p2[:, :, :: stride[0], :: stride[1]]
            corr[:, i // dilation_patch[0], j // dilation_patch[1]] = (input1 * p2).sum(
                dim=1
            )

    return corr


class IterTranslatedSpatialCorrelationSampler(nn.Module):
    """Apply translated spatial correlation sampling from two inputs using iteration in PyTorch.

    This operation is equivalent to first translating the points from input1 using the given flow, and then doing a local
    correlation sampling around the translated points.

    This allows us to do correlation sampling without warping the second input.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        patch_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        dilation_patch: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        """Initialize IterTranslatedSpatialCorrelationSampler.

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
        super(IterTranslatedSpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

        self.coords_grid = None

    def forward(
        self, input1: torch.Tensor, input2: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        """Compute the correlation sampling from input1 to input2.

        Parameters
        ----------
        input1 : torch.Tensor
            The origin feature map.
        input2 : torch.Tensor
            The target feature map.
        flow : torch.Tensor
            The optical flow field to translate the points from input1. The flow values should be represented in number of pixels
            (do not provide normalized values, e.g. between -1 and 1). It should be a 4D tensor (b, 2, h, w), where
            flow[:, 0] represent the horizontal flow and flow[:, 1] the vertical ones.

        Returns
        -------
        torch.Tensor
            Result of correlation sampling.
        """
        b, _, h, w = flow.shape
        if (
            self.coords_grid is None
            or self.coords_grid.shape[2] != h
            or self.coords_grid.shape[3] != w
        ):
            self.coords_grid = _init_coords_grid(flow)
        if self.coords_grid.shape[0] != b:
            self.coords_grid = self.coords_grid[:1].repeat(b, 1, 1, 1)

        return iter_translated_spatial_correlation_sample(
            input1=input1,
            input2=input2,
            flow=flow,
            kernel_size=self.kernel_size,
            patch_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            dilation_patch=self.dilation_patch,
            coords_grid=self.coords_grid,
        )


class IterativeCorrBlock(nn.Module):
    """Another wrapper for iter_translated_spatial_correlation_sample.

    This block is designed to mimic the operations of RAFT's AlternateCorrBlock package (see ptlflow/models/raft/corr.py).
    This block can be used when alt_cuda_corr has not been compiled (see ptlflow/utils/external/alt_cuda_corr).

    IMPORTANT: this implementation is slower than alt_cuda_corr.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        radius: int = 1,
        num_levels: int = 1,
    ):
        """Initialize IterativeCorrBlock.

        Parameters
        ----------
        fmap1 : torch.Tensor
            The origin feature map.
        fmap2 : torch.Tensor
            The target feature map.
        radius : int, default 1
            The radius if the correlation patch. The patch_size will be 2 * radius + 1.
        num_levels : int, default 1
            Number of correlation pooling levels to use (see ptlflow/models/raft/corr.py).
        """
        super(IterativeCorrBlock, self).__init__()

        self.patch_size = 2 * radius + 1
        self.num_levels = num_levels

        self.pyramid = [(fmap1, fmap2)]
        for _ in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def forward(self, coords):
        """Compute the correlation sampling from input1 to input2.

        Parameters
        ----------
        coords : torch.Tensor
            The addition (optical flow + coords_grid) to translate the points from input1. The coords values should be represented in number of pixels
            (do not provide normalized values, e.g. between -1 and 1). It should be a 4D tensor (b, 2, h, w), where
            coords[:, 0] represent the x axis and flow[:, 1] the y axis.

        Returns
        -------
        torch.Tensor
            Result of correlation sampling.
        """
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            fmap1_i = self.pyramid[0][0]
            fmap2_i = self.pyramid[i][1]

            coords_i = coords / 2**i
            corr = iter_translated_spatial_correlation_sample(
                input1=fmap1_i,
                input2=fmap2_i,
                coords=coords_i,
                patch_size=self.patch_size,
            )
            corr = rearrange(corr, "b c d h w -> b (d c) h w")
            corr_list.append(corr)

        corr = torch.cat(corr_list, dim=1)
        return corr / math.sqrt(dim)
