"""Functions taken and adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py."""

# Original license below

# BSD 3-Clause License

# Copyright (c) 2020, princeton-vl
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re

import numpy as np
from scipy import interpolate
import torch
import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by stride."""

    def __init__(
        self,
        dims,
        stride=8,
        two_side_pad=True,
        pad_mode="replicate",
        pad_value=0.0,
        size=None,
    ):
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        ht, wd = dims[-2:]
        if size is None:
            pad_ht = (((ht // stride) + 1) * stride - ht) % stride
            pad_wd = (((wd // stride) + 1) * stride - wd) % stride
        else:
            pad_ht = size[0] - ht
            pad_wd = size[1] - wd
        if two_side_pad:
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, x):
        in_shape = x.shape
        if len(in_shape) > 4:
            x = x.view(-1, *in_shape[-3:])
        x = F.pad(x, self._pad, mode=self.pad_mode, value=self.pad_value)
        if len(in_shape) > 4:
            x = x.view(*in_shape[:-2], *x.shape[-2:])
        return x

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def read_pfm(file_path: str) -> np.ndarray:
    """Read optical flow files in PFM format.

    Taken and adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py.

    Parameters
    ----------
    file_path : str
        Path to the optical flow file.

    Returns
    -------
    np.ndarray
        The optical flow in HC format.

    Raises
    ------
    ValueError
        If the file is not in valid PFM format.
    ValueError
        If the file header is corrupted.
    """
    with open(file_path, "rb") as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(rb"^(\d+)\s(\d+)\s$", f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError("Malformed PFM header.")

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

    # Mark invalid pixels as NaN
    mask = np.tile(data[:, :, 2:3], (1, 1, 2))
    flow = data[:, :, :2].astype(np.float32)
    flow[mask > 0.5] = float("nan")
    return flow


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()
