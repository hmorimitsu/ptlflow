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
import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by stride."""
    def __init__(self, dims, stride=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // stride) + 1) * stride - self.ht) % stride
        pad_wd = (((self.wd // stride) + 1) * stride - self.wd) % stride
        if len(dims) < 5:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht, 0, 0]

    def pad(self, x):
        return F.pad(x, self._pad, mode='replicate')

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def read_pfm(
    file_path: str
) -> np.ndarray:
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
    with open(file_path, 'rb') as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

    # Mark invalid pixels as NaN
    mask = np.tile(data[:, :, 2:3], (1, 1, 2))
    flow = data[:, :, :2].astype(np.float32)
    flow[mask > 0.5] = float('nan')
    return flow
