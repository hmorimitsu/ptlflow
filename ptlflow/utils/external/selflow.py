"""Function taken and adapted from https://github.com/ppliuboy/SelFlow/blob/master/flowlib.py."""

# Original license below

# MIT License

# Copyright (c) 2019 Pengpeng Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

import numpy as np


def write_pfm(output_path, flow, scale=1):
    with open(output_path, 'wb') as file:
        if flow.dtype.name != 'float32':
            raise TypeError('flow dtype must be float32.')
        if not (len(flow.shape) == 3 and flow.shape[2] == 2):
            raise ValueError('flow must have H x W x 2 shape.')

        file.write(b'PF\n')
        file.write(b'%d %d\n' % (flow.shape[1], flow.shape[0]))

        endian = flow.dtype.byteorder
        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write(b'%f\n' % scale)

        invalid = np.isnan(flow[..., 0]) | np.isnan(flow[..., 1])
        flow = np.dstack([flow, invalid.astype(np.float32)])
        flow = np.flipud(flow)
        flow.tofile(file)
