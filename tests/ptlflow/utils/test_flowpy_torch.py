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

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from ptlflow.utils import flowpy_torch

IMG_SIDE = 29
IMG_MIDDLE = IMG_SIDE // 2 + 1


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_bright() -> None:
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/flowpy_torch_rgb_bg_bright_{IMG_SIDE}x{IMG_SIDE}.png')))
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=0).astype(np.float32)
    flow = torch.from_numpy(flow)

    for i in range(5):
        flow_tmp = flow.clone()
        for _ in range(i):
            flow_tmp = flow_tmp[None]
        rgb = flowpy_torch.flow_to_rgb(flow_tmp, background='bright')
        for _ in range(i):
            rgb = rgb[0]
        rgb = (255 * rgb.permute(1, 2, 0).numpy()).astype(np.uint8)
        assert np.array_equal(rgb, rgb_gt)


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_dark() -> None:
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/flowpy_torch_rgb_bg_dark_{IMG_SIDE}x{IMG_SIDE}.png')))
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=0).astype(np.float32)
    flow = torch.from_numpy(flow)

    for i in range(5):
        flow_tmp = flow.clone()
        for _ in range(i):
            flow_tmp = flow_tmp[None]
        rgb = flowpy_torch.flow_to_rgb(flow_tmp, background='dark')
        for _ in range(i):
            rgb = rgb[0]
        rgb = (255 * rgb.permute(1, 2, 0).numpy()).astype(np.uint8)
        assert np.array_equal(rgb, rgb_gt)


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_dark_max5() -> None:
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/flowpy_torch_rgb_bg_dark_max5_{IMG_SIDE}x{IMG_SIDE}.png')))
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=0).astype(np.float32)
    flow = torch.from_numpy(flow)

    for i in range(5):
        flow_tmp = flow.clone()
        for _ in range(i):
            flow_tmp = flow_tmp[None]
        rgb = flowpy_torch.flow_to_rgb(flow_tmp, background='dark', flow_max_radius=5)
        for _ in range(i):
            rgb = rgb[0]
        rgb = (255 * rgb.permute(1, 2, 0).numpy()).astype(np.uint8)
        assert np.array_equal(rgb, rgb_gt)
