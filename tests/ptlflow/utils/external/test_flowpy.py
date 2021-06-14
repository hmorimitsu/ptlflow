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
import shutil

import cv2
import numpy as np
import pytest

from ptlflow.utils.external import flowpy

IMG_SIDE = 29
IMG_MIDDLE = IMG_SIDE // 2 + 1


def test_read_write_flo(tmp_path: Path) -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    file_path = tmp_path / 'flow.flo'
    flowpy.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flowpy.flow_read(file_path)
    assert np.array_equal(flow, loaded_flow)

    shutil.rmtree(tmp_path)


def test_read_write_png(tmp_path: Path) -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    file_path = tmp_path / 'flow.png'
    flowpy.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flowpy.flow_read(file_path)
    assert np.array_equal(flow, loaded_flow)

    shutil.rmtree(tmp_path)

# Skip these tests, as in some machines the colors are sightly different


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_bright() -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    rgb = flowpy.flow_to_rgb(flow, background='bright')
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/external/flowpy_rgb_bg_bright_{IMG_SIDE}x{IMG_SIDE}.png')))
    assert np.array_equal(rgb, rgb_gt)


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_dark() -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    rgb = flowpy.flow_to_rgb(flow, background='dark')
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/external/flowpy_rgb_bg_dark_{IMG_SIDE}x{IMG_SIDE}.png')))
    assert np.array_equal(rgb, rgb_gt)


@pytest.mark.skip(reason='In some machines the colors are slightly off.')
def test_convert_rgb_bg_dark_max5() -> None:
    flow = np.stack(np.meshgrid(np.arange(IMG_SIDE)-IMG_MIDDLE, np.arange(IMG_SIDE)-IMG_MIDDLE), axis=2).astype(np.float32)
    rgb = flowpy.flow_to_rgb(flow, background='dark', flow_max_radius=5)
    rgb_gt = cv2.imread(str(
        Path(f'tests/data/ptlflow/utils/external/flowpy_rgb_bg_dark_max5_{IMG_SIDE}x{IMG_SIDE}.png')))
    assert np.array_equal(rgb, rgb_gt)
