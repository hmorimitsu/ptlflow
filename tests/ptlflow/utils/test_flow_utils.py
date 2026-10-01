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
import json
import shutil

import cv2 as cv
import numpy as np
import torch

from ptlflow.utils import flow_utils

IMG_SIDE = 29
IMG_MIDDLE = IMG_SIDE // 2 + 1


def _dummy_flow(h: int = 13, w: int = 17) -> np.ndarray:
    return np.random.rand(h, w, 2).astype(np.float32)


def test_read_write_pfm(tmp_path: Path) -> None:
    flow = np.stack(
        np.meshgrid(np.arange(IMG_SIDE) - IMG_MIDDLE, np.arange(IMG_SIDE) - IMG_MIDDLE),
        axis=2,
    ).astype(np.float32)
    file_path = tmp_path / "flow.pfm"
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert np.array_equal(flow, loaded_flow)

    shutil.rmtree(tmp_path)


def test_read_write_flo(tmp_path: Path) -> None:
    flow = _dummy_flow()
    file_path = tmp_path / "flow.flo"
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert loaded_flow.shape == flow.shape
    assert np.abs(loaded_flow - flow).max() < 1e-4

    shutil.rmtree(tmp_path)


def test_read_write_png(tmp_path: Path) -> None:
    flow = _dummy_flow()
    file_path = tmp_path / "flow.png"
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert loaded_flow.shape == flow.shape
    # the png format is quantized, so a small error is expected
    assert np.abs(loaded_flow - flow).max() < 0.1

    shutil.rmtree(tmp_path)


def test_read_write_npy(tmp_path: Path) -> None:
    flow = _dummy_flow()
    file_path = tmp_path / "flow.npy"
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert np.array_equal(flow, loaded_flow)

    shutil.rmtree(tmp_path)


def test_read_write_flo5(tmp_path: Path) -> None:
    flow = _dummy_flow()
    file_path = tmp_path / "flow.flo5"
    flow_utils.flow_write(file_path, flow)
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path)
    assert loaded_flow.shape == flow.shape
    assert np.abs(loaded_flow - flow).max() < 1e-4

    shutil.rmtree(tmp_path)


def test_read_write_viper_npz(tmp_path: Path) -> None:
    flow = _dummy_flow()
    file_path = tmp_path / "flow.npz"
    flow_utils.flow_write(file_path, flow, "viper_npz")
    assert file_path.exists()

    loaded_flow = flow_utils.flow_read(file_path, "viper_npz")
    assert loaded_flow.shape == flow.shape
    # the viper format stores the flow as float16
    assert np.abs(loaded_flow - flow).max() < 0.1

    shutil.rmtree(tmp_path)


def test_read_kubric_flow(tmp_path: Path) -> None:
    seq_dir = tmp_path / "sequence"
    seq_dir.mkdir(parents=True)
    with open(seq_dir / "data_ranges.json", "w") as f:
        json.dump(
            {
                "forward_flow": {"min": -100, "max": 100},
                "backward_flow": {"min": -100, "max": 100},
            },
            f,
        )

    flow = ((np.random.rand(13, 17, 2) - 0.5) * 200).astype(np.float32)
    encoded = np.zeros((13, 17, 3), np.uint16)
    encoded[..., 1:] = np.round((flow + 100) / 200 * 65535)
    file_path = seq_dir / "forward_flow_00000.png"
    cv.imwrite(str(file_path), encoded)

    loaded_flow = flow_utils.read_kubric_flow(file_path, "forward_flow")
    assert loaded_flow.shape == flow.shape
    assert np.abs(loaded_flow - flow).max() < 0.01

    shutil.rmtree(tmp_path)


def test_fb_check() -> None:
    h, w = 8, 8
    forward_flow = np.zeros((h, w, 2), np.float32)
    backward_flow = np.zeros((h, w, 2), np.float32)

    # consistent zero flows
    mask = flow_utils.fb_check(forward_flow, backward_flow)
    assert mask.shape == (h, w)
    # the borders are always marked as out-of-bounds
    assert mask[1:-1, 1:-1].all()
    assert not mask[0, :].any()
    assert not mask[-1, :].any()

    # inconsistent backward flow
    backward_flow[..., 0] = 5.0
    mask = flow_utils.fb_check(forward_flow, backward_flow)
    assert not mask[1:-1, 1:-1].any()


def test_fb_check_torch() -> None:
    forward_flow = torch.zeros(1, 2, 8, 8)
    backward_flow = torch.zeros(1, 2, 8, 8)
    mask = flow_utils.fb_check(forward_flow, backward_flow)
    assert isinstance(mask, torch.Tensor)
    assert mask[1:-1, 1:-1].all()


def test_flow_to_rgb() -> None:
    flow = _dummy_flow(20, 30)
    rgb = flow_utils.flow_to_rgb(flow)
    assert rgb.shape == (20, 30, 3)
    # the numpy version returns an uint8 image
    assert rgb.min() >= 0
    assert rgb.max() <= 255

    flow_torch = torch.from_numpy(flow.transpose(2, 0, 1))
    rgb_torch = flow_utils.flow_to_rgb(flow_torch)
    assert rgb_torch.shape == (3, 20, 30)
    assert rgb_torch.min() >= 0.0
    assert rgb_torch.max() <= 1.0
    # torch and numpy results should be close (allowing for the uint8
    # quantization of the numpy version and small machine-dependent variations)
    assert (
        np.abs(
            rgb_torch.numpy().transpose(1, 2, 0) - rgb.astype(np.float32) / 255
        ).max()
        < 0.01
    )
