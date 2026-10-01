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

import numpy as np
import pytest
import torch
import torch.nn as nn

from ptlflow.utils.utils import (
    InputPadder,
    InputScaler,
    are_shapes_compatible,
    bgr_val_as_tensor,
    count_parameters,
    forward_interpolate_batch,
    make_divisible,
    release_gpu,
    tensor_dict_to_numpy,
)


def test_input_scaler_stride() -> None:
    scaler = InputScaler((30, 40), stride=8)
    assert (scaler.tgt_height, scaler.tgt_width) == (32, 40)

    ys = torch.linspace(0, 1, 30)[:, None]
    xs = torch.linspace(0, 1, 40)[None, :]
    images = (ys + 2 * xs).expand(2, 3, 30, 40).contiguous()

    scaled = scaler.fill(images)
    assert scaled.shape == (2, 3, 32, 40)

    restored = scaler.unfill(scaled)
    assert restored.shape == images.shape
    assert (restored - images).abs().max() < 2e-3


def test_input_scaler_flow_values() -> None:
    scaler = InputScaler((10, 20), size=(20, 40))

    flow = torch.full((1, 2, 10, 20), 2.0)
    scaled = scaler.fill(flow, is_flow=True)
    assert torch.allclose(scaled, torch.full_like(scaled, 4.0))

    restored = scaler.unfill(scaled, is_flow=True)
    assert torch.allclose(restored, torch.full_like(restored, 2.0))


def test_input_scaler_scale_factor() -> None:
    scaler = InputScaler((10, 20), scale_factor=0.5)
    images = torch.rand(1, 3, 10, 20)
    assert scaler.fill(images).shape == (1, 3, 5, 10)


def test_input_scaler_nearest_mode() -> None:
    scaler = InputScaler((10, 20), size=(20, 40), interpolation_mode="nearest")
    images = torch.full((1, 3, 10, 20), 0.5)
    scaled = scaler.fill(images)
    assert scaled.shape == (1, 3, 20, 40)
    assert torch.allclose(scaled, torch.full_like(scaled, 0.5))


def test_input_scaler_non_contiguous_input() -> None:
    scaler = InputScaler((10, 20), size=(20, 40))
    images = torch.rand(2, 3, 10, 20).transpose(0, 1)
    assert not images.is_contiguous()
    scaled = scaler.fill(images)
    assert scaled.shape == (3, 2, 20, 40)


def test_input_padder_roundtrip() -> None:
    padder = InputPadder([30, 40], stride=8)
    assert padder.tgt_size == (32, 40)

    images = torch.rand(2, 3, 30, 40)
    padded = padder.fill(images)
    assert padded.shape == (2, 3, 32, 40)

    restored = padder.unfill(padded)
    assert restored.shape == images.shape
    assert torch.equal(restored, images)


def test_input_padder_unfill_noop() -> None:
    padder = InputPadder([30, 40], stride=8)
    images = torch.rand(2, 3, 30, 40)
    # tensors already in the original size are not modified
    assert torch.equal(padder.unfill(images), images)


def test_make_divisible() -> None:
    assert make_divisible(100, 8) == 96
    assert make_divisible(16, 8) == 16
    assert make_divisible(5, 8) == 8


def test_are_shapes_compatible() -> None:
    assert are_shapes_compatible((1, 3, 8, 8), (2, 3, 8, 8))
    assert not are_shapes_compatible((1, 3, 8, 8), (2, 3, 8, 9))
    assert not are_shapes_compatible((3, 8, 8), (1, 3, 8, 8))


def test_bgr_val_as_tensor() -> None:
    ref = torch.rand(2, 3, 8, 8)

    out = bgr_val_as_tensor(0.5, ref)
    assert out.shape == (1, 3, 1, 1)
    assert torch.allclose(out, torch.full_like(out, 0.5))

    out = bgr_val_as_tensor((0.1, 0.2, 0.3), ref)
    assert out.shape == (1, 3, 1, 1)
    assert torch.allclose(out, torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1))

    out = bgr_val_as_tensor(np.array([0.1, 0.2, 0.3]), ref)
    assert out.shape == (1, 3, 1, 1)

    out = bgr_val_as_tensor(torch.ones(2, 3, 1, 1), ref)
    assert out.shape == (2, 3, 1, 1)


def test_tensor_dict_to_numpy() -> None:
    inputs = {
        "images": torch.rand(2, 3, 8, 8),
        "flows": torch.rand(2, 2, 8, 8),
        "meta": "keepme",
    }
    outputs = tensor_dict_to_numpy(inputs)
    assert outputs["images"].shape == (8, 8, 3)
    assert outputs["flows"].shape == (8, 8, 2)
    assert outputs["meta"] == "keepme"


def test_tensor_dict_to_numpy_with_padder() -> None:
    padder = InputPadder([8, 10], stride=8)
    inputs = {"images": torch.rand(2, 3, 8, 10)}
    outputs = tensor_dict_to_numpy({"images": padder.fill(inputs["images"])}, padder)
    assert outputs["images"].shape == (8, 10, 3)


def test_forward_interpolate_batch() -> None:
    zeros = torch.zeros(2, 2, 16, 24)
    out = forward_interpolate_batch(zeros)
    assert out.shape == zeros.shape
    assert out.abs().max() == 0.0

    flow = torch.zeros(1, 2, 16, 24)
    flow[:, 0] = 1.0
    flow[:, 1] = 0.5
    out = forward_interpolate_batch(flow)
    # constant flows are preserved in the interior
    assert torch.allclose(out[:, :, 2:-2, 2:-2], flow[:, :, 2:-2, 2:-2])


def test_count_parameters() -> None:
    model = nn.Linear(4, 2)
    n_params = 4 * 2 + 2
    assert count_parameters(model) == n_params

    for p in model.parameters():
        p.requires_grad = False
    assert count_parameters(model) == 0


def test_release_gpu() -> None:
    tensors_dict = {"images": torch.rand(2, 3, 8, 8), "meta": "keepme"}
    outputs = release_gpu(tensors_dict)
    assert outputs["images"].device.type == "cpu"
    assert outputs["meta"] == "keepme"


def test_release_gpu_detaches() -> None:
    tensor = torch.rand(2, requires_grad=True)
    outputs = release_gpu({"t": tensor})
    assert not outputs["t"].requires_grad


def test_input_scaler_size_and_stride_conflict() -> None:
    with pytest.raises(AssertionError):
        InputScaler((10, 20), stride=8, size=(20, 40))
