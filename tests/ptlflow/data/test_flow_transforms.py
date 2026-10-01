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

import random
from typing import Dict

import numpy as np
import torch

from ptlflow.data import flow_transforms as ft


def _generate_tensors(
    h: int = 32, w: int = 48, seed: int = 0
) -> Dict[str, torch.Tensor]:
    """Create a dict of tensors in the format produced by ToTensor."""
    torch.manual_seed(seed)
    inputs = {
        "images": torch.rand(2, 3, h, w),
        "flows": torch.randn(1, 2, h, w),
        "valids": torch.ones(1, 1, h, w),
        "occs": (torch.rand(1, 1, h, w) > 0.5).float(),
        "mbs": (torch.rand(1, 1, h, w) > 0.5).float(),
        "flows_b": torch.randn(1, 2, h, w),
        "valids_b": torch.ones(1, 1, h, w),
        "occs_b": (torch.rand(1, 1, h, w) > 0.5).float(),
        "mbs_b": (torch.rand(1, 1, h, w) > 0.5).float(),
    }
    return inputs


def _generate_raw(h: int = 24, w: int = 32) -> Dict[str, list]:
    """Create a dict of numpy arrays in the format produced by the datasets."""
    return {
        "images": [np.random.randint(0, 256, (h, w, 3), np.uint8) for _ in range(2)],
        "flows": [np.random.rand(h, w, 2).astype(np.float32)],
        "valids": [np.full((h, w, 1), 255, np.uint8)],
    }


def _clone(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in inputs.items()}


def test_compose() -> None:
    transform = ft.Compose([ft.Resize((16, 24)), None])
    assert len(transform.transforms_list) == 1

    inputs = _generate_tensors()
    outputs = transform(inputs)
    assert outputs["images"].shape == (2, 3, 16, 24)


def test_to_tensor() -> None:
    inputs = _generate_raw()

    outputs = ft.ToTensor()(inputs)
    assert outputs["images"].shape == (2, 3, 24, 32)
    assert outputs["images"].dtype == torch.float32
    assert outputs["images"].min() >= 0.0
    assert outputs["images"].max() <= 1.0
    assert outputs["flows"].shape == (1, 2, 24, 32)
    assert outputs["valids"].shape == (1, 1, 24, 32)
    assert outputs["valids"].max() == 1.0


def test_to_tensor_fp16() -> None:
    outputs = ft.ToTensor(fp16=True)({"images": _generate_raw()["images"]})
    assert outputs["images"].dtype == torch.float16


def test_to_tensor_ignore_keys() -> None:
    outputs = ft.ToTensor(ignore_keys=["flows"])(_generate_raw())
    assert isinstance(outputs["flows"], list)
    assert outputs["images"].shape == (2, 3, 24, 32)


def test_center_crop() -> None:
    inputs = _generate_tensors()
    crop_size = (16, 24)

    outputs = ft.CenterCrop(crop_size)(_clone(inputs))
    for v in outputs.values():
        assert v.shape[2:] == crop_size

    y0 = (inputs["images"].shape[2] - crop_size[0]) // 2
    x0 = (inputs["images"].shape[3] - crop_size[1]) // 2
    assert torch.equal(
        outputs["images"],
        inputs["images"][:, :, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]],
    )


def test_center_crop_ignore_keys() -> None:
    inputs = _generate_tensors()

    outputs = ft.CenterCrop((16, 24), ignore_keys=["mbs"])(_clone(inputs))
    assert outputs["images"].shape[2:] == (16, 24)
    assert outputs["mbs"].shape[2:] == (32, 48)


def test_center_crop_updates_oob_occlusion() -> None:
    h, w = 32, 48
    inputs = {
        "images": torch.rand(2, 3, h, w),
        "flows": torch.zeros(1, 2, h, w),
        "valids": torch.ones(1, 1, h, w),
        "occs": torch.zeros(1, 1, h, w),
    }
    inputs["flows"][0, 0, :8] = 1000.0  # flow pointing outside of the image

    outputs = ft.CenterCrop((h, w))(inputs)
    assert outputs["occs"][0, 0, :8].min() == 1.0
    assert outputs["occs"][0, 0, 8:].max() == 0.0


def test_color_jitter() -> None:
    inputs = _generate_tensors()

    random.seed(1)
    outputs = ft.ColorJitter(0.4, 0.4, 0.4, 0.1, asymmetric_prob=0.0)(_clone(inputs))
    assert outputs["images"].shape == inputs["images"].shape
    assert outputs["images"].min() >= 0.0
    assert outputs["images"].max() <= 1.0
    assert not torch.equal(outputs["images"], inputs["images"])
    # only the images are modified
    assert torch.equal(outputs["flows"], inputs["flows"])

    random.seed(2)
    outputs = ft.ColorJitter(0.4, 0.4, 0.4, 0.1, asymmetric_prob=1.0)(_clone(inputs))
    assert outputs["images"].shape == inputs["images"].shape
    assert outputs["images"].min() >= 0.0
    assert outputs["images"].max() <= 1.0


def test_gaussian_noise() -> None:
    random.seed(3)
    inputs = _generate_tensors()

    outputs = ft.GaussianNoise(0.5)(_clone(inputs))
    assert outputs["images"].shape == inputs["images"].shape
    assert not torch.equal(outputs["images"], inputs["images"])
    assert outputs["images"].min() >= 0.0
    assert outputs["images"].max() <= 1.0
    # only the images are modified
    assert torch.equal(outputs["flows"], inputs["flows"])


def test_random_patch_eraser() -> None:
    random.seed(4)
    inputs = _generate_tensors()

    outputs = ft.RandomPatchEraser(1.0, (1, 2), (8, 16), "mean")(_clone(inputs))
    # only the second image is modified
    assert torch.equal(outputs["images"][0], inputs["images"][0])
    assert not torch.equal(outputs["images"][1], inputs["images"][1])

    outputs = ft.RandomPatchEraser(1.0, 1, (8, 16), "noise")(_clone(inputs))
    assert outputs["images"].shape == inputs["images"].shape


def test_random_flip_hflip() -> None:
    inputs = _generate_tensors()

    outputs = ft.RandomFlip(hflip_prob=1.0, vflip_prob=0.0)(_clone(inputs))
    assert torch.equal(outputs["images"], torch.flip(inputs["images"], [3]))

    expected_flows = torch.flip(inputs["flows"], [3]).clone()
    expected_flows[:, 0] *= -1
    assert torch.allclose(outputs["flows"], expected_flows)

    expected_flows_b = torch.flip(inputs["flows_b"], [3]).clone()
    expected_flows_b[:, 0] *= -1
    assert torch.allclose(outputs["flows_b"], expected_flows_b)

    assert torch.equal(outputs["valids"], torch.flip(inputs["valids"], [3]))


def test_random_flip_vflip() -> None:
    inputs = _generate_tensors()

    outputs = ft.RandomFlip(hflip_prob=0.0, vflip_prob=1.0)(_clone(inputs))
    assert torch.equal(outputs["images"], torch.flip(inputs["images"], [2]))

    expected_flows = torch.flip(inputs["flows"], [2]).clone()
    expected_flows[:, 1] *= -1
    assert torch.allclose(outputs["flows"], expected_flows)

    expected_flows_b = torch.flip(inputs["flows_b"], [2]).clone()
    expected_flows_b[:, 1] *= -1
    assert torch.allclose(outputs["flows_b"], expected_flows_b)

    assert torch.equal(outputs["valids"], torch.flip(inputs["valids"], [2]))


def test_random_flip_identity() -> None:
    inputs = _generate_tensors()

    outputs = ft.RandomFlip(hflip_prob=0.0, vflip_prob=0.0)(_clone(inputs))
    for k in inputs:
        assert torch.equal(outputs[k], inputs[k])


def test_random_scale_and_crop() -> None:
    random.seed(5)
    inputs = _generate_tensors()

    outputs = ft.RandomScaleAndCrop((16, 24), (0.0, 0.0), (0.0, 0.0))(_clone(inputs))
    for v in outputs.values():
        assert v.shape[2:] == (16, 24)

    # With scale 1, a constant flow must keep its value after cropping
    inputs = _generate_tensors()
    inputs["flows"] = torch.full_like(inputs["flows"], 3.0)
    outputs = ft.RandomScaleAndCrop((16, 24), (0.0, 0.0), (0.0, 0.0))(_clone(inputs))
    assert torch.allclose(outputs["flows"], torch.full_like(outputs["flows"], 3.0))


def test_random_scale_and_crop_sparse() -> None:
    random.seed(6)
    inputs = {
        "images": torch.rand(2, 3, 32, 48),
        "flows": torch.zeros(1, 2, 32, 48),
        "valids": torch.zeros(1, 1, 32, 48),
    }
    inputs["valids"][0, 0, 8:24, 12:36] = 1
    inputs["flows"][0, 0, 8:24, 12:36] = 2.0

    outputs = ft.RandomScaleAndCrop((32, 48), (0.0, 0.0), (0.0, 0.0), sparse=True)(
        inputs
    )
    assert outputs["images"].shape == (2, 3, 32, 48)
    assert set(outputs["valids"].unique().tolist()) <= {0.0, 1.0}
    # flows are only nonzero where the mask is valid
    assert torch.equal(outputs["flows"] * outputs["valids"], outputs["flows"])
    assert outputs["flows"].max() == 2.0


def test_resize() -> None:
    inputs = {
        "images": torch.rand(2, 3, 16, 24),
        "flows": torch.full((1, 2, 16, 24), 2.0),
        "valids": torch.ones(1, 1, 16, 24),
        "occs": (torch.rand(1, 1, 16, 24) > 0.5).float(),
    }

    outputs = ft.Resize((32, 48))(_clone(inputs))
    assert outputs["images"].shape == (2, 3, 32, 48)
    # flows must be multiplied by the resize factor
    assert torch.allclose(outputs["flows"], torch.full_like(outputs["flows"], 4.0))
    # binary inputs remain binary
    assert set(outputs["occs"].unique().tolist()) <= {0.0, 1.0}

    outputs = ft.Resize(scale=0.5)(_clone(inputs))
    assert outputs["images"].shape == (2, 3, 8, 12)
    assert torch.allclose(outputs["flows"], torch.full_like(outputs["flows"], 1.0))


def test_resize_sparse() -> None:
    inputs = {
        "images": torch.rand(2, 3, 16, 24),
        "flows": torch.zeros(1, 2, 16, 24),
        "valids": torch.zeros(1, 1, 16, 24),
    }
    inputs["valids"][0, 0, 8, 8] = 1
    inputs["flows"][0, 0, 8, 8] = 2.0

    outputs = ft.Resize((32, 48), sparse=True)(inputs)
    assert set(outputs["valids"].unique().tolist()) <= {0.0, 1.0}
    # flows are only nonzero where the mask is valid
    assert torch.equal(outputs["flows"] * outputs["valids"], outputs["flows"])
    assert outputs["flows"].max() == 4.0


def test_random_translate_identity() -> None:
    inputs = _generate_tensors()

    outputs = ft.RandomTranslate(0)(_clone(inputs))
    for k in inputs:
        assert torch.equal(outputs[k], inputs[k])


def test_random_translate_args() -> None:
    assert ft.RandomTranslate(4).translation == (4, 4)
    assert ft.RandomTranslate([3, 5]).translation == (3, 5)
    assert ft.RandomTranslate((3, 5)).translation == (3, 5)


def test_random_translate_shifts() -> None:
    inputs = _generate_tensors()
    h, w = inputs["images"].shape[2:]

    # Replicate the translation sampled by the transform
    random.seed(1)
    tw = random.randint(-5, 5)
    th = random.randint(-3, 3)
    assert th != 0 and tw != 0  # guaranteed for this seed

    random.seed(1)
    outputs = ft.RandomTranslate((3, 5))(_clone(inputs))

    out_h, out_w = outputs["images"].shape[2:]
    assert out_h == h - abs(th)
    assert out_w == w - abs(tw)

    # even inputs are cropped with a shift of +t, odd inputs with -t
    x1, x2 = max(0, tw), min(w + tw, w)
    y1, y2 = max(0, th), min(h + th, h)
    assert torch.equal(outputs["images"][0], inputs["images"][0, :, y1:y2, x1:x2])
    # the flow values are compensated by the translation
    expected_flows = inputs["flows"][0, :, y1:y2, x1:x2].clone()
    expected_flows[0] += tw
    expected_flows[1] += th
    assert torch.allclose(outputs["flows"][0], expected_flows)


def test_random_rotate_zero_angle() -> None:
    inputs = _generate_tensors()
    # the flows must be exactly zero, otherwise the out-of-bounds occlusion
    # update would legitimately modify the occlusion masks at the borders
    inputs["flows"] = torch.zeros_like(inputs["flows"])
    inputs["flows_b"] = torch.zeros_like(inputs["flows_b"])

    outputs = ft.RandomRotate(angle=0.0, diff_angle=0.0)(_clone(inputs))
    for k in inputs:
        assert torch.allclose(outputs[k], inputs[k], atol=1e-4)


def test_random_rotate_flow_vectors_rotated_once() -> None:
    h, w = 64, 96
    angle = 45.0
    inputs = {
        "images": torch.rand(2, 3, h, w),
        "flows": torch.zeros(1, 2, h, w),
        "valids": torch.ones(1, 1, h, w),
    }
    inputs["flows"][0, 0] = 4.0  # constant flow (4, 0)

    for seed in [10, 11, 12]:
        # Replicate the angle sampled by the transform
        random.seed(seed)
        sampled_angle = random.uniform(-angle, angle)

        random.seed(seed)
        outputs = ft.RandomRotate(angle=angle, diff_angle=0.0)(
            {k: v.clone() for k, v in inputs.items()}
        )

        cy, cx = h // 2, w // 2
        angle_rad = np.deg2rad(sampled_angle)
        expected = np.array([4.0 * np.cos(angle_rad), -4.0 * np.sin(angle_rad)])
        assert np.allclose(
            outputs["flows"][0, :, cy, cx].numpy(), expected, atol=1e-4
        ), f"seed={seed}, sampled_angle={sampled_angle}"


def test_random_rotate_binary_keys_remain_binary() -> None:
    random.seed(13)
    inputs = _generate_tensors()

    outputs = ft.RandomRotate(angle=10.0, diff_angle=5.0)(_clone(inputs))
    assert outputs["images"].shape == inputs["images"].shape
    for k in ["valids", "occs", "mbs", "valids_b", "occs_b", "mbs_b"]:
        assert set(outputs[k].unique().tolist()) <= {0.0, 1.0}


def test_generate_fbcheck_flow_occlusion() -> None:
    h, w = 32, 48
    inputs = {
        "images": torch.rand(2, 3, h, w),
        "flows": torch.zeros(1, 2, h, w),
        "flows_b": torch.zeros(1, 2, h, w),
    }

    outputs = ft.GenerateFBCheckFlowOcclusion(threshold=1.0)(inputs)
    # zero forward and backward flows are consistent, so nothing is occluded
    assert outputs["occs"].max() == 0.0
    assert outputs["occs_b"].max() == 0.0
    assert outputs["occs"].shape == (1, 1, h, w)

    # inconsistent backward flow marks all pixels as occluded
    inputs["flows_b"][0, 0] = 5.0
    outputs = ft.GenerateFBCheckFlowOcclusion(threshold=1.0)(inputs)
    assert outputs["occs"].min() == 1.0

    # out-of-bounds flows are marked as occluded
    inputs["flows_b"][0, 0] = 0.0
    inputs["flows"][0, 0] = 1000.0
    outputs = ft.GenerateFBCheckFlowOcclusion(threshold=1.0)(inputs)
    assert outputs["occs"].max() == 1.0

    # forward-only mode (use a fresh dict, since the transform adds keys in place)
    outputs = ft.GenerateFBCheckFlowOcclusion(
        threshold=1.0, compute_backward_occlusion=False
    )(
        {
            "images": torch.rand(2, 3, h, w),
            "flows": inputs["flows"].clone(),
            "flows_b": inputs["flows_b"].clone(),
        }
    )
    assert "occs_b" not in outputs


def test_full_transform_chain() -> None:
    random.seed(20)
    raw_inputs = _generate_raw()

    transform = ft.Compose(
        [
            ft.ToTensor(),
            ft.RandomScaleAndCrop((16, 24), (-0.1, 0.1), (-0.1, 0.1)),
            ft.ColorJitter(0.4, 0.4, 0.4, 0.1, 0.2),
            ft.GaussianNoise(0.02),
            ft.RandomPatchEraser(0.5, (1, 2), (4, 8), "mean"),
            ft.RandomFlip(0.5, 0.5),
        ]
    )
    outputs = transform(raw_inputs)

    assert outputs["images"].shape == (2, 3, 16, 24)
    assert outputs["flows"].shape == (1, 2, 16, 24)
    assert outputs["valids"].shape == (1, 1, 16, 24)
    assert outputs["images"].min() >= 0.0
    assert outputs["images"].max() <= 1.0
