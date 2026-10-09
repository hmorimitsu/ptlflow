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

from typing import Dict

import pytest
import torch

from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.flow_metrics import FlowMetrics
from ptlflow.utils.utils import InputPadder, InputScaler


class _DummyModel(BaseModel):
    """Minimal concrete model that always predicts a constant flow."""

    def __init__(self) -> None:
        super().__init__(output_stride=8)
        self._pred_flow = None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"flows": self._pred_flow}


def _make_batch(dataset_name: str, epe_value: float) -> Dict[str, torch.Tensor]:
    h, w = 32, 40
    target = torch.zeros(1, 1, 2, h, w)
    target[:, :, 0] = epe_value
    return {
        "images": torch.rand(1, 2, 3, h, w),
        "flows": target,
        "valids": torch.ones(1, 1, 1, h, w),
        "meta": {
            "dataset_name": [dataset_name],
            "split_name": ["val"],
            "is_val": [True],
            "is_seq_start": [True],
            "misc": [""],
        },
    }


def _attach_log_recorder(model: _DummyModel) -> dict:
    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, value)
    return logged


def test_main_val_metric_uses_dataset_main_metric() -> None:
    # epe = 4 (> 3), so flall = 100 while epe = 4. For KITTI the main
    # metric is flall, so main_val_metric must be 100 (not 4).
    model = _DummyModel()
    model._pred_flow = torch.zeros(1, 1, 2, 32, 40)
    logged = _attach_log_recorder(model)

    model.validation_step(_make_batch("KITTI_2015", 4.0), 0, 0)
    model.on_validation_epoch_end()

    assert "main_val_metric" in logged
    assert logged["main_val_metric"].item() == pytest.approx(100.0)
    assert "kitti_2015-val" in logged


def test_main_val_metric_fallback_is_epe() -> None:
    # Sintel main metric is epe, so main_val_metric must be the epe value.
    model = _DummyModel()
    model._pred_flow = torch.zeros(1, 1, 2, 32, 40)
    logged = _attach_log_recorder(model)

    model.validation_step(_make_batch("Sintel_clean", 4.0), 0, 0)
    model.on_validation_epoch_end()

    assert logged["main_val_metric"].item() == pytest.approx(4.0)
    assert "sintel_clean-val" in logged


def test_validation_step_accumulates_metrics() -> None:
    model = _DummyModel()
    model._pred_flow = torch.zeros(1, 1, 2, 32, 40)
    _attach_log_recorder(model)

    outputs1 = model.validation_step(_make_batch("KITTI_2015", 2.0), 0, 0)
    outputs2 = model.validation_step(_make_batch("KITTI_2015", 4.0), 1, 0)

    # the metrics returned at each step correspond to that step only
    assert outputs1["metrics"]["val/epe"].item() == pytest.approx(2.0)
    assert outputs2["metrics"]["val/epe"].item() == pytest.approx(4.0)

    # after the epoch ends, the accumulated value is the mean
    model.on_validation_epoch_end()


def test_on_validation_epoch_end_with_empty_dataloader() -> None:
    model = _DummyModel()
    model._pred_flow = torch.zeros(1, 1, 2, 32, 40)
    _attach_log_recorder(model)

    model.validation_step(_make_batch("KITTI_2015", 2.0), 0, 0)
    # simulate a second dataloader that never saw any batch
    model.val_metrics.append(FlowMetrics(prefix="val/"))
    model.val_dataset_names.append(None)

    # must not raise, even though the name of the second loader is unknown
    model.on_validation_epoch_end()


def test_split_train_val_metrics() -> None:
    model = _DummyModel()

    out = model._split_train_val_metrics(
        {"val/epe": 1.0},
        {"dataset_name": ["Sintel_clean"], "is_val": [True]},
    )
    assert out == {
        "val_sintel_clean/full/epe": 1.0,
        "val_sintel_clean/val/epe": 1.0,
    }

    out = model._split_train_val_metrics(
        {"val/epe": 1.0},
        {"dataset_name": ["Sintel_clean"], "is_val": [False]},
    )
    assert out == {
        "val_sintel_clean/full/epe": 1.0,
        "val_sintel_clean/train/epe": 1.0,
    }

    out = model._split_train_val_metrics({"val/epe": 1.0}, None)
    assert out == {"val/full/epe": 1.0}


def test_preprocess_images_pad() -> None:
    model = _DummyModel()
    images = torch.rand(1, 2, 3, 30, 40)

    outputs, padder = model.preprocess_images(images)
    assert isinstance(padder, InputPadder)
    assert outputs.shape == (1, 2, 3, 32, 40)

    restored = model.postprocess_predictions(outputs, padder, is_flow=False)
    assert restored.shape == images.shape
    assert torch.equal(restored, images)


def test_preprocess_images_interpolation() -> None:
    model = _DummyModel()
    images = torch.rand(1, 2, 3, 30, 40)

    outputs, scaler = model.preprocess_images(images, resize_mode="interpolation")
    assert isinstance(scaler, InputScaler)
    assert outputs.shape == (1, 2, 3, 32, 40)

    flow = torch.full((1, 2, 32, 40), 2.0)
    restored = model.postprocess_predictions(flow, scaler, is_flow=True)
    assert restored.shape == (1, 2, 30, 40)
    # only the height changed (30 -> 32), so only the v channel is scaled
    expected = torch.empty_like(restored)
    expected[:, 0] = 2.0 * 40.0 / 40.0
    expected[:, 1] = 2.0 * 30.0 / 32.0
    assert torch.allclose(restored, expected)


def test_preprocess_images_bgr_ops() -> None:
    model = _DummyModel()
    images = torch.rand(1, 2, 3, 32, 40)  # already divisible by 8

    outputs, _ = model.preprocess_images(images, bgr_add=0.1, bgr_mult=2.0)
    expected = (images + 0.1) * 2.0
    assert torch.allclose(outputs, expected)

    channels = (
        torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1, 1).expand(1, 2, 3, 32, 40)
    )
    outputs, _ = model.preprocess_images(channels.contiguous(), bgr_to_rgb=True)
    assert torch.allclose(outputs, channels.flip([-3]))


def test_train_size_validation() -> None:
    model = _DummyModel()

    model.train_size = (360, 480)
    assert model.train_size == (360, 480)

    model.train_size = None
    assert model.train_size is None

    with pytest.raises(AssertionError):
        model.train_size = 360

    with pytest.raises(AssertionError):
        model.train_size = (360.0, 480.0)


def test_add_extra_param() -> None:
    model = _DummyModel()

    assert model.extra_params is None
    model.add_extra_param("foo", 42)
    model.add_extra_param("bar", "value")
    assert model.extra_params == {"foo": 42, "bar": "value"}
