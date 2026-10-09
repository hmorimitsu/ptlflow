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

import pytest
import torch

from ptlflow.utils.flow_metrics import FlowMetrics


def test_zero_epe() -> None:
    metric = FlowMetrics()
    flow = torch.rand(2, 2, 8, 8)
    metric.update({"flows": flow.clone()}, {"flows": flow.clone()})

    metrics = metric.calculate_metrics()
    assert metrics["epe"].item() == pytest.approx(0.0)
    assert metrics["px1"].item() == pytest.approx(1.0)
    assert metrics["px3"].item() == pytest.approx(1.0)
    assert metrics["px5"].item() == pytest.approx(1.0)
    assert metrics["flall"].item() == pytest.approx(0.0)
    assert metrics["wauc"].item() == pytest.approx(100.0, abs=1e-3)


def test_known_epe_values() -> None:
    metric = FlowMetrics()
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0] = 2.0  # constant epe of 2

    metric.update({"flows": pred}, {"flows": target})
    metrics = metric.calculate_metrics()
    assert metrics["epe"].item() == pytest.approx(2.0)
    assert metrics["px1"].item() == pytest.approx(0.0)
    assert metrics["px3"].item() == pytest.approx(1.0)
    assert metrics["px5"].item() == pytest.approx(1.0)
    assert metrics["flall"].item() == pytest.approx(0.0)


def test_flall() -> None:
    metric = FlowMetrics()
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0] = 10.0  # epe 10 > 3 and > 0.05 * 10

    metric.update({"flows": pred}, {"flows": target})
    metrics = metric.calculate_metrics()
    assert metrics["flall"].item() == pytest.approx(100.0)


def test_valid_mask() -> None:
    metric = FlowMetrics()
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0, :, :2] = 10.0  # large error only on the invalid half
    valids = torch.ones(1, 1, 4, 4)
    valids[..., :2] = 0

    metric.update({"flows": pred}, {"flows": target, "valids": valids})
    metrics = metric.calculate_metrics()
    assert metrics["epe"].item() == pytest.approx(0.0)
    assert metrics["flall"].item() == pytest.approx(0.0)


def test_occlusion_keys() -> None:
    metric = FlowMetrics()
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0] = 2.0
    occs = torch.zeros(1, 1, 4, 4)
    occs[..., 2:] = 1  # right half occluded

    metric.update({"flows": pred}, {"flows": target, "occs": occs})
    metrics = metric.calculate_metrics()
    assert "epe_occ" in metrics
    assert "epe_non_occ" in metrics
    assert metrics["epe"].item() == pytest.approx(2.0)


def test_occ_f1_perfect() -> None:
    metric = FlowMetrics()
    flow = torch.zeros(1, 2, 4, 4)
    occs = torch.zeros(1, 1, 4, 4)
    occs[..., 2:] = 1

    metric.update({"flows": flow, "occs": occs}, {"flows": flow, "occs": occs})
    metrics = metric.calculate_metrics()
    assert metrics["occ_f1"].item() == pytest.approx(1.0)


def test_occ_f1_macro() -> None:
    metric = FlowMetrics()
    flow = torch.zeros(1, 2, 4, 4)
    occ_pred = torch.zeros(1, 1, 4, 4)  # nothing predicted as occluded
    occ_target = torch.zeros(1, 1, 4, 4)
    occ_target[..., 2:] = 1  # half of the pixels are occluded

    metric.update(
        {"flows": flow, "occs": occ_pred},
        {"flows": flow, "occs": occ_target},
    )
    metrics = metric.calculate_metrics()
    # macro f1: f1_pos == 0, f1_neg == 2/3, so the mean is 1/3
    assert metrics["occ_f1"].item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_mb_and_conf_metrics() -> None:
    metric = FlowMetrics()
    flow = torch.zeros(1, 2, 4, 4)
    mbs = torch.zeros(1, 1, 4, 4)
    mbs[..., 2:] = 1

    metric.update(
        {"flows": flow, "mbs": mbs, "confs": mbs},
        {"flows": flow, "mbs": mbs},
    )
    metrics = metric.calculate_metrics()
    assert metrics["mb_f1"].item() == pytest.approx(1.0)
    # the confidence groundtruth is exp(-epe^2) == 1 everywhere, while the
    # prediction covers only half of the pixels, so macro f1 == 1/3
    assert metrics["conf_f1"].item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_epoch_mean_accumulates_two_steps() -> None:
    metric = FlowMetrics()
    flow = torch.zeros(1, 2, 4, 4)

    target1 = torch.zeros(1, 2, 4, 4)
    target1[0, 0] = 1.0
    metric.update({"flows": flow}, {"flows": target1})

    target2 = torch.zeros(1, 2, 4, 4)
    target2[0, 0] = 3.0
    metric.update({"flows": flow}, {"flows": target2})

    metrics = metric.calculate_metrics()
    assert metrics["epe"].item() == pytest.approx(2.0)
    assert metric.step_count == 2


def test_ema_average_mode() -> None:
    decay = 0.9
    metric = FlowMetrics(average_mode="ema", ema_decay=decay)
    flow = torch.zeros(1, 2, 4, 4)

    target1 = torch.zeros(1, 2, 4, 4)
    target1[0, 0] = 1.0
    metric.update({"flows": flow}, {"flows": target1})

    target2 = torch.zeros(1, 2, 4, 4)
    target2[0, 0] = 3.0
    metric.update({"flows": flow}, {"flows": target2})

    metrics = metric.calculate_metrics()
    # after two steps, the ema normalization gives (decay*m1 + m2) / (1 + decay)
    expected = (decay * 1.0 + 3.0) / (1.0 + decay)
    assert metrics["epe"].item() == pytest.approx(expected)


def test_prefix() -> None:
    metric = FlowMetrics(prefix="test_")
    flow = torch.zeros(1, 2, 4, 4)
    metric.update({"flows": flow}, {"flows": flow})
    metrics = metric.calculate_metrics()
    assert "test_epe" in metrics
