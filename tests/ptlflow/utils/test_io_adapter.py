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
import torch

from ptlflow.utils.io_adapter import IOAdapter


def _dummy_images(h: int = 30, w: int = 40) -> list:
    return [np.random.randint(0, 256, (h, w, 3), np.uint8) for _ in range(2)]


def test_prepare_and_unscale() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(30, 40), target_size=(32, 48))

    inputs = adapter.prepare_inputs(images=_dummy_images())
    assert inputs["images"].shape == (1, 2, 3, 32, 48)
    assert inputs["images"].dtype == torch.float32
    assert inputs["images"].min() >= 0.0
    assert inputs["images"].max() <= 1.0

    outputs = adapter.unscale({"flows": torch.rand(1, 2, 32, 48)})
    assert outputs["flows"].shape == (1, 2, 30, 40)


def test_scale_factor() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(10, 20), target_scale_factor=0.5)

    inputs = adapter.prepare_inputs(images=_dummy_images(10, 20))
    assert inputs["images"].shape == (1, 2, 3, 5, 10)


def test_no_scaling() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(30, 40))

    inputs = adapter.prepare_inputs(images=_dummy_images())
    assert inputs["images"].shape == (1, 2, 3, 30, 40)


def test_flows_and_kwargs() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(30, 40), target_size=(32, 48))

    flow = np.random.rand(30, 40, 2).astype(np.float32)
    inputs = adapter.prepare_inputs(
        images=_dummy_images(),
        flows=flow,
        occs=[np.random.randint(0, 2, (30, 40, 1), np.uint8) * 255],
    )
    assert inputs["flows"].shape == (1, 1, 2, 32, 48)
    assert inputs["occs"].shape == (1, 1, 1, 32, 48)

    # flow values are multiplied by the scaling factors
    assert inputs["flows"].max() > flow.max()


def test_prepared_inputs_dict() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(30, 40), target_size=(32, 48))

    # when a dict is given, the tensors are used as they are (no numpy conversion)
    inputs = adapter.prepare_inputs(
        inputs={"images": torch.rand(2, 3, 30, 40), "meta": "keepme"}
    )
    assert inputs["images"].shape == (1, 2, 3, 32, 48)
    # non-tensor values are kept as they are
    assert inputs["meta"] == "keepme"


def test_image_only() -> None:
    adapter = IOAdapter(output_stride=8, input_size=(30, 40), target_size=(32, 48))

    inputs = adapter.prepare_inputs(
        images=_dummy_images(), flows=np.random.rand(30, 40, 2).astype(np.float32)
    )
    # scale everything back, but only for the images
    outputs = adapter.unscale(inputs, image_only=True)
    assert outputs["images"].shape == (1, 2, 3, 30, 40)
    assert outputs["flows"].shape == (1, 1, 2, 32, 48)
