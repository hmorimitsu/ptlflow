# =============================================================================
# Copyright 2024 Henrique Morimitsu
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

from train import _gen_dataset_id


def test_gen_dataset_id_single() -> None:
    assert _gen_dataset_id("chairs-train") == "chairs"


def test_gen_dataset_id_multiple() -> None:
    assert _gen_dataset_id("chairs-train+sintel-clean-trainval") == "chairs_sintel"


def test_gen_dataset_id_multiplier_first() -> None:
    assert _gen_dataset_id("3*sintel-clean-trainval") == "sintel"


def test_gen_dataset_id_multiplier_last() -> None:
    assert _gen_dataset_id("kitti-2012-train*5") == "kitti"


def test_gen_dataset_id_mixed() -> None:
    dataset_id = _gen_dataset_id(
        "chairs-train+3*sintel-clean-trainval+kitti-2012-train*5"
    )
    assert dataset_id == "chairs_sintel_kitti"
