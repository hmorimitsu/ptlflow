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

import pytest

from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils import dummy_datasets

DATASET_CONFIG_PATH = Path(__file__).resolve().parents[3] / "datasets.yaml"


def test_parse_dataset_selection() -> None:
    datamodule = FlowDataModule()

    parsed = datamodule._parse_dataset_selection(
        "chairs-train+3*sintel-clean-trainval+kitti-2012-train*5"
    )
    assert parsed == [
        (1, "chairs", "train"),
        (3, "sintel", "clean", "trainval"),
        (5, "kitti", "2012", "train"),
    ]


def test_parse_dataset_selection_spaces() -> None:
    datamodule = FlowDataModule()

    parsed = datamodule._parse_dataset_selection(" chairs-train + 2*chairs-val ")
    assert parsed == [(1, "chairs", "train"), (2, "chairs", "val")]


def test_parse_dataset_selection_none() -> None:
    assert FlowDataModule()._parse_dataset_selection(None) == []


def test_parse_dataset_selection_invalid() -> None:
    with pytest.raises(ValueError):
        FlowDataModule()._parse_dataset_selection("chairs*train*val")


def test_train_and_val_dataloaders(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs(tmp_path)

    datamodule = FlowDataModule(
        train_dataset="chairs-train",
        val_dataset="chairs-trainval",
        train_batch_size=1,
        train_num_workers=0,
        flying_chairs_root_dir=str(tmp_path / "FlyingChairs_release"),
        dataset_config_path=str(DATASET_CONFIG_PATH),
    )
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    assert batch["images"].shape[0] == 1
    assert batch["images"].shape[1] == 2  # two frames per sample
    assert batch["flows"].shape[1] == 1
    assert batch["images"].shape[-2:] == batch["flows"].shape[-2:]

    val_loaders = datamodule.val_dataloader()
    assert len(val_loaders) == 1
    assert datamodule.val_dataloader_names == ["chairs-trainval"]
    val_batch = next(iter(val_loaders[0]))
    assert val_batch["images"].shape[1] == 2

    shutil.rmtree(tmp_path)


def test_train_dataloader_multiplier(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs(tmp_path)

    plain_datamodule = FlowDataModule(
        train_dataset="chairs-trainval",
        val_dataset="chairs-trainval",
        train_batch_size=1,
        train_num_workers=0,
        flying_chairs_root_dir=str(tmp_path / "FlyingChairs_release"),
        dataset_config_path=str(DATASET_CONFIG_PATH),
    )
    plain_datamodule.setup("fit")
    plain_datamodule.train_dataloader()

    mult_datamodule = FlowDataModule(
        train_dataset="2*chairs-trainval",
        val_dataset="chairs-trainval",
        train_batch_size=1,
        train_num_workers=0,
        flying_chairs_root_dir=str(tmp_path / "FlyingChairs_release"),
        dataset_config_path=str(DATASET_CONFIG_PATH),
    )
    mult_datamodule.setup("fit")
    mult_datamodule.train_dataloader()

    assert (
        mult_datamodule.train_dataloader_length
        == 2 * plain_datamodule.train_dataloader_length
    )

    shutil.rmtree(tmp_path)


def test_test_dataloader(tmp_path: Path) -> None:
    dummy_datasets.write_sintel(tmp_path)

    datamodule = FlowDataModule(
        test_dataset="sintel",
        mpi_sintel_root_dir=str(tmp_path / "MPI-Sintel"),
        dataset_config_path=str(DATASET_CONFIG_PATH),
    )
    datamodule.setup("test")

    dataloaders = datamodule.test_dataloader()
    assert datamodule.test_dataloader_names == [
        "sintel-clean-test",
        "sintel-final-test",
    ]
    assert len(dataloaders) == 2

    batch = next(iter(dataloaders[0]))
    assert batch["images"].shape[0] == 1
    assert batch["images"].shape[1] == 2
    # the test split has no groundtruth
    assert "flows" not in batch

    shutil.rmtree(tmp_path)


def test_get_overfit_dataset(tmp_path: Path) -> None:
    dummy_datasets.write_sintel(tmp_path)

    datamodule = FlowDataModule(
        mpi_sintel_root_dir=str(tmp_path / "MPI-Sintel"),
        dataset_config_path=str(DATASET_CONFIG_PATH),
    )

    dataset = datamodule._get_overfit_dataset(True)
    assert len(dataset) == 1

    inputs = dataset[0]
    assert inputs["images"].shape == (2, 3, 436, 1024)
    assert inputs["flows"].shape == (1, 2, 436, 1024)

    shutil.rmtree(tmp_path)


def test_missing_train_dataset_raises(tmp_path: Path) -> None:
    datamodule = FlowDataModule(dataset_config_path=str(DATASET_CONFIG_PATH))
    with pytest.raises(AssertionError):
        datamodule.setup("fit")
