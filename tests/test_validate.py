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

from jsonargparse import ArgumentParser

import ptlflow
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.dummy_datasets import write_kitti, write_sintel
import validate

TEST_MODEL = "raft_small"


def test_validate(tmp_path: Path) -> None:
    model = ptlflow.get_model(TEST_MODEL)

    data_parser = ArgumentParser()
    data_parser.add_class_arguments(FlowDataModule, "data")
    data_args = data_parser.parse_args([])
    data_args.data.val_dataset = "sintel-clean+sintel-final+kitti-2015"
    data_args.data.mpi_sintel_root_dir = str(tmp_path / "MPI-Sintel")
    data_args.data.kitti_2012_root_dir = str(tmp_path / "KITTI/2012")
    data_args.data.kitti_2015_root_dir = str(tmp_path / "KITTI/2015")

    data_parser = ArgumentParser(exit_on_error=False)
    data_parser.add_argument("--data", type=FlowDataModule)
    data_cfg = data_parser.parse_object({"data": data_args.data})
    datamodule = data_parser.instantiate_classes(data_cfg).data
    datamodule.setup("validate")

    parser = ArgumentParser(parents=[validate._init_parser()])
    args = parser.parse_args([])
    args.output_path = str(tmp_path)
    args.write_outputs = True
    args.max_samples = 1
    args.model_name = TEST_MODEL

    write_kitti(tmp_path)
    write_sintel(tmp_path)

    args.flow_format = "flo"
    metrics_df = validate.validate(args, model, datamodule)
    assert min(metrics_df.shape) > 0

    dataset_name_path = [
        ("kitti-2015", "000000_10"),
        ("sintel-clean", "sequence_1/frame_0001"),
        ("sintel-final", "sequence_1/frame_0001"),
    ]
    for dname, dpath in dataset_name_path:
        assert (tmp_path / dname / "flows" / (dpath + ".flo")).exists()
        assert (tmp_path / dname / "flows_viz" / (dpath + ".png")).exists()

    args.flow_format = "png"
    validate.validate(args, model, datamodule)
    for dname, dpath in dataset_name_path:
        assert (tmp_path / dname / "flows" / (dpath + ".png")).exists()

    shutil.rmtree(tmp_path)
