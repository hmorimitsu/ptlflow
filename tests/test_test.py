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
from ptlflow.utils.dummy_datasets import write_kitti
import test

TEST_MODEL = "raft_small"


def test_test(tmp_path: Path) -> None:
    model = ptlflow.get_model(TEST_MODEL)

    data_parser = ArgumentParser()
    data_parser.add_class_arguments(FlowDataModule, "data")
    data_args = data_parser.parse_args([])
    data_args.data.test_dataset = "kitti-2015"
    data_args.data.kitti_2015_root_dir = str(tmp_path / "KITTI/2015")

    data_parser = ArgumentParser(exit_on_error=False)
    data_parser.add_argument("--data", type=FlowDataModule)
    data_cfg = data_parser.parse_object({"data": data_args.data})
    datamodule = data_parser.instantiate_classes(data_cfg).data
    datamodule.setup("test")

    parser = ArgumentParser(parents=[test._init_parser()])
    args = parser.parse_args([])
    args.output_path = str(tmp_path)

    write_kitti(tmp_path)

    test.test(args, model, datamodule)

    dataset_name_path = [
        ("kitti2015", "flow/000000_10.png"),
    ]
    for dname, dpath in dataset_name_path:
        print(tmp_path / dname / dpath)
        assert (tmp_path / dname / dpath).exists()

    shutil.rmtree(tmp_path)


def test_test_max_forward_side_downscales(tmp_path: Path, monkeypatch) -> None:
    model = ptlflow.get_model(TEST_MODEL)

    data_parser = ArgumentParser()
    data_parser.add_class_arguments(FlowDataModule, "data")
    data_args = data_parser.parse_args([])
    data_args.data.test_dataset = "kitti-2015"
    data_args.data.kitti_2015_root_dir = str(tmp_path / "KITTI/2015")

    data_parser = ArgumentParser(exit_on_error=False)
    data_parser.add_argument("--data", type=FlowDataModule)
    data_cfg = data_parser.parse_object({"data": data_args.data})
    datamodule = data_parser.instantiate_classes(data_cfg).data
    datamodule.setup("test")

    parser = ArgumentParser(parents=[test._init_parser()])
    args = parser.parse_args([])
    args.output_path = str(tmp_path)

    write_kitti(tmp_path)  # dummy KITTI images are 375x1242

    # spy on the IOAdapter to check the scale factor given to the model
    recorded_scales = []
    real_io_adapter = test.IOAdapter

    class SpyIOAdapter(real_io_adapter):
        def __init__(self, *io_args, **io_kwargs):
            recorded_scales.append(io_kwargs.get("target_scale_factor"))
            super().__init__(*io_args, **io_kwargs)

    monkeypatch.setattr(test, "IOAdapter", SpyIOAdapter)

    args.max_forward_side = 500
    test.test(args, model, datamodule)

    # the larger side (1242) must be downscaled to 500, so the factor is < 1
    assert len(recorded_scales) > 0
    for scale in recorded_scales:
        assert scale is not None
        assert scale < 1.0
        assert scale == 500.0 / 1242

    shutil.rmtree(tmp_path)
