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

import torch

from ptlflow.data.datasets import (
    AutoFlowDataset,
    FlyingChairsDataset,
    FlyingChairs2Dataset,
    FlyingThings3DDataset,
    FlyingThings3DSubsetDataset,
    Hd1kDataset,
    KittiDataset,
    KubricDataset,
    MiddleburyDataset,
    MiddleburySTDataset,
    MonkaaDataset,
    SintelDataset,
    SpringDataset,
    TartanAirDataset,
    ViperDataset,
)
from ptlflow.data.flow_transforms import ToTensor
from ptlflow.utils import dummy_datasets


def test_autoflow(tmp_path: Path) -> None:
    dummy_datasets.write_autoflow(tmp_path)

    for split in ["trainval"]:
        dataset = AutoFlowDataset(
            root_dir=tmp_path / "autoflow",
            split=split,
            transform=ToTensor(),
            get_valid_mask=True,
            get_meta=True,
        )

        inputs = dataset[0]

        assert inputs.get("meta") is not None

        keys = ["images", "flows", "valids"]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_chairs(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs(tmp_path)

    for split in ["trainval"]:
        dataset = FlyingChairsDataset(
            root_dir=tmp_path / "FlyingChairs_release",
            split=split,
            transform=ToTensor(),
            get_valid_mask=True,
            get_meta=True,
        )

        inputs = dataset[0]

        assert inputs.get("meta") is not None

        keys = ["images", "flows", "valids"]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_chairs2(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs2(tmp_path)

    for split in ["trainval"]:
        dataset = FlyingChairs2Dataset(
            root_dir=tmp_path / "FlyingChairs2",
            split=split,
            transform=ToTensor(),
            add_reverse=True,
            get_backward=True,
            get_occlusion_mask=True,
            get_motion_boundary_mask=True,
            get_valid_mask=True,
            get_meta=True,
        )

        inputs = dataset[0]

        assert inputs.get("meta") is not None

        keys = [
            "images",
            "flows",
            "valids",
            "occs",
            "mbs",
            "flows_b",
            "valids_b",
            "occs_b",
            "mbs_b",
        ]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_hd1k(tmp_path: Path) -> None:
    dummy_datasets.write_hd1k(tmp_path)

    for split in ["trainval", "test"]:
        dataset = Hd1kDataset(
            root_dir=tmp_path / "HD1K",
            split=split,
            transform=ToTensor(),
            get_valid_mask=True,
            get_meta=True,
        )

        inputs = dataset[0]

        assert inputs.get("meta") is not None

        if split == "test":
            keys = ["images"]
        else:
            keys = ["images", "flows", "valids"]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_kitti(tmp_path: Path) -> None:
    dummy_datasets.write_kitti(tmp_path)

    for version in ["2012", "2015"]:
        for split in ["trainval", "test"]:
            dataset = KittiDataset(
                root_dir_2012=tmp_path / "KITTI/2012",
                root_dir_2015=tmp_path / "KITTI/2015",
                versions=[version],
                split=split,
                transform=ToTensor(),
                get_valid_mask=True,
                get_meta=True,
            )

            inputs = dataset[0]

            assert inputs.get("meta") is not None

            if split == "test":
                keys = ["images"]
            else:
                keys = ["images", "flows", "valids"]
            for k in keys:
                assert inputs.get(k) is not None
                assert isinstance(inputs[k], torch.Tensor)
                assert len(inputs[k].shape) == 4
                assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_kubric(tmp_path: Path) -> None:
    dummy_datasets.write_kubric(tmp_path)

    dataset = KubricDataset(
        root_dir=tmp_path / "kubric",
        transform=ToTensor(),
        get_backward=True,
        get_valid_mask=True,
        get_meta=True,
    )

    inputs = dataset[0]

    assert inputs.get("meta") is not None

    keys = [
        "images",
        "flows",
        "valids",
        "flows_b",
        "valids_b",
    ]
    for k in keys:
        assert inputs.get(k) is not None
        assert isinstance(inputs[k], torch.Tensor)
        assert len(inputs[k].shape) == 4
        assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_middlebury_st(tmp_path: Path) -> None:
    dummy_datasets.write_middlebury_st(tmp_path)

    dataset = MiddleburySTDataset(
        root_dir=tmp_path / "middlebury_st",
        transform=ToTensor(),
        get_valid_mask=True,
        get_meta=True,
    )

    inputs = dataset[0]

    assert inputs.get("meta") is not None

    keys = [
        "images",
        "flows",
        "valids",
    ]
    for k in keys:
        assert inputs.get(k) is not None
        assert isinstance(inputs[k], torch.Tensor)
        assert len(inputs[k].shape) == 4
        assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_sintel(tmp_path: Path) -> None:
    dummy_datasets.write_sintel(tmp_path)

    for pass_name in ["clean", "final"]:
        for split in ["trainval", "test"]:
            dataset = SintelDataset(
                root_dir=tmp_path / "MPI-Sintel",
                split=split,
                pass_names=[pass_name],
                transform=ToTensor(),
                get_valid_mask=True,
                get_meta=True,
                get_occlusion_mask=True,
            )

            assert len(dataset)
            inputs = dataset[0]

            assert inputs.get("meta") is not None

            if split == "test":
                keys = ["images"]
            else:
                keys = ["images", "flows", "valids", "occs"]
            for k in keys:
                assert inputs.get(k) is not None
                assert isinstance(inputs[k], torch.Tensor)
                assert len(inputs[k].shape) == 4
                assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_spring(tmp_path: Path) -> None:
    dummy_datasets.write_spring(tmp_path, write_4k_image=True)

    for split in ["train", "test"]:
        for side in ["left", "right"]:
            dataset = SpringDataset(
                root_dir=tmp_path / "spring",
                split=split,
                side_names=side,
                add_reverse=True,
                transform=ToTensor(),
                get_valid_mask=True,
                get_backward=True,
                get_meta=True,
                subsample=False,
                is_image_4k=False,
            )

            assert len(dataset)
            inputs = dataset[0]

            assert inputs.get("meta") is not None

            if split == "test":
                keys = ["images"]
            else:
                keys = ["images", "flows", "valids", "flows_b", "valids_b"]
            for k in keys:
                assert inputs.get(k) is not None
                assert isinstance(inputs[k], torch.Tensor)
                if "flow" in k:
                    assert len(inputs[k].shape) == 5
                    assert inputs[k].shape[1] == 4
                    assert inputs[k].max() == 1.0
                    assert inputs["images"].shape[-2] == inputs[k].shape[-2]
                    assert inputs["images"].shape[-1] == inputs[k].shape[-1]
                else:
                    assert len(inputs[k].shape) == 4
                assert min(inputs[k].shape) > 0

            if split == "train":
                dataset_sub = SpringDataset(
                    root_dir=tmp_path / "spring",
                    split=split,
                    side_names=side,
                    add_reverse=True,
                    transform=ToTensor(),
                    get_valid_mask=True,
                    get_backward=True,
                    get_meta=True,
                    subsample=True,
                    is_image_4k=False,
                )
                assert len(dataset_sub)
                inputs_sub = dataset_sub[0]
                for k in ["flows", "flows_b"]:
                    assert len(inputs_sub[k].shape) == 4
                    assert inputs[k].max() == 1.0
                    assert inputs_sub["images"].shape[-2] == inputs_sub[k].shape[-2]
                    assert inputs_sub["images"].shape[-1] == inputs_sub[k].shape[-1]

                dataset_4k = SpringDataset(
                    root_dir=tmp_path / "spring",
                    split=split,
                    side_names=side,
                    add_reverse=True,
                    transform=ToTensor(),
                    get_valid_mask=True,
                    get_backward=True,
                    get_meta=True,
                    subsample=False,
                    is_image_4k=True,
                )

                assert len(dataset_4k)
                inputs_4k = dataset_4k[0]

                assert inputs_4k.get("meta") is not None

                keys = ["images", "flows", "valids", "flows_b", "valids_b"]
                for k in keys:
                    assert inputs_4k.get(k) is not None
                    assert isinstance(inputs[k], torch.Tensor)
                    if "flow" in k:
                        assert len(inputs_4k[k].shape) == 4
                        assert inputs_4k[k].max() == 2.0
                        assert inputs_4k["images"].shape[-2] == inputs_4k[k].shape[-2]
                        assert inputs_4k["images"].shape[-1] == inputs_4k[k].shape[-1]
                    else:
                        assert len(inputs_4k[k].shape) == 4
                    assert min(inputs_4k[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_tartanair(tmp_path: Path) -> None:
    dummy_datasets.write_tartanair(tmp_path)

    for difficulty in ["Easy", "Hard"]:
        dataset = TartanAirDataset(
            root_dir=tmp_path / "tartanair",
            difficulties=difficulty,
            transform=ToTensor(),
            get_valid_mask=True,
            get_meta=True,
            get_occlusion_mask=True,
        )

        assert len(dataset)
        inputs = dataset[0]

        assert inputs.get("meta") is not None

        keys = ["images", "flows", "valids", "occs"]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_things(tmp_path: Path) -> None:
    dummy_datasets.write_things(tmp_path)

    for side_name in ["left", "right"]:
        for pass_name in ["clean", "final"]:
            for split in ["trainval"]:
                dataset = FlyingThings3DDataset(
                    root_dir=tmp_path / "FlyingThings3D",
                    split=split,
                    pass_names=[pass_name],
                    side_names=[side_name],
                    transform=ToTensor(),
                    add_reverse=True,
                    get_backward=True,
                    get_occlusion_mask=True,
                    get_motion_boundary_mask=True,
                    get_valid_mask=True,
                    get_meta=True,
                )

                inputs = dataset[0]

                assert inputs.get("meta") is not None

                # keys = ['images', 'flows', 'valids', 'occs', 'mbs', 'flows_b', 'valids_b', 'occs_b', 'mbs_b']
                keys = ["images", "flows", "valids", "flows_b", "valids_b"]
                for k in keys:
                    assert inputs.get(k) is not None
                    assert isinstance(inputs[k], torch.Tensor)
                    assert len(inputs[k].shape) == 4
                    assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_things_subset(tmp_path: Path) -> None:
    dummy_datasets.write_things_subset(tmp_path)

    for side_name in ["left", "right"]:
        for pass_name in ["clean", "final"]:
            for split in ["trainval"]:
                dataset = FlyingThings3DSubsetDataset(
                    root_dir=tmp_path / "FlyingThings3D_subset",
                    split=split,
                    pass_names=[pass_name],
                    side_names=[side_name],
                    transform=ToTensor(),
                    add_reverse=True,
                    get_backward=True,
                    get_occlusion_mask=True,
                    get_motion_boundary_mask=True,
                    get_valid_mask=True,
                    get_meta=True,
                )

                inputs = dataset[0]

                assert inputs.get("meta") is not None

                # keys = ['images', 'flows', 'valids', 'occs', 'mbs', 'flows_b', 'valids_b', 'occs_b', 'mbs_b']
                keys = ["images", "flows", "valids", "flows_b", "valids_b"]
                for k in keys:
                    assert inputs.get(k) is not None
                    assert isinstance(inputs[k], torch.Tensor)
                    assert len(inputs[k].shape) == 4
                    assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)


def test_viper(tmp_path: Path) -> None:
    dummy_datasets.write_viper(tmp_path)

    for split in ["trainval"]:
        dataset = ViperDataset(
            root_dir=tmp_path / "viper",
            split=split,
            transform=ToTensor(),
            get_valid_mask=True,
            get_meta=True,
        )

        inputs = dataset[0]

        assert inputs.get("meta") is not None

        keys = [
            "images",
            "flows",
            "valids",
        ]
        for k in keys:
            assert inputs.get(k) is not None
            assert isinstance(inputs[k], torch.Tensor)
            assert len(inputs[k].shape) == 4
            assert min(inputs[k].shape) > 0

    shutil.rmtree(tmp_path)
