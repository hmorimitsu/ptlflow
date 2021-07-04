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

import ptlflow
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import dummy_datasets

MODEL_NAME = 'raft_small'


def test_chairs(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs(tmp_path)
    model = _get_model(flying_chairs_root_dir=tmp_path / 'FlyingChairs_release')
    for is_train in [True, False]:
        model._get_chairs_dataset(is_train)
    shutil.rmtree(tmp_path)


def test_chairs2(tmp_path: Path) -> None:
    dummy_datasets.write_flying_chairs(tmp_path)
    model = _get_model(flying_chairs2_root_dir=tmp_path / 'FlyingChairs2')
    for is_train in [True, False]:
        model._get_chairs2_dataset(is_train)
    shutil.rmtree(tmp_path)


def test_hd1k(tmp_path: Path) -> None:
    dummy_datasets.write_hd1k(tmp_path)
    model = _get_model(hd1k_root_dir=tmp_path / 'HD1K')
    for is_train in [True, False]:
        model._get_hd1k_dataset(is_train)
    shutil.rmtree(tmp_path)


def test_kitti(tmp_path: Path) -> None:
    dummy_datasets.write_kitti(tmp_path)
    model = _get_model(kitti_2012_root_dir=tmp_path / 'KITTI/2012', kitti_2015_root_dir=tmp_path / 'KITTI/2015')
    for is_train in [True, False]:
        model._get_kitti_dataset(is_train, '2012')
        model._get_kitti_dataset(is_train, '2015')
    shutil.rmtree(tmp_path)


def test_sintel(tmp_path: Path) -> None:
    dummy_datasets.write_sintel(tmp_path)
    model = _get_model(mpi_sintel_root_dir=tmp_path / 'MPI-Sintel')
    for is_train in [True, False]:
        model._get_sintel_dataset(is_train)
    shutil.rmtree(tmp_path)


def test_things(tmp_path: Path) -> None:
    dummy_datasets.write_things(tmp_path)
    model = _get_model(flying_things3d_root_dir=tmp_path / 'FlyingThings3D')
    for is_train in [True, False]:
        model._get_things_dataset(is_train)
    shutil.rmtree(tmp_path)


def test_things_subset(tmp_path: Path) -> None:
    dummy_datasets.write_things_subset(tmp_path)
    model = _get_model(flying_things3d_subset_root_dir=tmp_path / 'FlyingThings3D_subset')
    for is_train in [True, False]:
        model._get_things_dataset(is_train, 'subset')
    shutil.rmtree(tmp_path)


def _get_model(
    **kwargs: Path
) -> BaseModel:
    model_ref = ptlflow.get_model_reference(MODEL_NAME)
    parser = model_ref.add_model_specific_args()
    args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)
    model = ptlflow.get_model(MODEL_NAME, None, args)
    return model
