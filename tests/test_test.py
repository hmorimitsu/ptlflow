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
import test
from ptlflow.utils.dummy_datasets import write_kitti, write_sintel

TEST_MODEL = 'raft_small'


def test_test(tmp_path: Path) -> None:
    parser = test._init_parser()

    model_ref = ptlflow.get_model_reference(TEST_MODEL)
    parser = model_ref.add_model_specific_args(parser)

    args = parser.parse_args([TEST_MODEL])

    args.test_dataset = ['kitti-2012', 'kitti-2015', 'sintel']
    args.output_path = tmp_path
    args.mpi_sintel_root_dir = tmp_path / 'MPI-Sintel'
    args.kitti_2012_root_dir = tmp_path / 'KITTI/2012'
    args.kitti_2015_root_dir = tmp_path / 'KITTI/2015'

    write_kitti(tmp_path)
    write_sintel(tmp_path)

    model = ptlflow.get_model(TEST_MODEL, None, args)
    test.test(args, model)

    dataset_name_path = [
        ('kitti2012', '000000_10.png'),
        ('kitti2015', '000000_10.png'),
        ('sintel/clean', 'sequence_1/frame_0001.flo'),
        ('sintel/final', 'sequence_1/frame_0001.flo')]
    for dname, dpath in dataset_name_path:
        assert (tmp_path / dname / dpath).exists()

    shutil.rmtree(tmp_path)
