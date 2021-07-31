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
import validate
from ptlflow.utils.dummy_datasets import write_kitti, write_sintel

TEST_MODEL = 'raft_small'


def test_validate(tmp_path: Path) -> None:
    parser = validate._init_parser()

    model_ref = ptlflow.get_model_reference(TEST_MODEL)
    parser = model_ref.add_model_specific_args(parser)

    args = parser.parse_args([TEST_MODEL])

    args.output_path = tmp_path
    args.write_outputs = True
    args.max_samples = 1
    args.mpi_sintel_root_dir = tmp_path / 'MPI-Sintel'
    args.kitti_2012_root_dir = tmp_path / 'KITTI/2012'
    args.kitti_2015_root_dir = tmp_path / 'KITTI/2015'

    write_kitti(tmp_path)
    write_sintel(tmp_path)

    args.flow_format = 'flo'
    model = ptlflow.get_model(TEST_MODEL, None, args)
    metrics_df = validate.validate(args, model)
    assert min(metrics_df.shape) > 0

    dataset_name_path = [
        ('kitti-2012-trainval', '000000_10'),
        ('kitti-2015-trainval', '000000_10'),
        ('sintel-clean-trainval-occ', 'sequence_1/frame_0001'),
        ('sintel-final-trainval-occ', 'sequence_1/frame_0001')]
    for dname, dpath in dataset_name_path:
        assert (tmp_path / dname / 'flows' / (dpath+'.flo')).exists()
        assert (tmp_path / dname / 'flows_viz' / (dpath+'.png')).exists()

    args.flow_format = 'png'
    model = ptlflow.get_model(TEST_MODEL, None, args)
    validate.validate(args, model)
    for dname, dpath in dataset_name_path:
        assert (tmp_path / dname / 'flows' / (dpath+'.png')).exists()

    shutil.rmtree(tmp_path)
