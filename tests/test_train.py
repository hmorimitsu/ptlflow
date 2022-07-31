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
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # Workaround until pl stop raising the metrics deprecation warning
    import pytorch_lightning as pl
import torch

import ptlflow
import train
from ptlflow.utils.dummy_datasets import write_flying_chairs, write_kitti, write_sintel

TEST_MODEL = 'raft_small'
TRAIN_DATASET = 'chairs-train'
TRAIN_LOG_SUFFIX = 'chairs'


def test_train(tmp_path: Path) -> None:
    parser = train._init_parser()

    model_ref = ptlflow.get_model_reference(TEST_MODEL)
    parser = model_ref.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args([TEST_MODEL])

    args.log_dir = tmp_path
    args.max_epochs = 1
    args.train_batch_size = 1
    args.limit_train_batches = 1
    args.limit_val_batches = 1
    args.num_sanity_val_steps = 0
    args.train_dataset = TRAIN_DATASET
    args.mpi_sintel_root_dir = tmp_path / 'MPI-Sintel'
    args.kitti_2012_root_dir = tmp_path / 'KITTI/2012'
    args.kitti_2015_root_dir = tmp_path / 'KITTI/2015'
    args.flying_chairs_root_dir = tmp_path / 'FlyingChairs_release'
    if torch.cuda.is_available():
        args.gpus = 1

    write_flying_chairs(tmp_path)
    write_kitti(tmp_path)
    write_sintel(tmp_path)

    train.train(args)

    dir_names = ['default', 'lightning_logs']  # Name changes depending on PL version
    hparams_res = []
    last_res = []
    train_res = []
    for dname in dir_names:
        log_dirs = Path(f'{dname}/version_0')

        hparams_res.append((tmp_path / f'{TEST_MODEL}-{TRAIN_LOG_SUFFIX}' / log_dirs / 'hparams.yaml').exists())
        last_res.append(len(list((tmp_path / f'{TEST_MODEL}-{TRAIN_LOG_SUFFIX}' / log_dirs / 'checkpoints').glob('*_last_*.ckpt'))))
        train_res.append(len(list((tmp_path / f'{TEST_MODEL}-{TRAIN_LOG_SUFFIX}' / log_dirs / 'checkpoints').glob('*_train_*.ckpt'))))

    assert max(hparams_res) is True
    assert max(last_res) > 0
    assert max(train_res) > 0

    shutil.rmtree(tmp_path)
