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

import pandas as pd
import pytest
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # Workaround until pl stop raising the metrics deprecation warning
    import pytorch_lightning as pl
import torch

import ptlflow
import train
import validate
from ptlflow.utils.dummy_datasets import write_flying_chairs2
from ptlflow.utils.utils import make_divisible

TRAIN_EPOCHS = 1
DATASET = 'overfit'

EXCLUDE_MODELS = [
    'scv4', 'scv8'  # Has additional requirements
]


def test_forward() -> None:
    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        try:
            model = ptlflow.get_model(mname)
            model = model.eval()

            s = make_divisible(128, model.output_stride)
            inputs = {'images': torch.rand(1, 2, 3, s, s)}

            if torch.cuda.is_available():
                model = model.cuda()
                inputs['images'] = inputs['images'].cuda()

            model(inputs)
        except (ImportError, RuntimeError):
            continue


@pytest.mark.skip(reason='Requires too many resources. Use only on machines with large GPUs.')
def test_train(tmp_path: Path):
    write_flying_chairs2(tmp_path)

    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        print(mname)
        _train_one_pass(tmp_path, mname)

    shutil.rmtree(tmp_path)


def _train_one_pass(tmp_path: Path, model_name: str) -> None:
    parser = train._init_parser()

    model_ref = ptlflow.get_model_reference(model_name)
    parser = model_ref.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args([model_name])

    args.log_dir = tmp_path / model_name
    args.max_epochs = 1
    args.train_batch_size = 1
    args.train_dataset = 'overfit-chairs2'
    args.val_dataset = 'none'
    args.flying_chairs2_root_dir = tmp_path / 'FlyingChairs2'
    args.train_crop_size = (256, 256)
    if torch.cuda.is_available():
        args.gpus = 1

    try:
        train.train(args)
    except AssertionError:
        pass  # When the model has no loss function


@pytest.mark.skip(reason='It takes too long. To be used sporadically.')
def test_overfit(tmp_path: Path) -> None:
    print('Saving outputs to ' + str(tmp_path))

    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        print(mname)
        try:
            epe = _overfit_model(tmp_path, mname)
        except KeyError:
            # Models that require backward will raise this, since there is no backward flows in sintel
            continue
        assert epe < 2

    shutil.rmtree(tmp_path)


def _overfit_model(tmp_path: Path, model_name: str) -> float:
    try:
        _train_overfit(tmp_path, model_name)

        metrics_df = _validate(tmp_path, model_name)
        epe = metrics_df.loc[0, 'overfit-val/epe']
    except AssertionError:
        epe = -1
    return epe


def _train_overfit(tmp_path: Path, model_name: str) -> None:
    parser = train._init_parser()

    model_ref = ptlflow.get_model_reference(model_name)
    parser = model_ref.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args([model_name])

    args.log_dir = tmp_path / model_name
    args.max_epochs = 100
    args.train_batch_size = 1
    args.train_dataset = 'overfit-sintel'
    args.val_dataset = 'none'
    args.mpi_sintel_root_dir = Path('tests/data/ptlflow/models/sintel')
    if torch.cuda.is_available():
        args.gpus = 1

    train.train(args)


def _validate(tmp_path: Path, model_name: str) -> pd.DataFrame:
    parser = validate._init_parser()

    model_ref = ptlflow.get_model_reference(model_name)
    parser = model_ref.add_model_specific_args(parser)

    args = parser.parse_args([model_name])

    args.output_path = tmp_path / model_name
    args.mpi_sintel_root_dir = Path('tests/data/ptlflow/models/sintel')
    args.val_dataset = DATASET
    args.pretrained_ckpt = str(list((tmp_path / model_name).glob('**/*_last_*.ckpt'))[0])
    args.write_viz = True

    model = ptlflow.get_model(model_name, args.pretrained_ckpt, args)
    metrics_df = validate.validate(args, model)
    return metrics_df
