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

import pandas as pd
import pytest

import lightning.pytorch as pl
import torch

import ptlflow
import train
import validate
from ptlflow.utils.dummy_datasets import write_flying_chairs2
from ptlflow.utils.utils import make_divisible

TRAIN_EPOCHS = 1
DATASET = "overfit"

EXCLUDE_MODELS = [
    "matchflow",
    "matchflow_raft",
    "neuflow",  # requires torch 2.X
    "neuflow2",  # requires torch 2.X
    "scv4",
    "scv8",
    "separableflow",
    "splatflow",
]  # Has additional requirements

EXCLUDE_MODELS_FP16 = [
    "lcv_raft",
    "lcv_raft_small",
    "matchflow",
    "matchflow_raft",
    "neuflow",  # requires torch 2.X
    "neuflow2",  # requires torch 2.X
    "scv4",
    "scv8",
    "separableflow",
    "splatflow",
]  # Some operations do not support fp16

MODEL_ARGS = {
    "ccmr": {"alternate_corr": False},
    "ccmr+": {"alternate_corr": False},
    "flowformer": {"use_tile_input": False},
    "flowformer++": {"use_tile_input": False},
    "ms_raft+": {"alternate_corr": False},
    "rapidflow": {"corr_mode": "allpairs"},
    "rapidflow_it1": {"corr_mode": "allpairs"},
    "rapidflow_it2": {"corr_mode": "allpairs"},
    "rapidflow_it3": {"corr_mode": "allpairs"},
    "rapidflow_it6": {"corr_mode": "allpairs"},
    "rapidflow_it12": {"corr_mode": "allpairs"},
    "rpknet": {"corr_mode": "allpairs"},
}


def test_forward() -> None:
    model_names = ptlflow.models_dict.keys()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        print(mname)
        model_ref = ptlflow.get_model_reference(mname)
        parser = model_ref.add_model_specific_args()
        args = parser.parse_args([])

        if mname in MODEL_ARGS:
            for name, val in MODEL_ARGS[mname].items():
                setattr(args, name, val)

        model = ptlflow.get_model(mname, args=args)
        model = model.eval()

        s = make_divisible(256, model.output_stride)
        num_images = 2
        if mname in ["videoflow_bof", "videoflow_mof"]:
            num_images = 3
        inputs = {"images": torch.rand(1, num_images, 3, s, s)}

        if torch.cuda.is_available():
            model = model.cuda()
            inputs["images"] = inputs["images"].cuda()

        model(inputs)


def test_forward_fp16() -> None:
    if torch.cuda.is_available():
        model_names = ptlflow.models_dict.keys()
        for mname in model_names:
            if mname in EXCLUDE_MODELS_FP16:
                continue

            print(mname)
            model_ref = ptlflow.get_model_reference(mname)
            parser = model_ref.add_model_specific_args()
            args = parser.parse_args([])

            if mname in MODEL_ARGS:
                for name, val in MODEL_ARGS[mname].items():
                    setattr(args, name, val)

            model = ptlflow.get_model(mname, args=args)
            model = model.eval()
            model = model.half()

            s = make_divisible(256, model.output_stride)
            num_images = 2
            if mname in ["videoflow_bof", "videoflow_mof"]:
                num_images = 3
            inputs = {"images": torch.rand(1, num_images, 3, s, s)}

            if torch.cuda.is_available():
                model = model.cuda()
                inputs["images"] = inputs["images"].cuda().half()

            model(inputs)


@pytest.mark.skip(
    reason="Requires too many resources. Use only on machines with large GPUs."
)
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
    args.train_dataset = "overfit-chairs2"
    args.val_dataset = "none"
    args.flying_chairs2_root_dir = tmp_path / "FlyingChairs2"
    args.train_crop_size = (256, 256)
    if torch.cuda.is_available():
        args.gpus = 1

    try:
        train.train(args)
    except AssertionError:
        pass  # When the model has no loss function


@pytest.mark.skip(reason="It takes too long. To be used sporadically.")
def test_overfit(tmp_path: Path) -> None:
    print("Saving outputs to " + str(tmp_path))

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
        epe = metrics_df.loc[0, "overfit-val/epe"]
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
    args.train_dataset = "overfit-sintel"
    args.val_dataset = "none"
    args.mpi_sintel_root_dir = Path("tests/data/ptlflow/models/sintel")
    if torch.cuda.is_available():
        args.gpus = 1

    train.train(args)


def _validate(tmp_path: Path, model_name: str) -> pd.DataFrame:
    parser = validate._init_parser()

    model_ref = ptlflow.get_model_reference(model_name)
    parser = model_ref.add_model_specific_args(parser)

    args = parser.parse_args([model_name])

    args.output_path = tmp_path / model_name
    args.mpi_sintel_root_dir = Path("tests/data/ptlflow/models/sintel")
    args.val_dataset = DATASET
    args.pretrained_ckpt = str(
        list((tmp_path / model_name).glob("**/*_last_*.ckpt"))[0]
    )
    args.write_viz = True

    model = ptlflow.get_model(model_name, args.pretrained_ckpt, args)
    metrics_df = validate.validate(args, model)
    return metrics_df
