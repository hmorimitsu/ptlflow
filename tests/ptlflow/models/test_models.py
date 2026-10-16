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
import lightning.pytorch as pl
import pandas as pd
import pytest
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
    "scv4",
    "scv8",
    "separableflow",
    "splatflow",
    "streamflow",
    "waft_dinov3_a2",
]  # Has additional requirements. Mode variants (see register_model_variants) are also excluded.

EXCLUDE_MODELS_FP16 = [
    "lcv_raft",
    "lcv_raft_small",
    "matchflow",
    "matchflow_raft",
    "scv4",
    "scv8",
    "separableflow",
    "splatflow",
    "streamflow",
    "waft_dinov3_a2",
]  # Some operations do not support fp16

MODEL_ARGS = {
    "ccmr": {"alternate_corr": False},
    "ccmr_p": {"alternate_corr": False},
    "flowformer": {"use_tile_input": False},
    "flowformer_pp": {"use_tile_input": False},
    "ms_raft_p": {"alternate_corr": False},
    "rapidflow": {"corr_mode": "allpairs"},
    "rapidflow_it1": {"corr_mode": "allpairs"},
    "rapidflow_it2": {"corr_mode": "allpairs"},
    "rapidflow_it3": {"corr_mode": "allpairs"},
    "rapidflow_it6": {"corr_mode": "allpairs"},
    "rapidflow_it12": {"corr_mode": "allpairs"},
    "rpknet": {"corr_mode": "allpairs"},
}

# Only models with additional (not installed) requirements are skipped here.
# streamflow is not skipped: unlike test_forward, the input below has enough
# frames for it (see SIMPLE_IO_NUM_IMAGES).
EXCLUDE_MODELS_SIMPLE_IO = [m for m in EXCLUDE_MODELS if m != "streamflow"]

# Models that require more than two input frames.
SIMPLE_IO_NUM_IMAGES = {
    "memfof": 3,
    "streamflow": 4,  # its update block is built for T=4 frames by default
    "videoflow_bof": 3,
    "videoflow_mof": 3,
}


def test_forward() -> None:
    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        print(mname)
        model_ref = ptlflow.get_model_reference(mname)
        parser = ArgumentParser()
        parser.add_class_arguments(model_ref, "model")
        args = parser.parse_args([])

        if mname in MODEL_ARGS:
            for name, val in MODEL_ARGS[mname].items():
                setattr(args.model, name, val)

        model = ptlflow.get_model(mname, args=args)
        model = model.eval()

        out_stride = model.output_stride
        if out_stride is None:
            out_stride = 64
        s = make_divisible(256, out_stride)
        num_images = 2
        if mname in ["videoflow_bof", "videoflow_mof", "memfof"]:
            num_images = 3
        inputs = {"images": torch.rand(1, num_images, 3, s, s)}

        if torch.cuda.is_available():
            model = model.cuda()
            inputs["images"] = inputs["images"].cuda()

        model(inputs)


def test_forward_fp16() -> None:
    if torch.cuda.is_available():
        model_names = [
            n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
        ]
        for mname in model_names:
            if mname in EXCLUDE_MODELS_FP16:
                continue

            print(mname)
            model_ref = ptlflow.get_model_reference(mname)
            parser = ArgumentParser()
            parser.add_class_arguments(model_ref, "model")
            args = parser.parse_args([])

            if mname in MODEL_ARGS:
                for name, val in MODEL_ARGS[mname].items():
                    setattr(args.model, name, val)

            model = ptlflow.get_model(mname, args=args)
            model = model.eval()
            model = model.half()

            out_stride = model.output_stride
            if out_stride is None:
                out_stride = 64
            s = make_divisible(256, out_stride)
            num_images = 2
            if mname in ["videoflow_bof", "videoflow_mof", "memfof"]:
                num_images = 3
            inputs = {"images": torch.rand(1, num_images, 3, s, s)}

            if torch.cuda.is_available():
                model = model.cuda()
                inputs["images"] = inputs["images"].cuda().half()

            model(inputs)


def test_simple_io() -> None:
    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
    for mname in model_names:
        if mname in EXCLUDE_MODELS_SIMPLE_IO:
            continue

        print(mname)
        model_ref = ptlflow.get_model_reference(mname)
        parser = ArgumentParser()
        parser.add_class_arguments(model_ref, "model")
        args = parser.parse_args([])

        if mname in MODEL_ARGS:
            for name, val in MODEL_ARGS[mname].items():
                setattr(args.model, name, val)
        setattr(args.model, "simple_io", True)

        model = ptlflow.get_model(mname, args=args)
        model = model.eval()

        out_stride = model.output_stride
        if out_stride is None:
            out_stride = 64
        s = make_divisible(256, out_stride)
        num_images = SIMPLE_IO_NUM_IMAGES.get(mname, 2)
        inputs = torch.rand(1, num_images, 3, s, s)

        if torch.cuda.is_available():
            model = model.cuda()
            inputs = inputs.cuda()

        with torch.no_grad():
            # A raw image tensor must be accepted, producing a 4D flow tensor.
            out_tensor = model(inputs)
            # A dict input must also be accepted while simple_io is enabled.
            out_dict = model({"images": inputs})
            # With simple_io disabled, the output must be the standard dict.
            model.simple_io = False
            out_default = model({"images": inputs})

        assert isinstance(out_tensor, torch.Tensor) and out_tensor.ndim == 4, (
            f"{mname}: simple_io with a tensor input must return a 4D flow tensor, "
            f"got {type(out_tensor)} with shape {getattr(out_tensor, 'shape', None)}"
        )
        assert isinstance(out_dict, torch.Tensor), (
            f"{mname}: simple_io with a dict input must return a tensor, "
            f"got {type(out_dict)}"
        )
        assert torch.allclose(
            out_tensor, out_dict, rtol=0, atol=0, equal_nan=True
        ), f"{mname}: tensor and dict inputs must produce the same output"

        assert isinstance(out_default, dict) and "flows" in out_default, (
            f"{mname}: the default output must be a dict with a flows key, "
            f"got {type(out_default)}"
        )
        flows = out_default["flows"]
        expected = flows[:, 0] if flows.ndim == 5 else flows
        assert torch.allclose(
            out_tensor, expected, rtol=0, atol=0, equal_nan=True
        ), f"{mname}: the simple_io output must match the default flows output"


@pytest.mark.skip(
    reason="Requires too many resources. Use only on machines with large GPUs."
)
def test_train(tmp_path: Path):
    write_flying_chairs2(tmp_path)

    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
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

    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
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
