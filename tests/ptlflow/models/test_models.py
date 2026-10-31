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
import pandas as pd
import pytest
import torch

import ptlflow
import train
import validate
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.dummy_datasets import write_flying_chairs2, write_sintel
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
from ptlflow.utils.registry import RegisteredModel
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
    "flowformer": {"use_tile_input": False},
    "flowformer_pp": {"use_tile_input": False},
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

# Very demanding models, skipped on machines without enough resources (CPU-only
# runners such as GitHub Actions, or GPUs with little VRAM) to keep the test
# runtime and memory usage acceptable.
EXCLUDE_MODELS_LOW_MEM = [
    "ufm_base_dinov2g",  # more than 1B parameters (inference needs ~8 GiB)
]

# Minimum GPU memory required to run the models in EXCLUDE_MODELS_LOW_MEM.
# ufm_base_dinov2g needs around 8 GiB for inference alone, so add a margin to
# avoid flaky out-of-memory errors caused by fragmentation.
MIN_GPU_MEM_GIB = 12.0


def _skip_low_mem_model(mname: str) -> bool:
    if mname not in EXCLUDE_MODELS_LOW_MEM:
        return False
    if not torch.cuda.is_available():
        return True
    total_mem_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    return total_mem_gib < MIN_GPU_MEM_GIB


# Models that require more than two input frames.
SIMPLE_IO_NUM_IMAGES = {
    "memfof": 3,
    "streamflow": 4,  # its update block is built for T=4 frames by default
    "videoflow_bof": 3,
    "videoflow_mof": 3,
}

# Repeated forward passes must produce identical outputs on CPU. On CUDA some
# kernels are non-deterministic, so a small tolerance is required instead.
SIMPLE_IO_ATOL = 0.1 if torch.cuda.is_available() else 0.0


def _get_trainable_model_names() -> list[str]:
    # Only models annotated with @trainable define a loss function, so they
    # are the only ones that can be trained and overfitted.
    trainable_names = set(ptlflow.get_trainable_model_names())
    return [
        n
        for n in ptlflow._models_dict.keys()
        if not ptlflow.is_model_variant(n) and n in trainable_names
    ]


def test_forward() -> None:
    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue
        if _skip_low_mem_model(mname):
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

        with torch.no_grad():
            # Gradients are not needed here, and keeping them would store all
            # the intermediate activations, requiring much more GPU memory
            # (some models can then exceed the capacity of consumer GPUs).
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

            with torch.no_grad():
                model(inputs)


def test_simple_io() -> None:
    model_names = [
        n for n in ptlflow._models_dict.keys() if not ptlflow.is_model_variant(n)
    ]
    for mname in model_names:
        if mname in EXCLUDE_MODELS_SIMPLE_IO:
            continue
        if _skip_low_mem_model(mname):
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
            out_tensor, out_dict, rtol=0, atol=SIMPLE_IO_ATOL, equal_nan=True
        ), f"{mname}: tensor and dict inputs must produce the same output"

        assert isinstance(out_default, dict) and "flows" in out_default, (
            f"{mname}: the default output must be a dict with a flows key, "
            f"got {type(out_default)}"
        )
        flows = out_default["flows"]
        expected = flows[:, 0] if flows.ndim == 5 else flows
        assert torch.allclose(
            out_tensor, expected, rtol=0, atol=SIMPLE_IO_ATOL, equal_nan=True
        ), f"{mname}: the simple_io output must match the default flows output"


@pytest.mark.skip(
    reason="Requires too many resources. Use only on machines with large GPUs."
)
def test_train(tmp_path: Path):
    write_flying_chairs2(tmp_path)

    model_names = _get_trainable_model_names()
    for mname in model_names:
        if mname in EXCLUDE_MODELS:
            continue

        print(mname)
        _train_one_pass(tmp_path, mname)

    shutil.rmtree(tmp_path)


def _train_one_pass(tmp_path: Path, model_name: str) -> None:
    parser = train._init_parser()

    train_args = [
        "--model",
        model_name,
        "--log_dir",
        str(tmp_path / model_name),
        "--trainer.max_epochs",
        str(TRAIN_EPOCHS),
        "--trainer.default_root_dir",
        str(tmp_path / model_name),
        "--trainer.logger",
        "False",
        "--trainer.limit_val_batches",
        "0",
        "--trainer.num_sanity_val_steps",
        "0",
        "--data.train_dataset",
        "overfit-chairs2",
        "--data.val_dataset",
        "overfit-chairs2",
        "--data.flying_chairs2_root_dir",
        str(tmp_path / "FlyingChairs2"),
        "--data.train_batch_size",
        "1",
        "--data.train_crop_size",
        "[256, 256]",
    ]
    if torch.cuda.is_available():
        train_args += ["--trainer.devices", "1"]

    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModule,
        trainer_class=PTLFlowTrainer,
        auto_configure_optimizers=False,
        parser_kwargs={"parents": [parser]},
        run=False,
        ignore_sys_argv=True,
        args=train_args,
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


@pytest.mark.skip(reason="It takes too long. To be used sporadically.")
def test_overfit(tmp_path: Path) -> None:
    print("Saving outputs to " + str(tmp_path))
    write_sintel(tmp_path)

    model_names = _get_trainable_model_names()
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

    train_args = [
        "--model",
        model_name,
        "--log_dir",
        str(tmp_path / model_name),
        "--trainer.max_epochs",
        "100",
        "--trainer.default_root_dir",
        str(tmp_path / model_name),
        "--trainer.logger",
        "False",
        "--trainer.limit_val_batches",
        "0",
        "--trainer.num_sanity_val_steps",
        "0",
        "--data.train_dataset",
        "overfit-sintel",
        "--data.val_dataset",
        "overfit-sintel",
        "--data.mpi_sintel_root_dir",
        str(tmp_path / "MPI-Sintel"),
        "--data.train_batch_size",
        "1",
    ]
    if torch.cuda.is_available():
        train_args += ["--trainer.devices", "1"]

    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModule,
        trainer_class=PTLFlowTrainer,
        auto_configure_optimizers=False,
        parser_kwargs={"parents": [parser]},
        run=False,
        ignore_sys_argv=True,
        args=train_args,
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


def _validate(tmp_path: Path, model_name: str) -> pd.DataFrame:
    ckpt_path = list((tmp_path / model_name).glob("**/*.ckpt"))[0]

    model = ptlflow.get_model(model_name, str(ckpt_path))

    data_parser = ArgumentParser()
    data_parser.add_class_arguments(FlowDataModule, "data")
    data_args = data_parser.parse_args([])
    data_args.data.val_dataset = DATASET
    data_args.data.mpi_sintel_root_dir = str(tmp_path / "MPI-Sintel")

    data_parser = ArgumentParser(exit_on_error=False)
    data_parser.add_argument("--data", type=FlowDataModule)
    data_cfg = data_parser.parse_object({"data": data_args.data})
    datamodule = data_parser.instantiate_classes(data_cfg).data

    parser = ArgumentParser(parents=[validate._init_parser()])
    args = parser.parse_args([])
    args.output_path = str(tmp_path / model_name / "validate")
    args.write_outputs = True
    args.max_samples = 1
    args.model_name = model_name
    args.ckpt_path = str(ckpt_path)

    metrics_df = validate.validate(args, model, datamodule)
    return metrics_df
