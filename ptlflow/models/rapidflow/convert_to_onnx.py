"""Validate optical flow estimation performance on standard datasets."""

# =============================================================================
# Copyright 2024 Henrique Morimitsu
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

import sys
from pathlib import Path

from jsonargparse import ArgumentParser
from loguru import logger
import torch
import torch.onnx

this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(this_dir.parent.parent.parent))

from ptlflow import get_model, load_checkpoint
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint to be loaded. It can also be one of the following names: {chairs, things, sintel, kitti}, in which case the respective pretrained checkpoint will be downloaded.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path to the directory where the converted onnx model will be saved.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[384, 1280],
        help="Size of the input image.",
    )
    return parser


def fuse_checkpoint_next1d_layers(state_dict):
    fused_sd = {}
    hv_pairs = {}
    for name, param in state_dict.items():
        if name.endswith("weight_h") or name.endswith("weight_v"):
            name_prefix = name[: -(len("weight_h") + 1)]
            orientation = name[-1]
            if name_prefix not in hv_pairs:
                hv_pairs[name_prefix] = {}
            hv_pairs[name_prefix][orientation] = param
        else:
            fused_sd[name] = param

    for name_prefix, param_pairs in hv_pairs.items():
        weight = torch.einsum("cijk,cimj->cimk", param_pairs["h"], param_pairs["v"])
        fused_sd[f"{name_prefix}.weight"] = weight
    return fused_sd


def load_model(cfg):
    ckpt_path = cfg.ckpt_path
    cfg.ckpt_path = None  # This is to avoid letting get_model to load the ckpt
    model = get_model(cfg.model_name, args=cfg)
    # Since we set fuse_next1d_weights to True in the model,
    # the ckpt 1D layers also need to be fused before loading
    ckpt = load_checkpoint(ckpt_path, model.__class__)
    state_dict = fuse_checkpoint_next1d_layers(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    return model


def _show_v04_warning():
    ignore_args = ["-h", "--help", "--model", "--config"]
    for arg in ignore_args:
        if arg in sys.argv:
            return

    logger.warning(
        "Since v0.4, it is now necessary to inform the model using the --model argument. For example, use: python infer.py --model raft --ckpt_path things"
    )


if __name__ == "__main__":
    _show_v04_warning()

    parser = _init_parser()

    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        parser_kwargs={"parents": [parser]},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )

    cfg = cli.config
    cfg.model.init_args.corr_mode = "allpairs"
    cfg.model.init_args.fuse_next1d_weights = True
    cfg.model.init_args.simple_io = True
    cfg.model_name = cfg.model.class_path.split(".")[-1]

    model = load_model(cfg)

    # model = get_model(cfg.model_name, args=cfg)
    sample_inputs = torch.randn(1, 2, 3, cfg.input_size[0], cfg.input_size[1])
    if torch.cuda.is_available():
        model = model.cuda()
        sample_inputs = sample_inputs.cuda()

    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{cfg.model_name}.onnx")
    torch.onnx.export(
        model, sample_inputs, output_path, verbose=False, opset_version=16
    )
    print(f"ONNX model saved to: {output_path}")
