"""Validate optical flow estimation performance on standard datasets."""

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

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.onnx

this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(this_dir.parent.parent.parent))

from ptlflow import get_model, load_checkpoint
from ptlflow.models.rapidflow.rapidflow import RAPIDFlow


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        choices=(
            "rapidflow",
            "rapidflow_it1",
            "rapidflow_it2",
            "rapidflow_it3",
            "rapidflow_it6",
            "rapidflow_it12",
        ),
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to be loaded. It can also be one of the following names: \{chairs, things, sintel, kitti\}, in which case the respective pretrained checkpoint will be downloaded.",
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
        default=(384, 1280),
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


def load_model(args):
    model = get_model(args.model, args=args)
    ckpt = load_checkpoint(args.checkpoint, RAPIDFlow, "rapidflow")
    state_dict = fuse_checkpoint_next1d_layers(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    parser = _init_parser()
    parser = RAPIDFlow.add_model_specific_args(parser)
    args = parser.parse_args()
    args.corr_mode = "allpairs"
    args.fuse_next1d_weights = True
    args.simple_io = True

    model = load_model(args)
    sample_inputs = torch.randn(1, 2, 3, args.input_size[0], args.input_size[1])
    if torch.cuda.is_available():
        model = model.cuda()
        sample_inputs = sample_inputs.cuda()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{args.model}.onnx")
    torch.onnx.export(
        model, sample_inputs, output_path, verbose=False, opset_version=16
    )
    print(f"ONNX model saved to: {output_path}")
