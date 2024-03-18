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

from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import onnx
import onnxruntime

this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(this_dir.parent.parent.parent))

from ptlflow.utils import flow_utils


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs=2,
        default=(
            str(this_dir / "image_samples" / "000000_10.png"),
            str(this_dir / "image_samples" / "000000_11.png"),
        ),
        help="Path to two images to estimate the optical flow with the ONNX model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path to the directory where the output flow will be saved.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=(384, 1280),
        help="Size of the input image.",
    )
    return parser


def load_images(image_paths):
    images = [cv.imread(p) for p in image_paths]
    images = [cv.resize(im, args.input_size[::-1]) for im in images]
    images = np.stack(images)
    images = images.transpose(0, 3, 1, 2)[None]
    images = images.astype(np.float32) / 255.0
    return images


if __name__ == "__main__":
    parser = _init_parser()
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_path)
    # onnx.checker.check_model(onnx_model)

    inputs = load_images(args.image_paths)
    print(inputs.shape, inputs.max())

    ort_session = onnxruntime.InferenceSession(
        args.onnx_path, providers=["CUDAExecutionProvider"]
    )

    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    flow_pred = ort_outs[0][0].transpose(1, 2, 0)
    print(flow_pred.shape, flow_pred.min(), flow_pred.max())

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    flo_output_path = output_dir / f"flow_pred.flo"
    flow_utils.flow_write(flo_output_path, flow_pred)
    print(f"Saved flow prediction to: {flo_output_path}")

    viz_output_path = output_dir / f"flow_pred_viz.png"
    flow_viz = flow_utils.flow_to_rgb(flow_pred)
    cv.imwrite(str(viz_output_path), cv.cvtColor(flow_viz, cv.COLOR_RGB2BGR))
    print(f"Saved flow prediction visualization to: {viz_output_path}")
