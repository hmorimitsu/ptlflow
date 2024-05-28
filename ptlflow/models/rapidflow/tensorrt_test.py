# TensorRT conversion code comes from the tutorial:
# https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_resnet_example.html


import sys
from argparse import ArgumentParser
from pathlib import Path
import time

import cv2 as cv
import numpy as np
import torch
import torch_tensorrt

this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(this_dir.parent.parent.parent))

from ptlflow import get_model, load_checkpoint
from ptlflow.models.rapidflow.rapidflow import RAPIDFlow
from ptlflow.utils import flow_utils


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
        "--image_paths",
        type=str,
        nargs=2,
        default=(
            str(this_dir / "image_samples" / "000000_10.png"),
            str(this_dir / "image_samples" / "000000_11.png"),
        ),
        help="Path to two images to estimate the optical flow with the TensorRT model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path to the directory where the predictions will be saved.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=(384, 1280),
        help="Size of the input image.",
    )
    return parser


def compile_engine_and_infer(args):
    # Initialize model with half precision and sample inputs
    model = load_model(args).half().eval().to("cuda")
    images = [torch.from_numpy(load_images(args.image_paths)).half().to("cuda")]

    num_tries = 11
    total_time_orig = 0.0
    for i in range(num_tries):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(images[0])
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i > 0:
            total_time_orig += end - start

    # Enabled precision for TensorRT optimization
    enabled_precisions = {torch.half}

    # Whether to print verbose logs
    debug = True

    # Workspace size for TensorRT
    workspace_size = 20 << 30

    # Maximum number of TRT Engines
    # (Lower value allows more graph segmentation)
    min_block_size = 7

    # Operations to Run in Torch, regardless of converter support
    torch_executed_ops = {}

    # Build and compile the model with torch.compile, using Torch-TensorRT backend
    compiled_model = torch_tensorrt.compile(
        model,
        ir="torch_compile",
        inputs=images,
        enabled_precisions=enabled_precisions,
        debug=debug,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
    )

    total_time_optimized = 0.0
    for i in range(num_tries):
        torch.cuda.synchronize()
        start = time.perf_counter()
        flow_pred = compiled_model(*images)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i > 0:
            total_time_optimized += end - start

    try:
        torch_tensorrt.save(compiled_model, f"{args.model}.tc", inputs=images)
        print(f"Saving compiled model to {args.model}.tc")
        compiled_model = torch_tensorrt.load(f"{args.model}.tc")
        print(f"Loading compiled model from {args.model}.tc")
    except Exception as e:
        print("WARNING: The compiled model was not saved due to the error:")
        print(e)

    print(f"Model: {args.model}. Average time of {num_tries - 1} runs:")
    print(f"Time (original): {(1000 * total_time_orig / (num_tries - 1)):.2f} ms.")
    print(f"Time (compiled): {(1000 * total_time_optimized / (num_tries - 1)):.2f} ms.")

    flow_pred_npy = flow_pred[0].permute(1, 2, 0).detach().cpu().numpy()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    flo_output_path = output_dir / f"flow_pred.flo"
    flow_utils.flow_write(flo_output_path, flow_pred_npy)
    print(f"Saved flow prediction to: {flo_output_path}")

    viz_output_path = output_dir / f"flow_pred_viz.png"
    flow_viz = flow_utils.flow_to_rgb(flow_pred_npy)
    cv.imwrite(str(viz_output_path), cv.cvtColor(flow_viz, cv.COLOR_RGB2BGR))
    print(f"Saved flow prediction visualization to: {viz_output_path}")

    # Finally, we use Torch utilities to clean up the workspace
    torch._dynamo.reset()


def load_images(image_paths):
    images = [cv.imread(p) for p in image_paths]
    images = [cv.resize(im, args.input_size[::-1]) for im in images]
    images = np.stack(images)
    images = images.transpose(0, 3, 1, 2)[None]
    images = images.astype(np.float32) / 255.0
    return images


def load_model(args):
    model = get_model(args.model, args=args)
    ckpt = load_checkpoint(args.checkpoint, RAPIDFlow, "rapidflow")
    state_dict = fuse_checkpoint_next1d_layers(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    return model


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


if __name__ == "__main__":
    parser = _init_parser()
    parser = RAPIDFlow.add_model_specific_args(parser)
    args = parser.parse_args()
    args.corr_mode = "allpairs"
    args.fuse_next1d_weights = True
    args.simple_io = True

    compile_engine_and_infer(args)
