"""

Generate optical flow with one of the available models.

This script can display and save optical flow estimated by any of the available models. It accepts multiple types of inputs,
including: individual images, a folder of images, a video, or a webcam stream.

"""

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
from jsonargparse import ArgumentParser, Namespace
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

import ptlflow
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.flow_utils import flow_to_rgb, flow_write, flow_read
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel
from ptlflow.utils.utils import tensor_dict_to_numpy


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=("Path to a ckpt file for the chosen model."),
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Path to the inputs. It can be in any of these formats: 1. list of paths of images; 2. path to a folder "
            + "containing images; 3. path to a video; 4. the index of a webcam."
        ),
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help=(
            "(Optional) Path to the flow groundtruth. The path must point to one file, and --input_path must be composed of paths to two images only."
        ),
    )
    parser.add_argument(
        "--not_write_outputs",
        action="store_false",
        help="If set, the model outputs are saved to disk.",
        dest="write_outputs",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/inference")),
        help="Path to a folder where the results will be saved.",
    )
    parser.add_argument(
        "--flow_format",
        type=str,
        default="flo",
        choices=["flo", "png"],
        help="The format to use when saving the estimated optical flow.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, the results are shown on the screen.",
    )
    parser.add_argument(
        "--auto_forward",
        action="store_true",
        help=(
            "Only relevant if used with --show. If set, consecutive results will be shown without stopping. "
            + "Otherwise, each result remain on the screen until the user press a button."
        ),
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[0, 0],
        help="If larger than zero, resize the input image before forwarding.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help=("Multiply the input image by this scale factor before forwarding."),
    )
    parser.add_argument(
        "--max_show_side",
        type=int,
        default=1000,
        help=(
            "If max(height, width) of the output image is larger than this value, then the image is downscaled "
            "before showing it on the screen."
        ),
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    return parser


@torch.no_grad()
def infer(args: Namespace, model: BaseModel) -> None:
    """Perform the inference.

    Parameters
    ----------
    model : BaseModel
        The model to be used for inference.
    args : Namespace
        Arguments to configure the model and the inference.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        if args.fp16:
            model = model.half()

    cap, img_paths, num_imgs, prev_img = init_input(args.input_path)
    flow_gt = None
    if args.gt_path is not None:
        assert num_imgs == 2
        flow_gt = flow_read(args.gt_path)

    if args.scale_factor is not None:
        io_adapter = IOAdapter(
            output_stride=model.output_stride,
            input_size=prev_img.shape[:2],
            target_scale_factor=args.scale_factor,
            cuda=torch.cuda.is_available(),
            fp16=args.fp16,
        )
    else:
        io_adapter = IOAdapter(
            output_stride=model.output_stride,
            input_size=prev_img.shape[:2],
            target_size=args.input_size,
            cuda=torch.cuda.is_available(),
            fp16=args.fp16,
        )

    prev_dir_name = None
    for i in tqdm(range(1, num_imgs)):
        img, img_dir_name, img_name, is_img_valid = _read_image(cap, img_paths, i)
        if prev_dir_name is None:
            prev_dir_name = img_dir_name

        if not is_img_valid:
            break

        if img_dir_name == prev_dir_name:
            inputs = io_adapter.prepare_inputs([prev_img, img])
            preds = model(inputs)

            preds["images"] = inputs["images"]
            preds = io_adapter.unscale(preds)
            preds_npy = tensor_dict_to_numpy(preds)

            if flow_gt is not None:
                flow_pred = preds_npy["flows"]
                valid = ~np.isnan(flow_gt[..., 0])

                sq_dist = np.power(flow_pred - flow_gt, 2).sum(2)
                epe = np.sqrt(sq_dist[valid])

                gt_sq_dist = np.power(flow_gt, 2).sum(2)
                gt_dist_valid = np.sqrt(gt_sq_dist[valid])
                flall = (epe > 3) & (epe > 0.05 * gt_dist_valid)
                print(
                    f"EPE: {epe.mean():.03f}, Fl-All: {100*flall.mean():.03f}",
                )

            preds_npy["flows_viz"] = flow_to_rgb(preds_npy["flows"])[:, :, ::-1]
            if preds_npy.get("flows_b") is not None:
                preds_npy["flows_b_viz"] = flow_to_rgb(preds_npy["flows_b"])[:, :, ::-1]
            if args.write_outputs:
                write_outputs(
                    preds_npy,
                    args.output_path,
                    img_name,
                    args.flow_format,
                    img_dir_name,
                )
            if args.show:
                img1 = prev_img
                img2 = img
                if min(args.input_size) > 0:
                    img1 = cv.resize(prev_img, args.input_size[::-1])
                    img2 = cv.resize(img, args.input_size[::-1])
                key = show_outputs(
                    img1, img2, preds_npy, args.auto_forward, args.max_show_side
                )
                if key == 27:
                    break
        prev_dir_name = img_dir_name
        prev_img = img


def init_input(
    input_path: Union[str, List[str]]
) -> Tuple[cv.VideoCapture, List[Path], int, np.ndarray]:
    """Initialize the required variable to start loading the inputs.

    This function will detect which type of input_path was given (list of images, folder of images, video, or webcam).
    Then it will establish its length and also get the first frame of the input.

    Parameters
    ----------
    input_path : str
        The path to the input(s).

    Returns
    -------
    tuple[cv.VideoCapture, List[Path], int, np.ndarray]
        The initialized variables
        - a cv.VideoCapture if the input is a video OR
        - a list of paths to the images otherwise,
        - the maximum number of images, and
        - the first image.
    """
    cap = None
    img_paths = None
    if len(input_path) > 1:
        # Assumes it is a list of images
        img_paths = [Path(p) for p in input_path]
    else:
        input_path = Path(input_path[0])
        if input_path.is_dir():
            # Assumes it is a folder of images
            img_paths = sorted([p for p in input_path.glob("**/*") if not p.is_dir()])
        else:
            inp = str(input_path)
            try:
                inp = int(inp)
            except (ValueError, TypeError):
                pass
            cap = cv.VideoCapture(inp)

    if img_paths is not None:
        num_imgs = len(img_paths)
    else:
        # cv.VideoCapture does not always know the correct number of frames,
        # so we just set it as a high value
        num_imgs = 9999999

    if cap is not None:
        prev_img = cap.read()[1]
    else:
        prev_img = cv.imread(str(img_paths[0]))

    return cap, img_paths, num_imgs, prev_img


def show_outputs(
    img1: np.ndarray,
    img2: np.ndarray,
    preds_npy: Dict[str, np.ndarray],
    auto_forward: bool,
    max_show_side: int,
) -> int:
    """Show the images on the screen.

    Parameters
    ----------
    img1 : np.ndarray
        First image for estimating the optical flow.
    img2 : np.ndarray
        Second image for estimating the optical flow.
    preds_npy : dict[str, np.ndarray]
        The model predictions converted to numpy format.
    auto_forward : bool
        If false, the user needs to press a key to move to the next image.
    max_show_side : int
        If max(height, width) of the image is larger than this value, then it is downscaled before showing.

    Returns
    -------
    int
        A value representing which key the user pressed.

    See Also
    --------
    ptlflow.utils.utils.tensor_dict_to_numpy : This function can generate preds_npy.
    """
    preds_npy["img1"] = img1
    preds_npy["img2"] = img2
    for k, v in preds_npy.items():
        if len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3:
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(
                    v, (int(scale_factor * v.shape[1]), int(scale_factor * v.shape[0]))
                )
            cv.imshow(k, v)

    if auto_forward:
        w = 1
    else:
        w = 0
    key = cv.waitKey(w)
    return key


def write_outputs(
    preds_npy: Dict[str, np.ndarray],
    output_dir: str,
    img_name: str,
    flow_format: str,
    img_dir_name: Optional[str] = None,
) -> None:
    """Show the images on the screen.

    Parameters
    ----------
    preds_npy : dict[str, np.ndarray]
        The model predictions converted to numpy format.
    output_dir : str
        The path to the root dir where the outputs will be saved.
    img_name : str
        The name to be used to save each image (without extension).
    flow_format : str
        The format (extension) of the flow file to be saved. It can one of {flo, png}.

    See Also
    --------
    ptlflow.utils.utils.tensor_dict_to_numpy : This function can generate preds_npy.
    """
    for k, v in preds_npy.items():
        out_dir = Path(output_dir) / k
        if img_dir_name is not None:
            out_dir /= img_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_name
        if k == "flows" or k == "flows_b":
            if flow_format[0] != ".":
                flow_format = "." + flow_format
            flow_write(out_path.with_suffix(flow_format), v)
        elif isinstance(v, np.ndarray) and (
            len(v.shape) == 2
            or (len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3))
        ):
            if v.max() <= 1:
                v = v * 255
            cv.imwrite(str(out_path.with_suffix(".png")), v.astype(np.uint8))


def _read_image(
    cap: cv.VideoCapture, img_paths: List[Union[str, Path]], i: int
) -> Tuple[np.ndarray, str, bool]:
    if cap is not None:
        is_img_valid, img = cap.read()
        img_dir_name = None
        img_name = "{:08d}".format(i)
    else:
        img = cv.imread(str(img_paths[i]))
        img_dir_name = None
        if len(img_paths[i].parent.name) > 0:
            img_dir_name = img_paths[i].parent.name
        img_name = img_paths[i - 1].stem
        is_img_valid = True
    return img, img_dir_name, img_name, is_img_valid


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

    assert (
        cfg.input_path is not None and len(cfg.input_path) > 0
    ), "You need to provide the --input_path argument"

    cfg.model_name = cfg.model.class_path.split(".")[-1]
    model_id = cfg.model_name
    if cfg.ckpt_path is not None:
        model_id += f"_{Path(cfg.ckpt_path).stem}"
    cfg.output_path = str(Path(cfg.output_path) / model_id)

    logger.info("The outputs will be saved to {}.", cfg.output_path)

    model = cli.model
    model = ptlflow.restore_model(model, cfg.ckpt_path)

    infer(cfg, model)
