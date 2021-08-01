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
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from ptlflow import get_model, get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.flow_utils import flow_to_rgb, flow_write
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import get_list_of_available_models_list, tensor_dict_to_numpy


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'model', type=str, choices=get_list_of_available_models_list(),
        help='Name of the model to use.')
    parser.add_argument(
        '--input_path', type=str, nargs='+', required=True,
        help=('Path to the inputs. It can be in any of these formats: 1. list of paths of images; 2. path to a folder '
              + 'containing images; 3. path to a video; 4. the index of a webcam.'))
    parser.add_argument(
        '--write_outputs', action='store_true',
        help='If set, the model outputs are saved to disk.')
    parser.add_argument(
        '--output_path', type=str, default=str(Path('outputs/inference')),
        help='Path to a folder where the results will be saved.')
    parser.add_argument(
        '--flow_format', type=str, default='flo', choices=['flo', 'png'],
        help='The format to use when saving the estimated optical flow.')
    parser.add_argument(
        '--show', action='store_true',
        help='If set, the results are shown on the screen.')
    parser.add_argument(
        '--auto_forward', action='store_true',
        help=('Only relevant if used with --show. If set, consecutive results will be shown without stopping. '
              + 'Otherwise, each result remain on the screen until the user press a button.'))
    parser.add_argument(
        '--input_size', type=int, nargs=2, default=[0, 0],
        help='If larger than zero, resize the input image before forwarding.')
    parser.add_argument(
        '--max_show_side', type=int, default=1000,
        help=('If max(height, width) of the output image is larger than this value, then the image is downscaled '
              'before showing it on the screen.'))
    return parser


@torch.no_grad()
def infer(
    args: Namespace,
    model: BaseModel
) -> None:
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

    cap, img_paths, num_imgs, prev_img = init_input(args.input_path)

    io_adapter = IOAdapter(model, prev_img.shape[:2], args.input_size, cuda=torch.cuda.is_available())

    for i in tqdm(range(1, num_imgs)):
        img, img_name, is_img_valid = _read_image(cap, img_paths, i)

        if not is_img_valid:
            break

        inputs = io_adapter.prepare_inputs([prev_img, img])
        preds = model(inputs)

        preds = io_adapter.unpad_and_unscale(preds)
        preds_npy = tensor_dict_to_numpy(preds)
        preds_npy['flows_viz'] = flow_to_rgb(preds_npy['flows'])[:, :, ::-1]
        if preds_npy.get('flows_b') is not None:
            preds_npy['flows_b_viz'] = flow_to_rgb(preds_npy['flows_b'])[:, :, ::-1]
        if args.write_outputs:
            write_outputs(preds_npy, args.output_path, img_name, args.flow_format)
        if args.show:
            img1 = prev_img
            img2 = img
            if min(args.input_size) > 0:
                img1 = cv.resize(prev_img, args.input_size[::-1])
                img2 = cv.resize(img, args.input_size[::-1])
            key = show_outputs(
                img1, img2, preds_npy, args.auto_forward, args.max_show_side)
            if key == 27:
                break
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
            img_paths = sorted(input_path.glob('*'))
        else:
            # Assumes it is a video or webcam index
            try:
                inp = int(input_path)
            except ValueError:
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
    max_show_side: int
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
    preds_npy['img1'] = img1
    preds_npy['img2'] = img2
    for k, v in preds_npy.items():
        if len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3:
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(v, (int(scale_factor*v.shape[1]), int(scale_factor*v.shape[0])))
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
    flow_format: str
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
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_name
        if k == 'flows' or k == 'flows_b':
            if flow_format[0] != '.':
                flow_format = '.' + flow_format
            flow_write(out_path.with_suffix(flow_format), v)
        elif len(v.shape) == 2 or (len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3)):
            if v.max() <= 1:
                v = v * 255
            cv.imwrite(str(out_path.with_suffix('.png')), v.astype(np.uint8))


def _read_image(
    cap: cv.VideoCapture,
    img_paths: List[Union[str, Path]],
    i: int
) -> Tuple[np.ndarray, str, bool]:
    if cap is not None:
        is_img_valid, img = cap.read()
        img_name = '{:08d}'.format(i)
    else:
        img = cv.imread(str(img_paths[i]))
        img_name = img_paths[i-1].stem
        is_img_valid = True
    return img, img_name, is_img_valid


if __name__ == '__main__':
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] != '-h' and sys.argv[1] != '--help':
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    args = parser.parse_args()

    model = get_model(sys.argv[1], args.pretrained_ckpt, args)

    infer(args, model)
