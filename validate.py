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

import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ptlflow
from ptlflow import get_model, get_model_reference
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.utils import (
    add_datasets_to_parser, config_logging, get_list_of_available_models_list, tensor_dict_to_numpy)

config_logging()


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'model', type=str, choices=['all', 'select']+get_list_of_available_models_list(),
        help='Name of the model to use.')
    parser.add_argument(
        '--selection', type=str, nargs='+', default=None,
        help=('Used in combination with model=select. The select mode can be used to run the validation on multiple models '
              'at once. Put a list of model names here separated by spaces.'))
    parser.add_argument(
        '--output_path', type=str, default=str(Path('outputs/validate')),
        help='Path to the directory where the validation results will be saved.')
    parser.add_argument(
        '--write_outputs', action='store_true',
        help='If set, the estimated flow is saved to disk.')
    parser.add_argument(
        '--show', action='store_true',
        help='If set, the results are shown on the screen.')
    parser.add_argument(
        '--flow_format', type=str, default='original', choices=['flo', 'png', 'original'],
        help=('The format to use when saving the estimated optical flow. If \'original\', then the format will be the same '
              + 'one the dataset uses for the groundtruth.'))
    parser.add_argument(
        '--max_forward_side', type=int, default=None,
        help=('If max(height, width) of the input image is larger than this value, then the image is downscaled '
              'before the forward and the outputs are bilinearly upscaled to the original resolution.'))
    parser.add_argument(
        '--max_show_side', type=int, default=1000,
        help=('If max(height, width) of the output image is larger than this value, then the image is downscaled '
              'before showing it on the screen.'))
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help=('Maximum number of samples per dataset will be used for calculating the metrics.'))
    return parser


def generate_outputs(
    args: Namespace,
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Display on screen and/or save outputs to disk, if required.

    Parameters
    ----------
    args : Namespace
        The arguments with the required values to manage the outputs.
    inputs : Dict[str, torch.Tensor]
        The inputs loaded from the dataset (images, groundtruth).
    preds : Dict[str, torch.Tensor]
        The model predictions (optical flow and others).
    dataloader_name : str
        A string to identify from which dataloader these inputs came from.
    batch_idx : int
        Indicates in which position of the loader this input is.
    metadata : Dict[str, Any], optional
        Metadata about this input, if available.
    """
    inputs = tensor_dict_to_numpy(inputs)
    inputs['flows_viz'] = flow_utils.flow_to_rgb(inputs['flows'])[:, :, ::-1]
    if inputs.get('flows_b') is not None:
        inputs['flows_b_viz'] = flow_utils.flow_to_rgb(inputs['flows_b'])[:, :, ::-1]
    preds = tensor_dict_to_numpy(preds)
    preds['flows_viz'] = flow_utils.flow_to_rgb(preds['flows'])[:, :, ::-1]
    if preds.get('flows_b') is not None:
        preds['flows_b_viz'] = flow_utils.flow_to_rgb(preds['flows_b'])[:, :, ::-1]

    if args.show:
        _show(inputs, preds, args.max_show_side)

    if args.write_outputs:
        _write_to_file(args, preds, dataloader_name, batch_idx, metadata)


def validate(
    args: Namespace,
    model: BaseModel
) -> pd.DataFrame:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the metric results.

    See Also
    --------
    ptlflow.models.base_model.base_model.BaseModel : The parent class of the available models.
    """
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataloaders = model.val_dataloader()
    dataloaders = {model.val_dataloader_names[i]: dataloaders[i] for i in range(len(dataloaders))}

    metrics_df = pd.DataFrame()
    metrics_df['model'] = [args.model]
    metrics_df['checkpoint'] = [args.pretrained_ckpt]

    for dataset_name, dl in dataloaders.items():
        metrics_mean = validate_one_dataloader(args, model, dl, dataset_name)
        metrics_df[[f'{dataset_name}-{k}' for k in metrics_mean.keys()]] = list(metrics_mean.values())
        args.output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.T.to_csv(args.output_path / 'metrics.csv', header=False)
    metrics_df = metrics_df.round(3)
    return metrics_df


@torch.no_grad()
def validate_one_dataloader(
    args: Namespace,
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_name: str,
) -> Dict[str, float]:
    """Perform validation for all examples of one dataloader.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the model and the validation.
    model : BaseModel
        The model to be used for validation.
    dataloader : DataLoader
        The dataloader for the validation.
    dataloader_name : str
        A string to identify this dataloader.

    Returns
    -------
    Dict[str, float]
        The average metric values for this dataloader.
    """
    metrics_sum = {}
    with tqdm(dataloader) as tdl:
        for i, inputs in enumerate(tdl):
            scale_factor = (
                None if args.max_forward_side is None else float(args.max_forward_side) / min(inputs['images'].shape[-2:]))

            io_adapter = IOAdapter(
                model, inputs['images'].shape[-2:], target_scale_factor=scale_factor, cuda=torch.cuda.is_available())
            inputs = io_adapter.prepare_inputs(inputs=inputs)

            preds = model(inputs)

            inputs = io_adapter.unpad_and_unscale(inputs)
            preds = io_adapter.unpad_and_unscale(preds)

            metrics = model.val_metrics(preds, inputs)

            for k in metrics.keys():
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()
            tdl.set_postfix(epe=metrics_sum['val/epe']/(i+1))

            generate_outputs(args, inputs, preds, dataloader_name, i, inputs.get('meta'))

            if args.max_samples is not None and i >= (args.max_samples - 1):
                break

    metrics_mean = {}
    for k, v in metrics_sum.items():
        metrics_mean[k] = v / len(dataloader)
    return metrics_mean


def _get_model_names(
    args: Namespace
) -> List[str]:
    if args.model == 'all':
        model_names = ptlflow.models_dict.keys()
    elif args.model == 'select':
        if args.selection is None:
            raise ValueError('When select is chosen, model names must be provided to --selection.')
        model_names = args.selection
    return model_names


def _show(
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    max_show_side: int
) -> None:
    for k, v in inputs.items():
        if isinstance(v, np.ndarray) and (len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3):
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(v, (int(scale_factor*v.shape[1]), int(scale_factor*v.shape[0])))
            cv.imshow(k, v)
    for k, v in preds.items():
        if isinstance(v, np.ndarray) and (len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3):
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(v, (int(scale_factor*v.shape[1]), int(scale_factor*v.shape[0])))
            cv.imshow('pred_'+k, v)
    cv.waitKey(1)


def _write_to_file(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    out_root_dir = Path(args.output_path) / dataloader_name

    extra_dirs = ''
    if metadata is not None:
        img_path = Path(metadata['image_paths'][0][0])
        image_name = img_path.stem
        if 'sintel' in dataloader_name:
            seq_name = img_path.parts[-2]
            extra_dirs = seq_name
    else:
        image_name = f'{batch_idx:08d}'

    if args.flow_format != 'original':
        flow_ext = args.flow_format
    else:
        if 'kitti' in dataloader_name or 'hd1k' in dataloader_name:
            flow_ext = 'png'
        else:
            flow_ext = 'flo'

    for k, v in preds.items():
        if isinstance(v, np.ndarray):
            out_dir = out_root_dir / k / extra_dirs
            out_dir.mkdir(parents=True, exist_ok=True)
            if k == 'flows' or k == 'flows_b':
                flow_utils.flow_write(out_dir / f'{image_name}.{flow_ext}', v)
            elif len(v.shape) == 2 or (len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3)):
                if v.max() <= 1:
                    v = v * 255
                cv.imwrite(str(out_dir / f'{image_name}.png'), v.astype(np.uint8))


if __name__ == '__main__':
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help', 'all', 'select']:
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    add_datasets_to_parser(parser, 'datasets.yml')

    args = parser.parse_args()

    if args.model not in ['all', 'select']:
        model_id = args.model
        if args.pretrained_ckpt is not None:
            model_id += f'_{args.pretrained_ckpt}'
        args.output_path = Path(args.output_path) / model_id
        model = get_model(sys.argv[1], args.pretrained_ckpt, args)
        args.output_path.mkdir(parents=True, exist_ok=True)

        metrics_df = validate(args, model)
    else:
        # Run validation on all models and checkpoints
        metrics_df = pd.DataFrame()

        model_names = _get_model_names(args)

        for mname in model_names:
            logging.info(mname)
            model_ref = ptlflow.get_model_reference(mname)

            if hasattr(model_ref, 'pretrained_checkpoints'):
                ckpt_names = model_ref.pretrained_checkpoints.keys()
                for cname in ckpt_names:
                    logging.info(cname)
                    parser_tmp = model_ref.add_model_specific_args(parser)
                    args = parser_tmp.parse_args()

                    args.model = mname
                    args.pretrained_ckpt = cname

                    model_id = args.model
                    if args.pretrained_ckpt is not None:
                        model_id += f'_{args.pretrained_ckpt}'
                    args.output_path = Path(args.output_path) / model_id

                    model = get_model(mname, cname, args)
                    instance_metrics_df = validate(args, model)
                    metrics_df = pd.concat([metrics_df, instance_metrics_df])
                    args.output_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_df.to_csv(args.output_path.parent / 'metrics_all.csv', index=False)
