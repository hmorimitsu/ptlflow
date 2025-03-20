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

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2 as cv
from jsonargparse import ArgumentParser, Namespace
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import ptlflow
from ptlflow import get_model
from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.registry import RegisteredModel
from ptlflow.utils.utils import tensor_dict_to_numpy


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, run validation on all available models.",
    )
    parser.add_argument(
        "--select",
        type=str,
        nargs="+",
        default=None,
        help=("Used to provide a list of model names to be validated."),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used in combination with --all. A list of model names that will not be validated."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=("Path to a ckpt file for the chosen model."),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("outputs/validate")),
        help="Path to the directory where the validation results will be saved.",
    )
    parser.add_argument(
        "--write_outputs",
        action="store_true",
        help="If set, the estimated flow is saved to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, the results are shown on the screen.",
    )
    parser.add_argument(
        "--flow_format",
        type=str,
        default="original",
        choices=["flo", "png", "original"],
        help=(
            "The format to use when saving the estimated optical flow. If 'original', then the format will be the same "
            + "one the dataset uses for the groundtruth."
        ),
    )
    parser.add_argument(
        "--max_forward_side",
        type=int,
        default=None,
        help=(
            "If max(height, width) of the input image is larger than this value, then the image is downscaled "
            "before the forward and the outputs are bilinearly upscaled to the original resolution."
        ),
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
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples per dataset will be used for calculating the metrics."
        ),
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="To be combined with model all or select. Iterates over the list of models in reversed order",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If set, use half floating point precision."
    )
    parser.add_argument(
        "--seq_val_mode",
        type=str,
        default="all",
        choices=("all", "first", "middle", "last"),
        help=(
            "Used only when the model predicts outputs for more than one frame. Select which predictions will be used for evaluation."
        ),
    )
    parser.add_argument(
        "--write_individual_metrics",
        action="store_true",
        help="If set, save a table of metrics for every image.",
    )
    parser.add_argument(
        "--epe_clip",
        type=float,
        default=5.0,
        help=("Maximum EPE value to before clipping. Used for EPE visualization."),
    )
    parser.add_argument(
        "--metric_exclude",
        type=str,
        default=None,
        nargs="+",
        help=("Names of metrics to not be included in the saved results."),
    )
    return parser


def generate_outputs(
    args: Namespace,
    inputs: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
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
    inputs["flows_viz"] = flow_utils.flow_to_rgb(inputs["flows"])[:, :, ::-1]
    if inputs.get("flows_b") is not None:
        inputs["flows_b_viz"] = flow_utils.flow_to_rgb(inputs["flows_b"])[:, :, ::-1]
    preds = tensor_dict_to_numpy(preds)
    preds["flows_viz"] = flow_utils.flow_to_rgb(preds["flows"])[:, :, ::-1]
    if preds.get("flows_b") is not None:
        preds["flows_b_viz"] = flow_utils.flow_to_rgb(preds["flows_b"])[:, :, ::-1]
    epe = np.sqrt(np.square(preds["flows"] - inputs["flows"]).sum(-1))
    epe = np.clip(epe, 0, args.epe_clip)
    epe_img = ((255.0 / args.epe_clip) * epe).astype(np.uint8)
    epe_img = cv.applyColorMap(epe_img, cv.COLORMAP_CIVIDIS)
    invalid_mask = inputs["valids"] < 0.5
    invalid_mask = np.concatenate([invalid_mask, invalid_mask, invalid_mask], -1)
    epe_img[invalid_mask] = 0
    preds["epe"] = epe_img

    if args.show:
        _show(inputs, preds, args.max_show_side)

    if args.write_outputs:
        _write_to_file(args, preds, dataloader_name, batch_idx, metadata)


def validate(
    args: Namespace, model: BaseModel, data_module: FlowDataModule
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
        if args.fp16:
            model = model.half()

    if args.scale_factor is not None and args.scale_factor != 1.0:
        model.metric_interpolate_pred_to_target_size = True

    data_module.setup("validate")
    dataloaders = data_module.val_dataloader()
    dataloaders = {
        data_module.val_dataloader_names[i]: dataloaders[i]
        for i in range(len(dataloaders))
    }

    metrics_df = pd.DataFrame()
    metrics_df["model"] = [args.model_name]
    metrics_df["checkpoint"] = [args.ckpt_path]

    if args.write_outputs:
        logger.info("Outputs will be saved to {}.", args.output_path)

    output_path = Path(args.output_path)
    for i, (dataset_name, dl) in enumerate(dataloaders.items()):
        metrics_mean = validate_one_dataloader(args, model, dl, i, dataset_name)
        metrics_df[[f"{dataset_name}-{k}" for k in metrics_mean.keys()]] = list(
            metrics_mean.values()
        )

        output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.T.to_csv(output_path / "metrics.csv", header=False)
    metrics_df = metrics_df.round(3)
    return metrics_df


def validate_list_of_models(args: Namespace, data_module: FlowDataModule) -> None:
    """Perform the validation.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the list of models and the validation.
    """
    metrics_df = pd.DataFrame()

    model_names = _get_model_names(args)
    if args.reversed:
        model_names = reversed(model_names)

    exclude = args.exclude
    if exclude is None:
        exclude = []
    else:
        available_model_names = ptlflow.get_model_names()
        for name in exclude:
            assert name in available_model_names

    for mname in model_names:
        if mname in exclude:
            continue

        logger.info("Model: {}", mname)
        model_ref = ptlflow.get_model_reference(mname)

        ckpt_names = []
        if args.ckpt_path is None and hasattr(model_ref, "pretrained_checkpoints"):
            ckpt_names = model_ref.pretrained_checkpoints.keys()
        elif args.ckpt_path is not None:
            ckpt_names = [args.ckpt_path]

        for cname in ckpt_names:
            try:
                logger.info("Checkpoint: {}", cname)

                model_id = f"{mname}_{cname}"
                output_path = Path(args.output_path) / model_id

                local_args = deepcopy(args)
                local_args.model_name = mname
                local_args.ckpt_path = cname
                local_args.output_path = str(output_path)

                model = get_model(mname, cname)
                instance_metrics_df = validate(local_args, model, data_module)
                metrics_df = pd.concat([metrics_df, instance_metrics_df])
                output_path.parent.mkdir(parents=True, exist_ok=True)

                file_name = "metrics"
                if args.all:
                    file_name += "_all"
                else:
                    file_name += "_select"

                if local_args.reversed:
                    file_name += "_rev"

                metrics_path = output_path.parent / f"{file_name}.csv"
                metrics_df.to_csv(metrics_path, index=False)
                logger.info("Saved metrics to {}", metrics_path)

            except Exception as e:  # noqa: B902
                logger.warning(
                    "Skipping model {} with ckpt {} due to exception {}",
                    mname,
                    cname,
                    e,
                )
                break


@torch.no_grad()
def validate_one_dataloader(
    args: Namespace,
    model: BaseModel,
    dataloader: DataLoader,
    dataloader_idx: int,
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
    dataloader_idx : index
        The index of this dataloader.
    dataloader_name : str
        A string to identify this dataloader.

    Returns
    -------
    Dict[str, float]
        The average metric values for this dataloader.
    """
    metrics_sum = {}

    metrics_individual = None
    if args.write_individual_metrics:
        metrics_individual = {
            "filename": [],
            "epe": [],
            "flall": [],
            "wauc": [],
            "px1": [],
        }

    with tqdm(dataloader) as tdl:
        for i, inputs in enumerate(tdl):
            if args.scale_factor is not None:
                scale_factor = args.scale_factor
            else:
                scale_factor = (
                    None
                    if args.max_forward_side is None
                    else float(args.max_forward_side) / max(inputs["images"].shape[-2:])
                )

            io_adapter = IOAdapter(
                output_stride=model.output_stride,
                input_size=inputs["images"].shape[-2:],
                target_scale_factor=scale_factor,
                cuda=torch.cuda.is_available(),
                fp16=args.fp16,
            )
            inputs = io_adapter.prepare_inputs(inputs=inputs, image_only=True)

            outputs = model.validation_step(inputs, i, dataloader_idx)

            inputs = io_adapter.unscale(inputs, image_only=True)
            preds = outputs["preds"]
            preds = io_adapter.unscale(preds)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and args.fp16:
                    inputs[k] = v.float()
            for k, v in preds.items():
                if isinstance(v, torch.Tensor) and args.fp16:
                    preds[k] = v.float()

            if inputs["flows"].shape[1] > 1 and args.seq_val_mode != "all":
                if args.seq_val_mode == "first":
                    k = 0
                elif args.seq_val_mode == "middle":
                    k = inputs["images"].shape[1] // 2
                elif args.seq_val_mode == "last":
                    k = inputs["flows"].shape[1] - 1
                for key, val in inputs.items():
                    if key == "meta":
                        inputs["meta"]["image_paths"] = inputs["meta"]["image_paths"][
                            k : k + 1
                        ]
                    elif key == "images":
                        inputs[key] = val[:, k : k + 2]
                    elif isinstance(val, torch.Tensor) and len(val.shape) == 5:
                        inputs[key] = val[:, k : k + 1]

            metrics = outputs["metrics"]
            for k in metrics.keys():
                if metrics_sum.get(k) is None:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += metrics[k].item()
            progress_bar_values = {
                "epe": metrics_sum["val/epe"] / (i + 1),
                "flall": metrics_sum["val/flall"] / (i + 1),
                "wauc": metrics_sum["val/wauc"] / (i + 1),
                "px1": 100 * (((i + 1) - metrics_sum["val/px1"]) / (i + 1)),
            }
            tdl.set_postfix(**progress_bar_values)

            filename = ""
            dataloader_suffix = ""
            if "sintel" in inputs["meta"]["dataset_name"][0].lower():
                filename = f'{Path(inputs["meta"]["image_paths"][0][0]).parent.name}/'
            elif "spring" in inputs["meta"]["dataset_name"][0].lower():
                filename = (
                    f'{Path(inputs["meta"]["image_paths"][0][0]).parent.parent.name}/'
                )
            elif "kubric" in inputs["meta"]["dataset_name"][0].lower():
                filename = f'{Path(inputs["meta"]["image_paths"][0][0]).parent.name}/'
                dataloader_suffix = (
                    f'_{Path(inputs["meta"]["image_paths"][0][0]).parent.parent.name}'
                )
            filename += Path(inputs["meta"]["image_paths"][0][0]).stem

            if metrics_individual is not None:
                metrics_individual["filename"].append(filename)
                metrics_individual["epe"].append(metrics["val/epe"].item())
                metrics_individual["flall"].append(metrics["val/flall"].item())
                metrics_individual["wauc"].append(metrics["val/wauc"].item())
                metrics_individual["px1"].append(metrics["val/px1"].item())

            generate_outputs(
                args, inputs, preds, dataloader_name, i, inputs.get("meta")
            )

            if args.max_samples is not None and i >= (args.max_samples - 1):
                break

    if args.write_individual_metrics:
        ind_df = pd.DataFrame(metrics_individual)
        args.output_path.mkdir(parents=True, exist_ok=True)
        csv_path = (
            Path(args.output_path)
            / f"{dataloader_name}{dataloader_suffix}_epe_flall.csv"
        )
        ind_df.to_csv(
            csv_path,
            index=None,
        )
        logger.info("Saved individual metrics to: {}", csv_path)

    metrics_mean = {}
    for k, v in metrics_sum.items():
        is_exclude = False
        if args.metric_exclude is not None:
            for ex_metric in args.metric_exclude:
                if ex_metric in k:
                    is_exclude = True
                    break
        if not is_exclude:
            metrics_mean[k] = v / len(dataloader)
    return metrics_mean


def _get_model_names(args: Namespace) -> List[str]:
    available_model_names = ptlflow.get_model_names()
    if args.all:
        model_names = available_model_names
    else:
        assert len(args.select) > 0
        model_names = args.select
        for name in model_names:
            assert name in available_model_names
    return model_names


def _show(
    inputs: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor], max_show_side: int
) -> None:
    for k, v in inputs.items():
        if isinstance(v, np.ndarray) and (
            len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3
        ):
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(
                    v, (int(scale_factor * v.shape[1]), int(scale_factor * v.shape[0]))
                )
            cv.imshow(k, v)
    for k, v in preds.items():
        if isinstance(v, np.ndarray) and (
            len(v.shape) == 2 or v.shape[2] == 1 or v.shape[2] == 3
        ):
            if max(v.shape[:2]) > max_show_side:
                scale_factor = float(max_show_side) / max(v.shape[:2])
                v = cv.resize(
                    v, (int(scale_factor * v.shape[1]), int(scale_factor * v.shape[0]))
                )
            cv.imshow("pred_" + k, v)
    cv.waitKey(1)


def _write_to_file(
    args: Namespace,
    preds: Dict[str, torch.Tensor],
    dataloader_name: str,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    out_root_dir = Path(args.output_path) / dataloader_name

    extra_dirs = ""
    if metadata is not None:
        img_path = Path(metadata["image_paths"][0][0])
        image_name = img_path.stem
        if (
            "sintel" in dataloader_name
            or "middlebury_st" in dataloader_name
            or "kubric" in dataloader_name
        ):
            seq_name = img_path.parts[-2]
            extra_dirs = seq_name
        elif "spring" in dataloader_name:
            seq_name = img_path.parts[-3]
            extra_dirs = seq_name
    else:
        image_name = f"{batch_idx:08d}"

    if args.flow_format != "original":
        flow_ext = args.flow_format
    else:
        if "kitti" in dataloader_name or "hd1k" in dataloader_name:
            flow_ext = "png"
        else:
            flow_ext = "flo"

    for k, v in preds.items():
        if isinstance(v, np.ndarray):
            out_dir = out_root_dir / k / extra_dirs
            out_dir.mkdir(parents=True, exist_ok=True)
            if k == "flows" or k == "flows_b":
                flow_utils.flow_write(out_dir / f"{image_name}.{flow_ext}", v)
            elif len(v.shape) == 2 or (
                len(v.shape) == 3 and (v.shape[2] == 1 or v.shape[2] == 3)
            ):
                if v.max() <= 1:
                    v = v * 255
                cv.imwrite(str(out_dir / f"{image_name}.png"), v.astype(np.uint8))


def _show_v04_warning():
    ignore_args = ["-h", "--help", "--model", "--config", "--all", "--select"]
    for arg in ignore_args:
        if arg in sys.argv:
            return

    logger.warning(
        "Since v0.4, it is now necessary to inform the model using the --model argument. For example, use: python infer.py --model raft --ckpt_path things"
    )


if __name__ == "__main__":
    _show_v04_warning()

    parser = _init_parser()

    is_validate_list = False
    if "--config" in sys.argv:
        config_file_idx = sys.argv.index("--config") + 1
        with open(sys.argv[config_file_idx], "r") as f:
            config = yaml.safe_load(f)
        if config["all"] or config["select"] is not None:
            is_validate_list = True

    if "--all" in sys.argv or "--select" in sys.argv:
        is_validate_list = True

    if is_validate_list:
        model_class = None
        subclass_mode_model = False
    else:
        model_class = RegisteredModel
        subclass_mode_model = True

    cli = PTLFlowCLI(
        model_class=model_class,
        subclass_mode_model=subclass_mode_model,
        datamodule_class=FlowDataModule,
        parser_kwargs={"parents": [parser]},
        run=False,
        parse_only=False,
        auto_configure_optimizers=False,
    )

    if is_validate_list:
        validate_list_of_models(cli.config, cli.datamodule)
    else:
        cfg = cli.config
        cfg.model_name = cfg.model.class_path.split(".")[-1]
        model_id = cfg.model_name
        if cfg.ckpt_path is not None:
            model_id += f"_{Path(cfg.ckpt_path).stem}"
        if cfg.max_forward_side is not None:
            model_id += f"_maxside{cfg.max_forward_side}"
        if cfg.scale_factor is not None:
            model_id += f"_scale{cfg.scale_factor}"
        cfg.output_path = str(Path(cfg.output_path) / model_id)
        Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

        model = cli.model
        model = ptlflow.restore_model(model, cfg.ckpt_path)

        validate(cfg, model, cli.datamodule)
