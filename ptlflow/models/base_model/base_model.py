"""Abstract model to be used as a parent of concrete optical flow networks.

In order to use this module, the concrete model should just need to:
1. Specify the network structure in their __init__() method and call super().__init__() with the required args.
2. Implement a forward method which receives inputs and outputs as specified. See the forward method docs for more details.
3. Optionally, define a loss function, if the model is able to be trained.
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

import logging
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ptlflow.data import flow_transforms as ft
from ptlflow.data.datasets import (
    AutoFlowDataset,
    FlyingChairsDataset,
    FlyingChairs2Dataset,
    Hd1kDataset,
    KittiDataset,
    KubricDataset,
    SintelDataset,
    FlyingThings3DDataset,
    FlyingThings3DSubsetDataset,
    SpringDataset,
    TartanAirDataset,
)
from ptlflow.utils.utils import InputPadder, InputScaler
from ptlflow.utils.utils import config_logging, make_divisible, bgr_val_as_tensor
from ptlflow.utils.flow_metrics import FlowMetrics

config_logging()


class BaseModel(pl.LightningModule):
    """A base abstract optical flow model."""

    def __init__(self, args: Namespace, loss_fn: Callable, output_stride: int) -> None:
        """Initialize BaseModel.

        Parameters
        ----------
        args : Namespace
            A namespace with the required arguments. Typically, this can be gotten from add_model_specific_args().
        loss_fn : Callable
            A function to be used to compute the loss for the training. The input of this function must match the output of the
            forward() method. The output of this function must be a tensor with a single value.
        output_stride : int
            How many times the output of the network is smaller than the input.
        """
        super(BaseModel, self).__init__()

        self.args = args
        self.loss_fn = loss_fn
        self.output_stride = output_stride

        self.train_size = None
        self.train_avg_length = None

        self.extra_params = None

        self.train_metrics = FlowMetrics(prefix="train/")
        self.val_metrics = FlowMetrics(prefix="val/")

        self.train_dataloader_length = 0
        self.train_epoch_step = 0

        self.val_dataloader_names = []
        self.val_dataloader_lengths = []

        self.test_dataloader_names = []

        self.last_inputs = None
        self.last_predictions = None

        if version.parse(pl.__version__) >= version.parse("1.6.0"):
            self.save_hyperparameters(
                ignore=["loss_fn"],
            )
        else:
            self.save_hyperparameters()

    @property
    def train_size(self):
        return self._train_size

    @train_size.setter
    def train_size(self, value):
        if value is not None:
            assert isinstance(value, (tuple, list))
            assert len(value) == 2
            assert isinstance(value[0], int) and isinstance(value[1], int)
        self._train_size = value

    def add_extra_param(self, name, value):
        if self.extra_params is None:
            self.extra_params = {}
        self.extra_params[name] = value

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[ArgumentParser] = None,
    ) -> ArgumentParser:
        """Generate a parser for the arguments required by this model.

        Parameters
        ----------
        parent_parser : ArgumentParser
            An existing parser, to be extended with the arguments from this model.

        Returns
        -------
        ArgumentParser
            The parser after extending its arguments.

        Notes
        -----
        If the concrete model needs to add more arguments than these defined in this BaseModel, then it should create its
        own method defined as follows:

        >>>
        @staticmethod
        def add_model_specific_args(parent_parser):
            parent_parser = BaseModel.add_model_specific_args(parent_parser)
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument(...add new arguments...)
            return parser
        """
        parents = [parent_parser] if parent_parser is not None else []
        add_help = False if parent_parser is not None else True
        parser = ArgumentParser(parents=parents, add_help=add_help)
        parser.add_argument("--train_batch_size", type=int, default=0, help="")
        parser.add_argument("--train_num_workers", type=int, default=4, help="")
        parser.add_argument(
            "--train_transform_cuda", action="store_true", default=False, help=""
        )
        parser.add_argument(
            "--train_transform_fp16", action="store_true", default=False, help=""
        )
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--wdecay", type=float, default=1e-4)
        parser.add_argument(
            "--train_dataset",
            type=str,
            default=None,
            help=(
                "String specifying the dataset for training. See the docs of "
                "ptlflow.models.base_model.base_model.BaseModel.parse_dataset_selection for more details about this "
                "string."
            ),
        )
        parser.add_argument(
            "--train_crop_size", type=int, nargs=2, default=None, help=""
        )
        parser.add_argument(
            "--val_dataset",
            type=str,
            default=None,
            help=(
                "String specifying the dataset for validation. See the docs of "
                "ptlflow.models.base_model.base_model.BaseModel.parse_dataset_selection for more details about this "
                "string."
            ),
        )
        parser.add_argument(
            "--test_dataset",
            type=str,
            nargs="+",
            default=None,
            choices=["kitti-2012", "kitti-2015", "sintel", "spring"],
            help=(
                "Specify the datasets for testing. This option should be used to generate predictions on the"
                "test set of datasets that have official benchmarks."
            ),
        )
        parser.add_argument(
            "--pretrained_ckpt",
            type=str,
            default=None,
            help=(
                "A string identifier of the pretrained checkpoint to load. The string will be first checked against the "
                "key names in the model pretrained_checkpoints class attribute. If it is not found, then it will be treated "
                "as a path to a local file."
            ),
        )
        return parser

    def preprocess_images(
        self,
        images: torch.Tensor,
        bgr_add: Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor] = 0,
        bgr_mult: Union[
            float, Tuple[float, float, float], np.ndarray, torch.Tensor
        ] = 1,
        bgr_to_rgb: bool = False,
        image_resizer: Optional[Union[InputPadder, InputScaler]] = None,
        resize_mode: str = "pad",
        target_size: Optional[Tuple[int, int]] = None,
        pad_mode: str = "replicate",
        pad_value: float = 0.0,
        pad_two_side: bool = True,
        interpolation_mode: str = "bilinear",
        interpolation_align_corners: bool = True,
    ) -> Tuple[torch.Tensor, Union[InputPadder, InputScaler]]:
        """Applies basic pre-processing to the images.

        The pre-processing is done in this order:
        1. images = images + bgr_add
        2. images = images * bgr_mult
        3. (optional) Convert BGR channels to RGB
        4. Pad or resize the input to the closest larger size multiple of self.output_stride

        Parameters
        ----------
        images : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., 3, H, W].
        bgr_add : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 0
            BGR values to be added to the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_mult : Union[float, Tuple[float, float, float], np.ndarray, torch.Tensor], default 1
            BGR values to be multiplied by the images. It can be a single value, a triple, or a tensor with a shape compatible with images.
        bgr_to_rgb : bool, default False
            If True, flip the channels to convert from BGR to RGB.
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to resize the images.
            If not provided, a new one will be created based on the given resize_mode.
        resize_mode : str, default "pad"
            How to resize the input. Accepted values are "pad" and "interpolation".
        target_size : Optional[Tuple[int, int]], default None
            If given, the images will be resized to this size, instead of calculating a multiple of self.output_stride.
        pad_mode : str, default "replicate"
            Used if resize_mode == "pad". How to pad the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.pad.
        pad_value : float, default 0.0
            Used if resize_mode == "pad" and pad_mode == "constant". The value to fill in the padded area.
        pad_two_side : bool, default True
            Used if resize_mode == "pad". If True, half of the padding goes to left/top and the rest to right/bottom. Otherwise, all the padding goes to the bottom right.
        interpolation_mode : str, default "bilinear"
            Used if resize_mode == "interpolation". How to interpolate the input. Must be one of the values accepted by the 'mode' argument of torch.nn.functional.interpolate.
        interpolation_align_corners : bool, default True
            Used if resize_mode == "interpolation". See 'align_corners' in https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

        Returns
        -------
        torch.Tensor
            A copy of the input images after applying all of the pre-processing steps.
        Union[InputPadder, InputScaler]
            An instance of InputPadder or InputScaler that was used to resize the images.
            Can be used to reverse the resizing operations.
        """
        bgr_add = bgr_val_as_tensor(
            bgr_add, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images = images + bgr_add
        bgr_mult = bgr_val_as_tensor(
            bgr_mult, reference_tensor=images, bgr_tensor_shape_position=-3
        )
        images *= bgr_mult
        if bgr_to_rgb:
            images = torch.flip(images, [-3])

        stride = self.output_stride
        if target_size is not None:
            stride = None

        if image_resizer is None:
            if resize_mode == "pad":
                image_resizer = InputPadder(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    pad_mode=pad_mode,
                    two_side_pad=pad_two_side,
                    pad_value=pad_value,
                )
            elif resize_mode == "interpolation":
                image_resizer = InputScaler(
                    images.shape,
                    stride=stride,
                    size=target_size,
                    interpolation_mode=interpolation_mode,
                    interpolation_align_corners=interpolation_align_corners,
                )
            else:
                raise ValueError(
                    f"resize_mode must be one of (pad, interpolation). Found: {resize_mode}."
                )

        images = image_resizer.fill(images)
        images = images.contiguous()
        return images, image_resizer

    def postprocess_predictions(
        self,
        prediction: torch.Tensor,
        image_resizer: Optional[Union[InputPadder, InputScaler]],
        is_flow: bool,
    ) -> torch.Tensor:
        """Simple resizing post-processing. Just use image_resizer to revert the resizing operations.

        Parameters
        ----------
        prediction : torch.Tensor
            A tensor with at least 3 dimensions in this order: [..., C, H, W].
        image_resizer : Optional[Union[InputPadder, InputScaler]]
            An instance of InputPadder or InputScaler that will be used to reverse the resizing done to the inputs.
            Typically, this will be the instance returned by self.preprocess_images().
        is_flow : bool
            Indicates if prediction is an optical flow prediction of not.
            Only used if image_resizer is an instance of InputScaler, in which case the flow values need to be scaled.

        Returns
        -------
        torch.Tensor
            A copy of the prediction after reversing the resizing.
        """
        if isinstance(image_resizer, InputScaler):
            return image_resizer.unfill(prediction, is_flow=is_flow)
        else:
            return image_resizer.unfill(prediction)

    @abstractmethod
    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward the inputs through the network and produce the predictions.

        The method inputs can be anything, up to the implementation of the concrete model. However, the recommended input is
        to receive only dict[str, Any] as argument. This dict should contain everything required for one pass of the network
        (images, etc.). Arguments which do not change for each forward should be defined as arguments in the parser
        (see add_model_specific_args()).

        Parameters
        ----------
        args : Any
            Any arguments.
        kwargs : Any
            Any named arguments.

        Returns
        -------
        Dict[str, torch.Tensor]
            For compatibility with the framework, the output should be a dict containing at least the following keys:

            - 'flows': a 5D tensor BNCHW containing the predicted flows. The flows must be at the original scale, i.e.,
              their size is the same as the input images, and the flow magnitudes are scaled accordingly. Most networks only
              produce a single 2D optical flow per batch element, so the output shape will be B12HW. N can be larger than one
              if the network produces predictions for a larger temporal window.

            - 'occs': optional, and only included if the network also predicts occlusion masks. It is a 5D tensor following the
              same structure as 'flows'.

            - 'mbs': same as 'occs' but for occlusion masks.

            - 'confs': same as 'occs' but for flow confidence predictions.


            The keys above are used by other parts of PTLFlow (if available). The output can also include any other keys as
            well. For example, by default, the output of forward() will be passed to the loss function. So the output may
            include keys which are going to be used for computing the training loss.
        """
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Perform one step of the training.

        This function is called internally by Pytorch-Lightning during training.

        Parameters
        ----------
        batch : Dict[str, Any]
            One batch of data, that is going to be given as input to the network.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        Dict[str, Any]
            A dict with two keys:

            - 'loss': torch.Tensor, containing the loss value. Required by Pytorch-Lightning for the optimization step.

            - 'dataset_name': str, a string representing the name of the dataset from where this batch came from. Used only for
              logging purposes.
        """
        preds = self(batch)
        self.last_inputs = batch
        self.last_predictions = preds
        loss = self.loss_fn(preds, batch)
        metrics = self.train_metrics(preds, batch)
        if isinstance(loss, dict):
            for k, v in loss.items():
                metrics[f"train/{k}"] = v.item()
            loss = loss["loss"]
        else:
            metrics["train/loss"] = loss.item()
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.log(
            "epe", metrics["train/epe"], prog_bar=True, on_step=True, on_epoch=True
        )

        outputs = {"loss": loss, "dataset_name": batch["meta"]["dataset_name"]}
        return outputs

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Perform one step of the validation.

        This function is called internally by Pytorch-Lightning during validation.

        Parameters
        ----------
        batch : Dict[str, Any]
            One batch of data, that is going to be given as input to the network.
        batch_idx : int
            The index of the current batch.
        dataloader_idx : int, default 0
            When using multiple loaders, indicate from which loader this input is coming from.

        Returns
        -------
        Dict[str, Any]
            A dict with two keys:

            - 'metrics': dict[str, torch.Tensor], the metrics computed during this validation step.

            - 'dataset_name': str, a string representing the name of the dataset from where this batch came from. Used only for
              logging purposes.

        See Also
        --------
        ptlflow.utils.flow_metrics.FlowMetrics : class to manage and compute the optical flow metrics.
        """
        preds = self(batch)
        self.last_inputs = batch
        self.last_predictions = preds
        metrics = self.val_metrics(preds, batch)
        train_val_metrics = self._split_train_val_metrics(metrics, batch.get("meta"))

        outputs = {
            "metrics": train_val_metrics,
            "dataset_name": batch["meta"]["dataset_name"],
        }
        return outputs

    def validation_epoch_end(
        self,
        outputs: Union[
            List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]
        ],
    ) -> None:
        """Perform operations at the end of one validation epoch.

        This function is called internally by Pytorch-Lightning during validation.

        Parameters
        ----------
        outputs : Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
            A list in which each element is the output of validation_step().

        See Also
        --------
        validation_step
        """
        if len(outputs) > 0:
            if not isinstance(outputs[0], list):
                outputs = [outputs]

            for i, loader_outputs in enumerate(outputs):
                # Log mean metrics for one val dataloader
                metrics_cum = {}
                for output_dict in loader_outputs:
                    for k, v in output_dict["metrics"].items():
                        if metrics_cum.get(k) is None:
                            metrics_cum[k] = [0.0, 0]
                        metrics_cum[k][0] += v
                        metrics_cum[k][1] += 1
                metrics_mean = {}
                for k, v in metrics_cum.items():
                    metrics_mean[k] = v[0] / v[1]
                self.log_dict(metrics_mean, sync_dist=True)

                # Find the EPE metric on the full split
                epe_key = [k for k in metrics_mean if "full" in k and "epe" in k][0]
                self.log(
                    self.val_dataloader_names[i],
                    metrics_mean[epe_key],
                    prog_bar=True,
                    sync_dist=True,
                )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizers and LR schedulers.

        This function is called internally by Pytorch-Lightning at the beginning of the training.

        Returns
        -------
        Dict[str, Any]
            A dict with two keys:
            - 'optimizer': an optimizer from PyTorch.
            - 'lr_scheduler': Dict['str', Any], a dict with the selected scheduler and its required arguments.
        """
        assert (
            self.loss_fn is not None
        ), f"Model {self.__class__.__name__} cannot be trained. It does not have loss function."

        self.train_dataloader()  # Just to initialize dataloader variables

        if self.args.max_steps is None or self.args.max_steps <= 0:
            if self.args.max_epochs is None:
                self.args.max_epochs = 10
                logging.warning(
                    "--max_epochs is not set. It will be set to %d.",
                    self.args.max_epochs,
                )

            gpu_divider = self.args.gpus
            if isinstance(gpu_divider, list) or isinstance(gpu_divider, tuple):
                gpu_divider = len(gpu_divider)
            elif isinstance(gpu_divider, str):
                gpu_divider = len(gpu_divider.split(","))
            elif not isinstance(gpu_divider, int):
                gpu_divider = 1

            self.args.max_steps = self.args.max_epochs * (
                self.train_dataloader_length // gpu_divider
            )

        optimizer = optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay
        )

        accumulate_grad_batches = (
            1
            if self.args.accumulate_grad_batches is None
            else self.args.accumulate_grad_batches
        )
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.args.lr,
            total_steps=self.args.max_steps // accumulate_grad_batches,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def train_dataloader(self) -> DataLoader:
        """Initialize and return the training dataloader.

        self.args.train_dataset will be parsed into the selected datasets and their parameters. parse_dataset_selection() is
        used to parse, and the parsed outputs are used as follows: (dataset_multiplier, dataset_name, dataset_params...).

        A method called _get_[dataset_name]_dataset(dataset_params) will be called to get the instance of each dataset.

        Each dataset will be repeated dataset_multiplier times and then mixed together into a single dataloader with samples
        of all the datasets.

        Returns
        -------
        DataLoader
            A single dataloader with all the selected training datasets.

        Notes
        -----
        This function could be easily modified to return one dataloader for each dataset, which may be preferable in some
        cases. However, this would require that the loading function have some logic to mix all the dataloaders internally.
        In order to keep the whole BaseModel more generic, we use just a single dataloader randomly mixed to train.

        See Also
        --------
        parse_dataset_selection
            This function does the parsing of self.args.train_dataset. You can read its documentation to learn how to write
            valid dataset strings.
        val_dataloader
            Similar to this method, but val_dataloader returns one dataloader for each dataset.
        """
        if self.args.train_dataset is None:
            self.args.train_dataset = "chairs-train"
            logging.warning(
                "--train_dataset is not set. It will be set as %s",
                self.args.train_dataset,
            )
        if self.args.train_batch_size == 0:
            self.args.train_batch_size = 8
            logging.warning(
                "--train_batch_size is not set. It will be set to %d",
                self.args.train_batch_size,
            )

        if self.args.train_dataset is not None:
            parsed_datasets = self.parse_dataset_selection(self.args.train_dataset)
            train_dataset = None
            for parsed_vals in parsed_datasets:
                multiplier = parsed_vals[0]
                dataset_name = parsed_vals[1]
                dataset = getattr(self, f"_get_{dataset_name}_dataset")(
                    True, *parsed_vals[2:]
                )
                for _ in range(multiplier):
                    if train_dataset is None:
                        train_dataset = dataset
                    else:
                        train_dataset += dataset

            train_pin_memory = False if self.args.train_transform_cuda else True
            train_dataloader = DataLoader(
                train_dataset,
                self.args.train_batch_size,
                shuffle=True,
                num_workers=self.args.train_num_workers,
                pin_memory=train_pin_memory,
                drop_last=False,
            )
            self.train_dataloader_length = len(train_dataloader)
            return train_dataloader

    def val_dataloader(self) -> Optional[List[DataLoader]]:
        """Initialize and return the list of validation dataloaders.

        self.args.val_dataset will be parsed into the selected datasets and their parameters. parse_dataset_selection() is
        used to parse, and the parsed outputs are used as follows: (dataset_multiplier, dataset_name, dataset_params...).

        A method called _get_[dataset_name]_dataset(dataset_params) will be called to get the instance of each dataset.

        If there is a dataset_multiplier, it will be ignored. Each dataset will be returned in a separated dataloader.

        Returns
        -------
        Optional[List[DataLoader]]
            A list of dataloaders each for one dataset.

        See Also
        --------
        parse_dataset_selection
            This function does the parsing of self.args.val_dataset. You can read its documentation to learn how to write
            valid dataset strings.
        """
        if self.args.val_dataset is None:
            self.args.val_dataset = "sintel-clean+sintel-final+kitti-2015"
            logging.warning(
                "--val_dataset is not set. It will be set as %s. If you want to skip validation, then set --val_dataset none.",
                self.args.val_dataset,
            )
        elif self.args.val_dataset.lower() == "none":
            return None

        parsed_datasets = self.parse_dataset_selection(self.args.val_dataset)
        dataloaders = []
        self.val_dataloader_names = []
        self.val_dataloader_lengths = []
        for parsed_vals in parsed_datasets:
            dataset_name = parsed_vals[1]
            dataset = getattr(self, f"_get_{dataset_name}_dataset")(
                False, *parsed_vals[2:]
            )
            dataloaders.append(
                DataLoader(
                    dataset,
                    1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                )
            )

            self.val_dataloader_names.append("-".join(parsed_vals[1:]))
            self.val_dataloader_lengths.append(len(dataset))

        return dataloaders

    def test_dataloader(self) -> List[DataLoader]:
        """Initialize and return the list of test dataloaders.

        The list of datasets to load is provided by the --test_dataset argument.

        Returns
        -------
        Optional[List[DataLoader]]
            A list of dataloaders each for one dataset.

        Raises
        ------
        ValueError
            If --test_dataset is not provided.
        """
        if self.args.test_dataset is None:
            raise ValueError(
                "--test_dataset is not set. Please inform at least one dataset for the testing."
            )

        dataset_ids = self.args.test_dataset
        if "sintel" in dataset_ids:
            dataset_ids.remove("sintel")
            dataset_ids.extend(["sintel-clean", "sintel-final"])
        elif "spring" in dataset_ids:
            dataset_ids.append("spring-revonly")

        dataloaders = []
        for dataset_id in dataset_ids:
            dataset_id += "-test"
            dataset_tokens = dataset_id.split("-")
            dataset = getattr(self, f"_get_{dataset_tokens[0]}_dataset")(
                False, *dataset_tokens[1:]
            )
            dataloaders.append(
                DataLoader(
                    dataset,
                    1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False,
                )
            )

            self.test_dataloader_names.append(dataset_id)

        return dataloaders

    def parse_dataset_selection(
        self,
        dataset_selection: str,
    ) -> List[Tuple[str, int]]:
        """Parse the input string into the selected dataset and their multipliers and parameters.

        For example, 'chairs-train+3*sintel-clean-trainval+kitti-2012-train*5' will be parsed into
        [(1, 'chairs', 'train'), (3, 'sintel', 'clean', 'trainval'), (5, 'kitti', '2012', 'train')].

        Parameters
        ----------
        dataset_selection : str
            The string defining the dataset selection. Each dataset is separated by a '+' sign. The multiplier must be either
            in the beginning or the end of one dataset string, connected to a '*' sign. The remaining content must be separated
            by '-' symbols.

        Returns
        -------
        List[Tuple[str, int]]
            The parsed choice of datasets and their number of repetitions.

        Raises
        ------
        ValueError
            If the given string is invalid.
        """
        if dataset_selection is None:
            return []

        dataset_selection = dataset_selection.replace(" ", "")
        datasets = dataset_selection.split("+")
        for i in range(len(datasets)):
            tokens = datasets[i].split("*")
            if len(tokens) == 1:
                datasets[i] = (1,) + tuple(tokens[0].split("-"))
            elif len(tokens) == 2:
                try:
                    mult, params = int(tokens[0]), tokens[1]
                except ValueError:
                    params, mult = tokens[0], int(
                        tokens[1]
                    )  # if the multiplier comes last.
                datasets[i] = (mult,) + tuple(params.split("-"))
            else:
                raise ValueError(
                    "The specified dataset string {:} is invalid. Check the BaseModel.parse_dataset_selection() documentation "
                    "to see how to write a valid selection string."
                )
        return datasets

    ###########################################################################
    # Logging
    ###########################################################################

    def _split_train_val_metrics(
        self, metrics: Dict[str, float], inputs_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Split the val metrics into 'train', 'val', and 'full' categories.

        This is useful for producing metrics for different subsets without having to forward the same sample multiple times.
        When computing metrics for the full validation dataset, the metrics for the 'train' and 'val' subsets are also
        computed at the same time.

        Parameters
        ----------
        metrics : dict[str, float]
            The metrics computed after one validation step.
        inputs_meta : Optional[Dict[str, Any]], optional
            If this is not provided, then the metrics are not split. This is a dict with two keys:
            - 'dataset_name': str, a string representing the name of the dataset these metrics correspond to.
            - 'is_val': bool, indicating whether these metrics come from a sample in the 'val' split or not.

        Returns
        -------
        dict[str, float]
            The metrics after being split.

        See Also
        --------
        validation_step
        """
        dataset_name = None
        if inputs_meta is not None and inputs_meta.get("dataset_name") is not None:
            dataset_name = inputs_meta["dataset_name"][0]

        log_metrics = {}
        for k, v in metrics.items():
            if dataset_name is not None:
                log_metrics[f"val_{dataset_name}/full/{k}"] = v
            else:
                log_metrics[f"val/full/{k}"] = v

            if inputs_meta is not None and inputs_meta.get("is_val") is not None:
                if inputs_meta["is_val"][0]:
                    split = "val"
                else:
                    split = "train"

                if dataset_name is not None:
                    log_metrics[f"val_{dataset_name}/{split}/{k}"] = v
                else:
                    log_metrics[f"val/{split}/{k}"] = v

        return log_metrics

    ###########################################################################
    # _get_datasets
    ###########################################################################

    def _get_autoflow_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(496, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "trainval"
        if len(args) > 0 and args[0] in ["train", "val", "trainval"]:
            split = args[0]
        dataset = AutoFlowDataset(
            self.args.autoflow_root_dir, split=split, transform=transform
        )
        return dataset

    def _get_chairs_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(496, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "trainval"
        if len(args) > 0 and args[0] in ["train", "val", "trainval"]:
            split = args[0]
        dataset = FlyingChairsDataset(
            self.args.flying_chairs_root_dir, split=split, transform=transform
        )
        return dataset

    def _get_chairs2_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(496, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx),
                        (-0.1, 1.0),
                        (-0.2, 0.2),
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "trainval"
        add_reverse = False
        get_occlusion_mask = False
        get_motion_boundary_mask = False
        get_backward = False
        for v in args:
            if v in ["train", "val", "trainval"]:
                split = v
            elif v == "rev":
                add_reverse = True
            elif v == "occ":
                get_occlusion_mask = True
            elif v == "mb":
                get_motion_boundary_mask = True
            elif v == "back":
                get_backward = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        dataset = FlyingChairs2Dataset(
            self.args.flying_chairs2_root_dir,
            split=split,
            transform=transform,
            add_reverse=add_reverse,
            get_occlusion_mask=get_occlusion_mask,
            get_motion_boundary_mask=get_motion_boundary_mask,
            get_backward=get_backward,
        )
        return dataset

    def _get_hd1k_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(768, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.5, 0.2), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "trainval"
        sequence_length = 2
        sequence_position = "first"
        for v in args:
            if v in ["train", "val", "trainval", "test"]:
                split = args[0]
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            else:
                raise ValueError(f"Invalid arg: {v}")

        dataset = Hd1kDataset(
            self.args.hd1k_root_dir,
            split=split,
            transform=transform,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_kitti_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(288, self.output_stride), md(960, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop(
                        (cy, cx), (-0.2, 0.4), (-0.2, 0.2), sparse=True
                    ),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                ]
            )
        else:
            transform = ft.ToTensor()

        versions = ["2012", "2015"]
        split = "trainval"
        get_occlusion_mask = False
        for v in args:
            if v in ["2012", "2015"]:
                versions = [v]
            elif v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "occ":
                get_occlusion_mask = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        dataset = KittiDataset(
            self.args.kitti_2012_root_dir,
            self.args.kitti_2015_root_dir,
            get_occlusion_mask=get_occlusion_mask,
            versions=versions,
            split=split,
            transform=transform,
        )
        return dataset

    def _get_kubric_dataset(self, is_train: bool, *args: str) -> Dataset:
        if is_train:
            raise NotImplementedError()
        else:
            transform = ft.ToTensor()

        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        for v in args:
            if v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]

        dataset = KubricDataset(
            self.args.kubric_root_dir,
            transform=transform,
            get_backward=get_backward,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_sintel_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(768, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.6), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        pass_names = ["clean", "final"]
        split = "trainval"
        get_occlusion_mask = False
        sequence_length = 2
        sequence_position = "first"
        for v in args:
            if v in ["clean", "final"]:
                pass_names = [v]
            elif v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "occ":
                get_occlusion_mask = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            else:
                raise ValueError(f"Invalid arg: {v}")

        dataset = SintelDataset(
            self.args.mpi_sintel_root_dir,
            split=split,
            pass_names=pass_names,
            transform=transform,
            get_occlusion_mask=get_occlusion_mask,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_spring_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(368, self.output_stride), md(768, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # Transforms copied from Sintel config
            # Untested! Not sure if they are optimal!
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.2, 0.6), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        split = "train"
        add_reverse = False
        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        reverse_only = False
        side_names = []
        for v in args:
            if v in ["train", "val", "trainval", "test"]:
                split = v
            elif v == "rev":
                add_reverse = True
            elif v == "revonly":
                reverse_only = True
            elif v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "left":
                side_names.append("left")
            elif v == "right":
                side_names.append("right")
            else:
                raise ValueError(f"Invalid arg: {v}")

        if len(side_names) == 0:
            side_names = ["left", "right"]

        dataset = SpringDataset(
            self.args.spring_root_dir,
            split=split,
            side_names=side_names,
            add_reverse=add_reverse,
            transform=transform,
            get_backward=get_backward,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
            reverse_only=reverse_only,
        )
        return dataset

    def _get_tartanair_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        get_occlusion_mask = False
        sequence_length = 2
        sequence_position = "first"
        difficulties = []
        for v in args:
            if v in ["easy", "hard"]:
                difficulties.append(v)
            elif v == "occ":
                get_occlusion_mask = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            else:
                raise ValueError(f"Invalid arg: {v}")

        if len(difficulties) == 0:
            difficulties = ["easy"]

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(360, self.output_stride), md(480, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), (-0.4, 0.8), (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        dataset = TartanAirDataset(
            self.args.tartanair_root_dir,
            difficulties=difficulties,
            transform=transform,
            get_occlusion_mask=get_occlusion_mask,
            sequence_length=sequence_length,
            sequence_position=sequence_position,
        )
        return dataset

    def _get_things_dataset(self, is_train: bool, *args: str) -> Dataset:
        device = "cuda" if self.args.train_transform_cuda else "cpu"
        md = make_divisible

        pass_names = ["clean", "final"]
        split = "trainval"
        is_subset = False
        add_reverse = False
        get_occlusion_mask = False
        get_motion_boundary_mask = False
        get_backward = False
        sequence_length = 2
        sequence_position = "first"
        sintel_transform = False
        for v in args:
            if v in ["clean", "final"]:
                pass_names = [v]
            elif v in ["train", "val", "trainval"]:
                split = v
            elif v == "subset":
                is_subset = True
            elif v == "rev":
                add_reverse = True
            elif v == "occ":
                get_occlusion_mask = True
            elif v == "mb":
                get_motion_boundary_mask = True
            elif v == "back":
                get_backward = True
            elif v.startswith("seqlen"):
                sequence_length = int(v.split("_")[1])
            elif v.startswith("seqpos"):
                sequence_position = v.split("_")[1]
            elif v == "sinteltransform":
                sintel_transform = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_train:
            if self.args.train_crop_size is None:
                cy, cx = (md(400, self.output_stride), md(720, self.output_stride))
                self.args.train_crop_size = (cy, cx)
                logging.warning(
                    "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
                )
            else:
                cy, cx = (
                    md(self.args.train_crop_size[0], self.output_stride),
                    md(self.args.train_crop_size[1], self.output_stride),
                )

            # These transforms are based on RAFT: https://github.com/princeton-vl/RAFT
            if sintel_transform:
                major_scale = (-0.2, 0.6)
            else:
                major_scale = (-0.4, 0.8)
            transform = ft.Compose(
                [
                    ft.ToTensor(device=device, fp16=self.args.train_transform_fp16),
                    ft.RandomScaleAndCrop((cy, cx), major_scale, (-0.2, 0.2)),
                    ft.ColorJitter(0.4, 0.4, 0.4, 0.5 / 3.14, 0.2),
                    ft.GaussianNoise(0.02),
                    ft.RandomPatchEraser(
                        0.5, (int(1), int(3)), (int(50), int(100)), "mean"
                    ),
                    ft.RandomFlip(min(0.5, 0.5), min(0.1, 0.5)),
                ]
            )
        else:
            transform = ft.ToTensor()

        if is_subset:
            dataset = FlyingThings3DSubsetDataset(
                self.args.flying_things3d_subset_root_dir,
                split=split,
                pass_names=pass_names,
                side_names=["left", "right"],
                add_reverse=add_reverse,
                transform=transform,
                get_occlusion_mask=get_occlusion_mask,
                get_motion_boundary_mask=get_motion_boundary_mask,
                get_backward=get_backward,
                sequence_length=sequence_length,
                sequence_position=sequence_position,
            )
        else:
            dataset = FlyingThings3DDataset(
                self.args.flying_things3d_root_dir,
                split=split,
                pass_names=pass_names,
                side_names=["left", "right"],
                add_reverse=add_reverse,
                transform=transform,
                get_occlusion_mask=get_occlusion_mask,
                get_motion_boundary_mask=get_motion_boundary_mask,
                get_backward=get_backward,
                sequence_length=sequence_length,
                sequence_position=sequence_position,
            )
        return dataset

    def _get_overfit_dataset(self, is_train: bool, *args: str) -> Dataset:
        md = make_divisible
        if self.args.train_crop_size is None:
            cy, cx = (md(436, self.output_stride), md(1024, self.output_stride))
            self.args.train_crop_size = (cy, cx)
            logging.warning(
                "--train_crop_size is not set. It will be set as (%d, %d).", cy, cx
            )
        else:
            cy, cx = (
                md(self.args.train_crop_size[0], self.output_stride),
                md(self.args.train_crop_size[1], self.output_stride),
            )
        transform = ft.Compose([ft.ToTensor(), ft.Resize((cy, cx))])

        dataset_name = "sintel"
        if len(args) > 0 and args[0] in ["chairs2"]:
            dataset_name = args[0]

        if dataset_name == "sintel":
            dataset = SintelDataset(
                self.args.mpi_sintel_root_dir,
                split="trainval",
                pass_names="clean",
                transform=transform,
                get_occlusion_mask=False,
            )
        elif dataset_name == "chairs2":
            dataset = FlyingChairs2Dataset(
                self.args.flying_chairs2_root_dir,
                split="trainval",
                transform=transform,
                add_reverse=False,
                get_occlusion_mask=True,
                get_motion_boundary_mask=True,
                get_backward=True,
            )

        dataset.img_paths = dataset.img_paths[:1]
        dataset.flow_paths = dataset.flow_paths[:1]
        dataset.occ_paths = dataset.occ_paths[:1]
        dataset.mb_paths = dataset.mb_paths[:1]
        dataset.flow_b_paths = dataset.flow_b_paths[:1]
        dataset.occ_b_paths = dataset.occ_b_paths[:1]
        dataset.mb_b_paths = dataset.mb_b_paths[:1]
        dataset.metadata = dataset.metadata[:1]

        return dataset
