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

from abc import abstractmethod
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import lightning.pytorch as pl
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ptlflow.utils.utils import InputPadder, InputScaler
from ptlflow.utils.utils import bgr_val_as_tensor
from ptlflow.utils.flow_metrics import FlowMetrics

DATASET_MAIN_METRIC = {
    "autoflow": "epe",
    "flyingchairs": "epe",
    "flyingchairs2": "epe",
    "flyingthings3d": "epe",
    "flyingthings3dsubset": "epe",
    "hd1k": "flall",
    "kitti_2012": "flall",
    "kitti_2015": "flall",
    "kubric": "epe",
    "middlebury": "epe",
    "middleburyst": "epe",
    "monkaa": "epe",
    "sintel_clean": "epe",
    "sintel_final": "epe",
    "spring": "px1",
    "tartanair_easy": "epe",
    "tartanair_hard": "epe",
    "viper": "wauc",
}


class BaseModel(pl.LightningModule):
    """A base abstract optical flow model."""

    def __init__(
        self,
        output_stride: int,
        loss_fn: Optional[Callable] = None,
        lr: Optional[float] = None,
        wdecay: Optional[float] = None,
        warm_start: bool = False,
        metric_interpolate_pred_to_target_size: bool = False,
    ) -> None:
        """Initialize BaseModel.

        Parameters
        ----------
        output_stride : int
            How many times the output of the network is smaller than the input.
        loss_fn : Optional[Callable]
            A function to be used to compute the loss for the training. The input of this function must match the output of the
            forward() method. The output of this function must be a tensor with a single value.
        lr : Optional[float]
            The learning rate to be used for training the model. If not provided, it will be set as 1e-4.
        wdecay : Optional[float]
            The weight decay to be used for training the model. If not provided, it will be set as 1e-4.
        warm_start : bool, default False
            If True, use warm start to initialize the flow prediction. The warm_start strategy was presented by the RAFT method and forward interpolates the prediction from the last frame.
        metric_interpolate_pred_to_target_size : bool, default False
            If True, the prediction is bilinearly interpolated to match the target size during metric calculation, if their sizes are different.
        """
        super(BaseModel, self).__init__()

        self.output_stride = output_stride
        self.loss_fn = loss_fn
        self.lr = lr
        self.wdecay = wdecay
        self.warm_start = warm_start
        self.metric_interpolate_pred_to_target_size = (
            metric_interpolate_pred_to_target_size
        )

        self.train_size = None
        self.train_avg_length = None

        self.extra_params = None

        self.train_metrics = FlowMetrics(
            prefix="train/",
            interpolate_pred_to_target_size=self.metric_interpolate_pred_to_target_size,
        )
        self.val_metrics = nn.ModuleList()
        self.val_dataset_names = []

        self.last_inputs = None
        self.last_predictions = None

        self.prev_preds = None

        self.has_trained_on_ptlflow = False

        self.has_logged_main_val_metric_message = False

        self.save_hyperparameters(
            ignore=["loss_fn"],
        )

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

    def preprocess_images(
        self,
        images: torch.Tensor,
        stride: Optional[int] = None,
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

        stride = self.output_stride if stride is None else stride
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

        This function is called internally by Pytorch Lightning during training.

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

            - 'loss': torch.Tensor, containing the loss value. Required by Pytorch Lightning for the optimization step.

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

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Perform one step of the validation.

        This function is called internally by Pytorch Lightning during validation.

        Parameters
        ----------
        batch : Dict[str, Any]
            One batch of data, that is going to be given as input to the network.
        batch_idx : int
            The index of the current batch.
        dataloader_idx : int, default 0
            When using multiple loaders, indicate from which loader this input is coming from.

        See Also
        --------
        ptlflow.utils.flow_metrics.FlowMetrics : class to manage and compute the optical flow metrics.
        """
        if len(self.val_metrics) <= dataloader_idx:
            self.val_metrics.append(
                FlowMetrics(
                    prefix="val/",
                    interpolate_pred_to_target_size=self.metric_interpolate_pred_to_target_size,
                ).to(device=batch["flows"].device)
            )
            self.val_dataset_names.append(None)

        if self.warm_start:
            batch["prev_preds"] = self.prev_preds

        preds = self(batch)
        self.last_inputs = batch
        self.last_predictions = preds
        metrics = self.val_metrics[dataloader_idx](preds, batch)
        inputs_meta = batch.get("meta")
        train_val_metrics = self._split_train_val_metrics(metrics, inputs_meta)
        if (
            self.val_dataset_names[dataloader_idx] is None
            and inputs_meta is not None
            and inputs_meta.get("dataset_name") is not None
        ):
            name = inputs_meta.get("dataset_name")[0]
            if inputs_meta.get("split_name") is not None:
                name += f"-{inputs_meta.get('split_name')[0]}"
            self.val_dataset_names[dataloader_idx] = name
        self.log_dict(
            train_val_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if self.warm_start:
            if "is_seq_start" in batch["meta"] and batch["meta"]["is_seq_start"][0]:
                self.prev_preds = None
            else:
                self.prev_preds = preds
                for k, v in self.prev_preds.items():
                    if isinstance(v, torch.Tensor):
                        self.prev_preds[k] = v.detach()

        return {"preds": preds, "metrics": metrics}

    def on_validation_epoch_end(self) -> None:
        for i in range(len(self.val_metrics)):
            metrics = self.val_metrics[i].compute()
            dset_name = self.val_dataset_names[i].lower()
            for name, val in metrics.items():
                main_metric = (
                    DATASET_MAIN_METRIC[dset_name]
                    if dset_name in DATASET_MAIN_METRIC
                    else "epe"
                )
                main_metric = f"val/{main_metric}"
                if name == main_metric:
                    self.log(dset_name, val, sync_dist=True, prog_bar=True)

                    if i == 0:
                        self.log("main_val_metric", val, sync_dist=True)
                        if not self.has_logged_main_val_metric_message:
                            logger.info(
                                "main_val_metric is tracking the metric {}/{}",
                                dset_name,
                                name,
                            )
                            self.has_logged_main_val_metric_message = True
            self.val_metrics[i].reset()

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        """Perform one step of the test.

        This function is called internally by Pytorch Lightning during testing.

        Parameters
        ----------
        batch : Dict[str, Any]
            One batch of data, that is going to be given as input to the network.
        batch_idx : int
            The index of the current batch.
        dataloader_idx : int, default 0
            When using multiple loaders, indicate from which loader this input is coming from.
        """
        if self.warm_start:
            batch["prev_preds"] = self.prev_preds

        preds = self(batch)
        self.last_inputs = batch
        self.last_predictions = preds

        if self.warm_start:
            if "is_seq_start" in batch["meta"] and batch["meta"]["is_seq_start"][0]:
                self.prev_preds = None
            else:
                self.prev_preds = preds
                for k, v in self.prev_preds.items():
                    if isinstance(v, torch.Tensor):
                        self.prev_preds[k] = v.detach()

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizers and LR schedulers.

        This function is called internally by Pytorch Lightning at the beginning of the training.

        Returns
        -------
        Dict[str, Any]
            A dict with two keys:
            - 'optimizer': an optimizer from PyTorch.
            - 'lr_scheduler': Dict['str', Any], a dict with the selected scheduler and its required arguments.
        """
        assert (
            self.loss_fn is not None
        ), f"Model {self.__class__.__name__} cannot be trained. It does not have a loss function."

        if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
            device_divider = self.trainer.device_ids
            if isinstance(device_divider, list) or isinstance(device_divider, tuple):
                device_divider = len(device_divider)
            elif isinstance(device_divider, str):
                device_idx = [v for v in device_divider.split(",") if len(v) > 0]
                device_divider = len(device_idx)
            elif not isinstance(device_divider, int):
                device_divider = 1

            grad_batches_divider = (
                1
                if self.trainer.accumulate_grad_batches is None
                else self.trainer.accumulate_grad_batches
            )

            # Hack to get train_dataloader before it is set to self.trainer.train_dataloader
            steps_per_epoch = len(
                self.trainer.fit_loop._data_source.instance.train_dataloader()
            )
            max_steps = self.trainer.max_epochs * int(
                math.ceil(
                    float(steps_per_epoch) / device_divider / grad_batches_divider
                )
            )
            logger.info(
                "--trainer.max_steps is not provided. max_steps will be set as {} ({} epochs * {} steps per epoch / {} devices / {} accumulate grad steps)",
                max_steps,
                self.trainer.max_epochs,
                steps_per_epoch,
                device_divider,
                grad_batches_divider,
            )
        else:
            max_steps = self.trainer.max_steps

        if self.lr is None:
            self.lr = 1e-4
            logger.warning("--model.lr is not set. It will be set as {}", self.lr)
        if self.wdecay is None:
            self.wdecay = 1e-4
            logger.warning(
                "--model.wdecay is not set. It will be set as {}", self.wdecay
            )

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wdecay)

        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            total_steps=max_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

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
            dataset_name = inputs_meta["dataset_name"][0].lower()

        log_metrics = {}
        for k, v in metrics.items():
            if k.startswith("val/"):
                k = k[4:]

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
