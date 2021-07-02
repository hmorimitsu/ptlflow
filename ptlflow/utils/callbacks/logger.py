"""Implement a callback to log images."""

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

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

try:
    from neptune.new.types import File as NeptuneFile
except ImportError:
    NeptuneFile = None
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
try:
    import wandb
except ImportError:
    wandb = None

from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.utils import config_logging

config_logging()


class LoggerCallback(Callback):
    """Callback to collect and log images during training and validation.

    For each dataloader, num_images samples will be collected. The samples are collected by trying to retrieve from both
    inputs and outputs tensors whose keys match the values provided in log_keys.

    num_images samples are uniformly sampled from the whole dataloader.
    """

    def __init__(
        self,
        num_images: int = 5,
        image_size: Tuple[int, int] = (200, 400),
        log_keys: Sequence[str] = ('images', 'flows', 'occs', 'mbs', 'confs'),
        epe_clip: float = 5.0
    ) -> None:
        """Initialize LoggerCallback.

        Parameters
        ----------
        num_images : int, default 5
            Number of images to log during one epoch.
        image_size : Tuple[int, int], default (200, 400)
            The size of the stored images.
        log_keys : Sequence[str], default ('images', 'flows', 'occs', 'mbs', 'confs')
            The keys to use to collect the images from the inputs and outputs of the model. If a key is not found, it is
            ignored.
        epe_clip : float, default 5.0
            The maximum EPE value that is shown on EPE image. All EPE values above this will be clipped.
        """
        super().__init__()

        self.num_images = num_images
        self.image_size = image_size
        self.log_keys = log_keys
        self.epe_clip = epe_clip

        self.train_collect_img_idx = []
        self.train_images = {}

        self.val_dataloader_names = []
        self.val_collect_image_idx = {}
        self.val_images = {}

    def log_image(
        self,
        title: str,
        image: torch.Tensor,
        pl_module: BaseModel
    ) -> None:
        """Log the image in all of the pl_module loggers.

        Note, however, that not all loggers may be able to log images.

        Parameters
        ----------
        title : str
            A title for the image.
        image : torch.Tensor
            The image to log. It must be a 3D tensor CHW (typically C=3).
        pl_module : BaseModel
            An instance of the optical flow model to get the logger from.
        """
        image_npy = image.permute(1, 2, 0).numpy()

        logger_collection = pl_module.logger
        if logger_collection is not None:
            if not isinstance(logger_collection, LoggerCollection):
                logger_collection = LoggerCollection([logger_collection])

            for logger in logger_collection:
                if isinstance(logger, CometLogger):
                    logger.experiment.log_image(image_npy, name=title)
                elif isinstance(logger, NeptuneLogger):
                    logger.experiment[title].log(NeptuneFile.as_image(image))
                elif isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image(title, image, pl_module.global_step)
                elif isinstance(logger, WandbLogger) and wandb is not None:
                    title_wb = title.replace('/', '-')
                    image_wb = wandb.Image(image_npy)
                    logger.experiment.log({title_wb: image_wb})

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModel,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """Store one image to be logged, if the current batch_idx is in the log selection group.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        outputs : Dict[str, torch.Tensor]
            The outputs of the current training batch.
        batch : Dict[str, torch.Tensor]
            The inputs of the current training batch.
        batch_idx : int
            The counter value of the current batch.
        dataloader_idx : int
            The index number of the current dataloader.
        """
        if batch_idx in self.train_collect_img_idx:
            self._append_images(self.train_images, pl_module.last_inputs, pl_module.last_predictions)

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: BaseModel
    ) -> None:
        """Reset the training log params and accumulators.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        """
        self.train_images = {}
        collect_idx = np.unique(np.linspace(
            0, self._compute_max_range(pl_module.train_dataloader_length, pl_module.args.limit_train_batches),
            self.num_images, dtype=np.int32))
        self.train_collect_img_idx = collect_idx

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModel,
        outputs: Any = None  # This arg does not exist anymore, but it is kept here for compatibility
    ) -> None:
        """Log the images accumulated during the training.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        outputs : Any
            Outputs of the training epoch.
        """
        img_grid = self._make_image_grid(self.train_images)
        self.log_image('train', img_grid, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModel,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """Store one image to be logged, if the current batch_idx is in the log selection group.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        outputs : Dict[str, torch.Tensor]
            The outputs of the current validation batch.
        batch : Dict[str, torch.Tensor]
            The inputs of the current validation batch.
        batch_idx : int
            The counter value of the current batch.
        dataloader_idx : int
            The index number of the current dataloader.
        """
        dl_name = self.val_dataloader_names[dataloader_idx]
        if batch_idx in self.val_collect_image_idx[dl_name]:
            self._append_images(self.val_images[dl_name], pl_module.last_inputs, pl_module.last_predictions)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: BaseModel
    ) -> None:
        """Reset the validation log params and accumulators.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        """
        self.val_dataloader_names = pl_module.val_dataloader_names
        for dl_name in self.val_dataloader_names:
            self.val_images[dl_name] = {}

        for dname, dlen in zip(pl_module.val_dataloader_names, pl_module.val_dataloader_lengths):
            collect_idx = np.unique(np.linspace(
                0, self._compute_max_range(dlen, pl_module.args.limit_val_batches), self.num_images, dtype=np.int32))
            self.val_collect_image_idx[dname] = collect_idx

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: BaseModel
    ) -> None:
        """Log the images accumulated during the validation.

        Parameters
        ----------
        trainer : Trainer
            An instance of the PyTorch Lightning trainer.
        pl_module : BaseModel
            An instance of the optical flow model.
        """
        for dl_name, dl_images in self.val_images.items():
            img_grid = self._make_image_grid(dl_images)
            self.log_image(f'val/{dl_name}', img_grid, pl_module)

    def _add_title(
        self,
        image: torch.Tensor,
        img_title: str
    ) -> torch.Tensor:
        """Add a title to an image.

        Parameters
        ----------
        image : torch.Tensor
            The image where the title will be added.
        img_title : str
            The title to be added.

        Returns
        -------
        torch.Tensor
            The input image with the title superposed on it.
        """
        size = min(image.shape[1:3])
        image = (255*image.permute(1, 2, 0).numpy()).astype(np.uint8)
        image = Image.fromarray(image)

        this_dir = Path(__file__).resolve().parent
        title_font = ImageFont.truetype(str(this_dir / 'RobotoMono-Regular.ttf'), size//10)

        draw = ImageDraw.Draw(image)
        bb = (size//25, size//25, size//25+len(img_title)*size//15, size//25+size//8)
        draw.rectangle(bb, fill='black')
        draw.text((size//20, size//30), img_title, (237, 230, 211), font=title_font)

        image = np.array(image)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255

        return image

    def _append_images(  # noqa: C901
        self,
        images: Dict[str, List[torch.Tensor]],
        inputs: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor]
    ) -> None:
        """Append samples to the images accumulator.

        Parameters
        ----------
        images : Dict[str, List[torch.Tensor]]
            The accumulator where the samples will be appended to.
        inputs : Dict[str, torch.Tensor]
            The inputs of the model.
        preds : Dict[str, torch.Tensor]
            The outrputs of the model.
        """
        for k in self.log_keys:
            log_names = []
            log_sources = []
            if k in inputs or (k == 'confs' and k in preds):
                log_names.append(f'i_{k}')
                log_sources.append(inputs)
            if k in preds:
                log_names.append(f'o_{k}')
                log_sources.append(preds)

                if k == 'flows':
                    log_names.append(f'epe<{self.epe_clip:.1f}')
                    log_sources.append(None)

            for name, source in zip(log_names, log_sources):
                if images.get(name) is None:
                    images[name] = []

                if name == 'i_confs':
                    img = self._compute_confidence_gt(preds['flows'], inputs['flows'])
                elif name.startswith('epe'):
                    epe = torch.norm(preds[k] - inputs[k], p=2, dim=2, keepdim=True)
                    img = torch.clamp(epe, 0, self.epe_clip) / self.epe_clip
                    if inputs.get('valids') is not None:
                        img[inputs['valids'] < 0.5] = 0
                else:
                    img = source[k]

                img = img[:1, 0].detach().cpu()
                img = F.interpolate(img, self.image_size)
                img = img[0]

                if 'images' in name:
                    img = img.flip([0])  # BGR to RGB
                elif 'flows' in name:
                    img = flow_utils.flow_to_rgb(img)

                images[name].append(img)

    def _compute_confidence_gt(
        self,
        pred_flows: torch.Tensor,
        target_flows: torch.Tensor
    ) -> torch.Tensor:
        """Compute a confidence score for the flow predictions.

        This score was proposed in https://arxiv.org/abs/2007.09319.

        Parameters
        ----------
        pred_flows : torch.Tensor
            The predicted optical flow.
        target_flows : torch.Tensor
            The groundtruth optical flow.

        Returns
        -------
        torch.Tensor
            The confidence score for each pixel of the input.
        """
        conf_gt = torch.exp(-torch.pow(pred_flows - target_flows, 2).sum(dim=2, keepdim=True))
        return conf_gt

    def _compute_max_range(
        self,
        dataloader_length: int,
        limit_batches: Union[float, int]
    ) -> int:
        """Find the maximum number of samples that will be drawn from a dataloader.

        Parameters
        ----------
        dataloader_length : int
            Total size of the dataloader.
        limit_batches : Union[float, int]
            A value that may decrease the samples in the dataloader. See --limit_val_batches or --limit_train_batches from
            PyTorch Lightning for more information.

        Returns
        -------
        int
            The maximum number of samples that will be drawn from the dataloader.
        """
        if isinstance(limit_batches, int):
            max_range = limit_batches - 1
        else:
            max_range = int(limit_batches * dataloader_length) - 1
        return max_range

    def _make_image_grid(
        self,
        dl_images: Dict[str, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Transform a bunch of images into a single one by adding them to a grid.

        Parameters
        ----------
        dl_images : Dict[str, List[torch.Tensor]]
            Lists of images, each identified by a title name.

        Returns
        -------
        torch.Tensor
            A single 3D tensor image 3HW.
        """
        imgs = []
        for img_label, img_list in dl_images.items():
            for j, im in enumerate(img_list):
                if len(im.shape) == 2:
                    im = im[None]

                if im.shape[0] == 1:
                    im = im.repeat(3, 1, 1)

                if j == 0:
                    im = self._add_title(im, img_label)

                imgs.append(im)

        grid = make_grid(imgs, len(imgs)//len(dl_images))
        return grid
