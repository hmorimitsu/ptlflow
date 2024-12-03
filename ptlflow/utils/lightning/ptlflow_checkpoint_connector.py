# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from pathlib import Path
import re
from typing import Optional

import torch
from torch import hub
from fsspec.core import url_to_fs

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info

log = logging.getLogger(__name__)


class _PTLFlowCheckpointConnector(_CheckpointConnector):
    def __init__(self, trainer: pl.Trainer) -> None:
        super().__init__(trainer)

    def resume_start(
        self,
        checkpoint_path: Optional[_PATH] = None,
        model: Optional[pl.LightningModule] = None,
    ) -> None:
        """Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:

        1. from HPC weights if `checkpoint_path` is ``None`` and on SLURM or passed keyword `"hpc"`.
        2. from fault-tolerant auto-saved checkpoint if found
        3. from `checkpoint_path` file if provided
        4. don't restore

        """
        self._ckpt_path = checkpoint_path
        if not checkpoint_path:
            log.debug("`checkpoint_path` not specified. Skipping checkpoint loading.")
            return

        if not Path(checkpoint_path).exists():
            if model is not None:
                model_ref = model.__class__
                if hasattr(model_ref, "pretrained_checkpoints"):
                    checkpoint_path = model_ref.pretrained_checkpoints.get(
                        checkpoint_path
                    )
                    if checkpoint_path is None:
                        raise ValueError(
                            f"Invalid checkpoint name {checkpoint_path}. "
                            f'Choose one from {{{",".join(model_ref.pretrained_checkpoints.keys())}}}'
                        )

                    cache_path = (
                        Path(hub.get_dir())
                        / "checkpoints"
                        / checkpoint_path.split("/")[-1]
                    )
                    if cache_path.exists():
                        checkpoint_path = cache_path
            else:
                raise ValueError(
                    f"Cannot find checkpoint {checkpoint_path} for model {model.__class__.__name__}"
                )

        rank_zero_info(
            f"Restoring states from the checkpoint path at {checkpoint_path}"
        )
        with pl_legacy_patch():
            loaded_checkpoint = self.trainer.strategy.load_checkpoint(checkpoint_path)
        if not "pytorch-lightning_version" in loaded_checkpoint:
            loaded_checkpoint["pytorch-lightning_version"] = "1.9.5"
        self._loaded_checkpoint = _pl_migrate_checkpoint(
            loaded_checkpoint, checkpoint_path
        )

    def resume_end(self) -> None:
        """Signal the connector that all states have resumed and memory for the checkpoint object can be released."""
        assert self.trainer.state.fn is not None
        if self._ckpt_path:
            message = (
                "Restored all states"
                if self.trainer.state.fn == TrainerFn.FITTING
                else "Loaded model weights"
            )
            rank_zero_info(f"{message} from the checkpoint at {self._ckpt_path}")

        # free memory
        self._loaded_checkpoint = {}
        torch.cuda.empty_cache()

        # wait for all to catch up
        self.trainer.strategy.barrier("_PTLFlowCheckpointConnector.resume_end")

    def restore_training_state(self) -> None:
        """Restore the trainer state from the pre-loaded checkpoint.

        This includes the precision settings, loop progress, optimizer states and learning rate scheduler states.

        Modifications by Henrique Morimitsu:
        - restore_optimizers_and_schedulers before other states to raise an error earlier in case the ckpt does not have training states
        """
        if not self._loaded_checkpoint:
            return

        assert self.trainer.state.fn is not None
        if self.trainer.state.fn == TrainerFn.FITTING:
            # restore optimizers and schedulers state
            self.restore_optimizers_and_schedulers()

        # restore precision plugin (scaler etc.)
        self.restore_precision_plugin_state()

        # restore loops and their progress
        self.restore_loops()

    def _restore_modules_and_callbacks(
        self,
        checkpoint_path: Optional[_PATH] = None,
        model: Optional[pl.LightningModule] = None,
    ) -> None:
        # restore modules after setup
        self.resume_start(checkpoint_path, model)
        self.restore_model()
        self.restore_datamodule()
        if self.trainer.state.fn == TrainerFn.FITTING:
            # restore callback states
            self.restore_callbacks()

    @staticmethod
    def __max_ckpt_version_in_folder(
        dir_path: _PATH, name_key: str = "ckpt_"
    ) -> Optional[int]:
        """List up files in `dir_path` with `name_key`, then yield maximum suffix number.

        Args:
            dir_path: path of directory which may contain files whose name include `name_key`
            name_key: file name prefix
        Returns:
            None if no-corresponding-file else maximum suffix number

        """
        # check directory existence
        fs, uri = url_to_fs(str(dir_path))
        if not fs.exists(dir_path):
            return None

        # check corresponding file existence
        files = [os.path.basename(f["name"]) for f in fs.listdir(uri)]
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return None

        # extract suffix number
        ckpt_vs = []
        for name in files:
            name = name.split(name_key)[-1]
            name = re.sub("[^0-9]", "", name)
            ckpt_vs.append(int(name))

        return max(ckpt_vs)

    @staticmethod
    def __get_max_ckpt_path_from_folder(folder_path: _PATH) -> str:
        """Get path of maximum-epoch checkpoint in the folder."""
        max_suffix = _PTLFlowCheckpointConnector.__max_ckpt_version_in_folder(
            folder_path
        )
        ckpt_number = max_suffix if max_suffix is not None else 0
        return f"{folder_path}/hpc_ckpt_{ckpt_number}.ckpt"

    @staticmethod
    def hpc_save_path(folderpath: _PATH) -> str:
        max_suffix = _PTLFlowCheckpointConnector.__max_ckpt_version_in_folder(
            folderpath
        )
        ckpt_number = (max_suffix if max_suffix is not None else 0) + 1
        return os.path.join(folderpath, f"hpc_ckpt_{ckpt_number}.ckpt")
