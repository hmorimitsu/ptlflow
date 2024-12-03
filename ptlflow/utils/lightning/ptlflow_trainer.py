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

# THIS FILE MUST READ EASILY, FOR UNDERSTANDING AND DEBUGGING PURPOSES.
# DO NOT OBSCURE THE TRAINING LOOP
# THIS IS A HARD REQUIREMENT TO CONTRIBUTING TO LIGHTNING
# WE FAVOR READABILITY OVER ENGINEERING-CONSTRUCTS BY DESIGN
# DO NOT REMOVE THIS NOTICE
# - WILLIAM FALCON
"""Trainer to automate the training."""

import logging
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Union
from weakref import proxy

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.utilities import _log_hyperparams
from lightning.pytorch.loops.utilities import _parse_loop_limits
from lightning.pytorch.plugins import _PLUGIN_INPUT
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.configuration_validator import (
    _verify_loop_configurations,
)
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
)
from lightning.pytorch.trainer.states import (
    TrainerFn,
    TrainerStatus,
)
from lightning.pytorch.utilities import parsing
from lightning.pytorch.utilities.argparse import _defaults_from_env_vars
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
)

from .ptlflow_checkpoint_connector import (
    _PTLFlowCheckpointConnector,
)

log = logging.getLogger(__name__)


class PTLFlowTrainer(pl.Trainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
    ) -> None:
        r"""Customize every aspect of training via flags.

        Args:
            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "hpu", "mps", "auto")
                as well as custom accelerator instances.

            strategy: Supports different training strategies with aliases as well custom strategies.
                Default: ``"auto"``.

            devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
                (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
                automatic selection based on the chosen accelerator. Default: ``"auto"``.

            num_nodes: Number of GPU nodes for distributed training.
                Default: ``1``.

            precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
                16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
                Can be used on CPU, GPU, TPUs, or HPUs.
                Default: ``'32-true'``.

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
                ``False`` will disable logging. If multiple loggers are provided, local files
                (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
                Default: ``True``.

            callbacks: Add a callback or list of callbacks.
                Default: ``None``.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (``None``).

            max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
                Default: ``0.0``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during iteration-based training.
                Default: ``1.0``.

            check_val_every_n_epoch: Perform a validation loop after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.

            log_every_n_steps: How often to log within steps.
                Default: ``50``.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
                Default: ``True``.

            enable_progress_bar: Whether to enable to progress bar by default.
                Default: ``True``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

            accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
                Default: 1.

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
                that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.

            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic`
                is set to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.

            inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
                evaluation (``validate``/``test``/``predict``).

            use_distributed_sampler: Whether to wrap the DataLoader's sampler with
                :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
                strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
                ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
                ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
                sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
                we don't do this automatically.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            barebones: Whether to run in "barebones mode", where all features that may impact raw speed are
                disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
                runs. The following features are deactivated:
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`,
                :meth:`~lightning.pytorch.core.LightningModule.log`,
                :meth:`~lightning.pytorch.core.LightningModule.log_dict`.
            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
                Default: ``None``.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
                Default: ``False``.

            reload_dataloaders_every_n_epochs: Set to a positive integer to reload dataloaders every n epochs.
                Default: ``0``.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

        Raises:
            TypeError:
                If ``gradient_clip_val`` is not an int or float.

            MisconfigurationException:
                If ``gradient_clip_algorithm`` is invalid.

        """
        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
        )
        self._checkpoint_connector = _PTLFlowCheckpointConnector(self)

    def _run(
        self, model: "pl.LightningModule", ckpt_path: Optional[_PATH] = None
    ) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        if self.state.fn == TrainerFn.FITTING:
            min_epochs, max_epochs = _parse_loop_limits(
                self.min_steps, self.max_steps, self.min_epochs, self.max_epochs, self
            )
            self.fit_loop.min_epochs = min_epochs
            self.fit_loop.max_epochs = max_epochs

        if self.barebones:
            # no progress bar in barebones can make it look like the Trainer hung
            rank_zero_info(
                "`Trainer(barebones=True)` started running. The progress bar is disabled so you might want to"
                " manually print the progress in your model."
            )

        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # attach model to the strategy
        self.strategy.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        _verify_loop_configurations(self)

        # ----------------------------
        # SET UP THE TRAINER
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: setting up strategy environment")
        self.strategy.setup_environment()
        self.__setup_profiler()

        log.debug(f"{self.__class__.__name__}: preparing data")
        self._data_connector.prepare_data()

        call._call_setup_hook(
            self
        )  # allow user to set up LightningModule in accelerator environment
        log.debug(f"{self.__class__.__name__}: configuring model")
        call._call_configure_model(self)

        # check if we should delay restoring checkpoint till later
        if not self.strategy.restore_checkpoint_after_setup:
            log.debug(
                f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}"
            )
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path, model)

        # reset logger connector
        self._logger_connector.reset_results()
        self._logger_connector.reset_metrics()

        # strategy will configure model and move it to the device
        self.strategy.setup(self)

        # hook
        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_start")
            call._call_lightning_module_hook(self, "on_fit_start")

        _log_hyperparams(self)

        if self.strategy.restore_checkpoint_after_setup:
            log.debug(
                f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}"
            )
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path)

        # restore optimizers, etc.
        try:
            log.debug(f"{self.__class__.__name__}: restoring training state")
            self._checkpoint_connector.restore_training_state()
        except KeyError:
            log.info(
                "The provided checkpoint does not contain the training state. Only the model weights will be loaded."
            )

        self._checkpoint_connector.resume_end()

        self._signal_connector.register_signal_handlers()

        # ----------------------------
        # RUN THE TRAINER
        # ----------------------------
        results = self._run_stage()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: trainer tearing down")
        self._teardown()

        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_end")
            call._call_lightning_module_hook(self, "on_fit_end")

        log.debug(f"{self.__class__.__name__}: calling teardown hooks")
        call._call_teardown_hook(self)

        self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return results

    def __setup_profiler(self) -> None:
        assert self.state.fn is not None
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(
            stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir
        )
