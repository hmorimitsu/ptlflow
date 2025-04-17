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
import sys
from typing import Any, Callable, Dict, Optional, Type, Union

from jsonargparse import Namespace
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class PTLFlowCLI(LightningCLI):
    parser_class = LightningArgumentParser

    def __init__(
        self,
        model_class: Optional[
            Union[Type[LightningModule], Callable[..., LightningModule]]
        ] = None,
        datamodule_class: Optional[
            Union[Type[LightningDataModule], Callable[..., LightningDataModule]]
        ] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Optional[Union[Type[Trainer], Callable[..., Trainer]]] = None,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[
            Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        parse_only: bool = False,
        ignore_sys_argv: bool = False,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which are
        called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``parser_kwargs={"default_env":
        True}``. A full configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed
        from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        Args:
            model_class: An optional :class:`~lightning.pytorch.core.LightningModule` class to train on or a
                callable which returns a :class:`~lightning.pytorch.core.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~lightning.pytorch.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~lightning.pytorch.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning.fabric.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~lightning.pytorch.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            parse_only: If set to ``True``, the CLI acts as a parser only. The classes are not instantiated.
        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}
        self.auto_configure_optimizers = auto_configure_optimizers

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class
        self.subclass_mode_model = subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args, ignore_sys_argv)

        if not parse_only:
            self.subcommand = self.config["subcommand"] if run else None

            self._set_seed()

            self._add_instantiators()
            self.before_instantiate_classes()
            self.instantiate_classes()

            if self.subcommand is not None:
                self._run_subcommand(self.subcommand)

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        if self.trainer_class is not None:
            parser.add_lightning_class_args(self.trainer_class, "trainer")
            trainer_defaults = {
                "trainer." + k: v
                for k, v in self.trainer_defaults.items()
                if k != "callbacks"
            }
            parser.set_defaults(trainer_defaults)

        if self._model_class is not None:
            parser.add_lightning_class_args(
                self._model_class, "model", subclass_mode=self.subclass_mode_model
            )

        if self.datamodule_class is not None:
            parser.add_lightning_class_args(
                self._datamodule_class, "data", subclass_mode=self.subclass_mode_data
            )

    def parse_arguments(
        self, parser: LightningArgumentParser, args: ArgsType, ignore_sys_argv: bool
    ) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if args is not None and len(sys.argv) > 1 and not ignore_sys_argv:
            rank_zero_warn(
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)

        if self.trainer_class is not None:
            self.trainer = self.instantiate_trainer()
