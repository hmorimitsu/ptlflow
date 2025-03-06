# =============================================================================
# Copyright 2024 Henrique Morimitsu
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

from datetime import datetime
from pathlib import Path
import sys

from jsonargparse import ArgumentParser
from loguru import logger

from ptlflow.data.flow_datamodule import FlowDataModule
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI
from ptlflow.utils.lightning.ptlflow_trainer import PTLFlowTrainer
from ptlflow.utils.registry import RegisteredModel


def _init_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wdecay", type=float, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--project", type=str, default="ptlflow")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--train_ckpt_topk", type=int, default=0)
    parser.add_argument("--train_ckpt_metric", type=str, default="train/loss_epoch")
    parser.add_argument("--infer_ckpt_topk", type=int, default=1)
    parser.add_argument("--infer_ckpt_metric", type=str, default="main_val_metric")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--log_dir", type=str, default="ptlflow_logs")
    return parser


def cli_main():
    parser = _init_parser()

    cfg = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        datamodule_class=FlowDataModule,
        trainer_class=PTLFlowTrainer,
        auto_configure_optimizers=False,
        parser_kwargs={"parents": [parser]},
        run=False,
        parse_only=True,
    ).config

    model_name = cfg.model.class_path.split(".")[-1]

    # Setup loggers and callbacks

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    log_model_name = (
        f"{model_name}-{_gen_dataset_id(cfg.data.train_dataset)}-{timestamp}"
    )
    log_model_dir = Path(cfg.log_dir) / log_model_name
    log_model_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logger == "tensorboard":
        trainer_logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": str(log_model_dir),
                "name": f"{model_name}-{cfg.data.train_dataset}",
                "version": cfg.version,
            },
        }
    elif cfg.logger == "wandb":
        trainer_logger = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": {
                "save_dir": str(log_model_dir),
                "project": cfg.project,
                "version": cfg.version,
                "name": f"{model_name}-{cfg.data.train_dataset}",
            },
        }

    callbacks = []

    callbacks.append(
        {
            "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {
                "filename": model_name + "_last_{epoch}_{step}",
                "save_weights_only": True,
                "mode": "max",
            },
        }
    )

    callbacks.append(
        {
            "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
            "init_args": {
                "filename": model_name + "_train_{epoch}_{step}",
            },
        }
    )

    if cfg.train_ckpt_topk > 0:
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
                "init_args": {
                    "filename": model_name
                    + "_{"
                    + cfg.train_ckpt_metric
                    + ":.2f}_{epoch}",
                    "save_weights_only": False,
                    "save_top_k": cfg.train_ckpt_topk,
                    "monitor": "train/loss_epoch",
                },
            }
        )

    if cfg.infer_ckpt_topk > 0:
        assert (
            cfg.infer_ckpt_metric is not None
        ), "You must provide a metric name for --infer_ckpt_topk_metric"
        callbacks.append(
            {
                "class_path": "lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint",
                "init_args": {
                    "filename": model_name
                    + "_best_{"
                    + cfg.infer_ckpt_metric
                    + ":.2f}_{epoch}_{step}",
                    "save_weights_only": True,
                    "save_top_k": cfg.infer_ckpt_topk,
                    "monitor": cfg.infer_ckpt_metric,
                },
            }
        )

    callbacks.append(
        {
            "class_path": "ptlflow.utils.callbacks.logger.LoggerCallback",
        }
    )

    cfg.trainer.logger = trainer_logger
    cfg.trainer.callbacks = callbacks
    cfg.model.init_args.lr = cfg.lr
    cfg.model.init_args.wdecay = cfg.wdecay
    cli = PTLFlowCLI(
        model_class=RegisteredModel,
        subclass_mode_model=True,
        trainer_class=PTLFlowTrainer,
        datamodule_class=FlowDataModule,
        auto_configure_optimizers=False,
        args=cfg,
        run=False,
        ignore_sys_argv=True,
        parser_kwargs={"parents": [parser]},
    )

    if not cli.model.has_trained_on_ptlflow:
        _print_untested_warning()

    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=cfg.ckpt_path)


def _gen_dataset_id(dataset_string: str) -> str:
    sep_datasets = dataset_string.split("+")
    names_list = []
    for dataset in sep_datasets:
        if "*" in dataset:
            tokens = dataset.split("*")
            try:
                _, dataset_params = int(tokens[0]), tokens[1]
            except ValueError:  # the multiplier is at the end
                dataset_params = tokens[0]
        else:
            dataset_params = dataset

        dataset_name = dataset_params.split("-")[0]
        names_list.append(dataset_name)

    dataset_id = "_".join(names_list)
    return dataset_id


def _print_untested_warning():
    print("###########################################################################")
    print("# WARNING, please read!                                                   #")
    print("#                                                                         #")
    print("# This training script has not been tested for this model!                #")
    print("# Therefore, there is no guarantee that training it with this script      #")
    print("# will produce good results!                                              #")
    print("#                                                                         #")
    print("# You can find more information at                                        #")
    print("# https://ptlflow.readthedocs.io/en/latest/starting/training.html         #")
    print("###########################################################################")


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

    cli_main()
