"""Train one of the available models."""

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

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # Workaround until pl stop raising the metrics deprecation warning
    import pytorch_lightning as pl
import torch

from ptlflow import get_model, get_model_reference
from ptlflow.utils.callbacks.logger import LoggerCallback
from ptlflow.utils.utils import add_datasets_to_parser, get_list_of_available_models_list


def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        'model', type=str, choices=get_list_of_available_models_list(),
        help='Name of the model to use.')
    parser.add_argument(
        '--random_seed', type=int, default=1234,
        help='A number to seed the pseudo-random generators.')
    parser.add_argument(
        '--clear_train_state', action='store_true',
        help=('Only used if --resume_from_checkpoint is not None. If set, only the weights are loaded from the checkpoint '
              'and the training state is ignored. Set it when you want to finetune the model from a previous checkpoint.'))
    parser.add_argument(
        '--log_dir', type=str, default='ptlflow_logs',
        help='The path to the directory where the logs will be saved.')
    return parser


def train(args: Namespace) -> None:
    """Run the training.

    Parameters
    ----------
    args : Namespace
        Arguments to configure the training.
    """
    _print_untested_warning()

    pl.utilities.seed.seed_everything(args.random_seed)

    if args.train_transform_cuda:
        from torch.multiprocessing import set_start_method
        set_start_method('spawn')

    if args.train_dataset is None:
        args.train_dataset = 'chairs-train'
        print('INFO: --train_dataset is not set. It will be set to "chairs-train"')

    log_model_name = f'{args.model}-{_gen_dataset_id(args.train_dataset)}'

    model = get_model(args.model, args.pretrained_ckpt, args)

    if args.resume_from_checkpoint is not None and args.clear_train_state:
        # Restore model weights, but not the train state
        pl_ckpt = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(pl_ckpt['state_dict'])
        args.resume_from_checkpoint = None

    # Setup loggers and callbacks

    callbacks = []

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_logger)

    log_model_dir = str(Path(args.log_dir) / log_model_name)
    tb_logger = pl.loggers.TensorBoardLogger(log_model_dir)

    model.val_dataloader()  # Called just to populate model.val_dataloader_names

    model_ckpt_last = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model+'_last_{epoch}_{step}', save_weights_only=True)
    callbacks.append(model_ckpt_last)
    model_ckpt_train = pl.callbacks.model_checkpoint.ModelCheckpoint(filename=args.model+'_train_{epoch}_{step}')
    callbacks.append(model_ckpt_train)

    if len(model.val_dataloader_names) > 0:
        model_ckpt_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
            filename=args.model+'_best_{'+model.val_dataloader_names[0]+':.2f}_{epoch}_{step}', save_weights_only=True,
            save_top_k=1, monitor=model.val_dataloader_names[0])
        callbacks.append(model_ckpt_best)

    callbacks.append(LoggerCallback())

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=callbacks)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Workaround until pl stop the LightningModule.datamodule` property warning
        trainer.tune(model)
        trainer.fit(model)


def _gen_dataset_id(
    dataset_string: str
) -> str:
    sep_datasets = dataset_string.split('+')
    names_list = []
    for dataset in sep_datasets:
        if '*' in dataset:
            tokens = dataset.split('*')
            try:
                _, dataset_params = int(tokens[0]), tokens[1]
            except ValueError:  # the multiplier is at the end
                dataset_params = tokens[0]
        else:
            dataset_params = dataset

        dataset_name = dataset_params.split('-')[0]
        names_list.append(dataset_name)

    dataset_id = '_'.join(names_list)
    return dataset_id


def _print_untested_warning():
    print('###############################################################################')
    print('# WARNING, please read!                                                       #')
    print('#                                                                             #')
    print('# This training script has not been tested!                                   #')
    print('# Therefore, there is no guarantee that a model trained with this script      #')
    print('# will produce good results after the training!                               #')
    print('#                                                                             #')
    print('# You can find more information at                                            #')
    print('# https://ptlflow.readthedocs.io/en/latest/starting/training.html             #')
    print('###############################################################################')


if __name__ == '__main__':
    parser = _init_parser()

    # TODO: It is ugly that the model has to be gotten from the argv rather than the argparser.
    # However, I do not see another way, since the argparser requires the model to load some of the args.
    FlowModel = None
    if len(sys.argv) > 1 and sys.argv[1] != '-h' and sys.argv[1] != '--help':
        FlowModel = get_model_reference(sys.argv[1])
        parser = FlowModel.add_model_specific_args(parser)

    add_datasets_to_parser(parser, 'datasets.yml')

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    train(args)
