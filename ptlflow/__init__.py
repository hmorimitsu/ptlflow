"""Provide useful functions for using PTLFlow."""

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

__version__ = "0.4.2"

from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonargparse import ArgumentParser, Namespace
from loguru import logger
import requests
import torch
from torch import hub

from ptlflow.models.base_model.base_model import BaseModel

from ptlflow.utils.registry import (
    _models_dict,
    _ptlflow_trained_models,
    _trainable_models,
)


def download_scripts(destination_dir: Path = Path("ptlflow_scripts")) -> None:
    """Download the main scripts and configs to start working with PTLFlow."""
    github_url = "https://raw.githubusercontent.com/hmorimitsu/ptlflow/main/"
    script_names = [
        "datasets.yaml",
        "infer.py",
        "model_benchmark.py",
        "test.py",
        "train.py",
        "validate.py",
    ]

    destination_dir.mkdir(parents=True, exist_ok=True)

    for sname in script_names:
        script_url = github_url + sname
        data = requests.get(script_url)
        if data.status_code == 200:
            with open(destination_dir / sname, "wb") as f:
                f.write(data.content)
        else:
            logger.warning("Script {} was not found.", script_url)

    logger.info("Downloaded scripts to {}.", str(destination_dir))


def get_model(
    model_name: str,
    ckpt_path: Optional[str] = None,
    args: Optional[Namespace] = None,
) -> BaseModel:
    """Return an instance of a chosen model.

    The instance can have configured by he arguments, and load some existing pretrained weights.

    Note that this is different from get_model_reference(), which returns a reference to the model class. The instance,
    returned by this function, is a class already instantiated. Therefore, the return of this function is equivalent to
    "return get_model_reference()()", which looks confusing. This can be rewritten as
    "model_ref = get_model_reference(); return model_ref()".

    Parameters
    ----------
    model_name : str
        Name of the model to get an instance of.
    ckpt_path : Optional[str], optional
        Name of the pretrained weight to load or a path to a local checkpoint file.
    args : Optional[Namespace], optional
        Some arguments that ill be provided to the model.

    Returns
    -------
    BaseModel
        The instance of the chosen model.

    Raises
    ------
    ValueError
        If the given checkpoint name is not a valid choice.
    ValueError
        If a checkpoint name is given, but the model does not have any pretrained weights available.

    See Also
    --------
    get_model_reference : To get a reference to the class of a model.
    """
    model_ref = get_model_reference(model_name)
    if args is None:
        parser = ArgumentParser()
        parser.add_class_arguments(model_ref, "model")
        args = parser.parse_args([])

    model_parser = ArgumentParser(exit_on_error=False)
    model_parser.add_argument("--model", type=model_ref)
    model_cfg = model_parser.parse_object({"model": args.model})
    model = model_parser.instantiate_classes(model_cfg).model

    if (
        ckpt_path is None
        and args is not None
        and hasattr(args, "ckpt_path")
        and args.ckpt_path is not None
    ):
        ckpt_path = args.ckpt_path

    model = restore_model(model, ckpt_path)

    return model


def get_model_reference(model_name: str) -> BaseModel:
    """Return a reference to the class of a chosen model.

    Note that this is different from get_model(), which returns an instance of a model. The reference, returned by this
    function, is a class before instantiation. Therefore, the return of this function can be used to instantiate a model as
    "model_ref = get_model_reference(); model_instance = model_ref()".

    Parameters
    ----------
    model_name : str
        Name of the model to get a reference of.

    Returns
    -------
    BaseModel
        A reference to the chosen model.

    Raises
    ------
    ValueError
        If the given name is not a valid choice.

    See Also
    --------
    get_model : To get an instance of a model.
    """
    try:
        return _models_dict[model_name]
    except KeyError:
        raise ValueError(
            f'Unknown model name: {model_name}. Choose from [{", ".join(_models_dict.keys())}]'
        )


def get_model_names() -> List[str]:
    """Return a list of all model names that are registered in this platform.

    Models are added to this list by decorating their classes with @ptlflow.utils.registry.register_model.

    Returns
    -------
    List[str]
        The list of the all registered model names.
    """
    return sorted(list(_models_dict.keys()))


def get_trainable_model_names() -> List[str]:
    """Return a list of model names that are able to be trained.

    This function return the names of the model that have a loss function defined.

    Returns
    -------
    List[str]
        The list of the model names that can be trained.
    """
    return _trainable_models


def get_ptlflow_trained_model_names() -> List[str]:
    """Return a list of model names that have been trained on PTLFlow.

    This function return the names of the model that have set model.has_trained_on_ptlflow = True.

    Returns
    -------
    List[str]
        The list of the model names that has been trained on PTLFlow.
    """
    return _ptlflow_trained_models


def load_checkpoint(ckpt_path: str, model_ref: BaseModel) -> Dict[str, Any]:
    """Try to load the checkpoint specified in ckpt_path.

    Parameters
    ----------
    ckpt_path : str
        Path to a local file or name of a pretrained checkpoint.
    model_ref : BaseModel
        A reference to the model class. See the function get_model_reference() for more details.

    Returns
    -------
    Dict[str, Any]
        A dictionary of the loaded checkpoint. The output of torch.load().

    See Also
    --------
    get_model_reference : To get a reference to the class of a model.
    """
    if Path(ckpt_path).exists():
        ckpt_path = ckpt_path
    elif hasattr(model_ref, "pretrained_checkpoints"):
        tmp_ckpt_path = model_ref.pretrained_checkpoints.get(ckpt_path)
        if tmp_ckpt_path is None:
            raise ValueError(
                f"Invalid checkpoint name {ckpt_path}. "
                f'Choose one from {{{",".join(model_ref.pretrained_checkpoints.keys())}}}'
            )
        else:
            ckpt_path = tmp_ckpt_path
    else:
        raise ValueError(
            f"Cannot find checkpoint {ckpt_path} for model {model_ref.__name__}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if Path(ckpt_path).exists():
        ckpt = torch.load(
            ckpt_path, map_location=torch.device(device), weights_only=True
        )
    else:
        model_dir = Path(hub.get_dir()) / "checkpoints"
        ckpt = hub.load_state_dict_from_url(
            ckpt_path,
            model_dir=model_dir,
            map_location=torch.device(device),
            check_hash=True,
            weights_only=True,
        )
    return ckpt


def restore_model(model, ckpt_path):
    """Load model state from the checkpoint.

    Parameters
    ----------
    model : BaseModel
        An instance of the model to be restored.
    ckpt_path : str
        Path to a local file or name of a pretrained checkpoint.

    Returns
    -------
    BaseModel
        An instance of the restored model.
    """
    if ckpt_path is not None:
        ckpt = load_checkpoint(ckpt_path, model.__class__)

        state_dict = ckpt["state_dict"]
        if "hyper_parameters" in ckpt:
            if "train_size" in ckpt["hyper_parameters"]:
                model.train_size = ckpt["hyper_parameters"]["train_size"]
            if "train_avg_length" in ckpt["hyper_parameters"]:
                model.train_avg_length = ckpt["hyper_parameters"]["train_avg_length"]
            if "extra_params" in ckpt["hyper_parameters"]:
                extra_params = ckpt["hyper_parameters"]["extra_params"]
                for name, value in extra_params.items():
                    model.add_extra_param(name, value)
        model.load_state_dict(state_dict)
        logger.info("Restored model state from checkpoint: {}", ckpt_path)

    return model
