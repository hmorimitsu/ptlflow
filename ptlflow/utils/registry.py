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

import sys

import lightning.pytorch as pl

from ptlflow.models.base_model.base_model import BaseModel


_models_dict = {}
_trainable_models = []
_ptlflow_trained_models = []


class RegisteredModel(pl.LightningModule):
    pass


def register_model(model_class: BaseModel) -> BaseModel:
    # lookup containing module
    model_dir = ".".join(model_class.__module__.split(".")[:-1])
    mod = sys.modules[model_dir]
    # add model to __all__ in module
    model_name = model_class.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]  # type: ignore

    _models_dict[model_class.__name__] = model_class
    registered_class = type(model_class.__name__, (model_class, RegisteredModel), {})
    registered_class.__module__ = model_class.__module__
    return registered_class


def trainable(model_class: BaseModel) -> BaseModel:
    _trainable_models.append(model_class.__name__)
    return model_class


def ptlflow_trained(model_class: BaseModel) -> BaseModel:
    _ptlflow_trained_models.append(model_class.__name__)
    return model_class
