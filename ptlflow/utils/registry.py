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

import inspect
import sys
from typing import Any, Dict, Type

import lightning.pytorch as pl
from jsonargparse import ArgumentParser

from ptlflow.models.base_model.base_model import BaseModel

_models_dict = {}
_trainable_models = []
_ptlflow_trained_models = []
_model_variants = set()

# Resolved __init__ parameters of the base classes, cached so that generating
# several variants of one family does not repeat the signature resolution.
_variant_base_params = {}


class RegisteredModel(pl.LightningModule):
    pass


def register_model(model_class: Type[BaseModel]) -> Type[BaseModel]:
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


def is_model_variant(model_name: str) -> bool:
    """Check whether a model name refers to a generated preset variant.

    Variants are created with register_model_variants(). They are resolved like any
    other model by ptlflow.get_model() and by the --model argument, but they are
    hidden from ptlflow.get_model_names() unless include_variants=True is given.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    bool
        True if the name refers to a generated variant model.
    """
    return model_name in _model_variants


def register_model_variants(
    base_classes: Dict[str, Type[BaseModel]],
    presets: Dict[str, Dict[str, Any]],
) -> None:
    """Register one preset variant model for each (base model, preset) combination.

    For each name in base_classes and each entry of presets, a model class named
    <base_name>_<preset_suffix> is generated. The variant subclasses base_classes[base_name]
    and presets its __init__ with the given keyword arguments (which can still be overridden
    by explicitly passing them). The generated classes are registered like regular models,
    so they can be selected by name anywhere (ptlflow.get_model(), --model argument), but
    they are tagged as variants and hidden from ptlflow.get_model_names() by default.

    Parameters
    ----------
    base_classes : Dict[str, Type[BaseModel]]
        Maps registered model names to the base classes that the variants subclass.
    presets : Dict[str, Dict[str, Any]]
        Maps the variant name suffixes to the __init__ keyword arguments preset by the
        variant. For example, {"localcorr_triton": {"corr_mode": "localcorr_triton"}}
        registers <base_name>_localcorr_triton models.
    """
    for base_name, base_class in base_classes.items():
        for suffix, preset_kwargs in presets.items():
            variant_name = f"{base_name}_{suffix}"
            if variant_name in _models_dict:
                raise ValueError(f"Model name {variant_name} is already registered.")

            def _make_init(base_class, kwargs_preset):
                def __init__(self, **kwargs):
                    for key, value in kwargs_preset.items():
                        kwargs.setdefault(key, value)
                    base_class.__init__(self, **kwargs)

                # jsonargparse and inspect cannot resolve the **kwargs of a
                # generated function against the base class signature, so the
                # parameters are exposed explicitly through __signature__: the
                # same parameters resolved for the base class, with the preset
                # values as defaults.
                if base_class not in _variant_base_params:
                    parser = ArgumentParser()
                    parser.add_class_arguments(base_class, "model")
                    args = parser.parse_args([])
                    _variant_base_params[base_class] = (
                        dict(args.model) if args.model else {}
                    )
                parameters = []
                seen = set()
                for name, default in _variant_base_params[base_class].items():
                    if name in kwargs_preset:
                        default = kwargs_preset[name]
                    parameters.append(
                        inspect.Parameter(
                            name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            default=default,
                        )
                    )
                    seen.add(name)
                for name, default in kwargs_preset.items():
                    if name not in seen:
                        parameters.append(
                            inspect.Parameter(
                                name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=default,
                            )
                        )
                __init__.__signature__ = inspect.Signature(parameters)
                return __init__

            variant_class = type(
                variant_name,
                (base_class,),
                {
                    "__init__": _make_init(base_class, preset_kwargs),
                    "__module__": base_class.__module__,
                    "__doc__": (
                        f"{base_name} with the preset arguments {preset_kwargs}.\n\n"
                        "This is a generated model variant (see "
                        "ptlflow.utils.registry.register_model_variants)."
                    ),
                },
            )
            registered_class = register_model(variant_class)
            # Make the variant importable from the module that defines the family,
            # so it is found by the --model argument subclass resolution.
            setattr(sys.modules[base_class.__module__], variant_name, registered_class)
            _model_variants.add(variant_name)


def trainable(model_class: Type[BaseModel]) -> Type[BaseModel]:
    _trainable_models.append(model_class.__name__)
    return model_class


def ptlflow_trained(model_class: Type[BaseModel]) -> Type[BaseModel]:
    _ptlflow_trained_models.append(model_class.__name__)
    return model_class
