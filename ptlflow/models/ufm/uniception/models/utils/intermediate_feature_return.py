"""
Utils for Intermediate Feature Returner
References:
HuggingFace PyTorch Image Models (Timm)
"""

from typing import List, Optional, Tuple, Union

import torch

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


class IntermediateFeatureReturner:
    "Intermediate Feature Returner Class"

    def __init__(
        self,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        stop_early: bool = False,
        intermediates_only: bool = True,
    ):
        """
        Init class for returning intermediate features from the encoder.

        Args:
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. Defaults to None. Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. Defaults to True.
            stop_early (bool, optional): Whether to stop early. Defaults to False.
            intermediates_only (bool, optional): Whether to return only the intermediate features. Defaults to True.
        """
        self.indices = indices
        self.norm_intermediate = norm_intermediate
        self.stop_early = stop_early
        self.intermediates_only = intermediates_only


def feature_take_indices(
    num_features: int,
    indices: Optional[Union[int, List[int]]] = None,
    as_set: bool = False,
) -> Tuple[List[int], int]:
    """Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forwar() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        _assert(0 < indices <= num_features, f"last-n ({indices}) is out of range (1 to {num_features})")
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            _assert(0 <= idx < num_features, f"feature index {idx} is out of range (0 to {num_features - 1})")
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)
