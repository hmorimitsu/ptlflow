"""Models subpackage for the vendored `uniflowmatch` (from https://github.com/UniFlowMatch/UFM)."""

from .base import (
    UFMClassificationRefinementOutput,
    UFMFlowFieldOutput,
    UFMMaskFieldOutput,
    UFMOutputInterface,
    UniFlowMatchModelsBase,
)
from .ufm import (
    UniFlowMatch,
    UniFlowMatchClassificationRefinement,
    UniFlowMatchConfidence,
)

__all__ = [
    "UFMClassificationRefinementOutput",
    "UFMFlowFieldOutput",
    "UFMMaskFieldOutput",
    "UFMOutputInterface",
    "UniFlowMatchModelsBase",
    "UniFlowMatch",
    "UniFlowMatchClassificationRefinement",
    "UniFlowMatchConfidence",
]
