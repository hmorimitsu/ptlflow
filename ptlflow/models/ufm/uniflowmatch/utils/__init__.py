"""Utilities subpackage for the vendored `uniflowmatch` (from https://github.com/UniFlowMatch/UFM).

Only the utilities required for inference are kept from the original repository.
"""

from .flow_resizing import (
    AutomaticShapeSelection,
    ResizeToFixedManipulation,
    unmap_predicted_channels,
    unmap_predicted_flow,
)

__all__ = [
    "AutomaticShapeSelection",
    "ResizeToFixedManipulation",
    "unmap_predicted_channels",
    "unmap_predicted_flow",
]
