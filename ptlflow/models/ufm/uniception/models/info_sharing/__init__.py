"""Info sharing modules of the vendored UniCeption subset."""

from .base import (
    MultiSetTransformerInput,
    MultiSetTransformerOutput,
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from .global_attention_transformer import (
    MultiViewGlobalAttentionTransformer,
    MultiViewGlobalAttentionTransformerIFR,
)

INFO_SHARING_CLASSES = {
    "global_attention": (
        MultiViewGlobalAttentionTransformer,
        MultiViewGlobalAttentionTransformerIFR,
    ),
}
