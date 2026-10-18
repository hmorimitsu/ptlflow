"""Encoders of the vendored UniCeption subset.

Only the DINOv2 encoder (used by UFM) is provided here. Contrary to the original UniCeption code,
the DINOv2 architecture is instantiated locally (no torch.hub download). The weights are expected
to come from the UFM checkpoints, which already include the DINOv2 weights.
"""

from .base import (
    EncoderGlobalRepInput,
    EncoderInput,
    UniCeptionEncoderBase,
    UniCeptionViTEncoderBase,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
    ViTEncoderOutput,
)
from .dinov2 import DINOv2Encoder, DINOv2IntermediateFeatureReturner

# Define encoder configurations
ENCODER_CONFIGS = {
    "dinov2": {
        "class": DINOv2Encoder,
        "intermediate_feature_returner_class": DINOv2IntermediateFeatureReturner,
        "supported_models": ["DINOv2", "DINOv2-Registers"],
    },
}


def feature_returner_encoder_factory(encoder_str: str, **kwargs) -> UniCeptionEncoderBase:
    """Factory for the vendored UniCeption encoders with support for intermediate feature returning.

    Args:
        encoder_str (str): Name of the encoder to create.
        **kwargs: Additional keyword arguments to pass to the encoder constructor.

    Returns:
        UniCeptionEncoderBase: An instance of the specified encoder.
    """
    if encoder_str not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_str}. The vendored UniCeption subset only supports: {list(ENCODER_CONFIGS.keys())}"
        )

    encoder_config = ENCODER_CONFIGS[encoder_str]
    encoder_class = encoder_config["intermediate_feature_returner_class"]

    return encoder_class(**kwargs)


__all__ = [
    "feature_returner_encoder_factory",
    "UniCeptionEncoderBase",
    "UniCeptionViTEncoderBase",
    "EncoderInput",
    "ViTEncoderInput",
    "ViTEncoderOutput",
]
