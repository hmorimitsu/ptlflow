"""
Base Encoder Class for UniCeption
"""

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@dataclass
class EncoderInput:
    "Data class for Encoder Input"

    data_norm_type: str
    # Add other fields that are required by the specific implementation of the encoder.


@dataclass
class EncoderOutput:
    "Data class for Encoder Output"

    pass


@dataclass
class EncoderGlobalRepInput:
    "Data class for Encoder Global Representation Input"

    data: Tensor


@dataclass
class EncoderGlobalRepOutput:
    "Data class for Encoder Global Representation Output"

    features: Tensor


class UniCeptionEncoderBase(nn.Module):
    "Encoder Base Class for UniCeption"

    def __init__(
        self,
        name: str,
        data_norm_type: str,
        size: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Base class for all encoders in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.size: Optional[str] = size

        self.data_norm_type: str = data_norm_type

    def forward(
        self,
        encoder_input: EncoderInput,
    ) -> EncoderOutput:
        """
        Forward interface for the UniCeption encoders.

        We expect the "data_norm_type" field to be present in the encoder_input to check for normalization type.

        Args:
            encoder_input (EncoderInput): Input to the encoder. We expect the following fields: "data_norm_type: str".
                This is also includes the other fields that are required by the specific implementation of the encoder.

        Returns:
            EncoderOutput: Output of the encoder.
        """

        raise NotImplementedError

    def _check_data_normalization_type(self, data_norm_type: str):
        """
        Check if the input normalization type matches the encoder's expected input data normalization type.

        Args:
            data_norm_type (str): Data normalization type.

        Raises:
            AssertionError: If the data normalization type does not match the encoder's expected input data normalization type.
        """

        assert (
            data_norm_type == self.data_norm_type
        ), f"Input normalization type {data_norm_type} does not match the encoder's normalization type {self.data_norm_type}."


@dataclass
class ViTEncoderInput(EncoderInput):
    "Data class for Vision Transformer Encoder Input"

    image: Tensor


@dataclass
class ViTEncoderNonImageInput:
    "Data class for Vision (2D-Grid) Transformer Encoder Non-Image Input"

    data: Tensor


@dataclass
class ViTEncoderOutput(EncoderOutput):
    "Data class for Vision Transformer Encoder Output"

    features: Tensor
    registers: Optional[Tensor] = None


class UniCeptionViTEncoderBase(UniCeptionEncoderBase):
    "Vision Transformer Encoder Base Class for UniCeption"

    def __init__(
        self,
        patch_size: int,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Base class for all Vision Transformer encoders in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        """
        Wrapper for Gradient Checkpointing
        References: https://github.com/microsoft/MoGe
        """

        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module


if __name__ == "__main__":
    dummy_model = UniCeptionEncoderBase(name="name", data_norm_type="norm")
    dummy_vit_model = UniCeptionViTEncoderBase(name="name", data_norm_type="norm", patch_size=16)
    print("Dummy Base Encoders created successfully!")
