"""
Base Prediction Head Class for UniCeption
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PredictionHeadInput:
    last_feature: Tensor


@dataclass
class PredictionHeadLayeredInput:
    list_features: List[Tensor]
    target_output_shape: Tuple[int, int]


@dataclass
class PredictionHeadTokenInput:
    last_feature: Tensor


@dataclass
class PixelTaskOutput:
    """
    PixelTaskOutput have dense pixel-wise output in BCHW format,
    with the same spatial resolution as the input image.
    """

    decoded_channels: Tensor


@dataclass
class SummaryTaskOutput:
    """
    SummaryTaskOutput have a single latent output for each image in BC format.
    """

    decoded_channels: Tensor


@dataclass
class AdaptorInput:
    adaptor_feature: Tensor
    output_shape_hw: Tuple[int, int]


@dataclass
class AdaptorOutput:
    value: Tensor


@dataclass
class PredictionHeadOutput:
    adaptor_output: Dict[str, AdaptorOutput]


@dataclass
class MaskAdaptorOutput:
    logits: Tensor
    mask: Tensor


@dataclass
class Covariance2DAdaptorOutput:
    covariance: Tensor  # the 3 channels are s_x^2, s_y^2, and rho_xy
    log_det: Tensor  # log determinant of the covariance matrix
    inv_covariance: Tensor  # the channels are [0,0], [1,1], and [0,1] of the inverse covariance matrix
    log_representation: Tensor  # c1, c2, s representation of 2D covariance


@dataclass
class RegressionAdaptorOutput:
    value: Tensor


@dataclass
class RegressionWithConfidenceAdaptorOutput:
    value: Tensor
    confidence: Tensor


@dataclass
class RegressionWithMaskAdaptorOutput:
    value: Tensor
    logits: Tensor
    mask: Tensor


@dataclass
class RegressionWithConfidenceAndMaskAdaptorOutput:
    value: Tensor
    confidence: Tensor
    logits: Tensor
    mask: Tensor


class UniCeptionPredictionHeadBase(nn.Module):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Base class for all prediction heads in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name

    def forward(
        self,
        head_input: PredictionHeadInput,
    ) -> PredictionHeadOutput:
        """
        Forward interface for the UniCeption prediction heads.


        Args:
            head_input (PredictionHeadInput): Input to the prediction head.

        Returns:
            head_output (PredictionHeadOutput): Output of the prediction head.
        """

        raise NotImplementedError


class UniCeptionAdaptorBase(nn.Module):
    def __init__(
        self,
        name: str,
        required_channels: int,
        *args,
        **kwargs,
    ):
        """
        Base class for all adaptors in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.required_channels: int = required_channels

    def forward(
        self,
        adaptor_input: AdaptorInput,
    ) -> AdaptorOutput:
        """
        Forward interface for the UniCeption adaptors.


        Args:
            adaptor_input (AdaptorInput): Input to the adaptor.

        Returns:
            adaptor_output (AdaptorOutput): Output of the adaptor.
        """

        raise NotImplementedError


class AdaptorMap(nn.Module):
    def __init__(self, *adaptors: UniCeptionAdaptorBase):
        """
        AdaptorMap slices the input tensor and passes it to the corresponding adaptors.

        Args:
            *adaptors (List[UniCeptionAdaptorBase]): List of adaptors in the Adaptor
        """

        super().__init__()
        self.adaptors = nn.ModuleDict({adaptor.name: adaptor for adaptor in adaptors})

        self.required_channels = sum([adaptor.required_channels for adaptor in adaptors])

    def forward(
        self,
        adaptor_input: AdaptorInput,
    ) -> Dict[str, AdaptorOutput]:
        """
        Run the input through the adaptors and return the output.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptors.

        Returns:
            Dict[str, AdaptorOutput]: Output of the adaptors, from adaptor name to AdaptorOutput.
        """

        # split adaptor input into chunks
        adaptor_features = torch.split(
            adaptor_input.decoded_channels, [adaptor.required_channels for adaptor in self.adaptors.values()], dim=1
        )

        result = {
            adaptor_name: adaptor(AdaptorInput(adaptor_features[i], adaptor_features[i].shape[2:]))
            for i, (adaptor_name, adaptor) in enumerate(self.adaptors.items())
        }

        return result
