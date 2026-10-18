"""
Linear head with MLP implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width)
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PixelTaskOutput, PredictionHeadInput
from ..utils.transformer_blocks import Mlp


class MLPFeature(nn.Module):
    """
    This class implements a linear mapping from the low resolution patch features
    to pixel-wise features with an additional intermediate MLP layer.
    """

    def __init__(
        self,
        input_feature_dim: Union[int, str],
        patch_size: int,
        output_dim: int,
        mlp_ratio: int = 4,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the linear feature mapping.

        Args:
            input_feature_dim : int, the input feature dimension
            output_dim : int, the output feature dimension
            patch_size : int, the patch size
            pretrained_checkpoint_path : str, path to pretrained checkpoint (default: None)
        """

        super().__init__(*args, **kwargs)

        if isinstance(input_feature_dim, str):
            input_feature_dim = eval(input_feature_dim)

        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

        self.mlp = Mlp(
            in_features=self.input_feature_dim,
            hidden_features=int(mlp_ratio * self.input_feature_dim),
            act_layer=act_layer,
            drop=drop,
            bias=bias,
        )

        self.linear = nn.Conv2d(
            in_channels=self.input_feature_dim,
            out_channels=self.output_dim * (self.patch_size**2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained linear dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, feature_input: PredictionHeadInput):
        """
        Forward interface for the linear feature mapping.

        Args:
            feature_input : PredictionHeadInput, the input features
            - last_feature : torch.Tensor, the last feature tensor

        Returns:
            PixelTaskOutput, the output of the linear feature mapping
            - decoded_channels : torch.Tensor, the decoded channels

        """

        x = feature_input.last_feature

        assert (
            x.shape[1] == self.input_feature_dim
        ), f"Input feature dimension mismatch: {x.shape[1]} != {self.input_feature_dim}"

        x = self.mlp(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.linear(x)
        x = F.pixel_shuffle(x, self.patch_size)

        return PixelTaskOutput(decoded_channels=x)


if __name__ == "__main__":
    # Init an example linear feature head
    linear_prediction_head = MLPFeature(
        input_feature_dim=768, mlp_ratio=4, act_layer=nn.GELU, output_dim=4, patch_size=16
    )

    # Create a dummy input tensor with shape (B, C, H, W)
    dummy_input = torch.randn(1, 768, 14, 14)  # Example input

    # Run dummy forward pass
    output = linear_prediction_head(PredictionHeadInput(last_feature=dummy_input))
