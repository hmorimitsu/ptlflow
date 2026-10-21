"""
Image normalizations for the different UniCeption image encoders.
Image encoders defined in UniCeption must have their corresponding image normalization defined here.
"""

from dataclasses import dataclass

import torch


@dataclass
class ImageNormalization:
    mean: torch.Tensor
    std: torch.Tensor


IMAGE_NORMALIZATION_DICT = {
    "dummy": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "croco": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    "dune": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dinov3": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "perception_encoder": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    "identity": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "patch_embedder": ImageNormalization(
        mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])
    ),
    "radio": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "sea_raft": ImageNormalization(
        mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]) / 255
    ),  # Sea-RAFT uses 0-255 in FP32
    "unimatch": ImageNormalization(
        mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]) / 255
    ),  # UniMatch uses 0-255 in FP32
    "roma": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "cosmos": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([0.5, 0.5, 0.5])),
}
