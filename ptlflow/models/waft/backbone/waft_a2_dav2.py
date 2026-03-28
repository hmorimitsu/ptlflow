import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .head import DPTHead
from ..thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

DEPTH_ANYTHING_CONFIGS = {
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


class DepthAnythingFeature(nn.Module):
    def __init__(self, model_name="vits", pretrained=False, lvl=-3):
        super().__init__()
        self.dpt_configs = {
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }
        self.model_name = model_name
        depth_anything = DepthAnythingV2(**DEPTH_ANYTHING_CONFIGS[model_name])
        if pretrained:
            depth_anything.load_state_dict(
                torch.load(
                    f"depth-anything-ckpts/depth_anything_v2_{model_name}.pth",
                    map_location="cpu",
                )
            )

        self.encoder = self.freeze_(depth_anything.pretrained)
        self.embed_dim = depth_anything.pretrained.embed_dim
        self.output_dim = self.dpt_configs[model_name]["features"]
        self.out_channels = self.dpt_configs[model_name]["out_channels"]
        del depth_anything
        self.dpt_head = DPTHead(
            self.embed_dim,
            features=self.output_dim,
            out_channels=self.out_channels,
            lvl=lvl,
        )

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2], x.shape[-1]
        features = self.encoder.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.model_name], return_class_token=True
        )
        patch_size = self.encoder.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        outs = self.dpt_head.forward(features, patch_h, patch_w)
        final = F.interpolate(
            outs[0], size=(h // 2, w // 2), mode="bilinear", align_corners=True
        )
        return final


def normalize_image(img):
    """
    @img: (B,C,H,W) in range 0-255, RGB order
    """
    tf = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
    )
    return tf(img / 255.0).contiguous()
