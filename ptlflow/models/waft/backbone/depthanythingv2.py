import torch
import torch.nn as nn
import torchvision

from ..thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingFeature(nn.Module):
    def __init__(self, encoder="vits", pretrained=False):
        super().__init__()
        self.model_configs = {
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
        self.encoder = encoder
        depth_anything = DepthAnythingV2(**self.model_configs[encoder])
        if pretrained:
            depth_anything.load_state_dict(
                torch.load(
                    f"depth-anything-ckpts/depth_anything_v2_{encoder}.pth",
                    map_location="cpu",
                )
            )
        self.depth_anything = depth_anything

    def forward(self, x):
        """
        @x: (B,C,H,W)
        """
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(
            x,
            self.depth_anything.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size

        out, path_1, path_2, path_3, path_4 = self.depth_anything.depth_head.forward(
            features, patch_h, patch_w, return_intermediate=True
        )

        return {
            "out": out,
            "path_1": path_1,
            "path_2": path_2,
            "path_3": path_3,
            "path_4": path_4,
            "features": features,
        }  # path_1 is 1/2; path_2 is 1/4


def normalize_image(img):
    """
    @img: (B,C,H,W) in range 0-255, RGB order
    """
    tf = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
    )
    return tf(img / 255.0).contiguous()
