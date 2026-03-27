import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    convnext_tiny,
    convnext_small,
    ConvNeXt_Large_Weights,
    convnext_large,
)
from torchvision.models.convnext import (
    CNBlockConfig,
    LayerNorm2d,
    CNBlock,
    Conv2dNormActivation,
)

from functools import partial
from typing import List


class ConvNeXt_Extractor(nn.Module):
    def __init__(
        self,
        size="s",
        norm_layer=None,
        stochastic_depth_prob=0,
        layer_scale: float = 1e-6,
        input_dim=3,
        output_dim=256,
        pretrain=False,
    ):
        super().__init__()

        if size == "t":
            # tiny
            block_setting = [
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 9),
                CNBlockConfig(768, output_dim, 3),
            ]
        elif size == "s":
            # small
            block_setting = [
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 27),
                CNBlockConfig(768, output_dim, 3),
            ]
        elif size == "l":
            block_setting = [
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 3),
                CNBlockConfig(768, 1536, 27),
                CNBlockConfig(1536, output_dim, 3),
            ]

        block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                input_dim,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                # stride=1,
                padding=0,
                # padding=2,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for i, cnf in enumerate(block_setting):
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i < len(block_setting) - 3:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(
                            cnf.input_channels,
                            cnf.out_channels,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.ZeroPad2d((0, 1, 0, 1)),
                        nn.Conv2d(
                            cnf.input_channels,
                            cnf.out_channels,
                            kernel_size=2,
                            stride=1,
                        ),
                    )
                )
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if pretrain and size == "t":
            pretrained_dict = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            ).state_dict()
        elif pretrain and size == "s":
            pretrained_dict = convnext_small(
                weights=ConvNeXt_Small_Weights.IMAGENET1K_V1
            ).state_dict()
        elif pretrain and size == "l":
            pretrained_dict = convnext_large(
                weights=ConvNeXt_Large_Weights.IMAGENET1K_V1
            ).state_dict()

        if pretrain:
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }

            if input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == "features.0.0.weight":
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=True)
            print("loaded pretrained checkpoints")

    def forward(self, x):
        return self.features(x)
