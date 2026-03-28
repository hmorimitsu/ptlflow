import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    mobilenet_v3_small,
    mobilenet_v3_large,
)
from torchvision.models.mobilenetv3 import (
    InvertedResidualConfig,
    InvertedResidual,
    Conv2dNormActivation,
)

from functools import partial
from typing import List


class MobileNetV3Extractor(nn.Module):
    def __init__(
        self,
        size="l",
        input_dim=3,
        output_dim=256,
        block=None,
        norm_layer=None,
        pretrain=True,
    ) -> None:
        super().__init__()

        dilation = 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=1)

        if size == "l":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 1, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(112, 5, 672, 160, True, "HS", 1, dilation),  # C4
                bneck_conf(160, 5, 960, 160, True, "HS", 1, dilation),
                bneck_conf(160, 5, 960, 160, True, "HS", 1, dilation),
            ]

        elif size == "s":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 1, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 288, 96, True, "HS", 1, dilation),  # C4
                bneck_conf(96, 5, 576, 96, True, "HS", 1, dilation),
                bneck_conf(96, 5, 576, 96, True, "HS", 1, dilation),
            ]

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                input_dim,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.final = nn.Conv2d(960 if size == "l" else 576, output_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrain and size == "l":
            pretrained_dict = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            ).state_dict()
        elif pretrain and size == "s":
            pretrained_dict = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
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
        return self.final(self.features(x))


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
