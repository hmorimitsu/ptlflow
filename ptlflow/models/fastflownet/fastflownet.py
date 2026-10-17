"""
Portions of this code copyright 2017, Clement Pinard
"""

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.corr_modes import (
    get_spatial_corr_sampler,
    log_corr_mode_warnings,
    SpatialCorrMode,
    SPATIAL_CORR_VARIANTS,
)
from ptlflow.utils.registry import register_model, register_model_variants, trainable
from ..base_model.base_model import BaseModel
from ..flownet.losses import MultiScale


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(
                self.conv6(
                    self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
                )
            )
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class FastFlowNet(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/fastflownet-chairs-89e7a48e.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/fastflownet-kitti-6d3526a8.ckpt",
        "mix": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/fastflownet-mix-fd9b8c0d.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/fastflownet-sintel-6475ea96.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/fastflownet-things3d-fc093d29.ckpt",
    }

    def __init__(
        self,
        div_flow: float = 20.0,
        md: int = 4,
        groups: int = 3,
        corr_mode: SpatialCorrMode = "iter",
        loss_start_scale: int = 4,
        loss_num_scales: int = 5,
        loss_base_weight: float = 0.32,
        loss_norm: str = "L2",
        simple_io: bool = False,
        **kwargs,
    ):
        super(FastFlowNet, self).__init__(
            loss_fn=MultiScale(
                startScale=loss_start_scale,
                numScales=loss_num_scales,
                l_weight=loss_base_weight,
                norm=loss_norm,
            ),
            output_stride=64,
            simple_io=simple_io,
            **kwargs,
        )

        self.div_flow = div_flow
        self.md = md
        self.groups = groups
        self.corr_mode = corr_mode

        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        log_corr_mode_warnings(self.corr_mode)
        self.corr_layer, self._corr_error = get_spatial_corr_sampler(
            self.corr_mode, patch_size=2 * self.md + 1
        )
        self.index = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                21,
                22,
                23,
                24,
                26,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                36,
                38,
                39,
                40,
                41,
                42,
                44,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                54,
                56,
                57,
                58,
                59,
                60,
                62,
                64,
                66,
                68,
                70,
                72,
                74,
                76,
                78,
                80,
            ]
        )
        # One-hot channel-selection kernel equivalent to
        # torch.index_select(corr, dim=1, index=self.index). The conv2d form
        # exports (and fuses in TensorRT) better than Gather/index_select.
        index_weight = torch.zeros(len(self.index), (2 * self.md + 1) ** 2, 1, 1)
        for i, idx in enumerate(self.index):
            index_weight[i, idx, 0, 0] = 1.0
        self.register_buffer("index_weight", index_weight, persistent=False)

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, self.groups)
        self.decoder3 = Decoder(87, self.groups)
        self.decoder4 = Decoder(87, self.groups)
        self.decoder5 = Decoder(87, self.groups)
        self.decoder6 = Decoder(87, self.groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def corr(self, f1, f2):
        corr = self.corr_layer(f1, f2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / f1.shape[1]
        return corr

    def warp(self, x, flo):
        B, C, H, W = x.size()
        # Create the meshgrid directly on the input device/dtype to avoid the
        # ONNX export freezing it as a CPU constant, and compute the sampling
        # grid without in-place slice assignments, which are not
        # ONNX/TensorRT-exportable.
        xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1)

        grid_x = 2.0 * grid[:, 0:1, :, :] / max(W - 1, 1) - 1.0
        grid_y = 2.0 * grid[:, 1:2, :, :] / max(H - 1, 1) - 1.0

        flo_x = 2.0 * flo[:, 0:1, :, :] / max(W - 1, 1)
        flo_y = 2.0 * flo[:, 1:2, :, :] / max(H - 1, 1)

        vgrid = torch.cat([grid_x + flo_x, grid_y + flo_y], dim=1).permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode="bilinear", align_corners=True)
        return output

    def forward(self, inputs):
        inputs = self._normalize_inputs(inputs)
        if self.corr_layer is None:
            raise self._corr_error
        mean_bgr = rearrange(inputs["images"], "b n c h w -> b c (n h w)").mean(2)[
            :, None, :, None, None
        ]
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-mean_bgr,
            bgr_mult=1.0,
            bgr_to_rgb=False,
            resize_mode="interpolation",
            interpolation_mode="bilinear",
            interpolation_align_corners=False,
        )
        img1 = images[:, 0]
        img2 = images[:, 1]

        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = F.conv2d(self.corr(f16, f26), self.index_weight)
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up * 0.625)
        cv5 = F.conv2d(self.corr(f15, f25_w), self.index_weight)
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up * 1.25)
        cv4 = F.conv2d(self.corr(f14, f24_w), self.index_weight)
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up * 2.5)
        cv3 = F.conv2d(self.corr(f13, f23_w), self.index_weight)
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up * 5.0)
        cv2 = F.conv2d(self.corr(f12, f22_w), self.index_weight)
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        flow_up = self.div_flow * F.interpolate(
            flow2, size=img2.shape[-2:], mode="bilinear", align_corners=False
        )
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)

        outputs = {}
        if self.training:
            outputs["flow_preds"] = [flow2, flow3, flow4, flow5, flow6]
            outputs["flows"] = flow_up[:, None]
        else:
            outputs["flows"] = flow_up[:, None]
        return self._maybe_return_simple(outputs)


@register_model
@trainable
class fastflownet(FastFlowNet):
    pass


register_model_variants({"fastflownet": FastFlowNet}, SPATIAL_CORR_VARIANTS)
