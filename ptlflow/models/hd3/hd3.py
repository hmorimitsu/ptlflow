from typing import Sequence

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import (
        IterSpatialCorrelationSampler as SpatialCorrelationSampler,
    )
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from .hd3_ops import *
from .dla_up import DLAEncoder as dlaup_encoder
from .vgg import VGGEncoder as vgg_encoder
from . import decoder as PreDecoder
from .hd3losses import LossCalculator
from ..base_model.base_model import BaseModel

BatchNorm = nn.BatchNorm2d


class Context(nn.Module):
    def __init__(self, inplane, classes):
        super(Context, self).__init__()
        self.num_convs = 7
        ch = [inplane, 128, 128, 128, 128, 128, 128, 128]
        dilations = [1, 1, 2, 4, 8, 16, 1]
        for i in range(self.num_convs):
            setattr(
                self,
                "dc_conv_{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        ch[i],
                        ch[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=dilations[i],
                        dilation=dilations[i],
                        bias=False,
                    ),
                    BatchNorm(ch[i + 1]),
                    nn.ReLU(inplace=True),
                ),
            )
        self.cls = nn.Conv2d(
            ch[-1], classes, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        out = x
        for i in range(self.num_convs):
            dc_conv = getattr(self, "dc_conv_" + str(i))
            out = dc_conv(out)
        out = self.cls(out)
        return out, None


class Decoder(nn.Module):
    def __init__(self, inplane, block, classes, up_classes):
        super(Decoder, self).__init__()
        self.mapping = block(inplane, 128)
        self.cls = nn.Sequential(
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.up = None
        if up_classes > 0:
            self.up = nn.Sequential(
                BatchNorm(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    128, up_classes, kernel_size=4, stride=2, padding=1, bias=False
                ),
                BatchNorm(up_classes),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.mapping(x)
        prob = self.cls(out)
        up_feat = self.up(out) if self.up else None
        return prob, up_feat


class HD3(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3-chairs-0d46c9fd.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3-things-afcd2eb4.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3-sintel-10689995.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3-kitti-6eb77dd3.ckpt",
    }

    def __init__(
        self,
        task: str = "flow",
        encoder: str = "dlaup",
        decoder: str = "hda",
        downsample: int = 6,
        corr_range: Sequence[int] = (4, 4, 4, 4, 4, 4),
        context: bool = False,
        **kwargs,
    ):
        super(HD3, self).__init__(
            loss_fn=LossCalculator(task), output_stride=int(2**downsample), **kwargs
        )

        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        self.downsample = downsample
        self.corr_range = corr_range
        self.context = context

        if self.task == "flow":
            self.corr_range = self.corr_range[:5]

        self.task = self.task
        self.dim = 1 if self.task == "stereo" else 2
        self.levels = len(self.corr_range)
        if self.task == "flow":
            self.classes = [(2 * d + 1) ** 2 for d in self.corr_range]
        else:
            self.classes = [2 * d + 1 for d in self.corr_range]

        if self.encoder == "vgg":
            pyr_channels = [16, 32, 64, 96, 128, 196]
            assert self.levels <= len(pyr_channels)
            self.encoder = vgg_encoder(pyr_channels)
        elif self.encoder == "dlaup":
            pyr_channels = [16, 32, 64, 128, 256, 512, 512]
            self.encoder = dlaup_encoder(pyr_channels)
        else:
            raise ValueError("Unknown encoder {}".format(self.encoder))

        if self.decoder == "resnet":
            dec_block = PreDecoder.ResnetDecoder
        elif self.decoder == "hda":
            dec_block = PreDecoder.HDADecoder
        else:
            raise ValueError("Unknown decoder {}".format(self.decoder))

        feat_d_offset = pyr_channels[::-1]
        feat_d_offset[0] = 0
        up_d_offset = [0] + self.classes[1:]

        for l in range(self.levels):
            setattr(self, "cost_bn_{}".format(l), BatchNorm(self.classes[l]))
            input_d = (
                self.classes[l] + feat_d_offset[l] + up_d_offset[l] + self.dim * (l > 0)
            )
            if l < self.levels - 1:
                up_classes = self.classes[l + 1]
            else:
                up_classes = -1
            if self.context and l == self.levels - 1:
                setattr(self, "Decoder_{}".format(l), Context(input_d, self.classes[l]))
            else:
                setattr(
                    self,
                    "Decoder_{}".format(l),
                    Decoder(input_d, dec_block, self.classes[l], up_classes=up_classes),
                )

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=9, padding=0)

        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find("BatchNorm") != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def shift(self, x, vect):
        if vect.size(1) < 2:
            vect = disp2flow(vect)
        return flow_warp(x, vect)

    def forward(self, inputs):
        x, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=[-0.485, -0.456, -0.406],
            bgr_mult=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            bgr_to_rgb=False,
            resize_mode="interpolation",
            interpolation_mode="bilinear",
            interpolation_align_corners=True,
        )

        # extract pyramid features
        bs = x.size(0)
        feat_list = self.encoder(torch.cat([x[:, 0], x[:, 1]], 0))
        fp_0 = [f[:bs, :, :, :] for f in feat_list[::-1]]
        fp_1 = [f[bs:, :, :, :] for f in feat_list[::-1]]

        ms_pred = []
        for l in range(self.levels):
            ref_feat = fp_0[l]
            tar_feat = fp_1[l]

            if l == 0:
                tar_feat_corr = tar_feat
            else:
                tar_feat_corr = self.shift(tar_feat, up_curr_vect)

            cost_vol = self.corr(ref_feat, tar_feat_corr)
            cost_vol = cost_vol.view(
                cost_vol.shape[0], -1, cost_vol.shape[3], cost_vol.shape[4]
            )
            cost_vol = cost_vol / ref_feat.shape[1]
            if self.task == "stereo":
                c = self.classes[l] // 2
                cost_vol = cost_vol[
                    :, c * (2 * c + 1) : (c + 1) * (2 * c + 1), :, :
                ].contiguous()
            cost_bn = getattr(self, "cost_bn_" + str(l))
            cost_vol = cost_bn(cost_vol)

            if l == 0:
                decoder_input = cost_vol
            else:
                decoder_input = torch.cat(
                    [cost_vol, ref_feat, ms_pred[-1][-1], up_curr_vect], 1
                )

            decoder = getattr(self, "Decoder_" + str(l))
            prob_map, up_feat = decoder(decoder_input)

            curr_vect = density2vector(prob_map, self.dim, True)
            if l > 0:
                curr_vect += up_curr_vect
            if self.task == "stereo":
                curr_vect = torch.clamp(curr_vect, max=0)
            ms_pred.append([prob_map, curr_vect * 2 ** (self.downsample - l), up_feat])

            if l < self.levels - 1:
                up_curr_vect = 2 * F.interpolate(
                    curr_vect, scale_factor=2, mode="bilinear", align_corners=True
                )

        ms_prob = [l[0] for l in ms_pred]
        ms_vect = [l[1] for l in ms_pred]

        flow_up = F.interpolate(
            ms_vect[-1], scale_factor=4, mode="bilinear", align_corners=False
        )
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)

        outputs = {}
        if self.training:
            outputs["ms_prob"] = ms_prob
            outputs["ms_pred"] = ms_vect
            outputs["corr_range"] = self.corr_range
            outputs["downsample"] = self.downsample
            outputs["flows"] = flow_up[:, None]
        else:
            outputs["flows"] = flow_up[:, None]
        return outputs


class HD3Context(HD3):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3_ctxt-chairs-d7448468.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3_ctxt-things-f1681f75.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3_ctxt-sintel-22e655e9.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/hd3_ctxt-kitti-e7d69776.ckpt",
    }

    def __init__(
        self,
        task: str = "flow",
        encoder: str = "dlaup",
        decoder: str = "hda",
        downsample: int = 6,
        corr_range: Sequence[int] = (4, 4, 4, 4, 4, 4),
        context: bool = True,
        **kwargs,
    ):
        super().__init__(
            task, encoder, decoder, downsample, corr_range, context, **kwargs
        )


@register_model
@trainable
class hd3(HD3):
    pass


@register_model
@trainable
class hd3_ctxt(HD3Context):
    pass
