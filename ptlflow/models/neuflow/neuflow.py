from importlib.metadata import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from . import backbone
from . import transformer
from . import matching
from . import refine
from . import upsample
from . import utils
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        weights = [0.2, 1]
        for i in range(n_predictions):
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += weights[i] * (valid * i_loss).mean()

        return flow_loss


class NeuFlow(BaseModel):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/neuflow-things-c402aa7a.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/neuflow-sintel-0d969ea2.ckpt",
    }

    def __init__(
        self,
        gamma: float = 0.8,
        max_flow: float = 400.0,
        feature_dim: int = 90,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=16, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.feature_dim = feature_dim

        if not version("torch").startswith("2"):
            raise ImportError(
                f"NeuFlow requires torch 2.X. Your current version is {version('torch')}"
            )

        self.backbone = backbone.CNNEncoder(feature_dim)
        self.cross_attn_s16 = transformer.FeatureAttention(
            feature_dim + 2,
            num_layers=2,
            bidir=True,
            ffn=True,
            ffn_dim_expansion=1,
            post_norm=True,
        )

        self.matching_s16 = matching.Matching()

        self.flow_attn_s16 = transformer.FlowAttention(feature_dim + 2)

        self.merge_s8 = torch.nn.Sequential(
            torch.nn.Conv2d(
                (feature_dim + 2) * 2,
                feature_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                feature_dim * 2,
                feature_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.refine_s8 = refine.Refine(feature_dim, patch_size=7, num_layers=6)

        self.conv_s8 = backbone.ConvBlock(
            3, feature_dim, kernel_size=8, stride=8, padding=0
        )

        self.upsample_s1 = upsample.UpSample(feature_dim, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.curr_bhw = None

    def init_bhw(self, batch_size, height, width):
        self.backbone.init_pos_12(
            batch_size, height // 8, width // 8, dtype=self.dtype, device=self.device
        )
        self.matching_s16.init_grid(
            batch_size, height // 16, width // 16, dtype=self.dtype, device=self.device
        )

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=[-0.406, -0.456, -0.485],
            bgr_mult=[1 / 0.225, 1 / 0.224, 1 / 0.229],
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        img0 = images[:, 0]
        img1 = images[:, 1]

        bhw = (img0.shape[0], img0.shape[-2], img0.shape[-1])
        if bhw != self.curr_bhw:
            self.init_bhw(*bhw)
            self.curr_bhw = bhw

        feature0_s8, feature0_s16 = self.backbone(img0)
        feature1_s8, feature1_s16 = self.backbone(img1)

        feature0_s16, feature1_s16 = self.cross_attn_s16(feature0_s16, feature1_s16)
        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        flow0 = self.flow_attn_s16(feature0_s16, flow0)

        feature0_s16 = F.interpolate(feature0_s16, scale_factor=2, mode="nearest")
        feature1_s16 = F.interpolate(feature1_s16, scale_factor=2, mode="nearest")

        feature0_s8 = self.merge_s8(torch.cat([feature0_s8, feature0_s16], dim=1))
        feature1_s8 = self.merge_s8(torch.cat([feature1_s8, feature1_s16], dim=1))

        flow0 = F.interpolate(flow0, scale_factor=2, mode="nearest") * 2

        delta_flow = self.refine_s8(
            feature0_s8, utils.flow_warp(feature1_s8, flow0), flow0
        )
        flow0 = flow0 + delta_flow

        flow_list = []
        if self.training:
            up_flow0 = (
                F.interpolate(
                    flow0, scale_factor=8, mode="bilinear", align_corners=True
                )
                * 8
            )
            up_flow0 = self.postprocess_predictions(
                up_flow0, image_resizer, is_flow=True
            )
            flow_list.append(up_flow0)

        feature0_s8 = self.conv_s8(img0)

        flow0 = self.upsample_s1(feature0_s8, flow0)
        flow0 = self.postprocess_predictions(flow0, image_resizer, is_flow=True)
        flow_list.append(flow0)

        if self.training:
            outputs = {"flows": flow0[:, None], "flow_preds": flow_list}
        else:
            outputs = {"flows": flow0[:, None]}

        return outputs


@register_model
@trainable
class neuflow(NeuFlow):
    pass
