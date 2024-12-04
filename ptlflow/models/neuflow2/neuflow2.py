from importlib.metadata import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from . import backbone_v7
from . import transformer
from . import matching
from . import corr
from . import refine
from . import upsample
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


class NeuFlow2(BaseModel):
    pretrained_checkpoints = {
        "mixed": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/neuflow2-mixed-acac1a70.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/neuflow2-sintel-15c625f8.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/neuflow2-things-6ed47437.ckpt",
    }

    def __init__(
        self,
        gamma: float = 0.8,
        max_flow: float = 400,
        feature_dim_s16: int = 128,
        context_dim_s16: int = 64,
        iter_context_dim_s16: int = 64,
        feature_dim_s8: int = 128,
        context_dim_s8: int = 64,
        iter_context_dim_s8: int = 64,
        feature_dim_s1: int = 128,
        iters_s16: int = 1,
        iters_s8: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=16, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.gamma = gamma
        self.max_flow = max_flow
        self.feature_dim_s16 = feature_dim_s16
        self.context_dim_s16 = context_dim_s16
        self.iter_context_dim_s16 = iter_context_dim_s16
        self.feature_dim_s8 = feature_dim_s8
        self.context_dim_s8 = context_dim_s8
        self.iter_context_dim_s8 = iter_context_dim_s8
        self.feature_dim_s1 = feature_dim_s1
        self.iters_s16 = iters_s16
        self.iters_s8 = iters_s8

        if not version("torch").startswith("2"):
            raise ImportError(
                f"NeuFlow 2 requires torch 2.X. Your current version is {version('torch')}"
            )

        self.backbone = backbone_v7.CNNEncoder(
            self.feature_dim_s16,
            self.context_dim_s16,
            self.feature_dim_s8,
            self.context_dim_s8,
        )

        self.cross_attn_s16 = transformer.FeatureAttention(
            self.feature_dim_s16 + self.context_dim_s16,
            num_layers=2,
            ffn=True,
            ffn_dim_expansion=1,
            post_norm=True,
        )

        self.matching_s16 = matching.Matching()

        # self.flow_attn_s16 = transformer.FlowAttention(self.feature_dim_s16)

        self.corr_block_s16 = corr.CorrBlock(radius=4, levels=1)
        self.corr_block_s8 = corr.CorrBlock(radius=4, levels=1)

        self.merge_s8 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.feature_dim_s16 + self.feature_dim_s8,
                self.feature_dim_s8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                self.feature_dim_s8,
                self.feature_dim_s8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(self.feature_dim_s8),
        )

        self.context_merge_s8 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.context_dim_s16 + self.context_dim_s8,
                self.context_dim_s8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                self.context_dim_s8,
                self.context_dim_s8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(self.context_dim_s8),
        )

        self.refine_s16 = refine.Refine(
            self.context_dim_s16,
            self.iter_context_dim_s16,
            num_layers=5,
            levels=1,
            radius=4,
            inter_dim=128,
        )
        self.refine_s8 = refine.Refine(
            self.context_dim_s8,
            self.iter_context_dim_s8,
            num_layers=5,
            levels=1,
            radius=4,
            inter_dim=96,
        )

        self.conv_s8 = backbone_v7.ConvBlock(
            3, self.feature_dim_s1, kernel_size=8, stride=8, padding=0
        )
        self.upsample_s8 = upsample.UpSample(self.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.has_init_bhwd = False

    def init_bhwd(self, batch_size, height, width, device, amp=True):
        self.backbone.init_bhwd(batch_size * 2, height // 16, width // 16, device, amp)

        self.matching_s16.init_bhwd(batch_size, height // 16, width // 16, device, amp)

        self.corr_block_s16.init_bhwd(
            batch_size, height // 16, width // 16, device, amp
        )
        self.corr_block_s8.init_bhwd(batch_size, height // 8, width // 8, device, amp)

        self.refine_s16.init_bhwd(batch_size, height // 16, width // 16, device, amp)
        self.refine_s8.init_bhwd(batch_size, height // 8, width // 8, device, amp)

        self.init_iter_context_s16 = torch.zeros(
            batch_size,
            self.iter_context_dim_s16,
            height // 16,
            width // 16,
            device=device,
            dtype=torch.half if amp else torch.float,
        )
        self.init_iter_context_s8 = torch.zeros(
            batch_size,
            self.iter_context_dim_s8,
            height // 8,
            width // 8,
            device=device,
            dtype=torch.half if amp else torch.float,
        )

    def split_features(self, features, context_dim, feature_dim):
        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=[0.0, 0.0, 0.0],
            bgr_mult=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            resize_mode="interpolation",
            interpolation_align_corners=False,
        )

        img0 = images[:, 0]
        img1 = images[:, 1]

        flow_list = []

        if not self.has_init_bhwd:
            self.init_bhwd(
                img0.shape[0],
                img0.shape[2],
                img0.shape[3],
                img0.device,
                img0.dtype == torch.float16,
            )

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s16 = self.cross_attn_s16(features_s16)

        features_s16, context_s16 = self.split_features(
            features_s16, self.context_dim_s16, self.feature_dim_s16
        )
        features_s8, context_s8 = self.split_features(
            features_s8, self.context_dim_s8, self.feature_dim_s8
        )

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        flow0 = self.matching_s16.global_correlation_softmax(feature0_s16, feature1_s16)

        # flow0 = self.flow_attn_s16(feature0_s16, flow0)

        corr_pyr_s16 = self.corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)

        iter_context_s16 = self.init_iter_context_s16

        for i in range(self.iters_s16):
            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s16 = iter_context_s16.detach()

            corrs = self.corr_block_s16(corr_pyr_s16, flow0)

            iter_context_s16, delta_flow = self.refine_s16(
                corrs, context_s16, iter_context_s16, flow0
            )

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode="bilinear") * 16
                up_flow0 = self.postprocess_predictions(
                    up_flow0, image_resizer, is_flow=True
                )
                flow_list.append(up_flow0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode="nearest") * 2

        features_s16 = F.interpolate(features_s16, scale_factor=2, mode="nearest")

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))

        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        corr_pyr_s8 = self.corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode="nearest")

        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        iter_context_s8 = self.init_iter_context_s8

        for i in range(self.iters_s8):
            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s8 = iter_context_s8.detach()

            corrs = self.corr_block_s8(corr_pyr_s8, flow0)

            iter_context_s8, delta_flow = self.refine_s8(
                corrs, context_s8, iter_context_s8, flow0
            )

            flow0 = flow0 + delta_flow

            if self.training or i == self.iters_s8 - 1:
                feature0_s1 = self.conv_s8(img0)
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                up_flow0 = self.postprocess_predictions(
                    up_flow0, image_resizer, is_flow=True
                )
                flow_list.append(up_flow0)

        if self.training:
            outputs = {"flows": up_flow0[:, None], "flow_preds": flow_list}
        else:
            outputs = {"flows": up_flow0[:, None]}

        return outputs


@register_model
@trainable
class neuflow2(NeuFlow2):
    pass
