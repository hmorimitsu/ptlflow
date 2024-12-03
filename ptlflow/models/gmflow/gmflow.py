from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import feature_add_position
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

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)

            i_loss = (flow_preds[i] - flow_gt).abs()

            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class GMFlow(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-chairs-4922131e.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-things-5a18a9e8.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-sintel-d6f83ccd.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-kitti-af50eb2e.ckpt",
    }

    def __init__(
        self,
        attention_type: str = "swin",
        attn_splits_list: Sequence[int] = (2,),
        corr_radius_list: Sequence[int] = (-1,),
        feature_channels: int = 128,
        ffn_dim_expansion: int = 4,
        gamma: float = 0.9,
        max_flow: float = 400.0,
        num_head: int = 1,
        num_scales: int = 1,
        num_transformer_layers: int = 6,
        pred_bidir_flow: bool = False,
        prop_radius_list: Sequence[int] = (-1,),
        upsample_factor: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=32, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.attention_type = attention_type
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.feature_channels = feature_channels
        self.ffn_dim_expansion = ffn_dim_expansion
        self.gamma = gamma
        self.max_flow = max_flow
        self.num_head = num_head
        self.num_scales = num_scales
        self.num_transformer_layers = num_transformer_layers
        self.pred_bidir_flow = pred_bidir_flow
        self.prop_radius_list = prop_radius_list
        self.upsample_factor = upsample_factor

        # CNN backbone
        self.backbone = CNNEncoder(
            output_dim=self.feature_channels,
            num_output_scales=self.num_scales,
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=self.num_transformer_layers,
            d_model=self.feature_channels,
            nhead=self.num_head,
            attention_type=self.attention_type,
            ffn_dim_expansion=self.ffn_dim_expansion,
        )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=self.feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(
            nn.Conv2d(2 + self.feature_channels, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.upsample_factor**2 * 9, 1, 1, 0),
        )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(
            concat
        )  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(
        self,
        flow,
        feature,
        bilinear=False,
        upsample_factor=8,
    ):
        if bilinear:
            up_flow = (
                F.interpolate(
                    flow,
                    scale_factor=upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                * upsample_factor
            )

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(
                b, 1, 9, self.upsample_factor, self.upsample_factor, h, w
            )  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(
                b, flow_channel, 9, 1, 1, h, w
            )  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(
                b,
                flow_channel,
                self.upsample_factor * h,
                self.upsample_factor * w,
            )  # [B, 2, K*H, K*W]

        return up_flow

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

        flow_preds = []

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(
            img0, img1
        )  # list of features

        flow = None

        assert (
            len(self.attn_splits_list)
            == len(self.corr_radius_list)
            == len(self.prop_radius_list)
            == self.num_scales
        )

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if self.pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat(
                    (feature1, feature0), dim=0
                )

            upsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            if scale_idx > 0:
                flow = (
                    F.interpolate(
                        flow, scale_factor=2, mode="bilinear", align_corners=True
                    )
                    * 2
                )

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = self.attn_splits_list[scale_idx]
            corr_radius = self.corr_radius_list[scale_idx]
            prop_radius = self.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(
                feature0, feature1, attn_splits, self.feature_channels
            )

            # Transformer
            feature0, feature1 = self.transformer(
                feature0, feature1, attn_num_splits=attn_splits
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(
                    feature0, feature1, self.pred_bidir_flow
                )[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[
                    0
                ]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if (
                self.training
            ):  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(
                    flow, None, bilinear=True, upsample_factor=upsample_factor
                )
                flow_bilinear = self.postprocess_predictions(
                    flow_bilinear, image_resizer, is_flow=True
                )
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if self.pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat(
                    (feature0, feature1), dim=0
                )  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(
                feature0,
                flow.detach(),
                local_window_attn=prop_radius > 0,
                local_window_radius=prop_radius,
            )

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(
                    flow, feature0, bilinear=True, upsample_factor=upsample_factor
                )
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_preds.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_preds}
        else:
            outputs = {"flows": flow_up[:, None]}

        return outputs


class GMFlowWithRefinement(GMFlow):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-chairs-88cdc009.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-things-e40899f5.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-sintel-ee46a2c4.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-kitti-b7bf2fda.ckpt",
    }

    def __init__(
        self,
        attention_type: str = "swin",
        attn_splits_list: Sequence[int] = (2, 8),
        corr_radius_list: Sequence[int] = (-1, 4),
        feature_channels: int = 128,
        ffn_dim_expansion: int = 4,
        gamma: float = 0.9,
        max_flow: float = 400,
        num_head: int = 1,
        num_scales: int = 2,
        num_transformer_layers: int = 6,
        pred_bidir_flow: bool = False,
        prop_radius_list: Sequence[int] = (-1, 1),
        upsample_factor: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            attention_type,
            attn_splits_list,
            corr_radius_list,
            feature_channels,
            ffn_dim_expansion,
            gamma,
            max_flow,
            num_head,
            num_scales,
            num_transformer_layers,
            pred_bidir_flow,
            prop_radius_list,
            upsample_factor,
            **kwargs,
        )


@register_model
@trainable
class gmflow(GMFlow):
    pass


@register_model
@trainable
class gmflow_refine(GMFlowWithRefinement):
    pass
