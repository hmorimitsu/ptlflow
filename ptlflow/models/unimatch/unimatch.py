from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (
    global_correlation_softmax,
    local_correlation_softmax,
    local_correlation_with_flow,
)
from .attention import SelfAttnPropagation
from .geometry import flow_warp
from .reg_refine import BasicUpdateBlock
from .utils import (
    feature_add_position,
    upsample_flow_with_mask,
)
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
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class UniMatch(BaseModel):
    pretrained_checkpoints = {
        "mix": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch-mixdata-9d7c1e4d.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch-things-2433864a.ckpt",
    }

    def __init__(
        self,
        gamma: float = 0.9,
        max_flow: float = 400.0,
        corr_radius: int = 4,
        feature_channels: int = 128,
        num_scales: int = 1,
        upsample_factor: int = 8,
        reg_refine: bool = False,
        num_transformer_layers: int = 6,
        num_head: int = 1,
        ffn_dim_expansion: int = 4,
        pred_bidir_flow: bool = False,
        num_reg_refine: int = 1,
        attn_type: str = "swin",
        attn_splits_list: Sequence[int] = (2,),
        corr_radius_list: Sequence[int] = (-1,),
        prop_radius_list: Sequence[int] = (-1,),
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=32, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.gamma = gamma
        self.max_flow = max_flow
        self.corr_radius = corr_radius
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine
        self.num_transformer_layers = num_transformer_layers
        self.num_head = num_head
        self.ffn_dim_expansion = ffn_dim_expansion
        self.pred_bidir_flow = pred_bidir_flow
        self.num_reg_refine = num_reg_refine
        self.attn_type = attn_type
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.prop_radius_list = prop_radius_list

        # CNN
        self.backbone = CNNEncoder(
            output_dim=self.feature_channels,
            num_output_scales=self.num_scales,
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=self.num_transformer_layers,
            d_model=self.feature_channels,
            nhead=self.num_head,
            ffn_dim_expansion=self.ffn_dim_expansion,
        )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=self.feature_channels)

        if not self.reg_refine:
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(
                nn.Conv2d(2 + self.feature_channels, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.upsample_factor**2 * 9, 1, 1, 0),
            )
            # thus far, all the learnable parameters are task-agnostic

        if self.reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(
                corr_channels=(2 * 4 + 1) ** 2,
                downsample_factor=self.upsample_factor,
                flow_dim=2,
                bilinear_up=False,
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

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8):
        if bilinear:
            multiplier = upsample_factor
            up_flow = (
                F.interpolate(
                    flow,
                    scale_factor=upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                * multiplier
            )
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(
                flow, mask, upsample_factor=self.upsample_factor
            )

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

        results_dict = {}
        flow_preds = []

        # list of features, resolution low to high
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

            feature0_ori, feature1_ori = feature0, feature1

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
                feature0,
                feature1,
                attn_type=self.attn_type,
                attn_num_splits=attn_splits,
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

            # upsample to the original resolution for supervison at training time only
            if self.training:
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

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(
                    flow, feature0, bilinear=True, upsample_factor=upsample_factor
                )
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    flow_up = self.upsample_flow(flow, feature0)
                    flow_up = self.postprocess_predictions(
                        flow_up, image_resizer, is_flow=True
                    )
                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(
                            flow,
                            feature0,
                            bilinear=True,
                            upsample_factor=upsample_factor,
                        )
                        flow_up = self.postprocess_predictions(
                            flow_up, image_resizer, is_flow=True
                        )
                        flow_preds.append(flow_up)

                    assert self.num_reg_refine > 0
                    for refine_iter_idx in range(self.num_reg_refine):
                        flow = flow.detach()
                        correlation = local_correlation_with_flow(
                            feature0_ori,
                            feature1_ori,
                            flow=flow,
                            local_radius=4,
                        )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(
                            net,
                            inp,
                            correlation,
                            flow.clone(),
                        )
                        flow = flow + residual_flow

                        if self.training or refine_iter_idx == self.num_reg_refine - 1:
                            flow_up = upsample_flow_with_mask(
                                flow, up_mask, upsample_factor=self.upsample_factor
                            )

                            flow_up = self.postprocess_predictions(
                                flow_up, image_resizer, is_flow=True
                            )
                            flow_preds.append(flow_up)

        results_dict.update({"flow_preds": flow_preds})

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_preds}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": flow}

        return outputs


class UniMatchScale2(UniMatch):
    pretrained_checkpoints = {
        "mix": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2-mixdata-b514dde2.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2-things-e75ae2f7.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2-sintel-f43b76ab.ckpt",
    }

    def __init__(
        self,
        gamma: float = 0.9,
        max_flow: float = 400,
        corr_radius: int = 4,
        feature_channels: int = 128,
        num_scales: int = 2,
        upsample_factor: int = 4,
        reg_refine: bool = False,
        num_transformer_layers: int = 6,
        num_head: int = 1,
        ffn_dim_expansion: int = 4,
        pred_bidir_flow: bool = False,
        num_reg_refine: int = 1,
        attn_type: str = "swin",
        attn_splits_list: Sequence[int] = (2, 8),
        corr_radius_list: Sequence[int] = (-1, 4),
        prop_radius_list: Sequence[int] = (-1, 1),
        **kwargs,
    ) -> None:
        super().__init__(
            gamma,
            max_flow,
            corr_radius,
            feature_channels,
            num_scales,
            upsample_factor,
            reg_refine,
            num_transformer_layers,
            num_head,
            ffn_dim_expansion,
            pred_bidir_flow,
            num_reg_refine,
            attn_type,
            attn_splits_list,
            corr_radius_list,
            prop_radius_list,
            **kwargs,
        )


class UniMatchScale2With6Refinements(UniMatch):
    pretrained_checkpoints = {
        "mix": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-mixdata-398760b1.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-things-54d7505b.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-sintel-95ab1410.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-kitti-0626279a.ckpt",
    }

    def __init__(
        self,
        gamma: float = 0.9,
        max_flow: float = 400,
        corr_radius: int = 4,
        feature_channels: int = 128,
        num_scales: int = 2,
        upsample_factor: int = 4,
        reg_refine: bool = True,
        num_transformer_layers: int = 6,
        num_head: int = 1,
        ffn_dim_expansion: int = 4,
        pred_bidir_flow: bool = False,
        num_reg_refine: int = 6,
        attn_type: str = "swin",
        attn_splits_list: Sequence[int] = (2, 8),
        corr_radius_list: Sequence[int] = (-1, 4),
        prop_radius_list: Sequence[int] = (-1, 1),
        **kwargs,
    ) -> None:
        super().__init__(
            gamma,
            max_flow,
            corr_radius,
            feature_channels,
            num_scales,
            upsample_factor,
            reg_refine,
            num_transformer_layers,
            num_head,
            ffn_dim_expansion,
            pred_bidir_flow,
            num_reg_refine,
            attn_type,
            attn_splits_list,
            corr_radius_list,
            prop_radius_list,
            **kwargs,
        )


@register_model
@trainable
class unimatch(UniMatch):
    pass


@register_model
@trainable
class unimatch_sc2(UniMatchScale2):
    pass


@register_model
@trainable
class unimatch_sc2_ref6(UniMatchScale2With6Refinements):
    pass


@register_model
@trainable
class gmflow_p(UniMatch):
    pass


@register_model
@trainable
class gmflow_p_sc2(UniMatchScale2):
    pass


@register_model
@trainable
class gmflow_p_sc2_ref6(UniMatchScale2With6Refinements):
    pass
