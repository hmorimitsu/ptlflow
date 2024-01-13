from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    normalize_img,
    feature_add_position,
    upsample_flow_with_mask,
    InputPadder,
)
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

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

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=32)

        # CNN
        self.backbone = CNNEncoder(
            output_dim=self.args.feature_channels,
            num_output_scales=self.args.num_scales,
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=self.args.num_transformer_layers,
            d_model=self.args.feature_channels,
            nhead=self.args.num_head,
            ffn_dim_expansion=self.args.ffn_dim_expansion,
        )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(
            in_channels=self.args.feature_channels
        )

        if not self.args.reg_refine:
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(
                nn.Conv2d(2 + self.args.feature_channels, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.args.upsample_factor**2 * 9, 1, 1, 0),
            )
            # thus far, all the learnable parameters are task-agnostic

        if self.args.reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(
                corr_channels=(2 * 4 + 1) ** 2,
                downsample_factor=self.args.upsample_factor,
                flow_dim=2,
                bilinear_up=False,
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument("--feature_channels", type=int, default=128)
        parser.add_argument("--num_scales", type=int, default=1)
        parser.add_argument("--upsample_factor", type=int, default=8)
        parser.add_argument("--reg_refine", action="store_true")
        parser.add_argument("--num_transformer_layers", type=int, default=6)
        parser.add_argument("--num_head", type=int, default=1)
        parser.add_argument("--ffn_dim_expansion", type=int, default=4)
        parser.add_argument("--pred_bidir_flow", action="store_true")
        parser.add_argument("--num_reg_refine", type=int, default=1)
        parser.add_argument("--attn_type", type=str, default="swin")
        parser.add_argument("--attn_splits_list", type=int, nargs="+", default=(2,))
        parser.add_argument("--corr_radius_list", type=int, nargs="+", default=(-1,))
        parser.add_argument("--prop_radius_list", type=int, nargs="+", default=(-1,))
        return parser

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
                flow, mask, upsample_factor=self.args.upsample_factor
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
            len(self.args.attn_splits_list)
            == len(self.args.corr_radius_list)
            == len(self.args.prop_radius_list)
            == self.args.num_scales
        )

        for scale_idx in range(self.args.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if self.args.pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat(
                    (feature1, feature0), dim=0
                )

            feature0_ori, feature1_ori = feature0, feature1

            upsample_factor = self.args.upsample_factor * (
                2 ** (self.args.num_scales - 1 - scale_idx)
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

            attn_splits = self.args.attn_splits_list[scale_idx]
            corr_radius = self.args.corr_radius_list[scale_idx]
            prop_radius = self.args.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(
                feature0, feature1, attn_splits, self.args.feature_channels
            )

            # Transformer
            feature0, feature1 = self.transformer(
                feature0,
                feature1,
                attn_type=self.args.attn_type,
                attn_num_splits=attn_splits,
            )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(
                    feature0, feature1, self.args.pred_bidir_flow
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
            if self.args.pred_bidir_flow and scale_idx == 0:
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
            if self.training and scale_idx < self.args.num_scales - 1:
                flow_up = self.upsample_flow(
                    flow, feature0, bilinear=True, upsample_factor=upsample_factor
                )
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_preds.append(flow_up)

            if scale_idx == self.args.num_scales - 1:
                if not self.args.reg_refine:
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

                    assert self.args.num_reg_refine > 0
                    for refine_iter_idx in range(self.args.num_reg_refine):
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

                        if (
                            self.training
                            or refine_iter_idx == self.args.num_reg_refine - 1
                        ):
                            flow_up = upsample_flow_with_mask(
                                flow, up_mask, upsample_factor=self.args.upsample_factor
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

    def __init__(self, args: Namespace) -> None:
        args.num_scales = 2
        args.upsample_factor = 4
        args.attn_splits_list = (2, 8)
        args.corr_radius_list = (-1, 4)
        args.prop_radius_list = (-1, 1)
        super().__init__(args)


class UniMatchScale2With6Refinements(UniMatch):
    pretrained_checkpoints = {
        "mix": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-mixdata-398760b1.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-things-54d7505b.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-sintel-95ab1410.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/unimatch_scale2_refine6-kitti-0626279a.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        args.num_scales = 2
        args.upsample_factor = 4
        args.attn_splits_list = (2, 8)
        args.corr_radius_list = (-1, 4)
        args.prop_radius_list = (-1, 1)
        args.reg_refine = True
        args.num_reg_refine = 6
        super().__init__(args)
