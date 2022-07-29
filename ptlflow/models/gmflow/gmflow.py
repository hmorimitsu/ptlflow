from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

    def forward(self, outputs, inputs):
        """ Loss function defined over sequence of flow predictions """

        flow_preds = outputs['flow_preds']
        flow_gt = inputs['flows'][:, 0]
        valid = inputs['valids'][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)

            i_loss = (flow_preds[i] - flow_gt).abs()

            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class GMFlow(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-chairs-4922131e.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-things-5a18a9e8.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-sintel-d6f83ccd.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow-kitti-af50eb2e.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=16)

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=self.args.feature_channels, num_output_scales=self.args.num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=self.args.num_transformer_layers,
                                              d_model=self.args.feature_channels,
                                              nhead=self.args.num_head,
                                              attention_type=self.args.attention_type,
                                              ffn_dim_expansion=self.args.ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=self.args.feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + self.args.feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, self.args.upsample_factor ** 2 * 9, 1, 1, 0))

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--attention_type', type=str, choices=('full', 'swin'), default='swin')
        parser.add_argument('--attn_splits_list', type=int, nargs='+', default=(2,))
        parser.add_argument('--corr_radius_list', type=int, nargs='+', default=(-1,))
        parser.add_argument('--feature_channels', type=int, default=128)
        parser.add_argument('--ffn_dim_expansion', type=int, default=4)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument('--max_flow', type=float, default=400.0)
        parser.add_argument('--num_head', type=int, default=1)
        parser.add_argument('--num_scales', type=int, default=1)
        parser.add_argument('--num_transformer_layers', type=int, default=6)
        parser.add_argument('--pred_bidir_flow', action='store_true')
        parser.add_argument('--prop_radius_list', type=int, nargs='+', default=(-1,))
        parser.add_argument('--upsample_factor', type=int, default=8)
        return parser

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.args.upsample_factor, self.args.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.args.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.args.upsample_factor * h,
                                      self.args.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, inputs):
        """ Estimate optical flow between pair of frames """
        img0 = inputs['images'][:, 0]
        img1 = inputs['images'][:, 1]

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]

        flow_preds = []

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        flow = None

        assert len(self.args.attn_splits_list) == len(self.args.corr_radius_list) == len(self.args.prop_radius_list) == self.args.num_scales

        for scale_idx in range(self.args.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if self.args.pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.args.upsample_factor * (2 ** (self.args.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = self.args.attn_splits_list[scale_idx]
            corr_radius = self.args.corr_radius_list[scale_idx]
            prop_radius = self.args.prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.args.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, self.args.pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if self.args.pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.args.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.args.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)


        if self.training:
            outputs = {
                'flows': flow_up[:, None],
                'flow_preds': flow_preds
            }
        else:
            outputs = {
                'flows': flow_up[:, None]
            }
            
        return outputs


class GMFlowWithRefinement(GMFlow):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-chairs-88cdc009.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-things-e40899f5.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-sintel-ee46a2c4.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-kitti-b7bf2fda.ckpt'
    }

    def __init__(self, args: Namespace) -> None:
        args.attn_splits_list = (2, 2)
        args.corr_radius_list = (-1, 4)
        args.num_scales = 2
        args.prop_radius_list = (-1, 1)
        args.upsample_factor = 4
        super().__init__(args)
