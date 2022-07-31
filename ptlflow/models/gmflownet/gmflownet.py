from argparse import ArgumentParser, Namespace
import configparser
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder, BasicConvEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from .swin_transformer import POLAUpdate, MixAxialPOLAUpdate
from .loss import compute_supervision_coarse, compute_coarse_loss, backwarp
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow
        self.use_matching_loss = args.use_matching_loss

    def forward(self, outputs, inputs):
        """ Loss function defined over sequence of flow predictions """

        flow_preds = outputs['flow_preds']
        soft_corr_map = outputs['soft_corr_map']
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]
        flow_gt = inputs['flows'][:, 0]
        valid = inputs['valids'][:, 0]

        # original RAFT loss
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None].float()  * i_loss).mean()

        if self.use_matching_loss:
            # enable global matching loss. Try to use it in late stages of the trianing
            img_2back1 = backwarp(image2, flow_gt)
            occlusionMap = (image1 - img_2back1).mean(1, keepdims=True) #(N, H, W)
            occlusionMap = torch.abs(occlusionMap) > 20
            occlusionMap = occlusionMap.float()

            conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8) # 8 from RAFT downsample

            matchLossCfg = configparser.ConfigParser()
            matchLossCfg.POS_WEIGHT = 1
            matchLossCfg.NEG_WEIGHT = 1
            matchLossCfg.FOCAL_ALPHA = 0.25
            matchLossCfg.FOCAL_GAMMA = 2.0
            matchLossCfg.COARSE_TYPE = 'cross_entropy'
            match_loss = compute_coarse_loss(soft_corr_map, conf_matrix_gt, matchLossCfg)

            flow_loss = flow_loss + 0.01*match_loss

        return flow_loss


class GMFlowNet(BaseModel):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflownet-things-9f061ac7.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflownet-kitti-712b4660.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if not hasattr(self.args, 'dropout'):
            self.args.dropout = 0

        # feature network, context network, and update block
        if self.args.use_mix_attn:
            self.fnet = nn.Sequential(
                            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                            MixAxialPOLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7)
                        )
        else:
            self.fnet = nn.Sequential(
                BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
                POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
            )

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8)
        parser.add_argument('--max_flow', type=float, default=400.0)
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--use_matching_loss', action='store_true')
        parser.add_argument('--use_mix_attn', action='store_true')
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, inputs, flow_init=None):
        """ Estimate optical flow between pair of frames """
        inputs['images'] = 2 * inputs['images'] - 1.0
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # # Self-attention update
        # fmap1 = self.transEncoder(fmap1)
        # fmap2 = self.transEncoder(fmap2)

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # Correlation as initialization
        N, fC, fH, fW = fmap1.shape
        corrMap = corr_fn.corrMap

        #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
        softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        if flow_init is not None:
            coords1 = coords1 + flow_init
        else:
            # print('matching as init')
            # mutual match selection
            match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            match21, match_idx21 = softCorrMap.max(dim=1)

            for b_idx in range(N):
                match21_b = match21[b_idx,:]
                match_idx12_b = match_idx12[b_idx,:]
                match21[b_idx,:] = match21_b[match_idx12_b]

            matched = (match12 - match21) == 0  # (N, fH*fW)
            coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            coords_index[matched] = match_idx12[matched]

            # matched coords
            coords_index = coords_index.reshape(N, fH, fW)
            coords_x = coords_index % fW
            coords_y = coords_index // fW

            coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            coords1 = coords_xy

        # Iterative update
        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if self.training:
            outputs = {
                'flows': flow_up[:, None],
                'flow_preds': flow_predictions,
                'soft_corr_map': softCorrMap
            }
        else:
            outputs = {
                'flows': flow_up[:, None],
                'flow_small': coords1 - coords0
            }
            
        return outputs


class GMFlowNetMix(GMFlowNet):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflownet_mix-things-8396f0a1.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflownet_mix-sintel-33492618.ckpt',
    }

    def __init__(self,
                 args: Namespace) -> None:
        args.use_mix_attn = True
        super().__init__(
            args=args)
