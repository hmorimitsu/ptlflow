from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils import bilinear_sampler, coords_grid, upflow8
from .gma_utils import Attention, Aggregate

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
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class GMA(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gma-chairs-d4ec321d.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gma-things-90aafb63.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gma-sintel-98d6f3d0.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gma-kitti-8ca3ec80.ckpt'
    }
    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8)
        parser.add_argument('--max_flow', type=float, default=1000.0)
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=1)
        parser.add_argument('--position_only', action='store_true')
        parser.add_argument('--position_and_content', action='store_true')
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, inputs, flow_init=None):
        """ Estimate optical flow between pair of frames """
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        # attention, att_c, att_p = self.att(inp)
        attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

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
                'flow_preds': flow_predictions
            }
        else:
            outputs = {
                'flows': flow_up[:, None],
                'flow_small': coords1 - coords0
            }
            
        return outputs