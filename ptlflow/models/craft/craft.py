from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock, TransCorrBlock
from .utils import coords_grid, upflow8, print0
from .gma import Attention
from .setrans import SETransConfig, SelfAttVisPosTrans
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

        # n_predictions = args.iters = 12
        n_predictions = len(flow_preds)    
        flow_loss = 0.0

        # exclude invalid pixels and extremely large displacements. 
        # MAX_FLOW = 400.
        valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < self.max_flow)

        for i in range(n_predictions):
            # Exponentially increasing weights. (Eq.7 in RAFT paper)
            # As i increases, flow_preds[i] is expected to be more and more accurate, 
            # so we are less and less tolerant to errors through gradually increased i_weight.
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class CRAFT(BaseModel):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-things-5a41930c.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-sintel-ff8e6563.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-kitti-4d99b0c1.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # default CRAFT corr_radius: 4
        if args.corr_radius == -1:
            args.corr_radius = 4
        print0("Lookup radius: %d" %args.corr_radius)
                
        if args.craft:
            self.inter_trans_config = SETransConfig()
            self.inter_trans_config.update_config(args)
            self.inter_trans_config.in_feat_dim = 256
            self.inter_trans_config.feat_dim    = 256
            self.inter_trans_config.max_pos_size     = 160
            # out_attn_scores_only implies no FFN nor V projection.
            self.inter_trans_config.out_attn_scores_only    = True                  # implies no FFN and no skip.
            self.inter_trans_config.attn_diag_cycles = 1000
            self.inter_trans_config.num_modes       = args.inter_num_modes          # default: 4
            self.inter_trans_config.tie_qk_scheme   = 'shared'                      # Symmetric Q/K
            self.inter_trans_config.qk_have_bias    = args.inter_qk_have_bias       # default: True
            self.inter_trans_config.pos_code_type   = args.inter_pos_code_type      # default: bias
            self.inter_trans_config.pos_code_weight = args.inter_pos_code_weight    # default: 0.5
            self.args.inter_trans_config = self.inter_trans_config
            print0("Inter-frame trans config:\n{}".format(self.inter_trans_config.__dict__))
            
            self.corr_fn = TransCorrBlock(self.inter_trans_config, radius=self.args.corr_radius,
                                          do_corr_global_norm=True)
        
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256,         norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch',    dropout=args.dropout)

        if args.f2trans != 'none':
            # f2_trans has the same configuration as GMA att, 
            # except that the feature dimension is doubled, and not out_attn_probs_only.
            self.f2_trans_config = SETransConfig()
            self.f2_trans_config.update_config(args)
            self.f2_trans_config.in_feat_dim = 256
            self.f2_trans_config.feat_dim  = 256
            # f2trans(x) = attn_aggregate(v(x)) + x. Here attn_aggregate and v (first_linear) both have 4 modes.
            self.f2_trans_config.has_input_skip = True
            # No FFN. f2trans simply aggregates similar features.
            # But there's still a V projection.
            self.f2_trans_config.has_FFN = False
            # When doing feature aggregation, set attn_mask_radius > 0 to exclude points that are too far apart, to reduce noises.
            # E.g., 64 corresponds to 64*8=512 pixels in the image space.
            self.f2_trans_config.attn_mask_radius = args.f2_attn_mask_radius
            # Not tying QK performs slightly better.
            self.f2_trans_config.tie_qk_scheme = None
            self.f2_trans_config.qk_have_bias  = False
            self.f2_trans_config.out_attn_probs_only    = False
            self.f2_trans_config.attn_diag_cycles   = 1000
            self.f2_trans_config.num_modes          = args.f2_num_modes             # default: 4
            self.f2_trans_config.pos_code_type      = args.intra_pos_code_type      # default: bias
            self.f2_trans_config.pos_code_weight    = args.f2_pos_code_weight       # default: 0.5
            self.f2_trans = SelfAttVisPosTrans(self.f2_trans_config, "F2 transformer")
            print0("F2-trans config:\n{}".format(self.f2_trans_config.__dict__))
            self.args.f2_trans_config = self.f2_trans_config
            
            if args.f1trans != 'none':
                args.corr_multiplier = 2
                if args.f1trans == 'shared':
                    # f1_trans and f2_trans are shared.
                    self.f1_trans = self.f2_trans
                elif args.f1trans == 'private':
                    # f1_trans is a private instance of SelfAttVisPosTrans.
                    self.f1_trans = SelfAttVisPosTrans(self.f2_trans_config, "F1 transformer")
                else:
                    breakpoint()
            else:
                self.f1_trans = None
                args.corr_multiplier = 1
                
        if args.use_setrans:
            self.intra_trans_config = SETransConfig()
            self.intra_trans_config.update_config(args)
            self.intra_trans_config.in_feat_dim = 128
            self.intra_trans_config.feat_dim  = 128
            # has_FFN & has_input_skip are for GMAUpdateBlock.aggregator.
            # Having FFN reduces performance. FYI, GMA also has no FFN.
            self.intra_trans_config.has_FFN = False
            self.intra_trans_config.has_input_skip = True
            self.intra_trans_config.attn_mask_radius = -1
            # Not tying QK performs slightly better.
            self.intra_trans_config.tie_qk_scheme = None
            self.intra_trans_config.qk_have_bias  = False
            self.intra_trans_config.out_attn_probs_only    = True
            self.intra_trans_config.attn_diag_cycles = 1000
            self.intra_trans_config.num_modes           = args.intra_num_modes          # default: 4
            self.intra_trans_config.pos_code_type       = args.intra_pos_code_type      # default: bias
            self.intra_trans_config.pos_code_weight     = args.intra_pos_code_weight    # default: 1
            self.att = SelfAttVisPosTrans(self.intra_trans_config, "Intra-frame attention")
            self.args.intra_trans_config = self.intra_trans_config
            print0("Intra-frame trans config:\n{}".format(self.intra_trans_config.__dict__))
        else:
            self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        # if args.use_setrans, initialization of GMAUpdateBlock.aggregator needs to access self.args.intra_trans_config.
        # So GMAUpdateBlock() construction has to be done after initializing intra_trans_config.
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.call_counter = 0

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
        parser.add_argument('--f1trans', type=str, choices=['none', 'shared', 'private'], default='none')
        parser.add_argument('--f2trans', type=str, choices=('none', 'full'), default='full')
        parser.add_argument('--f2_attn_mask_radius', type=int, default=-1)
        parser.add_argument('--f2_num_modes', type=int, default=4)
        parser.add_argument('--f2_pos_code_weight', type=float, default=0.5)
        parser.add_argument('--inter_num_modes', type=int, default=4)
        parser.add_argument('--inter_pos_code_type', type=str, choices=('bias', 'lsinu'), default='bias')
        parser.add_argument('--inter_pos_code_weight', type=float, default=0.5)
        parser.add_argument('--intra_pos_code_type', type=str, choices=('bias', 'lsinu'), default='bias')
        parser.add_argument('--intra_pos_code_weight', type=float, default=1.0)
        parser.add_argument('--intra_num_modes', type=int, default=4)
        parser.add_argument('--no_craft', action='store_false', dest='craft')
        parser.add_argument('--no_inter_qk_have_bias', action='store_false', dest='inter_qk_have_bias')
        parser.add_argument('--num_heads', type=int, default=1)
        parser.add_argument('--pos_bias_radius', type=int, default=7)
        parser.add_argument('--no_use_setrans', action='store_false', dest='use_setrans')
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
        fmap1o, fmap2o = None, None
        if self.args.f1trans != 'none':
            fmap1o = fmap1
            fmap1 = self.f1_trans(fmap1)
        if self.args.f2trans != 'none':
            fmap2o = fmap2
            fmap2  = self.f2_trans(fmap2)

        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        # correlation matrix: 7040*7040 (55*128=7040).
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # If not craft, the correlation volume is computed in the ctor.
        # If craft, the correlation volume is computed in corr_fn.update().
        if not self.args.craft:
            self.corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        # cnet: context network to extract features from image1 only.
        # cnet arch is the same as fnet. 
        # fnet extracts features specifically for correlation computation.
        # cnet_feat: extracted features focus on semantics of image1? 
        # (semantics of each pixel, used to guess its motion?)
        cnet_feat = self.cnet(image1)
            
        # Both fnet and cnet are BasicEncoder. output is from conv (no activation function yet).
        # net_feat, inp_feat: [1, 128, 55, 128]
        net_feat, inp_feat = torch.split(cnet_feat, [hdim, cdim], dim=1)
        net_feat = torch.tanh(net_feat)
        inp_feat = torch.relu(inp_feat)
        # attention, att_c, att_p = self.att(inp_feat)
        attention = self.att(inp_feat)
                
        # coords0 is always fixed as original coords.
        # coords1 is iteratively updated as coords0 + current estimated flow.
        # At this moment coords0 == coords1.
        coords0, coords1 = self.initialize_flow(image1)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        # If craft, the correlation volume is computed in corr_fn.update().
        if self.args.craft:
            # only update() once, instead of dynamically updating coords1.
            self.corr_fn.update(fmap1, fmap2, fmap1o, fmap2o, coords1, coords2=None)

        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            # corr: [6, 324, 50, 90]. 324: number of points in the neighborhood. 
            # radius = 4 -> neighbor points = (4*2+1)^2 = 81. 4 levels: x4 -> 324.
            corr = self.corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            
            # net_feat: hidden features of ConvGRU. 
            # inp_feat: input  features to ConvGRU.
            # up_mask is scaled to 0.25 of original values.
            # update_block: GMAUpdateBlock
            # In the first few iterations, delta_flow.abs().max() could be 1.3 or 0.8. Later it becomes 0.2~0.3.
            net_feat, up_mask, delta_flow = self.update_block(net_feat, inp_feat, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                # coords0 is fixed as original coords.
                # upflow8: upsize to 8 * height, 8 * width. 
                # flow value also *8 (scale the offsets proportionally to the resolution).
                flow_up = upflow8(coords1 - coords0)
            else:
                # The final high resolution flow field is found 
                # by using the mask to take a weighted combination over the neighborhood.
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
