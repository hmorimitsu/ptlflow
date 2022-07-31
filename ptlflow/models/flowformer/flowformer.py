from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn

from .cnn import BasicEncoder
from .encoder import MemoryEncoder
from .encoders import twins_svt_large
from .decoder import MemoryDecoder
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
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class FlowFormer(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-chairs-2b34ea4b.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-things-ab5f3255.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-sintel-27cc959a.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-kitti-1e45a6c8.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=8)

        if self.args.gma is None:
            self.args.gma = True  # Use GMA by default, unless

        self.memory_encoder = MemoryEncoder(args)
        self.memory_decoder = MemoryDecoder(args)
        if args.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.args.pretrain)
        elif args.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--add_flow_token', action='store_true')
        parser.add_argument('--cnet', type=str, choices=('basicencoder', 'twins'), default='twins')
        parser.add_argument('--context_concat', action='store_true')
        parser.add_argument('--cost_encoder_res', action='store_true')
        parser.add_argument('--cost_heads_num', type=int, default=1)
        parser.add_argument('--cost_latent_dim', type=int, default=128)
        parser.add_argument('--cost_latent_input_dim', type=int, default=64)
        parser.add_argument('--cost_latent_token_num', type=int, default=8)
        parser.add_argument('--decoder_depth', type=int, default=12)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--encoder_depth', type=int, default=3)
        parser.add_argument('--encoder_latent_dim', type=int, default=256)
        parser.add_argument('--feat_cross_attn', action='store_true')
        parser.add_argument('--fnet', type=str, choices=('basicencoder', 'twins'), default='twins')
        parser.add_argument('--gamma', type=float, default=0.8)
        parser.add_argument('--max_flow', type=float, default=400.0)
        parser.add_argument('--no_gma', action='store_false', dest='gma')
        parser.add_argument('--only_global', action='store_true')
        parser.add_argument('--patch_size', type=int, default=8)
        parser.add_argument('--pe', type=str, choices=('exp', 'linear'), default='linear')
        parser.add_argument('--pretrain', action='store_true')
        parser.add_argument('--query_latent_dim', type=int, default=64)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--vert_c_dim', type=int, default=64)
        parser.add_argument('--vertical_conv', action='store_true')
        return parser


    def forward(self, inputs, flow_init=None):
        """ Estimate optical flow between pair of frames """
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]

        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        data = {}

        if self.args.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        if self.training:
            outputs = {
                'flows': flow_predictions[0][:, None],
                'flow_preds': flow_predictions
            }
        else:
            outputs = {
                'flows': flow_predictions[0][:, None]
            }
            
        return outputs
