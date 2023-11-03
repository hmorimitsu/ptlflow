from argparse import ArgumentParser, Namespace

from .FlowFormer.encoders import twins_svt_large, convnext_large
from .FlowFormer.PerCostFormer3.encoder import MemoryEncoder
from .FlowFormer.PerCostFormer3.decoder import MemoryDecoder
from .FlowFormer.PerCostFormer3.cnn import BasicEncoder
from ..base_model.base_model import BaseModel


class FlowFormerPlusPlus(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-chairs-228c2fec.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things-71639183.ckpt',
        'things288960': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things_288960-1a21a884.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-sintel-90b72ab7.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-kitti-453e8476.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=None,
            output_stride=32)

        H1, W1, H2, W2 = args.pic_size
        H_offset = (H1-H2) // 2
        W_offset = (W1-W2) // 2
        args.H_offset = H_offset
        args.W_offset = W_offset

        self.memory_encoder = MemoryEncoder(args)
        self.memory_decoder = MemoryDecoder(args)
        if args.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.args.pretrain, del_layers=args.del_layers)
        elif args.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        elif args.cnet == 'convnext':
            self.context_encoder = convnext_large(pretrained=self.args.pretrain)

        if args.pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--add_flow_token', action='store_true')
        # parser.add_argument('--context_concat', action='store_true')
        # parser.add_argument('--feat_cross_attn', action='store_true')
        # parser.add_argument('--gamma', type=float, default=0.8)
        # parser.add_argument('--max_flow', type=float, default=400.0)
        # parser.add_argument('--only_global', action='store_true')
        # parser.add_argument('--use_mlp', action='store_true')
        # parser.add_argument('--vertical_conv', action='store_true')
        parser.add_argument('--cnet', type=str, choices=('basicencoder', 'twins', 'convnext'), default='twins')
        parser.add_argument('--fnet', type=str, choices=('basicencoder', 'twins', 'convnext'), default='twins')
        parser.add_argument('--no_pretrain', action='store_false', dest='pretrain')
        parser.add_argument('--patch_size', type=int, default=8)
        parser.add_argument('--cost_heads_num', type=int, default=1)
        parser.add_argument('--cost_latent_input_dim', type=int, default=64)
        parser.add_argument('--cost_latent_token_num', type=int, default=8)
        parser.add_argument('--cost_latent_dim', type=int, default=128)
        parser.add_argument('--pe', type=str, choices=('exp', 'linear'), default='linear')
        parser.add_argument('--encoder_depth', type=int, default=3)
        parser.add_argument('--encoder_latent_dim', type=int, default=256)
        parser.add_argument('--decoder_depth', type=int, default=12)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--vert_c_dim', type=int, default=64)
        parser.add_argument('--query_latent_dim', type=int, default=64)
        parser.add_argument('--no_cost_encoder_res', action='store_false', dest='cost_encoder_res')

        parser.add_argument('--pic_size', type=int, nargs=4, default=(368, 496, 368, 496))
        parser.add_argument('--not_del_layers', action='store_false', dest='del_layers')
        parser.add_argument('--pretrain_mode', action='store_true')
        parser.add_argument('--use_convertor', action='store_true')
        parser.add_argument('--patch_embed', type=str, choices=('single', 'no_relu'), default='single')
        parser.add_argument('--cross_attn', type=str, choices=('all', 'part', 'rep', 'k3s2', '34'), default='all')
        parser.add_argument('--droppath', type=float, default=0.0)
        parser.add_argument('--vertical_encoder_attn', type=str, choices=('twins', 'NA', 'NA-twins'), default='twins')
        parser.add_argument('--use_patch', action='store_true')
        parser.add_argument('--fix_pe', action='store_true')
        parser.add_argument('--gt_r', type=int, default=15)
        parser.add_argument('--flow_or_pe', type=str, choices=('and', 'pe', 'flow'), default='and')
        parser.add_argument('--no_sc', action='store_true')
        parser.add_argument('--r_16', type=int, default=-1)
        parser.add_argument('--quater_refine', action='store_true')
        parser.add_argument('--use_rpe', action='store_true')
        parser.add_argument('--gma', type=str, choices=('GMA', 'GMA-SK'), default='GMA')
        parser.add_argument('--detach_local', action='store_true')
        return parser    

    def forward(self, inputs, flow_init=None, mask=None, output=None):
        """ Estimate optical flow between pair of frames """
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        if self.args.pretrain_mode:
            image1 = image1 * 255.0
            image2 = image2 * 255.0
            loss = self.pretrain_forward(image1, image2, mask=mask, output=output)
            return loss
        else:
            # Following https://github.com/princeton-vl/RAFT/
            image1 = 2 * image1 - 1.0
            image2 = 2 * image2 - 1.0

            data = {}
            
            context, _ = self.context_encoder(image1)
            context_quater = None

            cost_memory, cost_patches, feat_s_quater, feat_t_quater = self.memory_encoder(image1, image2, data, context)

            flow_predictions = self.memory_decoder(cost_memory, context, context_quater, feat_s_quater, feat_t_quater, data, flow_init=flow_init, cost_patches=cost_patches)

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
