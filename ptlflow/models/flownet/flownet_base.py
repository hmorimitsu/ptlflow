from argparse import ArgumentParser, Namespace

from torch.functional import norm

from ...base_model.base_model import BaseModel
from .losses import MultiScale


class FlowNetBase(BaseModel):
    def __init__(self,
                 args: Namespace):
        super(FlowNetBase, self).__init__(
            args=args,
            loss_fn=MultiScale(
                startScale=args.loss_start_scale,
                numScales=args.loss_num_scales,
                l_weight=args.loss_base_weight,
                norm=args.loss_norm
            ),
            output_stride=64)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        parser.add_argument('--input_channels', type=int, default=6)
        parser.add_argument('--batch_norm', action='store_true')
        parser.add_argument('--loss_start_scale', type=float, default=4)
        parser.add_argument('--loss_num_scales', type=int, default=5)
        parser.add_argument('--loss_base_weight', type=float, default=0.32)
        parser.add_argument('--loss_norm', type=str, default='L2')
        return parser
