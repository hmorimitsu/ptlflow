from argparse import ArgumentParser, Namespace
from pathlib import Path

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler
import torch
import torch.nn as nn

from .pwc_modules import conv, rescale_flow, upsample2d_as, initialize_msra
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense
from ..base_model.base_model import BaseModel
from .losses import MultiScaleEPE_PWC

class IRRPWCNetIRR(BaseModel):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/irr_pwcnet_irr-things-41a07190.ckpt'
    }

    def __init__(self, args):
        super(IRRPWCNetIRR, self).__init__(
            args=args,
            loss_fn=MultiScaleEPE_PWC(args),
            output_stride=64)
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.args.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.args.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.num_ch_in + 448 + 2)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.args.search_range+1, padding=0)

        initialize_msra(self.modules())

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=0.05)
        parser.add_argument('--search_range', type=int, default=4)
        parser.add_argument('--output_level', type=int, default=4)
        parser.add_argument('--num_chs', type=int, nargs='+', default=[3, 16, 32, 64, 96, 128, 196])
        parser.add_argument('--num_iters', type=int, default=1)
        return parser

    def forward(self, inputs):

        x1_raw = inputs['images'][:, 0]
        x2_raw = inputs['images'][:, 1]
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self.args.div_flow)

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr = out_corr.view(out_corr.shape[0], -1, out_corr.shape[3], out_corr.shape[4])
            out_corr = out_corr / x1.shape[1]
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            flow = rescale_flow(flow, self.args.div_flow, width_im, height_im, to_local=True)

            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flow = rescale_flow(flow, self.args.div_flow, width_im, height_im, to_local=False)
            flows.append(flow)

            # upsampling or post-processing
            if l == self.args.output_level:
                break

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flows
            outputs['flows'] = (upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
        else:
            outputs['flows'] = (upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
        return outputs
