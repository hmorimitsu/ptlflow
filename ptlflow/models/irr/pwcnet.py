from argparse import ArgumentParser, Namespace
from pathlib import Path

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler
import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense
from ..base_model.base_model import BaseModel
from .losses import MultiScaleEPE_PWC

class IRRPWCNet(BaseModel):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/irr_pwcnet-things-3f7fb8ca.ckpt'
    }

    def __init__(self, args):
        super(IRRPWCNet, self).__init__(
            args=args,
            loss_fn=MultiScaleEPE_PWC(args),
            output_stride=64)
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.args.num_chs)
        self.warping_layer = WarpingLayer()

        self.flow_estimators = nn.ModuleList()
        self.dim_corr = (self.args.search_range * 2 + 1) ** 2
        for l, ch in enumerate(self.args.num_chs[::-1]):
            if l > self.args.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr
            else:
                num_ch_in = self.dim_corr + ch + 2

            layer = FlowEstimatorDense(num_ch_in)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetwork(self.dim_corr + 32 + 2 + 448 + 2)

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

            # flow estimator
            if l == 0:
                x_intm, flow = self.flow_estimators[l](out_corr_relu)
            else:
                x_intm, flow = self.flow_estimators[l](torch.cat([out_corr_relu, x1, flow], dim=1))

            # upsampling or post-processing
            if l != self.args.output_level:
                flows.append(flow)
            else:
                flow_res = self.context_networks(torch.cat([x_intm, flow], dim=1))
                flow = flow + flow_res
                flows.append(flow)                
                break

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flows
            outputs['flows'] = (upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
        else:
            outputs['flows'] = (upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
        return outputs
