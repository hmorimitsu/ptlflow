from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn

from .pwc_modules import conv, upsample2d_as, rescale_flow, initialize_msra, compute_cost_volume
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense, OccContextNetwork, OccEstimatorDense
from .irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc
from ...base_model.base_model import BaseModel
from .losses import MultiScaleEPE_PWC_Bi_Occ_upsample


class ExternalIRRPWC(BaseModel):
    pretrained_checkpoints = {
        'chairs_occ': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_irr_pwc-chairs_occ-02066cc4.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_irr_pwc-things-c143e848.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_irr_pwc-sintel-6ad65777.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_irr_pwc-kitti-74d8868f.ckpt'
    }

    def __init__(self, args):
        super(ExternalIRRPWC, self).__init__(
            args=args,
            loss_fn=MultiScaleEPE_PWC_Bi_Occ_upsample(args),
            output_stride=64)
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.args.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.args.search_range * 2 + 1) ** 2
        self.num_ch_in_flo = self.dim_corr + 32 + 2
        self.num_ch_in_occ = self.dim_corr + 32 + 1

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in_flo)
        self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
        self.occ_estimators = OccEstimatorDense(self.num_ch_in_occ)
        self.occ_context_networks = OccContextNetwork(self.num_ch_in_occ + 448 + 1)
        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1)])

        self.conv_1x1_1 = conv(16, 3, kernel_size=1, stride=1, dilation=1)

        self.refine_flow = RefineFlow(2 + 1 + 32)
        self.refine_occ = RefineOcc(1 + 32 + 32)

        self.corr_params = {"pad_size": self.args.search_range, "kernel_size": 1, "max_disp": self.args.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}

        initialize_msra(self.modules())

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=0.05)
        parser.add_argument('--search_range', type=int, default=4)
        parser.add_argument('--output_level', type=int, default=4)
        parser.add_argument('--num_levels', type=int, default=7)
        parser.add_argument('--num_chs', type=int, nargs='+', default=[3, 16, 32, 64, 96, 128, 196])
        parser.add_argument('--num_iters', type=int, default=1)
        return parser

    def forward(self, inputs):

        x1_raw = inputs['images'][:, 0]
        x2_raw = inputs['images'][:, 1]
        batch_size, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        output_dict_eval = {}
        flows = []
        occs = []

        _, _, h_x1, w_x1, = x1_pyramid[0].size()
        flow_f = torch.zeros(batch_size, 2, h_x1, w_x1).to(dtype=x1_raw.dtype, device=x1_raw.device)
        flow_b = torch.zeros(batch_size, 2, h_x1, w_x1).to(dtype=x1_raw.dtype, device=x1_raw.device)
        occ_f = torch.zeros(batch_size, 1, h_x1, w_x1).to(dtype=x1_raw.dtype, device=x1_raw.device)
        occ_b = torch.zeros(batch_size, 1, h_x1, w_x1).to(dtype=x1_raw.dtype, device=x1_raw.device)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            if l <= self.args.output_level:

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    occ_f = upsample2d_as(occ_f, x1, mode="bilinear")
                    occ_b = upsample2d_as(occ_b, x2, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self.args.div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self.args.div_flow)

                # correlation
                out_corr_f = compute_cost_volume(x1, x2_warp, self.corr_params)
                out_corr_b = compute_cost_volume(x2, x1_warp, self.corr_params)

                out_corr_relu_f = self.leakyRELU(out_corr_f)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                if l != self.args.output_level:
                    x1_1by1 = self.conv_1x1[l](x1)
                    x2_1by1 = self.conv_1x1[l](x2)
                else:
                    x1_1by1 = x1
                    x2_1by1 = x2

                # concat and estimate flow
                flow_f = rescale_flow(flow_f, self.args.div_flow, width_im, height_im, to_local=True)
                flow_b = rescale_flow(flow_b, self.args.div_flow, width_im, height_im, to_local=True)

                x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
                x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
                flow_est_f = flow_f + flow_res_f
                flow_est_b = flow_b + flow_res_b

                flow_cont_f = flow_est_f + self.context_networks(torch.cat([x_intm_f, flow_est_f], dim=1))
                flow_cont_b = flow_est_b + self.context_networks(torch.cat([x_intm_b, flow_est_b], dim=1))

                # occ estimation
                x_intm_occ_f, occ_res_f = self.occ_estimators(torch.cat([out_corr_relu_f, x1_1by1, occ_f], dim=1))
                x_intm_occ_b, occ_res_b = self.occ_estimators(torch.cat([out_corr_relu_b, x2_1by1, occ_b], dim=1))
                occ_est_f = occ_f + occ_res_f
                occ_est_b = occ_b + occ_res_b

                occ_cont_f = occ_est_f + self.occ_context_networks(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                occ_cont_b = occ_est_b + self.occ_context_networks(torch.cat([x_intm_occ_b, occ_est_b], dim=1))

                # refinement
                img1_resize = upsample2d_as(x1_raw, flow_f, mode="bilinear")
                img2_resize = upsample2d_as(x2_raw, flow_b, mode="bilinear")
                flow_cont_f = rescale_flow(flow_cont_f, self.args.div_flow, width_im, height_im, to_local=False)
                flow_cont_b = rescale_flow(flow_cont_b, self.args.div_flow, width_im, height_im, to_local=False)
                img2_warp = self.warping_layer(img2_resize, flow_cont_f, height_im, width_im, self.args.div_flow)
                img1_warp = self.warping_layer(img1_resize, flow_cont_b, height_im, width_im, self.args.div_flow)
                
                # flow refine
                flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
                flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)

                flow_f = rescale_flow(flow_f, self.args.div_flow, width_im, height_im, to_local=False)
                flow_b = rescale_flow(flow_b, self.args.div_flow, width_im, height_im, to_local=False)

                # occ refine
                x2_1by1_warp = self.warping_layer(x2_1by1, flow_f, height_im, width_im, self.args.div_flow)
                x1_1by1_warp = self.warping_layer(x1_1by1, flow_b, height_im, width_im, self.args.div_flow)

                occ_f = self.refine_occ(occ_cont_f.detach(), x1_1by1, x1_1by1 - x2_1by1_warp)
                occ_b = self.refine_occ(occ_cont_b.detach(), x2_1by1, x2_1by1 - x1_1by1_warp)

                flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])
                occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])

            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                flows.append([flow_f, flow_b])

                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self.args.div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self.args.div_flow)
                flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im, self.args.div_flow)
                flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im, self.args.div_flow)

                if l != self.args.num_levels-1:
                    x1_in = self.conv_1x1_1(x1)
                    x2_in = self.conv_1x1_1(x2)
                    x1_w_in = self.conv_1x1_1(x1_warp)
                    x2_w_in = self.conv_1x1_1(x2_warp)
                else:
                    x1_in = x1
                    x2_in = x2
                    x1_w_in = x1_warp
                    x2_w_in = x2_warp

                occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([x1_in, x2_w_in, flow_f, flow_b_warp], dim=1))
                occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([x2_in, x1_w_in, flow_b, flow_f_warp], dim=1))

                occs.append([occ_f, occ_b])

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flows
            outputs['occ_preds'] = occs
            outputs['flows'] = (upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
            outputs['occs'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")[:, None]
        else:
            outputs['flows'] = (upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self.args.div_flow))[:, None]
            outputs['occs'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")[:, None]
        return outputs
