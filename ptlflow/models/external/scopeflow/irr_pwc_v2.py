from argparse import ArgumentParser
from pathlib import Path
import random

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler
import torch
import torch.nn as nn

from .pwc_modules import conv, upsample2d_as, rescale_flow, initialize_msra
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork,\
    FlowEstimatorDense, OccContextNetwork, OccEstimatorDense
from .irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc
from ...base_model.base_model import BaseModel
from .losses import MultiScaleEPE_PWC_Bi_Occ_upsample


class ExternalScopeFlow(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_scopeflow-chairs-ebfaa62d.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_scopeflow-things-70e22d63.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_scopeflow-kitti-a20c434d.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_scopeflow-sintel-21a91683.ckpt'
    }

    def __init__(self, args):
        super(ExternalScopeFlow, self).__init__(
            args=args,
            loss_fn=MultiScaleEPE_PWC_Bi_Occ_upsample(args),
            output_stride=64)
        self.pwc_groups = ['extractor', '1x1',
                           'flow_estimator',
                           'context', 'refine_flow',
                           'attention'] + ['occ'] * 3

        # Init sizes
        self.args = args
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.ch_proj_size = 32
        self.min_layer = len(self.num_chs) - self.output_level - 1

        # Init generic layers
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()
        self.correlation = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.args.search_range+1, padding=0)

        # Calc dimensions
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in_flo = self.dim_corr + self.ch_proj_size + 2
        self.num_ch_in_occ = self.dim_corr + self.ch_proj_size + 1

        # Init PWC modules
        self.flow_estimators = FlowEstimatorDense(self.num_ch_in_flo)
        self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
        self.occ_estimators = OccEstimatorDense(self.num_ch_in_occ)
        self.occ_context_networks = OccContextNetwork(self.num_ch_in_occ + 448 + 1)
        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)
        if hasattr(args, 'dropout') and self.args.dropout:
            print("Starting dropout!")
            self.dropout = nn.Dropout2d(p=self.args.dropout, inplace=False)

        self.conv_1x1 = nn.ModuleList([
            conv(self.num_chs[ch], self.ch_proj_size, kernel_size=1, stride=1, dilation=1)
            for ch in range(len(self.num_chs) - 1, self.min_layer, -1)])

        self.conv_1x1_1 = conv(self.num_chs[self.min_layer - 1], 3, kernel_size=1, stride=1, dilation=1)

        # Initialize attention
        if hasattr(args, 'self_attention') and self.args.self_attention:
            print("Initializing self attention!")
            from models.self_attention import AttentionConv, AttentionStem, SelfAttentionConv2d
            self.attention_conv = SelfAttentionConv2d(2, 2, kernel_size=5, padding=2)
        self.refine_flow = RefineFlow(2 + 1 + self.ch_proj_size, args.bilateral_mask)
        self.refine_occ = RefineOcc(1 + self.ch_proj_size + self.ch_proj_size)

        # Init weights
        initialize_msra(self.modules())
        self.param_groups = self._get_param_groups(self.pwc_groups)

        if hasattr(args, 'freeze_list') and args.freeze_list:
            self.freezed_params = args.freeze_list.split(',')
            self._freeze()
            self.random_freeze = False
        elif hasattr(args, 'random_freeze') and args.random_freeze:
            self.random_freeze = True
        self.mean_per_module = None

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=0.05)
        parser.add_argument('--search_range', type=int, default=4)
        parser.add_argument('--output_level', type=int, default=4)
        parser.add_argument('--num_levels', type=int, default=7)
        parser.add_argument('--num_chs', type=int, nargs='+', default=[3, 16, 32, 64, 96, 128, 196])
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--self_attention', action='store_true')
        parser.add_argument('--bilateral_mask', action='store_true')
        parser.add_argument('--random_freeze', action='store_true')
        parser.add_argument('--seploss', action='store_true')
        parser.add_argument('--loss_perc', action='store_true')
        return parser

    def forward(self, inputs):

        # Extract input
        x1_raw = inputs['images'][:, 0]
        x2_raw = inputs['images'][:, 1]
        batch_size, _, height_im, width_im = x1_raw.size()

        # Get pyramid, on the bottom level are original images
        cur_x1 = self.feature_pyramid_extractor(x1_raw)
        cur_x2 = self.feature_pyramid_extractor(x2_raw)
        if hasattr(self.args, 'dropout') and self.args.dropout:
            # print("Dropout in ctx!")
            for xl in range(len(cur_x1)):
                # cur_x1[xl] = self.dropout(cur_x1[xl])
                cur_x2[xl] = self.dropout(cur_x2[xl])
        x1_pyramid = cur_x1 + [x1_raw]
        x2_pyramid = cur_x2 + [x2_raw]

        # Set output data structures
        output_dict = {}
        output_dict_eval = {}
        flows = []
        occs = []

        # Pre allocate output tensors
        flow_f, flow_b, occ_f, occ_b = self._allocate_out_tensors(x1_pyramid, batch_size)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            if l <= self.output_level:
                # Warp
                x1_warp, x2_warp, flow_f, flow_b, occ_f, occ_b = self._warp(l, x1, x2,
                                                                            flow_f,
                                                                            flow_b, occ_f,
                                                                            occ_b,
                                                                            height_im,
                                                                            width_im)

                # Correlate
                out_corr_relu_f = self._correlate(x1, x2_warp)
                out_corr_relu_b = self._correlate(x2, x1_warp)

                # Squash to channels projection size
                x1_1by1, x2_1by1 = self._squash(l, x1, x2)

                # Rescale Flow
                flow_f, flow_b = self._rescale_flow(flow_f, flow_b, height_im, width_im)

                # Estimate flow
                flow_cont_f, flow_cont_b = self._estimate_flow(out_corr_relu_f,
                                                               out_corr_relu_b,
                                                               x1_1by1, x2_1by1,
                                                               flow_f, flow_b)

                # Estimate occlusions
                occ_cont_f, occ_cont_b = self._estimate_occ(out_corr_relu_f,
                                                            out_corr_relu_b,
                                                            x1_1by1, x2_1by1,
                                                            occ_f, occ_b)

                # Prepare refinement inputs
                img1_resize, img2_resize, img1_warp, img2_warp, flow_cont_f, flow_cont_b = \
                    self._resize_and_warp_inputs(
                    x1_raw, x2_raw, flow_f, flow_b, flow_cont_f, flow_cont_b, height_im,
                    width_im)

                # Refine Flow
                flow_f, flow_cont_f, flow_b, flow_cont_b = self._refine_flow(
                    flow_cont_f, flow_cont_b, img1_resize, img2_resize, img1_warp,
                    img2_warp, x1_1by1, x2_1by1, height_im, width_im)

                # Refine Occlusions
                occ_f, occ_cont_f, occ_b, occ_cont_f = self._refine_occ(
                    x1_1by1, x2_1by1, flow_f, flow_b, occ_cont_f, occ_cont_b,
                    height_im, width_im)

                # Collect layer's outputs
                flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])
                occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])

            else:
                # Final flow upsampling
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")

                # Final occ upsampling
                occ_f, occ_b = self.occ_upsampling(l, x1, x2, flow_f, flow_b,
                                                   occ_f, occ_b, height_im, width_im)

                # Apply self attention
                if hasattr(self.args, 'self_attention') and self.args.self_attention:
                    flow_f = self.attention_conv(flow_f)
                    flow_b = self.attention_conv(flow_b)

                # Aggregate outputs
                flows.append([flow_f, flow_b])
                occs.append([occ_f, occ_b])

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flows
            outputs['occ_preds'] = occs
            outputs['flows'] = (upsample2d_as(flow_f, x1_raw, mode="bilinear") / self.args.div_flow)[:, None]
            outputs['occs'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")[:, None]
        else:
            outputs['flows'] = (upsample2d_as(flow_f, x1_raw, mode="bilinear") / self.args.div_flow)[:, None]
            outputs['occs'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")[:, None]
        return outputs

    def submodules_summary(self):
        print("Trainable submodules: {}".format({p[0] for p in self.named_parameters()
                                                 if p[1].requires_grad}))
        if not self.mean_per_module:
            print("Initializing mean per module")
            self.mean_per_module = {p[0]:p[1].data.mean()
                                    for p in self.named_parameters()}
        else:

            changes = {p[0]:p[1].data.mean() for p in self.named_parameters()
                       if p[1].data.mean() != self.mean_per_module[p[0]]}
            self.mean_per_module = {p[0]: p[1].data.mean() for p in
                                    self.named_parameters()}
            # print("Mean per submodule: {}".format(self.mean_per_module))
            print(f"Changes: {changes}")

    def freeze_random_weights(self):
        self.freezed_params = random.sample(self.pwc_groups,
                                            random.randrange(1, len(self.pwc_groups) / 2))
        self._freeze(verify=True)

    def _freeze(self, verify=False):
        print(f"Freezing groups {self.freezed_params}")

        if verify:
            for param in self.named_parameters():
                if not param[1].requires_grad:
                    param[1].requires_grad = True

        fkeys = set()
        for freezing_group in self.freezed_params:
            for key, param in self.param_groups[freezing_group].items():
                param.requires_grad = False
                if param.grad is not None:
                    param.grad.data.zero_()
                    # print("Freezing {}, grad sum is {}".format(key, param.grad.sum()))
                fkeys.add(key)

        assert len({p[0] for p in self.named_parameters() if not p[1].requires_grad}) == \
               len(fkeys)
        print(f"Freezed keys {fkeys}")
        print("Params with grad {}".format({p[0] for p in self.named_parameters()
                                            if p[1].requires_grad}))

    def _get_param_groups(self, keys):
        pgroups = {}
        for substr in keys:
            pgroups[substr] = {k: t for k, t in self.named_parameters() if substr in k}

        # Verify each param has at least one group
        unique_group_params = set([s for p in [list(pgroups[k].keys()) for k in keys]
                                       for s in p])
        assert len(unique_group_params) == len(list(self.named_parameters()))

        return pgroups

    @staticmethod
    def _allocate_out_tensors(x1_pyramid, batch_size):
        _, _, h_x1, w_x1, = x1_pyramid[0].size()
        flow_f = torch.zeros(batch_size, 2, h_x1, w_x1).to(dtype=x1_pyramid[0].dtype, device=x1_pyramid[0].device)
        flow_b = torch.zeros(batch_size, 2, h_x1, w_x1).to(dtype=x1_pyramid[0].dtype, device=x1_pyramid[0].device)
        occ_f = torch.zeros(batch_size, 1, h_x1, w_x1).to(dtype=x1_pyramid[0].dtype, device=x1_pyramid[0].device)
        occ_b = torch.zeros(batch_size, 1, h_x1, w_x1).to(dtype=x1_pyramid[0].dtype, device=x1_pyramid[0].device)
        return flow_f, flow_b, occ_f, occ_b

    def _warp(self, l, x1, x2, flow_f, flow_b, occ_f, occ_b, height_im, width_im):
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
        return x1_warp, x2_warp, flow_f, flow_b, occ_f, occ_b

    @staticmethod
    def collate_corr(corr, first_inp):
        b, ph, pw, h, w = corr.size()
        output_collated = corr.view(b, ph * pw, h, w)
        return output_collated / float(first_inp.size(1))

    def _correlate(self, x1, x2_warp):
        out_corr = self.correlation(x1, x2_warp)
        out_corr = out_corr.view(out_corr.shape[0], -1, out_corr.shape[3], out_corr.shape[4])
        out_corr = out_corr / x1.shape[1]
        # out_corr = self.collate_corr(out_corr, x1)
        out_corr_relu = self.leakyRELU(out_corr)
        return out_corr_relu

    def _squash(self, l, x1, x2):
        if l != self.output_level:
            x1_1by1 = self.conv_1x1[l](x1)
            x2_1by1 = self.conv_1x1[l](x2)
        else:
            x1_1by1 = x1
            x2_1by1 = x2
        return x1_1by1, x2_1by1

    def _rescale_flow(self, flow_f, flow_b, height_im, width_im):
        flow_f = rescale_flow(flow_f, self.args.div_flow, width_im, height_im, to_local=True)
        flow_b = rescale_flow(flow_b, self.args.div_flow, width_im, height_im, to_local=True)
        return flow_f, flow_b

    def _estimate_flow(self, out_corr_relu_f, out_corr_relu_b, x1_1by1, x2_1by1,
                       flow_f, flow_b):

        # Estimate forward and backward residual flows
        x_intm_f, flow_res_f = self.flow_estimators(
            torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
        x_intm_b, flow_res_b = self.flow_estimators(
            torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))

        # Add residual flow to the aggregated flow
        flow_est_f = flow_f + flow_res_f
        flow_est_b = flow_b + flow_res_b

        # Estimate residual context based refinement
        ctxf = self.context_networks(torch.cat([x_intm_f, flow_est_f], dim=1))
        ctxb = self.context_networks(torch.cat([x_intm_b, flow_est_b], dim=1))
        flow_cont_f = flow_est_f + ctxf
        flow_cont_b = flow_est_b + ctxb

        return flow_cont_f, flow_cont_b

    def _estimate_occ(self, out_corr_relu_f, out_corr_relu_b,  x1_1by1, x2_1by1,
                      occ_f, occ_b):

        # Estimate residual occlusions
        x_intm_occ_f, occ_res_f = self.occ_estimators(
            torch.cat([out_corr_relu_f, x1_1by1, occ_f], dim=1))
        x_intm_occ_b, occ_res_b = self.occ_estimators(
            torch.cat([out_corr_relu_b, x2_1by1, occ_b], dim=1))

        # Add residual occlusions to the aggregated occlusions
        occ_est_f = occ_f + occ_res_f
        occ_est_b = occ_b + occ_res_b

        # Estimate residual context based refinement
        occ_cont_f = occ_est_f + self.occ_context_networks(
            torch.cat([x_intm_occ_f, occ_est_f], dim=1))
        occ_cont_b = occ_est_b + self.occ_context_networks(
            torch.cat([x_intm_occ_b, occ_est_b], dim=1))
        return occ_cont_f, occ_cont_b

    def _resize_and_warp_inputs(self, x1_raw, x2_raw, flow_f, flow_b,
                                flow_cont_f, flow_cont_b,
                                height_im, width_im):

        # Resizing input images for warping
        img1_resize = upsample2d_as(x1_raw, flow_f, mode="bilinear")
        img2_resize = upsample2d_as(x2_raw, flow_b, mode="bilinear")

        # Rescaling flow to the layer's size
        rescaled_f = rescale_flow(flow_cont_f, self.args.div_flow, width_im, height_im,
                                  to_local=False)
        rescaled_b = rescale_flow(flow_cont_b, self.args.div_flow, width_im, height_im,
                                  to_local=False)

        # Warping resized images
        img2_warp = self.warping_layer(img2_resize, rescaled_f, height_im, width_im,
                                       self.args.div_flow)
        img1_warp = self.warping_layer(img1_resize, rescaled_b, height_im, width_im,
                                       self.args.div_flow)

        return img1_resize, img2_resize, img1_warp, img2_warp, rescaled_f, rescaled_b

    def _refine_flow(self, flow_cont_f, flow_cont_b, img1_resize, img2_resize, img1_warp,
                     img2_warp, x1_1by1, x2_1by1, height_im, width_im):
        flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
        flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)

        flow_cont_f = rescale_flow(flow_cont_f, self.args.div_flow, width_im, height_im,
                                   to_local=False)
        flow_cont_b = rescale_flow(flow_cont_b, self.args.div_flow, width_im, height_im,
                                   to_local=False)
        flow_f = rescale_flow(flow_f, self.args.div_flow, width_im, height_im, to_local=False)
        flow_b = rescale_flow(flow_b, self.args.div_flow, width_im, height_im, to_local=False)
        return flow_f, flow_cont_f, flow_b, flow_cont_b

    def _refine_occ(self, x1_1by1, x2_1by1, flow_f, flow_b, occ_cont_f, occ_cont_b,
                    height_im, width_im):
        x2_1by1_warp = self.warping_layer(x2_1by1, flow_f, height_im, width_im,
                                          self.args.div_flow)
        x1_1by1_warp = self.warping_layer(x1_1by1, flow_b, height_im, width_im,
                                          self.args.div_flow)

        occ_f = self.refine_occ(occ_cont_f.detach(), x1_1by1, x1_1by1 - x2_1by1_warp)
        occ_b = self.refine_occ(occ_cont_b.detach(), x2_1by1, x2_1by1 - x1_1by1_warp)
        return occ_f, occ_cont_f, occ_b, occ_cont_f

    def occ_upsampling(self, l, x1, x2, flow_f, flow_b, occ_f, occ_b,
                       height_im, width_im):
        x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self.args.div_flow)
        x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self.args.div_flow)
        flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im,
                                         self.args.div_flow)
        flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im,
                                         self.args.div_flow)

        if l != self.num_levels - 1:
            x1_in = self.conv_1x1_1(x1)
            x2_in = self.conv_1x1_1(x2)
            x1_w_in = self.conv_1x1_1(x1_warp)
            x2_w_in = self.conv_1x1_1(x2_warp)
        else:
            x1_in = x1
            x2_in = x2
            x1_w_in = x1_warp
            x2_w_in = x2_warp

        occ_f = self.occ_shuffle_upsample(occ_f,
                                          torch.cat([x1_in, x2_w_in, flow_f, flow_b_warp],
                                                    dim=1))
        occ_b = self.occ_shuffle_upsample(occ_b,
                                          torch.cat([x2_in, x1_w_in, flow_b, flow_f_warp],
                                                    dim=1))
        return occ_f, occ_b
