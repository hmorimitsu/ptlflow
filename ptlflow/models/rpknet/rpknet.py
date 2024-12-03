# =============================================================================
# Copyright 2023 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable, ptlflow_trained
from ptlflow.utils.utils import forward_interpolate_batch
from .pwc_modules import rescale_flow, upsample2d_as
from .update_partial import UpdatePartialBlock
from .corr import get_corr_block
from .pkconv_slk_encoder import PKConvSLKEncoder
from .utils import ResidualPartialBlock, InterpolationTransition, get_norm_layer
from .pkconv import PKConv2d
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exclude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            pred = flow_preds[i]
            if (
                pred.shape[-2] != flow_gt.shape[-2]
                or pred.shape[-1] != flow_gt.shape[-1]
            ):
                pred = F.interpolate(
                    pred, size=flow_gt.shape[-2:], mode="bilinear", align_corners=True
                )
            i_weight = self.gamma ** (n_predictions - i - 1)
            diff = pred - flow_gt
            i_loss = (diff).abs()
            valid_loss = valid * i_loss
            flow_loss += i_weight * valid_loss.mean()

        return flow_loss


class UpNetPartial(nn.Module):
    def __init__(
        self,
        net_chs_fixed: int,
        enc_norm_type: str,
        use_norm_affine: bool,
        group_norm_num_groups: int,
        cache_pkconv_weights: bool,
    ) -> None:
        super().__init__()
        self.conv = PKConv2d(
            2 * net_chs_fixed,
            net_chs_fixed,
            1,
            cache_weights=cache_pkconv_weights,
        )
        self.act = nn.ReLU(inplace=True)
        self.res = ResidualPartialBlock(
            net_chs_fixed,
            net_chs_fixed,
            norm_layer=get_norm_layer(
                enc_norm_type,
                affine=use_norm_affine,
                num_groups=group_norm_num_groups,
            ),
            use_out_activation=False,
            cache_pkconv_weights=cache_pkconv_weights,
        )

    def forward(self, x):
        x = self.conv(x, x.shape[1] // 2)
        x = self.act(x)
        x = self.res(x, x.shape[1])
        return x


class RPKNet(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rpknet-chairs-a705b345.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rpknet-kitti-39504eb4.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rpknet-sintel-e7cc969e.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rpknet-things-f79b0d81.ckpt",
    }

    def __init__(
        self,
        pyramid_ranges: tuple[int, int] = (32, 8),
        iters: int = 12,
        input_pad_one_side: bool = False,
        input_bgr_to_rgb: bool = False,
        detach_flow: bool = True,
        corr_mode: str = "allpairs",
        upgate_norm_type: str = "group",
        use_norm_affine: bool = False,
        group_norm_num_groups: int = 8,
        corr_levels: int = 1,
        corr_range: int = 4,
        enc_norm_type: str = "group",
        enc_stem_stride: int = 2,
        enc_depth: int = 2,
        enc_mlp_ratio: float = 4.0,
        enc_hidden_chs: Sequence[int] = (32, 64, 96),
        enc_out_1x1_chs: str = "2.0",
        dec_gru_iters: int = 2,
        dec_gru_depth: int = 2,
        dec_gru_mlp_ratio: float = 4.0,
        dec_net_chs: Optional[int] = None,
        dec_inp_chs: Optional[int] = None,
        dec_motion_chs: int = 128,
        use_upsample_mask: bool = True,
        upmask_gradient_scale: float = 1.0,
        cache_pkconv_weights: bool = True,
        gamma: float = 0.8,
        max_flow: float = 400,
        **kwargs,
    ) -> None:
        for v in pyramid_ranges:
            assert (
                v > 0
            ), f"--pyramid_ranges values must be larger than 0, but found {v}"
            log_res = math.log2(v)
            assert (log_res) - int(
                log_res
            ) < 1e-3, f"--pyramid_ranges values must be powers of 2, but found {v}"
        num_recurrent_layers = int(math.log2(max(pyramid_ranges))) - 1
        output_stride = int(2 ** (num_recurrent_layers + 1))
        super().__init__(
            output_stride=output_stride,
            loss_fn=SequenceLoss(gamma, max_flow),
            **kwargs,
        )

        self.pyramid_ranges = pyramid_ranges
        self.iters = iters
        self.input_pad_one_side = input_pad_one_side
        self.input_bgr_to_rgb = input_bgr_to_rgb
        self.detach_flow = detach_flow
        self.corr_mode = corr_mode
        self.upgate_norm_type = upgate_norm_type
        self.use_norm_affine = use_norm_affine
        self.group_norm_num_groups = group_norm_num_groups
        self.corr_levels = corr_levels
        self.corr_range = corr_range
        self.enc_norm_type = enc_norm_type
        self.enc_stem_stride = enc_stem_stride
        self.enc_depth = enc_depth
        self.enc_mlp_ratio = enc_mlp_ratio
        self.enc_hidden_chs = enc_hidden_chs
        self.enc_out_1x1_chs = enc_out_1x1_chs
        self.dec_gru_iters = dec_gru_iters
        self.dec_gru_depth = dec_gru_depth
        self.dec_gru_mlp_ratio = dec_gru_mlp_ratio
        self.dec_net_chs = dec_net_chs
        self.dec_inp_chs = dec_inp_chs
        self.dec_motion_chs = dec_motion_chs
        self.use_upsample_mask = use_upsample_mask
        self.upmask_gradient_scale = upmask_gradient_scale
        self.cache_pkconv_weights = cache_pkconv_weights

        if isinstance(self.enc_out_1x1_chs, str):
            self.enc_out_1x1_chs = (
                float(self.enc_out_1x1_chs)
                if "." in self.enc_out_1x1_chs
                else int(self.enc_out_1x1_chs)
            )

        if isinstance(self.enc_out_1x1_chs, float):
            self.out_1x1_factor = self.enc_out_1x1_chs
            self.out_1x1_abs_chs = int(self.enc_out_1x1_chs * self.enc_hidden_chs[-1])
        else:
            self.out_1x1_factor = None
            self.out_1x1_abs_chs = self.enc_out_1x1_chs

        self.max_feat_chs = max(
            self.enc_hidden_chs[-1],
            self.out_1x1_abs_chs,
        )

        net_chs = self.dec_net_chs
        inp_chs = self.dec_inp_chs
        if net_chs is None or inp_chs is None:
            base_chs = self.out_1x1_abs_chs
            if base_chs < 1:
                base_chs = enc_hidden_chs[-1]

            base_chs = base_chs // 3 * 2

            if net_chs is None and inp_chs is None:
                net_chs = inp_chs = base_chs // 2
            elif net_chs is None and inp_chs is not None:
                net_chs = base_chs - inp_chs
            elif net_chs is not None and inp_chs is None:
                inp_chs = base_chs - net_chs
        self.net_chs_fixed = net_chs
        self.inp_chs_fixed = inp_chs

        self.pyramid_levels = [
            num_recurrent_layers + 1 - int(math.log2(v)) for v in self.pyramid_ranges
        ]
        self.min_pyr_level = min(self.pyramid_levels)
        self.max_pyr_level = max(self.pyramid_levels)

        pyr_range = [min(self.pyramid_ranges), max(self.pyramid_ranges)]

        self.fnet = PKConvSLKEncoder(
            pyr_range=pyr_range,
            hidden_chs=self.enc_hidden_chs,
            out_1x1_abs_chs=self.out_1x1_abs_chs,
            out_1x1_factor=self.out_1x1_factor,
            mlp_ratio=self.enc_mlp_ratio,
            depth=self.enc_depth,
            norm_layer=get_norm_layer(
                self.enc_norm_type,
                affine=self.use_norm_affine,
                num_groups=self.group_norm_num_groups,
            ),
            stem_stride=self.enc_stem_stride,
            cache_pkconv_weights=self.cache_pkconv_weights,
        )

        self.dim_corr = (self.corr_range * 2 + 1) ** 2 * self.corr_levels

        self.update_block = UpdatePartialBlock(
            pyramid_ranges=self.pyramid_ranges,
            corr_levels=self.corr_levels,
            corr_range=self.corr_range,
            net_chs_fixed=self.net_chs_fixed,
            inp_chs_fixed=self.inp_chs_fixed,
            group_norm_num_groups=self.group_norm_num_groups,
            use_norm_affine=self.use_norm_affine,
            dec_motion_chs=self.dec_motion_chs,
            dec_gru_depth=self.dec_gru_depth,
            dec_gru_iters=self.dec_gru_iters,
            dec_gru_mlp_ratio=self.dec_gru_mlp_ratio,
            use_upsample_mask=self.use_upsample_mask,
            upmask_gradient_scale=self.upmask_gradient_scale,
            cache_pkconv_weights=self.cache_pkconv_weights,
        )

        self.upnet_layer = InterpolationTransition(False, 2)

        self.upnet_gate_layer = UpNetPartial(
            net_chs_fixed=self.net_chs_fixed,
            enc_norm_type=enc_norm_type,
            use_norm_affine=use_norm_affine,
            group_norm_num_groups=group_norm_num_groups,
            cache_pkconv_weights=cache_pkconv_weights,
        )

        self.input_act = nn.ReLU()

        self.has_trained_on_ptlflow = True

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=self.dtype, device=self.device),
            torch.arange(wd, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def upsample_flow(self, flow, mask, factor, ch=2):
        """Upsample flow field [H/f, W/f, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, ch, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, ch, factor * H, factor * W)

    def forward(self, inputs):
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=self.input_bgr_to_rgb,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=(not self.input_pad_one_side),
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        flow_init = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            flow_init = inputs["prev_preds"]["flow_small"]

        flow_predictions, flow_small, flow_up = self.predict(image1, image2, flow_init)
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
        for i, p in enumerate(flow_predictions):
            flow_predictions[i] = self.postprocess_predictions(
                p, image_resizer, is_flow=True
            )
        outputs = {"flows": flow_up[:, None], "flow_small": flow_small}

        if self.training:
            outputs["flow_preds"] = flow_predictions

        return outputs

    def predict(self, x1_raw, x2_raw, flow_init):
        b, _, height_im, width_im = x1_raw.size()

        x1_pyramid = self.fnet(x1_raw)
        x2_pyramid = self.fnet(x2_raw)

        # outputs
        flows = []

        pred_stride = min(self.pyramid_ranges)

        ipyr = 0
        if self.training:
            ipyr = self.global_step % (len(self.pyramid_ranges) // 2)
        pyr_levels = self.pyramid_levels[2 * ipyr : 2 * ipyr + 2]

        start_level, output_level = pyr_levels
        pass_pyramid1 = x1_pyramid[start_level : output_level + 1]
        pass_pyramid2 = x2_pyramid[start_level : output_level + 1]

        level_diff = output_level - start_level
        iters_per_level = [int(math.ceil(float(self.iters) / (level_diff + 1)))] * (
            level_diff + 1
        )

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = pass_pyramid1[0].size()

        if flow_init is not None:
            flow = forward_interpolate_batch(flow_init)
        else:
            flow = torch.zeros(
                b_size,
                2,
                h_x1,
                w_x1,
                dtype=pass_pyramid1[0].dtype,
                device=pass_pyramid1[0].device,
            )

        net = None
        for l, (x1, x2) in enumerate(zip(pass_pyramid1, pass_pyramid2)):
            # Split feature channels into matching (x) and context (c)
            xh = x1.shape[1]
            ch = xh // 3
            x1, cn1 = torch.split(x1, [xh - ch, ch], dim=1)
            x2, cn2 = torch.split(x2, [xh - ch, ch], dim=1)
            halfch = ch // 2
            i1, n1 = torch.split(cn1, [ch - halfch, halfch], dim=1)
            i2, n2 = torch.split(cn2, [ch - halfch, halfch], dim=1)
            inp = torch.cat([i1, i2], 1)
            inp = self.input_act(inp)
            net_tmp = torch.cat([n1, n2], 1)

            coords0 = self.coords_grid(x1.shape[0], x1.shape[2], x1.shape[3])

            corr_fn = get_corr_block(
                x1,
                x2,
                self.corr_levels,
                self.corr_range,
                alternate_corr=self.corr_mode == "local",
            )

            if net is None:
                net = torch.tanh(net_tmp)
            else:
                net = self.upnet_layer(net, net_tmp.shape[1])
                net = torch.tanh(net)

                net_skip = torch.tanh(net_tmp)
                gate = torch.sigmoid(
                    self.upnet_gate_layer(torch.cat([net, net_skip], dim=1))
                )
                net = gate * net + (1.0 - gate) * net_skip

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = upsample2d_as(flow, x1, mode="bilinear")

            for _ in range(iters_per_level[l]):
                if self.detach_flow:
                    flow = flow.detach()

                # correlation
                out_corr = corr_fn(coords0 + flow)

                flow_res, net, mask = self.update_block(net, inp, out_corr, flow)

                flow = flow + flow_res

                out_flow = flow
                small_flow = out_flow
                out_flow = rescale_flow(out_flow, width_im, height_im, to_local=False)

                if l < (output_level - start_level) or mask is None:
                    out_flow = upsample2d_as(out_flow, x1_raw, mode="bilinear")
                else:
                    out_flow = self.upsample_flow(out_flow, mask, pred_stride)

                flows.append(out_flow)

        small_flow = rescale_flow(
            small_flow,
            pass_pyramid1[0].shape[-1],
            pass_pyramid1[0].shape[-2],
            to_local=False,
        )
        small_flow = upsample2d_as(small_flow, pass_pyramid1[0], mode="bilinear")

        return flows, small_flow, out_flow


@register_model
@trainable
@ptlflow_trained
class rpknet(RPKNet):
    pass
