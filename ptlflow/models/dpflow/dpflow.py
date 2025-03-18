# =============================================================================
# Copyright 2025 Henrique Morimitsu
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
from typing import Optional

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model.base_model import BaseModel
from .corr import CorrBlock, AlternateCorrBlock
from .pwc_modules import rescale_flow, upsample2d_as
from .cgu_bidir_dual_encoder import CGUBidirDualEncoder
from .update import UpdateBlock
from .utils import (
    compute_pyramid_levels,
    get_activation,
    get_norm,
)
from ptlflow.utils.utils import forward_interpolate_batch
from ptlflow.utils.registry import register_model, trainable, ptlflow_trained

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


class SequenceLoss(nn.Module):
    def __init__(self, loss: str, max_flow: float, gamma: float):
        super().__init__()
        self.loss = loss
        self.max_flow = max_flow
        self.gamma = gamma

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

            if self.loss == "l1" or outputs["nf_preds"][i] is None:
                diff = pred - flow_gt
                i_loss = (diff).abs()
                valid_loss = valid * i_loss
                flow_loss += i_weight * valid_loss.mean()
            elif self.loss == "laplace":
                loss_i = outputs["nf_preds"][i]
                final_mask = (
                    (~torch.isnan(loss_i.detach()))
                    & (~torch.isinf(loss_i.detach()))
                    & valid
                )
                flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

        return flow_loss


class DPFlow(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-chairs-f94e717a.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-kitti-4e97eac6.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-sintel-b44b072c.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-spring-69bac7fa.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-things-2012b5d6.ckpt",
    }

    def __init__(
        self,
        pyramid_levels: Optional[int] = None,
        iters_per_level: int = 4,
        detach_flow: bool = True,
        use_norm_affine: bool = False,
        group_norm_num_groups: int = 8,
        corr_mode: str = "allpairs",  # "allpairs" or "local"
        corr_levels: int = 1,
        corr_range: int = 4,
        activation_function: str = "orig",  # "orig", "relu", "gelu", "silu", or "mish"
        enc_network: str = "cgu_bidir_dual",  # "cgu", "cgu_bidir", "cgu_bidir_dual", "cgu_dual", "next_bidir_dual", "swin"
        enc_norm_type: str = "group",  # "none", "group", "layer", or "batch"
        enc_depth: int = 4,
        enc_mlp_ratio: float = 2.0,
        enc_mlp_in_kernel_size: int = 1,
        enc_mlp_out_kernel_size: int = 1,
        enc_hidden_chs: list[int] = (64, 96, 128),
        enc_num_out_stages: int = 1,
        enc_out_1x1_chs: str = "384",
        dec_gru_norm_type: str = "layer",  # "none", "group", "layer", or "batch"
        dec_gru_iters: int = 1,
        dec_gru_depth: int = 4,
        dec_gru_mlp_ratio: float = 2.0,
        dec_gru_mlp_in_kernel_size: int = 1,
        dec_gru_mlp_out_kernel_size: int = 1,
        dec_net_chs: int = 128,
        dec_inp_chs: int = 128,
        dec_motion_chs: int = 128,
        dec_flow_kernel_size: int = 7,
        dec_flow_head_chs: int = 256,
        dec_motenc_corr_hidden_chs: int = 256,
        dec_motenc_corr_out_chs: int = 192,
        dec_motenc_flow_hidden_chs: int = 128,
        dec_motenc_flow_out_chs: int = 64,
        use_upsample_mask: bool = True,
        upmask_gradient_scale: float = 1.0,
        cgu_mlp_dw_kernel_size: int = 7,
        cgu_fusion_gate_activation: str = "gelu",  # "linear", "sigmoid", "relu", "gelu", "silu", or "mish"
        cgu_mlp_use_dw_conv: bool = True,
        cgu_mlp_activation_function: str = "gelu",  # "linear", "sigmoid", "relu", "gelu", "silu", or "mish"
        cgu_layer_scale_init_value: float = 0.01,
        loss: str = "laplace",  # "l1" or "laplace"
        gamma: float = 0.8,
        max_flow: float = 400.0,
        use_var: bool = True,
        var_min: float = 0.0,
        var_max: float = 10.0,
        **kwargs,
    ):
        if pyramid_levels is not None:
            assert pyramid_levels > 2, "Only --model.pyramid_levels >= 3 is supported."
            output_stride = int(2 ** (pyramid_levels + 2))
            if enc_network == "swin_bidir_dual":
                output_stride *= 2
        else:
            logger.info(
                f"DPFlow: --model.pyramid_levels is not set, the number of pyramid levels will be inferred from the input size."
            )
            output_stride = None
            self.extra_output_stride = 1 if enc_network == "swin_bidir_dual" else 0

        super(DPFlow, self).__init__(
            loss_fn=SequenceLoss(loss=loss, max_flow=max_flow, gamma=gamma),
            output_stride=output_stride,
            **kwargs,
        )

        self.pyramid_levels = pyramid_levels
        self.iters_per_level = iters_per_level
        self.corr_mode = corr_mode
        self.corr_range = corr_range
        self.corr_levels = corr_levels
        self.detach_flow = detach_flow
        self.loss = loss
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max

        activation_function = get_activation(activation_function)

        enc_out_1x1_chs = (
            float(enc_out_1x1_chs)
            if (isinstance(enc_out_1x1_chs, str) and "." in enc_out_1x1_chs)
            else int(enc_out_1x1_chs)
        )

        if isinstance(enc_out_1x1_chs, float):
            out_1x1_factor = enc_out_1x1_chs
            out_1x1_abs_chs = int(enc_out_1x1_chs * enc_hidden_chs[-1])
        else:
            out_1x1_factor = None
            out_1x1_abs_chs = enc_out_1x1_chs

        self.max_feat_chs = max(
            enc_hidden_chs[-1],
            out_1x1_abs_chs,
        )

        net_chs = dec_net_chs
        inp_chs = dec_inp_chs
        if net_chs is None or inp_chs is None:
            base_chs = out_1x1_abs_chs
            if base_chs < 1:
                base_chs = enc_hidden_chs[-1]

            base_chs = base_chs // 3 * 2

            if net_chs is None and inp_chs is None:
                net_chs = inp_chs = base_chs // 2
            elif net_chs is None and inp_chs is not None:
                net_chs = base_chs - inp_chs
            elif net_chs is not None and inp_chs is None:
                inp_chs = base_chs - net_chs
        net_chs_fixed = net_chs
        inp_chs_fixed = inp_chs

        enc_norm_layer = get_norm(
            enc_norm_type,
            affine=use_norm_affine,
            num_groups=group_norm_num_groups,
        )
        self.fnet = CGUBidirDualEncoder(
            pyramid_levels=pyramid_levels,
            hidden_chs=enc_hidden_chs,
            out_1x1_abs_chs=out_1x1_abs_chs,
            out_1x1_factor=out_1x1_factor,
            num_out_stages=enc_num_out_stages,
            activation_function=activation_function,
            norm_layer=enc_norm_layer,
            depth=enc_depth,
            mlp_ratio=enc_mlp_ratio,
            mlp_use_dw_conv=cgu_mlp_use_dw_conv,
            mlp_dw_kernel_size=cgu_mlp_dw_kernel_size,
            mlp_in_kernel_size=enc_mlp_in_kernel_size,
            mlp_out_kernel_size=enc_mlp_out_kernel_size,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
        )

        self.dim_corr = (corr_range * 2 + 1) ** 2 * corr_levels

        dec_gru_norm_layer = get_norm(
            dec_gru_norm_type,
            affine=use_norm_affine,
            num_groups=group_norm_num_groups,
        )
        self.update_block = UpdateBlock(
            dec_motenc_corr_hidden_chs=dec_motenc_corr_hidden_chs,
            dec_motenc_corr_out_chs=dec_motenc_corr_out_chs,
            dec_motenc_flow_hidden_chs=dec_motenc_flow_hidden_chs,
            dec_motenc_flow_out_chs=dec_motenc_flow_out_chs,
            corr_levels=corr_levels,
            corr_range=corr_range,
            dec_flow_kernel_size=dec_flow_kernel_size,
            dec_motion_chs=dec_motion_chs,
            activation_function=activation_function,
            net_chs_fixed=net_chs_fixed,
            inp_chs_fixed=inp_chs_fixed,
            dec_gru_norm_layer=dec_gru_norm_layer,
            dec_gru_depth=dec_gru_depth,
            dec_gru_iters=dec_gru_iters,
            dec_gru_mlp_ratio=dec_gru_mlp_ratio,
            cgu_mlp_use_dw_conv=cgu_mlp_use_dw_conv,
            cgu_mlp_dw_kernel_size=cgu_mlp_dw_kernel_size,
            dec_gru_mlp_in_kernel_size=dec_gru_mlp_in_kernel_size,
            dec_gru_mlp_out_kernel_size=dec_gru_mlp_out_kernel_size,
            cgu_layer_scale_init_value=cgu_layer_scale_init_value,
            dec_flow_head_chs=dec_flow_head_chs,
            loss=loss,
            use_upsample_mask=use_upsample_mask,
            upmask_gradient_scale=upmask_gradient_scale,
        )

        act = nn.ReLU if activation_function is None else activation_function
        self.input_act = act(inplace=True)

        self.current_output_stride = output_stride

        self.has_shown_input_message = False
        self.has_shown_altcuda_message = False

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=self.dtype, device=self.device),
            torch.arange(wd, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).to(dtype=self.dtype)
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

    def _show_input_message(self, images):
        pyr_levels = compute_pyramid_levels(images)
        recommended_pyr_levels = pyr_levels  # 3 for 1K, 4 for 2K, etc.

        logger.info(
            f"DPFlow: Using {self.pyramid_levels} pyramid levels and {self.iters_per_level} iterations per level."
        )
        logger.info(
            f"DPFlow: Processing inputs of resolution {images.shape[-1]} x {images.shape[-2]}"
        )
        logger.info(f"DPFlow: Correlation mode: {self.corr_mode}")

        if recommended_pyr_levels != self.pyramid_levels:
            logger.info(
                "DPFlow: For this input size, you may get better results by setting --pyramid_levels {}",
                recommended_pyr_levels,
            )

    def _show_altcuda_message(self):
        if self.corr_mode == "local" and alt_cuda_corr is None:
            logger.warning(
                f"DPFlow: You are running with --corr_mode local, but alt_cuda_corr is not installed. Please install alt_cuda_corr to increase the speed."
            )

    def forward(self, inputs):
        try:
            return self.forward_flow(inputs)
        except torch.OutOfMemoryError:
            if self.corr_mode == "allpairs":
                logger.warning(
                    "DPFlow: CUDA out of memory error with input size {}. DPFlow will set --model.corr_mode to 'local' and re-attempt inference. This decreases memory consumption, but it is also slower.",
                    list(inputs["images"].shape[-2:]),
                )
                self.corr_mode = "local"
                try:
                    return self.forward_flow(inputs)
                except torch.OutOfMemoryError:
                    logger.error(
                        "DPFlow: CUDA out of memory error even after setting --model.corr_mode to 'local'. DPFlow cannot process this input size: {} on this device.",
                        list(inputs["images"].shape[-2:]),
                    )
            else:
                logger.error(
                    "DPFlow: CUDA out of memory error even with --model.corr_mode set to 'local'. DPFlow cannot process this input size: {} on this device.",
                    list(inputs["images"].shape[-2:]),
                )

    def forward_flow(self, inputs):
        if self.corr_mode == "local" and not self.has_shown_altcuda_message:
            self._show_altcuda_message()
            self.has_shown_altcuda_message = True

        if self.pyramid_levels is not None and not self.has_shown_input_message:
            self._show_input_message(inputs["images"])
            self.has_shown_input_message = True

        if self.pyramid_levels is None:
            pyr_levels = compute_pyramid_levels(inputs["images"])
            output_stride = 2 ** (pyr_levels + 2 + self.extra_output_stride)

            if output_stride != self.current_output_stride:
                logger.info(
                    "DPFlow: Detected change in input size. The number of pyramid levels will change to {}, corresponding to output stride {}.",
                    pyr_levels,
                    output_stride,
                )
                self.current_output_stride = output_stride
        else:
            pyr_levels = self.pyramid_levels
            output_stride = self.output_stride

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            stride=output_stride,
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )
        image1 = images[:, 0]
        image2 = images[:, 1]

        flow_init = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            flow_init = inputs["prev_preds"]["flow_small"]

        flow_predictions, flow_small, flow_up, info_predictions = self.predict(
            image1,
            image2,
            pyr_levels=pyr_levels,
            image_resizer=image_resizer,
            flow_init=flow_init,
        )

        nf_predictions = []
        if self.training and self.loss == "laplace":
            # exlude invalid pixels and extremely large diplacements
            for i in range(len(info_predictions)):
                if not self.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min

                if info_predictions[i] is None:
                    nf_predictions.append(None)
                else:
                    raw_b = info_predictions[i][:, 2:]
                    log_b = torch.zeros_like(raw_b)
                    weight = info_predictions[i][:, :2]
                    # Large b Component
                    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                    # Small b Component
                    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                    # term2: [N, 2, m, H, W]
                    term2 = (
                        (inputs["flows"][:, 0] - flow_predictions[i]).abs().unsqueeze(2)
                    ) * (torch.exp(-log_b).unsqueeze(1))
                    # term1: [N, m, H, W]
                    term1 = weight - math.log(2) - log_b
                    nf_loss = torch.logsumexp(
                        weight, dim=1, keepdim=True
                    ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                    nf_predictions.append(nf_loss)

        outputs = {"flows": flow_up[:, None], "flow_small": flow_small}

        if self.training:
            outputs["flow_preds"] = flow_predictions
            outputs["nf_preds"] = nf_predictions

        return outputs

    def predict(self, x1_raw, x2_raw, pyr_levels, image_resizer, flow_init=None):
        b, _, height_im, width_im = x1_raw.size()

        x1_pyramid, x2_pyramid = self.fnet(x1_raw, x2_raw, pyr_levels=pyr_levels)

        # outputs
        flows = []
        infos = []

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = x1_pyramid[0].size()
        init_device = x1_pyramid[0].device

        if flow_init is not None:
            flow = flow_init
            flow = rescale_flow(
                flow,
                x1_pyramid[0].shape[-1],
                x1_pyramid[0].shape[-2],
                to_local=False,
            )
            flow = upsample2d_as(flow, x1_pyramid[0], mode="bilinear")
            flow = forward_interpolate_batch(flow)
        else:
            flow = torch.zeros(
                b_size, 2, h_x1, w_x1, dtype=self.dtype, device=init_device
            )

        net = None
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
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

            if self.corr_mode == "allpairs":
                corr_fn = CorrBlock(x1, x2, self.corr_levels, self.corr_range)
            else:
                corr_fn = AlternateCorrBlock(x1, x2, self.corr_levels, self.corr_range)

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = upsample2d_as(flow, x1, mode="bilinear")

            net = torch.tanh(net_tmp)

            for it in range(self.iters_per_level):
                if self.detach_flow:
                    flow = flow.detach()

                # correlation
                out_corr = corr_fn(coords0 + flow)

                flow_res, net, mask = self.update_block(net, inp, out_corr, flow)

                info = None
                if self.loss == "laplace":
                    info = flow_res[:, 2:]
                    flow_res = flow_res[:, :2]

                flow = flow + flow_res

                if self.training or (
                    l == len(x1_pyramid) - 1 and it == self.iters_per_level - 1
                ):
                    out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                    if mask is not None:
                        out_flow = self.upsample_flow(out_flow, mask, factor=8)
                    out_flow = upsample2d_as(out_flow, x1_raw, mode="bilinear")
                    out_flow = self.postprocess_predictions(
                        out_flow, image_resizer, is_flow=True
                    )
                    flows.append(out_flow)

                    out_info = None
                    if info is not None:
                        if mask is not None:
                            out_info = self.upsample_flow(info, mask, factor=8, ch=4)
                        out_info = upsample2d_as(out_info, x1_raw, mode="bilinear")
                        out_info = self.postprocess_predictions(
                            out_info, image_resizer, is_flow=False
                        )
                    infos.append(out_info)

        return flows, flow, out_flow, infos


@register_model
@trainable
@ptlflow_trained
class dpflow(DPFlow):
    pass
