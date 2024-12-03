from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from ptlflow.utils.utils import forward_interpolate_batch
from .update import GMAUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock, TransCorrBlock
from .utils import coords_grid, upflow8, print0
from .gma import Attention
from .setrans import SETransConfig, SelfAttVisPosTrans
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

        # exclude invalid pixels and extremely large displacements.
        # MAX_FLOW = 400.
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            # Exponentially increasing weights. (Eq.7 in RAFT paper)
            # As i increases, flow_preds[i] is expected to be more and more accurate,
            # so we are less and less tolerant to errors through gradually increased i_weight.
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class CRAFT(BaseModel):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-things-5a41930c.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-sintel-ff8e6563.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/craft-kitti-4d99b0c1.ckpt",
    }

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout: float = 0.0,
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
        f1trans: str = "none",
        f2trans: str = "full",
        f2_attn_mask_radius: int = -1,
        f2_num_modes: int = 4,
        f2_pos_code_weight: float = 0.5,
        inter_num_modes: int = 4,
        inter_pos_code_type: str = "bias",
        inter_pos_code_weight: float = 0.5,
        intra_pos_code_type: str = "bias",
        intra_pos_code_weight: float = 1.0,
        intra_num_modes: int = 4,
        craft: bool = True,
        inter_qk_have_bias: bool = True,
        num_heads: int = 1,
        pos_bias_radius: int = 7,
        use_setrans: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=8, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.dropout = dropout
        self.gamma = gamma
        self.max_flow = max_flow
        self.iters = iters
        self.f1trans = f1trans
        self.f2trans = f2trans
        self.f2_attn_mask_radius = f2_attn_mask_radius
        self.f2_num_modes = f2_num_modes
        self.f2_pos_code_weight = f2_pos_code_weight
        self.inter_num_modes = inter_num_modes
        self.inter_pos_code_type = inter_pos_code_type
        self.inter_pos_code_weight = inter_pos_code_weight
        self.intra_pos_code_type = intra_pos_code_type
        self.intra_pos_code_weight = intra_pos_code_weight
        self.intra_num_modes = intra_num_modes
        self.craft = craft
        self.inter_qk_have_bias = inter_qk_have_bias
        self.num_heads = num_heads
        self.pos_bias_radius = pos_bias_radius
        self.use_setrans = use_setrans

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        # default CRAFT corr_radius: 4
        if corr_radius == -1:
            corr_radius = 4
        print0("Lookup radius: %d" % corr_radius)

        if craft:
            self.inter_trans_config = SETransConfig()
            self.inter_trans_config.update_config(self.__dict__["_hparams"])
            self.inter_trans_config.in_feat_dim = 256
            self.inter_trans_config.feat_dim = 256
            self.inter_trans_config.max_pos_size = 160
            # out_attn_scores_only implies no FFN nor V projection.
            self.inter_trans_config.out_attn_scores_only = (
                True  # implies no FFN and no skip.
            )
            self.inter_trans_config.attn_diag_cycles = 1000
            self.inter_trans_config.num_modes = inter_num_modes  # default: 4
            self.inter_trans_config.tie_qk_scheme = "shared"  # Symmetric Q/K
            self.inter_trans_config.qk_have_bias = inter_qk_have_bias  # default: True
            self.inter_trans_config.pos_code_type = inter_pos_code_type  # default: bias
            self.inter_trans_config.pos_code_weight = (
                inter_pos_code_weight  # default: 0.5
            )
            print0(
                "Inter-frame trans config:\n{}".format(self.inter_trans_config.__dict__)
            )

            self.corr_fn = TransCorrBlock(
                self.inter_trans_config,
                radius=corr_radius,
                do_corr_global_norm=True,
            )

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=dropout)
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=dropout
        )

        if f2trans != "none":
            # f2_trans has the same configuration as GMA att,
            # except that the feature dimension is doubled, and not out_attn_probs_only.
            self.f2_trans_config = SETransConfig()
            self.f2_trans_config.update_config(self.__dict__["_hparams"])
            self.f2_trans_config.in_feat_dim = 256
            self.f2_trans_config.feat_dim = 256
            # f2trans(x) = attn_aggregate(v(x)) + x. Here attn_aggregate and v (first_linear) both have 4 modes.
            self.f2_trans_config.has_input_skip = True
            # No FFN. f2trans simply aggregates similar features.
            # But there's still a V projection.
            self.f2_trans_config.has_FFN = False
            # When doing feature aggregation, set attn_mask_radius > 0 to exclude points that are too far apart, to reduce noises.
            # E.g., 64 corresponds to 64*8=512 pixels in the image space.
            self.f2_trans_config.attn_mask_radius = f2_attn_mask_radius
            # Not tying QK performs slightly better.
            self.f2_trans_config.tie_qk_scheme = None
            self.f2_trans_config.qk_have_bias = False
            self.f2_trans_config.out_attn_probs_only = False
            self.f2_trans_config.attn_diag_cycles = 1000
            self.f2_trans_config.num_modes = f2_num_modes  # default: 4
            self.f2_trans_config.pos_code_type = intra_pos_code_type  # default: bias
            self.f2_trans_config.pos_code_weight = f2_pos_code_weight  # default: 0.5
            self.f2_trans = SelfAttVisPosTrans(self.f2_trans_config, "F2 transformer")
            print0("F2-trans config:\n{}".format(self.f2_trans_config.__dict__))
            self.f2_trans_config = self.f2_trans_config

            if f1trans != "none":
                corr_multiplier = 2
                if f1trans == "shared":
                    # f1_trans and f2_trans are shared.
                    self.f1_trans = self.f2_trans
                elif f1trans == "private":
                    # f1_trans is a private instance of SelfAttVisPosTrans.
                    self.f1_trans = SelfAttVisPosTrans(
                        self.f2_trans_config, "F1 transformer"
                    )
                else:
                    breakpoint()
            else:
                self.f1_trans = None
                corr_multiplier = 1

        if use_setrans:
            self.intra_trans_config = SETransConfig()
            self.intra_trans_config.update_config(self.__dict__["_hparams"])
            self.intra_trans_config.in_feat_dim = 128
            self.intra_trans_config.feat_dim = 128
            # has_FFN & has_input_skip are for GMAUpdateBlock.aggregator.
            # Having FFN reduces performance. FYI, GMA also has no FFN.
            self.intra_trans_config.has_FFN = False
            self.intra_trans_config.has_input_skip = True
            self.intra_trans_config.attn_mask_radius = -1
            # Not tying QK performs slightly better.
            self.intra_trans_config.tie_qk_scheme = None
            self.intra_trans_config.qk_have_bias = False
            self.intra_trans_config.out_attn_probs_only = True
            self.intra_trans_config.attn_diag_cycles = 1000
            self.intra_trans_config.num_modes = intra_num_modes  # default: 4
            self.intra_trans_config.pos_code_type = intra_pos_code_type  # default: bias
            self.intra_trans_config.pos_code_weight = (
                intra_pos_code_weight  # default: 1
            )
            self.att = SelfAttVisPosTrans(
                self.intra_trans_config, "Intra-frame attention"
            )
            print0(
                "Intra-frame trans config:\n{}".format(self.intra_trans_config.__dict__)
            )
        else:
            self.att = Attention(
                dim=cdim,
                heads=num_heads,
                max_pos_size=160,
                dim_head=cdim,
            )

        # if args.use_setrans, initialization of GMAUpdateBlock.aggregator needs to access self.intra_trans_config.
        # So GMAUpdateBlock() construction has to be done after initializing intra_trans_config.
        self.update_block = GMAUpdateBlock(
            corr_levels=corr_levels,
            corr_multiplier=corr_multiplier,
            corr_radius=corr_radius,
            use_setrans=use_setrans,
            intra_trans_config=self.intra_trans_config,
            num_heads=num_heads,
            hidden_dim=hdim,
        )
        self.call_counter = 0

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        # correlation matrix: 7040*7040 (55*128=7040).
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1o, fmap2o = None, None
        if self.f1trans != "none":
            fmap1o = fmap1
            fmap1 = self.f1_trans(fmap1)
        if self.f2trans != "none":
            fmap2o = fmap2
            fmap2 = self.f2_trans(fmap2)

        # If not craft, the correlation volume is computed in the ctor.
        # If craft, the correlation volume is computed in corr_fn.update().
        if not self.craft:
            self.corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

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

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        # If craft, the correlation volume is computed in corr_fn.update().
        if self.craft:
            # only update() once, instead of dynamically updating coords1.
            self.corr_fn.update(fmap1, fmap2, fmap1o, fmap2o, coords1, coords2=None)

        flow_predictions = []
        for itr in range(self.iters):
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
            net_feat, up_mask, delta_flow = self.update_block(
                net_feat, inp_feat, corr, flow, attention
            )

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

            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs


@register_model
@trainable
class craft(CRAFT):
    pass
