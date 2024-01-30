from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .update import *
from .extractor import BasicEncoder
from .corr import get_corr_block
from .utils import upflow8
from .gma import Attention
from ..base_model.base_model import BaseModel

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class SKFlow(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/skflow-kitti-4e1f8b63.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/skflow-sintel-98fb67cf.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/skflow-things-f84e6538.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if "dropout" not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=args.dropout
        )
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
        )
        self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
            self.args, hidden_dim=hdim
        )
        self.att = Attention(
            args=self.args,
            dim=cdim,
            heads=self.args.num_heads,
            max_pos_size=160,
            dim_head=cdim,
        )

        if self.args.alternate_corr and alt_cuda_corr is None:
            print(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--iters", type=int, default=32)
        parser.add_argument("--k_conv", type=int, nargs="+", default=(1, 15))
        parser.add_argument("--PCUpdater_conv", type=int, nargs="+", default=(1, 7))
        parser.add_argument(
            "--num_heads",
            default=1,
            type=int,
            help="number of heads in attention and aggregation",
        )
        parser.add_argument(
            "--position_only",
            default=False,
            action="store_true",
            help="only use position-wise attention",
        )
        parser.add_argument(
            "--position_and_content",
            default=False,
            action="store_true",
            help="use position and content-wise attention",
        )
        parser.add_argument("--alternate_corr", action="store_true")
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=self.dtype, device=self.device),
            torch.arange(wd, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = self.coords_grid(N, H // 8, W // 8)
        coords1 = self.coords_grid(N, H // 8, W // 8)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upflow8(self, flow, mode="bilinear"):
        new_size = (8 * flow.shape[2], 8 * flow.shape[3])
        return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

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
        fmap1, fmap2 = self.fnet([image1, image2])

        corr_fn = get_corr_block(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels,
            alternate_corr=self.args.alternate_corr,
        )

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        # attention, att_c, att_p = self.att(inp)
        attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(
                net, inp, corr, flow, attention
            )

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs
