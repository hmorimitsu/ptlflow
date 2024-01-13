from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .extractor import BasicEncoder
from .attention import Attention1D
from .position import PositionEmbeddingSine
from .correlation import Correlation1D
from .update import BasicUpdateBlock
from .utils import coords_grid
from ..base_model.base_model import BaseModel


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

            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class Flow1D(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flow1d-chairs-75cd85a1.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flow1d-things-bcd92815.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flow1d-sintel-28a093d3.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flow1d-kitti-803a0181.ckpt",
        "highres": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flow1d-highres-7ab476dc.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=8)
        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=self.args.feature_channels,
            norm_fn="instance",
        )

        self.cnet = BasicEncoder(
            output_dim=self.args.hidden_dim + self.args.context_dim,
            norm_fn="batch",
        )

        # 1D attention
        corr_channels = (2 * self.args.corr_radius + 1) * 2

        self.attn_x = Attention1D(
            self.args.feature_channels,
            y_attention=False,
            double_cross_attn=True,
        )
        self.attn_y = Attention1D(
            self.args.feature_channels,
            y_attention=True,
            double_cross_attn=True,
        )

        # Update block
        self.update_block = BasicUpdateBlock(
            corr_channels=corr_channels,
            hidden_dim=self.args.hidden_dim,
            context_dim=self.args.context_dim,
            downsample_factor=self.args.downsample_factor,
        )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--downsample_factor", type=int, default=8)
        parser.add_argument("--feature_channels", type=int, default=256)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--context_dim", type=int, default=128)
        parser.add_argument("--corr_radius", type=int, default=32)
        parser.add_argument("--iters", type=int, default=32)
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=400)
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, downsample=None):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        n, c, h, w = img.shape
        downsample_factor = (
            self.args.downsample_factor if downsample is None else downsample
        )
        coords0 = coords_grid(
            n, h // downsample_factor, w // downsample_factor, dtype=img.dtype
        ).to(img.device)
        coords1 = coords_grid(
            n, h // downsample_factor, w // downsample_factor, dtype=img.dtype
        ).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def learned_upflow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        n, _, h, w = flow.shape
        mask = mask.view(
            n, 1, 9, self.args.downsample_factor, self.args.downsample_factor, h, w
        )
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.args.downsample_factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(
            n, 2, self.args.downsample_factor * h, self.args.downsample_factor * w
        )

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

        # run the feature network
        feature1, feature2 = self.fnet([image1, image2])

        hdim = self.args.hidden_dim
        cdim = self.args.context_dim

        # position encoding
        pos_channels = self.args.feature_channels // 2
        pos_enc = PositionEmbeddingSine(pos_channels)

        position = pos_enc(feature1)  # [B, C, H, W]

        # 1D correlation
        feature2_x, attn_x = self.attn_x(feature1, feature2, position)
        corr_fn_y = Correlation1D(
            feature1,
            feature2_x,
            radius=self.args.corr_radius,
            x_correlation=False,
        )

        feature2_y, attn_y = self.attn_y(feature1, feature2, position)
        corr_fn_x = Correlation1D(
            feature1,
            feature2_y,
            radius=self.args.corr_radius,
            x_correlation=True,
        )

        # run the context network
        cnet = self.cnet(image1)  # list of feature pyramid, low scale to high scale

        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)  # 1/8 resolution or 1/4

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()  # stop gradient

            corr_x = corr_fn_x(coords1)
            corr_y = corr_fn_y(coords1)
            corr = torch.cat((corr_x, corr_y), dim=1)  # [B, 2(2R+1), H, W]

            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(
                net,
                inp,
                corr,
                flow,
                upsample=(self.training or itr == self.args.iters - 1),
            )

            coords1 = coords1 + delta_flow

            if self.training:
                # upsample predictions
                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_predictions.append(flow_up)
            elif itr == self.args.iters - 1:
                flow_up = self.learned_upflow(coords1 - coords0, up_mask)
                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs
