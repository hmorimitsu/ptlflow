from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .update import BasicUpdateBlock, GMAUpdateBlock
from .extractor import BasicEncoder
from .matching_encoder import MatchingModel
from .corr import get_corr_block
from .utils import coords_grid, upflow8, compute_grid_indices, compute_weight
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


class MatchFlow(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-chairs-02519b53.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-kitti-bc72ce81.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-sintel-683422f4.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-things-49295bd8.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=32)
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if "dropout" not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = MatchingModel(cfg=args)

        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
        )
        if self.args.raft is False:
            self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
            self.att = Attention(
                args=self.args,
                dim=cdim,
                heads=self.args.num_heads,
                max_pos_size=160,
                dim_head=cdim,
            )
        else:
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        if self.args.alternate_corr and alt_cuda_corr is None:
            print(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

        self.showed_warning = False

    @property
    def train_size(self):
        return self._train_size

    @train_size.setter
    def train_size(self, value):
        if value is not None:
            assert isinstance(value, (tuple, list))
            assert len(value) == 2
            assert isinstance(value[0], int) and isinstance(value[1], int)
        self._train_size = value
        self.fnet = MatchingModel(cfg=self.args, train_size=self.train_size)

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
        parser.add_argument("--matching_model_path", type=str, default="")
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--raft", action="store_true")
        parser.add_argument(
            "--not_use_tile_input", action="store_false", dest="use_tile_input"
        )
        parser.add_argument("--tile_height", type=int, default=416)
        parser.add_argument("--tile_sigma", type=float, default=0.05)
        parser.add_argument("--position_only", action="store_true")
        parser.add_argument("--position_and_content", action="store_true")
        parser.add_argument("--alternate_corr", action="store_true")
        parser.add_argument(
            "--train_size",
            type=int,
            nargs=2,
            default=None,
            help="train_size will be normally loaded from the checkpoint. However, if you provide this value, it will override the value from the checkpoint.",
        )
        return parser

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
        if self.args.train_size is not None:
            train_size = self.args.train_size
        else:
            train_size = self.train_size

        if self.args.use_tile_input and train_size is None and not self.showed_warning:
            print(
                "WARNING: --train_size is not provided and it cannot be loaded from the checkpoint either. Matchflow will run without input tile."
            )
            self.showed_warning = True

        if self.args.use_tile_input and train_size is not None:
            return self.forward_tile(inputs, train_size)
        else:
            return self.forward_resize(inputs)

    def forward_resize(self, inputs):
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="interpolation",
            interpolation_mode="bilinear",
            interpolation_align_corners=True,
        )

        flow_prev = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            flow_prev = inputs["prev_preds"]["flow_small"]

        flow_predictions, flow_small = self.predict(
            images[:, 0], images[:, 1], flow_prev
        )
        output_flow = flow_predictions[-1]

        if self.training:
            for i, p in enumerate(flow_predictions):
                flow_predictions[i] = self.postprocess_predictions(
                    p, image_resizer, is_flow=True
                )
            outputs = {
                "flows": flow_predictions[-1][:, None],
                "flow_preds": flow_predictions,
            }
        else:
            output_flow = self.postprocess_predictions(
                output_flow, image_resizer, is_flow=True
            )
            outputs = {"flows": output_flow[:, None], "flow_small": flow_small}

        return outputs

    def forward_tile(self, inputs, train_size):
        input_size = inputs["images"].shape[-2:]
        image_size = (max(self.args.tile_height, input_size[-2]), input_size[-1])
        hws = compute_grid_indices(image_size, train_size)
        device = inputs["images"].device
        weights = compute_weight(
            hws, image_size, train_size, self.args.tile_sigma, device=device
        )

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="interpolation",
            target_size=image_size,
            interpolation_mode="bilinear",
            interpolation_align_corners=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + train_size[0], w : w + train_size[1]]
            image2_tile = image2[:, :, h : h + train_size[0], w : w + train_size[1]]

            flow_predictions, _ = self.predict(image1_tile, image2_tile)
            flow_pre = flow_predictions[-1]

            padding = (
                w,
                image_size[1] - w - train_size[1],
                h,
                image_size[0] - h - train_size[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        output_flow = flows / flow_count

        output_flow = self.postprocess_predictions(
            output_flow, image_resizer, is_flow=True
        )
        return {"flows": output_flow[:, None]}

    def predict(self, image1, image2, flow_prev=None):
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
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
        if self.args.raft is False:
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_prev is not None:
            forward_flow = forward_interpolate_batch(flow_prev)
            coords1 = coords1 + forward_flow

        flow_predictions = []
        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            if self.args.raft is False:
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow, attention
                )
            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        return flow_predictions, coords1 - coords0


class MatchFlowRAFT(MatchFlow):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_raft-things-bf560032.ckpt"
    }

    def __init__(self, args: Namespace) -> None:
        args.raft = True
        super().__init__(args=args)
