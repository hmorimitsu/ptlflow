from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .update import BasicUpdateBlock
from .extractor import BasicEncoder, Basic_Context_Encoder
from .corr import get_corr_block
from .utils import coords_grid, upflow2, get_correlation_depth
from ..base_model.base_model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


def downflow(flow, mode="bilinear", factor=0.125):
    old_size = (flow.shape[2], flow.shape[3])
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    u_scale = new_size[1] / old_size[1]
    v_scale = new_size[0] / old_size[0]
    resized_flow = F.interpolate(
        flow, size=new_size, mode=mode, align_corners=True
    )  # b 2 h w
    resized_flow_split = torch.split(resized_flow, 1, dim=1)
    rescaled_flow = torch.cat(
        [u_scale * resized_flow_split[0], v_scale * resized_flow_split[1]], dim=1
    )

    return rescaled_flow


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


class MSRAFTPlus(BaseModel):
    pretrained_checkpoints = {
        "mixed": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ms_raft_plus-mixed-2bb01f62.ckpt"
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=16)

        self.correlation_depth = get_correlation_depth(
            self.args.lookup_pyramid_levels, self.args.lookup_radius
        )

        self.hidden_dim = 128
        self.context_dim = 128

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="group")
        self.cnet = Basic_Context_Encoder(output_dim=256, norm_fn="group")
        self.update_block = BasicUpdateBlock(
            self.args, self.correlation_depth, hidden_dim=128, scale=2
        )

        if self.args.alternate_corr and alt_cuda_corr is None:
            print(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--iters", type=int, nargs="+", default=[4, 6, 5, 10])
        parser.add_argument("--lookup_pyramid_levels", type=int, default=2)
        parser.add_argument("--lookup_radius", default=4)
        parser.add_argument(
            "--no_alternate_corr", action="store_false", dest="alternate_corr"
        )
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow16(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 16, W // 16, dtype=img.dtype, device=img.device)
        coords1 = coords_grid(N, H // 16, W // 16, dtype=img.dtype, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def get_grid(self, img, scale):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(
            N, H // scale, W // scale, dtype=img.dtype, device=img.device
        )
        return coords0

    def upsample_flow(self, flow, mask, scale=8):
        """Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale * H, scale * W)

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
        fnet_pyramid = self.fnet([image1, image2])
        # run the context network
        cnet_pyramid = self.cnet(image1)

        coords0, coords1 = self.initialize_flow16(image1)
        flow_predictions = []

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        assert len(fnet_pyramid) == len(
            cnet_pyramid
        ), "fnet and cnet pyramid should have the same length."
        assert len(fnet_pyramid) == len(
            self.args.iters
        ), "pyramid levels and the length of GRU iteration lists should be the same."

        for index, (fmap1, fmap2) in enumerate(fnet_pyramid):
            corr_fn = get_corr_block(
                fmap1=fmap1,
                fmap2=fmap2,
                radius=self.args.lookup_radius,
                num_levels=self.args.lookup_pyramid_levels,
                alternate_corr=self.args.alternate_corr,
            )

            net, inp = torch.split(cnet_pyramid[index], [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            for itr in range(self.args.iters[index]):
                coords1 = coords1.detach()
                if index >= 1 and itr == 0:
                    coords1 = self.upsample_flow(coords1, up_mask, scale=2)
                    coords0 = self.get_grid(image1, scale=16 / (2**index))

                corr = corr_fn(coords1)
                flow = coords1 - coords0
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow
                # upsample predictions
                flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                for i in range(len(fnet_pyramid) - index - 1):
                    flow_up = upflow2(flow_up)

                flow_up = self.postprocess_predictions(
                    flow_up, image_resizer, is_flow=True
                )
                flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {
                "flows": flow_up[:, None],
                "flow_small": downflow(flow_up, factor=0.0625),
            }

        return outputs
