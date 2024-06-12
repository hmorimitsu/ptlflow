from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

from .attention import Attention
from .extractor import BasicEncoder
from .corr import CorrBlock

from .update import Update
from ..base_model.base_model import BaseModel

try:
    from .softsplat import FunctionSoftsplat as forward_warping
except ModuleNotFoundError:
    forward_warping = None


class SplatFlow(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/splatflow-kitti-2aa8e145.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=None, output_stride=8)

        self.hdim = self.cdim = 128
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance")
        self.cnet = BasicEncoder(output_dim=self.hdim + self.cdim, norm_fn="batch")
        self.att = Attention(dim=self.cdim, heads=1, dim_head=self.cdim)
        self.update = Update(hidden_dim=self.hdim)

        self.has_shown_warning = False

        if forward_warping is None:
            raise ModuleNotFoundError("No module named 'cupy'")

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
        parser.add_argument("--alternate_corr", action="store_true")
        parser.add_argument(
            "--not_fast_inference", action="store_false", dest="fast_inference"
        )
        return parser

    def init_coord(self, fmap):
        f_shape = fmap.shape
        H, W = f_shape[-2:]
        y0, x0 = torch.meshgrid(
            torch.arange(H).to(device=fmap.device, dtype=fmap.dtype),
            torch.arange(W).to(device=fmap.device, dtype=fmap.dtype),
            indexing="ij",
        )
        coord = torch.stack([x0, y0], dim=0)  # shape: (2, H, W)
        coord = coord.unsqueeze(0).repeat(f_shape[0], 1, 1, 1)
        return coord

    def initialize_flow(self, fmap):
        coords0 = self.init_coord(fmap)
        coords1 = self.init_coord(fmap)

        return coords0, coords1

    def cvx_upsample(self, data, mask):
        N, C, H, W = data.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(data, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

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

        if images.shape[1] == 2 and not self.has_shown_warning:
            print(
                "WARNING: splatflow is running in two-frame mode. Provide 3 input images to run in three-frame mode."
            )
            self.has_shown_warning = True

        flow_prs_01, mf_01, low_01 = self.forward_one_pair(images[:, 0], images[:, 1])
        if images.shape[1] > 2:
            mf_t = forward_warping(mf_01, low_01)
            flow_prs_12, mf_12, low_12 = self.forward_one_pair(
                images[:, 1], images[:, 2], mf_t=mf_t
            )
            out_flow = flow_prs_12[-1]
            flow_small = low_12
        else:
            out_flow = flow_prs_01[-1]
            flow_small = low_01

        out_flow = self.postprocess_predictions(out_flow, image_resizer, is_flow=True)
        outputs = {"flows": out_flow[:, None], "flow_small": flow_small}
        return outputs

    def forward_one_pair(self, image1, image2, mf_t=None):
        fmap1, fmap2 = self.fnet([image1, image2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=4)

        coords0, coords1 = self.initialize_flow(fmap1)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        atte_s = self.att(inp)

        flow_predictions = []

        for itr in range(self.args.iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)

            flow = coords1 - coords0
            net, up_mask, delta_flow, mf = self.update(
                net, inp, corr, flow, atte_s, mf_t
            )
            coords1 = coords1 + delta_flow

            if (self.args.fast_inference and (itr == self.args.iters - 1)) or (
                not self.args.fast_inference
            ):
                flow_up = self.cvx_upsample(8 * (coords1 - coords0), up_mask)
                flow_predictions.append(flow_up)

            low = coords1 - coords0

        return flow_predictions, mf, low
