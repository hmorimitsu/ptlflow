from argparse import Namespace

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import init

from .flownetc import FlowNetC
from .flownets import FlowNetS
from .submodules import *
from .flownet_base import FlowNetBase


class FlowNetCSS(FlowNetBase):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flownetcss-things-b42e67d0.ckpt"
    }

    def __init__(self, args: Namespace):
        args.input_channels = 12
        super(FlowNetCSS, self).__init__(args)

        self.args = args
        self.rgb_max = 1

        # First Block (FlowNetC)
        self.flownetc = FlowNetC(args)

        # Block (FlowNetS)
        self.flownets_1 = FlowNetS(args)
        self.flownets_2 = FlowNetS(args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)

        if x.is_cuda:
            grid = grid.to(dtype=x.dtype, device=x.device)
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).to(dtype=x.dtype, device=x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, inputs):
        orig_images = inputs["images"]
        bgr_mean = rearrange(orig_images, "b n c h w -> b n c (h w)").mean(-1)
        bgr_mean = bgr_mean[..., None, None]
        images, image_resizer = self.preprocess_images(
            orig_images,
            bgr_add=-bgr_mean,
            bgr_mult=1.0,
            bgr_to_rgb=True,
            resize_mode="interpolation",
            interpolation_mode="bilinear",
            interpolation_align_corners=True,
        )
        inputs["images"] = images

        # flownetc
        flownetc_flow = self.flownetc(inputs, skip_preprocess=True)["flows"][:, 0]

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.warp(inputs["images"][:, 1], flownetc_flow)
        diff_img0 = inputs["images"][:, 0] - resampled_img1
        norm_diff_img0 = torch.norm(diff_img0, p=2, dim=1, keepdim=True)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (
                inputs["images"][:, 0],
                inputs["images"][:, 1],
                resampled_img1,
                flownetc_flow / self.args.div_flow,
                norm_diff_img0,
            ),
            dim=1,
        )

        # flownets1
        flownets1_flow = self.flownets_1(
            {"images": concat1[:, None]}, skip_preprocess=True
        )["flows"][:, 0]

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.warp(inputs["images"][:, 1], flownets1_flow)
        diff_img0 = inputs["images"][:, 0] - resampled_img1
        norm_diff_img0 = torch.norm(diff_img0, p=2, dim=1, keepdim=True)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (
                inputs["images"][:, 0],
                inputs["images"][:, 1],
                resampled_img1,
                flownets1_flow / self.args.div_flow,
                norm_diff_img0,
            ),
            dim=1,
        )

        # flownets2
        flownets2_preds = self.flownets_2(
            {"images": concat1[:, None]}, skip_preprocess=True
        )
        flownets2_preds["flows"] = self.postprocess_predictions(
            flownets2_preds["flows"], image_resizer, is_flow=True
        )

        return flownets2_preds
