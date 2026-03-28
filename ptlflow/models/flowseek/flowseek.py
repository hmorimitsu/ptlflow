from typing import Literal, Sequence

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model
from .depth_anything_v2.dpt import DepthAnythingV2
from .update import BasicUpdateBlock
from .corr import CorrBlock
from .utils import coords_grid, InputPadder
from .extractor import ResNetFPN
from .layer import conv3x3
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
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss_i = outputs["nf_preds"][i]
            final_mask = (
                (~torch.isnan(loss_i.detach()))
                & (~torch.isinf(loss_i.detach()))
                & valid
            )
            flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

        return flow_loss


class FlowSeek(BaseModel):
    def __init__(
        self,
        corr_levels: int = 4,
        radius: int = 4,
        pretrain: Literal["resnet18", "resnet34"] = "resnet18",
        da_size: Literal["vits", "vitb"] = "vits",
        dim: int = 128,
        initial_dim: int = 64,
        num_blocks: int = 2,
        block_dims: Sequence[int] = (64, 128, 256),
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 4,
        use_var: bool = True,
        var_min: float = 0,
        var_max: float = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=8, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.dim = dim
        self.iters = iters
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max
        self.output_dim = dim * 2

        self.da_size = da_size

        self.corr_levels = corr_levels
        self.corr_radius = radius
        self.corr_channel = corr_levels * (radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(
            initial_dim=initial_dim,
            block_dims=block_dims,
            pretrain=pretrain,
            input_dim=6,
            output_dim=2 * dim,
            norm_layer=nn.BatchNorm2d,
            init_weight=False,
        )

        self.da_model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
        }

        self.dav2 = DepthAnythingV2(**self.da_model_configs[da_size])
        # self.dav2.load_state_dict(
        #     torch.load(
        #         f"weights/depth_anything_v2_%s.pth" % da_size, map_location="cpu"
        #     )
        # )
        self.dav2 = self.dav2.eval()
        for param in self.dav2.parameters():
            param.requires_grad = False

        self.merge_head = nn.Sequential(
            nn.Conv2d(
                self.da_model_configs[da_size]["features"],
                self.da_model_configs[da_size]["features"] // 2 * 3,
                3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.da_model_configs[da_size]["features"] // 2 * 3,
                self.da_model_configs[da_size]["features"] * 2,
                3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.da_model_configs[da_size]["features"] * 2,
                self.da_model_configs[da_size]["features"] * 2,
                3,
                stride=2,
                padding=1,
            ),
        )

        self.bnet = ResNetFPN(
            initial_dim=initial_dim,
            block_dims=block_dims,
            pretrain=pretrain,
            input_dim=16,
            output_dim=2 * dim,
            norm_layer=nn.BatchNorm2d,
            init_weight=False,
        )

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * dim, 2 * dim)

        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0),
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim * 2, 2 * dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, 6, 3, padding=1),
        )
        if iters > 0:
            self.fnet = ResNetFPN(
                initial_dim=initial_dim,
                block_dims=block_dims,
                pretrain=pretrain,
                input_dim=3,
                output_dim=self.output_dim,
                norm_layer=nn.BatchNorm2d,
                init_weight=False,
            )
            self.update_block = BasicUpdateBlock(
                hdim=dim * 2,
                cdim=dim * 2,
                num_blocks=num_blocks,
                corr_channel=self.corr_channel,
            )

    def create_bases(self, disp):
        B, C, H, W = disp.shape
        assert C == 1
        cx = 0.5
        cy = 0.5
        dtype = disp.dtype

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, dtype=dtype)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, dtype=dtype)
        u, v = torch.meshgrid(xs, ys, indexing="xy")
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1)
        v = v.repeat(B, 1, 1, 1)

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1, 2, 3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1, 2, 3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1, 2, 3), keepdim=True)

        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz = torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1, 2, 3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1, 2, 3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1, 2, 3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1, 2, 3), keepdim=True)
        Rz = Rz / torch.linalg.vector_norm(Rz, dim=(1, 2, 3), keepdim=True)

        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1)  # Bx(8x2)xHxW
        return M

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""

        images_res, _ = self.preprocess_images(
            inputs["images"],
            bgr_add=[-0.406, -0.456, -0.485],
            bgr_mult=[1 / 0.225, 1 / 0.224, 1 / 0.229],
            bgr_to_rgb=True,
            target_size=[518, 518],
            resize_mode="interpolation",
            interpolation_align_corners=False,
        )

        image1_res = images_res[:, 0]
        image2_res = images_res[:, 1]

        im1_path1, depth1 = self.dav2.forward(image1_res)
        im2_path1, _ = self.dav2.forward(image2_res)

        N, _, _, H, W = inputs["images"].shape

        if "flows" in inputs:
            flow_gt = inputs["flows"][:, 0]
        else:
            flow_gt = torch.zeros(N, 2, H, W, device=image1_res.device)

        im1_path1 = F.interpolate(
            im1_path1, (H, W), mode="bilinear", align_corners=False
        )
        im2_path1 = F.interpolate(
            im2_path1, (H, W), mode="bilinear", align_corners=False
        )
        bases1 = self.create_bases(
            F.interpolate(depth1, (H, W), mode="bilinear", align_corners=False)
        )

        mono1 = self.merge_head(im1_path1)
        mono2 = self.merge_head(im2_path1)

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="constant",
            pad_two_side=True,
        )
        image1 = images[:, 0]
        image2 = images[:, 1]
        flow_predictions = []
        info_predictions = []

        N, _, H, W = image1.shape
        dilation = torch.ones(
            N, 1, H // 8, W // 8, device=image1.device, dtype=image1.dtype
        )

        # run the context network
        cnet_inputs = torch.cat([image1, image2], dim=1)
        # if self.use_da:

        cnet = self.cnet(cnet_inputs)

        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)

        bases1 = image_resizer.pad(bases1)
        bnet_inputs = bases1
        bnet = self.bnet(bnet_inputs)
        bnet = self.init_conv(bnet)
        netbases, ctxbases = torch.split(bnet, [self.dim, self.dim], dim=1)

        context = torch.cat((context, ctxbases), 1)
        net = torch.cat((net, netbases), 1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
        info_up = self.postprocess_predictions(info_up, image_resizer, is_flow=False)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)

            fmap1_8x = torch.cat((fmap1_8x, mono1), 1)
            fmap2_8x = torch.cat((fmap2_8x, mono2), 1)

            corr_fn = CorrBlock(
                fmap1_8x,
                fmap2_8x,
                corr_levels=self.corr_levels,
                corr_radius=self.corr_radius,
            )

        for itr in range(self.iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (
                coords_grid(N, H, W, device=image1.device, dtype=image1.dtype) + flow_8x
            ).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            info_up = self.postprocess_predictions(
                info_up, image_resizer, is_flow=False
            )
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if self.training:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min

                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (
                    torch.exp(-log_b).unsqueeze(1)
                )
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(
                    weight, dim=1, keepdim=True
                ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {
                "flows": flow_up[:, None],
                "flow_preds": flow_predictions,
                "info_preds": info_predictions,
                "nf_preds": nf_predictions,
            }
        else:
            return {"flows": flow_up[:, None], "flow_small": flow_8x}


class FlowSeekT(FlowSeek):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_t-things-16757c61.ckpt",
        "tar": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_t-tar-2a711278.ckpt",
        "tar-c": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_t-tar-c-dc6718fb.ckpt",
        "tar-c-t": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_t-tar-c-t-6be37a8c.ckpt",
        "tar-c-t-tskh": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_t-tar-c-t-tskh-41a591c8.ckpt",
    }

    def __init__(
        self,
        pretrain: Literal["resnet18", "resnet34"] = "resnet18",
        da_size: Literal["vits", "vitb"] = "vits",
        **kwargs,
    ):
        super().__init__(pretrain=pretrain, da_size=da_size, **kwargs)


class FlowSeekM(FlowSeek):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_m-things-503e3693.ckpt",
        "tar": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_m-tar-78daff58.ckpt",
        "tar-c": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_m-tar-c-84dc2106.ckpt",
        "tar-c-t": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_m-tar-c-t-261fd770.ckpt",
        "tar-c-t-tskh": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowseek_m-tar-c-t-tskh-7600557a.ckpt",
    }

    def __init__(
        self,
        pretrain: Literal["resnet18", "resnet34"] = "resnet34",
        da_size: Literal["vits", "vitb"] = "vitb",
        **kwargs,
    ):
        super().__init__(pretrain=pretrain, da_size=da_size, **kwargs)


@register_model
class flowseek_t(FlowSeekT):
    pass


@register_model
class flowseek_m(FlowSeekM):
    pass
