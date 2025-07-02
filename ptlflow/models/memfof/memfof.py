from typing import Literal
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torchvision.models import (
    ResNet34_Weights,
    WeightsEnum,
)

from .update import GMAUpdateBlock
from .corr import CorrBlock
from .utils import coords_grid, InputPadder
from .extractor import ResNetFPN16x
from .layer import conv3x3
from .gma import Attention

from ptlflow.utils.registry import register_model
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
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss_i = outputs["nf_preds"][i]
            final_mask = (
                (~torch.isnan(loss_i.detach()))
                & (~torch.isinf(loss_i.detach()))
                & valid[:, :, None]
            )

            fms = final_mask.sum()
            if fms > 0.5:
                flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
            else:
                flow_loss += (0.0 * loss_i).sum().nan_to_num(0.0)

        return flow_loss


class MEMFOF(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-kitti-ed27d6f1.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-sintel-cbb45e24.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-spring-f8a968f7.ckpt",
        "tartan": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-tartan-7ca03da2.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-things-11146736.ckpt",
        "tskh": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memfof-tskh-6fb0c129.ckpt",
    }

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet34", "resnet50"] = "resnet34",
        backbone_weights: WeightsEnum = ResNet34_Weights.IMAGENET1K_V1,
        dim: int = 512,
        corr_levels: int = 4,
        corr_radius: int = 4,
        iters: int = 8,
        num_blocks: int = 2,
        gamma: float = 0.8,
        max_flow: float = 400,
        use_var: bool = True,
        var_min: float = 0.0,
        var_max: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=32, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )
        self.dim = dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.iters = iters
        self.use_var = use_var
        self.var_min = var_min
        self.var_max = var_max

        self.cnet = ResNetFPN16x(9, dim * 2, backbone, backbone_weights)

        self.init_conv = conv3x3(2 * dim, 2 * dim)

        self.upsample_weight = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim * 2, 2 * 16 * 16 * 9, 1, padding=0),
        )

        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim, 2 * 6, 3, padding=1),
        )

        self.fnet = ResNetFPN16x(3, dim * 2, backbone, backbone_weights)

        corr_channel = corr_levels * (corr_radius * 2 + 1) ** 2
        self.update_block = GMAUpdateBlock(num_blocks, corr_channel, hdim=dim, cdim=dim)

        self.att = Attention(dim=dim, heads=1, dim_head=dim)

    def forward(self, inputs, fmap_cache=[None, None, None]):
        """Forward pass of the MEMFOF model.

        Parameters
        ----------
        fmap_cache : list[torch.Tensor | None], optional
            Cache for feature maps to be used in current forward pass, by default [None, None, None]
        """

        B, _, _, H, W = inputs["images"].shape

        if "flows" in inputs:
            flow_gt = inputs["flows"][:, 0]
        else:
            flow_gts = torch.zeros(B, 2, 2, H, W, device=inputs["images"].device)

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )
        B, _, _, H, W = images.shape

        flow_predictions = []
        info_predictions = []

        dilation = torch.ones(B, 1, H // 16, W // 16, device=images.device)

        # run the context network
        cnet = self.cnet(torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)
        attention = self.att(context)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)

        flow_16x_21 = flow_update[:, 0:2]
        info_16x_21 = flow_update[:, 2:6]

        flow_16x_23 = flow_update[:, 6:8]
        info_16x_23 = flow_update[:, 8:12]

        if self.training or self.iters == 0:
            flow_up_21, info_up_21 = self._upsample_data(
                flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
            )
            flow_up_23, info_up_23 = self._upsample_data(
                flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
            )
            flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
            info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        if self.iters > 0:
            # run the feature network
            fmap1_16x = (
                self.fnet(images[:, 0])
                if fmap_cache[0] is None
                else fmap_cache[0].clone().to(cnet)
            )
            fmap2_16x = (
                self.fnet(images[:, 1])
                if fmap_cache[1] is None
                else fmap_cache[1].clone().to(cnet)
            )
            fmap3_16x = (
                self.fnet(images[:, 2])
                if fmap_cache[2] is None
                else fmap_cache[2].clone().to(cnet)
            )
            corr_fn_21 = CorrBlock(
                fmap2_16x, fmap1_16x, self.corr_levels, self.corr_radius
            )
            corr_fn_23 = CorrBlock(
                fmap2_16x, fmap3_16x, self.corr_levels, self.corr_radius
            )

        for itr in range(self.iters):
            B, _, H, W = flow_16x_21.shape
            flow_16x_21 = flow_16x_21.detach()
            flow_16x_23 = flow_16x_23.detach()

            coords21 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_21
            ).detach()
            coords23 = (
                coords_grid(B, H, W, device=images.device) + flow_16x_23
            ).detach()

            corr_21 = corr_fn_21(coords21, dilation=dilation)
            corr_23 = corr_fn_23(coords23, dilation=dilation)

            corr = torch.cat([corr_21, corr_23], dim=1)
            flow_16x = torch.cat([flow_16x_21, flow_16x_23], dim=1)

            net = self.update_block(net, context, corr, flow_16x, attention)

            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)

            flow_16x_21 = flow_16x_21 + flow_update[:, 0:2]
            info_16x_21 = flow_update[:, 2:6]

            flow_16x_23 = flow_16x_23 + flow_update[:, 6:8]
            info_16x_23 = flow_update[:, 8:12]

            if self.training or itr == self.iters - 1:
                flow_up_21, info_up_21 = self._upsample_data(
                    flow_16x_21, info_16x_21, weight_update[:, : 16 * 16 * 9]
                )
                flow_up_23, info_up_23 = self._upsample_data(
                    flow_16x_23, info_16x_23, weight_update[:, 16 * 16 * 9 :]
                )
                flow_predictions.append(torch.stack([flow_up_21, flow_up_23], dim=1))
                info_predictions.append(torch.stack([info_up_21, info_up_23], dim=1))

        for i in range(len(info_predictions)):
            flow_predictions[i] = self.postprocess_predictions(
                flow_predictions[i], image_resizer, is_flow=True
            )
            info_predictions[i] = self.postprocess_predictions(
                info_predictions[i], image_resizer, is_flow=False
            )

        new_fmap_cache = [None, None, None]
        if self.iters > 0:
            new_fmap_cache[0] = fmap1_16x.clone().cpu()
            new_fmap_cache[1] = fmap2_16x.clone().cpu()
            new_fmap_cache[2] = fmap3_16x.clone().cpu()

        if not self.training:
            return {
                "flows": flow_predictions[-1][:, 1:],
                "fmap_cache": new_fmap_cache,
            }
        else:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.var_max
                    var_min = self.var_min

                nf_losses = []
                for k in range(2):
                    raw_b = info_predictions[i][:, k, 2:]
                    log_b = torch.zeros_like(raw_b)
                    weight = info_predictions[i][:, k, :2]
                    # Large b Component
                    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                    # Small b Component
                    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                    # term2: [N, 2, m, H, W]
                    term2 = (
                        (flow_gt[:, k] - flow_predictions[i][:, k]).abs().unsqueeze(2)
                    ) * (torch.exp(-log_b).unsqueeze(1))
                    # term1: [N, m, H, W]
                    term1 = weight - math.log(2) - log_b
                    nf_loss = torch.logsumexp(
                        weight, dim=1, keepdim=True
                    ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                    nf_losses.append(nf_loss)

                nf_predictions.append(torch.stack(nf_losses, dim=1))

            return {
                "flow_preds": flow_predictions,
                "info_preds": info_predictions,
                "nf_preds": nf_predictions,
                "fmap_cache": new_fmap_cache,
            }

    def _upsample_data(self, flow, info, mask):
        """Upsample [H/16, W/16, C] -> [H, W, C] using convex combination"""
        """Forward pass of the MEMFOF model.

        Parameters
        ----------
        flow : torch.Tensor
            Tensor of shape [B, 2, H / 16, W / 16].
        info : torch.Tensor
            Tensor of shape [B, 4, H / 16, W / 16].
        mask : torch.Tensor
            Tensor of shape [B, 9 * 16 * 16, H / 16, W / 16]
        Returns
        -------
        flow : torch.Tensor
            Tensor of shape [B, 2, H, W]
        info : torch.Tensor
            Tensor of shape [B, 4, H, W]
        """
        B, C, H, W = info.shape
        mask = mask.view(B, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(16 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(B, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(B, 2, 16 * H, 16 * W), up_info.reshape(
            B, C, 16 * H, 16 * W
        )


@register_model
class memfof(MEMFOF):
    pass
