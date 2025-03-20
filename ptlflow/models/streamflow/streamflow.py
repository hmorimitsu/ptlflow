from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from .corr import CorrBlock
from .gma import Attention
from .twins_csc import Twins_CSC
from .update import SKUpdateBlock_TAM_v3
from .utils import coords_grid
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
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class StreamFlow(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/streamflow-kitti-eaafa6ed.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/streamflow-sintel-af557e5e.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/streamflow-spring-092f8a17.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/streamflow-things-c640255a.ckpt",
    }

    def __init__(
        self,
        decoder_dim: int = 256,
        corr_levels: int = 4,
        corr_radius: int = 4,
        num_heads: int = 1,
        pcupdater_conv: list[int] = [1, 7],
        T: int = 4,
        k_conv: list[int] = [1, 15],
        use_gma: bool = True,
        iters: int = 15,
        twins_pretrained_ckpt: Optional[str] = None,
        gamma: float = 0.8,
        max_flow: float = 400,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=8, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.ratio = 8
        self.corr_radius = corr_radius
        self.iters = iters
        self.context_dim = cdim = decoder_dim // 2
        self.hidden_dim = decoder_dim // 2

        # feature network, context network, and update block
        self.fnet = Twins_CSC(pretrained_ckpt=twins_pretrained_ckpt)
        self.cnet = Twins_CSC(pretrained_ckpt=twins_pretrained_ckpt)
        self.update_block = SKUpdateBlock_TAM_v3(
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            use_gma=use_gma,
            pcupdater_conv=pcupdater_conv,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            T=T,
            k_conv=k_conv,
        )
        if use_gma:
            self.att = Attention(
                dim=cdim,
                heads=num_heads,
                dim_head=cdim,
            )
        else:
            self.att = None

    def freeze_untemporal(self):
        for name, param in self.named_parameters():
            if "temporal" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, ratio=8):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(
            N, H // ratio, W // ratio, dtype=img.dtype, device=img.device
        )

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0

    def upsample_flow(self, flow, mask, ratio=8):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(ratio * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, ratio * H, ratio * W)

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

        T = images.shape[1]

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmaps = self.fnet(images)
        cnets = self.cnet(images[:, :-1])

        corr_fns = [
            CorrBlock(fmaps[:, i], fmaps[:, i + 1], radius=self.corr_radius)
            for i in range(T - 1)
        ]
        coord_0s = [
            self.initialize_flow(images[:, i], ratio=self.ratio) for i in range(T - 1)
        ]
        coord_1s = [
            self.initialize_flow(images[:, i], ratio=self.ratio) for i in range(T - 1)
        ]

        # if flow_init is not None:
        #     coord_1s = [coord_1s[i] + flow_init[i] for i in range(len(flow_init))]

        nets, inps, attentions = [], [], []
        nets, inps = torch.split(cnets, [hdim, cdim], dim=2)
        nets = torch.tanh(rearrange(nets, "B T C H W -> (B T) C H W"))
        inps = torch.relu(inps)
        inps = rearrange(inps, "B T C H W -> (B T) C H W")
        if self.att is not None:
            attentions = self.att(inps)
        else:
            attentions = None

        flow_predictions_list = [[] for i in range(T - 1)]
        for itr in range(self.iters):
            coord_1s = [coord.detach() for coord in coord_1s]
            corrs = rearrange(
                torch.stack([corr_fns[i](coord_1s[i]) for i in range(T - 1)], dim=1),
                "B T C H W -> (B T) C H W",
            )
            flows = rearrange(
                torch.stack([coord_1s[i] - coord_0s[i] for i in range(T - 1)], dim=1),
                "B T C H W -> (B T) C H W",
            )

            nets, up_masks, delta_flows = self.update_block(
                nets, inps, corrs, flows, attentions, T=T - 1
            )

            coord_1s = [coord_1s[i] + delta_flows[:, i] for i in range(T - 1)]
            for i in range(T - 1):
                flow_predictions_list[i].append(
                    self.upsample_flow(
                        coord_1s[i] - coord_0s[i], up_masks[:, i], ratio=self.ratio
                    )
                )

        out_flow = [
            self.postprocess_predictions(
                flow_predictions[-1], image_resizer, is_flow=True
            )
            for flow_predictions in flow_predictions_list
        ]
        out_flow = torch.stack(out_flow, 1)
        if self.training:
            outputs = {"flows": out_flow, "flow_preds": flow_predictions_list}
        else:
            flow_small = [coord_1s[i] - coord_0s[i] for i in range(T - 1)]
            flow_small = torch.stack(flow_small, 1)
            outputs = {"flows": out_flow, "flow_small": flow_small}

        return outputs


@register_model
@trainable
class streamflow(StreamFlow):
    pass
