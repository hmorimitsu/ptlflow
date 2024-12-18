import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable, ptlflow_trained
from ptlflow.utils.utils import forward_interpolate_batch
from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr_lcv import LearnableCorrBlock
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


class LCV_RAFT(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/lcv_raft-chairs-8063d698.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/lcv_raft-things-4c7233b8.ckpt",
    }

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout: float = 0.0,
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
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

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=dropout)
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=dropout
        )
        self.update_block = BasicUpdateBlock(
            corr_levels=corr_levels, corr_radius=corr_radius, hidden_dim=hdim
        )

        self.corr_block = LearnableCorrBlock(256, self.corr_levels, self.corr_radius)

        self.has_trained_on_ptlflow = True

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
        coords = torch.stack(coords[::-1], dim=0).float()
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
            bgr_to_rgb=False,
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

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_pyramid = self.corr_block.compute_cost_volume(fmap1, fmap2)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            forward_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])
            coords1 = coords1 + forward_flow

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = self.corr_block(corr_pyramid, coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = self.upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs


class LCV_RAFTSmall(LCV_RAFT):
    pretrained_checkpoints = {}

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 3,
        dropout: float = 0,
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(
            corr_levels, corr_radius, dropout, gamma, max_flow, iters, **kwargs
        )

        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64

        # feature network, context network, and update block
        self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=dropout)
        self.cnet = SmallEncoder(
            output_dim=hdim + cdim, norm_fn="none", dropout=dropout
        )
        self.update_block = SmallUpdateBlock(
            corr_levels=corr_levels, corr_radius=corr_radius, hidden_dim=hdim
        )

        self.corr_block = LearnableCorrBlock(128, self.corr_levels, self.corr_radius)


@register_model
@trainable
@ptlflow_trained
class lcv_raft(LCV_RAFT):
    pass


@register_model
@trainable
@ptlflow_trained
class lcv_raft_small(LCV_RAFTSmall):
    pass
