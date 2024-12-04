from typing import Optional

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
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


class MatchFlow(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-chairs-02519b53.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-kitti-bc72ce81.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-sintel-683422f4.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/matchflow_gma-things-49295bd8.ckpt",
    }

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout: float = 0.0,
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
        matching_model_path: str = "",
        num_heads: int = 1,
        raft: bool = False,
        use_tile_input: bool = True,
        tile_height: int = 416,
        tile_sigma: float = 0.05,
        position_only: bool = False,
        position_and_content: bool = False,
        alternate_corr: bool = False,
        train_size: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=32, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.dropout = dropout
        self.gamma = gamma
        self.iters = iters
        self.matching_model_path = matching_model_path
        self.num_heads = num_heads
        self.raft = raft
        self.use_tile_input = use_tile_input
        self.tile_height = tile_height
        self.tile_sigma = tile_sigma
        self.position_only = position_only
        self.position_and_content = position_and_content
        self.alternate_corr = alternate_corr
        self.train_size = train_size

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        # feature network, context network, and update block
        self.fnet = MatchingModel()

        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=dropout
        )
        if self.raft is False:
            self.update_block = GMAUpdateBlock(
                corr_levels=corr_levels,
                corr_radius=corr_radius,
                num_heads=num_heads,
                hidden_dim=hdim,
            )
            self.att = Attention(
                position_only=position_only,
                position_and_content=position_and_content,
                dim=cdim,
                heads=self.num_heads,
                max_pos_size=160,
                dim_head=cdim,
            )
        else:
            self.update_block = BasicUpdateBlock(
                corr_levels=corr_levels, corr_radius=corr_radius, hidden_dim=hdim
            )

        if self.alternate_corr and alt_cuda_corr is None:
            logger.warning(
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
        self.fnet = MatchingModel(train_size=self.train_size)

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
        if self.train_size is not None:
            train_size = self.train_size
        else:
            train_size = self.train_size

        if self.use_tile_input and train_size is None and not self.showed_warning:
            logger.warning(
                "--train_size is not provided and it cannot be loaded from the checkpoint either. Matchflow will run without input tile."
            )
            self.showed_warning = True

        if self.use_tile_input and train_size is not None:
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
        image_size = (max(self.tile_height, input_size[-2]), input_size[-1])
        hws = compute_grid_indices(image_size, train_size)
        device = inputs["images"].device
        weights = compute_weight(
            hws, image_size, train_size, self.tile_sigma, device=device
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
            radius=self.corr_radius,
            num_levels=self.corr_levels,
            alternate_corr=self.alternate_corr,
        )

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        # attention, att_c, att_p = self.att(inp)
        if self.raft is False:
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_prev is not None:
            forward_flow = forward_interpolate_batch(flow_prev)
            coords1 = coords1 + forward_flow

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            if self.raft is False:
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

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        dropout: float = 0,
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
        matching_model_path: str = "",
        num_heads: int = 1,
        raft: bool = True,
        use_tile_input: bool = True,
        tile_height: int = 416,
        tile_sigma: float = 0.05,
        position_only: bool = False,
        position_and_content: bool = False,
        alternate_corr: bool = False,
        train_size: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            corr_levels,
            corr_radius,
            dropout,
            gamma,
            max_flow,
            iters,
            matching_model_path,
            num_heads,
            raft,
            use_tile_input,
            tile_height,
            tile_sigma,
            position_only,
            position_and_content,
            alternate_corr,
            train_size,
            **kwargs,
        )


@register_model
@trainable
class matchflow(MatchFlow):
    pass


@register_model
@trainable
class matchflow_raft(MatchFlowRAFT):
    pass
