from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.registry import register_model, trainable
from .cnn import BasicEncoder
from .encoder import MemoryEncoder
from .encoders import twins_svt_large
from .decoder import MemoryDecoder
from .utils import compute_grid_indices, compute_weight
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
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class FlowFormer(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-chairs-84881320.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-things-dbe62dd3.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-sintel-cce498f8.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-kitti-d4225180.ckpt",
    }

    def __init__(
        self,
        add_flow_token: bool = True,
        cnet: str = "twins",
        context_concat: bool = False,
        cost_encoder_res: bool = True,
        cost_heads_num: int = 1,
        cost_latent_dim: int = 128,
        cost_latent_input_dim: int = 64,
        cost_latent_token_num: int = 8,
        decoder_depth: int = 32,
        dropout: float = 0.0,
        encoder_depth: int = 3,
        encoder_latent_dim: int = 256,
        feat_cross_attn: bool = False,
        fnet: str = "twins",
        gamma: float = 0.8,
        max_flow: float = 400.0,
        gma: bool = True,
        only_global: bool = False,
        patch_size: int = 8,
        pe: str = "linear",
        pretrain: bool = False,
        query_latent_dim: int = 64,
        use_mlp: bool = False,
        mlp_expansion_factor: int = 4,
        vert_c_dim: int = 64,
        vertical_conv: bool = False,
        cost_scale_aug: Optional[tuple[float, float]] = None,
        use_tile_input: bool = True,
        tile_height: int = 432,
        tile_sigma: float = 0.05,
        train_size: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=8, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.add_flow_token = add_flow_token
        self.cnet = cnet
        self.context_concat = context_concat
        self.cost_encoder_res = cost_encoder_res
        self.cost_heads_num = cost_heads_num
        self.cost_latent_dim = cost_latent_dim
        self.cost_latent_input_dim = cost_latent_input_dim
        self.cost_latent_token_num = cost_latent_token_num
        self.decoder_depth = decoder_depth
        self.dropout = dropout
        self.encoder_depth = encoder_depth
        self.encoder_latent_dim = encoder_latent_dim
        self.feat_cross_attn = feat_cross_attn
        self.fnet = fnet
        self.gma = gma
        self.only_global = only_global
        self.patch_size = patch_size
        self.pe = pe
        self.pretrain = pretrain
        self.query_latent_dim = query_latent_dim
        self.use_mlp = use_mlp
        self.mlp_expansion_factor = mlp_expansion_factor
        self.vert_c_dim = vert_c_dim
        self.vertical_conv = vertical_conv
        self.cost_scale_aug = cost_scale_aug
        self.use_tile_input = use_tile_input
        self.tile_height = tile_height
        self.tile_sigma = tile_sigma
        self.train_size = train_size

        if self.gma is None:
            self.gma = True  # Use GMA by default, unless

        self.memory_encoder = MemoryEncoder(
            fnet,
            encoder_latent_dim=encoder_latent_dim,
            pretrain=pretrain,
            feat_cross_attn=feat_cross_attn,
            cost_heads_num=cost_heads_num,
            patch_size=patch_size,
            cost_latent_input_dim=cost_latent_input_dim,
            pe=pe,
            encoder_depth=encoder_depth,
            cost_latent_dim=cost_latent_dim,
            dropout=dropout,
            use_mlp=use_mlp,
            cost_scale_aug=cost_scale_aug,
            mlp_expansion_factor=mlp_expansion_factor,
            vert_c_dim=vert_c_dim,
            vertical_conv=vertical_conv,
            cost_latent_token_num=cost_latent_token_num,
            cost_encoder_res=cost_encoder_res,
        )
        self.memory_decoder = MemoryDecoder(
            query_latent_dim=query_latent_dim,
            cost_heads_num=cost_heads_num,
            decoder_depth=decoder_depth,
            gma=gma,
            only_global=only_global,
            patch_size=patch_size,
            cost_latent_dim=cost_latent_dim,
            add_flow_token=add_flow_token,
            dropout=dropout,
        )
        if cnet == "twins":
            self.context_encoder = twins_svt_large(pretrained=self.pretrain)
        elif cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")

        self.showed_warning = False

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        if self.train_size is not None:
            train_size = self.train_size
        else:
            train_size = self.train_size

        if self.use_tile_input and train_size is None and not self.showed_warning:
            print(
                "WARNING: --train_size is not provided and it cannot be loaded from the checkpoint either. Flowformer will run without input tile."
            )
            self.showed_warning = True

        if self.use_tile_input and train_size is not None:
            return self.forward_tile(inputs, train_size)
        else:
            return self.forward_pad(inputs)

    def forward_pad(self, inputs):
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        prev_flow = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            prev_flow = inputs["prev_preds"]["flow_small"]

        flow_predictions, flow_small = self.predict(
            images[:, 0], images[:, 1], prev_flow
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
            resize_mode="pad",
            target_size=image_size,
            pad_two_side=False,
            pad_mode="constant",
            pad_value=-1,
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

    def predict(self, image1, image2, prev_flow=None):
        """Estimate optical flow between pair of frames"""

        data = {}

        if self.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions, flow_small = self.memory_decoder(
            cost_memory, context, data, prev_flow=prev_flow
        )

        return flow_predictions, flow_small


@register_model
@trainable
class flowformer(FlowFormer):
    pass
