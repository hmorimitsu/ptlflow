from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import BasicEncoder
from .encoder import MemoryEncoder
from .encoders import twins_svt_large
from .decoder import MemoryDecoder
from .utils import compute_grid_indices, compute_weight
from ..base_model.base_model import BaseModel


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
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class FlowFormer(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-chairs-84881320.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-things-dbe62dd3.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-sintel-cce498f8.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformer-kitti-d4225180.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=8)

        if self.args.gma is None:
            self.args.gma = True  # Use GMA by default, unless

        self.memory_encoder = MemoryEncoder(args)
        self.memory_decoder = MemoryDecoder(args)
        if args.cnet == "twins":
            self.context_encoder = twins_svt_large(pretrained=self.args.pretrain)
        elif args.cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")

        self.showed_warning = False

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--no_add_flow_token", action="store_false", dest="add_flow_token"
        )
        parser.add_argument(
            "--cnet", type=str, choices=("basicencoder", "twins"), default="twins"
        )
        parser.add_argument("--context_concat", action="store_true")
        parser.add_argument(
            "--no_cost_encoder_res", action="store_false", dest="cost_encoder_res"
        )
        parser.add_argument("--cost_heads_num", type=int, default=1)
        parser.add_argument("--cost_latent_dim", type=int, default=128)
        parser.add_argument("--cost_latent_input_dim", type=int, default=64)
        parser.add_argument("--cost_latent_token_num", type=int, default=8)
        parser.add_argument("--decoder_depth", type=int, default=32)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--encoder_depth", type=int, default=3)
        parser.add_argument("--encoder_latent_dim", type=int, default=256)
        parser.add_argument("--feat_cross_attn", action="store_true")
        parser.add_argument(
            "--fnet", type=str, choices=("basicencoder", "twins"), default="twins"
        )
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=400.0)
        parser.add_argument("--no_gma", action="store_false", dest="gma")
        parser.add_argument("--only_global", action="store_true")
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument(
            "--pe", type=str, choices=("exp", "linear"), default="linear"
        )
        parser.add_argument("--pretrain", action="store_true")
        parser.add_argument("--query_latent_dim", type=int, default=64)
        parser.add_argument("--use_mlp", action="store_true")
        parser.add_argument("--vert_c_dim", type=int, default=64)
        parser.add_argument("--vertical_conv", action="store_true")
        parser.add_argument(
            "--not_use_tile_input", action="store_false", dest="use_tile_input"
        )
        parser.add_argument("--tile_height", type=int, default=432)
        parser.add_argument("--tile_sigma", type=float, default=0.05)
        parser.add_argument(
            "--train_size",
            type=int,
            nargs=2,
            default=None,
            help="train_size will be normally loaded from the checkpoint. However, if you provide this value, it will override the value from the checkpoint.",
        )
        return parser

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        if self.args.train_size is not None:
            train_size = self.args.train_size
        else:
            train_size = self.train_size

        if self.args.use_tile_input and train_size is None and not self.showed_warning:
            print(
                "WARNING: --train_size is not provided and it cannot be loaded from the checkpoint either. Flowformer will run without input tile."
            )
            self.showed_warning = True

        if self.args.use_tile_input and train_size is not None:
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

        if self.args.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions, flow_small = self.memory_decoder(
            cost_memory, context, data, prev_flow=prev_flow
        )

        return flow_predictions, flow_small
