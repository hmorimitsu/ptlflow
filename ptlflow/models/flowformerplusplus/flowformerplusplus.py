from argparse import ArgumentParser, Namespace

import torch.nn.functional as F

from .FlowFormer.encoders import twins_svt_large, convnext_large
from .FlowFormer.PerCostFormer3.encoder import MemoryEncoder
from .FlowFormer.PerCostFormer3.decoder import MemoryDecoder
from .FlowFormer.PerCostFormer3.cnn import BasicEncoder
from .utils import compute_grid_indices, compute_weight
from ..base_model.base_model import BaseModel


class FlowFormerPlusPlus(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-chairs-a7745dd5.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things-4db3ecff.ckpt",
        "things288960": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things_288960-a4291d41.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-sintel-d14a1968.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-kitti-65b828c3.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=None, output_stride=32)

        H1, W1, H2, W2 = args.pic_size
        H_offset = (H1 - H2) // 2
        W_offset = (W1 - W2) // 2
        args.H_offset = H_offset
        args.W_offset = W_offset

        self.memory_encoder = MemoryEncoder(args)
        self.memory_decoder = MemoryDecoder(args)
        if args.cnet == "twins":
            self.context_encoder = twins_svt_large(
                pretrained=self.args.pretrain, del_layers=args.del_layers
            )
        elif args.cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")
        elif args.cnet == "convnext":
            self.context_encoder = convnext_large(pretrained=self.args.pretrain)

        if args.pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False

        self.showed_warning = False

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--cnet",
            type=str,
            choices=("basicencoder", "twins", "convnext"),
            default="twins",
        )
        parser.add_argument(
            "--fnet",
            type=str,
            choices=("basicencoder", "twins", "convnext"),
            default="twins",
        )
        parser.add_argument("--no_pretrain", action="store_false", dest="pretrain")
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument("--cost_heads_num", type=int, default=1)
        parser.add_argument("--cost_latent_input_dim", type=int, default=64)
        parser.add_argument("--cost_latent_token_num", type=int, default=8)
        parser.add_argument("--cost_latent_dim", type=int, default=128)
        parser.add_argument(
            "--pe", type=str, choices=("exp", "linear"), default="linear"
        )
        parser.add_argument("--encoder_depth", type=int, default=3)
        parser.add_argument("--encoder_latent_dim", type=int, default=256)
        parser.add_argument("--decoder_depth", type=int, default=32)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--vert_c_dim", type=int, default=64)
        parser.add_argument("--query_latent_dim", type=int, default=64)
        parser.add_argument(
            "--no_cost_encoder_res", action="store_false", dest="cost_encoder_res"
        )

        parser.add_argument(
            "--pic_size", type=int, nargs=4, default=(368, 496, 368, 496)
        )
        parser.add_argument("--not_del_layers", action="store_false", dest="del_layers")
        parser.add_argument("--pretrain_mode", action="store_true")
        parser.add_argument("--use_convertor", action="store_true")
        parser.add_argument(
            "--patch_embed", type=str, choices=("single", "no_relu"), default="single"
        )
        parser.add_argument(
            "--cross_attn",
            type=str,
            choices=("all", "part", "rep", "k3s2", "34"),
            default="all",
        )
        parser.add_argument("--droppath", type=float, default=0.0)
        parser.add_argument(
            "--vertical_encoder_attn",
            type=str,
            choices=("twins", "NA", "NA-twins"),
            default="twins",
        )
        parser.add_argument("--use_patch", action="store_true")
        parser.add_argument("--fix_pe", action="store_true")
        parser.add_argument("--gt_r", type=int, default=15)
        parser.add_argument(
            "--flow_or_pe", type=str, choices=("and", "pe", "flow"), default="and"
        )
        parser.add_argument("--no_sc", action="store_true")
        parser.add_argument("--r_16", type=int, default=-1)
        parser.add_argument("--quater_refine", action="store_true")
        parser.add_argument("--use_rpe", action="store_true")
        parser.add_argument("--gma", type=str, choices=("GMA", "GMA-SK"), default="GMA")
        parser.add_argument("--detach_local", action="store_true")
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

    def forward(self, inputs, mask=None, output=None):
        """Estimate optical flow between pair of frames"""
        if self.args.pretrain_mode:
            image1 = (image1 + 1) * 127.5
            image2 = (image2 + 1) * 127.5
            loss = self.pretrain_forward(image1, image2, mask=mask, output=output)
            return loss

        if self.args.train_size is not None:
            train_size = self.args.train_size
        else:
            train_size = self.train_size

        if self.args.use_tile_input and train_size is None and not self.showed_warning:
            print(
                "WARNING: --train_size is not provided and it cannot be loaded from the checkpoint either. Flowformer++ will run without input tile."
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

        context, _ = self.context_encoder(image1)
        context_quater = None

        (
            cost_memory,
            cost_patches,
            feat_s_quater,
            feat_t_quater,
        ) = self.memory_encoder(image1, image2, data, context)

        flow_predictions, flow_small = self.memory_decoder(
            cost_memory,
            context,
            context_quater,
            feat_s_quater,
            feat_t_quater,
            data,
            prev_flow=prev_flow,
            cost_patches=cost_patches,
        )

        return flow_predictions, flow_small
