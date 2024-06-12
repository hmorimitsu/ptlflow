from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn

from ptlflow.utils.utils import forward_interpolate_batch
from .optimizer import fetch_optimizer
from .MemFlowNet.corr import CorrBlock
from .memory_manager_skflow import MemoryManager
from .MemFlowNet.MemFlow import MemFlowNet
from .MemFlowNet.memory_util import *
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow
        self.filter_epe = args.filter_epe

    # def forward(self, flow_preds, flow_gt, valid):
    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""

        # print(flow_gt.shape, valid.shape, flow_preds[0].shape)
        # exit()

        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=2).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)

            flow_pre = flow_preds[i]
            i_loss = (flow_pre - flow_gt).abs()

            _valid = valid[:, :, None]
            if self.filter_epe:
                loss_mag = torch.sum(i_loss**2, dim=2).sqrt()
                mask = loss_mag > 1000
                # print(mask.shape, _valid.shape)
                if torch.any(mask):
                    print(
                        "[Found extrem epe. Filtered out. Max is {}. Ratio is {}]".format(
                            torch.max(loss_mag), torch.mean(mask.float())
                        )
                    )
                    _valid = _valid & (~mask[:, :, None])

            flow_loss += i_weight * (_valid * i_loss).mean()

        return flow_loss


class MemFlow(BaseModel):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow-things-90d0b74c.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow-sintel-38621d84.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow-kitti-ee6cbf09.ckpt",
        "spring": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow-spring-7ee1b984.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=None, output_stride=8)
        self.network = MemFlowNet(args)

        self.clear_memory()

        self.showed_warning = False

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument(
            "--cnet",
            type=str,
            default="basicencoder",
            choices=("basicencoder", "twins"),
        )
        parser.add_argument(
            "--fnet",
            type=str,
            default="basicencoder",
            choices=("basicencoder", "twins"),
        )
        parser.add_argument("--feat_dim", type=int, default=256)
        parser.add_argument(
            "--gma", type=str, default="GMA-SK2", choices=("GMA", "GMA-SK", "GMA-SK2")
        )
        parser.add_argument(
            "--corr_fn", type=str, default="default", choices=("default",)
        )
        parser.add_argument("--down_ratio", type=int, default=8)
        parser.add_argument("--decoder_depth", type=int, default=15)
        parser.add_argument("--cost_heads_num", type=int, default=1)
        parser.add_argument("--mem_every", type=int, default=1)
        parser.add_argument("--enable_long_term", action="store_true")
        parser.add_argument(
            "--not_enable_long_term_count_usage",
            action="store_false",
            dest="enable_long_term_count_usage",
        )
        parser.add_argument("--max_mid_term_frames", type=int, default=2)
        parser.add_argument("--min_mid_term_frames", type=int, default=1)
        parser.add_argument("--num_prototypes", type=int, default=128)
        parser.add_argument("--max_long_term_elements", type=int, default=10000)
        parser.add_argument("--top_k", type=int, default=None)
        parser.add_argument(
            "--critical_params",
            type=str,
            nargs="+",
            default=(
                "cnet",
                "fnet",
                "pretrain",
                "corr_levels",
                "decoder_depth",
                "train_avg_length",
            ),
            choices=(
                "cnet",
                "fnet",
                "pretrain",
                "corr_levels",
                "decoder_depth",
                "train_avg_length",
            ),
        )
        parser.add_argument("--filter_epe", action="store_true")
        parser.add_argument("--not_pretrain", action="store_false", dest="pretrain")
        parser.add_argument(
            "--train_avg_length",
            type=int,
            default=None,
            help="train_avg_length will be normally loaded from the checkpoint. However, if you provide this value, it will override the value from the checkpoint.",
        )
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=400.0)
        return parser

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = -self.args.mem_every
        self.memory = MemoryManager(config=self.args)

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        if self.args.train_avg_length is not None:
            train_avg_length = self.args.train_avg_length
        else:
            train_avg_length = self.train_avg_length
            self.args.train_avg_length = train_avg_length

        if train_avg_length is None and not self.showed_warning:
            print(
                "WARNING: --train_avg_length is not provided and it cannot be loaded from the checkpoint either. It will be set as 6750, but this may not be the optimal value."
            )
            train_avg_length = 6750
            self.args.train_avg_length = train_avg_length
            self.showed_warning = True

        assert (
            train_avg_length is not None
        ), "train_avg_length could not be loaded from the checkpoint and it was not provided as an argument."

        if (
            "meta" in inputs
            and "is_seq_start" in inputs["meta"]
            and inputs["meta"]["is_seq_start"]
        ):
            self.clear_memory()

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        flow_init = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            flow_init = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])

        # image: 1*2*3*H*W
        self.curr_ti += 1

        end = True
        if "meta" in inputs and "is_seq_end" in inputs["meta"]:
            end = inputs["meta"]["is_seq_end"]
        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.args.mem_every) and (
            not end
        )

        # B, C, H, W
        query, key, net, inp = self.network.encode_context(images[:, 0, ...])
        # B, T, C, H, W
        coords0, coords1, fmaps = self.network.encode_features(
            images, flow_init=flow_init
        )

        # predict flow
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...], num_levels=4, radius=4)

        for itr in range(self.args.decoder_depth):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            (
                motion_features,
                current_value,
            ) = self.network.update_block.get_motion_and_value(flow, corr)
            # get global motion
            memory_readout = self.memory.match_memory(
                query, key, current_value, scale=self.network.att.scale
            )
            motion_features_global = (
                motion_features
                + self.network.update_block.aggregator.gamma * memory_readout
            )
            net, up_mask, delta_flow = self.network.update_block(
                net, inp, motion_features, motion_features_global
            )
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
        # upsample predictions
        flow_up = self.network.upsample_flow(coords1 - coords0, up_mask)
        flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)

        # save as memory if needed
        if is_mem_frame:
            self.memory.add_memory(key, current_value)
            self.last_mem_ti = self.curr_ti

        # if self.training:
        #     outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        # else:
        outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs

    def configure_optimizers(self):
        optimizer, lr_scheduler = fetch_optimizer(self, self.args)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


class MemFlowT(MemFlow):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow_t-things-6028d89f.ckpt",
        "things_kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow_t-things_kitti-542e0a1c.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow_t-sintel-d2df0424.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/memflow_t-kitti-9eeabb65.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        args.cnet = "twins"
        args.fnet = "twins"
        super().__init__(args)
