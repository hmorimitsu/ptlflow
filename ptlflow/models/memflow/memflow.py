from typing import Optional, Sequence

import torch
import torch.nn as nn

from ptlflow.utils.registry import register_model, trainable
from ptlflow.utils.utils import forward_interpolate_batch
from .optimizer import fetch_optimizer
from .MemFlowNet.corr import CorrBlock
from .memory_manager_skflow import MemoryManager
from .MemFlowNet.MemFlow import MemFlowNet
from .MemFlowNet.memory_util import *
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float, filter_epe: bool):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.filter_epe = filter_epe

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

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        cnet: str = "basicencoder",
        fnet: str = "basicencoder",
        feat_dim: int = 256,
        gma: str = "GMA-SK2",
        corr_fn: str = "default",
        down_ratio: int = 8,
        decoder_depth: int = 15,
        cost_heads_num: int = 1,
        mem_every: int = 1,
        enable_long_term: bool = False,
        enable_long_term_count_usage: bool = True,
        max_mid_term_frames: int = 2,
        min_mid_term_frames: int = 2,
        num_prototypes: int = 128,
        max_long_term_elements: int = 10000,
        top_k: Optional[int] = None,
        critical_params: Sequence[str] = (
            "cnet",
            "fnet",
            "pretrain",
            "corr_levels",
            "decoder_depth",
            "train_avg_length",
        ),
        filter_epe: bool = False,
        pretrain: bool = True,
        train_avg_length: Optional[int] = None,
        gamma: float = 0.8,
        max_flow: float = 400,
        optimizer_name="adamw",
        scheduler_name="OneCycleLR",
        canonical_lr=2.1e-4,
        twins_lr_factor=2.1e-4,
        adam_decay=1e-4,
        adamw_decay=1e-4,
        epsilon=1e-8,
        num_steps=600000,
        anneal_strategy="linear",
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=8, loss_fn=SequenceLoss(gamma, max_flow, filter_epe), **kwargs
        )

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.cnet = cnet
        self.fnet = fnet
        self.feat_dim = feat_dim
        self.gma = gma
        self.corr_fn = corr_fn
        self.down_ratio = down_ratio
        self.decoder_depth = decoder_depth
        self.cost_heads_num = cost_heads_num
        self.mem_every = mem_every
        self.enable_long_term = enable_long_term
        self.enable_long_term_count_usage = enable_long_term_count_usage
        self.max_mid_term_frames = max_mid_term_frames
        self.min_mid_term_frames = min_mid_term_frames
        self.num_prototypes = num_prototypes
        self.max_long_term_elements = max_long_term_elements
        self.top_k = top_k
        self.critical_params = critical_params
        self.filter_epe = filter_epe
        self.pretrain = pretrain
        self.train_avg_length = train_avg_length
        self.gamma = gamma
        self.max_flow = max_flow

        self.optimizer_name = (optimizer_name,)
        self.scheduler_name = (scheduler_name,)
        self.canonical_lr = (canonical_lr,)
        self.twins_lr_factor = (twins_lr_factor,)
        self.adam_decay = (adam_decay,)
        self.adamw_decay = (adamw_decay,)
        self.epsilon = (epsilon,)
        self.num_steps = (num_steps,)
        self.anneal_strategy = (anneal_strategy,)

        self.network = MemFlowNet(
            cnet=cnet,
            fnet=fnet,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            decoder_depth=decoder_depth,
            pretrain=pretrain,
            gma=gma,
            corr_fn=corr_fn,
            train_avg_length=train_avg_length,
        )

        if self.train_avg_length is None:
            train_avg_length = 6750
            print(
                f"WARNING: --train_avg_length is not provided and it cannot be loaded from the checkpoint either. It will be set as {train_avg_length}, but this may not be the optimal value."
            )
            self.train_avg_length = train_avg_length

        self.clear_memory()

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = -self.mem_every
        self.memory = MemoryManager(
            train_avg_length=self.train_avg_length,
            enable_long_term=self.enable_long_term,
            enable_long_term_count_usage=self.enable_long_term_count_usage,
            top_k=self.top_k,
            max_mid_term_frames=self.max_mid_term_frames,
            min_mid_term_frames=self.min_mid_term_frames,
        )

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
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
        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)

        # B, C, H, W
        query, key, net, inp = self.network.encode_context(images[:, 0, ...])
        # B, T, C, H, W
        coords0, coords1, fmaps = self.network.encode_features(
            images, flow_init=flow_init
        )

        # predict flow
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...], num_levels=4, radius=4)

        for itr in range(self.decoder_depth):
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
        optimizer, lr_scheduler = fetch_optimizer(
            model=self,
            optimizer_name=self.optimizer_name,
            scheduler_name=self.scheduler_name,
            canonical_lr=self.canonical_lr,
            twins_lr_factor=self.twins_lr_factor,
            adam_decay=self.adam_decay,
            adamw_decay=self.adamw_decay,
            epsilon=self.epsilon,
            num_steps=self.num_steps,
            anneal_strategy=self.anneal_strategy,
        )
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

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        cnet: str = "twins",
        fnet: str = "twins",
        feat_dim: int = 256,
        gma: str = "GMA-SK2",
        corr_fn: str = "default",
        down_ratio: int = 8,
        decoder_depth: int = 15,
        cost_heads_num: int = 1,
        mem_every: int = 1,
        enable_long_term: bool = False,
        enable_long_term_count_usage: bool = True,
        max_mid_term_frames: int = 2,
        min_mid_term_frames: int = 2,
        num_prototypes: int = 128,
        max_long_term_elements: int = 10000,
        top_k: int | None = None,
        critical_params: Sequence[str] = (
            "cnet",
            "fnet",
            "pretrain",
            "corr_levels",
            "decoder_depth",
            "train_avg_length",
        ),
        filter_epe: bool = False,
        pretrain: bool = True,
        train_avg_length: int | None = None,
        gamma: float = 0.8,
        max_flow: float = 400,
        **kwargs,
    ) -> None:
        super().__init__(
            corr_levels,
            corr_radius,
            cnet,
            fnet,
            feat_dim,
            gma,
            corr_fn,
            down_ratio,
            decoder_depth,
            cost_heads_num,
            mem_every,
            enable_long_term,
            enable_long_term_count_usage,
            max_mid_term_frames,
            min_mid_term_frames,
            num_prototypes,
            max_long_term_elements,
            top_k,
            critical_params,
            filter_epe,
            pretrain,
            train_avg_length,
            gamma,
            max_flow,
            **kwargs,
        )


@register_model
@trainable
class memflow(MemFlow):
    pass


@register_model
@trainable
class memflow_t(MemFlowT):
    pass
