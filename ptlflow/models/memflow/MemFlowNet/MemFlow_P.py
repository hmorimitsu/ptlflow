import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from ..encoders import twins_svt_large
from .cnn import BasicEncoder
from .corr import CorrBlock
from ..utils import coords_grid
from .gma import Attention
from .sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder
from .sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_predict
from .memory_util import *

try:
    from flash_attn import flash_attn_func
except:
    pass


class MemFlowNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = 128
        self.context_dim = 128

        # feature network, context network, and update block
        if cfg.cnet == "twins":
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == "basicencoder":
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn="batch")

        if cfg.fnet == "twins":
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == "basicencoder":
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance")

        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
        elif self.cfg.gma == "GMA-SK":
            print("[Using GMA-SK]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.cfg, hidden_dim=128
            )
        elif self.cfg.gma == "GMA-SK2":
            print("[Using GMA-SK2]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_predict(
                args=self.cfg, hidden_dim=128
            )

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        self.att = Attention(
            args=self.cfg,
            dim=self.context_dim,
            heads=1,
            max_pos_size=160,
            dim_head=self.context_dim,
        )
        self.train_avg_length = cfg.train_avg_length

        self.motion_prompt = nn.Parameter(torch.randn(1, 128, 1, 1))

    def encode_features(self, frame, flow_init=None):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        fmaps = self.fnet(frame)
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        return coords0, coords1, fmaps

    def encode_context(self, frame):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        # shape is b*c*h*w
        inp = self.cnet(frame)
        inp = torch.relu(inp)
        query, key = self.att.to_qk(inp).chunk(2, dim=1)
        # query = query * self.att.scale
        if need_reshape:
            # B*C*T*H*W
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            inp = inp.view(b, t, *inp.shape[-3:])

        return query, key, inp

    def get_motion_feature(self, flow, coords1, fmaps):
        corr_fn = CorrBlock(
            fmaps[:, 0, ...],
            fmaps[:, 1, ...],
            num_levels=self.cfg.corr_levels,
            radius=self.cfg.corr_radius,
        )
        corr = corr_fn(coords1 + flow)  # index correlation volume
        _, current_value = self.update_block.get_motion_and_value(flow, corr)
        return current_value

    def predict_flow(
        self, inp, query, ref_keys, value, forward_warp_flow=None, test_mode=False
    ):
        B, _, H, W = inp.shape
        if ref_keys is not None and value is not None:
            query = query.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            ref_keys = ref_keys.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            # get global motion
            # B, L, N, C
            value = value.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            scale = self.att.scale * math.log(ref_keys.shape[1], self.train_avg_length)
            hidden_states = flash_attn_func(
                query, ref_keys, value, dropout_p=0.0, softmax_scale=scale, causal=False
            )
            hidden_states = (
                hidden_states.squeeze(2).permute(0, 2, 1).reshape(B, -1, H, W)
            )

            motion_features_global = (
                self.motion_prompt.repeat(B, 1, H, W)
                + self.update_block.aggregator.gamma * hidden_states
            )
        else:
            motion_features_global = self.motion_prompt.repeat(B, 1, H, W)
        if "concat_flow" in self.cfg and self.cfg.concat_flow:
            motion_features_global = torch.cat(
                [motion_features_global, forward_warp_flow], dim=1
            )
        _, up_mask, delta_flow = self.update_block(inp, motion_features_global)
        if "concat_flow" in self.cfg and self.cfg.concat_flow:
            delta_flow = delta_flow + forward_warp_flow
        # upsample predictions
        flow_up = self.upsample_flow(delta_flow, up_mask)

        return flow_up

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device, dtype=img.dtype)

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
