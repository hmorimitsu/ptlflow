# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import torch

from torch import Tensor
from torch import nn

import torch.nn.functional as F

logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


USE_FLASH_ATTENTION3 = True
try:
    from flash_attn_interface import flash_attn_func

    FA3_AVAILABLE = True
    warnings.warn("flash attention 3 is available (ViT)")
except ImportError:
    FA3_AVAILABLE = False
    warnings.warn("flash attention 3 is not available (ViT)")


USE_PYTORCH_ATTN = True  # flash attention 2


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if USE_PYTORCH_ATTN and self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, y=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE and not (USE_FLASH_ATTENTION3 and FA3_AVAILABLE):
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        if y is not None:

            B, Nq, C = x.shape
            context = x if y is None else y
            Nk = context.shape[1]

            # Lazy conversion from qkv to q_proj and kv_proj
            if not hasattr(self, "q_proj") or self.q_proj is None:
                self.q_proj, self.kv_proj = convert_qkv_to_q_and_kv_proj(self.qkv)
                # del self.qkv  # Optional: free memory

            # Project q, k, v
            q = self.q_proj(x).reshape(
                B, Nq, self.num_heads, C // self.num_heads
            )  # [B, N, H, d]
            kv = self.kv_proj(context).reshape(
                B, Nk, 2, self.num_heads, C // self.num_heads
            )  # [B, Nk, 2, H, d]
            k, v = kv.unbind(dim=2)  # [B, h, Nk, d]
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            # FlashAttention3 or memory-efficient fallback
            if USE_FLASH_ATTENTION3 and FA3_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError("attn_bias is not supported in FA3")
                out = flash_attn_func(q, k, v)[0]  # [B, h, Nq, d]
            else:
                out = memory_efficient_attention(
                    q, k, v, attn_bias=attn_bias
                )  # [B, h, Nq, d]

            # Merge heads
            out = out.reshape(B, Nq, C)  # [B, Nq, C]
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = torch.unbind(qkv, 2)

            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            if USE_FLASH_ATTENTION3 and FA3_AVAILABLE:
                if attn_bias is not None:
                    raise AssertionError("attn_bias is not supported in FA3")
                x = flash_attn_func(q, k, v)[0]
            else:
                x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x


def convert_qkv_to_q_and_kv_proj(qkv_layer: nn.Linear):
    """
    Convert a self-attention qkv projection layer (dim -> 3*dim) into
    separate q_proj (dim -> dim) and kv_proj (dim -> 2*dim) layers.

    Returns:
        q_proj (nn.Linear): projection for query
        kv_proj (nn.Linear): projection for key and value
    """
    assert isinstance(qkv_layer, nn.Linear), "Expected nn.Linear for qkv_layer"
    in_features = qkv_layer.in_features
    out_features = qkv_layer.out_features
    assert out_features % 3 == 0, "Output features must be divisible by 3"

    dim = out_features // 3
    device = qkv_layer.weight.device
    dtype = qkv_layer.weight.dtype

    q_proj = nn.Linear(in_features, dim, bias=qkv_layer.bias is not None).to(
        device=device, dtype=dtype
    )
    kv_proj = nn.Linear(in_features, dim * 2, bias=qkv_layer.bias is not None).to(
        device=device, dtype=dtype
    )

    # Split weights and biases
    q_weight, k_weight, v_weight = qkv_layer.weight.chunk(3, dim=0)
    q_proj.weight.data.copy_(q_weight)
    kv_proj.weight.data.copy_(torch.cat([k_weight, v_weight], dim=0))

    if qkv_layer.bias is not None:
        q_bias, k_bias, v_bias = qkv_layer.bias.chunk(3, dim=0)
        q_proj.bias.data.copy_(q_bias)
        kv_proj.bias.data.copy_(torch.cat([k_bias, v_bias], dim=0))

    return q_proj, kv_proj
