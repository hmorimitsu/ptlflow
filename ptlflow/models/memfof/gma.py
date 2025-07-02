# https://github.com/XiaoyuShi97/VideoFlow/blob/main/core/Networks/BOFNet/gma.py

import torch
import math
from torch import nn, einsum
from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

    def forward(self, fmap):
        heads, _, _, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=heads), (q, k))

        # Small change based on MemFlow Paper
        q = self.scale * q * math.log(h * w, 3)

        sim = einsum("b h x y d, b h u v d -> b h x y u v", q, k)

        sim = rearrange(sim, "b h x y u v -> b h (x y) (u v)")
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.heads = heads

        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, _, _, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out
