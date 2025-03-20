import torch
from torch import nn, einsum
from einops import rearrange
import math


class RelPosEmb(nn.Module):
    def __init__(self, max_pos_size, dim_head):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(
            max_pos_size
        ).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer("rel_ind", rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, "(x u) d -> x u () d", x=h)
        width_emb = rearrange(width_emb, "(y v) d -> y () v d", y=w)

        height_score = einsum("b h x y d, x u v d -> b h x y u v", q, height_emb)
        width_score = einsum("b h x y d, y u v d -> b h x y u v", q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        # self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=heads), (q, k))
        q = self.scale * q
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
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class TemporalAggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.temporal_gamma = nn.Parameter(torch.zeros(1))

        self.temporal_project = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, temporal_attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", temporal_attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = self.temporal_project(out)

        out = fmap + self.temporal_gamma * out
        return out


class SpatioTemporalAggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.temporal_gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

        if args.no_temporal_project:
            self.temporal_project = None
        else:
            self.temporal_project = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, attn, temporal_attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        # temporal
        if temporal_attn is not None:
            temporal_out = rearrange(out, "b (h d) x y -> b h (x y) d", h=heads)
            temporal_out = einsum(
                "b h i j, b h j d -> b h i d", temporal_attn, temporal_out
            )
            temporal_out = rearrange(
                temporal_out, "b h (x y) d -> b (h d) x y", x=h, y=w
            )
            if self.temporal_project is not None:
                temporal_out = self.temporal_project(temporal_out)
            out = out + self.temporal_gamma * temporal_out
        return out


class TemporalAttention(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        T = args.T - 1
        self.to_qk = nn.Conv2d(dim * T, dim * T * 2, 1, bias=False)
        self.scale = (dim * T) ** -0.5

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> b (t c) h w")
        q, k = self.to_qk(x).chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, "b c h w -> b h w c"), (q, k))
        q = self.scale * q

        out = einsum("b h w c, b u v c -> b h w u v", q, k)
        out = rearrange(out, "b h w u v -> b (h w) (u v)").softmax(dim=-1)

        return out.unsqueeze(1)


class SpatioTemporalAggregate2(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        # self.temporal_gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, temporal_attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape
        T = self.args.T - 1
        B = b // T

        v = self.to_v(fmap)
        v = rearrange(v, "(B T) C H W -> B C H (T W)", T=T, W=w)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=T * w)
        out = rearrange(out, "B C H (T W) -> (B T) C H W", T=T, W=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class TMMAggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=1,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.temporal_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn, temporal_attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        out = fmap + self.gamma * out

        # temporal
        T = self.args.T - 1
        B = b // T
        out = rearrange(out, "(b t) c h w -> b (t c) h w", b=B, t=T)
        temporal_out = rearrange(out, "b (h d) x y -> b h (x y) d", h=heads)
        temporal_out = einsum(
            "b h i j, b h j d -> b h i d", temporal_attn, temporal_out
        )
        temporal_out = rearrange(temporal_out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = out + self.temporal_gamma * temporal_out
        out = rearrange(out, "b (t c) h w -> (b t) c h w", b=B, t=T)
        return out


def kaiming_init(
    module, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


if __name__ == "__main__":
    att = Attention(dim=128, heads=1)
    fmap = torch.randn(2, 128, 40, 90)
    out = att(fmap)

    print(out.shape)
