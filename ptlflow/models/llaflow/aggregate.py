import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
import math


# The position encoding block is from https://github.com/haofeixu/gmflow
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=x.dtype)
        x_embed = mask.cumsum(2, dtype=x.dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def PatchExtra(x, size):
    B, C, H, W = x.shape
    x = F.pad(x, (size // 2, size // 2, size // 2, size // 2), "replicate")
    return F.unfold(x, [size, size], stride=1, padding=0).reshape(
        B, C, size * size, H, W
    )


def ImgShift(x, size):
    B, C, H, W = x.shape
    x = F.pad(x, (size // 2, size // 2, size // 2, size // 2), "replicate")
    return (
        F.unfold(x, [H, W], stride=1, padding=0)
        .reshape(B, C, H, W, size**2)
        .permute(4, 0, 1, 2, 3)
    )


class GlobalSimilar(nn.Module):
    def __init__(
        self,
        args,
        dim=128,
        heads=1,
    ):
        super().__init__()
        self.args = args
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qk = nn.Conv2d(dim, dim * 2, 1, bias=False)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)
        q, k = map(lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=heads), (q, k))
        q = self.scale * q
        sim = einsum("b h x y d, b h u v d -> b h x y u v", q, k)
        sim = rearrange(sim, "b h x y u v -> b h (x y) (u v)")
        attn = sim.softmax(dim=-1)

        return attn


class LocalSimilar(nn.Module):
    def __init__(
        self,
        args,
        dim=128,
        heads=1,
        size=5,
    ):
        super().__init__()
        self.args = args
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.size = size

        self.to_qk = nn.Conv2d(dim, dim * 2, 1, bias=False)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        # pos_enc = PositionEmbeddingSine(num_pos_feats=c//2)
        # fmap = fmap + pos_enc(fmap)

        q, k = self.to_qk(fmap).chunk(2, dim=1)  # B, C, H, W
        k = PatchExtra(k, self.size)  # B, C, L, H, W

        q = rearrange(q, "b (h d) x y -> b h x y d", h=heads)
        k = rearrange(k, "b (h d) l x y -> b h x y d l", h=heads)
        q = self.scale * q

        sim = einsum("b h x y d, b h x y d l -> b h x y l", q, k)
        sim = rearrange(sim, "b h x y l -> b h (x y) l")
        attn = sim.softmax(dim=-1)

        return attn


class GSA(nn.Module):
    def __init__(
        self,
        args,
        dim=128,
        heads=1,
    ):
        super().__init__()
        self.args = args
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return fmap + self.gamma * out


class ShiftLSA(nn.Module):
    def __init__(
        self,
        args,
        dim=128,
        heads=1,
        size=5,
    ):
        super().__init__()
        self.args = args
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.size = size

        self.to_f1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_f2 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, attn, fmap, fmap2):
        heads, b, c, h, w = self.heads, *fmap.shape

        # pos_enc = PositionEmbeddingSine(num_pos_feats=c//2)
        # position = pos_enc(fmap)
        # fmap = fmap + position
        # fmap2 = fmap2 + position

        f1 = self.to_f1(fmap)
        f1s = PatchExtra(f1, self.size)  # b c l h w
        f1s = rearrange(f1s, "b (h d) l x y -> b h (x y) d l", h=heads)
        f1s = einsum("b h n l, b h n d l -> b h n d l", attn, f1s)

        f2 = self.to_f2(fmap2)
        f2s = ImgShift(f2, self.size)
        f2s = rearrange(f2s, "l b (h d) x y -> b h (x y) d l", h=heads)
        corr = einsum("b h n d l, b h m d l -> b h n m", f1s, f2s)
        corr = rearrange(corr, "b h n m -> b n h m").view(b, h, w, heads, h, w)

        return corr / torch.sqrt(torch.tensor(self.dim // self.heads))


class LSA(nn.Module):
    def __init__(self, args, dim=128, heads=1, size=5):
        super().__init__()
        self.args = args
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.size = size
        # self.bias = nn.Parameter(torch.ones(5,5))

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        # pos_enc = PositionEmbeddingSine(num_pos_feats=c//2)
        # position = pos_enc(fmap)
        # fmap = fmap + position

        v = self.to_v(fmap)
        v = PatchExtra(v, self.size)  # b c l h w
        v = rearrange(v, "b (h d) l x y -> b h (x y) d l", h=heads)
        out = einsum("b h n l, b h n d l -> b h n d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        out = fmap + self.gamma * out
        return out


if __name__ == "__main__":
    att = GlobalSimilar(dim=128, heads=1)
    fmap = torch.randn(2, 128, 40, 90)
    out = att(fmap)

    print(out.shape)
