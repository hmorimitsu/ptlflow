import torch
import torch.nn as nn
import timm
from einops import rearrange
from torch import nn
from timm.layers import to_2tuple


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, T, C, H, W
        B, T, _, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b (t h w) c", b=B, t=T)
        x = self.norm(x)
        out_size = ((H * T) // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins_CSC(nn.Module):
    def __init__(self, pretrained_ckpt=None):
        super().__init__()
        self.svt = timm.create_model("twins_svt_large", pretrained=False)
        self.svt.patch_embeds = nn.ModuleList()
        embed_dims = [128, 256, 512, 1024]

        self.svt.patch_embeds.append(PatchEmbed(4, 3, embed_dims[0]))  # 4倍下采样
        self.svt.patch_embeds.append(
            PatchEmbed(2, embed_dims[0], embed_dims[1])
        )  # 2倍下采样
        self.svt.patch_embeds.append(
            PatchEmbed(2, embed_dims[1], embed_dims[2])
        )  # 2倍下采样
        self.svt.patch_embeds.append(
            PatchEmbed(2, embed_dims[2], embed_dims[3])
        )  # 2倍下采样

        if pretrained_ckpt is not None:
            self.svt.load_state_dict(torch.load(pretrained_ckpt), strict=True)

        del self.svt.head
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]

    def forward(self, x):
        layer = 2
        # if input is list, combine batch dimension
        B, T, C, H, W = x.shape
        ratios = [4, 2]

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):

            x, size = embed(x)  # x: (B, T*h*w, C)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)

            # self.svt.depths实质为2
            H, W = H // ratios[i], W // ratios[i]
            x = rearrange(x, "b (t h w) c -> b t c h w", t=T, h=H, w=W)

            # x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x
