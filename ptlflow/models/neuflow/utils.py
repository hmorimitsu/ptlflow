import torch
import torch.nn.functional as F


def normalize_img(img, mean, std):
    return (img / 255.0 - mean) / std


def coords_grid(b, h, w, dtype, device):
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )  # [H, W]

    stacks = [xs, ys]

    grid = torch.stack(stacks, dim=0)  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    return grid


def bilinear_sample(img, sample_coords):
    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return img


def flow_warp(feature, flow):
    b, c, h, w = feature.size()

    grid = (
        coords_grid(b, h, w, dtype=flow.dtype, device=flow.device) + flow
    )  # [B, 2, H, W]

    return bilinear_sample(feature, grid)
