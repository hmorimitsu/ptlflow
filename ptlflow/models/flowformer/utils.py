import math

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, pad_mode, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        self.pad_mode = pad_mode
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        elif mode == "kitti432":
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == "kitti400":
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == "kitti376":
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode=self.pad_mode, value=0.0) for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def forward_interpolate(flow):
    dtype = flow.dtype
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).to(dtype=dtype)


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.to(dtype=coords.dtype)

    return img


def indexing(img, coords, mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    """
        TODO: directly indexing features instead of sampling
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode="nearest")

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.to(dtype=coords.dtype)

    return img


def coords_grid(batch, ht, wd, dtype, device):
    coords = torch.meshgrid(
        torch.arange(ht, dtype=dtype, device=device),
        torch.arange(wd, dtype=dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0)
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def compute_grid_indices(image_shape, patch_size, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(
    hws,
    image_shape,
    patch_size,
    sigma=1.0,
    wtype="gaussian",
    device: torch.device = torch.device("cpu"),
):
    patch_num = len(hws)
    h, w = torch.meshgrid(
        torch.arange(patch_size[0], device=device),
        torch.arange(patch_size[1], device=device),
        indexing="ij",
    )
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h**2 + w**2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape, device=device)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h : h + patch_size[0], w : w + patch_size[1]] = weights_hw
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(
            weights[:, idx : idx + 1, h : h + patch_size[0], w : w + patch_size[1]]
        )

    return patch_weights
