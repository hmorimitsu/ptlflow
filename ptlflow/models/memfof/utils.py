import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def load_ckpt(model, path):
    """Load checkpoint"""
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)[
        "state_dict"
    ]
    model.load_state_dict(state_dict, strict=True)


def load_ckpt_submission(model, path):
    """Load checkpoint"""
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)[
        "state_dict"
    ]
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def resize_data(img1, img2, flow, factor=1.0):
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode="area")
    img2 = F.interpolate(img2, (h, w), mode="area")
    flow = F.interpolate(flow, (h, w), mode="area") * factor
    return img1, img2, flow


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 16) + 1) * 16 - self.ht) % 16
        pad_wd = (((self.wd // 16) + 1) * 16 - self.wd) % 16
        self.mode = mode
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
                0,
                0,
            ]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht, 0, 0]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht, 0, 0]

    def pad(self, input):
        if self.mode == "downzero":
            return F.pad(input, self._pad)
        else:
            return F.pad(input, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


# Intended to be used before converting to torch.Tensor
def merge_flows(flow1, valid1, flow2, valid2, method="nearest"):
    flow1 = np.transpose(flow1, axes=[2, 0, 1])

    _, ht, wd = flow1.shape

    x1, y1 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1_f = x1 + flow1[0]
    y1_f = y1 + flow1[1]

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x1_f = x1_f.reshape(-1)
    y1_f = y1_f.reshape(-1)
    valid1 = valid1.reshape(-1)

    mask1 = (
        (valid1 > 0.5) & (x1_f >= 0) & (x1_f <= wd - 1) & (y1_f >= 0) & (y1_f <= ht - 1)
    )
    x1 = x1[mask1]
    y1 = y1[mask1]
    x1_f = x1_f[mask1]
    y1_f = y1_f[mask1]
    valid1 = valid1[mask1]

    # STEP 1: interpolate valid values
    new_valid1 = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        valid2,
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    valid1 = new_valid1.round()

    mask1 = valid1 > 0.5
    x1 = x1[mask1]
    y1 = y1[mask1]
    x1_f = x1_f[mask1]
    y1_f = y1_f[mask1]
    valid1 = valid1[mask1]

    flow2_filled = fill_invalid(flow2, valid2)

    # STEP 2: interpolate flow values
    flow_x = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        flow2_filled[:, :, 0],
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )
    flow_y = interpolate.interpn(
        (np.arange(ht), np.arange(wd)),
        flow2_filled[:, :, 1],
        (y1_f, x1_f),
        method=method,
        bounds_error=False,
        fill_value=0,
    )

    new_flow_x = np.zeros_like(flow1[0])
    new_flow_y = np.zeros_like(flow1[1])
    new_flow_x[(y1, x1)] = flow_x + x1_f - x1
    new_flow_y[(y1, x1)] = flow_y + y1_f - y1

    new_flow = np.stack([new_flow_x, new_flow_y], axis=0)

    new_valid = np.zeros_like(flow1[0])
    new_valid[(y1, x1)] = valid1

    new_flow = np.transpose(new_flow, axes=[1, 2, 0])
    return new_flow, new_valid


def fill_invalid(flow, valid):
    return fill_invalid_slow(flow, valid)


# Intended to be used before converting to torch.Tensor, slightly modification of forward_interpolate
def fill_invalid_slow(flow, valid):
    flow = np.transpose(flow, axes=[2, 0, 1])
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0.copy()
    y1 = y0.copy()

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    valid_flat = valid.reshape(-1)

    mask = valid_flat > 0.5
    x1 = x1[mask]
    y1 = y1[mask]
    dx = dx[mask]
    dy = dy[mask]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method="nearest", fill_value=0
    )

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method="nearest", fill_value=0
    )

    flow = np.stack([flow_x, flow_y], axis=0)
    flow = np.transpose(flow, axes=[1, 2, 0])
    return flow


def forward_interpolate(flow):
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
    return torch.from_numpy(flow).float()


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
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def transform(T, p):
    assert T.shape == (4, 4)
    return np.einsum("H W j, i j -> H W i", p, T[:3, :3]) + T[:3, 3]


def from_homog(x):
    return x[..., :-1] / x[..., [-1]]


def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum(
        "H W, H W j, i j -> H W i", depth1, img_1_coords, np.linalg.inv(K1)
    )
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum("H W j, i j -> H W i", cam2_coords, K2))


def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data["T0"], data["T1"], data["K0"], data["K1"])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data["T1"], data["T0"], data["K1"], data["K0"])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0

    return flow_01, flow_10


def check_cycle_consistency(flow_01, flow_10):
    flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.1 * min(H, W)).float()
    return mask[0].numpy()
