import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate

# from torch_scatter import scatter_softmax, scatter_add


class InputPadder:
    """Pads images such that dimensions are divisible by 32"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 32) + 1) * 32 - self.ht) % 32
        pad_wd = (((self.wd // 32) + 1) * 32 - self.wd) % 32
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


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


def bilinear_sample(
    img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False
):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(
        img, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )

    if return_mask:
        mask = (
            (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        )  # [B, H, W]

        return img, mask

    return img


def coords_grid(batch, ht, wd, dtype, device):
    coords = torch.meshgrid(
        torch.arange(ht, dtype=dtype, device=device),
        torch.arange(wd, dtype=dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack(coords[::-1], dim=0)
    return coords[None].repeat(batch, 1, 1, 1)


def flow_warp(image, flow, mask=False, padding_mode="zeros"):
    b, c, h, w = image.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(image, grid, padding_mode=padding_mode, return_mask=mask)


def coords_grid_y_first(batch, ht, wd):
    """Place y grid before x grid"""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords, dim=0).int()
    return coords[None].expand(batch, -1, -1, -1)


def soft_argmax(corr_me, B, H1, W1):
    # Implement soft argmin
    coords, feats = corr_me.decomposed_coordinates_and_features

    # Computing soft argmin
    flow_pred = torch.zeros(B, 2, H1, W1).to(corr_me.device)
    for batch, (coord, feat) in enumerate(zip(coords, feats)):
        coord_img_1 = coord[:, :2].to(corr_me.device)
        coord_img_2 = coord[:, 2:].to(corr_me.device)
        # relative positions (flow hypotheses)
        rel_pos = coord_img_2 - coord_img_1
        # augmented indices
        aug_coord_img_1 = (coord_img_1[:, 0:1] * W1 + coord_img_1[:, 1:2]).long()
        # run softmax on the score
        weight = scatter_softmax(feat, aug_coord_img_1, dim=0)
        rel_pos_weighted = weight * rel_pos
        out = scatter_add(rel_pos_weighted, aug_coord_img_1, dim=0)
        # Need to permute (y, x) to (x, y) for flow
        flow_pred[batch] = out[:, [1, 0]].view(H1, W1, 2).permute(2, 0, 1)
    return flow_pred


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow4(flow, mode="bilinear"):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow2(flow, mode="bilinear"):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def downflow8(flow, mode="bilinear"):
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8


def downflow4(flow, mode="bilinear"):
    new_size = (flow.shape[2] // 4, flow.shape[3] // 4)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 4


def compute_interpolation_weights(yx_warped):
    # yx_warped: [N, 2]
    y_warped = yx_warped[:, 0]
    x_warped = yx_warped[:, 1]

    # elementwise operations below
    y_f = torch.floor(y_warped)
    y_c = y_f + 1
    x_f = torch.floor(x_warped)
    x_c = x_f + 1

    w0 = (y_c - y_warped) * (x_c - x_warped)
    w1 = (y_warped - y_f) * (x_c - x_warped)
    w2 = (y_c - y_warped) * (x_warped - x_f)
    w3 = (y_warped - y_f) * (x_warped - x_f)

    weights = [w0, w1, w2, w3]
    indices = [
        torch.stack([y_f, x_f], dim=1),
        torch.stack([y_c, x_f], dim=1),
        torch.stack([y_f, x_c], dim=1),
        torch.stack([y_c, x_c], dim=1),
    ]
    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=2)
    # indices = torch.cat(indices, dim=0)  # [4*N, 2]

    return weights, indices


# weights, indices = compute_interpolation_weights(xy_warped, b, h_i, w_i)


def compute_inverse_interpolation_img(weights, indices, img, b, h_i, w_i):
    """
    weights: [b, h*w]
    indices: [b, h*w]
    img: [b, h*w, a, b, c, ...]
    """
    w0, w1, w2, w3 = weights
    ff_idx, cf_idx, fc_idx, cc_idx = indices

    k = len(img.size()) - len(w0.size())
    img_0 = w0[(...,) + (None,) * k] * img
    img_1 = w1[(...,) + (None,) * k] * img
    img_2 = w2[(...,) + (None,) * k] * img
    img_3 = w3[(...,) + (None,) * k] * img

    img_out = torch.zeros(b, h_i * w_i, *img.shape[2:]).type_as(img)

    ff_idx = torch.clamp(ff_idx, min=0, max=h_i * w_i - 1)
    cf_idx = torch.clamp(cf_idx, min=0, max=h_i * w_i - 1)
    fc_idx = torch.clamp(fc_idx, min=0, max=h_i * w_i - 1)
    cc_idx = torch.clamp(cc_idx, min=0, max=h_i * w_i - 1)

    img_out.scatter_add_(1, ff_idx[(...,) + (None,) * k].expand_as(img_0), img_0)
    img_out.scatter_add_(1, cf_idx[(...,) + (None,) * k].expand_as(img_1), img_1)
    img_out.scatter_add_(1, fc_idx[(...,) + (None,) * k].expand_as(img_2), img_2)
    img_out.scatter_add_(1, cc_idx[(...,) + (None,) * k].expand_as(img_3), img_3)

    return img_out  # [b, h_i*w_i, ...]


def compute_grid_indices(image_shape, patch_size, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    # unique
    hs = np.unique(hs)
    # ws.append(32)
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
