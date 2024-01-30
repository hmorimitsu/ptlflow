import numpy as np
import torch
import torch.nn.functional as F
from .position import PositionEmbeddingSine


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (
            ((self.ht // padding_factor) + 1) * padding_factor - self.ht
        ) % padding_factor
        pad_wd = (
            ((self.wd // padding_factor) + 1) * padding_factor - self.wd
        ) % padding_factor
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


def bilinear_sampler(img, coords, mode="bilinear", mask=False, padding_mode="zeros"):
    """Wrapper for grid_sample, uses pixel coordinates"""
    if coords.size(-1) != 2:  # [B, 2, H, W] -> [B, H, W, 2]
        coords = coords.permute(0, 2, 3, 1)

    H, W = img.shape[-2:]
    # H = height if height is not None else img.shape[-2]
    # W = width if width is not None else img.shape[-1]

    xgrid, ygrid = coords.split([1, 1], dim=-1)

    # To handle H or W equals to 1 by explicitly defining height and width
    if H == 1:
        assert ygrid.abs().max() < 1e-8
        H = 10
    if W == 1:
        assert xgrid.abs().max() < 1e-8
        W = 10

    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(
        img, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.squeeze(-1).to(dtype=coords.dtype)

    return img


def coords_grid(batch, ht, wd, normalize=False, dtype=None, device=None):
    if normalize:  # [-1, 1]
        coords = torch.meshgrid(
            2 * torch.arange(ht, dtype=dtype, device=device) / (ht - 1) - 1,
            2 * torch.arange(wd, dtype=dtype, device=device) / (wd - 1) - 1,
            indexing="ij",
        )
    else:
        coords = torch.meshgrid(
            torch.arange(ht, dtype=dtype, device=device),
            torch.arange(wd, dtype=dtype, device=device),
            indexing="ij",
        )
    coords = torch.stack(coords[::-1], dim=0)
    return coords[None].repeat(batch, 1, 1, 1)  # [B, 2, H, W]


def coords_grid_np(h, w):  # used for accumulating high speed sintel flow testdata
    coords = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    coords = np.stack(coords[::-1], axis=-1)  # [H, W, 2]

    return coords


def compute_out_of_boundary_mask(flow, downsample_factor=None):
    # flow: [B, 2, H, W]
    assert flow.dim() == 4 and flow.size(1) == 2
    b, _, h, w = flow.shape
    init_coords = coords_grid(b, h, w).to(flow.device)
    corres = init_coords + flow  # [B, 2, H, W]

    if downsample_factor is not None:
        assert w % downsample_factor == 0 and h % downsample_factor == 0
        # the actual max disp can predict is in the downsampled feature resolution, then upsample
        max_w = (w // downsample_factor - 1) * downsample_factor
        max_h = (h // downsample_factor - 1) * downsample_factor
        # print('max_w: %d, max_h: %d' % (max_w, max_h))
    else:
        max_w = w - 1
        max_h = h - 1

    valid_mask = (
        (corres[:, 0] >= 0)
        & (corres[:, 0] <= max_w)
        & (corres[:, 1] >= 0)
        & (corres[:, 1] <= max_h)
    )

    # in case very large flow
    flow_mask = (flow[:, 0].abs() <= max_w) & (flow[:, 1].abs() <= max_h)

    valid_mask = valid_mask & flow_mask

    return valid_mask  # [B, H, W]


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    # grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def flow_warp(feature, flow, mask=False, padding_mode="zeros"):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sampler(feature, grid, mask=mask, padding_mode=padding_mode)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def bilinear_upflow(flow, scale_factor=8):
    assert flow.size(1) == 2
    flow = (
        F.interpolate(
            flow, scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        * scale_factor
    )

    return flow


def upsample_flow(flow, img):
    if flow.size(-1) != img.size(-1):
        scale_factor = img.size(-1) / flow.size(-1)
        flow = (
            F.interpolate(
                flow, size=img.size()[-2:], mode="bilinear", align_corners=True
            )
            * scale_factor
        )
    return flow


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def generate_window_grid(
    h_min, h_max, w_min, w_max, len_h, len_w, dtype=None, device=None
):
    assert device is not None

    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, dtype=dtype, device=device),
            torch.linspace(h_min, h_max, len_h, dtype=dtype, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack((x, y), -1).transpose(0, 1)  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).to(
        dtype=coords.dtype, device=coords.device
    )
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = (
        torch.tensor([0.485, 0.456, 0.406])
        .view(1, 3, 1, 1)
        .to(dtype=img1.dtype, device=img1.device)
    )
    std = (
        torch.tensor([0.229, 0.224, 0.225])
        .view(1, 3, 1, 1)
        .to(dtype=img1.dtype, device=img1.device)
    )
    img0 = (img0 - mean) / std
    img1 = (img1 - mean) / std

    return img0, img1


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )  # [B, C, H, W]

    return merge


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def upsample_flow_with_mask(flow, up_mask, upsample_factor):
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(
        b, 1, 9, upsample_factor, upsample_factor, h, w
    )  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(
        b, flow_channel, upsample_factor * h, upsample_factor * w
    )  # [B, 2, K*H, K*W]

    return up_flow


def split_feature_1d(
    feature,
    num_splits=2,
):
    # feature: [B, W, C]
    b, w, c = feature.size()
    assert w % num_splits == 0

    b_new = b * num_splits
    w_new = w // num_splits

    feature = feature.view(b, num_splits, w // num_splits, c).view(
        b_new, w_new, c
    )  # [B*K, W/K, C]

    return feature


def merge_splits_1d(
    splits,
    h,
    num_splits=2,
):
    b, w, c = splits.size()
    new_b = b // num_splits // h

    splits = splits.view(new_b, h, num_splits, w, c)
    merge = splits.view(new_b, h, num_splits * w, c)  # [B, H, W, C]

    return merge


def window_partition_1d(x, window_size_w):
    """
    Args:
        x: (B, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, W, C = x.shape
    x = x.view(B, W // window_size_w, window_size_w, C).view(-1, window_size_w, C)
    return x


def generate_shift_window_attn_mask_1d(
    input_w, window_size_w, shift_size_w, device=torch.device("cuda")
):
    # calculate attention mask for SW-MSA
    img_mask = torch.zeros((1, input_w, 1)).to(device)  # 1 W 1
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for w in w_slices:
        img_mask[:, w, :] = cnt
        cnt += 1

    mask_windows = window_partition_1d(img_mask, window_size_w)  # nW, window_size, 1
    mask_windows = mask_windows.view(-1, window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
        2
    )  # nW, window_size, window_size
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask
