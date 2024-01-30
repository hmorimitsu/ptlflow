import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(
    feature0,
    feature1,
    pred_bidir_flow=False,
):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (
        c**0.5
    )  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(
        b, h, w, dtype=correlation.dtype, device=correlation.device
    )  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat(
            (correlation, correlation.permute(0, 2, 1)), dim=0
        )  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = (
        torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob


def local_correlation_softmax(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(
        b, h, w, dtype=feature0.dtype, device=feature0.device
    )  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius,
        local_radius,
        -local_radius,
        local_radius,
        local_h,
        local_w,
        dtype=feature0.dtype,
        device=feature0.device,
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (
        sample_coords[:, :, :, 0] < w
    )  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (
        sample_coords[:, :, :, 1] < h
    )  # [B, H*W, (2R+1)^2]

    valid = (
        valid_x & valid_y
    )  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(
        feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True
    ).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (
        c**0.5
    )  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    if feature0.dtype == torch.float16:
        val = -1e4
    else:
        val = -1e9
    corr[~valid] = val

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = (
        torch.matmul(prob.unsqueeze(-2), sample_coords_softmax)
        .squeeze(-2)
        .view(b, h, w, 2)
        .permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob


def local_correlation_with_flow(
    feature0,
    feature1,
    flow,
    local_radius,
    padding_mode="zeros",
    dilation=1,
):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(
        b, h, w, dtype=feature0.dtype, device=feature0.device
    )  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius,
        local_radius,
        -local_radius,
        local_radius,
        local_h,
        local_w,
        dtype=feature0.dtype,
        device=feature0.device,
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = (
        coords.unsqueeze(-2) + window_grid * dilation
    )  # [B, H*W, (2R+1)^2, 2]

    # flow can be zero when using features after transformer
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(b, 2, -1).permute(0, 2, 1).unsqueeze(
            -2
        )  # [B, H*W, (2R+1)^2, 2]
    else:
        assert flow == 0.0

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(
        feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True
    ).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (
        c**0.5
    )  # [B, H*W, (2R+1)^2]

    corr = (
        corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
    )  # [B, (2R+1)^2, H, W]

    return corr


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, dtype=depth.dtype, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature
