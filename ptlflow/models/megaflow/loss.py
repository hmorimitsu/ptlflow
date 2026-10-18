import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
import numpy as np

from megaflow.utils.basic import gridcloud2d, reduce_masked_mean
from megaflow.model.geometry import bilinear_sample_video


def pointflow_loss_func(
    flow_preds, trajs_g, valids, vis_g, all_flow_e, seq_len=12, gamma=0.8
):
    B, T, _, H, W = all_flow_e.shape
    device = all_flow_e.device
    B, T1, N, D = trajs_g.shape
    assert T == T1

    S = seq_len
    vis_g = vis_g.float()

    # calculate final tracking traj for all frams
    grid_xy = gridcloud2d(1, H, W, norm=False, device=device).float()  # 1, H*W, 2
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)  # 1,1,2,H,W
    traj_maps_e = all_flow_e + grid_xy  # B,T,2,H,W

    xy0 = trajs_g[:, 0]  # B,N,2
    xy0[:, :, 0] = xy0[:, :, 0].clamp(0, W - 1)
    xy0[:, :, 1] = xy0[:, :, 1].clamp(0, H - 1)
    xy0_ = xy0.reshape(B, 1, N, 2).repeat(1, T, 1, 1).reshape(B * T, N, 1, 2)
    trajs_e_ = bilinear_sample_video(
        traj_maps_e.reshape(B * T, 2, H, W), xy0_
    )  # B*T,2,N,1
    trajs_e = trajs_e_.reshape(B, T, 2, N).permute(0, 1, 3, 2)  # B,T,N,2

    xy0_ = xy0.reshape(B, 1, N, 2).repeat(1, S, 1, 1).reshape(B * S, N, 1, 2)

    coord_predictions = []
    assert S <= T and B == 1

    for fpl in flow_preds:
        cps = []
        for fp in fpl:
            traj_map = fp + grid_xy  # B,S,2,H,W
            traj_map_ = traj_map.reshape(B * S, 2, H, W)
            traj_e_ = bilinear_sample_video(traj_map_, xy0_)  # B*S,2,N,1
            traj_e = traj_e_.reshape(B, S, 2, N).permute(0, 1, 3, 2)  # B,S,N,2
            cps.append(traj_e)
        coord_predictions.append(cps)

    # visconf is upset when i bilinearly sample. says the data is outside [0,1]
    # so here we do NN sampling
    assert B == 1
    x0, y0 = xy0[0, :, 0], xy0[0, :, 1]  # N
    x0 = torch.clamp(x0, 0, W - 1).round().long()
    y0 = torch.clamp(y0, 0, H - 1).round().long()

    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    for ind in range(0, T - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        valids_gts.append(valids[:, ind : ind + S])

    seq_loss_visible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=gamma,
        loss_only_for_visible=True,
    )

    seq_loss_invisible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=invis_gts,
        gamma=gamma,
        loss_only_for_visible=True,
    )

    total_loss = seq_loss_visible.mean() * 0.05 + seq_loss_invisible.mean() * 0.01

    epe = torch.sum((trajs_e - trajs_g) ** 2, dim=-1).sqrt()
    epe = epe.view(-1)[(valids * vis_g).view(-1).bool()]

    metrics = {
        "epe": epe.mean().item(),
        "seq_loss_visible": seq_loss_visible.mean().item(),
        "seq_loss_invisible": seq_loss_invisible.mean().item(),
    }

    thrs = [1, 2, 4, 8, 16]
    sx_ = (W - 1) / 255.0
    sy_ = (H - 1) / 255.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().to(device)
    d_sum = 0.0
    for thr in thrs:
        d_ = (
            (torch.norm(trajs_e / sc_pt - trajs_g / sc_pt, dim=-1) < thr)
            .float()
            .mean()
            .item()
        )
        d_sum += d_
        metrics["d_%d" % thr] = d_
    d_avg = d_sum / len(thrs)
    metrics["d_avg"] = d_avg

    return total_loss, metrics


def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def resize_gtflow(flow_gt, valid_mask, target_h, target_w):
    """
    Resize flow_gt and mask to target resolution for loss computation.
    """
    ori_h, ori_w = flow_gt.shape[-2:]
    scale_x = target_w / ori_w
    scale_y = target_h / ori_h

    # resize flow field
    flow_gt_rs = F.interpolate(
        flow_gt, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    flow_gt_rs[:, 0] *= scale_x
    flow_gt_rs[:, 1] *= scale_y

    # resize mask
    valid_rs = (
        F.interpolate(
            valid_mask.float()[:, None], size=(target_h, target_w), mode="nearest"
        )
        >= 0.5
    )

    return flow_gt_rs, valid_rs


def flow_loss_func(
    flow_preds, flow_gt, valid, flow_init=None, gamma=0.9, max_flow=512, **kwargs
):
    """
    Compute hierarchical flow loss with optional initial prediction loss.

    Parameters
    ----------
    outputs : Dict[str, List[torch.Tensor]]
        Dictionary containing model outputs:
        - 'flow': List of predicted flow fields, each of shape (B, 2, 2, H, W)
        - 'nf': List of normalized flow losses, each of shape (B, 2, 2, H, W)
    flow_gts : torch.Tensor
        Ground truth flow fields of shape (B, 2, 2, H, W)
    valids : torch.Tensor
        Validity masks of shape (B, 2, H, W)
    gamma : float, optional
        Weight decay factor for sequence loss, by default 0.8
    max_flow : float, optional
        Maximum flow magnitude threshold, by default MAX_FLOW

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # valid pixels and exclude extremely large displacements
    mag = torch.sqrt(torch.sum(flow_gt**2, dim=2))  # [B,T,H,W]
    valid_mask = (valid >= 0.5) & (mag < max_flow)

    # Hierarchical predictions loss
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid_mask[:, :, None] * i_loss).mean()

    # Optional initial prediction loss
    if flow_init is not None:
        # Downsample GT to match flow_init resolution
        flow_gt_ds, valid_ds = resize_gtflow(
            flow_gt.reshape(-1, *flow_gt.shape[2:]),
            valid_mask.reshape(-1, *valid_mask.shape[2:]),
            *flow_init.shape[-2:],
        )

        i_loss = F.smooth_l1_loss(
            flow_init.reshape(-1, *flow_init.shape[2:]), flow_gt_ds, reduction="none"
        )  # [B,2,H,W]
        flow_loss += (valid_ds * i_loss).mean()

    epe = torch.sqrt(torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=2))
    epe = torch.nan_to_num(epe, nan=0.0, posinf=0.0, neginf=0.0)
    epe = epe.view(-1)[valid_mask.view(-1)]

    return flow_loss, epe.mean()
