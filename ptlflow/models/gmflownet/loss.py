import torch
import numpy as np


from typing import Optional
def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.
    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.
    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])
        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False,).to(flow.device)
    gridY = torch.tensor(gridY, requires_grad=False,).to(flow.device)
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=False)

    return imgOut


@torch.no_grad()
def compute_supervision_coarse(flow, occlusions, scale: int):
    N, _, H, W = flow.shape
    Hc, Wc = int(np.ceil(H / scale)), int(np.ceil(W / scale))

    occlusions_c = occlusions[:, :, ::scale, ::scale]
    flow_c = flow[:, :, ::scale, ::scale] / scale
    occlusions_c = occlusions_c.reshape(N, Hc * Wc)

    grid_c = create_meshgrid(Hc, Wc, False, device=flow.device).reshape(1, Hc * Wc, 2).repeat(N, 1, 1)
    warp_c = grid_c + flow_c.permute(0, 2, 3, 1).reshape(N, Hc * Wc, 2)
    warp_c = warp_c.round().long()

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    occlusions_c[out_bound_mask(warp_c, Wc, Hc)] = 1
    warp_c = warp_c[..., 0] + warp_c[..., 1] * Wc

    b_ids, i_ids = torch.split(torch.nonzero(occlusions_c == 0), 1, dim=1)
    conf_matrix_gt = torch.zeros(N, Hc * Wc, Hc * Wc, device=flow.device)
    j_ids = warp_c[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    return conf_matrix_gt


def compute_coarse_loss(conf, conf_gt, cfg):
    c_pos_w, c_neg_w = cfg.POS_WEIGHT, cfg.NEG_WEIGHT
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0

    if cfg.COARSE_TYPE == 'cross_entropy':
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        loss_pos = -torch.log(conf[pos_mask])
        loss_neg = -torch.log(1 - conf[neg_mask])

        return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
    elif cfg.COARSE_TYPE == 'focal':
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = cfg.FOCAL_ALPHA
        gamma = cfg.FOCAL_GAMMA
        loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
        loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
        return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
    else:
        raise ValueError('Unknown coarse loss: {type}'.format(type=cfg.COARSE_TYPE))


def compute_fine_loss(kflow, kflow_gt, cfg):
    fine_correct_thr = cfg.WINDOW_SIZE // 2 * 2
    error = (kflow - kflow_gt).abs()
    correct = torch.max(error, dim=1)[0] < fine_correct_thr
    rate = torch.sum(correct).float() / correct.shape[0]
    num = correct.shape[0]
    return error[correct].mean(), rate.item(), num


def compute_flow_loss(flow, flow_gt):
    loss = (flow - flow_gt).abs().mean()
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return loss, metrics
