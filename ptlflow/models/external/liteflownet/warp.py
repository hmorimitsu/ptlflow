# Code taken from IRR: https://github.com/visinf/irr
# Licensed under the Apache 2.0 license (see LICENSE_IRR).

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.requires_grad_(False).to(dtype=x.dtype, device=x.device)
    return grids_cuda


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
        x_warp = F.grid_sample(x, grid, align_corners=True)

        mask = torch.ones(x.size(), requires_grad=False).to(dtype=x.dtype, device=x.device)
        mask = F.grid_sample(mask, grid, align_corners=True)
        mask = (mask >= 1.0).float()

        return x_warp * mask
