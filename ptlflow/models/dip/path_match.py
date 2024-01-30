import torch
import torch.nn.functional as F
from .utils import coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class PathMatch:
    def __init__(self, fmap1, fmap2):
        self.map1 = fmap1
        self.map2 = fmap2
        self.N, self.C, self.H, self.W = fmap1.shape
        self.single_planes = self.C // 2
        self.coords = coords_grid(
            self.N, self.H, self.W, dtype=fmap1.dtype, device=fmap1.device
        )
        self.shift_map2 = self.get_inv_patch_map(fmap2)
        self.view_map1 = fmap1.view(self.N, self.C // 2, 2, 1, self.H, self.W)

    def get_inv_patch_map(self, fmap2):
        #
        r = 1
        fmap2_tl = F.pad(fmap2, [r, 0, r, 0], mode="replicate")[
            :, :, 0 : self.H, 0 : self.W
        ]
        fmap2_tr = F.pad(fmap2, [0, r, r, 0], mode="replicate")[:, :, 0 : self.H, r:]
        fmap2_dl = F.pad(fmap2, [r, 0, 0, r], mode="replicate")[:, :, r:, 0 : self.W]
        fmap2_dr = F.pad(fmap2, [0, r, 0, r], mode="replicate")[:, :, r:, r:]
        return torch.cat((fmap2, fmap2_tl, fmap2_tr, fmap2_dl, fmap2_dr), dim=1)

    def warp(self, coords, image, h, w):
        coords[:, 0, :, :] = 2.0 * coords[:, 0, :, :].clone() / max(self.W - 1, 1) - 1.0
        coords[:, 1, :, :] = 2.0 * coords[:, 1, :, :].clone() / max(self.H - 1, 1) - 1.0

        coords = coords.permute(0, 2, 3, 1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
        return output

    def search(self, flow, scale=1):
        corrs = []
        temp_coord = self.coords + flow
        map2_warp = self.warp(temp_coord, self.map2, self.H, self.W)
        padd_map2 = F.pad(map2_warp, [2, 2, 2, 2], mode="replicate")
        # #current
        # random search
        for i in range(5):
            for j in range(5):
                map2 = padd_map2[:, :, j : j + self.H, i : i + self.W]
                cost = torch.mean(self.map1 * map2, dim=1, keepdim=True)
                corrs.append(cost)

        out_corrs = torch.cat(corrs, dim=1)
        return out_corrs

    def inverse_propagation(self, flow):
        corrs = []
        temp_coord = self.coords + flow
        map2_warp = self.warp(temp_coord, self.shift_map2, self.H, self.W)

        map2_warp = map2_warp.view(self.N, self.C // 2, 2, 5, self.H, self.W)
        corr = torch.mean(map2_warp * self.view_map1, dim=1)
        corr = corr.view(self.N, 10, self.H, self.W)
        return corr

    def __call__(self, flow, is_search=True):
        if is_search:
            out_corrs = self.search(flow)
        else:
            out_corrs = self.inverse_propagation(flow)
        return out_corrs
