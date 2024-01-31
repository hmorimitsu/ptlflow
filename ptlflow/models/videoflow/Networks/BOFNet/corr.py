import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import bilinear_sampler

# from compute_sparse_correlation import compute_sparse_corr, compute_sparse_corr_torch, compute_sparse_corr_mink

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class DirectCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords):
        ctx.save_for_backward(fmap1, fmap2, coords)
        (corr,) = alt_cuda_corr.forward(fmap1, fmap2, coords, 4)
        return corr

    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = alt_cuda_corr.backward(
            fmap1, fmap2, coords, grad_output, 4
        )

        return fmap1_grad, fmap2_grad, coords_grad


class OLCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        batch, dim, ht, wd = fmap1.shape
        self.fmap1 = fmap1.permute(0, 2, 3, 1).view(batch * ht * wd, 1, dim)

        self.fmap2_pyramid = []
        self.fmap2_pyramid.append(fmap2)
        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.fmap2_pyramid.append(fmap2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        _, _, dim = self.fmap1.shape

        out_pyramid = []
        for i in range(self.num_levels):
            fmap2 = self.fmap2_pyramid[i]
            dx = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            dy = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch, h1 * w1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 1, (2 * r + 1) ** 2, 2)
            coords_lvl = centroid_lvl + delta_lvl

            fmap2 = bilinear_sampler(fmap2, coords_lvl)  # B, 256, h*w, 9*9
            fmap2 = fmap2.permute(0, 2, 1, 3).view(
                batch * h1 * w1, dim, (2 * r + 1) ** 2
            )
            # print(self.fmap1.shape, fmap2.shape)
            corr = torch.bmm(self.fmap1, fmap2) / torch.sqrt(torch.tensor(dim))

            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            dy = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim))


class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim))


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            # fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((None, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            if coords.dtype == torch.float16:
                fmap1_i = fmap1_i.float()
                fmap2_i = fmap2_i.float()
                coords_i = coords_i.float()
            (corr,) = DirectCorr.apply(fmap1_i, fmap2_i, coords_i)
            if coords.dtype == torch.float16:
                corr = corr.half()
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim))
