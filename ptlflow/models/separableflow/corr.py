import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import bilinear_sampler

try:
    from .libs.GANet.modules.GANet import NLFIter
except ModuleNotFoundError:
    NLFIter = None

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class NLF(nn.Module):
    def __init__(self, in_channel=32):
        super(NLF, self).__init__()
        self.nlf = NLFIter()

    def forward(self, x, g):
        N, D1, D2, H, W = x.shape
        x = x.reshape(N, D1 * D2, H, W).contiguous()
        rem = x
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)
        #        k1, k2, k3, k4 = self.getweights(x)
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)

        x = self.nlf(x, k1, k2, k3, k4)
        #        x = x + rem
        x = x.reshape(N, D1, D2, H, W)
        return x


class CorrBlock:
    def __init__(self, fmap1, fmap2, guid, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        self.nlf = NLF()
        corr = self.corr_compute(fmap1, fmap2, guid, reverse=True)
        # corr = self.nlf(corr, g)
        # corr = corr.permute(0, 3,4, 1,2)

        batch, h1, w1, h2, w2 = corr.shape
        self.shape = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def separate(self):
        sep_u = []
        sep_v = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            m1, _ = corr.max(dim=2, keepdim=True)
            m2 = corr.mean(dim=2, keepdim=True)
            sep = torch.cat((m1, m2), dim=2)
            sep = sep.reshape(
                self.shape[0], self.shape[1], self.shape[2], sep.shape[2], sep.shape[3]
            ).permute(0, 3, 4, 1, 2)
            sep = F.interpolate(
                sep,
                [self.shape[4], self.shape[1], self.shape[2]],
                mode="trilinear",
                align_corners=True,
            )
            sep_u.append(sep)
            m1, _ = corr.max(dim=3, keepdim=True)
            m2 = corr.mean(dim=3, keepdim=True)
            sep = torch.cat((m1, m2), dim=3)
            sep = sep.reshape(
                self.shape[0], self.shape[1], self.shape[2], sep.shape[2], sep.shape[3]
            ).permute(0, 4, 3, 1, 2)
            sep = F.interpolate(
                sep,
                [self.shape[3], self.shape[1], self.shape[2]],
                mode="trilinear",
                align_corners=True,
            )
            sep_v.append(sep)
        sep_u = torch.cat(sep_u, dim=1)
        sep_v = torch.cat(sep_v, dim=1)
        return sep_u, sep_v

    def __call__(self, coords, sep=False):
        if sep:
            return self.separate()
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    # @staticmethod
    def corr_compute(self, fmap1, fmap2, guid, reverse=True):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        if reverse:
            corr = torch.matmul(fmap2.transpose(1, 2), fmap1) / torch.sqrt(
                torch.tensor(dim).float()
            )
            corr = corr.view(batch, ht, wd, ht, wd)
            corr = self.nlf(corr, guid)
            corr = corr.permute(0, 3, 4, 1, 2)
        else:
            corr = torch.matmul(fmap1.transpose(1, 2), fmap2) / torch.sqrt(
                torch.tensor(dim).float()
            )
            corr = corr.view(batch, ht, wd, ht, wd)
            corr = self.nlf(corr, guid)

        return corr


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

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
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            if coords.dtype == torch.float16:
                corr = corr.half()
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlock1D:
    def __init__(self, corr1, corr2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid1 = []
        self.corr_pyramid2 = []

        corr1 = corr1.permute(0, 3, 4, 1, 2)
        corr2 = corr2.permute(0, 3, 4, 1, 2)
        batch, h1, w1, dim, w2 = corr1.shape
        batch, h1, w1, dim, h2 = corr2.shape
        assert corr1.shape[:-1] == corr2.shape[:-1]
        assert h1 == h2 and w1 == w2

        # self.coords = coords_grid(batch, h2, w2).to(corr1.device)

        corr1 = corr1.reshape(batch * h1 * w1, dim, 1, w2)
        corr2 = corr2.reshape(batch * h1 * w1, dim, 1, h2)

        self.corr_pyramid1.append(corr1)
        self.corr_pyramid2.append(corr2)
        for i in range(self.num_levels):
            corr1 = F.avg_pool2d(corr1, [1, 2], stride=[1, 2])
            self.corr_pyramid1.append(corr1)
            corr2 = F.avg_pool2d(corr2, [1, 2], stride=[1, 2])
            self.corr_pyramid2.append(corr2)
            # print(corr1.shape, corr1.mean().item(), corr2.shape, corr2.mean().item())

    def bilinear_sampler(self, img, coords, mode="bilinear", mask=False):
        """Wrapper for grid_sample, uses pixel coordinates"""
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def __call__(self, coords):
        coords_org = coords.clone()
        coords = coords_org[:, :1, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid1[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            coords_lvl = torch.clamp(coords_lvl, -1, 1)
            # print("corri:", corr.shape, corr.mean().item(), coords_lvl.shape, coords_lvl.mean().item())
            corr = self.bilinear_sampler(corr, coords_lvl)
            # print("corri:", corr.shape, corr.mean().item())
            corr = corr.view(batch, h1, w1, -1)
            # print("corri:", corr.shape, corr.mean().item())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out1 = out.permute(0, 3, 1, 2).contiguous().float()

        coords = coords_org[:, 1:, :, :]
        coords = coords.permute(0, 2, 3, 1)
        r = self.radius
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid2[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out2 = out.permute(0, 3, 1, 2).contiguous().float()
        return out1, out2
