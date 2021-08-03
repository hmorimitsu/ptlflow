import math

import torch
import torch.nn.functional as F
from .utils import bilinear_sampler


class LearnableCorrBlock(torch.nn.Module):
    def __init__(self, dim, num_levels=4, radius=4):
        super(LearnableCorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.dim = dim

        self.raw_P = torch.nn.Parameter(torch.eye(self.dim), requires_grad=True)
        self.raw_D = torch.nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.register_buffer('eye', torch.eye(self.dim))

    def compute_cost_volume(self, fmap1, fmap2):
        # get matrix W
        self.raw_P_upper = torch.triu(self.raw_P)
        self.skew_P = (self.raw_P_upper - self.raw_P_upper.t()) / 2
        # Cayley representation, P is in Special Orthogonal Group SO(n)
        self.P = torch.matmul((self.eye + self.skew_P), torch.inverse(self.eye - self.skew_P))

        # obtain the diagonal matrix with positive elements
        self.trans_D = torch.atan(self.raw_D) * 2 / math.pi
        self.D = torch.diag((1 + self.trans_D) / (1 - self.trans_D))
        self.W = torch.matmul(torch.matmul(self.P.t(), self.D), self.P)

        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(torch.tensordot(fmap1, self.W, dims=[[1],[0]]), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        corr = corr / torch.sqrt(torch.tensor(dim).float())

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.view(batch*h1*w1, dim, h2, w2)

        corr_pyramid = []
        corr_pyramid.append(corr)
        for i in range(self.num_levels):
            if min(corr.shape[2:4]) > 2*self.radius+1:
                corr = F.avg_pool2d(corr, 2, stride=2)
            corr_pyramid.append(corr)
        return corr_pyramid

    def forward(self, corr_pyramid, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)


# class CorrBlock:
#     def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []

#         # all pairs correlation
#         corr = CorrBlock.corr(fmap1, fmap2)
#         corr = corr.float()

#         batch, h1, w1, dim, h2, w2 = corr.shape
#         corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
#         self.corr_pyramid.append(corr)
#         for i in range(self.num_levels-1):
#             if min(corr.shape[2:4]) > 2*radius+1:
#                 corr = F.avg_pool2d(corr, 2, stride=2)
#             self.corr_pyramid.append(corr)

#     def __call__(self, coords):
#         r = self.radius
#         coords = coords.permute(0, 2, 3, 1)
#         batch, h1, w1, _ = coords.shape

#         out_pyramid = []
#         for i in range(self.num_levels):
#             corr = self.corr_pyramid[i]
#             dx = torch.linspace(-r, r, 2*r+1)
#             dy = torch.linspace(-r, r, 2*r+1)
#             delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

#             centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
#             delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
#             coords_lvl = centroid_lvl + delta_lvl

#             corr = bilinear_sampler(corr, coords_lvl)
#             corr = corr.view(batch, h1, w1, -1)
#             out_pyramid.append(corr)

#         out = torch.cat(out_pyramid, dim=-1)
#         return out.permute(0, 3, 1, 2).contiguous().float()

#     @staticmethod
#     def corr(fmap1, fmap2):
#         batch, dim, ht, wd = fmap1.shape
#         fmap1 = fmap1.view(batch, dim, ht*wd)
#         fmap2 = fmap2.view(batch, dim, ht*wd) 
        
#         corr = torch.matmul(fmap1.transpose(1,2), fmap2)
#         corr = corr.view(batch, ht, wd, 1, ht, wd)
#         return corr  / torch.sqrt(torch.tensor(dim).float())
