import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import bilinear_sampler
from .setrans import CrossAttFeatTrans, gen_all_indices, SETransInputFeatEncoder
import os


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, do_corr_global_norm=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.do_corr_global_norm = do_corr_global_norm
        
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        if self.do_corr_global_norm:
            corr_3d = corr.permute(0, 3, 1, 2, 4, 5).view(batch, dim, -1)
            corr_normed = F.layer_norm( corr_3d, (corr_3d.shape[2],), eps=1e-12 )
            corr = corr_normed.view(batch, dim, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)

        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        # Save corr for visualization
        if 'SAVECORR' in os.environ:
            corr_savepath = os.environ['SAVECORR']
            # corr2: batch, dim, h1, w1, h2, w2.
            corr2 = corr.reshape(batch, h1, w1, dim, h2, w2).permute(0, 3, 1, 2, 4, 5).detach().cpu()
            torch.save(corr2, corr_savepath)
            print(f"Corr tensor saved to {corr_savepath}")

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
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        # Concatenate the four levels (4 resolutions of neighbors), 
        # and permute the neighbors to the channel dimension.
        out = torch.cat(out_pyramid, dim=-1)
        # [batch, number of neighbors, h1, w1]
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, do_corr_global_norm=False):
        super().__init__()
        self.radius = radius
        self.do_corr_global_norm = do_corr_global_norm
        
        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        if self.do_corr_global_norm:
            corr_3d = corr.permute(0, 3, 1, 2, 4, 5).view(B, dim, -1)
            corr_normed = F.layer_norm( corr_3d, (corr_3d.shape[2],), eps=1e-12 )
            corr = corr_normed.view(batch, dim, h1, w1, h2, w2).permute(0, 2, 3, 1, 4, 5)
            
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        self.do_corr_global_norm = do_corr_global_norm
        
    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

# TransCorrBlock instance is created and destroyed in each call of raft.forward().
# It is only for a particular pair of image features fmap1, fmap2
class TransCorrBlock(CorrBlock, nn.Module):
    def __init__(self, config, num_levels=4, radius=4, do_corr_global_norm=False):
        # Do not call CorrBlock.__init__(), as corr is computed differently.
        nn.Module.__init__(self)
        self.num_levels = num_levels
        self.radius = radius
        self.config = config
        self.setrans = CrossAttFeatTrans(self.config, "Inter-frame correlation block")
        self.vispos_encoder = SETransInputFeatEncoder(self.config)
        self.coords2 = None
        self.do_corr_global_norm = do_corr_global_norm

    # Compute the correlation volume between all pairs here.    
    # If fmap1o, fmap2o are None, then fmap1 is conv features, fmap2 is transformer features.
    # If fmap1o, fmap2o are not None, then fmap1, fmap2 are transformer features, 
    # and fmap1o, fmap2o are conv features.
    def update(self, fmap1, fmap2, fmap1o, fmap2o, coords1, coords2=None):
        self.corr_pyramid = []
        # coords1 is generated by coords_grid(), with the format 
        #           (width index,  height index) 
        # flip  =>  (height index, width index)
        coords1     = coords1.permute(0, 2, 3, 1).flip(-1)
        if coords2 is None:
            coords2 = gen_all_indices(fmap2.shape[2:], device=fmap2.device)
            coords2 = coords2.unsqueeze(0).repeat(fmap2.shape[0], 1, 1, 1)
        
        vispos1, pos_biases = self.vispos_encoder(fmap1, coords1, return_pos_biases=True)
        vispos2             = self.vispos_encoder(fmap2, coords2, return_pos_biases=False)
        batch, dim, ht, wd = fmap1.shape

        # If both f1_trans and f2_trans are used, then compute the two-way correlation.
        # Otherwise, only fmap2o is not None, and compute the single-way correlation.  
        if fmap1o is not None and fmap2o is not None:
            vispos1o = self.vispos_encoder(fmap1o, coords1, return_pos_biases=False)
            vispos2o = self.vispos_encoder(fmap2o, coords2, return_pos_biases=False)
            # Two-way correlation.
            corr_1t2o = self.corr(ht, wd, vispos1,  vispos2o, pos_biases)
            corr_1o2t = self.corr(ht, wd, vispos1o, vispos2,  pos_biases)
            # Try concatenation instead of averaging, as averaging may lose info.
            corr = torch.cat([corr_1t2o, corr_1o2t], dim=3)
        else:
            # single-way correlation. 
            corr = self.corr(ht, wd, vispos1, vispos2, pos_biases)

        batch, h1, w1, dim, h2, w2 = corr.shape
        # Merge batch with h1 and w1 to improve efficiency. They will be separate later.
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        if 'SAVECORR' in os.environ:
            corr_savepath = os.environ['SAVECORR']
            corr2 = corr.detach().cpu().reshape(batch, h1, w1, h2, w2)
            torch.save(corr2, corr_savepath)
            print(f"Corr tensor saved to {corr_savepath}")

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def corr(self, ht, wd, vispos1, vispos2, pos_biases):
        batch, ht_wd, dim = vispos1.shape
        assert ht_wd == ht * wd
        # if out_attn_only, output attention matrix is in the shape of (query unit number, key unit number)
        # otherwise, output features are in the same shape as the query features.
        # key features are recombined to get new query features by matmul(attention_probs, V(key features))
        #             frame1 frame2
        # corr: [1, 1, 7040, 7040]
        corr = self.setrans(vispos1, vispos2, pos_biases)
        if self.do_corr_global_norm:
            B, C, H, W = corr.shape
            corr_3d = corr.view(B, C, H*W)
            corr_normed = F.layer_norm( corr_3d, (corr_3d.shape[2],), eps=1e-12 )
            corr = corr_normed.view(B, C, H, W)
                
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr

    # __call__() inherits from CorrBlock.
    