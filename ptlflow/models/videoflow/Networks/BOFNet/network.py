import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import GMAUpdateBlock
from ..encoders import twins_svt_large
from .cnn import BasicEncoder
from .corr import CorrBlock, OLCorrBlock, AlternateCorrBlock
from ...utils.utils import bilinear_sampler, coords_grid, upflow8
from .gma import Attention, Aggregate
from .sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder
from .sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2

from torchvision.utils import save_image

autocast = torch.cuda.amp.autocast

class BOFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        cfg.corr_radius = 4
        cfg.corr_levels = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='instance')

        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')

        if self.cfg.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
        elif self.cfg.gma == 'GMA-SK':
            print("[Using GMA-SK]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(args=self.cfg, hidden_dim=128)
        elif self.cfg.gma == 'GMA-SK2':
            print("[Using GMA-SK2]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(args=self.cfg, hidden_dim=128)
        
        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        self.att = Attention(args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, images, data={}, flow_init=None):

        B, N, _, H, W = images.shape

        images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.cfg.mixed_precision):
            fmaps = self.fnet(images.reshape(B*N, 3, H, W)).reshape(B, N, -1, H//8, W//8)
        fmaps = fmaps.float()
        fmap1 = fmaps[:, 0, ...]
        fmap2 = fmaps[:, 1, ...]
        fmap3 = fmaps[:, 2, ...]
        
        if self.cfg.corr_fn == "efficient":
            corr_fn_21 = AlternateCorrBlock(fmap2, fmap1, radius=self.cfg.corr_radius)
            corr_fn_23 = AlternateCorrBlock(fmap2, fmap3, radius=self.cfg.corr_radius)
        else:
            corr_fn_21 = CorrBlock(fmap2, fmap1, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
            corr_fn_23 = CorrBlock(fmap2, fmap3, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
    
        with autocast(enabled=self.cfg.mixed_precision):
            cnet = self.cnet(images[:, 1, ...])
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        coords0_21, coords1_21 = self.initialize_flow(images[:, 0, ...])
        coords0_23, coords1_23 = self.initialize_flow(images[:, 0, ...])

        flow_predictions = []
        for itr in range(self.cfg.decoder_depth):
            coords1_21 = coords1_21.detach()
            coords1_23 = coords1_23.detach()
            
            corr21 = corr_fn_21(coords1_21)
            corr23 = corr_fn_23(coords1_23)
            corr =  torch.cat([corr23, corr21], dim=1)
            
            flow21 = coords1_21 - coords0_21
            flow23 = coords1_23 - coords0_23
            flow = torch.cat([flow23, flow21], dim=1)
            
            with autocast(enabled=self.cfg.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            up_mask_21, up_mask_23 = torch.split(up_mask, [64*9, 64*9], dim=1)

            coords1_23 = coords1_23 + delta_flow[:, 0:2, ...]
            coords1_21 = coords1_21 + delta_flow[:, 2:4, ...]

            # upsample predictions
            flow_up_23 = self.upsample_flow(coords1_23 - coords0_23, up_mask_23)
            flow_up_21 = self.upsample_flow(coords1_21 - coords0_21, up_mask_21)

            flow_predictions.append(torch.stack([flow_up_23, flow_up_21], dim=1))

        if self.training:
            return flow_predictions
        else:
            return flow_predictions[-1], torch.stack([coords1_23-coords0_23, coords1_21-coords0_21], dim=1)
