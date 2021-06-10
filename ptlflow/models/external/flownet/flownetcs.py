from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import init

from .flownetc import ExternalFlowNetC
from .flownets import ExternalFlowNetS
from .submodules import *
from .flownet_base import FlowNetBase

class ExternalFlowNetCS(FlowNetBase):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_flownetcs-things-4bdecffa.ckpt'
    }

    def __init__(self, args: Namespace):
        args.input_channels = 12
        super(ExternalFlowNetCS,self).__init__(args)

        self.args = args
        self.rgb_max = 1

        # First Block (FlowNetC)
        self.flownetc = ExternalFlowNetC(args)

        # Block (FlowNetS)
        self.flownets_1 = ExternalFlowNetS(args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.to(dtype=x.dtype, device=x.device)
        vgrid = grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size()).to(dtype=x.dtype, device=x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def normalize(self, im):
        im = im - 0.5
        im = im / 0.5
        return im

    def forward(self, inputs):
        # flownetc
        flownetc_flow = self.flownetc(inputs)['flows'][:, 0]
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.warp(inputs['images'][:, 1], flownetc_flow)
        diff_img0 = inputs['images'][:, 0] - resampled_img1 
        norm_diff_img0 = torch.norm(diff_img0, p=2, dim=1, keepdim=True)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((inputs['images'][:, 0], inputs['images'][:, 1], resampled_img1, flownetc_flow/self.args.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_preds = self.flownets_1({'images': concat1[:, None]})

        return flownets1_preds
