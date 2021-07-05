"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
import numpy as np
import math
import pdb
import time

from .submodule import pspnet_s, pspnet, conv
from .conv4d import sepConv4d, butterfly4D
from ..base_model.base_model import BaseModel


class VCNLoss(nn.Module):
    def __init__(self, args):
        super(VCNLoss, self).__init__()
        self.maxdisp = args.maxdisp
        self.fac = args.fac
        self.bs = args.train_batch_size

        self.warpx = WarpModule()

    def forward(self, preds, inputs):
        flowl0 = inputs['flows'][:, 0].clone()
        mask = (inputs['valids'][:, 0, 0] == 1) & (flowl0[:,0].abs() < self.maxdisp) & (flowl0[:,1].abs() < (self.maxdisp//self.fac))
        mask = mask.detach().clone()

        im = inputs['images']
        im = im.view(im.shape[0]*im.shape[1], im.shape[2], im.shape[3], im.shape[4])

        # out-of-range loss
        im_warp, _ = self.warpx(im[self.bs:], flowl0[:,:2])
        occ_mask = (im_warp - im[:self.bs]).norm(dim=1)>0.3

        m = 64.0
        loss = 0
        ws = [0.25, 0.25, 0.25, 0.5, 1.0]
        for i in range(len(ws)):
            loss += ws[i]*torch.norm((preds['flow_preds'][i]*m-flowl0[:, :2]), 0, 1)[mask].mean()
            m /= 2

        im_size = flowl0.shape[2:4]

        m = 32.0
        up_flows = preds['up_flows_preds']
        for i in range(len(up_flows)):
            up_flows[i] = F.interpolate(up_flows[i], [im_size[0], im_size[1]], mode='bilinear', align_corners=False)*m
            m /= 2
        up_flows.insert(0, 0)

        oors = preds['oors_preds']
        for i in range(len(oors)):
            oors[i] = F.interpolate(oors[i][:,np.newaxis], [im_size[0], im_size[1]], mode='bilinear', align_corners=False)[:,0]

        m = 64
        for i in range(len(oors)):
            loss += self.get_oor_loss(flowl0[:,:2]-up_flows[i], oors[i], (m * preds['flow_reg_maxs_preds'][i]), occ_mask)
            m /= 2

        return loss

    def get_oor_loss(self, flowl0, oor3, maxdisp, occ_mask):
        """ 
        return out-of-range loss
        """
        oor3_gt = (flowl0.abs() > maxdisp).detach() #  (8*self.md[3])
        oor3_gt = (((oor3_gt.sum(1)>0) + occ_mask)>0).float()  # oor, or occluded
        weights = oor3_gt.sum().float()/(oor3_gt.shape[0]*oor3_gt.shape[1]*oor3_gt.shape[2])
        weights = oor3_gt * (1-weights) + (1-oor3_gt) * weights
        loss_oor3 = F.binary_cross_entropy_with_logits(oor3,oor3_gt,size_average=True, weight=weights)
        return loss_oor3


class flow_reg(nn.Module):
    """
    Soft winner-take-all that selects the most likely diplacement.
    Set ent=True to enable entropy output.
    Set maxdisp to adjust maximum allowed displacement towards one side.
        maxdisp=4 searches for a 9x9 region.
    Set fac to squeeze search window.
        maxdisp=4 and fac=2 gives search window of 9x5
    """
    def __init__(self, ent=False, maxdisp = int(4), fac=1):
        super(flow_reg, self).__init__()
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3  # by default using truncation 7x7

        self.create_flow([1, 1, 1])

        self.pool3d = nn.MaxPool3d((self.wsize*2+1,self.wsize*2+1,1),stride=1,padding=(self.wsize,self.wsize,0))

    def create_flow(self, size):
        B,W,H = size
        flowrangey = range(-self.md,self.md+1)
        flowrangex = range(-int(self.md//self.fac),int(self.md//self.fac)+1)
        meshgrid = np.meshgrid(flowrangex,flowrangey)
        flowy = np.tile( np.reshape(meshgrid[0],[1,2*self.md+1,2*int(self.md//self.fac)+1,1,1]), (B,1,1,H,W) )
        flowx = np.tile( np.reshape(meshgrid[1],[1,2*self.md+1,2*int(self.md//self.fac)+1,1,1]), (B,1,1,H,W) )
        self.flowx = torch.Tensor(flowx)
        self.flowy = torch.Tensor(flowy)

    def forward(self, x):
        b,u,v,h,w = x.shape
        oldx = x

        if b != self.flowx.shape[0] or h != self.flowx.shape[3] or w != self.flowx.shape[4]:
            self.create_flow((b, w, h))

        if self.truncated:
            # truncated softmax
            x = x.view(b,u*v,h,w)

            idx = x.argmax(1)[:,np.newaxis]
            if x.is_cuda:
                mask = Variable(torch.cuda.HalfTensor(b,u*v,h,w)).fill_(0)
            else:
                mask = Variable(torch.FloatTensor(b,u*v,h,w)).fill_(0)
            mask.scatter_(1,idx,1)
            mask = mask.view(b,1,u,v,-1)
            mask = self.pool3d(mask)[:,0].view(b,u,v,h,w)

            ninf = x.clone().fill_(-np.inf).view(b,u,v,h,w)
            x = torch.where(mask.byte(),oldx,ninf)
        else:
            self.wsize = (np.sqrt(u*v)-1)/2

        b,u,v,h,w = x.shape
        x = F.softmax(x.view(b,-1,h,w),1).view(b,u,v,h,w)
        outx = torch.sum(torch.sum(x*self.flowx.to(x.device),1),1,keepdim=True)
        outy = torch.sum(torch.sum(x*self.flowy.to(x.device),1),1,keepdim=True)

        if self.ent:
            # local
            local_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.
            else:
                local_entropy /= np.log((self.wsize*2+1)**2)

            # global
            x = F.softmax(oldx.view(b,-1,h,w),1).view(b,u,v,h,w)
            global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            global_entropy /= np.log(x.shape[1]*x.shape[2])
            return torch.cat([outx,outy],1),torch.cat([local_entropy, global_entropy],1)
        else:
            return torch.cat([outx,outy],1),None


class WarpModule(nn.Module):
    """
    taken from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    """
    def __init__(self):
        super(WarpModule, self).__init__()
        self.create_grid([1, 1, 1])

    def create_grid(self, size):
        B,W,H = size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        
        # self.register_buffer('grid',torch.cat((xx,yy),1).float())
        self.grid = torch.cat((xx,yy),1).float()

    def forward(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        if B != self.grid.shape[0] or H != self.grid.shape[2] or W != self.grid.shape[3]:
            self.create_grid((B, W, H))

        vgrid = self.grid.to(flo.device) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = ((vgrid[:,:,:,0].abs()<1) * (vgrid[:,:,:,1].abs()<1)) >0
        return output*mask.unsqueeze(1).float(), mask


class VCNSmall(BaseModel):
    """
    VCN.
    md defines maximum displacement for each level, following a coarse-to-fine-warping scheme
    fac defines squeeze parameter for the coarsest level
    """

    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn_small-chairs-2a5f72ba.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn_small-things-e5a44ada.ckpt'
    }

    def __init__(self, args, md=[4,4,4,4,4]):
        super(VCNSmall, self).__init__(
            args=args,
            loss_fn=VCNLoss(args),
            output_stride=64)
        self.md = md
        self.fac = self.args.fac
        use_entropy = True
        withbn = True
        
        ## pspnet
        self.pspnet = pspnet_s(is_proj=False)

        ## Volumetric-UNet
        fdima1 = 128 # 6/5/4
        fdima2 = 64 # 3/2
        fdimb1 = 16 # 6/5/4/3
        fdimb2 = 12 # 2

        full=False
        self.f6 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p6 = sepConv4d(fdimb1,fdimb1, with_bn=False, full=full)

        self.f5 = butterfly4D(fdima1, fdimb1,withbn=withbn, full=full)
        self.p5 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f4 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p4 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f3 = butterfly4D(fdima2, fdimb1,withbn=withbn,full=full)
        self.p3 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        ## soft WTA modules
        self.flow_reg64 = flow_reg(ent=use_entropy, maxdisp=self.md[0], fac=self.fac)
        self.flow_reg32 = flow_reg(ent=use_entropy, maxdisp=self.md[1])
        self.flow_reg16 = flow_reg(ent=use_entropy, maxdisp=self.md[2])
        self.flow_reg8 =  flow_reg(ent=use_entropy, maxdisp=self.md[3])

        ## warping modules
        self.warp5 = WarpModule()
        self.warp4 = WarpModule()
        self.warp3 = WarpModule()
        if self.training:
            self.warpx = WarpModule()
       

        ## hypotheses fusion modules, adopted from the refinement module of PWCNet
        # https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
        # c6
        self.dc6_conv1 = conv(128+4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc6_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc6_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc6_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc6_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv7 = nn.Conv2d(32,2*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)

        # c5
        self.dc5_conv1 = conv(128+4*fdimb1*2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc5_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc5_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc5_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc5_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv7 = nn.Conv2d(32,2*fdimb1*2,kernel_size=3,stride=1,padding=1,bias=True)

        # c4
        self.dc4_conv1 = conv(128+4*fdimb1*3, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc4_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc4_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc4_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc4_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv7 = nn.Conv2d(32,2*fdimb1*3,kernel_size=3,stride=1,padding=1,bias=True)

        # c3
        self.dc3_conv1 = conv(64+16*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc3_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc3_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc3_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc3_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv7 = nn.Conv2d(32,8*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)

        ## Out-of-range detection
        # if size[0]>1:  # only in train mode
        self.dc6_convo = nn.Sequential(conv(128+4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc5_convo = nn.Sequential(conv(128+2*4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc4_convo = nn.Sequential(conv(128+3*4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc3_convo = nn.Sequential(conv(64+16*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--fac', type=float, default=1.0)
        parser.add_argument('--maxdisp', type=int, default=256)
        return parser
   
    #@profile
    def corr(self, refimg_fea, targetimg_fea, maxdisp, fac=1):
        """
        correlation function. Adopted from https://github.com/ClementPinard/Pytorch-Correlation-extension
        faster, but backwards not implemented.
        """
        from spatial_correlation_sampler import SpatialCorrelationSampler
        corr = SpatialCorrelationSampler(kernel_size=1,patch_size=(int(1+2*maxdisp//fac),int(1+2*maxdisp)),stride=1,padding=0,dilation_patch=1)
        cost = corr(refimg_fea, targetimg_fea)
        cost = F.leaky_relu(cost, 0.1,inplace=True)
        return cost


    def corrf(self, refimg_fea, targetimg_fea,maxdisp, fac=1):
        """
        another correlation function giving the same result as corr()
        supports backwards
        """
        b,c,height,width = refimg_fea.shape
        if refimg_fea.is_cuda:
            cost = Variable(torch.cuda.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//fac)+1,height,width)).fill_(0.) # b,c,u,v,h,w
        else:
            cost = Variable(torch.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//fac)+1,height,width)).fill_(0.) # b,c,u,v,h,w
        for i in range(2*maxdisp+1):
            ind = i-maxdisp
            for j in range(2*int(maxdisp//fac)+1):
                indd = j-int(maxdisp//fac)
                feata = refimg_fea[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                featb = targetimg_fea[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
                diff = (feata*featb)
                cost[:, :, i,j,max(0,-indd):height-indd,max(0,-ind):width-ind]   = diff
        cost = F.leaky_relu(cost, 0.1,inplace=True)
        return cost

    def get_oor_loss(self, flowl0, oor3, maxdisp, occ_mask):
        """ 
        return out-of-range loss
        """
        oor3_gt = (flowl0.abs() > maxdisp).detach() #  (8*self.md[3])
        oor3_gt = (((oor3_gt.sum(1)>0) + occ_mask)>0).float()  # oor, or occluded
        weights = oor3_gt.sum().float()/(oor3_gt.shape[0]*oor3_gt.shape[1]*oor3_gt.shape[2])
        weights = oor3_gt * (1-weights) + (1-oor3_gt) * weights
        loss_oor3 = F.binary_cross_entropy_with_logits(oor3,oor3_gt,size_average=True, weight=weights)
        return loss_oor3

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, inputs, disc_aux=None):
        im = inputs['images']
        im = im.view(im.shape[0]*im.shape[1], im.shape[2], im.shape[3], im.shape[4])

        bs = im.shape[0]//2

        c06,c05,c04,c03 = self.pspnet(im)
        c16 = c06[:bs];  c26 = c06[bs:]
        c15 = c05[:bs];  c25 = c05[bs:]
        c14 = c04[:bs];  c24 = c04[bs:]
        c13 = c03[:bs];  c23 = c03[bs:]

        # normalize
        c16n = c16 / (c16.norm(dim=1, keepdim=True)+1e-9)
        c26n = c26 / (c26.norm(dim=1, keepdim=True)+1e-9)
        c15n = c15 / (c15.norm(dim=1, keepdim=True)+1e-9)
        c25n = c25 / (c25.norm(dim=1, keepdim=True)+1e-9)
        c14n = c14 / (c14.norm(dim=1, keepdim=True)+1e-9)
        c24n = c24 / (c24.norm(dim=1, keepdim=True)+1e-9)
        c13n = c13 / (c13.norm(dim=1, keepdim=True)+1e-9)
        c23n = c23 / (c23.norm(dim=1, keepdim=True)+1e-9)

        up_flows = []
        oors = []
        flow_reg_maxs = []

        ## matching 6
        if self.training or (not im.is_cuda):
            feat6 = self.corrf(c16n,c26n,self.md[0],fac=self.fac)
        else:
            feat6 = self.corrf(c16n,c26n,self.md[0],fac=self.fac)
            #feat6 = self.corr(c16n,c26n,self.md[0],fac=self.fac)
        feat6 = self.f6(feat6) 
        cost6 = self.p6(feat6) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost6.shape
        cost6 = cost6.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow6h,ent6h = self.flow_reg64(cost6) # bx16, 2, h, w
        flow6h =  flow6h.view(bs,-1,h,w) # b, 16*2, h, w
        ent6h =  ent6h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg64.flowx.max())

        if self.training:
            x = torch.cat((ent6h.detach(), flow6h.detach(), c16),1)
            oor6 = self.dc6_convo(x)[:,0]
            oors.append(oor6)

        # hypotheses fusion
        x = torch.cat((ent6h.detach(), flow6h.detach(), c16),1)
        x = self.dc6_conv4(self.dc6_conv3(self.dc6_conv2(self.dc6_conv1(x))))
        va = self.dc6_conv7(self.dc6_conv6(self.dc6_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow6 = ( flow6h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        ## matching 5
        up_flow6 = F.interpolate(flow6, [im.size()[2]//32,im.size()[3]//32], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow6)
        warp5,_ = self.warp5(c25n, up_flow6)
        if self.training or (not im.is_cuda):
            feat5 = self.corrf(c15n,warp5,self.md[1])
        else:
            feat5 = self.corrf(c15n,warp5,self.md[1])
            #feat5 = self.corr(c15n,warp5,self.md[1])
        feat5 = self.f5(feat5) 
        cost5 = self.p5(feat5) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost5.shape
        cost5 = cost5.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow5h,ent5h = self.flow_reg32(cost5) # bx16, 2, h, w
        flow5h = flow5h.view(b,c,2,h,w) + up_flow6[:,np.newaxis]
        flow5h = flow5h.view(bs,-1,h,w) # b, 16*2, h, w
        ent5h =  ent5h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg32.flowx.max())

        # append coarse hypotheses
        flow5h = torch.cat((flow5h, F.interpolate(flow6h.detach()*2, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear', align_corners=False)),1) # b, k2--k2, h, w
        ent5h = torch.cat((ent5h, F.interpolate(ent6h, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent5h.detach(), flow5h.detach(), c15),1)
            oor5 = self.dc5_convo(x)[:,0]
            oors.append(oor5)

        # hypotheses fusion
        x = torch.cat((ent5h.detach(), flow5h.detach(), c15),1)
        x = self.dc5_conv4(self.dc5_conv3(self.dc5_conv2(self.dc5_conv1(x))))
        va5 = self.dc5_conv7(self.dc5_conv6(self.dc5_conv5(x)))
        va5 = va5.view(b,-1,2,h,w)
        flow5 = ( flow5h.view(b,-1,2,h,w) * F.softmax(va5,1) ).sum(1) # b, 2k, 2, h, w
        

        ## matching 4
        up_flow5 = F.interpolate(flow5, [im.size()[2]//16,im.size()[3]//16], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow5)
        warp4,_ = self.warp4(c24n, up_flow5)
        if self.training or (not im.is_cuda):
            feat4 = self.corrf(c14n,warp4,self.md[2])
        else:
            feat4 = self.corrf(c14n,warp4,self.md[2])
            #feat4 = self.corr(c14n,warp4,self.md[2])
        feat4 = self.f4(feat4) 
        cost4 = self.p4(feat4) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost4.shape
        cost4 = cost4.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow4h,ent4h = self.flow_reg16(cost4) # bx16, 2, h, w
        flow4h = flow4h.view(b,c,2,h,w) + up_flow5[:,np.newaxis]
        flow4h =  flow4h.view(bs,-1,h,w) # b, 16*2, h, w
        ent4h =  ent4h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg16.flowx.max())

        # append coarse hypotheses
        flow4h = torch.cat((flow4h, F.interpolate(flow5h.detach()*2, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear', align_corners=False)),1)
        ent4h = torch.cat((ent4h, F.interpolate(ent5h, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent4h.detach(), flow4h.detach(), c14),1)
            oor4 = self.dc4_convo(x)[:,0]
            oors.append(oor4)

        # hypotheses fusion
        x = torch.cat((ent4h.detach(), flow4h.detach(), c14),1)
        x =  self.dc4_conv4(self.dc4_conv3(self.dc4_conv2(self.dc4_conv1(x))))
        va = self.dc4_conv7(self.dc4_conv6(self.dc4_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow4 = ( flow4h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)


        ## matching 3
        up_flow4 = F.interpolate(flow4, [im.size()[2]//8,im.size()[3]//8], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow4)
        warp3,_ = self.warp3(c23n, up_flow4)
        if self.training or (not im.is_cuda):
            feat3 = self.corrf(c13n,warp3,self.md[3])
        else:
            feat3 = self.corrf(c13n,warp3,self.md[3])
            #feat3 = self.corr(c13n,warp3,self.md[3])
        feat3 = self.f3(feat3) 
        cost3 = self.p3(feat3) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost3.shape
        cost3 = cost3.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow3h,ent3h = self.flow_reg8(cost3) # bx16, 2, h, w
        flow3h = flow3h.view(b,c,2,h,w) + up_flow4[:,np.newaxis]
        flow3h = flow3h.view(bs,-1,h,w) # b, 16*2, h, w
        ent3h =  ent3h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg8.flowx.max())

        # append coarse hypotheses
        flow3h = torch.cat((flow3h, F.interpolate(flow4h.detach()*2, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear', align_corners=False)),1)
        ent3h = torch.cat((ent3h, F.interpolate(ent4h, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent3h.detach(), flow3h.detach(), c13),1)
            oor3 = self.dc3_convo(x)[:,0]
            oors.append(oor3)

        # hypotheses fusion
        x = torch.cat((ent3h.detach(), flow3h.detach(), c13),1)
        x = self.dc3_conv4(self.dc3_conv3(self.dc3_conv2(self.dc3_conv1(x))))
        va = self.dc3_conv7(self.dc3_conv6(self.dc3_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow3 = ( flow3h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        flow3 = F.interpolate(flow3, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow4 = F.interpolate(flow4, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow5 = F.interpolate(flow5, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow6 = F.interpolate(flow6, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow2 = flow3*2
        flows = [flow6, flow5, flow4, flow3, flow2]

        outputs = {}

        if self.training:
            outputs['flow_preds'] = flows
            outputs['up_flows_preds'] = up_flows
            outputs['oors_preds'] = oors
            outputs['flow_reg_maxs_preds'] = flow_reg_maxs
            outputs['flows'] = flow2[:, None]*4
        else:
            outputs['flows'] = flow2[:, None]*4

        return outputs


class VCN(VCNSmall):
    """
    VCN.
    md defines maximum displacement for each level, following a coarse-to-fine-warping scheme
    fac defines squeeze parameter for the coarsest level
    """

    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn-chairs-792544a0.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn-things-afc0f455.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn-sintel-5aac0540.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/vcn-kitti-29aa0c1a.ckpt'
    }

    def __init__(self, args, md=[4,4,4,4,4]):
        super(VCN, self).__init__(
            args=args, md=md)
        use_entropy = True
        withbn = True
        
        ## pspnet
        self.pspnet = pspnet(is_proj=False)

        ## Volumetric-UNet
        fdima1 = 128 # 6/5/4
        fdima2 = 64 # 3/2
        fdimb1 = 16 # 6/5/4/3
        fdimb2 = 12 # 2

        full=True
        self.f2 = butterfly4D(fdima2, fdimb2,withbn=withbn,full=full)
        self.p2 = sepConv4d(fdimb2,fdimb2, with_bn=False,full=full)

        ## soft WTA modules
        self.flow_reg4 =  flow_reg(ent=use_entropy, maxdisp=self.md[4])

        ## warping modules
        self.warp2 = WarpModule()
       

        ## hypotheses fusion modules, adopted from the refinement module of PWCNet
        # https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
        # c2
        self.dc2_conv1 = conv(64+16*fdimb1+4*fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc2_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc2_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc2_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc2_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv7 = nn.Conv2d(32,4*2*fdimb1 + 2*fdimb2,kernel_size=3,stride=1,padding=1,bias=True)

        ## Out-of-range detection
        # if size[0]>1:  # only in train mode
        self.dc2_convo = nn.Sequential(conv(64+16*fdimb1+4*fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, inputs, disc_aux=None):
        im = inputs['images']
        im = im.view(im.shape[0]*im.shape[1], im.shape[2], im.shape[3], im.shape[4])

        bs = im.shape[0]//2

        c06,c05,c04,c03,c02 = self.pspnet(im)
        c16 = c06[:bs];  c26 = c06[bs:]
        c15 = c05[:bs];  c25 = c05[bs:]
        c14 = c04[:bs];  c24 = c04[bs:]
        c13 = c03[:bs];  c23 = c03[bs:]
        c12 = c02[:bs];  c22 = c02[bs:]

        # normalize
        c16n = c16 / (c16.norm(dim=1, keepdim=True)+1e-9)
        c26n = c26 / (c26.norm(dim=1, keepdim=True)+1e-9)
        c15n = c15 / (c15.norm(dim=1, keepdim=True)+1e-9)
        c25n = c25 / (c25.norm(dim=1, keepdim=True)+1e-9)
        c14n = c14 / (c14.norm(dim=1, keepdim=True)+1e-9)
        c24n = c24 / (c24.norm(dim=1, keepdim=True)+1e-9)
        c13n = c13 / (c13.norm(dim=1, keepdim=True)+1e-9)
        c23n = c23 / (c23.norm(dim=1, keepdim=True)+1e-9)
        c12n = c12 / (c12.norm(dim=1, keepdim=True)+1e-9)
        c22n = c22 / (c22.norm(dim=1, keepdim=True)+1e-9)

        up_flows = []
        oors = []
        flow_reg_maxs = []

        ## matching 6
        if self.training or (not im.is_cuda):
            feat6 = self.corrf(c16n,c26n,self.md[0],fac=self.fac)
        else:
            feat6 = self.corrf(c16n,c26n,self.md[0],fac=self.fac)
            #feat6 = self.corr(c16n,c26n,self.md[0],fac=self.fac)
        feat6 = self.f6(feat6) 
        cost6 = self.p6(feat6) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost6.shape
        cost6 = cost6.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow6h,ent6h = self.flow_reg64(cost6) # bx16, 2, h, w
        flow6h =  flow6h.view(bs,-1,h,w) # b, 16*2, h, w
        ent6h =  ent6h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg64.flowx.max())

        if self.training:
            x = torch.cat((ent6h.detach(), flow6h.detach(), c16),1)
            oor6 = self.dc6_convo(x)[:,0]
            oors.append(oor6)

        # hypotheses fusion
        x = torch.cat((ent6h.detach(), flow6h.detach(), c16),1)
        x = self.dc6_conv4(self.dc6_conv3(self.dc6_conv2(self.dc6_conv1(x))))
        va = self.dc6_conv7(self.dc6_conv6(self.dc6_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow6 = ( flow6h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        ## matching 5
        up_flow6 = F.interpolate(flow6, [im.size()[2]//32,im.size()[3]//32], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow6)
        warp5,_ = self.warp5(c25n, up_flow6)
        if self.training or (not im.is_cuda):
            feat5 = self.corrf(c15n,warp5,self.md[1])
        else:
            feat5 = self.corrf(c15n,warp5,self.md[1])
            #feat5 = self.corr(c15n,warp5,self.md[1])
        feat5 = self.f5(feat5) 
        cost5 = self.p5(feat5) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost5.shape
        cost5 = cost5.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow5h,ent5h = self.flow_reg32(cost5) # bx16, 2, h, w
        flow5h = flow5h.view(b,c,2,h,w) + up_flow6[:,np.newaxis]
        flow5h = flow5h.view(bs,-1,h,w) # b, 16*2, h, w
        ent5h =  ent5h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg32.flowx.max())

        # append coarse hypotheses
        flow5h = torch.cat((flow5h, F.interpolate(flow6h.detach()*2, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear', align_corners=False)),1) # b, k2--k2, h, w
        ent5h = torch.cat((ent5h, F.interpolate(ent6h, [flow5h.shape[2],flow5h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent5h.detach(), flow5h.detach(), c15),1)
            oor5 = self.dc5_convo(x)[:,0]
            oors.append(oor5)

        # hypotheses fusion
        x = torch.cat((ent5h.detach(), flow5h.detach(), c15),1)
        x = self.dc5_conv4(self.dc5_conv3(self.dc5_conv2(self.dc5_conv1(x))))
        va5 = self.dc5_conv7(self.dc5_conv6(self.dc5_conv5(x)))
        va5 = va5.view(b,-1,2,h,w)
        flow5 = ( flow5h.view(b,-1,2,h,w) * F.softmax(va5,1) ).sum(1) # b, 2k, 2, h, w
        

        ## matching 4
        up_flow5 = F.interpolate(flow5, [im.size()[2]//16,im.size()[3]//16], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow5)
        warp4,_ = self.warp4(c24n, up_flow5)
        if self.training or (not im.is_cuda):
            feat4 = self.corrf(c14n,warp4,self.md[2])
        else:
            feat4 = self.corrf(c14n,warp4,self.md[2])
            #feat4 = self.corr(c14n,warp4,self.md[2])
        feat4 = self.f4(feat4) 
        cost4 = self.p4(feat4) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost4.shape
        cost4 = cost4.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow4h,ent4h = self.flow_reg16(cost4) # bx16, 2, h, w
        flow4h = flow4h.view(b,c,2,h,w) + up_flow5[:,np.newaxis]
        flow4h =  flow4h.view(bs,-1,h,w) # b, 16*2, h, w
        ent4h =  ent4h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg16.flowx.max())

        # append coarse hypotheses
        flow4h = torch.cat((flow4h, F.interpolate(flow5h.detach()*2, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear', align_corners=False)),1)
        ent4h = torch.cat((ent4h, F.interpolate(ent5h, [flow4h.shape[2],flow4h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent4h.detach(), flow4h.detach(), c14),1)
            oor4 = self.dc4_convo(x)[:,0]
            oors.append(oor4)

        # hypotheses fusion
        x = torch.cat((ent4h.detach(), flow4h.detach(), c14),1)
        x =  self.dc4_conv4(self.dc4_conv3(self.dc4_conv2(self.dc4_conv1(x))))
        va = self.dc4_conv7(self.dc4_conv6(self.dc4_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow4 = ( flow4h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)


        ## matching 3
        up_flow4 = F.interpolate(flow4, [im.size()[2]//8,im.size()[3]//8], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow4)
        warp3,_ = self.warp3(c23n, up_flow4)
        if self.training or (not im.is_cuda):
            feat3 = self.corrf(c13n,warp3,self.md[3])
        else:
            feat3 = self.corrf(c13n,warp3,self.md[3])
            #feat3 = self.corr(c13n,warp3,self.md[3])
        feat3 = self.f3(feat3) 
        cost3 = self.p3(feat3) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost3.shape
        cost3 = cost3.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flow3h,ent3h = self.flow_reg8(cost3) # bx16, 2, h, w
        flow3h = flow3h.view(b,c,2,h,w) + up_flow4[:,np.newaxis]
        flow3h = flow3h.view(bs,-1,h,w) # b, 16*2, h, w
        ent3h =  ent3h.view(bs,-1,h,w) # b, 16*1, h, w
        flow_reg_maxs.append(self.flow_reg8.flowx.max())

        # append coarse hypotheses
        flow3h = torch.cat((flow3h, F.interpolate(flow4h.detach()*2, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear', align_corners=False)),1)
        ent3h = torch.cat((ent3h, F.interpolate(ent4h, [flow3h.shape[2],flow3h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent3h.detach(), flow3h.detach(), c13),1)
            oor3 = self.dc3_convo(x)[:,0]
            oors.append(oor3)

        # hypotheses fusion
        x = torch.cat((ent3h.detach(), flow3h.detach(), c13),1)
        x = self.dc3_conv4(self.dc3_conv3(self.dc3_conv2(self.dc3_conv1(x))))
        va = self.dc3_conv7(self.dc3_conv6(self.dc3_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow3 = ( flow3h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)

        ## matching 2
        up_flow3 = F.interpolate(flow3, [im.size()[2]//4,im.size()[3]//4], mode='bilinear', align_corners=False)*2
        up_flows.append(up_flow3)
        warp2,_ = self.warp2(c22n, up_flow3)
        if self.training or (not im.is_cuda):
            feat2 = self.corrf(c12n,warp2,self.md[4])
        else:
            feat2 = self.corrf(c12n,warp2,self.md[4])
            #feat2 = self.corr(c12n,warp2,self.md[4])
        feat2 = self.f2(feat2) 
        cost2 = self.p2(feat2) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost2.shape
        cost2 = cost2.view(-1,u,v,h,w)  # bx12, 9,9,h,w, also predict uncertainty from here
        flow2h,ent2h = self.flow_reg4(cost2) # bx12, 2, h, w
        flow2h = flow2h.view(b,c,2,h,w) + up_flow3[:,np.newaxis]
        flow2h = flow2h.view(bs,-1,h,w) # b, 12*2, h, w
        ent2h =  ent2h.view(bs,-1,h,w) # b, 12*1, h, w
        flow_reg_maxs.append(self.flow_reg4.flowx.max())

        # append coarse hypotheses
        flow2h = torch.cat((flow2h, F.interpolate(flow3h.detach()*2, [flow2h.shape[2],flow2h.shape[3]], mode='bilinear', align_corners=False)),1)
        ent2h = torch.cat((ent2h, F.interpolate(ent3h, [flow2h.shape[2],flow2h.shape[3]], mode='bilinear', align_corners=False)),1)

        if self.training:
            x = torch.cat((ent2h.detach(), flow2h.detach(), c12),1)
            oor2 = self.dc2_convo(x)[:,0]
            oors.append(oor2)

        # hypotheses fusion
        x = torch.cat((ent2h.detach(), flow2h.detach(), c12),1)
        x = self.dc2_conv4(self.dc2_conv3(self.dc2_conv2(self.dc2_conv1(x))))
        va = self.dc2_conv7(self.dc2_conv6(self.dc2_conv5(x)))
        va = va.view(b,-1,2,h,w)
        flow2 = ( flow2h.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1)


        flow2 = F.interpolate(flow2, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow3 = F.interpolate(flow3, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow4 = F.interpolate(flow4, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow5 = F.interpolate(flow5, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flow6 = F.interpolate(flow6, [im.size()[2],im.size()[3]], mode='bilinear', align_corners=False)
        flows = [flow6, flow5, flow4, flow3, flow2]

        outputs = {}

        if self.training:
            outputs['flow_preds'] = flows
            outputs['up_flows_preds'] = up_flows
            outputs['oors_preds'] = oors
            outputs['flow_reg_maxs_preds'] = flow_reg_maxs
            outputs['flows'] = flow2[:, None]*4
        else:
            outputs['flows'] = flow2[:, None]*4

        return outputs
