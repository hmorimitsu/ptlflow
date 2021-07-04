'''
Portions of this code copyright 2017, Clement Pinard
'''

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler
import torch
import torch.nn as nn

from ...base_model.base_model import BaseModel
from ..flownet.losses import MultiScale


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ExternalPWCNet(BaseModel):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_pwcnet-things-6a2e540b.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_pwcnet-sintel-0916cff4.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        super(ExternalPWCNet, self).__init__(
            args=args,
            loss_fn=MultiScale(
                startScale=args.loss_start_scale,
                numScales=args.loss_num_scales,
                l_weight=args.loss_base_weight,
                norm=args.loss_norm
            ),
            output_stride=64)

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.args.md+1, padding=0)
        
        nd = (2*self.args.md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        parser.add_argument('--md', type=int, default=4)
        parser.add_argument('--loss_start_scale', type=float, default=4)
        parser.add_argument('--loss_num_scales', type=int, default=5)
        parser.add_argument('--loss_base_weight', type=float, default=0.32)
        parser.add_argument('--loss_norm', type=str, default='L2')
        return parser

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


    def forward(self, inputs):
        im1 = inputs['images'][:, 0]
        im2 = inputs['images'][:, 1]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26)
        corr6 = corr6.view(corr6.shape[0], -1, corr6.shape[3], corr6.shape[4])
        corr6 = corr6 / c16.shape[1]
        corr6 = self.leakyRELU(corr6)


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = corr5.view(corr5.shape[0], -1, corr5.shape[3], corr5.shape[4])
        corr5 = corr5 / c15.shape[1]
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = corr4.view(corr4.shape[0], -1, corr4.shape[3], corr4.shape[4])
        corr4 = corr4 / c14.shape[1]
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = corr3.view(corr3.shape[0], -1, corr3.shape[3], corr3.shape[4])
        corr3 = corr3 / c13.shape[1]
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = corr2.view(corr2.shape[0], -1, corr2.shape[3], corr2.shape[4])
        corr2 = corr2 / c12.shape[1]
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)

        flow_up = self.upsample1(flow2*self.args.div_flow)

        outputs = {}
        if self.training:
            outputs['flow_preds'] = [flow2, flow3, flow4, flow5, flow6]
            outputs['flows'] = flow_up[:, None]
        else:
            outputs['flows'] = flow_up[:, None]
        return outputs


class ExternalPWCDCNet(ExternalPWCNet):
    pretrained_checkpoints = {
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_pwcdcnet-things-cc223701.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ext_pwcdcnet-sintel-0f9cc3bf.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        super(ExternalPWCDCNet, self).__init__(args=args)
        
        nd = (2*self.args.md+1)**2
        dd = np.cumsum([128,128,96,64,32])
        od = nd+32+4
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, inputs):
        im1 = inputs['images'][:, 0]
        im2 = inputs['images'][:, 1]
        
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))


        corr6 = self.corr(c16, c26)
        corr6 = corr6.view(corr6.shape[0], -1, corr6.shape[3], corr6.shape[4])
        corr6 = corr6 / c16.shape[1]
        corr6 = self.leakyRELU(corr6)


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = corr5.view(corr5.shape[0], -1, corr5.shape[3], corr5.shape[4])
        corr5 = corr5 / c15.shape[1]
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

       
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = corr4.view(corr4.shape[0], -1, corr4.shape[3], corr4.shape[4])
        corr4 = corr4 / c14.shape[1]
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)


        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = corr3.view(corr3.shape[0], -1, corr3.shape[3], corr3.shape[4])
        corr3 = corr3 / c13.shape[1]
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)


        warp2 = self.warp(c22, up_flow3*5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = corr2.view(corr2.shape[0], -1, corr2.shape[3], corr2.shape[4])
        corr2 = corr2 / c12.shape[1]
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        flow_up = self.upsample1(flow2*self.args.div_flow)

        outputs = {}
        if self.training:
            outputs['flow_preds'] = [flow2, flow3, flow4, flow5, flow6]
            outputs['flows'] = flow_up[:, None]
        else:
            outputs['flows'] = flow_up[:, None]
        return outputs
