from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import ops

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler

from ..base_model.base_model import BaseModel


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=True):
    return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)


def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w,-1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def downsample_kernel2d(w, device):
    kernel = ((w + 1) - torch.abs(w - torch.arange(w * 2 + 1, dtype=torch.float32, device=device))) / (2 * w + 1)
    kernel = kernel.repeat(w).view(w,-1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w * 2 + 1, w * 2 + 1)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=(factor-1))
    upsamp_img = upsamp_img[:, :, : -1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)


def Downsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    kernel = downsample_kernel2d(factor // 2, img.device)
    upsamp_img = F.conv2d(batch_img, kernel, stride=factor, padding=factor//2)
    upsamp_nom = F.conv2d(torch.ones_like(batch_img), kernel, stride=factor, padding=factor//2)
    _, _, H_up, W_up = upsamp_img.shape
    upsamp_img = upsamp_img.view(B, C, H_up, W_up)
    upsamp_nom = upsamp_nom.view(B, C, H_up, W_up)
    return upsamp_img / upsamp_nom



class MaskFlownet_S(BaseModel):
    pretrained_checkpoints = {
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/maskflownet_s-sintel-79196ff3.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/maskflownet_s-things-a8a8ed9e.ckpt'
    }

    """
    PWC-DC net. add dilation convolution and densenet connections
    """
    def __init__(self, args):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(MaskFlownet_S, self).__init__(
            args=args,
            loss_fn=MultiscaleEpe(
                scales=args.strides,
                weights=args.loss_weights,
                match=args.loss_match,
                eps=args.loss_eps,
                q=args.loss_q
            ),
            output_stride=64)
        self.scale = self.args.div_flow * self.args.flow_multiplier
        self.deform_bias = self.args.deform_bias

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv1c  = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv2c  = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv3c  = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv4c  = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv5c  = conv(128, 128, kernel_size=3, stride=1)
        self.conv6a = conv(128, 196, kernel_size=3, stride=2)
        self.conv6b  = conv(196, 196, kernel_size=3, stride=1)
        self.conv6c  = conv(196, 196, kernel_size=3, stride=1)

        self.corr_fn = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.args.md+1, padding=0)
        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2*self.args.md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow6 = predict_flow(od + dd[4])
        self.pred_mask6 = predict_mask(od + dd[4])
        self.upfeat5 = deconv(od+dd[4], self.args.upfeat_ch[0], kernel_size=4, stride=2, padding=1)
        # self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+128+4
        od = nd+128+18
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow5 = predict_flow(od + dd[4])
        self.pred_mask5 = predict_mask(od + dd[4])
        self.upfeat4 = deconv(od+dd[4], self.args.upfeat_ch[1], kernel_size=4, stride=2, padding=1)
        # self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+96+4
        od = nd+96+18
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(od + dd[4])
        self.pred_mask4 = predict_mask(od + dd[4])
        self.upfeat3 = deconv(od+dd[4], self.args.upfeat_ch[2], kernel_size=4, stride=2, padding=1)
        # self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+64+4
        od = nd+64+18
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow3 = predict_flow(od + dd[4])
        self.pred_mask3 = predict_mask(od + dd[4])
        self.upfeat2 = deconv(od+dd[4], self.args.upfeat_ch[3], kernel_size=4, stride=2, padding=1)
        # self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+32+4
        od = nd+32+18
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow2 = predict_flow(od + dd[4])

        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        # self.upfeat5 = deconv()

        self.deform5 = deformable_conv(128, 128)
        self.deform4 = deformable_conv(96, 96)
        self.deform3 = deformable_conv(64, 64)
        self.deform2 = deformable_conv(32, 32)

        self.conv5f = conv(16, 128, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv4f = conv(16, 96, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv3f = conv(16, 64, kernel_size=3, stride=1, padding=1, activation=False)
        self.conv2f = conv(16, 32, kernel_size=3, stride=1, padding=1, activation=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        parser.add_argument('--md', type=int, default=4)
        parser.add_argument('--flow_multiplier', type=float, default=1.0)
        parser.add_argument('--deform_bias', type=bool, default=True)
        parser.add_argument('--strides', type=int, nargs='+', default=(64, 32, 16, 8, 4))
        parser.add_argument('--upfeat_ch', type=int, nargs='+', default=(16, 16, 16, 16))
        parser.add_argument('--loss_weights', type=float, nargs='+', default=(.005, .01, .02, .08, .32))
        parser.add_argument('--loss_match', type=str, default='upsampling', choices=('upsampling', 'downsampling'))
        parser.add_argument('--loss_eps', type=float, default=1e-8)
        parser.add_argument('--loss_q', type=float, default=None)
        return parser

    def corr(self, f1, f2):
        corr = self.corr_fn(f1, f2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / f2.shape[1]
        return corr

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

        device = x.device
        grid = grid.to(device)
        # vgrid = Variable(grid) + flo
        vgrid = Variable(grid) + torch.flip(flo, [1])

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        # vgrid = vgrid.permute(0,2,3,1).clamp(-1.1, 1.1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask


    def forward(self, inputs):
        im1 = inputs['images'][:, 0]
        im2 = inputs['images'][:, 1]

        c11 = self.conv1c(self.conv1b(self.conv1a(im1)))
        c21 = self.conv1c(self.conv1b(self.conv1a(im2)))
        c12 = self.conv2c(self.conv2b(self.conv2a(c11)))
        c22 = self.conv2c(self.conv2b(self.conv2a(c21)))
        c13 = self.conv3c(self.conv3b(self.conv3a(c12)))
        c23 = self.conv3c(self.conv3b(self.conv3a(c22)))
        c14 = self.conv4c(self.conv4b(self.conv4a(c13)))
        c24 = self.conv4c(self.conv4b(self.conv4a(c23)))
        c15 = self.conv5c(self.conv5b(self.conv5a(c14)))
        c25 = self.conv5c(self.conv5b(self.conv5a(c24)))
        c16 = self.conv6c(self.conv6b(self.conv6a(c15)))
        c26 = self.conv6c(self.conv6b(self.conv6a(c25)))


        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)


        x = torch.cat((self.conv6_0(corr6), corr6),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = self.pred_flow6(x)
        mask6 = self.pred_mask6(x)

        feat5 = self.leakyRELU(self.upfeat5(x))
        flow5 = Upsample(flow6, 2)
        mask5 = Upsample(mask6, 2)
        warp5 = (flow5*self.scale/self.args.strides[1]).unsqueeze(1)
        warp5 = torch.repeat_interleave(warp5, 9, 1)
        S1, S2, S3, S4, S5 = warp5.shape
        warp5 = warp5.view(S1, S2*S3, S4, S5)
        warp5 = self.deform5(c25, warp5)
        tradeoff5 = feat5
        warp5 = (warp5 * torch.sigmoid(mask5)) + self.conv5f(tradeoff5)
        warp5 = self.leakyRELU(warp5)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, feat5, flow5), 1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = flow5 + self.pred_flow5(x)
        mask5 = self.pred_mask5(x)

        feat4 = self.leakyRELU(self.upfeat4(x))
        flow4 = Upsample(flow5, 2)
        mask4 = Upsample(mask5, 2)
        warp4 = (flow4*self.scale/self.args.strides[2]).unsqueeze(1)
        warp4 = torch.repeat_interleave(warp4, 9, 1)
        S1, S2, S3, S4, S5 = warp4.shape
        warp4 = warp4.view(S1, S2*S3, S4, S5)
        warp4 = self.deform4(c24, warp4)
        tradeoff4 = feat4
        warp4 = (warp4 * torch.sigmoid(mask4)) + self.conv4f(tradeoff4)
        warp4 = self.leakyRELU(warp4)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, feat4, flow4), 1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = flow4 + self.pred_flow4(x)
        mask4 = self.pred_mask4(x)

        feat3 = self.leakyRELU(self.upfeat3(x))
        flow3 = Upsample(flow4, 2)
        mask3 = Upsample(mask4, 2)
        warp3 = (flow3*self.scale/self.args.strides[3]).unsqueeze(1)
        warp3 = torch.repeat_interleave(warp3, 9, 1)
        S1, S2, S3, S4, S5 = warp3.shape
        warp3 = warp3.view(S1, S2*S3, S4, S5)
        warp3 = self.deform3(c23, warp3)
        tradeoff3 = feat3
        warp3 = (warp3 * torch.sigmoid(mask3)) + self.conv3f(tradeoff3)
        warp3 = self.leakyRELU(warp3)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, feat3, flow3), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = flow3 + self.pred_flow3(x)
        mask3 = self.pred_mask3(x)

        feat2 = self.leakyRELU(self.upfeat2(x))
        flow2 = Upsample(flow3, 2)
        mask2 = Upsample(mask3, 2)
        warp2 = (flow2*self.scale/self.args.strides[4]).unsqueeze(1)
        warp2 = torch.repeat_interleave(warp2, 9, 1)
        S1, S2, S3, S4, S5 = warp2.shape
        warp2 = warp2.view(S1, S2*S3, S4, S5)
        warp2 = self.deform2(c22, warp2)
        tradeoff2 = feat2
        warp2 = (warp2 * torch.sigmoid(mask2)) + self.conv2f(tradeoff2)
        warp2 = self.leakyRELU(warp2)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, feat2, flow2), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = flow2 + self.pred_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        predictions = [flow.flip(1) * self.scale for flow in [flow6, flow5, flow4, flow3, flow2]]
        occlusion_masks = []
        occlusion_masks.append(1 - torch.sigmoid(mask2))
        c1s = [c11, c12, c13, c14, c15, c16]
        c2s = [c21, c12, c13, c24, c25, c26]
        flows = [flow6, flow5, flow4, flow3, flow2]
        mask0 = Upsample(mask2, 4)
        mask0 = torch.sigmoid(mask0) - 0.5
        c30 = im1
        c40 = self.warp(im2, Upsample(flow2, 4)*self.scale)
        c30 = torch.cat((c30, torch.zeros_like(mask0)), 1)
        c40 = torch.cat((c40, mask0), 1)
        srcs = [c1s, c2s, flows, c30, c40]

        output = {
            'flows': F.interpolate(predictions[-1], size=im1.shape[-2:], mode='bilinear', align_corners=True)[:, None],
            'occs': F.interpolate(occlusion_masks[-1], size=im1.shape[-2:], mode='bilinear', align_corners=True)[:, None],
            'srcs': srcs
        }
        if self.training:
            output['flow_preds'] = predictions
            output['occ_preds'] = occlusion_masks

        return output


class MaskFlownet(BaseModel):
    pretrained_checkpoints = {
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/maskflownet-kitti-d2ccb09b.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/maskflownet-sintel-c52423aa.ckpt'
    }

    def __init__(self, args):
        super(MaskFlownet, self).__init__(
            args=args,
            loss_fn=MultiscaleEpe(
                scales=args.strides,
                weights=args.loss_weights,
                match=args.loss_match,
                eps=args.loss_eps,
                q=args.loss_q
            ),
            output_stride=64)
        self.scale = self.args.div_flow * self.args.flow_multiplier
        self.deform_bias = self.args.deform_bias

        self.MaskFlownet_S = MaskFlownet_S(args)
        self.activate = nn.LeakyReLU(0.1)

        self.conv1x = conv(4,   16, stride=2)
        self.conv1y = conv(16,  16, stride=1)
        self.conv1z = conv(16,  16, stride=1)
        self.conv2x = conv(16,  32, stride=2)
        self.conv2y = conv(32,  32, stride=1)
        self.conv2z = conv(32,  32, stride=1)
        self.conv3x = conv(32,  64, stride=2)
        self.conv3y = conv(64,  64, stride=1)
        self.conv3z = conv(64,  64, stride=1)
        self.conv4x = conv(64,  96, stride=2)
        self.conv4y = conv(96,  96, stride=1)
        self.conv4z = conv(96,  96, stride=1)
        self.conv5x = conv(96,  128, stride=2)
        self.conv5y = conv(128, 128, stride=1)
        self.conv5z = conv(128, 128, stride=1)
        self.conv6x = conv(128, 196, stride=2)
        self.conv6y = conv(196, 196, stride=1)
        self.conv6z = conv(196, 196, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr_fn = SpatialCorrelationSampler(kernel_size=1, patch_size=2*self.args.md2+1, padding=0)
        # self.corr    = Correlation(pad_size=self.md, kernel_size=1, max_displacement=self.md, stride1=1, stride2=1, corr_multiply=1)

        nd = (2*self.args.md2+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd+nd+2
        self.conv6_0 = conv(od,       128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1], 96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2], 64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3], 32,  kernel_size=3, stride=1)
        self.pred_flow6 = predict_flow(od + dd[4])
        self.upfeat5 = deconv(od+dd[4], self.args.upfeat_ch[0], kernel_size=4, stride=2, padding=1)

        # od = nd+128+4
        od = nd+nd+128+16+2+2
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow5 = predict_flow(od + dd[4])
        self.upfeat4 = deconv(od+dd[4], self.args.upfeat_ch[1], kernel_size=4, stride=2, padding=1)
        # self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+96+4
        # od = nd+96+18
        od = nd+nd+96+16+2+2
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow4 = predict_flow(od + dd[4])
        self.upfeat3 = deconv(od+dd[4], self.args.upfeat_ch[2], kernel_size=4, stride=2, padding=1)
        # self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+64+4
        # od = nd+64+18
        od = nd+nd+64+16+2+2
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow3 = predict_flow(od + dd[4])
        self.upfeat2 = deconv(od+dd[4], self.args.upfeat_ch[3], kernel_size=4, stride=2, padding=1)
        # self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd+32+4
        # od = nd+32+18
        od = nd+nd+32+16+2+2
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.pred_flow2 = predict_flow(od + dd[4])

        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        # self.upfeat5 = deconv()

        self.deform6 = deformable_conv(196, 196)
        self.deform5 = deformable_conv(128, 128)
        self.deform4 = deformable_conv(96, 96)
        self.deform3 = deformable_conv(64, 64)
        self.deform2 = deformable_conv(32, 32)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        parser.add_argument('--md', type=int, default=4)
        parser.add_argument('--md2', type=int, default=2)
        parser.add_argument('--flow_multiplier', type=float, default=1.0)
        parser.add_argument('--deform_bias', type=bool, default=True)
        parser.add_argument('--strides', type=int, nargs='+', default=(64, 32, 16, 8, 4))
        parser.add_argument('--upfeat_ch', type=int, nargs='+', default=(16, 16, 16, 16))
        parser.add_argument('--loss_weights', type=float, nargs='+', default=(.005, .01, .02, .08, .32))
        parser.add_argument('--loss_match', type=str, default='upsampling', choices=('upsampling', 'downsampling'))
        parser.add_argument('--loss_eps', type=float, default=1e-8)
        parser.add_argument('--loss_q', type=float, default=None)
        return parser

    def corr(self, f1, f2):
        corr = self.corr_fn(f1, f2)
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / f2.shape[1]
        return corr

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
            grid = grid.cuda()
        # vgrid = Variable(grid) + flo
        vgrid = Variable(grid) + torch.flip(flo, [1])

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        # vgrid = vgrid.permute(0,2,3,1)
        vgrid = vgrid.permute(0,2,3,1).clamp(-1.1, 1.1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask

    def forward(self, inputs):
        mfns_out = self.MaskFlownet_S(inputs)
        srcs = mfns_out['srcs']
        c1s, c2s, flows, c30, c40 = srcs
        c11, c12, c13, c14, c15, c16 = c1s
        c21, c22, c23, c24, c25, c26 = c2s

        c31 = self.conv1z(self.conv1y(self.conv1x(c30)))
        c32 = self.conv2z(self.conv2y(self.conv2x(c31)))
        c33 = self.conv3z(self.conv3y(self.conv3x(c32)))
        c34 = self.conv4z(self.conv4y(self.conv4x(c33)))
        c35 = self.conv5z(self.conv5y(self.conv5x(c34)))
        c36 = self.conv6z(self.conv6y(self.conv6x(c35)))

        c41 = self.conv1z(self.conv1y(self.conv1x(c40)))
        c42 = self.conv2z(self.conv2y(self.conv2x(c41)))
        c43 = self.conv3z(self.conv3y(self.conv3x(c42)))
        c44 = self.conv4z(self.conv4y(self.conv4x(c43)))
        c45 = self.conv5z(self.conv5y(self.conv5x(c44)))
        c46 = self.conv6z(self.conv6y(self.conv6x(c45)))

        flow6 = flows[0]

        warp6u = (flow6*self.scale/self.args.strides[0]).unsqueeze(1)
        warp6u = torch.repeat_interleave(warp6u, 9, 1)
        S1, S2, S3, S4, S5 = warp6u.shape
        warp6u = warp6u.view(S1, S2*S3, S4, S5)
        warp6u = self.deform6(c26, warp6u)
        warp6u = self.leakyRELU(warp6u)
        corr6u = self.leakyRELU(self.corr(c16, warp6u))
        warp6v = c46
        corr6v = self.leakyRELU(self.corr(c36, warp6v))
        x = torch.cat((corr6u, corr6v, flow6),1)
        x = torch.cat((self.conv6_0(x), x),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((self.conv6_2(x), x),1)
        x = torch.cat((self.conv6_3(x), x),1)
        x = torch.cat((self.conv6_4(x), x),1)
        flow6 = flow6 + self.pred_flow6(x)

        feat5 = self.leakyRELU(self.upfeat5(x))
        flow5 = Upsample(flow6, 2)
        warp5u = (flow5*self.scale/self.args.strides[1]).unsqueeze(1)
        warp5u = torch.repeat_interleave(warp5u, 9, 1)
        S1, S2, S3, S4, S5 = warp5u.shape
        warp5u = warp5u.view(S1, S2*S3, S4, S5)
        warp5u = self.deform5(c25, warp5u)
        warp5u = self.leakyRELU(warp5u)
        corr5u = self.leakyRELU(self.corr(c15, warp5u))
        warp5v = c45
        corr5v = self.leakyRELU(self.corr(c35, warp5v))
        x = torch.cat((c15, feat5, corr5u, corr5v, flow5, flows[1]),1)
        x = torch.cat((self.conv5_0(x), x),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((self.conv5_2(x), x),1)
        x = torch.cat((self.conv5_3(x), x),1)
        x = torch.cat((self.conv5_4(x), x),1)
        flow5 = flow5 + self.pred_flow5(x)

        feat4 = self.leakyRELU(self.upfeat4(x))
        flow4 = Upsample(flow5, 2)
        warp4u = (flow4*self.scale/self.args.strides[2]).unsqueeze(1)
        warp4u = torch.repeat_interleave(warp4u, 9, 1)
        S1, S2, S3, S4, S5 = warp4u.shape
        warp4u = warp4u.view(S1, S2*S3, S4, S5)
        warp4u = self.deform4(c24, warp4u)
        warp4u = self.leakyRELU(warp4u)
        corr4u = self.leakyRELU(self.corr(c14, warp4u))
        warp4v = c44
        corr4v = self.leakyRELU(self.corr(c34, warp4v))
        x = torch.cat((c14, feat4, corr4u, corr4v, flow4, flows[2]),1)
        x = torch.cat((self.conv4_0(x), x),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        flow4 = flow4 + self.pred_flow4(x)

        feat3 = self.leakyRELU(self.upfeat3(x))
        flow3 = Upsample(flow4, 2)
        warp3u = (flow3*self.scale/self.args.strides[3]).unsqueeze(1)
        warp3u = torch.repeat_interleave(warp3u, 9, 1)
        S1, S2, S3, S4, S5 = warp3u.shape
        warp3u = warp3u.view(S1, S2*S3, S4, S5)
        warp3u = self.deform3(c23, warp3u)
        warp3u = self.leakyRELU(warp3u)
        corr3u = self.leakyRELU(self.corr(c13, warp3u))
        warp3v = c43
        corr3v = self.leakyRELU(self.corr(c33, warp3v))
        x = torch.cat((c13, feat3, corr3u, corr3v, flow3, flows[3]),1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        flow3 = flow3 + self.pred_flow3(x)

        feat2 = self.leakyRELU(self.upfeat2(x))
        flow2 = Upsample(flow3, 2)
        warp2u = (flow2*self.scale/self.args.strides[4]).unsqueeze(1)
        warp2u = torch.repeat_interleave(warp2u, 9, 1)
        S1, S2, S3, S4, S5 = warp2u.shape
        warp2u = warp2u.view(S1, S2*S3, S4, S5)
        warp2u = self.deform2(c22, warp2u)
        warp2u = self.leakyRELU(warp2u)
        corr2u = self.leakyRELU(self.corr(c12, warp2u))
        warp2v = c42
        corr2v = self.leakyRELU(self.corr(c32, warp2v))
        x = torch.cat((c12, feat2, corr2u, corr2v, flow2, flows[4]),1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        flow2 = flow2 + self.pred_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        preds = [flow.flip(1) * self.scale for flow in [flow6, flow5, flow4, flow3, flow2]]
        visuals = []
        visuals.append(flow2[:,:1])

        output = {
            'flows': F.interpolate(preds[-1], size=inputs['images'].shape[-2:], mode='bilinear', align_corners=True)[:, None],
            'visuals': visuals[-1][:, None]
        }
        if self.training:
            output['flow_preds'] = preds

        return output


class EpeLoss(nn.Module):
    def __init__(self, eps = 0):
        super(EpeLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        return loss.view(loss.shape[0], -1).mean(1)


class EpeLossWithMask(nn.Module):
    def __init__(self, eps=1e-8, q=None):
        super(EpeLossWithMask, self).__init__()
        self.eps = eps
        self.q = q

    def forward(self, pred, label, mask):
        if self.q is not None:
            loss = ((pred - label).abs().sum(1) + self.eps) ** self.q
        else:
            loss = ((pred - label).pow(2).sum(1) + self.eps).sqrt()
        loss = loss * mask.squeeze(1)
        loss = loss.view(loss.shape[0], -1).sum(1) / mask.view(mask.shape[0], -1).sum(1)
        return loss


class MultiscaleEpe(nn.Module):
    def __init__(self, scales, weights, match, eps = 1e-8, q = None):
        super(MultiscaleEpe, self).__init__()

        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q

    def forward(self, output, target):
        flow = target['flows'][:, 0]
        mask = target['valids'][:, 0]
        predictions = output['flow_preds']
        losses = 0
        if self.match == 'upsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(Upsample(p, s), flow, mask) * w
        elif self.match == 'downsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                losses += EpeLossWithMask(eps=self.eps, q=self.q)(p, Downsample(flow, s), Downsample(mask, s)) * w
        else:
            raise NotImplementedError
        return losses