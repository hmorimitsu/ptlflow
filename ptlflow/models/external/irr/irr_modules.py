from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


def upsample_factor2(inputs, target_as):
    inputs = tf.interpolate(inputs, scale_factor=2, mode="nearest")
    _, _, h, w = target_as.size()
    if inputs.size(2) != h or inputs.size(3) != w:
        return tf.interpolate(inputs, [h, w], mode="bilinear", align_corners=False)
    else:
        return inputs


class OccUpsampleNetwork(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(OccUpsampleNetwork, self).__init__()

        self.feat_dim = 32
        self.init_conv = conv(ch_in, self.feat_dim)

        self.res_convs = nn.Sequential(
            conv(self.feat_dim, self.feat_dim),
            conv(self.feat_dim, self.feat_dim, isReLU=False)
        )
        self.res_end_conv = conv(self.feat_dim, self.feat_dim)
        self.mul_const = 0.1

        self.out_convs = conv(self.feat_dim, ch_out)

    def forward(self, occ, x):
        occ = upsample_factor2(occ, x)
        x_in = torch.cat([occ, x], dim=1)
        x_init = self.init_conv(x_in)
        x_res = x_init
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_res = x_res + self.res_convs(x_res) * self.mul_const
        x_init = x_init + self.res_end_conv(x_res)

        return self.out_convs(x_init) + occ


def subtract_mean(input):
    return input - input.mean(2).mean(2).unsqueeze(2).unsqueeze(2).expand_as(input)

    
class RefineFlow(nn.Module):
    def __init__(self, ch_in):
        super(RefineFlow, self).__init__()

        self.kernel_size = 3
        self.pad_size = 1
        self.pad_ftn = nn.ReplicationPad2d(self.pad_size)

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 64, 3, 1, 1),
            conv(64, 64, 3, 1, 1),
            conv(64, 32, 3, 1, 1),
            conv(32, 32, 3, 1, 1),
            conv(32, self.kernel_size * self.kernel_size, 3, 1, 1)
        )

        self.softmax_feat = nn.Softmax(dim=1)
        self.unfold_flow = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size))
        self.unfold_kernel = nn.Unfold(kernel_size=(1, 1))

    def forward(self, flow, diff_img, feature):
        b, _, h, w = flow.size()

        flow_m = subtract_mean(flow)
        norm2_img = torch.norm(diff_img, p=2, dim=1, keepdim=True)

        feat = self.convs(torch.cat([flow_m, norm2_img, feature], dim=1))
        feat_kernel = self.softmax_feat(-feat ** 2)

        flow_x = flow[:, 0].unsqueeze(1)
        flow_y = flow[:, 1].unsqueeze(1)

        flow_x_unfold = self.unfold_flow(self.pad_ftn(flow_x))
        flow_y_unfold = self.unfold_flow(self.pad_ftn(flow_y))
        feat_kernel_unfold = self.unfold_kernel(feat_kernel)

        flow_out_x = torch.sum(flow_x_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)
        flow_out_y = torch.sum(flow_y_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)

        return torch.cat([flow_out_x, flow_out_y], dim=1)


class RefineOcc(nn.Module):
    def __init__(self, ch_in):
        super(RefineOcc, self).__init__()

        self.kernel_size = 3
        self.pad_size = 1
        self.pad_ftn = nn.ReplicationPad2d(self.pad_size)

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 64, 3, 1, 1),
            conv(64, 64, 3, 1, 1),
            conv(64, 32, 3, 1, 1),
            conv(32, 32, 3, 1, 1),
            conv(32, self.kernel_size * self.kernel_size, 3, 1, 1)
        )

        self.softmax_feat = nn.Softmax(dim=1)
        self.unfold_occ = nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size))
        self.unfold_kernel = nn.Unfold(kernel_size=(1, 1))

    def forward(self, occ, feat1, feat2):
        b, _, h, w = occ.size()

        feat = self.convs(torch.cat([occ, feat1, feat2], dim=1))
        feat_kernel = self.softmax_feat(-feat ** 2)

        occ_unfold = self.unfold_occ(self.pad_ftn(occ))
        feat_kernel_unfold = self.unfold_kernel(feat_kernel)

        occ_out = torch.sum(occ_unfold * feat_kernel_unfold, dim=1).unsqueeze(1).view(b, 1, h, w)

        return occ_out