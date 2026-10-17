from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            )
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
            conv(self.feat_dim, self.feat_dim, isReLU=False),
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


def weighted_local_sum(x_padded, kernel, size):
    """Aggregate a padded map with spatially-varying per-position weights.

    Computes the same as
        torch.sum(nn.Unfold(size, size)(x_padded) * nn.Unfold(1, 1)(kernel), dim=1)
    but with shifted slices instead of unfold: the Unfold -> Mul ->
    ReduceSum chains crash the OpenCL compiler of the OpenVINO GPU plugin
    when many of them are fused (irr/scopeflow refine blocks).

    x_padded: [B, 1, H + pad, W + pad] (already padded with pad = size // 2)
    kernel: [B, size * size, H, W] weights in row-major kernel order
    Returns [B, 1, H, W].
    """
    b, _, h, w = kernel.shape
    out = x_padded[:, :, 0:h, 0:w] * kernel[:, 0:1]
    k = 1
    for i in range(size):
        for j in range(size):
            if i == 0 and j == 0:
                continue
            out = out + x_padded[:, :, i : i + h, j : j + w] * kernel[:, k : k + 1]
            k += 1
    return out


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
            conv(32, self.kernel_size * self.kernel_size, 3, 1, 1),
        )

        self.softmax_feat = nn.Softmax(dim=1)

    def forward(self, flow, diff_img, feature):
        b, _, h, w = flow.size()

        flow_m = subtract_mean(flow)
        norm2_img = torch.norm(diff_img, p=2, dim=1, keepdim=True)

        feat = self.convs(torch.cat([flow_m, norm2_img, feature], dim=1))
        feat_kernel = self.softmax_feat(-(feat**2))

        flow_x = flow[:, 0].unsqueeze(1)
        flow_y = flow[:, 1].unsqueeze(1)

        flow_out_x = weighted_local_sum(
            self.pad_ftn(flow_x), feat_kernel, self.kernel_size
        )
        flow_out_y = weighted_local_sum(
            self.pad_ftn(flow_y), feat_kernel, self.kernel_size
        )

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
            conv(32, self.kernel_size * self.kernel_size, 3, 1, 1),
        )

        self.softmax_feat = nn.Softmax(dim=1)

    def forward(self, occ, feat1, feat2):
        b, _, h, w = occ.size()

        feat = self.convs(torch.cat([occ, feat1, feat2], dim=1))
        feat_kernel = self.softmax_feat(-(feat**2))

        occ_out = weighted_local_sum(self.pad_ftn(occ), feat_kernel, self.kernel_size)

        return occ_out
