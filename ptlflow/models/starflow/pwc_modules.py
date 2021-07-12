import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

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


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)


def rescale_flow(flow, div_flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im / div_flow)
        v_scale = float(flow.size(2) / height_im / div_flow)
    else:
        u_scale = float(width_im * div_flow / flow.size(3))
        v_scale = float(height_im * div_flow / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u = u * u_scale
    v = v * v_scale

    return torch.cat([u, v], dim=1)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False)
    if x.is_cuda:
            grids_cuda = grids_cuda.cuda()
    return grids_cuda


class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid, align_corners=True)

        mask = torch.ones(x.size(), requires_grad=False)
        if x.is_cuda:
            mask = mask.cuda()
        mask = tf.grid_sample(mask, grid, align_corners=True)
        mask = (mask >= 1.0).float()

        return x_warp * mask

class OpticalFlowEstimator(nn.Module):
    def __init__(self, ch_in):
        super(OpticalFlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_last = conv(32, 2, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out

class OcclusionEstimator(nn.Module):
    def __init__(self, ch_in):
        super(OcclusionEstimator, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_last = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_intm = self.convs(x)
        return x_intm, self.conv_last(x_intm)


class OccEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(OccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 1, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class OccContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(OccContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 1, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)

# -------------------------------------------

class FlowAndOccEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowAndOccEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 3, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out[:,:2,:,:], x_out[:,2,:,:].unsqueeze(1)


class FlowAndOccContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(FlowAndOccContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 3, isReLU=False)
        )

    def forward(self, x):
        x_out = self.convs(x)
        return x_out[:,:2,:,:], x_out[:,2,:,:].unsqueeze(1)