import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DomainNorm2(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=True)
        self.l2 = l2

    def forward(self, x):
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        x = self.normalize(x)
        return x


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        bn=True,
        l2=True,
        relu=True,
        **kwargs,
    ):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        self.l2 = l2
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = DomainNorm(channel=out_channels, l2=self.l2)

    #            self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        concat=True,
        bn=True,
        relu=True,
        kernel=None,
    ):
        super(Conv2x, self).__init__()
        self.concat = concat
        if kernel is not None:
            self.kernel = kernel
        # elif deconv and is_3d:
        #    kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(
            in_channels,
            out_channels,
            deconv,
            is_3d,
            bn=True,
            relu=True,
            kernel_size=kernel,
            stride=2,
            padding=1,
        )

        if self.concat:
            self.conv2 = BasicConv(
                out_channels * 2,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x, rem):
        x = self.conv1(x)
        # print(x.shape, rem.shape)
        assert x.size() == rem.size(), [x.size(), rem.size()]
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(
                nn.BatchNorm3d(channels), nn.ReLU(inplace=True)
            )
            self.conv_refine = BasicConv(
                channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False
            )
        #            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = nn.BatchNorm3d(channels)

        try:
            from .libs.GANet.modules.GANet import SGA
        except ImportError:
            raise ImportError(
                "ERROR: Could not import libs.GANet required for SeparableFlow."
                + " Go to ptlflow/models/separableflow/ and then install GANet by running: bash compile.sh"
            )
        self.SGA = SGA()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        rem = x
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert x.size() == rem.size()
        x += rem
        return self.relu(x)


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False


class ShiftRegression(nn.Module):
    def __init__(self, max_shift=192):
        super(ShiftRegression, self).__init__()
        self.max_shift = max_shift

    #        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x, max_shift=None):
        if max_shift is not None:
            self.max_shift = max_shift
        assert x.is_contiguous() == True
        with torch.cuda.device_of(x):
            shift = Variable(
                torch.Tensor(
                    np.reshape(
                        np.array(range(-self.max_shift, self.max_shift + 1)),
                        [1, self.max_shift * 2 + 1, 1, 1],
                    )
                ).cuda(),
                requires_grad=False,
            )
            shift = shift.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * shift, dim=1, keepdim=True)
        return out


class ShiftEstimate(nn.Module):
    def __init__(self, max_shift=192, InChannel=24):
        super(ShiftEstimate, self).__init__()
        self.max_shift = int(max_shift / 2)
        self.softmax = nn.Softmax(dim=1)
        self.regression = ShiftRegression(max_shift=self.max_shift + 1)
        self.conv3d_2d = nn.Conv3d(
            InChannel, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=True
        )
        # self.upsample_cost = FilterUpsample()

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/3, W/3, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 3, 3, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(3 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 3 * H, 3 * W)

    def forward(self, x):
        # N, _,  _, H, W = g.size()
        # assert (x.size(3)==H and x.size(4)==W)
        # x = self.upsample_cost(x, g)
        # print(x.size(), g.size())
        x = F.interpolate(
            self.conv3d_2d(x),
            [self.max_shift * 2 + 1, x.size()[3] * 4, x.size()[4] * 4],
            mode="trilinear",
            align_corners=True,
        )
        x = torch.squeeze(x, 1)
        x = self.softmax(x)
        x = self.regression(x)
        x = F.interpolate(
            x, [x.size()[2] * 2, x.size()[3] * 2], mode="bilinear", align_corners=True
        )
        return x * 2


class ShiftEstimate2(nn.Module):
    def __init__(self, max_shift=100, InChannel=24):
        super(ShiftEstimate2, self).__init__()
        self.max_shift = int(max_shift // 4)
        self.softmax = nn.Softmax(dim=1)
        self.regression = ShiftRegression()
        self.conv3d_2d = nn.Conv3d(
            InChannel, 1, kernel_size=3, stride=1, padding=1, bias=True
        )
        # self.upsample_cost = FilterUpsample()

    def forward(self, x, max_shift=None):
        if max_shift is not None:
            assert (max_shift // 8 * 2 + 1) == x.shape[2], [
                x.shape,
                max_shift,
                max_shift // 8 * 2 + 1,
            ]
            # assert(x.size() == rem.size()),[x.size(), rem.size()]
            self.max_shift = max_shift // 4
        x = F.interpolate(
            self.conv3d_2d(x),
            [self.max_shift * 2 + 1, x.size()[3] * 2, x.size()[4] * 2],
            mode="trilinear",
            align_corners=True,
        )
        #        x = self.conv3d_2d(x)
        x = torch.squeeze(x, 1)

        x = self.softmax(x)
        x = self.regression(x, self.max_shift)
        x = F.interpolate(
            x, [x.size()[2] * 4, x.size()[3] * 4], mode="bilinear", align_corners=True
        )

        return x * 4


class CostAggregation(nn.Module):
    def __init__(self, max_shift=400, in_channel=8):
        super(CostAggregation, self).__init__()
        self.max_shift = max_shift
        self.in_channel = in_channel  # t(self.max_shift / 6) * 2 + 1
        self.inner_channel = 8
        self.conv0 = BasicConv(
            self.in_channel,
            self.inner_channel,
            is_3d=True,
            kernel_size=3,
            padding=1,
            relu=True,
        )

        self.conv1a = BasicConv(
            self.inner_channel,
            self.inner_channel * 2,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2a = BasicConv(
            self.inner_channel * 2,
            self.inner_channel * 4,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3a = BasicConv(
            self.inner_channel * 4,
            self.inner_channel * 6,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.deconv1a = Conv2x(
            self.inner_channel * 2,
            self.inner_channel,
            deconv=True,
            is_3d=True,
            relu=True,
        )
        self.deconv2a = Conv2x(
            self.inner_channel * 4, self.inner_channel * 2, deconv=True, is_3d=True
        )
        self.deconv3a = Conv2x(
            self.inner_channel * 6, self.inner_channel * 4, deconv=True, is_3d=True
        )

        self.conv1b = BasicConv(
            self.inner_channel,
            self.inner_channel * 2,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2b = BasicConv(
            self.inner_channel * 2,
            self.inner_channel * 4,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3b = BasicConv(
            self.inner_channel * 4,
            self.inner_channel * 6,
            is_3d=True,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.deconv1b = Conv2x(
            self.inner_channel * 2,
            self.inner_channel,
            deconv=True,
            is_3d=True,
            relu=True,
            kernel=(3, 4, 4),
        )
        self.deconv2b = Conv2x(
            self.inner_channel * 4,
            self.inner_channel * 2,
            deconv=True,
            is_3d=True,
            kernel=(3, 4, 4),
        )
        self.deconv3b = Conv2x(
            self.inner_channel * 6,
            self.inner_channel * 4,
            deconv=True,
            is_3d=True,
            kernel=(3, 4, 4),
        )
        self.shift0 = ShiftEstimate2(self.max_shift, self.inner_channel)
        self.shift1 = ShiftEstimate2(self.max_shift, self.inner_channel)
        self.shift2 = ShiftEstimate2(self.max_shift, self.inner_channel)
        self.sga1 = SGABlock(channels=self.inner_channel, refine=True)
        self.sga2 = SGABlock(channels=self.inner_channel, refine=True)
        self.sga3 = SGABlock(channels=self.inner_channel, refine=True)
        self.sga11 = SGABlock(channels=self.inner_channel * 2, refine=True)
        self.sga12 = SGABlock(channels=self.inner_channel * 2, refine=True)
        self.corr_output = BasicConv(
            self.inner_channel, 1, is_3d=True, kernel_size=3, padding=1, relu=False
        )
        self.corr2cost = Corr2Cost()

    def forward(self, x, g, max_shift=400, is_ux=True):
        x = self.conv0(x)
        x = self.sga1(x, g["sg1"])
        rem0 = x

        if self.training:
            cost = self.corr2cost(x, max_shift // 8, is_ux)
            shift0 = self.shift0(cost, max_shift)

        x = self.conv1a(x)
        x = self.sga11(x, g["sg11"])
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g["sg12"])
        rem1 = x
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g["sg2"])
        rem0 = x
        cost = self.corr2cost(x, max_shift // 8, is_ux)
        if self.training:
            shift1 = self.shift1(cost, max_shift)
        corr = self.corr_output(x)
        rem0 = cost
        x = self.conv1b(cost)
        rem1 = x
        x = self.conv2b(x)
        rem2 = x
        x = self.conv3b(x)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)
        x = self.sga3(x, g["sg3"])
        shift2 = self.shift2(x, max_shift)
        if self.training:
            return shift0, shift1, shift2, corr
        else:
            return shift2, corr


class Corr2Cost(nn.Module):
    def __init__(self):
        super(Corr2Cost, self).__init__()

    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(
            torch.arange(ht, device=device),
            torch.arange(wd, device=device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def bilinear_sampler(self, img, coords, mode="bilinear", mask=False):
        """Wrapper for grid_sample, uses pixel coordinates"""
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        assert torch.unique(ygrid).numel() == 1 and H == 1  # This is a stereo problem

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def forward(self, corr, maxdisp=50, is_ux=True):
        batch, dim, d, h, w = corr.shape
        corr = corr.permute(0, 3, 4, 1, 2).reshape(batch * h * w, dim, 1, d)
        with torch.no_grad():
            coords = self.coords_grid(batch, h, w, corr.device)
            if is_ux:
                coords = coords[:, :1, :, :]
            else:
                coords = coords[:, 1:, :, :]
            dx = torch.linspace(-maxdisp, maxdisp, maxdisp * 2 + 1)
            dx = dx.view(1, 1, 2 * maxdisp + 1, 1).to(corr.device)
            x0 = dx + coords.reshape(batch * h * w, 1, 1, 1)
            y0 = torch.zeros_like(x0)
            # if is_ux:
            coords_lvl = torch.cat([x0, y0], dim=-1)
        # else:
        #     coords_lvl = torch.cat([y0, x0], dim=-1)
        corr = self.bilinear_sampler(corr, coords_lvl)
        # print(corr.shape)
        corr = corr.view(batch, h, w, dim, maxdisp * 2 + 1)
        corr = corr.permute(0, 3, 4, 1, 2).contiguous().float()
        return corr
