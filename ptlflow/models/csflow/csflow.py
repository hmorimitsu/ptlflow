from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

    def forward(self, outputs, inputs):
        """ Loss function defined over sequence of flow predictions """

        flow_preds = outputs['flow_preds']
        flow_gt = inputs['flows'][:, 0]
        valid = inputs['valids'][:, 0]
        
        n_predictions = len(flow_preds)
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        for i in range(n_predictions):
            i_weight = self.gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        return flow_loss


class CSFlow(BaseModel):
    pretrained_checkpoints = {
        'chairs': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/csflow-chairs-458a9436.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/csflow-things-ebdd403b.ckpt',
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/csflow-kitti-dc66357a.ckpt'
    }

    def __init__(self,
                 args: Namespace) -> None:
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='instance', dropout=args.dropout)

        self.cnet = BasicEncoder(
            output_dim=hdim + cdim,
            norm_fn='batch',
            dropout=args.dropout)

        self.strip_corr_block_v2 = StripCrossCorrMap_v2(
            in_chan=256, out_chan=256)
        self.update_block = BasicUpdateBlock(
            self.args, hidden_dim=hdim)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8)
        parser.add_argument('--max_flow', type=float, default=400.0)
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--gen_fmap', action='store_true')
        parser.add_argument('--skip_encode', action='store_true')
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids.
        flow = coords1 - coords0, Modified by Hao
        """
        N, C, H, W = img.shape

        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex
        combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, inputs, flow_init=None):
        """ Estimate optical flow between pair of frames """
        image1 = inputs['images'][:, 0]
        image2 = inputs['images'][:, 1]
        """Estimate optical flow between pair of frames."""

        if not self.args.skip_encode:
            # Modified, take image pairs as input
            image1 = 2 * image1 - 1.0
            image2 = 2 * image2 - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()

            # run the feature network
            fmap1, fmap2 = self.fnet([image1, image2])

            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
        else:
            fmap1 = image1 * 255.0
            fmap2 = image2 * 255.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the context network
        if not self.args.skip_encode:
            cnet = self.cnet(image1)

            if not self.training:
                if self.args.gen_fmap:
                    return fmap1, fmap2, cnet
        else:
            cnet = inputs['images'][:, 2] * 255.0

        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        strip_coor_map, strip_corr_map_w, strip_corr_map_h = self.strip_corr_block_v2(
            [fmap1, fmap2])
        corr_fn = CorrBlock_v2(
            fmap1, fmap2, strip_coor_map, radius=self.args.corr_radius)

        if not self.args.skip_encode:
            coords0, coords1 = self.initialize_flow(image1)
        else:
            b, c, h, w = fmap1.shape
            image1 = torch.zeros(b, c, 8 * h, 8 * w).cuda()
            coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        # init flow with regression before GRU iters
        corr_w_act = torch.nn.functional.softmax(
            strip_corr_map_w, dim=3)  # B H1 W1 1 W2
        corr_h_act = torch.nn.functional.softmax(
            strip_corr_map_h, dim=4)  # B H1 W1 H2 1

        flo_v = corr_w_act.mul(strip_corr_map_w)  # B H1 W1 1 W2
        flo_u = corr_h_act.mul(strip_corr_map_h)  # B H1 W1 H2 1

        flow_v = torch.sum(flo_v, dim=4).squeeze(dim=3)  # B H1 W1
        flow_u = torch.sum(flo_u, dim=3).squeeze(dim=3)  # B H1 W1

        corr_init = torch.stack((flow_u, flow_v), dim=1)  # B 2 H1 W1

        coords1 = coords1.detach()
        coords1 = coords1 + corr_init

        # add loss
        flow_up = upflow8(coords1 - coords0)
        flow_predictions.append(flow_up)

        for itr in range(self.args.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(
                net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if self.training:
            outputs = {
                'flows': flow_up[:, None],
                'flow_preds': flow_predictions
            }
        else:
            outputs = {
                'flows': flow_up[:, None],
                'flow_small': coords1 - coords0
            }
            
        return outputs


class StripCrossCorrMap_v2(nn.Module):
    """Strip Cross Corr Augmentation Module by Hao, version2.0"""

    def __init__(self, in_chan=256, out_chan=256, *args, **kwargs):
        super(StripCrossCorrMap_v2, self).__init__()
        self.conv1_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv2_2 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        fmap1, fmap2 = x

        # vertical query map
        fmap1_w = self.conv1_1(fmap1)  # B, 64, H, W
        batchsize, c_middle, h, w = fmap1_w.size()
        fmap1_w = fmap1_w.view(batchsize, c_middle, -1)

        # horizontal query map
        fmap1_h = self.conv1_2(fmap1)  # B, 64, H, W
        batchsize, c_middle, h, w = fmap1_h.size()
        fmap1_h = fmap1_h.view(batchsize, c_middle, -1)

        # vertical striping map
        fmap2_w = self.conv2_1(fmap2)  # B, 64, H, W
        fmap2_w = F.avg_pool2d(fmap2_w, [h, 1])
        fmap2_w = fmap2_w.view(batchsize, c_middle, -1).permute(0, 2, 1)

        # horizontal striping map
        fmap2_h = self.conv2_2(fmap2)  # B, 64, H, W
        fmap2_h = F.avg_pool2d(fmap2_h, [1, w])
        fmap2_h = fmap2_h.view(batchsize, c_middle, -1).permute(0, 2, 1)

        # cross strip corr map
        strip_corr_map_w = torch.bmm(fmap2_w, fmap1_w).\
            view(batchsize, w, h, w, 1).permute(0, 2, 3, 4, 1)      # B H1 W1 1 W2
        strip_corr_map_h = torch.bmm(fmap2_h, fmap1_h).\
            view(batchsize, h, h, w, 1).permute(0, 2, 3, 1, 4)      # B H1 W1 H2 1

        return (strip_corr_map_w + strip_corr_map_h).view(
            batchsize, h, w, 1, h, w), strip_corr_map_w, strip_corr_map_h

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(nn.Module):
    """Conv with BN and ReLU, used for Strip Corr Module"""

    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 *args,
                 **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = torch.nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SmallUpdateBlock(nn.Module):

    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    """Modified by Hao, support for CSFlow"""

    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_v2(args)
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        from torch.nn.modules.utils import _pair
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(
            self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)

        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(
            self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class CorrBlock_v2:
    """Corr Block, modified by Hao, concat SC with 4D corr"""

    def __init__(self,
                 fmap1,
                 fmap2,
                 strip_coor_map=None,
                 num_levels=4,
                 radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock_v2.corr(fmap1, fmap2)

        if strip_coor_map is not None:
            # strip correlation augmentation with concat
            corr = torch.cat((corr, strip_coor_map), dim=3)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

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
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for grid_sample, uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(
        flow, size=new_size, mode=mode, align_corners=True)


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicMotionEncoder_v2(nn.Module):
    """Get Motion Feature from CSFlow, by Hao"""

    def __init__(self, args):
        super(BasicMotionEncoder_v2, self).__init__()
        # double cor_plances due to concat aug
        cor_planes = 2 * (args.corr_levels * (2 * args.corr_radius + 1)**2)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallMotionEncoder(nn.Module):

    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h
