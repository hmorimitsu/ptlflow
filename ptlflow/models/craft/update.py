import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import Aggregate
from .setrans import ExpandedFeatTrans

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        # When both f1 and f2 are applied SS-Trans, corr_multiplier = 2. 
        # Otherwise corr_multiplier = 1.
        cor_planes = args.corr_levels * args.corr_multiplier * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net_feat, inp_feat, corr, flow, upsample=True):
        # motion_features: (256+2)-dimensional.
        motion_features = self.encoder(flow, corr)
        inp_feat = torch.cat([inp_feat, motion_features], dim=1)

        net_feat = self.gru(net_feat, inp_feat)
        delta_flow = self.flow_head(net_feat)

        # scale mask to balance gradients
        mask = .25 * self.mask(net_feat)
        return net_feat, mask, delta_flow


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.use_setrans = args.use_setrans
        if self.use_setrans:
            self.intra_trans_config = args.intra_trans_config
            self.aggregator = ExpandedFeatTrans(self.intra_trans_config, 'Motion Aggregator')
        else:
            # Aggregate is attention with a (learnable-weighted) skip connection, without FFN.
            self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp, corr, flow, attention):
        # encoder: BasicMotionEncoder
        # corr: [3, 676, 50, 90]
        motion_features = self.encoder(flow, corr)
        # motion_features: 128-dim
        if self.use_setrans:
            # attention is multi-mode. ExpandedFeatTrans takes multi-mode attention.
            B, C, H, W = motion_features.shape
            motion_features_3d = motion_features.view(B, C, H*W).permute(0, 2, 1)
            # motion_features_3d: [1, 7040, 128], attention: [1, 4, 7040, 7040]
            motion_features_global_3d = self.aggregator(motion_features_3d, attention)
            motion_features_global = motion_features_global_3d.view(B, H, W, C).permute(0, 3, 1, 2)
        else:
            # attention: [8, 1, 2852, 2852]. motion_features: [8, 128, 46, 62].
            motion_features_global = self.aggregator(attention, motion_features)
            
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



