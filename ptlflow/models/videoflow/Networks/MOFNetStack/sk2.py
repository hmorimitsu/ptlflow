import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import Aggregate

class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius*2+1)**2*args.cost_heads_num*args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1_ = nn.Conv2d(4, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-4, k_conv=args.k_conv)


    def forward(self, forward_flow, backward_flow, forward_corr, backward_corr):
        cor = F.gelu(torch.cat([self.convc1(forward_corr), self.convc1(backward_corr)], dim=1))

        cor = self.convc2(cor)

        flow = torch.cat([forward_flow, backward_flow], dim=1)
        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args

        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]
        
        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 4, k_conv=args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, forward_corr, backward_corr, forward_flow, backward_flow, coords0, attention, bs):

        motion_features = self.encoder(forward_flow, backward_flow, forward_corr, backward_corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow 
