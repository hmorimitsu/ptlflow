import torch
import torch.nn as nn
import torch.nn.functional as F
from .gma import Aggregate

from ...utils.utils import bilinear_sampler

class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.3*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.3*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.3*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.3*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x

class velocity_update_block(nn.Module):
    def __init__(self, C_in=43+128+43, C_out=43, C_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_out, 3, padding=1),
        )
    def forward(self, x):
        return self.mlp(x)


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius*2+1)**2*args.cost_heads_num*args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, 48, 1, 1))

        self.convf1_ = nn.Conv2d(4, 96, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(96, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192+48*3, 128-4+48, k_conv=args.k_conv)

        self.velocity_update_block = velocity_update_block()

    def sample_flo_feat(self, flow, feat):
        
        sampled_feat = bilinear_sampler(feat.float(), flow.permute(0, 2, 3, 1))
        return sampled_feat

    def forward(self, motion_hidden_state, forward_flow, backward_flow, coords0, forward_corr, backward_corr, bs):
        
        BN, _, H, W = forward_flow.shape
        N = BN // bs

        if motion_hidden_state is None:
            #print("initialized as None")
            motion_hidden_state = self.init_hidden_state.repeat(bs, N, 1, H, W)
        else:
            #print("later iterations")
            motion_hidden_state = motion_hidden_state.reshape(bs, N, -1, H, W)

        motion_hidden_state_sc = motion_hidden_state.clone()
        
        forward_loc = forward_flow+coords0
        backward_loc = backward_flow+coords0

        forward_motion_hidden_state = torch.cat([motion_hidden_state[:, 1:, ...], torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device)], dim=1).reshape(BN, -1, H, W)
        forward_motion_hidden_state = self.sample_flo_feat(forward_loc, forward_motion_hidden_state)
        backward_motion_hidden_state = torch.cat([torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device), motion_hidden_state[:, :N-1, ...]], dim=1).reshape(BN, -1, H, W)
        backward_motion_hidden_state = self.sample_flo_feat(backward_loc, backward_motion_hidden_state)

        forward_cor = self.convc1(forward_corr)
        backward_cor = self.convc1(backward_corr)
        cor = F.gelu(torch.cat([forward_cor, backward_cor], dim=1))
        cor = self.convc2(cor)

        flow = torch.cat([forward_flow, backward_flow], dim=1)
        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo, forward_motion_hidden_state, backward_motion_hidden_state, motion_hidden_state.reshape(BN, -1, H, W)], dim=1)
        out = self.conv(cor_flo)

        out, motion_hidden_state = torch.split(out, [124, 48], dim=1)

        motion_hidden_state = motion_hidden_state + motion_hidden_state_sc

        return torch.cat([out, flow], dim=1), motion_hidden_state


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

    def forward(self, net, motion_hidden_state, inp, forward_corr, backward_corr, forward_flow, backward_flow, coords0, attention, bs):

        motion_features, motion_hidden_state = self.encoder(motion_hidden_state, forward_flow, backward_flow, coords0, forward_corr, backward_corr, bs=bs)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 100.0 * self.mask(net)
        return net, motion_hidden_state, mask, delta_flow 
