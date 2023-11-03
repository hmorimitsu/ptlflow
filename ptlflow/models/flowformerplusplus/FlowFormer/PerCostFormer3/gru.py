import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
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

class ConvAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        #self.conv1_1 = nn.Conv2d(dim, dim, (15, 15), padding=(7, 7), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        #attn = attn + attn_1 

        attn = self.conv3(attn)

        return attn * u

class SKBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

        self.conv_s = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.conv_l = nn.Conv2d(hidden_dim, hidden_dim, 15, padding=7, groups=hidden_dim)
        self.conv_pw = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)

    def forward(self, x):
        
        x = self.proj_1(x)
        shortcut = x.clone()
        x = shortcut + self.conv_s(x)
        shortcut = x.clone()
        x = shortcut + self.conv_l(x)
        short_cut = x.clone()
        x = shortcut + sefl.conv_pw(x)
        return x
        

class ConvAttWoGRUBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ConvAttWoGRUBlock, self).__init__()

        self.proj_1 = nn.Conv2d(hidden_dim, hidden_dim, 1)   
        self.activation = nn.GELU()
        self.spatial_gating_unit = ConvAtt(hidden_dim)
        self.proj_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class ConvAttWoGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvAttWoGRU, self).__init__()

        self.proj_1 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = ConvAttWoGRUBlock(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim*4, hidden_dim, 1),
        )

    def forward(self, x):

        x = self.proj_1(x)

        shortcut = x.clone()
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.attn(x)

        shortcut = x.clone()

        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.mlp(x)

        return x

class SKBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

        self.conv_s = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.conv_l = nn.Conv2d(hidden_dim, hidden_dim, 15, padding=7, groups=hidden_dim)
        self.conv_pw = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)

    def forward(self, x):
        
        x = self.proj_1(x)
        shortcut = x.clone()
        x = shortcut + self.conv_s(x)
        shortcut = x.clone()
        x = shortcut + self.conv_l(x)
        short_cut = x.clone()
        x = shortcut + self.conv_pw(x)
        return x

class SKMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SKMotionEncoder, self).__init__()
        
        if args.r_16 > 0:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim+args.r_16**2
        else:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim
        
        self.sk_c = SKBlock(input_dim=cor_planes, hidden_dim=192)
        self.sk_f = SKBlock(input_dim=2, hidden_dim=64)
        self.sk_fusion = SKBlock(input_dim=192+64, hidden_dim=128-2)

    def forward(self, flow, corr):
        cor = self.sk_c(corr)
        flo = self.sk_f(flow)
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.sk_fusion(cor_flo)
        return torch.cat([out, flow], dim=1)

class SKWoGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SKWoGRU, self).__init__()

        self.updator = SKBlock(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, x):

        x = self.updator(x)

        return x

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        
        if args.r_16 > 0:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim+args.r_16**2
        else:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim
        
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

class ConvAttMotionEncoder(nn.Module):
    def __init__(self, args):
        super(ConvAttMotionEncoder, self).__init__()
        
        if args.r_16 > 0:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim+args.r_16**2
        else:
            cor_planes = 81*args.cost_heads_num+args.query_latent_dim
        
        self.convc = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convf = nn.Conv2d(2, 64, 7, padding=3)

        self.motion_fusion = nn.Sequential(
            ConvAttWoGRU(hidden_dim=128-2, input_dim=128+64),
            #ConvAttWoGRU(hidden_dim=128-2, input_dim=128)
        )
  
    def forward(self, flow, corr):
        cor = self.convc(corr)
        flo = self.convf(flow)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.motion_fusion(cor_flo)

        return torch.cat([out, flow], dim=1)

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

from .gma import Aggregate
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

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class ConvAttWoGRUMOnlyGMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = ConvAttMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

from .gma import Aggregate
class ConvAttWoGRUGMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = ConvAttWoGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        out = self.gru(inp_cat)

        delta_flow = self.flow_head(out)

        # scale mask to balence gradients
        mask = .25 * self.mask(out)
        return out, mask, delta_flow

class ConvAttWoGRUUMGMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = ConvAttMotionEncoder(args)
        self.gru = nn.Sequential(
            ConvAttWoGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim),
            #ConvAttWoGRU(hidden_dim=hidden_dim, input_dim=hidden_dim),
        )
        
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        out = self.gru(inp_cat)

        delta_flow = self.flow_head(out)

        # scale mask to balence gradients
        mask = .25 * self.mask(out)
        return out, mask, delta_flow

class SKGMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = SKMotionEncoder(args)
        self.gru = SKWoGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        out = self.gru(inp_cat)

        delta_flow = self.flow_head(out)

        # scale mask to balence gradients
        mask = .25 * self.mask(out)
        return out, mask, delta_flow
