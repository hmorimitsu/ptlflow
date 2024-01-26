import torch
import torch.nn as nn
import torch.nn.functional as F
from .xcit import XCiT


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class RefineHead(nn.Module):
    def __init__(self, input_dim=4 * 128, hidden_dim=128):
        super(RefineHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

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


class BasicMotionEncoder(nn.Module):
    def __init__(self, correlation_depth):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(correlation_depth, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.act = F.relu

        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = self.act(self.convc1(corr))
        cor = self.act(self.convc2(cor))
        flo = self.act(self.convf1(flow))
        flo = self.act(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.act(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        args,
        correlation_depth,
        hidden_dim=128,
        input_dim=256,
        scale=8,
        num_heads=8,
        depth=1,
        mlp_ratio=1,
        num_scales=4,
    ):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(correlation_depth)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale * scale * 9, 1, padding=0),
        )

        self.aggregator = nn.ModuleList(
            [
                XCiT(
                    embed_dim=128,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    separate=True,
                )
            ]
        )
        for i in range(num_scales - 1):
            self.aggregator.extend(
                [
                    XCiT(
                        embed_dim=128,
                        depth=depth,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        separate=True,
                    )
                ]
            )

    def forward(self, net, inp, corr, flow, global_context, level_index=0):
        motion_features = self.encoder(flow, corr)  # motion feature depth: 128
        motion_features_global = self.aggregator[level_index](
            global_context, motion_features
        )
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
