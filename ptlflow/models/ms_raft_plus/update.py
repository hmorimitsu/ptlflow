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
    def __init__(
        self, hidden_dim=128, input_dim=256 + 128
    ):  # input dim=256 + 128, hdim=64
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(input_dim, hidden_dim, (5, 1), padding=(2, 0))

    # h:net:128, x = 128 + 128 = 256
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
    def __init__(self, correlation_depth, stack_coords=False):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(correlation_depth, 256, 1, padding=0)
        self.stack_coords = stack_coords
        # self.convc1 = nn.Conv2d(324, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        if not stack_coords:
            self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

        else:
            self.conv_coords_x_1 = nn.Conv2d(correlation_depth, 64, 3, padding=1)
            self.conv_coords_x_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv_coords_y_1 = nn.Conv2d(correlation_depth, 64, 3, padding=1)
            self.conv_coords_y_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.conv_corr_coords = nn.Conv2d(192 + 128, 256, 3, padding=1)
            self.conv_corr_coords_flow_1 = nn.Conv2d(256 + 64, 256, 3, padding=1)
            self.conv_corr_coords_flow_2 = nn.Conv2d(256, 128 - 2, 1, padding=0)

    def forward(self, flow, corr, coords_x=None, coords_y=None):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        if not self.stack_coords:
            cor_flo = torch.cat([cor, flo], dim=1)
            out = F.relu(self.conv(cor_flo))
        else:
            conved_coords_x_1 = F.relu(self.conv_coords_x_1(coords_x))
            conved2_coords_x = F.relu(self.conv_coords_x_2(conved_coords_x_1))

            conved_coords_y_1 = F.relu(self.conv_coords_y_1(coords_y))
            conved2_coords_y = F.relu(self.conv_coords_y_2(conved_coords_y_1))

            conved_corr_coords = F.relu(
                self.conv_corr_coords(
                    torch.cat([conved2_coords_x, conved2_coords_y, cor], dim=1)
                )
            )
            conved_corr_coords_flow_1 = F.relu(
                self.conv_corr_coords_flow_1(
                    torch.cat([conved_corr_coords, flo], dim=1)
                )
            )
            out = F.relu(self.conv_corr_coords_flow_2(conved_corr_coords_flow_1))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        args,
        correlation_depth,
        stack_coords=False,
        hidden_dim=128,
        input_dim=256,
        scale=8,
    ):
        super(BasicUpdateBlock, self).__init__()
        # hidden_dim = 64
        # input_dim = 192
        self.args = args
        self.encoder = BasicMotionEncoder(correlation_depth, stack_coords=stack_coords)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, scale * scale * 9, 1, padding=0),
        )

    def forward(
        self, net, inp, corr, flow, coords_x=None, coords_y=None, level_index=0
    ):
        # net: 128
        # inp depth: 128
        motion_features = self.encoder(
            flow, corr, coords_x, coords_y
        )  # motion feature depth: 128
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)  # output:net.depth:128

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow
