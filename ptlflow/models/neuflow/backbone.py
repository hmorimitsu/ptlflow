import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
            bias=False,
        )

        self.conv2 = torch.nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.norm = torch.nn.BatchNorm2d(out_planes, eps=1e-06, affine=False)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        # x = self.dropout(x)

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        return self.norm(x1 + x2)


class DownDimBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownDimBlock, self).__init__()

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.conv_block = ConvBlock(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        return self.conv_block(self.relu(x))


class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim):
        super(CNNEncoder, self).__init__()

        # self.conv0 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=False) # rgb2gray
        # self.norm0 = torch.nn.BatchNorm2d(16, eps=1e-06, affine=False)

        self.block1_1 = ConvBlock(
            3, feature_dim, kernel_size=8, stride=8, padding=0
        )  # 1/1

        self.block1_2 = ConvBlock(
            3, feature_dim, kernel_size=8, stride=4, padding=2
        )  # 1/2

        self.block1_3 = ConvBlock(
            3, feature_dim, kernel_size=8, stride=2, padding=3
        )  # 1/4

        self.block1_4 = ConvBlock(
            3, feature_dim, kernel_size=7, stride=1, padding=3
        )  # 1/8

        self.block1_dd = DownDimBlock(feature_dim * 4, feature_dim)  # pick features
        self.block1_ds = ConvBlock(
            feature_dim, feature_dim, kernel_size=2, stride=2, padding=0
        )

        self.block2 = ConvBlock(
            3, feature_dim, kernel_size=7, stride=1, padding=3
        )  # 1/16
        self.block2_dd = DownDimBlock(feature_dim * 2, feature_dim)  # pick features

    def init_pos(self, batch_size, height, width, dtype, device):
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=dtype, device=device),
            torch.arange(width, dtype=dtype, device=device),
            indexing="ij",
        )
        ys = ys / (height - 1)
        xs = xs / (width - 1)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size, 1, 1, 1)

    def init_pos_12(self, batch_size, height, width, dtype, device):
        self.pos_1 = self.init_pos(
            batch_size, height, width, dtype=dtype, device=device
        )
        self.pos_2 = self.init_pos(
            batch_size, height // 2, width // 2, dtype=dtype, device=device
        )

    def forward(self, img):
        b = img.shape[0]

        # x = self.relu(self.norm0(self.conv0(x)))

        x1_1 = self.block1_1(img)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x1_2 = self.block1_2(img)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x1_3 = self.block1_3(img)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x1_4 = self.block1_4(img)

        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.block1_dd(x1)

        img = F.avg_pool2d(img, kernel_size=2, stride=2)

        x2 = self.block2(img)

        x2 = torch.cat([self.block1_ds(x1), x2], dim=1)
        x2 = self.block2_dd(x2)

        x1 = torch.cat([x1, self.pos_1], dim=1)
        x2 = torch.cat([x2, self.pos_2], dim=1)

        # x2 = self.self_attn(x2, x2)

        return [x1, x2]
