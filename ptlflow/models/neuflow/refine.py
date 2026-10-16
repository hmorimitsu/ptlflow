import torch

from ptlflow.utils.corr_modes import get_spatial_corr_sampler


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="zeros",
            bias=True,
        )
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))


class Refine(torch.nn.Module):
    def __init__(self, feature_dim, patch_size, num_layers, corr_mode="iter"):
        super(Refine, self).__init__()

        self.patch_size = patch_size
        self.corr_mode = corr_mode

        self.correlation_sampler, self._corr_error = get_spatial_corr_sampler(
            corr_mode, patch_size=patch_size
        )

        self.conv1 = ConvBlock(
            patch_size**2 + feature_dim + 2, 96, kernel_size=3, stride=1, padding=1
        )

        self.conv_layers = torch.nn.ModuleList(
            [
                ConvBlock(96, 96, kernel_size=3, stride=1, padding=1)
                for i in range(num_layers)
            ]
        )

        self.conv2 = ConvBlock(96, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(
            32, 2, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=True
        )

    def forward(self, feature_0, feature_1, flow_0):
        b, c, h, w = feature_0.shape

        if self.correlation_sampler is None:
            raise self._corr_error
        attn = self.correlation_sampler(feature_0, feature_1).view(b, -1, h, w)
        # attn = F.softmax(attn, dim=1)

        x = torch.cat([attn, feature_0, flow_0], dim=1)

        x = self.conv1(x)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.conv2(x)
        x = self.conv3(x)

        return self.conv4(x)
