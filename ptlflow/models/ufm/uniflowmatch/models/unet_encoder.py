"""
UNet Encoder for Fine Feature Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => ReLU) * 2 with padding"""

    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()

        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # preserve spatial
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # preserve spatial
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # preserve spatial
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # preserve spatial
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512], use_batch_norm=False):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Downsampling part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, batch_norm=use_batch_norm))
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, batch_norm=use_batch_norm)

        # Upsampling part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, batch_norm=use_batch_norm))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])  # Fix mismatched shapes
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=16)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
