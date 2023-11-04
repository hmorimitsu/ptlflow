import torch
import torch.nn.functional as F


class Correlation1D:
    def __init__(
        self,
        feature1,
        feature2,
        radius=32,
        x_correlation=False,
    ):
        self.radius = radius
        self.x_correlation = x_correlation

        if self.x_correlation:
            self.corr = self.corr_x(feature1, feature2)  # [B*H*W, 1, 1, W]
        else:
            self.corr = self.corr_y(feature1, feature2)  # [B*H*W, 1, H, 1]

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        b, h, w = coords.shape[:3]

        if self.x_correlation:
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.zeros_like(dx)
            delta_x = torch.stack((dx, dy), dim=-1).to(
                device=coords.device, dtype=coords.dtype
            )  # [2r+1, 2]

            coords_x = coords[:, :, :, 0]  # [B, H, W]
            coords_x = torch.stack(
                (coords_x, torch.zeros_like(coords_x)), dim=-1
            )  # [B, H, W, 2]

            centroid_x = coords_x.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_x = centroid_x + delta_x  # [B*H*W, 1, 2r+1, 2]

            coords_x = 2 * coords_x / (w - 1) - 1  # [-1, 1], y is always 0

            corr_x = F.grid_sample(
                self.corr, coords_x, mode="bilinear", align_corners=True
            )  # [B*H*W, G, 1, 2r+1]

            corr_x = corr_x.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_x = corr_x.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]
            return corr_x
        else:  # y correlation
            dy = torch.linspace(-r, r, 2 * r + 1).to(
                device=coords.device, dtype=coords.dtype
            )
            dx = torch.zeros_like(dy)
            delta_y = torch.stack((dx, dy), dim=-1).to(coords.device)  # [2r+1, 2]
            delta_y = delta_y.unsqueeze(1).unsqueeze(0)  # [1, 2r+1, 1, 2]

            coords_y = coords[:, :, :, 1]  # [B, H, W]
            coords_y = torch.stack(
                (torch.zeros_like(coords_y), coords_y), dim=-1
            )  # [B, H, W, 2]

            centroid_y = coords_y.view(b * h * w, 1, 1, 2)  # [B*H*W, 1, 1, 2]
            coords_y = centroid_y + delta_y  # [B*H*W, 2r+1, 1, 2]

            coords_y = 2 * coords_y / (h - 1) - 1  # [-1, 1], x is always 0

            corr_y = F.grid_sample(
                self.corr, coords_y, mode="bilinear", align_corners=True
            )  # [B*H*W, G, 2r+1, 1]

            corr_y = corr_y.view(b, h, w, -1)  # [B, H, W, (2r+1)*G]
            corr_y = corr_y.permute(0, 3, 1, 2).contiguous()  # [B, (2r+1)*G, H, W]

            return corr_y

    def corr_x(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c**0.5

        # x direction
        feature1 = feature1.permute(0, 2, 3, 1)  # [B, H, W, C]
        feature2 = feature2.permute(0, 2, 1, 3)  # [B, H, C, W]
        corr = torch.matmul(feature1, feature2)  # [B, H, W, W]

        corr = corr.unsqueeze(3).unsqueeze(3)  # [B, H, W, 1, 1, W]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, 1, W]

        return corr

    def corr_y(self, feature1, feature2):
        b, c, h, w = feature1.shape  # [B, C, H, W]
        scale_factor = c**0.5

        # y direction
        feature1 = feature1.permute(0, 3, 2, 1)  # [B, W, H, C]
        feature2 = feature2.permute(0, 3, 1, 2)  # [B, W, C, H]
        corr = torch.matmul(feature1, feature2)  # [B, W, H, H]

        corr = (
            corr.permute(0, 2, 1, 3).contiguous().view(b, h, w, 1, h, 1)
        )  # [B, H, W, 1, H, 1]
        corr = corr / scale_factor
        corr = corr.flatten(0, 2)  # [B*H*W, 1, H, 1]

        return corr
