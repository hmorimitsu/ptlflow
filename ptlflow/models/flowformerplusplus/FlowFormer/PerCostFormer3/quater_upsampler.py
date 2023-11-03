import torch
import torch.nn.functional as F
import torch.nn as nn

from ...utils.utils import coords_grid

def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]

class quater_upsampler(nn.Module):
    def __init__(self):
        super(quater_upsampler, self).__init__()

        hidden_dim = 64

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim-2, 3, padding=1),
        )
        
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(9, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim*2+128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 16*9, 1, padding=0))
        
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim*2+128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1, padding=0))
        

    def get_local_cost(self, coords, feature0, feature1, local_radius=1):

        b, c, h, w = feature0.size()

        coords = coords.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1

        window_grid = generate_window_grid(-local_radius, local_radius,
                                        -local_radius, local_radius,
                                        local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
        sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

        # normalize coordinates to [-1, 1]
        sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
        window_feature = F.grid_sample(feature1, sample_coords_norm,
                                    padding_mode="zeros", align_corners=True
                                    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
        feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

        corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

        corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2)

        return corr
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4*H, 4*W)

    def forward(self, flow, context_quater, feat_s_quater, feat_t_quater, r=1):
        coords0, _ = initialize_flow(flow)
        coords1 = coords0 + flow
        local_cost = self.get_local_cost(coords1, feat_s_quater, feat_t_quater, local_radius=r)

        corr_feat = self.corr_encoder(local_cost)
        feat = torch.cat([flow, self.flow_encoder(flow), corr_feat, context_quater], dim=1)

        delta_flow = self.flow_head(feat)
        mask = self.mask_head(feat)

        flow = self.upsample_flow(flow+delta_flow, mask)
        
        return flow



        

