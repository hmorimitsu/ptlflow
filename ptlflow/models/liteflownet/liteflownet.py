from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    from ptlflow.utils.correlation import IterSpatialCorrelationSampler as SpatialCorrelationSampler
import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import WarpingLayer
from ..base_model.base_model import BaseModel


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 7, 1, 3),
                leaky_relu
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 2, 1),
                leaky_relu,
                nn.Conv2d(32, 32, 3, 1, 1),
                leaky_relu,
                nn.Conv2d(32, 32, 3, 1, 1),
                leaky_relu
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1),
                leaky_relu,
                nn.Conv2d(64, 64, 3, 1, 1),
                leaky_relu
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1),
                leaky_relu,
                nn.Conv2d(96, 96, 3, 1, 1),
                leaky_relu
            ),
            nn.Sequential(
                nn.Conv2d(96, 128, 3, 2, 1),
                leaky_relu
            ),
            nn.Sequential(
                nn.Conv2d(128, 192, 3, 2, 1),
                leaky_relu
            )
        ])

    def forward(
        self,
        images: torch.Tensor
    ) -> List[torch.Tensor]:
        features = []

        x = images.view(-1, *images.shape[2:])
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i > 0:
                features.append(x.view(*images.shape[:2], *x.shape[1:]))

        return features[::-1]


class Matching(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 5,
        div_flow: float = 20.0
    ) -> None:
        super(Matching, self).__init__()

        corr_stride = [1, 1, 1, 2, 2][level]
        flow_kernel_size = [3, 3, 5, 5, 7][level]
        self.mult = [div_flow / 2**(num_levels-i) for i in range(num_levels)][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        if level == 0:
            self.up_flow = None
        else:
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)

        if level < 3:
            self.up_corr = None
        else:
            self.up_corr = nn.ConvTranspose2d(49, 49, 4, 2, 1, bias=False, groups=49)

        self.flow_net = nn.Sequential(
            nn.Conv2d(49, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size//2)
        )

        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=7, padding=0, stride=corr_stride, dilation_patch=corr_stride)

        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: Optional[torch.Tensor]
    ) -> torch.Tensor:
        warped_feat2 = feats[:, 1]
        if flow is not None:
            flow = self.up_flow(flow)
            warped_feat2 = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0/self.mult)

        corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / feats.shape[2]
        if self.up_corr is not None:
            corr = self.up_corr(corr)
        new_flow = self.flow_net(corr)
        if flow is not None:
            new_flow = flow + new_flow
        return new_flow


class SubPixel(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 5,
        div_flow: float = 20.0
    ) -> None:
        super(SubPixel, self).__init__()

        inputs_dims = [386, 258, 194, 130, 130][level]
        flow_kernel_size = [3, 3, 5, 5, 7][level]
        self.mult = [div_flow / 2**(num_levels-i) for i in range(num_levels)][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.flow_net = nn.Sequential(
            nn.Conv2d(inputs_dims, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size//2)
        )

        self.warp = WarpingLayer()


    def forward(
        self,
        feats: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        feat_warped = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0/self.mult)
        x = torch.cat([feats[:, 0], feat_warped, flow], dim=1)
        new_flow = self.flow_net(x)
        new_flow = flow + new_flow
        return new_flow


class Regularization(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 5,
        div_flow: float = 20.0
    ) -> None:
        super(Regularization, self).__init__()

        inputs_dims = [195, 131, 99, 67, 35][level]
        flow_kernel_size = [3, 3, 5, 5, 7][level]
        self.mult = [div_flow / 2**(num_levels-i) for i in range(num_levels)][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        if level < 2:
            self.feat_conv = nn.Sequential()
        else:
            self.feat_conv = nn.Sequential(
                nn.Conv2d(inputs_dims-3, 128, 1, 1, 0),
                self.leaky_relu
            )
            inputs_dims = 131

        self.feat_net = nn.Sequential(
            nn.Conv2d(inputs_dims, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 32, 3, 1, 1),
            self.leaky_relu,
        )

        if level < 2:
            self.dist = nn.Conv2d(32, flow_kernel_size**2, 3, 1, 1)
        else:
            self.dist = nn.Sequential(
                nn.Conv2d(32, flow_kernel_size**2, (flow_kernel_size, 1), 1, (flow_kernel_size//2, 0)),
                nn.Conv2d(flow_kernel_size**2, flow_kernel_size**2, (1, flow_kernel_size), 1, (0, flow_kernel_size//2))
            )

        self.unfold = nn.Unfold(flow_kernel_size, padding=flow_kernel_size//2)

        self.warp = WarpingLayer()

    def forward(
        self,
        images: torch.Tensor,
        feats: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        img2_warped = self.warp(images[:, 1], flow, images.shape[-2], images.shape[-1], 1.0/self.mult)
        img_diff_norm = torch.norm(images[:, 0] - img2_warped[:, 1], p=2, dim=1, keepdim=True)

        flow_mean = flow.view(*flow.shape[:2], -1).mean(dim=-1)[..., None, None]
        flow_nomean = flow - flow_mean
        feat = self.feat_conv(feats[:, 0])
        x = torch.cat([img_diff_norm, flow_nomean, feat], dim=1)
        x = self.feat_net(x)
        dist = self.dist(x)
        dist = dist.square().neg()
        dist = (dist - dist.max(dim=1, keepdim=True)[0]).exp()
        div = dist.sum(dim=1, keepdim=True)

        reshaped_flow_x = self.unfold(flow[:, :1])
        reshaped_flow_x = reshaped_flow_x.view(*reshaped_flow_x.shape[:2], *flow.shape[2:4])
        flow_smooth_x = (reshaped_flow_x * dist).sum(dim=1, keepdim=True) / div

        reshaped_flow_y = self.unfold(flow[:, 1:2])
        reshaped_flow_y = reshaped_flow_y.view(*reshaped_flow_y.shape[:2], *flow.shape[2:4])
        flow_smooth_y = (reshaped_flow_y * dist).sum(dim=1, keepdim=True) / div

        flow = torch.cat([flow_smooth_x, flow_smooth_y], dim=1)

        return flow

class LiteFlowNet(BaseModel):
    pretrained_checkpoints = {
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet-kitti-49f1991a.ckpt',
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet-sintel-17991e50.ckpt',
        'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet-things-a4d066e2.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        super(LiteFlowNet, self).__init__(
            args=args,
            loss_fn=None,
            output_stride=32)

        self.num_levels = 5

        self.feature_net = FeatureExtractor()
        self.matching_nets = nn.ModuleList([Matching(i, self.num_levels, self.args.div_flow) for i in range(self.num_levels)])
        self.subpixel_nets = nn.ModuleList([SubPixel(i, self.num_levels, self.args.div_flow) for i in range(self.num_levels)])
        self.regularization_nets = nn.ModuleList(
            [Regularization(i, self.num_levels, self.args.div_flow) for i in range(self.num_levels)])
        self.feat2_conv = nn.Sequential(
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        return parser

    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        images = inputs['images']
        images_mean = images.view(*images.shape[:3], -1).mean(dim=-1)[..., None, None]
        images = images - images_mean
        
        feats_pyr = self.feature_net(images)
        images_pyr = self._create_images_pyr(images, feats_pyr)
        
        flow_preds = []
        flow = None

        for i in range(self.num_levels):
            feats2 = feats_pyr[i]
            if i == (self.num_levels - 1):
                feats2 = self.feat2_conv(feats2.view(-1, *feats2.shape[2:])).view(*feats2.shape[:2], -1, *feats2.shape[3:])
            flow = self.matching_nets[i](feats2, flow)
            flow = self.subpixel_nets[i](feats2, flow)
            flow = self.regularization_nets[i](images_pyr[i], feats_pyr[i], flow)
            flow_preds.append(flow)
        
        flow = flow * self.args.div_flow
        flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flow_preds
            outputs['flows'] = flow[:, None]
        else:
            outputs['flows'] = flow[:, None]
        return outputs


    def _create_images_pyr(
        self,
        images: torch.Tensor,
        feats_pyr: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        batch_size = images.shape[0]
        images = images.view(-1, *images.shape[2:]).detach()
        images_pyr = [
            F.interpolate(images, size=feats_pyr[i].shape[-2:], mode='bilinear', align_corners=False)
            for i in range(len(feats_pyr))]
        images_pyr = [im.view(batch_size, -1, *im.shape[1:]) for im in images_pyr]
        return images_pyr
