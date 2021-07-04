from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Union

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
            if i > 1:
                features.append(x.view(*images.shape[:2], *x.shape[1:]))

        return features[::-1]


class FlowFieldDeformation(nn.Module):
    def __init__(
        self,
        level: int
    ) -> None:
        super(FlowFieldDeformation, self).__init__()

        patch_size = [None, 5, 7, 9][level]
        pred_kernel_size = [None, 3, 5, 5][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.up_conf = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)

        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=patch_size, padding=0, stride=1, dilation_patch=2)

        self.feat_net = nn.Sequential(
            nn.Conv2d(patch_size**2+1, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
        )

        self.disp_pred = nn.Conv2d(32, 2, pred_kernel_size, 1, pred_kernel_size//2)

        self.conf_pred = nn.Sequential(
            nn.Conv2d(32, 1, pred_kernel_size, 1, pred_kernel_size//2),
            nn.Sigmoid()
        )

        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: torch.Tensor,
        conf: torch.Tensor
    ) -> torch.Tensor:
        conf = self.up_conf(conf)
        flow = self.up_flow(flow)

        self_corr = self.leaky_relu(self.corr(feats[:, 0], feats[:, 0]))
        self_corr = self_corr.view(self_corr.shape[0], -1, self_corr.shape[3], self_corr.shape[4])
        self_corr = self_corr / feats.shape[2]

        x = torch.cat([self_corr, conf], dim=1)
        x = self.feat_net(x)

        disp = self.disp_pred(x)

        flow = self.warp(flow, disp, flow.shape[-2], flow.shape[-1], 1.0)

        conf = self.conf_pred(x)

        return flow, conf


class CostVolumeModulation(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0
    ) -> None:
        super().__init__()

        input_dims = [None, 210, 178, 146][level]
        self.mult = [div_flow / 2**(num_levels-i+1) for i in range(num_levels)][level]
        
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1)

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.feat_net = nn.Sequential(
            nn.Conv2d(input_dims, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 64, 3, 1, 1),
            self.leaky_relu
        )

        self.mod_scalar_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 81, 1, 1, 0)
        )

        self.mod_offset_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 81, 1, 1, 0)
        )

        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: torch.Tensor,
        conf: torch.Tensor
    ) -> torch.Tensor:
        warped_feat2 = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0/self.mult)

        corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
        corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
        corr = corr / feats.shape[2]

        x = torch.cat([feats[:, 0], corr, conf], dim=1)
        x = self.feat_net(x)

        mod_scalar = self.mod_scalar_net(x)
        mod_offset = self.mod_offset_net(x)

        corr = mod_scalar * corr + mod_offset

        return corr

class Matching(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0,
        use_s_version: bool = False
    ) -> None:
        super(Matching, self).__init__()

        flow_kernel_size = [3, 3, 5, 5][level]
        self.mult = [div_flow / 2**(num_levels-i+1) for i in range(num_levels)][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        if level == 1 and not use_s_version:
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        else:
            self.up_flow = None

        if level < 2:
            self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=9, padding=0, stride=1, dilation_patch=1)
        else:
            self.corr = None

        self.flow_net = nn.Sequential(
            nn.Conv2d(81, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 96, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(96, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size//2)
        )

        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: Optional[torch.Tensor],
        corr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.up_flow is not None:
            flow = self.up_flow(flow)

        if corr is None:
            warped_feat2 = feats[:, 1]
            if flow is not None:
                warped_feat2 = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0/self.mult)

            corr = self.leaky_relu(self.corr(feats[:, 0], warped_feat2))
            corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
            corr = corr / feats.shape[2]

        new_flow = self.flow_net(corr)
        if flow is not None:
            new_flow = flow + new_flow
        return new_flow


class SubPixel(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0
    ) -> None:
        super(SubPixel, self).__init__()

        inputs_dims = [386, 258, 194, 130][level]
        flow_kernel_size = [3, 3, 5, 5][level]
        self.mult = [div_flow / 2**(num_levels-i+1) for i in range(num_levels)][level]

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.feat_net = nn.Sequential(
            nn.Conv2d(inputs_dims, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 128, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(128, 96, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(96, 64, 3, 1, 1),
            self.leaky_relu,
            nn.Conv2d(64, 32, 3, 1, 1),
            self.leaky_relu
        )

        self.flow_net = nn.Conv2d(32, 2, flow_kernel_size, 1, flow_kernel_size//2)

        self.warp = WarpingLayer()

    def forward(
        self,
        feats: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        feat_warped = self.warp(feats[:, 1], flow, feats.shape[-2], feats.shape[-1], 1.0/self.mult)
        x = torch.cat([feats[:, 0], feat_warped, flow], dim=1)
        x = self.feat_net(x)
        new_flow = self.flow_net(x)
        new_flow = flow + new_flow
        return new_flow, x


class Regularization(nn.Module):
    def __init__(
        self,
        level: int,
        num_levels: int = 4,
        div_flow: float = 20.0,
        use_s_version: bool = False
    ) -> None:
        super(Regularization, self).__init__()

        inputs_dims = [195, 131, 99, 67][level]
        flow_kernel_size = [3, 3, 5, 5][level]
        conf_kernel_size = [3, 3, 5, None][level]
        self.mult = [div_flow / 2**(num_levels-i+1) for i in range(num_levels)][level]

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

        if (level == 0 and not use_s_version) or level == 3:
            self.conf_pred = None
        else:
            self.conf_pred = nn.Sequential(
                nn.Conv2d(32, 1, conf_kernel_size, 1, conf_kernel_size//2),
                nn.Sigmoid()
            )

        self.warp = WarpingLayer()

    def forward(
        self,
        images: torch.Tensor,
        feats: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        img2_warped = self.warp(images[:, 1], flow, images.shape[-2], images.shape[-1], 1.0/self.mult)
        img_diff_norm = torch.norm(images[:, 0] - img2_warped, p=2, dim=1, keepdim=True)

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

        conf = None
        if self.conf_pred is not None:
            conf = self.conf_pred(x)

        return flow, conf, x


class PseudoSubpixel(nn.Module):
    def __init__(
        self
    ) -> None:
        super(PseudoSubpixel, self).__init__()

        self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)

        self.flow_net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.Conv2d(32, 2, 7, 1, 3)
        )

    def forward(
        self,
        sub_feat: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        return self.up_flow(flow) + self.flow_net(sub_feat)


class PseudoRegularization(nn.Module):
    def __init__(
        self
    ) -> None:
        super(PseudoRegularization, self).__init__()
        
        self.feat_net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.Conv2d(32, 49, (7, 1), 1, (3, 0)),
            nn.Conv2d(49, 49, (1, 7), 1, (0, 3))
        )

        self.unfold = nn.Unfold(7, padding=3)

    def forward(
        self,
        reg_feat: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        dist = self.feat_net(reg_feat)
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


class LiteFlowNet3(BaseModel):
    pretrained_checkpoints = {
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3-sintel-d985929f.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        super(LiteFlowNet3, self).__init__(
            args=args,
            loss_fn=None,
            output_stride=32)

        self.num_levels = 4

        if args.use_s_version:
            self.min_mod_level = 1
        else:
            self.min_mod_level = 2

        self.feature_net = FeatureExtractor()
        self.deformation_nets = nn.ModuleList([FlowFieldDeformation(i) for i in range(self.min_mod_level, self.num_levels)])
        self.modulation_nets = nn.ModuleList(
            [CostVolumeModulation(i, self.num_levels, self.args.div_flow) for i in range(self.min_mod_level, self.num_levels)])
        self.matching_nets = nn.ModuleList(
            [Matching(i, self.num_levels, self.args.div_flow, self.args.use_s_version) for i in range(self.num_levels)])
        self.subpixel_nets = nn.ModuleList([SubPixel(i, self.num_levels, self.args.div_flow) for i in range(self.num_levels)])
        self.regularization_nets = nn.ModuleList(
            [Regularization(i, self.num_levels, self.args.div_flow, self.args.use_s_version) for i in range(self.num_levels)])

        if self.args.use_pseudo_regularization:
            self.pseudo_subpixel = PseudoSubpixel()
            self.pseudo_regularization = PseudoRegularization()
            self.up_flow = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False, groups=2)
        else:
            self.up_flow = nn.ConvTranspose2d(2, 2, 8, 4, 2, bias=False, groups=2)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--div_flow', type=float, default=20.0)
        parser.add_argument('--use_pseudo_regularization', action='store_true')
        parser.add_argument('--use_s_version', action='store_true')
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
        conf_preds = []
        flow = None
        conf = None
        corr = None

        for i in range(self.num_levels):
            if i >= self.min_mod_level:
                flow, conf = self.deformation_nets[i-self.min_mod_level](feats_pyr[i], flow, conf)
                if conf is not None:
                    conf_preds.append(conf)
                corr = self.modulation_nets[i-self.min_mod_level](feats_pyr[i], flow, conf)
            flow = self.matching_nets[i](feats_pyr[i], flow, corr)
            flow, sub_feat = self.subpixel_nets[i](feats_pyr[i], flow)
            flow, conf, reg_feat = self.regularization_nets[i](images_pyr[i], feats_pyr[i], flow)
            flow_preds.append(flow)
            if conf is not None:
                conf_preds.append(conf)
        
        if self.args.use_pseudo_regularization:
            flow = self.pseudo_subpixel(sub_feat, flow)
            flow = self.pseudo_regularization(reg_feat, flow)
            flow = self.up_flow(flow)
        else:
            flow = self.up_flow(flow)
        flow = flow * self.args.div_flow

        conf = F.interpolate(conf_preds[-1], scale_factor=4, mode='bilinear', align_corners=False)

        outputs = {}
        if self.training:
            outputs['flow_preds'] = flow_preds
            outputs['conf_preds'] = conf_preds
            outputs['flows'] = flow[:, None]
            outputs['confs'] = conf[:, None]
        else:
            outputs['flows'] = flow[:, None]
            outputs['confs'] = conf[:, None]
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


class LiteFlowNet3PseudoReg(LiteFlowNet3):
    pretrained_checkpoints = {
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3-kitti-b5d32443.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        args.use_pseudo_regularization = True
        super(LiteFlowNet3PseudoReg, self).__init__(args=args)


class LiteFlowNet3S(LiteFlowNet3):
    pretrained_checkpoints = {
        'sintel': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3s-sintel-89793e34.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        args.use_s_version = True
        super(LiteFlowNet3S, self).__init__(args=args)


class LiteFlowNet3SPseudoReg(LiteFlowNet3):
    pretrained_checkpoints = {
        'kitti': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/liteflownet3s-kitti-5dffb261.ckpt'
    }

    def __init__(self,
                 args: Namespace):
        args.use_s_version = True
        args.use_pseudo_regularization = True
        super(LiteFlowNet3SPseudoReg, self).__init__(args=args)
