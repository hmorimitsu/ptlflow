import torch
import math
import timm
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .backbone.depthanythingv2 import DepthAnythingFeature
from .backbone.vit import VisionTransformer, MODEL_CONFIGS

from .utils import coords_grid, bilinear_sampler

from ptlflow.utils.registry import register_model
from ..base_model.base_model import BaseModel


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(
                inp, oup, kernel_size=1, stride=s, padding=0, bias=True
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer1
        self.conv2 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer2
        self.conv3 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer3
        self.conv4 = timm.create_model(
            "resnet18.a3_in1k", pretrained=True, features_only=True
        ).layer4
        self.up_4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]


class SequenceLoss(nn.Module):
    def __init__(self, gamma: float, max_flow: float):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow

    def forward(self, outputs, inputs):
        """Loss function defined over sequence of flow predictions"""
        flow_preds = outputs["flow_preds"]
        flow_gt = inputs["flows"][:, 0]
        valid = inputs["valids"][:, 0]

        n_predictions = len(flow_preds)

        flow_loss = 0.0
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            loss_i = outputs["nf_preds"][i]
            final_mask = (
                (~torch.isnan(loss_i.detach()))
                & (~torch.isinf(loss_i.detach()))
                & valid
            )
            flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

        return flow_loss


class WAFT(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-chairs-16b9cbc4.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-things-24bd04dc.ckpt",
        "tar": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-48597867.ckpt",
        "tar-c": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-49c9625b.ckpt",
        "tar-c-t": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-t-eaa5c133.ckpt",
        "tar-c-t-kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-t-kitti-e5004e90.ckpt",
        "tar-c-t-sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-t-sintel-e582a3e6.ckpt",
        "tar-c-t-spring-540p": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-t-spring-540p-590939bf.ckpt",
        "tar-c-t-spring-1080p": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft-tar-c-t-spring-1080p-56700f83.ckpt",
    }

    def __init__(
        self,
        dav2_backbone: str = "vits",
        network_backbone: str = "vits",
        gamma: float = 0.8,
        max_flow: float = 400,
        iters: int = 32,
        var_min: float = 0,
        var_max: float = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            output_stride=112, loss_fn=SequenceLoss(gamma, max_flow), **kwargs
        )

        self.iters = iters
        self.var_min = var_min
        self.var_max = var_max

        self.da_feature = self.freeze_(DepthAnythingFeature(encoder=dav2_backbone))
        self.pretrain_dim = self.da_feature.model_configs[dav2_backbone]["features"]
        self.network_dim = MODEL_CONFIGS[network_backbone]["features"]
        self.refine_net = VisionTransformer(
            network_backbone, self.network_dim, patch_size=8
        )
        self.fnet = ResNet18Deconv(self.pretrain_dim // 2 + 3, 64)
        self.fmap_conv = nn.Conv2d(
            self.pretrain_dim // 2 + 64,
            self.network_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.hidden_conv = nn.Conv2d(
            self.network_dim * 2,
            self.network_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.warp_linear = nn.Conv2d(
            3 * self.network_dim + 2, self.network_dim, 1, 1, 0, bias=True
        )
        self.refine_transform = nn.Conv2d(
            self.network_dim // 2 * 3, self.network_dim, 1, 1, 0, bias=True
        )
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 4 * 9, 1, padding=0, bias=True),
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(self.network_dim, 2 * self.network_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.network_dim, 6, 1, padding=0, bias=True),
        )

    def freeze_(self, model):
        model = model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for p in model.buffers():
            p.requires_grad = False
        return model

    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)

    def normalize_image(self, img):
        """
        @img: (B,C,H,W) in range 0-255, RGB order
        """
        tf = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        )
        return tf(img / 255.0).contiguous()

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        N, _, H, W = image1.shape
        # initial feature
        da_feature1 = self.da_feature(image1)
        da_feature2 = self.da_feature(image2)
        fmap1_feats = self.fnet(torch.cat([da_feature1["out"], image1], dim=1))
        fmap2_feats = self.fnet(torch.cat([da_feature2["out"], image2], dim=1))
        da_feature1_2x = F.interpolate(
            da_feature1["out"], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        da_feature2_2x = F.interpolate(
            da_feature2["out"], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_feats[0], da_feature1_2x], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_feats[0], da_feature2_2x], dim=1))
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)

        flow_predictions = []
        info_predictions = []
        for itr in range(self.iters):
            flow_2x = flow_2x.detach()
            coords2 = (
                coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x
            ).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(
                torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1)
            )
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs["out"], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            info_up = self.postprocess_predictions(
                info_up, image_resizer, is_flow=False
            )
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        if self.training:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.var_min, max=0)
                term2 = (
                    (inputs["flows"][:, 0] - flow_predictions[i]).abs().unsqueeze(2)
                ) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(
                    weight, dim=1, keepdim=True
                ) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            outputs = {
                "flows": flow_up[:, None],
                "flow_preds": flow_predictions,
                "info_preds": info_predictions,
                "nf_preds": nf_predictions,
            }
        else:
            outputs = {"flows": flow_up[:, None]}

        return outputs


@register_model
class waft(WAFT):
    pass
