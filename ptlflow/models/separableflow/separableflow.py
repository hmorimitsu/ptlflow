from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, CorrBlock1D
from .cost_agg import CostAggregation
from .utils import coords_grid, upflow8, InputPadder
from ..base_model.base_model import BaseModel


class SequenceLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args.gamma
        self.max_flow = args.max_flow

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
            i_loss = (flow_preds[i] - flow_gt).abs()
            flow_loss += i_weight * (valid * i_loss).mean()

        return flow_loss


class Guidance(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(Guidance, self).__init__()
        self.bn_relu = nn.Sequential(nn.InstanceNorm2d(channels), nn.ReLU(inplace=True))
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, int(channels / 4), kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(int(channels / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(channels / 4), int(channels / 2), kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(int(channels / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(channels / 2), channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        inner_channels = channels // 4
        self.wsize = 20
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 2, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels, inner_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels, inner_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(
                inner_channels, inner_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(
                inner_channels * 2,
                inner_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels * 2,
                inner_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.weights = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, self.wsize, kernel_size=3, stride=1, padding=1),
        )
        self.weight_sg1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels, self.wsize * 2, kernel_size=3, stride=1, padding=1
            ),
        )
        self.weight_sg2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels, self.wsize * 2, kernel_size=3, stride=1, padding=1
            ),
        )
        self.weight_sg11 = nn.Sequential(
            nn.Conv2d(inner_channels * 2, inner_channels * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels * 2, self.wsize * 2, kernel_size=3, stride=1, padding=1
            ),
        )
        self.weight_sg12 = nn.Sequential(
            nn.Conv2d(inner_channels * 2, inner_channels * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels * 2, self.wsize * 2, kernel_size=3, stride=1, padding=1
            ),
        )
        self.weight_sg3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels, self.wsize * 2, kernel_size=3, stride=1, padding=1
            ),
        )
        # self.getweights = nn.Sequential(GetFilters(radius=1),
        #                                nn.Conv2d(9, 20, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, fea, img):
        x = self.conv0(img)
        x = torch.cat((self.bn_relu(fea), x), 1)
        x = self.conv1(x)
        rem = x
        x = self.conv2(x) + rem
        rem = x
        guid = self.weights(x)
        x = self.conv3(x) + rem
        sg1 = self.weight_sg1(x)
        sg1_u, sg1_v = torch.split(sg1, (self.wsize, self.wsize), dim=1)
        sg2 = self.weight_sg2(x)
        sg2_u, sg2_v = torch.split(sg2, (self.wsize, self.wsize), dim=1)
        sg3 = self.weight_sg3(x)
        sg3_u, sg3_v = torch.split(sg3, (self.wsize, self.wsize), dim=1)
        x = self.conv11(x)
        rem = x
        x = self.conv12(x) + rem
        sg11 = self.weight_sg11(x)
        sg11_u, sg11_v = torch.split(sg11, (self.wsize, self.wsize), dim=1)
        sg12 = self.weight_sg12(x)
        sg12_u, sg12_v = torch.split(sg12, (self.wsize, self.wsize), dim=1)
        guid_u = dict(
            [
                ("sg1", sg1_u),
                ("sg2", sg2_u),
                ("sg3", sg3_u),
                ("sg11", sg11_u),
                ("sg12", sg12_u),
            ]
        )
        guid_v = dict(
            [
                ("sg1", sg1_v),
                ("sg2", sg2_v),
                ("sg3", sg3_v),
                ("sg11", sg11_v),
                ("sg12", sg12_v),
            ]
        )
        return guid, guid_u, guid_v


class SeparableFlow(BaseModel):
    pretrained_checkpoints = {
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/separableflow-things-31fe3b2d.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/separableflow-sintel-4c9a8c03.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/separableflow-kitti-c9395318.ckpt",
        "universal": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/separableflow-universal-87350d91.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=64)

        hdim = self.args.hidden_dim
        cdim = self.args.context_dim

        if "dropout" not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=args.dropout
        )
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
        )
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.guidance = Guidance(channels=256)
        self.cost_agg1 = CostAggregation(in_channel=8)
        self.cost_agg2 = CostAggregation(in_channel=8)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--iters", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--context_dim", type=int, default=128)
        return parser

    def freeze_bn(self):
        count1, count2, count3 = 0, 0, 0
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                count1 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm2d):
                count2 += 1
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                count3 += 1
                # print(m)
                m.eval()
        # print(count1, count2, count3)
        # print(m)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

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

        hdim = self.args.hidden_dim
        cdim = self.args.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        guid, guid_u, guid_v = self.guidance(fmap1.detach(), image1)
        corr_fn = CorrBlock(fmap1, fmap2, guid, radius=self.args.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        corr1, corr2 = corr_fn(None, sep=True)
        coords0, coords1 = self.initialize_flow(image1)
        if self.training:
            u0, u1, flow_u, corr1 = self.cost_agg1(
                corr1, guid_u, max_shift=384, is_ux=True
            )
            v0, v1, flow_v, corr2 = self.cost_agg2(
                corr2, guid_v, max_shift=384, is_ux=False
            )
            flow_init = torch.cat((flow_u, flow_v), dim=1)

            flow_predictions = []
            flow_predictions.append(torch.cat((u0, v0), dim=1))
            flow_predictions.append(torch.cat((u1, v1), dim=1))
            flow_predictions.append(flow_init)

        else:
            flow_u, corr1 = self.cost_agg1(corr1, guid_u, max_shift=384, is_ux=True)
            flow_v, corr2 = self.cost_agg2(corr2, guid_v, max_shift=384, is_ux=False)
            flow_init = torch.cat((flow_u, flow_v), dim=1)
        flow_init = F.interpolate(
            flow_init.detach() / 8.0,
            [cnet.shape[2], cnet.shape[3]],
            mode="bilinear",
            align_corners=True,
        )
        corr1d_fn = CorrBlock1D(corr1, corr2, radius=self.args.corr_radius)
        coords1 = coords1 + flow_init
        for itr in range(self.args.iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)  # index correlation volume
            corr1, corr2 = corr1d_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(
                net, inp, corr, corr1, corr2, flow
            )

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            if self.training:
                flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": coords1 - coords0}

        return outputs
