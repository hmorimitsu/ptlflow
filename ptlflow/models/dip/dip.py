from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoderQuarter
from .utils import upflow4
from .path_match import PathMatch
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


class DIP(BaseModel):
    pretrained_checkpoints = {
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dip-kitti-b0b678b4.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dip-sintel-7abeb652.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dip-things-688d52a0.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=SequenceLoss(args), output_stride=16)

        self.hidden_dim = 128
        self.context_dim = 128

        self.dropout = 0

        # feature network, and update block
        self.fnet = BasicEncoderQuarter(
            output_dim=256, norm_fn="instance", dropout=self.dropout
        )

        self.update_block_s = SmallUpdateBlock(hidden_dim=self.hidden_dim)
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim)

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--gamma", type=float, default=0.8)
        parser.add_argument("--max_flow", type=float, default=1000.0)
        parser.add_argument("--iters", type=int, default=20)
        parser.add_argument("--max_offset", type=int, default=256)
        return parser

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def random_init_flow(self, fmap, max_offset, test_mode=False):
        N, C, H, W = fmap.shape
        if test_mode:
            init_seed = 20
            torch.manual_seed(init_seed)
            torch.cuda.manual_seed(init_seed)
        flow = (torch.rand(N, 2, H, W) - 0.5) * 2
        flow = flow.to(dtype=fmap.dtype, device=fmap.device) * max_offset
        return flow

    def upsample_flow(self, flow, mask, rate=4):
        """Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def build_pyramid(self, fmap1, fmap2, cnet, max_layers=5, min_width=40):
        py_fmap1 = []
        py_fmap2 = []
        py_cnet = []
        py_fmap1.append(fmap1)
        py_fmap2.append(fmap2)
        py_cnet.append(cnet)

        curr_fmap1 = fmap1
        curr_fmap2 = fmap2
        curr_cnet = cnet
        for i in range(max_layers - 1):
            if (curr_fmap1.shape[2] < min_width) and (curr_fmap1.shape[3] < min_width):
                break
            down_scale = 2 ** (i + 1)
            curr_fmap1 = F.avg_pool2d(curr_fmap1, 2, stride=2)
            curr_fmap2 = F.avg_pool2d(curr_fmap2, 2, stride=2)
            curr_cnet = F.avg_pool2d(curr_cnet, 2, stride=2)
            py_fmap1.append(curr_fmap1)
            py_fmap2.append(curr_fmap2)
            py_cnet.append(curr_cnet)

        return py_fmap1, py_fmap2, py_cnet

    def upflow(self, flow, targetMap, mode="bilinear"):
        """Upsample flow"""
        new_size = (targetMap.shape[2], targetMap.shape[3])
        factor = 1.0 * targetMap.shape[2] / flow.shape[2]
        return factor * F.interpolate(
            flow, size=new_size, mode=mode, align_corners=True
        )

    def inference(self, image1, image2, iters=6, max_layers=3, init_flow=None):
        max_layers = 3
        max_offset = 256
        auto_layer = 3
        if init_flow is not None:
            mag = torch.norm(init_flow, dim=1)
            max_offset = torch.max(mag)
            auto_layer = max_offset / 32
            auto_layer = int(auto_layer.ceil().cpu().numpy())
            if auto_layer < max_layers:
                max_layers = auto_layer
            # print("mag:", max_offset, "layers:", max_layers)

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        new_size = (image1.shape[2], image1.shape[3])

        # build layers
        min_width = 40
        py_fmap1, py_fmap2, py_cnet = self.build_pyramid(
            fmap1, fmap2, fmap1, max_layers=max_layers, min_width=min_width
        )
        n_levels = len(py_fmap1)

        # build layers
        s_fmap1 = py_fmap1[n_levels - 1]
        # init
        if init_flow is not None:
            new_size = (s_fmap1.shape[2], s_fmap1.shape[3])
            scale = s_fmap1.shape[2] / (init_flow.shape[2] * 1.0)
            s_flow = scale * F.interpolate(
                init_flow, size=new_size, mode="bilinear", align_corners=True
            )
            initail_flow_max = 2 ** (auto_layer + 1) * 1.0
            noise = self.random_init_flow(
                s_fmap1, max_offset=16, test_mode=(not self.training)
            )
            s_flow = s_flow + noise
        else:
            scale = 2 ** (n_levels - 1 + 2) * 1.0
            s_flow = self.random_init_flow(
                s_fmap1,
                max_offset=self.args.max_offset / scale,
                test_mode=(not self.training),
            )

        up_mask = None
        for i in range(n_levels):
            # print('i: ', i)
            curr_fmap1 = py_fmap1[n_levels - i - 1]
            curr_fmap2 = py_fmap2[n_levels - i - 1]
            curr_cnet = py_cnet[n_levels - i - 1]
            patch_fn = PathMatch(curr_fmap1, curr_fmap2)
            net, inp = torch.split(
                curr_cnet, [self.hidden_dim, self.context_dim], dim=1
            )
            net = torch.tanh(net)
            inp = torch.relu(inp)
            if i > 0:
                s_flow = self.upflow(flow_up, curr_fmap1)
                noise = self.random_init_flow(
                    curr_fmap1, max_offset=4, test_mode=(not self.training)
                )
                s_flow = s_flow + noise

            for itr in range(iters):
                s_flow = s_flow.detach()
                out_corrs = patch_fn(s_flow, is_search=False)

                net, up_mask, delta_flow = self.update_block_s(
                    net, inp, out_corrs, s_flow
                )

                s_flow = s_flow + delta_flow

                s_flow = s_flow.detach()
                out_corrs = patch_fn(s_flow, is_search=True)
                net, up_mask, delta_flow = self.update_block(
                    net, inp, out_corrs, s_flow
                )

                s_flow = s_flow + delta_flow
                flow_up = self.upsample_flow(s_flow, up_mask, rate=4)

        return flow_up

    def forward(self, inputs):
        """Estimate optical flow between pair of frames"""
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=False,
            resize_mode="pad",
            pad_mode="constant",
            pad_two_side=True,
            pad_value=-1,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        init_flow = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            init_flow = forward_interpolate_batch(inputs["prev_preds"]["flow_small"])

        if not self.training and init_flow is not None:
            flow_up = self.inference(
                image1, image2, iters=self.args.iters, init_flow=init_flow
            )
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            return {"flows": flow_up[:, None]}

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        # run the context network
        net, inp = torch.split(fmap1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # 1/4 -> 1/16
        # feature
        s_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
        s_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)
        # context(left)
        s_net = F.avg_pool2d(net, 4, stride=4)
        s_inp = F.avg_pool2d(inp, 4, stride=4)

        # 1/16
        s_patch_fn = PathMatch(s_fmap1, s_fmap2)

        # init flow
        s_flow = None
        s_flow = self.random_init_flow(
            s_fmap1,
            max_offset=self.args.max_offset // 16,
            test_mode=(not self.training),
        )

        # small initial: 1/16
        flow = None
        flow_up = None
        flow_predictions = []
        for itr in range(self.args.iters):
            # --------------- update1 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_search=False)

            s_net, up_mask, delta_flow = self.update_block_s(
                s_net, s_inp, out_corrs, s_flow
            )

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up = upflow4(flow)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_search=True)

            s_net, up_mask, delta_flow = self.update_block(
                s_net, s_inp, out_corrs, s_flow
            )

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up = upflow4(flow)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

        patch_fn = PathMatch(fmap1, fmap2)
        # large refine: 1/4
        for itr in range(self.args.iters):
            # --------------- update1 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_search=False)
            net, up_mask, delta_flow = self.update_block_s(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = self.upsample_flow(flow, up_mask, rate=4)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_search=True)
            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = self.upsample_flow(flow, up_mask, rate=4)
            flow_up = self.postprocess_predictions(flow_up, image_resizer, is_flow=True)
            flow_predictions.append(flow_up)

        if self.training:
            outputs = {"flows": flow_up[:, None], "flow_preds": flow_predictions}
        else:
            outputs = {"flows": flow_up[:, None], "flow_small": flow}

        return outputs
