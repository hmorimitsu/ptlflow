# =============================================================================
# Copyright 2024 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from RAFT: https://github.com/princeton-vl/RAFT
# =============================================================================

from argparse import ArgumentParser
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptlflow.utils.utils import forward_interpolate_batch
from .pwc_modules import rescale_flow
from .update import UpdateBlock
from .corr import get_corr_block
from .local_timm.norm import LayerNorm2d
from ..base_model.base_model import BaseModel

from .next1d_encoder import NeXt1DEncoder
from .next1d import NeXt1DStage

try:
    import alt_cuda_corr
except:
    alt_cuda_corr = None


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


class RAPIDFlow(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rapidflow-chairs-9c8c182a.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rapidflow-things-0377c8fa.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rapidflow-sintel-89a21262.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/rapidflow-kitti-2561329f.ckpt",
    }

    def __init__(self, args):
        num_recurrent_layers = int(math.log2(max(args.pyramid_ranges))) - 1
        super().__init__(
            args=args,
            loss_fn=SequenceLoss(args),
            output_stride=int(2 ** (num_recurrent_layers + 1)),
        )

        for v in self.args.pyramid_ranges:
            assert (
                v > 1
            ), f"--pyramid_ranges values must be larger than 1, but found {v}"
            log_res = math.log2(v)
            assert (log_res) - int(
                log_res
            ) < 1e-3, f"--pyramid_ranges values must be powers of 2, but found {v}"
        num_recurrent_layers = int(math.log2(max(self.args.pyramid_ranges))) - 1

        self.pyramid_levels = [
            num_recurrent_layers + 1 - int(math.log2(v))
            for v in self.args.pyramid_ranges
        ]

        max_pyr_range = (min(self.args.pyramid_ranges), max(self.args.pyramid_ranges))
        self.fnet = NeXt1DEncoder(
            max_pyr_range=max_pyr_range,
            stem_stride=self.args.enc_stem_stride,
            num_recurrent_layers=num_recurrent_layers,
            hidden_chs=self.args.enc_hidden_chs,
            out_chs=self.args.enc_out_chs,
            mlp_ratio=self.args.enc_mlp_ratio,
            norm_layer=LayerNorm2d,
            depth=self.args.enc_depth,
            fuse_next1d_weights=self.args.fuse_next1d_weights,
        )

        self.cnet = NeXt1DEncoder(
            max_pyr_range=max_pyr_range,
            stem_stride=self.args.enc_stem_stride,
            num_recurrent_layers=num_recurrent_layers,
            hidden_chs=self.args.enc_hidden_chs,
            out_chs=self.args.enc_out_chs,
            mlp_ratio=self.args.enc_mlp_ratio,
            norm_layer=LayerNorm2d,
            depth=self.args.enc_depth,
            fuse_next1d_weights=self.args.fuse_next1d_weights,
        )

        self.dim_corr = (self.args.corr_range * 2 + 1) ** 2 * self.args.corr_levels

        self.update_block = UpdateBlock(self.args)

        self.upnet_layer = nn.Sequential(
            nn.Conv2d(2 * self.args.dec_net_chs, self.args.dec_net_chs, 1),
            nn.ReLU(inplace=True),
            NeXt1DStage(
                self.args.dec_net_chs,
                self.args.dec_net_chs,
                stride=1,
                depth=2,
                mlp_ratio=args.dec_mlp_ratio,
                norm_layer=LayerNorm2d,
                fuse_next1d_weights=self.args.fuse_next1d_weights,
            ),
        )

        if self.args.corr_mode == "local" and alt_cuda_corr is None:
            print(
                "!!! alt_cuda_corr is not compiled! The slower IterativeCorrBlock will be used instead !!!"
            )

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--pyramid_ranges",
            type=int,
            nargs="+",
            default=(32, 8),
            help="(maximum, minimum) feature pyramid strides.",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=12,
            help="Total number of refinement iterations.",
        )

        parser.add_argument(
            "--corr_mode",
            type=str,
            choices=("local", "allpairs"),
            default="local",
            help="Correlation mode. Use local for low memory consumption or allpairs for maximum speed.",
        )
        parser.add_argument(
            "--corr_levels",
            type=int,
            default=1,
            help="Number or correlation pooling levels.",
        )
        parser.add_argument(
            "--corr_range",
            type=int,
            default=4,
            help="The correlation range will be 2*corr_range+1.",
        )

        parser.add_argument(
            "--enc_hidden_chs",
            type=int,
            default=64,
            help="Number of hidden channels in the encoder.",
        )
        parser.add_argument(
            "--enc_out_chs",
            type=int,
            default=128,
            help="Number of channels of the encoder features.",
        )
        parser.add_argument(
            "--enc_stem_stride",
            type=int,
            default=4,
            help="Stride of the stem layer, must be a power of 2.",
        )
        parser.add_argument(
            "--enc_mlp_ratio",
            type=float,
            default=4.0,
            help="Reverse bottleneck ratio in the encoder MLP.",
        )
        parser.add_argument(
            "--enc_depth",
            type=int,
            default=4,
            help="Number of NeXt1D blocks in the encoder.",
        )

        parser.add_argument(
            "--dec_net_chs",
            type=int,
            default=64,
            help="Number of net hidden channels in the decoder. Must follow: enc_out_chs=dec_net_chs+dec_inp_chs.",
        )
        parser.add_argument(
            "--dec_inp_chs",
            type=int,
            default=64,
            help="Number of input hidden channels in the decoder. Must follow: enc_out_chs=dec_net_chs+dec_inp_chs.",
        )
        parser.add_argument(
            "--dec_motion_chs",
            type=int,
            default=128,
            help="Number of channels of the motion encoder features.",
        )
        parser.add_argument(
            "--dec_depth",
            type=int,
            default=2,
            help="Number of NeXt1D blocks in the decoder.",
        )
        parser.add_argument(
            "--dec_mlp_ratio",
            type=float,
            default=4.0,
            help="Reverse bottleneck ratio in the decoder MLP.",
        )
        parser.add_argument(
            "--not_use_upsample_mask",
            action="store_false",
            dest="use_upsample_mask",
            help="If set, does not use convex upsampling.",
        )
        parser.add_argument(
            "--fuse_next1d_weights",
            action="store_true",
            help="If set, the NeXt1D conv layers will be fused into a single 2D layer. This requires to adapt the pretrained checkpoints before loading.",
        )
        parser.add_argument(
            "--simple_io",
            action="store_true",
            help="If set, the inputs and outputs will be simplified from dict to single torch.Tensor. This option should be used when exporting the model to ONNX. This will make the model incompatible with the PTLFlow framework.",
        )

        parser.add_argument(
            "--gamma",
            type=float,
            default=0.8,
            help="Used to compute the loss. Decaying factor for intermediate predictions.",
        )
        parser.add_argument(
            "--max_flow",
            type=float,
            default=400.0,
            help="Used to compute the loss. Groundtruth flows with magnitudes larger than this value are ignored.",
        )

        return parser

    def coords_grid(self, batch, ht, wd, dtype):
        coords = torch.meshgrid(
            torch.arange(ht, dtype=self.dtype, device=self.device),
            torch.arange(wd, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        coords = torch.stack(coords[::-1], dim=0).to(dtype=dtype)
        return coords[None].repeat(batch, 1, 1, 1)

    def upsample_flow(self, flow, mask, factor):
        """Upsample flow field [H/f, W/f, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, factor * H, factor * W)

    def forward(self, inputs):
        if self.args.simple_io:
            images = inputs
        else:
            images = inputs["images"]
        images, image_resizer = self.preprocess_images(
            images,
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=False,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        x1_raw = images[:, 0]
        x2_raw = images[:, 1]
        b, _, height_im, width_im = x1_raw.size()

        x_pyramid = self.fnet(torch.cat([x1_raw, x2_raw], 0))
        x1_pyramid = [x[:b] for x in x_pyramid]
        x2_pyramid = [x[b:] for x in x_pyramid]

        # outputs
        flows = []

        cnet_pyramid = self.cnet(x1_raw)

        pred_stride = min(self.args.pyramid_ranges)

        start_level, output_level = self.pyramid_levels
        pass_pyramid1 = x1_pyramid[start_level : output_level + 1]
        pass_pyramid2 = x2_pyramid[start_level : output_level + 1]
        pass_pyramid_cnet = cnet_pyramid[start_level : output_level + 1]

        iters_per_level = [
            int(math.ceil(float(self.args.iters) / (output_level - start_level + 1)))
        ] * (output_level - start_level + 1)

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = pass_pyramid1[0].size()
        init_device = pass_pyramid1[0].device

        if (
            not self.args.simple_io
            and "prev_flows" in inputs
            and inputs["prev_flows"] is not None
        ):
            flow = F.interpolate(
                inputs["prev_flows"][:, 0],
                [pass_pyramid1[0].shape[-2], pass_pyramid1[0].shape[-1]],
                mode="bilinear",
                align_corners=True,
            )
            flow = rescale_flow(flow, width_im, height_im, to_local=True)
            flow = forward_interpolate_batch(flow)
        else:
            flow = torch.zeros(
                b_size, 2, h_x1, w_x1, dtype=x1_raw.dtype, device=init_device
            )

        net = None
        for l, (x1, x2, cnet) in enumerate(
            zip(pass_pyramid1, pass_pyramid2, pass_pyramid_cnet)
        ):
            coords0 = self.coords_grid(x1.shape[0], x1.shape[2], x1.shape[3], x1.dtype)

            corr_fn = get_corr_block(
                x1,
                x2,
                self.args.corr_levels,
                self.args.corr_range,
                alternate_corr=self.args.corr_mode == "local",
            )

            net_tmp, inp = torch.split(
                cnet, [self.args.dec_net_chs, self.args.dec_inp_chs], dim=1
            )
            inp = torch.relu(inp)

            if net is None:
                net = torch.tanh(net_tmp)
            else:
                net = F.interpolate(
                    net,
                    [x1.shape[-2], x1.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )

                net_skip = torch.tanh(net_tmp)
                gate = torch.sigmoid(
                    self.upnet_layer(torch.cat([net, net_skip], dim=1))
                )
                net = gate * net + (1.0 - gate) * net_skip

            if l > 0:
                flow = rescale_flow(flow, x1.shape[-1], x1.shape[-2], to_local=False)
                flow = F.interpolate(
                    flow,
                    [x1.shape[-2], x1.shape[-1]],
                    mode="bilinear",
                    align_corners=True,
                )

            for k in range(iters_per_level[l]):
                flow = flow.detach()

                # correlation
                out_corr = corr_fn(coords0 + flow)

                get_mask = self.training or (
                    l == (output_level - start_level) and k == (iters_per_level[l] - 1)
                )
                flow_res, net, mask = self.update_block(
                    net, inp, out_corr, flow, get_mask=get_mask
                )
                flow = flow + flow_res

                out_flow = rescale_flow(flow, width_im, height_im, to_local=False)
                if self.training:
                    if mask is not None and l == (output_level - start_level):
                        if self.args.simple_io:
                            # Just copied the code from self.upsample_flow to here.
                            # For some reason, TensorRT backend does not compile when calling the function
                            N, _, H, W = out_flow.shape
                            mask = mask.view(N, 1, 9, pred_stride, pred_stride, H, W)
                            mask = torch.softmax(mask, dim=2)

                            up_flow = F.unfold(flow, [3, 3], padding=1)
                            up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

                            up_flow = torch.sum(mask * up_flow, dim=2)
                            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
                            up_flow = up_flow.reshape(
                                N, 2, pred_stride * H, pred_stride * W
                            )
                            out_flow = up_flow
                        else:
                            out_flow = self.upsample_flow(out_flow, mask, pred_stride)
                    else:
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                elif l == (output_level - start_level) and k == (
                    iters_per_level[l] - 1
                ):
                    if mask is not None:
                        if self.args.simple_io:
                            # Just copied the code from self.upsample_flow to here.
                            # For some reason, TensorRT backend does not compile when calling the function
                            N, _, H, W = out_flow.shape
                            mask = mask.view(N, 1, 9, pred_stride, pred_stride, H, W)
                            mask = torch.softmax(mask, dim=2)

                            up_flow = F.unfold(flow, [3, 3], padding=1)
                            up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

                            up_flow = torch.sum(mask * up_flow, dim=2)
                            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
                            up_flow = up_flow.reshape(
                                N, 2, pred_stride * H, pred_stride * W
                            )
                            out_flow = up_flow
                        else:
                            out_flow = self.upsample_flow(out_flow, mask, pred_stride)
                    else:
                        out_flow = F.interpolate(
                            out_flow,
                            [x1_raw.shape[-2], x1_raw.shape[-1]],
                            mode="bilinear",
                            align_corners=True,
                        )
                out_flow = self.postprocess_predictions(
                    out_flow, image_resizer, is_flow=True
                )
                flows.append(out_flow)

        if self.args.simple_io:
            return flows[-1]
        else:
            outputs = {}
            outputs["flows"] = flows[-1][:, None]
            if self.training:
                outputs["flow_preds"] = flows
            return outputs


class RAPIDFlow_it1(RAPIDFlow):
    def __init__(self, args):
        args.pyramid_ranges = (32, 32)
        args.iters = 1
        args.use_upsample_mask = False
        super().__init__(args)


class RAPIDFlow_it2(RAPIDFlow):
    def __init__(self, args):
        args.pyramid_ranges = (32, 16)
        args.iters = 2
        args.use_upsample_mask = False
        super().__init__(args)


class RAPIDFlow_it3(RAPIDFlow):
    def __init__(self, args):
        args.pyramid_ranges = (32, 8)
        args.iters = 3
        args.use_upsample_mask = True
        super().__init__(args)


class RAPIDFlow_it6(RAPIDFlow):
    def __init__(self, args):
        args.pyramid_ranges = (32, 8)
        args.iters = 6
        args.use_upsample_mask = True
        super().__init__(args)


class RAPIDFlow_it12(RAPIDFlow):
    def __init__(self, args):
        args.pyramid_ranges = (32, 8)
        args.iters = 12
        args.use_upsample_mask = True
        super().__init__(args)
