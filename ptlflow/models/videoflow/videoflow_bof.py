from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

from .Networks.BOFNet.update import GMAUpdateBlock
from .Networks.encoders import twins_svt_large
from .Networks.BOFNet.cnn import BasicEncoder
from .Networks.BOFNet.corr import CorrBlock, AlternateCorrBlock
from .utils import coords_grid
from .Networks.BOFNet.gma import Attention
from .Networks.BOFNet.sk import SKUpdateBlock6_Deep_nopoolres_AllDecoder
from .Networks.BOFNet.sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
from ..base_model.base_model import BaseModel


class VideoFlowBOF(BaseModel):
    pretrained_checkpoints = {
        "things_288960": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/videoflow_bof-things_288960noise-d581490a.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/videoflow_bof-sintel-c2010097.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/videoflow_bof-kitti-fa9af79c.ckpt",
    }

    def __init__(self, args: Namespace) -> None:
        super().__init__(args=args, loss_fn=None, output_stride=8)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        args.corr_radius = 4
        args.corr_levels = 4

        # feature network, context network, and update block
        if args.cnet == "twins":
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.args.pretrain)
        elif args.cnet == "basicencoder":
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn="instance")

        if args.fnet == "twins":
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.args.pretrain)
        elif args.fnet == "basicencoder":
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance")

        if self.args.gma == "GMA":
            print("[Using GMA]")
            self.update_block = GMAUpdateBlock(self.args, hidden_dim=128)
        elif self.args.gma == "GMA-SK":
            print("[Using GMA-SK]")
            self.args.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(
                args=self.args, hidden_dim=128
            )
        elif self.args.gma == "GMA-SK2":
            print("[Using GMA-SK2]")
            self.args.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(
                args=self.args, hidden_dim=128
            )

        print("[Using corr_fn {}]".format(self.args.corr_fn))

        self.att = Attention(
            args=self.args, dim=128, heads=1, max_pos_size=160, dim_head=128
        )

        self.has_showed_warning = False

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--corr_levels", type=int, default=4)
        parser.add_argument("--corr_radius", type=int, default=4)
        parser.add_argument(
            "--cnet", type=str, choices=("twins", "basicencoder"), default="twins"
        )
        parser.add_argument(
            "--fnet", type=str, choices=("twins", "basicencoder"), default="twins"
        )
        parser.add_argument(
            "--gma", type=str, choices=("GMA", "GMA-SK", "GMA-SK2"), default="GMA-SK2"
        )
        parser.add_argument("--no_pretrain", action="store_false", dest="pretrain")
        parser.add_argument(
            "--corr_fn", type=str, choices=("default", "efficient"), default="default"
        )
        parser.add_argument("--decoder_depth", type=int, default=32)
        return parser

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, dtype=img.dtype, device=img.device)

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
        images = self._check_input_shape(inputs["images"])

        images, image_resizer = self.preprocess_images(
            images,
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        B, N, _, H, W = images.shape

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmaps = self.fnet(images.reshape(B * N, 3, H, W)).reshape(
            B, N, -1, H // 8, W // 8
        )
        fmap1 = fmaps[:, 0, ...]
        fmap2 = fmaps[:, 1, ...]
        fmap3 = fmaps[:, 2, ...]

        if self.args.corr_fn == "efficient":
            corr_fn_21 = AlternateCorrBlock(fmap2, fmap1, radius=self.args.corr_radius)
            corr_fn_23 = AlternateCorrBlock(fmap2, fmap3, radius=self.args.corr_radius)
        else:
            corr_fn_21 = CorrBlock(
                fmap2,
                fmap1,
                num_levels=self.args.corr_levels,
                radius=self.args.corr_radius,
            )
            corr_fn_23 = CorrBlock(
                fmap2,
                fmap3,
                num_levels=self.args.corr_levels,
                radius=self.args.corr_radius,
            )

        cnet = self.cnet(images[:, 1, ...])
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        attention = self.att(inp)

        coords0_21, coords1_21 = self.initialize_flow(images[:, 0, ...])
        coords0_23, coords1_23 = self.initialize_flow(images[:, 0, ...])

        flow_predictions = []
        for itr in range(self.args.decoder_depth):
            coords1_21 = coords1_21.detach()
            coords1_23 = coords1_23.detach()

            corr21 = corr_fn_21(coords1_21)
            corr23 = corr_fn_23(coords1_23)
            corr = torch.cat([corr23, corr21], dim=1)

            flow21 = coords1_21 - coords0_21
            flow23 = coords1_23 - coords0_23
            flow = torch.cat([flow23, flow21], dim=1)

            net, up_mask, delta_flow = self.update_block(
                net, inp, corr, flow, attention
            )

            up_mask_21, up_mask_23 = torch.split(up_mask, [64 * 9, 64 * 9], dim=1)

            coords1_23 = coords1_23 + delta_flow[:, 0:2, ...]
            coords1_21 = coords1_21 + delta_flow[:, 2:4, ...]

            # upsample predictions
            flow_up_23 = self.upsample_flow(coords1_23 - coords0_23, up_mask_23)
            flow_up_21 = self.upsample_flow(coords1_21 - coords0_21, up_mask_21)
            flow_up_23 = self.postprocess_predictions(
                flow_up_23, image_resizer, is_flow=True
            )
            flow_up_21 = self.postprocess_predictions(
                flow_up_21, image_resizer, is_flow=True
            )

            flow_predictions.append(torch.stack([flow_up_23, flow_up_21], dim=1))

        if self.training:
            outputs = {
                "flows": flow_up_23[:, None],
                "flows_bw": flow_up_21[:, None],
                "flow_preds": flow_predictions,
            }
        else:
            outputs = {
                "flows": flow_up_23[:, None],
                "flow_small": coords1_23 - coords0_23,
                "flow_bw_small": coords1_21 - coords0_21,
            }

        return outputs

    def _check_input_shape(self, images):
        assert (
            images.shape[1] <= 3
        ), f"videoflow_bof requires inputs of 3 frames. The current input has too many frames (found {images.shape[1]}), decrease it to 3 to run."
        if images.shape[1] == 2:
            if not self.has_showed_warning:
                print(
                    "Warning: videoflow_bof requires inputs of 3 frames, but the current input has only 2. "
                    "The first frame will be replicated to run, which may decrease the prediction accuracy. "
                    "If using validate.py or test.py, add the arguments seqlen_3-seqpos_middle to the [val/test]_dataset arg."
                )
                self.has_showed_warning = True
            images = torch.cat([images[:, :1], images], 1)
        return images
