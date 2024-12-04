from typing import Optional

import torch.nn.functional as F

from ptlflow.utils.registry import register_model
from .FlowFormer.encoders import twins_svt_large, convnext_large
from .FlowFormer.PerCostFormer3.encoder import MemoryEncoder
from .FlowFormer.PerCostFormer3.decoder import MemoryDecoder
from .FlowFormer.PerCostFormer3.cnn import BasicEncoder
from .utils import compute_grid_indices, compute_weight
from ..base_model.base_model import BaseModel


class FlowFormerPlusPlus(BaseModel):
    pretrained_checkpoints = {
        "chairs": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-chairs-a7745dd5.ckpt",
        "things": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things-4db3ecff.ckpt",
        "things288960": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-things_288960-a4291d41.ckpt",
        "sintel": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-sintel-d14a1968.ckpt",
        "kitti": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flowformerplusplus-kitti-65b828c3.ckpt",
    }

    def __init__(
        self,
        cnet: str = "twins",
        fnet: str = "twins",
        pretrain: bool = False,
        patch_size: int = 8,
        cost_heads_num: int = 1,
        cost_latent_input_dim: int = 64,
        cost_latent_token_num: int = 8,
        cost_latent_dim: int = 128,
        pe: str = "linear",
        encoder_depth: int = 3,
        encoder_latent_dim: int = 256,
        decoder_depth: int = 32,
        dropout: float = 0.0,
        vert_c_dim: int = 64,
        query_latent_dim: int = 64,
        cost_encoder_res: bool = True,
        pic_size: tuple[int, int, int, int] = (368, 496, 368, 496),
        del_layers: bool = True,
        pretrain_mode: bool = False,
        use_convertor: bool = False,
        patch_embed: str = "single",
        cross_attn: str = "all",
        droppath: float = 0.0,
        vertical_encoder_attn: str = "twins",
        use_patch: bool = False,
        fix_pe: bool = False,
        gt_r: int = 15,
        flow_or_pe: str = "and",
        no_sc: bool = False,
        r_16: int = -1,
        quater_refine: bool = False,
        use_rpe: bool = False,
        gma: str = "GMA",
        detach_local: bool = False,
        use_tile_input: bool = True,
        tile_height: int = 432,
        tile_sigma: float = 0.05,
        train_size: Optional[tuple[int, int]] = None,
        crop_cost_volume: bool = False,
        mask_ratio: float = 0.0,
        attn_dim: int = 0,
        expand_factor: float = 4.0,
        query_num: int = 0,
        no_border: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(output_stride=32, loss_fn=None, **kwargs)

        self.cnet = cnet
        self.fnet = fnet
        self.cost_encoder_res = cost_encoder_res
        self.cost_heads_num = cost_heads_num
        self.cost_latent_dim = cost_latent_dim
        self.cost_latent_input_dim = cost_latent_input_dim
        self.cost_latent_token_num = cost_latent_token_num
        self.decoder_depth = decoder_depth
        self.dropout = dropout
        self.encoder_depth = encoder_depth
        self.encoder_latent_dim = encoder_latent_dim
        self.gma = gma
        self.patch_size = patch_size
        self.pe = pe
        self.pretrain = pretrain
        self.query_latent_dim = query_latent_dim
        self.vert_c_dim = vert_c_dim
        self.use_tile_input = use_tile_input
        self.tile_height = tile_height
        self.tile_sigma = tile_sigma
        self.train_size = train_size
        self.pic_size = pic_size
        self.del_layers = del_layers
        self.pretrain_mode = pretrain_mode
        self.use_convertor = use_convertor
        self.patch_embed = patch_embed
        self.cross_attn = cross_attn
        self.droppath = droppath
        self.vertical_encoder_attn = vertical_encoder_attn
        self.use_patch = use_patch
        self.fix_pe = fix_pe
        self.gt_r = gt_r
        self.flow_or_pe = flow_or_pe
        self.no_sc = no_sc
        self.r_16 = r_16
        self.quater_refine = quater_refine
        self.use_rpe = use_rpe
        self.detach_local = detach_local
        self.crop_cost_volume = crop_cost_volume
        self.mask_ratio = mask_ratio
        self.attn_dim = attn_dim
        self.expand_factor = expand_factor
        self.query_num = query_num
        self.no_border = no_border

        H1, W1, H2, W2 = pic_size
        H_offset = (H1 - H2) // 2
        W_offset = (W1 - W2) // 2

        self.memory_encoder = MemoryEncoder(
            cost_heads_num=cost_heads_num,
            use_convertor=use_convertor,
            r_16=r_16,
            crop_cost_volume=crop_cost_volume,
            del_layers=del_layers,
            H_offset=H_offset,
            W_offset=W_offset,
            fnet=fnet,
            pretrain=pretrain,
            pretrain_mode=pretrain_mode,
            encoder_latent_dim=encoder_latent_dim,
            vertical_encoder_attn=vertical_encoder_attn,
            cost_latent_token_num=cost_latent_token_num,
            cost_latent_dim=cost_latent_dim,
            cost_encoder_res=cost_encoder_res,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            cost_latent_input_dim=cost_latent_input_dim,
            pe=pe,
            encoder_depth=encoder_depth,
            cross_attn=cross_attn,
            dropout=dropout,
            vert_c_dim=vert_c_dim,
            patch_embed=patch_embed,
            use_rpe=use_rpe,
            droppath=droppath,
            attn_dim=attn_dim,
            expand_factor=expand_factor,
        )
        self.memory_decoder = MemoryDecoder(
            gma=gma,
            use_patch=use_patch,
            detach_local=detach_local,
            use_rpe=use_rpe,
            r_16=r_16,
            quater_refine=quater_refine,
            fix_pe=fix_pe,
            gt_r=gt_r,
            query_num=query_num,
            no_border=no_border,
            W_offset=W_offset,
            H_offset=H_offset,
            query_latent_dim=query_latent_dim,
            cost_latent_input_dim=cost_latent_input_dim,
            cost_heads_num=cost_heads_num,
            encoder_latent_dim=encoder_latent_dim,
            decoder_depth=decoder_depth,
            cost_latent_dim=cost_latent_dim,
            patch_size=patch_size,
            flow_or_pe=flow_or_pe,
            dropout=dropout,
            pe=pe,
            no_sc=no_sc,
        )
        if cnet == "twins":
            self.context_encoder = twins_svt_large(
                pretrained=self.pretrain, del_layers=del_layers
            )
        elif cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")
        elif cnet == "convnext":
            self.context_encoder = convnext_large(pretrained=self.pretrain)

        if pretrain_mode:
            print("[In pretrain mode, freeze context encoder]")
            for param in self.context_encoder.parameters():
                param.requires_grad = False

        self.showed_warning = False

    def forward(self, inputs, mask=None, output=None):
        """Estimate optical flow between pair of frames"""
        if self.pretrain_mode:
            image1 = (image1 + 1) * 127.5
            image2 = (image2 + 1) * 127.5
            loss = self.pretrain_forward(image1, image2, mask=mask, output=output)
            return loss

        if self.train_size is not None:
            train_size = self.train_size
        else:
            train_size = self.train_size

        if self.use_tile_input and train_size is None and not self.showed_warning:
            print(
                "WARNING: --train_size is not provided and it cannot be loaded from the checkpoint either. Flowformer++ will run without input tile."
            )
            self.showed_warning = True

        if self.use_tile_input and train_size is not None:
            return self.forward_tile(inputs, train_size)
        else:
            return self.forward_pad(inputs)

    def forward_pad(self, inputs):
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        prev_flow = None
        if (
            inputs.get("prev_preds") is not None
            and inputs["prev_preds"].get("flow_small") is not None
        ):
            prev_flow = inputs["prev_preds"]["flow_small"]

        flow_predictions, flow_small = self.predict(
            images[:, 0], images[:, 1], prev_flow
        )
        output_flow = flow_predictions[-1]

        if self.training:
            for i, p in enumerate(flow_predictions):
                flow_predictions[i] = self.postprocess_predictions(
                    p, image_resizer, is_flow=True
                )
            outputs = {
                "flows": flow_predictions[-1][:, None],
                "flow_preds": flow_predictions,
            }
        else:
            output_flow = self.postprocess_predictions(
                output_flow, image_resizer, is_flow=True
            )
            outputs = {"flows": output_flow[:, None], "flow_small": flow_small}

        return outputs

    def forward_tile(self, inputs, train_size):
        input_size = inputs["images"].shape[-2:]
        image_size = (max(self.tile_height, input_size[-2]), input_size[-1])
        hws = compute_grid_indices(image_size, train_size)
        device = inputs["images"].device
        weights = compute_weight(
            hws, image_size, train_size, self.tile_sigma, device=device
        )

        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            target_size=image_size,
            pad_two_side=False,
            pad_mode="constant",
            pad_value=-1,
        )

        image1 = images[:, 0]
        image2 = images[:, 1]

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + train_size[0], w : w + train_size[1]]
            image2_tile = image2[:, :, h : h + train_size[0], w : w + train_size[1]]

            flow_predictions, _ = self.predict(image1_tile, image2_tile)
            flow_pre = flow_predictions[-1]

            padding = (
                w,
                image_size[1] - w - train_size[1],
                h,
                image_size[0] - h - train_size[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        output_flow = flows / flow_count

        output_flow = self.postprocess_predictions(
            output_flow, image_resizer, is_flow=True
        )
        return {"flows": output_flow[:, None]}

    def predict(self, image1, image2, prev_flow=None):
        """Estimate optical flow between pair of frames"""
        data = {}

        context, _ = self.context_encoder(image1)
        context_quater = None

        (
            cost_memory,
            cost_patches,
            feat_s_quater,
            feat_t_quater,
        ) = self.memory_encoder(image1, image2, data, context)

        flow_predictions, flow_small = self.memory_decoder(
            cost_memory,
            context,
            context_quater,
            feat_s_quater,
            feat_t_quater,
            data,
            prev_flow=prev_flow,
            cost_patches=cost_patches,
        )

        return flow_predictions, flow_small


@register_model
class flowformer_pp(FlowFormerPlusPlus):
    pass
