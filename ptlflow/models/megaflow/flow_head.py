import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple, Union, Optional
from .model_utils import create_uv_grid, position_grid_to_embed


class FlowFeature(nn.Module):
    """
    Flow head that processes input tokens and predicts downsample features for flow estimation.

    Behavior:
    In forward, it will take in a list of Feature Tensors in BCHW (B, C, H//P, W//P)format,
    and return a upsampled feature tensor of shape (B, C, (H//8), (W//8)).
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 128,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: Optional[List[int]] = None,
        num_layers: int = 24,
        pos_embed: bool = True,
        down_ratio: int = 0,
        fuse_cnn: bool = False,
        cnn_feature_channels: List[int] = [64, 128, 256],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        if intermediate_layer_idx is None:
            if num_layers == 24:
                self.intermediate_layer_idx = [4, 11, 17, 23]
            elif num_layers == 12:
                self.intermediate_layer_idx = [2, 5, 8, 11]
            elif num_layers == 6:
                self.intermediate_layer_idx = [1, 2, 4, 5]
            else:
                step = num_layers // 4
                self.intermediate_layer_idx = [((i + 1) * step) - 1 for i in range(4)]
        else:
            self.intermediate_layer_idx = intermediate_layer_idx
        self.fuse_cnn = fuse_cnn

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        resize_ratio = 2 if down_ratio == 0 else 4

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=resize_ratio,
                    stride=resize_ratio,
                    padding=0,
                ),  # upsample 2
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),  # upsample 2
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),  # downsample 2
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1, kernel_size=3, stride=1, padding=1
        )

        if self.fuse_cnn:
            self.concat_project = nn.ModuleList(
                [
                    nn.Conv2d(
                        cnn_feature_channels[0] * 4**2 + out_channels[0],
                        out_channels[0],
                        1,
                    ),  # 1/2 -> 1/8 -> 1/7
                    nn.Conv2d(
                        cnn_feature_channels[1] * 2**2 + out_channels[1],
                        out_channels[1],
                        1,
                    ),  # 1/4 -> 1/8 -> 1/7
                ]
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int = 4,
        frames_chunk_size: int = 8,
        downsample_factor: int = 0,
        cnn_features=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        down_ratio = self.down_ratio if downsample_factor == 0 else downsample_factor
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                down_ratio=down_ratio,
                cnn_features=cnn_features,
            )

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            cnn_features_chunk = []
            for i, feat in enumerate(cnn_features):
                feat = feat.view(B, -1, *feat.shape[1:])[
                    :, frames_start_idx:frames_end_idx
                ].contiguous()
                cnn_features_chunk.append(feat.view(-1, *feat.shape[2:]))

            # Process batch of frames
            chunk_output = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
                down_ratio=down_ratio,
                cnn_features=cnn_features_chunk,
            )
            all_preds.append(chunk_output)  # Only supports scale 1 for now

        # return feature only
        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
        down_ratio: int = 8,
        cnn_features=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(
            out,
            add_size=(H // down_ratio, W // down_ratio) if down_ratio > 0 else None,
            cnn_features=cnn_features,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        return out.view(B, S, *out.shape[1:])

    def _apply_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(
        self, features: List[torch.Tensor], add_size=None, cnn_features=None
    ) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        if self.fuse_cnn and cnn_features is not None:

            # Use pixel unshuffle for downsampling
            cnn1_8 = F.pixel_unshuffle(cnn_features[0], downscale_factor=4)
            cnn2_8 = F.pixel_unshuffle(cnn_features[1], downscale_factor=2)
            cnn_1 = custom_interpolate(
                cnn1_8, size=layer_1.shape[-2:], mode="bilinear", align_corners=True
            )
            cnn_2 = custom_interpolate(
                cnn2_8, size=layer_2.shape[-2:], mode="bilinear", align_corners=True
            )
            layer_1 = self.concat_project[0](torch.cat([cnn_1, layer_1], dim=1))
            layer_2 = self.concat_project[1](torch.cat([cnn_2, layer_2], dim=1))

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])  # 1/14
        del layer_4_rn, layer_4

        out_7 = self.scratch.refinenet3(
            out, layer_3_rn, size=layer_2_rn.shape[2:]
        )  # 1/7
        del layer_3_rn, layer_3

        out_3 = self.scratch.refinenet2(
            out_7, layer_2_rn, size=layer_1_rn.shape[2:]
        )  # 1/3.5

        del layer_2_rn, layer_2

        out_4 = self.scratch.refinenet1(
            out_3,
            layer_1_rn,
            size=add_size if add_size is not None else layer_1_rn.shape[2:],
        )  # resize to 1/8

        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out_4)

        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(
    features: int, size: int = None, has_residual: bool = True, groups: int = 1
) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=self.groups,
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(
                features, activation, bn, groups=self.groups
            )

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(
            features, activation, bn, groups=self.groups
        )

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk, size=size, mode=mode, align_corners=align_corners
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(
            x, size=size, mode=mode, align_corners=align_corners
        )


def soft_match_features(prob, f1):
    B, C, H, W = f1.shape
    f1_flat = f1.view(B, C, -1)  # [B, C, H*W]
    matched = torch.matmul(prob, f1_flat.permute(0, 2, 1))  # [B, H*W, C]
    return matched.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]


class VisConfHead(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(2 * C + 1, 2 * C, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * C, 2, 3, padding=1),  # [vis, conf]
        )

    def forward(self, f0, f1, prob):
        matched_f1 = soft_match_features(prob, f1)
        max_prob, _ = prob.max(dim=-1)  # [B, H*W]
        B, C, H, W = f0.shape
        max_prob_map = max_prob.view(B, 1, H, W)
        inp = torch.cat([f0, matched_f1, max_prob_map], dim=1)
        return self.head(inp)
