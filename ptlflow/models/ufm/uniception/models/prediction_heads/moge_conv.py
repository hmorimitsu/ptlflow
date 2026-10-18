"""
MoGe Conv Decoder Implementation
References: https://github.com/microsoft/MoGe/blob/main/moge/model/v1.py
"""

from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint

from .base import PixelTaskOutput, PredictionHeadLayeredInput

class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
    ):
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "silu":
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == "elu":
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == "group_norm" else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv


class MoGeConvFeature(nn.Module):
    def __init__(
        self,
        patch_size: int,
        # MoGe parameters
        num_features: int,
        input_feature_dims: Union[int, List[int]],
        dim_out: List[int],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 64],
        dim_times_res_block_hidden: int = 2,
        num_res_blocks: int = 2,
        res_block_norm: Literal["group_norm", "layer_norm"] = "group_norm",
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        # UniCeption parameters
        pretrained_checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        if isinstance(input_feature_dims, int):
            input_feature_dims = [input_feature_dims] * num_features
        self.input_feature_dims = input_feature_dims

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.input_feature_dims[i],
                    out_channels=dim_proj,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for i in range(num_features)
            ]
        )

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch + 2, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
            ]
        )

        self.output_block = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1] + 2,
                    dim_out_,
                    dim_times_res_block_hidden,
                    last_res_blocks,
                    last_conv_channels,
                    last_conv_size,
                    res_block_norm,
                )
                for dim_out_ in dim_out
            ]
        )

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: Literal["group_norm", "layer_norm"],
    ):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode="replicate",
            ),
        )

    # @torch.compile(fullgraph=True, options={}, dynamic=True)
    def forward(self, head_input: PredictionHeadLayeredInput) -> PixelTaskOutput:
        img_h, img_w = head_input.target_output_shape
        patch_h, patch_w = img_h // self.patch_size, img_w // self.patch_size

        # Process the hidden states
        x: torch.Tensor = torch.stack(
            [proj(feat.contiguous()) for proj, feat in zip(self.projects, head_input.list_features)], dim=1
        ).sum(dim=1)

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(
                width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(
            width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            raise NotImplementedError()

        return PixelTaskOutput(decoded_channels=torch.cat(output, dim=1))


if __name__ == "__main__":
    import time

    import numpy as np
    import torch.cuda.profiler as profiler

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Ensure the model is on GPU
    num_runs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and move to GPU
    head = MoGeConvFeature(
        patch_size=14,
        num_features=4,
        input_feature_dims=[1024, 768, 768, 768],
        dim_out=[2, 1],
        dim_proj=512,
        dim_upsample=[256, 128, 64],
        dim_times_res_block_hidden=2,
        num_res_blocks=2,
        res_block_norm="group_norm",
        last_res_blocks=0,
        last_conv_channels=32,
        last_conv_size=1,
        pretrained_checkpoint_path=None,
    ).to(device)

    # Define input shape
    image_shape = (560, 420)
    batch_size = 10
    patch_size = 14
    patch_num = (image_shape[0] // patch_size, image_shape[1] // patch_size)

    # Generate input features and move to GPU
    input_feats = [
        torch.randn(batch_size, dim, *patch_num, device=device, requires_grad=True) for dim in [1024, 768, 768, 768]
    ]

    # Wrap input into PredictionHeadLayeredInput
    model_input = PredictionHeadLayeredInput(list_features=input_feats, target_output_shape=image_shape)

    with torch.autocast("cuda", dtype=torch.float16):
        # Warm-up to stabilize GPU performance
        for _ in range(3):
            output = head(model_input)
            output.decoded_channels.sum().backward()
            torch.cuda.synchronize()

        # Clear memory cache
        torch.cuda.empty_cache()

        # Lists to store results
        forward_times = []
        backward_times = []
        memory_usages = []

        for _ in range(num_runs):
            # Start measuring time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.max_memory_allocated(device)

            # Forward pass
            start_event.record()
            output = head(model_input)
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)  # Time in milliseconds

            # Backward pass
            start_event.record()
            output.decoded_channels.sum().backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)

            # Memory usage
            memory_after = torch.cuda.max_memory_allocated(device)
            peak_memory = memory_after - memory_before

            forward_times.append(forward_time)
            backward_times.append(backward_time)
            memory_usages.append(peak_memory / 1e6)  # Convert to MB

        # Compute mean and standard deviation
        fwd_mean, fwd_std = np.mean(forward_times), np.std(forward_times)
        bwd_mean, bwd_std = np.mean(backward_times), np.std(backward_times)
        mem_mean, mem_std = np.mean(memory_usages), np.std(memory_usages)

        print(f"Forward Pass Time: {fwd_mean:.2f} ± {fwd_std:.2f} ms")
        print(f"Backward Pass Time: {bwd_mean:.2f} ± {bwd_std:.2f} ms")
        print(f"Peak GPU Memory Usage: {mem_mean:.2f} ± {mem_std:.2f} MB")
