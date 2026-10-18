"""
DPT head implementation
Downstream heads assume inputs of size BCHW (B: batch, C: channels, H: height, W: width);
The DPT head implementation is based on DUSt3R and CroCoV2
References: https://github.com/naver/dust3r
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..libs.croco.dpt_block import make_fusion_block, make_nonlinearity, make_scratch, pair
from .base import PixelTaskOutput, PredictionHeadLayeredInput


@dataclass
class DPTFeatureInput:
    features_upsampled_8x: Tensor
    target_output_shape: Tuple[int, int]


# -------------------------------------------------------- DPT Feature --------------------------------------------------------


class DPTFeature(nn.Module):
    """
    DPT head implementation based on DUSt3R and CroCoV2

    Behavior:
    In forward, it will take in a list of Feature Tensors in BCHW (B, C, H//P, W//P)format,
    and return a upsampled feature tensor of shape (B, C, 8*(H//P), 8*(W//P)). This module
    should be used together with DPT[*]Processor to upsample the feature and
    interpolate when P is not 2^n to match the image shape exactly.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        main_tasks: Iterable[str] = ("rgb",),
        hooks: List[int] = [2, 5, 8, 11],
        input_feature_dims: Optional[Union[int, List[int]]] = 768,
        layer_dims: List[int] = [96, 192, 384, 768],
        feature_dim: int = 256,
        use_bn: bool = False,
        output_width_ratio=1,
        pretrained_checkpoint_path: str = None,
        checkpoint_gradient: bool = False,
        nonlinearity: str = "relu",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.checkpoint_gradient = checkpoint_gradient

        if isinstance(input_feature_dims, int):
            input_feature_dims = 4 * [input_feature_dims]
        else:
            input_feature_dims = input_feature_dims
            assert isinstance(input_feature_dims, List) and len(input_feature_dims) == 4

        self.input_feature_dims = input_feature_dims

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio, nonlinearity=nonlinearity)

        # delete resconfunit1 in refinement 4 because it is not used, and will cause error in DDP.
        del self.scratch.refinenet4.resConfUnit1

        if self.input_feature_dims is not None:
            self.init(input_feature_dims=input_feature_dims)

            # The rn convs are shared between scratch and input_process, so they would appear
            # multiple times in the state dict. The official UFM checkpoints only store the
            # input_process keys, so the duplicate scratch registrations are removed here.
            for rn_idx in range(1, 5):
                delattr(self.scratch, f"layer{rn_idx}_rn")
            delattr(self.scratch, "layer_rn")

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def init(self, input_feature_dims: Union[int, List[int]] = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.

        Args:
            input_feature_dims: Dimension of tokens coming from encoder
        """
        # Set up activation postprocessing layers
        if isinstance(input_feature_dims, int):
            input_feature_dims = 4 * [input_feature_dims]

        self.input_feature_dims = [dt * len(self.main_tasks) for dt in input_feature_dims]

        act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[2],
                out_channels=self.layer_dims[2],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        act_postprocess = [act_1_postprocess, act_2_postprocess, act_3_postprocess, act_4_postprocess]

        self.input_process = nn.ModuleList(
            [nn.Sequential(act_, layer_rn_) for act_, layer_rn_ in zip(act_postprocess, self.scratch.layer_rn)]
        )

    def forward(self, dpt_input: PredictionHeadLayeredInput) -> DPTFeatureInput:
        """
        DPT Feature forward pass from 4 layers in the transformer to 8x sampled feature output.

        Args:
            dpt_input (PredictionHeadLayeredInput): Input to the DPT feature head
            - list_features: List of 4 BCHW Tensors representing the features from 4 layers of the transformer

        Returns:
            DPTFeatureInput: Output of the DPT feature head
            - features_upsampled_8x: BCHW Tensor representing the 8x upsampled feature.
        """

        assert self.input_feature_dims is not None, "Need to call init(input_feature_dims) function first"

        layered_feats = dpt_input.list_features

        # check input dimensions
        for hook_idx, hook in enumerate(self.hooks):
            assert (
                layered_feats[hook].shape[1] == self.input_feature_dims[hook_idx]
            ), f"Input feature dimension mismatch at hook {hook}. Expected BCHW"

        if not self.checkpoint_gradient:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            # layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
            # # Project layers to chosen feature dim
            # layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
            layers = [self.input_process[idx](l) for idx, l in enumerate(layers)]

            # Fuse layers using refinement stages
            path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
            path_3 = self.scratch.refinenet3(path_4, layers[2])
            path_2 = self.scratch.refinenet2(path_3, layers[1])
            feature_upsampled_8x = self.scratch.refinenet1(path_2, layers[0])
        else:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            layers = [checkpoint(self.input_process[idx], l, use_reentrant=False) for idx, l in enumerate(layers)]

            path_4 = checkpoint(self.scratch.refinenet4, layers[3], use_reentrant=False)[
                :, :, : layers[2].shape[2], : layers[2].shape[3]
            ]
            path_3 = checkpoint(self.scratch.refinenet3, path_4, layers[2], use_reentrant=False)
            path_2 = checkpoint(self.scratch.refinenet2, path_3, layers[1], use_reentrant=False)
            feature_upsampled_8x = checkpoint(self.scratch.refinenet1, path_2, layers[0], use_reentrant=False)

        return DPTFeatureInput(
            features_upsampled_8x=feature_upsampled_8x, target_output_shape=dpt_input.target_output_shape
        )


# -------------------------------------------------------- DPT Processors --------------------------------------------------------


class DPTRegressionProcessor(nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,  # when not given, use input_feature_dim//2
        pretrained_checkpoint_path: str = None,
        checkpoint_gradient: bool = False,
        nonlinearity: str = "relu",
        *args,
        **kwargs,
    ):
        """
        DPT regression processor, takes 8x upsampled feature from DPT and furture upsamples to target shape

        It will interpolate the feature to match the target shape exactly, handling patch size not 2^n

        Args:
            input_feature_dim: Dimension of input feature
            output_dim: Dimension of output regression
            hidden_dims: [h1, h2] List of 2 hidden dimensions for intermediate. default is [input_feature_dim//2] * 2
            pretrained_checkpoint_path: Path to pretrained checkpoint (default: None)
        """

        super().__init__(*args, **kwargs)

        if hidden_dims is None:
            hidden_dims = [input_feature_dim // 2] * 2
        else:
            assert isinstance(hidden_dims, List) and len(hidden_dims) == 2

        self.checkpoint_gradient = checkpoint_gradient

        self.conv1 = nn.Conv2d(input_feature_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        # interpolate is dependent on target output size
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            make_nonlinearity(nonlinearity),
            nn.Conv2d(hidden_dims[1], output_dim, kernel_size=1, stride=1, padding=0),
        )

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT regression processor from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, dpt_processor_input: DPTFeatureInput):
        """
        DPT regression processor, process DPT output into channels to be adapted into regression output.

        Args:
            dpt_processor_input (DPTFeatureInput): Input to the processor
            - features_upsampled_8x: BCHW Tensor representing the upsampled feature
            - target_output_shape: Tuple of (H, W) representing the target output shape

        Returns:
            PixelTaskOutput: Output of the processor
            - decoded_channels: BCHW Tensor representing the regression output
        """

        x = dpt_processor_input.features_upsampled_8x
        output_shape = dpt_processor_input.target_output_shape

        if not self.checkpoint_gradient:
            x = self.conv1(x)
            x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=True)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=True)
            x = checkpoint(self.conv2, x, use_reentrant=False)

        return PixelTaskOutput(decoded_channels=x)


class DPTSegmentationProcessor(nn.Module):
    def __init__(
        self,
        input_feature_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,  # when not given, use input_feature_dim
        use_bn: bool = False,
        pretrained_checkpoint_path: str = None,
        *args,
        **kwargs,
    ):
        """
        DPT segmentation processor, takes 8x upsampled feature from DPT and furture upsamples to target shape.
        This version differs slightly from the regression processor.

        It will interpolate the feature to match the target shape exactly, handling patch size not 2^n

        Args:
            input_feature_dim: Dimension of input feature
            output_dim: Dimension of output regression
            hidden_dim: h1 Hidden dimension for intermediate. default is input_feature_dim
            use_bn: Whether to use batch normalization, default is False
            pretrained_checkpoint_path: Path to pretrained checkpoint (default: None)
        """

        super().__init__(*args, **kwargs)

        if hidden_dim is None:
            hidden_dim = input_feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT segmentation processor from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def forward(self, dpt_processor_input: DPTFeatureInput):
        """
        Forward pass for the DPT segmentation processor, process DPT output into channels
        to be adapted into segmentation mask.

        Args:
            dpt_processor_input (DPTFeatureInput): Input to the processor
            - features_upsampled_8x: BCHW Tensor representing the upsampled feature
            - target_output_shape: Tuple of (H, W) representing the target output shape

        Returns:
            PixelTaskOutput: Output of the processor
            - decoded_channels: BCHW Tensor representing the segmentation mask
        """

        x = dpt_processor_input.features_upsampled_8x
        output_shape = dpt_processor_input.target_output_shape

        x = self.conv(x)
        x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=True)

        return PixelTaskOutput(decoded_channels=x)


# ---------------------------------------- DPT Feature 2x upsample ----------------------------------------
class DPTFeatureDoubleUpsampling(nn.Module):
    """
    DPT head implementation based on DUSt3R and CroCoV2

    Behavior:
    In forward, it will take in a list of Feature Tensors in BCHW (B, C, H//P, W//P)format,
    and return a upsampled feature tensor of shape (B, C, 8*(H//P), 8*(W//P)). This module
    should be used together with DPT[*]Processor to upsample the feature and
    interpolate when P is not 2^n to match the image shape exactly.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        main_tasks: Iterable[str] = ("rgb",),
        hooks: List[int] = [0, 1],
        input_feature_dims: Optional[Union[int, List[int]]] = 768,
        layer_dims: List[int] = [384, 768],
        feature_dim: int = 256,
        use_bn: bool = False,
        output_width_ratio=1,
        pretrained_checkpoint_path: str = None,
        checkpoint_gradient: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.checkpoint_gradient = checkpoint_gradient

        if isinstance(input_feature_dims, int):
            input_feature_dims = 2 * [input_feature_dims]
        else:
            input_feature_dims = input_feature_dims
            assert isinstance(input_feature_dims, List) and len(input_feature_dims) == 2

        self.input_feature_dims = input_feature_dims

        self.scratch = self.make_scratch_2(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)

        # delete resconfunit1 in refinement 4 because it is not used, and will cause error in DDP.
        del self.scratch.refinenet4.resConfUnit1

        if self.input_feature_dims is not None:
            self.init(input_feature_dims=input_feature_dims)

        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained DPT dense feature head from {self.pretrained_checkpoint_path}")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

    def make_scratch_2(self, in_shape, out_shape, groups=1, expand=False):
        scratch = nn.Module()

        out_shape3 = out_shape
        out_shape4 = out_shape
        if expand == True:
            out_shape3 = out_shape * 4
            out_shape4 = out_shape * 8

        scratch.layer3_rn = nn.Conv2d(
            in_shape[0],
            out_shape3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        scratch.layer4_rn = nn.Conv2d(
            in_shape[1],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        scratch.layer_rn = nn.ModuleList(
            [
                scratch.layer3_rn,
                scratch.layer4_rn,
            ]
        )

        return scratch

    def init(self, input_feature_dims: Union[int, List[int]] = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.

        Args:
            input_feature_dims: Dimension of tokens coming from encoder
        """
        # Set up activation postprocessing layers
        if isinstance(input_feature_dims, int):
            input_feature_dims = 2 * [input_feature_dims]

        self.input_feature_dims = [dt * len(self.main_tasks) for dt in input_feature_dims]

        act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_feature_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        act_postprocess = [act_3_postprocess, act_4_postprocess]

        self.input_process = nn.ModuleList(
            [nn.Sequential(act_, layer_rn_) for act_, layer_rn_ in zip(act_postprocess, self.scratch.layer_rn)]
        )

    def forward(self, dpt_input: PredictionHeadLayeredInput) -> DPTFeatureInput:
        """
        DPT Feature forward pass from 4 layers in the transformer to 8x sampled feature output.

        Args:
            dpt_input (PredictionHeadLayeredInput): Input to the DPT feature head
            - list_features: List of 4 BCHW Tensors representing the features from 4 layers of the transformer

        Returns:
            DPTFeatureInput: Output of the DPT feature head
            - features_upsampled_8x: BCHW Tensor representing the 8x upsampled feature.
        """

        assert self.input_feature_dims is not None, "Need to call init(input_feature_dims) function first"

        layered_feats = dpt_input.list_features

        # check input dimensions
        for hook_idx, hook in enumerate(self.hooks):
            assert (
                layered_feats[hook].shape[1] == self.input_feature_dims[hook_idx]
            ), f"Input feature dimension mismatch at hook {hook}. Expected BCHW"

        if not self.checkpoint_gradient:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            # layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
            # # Project layers to chosen feature dim
            # layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
            layers = [self.input_process[idx](l) for idx, l in enumerate(layers)]

            # Fuse layers using refinement stages
            path_4 = self.scratch.refinenet4(layers[1])[:, :, : layers[0].shape[2], : layers[0].shape[3]]
            feature_upsampled_2x = self.scratch.refinenet3(path_4, layers[0])
        else:
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layered_feats[hook] for hook in self.hooks]

            layers = [checkpoint(self.input_process[idx], l, use_reentrant=False) for idx, l in enumerate(layers)]

            path_4 = checkpoint(self.scratch.refinenet4, layers[1], use_reentrant=False)[
                :, :, : layers[0].shape[2], : layers[0].shape[3]
            ]
            feature_upsampled_2x = checkpoint(self.scratch.refinenet3, path_4, layers[0], use_reentrant=False)

        return DPTFeatureInput(
            features_upsampled_8x=feature_upsampled_2x, target_output_shape=dpt_input.target_output_shape
        )


if __name__ == "__main__":
    import numpy as np

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Ensure the model is on GPU
    num_runs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model and move to GPU
    dpt_feature_output = DPTFeature(
        patch_size=16,
        main_tasks=("rgb",),
        hooks=[2, 5, 8, 11],
        input_feature_dims=[1024, 768, 768, 768],
        layer_dims=[96, 192, 384, 768],
        feature_dim=256,
        use_bn=False,
        output_width_ratio=1,
        checkpoint_gradient=True,
    ).to(device)

    postprocess = DPTRegressionProcessor(input_feature_dim=256, output_dim=3, checkpoint_gradient=True).to(device)

    # Define input shape
    image_shape = (560, 420)
    batch_size = 12
    patch_size = 14

    patch_num = (image_shape[0] // patch_size, image_shape[1] // patch_size)

    input_feats = [None for _ in range(12)]

    input_feats[2] = torch.randn(batch_size, 1024, *patch_num, device=device, requires_grad=True)
    input_feats[5] = torch.randn(batch_size, 768, *patch_num, device=device, requires_grad=True)
    input_feats[8] = torch.randn(batch_size, 768, *patch_num, device=device, requires_grad=True)
    input_feats[11] = torch.randn(batch_size, 768, *patch_num, device=device, requires_grad=True)

    # Warm-up to stabilize GPU performance
    for _ in range(3):
        output = dpt_feature_output(
            PredictionHeadLayeredInput(list_features=input_feats, target_output_shape=image_shape)
        )
        output2 = postprocess(output)
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
        output = dpt_feature_output(
            PredictionHeadLayeredInput(list_features=input_feats, target_output_shape=image_shape)
        )
        output2 = postprocess(output)
        end_event.record()
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)  # Time in milliseconds

        # Backward pass
        start_event.record()
        output = dpt_feature_output(
            PredictionHeadLayeredInput(list_features=input_feats, target_output_shape=image_shape)
        )
        output2 = postprocess(output)
        output2.decoded_channels.sum().backward()
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
