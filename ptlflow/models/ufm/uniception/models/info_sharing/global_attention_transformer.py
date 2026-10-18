"""
UniCeption Global-Attention Transformer for Information Sharing
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

from .base import (
    MultiSetTransformerInput,
    MultiSetTransformerOutput,
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
    UniCeptionInfoSharingBase,
)
from ..libs.croco.pos_embed import RoPE2D
from ..utils.intermediate_feature_return import IntermediateFeatureReturner, feature_take_indices
from ..utils.positional_encoding import PositionGetter
from ..utils.transformer_blocks import Mlp, SelfAttentionBlock, SwiGLUFFNCompressed


class MultiViewGlobalAttentionTransformer(UniCeptionInfoSharingBase):
    "UniCeption Multi-View Global-Attention Transformer for information sharing across image features from different views."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = True,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: Type[nn.Module] = Mlp,
        custom_positional_encoding: Optional[Union[str, Callable]] = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Global-Attention Transformer for information sharing across image features from different views.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: True)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views.
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: True)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Initialize the base class
        super().__init__(name=name, size=size, *args, **kwargs)

        # Initialize the specific attributes of the transformer
        self.input_embed_dim = input_embed_dim
        self.distinguish_ref_and_non_ref_views = distinguish_ref_and_non_ref_views
        self.use_pe_for_non_reference_views = use_pe_for_non_reference_views
        self.max_num_views_for_pe = max_num_views_for_pe
        self.use_rand_idx_pe_for_non_reference_views = use_rand_idx_pe_for_non_reference_views
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # Resolve mlp_layer given as a string, as stored in the configs of some released UFM
        # checkpoints (e.g., UFM-Base-DINOv2G-init).
        if isinstance(mlp_layer, str):
            if mlp_layer == "SwiGLU":
                mlp_layer = SwiGLUFFNCompressed
            else:
                raise ValueError(f"Unknown mlp_layer string: {mlp_layer}")
        self.mlp_layer = mlp_layer
        self.custom_positional_encoding = custom_positional_encoding
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize the projection layer for input embeddings
        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        # Initialize custom position encodings
        if self.custom_positional_encoding is not None and isinstance(self.custom_positional_encoding, str):
            if self.custom_positional_encoding == "rope":
                self.rope = RoPE2D(freq=100.0, F0=1.0)
                self.custom_positional_encoding = self.rope
            else:
                raise ValueError(f"Unknown custom positional encoding: {self.custom_positional_encoding}")

        # Initialize the self-attention blocks which ingest all views at once
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the position getter for patch positions if required
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        # Initialize the positional encoding table for the views if required
        if self.distinguish_ref_and_non_ref_views:
            if self.use_pe_for_non_reference_views:
                # Initialize the positional encoding table for the different views
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(self.max_num_views_for_pe, self.dim, 10000),
                )
            else:
                # Initialize the positional encoding table for the reference view
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(1, self.dim, 10000),
                )

        # Initialize random weights
        self.initialize_weights()

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained multi-view global-attention transformer weights from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.self_attention_blocks):
                self.self_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def initialize_weights(self):
        "Initialize weights of the transformer."
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer linear and layer norm weights."
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> MultiViewTransformerOutput:
        """
        Forward interface for the Multi-View Global-Attention Transformer.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include:
                - additional_input_tokens: Global additional tokens (e.g., scale token)
                  which are appended to the token set from all multi-view features. Shape: (batch, input_embed_dim, num_of_additional_tokens).
                - additional_input_tokens_per_view: Per-view additional tokens (e.g., view-specific tokens like per-view registers)
                  which are appended to each view's token set. List of tensors, each of shape (batch, input_embed_dim, num_of_additional_tokens_per_view).

        Returns:
            MultiViewTransformerOutput: Output of the model post information sharing.
                Returns view features and optionally additional_token_features (global) and additional_token_features_per_view (per-view).
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) <= self.max_num_views_for_pe
        ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            curr_view_features.shape[1] == self.input_embed_dim for curr_view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            curr_view_features.ndim == 4 for curr_view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Process per-view additional tokens if provided
        num_of_additional_tokens_per_view = 0
        if model_input.additional_input_tokens_per_view is not None:
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            assert len(additional_tokens_per_view) == num_of_views, (
                f"Number of additional token tensors ({len(additional_tokens_per_view)}) "
                f"must match number of views ({num_of_views})"
            )
            assert all(
                tokens.ndim == 3 for tokens in additional_tokens_per_view
            ), "Additional tokens per view must have 3 dimensions (N, C, T)"
            assert all(
                tokens.shape[1] == self.input_embed_dim for tokens in additional_tokens_per_view
            ), f"Additional tokens per view must have input dimension {self.input_embed_dim}"
            assert all(
                tokens.shape[0] == batch_size for tokens in additional_tokens_per_view
            ), "Batch size mismatch for additional tokens per view"

            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]

            # Concatenate per-view additional tokens to each view's features
            multi_view_features_with_tokens = []
            for view_idx, (view_features, view_tokens) in enumerate(
                zip(multi_view_features, additional_tokens_per_view)
            ):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()

            # Update tokens per view to include additional tokens
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
        else:
            # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
            multi_view_features = torch.stack(multi_view_features, dim=1)

            # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
            multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
            multi_view_features = multi_view_features.reshape(
                batch_size, num_of_views * height * width, self.input_embed_dim
            ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Raise error if custom positional encoding is used with additional tokens
        if self.custom_positional_encoding is not None:
            if (
                model_input.additional_input_tokens is not None
                or model_input.additional_input_tokens_per_view is not None
            ):
                raise ValueError(
                    "Custom positional encoding is not supported when additional_input_tokens or "
                    "additional_input_tokens_per_view are provided. Please set custom_positional_encoding=None "
                    "or remove additional tokens from the input."
                )

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[2]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            # Apply the self-attention block and update the multi-view features
            multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding global additional tokens)
        view_features_flat = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Extract per-view additional tokens if they were provided
        additional_token_features_per_view = None
        if model_input.additional_input_tokens_per_view is not None:
            # Reshape to (N, V, H*W + T_per_view, C)
            view_features_with_tokens = view_features_flat.reshape(
                batch_size, num_of_views, num_of_tokens_per_view, self.dim
            )

            # Split into spatial features and per-view additional tokens
            spatial_tokens_per_view = height * width
            view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
            per_view_additional = view_features_with_tokens[:, :, spatial_tokens_per_view:, :]  # (N, V, T_per_view, C)

            # Reshape view features to (N, V, H, W, C)
            view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

            # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
            per_view_additional = per_view_additional.split(1, dim=1)
            additional_token_features_per_view = [
                tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                for tokens in per_view_additional
            ]
        else:
            # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
            view_features = view_features_flat.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the output multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features (global) if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

        return MultiViewTransformerOutput(
            features=view_features,
            additional_token_features=additional_token_features,
            additional_token_features_per_view=additional_token_features_per_view,
        )


class MultiViewGlobalAttentionTransformerIFR(MultiViewGlobalAttentionTransformer, IntermediateFeatureReturner):
    "Intermediate Feature Returner for UniCeption Multi-View Global-Attention Transformer"

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = True,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        intermediates_only: bool = False,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Global-Attention Transformer for information sharing across image features from different views.
        Extends the base class to return intermediate features.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: True)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views.
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax. (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. (default: None) Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. (default: True)
            intermediates_only (bool, optional): Whether to return only the intermediate features. (default: False)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Init the base classes
        MultiViewGlobalAttentionTransformer.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            distinguish_ref_and_non_ref_views=distinguish_ref_and_non_ref_views,
            use_pe_for_non_reference_views=use_pe_for_non_reference_views,
            max_num_views_for_pe=max_num_views_for_pe,
            use_rand_idx_pe_for_non_reference_views=use_rand_idx_pe_for_non_reference_views,
            size=size,
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            intermediates_only=intermediates_only,
        )

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]],]:
        """
        Forward interface for the Multi-View Global-Attention Transformer with Intermediate Feature Return.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include:
                - additional_input_tokens: Global additional tokens (e.g., scale token)
                  which are appended to the token set from all multi-view features. Shape: (batch, input_embed_dim, num_of_additional_tokens).
                - additional_input_tokens_per_view: Per-view additional tokens (e.g., view-specific tokens like per-view registers)
                  which are appended to each view's token set. List of tensors, each of shape (batch, input_embed_dim, num_of_additional_tokens_per_view).

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                If intermediates_only is True, returns a list of intermediate outputs containing:
                    - features: List of view features
                    - additional_token_features: Global additional token features (if provided)
                    - additional_token_features_per_view: Per-view additional token features (if provided)
                If intermediates_only is False, returns a tuple of:
                    - Final output (MultiViewTransformerOutput) with features, additional_token_features, and additional_token_features_per_view
                    - List of intermediate outputs (MultiViewTransformerOutput) from specified layers
        """
        # Check that the number of views matches the input and the features are of expected shape
        assert (
            len(model_input.features) <= self.max_num_views_for_pe
        ), f"Expected {self.num_views} views, got {len(model_input.features)}"
        assert all(
            curr_view_features.shape[1] == self.input_embed_dim for curr_view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            curr_view_features.ndim == 4 for curr_view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Process per-view additional tokens if provided
        num_of_additional_tokens_per_view = 0
        if model_input.additional_input_tokens_per_view is not None:
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            assert len(additional_tokens_per_view) == num_of_views, (
                f"Number of additional token tensors ({len(additional_tokens_per_view)}) "
                f"must match number of views ({num_of_views})"
            )
            assert all(
                tokens.ndim == 3 for tokens in additional_tokens_per_view
            ), "Additional tokens per view must have 3 dimensions (N, C, T)"
            assert all(
                tokens.shape[1] == self.input_embed_dim for tokens in additional_tokens_per_view
            ), f"Additional tokens per view must have input dimension {self.input_embed_dim}"
            assert all(
                tokens.shape[0] == batch_size for tokens in additional_tokens_per_view
            ), "Batch size mismatch for additional tokens per view"

            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]

            # Concatenate per-view additional tokens to each view's features
            multi_view_features_with_tokens = []
            for view_idx, (view_features, view_tokens) in enumerate(
                zip(multi_view_features, additional_tokens_per_view)
            ):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()

            # Update tokens per view to include additional tokens
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
        else:
            # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
            multi_view_features = torch.stack(multi_view_features, dim=1)

            # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
            multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
            multi_view_features = multi_view_features.reshape(
                batch_size, num_of_views * height * width, self.input_embed_dim
            ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Raise error if custom positional encoding is used with additional tokens
        if self.custom_positional_encoding is not None:
            if (
                model_input.additional_input_tokens is not None
                or model_input.additional_input_tokens_per_view is not None
            ):
                raise ValueError(
                    "Custom positional encoding is not supported when additional_input_tokens or "
                    "additional_input_tokens_per_view are provided. Please set custom_positional_encoding=None "
                    "or remove additional tokens from the input."
                )

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[2]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            # Apply the self-attention block and update the multi-view features
            multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    self.norm(multi_view_features) if self.norm_intermediate else multi_view_features
                )

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            # Get the current intermediate features
            current_features = intermediate_multi_view_features[idx]

            # Extract view features (excluding global additional tokens)
            view_features_flat = current_features[:, : num_of_views * num_of_tokens_per_view, :]

            # Extract per-view additional tokens if they were provided
            additional_token_features_per_view = None
            if model_input.additional_input_tokens_per_view is not None:
                # Reshape to (N, V, H*W + T_per_view, C)
                view_features_with_tokens = view_features_flat.reshape(
                    batch_size, num_of_views, num_of_tokens_per_view, self.dim
                )

                # Split into spatial features and per-view additional tokens
                spatial_tokens_per_view = height * width
                view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
                per_view_additional = view_features_with_tokens[
                    :, :, spatial_tokens_per_view:, :
                ]  # (N, V, T_per_view, C)

                # Reshape view features to (N, V, H, W, C)
                view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

                # Split view features into separate views
                view_features = view_features.split(1, dim=1)
                view_features = [
                    intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
                ]

                # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
                per_view_additional = per_view_additional.split(1, dim=1)
                additional_token_features_per_view = [
                    tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                    for tokens in per_view_additional
                ]
            else:
                # Reshape the intermediate multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
                view_features = view_features_flat.reshape(
                    batch_size, num_of_views, height, width, self.dim
                )  # (N, V, H, W, C)
                view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

                # Split the intermediate multi-view features into separate views
                view_features = view_features.split(1, dim=1)
                view_features = [
                    intermediate_view_features.squeeze(dim=1) for intermediate_view_features in view_features
                ]

            # Extract and return additional token features (global) if provided
            additional_token_features = None
            if model_input.additional_input_tokens is not None:
                additional_token_features = current_features[:, num_of_views * num_of_tokens_per_view :, :]
                additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=view_features,
                additional_token_features=additional_token_features,
                additional_token_features_per_view=additional_token_features_per_view,
            )

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_multi_view_features

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding global additional tokens)
        view_features_flat = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Extract per-view additional tokens if they were provided
        additional_token_features_per_view = None
        if model_input.additional_input_tokens_per_view is not None:
            # Reshape to (N, V, H*W + T_per_view, C)
            view_features_with_tokens = view_features_flat.reshape(
                batch_size, num_of_views, num_of_tokens_per_view, self.dim
            )

            # Split into spatial features and per-view additional tokens
            spatial_tokens_per_view = height * width
            view_features = view_features_with_tokens[:, :, :spatial_tokens_per_view, :]  # (N, V, H*W, C)
            per_view_additional = view_features_with_tokens[:, :, spatial_tokens_per_view:, :]  # (N, V, T_per_view, C)

            # Reshape view features to (N, V, H, W, C)
            view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

            # Split per-view additional tokens and reshape to (N, C, T_per_view) for each view
            per_view_additional = per_view_additional.split(1, dim=1)
            additional_token_features_per_view = [
                tokens.squeeze(dim=1).permute(0, 2, 1).contiguous()  # (N, T_per_view, C) -> (N, C, T_per_view)
                for tokens in per_view_additional
            ]
        else:
            # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, H, W, C)
            view_features = view_features_flat.reshape(batch_size, num_of_views, height, width, self.dim)
            view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the output multi-view features into separate views
            view_features = view_features.split(1, dim=1)
            view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features (global) if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)

        output_multi_view_features = MultiViewTransformerOutput(
            features=view_features,
            additional_token_features=additional_token_features,
            additional_token_features_per_view=additional_token_features_per_view,
        )

        return output_multi_view_features, intermediate_multi_view_features


class GlobalAttentionTransformer(UniCeptionInfoSharingBase):
    "UniCeption Global-Attention Transformer for information sharing across different set of features."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        max_num_sets: int,
        use_rand_idx_pe_for_non_reference_sets: bool,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: Type[nn.Module] = Mlp,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Global-Attention Transformer for information sharing across features from different sets.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            max_num_sets (int): Maximum number of sets for positional encoding.
            use_rand_idx_pe_for_non_reference_sets (bool): Whether to use random index positional encoding for non-reference sets.
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: True)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Initialize the base class
        super().__init__(name=name, size=size, *args, **kwargs)

        # Initialize the specific attributes of the transformer
        self.input_embed_dim = input_embed_dim
        self.max_num_sets = max_num_sets
        self.use_rand_idx_pe_for_non_reference_sets = use_rand_idx_pe_for_non_reference_sets
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize the projection layer for input embeddings
        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        # Initialize the self-attention blocks which ingest all sets at once
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the positional encoding table for the different sets
        self.register_buffer(
            "set_pos_table",
            self._get_sinusoid_encoding_table(self.max_num_sets, self.dim, 10000),
        )

        # Initialize random weights
        self.initialize_weights()

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(f"Loading pretrained global-attention transformer weights from {self.pretrained_checkpoint_path} ...")
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.self_attention_blocks):
                self.self_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def initialize_weights(self):
        "Initialize weights of the transformer."
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer linear and layer norm weights."
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        model_input: MultiSetTransformerInput,
    ) -> MultiSetTransformerOutput:
        """
        Forward interface for the Multi-Set Global-Attention Transformer.

        Args:
            model_input (MultiSetTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, num_tokens),
                where each entry corresponds to a different set of tokens and
                the number of tokens can be different for each set.
                Optionally, the input can also include additional_input_tokens (e.g., class token, registers, pose tokens, scale token)
                which are appended to the token set from the multi-view features. The tokens are of size (batch, input_embed_dim, num_of_additional_tokens).

        Returns:
            MultiSetTransformerOutput: Output of the model post information sharing.
        """
        # Check that the number of sets matches the input and the features are of expected shape
        assert (
            len(model_input.features) <= self.max_num_sets
        ), f"Expected less than {self.max_num_sets} sets, got {len(model_input.features)}"
        assert all(
            set_features.shape[1] == self.input_embed_dim for set_features in model_input.features
        ), f"All sets must have input dimension {self.input_embed_dim}"
        assert all(
            set_features.ndim == 3 for set_features in model_input.features
        ), "All sets must have 3 dimensions (N, C, T)"

        # Initialize the multi-set features from the model input and number of sets for current input
        multi_set_features = model_input.features
        num_of_sets = len(multi_set_features)
        batch_size, _, _ = multi_set_features[0].shape
        num_of_tokens_per_set = [set_features.shape[2] for set_features in multi_set_features]

        # Permute the multi-set features from (N, C, T) to (N, T, C)
        multi_set_features = [set_features.permute(0, 2, 1).contiguous() for set_features in multi_set_features]

        # Stack the multi-set features along the number of tokens dimension
        multi_set_features = torch.cat(multi_set_features, dim=1)

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:
            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-set features
            multi_set_features = torch.cat([multi_set_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_set_features = self.proj_embed(multi_set_features)

        # Create dummy patch positions for each set
        multi_set_positions = [None] * num_of_sets

        # Add positional encoding for reference set (idx 0)
        ref_set_pe = self.set_pos_table[0].clone().detach()
        ref_set_pe = ref_set_pe.reshape((1, 1, self.dim))
        ref_set_pe = ref_set_pe.repeat(batch_size, num_of_tokens_per_set[0], 1)
        ref_set_features = multi_set_features[:, : num_of_tokens_per_set[0], :]
        ref_set_features = ref_set_features + ref_set_pe

        # Add positional encoding for non-reference sets (sequential indices starting from idx 1 or random indices which are uniformly sampled)
        if self.use_rand_idx_pe_for_non_reference_sets:
            non_ref_set_pe_indices = torch.randint(low=1, high=self.max_num_sets, size=(num_of_sets - 1,))
        else:
            non_ref_set_pe_indices = torch.arange(1, num_of_sets)
        non_ref_set_pe_list = []
        for non_ref_set_idx in range(1, num_of_sets):
            non_ref_set_pe_for_idx = self.set_pos_table[non_ref_set_pe_indices[non_ref_set_idx - 1]].clone().detach()
            non_ref_set_pe_for_idx = non_ref_set_pe_for_idx.reshape((1, 1, self.dim))
            non_ref_set_pe_for_idx = non_ref_set_pe_for_idx.repeat(
                batch_size, num_of_tokens_per_set[non_ref_set_idx], 1
            )
            non_ref_set_pe_list.append(non_ref_set_pe_for_idx)
        non_ref_set_pe = torch.cat(non_ref_set_pe_list, dim=1)
        non_ref_set_features = multi_set_features[:, num_of_tokens_per_set[0] : sum(num_of_tokens_per_set), :]
        non_ref_set_features = non_ref_set_features + non_ref_set_pe

        # Concatenate the reference and non-reference set features
        # Handle additional tokens (no set-based positional encoding for them)
        if model_input.additional_input_tokens is not None:
            additional_features = multi_set_features[:, sum(num_of_tokens_per_set) :, :]
            multi_set_features = torch.cat([ref_set_features, non_ref_set_features, additional_features], dim=1)
        else:
            multi_set_features = torch.cat([ref_set_features, non_ref_set_features], dim=1)

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:
            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[2]
            multi_set_positions = multi_set_positions + additional_tokens_positions

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            # Apply the self-attention block and update the multi-set features
            multi_set_features = self.self_attention_blocks[depth_idx](multi_set_features, multi_set_positions)

        # Normalize the output features
        output_multi_set_features = self.norm(multi_set_features)

        # Extract additional token features if provided
        additional_token_features = None
        if model_input.additional_input_tokens is not None:
            additional_token_features = output_multi_set_features[:, sum(num_of_tokens_per_set) :, :]
            additional_token_features = additional_token_features.permute(
                0, 2, 1
            ).contiguous()  # (N, T, C) -> (N, C, T)
            # Only keep the set features for reshaping
            output_multi_set_features = output_multi_set_features[:, : sum(num_of_tokens_per_set), :]

        # Reshape the output multi-set features from (N, T, C) to (N, C, T)
        output_multi_set_features = output_multi_set_features.permute(0, 2, 1).contiguous()

        # Split the output multi-set features into separate sets using the list of number of tokens per set
        output_multi_set_features = torch.split(output_multi_set_features, num_of_tokens_per_set, dim=2)

        # Return the output multi-set features with additional token features if provided
        return MultiSetTransformerOutput(
            features=output_multi_set_features,
            additional_token_features=additional_token_features,
        )


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x


if __name__ == "__main__":
    # Init multi-view global-attention transformer with no custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(f"Testing MultiViewGlobalAttentionTransformer with {num_views} views ...")
        # Sequential idx based positional encoding
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT",
            input_embed_dim=1024,
            max_num_views_for_pe=1000,
            use_rand_idx_pe_for_non_reference_views=False,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
        # Random idx based positional encoding
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT",
            input_embed_dim=1024,
            max_num_views_for_pe=1000,
            use_rand_idx_pe_for_non_reference_views=True,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    # Init multi-view global-attention transformer with custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(f"Testing MultiViewGlobalAttentionTransformer with {num_views} views and custom positional encoding ...")
        model = MultiViewGlobalAttentionTransformer(
            name="MV-GAT",
            input_embed_dim=1024,
            max_num_views_for_pe=1000,
            use_rand_idx_pe_for_non_reference_views=True,
            custom_positional_encoding=dummy_positional_encoding,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    print("All multi-view global-attention transformers initialized and tested successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests ...")

    # Run the intermediate feature returner with last-n index
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=6,  # Last 6 layers
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 6
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Run the intermediate feature returner with specific indices
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[0, 2, 4, 6],  # Specific indices
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 4
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Test the normalizing of intermediate features
    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[-1],  # Last layer
        norm_intermediate=False,  # Disable normalization
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert not torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be different."

    model_intermediate_feature_returner = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[-1],  # Last layer
        norm_intermediate=True,
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be same."

    print("All Intermediate Feature Returner Tests passed!")

    # Init multi-set global-attention transformer and run a forward pass with different number of sets and set token sizes
    import random

    model = GlobalAttentionTransformer(
        name="GAT",
        input_embed_dim=1024,
        max_num_sets=3,
        use_rand_idx_pe_for_non_reference_sets=False,
    )
    for num_sets in [2, 3]:
        print(f"Testing GlobalAttentionTransformer with {num_sets} sets ...")
        model_input = [torch.rand(1, 1024, random.randint(256, 513)) for _ in range(num_sets)]
        model_input = MultiSetTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_sets
        for feat, rand_input in zip(model_output.features, model_input.features):
            assert feat.shape[2] == rand_input.shape[2]
            assert feat.shape[1] == model.dim
            assert feat.shape[0] == rand_input.shape[0]
    # Random idx based positional encoding
    model = GlobalAttentionTransformer(
        name="GAT",
        input_embed_dim=1024,
        max_num_sets=1000,
        use_rand_idx_pe_for_non_reference_sets=True,
    )
    for num_sets in [2, 3, 4]:
        print(f"Testing GlobalAttentionTransformer with {num_sets} sets ...")
        model_input = [torch.rand(1, 1024, random.randint(256, 513)) for _ in range(num_sets)]
        model_input = MultiSetTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_sets
        for feat, rand_input in zip(model_output.features, model_input.features):
            assert feat.shape[2] == rand_input.shape[2]
            assert feat.shape[1] == model.dim
            assert feat.shape[0] == rand_input.shape[0]

    print("All Global Attention Transformer Tests passed!")

    # Test additional input tokens for MultiViewGlobalAttentionTransformer
    print("Testing MultiViewGlobalAttentionTransformer with additional input tokens...")
    model = MultiViewGlobalAttentionTransformer(
        name="MV-GAT",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=False,
    )
    num_views = 2
    num_additional_tokens = 5
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    model_input = MultiViewTransformerInput(features=model_input, additional_input_tokens=additional_tokens)
    model_output = model(model_input)
    assert len(model_output.features) == num_views
    assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
    assert model_output.additional_token_features is not None
    assert model_output.additional_token_features.shape == (
        1,
        model.dim,
        num_additional_tokens,
    )

    # Test additional input tokens for MultiViewGlobalAttentionTransformerIFR
    print("Testing MultiViewGlobalAttentionTransformerIFR with additional input tokens...")
    model_ifr = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[0, 2, 4],
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    model_input = MultiViewTransformerInput(features=model_input, additional_input_tokens=additional_tokens)
    output = model_ifr(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert output[0].additional_token_features is not None
    assert output[0].additional_token_features.shape == (
        1,
        model_ifr.dim,
        num_additional_tokens,
    )
    assert len(output[1]) == 3
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert all(intermediate.additional_token_features is not None for intermediate in output[1])
    assert all(
        intermediate.additional_token_features.shape == (1, model_ifr.dim, num_additional_tokens)
        for intermediate in output[1]
    )

    # Test additional input tokens for GlobalAttentionTransformer
    print("Testing GlobalAttentionTransformer with additional input tokens...")
    model = GlobalAttentionTransformer(
        name="GAT",
        input_embed_dim=1024,
        max_num_sets=1000,
        use_rand_idx_pe_for_non_reference_sets=False,
    )
    num_sets = 3
    num_additional_tokens = 8
    model_input = [torch.rand(1, 1024, random.randint(256, 513)) for _ in range(num_sets)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    model_input = MultiSetTransformerInput(features=model_input, additional_input_tokens=additional_tokens)
    model_output = model(model_input)
    assert len(model_output.features) == num_sets
    for feat, rand_input in zip(model_output.features, model_input.features):
        assert feat.shape[2] == rand_input.shape[2]
        assert feat.shape[1] == model.dim
        assert feat.shape[0] == rand_input.shape[0]
    assert model_output.additional_token_features is not None
    assert model_output.additional_token_features.shape == (
        1,
        model.dim,
        num_additional_tokens,
    )

    print("All tests using additional input tokens passed!")

    # Test both additional_input_tokens and additional_input_tokens_per_view together for MultiViewGlobalAttentionTransformer
    print(
        "Testing MultiViewGlobalAttentionTransformer with both additional_input_tokens and additional_input_tokens_per_view..."
    )
    model = MultiViewGlobalAttentionTransformer(
        name="MV-GAT",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=False,
    )
    num_views = 3
    num_additional_tokens = 5
    num_additional_tokens_per_view = 8
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    additional_tokens_per_view = [torch.rand(1, 1024, num_additional_tokens_per_view) for _ in range(num_views)]
    model_input = MultiViewTransformerInput(
        features=model_input,
        additional_input_tokens=additional_tokens,
        additional_input_tokens_per_view=additional_tokens_per_view,
    )
    model_output = model(model_input)
    assert len(model_output.features) == num_views
    assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
    assert model_output.additional_token_features is not None
    assert model_output.additional_token_features.shape == (
        1,
        model.dim,
        num_additional_tokens,
    )
    assert model_output.additional_token_features_per_view is not None
    assert len(model_output.additional_token_features_per_view) == num_views
    assert all(
        tokens.shape == (1, model.dim, num_additional_tokens_per_view)
        for tokens in model_output.additional_token_features_per_view
    )

    # Test both additional_input_tokens and additional_input_tokens_per_view together for MultiViewGlobalAttentionTransformerIFR
    print(
        "Testing MultiViewGlobalAttentionTransformerIFR with both additional_input_tokens and additional_input_tokens_per_view..."
    )
    model_ifr = MultiViewGlobalAttentionTransformerIFR(
        name="MV-GAT-IFR",
        input_embed_dim=1024,
        max_num_views_for_pe=1000,
        use_rand_idx_pe_for_non_reference_views=True,
        indices=[0, 5, 11],
    )
    num_views = 2
    num_additional_tokens = 3
    num_additional_tokens_per_view = 6
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    additional_tokens_per_view = [torch.rand(1, 1024, num_additional_tokens_per_view) for _ in range(num_views)]
    model_input = MultiViewTransformerInput(
        features=model_input,
        additional_input_tokens=additional_tokens,
        additional_input_tokens_per_view=additional_tokens_per_view,
    )
    output = model_ifr(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[0].features) == num_views
    assert all(f.shape == (1, model_ifr.dim, 14, 14) for f in output[0].features)
    assert output[0].additional_token_features is not None
    assert output[0].additional_token_features.shape == (
        1,
        model_ifr.dim,
        num_additional_tokens,
    )
    assert output[0].additional_token_features_per_view is not None
    assert len(output[0].additional_token_features_per_view) == num_views
    assert all(
        tokens.shape == (1, model_ifr.dim, num_additional_tokens_per_view)
        for tokens in output[0].additional_token_features_per_view
    )
    assert len(output[1]) == 3
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert all(intermediate.additional_token_features is not None for intermediate in output[1])
    assert all(
        intermediate.additional_token_features.shape == (1, model_ifr.dim, num_additional_tokens)
        for intermediate in output[1]
    )
    assert all(intermediate.additional_token_features_per_view is not None for intermediate in output[1])
    assert all(len(intermediate.additional_token_features_per_view) == num_views for intermediate in output[1])
    assert all(
        all(
            tokens.shape == (1, model_ifr.dim, num_additional_tokens_per_view)
            for tokens in intermediate.additional_token_features_per_view
        )
        for intermediate in output[1]
    )

    print("All tests using both additional_input_tokens and additional_input_tokens_per_view passed!")
