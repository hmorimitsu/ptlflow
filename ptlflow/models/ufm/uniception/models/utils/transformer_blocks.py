"""
Utils for Common Transformer Blocks used in UniCeption
References:
HuggingFace PyTorch Image Models (Timm)
CroCoV2
DINOv2
"""

import collections.abc
import math
from itertools import repeat
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from .config import use_fused_attn

torch.backends.cuda.matmul.allow_tf32 = True


def _ntuple(n):
    "Helper function to create n-tuple."

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class SwiGLUFFNCompressed(SwiGLUFFN):
    """SwiGLUFFN with the hidden dimension compressed to two thirds, without rounding.

    Added for compatibility with the UFM checkpoints whose configs specify mlp_layer="SwiGLU"
    (e.g., UFM-Base-DINOv2G-init), which were trained with a newer UniCeption version.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features * 2 / 3)
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class Attention(nn.Module):
    "Self-Attention Layer"

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        latent_attn_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
    ):
        """
        Initialize the Attention layer.

        Args:
            dim (int): Dimension of input features
            latent_attn_dim (int): Dimension of latent attention features (default: None)
            num_heads (int): Number of attention heads (default: 8)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            proj_drop (float): Dropout rate for output (default: 0.)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
        """
        super().__init__()

        if latent_attn_dim is not None:
            assert latent_attn_dim % num_heads == 0, "latent_attn_dim should be divisible by num_heads"
            self.latent_attn_dim = latent_attn_dim
            self.latent_attn = True
        else:
            self.latent_attn = False
            assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads if not self.latent_attn else latent_attn_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = (
            nn.Linear(dim, dim * 3, bias=qkv_bias)
            if not self.latent_attn
            else nn.Linear(dim, latent_attn_dim * 3, bias=qkv_bias)
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) if not self.latent_attn else nn.Linear(latent_attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.custom_positional_encoding = custom_positional_encoding
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor

    def forward(self, x: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input features
            xpos (torch.Tensor): Positions of tokens (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.custom_positional_encoding is not None:
            assert (
                xpos is not None
            ), "Positions of tokens (xpos) are a required input when using custom positional encoding"
            q = self.custom_positional_encoding(q, xpos)
            k = self.custom_positional_encoding(k, xpos)

        if self.use_scalable_softmax:
            # Scales the exponential base using the number of tokens (https://arxiv.org/pdf/2501.19399)
            q = q * torch.log(torch.tensor(N, device=q.device))

        if self.use_entropy_scaling:
            # Scales the exponential base using the number of tokens (https://arxiv.org/pdf/2502.07785#page=7.35)
            scaling_factor = torch.sqrt(
                (self.entropy_scaling_growth_factor * torch.log(torch.tensor(N, device=q.device)))
                / torch.log(torch.tensor(self.base_token_count_for_entropy_scaling, device=q.device))
            )
            q = q * scaling_factor

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    "Cross-Attention Layer"

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
    ):
        """
        Initialize the Cross-Attention layer.

        Args:
            dim (int): Dimension of input features
            num_heads (int): Number of attention heads (default: 8)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            proj_drop (float): Dropout rate for output (default: 0.)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.custom_positional_encoding = custom_positional_encoding
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        qpos: torch.Tensor = None,
        kpos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Cross-Attention layer.

        Args:
            query (torch.Tensor): Query features
            key (torch.Tensor): Key features
            value (torch.Tensor): Value features
            qpos (torch.Tensor): Positions of queries (required when using custom positional encoding)
            kpos (torch.Tensor): Positions of keys (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.custom_positional_encoding is not None:
            assert (
                qpos is not None
            ), "Positions of queries (qpos) are a required input when using custom positional encoding"
            assert (
                kpos is not None
            ), "Positions of keys (kpos) are a required input when using custom positional encoding"
            q = self.custom_positional_encoding(q, qpos)
            k = self.custom_positional_encoding(k, kpos)

        if self.use_scalable_softmax:
            # Scales the exponential base using the number of tokens (https://arxiv.org/pdf/2501.19399)
            q = q * torch.log(torch.tensor(Nq, device=q.device))

        if self.use_entropy_scaling:
            # Scales the exponential base using the number of tokens (https://arxiv.org/pdf/2502.07785#page=7.35)
            scaling_factor = torch.sqrt(
                (self.entropy_scaling_growth_factor * torch.log(torch.tensor(Nq, device=q.device)))
                / torch.log(torch.tensor(self.base_token_count_for_entropy_scaling, device=q.device))
            )
            q = q * scaling_factor

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    "Layer Scale Layer"

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ):
        """
        Initialize the Layer Scale layer

        Args:
            dim (int): Dimension of input features
            init_values (float): Initial value for LayerScale gamma (default: 1e-5)
            inplace (bool): Whether to perform inplace operations (default: False)
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Forward pass of the Layer Scale layer"
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SelfAttentionBlock(nn.Module):
    "Self-Attention Block"

    def __init__(
        self,
        dim: int,
        num_heads: int,
        latent_attn_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
    ):
        """
        Initialize the Self-Attention Block.

        Args:
            dim (int): Dimension of input features
            num_heads (int): Number of attention heads
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
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            latent_attn_dim=latent_attn_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.custom_positional_encoding = custom_positional_encoding

    def forward(self, x: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Self-Attention Block.

        Args:
            x (torch.Tensor): Input features
            xpos (torch.Tensor): Positions of tokens (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        if self.custom_positional_encoding is not None:
            assert (
                xpos is not None
            ), "Positions of tokens (xpos) are a required input when using custom positional encoding"
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), xpos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttentionBlock(nn.Module):
    "Cross-Attention Block"

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        norm_cross_tokens: bool = True,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
    ):
        """
        Initialize the Cross-Attention Block.

        Args:
            dim (int): Dimension of input features
            num_heads (int): Number of attention heads
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
            norm_cross_tokens (bool): Whether to normalize cross tokens (default: True)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_y = norm_layer(dim) if norm_cross_tokens else nn.Identity()
        self.custom_positional_encoding = custom_positional_encoding
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xpos: torch.Tensor = None,
        ypos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Cross-Attention Block.

        Args:
            x (torch.Tensor): Input features
            y (torch.Tensor): Cross features
            xpos (torch.Tensor): Positions of tokens (required when using custom positional encoding)
            ypos (torch.Tensor): Positions of cross tokens (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        if self.custom_positional_encoding is not None:
            assert (
                xpos is not None
            ), "Positions of tokens (xpos) are a required input when using custom positional encoding"
            assert (
                ypos is not None
            ), "Positions of cross tokens (ypos) are a required input when using custom positional encoding"
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), xpos)))
        y_ = self.norm_y(y)
        x = x + self.drop_path2(self.ls2(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos)))
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x


# copied from DiffTrsformer
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)  # copied from DiffTrsformer


class DiffAttention(nn.Module):
    "Differential Self-Attention Layer"

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        custom_positional_encoding: Callable = None,
    ):
        """
        Initialize the DiffAttention layer.

        Args:
            dim (int): Dimension of input features
            depth (int): Depth of the current layer, used in lambda initialization (default: 0)
            num_heads (int): Number of attention heads (default: 8)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            proj_drop (float): Dropout rate for output (default: 0.)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.custom_positional_encoding = custom_positional_encoding

        # DiffTransformer specific
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, xpos: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input features
            xpos (torch.Tensor): Positions of tokens (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim * 2)
        q, k, v = torch.chunk(qkv, 3, dim=2)  # B, N, Nh, Dh

        q = q.view(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.custom_positional_encoding is not None:
            assert (
                xpos is not None
            ), "Positions of tokens (xpos) are a required input when using custom positional encoding"
            q = self.custom_positional_encoding(q, xpos)
            k = self.custom_positional_encoding(k, xpos)

        q1, q2 = q.chunk(2, dim=1)  # split heads dimension into two
        k1, k2 = k.chunk(2, dim=1)  # split heads dimension into two

        if self.fused_attn:
            attn1 = F.scaled_dot_product_attention(
                q1, k1, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
            attn2 = F.scaled_dot_product_attention(
                q2, k2, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
        else:
            q1 = q1 * self.scale
            attn = q1 @ k1.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn1 = attn @ v

            q2 = q2 * self.scale
            attn = q2 @ k2.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn2 = attn @ v

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(B, N, self.num_heads * 2 * self.head_dim)

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class DiffCrossAttention(nn.Module):
    "Differential Cross-Attention Layer, following https://arxiv.org/abs/2410.05258"

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        custom_positional_encoding: Callable = None,
    ):
        """
        Initialize the Cross-Attention layer.

        Args:
            dim (int): Dimension of input features
            depth (int): Depth of the current layer, used in lambda initialization (default: 0)
            num_heads (int): Number of attention heads (default: 8)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            proj_drop (float): Dropout rate for output (default: 0.)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # DiffTransformer specific
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.custom_positional_encoding = custom_positional_encoding

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)  # copied from DiffTrsformer

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        qpos: torch.Tensor = None,
        kpos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Cross-Attention layer.

        Args:
            query (torch.Tensor): Query features
            key (torch.Tensor): Key features
            value (torch.Tensor): Value features
            qpos (torch.Tensor): Positions of queries (required when using custom positional encoding)
            kpos (torch.Tensor): Positions of keys (required when using custom positional encoding)

        Returns:
            torch.Tensor: Output features of same shape as input
        """
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.custom_positional_encoding is not None:
            assert (
                qpos is not None
            ), "Positions of queries (qpos) are a required input when using custom positional encoding"
            assert (
                kpos is not None
            ), "Positions of keys (kpos) are a required input when using custom positional encoding"
            q = self.custom_positional_encoding(q, qpos)
            k = self.custom_positional_encoding(k, kpos)

        q1, q2 = q.chunk(2, dim=1)  # split heads dimension into two
        k1, k2 = k.chunk(2, dim=1)  # split heads dimension into two

        if self.fused_attn:
            attn1 = F.scaled_dot_product_attention(
                q1, k1, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
            attn2 = F.scaled_dot_product_attention(
                q2, k2, v, dropout_p=(self.attn_drop.p if self.training else 0.0), scale=self.scale
            )
        else:
            q1 = q1 * self.scale
            attn = q1 @ k1.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn1 = attn @ v

            q2 = q2 * self.scale
            attn = q2 @ k2.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn2 = attn @ v

        attn1 = attn1.transpose(1, 2)  # B, Nq, Nh, Dh
        attn2 = attn2.transpose(1, 2)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(B, Nq, self.num_heads * 2 * self.head_dim)

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class DiffSelfAttentionBlock(SelfAttentionBlock):
    "Differential Self-Attention Block"

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
    ):
        super().__init__(
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
        )

        self.attn = DiffAttention(
            dim,
            depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            custom_positional_encoding=custom_positional_encoding,
        )


class DiffCrossAttentionBlock(CrossAttentionBlock):
    "Differential Cross-Attention Block"

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        norm_cross_tokens: bool = True,
    ):
        super().__init__(
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
            norm_cross_tokens=norm_cross_tokens,
        )

        self.cross_attn = DiffCrossAttention(
            dim,
            depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            custom_positional_encoding=custom_positional_encoding,
        )


if __name__ == "__main__":
    # Init Attention & CrossAttention classes
    self_attn = Attention(dim=768, custom_positional_encoding=dummy_positional_encoding)
    cross_attn = CrossAttention(dim=768, custom_positional_encoding=dummy_positional_encoding)

    # Perform dummy inference with the Attention classes
    dummy_input = torch.randn((1, 256, 768))
    dummy_x = torch.arange(16)
    dummy_y = torch.arange(16)
    dummy_xpos = torch.cartesian_prod(dummy_y, dummy_x).view(1, 256, 2).expand(1, -1, 2).clone()
    self_attn_output = self_attn(dummy_input, dummy_xpos)
    cross_attn_output = cross_attn(dummy_input, dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of Attention & CrossAttention classes is successful!")

    # Init SelfAttentionBlock & CrossAttentionBlock
    self_attn_block = SelfAttentionBlock(dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding)
    cross_attn_block = CrossAttentionBlock(dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding)

    # Perform dummy inference with the Attention blocks
    self_attn_block_output = self_attn_block(dummy_input, dummy_xpos)
    cross_attn_block_output = cross_attn_block(dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of SelfAttentionBlock & CrossAttentionBlock is successful!")

    # Init DiffAttention & DiffCrossAttention classes
    diff_self_attn = DiffAttention(dim=768, depth=0, custom_positional_encoding=dummy_positional_encoding)
    diff_cross_attn = DiffCrossAttention(dim=768, depth=0, custom_positional_encoding=dummy_positional_encoding)

    # Perform dummy inference with the DiffAttention classes
    diff_self_attn_output = diff_self_attn(dummy_input, dummy_xpos)
    diff_cross_attn_output = diff_cross_attn(dummy_input, dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of DiffAttention & DiffCrossAttention classes is successful!")

    # Init DiffSelfAttentionBlock & DiffCrossAttentionBlock
    diff_self_attn_block = DiffSelfAttentionBlock(
        dim=768, depth=0, num_heads=8, custom_positional_encoding=dummy_positional_encoding
    )
    diff_cross_attn_block = DiffCrossAttentionBlock(
        dim=768, depth=0, num_heads=8, custom_positional_encoding=dummy_positional_encoding
    )

    # Perform dummy inference with the DiffAttention blocks
    diff_self_attn_block_output = diff_self_attn_block(dummy_input, dummy_xpos)
    diff_cross_attn_block_output = diff_cross_attn_block(dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of DiffSelfAttentionBlock & DiffCrossAttentionBlock is successful!")

    # Init SelfAttentionBlock & CrossAttentionBlock with scalable softmax
    self_attn_block = SelfAttentionBlock(
        dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding, use_scalable_softmax=True
    )
    cross_attn_block = CrossAttentionBlock(
        dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding, use_scalable_softmax=True
    )

    # Perform dummy inference with the Attention blocks
    self_attn_block_output = self_attn_block(dummy_input, dummy_xpos)
    cross_attn_block_output = cross_attn_block(dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of SelfAttentionBlock & CrossAttentionBlock with scalable softmax is successful!")

    # Init SelfAttentionBlock & CrossAttentionBlock with entropy scaling
    self_attn_block = SelfAttentionBlock(
        dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding, use_entropy_scaling=True
    )
    cross_attn_block = CrossAttentionBlock(
        dim=768, num_heads=16, custom_positional_encoding=dummy_positional_encoding, use_entropy_scaling=True
    )

    # Perform dummy inference with the Attention blocks
    self_attn_block_output = self_attn_block(dummy_input, dummy_xpos)
    cross_attn_block_output = cross_attn_block(dummy_input, dummy_input, dummy_xpos, dummy_xpos)
    print("Init of SelfAttentionBlock & CrossAttentionBlock with entropy scaling is successful!")
