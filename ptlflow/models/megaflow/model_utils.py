import torch
import torch.nn.functional as F
import torch.nn as nn


class SelfAttnPropagation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(
        self,
        in_channels,
        **kwargs,
    ):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feature0,
        flow,
        local_window_attn=False,
        local_window_radius=1,
        **kwargs,
    ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(
                feature0, flow, local_window_radius=local_window_radius
            )

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c**0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(
        self,
        feature0,
        flow,
        local_window_radius=1,
    ):
        assert flow.size(1) == 2 or flow.size(1) == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(
            feature0.view(b, c, -1).permute(0, 2, 1)
        ).reshape(
            b * h * w, 1, c
        )  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = (
            self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(b, c, h, w)
        )

        feature0_window = F.unfold(
            feature0_proj, kernel_size=kernel_size, padding=local_window_radius
        )  # [B, C*(2R+1)^2), H*W]

        feature0_window = (
            feature0_window.view(b, c, kernel_size**2, h, w)
            .permute(0, 3, 4, 1, 2)
            .reshape(b * h * w, c, kernel_size**2)
        )  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(
            flow, kernel_size=kernel_size, padding=local_window_radius
        )  # [B, 2*(2R+1)^2), H*W]

        flow_window = (
            flow_window.view(b, value_channel, kernel_size**2, h, w)
            .permute(0, 3, 4, 2, 1)
            .reshape(b * h * w, kernel_size**2, value_channel)
        )  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (
            c**0.5
        )  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = (
            torch.matmul(prob, flow_window)
            .view(b, h, w, value_channel)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [B, 2, H, W]

        return out


def upsample_flow_with_mask(flow, up_mask, upsample_factor, is_depth=False):
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(
        b, 1, 9, upsample_factor, upsample_factor, h, w
    )  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = 1 if is_depth else upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(
        b, flow_channel, upsample_factor * h, upsample_factor * w
    )  # [B, 2, K*H, K*W]

    return up_flow


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(
        embed_dim // 2, pos_flat[:, 0], omega_0=omega_0
    )  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(
        embed_dim // 2, pos_flat[:, 1], omega_0=omega_0
    )  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(
        embed_dim // 2,
        dtype=torch.float32 if device.type == "mps" else torch.double,
        device=device,
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid
