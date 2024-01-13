import torch
import torch.nn as nn
import copy


class Attention1D(nn.Module):
    """Cross-Attention on x or y direction,
    without multi-head and dropout support for faster speed
    """

    def __init__(
        self,
        in_channels,
        y_attention=False,
        double_cross_attn=False,  # cross attn feature1 before computing cross attn feature2
        **kwargs,
    ):
        super(Attention1D, self).__init__()

        self.y_attention = y_attention
        self.double_cross_attn = double_cross_attn

        # self attn feature1 before cross attn
        if double_cross_attn:
            self.self_attn = copy.deepcopy(
                Attention1D(
                    in_channels=in_channels,
                    y_attention=not y_attention,
                )
            )

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Initialize: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L138
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # original Transformer initialization

    def forward(self, feature1, feature2, position=None, value=None):
        b, c, h, w = feature1.size()

        # self attn before cross attn
        if self.double_cross_attn:
            feature1 = self.self_attn(feature1, feature1, position)[
                0
            ]  # self attn feature1

        query = feature1 + position if position is not None else feature1
        query = self.query_conv(query)  # [B, C, H, W]

        key = feature2 + position if position is not None else feature2

        key = self.key_conv(key)  # [B, C, H, W]
        value = feature2 if value is None else value  # [B, C, H, W]
        scale_factor = c**0.5

        if self.y_attention:
            query = query.permute(0, 3, 2, 1)  # [B, W, H, C]
            key = key.permute(0, 3, 1, 2)  # [B, W, C, H]
            value = value.permute(0, 3, 2, 1)  # [B, W, H, C]
        else:  # x attention
            query = query.permute(0, 2, 3, 1)  # [B, H, W, C]
            key = key.permute(0, 2, 1, 3)  # [B, H, C, W]
            value = value.permute(0, 2, 3, 1)  # [B, H, W, C]

        scores = torch.matmul(query, key) / scale_factor  # [B, W, H, H] or [B, H, W, W]

        attention = torch.softmax(scores, dim=-1)  # [B, W, H, H] or [B, H, W, W]

        out = torch.matmul(attention, value)  # [B, W, H, C] or [B, H, W, C]

        if self.y_attention:
            out = out.permute(0, 3, 2, 1).contiguous()  # [B, C, H, W]
        else:
            out = out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return out, attention
