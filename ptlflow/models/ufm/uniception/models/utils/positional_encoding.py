"""
Helper function for positional encoding in UniCeption
"""

import torch


class PositionGetter(object):
    "Helper class to return positions of patches."

    def __init__(self):
        "Initialize the position getter."
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        "Get the positions for a given batch size, height, and width. Uses caching."
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()

        return pos
