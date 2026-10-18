"""
UFM Model Utils
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=10)
def get_meshgrid_torch(W, H, device):
    u, v = torch.meshgrid(torch.arange(W, device=device).float(), torch.arange(H, device=device).float(), indexing="xy")

    uv = torch.stack((u, v), dim=-1)

    return uv
