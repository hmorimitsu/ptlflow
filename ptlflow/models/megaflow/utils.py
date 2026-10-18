import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download

EPS = 1e-6


def load_ckpt(
    module, path: str | None, use_dinov3: bool = False, pretrain_embed: bool = True
):
    """
    Load checkpoint into a LightningModule or nn.Module.

    Supports:
        - None: load default pretrained weights (e.g., safetensors)
        - .ckpt: Lightning checkpoint
        - .pth/.bin: Accelerate/HF-style checkpoint
        - .safetensors: VGGT pretrained checkpoint

    Automatically handles `model.` prefix if module wraps a submodule.
    """
    module_dict = module.state_dict()

    if path is None:
        print("Local VGGT weights not found. Auto-downloading from Hugging Face...")
        repo_id = "facebook/VGGT-1B"
        filename = "model.pt"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # Safely load from the cached local path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        filtered_state_dict = {}
        for k, v in ckpt.items():
            # Only load matching keys
            if k.startswith("aggregator.") or k.startswith(
                "track_head.feature_extractor"
            ):
                key = (
                    "model." + k[len("aggregator.") :]
                    if k.startswith("aggregator.")
                    else "model.flow_head." + k[len("track_head.feature_extractor.") :]
                )
                if (use_dinov3 or not pretrain_embed) and "patch_embed" in key:
                    print(f"Skipping original patch_embed weight: {key}")
                    continue
                if key in module_dict and module_dict[key].shape == v.shape:
                    filtered_state_dict[key] = v
        module_dict.update(filtered_state_dict)
        module.load_state_dict(module_dict, strict=False)
        print(f"Loaded safetensors: {len(filtered_state_dict)} tensors")

    elif path.endswith(".ckpt"):
        # Lightning checkpoint
        state_dict = torch.load(path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        module.load_state_dict(state_dict, strict=False)
        print(f"Loaded Lightning checkpoint: {path}")

    elif path.endswith((".pth", ".bin")):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]

        filtered_state_dict = {}
        for k, v in ckpt.items():
            # Add "model." prefix if needed
            prefixed_k = f"model.{k}" if f"model.{k}" in module_dict else k
            if prefixed_k in module_dict and module_dict[prefixed_k].shape == v.shape:
                filtered_state_dict[prefixed_k] = v
            else:
                print(f"Skipping: {k} -> {prefixed_k}")

        missing, unexpected = module.load_state_dict(filtered_state_dict, strict=False)
        print(
            f"Loaded {len(filtered_state_dict)} tensors. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    elif path == "None":
        return  # No checkpoint to load
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")


class InputPadderMF:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (
            ((self.ht // padding_factor) + 1) * padding_factor - self.ht
        ) % padding_factor
        pad_wd = (
            ((self.wd // padding_factor) + 1) * padding_factor - self.wd
        ) % padding_factor
        self.mode = mode
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
                0,
                0,
            ]
        elif mode == "downzero":
            self._pad = [0, pad_wd, 0, pad_ht, 0, 0]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht, 0, 0]

    def pad(self, input):
        if self.mode == "downzero":
            return F.pad(input, self._pad)
        else:
            return F.pad(input, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def meshgrid2d(B, Y, X, stack=False, device="cuda", on_chans=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        if on_chans:
            grid = torch.stack([grid_x, grid_y], dim=1)
        else:
            grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def gridcloud2d(B, Y, X, norm=False, device="cuda"):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2d(B, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy


def reduce_masked_mean(x, mask, dim=None, keepdim=False, broadcast=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    if not broadcast:
        for a, b in zip(x.size(), mask.size()):
            if not a == b:
                print("some shape mismatch:", x.shape, mask.shape)
            assert a == b  # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)
    mean = numer / denom
    return mean
