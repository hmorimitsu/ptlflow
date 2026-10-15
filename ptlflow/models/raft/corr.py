from typing import Literal

import torch
import torch.nn.functional as F
from .utils import bilinear_sampler

try:
    import alt_cuda_corr
except ImportError:
    alt_cuda_corr = None
from ptlflow.utils.correlation import IterativeCorrBlock

try:
    import localcorr
except ImportError:
    localcorr = None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            dy = torch.linspace(
                -r, r, 2 * r + 1, dtype=coords.dtype, device=coords.device
            )
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim))


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            if coords.dtype == torch.float16:
                fmap1_i = fmap1_i.float()
                fmap2_i = fmap2_i.float()
                coords_i = coords_i.float()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            if coords.dtype == torch.float16:
                corr = corr.half()
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim))


def get_corr_block(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    num_levels: int = 4,
    radius: int = 4,
    corr_mode: Literal[
        "allpairs",
        "alt_cuda_corr",
        "torch",
        "localcorr_triton",
        "localcorr_cuda",
        "localcorr_onnx",
        "localcorr_sycl",
    ] = "allpairs",
):
    """Create the correlation block for the given mode.

    Modes:
    - allpairs: full correlation volume computed once with a matmul (the RAFT default; pure PyTorch).
    - alt_cuda_corr: local correlation with the compiled alt_cuda_corr CUDA extension. It raises an error
      if the extension is not compiled or if the feature maps are not on a CUDA device.
    - torch: local correlation in pure PyTorch (IterativeCorrBlock). Slower, but it has no extra requirements.
    - localcorr_triton, localcorr_onnx: local correlation from the localcorr package computed with its fused
      Triton kernel (all pyramid levels in one launch), with forward and backward passes (triton only has
      Linux wheels, so these modes require Linux and a supported accelerator device). The localcorr_onnx
      mode only differs from localcorr_triton during ONNX export, which emits standard ops instead of the
      TensorRT plugin node.
    - localcorr_cuda: eager inference with the AOT CUDA kernel of the localcorr package, called through
      ctypes (build it with python -m localcorr.build_native cuda). Inference-only and CUDA-only: it raises
      an error for training or on other devices. ONNX export emits the TensorRT plugin node.
    - localcorr_sycl: the Intel counterpart of localcorr_cuda. Eager inference on torch.xpu devices with
      the fused SYCL kernel of the localcorr package (build it with python -m localcorr.build_native xpu);
      inference-only. ONNX export emits the node resolved by the OpenVINO SYCL custom-op extension.
    """
    if corr_mode.startswith("localcorr"):
        if localcorr is None:
            raise ImportError(
                f"corr_mode={corr_mode!r} requires the localcorr package, but it could not be imported. "
                "Install localcorr or use another corr_mode."
            )
        return localcorr.get_localcorr_block(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
            corr_mode=corr_mode,
        )

    if corr_mode == "allpairs":
        corr_fn = CorrBlock
    elif corr_mode == "alt_cuda_corr":
        if alt_cuda_corr is None:
            raise ImportError(
                'corr_mode="alt_cuda_corr" requires the alt_cuda_corr CUDA extension, but it could not be '
                "imported. Compile it (see ptlflow/utils/external/alt_cuda_corr) or use another corr_mode."
            )
        if fmap1.device.type != "cuda":
            raise RuntimeError(
                'corr_mode="alt_cuda_corr" requires CUDA tensors, but the feature maps are on device '
                f"{fmap1.device}."
            )
        corr_fn = AlternateCorrBlock
    elif corr_mode == "torch":
        corr_fn = IterativeCorrBlock
    else:
        raise ValueError(f"Invalid corr_mode {corr_mode}.")

    return corr_fn(fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels)
