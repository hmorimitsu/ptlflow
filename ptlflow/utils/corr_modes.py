"""Shared corr_mode selection for the ptlflow models.

Many models can compute their correlation with swappable engines (compiled
extensions, the localcorr package or pure PyTorch). They expose a corr_mode
argument with two standard sets of choices:

- RAFT-style models (all-pairs correlation volume + local windows):
  "allpairs", "alt_cuda_corr", "torch", "localcorr_cuda", "localcorr_onnx",
  "localcorr_sycl", "localcorr_triton".
- PWCNet-style models (single-level spatial correlation):
  "iter", "spatial", "localcorr_cuda", "localcorr_onnx", "localcorr_sycl",
  "localcorr_triton".

This module centralizes the dispatch and the guards, so that each model only
has to provide its own correlation classes. The model variants for each
corr_mode (e.g. raft_localcorr_triton) are generated with
ptlflow.utils.registry.register_model_variants using the preset dicts defined
here.
"""

from typing import Any, Dict, Literal, Optional, Tuple

from loguru import logger
import torch

from ptlflow.utils.correlation import IterSpatialCorrelationSampler, IterativeCorrBlock

try:
    import alt_cuda_corr
except ImportError:
    alt_cuda_corr = None

try:
    import localcorr
except ImportError:
    localcorr = None

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except ModuleNotFoundError:
    SpatialCorrelationSampler = None


RAFT_CORR_MODES = (
    "allpairs",
    "alt_cuda_corr",
    "torch",
    "localcorr_cuda",
    "localcorr_onnx",
    "localcorr_sycl",
    "localcorr_triton",
)

PWCNET_CORR_MODES = (
    "iter",
    "spatial",
    "localcorr_cuda",
    "localcorr_onnx",
    "localcorr_sycl",
    "localcorr_triton",
)

RaftCorrMode = Literal[
    "allpairs",
    "alt_cuda_corr",
    "torch",
    "localcorr_cuda",
    "localcorr_onnx",
    "localcorr_sycl",
    "localcorr_triton",
]

SpatialCorrMode = Literal[
    "iter",
    "spatial",
    "localcorr_cuda",
    "localcorr_onnx",
    "localcorr_sycl",
    "localcorr_triton",
]

# Presets for register_model_variants: suffix -> preset __init__ kwargs.
RAFT_CORR_VARIANTS = {
    "alt_cuda_corr": {"corr_mode": "alt_cuda_corr"},
    "torch_corr": {"corr_mode": "torch"},
    "localcorr_cuda": {"corr_mode": "localcorr_cuda"},
    "localcorr_onnx": {"corr_mode": "localcorr_onnx"},
    "localcorr_sycl": {"corr_mode": "localcorr_sycl"},
    "localcorr_triton": {"corr_mode": "localcorr_triton"},
}

SPATIAL_CORR_VARIANTS = {
    "spatial_corr": {"corr_mode": "spatial"},
    "localcorr_cuda": {"corr_mode": "localcorr_cuda"},
    "localcorr_onnx": {"corr_mode": "localcorr_onnx"},
    "localcorr_sycl": {"corr_mode": "localcorr_sycl"},
    "localcorr_triton": {"corr_mode": "localcorr_triton"},
}


def log_corr_mode_warnings(corr_mode: str) -> None:
    """Log the standard warnings for corr modes whose engine is unavailable.

    Call this from the model __init__. Each engine still raises a clear error
    when it is actually used; these warnings only give an early notice.

    Parameters
    ----------
    corr_mode : str
        The chosen correlation mode.
    """
    if corr_mode == "alt_cuda_corr" and alt_cuda_corr is None:
        logger.warning(
            "!!! alt_cuda_corr is not compiled! Using corr_mode=alt_cuda_corr will raise an error! "
            "Compile it from ptlflow/utils/external/alt_cuda_corr or choose another corr_mode !!!"
        )

    if corr_mode == "spatial" and SpatialCorrelationSampler is None:
        logger.warning(
            "!!! spatial_correlation_sampler is not compiled! Using corr_mode=spatial will raise "
            "an error! Compile it from ptlflow/utils/external/Pytorch-Correlation-extension "
            "or choose another corr_mode !!!"
        )

    if corr_mode.startswith("localcorr") and localcorr is None:
        logger.warning(
            f"!!! localcorr is not installed, so corr_mode={corr_mode} will not work! "
            "Install localcorr (its triton kernels are Linux-only) or choose another "
            "corr_mode !!!"
        )

    if (
        corr_mode == "localcorr_cuda"
        and localcorr is not None
        and localcorr.get_cuda_pyramid_library() is None
    ):
        logger.warning(
            "!!! the localcorr CUDA kernel library is not built! Using corr_mode=localcorr_cuda "
            "will raise an error! Build it with 'python -m localcorr.build_native cuda' or choose "
            "another corr_mode !!!"
        )

    if (
        corr_mode == "localcorr_sycl"
        and localcorr is not None
        and localcorr.get_sycl_xpu_extension() is None
    ):
        logger.warning(
            "!!! the localcorr SYCL XPU extension is not built! Eager execution with "
            "corr_mode=localcorr_sycl will raise an error (ONNX export for OpenVINO still works)! "
            "Build it with 'python -m localcorr.build_native xpu' or choose another corr_mode !!!"
        )


def get_raft_corr_block(
    corr_mode: str,
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    radius: int = 4,
    num_levels: int = 4,
    allpairs_cls: Optional[Any] = None,
    alt_cls: Optional[Any] = None,
):
    """Create a RAFT-style correlation block for the given mode.

    The allpairs and alt_cuda_corr engines are model-specific classes; the torch and
    localcorr_* engines are shared. alt_cuda_corr raises an error if the extension is
    not compiled or if the feature maps are not on a CUDA device.

    Parameters
    ----------
    corr_mode : str
        One of RAFT_CORR_MODES.
    fmap1, fmap2 : torch.Tensor
        The input feature maps, [B, C, H, W].
    radius, num_levels : int
        Correlation window radius and number of pyramid levels.
    allpairs_cls : Optional[Any]
        The model class computing the full correlation volume (used by "allpairs").
    alt_cls : Optional[Any]
        The model class using the alt_cuda_corr extension (used by "alt_cuda_corr").

    Returns
    -------
    The correlation block, callable with coords [B, 2, H, W].
    """
    if corr_mode.startswith("localcorr"):
        if localcorr is None:
            raise ImportError(
                f"corr_mode={corr_mode!r} requires the localcorr package, but it could not be "
                "imported. Install localcorr or use another corr_mode."
            )
        return localcorr.get_localcorr_block(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
            corr_mode=corr_mode,
        )

    if corr_mode == "allpairs":
        if allpairs_cls is None:
            raise ValueError(
                'corr_mode="allpairs" is not supported by this model (it has no '
                "all-pairs correlation implementation). Use another corr_mode."
            )
        corr_fn = allpairs_cls
    elif corr_mode == "alt_cuda_corr":
        if alt_cls is None:
            raise ValueError(
                'corr_mode="alt_cuda_corr" is not supported by this model. '
                "Use another corr_mode."
            )
        if alt_cuda_corr is None:
            raise ImportError(
                'corr_mode="alt_cuda_corr" requires the alt_cuda_corr CUDA extension, but it '
                "could not be imported. Compile it (see ptlflow/utils/external/alt_cuda_corr) "
                "or use another corr_mode."
            )
        if fmap1.device.type != "cuda":
            raise RuntimeError(
                'corr_mode="alt_cuda_corr" requires CUDA tensors, but the feature maps are on '
                f"device {fmap1.device}."
            )
        corr_fn = alt_cls
    elif corr_mode == "torch":
        corr_fn = IterativeCorrBlock
    else:
        raise ValueError(f"Invalid corr_mode {corr_mode}.")

    return corr_fn(fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels)


def get_spatial_corr_sampler(
    corr_mode: str,
    patch_size: int,
    dilation_patch: int = 1,
) -> Tuple[Any, Optional[Exception]]:
    """Create a PWCNet-style spatial correlation sampler for the given mode.

    Parameters
    ----------
    corr_mode : str
        One of PWCNET_CORR_MODES. "iter" is the pure-PyTorch
        IterSpatialCorrelationSampler (the default); "spatial" is the compiled
        spatial_correlation_sampler extension; the localcorr_* modes use the
        localcorr package.
    patch_size : int
        The correlation patch size (2 * search_range + 1).
    dilation_patch : int, default 1
        The step between consecutive correlation window positions.

    Returns
    -------
    Tuple[Any, Optional[Exception]]
        The sampler, and an error to be raised by the model forward if the sampler
        could not be created (None on success).
    """
    if corr_mode == "iter":
        return (
            IterSpatialCorrelationSampler(
                kernel_size=1,
                patch_size=patch_size,
                padding=0,
                dilation_patch=dilation_patch,
            ),
            None,
        )
    if corr_mode == "spatial":
        if SpatialCorrelationSampler is None:
            error = ImportError(
                'corr_mode="spatial" requires the spatial_correlation_sampler extension, '
                "but it could not be imported. Compile it (see "
                "ptlflow/utils/external/Pytorch-Correlation-extension) or use another "
                "corr_mode."
            )
            return None, error
        return (
            SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=patch_size,
                padding=0,
                dilation_patch=dilation_patch,
            ),
            None,
        )
    if corr_mode.startswith("localcorr"):
        if localcorr is None:
            error = ImportError(
                f"corr_mode={corr_mode!r} requires the localcorr package, but it could "
                "not be imported. Install localcorr or use another corr_mode."
            )
            return None, error
        return (
            localcorr.get_localcorr_sampler(
                patch_size=patch_size,
                dilation_patch=dilation_patch,
                corr_mode=corr_mode,
            ),
            None,
        )
    raise ValueError(f"Invalid corr_mode {corr_mode}.")
