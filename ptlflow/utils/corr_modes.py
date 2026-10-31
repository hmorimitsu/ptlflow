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

If the engine chosen with corr_mode is not available in the current
environment (e.g. an extension is not compiled or the inputs are on the wrong
device), the factories below log a warning and fall back to the pure-PyTorch
engines, which are always available: IterativeCorrBlock ("torch") for the
RAFT-style models and IterSpatialCorrelationSampler ("iter") for the
PWCNet-style models.

For backward compatibility, interpret_corr_mode() translates the pre-v0.5
arguments alternate_corr and corr_fn (and the corr_mode="local" value used by
rapidflow, dpflow and rpknet) into corr_mode values, logging a deprecation
warning.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

from loguru import logger
import torch
import torch.nn as nn

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


# Accepted for backward compatibility: before v0.5, rapidflow, dpflow and
# rpknet used corr_mode="local" for their alt_cuda_corr engine.
LegacyRaftCorrMode = Union[RaftCorrMode, Literal["local"]]


_TRITON_AVAILABLE: Optional[bool] = None
_FALLBACKS_LOGGED = set()


def _triton_available() -> bool:
    """Check whether the triton package can be imported (cached).

    triton publishes Linux-only wheels, so it is an optional dependency of
    localcorr: without it the localcorr eager kernels cannot run.
    """
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401

            _TRITON_AVAILABLE = True
        except (ImportError, OSError):
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _triton_gpu_available() -> bool:
    """Check whether there is any GPU device the triton kernels can run on.

    The triton kernels cannot run on CPU tensors, so the localcorr triton
    modes are only usable if a CUDA or XPU device exists.
    """
    if torch.cuda.is_available():
        return True
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def _log_corr_mode_fallback(corr_mode: str, fallback_mode: str, reason: str) -> None:
    """Log (only once) that corr_mode falls back to the pure-PyTorch engine."""
    key = (corr_mode, fallback_mode, reason)
    if key in _FALLBACKS_LOGGED:
        return
    _FALLBACKS_LOGGED.add(key)
    logger.warning(
        f"!!! corr_mode={corr_mode!r} is not available ({reason}). "
        f"Falling back to corr_mode={fallback_mode!r}, the pure-PyTorch engine !!!"
    )


def _localcorr_unavailable_reason(
    corr_mode: str,
    fmap1: Optional[torch.Tensor] = None,
    is_sampler: bool = False,
) -> Optional[str]:
    """Return why a localcorr corr_mode cannot be used now, or None if it can.

    The localcorr engines only check their requirements when they are first
    called, so this centralizes the checks needed to decide the fallback:

    - the localcorr package must be importable;
    - for eager execution, the triton modes need the triton package and GPU
      (CUDA or XPU) tensors, localcorr_cuda needs the built CUDA kernel
      library and CUDA tensors, and localcorr_sycl needs the built SYCL XPU
      extension and XPU tensors;
    - during ONNX export the engines emit ONNX nodes without loading their
      eager kernels, so the eager-only requirements do not apply (the
      exception is the PWCNet-style localcorr_triton sampler, whose class
      imports triton when it is created).

    Parameters
    ----------
    corr_mode : str
        One of the localcorr_* corr modes.
    fmap1 : Optional[torch.Tensor]
        The first input feature map, used for the device checks of the RAFT
        blocks. None for the PWCNet-style samplers, which are created before
        any tensor is available.
    is_sampler : bool, default False
        Whether the engine is a PWCNet-style spatial correlation sampler.
    """
    if localcorr is None:
        return (
            "the localcorr package is not installed. Install localcorr (its triton "
            "kernels are Linux-only) to use it"
        )
    if torch.onnx.is_in_onnx_export():
        if is_sampler and corr_mode == "localcorr_triton" and not _triton_available():
            return (
                "the triton package is not installed, which the localcorr triton "
                "sampler requires"
            )
        return None
    if corr_mode in ("localcorr_triton", "localcorr_onnx") and not _triton_available():
        return (
            "the triton package is not installed, which the localcorr eager kernels "
            "require"
        )
    if corr_mode in ("localcorr_triton", "localcorr_onnx"):
        # The triton kernels only run on GPU tensors, never on CPU ones.
        if fmap1 is not None:
            if not (fmap1.is_cuda or getattr(fmap1, "is_xpu", False)):
                return (
                    "the triton kernels require CUDA or XPU tensors, but the feature "
                    f"maps are on device {fmap1.device}"
                )
        elif not _triton_gpu_available():
            return (
                "the triton kernels require CUDA or XPU tensors, but no CUDA or XPU "
                "device is available"
            )
    if corr_mode == "localcorr_cuda":
        if localcorr.get_cuda_pyramid_library() is None:
            return (
                "the localcorr CUDA kernel library is not built. Build it with "
                "'python -m localcorr.build_native cuda' to use it"
            )
        if fmap1 is not None and not fmap1.is_cuda:
            return (
                "it requires CUDA tensors, but the feature maps are on device "
                f"{fmap1.device}"
            )
    if corr_mode == "localcorr_sycl":
        if localcorr.get_sycl_xpu_extension() is None:
            return (
                "the localcorr SYCL XPU extension is not built. Build it with "
                "'python -m localcorr.build_native xpu' to use it"
            )
        if fmap1 is not None and not getattr(fmap1, "is_xpu", False):
            return (
                "it requires torch.xpu tensors, but the feature maps are on device "
                f"{fmap1.device}"
            )
    return None


class _LocalcorrSamplerFallbackWrapper(nn.Module):
    """Run a localcorr sampler on GPU tensors, IterSpatialCorrelationSampler on CPU ones.

    The localcorr samplers are created in the model __init__, before any tensor
    exists, so the input device can only be checked when they are called. The
    localcorr kernels cannot run on CPU tensors, so on CPU inputs (eager
    execution) this wrapper logs a warning and falls back to the pure-PyTorch
    IterSpatialCorrelationSampler. During ONNX export the inputs are CPU
    tensors by design, so the fallback must not (and does not) trigger then.
    """

    def __init__(self, sampler, corr_mode, patch_size, dilation_patch):
        super().__init__()
        self.sampler = sampler
        self.corr_mode = corr_mode
        self.iter_sampler = IterSpatialCorrelationSampler(
            kernel_size=1,
            patch_size=patch_size,
            padding=0,
            dilation_patch=dilation_patch,
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        if input1.device.type not in ("cuda", "xpu") and not (
            torch.onnx.is_in_onnx_export()
        ):
            _log_corr_mode_fallback(
                self.corr_mode,
                "iter",
                "the localcorr kernels require CUDA or XPU tensors, but the feature "
                f"maps are on device {input1.device}",
            )
            return self.iter_sampler(input1, input2)
        return self.sampler(input1, input2)


def interpret_corr_mode(
    corr_mode: str,
    alternate_corr: Optional[bool] = None,
    corr_fn: Optional[str] = None,
) -> str:
    """Resolve the corr_mode value from the new argument and the deprecated ones.

    Before v0.5, the correlation engine of the models was configured with the
    alternate_corr (bool) or corr_fn (str) arguments, and rapidflow, dpflow and
    rpknet used corr_mode="local" for the engine that is now called
    alt_cuda_corr. These old arguments and value are still accepted for
    backward compatibility: they are translated to the equivalent corr_mode
    value with a deprecation warning, but they may be removed in the future.

    Parameters
    ----------
    corr_mode : str
        The value given to the corr_mode argument.
    alternate_corr : Optional[bool], default None
        Deprecated. If given, it overrides corr_mode: True maps to
        "alt_cuda_corr" and False maps to "allpairs".
    corr_fn : Optional[str], default None
        Deprecated. If given, it overrides corr_mode: "efficient" maps to
        "alt_cuda_corr" and "default" maps to "allpairs".

    Returns
    -------
    str
        The resolved corr_mode value.

    Raises
    ------
    ValueError
        If corr_fn is not one of "default" or "efficient".
    """
    if alternate_corr is not None:
        resolved = "alt_cuda_corr" if alternate_corr else "allpairs"
        logger.warning(
            f"!!! The argument alternate_corr is deprecated and will be removed in "
            f"the future! Its value {alternate_corr} was converted to "
            f"corr_mode={resolved!r}. Please use the corr_mode argument instead !!!"
        )
        corr_mode = resolved
    if corr_fn is not None:
        if corr_fn == "efficient":
            resolved = "alt_cuda_corr"
        elif corr_fn == "default":
            resolved = "allpairs"
        else:
            raise ValueError(
                f"Invalid corr_fn {corr_fn}. Choose from ['default', 'efficient']."
            )
        logger.warning(
            f"!!! The argument corr_fn is deprecated and will be removed in the "
            f"future! Its value {corr_fn!r} was converted to corr_mode={resolved!r}. "
            f"Please use the corr_mode argument instead !!!"
        )
        corr_mode = resolved
    if corr_mode == "local":
        logger.warning(
            "!!! corr_mode='local' is deprecated and will be removed in the future! "
            "It was converted to corr_mode='alt_cuda_corr'. Please use "
            "corr_mode='alt_cuda_corr' instead !!!"
        )
        corr_mode = "alt_cuda_corr"
    return corr_mode


def log_corr_mode_warnings(corr_mode: str) -> None:
    """Log the standard warnings for corr modes whose engine is unavailable.

    Call this from the model __init__. When the engine is unavailable, the
    factories in this module (get_raft_corr_block and
    get_spatial_corr_sampler) log a warning and fall back to the pure-PyTorch
    engine ("torch" or "iter") instead of raising an error; these warnings
    only give an early notice of that.

    Parameters
    ----------
    corr_mode : str
        The chosen correlation mode.
    """
    if corr_mode == "alt_cuda_corr" and alt_cuda_corr is None:
        logger.warning(
            "!!! alt_cuda_corr is not compiled! corr_mode=alt_cuda_corr will fall back "
            "to the pure-PyTorch engine (corr_mode=torch)! Compile it from "
            "ptlflow/utils/external/alt_cuda_corr to use it !!!"
        )

    if corr_mode == "spatial" and SpatialCorrelationSampler is None:
        logger.warning(
            "!!! spatial_correlation_sampler is not compiled! Using corr_mode=spatial "
            "will fall back to the pure-PyTorch engine (corr_mode=iter)! Compile it "
            "from ptlflow/utils/external/Pytorch-Correlation-extension to use it !!!"
        )

    if corr_mode.startswith("localcorr") and localcorr is None:
        logger.warning(
            f"!!! localcorr is not installed, so corr_mode={corr_mode} will fall back "
            "to the pure-PyTorch engine (corr_mode=torch or iter)! Install localcorr "
            "(its triton kernels are Linux-only) to use it !!!"
        )

    if (
        corr_mode == "localcorr_cuda"
        and localcorr is not None
        and localcorr.get_cuda_pyramid_library() is None
    ):
        logger.warning(
            "!!! the localcorr CUDA kernel library is not built! Eager execution with "
            "corr_mode=localcorr_cuda will fall back to the pure-PyTorch engine (ONNX "
            "export still emits the plugin node)! Build it with "
            "'python -m localcorr.build_native cuda' to use it !!!"
        )

    if (
        corr_mode == "localcorr_sycl"
        and localcorr is not None
        and localcorr.get_sycl_xpu_extension() is None
    ):
        logger.warning(
            "!!! the localcorr SYCL XPU extension is not built! Eager execution with "
            "corr_mode=localcorr_sycl will fall back to the pure-PyTorch engine (ONNX "
            "export for OpenVINO still works)! Build it with "
            "'python -m localcorr.build_native xpu' to use it !!!"
        )


def _raft_corr_unavailable_reason(
    corr_mode: str,
    fmap1: torch.Tensor,
    allpairs_cls: Optional[Any],
    alt_cls: Optional[Any],
) -> Optional[str]:
    """Return why a RAFT-style corr_mode cannot be used now, or None if it can."""
    if corr_mode == "torch":
        return None
    if corr_mode == "allpairs":
        if allpairs_cls is None:
            return "this model has no all-pairs correlation implementation"
        return None
    if corr_mode == "alt_cuda_corr":
        if alt_cls is None:
            return "this model does not support corr_mode='alt_cuda_corr'"
        if alt_cuda_corr is None:
            return (
                "the alt_cuda_corr CUDA extension is not compiled. Compile it from "
                "ptlflow/utils/external/alt_cuda_corr to use it"
            )
        if fmap1.device.type != "cuda":
            return (
                "it requires CUDA tensors, but the feature maps are on device "
                f"{fmap1.device}"
            )
        return None
    if corr_mode.startswith("localcorr"):
        return _localcorr_unavailable_reason(corr_mode, fmap1=fmap1)
    return None


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
    localcorr_* engines are shared. If the chosen engine is not available (missing
    package or compiled extension, or feature maps on a device it does not
    support), a warning is logged and the pure-PyTorch IterativeCorrBlock
    (corr_mode="torch") is used instead.

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
    fallback_reason = _raft_corr_unavailable_reason(
        corr_mode, fmap1, allpairs_cls, alt_cls
    )
    if fallback_reason is not None:
        _log_corr_mode_fallback(corr_mode, "torch", fallback_reason)
        corr_fn = IterativeCorrBlock
    elif corr_mode.startswith("localcorr"):
        return localcorr.get_localcorr_block(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
            corr_mode=corr_mode,
        )
    elif corr_mode == "allpairs":
        corr_fn = allpairs_cls
    elif corr_mode == "alt_cuda_corr":
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

    If the chosen engine is not available (missing package or compiled
    extension), a warning is logged and the pure-PyTorch
    IterSpatialCorrelationSampler (corr_mode="iter") is used instead.

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
        The sampler and an error to be raised by the model forward if the
        sampler could not be created (None on success; with the fallback it
        is always None, kept for backward compatibility).
    """
    fallback_reason = None
    if corr_mode == "spatial":
        if SpatialCorrelationSampler is None:
            fallback_reason = (
                "the spatial_correlation_sampler extension is not compiled. Compile it "
                "from ptlflow/utils/external/Pytorch-Correlation-extension to use it"
            )
    elif corr_mode.startswith("localcorr"):
        fallback_reason = _localcorr_unavailable_reason(corr_mode, is_sampler=True)

    if fallback_reason is not None:
        _log_corr_mode_fallback(corr_mode, "iter", fallback_reason)
        corr_mode = "iter"

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
        return (
            _LocalcorrSamplerFallbackWrapper(
                localcorr.get_localcorr_sampler(
                    patch_size=patch_size,
                    dilation_patch=dilation_patch,
                    corr_mode=corr_mode,
                ),
                corr_mode,
                patch_size,
                dilation_patch,
            ),
            None,
        )
    raise ValueError(f"Invalid corr_mode {corr_mode}.")
