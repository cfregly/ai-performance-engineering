from __future__ import annotations

"""
Utility helpers to integrate Transformer Engine FP8 layers into Blackwell benchmarks.

The goal of this module is to make it trivial for benchmarks (e.g. chapter 16)
to opt into *real* FP8 execution using NVIDIA's Transformer Engine (TE) when the
library is available at runtime. The helpers here are intentionally defensive:
they report a clear diagnostic when TE cannot be imported (common on systems
without the compiled extension) and they provide convenience functions for
replacing standard `nn.Linear` layers with TE equivalents as well as for entering
the FP8 autocast region.
"""

import contextlib
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "transformer_engine_available",
    "transformer_engine_warning",
    "fp8_autocast",
    "convert_linear_layers",
    "make_delayed_scaling_recipe",
    "TransformerEngineUnavailable",
]


class TransformerEngineUnavailable(RuntimeError):
    """Raised when Transformer Engine is required but not installed."""


try:
    import transformer_engine.pytorch as te  # type: ignore
    from transformer_engine.common.recipe import DelayedScaling, Format  # type: ignore

    _TE_AVAILABLE = True
    _TE_IMPORT_ERROR: Optional[BaseException] = None
except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError, RuntimeError) as exc:  # pragma: no cover - environment dependent
    te = None  # type: ignore
    DelayedScaling = None  # type: ignore
    Format = None  # type: ignore
    _TE_AVAILABLE = False
    _TE_IMPORT_ERROR = exc


def transformer_engine_available() -> bool:
    """Return True if Transformer Engine imported successfully."""
    return _TE_AVAILABLE


def transformer_engine_warning() -> str:
    """Return a human-readable explanation when TE is missing."""
    if _TE_AVAILABLE:
        return ""
    reason = f"{_TE_IMPORT_ERROR}" if _TE_IMPORT_ERROR else "Transformer Engine not installed"
    return (
        "Transformer Engine FP8 disabled. "
        "Install with `pip install transformer-engine[pytorch]` and ensure "
        "the CUDA extensions build successfully. "
        f"Reason: {reason}"
    )


def _default_recipe(
    *,
    margin: int = 0,
    history: int = 64,
    format_name: str = "hybrid",
    fp8_mha: bool = True,
    fp8_dpa: bool = True,
):
    """Create a sensible DelayedScaling recipe."""
    if not _TE_AVAILABLE:
        raise TransformerEngineUnavailable(transformer_engine_warning())
    format_map = {
        "hybrid": Format.HYBRID,
        "e4m3": Format.E4M3,
        "e5m2": Format.E5M2,
    }
    fmt = format_map.get(format_name.lower(), Format.HYBRID)
    return DelayedScaling(
        margin=margin,
        fp8_format=fmt,
        amax_history_len=history,
        fp8_mha=fp8_mha,
        fp8_dpa=fp8_dpa,
    )


def make_delayed_scaling_recipe(
    *,
    margin: int = 0,
    history: int = 64,
    format_name: str = "hybrid",
    fp8_mha: bool = True,
    fp8_dpa: bool = True,
):
    """Public helper for callers that need to customize the FP8 recipe."""
    return _default_recipe(
        margin=margin,
        history=history,
        format_name=format_name,
        fp8_mha=fp8_mha,
        fp8_dpa=fp8_dpa,
    )


@contextlib.contextmanager
def fp8_autocast(enabled: bool = True, *, recipe: Optional["DelayedScaling"] = None):
    """
    Context manager that mirrors `transformer_engine.pytorch.fp8_autocast` but
    gracefully degrades when TE is unavailable.
    """
    if not enabled or not _TE_AVAILABLE:
        yield
        return
    if recipe is None:
        recipe = _default_recipe()
    assert te is not None  # mypy guard
    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            yield
    except TypeError:
        # Older TE releases expect `recipe`; fall back for compatibility.
        with te.fp8_autocast(enabled=True, recipe=recipe):
            yield


def convert_linear_layers(
    module: nn.Module,
    *,
    params_dtype: torch.dtype = torch.float16,
    skip_if_cpu: bool = True,
) -> int:
    """
    Recursively replace `nn.Linear` instances in `module` with Transformer Engine
    equivalents so they participate in FP8 execution.

    Returns the number of layers converted. Raises `TransformerEngineUnavailable`
    when TE is not installed.
    """
    if not _TE_AVAILABLE:
        raise TransformerEngineUnavailable(transformer_engine_warning())
    assert te is not None  # mypy guard

    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            device = child.weight.device
            if skip_if_cpu and device.type != "cuda":
                # TE only supports CUDA devices; silently skip CPU layers.
                continue
            te_linear = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=params_dtype,
                device=device,
            )
            with torch.inference_mode():
                te_linear.weight.copy_(child.weight.detach().to(te_linear.weight.dtype))
                if child.bias is not None and te_linear.use_bias:
                    te_linear.bias.copy_(child.bias.detach().to(te_linear.bias.dtype))
            setattr(module, name, te_linear)
            converted += 1
        else:
            converted += convert_linear_layers(
                child,
                params_dtype=params_dtype,
                skip_if_cpu=skip_if_cpu,
            )
    return converted
