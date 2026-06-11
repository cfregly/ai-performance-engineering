"""Utilities for safely applying torch.compile and precision defaults."""

from __future__ import annotations

import importlib
import logging
import threading
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, cast
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

_LEGACY_TF32_PATCHED = False


def _configure_compiler_defaults() -> None:
    """Enable Blackwell-friendly torch.compile defaults (TMA, autotune, unique kernel names)."""
    # First, configure inductor config if available (this sets unique_kernel_names
    # which prevents the "Placeholder.DESCRIPTIVE_NAME" Triton compilation error)
    try:
        import torch._inductor.config as inductor_config
        triton_inductor_cfg = getattr(inductor_config, "triton", None)
        if triton_inductor_cfg is not None:
            # CRITICAL: Enable unique_kernel_names to avoid "Placeholder.DESCRIPTIVE_NAME" errors
            # This ensures Triton kernels get proper names instead of placeholders
            if hasattr(triton_inductor_cfg, "unique_kernel_names"):
                triton_inductor_cfg.unique_kernel_names = True
    except Exception:
        pass  # Inductor config may not be available in all builds

    compiler_api = getattr(torch, "compiler", None)
    if compiler_api is None or not hasattr(compiler_api, "config"):
        logger.warning(
            "torch.compiler.config is missing; skipping Blackwell compiler defaults. "
            "Upgrade to a PyTorch build with compiler config support to auto-enable TMA."
        )
        return

    config = compiler_api.config  # type: ignore[attr-defined]

    cuda_cfg = getattr(config, "cuda", None)
    if cuda_cfg is None or not hasattr(cuda_cfg, "enable_tma"):
        logger.debug(
            "torch.compiler.config.cuda.enable_tma is unavailable; leaving TMA defaults untouched."
        )
    else:
        try:
            cuda_cfg.enable_tma = True
        except Exception:
            logger.warning("Failed to enable TMA via torch.compiler.config.cuda.enable_tma", exc_info=True)

    triton_cfg = getattr(config, "triton", None)
    if triton_cfg is None:
        logger.debug("torch.compiler.config.triton is unavailable; Triton defaults unchanged.")
        return
    if hasattr(triton_cfg, "tma_support"):
        try:
            triton_cfg.tma_support = True
        except Exception:
            logger.warning("Failed to enable Triton TMA support via compiler.config", exc_info=True)
    else:
        logger.debug("torch.compiler.config.triton.tma_support is unavailable; leaving default.")
    if hasattr(triton_cfg, "autotune_mode"):
        try:
            triton_cfg.autotune_mode = "max-autotune"
        except Exception:
            logger.warning("Failed to set Triton autotune_mode=max-autotune", exc_info=True)
    else:
        logger.debug("torch.compiler.config.triton.autotune_mode is unavailable; leaving default.")
    # Also set unique_kernel_names on compiler.config.triton if available
    if hasattr(triton_cfg, "unique_kernel_names"):
        try:
            triton_cfg.unique_kernel_names = True
        except Exception:
            pass


_configure_compiler_defaults()


def _patch_legacy_tf32_attributes() -> None:
    """
    Override legacy TF32 accessors so reads/writes no-op instead of calling the
    deprecated CUDA context APIs (which cause mixing errors once the new API is used).
    """
    global _LEGACY_TF32_PATCHED
    if _LEGACY_TF32_PATCHED:
        return

    try:
        cu_blas_cls = torch.backends.cuda.cuBLASModule
    except AttributeError:
        cu_blas_cls = None

    if cu_blas_cls is not None:
        original_getattr = cu_blas_cls.__getattr__
        original_setattr = cu_blas_cls.__setattr__

        def _patched_getattr(self, name):
            if name == "allow_tf32":
                return getattr(self, "_legacy_allow_tf32", False)
            return original_getattr(self, name)

        def _patched_setattr(self, name, value):
            if name == "allow_tf32":
                object.__setattr__(self, "_legacy_allow_tf32", bool(value))
                return value
            return original_setattr(self, name, value)

        cu_blas_cls.__getattr__ = _patched_getattr  # type: ignore[assignment]
        cu_blas_cls.__setattr__ = _patched_setattr  # type: ignore[assignment]

    try:
        cudnn_cls = type(torch.backends.cudnn)
    except AttributeError:
        cudnn_cls = None

    if cudnn_cls is not None:
        def _cudnn_allow_tf32_get(self):
            return getattr(self, "_legacy_allow_tf32", False)

        def _cudnn_allow_tf32_set(self, value):
            object.__setattr__(self, "_legacy_allow_tf32", bool(value))

        cudnn_cls.allow_tf32 = property(  # type: ignore[assignment]
            _cudnn_allow_tf32_get,
            _cudnn_allow_tf32_set,
        )

    _LEGACY_TF32_PATCHED = True


def _mirror_legacy_tf32_flags(enable_matmul: bool, enable_cudnn: bool) -> None:
    """
    Store the TF32 state on the patched legacy accessors so callers that still read
    torch.backends.*.allow_tf32 observe the right boolean without touching the old API.
    """
    # These attributes may not exist in older PyTorch versions
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = enable_matmul  # type: ignore[attr-defined]
    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = enable_cudnn  # type: ignore[attr-defined]


def get_optimal_compile_mode(preferred_mode: str = "max-autotune", sm_threshold: int = 68) -> str:
    """
    Get the optimal torch.compile mode based on GPU SM count.
    
    max-autotune requires >= 68 SMs (Streaming Multiprocessors) for GEMM operations.
    GPUs with fewer SMs will fall back to reduce-overhead to avoid warnings.
    
    Parameters
    ----------
    preferred_mode:
        The preferred compile mode (default: "max-autotune")
    sm_threshold:
        Minimum number of SMs required for max-autotune (default: 68)
    
    Returns
    -------
    str
        The compile mode to use: "max-autotune" if GPU has enough SMs,
        otherwise "reduce-overhead"
    """
    # Testing / A-B facility: force a specific compile mode regardless of the
    # caller's preference (used by the GB300 sm_103a-unlock re-sweep to measure
    # incumbent arms, e.g. AISP_COMPILE_MODE_OVERRIDE=reduce-overhead). Not set
    # in normal operation.
    override = os.environ.get("AISP_COMPILE_MODE_OVERRIDE", "").strip()
    if override:
        return override

    if preferred_mode != "max-autotune":
        return preferred_mode

    if not torch.cuda.is_available():
        return "reduce-overhead"

    # HISTORICAL (root-caused 2026-06-11, guard retired same day): the "sm_103
    # tcgen05.wait.st LLVM abort" under max-autotune was NOT a Triton >= 3.6 bug.
    # It was self-inflicted by core/benchmark/triton_compat.py, which de-suffixed
    # sm_103a -> sm_103; the arch-conditional tcgen05 intrinsics are only
    # selectable for the 'a' targets, so LLVM hard-aborted (uncatchable SIGABRT).
    # triton_compat.py now preserves the 'a' suffix for all arches and the
    # max-autotune paths were re-verified clean on GB300 / Triton 3.7 (see
    # code/upstream/triton-tcgen05-wait-st/STATUS.md), so the sm_103 downgrade
    # to "default" is gone.
    try:
        device_index = torch.cuda.current_device()
        num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
        
        if num_sms >= sm_threshold:
            return "max-autotune"
        else:
            return "reduce-overhead"
    except Exception:
        # If we can't determine SM count, use reduce-overhead to be safe
        return "reduce-overhead"


_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


def _get_dynamo_error_on_graph_break() -> Optional[Callable[[bool], contextmanager]]:
    try:
        import torch._dynamo as _dynamo  # type: ignore[attr-defined]

        return getattr(_dynamo, "error_on_graph_break", None)
    except Exception:
        return None


@contextmanager
def error_on_graph_break(enable: Optional[bool]) -> None:
    """Contextually toggle Dynamo graph-break behavior when available."""

    cm = _get_dynamo_error_on_graph_break()
    if cm is None or enable is None:
        yield
    else:
        with cm(enable):
            yield


def maybe_nested_compile_region(fn: _CallableT) -> _CallableT:
    """
    Decorate a callable with torch.compiler.nested_compile_region when available.

    Safe to use on any callable; it is a no-op on Torch builds that lack the API.
    """
    compiler_api = getattr(torch, "compiler", None)
    nested = getattr(compiler_api, "nested_compile_region", None)
    if nested is None:
        return fn
    try:
        return cast(_CallableT, nested(fn))
    except Exception:
        # Defensive: fall back to original callable if decorator wrapping fails.
        return fn


def _get_torch_compile() -> Callable[..., Any]:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover - depends on PyTorch build
        raise RuntimeError("SKIPPED: torch.compile is unavailable in this PyTorch build.")
    return compile_fn


_ALLOWED_COMPILE_KWARGS = {"fullgraph", "dynamic", "backend", "options", "disable"}


def _normalize_compile_kwargs(kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    extra: Dict[str, Any] = dict(kwargs)  # defensive copy so callers can reuse dicts
    preferred_mode = extra.pop("mode", "max-autotune")
    chosen_mode = get_optimal_compile_mode(preferred_mode)
    filtered = {key: value for key, value in extra.items() if key in _ALLOWED_COMPILE_KWARGS}
    return chosen_mode, filtered


def compile_callable(fn: _CallableT, **kwargs: Any) -> _CallableT:
    """
    Compile an arbitrary callable with torch.compile and propagate failures.

    Parameters mirror torch.compile's kwargs, plus ``mode`` which defaults to
    ``max-autotune`` but is automatically downgraded via ``get_optimal_compile_mode``.
    """
    err_on_graph_break = kwargs.pop("error_on_graph_break", None)
    use_nested_region = kwargs.pop("nested_compile_region", False)
    compile_fn = _get_torch_compile()
    chosen_mode, extra = _normalize_compile_kwargs(kwargs)
    target_fn = maybe_nested_compile_region(fn) if use_nested_region else fn
    try:
        with error_on_graph_break(err_on_graph_break):
            compiled = compile_fn(target_fn, mode=chosen_mode, **extra)
    except Exception as exc:  # pragma: no cover - propagate compile failures
        raise RuntimeError(
            f"torch.compile failed in compile_callable (mode={chosen_mode}): {exc}"
        ) from exc
    return cast(_CallableT, compiled)


def compile_model(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    """
    Compile a module with torch.compile when available.

    Parameters follow the historical signature used throughout the chapters.
    Unknown keyword arguments are ignored intentionally to preserve backwards
    compatibility with chapter-specific wrappers.
    
    FAIL FAST: If compilation fails, we raise an error with "SKIPPED:" prefix
    so the benchmark harness properly skips the benchmark. We do NOT silently
    fall back to eager mode - that would produce invalid benchmark results.
    """
    if getattr(module, "_is_compiled_benchmark_module", False):
        return module

    try:
        compiled = compile_callable(module, **kwargs)
    except Exception as exc:
        message = str(exc)
        major_minor = None
        if torch.cuda.is_available():
            try:
                major_minor = torch.cuda.get_device_capability()
            except Exception:
                major_minor = None
        major = major_minor[0] if major_minor else None
        
        # Identify known torch.compile issues - SKIP (don't silently fallback!)
        # These are hardware/toolchain limitations, not code bugs
        skip_reasons = []
        
        if "NoTritonConfigsError" in message:
            skip_reasons.append("Triton has no valid configs for this kernel")
        if "ptxas fatal" in message:
            skip_reasons.append("PTX assembler error (toolchain incompatibility)")
        if "sm_121" in message:
            skip_reasons.append("SM 12.1 (Blackwell) not supported by current Triton")
        if major is not None and major >= 12:
            skip_reasons.append(f"SM {major}.x architecture not fully supported")
        
        # These indicate code bugs that should be FIXED, not skipped
        # But we still skip with clear error so developer knows to fix
        if "SymNodeVariable" in message:
            skip_reasons.append("SymNodeVariable bug - symbolic tracing incompatible (FIX THE CODE)")
        if "SymInt" in message and "cannot be" in message:
            skip_reasons.append("Dynamic shape tracing issue (FIX THE CODE)")
        if "Unsupported: call_function aten" in message:
            skip_reasons.append("Unsupported aten operation in torch.compile (FIX THE CODE)")
        
        if skip_reasons:
            reason_str = "; ".join(skip_reasons)
            raise RuntimeError(
                f"SKIPPED: torch.compile failed - {reason_str}. "
                f"Original error: {message[:200]}"
            ) from exc
        
        # Unknown error - re-raise as-is
        raise
    
    setattr(compiled, "_is_compiled_benchmark_module", True)
    return compiled


def _current_matmul_precision() -> Optional[str]:
    backend = getattr(torch.backends.cuda, "matmul", None)
    if backend is not None and hasattr(backend, "fp32_precision"):
        return str(getattr(backend, "fp32_precision"))
    if hasattr(torch, 'get_float32_matmul_precision'):
        return torch.get_float32_matmul_precision()
    return None


def _current_cudnn_precision() -> Optional[str]:
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
        return str(getattr(cudnn_conv, "fp32_precision"))
    return None


def _map_precision_to_backend(precision: str) -> str:
    """Map torch precision strings to CUDA backend values (ieee/tf32/none)."""
    p = precision.lower()
    if p in ("high", "tf32"):
        return "tf32"
    if p in ("highest", "fp32", "ieee"):
        return "ieee"
    if p in ("medium",):
        return "tf32"  # medium also uses tf32
    if p in ("none",):
        return "none"
    # Return as-is if already a valid backend value
    return precision


def _set_matmul_precision(precision: str) -> None:
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        backend_mut = cast(Any, matmul_backend)
        backend_mut.fp32_precision = _map_precision_to_backend(precision)
    if hasattr(torch, 'set_float32_matmul_precision'):
        # Normalize precision to valid PyTorch API values: 'highest', 'high', or 'medium'
        # Map legacy/internal "tf32" to "high" for PyTorch 2.0+ compatibility
        api_precision = precision.lower()
        if api_precision in ("tf32", "enable", "on", "true"):
            api_precision = "high"
        elif api_precision in ("fp32", "disable", "off", "false", "none"):
            api_precision = "highest"
        elif api_precision not in ("highest", "high", "medium"):
            api_precision = "high"  # Default fallback
        torch.set_float32_matmul_precision(api_precision)


def _set_cudnn_precision(precision: str) -> None:
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if cudnn_conv is not None and hasattr(cudnn_conv, "fp32_precision"):
        try:
            cudnn_mut = cast(Any, cudnn_conv)
            cudnn_mut.fp32_precision = _map_precision_to_backend(precision)
        except RuntimeError:
            pass


def _normalize_precision(value: Optional[str], enable: Optional[bool]) -> Optional[str]:
    if value is not None:
        normalized = value.lower()
        if normalized in ("tf32", "high", "enable", "on", "true"):
            return "high"
        if normalized in ("fp32", "highest", "disable", "off", "false"):
            return "highest"
        if normalized in ("medium",):
            return normalized
        return value
    if enable is None:
        return None
    return "high" if enable else "highest"


def configure_tf32(
    *,
    enable_matmul: Optional[bool] = None,
    enable_cudnn: Optional[bool] = None,
    matmul_precision: Optional[str] = None,
    cudnn_precision: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Apply TF32 settings via the PyTorch 2.10 precision APIs."""
    if not torch.cuda.is_available():
        return (None, None)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Please use the new API settings to control TF32.*",
            category=UserWarning,
        )
        _patch_legacy_tf32_attributes()
        previous = (_current_matmul_precision(), _current_cudnn_precision())

        new_matmul = _normalize_precision(matmul_precision, enable_matmul)
        new_cudnn = _normalize_precision(cudnn_precision, enable_cudnn)

        if new_matmul is not None:
            _set_matmul_precision(new_matmul)
        if new_cudnn is not None:
            _set_cudnn_precision(new_cudnn)

        matmul_enabled = (new_matmul or previous[0] or "").lower() == "high"
        cudnn_enabled = (new_cudnn or previous[1] or "").lower() == "high"
        _mirror_legacy_tf32_flags(matmul_enabled, cudnn_enabled)

    return previous


def restore_tf32(state: Tuple[Optional[str], Optional[str]]) -> None:
    """Restore TF32 state captured via configure_tf32."""
    if not torch.cuda.is_available():
        return
    prev_matmul, prev_cudnn = state
    if prev_matmul is not None:
        _set_matmul_precision(prev_matmul)
    if prev_cudnn is not None:
        _set_cudnn_precision(prev_cudnn)
    matmul_enabled = (prev_matmul or "").lower() == "high"
    cudnn_enabled = (prev_cudnn or "").lower() == "high"
    _mirror_legacy_tf32_flags(matmul_enabled, cudnn_enabled)


@contextmanager
def tf32_override(
    *,
    enable_matmul: Optional[bool] = None,
    enable_cudnn: Optional[bool] = None,
    matmul_precision: Optional[str] = None,
    cudnn_precision: Optional[str] = None,
):
    state = configure_tf32(
        enable_matmul=enable_matmul,
        enable_cudnn=enable_cudnn,
        matmul_precision=matmul_precision,
        cudnn_precision=cudnn_precision,
    )
    try:
        yield
    finally:
        restore_tf32(state)


def enable_tf32(
    *,
    matmul_precision: str = "tf32",
    cudnn_precision: str = "tf32",
    set_global_precision: bool = True,
) -> None:
    """
    Configure TF32 execution using the new PyTorch 2.10 APIs only.
    """
    if not torch.cuda.is_available():
        return

    state = configure_tf32(
        matmul_precision=matmul_precision,
        cudnn_precision=cudnn_precision,
    )
    if set_global_precision:
        target = _normalize_precision(matmul_precision, None) or "high"
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(target)
    # state unused but kept for signature compatibility


@lru_cache()
def _get_compiled_architectures() -> Tuple[str, ...]:
    """Return the set of architectures baked into the current PyTorch build."""
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if get_arch_list is None:
        return tuple()
    return tuple(get_arch_list())


def _format_arch(major: int, minor: int) -> str:
    return f"sm_{major}{minor}"


def _supported_arch_aliases(major: int, minor: int) -> Tuple[str, ...]:
    """Return possible arch strings (sm_X, sm_Xa, compute_X) for a device."""
    digits = f"{major}{minor}"
    aliases = []
    for suffix in ("", "a", "b"):
        aliases.append(f"sm_{digits}{suffix}")
        aliases.append(f"compute_{digits}{suffix}")
    return tuple(aliases)


def is_torch_compile_supported_on_device(device_index: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """
    Determine if torch.compile has kernel support for the active CUDA architecture.
    
    Returns (True, None) when the architecture is baked into the current PyTorch build,
    otherwise (False, reason) describing the capability gap.
    """
    if not torch.cuda.is_available():
        return True, None
    
    try:
        if device_index is None:
            if getattr(torch.cuda, "_initialized", False):
                device_index = torch.cuda.current_device()
            else:
                device_index = 0
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception as exc:  # pragma: no cover - depends on runtime
        return False, f"torch.compile: unable to query CUDA capability ({exc})."
    
    supported_arches = _get_compiled_architectures()
    aliases = _supported_arch_aliases(major, minor)
    for alias in aliases:
        if alias in supported_arches:
            return True, None
    
    arch_display = _format_arch(major, minor)
    supported_display = ", ".join(supported_arches) if supported_arches else "unknown"
    reason = (
        f"GPU {arch_display} (compute capability {major}.{minor}) is missing from the "
        f"PyTorch {torch.__version__} SASS list [{supported_display}]."
    )
    return False, reason


_WARNED_MESSAGES: set[str] = set()


def _log_once(message: Optional[str]) -> None:
    if not message:
        return
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    logger.warning(message)


_CUDAGRAPH_TLS_PATCHED = False


def _make_tls_default(attr_name: str) -> Optional[Any]:
    if attr_name == "tree_manager_containers":
        return {}
    if attr_name == "tree_manager_locks":
        return defaultdict(threading.Lock)
    return None


def _patch_cudagraph_tls_bug() -> None:
    """
    Work around a PyTorch TLS regression on newly added architectures (e.g. SM12.1).
    
    When CUDA graph trees try to read thread-local state before the Inductor module
    has stashed the default dictionaries, an assertion is raised. We defensively
    recreate the missing structures and stash them so later reads succeed.
    """
    global _CUDAGRAPH_TLS_PATCHED
    if _CUDAGRAPH_TLS_PATCHED:
        return

    try:
        trees = importlib.import_module("torch._inductor.cudagraph_trees")
    except Exception as exc:  # pragma: no cover - safety path
        logger.debug("Unable to import torch._inductor.cudagraph_trees: %s", exc)
        return

    original_get_obj = getattr(trees, "get_obj", None)
    if original_get_obj is None:
        return

    def patched_get_obj(local: Any, attr_name: str) -> Any:
        if hasattr(local, attr_name):
            return getattr(local, attr_name)

        if torch._C._is_key_in_tls(attr_name):
            return torch._C._get_obj_in_tls(attr_name)

        fallback = _make_tls_default(attr_name)
        if fallback is None:
            # Mirror the upstream assertion so failures are loud
            raise AssertionError(f"Missing TLS object for {attr_name}")

        setattr(local, attr_name, fallback)
        try:
            torch._C._stash_obj_in_tls(attr_name, fallback)
        except Exception:
            logger.debug("Unable to stash TLS object for %s", attr_name, exc_info=True)
        _log_once(f"Rebuilt missing CUDA graph TLS bucket '{attr_name}' to avoid torch.compile crashes.")
        return fallback

    trees.get_obj = patched_get_obj  # type: ignore[assignment]
    _CUDAGRAPH_TLS_PATCHED = True


_patch_cudagraph_tls_bug()
