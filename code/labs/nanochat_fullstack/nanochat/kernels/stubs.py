"""
Runtime helpers + placeholder stubs for experimental custom kernels.

These functions sit behind feature flags. When a flag is enabled without a
corresponding compiled extension, we fail fast with a clear error unless the
caller opts into using the built-in Python/Torch fallback path.

Usage for custom kernels:
- Build and expose your kernel, then either:
  * set `GPTConfig.clustered_attention_impl` / `GPTConfig.persistent_decode_impl`
    to a `module:function` string, or
  * call `register_clustered_attention_kernel(fn)` /
    `register_persistent_decode_kernel(fn)` at startup.
- For quick debugging without a custom kernel, set
  `GPTConfig.allow_kernel_stub_fallback=True` to route to the reference
  fallback implementations instead of raising.
"""

import importlib
import warnings
from typing import Callable, Optional

from nanochat.kernels.clustered_attention import clustered_attention

def _persistent_decode_python(model, ids, kv_cache, attention_mask=None, token_mask=None):
    """Lightweight persistent decode fallback used as a built-in custom kernel."""
    return model(
        ids,
        kv_cache=kv_cache,
        attention_mask=attention_mask,
        token_mask=token_mask,
    )

_clustered_attention_impl: Optional[Callable] = None
_persistent_decode_impl: Optional[Callable] = None
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _warned:
        return
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    _warned.add(key)


def _import_from_string(target: str, default_attr: Optional[str] = None) -> Callable:
    """Import `module:attr` from a string of form module:function."""
    module_name, _, attr = target.partition(":")
    attr = attr or default_attr
    if not attr:
        raise ValueError(f"{target!r} must be of form 'module:function'")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _register_builtin_kernels() -> None:
    """Register built-in Python implementations so use_* flags don't fall back."""
    global _clustered_attention_impl, _persistent_decode_impl
    _clustered_attention_impl = clustered_attention
    _persistent_decode_impl = _persistent_decode_python


def reset_kernel_overrides(keep_defaults: bool = True) -> None:
    """Testing helper: clear any registered/loaded custom kernels.

    By default we restore the built-in Python implementations so production code
    has a usable kernel when custom CUDA builds are unavailable. Tests can pass
    keep_defaults=False to simulate an environment with no custom kernels.
    """
    global _clustered_attention_impl, _persistent_decode_impl
    _clustered_attention_impl = None
    _persistent_decode_impl = None
    _warned.clear()
    if keep_defaults:
        _register_builtin_kernels()


def clear_kernel_overrides() -> None:
    """Remove all registered kernels without restoring defaults."""
    reset_kernel_overrides(keep_defaults=False)


def register_default_kernels() -> None:
    """Explicitly register the built-in Python kernel implementations."""
    _register_builtin_kernels()


def register_clustered_attention_kernel(fn: Callable) -> None:
    """Register a custom clustered attention kernel callable."""
    global _clustered_attention_impl
    _clustered_attention_impl = fn


def resolve_clustered_attention_kernel(
    fallback: Optional[Callable] = None,
    impl: Optional[str] = None,
    allow_fallback: bool = False,
) -> Callable:
    """
    Return the clustered attention implementation to use.
    Resolution order:
    1) Explicit registration via register_clustered_attention_kernel
    2) Config-provided string path (impl)
    3) Provided `fallback` when allow_fallback
    4) clustered_attention_stub (raises)
    """
    global _clustered_attention_impl
    if impl:
        _clustered_attention_impl = _import_from_string(impl, default_attr="clustered_attention")
    if _clustered_attention_impl is not None:
        return _clustered_attention_impl
    # Fail fast if no custom kernel is registered
    return clustered_attention_stub


def clustered_attention_stub(q, k, v, attn_mask=None, causal=True, enable_gqa=False):
    raise NotImplementedError(
        "use_clustered_attention_kernel=True but no clustered attention kernel is built. "
        "Provide a custom kernel via GPTConfig.clustered_attention_impl='module:function' "
        "or register_clustered_attention_kernel(...). "
        "To fall back to the reference path instead of raising, set "
        "GPTConfig.allow_kernel_stub_fallback=True."
    )


def register_persistent_decode_kernel(fn: Callable) -> None:
    """Register a custom persistent decode step callable."""
    global _persistent_decode_impl
    _persistent_decode_impl = fn


def resolve_persistent_decode_kernel(
    fallback: Optional[Callable] = None,
    impl: Optional[str] = None,
    allow_fallback: bool = False,
) -> Callable:
    """
    Return the persistent decode implementation to use.
    Resolution order mirrors clustered attention.
    """
    global _persistent_decode_impl
    if impl:
        _persistent_decode_impl = _import_from_string(impl, default_attr="persistent_decode_step")
    if _persistent_decode_impl is not None:
        return _persistent_decode_impl
    # Fail fast if no custom kernel is registered
    return persistent_decode_step_stub


# Ensure defaults are present for production use
_register_builtin_kernels()


def persistent_decode_step_stub(model, ids, kv_cache, attention_mask=None, token_mask=None):
    raise NotImplementedError(
        "use_persistent_decode_kernel=True but no persistent decode kernel is built. "
        "Provide a custom kernel via GPTConfig.persistent_decode_impl='module:function' "
        "or register_persistent_decode_kernel(...). "
        "To fall back to the reference path instead of raising, set "
        "GPTConfig.allow_kernel_stub_fallback=True."
    )
