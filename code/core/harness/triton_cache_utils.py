from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

_ACTIVE_PROCESS_TRITON_CACHE_ROOT: Path | None = None


def _cleanup_active_triton_cache_root() -> None:
    global _ACTIVE_PROCESS_TRITON_CACHE_ROOT

    cache_root = _ACTIVE_PROCESS_TRITON_CACHE_ROOT
    _ACTIVE_PROCESS_TRITON_CACHE_ROOT = None
    if cache_root is None:
        return
    shutil.rmtree(cache_root, ignore_errors=True)


atexit.register(_cleanup_active_triton_cache_root)


def _prepare_process_local_triton_cache_root() -> tuple[Path, Path, Path]:
    global _ACTIVE_PROCESS_TRITON_CACHE_ROOT

    previous_root = _ACTIVE_PROCESS_TRITON_CACHE_ROOT
    cache_root = Path(tempfile.mkdtemp(prefix=f"aisp_triton_cache_pid{os.getpid()}_"))
    _ACTIVE_PROCESS_TRITON_CACHE_ROOT = cache_root
    if previous_root is not None and previous_root != cache_root:
        shutil.rmtree(previous_root, ignore_errors=True)

    cache_dir = cache_root / "cache"
    override_dir = cache_root / "override"
    dump_dir = cache_root / "dump"
    for directory in (cache_dir, override_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return cache_dir, override_dir, dump_dir


def reset_triton_runtime_cache(
    emit_warning: Callable[[str, Exception], None] | None = None,
) -> str | None:
    """Reset Triton JIT cache safely across Triton versions.

    Triton 3.5.1 no longer exposes ``triton.runtime.cache.clear()``. When the
    clear API is missing, switch only the current process to a fresh private
    cache root so subsequent compilations cannot reuse stale on-disk artifacts.
    """
    try:
        import triton
    except ImportError:
        return None

    runtime = getattr(triton, "runtime", None)
    cache_module = getattr(runtime, "cache", None)
    clear_cache = getattr(cache_module, "clear", None)
    if callable(clear_cache):
        try:
            clear_cache()
            return None
        except Exception as exc:
            if emit_warning is not None:
                emit_warning("Failed to clear Triton runtime cache", exc)
            raise RuntimeError("Failed to clear Triton runtime cache") from exc

    try:
        from triton import knobs

        cache_dir, override_dir, dump_dir = _prepare_process_local_triton_cache_root()
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
        os.environ["TRITON_OVERRIDE_DIR"] = str(override_dir)
        os.environ["TRITON_DUMP_DIR"] = str(dump_dir)

        cache_knobs = getattr(knobs, "cache", None)
        if cache_knobs is not None:
            reset = getattr(cache_knobs, "reset", None)
            if callable(reset):
                reset()
            getattr(cache_knobs, "dir", None)
            getattr(cache_knobs, "override_dir", None)
            getattr(cache_knobs, "dump_dir", None)
        return str(cache_dir)
    except Exception as exc:
        if emit_warning is not None:
            emit_warning("Failed to rotate Triton runtime cache", exc)
        raise RuntimeError("Failed to rotate Triton runtime cache") from exc


def get_triton_runtime_env_overrides() -> dict[str, str]:
    """Return active Triton runtime env overrides for the current process."""
    if _ACTIVE_PROCESS_TRITON_CACHE_ROOT is None:
        return {}
    keys = ("TRITON_CACHE_DIR", "TRITON_OVERRIDE_DIR", "TRITON_DUMP_DIR")
    return {
        key: value
        for key in keys
        if (value := os.environ.get(key))
    }
