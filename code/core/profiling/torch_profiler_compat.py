"""Compatibility initialization for Torch profiler CUDA activity capture."""

from __future__ import annotations

import ctypes
import os
import re
import threading
from typing import Any

import torch

_CUPTI_SUCCESS = 0
_CUPTI_PER_THREAD_ACTIVITY_BUFFER = 9
_PREPARE_LOCK = threading.Lock()
_prepared_pid: int | None = None


def _major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _requires_cupti_buffer_initialization(torch_version: str, cuda_version: str | None) -> bool:
    return _major_minor(torch_version) == (2, 9) and _major_minor(cuda_version) == (
        13,
        0,
    )


def _call_cupti_attribute(
    function: Any,
    *,
    value: ctypes.c_uint8,
    operation: str,
) -> None:
    value_size = ctypes.c_size_t(ctypes.sizeof(value))
    result = int(
        function(
            _CUPTI_PER_THREAD_ACTIVITY_BUFFER,
            ctypes.byref(value_size),
            ctypes.byref(value),
        )
    )
    if result != _CUPTI_SUCCESS:
        raise RuntimeError(
            "Torch profiler CUDA compatibility initialization failed while "
            f"calling {operation}. CUPTI returned status {result}."
        )
    if value_size.value != ctypes.sizeof(value):
        raise RuntimeError(
            "Torch profiler CUDA compatibility initialization received an "
            f"unexpected {operation} value size of {value_size.value}."
        )


def _initialize_cupti_per_thread_buffer(cupti: Any) -> None:
    get_attribute = cupti.cuptiActivityGetAttribute
    set_attribute = cupti.cuptiActivitySetAttribute
    get_attribute.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    get_attribute.restype = ctypes.c_int
    set_attribute.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    set_attribute.restype = ctypes.c_int

    before = ctypes.c_uint8()
    _call_cupti_attribute(
        get_attribute,
        value=before,
        operation="cuptiActivityGetAttribute before initialization",
    )
    if before.value not in (0, 1):
        raise RuntimeError(
            "Torch profiler CUDA compatibility initialization read an invalid "
            f"per-thread activity-buffer value of {before.value}."
        )
    if before.value == 0:
        return

    disabled = ctypes.c_uint8(0)
    _call_cupti_attribute(
        set_attribute,
        value=disabled,
        operation="cuptiActivitySetAttribute",
    )

    after = ctypes.c_uint8()
    _call_cupti_attribute(
        get_attribute,
        value=after,
        operation="cuptiActivityGetAttribute after initialization",
    )
    if after.value != 0:
        raise RuntimeError(
            "Torch profiler CUDA compatibility initialization did not disable "
            f"per-thread activity buffers. CUPTI reported {after.value}."
        )


def prepare_torch_cuda_profiler(*, cuda_profile_requested: bool) -> None:
    """Prepare the affected Torch 2.9 and CUDA 13 profiler before capture.

    CUDA 13 defaults to per-thread CUPTI activity buffers. The Kineto revision
    bundled with Torch 2.9 changes that attribute before flushing its first
    trace, which can produce an empty CUDA trace. Kineto fixed the ordering in
    https://github.com/pytorch/kineto/pull/1115.

    NVIDIA requires this attribute to be configured before CUPTI registers
    buffer callbacks or enables activity kinds. Call this function immediately
    before the first repository-owned CUDA ``torch.profiler.profile`` context.
    Remove the workaround when the B200 baseline no longer uses Torch 2.9.
    """

    if not cuda_profile_requested:
        return
    if not _requires_cupti_buffer_initialization(torch.__version__, torch.version.cuda):
        return
    if not torch.cuda.is_available():
        return

    global _prepared_pid
    current_pid = os.getpid()
    with _PREPARE_LOCK:
        if _prepared_pid == current_pid:
            return
        if torch.autograd._profiler_enabled():
            raise RuntimeError(
                "Torch 2.9 with CUDA 13 requires CUPTI activity-buffer "
                "initialization before the first CUDA profiler starts, but a "
                "PyTorch profiler is already active on this thread."
            )
        try:
            cupti = ctypes.CDLL("libcupti.so.13")
        except OSError as exc:
            raise RuntimeError(
                "Torch 2.9 with CUDA 13 requires CUPTI activity-buffer "
                "initialization before CUDA profiling, but libcupti.so.13 "
                "could not be loaded."
            ) from exc
        _initialize_cupti_per_thread_buffer(cupti)
        _prepared_pid = current_pid
