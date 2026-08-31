"""CUDA thread-scope asynchronous copy using 4-byte cuda::memcpy_async.

This is not Tensor Memory Accelerator (TMA) code. Depending on the compiler and
GPU it lowers to cp.async/LDGSTS or ordinary loads/stores. Legacy module/API names
remain for compatibility; no native-TMA capability or performance is claimed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_EXT_NAME = "persistent_decode_thread_async_copy_ext"


def _require_async_copy_hardware() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: Thread-scope CUDA copy path requires a CUDA GPU.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        raise RuntimeError(f"SKIPPED: Thread-scope CUDA copy extension is built for sm_100/103/120/121 (got sm_{major}{minor}).")


def _try_build_extension() -> Optional[object]:
    cpp_src = r"""
#include <torch/extension.h>

void tma_copy(torch::Tensor src, torch::Tensor dst);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("async_copy", &tma_copy, "Thread-scope async copy (CUDA)");
  m.def("tma_copy", &tma_copy, "Legacy alias: thread-scope copy, not TMA");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <cstdint>
#include <cuda/pipeline>

namespace {

__global__ void tma_copy_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                int n) {
    extern __shared__ float smem[];
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in_bounds = global_idx < n;

    if (in_bounds) {
        pipe.producer_acquire();
        cuda::memcpy_async(&smem[threadIdx.x], &src[global_idx], sizeof(float), pipe);
        pipe.producer_commit();
        pipe.consumer_wait();
    }
    __syncthreads();

    if (in_bounds) {
        dst[global_idx] = smem[threadIdx.x];
        pipe.consumer_release();
    }
}

} // namespace

void tma_copy(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.scalar_type() == torch::kFloat, "src must be float32");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat, "dst must be float32");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "size mismatch");
    TORCH_CHECK(src.device() == dst.device(), "src/dst must share a CUDA device");
    TORCH_CHECK(src.numel() <= std::numeric_limits<int>::max(), "copy size exceeds kernel index range");
    if (src.numel() == 0 || src.data_ptr() == dst.data_ptr()) return;
    const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
    const auto dst_begin = reinterpret_cast<uintptr_t>(dst.data_ptr());
    const auto bytes = src.numel() * sizeof(float);
    TORCH_CHECK(src_begin + bytes <= dst_begin || dst_begin + bytes <= src_begin,
                "source and destination must not partially overlap");

    const int n = static_cast<int>(src.numel());
    const int threads = 128;
    const int blocks = ((n - 1) / threads) + 1;

    c10::cuda::CUDAGuard guard(src.get_device());
    // Use the CURRENT stream, not getDefaultCUDAStream(): an explicit-but-default
    // stream is invisible to side-stream CUDA graph capture exactly like a raw
    // <<<>>> legacy launch (B45/B62/B65 silent-drop class). Eager behavior is
    // identical (current == default outside capture).
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    tma_copy_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

    return load_inline(
        _EXT_NAME,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=[
            "--std=c++17",
            "--use_fast_math",
            "-lineinfo",
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_103,code=sm_103",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
        ],
        extra_include_paths=torch.utils.cpp_extension.include_paths(),
        verbose=False,
    )


_EXT_INSTANCE: Optional[object] = None


def load_async_copy() -> Optional[object]:
    """Return the thread-scope CUDA copy extension or raise on unsupported hardware."""
    global _EXT_INSTANCE
    if _EXT_INSTANCE is not None:
        return _EXT_INSTANCE
    _require_async_copy_hardware()
    try:
        _EXT_INSTANCE = _try_build_extension()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"SKIPPED: Thread-scope CUDA copy extension unavailable ({exc})") from exc
    return _EXT_INSTANCE


def load_native_tma() -> Optional[object]:
    """Compatibility alias for load_async_copy(); this does not use the TMA engine."""
    return load_async_copy()
