"""Optional native TMA-style copy path using a tiny CUDA extension.

This attempts to build a CUDA extension that issues hardware async copies
via the CUDA pipeline API. On Hopper/Blackwell parts with TMA enabled,
the compiler lowers these async copies to the TMA engine. On GPUs without
support, the extension is skipped and callers fall back to the Python
pseudo-TMA path.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_EXT_NAME = "persistent_decode_tma_ext"


def _require_tma_hardware() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: Native TMA path requires a CUDA GPU.")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"SKIPPED: Native TMA path requires Hopper/Blackwell-class GPUs (got sm_{major}{minor}).")


def _try_build_extension() -> Optional[object]:
    cpp_src = r"""
#include <torch/extension.h>

void tma_copy(torch::Tensor src, torch::Tensor dst);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tma_copy", &tma_copy, "TMA-style async copy (CUDA)");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
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

    const int n = static_cast<int>(src.numel());
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(src.get_device());
    // Use the CURRENT stream, not getDefaultCUDAStream(): an explicit-but-default
    // stream is invisible to side-stream CUDA graph capture exactly like a raw
    // <<<>>> legacy launch (B45/B62/B65 silent-drop class). Eager behavior is
    // identical (current == default outside capture).
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    tma_copy_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
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


def load_native_tma() -> Optional[object]:
    """Return the compiled native TMA extension or raise on unsupported hardware."""
    global _EXT_INSTANCE
    if _EXT_INSTANCE is not None:
        return _EXT_INSTANCE
    _require_tma_hardware()
    try:
        _EXT_INSTANCE = _try_build_extension()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"SKIPPED: Native TMA extension unavailable ({exc})") from exc
    return _EXT_INSTANCE
