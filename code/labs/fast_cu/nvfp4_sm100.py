"""B200 K64 port of fast.cu's NVFP4 GEMM."""

from __future__ import annotations

from pathlib import Path

import torch

from labs.fast_cu.build import load_cuda_extension
from labs.fast_cu.nvfp4 import FastCuNvfp4Benchmark, Nvfp4Workload, _version_tuple
from labs.fast_cu.source import UPSTREAM_DIR

CUDA_VERSION = (13, 0)
COMPUTE_CAPABILITY = (10, 0)
ORIGIN_RUNG = 5
_LAB_DIR = Path(__file__).parent


def require_sm100_runtime() -> int:
    """Require B200 and CUDA 13.0 or newer."""
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: fast.cu SM100 NVFP4 requires a B200 CUDA GPU")
    runtime = _version_tuple(torch.version.cuda)
    if runtime is None or runtime < CUDA_VERSION:
        raise RuntimeError(
            "SKIPPED: fast.cu SM100 NVFP4 requires CUDA runtime 13.0 or newer. "
            f"Found {torch.version.cuda}"
        )
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability != COMPUTE_CAPABILITY:
        raise RuntimeError(
            "SKIPPED: fast.cu SM100 NVFP4 requires B200 (SM100). "
            f"Found SM{capability[0]}{capability[1]}"
        )
    return device


def load_sm100_extension():
    """Build the K64 port with explicit SM100 flags."""
    require_sm100_runtime()
    return load_cuda_extension(
        "fast_cu_nvfp4_sm100",
        [_LAB_DIR / "nvfp4_native.cu"],
        dependencies=[_LAB_DIR / "nvfp4_sm100.cuh"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-DFAST_CU_NVFP4_SM100=1",
            f"-DFAST_CU_NVFP4_RUNG={ORIGIN_RUNG}",
            f"-I{_LAB_DIR}",
            f"-I{UPSTREAM_DIR / 'gb300' / 'nvfp4'}",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcublasLt", "-lcuda"],
        minimum_cuda=CUDA_VERSION,
    )


class FastCuNvfp4Sm100Benchmark(FastCuNvfp4Benchmark):
    """Compare the complete K64 GEMM with cuBLASLt on identical NVFP4 inputs."""

    def __init__(
        self, *, optimized: bool, workload: Nvfp4Workload | None = None
    ) -> None:
        super().__init__(optimized=optimized, workload=workload, rung=ORIGIN_RUNG)

    def _load_native_module(self):
        return load_sm100_extension()
