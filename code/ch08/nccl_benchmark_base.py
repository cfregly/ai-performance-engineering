"""Shared base for NCCL-style reduction benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension


class NcclBenchmarkBase(BaseBenchmark):
    world_size: int = 4
    chunk_elems: int = 1 << 15
    nvtx_label: str = "nccl"

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for NCCL benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.device_chunks: Optional[torch.Tensor] = None
        self.host_chunks: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_nccl_kernels",
            cuda_source_file=str(Path(__file__).with_name("nccl_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        self.device_chunks = torch.randn(
            self.world_size,
            self.chunk_elems,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        self.host_chunks = self.device_chunks.cpu().pin_memory()
        self.output = torch.empty(self.chunk_elems, device=self.device, dtype=torch.float32)

        self._invoke_kernel()
        torch.cuda.synchronize()
        self._validate_correctness()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            self._invoke_kernel()

    def teardown(self) -> None:
        self.device_chunks = None
        self.host_chunks = None
        self.output = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _validate_correctness(self) -> None:
        assert self.device_chunks is not None
        assert self.output is not None
        reference = self.device_chunks.sum(dim=0)
        torch.cuda.synchronize()
        max_error = scalar_tensor_to_float(torch.max(torch.abs(reference - self.output)))
        if max_error > 1e-3:
            raise RuntimeError(f"NCCL reduction validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NCCL reduction benchmark metrics."""
        total_elements = self.world_size * self.chunk_elems
        bytes_transferred = float(total_elements * 4)  # float32
        return {
            f"{self.nvtx_label}.world_size": float(self.world_size),
            f"{self.nvtx_label}.chunk_elems": float(self.chunk_elems),
            f"{self.nvtx_label}.total_elements": float(total_elements),
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
        }
