"""optimized_bandwidth_naive.py - Coalesced bandwidth optimization (optimized).

Optimized memory access patterns with coalesced access.
Efficient bandwidth utilization through contiguous memory operations.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


from typing import Optional

from core.benchmark.verification import simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.utils.extension_loader_template import load_cuda_extension_v2


_EXT_NAME = "ch13_bandwidth_naive_ext"


@lru_cache(maxsize=1)
def _load_extension():
    return load_cuda_extension_v2(
        name=_EXT_NAME,
        sources=[Path(__file__).with_name("bandwidth_naive_extension.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


class OptimizedBandwidthCoalescedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized bandwidth usage - coalesced memory access."""
    
    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self._verify_indices: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self.rows = 4096
        self.cols = 4096
        self.size = self.rows * self.cols  # Same size for fair comparison
        self.passes = 16
        self.stride = 1
        bytes_per_iter = float(self.size * 4 * 3 * self.passes)  # read A/B, write C
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size * self.passes),
            bytes_per_iteration=bytes_per_iter,
        )
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Large tensors for bandwidth measurement
        self.A = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty_like(self.A)
        verify_count = min(4096, self.size)
        verify_step = max(self.size // verify_count, 1)
        self._verify_indices = torch.arange(
            0,
            verify_step * verify_count,
            verify_step,
            device=self.device,
            dtype=torch.long,
        )
        self._verify_output_buffer = torch.empty(verify_count, device=self.device, dtype=torch.float32)
        self.ext = _load_extension()
        
        # Warmup
        self.ext.bandwidth_add_mul(self.A, self.B, self.C, int(self.stride), 1)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized bandwidth usage."""
        assert self.A is not None and self.B is not None and self.C is not None
        with torch.inference_mode(), self._nvtx_range("optimized_bandwidth_coalesced"):
            self.ext.bandwidth_add_mul(self.A, self.B, self.C, int(self.stride), int(self.passes))

    def capture_verification_payload(self) -> None:
        if self.C is None or self._verify_indices is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        torch.index_select(self.C, 0, self._verify_indices, out=self._verify_output_buffer)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self._verify_output_buffer,
            batch_size=self.size,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        """Cleanup."""
        self.A = None
        self.B = None
        self.C = None
        self._verify_indices = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            timing_method="wall_clock",
        )

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.size,
            dtype="float32",
            rows=self.rows,
            cols=self.cols,
            passes=self.passes,
        ).to_dict()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp32",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedBandwidthCoalescedBenchmark()
