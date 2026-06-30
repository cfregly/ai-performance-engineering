"""optimized_pageable_copy.py - Pinned async host-transfer optimization.

This benchmark demonstrates efficient memory transfer - using pinned memory
with async copies on the measured CUDA stream.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


class OptimizedPageableCopyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pinned async variant for the pageable-copy benchmark pair.

    Actual CPU/GPU NUMA binding helpers live elsewhere in the chapter. This
    pair isolates the pageable-vs-pinned transfer behavior.
    """

    def __init__(self):
        super().__init__()
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        # Memory copy benchmark - jitter check not applicable
        bytes_per_iter = 128_000_000 * 4  # float32 bytes (same as baseline)
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Pinned memory is the optimization: it allows the H2D copy to remain
        # non-blocking while still being fully measured on the benchmark stream.
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float32, pin_memory=True)
        self.device_buffer = torch.empty_like(self.host_tensor, device=self.device)
        self._output_buffer = torch.empty(1, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Copy data and compute sum using a pinned async H2D transfer."""
        assert (
            self.host_tensor is not None
            and self.device_buffer is not None
            and self._output_buffer is not None
        )
        with torch.inference_mode(), self._nvtx_range("optimized_pageable_copy"):
            self.device_buffer.copy_(self.host_tensor, non_blocking=True)
            torch.sum(self.device_buffer, dim=0, keepdim=True, out=self._output_buffer)
            self.output = self._output_buffer
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "host_tensor": self.host_tensor,
                "device_buffer": self.device_buffer,
            },
            output=self._verify_output_buffer,
            batch_size=self.host_tensor.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffer = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPageableCopyBenchmark()
