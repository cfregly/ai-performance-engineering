"""baseline_memory_bound.py - Memory-bound kernel (low arithmetic intensity)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineMemoryBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Simple element-wise ops with low arithmetic intensity."""

    def __init__(self):
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.repeats = 64
        self.N = 16_777_216  # ~64 MB
        self._repeat_range = range(self.repeats)
        # Memory-bound benchmark - fixed dimensions for roofline analysis
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty(4096, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range("baseline_memory_bound"):
            t = self.tensor
            for _ in self._repeat_range:
                t = t * 1.0001 + 0.0001
            self.output = t
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self.tensor is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        # Keep verification lightweight: slice the large output tensor.
        # This avoids serializing ~64MB outputs in subprocess mode while still
        # validating correctness on representative data.
        output_slice = self.output[: self._verify_output_buffer.numel()].detach()
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"tensor": self.tensor},
            output=self._verify_output_buffer,
            batch_size=self.tensor.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-1, 1e-1),
        )

    def teardown(self) -> None:
        self.tensor = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            timing_method="wall_clock",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp32",
        )

    def validate_result(self) -> Optional[str]:
        if self.tensor is None:
            return "Tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return BaselineMemoryBoundBenchmark()
