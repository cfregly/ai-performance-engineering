"""baseline_host_staged_reduction.py - Baseline reduction with full host staging."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineHostStagedReductionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Transfer the full tensor to the CPU before reducing it."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._host_buffer: Optional[torch.Tensor] = None
        self._host_sum: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.num_elements = 10_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.num_elements),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.randn(self.num_elements, device=self.device)
        self._host_buffer = self._make_host_buffer(self.data)
        self._host_sum = torch.empty((), dtype=self.data.dtype)
        self._output_buffer = torch.empty((), device=self.data.device, dtype=self.data.dtype)
        self._synchronize()

    def _make_host_buffer(self, data: torch.Tensor) -> torch.Tensor:
        use_pinned_host = data.device.type == "cuda" and torch.cuda.is_available()
        return torch.empty_like(data, device="cpu", pin_memory=use_pinned_host)

    def _host_buffer_for_data(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Data not initialized")
        if (
            self._host_buffer is None
            or self._host_buffer.shape != self.data.shape
            or self._host_buffer.dtype != self.data.dtype
        ):
            self._host_buffer = self._make_host_buffer(self.data)
        return self._host_buffer

    def _host_sum_for_data(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Data not initialized")
        if self._host_sum is None or self._host_sum.dtype != self.data.dtype:
            self._host_sum = torch.empty((), dtype=self.data.dtype)
        return self._host_sum

    def _output_for_data(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Data not initialized")
        if (
            self._output_buffer is None
            or self._output_buffer.device != self.data.device
            or self._output_buffer.dtype != self.data.dtype
        ):
            self._output_buffer = torch.empty((), device=self.data.device, dtype=self.data.dtype)
        return self._output_buffer

    def benchmark_fn(self) -> None:
        """Benchmark: Single-node operations."""
        assert self.data is not None
        host_buffer = self._host_buffer_for_data()
        host_sum = self._host_sum_for_data()
        output_buffer = self._output_for_data()
        with torch.inference_mode(), self._nvtx_range("baseline_host_staged_reduction"):
            host_buffer.copy_(self.data, non_blocking=False)
            torch.sum(host_buffer, dim=0, out=host_sum)
            output_buffer.copy_(host_sum, non_blocking=False)
        self.output = output_buffer

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self.output,
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self._host_buffer = None
        self._host_sum = None
        self._output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual host-staging behavior of the reduction path."""
        from ch05.metrics_common import compute_host_reduction_metrics

        return compute_host_reduction_metrics(
            num_elements=self.num_elements,
            host_staging_round_trips=2,
            keeps_reduction_on_device=False,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineHostStagedReductionBenchmark()
