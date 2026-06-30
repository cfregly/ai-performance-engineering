"""baseline_quantization_ilp.py - Baseline ILP without quantization."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


class BaselineQuantizationILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Full precision ILP (no quantization)."""

    signature_equivalence_group = "ch06_quantization_ilp_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.quantization_elements
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
        # ILP benchmark: fixed dimensions for measurement
    
    def setup(self) -> None:
        """Setup: Initialize full precision tensors."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._output_buffer = torch.empty_like(self.input)
        self.output = None
    
    def benchmark_fn(self) -> None:
        """Benchmark: Full precision ILP operations."""
        assert self.input is not None and self._output_buffer is not None
        with torch.inference_mode(), self._nvtx_range("baseline_quantization_ilp"):
            torch.mul(self.input, 2.0, out=self._output_buffer)
            self._output_buffer.add_(1.0)
            self.output = self._output_buffer

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach(),
            batch_size=self.N,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-2, 1e-2),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineQuantizationILPBenchmark()
