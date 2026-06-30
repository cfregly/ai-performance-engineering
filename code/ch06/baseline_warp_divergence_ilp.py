"""baseline_warp_divergence_ilp.py - Baseline ILP with warp divergence."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


class BaselineWarpDivergenceILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: ILP limited by warp divergence."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.warp_elements
        self.branch_iterations = self.workload.warp_branch_iterations
        self._branch_iteration_range = range(self.branch_iterations)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.branch_iterations),
            tokens_per_iteration=float(self.N * self.branch_iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None  # Will be set by benchmark_fn
        self._output_buffer = torch.empty_like(self.input)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._branch_iteration_range = range(self.branch_iterations)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: ILP operations with warp divergence."""
        assert self.input is not None and self.routing_logits is not None and self._output_buffer is not None
        with torch.inference_mode(), self._nvtx_range("baseline_warp_divergence_ilp"):
            mask_source = self.routing_logits
            result = self._output_buffer
            result.copy_(self.input)
            for iteration in self._branch_iteration_range:
                activations = torch.sigmoid(mask_source)
                mask = activations > 0.5

                positive = result[mask]
                negative = result[~mask]

                positive = torch.tanh(positive * 1.11 + 0.25)
                positive = positive * 1.003 + 0.0005 * positive * positive

                negative = torch.sin(negative * 0.77 - 0.35)
                negative = negative * 0.997 - 0.0004 * negative * negative

                result[mask] = positive
                result[~mask] = negative

                mask_source = 0.92 * mask_source + 0.08 * torch.roll(result, shifts=iteration + 1, dims=0)

            self.output = result
            self.routing_logits = mask_source

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input, "routing_logits": self.routing_logits},
            output=self.output.detach(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-5, 1e-5),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self._output_buffer = None
        self.routing_logits = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
            adaptive_iterations=False,
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
    """Factory function for harness discovery."""
    return BaselineWarpDivergenceILPBenchmark()
