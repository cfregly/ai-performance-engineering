"""optimized_batch.py - Optimized large batch size in GEMM context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch10.workload_config import WORKLOAD


class OptimizedBatchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: large batch size to maximize GPU utilization.
    
    Processes all data in a single forward pass, achieving better
    GPU utilization through larger matrix operations.
    """
    
    def __init__(self):
        super().__init__()
        self.model: nn.Sequential | None = None
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.workload = WORKLOAD
        self.total_batch_size = self.workload.optimized_batch_size  # 512
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        tokens = self.total_batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with optimized batch size."""
        # Harness provides seeding - model and input creation order must match baseline
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Generate input (same shape/order as baseline for verification)
        self.input = torch.randn(self.total_batch_size, self.hidden_dim, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with optimized batch size (single kernel launch)."""
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("batch_optimized"):
            with torch.inference_mode():
                # Single large forward pass (one kernel launch, better GPU utilization)
                self.output = self.model(self.input)
        if self.output is None or self.input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach().float().clone(),
            batch_size=self.total_batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.05, 0.05),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            timing_method="wall_clock",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual batching structure."""
        from ch10.benchmark_metrics_common import compute_batch_workload_metrics

        return compute_batch_workload_metrics(
            total_batch_size=self.total_batch_size,
            micro_batch_size=self.total_batch_size,
            micro_batches=1,
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> OptimizedBatchBenchmark:
    """Factory function for harness discovery."""
    return OptimizedBatchBenchmark()
