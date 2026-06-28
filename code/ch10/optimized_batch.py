"""optimized_batch.py - Optimized large batch size in GEMM context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch10.workload_config import WORKLOAD


class BufferedBatchMlp(nn.Module):
    """Two-layer MLP that reuses forward buffers in inference mode."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self._fc1_buffer: torch.Tensor | None = None
        self._fc2_buffer: torch.Tensor | None = None

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        fc1_shape = (rows, self.fc1.out_features)
        fc2_shape = (rows, self.fc2.out_features)
        if (
            self._fc1_buffer is None
            or self._fc1_buffer.shape != fc1_shape
            or self._fc1_buffer.device != x.device
            or self._fc1_buffer.dtype != x.dtype
        ):
            self._fc1_buffer = torch.empty(fc1_shape, device=x.device, dtype=x.dtype)
        if (
            self._fc2_buffer is None
            or self._fc2_buffer.shape != fc2_shape
            or self._fc2_buffer.device != x.device
            or self._fc2_buffer.dtype != x.dtype
        ):
            self._fc2_buffer = torch.empty(fc2_shape, device=x.device, dtype=x.dtype)
        return self._fc1_buffer, self._fc2_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        torch.mm(x, self.fc1.weight.t(), out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.mm(fc1_out, self.fc2.weight.t(), out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out


class OptimizedBatchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: large batch size to maximize GPU utilization.
    
    Processes all data in a single forward pass, achieving better
    GPU utilization through larger matrix operations.
    """
    
    def __init__(self):
        super().__init__()
        self.model: BufferedBatchMlp | None = None
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
        self.model = BufferedBatchMlp(self.hidden_dim, self.ffn_dim).to(self.device).eval()
        
        # Generate input (same shape/order as baseline for verification)
        self.input = torch.randn(self.total_batch_size, self.hidden_dim, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with optimized batch size (single kernel launch)."""
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("batch_optimized"):
            with torch.inference_mode():
                # Single large forward pass with scratch buffers reused between iterations.
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
