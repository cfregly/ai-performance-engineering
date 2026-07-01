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
        self._fc1_forward_view: torch.Tensor | None = None
        self._fc2_forward_view: torch.Tensor | None = None
        self._fc1_weight_t: torch.Tensor | None = None
        self._fc2_weight_t: torch.Tensor | None = None

    def cache_weight_views(self) -> None:
        self._fc1_weight_t = self.fc1.weight.t()
        self._fc2_weight_t = self.fc2.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, width = (int(shape[0]), int(shape[1]))
        numel = rows * width
        buffer = getattr(self, name)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, buffer)
        return buffer[:numel].view(rows, width)

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        fc1_shape = (rows, self.fc1.out_features)
        fc2_shape = (rows, self.fc2.out_features)
        fc1_out = self._workspace("_fc1_buffer", fc1_shape, device=x.device, dtype=x.dtype)
        fc2_out = self._workspace("_fc2_buffer", fc2_shape, device=x.device, dtype=x.dtype)
        return fc1_out, fc2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def prepare_forward_buffers(self, x: torch.Tensor) -> None:
        self._fc1_forward_view, self._fc2_forward_view = self._ensure_forward_buffers(x)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.forward(x)
        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out = self._fc1_forward_view
        fc2_out = self._fc2_forward_view
        if fc1_out is None or fc2_out is None:
            raise RuntimeError("forward_prepared() requires prepare_forward_buffers()")
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def _forward_into_buffers(
        self,
        x: torch.Tensor,
        fc1_out: torch.Tensor,
        fc2_out: torch.Tensor,
    ) -> torch.Tensor:
        torch.mm(x, self._fc1_weight_t, out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.mm(fc1_out, self._fc2_weight_t, out=fc2_out)
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
        self._verify_output_buffer: torch.Tensor | None = None
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
        self.model.cache_weight_views()
        
        # Generate input (same shape/order as baseline for verification)
        self.input = torch.randn(self.total_batch_size, self.hidden_dim, device=self.device)
        self.model.prepare_forward_buffers(self.input)
        self._verify_output_buffer = torch.empty_like(self.input)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with optimized batch size (single kernel launch)."""
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("batch_optimized"):
            with torch.inference_mode():
                # Single large forward pass with scratch buffers reused between iterations.
                self.output = self.model.forward_prepared(self.input)
        if self.output is None or self.input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
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
        self._verify_output_buffer = None
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
