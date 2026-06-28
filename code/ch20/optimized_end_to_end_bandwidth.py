"""optimized_end_to_end_bandwidth.py - Flattened end-to-end bandwidth path."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimplePipeline(nn.Module):
    """Simple inference pipeline for bandwidth analysis."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None

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
            x = self.fc2(x)
            return x
        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        torch.mm(x, self.fc1.weight.t(), out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.mm(fc1_out, self.fc2.weight.t(), out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out


class OptimizedEndToEndBandwidthBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized end-to-end bandwidth by flattening the batch stream."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.stacked_inputs: Optional[torch.Tensor] = None
        self.flat_inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
        tokens = self.batch_size * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(tokens),
            tokens_per_iteration=float(tokens),
        )
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.float32).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32).contiguous()
            for _ in range(self.num_batches)
        ]
        self.stacked_inputs = torch.stack(self.inputs, dim=0)
        self.flat_inputs = self.stacked_inputs.view(self.num_batches * self.batch_size, self.hidden_dim).contiguous()
        self.output = None
        with torch.inference_mode():
            _ = self.model(self.flat_inputs)
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.flat_inputs is not None
        with self._nvtx_range("optimized_end_to_end_bandwidth"):
            with torch.inference_mode():
                flat_output = self.model(self.flat_inputs)
                self.output = flat_output.view(self.num_batches, self.batch_size, self.hidden_dim)

    def capture_verification_payload(self) -> None:
        if self.model is None or self.stacked_inputs is None:
            raise RuntimeError("capture_verification_payload() requires completed benchmark run")
        self._set_verification_payload(
            inputs={"inputs": self.stacked_inputs.detach()},
            output=self.output.detach().clone(),
            batch_size=int(self.stacked_inputs.shape[0]),
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.stacked_inputs = None
        self.flat_inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=None,
            ai_optimized_time_ms=getattr(self, '_last_elapsed_ms', None),
            suggestions_applied=None,
            suggestions_total=None,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.output is None:
            return "Output not initialized"
        if tuple(self.output.shape) != (self.num_batches, self.batch_size, self.hidden_dim):
            return f"Output shape mismatch: expected {(self.num_batches, self.batch_size, self.hidden_dim)}, got {tuple(self.output.shape)}"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return OptimizedEndToEndBandwidthBenchmark()
