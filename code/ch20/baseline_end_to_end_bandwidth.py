"""baseline_end_to_end_bandwidth.py - Baseline end-to-end bandwidth (sequential)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimplePipeline(nn.Module):
    """Simple inference pipeline."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineEndToEndBandwidthBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline end-to-end bandwidth - sequential processing."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self.stacked_inputs: Optional[torch.Tensor] = None
        self.outputs: Optional[list[torch.Tensor | None]] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self.num_batches = 10
        tokens = self.batch_size * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.num_batches),
            tokens_per_iteration=float(tokens),
        )
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = SimplePipeline(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.float32).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.stacked_inputs = torch.empty(
            self.num_batches,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.inputs = list(self.stacked_inputs.unbind(0))
        for batch_input in self.inputs:
            batch_input.normal_()
        self._output_buffer = torch.empty_like(self.stacked_inputs)
        self.outputs = [None for _ in range(self.num_batches)]
        self.output = None
        for inp in self.inputs[:3]:
            _ = self.model(inp)
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None and self.outputs is not None
        with self._nvtx_range("baseline_end_to_end_bandwidth"):
            with torch.inference_mode():
                for batch_idx, inp in enumerate(self.inputs):
                    out = self.model(inp)
                    self.outputs[batch_idx] = out

    def capture_verification_payload(self) -> None:
        if (
            self.model is None
            or self.stacked_inputs is None
            or self.outputs is None
            or self._output_buffer is None
            or any(out is None for out in self.outputs)
        ):
            raise RuntimeError("capture_verification_payload() requires completed benchmark run")
        outputs = [out for out in self.outputs if out is not None]
        torch.stack(outputs, dim=0, out=self._output_buffer)
        self.output = self._output_buffer.detach()
        self._set_verification_payload(
            inputs={"inputs": self.stacked_inputs},
            output=self.output,
            batch_size=int(self.stacked_inputs.shape[0]),
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.stacked_inputs = None
        self.outputs = None
        self._output_buffer = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_last_elapsed_ms', None),
            ai_optimized_time_ms=None,
            suggestions_applied=None,
            suggestions_total=None,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.outputs is None or len(self.outputs) != self.num_batches:
            return f"Expected {self.num_batches} outputs"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return BaselineEndToEndBandwidthBenchmark()
