"""baseline_bf16_mlp.py - FP32 MLP baseline for the BF16 precision comparison.

Chapter 20: AI-Assisted Performance Optimizations

Baseline demonstrates common anti-patterns that AI optimization can fix:
1. FP32 instead of FP16/BF16 (no tensor core acceleration)
2. Redundant computation (no caching of repeated values)
3. Synchronous execution (no async/pipelining)
4. Multiple kernel launches instead of fused operations
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class UnoptimizedModel(nn.Module):
    """Model with intentional inefficiencies that AI optimization can fix."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x):
        # Anti-pattern 1: Separate operations that could be fused
        x = self.fc1(x)
        x = torch.relu_(x)  # Separate kernel
        x = self.fc2(x)
        x = torch.relu_(x)  # Another separate kernel
        x = self.fc3(x)
        # Anti-pattern 2: Redundant normalization
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x


class BaselineBF16MLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP32 eager MLP used for the BF16 precision comparison."""

    signature_equivalence_group = "ch20_bf16_mlp_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._materialization_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.batch_size = 128
        self.hidden_dim = 2048  # Smaller to make differences more visible
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # FP32 - no tensor core acceleration
        self.model = UnoptimizedModel(hidden_dim=self.hidden_dim).to(self.device).float().eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.output = None
        self._materialization_buffer = torch.empty((), device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self.x, dtype=torch.float32)
        # Warmup
        for _ in range(5):
            with torch.inference_mode():
                _ = self.model(self.x)
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None
        assert self._materialization_buffer is not None
        with self._nvtx_range("multiple_techniques_baseline"):
            with torch.inference_mode():
                out = self.model(self.x)
                torch.sum(out, dim=(0, 1), out=self._materialization_buffer)
                self.output = out

    def capture_verification_payload(self) -> None:
        assert (
            self.model is not None
            and self.x is not None
            and self.output is not None
            and self._verify_output_buffer is not None
        )
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.5, 6.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.x = None
        self._materialization_buffer = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def get_workload_metadata(self):
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
        if self.x is None:
            return "Input tensor not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return BaselineBF16MLPBenchmark()
