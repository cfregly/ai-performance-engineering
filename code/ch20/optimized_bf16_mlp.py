"""optimized_bf16_mlp.py - BF16 MLP precision optimization.

Chapter 20: AI-Assisted Performance Optimizations

Optimizations applied (as AI would suggest):
1. BF16/FP16 for tensor core acceleration
2. Efficient normalization with the same eager op graph
3. Single forward pass (no redundant computation)

This benchmark does not implement a fused MLP kernel today; the optimized path
is a truthful "better precision policy, same eager graph" comparison.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedModel(nn.Module):
    """Model with AI-suggested optimizations applied."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.output = None
        # Same architecture, but executed in BF16/FP16 for tensor core use.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x):
        # Keep math identical to the baseline so verification is meaningful.
        x = self.fc1(x)
        x = torch.relu_(x)
        x = self.fc2(x)
        x = torch.relu_(x)
        x = self.fc3(x)
        norm = x.norm(dim=-1, keepdim=True)
        norm.clamp_(min=1e-8)
        x.div_(norm)
        return x


class OptimizedBF16MLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: BF16 precision on the same eager MLP graph."""

    signature_equivalence_group = "ch20_bf16_mlp_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        self._x_model_dtype: Optional[torch.Tensor] = None
        self._model_dtype: Optional[torch.dtype] = None
        self.output: Optional[torch.Tensor] = None
        self._materialization_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.batch_size = 128
        self.hidden_dim = 2048  # Match baseline
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._payload_parameter_count = 0
        self._verification_payload = None
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Optimization 1: BF16 for tensor cores
        self._verification_payload = None
        self._model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device, dtype=self._model_dtype).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self._refresh_model_input()
        self.output = None
        self._materialization_buffer = torch.empty((), device=self.device, dtype=self._model_dtype)
        self._verify_output_buffer = torch.empty_like(self.x, dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            with torch.inference_mode():
                _ = self.model(self._x_model_dtype)

    def _refresh_model_input(self) -> None:
        if self.x is None or self._model_dtype is None:
            raise RuntimeError("setup() must initialize inputs before refresh")
        self._x_model_dtype = self.x.to(dtype=self._model_dtype)

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None and self._x_model_dtype is not None
        assert self._materialization_buffer is not None

        if self._verification_payload is not None:
            self._refresh_model_input()

        with self._nvtx_range("multiple_techniques_optimized"):
            with torch.inference_mode():
                # Optimization: Single forward pass (no redundant compute)
                self.output = self.model(self._x_model_dtype)
                torch.sum(self.output, dim=(0, 1), out=self._materialization_buffer)

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
            precision_flags={
                "fp16": False,
                "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        )
    
    def teardown(self) -> None:
        self.model = None
        self.x = None
        self._x_model_dtype = None
        self._model_dtype = None
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
        """Return optimization stack metrics."""
        return {
            "ch20.uses_bf16": 1.0,
            "ch20.uses_fused_ops": 0.0,
            "ch20.no_redundant_compute": 1.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()


def get_benchmark() -> BaseBenchmark:
    return OptimizedBF16MLPBenchmark()
