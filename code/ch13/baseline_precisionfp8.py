"""baseline_precisionfp8.py - FP16 precision baseline (baseline).

Training with FP16 precision so the optimized variant isolates the additional
benefit of torchao FP8 kernels instead of conflating FP32->FP16 and FP16->FP8.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

resolve_device = partial(require_cuda_device, "CUDA required for ch13")


class SimpleModel(nn.Module):
    """Simple model for precision comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselinePrecisionFP8Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP16 precision baseline for the FP8 training comparison."""

    signature_equivalence_group = "ch13_precisionfp8_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.targets = None
        self.inputs_fp16 = None
        self.targets_fp16 = None
        self.optimizer = None
        self.criterion = None
        self.output = None  # For output verification
        self.batch_size = 8192
        self.hidden_dim = 8192
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_input_fp16: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize FP16 model and data."""
        # Harness provides seeding - creation order must match optimized
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self._verify_input = self.inputs.detach().clone()
        self._verify_input_fp16 = self._verify_input.to(torch.float16)
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.hidden_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.inputs_fp16 = self.inputs.to(torch.float16)
        self.targets_fp16 = self.targets.to(torch.float16)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup (will modify model weights, but output already saved)
        for _ in range(5):
            self._train_step()
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _train_step(self) -> None:
        assert self.model is not None
        assert self.inputs_fp16 is not None and self.targets_fp16 is not None
        assert self.optimizer is not None and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(self.inputs_fp16)
        loss = self.criterion(outputs, self.targets_fp16)
        loss.backward()
        self.optimizer.step()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP16 precision."""
        if (
            self.model is None
            or self.optimizer is None
            or self.criterion is None
            or self._verify_input is None
            or self._verify_input_fp16 is None
        ):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_precisionfp8"):
            self._train_step()
            self.output = None

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self._verify_input_fp16 is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        with torch.inference_mode():
            verify_out = self.model(self._verify_input_fp16)
            output_slice = verify_out[
                : self._verify_output_buffer.shape[0],
                : self._verify_output_buffer.shape[1],
            ]
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.inputs_fp16, self.targets_fp16, self.optimizer, self.criterion
        self.output = None
        self._verify_input = None
        self._verify_input_fp16 = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            backend_policy="fp32_strict",
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePrecisionFP8Benchmark()
