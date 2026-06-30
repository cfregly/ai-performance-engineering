"""baseline_precisionmixed.py - FP32 baseline for mixed-precision comparison."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Deeper MLP to stress GEMM throughput."""

    def __init__(self, hidden_dim: int = 1024, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(depth)
        ])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_proj(x))
        for layer in self.hidden:
            x = self.act(layer(x))
        x = self.out_proj(x)
        return x


class BaselinePrecisionMixedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP32 baseline used to compare against mixed-precision autocast."""

    signature_equivalence_group = "ch13_precisionmixed_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.batch_size = 512
        # Retuned upward so the FP32-vs-BF16 training comparison spends more
        # time in GEMM-heavy math where mixed precision should pay off.
        self.hidden_dim = 3072
        self.micro_steps = 4
        self._micro_step_range = range(self.micro_steps)
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_steps),
            tokens_per_iteration=float(tokens * self.micro_steps),
        )
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.micro_steps),
            tokens_per_iteration=float(tokens * self.micro_steps),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn_like(self.inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self._micro_step_range = range(self.micro_steps)
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        
        for _ in range(3):
            self.optimizer.zero_grad()
            _ = self.model(self.inputs)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - FP32 training."""
        if (
            self.model is None
            or self.inputs is None
            or self.targets is None
            or self.optimizer is None
            or self.criterion is None
        ):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_precision_mixed"):
            for _ in self._micro_step_range:
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
                loss.backward()
                self.optimizer.step()
            self.output = outputs.detach_()
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("Verification input/output not initialized")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.output = None
        self._verify_input = None
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

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

def get_benchmark() -> BaselinePrecisionMixedBenchmark:
    """Factory function for harness discovery."""
    return BaselinePrecisionMixedBenchmark()
