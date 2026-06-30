"""baseline_autograd_standard.py - Standard autograd baseline (baseline)."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineAutogradStandardBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Standard autograd - no compilation."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        # Smaller batch to increase launch overhead share and highlight graph/optimized paths.
        self.batch_size = 16
        self.hidden_dim = 1024
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and data."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.output_buffer = torch.empty_like(self.inputs)
        self._verify_output_buffer = torch.empty_like(self.inputs, dtype=torch.float32)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        saved_model_state = copy.deepcopy(self.model.state_dict())
        saved_opt_state = copy.deepcopy(self.optimizer.state_dict())

        for _ in range(3):
            self._train_step(self.inputs, self.targets)
        self._synchronize()
        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)

    def _train_step(self, batch: torch.Tensor, target: torch.Tensor, capture_output: bool = False) -> None:
        assert self.model is not None and self.optimizer is not None and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=False)
        outputs = self.model(batch)
        if capture_output and self.output_buffer is not None:
            self.output_buffer.copy_(outputs)
            self.output = self.output_buffer
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - standard autograd."""
        if (
            self.model is None
            or self.inputs is None
            or self.targets is None
            or self.optimizer is None
            or self.criterion is None
        ):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_autograd_standard"):
            self._train_step(self.inputs, self.targets, capture_output=True)
        if self.inputs is None or self.targets is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.inputs, "targets": self.targets},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.output = None
        self.output_buffer = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()


def get_benchmark() -> BaselineAutogradStandardBenchmark:
    """Factory function for harness discovery."""
    return BaselineAutogradStandardBenchmark()
