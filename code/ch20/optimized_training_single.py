"""optimized_training_single.py - Optimized single-GPU training loop.

Optimizations (Ch20):
- Enable TF32 matmul where appropriate
- Prefer fused AdamW when available
- Use set_to_none=True for zero_grad
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for training demonstration."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedTrainingSingleBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized single-GPU training loop with TF32-enabled matmul and fused optimizer."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.output: Optional[torch.Tensor] = None
        self._tf32_state: Optional[tuple[bool, bool]] = None
        self.batch_size = 32
        self.hidden_dim = 8192
        self.train_steps = 6
        self._train_step_range = range(self.train_steps)
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._tf32_state = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).float().train()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        except TypeError:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self._train_step_range = range(self.train_steps)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None and self.targets is not None
        assert self.optimizer is not None and self.criterion is not None
        with self._nvtx_range("training_optimized"):
            for _ in self._train_step_range:
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
                loss.backward()
                self.optimizer.step()

    def capture_verification_payload(self) -> None:
        if self.model is None or self.inputs is None or self.targets is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        output = self.model.fc1.weight[:16, :16].detach()
        self._set_verification_payload(
            inputs={"inputs": self.inputs, "targets": self.targets},
            output=output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(1e-2, 1e-2),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        if self._tf32_state is not None:
            matmul_tf32, cudnn_tf32 = self._tf32_state
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
        self._tf32_state = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self):
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
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedTrainingSingleBenchmark()
