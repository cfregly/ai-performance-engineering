"""optimized_memory_profiling.py - Optimized memory profiling (optimized)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedModel(nn.Module):
    """Model with gradient checkpointing for memory optimization."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = checkpoint(
            self._fc1_relu,
            x,
            preserve_rng_state=False,
            use_reentrant=False,
        )
        x = self.fc2(x)
        return x
    
    def _fc1_relu(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc1(x))


class OptimizedMemoryProfilingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized memory profiling - uses gradient checkpointing only."""

    signature_equivalence_group = "ch13_memory_profiling_checkpointing"
    signature_equivalence_ignore_fields: tuple[str, ...] = ()
    
    def __init__(self):
        super().__init__()
        self.model: Optional[OptimizedModel] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.criterion: Optional[nn.Module] = None
        self.peak_memory_mb = 0.0
        self._memory_bytes_to_mb = 1.0 / (1024 ** 2)
        self.batch_size = 32
        self.hidden_dim = 2048
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()

        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device)
        self.model.train()
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.criterion = nn.MSELoss()
        self._verify_output_buffer = torch.empty_like(self.inputs, dtype=torch.float32)

        _ = self.model(self.inputs)
        self._synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None or self.targets is None or self.criterion is None:
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("optimized_memory_profiling"):
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.output = outputs.detach_()
        if self.inputs is None or self.targets is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.inputs, "targets": self.targets},
            output=self._verify_output_buffer,
            batch_size=self.inputs.shape[0],
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.5, 5.0),
        )

    def finalize_iteration_metrics(self) -> Optional[dict]:
        """Poll allocator peak after harness timing has already finalized."""
        self.peak_memory_mb = torch.cuda.max_memory_allocated() * self._memory_bytes_to_mb
        return None

    def teardown(self) -> None:
        if torch.cuda.is_available():
            self.finalize_iteration_metrics()
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
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
        return {
            "memory.peak_allocated_mb": float(self.peak_memory_mb),
            "memory.gradient_checkpointing": 1.0,
            "memory.cuda_graph_replay": 0.0,
            "memory.compute_dtype_fp32": 1.0,
        }

    def get_optimization_goal(self) -> str:
        """Gradient checkpointing is measured here as a memory optimization."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedMemoryProfilingBenchmark:
    return OptimizedMemoryProfilingBenchmark()
