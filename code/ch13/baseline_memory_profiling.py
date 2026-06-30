"""baseline_memory_profiling.py - Baseline memory profiling (baseline)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for memory profiling."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineMemoryProfilingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline memory profiling - tracks usage without optimization."""

    signature_equivalence_group = "ch13_memory_profiling_checkpointing"
    signature_equivalence_ignore_fields: tuple[str, ...] = ()
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
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
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
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
        with self._nvtx_range("baseline_memory_profiling"):
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
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "memory.peak_allocated_mb": float(self.peak_memory_mb),
            "memory.gradient_checkpointing": 0.0,
            "memory.cuda_graph_replay": 0.0,
            "memory.compute_dtype_fp32": 1.0,
        }

    def get_optimization_goal(self) -> str:
        """Memory profiling is a memory-reduction study, not a speed race."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineMemoryProfilingBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryProfilingBenchmark()
