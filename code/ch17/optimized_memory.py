"""optimized_memory.py - Optimized GPU memory management."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

BATCH_SIZE = 1024
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 10


class OptimizedMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: GPU memory management with capture and buffer reuse."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.device_buffer: Optional[torch.Tensor] = None
        self.transform_buffer: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None
        self.repetitions = REPETITIONS
        tokens = self.batch_size * self.input_dim * self.repetitions
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, self.input_dim),
        ).to(self.device, dtype=torch.float32).eval()
        
        self.device_buffer = torch.empty(
            self.batch_size,
            self.input_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.transform_buffer = torch.empty_like(self.device_buffer)
        self.graph_output = torch.empty_like(self.device_buffer)
        self._synchronize()

        with torch.no_grad():
            _ = self.model(self.device_buffer)
        self._synchronize()

        self.graph = torch.cuda.CUDAGraph()
        # Keep the float buffer on the same discrete 0..255 population as the
        # baseline's uint8 staging path. random_(low, high) already emits
        # integer-valued floats, so no extra floor kernel is needed.
        self.device_buffer.random_(0, 256)
        self._synchronize()
        with torch.cuda.graph(self.graph):
            self.transform_buffer.copy_(self.device_buffer)
            self.transform_buffer.mul_(1.0 / 255.0)
            self.transform_buffer.add_(-0.5)
            self.transform_buffer.mul_(2.0)
            self.transform_buffer.tanh_()
            self.graph_output.copy_(self.model(self.transform_buffer))
        self._synchronize()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
    
    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.device_buffer is None
            or self.graph is None
            or self.graph_output is None
        ):
            raise RuntimeError("Optimized memory benchmark not initialized")

        with self._nvtx_range("optimized_memory"):
            with torch.no_grad():
                for _ in range(self.repetitions):
                    # Make the discrete input population explicit on every replay.
                    self.device_buffer.random_(0, 256)
                    self.graph.replay()
                self.output = self.graph_output
        if self.output is None or self.device_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.device_buffer is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.device_buffer},
            output=self.output.detach().clone(),
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.device_buffer = None
        self.transform_buffer = None
        self.graph_output = None
        self.graph = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedMemoryBenchmark:
    return OptimizedMemoryBenchmark()
