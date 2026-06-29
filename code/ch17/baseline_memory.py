"""baseline_memory.py - Baseline standard GPU memory allocation."""

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


class BaselineMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Standard GPU memory allocation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.repetitions = REPETITIONS
        self.host_batches: list[torch.Tensor] = []
        tokens = self.batch_size * self.input_dim * self.repetitions
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._last_input: Optional[torch.Tensor] = None
        self._host_float_buffer: Optional[torch.Tensor] = None
        self._device_batch_buffer: Optional[torch.Tensor] = None
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
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.host_batches = [
            torch.randint(
                0,
                256,
                (self.batch_size, self.input_dim),
                device="cpu",
                dtype=torch.uint8,
            )
            for _ in range(self.repetitions)
        ]
        self._host_float_buffer = torch.empty(
            (self.batch_size, self.input_dim),
            device="cpu",
            dtype=torch.float32,
        )
        self._device_batch_buffer = torch.empty(
            (self.batch_size, self.input_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Benchmark not configured")
        if self._host_float_buffer is None or self._device_batch_buffer is None:
            raise RuntimeError("Benchmark buffers not configured")
        with self._nvtx_range("baseline_memory"):
            with torch.inference_mode():
                host_batch = self._host_float_buffer
                device_batch = self._device_batch_buffer
                for compressed in self.host_batches:
                    host_batch.copy_(compressed)
                    host_batch.mul_(1.0 / 255.0)
                    host_batch.add_(-0.5)
                    host_batch.mul_(2.0)
                    host_batch.tanh_()
                    device_batch.copy_(host_batch, non_blocking=False)
                    self._last_input = device_batch
                    self.output = self.model(device_batch)
        if self.output is None or self._last_input is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._last_input},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.host_batches = []
        self._host_float_buffer = None
        self._device_batch_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
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


def get_benchmark() -> BaselineMemoryBenchmark:
    return BaselineMemoryBenchmark()
