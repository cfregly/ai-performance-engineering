"""baseline_memory_double_buffering.py - Single-stream baseline."""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class MemoryDoubleBufferingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: single stream, single buffer (no overlap)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
        self.host_batches: List[torch.Tensor] = []
        self.micro_batches = 16
        tokens = self.batch_size * self.seq_len * self.micro_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self._verification_payload = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        ).to(self.device).half().eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.buffer = torch.empty(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.host_batches = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device="cpu",
                dtype=torch.float16,
            ).pin_memory()
            for _ in range(self.micro_batches)
        ]
        self.stream = torch.cuda.Stream()

    def benchmark_fn(self) -> None:
        """Benchmark: Single-GPU stream-ordered operations."""
        assert (
            self.model is not None
            and self.buffer is not None
            and self.stream is not None
            and self.host_batches
        )
        with nvtx_range("baseline_memory_double_buffering", enable=self._enable_nvtx):
            with torch.inference_mode():
                for host_batch in self.host_batches:
                    self.buffer.copy_(host_batch, non_blocking=False)
                    with torch.cuda.stream(self.stream):
                        self.output = self.model(self.buffer)
                    torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        if self.output is None or self.buffer is None or self.model is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"buffer": self.buffer},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.buffer = None
        self.output = None
        self.stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return MemoryDoubleBufferingBenchmark()
