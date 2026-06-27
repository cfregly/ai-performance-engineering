"""optimized_memory_double_buffering.py - Optimized memory management with double buffering.

Demonstrates double buffering (ping-pong) for overlapping memory operations.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

resolve_device = partial(require_cuda_device, "CUDA required for ch19")

class OptimizedMemoryDoubleBufferingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: double buffering for overlapping operations."""
    
    def __init__(self):
        super().__init__()
        self.output = None
        self.device = resolve_device()
        self.model = None

        self.buffer_a = None
        self.buffer_b = None
        self.buffers: list[torch.Tensor] = []
        self.copy_stream = None
        self.compute_stream = None
        self.copy_events: list[torch.cuda.Event] = []
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
        self.micro_batches = 16
        self.host_batches: list[torch.Tensor] = []
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=float(self.micro_batches))
    
    def setup(self) -> None:
        """Setup: Initialize model and double buffers."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        # Optimization: Use FP16 for faster computation - FAIL FAST if not supported
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for optimized_memory_double_buffering benchmark")
        self.model = self.model.to(self.device).half()
        self.model.eval()
        
        # Optimization: Double buffering (ping-pong buffers)
        # Two buffers allow overlapping copy and compute operations
        # Ensure buffer dtype matches model dtype - FAIL FAST if model has no parameters
        params = list(self.model.parameters())
        if not params:
            raise RuntimeError("Model has no parameters - cannot determine dtype")
        model_dtype = params[0].dtype
        self.buffer_a = torch.empty(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=model_dtype
        )
        self.buffer_b = torch.empty_like(self.buffer_a)
        self.buffers = [self.buffer_a, self.buffer_b]
        self.host_batches = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device="cpu",
                dtype=model_dtype,
            ).pin_memory()
            for _ in range(self.micro_batches)
        ]
        
        # Create streams for overlapping operations
        self.copy_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.copy_events = [torch.cuda.Event(blocking=False) for _ in range(2)]
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapping operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.copy_stream is not None and self.compute_stream is not None
        with nvtx_range("optimized_memory_double_buffering", enable=enable_nvtx):
            with torch.inference_mode():
                buffers = self.buffers
                copy_events = self.copy_events
                if len(buffers) != 2 or len(copy_events) < len(buffers):
                    raise RuntimeError("Double buffers or copy events not initialized")

                # Preload first buffer on the copy stream and signal completion.
                with torch.cuda.stream(self.copy_stream):
                    buffers[0].copy_(self.host_batches[0], non_blocking=True)
                    copy_events[0].record()

                for i in range(self.micro_batches):
                    current_buffer = buffers[i % 2]
                    current_event = copy_events[i % 2]

                    with torch.cuda.stream(self.compute_stream):
                        # Ensure the compute stream only waits when the copy has finished.
                        self.compute_stream.wait_event(current_event)
                        self.output = self.model(current_buffer)

                    if i + 1 < self.micro_batches:
                        next_buffer = buffers[(i + 1) % 2]
                        next_event = copy_events[(i + 1) % 2]
                        with torch.cuda.stream(self.copy_stream):
                            next_buffer.copy_(self.host_batches[i + 1], non_blocking=True)
                            next_event.record()
                current = torch.cuda.current_stream(device=self.device)
                current.wait_stream(self.compute_stream)
                current.wait_stream(self.copy_stream)
        if self.output is None or (self.buffer_a is None and self.buffer_b is None):
            raise RuntimeError("benchmark_fn() must produce output and buffers")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"buffer": self.buffer_a if self.buffer_a is not None else self.buffer_b},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
            output_tolerance=(0.1, 1.0),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
        )

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.buffer_a, self.buffer_b
        self.copy_stream = None
        self.compute_stream = None
        self.buffers = []
        self.copy_events = []
        self.host_batches = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.buffer_a is None or self.buffer_b is None:
            return "Buffers not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedMemoryDoubleBufferingBenchmark()
