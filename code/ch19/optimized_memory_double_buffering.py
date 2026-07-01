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


class BufferedMicrobatchMlp(nn.Module):
    """Microbatch MLP that reuses forward buffers in inference mode."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None
        self._fc1_forward_view: Optional[torch.Tensor] = None
        self._fc2_forward_view: Optional[torch.Tensor] = None
        self._fc1_weight_t: Optional[torch.Tensor] = None
        self._fc2_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._fc1_weight_t = self.fc1.weight.t()
        self._fc2_weight_t = self.fc2.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        normalized_shape = tuple(int(dim) for dim in shape)
        numel = 1
        for dim in normalized_shape:
            numel *= dim
        buffer = getattr(self, name)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, buffer)
        return buffer[:numel].view(normalized_shape)

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = tuple(x.shape[:-1])
        fc1_shape = (*prefix, self.fc1.out_features)
        fc2_shape = (*prefix, self.fc2.out_features)
        fc1_out = self._workspace("_fc1_buffer", fc1_shape, device=x.device, dtype=x.dtype)
        fc2_out = self._workspace("_fc2_buffer", fc2_shape, device=x.device, dtype=x.dtype)
        return fc1_out, fc2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def prepare_forward_buffers(self, x: torch.Tensor) -> None:
        self._fc1_forward_view, self._fc2_forward_view = self._ensure_forward_buffers(x)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.forward(x)
        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out = self._fc1_forward_view
        fc2_out = self._fc2_forward_view
        if fc1_out is None or fc2_out is None:
            raise RuntimeError("forward_prepared() requires prepare_forward_buffers()")
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def _forward_into_buffers(
        self,
        x: torch.Tensor,
        fc1_out: torch.Tensor,
        fc2_out: torch.Tensor,
    ) -> torch.Tensor:
        torch.matmul(x, self._fc1_weight_t, out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.matmul(fc1_out, self._fc2_weight_t, out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out


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
        self._buffer_event_counts: tuple[int, int] = (0, 0)
        self._expected_buffer_event_counts: tuple[int, int] = (0, 0)
        self._micro_batch_schedule: list[tuple[int, int, Optional[int], Optional[int]]] = []
        self._micro_batch_schedule_count = 0
        self._expected_micro_batch_schedule_count = 0
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 1024
        self.micro_batches = 16
        self.host_batches: list[torch.Tensor] = []
        self._verification_payload = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(requests_per_iteration=float(self.micro_batches))
    
    def setup(self) -> None:
        """Setup: Initialize model and double buffers."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = BufferedMicrobatchMlp(self.hidden_dim)
        # Optimization: Use FP16 for faster computation - FAIL FAST if not supported
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for optimized_memory_double_buffering benchmark")
        self.model = self.model.to(self.device).half()
        self.model.eval()
        self.model.cache_weight_views()
        
        # Optimization: Double buffering (ping-pong buffers)
        # Two buffers allow overlapping copy and compute operations
        # Ensure buffer dtype matches model dtype - FAIL FAST if model has no parameters
        params = list(self.model.parameters())
        if not params:
            raise RuntimeError("Model has no parameters - cannot determine dtype")
        self._payload_parameter_count = sum(p.numel() for p in params)
        model_dtype = params[0].dtype
        self.buffer_a = torch.empty(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=model_dtype
        )
        self.buffer_b = torch.empty_like(self.buffer_a)
        self.buffers = [self.buffer_a, self.buffer_b]
        self.model.prepare_forward_buffers(self.buffer_a)
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
        self._buffer_event_counts = (len(self.buffers), len(self.copy_events))
        self._expected_buffer_event_counts = (2, 2)
        self._micro_batch_schedule = [
            (
                batch_idx,
                batch_idx & 1,
                batch_idx + 1 if batch_idx + 1 < self.micro_batches else None,
                (batch_idx + 1) & 1 if batch_idx + 1 < self.micro_batches else None,
            )
            for batch_idx in range(self.micro_batches)
        ]
        self._micro_batch_schedule_count = len(self._micro_batch_schedule)
        self._expected_micro_batch_schedule_count = self.micro_batches
    
    def benchmark_fn(self) -> None:
        """Benchmark: Double buffering with overlapping operations."""
        assert self.copy_stream is not None and self.compute_stream is not None
        with nvtx_range("optimized_memory_double_buffering", enable=self._enable_nvtx):
            with torch.inference_mode():
                buffers = self.buffers
                copy_events = self.copy_events
                if self._buffer_event_counts != self._expected_buffer_event_counts:
                    raise RuntimeError("Double buffers or copy events not initialized")
                if self._micro_batch_schedule_count != self._expected_micro_batch_schedule_count:
                    raise RuntimeError("Double-buffer microbatch schedule not initialized")

                # Preload first buffer on the copy stream and signal completion.
                with torch.cuda.stream(self.copy_stream):
                    buffers[0].copy_(self.host_batches[0], non_blocking=True)
                    copy_events[0].record(self.copy_stream)

                for batch_idx, slot_idx, next_batch_idx, next_slot_idx in self._micro_batch_schedule:
                    current_buffer = buffers[slot_idx]
                    current_event = copy_events[slot_idx]

                    with torch.cuda.stream(self.compute_stream):
                        # Ensure the compute stream only waits when the copy has finished.
                        self.compute_stream.wait_event(current_event)
                        self.output = self.model.forward_prepared(current_buffer)

                    if next_batch_idx is not None and next_slot_idx is not None:
                        next_buffer = buffers[next_slot_idx]
                        next_event = copy_events[next_slot_idx]
                        with torch.cuda.stream(self.copy_stream):
                            next_buffer.copy_(self.host_batches[next_batch_idx], non_blocking=True)
                            next_event.record(self.copy_stream)
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
            parameter_count=self._payload_parameter_count,
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
        self._buffer_event_counts = (0, 0)
        self._expected_buffer_event_counts = (0, 0)
        self._micro_batch_schedule = []
        self._micro_batch_schedule_count = 0
        self._expected_micro_batch_schedule_count = 0
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
