"""Shared NVLink offload helper for the persistent decode lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass
class OffloadConfig:
    use_pinned: bool
    non_blocking: bool
    use_copy_stream: bool
    batch_size: int = 2
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    chunk_tokens: int = 256
    dtype: torch.dtype = torch.float16


class NvlinkOffloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Measure H2D/D2H swap time for a KV-cache slice."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self, cfg: Optional[OffloadConfig] = None, label: str = "nvlink_offload"):
        super().__init__()
        self.cfg = cfg or OffloadConfig(use_pinned=False, non_blocking=False, use_copy_stream=False)
        self.label = label

        self.cpu_cache: Optional[torch.Tensor] = None
        self.gpu_cache: Optional[torch.Tensor] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.next_start: int = 0
        self._next_chunk_idx: int = 0
        self._chunk_count: int = 0
        self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._bytes_per_iteration: float = 0.0
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        shape = (
            self.cfg.num_layers,
            2,  # k/v
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.max_seq_len,
            self.cfg.head_dim,
        )

        self.cpu_cache = torch.zeros(shape, dtype=self.cfg.dtype, pin_memory=self.cfg.use_pinned)
        self.gpu_cache = torch.zeros(shape, dtype=self.cfg.dtype, device=self.device)
        self.copy_stream = torch.cuda.Stream() if self.cfg.use_copy_stream else None
        self._next_chunk_idx = 0
        self._chunk_count = 0
        self.next_start = 0
        self._chunk_views = []
        for start in range(0, self.cfg.max_seq_len, self.cfg.chunk_tokens):
            end = min(start + self.cfg.chunk_tokens, self.cfg.max_seq_len)
            slice_len = end - start
            if slice_len <= 0:
                continue
            self._chunk_views.append(
                (
                    self.cpu_cache[..., start:end, :],
                    self.gpu_cache[..., :slice_len, :],
                )
            )
        self._chunk_count = len(self._chunk_views)
        self._output_view = self.gpu_cache[
            ...,
            : min(1, self.cfg.max_seq_len),
            : min(8, self.cfg.head_dim),
        ]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)

        elements_per_chunk = (
            self.cfg.num_layers
            * 2
            * self.cfg.batch_size
            * self.cfg.num_heads
            * self.cfg.chunk_tokens
            * self.cfg.head_dim
        )
        # H2D + D2H each iteration
        self._bytes_per_iteration = elements_per_chunk * self.cfg.dtype.itemsize * 2.0
        self.register_workload_metadata(bytes_per_iteration=self._bytes_per_iteration)

    def benchmark_fn(self) -> None:
        assert self.cpu_cache is not None and self.gpu_cache is not None
        if not self._chunk_views or not self._chunk_count or self._output_view is None:
            raise RuntimeError("Offload chunk views not initialized")

        cpu_slice, gpu_slice = self._chunk_views[self._next_chunk_idx]
        copy_stream = self.copy_stream
        current_stream = torch.cuda.current_stream() if copy_stream is not None else None
        if copy_stream is not None and current_stream is not None:
            with torch.cuda.stream(copy_stream):
                gpu_slice.copy_(cpu_slice, non_blocking=self.cfg.non_blocking)
            current_stream.wait_stream(copy_stream)
        else:
            gpu_slice.copy_(cpu_slice, non_blocking=self.cfg.non_blocking)

        # Lightweight compute to keep the slice "hot"
        gpu_slice.mul_(1.0001)

        if copy_stream is not None and current_stream is not None:
            copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(copy_stream):
                cpu_slice.copy_(gpu_slice, non_blocking=self.cfg.non_blocking)
            current_stream.wait_stream(copy_stream)
        else:
            cpu_slice.copy_(gpu_slice, non_blocking=self.cfg.non_blocking)

        # Capture a representative slice for verification (GPU slice to avoid host sync patterns)
        self.output = self._output_view
        self._next_chunk_idx = (self._next_chunk_idx + 1) % self._chunk_count
        self.next_start = self._next_chunk_idx * self.cfg.chunk_tokens
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"gpu_cache": self.gpu_cache.detach()},
            output=self._verify_output_buffer,
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.cfg.dtype == torch.float16,
                "bf16": self.cfg.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.cpu_cache = None
        self.gpu_cache = None
        self.copy_stream = None
        self._next_chunk_idx = 0
        self._chunk_count = 0
        self._chunk_views = []
        self._output_view = None
        self._verify_output_buffer = None
        self.output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Keep iterations small to avoid large host memory pressure during sweeps.
        return BenchmarkConfig(
            iterations=16,
            warmup=5,
            timeout_seconds=180,
            measurement_timeout_seconds=180,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVLink offload performance metrics."""
        return {
            f"{self.label}.bytes_per_iteration": self._bytes_per_iteration,
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_layers": float(self.cfg.num_layers),
            f"{self.label}.chunk_tokens": float(self.cfg.chunk_tokens),
            f"{self.label}.use_pinned": float(self.cfg.use_pinned),
            f"{self.label}.non_blocking": float(self.cfg.non_blocking),
        }

 
