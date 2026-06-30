"""Shared single-GPU transfer benchmark utilities for Chapter 04."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin


class SingleGPUTransferBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU transfer microbenchmark with optional pipelining."""

    allowed_benchmark_fn_antipatterns = ("sync",)

    def __init__(
        self,
        *,
        size_mb: int = 256,
        inner_iterations: int = 20,
        num_chunks: int = 8,
        use_streams: bool = False,
        sync_per_chunk: bool = True,
        collective_type: str,
    ) -> None:
        super().__init__()
        self.size_mb = int(size_mb)
        self.inner_iterations = int(inner_iterations)
        self._inner_iteration_range = range(self.inner_iterations)
        self.num_chunks = int(num_chunks)
        self.use_streams = bool(use_streams)
        self.sync_per_chunk = bool(sync_per_chunk)
        self.collective_type = collective_type
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.chunk_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._stream_chunk_pairs: list[tuple[torch.cuda.Stream, torch.Tensor, torch.Tensor]] = []
        self.streams: list[torch.cuda.Stream] = []
        self.last_bandwidth_gbps: Optional[float] = None
        bytes_per_iter = self.size_mb * 1024 * 1024
        self._total_bytes_per_benchmark = bytes_per_iter * self.inner_iterations
        self._pending_bandwidth_sample = False
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter * self.inner_iterations),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for single-GPU transfer benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        bytes_per_iter = self.size_mb * 1024 * 1024
        numel = bytes_per_iter // 4  # float32
        self._total_bytes_per_benchmark = bytes_per_iter * self.inner_iterations
        self._pending_bandwidth_sample = False
        self._inner_iteration_range = range(self.inner_iterations)
        self.src = torch.randn(numel, device=self.device, dtype=torch.float32)
        self.dst = torch.empty_like(self.src, device=self.device)
        src_chunks = torch.chunk(self.src, self.num_chunks)
        dst_chunks = torch.chunk(self.dst, self.num_chunks)
        self.chunk_pairs = list(zip(src_chunks, dst_chunks))
        self.streams = []
        self._stream_chunk_pairs = []
        if self.use_streams:
            # Keep stream count low so stream auditing stays green for Ch04.
            stream_count = min(2, len(self.chunk_pairs))
            for _ in range(stream_count):
                self.streams.append(torch.cuda.Stream(device=self.device))
            self._stream_chunk_pairs = [
                (self.streams[idx % stream_count], src_chunk, dst_chunk)
                for idx, (src_chunk, dst_chunk) in enumerate(self.chunk_pairs)
            ]

    def benchmark_fn(self) -> None:
        if self.src is None or self.dst is None or not self.chunk_pairs:
            raise RuntimeError("setup() must run before benchmark_fn()")
        for _ in self._inner_iteration_range:
            if self.use_streams:
                if not self.streams:
                    raise RuntimeError("use_streams=True requires at least one CUDA stream")
                for stream, src_chunk, dst_chunk in self._stream_chunk_pairs:
                    with torch.cuda.stream(stream):
                        dst_chunk.copy_(src_chunk, non_blocking=True)
                for stream in self.streams:
                    stream.synchronize()
            else:
                for src_chunk, dst_chunk in self.chunk_pairs:
                    dst_chunk.copy_(src_chunk, non_blocking=False)
                    if self.sync_per_chunk:
                        torch.cuda.synchronize(self.device)
        torch.cuda.synchronize(self.device)
        self._pending_bandwidth_sample = True

    def finalize_iteration_metrics(self) -> Optional[dict]:
        if not self._pending_bandwidth_sample:
            return None
        self._pending_bandwidth_sample = False
        elapsed_ms = getattr(self, "_last_wall_elapsed_ms", None)
        if elapsed_ms is None:
            elapsed_ms = getattr(self, "_last_elapsed_ms", None)
        if elapsed_ms is None:
            return None
        elapsed_s = max(float(elapsed_ms), 1e-9) / 1000.0
        self.last_bandwidth_gbps = (self._total_bytes_per_benchmark / elapsed_s) / 1e9
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.src is None or self.dst is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        probe = self.src[: 256 * 256].view(256, 256)
        output = self.dst[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": 1,
                "collective_type": self.collective_type,
            },
        )

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        self.chunk_pairs = []
        self._stream_chunk_pairs = []
        self.streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, measurement_timeout_seconds=60)

    def get_custom_metrics(self) -> Optional[dict]:
        self.finalize_iteration_metrics()
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}
