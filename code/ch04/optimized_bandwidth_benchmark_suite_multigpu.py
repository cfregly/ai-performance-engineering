"""Optimized bandwidth benchmark suite; runs P2P copies across GPUs."""

from __future__ import annotations

import time
from typing import Optional

import torch

from core.benchmark.cuda_event_timing import max_elapsed_ms
from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


def measure_peer_bandwidth_pipelined(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    iterations: int,
    streams: list[torch.cuda.Stream],
) -> float:
    """Measure bandwidth using chunked async copies across multiple streams."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not streams:
        raise ValueError("streams must be non-empty")

    chunks = torch.chunk(src, len(streams))
    dst_chunks = torch.chunk(dst, len(streams))
    bytes_per_iter = src.numel() * src.element_size()

    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    start = time.perf_counter()
    for _ in range(iterations):
        for stream, src_chunk, dst_chunk in zip(streams, chunks, dst_chunks):
            with torch.cuda.stream(stream):
                dst_chunk.copy_(src_chunk, non_blocking=True)
        torch.cuda.synchronize(dst.device)
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    elapsed = (time.perf_counter() - start) / iterations
    gb_per_iter = bytes_per_iter / 1e9
    return gb_per_iter / elapsed


class OptimizedBandwidthSuiteMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        self.last_bandwidth_gbps: Optional[float] = None
        self.size_mb = 512
        self.inner_iterations = 12
        self._inner_iteration_range = range(self.inner_iterations)
        self.num_chunks = 8
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.chunk_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.streams: list[torch.cuda.Stream] = []
        self._timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._empty_timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = self._empty_timing_pairs
        self._stream_chunk_pairs: list[tuple[torch.cuda.Stream, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        self._stream_timing_pairs: list[tuple[torch.cuda.Stream, tuple[torch.cuda.Event, torch.cuda.Event]]] = []
        self._stream_timing_pair_count = 0
        self._expected_stream_timing_pair_count = 0
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        require_min_gpus(2, "optimized_bandwidth_benchmark_suite_multigpu.py")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        bytes_per_iter = int(self.size_mb * 1024 * 1024)
        numel = bytes_per_iter // 4  # float32
        device_count = torch.cuda.device_count()
        self._inner_iteration_range = range(self.inner_iterations)
        self.pairs = []
        self.chunk_pairs = []
        self.streams = []
        self._timing_pairs = []
        self._pending_timing_pairs = self._empty_timing_pairs
        self._stream_chunk_pairs = []
        self._stream_timing_pairs = []
        self._stream_timing_pair_count = 0
        self._expected_stream_timing_pair_count = 0
        src_buffers = [
            torch.randn(numel, device=f"cuda:{idx}", dtype=torch.float32)
            for idx in range(device_count)
        ]
        for idx, src in enumerate(src_buffers):
            dst_device = f"cuda:{(idx + 1) % device_count}"
            dst = torch.empty_like(src, device=dst_device)
            self.pairs.append((src, dst))
            src_chunks = torch.chunk(src, self.num_chunks)
            dst_chunks = torch.chunk(dst, self.num_chunks)
            self.chunk_pairs.append(list(zip(src_chunks, dst_chunks)))
            self.streams.append(torch.cuda.Stream(device=dst_device))
        self._timing_pairs = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in self.streams
        ]
        self._stream_chunk_pairs = list(zip(self.streams, self.chunk_pairs, strict=True))
        self._stream_timing_pairs = list(zip(self.streams, self._timing_pairs, strict=True))
        self._stream_timing_pair_count = len(self._stream_timing_pairs)
        self._expected_stream_timing_pair_count = len(self.streams)
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter * self.inner_iterations * len(self.pairs)),
        )

    def benchmark_fn(self) -> None:
        if not self.chunk_pairs:
            raise RuntimeError("Benchmark not initialized")
        if (
            not self._stream_timing_pairs
            or self._stream_timing_pair_count != self._expected_stream_timing_pair_count
        ):
            raise RuntimeError("Timing events not initialized")
        self._pending_timing_pairs = self._timing_pairs
        for stream, (start_event, _) in self._stream_timing_pairs:
            with torch.cuda.stream(stream):
                start_event.record()
        for _ in self._inner_iteration_range:
            for stream, chunk_list in self._stream_chunk_pairs:
                with torch.cuda.stream(stream):
                    for src_chunk, dst_chunk in chunk_list:
                        dst_chunk.copy_(src_chunk, non_blocking=True)
        for stream, (_, end_event) in self._stream_timing_pairs:
            with torch.cuda.stream(stream):
                end_event.record()

    def finalize_iteration_metrics(self) -> Optional[dict]:
        if not self._pending_timing_pairs:
            return None
        total_bytes = self.size_mb * 1024 * 1024 * len(self.pairs) * self.inner_iterations
        elapsed_ms_value = max_elapsed_ms(self._pending_timing_pairs)
        self._pending_timing_pairs = self._empty_timing_pairs
        elapsed_s = max(elapsed_ms_value, 1e-9) / 1000.0
        self.last_bandwidth_gbps = (total_bytes / elapsed_s) / 1e9
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if not self.pairs:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.pairs[0][0][: 256 * 256].view(256, 256)
        output = self.pairs[0][1][: 256 * 256].view(256, 256)
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
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            measurement_timeout_seconds=30,
            multi_gpu_required=True,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        self.finalize_iteration_metrics()
        """Return measured P2P bandwidth."""
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}

    def get_custom_streams(self) -> list[torch.cuda.Stream]:
        return self.streams

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedBandwidthSuiteMultiGPU()
