"""Benchmark wrapper for bandwidth suite; skips on <2 GPUs."""

from __future__ import annotations

import torch

from core.benchmark.cuda_event_timing import max_elapsed_ms
from core.benchmark.gpu_requirements import require_min_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin

import time
from typing import Optional


def measure_peer_bandwidth(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    iterations: int = 50,
    stream: Optional[torch.cuda.Stream] = None,
) -> float:
    """Measure GPU-to-GPU bandwidth by copying a tensor between two devices."""
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: bandwidth benchmark suite requires >=2 GPUs")
    if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
        raise TypeError("src and dst must be torch.Tensor")
    if src.numel() != dst.numel() or src.dtype != dst.dtype:
        raise ValueError("src and dst must have matching shape/dtype")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    bytes_per_iter = src.numel() * src.element_size()
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    start = time.perf_counter()
    if stream is not None:
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                dst.copy_(src, non_blocking=True)
        stream.synchronize()
    else:
        for _ in range(iterations):
            dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    elapsed = (time.perf_counter() - start) / iterations
    gb_per_iter = bytes_per_iter / 1e9
    return gb_per_iter / elapsed


class BandwidthSuiteMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        self.last_bandwidth_gbps: Optional[float] = None
        self.size_mb = 512
        self.inner_iterations = 12
        self._inner_iteration_range = range(self.inner_iterations)
        self.num_chunks = 32
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.chunk_pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self._flat_chunk_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._pair_count = 0
        self._total_bytes_per_benchmark = 0
        self._timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._empty_timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_timing_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = self._empty_timing_pairs
        self._timing_device_pairs: list[tuple[torch.device, tuple[torch.cuda.Event, torch.cuda.Event]]] = []
        self._timing_pair_count = 0
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        require_min_gpus(2, "baseline_bandwidth_benchmark_suite_multigpu.py")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        bytes_per_iter = int(self.size_mb * 1024 * 1024)
        numel = bytes_per_iter // 4  # float32
        device_count = torch.cuda.device_count()
        self._inner_iteration_range = range(self.inner_iterations)
        self.pairs = []
        self.chunk_pairs = []
        self._flat_chunk_pairs = []
        self._total_bytes_per_benchmark = 0
        self._timing_pairs = []
        self._pending_timing_pairs = self._empty_timing_pairs
        self._timing_device_pairs = []
        self._timing_pair_count = 0
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
            chunk_pairs = list(zip(src_chunks, dst_chunks))
            self.chunk_pairs.append(chunk_pairs)
            self._flat_chunk_pairs.extend(chunk_pairs)
            with torch.cuda.device(dst.device):
                self._timing_pairs.append(
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                )
        self._pair_count = len(self.pairs)
        self._total_bytes_per_benchmark = bytes_per_iter * self.inner_iterations * self._pair_count
        self._timing_device_pairs = [
            (dst.device, event_pair)
            for (_, dst), event_pair in zip(self.pairs, self._timing_pairs, strict=True)
        ]
        self._timing_pair_count = len(self._timing_device_pairs)
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter * self.inner_iterations * self._pair_count),
        )

    def benchmark_fn(self) -> None:
        if not self._flat_chunk_pairs:
            raise RuntimeError("Benchmark not initialized")
        if self._timing_pair_count != self._pair_count:
            raise RuntimeError("Timing events not initialized")
        self._pending_timing_pairs = self._timing_pairs
        for dst_device, (start_event, _) in self._timing_device_pairs:
            with torch.cuda.device(dst_device):
                start_event.record(torch.cuda.current_stream(dst_device))
        for _ in self._inner_iteration_range:
            for src_chunk, dst_chunk in self._flat_chunk_pairs:
                dst_chunk.copy_(src_chunk, non_blocking=False)
        for dst_device, (_, end_event) in self._timing_device_pairs:
            with torch.cuda.device(dst_device):
                end_event.record(torch.cuda.current_stream(dst_device))

    def finalize_iteration_metrics(self) -> Optional[dict]:
        if not self._pending_timing_pairs:
            return None
        elapsed_ms_value = max_elapsed_ms(self._pending_timing_pairs)
        self._pending_timing_pairs = self._empty_timing_pairs
        elapsed_s = max(elapsed_ms_value, 1e-9) / 1000.0
        self.last_bandwidth_gbps = (self._total_bytes_per_benchmark / elapsed_s) / 1e9
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
        """Return measured P2P bandwidth."""
        self.finalize_iteration_metrics()
        return {"p2p_bandwidth_gbps": float(self.last_bandwidth_gbps or 0.0)}

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
    return BandwidthSuiteMultiGPU()
