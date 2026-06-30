"""Shared helpers for Chapter 11 multi-stream benchmarks."""

from __future__ import annotations

from functools import partial
from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import canonicalize_nvtx_name

resolve_device = partial(require_cuda_device, "CUDA required for ch11")


class StridedStreamBaseline(VerificationPayloadMixin, BaseBenchmark):
    """Baseline workload that executes strided copies on a single stream."""

    declare_all_streams = False
    allowed_benchmark_fn_antipatterns = ("sync",)

    def __init__(
        self,
        nvtx_label: str,
        num_elements: int = 24_000_000,
        num_segments: int = 16,
    ):
        super().__init__()
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_segments = num_segments
        self.stream = None
        self.host_input = None
        self.host_output = None
        self._verify_output_buffer = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None
        self.chunk_triplets = None
        # Stream benchmark - fixed dimensions for overlap measurement
        bytes_transferred = float(num_elements * 4 * 2)  # H2D + D2H
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.stream = torch.cuda.Stream()
        self.host_input = torch.randn(
            self.N, device="cpu", dtype=torch.float32, pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        self._verify_output_buffer = torch.empty_like(self.host_output, pin_memory=True)
        self.host_in_chunks = list(torch.chunk(self.host_input, self.num_segments))
        self.host_out_chunks = list(torch.chunk(self.host_output, self.num_segments))
        self.device_chunks = [torch.empty_like(chunk, device=self.device) for chunk in self.host_in_chunks]
        self.chunk_triplets = list(zip(self.host_in_chunks, self.host_out_chunks, self.device_chunks, strict=True))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self.label):
            assert self.host_in_chunks is not None
            assert self.host_out_chunks is not None
            assert self.device_chunks is not None
            assert self.chunk_triplets is not None
            with torch.inference_mode():
                for h_in, h_out, d_buf in self.chunk_triplets:
                    with torch.cuda.stream(self.stream):
                        d_buf.copy_(h_in, non_blocking=True)
                        d_buf.mul_(2.0)
                        d_buf.add_(1.0)
                        d_buf.mul_(1.1)
                        d_buf.add_(0.5)
                        h_out.copy_(d_buf, non_blocking=True)
                    # Naive path blocks on each segment, preventing overlap.
                    self.stream.synchronize()
        if (
            self.host_input is None
            or self.host_output is None
            or self.host_in_chunks is None
            or self.host_out_chunks is None
        ):
            raise RuntimeError("benchmark_fn() must run after setup() initializes buffers")

    def capture_verification_payload(self) -> None:
        assert self.host_input is not None
        assert self.host_output is not None
        assert self._verify_output_buffer is not None
        self._verify_output_buffer.copy_(self.host_output)
        self._set_verification_payload(
            inputs={"host_input": self.host_input},
            output=self._verify_output_buffer,
            batch_size=self.host_input.numel(),
            parameter_count=0,
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.stream = None
        self.host_input = None
        self.host_output = None
        self._verify_output_buffer = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None
        self.chunk_triplets = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # Copy + compute per chunk is heavier; keep iteration count modest.
        nvtx_tag = canonicalize_nvtx_name(self.label)
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if self.host_output is None or self.host_input is None:
            return "Buffers not initialized"
        if self.host_output.shape != self.host_input.shape:
            return "Shape mismatch"
        if not torch.isfinite(self.host_output).all():
            return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the baseline (sequential) path."""
        bytes_transferred = float(self.N * 4 * 2)  # H2D + D2H
        return {
            f"{self.label}.elements": float(self.N),
            f"{self.label}.num_segments": float(self.num_segments),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.num_streams": 1.0,  # baseline uses 1 stream
            f"{self.label}.expected_overlap_pct": 0.0,
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.stream is None:
            return []
        return [self.stream]

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()


class ConcurrentStreamOptimized(VerificationPayloadMixin, BaseBenchmark):
    """Optimized workload that splits data across multiple CUDA streams."""

    declare_all_streams = False

    def __init__(
        self,
        nvtx_label: str = "concurrent_streams",
        num_elements: int = 24_000_000,
        num_segments: int = 16,
        num_streams: int = 2,
        chunk_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_segments = num_segments
        self.num_streams = num_streams
        self.dtype = chunk_dtype
        self.streams: List[torch.cuda.Stream] | None = None
        self.host_input: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self.host_in_chunks: List[torch.Tensor] | None = None
        self.host_out_chunks: List[torch.Tensor] | None = None
        self.device_chunks: List[torch.Tensor] | None = None
        self.stream_chunk_groups: List[tuple[torch.cuda.Stream, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        # Stream benchmark - fixed dimensions for overlap measurement
        element_size = float(torch.finfo(self.dtype).bits // 8)
        bytes_transferred = float(num_elements * element_size * 2)  # H2D + D2H
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        if self.num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        self.host_input = torch.randn(
            self.N, dtype=self.dtype, device="cpu", pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        self._verify_output_buffer = torch.empty_like(self.host_output, pin_memory=True)
        chunks = torch.chunk(self.host_input, self.num_segments)
        if len(chunks) < self.num_segments:
            chunks = list(chunks)
            for _ in range(self.num_segments - len(chunks)):
                empty = torch.empty(0, dtype=self.dtype, device="cpu", pin_memory=True)
                chunks.append(empty)
        self.host_in_chunks = list(chunks)
        self.host_out_chunks = list(torch.chunk(self.host_output, len(self.host_in_chunks)))
        self.device_chunks = [torch.empty_like(chunk, device=self.device) for chunk in self.host_in_chunks]
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self.stream_chunk_groups = [
            (self.streams[idx % self.num_streams], h_in, h_out, d_buf)
            for idx, (h_in, h_out, d_buf) in enumerate(
                zip(self.host_in_chunks, self.host_out_chunks, self.device_chunks, strict=True)
            )
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self.label):
            assert self.streams is not None
            assert self.host_in_chunks is not None
            assert self.host_out_chunks is not None
            assert self.device_chunks is not None
            assert self.stream_chunk_groups is not None
            with torch.inference_mode():
                for stream, h_in, h_out, d_buf in self.stream_chunk_groups:
                    if h_in.numel() == 0:
                        continue
                    with torch.cuda.stream(stream):
                        d_buf.copy_(h_in, non_blocking=True)
                        d_buf.mul_(2.0)
                        d_buf.add_(1.0)
                        d_buf.mul_(1.1)
                        d_buf.add_(0.5)
                        h_out.copy_(d_buf, non_blocking=True)

                # Join all work back onto the current stream so CUDA-event timing
                # accounts for multi-stream work (Locus/KernelBench 2025).
                current = torch.cuda.current_stream(self.device)
                for stream in self.streams:
                    current.wait_stream(stream)
        if (
            self.host_input is None
            or self.host_output is None
            or self.host_in_chunks is None
            or self.host_out_chunks is None
        ):
            raise RuntimeError("benchmark_fn() must run after setup() initializes buffers")

    def capture_verification_payload(self) -> None:
        assert self.host_input is not None
        assert self.host_output is not None
        assert self._verify_output_buffer is not None
        self._verify_output_buffer.copy_(self.host_output)
        self._set_verification_payload(
            inputs={"host_input": self.host_input},
            output=self._verify_output_buffer,
            batch_size=self.host_input.numel(),
            parameter_count=0,
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.streams = None
        self.host_input = None
        self.host_output = None
        self._verify_output_buffer = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None
        self.stream_chunk_groups = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # Streams overlap chunked work; fewer iterations keep runtime reasonable.
        nvtx_tag = canonicalize_nvtx_name(self.label)
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if not self.host_out_chunks:
            return "Chunks not initialized"
        for out in self.host_out_chunks:
            if not torch.isfinite(out).all():
                return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the optimized (concurrent) path."""
        bytes_transferred = float(self.N * 4 * 2)  # H2D + D2H
        return {
            f"{self.label}.elements": float(self.N),
            f"{self.label}.num_streams": float(self.num_streams),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.expected_overlap_pct": min(100.0, (self.num_streams - 1) / self.num_streams * 100),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.streams is None:
            return []
        return list(self.streams)

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()
