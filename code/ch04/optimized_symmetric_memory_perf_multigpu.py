#!/usr/bin/env python3
"""Optimized symmetric-memory perf microbench (SymmetricMemory puts only).

Measures latency/bandwidth of direct peer writes using torch.distributed.nn.SymmetricMemory.
Demonstrates the uplift from using direct GPU-to-GPU memory access vs NCCL collectives.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ch04.symmetric_memory_perf_common import build_square_verification_probe
from core.benchmark.cuda_event_timing import max_elapsed_ms
from core.benchmark.metrics import compute_memory_transfer_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import resolve_local_rank
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import (
    SymmetricMemoryHandle,
    create_symmetric_memory_handle,
    symmetric_memory_available,
)


def init_distributed() -> Tuple[int, int, int]:
    """Initialize process group for a single-node demo."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = resolve_local_rank()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
            device_id=local_rank,
        )
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


class OptimizedSymmetricMemoryPerfBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized SymmetricMemory peer-put benchmark for direct GPU memory access."""
    multi_gpu_required = True

    def __init__(self, size_mb: float = 1.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.local_tensor: Optional[torch.Tensor] = None
        self.peer_buffer: Optional[torch.Tensor] = None
        self.prev_buffer: Optional[torch.Tensor] = None
        self.handle: Optional[SymmetricMemoryHandle] = None
        self._buffer_count = 2
        self._local_buffers: Optional[torch.Tensor] = None
        self._peer_buffers: Optional[torch.Tensor] = None
        self._prev_buffers: Optional[torch.Tensor] = None
        self._copy_streams: Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream]] = None
        self._buffer_events: Optional[List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = None
        self._buffer_inflight: Optional[List[bool]] = None
        self.rank = 0
        self.world_size = 1
        self.peer_rank = 0
        self._last_avg_ms = 0.0
        self._last_gbps = 0.0
        self._bytes_transferred = 0.0
        self._inner_iterations = 2000
        self._inner_iteration_range = range(self._inner_iterations)
        self._timing_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._empty_timing_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_timing_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = self._empty_timing_pairs
        self._stream_timing_pairs: List[tuple[torch.cuda.Stream, tuple[torch.cuda.Event, torch.cuda.Event]]] = []
        self._timing_pair_count = 0
        self._stream_timing_pair_count = 0
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_numel = 0
        self._local_buffer: Optional[torch.Tensor] = None
        self._peer_buffer: Optional[torch.Tensor] = None
        self._prev_buffer: Optional[torch.Tensor] = None
        self._recv_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Initialize distributed and allocate symmetric memory."""
        if not symmetric_memory_available():
            raise RuntimeError(
                "SKIPPED: SymmetricMemory not available. "
                "Install PyTorch with SymmetricMemory support."
            )

        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory_perf requires >= 2 GPUs")

        self.rank, self.world_size, device_id = init_distributed()
        
        if self.world_size < 2:
            raise RuntimeError("SKIPPED: SymmetricMemory peer-put requires world_size >= 2")

        device = torch.device("cuda", device_id)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.local_tensor = torch.randn(self.numel, device=device, dtype=torch.float32)

        # Create symmetric memory handle for direct peer access.
        self.handle = create_symmetric_memory_handle(self.local_tensor)
        self.peer_rank = (self.rank + 1) % self.world_size
        self._local_buffer = self.handle.buffer
        self._peer_buffer = self.handle.get_buffer(self.peer_rank)
        self._prev_buffer = self.handle.get_buffer((self.rank - 1) % self.world_size)
        self._recv_buffer = torch.empty_like(self._local_buffer)
        self._verify_input, self._verify_numel = build_square_verification_probe(self._local_buffer)
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        if self._copy_streams is None:
            self._copy_streams = (
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            )
        self._timing_pairs = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(2)
        ]
        self._stream_timing_pairs = list(zip(self._copy_streams, self._timing_pairs, strict=True))
        self._timing_pair_count = len(self._timing_pairs)
        self._stream_timing_pair_count = len(self._stream_timing_pairs)
        self._inner_iteration_range = range(self._inner_iterations)
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        """Run direct peer copy via SymmetricMemory and measure performance."""
        if (
            self._local_buffer is None
            or self._peer_buffer is None
            or self._prev_buffer is None
            or self._recv_buffer is None
        ):
            raise RuntimeError("Tensors not initialized")

        timing_pairs = self._timing_pairs
        if (
            self._copy_streams is None
            or self._timing_pair_count < 2
            or self._stream_timing_pair_count < 2
        ):
            raise RuntimeError("Timing events not initialized")
        send_stream, recv_stream = self._copy_streams
        current_stream = torch.cuda.current_stream()
        send_stream.wait_stream(current_stream)
        recv_stream.wait_stream(current_stream)
        for stream, (start_event, _) in self._stream_timing_pairs:
            with torch.cuda.stream(stream):
                start_event.record()
        for _ in self._inner_iteration_range:
            with torch.cuda.stream(send_stream):
                self._peer_buffer.copy_(self._local_buffer, non_blocking=True)
            with torch.cuda.stream(recv_stream):
                self._recv_buffer.copy_(self._prev_buffer, non_blocking=True)
        current_stream.wait_stream(send_stream)
        current_stream.wait_stream(recv_stream)
        with torch.cuda.stream(send_stream):
            timing_pairs[0][1].record()
        with torch.cuda.stream(recv_stream):
            timing_pairs[1][1].record()
        self._pending_timing_pairs = timing_pairs
        self._verify_output = self._recv_buffer
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if not self._pending_timing_pairs:
            return None
        elapsed_ms_value = max_elapsed_ms(self._pending_timing_pairs)
        self._pending_timing_pairs = self._empty_timing_pairs
        bytes_per_iter = self.size_mb * 1024 * 1024 * 2
        bytes_moved = bytes_per_iter * self._inner_iterations
        gbps = (bytes_moved / (elapsed_ms_value / 1000.0)) / 1e9 if elapsed_ms_value > 0 else 0.0
        self._last_avg_ms = elapsed_ms_value
        self._last_gbps = gbps
        self._bytes_transferred = bytes_moved
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("Verification buffers not initialized")
        probe = self._verify_input
        if self._verify_output is not None:
            output_source = self._verify_output[: self._verify_numel].view_as(self._verify_input).detach()
        else:
            output_source = probe
        self._verify_output_buffer.copy_(output_source)

        self._set_verification_payload(
            inputs={"tensor": probe},
            output=self._verify_output_buffer,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={"world_size": torch.cuda.device_count()},
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.capture_verification_payload()
        self._subprocess_verify_output = self.get_verify_output()
        self._subprocess_output_tolerance = self.get_output_tolerance()
        self._subprocess_input_signature = self.get_input_signature()

    def teardown(self) -> None:
        """Cleanup distributed resources."""
        self.local_tensor = None
        self.peer_buffer = None
        self.prev_buffer = None
        self.handle = None
        self._local_buffers = None
        self._peer_buffers = None
        self._prev_buffers = None
        self._copy_streams = None
        self._buffer_events = None
        self._buffer_inflight = None
        self._timing_pairs = []
        self._pending_timing_pairs = self._empty_timing_pairs
        self._stream_timing_pairs = []
        self._timing_pair_count = 0
        self._stream_timing_pair_count = 0
        self._verify_output = None
        self._verify_output_buffer = None
        self._local_buffer = None
        self._peer_buffer = None
        self._prev_buffer = None
        self._recv_buffer = None
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(iterations=10, warmup=5)
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=10,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="optimized_symmetric_memory_perf_multigpu",
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        """Return memory transfer metrics for SymmetricMemory peer-put."""
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="nvlink",  # SymmetricMemory uses direct NVLink access
        )

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        """Validate benchmark ran successfully."""
        if self.local_tensor is None or self._local_buffer is None:
            return "Local buffer not initialized"
        if self._peer_buffer is None:
            return "Peer buffer not initialized"
        if self._recv_buffer is None:
            return "Receive buffer not initialized"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedSymmetricMemoryPerfBenchmark(size_mb=0.0625)
