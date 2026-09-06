#!/usr/bin/env python3
"""Baseline symmetric-memory perf microbench (NCCL only).

Compares simple NCCL AllReduce latency/bandwidth across payload sizes.
Use the optimized variant to see the uplift when using SymmetricMemory + direct puts.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ch04.symmetric_memory_perf_common import build_square_verification_probe
from ch04.symmetric_memory_perf_common import (
    SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE,
    SYMMETRIC_MEMORY_PERF_RESULT_CALLBACK,
    SymmetricMemoryPerfChildResultMixin,
)
from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.metrics import compute_memory_transfer_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import resolve_local_rank
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
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


class BaselineSymmetricMemoryPerfBenchmark(
    SymmetricMemoryPerfChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    """Baseline NCCL P2P send/recv benchmark for symmetric memory comparison."""
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"

    def __init__(self, size_mb: float = 1.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.tensor: Optional[torch.Tensor] = None
        self.recv_tensor: Optional[torch.Tensor] = None
        self.rank = 0
        self.world_size = 1
        self._last_avg_ms = 0.0
        self._last_gbps = 0.0
        self._bytes_transferred = 0.0
        self._inner_iterations = 2000
        self._inner_iteration_range = range(self._inner_iterations)
        self._timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_numel = 0

    def setup(self) -> None:
        """Initialize distributed and allocate tensor."""
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory_perf requires >= 2 GPUs")
        
        self.rank, self.world_size, device_id = init_distributed()
        device = torch.device("cuda", device_id)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(self.numel, device=device, dtype=torch.float32)
        self.recv_tensor = torch.empty_like(self.tensor)
        self._verify_input, self._verify_numel = build_square_verification_probe(self.tensor)
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        self._timing_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self._inner_iteration_range = range(self._inner_iterations)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        """Run NCCL P2P send/recv and measure performance."""
        if self.tensor is None or self.recv_tensor is None:
            raise RuntimeError("Tensor not initialized")

        if self._timing_pair is None:
            raise RuntimeError("Timing events not initialized")
        start, end = self._timing_pair

        start.record()
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        for _ in self._inner_iteration_range:
            if self.rank % 2 == 0:
                dist.send(self.tensor, dst=next_rank)
                dist.recv(self.recv_tensor, src=prev_rank)
            else:
                dist.recv(self.recv_tensor, src=prev_rank)
                dist.send(self.tensor, dst=next_rank)
        end.record()
        self._pending_timing_pair = (start, end)
        self._verify_output = self.recv_tensor
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if self._pending_timing_pair is None:
            return None
        elapsed_ms_value = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
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
        if self._verify_output is None:
            raise RuntimeError("Timed receive output was not produced")
        probe = self._verify_input
        output_source = self._verify_output[: self._verify_numel].view_as(probe).detach()
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
            signature_overrides={"world_size": self.world_size},
        )

    def _prepare_verification_payload(self) -> None:
        self.require_symmetric_memory_perf_child_result()

    def teardown(self) -> None:
        """Cleanup distributed resources."""
        self.tensor = None
        self.recv_tensor = None
        self._timing_pair = None
        self._pending_timing_pair = None
        self._verify_output = None
        self._verify_output_buffer = None
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(
                iterations=10,
                warmup=5,
                nsys_nvtx_include=[SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE],
                ncu_replay_mode="app-range",
                ncu_replay_mode_override=True,
            )
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=10,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
            nsys_nvtx_include=[SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        nnodes = int(effective_config.nnodes or 1)
        if nnodes != 1:
            raise RuntimeError(
                "Symmetric-memory perf child-result transport requires nnodes == 1"
            )
        nproc_per_node = int(
            effective_config.nproc_per_node or torch.cuda.device_count()
        )
        env = self.prepare_symmetric_memory_perf_child_result(
            variant="baseline",
            world_size=nproc_per_node,
        )
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve().with_name(
                "symmetric_memory_perf_worker.py"
            ),
            script_args=[
                "--variant",
                "baseline",
                "--warmup",
                str(effective_config.warmup),
                "--iterations",
                str(effective_config.iterations),
            ],
            env=env,
            multi_gpu_required=True,
            name="baseline_symmetric_memory_perf_multigpu",
            result_callback=SYMMETRIC_MEMORY_PERF_RESULT_CALLBACK,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        """Return memory transfer metrics for NCCL P2P send/recv."""
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="nvlink",  # NCCL uses NVLink when available
        )

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        """Validate benchmark ran successfully."""
        if self._symmetric_memory_perf_result_bundle is not None:
            return None
        if self.tensor is None:
            return "Tensor not initialized"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineSymmetricMemoryPerfBenchmark(size_mb=0.0625)
