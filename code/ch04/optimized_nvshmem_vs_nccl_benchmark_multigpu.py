"""Optimized NVSHMEM vs NCCL benchmark with NVLink5/NVLS knobs; skips on <2 GPUs."""

from __future__ import annotations

import os
from typing import Optional

import torch

from ch04.nccl_blackwell_config import (
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    configure_nccl_for_multigpu,
    detect_b200_multigpu_topology,
)
from ch04.nvshmem_child_result import (
    NVSHMEM_CHILD_RESULT_CALLBACK,
    NVSHMEMChildResultMixin,
)
from ch04.nvshmem_profile_ranges import (
    COLLECTIVE_OPTIMIZED_NVTX_RANGE,
    PROFILE_NVTX_RANGE_ENV,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import symmetric_memory_available


def _configure_blackwell_nccl() -> None:
    try:
        topo = detect_b200_multigpu_topology()
    except Exception:
        configure_nccl_for_blackwell(verbose=False)
        return

    if topo.get("has_grace_cpu"):
        configure_nccl_for_gb200_gb300(verbose=False)
    elif topo.get("num_gpus", 0) >= 2 and topo.get("is_b200_multigpu"):
        configure_nccl_for_multigpu(num_gpus=topo.get("num_gpus", 2), verbose=False)
    else:
        configure_nccl_for_blackwell(verbose=False)


class OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(
    NVSHMEMChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._benchmark_ready = False

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_vs_nccl_benchmark requires >=2 GPUs")
        if not symmetric_memory_available():
            raise RuntimeError(
                "SKIPPED: nvshmem_vs_nccl_benchmark requires NVSHMEM or SymmetricMemory support"
            )
        _configure_blackwell_nccl()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._benchmark_ready = True

    def benchmark_fn(self) -> None:
        if not self._benchmark_ready:
            raise RuntimeError("setup() must run before benchmark_fn()")

    def teardown(self) -> None:
        self._benchmark_ready = False
        torch.cuda.empty_cache()

    def capture_verification_payload(self) -> None:
        self.require_nvshmem_child_result()

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(
                iterations=1,
                warmup=5,
                measurement_timeout_seconds=300,
                nsys_nvtx_include=[COLLECTIVE_OPTIMIZED_NVTX_RANGE],
                ncu_replay_mode="app-range",
                ncu_replay_mode_override=True,
            )
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
            nsys_nvtx_include=[COLLECTIVE_OPTIMIZED_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        if int(effective_config.nnodes or 1) != 1:
            raise RuntimeError("NVSHMEM child-result transport requires nnodes == 1")
        world_size = int(
            effective_config.nproc_per_node or max(2, torch.cuda.device_count())
        )
        result_env = self.prepare_nvshmem_child_result(
            variant="optimized",
            workload="collective",
            world_size=world_size,
            iterations=500,
            configuration={
                "min_bytes": 1048576,
                "max_bytes": 1048576,
                "steps": 1,
                "iterations": 500,
                "mode": "nvshmem",
            },
        )
        return TorchrunLaunchSpec(
            module_name="core.harness.benchmark_worker",
            script_args=[
                "--module",
                "ch04.nvshmem_worker",
                "--callable",
                "main",
                "--",
                "--workload",
                "collective",
                "--variant",
                "optimized",
                "--min-bytes",
                "1048576",
                "--max-bytes",
                "1048576",
                "--steps",
                "1",
                "--iterations",
                "500",
                "--mode",
                "nvshmem",
            ],
            env={
                **result_env,
                "AISP_DISABLE_SYMMETRIC_MEMORY": "0",
                "AISP_BROADCAST_OVERLAP": "1",
                "AISP_BROADCAST_COMPUTE_PASSES": "8",
                PROFILE_NVTX_RANGE_ENV: COLLECTIVE_OPTIMIZED_NVTX_RANGE,
            },
            multi_gpu_required=True,
            name="optimized_nvshmem_vs_nccl_benchmark_multigpu",
            result_callback=NVSHMEM_CHILD_RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=500,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """The harness owns timing; the worker result carries verification tensors."""
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU()
