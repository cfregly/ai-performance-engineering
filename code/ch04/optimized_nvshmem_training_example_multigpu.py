"""Optimized NVSHMEM training example with NCCL 2.28 tuning; skips on <2 GPUs."""

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
    PROFILE_NVTX_RANGE_ENV,
    TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE,
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
    """Enable NCCL 2.28 knobs for Blackwell/Grace-Blackwell."""
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


class OptimizedNVSHMEMTrainingExampleMultiGPU(
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
            raise RuntimeError("SKIPPED: nvshmem_training_example requires >=2 GPUs")
        if not symmetric_memory_available():
            raise RuntimeError(
                "SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support"
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
                nsys_nvtx_include=[TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE],
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
            nsys_nvtx_include=[TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE],
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
            workload="training-example",
            world_size=world_size,
            iterations=240,
            configuration={
                "demo": "pipeline",
                "batch_size": 2,
                "seq_len": 256,
                "dim": 384,
                "steps": 240,
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
                "training-example",
                "--variant",
                "optimized",
                *self._training_args(),
            ],
            env={
                **result_env,
                "AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS": "1",
                PROFILE_NVTX_RANGE_ENV: TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE,
            },
            multi_gpu_required=True,
            name="optimized_nvshmem_training_example_multigpu",
            result_callback=NVSHMEM_CHILD_RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=240,
        )

    @staticmethod
    def _training_args() -> list[str]:
        return [
            "--demo",
            "pipeline",
            "--batch-size",
            "2",
            "--seq-len",
            "256",
            "--dim",
            "384",
            "--steps",
            "240",
        ]

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics

        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred
            if hasattr(self, "_bytes_transferred")
            else float(getattr(self, "N", 1024) * 4),
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            transfer_type="hbm",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMTrainingExampleMultiGPU()
