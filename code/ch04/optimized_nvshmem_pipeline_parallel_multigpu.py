"""Optimized NVSHMEM pipeline parallel wrapper with NVLink5/NVLS tuning; skips on <2 GPUs.

Uses symmetric-memory handoff on the 1F1B schedule to reduce pipeline stalls.
"""

from __future__ import annotations

import os

import torch

from ch04.nccl_blackwell_config import (
    configure_nccl_for_blackwell,
    configure_nccl_for_gb200_gb300,
    configure_nccl_for_multigpu,
    detect_b200_multigpu_topology,
)
from ch04.nvshmem_pipeline_result import (
    NVSHMEM_PIPELINE_RESULT_CALLBACK,
    NVSHMEMPipelineChildResultMixin,
)
from ch04.nvshmem_profile_ranges import (
    PIPELINE_OPTIMIZED_NVTX_RANGE,
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


class OptimizedNVSHMEMPipelineParallelMultiGPU(
    NVSHMEMPipelineChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"
    allowed_benchmark_fn_antipatterns = ("host_transfer", "random_input_regeneration", "sync")

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_pipeline_parallel_multigpu requires >=2 GPUs")
        if not symmetric_memory_available():
            raise RuntimeError(
                "SKIPPED: nvshmem_pipeline_parallel_multigpu requires NVSHMEM or SymmetricMemory support"
            )
        # NCCL tuning helps the real symmetric-memory pipeline path on supported hosts.
        _configure_blackwell_nccl()

    def benchmark_fn(self) -> None:
        raise RuntimeError(
            "NVSHMEM pipeline executes only through its torchrun worker; "
            "use get_torchrun_spec()"
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()

    def capture_verification_payload(self) -> None:
        self.require_nvshmem_pipeline_child_result()

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(
                iterations=1,
                warmup=5,
                measurement_timeout_seconds=300,
                nsys_nvtx_include=[PIPELINE_OPTIMIZED_NVTX_RANGE],
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
            nsys_nvtx_include=[PIPELINE_OPTIMIZED_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        if int(effective_config.nnodes or 1) != 1:
            raise RuntimeError("Pipeline child-result transport requires nnodes == 1")
        nproc_per_node = int(
            effective_config.nproc_per_node or max(2, torch.cuda.device_count())
        )
        env = self.prepare_nvshmem_pipeline_child_result(
            variant="optimized",
            world_size=nproc_per_node,
            iterations=1,
            configuration=self._pipeline_configuration(transport="nvshmem"),
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
                "pipeline",
                "--variant",
                "optimized",
                *self._pipeline_args(),
            ],
            env={
                **env,
                "AISP_DISABLE_SYMMEM_PIPELINE": "0",
                PROFILE_NVTX_RANGE_ENV: PIPELINE_OPTIMIZED_NVTX_RANGE,
            },
            multi_gpu_required=True,
            name="optimized_nvshmem_pipeline_parallel_multigpu",
            result_callback=NVSHMEM_PIPELINE_RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=1,
        )

    @staticmethod
    def _pipeline_args() -> list[str]:
        return [
            "--schedule",
            "1f1b",
            "--batch-size",
            "64",
            "--num-microbatches",
            "4",
            "--seq-len",
            "16",
            "--hidden-dim",
            "32",
            "--transport",
            "nvshmem",
        ]

    @staticmethod
    def _pipeline_configuration(*, transport: str) -> dict[str, int | str]:
        return {
            "schedule": "1f1b",
            "batch_size": 64,
            "num_microbatches": 4,
            "microbatch_size": 16,
            "seq_len": 16,
            "hidden_dim": 32,
            "transport": transport,
        }

    def get_custom_metrics(self) -> dict | None:
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
    return OptimizedNVSHMEMPipelineParallelMultiGPU()
