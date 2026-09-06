"""NCCL P2P baseline for the NVSHMEM pipeline comparison; skips on <2 GPUs."""

from __future__ import annotations

import os

import torch

from ch04.nvshmem_pipeline_result import (
    NVSHMEM_PIPELINE_RESULT_CALLBACK,
    NVSHMEMPipelineChildResultMixin,
)
from ch04.nvshmem_profile_ranges import (
    PIPELINE_BASELINE_NVTX_RANGE,
    PROFILE_NVTX_RANGE_ENV,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


class NVSHMEMPipelineParallelMultiGPU(
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
                nsys_nvtx_include=[PIPELINE_BASELINE_NVTX_RANGE],
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
            nsys_nvtx_include=[PIPELINE_BASELINE_NVTX_RANGE],
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
            variant="baseline",
            world_size=nproc_per_node,
            iterations=1,
            configuration=self._pipeline_configuration(transport="nccl"),
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
                "baseline",
                *self._pipeline_args(),
            ],
            env={
                **env,
                "AISP_DISABLE_SYMMEM_PIPELINE": "1",
                PROFILE_NVTX_RANGE_ENV: PIPELINE_BASELINE_NVTX_RANGE,
            },
            multi_gpu_required=True,
            name="baseline_nvshmem_pipeline_parallel_multigpu",
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
            "nccl",
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
    return NVSHMEMPipelineParallelMultiGPU()
