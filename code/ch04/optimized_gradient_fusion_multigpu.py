"""Optimized gradient fusion benchmark (one fused gradient average)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ch04.gradient_fusion_multigpu import (
    OPTIMIZED_PROFILE_NVTX_RANGE,
    PAIR_NUM_TENSORS,
    PAIR_TENSOR_KB,
    PAIR_TIMED_ITERATIONS,
)
from ch04.gradient_fusion_result import (
    GRADIENT_FUSION_RESULT_CALLBACK,
    GradientFusionChildResultMixin,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


class OptimizedGradientFusionMultiGPU(
    GradientFusionChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: gradient_fusion requires >=2 GPUs")

    def benchmark_fn(self) -> None:
        raise RuntimeError(
            "Gradient fusion executes only through its torchrun worker; "
            "use get_torchrun_spec()"
        )

    def capture_verification_payload(self) -> None:
        self.require_gradient_fusion_child_result()

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(
                iterations=1,
                warmup=5,
                measurement_timeout_seconds=300,
                nsys_nvtx_include=[OPTIMIZED_PROFILE_NVTX_RANGE],
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
            nsys_nvtx_include=[OPTIMIZED_PROFILE_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def _prepare_verification_payload(self) -> None:
        self.require_gradient_fusion_child_result()

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        nnodes = int(effective_config.nnodes or 1)
        if nnodes != 1:
            raise RuntimeError("Gradient-fusion child-result transport requires nnodes == 1")
        nproc_per_node = int(
            effective_config.nproc_per_node or torch.cuda.device_count()
        )
        env = self.prepare_gradient_fusion_child_result(
            variant="optimized",
            world_size=nproc_per_node,
            num_tensors=PAIR_NUM_TENSORS,
            tensor_kb=PAIR_TENSOR_KB,
            iterations=PAIR_TIMED_ITERATIONS,
        )
        script_path = Path(__file__).resolve().with_name("gradient_fusion_multigpu.py")
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[
                "--mode",
                "optimized",
                "--num-tensors",
                str(PAIR_NUM_TENSORS),
                "--tensor-kb",
                str(PAIR_TENSOR_KB),
                "--iterations",
                str(PAIR_TIMED_ITERATIONS),
            ],
            env=env,
            multi_gpu_required=True,
            name="optimized_gradient_fusion_multigpu",
            result_callback=GRADIENT_FUSION_RESULT_CALLBACK,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=PAIR_TIMED_ITERATIONS,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedGradientFusionMultiGPU()
