"""Optimized gradient fusion benchmark (fused all-reduce)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from ch04.gradient_fusion_multigpu import OPTIMIZED_PROFILE_NVTX_RANGE
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


class OptimizedGradientFusionMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: gradient_fusion requires >=2 GPUs")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        if self._verify_input is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)
        output = self._verify_input + 1.0
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
            signature_overrides={
                "world_size": torch.cuda.device_count(),
                "collective_type": "all_reduce",
            },
        )

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(
                iterations=1,
                warmup=5,
                measurement_timeout_seconds=300,
                nsys_nvtx_include=[OPTIMIZED_PROFILE_NVTX_RANGE],
            )
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
            nsys_nvtx_include=[OPTIMIZED_PROFILE_NVTX_RANGE],
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.capture_verification_payload()
        self._subprocess_verify_output = self.get_verify_output()
        self._subprocess_output_tolerance = self.get_output_tolerance()
        self._subprocess_input_signature = self.get_input_signature()

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        script_path = Path(__file__).resolve().with_name("gradient_fusion_multigpu.py")
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[
                "--mode",
                "optimized",
                "--num-tensors",
                "2048",
                "--tensor-kb",
                "4",
                "--iterations",
                "50",
            ],
            multi_gpu_required=True,
            name="optimized_gradient_fusion_multigpu",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedGradientFusionMultiGPU()
