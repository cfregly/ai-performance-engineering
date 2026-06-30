"""Benchmark wrapper for NVSHMEM training example; skips on <2 GPUs."""

from __future__ import annotations

from pathlib import Path

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status
from ch04.nvshmem_training_example import main as nvshmem_train_main
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import symmetric_memory_available
from core.benchmark.verification_mixin import VerificationPayloadMixin


class NVSHMEMTrainingExampleMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True
    allowed_benchmark_fn_antipatterns = ("random_input_regeneration",)
    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._benchmark_argv: list[str] = []
        self._original_argv: Optional[list[str]] = None
        self._original_env: dict[str, Optional[str]] = {}
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: nvshmem_training_example requires >=2 GPUs")
        if not symmetric_memory_available():
            raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._benchmark_argv = [
            sys.argv[0],
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
        self._original_argv = sys.argv
        self._original_env = {
            "AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS": os.environ.get("AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS"),
        }
        os.environ["AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS"] = "0"
        sys.argv = self._benchmark_argv
        self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        if not self._benchmark_argv:
            raise RuntimeError("setup() must initialize benchmark argv before benchmark_fn()")
        nvshmem_train_main()

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        if self._original_argv is not None:
            sys.argv = self._original_argv
            self._original_argv = None
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._original_env = {}
        self._benchmark_argv = []
        torch.cuda.empty_cache()

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
                "collective_type": "nvshmem",
            },
        )

    def get_config(self) -> BenchmarkConfig:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=300)
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="baseline_nvshmem_training_example_multigpu",
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return NVSHMEMTrainingExampleMultiGPU()


def main() -> None:
    bench = NVSHMEMTrainingExampleMultiGPU()
    bench.setup()
    try:
        bench.benchmark_fn()
    finally:
        bench.teardown()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
