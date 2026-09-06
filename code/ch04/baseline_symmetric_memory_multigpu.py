"""Benchmark wrapper for symmetric memory example; skips on <2 GPUs."""

from __future__ import annotations

from pathlib import Path

import torch

from ch04.symmetric_memory_example import TRADITIONAL_RING_NVTX_RANGE
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.optimization.symmetric_memory_patch import symmetric_memory_available
from typing import Optional


class SymmetricMemoryMultiGPU(VerificationPayloadMixin, BaseBenchmark):
    multi_gpu_required = True
    preferred_ncu_replay_mode = "app-range"
    def __init__(self) -> None:
        super().__init__()
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory requires >=2 GPUs")
        if not symmetric_memory_available():
            raise RuntimeError("SKIPPED: symmetric_memory requires SymmetricMemory support")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(64, 64, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        if self._verify_input is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._verify_output = self._verify_input + 1.0

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self._verify_output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self._verify_output,
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
                "collective_type": "symmetric_memory",
            },
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=torch.cuda.device_count(),
            iterations=1,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=300,
            nsys_nvtx_include=[TRADITIONAL_RING_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        script_path = Path(__file__).resolve().with_name("symmetric_memory_example.py")
        return TorchrunLaunchSpec(
            script_path=script_path,
            script_args=[
                "--benchmark-mode",
                "traditional",
                "--tensor-bytes",
                "2097152",
                "--iterations",
                "400",
            ],
            multi_gpu_required=True,
            name="baseline_symmetric_memory_multigpu",
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
    return SymmetricMemoryMultiGPU()
