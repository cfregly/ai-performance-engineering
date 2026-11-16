"""Compare host-submitted and fused device epochs with identical payloads."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from core.benchmark.cuda_binary_benchmark import BinaryRunResult, CudaBinaryBenchmark, detect_supported_arch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BenchmarkConfig
from labs.multi_gpu_communication.native_common import require_unprofiled_native_config
from labs.nvshmem_device.run_lab_nvshmem_device import (
    LAB_DIR, NativeUnavailable, NativeWorkload, invoke_native,
)


class NvshmemDeviceBenchmark(CudaBinaryBenchmark):
    """Use completed native epoch timing and capture every received element."""

    multi_gpu_required = True

    def __init__(self, *, persistent: bool) -> None:
        super().__init__(
            chapter_dir=LAB_DIR,
            binary_name="nvshmem_device_native",
            friendly_name="Persistent NVSHMEM signal exchange" if persistent else "Block NVSHMEM signal exchange",
            iterations=3,
            warmup=5,
            workload_params={"dtype": "float32", "elements": 263, "steps": 32, "ranks": 2},
        )
        self.workload = NativeWorkload(
            mode="persistent-signal" if persistent else "block-signal",
            elements=263, iterations=32, warmup=2,
        )
        self._directory: Path | None = None
        self._invocations = 0
        self._output: torch.Tensor | None = None
        self._parameters: torch.Tensor | None = None
        self._native_result: dict | None = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        self.workload.validate()
        config = getattr(self, "_config", None)
        require_unprofiled_native_config(config)
        if not os.environ.get("NVSHMEM_HOME"):
            raise NativeUnavailable("SKIPPED: Set NVSHMEM_HOME for the device NVSHMEM build")
        for tool in ("nvcc", "mpicxx", "mpirun"):
            if shutil.which(tool) is None:
                raise NativeUnavailable(f"SKIPPED: Device NVSHMEM requires {tool}")
        command = ["make", f"ARCH={detect_supported_arch()}", "NVTX_ENABLED=1", "nvshmem_device_native"]
        built = subprocess.run(command, cwd=LAB_DIR, capture_output=True, text=True, check=False)
        if built.returncode:
            raise RuntimeError(f"Device NVSHMEM build failed\n{built.stdout}\n{built.stderr}")
        self.exec_path = LAB_DIR / "nvshmem_device_native"
        configured = getattr(config, "profiling_output_dir", None)
        artifact_root = Path(configured) if configured else LAB_DIR.parents[1] / "artifacts" / "nvshmem_device"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix=self.workload.mode + "_", dir=artifact_root))
        self._parameters = torch.tensor(
            [self.workload.elements, self.workload.iterations, self.workload.ranks], dtype=torch.int64,
        )

    def _run_once(self) -> BinaryRunResult:
        if self._directory is None or self.exec_path is None:
            raise RuntimeError("setup() must complete before native execution")
        self._invocations += 1
        output_path = self._directory / f"output_{self._invocations}.f32"
        result = invoke_native(
            self.workload, binary=self.exec_path,
            json_out=self._directory / f"result_{self._invocations}.json", dump_output=output_path,
        )
        output = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=torch.float32).clone()
        shape = (self.workload.ranks, self.workload.iterations, 2, self.workload.elements)
        if output.numel() != self.workload.expected_output_elements or not torch.isfinite(output).all():
            raise RuntimeError("The timed received history is incomplete or nonfinite")
        self._output = output.reshape(shape)
        self._native_result = result
        return BinaryRunResult(time_ms=result["time_per_iteration_ms"], raw_stdout="", raw_stderr="")

    def capture_verification_payload(self) -> None:
        if self._output is None or self._parameters is None:
            raise RuntimeError("A timed solve must finish before verification capture")
        self._set_verification_payload(
            inputs={"payload_parameters": self._parameters}, output=self._output,
            batch_size=1, parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
            signature_overrides={"world_size": self.workload.ranks, "collective_type": "nvshmem_neighbor_tiles"},
        )

    def get_verify_output(self) -> torch.Tensor:
        return VerificationPayloadMixin.get_verify_output(self)

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return VerificationPayloadMixin.get_verify_inputs(self)

    def get_output_tolerance(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_ncu_profile_command(self) -> tuple[list[str], str]:
        raise NativeUnavailable(
            "SKIPPED: Device synchronization requires whole-MPI profiling. "
            "Use the lab's validated native profile commands and --profile none for this wrapper"
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.iterations, warmup=self.warmup, device=torch.device("cuda"),
            multi_gpu_required=True, enable_memory_tracking=False, adaptive_iterations=False,
            cross_validate_timing=False, enable_profiling=False, profile_type="none",
        )

    def get_custom_metrics(self) -> dict[str, float] | None:
        if self._native_result is None:
            return None
        return {f"native.{name}": float(self._native_result[name]) for name in (
            "setup_ms", "allocation_ms", "warmup_ms", "reset_ms", "steady_ms",
            "verification_ms", "cleanup_ms", "max_abs_error",
        )}

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--elements", type=int, default=self.workload.elements)
        parser.add_argument("--steps", type=int, default=self.workload.iterations)
        parser.add_argument("--native-warmup", type=int, default=self.workload.warmup)
        parser.add_argument("--ranks", type=int, default=self.workload.ranks)
        args = parser.parse_args(argv)
        self.workload = replace(self.workload, elements=args.elements, iterations=args.steps,
                                warmup=args.native_warmup, ranks=args.ranks)
        self.workload.validate()
        self._workload_params = {"dtype": "float32", "elements": self.workload.elements,
                                 "steps": self.workload.iterations, "ranks": self.workload.ranks}

    def teardown(self) -> None:
        self._directory = None

