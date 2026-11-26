"""Harness integration for the equivalent serialized and overlapped native solves."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from core.benchmark.cuda_binary_benchmark import (
    BinaryRunResult,
    CudaBinaryBenchmark,
    detect_supported_arch,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.cooperative_process_artifacts import (
    require_owned_uncapped_native_config,
    run_recorded_cooperative_process,
)
from labs.multi_gpu_communication.native_common import (
    LAB_DIR,
    NativeUnavailable,
    NativeWorkload,
    invoke_native,
    require_unprofiled_native_config,
)


class NativeJacobiBenchmark(CudaBinaryBenchmark):
    """Use maximum-rank native timing and retain every element of the timed output."""

    multi_gpu_required = True
    supports_uncapped_execution = True

    def __init__(self, *, overlapped: bool) -> None:
        super().__init__(
            chapter_dir=LAB_DIR,
            binary_name="jacobi_native",
            friendly_name="Overlapped native Jacobi" if overlapped else "Serialized native Jacobi",
            iterations=3,
            warmup=1,
            workload_params={"dtype": "float32", "nx": 256, "ny": 257, "steps": 20, "ranks": 2},
        )
        self.workload = NativeWorkload(schedule="overlap" if overlapped else "serialized")
        self._directory: Path | None = None
        self._invocations = 0
        self._output: torch.Tensor | None = None
        self._native_result: dict | None = None
        self._input_parameters: torch.Tensor | None = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _build_architecture(self) -> str:
        return detect_supported_arch()

    def setup(self) -> None:
        config = getattr(self, "_config", None)
        require_owned_uncapped_native_config(config, launcher_name="Native Jacobi harness execution")
        self.workload.validate()
        require_unprofiled_native_config(config)
        if shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").is_file():
            raise NativeUnavailable("SKIPPED: The native communication lab requires nvcc")
        if shutil.which("mpicxx") is None or shutil.which("mpirun") is None:
            raise NativeUnavailable("SKIPPED: The native communication lab requires mpicxx and mpirun")
        if self.workload.backend == "nvshmem" and not os.environ.get("NVSHMEM_HOME"):
            raise NativeUnavailable("SKIPPED: Set NVSHMEM_HOME for the explicit native NVSHMEM build")
        architecture = self._build_architecture()
        configured_output = getattr(config, "profiling_output_dir", None)
        artifact_root = (
            Path(configured_output)
            if configured_output
            else LAB_DIR.parents[1] / "artifacts" / "multi_gpu_communication"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(
            tempfile.mkdtemp(prefix=f"{self.workload.schedule}_", dir=artifact_root)
        )
        command = [
            "make",
            "-C",
            str(LAB_DIR),
            f"ARCH={architecture}",
            "NVTX_ENABLED=1",
            "WITH_NCCL=1",
            f"WITH_NVSHMEM={int(self.workload.backend == 'nvshmem')}",
            "jacobi_native",
        ]
        completed = run_recorded_cooperative_process(
            command,
            output_prefix=self._directory / "native_build",
        )
        if completed["returncode"] != 0:
            raise RuntimeError(
                f"Native build failed\n{completed['stdout']}\n{completed['stderr']}"
            )
        self.exec_path = LAB_DIR / "jacobi_native"
        self._input_parameters = torch.tensor(
            [self.workload.nx, self.workload.ny, self.workload.iterations, self.workload.ranks],
            dtype=torch.int64,
        )

    def _run_once(self) -> BinaryRunResult:
        require_owned_uncapped_native_config(
            getattr(self, "_config", None), launcher_name="Native Jacobi harness execution"
        )
        if self._directory is None or self.exec_path is None:
            raise RuntimeError("setup() must complete before running native Jacobi")
        self._invocations += 1
        directory = self._directory
        output_path = directory / f"output_{self._invocations}.f32"
        result = invoke_native(
            self.workload,
            binary=self.exec_path,
            json_out=directory / f"result_{self._invocations}.json",
            dump_output=output_path,
        )
        output = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=torch.float32).clone()
        if output.numel() != self.workload.nx * self.workload.ny or not torch.isfinite(output).all():
            raise RuntimeError("Native timed output is incomplete or nonfinite")
        self._output = output.reshape(self.workload.ny, self.workload.nx)
        self._native_result = result
        return BinaryRunResult(time_ms=result["time_per_iteration_ms"], raw_stdout="", raw_stderr="")

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input_parameters is None:
            raise RuntimeError("A timed native solve must finish before verification capture")
        self._set_verification_payload(
            inputs={"workload_parameters": self._input_parameters},
            output=self._output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 2e-6),
            signature_overrides={"world_size": self.workload.ranks, "collective_type": "jacobi_halo"},
        )

    def get_verify_output(self) -> torch.Tensor:
        return VerificationPayloadMixin.get_verify_output(self)

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return VerificationPayloadMixin.get_verify_inputs(self)

    def get_output_tolerance(self) -> tuple[float, float]:
        return 0.0, 2e-6

    def get_ncu_profile_command(self) -> tuple[list[str], str]:
        raise NativeUnavailable(
            "SKIPPED: Native multi-rank NCU collection needs a validated launch that preserves collective progress. "
            "This wrapper supports --profile none"
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.iterations,
            warmup=self.warmup,
            device=torch.device("cuda"),
            multi_gpu_required=True,
            enable_memory_tracking=False,
            adaptive_iterations=False,
            cross_validate_timing=False,
            profile_type="none",
        )

    def get_custom_metrics(self) -> dict[str, float] | None:
        if self._native_result is None:
            return None
        return {f"native.{name}": float(self._native_result[name]) for name in (
            "setup_ms", "allocation_ms", "registration_ms", "capture_ms", "warmup_ms",
            "graph_warmup_ms", "steady_ms", "verification_ms", "cleanup_ms", "max_abs_error",
        )}

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--nx", type=int, default=self.workload.nx)
        parser.add_argument("--ny", type=int, default=self.workload.ny)
        parser.add_argument("--steps", type=int, default=self.workload.iterations)
        parser.add_argument("--native-warmup", type=int, default=self.workload.warmup)
        parser.add_argument("--ranks", type=int, default=self.workload.ranks)
        parser.add_argument("--backend", choices=("mpi", "nccl", "nvshmem"), default=self.workload.backend)
        parser.add_argument("--registered-buffers", action="store_true")
        parser.add_argument("--graph", action="store_true")
        parser.add_argument("--assert-cuda-aware-mpi", action="store_true")
        args = parser.parse_args(argv)
        self.workload = replace(
            self.workload, nx=args.nx, ny=args.ny, iterations=args.steps, warmup=args.native_warmup,
            ranks=args.ranks, backend=args.backend, registered_buffers=args.registered_buffers,
            graph=args.graph, assert_cuda_aware_mpi=args.assert_cuda_aware_mpi,
        )
        self.workload.validate()
        self._workload_params = {
            "dtype": "float32", "nx": self.workload.nx, "ny": self.workload.ny,
            "steps": self.workload.iterations, "ranks": self.workload.ranks,
            "backend_kind": {"mpi": 0, "nccl": 1, "nvshmem": 2}[self.workload.backend],
            "registered": int(self.workload.registered_buffers), "graph": int(self.workload.graph),
        }

    def teardown(self) -> None:
        self._directory = None
