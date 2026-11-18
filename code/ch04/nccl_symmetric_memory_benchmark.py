"""Harness integration for the native NCCL symmetric-memory AllReduce pair."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from ch04.run_nccl_symmetric_memory import (
    CHAPTER_DIR,
    NativeUnavailableError,
    NativeWorkload,
    invoke_native,
)
from core.benchmark.cuda_binary_benchmark import (
    BinaryRunResult,
    CudaBinaryBenchmark,
    detect_supported_arch,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.cooperative_process_artifacts import require_owned_uncapped_native_config
from core.utils.cooperative_subprocess import run_cooperative_process


class NcclSymmetricMemoryBenchmark(CudaBinaryBenchmark):
    """Return native MPI_MAX timing while retaining every output element."""

    multi_gpu_required = True
    supports_uncapped_execution = True

    def __init__(self, *, arm: str) -> None:
        if arm not in {"cuda_malloc", "window_register"}:
            raise ValueError("The public harness pair supports cuda_malloc and window_register")
        name = (
            "Symmetric window NCCL AllReduce"
            if arm == "window_register"
            else "cudaMalloc NCCL AllReduce"
        )
        super().__init__(
            chapter_dir=CHAPTER_DIR,
            binary_name="nccl_symmetric_memory_native",
            friendly_name=name,
            iterations=3,
            warmup=1,
            workload_params={
                "dtype": "float32",
                "elements": 16 * 1024 * 1024,
                "ranks": 2,
                "native_iterations": 20,
            },
        )
        self.arm = arm
        self.elements = 16 * 1024 * 1024
        self.native_iterations = 20
        self.native_warmup = 5
        self.requested_ranks: int | None = None
        self.launcher = "mpirun"
        self.launcher_args: tuple[str, ...] = ()
        self.workload: NativeWorkload | None = None
        self._directory: Path | None = None
        self._invocations = 0
        self._output: torch.Tensor | None = None
        self._input_parameters: torch.Tensor | None = None
        self._native_result: dict | None = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    @staticmethod
    def _require_unprofiled_config(config: object) -> None:
        if (
            getattr(config, "enable_profiling", False)
            or getattr(config, "profile_mode", "none") not in {None, "none"}
            or getattr(config, "profile_type", "none") not in {None, "none"}
        ):
            raise NativeUnavailableError(
                "SKIPPED: The generic harness profiler cannot launch this native MPI child. "
                "Use --profile none here. Run ch04/run_nccl_symmetric_memory.py with --profile nsys or --profile ncu for per-rank captures."
            )

    def setup(self) -> None:
        config = getattr(self, "_config", None)
        require_owned_uncapped_native_config(
            config, launcher_name="NCCL symmetric-memory harness execution"
        )
        self._require_unprofiled_config(config)
        if not torch.cuda.is_available():
            raise NativeUnavailableError("SKIPPED: The NCCL symmetric-memory pair requires CUDA")
        visible_gpus = torch.cuda.device_count()
        ranks = self.requested_ranks if self.requested_ranks is not None else visible_gpus
        if ranks < 2:
            raise NativeUnavailableError("SKIPPED: At least two visible GPUs are required")
        if ranks > visible_gpus:
            raise NativeUnavailableError("SKIPPED: --ranks exceeds the visible CUDA device count")
        self.workload = NativeWorkload(
            arm=self.arm,
            ranks=ranks,
            elements=self.elements,
            iterations=self.native_iterations,
            warmup=self.native_warmup,
        )
        self.workload.validate()
        if shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").is_file():
            raise NativeUnavailableError(
                "SKIPPED: The native NCCL symmetric-memory pair requires nvcc"
            )
        if shutil.which("mpicxx") is None or shutil.which(self.launcher) is None:
            raise NativeUnavailableError(
                "SKIPPED: The native NCCL symmetric-memory pair requires mpicxx and an MPI launcher"
            )

        architecture = detect_supported_arch()
        command = [
            "make",
            f"ARCH={architecture}",
            "NVTX_ENABLED=1",
            "nccl_symmetric_memory_native",
        ]
        completed = run_cooperative_process(command, cwd=CHAPTER_DIR)
        if completed["returncode"] != 0:
            raise RuntimeError(
                "Native NCCL symmetric-memory build failed\n"
                f"{completed['stdout']}\n{completed['stderr']}"
            )
        self.exec_path = CHAPTER_DIR / "nccl_symmetric_memory_native"
        configured_output = getattr(config, "profiling_output_dir", None)
        artifact_root = (
            Path(configured_output)
            if configured_output
            else CHAPTER_DIR.parent / "artifacts" / "nccl_symmetric_memory"
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix=self.arm + "_", dir=artifact_root))
        self._input_parameters = torch.tensor(
            [self.elements, ranks, self.native_iterations], dtype=torch.int64
        )
        self._workload_params = {
            "dtype": "float32",
            "elements": self.elements,
            "ranks": ranks,
            "native_iterations": self.native_iterations,
        }

    def _run_once(self) -> BinaryRunResult:
        require_owned_uncapped_native_config(
            getattr(self, "_config", None),
            launcher_name="NCCL symmetric-memory harness execution",
        )
        if self._directory is None or self.exec_path is None or self.workload is None:
            raise RuntimeError("setup() must complete before the native AllReduce runs")
        self._invocations += 1
        output_path = self._directory / f"output_{self._invocations}.f32"
        result = invoke_native(
            self.workload,
            binary=self.exec_path,
            json_out=self._directory / f"result_{self._invocations}.json",
            dump_output=output_path,
            launcher=self.launcher,
            launcher_args=self.launcher_args,
        )
        output = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=torch.float32).clone()
        if output.numel() != self.workload.elements or not torch.isfinite(output).all():
            raise RuntimeError("Native timed output is incomplete or nonfinite")
        self._output = output
        self._native_result = result
        return BinaryRunResult(
            time_ms=result["time_per_iteration_ms"], raw_stdout="", raw_stderr=""
        )

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input_parameters is None or self.workload is None:
            raise RuntimeError("A timed native AllReduce must finish before verification capture")
        self._set_verification_payload(
            inputs={"workload_parameters": self._input_parameters},
            output=self._output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": self.workload.ranks,
                "collective_type": "out_of_place_sum_allreduce",
            },
        )

    def get_verify_output(self) -> torch.Tensor:
        return VerificationPayloadMixin.get_verify_output(self)

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return VerificationPayloadMixin.get_verify_inputs(self)

    def get_output_tolerance(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_ncu_profile_command(self) -> tuple[list[str], str]:
        raise NativeUnavailableError(
            "SKIPPED: Native multi-rank NCU collection needs a validated per-rank launch. "
            "This harness pair supports --profile none"
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
        metrics = {
            f"native.{name}": float(self._native_result[name])
            for name in (
                "setup_ms",
                "topology_gate_ms",
                "communicator_ms",
                "allocation_ms",
                "registration_ms",
                "initialization_ms",
                "warmup_ms",
                "verification_ms",
                "output_write_ms",
                "cleanup_ms",
                "max_abs_error",
            )
        }
        metrics.update(
            {
                f"native.{name}_ms": float(self._native_result["statistics_ms"][name])
                for name in ("min", "max", "mean", "median", "p95", "stddev")
            }
        )
        return metrics

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--elements", type=int, default=self.elements)
        parser.add_argument("--native-iterations", type=int, default=self.native_iterations)
        parser.add_argument("--native-warmup", type=int, default=self.native_warmup)
        parser.add_argument("--ranks", type=int)
        parser.add_argument("--launcher", default=self.launcher)
        parser.add_argument("--launcher-arg", action="append", default=[])
        args = parser.parse_args(argv)
        validation_ranks = args.ranks if args.ranks is not None else 2
        NativeWorkload(
            arm=self.arm,
            ranks=validation_ranks,
            elements=args.elements,
            iterations=args.native_iterations,
            warmup=args.native_warmup,
        ).validate()
        self.elements = args.elements
        self.native_iterations = args.native_iterations
        self.native_warmup = args.native_warmup
        self.requested_ranks = args.ranks
        self.launcher = args.launcher
        self.launcher_args = tuple(args.launcher_arg)
        self._workload_params = {
            "dtype": "float32",
            "elements": self.elements,
            "ranks": validation_ranks,
            "native_iterations": self.native_iterations,
        }

    def teardown(self) -> None:
        self._directory = None
        self.workload = None
        self._output = None
        self._input_parameters = None
        self._native_result = None
