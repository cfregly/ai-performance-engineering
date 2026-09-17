"""Native harness adapter for fast.cu's GB300 NVFP4 optimization ladder."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.fast_cu.build import load_cuda_extension, parse_nvcc_release
from labs.fast_cu.source import UPSTREAM_DIR, verify_upstream

DEFAULT_M = 8192
DEFAULT_N = 8192
DEFAULT_K = 8192
DEFAULT_RUNG = 9
CUDA_VERSION = (13, 1)
COMPUTE_CAPABILITY = (10, 3)
OUTPUT_RTOL = 2e-3
OUTPUT_ATOL = 0.5
_NATIVE_SOURCE = Path(__file__).with_name("nvfp4_native.cu")
_UPSTREAM_NVFP4_DIR = UPSTREAM_DIR / "gb300" / "nvfp4"


@dataclass(frozen=True)
class Nvfp4Workload:
    """The exact MxNxK workload shared by both comparison arms."""

    m: int = DEFAULT_M
    n: int = DEFAULT_N
    k: int = DEFAULT_K

    def validate(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.k % 32:
            raise ValueError(
                "k must be divisible by 32 because the cuBLASLt NVFP4 baseline "
                "has no supported algorithm otherwise"
            )


def validate_rung(rung: int) -> int:
    if type(rung) is not int or not 0 <= rung <= 9:
        raise ValueError("fast.cu NVFP4 rung must be an integer in [0, 9]")
    return rung


def workload_bytes(workload: Nvfp4Workload) -> int:
    """Packed A/B + VEC16 scales read and full FP16 C written per launch."""
    workload.validate()
    ab_bytes = (workload.m + workload.n) * (workload.k // 2)
    packed_scale_inner = ((workload.k + 63) // 64) * 4
    packed_scale_rows = ((workload.m + 127) // 128) * 128 + ((workload.n + 127) // 128) * 128
    scale_bytes = packed_scale_inner * packed_scale_rows
    output_bytes = workload.m * workload.n * 2
    return ab_bytes + scale_bytes + output_bytes


def workload_flops(workload: Nvfp4Workload) -> int:
    """Conventional GEMM operation count: one multiply and add per K."""
    workload.validate()
    return 2 * workload.m * workload.n * workload.k


def _version_tuple(version: str | None) -> tuple[int, int] | None:
    if not isinstance(version, str):
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", version)
    if match is None:
        return None
    return int(match[1]), int(match[2])


def require_nvfp4_runtime() -> int:
    """Reject runtimes older than CUDA 13.1 and every GPU except SM103."""
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: fast.cu NVFP4 requires a CUDA 13.1 GB300/B300 (SM103) GPU")
    runtime = _version_tuple(torch.version.cuda)
    if runtime is None or runtime < CUDA_VERSION:
        found = torch.version.cuda if torch.version.cuda is not None else "none"
        raise RuntimeError(
            f"SKIPPED: fast.cu NVFP4 requires CUDA runtime 13.1 or newer; found {found}"
        )
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability != COMPUTE_CAPABILITY:
        raise RuntimeError(
            "SKIPPED: fast.cu NVFP4 requires exact SM103 GB300/B300; "
            f"found sm_{capability[0]}{capability[1]} on "
            f"{torch.cuda.get_device_name(device)}"
        )
    return device


def require_nvfp4_toolkit() -> Path:
    """Require nvcc 13.1+ before starting the content-keyed JIT build."""
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError("SKIPPED: fast.cu NVFP4 requires CUDA toolkit 13.1 with nvcc")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"SKIPPED: fast.cu NVFP4 nvcc is missing: {nvcc}")
    completed = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version = parse_nvcc_release(completed.stdout)
    if version < CUDA_VERSION:
        raise RuntimeError(
            "SKIPPED: fast.cu NVFP4 requires CUDA toolkit 13.1 or newer; "
            f"found {version[0]}.{version[1]}"
        )
    return nvcc


def load_nvfp4_extension(rung: int = DEFAULT_RUNG):
    """Verify provenance and runtime gates, then compile one explicit rung."""
    rung = validate_rung(rung)
    manifest = verify_upstream()
    hardware = manifest.get("hardware", {}).get("nvfp4", {})
    if hardware.get("compute_capability") != [10, 3] or hardware.get("minimum_cuda") != "13.1":
        raise RuntimeError("Pinned fast.cu NVFP4 hardware manifest is inconsistent")
    require_nvfp4_runtime()
    require_nvfp4_toolkit()
    return load_cuda_extension(
        f"fast_cu_nvfp4_r{rung}",
        [_NATIVE_SOURCE],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            f"-DFAST_CU_NVFP4_RUNG={rung}",
            f"-I{_UPSTREAM_NVFP4_DIR}",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        extra_ldflags=["-lcublasLt", "-lcuda"],
        minimum_cuda=CUDA_VERSION,
    )


class FastCuNvfp4Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """One arm of the native single-buffer cuBLASLt-versus-fast.cu pair.

    The r9 route table is translation-unit global. Pybind keeps the GIL for all
    context calls, setup synchronizes before replacing that table, and launch
    rejects a table configured for another shape. Concurrent native calls that
    bypass this Python/GIL boundary are unsupported.
    """

    allow_cpu = False
    multi_gpu_required = False
    required_world_size = 1

    def __init__(
        self,
        *,
        optimized: bool,
        workload: Nvfp4Workload | None = None,
        rung: int = DEFAULT_RUNG,
    ) -> None:
        super().__init__()
        self.optimized = bool(optimized)
        self.workload = workload or Nvfp4Workload()
        self.workload.validate()
        self.rung = validate_rung(rung)
        self.output: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self._native_module: Any | None = None
        self._native_context: Any | None = None
        self._inputs: dict[str, torch.Tensor] = {}
        self._caller_seed: int | None = None
        self._setup_gate_passed = False
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(workload_bytes(self.workload)),
            custom_units_per_iteration=float(workload_flops(self.workload)),
            custom_unit_name="nvfp4_flops",
        )

    def setup(self) -> None:
        self._verification_payload = None
        self.output = None
        self._output_buffer = None
        self._inputs = {}
        self._native_context = None
        self._native_module = None
        self._setup_gate_passed = False
        require_nvfp4_runtime()
        self._native_module = load_nvfp4_extension(self.rung)
        self._caller_seed = int(torch.initial_seed())
        m, n, k = self.workload.m, self.workload.n, self.workload.k
        self._output_buffer = torch.empty((m, n), device=self.device, dtype=torch.float16)
        self.output = None
        self._native_context = self._native_module.Nvfp4Context(
            m,
            n,
            k,
            self._caller_seed,
            self.optimized,
        )
        if int(self._native_context.rung) != self.rung:
            raise RuntimeError(
                f"Compiled fast.cu rung {self._native_context.rung} does not match requested r{self.rung}"
            )
        self._inputs = {
            "a_packed_e2m1": self._native_context.a_packed,
            "b_packed_e2m1": self._native_context.b_packed,
            "sfa_vec16_ue4m3": self._native_context.sfa_packed,
            "sfb_vec16_ue4m3": self._native_context.sfb_packed,
        }
        self._run_untimed_arm_gate()

    def _run_untimed_arm_gate(self) -> None:
        """Prove full overwrite and determinism on separately poisoned outputs."""
        if self._native_context is None:
            raise RuntimeError("native context is unavailable for setup validation")
        m, n = self.workload.m, self.workload.n
        first = torch.full((m, n), float("nan"), device=self.device, dtype=torch.float16)
        second = torch.full((m, n), float("nan"), device=self.device, dtype=torch.float16)
        launch = (
            self._native_context.launch_fast
            if self.optimized
            else self._native_context.launch_cublaslt
        )
        launch(first)
        launch(second)
        torch.cuda.synchronize(self.device)
        if not bool(torch.isfinite(first).all().item()) or not bool(
            torch.isfinite(second).all().item()
        ):
            raise RuntimeError("untimed NVFP4 setup gate found an incomplete or non-finite output")
        if not torch.equal(first, second):
            raise RuntimeError("untimed NVFP4 setup gate found nondeterministic output")
        if self.optimized:
            self._native_context.validate_schedule()
        self._setup_gate_passed = True

    def benchmark_fn(self) -> None:
        if self._native_context is None or self._output_buffer is None:
            raise RuntimeError("setup() must complete before benchmark_fn()")
        if self.optimized:
            self._native_context.launch_fast(self._output_buffer)
        else:
            self._native_context.launch_cublaslt(self._output_buffer)
        self.output = self._output_buffer

    def capture_verification_payload(self) -> None:
        if self._native_context is None or self.output is None or not self._inputs:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        if not self._setup_gate_passed:
            raise RuntimeError("untimed NVFP4 setup correctness gate did not pass")
        if self.optimized:
            self._native_context.validate_schedule()
        self._set_verification_payload(
            inputs=self._inputs,
            output=self.output,
            batch_size=self.workload.m,
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(OUTPUT_RTOL, OUTPUT_ATOL),
            signature_overrides={"quantization_mode": "nvfp4_e2m1_vec16_ue4m3"},
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce an output"
        if self.output.dtype != torch.float16:
            return f"expected FP16 output, found {self.output.dtype}"
        if tuple(self.output.shape) != (self.workload.m, self.workload.n):
            return f"unexpected output shape {tuple(self.output.shape)}"
        if not bool(torch.isfinite(self.output).all().item()):
            return "NVFP4 GEMM output contains non-finite values"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=25,
            warmup=5,
            min_run_time_ms=100.0,
            setup_timeout_seconds=1800,
            measurement_timeout_seconds=600,
            single_gpu=True,
            ncu_metric_set="minimal",
            ncu_replay_mode="kernel",
            ncu_replay_mode_override=True,
        )

    def get_input_signature(self):
        return super().get_input_signature()

    def get_verify_output(self):
        return super().get_verify_output()

    def get_output_tolerance(self):
        return super().get_output_tolerance()

    def teardown(self) -> None:
        self._inputs = {}
        self._native_context = None
        self._native_module = None
        self.output = None
        self._output_buffer = None
        self._verification_payload = None
        self._setup_gate_passed = False
        super().teardown()


__all__ = [
    "COMPUTE_CAPABILITY",
    "CUDA_VERSION",
    "DEFAULT_K",
    "DEFAULT_M",
    "DEFAULT_N",
    "DEFAULT_RUNG",
    "FastCuNvfp4Benchmark",
    "Nvfp4Workload",
    "load_nvfp4_extension",
    "require_nvfp4_runtime",
    "require_nvfp4_toolkit",
    "validate_rung",
    "workload_bytes",
    "workload_flops",
]
