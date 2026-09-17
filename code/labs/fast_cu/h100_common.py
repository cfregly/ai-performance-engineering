"""Shared runtime gates, JIT loaders, and benchmark contracts for fast.cu."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.fast_cu.build import load_cuda_extension

_LAB_DIR = Path(__file__).resolve().parent

H100_CAPABILITY = (9, 0)
REDUCTION_CAPABILITIES = frozenset({H100_CAPABILITY, (10, 0)})

GEMM_M = 8192
GEMM_N = 8192
GEMM_K = 8192
REDUCTION_ELEMENTS = 1 << 28


def ensure_h100_capability_supported(
    capability: tuple[int, int], *, module_name: str = "fast.cu H100 BF16 GEMM"
) -> None:
    """Require the exact Hopper capability targeted by the upstream WGMMA kernel."""
    if capability == H100_CAPABILITY:
        return
    major, minor = capability
    raise RuntimeError(
        f"SKIPPED: {module_name} requires exact SM 9.0 (sm_90a); got SM {major}.{minor}."
    )


def ensure_reduction_capability_supported(
    capability: tuple[int, int], *, module_name: str = "fast.cu int32 reduction"
) -> None:
    """Require an architecture explicitly compiled and exercised by this adapter."""
    if capability in REDUCTION_CAPABILITIES:
        return
    major, minor = capability
    raise RuntimeError(
        f"SKIPPED: {module_name} supports exact SM 9.0 (H100) and SM 10.0 (B200); "
        f"got SM {major}.{minor}."
    )


def _visible_capability(module_name: str) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"SKIPPED: {module_name} requires an NVIDIA CUDA GPU")
    return tuple(torch.cuda.get_device_capability())


def ensure_h100_gemm_supported() -> tuple[int, int]:
    capability = _visible_capability("fast.cu H100 BF16 GEMM")
    ensure_h100_capability_supported(capability)
    return capability


def ensure_int32_reduction_supported() -> tuple[int, int]:
    capability = _visible_capability("fast.cu int32 reduction")
    ensure_reduction_capability_supported(capability)
    return capability


def _gemm_cuda_flags(capability: tuple[int, int]) -> list[str]:
    ensure_h100_capability_supported(capability)
    return [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "--expt-relaxed-constexpr",
        # torch cpp_extension disables this conversion by default, but pinned
        # matmul_12.cuh converts FP32 accumulators to BF16 at its store epilogue.
        # Extra CUDA flags follow torch's defaults, so this narrowly restores
        # the constructor used by the upstream conversion code.
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-gencode=arch=compute_90a,code=sm_90a",
        f"-I{_LAB_DIR}",
    ]


def _reduction_build_contract(
    capability: tuple[int, int],
) -> tuple[list[str], tuple[int, int]]:
    ensure_reduction_capability_supported(capability)
    if capability == H100_CAPABILITY:
        architecture = "90a"
        minimum_cuda = (12, 0)
    else:
        architecture = "100a"
        minimum_cuda = (12, 8)
    return (
        [
            "-O3",
            "-std=c++17",
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
        ],
        minimum_cuda,
    )


@lru_cache(maxsize=1)
def load_h100_gemm_extension() -> ModuleType:
    capability = ensure_h100_gemm_supported()
    return load_cuda_extension(
        "fast_cu_h100_gemm",
        [_LAB_DIR / "h100_gemm_extension.cu"],
        extra_cuda_cflags=_gemm_cuda_flags(capability),
        extra_ldflags=["-lcublas", "-lcuda"],
        minimum_cuda=(12, 0),
    )


@lru_cache(maxsize=2)
def _load_reduction_extension(capability: tuple[int, int]) -> ModuleType:
    flags, minimum_cuda = _reduction_build_contract(capability)
    return load_cuda_extension(
        "fast_cu_int32_reduction",
        [_LAB_DIR / "h100_reduction_kernels.cu"],
        extra_cuda_cflags=flags,
        extra_ldflags=[],
        minimum_cuda=minimum_cuda,
    )


def load_int32_reduction_extension() -> ModuleType:
    return _load_reduction_extension(ensure_int32_reduction_supported())


class H100Bf16GemmBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Common full-output contract for A[M,K] @ B[N,K].T."""

    allow_cpu = False
    matrix_rows = GEMM_M
    matrix_cols = GEMM_N
    shared_dim = GEMM_K
    # Upstream's own full-matrix check uses an absolute 0.1 bound. Retain that
    # bound here; target-H100 qualification must establish any future change.
    output_tolerance = (1e-2, 1e-1)

    def __init__(self) -> None:
        super().__init__()
        self.matrix_a: torch.Tensor | None = None
        self.matrix_b: torch.Tensor | None = None
        self._physical_output: torch.Tensor | None = None
        self._logical_output: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.extension: ModuleType | None = None
        element_size = torch.finfo(torch.bfloat16).bits // 8
        self.register_workload_metadata(
            bytes_per_iteration=float(
                (
                    self.matrix_rows * self.shared_dim
                    + self.matrix_cols * self.shared_dim
                    + self.matrix_rows * self.matrix_cols
                )
                * element_size
            ),
            custom_units_per_iteration=float(
                2 * self.matrix_rows * self.matrix_cols * self.shared_dim
            ),
            custom_unit_name="FLOPs",
        )

    def _clear_runtime_state(self) -> None:
        """Release tensors retained directly or through the verification payload."""
        self._verification_payload = None
        self.matrix_a = None
        self.matrix_b = None
        self._physical_output = None
        self._logical_output = None
        self.output = None
        self.extension = None

    def _setup_tensors(self) -> None:
        # VerifyRunner can set up the same instance again with fresh inputs. Drop
        # the previous full payload before allocating another multi-GiB workload.
        self._clear_runtime_state()
        ensure_h100_gemm_supported()
        self.extension = load_h100_gemm_extension()
        # The harness owns the active RNG seed. Do not reset it here.
        self.matrix_a = torch.empty(
            (self.matrix_rows, self.shared_dim),
            device=self.device,
            dtype=torch.bfloat16,
        ).uniform_(-1.0, 1.0)
        self.matrix_b = torch.empty(
            (self.matrix_cols, self.shared_dim),
            device=self.device,
            dtype=torch.bfloat16,
        ).uniform_(-1.0, 1.0)
        # The upstream TMA store writes C physically as [N, M]. The transpose is
        # a metadata-only logical [M, N] view and is shared by both variants.
        self._physical_output = torch.empty(
            (self.matrix_cols, self.matrix_rows),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._logical_output = self._physical_output.transpose(0, 1)
        self.output = None

    def capture_verification_payload(self) -> None:
        if self.matrix_a is None or self.matrix_b is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"matrix_a": self.matrix_a, "matrix_b": self.matrix_b},
            output=self.output,
            batch_size=self.matrix_rows,
            parameter_count=self.matrix_a.numel() + self.matrix_b.numel(),
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": False},
            output_tolerance=self.output_tolerance,
        )

    def get_input_signature(self) -> InputSignature:
        return InputSignature(
            shapes={
                "matrix_a": (self.matrix_rows, self.shared_dim),
                "matrix_b": (self.matrix_cols, self.shared_dim),
                "output": (self.matrix_rows, self.matrix_cols),
            },
            dtypes={
                "matrix_a": str(torch.bfloat16),
                "matrix_b": str(torch.bfloat16),
                "output": str(torch.bfloat16),
            },
            batch_size=self.matrix_rows,
            parameter_count=(
                self.matrix_rows * self.shared_dim + self.matrix_cols * self.shared_dim
            ),
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if self.output.shape != (self.matrix_rows, self.matrix_cols):
            return f"unexpected output shape: {tuple(self.output.shape)}"
        if self.output.dtype is not torch.bfloat16:
            return f"unexpected output dtype: {self.output.dtype}"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5, use_subprocess=False)

    def teardown(self) -> None:
        self._clear_runtime_state()
        super().teardown()


class Int32ReductionBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Common overflow-safe input and full scalar-output reduction contract."""

    allow_cpu = False
    element_count = REDUCTION_ELEMENTS

    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.extension: ModuleType | None = None
        self.register_workload_metadata(
            bytes_per_iteration=float(self.element_count * torch.iinfo(torch.int32).bits // 8),
            custom_units_per_iteration=float(self.element_count),
            custom_unit_name="elements",
        )

    def _clear_runtime_state(self) -> None:
        """Release tensors retained directly or through the verification payload."""
        self._verification_payload = None
        self.input = None
        self._output_buffer = None
        self.output = None
        self.extension = None

    def _setup_tensors(self) -> None:
        # Verification may repeat setup on this instance with a fresh input.
        # Clear the previous 1 GiB input payload before allocating its successor.
        self._clear_runtime_state()
        ensure_int32_reduction_supported()
        self.extension = load_int32_reduction_extension()
        # Caller-owned RNG, with a bounded domain: every possible sum fits int32,
        # so both CUB and the upstream-derived atomic tree have defined arithmetic.
        self.input = torch.randint(
            -1,
            2,
            (self.element_count,),
            device=self.device,
            dtype=torch.int32,
        )
        self._output_buffer = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.output = None

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=self.element_count,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def get_input_signature(self) -> InputSignature:
        return InputSignature(
            shapes={"input": (self.element_count,), "output": (1,)},
            dtypes={"input": str(torch.int32), "output": str(torch.int32)},
            batch_size=self.element_count,
            parameter_count=0,
            precision_flags=PrecisionFlags(tf32=False),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if self.output.shape != (1,):
            return f"unexpected output shape: {tuple(self.output.shape)}"
        if self.output.dtype is not torch.int32:
            return f"unexpected output dtype: {self.output.dtype}"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10, use_subprocess=False)

    def teardown(self) -> None:
        self._clear_runtime_state()
        super().teardown()


__all__ = [
    "GEMM_K",
    "GEMM_M",
    "GEMM_N",
    "H100Bf16GemmBenchmarkBase",
    "H100_CAPABILITY",
    "Int32ReductionBenchmarkBase",
    "REDUCTION_CAPABILITIES",
    "REDUCTION_ELEMENTS",
    "ensure_h100_capability_supported",
    "ensure_h100_gemm_supported",
    "ensure_int32_reduction_supported",
    "ensure_reduction_capability_supported",
    "load_h100_gemm_extension",
    "load_int32_reduction_extension",
]
