"""SM100 epilogue store-width isolation derived from fast.cu r4/r5.

This benchmark is an FP32-accumulator-to-FP16 epilogue microbenchmark. It is
not an NVFP4 GEMM and cannot reproduce fast.cu's end-to-end headline result.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.fast_cu.build import load_cuda_extension

CUDA_SOURCE = Path(__file__).with_suffix(".cu")
DEFAULT_NUM_ELEMENTS = 1 << 24
ELEMENTS_PER_THREAD = 16
OUTPUT_ALIGNMENT_BYTES = 32
MINIMUM_CUDA = (12, 9)
EXPECTED_COMPUTE_CAPABILITY = (10, 0)


def parse_cuda_build(version: str | None) -> tuple[int, int]:
    if not isinstance(version, str):
        raise RuntimeError("SKIPPED: fast.cu B200 epilogue requires a CUDA-enabled PyTorch build")
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse PyTorch CUDA build version: {version!r}")
    return int(match[1]), int(match[2])


def validate_num_elements(num_elements: int) -> None:
    if type(num_elements) is not int or num_elements <= 0:
        raise ValueError("num_elements must be a positive integer")
    if num_elements % ELEMENTS_PER_THREAD:
        raise ValueError("num_elements must be divisible by 16")


def ensure_b200_capability_supported(capability: tuple[int, int]) -> None:
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "SKIPPED: fast.cu B200 epilogue requires exact SM100 "
            f"(compute capability 10.0); found {capability[0]}.{capability[1]}"
        )


def ensure_cuda_build_supported(cuda_build: tuple[int, int]) -> None:
    if cuda_build < MINIMUM_CUDA:
        raise RuntimeError(
            "SKIPPED: st.global.v8.b32 requires a CUDA 12.9+ PyTorch build; "
            f"found {cuda_build[0]}.{cuda_build[1]}"
        )


def require_b200_runtime(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: fast.cu B200 epilogue requires CUDA")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("SKIPPED: fast.cu B200 epilogue requires a CUDA device")
    ensure_b200_capability_supported(torch.cuda.get_device_capability(device))
    ensure_cuda_build_supported(parse_cuda_build(torch.version.cuda))


def validate_epilogue_tensors(
    accumulators: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not isinstance(accumulators, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("accumulators and output must be torch.Tensor instances")
    if accumulators.dtype != torch.float32:
        raise TypeError("accumulators must have dtype torch.float32")
    if output.dtype != torch.float16:
        raise TypeError("output must have dtype torch.float16")
    if not accumulators.is_contiguous() or not output.is_contiguous():
        raise ValueError("accumulators and output must be contiguous")
    if accumulators.shape != output.shape:
        raise ValueError("accumulators and output must have matching shapes")
    validate_num_elements(accumulators.numel())
    if output.data_ptr() % OUTPUT_ALIGNMENT_BYTES:
        raise ValueError("output must be 32-byte aligned")
    if accumulators.device != output.device:
        raise ValueError("accumulators and output must be on the same device")
    if not accumulators.is_cuda:
        raise ValueError("accumulators and output must be CUDA tensors")


@lru_cache(maxsize=1)
def load_b200_epilogue_extension() -> Any:
    return load_cuda_extension(
        "fast_cu_b200_epilogue",
        [CUDA_SOURCE],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_100,code=sm_100",
        ],
        extra_ldflags=[],
        minimum_cuda=MINIMUM_CUDA,
    )


class B200EpilogueBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compare equal epilogues whose only device-code difference is store width."""

    def __init__(self, *, optimized: bool, num_elements: int = DEFAULT_NUM_ELEMENTS):
        if type(optimized) is not bool:
            raise TypeError("optimized must be bool")
        validate_num_elements(num_elements)
        super().__init__()
        self.optimized = optimized
        self.num_elements = num_elements
        self.accumulators: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._verification_output: torch.Tensor | None = None
        self._extension: Any = None
        self._launch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self._caller_seed: int | None = None
        self._ran = False
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            samples_per_iteration=float(num_elements),
            bytes_per_iteration=float(num_elements * (4 + 2)),
        )

    def setup(self) -> None:
        self._ran = False
        self._verification_output = None
        self._verification_payload = None
        self._output_buffer = None
        self.output = None
        require_b200_runtime(self.device)
        self._extension = load_b200_epilogue_extension()
        self._extension.require_exact_sm100()
        self._caller_seed = int(torch.initial_seed())
        self.accumulators = torch.randn(
            self.num_elements,
            device=self.device,
            dtype=torch.float32,
        )
        self._output_buffer = torch.empty(
            self.num_elements,
            device=self.device,
            dtype=torch.float16,
        )
        validate_epilogue_tensors(self.accumulators, self._output_buffer)
        self._launch = self._extension.optimized if self.optimized else self._extension.baseline

    def benchmark_fn(self) -> None:
        if self.accumulators is None or self._output_buffer is None or self._launch is None:
            raise RuntimeError("setup() must complete before benchmark_fn()")
        label = (
            "fast_cu_b200_epilogue_optimized"
            if self.optimized
            else "fast_cu_b200_epilogue_baseline"
        )
        with self._nvtx_range(label):
            self._launch(self.accumulators, self._output_buffer)
        self.output = self._output_buffer
        self._ran = True

    def capture_verification_payload(self) -> None:
        if not self._ran or self.accumulators is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        self._verification_output = self.output.clone()
        self._set_verification_payload(
            inputs={"accumulators": self.accumulators},
            output=self._verification_output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self) -> str | None:
        if not self._ran or self.accumulators is None or self.output is None:
            return "No timed epilogue output"
        expected = self.accumulators.to(torch.float16)
        if not torch.equal(self.output, expected):
            return "Epilogue output differs from full FP32-to-FP16 reference"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
            single_gpu=True,
            measurement_timeout_seconds=180,
        )

    def get_workload_metadata(self) -> WorkloadMetadata:
        return self._workload

    def get_custom_metrics(self) -> dict[str, float]:
        return {
            "fast_cu.elements_per_thread": float(ELEMENTS_PER_THREAD),
            "fast_cu.store_width_bytes": 32.0 if self.optimized else 16.0,
            "fast_cu.store_instructions_per_thread": 1.0 if self.optimized else 2.0,
            "fast_cu.full_nvfp4_gemm": 0.0,
        }

    def teardown(self) -> None:
        self.accumulators = None
        self._output_buffer = None
        self.output = None
        self._verification_output = None
        self._extension = None
        self._launch = None
        self._caller_seed = None
        self._ran = False
        self._verification_payload = None
        super().teardown()


__all__ = [
    "B200EpilogueBenchmark",
    "DEFAULT_NUM_ELEMENTS",
    "ELEMENTS_PER_THREAD",
    "EXPECTED_COMPUTE_CAPABILITY",
    "MINIMUM_CUDA",
    "ensure_b200_capability_supported",
    "ensure_cuda_build_supported",
    "load_b200_epilogue_extension",
    "parse_cuda_build",
    "require_b200_runtime",
    "validate_epilogue_tensors",
    "validate_num_elements",
]
