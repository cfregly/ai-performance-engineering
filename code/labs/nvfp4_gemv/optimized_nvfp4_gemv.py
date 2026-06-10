"""Benchmark harness wrapper for NVFP4 GEMV optimized submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch

LAB_DIR = Path(__file__).resolve().parent

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.nvfp4_gemv.local_eval_loader import (
    load_reference_module,
    load_submission_module,
    load_utils_module,
)


class OptimizedNvfp4GemvBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the optimized NVFP4 GEMV submission through the benchmark harness."""

    def __init__(self) -> None:
        super().__init__()
        self.m = 7168
        self.k = 2048
        self.l = 4
        self.seed = 42
        self._input_data: Optional[Any] = None
        self._kernel_fn: Optional[Callable[[Any], torch.Tensor]] = None
        self._generate_input: Optional[Callable[..., Any]] = None
        self.output: Optional[torch.Tensor] = None
        tokens = float(self.m * self.l)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA is required for NVFP4 GEMV benchmarks")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        utils_mod = load_utils_module(LAB_DIR / "utils.py", "nvfp4_gemv_utils_for_optimized")
        reference_mod = load_reference_module(
            LAB_DIR / "reference_submission.py",
            module_name="nvfp4_gemv_reference_for_optimized",
            utils_module=utils_mod,
        )
        optimized_mod = load_submission_module(
            LAB_DIR / "optimized_submission.py",
            module_name="nvfp4_gemv_optimized_submission",
            reference_module=reference_mod,
            utils_module=utils_mod,
        )
        generate_input = getattr(reference_mod, "generate_input", None)
        if not callable(generate_input):
            raise RuntimeError("reference_submission.py must expose generate_input()")
        self._generate_input = generate_input
        self._input_data = self._generate_input(
            m=self.m,
            k=self.k,
            l=self.l,
            seed=self.seed,
        )
        module = optimized_mod
        kernel_fn = getattr(module, "custom_kernel", None)
        if not callable(kernel_fn):
            raise RuntimeError("optimized_submission.py does not expose callable custom_kernel(data)")
        self._kernel_fn = kernel_fn
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._input_data is None or self._kernel_fn is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("optimized_nvfp4_gemv"):
            self.output = self._kernel_fn(self._input_data)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:64, :1, :1].float().detach().clone()
        self._set_verification_payload(
            inputs={
                "shape_signature": torch.tensor(
                    [self.m, self.k, self.l, self.seed],
                    dtype=torch.int64,
                )
            },
            output=verify_output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self._input_data = None
        self._kernel_fn = None
        self._generate_input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # The optimized submission can wedge when benchmark_fn() executes on the harness worker thread.
        # Keep this benchmark in a subprocess so CUDA work runs on the child main thread.
        # warmup>=15 so the torch.compile-fused dequant GEMV (default path) is fully warm before
        # timing (otherwise residual JIT/guard cost leaks into the few measured iters).
        return BenchmarkConfig(iterations=10, warmup=15, use_subprocess=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_input_signature(self) -> dict:
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        return simple_signature(
            batch_size=1,
            dtype="float16",
            m=self.m,
            k=self.k,
            l=self.l,
            seed=self.seed,
            precision_flags=PrecisionFlags(fp16=True, tf32=tf32_enabled),
        ).to_dict()

    def validate_result(self) -> Optional[str]:
        if self._input_data is None or self.output is None:
            return "Benchmark output missing"
        if self.output.dtype != torch.float16:
            return f"Unexpected output dtype: {self.output.dtype}"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNvfp4GemvBenchmark()
