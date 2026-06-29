#!/usr/bin/env python3
"""Baseline: tcgen05 GEMM stage-2 kernel (Blackwell).

This lab demonstrates progressive tcgen05 optimizations. The harness pair compares:
  - baseline: basic tcgen05 GEMM, and
  - optimized: clustered tcgen05 GEMM (higher performance).
"""

from __future__ import annotations

from typing import Optional
from types import ModuleType

import torch

from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.custom_vs_cublas.tcgen05_loader import load_tcgen05_module, matmul_tcgen05


class BaselineTcgen05MatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self) -> None:
        super().__init__()
        self.size = 8192
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._module: Optional[ModuleType] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            bytes_per_iteration=float(6 * self.size * self.size),
            custom_units_per_iteration=float(2 * self.size * self.size * self.size),
            custom_unit_name="FLOPs",
        )

    def setup(self) -> None:
        ensure_tcgen05_supported(module_name="labs/custom_vs_cublas tcgen05 kernels")

        # Compile the extension (first time only) outside the timed hot path.
        self._module = load_tcgen05_module()

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        dtype = torch.float16
        self.a = torch.randn((self.size, self.size), device=self.device, dtype=dtype)
        self.b = torch.randn((self.size, self.size), device=self.device, dtype=dtype)
        self._verify_output_buffer = torch.empty(
            min(128, self.size),
            min(256, self.size),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if self.a is None or self.b is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        if self._module is None:
            raise RuntimeError("tcgen05 module was not compiled in setup()")
        with torch.inference_mode():
            self.output = self._module.matmul_tcgen05(self.a, self.b)

    def capture_verification_payload(self) -> None:
        if self.a is None or self.b is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"A": self.a, "B": self.b},
            output=self._verify_output_buffer,
            batch_size=self.size,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.1, 0.5),
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.size,
            dtype="float16",
            m=self.size,
            n=self.size,
            k=self.size,
            precision_flags=PrecisionFlags(fp16=True, tf32=False),
        ).to_dict()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5, use_subprocess=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineTcgen05MatmulBenchmark()
