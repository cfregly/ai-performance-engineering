#!/usr/bin/env python3
"""Optimized: tcgen05 GEMM with clustered launch (Blackwell)."""

from __future__ import annotations

import os
from typing import Optional
from types import ModuleType

import torch

from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.custom_vs_cublas.tcgen05_loader import (
    load_tcgen05_cluster_module,
    load_tcgen05_dual_cta_2sm_module,
    load_tcgen05_dual_cta_module,
    matmul_tcgen05_cluster,
)

# Variant switch (default: committed cluster kernel, 2.329x vs naive).
#   AISP_TCGEN05_VARIANT=dual_cta     -> 2 CTAs/SM occupancy rewrite
#   AISP_TCGEN05_VARIANT=dual_cta_2sm -> 2-SM UMMA pair (cta_group::2)
#   AISP_TCGEN05_VARIANT=cluster      -> incumbent (default)
_VARIANT = os.environ.get("AISP_TCGEN05_VARIANT", "cluster")


class OptimizedTcgen05MatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self) -> None:
        super().__init__()
        self.size = 8192
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._module: Optional[ModuleType] = None
        self._kernel_fn = None
        self.register_workload_metadata(
            bytes_per_iteration=float(6 * self.size * self.size),
            custom_units_per_iteration=float(2 * self.size * self.size * self.size),
            custom_unit_name="FLOPs",
        )

    def setup(self) -> None:
        ensure_tcgen05_supported(module_name="labs/custom_vs_cublas tcgen05 kernels")

        # Compile the extension (first time only) outside the timed hot path.
        if _VARIANT == "dual_cta":
            self._module = load_tcgen05_dual_cta_module()
            self._kernel_fn = self._module.matmul_tcgen05_dual_cta
        elif _VARIANT == "dual_cta_2sm":
            self._module = load_tcgen05_dual_cta_2sm_module()
            self._kernel_fn = self._module.matmul_tcgen05_dual_cta_2sm
        else:
            self._module = load_tcgen05_cluster_module()
            self._kernel_fn = self._module.matmul_tcgen05_cluster

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        dtype = torch.float16
        self.a = torch.randn((self.size, self.size), device=self.device, dtype=dtype)
        self.b = torch.randn((self.size, self.size), device=self.device, dtype=dtype)

    def benchmark_fn(self) -> None:
        if self.a is None or self.b is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        if self._module is None:
            raise RuntimeError("tcgen05 module was not compiled in setup()")
        with torch.inference_mode():
            self.output = self._kernel_fn(self.a, self.b)

    def capture_verification_payload(self) -> None:
        if self.a is None or self.b is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"A": self.a, "B": self.b},
            output=self.output.detach().float().clone(),
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
    return OptimizedTcgen05MatmulBenchmark()
