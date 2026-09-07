#!/usr/bin/env python3
"""Optimized: compiled Llama 3.1 8B SDPA execution.

The comparison isolates max-autotune ``torch.compile`` while preserving the
baseline's preferred SDPA, FP32 residual, and stable RMSNorm implementation.
"""

from __future__ import annotations

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.real_world_models.llama_3_1_8b_optimization import (
    LLAMA_BF16_OUTPUT_TOLERANCE,
    Llama31_8B_Optimization,
)


class OptimizedLlama31_8B(VerificationPayloadMixin, BaseBenchmark):
    """Max-autotune compiled arm for the compile-only comparison."""

    def __init__(self, batch_size: int = 1, seq_length: int = 2048):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.model_wrapper = None
        self.output: torch.Tensor | None = None
        self.parameter_count = 0
        self._last_metrics: dict[str, float] = {
            "llama.use_compile": 1.0,
            "llama.attention.preferred_sdpa": 1.0,
            "llama.fp32_residual": 1.0,
            "llama.stable_rms_norm": 1.0,
            "llama.emulate_precision_casts": 1.0,
        }
        self.register_workload_metadata(requests_per_iteration=float(batch_size))

    def setup(self) -> None:
        self.model_wrapper = Llama31_8B_Optimization(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            use_compile=True,
            attention_mode="preferred_sdpa",
        )
        self.model_wrapper.setup()
        self.parameter_count = sum(p.numel() for p in self.model_wrapper.layers.parameters())

    def benchmark_fn(self) -> None:
        if self.model_wrapper is None:
            raise RuntimeError("Model wrapper not initialized")
        self.model_wrapper.run()
        self.output = self.model_wrapper.output
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.model_wrapper is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        self._set_verification_payload(
            inputs={"input": self.model_wrapper.input.detach()},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "bf16": True,
                "fp16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=LLAMA_BF16_OUTPUT_TOLERANCE,
        )

    def teardown(self) -> None:
        if self.model_wrapper:
            self.model_wrapper.teardown()
        self.model_wrapper = None
        self.output = None
        self._verification_payload = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_custom_metrics(self) -> dict:
        return self._last_metrics


def get_benchmark() -> BaseBenchmark:
    return OptimizedLlama31_8B()
