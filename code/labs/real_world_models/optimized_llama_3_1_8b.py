#!/usr/bin/env python3
"""Optimized: Llama 3.1 8B with all Blackwell optimizations.

Enables torch.compile, FlexAttention, and optional FP8 for maximum performance.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.real_world_models.llama_3_1_8b_optimization import Llama31_8B_Optimization


class OptimizedLlama31_8B(VerificationPayloadMixin, BaseBenchmark):
    """Optimized Llama 3.1 8B - torch.compile + FlexAttention + optional FP8.
    
    Optimizations:
    - torch.compile with max-autotune
    - FlexAttention for efficient attention
    - Optional FP8 precision (2x throughput on Blackwell)
    """

    def __init__(self, batch_size: int = 1, seq_length: int = 2048, use_fp8: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_fp8 = use_fp8
        self.model_wrapper = None
        self.output: Optional[torch.Tensor] = None
        self.parameter_count = 0
        self._last_metrics: dict[str, float] = {
            "llama.use_compile": 1.0,
            "llama.use_flex_attention": 1.0,
            "llama.use_fp8": 1.0 if self.use_fp8 else 0.0,
        }
        self.register_workload_metadata(requests_per_iteration=float(batch_size))

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model_wrapper = Llama31_8B_Optimization(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            use_compile=True,         # Enable torch.compile
            use_fp8=self.use_fp8,     # Optional FP8
            use_flex_attention=True,  # Enable FlexAttention
            prefer_sdpa=True,
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
        self._set_verification_payload(
            inputs={"input": self.model_wrapper.input.detach()},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "bf16": True,
                "fp16": False,
                "fp8": bool(self.use_fp8),
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        if self.model_wrapper:
            self.model_wrapper.teardown()
        self.model_wrapper = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_custom_metrics(self) -> dict:
        return self._last_metrics


def get_benchmark() -> BaseBenchmark:
    return OptimizedLlama31_8B()

