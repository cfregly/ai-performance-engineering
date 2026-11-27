"""Baseline BF16 MLP for the Chapter 16 NVFP4 quantization comparison."""

from __future__ import annotations

from core.benchmark.nvfp4_mlp import NVFP4MLPBenchmark, NVFP4MLPConfig
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    config = NVFP4MLPConfig(
        batch_size=512,
        d_model=8192,
        d_ff=32768,
        num_layers=2,
        iterations=20,
        warmup=10,
        name="ch16_nvfp4_mlp",
    )
    return attach_benchmark_metadata(NVFP4MLPBenchmark(config, use_nvfp4=False), __file__)
