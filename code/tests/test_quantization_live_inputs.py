"""Real forward paths must consume the inputs exposed to verification."""

from __future__ import annotations

import pytest
import torch

from ch13.baseline_precisionfp8_pad_inner import BaselinePrecisionFP8PadInnerBenchmark
from ch13.baseline_torchao_quantization import BaselineTorchAOQuantizationBenchmark
from ch13.optimized_precisionfp8_pad_inner import (
    TORCHAO_IMPORT_ERROR as FP8_IMPORT_ERROR,
)
from ch13.optimized_precisionfp8_pad_inner import OptimizedFP8PadInnerBenchmark
from ch13.optimized_torchao_quantization import (
    TORCHAO_IMPORT_ERROR as INT8_IMPORT_ERROR,
)
from ch13.optimized_torchao_quantization import OptimizedTorchAOQuantizationBenchmark


def _small_workload(benchmark) -> None:
    benchmark.batch_size = 32
    if isinstance(benchmark, BaselinePrecisionFP8PadInnerBenchmark | OptimizedFP8PadInnerBenchmark):
        benchmark.input_dim = 40  # Exercise actual inner-dimension padding.
        benchmark.hidden_dim = benchmark.output_dim = 64
    else:
        benchmark.in_features = benchmark.hidden_features = benchmark.out_features = 64


def _check_live_forward(benchmark, *, seed: int) -> torch.Tensor:
    _small_workload(benchmark)
    torch.manual_seed(seed)
    benchmark.setup()
    try:
        assert torch.initial_seed() == seed
        declared = benchmark._verify_input
        consumed = benchmark.inputs if hasattr(benchmark, "inputs") else benchmark.data
        assert declared is consumed
        original_input = declared.clone()
        benchmark.benchmark_fn()
        original_output = benchmark.output.clone()
        declared.add_(0.75)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        assert not torch.equal(original_output, benchmark.output)
        assert torch.equal(benchmark.output.float(), benchmark._verify_output_buffer)
        if isinstance(benchmark, OptimizedFP8PadInnerBenchmark):
            assert torch.equal(benchmark.inputs_fp16, declared.half())
        declared.copy_(original_input)
        benchmark.benchmark_fn()
        assert torch.equal(original_output, benchmark.output)
        return original_input.cpu()
    finally:
        benchmark.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Benchmark setup requires CUDA")
def test_padding_baseline_preserves_seed_and_consumes_declared_input() -> None:
    first = _check_live_forward(BaselinePrecisionFP8PadInnerBenchmark(), seed=42)
    second = _check_live_forward(BaselinePrecisionFP8PadInnerBenchmark(), seed=1042)
    assert not torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual quantized paths require CUDA")
@pytest.mark.parametrize(
    "benchmark_type,import_error",
    [
        (OptimizedFP8PadInnerBenchmark, FP8_IMPORT_ERROR),
        (BaselineTorchAOQuantizationBenchmark, None),
        (OptimizedTorchAOQuantizationBenchmark, INT8_IMPORT_ERROR),
    ],
)
def test_quantized_forward_consumes_live_input_and_preserves_seed(benchmark_type, import_error) -> None:
    if import_error is not None:
        pytest.skip(f"torchao unavailable: {import_error}")
    first = _check_live_forward(benchmark_type(), seed=42)
    second = _check_live_forward(benchmark_type(), seed=1042)
    assert not torch.equal(first, second)
