"""Real forward paths must consume the inputs exposed to verification."""

from __future__ import annotations

import pytest
import torch

from ch13.baseline_precisionfp8_pad_inner import BaselinePrecisionFP8PadInnerBenchmark
from ch13.baseline_precisionfp8_pad_inner_matmul import BaselinePrecisionFP8PadInnerMatmulBenchmark
from ch13.baseline_torchao_quantization import BaselineTorchAOQuantizationBenchmark
from ch13.optimized_precisionfp8_pad_inner import (
    TORCHAO_IMPORT_ERROR as FP8_IMPORT_ERROR,
)
from ch13.optimized_precisionfp8_pad_inner import OptimizedFP8PadInnerBenchmark
from ch13.optimized_torchao_quantization import (
    TORCHAO_IMPORT_ERROR as INT8_IMPORT_ERROR,
)
from ch13.optimized_torchao_quantization import OptimizedTorchAOQuantizationBenchmark


def test_matmul_payload_covers_rows_and_columns_beyond_old_crop() -> None:
    benchmark = BaselinePrecisionFP8PadInnerMatmulBenchmark()
    benchmark.device = torch.device("cpu")
    benchmark.m, benchmark.k, benchmark.n = 129, 24, 257
    torch.manual_seed(1042)
    benchmark.setup()
    try:
        assert torch.initial_seed() == 1042
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        output = benchmark.get_verify_output()
        assert output.shape == (129, 257)
        torch.testing.assert_close(output, benchmark.a @ benchmark.b, rtol=0, atol=0)
        assert torch.equal(output[-1], benchmark.output[-1])
    finally:
        benchmark.teardown()


def _small_workload(benchmark) -> None:
    # Cross both former verification crop boundaries (128 rows, 256 columns).
    benchmark.batch_size = 144
    if isinstance(benchmark, BaselinePrecisionFP8PadInnerBenchmark | OptimizedFP8PadInnerBenchmark):
        benchmark.input_dim = 40  # Exercise actual inner-dimension padding.
        benchmark.hidden_dim = 64
        benchmark.output_dim = 272
    else:
        benchmark.in_features = benchmark.hidden_features = 64
        benchmark.out_features = 272


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
        assert benchmark.get_verify_output().shape == (144, 272)
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
