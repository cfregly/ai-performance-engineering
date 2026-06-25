from __future__ import annotations

import pytest
import torch

from ch05.baseline_decompression import CPUDecompressionBenchmark
from ch05.optimized_decompression import GPUDecompressionBenchmark


def _assert_decompression_clone_deferred(bench) -> None:
    bench.setup()
    try:
        result = bench.benchmark_fn()
        assert result is not None
        assert bench.output is not None
        assert bench.output.numel() == result["decompressed_len"]
        output_ptr = bench.output.data_ptr()

        bench.capture_verification_payload()
        payload = bench._verification_payload
        assert payload.output.numel() == 4096
        assert payload.output.data_ptr() != output_ptr
    finally:
        bench.teardown()


def test_cpu_decompression_defers_full_output_clone_to_capture() -> None:
    _assert_decompression_clone_deferred(CPUDecompressionBenchmark())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU decompression")
def test_gpu_decompression_defers_full_output_clone_to_capture() -> None:
    _assert_decompression_clone_deferred(GPUDecompressionBenchmark())
