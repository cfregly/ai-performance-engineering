from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ch05.baseline_decompression import CPUDecompressionBenchmark
from ch05.optimized_decompression import GPUDecompressionBenchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_gpu_decompression_reuses_preallocated_broadcast_output() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_decompression.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def benchmark_fn", maxsplit=1
    )[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._output_matrix = torch.empty((num_runs, run_len)" in setup_section
    assert "self._output_flat = self._output_matrix.reshape(-1)" in setup_section
    assert "torch.repeat_interleave" not in benchmark_section
    assert "self._output_matrix.copy_(self.values.unsqueeze(1))" in benchmark_section
    assert "out = self._output_flat" in benchmark_section


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU decompression")
def test_gpu_decompression_defers_full_output_clone_to_capture() -> None:
    _assert_decompression_clone_deferred(GPUDecompressionBenchmark())
