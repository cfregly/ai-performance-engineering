from __future__ import annotations

import inspect

import pytest
import torch

import ch06.baseline_bank_conflicts as baseline_bank_conflicts
import ch06.optimized_bank_conflicts as optimized_bank_conflicts
from ch06.baseline_adaptive import BaselineAdaptiveBenchmark
from ch06.baseline_autotuning import BaselineAutotuningBenchmark
from ch06.baseline_bank_conflicts import BaselineBankConflictsBenchmark
from ch06.optimized_adaptive import OptimizedAdaptiveBenchmark
from ch06.optimized_autotuning import OptimizedAutotuningBenchmark
from ch06.optimized_bank_conflicts import OptimizedBankConflictsBenchmark


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_optimized_autotuning_writes_transform_into_reused_buffers() -> None:
    source = inspect.getsource(OptimizedAutotuningBenchmark)
    transform_section = source.split("def _transform", maxsplit=1)[1].split(
        "def _autotune_chunk_size",
        maxsplit=1,
    )[0]

    assert "def _transform(self, tensor: torch.Tensor, out: torch.Tensor)" in source
    assert "torch.mul(tensor, 1.75, out=out)" in transform_section
    assert "out.add_(0.1)" in transform_section
    assert "F.silu(out, inplace=True)" in transform_section
    assert "transformed = self._transform" not in source
    assert ".copy_(transformed)" not in source
    assert "self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []" in source
    assert "def _build_chunk_views(self, chunk: int) -> list[tuple[torch.Tensor, torch.Tensor]]" in source
    assert "self._chunk_views = self._build_chunk_views(self.optimal_chunk)" in source
    assert "self._transform(window, scratch[offset : offset + span])" in source
    assert "for window, out_window in self._chunk_views:" in source
    assert "self._transform(window, out_window)" in source
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    assert "offset : offset + span" not in benchmark_section

    bench = OptimizedAutotuningBenchmark()
    x = torch.randn(16, dtype=torch.float32)
    out = torch.empty_like(x)
    actual = bench._transform(x, out)
    expected = torch.nn.functional.silu(x * 1.75 + 0.1)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_optimized_adaptive_writes_transform_into_reused_buffer() -> None:
    source = inspect.getsource(OptimizedAdaptiveBenchmark)
    transform_section = source.split("def _transform", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "def _transform(self, tensor: torch.Tensor, out: torch.Tensor)" in source
    assert "torch.mul(tensor, 1.75, out=out)" in transform_section
    assert "out.add_(0.1)" in transform_section
    assert "F.silu(out, inplace=True)" in transform_section
    assert "transformed = self._transform" not in source
    assert ".copy_(transformed)" not in source
    assert "self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []" in source
    assert "self._chunk_views = [" in source
    assert "for window, out_window in self._chunk_views:" in source
    assert "self._transform(window, out_window)" in source
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    assert "self.input[start:end]" not in benchmark_section
    assert "self._output_buffer[start:end]" not in benchmark_section

    bench = OptimizedAdaptiveBenchmark()
    x = torch.randn(16, dtype=torch.float32)
    out = torch.empty_like(x)
    actual = bench._transform(x, out)
    expected = torch.nn.functional.silu(x * 1.75 + 0.1)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_baseline_autotuning_writes_transform_into_reused_buffers() -> None:
    source = inspect.getsource(BaselineAutotuningBenchmark)
    transform_section = source.split("def _transform", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "def _transform(self, tensor: torch.Tensor, out: torch.Tensor)" in source
    assert "torch.mul(tensor, 1.75, out=out)" in transform_section
    assert "out.add_(0.1)" in transform_section
    assert "F.silu(out, inplace=True)" in transform_section
    assert "transformed = self._transform" not in source
    assert ".copy_(transformed)" not in source
    assert "self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []" in source
    assert "for window, out_window in self._chunk_views:" in source
    assert "self._transform(window, out_window)" in source
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    assert "self.input[start:end]" not in benchmark_section
    assert "self._output_buffer[start:end]" not in benchmark_section

    bench = BaselineAutotuningBenchmark()
    x = torch.randn(16, dtype=torch.float32)
    out = torch.empty_like(x)
    actual = bench._transform(x, out)
    expected = torch.nn.functional.silu(x * 1.75 + 0.1)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_baseline_adaptive_writes_transform_into_reused_buffer() -> None:
    source = inspect.getsource(BaselineAdaptiveBenchmark)
    transform_section = source.split("def _transform", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "def _transform(self, tensor: torch.Tensor, out: torch.Tensor)" in source
    assert "torch.mul(tensor, 1.75, out=out)" in transform_section
    assert "out.add_(0.1)" in transform_section
    assert "F.silu(out, inplace=True)" in transform_section
    assert "transformed = self._transform" not in source
    assert ".copy_(transformed)" not in source
    assert "self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []" in source
    assert "for window, out_window in self._chunk_views:" in source
    assert "self._transform(window, out_window)" in source
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]
    assert "self.input[start:end]" not in benchmark_section
    assert "self._output_buffer[start:end]" not in benchmark_section

    bench = BaselineAdaptiveBenchmark()
    x = torch.randn(16, dtype=torch.float32)
    out = torch.empty_like(x)
    actual = bench._transform(x, out)
    expected = torch.nn.functional.silu(x * 1.75 + 0.1)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineAdaptiveBenchmark,
        OptimizedAdaptiveBenchmark,
        BaselineAutotuningBenchmark,
        OptimizedAutotuningBenchmark,
    ],
)
def test_chunked_processing_setup_keeps_public_output_empty(benchmark_cls: type) -> None:
    bench = benchmark_cls()
    bench.N = 4096
    if hasattr(bench, "static_chunk"):
        bench.static_chunk = 256
    if hasattr(bench, "candidates"):
        bench.candidates = [128, 256]
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()


class _FakeBankConflictsExtension:
    def bank_conflicts(self, output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output.copy_(input_tensor)

    def bank_conflicts_padded(self, output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output.copy_(input_tensor)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("benchmark_cls", "module"),
    [
        (BaselineBankConflictsBenchmark, baseline_bank_conflicts),
        (OptimizedBankConflictsBenchmark, optimized_bank_conflicts),
    ],
)
def test_bank_conflicts_setup_keeps_public_output_empty(
    benchmark_cls: type,
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "load_bank_conflicts_extension", lambda: _FakeBankConflictsExtension())

    bench = benchmark_cls()
    bench.N = 4096
    bench.repeats = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()
