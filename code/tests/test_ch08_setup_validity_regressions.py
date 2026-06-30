from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ch08.ai_optimization_benchmark_base as ai_base
import ch08.hbm_benchmark_base as hbm_base
import ch08.loop_unrolling_benchmark_base as loop_base
import ch08.tiling_benchmark_base as tiling_base
import ch08.tiling_benchmark_base_tcgen05 as tiling_tcgen05_base
from ch08.baseline_ai_optimization import BaselineAiOptimizationBenchmark
from ch08.baseline_hbm import BaselineHBMBenchmark
from ch08.baseline_loop_unrolling import BaselineLoopUnrollingBenchmark
from ch08.baseline_tiling import BaselineTilingBenchmark
from ch08.baseline_tiling_tcgen05 import BaselineTilingBenchmarkTCGen05
from ch08.optimized_ai_optimization import OptimizedAiOptimizationBenchmark
from ch08.optimized_hbm import OptimizedHBMBenchmark
from ch08.optimized_loop_unrolling import OptimizedLoopUnrollingBenchmark
from ch08.optimized_tiling import OptimizedTilingBenchmark
from ch08.optimized_tiling_tcgen05 import OptimizedTilingBenchmarkTCGen05


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _FakeAiOptimizationExtension:
    def ai_baseline(self, inputs: torch.Tensor, weights: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(torch.tanh(inputs @ weights))

    def ai_optimized(self, inputs: torch.Tensor, weights: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(torch.tanh(inputs @ weights))


class _FakeHBMExtension:
    def hbm_baseline(self, matrix: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix.sum(dim=0))

    def hbm_optimized(self, matrix: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix.sum(dim=1))


class _FakeLoopUnrollingExtension:
    def loop_unrolling_baseline(
        self, inputs: torch.Tensor, weights: torch.Tensor, output: torch.Tensor
    ) -> None:
        tiled = weights.repeat((inputs.shape[1] + weights.numel() - 1) // weights.numel())[: inputs.shape[1]]
        output.copy_((inputs * tiled).sum(dim=1))

    def loop_unrolling_optimized(
        self, inputs: torch.Tensor, weights: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.loop_unrolling_baseline(inputs, weights, output)


class _FakeTilingExtension:
    def matmul_naive(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix_a @ matrix_b)

    def matmul_tiled(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix_a @ matrix_b)

    def matmul_tiled_fast(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix_a @ matrix_b)

    def tiling_baseline(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix_a @ matrix_b)

    def tiling_optimized(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor, output: torch.Tensor) -> None:
        output.copy_(matrix_a @ matrix_b)

    def matmul_tiling_tcgen05(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        return matrix_a @ matrix_b

    def matmul_tiling_tcgen05_pretransposed(
        self, matrix_a: torch.Tensor, matrix_b_t: torch.Tensor
    ) -> torch.Tensor:
        return matrix_a @ matrix_b_t.t()


def test_ch08_base_benchmarks_use_cached_nvtx_range() -> None:
    base_files = (
        "ai_optimization_benchmark_base.py",
        "hbm_benchmark_base.py",
        "loop_unrolling_benchmark_base.py",
        "nccl_benchmark_base.py",
        "threshold_benchmark_base.py",
        "tiling_benchmark_base.py",
    )

    for filename in base_files:
        source = (REPO_ROOT / "ch08" / filename).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload" if "def capture_verification_payload" in source else "def teardown",
            maxsplit=1,
        )[0]

        assert "self._nvtx_range(self.nvtx_label)" in benchmark_section
        assert "torch.inference_mode()" in benchmark_section
        assert "torch.no_grad()" not in benchmark_section
        assert "with self._nvtx_range(" not in benchmark_section
        assert "get_nvtx_enabled(" not in benchmark_section
        assert "with nvtx_range(" not in benchmark_section
        assert "from core.profiling.nvtx_helper" not in source


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineAiOptimizationBenchmark,
        OptimizedAiOptimizationBenchmark,
    ],
)
def test_ai_optimization_setup_keeps_public_output_empty(
    benchmark_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_base, "load_cuda_extension", lambda **_: _FakeAiOptimizationExtension())

    bench = benchmark_cls()
    bench.rows = 256
    bench.cols = 64
    bench.inner_iterations = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineHBMBenchmark,
        OptimizedHBMBenchmark,
    ],
)
def test_hbm_setup_keeps_public_output_empty(
    benchmark_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hbm_base, "load_cuda_extension", lambda **_: _FakeHBMExtension())

    bench = benchmark_cls()
    bench.rows = 128
    bench.cols = 64
    bench.inner_iterations = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineLoopUnrollingBenchmark,
        OptimizedLoopUnrollingBenchmark,
    ],
)
def test_loop_unrolling_setup_keeps_public_output_empty(
    benchmark_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop_base, "load_cuda_extension", lambda **_: _FakeLoopUnrollingExtension())

    bench = benchmark_cls()
    bench.rows = 256
    bench.elements_per_row = 64
    bench.weight_period = 8
    bench.inner_iterations = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineTilingBenchmark,
        OptimizedTilingBenchmark,
        BaselineTilingBenchmarkTCGen05,
        OptimizedTilingBenchmarkTCGen05,
    ],
)
def test_tiling_setup_keeps_public_output_empty(
    benchmark_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tiling_base, "load_cuda_extension", lambda **_: _FakeTilingExtension())
    monkeypatch.setattr(tiling_tcgen05_base, "_check_tcgen05_extension_available", lambda: (True, None))
    monkeypatch.setattr(
        tiling_tcgen05_base.TilingBenchmarkBaseTCGen05,
        "_load_extension",
        lambda self: setattr(self, "extension", _FakeTilingExtension()),
    )

    bench = benchmark_cls()
    bench.matrix_rows = 64
    bench.matrix_cols = 64
    bench.shared_dim = 32
    bench.inner_iterations = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()
