"""Caller-owned RNG controls for the reusable Chapter 8 benchmark bases."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import ch08.ai_optimization_benchmark_base as ai_base
import ch08.hbm_benchmark_base as hbm_base
import ch08.loop_unrolling_benchmark_base as loop_base
import ch08.threshold_benchmark_base as threshold_base
import ch08.tiling_benchmark_base as tiling_base
from ch08.ai_optimization_benchmark_base import AiOptimizationBenchmarkBase
from ch08.hbm_benchmark_base import HBMBenchmarkBase
from ch08.loop_unrolling_benchmark_base import LoopUnrollingBenchmarkBase
from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from ch08.threshold_benchmark_base import (
    THRESHOLD_INNER_SCALE,
    THRESHOLD_OUTER_SCALE,
    THRESHOLD_SECONDARY_SCALE,
    ThresholdBenchmarkBase,
)
from ch08.tiling_benchmark_base import TilingBenchmarkBase
from core.harness.benchmark_harness import BaseBenchmark
from tests.protection_test_utils import preserve_rng_state


CODE_ROOT = Path(__file__).resolve().parents[1]
SETUP_CLASSES = {
    "ch08/threshold_benchmark_base.py": "ThresholdBenchmarkBase",
    "ch08/hbm_benchmark_base.py": "HBMBenchmarkBase",
    "ch08/tiling_benchmark_base.py": "TilingBenchmarkBase",
    "ch08/loop_unrolling_benchmark_base.py": "LoopUnrollingBenchmarkBase",
    "ch08/ai_optimization_benchmark_base.py": "AiOptimizationBenchmarkBase",
    "ch08/tcgen05_custom_vs_cublas_benchmark_base.py": "Tcgen05CustomVsCublasBase",
}
GLOBAL_SEED_CALLS = {
    "torch.manual_seed",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
}


class _CpuThreshold(ThresholdBenchmarkBase):
    rows = 32
    inner_iterations = 1

    def _resolve_device(self) -> torch.device:
        return torch.device("cpu")

    def _invoke_kernel(self) -> None:
        assert self.inputs is not None and self.outputs is not None
        magnitude = self.inputs.abs() + torch.sin(self.inputs) * torch.cos(self.inputs) * 0.0001
        active = self.inputs.abs() > self.threshold
        outer = self.inputs.abs() > self.threshold * THRESHOLD_SECONDARY_SCALE
        scale = torch.where(outer, THRESHOLD_OUTER_SCALE, THRESHOLD_INNER_SCALE)
        self.outputs.copy_(torch.copysign(magnitude * scale, self.inputs))
        self.outputs.masked_fill_(active.logical_not_(), 0.0)


class _CpuHBM(HBMBenchmarkBase):
    rows = 8
    cols = 6
    inner_iterations = 1

    def __init__(self) -> None:
        BaseBenchmark.__init__(self)
        self.device = torch.device("cpu")
        self.extension = None
        self.matrix_row = None
        self.matrix_col = None
        self.output = None
        self._output_buffer = None
        self.host_col = None
        self._inner_iteration_range = range(self.inner_iterations)


class _CpuTiling(TilingBenchmarkBase):
    matrix_rows = 5
    matrix_cols = 3
    shared_dim = 4
    inner_iterations = 1

    def _resolve_device(self) -> torch.device:
        return torch.device("cpu")

    def _load_extension(self) -> None:
        self.extension = object()


class _CpuLoop(LoopUnrollingBenchmarkBase):
    rows = 7
    elements_per_row = 6
    weight_period = 3
    inner_iterations = 1

    def __init__(self) -> None:
        BaseBenchmark.__init__(self)
        self.device = torch.device("cpu")
        self.extension = None
        self.inputs = None
        self.weights = None
        self.output = None
        self._output_buffer = None
        self._inner_iteration_range = range(self.inner_iterations)


class _CpuAi(AiOptimizationBenchmarkBase):
    rows = 8
    cols = 6
    inner_iterations = 1

    def __init__(self) -> None:
        BaseBenchmark.__init__(self)
        self.device = torch.device("cpu")
        self.extension = None
        self.inputs = None
        self.weights = None
        self.output = None
        self._output_buffer = None
        self._inner_iteration_range = range(self.inner_iterations)


class _CpuTcgen05(Tcgen05CustomVsCublasBase):
    matrix_rows = 5
    matrix_cols = 3
    shared_dim = 4
    tensor_dtype = torch.float32

    def _resolve_device(self) -> torch.device:
        return torch.device("cpu")


def _setup_snapshot(name: str, seed: int) -> tuple[tuple[torch.Tensor, ...], int]:
    torch.manual_seed(seed)
    if name == "threshold":
        benchmark = _CpuThreshold()
        fields = ("inputs",)
    elif name == "hbm":
        benchmark = _CpuHBM()
        fields = ("matrix_row", "matrix_col")
    elif name == "tiling":
        benchmark = _CpuTiling()
        fields = ("matrix_a", "matrix_b")
    elif name == "loop":
        benchmark = _CpuLoop()
        fields = ("inputs", "weights")
    elif name == "ai":
        benchmark = _CpuAi()
        fields = ("inputs", "weights")
    else:
        benchmark = _CpuTcgen05()
        fields = ("matrix_a", "matrix_b")
    benchmark.setup()
    tensors = tuple(getattr(benchmark, field).detach().clone() for field in fields)
    return tensors, int(torch.initial_seed())


def test_ch08_base_setups_do_not_reset_the_harness_seed() -> None:
    for relative_path, class_name in SETUP_CLASSES.items():
        tree = ast.parse((CODE_ROOT / relative_path).read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        setup = next(
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "setup"
        )
        calls = {ast.unparse(node.func) for node in ast.walk(setup) if isinstance(node, ast.Call)}
        assert calls.isdisjoint(GLOBAL_SEED_CALLS), relative_path


@pytest.mark.parametrize("name", ["threshold", "hbm", "tiling", "loop", "ai", "tcgen05"])
def test_ch08_cpu_setup_preserves_default_and_fresh_seed(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(threshold_base, "load_cuda_extension", lambda **_: object())
    monkeypatch.setattr(hbm_base, "load_cuda_extension", lambda **_: object())
    monkeypatch.setattr(loop_base, "load_cuda_extension", lambda **_: object())
    monkeypatch.setattr(ai_base, "load_cuda_extension", lambda **_: object())

    with preserve_rng_state():
        first_42, observed_42 = _setup_snapshot(name, 42)
        second_42, repeated_42 = _setup_snapshot(name, 42)
        fresh_1042, observed_1042 = _setup_snapshot(name, 1042)

    assert observed_42 == repeated_42 == 42
    assert observed_1042 == 1042
    assert all(torch.equal(left, right) for left, right in zip(first_42, second_42, strict=True))
    assert any(not torch.equal(left, right) for left, right in zip(first_42, fresh_1042, strict=True))
