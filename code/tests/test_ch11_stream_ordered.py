from pathlib import Path

import torch

from ch11.baseline_stream_ordered import BaselineStreamOrderedBenchmark
from ch11.optimized_stream_ordered import OptimizedStreamOrderedBenchmark
from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeStreamOrderedModule:
    def __init__(self) -> None:
        self.output = torch.ones(1, dtype=torch.float32)

    def run_standard_allocator_capture(self, elements: int, inner_iterations: int) -> torch.Tensor:
        return self.output

    def run_stream_ordered_allocator_capture(self, elements: int, inner_iterations: int) -> torch.Tensor:
        return self.output


def test_stream_ordered_benchmarks_prefer_kernel_replay_for_ncu() -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()

    assert baseline.preferred_ncu_replay_mode == "kernel"
    assert optimized.preferred_ncu_replay_mode == "kernel"
    assert baseline.preferred_ncu_metric_set == "minimal"
    assert optimized.preferred_ncu_metric_set == "minimal"
    assert baseline.get_config().ncu_replay_mode == "application"
    assert optimized.get_config().ncu_replay_mode == "application"


def test_stream_ordered_reduces_inner_iterations_only_during_profiling() -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()

    assert baseline._active_inner_iterations() == 500
    assert optimized._active_inner_iterations() == 500

    profiling_config = BenchmarkConfig(enable_profiling=True, enable_ncu=True, enable_nvtx=True)
    baseline._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)
    optimized._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)

    assert baseline._active_inner_iterations() == 8
    assert optimized._active_inner_iterations() == 8


def test_stream_ordered_benchmark_fn_and_capture_reuse_payload_tensors(monkeypatch) -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()
    baseline._module = _FakeStreamOrderedModule()
    optimized._module = _FakeStreamOrderedModule()

    def fail_tensor(*args, **kwargs):
        raise AssertionError("benchmark_fn() and capture should reuse verification tensors")

    monkeypatch.setattr(torch, "tensor", fail_tensor)
    baseline.benchmark_fn()
    optimized.benchmark_fn()
    baseline.capture_verification_payload()
    optimized.capture_verification_payload()

    baseline_inputs = baseline.get_verify_inputs()
    optimized_inputs = optimized.get_verify_inputs()
    assert baseline_inputs["elements"].item() == baseline.elements
    assert optimized_inputs["elements"].item() == optimized.elements
    assert baseline_inputs["inner_iterations"].item() == baseline.inner_iterations
    assert optimized_inputs["inner_iterations"].item() == optimized.inner_iterations


def test_stream_ordered_benchmark_fn_uses_setup_cached_iteration_count() -> None:
    for relative in (
        "ch11/baseline_stream_ordered.py",
        "ch11/optimized_stream_ordered.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        setup_section = source.split("def setup", maxsplit=1)[1].split("def benchmark_fn", maxsplit=1)[0]
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def capture_verification_payload", maxsplit=1
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def get_config", maxsplit=1
        )[0]

        assert "self._elements_tensor = torch.tensor([self.elements], dtype=torch.int64)" in source
        assert "self._inner_iterations_tensor = torch.empty((1,), dtype=torch.int64)" in source
        assert "self._inner_iterations_tensor[0] = self.inner_iterations" in source
        assert "self._active_inner_iterations_count = self._active_inner_iterations()" in setup_section
        assert "inner_iterations = self._active_inner_iterations_count" in benchmark_section
        assert "with torch.inference_mode(), self._nvtx_range(" in benchmark_section
        assert "self._inner_iterations_tensor[0] = inner_iterations" in benchmark_section
        assert '"elements": self._elements_tensor' in capture_section
        assert '"inner_iterations": self._inner_iterations_tensor' in capture_section
        assert "torch.tensor([self.elements]" not in capture_section
        assert "torch.tensor([self._last_inner_iterations]" not in capture_section
        assert "torch.no_grad()" not in benchmark_section
        assert "with self._nvtx_range(" not in benchmark_section
        assert "_active_inner_iterations()" not in benchmark_section
        assert "getattr(self, \"_config\"" not in benchmark_section
