import torch

from ch11.baseline_stream_ordered import BaselineStreamOrderedBenchmark
from ch11.optimized_stream_ordered import OptimizedStreamOrderedBenchmark
from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView


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


def test_stream_ordered_benchmark_fn_defers_payload_tensor_allocation(monkeypatch) -> None:
    baseline = BaselineStreamOrderedBenchmark()
    optimized = OptimizedStreamOrderedBenchmark()
    baseline._module = _FakeStreamOrderedModule()
    optimized._module = _FakeStreamOrderedModule()

    def fail_tensor(*args, **kwargs):
        raise AssertionError("benchmark_fn() should not allocate verification tensors")

    monkeypatch.setattr(torch, "tensor", fail_tensor)
    baseline.benchmark_fn()
    optimized.benchmark_fn()
    monkeypatch.undo()

    baseline.capture_verification_payload()
    optimized.capture_verification_payload()

    baseline_inputs = baseline.get_verify_inputs()
    optimized_inputs = optimized.get_verify_inputs()
    assert baseline_inputs["elements"].item() == baseline.elements
    assert optimized_inputs["elements"].item() == optimized.elements
    assert baseline_inputs["inner_iterations"].item() == baseline.inner_iterations
    assert optimized_inputs["inner_iterations"].item() == optimized.inner_iterations
