"""Memory-bound gates expose metadata before setup and verify the full result."""

from types import SimpleNamespace

import pytest
import torch

from ch09.baseline_memory_bound import BaselineMemoryBoundBenchmark
from ch09.optimized_memory_bound import OptimizedMemoryBoundBenchmark


def test_architecture_policy_keeps_modern_cudagraph_ownership() -> None:
    from core.harness.arch_config import _configure_triton_compile_policy

    triton_config = SimpleNamespace(
        unique_kernel_names=False,
        cudagraphs=True,
        cudagraph_trees=False,
    )
    _configure_triton_compile_policy(triton_config)
    assert triton_config.unique_kernel_names is True
    assert triton_config.cudagraphs is False
    assert triton_config.cudagraph_trees is True


def test_memory_bound_metadata_is_available_before_setup() -> None:
    baseline = BaselineMemoryBoundBenchmark()
    optimized = OptimizedMemoryBoundBenchmark()
    assert optimized.get_workload_metadata() == baseline.get_workload_metadata()
    assert optimized.get_workload_metadata().tokens_per_iteration == 16_777_216 * 64


def test_memory_bound_tolerance_rejects_a_missing_repeat() -> None:
    benchmark = BaselineMemoryBoundBenchmark()
    benchmark.device = torch.device("cpu")
    benchmark.N = 8193
    try:
        benchmark.setup()
        benchmark.tensor.fill_(1.0)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        complete = benchmark.get_verify_output().clone()
        rtol, atol = benchmark.get_output_tolerance()
        benchmark._repeat_range = range(benchmark.repeats - 1)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        assert not torch.allclose(benchmark.get_verify_output(), complete, rtol=rtol, atol=atol)
    finally:
        benchmark.teardown()


@pytest.mark.parametrize("optimized", [False, True])
def test_memory_bound_captures_live_tail_beyond_old_crop(optimized: bool) -> None:
    if optimized and not torch.cuda.is_available():
        pytest.skip("The production compiled memory-bound path requires CUDA")
    if optimized:
        # Exercise the same process-global policy loaded by the benchmark CLI.
        # The policy must disable implicit graph wrapping while retaining the
        # tree owner used when reduce-overhead explicitly enables graphs.
        import core.harness.arch_config  # noqa: F401

        triton_config = torch._inductor.config.triton
        assert triton_config.cudagraphs is False
        assert triton_config.cudagraph_trees is True
    benchmark = OptimizedMemoryBoundBenchmark() if optimized else BaselineMemoryBoundBenchmark()
    benchmark.device = torch.device("cuda" if optimized else "cpu")
    benchmark.N = 8193
    torch.manual_seed(1042)
    benchmark.setup()
    try:
        assert torch.initial_seed() == 1042
        source = benchmark.data if optimized else benchmark.tensor
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        first = benchmark.get_verify_output().clone()
        assert first.shape == (8193,)
        torch.testing.assert_close(first, benchmark.output, rtol=0, atol=0)
        source[-1].add_(1.0)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        changed = benchmark.get_verify_output()
        assert changed[-1] != first[-1]
        torch.testing.assert_close(changed[:-1], first[:-1], rtol=0, atol=0)
    finally:
        benchmark.teardown()
    if optimized:
        assert benchmark._compiled_run is None
