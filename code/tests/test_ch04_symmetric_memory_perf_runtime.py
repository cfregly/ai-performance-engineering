import inspect

import pytest
import torch

from ch04.baseline_symmetric_memory_perf import BaselineSymmetricMemoryPerfBenchmark
from ch04.optimized_symmetric_memory_perf import OptimizedSymmetricMemoryPerfBenchmark


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "benchmark_cls",
    (
        BaselineSymmetricMemoryPerfBenchmark,
        OptimizedSymmetricMemoryPerfBenchmark,
    ),
)
def test_ch04_symmetric_memory_perf_reuses_timing_pair(benchmark_cls) -> None:
    source = inspect.getsource(benchmark_cls.benchmark_fn)
    assert "current_stream = torch.cuda.current_stream(self.device)" in source
    assert "start.record(current_stream)" in source
    assert "end.record(current_stream)" in source
    assert "start.record()" not in source
    assert "end.record()" not in source

    benchmark = benchmark_cls(size_mb=0.0625)
    benchmark.setup()
    try:
        benchmark.benchmark_fn()
        torch.cuda.synchronize(benchmark.device)
        timing_pair = benchmark._timing_pair
        assert timing_pair is not None
        assert benchmark._pending_timing_pair is timing_pair

        benchmark.finalize_iteration_metrics()
        benchmark.benchmark_fn()
        torch.cuda.synchronize(benchmark.device)
        assert benchmark._timing_pair is timing_pair
        assert benchmark._pending_timing_pair is timing_pair

        benchmark.capture_verification_payload()
        assert benchmark._verification_payload is not None
        assert benchmark._verification_payload.output.numel() == benchmark._verify_numel
    finally:
        benchmark.teardown()
