"""Optimized legacy thresholdtma benchmark using per-thread async-copy staging."""

from __future__ import annotations

from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA


class OptimizedThresholdTMABenchmark(ThresholdBenchmarkBaseTMA):
    nvtx_label = "optimized_threshold_tma"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_tma_optimized(self.inputs, self.outputs, self.threshold)



def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return OptimizedThresholdTMABenchmark()


