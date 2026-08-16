"""Loop-unrolling variant with ILP and vectorized loads."""

from __future__ import annotations

from typing import Optional

from ch08.loop_unrolling_benchmark_base import LoopUnrollingBenchmarkBase


class OptimizedLoopUnrollingBenchmark(LoopUnrollingBenchmarkBase):
    nvtx_label = "optimized_loop_unrolling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None
        self.extension.loop_unrolling_optimized(self.inputs, self.weights, self.output)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for loop_unrolling."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=None,
            optimized_ms=getattr(self, '_last_elapsed_ms', None),
            name="loop_unrolling",
        )



def get_benchmark() -> LoopUnrollingBenchmarkBase:
    return OptimizedLoopUnrollingBenchmark()
