"""baseline_performance_fp16.py - Compute-heavy FP32 baseline for the precision-only pair."""

from __future__ import annotations

import torch

from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from core.harness.benchmark_harness import BaseBenchmark
from ch01.baseline_performance import BaselinePerformanceBenchmark
from ch01.performance_fp16_common import PERFORMANCE_FP16_WORKLOAD


class BaselinePerformanceFP16Benchmark(BaselinePerformanceBenchmark):
    """Keep the Chapter 1 baseline logic, but use the local FP16-comparison workload."""

    def __init__(self):
        super().__init__()
        self.batch_size = PERFORMANCE_FP16_WORKLOAD.batch_size
        self.num_microbatches = PERFORMANCE_FP16_WORKLOAD.num_microbatches
        self.hidden_dim = PERFORMANCE_FP16_WORKLOAD.hidden_dim
        self.register_workload_metadata(
            samples_per_iteration=PERFORMANCE_FP16_WORKLOAD.samples_per_iteration,
        )
        self._enable_nvtx = False

    def setup(self) -> None:
        super().setup()
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

    def benchmark_fn(self) -> None:
        """Mirror the inherited baseline loop explicitly for pair-audit equivalence."""
        assert (
            self._microbatch_groups is not None
            and self._target_groups is not None
            and self._group_sizes is not None
            and self._training_groups is not None
        )
        with nvtx_range("baseline_performance_fp16", enable=self._enable_nvtx):
            for paired_group, group_size in self._training_groups:
                self.optimizer.zero_grad(set_to_none=True)
                for data, target in paired_group:
                    logits = self.model(data)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    (loss / group_size).backward()
                self.optimizer.step()


def get_benchmark() -> BaseBenchmark:
    return BaselinePerformanceFP16Benchmark()
