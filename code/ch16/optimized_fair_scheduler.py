"""Token-fair admission policy benchmark."""

from ch16.fair_scheduler_benchmarks import FairSchedulerBenchmark
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(FairSchedulerBenchmark(optimized=True), __file__)
