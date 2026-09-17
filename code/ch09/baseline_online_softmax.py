"""Online-normalizer fusion comparison."""

from ch09.online_softmax_common import OnlineSoftmaxBenchmark
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(OnlineSoftmaxBenchmark(optimized=False), __file__)
