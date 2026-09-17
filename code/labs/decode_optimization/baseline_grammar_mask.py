"""Legal-token mask compilation and reuse benchmark."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.decode_optimization.grammar_mask_benchmark import GrammarMaskBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(GrammarMaskBenchmark(optimized=False), __file__)
