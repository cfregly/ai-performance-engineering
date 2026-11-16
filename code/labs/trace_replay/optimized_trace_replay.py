"""Replay the same retained operator with its existing optimized kernel."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.trace_replay.adapters import ReplayBenchmark


def get_benchmark():
    return attach_benchmark_metadata(ReplayBenchmark(optimized=True), __file__)
