"""Replay one retained operator with its materialized native baseline."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.trace_replay.adapters import ReplayBenchmark


def get_benchmark():
    return attach_benchmark_metadata(ReplayBenchmark(optimized=False), __file__)
