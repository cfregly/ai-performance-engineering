"""Optimized EOS early-exit decode: stop once every row is complete."""

from __future__ import annotations

from ch18.eos_early_exit_common import EosEarlyExitBenchmark, EosEarlyExitConfig
from core.benchmark.wrapper_utils import attach_benchmark_metadata


def get_benchmark() -> EosEarlyExitBenchmark:
    cfg = EosEarlyExitConfig(
        stop_on_all_done=True,
        label="optimized_eos_early_exit",
    )
    return attach_benchmark_metadata(EosEarlyExitBenchmark(cfg), __file__)  # type: ignore[return-value]
