"""Baseline EOS early-exit decode: keep decoding after the batch is complete."""

from __future__ import annotations

from ch18.eos_early_exit_common import EosEarlyExitBenchmark, EosEarlyExitConfig
from core.benchmark.wrapper_utils import attach_benchmark_metadata


def get_benchmark() -> EosEarlyExitBenchmark:
    cfg = EosEarlyExitConfig(
        stop_on_all_done=False,
        label="baseline_eos_early_exit",
    )
    return attach_benchmark_metadata(EosEarlyExitBenchmark(cfg), __file__)  # type: ignore[return-value]
