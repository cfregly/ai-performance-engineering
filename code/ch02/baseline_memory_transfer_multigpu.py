"""Python harness wrapper for baseline_memory_transfer_multigpu.cu."""

from __future__ import annotations

from pathlib import Path

from ch02.memory_transfer_multigpu_common import MemoryTransferMultiGPUBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class BaselineMemoryTransferMultigpuBenchmark(MemoryTransferMultiGPUBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_memory_transfer_multigpu",
            friendly_name="Baseline Memory Transfer Multigpu",
        )

    def get_custom_metrics(self) -> dict | None:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMemoryTransferMultigpuBenchmark()

