"""Submit equivalent device NVSHMEM exchanges one epoch at a time."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.nvshmem_device.native_benchmark import NvshmemDeviceBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(NvshmemDeviceBenchmark(persistent=False), __file__)
