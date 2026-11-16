"""Fuse equivalent device NVSHMEM exchanges into one resident kernel."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.nvshmem_device.native_benchmark import NvshmemDeviceBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(NvshmemDeviceBenchmark(persistent=True), __file__)
