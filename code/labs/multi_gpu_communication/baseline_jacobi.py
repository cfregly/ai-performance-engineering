"""Serialized native Jacobi halo exchange over the complete numerical workload."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.multi_gpu_communication.native_benchmark import NativeJacobiBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(NativeJacobiBenchmark(overlapped=False), __file__)
