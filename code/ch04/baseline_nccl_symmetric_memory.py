"""Out-of-place NCCL AllReduce over ordinary cudaMalloc buffers."""

from ch04.nccl_symmetric_memory_benchmark import NcclSymmetricMemoryBenchmark
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(
        NcclSymmetricMemoryBenchmark(arm="cuda_malloc"), __file__
    )
