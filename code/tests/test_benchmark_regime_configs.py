import pytest
import torch

from ch04.baseline_symmetric_memory_perf import BaselineSymmetricMemoryPerfBenchmark
from ch04.optimized_symmetric_memory_perf import OptimizedSymmetricMemoryPerfBenchmark
from ch05.baseline_ai import BaselineAIBenchmark
from ch05.optimized_ai import OptimizedAIBenchmark
from ch11.baseline_tensor_cores_streams import BaselineTensorCoresStreamsBenchmark
from ch11.optimized_tensor_cores_streams import OptimizedTensorCoresStreamsBenchmark
from ch15.allreduce_rmsnorm_common import AllReduceRMSNormConfig
from ch15.baseline_allreduce_rmsnorm import BaselineAllReduceRMSNormBenchmark
from ch15.optimized_allreduce_rmsnorm import OptimizedAllReduceRMSNormBenchmark
from ch17.baseline_prefill_decode_disagg_tpot_long import TPOT_LONG_CONFIG as TPOT_LONG_BASELINE
from ch17.optimized_prefill_decode_disagg_tpot_long import TPOT_LONG_CONFIG as TPOT_LONG_OPTIMIZED
from ch19.baseline_vectorization_memory import VectorizationBenchmark
from ch19.optimized_vectorization_memory import OptimizedVectorizationMemoryBenchmark


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required by setup")
def test_symmetric_memory_perf_uses_small_allocation_dominated_regime() -> None:
    baseline = BaselineSymmetricMemoryPerfBenchmark()
    optimized = OptimizedSymmetricMemoryPerfBenchmark()

    assert baseline.size_mb == optimized.size_mb == 0.0625
    assert baseline.get_config().iterations == optimized.get_config().iterations == 50
    assert baseline.get_config().warmup == optimized.get_config().warmup == 10

    baseline.setup()
    optimized.setup()
    try:
        assert tuple(baseline._verify_input.shape) == tuple(optimized._verify_input.shape) == (128, 128)
    finally:
        baseline.teardown()
        optimized.teardown()


def test_ai_benchmarks_use_launch_bound_regime() -> None:
    baseline = BaselineAIBenchmark()
    optimized = OptimizedAIBenchmark()

    assert baseline.batch == optimized.batch == 64
    assert baseline.hidden == optimized.hidden == 32
    assert baseline.num_blocks == optimized.num_blocks == 256


def test_tensor_core_streams_use_overlap_friendly_regime() -> None:
    baseline = BaselineTensorCoresStreamsBenchmark()
    optimized = OptimizedTensorCoresStreamsBenchmark()

    assert baseline.matrix_dim == optimized.matrix_dim == 768
    assert baseline.num_segments == optimized.num_segments == 24
    assert optimized.num_streams == 6


@pytest.mark.skipif(torch.cuda.is_available(), reason="Checks the real missing-CUDA error")
@pytest.mark.parametrize("benchmark_cls", [
    BaselineTensorCoresStreamsBenchmark, OptimizedTensorCoresStreamsBenchmark,
])
def test_tensor_core_streams_require_cuda_before_setup_allocations(benchmark_cls) -> None:
    benchmark = benchmark_cls()
    with pytest.raises(RuntimeError, match="CUDA required for ch11"):
        benchmark.setup()
    assert benchmark.host_A is None


def test_allreduce_rmsnorm_uses_larger_fusion_workload() -> None:
    cfg = AllReduceRMSNormConfig()

    assert cfg.tp_size == 8
    assert cfg.batch_size == 16
    assert cfg.hidden_size == 4096

    baseline = BaselineAllReduceRMSNormBenchmark()
    optimized = OptimizedAllReduceRMSNormBenchmark()
    assert baseline.get_config().iterations == optimized.get_config().iterations == 20
    assert baseline.get_config().warmup == optimized.get_config().warmup == 10


def test_tpot_long_uses_transfer_heavy_config() -> None:
    assert TPOT_LONG_BASELINE == TPOT_LONG_OPTIMIZED
    assert TPOT_LONG_BASELINE.hidden_size == 1024
    assert TPOT_LONG_BASELINE.num_layers == 1
    assert TPOT_LONG_BASELINE.batch_size == 4
    assert TPOT_LONG_BASELINE.requests_per_rank == 8
    assert TPOT_LONG_BASELINE.context_window == 4096
    assert TPOT_LONG_BASELINE.decode_tokens == 1024


def test_vectorization_memory_uses_same_bandwidth_regime() -> None:
    baseline = VectorizationBenchmark()
    optimized = OptimizedVectorizationMemoryBenchmark()

    assert baseline.N == optimized.N == 67_108_864
    assert baseline.repeats == optimized.repeats == 12
