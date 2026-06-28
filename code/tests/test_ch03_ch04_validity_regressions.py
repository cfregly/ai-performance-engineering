from __future__ import annotations

import pytest
import torch

from ch03.optimized_rack_prep import OptimizedRackPrepBenchmark
from ch04.baseline_cpu_reduction import BaselineCpuReductionBenchmark
from ch04.baseline_nccl import BaselineNcclBenchmark
from ch04.gradient_compression_common import GradientCompressionBenchmark
from ch04.optimized_cpu_reduction import OptimizedGpuReductionBenchmark


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_optimized_rack_prep_uses_wall_clock_timing() -> None:
    bench = OptimizedRackPrepBenchmark()
    config = bench.get_config()
    assert config.timing_method == "wall_clock"
    assert config.full_device_sync is True


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineCpuReductionBenchmark,
        BaselineNcclBenchmark,
        OptimizedGpuReductionBenchmark,
    ],
)
def test_cpu_reduction_setup_keeps_public_output_empty(benchmark_cls: type) -> None:
    bench = benchmark_cls()
    bench.batch_size = 32
    bench.hidden_dim = 16
    bench.inner_dim = 32
    bench.num_shards = 4
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        output_ptr = bench._output_buffer.data_ptr()
        if isinstance(bench, (BaselineCpuReductionBenchmark, BaselineNcclBenchmark)):
            assert len(bench._cpu_shard_buffers) == bench.num_shards
            shard_ptrs = [shard.data_ptr() for shard in bench._cpu_shard_buffers]
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
        assert bench.output.data_ptr() == output_ptr
        if isinstance(bench, (BaselineCpuReductionBenchmark, BaselineNcclBenchmark)):
            assert [shard.data_ptr() for shard in bench._cpu_shard_buffers] == shard_ptrs
    finally:
        bench.teardown()


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("compression", "use_prealloc_buffers", "bucket_mb"),
    [
        ("none", True, 0),
        ("fp16", False, 1),
        ("int8", False, 1),
    ],
)
def test_gradient_compression_setup_keeps_public_output_empty(
    compression: str,
    use_prealloc_buffers: bool,
    bucket_mb: int,
) -> None:
    bench = GradientCompressionBenchmark(
        compression=compression,
        equivalence_group="test_gradient_compression_validity",
        output_tolerance=(1e-3, 1e-3),
        tensor_size_mb=1,
        multi_gpu=False,
        simulate_single_gpu_transfer=True,
        use_prealloc_buffers=use_prealloc_buffers,
        bucket_mb=bucket_mb,
    )
    try:
        bench.setup()
        assert bench.output is None
        if compression == "none":
            assert bench._fp32_output is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()
