from __future__ import annotations

import math
from pathlib import Path

import torch

from ch01.baseline_performance import BaselinePerformanceBenchmark
from ch01.optimized_performance_fusion import OptimizedPerformanceFusionBenchmark
from ch02.baseline_cublas import BaselineCublasBenchmark
from ch02.baseline_memory_transfer import BaselineMemoryTransferBenchmark
from ch02.optimized_cublas import OptimizedCublasBenchmark
from ch02.optimized_memory_transfer import OptimizedMemoryTransferBenchmark
from ch05.ai_common import TinyBlock as CommonTinyBlock
from ch05.baseline_ai import BaselineAIBenchmark
from ch05.baseline_ai import TinyBlock as BaselineTinyBlock
from ch05.baseline_host_staged_reduction import BaselineHostStagedReductionBenchmark
from ch05.baseline_storage_cpu import BaselineStorageCpuBenchmark
from ch05.baseline_vectorization import BaselineVectorizationBenchmark
from ch05.optimized_ai import OptimizedAIBenchmark
from ch05.optimized_ai import TinyBlock as OptimizedTinyBlock
from ch05.optimized_host_staged_reduction import OptimizedHostStagedReductionBenchmark
from ch05.optimized_storage_cpu import OptimizedStorageCpuBenchmark
from ch05.optimized_vectorization import OptimizedVectorizationBenchmark
from ch10.baseline_batch import BaselineBatchBenchmark
from ch10.baseline_double_buffered_pipeline import BaselineDoubleBufferedPipelineBenchmark
from ch10.baseline_flash_attention import BaselineFlashAttentionBenchmark
from ch10.baseline_matmul_tcgen05_pipelined import BaselineMatmulTCGen05PipelinedBenchmark
from ch10.baseline_warp_spec_pingpong import BaselineWarpSpecPingPongBenchmark
from ch10.baseline_warp_specialized_cluster_pipeline import (
    BaselineWarpSpecializedClusterPipelineBenchmark,
)
from ch10.baseline_warp_specialized_pipeline import BaselineWarpSpecializedPipelineBenchmark
from ch10.baseline_warp_specialized_pipeline_enhanced import (
    BaselineWarpSpecializedPipelineEnhancedBenchmark,
)
from ch10.optimized_batch import OptimizedBatchBenchmark
from ch10.optimized_double_buffered_pipeline import OptimizedDoubleBufferedPipelineBenchmark
from ch10.optimized_flash_attention import OptimizedFlashAttentionBenchmark
from ch10.optimized_matmul_tcgen05_pipelined import OptimizedMatmulTCGen05PipelinedBenchmark
from ch10.optimized_tcgen05_cluster_pipeline import OptimizedTcgen05ClusterPipelineBenchmark
from ch10.optimized_warp_spec_pingpong import OptimizedWarpSpecPingPongBenchmark
from ch10.optimized_warp_specialized_cluster_pipeline import (
    OptimizedWarpSpecializedClusterPipelineBenchmark,
)
from ch10.optimized_warp_specialized_pipeline import OptimizedWarpSpecializedPipelineBenchmark
from ch10.optimized_warp_specialized_pipeline_enhanced import (
    OptimizedWarpSpecializedPipelineEnhancedBenchmark,
)
from ch15.baseline_inference_monolithic import SimpleLLM as BaselineMonolithicLLM
from ch15.inference_monolithic_common import SimpleLLM as CommonMonolithicLLM
from ch15.optimized_inference_monolithic import SimpleLLM as OptimizedMonolithicLLM
from ch17.baseline_prefill_decode_disagg import SimpleLLM as BaselineDisaggLLM
from ch17.optimized_prefill_decode_disagg import SimpleLLM as OptimizedDisaggLLM
from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM as CommonDisaggLLM
from core.scripts.update_custom_metrics import CHAPTER_METRIC_HELPERS, audit_repo_custom_metrics
from labs.flexattention.baseline_flex_attention import BaselineFlexAttentionBenchmark
from labs.flexattention.optimized_flex_attention import OptimizedFlexAttentionBenchmark


def test_ch01_environment_metrics_use_runtime_detection() -> None:
    expected_gpu_count = float(torch.cuda.device_count() if torch.cuda.is_available() else 0)

    baseline_metrics = BaselinePerformanceBenchmark().get_custom_metrics()
    optimized_metrics = OptimizedPerformanceFusionBenchmark().get_custom_metrics()

    assert baseline_metrics["env.gpu_count"] == expected_gpu_count
    assert optimized_metrics["env.gpu_count"] == expected_gpu_count
    if torch.cuda.is_available():
        assert baseline_metrics["env.gpu_memory_gb"] > 0.0
    else:
        assert baseline_metrics["env.gpu_memory_gb"] == 0.0


def test_ch05_ai_metrics_report_workload_shape_not_storage_defaults() -> None:
    baseline = BaselineAIBenchmark()
    baseline.parameter_count = 123_456
    optimized = OptimizedAIBenchmark()
    optimized.parameter_count = 654_321

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert "storage.bytes_read" not in baseline_metrics
    assert baseline_metrics["ai.num_blocks"] == float(baseline.num_blocks)
    assert baseline_metrics["ai.activation_elements_per_iteration"] == float(
        baseline.batch * baseline.hidden * baseline.num_blocks
    )
    assert optimized_metrics["ai.parameters_millions"] == optimized.parameter_count / 1_000_000.0


def test_ch10_flash_attention_metrics_report_attention_shape() -> None:
    baseline = BaselineFlashAttentionBenchmark()
    optimized = OptimizedFlashAttentionBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    expected_score_elements = float(
        baseline.batch_size * baseline.num_heads * baseline.seq_len * baseline.seq_len
    )
    assert "pipeline.max_stage_ms" not in baseline_metrics
    assert baseline_metrics["attention.score_matrix_elements"] == expected_score_elements
    assert optimized_metrics["attention.output_elements"] == float(
        optimized.batch_size * optimized.seq_len * optimized.hidden_dim
    )


def test_ch02_gemm_benchmarks_report_gemm_and_roofline_metrics() -> None:
    baseline = BaselineCublasBenchmark()
    optimized = OptimizedCublasBenchmark()
    baseline._last_elapsed_ms = 2.5
    optimized._last_elapsed_ms = 1.25

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    expected_flops = float(2 * baseline.m * baseline.n * baseline.k)

    assert baseline_metrics["gemm.total_flops"] == expected_flops
    assert optimized_metrics["gemm.total_flops"] == expected_flops
    assert "roofline.achieved_tflops" in baseline_metrics
    assert "roofline.achieved_tflops" in optimized_metrics
    assert "transfer.achieved_gbps" not in baseline_metrics
    assert optimized_metrics["roofline.achieved_tflops"] > baseline_metrics["roofline.achieved_tflops"]


def test_ch02_transfer_benchmarks_still_report_transfer_metrics() -> None:
    baseline = BaselineMemoryTransferBenchmark()
    optimized = OptimizedMemoryTransferBenchmark()
    baseline._last_elapsed_ms = 4.0
    optimized._last_elapsed_ms = 2.0

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert baseline_metrics["transfer.bytes"] == float(baseline.N * 4)
    assert "gemm.total_flops" not in baseline_metrics
    assert optimized_metrics["transfer.achieved_gbps"] > baseline_metrics["transfer.achieved_gbps"]


def test_flexattention_metrics_use_sparse_attention_workload_formula() -> None:
    baseline = BaselineFlexAttentionBenchmark()
    optimized = OptimizedFlexAttentionBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    docs = math.ceil(baseline.seq_len / baseline.doc_span)
    active_pairs_per_head = 0
    for doc_idx in range(docs):
        start = doc_idx * baseline.doc_span
        stop = min(baseline.seq_len, start + baseline.doc_span)
        active_pairs_per_head += (stop - start) ** 2
    expected_pairs = float(baseline.batch * baseline.heads * active_pairs_per_head)
    expected_flops = 4.0 * expected_pairs * float(baseline.head_dim)

    assert baseline_metrics["flex_attention.active_score_pairs"] == expected_pairs
    assert baseline_metrics["flex_attention.total_flops"] == expected_flops
    assert optimized_metrics["flex_attention.total_flops"] == expected_flops
    assert baseline_metrics["flex_attention.arithmetic_intensity"] > 0.0


def test_ch05_remaining_metrics_no_longer_emit_storage_bandwidth_placeholders() -> None:
    baseline_vector = BaselineVectorizationBenchmark().get_custom_metrics()
    optimized_vector = OptimizedVectorizationBenchmark().get_custom_metrics()
    baseline_storage = BaselineStorageCpuBenchmark().get_custom_metrics()
    optimized_storage = OptimizedStorageCpuBenchmark().get_custom_metrics()
    baseline_reduction = BaselineHostStagedReductionBenchmark().get_custom_metrics()
    optimized_reduction = OptimizedHostStagedReductionBenchmark().get_custom_metrics()

    assert "storage.read_gbps" not in baseline_vector
    assert baseline_vector["vectorization.is_vectorized"] == 0.0
    assert optimized_vector["vectorization.is_vectorized"] == 1.0

    assert baseline_storage["storage.uses_cpu_staging"] == 1.0
    assert optimized_storage["storage.uses_cpu_staging"] == 1.0
    assert optimized_storage["storage.simulates_gpu_direct"] == 0.0
    assert "storage.total_gbps" not in baseline_storage

    assert baseline_reduction["reduction.host_staging_round_trips"] == 2.0
    assert optimized_reduction["reduction.keeps_reduction_on_device"] == 1.0


def test_ch05_host_staged_reduction_keeps_scalar_output_without_hot_path_clone() -> None:
    data = torch.arange(16, dtype=torch.float32)
    for benchmark in (
        BaselineHostStagedReductionBenchmark(),
        OptimizedHostStagedReductionBenchmark(),
    ):
        benchmark.device = torch.device("cpu")
        benchmark.data = data
        benchmark.benchmark_fn()

        assert benchmark.output is not None
        assert benchmark.output.shape == ()
        output_ptr = benchmark.output.data_ptr()
        torch.testing.assert_close(benchmark.output, data.sum())
        benchmark.benchmark_fn()
        assert benchmark.output.data_ptr() == output_ptr
        torch.testing.assert_close(benchmark.output, data.sum())


def test_ch10_remaining_metrics_no_longer_emit_fake_pipeline_timing() -> None:
    baseline_batch = BaselineBatchBenchmark().get_custom_metrics()
    optimized_batch = OptimizedBatchBenchmark().get_custom_metrics()
    baseline_binary = BaselineDoubleBufferedPipelineBenchmark().get_custom_metrics()
    optimized_binary = OptimizedDoubleBufferedPipelineBenchmark().get_custom_metrics()

    assert "pipeline.max_stage_ms" not in baseline_batch
    assert baseline_batch["batch.micro_batches"] == float(BaselineBatchBenchmark().micro_batches)
    assert optimized_batch["batch.micro_batches"] == 1.0

    assert "pipeline.max_stage_ms" not in baseline_binary
    assert baseline_binary["pipeline.num_stages"] == 1.0
    assert optimized_binary["pipeline.double_buffered"] == 1.0
    assert optimized_binary["workload.M"] == 2048.0


def test_ch10_tcgen05_metrics_no_longer_emit_legacy_string_placeholders() -> None:
    baseline_pipelined = BaselineMatmulTCGen05PipelinedBenchmark().get_custom_metrics()
    optimized_pipelined = OptimizedMatmulTCGen05PipelinedBenchmark().get_custom_metrics()

    assert "library" not in baseline_pipelined
    assert "matrix_size" not in baseline_pipelined
    assert baseline_pipelined["gemm.uses_tcgen05"] == 1.0
    assert baseline_pipelined["gemm.pipeline_stages"] == 1.0
    assert optimized_pipelined["gemm.no_wait_pipeline"] == 1.0
    assert optimized_pipelined["gemm.total_flops"] == baseline_pipelined["gemm.total_flops"]


def test_ch10_cluster_pipeline_metrics_report_real_workload_shape() -> None:
    if not torch.cuda.is_available():
        return

    metrics = OptimizedTcgen05ClusterPipelineBenchmark().get_custom_metrics()

    assert "cuda_graph_replay" not in metrics
    assert metrics["gemm.uses_tcgen05"] == 1.0
    assert metrics["gemm.cluster_launch"] == 1.0
    assert metrics["gemm.cuda_graph_replay"] == 1.0
    assert metrics["gemm.m"] == float(OptimizedTcgen05ClusterPipelineBenchmark.matrix_rows)


def test_ch10_warp_specialization_wrappers_no_longer_return_empty_placeholder_metrics() -> None:
    baseline_metrics = BaselineWarpSpecializedPipelineBenchmark().get_custom_metrics()
    optimized_metrics = OptimizedWarpSpecializedPipelineBenchmark().get_custom_metrics()
    baseline_enhanced_metrics = BaselineWarpSpecializedPipelineEnhancedBenchmark().get_custom_metrics()
    optimized_enhanced_metrics = OptimizedWarpSpecializedPipelineEnhancedBenchmark().get_custom_metrics()

    assert baseline_metrics["pipeline.warp_specialized"] == 1.0
    assert baseline_metrics["workload.elements"] == float(512 * 64 * 64)
    assert baseline_metrics["pipeline.async_staging"] == 0.0
    assert optimized_metrics["pipeline.num_stages"] == 2.0
    assert optimized_metrics["pipeline.async_staging"] == 1.0

    assert baseline_enhanced_metrics["workload.tile_candidate_count"] == 3.0
    assert baseline_enhanced_metrics["pipeline.adaptive_tile_selection"] == 1.0
    assert optimized_enhanced_metrics["pipeline.uses_pipeline_api"] == 1.0
    assert optimized_enhanced_metrics["pipeline.num_stages"] == 2.0


def test_ch10_cluster_and_pingpong_metrics_report_real_warp_roles() -> None:
    baseline_cluster_metrics = BaselineWarpSpecializedClusterPipelineBenchmark().get_custom_metrics()
    optimized_cluster_metrics = OptimizedWarpSpecializedClusterPipelineBenchmark().get_custom_metrics()
    baseline_pingpong_metrics = BaselineWarpSpecPingPongBenchmark().get_custom_metrics()
    optimized_pingpong_metrics = OptimizedWarpSpecPingPongBenchmark().get_custom_metrics()

    assert baseline_cluster_metrics["workload.cluster_blocks"] == 4.0
    assert baseline_cluster_metrics["pipeline.uses_cluster"] == 1.0
    assert baseline_cluster_metrics["pipeline.async_staging"] == 0.0
    assert optimized_cluster_metrics["pipeline.async_staging"] == 1.0
    assert optimized_cluster_metrics["pipeline.uses_dsmem"] == 1.0

    assert baseline_pingpong_metrics["pipeline.consumer_warps"] == 1.0
    assert baseline_pingpong_metrics["pipeline.pingpong_enabled"] == 0.0
    assert baseline_pingpong_metrics["workload.tile_size"] == 64.0
    assert optimized_pingpong_metrics["pipeline.consumer_warps"] == 2.0
    assert optimized_pingpong_metrics["pipeline.pingpong_enabled"] == 1.0


def test_chapter_models_import_shared_classes() -> None:
    assert BaselineTinyBlock is CommonTinyBlock
    assert OptimizedTinyBlock is CommonTinyBlock
    assert BaselineMonolithicLLM is CommonMonolithicLLM
    assert OptimizedMonolithicLLM is CommonMonolithicLLM
    assert BaselineDisaggLLM is CommonDisaggLLM
    assert OptimizedDisaggLLM is CommonDisaggLLM


def test_update_custom_metrics_no_longer_auto_suggests_bogus_helpers_for_ch5_and_ch10() -> None:
    assert CHAPTER_METRIC_HELPERS[5] is None
    assert CHAPTER_METRIC_HELPERS[10] is None
    assert CHAPTER_METRIC_HELPERS[14] is None
    assert CHAPTER_METRIC_HELPERS[18] is None


def test_repo_wide_custom_metrics_audit_has_no_phantoms() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = audit_repo_custom_metrics(repo_root)
    phantoms = [result for result in results if result["analysis"]["classification"] == "phantom"]
    assert not phantoms
