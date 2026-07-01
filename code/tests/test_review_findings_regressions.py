from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from ch08.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA
from ch08.tiling_benchmark_base import TilingBenchmarkBase
from ch15.speculative_decoding_benchmarks import SpeculativeDecodingBenchmark
from core.discovery import chapter_slug, discover_all_chapters, discover_benchmarks
from core.harness.run_benchmarks import INFORMATIONAL_BENCHMARKS
from scripts.canonical_queue_batches import (
    CAPABILITY_VALIDATION_BATCH,
    CHAPTER_DRIFT_TRIAGE,
    CHAPTER_EXPECTATION_BATCH,
    LAB_FAMILY_BATCHES,
)
from scripts.full_virtualized_rerun import (
    EXPECTED_UNSUPPORTED_PORTABLE_REASON,
    EXPECTED_UNSUPPORTED_RUNTIME_REASON,
    _backfill_written_expectation_total,
    _canonicalize_state,
    _expectation_example_key,
    _expected_unsupported_portable_reason,
    _is_informational_benchmark,
    _persist_state,
    _queue_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _setup_section(rel_path: str) -> str:
    text = _read(rel_path)
    return text.split("def benchmark_fn", 1)[0]


def _benchmark_section(rel_path: str) -> str:
    text = _read(rel_path)
    return text.split("def benchmark_fn", 1)[1].split("def capture_verification_payload", 1)[0]


def _registered_targets() -> set[str]:
    targets: set[str] = set()
    for chapter_dir in discover_all_chapters(REPO_ROOT, bench_roots=[REPO_ROOT]):
        chapter_id = chapter_slug(chapter_dir, REPO_ROOT, bench_root=REPO_ROOT)
        for _, _, example in discover_benchmarks(chapter_dir):
            targets.add(f"{chapter_id}:{example}")
    return targets


def test_ch02_cublas_setup_keeps_warmup_symmetric() -> None:
    baseline_setup = _setup_section("ch02/baseline_cublas.py")
    optimized_setup = _setup_section("ch02/optimized_cublas.py")

    for setup_text in (baseline_setup, optimized_setup):
        assert "for _ in range(10):" in setup_text
        assert "_ = torch.matmul(self.A, self.B)" in setup_text

    assert "warm cuBLAS identically" in baseline_setup
    assert "warmed-up heuristics" in optimized_setup


def test_ch02_cublas_benchmark_fn_uses_shared_nvtx_helper_symmetrically() -> None:
    baseline_bench = _benchmark_section("ch02/baseline_cublas.py")
    optimized_bench = _benchmark_section("ch02/optimized_cublas.py")

    assert 'with torch.inference_mode(), self._nvtx_range("baseline_cublas_fp32"):' in baseline_bench
    assert 'with torch.inference_mode(), self._nvtx_range("optimized_cublas_tf32"):' in optimized_bench
    assert "torch.no_grad()" not in baseline_bench
    assert "torch.no_grad()" not in optimized_bench
    assert "core.profiling.nvtx_helper" not in optimized_bench
    assert "get_config()" not in optimized_bench
    assert "get_nvtx_enabled" not in optimized_bench


def test_ch01_performance_workload_stays_on_retuned_hidden_dim() -> None:
    workload_text = _read("ch01/workload_config.py")

    assert "performance_microbatches: int = 128" in workload_text
    assert "performance_hidden_dim: int = 16384" in workload_text


def test_ch06_attention_ilp_pair_keeps_math_fixed_and_only_changes_ilp_schedule() -> None:
    baseline_text = _read("ch06/baseline_attention_ilp.py")
    optimized_text = _read("ch06/optimized_attention_ilp.py")
    workload_text = _read("ch06/workload_config.py")
    readme_text = _read("ch06/README.md")

    for source in (baseline_text, optimized_text):
        assert "load_ilp_extension" in source
        assert "self.attention_terms = (query * key * 0.1).contiguous().reshape(-1)" in source
        assert "WORKLOAD" in source
        assert "MultiheadAttention" not in source
        assert "scaled_dot_product_attention" not in source
        assert "torch.cuda.Stream" not in source

    assert "self._extension.sequential_ops(dst, src)" in baseline_text
    assert "self._extension.unrolled_ilp(dst, src)" in optimized_text
    assert '"attention_ilp.independent_chains_per_thread": 1.0' in baseline_text
    assert '"attention_ilp.independent_chains_per_thread": 4.0' in optimized_text
    assert "attention_batch: int = 8" in workload_text
    assert "attention_embed_dim: int = 1024" in workload_text
    assert "attention_heads: int = 16" in workload_text
    assert "attention_tokens: int = 2048" in workload_text
    assert "keep the math fixed while changing independent chains per thread" in readme_text


def test_ch06_ilp_benchmarks_defer_verification_clone_out_of_hot_path() -> None:
    for relative_path in (
        "ch06/baseline_elementwise_ilp.py",
        "ch06/optimized_elementwise_ilp.py",
        "ch06/baseline_attention_ilp.py",
        "ch06/optimized_attention_ilp.py",
    ):
        source_text = _read(relative_path)
        benchmark_text = _benchmark_section(relative_path)
        capture_text = source_text.split("def capture_verification_payload", 1)[1]
        probe_size = "4096" if "attention" in relative_path else "1024"

        assert "self._output_view0: Optional[torch.Tensor] = None" in source_text
        assert "self._output_view1: Optional[torch.Tensor] = None" in source_text
        assert f"self._output_view0 = self._buf0[:{probe_size}]" in source_text
        assert f"self._output_view1 = self._buf1[:{probe_size}]" in source_text
        assert ".clone()" not in benchmark_text
        assert "self.output = src[:" not in benchmark_text
        assert "self.output = self._output_view0 if src is buf0 else self._output_view1" in benchmark_text
        assert "output=self.output.detach().clone()" in capture_text


def test_ch17_static_routing_reuses_verification_output_buffer() -> None:
    for relative_path in (
        "ch17/baseline_routing_static.py",
        "ch17/optimized_routing_static.py",
    ):
        source_text = _read(relative_path)
        benchmark_text = _benchmark_section(relative_path)
        setup_text = source_text.split("def setup", 1)[1].split("def benchmark_fn", 1)[0]
        capture_text = source_text.split("def capture_verification_payload", 1)[1]

        assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source_text
        assert "self._verify_output_buffer = torch.empty(" in setup_text
        assert ".clone()" not in benchmark_text
        assert ".float()" not in benchmark_text
        assert "self._verify_output_buffer.copy_(self.output)" in capture_text
        assert "output=self._verify_output_buffer" in capture_text
        assert "output=self.output.detach().float().clone()" not in capture_text


def test_ch06_optimized_adaptive_uses_chunk_plan_without_extra_staging_buffers() -> None:
    optimized_text = _read("ch06/optimized_adaptive.py")

    assert "self.chunk_plan: list[tuple[int, int]] = []" in optimized_text
    assert "self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []" in optimized_text
    assert "self._output_buffer = torch.empty_like(self.input)" in optimized_text
    assert "for start, end in self.chunk_plan" in optimized_text
    assert "self._chunk_views = [" in optimized_text
    assert "for window, out_window in self._chunk_views:" in optimized_text
    assert "self._transform(window, out_window)" in optimized_text
    assert "self._output_buffer[start:end].copy_(transformed)" not in optimized_text

    for forbidden in ("host_buffer", "device_buffer", "pin_memory", "torch.cuda.Stream"):
        assert forbidden not in optimized_text


def test_ch08_bridge_comparison_pairs_are_explicitly_marked_in_structured_metrics() -> None:
    threshold = object.__new__(ThresholdBenchmarkBaseTMA)
    threshold.rows = ThresholdBenchmarkBaseTMA.rows
    threshold.threshold = ThresholdBenchmarkBaseTMA.threshold
    threshold.inner_iterations = ThresholdBenchmarkBaseTMA.inner_iterations
    threshold_metrics = threshold.get_custom_metrics()
    assert threshold_metrics["story.comparison_pair"] == 1.0
    assert threshold_metrics["story.chapter_native_exemplar"] == 0.0
    assert threshold_metrics["story.bridge_to_ch10"] == 1.0

    tiling = object.__new__(TilingBenchmarkBase)
    tiling.matrix_rows = TilingBenchmarkBase.matrix_rows
    tiling.shared_dim = TilingBenchmarkBase.shared_dim
    tiling.matrix_cols = TilingBenchmarkBase.matrix_cols
    tiling.inner_iterations = TilingBenchmarkBase.inner_iterations
    tiling.nvtx_label = "tiling"
    tiling_metrics = tiling.get_custom_metrics()
    assert tiling_metrics["story.comparison_pair"] == 1.0
    assert tiling_metrics["story.chapter_native_exemplar"] == 0.0
    assert tiling_metrics["story.bridge_to_ch09"] == 1.0

    tcgen05 = object.__new__(Tcgen05CustomVsCublasBase)
    tcgen05.matrix_rows = Tcgen05CustomVsCublasBase.matrix_rows
    tcgen05.shared_dim = Tcgen05CustomVsCublasBase.shared_dim
    tcgen05.matrix_cols = Tcgen05CustomVsCublasBase.matrix_cols
    tcgen05_metrics = tcgen05.get_custom_metrics()
    assert tcgen05_metrics["story.comparison_pair"] == 1.0
    assert tcgen05_metrics["story.chapter_native_exemplar"] == 0.0
    assert tcgen05_metrics["story.bridge_to_ch09"] == 1.0

    baseline_nvfp4_text = _read("ch08/baseline_nvfp4_mlp.py")
    optimized_nvfp4_text = _read("ch08/optimized_nvfp4_mlp.py")
    for source in (baseline_nvfp4_text, optimized_nvfp4_text):
        assert '"story.comparison_pair": 1.0' in source
        assert '"story.chapter_native_exemplar": 0.0' in source
        assert '"story.bridge_to_ch09": 1.0' in source


def test_ch08_threshold_tma_bridge_workload_uses_larger_row_count() -> None:
    threshold_base_text = _read("ch08/threshold_benchmark_base.py")
    validate_section = threshold_base_text.split("def _validate_correctness", maxsplit=1)[1].split(
        "def get_config",
        maxsplit=1,
    )[0]

    assert "rows: int = 1 << 26" in threshold_base_text
    assert "torch.full_like(self.inputs" not in validate_section
    assert "torch.zeros_like(self.inputs)" not in validate_section
    assert "scale = torch.where(outer, THRESHOLD_OUTER_SCALE, THRESHOLD_INNER_SCALE)" in validate_section
    assert "reference.copysign_(self.inputs)" in validate_section
    assert "reference.masked_fill_(active.logical_not_(), 0.0)" in validate_section


def test_ch08_threshold_demos_use_relu_without_zero_tensor() -> None:
    for filename in ("threshold_op.py", "jit_threshold_op.py"):
        source = _read(f"ch08/{filename}")
        function_section = source.split("def threshold_op", maxsplit=1)[1].split(
            "def main",
            maxsplit=1,
        )[0]

        assert "torch.zeros_like(x)" not in function_section
        assert "torch.maximum(x" not in function_section
        assert "return torch.relu(x)" in function_section


def test_ch08_mask_strategy_demo_reuses_output_workspaces() -> None:
    source = _read("ch08/warp_divergence_pytorch.py")
    mask_section = source.split("def compare_mask_strategies", maxsplit=1)[1].split(
        "def compiled_conditionals",
        maxsplit=1,
    )[0]

    assert "zeros = torch.zeros_like(data)" not in mask_section
    assert "return torch.where(mask, processed, zeros)" not in mask_section
    assert "result = zeros.clone()" not in mask_section
    assert "all_output = torch.empty_like(data)" in mask_section
    assert "active_output = torch.empty_like(data)" in mask_section
    assert "active_data = torch.empty(active_indices.numel(), device=device, dtype=data.dtype)" in mask_section
    assert "active_scratch = torch.empty_like(active_data)" in mask_section
    assert "torch.sin(data, out=all_output)" in mask_section
    assert "torch.index_select(data, 0, active_indices, out=active_data)" in mask_section
    assert "torch.cos(active_data, out=active_scratch)" in mask_section
    assert "torch.sin(active_data, out=active_data)" in mask_section
    assert "active_data.mul_(active_scratch)" in mask_section
    assert "active_output.index_copy_(0, active_indices, active_data)" in mask_section
    assert "active_data = data[active_indices]" not in mask_section
    assert "processed = torch.sin(active_data) * torch.cos(active_data)" not in mask_section
    assert "torch.sin(data[active_indices])" not in mask_section
    assert "torch.cos(data[active_indices])" not in mask_section
    assert "all_output.masked_fill_(inactive, 0.0)" in mask_section
    assert "active_output.zero_()" in mask_section
    assert "scalar_tensor_to_float(torch.max(torch.abs(res_all - res_active)))" in mask_section
    assert "torch.max(torch.abs(res_all - res_active)).item()" not in mask_section

    compiled_section = source.split("def compiled_conditionals", maxsplit=1)[1]
    assert "max_diff = scalar_tensor_to_float(" in compiled_section
    assert (
        "torch.max(torch.abs(uncompiled(x, y, threshold) - compiled(x, y, threshold))).item()"
        not in compiled_section
    )


def test_ch08_tiling_optimized_wrapper_uses_strict_fast_path() -> None:
    optimized_tiling = _read("ch08/optimized_tiling.py")

    assert "matmul_tiled_fast(self.matrix_a, self.matrix_b, self.output)" in optimized_tiling
    assert "matmul_tiled(self.matrix_a, self.matrix_b, self.output)" not in optimized_tiling


def test_ch08_loop_unrolling_binaries_share_identical_input_initialization() -> None:
    baseline_text = _read("ch08/baseline_loop_unrolling.cu")
    optimized_text = _read("ch08/optimized_loop_unrolling.cu")
    common_text = _read("ch08/loop_unrolling_common.cuh")

    for source in (baseline_text, optimized_text):
        assert "init_input_value(i)" in source
        assert "init_weight_value(i)" in source

    assert "constexpr int kInputModulo = 1024" in common_text
    assert "constexpr float kWeightBase = 0.5f" in common_text


def test_ch08_readme_calls_out_bridge_comparisons_and_historical_tcgen05_naming() -> None:
    readme_text = _read("ch08/README.md")
    baseline_tcgen05_text = _read("ch08/baseline_tcgen05_custom_vs_cublas.py")
    optimized_tcgen05_text = _read("ch08/optimized_tcgen05_custom_vs_cublas.py")

    assert "chapter-native exemplars" in readme_text
    assert "`thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp`" in readme_text
    assert "custom-versus-library comparison target" in readme_text
    assert "supplementary comparison benchmark with a local contract" in readme_text
    assert "matmul_tiled_fast" in readme_text
    assert "historical baseline/optimized filenames" not in readme_text
    assert "tcgen05-versus-cuBLAS bridge comparison" in baseline_tcgen05_text
    assert "Vendor cuBLAS reference side of the comparison pair." in baseline_tcgen05_text
    assert "Custom tcgen05 kernel side of the comparison pair." in optimized_tcgen05_text


def test_ch08_tcgen05_custom_vs_cublas_is_not_informational() -> None:
    assert "tcgen05_custom_vs_cublas" not in INFORMATIONAL_BENCHMARKS.get("ch08", set())


def test_ch06_launch_bounds_pair_uses_local_contract_small_effect_case() -> None:
    readme_text = _read("ch06/README.md")
    baseline_text = _read("ch06/baseline_launch_bounds.py")
    optimized_text = _read("ch06/optimized_launch_bounds.py")

    assert "launch_bounds" not in INFORMATIONAL_BENCHMARKS.get("ch06", set())
    assert "launch_bounds_cuda" not in INFORMATIONAL_BENCHMARKS.get("ch06", set())
    assert "small-effect teaching cases" in readme_text
    assert "local expectation-backed contracts" in readme_text
    assert "self.kernel_launches_per_timed_call = 96" in baseline_text
    assert "self.kernel_launches_per_timed_call = 96" in optimized_text


def test_ch04_gradient_fusion_batches_reductions_per_timed_call() -> None:
    common_text = _read("ch04/gradient_fusion_common.py")

    assert "reduction_repeats: int = 16" in common_text
    assert "requests_per_iteration=float(self.reduction_repeats)" in common_text
    assert "tokens_per_iteration=float(total_bytes * self.reduction_repeats)" in common_text
    assert "self._repeat_tail_range = range(1, self.reduction_repeats)" in common_text
    assert "for _ in self._repeat_tail_range:" in common_text
    assert "self._sum_buffer = torch.empty_like(self._accum_buffer)" in common_text
    assert "torch.sum(self.fused_tensor, dim=None, out=accum)" in common_text
    assert "self._seed_tensor = self.tensors[0]" in common_text
    assert "self._tail_tensors = self.tensors[1:]" in common_text
    assert "torch.sum(self._seed_tensor, dim=None, out=accum)" in common_text
    assert "for tensor in self._tail_tensors:" in common_text
    assert "torch.sum(self.fused_tensor, dim=None, out=sum_buffer)" in common_text
    assert "torch.sum(tensor, dim=None, out=sum_buffer)" in common_text
    assert ".sum()" not in common_text.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]


def test_ch08_tiling_bridge_comparison_batches_enough_inner_iterations() -> None:
    tiling_base = _read("ch08/tiling_benchmark_base.py")

    assert "inner_iterations: int = 12" in tiling_base


def test_occupancy_tuning_low_warp_reference_schedule_uses_local_contract() -> None:
    readme_text = _read("labs/occupancy_tuning/README.md")
    schedule_text = _read("labs/occupancy_tuning/triton_matmul_schedules.py")

    assert "proton_matmul_bm64_bn64_bk32_nw2" not in INFORMATIONAL_BENCHMARKS.get("occupancy_tuning", set())
    assert "verifying Proton vs Nsight agreement" in schedule_text
    assert "supplementary comparison schedule benchmark" in readme_text
    assert "canonical speed claims stay on" in readme_text


def test_nvfp4_group_gemm_shape_surface_uses_frontdoor_and_comparison_companions() -> None:
    readme_text = _read("labs/nvfp4_group_gemm/README.md")

    assert "nvfp4_group_gemm" not in INFORMATIONAL_BENCHMARKS.get("nvfp4_group_gemm", set())
    assert "nvfp4_group_gemm_g8_n7168_k2048" not in INFORMATIONAL_BENCHMARKS.get("nvfp4_group_gemm", set())
    assert "nvfp4_group_gemm_g2_n3072_k4096" not in INFORMATIONAL_BENCHMARKS.get("nvfp4_group_gemm", set())
    assert "nvfp4_group_gemm_g2_n4096_k1536" not in INFORMATIONAL_BENCHMARKS.get("nvfp4_group_gemm", set())
    assert "canonical local-contract speed benchmark" in readme_text
    assert "supplementary comparison benchmark" in readme_text
    assert "older strict all-case snapshots" in readme_text
    assert "former competition `caseN` numbering is retired" in readme_text


def test_ch10_double_buffered_pipeline_baseline_is_book_aligned_naive_gemm() -> None:
    baseline_source = _read("ch10/baseline_double_buffered_pipeline.cu")
    optimized_source = _read("ch10/optimized_double_buffered_pipeline.cu")
    baseline_wrapper = _read("ch10/baseline_double_buffered_pipeline.py")

    assert "gemm_naive_kernel" in baseline_source
    assert "global memory" in baseline_source
    assert "gemm_single_buffered_kernel" not in baseline_source
    assert "constexpr int CHUNK_K = 32;" in optimized_source
    assert "const bool full_tile =" in optimized_source
    assert "if (full_tile) {" in optimized_source
    assert "if (chunk_base + kk >= K)" in optimized_source
    assert 'double_buffered=False' in baseline_wrapper
    assert 'num_stages=1' in baseline_wrapper


def test_ch10_atomic_reduction_explicitly_reports_timed_memset_cost() -> None:
    optimized_wrapper = _read("ch10/optimized_atomic_reduction.py")
    optimized_source = _read("ch10/optimized_atomic_reduction.cu")

    assert "timed_output_reset_memset=True" in optimized_wrapper
    assert "timed_output_reset_bytes=4096.0" in optimized_wrapper
    assert "Timing includes cudaMemset(d_output, 0, ...)" in optimized_source


def test_ch12_conditional_graphs_optimized_path_keeps_runtime_condition_inside_graph() -> None:
    optimized_source = _read("ch12/optimized_cuda_graphs_conditional.cu")

    assert "conditional_dispatch_kernel" in optimized_source
    assert "predicate_kernel<<<1, 1, 0, graph_stream>>>(d_condition, d_data, THRESHOLD);" in optimized_source
    assert "conditional_dispatch_kernel<<<grid, block, 0, graph_stream>>>(" in optimized_source
    assert "expensive_kernel<<<grid, block, 0, graph_stream>>>(d_data, N, 1.01f);" not in optimized_source


def test_ch14_cutlass_pair_is_renamed_to_explicit_cublas_vs_cutlass() -> None:
    baseline_source = _read("ch14/baseline_cublas_vs_cutlass.py")
    binding_source = _read("core/benchmark/cutlass_binding.py")
    extension_source = _read("core/benchmark/cuda/cutlass_gemm_extension.cu")

    assert "BaselineCublasVsCutlassBenchmark" in baseline_source
    assert "from core.benchmark.cutlass_binding import cublas_gemm_fp16" in baseline_source
    assert "def cublas_gemm_fp16" in binding_source
    assert "torch::Tensor cublas_gemm_fp16" in extension_source
    assert "cublas_vs_cutlass" in INFORMATIONAL_BENCHMARKS["ch14"]


def test_ch14_model_compile_pair_uses_reduced_precision_name_not_bf16_alias() -> None:
    baseline_source = _read("ch14/baseline_model_compile_reduced_precision.py")
    optimized_source = _read("ch14/optimized_model_compile_reduced_precision.py")

    assert "BaselineModelCompileReducedPrecisionBenchmark" in baseline_source
    assert "OptimizedModelCompileReducedPrecisionBenchmark" in optimized_source
    assert "signature_equivalence_group = \"ch14_model_compile_reduced_precision\"" in baseline_source
    assert "model_compile_reduced_precision_optimized" in optimized_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for speculative decoding stability check")
def test_ch15_speculative_decoding_acceptance_metrics_are_stable_run_to_run() -> None:
    def _run_acceptance_rate() -> float:
        bench = SpeculativeDecodingBenchmark(use_speculative=True, label="speculative_decode_stability")
        bench.workload = replace(
            bench.workload,
            target_hidden=512,
            target_layers=1,
            draft_hidden=128,
            speculative_k=8,
            total_tokens=32,
        )
        try:
            bench.setup()
            bench.benchmark_fn()
            metrics = bench.get_custom_metrics()
            assert metrics is not None
            return metrics["speculative.acceptance_rate_pct"]
        finally:
            bench.teardown()

    first = _run_acceptance_rate()
    second = _run_acceptance_rate()
    assert first == second


def test_ch15_split_moe_targets_isolate_dispatch_from_routing() -> None:
    dispatch_baseline = _read("ch15/baseline_moe_dispatch.py")
    dispatch_optimized = _read("ch15/optimized_moe_dispatch.py")
    routing_baseline = _read("ch15/baseline_moe_routing_topology_aware.py")
    routing_optimized = _read("ch15/optimized_moe_routing_topology_aware.py")
    readme_text = _read("ch15/README.md")

    assert 'route_mode = "uniform"' in dispatch_baseline
    assert 'route_mode = "uniform"' in dispatch_optimized
    assert 'dispatch_mode = "mask_scan"' in dispatch_baseline
    assert 'dispatch_mode = "active_experts"' in dispatch_optimized
    assert 'dispatch_mode = "mask_scan"' in routing_baseline
    assert 'dispatch_mode = "mask_scan"' in routing_optimized
    assert 'route_mode = "uniform"' in routing_baseline
    assert 'route_mode = "topology_aware"' in routing_optimized
    assert "baseline_moe_routing_simple.py" not in readme_text
    assert "baseline_moe_dispatch.py" in readme_text
    assert "baseline_moe_routing_topology_aware.py" in readme_text


def test_ch15_guided_decoding_defaults_stay_on_heavier_mask_reuse_workload() -> None:
    common_text = _read("ch15/guided_decoding_common.py")

    assert "batch_size: int = 32" in common_text
    assert "steps: int = 96" in common_text
    assert "vocab_size: int = 65536" in common_text
    assert "allowed_count: int = 8192" in common_text


def test_ch18_split_paged_attention_targets_isolate_backend_from_layout() -> None:
    common_text = _read("ch18/paged_attn_split_common.py")
    readme_text = _read("ch18/README.md")

    assert "class DensePagedAttnBase" in common_text
    assert "class LayoutPagedAttnBase" in common_text
    assert 'metrics["paged_attn.backend_math"] = 1.0 if self.backend == "math" else 0.0' in common_text
    assert "def _build_block_table" in common_text
    assert "return (block_ids.unsqueeze(0) - batch_offsets).remainder_(num_blocks)" in common_text
    assert "return torch.stack(" not in common_text
    assert "return create_block_mask(" in common_text
    assert "dense masked decode versus block-table-driven FlexAttention sparse kernels" in readme_text
    assert "fused FlexAttention block-mask kernel" in readme_text
    assert "baseline_paged_attn_backend.py" in readme_text
    assert "baseline_paged_attn_layout.py" in readme_text
    assert "optimized_paged_attn_vllm.py" not in readme_text


def test_ch16_blackwell_dense_attention_variant_is_explicitly_noncanonical() -> None:
    readme_text = _read("ch16/README.md")
    source = _read("ch16/optimized_dense_attention_flash_blackwell_variant.py")

    assert "dense_attention_flash_blackwell_variant" in readme_text
    assert "non-canonical hardware variant" in readme_text
    assert "story_metadata" in source
    assert '"variant"' in source
    assert "dense_attention_flash_blackwell_variant" in INFORMATIONAL_BENCHMARKS["ch16"]


def test_reviewed_pair_fixes_remain_applied() -> None:
    baseline_regional = _read("ch14/baseline_regional_triton.py")
    baseline_regional_setup = _setup_section("ch14/baseline_regional_triton.py")
    baseline_regional_bench = _benchmark_section("ch14/baseline_regional_triton.py")
    sliding_window = _read("ch14/optimized_sliding_window.py")
    blackwell = _read("ch16/optimized_dense_attention_flash_blackwell_variant.py")
    baseline_memory = _read("ch17/baseline_memory.py")
    optimized_memory = _read("ch17/optimized_memory.py")
    fp4_baseline = _read("ch19/baseline_fp4_weight_quantization.py")
    baseline_kv = _read("ch20/baseline_integrated_kv_cache.py")
    optimized_kv = _read("ch20/optimized_integrated_kv_cache.py")
    optimized_memory_standard = _read("ch20/optimized_memory_standard.py")
    baseline_memory_standard = _read("ch20/baseline_memory_standard.py")
    baseline_pipeline_bench = _benchmark_section("ch20/baseline_pipeline_sequential.py")

    assert "self._compiled_model = torch.compile(" in baseline_regional_setup
    assert "torch.compile(" not in baseline_regional_bench
    assert "allowed_benchmark_fn_antipatterns" not in baseline_regional

    assert "iterations=20" in sliding_window
    assert "warmup=5" in sliding_window
    assert "full causal SDPA" in sliding_window
    assert "historical" in sliding_window

    assert "fp8_kv" not in blackwell
    assert "FP8 KV cache benefits" not in blackwell
    assert '"fp8": False' in blackwell

    for source in (baseline_memory, optimized_memory):
        assert "iterations=10" in source
        assert "warmup=5" in source

    assert "_ = weight.sum()" not in fp4_baseline

    assert "self._request_token_groups = [" in baseline_kv
    assert "for request_id, token_views in self._request_token_groups:" in baseline_kv
    assert "for pos, token in token_views:" in baseline_kv
    assert "self._request_block_groups = [" in optimized_kv
    assert "for request_id, seq_len, block_views in self._request_block_groups:" in optimized_kv
    assert "for pos, block_view in block_views:" in optimized_kv
    assert "self.output = hidden[:, -1:, :] if hidden is not None else None" in optimized_kv
    assert "hidden[:, -1:, :].detach()" not in optimized_kv

    assert "class OptimizedMemoryStandardBenchmark" in optimized_memory_standard
    assert "OptimizedMemoryHBM3eBenchmark" not in optimized_memory_standard
    assert "HBM3e" not in baseline_memory_standard

    assert "with torch.inference_mode():" in baseline_pipeline_bench
    assert "with torch.no_grad():" not in baseline_pipeline_bench


def test_ch04_torchrun_wrappers_keep_entrypoints_and_side_effect_free_specs() -> None:
    self_target_wrappers = [
        "ch04/ddp_no_overlap.py",
        "ch04/ddp_overlap.py",
        "ch04/baseline_nvshmem_training_example_multigpu.py",
        "ch04/optimized_nvshmem_training_example_multigpu.py",
        "ch04/baseline_nvshmem_training_patterns_multigpu.py",
        "ch04/optimized_nvshmem_training_patterns_multigpu.py",
        "ch04/baseline_nvshmem_pipeline_parallel_multigpu.py",
        "ch04/optimized_nvshmem_pipeline_parallel_multigpu.py",
        "ch04/baseline_nvshmem_vs_nccl_benchmark_multigpu.py",
        "ch04/optimized_nvshmem_vs_nccl_benchmark_multigpu.py",
    ]
    side_effect_free_specs = self_target_wrappers + [
        "ch04/baseline_symmetric_memory_multigpu.py",
        "ch04/optimized_symmetric_memory_multigpu.py",
    ]

    for rel_path in self_target_wrappers:
        text = _read(rel_path)
        assert 'if __name__ == "__main__":' in text
        assert "run_main_with_skip_status(main)" in text

    for rel_path in side_effect_free_specs:
        text = _read(rel_path)
        spec_section = text.split("def get_torchrun_spec", 1)[1].split("def get_custom_metrics", 1)[0]
        assert "_prepare_verification_payload" not in spec_section

    for rel_path in (
        "ch04/baseline_nvshmem_pipeline_parallel_multigpu.py",
        "ch04/optimized_nvshmem_pipeline_parallel_multigpu.py",
    ):
        spec_section = _read(rel_path).split("def get_torchrun_spec", 1)[1].split("def get_custom_metrics", 1)[0]
        assert "config_arg_map" not in spec_section


def test_ch13_pair_remediations_keep_canonical_and_informational_targets_split() -> None:
    canonical_quant = _read("ch13/optimized_torchao_quantization.py")
    baseline_quant = _read("ch13/baseline_torchao_quantization.py")
    compiled_quant = _read("ch13/optimized_torchao_quantization_compiled.py")
    canonical_kv = _read("ch13/optimized_kv_cache_naive.py")
    flash_kv = _read("ch13/optimized_kv_cache_naive_flash_blockwise.py")
    memory_baseline = _read("ch13/baseline_memory_profiling.py")
    memory_optimized = _read("ch13/optimized_memory_profiling.py")

    assert 'configure_tf32(' in baseline_quant
    assert 'matmul_precision="highest"' in baseline_quant
    assert "restore_tf32(self._tf32_state)" in baseline_quant
    assert "torch.compile(self.model" not in canonical_quant
    assert 'configure_tf32(' in canonical_quant
    assert 'matmul_precision="highest"' in canonical_quant
    assert "restore_tf32(self._tf32_state)" in canonical_quant
    assert "torch.compile(self.model" in compiled_quant
    assert "torchao_quantization_compiled" in INFORMATIONAL_BENCHMARKS["ch13"]
    assert "precisionfp8" in INFORMATIONAL_BENCHMARKS["ch13"]
    assert "precisionfp8_rowwise" in INFORMATIONAL_BENCHMARKS["ch13"]
    assert "precisionfp8_rowwise_gw_hp" in INFORMATIONAL_BENCHMARKS["ch13"]
    assert '"tf32": False' in baseline_quant
    assert '"tf32": False' in canonical_quant

    assert "self._request_token_groups = [" in canonical_kv
    assert "for request_id, seq_len, token_views in self._request_token_groups:" in canonical_kv
    assert "self.output = hidden" in canonical_kv
    assert "self.output = hidden.detach()" not in canonical_kv
    assert "range(0, seq_len, self.block_size)" not in canonical_kv
    assert 'return "memory"' in canonical_kv
    assert 'return "memory"' in memory_baseline
    assert 'return "memory"' in memory_optimized
    assert "self._request_block_groups = [" in flash_kv
    assert "for request_id, seq_len, block_views in self._request_block_groups:" in flash_kv
    assert "self.output = hidden[:, -1:, :]" in flash_kv
    assert "hidden[:, -1:, :].detach()" not in flash_kv
    assert "kv_cache_naive_flash_blockwise" in INFORMATIONAL_BENCHMARKS["ch13"]


def test_ch02_grace_coherent_memory_requires_grace_and_never_advertises_fallback() -> None:
    baseline_source = _read("ch02/baseline_grace_coherent_memory.py")
    optimized_source = _read("ch02/optimized_grace_coherent_memory.py")
    readme_text = _read("ch02/README.md")
    refresh_text = _read("core/scripts/refresh_readmes.py")
    rerun_text = _read("scripts/full_virtualized_rerun.py")

    for source in (baseline_source, optimized_source):
        assert "SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support" in source
        assert "using fallback path" not in source

    assert "falls back to a host/device transfer-strategy comparison" not in readme_text
    assert "fallback transfer path" not in readme_text
    assert "fails fast with `SKIPPED:`" in readme_text

    assert "falls back to a host/device transfer-strategy comparison" not in refresh_text
    assert "fallback transfer path" not in refresh_text
    assert "fails fast with `SKIPPED:`" in refresh_text

    assert "requires grace-blackwell coherent memory support" in rerun_text


def test_ch04_no_overlap_and_nvshmem_surfaces_do_not_advertise_single_gpu_fallbacks() -> None:
    ddp_no_overlap = _read("ch04/ddp_no_overlap.py")
    ddp_overlap = _read("ch04/ddp_overlap.py")
    baseline_no_overlap = _read("ch04/baseline_no_overlap.py")
    readme_text = _read("ch04/README.md")
    example_wrapper = _read("ch04/baseline_nvshmem_training_example.py")
    patterns_wrapper = _read("ch04/baseline_nvshmem_training_patterns.py")
    pipeline_wrapper = _read("ch04/baseline_nvshmem_pipeline_parallel.py")
    bandwidth_wrapper = _read("ch04/baseline_bandwidth_benchmark_suite.py")
    symmem_wrapper = _read("ch04/baseline_symmetric_memory.py")
    nvshmem_vs_nccl_wrapper = _read("ch04/baseline_nvshmem_vs_nccl_benchmark.py")

    assert "Single-GPU simulation" not in ddp_no_overlap
    assert "stand-in for" not in ddp_no_overlap
    assert "stand-in for" not in ddp_overlap
    assert "stand-in for" not in baseline_no_overlap
    assert 'if __name__ == "__main__":' in ddp_no_overlap
    assert 'if __name__ == "__main__":' in ddp_overlap
    assert "setup_single_gpu_env(\n            \"ddp_no_overlap\"" in ddp_no_overlap
    assert "setup_single_gpu_env(\n            \"ddp_overlap\"" in ddp_overlap
    ddp_no_overlap_spec = ddp_no_overlap.split("def get_torchrun_spec", 1)[1].split("def get_benchmark", 1)[0]
    ddp_overlap_spec = ddp_overlap.split("def get_torchrun_spec", 1)[1].split("def get_benchmark", 1)[0]
    assert "_prepare_verification_payload()" not in ddp_no_overlap_spec
    assert "_prepare_verification_payload()" not in ddp_overlap_spec
    assert '"iterations": "--iterations"' in ddp_no_overlap_spec
    assert '"warmup": "--warmup"' in ddp_no_overlap_spec
    assert '"iterations": "--iterations"' in ddp_overlap_spec
    assert '"warmup": "--warmup"' in ddp_overlap_spec
    assert "SingleGPUTransferBenchmark" not in example_wrapper
    assert "SingleGPUTransferBenchmark" not in patterns_wrapper
    assert "SingleGPUTransferBenchmark" not in pipeline_wrapper
    assert "SingleGPUTransferBenchmark" not in bandwidth_wrapper
    assert "SingleGPUTransferBenchmark" not in symmem_wrapper
    assert "SingleGPUTransferBenchmark" not in nvshmem_vs_nccl_wrapper
    assert "host-buffer round-trip" not in readme_text
    assert "require `torchrun` plus `>=2` GPUs" in readme_text


def test_ch04_nvshmem_vs_nccl_wrapper_keeps_collective_metadata_aligned_to_mode() -> None:
    baseline_source = _read("ch04/baseline_nvshmem_vs_nccl_benchmark_multigpu.py")
    optimized_source = _read("ch04/optimized_nvshmem_vs_nccl_benchmark_multigpu.py")

    assert 'mode="nccl"' in baseline_source
    assert '"collective_type": "nccl"' in baseline_source
    assert 'mode="nvshmem"' in optimized_source
    assert '"collective_type": "nvshmem"' in optimized_source


def test_ch05_gds_probe_and_ch07_tma_copy_never_advertise_fallback_paths() -> None:
    gds_source = _read("ch05/gds_cufile_minimal.py")
    gds_readme = _read("ch05/README.md")
    refresh_text = _read("core/scripts/refresh_readmes.py")
    tma_cuda = _read("ch07/optimized_tma_copy.cu")
    tma_readme = _read("ch07/README.md")

    assert "standard I/O fallback" not in gds_source
    assert "publish host-mediated fallback numbers" in gds_readme
    assert "publish host-mediated fallback numbers" in refresh_text
    assert "Async-pipeline 2D copy fallback" not in tma_cuda
    assert "legacy async-neighbor demo" not in tma_readme
    assert "strict tensor-map/TMA-capable run only" in tma_readme


def test_ch01_training_loop_targets_keep_combined_and_fusion_only_stories_separate() -> None:
    performance = _read("ch01/optimized_performance.py")
    performance_fusion = _read("ch01/optimized_performance_fusion.py")
    workload_config = _read("ch01/workload_config.py")
    readme_text = _read("ch01/README.md")

    assert "self.model = self.model.half()" in performance
    assert "dtype = torch.float16" in performance
    assert 'with self._nvtx_range("optimized_performance"):' in performance
    assert '"fp16": model_dtype == torch.float16' in performance

    assert "self.model = self.model.half()" not in performance_fusion
    assert "dtype=torch.float32" in performance_fusion
    assert 'with self._nvtx_range("optimized_performance_fusion"):' in performance_fusion
    assert "performance_microbatches: int = 128" in workload_config

    assert "| `performance` | FP16 math + fused microbatches | the combined goodput story |" in readme_text
    assert "| `performance_fusion` | fused microbatches only | what launch amortization buys you without changing math precision |" in readme_text


def test_ch05_and_ch20_noncanonical_pairs_are_marked_informational() -> None:
    assert "ai" in INFORMATIONAL_BENCHMARKS["ch05"]
    assert "cuda_graphs_conditional" in INFORMATIONAL_BENCHMARKS["ch12"]
    assert "pipeline_sequential" in INFORMATIONAL_BENCHMARKS["ch20"]


def test_ch11_stream_ordered_kv_cache_uses_three_streams_without_changing_segments() -> None:
    source = _read("ch11/optimized_stream_ordered_kv_cache.py")

    assert "num_segments=8" in source
    assert "num_streams=3" in source
    assert "same chunked workload and update ordering" in source


def test_ch10_attention_and_ch13_precisionmixed_retuned_workloads_match_between_pairs() -> None:
    ch10_baseline = _read("ch10/baseline_attention.py")
    ch10_optimized = _read("ch10/optimized_attention.py")
    ch13_baseline = _read("ch13/baseline_precisionmixed.py")
    ch13_optimized = _read("ch13/optimized_precisionmixed.py")

    assert "self.seq_len = 1280" in ch10_baseline
    assert "self.seq_len = 1280" in ch10_optimized
    assert "self.hidden_dim = 3072" in ch13_baseline
    assert "self.hidden_dim = 3072" in ch13_optimized
    assert "same workload" in ch10_optimized
    assert "same training shape" in ch13_optimized


def test_portable_rerun_ignores_informational_targets_for_expectation_queueing() -> None:
    assert _is_informational_benchmark("ch05", {"example": "ai"}) is True
    assert _is_informational_benchmark("ch12", {"example": "cuda_graphs_conditional"}) is True
    assert _is_informational_benchmark("ch20", {"example": "pipeline_sequential"}) is True
    assert _is_informational_benchmark("labs_decode_optimization", {"example": "decode_pinned"}) is False
    assert _is_informational_benchmark("labs_fullstack_cluster", {"example": "cluster_gemm_tcgen05"}) is False
    assert _is_informational_benchmark("labs_nvfp4_group_gemm", {"example": "nvfp4_group_gemm"}) is False
    assert _is_informational_benchmark("labs_nvfp4_group_gemm", {"example": "nvfp4_group_gemm_g8_n7168_k2048"}) is False
    assert _is_informational_benchmark("labs_nvfp4_group_gemm", {"example": "nvfp4_group_gemm_g2_n3072_k4096"}) is False
    assert _is_informational_benchmark("labs_nvfp4_group_gemm", {"example": "nvfp4_group_gemm_g2_n4096_k1536"}) is False
    assert _is_informational_benchmark("labs_persistent_decode", {"example": "nvlink_offload"}) is False
    assert _is_informational_benchmark("labs_persistent_decode", {"example": "paged_kv_offload"}) is False
    assert _is_informational_benchmark("ch13", {"example": "kv_cache_naive"}) is False


def test_portable_rerun_classifies_runtime_capability_skips_separately() -> None:
    reason = _expected_unsupported_portable_reason(
        {
            "status": "skipped",
            "error": "SKIPPED: PyTorch build missing batched_reduce_scatter_hook required for optimized ZeRO-2.",
        }
    )
    assert reason == EXPECTED_UNSUPPORTED_RUNTIME_REASON
    rerun_text = _read("scripts/full_virtualized_rerun.py")
    assert "requires torchrun/distributed launch context" in rerun_text
    assert "requires usable cufile/gds support" in rerun_text
    assert "requires usable tensor-map/tma support" in rerun_text
    assert "requires sm100+ blackwell-class hardware" in rerun_text


def test_portable_rerun_reclassifies_pre_sm100_cutlass_fp8_as_expected_unsupported() -> None:
    state = _canonicalize_state(
        {
            "target_records": {
                "ch09:cutlass_gemm_fp8": {
                    "target": "ch09:cutlass_gemm_fp8",
                    "benchmarks": [
                        {
                            "target": "ch09:cutlass_gemm_fp8",
                            "benchmark_status": "skipped",
                            "error": "HARDWARE/SOFTWARE LIMITATION: baseline_cutlass_gemm_fp8 requires SM100+ Blackwell-class hardware.",
                            "queue_reasons": ["skipped", "missing_successful_optimization"],
                        }
                    ],
                    "queued_problem_count": 1,
                    "expected_unsupported_count": 0,
                    "written_expectation_count": 0,
                }
            }
        }
    )
    record = state["target_records"]["ch09:cutlass_gemm_fp8"]
    bench = record["benchmarks"][0]

    assert bench["classification"] == EXPECTED_UNSUPPORTED_RUNTIME_REASON
    assert bench["queue_reasons"] == [EXPECTED_UNSUPPORTED_RUNTIME_REASON]
    assert record["queued_problem_count"] == 0
    assert record["expected_unsupported_count"] == 1


def test_ch09_cutlass_fp8_pair_is_retuned_for_blackwell_sm100() -> None:
    baseline_wrapper = _read("ch09/baseline_cutlass_gemm_fp8.py")
    optimized_wrapper = _read("ch09/optimized_cutlass_gemm_fp8.py")
    baseline_source = _read("ch09/baseline_cutlass_gemm_fp8.cu")
    optimized_source = _read("ch09/optimized_cutlass_gemm_fp8.cu")

    for wrapper in (baseline_wrapper, optimized_wrapper):
        assert "requires SM100+ Blackwell-class hardware" in wrapper
        assert "requires SM90 Hopper hardware" not in wrapper
        assert "major < 10" in wrapper

    assert 'self._selected_backend = "cutlass_sm100_1sm"' in baseline_wrapper
    assert 'self._selected_backend = "cutlass_sm100_2sm"' in optimized_wrapper

    for source in (baseline_source, optimized_source):
        assert "CUTLASS_ARCH_MMA_SM100_SUPPORTED" in source
        assert "cutlass::arch::Sm100" in source
        assert "CUTLASS_ARCH_MMA_SM90_SUPPORTED" not in source
        assert "cutlass::arch::Sm90" not in source

    assert "KernelTmaWarpSpecialized1SmSm100" in baseline_source
    assert "Shape<_128, _128, _64>" in baseline_source
    assert "KernelTmaWarpSpecialized2SmSm100" in optimized_source
    assert "Shape<_256, _128, _64>" in optimized_source


def test_portable_rerun_reclassifies_multi_gpu_skip_records_on_load() -> None:
    state = _canonicalize_state(
        {
            "target_records": {
                "ch04:no_overlap": {
                    "target": "ch04:no_overlap",
                    "benchmarks": [
                        {
                            "target": "ch04:no_overlap",
                            "benchmark_status": "skipped",
                            "error": "HARDWARE/SOFTWARE LIMITATION: Distributed benchmark requires multiple GPUs (insufficient GPUs available)",
                            "queue_reasons": ["missing_expectation"],
                        }
                    ],
                    "queued_problem_count": 1,
                    "expected_unsupported_count": 0,
                    "written_expectation_count": 0,
                }
            }
        }
    )
    record = state["target_records"]["ch04:no_overlap"]
    bench = record["benchmarks"][0]

    assert bench["classification"] == EXPECTED_UNSUPPORTED_PORTABLE_REASON
    assert bench["queue_reasons"] == [EXPECTED_UNSUPPORTED_PORTABLE_REASON]
    assert record["queued_problem_count"] == 0
    assert record["expected_unsupported_count"] == 1


def test_portable_rerun_reclassifies_nested_optimization_capability_skips(tmp_path: Path) -> None:
    results_json = tmp_path / "results.json"
    results_json.write_text(
        '{"results":[{"chapter":"ch09","benchmarks":[{"example":"cublaslt_gemm_fp4","status":"succeeded","optimizations":[{"file":"optimized_cublaslt_gemm_fp4.py","status":"skipped","error":"SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain. Block-scaled VEC16_UE4M3 requires a native cuBLASLt heuristic for this exact benchmark."}]}]}]}',
        encoding="utf-8",
    )
    state = _canonicalize_state(
        {
            "target_records": {
                "ch09:cublaslt_gemm_fp4": {
                    "target": "ch09:cublaslt_gemm_fp4",
                    "benchmarks": [
                        {
                            "target": "ch09:cublaslt_gemm_fp4",
                            "example": "cublaslt_gemm_fp4",
                            "benchmark_status": "succeeded",
                            "queue_reasons": ["missing_successful_optimization"],
                            "results_json": str(results_json),
                        }
                    ],
                    "queued_problem_count": 1,
                    "expected_unsupported_count": 0,
                    "written_expectation_count": 0,
                }
            }
        }
    )
    record = state["target_records"]["ch09:cublaslt_gemm_fp4"]
    bench = record["benchmarks"][0]

    assert bench["classification"] == EXPECTED_UNSUPPORTED_RUNTIME_REASON
    assert bench["queue_reasons"] == [EXPECTED_UNSUPPORTED_RUNTIME_REASON]
    assert "algorithm unavailable on this driver/toolchain" in bench["error"]
    assert record["queued_problem_count"] == 0
    assert record["expected_unsupported_count"] == 1


def test_portable_rerun_backfills_cumulative_expectation_writes_from_worker_log(tmp_path: Path) -> None:
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "\n".join(
            [
                "[2026-03-21T07:19:07+00:00] finished target ch01:nvfp4_mlp: rc=0 written_expectations=1 queued_problems=0",
                "[2026-03-21T07:19:48+00:00] finished target ch01:performance: rc=0 written_expectations=0 queued_problems=1",
                "[2026-03-21T07:20:33+00:00] finished target ch01:performance_fp16: rc=0 written_expectations=1 queued_problems=0",
            ]
        ),
        encoding="utf-8",
    )
    state = {"written_expectation_total": 0}

    _backfill_written_expectation_total(worker_log, state)

    assert state["written_expectation_total"] == 2


def test_portable_rerun_persist_state_keeps_written_totals_without_worker_log(tmp_path: Path) -> None:
    paths = _queue_paths(tmp_path)
    state = {
        "target_records": {
            "ch01:performance": {
                "target": "ch01:performance",
                "benchmarks": [],
                "written_expectation_count": 2,
                "queued_problem_count": 0,
                "expected_unsupported_count": 0,
            }
        },
        "written_expectation_total": 0,
        "queued_problem_total": 0,
        "expected_unsupported_total": 0,
    }

    _persist_state(paths, state)

    persisted = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert persisted["written_expectation_total"] == 2
    assert persisted["target_records"]["ch01:performance"]["written_expectation_count"] == 2


def test_portable_rerun_uses_typed_expectation_keys_for_cuda_examples() -> None:
    assert _expectation_example_key({"example": "cuda_graphs_conditional_enhanced", "type": "cuda"}) == (
        "cuda_graphs_conditional_enhanced_cuda"
    )
    assert _expectation_example_key({"example": "regional_triton", "type": "python"}) == "regional_triton"


def test_canonical_queue_batch_helper_tracks_planned_chapter_targets() -> None:
    helper = _read("scripts/canonical_queue_batches.py")
    registered_targets = _registered_targets()
    queued_targets = [
        target
        for group in CHAPTER_EXPECTATION_BATCH.values()
        for target in group
    ] + list(CHAPTER_DRIFT_TRIAGE) + [
        target
        for group in CAPABILITY_VALIDATION_BATCH.values()
        for target in group
    ] + [
        target
        for group in LAB_FAMILY_BATCHES.values()
        for target in group
    ]

    assert '"ch07:tma_bulk_tensor_2d"' in helper
    assert '"ch10:dsmem_reduction"' in helper
    assert '"ch13:regional_compile"' in helper
    assert '"ch09:cutlass_gemm_fp8"' in helper
    assert '"labs/train_distributed:ddp"' in helper
    assert sorted(set(queued_targets) - registered_targets) == []


def test_ch14_optimized_regional_triton_warms_all_sequence_buckets_in_setup() -> None:
    baseline_text = _read("ch14/baseline_regional_triton.py")
    optimized_text = _read("ch14/optimized_regional_triton.py")
    setup_section = _setup_section("ch14/optimized_regional_triton.py")

    assert "self.hidden = 1536" in baseline_text
    assert "self.hidden = 1536" in optimized_text
    assert "self.num_heads = 12" in baseline_text
    assert "self.num_heads = 12" in optimized_text
    assert "self.mlp_hidden = 12288" in baseline_text
    assert "self.mlp_hidden = 12288" in optimized_text
    assert "for _ in range(3):" in setup_section
    assert "for seq in self.sequence_schedule:" in setup_section
    assert "_ = self._compiled_model(self.inputs[seq])" in setup_section
    assert "timed path measures steady" in setup_section


def test_ch13_regional_compile_uses_heavier_bf16_block_shape_in_both_variants() -> None:
    baseline_text = _read("ch13/baseline_regional_compile.py")
    optimized_text = _read("ch13/optimized_regional_compile.py")

    for source in (baseline_text, optimized_text):
        assert "self.hidden = 2048" in source
        assert "self.num_heads = 16" in source
        assert "self.mlp_hidden = 16384" in source
        assert "self.batch_size = 16" in source


def test_ch14_triton_persistent_uses_deeper_batched_gemm_workload() -> None:
    baseline_text = _read("ch14/baseline_triton_persistent.py")
    optimized_text = _read("ch14/optimized_triton_persistent.py")

    assert "self.batch_size = 64" in baseline_text
    assert "self.batch_size = 64" in optimized_text


def test_ch14_flex_attention_sparse_uses_longer_and_sparser_window() -> None:
    baseline_text = _read("ch14/baseline_flex_attention_sparse.py")
    optimized_text = _read("ch14/optimized_flex_attention_sparse.py")

    for source in (baseline_text, optimized_text):
        assert "self.seq_len = 4096" in source
        assert "self.window_size = 128" in source
    assert "scores.masked_fill_(~allowed_mask, float(\"-inf\"))" in baseline_text
    assert "scores = scores.masked_fill(~allowed_mask, float(\"-inf\"))" not in baseline_text


def test_ch17_memory_uses_larger_replayed_transfer_workload() -> None:
    baseline_text = _read("ch17/baseline_memory.py")
    optimized_text = _read("ch17/optimized_memory.py")

    for source in (baseline_text, optimized_text):
        assert "BATCH_SIZE = 1024" in source
        assert "REPETITIONS = 10" in source


def test_ch13_regional_compile_retunes_shared_mlp_heavy_shape() -> None:
    baseline_text = _read("ch13/baseline_regional_compile.py")
    optimized_text = _read("ch13/optimized_regional_compile.py")

    for source in (baseline_text, optimized_text):
        assert "self.hidden = 2048" in source
        assert "self.num_heads = 16" in source
        assert "self.mlp_hidden = 16384" in source
        assert "self.batch_size = 16" in source
        assert "self.sequence_schedule: List[int] = [256, 512, 1024, 1536]" in source


def test_ch13_dataloader_default_uses_heavier_shared_preprocessing_workload() -> None:
    baseline_text = _read("ch13/baseline_dataloader_default.py")
    optimized_text = _read("ch13/optimized_dataloader_default.py")

    for source in (baseline_text, optimized_text):
        assert "self.dataset_size = 4000" in source
        assert "self.batch_size = 64" in source
        assert "self.feature_dim = 1024" in source
        assert "self.preprocess_steps = 16" in source


def test_parameterized_graph_verification_capture_uses_fixed_request_slot() -> None:
    source = _read("labs/parameterized_cuda_graphs/parameterized_cuda_graphs_common.py")
    assert "slot_idx = 0" in source
    assert "self._run_verification_slot(slot_idx)" in source
    setup_section = source.split("def setup", maxsplit=1)[1].split(
        "def _build_request_slots",
        maxsplit=1,
    )[0]
    build_slots = source.split("def _build_request_slots", maxsplit=1)[1].split(
        "def _warmup_eager_path",
        maxsplit=1,
    )[0]
    output_slice = source.split("def _current_output_slice", maxsplit=1)[1].split(
        "def _run_verification_slot",
        maxsplit=1,
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def get_config",
        maxsplit=1,
    )[0]
    teardown_section = source.split("def teardown", maxsplit=1)[1].split(
        "class ParameterizedGraphRecaptureBenchmark",
        maxsplit=1,
    )[0]
    assert "self.host_inputs = [torch.empty(0) for _ in range(self.cfg.request_slots)]" in build_slots
    assert "self.host_scales = [torch.empty(0) for _ in range(self.cfg.request_slots)]" in build_slots
    assert "self.host_outputs = [torch.empty(0) for _ in range(self.cfg.request_slots)]" in build_slots
    assert "self.host_inputs[slot_idx] = host_input" in build_slots
    assert "self.host_scales[slot_idx] = host_scale" in build_slots
    assert "self.host_outputs[slot_idx] = host_output" in build_slots
    assert "self._refresh_slot_memcpy_bindings()" in build_slots
    assert ".append(" not in build_slots
    assert "self._verify_input_buffer: Optional[torch.Tensor] = None" in source
    assert "self._verify_scale_buffer: Optional[torch.Tensor] = None" in source
    assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source
    assert "self._verify_input_buffer = torch.empty_like(self.host_inputs[0])" in setup_section
    assert "self._verify_scale_buffer = torch.empty_like(self.host_scales[0])" in setup_section
    assert 'self._verify_output_buffer = torch.empty((2, 16), dtype=torch.float32, device="cpu")' in setup_section
    assert "self._verify_output_buffer.copy_(host_output[:2, :16])" in output_slice
    assert "return self._verify_output_buffer" in output_slice
    assert "self._verify_input_buffer.copy_(self.host_inputs[slot_idx])" in capture_section
    assert "self._verify_scale_buffer.copy_(self.host_scales[slot_idx])" in capture_section
    assert '"x": self._verify_input_buffer' in capture_section
    assert '"scale": self._verify_scale_buffer' in capture_section
    assert "self.host_inputs[slot_idx].clone()" not in capture_section
    assert "self.host_scales[slot_idx].clone()" not in capture_section
    assert ".to(dtype=torch.float32).clone()" not in output_slice
    assert "self._verify_input_buffer = None" in teardown_section
    assert "self._verify_scale_buffer = None" in teardown_section
    assert "self._verify_output_buffer = None" in teardown_section


def test_parameterized_graph_residual_block_writes_directly_to_output_buffer() -> None:
    source = _read("labs/parameterized_cuda_graphs/parameterized_cuda_graphs_common.py")
    program_section = source.split("def _schedule_request_program", maxsplit=1)[1].split(
        "def _refresh_slot_memcpy_bindings",
        maxsplit=1,
    )[0]

    assert "def forward_into(self, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor)" in source
    assert "torch.mul(hidden, scale, out=out)" in source
    assert "out.add_(x)" in source
    assert "model.forward_into(device_input, device_scale, device_output)" in program_section
    assert "device_output.copy_(model(" not in program_section


def test_parameterized_graph_residual_block_forward_into_matches_forward() -> None:
    from labs.parameterized_cuda_graphs.parameterized_cuda_graphs_common import _ResidualScaleBlock

    torch.manual_seed(1234)
    model = _ResidualScaleBlock(hidden_size=8, expansion_factor=2).eval()
    x = torch.randn(3, 8)
    scale = torch.tensor([0.75])
    out = torch.empty_like(x)

    with torch.inference_mode():
        expected = model(x, scale)
        actual = model.forward_into(x, scale, out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_parameterized_graph_replay_uses_cached_memcpy_bindings() -> None:
    source = _read("labs/parameterized_cuda_graphs/parameterized_cuda_graphs_common.py")
    cache_section = source.split("def _refresh_slot_memcpy_bindings", maxsplit=1)[1].split(
        "def _next_slot",
        maxsplit=1,
    )[0]
    bind_section = source.split("def _bind_memcpy_nodes", maxsplit=1)[1].split(
        "def _check_cudart",
        maxsplit=1,
    )[0]
    update_section = source.split("def _update_exec_params_for_slot", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "SlotMemcpyBinding = Tuple[int, int, int, int, int, int, int, int, int]" in source
    assert "self._slot_memcpy_bindings: List[SlotMemcpyBinding] = []" in source
    assert "self._slot_memcpy_bindings = [" in cache_section
    assert "host_input.numel() * host_input.element_size()" in cache_section
    assert "host_scale.numel() * host_scale.element_size()" in cache_section
    assert "host_output.numel() * host_output.element_size()" in cache_section
    assert ") = self._slot_memcpy_bindings[slot_idx]" in bind_section
    assert ") = self._slot_memcpy_bindings[slot_idx]" in update_section
    assert "slot_input = self.host_inputs[slot_idx]" not in update_section
    assert "slot_scale = self.host_scales[slot_idx]" not in update_section
    assert "slot_output = self.host_outputs[slot_idx]" not in update_section
    assert ".data_ptr()" not in update_section
    assert ".numel()" not in update_section
    assert ".element_size()" not in update_section


def test_ch18_and_fullstack_pairs_keep_semantics_fixed() -> None:
    baseline_flexdecode = _read("ch18/baseline_flexdecoding.py")
    optimized_flexdecode = _read("ch18/optimized_flexdecoding.py")
    moe_common = _read("labs/fullstack_cluster/moe_hybrid_ep_common.py")
    fullstack_readme = _read("labs/fullstack_cluster/README.md")

    assert '"comparison_axis": "full_kv_mask_vs_windowed_kv_slice"' in baseline_flexdecode
    assert '"execution_pattern": "masked_full_cache_decode"' in baseline_flexdecode
    assert "self.model.decode(token, position)" not in optimized_flexdecode
    assert "full KV cache with a sliding-window mask" in baseline_flexdecode
    assert "window_slice_decode" in optimized_flexdecode
    assert "k_slice = self.model.k_cache[:, start:end]" not in baseline_flexdecode
    assert "k_slice = self.model.k_cache[:, start:end]" in optimized_flexdecode
    assert 'route_mode="uniform"' in moe_common
    assert 'route_mode="topology_aware" if optimized else "uniform"' not in moe_common
    assert "cluster_gemm_tcgen05" not in INFORMATIONAL_BENCHMARKS.get("fullstack_cluster", set())
    assert "supplementary comparison benchmark with a local contract" in fullstack_readme
    assert "canonical speed claim stays on `cluster_gemm`" in fullstack_readme


def test_ozaki_lab_documents_slide_narrative_and_pins_emulation_strategy() -> None:
    dynamic_text = _read("labs/ozaki_scheme/optimized_ozaki_scheme_dynamic.py")
    fixed_text = _read("labs/ozaki_scheme/optimized_ozaki_scheme_fixed.py")
    readme_text = _read("labs/ozaki_scheme/README.md")

    for source in (dynamic_text, fixed_text):
        assert '"--emulation-strategy", "eager"' in source
        assert '"emulation_strategy": "eager"' in source

    assert "Coverage Against The Ozaki Narrative" in readme_text
    assert "Ozaki-II Context" in readme_text
    assert "Controllable Accuracy" in readme_text
    assert "Adaptive Behavior" in readme_text
    assert "Reproducibility" in readme_text
    assert "Disadvantages" in readme_text
    assert "Papers and Code" in readme_text
    assert "python labs/ozaki_scheme/narrative_checks.py --section all" in readme_text
    assert "CUBLAS_EMULATE_DOUBLE_PRECISION=1" in readme_text
    assert "CUBLAS_EMULATION_STRATEGY=performant" in readme_text


def test_ch17_memory_pair_keeps_discrete_input_distribution() -> None:
    baseline_memory = _read("ch17/baseline_memory.py")
    optimized_memory = _read("ch17/optimized_memory.py")

    assert "torch.randint(" in baseline_memory
    assert "256," in baseline_memory
    assert "dtype=torch.uint8" in baseline_memory
    assert "random_(0, 256)" in optimized_memory
    assert ".floor_()" not in optimized_memory
    assert "discrete 0..255 population" in optimized_memory


def test_ch10_flashattention3_pair_keeps_shared_warmup_and_unfused_qkv_structure() -> None:
    baseline_source = _read("ch10/baseline_flashattention3_pipeline.py")
    optimized_source = _read("ch10/optimized_flashattention3_pipeline.py")

    assert "for _ in range(3):" in baseline_source
    assert "for _ in range(3):" in optimized_source
    for source in (baseline_source, optimized_source):
        assert "self.q_proj = nn.Linear(" in source
        assert "self.k_proj = nn.Linear(" in source
        assert "self.v_proj = nn.Linear(" in source
        assert "qkv_proj" not in source
    assert "self._q_buffer: Optional[torch.Tensor] = None" in optimized_source
    assert "self._k_buffer: Optional[torch.Tensor] = None" in optimized_source
    assert "self._v_buffer: Optional[torch.Tensor] = None" in optimized_source
    assert "self._output_buffer: Optional[torch.Tensor] = None" in optimized_source
    assert "self._q_forward_view: Optional[torch.Tensor] = None" in optimized_source
    assert "self._k_forward_view: Optional[torch.Tensor] = None" in optimized_source
    assert "self._v_forward_view: Optional[torch.Tensor] = None" in optimized_source
    assert "self._output_forward_view: Optional[torch.Tensor] = None" in optimized_source
    assert "self._q_proj_weight_t: Optional[torch.Tensor] = None" in optimized_source
    assert "self._k_proj_weight_t: Optional[torch.Tensor] = None" in optimized_source
    assert "self._v_proj_weight_t: Optional[torch.Tensor] = None" in optimized_source
    assert "self._out_proj_weight_t: Optional[torch.Tensor] = None" in optimized_source
    assert "def cache_weight_views(self) -> None:" in optimized_source
    assert "self.model.cache_weight_views()" in optimized_source
    assert "def _projection_workspace(" in optimized_source
    assert "def prepare_projection_buffers(self, x: torch.Tensor) -> None:" in optimized_source
    assert "def forward_prepared(" in optimized_source
    assert 'raise RuntimeError("forward_prepared() requires prepare_projection_buffers()")' in optimized_source
    assert "self.model.prepare_projection_buffers(self.input)" in optimized_source
    assert "self.model.forward_prepared(self.input, is_causal=self.use_causal)" in optimized_source
    assert "or buffer.numel() < numel" in optimized_source
    assert "return buffer[:numel].view(shape)" in optimized_source
    assert "self._q_buffer.shape != q_shape" not in optimized_source
    assert "self._output_buffer.shape != output_shape" not in optimized_source
    optimized_forward = optimized_source.split("def forward", maxsplit=1)[1].split(
        "class OptimizedFlashAttention3Benchmark",
        maxsplit=1,
    )[0]
    assert "enable_gqa=enable_gqa" in optimized_forward
    assert "repeat_interleave(n_rep" not in optimized_forward
    assert "q_proj = torch.matmul(x, self._q_proj_weight_t, out=q_buffer)" in optimized_forward
    assert "k_proj = torch.matmul(x, self._k_proj_weight_t, out=k_buffer)" in optimized_forward
    assert "v_proj = torch.matmul(x, self._v_proj_weight_t, out=v_buffer)" in optimized_forward
    assert "return torch.matmul(attn_output, self._out_proj_weight_t, out=output_buffer)" in optimized_forward
    assert "self.q_proj.weight.t()" not in optimized_forward
    assert "self.k_proj.weight.t()" not in optimized_forward
    assert "self.v_proj.weight.t()" not in optimized_forward
    assert "self.out_proj.weight.t()" not in optimized_forward


def test_ch10_flashattention3_projection_buffers_reuse_capacity() -> None:
    from ch10.optimized_flashattention3_pipeline import FA3PipelinedAttention

    module = FA3PipelinedAttention(
        hidden_dim=8,
        num_heads=2,
        num_kv_heads=1,
        use_fp8=False,
    ).eval()
    large_input = torch.empty(4, 5, 8)
    small_input = torch.empty(2, 3, 8)
    grown_input = torch.empty(5, 5, 8)

    large_q, large_k, large_v, large_out = module._ensure_projection_buffers(
        large_input,
        4,
        5,
    )
    ptrs = (
        module._q_buffer.data_ptr(),
        module._k_buffer.data_ptr(),
        module._v_buffer.data_ptr(),
        module._output_buffer.data_ptr(),
    )
    small_q, small_k, small_v, small_out = module._ensure_projection_buffers(
        small_input,
        2,
        3,
    )
    module.prepare_projection_buffers(large_input)
    grown_q, grown_k, grown_v, grown_out = module._ensure_projection_buffers(
        grown_input,
        5,
        5,
    )

    assert large_q.shape == (4, 5, 8)
    assert large_k.shape == (4, 5, 4)
    assert large_v.shape == (4, 5, 4)
    assert large_out.shape == (4, 5, 8)
    assert small_q.shape == (2, 3, 8)
    assert small_k.shape == (2, 3, 4)
    assert small_v.shape == (2, 3, 4)
    assert small_out.shape == (2, 3, 8)
    assert small_q.data_ptr() == ptrs[0]
    assert small_k.data_ptr() == ptrs[1]
    assert small_v.data_ptr() == ptrs[2]
    assert small_out.data_ptr() == ptrs[3]
    assert module._q_forward_view is not None
    assert module._k_forward_view is not None
    assert module._v_forward_view is not None
    assert module._output_forward_view is not None
    assert module._q_forward_view.data_ptr() == ptrs[0]
    assert module._k_forward_view.data_ptr() == ptrs[1]
    assert module._v_forward_view.data_ptr() == ptrs[2]
    assert module._output_forward_view.data_ptr() == ptrs[3]
    assert grown_q.shape == (5, 5, 8)
    assert grown_k.shape == (5, 5, 4)
    assert grown_v.shape == (5, 5, 4)
    assert grown_out.shape == (5, 5, 8)
    assert module._q_buffer.numel() == 5 * 5 * 8
    assert module._k_buffer.numel() == 5 * 5 * 4
    assert module._v_buffer.numel() == 5 * 5 * 4
    assert module._output_buffer.numel() == 5 * 5 * 8


def test_persistent_decode_keeps_canonical_iteration_parity_and_marks_cuda_variant_informational() -> None:
    baseline_source = _read("labs/persistent_decode/baseline_persistent_decode.py")
    triton_source = _read("labs/persistent_decode/optimized_persistent_decode_triton.py")
    cuda_source = _read("labs/persistent_decode/optimized_persistent_decode_cuda.py")

    assert "iterations=12" in baseline_source
    assert "iterations=12" in triton_source
    assert "warmup=5" in baseline_source
    assert "warmup=5" in triton_source
    assert "iterations=5" in cuda_source
    assert "use_subprocess=True" in cuda_source
    assert "persistent_decode_cuda" in INFORMATIONAL_BENCHMARKS.get("persistent_decode", set())
    assert "nvlink_offload" not in INFORMATIONAL_BENCHMARKS.get("persistent_decode", set())
    assert "paged_kv_offload" not in INFORMATIONAL_BENCHMARKS.get("persistent_decode", set())


def test_decode_optimization_keeps_decode_streams_canonical_and_marks_decode_pinned_local_contract() -> None:
    baseline_source = _read("labs/decode_optimization/baseline_decode.py")
    pinned_baseline = _read("labs/decode_optimization/baseline_decode_pinned.py")
    pinned_source = _read("labs/decode_optimization/optimized_decode_pinned.py")
    streams_baseline = _read("labs/decode_optimization/baseline_decode_streams.py")
    streams_optimized = _read("labs/decode_optimization/optimized_decode_streams.py")
    readme_text = _read("labs/decode_optimization/README.md")

    assert "decode_pinned" not in INFORMATIONAL_BENCHMARKS.get("decode_optimization", set())
    assert "host_payload_mb=512" not in baseline_source
    assert "host_payload_mb=512" in pinned_baseline
    assert "use_pinned_host=False" in pinned_baseline
    assert "host_payload_mb=512" in pinned_source
    assert "use_pinned_host=True" in pinned_source
    assert "host_payload_mb=512" in streams_baseline
    assert "host_payload_mb=512" in streams_optimized
    assert "decode_pinned" in readme_text
    assert "supplementary local-contract speed benchmark" in readme_text
    assert "decode_streams" in readme_text
    assert "large host payload" in readme_text
    assert "non-headline benchmarks" in readme_text


def test_ch20_bf16_mlp_no_longer_claims_fused_ops() -> None:
    source = _read("ch20/optimized_bf16_mlp.py")

    assert "does not implement a fused MLP kernel today" in source
    assert '"ch20.uses_fused_ops": 0.0' in source


def test_ch20_integrated_kv_cache_uses_two_layers_in_both_variants() -> None:
    baseline_text = _read("ch20/baseline_integrated_kv_cache.py")
    optimized_text = _read("ch20/optimized_integrated_kv_cache.py")

    assert "self.num_layers = 2" in baseline_text
    assert "self.num_layers = 2" in optimized_text
