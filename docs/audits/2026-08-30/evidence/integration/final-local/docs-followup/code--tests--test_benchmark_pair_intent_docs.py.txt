from __future__ import annotations

from pathlib import Path

from ch02.optimized_memory_transfer import OptimizedMemoryTransferBenchmark
from ch05.optimized_storage_cpu import OptimizedStorageCpuBenchmark
from core.discovery import discover_benchmarks

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ch02_optimized_memory_transfer_docstring_matches_async_pinned_copy() -> None:
    module_doc = (REPO_ROOT / "ch02" / "optimized_memory_transfer.py").read_text(encoding="utf-8")
    class_doc = OptimizedMemoryTransferBenchmark.__doc__

    assert "Pinned host memory with async H2D transfer" in module_doc
    assert class_doc is not None
    assert "asynchronous non-blocking copy" in class_doc
    assert "NVLink-C2C" not in module_doc
    assert "NVLink-C2C" not in class_doc


def test_ch04_no_overlap_docs_require_real_distributed_launch() -> None:
    ddp_no_overlap_text = (REPO_ROOT / "ch04" / "ddp_no_overlap.py").read_text(encoding="utf-8")
    wrapper_text = (REPO_ROOT / "ch04" / "baseline_no_overlap.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch04" / "README.md").read_text(encoding="utf-8")

    assert "Single-GPU simulation" not in ddp_no_overlap_text
    assert "stand-in for" not in ddp_no_overlap_text
    assert "communication overlap" in ddp_no_overlap_text
    assert "single-GPU simulation" not in wrapper_text
    assert "stand-in for" not in wrapper_text
    assert "require `torchrun` plus `>=2` GPUs" in readme_text


def test_ch05_optimized_storage_cpu_names_match_cpu_staged_path() -> None:
    source = (REPO_ROOT / "ch05" / "optimized_storage_cpu.py").read_text(encoding="utf-8")

    assert OptimizedStorageCpuBenchmark.__name__ == "OptimizedStorageCpuBenchmark"
    assert 'storage_cpu_optimized' in source
    assert "OptimizedStorageGdsBenchmark" not in source
    assert '"storage_gds"' not in source


def test_ch14_demo_files_stay_out_of_benchmark_discovery() -> None:
    ch14_dir = REPO_ROOT / "ch14"
    pairs = discover_benchmarks(ch14_dir)
    names = {example_name for _, _, example_name in pairs}

    assert (ch14_dir / "triton_persistent_demo.py").exists()
    assert (ch14_dir / "sliding_window_demo.py").exists()
    assert "triton_persistent" in names
    assert "sliding_window" in names
    assert "triton_persistent_demo" not in names
    assert "sliding_window_demo" not in names


def test_ch06_attention_ilp_docs_call_out_attention_score_microbenchmark() -> None:
    baseline_text = (REPO_ROOT / "ch06" / "baseline_attention_ilp.py").read_text(encoding="utf-8")
    optimized_text = (REPO_ROOT / "ch06" / "optimized_attention_ilp.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch06" / "README.md").read_text(encoding="utf-8")

    assert "attention-score" in baseline_text
    assert "attention-score" in optimized_text
    assert "multi-stream" not in optimized_text
    assert "scaled_dot_product_attention" not in optimized_text
    assert "attention-score preprocessing microbenchmark" in readme_text


def test_ch08_readme_marks_bridge_comparisons_vs_native_exemplars() -> None:
    readme_text = (REPO_ROOT / "ch08" / "README.md").read_text(encoding="utf-8")

    assert "chapter-native exemplars" in readme_text
    assert "bridge comparison" in readme_text
    assert "`threshold`, `loop_unrolling`, and `ai_optimization` are the chapter-native exemplars" in readme_text
    assert "`thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp` remain real baseline/optimized bridge comparisons" in readme_text
    assert "`tcgen05_custom_vs_cublas`" in readme_text
    assert "supplementary comparison benchmark with a local contract" in readme_text
    assert "historical baseline/optimized filenames" not in readme_text


def test_ch04_readme_calls_single_gpu_staging_pair_pcie_not_nvlink() -> None:
    readme_text = (REPO_ROOT / "ch04" / "README.md").read_text(encoding="utf-8")

    assert "baseline_pcie_staging.py" in readme_text
    assert "optimized_pcie_staging.py" in readme_text
    assert "PCIe host-staging comparison pair" in readme_text


def test_ch14_readme_uses_renamed_compile_and_library_targets() -> None:
    readme_text = (REPO_ROOT / "ch14" / "README.md").read_text(encoding="utf-8")

    assert "model_compile_reduced_precision" in readme_text
    assert "baseline_model_compile_reduced_precision.py" in readme_text
    assert "baseline_cublas_vs_cutlass.py" in readme_text
    assert "informational comparison surface" in readme_text
    assert "supplementary comparison pair" in readme_text
    assert "model_compile_bf16" not in readme_text
    assert "baseline_cutlass.py" not in readme_text


def test_ch08_readme_marks_tcgen05_custom_vs_cublas_informational_and_tiling_fast_path() -> None:
    readme_text = (REPO_ROOT / "ch08" / "README.md").read_text(encoding="utf-8")

    assert "tcgen05_custom_vs_cublas" in readme_text
    assert "supplementary comparison benchmark with a local contract" in readme_text
    assert "matmul_tiled_fast" in readme_text


def test_ch15_readme_calls_single_gpu_pair_a_comparison_handoff() -> None:
    readme_text = (REPO_ROOT / "ch15" / "README.md").read_text(encoding="utf-8")

    assert "baseline_single_gpu_kv_handoff.py" in readme_text
    assert "supplementary single-GPU KV-handoff comparison pair" in readme_text
    assert "baseline_disaggregated_inference.py" not in readme_text


def test_nvfp4_group_gemm_readme_uses_shape_names_and_frontdoor() -> None:
    readme_text = (REPO_ROOT / "labs" / "nvfp4_group_gemm" / "README.md").read_text(encoding="utf-8")
    names = {example_name for _, _, example_name in discover_benchmarks(REPO_ROOT / "labs" / "nvfp4_group_gemm")}

    assert "nvfp4_group_gemm" in readme_text
    assert "nvfp4_group_gemm_g8_n4096_k7168" in readme_text
    assert "nvfp4_group_gemm_g8_n7168_k2048" in readme_text
    assert "nvfp4_group_gemm_g2_n3072_k4096" in readme_text
    assert "nvfp4_group_gemm_g2_n4096_k1536" in readme_text
    assert "former competition `caseN` numbering is retired" in readme_text
    assert "supplementary comparison benchmark" in readme_text
    assert "canonical local-contract speed benchmark" in readme_text
    assert "nvfp4_group_gemm_case0" not in readme_text
    assert "nvfp4_group_gemm_case1" not in readme_text
    assert "nvfp4_group_gemm_case2" not in readme_text
    assert "nvfp4_group_gemm_case3" not in readme_text
    assert "nvfp4_group_gemm" in names
    assert "nvfp4_group_gemm_g8_n4096_k7168" in names
    assert "nvfp4_group_gemm_g8_n7168_k2048" in names
    assert "nvfp4_group_gemm_g2_n3072_k4096" in names
    assert "nvfp4_group_gemm_g2_n4096_k1536" in names
    assert "nvfp4_group_gemm_case0" not in names
    assert "nvfp4_group_gemm_case1" not in names
    assert "nvfp4_group_gemm_case2" not in names
    assert "nvfp4_group_gemm_case3" not in names


def test_ch17_inference_full_docs_mark_comparison_pair_not_disagg_exemplar() -> None:
    baseline_text = (REPO_ROOT / "ch17" / "baseline_inference_full.py").read_text(encoding="utf-8")
    optimized_text = (REPO_ROOT / "ch17" / "optimized_inference_full.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch17" / "README.md").read_text(encoding="utf-8")

    assert "comparison pair for model-side work reduction" in baseline_text
    assert "comparison pair for model-side work reduction" in optimized_text
    assert "disaggregated prefill/decode" in baseline_text
    assert "disaggregated prefill/decode" in optimized_text
    assert "It is not the" in baseline_text
    assert "It is not the" in optimized_text
    assert "chapter's native" in baseline_text
    assert "chapter's native" in optimized_text
    assert "inference_full` remains a comparison pair" in readme_text
    assert "chapter-native exemplars" in readme_text


def test_ch10_readme_stays_on_book_native_targets() -> None:
    readme_text = (REPO_ROOT / "ch10" / "README.md").read_text(encoding="utf-8")
    pairs = discover_benchmarks(REPO_ROOT / "ch10")
    names = {example_name for _, _, example_name in pairs}

    assert not (REPO_ROOT / "ch10" / "baseline_matmul.py").exists()
    assert not (REPO_ROOT / "ch10" / "optimized_matmul.py").exists()
    assert "baseline_matmul.py" not in readme_text
    assert "optimized_matmul.py" not in readme_text
    assert "double_buffered_pipeline" in readme_text
    assert "warp_specialized_pipeline" in readme_text
    assert "cluster_group" in readme_text
    assert "baseline_matmul_tcgen05_vs_cublas.py" in readme_text
    assert "baseline_tmem_tcgen05.py" not in readme_text
    assert "matmul" not in names


def test_ch11_legacy_stream_docs_call_out_actual_overlap_workload() -> None:
    baseline_adaptive = (REPO_ROOT / "ch11" / "baseline_adaptive_streams.py").read_text(encoding="utf-8")
    optimized_adaptive = (REPO_ROOT / "ch11" / "optimized_adaptive_streams.py").read_text(encoding="utf-8")
    baseline_gemm = (REPO_ROOT / "ch11" / "baseline_gemm_streams.py").read_text(encoding="utf-8")
    optimized_gemm = (REPO_ROOT / "ch11" / "optimized_gemm_streams.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "ch11" / "README.md").read_text(encoding="utf-8")

    assert "fixed round-robin" in optimized_adaptive
    assert "serialized copy/compute overlap work" in baseline_adaptive
    assert "copy+elementwise overlap work" in baseline_gemm
    assert "copy+elementwise stream work" in optimized_gemm
    assert "legacy target names" in readme_text
    assert "copy+elementwise overlap workload" in readme_text
    assert "runtime-adaptive scheduling" in readme_text


def test_ch13_readme_marks_canonical_vs_informational_variants() -> None:
    readme_text = (REPO_ROOT / "ch13" / "README.md").read_text(encoding="utf-8")

    assert "fairness-refreshed" in readme_text
    assert "token-by-token decode with naive concat cache versus paged cache allocation" in readme_text
    assert "kv_cache_naive_flash_blockwise" in readme_text
    assert "torchao_quantization_compiled" in readme_text
    assert "precisionfp8_rowwise" in readme_text
    assert "precisionfp8_rowwise_gw_hp" in readme_text
    assert "quantization-only canonical pair" in readme_text
    assert "memory-goal benchmark" in readme_text


def test_persistent_decode_readme_marks_direct_offload_comparisons_vs_prefetch_overlap() -> None:
    readme_text = (REPO_ROOT / "labs" / "persistent_decode" / "README.md").read_text(encoding="utf-8")

    assert "nvlink_offload" in readme_text
    assert "paged_kv_offload" in readme_text
    assert "transport-comparison benchmark" in readme_text
    assert "paged_kv_offload` as a real speed benchmark" in readme_text
    assert "paged_kv_offload_prefetch" in readme_text
    assert "canonical KV-offload overlap claim" in readme_text


def test_decode_optimization_readme_marks_decode_pinned_as_comparison_surface() -> None:
    readme_text = (REPO_ROOT / "labs" / "decode_optimization" / "README.md").read_text(encoding="utf-8")

    assert "decode_pinned" in readme_text
    assert "supplementary local-contract speed benchmark" in readme_text
    assert "decode_streams" in readme_text
    assert "large host payload" in readme_text


def test_training_hotpath_readme_marks_padding_aware_transformer_memory_goal() -> None:
    readme_text = (REPO_ROOT / "labs" / "training_hotpath" / "README.md").read_text(encoding="utf-8")

    assert "padding_aware_transformer" in readme_text
    assert "memory-goal benchmark" in readme_text
    assert "peak-memory reduction" in readme_text
    assert "not by raw speedup" in readme_text


def test_cache_aware_disagg_readme_marks_single_gpu_locality_comparison_contract() -> None:
    readme_text = (REPO_ROOT / "labs" / "cache_aware_disagg_inference" / "README.md").read_text(encoding="utf-8")

    assert "cache_aware_disagg" in readme_text
    assert "locality-comparison benchmark with a local comparison contract" in readme_text
    assert "cache hit rate, KV transfer volume, and worker affinity" in readme_text


def test_train_distributed_readme_marks_single_gpu_fsdp2_as_comparison_surface() -> None:
    readme_text = (REPO_ROOT / "labs" / "train_distributed" / "README.md").read_text(encoding="utf-8")

    assert "single-GPU `fsdp2` on `b200`" in readme_text
    assert "supplementary comparison surface with a local comparison contract" in readme_text
    assert "multi-GPU `2x_b200` contract" in readme_text


def test_occupancy_tuning_readme_marks_low_warp_schedule_informational() -> None:
    readme_text = (REPO_ROOT / "labs" / "occupancy_tuning" / "README.md").read_text(encoding="utf-8")

    assert "proton_matmul_bm64_bn64_bk32_nw2" in readme_text
    assert "supplementary local-contract schedule benchmark" in readme_text
    assert "proton_matmul_bm64_bn256_bk32" in readme_text
    assert "proton_matmul_bm128_bn128_bk32_nw8" in readme_text
    assert "proton_matmul_bm128_bn256_bk64" in readme_text


def test_ch18_flexdecoding_docs_call_out_intentional_work_reduction() -> None:
    readme_text = (REPO_ROOT / "ch18" / "README.md").read_text(encoding="utf-8")

    assert "chapter-native work-reduction story" in readme_text
    assert "full KV cache with a sliding-window mask" in readme_text
    assert "active window before attention" in readme_text
    assert "Re-measure it on your hardware" in readme_text


def test_fullstack_cluster_docs_call_out_uniform_default_and_topology_override() -> None:
    common_text = (REPO_ROOT / "labs" / "fullstack_cluster" / "moe_hybrid_ep_common.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "labs" / "fullstack_cluster" / "README.md").read_text(encoding="utf-8")

    assert 'route_mode="uniform"' in common_text
    assert "Single-GPU `moe_hybrid_ep` benchmark runs measure `HybridEPTrainer.run_step()` in-process" in readme_text


def test_block_scaling_readme_calls_out_local_floor_below_stale_single_run_peak() -> None:
    readme_text = (REPO_ROOT / "labs" / "block_scaling" / "README.md").read_text(encoding="utf-8")

    assert "roughly `1.76x`" in readme_text
    assert "local gating floor is `1.75x`" in readme_text
    assert "`1.784x` single-run best" in readme_text


def test_top_k_kernel_docs_and_defaults_align_to_large_forward_routing_case() -> None:
    common_text = (REPO_ROOT / "labs" / "top_k_kernel" / "top_k_kernel_common.py").read_text(encoding="utf-8")
    readme_text = (REPO_ROOT / "labs" / "top_k_kernel" / "README.md").read_text(encoding="utf-8")

    assert 'mode: str = "forward"' in common_text
    assert "q_len: int = 32768" in common_text
    assert "compressed_k_len: int = 32768" in common_text
    assert "head_dim: int = 128" in common_text
    assert "top_k: int = 16" in common_text
    assert "selection_block_size: int = 64" in common_text
    assert "compress_stride: int = 1" in common_text
    assert "The default harnessed workload is the large forward-routing case" in readme_text
    assert "Keep `fwd_bwd` as an explicit override" in readme_text
