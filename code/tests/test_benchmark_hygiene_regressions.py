from __future__ import annotations

import importlib
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import tempfile
import time

import torch

from ch01.optimized_performance import OptimizedPerformanceBatchBenchmark
from ch01.optimized_performance_fp16 import OptimizedPerformanceFP16Benchmark
from ch02.baseline_cublas import BaselineCublasBenchmark
from ch02.optimized_cublas import OptimizedCublasBenchmark
from core.benchmark.verification import coerce_input_signature
from core.harness.benchmark_harness import BaseBenchmark
from labs.flexattention.baseline_flex_attention import BaselineFlexAttentionBenchmark
from labs.flexattention.optimized_flex_attention import OptimizedFlexAttentionBenchmark
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn256_bk32 import (
    get_benchmark as get_wide_n_benchmark,
)
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn64_bk32_nw2 import (
    get_benchmark as get_latency_benchmark,
)
from labs.real_world_models.deepseek_r1_moe_optimization import (
    get_benchmark as get_deepseek_benchmark,
)
from labs.real_world_models.gpt4_architecture_optimization import (
    get_benchmark as get_gpt4_benchmark,
)
from core.harness.benchmark_harness import _cleanup_process_group
from core.harness.run_benchmarks import (
    INFORMATIONAL_BENCHMARKS,
    _collect_current_run_benchmark_orphan_pids,
    _collect_stale_benchmark_orphan_pids,
    _reap_benchmark_process_leftovers,
    _reap_run_descendants,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_FUSION_SIGNATURE_MODULES = (
    "ch12.baseline_kernel_fusion",
    "ch12.optimized_kernel_fusion",
    "ch12.optimized_kernel_fusion_llm_dedicated_stream_and_prefetch_for_blackwell",
    "ch12.optimized_kernel_fusion_llm_persistent_buffer_and_stream_friendly_setup",
    "ch12.optimized_kernel_fusion_llm_reuse_static_tensor_and_simplify_setup",
)
TIMEOUT_PRONE_SIGNATURE_CASES = (
    ("ch13.baseline_bandwidth_naive", 16_777_216, (4096, 4096, 16), "float32"),
    ("ch13.optimized_bandwidth_naive", 16_777_216, (4096, 4096, 16), "float32"),
    ("ch18.baseline_vllm_v1_integration", 8, (128,), "int64"),
    ("ch18.optimized_vllm_v1_integration", 8, (128,), "int64"),
)


def test_ch01_precision_benchmarks_disable_tf32_during_setup() -> None:
    initial_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    initial_cudnn = (
        bool(torch.backends.cudnn.allow_tf32)
        if torch.backends.cudnn.is_available()
        else None
    )

    for benchmark_cls in (OptimizedPerformanceBatchBenchmark, OptimizedPerformanceFP16Benchmark):
        bench = benchmark_cls()
        if isinstance(bench, OptimizedPerformanceBatchBenchmark):
            bench.workload = SimpleNamespace(performance_microbatches=2)
        bench.batch_size = 2
        bench.num_microbatches = 2
        bench.hidden_dim = 64
        bench.setup()
        try:
            assert torch.backends.cuda.matmul.allow_tf32 is False
            if torch.backends.cudnn.is_available():
                assert torch.backends.cudnn.allow_tf32 is False
        finally:
            bench.teardown()

    assert torch.backends.cuda.matmul.allow_tf32 == initial_matmul
    if initial_cudnn is not None:
        assert torch.backends.cudnn.allow_tf32 == initial_cudnn


def test_ch02_cublas_metrics_report_gemm_workload_not_transfer_placeholders() -> None:
    baseline_metrics = BaselineCublasBenchmark().get_custom_metrics()
    optimized_metrics = OptimizedCublasBenchmark().get_custom_metrics()

    expected_flops = float(2 * 2048 * 2048 * 2048)
    assert baseline_metrics["gemm.total_flops"] == expected_flops
    assert optimized_metrics["gemm.total_flops"] == expected_flops
    assert "transfer.achieved_gbps" not in baseline_metrics
    assert "transfer.achieved_gbps" not in optimized_metrics


def test_ch07_and_ch08_sources_do_not_ship_artificial_baseline_penalties() -> None:
    hbm_copy_source = (REPO_ROOT / "ch07" / "baseline_hbm_copy.cu").read_text(encoding="utf-8")
    threshold_source = (REPO_ROOT / "ch08" / "threshold_common.cuh").read_text(encoding="utf-8")

    assert "scalar_copy_kernel<<<64, 64>>>" not in hbm_copy_source
    assert "scalar_copy_kernel<<<blocks, threads>>>" in hbm_copy_source
    assert "const volatile float* volatile_inputs" not in threshold_source
    assert "volatile float redundant_eval" not in threshold_source
    assert "expensive_transform(-value" not in threshold_source


def test_ch07_tma_copy_surfaces_scalar_vs_strict_descriptor_tma_story() -> None:
    optimized_wrapper = (REPO_ROOT / "ch07" / "optimized_tma_copy.py").read_text(encoding="utf-8")
    optimized_cuda = (REPO_ROOT / "ch07" / "optimized_tma_copy.cu").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "ch07" / "README.md").read_text(encoding="utf-8")

    assert "Pipeline + Tensor-Map Neighbor Copy" in optimized_wrapper
    assert "strict `tma_copy` path" in optimized_wrapper
    assert "dst[global_row * N + global_col] = combine_values(" in optimized_cuda
    assert "output_tile[local_row][local_col] = combine_values(" in optimized_cuda
    assert "output_tile" in optimized_cuda
    assert "usable tensor-map/TMA support" in optimized_cuda
    assert "legacy async-neighbor demo" not in readme
    assert "strict tensor-map/TMA-capable run only" in readme


def test_occupancy_tuning_variants_match_their_filenames() -> None:
    wide_n = get_wide_n_benchmark()
    latency = get_latency_benchmark()

    assert wide_n.schedule.name == "bm64_bn256_bk32"
    assert wide_n.schedule.block_m == 64
    assert wide_n.schedule.block_n == 256
    assert wide_n.schedule.block_k == 32

    assert latency.schedule.name == "bm64_bn64_bk32_nw2"
    assert latency.schedule.block_m == 64
    assert latency.schedule.block_n == 64
    assert latency.schedule.block_k == 32
    assert latency.schedule.num_warps == 2


def test_real_world_model_entrypoints_return_harness_benchmarks() -> None:
    assert isinstance(get_deepseek_benchmark(), BaseBenchmark)
    assert isinstance(get_gpt4_benchmark(), BaseBenchmark)


def test_cleanup_process_group_escalates_when_group_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []

    def _fake_killpg(_pgid: int, sig: int) -> None:
        if sig == 0:
            return
        signals.append(sig)

    monkeypatch.setattr("core.harness.benchmark_harness.os.killpg", _fake_killpg)

    _cleanup_process_group(4242, grace_seconds=0.0)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_cleanup_process_group_ignores_missing_group(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_killpg(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("core.harness.benchmark_harness.os.killpg", _missing_killpg)

    _cleanup_process_group(4242)


def test_flexattention_metrics_use_attention_formula_and_hot_paths_skip_clone() -> None:
    baseline = BaselineFlexAttentionBenchmark()
    optimized = OptimizedFlexAttentionBenchmark()

    docs = baseline.seq_len // baseline.doc_span
    active_pairs = float(baseline.batch * baseline.heads * docs * baseline.doc_span * baseline.doc_span)
    expected_flops = float(4 * active_pairs * baseline.head_dim)
    assert baseline.get_custom_metrics()["flex_attention.total_flops"] == expected_flops
    assert optimized.get_custom_metrics()["flex_attention.total_flops"] == expected_flops

    flex_baseline_source = (REPO_ROOT / "labs" / "flexattention" / "baseline_flex_attention.py").read_text(
        encoding="utf-8"
    )
    flex_optimized_source = (REPO_ROOT / "labs" / "flexattention" / "optimized_flex_attention.py").read_text(
        encoding="utf-8"
    )
    flash4_source = (REPO_ROOT / "labs" / "flashattention4" / "flashattention4_benchmarks.py").read_text(
        encoding="utf-8"
    )

    assert "self.output = output_tensor.detach().float().clone()" not in flex_baseline_source
    assert "self.output = output_tensor.detach().float().clone()" not in flex_optimized_source
    assert "self.output = result.detach().float().clone()" not in flash4_source


def test_ch10_flash_attention_requires_real_flashattention_on_sm100() -> None:
    source = (REPO_ROOT / "ch10" / "optimized_flash_attention.py").read_text(encoding="utf-8")
    resolve_section = source.split("def _resolve_attention_runner", maxsplit=1)[1].split(
        "def _run_attention", maxsplit=1
    )[0]
    assert "candidates.append([SDPBackend.FLASH_ATTENTION])" in source
    assert "candidates.append([SDPBackend.EFFICIENT_ATTENTION])" in source
    assert "if major >= 10" in resolve_section
    assert "FAIL FAST: FlashAttention required for ch10" in resolve_section
    assert "self._selected_backend_name = candidate[0].name.lower()" in source


def test_ch10_flash_attention_prefers_external_flash_engines_before_sdpa_fallback() -> None:
    source = (REPO_ROOT / "ch10" / "optimized_flash_attention.py").read_text(encoding="utf-8")

    flash3_idx = source.index("flash_attn_3.flash_attn_interface")
    flash2_idx = source.index("flash_attn.flash_attn_interface")

    assert flash3_idx < flash2_idx
    assert "self._selected_engine_name = \"sdpa\"" in source


def test_ch18_paged_attention_uses_real_block_table_sparse_kernel() -> None:
    common_source = (REPO_ROOT / "ch18" / "paged_attn_split_common.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch18" / "optimized_paged_attn_layout.py").read_text(encoding="utf-8")

    assert "self.block_table" in common_source
    assert "torch.roll(block_ids" in common_source
    assert "create_block_mask, flex_attention" in common_source
    assert "dense_mask[:, 0][allowed] = 0.0" in common_source
    assert "return create_block_mask(" in common_source
    assert 'torch.compile(flex_attention, mode="max-autotune")' in common_source
    assert "return self._flex_attention_fn(self.q, self.k_dense, self.v_dense, block_mask=self.block_mask)" in common_source
    assert "_gather_paged_kv" not in common_source
    assert "LayoutPagedAttnBase" in optimized_source


def test_ch04_nvshmem_symmetric_broadcast_overlap_defines_done_event() -> None:
    source = (REPO_ROOT / "ch04" / "nvshmem_vs_nccl_benchmark.py").read_text(encoding="utf-8")
    symmetric_section = source.split("def _measure_symmetric_broadcast", maxsplit=1)[1].split(
        "def sweep_sizes", maxsplit=1
    )[0]

    assert "done = torch.cuda.Event() if overlap_compute and comm_stream is not None else None" in symmetric_section
    assert "if overlap_compute and comm_stream is not None and done is not None:" in symmetric_section


def test_ch19_vectorization_memory_preconverts_fp16_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch19" / "optimized_vectorization_memory.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "_cached_a_fp16" not in source
    assert "_cached_b_fp16" not in source
    assert "self._tensor_a_fp16 = self.tensor_a.to(self._compute_dtype)" in setup_section
    assert "self._tensor_b_fp16 = self.tensor_b.to(self._compute_dtype)" in setup_section
    assert "torch.add(self._tensor_a_fp16, self._tensor_b_fp16, out=self._work)" in benchmark_section
    assert ".to(self._compute_dtype)" not in benchmark_section


def test_moe_cuda_decode_attention_preconverts_bf16_outside_hot_loop() -> None:
    source = (REPO_ROOT / "labs" / "moe_cuda" / "optimized_decode_attention.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def finalize_iteration_metrics", maxsplit=1
    )[0]

    assert "def _cached_bf16" not in source
    assert "self._refresh_bf16_cache(force=True)" in setup_section
    assert "self._q_bf16 = self.q.to(torch.bfloat16)" not in benchmark_section
    assert "self._refresh_bf16_cache()" in source
    assert "q = self._q_bf16" in benchmark_section
    assert "k = self._k_bf16" in benchmark_section
    assert "v = self._v_bf16" in benchmark_section


def test_ch20_bf16_mlp_preconverts_activation_dtype_outside_hot_loop() -> None:
    source = (REPO_ROOT / "ch20" / "optimized_bf16_mlp.py").read_text(encoding="utf-8")
    setup_section = source.split("def benchmark_fn", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self._model_dtype =" in setup_section
    assert "self._x_model_dtype = self.x.to(dtype=self._model_dtype)" in setup_section
    assert "next(self.model.parameters()).dtype" not in benchmark_section
    assert ".to(dtype=" not in benchmark_section
    assert "self.output = self.model(self._x_model_dtype)" in benchmark_section


def test_ch13_regional_compile_moves_fp32_verification_conversion_out_of_hot_loop() -> None:
    source = (REPO_ROOT / "ch13" / "optimized_regional_compile.py").read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]

    assert ".detach().float().clone()" not in benchmark_section
    assert "self.output = self.model(x).detach().clone()" in benchmark_section
    assert "output=self._verify_output.float().clone()" in capture_section


def test_ch13_memory_profiling_pair_keeps_compute_dtype_fixed_and_direct_output_capture() -> None:
    baseline_source = (REPO_ROOT / "ch13" / "baseline_memory_profiling.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch13" / "optimized_memory_profiling.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert 'signature_equivalence_group = "ch13_memory_profiling_checkpointing"' in baseline_source
    assert 'signature_equivalence_group = "ch13_memory_profiling_checkpointing"' in optimized_source
    assert "dtype=torch.bfloat16" not in optimized_source
    assert "self.inputs_fp32" not in optimized_source
    assert "self.targets_fp32" not in optimized_source
    for source in (baseline_source, optimized_source):
        assert "dtype=torch.float32" in source
    assert "self.output = outputs.detach().clone()" in baseline_benchmark
    assert "self.output = outputs.detach().clone()" in optimized_benchmark
    assert "self.output_buffer" not in optimized_source
    assert 'return "memory"' in baseline_source
    assert 'return "memory"' in optimized_source


def test_ch12_kernel_launches_pair_keeps_hot_path_work_fixed() -> None:
    baseline_source = (REPO_ROOT / "ch12" / "baseline_kernel_launches.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch12" / "optimized_kernel_launches.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert ".clone()" not in baseline_benchmark
    assert "self.x_input =" in optimized_source
    assert "self.x_capture" not in optimized_source
    assert "self.graph_output = self.work_a" in optimized_source


def test_ch13_precisionfp8_pad_inner_runs_single_forward_per_timed_iteration() -> None:
    baseline_source = (REPO_ROOT / "ch13" / "baseline_precisionfp8_pad_inner.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch13" / "optimized_precisionfp8_pad_inner.py").read_text(encoding="utf-8")
    baseline_benchmark = baseline_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]
    optimized_benchmark = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert baseline_benchmark.count("self.model(") == 1
    assert optimized_benchmark.count("self.model(") == 1


def test_ch13_matmul_and_regional_compile_keep_precision_fixed_across_pairs() -> None:
    baseline_matmul = (REPO_ROOT / "ch13" / "baseline_matmul_pytorch.py").read_text(encoding="utf-8")
    optimized_matmul = (REPO_ROOT / "ch13" / "optimized_matmul_pytorch.py").read_text(encoding="utf-8")
    baseline_regional = (REPO_ROOT / "ch13" / "baseline_regional_compile.py").read_text(encoding="utf-8")
    optimized_regional = (REPO_ROOT / "ch13" / "optimized_regional_compile.py").read_text(encoding="utf-8")

    assert "dtype=torch.float32" not in baseline_matmul
    assert "dtype=torch.float16" in baseline_matmul
    assert "dtype=torch.float16" in optimized_matmul
    assert "self.compiled_model = torch.compile(" in baseline_regional
    assert "dtype=torch.bfloat16" in baseline_regional
    assert "dtype=torch.bfloat16" in optimized_regional


def test_ch10_tcgen05_warp_specialized_kernel_uses_direct_epilogue_copy() -> None:
    optimized_wrapper = (REPO_ROOT / "ch10" / "optimized_tcgen05_warp_specialization.py").read_text(
        encoding="utf-8"
    )
    kernel_source = (REPO_ROOT / "ch10" / "tcgen05_warp_specialized.cu").read_text(encoding="utf-8")

    assert "matmul_tcgen05_warp_specialized(self.matrix_a, self.matrix_b)" in optimized_wrapper
    assert "torch::zeros({m, n}, options)" not in kernel_source
    assert "axpby(" not in kernel_source
    assert "copy(tDrAcc, tDgD);" in kernel_source


def test_moe_cuda_graphs_journey_uses_real_graph_capture_and_correct_leveling() -> None:
    benchmark_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "moe_benchmark.py").read_text(
        encoding="utf-8"
    )
    model_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "moe_model.py").read_text(
        encoding="utf-8"
    )
    cuda_graph_source = (REPO_ROOT / "labs" / "moe_optimization_journey" / "level5_cudagraphs.py").read_text(
        encoding="utf-8"
    )
    optimized_entrypoint = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "optimized_moe_cuda_graphs.py"
    ).read_text(
        encoding="utf-8"
    )
    optimized_main_entrypoint = (
        REPO_ROOT / "labs" / "moe_optimization_journey" / "optimized_moe.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "self._cuda_graph = torch.cuda.CUDAGraph()" in model_source
    assert "self._cuda_graph.replay()" in model_source
    assert "self.output = logits[:, :1, : min(8, logits.shape[-1])]" in benchmark_source
    assert ".float().clone()" not in benchmark_source.split("def capture_verification_payload", maxsplit=1)[0]
    assert "Level6CUDAGraphs" in cuda_graph_source
    assert "LEVEL = 6" in cuda_graph_source
    assert "Level6CUDAGraphs" in optimized_entrypoint
    assert "Level7Compiled" in optimized_main_entrypoint


def test_persistent_decode_verification_clone_stays_out_of_hot_path() -> None:
    targets = [
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_persistent_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_cuda.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_graphs.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_persistent_decode_triton.py",
        REPO_ROOT / "labs" / "persistent_decode" / "baseline_tma_prefill_decode.py",
        REPO_ROOT / "labs" / "persistent_decode" / "optimized_tma_prefill_decode.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        benchmark_section = text.split("def capture_verification_payload", maxsplit=1)[0]
        assert ".float().clone()" not in benchmark_section
        assert ".detach().clone()" not in benchmark_section


def test_iteration_seed_and_clone_fixes_for_reviewed_pairs_remain_applied() -> None:
    baseline_pipeline = (REPO_ROOT / "ch10" / "baseline_pipeline_3stage.py").read_text(encoding="utf-8")
    optimized_pipeline = (REPO_ROOT / "ch10" / "optimized_pipeline_3stage.py").read_text(encoding="utf-8")
    baseline_gluon = (
        REPO_ROOT / "labs" / "flashattention_gluon" / "baseline_flashattention_gluon.py"
    ).read_text(encoding="utf-8")
    optimized_gluon = (
        REPO_ROOT / "labs" / "flashattention_gluon" / "optimized_flashattention_gluon.py"
    ).read_text(encoding="utf-8")
    blackwell = (REPO_ROOT / "labs" / "blackwell_matmul" / "blackwell_benchmarks.py").read_text(
        encoding="utf-8"
    )
    baseline_double_buffer = (REPO_ROOT / "ch19" / "baseline_memory_double_buffering.py").read_text(
        encoding="utf-8"
    )
    optimized_double_buffer = (REPO_ROOT / "ch19" / "optimized_memory_double_buffering.py").read_text(
        encoding="utf-8"
    )
    optimized_rack_prep = (REPO_ROOT / "ch03" / "optimized_rack_prep.py").read_text(encoding="utf-8")

    for source in (baseline_pipeline, optimized_pipeline, baseline_gluon, optimized_gluon):
        assert "iterations=10" in source
        assert "warmup=5" in source

    assert "torch.manual_seed(42)" in blackwell
    assert "torch.cuda.manual_seed_all(42)" in blackwell
    for source in (baseline_double_buffer, optimized_double_buffer):
        assert "torch.manual_seed(42)" in source
        assert "torch.cuda.manual_seed_all(42)" in source

    assert "host_template.pin_memory()" in optimized_rack_prep
    assert "host_template.clone().pin_memory()" in optimized_rack_prep


def test_ch15_optimized_monolithic_uses_token_equivalent_decode_steps() -> None:
    common_source = (REPO_ROOT / "ch15" / "inference_monolithic_common.py").read_text(encoding="utf-8")
    optimized_source = (REPO_ROOT / "ch15" / "optimized_inference_monolithic.py").read_text(encoding="utf-8")

    assert "def decode_step(" in common_source
    assert "self.output = self.model.decode_autoregressive(" in optimized_source
    assert "output_buffer=self._decode_buffer" in optimized_source
    assert "self.output = self.model.decode(kv_cache, num_tokens=self.num_tokens)" not in optimized_source


def test_ch15_baseline_monolithic_uses_harness_timing_not_per_token_cuda_events() -> None:
    source = (REPO_ROOT / "ch15" / "baseline_inference_monolithic.py").read_text(encoding="utf-8")

    assert "torch.cuda.Event" not in source
    assert "self._last_elapsed_ms" in source
    assert "finalize_iteration_metrics" in source
    assert "self.model.decode(decode_state, num_tokens=1)" in source


def test_ch03_pageable_copy_is_not_marked_informational() -> None:
    assert "pageable_copy" not in INFORMATIONAL_BENCHMARKS.get("ch03", set())


def test_clean_all_benchmark_pairs_tracker_is_rebaselined() -> None:
    tracker = REPO_ROOT / ".cursor" / "plans" / "clean_all_benchmark_pairs_6db4c258.plan.md"
    text = tracker.read_text(encoding="utf-8")

    assert "status: pending" not in text
    assert "Rebaselined on 2026-03-16 against current repo truth" in text
    assert "tests/test_benchmark_hygiene_regressions.py" in text


def test_run_benchmarks_reaps_orphaned_benchmark_processes_from_older_runs() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=old-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (orphan_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        live_parent_dir = proc_root / "4321"
        live_parent_dir.mkdir()
        (live_parent_dir / "stat").write_text("4321 (python3) S 2222 0 0 0 0\n", encoding="utf-8")
        (live_parent_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=other-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (live_parent_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        parent_dir = proc_root / "2222"
        parent_dir.mkdir()
        (parent_dir / "stat").write_text("2222 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        current_run_dir = proc_root / "9999"
        current_run_dir.mkdir()
        (current_run_dir / "stat").write_text("9999 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (current_run_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (current_run_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        stale = _collect_stale_benchmark_orphan_pids(
            current_run_id="current-run",
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert stale == [1234]


def test_run_benchmarks_identifies_detached_current_run_processes_by_owner_pid() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        owner_dir = proc_root / "555"
        owner_dir.mkdir()
        (owner_dir / "stat").write_text("555 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=555\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (orphan_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        descendant_dir = proc_root / "1235"
        descendant_dir.mkdir()
        (descendant_dir / "stat").write_text("1235 (python3) S 555 0 0 0 0\n", encoding="utf-8")
        (descendant_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=555\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (descendant_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        other_owner_dir = proc_root / "1236"
        other_owner_dir.mkdir()
        (other_owner_dir / "stat").write_text("1236 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (other_owner_dir / "environ").write_bytes(
            b"AISP_BENCHMARK_OWNER_RUN_ID=current-run\0"
            b"AISP_BENCHMARK_OWNER_PID=999\0"
            b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0"
        )
        (other_owner_dir / "cmdline").write_bytes(b"/usr/bin/python3\0-m\0core.harness.isolated_runner\0")

        current_orphans = _collect_current_run_benchmark_orphan_pids(
            current_run_id="current-run",
            current_owner_pid=555,
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert current_orphans == [1234]


def test_run_benchmarks_identifies_detached_current_run_processes_by_cmdline_owner_markers() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        owner_dir = proc_root / "555"
        owner_dir.mkdir()
        (owner_dir / "stat").write_text("555 (python3) S 1 0 0 0 0\n", encoding="utf-8")

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0")
        (orphan_dir / "cmdline").write_bytes(
            b"/usr/bin/python3\0-m\0core.profiling.nsys_capture_helper\0"
            b"--aisp-owner-run-id\0current-run\0--aisp-owner-pid\0"
            b"555\0"
        )

        current_orphans = _collect_current_run_benchmark_orphan_pids(
            current_run_id="current-run",
            current_owner_pid=555,
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert current_orphans == [1234]


def test_run_benchmarks_reaps_orphaned_benchmark_processes_from_older_runs_via_cmdline_marker() -> None:
    with tempfile.TemporaryDirectory() as proc_dir:
        proc_root = Path(proc_dir)

        orphan_dir = proc_root / "1234"
        orphan_dir.mkdir()
        (orphan_dir / "stat").write_text("1234 (python3) S 1 0 0 0 0\n", encoding="utf-8")
        (orphan_dir / "environ").write_bytes(b"PWD=" + str(REPO_ROOT).encode("utf-8") + b"\0")
        (orphan_dir / "cmdline").write_bytes(
            b"/usr/bin/python3\0-m\0core.profiling.nsys_capture_helper\0"
            b"--aisp-owner-run-id\0old-run\0"
        )

        stale = _collect_stale_benchmark_orphan_pids(
            current_run_id="current-run",
            repo_root=REPO_ROOT,
            proc_root=proc_root,
        )

        assert stale == [1234]


def test_run_benchmarks_reaps_current_run_descendants() -> None:
    process = subprocess.Popen(
        ["python", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None

        _reap_run_descendants("unit_test_descendant_cleanup", grace_seconds=0.2)

        deadline = time.time() + 5.0
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.05)

        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def test_benchmark_leftover_cleanup_does_not_kill_unmarked_current_descendants() -> None:
    process = subprocess.Popen(
        ["python", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None

        _reap_benchmark_process_leftovers(
            "unit_test_preserve_unmarked_descendant",
            current_run_id="unit-test-run",
            current_owner_pid=999999,
            repo_root=REPO_ROOT,
        )

        time.sleep(0.5)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def test_ch12_kernel_fusion_variants_publish_static_input_signatures_without_execution() -> None:
    for module_name in KERNEL_FUSION_SIGNATURE_MODULES:
        module = importlib.import_module(module_name)
        benchmark = module.get_benchmark()
        signature = coerce_input_signature(benchmark.get_input_signature())

        assert signature.batch_size == 16_000_000
        assert signature.dtypes["workload"] == "float32"
        assert signature.shapes["workload"] == (16_000_000, 10)


def test_timeout_prone_pairs_publish_static_input_signatures_without_execution() -> None:
    for module_name, expected_batch_size, expected_shape, expected_dtype in TIMEOUT_PRONE_SIGNATURE_CASES:
        module = importlib.import_module(module_name)
        benchmark = module.get_benchmark()
        signature = coerce_input_signature(benchmark.get_input_signature())

        assert signature.batch_size == expected_batch_size
        assert signature.dtypes["workload"] == expected_dtype
        assert signature.shapes["workload"] == expected_shape
