from __future__ import annotations

import inspect

import torch

from core.harness.benchmark_harness import BenchmarkConfig, ReadOnlyBenchmarkConfigView
from labs.recsys_sequence_ranking.baseline_sequence_ranking import BaselineSequenceRankingBenchmark
from labs.recsys_sequence_ranking.compare_sequence_ranking import _measure
from labs.recsys_sequence_ranking.optimized_sequence_ranking import (
    OptimizedSequenceRankingBenchmark,
)
from labs.recsys_sequence_ranking.recsys_sequence_ranking_common import (
    SequenceRankingWorkload,
    apply_cli_overrides,
    baseline_forward,
    build_inputs,
    build_model_state,
    build_workspace,
    candidate_scores_torch,
    context_sum_baseline,
    context_sum_vectorized,
    optimized_forward,
    ranking_metrics,
    resolve_score_backend,
    sequence_mean_baseline,
    sequence_mean_vectorized,
    warm_optimized_path,
)


class _FakeMeasuredBenchmark:
    def __init__(self) -> None:
        self.calls = 0

    def benchmark_fn(self) -> None:
        self.calls += 1


def _small_workload() -> SequenceRankingWorkload:
    return SequenceRankingWorkload(
        batch_size=4,
        seq_len=6,
        num_tables=3,
        embedding_dim=16,
        hidden_dim=24,
        num_candidates=8,
        item_vocab_size=128,
        context_vocab_size=64,
        min_history_len=2,
        zipf_alpha=1.05,
        seed=7,
        dtype=torch.float32,
        use_compile=False,
        score_backend="torch",
    )


def test_apply_cli_overrides_clamps_history_length() -> None:
    workload = apply_cli_overrides(_small_workload(), ["--seq-len", "4", "--min-history-len", "9"])
    assert workload.seq_len == 4
    assert workload.min_history_len == 4


def test_build_inputs_is_deterministic() -> None:
    workload = _small_workload()
    source = inspect.getsource(build_inputs)
    inputs_a = build_inputs(workload, torch.device("cpu"))
    inputs_b = build_inputs(workload, torch.device("cpu"))

    assert "input_stats = torch.stack(" in source
    assert ".mean().item()" not in source
    assert torch.equal(inputs_a.sequence_ids, inputs_b.sequence_ids)
    assert torch.equal(inputs_a.sequence_mask, inputs_b.sequence_mask)
    assert torch.equal(inputs_a.sequence_lengths, inputs_b.sequence_lengths)
    assert torch.equal(inputs_a.context_ids, inputs_b.context_ids)
    assert torch.equal(inputs_a.candidate_ids, inputs_b.candidate_ids)
    assert inputs_a.avg_sequence_length == inputs_b.avg_sequence_length
    assert inputs_a.hot_candidate_share_pct == inputs_b.hot_candidate_share_pct


def test_ranking_metrics_reuse_cpu_generated_input_metadata() -> None:
    workload = _small_workload()
    inputs = build_inputs(workload, torch.device("cpu"))
    source = inspect.getsource(ranking_metrics)

    expected_avg = float(inputs.sequence_lengths.to(torch.float32).mean().item())
    hot_threshold = max(workload.item_vocab_size // 100, 1)
    expected_hot_share = float(
        (inputs.candidate_ids < hot_threshold).to(torch.float32).mean().item() * 100.0
    )
    metrics = ranking_metrics(
        workload,
        inputs,
        score_backend="torch",
        compile_enabled=False,
    )

    assert "inputs.avg_sequence_length" in source
    assert "inputs.hot_candidate_share_pct" in source
    assert ".item()" not in source
    assert metrics["ranking.avg_sequence_length"] == expected_avg
    assert metrics["ranking.hot_candidate_share_pct"] == expected_hot_share


def test_baseline_and_optimized_torch_paths_match_on_cpu() -> None:
    workload = _small_workload()
    inputs = build_inputs(workload, torch.device("cpu"))
    state = build_model_state(workload, torch.device("cpu"))
    baseline_workspace = build_workspace(workload, torch.device("cpu"))
    optimized_workspace = build_workspace(workload, torch.device("cpu"))

    baseline_scores = baseline_forward(inputs, state, baseline_workspace)
    optimized_scores = optimized_forward(
        inputs,
        state,
        compiled_tower=None,
        score_backend="torch",
        workspace=optimized_workspace,
    )

    torch.testing.assert_close(baseline_scores, optimized_scores, rtol=1e-6, atol=1e-6)


def test_baseline_poolers_seed_reusable_outputs_before_accumulating() -> None:
    sequence_source = inspect.getsource(sequence_mean_baseline)
    context_source = inspect.getsource(context_sum_baseline)

    assert "out.copy_(token_vec * mask[:, 0:1])" in sequence_source
    assert "for t in range(1, inputs.sequence_ids.shape[1]):" in sequence_source
    assert "out.copy_(state.context_embeddings[0, inputs.context_ids[:, 0]])" in context_source
    assert "for table_idx in range(1, inputs.context_ids.shape[1]):" in context_source


def test_workspace_backed_vectorized_helpers_match_fallback_on_cpu() -> None:
    workload = _small_workload()
    inputs = build_inputs(workload, torch.device("cpu"))
    state = build_model_state(workload, torch.device("cpu"))
    workspace = build_workspace(workload, torch.device("cpu"))
    build_workspace_source = inspect.getsource(build_workspace)

    fallback_sequence = sequence_mean_vectorized(inputs, state)
    workspace_sequence = sequence_mean_vectorized(inputs, state, workspace)
    fallback_context = context_sum_vectorized(inputs, state)
    workspace_context = context_sum_vectorized(inputs, state, workspace)

    assert workspace.sequence_metadata_key is not None
    assert workspace.context_table_index.shape == (1, workload.num_tables)
    assert ".expand(workload.batch_size, -1).clone()" not in build_workspace_source
    assert workspace_sequence.data_ptr() == workspace.sequence_accum.data_ptr()
    assert workspace_context.data_ptr() == workspace.context_accum.data_ptr()
    torch.testing.assert_close(workspace_sequence, fallback_sequence, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(workspace_context, fallback_context, rtol=1e-6, atol=1e-6)


def test_torch_candidate_scoring_reuses_workspace_output_on_cpu() -> None:
    workload = _small_workload()
    inputs = build_inputs(workload, torch.device("cpu"))
    state = build_model_state(workload, torch.device("cpu"))
    workspace = build_workspace(workload, torch.device("cpu"))
    user_vec = torch.randn(
        workload.batch_size,
        workload.embedding_dim,
        dtype=workload.dtype,
    )
    source = inspect.getsource(candidate_scores_torch)

    fallback_scores = candidate_scores_torch(user_vec, inputs, state)
    workspace_scores = candidate_scores_torch(user_vec, inputs, state, workspace.score_output)

    assert "out=out.unsqueeze(2)" in source
    assert workspace_scores.data_ptr() == workspace.score_output.data_ptr()
    torch.testing.assert_close(workspace_scores, fallback_scores, rtol=1e-6, atol=1e-6)


def test_ranking_hot_paths_use_inference_mode() -> None:
    baseline_source = inspect.getsource(BaselineSequenceRankingBenchmark.benchmark_fn)
    optimized_source = inspect.getsource(OptimizedSequenceRankingBenchmark.benchmark_fn)
    warmup_source = inspect.getsource(warm_optimized_path)

    assert "with torch.inference_mode():" in baseline_source
    assert "with torch.inference_mode():" in optimized_source
    assert "with torch.inference_mode():" in warmup_source
    assert "with torch.no_grad():" not in baseline_source
    assert "with torch.no_grad():" not in optimized_source
    assert "with torch.no_grad():" not in warmup_source


def test_resolve_score_backend_respects_availability() -> None:
    resolved = resolve_score_backend("auto")
    assert resolved in {"torch", "triton"}
    assert resolve_score_backend("torch") == "torch"


def test_benchmarks_prefer_application_replay_and_explicit_nvtx_scope_for_ncu() -> None:
    baseline = BaselineSequenceRankingBenchmark()
    optimized = OptimizedSequenceRankingBenchmark()

    assert baseline.preferred_ncu_replay_mode == "application"
    assert optimized.preferred_ncu_replay_mode == "application"
    assert baseline.get_config().ncu_replay_mode == "application"
    assert optimized.get_config().ncu_replay_mode == "application"
    assert baseline.get_config().nsys_nvtx_include == ["compute_kernel:profile"]
    assert optimized.get_config().nsys_nvtx_include == ["compute_kernel:profile"]
    assert baseline.get_config().profiling_warmup == 0
    assert optimized.get_config().profiling_warmup == 0
    assert baseline.get_config().profiling_iterations == 1
    assert optimized.get_config().profiling_iterations == 1


def test_optimized_benchmark_disables_compile_for_ncu_wrapper_runs() -> None:
    optimized = OptimizedSequenceRankingBenchmark()
    profiling_config = BenchmarkConfig(enable_profiling=True, enable_ncu=True, enable_nvtx=True)
    optimized._config = ReadOnlyBenchmarkConfigView.from_config(profiling_config)

    assert optimized._should_enable_compile() is False


def test_compare_measure_cpu_path_counts_warmup_and_timed_iterations(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bench = _FakeMeasuredBenchmark()

    latency_ms = _measure(bench, warmup=3, iterations=5)

    assert bench.calls == 8
    assert latency_ms >= 0.0
