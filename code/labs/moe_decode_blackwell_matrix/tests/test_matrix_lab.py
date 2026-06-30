from __future__ import annotations

import inspect
from pathlib import Path

import torch

from labs.moe_decode_blackwell_matrix.matrix_catalog import load_playbook
from labs.moe_decode_blackwell_matrix.matrix_types import MatrixScenario
from labs.moe_decode_blackwell_matrix.profiler.compare import auto_select_graph_pair
from labs.moe_decode_blackwell_matrix.runner import (
    DispatchBatch,
    _compare_outputs,
    _routing_stats,
    build_decode_batches,
    render_console_table,
    summarize_rows,
)


def test_smoke_playbook_loads() -> None:
    playbook = load_playbook("smoke_b200")
    assert playbook.name == "smoke_b200"
    assert playbook.hidden_size == 256
    assert playbook.decode_batches == (1, 8)


def test_build_decode_batches_cpu_contract() -> None:
    scenario = MatrixScenario(
        playbook_name="unit",
        description="unit",
        seed=17,
        dtype="bf16",
        hidden_size=64,
        intermediate_size=128,
        steps=3,
        warmup=1,
        repeats=1,
        num_experts=8,
        top_k=2,
        decode_batch=4,
        routing_policy="sticky",
        schedule_mode="persistent",
        launch_mode="eager",
    )
    batches = build_decode_batches(scenario, device=torch.device("cpu"))
    assert len(batches) == 3
    for batch in batches:
        assert batch.hidden_states.shape == (4, 64)
        assert batch.expert_indices.shape == (4, 2)
        assert batch.expert_weights.shape == (4, 2)
        assert torch.all(batch.expert_indices >= 0)
        assert torch.all(batch.expert_indices < 8)
        assert torch.allclose(
            batch.expert_weights.sum(dim=-1).float(),
            torch.ones(4),
            atol=1e-5,
        )


def test_routing_stats_batches_scalar_materialization() -> None:
    source = inspect.getsource(_routing_stats)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [2, 0]], dtype=torch.long)

    entropy, active, max_tokens = _routing_stats(indices, num_experts=4)

    assert "stats = torch.empty(4, device=counts.device, dtype=counts.dtype)" in source
    assert "stats[0].copy_(total_tensor)" in source
    assert "stats[1].copy_((counts > 0).sum().to(counts.dtype))" in source
    assert "stats[2].copy_(counts.max())" in source
    assert "stats[3].copy_(entropy_tensor)" in source
    assert "stats_host = stats.detach().cpu()" in source
    assert "stats_host = torch.stack(" not in source
    assert "total = float(stats_host[0])" in source
    assert "active_count = float(stats_host[1])" in source
    assert "max_tokens_float = float(stats_host[2])" in source
    assert "entropy = float(stats_host[3])" in source
    assert ".tolist()" not in source
    assert ".sum().item()" not in source
    assert ".max().item()" not in source
    assert entropy > 0.0
    assert active == 1.0
    assert max_tokens == 3


def test_profiler_capture_selects_top_ops_without_full_sort() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "profiler" / "capture.py"
    ).read_text(encoding="utf-8")
    top_ops_section = source.split("def _top_ops", maxsplit=1)[1].split(
        "def profile_scenario",
        maxsplit=1,
    )[0]

    assert "import heapq" in source
    assert "def _top_ops(events: Iterable[Any], *, top_ops: int)" in source
    assert "heapq.nlargest(top_ops, events, key=_self_device_time_us)" in top_ops_section
    assert "sorted(events, key=_self_device_time_us" not in top_ops_section
    assert "events = profile.key_averages()" in source
    assert "for event in events:" in source
    assert "total_self_cuda_time_us += _self_device_time_us(event)" in source
    assert "list(profile.key_averages())" not in source
    assert "all_events =" not in source
    assert "sum(_self_device_time_us(event) for event in events)" not in source


def test_profiler_compare_selects_graph_pair_without_full_sort() -> None:
    source = inspect.getsource(auto_select_graph_pair)
    rows = [
        {
            "config_id": "wk_a_eager",
            "workload_key": "wk_a",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "eager",
            "step_mean_ms": 8.0,
        },
        {
            "config_id": "wk_a_graph",
            "workload_key": "wk_a",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "cuda_graph",
            "step_mean_ms": 4.0,
        },
        {
            "config_id": "wk_b_eager",
            "workload_key": "wk_b",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "eager",
            "step_mean_ms": 9.0,
        },
        {
            "config_id": "wk_b_graph",
            "workload_key": "wk_b",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "cuda_graph",
            "step_mean_ms": 3.0,
        },
    ]

    eager_row, graph_row = auto_select_graph_pair(rows)

    assert eager_row["config_id"] == "wk_b_eager"
    assert graph_row["config_id"] == "wk_b_graph"
    assert "best_pair" in source
    assert "best_graph_by_workload" in source
    assert "for row in rows:" in source
    assert "ranked_pairs" not in source
    assert "ok_rows = [" not in source
    assert "graph_rows = [" not in source
    assert ".sort(" not in source


def test_compare_outputs_batches_diff_materialization() -> None:
    source = inspect.getsource(_compare_outputs)
    scenario = MatrixScenario(
        playbook_name="unit",
        description="unit",
        seed=17,
        dtype="bf16",
        hidden_size=4,
        intermediate_size=8,
        steps=2,
        warmup=1,
        repeats=1,
        num_experts=2,
        top_k=1,
        decode_batch=2,
        routing_policy="balanced",
        schedule_mode="dynamic",
        launch_mode="eager",
    )
    batches = [
        DispatchBatch(
            hidden_states=torch.zeros(2, 4, dtype=torch.bfloat16),
            expert_indices=torch.zeros(2, 1, dtype=torch.long),
            expert_weights=torch.ones(2, 1, dtype=torch.bfloat16),
            routing_entropy_norm=0.0,
            active_expert_fraction=0.5,
            max_tokens_per_expert=2,
        ),
        DispatchBatch(
            hidden_states=torch.ones(2, 4, dtype=torch.bfloat16),
            expert_indices=torch.zeros(2, 1, dtype=torch.long),
            expert_weights=torch.ones(2, 1, dtype=torch.bfloat16),
            routing_entropy_norm=0.0,
            active_expert_fraction=0.5,
            max_tokens_per_expert=2,
        ),
    ]
    refs = [batches[0].hidden_states.float() + 0.125, batches[1].hidden_states.float()]

    class EchoExperts:
        def forward_grouped(
            self,
            hidden_states: torch.Tensor,
            expert_indices: torch.Tensor,
            expert_weights: torch.Tensor,
        ) -> torch.Tensor:
            return hidden_states

    assert _compare_outputs(EchoExperts(), batches, refs, scenario=scenario) == 0.125

    assert "max_diff: torch.Tensor | None = None" in source
    assert "torch.maximum(max_diff, diff, out=max_diff)" in source
    assert "return float(max_diff.detach().cpu())" in source
    assert "max_diff.detach().cpu().tolist()" not in source
    assert "diff_tensors" not in source
    assert "torch.stack(diff_tensors)" not in source
    assert "diffs.append(float(" not in source
    assert "torch.max(torch.abs(out.float() - ref.float())).item()" not in source


def test_summary_builds_pairwise_sections() -> None:
    source = inspect.getsource(summarize_rows)
    runner_source = (Path(__file__).resolve().parents[1] / "runner.py").read_text(
        encoding="utf-8"
    )
    run_matrix_source = (Path(__file__).resolve().parents[1] / "run_matrix.py").read_text(
        encoding="utf-8"
    )
    rows = [
        {
            "config_id": "wk_dyn",
            "workload_key": "wk",
            "status": "ok",
            "schedule_mode": "dynamic",
            "launch_mode": "eager",
            "step_mean_ms": 2.0,
            "tokens_per_second": 500.0,
        },
        {
            "config_id": "wk_pst",
            "workload_key": "wk",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "eager",
            "step_mean_ms": 1.25,
            "tokens_per_second": 800.0,
            "capture_ms": None,
        },
        {
            "config_id": "wk_grf",
            "workload_key": "wk",
            "status": "ok",
            "schedule_mode": "persistent",
            "launch_mode": "cuda_graph",
            "step_mean_ms": 1.0,
            "tokens_per_second": 1000.0,
            "capture_ms": 4.5,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["best_overall"]["config_id"] == "wk_grf"
    assert summary["persistent_vs_dynamic"][0]["speedup"] == 1.6
    assert summary["graph_vs_eager"][0]["speedup"] == 1.25
    assert "for row in rows:" in source
    assert "ok_row_count += 1" in source
    assert "unsupported_row_count += 1" in source
    assert "error_row_count += 1" in source
    assert "by_config[(row[\"workload_key\"], row[\"schedule_mode\"], row[\"launch_mode\"])] = row" in source
    assert "workload_keys.add(row[\"workload_key\"])" in source
    assert "ok_rows = [" not in source
    assert "sum(1 for row in rows" not in source
    assert "min(ok_rows" not in source
    assert "for row in ok_rows" not in source
    assert "upper_tail_count = sample_count - p95_index" in runner_source
    assert "step_p95_ms = heapq.nlargest(upper_tail_count, elapsed_per_step_ms)[-1]" in runner_source
    assert "sorted_latencies = sorted(elapsed_per_step_ms)" not in runner_source
    assert 'return 0 if int(summary["error_row_count"]) == 0 else 2' in run_matrix_source
    assert "error_count = sum(1 for row in rows" not in run_matrix_source


def test_console_table_uses_bounded_selection() -> None:
    source = inspect.getsource(render_console_table)
    rows = [
        {
            "config_id": f"cfg_{idx}",
            "decode_batch": idx,
            "routing_policy": "balanced",
            "schedule_mode": "persistent",
            "launch_mode": "eager",
            "status": "ok",
            "step_mean_ms": float(ms),
            "tokens_per_second": 1000.0 / float(ms),
        }
        for idx, ms in enumerate((9.0, 3.0, 7.0, 1.0), start=1)
    ]
    rows.append({"status": "error", "config_id": "bad", "step_mean_ms": 0.1})

    table = render_console_table(rows, limit=2)

    assert "`cfg_4`" in table
    assert "`cfg_2`" in table
    assert "`cfg_3`" not in table
    assert "`bad`" not in table
    assert "heapq.nsmallest(" in source
    assert "sorted(" not in source
