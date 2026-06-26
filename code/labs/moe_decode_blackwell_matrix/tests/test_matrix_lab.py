from __future__ import annotations

import inspect

import torch

from labs.moe_decode_blackwell_matrix.matrix_catalog import load_playbook
from labs.moe_decode_blackwell_matrix.matrix_types import MatrixScenario
from labs.moe_decode_blackwell_matrix.runner import (
    _compare_outputs,
    _routing_stats,
    build_decode_batches,
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

    assert "torch.stack(" in source
    assert ".tolist()" in source
    assert ".sum().item()" not in source
    assert ".max().item()" not in source
    assert entropy > 0.0
    assert active == 1.0
    assert max_tokens == 3


def test_compare_outputs_batches_diff_materialization() -> None:
    source = inspect.getsource(_compare_outputs)

    assert "diff_tensors: list[torch.Tensor]" in source
    assert "torch.stack(diff_tensors).amax().tolist()" in source
    assert "diffs.append(float(" not in source
    assert "torch.max(torch.abs(out.float() - ref.float())).item()" not in source


def test_summary_builds_pairwise_sections() -> None:
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
