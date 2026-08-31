"""Focused host-side regressions for Wave 2 findings W2-133 through W2-141."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.engine as core_engine
import mcp.mcp_server as mcp_server
from labs.real_world_models.gpt4_architecture_optimization import (
    GPT4ArchitectureOptimization,
)
from labs.software_pipelining.pipeline_graph import (
    get_pipeline_example,
    validate_schedule,
)
from scripts.full_virtualized_rerun import (
    _enqueue_targets,
    _parse_args,
    _resolve_start_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"


def test_w2_133_outlier_corrupted_llama_expectation_is_retired() -> None:
    path = CODE_ROOT / "labs/real_world_models/expectations_4x_gb200.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["hardware_key"] == "4x_gb200"
    assert payload["schema_version"] == 2
    assert payload["examples"] == {}
    assert "15.055089060057934" not in path.read_text(encoding="utf-8")


def test_w2_134_moe_memory_estimate_stores_every_expert() -> None:
    estimate = GPT4ArchitectureOptimization(
        batch_size=1,
        seq_length=1,
        use_moe=True,
        use_fp8=False,
    )
    hidden = estimate.HIDDEN_SIZE
    attention_params = hidden * hidden * 4
    ffn_params_per_expert = hidden * hidden * 4 * 3
    expected_params = (
        attention_params + ffn_params_per_expert * estimate.NUM_EXPERTS_PER_LAYER
    ) * estimate.NUM_LAYERS

    assert estimate.estimated_parameter_count == expected_params
    assert estimate.estimated_parameter_memory_gb == pytest.approx(
        expected_params * 2 / 1024**3
    )
    assert estimate.estimated_min_b200_gpus == math.ceil(
        estimate.estimated_total_memory_gb / 192
    )


def test_w2_135_recsys_readme_names_both_triton_kernels() -> None:
    readme = (CODE_ROOT / "labs/recsys_sequence_ranking/README.md").read_text(
        encoding="utf-8"
    )

    assert "fused Triton kernels for sequence pooling and candidate scoring" in readme
    assert "uses Triton only for candidate scoring" not in readme


def test_w2_136_flash_attention_anti_dependency_matches_ring_depth() -> None:
    example = get_pipeline_example("fa_like_inner_loop")
    anti_dependency = next(
        edge
        for edge in example.edges
        if edge.kind == "anti_dependency"
        and edge.src == "pv_mma"
        and edge.dst == "load_kv"
    )

    assert anti_dependency.distance == example.stage_count == 3
    assert validate_schedule(example).is_valid


def test_w2_138_scaling_prediction_has_no_fixed_7_5x_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(analyze=SimpleNamespace(scaling=lambda: {"status": "ok"}))
    monkeypatch.setattr(core_engine, "get_engine", lambda: engine)

    result = mcp_server.tool_predict_scaling(
        {"current_gpus": 1, "target_gpus": 16}
    )

    assert result["prediction"]["scaling_efficiency"] == pytest.approx(0.85)
    assert result["prediction"]["estimated_speedup"] == pytest.approx(13.6)
    assert result["prediction"]["estimated_speedup"] > 7.5


def test_w2_139_force_rerun_never_duplicates_pending_targets() -> None:
    state = {
        "pending_targets": ["ch01:performance"],
        "target_records": {"ch01:performance": {"return_code": 0}},
        "discovered_targets": ["ch01:performance"],
        "active_target": "ch01:performance",
    }

    assert _enqueue_targets(
        state,
        ["ch01:performance", "ch01:performance"],
        force_rerun=True,
    ) == 0
    assert state["pending_targets"] == ["ch01:performance"]

    completed = {
        "pending_targets": [],
        "target_records": {"ch01:performance": {"return_code": 0}},
        "discovered_targets": ["ch01:performance"],
        "active_target": None,
    }
    assert _enqueue_targets(
        completed,
        ["ch01:performance", "ch01:performance"],
        force_rerun=True,
    ) == 1
    assert completed["pending_targets"] == ["ch01:performance"]


def test_w2_140_existing_queue_preserves_profile_when_flag_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["full_virtualized_rerun.py", "start", "--target", "ch01:performance"],
    )

    args = _parse_args()

    assert args.profile is None
    assert _resolve_start_profile("deep_dive", args.profile) == "deep_dive"
    assert _resolve_start_profile("deep_dive", "minimal") == "minimal"
    assert _resolve_start_profile(None, None) == "none"


def test_w2_141_nvl72_bandwidth_is_not_described_as_bisection() -> None:
    appendix = (REPO_ROOT / "docs/appendix.md").read_text(encoding="utf-8")
    paragraph = next(
        line for line in appendix.splitlines() if "130 TB/s" in line
    )

    assert "aggregate NVLink bandwidth" in paragraph
    assert "does not identify it as bisection bandwidth" in paragraph
    assert "130 TB/s total bisection bandwidth" not in paragraph
