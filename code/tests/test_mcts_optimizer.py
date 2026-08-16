"""CPU-only tests for the explicitly simulated MCTS search."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.optimization.search.mcts_optimizer import MCTSOptimizer, OptimizationState


def _hardware(**overrides: object) -> dict[str, object]:
    hardware: dict[str, object] = {
        "num_gpus": 1,
        "gpu_memory_gb": 192,
        "gpu_arch": "b200",
        "has_nvlink": True,
    }
    hardware.update(overrides)
    return hardware


def _model() -> dict[str, object]:
    return {"parameters_billions": 1, "model_id": "test-model"}


def test_mcts_requires_explicit_simulation_opt_in(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="simulation-only"):
        MCTSOptimizer(
            _hardware(),
            _model(),
            knowledge_base_path=tmp_path / "knowledge.json",
        )


def test_external_scalar_evaluator_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="campaign executor"):
        MCTSOptimizer(
            _hardware(),
            _model(),
            evaluator=lambda _state: 1.0,
            simulation=True,
            knowledge_base_path=tmp_path / "knowledge.json",
        )


def test_simulation_result_cannot_be_read_as_measurement(tmp_path: Path) -> None:
    optimizer = MCTSOptimizer(
        _hardware(),
        _model(),
        simulation=True,
        knowledge_base_path=tmp_path / "knowledge.json",
    )

    result = optimizer.search(budget=3, verbose=False)

    assert result["evaluation_mode"] == "simulation"
    assert result["performance_claim_allowed"] is False
    assert "estimated_speedup" not in result
    assert "simulated_throughput_delta_pct" in result
    assert 0.0 <= result["best_score"] <= 1.0


def test_cache_identity_includes_hardware_model_goal_and_context(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "knowledge.json"
    state = OptimizationState(applied_actions=["torch_compile"])
    first = MCTSOptimizer(
        _hardware(),
        _model(),
        simulation=True,
        evaluation_context={
            "source_revision": "control-sha",
            "workload_spec_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
        },
        knowledge_base_path=knowledge_path,
    )
    first._evaluate(state, "throughput")
    first._save_knowledge_base()

    same = MCTSOptimizer(
        _hardware(),
        _model(),
        simulation=True,
        evaluation_context={
            "source_revision": "control-sha",
            "workload_spec_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
        },
        knowledge_base_path=knowledge_path,
    )
    same._evaluate(state, "throughput")

    different_hardware = MCTSOptimizer(
        _hardware(gpu_arch="hopper"),
        _model(),
        simulation=True,
        evaluation_context={
            "source_revision": "control-sha",
            "workload_spec_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
        },
        knowledge_base_path=knowledge_path,
    )
    different_hardware._evaluate(state, "throughput")

    assert same.cache_hits == 1
    assert same.total_evaluations == 0
    assert different_hardware.cache_hits == 0
    assert different_hardware.total_evaluations == 1


def test_corrupt_knowledge_base_fails_closed(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text("not json")

    with pytest.raises(ValueError, match="Invalid MCTS knowledge base"):
        MCTSOptimizer(
            _hardware(),
            _model(),
            simulation=True,
            knowledge_base_path=knowledge_path,
        )


def test_cli_requires_simulation_and_json_is_machine_readable(tmp_path: Path) -> None:
    code_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(code_root)
    environment["HOME"] = str(tmp_path)
    base_command = [
        sys.executable,
        "-m",
        "core.optimization.search",
        "mcts",
        "--model-size",
        "1",
        "--num-gpus",
        "1",
        "--gpu-memory",
        "192",
        "--gpu-arch",
        "b200",
        "--budget",
        "1",
        "--json",
    ]

    rejected = subprocess.run(
        base_command,
        cwd=code_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    accepted = subprocess.run(
        [*base_command, "--simulate"],
        cwd=code_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 2
    assert "simulation-only" in rejected.stderr
    assert accepted.returncode == 0
    payload = json.loads(accepted.stdout)
    assert payload["evaluation_mode"] == "simulation"
    assert payload["performance_claim_allowed"] is False
