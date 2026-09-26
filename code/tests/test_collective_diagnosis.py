from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ch04.collective_diagnosis_tool import (
    ANALYSIS_SCHEMA,
    ARTIFACT_SCHEMA,
    CLASSIFICATION_DIAGNOSTIC,
    analyze_artifact,
    parse_message_size,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


def _round(
    rank: int,
    round_index: int,
    step_ms: float,
    cuda_ms: float,
    *,
    phase: str = "measured",
) -> dict[str, object]:
    ready_mono = 1_000_000_000 + round_index * 100_000_000 + rank * 1_000_000
    enqueue_mono = ready_mono + rank * 100_000
    completion_mono = ready_mono + int(step_ms * 1_000_000)
    ready_wall = 2_000_000_000 + round_index * 100_000_000 + rank * 4_000_000
    enqueue_wall = ready_wall + rank * 100_000
    completion_wall = ready_wall + int(step_ms * 1_000_000)
    return {
        "phase": phase,
        "round": round_index,
        "ready_mono_ns": ready_mono,
        "ready_wall_time_ns": ready_wall,
        "enqueue_mono_ns": enqueue_mono,
        "enqueue_wall_time_ns": enqueue_wall,
        "completion_mono_ns": completion_mono,
        "completion_wall_time_ns": completion_wall,
        "step_completion_mono_ns": completion_mono,
        "step_completion_wall_time_ns": completion_wall,
        "readiness_to_enqueue_ms": (enqueue_mono - ready_mono) / 1_000_000,
        "enqueue_to_completion_ms": (completion_mono - enqueue_mono) / 1_000_000,
        "cuda_collective_ms": cuda_ms,
        "compute_cuda_ms": 4.0,
        "competing_cuda_ms": None,
        "step_ms": step_ms,
        "correct": True,
    }


def _experiment(rank: int, scenario: str, step_ms: float, cuda_ms: float) -> dict[str, object]:
    injection = {
        "healthy": {"type": "none", "ranks": []},
        "delayed_rank": {
            "type": "pre_enqueue_host_sleep",
            "ranks": [1],
            "configured_delay_ms": 20.0,
        },
        "competing_gpu_workload": {
            "type": "extra_matrix_multiply_stream",
            "ranks": [1],
            "iterations": 12,
        },
        "forced_dependency": {
            "type": "collective_stream_waits_for_compute_event",
            "ranks": "all",
            "compute_iterations": 4,
        },
    }[scenario]
    return {
        "collective": "all_reduce",
        "requested_message_size_bytes": 1024,
        "logical_payload_bytes_per_rank": 1024,
        "input_buffer_bytes_per_rank": 1024,
        "output_buffer_bytes_per_rank": 1024,
        "world_size": 2,
        "dtype": "float32",
        "scenario": scenario,
        "injection": injection,
        "comparison_key": {
            "collective": "all_reduce",
            "logical_payload_bytes_per_rank": 1024,
            "world_size": 2,
            "dtype": "float32",
        },
        "warmup_rounds": [_round(rank, 0, step_ms, cuda_ms, phase="warmup")],
        "rounds": [
            _round(rank, 0, step_ms, cuda_ms),
            _round(rank, 1, step_ms + 1.0, cuda_ms + 0.1),
        ],
        "correctness": {"passed": True, "checks": ["retained output check"]},
    }


def _artifact(*, synchronized: bool = False) -> dict[str, object]:
    scenarios = {
        "healthy": (10.0, 3.0),
        "delayed_rank": (30.0, 3.2),
        "competing_gpu_workload": (18.0, 6.0),
        "forced_dependency": (16.0, 3.1),
    }
    clock_sync = (
        {"synchronized": True, "max_error_ms": 0.25, "method": "PTP receipt"}
        if synchronized
        else None
    )
    return {
        "schema": ARTIFACT_SCHEMA,
        "status": "completed",
        "classification": CLASSIFICATION_DIAGNOSTIC,
        "run_id": "fixture-run",
        "config": {
            "rounds": 2,
            "warmups": 1,
            "collectives": ["all_reduce"],
            "message_sizes_bytes": [1024],
            "scenarios": list(scenarios),
            "dtype": "float32",
        },
        "validity": {
            "correctness_passed": True,
            "harness_gates": None,
            "classification": CLASSIFICATION_DIAGNOSTIC,
        },
        "provenance": {"clock_sync_evidence": clock_sync},
        "rank_results": [
            {
                "rank": rank,
                "local_rank": 0,
                "host_id": f"host-{rank}",
                "experiments": [
                    _experiment(rank, scenario, step_ms, cuda_ms)
                    for scenario, (step_ms, cuda_ms) in scenarios.items()
                ],
            }
            for rank in range(2)
        ],
    }


def _subprocess_environment(fake_module_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(fake_module_dir), str(CODE_ROOT)))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        environment.pop(key, None)
    return environment


def test_parse_message_size_supports_binary_and_decimal_units() -> None:
    assert parse_message_size("256KiB") == 256 * 1024
    assert parse_message_size("1.5MiB") == int(1.5 * 1024**2)
    assert parse_message_size("4MB") == 4_000_000
    with pytest.raises(Exception, match="positive"):
        parse_message_size("0")


def test_analyzer_compares_explicit_injections_without_claiming_fabric_cause() -> None:
    analysis = analyze_artifact(_artifact())

    assert analysis["schema"] == ANALYSIS_SCHEMA
    assert analysis["classification"] == CLASSIFICATION_DIAGNOSTIC
    assert analysis["canonical"] is False
    delayed = next(item for item in analysis["comparisons"] if item["scenario"] == "delayed_rank")
    assert delayed["comparable"] is True
    assert delayed["step_p50_ratio_vs_healthy"] == pytest.approx(30.5 / 10.5)
    assert delayed["injected_cause"]["type"] == "pre_enqueue_host_sleep"
    assert "do not identify a fabric" in delayed["causal_boundary"]
    assert all(group["correctness_passed"] for group in analysis["groups"])
    slowdown = next(
        signal
        for signal in analysis["signals"]
        if signal["metric"] == "collective.step_slowdown_ratio_vs_healthy"
        and signal["scenario"] == "delayed_rank"
        and signal["scope"]["host"] == "host-0"
    )
    assert slowdown["role"] == "symptom"
    assert slowdown["injection"]["type"] == "pre_enqueue_host_sleep"
    assert slowdown["clock_domain"] == "collector_wall_clock:host-0"


def test_analyzer_refuses_cross_host_enqueue_spread_without_sync_evidence() -> None:
    analysis = analyze_artifact(_artifact())

    healthy = next(group for group in analysis["groups"] if group["scenario"] == "healthy")
    assert healthy["arrival_timing"]["available"] is False
    assert healthy["arrival_timing"]["enqueue_spread_ms"] is None
    assert "synchronization evidence" in healthy["arrival_timing"]["reason"]


def test_analyzer_uses_cross_host_wall_clock_only_with_bounded_sync_receipt() -> None:
    artifact = _artifact(synchronized=True)
    without_workload = analyze_artifact(artifact)
    assert not any(row["metric"] == "collective.enqueue_spread_p95_ms" for row in without_workload["signals"])
    artifact["config"]["workload_id"] = "shared-workload"
    analysis = analyze_artifact(artifact)

    healthy = next(group for group in analysis["groups"] if group["scenario"] == "healthy")
    timing = healthy["arrival_timing"]
    assert timing["available"] is True
    assert timing["method"] == "cross_host_wall_clock_with_sync_evidence"
    assert timing["clock_uncertainty_ms"] == 0.25
    assert timing["enqueue_spread_ms"]["count"] == 2
    skew_signal = next(
        signal
        for signal in analysis["signals"]
        if signal["metric"] == "collective.enqueue_spread_p95_ms"
        and signal["scenario"] == "healthy"
    )
    assert skew_signal["clock_domain"] == "cross_host_synchronized_wall"
    assert "host" not in skew_signal["scope"]
    assert skew_signal["scope"]["workload_id"] == "shared-workload"


def test_collective_signal_can_join_a_selected_fabric_path_on_its_host() -> None:
    from core.analysis.cross_layer_diagnosis import artifact_signals, correlate

    analysis = analyze_artifact(_artifact())
    symptom = next(row for row in artifact_signals(analysis) if row["role"] == "symptom")
    assert "path" not in symptom["scope"]
    counter = {**symptom, "role": "counter", "semantics": "delta", "metric": "pause_frames",
               "value": 5, "unit": "frames", "scope": {**symptom["scope"], "path": "selected-ib-port"}}
    assert len(correlate([symptom, counter])["correlations"]) == 1


def test_analyzer_accepts_explicit_measured_clock_bound() -> None:
    analysis = analyze_artifact(_artifact(), max_clock_skew_ms=0.5)

    healthy = next(group for group in analysis["groups"] if group["scenario"] == "healthy")
    assert healthy["arrival_timing"]["available"] is True
    assert healthy["arrival_timing"]["clock_uncertainty_ms"] == 0.5


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda artifact: artifact.update(rank_results=[]),
            "rank_results must contain at least two ranks",
        ),
        (
            lambda artifact: artifact["rank_results"].pop(),
            "rank_results has 1 entries",
        ),
        (
            lambda artifact: artifact["rank_results"][0]["experiments"][0]["rounds"].pop(),
            "records, expected 2",
        ),
        (
            lambda artifact: artifact["rank_results"][0]["experiments"][0]["rounds"][0].update(
                step_ms=float("nan")
            ),
            "step_ms must be a finite",
        ),
        (
            lambda artifact: artifact["rank_results"][0]["experiments"][0]["rounds"][0].update(
                completion_mono_ns=1
            ),
            "monotonic markers are out of order",
        ),
    ],
)
def test_analyzer_rejects_incomplete_or_invalid_artifacts(mutate, expected_reason: str) -> None:
    artifact = deepcopy(_artifact())
    mutate(artifact)

    analysis = analyze_artifact(artifact)

    assert analysis["status"] == "invalid"
    assert analysis["groups"] == []
    assert analysis["signals"] == []
    assert any(expected_reason in reason for reason in analysis["rejection_reasons"])


def test_analyzer_rejects_failed_correctness_without_emitting_signals() -> None:
    artifact = deepcopy(_artifact())
    artifact["rank_results"][1]["experiments"][0]["correctness"]["passed"] = False

    analysis = analyze_artifact(artifact)

    assert analysis["status"] == "rejected"
    assert analysis["signals"] == []
    assert analysis["comparisons"] == []
    assert any("correctness failed" in reason for reason in analysis["rejection_reasons"])


def test_analyzer_rejects_group_missing_from_every_rank() -> None:
    artifact = deepcopy(_artifact())
    for rank_result in artifact["rank_results"]:
        rank_result["experiments"].pop()

    analysis = analyze_artifact(artifact)

    assert analysis["status"] == "invalid"
    assert any("does not match config sweep" in reason for reason in analysis["rejection_reasons"])


def test_analyzer_never_promotes_external_gate_receipts() -> None:
    artifact = _artifact()
    artifact["classification"] = "canonical"
    artifact["validity"]["classification"] = "harness_gated"
    artifact["validity"]["harness_gates"] = {
        "all_required_passed": True,
        "gates": [{"name": "unverified", "status": "passed"}],
    }

    analysis = analyze_artifact(artifact)

    assert analysis["status"] == "completed"
    assert analysis["classification"] == CLASSIFICATION_DIAGNOSTIC
    assert analysis["canonical"] is False


def test_help_and_analyze_subprocesses_do_not_import_torch(tmp_path: Path) -> None:
    fake_module_dir = tmp_path / "fake-module"
    fake_module_dir.mkdir()
    (fake_module_dir / "torch.py").write_text(
        "raise RuntimeError('torch import attempted')\n",
        encoding="utf-8",
    )
    environment = _subprocess_environment(fake_module_dir)

    help_result = subprocess.run(
        [sys.executable, "-m", "ch04.collective_diagnosis_tool", "--help"],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "collective stalls" in help_result.stdout
    assert "lost" in help_result.stdout

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    analyze_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ch04.collective_diagnosis_tool",
            "analyze",
            str(artifact_path),
        ],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert analyze_result.returncode == 0, analyze_result.stderr
    assert json.loads(analyze_result.stdout)["schema"] == ANALYSIS_SCHEMA


def test_direct_run_emits_structured_skip_without_importing_torch(tmp_path: Path) -> None:
    fake_module_dir = tmp_path / "fake-module"
    fake_module_dir.mkdir()
    (fake_module_dir / "torch.py").write_text(
        "raise RuntimeError('torch import attempted')\n",
        encoding="utf-8",
    )
    environment = _subprocess_environment(fake_module_dir)
    artifact_path = tmp_path / "skipped.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ch04.collective_diagnosis_tool",
            "run",
            "--output",
            str(artifact_path),
        ],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "skipped"
    assert payload["diagnostic"].startswith("SKIPPED:")
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == payload


def test_analyze_cli_returns_nonzero_for_incomplete_artifact(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["rank_results"] = []
    artifact_path = tmp_path / "incomplete.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ch04.collective_diagnosis_tool",
            "analyze",
            str(artifact_path),
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    analysis = json.loads(result.stdout)
    assert analysis["status"] == "invalid"
    assert analysis["signals"] == []
