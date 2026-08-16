"""File-backed tests for artifact-derived optimization campaign records."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.optimization.campaign import CampaignConfig, ExperimentRecord, sha256_file
from core.optimization.campaign_evidence import (
    CampaignEvidenceError,
    derive_record_from_evidence_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest(commit: str, start: datetime, *, dirty: bool = False) -> dict[str, object]:
    return {
        "hardware": {
            "gpu_model": "NVIDIA B200",
            "cuda_version": "13.0",
            "driver_version": "580.0",
            "compute_capability": "10.0",
            "gpu_clock_mhz": 1800,
            "memory_clock_mhz": 3000,
            "gpu_app_clock_mhz": 1800,
            "memory_app_clock_mhz": 3000,
            "persistence_mode": True,
            "power_limit_w": 1000.0,
            "schemaVersion": "1.0",
        },
        "software": {
            "pytorch_version": "2.10.0",
            "triton_version": "3.6.0",
            "python_version": "3.12.1",
            "os": "Linux",
            "schemaVersion": "1.0",
        },
        "environment": {
            "cuda_visible_devices": "0",
            "relevant_env_vars": {},
            "schemaVersion": "1.0",
        },
        "git": {
            "commit": commit,
            "branch": "candidate",
            "dirty": dirty,
            "schemaVersion": "1.0",
        },
        "seeds": None,
        "verify": None,
        "collection_warnings": [],
        "runtime_capability_limitations": [],
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(seconds=2)).isoformat(),
        "duration_seconds": 2.0,
        "config": {
            "iterations": 20,
            "warmup_iterations": 5,
            "validity_profile": "strict",
            "profile_type": "minimal",
            "use_subprocess": True,
            "force_synchronize": True,
        },
        "schemaVersion": "1.0",
    }


def _benchmark_result(
    path: Path,
    *,
    commit: str,
    start: datetime,
    optimized_time_ms: float,
    verification_passed: bool = True,
    dirty: bool = False,
) -> None:
    target = "ch01:launch"
    example = "launch"
    optimized_file = "optimized_launch.py"
    status = "succeeded" if verification_passed else "failed_verification"
    baseline_manifest = _run_manifest(commit, start, dirty=dirty)
    optimized_manifest = _run_manifest(commit, start + timedelta(milliseconds=1), dirty=dirty)
    _write_json(
        path,
        {
            "timestamp": start.isoformat(),
            "results": [
                {
                    "chapter": "ch01",
                    "status": "completed",
                    "benchmarks": [
                        {
                            "example": example,
                            "type": "python",
                            "baseline_file": "baseline_launch.py",
                            "baseline_time_ms": 130.0,
                            "status": status,
                            "optimizations": [
                                {
                                    "file": optimized_file,
                                    "technique": "candidate",
                                    "status": status,
                                    "time_ms": optimized_time_ms,
                                    "input_verification": {
                                        "passed": verification_passed,
                                        "mismatches": [],
                                    },
                                    "verification": {
                                        "passed": verification_passed,
                                        "max_diff": 0.0,
                                    },
                                }
                            ],
                        }
                    ],
                    "manifests": [
                        {
                            "run_id": "baseline",
                            "timestamp": start.isoformat(),
                            "target_label": target,
                            "variant": "baseline",
                            "file": "baseline_launch.py",
                            "manifest": baseline_manifest,
                        },
                        {
                            "run_id": "optimized",
                            "timestamp": start.isoformat(),
                            "target_label": target,
                            "variant": "optimized",
                            "file": optimized_file,
                            "manifest": optimized_manifest,
                        },
                    ],
                    "summary": {
                        "total_benchmarks": 1,
                        "successful": 1 if verification_passed else 0,
                        "failed": 0 if verification_passed else 1,
                        "failed_verification": 0 if verification_passed else 1,
                    },
                }
            ],
        },
    )


def _draft() -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id="exp-derived",
        parent_id="a" * 40,
        beam="structural",
        hypothesis="Remove one repeated launch from the optimized path.",
        status="planned",
        changed_surface=["kernel.py"],
        primary_case="common",
        mechanism="The candidate removes one launch while keeping the workload fixed.",
        code_audit="The diff changes only the declared launch path.",
        outcome="Artifact-derived comparison.",
    )


def _make_bundle(
    root: Path,
    *,
    control_commit: str = "a" * 40,
    candidate_commit: str = "b" * 40,
    failed_candidate: bool = False,
) -> tuple[Path, Path, Path]:
    workload = root / "workload.yaml"
    workload.write_text("cases: [common, edge]\n", encoding="utf-8")
    environment = root / "environment.json"
    _write_json(environment, {"hardware": "B200", "validity_profile": "strict"})
    base_time = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
    cases: list[dict[str, object]] = []
    for case_index, (case_id, candidate_value) in enumerate((("common", 90.0), ("edge", 100.0))):
        pairs: list[dict[str, object]] = []
        for pair_index in range(1, 4):
            control_first = pair_index % 2 == 1
            pair_start = base_time + timedelta(minutes=case_index * 20 + pair_index * 2)
            control_start = pair_start if control_first else pair_start + timedelta(seconds=20)
            candidate_start = pair_start + timedelta(seconds=20) if control_first else pair_start
            control_path = root / f"{case_id}-{pair_index}-control.json"
            candidate_path = root / f"{case_id}-{pair_index}-candidate.json"
            _benchmark_result(
                control_path,
                commit=control_commit,
                start=control_start,
                optimized_time_ms=100.0,
            )
            _benchmark_result(
                candidate_path,
                commit=candidate_commit,
                start=candidate_start,
                optimized_time_ms=candidate_value,
                verification_passed=not (
                    failed_candidate and case_id == "common" and pair_index == 2
                ),
            )
            pairs.append(
                {
                    "sequence": pair_index,
                    "order": ["control", "candidate"]
                    if control_first
                    else ["candidate", "control"],
                    "control_result": {
                        "path": control_path.name,
                        "sha256": sha256_file(control_path),
                    },
                    "candidate_result": {
                        "path": candidate_path.name,
                        "sha256": sha256_file(candidate_path),
                    },
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "target_label": "ch01:launch",
                "example": "launch",
                "optimization_file": "optimized_launch.py",
                "pairs": pairs,
            }
        )
    bundle = root / "benchmark-evidence.json"
    _write_json(
        bundle,
        {
            "schema_version": "aisp.campaign-benchmark-evidence/v1",
            "experiment_id": "exp-derived",
            "metric": "latency_ms",
            "metric_source": "optimized_time_ms",
            "measurement_protocol": "paired_interleaved",
            "cases": cases,
            "profile_artifacts": [],
        },
    )
    return bundle, workload, environment


def _config(workload: Path, environment: Path, *, derived: bool = True) -> CampaignConfig:
    return CampaignConfig(
        objective="Reduce launch latency without regressing the edge case.",
        primary_metric="latency_ms",
        initial_control_commit="a" * 40,
        direction="lower",
        primary_cases=["common"],
        frozen_cases=["common", "edge"],
        min_trials=3,
        min_improvement_pct=2.0,
        max_case_regression_pct=0.5,
        max_cv_pct=5.0,
        require_derived_evidence=derived,
        workload_spec=str(workload),
        workload_sha256=sha256_file(workload),
        environment_spec=str(environment),
        environment_sha256=sha256_file(environment),
    )


def test_derives_measurements_correctness_and_manifest_provenance(tmp_path: Path) -> None:
    bundle, workload, environment = _make_bundle(tmp_path)

    record = derive_record_from_evidence_bundle(_draft(), _config(workload, environment), bundle)

    assert record.status == "completed"
    assert record.correctness == "passed"
    assert record.measurement_protocol == "interleaved"
    assert record.measurements["common"].control == [100.0, 100.0, 100.0]
    assert record.measurements["common"].candidate == [90.0, 90.0, 90.0]
    assert record.measurements["edge"].candidate == [100.0, 100.0, 100.0]
    assert record.provenance["derived_measurements"] is True
    assert record.provenance["derived_correctness"] is True
    assert record.provenance["control_run_commit"] == "a" * 40
    assert record.provenance["candidate_run_commit"] == "b" * 40
    assert bundle.name in record.raw_artifacts
    assert len(record.raw_artifacts) == 13
    assert all(len(digest) == 64 for digest in record.artifact_sha256.values())


def test_derives_failed_correctness_without_accepting_a_success_claim(tmp_path: Path) -> None:
    bundle, workload, environment = _make_bundle(tmp_path, failed_candidate=True)

    record = derive_record_from_evidence_bundle(_draft(), _config(workload, environment), bundle)

    assert record.correctness == "failed"
    assert record.measurements["common"].candidate[1] == 90.0


def test_rejects_tampered_results_and_same_revision_comparisons(tmp_path: Path) -> None:
    bundle, workload, environment = _make_bundle(tmp_path)
    tampered = tmp_path / "common-1-control.json"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(CampaignEvidenceError, match="hash does not match"):
        derive_record_from_evidence_bundle(_draft(), _config(workload, environment), bundle)

    second_root = tmp_path / "same-revision"
    second_root.mkdir()
    same_bundle, same_workload, same_environment = _make_bundle(
        second_root,
        control_commit="c" * 40,
        candidate_commit="c" * 40,
    )
    with pytest.raises(CampaignEvidenceError, match="same revision"):
        derive_record_from_evidence_bundle(
            _draft(), _config(same_workload, same_environment), same_bundle
        )


def test_record_evidence_cli_binds_results_to_git_and_passes_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "campaign@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Campaign Test"], cwd=repo, check=True)
    source = repo / "kernel.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "control"], cwd=repo, check=True)
    control_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "candidate"], cwd=repo, check=True)
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    bundle, workload, environment = _make_bundle(
        evidence_root,
        control_commit=control_commit,
        candidate_commit=candidate_commit,
    )
    draft_path = evidence_root / "experiment.json"
    draft_payload = _draft().to_dict()
    draft_payload["parent_id"] = control_commit
    _write_json(draft_path, draft_payload)
    workspace = tmp_path / "campaign"
    code_root = Path(__file__).parents[1]
    command_env = dict(os.environ)
    command_env["PYTHONPATH"] = str(code_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "core.optimization.campaign",
            "init",
            str(workspace),
            "--objective",
            "Reduce launch latency without regressing the edge case.",
            "--metric",
            "latency_ms",
            "--initial-control-commit",
            control_commit,
            "--primary-case",
            "common",
            "--frozen-case",
            "common",
            "--frozen-case",
            "edge",
            "--min-trials",
            "3",
            "--min-improvement-pct",
            "2",
            "--max-case-regression-pct",
            "0.5",
            "--require-derived-evidence",
            "--workload-spec",
            str(workload),
            "--environment-spec",
            str(environment),
        ],
        cwd=code_root,
        env=command_env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "core.optimization.campaign",
            "record-evidence",
            str(workspace),
            "--experiment",
            str(draft_path),
            "--evidence",
            str(bundle),
            "--repo",
            str(repo),
            "--control-revision",
            control_commit,
        ],
        cwd=code_root,
        env=command_env,
        check=True,
        capture_output=True,
        text=True,
    )
    gate = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.optimization.campaign",
            "gate",
            str(workspace),
            "exp-derived",
            "--json",
        ],
        cwd=code_root,
        env=command_env,
        check=True,
        capture_output=True,
        text=True,
    )

    decision = json.loads(gate.stdout)
    stored = json.loads((workspace / "experiments.jsonl").read_text(encoding="utf-8"))
    assert decision["decision"] == "promote"
    assert decision["improvement_pct"] == pytest.approx(10.0)
    assert stored["provenance"]["control_commit"] == control_commit
    assert stored["provenance"]["candidate_commit"] == candidate_commit
    assert stored["provenance"]["candidate_run_commit"] == candidate_commit
