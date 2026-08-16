"""Focused file-backed tests for opportunity queue execution evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from core.analysis.optimization_opportunities import (
    _render_job_plan_shell,
    normalize_candidates,
    rank_opportunities,
    summarize_run_queue_root,
)
from core.optimization.evidence_validation import validate_evidence_artifact


def _write_result(path: Path) -> str:
    manifest = {
        "schemaVersion": "1.0",
        "git": {"commit": "a" * 40, "dirty": False},
        "config": {"validity_profile": "strict"},
        "collection_warnings": [],
        "runtime_capability_limitations": [],
    }
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-16T00:00:00Z",
                "results": [
                    {
                        "chapter": "test",
                        "status": "completed",
                        "benchmarks": [
                            {
                                "example": "example",
                                "baseline_file": "baseline_example.py",
                                "baseline_time_ms": 10.0,
                                "status": "succeeded",
                                "optimizations": [
                                    {
                                        "file": "optimized_example.py",
                                        "status": "succeeded",
                                        "time_ms": 9.0,
                                        "input_verification": {"passed": True},
                                        "verification": {"passed": True},
                                    }
                                ],
                            }
                        ],
                        "manifests": [
                            {"variant": "baseline", "manifest": manifest},
                            {"variant": "optimized", "manifest": manifest},
                        ],
                        "summary": {
                            "total_benchmarks": 1,
                            "successful": 1,
                            "failed": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_repeat_manifest(root: Path, result_hash: str) -> Path:
    manifest_path = root / "repeat_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "aisp.queue-repeat-manifest/v1",
                "declared_repeat_count": 1,
                "executed_repeat_count": 1,
                "comparison_policy": "single_command",
                "runs": [
                    {
                        "sequence": 1,
                        "order": ["job"],
                        "roles": {
                            "job": {
                                "exit_code": 0,
                                "result": {
                                    "path": "result.json",
                                    "sha256": result_hash,
                                },
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_generated_candidate_commands_bind_distinct_worktrees() -> None:
    result = rank_opportunities(
        normalize_candidates(
            {
                "benchmarks": [],
                "target_catalog": ["labs/flexattention:flex_prefill"],
            }
        ),
        top_n=1,
    )
    control, candidate = result["run_queue"]["jobs"][:2]

    assert "AISP_CONTROL_CWD" in control["command"]
    assert "AISP_CANDIDATE_CWD" in candidate["command"]
    assert control["command"] != candidate["command"]
    assert candidate["paired_control_command"] == control["command"]
    assert candidate["comparison_policy"] == "paired_interleaved"
    assert candidate["repeat_count"] == 3
    assert candidate["require_distinct_source"] is True


def test_queue_records_failure_and_continues_independent_jobs(tmp_path: Path) -> None:
    queue = {
        "jobs": [
            {
                "id": "fails",
                "stage": "candidate",
                "target": "test:fails",
                "command": "false",
                "repeat_count": 2,
                "comparison_policy": "single_command",
                "result_artifact": "",
            },
            {
                "id": "continues",
                "stage": "control",
                "target": "test:continues",
                "command": "printf ok",
                "repeat_count": 2,
                "comparison_policy": "single_command",
                "result_artifact": "",
            },
        ]
    }
    queue_root = tmp_path / "queue"
    script = tmp_path / "run.sh"
    script.write_text(
        _render_job_plan_shell(
            queue,
            root_env_var="AISP_TEST_QUEUE_ROOT",
            default_root=str(queue_root),
            root_label="Test queue",
            empty_message="empty",
            completion_message="complete",
            review_title="Review",
        ),
        encoding="utf-8",
    )

    syntax = subprocess.run(["bash", "-n", str(script)], check=False)
    completed = subprocess.run(["bash", str(script)], cwd=tmp_path, check=False)

    assert syntax.returncode == 0
    assert completed.returncode == 1
    assert (queue_root / "fails" / "FAILED").is_file()
    assert (queue_root / "fails" / "EXIT_CODE").read_text().strip() == "1"
    assert (queue_root / "continues" / "DONE").is_file()
    manifest = json.loads((queue_root / "continues" / "repeat_run_manifest.json").read_text())
    assert manifest["executed_repeat_count"] == 2


def test_repeat_manifest_validates_result_content_and_hash(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_hash = _write_result(result_path)
    manifest_path = _write_repeat_manifest(tmp_path, result_hash)

    valid = validate_evidence_artifact(manifest_path, tmp_path)
    assert valid.valid is True

    result_path.write_text("{}", encoding="utf-8")
    tampered = validate_evidence_artifact(manifest_path, tmp_path)
    assert tampered.valid is False
    assert any("sha256 does not match" in error for error in tampered.errors)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["runs"][0]["roles"]["job"]["result"]["sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    malformed = validate_evidence_artifact(manifest_path, tmp_path)
    assert malformed.valid is False
    assert any("results must contain exactly one target" in error for error in malformed.errors)


def test_done_job_with_invalid_execution_evidence_is_incomplete(tmp_path: Path) -> None:
    job_dir = tmp_path / "queue" / "job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "job",
                "stage": "candidate",
                "command": "true",
                "repeat_count": 3,
                "comparison_policy": "paired_interleaved",
                "result_artifact": "benchmark_result.json",
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "DONE").write_text("2026-08-16T00:00:00Z\n", encoding="utf-8")
    (job_dir / "EXIT_CODE").write_text("0\n", encoding="utf-8")
    (job_dir / "repeat_run_manifest.json").write_text("{}\n", encoding="utf-8")

    summary = summarize_run_queue_root(tmp_path / "queue")

    assert summary["status_counts"] == {"failed_or_incomplete": 1}
    assert summary["jobs"][0]["completion_errors"]
