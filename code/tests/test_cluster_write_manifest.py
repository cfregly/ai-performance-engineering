from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from cluster.scripts import write_manifest


def _write_json(path: Path, payload) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_manifest_payload_marks_cluster_run_succeeded(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-28_cluster_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [
            {"name": "bootstrap_nodes", "exit_code": 0},
            {"name": "manifest_refresh", "exit_code": 0},
        ],
    )
    _write_json(
        run_dir / "progress" / "run_progress.json",
        {
            "run_id": run_id,
            "current": {
                "timestamp": "2026-03-28T21:15:57.376221+00:00",
                "step": "complete",
                "step_detail": "completed 2/2 suite steps",
                "percent_complete": 100.0,
                "metrics": {
                    "status": "completed",
                    "completed_steps": 2,
                    "total_steps": 2,
                    "suite_steps_path": str(run_dir / "structured" / f"{run_id}_suite_steps.json"),
                },
            },
        },
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
        finalize=True,
    )

    assert payload["manifest_version"] == 2
    assert payload["status"] == "succeeded"
    assert payload["suite_status"] == "succeeded"
    assert payload["success"] is True
    assert payload["issues"] == []
    assert payload["progress"]["percent_complete"] == 100.0
    assert payload["suite_steps"]["failed_step_count"] == 0


def test_build_manifest_payload_marks_fabric_run_partial(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-28_fabric_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [
            {"name": "build_fabric_eval", "exit_code": 0},
            {"name": "manifest_refresh", "exit_code": 0},
        ],
    )
    _write_json(
        run_dir / "progress" / "run_progress.json",
        {
            "run_id": run_id,
            "current": {
                "timestamp": "2026-03-28T21:40:33.028257+00:00",
                "step": "complete",
                "step_detail": "completed 2/2 suite steps",
                "percent_complete": 100.0,
                "metrics": {
                    "status": "completed",
                    "completed_steps": 2,
                    "total_steps": 2,
                    "suite_steps_path": str(run_dir / "structured" / f"{run_id}_suite_steps.json"),
                },
            },
        },
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_fabric_scorecard.json",
        {
            "status": "partial",
            "completeness": "runtime_verified",
            "families": {
                "nvlink": {"completeness": "runtime_verified"},
                "infiniband": {"completeness": "not_present"},
            },
        },
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
        finalize=True,
    )

    assert payload["status"] == "partial"
    assert payload["suite_status"] == "partial"
    assert payload["success"] is True
    assert payload["completeness"] == "runtime_verified"
    assert "fabric completeness is partial for one or more families" in payload["issues"]
    assert payload["fabric"]["scorecard_status"] == "partial"
    assert payload["fabric"]["degraded_families"] == ["infiniband"]


def test_build_manifest_payload_classifies_multinode_vllm_artifacts(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-29_2node_demo"
    run_dir = cluster_root / "runs" / run_id

    _write_json(run_dir / "structured" / f"{run_id}_preflight_services.json", {"status": "ok"})
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [{"name": "manifest_refresh", "exit_code": 0}],
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_leader_vllm_multinode_serve.json", {"status": "ok"}
    )
    (run_dir / "structured" / f"{run_id}_leader_vllm_multinode_serve.csv").write_text(
        "metric,value\n", encoding="utf-8"
    )
    (run_dir / "structured" / f"{run_id}_leader_vllm_multinode_serve.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_leader_vllm_multinode_slo_goodput.json",
        {"status": "ok"},
    )
    (run_dir / "structured" / f"{run_id}_leader_vllm_multinode_slo_goodput.csv").write_text(
        "metric,value\n", encoding="utf-8"
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_leader_vllm_multinode_leader_clock_lock.json",
        {"locked": True},
    )
    _write_json(
        run_dir / "structured" / f"{run_id}_worker_vllm_multinode_worker_clock_lock.json",
        {"locked": True},
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["leader", "worker"],
        labels=["leader", "worker"],
    )

    artifact_roles = payload["artifact_roles"]
    assert artifact_roles["preflight_services"] == [f"structured/{run_id}_preflight_services.json"]
    assert sorted(artifact_roles["vllm_multinode_serve"]) == [
        f"structured/{run_id}_leader_vllm_multinode_serve.csv",
        f"structured/{run_id}_leader_vllm_multinode_serve.json",
        f"structured/{run_id}_leader_vllm_multinode_serve.jsonl",
    ]
    assert sorted(artifact_roles["vllm_multinode_slo_goodput"]) == [
        f"structured/{run_id}_leader_vllm_multinode_slo_goodput.csv",
        f"structured/{run_id}_leader_vllm_multinode_slo_goodput.json",
    ]
    assert sorted(artifact_roles["vllm_multinode_clock_lock"]) == [
        f"structured/{run_id}_leader_vllm_multinode_leader_clock_lock.json",
        f"structured/{run_id}_worker_vllm_multinode_worker_clock_lock.json",
    ]


def test_build_manifest_payload_hashes_every_listed_file(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "hash_contract_demo"
    run_dir = cluster_root / "runs" / run_id
    structured_path = run_dir / "structured" / "result.json"
    raw_path = run_dir / "raw" / "metrics.csv"
    structured_path.parent.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    structured_path.write_bytes(b'{"status":"ok"}\n')
    raw_path.write_bytes(b"metric,value\nlatency,1\n")

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
    )

    assert payload["files"] == ["raw/metrics.csv", "structured/result.json"]
    assert payload["summary"]["file_count"] == len(payload["files"])
    assert set(payload["summary"]["sha256"]) == set(payload["files"])
    assert payload["summary"]["sha256"] == {
        "raw/metrics.csv": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "structured/result.json": hashlib.sha256(structured_path.read_bytes()).hexdigest(),
    }


def test_build_manifest_payload_is_nonterminal_until_explicit_finalization(
    tmp_path: Path,
) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "nonterminal_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [{"name": "validate_required_artifacts", "exit_code": 0}],
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
    )

    assert payload["finalized"] is False
    assert payload["status"] == "running"
    assert payload["suite_status"] == "running"
    assert payload["success"] is None


def test_build_manifest_payload_keeps_readiness_only_nonterminal(
    tmp_path: Path,
) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "readiness_only_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_multinode_readiness.json",
        {"status": "ok"},
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
    )

    assert payload["finalized"] is False
    assert payload["status"] == "unknown"
    assert payload["suite_status"] == "unknown"
    assert payload["success"] is None


def test_build_manifest_payload_uses_latest_resume_attempt_per_step(
    tmp_path: Path,
) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "resume_demo"
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_suite_steps.json",
        [
            {"name": "benchmark", "exit_code": 1, "log_path": "first.log"},
            {"name": "benchmark", "exit_code": 0, "log_path": "second.log"},
        ],
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=False,
        hosts=["localhost"],
        labels=["localhost"],
        finalize=True,
    )

    assert payload["status"] == "succeeded"
    assert payload["success"] is True
    assert payload["suite_steps"]["recorded_attempt_count"] == 2
    assert payload["suite_steps"]["step_count"] == 1
    assert payload["suite_steps"]["failed_step_count"] == 0


def test_build_manifest_payload_emits_actual_git_provenance() -> None:
    cluster_root = Path(__file__).resolve().parents[1] / "cluster"
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cluster_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    expected_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cluster_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    )

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id="actual_git_probe",
        run_dir=None,
        include_figures=False,
        hosts=[],
        labels=[],
    )

    assert payload["git"] == {
        "commit": expected_commit,
        "dirty": expected_dirty,
        "probe_ok": True,
        "errors": [],
    }


def test_git_probe_failure_is_observable(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    def _probe(command, **kwargs):  # type: ignore[no-untyped-def]
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n", stderr="")
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="status failed")

    monkeypatch.setattr(write_manifest.subprocess, "run", _probe)

    payload, _ = write_manifest.build_manifest_payload(
        cluster_root=tmp_path,
        run_id="failed_git_probe",
        run_dir=None,
        include_figures=False,
        hosts=[],
        labels=[],
    )

    assert payload["git"]["commit"] == "a" * 40
    assert payload["git"]["dirty"] is None
    assert payload["git"]["probe_ok"] is False
    assert payload["git"]["errors"] == ["git status exited with code 2"]


def test_cluster_shell_final_manifest_write_follows_all_recorded_steps() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "run_cluster_eval_suite.sh"
    )
    script = script_path.read_text(encoding="utf-8")
    preliminary_index = script.index(
        'run_step "manifest_refresh" python3 "${ROOT_DIR}/scripts/write_manifest.py"'
    )
    report_index = script.index('run_step "render_localhost_field_report_package"')
    final_index = script.index('final_manifest_args=("${manifest_args[@]}" --finalize)')
    exit_index = script.index('if [[ "$fail" -ne 0 ]]', final_index)
    readiness_block = script[
        script.index('if [[ "$MULTINODE_READINESS_CHECK_ONLY" -eq 1 ]]') : script.index("fail=0")
    ]

    assert preliminary_index < report_index < final_index < exit_index
    assert script.rfind("run_step ", 0, final_index) == report_index
    assert (
        'python3 "${ROOT_DIR}/scripts/write_manifest.py" "${final_manifest_args[@]}"'
        in script[final_index:exit_index]
    )
    assert "--finalize" not in readiness_block
