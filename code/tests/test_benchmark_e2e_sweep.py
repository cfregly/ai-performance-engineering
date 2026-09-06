from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import mcp.mcp_server as mcp_server
from core.api import handlers
from core.benchmark import bench_commands, e2e_sweep

TEST_GIT_COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def _clean_git_provenance(monkeypatch) -> None:
    git = {"commit": TEST_GIT_COMMIT, "branch": "test", "dirty": False}
    monkeypatch.setattr(e2e_sweep, "get_git_info", lambda: dict(git))
    real_run = subprocess.run

    def _clean_status_probe(command, *args, **kwargs):
        if list(command) == ["git", "status", "--porcelain", "--untracked-files=normal"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(e2e_sweep.subprocess, "run", _clean_status_probe)


def _resume_provenance(tmp_path: Path, *, gpu_count: int = 0) -> dict[str, object]:
    return {
        "git": e2e_sweep.get_git_info(),
        "expectation_hardware_key": "test_gpu",
        "execution_environment": {
            "kind": "bare_metal",
            "virtualized": False,
            "dmi_product_name": "test-box",
        },
        "gpu_count": gpu_count,
        "bench_root_identity": e2e_sweep._bench_root_identity(tmp_path),
    }


def _resume_contract(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_tier1": True,
        "run_full_sweep": False,
        "run_cluster": True,
        "run_fabric": True,
        "cluster_preset": "common-answer-fast",
        "hosts": ["localhost"],
        "labels": ["localhost"],
        "ssh_user": getpass.getuser(),
        "ssh_key": None,
        "oob_if": None,
        "socket_ifname": None,
        "nccl_ib_hca": None,
        "nmx_url": None,
        "nmx_token": None,
        "ib_mgmt_host": None,
        "ib_mgmt_user": None,
        "ib_mgmt_ssh_key": None,
        "cumulus_hosts": [],
        "cumulus_user": None,
        "cumulus_ssh_key": None,
        "primary_label": None,
        "coverage_baseline_run_id": None,
        "extra_cluster_args": ["--skip-render-localhost-report"],
        "bench_root": str(tmp_path),
        "profile_type": "minimal",
        "output_format": "both",
        "suite_timeout": 14400,
        "full_sweep_suite_timeout": 0,
        "timeout_multiplier": 3.0,
        "timeout_seconds": None,
        "validity_profile": "strict",
        "allow_portable_expectations_update": False,
        "reproducible": False,
        "cold_start": False,
        "force_synchronize": False,
        "iterations": None,
        "warmup": None,
        "gpu_sm_clock_mhz": None,
        "gpu_mem_clock_mhz": None,
        "artifacts_dir": None,
        "log_level": "INFO",
        "log_file": None,
        "single_gpu": False,
        "accept_regressions": False,
        "update_expectations": False,
        "allow_mixed_provenance": False,
        "ncu_metric_set": "minimal",
        "ncu_replay_mode": None,
        "nsys_timeout_seconds": None,
        "ncu_timeout_seconds": None,
        "auto_resume": True,
        "max_auto_resumes": 3,
        "watch_poll_interval_seconds": 15,
    }
    values.update(overrides)
    return e2e_sweep._build_e2e_contract(**values)


def _write_cluster_manifest(path: Path, *, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "run_id": run_id,
                "finalized": True,
                "status": "succeeded",
                "suite_status": "succeeded",
                "success": True,
                "git": {"commit": TEST_GIT_COMMIT, "dirty": False},
                "files": [],
                "summary": {"file_count": 0, "artifact_counts": {}, "sha256": {}},
            }
        ),
        encoding="utf-8",
    )


def _write_cluster_manifest_with_file(
    path: Path,
    *,
    run_id: str,
    relative_path: str = "reports/result.json",
    content: bytes = b"{}",
) -> Path:
    run_dir = path.parent
    artifact_path = run_dir / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(content)
    path.write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "run_id": run_id,
                "finalized": True,
                "status": "succeeded",
                "suite_status": "succeeded",
                "success": True,
                "git": {"commit": TEST_GIT_COMMIT, "dirty": False},
                "files": [relative_path],
                "summary": {
                    "file_count": 1,
                    "artifact_counts": {},
                    "sha256": {relative_path: hashlib.sha256(content).hexdigest()},
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _write_fabric_scorecard(path: Path, *, run_id: str, **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "ok",
        "families": {"nvlink": {"completeness": "runtime_verified"}},
        "summary": {"runtime_verified_families": 1},
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_benchmark_attempt_artifacts(
    *,
    repo_root: Path,
    artifacts_dir: str | None,
    run_id: str,
    targets: list[str],
    results: list[dict[str, object]],
) -> Path:
    paths = e2e_sweep._benchmark_run_event_paths(
        run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["events"].write_text(
        json.dumps({"event_type": "run_start", "targets": targets}) + "\n",
        encoding="utf-8",
    )
    paths["output_json"].write_text(
        json.dumps({"run_id": run_id, "results": results}),
        encoding="utf-8",
    )
    _write_benchmark_manifest(paths["run_dir"] / "manifest.json", run_id=run_id)
    return paths["output_json"]


def _write_benchmark_manifest(path: Path, *, run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "git": {"commit": TEST_GIT_COMMIT, "dirty": False},
                "manifests": [
                    {
                        "run_id": "ch01_example",
                        "manifest": {"git": {"commit": TEST_GIT_COMMIT, "dirty": False}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _patch_minimal_e2e_runtime(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(
            kind="bare_metal",
            virtualized=False,
            dmi_product_name="test-box",
        ),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_benchmark_queue_lock",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )


def test_discover_benchmark_e2e_inventory_buckets_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        e2e_sweep,
        "_iter_discovered_targets",
        lambda _root: [
            {
                "target": "ch01:demo",
                "chapter": "ch01",
                "example": "demo",
                "bench_type": "python",
                "multi_gpu": False,
            },
            {
                "target": "ch02:dist",
                "chapter": "ch02",
                "example": "dist",
                "bench_type": "python",
                "multi_gpu": True,
            },
            {
                "target": "labs/foo:cuda_demo",
                "chapter": "labs/foo",
                "example": "cuda_demo",
                "bench_type": "cuda",
                "multi_gpu": False,
            },
        ],
    )

    payload = e2e_sweep.discover_benchmark_e2e_inventory()

    assert payload["counts"] == {"total": 3, "single_gpu": 2, "multi_gpu": 1}
    assert payload["single_gpu"] == ["ch01:demo", "labs/foo:cuda_demo"]
    assert payload["multi_gpu"] == ["ch02:dist"]


def test_benchmark_producer_binds_run_id_and_truthful_git_to_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir()
    producer_git = {"commit": TEST_GIT_COMMIT, "branch": "test", "dirty": True}
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(bench_commands, "get_git_info", lambda: dict(producer_git))
    monkeypatch.setattr(bench_commands, "get_gpu_state", lambda **kwargs: {})
    monkeypatch.setattr(
        bench_commands,
        "resolve_target_chapters",
        lambda targets, bench_root: ([chapter_dir], {"ch01": {"example"}}),
    )
    monkeypatch.setattr(
        bench_commands,
        "_preflight_target_coverage_and_assets",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        bench_commands,
        "test_chapter",
        lambda **kwargs: {
            "chapter": "ch01",
            "summary": {"successful": 1, "failed": 0, "total_skipped": 0},
            "manifests": [
                {
                    "run_id": "ch01_example",
                    "manifest": {"git": dict(producer_git)},
                }
            ],
        },
    )
    run_id = "producer_binding"

    result = bench_commands._execute_benchmarks(
        targets=["ch01:example"],
        bench_root=tmp_path,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id=run_id,
        suite_timeout=0,
        exit_on_failure=False,
    )

    output = json.loads(Path(result["output_json"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert output["run_id"] == run_id
    assert manifest["run_id"] == run_id
    assert manifest["git"] == producer_git


@pytest.mark.parametrize("run_id", ["../outside", "/tmp/outside", ".", "..", "bad/run"])
def test_e2e_run_paths_reject_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="Run id must start"):
        e2e_sweep.resolve_e2e_run_id(run_id, repo_root=tmp_path)
    with pytest.raises(ValueError, match="Run id must start"):
        e2e_sweep.e2e_run_dir(run_id, repo_root=tmp_path)


def _tier1_stage_result(
    tmp_path: Path,
    *,
    target_status: str = "succeeded",
    execution_failed: int = 0,
    regressions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    artifact_paths = {
        "summary_path": tmp_path / "summary.json",
        "regression_summary_path": tmp_path / "regression_summary.md",
        "regression_json_path": tmp_path / "regression_summary.json",
        "trend_snapshot_path": tmp_path / "trend_snapshot.json",
    }
    summary_payload = {
        "targets": [
            {
                "key": "example",
                "target": "ch01:example",
                "status": target_status,
            }
        ],
        "summary": {
            "target_count": 1,
            "succeeded": int(target_status == "succeeded"),
            "failed": int(target_status.startswith("failed")),
            "skipped": int(target_status.startswith("skipped")),
            "missing": int(target_status == "missing"),
        },
    }
    artifact_paths["summary_path"].write_text(json.dumps(summary_payload), encoding="utf-8")
    artifact_paths["regression_summary_path"].write_text(
        "# Tier-1 regression summary\n", encoding="utf-8"
    )
    artifact_paths["regression_json_path"].write_text(
        json.dumps({"regressions": regressions or [], "missing_targets": []}),
        encoding="utf-8",
    )
    artifact_paths["trend_snapshot_path"].write_text(
        json.dumps({"run_count": 1, "history": [], "evidence_history": []}),
        encoding="utf-8",
    )

    return {
        "run_accepted": True,
        "baseline_eligible": True,
        **{key: str(path) for key, path in artifact_paths.items()},
        "execution": {"total_failed": execution_failed},
        "summary": summary_payload,
        "comparison": {"regressions": regressions or [], "missing_targets": []},
    }


def _successful_tier1_invocation_result(
    *,
    run_id: str,
    repo_root: Path,
    suite_definitions: list[object],
    artifacts_dir: str | None = None,
) -> dict[str, object]:
    history_root = repo_root / "artifacts" / "history" / "tier1"
    run_history_root = history_root / run_id
    summary_path = run_history_root / "summary.json"
    regression_summary_path = run_history_root / "regression_summary.md"
    trend_snapshot_path = run_history_root / "trend_snapshot.json"
    run_history_root.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "run_id": run_id,
        "source_git_commit": TEST_GIT_COMMIT,
        "source_manifest_git_commit": TEST_GIT_COMMIT,
        "source_git_dirty": False,
        "targets": [{"target": "ch01:example", "status": "succeeded"}],
        "summary": {
            "target_count": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "missing": 0,
        },
    }
    regression_json_path = regression_summary_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    regression_summary_path.write_text("# Tier-1 regression summary\n", encoding="utf-8")
    regression_json_path.write_text(
        json.dumps(
            {
                "current_run_id": run_id,
                "regressions": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_snapshot_path.write_text(
        json.dumps({"run_count": 1, "history": [], "evidence_history": []}),
        encoding="utf-8",
    )
    output_json = _write_benchmark_attempt_artifacts(
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        targets=["ch01:example"],
        results=[
            {
                "chapter": "ch01",
                "benchmarks": [{"example": "example", "status": "succeeded"}],
            }
        ],
    )
    return {
        "run_accepted": True,
        "baseline_eligible": True,
        "execution": {
            "run_id": run_id,
            "output_json": str(output_json),
            "total_failed": 0,
        },
        "summary": summary_payload,
        "comparison": {
            "current_run_id": run_id,
            "regressions": [],
            "missing_targets": [],
        },
        "summary_path": str(summary_path),
        "regression_summary_path": str(regression_summary_path),
        "regression_json_path": str(regression_json_path),
        "trend_snapshot_path": str(trend_snapshot_path),
        "history_root": str(history_root),
        "suite_definitions": suite_definitions,
        "total_failed": 0,
        "total_skipped": 0,
    }


def test_full_sweep_completion_rejects_subset_attempt_for_multi_target_unit() -> None:
    attempts = [
        {
            "verified_targets": ["ch01:a"],
            "benchmark_summary": {"target_outcomes": [{"target": "ch01:a", "status": "succeeded"}]},
        }
    ]

    assert (
        e2e_sweep._completed_units_from_attempts(
            attempts,
            frozen_targets=["ch01:a", "ch01:b", "ch02:c"],
        )
        == []
    )


def test_full_sweep_stored_success_is_downgraded_without_exact_output(tmp_path: Path) -> None:
    stage = {
        "name": "full_sweep",
        "enabled": True,
        "status": "succeeded",
        "attempts": [
            {
                "bucket": "single_gpu",
                "status": "succeeded",
                "targets": ["ch01:a"],
                "completed_units": ["ch01"],
            }
        ],
    }

    e2e_sweep._revalidate_full_sweep_stage_from_frozen_plan(
        stage,
        {"single_gpu_targets": ["ch01:a"], "multi_gpu_targets": []},
        repo_root=tmp_path,
        artifacts_dir=None,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert stage["status"] == "aborted"
    assert "lacks exact frozen-target evidence" in " ".join(stage["issues"])


def test_resume_provenance_rejects_changed_git_or_hardware(tmp_path: Path) -> None:
    current = _resume_provenance(tmp_path, gpu_count=1)
    stored = json.loads(json.dumps(current))
    stored["git"]["commit"] = "b" * 40
    stored["gpu_count"] = 2

    error = e2e_sweep._validate_resume_provenance(current=current, stored=stored)

    assert error is not None
    assert "git.commit changed" in error
    assert "gpu_count changed" in error


@pytest.mark.parametrize(
    "failure_mode",
    [
        "missing_commit",
        "malformed_commit",
        "reported_dirty",
        "status_dirty",
        "status_nonzero",
        "status_timeout",
    ],
)
def test_fresh_evidence_run_fails_git_preflight_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    _patch_minimal_e2e_runtime(monkeypatch, tmp_path)
    stage_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: stage_calls.append(kwargs) or {},
    )
    git_payload: dict[str, object] = {
        "commit": TEST_GIT_COMMIT,
        "branch": "test",
        "dirty": False,
    }
    if failure_mode == "missing_commit":
        git_payload.pop("commit")
    elif failure_mode == "malformed_commit":
        git_payload["commit"] = "abc123"
    elif failure_mode == "reported_dirty":
        git_payload["dirty"] = True
    monkeypatch.setattr(e2e_sweep, "get_git_info", lambda: dict(git_payload))

    def _status_probe(command, *args, **kwargs):
        if failure_mode == "status_timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if failure_mode == "status_nonzero":
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="failed")
        if failure_mode == "status_dirty":
            return subprocess.CompletedProcess(command, 0, stdout=" M code/file.py\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(e2e_sweep.subprocess, "run", _status_probe)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id=f"e2e_git_preflight_{failure_mode}",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "Git" in result["error"]
    assert stage_calls == []


def test_clean_git_preflight_accepts_exact_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        e2e_sweep.subprocess,
        "run",
        lambda command, *args, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    commit, error = e2e_sweep._validated_clean_git_commit(
        repo_root=tmp_path,
        git_info={"commit": TEST_GIT_COMMIT, "dirty": False},
    )

    assert commit == TEST_GIT_COMMIT
    assert error is None


@pytest.mark.parametrize("change_mode", ["commit", "dirty"])
def test_e2e_finalization_rejects_mid_run_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_mode: str,
) -> None:
    _patch_minimal_e2e_runtime(monkeypatch, tmp_path)
    git_calls = 0

    def _git_info() -> dict[str, object]:
        nonlocal git_calls
        git_calls += 1
        commit = "b" * 40 if change_mode == "commit" and git_calls > 1 else TEST_GIT_COMMIT
        return {"commit": commit, "branch": "test", "dirty": False}

    monkeypatch.setattr(e2e_sweep, "get_git_info", _git_info)
    status_calls = 0

    def _status_probe(command, *args, **kwargs):
        nonlocal status_calls
        status_calls += 1
        stdout = " M code/file.py\n" if change_mode == "dirty" and status_calls > 1 else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(e2e_sweep.subprocess, "run", _status_probe)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: _successful_tier1_invocation_result(
            run_id=kwargs["run_id"],
            repo_root=tmp_path,
            suite_definitions=[],
        ),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id=f"e2e_mid_run_{change_mode}",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    assert result["success"] is False
    assert result["run_state"] == "aborted"
    assert "Git" in result["error"]


def test_e2e_contract_redacts_tokens_key_paths_and_bench_root(tmp_path: Path) -> None:
    contract = e2e_sweep._build_e2e_contract(
        bench_root=str(tmp_path / "private-checkout"),
        ssh_key="/private/keys/id_ed25519",
        nmx_token="top-secret-token",
        ib_mgmt_ssh_key="/private/keys/ib",
        cumulus_ssh_key="/private/keys/cumulus",
    )

    serialized = json.dumps(contract)
    assert "top-secret-token" not in serialized
    assert "/private/" not in serialized
    assert contract["ssh_key_configured"] is True
    assert contract["nmx_token_configured"] is True
    assert str(contract["bench_root_identity"]).startswith("sha256:")
    resume_command = e2e_sweep.build_benchmark_e2e_resume_command(
        "redacted_contract",
        contract=contract,
    )
    assert "--ssh-key" not in resume_command
    assert "--bench-root" not in resume_command


def test_persisted_e2e_artifacts_redact_runtime_secrets_and_custom_bench_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_root = tmp_path / "private-bench-root-sentinel"
    repo_root.mkdir()
    bench_root.mkdir()
    ssh_key = "/private/keys/ssh-key-sentinel"
    nmx_token = "nmx-token-sentinel"
    ib_key = "/private/keys/ib-key-sentinel"
    cumulus_key = "/private/keys/cumulus-key-sentinel"
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "generated_at": "2026-08-16T00:00:00Z",
            "bench_root_identity": e2e_sweep._bench_root_identity(bench_root),
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    def _cluster_result(**kwargs):
        run_id = str(kwargs["run_id"])
        cluster_run_dir = repo_root / "cluster" / "runs" / run_id
        manifest_path = cluster_run_dir / "manifest.json"
        _write_cluster_manifest(manifest_path, run_id=run_id)
        if run_id.endswith("__fabric"):
            _write_fabric_scorecard(
                cluster_run_dir / "structured" / f"{run_id}_fabric_scorecard.json",
                run_id=run_id,
            )
        return {
            "success": True,
            "run_id": run_id,
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "command": [
                "cluster-eval",
                "--ssh-key",
                ssh_key,
                "--nmx-token",
                nmx_token,
                "--ib-mgmt-ssh-key",
                ib_key,
                "--cumulus-ssh-key",
                cumulus_key,
                "--bench-root",
                str(bench_root),
            ],
            "nested": {
                "ssh_key": ssh_key,
                "token_echo": nmx_token,
                "bench_root": str(bench_root),
            },
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _cluster_result)
    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_fabric_eval", _cluster_result)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_redaction_sentinels",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=True,
        cluster_preset="common-answer-fast",
        hosts=["remote.example"],
        labels=["remote"],
        ssh_user="runner",
        ssh_key=ssh_key,
        nmx_token=nmx_token,
        ib_mgmt_ssh_key=ib_key,
        cumulus_ssh_key=cumulus_key,
        bench_root=bench_root,
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_redaction_sentinels", repo_root)
    persisted = {
        path.name: path.read_text(encoding="utf-8") for path in run_dir.iterdir() if path.is_file()
    }
    serialized_result = json.dumps(result)
    for sentinel in (ssh_key, nmx_token, ib_key, cumulus_key, str(bench_root)):
        assert sentinel not in serialized_result
        assert all(sentinel not in content for content in persisted.values())
    inventory = json.loads((run_dir / "target_inventory.json").read_text(encoding="utf-8"))
    assert "bench_root" not in inventory
    assert inventory["bench_root_identity"].startswith("sha256:")
    assert result["hosts"]["ssh_key_configured"] is True
    assert "ssh_key" not in result["hosts"]


def test_early_failure_persists_only_path_free_bench_root_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_root = tmp_path / "private-early-failure-root"
    repo_root.mkdir()
    bench_root.mkdir()
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "bench_root_identity": e2e_sweep._bench_root_identity(bench_root),
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_early_failure_identity",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
        hosts=["remote.example"],
        ssh_user="runner",
        ssh_key=None,
        bench_root=bench_root,
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_early_failure_identity", repo_root)
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in run_dir.iterdir() if path.is_file()
    )
    assert result["overall_status"] == "failed"
    assert str(bench_root) not in persisted_text
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["bench_root_identity"].startswith("sha256:")
    assert "bench_root" not in summary["provenance"]


def test_default_e2e_success_redacts_repo_root_from_all_run_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo-root-success-sentinel"
    repo_root.mkdir()
    _patch_minimal_e2e_runtime(monkeypatch, repo_root)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: _successful_tier1_invocation_result(
            run_id=kwargs["run_id"],
            repo_root=repo_root,
            suite_definitions=[],
        ),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_default_root_success",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_default_root_success", repo_root)
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in run_dir.iterdir() if path.is_file()
    )
    assert result["success"] is True
    assert str(repo_root) not in json.dumps(result)
    assert str(repo_root) not in persisted_text
    assert "<repo-root>" in json.dumps(result)
    assert "<repo-root>" in persisted_text


def test_default_e2e_failure_redacts_repo_root_from_all_run_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo-root-failure-sentinel"
    repo_root.mkdir()
    _patch_minimal_e2e_runtime(monkeypatch, repo_root)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_default_root_failure",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
        hosts=["remote.example"],
        labels=["one", "two"],
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_default_root_failure", repo_root)
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in run_dir.iterdir() if path.is_file()
    )
    assert result["success"] is False
    assert str(repo_root) not in json.dumps(result)
    assert str(repo_root) not in persisted_text


def test_custom_bench_root_resume_restores_locator_only_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    bench_root = tmp_path / "custom-bench-root-sentinel"
    repo_root.mkdir()
    bench_root.mkdir()
    _patch_minimal_e2e_runtime(monkeypatch, repo_root)
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 0)
    run_id = "e2e_custom_root_resume"
    run_dir = e2e_sweep.e2e_run_dir(run_id, repo_root)
    run_dir.mkdir(parents=True)
    contract = _resume_contract(
        bench_root,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-08-17T00:00:00Z",
                "provenance": _resume_provenance(bench_root),
                "contract": contract,
                "stages": [
                    {
                        "name": name,
                        "enabled": False,
                        "status": "skipped",
                        "run_id": f"{run_id}__{name}",
                        "attempts": [],
                    }
                    for name in ("tier1", "full_sweep", "cluster", "fabric")
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [],
                        "multi_gpu_targets": [],
                    },
                    "locator_probe": "<bench-root>/suite",
                },
            }
        ),
        encoding="utf-8",
    )
    restored_probe: dict[str, str] = {}
    real_restore = e2e_sweep._restore_persisted_path_locators

    def _capture_restore(value, **kwargs):
        restored = real_restore(value, **kwargs)
        restored_probe["value"] = restored["frozen_plan"]["locator_probe"]
        return restored

    monkeypatch.setattr(e2e_sweep, "_restore_persisted_path_locators", _capture_restore)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id=run_id,
        resume=True,
        dry_run=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        bench_root=bench_root,
    )

    assert result["overall_status"] == "dry_run"
    assert restored_probe["value"] == str(bench_root / "suite")
    assert result["frozen_plan"]["locator_probe"] == "<bench-root>/suite"
    assert str(bench_root) not in json.dumps(result)


def test_auto_resume_refuses_unreconstructible_credentials_and_custom_bench_root(
    tmp_path: Path,
) -> None:
    credential_contract = _resume_contract(
        tmp_path,
        run_cluster=True,
        run_fabric=False,
        ssh_key="/private/key",
    )
    custom_root_contract = _resume_contract(
        tmp_path,
        run_cluster=False,
        run_fabric=False,
        bench_root=str(tmp_path / "elsewhere"),
    )

    assert (
        e2e_sweep.build_benchmark_e2e_resume_command(
            "credential_resume", contract=credential_contract, repo_root=tmp_path
        )
        == []
    )
    assert "credentials are required" in str(
        e2e_sweep._auto_resume_reconstruction_error(
            credential_contract,
            repo_root=tmp_path,
        )
    )
    assert (
        e2e_sweep.build_benchmark_e2e_resume_command(
            "custom_root_resume", contract=custom_root_contract, repo_root=tmp_path
        )
        == []
    )
    assert "custom bench root" in str(
        e2e_sweep._auto_resume_reconstruction_error(
            custom_root_contract,
            repo_root=tmp_path,
        )
    )


@pytest.mark.parametrize(
    ("field_name", "stored_value", "requested_value"),
    [
        ("output_format", "both", "json"),
        ("extra_cluster_args", ["--skip-render-localhost-report"], ["--other"]),
        ("reproducible", False, True),
        ("cold_start", False, True),
        ("force_synchronize", False, True),
        ("oob_if", None, "eth0"),
        ("socket_ifname", None, "ib0"),
        ("nccl_ib_hca", None, "mlx5_0"),
        ("timeout_multiplier", 3.0, 4.0),
        ("timeout_seconds", 60, 90),
        ("full_sweep_suite_timeout", 0, 120),
        ("ncu_metric_set", "minimal", "full"),
        ("ncu_replay_mode", None, "kernel"),
        ("nsys_timeout_seconds", None, 30),
        ("ncu_timeout_seconds", None, 45),
    ],
)
def test_resume_contract_validates_safe_execution_options(
    tmp_path: Path,
    field_name: str,
    stored_value: object,
    requested_value: object,
) -> None:
    stored = _resume_contract(tmp_path, **{field_name: stored_value})
    requested = _resume_contract(tmp_path, **{field_name: requested_value})

    error = e2e_sweep._validate_resume_contract(requested=requested, stored=stored)

    assert error is not None
    assert field_name in error


@pytest.mark.parametrize("schema_version", [None, "0.9"])
def test_resume_contract_rejects_missing_or_legacy_schema(
    tmp_path: Path,
    schema_version: str | None,
) -> None:
    requested = _resume_contract(tmp_path)
    stored = _resume_contract(tmp_path)
    if schema_version is None:
        stored.pop("schema_version")
    else:
        stored["schema_version"] = schema_version

    error = e2e_sweep._validate_resume_contract(requested=requested, stored=stored)

    assert error is not None
    assert "stored schema_version" in error


def test_resume_provenance_refuses_dirty_stored_or_current_worktree(tmp_path: Path) -> None:
    clean = _resume_provenance(tmp_path)
    stored_dirty = json.loads(json.dumps(clean))
    stored_dirty["git"]["dirty"] = True
    current_dirty = json.loads(json.dumps(clean))
    current_dirty["git"]["dirty"] = True

    assert "stored worktree is dirty" in str(
        e2e_sweep._validate_resume_provenance(current=clean, stored=stored_dirty)
    )
    assert "current worktree is dirty" in str(
        e2e_sweep._validate_resume_provenance(current=current_dirty, stored=clean)
    )


@pytest.mark.parametrize("stage_name", ["tier1", "cluster", "fabric"])
def test_resume_downgrades_succeeded_stage_when_required_artifacts_are_missing(
    tmp_path: Path,
    stage_name: str,
) -> None:
    stages = [
        {
            "name": name,
            "enabled": name == stage_name,
            "status": "succeeded" if name == stage_name else "skipped",
            "attempts": [
                {
                    "run_id": f"run__{name}",
                    "status": "succeeded",
                    "result": {"success": True},
                }
            ]
            if name == stage_name
            else [],
        }
        for name in ("tier1", "full_sweep", "cluster", "fabric")
    ]

    e2e_sweep._revalidate_resumed_terminal_stages(
        stages,
        repo_root=tmp_path,
        artifacts_dir=None,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    stage = next(item for item in stages if item["name"] == stage_name)
    assert stage["status"] == "aborted"
    assert "failed resume revalidation" in " ".join(stage["issues"])


@pytest.mark.parametrize("stage_name", ["tier1", "cluster", "fabric"])
def test_resume_downgrades_succeeded_stage_when_evidence_is_corrupt(
    tmp_path: Path,
    stage_name: str,
) -> None:
    run_id = f"corrupt__{stage_name}"
    if stage_name == "tier1":
        result = _tier1_stage_result(tmp_path)
        Path(str(result["summary_path"])).write_text("{not-json", encoding="utf-8")
    else:
        run_dir = tmp_path / stage_name
        manifest_path = run_dir / "manifest.json"
        _write_cluster_manifest(manifest_path, run_id=run_id)
        result = {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
        }
        if stage_name == "cluster":
            manifest_path.write_text("{not-json", encoding="utf-8")
        else:
            scorecard_path = run_dir / "structured" / f"{run_id}_fabric_scorecard.json"
            scorecard_path.parent.mkdir(parents=True, exist_ok=True)
            scorecard_path.write_text("{not-json", encoding="utf-8")

    stages = [
        {
            "name": name,
            "enabled": name == stage_name,
            "status": "succeeded" if name == stage_name else "skipped",
            "attempts": [
                {
                    "run_id": run_id,
                    "status": "succeeded",
                    "result": result,
                }
            ]
            if name == stage_name
            else [],
        }
        for name in ("tier1", "full_sweep", "cluster", "fabric")
    ]

    e2e_sweep._revalidate_resumed_terminal_stages(
        stages,
        repo_root=tmp_path,
        artifacts_dir=None,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    stage = next(item for item in stages if item["name"] == stage_name)
    assert stage["status"] == "aborted"
    assert "failed resume revalidation" in " ".join(stage["issues"])


def _bound_tier1_status(
    result: dict[str, object],
    *,
    run_id: str,
    repo_root: Path,
    artifacts_dir: str | None = None,
) -> tuple[str, list[str], dict[str, object] | None]:
    return e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
        expected_run_id=run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        expected_git_commit=TEST_GIT_COMMIT,
    )


def test_tier1_evidence_binding_accepts_exact_attempt(tmp_path: Path) -> None:
    run_id = "tier1_exact_attempt"
    result = _successful_tier1_invocation_result(
        run_id=run_id,
        repo_root=tmp_path,
        suite_definitions=[],
    )

    status, issues, _ = _bound_tier1_status(
        result,
        run_id=run_id,
        repo_root=tmp_path,
    )

    assert status == "succeeded"
    assert issues == []


def test_tier1_evidence_binding_rejects_cross_run_artifacts(tmp_path: Path) -> None:
    result = _successful_tier1_invocation_result(
        run_id="tier1_attempt_b",
        repo_root=tmp_path,
        suite_definitions=[],
    )

    status, issues, _ = _bound_tier1_status(
        result,
        run_id="tier1_attempt_a",
        repo_root=tmp_path,
    )

    assert status == "failed"
    assert "run_id" in " ".join(issues)


@pytest.mark.parametrize(
    "tamper",
    [
        "execution_run_id",
        "summary_run_id",
        "comparison_run_id",
        "source_commit",
        "manifest_source_commit",
        "source_dirty",
        "outside_summary",
        "benchmark_manifest_run_id",
        "benchmark_manifest_commit",
        "benchmark_manifest_dirty",
        "benchmark_output_run_id",
    ],
)
def test_tier1_evidence_binding_rejects_tampered_identity_or_path(
    tmp_path: Path,
    tamper: str,
) -> None:
    run_id = f"tier1_tamper_{tamper}"
    result = _successful_tier1_invocation_result(
        run_id=run_id,
        repo_root=tmp_path,
        suite_definitions=[],
    )
    summary_path = Path(str(result["summary_path"]))
    comparison_path = Path(str(result["regression_json_path"]))
    benchmark_run_dir = tmp_path / "artifacts" / "runs" / run_id
    manifest_path = benchmark_run_dir / "manifest.json"
    output_path = benchmark_run_dir / "results" / "benchmark_test_results.json"

    if tamper == "execution_run_id":
        result["execution"]["run_id"] = "other_run"
    elif tamper == "summary_run_id":
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["run_id"] = "other_run"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "comparison_run_id":
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        payload["current_run_id"] = "other_run"
        comparison_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper in {"source_commit", "manifest_source_commit", "source_dirty"}:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if tamper == "source_commit":
            payload["source_git_commit"] = "b" * 40
        elif tamper == "manifest_source_commit":
            payload["source_manifest_git_commit"] = "b" * 40
        else:
            payload["source_git_dirty"] = True
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "outside_summary":
        outside_path = tmp_path / "outside_summary.json"
        outside_path.write_bytes(summary_path.read_bytes())
        result["summary_path"] = str(outside_path)
    elif tamper.startswith("benchmark_manifest_"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "benchmark_manifest_run_id":
            payload["run_id"] = "other_run"
        elif tamper == "benchmark_manifest_commit":
            payload["git"]["commit"] = "b" * 40
        else:
            payload["git"]["dirty"] = True
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "benchmark_output_run_id":
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["run_id"] = "other_run"
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = _bound_tier1_status(
        result,
        run_id=run_id,
        repo_root=tmp_path,
    )

    assert status == "failed"
    assert issues


def test_tier1_evidence_binding_rejects_empty_declared_targets(tmp_path: Path) -> None:
    result = _tier1_stage_result(tmp_path)
    summary_path = Path(str(result["summary_path"]))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["targets"] = []
    payload["summary"] = {
        "target_count": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
    )

    assert status == "failed"
    assert "targets must not be empty" in " ".join(issues)


@pytest.mark.parametrize("require_scorecard", [False, True])
def test_cluster_evidence_binding_rejects_cross_run_artifacts(
    tmp_path: Path,
    require_scorecard: bool,
) -> None:
    expected_run_id = "cluster_attempt_a"
    actual_run_id = "cluster_attempt_b"
    run_dir = tmp_path / "cluster" / "runs" / actual_run_id
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest_with_file(manifest_path, run_id=actual_run_id)
    if require_scorecard:
        _write_fabric_scorecard(
            run_dir / "structured" / f"{actual_run_id}_fabric_scorecard.json",
            run_id=actual_run_id,
        )
    result = {
        "success": True,
        "run_id": actual_run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
    }

    status, issues, _ = e2e_sweep._cluster_stage_status(
        result,
        require_scorecard=require_scorecard,
        expected_run_id=expected_run_id,
        repo_root=tmp_path,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert status == "failed"
    assert "run_id" in " ".join(issues)


@pytest.mark.parametrize(
    ("manifest_status", "success"),
    [
        ("running", None),
        ("unknown", None),
        ("failed", False),
        ("succeeded", None),
        ("succeeded", False),
    ],
)
def test_cluster_evidence_binding_rejects_nonterminal_or_unsuccessful_manifest(
    tmp_path: Path,
    manifest_status: str,
    success: bool | None,
) -> None:
    run_id = f"cluster_status_{manifest_status}_{success}"
    run_dir = tmp_path / "cluster" / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id=run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = manifest_status
    payload["suite_status"] = manifest_status
    payload["success"] = success
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = e2e_sweep._cluster_stage_status(
        {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
        },
        expected_run_id=run_id,
        repo_root=tmp_path,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert status == "failed"
    assert issues


def test_cluster_evidence_binding_requires_manifest_git_when_commit_is_expected(
    tmp_path: Path,
) -> None:
    run_id = "cluster_missing_git"
    run_dir = tmp_path / "cluster" / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id=run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("git")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = e2e_sweep._cluster_stage_status(
        {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
        },
        expected_run_id=run_id,
        repo_root=tmp_path,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert status == "failed"
    assert issues == ["cluster manifest is missing Git provenance"]


def test_cluster_evidence_binding_propagates_terminal_partial_manifest(
    tmp_path: Path,
) -> None:
    run_id = "cluster_partial_manifest"
    run_dir = tmp_path / "cluster" / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id=run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "partial"
    payload["suite_status"] = "partial"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = e2e_sweep._cluster_stage_status(
        {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
        },
        expected_run_id=run_id,
        repo_root=tmp_path,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert status == "partial"
    assert issues == ["cluster manifest reported partial completion"]


@pytest.mark.parametrize(
    "tamper",
    [
        "outside_run_dir",
        "outside_manifest",
        "outside_listed_file",
        "missing_file",
        "bad_digest",
        "bad_file_count",
    ],
)
def test_cluster_manifest_binding_rejects_invalid_file_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    run_id = f"cluster_tamper_{tamper}"
    run_dir = tmp_path / "cluster" / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest_with_file(manifest_path, run_id=run_id)
    result = {
        "success": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
    }
    if tamper == "outside_run_dir":
        outside_run_dir = tmp_path / "outside_cluster"
        outside_manifest = outside_run_dir / "manifest.json"
        _write_cluster_manifest_with_file(outside_manifest, run_id=run_id)
        result["run_dir"] = str(outside_run_dir)
        result["manifest_path"] = str(outside_manifest)
    elif tamper == "outside_manifest":
        outside_manifest = tmp_path / "outside_manifest.json"
        outside_manifest.write_bytes(manifest_path.read_bytes())
        result["manifest_path"] = str(outside_manifest)
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "outside_listed_file":
            outside_file = run_dir.parent / "outside.json"
            outside_file.write_text("{}", encoding="utf-8")
            payload["files"] = ["../outside.json"]
            payload["summary"]["sha256"] = {"../outside.json": hashlib.sha256(b"{}").hexdigest()}
        elif tamper == "missing_file":
            Path(run_dir / payload["files"][0]).unlink()
        elif tamper == "bad_digest":
            payload["summary"]["sha256"][payload["files"][0]] = "0" * 64
        elif tamper == "bad_file_count":
            payload["summary"]["file_count"] = 2
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status, issues, _ = e2e_sweep._cluster_stage_status(
        result,
        expected_run_id=run_id,
        repo_root=tmp_path,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert status == "failed"
    assert issues


@pytest.mark.parametrize(
    "tamper",
    ["run_id", "commit", "dirty", "output_run_id"],
)
def test_full_sweep_revalidation_rejects_tampered_bucket_manifest(
    tmp_path: Path,
    tamper: str,
) -> None:
    run_id = f"full_sweep_tamper_{tamper}"
    output_path = _write_benchmark_attempt_artifacts(
        repo_root=tmp_path,
        artifacts_dir=None,
        run_id=run_id,
        targets=["ch01:a"],
        results=[
            {
                "chapter": "ch01",
                "benchmarks": [{"example": "a", "status": "succeeded"}],
            }
        ],
    )
    manifest_path = output_path.parents[1] / "manifest.json"
    if tamper == "output_run_id":
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["run_id"] = "other_run"
        output_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "run_id":
            payload["run_id"] = "other_run"
        elif tamper == "commit":
            payload["git"]["commit"] = "b" * 40
        else:
            payload["git"]["dirty"] = True
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stage = {
        "name": "full_sweep",
        "enabled": True,
        "status": "succeeded",
        "attempts": [
            {
                "run_id": run_id,
                "bucket": "single_gpu",
                "status": "succeeded",
                "verified_targets": ["ch01:a"],
                "benchmark_summary": {
                    "target_outcomes": [{"target": "ch01:a", "status": "succeeded"}]
                },
            }
        ],
    }

    e2e_sweep._revalidate_full_sweep_stage_from_frozen_plan(
        stage,
        {"single_gpu_targets": ["ch01:a"], "multi_gpu_targets": []},
        repo_root=tmp_path,
        artifacts_dir=None,
        expected_git_commit=TEST_GIT_COMMIT,
    )

    assert stage["status"] == "aborted"
    assert "lacks exact frozen-target evidence" in " ".join(stage["issues"])


def test_benchmark_stage_status_fails_complete_tier1_with_comparison_regression(
    tmp_path: Path,
) -> None:
    result = _tier1_stage_result(
        tmp_path,
        regressions=[{"target": "ch01:example", "reason": "speedup"}],
    )

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
    )

    assert status == "failed"
    assert "1 benchmark comparison regression(s) detected" in issues


def test_benchmark_stage_status_allows_explicitly_accepted_tier1_comparison(
    tmp_path: Path,
) -> None:
    result = _tier1_stage_result(
        tmp_path,
        regressions=[{"target": "ch01:example", "reason": "speedup"}],
    )

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
        allow_comparison_regressions=True,
    )

    assert status == "succeeded"
    assert issues == []


def test_benchmark_stage_status_fails_complete_tier1_with_execution_failure(tmp_path: Path) -> None:
    result = _tier1_stage_result(tmp_path, execution_failed=1)

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
    )

    assert status == "failed"
    assert "benchmark execution reported 1 failure(s)" in issues


def test_benchmark_stage_status_requires_every_requested_target_outcome(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    output_path.write_text(json.dumps({"results": []}), encoding="utf-8")

    status, issues, details = e2e_sweep._benchmark_stage_status(
        {"output_json": str(output_path)},
        required_paths=["output_json"],
        required_targets=["ch01:demo"],
    )

    assert status == "failed"
    assert issues == ["missing terminal benchmark outcomes: ch01:demo"]
    assert details == {
        "status_counts": {},
        "failed_benchmarks": [],
        "skipped_benchmarks": [],
        "target_outcomes": [],
    }


def test_benchmark_stage_status_canonicalizes_lab_targets_and_rejects_non_success(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs_demo",
                        "benchmarks": [{"example": "kernel", "status": "unknown"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status, issues, details = e2e_sweep._benchmark_stage_status(
        {"output_json": str(output_path)},
        required_paths=["output_json"],
        required_targets=["labs/demo:kernel"],
    )

    assert status == "failed"
    assert issues == ["non-success terminal benchmark outcomes: labs/demo:kernel=unknown"]
    assert details["target_outcomes"] == [{"target": "labs/demo:kernel", "status": "unknown"}]


def test_benchmark_stage_status_rejects_duplicate_terminal_outcomes(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    output_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "ch01",
                        "benchmarks": [
                            {"example": "demo", "status": "succeeded"},
                            {"example": "demo", "status": "succeeded"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        {"output_json": str(output_path)},
        required_paths=["output_json"],
        required_targets=["ch01:demo"],
    )

    assert status == "failed"
    assert issues == ["unexpected terminal benchmark outcomes: ch01:demo"]


@pytest.mark.parametrize("target_status", ["skipped", "missing", "unknown"])
def test_benchmark_stage_status_fails_incomplete_tier1_target(
    tmp_path: Path,
    target_status: str,
) -> None:
    result = _tier1_stage_result(tmp_path, target_status=target_status)

    status, issues, _ = e2e_sweep._benchmark_stage_status(
        result,
        required_paths=[
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ],
        require_complete=True,
    )

    assert status == "failed"
    assert f"1 benchmark target(s) reported {target_status}" in issues


def test_run_benchmark_e2e_sweep_derives_stage_ids_and_skips_duplicate_fabric(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_001__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_001__cluster")

    class _Tier1SuiteDefinitionLike:
        pass

    cluster_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: _successful_tier1_invocation_result(
            run_id=kwargs["run_id"],
            repo_root=tmp_path,
            suite_definitions=[_Tier1SuiteDefinitionLike()],
            artifacts_dir=str(tmp_path / "artifacts"),
        ),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )

    def _fake_cluster_eval(**kwargs):
        cluster_calls.append(kwargs)
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_001",
        run_full_sweep=False,
        cluster_preset="fabric-systems",
        run_fabric=True,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    stages = {stage["name"]: stage for stage in result["stages"]}
    assert stages["tier1"]["run_id"] == "e2e_001__tier1"
    assert stages["cluster"]["run_id"] == "e2e_001__cluster"
    assert stages["fabric"]["run_id"] == "e2e_001__fabric"
    assert stages["fabric"]["status"] == "skipped_duplicate"
    assert result["overall_status"] == "succeeded"
    result_run_dir = e2e_sweep.e2e_run_dir("e2e_001", tmp_path)
    progress_path = e2e_sweep.e2e_progress_path(result_run_dir)
    assert progress_path.exists()
    assert "--skip-render-localhost-report" in cluster_calls[0]["extra_args"]
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["run_id"] == "e2e_001"
    assert progress_payload["current"]["metrics"]["run_state"] == "completed"
    assert progress_payload["current"]["metrics"]["overall_status"] == "succeeded"
    json.dumps(result)
    assert (result_run_dir / "manifest.json").exists()


def test_run_benchmark_e2e_sweep_detects_terminal_tier1_failure_from_nested_execution_output(
    tmp_path: Path, monkeypatch
) -> None:
    tier1_run_id = "e2e_tier1_nested_failure__tier1"
    history_root = tmp_path / "artifacts" / "history" / "tier1"
    run_history_root = history_root / tier1_run_id
    summary_path = run_history_root / "summary.json"
    regression_summary_path = run_history_root / "regression_summary.md"
    regression_json_path = regression_summary_path.with_suffix(".json")
    trend_snapshot_path = run_history_root / "trend_snapshot.json"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_id": tier1_run_id,
                "source_git_commit": TEST_GIT_COMMIT,
                "source_manifest_git_commit": TEST_GIT_COMMIT,
                "source_git_dirty": False,
                "suite_name": "tier1",
                "targets": [
                    {
                        "key": "block_scaling",
                        "target": "labs/block_scaling:block_scaling",
                        "status": "failed_no_speedup",
                        "best_speedup": 1.749,
                        "optimization_goal": "speed",
                    }
                ],
                "summary": {
                    "target_count": 1,
                    "failed": 1,
                    "skipped": 0,
                    "succeeded": 0,
                    "missing": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    regression_summary_path.write_text("# regressions\n", encoding="utf-8")
    regression_json_path.write_text(
        json.dumps(
            {
                "current_run_id": tier1_run_id,
                "regressions": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_snapshot_path.write_text(
        json.dumps({"run_count": 1, "history": [], "evidence_history": []}),
        encoding="utf-8",
    )
    output_json = _write_benchmark_attempt_artifacts(
        repo_root=tmp_path,
        artifacts_dir=None,
        run_id=tier1_run_id,
        targets=["labs/block_scaling:block_scaling"],
        results=[
            {
                "chapter": "labs_block_scaling",
                "benchmarks": [
                    {
                        "example": "block_scaling",
                        "status": "failed_no_speedup",
                        "best_speedup": 1.749,
                        "optimization_goal": "speed",
                        "minimum_required_speedup": 1.75,
                        "error": "Best speedup 1.749x below required 1.75x threshold for speed-goal benchmark",
                    }
                ],
            }
        ],
    )

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {
                "run_id": kwargs["run_id"],
                "output_json": str(output_json),
                "total_failed": 1,
                "total_skipped": 0,
            },
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "regression_json_path": str(regression_json_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_tier1_nested_failure",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    tier1_stage = next(stage for stage in result["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "failed"
    assert result["overall_status"] == "failed"
    assert tier1_stage["attempts"][-1]["status"] == "failed"
    assert tier1_stage["attempts"][-1]["benchmark_summary"]["failed_benchmarks"] == [
        {
            "target": "labs/block_scaling:block_scaling",
            "status": "failed_no_speedup",
            "error": "Best speedup 1.749x below required 1.75x threshold for speed-goal benchmark",
            "best_speedup": 1.749,
            "optimization_goal": "speed",
            "minimum_required_speedup": 1.75,
        }
    ]

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id="e2e_tier1_nested_failure", repo_root=tmp_path
    )
    aggregate_targets = {entry["target"] for entry in status["aggregate_failures"]}
    assert aggregate_targets == {"labs/block_scaling:block_scaling"}
    assert status["ledgers"]["summary"]["unresolved_count"] == 1


def test_run_benchmark_e2e_sweep_marks_partial_when_multi_gpu_bucket_is_skipped(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts_dir = str(tmp_path / "artifacts")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 1, "multi_gpu": 1},
            "single_gpu": ["ch01:demo"],
            "multi_gpu": ["ch02:dist"],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)

    def _fake_execute(**kwargs):
        output_json = _write_benchmark_attempt_artifacts(
            repo_root=tmp_path,
            artifacts_dir=artifacts_dir,
            run_id=kwargs["run_id"],
            targets=kwargs["targets"],
            results=[
                {
                    "chapter": "ch01",
                    "benchmarks": [{"example": "demo", "status": "succeeded"}],
                }
            ],
        )
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(output_json),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_partial",
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=artifacts_dir,
    )

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "partial"
    assert "multi-GPU bucket skipped" in " ".join(full_stage.get("issues", []))
    assert full_stage["result"]["buckets"]["multi_gpu"]["status"] == "skipped"
    assert result["overall_status"] == "partial"
    assert result["success"] is False


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_not_configured_scorecard(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    run_dir = tmp_path / "cluster" / "runs" / "e2e_fabric__fabric"
    structured_dir = run_dir / "structured"
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_fabric__fabric")
    scorecard_path = structured_dir / "e2e_fabric__fabric_fabric_scorecard.json"
    _write_fabric_scorecard(
        scorecard_path,
        run_id="e2e_fabric__fabric",
        families={
            "nvlink": {"completeness": "runtime_verified"},
            "infiniband": {"completeness": "not_configured"},
        },
        summary={"configured_management_planes": 0},
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert result["overall_status"] == "partial"
    assert result["success"] is False
    assert fabric_stage["artifacts"]["fabric_scorecard"]["degraded_families"] == [
        {"family": "infiniband", "completeness": "not_configured"}
    ]


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_partial_scorecard_status(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    run_dir = tmp_path / "cluster" / "runs" / "e2e_fabric_partial__fabric"
    structured_dir = run_dir / "structured"
    manifest_path = run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_fabric_partial__fabric")
    scorecard_path = structured_dir / "e2e_fabric_partial__fabric_fabric_scorecard.json"
    _write_fabric_scorecard(
        scorecard_path,
        run_id="e2e_fabric_partial__fabric",
        status="partial",
        families={
            "nvlink": {"completeness": "runtime_verified"},
            "spectrum-x": {"completeness": "not_present"},
        },
        summary={"runtime_verified_families": 1},
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_partial",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert "fabric completeness is partial" in " ".join(fabric_stage.get("issues", []))
    assert result["overall_status"] == "partial"
    assert result["success"] is False


def test_run_benchmark_e2e_sweep_mirrors_cluster_stage_progress(
    tmp_path: Path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_cluster_progress__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_cluster_progress__cluster")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_cluster_eval(**kwargs):
        child_progress_path = cluster_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(
            run_id=kwargs["run_id"], progress_path=child_progress_path
        )
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=2,
                total_phases=10,
                step="vllm_serve_sweep",
                step_detail="completed 2/10 suite steps",
                percent_complete=20.0,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_cluster_progress", tmp_path) / "progress.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_cluster_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "cluster:vllm_serve_sweep"
    assert payload["current"]["step_detail"] == "completed 2/10 suite steps"
    assert payload["current"]["percent_complete"] == 20.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_cluster_progress__cluster"


def test_run_benchmark_e2e_sweep_heartbeats_summary_during_stage_progress(
    tmp_path: Path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_cluster_heartbeat__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_cluster_heartbeat__cluster")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)
    monkeypatch.setattr(e2e_sweep, "_STATE_HEARTBEAT_SECONDS", 0.01)
    heartbeat_written = threading.Event()
    watch_heartbeat = threading.Event()
    summary_path = e2e_sweep.e2e_run_dir("e2e_cluster_heartbeat", tmp_path) / "summary.json"
    real_write_json = e2e_sweep._write_json

    def _observe_summary_write(path, payload):
        result = real_write_json(path, payload)
        if (
            watch_heartbeat.is_set()
            and Path(path) == summary_path
            and threading.current_thread().name.startswith("e2e-progress-")
        ):
            heartbeat_written.set()
        return result

    monkeypatch.setattr(e2e_sweep, "_write_json", _observe_summary_write)

    def _fake_cluster_eval(**kwargs):
        child_progress_path = cluster_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(
            run_id=kwargs["run_id"], progress_path=child_progress_path
        )
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=2,
                total_phases=10,
                step="vllm_serve_sweep",
                step_detail="completed 2/10 suite steps",
                percent_complete=20.0,
            )
        )
        # Wait for the real background write, rather than assuming the mirror
        # thread is scheduled within 50 ms or that filesystem mtimes advance.
        watch_heartbeat.set()
        observed["heartbeat_written"] = heartbeat_written.wait(timeout=2.0)
        observed["summary"] = json.loads(summary_path.read_text())
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_cluster_heartbeat",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
    )

    assert result["success"] is True
    assert result["overall_status"] == "succeeded"
    assert observed["heartbeat_written"] is True
    assert isinstance(observed["summary"], dict)


def test_run_benchmark_e2e_sweep_mirrors_fabric_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    fabric_run_dir = tmp_path / "cluster" / "runs" / "e2e_fabric_progress__fabric"
    manifest_path = fabric_run_dir / "manifest.json"
    _write_cluster_manifest(manifest_path, run_id="e2e_fabric_progress__fabric")
    _write_fabric_scorecard(
        fabric_run_dir / "structured" / "e2e_fabric_progress__fabric_fabric_scorecard.json",
        run_id="e2e_fabric_progress__fabric",
    )

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_fabric_eval(**kwargs):
        child_progress_path = fabric_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(
            run_id=kwargs["run_id"], progress_path=child_progress_path
        )
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=5,
                total_phases=12,
                step="build_fabric_eval",
                step_detail="completed 4/12 suite steps",
                percent_complete=33.3333,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_fabric_progress", tmp_path) / "progress.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(fabric_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-fabric"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_fabric_eval", _fake_fabric_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "fabric:build_fabric_eval"
    assert payload["current"]["step_detail"] == "completed 4/12 suite steps"
    assert payload["current"]["percent_complete"] == 33.3333
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_fabric_progress__fabric"


def test_emit_live_progress_includes_child_stage_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    recorder = e2e_sweep.ProgressRecorder(run_id="e2e_progress", progress_path=progress_path)
    stages = [
        {
            "name": "tier1",
            "enabled": True,
            "run_id": "e2e_progress__tier1",
            "status": "running",
            "attempts": [{"run_id": "e2e_progress__tier1", "status": "running"}],
        },
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": "e2e_progress__full_sweep",
            "status": "planned",
            "attempts": [],
        },
    ]

    e2e_sweep._emit_live_progress(
        recorder,
        stages=stages,
        run_state="running",
        overall_status="running",
        artifact_paths={"summary_path": tmp_path / "summary.json"},
        child_progress={
            "step": "ch04:gradient_fusion",
            "step_detail": "optimized timing (optimized_gradient_fusion)",
            "percent_complete": 50.0,
        },
        child_stage_name="tier1",
        child_run_id="e2e_progress__tier1",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["current"]["step"] == "tier1:ch04:gradient_fusion"
    assert payload["current"]["step_detail"] == "optimized timing (optimized_gradient_fusion)"
    assert payload["current"]["percent_complete"] == 25.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_progress__tier1"
    assert payload["current"]["metrics"]["child_progress"]["percent_complete"] == 50.0


def test_run_benchmark_e2e_sweep_writes_progress_and_checkpoint_before_stage_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, dict[str, object]] = {}
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )

    def _fake_tier1(**kwargs):
        run_dir = e2e_sweep.e2e_run_dir("e2e_checkpoint", tmp_path)
        observed["progress"] = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        observed["summary"] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        observed["checkpoint"] = json.loads(
            (run_dir / "checkpoint.json").read_text(encoding="utf-8")
        )
        return _successful_tier1_invocation_result(
            run_id=kwargs["run_id"],
            repo_root=tmp_path,
            suite_definitions=[],
        )

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _fake_tier1)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_checkpoint",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    assert observed["progress"]["run_id"] == "e2e_checkpoint"
    assert observed["progress"]["current"]["metrics"]["run_state"] == "running"
    assert observed["progress"]["current"]["metrics"]["overall_status"] == "running"
    assert observed["summary"]["run_state"] == "running"
    assert observed["summary"]["overall_status"] == "running"
    assert observed["summary"]["current_stage"] == "tier1"
    assert observed["summary"]["current_stage_run_id"] == "e2e_checkpoint__tier1"
    assert observed["checkpoint"]["run_id"] == "e2e_checkpoint"
    assert observed["checkpoint"]["run_state"] == "running"
    assert observed["checkpoint"]["overall_status"] == "running"
    assert observed["checkpoint"]["current_stage"] == "tier1"
    assert observed["checkpoint"]["current_stage_run_id"] == "e2e_checkpoint__tier1"
    assert observed["checkpoint"]["stages"][0]["name"] == "tier1"
    assert observed["checkpoint"]["stages"][0]["status"] == "running"


def test_run_benchmark_e2e_sweep_persists_aborted_state_on_unhandled_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tier1 exploded")),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "tier1 exploded" in result["error"]
    assert summary_payload["run_state"] == "aborted"
    assert summary_payload["overall_status"] == "aborted"
    assert summary_payload["resume_available"] is True
    assert checkpoint_payload["run_state"] == "aborted"
    assert checkpoint_payload["resume_available"] is True
    assert manifest_payload["checkpoint"]["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_persists_aborted_state_on_sighup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    if not hasattr(signal, "SIGHUP"):
        return

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )

    def _raise_hup(**_kwargs):
        signal.raise_signal(signal.SIGHUP)
        raise AssertionError("SIGHUP handler should have interrupted execution")

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _raise_hup)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_sighup_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_sighup_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "received SIGHUP" in result["error"]
    assert result["crash"]["signal"] == "SIGHUP"
    assert summary_payload["run_state"] == "aborted"
    assert checkpoint_payload["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_sigterm_abort_bypasses_inner_exception_handlers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )

    def _raise_term_inside_inner_exception_handler(**_kwargs):
        try:
            signal.raise_signal(signal.SIGTERM)
        except Exception as exc:  # pragma: no cover - regression guard
            return {"swallowed": str(exc)}
        raise AssertionError("SIGTERM handler should have aborted execution")

    monkeypatch.setattr(
        e2e_sweep, "_invoke_run_tier1_suite", _raise_term_inside_inner_exception_handler
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_sigterm_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_sigterm_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "received SIGTERM" in result["error"]
    assert result["crash"]["signal"] == "SIGTERM"
    assert summary_payload["run_state"] == "aborted"
    assert checkpoint_payload["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_resume_requires_explicit_run_id() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "requires an explicit run_id" in result["error"]


def test_run_benchmark_e2e_sweep_resume_rejects_contract_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_mismatch", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": _resume_contract(
                    tmp_path,
                    profile_type="none",
                    run_tier1=False,
                    run_full_sweep=False,
                    run_cluster=False,
                    run_fabric=False,
                ),
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "full_sweep", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_mismatch",
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        profile_type="minimal",
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert result["resume_available"] is True
    assert "Resume contract mismatch" in result["error"]
    assert "profile_type" in result["error"]


def test_run_benchmark_e2e_sweep_resume_allows_extending_suite_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_suite_timeout", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "run_state": "aborted",
                "overall_status": "aborted",
                "success": False,
                "resume_available": True,
                "provenance": _resume_provenance(tmp_path),
                "contract": _resume_contract(
                    tmp_path,
                    run_tier1=False,
                    run_full_sweep=False,
                    run_cluster=False,
                    run_fabric=False,
                    suite_timeout=14400,
                ),
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "full_sweep", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_suite_timeout",
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        profile_type="minimal",
        suite_timeout=0,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert result["run_state"] == "completed"
    assert result["contract"]["suite_timeout"] == 0


def test_run_benchmark_e2e_sweep_resume_marks_superseded_running_attempt_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    # Exercise only the in-process resume transition, independent of host GPUs or supervision.
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 0)
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: _successful_tier1_invocation_result(
            run_id=kwargs["run_id"],
            repo_root=tmp_path,
            suite_definitions=[],
        ),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_running", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "provenance": _resume_provenance(tmp_path),
                "contract": _resume_contract(
                    tmp_path,
                    run_tier1=True,
                    run_full_sweep=False,
                    run_cluster=False,
                    run_fabric=False,
                    auto_resume=False,
                ),
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_running__tier1",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_running__tier1",
                                "status": "running",
                            }
                        ],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_running__full_sweep",
                        "attempts": [],
                    },
                    {
                        "name": "cluster",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_running__cluster",
                        "attempts": [],
                    },
                    {
                        "name": "fabric",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_running__fabric",
                        "attempts": [],
                    },
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_running",
        resume=True,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    tier1_stage = next(stage for stage in result["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "succeeded"
    assert len(tier1_stage["attempts"]) == 2
    assert tier1_stage["attempts"][0]["status"] == "aborted"
    assert "resume superseded unfinished attempt" in tier1_stage["attempts"][0]["issues"]
    assert tier1_stage["attempts"][1]["status"] == "succeeded"


def test_normalize_stale_running_resume_state_marks_dead_orchestrator_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda _pid: False)

    resume_state = {
        "run_state": "running",
        "overall_status": "running",
        "success": False,
        "resume_available": False,
        "orchestrator_pid": 999999,
        "stages": [
            {
                "name": "tier1",
                "enabled": True,
                "status": "running",
                "run_id": "stale_e2e__tier1",
                "attempts": [
                    {
                        "run_id": "stale_e2e__tier1",
                        "status": "running",
                    }
                ],
            },
            {
                "name": "full_sweep",
                "enabled": False,
                "status": "skipped",
                "run_id": "stale_e2e__full_sweep",
                "attempts": [],
            },
        ],
    }

    reason = e2e_sweep._normalize_stale_running_resume_state(
        resume_state,
        repo_root=tmp_path,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert reason == "orchestrator process 999999 exited without finalizing run state"
    assert resume_state["run_state"] == "aborted"
    assert resume_state["overall_status"] == "aborted"
    assert resume_state["resume_available"] is True
    assert resume_state["stages"][0]["status"] == "aborted"
    assert resume_state["stages"][0]["attempts"][0]["status"] == "aborted"
    assert reason in resume_state["stages"][0]["attempts"][0]["issues"]


def test_run_benchmark_e2e_sweep_resume_rejects_renamed_frozen_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": ["ch13:torchao_quantization", "ch14:cublas_vs_cutlass"],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", lambda **kwargs: {})

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps(
                        {
                            "event_type": "chapter_end",
                            "chapter": "ch13",
                            "failed": 0,
                            "total_benchmarks": 1,
                            "successful": 1,
                            "skipped_hardware": 0,
                            "skipped_distributed": 0,
                            "informational": 0,
                        }
                    ),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps(
                        {
                            "event_type": "chapter_end",
                            "chapter": "ch14",
                            "failed": 0,
                            "total_benchmarks": 1,
                            "successful": 1,
                            "skipped_hardware": 0,
                            "skipped_distributed": 0,
                            "informational": 0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["run_id"],
                    "results": [
                        {
                            "chapter": "ch13",
                            "benchmarks": [
                                {"example": "torchao_quantization", "status": "succeeded"}
                            ],
                        },
                        {
                            "chapter": "ch14",
                            "benchmarks": [{"example": "cublas_vs_cutlass", "status": "succeeded"}],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_benchmark_manifest(paths["run_dir"] / "manifest.json", run_id=kwargs["run_id"])
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_unit_map", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "provenance": _resume_provenance(tmp_path, gpu_count=1),
                "contract": _resume_contract(
                    tmp_path,
                    run_tier1=False,
                    run_full_sweep=True,
                    run_cluster=False,
                    run_fabric=False,
                    artifacts_dir=str(runs_root),
                ),
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_unit_map__tier1",
                        "attempts": [],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume_unit_map__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_unit_map__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                                "units": ["ch13", "ch14"],
                                "completed_units": [],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {
                        "name": "cluster",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_unit_map__cluster",
                        "attempts": [],
                    },
                    {
                        "name": "fabric",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_unit_map__fabric",
                        "attempts": [],
                    },
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                        "single_gpu_units": ["ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_unit_map",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert result["overall_status"] == "failed"
    assert execute_calls == []
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert "resume could not resolve current benchmark targets for unit(s): ch14" in " ".join(
        full_stage["issues"]
    )


def test_run_benchmark_e2e_sweep_resume_reruns_partial_unit_with_resume_attempt_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    tier1_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 4, "single_gpu": 4, "multi_gpu": 0},
            "single_gpu": [
                "ch12:done",
                "ch13:torchao_quantization",
                "ch13:training_speed",
                "ch14:after",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: tier1_calls.append(kwargs) or {},
    )

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps(
                        {
                            "event_type": "chapter_end",
                            "chapter": "ch13",
                            "failed": 0,
                            "total_benchmarks": 2,
                            "successful": 2,
                            "skipped_hardware": 0,
                            "skipped_distributed": 0,
                            "informational": 0,
                        }
                    ),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps(
                        {
                            "event_type": "chapter_end",
                            "chapter": "ch14",
                            "failed": 0,
                            "total_benchmarks": 1,
                            "successful": 1,
                            "skipped_hardware": 0,
                            "skipped_distributed": 0,
                            "informational": 0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["run_id"],
                    "results": [
                        {
                            "chapter": "ch13",
                            "benchmarks": [
                                {"example": "torchao_quantization", "status": "succeeded"},
                                {"example": "training_speed", "status": "succeeded"},
                            ],
                        },
                        {
                            "chapter": "ch14",
                            "benchmarks": [
                                {"example": "after", "status": "succeeded"},
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_benchmark_manifest(paths["run_dir"] / "manifest.json", run_id=kwargs["run_id"])
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    _write_benchmark_attempt_artifacts(
        repo_root=tmp_path,
        artifacts_dir=str(runs_root),
        run_id="e2e_resume__full_sweep__single",
        targets=[
            "ch12:done",
            "ch13:torchao_quantization",
            "ch13:training_speed",
            "ch14:after",
        ],
        results=[
            {
                "chapter": "ch12",
                "benchmarks": [{"example": "done", "status": "succeeded"}],
            }
        ],
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "provenance": _resume_provenance(tmp_path, gpu_count=1),
                "contract": _resume_contract(
                    tmp_path,
                    run_tier1=True,
                    run_full_sweep=True,
                    run_cluster=False,
                    run_fabric=False,
                    artifacts_dir=str(runs_root),
                ),
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "succeeded",
                        "run_id": "e2e_resume__tier1",
                        "attempts": [
                            {
                                "run_id": "e2e_resume__tier1",
                                "status": "succeeded",
                                "result": _successful_tier1_invocation_result(
                                    run_id="e2e_resume__tier1",
                                    repo_root=tmp_path,
                                    suite_definitions=[],
                                    artifacts_dir=str(runs_root),
                                ),
                            }
                        ],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": [
                                    "ch12:done",
                                    "ch13:torchao_quantization",
                                    "ch13:training_speed",
                                    "ch14:after",
                                ],
                                "units": ["ch12", "ch13", "ch14"],
                                "completed_units": ["ch12"],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {
                        "name": "cluster",
                        "enabled": False,
                        "status": "planned",
                        "run_id": "e2e_resume__cluster",
                        "attempts": [],
                    },
                    {
                        "name": "fabric",
                        "enabled": False,
                        "status": "planned",
                        "run_id": "e2e_resume__fabric",
                        "attempts": [],
                    },
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "ch12:done",
                            "ch13:torchao_quantization",
                            "ch13:training_speed",
                            "ch14:after",
                        ],
                        "single_gpu_units": ["ch12", "ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume",
        resume=True,
        run_tier1=True,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert tier1_calls == []
    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == [
        "ch13:torchao_quantization",
        "ch13:training_speed",
        "ch14:after",
    ]

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "succeeded"
    assert len(full_stage["attempts"]) == 2
    assert full_stage["attempts"][0]["run_id"] == "e2e_resume__full_sweep__single"
    assert full_stage["attempts"][1]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert full_stage["attempts"][1]["completed_units"] == ["ch13", "ch14"]
    assert (
        full_stage["result"]["buckets"]["single_gpu"]["latest_attempt_run_id"]
        == "e2e_resume__full_sweep__single__resume1"
    )
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_resume_canonicalizes_lab_unit_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": [
                "labs/async_input_pipeline:async_input_pipeline",
                "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps(
                        {"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}
                    ),
                    json.dumps(
                        {
                            "event_type": "chapter_end",
                            "chapter": "labs_trtllm_phi_3_5_moe",
                            "failed": 0,
                            "total_benchmarks": 1,
                            "successful": 1,
                            "skipped_hardware": 0,
                            "skipped_distributed": 0,
                            "informational": 0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["run_id"],
                    "results": [
                        {
                            "chapter": "labs_trtllm_phi_3_5_moe",
                            "benchmarks": [
                                {"example": "trtllm_phi_3_5_moe", "status": "succeeded"},
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_benchmark_manifest(paths["run_dir"] / "manifest.json", run_id=kwargs["run_id"])
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    prior_paths = e2e_sweep._benchmark_run_event_paths(
        "e2e_resume_labs__full_sweep__single",
        repo_root=tmp_path,
        artifacts_dir=str(runs_root),
    )
    prior_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["events"].write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "run_start",
                        "targets": [
                            "labs/async_input_pipeline:async_input_pipeline",
                            "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                        ],
                    }
                ),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_async_input_pipeline"}),
                json.dumps(
                    {
                        "event_type": "chapter_end",
                        "chapter": "labs_async_input_pipeline",
                        "failed": 0,
                        "total_benchmarks": 1,
                        "successful": 1,
                        "skipped_hardware": 0,
                        "skipped_distributed": 0,
                        "informational": 0,
                    }
                ),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prior_paths["output_json"].write_text(
        json.dumps(
            {
                "run_id": "e2e_resume_labs__full_sweep__single",
                "results": [
                    {
                        "chapter": "labs_async_input_pipeline",
                        "benchmarks": [
                            {"example": "async_input_pipeline", "status": "succeeded"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_benchmark_manifest(
        prior_paths["run_dir"] / "manifest.json",
        run_id="e2e_resume_labs__full_sweep__single",
    )
    prior_paths["progress"].write_text("{}", encoding="utf-8")

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_labs", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "run_state": "running",
                "orchestrator_pid": 999999,
                "provenance": _resume_provenance(tmp_path, gpu_count=1),
                "contract": _resume_contract(
                    tmp_path,
                    run_tier1=False,
                    run_full_sweep=True,
                    run_cluster=False,
                    run_fabric=False,
                    artifacts_dir=str(runs_root),
                ),
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_labs__tier1",
                        "attempts": [],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_labs__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_labs__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "running",
                                "targets": [
                                    "labs/async_input_pipeline:async_input_pipeline",
                                    "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                                ],
                                "units": ["labs/async_input_pipeline", "labs/trtllm_phi_3_5_moe"],
                                "completed_units": [],
                                "active_unit": "labs/async_input_pipeline",
                            }
                        ],
                    },
                    {
                        "name": "cluster",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_labs__cluster",
                        "attempts": [],
                    },
                    {
                        "name": "fabric",
                        "enabled": False,
                        "status": "skipped",
                        "run_id": "e2e_resume_labs__fabric",
                        "attempts": [],
                    },
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "labs/async_input_pipeline:async_input_pipeline",
                            "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                        ],
                        "single_gpu_units": [
                            "labs/async_input_pipeline",
                            "labs/trtllm_phi_3_5_moe",
                        ],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_labs",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume_labs__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == ["labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe"]
    assert execute_calls[0]["enforce_external_assets"] is False
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["attempts"][0]["status"] == "aborted"
    assert full_stage["attempts"][0]["completed_units"] == ["labs/async_input_pipeline"]
    assert full_stage["attempts"][0]["active_unit"] == "labs/trtllm_phi_3_5_moe"
    assert full_stage["attempts"][1]["completed_units"] == ["labs/trtllm_phi_3_5_moe"]
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_rejects_portable_expectation_writes() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_invalid",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        validity_profile="portable",
        update_expectations=True,
        dry_run=True,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "allow-portable-expectations-update" in result["error"]


def test_run_benchmark_e2e_sweep_rejects_non_local_hosts_without_ssh_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0},
            "single_gpu": [],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_bad_hosts",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        hosts=["gpu-node-1"],
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "Non-local hosts require explicit ssh_user and ssh_key" in result["error"]


def test_benchmark_e2e_sweep_handler_returns_async_ticket(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.run_benchmark_e2e_sweep",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"]},
    )

    ticket = handlers.benchmark_e2e_sweep(
        {
            "async": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
            "run_id": "handler_async_e2e",
        }
    )

    assert ticket["status"] == "queued"
    assert ticket["run_id"] == "handler_async_e2e"
    assert ticket["run_dir"].endswith("artifacts/e2e_runs/handler_async_e2e")
    assert ticket["progress_path"].endswith("artifacts/e2e_runs/handler_async_e2e/progress.json")
    assert ticket["actions"]["preferred_mcp_tool"] == "benchmark_e2e_status"
    assert ticket["preferred_progress_source"]["kind"] == "normalized_e2e_status"
    assert ticket["actions"]["status_api_path"].endswith(
        "/api/benchmark/e2e-status?run_id=handler_async_e2e"
    )

    record = handlers.JobStore.get().get_status(ticket["job_id"])
    if record is not None:
        handlers.JobStore.get().update_job(ticket["job_id"], status="completed")


def test_tool_benchmark_e2e_sweep_delegates_to_handler(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_sweep",
        lambda params: {
            "success": True,
            "run_id": "tool_e2e",
            "echo": params.get("dry_run", False),
        },
    )

    payload = mcp_server.tool_benchmark_e2e_sweep({"dry_run": True})

    assert payload["success"] is True
    assert payload["run_id"] == "tool_e2e"
    assert payload["echo"] is True


def test_benchmark_e2e_sweep_handler_passes_resume_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs["run_id"], "resume": kwargs["resume"]}

    monkeypatch.setattr("core.benchmark.e2e_sweep.run_benchmark_e2e_sweep", _fake_run)

    payload = handlers.benchmark_e2e_sweep(
        {
            "run_id": "handler_resume_e2e",
            "resume": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
        }
    )

    assert payload["success"] is True
    assert payload["resume"] is True
    assert captured["resume"] is True


def test_run_benchmark_e2e_sweep_uses_full_sweep_suite_timeout_for_bucket_runs(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    artifacts_dir = str(tmp_path / "artifacts")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 1, "single_gpu": 1, "multi_gpu": 0},
            "single_gpu": ["ch10:atomic_reduction"],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "watch_benchmark_e2e_sweep_run",
        lambda **kwargs: {
            "success": True,
            "watcher_pid": 777,
            "watch_status_path": str(tmp_path / "watcher.json"),
        },
    )

    def _fake_execute(**kwargs):
        captured.update(kwargs)
        output_json = _write_benchmark_attempt_artifacts(
            repo_root=tmp_path,
            artifacts_dir=artifacts_dir,
            run_id=kwargs["run_id"],
            targets=kwargs["targets"],
            results=[
                {
                    "chapter": "ch10",
                    "benchmarks": [{"example": "atomic_reduction", "status": "succeeded"}],
                }
            ],
        )
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(output_json),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fs_timeout",
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        suite_timeout=14400,
        full_sweep_suite_timeout=0,
        artifacts_dir=artifacts_dir,
    )

    assert result["success"] is True
    assert captured["suite_timeout"] == 0
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert "--suite-timeout" in full_stage["attempts"][0]["command"]
    assert "0" in full_stage["attempts"][0]["command"]


def test_inspect_benchmark_e2e_sweep_run_detects_live_progress_and_child_events(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_live_status"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        f"{run_id}__full_sweep__single",
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps(
            {
                "event_type": "example_end",
                "chapter": "ch10",
                "example": "cluster_multicast",
                "status": "succeeded",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")
    watcher_status_path = e2e_sweep.e2e_watcher_status_path(run_dir)
    watcher_status_path.write_text(
        json.dumps({"watcher_pid": 456, "watch_state": "watching", "auto_resume_count": 0}),
        encoding="utf-8",
    )

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch10",
                    "completed_units": ["ch06"],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:10:00+00:00",
            "step": "full_sweep/single_gpu:ch10:cluster_multicast",
            "step_detail": "ncu profiling (optimized)",
            "percent_complete": 42.0,
            "elapsed_seconds": 100.0,
            "eta_seconds": 200.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": f"{run_id}__full_sweep__single",
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"ts": "2026-03-27T16:10:00Z", "event": "stage_started", "stage": "full_sweep"})
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) in {123, 456})

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["success"] is True
    assert status["inferred_state"] == "running_live"
    assert status["progress_source"]["kind"] == "live_child_progress"
    assert status["progress_source"]["preferred_mcp_tool"] == "benchmark_e2e_status"
    assert status["current"]["child_artifacts"]["events_path"].endswith("benchmark_events.jsonl")
    assert status["current"]["recent_child_events"][0]["event_type"] == "example_end"
    assert status["watcher"]["watcher_pid"] == 456
    assert status["actions"]["status_api_path"].endswith(
        f"/api/benchmark/e2e-status?run_id={run_id}"
    )
    assert status["actions"]["dashboard_path"].endswith(f"/e2e?run_id={run_id}")


def test_inspect_benchmark_e2e_sweep_run_downgrades_stale_active_watcher_state(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_dead_watcher_state"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_state": "aborted",
        "overall_status": "aborted",
        "updated_at": "2026-03-29T15:48:37Z",
        "resume_available": True,
        "contract": {"auto_resume": True},
        "stages": [],
        "orchestrator_pid": 999,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-29T15:48:37+00:00",
                    "metrics": {
                        "run_state": "aborted",
                        "overall_status": "aborted",
                        "orchestrator_pid": 999,
                        "stages": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    e2e_sweep.e2e_watcher_status_path(run_dir).write_text(
        json.dumps({"watcher_pid": 456, "watch_state": "resuming", "auto_resume_count": 1}),
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)

    assert status["watcher"]["stored_watch_state"] == "resuming"
    assert status["watcher"]["watch_state"] == "stale_dead"
    assert status["watcher"]["watcher_live"] is False
    assert (
        "stored watcher watch_state `resuming` corrected to `stale_dead` because the watcher pid is not live"
        in status["notes"]
    )
    assert "watcher_state=stale_dead" in rendered
    assert "stored_watcher_state=resuming" in rendered


def test_inspect_benchmark_e2e_sweep_run_syncs_active_issue_ledger_from_live_failures(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_live_issue_sync"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_run_id = f"{run_id}__full_sweep__single"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps(
            {
                "event_type": "example_end",
                "timestamp": "2026-03-27T16:20:34.180782",
                "run_id": child_run_id,
                "chapter": "ch13",
                "example": "memory_profiling",
                "status": "failed_no_speedup",
                "best_speedup": 1.0,
                "best_memory_savings_pct": 10.1673,
                "optimization_goal": "speed",
                "error": "Best speedup 1.00x below required 1.05x threshold for speed-goal benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")

    existing_ledger = {
        "summary": {"run_id": run_id, "issue_count": 1, "resolved_count": 1, "unresolved_count": 0},
        "rows": [
            {
                "issue_id": "tier1_goalaware_regression_summary",
                "stage": "tier1",
                "status": "resolved",
                "symptom": "Existing manual issue",
                "root_cause": "Fixed already.",
                "fixes": [],
                "verification": {"pytest": "pytest -q tests/test_tier1_suite.py"},
                "evidence_paths": {},
            }
        ],
    }
    (run_dir / "active_issue_ledger.json").write_text(json.dumps(existing_ledger), encoding="utf-8")
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch13",
                    "completed_units": ["ch06"],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:21:00+00:00",
            "step": "full_sweep/single_gpu:ch13:memory_profiling",
            "step_detail": "timing (optimized)",
            "percent_complete": 48.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["current"]["reported_failures"][0]["target"] == "ch13:memory_profiling"
    assert any(entry["target"] == "ch13:memory_profiling" for entry in status["aggregate_failures"])
    assert status["ledgers"]["active_issue_ledger_json"].endswith("active_issue_ledger.json")

    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    assert synced_ledger["summary"]["run_id"] == run_id
    assert synced_ledger["summary"]["issue_count"] == 2
    assert synced_ledger["summary"]["resolved_count"] == 1
    assert synced_ledger["summary"]["unresolved_count"] == 1
    assert synced_ledger["summary"]["reported_issue_count"] == 1
    assert synced_ledger["summary"]["issue_group_count"] == 1
    issue_ids = [row["issue_id"] for row in synced_ledger["rows"]]
    assert "tier1_goalaware_regression_summary" in issue_ids
    assert "reported_full_sweep_single_gpu_ch13_memory_profiling" in issue_ids
    assert synced_ledger["summary"]["issue_group_count"] == 1
    assert (
        synced_ledger["issue_groups"][0]["signature"]
        == "Best speedup 1.00x below required 1.05x threshold for speed-goal benchmark"
    )
    markdown = (run_dir / "active_issue_ledger.md").read_text(encoding="utf-8")
    assert "ch13:memory_profiling" in markdown


def test_inspect_benchmark_e2e_sweep_run_preserves_local_no_speedup_threshold_from_child_events(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_local_threshold_issue_sync"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_run_id = f"{run_id}__full_sweep__single"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps(
            {
                "event_type": "example_end",
                "timestamp": "2026-03-27T16:20:34.180782",
                "run_id": child_run_id,
                "chapter": "ch06",
                "example": "launch_bounds",
                "status": "failed_no_speedup",
                "best_speedup": 1.006,
                "best_memory_savings_pct": 0.0,
                "optimization_goal": "speed",
                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch06",
                    "completed_units": [],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:21:00+00:00",
            "step": "full_sweep/single_gpu:ch06:launch_bounds",
            "step_detail": "timing (optimized)",
            "percent_complete": 48.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["current"]["reported_failures"][0]["target"] == "ch06:launch_bounds"
    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["issue_groups"][0]["signature"] == (
        "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark"
    )
    row = next(
        item
        for item in synced_ledger["rows"]
        if item["issue_id"] == "reported_full_sweep_single_gpu_ch06_launch_bounds"
    )
    assert row["symptom"].endswith(
        "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark"
    )


def test_inspect_benchmark_e2e_sweep_run_detects_stale_running_and_builds_resume_command(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_stale_status"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = _resume_contract(
        tmp_path,
        run_tier1=True,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        validity_profile="portable",
    )
    payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": False,
        "contract": contract,
        "stages": [],
        "orchestrator_pid": 999,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-27T16:10:00+00:00",
                    "metrics": {
                        "run_state": "running",
                        "overall_status": "running",
                        "orchestrator_pid": 999,
                        "stages": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["inferred_state"] == "running_stale"
    assert "--resume" in status["actions"]["resume_command"]
    assert "--full-sweep-suite-timeout" in status["actions"]["resume_command"]
    assert status["resume_available"] is True
    assert status["stored_resume_available"] is False
    assert (
        "stored resume_available was false. Corrected to true for stale running package"
        in status["notes"]
    )


def test_inspect_benchmark_e2e_sweep_run_preserves_terminal_full_sweep_failures_from_all_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_full_sweep_failures"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T04:55:27Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": [
                    {
                        "run_id": f"{run_id}__full_sweep__single",
                        "bucket": "single_gpu",
                        "status": "aborted",
                        "artifacts": {
                            "events_path": str(tmp_path / "events_single.jsonl"),
                            "output_json": str(tmp_path / "single_results.json"),
                            "progress_path": str(tmp_path / "single_progress.json"),
                            "run_dir": str(tmp_path / "single_run"),
                        },
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch06:launch_bounds",
                                    "status": "failed_no_speedup",
                                    "error": "Best speedup 1.01x below required 1.05x threshold",
                                }
                            ],
                            "skipped_benchmarks": [],
                        },
                    },
                    {
                        "run_id": f"{run_id}__full_sweep__single__resume1",
                        "bucket": "single_gpu",
                        "status": "failed",
                        "artifacts": {
                            "events_path": str(tmp_path / "events_resume.jsonl"),
                            "output_json": str(tmp_path / "resume_results.json"),
                            "progress_path": str(tmp_path / "resume_progress.json"),
                            "run_dir": str(tmp_path / "resume_run"),
                        },
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch13:regional_compile",
                                    "status": "failed_no_speedup",
                                    "error": "Best speedup 1.05x below required 1.05x threshold",
                                }
                            ],
                            "skipped_benchmarks": [],
                        },
                    },
                    {
                        "run_id": f"{run_id}__full_sweep__multi",
                        "bucket": "multi_gpu",
                        "status": "skipped",
                    },
                ],
            }
        ],
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 1204046
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-28T04:55:27+00:00",
            "metrics": {
                "run_state": "completed",
                "overall_status": "failed",
                "orchestrator_pid": 1204046,
                "stages": [
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "run_id": f"{run_id}__full_sweep",
                        "status": "failed",
                        "attempts": [
                            {
                                "run_id": f"{run_id}__full_sweep__multi",
                                "bucket": "multi_gpu",
                                "status": "skipped",
                            }
                        ],
                    }
                ],
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    existing_ledger = {
        "summary": {"run_id": run_id, "issue_count": 1, "resolved_count": 1, "unresolved_count": 0},
        "rows": [
            {
                "issue_id": "tier1_goalaware_regression_summary",
                "stage": "tier1",
                "status": "resolved",
                "symptom": "Existing manual issue",
                "root_cause": "Fixed already.",
                "fixes": [],
                "verification": {"pytest": "pytest -q tests/test_tier1_suite.py"},
                "evidence_paths": {},
            }
        ],
    }
    (run_dir / "active_issue_ledger.json").write_text(json.dumps(existing_ledger), encoding="utf-8")
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    aggregate_targets = {entry["target"] for entry in status["aggregate_failures"]}
    assert "ch06:launch_bounds" in aggregate_targets
    assert "ch13:regional_compile" in aggregate_targets
    surfaced_targets = {entry["target"] for entry in status["current"]["reported_failures"]}
    assert surfaced_targets == aggregate_targets

    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    issue_ids = {row["issue_id"] for row in synced_ledger["rows"]}
    assert "reported_full_sweep_single_gpu_ch06_launch_bounds" in issue_ids
    assert "reported_full_sweep_single_gpu_ch13_regional_compile" in issue_ids
    assert synced_ledger["summary"]["issue_count"] == 3
    assert status["ledgers"]["summary"]["issue_count"] == 3
    assert status["ledgers"]["summary"]["resolved_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 2
    assert status["ledgers"]["summary"]["active_issue_count"] == 2
    assert status["ledgers"]["summary"]["historical_issue_count"] == 0
    assert status["issue_groups"]

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "reported_failures=2" in rendered
    assert "active_issue_counts=issues=2 unresolved=2" in rendered
    assert "ledger_totals=issues=3 resolved=1 unresolved=2" in rendered
    assert "issue_groups=" in rendered
    assert "ch06:launch_bounds[failed_no_speedup]" in rendered
    assert "ch13:regional_compile[failed_no_speedup]" in rendered


def test_inspect_benchmark_e2e_sweep_run_recovers_tier1_failures_from_nested_suite_result(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_tier1_nested_failure"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_json = tmp_path / "tier1_run" / "results" / "benchmark_test_results.json"
    summary_path = tmp_path / "history" / run_id / "summary.json"
    regression_summary_path = tmp_path / "history" / run_id / "regression_summary.md"
    trend_snapshot_path = tmp_path / "history" / run_id / "trend.json"
    for path in (output_json, summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs_block_scaling",
                        "benchmarks": [
                            {
                                "example": "block_scaling",
                                "status": "failed_no_speedup",
                                "best_speedup": 1.7492851550300643,
                                "best_memory_savings_pct": 0.0,
                                "optimization_goal": "speed",
                                "minimum_required_speedup": 1.75,
                                "error": "Best speedup 1.75x below required 1.75x threshold for speed-goal benchmark",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "targets": [
                    {
                        "target": "labs/block_scaling:block_scaling",
                        "status": "failed_no_speedup",
                        "best_speedup": 1.7492851550300643,
                        "best_memory_savings_pct": 0.0,
                        "optimization_goal": "speed",
                    }
                ],
                "summary": {"failed": 1, "skipped": 0, "succeeded": 5},
            }
        ),
        encoding="utf-8",
    )
    regression_summary_path.write_text("# regressions\n", encoding="utf-8")
    trend_snapshot_path.write_text("{}", encoding="utf-8")

    stage_payload = [
        {
            "name": "tier1",
            "enabled": True,
            "run_id": f"{run_id}__tier1",
            "status": "succeeded",
            "attempts": [
                {
                    "run_id": f"{run_id}__tier1",
                    "status": "succeeded",
                    "result": {
                        "execution": {
                            "run_id": f"{run_id}__tier1",
                            "output_json": str(output_json),
                            "total_failed": 1,
                            "total_skipped": 0,
                        },
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(regression_summary_path),
                        "trend_snapshot_path": str(trend_snapshot_path),
                    },
                    "artifacts": {
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(regression_summary_path),
                        "trend_snapshot_path": str(trend_snapshot_path),
                    },
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "succeeded",
        "updated_at": "2026-03-29T06:58:44Z",
        "resume_available": False,
        "stages": stage_payload,
        "contract": {"run_tier1": True},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 777
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T06:58:44+00:00",
            "metrics": {
                "run_state": "completed",
                "overall_status": "succeeded",
                "orchestrator_pid": 777,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert {entry["target"] for entry in status["aggregate_failures"]} == {
        "labs/block_scaling:block_scaling"
    }
    assert {entry["target"] for entry in status["current"]["reported_failures"]} == {
        "labs/block_scaling:block_scaling"
    }
    assert status["stages"][0]["status"] == "failed"
    assert status["stages"][0]["stored_status"] == "succeeded"
    assert status["overall_status"] == "failed"
    assert status["stored_overall_status"] == "succeeded"
    assert status["ledgers"]["summary"]["unresolved_count"] == 1


def test_inspect_benchmark_e2e_sweep_run_preserves_resolved_reported_issue_rows(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_preserve_resolved_reported"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_attempts = [
        {
            "run_id": f"{run_id}__full_sweep__single",
            "bucket": "single_gpu",
            "status": "failed",
            "artifacts": {
                "events_path": str(tmp_path / "events_single.jsonl"),
                "output_json": str(tmp_path / "single_results.json"),
            },
            "benchmark_summary": {
                "failed_benchmarks": [
                    {
                        "target": "ch13:regional_compile",
                        "status": "failed_no_speedup",
                        "error": "Best speedup 1.05x below required 1.05x threshold",
                    }
                ],
                "skipped_benchmarks": [],
            },
        },
        {
            "run_id": f"{run_id}__full_sweep__multi",
            "bucket": "multi_gpu",
            "status": "skipped",
        },
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T05:30:00Z",
        "resume_available": False,
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": stage_attempts,
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-28T05:30:00+00:00",
                    "metrics": {
                        "run_state": "completed",
                        "overall_status": "failed",
                        "stages": [
                            {
                                "name": "full_sweep",
                                "enabled": True,
                                "run_id": f"{run_id}__full_sweep",
                                "status": "failed",
                                "attempts": [
                                    {
                                        "run_id": f"{run_id}__full_sweep__multi",
                                        "bucket": "multi_gpu",
                                        "status": "skipped",
                                    }
                                ],
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "active_issue_ledger.json").write_text(
        json.dumps(
            {
                "summary": {
                    "run_id": run_id,
                    "issue_count": 1,
                    "resolved_count": 1,
                    "unresolved_count": 0,
                },
                "rows": [
                    {
                        "issue_id": "reported_full_sweep_single_gpu_ch13_regional_compile",
                        "stage": "full_sweep/single_gpu",
                        "status": "resolved",
                        "symptom": "regional_compile rerun passed at 1.08x",
                        "root_cause": "Initial full sweep result was a noisy borderline measurement.",
                        "fixes": [
                            "Re-ran ch13:regional_compile on an uncontended GPU and confirmed 1.08x."
                        ],
                        "verification": {
                            "command": "python -m cli.aisp bench run --targets ch13:regional_compile --profile minimal --validity-profile portable --single-gpu"
                        },
                        "evidence_paths": {"rerun_output_json": str(tmp_path / "rerun.json")},
                        "resolved_at": "2026-03-28T06:10:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert {entry["target"] for entry in status["aggregate_failures"]} == {"ch13:regional_compile"}
    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    row = next(
        item
        for item in synced_ledger["rows"]
        if item["issue_id"] == "reported_full_sweep_single_gpu_ch13_regional_compile"
    )
    assert row["status"] == "resolved"
    assert row["root_cause"] == "Initial full sweep result was a noisy borderline measurement."
    assert row["verification"]["command"].endswith("--single-gpu")
    assert row["evidence_paths"]["rerun_output_json"].endswith("rerun.json")
    assert status["ledgers"]["summary"]["resolved_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 0
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 0
    assert status["current"]["reported_failures"] == []
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "reported_failures=" not in rendered


def test_inspect_benchmark_e2e_sweep_run_splits_historical_reported_rows_from_active_counts_while_live(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_live_resume_historical_rows"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    child_run_id = f"{run_id}__full_sweep__single__resume2"
    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "failed",
                    "benchmark_summary": {
                        "failed_benchmarks": [
                            {
                                "target": "ch06:launch_bounds",
                                "status": "failed_no_speedup",
                                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
                            }
                        ],
                        "skipped_benchmarks": [],
                    },
                },
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch18",
                    "completed_units": ["ch17"],
                },
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-29T17:00:00Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": stage_payload,
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T17:00:05+00:00",
            "step": "full_sweep/single_gpu:ch18:flexdecoding",
            "step_detail": "optimized timing",
            "percent_complete": 66.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    benchmark_paths = e2e_sweep._benchmark_run_event_paths(  # type: ignore[attr-defined]
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    benchmark_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-29T17:00:04Z",
                        "event_type": "example_end",
                        "run_id": child_run_id,
                        "chapter": "ch17",
                        "example": "dynamic_routing",
                        "status": "succeeded",
                        "best_speedup": 33.98,
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["current"]["reported_failures"] == []
    assert status["issue_groups"] == []
    assert status["historical_issue_groups"]
    assert status["ledgers"]["summary"]["issue_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 1
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 1

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "active_issue_counts=issues=0 unresolved=0" in rendered
    assert "historical_issue_counts=issues=1 unresolved=1" in rendered
    assert "ledger_totals=issues=1 resolved=0 unresolved=1" in rendered
    assert "historical_issue_groups=1" in rendered


def test_failed_chapter_end_is_not_complete_for_resume(tmp_path: Path) -> None:
    run_id = "e2e_failed_chapter_resume"
    paths = e2e_sweep._benchmark_run_event_paths(  # type: ignore[attr-defined]
        run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    paths["events"].parent.mkdir(parents=True)
    paths["events"].write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "chapter_start",
                        "chapter": "ch06",
                    }
                ),
                json.dumps(
                    {
                        "event_type": "chapter_end",
                        "chapter": "ch06",
                        "status": "completed",
                        "total_benchmarks": 2,
                        "successful": 1,
                        "failed": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress = e2e_sweep._load_benchmark_unit_progress(  # type: ignore[attr-defined]
        run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )

    assert progress["completed_units"] == []
    assert progress["active_unit"] == "ch06"


@pytest.mark.parametrize(
    ("successful", "skipped_hardware", "informational"),
    [(0, 1, 0), (1, 0, 1)],
)
def test_incomplete_chapter_end_is_not_complete_for_resume(
    tmp_path: Path,
    successful: int,
    skipped_hardware: int,
    informational: int,
) -> None:
    run_id = "e2e_incomplete_chapter_resume"
    paths = e2e_sweep._benchmark_run_event_paths(
        run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    paths["events"].parent.mkdir(parents=True)
    paths["events"].write_text(
        "\n".join(
            [
                json.dumps({"event_type": "chapter_start", "chapter": "ch06"}),
                json.dumps(
                    {
                        "event_type": "chapter_end",
                        "chapter": "ch06",
                        "total_benchmarks": 1,
                        "successful": successful,
                        "failed": 0,
                        "skipped_hardware": skipped_hardware,
                        "skipped_distributed": 0,
                        "informational": informational,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress = e2e_sweep._load_benchmark_unit_progress(
        run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )

    assert progress["completed_units"] == []
    assert progress["active_unit"] == "ch06"


def test_latest_successful_retry_clears_active_failure_but_preserves_attempt_history() -> None:
    stage = {
        "name": "full_sweep",
        "enabled": True,
        "status": "succeeded",
        "attempts": [
            {
                "run_id": "full__single",
                "bucket": "single_gpu",
                "status": "failed",
                "benchmark_summary": {
                    "target_outcomes": [{"target": "ch06:demo", "status": "failed_no_speedup"}],
                    "failed_benchmarks": [
                        {
                            "target": "ch06:demo",
                            "status": "failed_no_speedup",
                            "error": "first attempt failed",
                        }
                    ],
                    "skipped_benchmarks": [],
                },
            },
            {
                "run_id": "full__single__resume1",
                "bucket": "single_gpu",
                "status": "succeeded",
                "benchmark_summary": {
                    "target_outcomes": [{"target": "ch06:demo", "status": "succeeded"}],
                    "failed_benchmarks": [],
                    "skipped_benchmarks": [],
                },
            },
        ],
    }

    snapshot = e2e_sweep._stage_snapshot(stage)  # type: ignore[attr-defined]

    assert snapshot["failed_benchmarks"] == []
    assert snapshot["status_counts"] == {"succeeded": 1}
    assert e2e_sweep._effective_stage_snapshot_status(snapshot) == "succeeded"  # type: ignore[attr-defined]
    assert stage["attempts"][0]["benchmark_summary"]["failed_benchmarks"]


def test_non_resume_e2e_run_rejects_existing_package_without_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "e2e_existing_package"
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.json"
    sentinel.write_bytes(b'{"preserve": true}')
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*")}

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id=run_id,
        resume=False,
        dry_run=True,
    )

    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*")}
    assert result["overall_status"] == "failed"
    assert "Refusing to overwrite existing E2E run" in result["error"]
    assert after == before


def test_inspect_benchmark_e2e_sweep_run_keeps_historical_rows_historical_when_running_stale_with_child_progress(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_stale_with_live_child_progress"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    child_run_id = f"{run_id}__full_sweep__single__resume2"
    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "failed",
                    "benchmark_summary": {
                        "failed_benchmarks": [
                            {
                                "target": "ch06:launch_bounds",
                                "status": "failed_no_speedup",
                                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
                            }
                        ],
                        "skipped_benchmarks": [],
                    },
                },
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch18",
                    "completed_units": ["ch17"],
                },
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-29T17:00:00Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": stage_payload,
        "orchestrator_pid": 999,
    }
    checkpoint_payload = dict(summary_payload)
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T17:00:05+00:00",
            "step": "full_sweep/single_gpu:ch18:flexdecoding",
            "step_detail": "optimized timing",
            "percent_complete": 66.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 999,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    benchmark_paths = e2e_sweep._benchmark_run_event_paths(  # type: ignore[attr-defined]
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    benchmark_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    benchmark_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["progress"].write_text(
        json.dumps(
            {
                "run_id": child_run_id,
                "current": {
                    "timestamp": "2026-03-29T17:00:06+00:00",
                    "phase": "optimized_timing",
                    "step": "ch18:flexdecoding",
                    "step_detail": "optimized timing",
                    "percent_complete": 12.0,
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark_paths["events"].write_text(
        json.dumps(
            {
                "timestamp": "2026-03-29T17:00:04Z",
                "event_type": "example_end",
                "run_id": child_run_id,
                "chapter": "ch17",
                "example": "dynamic_routing",
                "status": "succeeded",
                "best_speedup": 33.98,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["inferred_state"] == "running_stale"
    assert status["current"]["reported_failures"] == []
    assert status["issue_groups"] == []
    assert status["historical_issue_groups"]
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 1

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "active_issue_counts=issues=0 unresolved=0" in rendered
    assert "historical_issue_counts=issues=1 unresolved=1" in rendered


def test_inspect_benchmark_e2e_sweep_run_groups_cascade_failures(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_grouped_cascade"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T05:30:00Z",
        "resume_available": False,
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": [
                    {
                        "run_id": f"{run_id}__full_sweep__single",
                        "bucket": "single_gpu",
                        "status": "failed",
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch07:matmul_tiled",
                                    "status": "failed_error",
                                    "error": "Baseline execution failed: received SIGHUP",
                                },
                                {
                                    "target": "ch07:memory_access",
                                    "status": "failed_error",
                                    "error": "Baseline execution failed: [Errno 5] Input/output error",
                                },
                            ],
                            "skipped_benchmarks": [],
                        },
                    }
                ],
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-28T05:30:00+00:00",
                    "metrics": {
                        "run_state": "completed",
                        "overall_status": "failed",
                        "stages": summary_payload["stages"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    groups = status["issue_groups"]
    assert len(groups) == 2
    signatures = {group["signature_key"] for group in groups}
    assert "received_sighup" in signatures
    assert "errno5_input_output_error" in signatures
    errno_group = next(
        group for group in groups if group["signature_key"] == "errno5_input_output_error"
    )
    assert errno_group["cascade_from"] == "received_sighup"
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "issue_group_preview=" in rendered


def test_watch_benchmark_e2e_sweep_run_launches_detached_watcher(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    run_id = "e2e_watch_launch"
    run_dir = e2e_sweep.e2e_run_dir(run_id, repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: repo_root)

    captured: dict[str, object] = {}

    class DummyProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def _fake_popen(cmd, cwd=None, stdout=None, stderr=None, start_new_session=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout_name"] = getattr(stdout, "name", None)
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        return DummyProc(54321)

    monkeypatch.setattr(e2e_sweep.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    result = e2e_sweep.watch_benchmark_e2e_sweep_run(
        run_id=run_id, repo_root=repo_root, poll_interval_seconds=5, max_auto_resumes=2
    )

    assert result["success"] is True
    assert result["watcher_pid"] == 54321
    assert "--run-id" in captured["cmd"]
    assert "--poll-interval-seconds" in captured["cmd"]
    assert "--max-auto-resumes" in captured["cmd"]
    assert captured["cwd"] == str(repo_root)
    assert captured["start_new_session"] is True
    assert captured["stdout_name"].endswith(f"{run_id}_watcher.launch.log")


def test_benchmark_e2e_status_handler_and_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.inspect_benchmark_e2e_sweep_run",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "inferred_state": "running_live",
        },
    )

    payload = handlers.benchmark_e2e_status({"run_id": "handler_status"})
    assert payload["run_id"] == "handler_status"
    assert payload["inferred_state"] == "running_live"

    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_status",
        lambda params: {"success": True, "run_id": params.get("run_id"), "source": "tool"},
    )
    tool_payload = mcp_server.tool_benchmark_e2e_status({"run_id": "tool_status"})
    assert tool_payload["source"] == "tool"


def test_benchmark_e2e_watch_handler_and_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.watch_benchmark_e2e_sweep_run",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"], "watcher_pid": 123},
    )
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.resolve_latest_e2e_run_id",
        lambda **kwargs: "latest_watch",
    )

    payload = handlers.benchmark_e2e_watch({})
    assert payload["run_id"] == "latest_watch"
    assert payload["watcher_pid"] == 123

    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_watch",
        lambda params: {"success": True, "run_id": params.get("run_id"), "source": "tool-watch"},
    )
    tool_payload = mcp_server.tool_benchmark_e2e_watch({"run_id": "tool_watch"})
    assert tool_payload["source"] == "tool-watch"
