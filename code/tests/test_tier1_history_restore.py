from __future__ import annotations

import io
import json
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from core.scripts.ci import restore_tier1_history as restore


def _paired_artifacts(run_id: str, *, created_at: str) -> list[dict[str, object]]:
    return [
        {
            "id": int(run_id) * 10 + 1,
            "name": f"tier1-history-{run_id}",
            "created_at": created_at,
            "expired": False,
            "archive_download_url": f"https://api.example/history/{run_id}",
            "digest": f"sha256:{'a' * 64}",
        },
        {
            "id": int(run_id) * 10 + 2,
            "name": f"tier1-evidence-{run_id}",
            "created_at": created_at,
            "expired": False,
            "archive_download_url": f"https://api.example/evidence/{run_id}",
            "digest": f"sha256:{'b' * 64}",
        },
    ]


def _history_zip(
    path: Path,
    *,
    run_id: str = "tier1_run_a",
    legacy_absolute: bool = False,
    omit_summary: bool = False,
    history_warnings: list[str] | None = None,
    baseline_evidence_digest: object | None = None,
) -> None:
    prefix = "code/artifacts/history/tier1"
    run_prefix = f"{prefix}/{run_id}"
    paths = {
        "summary_path": f"{run_id}/summary.json",
        "regression_summary_path": f"{run_id}/regression_summary.md",
        "regression_json_path": f"{run_id}/regression_summary.json",
        "trend_snapshot_path": f"{run_id}/trend_snapshot.json",
    }
    if legacy_absolute:
        paths = {
            key: f"/retired/runner/artifacts/history/tier1/{value}" for key, value in paths.items()
        }
    entry = {
        "run_id": run_id,
        "run_accepted": True,
        "baseline_eligible": True,
        "baseline_acceptance": "clean",
        **paths,
    }
    if baseline_evidence_digest is not None:
        entry["baseline_evidence_digest"] = baseline_evidence_digest
    index = {
        "suite_name": "tier1",
        "history_root": "/retired/runner/artifacts/history/tier1" if legacy_absolute else ".",
        "runs": [entry],
    }
    summary = {
        "run_id": run_id,
        "source_git_commit": "a" * 40,
        "source_manifest_git_commit": "a" * 40,
        "source_git_dirty": False,
        "source_result_json": f"/retired/runner/artifacts/runs/{run_id}/results/results.json",
        "source_manifest_json": f"/retired/runner/artifacts/runs/{run_id}/manifest.json",
        "source_markdown_report": f"/retired/runner/artifacts/runs/{run_id}/report.md",
        "targets": [
            {
                "target": "ch01:demo",
                "status": "succeeded",
                "optimization_goal": "performance",
                "baseline_time_ms": 1.0,
                "best_speedup": 2.0,
                "best_optimized_time_ms": 0.5,
                "baseline_file": "/retired/runner/repo/ch01/baseline_demo.py",
                "artifacts": {
                    "nsys_rep": (f"/retired/runner/artifacts/runs/{run_id}/profiles/demo.nsys-rep")
                },
            }
        ],
        "summary": {
            "target_count": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "missing": 0,
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{prefix}/index.json", json.dumps(index))
        if not omit_summary:
            archive.writestr(f"{run_prefix}/summary.json", json.dumps(summary))
        regression_markdown = "# regression\n"
        if history_warnings:
            regression_markdown += "\n## Warnings\n\n" + "\n".join(
                f"- {warning}" for warning in history_warnings
            )
        archive.writestr(f"{run_prefix}/regression_summary.md", regression_markdown)
        archive.writestr(
            f"{run_prefix}/regression_summary.json",
            json.dumps(
                {
                    "regressions": [],
                    "missing_targets": [],
                    "warnings": history_warnings or [],
                }
            ),
        )
        archive.writestr(f"{run_prefix}/trend_snapshot.json", '{"run_count": 1}')


def test_find_latest_history_artifact_uses_failed_run_with_matching_evidence() -> None:
    def _request_json(url: str, token: str) -> dict[str, object]:
        assert token == "token"
        if "/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 30,
                        "status": "completed",
                        "conclusion": "failure",
                        "created_at": "2026-08-16T03:00:00Z",
                    },
                    {
                        "id": 20,
                        "status": "completed",
                        "conclusion": "success",
                        "created_at": "2026-08-16T02:00:00Z",
                    },
                ]
            }
        if "/runs/30/artifacts" in url:
            return {"artifacts": _paired_artifacts("30", created_at="2026-08-16T03:01:00Z")}
        if "/runs/20/artifacts" in url:
            return {"artifacts": _paired_artifacts("20", created_at="2026-08-16T02:01:00Z")}
        raise AssertionError(url)

    artifact = restore.find_latest_history_artifact(
        api_url="https://api.example",
        repository="owner/repo",
        workflow="tier1-nightly.yml",
        branch="main",
        current_run_id=40,
        token="token",
        allow_bootstrap=False,
        request_json=_request_json,
    )

    assert artifact is not None
    assert artifact["workflow_run_id"] == 30
    assert artifact["name"] == "tier1-history-30"
    assert artifact["evidence_artifact_name"] == "tier1-evidence-30"
    assert artifact["workflow_run_conclusion"] == "failure"


def test_find_latest_history_artifact_skips_run_without_evidence_pair() -> None:
    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 30,
                        "status": "completed",
                        "conclusion": "failure",
                        "created_at": "2026-08-16T03:00:00Z",
                    },
                    {
                        "id": 20,
                        "status": "completed",
                        "conclusion": "success",
                        "created_at": "2026-08-16T02:00:00Z",
                    },
                ]
            }
        if "/runs/30/artifacts" in url:
            return {"artifacts": _paired_artifacts("30", created_at="2026-08-16T03:01:00Z")[:1]}
        if "/runs/20/artifacts" in url:
            return {"artifacts": _paired_artifacts("20", created_at="2026-08-16T02:01:00Z")}
        raise AssertionError(url)

    artifact = restore.find_latest_history_artifact(
        api_url="https://api.example",
        repository="owner/repo",
        workflow="tier1-nightly.yml",
        branch="main",
        current_run_id=40,
        token="token",
        allow_bootstrap=False,
        request_json=_request_json,
    )

    assert artifact is not None
    assert artifact["workflow_run_id"] == 20


def test_history_candidates_are_sorted_by_artifact_creation_time() -> None:
    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 30,
                        "status": "completed",
                        "conclusion": "failure",
                        "created_at": "2026-08-16T03:00:00Z",
                    },
                    {
                        "id": 20,
                        "status": "completed",
                        "conclusion": "success",
                        "created_at": "2026-08-16T02:00:00Z",
                    },
                ]
            }
        if "/runs/30/artifacts" in url:
            return {"artifacts": _paired_artifacts("30", created_at="2026-08-16T03:01:00Z")}
        if "/runs/20/artifacts" in url:
            return {"artifacts": _paired_artifacts("20", created_at="2026-08-16T04:01:00Z")}
        raise AssertionError(url)

    candidates = restore.find_history_artifact_candidates(
        api_url="https://api.example",
        repository="owner/repo",
        workflow="tier1-nightly.yml",
        branch="main",
        current_run_id=40,
        token="token",
        request_json=_request_json,
    )

    assert [candidate["workflow_run_id"] for candidate in candidates] == [20, 30]


def test_find_latest_history_artifact_allows_explicit_first_bootstrap() -> None:
    artifact = restore.find_latest_history_artifact(
        api_url="https://api.example",
        repository="owner/repo",
        workflow="tier1-nightly.yml",
        branch="main",
        current_run_id=1,
        token="token",
        allow_bootstrap=True,
        request_json=lambda url, token: {"workflow_runs": []},
    )

    assert artifact is None


def test_evidence_artifact_requires_renewal_at_sixty_days() -> None:
    artifact = {"created_at": "2026-06-17T00:00:00Z"}

    assert restore.evidence_artifact_requires_renewal(
        artifact,
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def test_evidence_artifact_does_not_require_renewal_before_sixty_days() -> None:
    artifact = {"created_at": "2026-06-18T00:00:01Z"}

    assert not restore.evidence_artifact_requires_renewal(
        artifact,
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("created_at", [None, "", "not-a-time", "2026-08-16T00:00:00"])
def test_evidence_artifact_rejects_missing_or_invalid_timestamp(created_at: object) -> None:
    with pytest.raises(restore.HistoryRestoreError, match="timestamp"):
        restore.evidence_artifact_requires_renewal(
            {"created_at": created_at},
            now=datetime(2026, 8, 16, tzinfo=timezone.utc),
        )


def test_find_available_evidence_artifact_binds_workflow_branch_commit_and_digest() -> None:
    expected_sha = "a" * 40

    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/actions/artifacts?" in url:
            return {
                "artifacts": [
                    {
                        "id": 1,
                        "name": "tier1-evidence-tier1_run_a",
                        "expired": False,
                        "created_at": "2026-08-16T03:00:00Z",
                        "archive_download_url": "https://api.example/evidence/1",
                        "digest": f"sha256:{'b' * 64}",
                        "workflow_run": {"id": 100},
                    }
                ]
            }
        if url.endswith("/actions/runs/100"):
            return {
                "path": ".github/workflows/tier1-nightly.yml",
                "head_branch": "main",
                "head_sha": expected_sha,
                "repository": {"full_name": "owner/repo"},
            }
        raise AssertionError(url)

    artifact = restore.find_available_evidence_artifact(
        api_url="https://api.example",
        repository="owner/repo",
        artifact_name="tier1-evidence-tier1_run_a",
        workflow="tier1-nightly.yml",
        branch="main",
        expected_sha=expected_sha,
        token="token",
        request_json=_request_json,
    )

    assert artifact is not None
    assert artifact["id"] == 1


def test_find_available_evidence_artifact_rejects_name_collision_from_other_workflow() -> None:
    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/actions/artifacts?" in url:
            return {
                "artifacts": [
                    {
                        "id": 1,
                        "name": "tier1-evidence-tier1_run_a",
                        "expired": False,
                        "created_at": "2026-08-16T03:00:00Z",
                        "archive_download_url": "https://api.example/evidence/1",
                        "digest": f"sha256:{'b' * 64}",
                        "workflow_run": {"id": 100},
                    }
                ]
            }
        if url.endswith("/actions/runs/100"):
            return {
                "path": ".github/workflows/other.yml",
                "head_branch": "main",
                "head_sha": "a" * 40,
                "repository": {"full_name": "owner/repo"},
            }
        raise AssertionError(url)

    with pytest.raises(restore.HistoryRestoreError, match="provenance"):
        restore.find_available_evidence_artifact(
            api_url="https://api.example",
            repository="owner/repo",
            artifact_name="tier1-evidence-tier1_run_a",
            workflow="tier1-nightly.yml",
            branch="main",
            expected_sha="a" * 40,
            token="token",
            request_json=_request_json,
        )


def test_find_available_evidence_artifact_requires_persisted_digest_match() -> None:
    expected_sha = "a" * 40

    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/actions/artifacts?" in url:
            return {
                "artifacts": [
                    {
                        "id": 1,
                        "name": "tier1-evidence-tier1_run_a",
                        "expired": False,
                        "archive_download_url": "https://api.example/evidence/1",
                        "digest": f"sha256:{'b' * 64}",
                        "workflow_run": {"id": 100},
                    }
                ]
            }
        if url.endswith("/actions/runs/100"):
            return {
                "path": ".github/workflows/tier1-nightly.yml",
                "head_branch": "main",
                "head_sha": expected_sha,
                "repository": {"full_name": "owner/repo"},
            }
        raise AssertionError(url)

    with pytest.raises(restore.HistoryRestoreError, match="persisted digest"):
        restore.find_available_evidence_artifact(
            api_url="https://api.example",
            repository="owner/repo",
            artifact_name="tier1-evidence-tier1_run_a",
            workflow="tier1-nightly.yml",
            branch="main",
            expected_sha=expected_sha,
            expected_digest=f"sha256:{'c' * 64}",
            token="token",
            request_json=_request_json,
        )


def test_history_candidate_scan_is_bounded(monkeypatch) -> None:
    requested_runs: list[int] = []
    monkeypatch.setattr(restore, "MAX_RUNS_TO_SCAN", 3)
    monkeypatch.setattr(restore, "MAX_ARTIFACT_CANDIDATES", 2)

    def _request_json(url: str, token: str) -> dict[str, object]:
        if "/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": run_id,
                        "status": "completed",
                        "created_at": f"2026-08-16T0{run_id}:00:00Z",
                    }
                    for run_id in (3, 2, 1)
                ]
            }
        run_id = int(url.split("/runs/", 1)[1].split("/", 1)[0])
        requested_runs.append(run_id)
        return {
            "artifacts": _paired_artifacts(
                str(run_id),
                created_at=f"2026-08-16T0{run_id}:01:00Z",
            )
        }

    candidates = restore.find_history_artifact_candidates(
        api_url="https://api.example",
        repository="owner/repo",
        workflow="tier1-nightly.yml",
        branch="main",
        current_run_id=4,
        token="token",
        request_json=_request_json,
    )

    assert [candidate["workflow_run_id"] for candidate in candidates] == [3, 2]
    assert requested_runs == [3, 2, 1]
    assert all(candidate["_discovery_truncated"] for candidate in candidates)


def test_restore_history_archive_is_portable(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path)
    destination = tmp_path / "restored" / "tier1"

    restore.restore_history_archive(archive_path, destination)

    index = json.loads((destination / "index.json").read_text(encoding="utf-8"))
    assert index["history_root"] == "."
    assert index["runs"][0]["summary_path"] == "tier1_run_a/summary.json"
    assert (destination / "tier1_run_a" / "summary.json").is_file()


def test_restore_history_archive_returns_persisted_evidence_digest(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    digest = f"sha256:{'b' * 64}"
    _history_zip(archive_path, baseline_evidence_digest=digest)

    evidence_name, git_commit, restored_digest = restore.restore_history_archive(
        archive_path,
        tmp_path / "restored",
    )

    assert evidence_name == "tier1-evidence-tier1_run_a"
    assert git_commit == "a" * 40
    assert restored_digest == digest


def test_restore_history_archive_rejects_malformed_persisted_digest_as_integrity_failure(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path, baseline_evidence_digest=123)

    with pytest.raises(restore.HistoryRestoreError, match="invalid baseline evidence digest"):
        restore.restore_history_archive(archive_path, tmp_path / "restored")


def test_restore_history_archive_migrates_legacy_absolute_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path, legacy_absolute=True)
    destination = tmp_path / "restored"

    restore.restore_history_archive(archive_path, destination)

    index = json.loads((destination / "index.json").read_text(encoding="utf-8"))
    assert index["history_root"] == "."
    entry = index["runs"][0]
    assert entry["summary_path"] == "tier1_run_a/summary.json"
    assert not any(str(value).startswith("/") for value in entry.values())
    summary = json.loads((destination / "tier1_run_a" / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_result_json"] == "tier1_run_a/results/results.json"
    assert summary["source_manifest_json"] == "tier1_run_a/manifest.json"
    assert summary["source_markdown_report"] == "tier1_run_a/report.md"
    assert summary["targets"][0]["baseline_file"] == "ch01/baseline_demo.py"
    assert summary["targets"][0]["artifacts"]["nsys_rep"] == ("tier1_run_a/profiles/demo.nsys-rep")
    assert "/retired/runner" not in json.dumps(summary)


def test_restore_history_archive_rejects_warned_eligible_anchor_without_path_disclosure(
    tmp_path: Path,
) -> None:
    retired_path = "/retired/runner/private/history/index.json"
    archive_path = tmp_path / "history.zip"
    _history_zip(
        archive_path,
        history_warnings=[f"Failed to read {retired_path}"],
    )

    with pytest.raises(restore.HistoryCompatibilityError) as exc_info:
        restore.restore_history_archive(archive_path, tmp_path / "restored")

    assert "contains history warnings" in str(exc_info.value)
    assert retired_path not in str(exc_info.value)


def test_legacy_regression_warning_sanitizers_remove_retired_paths(tmp_path: Path) -> None:
    retired_path = "/retired/runner/private/history/index.json"
    regression_path = tmp_path / "regression_summary.json"
    regression = {
        "regressions": [],
        "missing_targets": [],
        "warnings": [f"Failed to read {retired_path}"],
    }
    regression_path.write_text(json.dumps(regression), encoding="utf-8")

    assert (
        restore._sanitize_restored_regression(
            regression_path,
            regression,
            run_id="tier1_run_a",
        )
        is True
    )
    markdown_path = tmp_path / "regression_summary.md"
    restore._rewrite_regression_markdown(
        markdown_path,
        summary={"run_id": "tier1_run_a"},
        regression=regression,
    )

    sanitized_json = regression_path.read_text(encoding="utf-8")
    sanitized_markdown = markdown_path.read_text(encoding="utf-8")
    assert retired_path not in sanitized_json
    assert retired_path not in sanitized_markdown
    assert "Restored package recorded one or more Tier-1 history warnings" in sanitized_json
    assert "Restored package recorded one or more Tier-1 history warnings" in sanitized_markdown


def test_restore_history_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.json", "{}")

    with pytest.raises(restore.HistoryRestoreError, match="Unsafe path"):
        restore.restore_history_archive(archive_path, tmp_path / "restored")


def test_restore_history_archive_rejects_duplicate_normalized_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("code/artifacts/history/tier1/index.json", "{}")
        archive.writestr("history/tier1/index.json", "{}")

    with pytest.raises(restore.HistoryRestoreError, match="Duplicate path"):
        restore.restore_history_archive(archive_path, tmp_path / "restored")


def test_restore_history_archive_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    link = zipfile.ZipInfo("code/artifacts/history/tier1/index.json")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "target")

    with pytest.raises(restore.HistoryRestoreError, match="symbolic link"):
        restore.restore_history_archive(archive_path, tmp_path / "restored")


def test_restore_history_archive_rejects_nonempty_destination(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path)
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "keep.json").write_text("{}", encoding="utf-8")

    with pytest.raises(restore.HistoryRestoreError, match="not empty"):
        restore.restore_history_archive(archive_path, destination)


def test_restore_history_archive_rejects_dot_only_run_id(tmp_path: Path) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path, run_id=".")

    with pytest.raises(restore.HistoryRestoreError, match="unsafe run id"):
        restore.restore_history_archive(archive_path, tmp_path / "restored")


def test_archive_redirect_does_not_forward_github_token(monkeypatch) -> None:
    observed: dict[str, object] = {}
    headers = Message()
    headers["Location"] = "https://artifact-storage.example/signed"

    class _InitialOpener:
        def open(self, request, timeout):  # noqa: ANN001
            observed["initial_headers"] = dict(request.header_items())
            raise HTTPError(request.full_url, 302, "Found", headers, None)

    def _storage_open(request, timeout):  # noqa: ANN001
        observed["storage_headers"] = dict(request.header_items())
        return io.BytesIO(b"archive")

    monkeypatch.setattr(restore, "build_opener", lambda handler: _InitialOpener())
    monkeypatch.setattr(restore, "urlopen", _storage_open)

    response = restore._open_archive_download(
        "https://api.github.example/artifact",
        "secret-token",
    )
    response.close()

    initial_headers = {
        str(key).lower(): value for key, value in observed["initial_headers"].items()
    }
    storage_headers = {
        str(key).lower(): value for key, value in observed["storage_headers"].items()
    }
    assert initial_headers["authorization"] == "Bearer secret-token"
    assert "authorization" not in storage_headers


def test_main_can_explicitly_bootstrap_after_incompatible_legacy_history(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    archive_path = tmp_path / "legacy.zip"
    _history_zip(archive_path, legacy_absolute=True, omit_summary=True)
    artifact = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "failure",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [artifact],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        shutil.copyfile(archive_path, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(tmp_path / "restored"),
            "--allow-bootstrap",
        ],
    )

    assert restore.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "bootstrap"
    assert payload["history_restored"] is False
    assert payload["rejections"]
    assert not (tmp_path / "restored").exists()


def test_main_can_explicitly_bootstrap_after_successful_history_loses_anchor_evidence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    archive_path = tmp_path / "history.zip"
    _history_zip(archive_path, run_id="20")
    artifact = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "success",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [artifact],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        shutil.copyfile(archive_path, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    monkeypatch.setattr(
        restore,
        "find_available_evidence_artifact",
        lambda **kwargs: None,
    )
    destination = tmp_path / "restored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(destination),
            "--allow-bootstrap",
        ],
    )

    assert restore.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "bootstrap"
    assert payload["history_restored"] is False
    assert payload["rejections"] == [
        "tier1-history-20: Tier-1 baseline evidence artifact is unavailable or expired: "
        "tier1-evidence-20"
    ]
    assert not destination.exists()


def test_main_does_not_bootstrap_after_truncated_candidate_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_path = tmp_path / "incompatible.zip"
    _history_zip(archive_path, run_id="30", omit_summary=True)
    artifact = {
        **_paired_artifacts("30", created_at="2026-08-16T03:00:00Z")[0],
        "evidence_artifact_id": 302,
        "evidence_artifact_name": "tier1-evidence-30",
        "workflow_run_conclusion": "failure",
        "_discovery_truncated": True,
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [artifact],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        shutil.copyfile(archive_path, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    destination = tmp_path / "restored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(destination),
            "--allow-bootstrap",
        ],
    )

    with pytest.raises(restore.HistoryRestoreError, match="bounded discovery limit"):
        restore.main()
    assert not destination.exists()


def test_main_falls_back_to_older_valid_history_candidate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    incompatible_archive = tmp_path / "incompatible.zip"
    _history_zip(incompatible_archive, run_id="30", omit_summary=True)
    valid_archive = tmp_path / "valid.zip"
    _history_zip(valid_archive, run_id="20")
    newest = {
        **_paired_artifacts("30", created_at="2026-08-16T03:00:00Z")[0],
        "evidence_artifact_id": 302,
        "evidence_artifact_name": "tier1-evidence-30",
        "workflow_run_conclusion": "failure",
    }
    older = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "success",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [newest, older],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        source = incompatible_archive if url.endswith("/30") else valid_archive
        shutil.copyfile(source, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    monkeypatch.setattr(
        restore,
        "find_available_evidence_artifact",
        lambda **kwargs: {
            "id": 202,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    destination = tmp_path / "restored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(destination),
        ],
    )

    assert restore.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "restored"
    assert payload["artifact_name"] == "tier1-history-20"
    assert len(payload["rejections"]) == 1
    assert (destination / "20" / "summary.json").is_file()


def test_main_requires_live_evidence_to_match_persisted_digest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    evidence_digest = f"sha256:{'b' * 64}"
    archive_path = tmp_path / "history.zip"
    _history_zip(
        archive_path,
        run_id="20",
        baseline_evidence_digest=evidence_digest,
    )
    artifact = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "success",
    }
    observed: dict[str, object] = {}

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [artifact],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        shutil.copyfile(archive_path, destination)
        return "a" * 64

    def _find_evidence(**kwargs):
        observed.update(kwargs)
        return {
            "id": 202,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "digest": evidence_digest,
        }

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    monkeypatch.setattr(restore, "find_available_evidence_artifact", _find_evidence)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(tmp_path / "restored"),
        ],
    )

    assert restore.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "restored"
    assert observed["artifact_name"] == "tier1-evidence-20"
    assert observed["expected_sha"] == "a" * 40
    assert observed["expected_digest"] == evidence_digest


def test_main_does_not_fallback_or_bootstrap_from_successful_structural_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    incompatible_archive = tmp_path / "incompatible.zip"
    _history_zip(incompatible_archive, run_id="30", omit_summary=True)
    valid_archive = tmp_path / "valid.zip"
    _history_zip(valid_archive, run_id="20")
    newest = {
        **_paired_artifacts("30", created_at="2026-08-16T03:00:00Z")[0],
        "evidence_artifact_id": 302,
        "evidence_artifact_name": "tier1-evidence-30",
        "workflow_run_conclusion": "success",
    }
    older = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "success",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [newest, older],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        source = incompatible_archive if url.endswith("/30") else valid_archive
        shutil.copyfile(source, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    destination = tmp_path / "restored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(destination),
            "--allow-bootstrap",
        ],
    )

    with pytest.raises(restore.HistoryRestoreError, match="successful or unverified"):
        restore.main()
    assert not destination.exists()


def test_main_does_not_bootstrap_or_fallback_after_archive_integrity_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corrupt_archive = tmp_path / "corrupt.zip"
    corrupt_archive.write_bytes(b"not-a-zip")
    valid_archive = tmp_path / "valid.zip"
    _history_zip(valid_archive, run_id="20")
    newest = {
        **_paired_artifacts("30", created_at="2026-08-16T03:00:00Z")[0],
        "evidence_artifact_id": 302,
        "evidence_artifact_name": "tier1-evidence-30",
        "workflow_run_conclusion": "failure",
    }
    older = {
        **_paired_artifacts("20", created_at="2026-08-16T02:00:00Z")[0],
        "evidence_artifact_id": 202,
        "evidence_artifact_name": "tier1-evidence-20",
        "workflow_run_conclusion": "success",
    }

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        restore,
        "find_history_artifact_candidates",
        lambda **kwargs: [newest, older],
    )

    def _copy_archive(url: str, token: str, destination: Path) -> str:
        source = corrupt_archive if url.endswith("/30") else valid_archive
        shutil.copyfile(source, destination)
        return "a" * 64

    monkeypatch.setattr(restore, "_download_archive", _copy_archive)
    destination = tmp_path / "restored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore_tier1_history.py",
            "--repository",
            "owner/repo",
            "--current-run-id",
            "40",
            "--destination",
            str(destination),
            "--allow-bootstrap",
        ],
    )

    with pytest.raises(restore.HistoryRestoreError, match="integrity validation"):
        restore.main()
    assert not destination.exists()
