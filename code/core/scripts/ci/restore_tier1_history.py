#!/usr/bin/env python3
"""Restore the latest trusted Tier-1 history artifact for a canonical run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

API_VERSION = "2026-03-10"
HISTORY_ARTIFACT_PREFIX = "tier1-history-"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_RUN_PAGES = 10
RUNS_PER_PAGE = 100
MAX_RUNS_TO_SCAN = 100
MAX_ARTIFACT_CANDIDATES = 20
ANCHOR_RENEWAL_DAYS = 60
NON_SUCCESS_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "neutral",
    "skipped",
    "stale",
    "startup_failure",
    "timed_out",
}
BASELINE_ACCEPTANCE_VALUES = {
    "accept_history_anchor",
    "accept_regressions",
    "clean",
    "update_expectations",
}
EXPLICIT_BASELINE_ACCEPTANCE_VALUES = {
    "accept_history_anchor",
    "accept_regressions",
    "update_expectations",
}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
ARCHIVE_PREFIXES = (
    ("code", "artifacts", "history", "tier1"),
    ("artifacts", "history", "tier1"),
    ("history", "tier1"),
)


class HistoryRestoreError(RuntimeError):
    """Raised when canonical history cannot be restored safely."""


class HistoryCompatibilityError(HistoryRestoreError):
    """Raised when a trusted legacy package cannot satisfy the portable contract."""


class BaselineEvidenceUnavailableError(HistoryRestoreError):
    """Raised when a valid history package references evidence that no longer exists."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _is_safe_component(value: str) -> bool:
    return value not in {".", ".."} and SAFE_COMPONENT.fullmatch(value) is not None


def _repository_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/", 1)
    if len(parts) != 2 or not all(_is_safe_component(part) for part in parts):
        raise HistoryRestoreError(f"Invalid GitHub repository name: {repository!r}")
    return parts[0], parts[1]


def _request_json(url: str, token: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "ai-performance-engineering-tier1-history",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:
        raise HistoryRestoreError(f"GitHub API request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HistoryRestoreError(
            f"GitHub API returned {type(payload).__name__}, expected an object"
        )
    return payload


def find_history_artifact_candidates(
    *,
    api_url: str,
    repository: str,
    workflow: str,
    branch: str,
    current_run_id: int,
    token: str,
    request_json: Callable[[str, str], dict[str, Any]] = _request_json,
) -> list[dict[str, Any]]:
    owner, repo = _repository_parts(repository)
    prior_runs: list[dict[str, Any]] = []
    for page in range(1, MAX_RUN_PAGES + 1):
        query = urlencode(
            {
                "branch": branch,
                "status": "completed",
                "per_page": RUNS_PER_PAGE,
                "page": page,
            }
        )
        runs_url = (
            f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}/actions/workflows/"
            f"{quote(workflow, safe='')}/runs?{query}"
        )
        runs_payload = request_json(runs_url, token)
        runs = runs_payload.get("workflow_runs")
        if not isinstance(runs, list):
            raise HistoryRestoreError("GitHub workflow-runs response has no workflow_runs list")
        prior_runs.extend(
            run
            for run in runs
            if isinstance(run, dict)
            and isinstance(run.get("id"), int)
            and not isinstance(run.get("id"), bool)
            and run["id"] > 0
            and run["id"] != current_run_id
            and run.get("status") == "completed"
        )
        if len(runs) < RUNS_PER_PAGE:
            break
        if len(prior_runs) > MAX_RUNS_TO_SCAN:
            break
    run_scan_truncated = len(prior_runs) > MAX_RUNS_TO_SCAN
    prior_runs.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
    prior_runs = prior_runs[:MAX_RUNS_TO_SCAN]
    candidates: list[dict[str, Any]] = []
    for run in prior_runs:
        run_id = run["id"]
        artifacts_url = (
            f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}/actions/runs/"
            f"{run_id}/artifacts?per_page=100"
        )
        artifacts_payload = request_json(artifacts_url, token)
        artifacts = artifacts_payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise HistoryRestoreError("GitHub artifacts response has no artifacts list")
        history_candidates = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and isinstance(artifact.get("name"), str)
            and artifact["name"].startswith(HISTORY_ARTIFACT_PREFIX)
            and not artifact.get("expired", False)
            and isinstance(artifact.get("archive_download_url"), str)
            and artifact["archive_download_url"]
        ]
        history_candidates.sort(
            key=lambda artifact: str(artifact.get("created_at") or ""),
            reverse=True,
        )
        for history_artifact in history_candidates:
            history_name = history_artifact["name"]
            evidence_name = f"tier1-evidence-{history_name.removeprefix(HISTORY_ARTIFACT_PREFIX)}"
            evidence_artifact = next(
                (
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                    and artifact.get("name") == evidence_name
                    and not artifact.get("expired", False)
                    and isinstance(artifact.get("archive_download_url"), str)
                    and artifact["archive_download_url"]
                ),
                None,
            )
            if evidence_artifact is not None:
                candidates.append(
                    {
                        **history_artifact,
                        "workflow_run_id": run_id,
                        "evidence_artifact_id": evidence_artifact.get("id"),
                        "evidence_artifact_name": evidence_name,
                        "evidence_artifact_digest": evidence_artifact.get("digest"),
                        "workflow_run_conclusion": run.get("conclusion"),
                        "_discovery_truncated": run_scan_truncated,
                    }
                )
    candidates.sort(
        key=lambda artifact: str(artifact.get("created_at") or ""),
        reverse=True,
    )
    candidate_scan_truncated = len(candidates) >= MAX_ARTIFACT_CANDIDATES
    candidates = candidates[:MAX_ARTIFACT_CANDIDATES]
    if run_scan_truncated or candidate_scan_truncated:
        for candidate in candidates:
            candidate["_discovery_truncated"] = True
    if run_scan_truncated and not candidates:
        raise HistoryRestoreError(
            "Tier-1 history discovery reached its run scan limit without finding a "
            "restorable artifact pair"
        )
    return candidates


def find_latest_history_artifact(
    *,
    api_url: str,
    repository: str,
    workflow: str,
    branch: str,
    current_run_id: int,
    token: str,
    allow_bootstrap: bool,
    request_json: Callable[[str, str], dict[str, Any]] = _request_json,
) -> dict[str, Any] | None:
    candidates = find_history_artifact_candidates(
        api_url=api_url,
        repository=repository,
        workflow=workflow,
        branch=branch,
        current_run_id=current_run_id,
        token=token,
        request_json=request_json,
    )
    if candidates:
        return candidates[0]

    if allow_bootstrap:
        return None
    raise HistoryRestoreError(
        "No prior completed Tier-1 run has an unexpired history artifact. Run one manual "
        "main-branch dispatch with bootstrap_history enabled to establish the canonical "
        "B200 anchor."
    )


def find_available_evidence_artifact(
    *,
    api_url: str,
    repository: str,
    artifact_name: str,
    workflow: str,
    branch: str,
    expected_sha: str,
    expected_digest: str | None = None,
    token: str,
    request_json: Callable[[str, str], dict[str, Any]] = _request_json,
) -> dict[str, Any] | None:
    if expected_digest is not None and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", expected_digest):
        raise HistoryRestoreError("Tier-1 persisted baseline evidence digest is malformed")
    owner, repo = _repository_parts(repository)
    query = urlencode({"name": artifact_name, "per_page": 100})
    artifacts_url = (
        f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}/actions/artifacts?{query}"
    )
    payload = request_json(artifacts_url, token)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise HistoryRestoreError("GitHub artifacts response has no artifacts list")
    candidates = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict)
        and artifact.get("name") == artifact_name
        and not artifact.get("expired", False)
        and artifact.get("archive_download_url")
    ]
    candidates.sort(
        key=lambda artifact: str(artifact.get("created_at") or ""),
        reverse=True,
    )
    expected_workflow_path = f".github/workflows/{workflow}"
    for artifact in candidates:
        workflow_run = artifact.get("workflow_run")
        if not isinstance(workflow_run, dict):
            continue
        try:
            workflow_run_id = int(workflow_run.get("id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if workflow_run_id <= 0:
            continue
        run_url = (
            f"{api_url.rstrip('/')}/repos/{quote(owner)}/{quote(repo)}/actions/runs/"
            f"{workflow_run_id}"
        )
        run = request_json(run_url, token)
        raw_run_path = run.get("path")
        run_path = raw_run_path.split("@", 1)[0] if isinstance(raw_run_path, str) else ""
        run_repository = run.get("repository")
        repository_name = (
            run_repository.get("full_name") if isinstance(run_repository, dict) else None
        )
        digest = artifact.get("digest")
        if (
            run_path != expected_workflow_path
            or run.get("head_branch") != branch
            or run.get("head_sha") != expected_sha
            or repository_name != repository
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest)
        ):
            continue
        if expected_digest is not None and digest.lower() != expected_digest.lower():
            continue
        return artifact
    if candidates:
        raise HistoryRestoreError(
            "Tier-1 evidence artifact provenance does not match the canonical workflow, "
            "branch, commit, repository, and persisted digest contract"
        )
    return None


def evidence_artifact_requires_renewal(
    artifact: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    raw_created_at = artifact.get("created_at")
    if not isinstance(raw_created_at, str) or not raw_created_at.strip():
        raise HistoryRestoreError("Tier-1 baseline evidence has no creation timestamp")
    try:
        created_at = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoryRestoreError(
            "Tier-1 baseline evidence has an invalid creation timestamp"
        ) from exc
    if created_at.tzinfo is None:
        raise HistoryRestoreError("Tier-1 baseline evidence creation timestamp has no timezone")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("Tier-1 renewal reference time must include a timezone")
    age_seconds = (
        current_time.astimezone(timezone.utc) - created_at.astimezone(timezone.utc)
    ).total_seconds()
    if age_seconds < -300:
        raise HistoryRestoreError("Tier-1 baseline evidence creation timestamp is in the future")
    return age_seconds >= ANCHOR_RENEWAL_DAYS * 24 * 60 * 60


def _open_archive_download(url: str, token: str):  # noqa: ANN201
    initial_request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "ai-performance-engineering-tier1-history",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    opener = build_opener(_NoRedirectHandler())
    try:
        return opener.open(initial_request, timeout=60)
    except HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        location = exc.headers.get("Location")
        if not location:
            raise HistoryRestoreError(
                "Tier-1 history download redirect has no Location header"
            ) from exc
        storage_request = Request(
            urljoin(url, location),
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "ai-performance-engineering-tier1-history",
            },
        )
        return urlopen(storage_request, timeout=60)


def _download_archive(url: str, token: str, destination: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with _open_archive_download(url, token) as response, destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise HistoryRestoreError(
                        f"Tier-1 history archive exceeds {MAX_ARCHIVE_BYTES} bytes"
                    )
                digest.update(chunk)
                handle.write(chunk)
    except HistoryRestoreError:
        raise
    except Exception as exc:
        raise HistoryRestoreError(
            f"Failed to download Tier-1 history artifact ({type(exc).__name__})"
        ) from exc
    return digest.hexdigest()


def _normalized_archive_path(name: str) -> PurePosixPath | None:
    raw = PurePosixPath(name)
    if raw.is_absolute() or ".." in raw.parts:
        raise HistoryRestoreError("Unsafe path in Tier-1 history archive")
    parts = tuple(part for part in raw.parts if part not in ("", "."))
    for prefix in ARCHIVE_PREFIXES:
        if parts[: len(prefix)] == prefix:
            parts = parts[len(prefix) :]
            break
    if not parts:
        return None
    if any(not _is_safe_component(part) for part in parts):
        raise HistoryRestoreError("Unsafe component in Tier-1 history archive")
    return PurePosixPath(*parts)


def _load_restored_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HistoryCompatibilityError(
            f"Restored Tier-1 {label} is unreadable ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, dict):
        raise HistoryCompatibilityError(f"Restored Tier-1 {label} must be a JSON object")
    return payload


def _portable_evidence_reference(raw_value: Any, *, run_id: str) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    path = Path(raw_value)
    parts = path.parts
    if path.is_absolute():
        indexes = [index for index, part in enumerate(parts) if part == run_id]
        if not indexes:
            return None
        parts = parts[indexes[-1] :]
    elif run_id in parts:
        parts = parts[parts.index(run_id) :]
    else:
        parts = (run_id, *parts)
    if ".." in parts:
        return None
    normalized = tuple(part for part in parts if part not in ("", "."))
    return PurePosixPath(*normalized).as_posix() if normalized else None


def _portable_source_reference(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    path = Path(raw_value)
    if not path.is_absolute():
        if ".." in path.parts:
            return None
        return PurePosixPath(*path.parts).as_posix()
    for index, part in enumerate(path.parts):
        if part == "labs" or re.fullmatch(r"ch\d+", part):
            return PurePosixPath(*path.parts[index:]).as_posix()
    return None


def _sanitize_restored_summary(
    summary_path: Path,
    summary: dict[str, Any],
    *,
    run_id: str,
) -> None:
    raw_evidence_artifact_name = summary.get("evidence_artifact_name")
    if raw_evidence_artifact_name is None:
        evidence_artifact_name = f"tier1-evidence-{run_id}"
    elif isinstance(raw_evidence_artifact_name, str):
        evidence_artifact_name = raw_evidence_artifact_name
    else:
        raise HistoryCompatibilityError(
            "Restored Tier-1 summary has a non-string evidence artifact name"
        )
    if not re.fullmatch(r"tier1-evidence-[A-Za-z0-9_.-]+", evidence_artifact_name):
        raise HistoryCompatibilityError(
            "Restored Tier-1 summary has an unsafe evidence artifact name"
        )
    summary["evidence_artifact_name"] = evidence_artifact_name
    for key in ("source_result_json", "source_manifest_json", "source_markdown_report"):
        summary[key] = _portable_evidence_reference(summary.get(key), run_id=run_id)
    targets = summary.get("targets")
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            raw_goal = target.get("optimization_goal")
            if raw_goal is not None and not isinstance(raw_goal, str):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 target for {run_id} has invalid optimization goal"
                )
            goal = (raw_goal or "performance").strip().lower()
            if goal == "memory" and target.get("best_optimized_memory_mb") is None:
                baseline_memory = target.get("baseline_memory_mb")
                memory_savings = target.get("best_memory_savings_pct")
                if _is_finite_json_number(baseline_memory) and _is_finite_json_number(
                    memory_savings
                ):
                    optimized_memory = float(baseline_memory) * (
                        1.0 - float(memory_savings) / 100.0
                    )
                    if math.isfinite(optimized_memory) and optimized_memory > 0.0:
                        target["best_optimized_memory_mb"] = optimized_memory
            target["baseline_file"] = _portable_source_reference(target.get("baseline_file"))
            artifacts = target.get("artifacts")
            if isinstance(artifacts, dict):
                target["artifacts"] = {
                    str(key): portable
                    for key, value in artifacts.items()
                    if (portable := _portable_evidence_reference(value, run_id=run_id))
                }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sanitize_recheck_record(record: dict[str, Any]) -> None:
    raw_recheck_run_id = record.get("recheck_run_id")
    recheck_run_id = raw_recheck_run_id if isinstance(raw_recheck_run_id, str) else ""
    if not _is_safe_component(recheck_run_id):
        record["recheck_output_json"] = None
        return
    record["recheck_output_json"] = _portable_evidence_reference(
        record.get("recheck_output_json"),
        run_id=recheck_run_id,
    )


def _sanitize_restored_regression(
    regression_path: Path,
    regression: dict[str, Any],
    *,
    run_id: str,
) -> bool:
    for key in ("anchor_declines", "suppressed_regressions"):
        if key not in regression:
            regression[key] = []
        elif not isinstance(regression[key], list):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 regression summary for {run_id} has malformed {key}"
            )
    had_history_warnings = bool(regression.get("warnings"))
    if had_history_warnings:
        regression["warnings"] = ["Restored package recorded one or more Tier-1 history warnings"]
    if regression.get("regression_rechecks_path"):
        regression["regression_rechecks_path"] = f"{run_id}/regression_rechecks.json"
    rechecks = regression.get("rechecks")
    if isinstance(rechecks, list):
        for record in rechecks:
            if isinstance(record, dict):
                _sanitize_recheck_record(record)
    for key in ("new_targets", "missing_targets"):
        rows = regression.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["baseline_file"] = _portable_source_reference(row.get("baseline_file"))
            artifacts = row.get("artifacts")
            if isinstance(artifacts, dict):
                row["artifacts"] = {
                    str(name): portable
                    for name, value in artifacts.items()
                    if (portable := _portable_evidence_reference(value, run_id=run_id))
                }
    regression_path.write_text(
        json.dumps(regression, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    rechecks_path = regression_path.parent / "regression_rechecks.json"
    if not rechecks_path.is_file():
        return had_history_warnings
    try:
        recheck_payload = json.loads(rechecks_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HistoryCompatibilityError(
            f"Restored Tier-1 recheck records for {run_id} are unreadable "
            f"({type(exc).__name__})"
        ) from exc
    if not isinstance(recheck_payload, list):
        raise HistoryCompatibilityError(
            f"Restored Tier-1 recheck records for {run_id} must be a JSON list"
        )
    for record in recheck_payload:
        if isinstance(record, dict):
            _sanitize_recheck_record(record)
    rechecks_path.write_text(
        json.dumps(recheck_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return had_history_warnings


def _rewrite_regression_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    regression: dict[str, Any],
) -> None:
    from core.analysis.regressions import render_regression_summary

    path.write_text(
        render_regression_summary(summary, {}, regression),
        encoding="utf-8",
    )


def _is_finite_json_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _summary_has_valid_metric_types(summary: dict[str, Any]) -> bool:
    counts = summary.get("summary")
    targets = summary.get("targets")
    if not isinstance(counts, dict) or not isinstance(targets, list):
        return False
    for key in ("target_count", "succeeded", "failed", "skipped", "missing"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    if counts["target_count"] != len(targets):
        return False
    derived_counts = {
        "succeeded": sum(
            isinstance(target, dict) and target.get("status") == "succeeded" for target in targets
        ),
        "failed": sum(
            isinstance(target, dict)
            and isinstance(target.get("status"), str)
            and target["status"].startswith("failed")
            for target in targets
        ),
        "skipped": sum(
            isinstance(target, dict)
            and isinstance(target.get("status"), str)
            and target["status"].startswith("skipped")
            for target in targets
        ),
        "missing": sum(
            isinstance(target, dict) and target.get("status") == "missing" for target in targets
        ),
    }
    if any(counts[key] != value for key, value in derived_counts.items()):
        return False
    if sum(derived_counts.values()) != counts["target_count"]:
        return False
    for key in (
        "avg_speedup",
        "geomean_speedup",
        "max_speedup",
        "median_speedup",
        "representative_speedup",
    ):
        if key in counts and not _is_finite_json_number(counts[key]):
            return False
    for target in targets:
        if not isinstance(target, dict):
            return False
        if not isinstance(target.get("status"), str):
            return False
        for key in (
            "baseline_memory_mb",
            "baseline_p75_ms",
            "baseline_time_ms",
            "best_memory_savings_pct",
            "best_optimized_memory_mb",
            "best_optimized_time_ms",
            "best_speedup",
        ):
            value = target.get(key)
            if value is not None and not _is_finite_json_number(value):
                return False
    return True


def _comparison_has_valid_shape(comparison: dict[str, Any]) -> bool:
    if not isinstance(comparison, dict):
        return False
    for key in ("regressions", "missing_targets"):
        rows = comparison.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return False
    for key in ("anchor_declines", "suppressed_regressions", "rechecks"):
        rows = comparison.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return False
    warnings = comparison.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(row, str) for row in warnings):
        return False
    for key in ("baseline_run_id", "current_run_id"):
        value = comparison.get(key)
        if value is not None and not isinstance(value, str):
            return False
    return True


def _summary_is_eligible_anchor(summary: dict[str, Any]) -> bool:
    if not _summary_has_valid_metric_types(summary):
        return False
    counts = summary.get("summary")
    targets = summary.get("targets")
    if not isinstance(counts, dict) or not isinstance(targets, list) or not targets:
        return False
    try:
        target_count = int(counts.get("target_count", 0) or 0)
        succeeded = int(counts.get("succeeded", 0) or 0)
    except (TypeError, ValueError):
        return False
    if target_count <= 0 or succeeded != target_count or len(targets) != target_count:
        return False
    for target in targets:
        if not isinstance(target, dict) or target.get("status") != "succeeded":
            return False
        try:
            baseline_time = float(target.get("baseline_time_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(baseline_time) or baseline_time <= 0.0:
            return False
        raw_goal = target.get("optimization_goal")
        if raw_goal is not None and not isinstance(raw_goal, str):
            return False
        goal = (raw_goal or "performance").strip().lower()
        metric_name = "best_memory_savings_pct" if goal == "memory" else "best_speedup"
        try:
            metric = float(target.get(metric_name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(metric) or (goal != "memory" and metric <= 0.0):
            return False
        if goal == "memory":
            try:
                optimized_memory = float(target.get("best_optimized_memory_mb", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(optimized_memory) or optimized_memory <= 0.0:
                return False
        else:
            try:
                optimized_time = float(target.get("best_optimized_time_ms", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(optimized_time) or optimized_time <= 0.0:
                return False
    return True


def _summary_has_accepted_provenance(summary: dict[str, Any]) -> bool:
    raw_source_commit = summary.get("source_git_commit")
    raw_manifest_commit = summary.get("source_manifest_git_commit")
    if not isinstance(raw_source_commit, str) or not isinstance(raw_manifest_commit, str):
        return False
    source_commit = raw_source_commit.lower()
    manifest_commit = raw_manifest_commit.lower()
    return (
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None
        and manifest_commit == source_commit
        and summary.get("source_git_dirty") is False
    )


def _validate_restored_index(history_root: Path) -> tuple[str, str, str | None]:
    index_path = history_root / "index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HistoryRestoreError(
            f"Restored Tier-1 index is unreadable ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise HistoryRestoreError("Restored Tier-1 index must contain a runs list")
    if not payload["runs"]:
        raise HistoryRestoreError("Restored Tier-1 index contains no runs")

    root = history_root.resolve()
    migrated = False
    latest_baseline_evidence_name: str | None = None
    latest_baseline_git_commit: str | None = None
    latest_baseline_evidence_digest: str | None = None
    initial_anchor_claimed = False
    for entry in payload["runs"]:
        if not isinstance(entry, dict):
            raise HistoryRestoreError("Restored Tier-1 index contains a non-object run entry")
        raw_run_id = entry.get("run_id")
        if not isinstance(raw_run_id, str) or not _is_safe_component(raw_run_id):
            raise HistoryRestoreError("Restored Tier-1 index has an unsafe run id")
        run_id = raw_run_id
        if "baseline_eligible" in entry and not isinstance(
            entry.get("baseline_eligible"),
            bool,
        ):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} has non-boolean baseline eligibility"
            )
        if "run_accepted" in entry and not isinstance(entry.get("run_accepted"), bool):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} has non-boolean run acceptance"
            )
        acceptance = entry.get("baseline_acceptance")
        if acceptance is not None and acceptance not in BASELINE_ACCEPTANCE_VALUES:
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} has invalid baseline acceptance"
            )
        evidence_digest = entry.get("baseline_evidence_digest")
        if evidence_digest is not None and (
            not isinstance(evidence_digest, str)
            or re.fullmatch(r"sha256:[0-9a-fA-F]{64}", evidence_digest) is None
        ):
            raise HistoryRestoreError(
                f"Restored Tier-1 entry {run_id} has invalid baseline evidence digest"
            )
        if acceptance is not None and entry.get("baseline_eligible") is not True:
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} attaches acceptance to an ineligible run"
            )
        if acceptance in EXPLICIT_BASELINE_ACCEPTANCE_VALUES and any(
            not isinstance(entry.get(key), str) or not entry.get(key, "").strip()
            for key in (
                "baseline_acceptance_actor",
                "baseline_acceptance_note",
                "baseline_acceptance_workflow_run",
            )
        ):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} has incomplete acceptance evidence"
            )
        resolved_paths: dict[str, Path] = {}
        for key in (
            "summary_path",
            "regression_summary_path",
            "regression_json_path",
            "trend_snapshot_path",
        ):
            raw_path = entry.get(key)
            if raw_path is None:
                raise HistoryCompatibilityError(f"Restored Tier-1 entry {run_id} has no {key}")
            if not isinstance(raw_path, str):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 entry {run_id} has non-string {key}"
                )
            path = Path(raw_path)
            if path.is_absolute():
                relocated = root / run_id / path.name
                if not relocated.is_file():
                    raise HistoryCompatibilityError(
                        f"Legacy Tier-1 {key} for {run_id} cannot be relocated"
                    )
                entry[key] = relocated.relative_to(root).as_posix()
                path = Path(entry[key])
                migrated = True
            resolved = (root / path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise HistoryRestoreError(
                    f"Restored Tier-1 {key} for {run_id} escapes the history root"
                ) from exc
            if not resolved.is_file():
                raise HistoryCompatibilityError(f"Restored Tier-1 {key} for {run_id} is missing")
            resolved_paths[key] = resolved

        summary = _load_restored_json(
            resolved_paths["summary_path"],
            label=f"summary for {run_id}",
        )
        if summary.get("run_id") != run_id or not isinstance(summary.get("summary"), dict):
            raise HistoryCompatibilityError(f"Restored Tier-1 summary does not match run {run_id}")
        if not _summary_has_valid_metric_types(summary):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 summary for {run_id} has invalid numeric fields"
            )
        regression = _load_restored_json(
            resolved_paths["regression_json_path"],
            label=f"regression summary for {run_id}",
        )
        if not _comparison_has_valid_shape(regression):
            raise HistoryCompatibilityError(
                f"Restored Tier-1 regression summary for {run_id} is malformed"
            )
        _load_restored_json(
            resolved_paths["trend_snapshot_path"],
            label=f"trend snapshot for {run_id}",
        )
        _sanitize_restored_summary(
            resolved_paths["summary_path"],
            summary,
            run_id=run_id,
        )
        had_history_warnings = _sanitize_restored_regression(
            resolved_paths["regression_json_path"],
            regression,
            run_id=run_id,
        )
        _rewrite_regression_markdown(
            resolved_paths["regression_summary_path"],
            summary=summary,
            regression=regression,
        )
        if entry.get("baseline_eligible") is True and had_history_warnings:
            raise HistoryCompatibilityError(
                f"Restored Tier-1 baseline {run_id} contains history warnings"
            )

        blocking_comparison_changes = bool(
            regression.get("regressions") or regression.get("missing_targets")
        )
        comparison_changes = any(
            regression.get(key)
            for key in (
                "anchor_declines",
                "missing_targets",
                "regressions",
                "suppressed_regressions",
            )
        )
        explicit_acceptance = acceptance in EXPLICIT_BASELINE_ACCEPTANCE_VALUES
        expected_run_accepted = (
            _summary_is_eligible_anchor(summary)
            and _summary_has_accepted_provenance(summary)
            and not had_history_warnings
            and (not blocking_comparison_changes or explicit_acceptance)
        )
        run_accepted = entry.get("run_accepted")
        if run_accepted is None:
            run_accepted = expected_run_accepted
            entry["run_accepted"] = run_accepted
            migrated = True
        elif run_accepted and not expected_run_accepted:
            raise HistoryCompatibilityError(
                f"Restored Tier-1 entry {run_id} has invalid run acceptance"
            )

        eligible = entry.get("baseline_eligible")
        if eligible is None:
            eligible = (
                run_accepted
                and not comparison_changes
                and regression.get("baseline_run_id") is None
                and not initial_anchor_claimed
            )
            entry["baseline_eligible"] = eligible
            migrated = True
        if eligible is True:
            if not run_accepted:
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} was not an accepted run"
                )
            if comparison_changes and acceptance not in EXPLICIT_BASELINE_ACCEPTANCE_VALUES:
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} has unaccepted comparison changes"
                )
            if regression.get("suppressed_regressions"):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} contains a recheck-suppressed regression"
                )
            if regression.get("baseline_run_id") is not None and not explicit_acceptance:
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} has no explicit anchor acceptance"
                )
            if (
                not comparison_changes
                and regression.get("baseline_run_id") is None
                and acceptance is None
            ):
                if initial_anchor_claimed:
                    raise HistoryCompatibilityError(
                        f"Restored Tier-1 baseline {run_id} is a disconnected initial anchor"
                    )
                acceptance = "clean"
                entry["baseline_acceptance"] = acceptance
                migrated = True
            if had_history_warnings:
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} contains history warnings"
                )
            if not _summary_is_eligible_anchor(summary):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} has incomplete metrics"
                )
            evidence_artifact_name = summary.get("evidence_artifact_name")
            source_git_commit = summary.get("source_git_commit")
            if not isinstance(evidence_artifact_name, str) or not isinstance(
                source_git_commit, str
            ):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} has invalid evidence identity"
                )
            latest_baseline_evidence_name = evidence_artifact_name
            if not _summary_has_accepted_provenance(summary):
                raise HistoryCompatibilityError(
                    f"Restored Tier-1 baseline {run_id} has invalid source provenance"
                )
            latest_baseline_git_commit = source_git_commit.lower()
            latest_baseline_evidence_digest = (
                evidence_digest.lower() if evidence_digest is not None else None
            )
            if regression.get("baseline_run_id") is None:
                initial_anchor_claimed = True

    if latest_baseline_evidence_name is None or latest_baseline_git_commit is None:
        raise HistoryCompatibilityError("Restored Tier-1 history contains no eligible baseline")

    if payload.get("history_root") != ".":
        payload["history_root"] = "."
        migrated = True
    if migrated:
        index_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return (
        latest_baseline_evidence_name,
        latest_baseline_git_commit,
        latest_baseline_evidence_digest,
    )


def restore_history_archive(
    archive_path: Path,
    destination: Path,
) -> tuple[str, str, str | None]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise HistoryRestoreError("Tier-1 history destination is not empty")

    with tempfile.TemporaryDirectory(prefix="tier1-history-", dir=destination.parent) as temp_dir:
        extracted_root = Path(temp_dir) / "history"
        extracted_root.mkdir()
        try:
            archive = zipfile.ZipFile(archive_path)
        except Exception as exc:
            raise HistoryRestoreError(
                f"Tier-1 history artifact is not a valid zip ({type(exc).__name__})"
            ) from exc
        with archive:
            members = archive.infolist()
            if len(members) > MAX_MEMBERS:
                raise HistoryRestoreError(
                    f"Tier-1 history archive has too many entries: {len(members)}"
                )
            expanded = sum(member.file_size for member in members)
            if expanded > MAX_EXPANDED_BYTES:
                raise HistoryRestoreError(
                    f"Tier-1 history archive expands beyond {MAX_EXPANDED_BYTES} bytes"
                )
            seen_paths: set[PurePosixPath] = set()
            for member in members:
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise HistoryRestoreError("Tier-1 history archive contains a symbolic link")
                relative = _normalized_archive_path(member.filename)
                if relative is None or member.is_dir():
                    continue
                if relative in seen_paths:
                    raise HistoryRestoreError("Duplicate path in Tier-1 history archive")
                seen_paths.add(relative)
                if relative.name != "index.json" and relative.suffix not in {".json", ".md"}:
                    raise HistoryRestoreError("Unexpected file in Tier-1 history archive")
                output_path = extracted_root.joinpath(*relative.parts)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

        baseline_evidence = _validate_restored_index(extracted_root)
        if destination.exists():
            destination.rmdir()
        os.replace(extracted_root, destination)
    return baseline_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"), required=False)
    parser.add_argument("--workflow", default="tier1-nightly.yml")
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME", "main"))
    parser.add_argument("--current-run-id", type=int, default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--allow-bootstrap", action="store_true")
    parser.add_argument("--allow-anchor-renewal", action="store_true")
    args = parser.parse_args()

    if not args.repository or args.current_run_id is None:
        raise HistoryRestoreError("GitHub repository and current run id are required")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise HistoryRestoreError("GITHUB_TOKEN is required to restore Tier-1 history")

    artifacts = find_history_artifact_candidates(
        api_url=args.api_url,
        repository=args.repository,
        workflow=args.workflow,
        branch=args.branch,
        current_run_id=args.current_run_id,
        token=token,
    )
    if not artifacts:
        if not args.allow_bootstrap:
            raise HistoryRestoreError(
                "No prior completed Tier-1 run has a matching unexpired history and evidence "
                "artifact pair. Run one manual main-branch dispatch with bootstrap_history "
                "enabled to establish the canonical B200 anchor."
            )
        print(json.dumps({"status": "bootstrap", "history_restored": False}))
        return 0

    rejection_reasons: list[str] = []
    restored_artifact: dict[str, Any] | None = None
    discovery_truncated = any(bool(artifact.get("_discovery_truncated")) for artifact in artifacts)
    destination_parent = args.destination.resolve().parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        try:
            with tempfile.TemporaryDirectory(
                prefix="tier1-history-download-",
                dir=destination_parent,
            ) as temp_dir:
                archive_path = Path(temp_dir) / "history.zip"
                archive_download_url = artifact.get("archive_download_url")
                if not isinstance(archive_download_url, str) or not archive_download_url:
                    raise HistoryRestoreError("Tier-1 history artifact has no valid download URL")
                actual_digest = _download_archive(
                    archive_download_url,
                    token,
                    archive_path,
                )
                expected_digest = artifact.get("digest")
                if not isinstance(expected_digest, str):
                    raise HistoryRestoreError(
                        "Tier-1 history artifact has no valid SHA-256 digest metadata"
                    )
                if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", expected_digest):
                    raise HistoryRestoreError(
                        "Tier-1 history artifact has no valid SHA-256 digest metadata"
                    )
                expected_value = expected_digest.split(":", 1)[1].lower()
                if actual_digest != expected_value:
                    raise HistoryRestoreError("Tier-1 history artifact digest mismatch")
                candidate_destination = Path(temp_dir) / "restored"
                (
                    baseline_evidence_name,
                    baseline_git_commit,
                    baseline_evidence_digest,
                ) = restore_history_archive(archive_path, candidate_destination)
                baseline_evidence = find_available_evidence_artifact(
                    api_url=args.api_url,
                    repository=args.repository,
                    artifact_name=baseline_evidence_name,
                    workflow=args.workflow,
                    branch=args.branch,
                    expected_sha=baseline_git_commit,
                    expected_digest=baseline_evidence_digest,
                    token=token,
                )
                if baseline_evidence is None:
                    raise BaselineEvidenceUnavailableError(
                        "Tier-1 baseline evidence artifact is unavailable or expired: "
                        f"{baseline_evidence_name}"
                    )
                anchor_renewal_required = evidence_artifact_requires_renewal(baseline_evidence)
                if anchor_renewal_required and not args.allow_anchor_renewal:
                    raise HistoryRestoreError(
                        "Tier-1 baseline evidence is at least 60 days old. Run a protected "
                        "manual anchor renewal before the 90-day evidence artifact expires."
                    )

                destination = args.destination.resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if not destination.is_dir() or any(destination.iterdir()):
                        raise HistoryRestoreError("Tier-1 history destination is not empty")
                    destination.rmdir()
                os.replace(candidate_destination, destination)
            restored_artifact = artifact
            restored_artifact["anchor_renewal_required"] = anchor_renewal_required
            break
        except BaselineEvidenceUnavailableError as exc:
            conclusion = str(artifact.get("workflow_run_conclusion") or "")
            rejection = f"{artifact.get('name') or artifact.get('id') or '<unknown>'}: {exc}"
            if conclusion == "success":
                if not args.allow_bootstrap:
                    raise HistoryRestoreError(
                        f"Tier-1 history artifact {artifact.get('name') or '<unknown>'} "
                        "references unavailable or expired baseline evidence. Run a protected "
                        "manual dispatch with bootstrap_history enabled."
                    ) from exc
                rejection_reasons.append(rejection)
                if discovery_truncated:
                    raise HistoryRestoreError(
                        "The latest validated Tier-1 history references unavailable baseline "
                        "evidence, but discovery reached its bounded limit. Refusing to reset "
                        "canonical history."
                    ) from exc
                print(
                    json.dumps(
                        {
                            "status": "bootstrap",
                            "history_restored": False,
                            "rejections": rejection_reasons,
                        }
                    )
                )
                return 0
            if conclusion not in NON_SUCCESS_CONCLUSIONS:
                raise HistoryRestoreError(
                    f"Tier-1 history artifact {artifact.get('name') or '<unknown>'} from an "
                    "unverified workflow run references unavailable baseline evidence"
                ) from exc
            rejection_reasons.append(rejection)
        except HistoryCompatibilityError as exc:
            conclusion = str(artifact.get("workflow_run_conclusion") or "")
            if conclusion not in NON_SUCCESS_CONCLUSIONS:
                raise HistoryRestoreError(
                    f"Tier-1 history artifact {artifact.get('name') or '<unknown>'} from a "
                    "successful or unverified workflow run failed structural validation"
                ) from exc
            rejection_reasons.append(
                f"{artifact.get('name') or artifact.get('id') or '<unknown>'}: {exc}"
            )
        except HistoryRestoreError as exc:
            raise HistoryRestoreError(
                f"Tier-1 history artifact {artifact.get('name') or '<unknown>'} "
                f"failed integrity validation: {exc}"
            ) from exc

    if restored_artifact is None:
        if discovery_truncated:
            raise HistoryRestoreError(
                "No candidate Tier-1 history artifact passed validation before the bounded "
                "discovery limit. Refusing to reset canonical history."
            )
        if not args.allow_bootstrap:
            raise HistoryRestoreError(
                "No candidate Tier-1 history artifact passed validation: "
                + " | ".join(rejection_reasons)
            )
        print(
            json.dumps(
                {
                    "status": "bootstrap",
                    "history_restored": False,
                    "rejections": rejection_reasons,
                }
            )
        )
        return 0

    print(
        json.dumps(
            {
                "status": "restored",
                "history_restored": True,
                "artifact_id": restored_artifact.get("id"),
                "artifact_name": restored_artifact.get("name"),
                "evidence_artifact_id": restored_artifact.get("evidence_artifact_id"),
                "evidence_artifact_name": restored_artifact.get("evidence_artifact_name"),
                "anchor_renewal_required": restored_artifact.get("anchor_renewal_required", False),
                "rejections": rejection_reasons,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
