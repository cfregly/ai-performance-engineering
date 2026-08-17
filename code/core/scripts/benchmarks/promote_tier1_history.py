"""Ratify an immutable Tier-1 candidate as the canonical history anchor."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from core.analysis.history_index import (
    SAFE_RUN_ID,
    load_history_index_with_warnings,
    resolve_history_entry_path,
)
from core.analysis.regressions import compare_suite_summaries
from core.analysis.trends import build_trend_snapshot
from core.scripts.ci.restore_tier1_history import (
    _comparison_has_valid_shape,
    _summary_has_accepted_provenance,
    _summary_is_eligible_anchor,
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_suite_contract(path: Path) -> tuple[str, object, list[tuple[str, str]]]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Tier-1 suite config is unreadable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tier-1 suite config must be a mapping")
    suite_name = payload.get("suite_name")
    suite_version = payload.get("version")
    targets = payload.get("targets")
    if not isinstance(suite_name, str) or not suite_name.strip():
        raise ValueError("Tier-1 suite config has no suite name")
    if isinstance(suite_version, bool) or not isinstance(suite_version, int):
        raise ValueError("Tier-1 suite config has no integer version")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Tier-1 suite config has no targets")
    configured_targets: list[tuple[str, str]] = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Tier-1 suite config contains a malformed target")
        key = target.get("key")
        target_name = target.get("target")
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(target_name, str)
            or not target_name.strip()
        ):
            raise ValueError("Tier-1 suite config contains an incomplete target")
        configured_targets.append((key, target_name))
    if len(configured_targets) != len(set(configured_targets)):
        raise ValueError("Tier-1 suite config contains duplicate targets")
    return suite_name, suite_version, configured_targets


def _resolve_evidence_reference(
    evidence_root: Path,
    raw_value: Any,
    *,
    expected_run_id: str | None = None,
) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("Tier-1 candidate has a missing evidence reference")
    relative = Path(raw_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Tier-1 candidate has an unsafe evidence reference")
    if expected_run_id is not None and (not relative.parts or relative.parts[0] != expected_run_id):
        raise ValueError("Tier-1 candidate primary evidence is not bound to its run id")
    root = evidence_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Tier-1 candidate evidence reference escapes its artifact") from exc
    if not resolved.is_file():
        raise ValueError("Tier-1 candidate evidence artifact is incomplete")
    return resolved


def _require_finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Tier-1 immutable result has invalid {label}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Tier-1 immutable result has invalid {label}")
    return parsed


def _optional_finite_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _require_finite_number(value, label=label)


def _validated_workflow_run_url(value: str) -> str:
    workflow_run = value.strip()
    parsed = urlsplit(workflow_run)
    path_parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or len(path_parts) < 5
        or path_parts[-3:-1] != ["actions", "runs"]
        or not path_parts[-1].isdecimal()
    ):
        raise ValueError("Tier-1 promotion requires an exact HTTPS GitHub Actions workflow run URL")
    return workflow_run


def _best_optimization_name(benchmark: dict[str, Any]) -> str | None:
    direct = benchmark.get("best_optimization")
    if direct is not None and not isinstance(direct, str):
        raise ValueError("Tier-1 immutable result has invalid best optimization")
    if direct:
        return direct
    optimizations = benchmark.get("optimizations", [])
    if not isinstance(optimizations, list) or any(
        not isinstance(item, dict) for item in optimizations
    ):
        raise ValueError("Tier-1 immutable result has malformed optimizations")
    if not optimizations:
        return None
    best_entry = max(
        optimizations,
        key=lambda item: _require_finite_number(
            item.get("speedup"),
            label="optimization speedup",
        ),
    )
    name = best_entry.get("technique") or best_entry.get("file") or ""
    if not isinstance(name, str):
        raise ValueError("Tier-1 immutable result has invalid optimization name")
    return name


def _expected_target_from_result(
    benchmark: dict[str, Any],
    *,
    key: str,
    target_name: str,
) -> dict[str, Any]:
    status = benchmark.get("status", "unknown")
    optimization_goal = benchmark.get("optimization_goal")
    if not isinstance(status, str) or (
        optimization_goal is not None and not isinstance(optimization_goal, str)
    ):
        raise ValueError("Tier-1 immutable result has invalid target identity fields")
    baseline_time = _optional_finite_number(
        benchmark.get("baseline_time_ms"), label="baseline_time_ms"
    )
    best_speedup = _optional_finite_number(benchmark.get("best_speedup"), label="best_speedup")
    baseline_memory = _optional_finite_number(
        benchmark.get("baseline_memory_mb"), label="baseline_memory_mb"
    )
    memory_savings = _optional_finite_number(
        benchmark.get("best_memory_savings_pct"),
        label="best_memory_savings_pct",
    )
    baseline_p75 = _optional_finite_number(
        benchmark.get("baseline_p75_ms"), label="baseline_p75_ms"
    )
    return {
        "key": key,
        "target": target_name,
        "status": status,
        "optimization_goal": optimization_goal,
        "baseline_time_ms": baseline_time,
        "best_speedup": best_speedup,
        "best_optimized_time_ms": (
            baseline_time / best_speedup
            if baseline_time is not None
            and best_speedup is not None
            and baseline_time > 0.0
            and best_speedup > 0.0
            else None
        ),
        "best_optimization": _best_optimization_name(benchmark),
        "baseline_memory_mb": baseline_memory,
        "best_memory_savings_pct": memory_savings,
        "best_optimized_memory_mb": (
            baseline_memory * (1.0 - memory_savings / 100.0)
            if baseline_memory is not None and memory_savings is not None and baseline_memory > 0.0
            else None
        ),
        "baseline_p75_ms": baseline_p75,
    }


def _validate_summary_against_result(
    *,
    summary: dict[str, Any],
    result: dict[str, Any],
    configured_targets: list[tuple[str, str]],
    run_id: str,
) -> None:
    if result.get("run_id") != run_id:
        raise ValueError("Tier-1 immutable result run id does not match the candidate")
    chapters = result.get("results")
    if not isinstance(chapters, list) or any(not isinstance(chapter, dict) for chapter in chapters):
        raise ValueError("Tier-1 immutable result has malformed chapter results")
    benchmark_map: dict[tuple[str, Any], dict[str, Any]] = {}
    for chapter in chapters:
        chapter_name = chapter.get("chapter")
        benchmarks = chapter.get("benchmarks")
        if not isinstance(chapter_name, str) or not isinstance(benchmarks, list):
            raise ValueError("Tier-1 immutable result has malformed chapter results")
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict):
                raise ValueError("Tier-1 immutable result has malformed benchmark results")
            example = benchmark.get("example")
            if example is not None and not isinstance(example, str):
                raise ValueError("Tier-1 immutable result has invalid benchmark identity")
            identity = (chapter_name, example)
            if identity in benchmark_map:
                raise ValueError("Tier-1 immutable result has duplicate benchmark identity")
            benchmark_map[identity] = benchmark

    expected_targets: list[dict[str, Any]] = []
    for key, target_name in configured_targets:
        chapter_name, separator, example = target_name.partition(":")
        result_identity = (
            chapter_name.strip().replace("/", "_").replace("-", "_"),
            example.strip() if separator else None,
        )
        benchmark = benchmark_map.get(result_identity)
        if benchmark is None:
            expected_targets.append({"key": key, "target": target_name, "status": "missing"})
        else:
            expected_targets.append(
                _expected_target_from_result(
                    benchmark,
                    key=key,
                    target_name=target_name,
                )
            )

    summary_targets = summary.get("targets")
    if not isinstance(summary_targets, list) or len(summary_targets) != len(expected_targets):
        raise ValueError("Tier-1 candidate summary does not match the immutable result")
    metric_fields = (
        "status",
        "optimization_goal",
        "baseline_time_ms",
        "best_speedup",
        "best_optimized_time_ms",
        "best_optimization",
        "baseline_memory_mb",
        "best_memory_savings_pct",
        "best_optimized_memory_mb",
        "baseline_p75_ms",
    )
    for observed, expected in zip(summary_targets, expected_targets, strict=True):
        if not isinstance(observed, dict) or any(
            observed.get(field) != expected.get(field) for field in metric_fields
        ):
            raise ValueError("Tier-1 candidate summary does not match the immutable result")

    succeeded_speedups = [
        float(target["best_speedup"] or 0.0)
        for target in expected_targets
        if target["status"] == "succeeded"
    ]
    expected_counts: dict[str, int | float] = {
        "target_count": len(expected_targets),
        "succeeded": sum(target["status"] == "succeeded" for target in expected_targets),
        "failed": sum(target["status"].startswith("failed") for target in expected_targets),
        "skipped": sum(target["status"].startswith("skipped") for target in expected_targets),
        "missing": sum(target["status"] == "missing" for target in expected_targets),
        "avg_speedup": (
            sum(succeeded_speedups) / len(succeeded_speedups) if succeeded_speedups else 0.0
        ),
        "median_speedup": statistics.median(succeeded_speedups) if succeeded_speedups else 0.0,
        "geomean_speedup": (
            math.exp(
                sum(math.log(value) for value in succeeded_speedups if value > 0.0)
                / sum(value > 0.0 for value in succeeded_speedups)
            )
            if any(value > 0.0 for value in succeeded_speedups)
            else 0.0
        ),
        "representative_speedup": (
            math.exp(
                sum(math.log(value) for value in succeeded_speedups if value > 0.0)
                / sum(value > 0.0 for value in succeeded_speedups)
            )
            if any(value > 0.0 for value in succeeded_speedups)
            else 0.0
        ),
        "max_speedup": max(succeeded_speedups) if succeeded_speedups else 0.0,
    }
    observed_counts = summary.get("summary")
    if not isinstance(observed_counts, dict) or any(
        observed_counts.get(key) != value for key, value in expected_counts.items()
    ):
        raise ValueError("Tier-1 candidate summary does not match the immutable result")


def _validate_evidence_package(
    *,
    evidence_root: Path,
    history_root: Path,
    summary: dict[str, Any],
    comparison: dict[str, Any],
    expected_git_commit: str,
    configured_targets: list[tuple[str, str]],
    run_id: str,
) -> None:
    manifest_path = _resolve_evidence_reference(
        evidence_root,
        summary.get("source_manifest_json"),
        expected_run_id=run_id,
    )
    result_path = _resolve_evidence_reference(
        evidence_root,
        summary.get("source_result_json"),
        expected_run_id=run_id,
    )
    _resolve_evidence_reference(
        evidence_root,
        summary.get("source_markdown_report"),
        expected_run_id=run_id,
    )
    manifest = _load_json_object(manifest_path, label="Tier-1 evidence manifest")
    result = _load_json_object(result_path, label="Tier-1 immutable result")
    manifest_git = manifest.get("git")
    if (
        not isinstance(manifest_git, dict)
        or not isinstance(manifest_git.get("commit"), str)
        or manifest_git["commit"].lower() != expected_git_commit
        or manifest_git.get("dirty") is not False
    ):
        raise ValueError("Tier-1 evidence manifest does not match the clean expected commit")
    if manifest.get("run_id") != run_id:
        raise ValueError("Tier-1 evidence manifest run id does not match the candidate")
    targets = summary.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Tier-1 candidate target evidence is malformed")
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Tier-1 candidate target evidence is malformed")
        artifacts = target.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("Tier-1 candidate target artifact references are malformed")
        for reference in artifacts.values():
            _resolve_evidence_reference(evidence_root, reference)
    for recheck in comparison.get("rechecks", []):
        reference = recheck.get("recheck_output_json")
        if reference is not None:
            _resolve_evidence_reference(evidence_root, reference)
    recheck_history_reference = comparison.get("regression_rechecks_path")
    if recheck_history_reference is not None:
        if not isinstance(recheck_history_reference, str):
            raise ValueError("Tier-1 recheck history reference is malformed")
        relative = Path(recheck_history_reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Tier-1 recheck history reference is unsafe")
        recheck_path = (history_root.resolve() / relative).resolve()
        try:
            recheck_path.relative_to(history_root.resolve())
        except ValueError as exc:
            raise ValueError("Tier-1 recheck history reference escapes history") from exc
        if not recheck_path.is_file():
            raise ValueError("Tier-1 recheck history evidence is missing")
    _validate_summary_against_result(
        summary=summary,
        result=result,
        configured_targets=configured_targets,
        run_id=run_id,
    )


def promote_tier1_history_anchor(
    *,
    history_root: Path,
    canonical_history_root: Path | None = None,
    output_history_root: Path | None = None,
    evidence_root: Path,
    suite_config: Path,
    run_id: str,
    requester: str,
    note: str,
    workflow_run: str,
    expected_git_commit: str,
    expected_evidence_artifact: str,
    expected_evidence_digest: str,
    allow_bootstrap: bool = False,
) -> dict[str, Any]:
    root = Path(history_root).resolve()
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("Tier-1 promotion run id is unsafe")
    audit_values: dict[str, str] = {}
    for key, value in {
        "requester": requester,
        "note": note,
        "workflow_run": workflow_run,
    }.items():
        if not isinstance(value, str):
            raise ValueError("Tier-1 promotion audit values must be strings")
        audit_values[key] = value.strip()
    if any(not value for value in audit_values.values()):
        raise ValueError("Tier-1 promotion requires requester, note, and workflow run evidence")
    audit_values["workflow_run"] = _validated_workflow_run_url(audit_values["workflow_run"])
    if not isinstance(expected_git_commit, str):
        raise ValueError("Tier-1 promotion requires an exact expected Git commit")
    expected_git_commit = expected_git_commit.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", expected_git_commit) is None:
        raise ValueError("Tier-1 promotion requires an exact expected Git commit")
    if not isinstance(expected_evidence_artifact, str):
        raise ValueError("Tier-1 promotion requires an exact evidence artifact name")
    if re.fullmatch(r"tier1-evidence-[A-Za-z0-9_.-]+", expected_evidence_artifact) is None:
        raise ValueError("Tier-1 promotion requires an exact evidence artifact name")
    if not isinstance(expected_evidence_digest, str):
        raise ValueError("Tier-1 promotion requires an exact evidence artifact digest")
    expected_evidence_digest = expected_evidence_digest.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_evidence_digest):
        expected_evidence_digest = f"sha256:{expected_evidence_digest}"
    if re.fullmatch(r"sha256:[0-9a-f]{64}", expected_evidence_digest) is None:
        raise ValueError("Tier-1 promotion requires an exact evidence artifact digest")
    suite_name, suite_version, configured_targets = _load_suite_contract(suite_config)

    index_path = root / "index.json"
    index, warnings = load_history_index_with_warnings(index_path)
    if warnings:
        raise ValueError("Tier-1 promotion rejected invalid history: " + " | ".join(warnings))
    runs = index.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Tier-1 promotion requires a nonempty history index")
    if runs[-1].get("run_id") != run_id:
        raise ValueError("Tier-1 promotion may ratify only the newest immutable candidate")
    entry = runs[-1]
    if entry.get("baseline_eligible") is True:
        raise ValueError("Tier-1 candidate is already a canonical history anchor")
    if index.get("suite_name") != suite_name or index.get("suite_version") != suite_version:
        raise ValueError("Tier-1 candidate history does not match the checked-out suite config")

    summary_path = resolve_history_entry_path(
        root,
        entry.get("summary_path"),
        run_id=run_id,
    )
    regression_path = resolve_history_entry_path(
        root,
        entry.get("regression_json_path"),
        run_id=run_id,
    )
    trend_path = resolve_history_entry_path(
        root,
        entry.get("trend_snapshot_path"),
        run_id=run_id,
    )
    if summary_path is None or regression_path is None or trend_path is None:
        raise ValueError("Tier-1 candidate is missing required history paths")
    if not summary_path.is_file() or not regression_path.is_file() or not trend_path.is_file():
        raise ValueError("Tier-1 candidate is missing required history files")

    summary = _load_json_object(summary_path, label="Tier-1 candidate summary")
    regression = _load_json_object(regression_path, label="Tier-1 candidate comparison")
    if summary.get("run_id") != run_id:
        raise ValueError("Tier-1 candidate summary run id does not match the index")
    if summary.get("suite_name") != suite_name or summary.get("suite_version") != suite_version:
        raise ValueError("Tier-1 candidate suite identity does not match the checked-out config")
    summary_targets = summary.get("targets")
    if not isinstance(summary_targets, list) or any(
        not isinstance(target, dict) for target in summary_targets
    ):
        raise ValueError("Tier-1 candidate target set is malformed")
    observed_targets: list[tuple[str, str]] = []
    for target in summary_targets:
        key = target.get("key")
        target_name = target.get("target")
        if not isinstance(key, str) or not isinstance(target_name, str):
            raise ValueError("Tier-1 candidate target set is malformed")
        observed_targets.append((key, target_name))
    if observed_targets != configured_targets or len(observed_targets) != len(
        set(observed_targets)
    ):
        raise ValueError("Tier-1 candidate target set does not match the checked-out suite")
    if not _summary_is_eligible_anchor(summary):
        raise ValueError("Tier-1 candidate is incomplete or has invalid benchmark metrics")
    if not _summary_has_accepted_provenance(summary):
        raise ValueError("Tier-1 candidate does not have clean manifest-bound Git provenance")
    source_git_commit = summary.get("source_git_commit")
    if not isinstance(source_git_commit, str) or source_git_commit.lower() != expected_git_commit:
        raise ValueError("Tier-1 candidate Git commit does not match the workflow commit")
    if not _comparison_has_valid_shape(regression):
        raise ValueError("Tier-1 candidate comparison is malformed")
    if regression.get("current_run_id") != run_id:
        raise ValueError("Tier-1 candidate comparison run id does not match the index")
    confirmed_regressions = regression.get("regressions", [])
    suppressed_regressions = regression.get("suppressed_regressions", [])
    warnings = regression.get("warnings", [])
    if suppressed_regressions:
        raise ValueError("Tier-1 candidate contains a regression cleared only by a recheck")
    if warnings:
        raise ValueError("Tier-1 candidate contains history integrity warnings")
    if summary.get("evidence_artifact_name") != expected_evidence_artifact:
        raise ValueError("Tier-1 candidate is not bound to the expected evidence artifact")

    live_root = (
        Path(canonical_history_root).resolve() if canonical_history_root is not None else root
    )
    live_index_path = live_root / "index.json"
    copy_live_history = live_index_path.is_file()
    if canonical_history_root is None:
        live_index = index
        live_runs = runs[:-1]
    elif live_index_path.is_file():
        live_index, live_warnings = load_history_index_with_warnings(live_index_path)
        if live_warnings:
            raise ValueError(
                "Tier-1 promotion rejected invalid live history: " + " | ".join(live_warnings)
            )
        live_runs = live_index.get("runs")
        if not isinstance(live_runs, list):
            raise ValueError("Tier-1 live history has no runs list")
        live_suite_name = live_index.get("suite_name")
        live_suite_version = live_index.get("suite_version")
        if live_suite_name != suite_name:
            raise ValueError("Tier-1 live history does not match the checked-out suite config")
        if live_suite_version != suite_version and not allow_bootstrap:
            raise ValueError("Tier-1 live history does not match the checked-out suite config")
        if live_suite_version != suite_version:
            live_index = {
                "suite_name": suite_name,
                "suite_version": suite_version,
                "history_root": ".",
                "runs": [],
            }
            live_runs = []
            copy_live_history = False
        elif any(live_entry.get("run_id") == run_id for live_entry in live_runs):
            raise ValueError("Tier-1 candidate run id already exists in live history")
    elif allow_bootstrap:
        live_index = {
            "suite_name": suite_name,
            "suite_version": suite_version,
            "history_root": ".",
            "runs": [],
        }
        live_runs = []
    else:
        raise ValueError("Tier-1 promotion could not load live canonical history")

    compatible_prior_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for prior_entry in live_runs:
        if prior_entry.get("baseline_eligible") is not True:
            continue
        prior_run_id = prior_entry.get("run_id")
        if not isinstance(prior_run_id, str) or not SAFE_RUN_ID.fullmatch(prior_run_id):
            raise ValueError("Tier-1 prior anchor has an unsafe run id")
        prior_summary_path = resolve_history_entry_path(
            live_root,
            prior_entry.get("summary_path"),
            run_id=prior_run_id,
        )
        if prior_summary_path is None or not prior_summary_path.is_file():
            raise ValueError("Tier-1 prior anchor summary is missing")
        prior_summary = _load_json_object(
            prior_summary_path,
            label="Tier-1 prior anchor summary",
        )
        if (
            prior_summary.get("suite_name") == suite_name
            and prior_summary.get("suite_version") == suite_version
        ):
            compatible_prior_entries.append((prior_entry, prior_summary))
    expected_baseline_run_id = (
        compatible_prior_entries[-1][0]["run_id"] if compatible_prior_entries else None
    )
    if expected_baseline_run_id is None and not allow_bootstrap:
        raise ValueError(
            "Tier-1 promotion has no compatible prior anchor and bootstrap is disabled"
        )
    if regression.get("baseline_run_id") != expected_baseline_run_id:
        raise ValueError("Tier-1 candidate comparison is not bound to the prior canonical anchor")
    expected_comparison = compare_suite_summaries(
        summary,
        compatible_prior_entries[-1][1] if compatible_prior_entries else None,
    )
    comparison_fields = (
        "baseline_run_id",
        "current_run_id",
        "regressions",
        "improvements",
        "anchor_declines",
        "new_targets",
        "missing_targets",
    )
    if any(regression.get(field) != expected_comparison.get(field) for field in comparison_fields):
        raise ValueError("Tier-1 candidate comparison does not match the canonical summaries")

    _validate_evidence_package(
        evidence_root=Path(evidence_root),
        history_root=root,
        summary=summary,
        comparison=regression,
        expected_git_commit=expected_git_commit,
        configured_targets=configured_targets,
        run_id=run_id,
    )

    updated_entry = {
        **entry,
        "run_accepted": True,
        "baseline_eligible": True,
        "baseline_acceptance": "accept_history_anchor",
        "baseline_acceptance_actor": audit_values["requester"],
        "baseline_acceptance_actor_role": "requester",
        "baseline_acceptance_note": audit_values["note"],
        "baseline_acceptance_workflow_run": audit_values["workflow_run"],
        "baseline_evidence_digest": expected_evidence_digest.lower(),
    }
    # Confirmed regressions can reach this point only through explicit post-benchmark
    # promotion. The public acceptance note records why the anchor moved.
    updated_index = {
        **live_index,
        "history_root": ".",
        "runs": [*live_runs, updated_entry],
    }
    updated_trend = build_trend_snapshot(updated_index)

    if output_history_root is not None:
        output_root = Path(output_history_root).resolve()
        if output_root.exists():
            raise ValueError("Tier-1 promotion output history already exists")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        candidate_run_dir = root / run_id
        if not candidate_run_dir.is_dir():
            raise ValueError("Tier-1 candidate run directory is missing")
        with tempfile.TemporaryDirectory(
            prefix="tier1-promote-",
            dir=output_root.parent,
        ) as temp_dir:
            staging_root = Path(temp_dir) / "history"
            if copy_live_history:
                shutil.copytree(live_root, staging_root)
            else:
                staging_root.mkdir()
            staging_run_dir = staging_root / run_id
            if staging_run_dir.exists():
                raise ValueError("Tier-1 candidate run id collides with live history")
            shutil.copytree(candidate_run_dir, staging_run_dir)
            staging_trend_path = resolve_history_entry_path(
                staging_root,
                updated_entry.get("trend_snapshot_path"),
                run_id=run_id,
            )
            if staging_trend_path is None:
                raise ValueError("Tier-1 candidate has an invalid trend path")
            _atomic_write_json(staging_trend_path, updated_trend)
            _atomic_write_json(staging_root / "index.json", updated_index)
            os.replace(staging_root, output_root)
    else:
        original_trend = trend_path.read_bytes()
        _atomic_write_json(trend_path, updated_trend)
        try:
            _atomic_write_json(index_path, updated_index)
        except Exception:
            temporary_restore: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=trend_path.parent,
                    prefix=f".{trend_path.name}.restore.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(original_trend)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary_restore = Path(handle.name)
                os.replace(temporary_restore, trend_path)
                temporary_restore = None
            finally:
                if temporary_restore is not None:
                    temporary_restore.unlink(missing_ok=True)
            raise

    return {
        "success": True,
        "run_id": run_id,
        "baseline_acceptance": "accept_history_anchor",
        "accepted_regression_count": len(confirmed_regressions),
        "history_root": ".",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--canonical-history-root", type=Path)
    parser.add_argument("--output-history-root", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=Path("configs/benchmark_suites/tier1.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--requester", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--workflow-run", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-evidence-artifact", required=True)
    parser.add_argument("--expected-evidence-digest", required=True)
    parser.add_argument("--allow-bootstrap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = promote_tier1_history_anchor(
            history_root=args.history_root,
            canonical_history_root=args.canonical_history_root,
            output_history_root=args.output_history_root,
            evidence_root=args.evidence_root,
            suite_config=args.suite_config,
            run_id=args.run_id,
            requester=args.requester,
            note=args.note,
            workflow_run=args.workflow_run,
            expected_git_commit=args.expected_git_commit,
            expected_evidence_artifact=args.expected_evidence_artifact,
            expected_evidence_digest=args.expected_evidence_digest,
            allow_bootstrap=args.allow_bootstrap,
        )
    except Exception as exc:
        print(f"Tier-1 history promotion failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
