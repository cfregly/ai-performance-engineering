#!/usr/bin/env python3
"""Refresh expectation files from eligible measurements in a saved benchmark run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.analysis.performance_analyzer import (
    has_success_status,
    is_finite_measurement,
    iter_benchmark_rows,
    read_benchmark_json,
    requires_variant_evidence,
)
from core.benchmark.expectations import (
    NON_OVERRIDABLE_ISSUE_TYPES,
    ExpectationEntry,
    ExpectationsStore,
    _slugify,
)
from core.harness.validity_profile import normalize_validity_profile

REBASELINE_STATUSES = {"failed_regression", "failed_no_speedup"}
APPLIED_STATUSES = {"updated", "improved", "regressed", "unchanged"}


@dataclass
class RefreshRecord:
    target: str
    expectation_file: str
    status: str
    message: str
    validity_profile_source: Optional[str] = None
    validity_profile: Optional[str] = None


def _artifact_run_dir(results_json: Path) -> Path:
    return results_json.parent.parent if results_json.parent.name == "results" else results_json.parent


def _load_run_manifest(results_json: Path, run_id: Optional[str], warnings: List[str]) -> dict:
    path = _artifact_run_dir(results_json) / "manifest.json"
    if not path.exists():
        warnings.append(f"No benchmark run manifest at {path}. Run identity is unbound.")
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to read benchmark run manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected JSON object in benchmark run manifest {path}")
    manifest_id = manifest.get("run_id")
    if manifest_id is not None and (not isinstance(manifest_id, str) or not manifest_id.strip()):
        raise ValueError(f"Invalid run_id in benchmark run manifest {path}")
    if run_id is not None and manifest_id is not None and manifest_id != run_id:
        raise ValueError(f"Benchmark run manifest {path} has run_id {manifest_id!r}. Results have {run_id!r}.")
    if not run_id or not manifest_id:
        warnings.append(f"Missing run_id in results or manifest {path}. Run identity is unbound.")
        return {}
    if not isinstance(manifest.get("git", {}), dict) or not isinstance(manifest.get("manifests", []), list):
        raise ValueError(f"Invalid Git or benchmark entries in run manifest {path}")
    for item in manifest.get("manifests", []):
        if not isinstance(item, dict) or not isinstance(item.get("manifest"), dict):
            raise ValueError(f"Invalid benchmark entry in run manifest {path}")
    return manifest


def _load_run_start(results_json: Path, run_id: Optional[str], warnings: List[str]) -> dict:
    run_dir = _artifact_run_dir(results_json)
    event_paths = (
        run_dir / "logs" / "benchmark_events.jsonl",
        results_json.parent / "benchmark_events.jsonl",
    )
    events_path = next((path for path in event_paths if path.exists()), None)
    if events_path is None:
        return {}
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        warnings.append(f"Failed to read benchmark events log {events_path}: {exc}")
        return {}
    run_start = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            warnings.append(f"Failed to parse benchmark events log {events_path}:{line_number}: {exc}")
            return {}
        if not isinstance(event, dict):
            warnings.append(f"Expected JSON object in benchmark events log {events_path}:{line_number}")
            return {}
        if event.get("event_type") != "run_start":
            continue
        if run_id is not None and event.get("run_id") != run_id:
            warnings.append(f"Ignored run_start at {events_path}:{line_number} with a missing or different run_id.")
            continue
        if run_id is not None:
            return event
        if run_start:
            warnings.append(f"Multiple run_start events in {events_path}. Historical results have no run_id.")
            return {}
        run_start = event
    return run_start


def _load_targets(targets: Sequence[str], targets_file: Optional[Path]) -> Optional[Set[str]]:
    loaded = set(targets)
    if targets_file:
        try:
            target_lines = targets_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            raise ValueError(f"Failed to read targets file {targets_file}: {exc}") from exc
        loaded.update(line.strip() for line in target_lines if line.strip())
    return loaded or None


def _chapter_dir(repo_root: Path, chapter: str) -> Path:
    for relative in (chapter, chapter.replace("_", "/")):
        path = (repo_root / relative).resolve()
        if path == repo_root or not path.is_relative_to(repo_root):
            raise ValueError(f"Chapter directory must be inside the repository: {chapter!r}")
        if path.is_dir():
            return path
    raise ValueError(f"Could not resolve chapter directory for {chapter!r}")


def _entry_targets(chapter: str, example: str, example_type: str) -> Set[str]:
    targets = {f"{chapter}:{example}"}
    if example_type == "cuda":
        targets.add(f"{chapter}:{example}_cuda")
    return targets


def _expectation_example_key(example: str, example_type: str) -> str:
    example_type_norm = (example_type or "python").strip().lower()
    if example_type_norm == "python":
        return example
    suffix = f"_{example_type_norm}"
    if example.lower().endswith(suffix):
        return example
    return f"{example}_{example_type_norm}"


def _ineligible_reason(bench: dict, expectation: dict, *, accept_regressions: bool) -> Optional[str]:
    status = bench.get("status")
    if isinstance(status, str):
        status = status.strip().lower()
    if "status" not in bench:
        if requires_variant_evidence(bench):
            return "Current benchmark result is missing its final status"
        if bench.get("error") or bench.get("skip_reason"):
            return "Historical result has an error or skip reason"

    if bench.get("required_profiler_failures") or expectation.get("required_profiler_failures"):
        return "Required profiler captures failed"
    preview_status = expectation.get("status")
    if "status" in expectation and (
        not isinstance(preview_status, str) or preview_status not in APPLIED_STATUSES | {"rejected"}
    ):
        return f"Expectation preview status {preview_status!r} is ineligible for refresh"
    issues = expectation.get("validation_issues", [])
    if not isinstance(issues, list) or any(
        not isinstance(issue, dict) or not isinstance(issue.get("issue_type"), str) for issue in issues
    ):
        return "Malformed expectation validation issues"
    issue_types = {issue["issue_type"] for issue in issues}
    non_overridable = issue_types & NON_OVERRIDABLE_ISSUE_TYPES
    if non_overridable:
        return f"Invalid expectation evidence ({', '.join(sorted(non_overridable))})"

    tracked_metric_dropped = "tracked_metric_missing" in issue_types
    if "status" in bench and not has_success_status(status):
        ordinary_rebaseline = isinstance(status, str) and status in REBASELINE_STATUSES
        throughput_rebaseline = (
            status == "failed_expectation_invalid"
            and preview_status == "rejected"
            and issue_types == {"tracked_metric_missing"}
        )
        if not (ordinary_rebaseline or throughput_rebaseline):
            return f"Benchmark status {status!r} is ineligible for refresh"
        if not accept_regressions:
            return "Rebaselining this result requires accepting regressions"
    if not accept_regressions and ("regression" in issue_types or tracked_metric_dropped):
        return "Rebaselining this result requires accepting regressions"
    if preview_status == "rejected" and (
        not issue_types or issue_types - {"provenance_mismatch", "regression", "tracked_metric_missing"}
    ):
        return "Expectation preview was rejected without an eligible rebaselining or provenance reason"
    return None


def _prepare_entry(
    entry_dict: Dict,
    *,
    bench: dict,
    example_type: str,
    target: str,
    manifest: dict,
    run_start: dict,
    run_start_source: str,
    validity_override: Optional[str],
) -> Tuple[ExpectationEntry, Optional[str], List[str]]:
    for key in ("metrics", "metadata", "provenance"):
        if not isinstance(entry_dict.get(key, {}), dict):
            raise ValueError(f"Expected an object in expectation preview {key}")
    entry = ExpectationEntry.from_dict(entry_dict)
    if entry.example != bench["example"] or entry.type != example_type:
        raise ValueError(f"Expectation preview identity does not match {target} ({example_type})")
    for value in (entry.baseline_time_ms, entry.best_optimized_time_ms):
        if not is_finite_measurement(value) or value <= 0:
            raise ValueError("Expectation preview needs positive finite baseline and optimized times")
    if bench.get("baseline_time_ms") is not None and bench["baseline_time_ms"] != entry.baseline_time_ms:
        raise ValueError("Expectation preview baseline time disagrees with the benchmark row")
    if "optimizations" not in bench and requires_variant_evidence(bench):
        raise ValueError("Current benchmark result is missing its optimization records")
    if "optimizations" in bench:
        optimizations = bench["optimizations"]
        if not isinstance(optimizations, list) or any(not isinstance(opt, dict) for opt in optimizations):
            raise ValueError("Malformed benchmark optimization results")
        if not any(
            has_success_status(opt.get("status"))
            and opt.get("time_ms") == entry.best_optimized_time_ms
            and (not entry.best_optimization_file or opt.get("file") == entry.best_optimization_file)
            for opt in optimizations
        ):
            raise ValueError("Expectation preview has no matching successful optimization measurement")

    provenance = entry.provenance
    if not isinstance(provenance.hardware_key, str) or not re.fullmatch(
        r"[a-z0-9]+(?:_[a-z0-9]+)*", provenance.hardware_key
    ):
        raise ValueError("Expectation preview needs a valid recorded hardware key")
    if provenance.execution_environment not in (None, "", "unknown", "virtualized", "bare_metal"):
        raise ValueError("Expectation preview has an invalid measuring-host execution environment")
    if provenance.dmi_product_name is not None and not isinstance(provenance.dmi_product_name, str):
        raise ValueError("Expectation preview has an invalid measuring-host DMI product name")
    manifest_commits = [manifest.get("git", {}).get("commit")]
    profile_values = [("entry", provenance.validity_profile)]
    if run_start_source == "run_start":
        profile_values.append((run_start_source, run_start.get("validity_profile")))
    model_key = re.sub(r"^[0-9]+x_", "", provenance.hardware_key)
    manifest_variants = set()
    for item in manifest.get("manifests", []):
        if item.get("target_label") != target:
            continue
        expected_file = bench.get("baseline_file") if item.get("variant") == "baseline" else entry.best_optimization_file
        if expected_file and item.get("file") and item["file"] != expected_file:
            continue
        measurement = item["manifest"]
        variant = item.get("variant")
        if variant in ("baseline", "optimized"):
            manifest_variants.add(variant)
        git = measurement.get("git") or {}
        config = measurement.get("config") or {}
        hardware = measurement.get("hardware") or {}
        if any(not isinstance(section, dict) for section in (git, config, hardware)):
            raise ValueError("Malformed measurement provenance in run manifest")
        gpu_model = hardware.get("gpu_model")
        if gpu_model is not None:
            if not isinstance(gpu_model, str):
                raise ValueError("Malformed GPU model in run manifest")
            measured_model_key = _slugify(gpu_model).removeprefix("nvidia_")
            if measured_model_key != "unknown" and model_key != measured_model_key:
                raise ValueError(
                    f"Expectation hardware key {provenance.hardware_key!r} disagrees with "
                    f"the measuring GPU model {gpu_model!r}"
                )
        manifest_commits.append(git.get("commit"))
        profile_values.append(("manifest", config.get("validity_profile")))
    if any(commit and commit != provenance.git_commit for commit in manifest_commits):
        raise ValueError("Expectation preview Git commit disagrees with the run manifest")

    measured_profiles = [
        (source, normalize_validity_profile(value, field_name=f"{source} validity_profile"))
        for source, value in profile_values if value is not None
    ]
    if len({value for _, value in measured_profiles}) > 1:
        raise ValueError("Recorded validity profiles disagree between the expectation preview and run evidence")
    if measured_profiles:
        profile_source, profile = measured_profiles[0]
    elif run_start.get("validity_profile") is not None:
        profile_source = run_start_source
        profile = normalize_validity_profile(run_start["validity_profile"], field_name="run_start validity_profile")
    else:
        profile_source, profile = "user_fallback", validity_override
    if profile == "portable" and run_start.get("allow_portable_expectations_update") is False:
        if run_start_source == "run_start" or profile_source == "legacy_unbound_run_start":
            raise ValueError("Portable run did not enable expectation updates")
    provenance.validity_profile = (
        None if profile_source in {"user_fallback", "legacy_unbound_run_start"} else profile
    )
    # Only the expectation preview records the measuring host. Keep missing
    # fields unknown when the preview predates measuring-host provenance.
    provenance.execution_environment = provenance.execution_environment or "unknown"
    return entry, profile_source if profile is not None else None, sorted(manifest_variants)


def refresh_expectations_from_run(
    *,
    results_json: Path,
    repo_root: Path,
    targets: Optional[Set[str]] = None,
    validity_profile: Optional[str] = None,
    accept_regressions: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    results = read_benchmark_json(results_json)
    rows = list(iter_benchmark_rows(results, results_json))
    repo_root = repo_root.resolve()
    run_id = results.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError(f"Invalid run_id in benchmark results {results_json}")

    warnings: List[str] = []
    run_start = _load_run_start(results_json, run_id, warnings)
    run_start_source = "run_start" if run_id is not None else "legacy_unbound_run_start"
    manifest = _load_run_manifest(results_json, run_id, warnings)
    if validity_profile is not None:
        validity_profile = normalize_validity_profile(validity_profile, field_name="--validity-profile")
    recorded_profile = run_start.get("validity_profile")
    if recorded_profile is not None:
        recorded_profile = normalize_validity_profile(recorded_profile, field_name="run_start validity_profile")
    stores: Dict[Tuple[Path, str], ExpectationsStore] = {}
    updated_files: Dict[Path, ExpectationsStore] = {}
    records: List[RefreshRecord] = []

    for chapter, raw_bench in rows:
        bench = {**raw_bench, "run_id": run_id}
        example = bench["example"]
        example_type = bench.get("type", "python")
        if not isinstance(example_type, str) or not example_type:
            raise ValueError(f"Invalid benchmark type for {chapter}:{example}")
        target = f"{chapter}:{example}"
        if targets is not None and _entry_targets(chapter, example, example_type).isdisjoint(targets):
            continue

        expectation = bench.get("expectation")
        if expectation is None:
            expectation = {}
        if not isinstance(expectation, dict):
            records.append(RefreshRecord(target, "", "rejected", "Malformed expectation preview"))
            continue
        entry_dict = expectation.get("entry")
        reason = _ineligible_reason(bench, expectation, accept_regressions=accept_regressions)
        if reason:
            records.append(RefreshRecord(target, "", "rejected", reason))
            continue
        if entry_dict is None:
            records.append(RefreshRecord(target, "", "skipped", "Missing expectation preview entry in run results"))
            continue
        if not isinstance(entry_dict, dict):
            records.append(RefreshRecord(target, "", "rejected", "Malformed expectation preview entry"))
            continue

        try:
            entry, profile_source, manifest_variants = _prepare_entry(
                entry_dict,
                bench=bench,
                example_type=example_type,
                target=target,
                manifest=manifest,
                run_start=run_start,
                run_start_source=run_start_source,
                validity_override=validity_profile,
            )
        except ValueError as exc:
            records.append(RefreshRecord(target, "", "rejected", str(exc)))
            continue
        if "status" not in bench:
            warnings.append(f"{target}: Benchmark status was unrecorded in the historical result.")
        if "optimizations" not in bench:
            warnings.append(f"{target}: Historical result has no optimization records to verify its preview.")
        if entry.provenance.execution_environment == "unknown":
            warnings.append(f"{target}: Measuring-host execution environment was unrecorded.")

        missing_manifests = {"baseline", "optimized"} - set(manifest_variants)
        if missing_manifests:
            warnings.append(f"{target}: No matching measurement manifest for {', '.join(sorted(missing_manifests))}.")

        try:
            chapter_dir = _chapter_dir(repo_root, chapter)
        except ValueError as exc:
            records.append(RefreshRecord(target, "", "rejected", str(exc)))
            continue
        store_key = (chapter_dir, entry.provenance.hardware_key)
        store = stores.get(store_key)
        if store is None:
            store = ExpectationsStore(
                chapter_dir,
                entry.provenance.hardware_key,
                accept_regressions=accept_regressions,
                allow_mixed_provenance=True,
            )
            if not store.path.resolve().is_relative_to(repo_root):
                raise ValueError(f"Expectation file must be inside the repository: {store.path}")
            stores[store_key] = store

        example_key = _expectation_example_key(entry.example, entry.type)
        try:
            update = store.update_entry(example_key, entry)
        except (TypeError, ValueError) as exc:
            records.append(RefreshRecord(target, "", "rejected", f"Invalid expectation metrics: {exc}"))
            continue
        if update.status in APPLIED_STATUSES:
            metadata = store.data["examples"][example_key]["metadata"]
            if profile_source == "user_fallback":
                metadata["validity_profile_source"] = profile_source
                metadata["validity_profile_override"] = validity_profile
            elif profile_source == "legacy_unbound_run_start":
                metadata["validity_profile_source"] = profile_source
                metadata["legacy_run_start_validity_profile"] = recorded_profile
            issues = {issue["issue_type"] for issue in expectation.get("validation_issues", [])}
            metadata["expectation_refresh"] = {
                "run_id": run_id,
                "benchmark_status": bench.get("status"),
                "preview_status": expectation.get("status"),
                "accept_regressions": accept_regressions,
                "rebaselined": accept_regressions and (
                    update.status == "regressed"
                    or bench.get("status") in REBASELINE_STATUSES
                    or bool(issues & {"regression", "tracked_metric_missing"})
                ),
                "measurement_manifest_variants": manifest_variants,
            }
            updated_files[store.path] = store
        records.append(RefreshRecord(
            target,
            str(store.path.relative_to(repo_root)),
            update.status,
            update.message,
            profile_source,
            entry.provenance.validity_profile,
        ))

    if not dry_run:
        for store in updated_files.values():
            store.save()

    applied = [record for record in records if record.status in APPLIED_STATUSES]
    profiles = {record.validity_profile for record in applied}
    profile_sources = {record.validity_profile_source for record in applied}
    summary = {
        "results_json": str(results_json),
        "dry_run": dry_run,
        "accept_regressions": accept_regressions,
        "run_id": run_id,
        "manifest_run_id_match": bool(manifest),
        "validity_profile": (
            next(iter(profiles)) if len(profiles) == 1
            else recorded_profile if run_start_source == "run_start" else None
        ),
        "legacy_run_start_validity_profile": (
            recorded_profile if run_start_source == "legacy_unbound_run_start" else None
        ),
        "validity_profile_source": (
            next(iter(profile_sources)) if len(profile_sources) == 1 else run_start_source if recorded_profile else None
        ),
        "validity_profile_override": validity_profile,
        "updated_files": [str(path.relative_to(repo_root)) for path in updated_files],
        "records": [asdict(record) for record in records],
        "warnings": warnings,
        "counts": {
            "applied": len(applied),
            "improved": sum(1 for r in records if r.status == "improved"),
            "regressed": sum(1 for r in records if r.status == "regressed"),
            "unchanged": sum(1 for r in records if r.status == "unchanged"),
            "updated": sum(1 for r in records if r.status == "updated"),
            "rejected": sum(1 for r in records if r.status == "rejected"),
            "skipped": sum(1 for r in records if r.status == "skipped"),
            "total": len(records),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh expectations from benchmark run artifacts.")
    parser.add_argument("--results-json", type=Path, required=True, help="Run results JSON (benchmark_test_results.json)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root")
    parser.add_argument("--target", action="append", default=[], help="Specific target(s) to refresh")
    parser.add_argument("--targets-file", type=Path, default=None, help="File containing one target per line")
    parser.add_argument("--validity-profile", default=None, help="Record a user-supplied profile when the historical run has no recorded profile")
    regression_policy = parser.add_mutually_exclusive_group()
    regression_policy.add_argument(
        "--accept-regressions", action="store_true",
        help="Allow eligible slower measurements and dropped throughput metrics to rebaseline expectations",
    )
    regression_policy.add_argument(
        "--no-accept-regressions", action="store_false", dest="accept_regressions",
        help="Reject downward expectation updates (default)",
    )
    parser.set_defaults(accept_regressions=False)
    parser.add_argument("--dry-run", action="store_true", help="Compute updates without writing files")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path to write summary JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = refresh_expectations_from_run(
            results_json=args.results_json,
            repo_root=args.repo_root,
            targets=_load_targets(args.target, args.targets_file),
            validity_profile=args.validity_profile,
            accept_regressions=args.accept_regressions,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
