"""Derive campaign evidence from hashed benchmark result bundles.

The campaign ledger accepts human-authored records for interactive work. This
module is the stricter boundary for unattended work. It parses paired benchmark
result files, validates their embedded run manifests, and derives measurements
and correctness instead of trusting those fields in an experiment draft.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.optimization.campaign import (
    CampaignConfig,
    CaseMeasurement,
    ExperimentRecord,
    sha256_file,
)

EVIDENCE_SCHEMA = "aisp.campaign-benchmark-evidence/v1"
SUPPORTED_METRIC_SOURCES = {"optimized_time_ms", "baseline_time_ms"}
_VOLATILE_MANIFEST_FIELDS = {
    "end_time",
    "duration_seconds",
    "git",
    "seeds",
    "start_time",
    "verify",
}
_VOLATILE_HARDWARE_FIELDS = {
    "fan_speed_pct",
    "gpu_clock_mhz",
    "memory_clock_mhz",
    "power_draw_w",
    "temperature_gpu_c",
    "temperature_memory_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
}


class CampaignEvidenceError(ValueError):
    """Raised when a benchmark evidence bundle fails closed validation."""


@dataclass(frozen=True)
class ParsedRun:
    metric_value: float
    correctness_passed: bool
    timestamp: datetime
    duration_s: float
    git_commit: str
    environment_fingerprint: str
    raw_manifest_hashes: tuple[str, str]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignEvidenceError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise CampaignEvidenceError(f"{label} must be a nonempty JSON array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignEvidenceError(f"{label} must be a nonempty string")
    return value.strip()


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignEvidenceError(f"{label} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise CampaignEvidenceError(f"{label} must be positive and finite")
    return parsed


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignEvidenceError(f"{label} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise CampaignEvidenceError(f"{label} must be finite and not negative")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignEvidenceError(f"{label} must be an integer that is not negative")
    return int(value)


def _parse_timestamp(value: Any, label: str) -> datetime:
    raw = _require_string(value, label)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise CampaignEvidenceError(f"{label} must be an ISO 8601 timestamp") from error


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CampaignEvidenceError(f"could not read {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise CampaignEvidenceError(f"{label} is not valid JSON: {path}") from error
    return _require_dict(payload, label)


def _resolve_hashed_file(reference: Any, base_dir: Path, label: str) -> tuple[Path, str, str]:
    entry = _require_dict(reference, label)
    raw_path = _require_string(entry.get("path"), f"{label}.path")
    expected_hash = _require_string(entry.get("sha256"), f"{label}.sha256").lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise CampaignEvidenceError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    path = (base_dir / raw_path).resolve()
    if not path.is_relative_to(base_dir):
        raise CampaignEvidenceError(f"{label}.path escapes the evidence bundle directory")
    if not path.is_file():
        raise CampaignEvidenceError(f"{label}.path is not a file: {raw_path}")
    actual_hash = sha256_file(path)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise CampaignEvidenceError(f"{label} hash does not match: {raw_path}")
    return path, raw_path, actual_hash


def _only(values: list[Any], label: str) -> Any:
    if len(values) != 1:
        raise CampaignEvidenceError(
            f"{label} must resolve to exactly one entry, found {len(values)}"
        )
    return values[0]


def _validate_manifest(manifest: Any, label: str) -> tuple[dict[str, Any], datetime, float, str]:
    payload = _require_dict(manifest, label)
    if not str(payload.get("schemaVersion") or "").strip():
        raise CampaignEvidenceError(f"{label}.schemaVersion is missing")
    hardware = _require_dict(payload.get("hardware"), f"{label}.hardware")
    software = _require_dict(payload.get("software"), f"{label}.software")
    _require_dict(payload.get("environment"), f"{label}.environment")
    git = _require_dict(payload.get("git"), f"{label}.git")
    config = _require_dict(payload.get("config"), f"{label}.config")
    if payload.get("collection_warnings"):
        raise CampaignEvidenceError(f"{label} contains provenance collection warnings")
    if payload.get("runtime_capability_limitations"):
        raise CampaignEvidenceError(f"{label} contains runtime capability limitations")
    validity_profile = str(config.get("validity_profile") or "").strip().lower()
    if validity_profile != "strict":
        raise CampaignEvidenceError(f"{label} did not use the strict validity profile")
    commit = _require_string(git.get("commit"), f"{label}.git.commit")
    if git.get("dirty") is not False:
        raise CampaignEvidenceError(f"{label} must come from a clean committed worktree")
    if hardware.get("gpu_model"):
        _positive_float(hardware.get("gpu_app_clock_mhz"), f"{label}.hardware.gpu_app_clock_mhz")
        _positive_float(
            hardware.get("memory_app_clock_mhz"),
            f"{label}.hardware.memory_app_clock_mhz",
        )
    _require_string(software.get("python_version"), f"{label}.software.python_version")
    _require_string(software.get("pytorch_version"), f"{label}.software.pytorch_version")
    start_time = _parse_timestamp(payload.get("start_time"), f"{label}.start_time")
    duration = _nonnegative_float(payload.get("duration_seconds", 0.0), f"{label}.duration_seconds")
    return payload, start_time, duration, commit


def _manifest_environment_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value for key, value in manifest.items() if key not in _VOLATILE_MANIFEST_FIELDS
    }
    hardware = dict(_require_dict(payload.get("hardware"), "manifest.hardware"))
    for field_name in _VOLATILE_HARDWARE_FIELDS:
        hardware.pop(field_name, None)
    payload["hardware"] = hardware
    return payload


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_manifest_environment_payload(manifest))).hexdigest()


def _extract_run(
    result_path: Path,
    *,
    target_label: str,
    example: str,
    optimization_file: str,
    metric_source: str,
    role: str,
) -> ParsedRun:
    payload = _load_json_object(result_path, f"{role} benchmark result")
    timestamp = _parse_timestamp(payload.get("timestamp"), f"{role} result.timestamp")
    results = _require_list(payload.get("results"), f"{role} result.results")
    if len(results) != 1:
        raise CampaignEvidenceError(f"{role} result must contain exactly one target result")
    chapter_result = _require_dict(results[0], f"{role} result.results[0]")
    if chapter_result.get("status") != "completed":
        raise CampaignEvidenceError(f"{role} target result status is not completed")
    benchmarks = _require_list(
        chapter_result.get("benchmarks"), f"{role} result.results[0].benchmarks"
    )
    if len(benchmarks) != 1:
        raise CampaignEvidenceError(f"{role} result must contain exactly one benchmark")
    benchmark = _require_dict(benchmarks[0], f"{role} benchmark")
    if benchmark.get("example") != example:
        raise CampaignEvidenceError(
            f"{role} benchmark example {benchmark.get('example')!r} does not match {example!r}"
        )
    optimizations = _require_list(benchmark.get("optimizations"), f"{role} optimizations")
    optimization = _only(
        [
            item
            for item in optimizations
            if isinstance(item, dict) and item.get("file") == optimization_file
        ],
        f"{role} optimization selector",
    )
    optimization = _require_dict(optimization, f"{role} optimization")
    result_status = str(benchmark.get("status") or "")
    optimization_status = str(optimization.get("status") or "")
    recognized_result_statuses = {
        "succeeded",
        "failed_verification",
        "failed_regression",
        "failed_no_speedup",
    }
    recognized_optimization_statuses = {"succeeded", "failed_verification"}
    if (
        result_status not in recognized_result_statuses
        or optimization_status not in recognized_optimization_statuses
    ):
        raise CampaignEvidenceError(
            f"{role} benchmark has unsupported status {result_status!r}/{optimization_status!r}"
        )
    verification = _require_dict(optimization.get("verification"), f"{role} verification")
    input_verification = _require_dict(
        optimization.get("input_verification"), f"{role} input verification"
    )
    if not isinstance(verification.get("passed"), bool):
        raise CampaignEvidenceError(f"{role} verification.passed must be boolean")
    if not isinstance(input_verification.get("passed"), bool):
        raise CampaignEvidenceError(f"{role} input_verification.passed must be boolean")
    correctness_passed = (
        result_status in {"succeeded", "failed_regression", "failed_no_speedup"}
        and optimization_status == "succeeded"
        and verification["passed"] is True
        and input_verification["passed"] is True
    )

    summary = _require_dict(chapter_result.get("summary"), f"{role} result summary")
    total_benchmarks = _nonnegative_int(
        summary.get("total_benchmarks"), f"{role} result summary.total_benchmarks"
    )
    if total_benchmarks != 1:
        raise CampaignEvidenceError(f"{role} result summary must describe one benchmark")
    successful = _nonnegative_int(summary.get("successful"), f"{role} result summary.successful")
    failed = _nonnegative_int(summary.get("failed"), f"{role} result summary.failed")
    if result_status == "succeeded":
        if successful != 1 or failed != 0:
            raise CampaignEvidenceError(
                f"{role} result summary contradicts its succeeded benchmark status"
            )
    else:
        status_count = _nonnegative_int(
            summary.get(result_status, 0),
            f"{role} result summary.{result_status}",
        )
        if successful != 0 or failed != 1 or status_count != 1:
            raise CampaignEvidenceError(
                f"{role} result summary contradicts benchmark status {result_status!r}"
            )
    metric_value = _positive_float(
        optimization.get("time_ms")
        if metric_source == "optimized_time_ms"
        else benchmark.get("baseline_time_ms"),
        f"{role} {metric_source}",
    )

    manifest_entries = _require_list(
        chapter_result.get("manifests"), f"{role} result.results[0].manifests"
    )
    baseline_file = _require_string(benchmark.get("baseline_file"), f"{role} baseline_file")
    baseline_entry = _only(
        [
            item
            for item in manifest_entries
            if isinstance(item, dict)
            and item.get("target_label") == target_label
            and item.get("variant") == "baseline"
            and item.get("file") == baseline_file
        ],
        f"{role} baseline run manifest",
    )
    optimized_entry = _only(
        [
            item
            for item in manifest_entries
            if isinstance(item, dict)
            and item.get("target_label") == target_label
            and item.get("variant") == "optimized"
            and item.get("file") == optimization_file
        ],
        f"{role} optimized run manifest",
    )
    baseline_manifest, baseline_start, baseline_duration, baseline_commit = _validate_manifest(
        _require_dict(baseline_entry, f"{role} baseline entry").get("manifest"),
        f"{role} baseline manifest",
    )
    optimized_manifest, optimized_start, optimized_duration, optimized_commit = _validate_manifest(
        _require_dict(optimized_entry, f"{role} optimized entry").get("manifest"),
        f"{role} optimized manifest",
    )
    if baseline_commit != optimized_commit:
        raise CampaignEvidenceError(
            f"{role} baseline and optimized manifests use different commits"
        )
    baseline_fingerprint = _manifest_fingerprint(baseline_manifest)
    optimized_fingerprint = _manifest_fingerprint(optimized_manifest)
    if baseline_fingerprint != optimized_fingerprint:
        raise CampaignEvidenceError(
            f"{role} baseline and optimized run manifests do not share one environment contract"
        )
    manifest_timestamp = min(baseline_start, optimized_start)
    if abs((timestamp - manifest_timestamp).total_seconds()) > 24 * 60 * 60:
        raise CampaignEvidenceError(f"{role} result timestamp is not tied to its run manifests")
    return ParsedRun(
        metric_value=metric_value,
        correctness_passed=correctness_passed,
        timestamp=manifest_timestamp,
        duration_s=max(baseline_duration, optimized_duration),
        git_commit=baseline_commit,
        environment_fingerprint=baseline_fingerprint,
        raw_manifest_hashes=(
            hashlib.sha256(_canonical_json(baseline_manifest)).hexdigest(),
            hashlib.sha256(_canonical_json(optimized_manifest)).hexdigest(),
        ),
    )


def derive_record_from_evidence_bundle(
    draft: ExperimentRecord,
    config: CampaignConfig,
    evidence_path: Path,
) -> ExperimentRecord:
    """Return a record whose measurement fields come only from validated artifacts."""

    evidence_path = Path(evidence_path).resolve()
    base_dir = evidence_path.parent
    bundle = _load_json_object(evidence_path, "campaign evidence bundle")
    if bundle.get("schema_version") != EVIDENCE_SCHEMA:
        raise CampaignEvidenceError(f"schema_version must be {EVIDENCE_SCHEMA!r}")
    if bundle.get("experiment_id") != draft.experiment_id:
        raise CampaignEvidenceError("evidence experiment_id does not match the draft")
    if bundle.get("metric") != config.primary_metric:
        raise CampaignEvidenceError("evidence metric does not match the frozen campaign metric")
    metric_source = str(bundle.get("metric_source") or "")
    if metric_source not in SUPPORTED_METRIC_SOURCES:
        raise CampaignEvidenceError(
            f"metric_source must be one of {sorted(SUPPORTED_METRIC_SOURCES)}"
        )
    if bundle.get("measurement_protocol") != "paired_interleaved":
        raise CampaignEvidenceError("measurement_protocol must be paired_interleaved")

    workload_path = Path(config.workload_spec)
    environment_path = Path(config.environment_spec)
    if not workload_path.is_file() or sha256_file(workload_path) != config.workload_sha256:
        raise CampaignEvidenceError("frozen workload spec is missing or its hash changed")
    if not environment_path.is_file() or sha256_file(environment_path) != config.environment_sha256:
        raise CampaignEvidenceError("frozen environment spec is missing or its hash changed")

    cases = _require_list(bundle.get("cases"), "evidence cases")
    expected_cases = set(dict.fromkeys([*config.primary_cases, *config.frozen_cases]))
    seen_cases: set[str] = set()
    measurements: dict[str, CaseMeasurement] = {}
    raw_artifacts: list[str] = [evidence_path.name]
    artifact_hashes: dict[str, str] = {evidence_path.name: sha256_file(evidence_path)}
    manifest_hashes: dict[str, list[str]] = {}
    control_commits: set[str] = set()
    candidate_commits: set[str] = set()
    environment_fingerprints: set[str] = set()
    all_correct = True
    total_duration_s = 0.0

    for case_index, case_value in enumerate(cases):
        case = _require_dict(case_value, f"cases[{case_index}]")
        case_id = _require_string(case.get("case_id"), f"cases[{case_index}].case_id")
        if case_id in seen_cases:
            raise CampaignEvidenceError(f"case {case_id!r} is duplicated")
        seen_cases.add(case_id)
        target_label = _require_string(case.get("target_label"), f"case {case_id}.target_label")
        example = _require_string(case.get("example"), f"case {case_id}.example")
        optimization_file = _require_string(
            case.get("optimization_file"), f"case {case_id}.optimization_file"
        )
        pairs = _require_list(case.get("pairs"), f"case {case_id}.pairs")
        control_values: list[float] = []
        candidate_values: list[float] = []
        orders: list[tuple[str, str]] = []
        for pair_index, pair_value in enumerate(pairs, start=1):
            pair = _require_dict(pair_value, f"case {case_id} pair {pair_index}")
            if pair.get("sequence") != pair_index:
                raise CampaignEvidenceError(
                    f"case {case_id} pair sequence must be contiguous and start at 1"
                )
            order_value = pair.get("order")
            if order_value not in (["control", "candidate"], ["candidate", "control"]):
                raise CampaignEvidenceError(
                    f"case {case_id} pair {pair_index} has an invalid execution order"
                )
            order = (str(order_value[0]), str(order_value[1]))
            orders.append(order)
            control_path, control_ref, control_hash = _resolve_hashed_file(
                pair.get("control_result"), base_dir, f"case {case_id} pair {pair_index} control"
            )
            candidate_path, candidate_ref, candidate_hash = _resolve_hashed_file(
                pair.get("candidate_result"),
                base_dir,
                f"case {case_id} pair {pair_index} candidate",
            )
            if control_path == candidate_path:
                raise CampaignEvidenceError(
                    f"case {case_id} pair {pair_index} reuses one file for both roles"
                )
            for reference, digest in ((control_ref, control_hash), (candidate_ref, candidate_hash)):
                if reference in artifact_hashes:
                    raise CampaignEvidenceError(f"benchmark result artifact is reused: {reference}")
                raw_artifacts.append(reference)
                artifact_hashes[reference] = digest
            control_run = _extract_run(
                control_path,
                target_label=target_label,
                example=example,
                optimization_file=optimization_file,
                metric_source=metric_source,
                role="control",
            )
            candidate_run = _extract_run(
                candidate_path,
                target_label=target_label,
                example=example,
                optimization_file=optimization_file,
                metric_source=metric_source,
                role="candidate",
            )
            first_timestamp, second_timestamp = (
                (control_run.timestamp, candidate_run.timestamp)
                if order[0] == "control"
                else (candidate_run.timestamp, control_run.timestamp)
            )
            if first_timestamp >= second_timestamp:
                raise CampaignEvidenceError(
                    f"case {case_id} pair {pair_index} timestamps contradict its declared order"
                )
            if control_run.environment_fingerprint != candidate_run.environment_fingerprint:
                raise CampaignEvidenceError(
                    f"case {case_id} pair {pair_index} changed the benchmark environment contract"
                )
            control_values.append(control_run.metric_value)
            candidate_values.append(candidate_run.metric_value)
            all_correct = (
                all_correct and control_run.correctness_passed and candidate_run.correctness_passed
            )
            total_duration_s += control_run.duration_s + candidate_run.duration_s
            control_commits.add(control_run.git_commit)
            candidate_commits.add(candidate_run.git_commit)
            environment_fingerprints.add(control_run.environment_fingerprint)
            manifest_hashes[control_ref] = list(control_run.raw_manifest_hashes)
            manifest_hashes[candidate_ref] = list(candidate_run.raw_manifest_hashes)
        if len(pairs) > 1:
            order_counts = {order: orders.count(order) for order in set(orders)}
            if max(order_counts.values()) - min(order_counts.values()) > 1 or len(order_counts) < 2:
                raise CampaignEvidenceError(
                    f"case {case_id} must alternate control-first and candidate-first pair order"
                )
        measurements[case_id] = CaseMeasurement(
            control=control_values,
            candidate=candidate_values,
        )

    missing_cases = sorted(expected_cases - seen_cases)
    extra_cases = sorted(seen_cases - expected_cases)
    if missing_cases or extra_cases:
        reasons = []
        if missing_cases:
            reasons.append("missing cases: " + ", ".join(missing_cases))
        if extra_cases:
            reasons.append("undeclared cases: " + ", ".join(extra_cases))
        raise CampaignEvidenceError(
            "evidence case set does not match the campaign: " + "; ".join(reasons)
        )
    if len(control_commits) != 1 or len(candidate_commits) != 1:
        raise CampaignEvidenceError(
            "control and candidate revisions must each stay fixed across all pairs"
        )
    control_commit = next(iter(control_commits))
    candidate_commit = next(iter(candidate_commits))
    if control_commit == candidate_commit:
        raise CampaignEvidenceError(
            "control and candidate benchmark manifests use the same revision"
        )
    if len(environment_fingerprints) != 1:
        raise CampaignEvidenceError("benchmark environment changed across evidence pairs")

    profile_artifacts: list[str] = []
    for profile_index, profile_value in enumerate(bundle.get("profile_artifacts", []) or []):
        _, reference, digest = _resolve_hashed_file(
            profile_value, base_dir, f"profile_artifacts[{profile_index}]"
        )
        if reference in artifact_hashes:
            raise CampaignEvidenceError(f"artifact is listed more than once: {reference}")
        profile_artifacts.append(reference)
        artifact_hashes[reference] = digest

    payload = draft.to_dict()
    payload.update(
        {
            "status": "completed",
            "primary_case": config.primary_cases[0],
            "measurements": {
                case_id: measurement.to_dict() for case_id, measurement in measurements.items()
            },
            "correctness": "passed" if all_correct else "failed",
            "measurement_protocol": "interleaved",
            "raw_artifacts": raw_artifacts,
            "profile_artifacts": profile_artifacts,
            "artifact_sha256": artifact_hashes,
            "duration_s": total_duration_s,
            "workload_sha256": config.workload_sha256,
            "environment_sha256": config.environment_sha256,
        }
    )
    provenance = dict(draft.provenance)
    provenance.update(
        {
            "evidence_adapter": EVIDENCE_SCHEMA,
            "evidence_manifest": evidence_path.name,
            "evidence_manifest_sha256": artifact_hashes[evidence_path.name],
            "benchmark_manifest_sha256": manifest_hashes,
            "benchmark_environment_sha256": next(iter(environment_fingerprints)),
            "control_run_commit": control_commit,
            "candidate_run_commit": candidate_commit,
            "derived_correctness": True,
            "derived_measurements": True,
        }
    )
    payload["provenance"] = provenance
    return ExperimentRecord.from_dict(payload)


def verify_evidence_git_binding(record: ExperimentRecord) -> None:
    """Check that captured source provenance matches the measured clean revisions."""

    control_run_commit = str(record.provenance.get("control_run_commit") or "")
    candidate_run_commit = str(record.provenance.get("candidate_run_commit") or "")
    control_commit = str(record.provenance.get("control_commit") or "")
    candidate_commit = str(record.provenance.get("candidate_commit") or "")
    if control_run_commit != control_commit:
        raise CampaignEvidenceError(
            "control revision captured from Git does not match the control benchmark manifest"
        )
    if candidate_run_commit != candidate_commit:
        raise CampaignEvidenceError(
            "candidate revision captured from Git does not match the candidate benchmark manifest"
        )
