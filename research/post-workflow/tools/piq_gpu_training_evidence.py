#!/usr/bin/env python3
"""Validate and privately import the current PIQ diagnostic envelope v2.

The closed variants cover Stage 0, two-host CPU functional evidence, and the
fixed TE GPU-training study. The contract is deliberately diagnostic. It
checks retained identities, custody, and complete file bytes. It never executes
a workload or qualifies a scientific result.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any


ENVELOPE = "envelope.json"
IMPORT_RECEIPT = "import-receipt.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
REQUIRED_LAYERS = (
    "host", "accelerator", "fabric", "storage", "orchestration", "runtime", "workload",
)
REQUIRED_ROLES = (
    "execution", "quality", "checkpoint_restart", "telemetry_summary",
    "telemetry_raw_custody_index", "cleanup", "readback", "export",
)
SOURCE_ROLES = (
    "runner", "adapter", "verifier", "evaluator", "checkpoint", "telemetry",
    "helper",
)
SCHEDULE = (
    ("admission-01", ["baseline", "candidate"]),
    ("pair-01", ["baseline", "candidate"]),
    ("pair-02", ["candidate", "baseline"]),
    ("pair-03", ["candidate", "baseline"]),
    ("pair-04", ["baseline", "candidate"]),
)
DESIGN_SCHEDULE = (
    {"id": "admission-01", "kind": "admission", "order": ["baseline", "candidate"]},
    {"id": "pair-01", "kind": "pair", "order": ["baseline", "candidate"]},
    {"id": "pair-02", "kind": "pair", "order": ["candidate", "baseline"]},
    {"id": "pair-03", "kind": "pair", "order": ["candidate", "baseline"]},
    {"id": "pair-04", "kind": "pair", "order": ["baseline", "candidate"]},
)
ENVELOPE_FIELDS = {
    "schema_version", "kind", "scope", "scientific_disposition",
    "experiment_kind", "workload_family", "study_id", "experiment", "stack",
    "objective", "executions", "attempts", "comparison", "correctness",
    "workload_evidence", "detached_objects", "artifacts",
}
EXPERIMENT_KINDS = {
    "checkpoint_write_batching_stage0": "checkpoint_write_batching",
    "cpu_two_host_functional": "cpu_two_host_functional",
    "te_mxfp8_gpu_training": "gpu_training",
}
PIQ_V2_TWO_HOST_ADMISSIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "trust"
    / "piq-v2-two-host-admissions.json"
)
PIQ_V2_TWO_HOST_ADMISSIONS_KIND = "piq_v2_two_host_admissions"
SPECIFIC_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{7,254}$")
GENERIC_IDENTIFIERS = {
    "admission", "authority", "controller", "custodian", "execution-owner",
    "execution_owner", "owner", "reviewer", "source", "system", "unknown",
}
TWO_HOST_ADMISSION_FIELDS = {
    "admission_id", "reviewed_at", "reviewer_id", "source", "study_id",
    "owner", "authority", "envelope_sha256", "owner_attestation_sha256",
    "current_admission_sha256", "run_id", "attempt_ids",
}
TWO_HOST_ADMITTED_TUPLE_FIELDS = (
    "study_id", "owner", "authority", "envelope_sha256",
    "owner_attestation_sha256", "current_admission_sha256", "run_id",
    "attempt_ids",
)


class BundleError(ValueError):
    """Invalid retained diagnostic bundle."""


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise BundleError(f"nonfinite JSON constant: {value}")


def strict_load(path: Path, label: str = "JSON") -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw, object_pairs_hook=_duplicates, parse_constant=_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, BundleError) as exc:
        raise BundleError(f"invalid {label} {path}: {exc}") from exc


def strict_load_jsonl(path: Path, label: str = "JSONL") -> list[Any]:
    values: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise BundleError(f"empty JSONL record at line {line_number}")
                values.append(json.loads(
                    raw, object_pairs_hook=_duplicates, parse_constant=_constant,
                ))
    except (OSError, UnicodeError, json.JSONDecodeError, BundleError) as exc:
        raise BundleError(f"invalid {label} {path}: {exc}") from exc
    if not values:
        raise BundleError(f"invalid {label} {path}: no records")
    return values


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _specific_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and SPECIFIC_IDENTIFIER_RE.fullmatch(value) is not None
        and value.casefold() not in GENERIC_IDENTIFIERS
    )


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def _relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def _fields(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    if set(value) != expected:
        errors.append(f"{label} fields differ (missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)})")
        return False
    return True


def _resolve(bundle: Path, relative: Any, label: str, errors: list[str]) -> Path | None:
    if not _relative(relative):
        errors.append(f"{label}.path must be a safe bundle-relative path")
        return None
    target = bundle
    try:
        for part in PurePosixPath(str(relative)).parts:
            target /= part
            mode = target.lstat().st_mode
            if stat.S_ISLNK(mode):
                errors.append(f"{label} must not use symlinks: {relative}")
                return None
        target.resolve(strict=True).relative_to(bundle.resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        errors.append(f"{label} is missing or escapes the bundle: {relative}")
        return None
    if not target.is_file():
        errors.append(f"{label} must name a regular file: {relative}")
        return None
    return target


def _ref(value: Any, label: str, artifacts: dict[str, dict], errors: list[str]) -> Path | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        errors.append(f"{label} must contain exactly path and sha256")
        return None
    relative, digest = value.get("path"), value.get("sha256")
    if not _relative(relative) or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        errors.append(f"{label} has an unsafe path or invalid SHA-256")
        return None
    indexed = artifacts.get(relative)
    if not isinstance(indexed, dict) or indexed.get("sha256") != digest:
        errors.append(f"{label} is absent from the exact artifact index")
        return None
    return indexed.get("_path")


def _artifact_index(bundle: Path, envelope: dict, errors: list[str]) -> dict[str, dict]:
    entries = envelope.get("artifacts")
    if not isinstance(entries, list) or not entries:
        errors.append("PIQ v2 artifacts must be a nonempty complete index")
        return {}
    result: dict[str, dict] = {}
    folded: set[str] = set()
    for index, item in enumerate(entries):
        label = f"PIQ v2 artifacts[{index}]"
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes", "role"}:
            errors.append(f"{label} fields are invalid")
            continue
        relative = item.get("path")
        if not _relative(relative) or relative in (ENVELOPE, IMPORT_RECEIPT):
            errors.append(f"{label}.path is unsafe or reserved")
            continue
        if relative in result or str(relative).casefold() in folded:
            errors.append(f"PIQ v2 artifact paths duplicate or collide by case: {relative}")
            continue
        digest, size = item.get("sha256"), item.get("size_bytes")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None or not _positive_int(size) or not _text(item.get("role")):
            errors.append(f"{label} requires a digest, positive byte count and role")
            continue
        path = _resolve(bundle, relative, label, errors)
        if path is not None:
            try:
                if path.stat().st_size != size or _digest(path) != digest:
                    errors.append(f"{label} retained size/hash mismatch")
                else:
                    result[str(relative)] = {**item, "_path": path}
                    folded.add(str(relative).casefold())
            except OSError as exc:
                errors.append(f"{label} is unreadable: {exc}")

    actual: set[str] = set()
    for current, directories, files in os.walk(bundle, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            relative = path.relative_to(bundle).as_posix()
            if path.is_symlink():
                errors.append(f"PIQ v2 bundle contains a symlink: {relative}")
            elif path.is_file() and relative not in (ENVELOPE, IMPORT_RECEIPT):
                actual.add(relative)
    indexed_paths = set(result)
    for relative in sorted(actual - indexed_paths):
        errors.append(f"PIQ v2 unindexed retained artifact: {relative}")
    for relative in sorted(indexed_paths - actual):
        errors.append(f"PIQ v2 indexed artifact is missing: {relative}")
    return result


def _load_ref_json(value: Any, label: str, artifacts: dict[str, dict], errors: list[str]) -> Any:
    path = _ref(value, label, artifacts, errors)
    if path is None:
        return None
    try:
        return strict_load(path, label)
    except BundleError as exc:
        errors.append(str(exc))
        return None


def _objective_window(envelope: dict, metric: str, errors: list[str]) -> tuple[datetime, datetime] | None:
    objective = envelope.get("objective")
    fields = {"metric", "unit", "direction", "started_at", "ended_at"}
    if not _fields(objective, fields, "PIQ v2 objective", errors):
        return None
    started = _timestamp(objective.get("started_at"))
    ended = _timestamp(objective.get("ended_at"))
    if ((objective.get("metric"), objective.get("unit"), objective.get("direction"))
            != (metric, "nanoseconds", "minimize")
            or started is None or ended is None or started >= ended):
        errors.append(f"PIQ v2 objective must use ordered {metric} nanoseconds")
        return None
    return started, ended


def _validate_two_host_admissions(document: Any) -> list[str]:
    """Validate the source-owned current-v2 two-host trust anchor."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["PIQ v2 two-host admissions must be a JSON object"]
    if set(document) != {"schema_version", "kind", "entries"}:
        errors.append("PIQ v2 two-host admissions fields are invalid")
    if type(document.get("schema_version")) is not int or document.get("schema_version") != 2:
        errors.append("PIQ v2 two-host admissions schema_version must equal 2")
    if document.get("kind") != PIQ_V2_TWO_HOST_ADMISSIONS_KIND:
        errors.append("PIQ v2 two-host admissions kind is unsupported")
    entries = document.get("entries")
    if not isinstance(entries, list):
        errors.append("PIQ v2 two-host admissions entries must be a list")
        entries = []
    admission_ids: set[str] = set()
    admitted_tuples: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"PIQ v2 two-host admissions entries[{index}]"
        if not isinstance(entry, dict) or set(entry) != TWO_HOST_ADMISSION_FIELDS:
            errors.append(f"{label} fields are invalid")
            continue
        for field in ("admission_id", "reviewer_id", "owner", "authority"):
            if not _specific_identifier(entry.get(field)):
                errors.append(f"{label}.{field} must be a specific non-generic identifier")
        admission_id = entry.get("admission_id")
        if isinstance(admission_id, str):
            folded = admission_id.casefold()
            if folded in admission_ids:
                errors.append("PIQ v2 two-host admission IDs must be distinct")
            admission_ids.add(folded)
        identities = [entry.get(field) for field in ("reviewer_id", "owner", "authority")]
        if all(isinstance(value, str) for value in identities) and len(
                {str(value).casefold() for value in identities}) != 3:
            errors.append(f"{label} reviewer, owner, and authority must be distinct")
        if _timestamp(entry.get("reviewed_at")) is None:
            errors.append(f"{label}.reviewed_at must be a canonical UTC timestamp")
        source = entry.get("source")
        if not isinstance(source, dict) or set(source) != {"id", "sha256"}:
            errors.append(f"{label}.source must contain exactly id and sha256")
        else:
            if not _specific_identifier(source.get("id")):
                errors.append(f"{label}.source.id must be a specific identifier")
            if not isinstance(source.get("sha256"), str) or SHA256_RE.fullmatch(source["sha256"]) is None:
                errors.append(f"{label}.source.sha256 must be a lowercase SHA-256")
        if not _text(entry.get("study_id")) or not _text(entry.get("run_id")):
            errors.append(f"{label} study_id and run_id must be nonempty")
        for field in (
            "envelope_sha256", "owner_attestation_sha256",
            "current_admission_sha256",
        ):
            value = entry.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                errors.append(f"{label}.{field} must be a lowercase SHA-256")
        attempt_ids = entry.get("attempt_ids")
        if (not isinstance(attempt_ids, list) or len(attempt_ids) != 1
                or not _text(attempt_ids[0])):
            errors.append(f"{label}.attempt_ids must contain the exact single attempt")
            attempt_ids = []
        try:
            identity = json.dumps(
                {
                    field: attempt_ids if field == "attempt_ids" else entry.get(field)
                    for field in TWO_HOST_ADMITTED_TUPLE_FIELDS
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            errors.append(f"{label} admitted tuple must contain JSON values")
            continue
        if identity in admitted_tuples:
            errors.append("PIQ v2 two-host admitted tuples must be distinct")
        admitted_tuples.add(identity)
    return errors


def _load_fixed_two_host_admissions(errors: list[str]) -> Any:
    """Load the repository-owned anchor. Bundle and CLI overrides are forbidden."""
    path = PIQ_V2_TWO_HOST_ADMISSIONS_PATH
    try:
        if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
            errors.append(f"missing fixed PIQ v2 two-host admissions trust anchor: {path}")
            return None
        document = strict_load(path, "PIQ v2 two-host admissions")
    except BundleError as exc:
        errors.append(str(exc))
        return None
    errors.extend(_validate_two_host_admissions(document))
    return document


def _require_fixed_two_host_admission(
    document: Any,
    envelope: dict,
    envelope_sha256: str,
    owner_ref: Any,
    admission_ref: Any,
    owner: Any,
    run_id: Any,
    attempt_id: Any,
    objective_end: datetime | None,
    errors: list[str],
) -> None:
    if _validate_two_host_admissions(document):
        return
    if not isinstance(owner, dict):
        return
    expected = {
        "study_id": envelope.get("study_id"),
        "owner": owner.get("owner"),
        "authority": owner.get("authority"),
        "envelope_sha256": envelope_sha256,
        "owner_attestation_sha256": owner_ref.get("sha256") if isinstance(owner_ref, dict) else None,
        "current_admission_sha256": admission_ref.get("sha256") if isinstance(admission_ref, dict) else None,
        "run_id": run_id,
        "attempt_ids": [attempt_id],
    }
    matches = [
        entry for entry in document.get("entries", [])
        if isinstance(entry, dict)
        and all(entry.get(field) == expected.get(field) for field in TWO_HOST_ADMITTED_TUPLE_FIELDS)
    ]
    if len(matches) != 1:
        errors.append(
            "PIQ v2 two-host owner/admission tuple is not admitted by the fixed source-reviewed trust anchor"
        )
        return
    reviewed_at = _timestamp(matches[0].get("reviewed_at"))
    owner_attested_at = _timestamp(owner.get("attested_at"))
    if (reviewed_at is None or owner_attested_at is None or objective_end is None
            or reviewed_at < owner_attested_at or reviewed_at < objective_end):
        errors.append(
            "PIQ v2 two-host source review must occur after owner attestation and objective completion"
        )


def _validate_variant_stack(envelope: dict, layers: tuple[str, ...],
                            errors: list[str]) -> None:
    stack = envelope.get("stack")
    fields = {"piq_revision", "runtime_identifier", "hardware_identifier", "layers"}
    if not _fields(stack, fields, "PIQ v2 stack", errors):
        return
    if (not isinstance(stack.get("piq_revision"), str)
            or REVISION_RE.fullmatch(stack["piq_revision"]) is None
            or not _text(stack.get("runtime_identifier"))
            or not _text(stack.get("hardware_identifier"))
            or stack.get("layers") != [
                {"name": name, "state": "observed"} for name in layers
            ]):
        errors.append("PIQ v2 stack does not match the experiment-kind layer census")


def _validate_variant_experiment(envelope: dict, artifacts: dict[str, dict],
                                 errors: list[str]) -> None:
    experiment = envelope.get("experiment")
    fields = {
        "id", "question", "hypothesis", "workload_scope", "design",
        "source_artifacts", "input",
    }
    if not _fields(experiment, fields, "PIQ v2 diagnostic experiment", errors):
        return
    if (experiment.get("id") != envelope.get("study_id")
            or any(not _text(experiment.get(field))
                   for field in ("question", "hypothesis", "workload_scope"))):
        errors.append("PIQ v2 diagnostic experiment identity or question is invalid")
    _ref(experiment.get("design"), "PIQ v2 diagnostic design", artifacts, errors)
    _ref(experiment.get("input"), "PIQ v2 diagnostic input", artifacts, errors)
    sources = experiment.get("source_artifacts")
    if (not isinstance(sources, list) or not sources
            or len({item.get("path") for item in sources if isinstance(item, dict)})
            != len(sources)):
        errors.append("PIQ v2 diagnostic experiment needs distinct source artifacts")
    else:
        for index, ref in enumerate(sources):
            _ref(ref, f"PIQ v2 diagnostic source_artifacts[{index}]", artifacts, errors)


def _validate_retention_document(
    envelope: dict,
    artifacts: dict[str, dict],
    role: str,
    ref: Any,
    run_id: str,
    attempt_ids: list[str],
    objective_end: datetime | None,
    errors: list[str],
    two_host_pids: list[int] | None = None,
) -> None:
    document = _load_ref_json(ref, f"PIQ v2 {role}", artifacts, errors)
    timestamp_field = "completed_at" if role == "cleanup" else "verified_at"
    fields = {
        "schema", "study_id", "run_id", "attempt_ids", timestamp_field,
        "complete",
    }
    if role != "cleanup":
        fields.add("immutable")
    elif two_host_pids is not None:
        fields.update({"pids", "workers_terminated", "transport_closed"})
    if not _fields(document, fields, f"PIQ v2 {role} document", errors):
        return
    expected_schema = f"piq.{envelope.get('experiment_kind')}-{role}.v2"
    if ((document.get("schema"), document.get("study_id"), document.get("run_id"),
         document.get("attempt_ids"), document.get("complete"))
            != (expected_schema, envelope.get("study_id"), run_id, attempt_ids, True)):
        errors.append(f"PIQ v2 {role} document does not bind the exact experiment census")
    if role != "cleanup" and document.get("immutable") is not True:
        errors.append(f"PIQ v2 {role} document must be immutable")
    if (role == "cleanup" and two_host_pids is not None
            and (document.get("pids") != two_host_pids
                 or document.get("workers_terminated") is not True
                 or document.get("transport_closed") is not True)):
        errors.append("PIQ v2 two-host cleanup must close both workers and transport")
    observed = _timestamp(document.get(timestamp_field))
    if observed is None or (objective_end is not None and observed < objective_end):
        errors.append(f"PIQ v2 {role} document must be verified after the objective")


def _validate_stage0(envelope: dict, artifacts: dict[str, dict],
                     errors: list[str]) -> None:
    _validate_variant_experiment(envelope, artifacts, errors)
    _validate_variant_stack(
        envelope, ("storage", "orchestration", "runtime", "workload"), errors,
    )
    window = _objective_window(envelope, "complete_stage0_elapsed", errors)
    objective_end = window[1] if window else None
    if envelope.get("scientific_disposition") != "DIAGNOSTIC_ONLY":
        errors.append("PIQ v2 Stage 0 must remain DIAGNOSTIC_ONLY")

    executions = envelope.get("executions")
    execution_fields = {"run_id", "order", "attempt_ids"}
    if not isinstance(executions, list) or len(executions) != 1:
        errors.append("PIQ v2 Stage 0 requires exactly one execution")
        executions = []
    execution = executions[0] if executions else {}
    if executions and not _fields(execution, execution_fields, "PIQ v2 Stage 0 execution", errors):
        execution = {}

    attempts = envelope.get("attempts")
    attempt_fields = {
        "id", "run_id", "role", "status", "accepted", "failure_reason",
        "raw_result",
    }
    if not isinstance(attempts, list) or len(attempts) != 2:
        errors.append("PIQ v2 Stage 0 requires exactly two arm attempts")
        attempts = []
    roles: list[Any] = []
    attempt_ids: list[str] = []
    for index, attempt in enumerate(attempts):
        if not _fields(attempt, attempt_fields, f"PIQ v2 Stage 0 attempts[{index}]", errors):
            continue
        roles.append(attempt.get("role"))
        attempt_ids.append(attempt.get("id"))
        if (attempt.get("run_id") != execution.get("run_id")
                or (attempt.get("status"), attempt.get("accepted"),
                    attempt.get("failure_reason")) != ("succeeded", True, None)
                or not _text(attempt.get("id"))):
            errors.append("PIQ v2 Stage 0 attempts must be accepted exact-run rows")
        raw = _load_ref_json(
            attempt.get("raw_result"), f"PIQ v2 Stage 0 raw attempt[{index}]",
            artifacts, errors,
        )
        raw_fields = {
            "schema", "study_id", "run_id", "attempt_id", "role", "status",
            "accepted",
        }
        if _fields(raw, raw_fields, f"PIQ v2 Stage 0 raw attempt[{index}]", errors):
            expected = (
                "piq.checkpoint-write-batching-stage0-attempt.v2",
                envelope.get("study_id"), execution.get("run_id"), attempt.get("id"),
                attempt.get("role"), "succeeded", True,
            )
            actual = tuple(raw.get(field) for field in (
                "schema", "study_id", "run_id", "attempt_id", "role", "status",
                "accepted",
            ))
            if actual != expected:
                errors.append("PIQ v2 Stage 0 raw attempt identity drifted")
    if roles != ["baseline", "candidate"] or len(set(attempt_ids)) != 2:
        errors.append("PIQ v2 Stage 0 arm order must be baseline then candidate")
    if execution and (
        execution.get("order") != ["baseline", "candidate"]
        or execution.get("attempt_ids") != attempt_ids
        or not _text(execution.get("run_id"))
    ):
        errors.append("PIQ v2 Stage 0 execution does not bind both ordered arms")

    comparison = envelope.get("comparison")
    comparison_fields = {
        "baseline_attempt_id", "candidate_attempt_id", "scientific_disposition",
        "reason",
    }
    if _fields(comparison, comparison_fields, "PIQ v2 Stage 0 comparison", errors):
        if (comparison.get("baseline_attempt_id") != (attempt_ids[0] if len(attempt_ids) == 2 else None)
                or comparison.get("candidate_attempt_id") != (attempt_ids[1] if len(attempt_ids) == 2 else None)
                or comparison.get("scientific_disposition") != "DIAGNOSTIC_ONLY"
                or not _text(comparison.get("reason"))):
            errors.append("PIQ v2 Stage 0 comparison does not bind both arms")

    correctness = envelope.get("correctness")
    correctness_fields = {"accepted", "oracle", "baseline", "candidate"}
    if _fields(correctness, correctness_fields, "PIQ v2 Stage 0 correctness", errors):
        oracle = _load_ref_json(correctness.get("oracle"), "PIQ v2 Stage 0 oracle", artifacts, errors)
        baseline = _load_ref_json(correctness.get("baseline"), "PIQ v2 Stage 0 baseline correctness", artifacts, errors)
        candidate = _load_ref_json(correctness.get("candidate"), "PIQ v2 Stage 0 candidate correctness", artifacts, errors)
        oracle_fields = {"schema", "study_id", "input_sha256", "expected_output_sha256"}
        result_fields = {
            "schema", "study_id", "run_id", "attempt_id", "role",
            "input_sha256", "output_sha256", "accepted",
        }
        oracle_ok = _fields(oracle, oracle_fields, "PIQ v2 Stage 0 oracle", errors)
        results_ok = all(
            _fields(value, result_fields, f"PIQ v2 Stage 0 {role} correctness", errors)
            for role, value in (("baseline", baseline), ("candidate", candidate))
        )
        if correctness.get("accepted") is not True:
            errors.append("PIQ v2 Stage 0 requires accepted exact two-arm correctness")
        if oracle_ok and (
            oracle.get("schema"), oracle.get("study_id")
        ) != (
            "piq.checkpoint-write-batching-stage0-oracle.v2",
            envelope.get("study_id"),
        ):
            errors.append("PIQ v2 Stage 0 oracle schema/study identity drifted")
        if oracle_ok and results_ok:
            for index, (role, value) in enumerate((("baseline", baseline), ("candidate", candidate))):
                if (value.get("schema") != "piq.checkpoint-write-batching-stage0-correctness.v2"
                        or value.get("study_id") != envelope.get("study_id")
                        or value.get("run_id") != execution.get("run_id")
                        or value.get("attempt_id") != (attempt_ids[index] if len(attempt_ids) == 2 else None)
                        or value.get("role") != role or value.get("accepted") is not True
                        or value.get("input_sha256") != oracle.get("input_sha256")
                        or value.get("output_sha256") != oracle.get("expected_output_sha256")):
                    errors.append(f"PIQ v2 Stage 0 {role} correctness differs from the exact oracle")

    evidence = envelope.get("workload_evidence")
    evidence_roles = {"correctness", "cleanup", "custody", "readback", "export"}
    if not isinstance(evidence, dict) or set(evidence) != evidence_roles:
        errors.append("PIQ v2 Stage 0 workload evidence roles are incomplete")
    else:
        if isinstance(correctness, dict) and evidence.get("correctness") != correctness.get("oracle"):
            errors.append("PIQ v2 Stage 0 correctness role must equal the exact oracle ref")
        run_id = execution.get("run_id") if isinstance(execution, dict) else None
        for role in ("cleanup", "custody", "readback", "export"):
            _validate_retention_document(
                envelope, artifacts, role, evidence[role], run_id, attempt_ids,
                objective_end, errors,
            )
    if envelope.get("detached_objects") != []:
        errors.append("PIQ v2 Stage 0 does not admit detached objects")


def _validate_two_host(envelope: dict, artifacts: dict[str, dict],
                       admissions: Any, envelope_sha256: str,
                       errors: list[str]) -> None:
    _validate_variant_experiment(envelope, artifacts, errors)
    _validate_variant_stack(
        envelope,
        ("host", "fabric", "storage", "orchestration", "runtime", "workload"),
        errors,
    )
    window = _objective_window(envelope, "functional_two_host_elapsed", errors)
    objective_start, objective_end = window if window else (None, None)
    if envelope.get("scientific_disposition") != "DIAGNOSTIC_ONLY":
        errors.append("PIQ v2 two-host functional evidence must remain DIAGNOSTIC_ONLY")

    executions = envelope.get("executions")
    execution_fields = {
        "run_id", "attempt_ids", "owner_attestation", "current_admission",
    }
    if not isinstance(executions, list) or len(executions) != 1:
        errors.append("PIQ v2 two-host evidence requires exactly one execution")
        executions = []
    execution = executions[0] if executions else {}
    if executions and not _fields(execution, execution_fields, "PIQ v2 two-host execution", errors):
        execution = {}

    attempts = envelope.get("attempts")
    attempt_fields = {
        "id", "run_id", "status", "accepted", "failure_reason", "raw_result",
        "ranks", "partition_rehash", "transport",
    }
    if not isinstance(attempts, list) or len(attempts) != 1:
        errors.append("PIQ v2 two-host evidence requires exactly one attempt")
        attempts = []
    attempt = attempts[0] if attempts else {}
    if attempts and not _fields(attempt, attempt_fields, "PIQ v2 two-host attempt", errors):
        attempt = {}
    attempt_id = attempt.get("id")
    run_id = execution.get("run_id")
    if (not _text(run_id) or execution.get("attempt_ids") != [attempt_id]
            or attempt.get("run_id") != run_id
            or (attempt.get("status"), attempt.get("accepted"),
                attempt.get("failure_reason")) != ("succeeded", True, None)
            or not _text(attempt_id)):
        errors.append("PIQ v2 two-host execution/attempt census is invalid")
    raw = _load_ref_json(attempt.get("raw_result"), "PIQ v2 two-host raw attempt", artifacts, errors)
    raw_fields = {"schema", "study_id", "run_id", "attempt_id", "status", "accepted"}
    if _fields(raw, raw_fields, "PIQ v2 two-host raw attempt", errors):
        if tuple(raw.get(field) for field in raw_fields) != tuple({
            "schema": "piq.cpu-two-host-attempt.v2", "study_id": envelope.get("study_id"),
            "run_id": run_id, "attempt_id": attempt_id, "status": "succeeded",
            "accepted": True,
        }.get(field) for field in raw_fields):
            errors.append("PIQ v2 two-host raw attempt identity drifted")

    ranks = attempt.get("ranks")
    rank_fields = {
        "rank", "local_rank", "pid", "physical_host_sha256", "boot_id_sha256",
        "output_ref",
    }
    if not isinstance(ranks, list) or len(ranks) != 2:
        errors.append("PIQ v2 two-host attempt requires exactly two ranks")
        ranks = []
    pids: dict[int, int] = {}
    hosts: list[Any] = []
    boots: list[Any] = []
    for index, rank in enumerate(ranks):
        if not _fields(rank, rank_fields, f"PIQ v2 two-host ranks[{index}]", errors):
            continue
        if rank.get("rank") != index or rank.get("local_rank") != 0 or not _positive_int(rank.get("pid")):
            errors.append("PIQ v2 two-host rank/local-rank/PID census is invalid")
        for field in ("physical_host_sha256", "boot_id_sha256"):
            value = rank.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                errors.append(f"PIQ v2 two-host rank {field} must be a SHA-256 identity")
        output = _load_ref_json(
            rank.get("output_ref"), f"PIQ v2 two-host rank {index} output",
            artifacts, errors,
        )
        output_fields = {
            "schema", "study_id", "run_id", "attempt_id", "rank", "pid",
            "physical_host_sha256", "boot_id_sha256",
        }
        if _fields(output, output_fields, f"PIQ v2 two-host rank {index} output", errors):
            expected_output = {
                "schema": "piq.cpu-two-host-rank-output.v2",
                "study_id": envelope.get("study_id"), "run_id": run_id,
                "attempt_id": attempt_id, "rank": rank.get("rank"),
                "pid": rank.get("pid"),
                "physical_host_sha256": rank.get("physical_host_sha256"),
                "boot_id_sha256": rank.get("boot_id_sha256"),
            }
            if output != expected_output:
                errors.append(f"PIQ v2 two-host rank {index} output identity drifted")
        pids[index] = rank.get("pid")
        hosts.append(rank.get("physical_host_sha256"))
        boots.append(rank.get("boot_id_sha256"))
    if len(set(hosts)) != 2 or len(set(boots)) != 2 or len(set(pids.values())) != 2:
        errors.append("PIQ v2 two-host evidence requires two distinct physical hosts, boots, and worker PIDs")

    partition = attempt.get("partition_rehash")
    partition_fields = {
        "input_records", "partition_records", "rehashed_records", "output_records",
        "artifact",
    }
    if _fields(partition, partition_fields, "PIQ v2 two-host partition/rehash", errors):
        counts = partition.get("partition_records")
        if (not _positive_int(partition.get("input_records"))
                or not isinstance(counts, list) or len(counts) != 2
                or not all(_positive_int(value) for value in counts)
                or sum(counts) != partition.get("input_records")
                or partition.get("rehashed_records") != partition.get("input_records")
                or partition.get("output_records") != partition.get("input_records")):
            errors.append("PIQ v2 two-host partition/rehash census is not exact")
        partition_doc = _load_ref_json(
            partition.get("artifact"), "PIQ v2 two-host partition/rehash artifact",
            artifacts, errors,
        )
        partition_doc_fields = {
            "schema", "study_id", "run_id", "attempt_id", "input_records",
            "partition_records", "rehashed_records", "output_records",
        }
        if _fields(partition_doc, partition_doc_fields,
                   "PIQ v2 two-host partition/rehash document", errors):
            expected_partition = {
                "schema": "piq.cpu-two-host-partition-rehash.v2",
                "study_id": envelope.get("study_id"), "run_id": run_id,
                "attempt_id": attempt_id,
                **{field: partition.get(field) for field in (
                    "input_records", "partition_records", "rehashed_records",
                    "output_records",
                )},
            }
            if partition_doc != expected_partition:
                errors.append("PIQ v2 two-host partition/rehash artifact identity drifted")

    transport = attempt.get("transport")
    transport_fields = {
        "family", "source_rank", "target_rank", "source_pid", "target_pid",
        "bytes_sent", "artifact",
    }
    if _fields(transport, transport_fields, "PIQ v2 two-host transport", errors):
        source_rank, target_rank = transport.get("source_rank"), transport.get("target_rank")
        if (transport.get("family") != "tcp" or {source_rank, target_rank} != {0, 1}
                or transport.get("source_pid") != pids.get(source_rank)
                or transport.get("target_pid") != pids.get(target_rank)
                or not _positive_int(transport.get("bytes_sent"))):
            errors.append("PIQ v2 two-host transport must bind real worker PID to worker PID")
        transport_doc = _load_ref_json(
            transport.get("artifact"), "PIQ v2 two-host transport artifact",
            artifacts, errors,
        )
        transport_doc_fields = {
            "schema", "study_id", "run_id", "attempt_id", "family",
            "source_rank", "target_rank", "source_pid", "target_pid",
            "source_host_sha256", "target_host_sha256", "bytes_sent",
        }
        if _fields(transport_doc, transport_doc_fields,
                   "PIQ v2 two-host transport document", errors):
            expected_transport = {
                "schema": "piq.cpu-two-host-worker-transport.v2",
                "study_id": envelope.get("study_id"), "run_id": run_id,
                "attempt_id": attempt_id,
                **{field: transport.get(field) for field in (
                    "family", "source_rank", "target_rank", "source_pid",
                    "target_pid", "bytes_sent",
                )},
                "source_host_sha256": hosts[source_rank]
                if source_rank in (0, 1) and len(hosts) == 2 else None,
                "target_host_sha256": hosts[target_rank]
                if target_rank in (0, 1) and len(hosts) == 2 else None,
            }
            if transport_doc != expected_transport:
                errors.append("PIQ v2 two-host transport artifact identity drifted")

    comparison = envelope.get("comparison")
    if not _fields(comparison, {"scientific_disposition", "reason"},
                   "PIQ v2 two-host comparison", errors):
        comparison = {}
    if (comparison.get("scientific_disposition") != "DIAGNOSTIC_ONLY"
            or not _text(comparison.get("reason"))):
        errors.append("PIQ v2 two-host comparison must remain diagnostic")
    correctness = envelope.get("correctness")
    if not _fields(correctness, {"accepted", "artifact"},
                   "PIQ v2 two-host correctness", errors):
        correctness = {}
    if correctness.get("accepted") is not True:
        errors.append("PIQ v2 two-host correctness must be accepted")
    correctness_doc = _load_ref_json(
        correctness.get("artifact"), "PIQ v2 two-host correctness artifact",
        artifacts, errors,
    )
    correctness_fields = {
        "schema", "study_id", "run_id", "attempt_id", "input_records",
        "output_records", "accepted",
    }
    if _fields(correctness_doc, correctness_fields, "PIQ v2 two-host correctness document", errors):
        if ((correctness_doc.get("schema"), correctness_doc.get("study_id"),
             correctness_doc.get("run_id"), correctness_doc.get("attempt_id"),
             correctness_doc.get("input_records"), correctness_doc.get("output_records"),
             correctness_doc.get("accepted"))
                != ("piq.cpu-two-host-correctness.v2", envelope.get("study_id"),
                    run_id, attempt_id, partition.get("input_records"),
                    partition.get("output_records"), True)):
            errors.append("PIQ v2 two-host correctness does not bind the partition census")

    evidence = envelope.get("workload_evidence")
    evidence_roles = {
        "execution", "correctness", "partition_rehash", "transport", "cleanup",
        "custody", "readback", "export", "owner_attestation", "current_admission",
    }
    if not isinstance(evidence, dict) or set(evidence) != evidence_roles:
        errors.append("PIQ v2 two-host workload evidence roles are incomplete")
        evidence = {}
    else:
        bindings = {
            "correctness": correctness.get("artifact"),
            "partition_rehash": partition.get("artifact"),
            "transport": transport.get("artifact"),
            "owner_attestation": execution.get("owner_attestation"),
            "current_admission": execution.get("current_admission"),
            "execution": attempt.get("raw_result"),
        }
        for role, ref in bindings.items():
            if evidence.get(role) != ref:
                errors.append(f"PIQ v2 two-host {role} role does not bind the exact evidence ref")
        for role in ("cleanup", "custody", "readback", "export"):
            _validate_retention_document(
                envelope, artifacts, role, evidence[role], run_id, [attempt_id],
                objective_end, errors,
                [pids[index] for index in (0, 1) if index in pids]
                if role == "cleanup" else None,
            )

    owner_ref = execution.get("owner_attestation")
    owner = _load_ref_json(owner_ref, "PIQ v2 two-host owner attestation", artifacts, errors)
    owner_fields = {
        "schema", "study_id", "run_id", "attempt_id", "owner", "authority",
        "attested_at", "hosts", "accepted",
    }
    if _fields(owner, owner_fields, "PIQ v2 two-host owner attestation", errors):
        expected_hosts = [
            {
                "rank": rank.get("rank"), "physical_host_sha256": rank.get("physical_host_sha256"),
                "boot_id_sha256": rank.get("boot_id_sha256"), "pid": rank.get("pid"),
            }
            for rank in ranks
        ]
        attested = _timestamp(owner.get("attested_at"))
        if ((owner.get("schema"), owner.get("study_id"), owner.get("run_id"),
             owner.get("attempt_id"), owner.get("hosts"), owner.get("accepted"))
                != ("piq.cpu-two-host-owner-attestation.v2", envelope.get("study_id"),
                    run_id, attempt_id, expected_hosts, True)
                or not _text(owner.get("owner")) or not _text(owner.get("authority"))
                or attested is None
                or (objective_start is not None and attested > objective_start)):
            errors.append("PIQ v2 two-host owner attestation is not exact or timely")

    admission = _load_ref_json(
        execution.get("current_admission"), "PIQ v2 two-host current admission",
        artifacts, errors,
    )
    admission_fields = {
        "schema", "study_id", "run_id", "attempt_id", "authority",
        "owner_attestation_sha256", "admitted_at", "expires_at", "status",
        "revoked",
    }
    if _fields(admission, admission_fields, "PIQ v2 two-host current admission", errors):
        admitted = _timestamp(admission.get("admitted_at"))
        expires = _timestamp(admission.get("expires_at"))
        owner_sha = owner_ref.get("sha256") if isinstance(owner_ref, dict) else None
        if ((admission.get("schema"), admission.get("study_id"), admission.get("run_id"),
             admission.get("attempt_id"), admission.get("owner_attestation_sha256"),
             admission.get("status"), admission.get("revoked"))
                != ("piq.cpu-two-host-current-admission.v2", envelope.get("study_id"),
                    run_id, attempt_id, owner_sha, "ADMITTED", False)
                or not _text(admission.get("authority"))
                or not isinstance(owner, dict)
                or admission.get("authority") != owner.get("authority")
                or admitted is None or expires is None
                or objective_start is None or objective_end is None
                or not admitted <= objective_start < objective_end <= expires):
            errors.append("PIQ v2 two-host admission is not current for the objective window")
    _require_fixed_two_host_admission(
        admissions, envelope, envelope_sha256, owner_ref,
        execution.get("current_admission"), owner, run_id, attempt_id,
        objective_end, errors,
    )
    if envelope.get("detached_objects") != []:
        errors.append("PIQ v2 two-host evidence does not admit detached objects")


def _validate_assignment(assignment: Any, execution: dict, attempts: list[dict],
                         envelope: dict, plan_hashes: dict[str, str],
                         errors: list[str]) -> None:
    fields = {
        "schema", "study_id", "group_id", "run_id", "scheduler_job_id",
        "owner", "node", "session_id", "cgroup", "device_uuids",
        "source_sha256s", "runtime_identifier", "executable_sha256",
        "plan_sha256", "window", "requested_resources", "observed_resources",
        "attempt_processes",
    }
    if not _fields(assignment, fields, "PIQ v2 assignment evidence", errors):
        return
    if (assignment.get("schema"), assignment.get("study_id"), assignment.get("group_id"),
            assignment.get("run_id")) != (
                "piq.gpu-training-assignment.v1", envelope.get("study_id"),
                execution.get("group_id"), execution.get("run_id")):
        errors.append("PIQ v2 assignment study/group/run identity mismatch")
    for field in ("scheduler_job_id", "owner", "node", "session_id", "cgroup",
                  "runtime_identifier"):
        if not _text(assignment.get(field)):
            errors.append(f"PIQ v2 assignment {field} must be nonempty")
    for field in ("executable_sha256", "plan_sha256"):
        value = assignment.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            errors.append(f"PIQ v2 assignment {field} must be a SHA-256 digest")
    if (assignment.get("requested_resources") != execution.get("requested_resources")
            or assignment.get("observed_resources") != execution.get("observed_resources")):
        errors.append("PIQ v2 assignment requested/observed resources drifted")
    resources = execution.get("observed_resources", {})
    if assignment.get("device_uuids") != resources.get("device_uuids"):
        errors.append("PIQ v2 assignment device UUIDs differ from observed resources")
    if assignment.get("node") != resources.get("host"):
        errors.append("PIQ v2 assignment node differs from the observed host")
    source_hashes = [item.get("sha256") for item in envelope.get("experiment", {}).get("source_artifacts", [])]
    if assignment.get("source_sha256s") != source_hashes:
        errors.append("PIQ v2 assignment source identity drifted")
    if (assignment.get("runtime_identifier")
            != envelope.get("stack", {}).get("runtime_identifier")):
        errors.append("PIQ v2 assignment runtime identity drifted")
    if assignment.get("plan_sha256") != plan_hashes.get(execution.get("group_id")):
        errors.append("PIQ v2 assignment plan identity drifted")
    window = assignment.get("window")
    objective = envelope.get("objective", {})
    if not isinstance(window, dict) or set(window) != {"started_at", "ended_at"}:
        errors.append("PIQ v2 assignment window fields are invalid")
    else:
        started, ended = _timestamp(window.get("started_at")), _timestamp(window.get("ended_at"))
        objective_start = _timestamp(objective.get("started_at"))
        objective_end = _timestamp(objective.get("ended_at"))
        if (started is None or ended is None or objective_start is None
                or objective_end is None or not objective_start <= started < ended <= objective_end):
            errors.append("PIQ v2 assignment window must be ordered inside the objective window")
        else:
            for attempt in attempts:
                raw = attempt.get("_raw", {})
                attempt_started = _timestamp(raw.get("started_at"))
                attempt_ended = _timestamp(raw.get("ended_at"))
                if (attempt_started is None or attempt_ended is None
                        or not started <= attempt_started < attempt_ended <= ended):
                    errors.append(
                        "PIQ v2 raw attempt interval must be contained by its assignment window"
                    )
    expected_processes = [
        {"attempt_id": attempt.get("id"), "pids": [rank.get("pid") for rank in attempt.get("_raw", {}).get("ranks", [])]}
        for attempt in attempts
    ]
    if assignment.get("attempt_processes") != expected_processes:
        errors.append("PIQ v2 assignment process census differs from raw ranks")


def _validate_raw_attempt(raw: Any, attempt: dict, execution: dict,
                          envelope: dict, artifacts: dict[str, dict],
                          arm_hashes: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(raw, dict):
        errors.append("PIQ v2 raw attempt must be a JSON object")
        return
    bindings = {
        "study_id": envelope.get("study_id"), "group_id": attempt.get("group_id"),
        "run_id": attempt.get("run_id"), "attempt_id": attempt.get("id"),
        "role": attempt.get("role"), "status": attempt.get("status"),
        "accepted": attempt.get("accepted"), "failure_reason": attempt.get("failure_reason"),
        "design_sha256": envelope.get("experiment", {}).get("design", {}).get("sha256"),
    }
    if any(raw.get(key) != value for key, value in bindings.items()):
        errors.append("PIQ v2 raw attempt identity/disposition mismatch")
    started = _timestamp(raw.get("started_at"))
    ended = _timestamp(raw.get("ended_at"))
    if started is None or ended is None or started >= ended:
        errors.append("PIQ v2 raw attempt requires an ordered UTC execution interval")
    if (raw.get("arm_configuration_sha256") != arm_hashes.get(attempt.get("role"))
            or not isinstance(raw.get("arm_configuration_sha256"), str)
            or SHA256_RE.fullmatch(raw["arm_configuration_sha256"]) is None):
        errors.append("PIQ v2 raw attempt arm identity differs from the immutable design")
    ranks = raw.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        errors.append("PIQ v2 attempt requires two complete rank rows")
        return
    expected_uuids = execution.get("observed_resources", {}).get("device_uuids")
    expected_host = execution.get("observed_resources", {}).get("host")
    output_paths: list[Any] = []
    for index, rank in enumerate(ranks):
        fields = {
            "rank", "local_rank", "pid", "host", "device_type", "device_uuid",
            "output_ref", "arm_configuration_sha256",
        }
        if not isinstance(rank, dict) or set(rank) != fields:
            errors.append("PIQ v2 rank row fields are invalid")
            continue
        if (rank.get("rank"), rank.get("local_rank"), rank.get("device_type"),
                rank.get("device_uuid")) != (index, index, "cuda", expected_uuids[index]):
            errors.append("PIQ v2 rank/local-rank/CUDA device mapping mismatch")
        if not _positive_int(rank.get("pid")) or rank.get("host") != expected_host:
            errors.append("PIQ v2 rank row requires the observed host and a positive PID")
        _ref(rank.get("output_ref"), "PIQ v2 rank output", artifacts, errors)
        output_paths.append(rank.get("output_ref", {}).get("path")
                            if isinstance(rank.get("output_ref"), dict) else None)
        if rank.get("arm_configuration_sha256") != raw.get("arm_configuration_sha256"):
            errors.append("PIQ v2 rank arm configuration hash mismatch")
    if ([rank.get("rank") for rank in ranks] != [0, 1]
            or [rank.get("local_rank") for rank in ranks] != [0, 1]
            or len({rank.get("pid") for rank in ranks}) != 2
            or len({rank.get("host") for rank in ranks}) != 1
            or len({rank.get("device_uuid") for rank in ranks}) != 2
            or len(set(output_paths)) != 2):
        errors.append("PIQ v2 attempt must have distinct ranks/PIDs/devices on one host")
    attempt["_raw"] = raw


def _validate_detached(envelope: dict, artifacts: dict[str, dict], errors: list[str]) -> None:
    values = envelope.get("detached_objects")
    if not isinstance(values, list) or not values:
        errors.append("PIQ v2 requires detached checkpoint/telemetry custody records")
        return
    ids: set[str] = set()
    for index, item in enumerate(values):
        label = f"PIQ v2 detached_objects[{index}]"
        fields = {
            "id", "role", "sha256", "size_bytes", "media_type", "custody_record",
            "local_readback", "piq_readback", "export_readback", "availability",
        }
        if not _fields(item, fields, label, errors):
            continue
        object_id = item.get("id")
        if not _text(object_id) or object_id in ids:
            errors.append(f"{label}.id must be nonempty and distinct")
        else:
            ids.add(object_id)
        if not _text(item.get("role")) or not _text(item.get("media_type")) or item.get("availability") != "retained" or not _positive_int(item.get("size_bytes")) or not isinstance(item.get("sha256"), str) or SHA256_RE.fullmatch(item["sha256"]) is None:
            errors.append(f"{label} identity/availability/size/hash is invalid")
        receipt_contracts = {
            "custody_record": ("piq.gpu-training-custody-record.v1", "custody"),
            "local_readback": ("piq.gpu-training-readback.v1", "local_durable"),
            "piq_readback": ("piq.gpu-training-readback.v1", "piq"),
            "export_readback": ("piq.gpu-training-readback.v1", "export"),
        }
        for field in ("custody_record", "local_readback", "piq_readback", "export_readback"):
            document = _load_ref_json(item.get(field), f"{label}.{field}", artifacts, errors)
            schema, scope = receipt_contracts[field]
            document_fields = {
                "schema", "scope", "verified_at", "immutable", "objects",
                "synthetic_fixture",
            }
            if (not isinstance(document, dict) or set(document) != document_fields
                    or document.get("schema") != schema
                    or document.get("scope") != scope
                    or _timestamp(document.get("verified_at")) is None
                    or document.get("immutable") is not True
                    or type(document.get("synthetic_fixture")) is not bool):
                errors.append(f"{label}.{field} is not an immutable scoped readback receipt")
                continue
            objective_end = _timestamp(envelope.get("objective", {}).get("ended_at"))
            if (objective_end is None
                    or _timestamp(document["verified_at"]) < objective_end):
                errors.append(f"{label}.{field} predates the complete study window")
            entries = document.get("objects") if isinstance(document, dict) else None
            if (not isinstance(entries, list) or not entries
                    or any(not isinstance(entry, dict)
                           or set(entry) != {"id", "sha256", "size_bytes", "available"}
                           for entry in entries)
                    or len({entry.get("id") for entry in entries}) != len(entries)):
                errors.append(f"{label}.{field} object census is malformed")
                continue
            match = next((entry for entry in entries or [] if isinstance(entry, dict) and entry.get("id") == object_id), None)
            if not isinstance(match, dict) or match.get("sha256") != item.get("sha256") or match.get("size_bytes") != item.get("size_bytes") or match.get("available") is not True:
                errors.append(f"{label}.{field} lacks exact durable object readback")


def _validate_bundle(bundle: Path) -> list[str]:
    errors: list[str] = []
    try:
        bundle = bundle.resolve(strict=True)
    except OSError as exc:
        return [f"PIQ v2 bundle is missing: {exc}"]
    if not bundle.is_dir() or bundle.is_symlink():
        return ["PIQ v2 bundle must be a real directory"]
    envelope_path = bundle / ENVELOPE
    if envelope_path.is_symlink():
        return ["PIQ v2 envelope must not be a symlink"]
    try:
        envelope = strict_load(envelope_path, "PIQ v2 envelope")
    except BundleError as exc:
        return [str(exc)]
    if not _fields(envelope, ENVELOPE_FIELDS, "PIQ v2 envelope", errors):
        return errors
    version = envelope.get("schema_version")
    if type(version) is not int or version != 2:
        errors.append(
            f"PIQ envelope schema_version {version!r} is unsupported. Only version 2 is active"
        )
    if (envelope.get("kind"), envelope.get("scope")) != (
            "piq_experiment_evidence", "diagnostic"):
        errors.append("PIQ v2 must be a diagnostic experiment envelope")
    experiment_kind = envelope.get("experiment_kind")
    expected_family = EXPERIMENT_KINDS.get(experiment_kind)
    if expected_family is None:
        errors.append(f"PIQ v2 experiment_kind {experiment_kind!r} is unsupported")
    elif envelope.get("workload_family") != expected_family:
        errors.append("PIQ v2 experiment_kind/workload_family binding is invalid")
    if envelope.get("scientific_disposition") not in {"DIAGNOSTIC_ONLY", "INCONCLUSIVE", "REJECTED"}:
        errors.append("PIQ v2 refuses qualified scientific dispositions")
    if not _text(envelope.get("study_id")):
        errors.append("PIQ v2 study_id must be nonempty")
    artifacts = _artifact_index(bundle, envelope, errors)
    # Every JSON artifact is parsed strictly even when its semantics are opaque here.
    for relative, item in artifacts.items():
        try:
            if relative.endswith(".json"):
                strict_load(item["_path"], f"PIQ v2 artifact {relative}")
            elif relative.endswith(".jsonl"):
                strict_load_jsonl(item["_path"], f"PIQ v2 artifact {relative}")
        except BundleError as exc:
            errors.append(str(exc))

    if experiment_kind == "checkpoint_write_batching_stage0":
        _validate_stage0(envelope, artifacts, errors)
        return list(dict.fromkeys(errors))
    if experiment_kind == "cpu_two_host_functional":
        admissions = _load_fixed_two_host_admissions(errors)
        _validate_two_host(
            envelope, artifacts, admissions, _digest(envelope_path), errors,
        )
        return list(dict.fromkeys(errors))
    if experiment_kind != "te_mxfp8_gpu_training":
        return list(dict.fromkeys(errors))

    arm_hashes: dict[str, Any] = {}
    plan_hashes: dict[str, str] = {}
    design: Any = None
    experiment = envelope.get("experiment")
    exp_fields = {
        "id", "question", "hypothesis", "workload_scope", "design",
        "source_artifacts", "data", "initial_checkpoint",
    }
    if _fields(experiment, exp_fields, "PIQ v2 experiment", errors):
        if experiment.get("id") != envelope.get("study_id") or any(not _text(experiment.get(field)) for field in ("question", "hypothesis", "workload_scope")):
            errors.append("PIQ v2 experiment identity/question/hypothesis/scope is invalid")
        design = _load_ref_json(experiment.get("design"), "PIQ v2 design", artifacts, errors)
        if not isinstance(design, dict) or design.get("study_id") != envelope.get("study_id") or design.get("contract_id") != "te_mxfp8_bf16_two_b200_v1":
            errors.append("PIQ v2 design does not bind the fixed GPU training study")
        else:
            arms = design.get("arms")
            if not isinstance(arms, dict) or not all(
                    isinstance(arms.get(role), dict)
                    and isinstance(arms[role].get("configuration_sha256"), str)
                    and SHA256_RE.fullmatch(arms[role]["configuration_sha256"]) is not None
                    for role in ("baseline", "candidate")):
                errors.append("PIQ v2 design must bind both immutable arm configurations")
            else:
                arm_hashes = {
                    role: arms[role]["configuration_sha256"]
                    for role in ("baseline", "candidate")
                }
            if design.get("schedule") != list(DESIGN_SCHEDULE):
                errors.append("PIQ v2 design schedule differs from the fixed five-group plan")
            else:
                plan_hashes = {
                    group["id"]: _hash_value(group) for group in DESIGN_SCHEDULE
                }
        sources = experiment.get("source_artifacts")
        if (not isinstance(sources, list) or len(sources) != len(SOURCE_ROLES)
                or len({ref.get("path") for ref in sources if isinstance(ref, dict)})
                != len(SOURCE_ROLES)):
            errors.append("PIQ v2 experiment must bind the complete first-party source closure")
        else:
            for index, ref in enumerate(sources):
                _ref(ref, f"PIQ v2 source_artifacts[{index}]", artifacts, errors)
            imports = design.get("source_imports") if isinstance(design, dict) else None
            if (not isinstance(imports, list)
                    or [item.get("role") for item in imports if isinstance(item, dict)]
                    != list(SOURCE_ROLES)
                    or [item.get("resolved_sha256") for item in imports]
                    != [ref.get("sha256") for ref in sources]):
                errors.append("PIQ v2 design/source import closure identity drifted")
        data_identity = _load_ref_json(
            experiment.get("data"), "PIQ v2 data identity", artifacts, errors,
        )
        design_data = design.get("data", {}) if isinstance(design, dict) else {}
        if (not isinstance(data_identity, dict) or any(
                data_identity.get(field) != design_data.get(field)
                for field in ("dataset_revision", "train_sha256",
                              "validation_sha256", "sampler_sha256")
        )):
            errors.append("PIQ v2 design/data identity drifted")
        initial = experiment.get("initial_checkpoint")
        if not isinstance(initial, dict) or set(initial) != {"object_id", "sha256", "size_bytes"}:
            errors.append("PIQ v2 initial checkpoint must be a detached object identity")
        elif initial.get("sha256") != (
                design.get("retention", {}).get("initial_checkpoint_sha256")
                if isinstance(design, dict) else None):
            errors.append("PIQ v2 design/initial-checkpoint identity drifted")

    stack = envelope.get("stack")
    if not isinstance(stack, dict) or set(stack) != {
        "piq_revision", "runtime_identifier", "hardware_identifier", "layers",
    }:
        errors.append("PIQ v2 stack fields are invalid")
    elif (not isinstance(stack.get("piq_revision"), str) or REVISION_RE.fullmatch(stack["piq_revision"]) is None
          or not _text(stack.get("runtime_identifier")) or not _text(stack.get("hardware_identifier"))
          or stack.get("layers") != [{"name": name, "state": "observed"} for name in REQUIRED_LAYERS]):
        errors.append("PIQ v2 stack must bind PIQ/runtime/hardware and every required layer")

    objective = envelope.get("objective")
    if not isinstance(objective, dict) or set(objective) != {
        "metric", "unit", "direction", "started_at", "ended_at",
    } or (objective.get("metric"), objective.get("unit"), objective.get("direction")) != (
        "complete_job_elapsed", "nanoseconds", "minimize") or not _text(objective.get("started_at")) or not _text(objective.get("ended_at")):
        errors.append("PIQ v2 objective must use the complete-job nanosecond boundary")

    executions = envelope.get("executions")
    if not isinstance(executions, list) or len(executions) != 5:
        errors.append("PIQ v2 executions must contain the five fixed groups")
        executions = []
    execution_map: dict[str, dict] = {}
    run_ids: set[str] = set()
    for index, execution in enumerate(executions):
        fields = {
            "group_id", "run_id", "order", "attempt_ids", "requested_resources",
            "observed_resources", "assignment_evidence",
        }
        if not _fields(execution, fields, f"PIQ v2 executions[{index}]", errors):
            continue
        if index < len(SCHEDULE) and (execution.get("group_id"), execution.get("order")) != SCHEDULE[index]:
            errors.append("PIQ v2 execution groups/orders differ from the fixed schedule")
        if not _text(execution.get("run_id")) or execution.get("run_id") in run_ids:
            errors.append("PIQ v2 execution run IDs must be nonempty and distinct")
        else:
            run_ids.add(execution["run_id"])
        resources = execution.get("requested_resources")
        observed = execution.get("observed_resources")
        expected_common = {"node_count": 1, "accelerator_count": 2, "process_count": 2}
        if not isinstance(resources, dict) or any(resources.get(k) != v for k, v in expected_common.items()) or resources.get("executor") != "assigned-gpu-host" or not _positive_int(resources.get("timeout_seconds")):
            errors.append("PIQ v2 requested resources must be one assigned host/two GPUs/two processes")
        if not isinstance(observed, dict) or any(observed.get(k) != v for k, v in expected_common.items()) or not _text(observed.get("host")) or not isinstance(observed.get("device_uuids"), list) or len(observed["device_uuids"]) != 2 or len(set(observed["device_uuids"])) != 2:
            errors.append("PIQ v2 observed resources must bind one host and two distinct devices")
        _ref(execution.get("assignment_evidence"), "PIQ v2 assignment evidence", artifacts, errors)
        execution_map[execution.get("group_id")] = execution

    attempts = envelope.get("attempts")
    expected_order = [(group, role) for group, order in SCHEDULE for role in order]
    if not isinstance(attempts, list) or len(attempts) != 10:
        errors.append("PIQ v2 attempts must retain the complete ten-attempt census")
        attempts = []
    actual_order: list[tuple[Any, Any]] = []
    attempt_ids: set[str] = set()
    for index, attempt in enumerate(attempts):
        fields = {
            "id", "run_id", "group_id", "role", "status", "accepted",
            "failure_reason", "raw_result", "assignment_evidence",
        }
        if not _fields(attempt, fields, f"PIQ v2 attempts[{index}]", errors):
            continue
        actual_order.append((attempt.get("group_id"), attempt.get("role")))
        if not _text(attempt.get("id")) or attempt.get("id") in attempt_ids:
            errors.append("PIQ v2 attempt IDs must be nonempty and distinct")
        else:
            attempt_ids.add(attempt["id"])
        execution = execution_map.get(attempt.get("group_id"), {})
        if attempt.get("run_id") != execution.get("run_id") or attempt.get("assignment_evidence") != execution.get("assignment_evidence"):
            errors.append("PIQ v2 attempt run/assignment differs from its execution group")
        if attempt.get("status") not in {"succeeded", "failed"} or type(attempt.get("accepted")) is not bool:
            errors.append("PIQ v2 attempt disposition is invalid")
        if (attempt.get("status") == "succeeded") != (attempt.get("failure_reason") is None):
            errors.append("PIQ v2 attempt failure reason/status mismatch")
        if attempt.get("status") == "failed" and attempt.get("accepted") is not False:
            errors.append("PIQ v2 failed attempt cannot be accepted")
        raw = _load_ref_json(attempt.get("raw_result"), f"PIQ v2 raw attempt[{index}]", artifacts, errors)
        _validate_raw_attempt(raw, attempt, execution, envelope, artifacts,
                              arm_hashes, errors)
    if actual_order != expected_order:
        errors.append("PIQ v2 attempt census is missing, duplicated, retried, or reordered")
    for execution in executions:
        group_attempts = [attempt for attempt in attempts if attempt.get("group_id") == execution.get("group_id")]
        if execution.get("attempt_ids") != [attempt.get("id") for attempt in group_attempts]:
            errors.append("PIQ v2 execution attempt_ids differ from the complete group order")
        assignment = _load_ref_json(execution.get("assignment_evidence"), "PIQ v2 assignment evidence", artifacts, errors)
        _validate_assignment(assignment, execution, group_attempts, envelope,
                             plan_hashes, errors)

    comparison = envelope.get("comparison")
    if not isinstance(comparison, dict) or set(comparison) != {
        "baseline_attempt_ids", "candidate_attempt_ids", "scientific_disposition", "reason",
    }:
        errors.append("PIQ v2 comparison fields are invalid")
    else:
        baseline_ids = [a.get("id") for a in attempts if a.get("role") == "baseline"]
        candidate_ids = [a.get("id") for a in attempts if a.get("role") == "candidate"]
        if comparison.get("baseline_attempt_ids") != baseline_ids or comparison.get("candidate_attempt_ids") != candidate_ids or comparison.get("scientific_disposition") != envelope.get("scientific_disposition") or not _text(comparison.get("reason")):
            errors.append("PIQ v2 comparison must bind every arm attempt and remain diagnostic")

    correctness = envelope.get("correctness")
    if not isinstance(correctness, dict) or set(correctness) != {"accepted", "artifact"} or type(correctness.get("accepted")) is not bool:
        errors.append("PIQ v2 correctness fields are invalid")
    else:
        _ref(correctness.get("artifact"), "PIQ v2 correctness artifact", artifacts, errors)
        if any(attempt.get("status") != "succeeded" or attempt.get("accepted") is not True
               for attempt in attempts) and correctness.get("accepted") is not False:
            errors.append("PIQ v2 failed/unaccepted attempts require an explicit false correctness disposition")

    roles = envelope.get("workload_evidence")
    if not isinstance(roles, dict) or set(roles) != set(REQUIRED_ROLES):
        errors.append("PIQ v2 workload evidence must bind every required role")
    else:
        for role in REQUIRED_ROLES:
            _ref(roles[role], f"PIQ v2 workload role {role}", artifacts, errors)
    _validate_detached(envelope, artifacts, errors)

    initial = envelope.get("experiment", {}).get("initial_checkpoint") if isinstance(envelope.get("experiment"), dict) else None
    if isinstance(initial, dict):
        detached = {item.get("id"): item for item in envelope.get("detached_objects", []) if isinstance(item, dict)}
        match = detached.get(initial.get("object_id"))
        if not isinstance(match, dict) or match.get("sha256") != initial.get("sha256") or match.get("size_bytes") != initial.get("size_bytes"):
            errors.append("PIQ v2 initial checkpoint lacks exact detached-object custody")
    return list(dict.fromkeys(errors))


def validate_bundle(bundle: Path) -> list[str]:
    """Public fail-closed bundle API."""
    try:
        return _validate_bundle(bundle)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return [f"malformed current PIQ v2 bundle ({type(exc).__name__}): {exc}"]


def import_bundle(source: Path, destination: Path) -> Path:
    """Atomically copy only a valid diagnostic bundle. Never create a result."""
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise BundleError("destination already exists")
    errors = validate_bundle(source)
    if errors:
        raise BundleError("source bundle is invalid: " + ". ".join(errors))
    destination_parent = destination.parent.resolve(strict=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.piq-v2-", dir=destination_parent))
    staged_bundle = stage / destination.name
    try:
        staged_bundle.mkdir()
        envelope = strict_load(source / ENVELOPE, "PIQ v2 envelope")
        shutil.copyfile(source / ENVELOPE, staged_bundle / ENVELOPE)
        for artifact in envelope["artifacts"]:
            relative = PurePosixPath(artifact["path"])
            target = staged_bundle.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.joinpath(*relative.parts), target)
        errors = validate_bundle(staged_bundle)
        if errors:
            raise BundleError("copied bundle failed revalidation: " + ". ".join(errors))
        envelope_hash = _digest(staged_bundle / ENVELOPE)
        receipt = {
            "schema_version": 1,
            "kind": "piq_experiment_diagnostic_import",
            "experiment_kind": envelope["experiment_kind"],
            "envelope_sha256": envelope_hash,
            "scope": "diagnostic",
            "qualification": "not_established",
            "publishable": False,
        }
        (staged_bundle / IMPORT_RECEIPT).write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise BundleError("destination appeared during import")
        staged_bundle.rename(destination)
        stage.rmdir()
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("bundle", type=Path)
    ingest = commands.add_parser("import")
    ingest.add_argument("bundle", type=Path)
    ingest.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            errors = validate_bundle(args.bundle)
            if errors:
                raise BundleError("\n".join(errors))
        else:
            import_bundle(args.bundle, args.destination)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: retained current PIQ v2 diagnostic evidence. Qualification and publication remain unestablished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
