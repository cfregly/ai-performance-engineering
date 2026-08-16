"""Semantic validation for optimization queue evidence artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ArtifactValidation:
    path: str
    valid: bool
    sha256: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "sha256": self.sha256,
            "errors": list(self.errors),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_timestamp(text: str) -> bool:
    try:
        datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return False
    return True


def _resolve_reference(root: Path, reference: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(reference, dict):
        errors.append(f"{label} must be an object with path and sha256")
        return None
    raw_path = reference.get("path")
    expected_hash = str(reference.get("sha256") or "").lower()
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{label}.path is missing")
        return None
    if not SHA256_PATTERN.fullmatch(expected_hash):
        errors.append(f"{label}.sha256 is invalid")
        return None
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        errors.append(f"{label}.path escapes the job directory")
        return None
    if not path.is_file():
        errors.append(f"{label}.path is missing")
        return None
    if not hmac.compare_digest(_sha256(path), expected_hash):
        errors.append(f"{label}.sha256 does not match the file")
        return None
    return path


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0


def _validate_benchmark_result(path: Path, label: str, errors: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(f"{label} is not valid JSON")
        return
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object")
        return
    if not _valid_timestamp(str(payload.get("timestamp") or "")):
        errors.append(f"{label}.timestamp is missing or invalid")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        errors.append(f"{label}.results must contain exactly one target")
        return
    result = results[0]
    if not isinstance(result, dict) or result.get("status") != "completed":
        errors.append(f"{label} target status is not completed")
        return
    benchmarks = result.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 1:
        errors.append(f"{label} must contain exactly one benchmark")
        return
    benchmark = benchmarks[0]
    if not isinstance(benchmark, dict):
        errors.append(f"{label} benchmark must be an object")
        return
    accepted_statuses = {"succeeded", "failed_regression", "failed_no_speedup"}
    if benchmark.get("status") not in accepted_statuses:
        errors.append(f"{label} benchmark status is not valid comparison evidence")
    if not isinstance(benchmark.get("example"), str) or not benchmark.get("example"):
        errors.append(f"{label} benchmark example is missing")
    if not _positive_number(benchmark.get("baseline_time_ms")):
        errors.append(f"{label} baseline_time_ms is not positive and finite")
    optimizations = benchmark.get("optimizations")
    if not isinstance(optimizations, list) or not optimizations:
        errors.append(f"{label} optimizations must be a nonempty array")
        return
    for index, optimization in enumerate(optimizations):
        opt_label = f"{label}.optimizations[{index}]"
        if not isinstance(optimization, dict):
            errors.append(f"{opt_label} must be an object")
            continue
        if not isinstance(optimization.get("file"), str) or not optimization.get("file"):
            errors.append(f"{opt_label}.file is missing")
        if optimization.get("status") != "succeeded":
            errors.append(f"{opt_label}.status is not succeeded")
        if not _positive_number(optimization.get("time_ms")):
            errors.append(f"{opt_label}.time_ms is not positive and finite")
        for verification_name in ("input_verification", "verification"):
            verification = optimization.get(verification_name)
            if not isinstance(verification, dict) or verification.get("passed") is not True:
                errors.append(f"{opt_label}.{verification_name}.passed is not true")
    manifests = result.get("manifests")
    if not isinstance(manifests, list) or len(manifests) < 2:
        errors.append(f"{label} must contain baseline and optimized run manifests")
    else:
        variants: set[str] = set()
        for index, entry in enumerate(manifests):
            manifest_label = f"{label}.manifests[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{manifest_label} must be an object")
                continue
            variants.add(str(entry.get("variant") or ""))
            manifest = entry.get("manifest")
            if not isinstance(manifest, dict) or not manifest.get("schemaVersion"):
                errors.append(f"{manifest_label}.manifest schema is missing")
                continue
            git = manifest.get("git")
            config = manifest.get("config")
            if not isinstance(git, dict) or not git.get("commit") or git.get("dirty") is not False:
                errors.append(f"{manifest_label}.manifest Git identity is not clean")
            if not isinstance(config, dict) or config.get("validity_profile") != "strict":
                errors.append(f"{manifest_label}.manifest did not use strict validity")
            if manifest.get("collection_warnings"):
                errors.append(f"{manifest_label}.manifest contains collection warnings")
            if manifest.get("runtime_capability_limitations"):
                errors.append(f"{manifest_label}.manifest contains runtime limitations")
        if not {"baseline", "optimized"}.issubset(variants):
            errors.append(f"{label} run manifests are missing baseline or optimized variants")


def _validate_repeat_manifest(payload: dict[str, Any], root: Path, errors: list[str]) -> None:
    if payload.get("schema_version") != "aisp.queue-repeat-manifest/v1":
        errors.append("repeat manifest schema_version is invalid")
    declared = payload.get("declared_repeat_count")
    executed = payload.get("executed_repeat_count")
    if not isinstance(declared, int) or declared <= 0:
        errors.append("declared_repeat_count must be positive")
    if executed != declared:
        errors.append("executed_repeat_count does not match the declared repeat count")
    policy = payload.get("comparison_policy")
    if policy not in {"single_command", "paired_interleaved"}:
        errors.append("comparison_policy is invalid")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != declared:
        errors.append("runs must contain one entry per declared repeat")
        return
    observed_orders: list[tuple[str, ...]] = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            errors.append(f"runs[{index - 1}] must be an object")
            continue
        if run.get("sequence") != index:
            errors.append("repeat sequences must be contiguous and start at 1")
        order = run.get("order")
        expected_roles = ["job"]
        if policy == "paired_interleaved":
            if order not in (["control", "candidate"], ["candidate", "control"]):
                errors.append(f"runs[{index - 1}].order is invalid")
                continue
            expected_roles = ["control", "candidate"]
            observed_orders.append(tuple(str(role) for role in order))
        role_entries = run.get("roles")
        if not isinstance(role_entries, dict):
            errors.append(f"runs[{index - 1}].roles must be an object")
            continue
        for role in expected_roles:
            role_entry = role_entries.get(role)
            if not isinstance(role_entry, dict):
                errors.append(f"runs[{index - 1}].roles.{role} is missing")
                continue
            if role_entry.get("exit_code") != 0:
                errors.append(f"runs[{index - 1}].roles.{role} did not exit successfully")
            result_path = _resolve_reference(
                root,
                role_entry.get("result"),
                f"runs[{index - 1}].roles.{role}.result",
                errors,
            )
            if result_path is not None:
                _validate_benchmark_result(
                    result_path,
                    f"runs[{index - 1}].roles.{role}.result",
                    errors,
                )
    if policy == "paired_interleaved" and declared > 1:
        order_counts = {order: observed_orders.count(order) for order in set(observed_orders)}
        if len(order_counts) != 2 or max(order_counts.values()) - min(order_counts.values()) > 1:
            errors.append("paired repeats do not alternate control-first and candidate-first order")


def _validate_hashed_artifact_manifest(
    payload: dict[str, Any], root: Path, errors: list[str]
) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifact manifest must contain a nonempty artifacts array")
        return
    for index, artifact in enumerate(artifacts):
        _resolve_reference(root, artifact, f"artifacts[{index}]", errors)


def _validate_json_payload(path: Path, payload: Any, root: Path, errors: list[str]) -> None:
    if not isinstance(payload, dict) or not payload:
        errors.append("JSON artifact must be a nonempty object")
        return
    name = path.name
    if name == "job.json":
        if not isinstance(payload.get("id"), str) or not payload.get("id"):
            errors.append("job.json id is missing")
        if not isinstance(payload.get("stage"), str) or not payload.get("stage"):
            errors.append("job.json stage is missing")
    elif name == "artifact_contract.json":
        if not payload.get("stage") and not payload.get("stage_contracts"):
            errors.append("artifact contract has no stage declaration")
        required_files = payload.get("required_files")
        if required_files is not None and not isinstance(required_files, list):
            errors.append("artifact contract required_files must be an array")
    elif name == "repeat_run_manifest.json":
        _validate_repeat_manifest(payload, root, errors)
    elif name in {
        "artifact_contract_manifest.json",
        "profile_artifacts.json",
    }:
        _validate_hashed_artifact_manifest(payload, root, errors)
    elif name == "profiler_preflight_manifest.json":
        if payload.get("schema_version") != "aisp.profiler-preflight/v1":
            errors.append("profiler preflight schema_version is invalid")
        checks = payload.get("checks")
        if not isinstance(checks, list) or not checks:
            errors.append("profiler preflight checks must be a nonempty array")
        else:
            for index, check in enumerate(checks):
                if not isinstance(check, dict) or not check.get("check"):
                    errors.append(f"profiler preflight checks[{index}] is invalid")
                    continue
                if check.get("required") is True and check.get("status") != "passed":
                    errors.append(f"required profiler preflight checks[{index}] did not pass")
    elif name == "output_verification_record.json":
        if payload.get("passed") is not True:
            errors.append("output verification did not pass")
        source_result = payload.get("source_result")
        _resolve_reference(root, source_result, "source_result", errors)
        source_hash = (
            str(source_result.get("sha256") or "").lower()
            if isinstance(source_result, dict)
            else ""
        )
        declared_hash = str(payload.get("source_result_sha256") or "").lower()
        if not SHA256_PATTERN.fullmatch(declared_hash):
            errors.append("source_result_sha256 is invalid")
        elif source_hash and not hmac.compare_digest(source_hash, declared_hash):
            errors.append("source_result_sha256 contradicts source_result.sha256")
    elif name == "control_candidate_comparison.json":
        if payload.get("comparison_policy") != "paired_interleaved":
            errors.append("comparison policy is not paired_interleaved")
        if payload.get("correctness_passed") is not True:
            errors.append("comparison correctness did not pass")
        repeat_hash = str(payload.get("repeat_manifest_sha256") or "").lower()
        if not SHA256_PATTERN.fullmatch(repeat_hash):
            errors.append("comparison is not bound to a repeat manifest hash")
        else:
            repeat_path = root / "repeat_run_manifest.json"
            if not repeat_path.is_file():
                errors.append("comparison repeat manifest is missing")
            elif not hmac.compare_digest(_sha256(repeat_path), repeat_hash):
                errors.append("comparison repeat manifest hash does not match")
    elif name == "claim_decision.json":
        if payload.get("claim_allowed") is not True:
            errors.append("claim decision does not allow promotion")
        if not payload.get("reviewer"):
            errors.append("claim decision reviewer is missing")


def validate_evidence_artifact(path: Path, root: Path) -> ArtifactValidation:
    """Validate one required artifact by content and return its digest."""

    path = Path(path)
    root = Path(root).resolve()
    relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    errors: list[str] = []
    if not path.is_file():
        return ArtifactValidation(path=relative, valid=False, errors=["file is missing"])
    digest = _sha256(path)
    if path.name in {"DONE", "APPROVED", "MANUAL_REVIEW_REQUIRED"}:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            errors.append("marker is empty")
        elif path.name != "APPROVED" and not _valid_timestamp(text):
            errors.append("marker does not contain an ISO 8601 timestamp")
    elif path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("JSON could not be parsed")
        else:
            _validate_json_payload(path, payload, root, errors)
    elif path.name not in {"stdout.log", "stderr.log"} and path.stat().st_size == 0:
        errors.append("artifact is empty")
    return ArtifactValidation(
        path=relative,
        valid=not errors,
        sha256=digest,
        errors=errors,
    )
