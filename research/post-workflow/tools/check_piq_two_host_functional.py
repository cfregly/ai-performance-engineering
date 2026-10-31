#!/usr/bin/env python3
"""Validate an optional, supplementary PIQ two-host functional receipt.

The receipt closes a narrow artifact-gate failure: declaring two nodes is not
evidence that one accepted attempt observed two hosts. It remains diagnostic,
producer-asserted evidence and never satisfies scientific qualification.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any

if __package__:
    from .piq_experiment_evidence import validate_bundle
else:
    from piq_experiment_evidence import validate_bundle


RECEIPT_PATH = "evidence/two-host-functional-receipt.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
READINESS_BLOCK = (
    "PIQ two-host retained producer evidence cannot promote package readiness. "
    "Admitted owner-attested physical-host, transport, and cleanup evidence is still required"
)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _same_json(left: Any, right: Any) -> bool:
    try:
        options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def _canonical_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _safe_relative(value: Any) -> bool:
    if not _nonempty(value) or "\\" in value:
        return False
    path = Path(value)
    raw_parts = value.split("/")
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in raw_parts)


def _load_json(path: Path, label: str, errors: list[str]) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = child
        return value

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{label}: cannot read JSON: {exc}")
        return None


def _validated_ref(
    package: Path, value: Any, label: str, errors: list[str]
) -> tuple[Path, dict[str, str]] | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        errors.append(f"{label} must contain exactly path and sha256")
        return None
    relative = value.get("path")
    digest = value.get("sha256")
    if not _safe_relative(relative) or not str(relative).startswith("evidence/"):
        errors.append(f"{label}.path must be a safe package-relative evidence/ path")
        return None
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        errors.append(f"{label}.sha256 must be a lowercase SHA-256 digest")
        return None
    target = package / str(relative)
    current = package
    for part in Path(str(relative)).parts:
        try:
            if part not in {entry.name for entry in current.iterdir()}:
                errors.append(f"{label}.path must preserve exact on-disk case: {relative}")
                return None
        except OSError as exc:
            errors.append(f"{label}.path cannot be inspected: {relative}: {exc}")
            return None
        current /= part
        if current.is_symlink():
            errors.append(f"{label}.path must not use symlinks: {relative}")
            return None
    try:
        target.resolve().relative_to(package.resolve())
    except ValueError:
        errors.append(f"{label}.path escapes the package")
        return None
    try:
        if not target.is_file():
            errors.append(f"missing {label} file: {relative}")
            return None
        if target.stat().st_size == 0:
            errors.append(f"{label} file is empty: {relative}")
            return None
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            errors.append(f"{label}.sha256 does not match {relative}")
            return None
    except OSError as exc:
        errors.append(f"{label} file cannot be inspected: {relative}: {exc}")
        return None
    return target, {"path": str(relative), "sha256": digest}


def _bundle_ref(
    envelope_relative: str, value: Any, label: str, errors: list[str]
) -> dict[str, str] | None:
    if not isinstance(value, dict):
        errors.append(f"PIQ two-host envelope {label} must be an artifact reference")
        return None
    path = value.get("path")
    digest = value.get("sha256")
    if not _safe_relative(path) or not isinstance(digest, str):
        errors.append(f"PIQ two-host envelope {label} is malformed")
        return None
    package_path = (Path(envelope_relative).parent / str(path)).as_posix()
    return {"path": package_path, "sha256": digest}


def validate_two_host_functional_receipt(package: Path, data: Any) -> list[str]:
    """Validate retained two-host census evidence without upgrading its authority."""
    package = package.resolve()
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["PIQ two-host functional receipt must be a JSON object"]
    expected_fields = {
        "schema_version",
        "kind",
        "authority",
        "qualification",
        "envelope",
        "run_id",
        "attempt_ids",
        "aggregations",
        "execution",
        "correctness",
        "custody",
    }
    if set(data) != expected_fields:
        errors.append("PIQ two-host functional receipt fields do not match schema version 1")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        errors.append("PIQ two-host functional receipt schema_version must equal 1")
    if data.get("kind") != "piq_two_host_functional_evidence":
        errors.append("PIQ two-host functional receipt kind is unsupported")
    if data.get("authority") != "retained_producer_assertion":
        errors.append(
            "PIQ two-host functional receipt authority must equal retained_producer_assertion"
        )
    if data.get("qualification") != "not_established":
        errors.append("PIQ two-host functional receipt cannot claim qualification")
    run_id = data.get("run_id")
    if not _nonempty(run_id):
        errors.append("PIQ two-host functional receipt run_id must be a nonempty string")

    envelope_result = _validated_ref(package, data.get("envelope"), "PIQ two-host envelope", errors)
    envelope: dict[str, Any] = {}
    envelope_relative = ""
    if envelope_result is not None:
        envelope_path, envelope_ref = envelope_result
        envelope_relative = envelope_ref["path"]
        if Path(envelope_relative).name != "envelope.json":
            errors.append("PIQ two-host envelope must name an exact envelope.json")
        bundle_errors = validate_bundle(envelope_path.parent)
        errors.extend(f"PIQ two-host bundle: {error}" for error in bundle_errors)
        loaded = _load_json(envelope_path, "PIQ two-host envelope", errors)
        if isinstance(loaded, dict):
            envelope = loaded

    execution_ref = _bundle_ref(
        envelope_relative,
        envelope.get("workload_evidence", {}).get("execution")
        if isinstance(envelope.get("workload_evidence"), dict)
        else None,
        "execution reference",
        errors,
    )
    correctness_ref = _bundle_ref(
        envelope_relative, envelope.get("correctness"), "correctness reference", errors
    )
    execution_result = _validated_ref(
        package, data.get("execution"), "PIQ two-host execution", errors
    )
    correctness_result = _validated_ref(
        package, data.get("correctness"), "PIQ two-host correctness", errors
    )
    if execution_ref is not None and (
        execution_result is None or execution_result[1] != execution_ref
    ):
        errors.append("PIQ two-host execution must exactly match the envelope execution reference")
    if correctness_ref is not None and (
        correctness_result is None or correctness_result[1] != correctness_ref
    ):
        errors.append("PIQ two-host correctness must exactly match the envelope correctness reference")

    execution = (
        _load_json(execution_result[0], "PIQ two-host execution", errors)
        if execution_result is not None
        else None
    )
    correctness = (
        _load_json(correctness_result[0], "PIQ two-host correctness", errors)
        if correctness_result is not None
        else None
    )
    if not isinstance(execution, dict):
        errors.append("PIQ two-host execution must be a JSON object")
        execution = {}
    if execution.get("run_id") != run_id:
        errors.append("PIQ two-host execution run_id must match the receipt")
    resources = execution.get("resources")
    if not isinstance(resources, dict) or (
        resources.get("executor"),
        resources.get("nodeCount"),
        resources.get("processCount"),
        resources.get("acceleratorCount"),
    ) != ("assigned-cpu-hosts", 2, 2, 0):
        errors.append(
            "PIQ two-host functional evidence requires assigned-cpu-hosts, two nodes, two processes, and zero accelerators"
        )

    attempt_ids = data.get("attempt_ids")
    if (
        not isinstance(attempt_ids, list)
        or not attempt_ids
        or not all(_nonempty(value) for value in attempt_ids)
        or len(attempt_ids) != len(set(attempt_ids))
    ):
        errors.append("PIQ two-host functional receipt attempt_ids must be a distinct nonempty string list")
        attempt_ids = []
    aggregations = data.get("aggregations")
    if not isinstance(aggregations, list) or len(aggregations) != len(attempt_ids):
        errors.append(
            "PIQ two-host functional receipt aggregations must contain one reference per selected attempt"
        )
        aggregations = []
    raw_envelope_attempts = envelope.get("attempts")
    if not isinstance(raw_envelope_attempts, list):
        errors.append("PIQ two-host envelope attempts must be a list")
        raw_envelope_attempts = []
    raw_execution_attempts = execution.get("attempts")
    if not isinstance(raw_execution_attempts, list):
        errors.append("PIQ two-host execution attempts must be a list")
        raw_execution_attempts = []
    envelope_attempts = {
        item.get("id"): item
        for item in raw_envelope_attempts
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    execution_attempts = {
        item.get("attempt_id"): item
        for item in raw_execution_attempts
        if isinstance(item, dict) and isinstance(item.get("attempt_id"), str)
    }
    correctness_check = correctness.get("check") if isinstance(correctness, dict) else None
    selected_bindings: dict[str, dict[str, Any]] = {}
    for attempt_id in attempt_ids:
        attempt = envelope_attempts.get(attempt_id)
        observed = execution_attempts.get(attempt_id)
        if attempt is None or observed is None:
            errors.append(f"PIQ two-host selected attempt is missing: {attempt_id}")
            continue
        if attempt.get("run_id") != run_id:
            errors.append(f"PIQ two-host selected attempt run_id differs: {attempt_id}")
        if attempt.get("status") != "succeeded" or attempt.get("accepted") is not True:
            errors.append(
                f"PIQ two-host selected attempt must be succeeded and accepted: {attempt_id}"
            )
        census = observed.get("rank_census")
        if not isinstance(census, list):
            errors.append(f"PIQ two-host selected attempt lacks a rank census: {attempt_id}")
            continue
        hosts = [item.get("host") for item in census if isinstance(item, dict)]
        ranks = [item.get("rank") for item in census if isinstance(item, dict)]
        if (
            len(census) != 2
            or len(ranks) != 2
            or not all(type(rank) is int for rank in ranks)
            or sorted(ranks) != [0, 1]
            or any(not _nonempty(host) or host != host.strip() for host in hosts)
            or len(set(hosts)) < 2
        ):
            errors.append(
                f"PIQ two-host selected attempt must report two distinct hosts in one complete rank census: {attempt_id}"
            )
        rank_outputs: list[dict[str, str]] = []
        rank_output_identities: set[tuple[int, int]] = set()
        for rank in sorted(
            (item for item in census if isinstance(item, dict)),
            key=lambda item: item.get("rank") if type(item.get("rank")) is int else 65,
        ):
            output = rank.get("output")
            if isinstance(output, dict):
                rank_output = _bundle_ref(
                    envelope_relative,
                    {
                        "path": output.get("artifactName"),
                        "sha256": output.get("sha256"),
                    },
                    f"selected attempt {attempt_id} rank output",
                    errors,
                )
                if rank_output is not None:
                    validated_rank_output = _validated_ref(
                        package,
                        rank_output,
                        f"selected attempt {attempt_id} rank output",
                        errors,
                    )
                    if validated_rank_output is not None:
                        identity_stat = validated_rank_output[0].stat()
                        identity = (identity_stat.st_dev, identity_stat.st_ino)
                        if identity in rank_output_identities:
                            errors.append(
                                f"PIQ two-host selected attempt rank outputs must be distinct physical files: {attempt_id}"
                            )
                        rank_output_identities.add(identity)
                        rank_outputs.append(validated_rank_output[1])

        raw_reference = attempt.get("raw_result")
        raw_ref = _bundle_ref(
            envelope_relative,
            raw_reference,
            f"selected attempt {attempt_id} raw result",
            errors,
        )
        raw_result = None
        if raw_ref is not None:
            raw_validated = _validated_ref(
                package, raw_ref, f"selected attempt {attempt_id} raw result", errors
            )
            if raw_validated is not None:
                raw_result = _load_json(
                    raw_validated[0], f"selected attempt {attempt_id} raw result", errors
                )
        native = raw_result.get("native_attempt") if isinstance(raw_result, dict) else None
        native_output = native.get("output") if isinstance(native, dict) else None
        aggregate_output = _bundle_ref(
            envelope_relative,
            {
                "path": native_output.get("artifactName") if isinstance(native_output, dict) else None,
                "sha256": native_output.get("sha256") if isinstance(native_output, dict) else None,
            },
            f"selected attempt {attempt_id} aggregate output",
            errors,
        )
        operand_name = "reference" if attempt.get("role") == "baseline" else "candidate"
        correctness_operand = _bundle_ref(
            envelope_relative,
            correctness_check.get(operand_name) if isinstance(correctness_check, dict) else None,
            f"selected attempt {attempt_id} correctness operand",
            errors,
        )
        if aggregate_output is not None and correctness_operand is not None and aggregate_output != correctness_operand:
            errors.append(
                f"PIQ two-host selected attempt aggregate output must equal its correctness operand: {attempt_id}"
            )
        selected_bindings[attempt_id] = {
            "rank_outputs": rank_outputs,
            "aggregate_output": aggregate_output,
            "correctness_operand": correctness_operand,
        }

    if not isinstance(correctness, dict) or correctness.get("accepted") is not True:
        errors.append("PIQ two-host envelope correctness must be accepted")
    check = correctness_check
    if not isinstance(check, dict) or check.get("verification") != "producer-asserted":
        errors.append("PIQ two-host correctness must retain producer-asserted formal evidence")
    report_ref = check.get("report") if isinstance(check, dict) else None
    report_result = _validated_ref(
        package,
        _bundle_ref(envelope_relative, report_ref, "correctness report", errors),
        "PIQ two-host correctness report",
        errors,
    ) if report_ref is not None else None
    if report_result is not None:
        report = _load_json(report_result[0], "PIQ two-host correctness report", errors)
        if not isinstance(report, dict) or report.get("status") != "pass":
            errors.append("PIQ two-host formal correctness report must pass")

    aggregation_ids: list[str] = []
    for index, aggregation_ref in enumerate(aggregations):
        result = _validated_ref(
            package,
            aggregation_ref,
            f"PIQ two-host aggregations[{index}]",
            errors,
        )
        if result is None:
            continue
        aggregation = _load_json(
            result[0], f"PIQ two-host aggregations[{index}]", errors
        )
        if not isinstance(aggregation, dict):
            errors.append(f"PIQ two-host aggregations[{index}] must be a JSON object")
            continue
        expected_aggregation_fields = {
            "schema_version",
            "kind",
            "authority",
            "status",
            "run_id",
            "attempt_id",
            "rank_outputs",
            "aggregate_output",
            "correctness_operand",
        }
        if set(aggregation) != expected_aggregation_fields:
            errors.append(
                f"PIQ two-host aggregations[{index}] fields do not match schema version 1"
            )
        if type(aggregation.get("schema_version")) is not int or aggregation.get("schema_version") != 1:
            errors.append(f"PIQ two-host aggregations[{index}] schema_version must equal 1")
        if aggregation.get("kind") != "piq_rank_aggregation_evidence":
            errors.append(f"PIQ two-host aggregations[{index}] kind is unsupported")
        if aggregation.get("authority") != "retained_producer_assertion":
            errors.append(
                f"PIQ two-host aggregations[{index}] authority must equal retained_producer_assertion"
            )
        if aggregation.get("status") != "pass":
            errors.append(f"PIQ two-host aggregations[{index}] status must equal pass")
        if aggregation.get("run_id") != run_id:
            errors.append(f"PIQ two-host aggregations[{index}] run_id must match the receipt")
        attempt_id = aggregation.get("attempt_id")
        if not _nonempty(attempt_id):
            errors.append(f"PIQ two-host aggregations[{index}] attempt_id must be nonempty")
            continue
        aggregation_ids.append(attempt_id)
        binding = selected_bindings.get(attempt_id)
        if binding is None:
            errors.append(
                f"PIQ two-host aggregation references an unselected attempt: {attempt_id}"
            )
            continue
        for field in ("rank_outputs", "aggregate_output", "correctness_operand"):
            if aggregation.get(field) != binding[field]:
                errors.append(
                    f"PIQ two-host aggregation {field} does not match selected attempt {attempt_id}"
                )
    if aggregation_ids != attempt_ids:
        errors.append(
            "PIQ two-host aggregation attempt IDs must exactly match selected attempt_ids in order"
        )

    custody = data.get("custody")
    if not isinstance(custody, dict) or set(custody) != {
        "readback",
        "export",
        "import_receipt",
        "cleanup",
    }:
        errors.append(
            "PIQ two-host custody must contain exactly readback, export, import_receipt, and cleanup"
        )
    else:
        custody_results = {
            name: _validated_ref(
                package, custody.get(name), f"PIQ two-host custody {name}", errors
            )
            for name in ("readback", "export", "import_receipt", "cleanup")
        }
        paths = [result[1]["path"] for result in custody_results.values() if result is not None]
        if len(paths) != len(set(paths)):
            errors.append("PIQ two-host custody roles must reference distinct retained files")
        envelope_sha256 = (
            data.get("envelope", {}).get("sha256")
            if isinstance(data.get("envelope"), dict)
            else None
        )
        for name in ("readback", "export", "cleanup"):
            result = custody_results.get(name)
            if result is None:
                continue
            document = _load_json(result[0], f"PIQ two-host custody {name}", errors)
            expected = {
                "schema_version": 1,
                "kind": f"piq_functional_{name}_receipt",
                "authority": "retained_producer_assertion",
                "state": "COMPLETE",
                "run_id": run_id,
                "attempt_ids": attempt_ids,
                "envelope_sha256": envelope_sha256,
            }
            if not _same_json(document, expected):
                errors.append(
                    f"PIQ two-host custody {name} does not match its exact run/envelope projection"
                )
        import_result = custody_results.get("import_receipt")
        if import_result is not None:
            imported = _load_json(
                import_result[0], "PIQ two-host custody import_receipt", errors
            )
            expected_import_path = (
                Path(envelope_relative).parent / "import-receipt.json"
            ).as_posix()
            expected_experiment = (
                envelope.get("experiment", {}).get("id")
                if isinstance(envelope.get("experiment"), dict)
                else None
            )
            if (
                import_result[1]["path"] != expected_import_path
                or not isinstance(imported, dict)
                or set(imported) != {
                    "schema_version",
                    "state",
                    "publishable",
                    "experiment_id",
                    "envelope_sha256",
                    "imported_at",
                    "qualification",
                }
                or type(imported.get("schema_version")) is not int
                or imported.get("schema_version") != 1
                or imported.get("state") != "diagnostic_imported"
                or imported.get("publishable") is not False
                or imported.get("qualification") != "not_established"
                or imported.get("experiment_id") != expected_experiment
                or imported.get("envelope_sha256") != envelope_sha256
                or not _canonical_utc_timestamp(imported.get("imported_at"))
            ):
                errors.append(
                    "PIQ two-host custody import_receipt must be the exact canonical private-import receipt for the diagnostic envelope"
                )
    return errors


def validate_declared_two_host_functional(
    package: Path, manifest: Any, *, admissions: Any | None = None
) -> list[str]:
    producer_errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    declared = "piq_two_host_functional_receipt" in manifest
    receipt_path = package / RECEIPT_PATH
    data: Any = None
    if not declared:
        if receipt_path.exists() or receipt_path.is_symlink():
            producer_errors.append(
                f"undeclared PIQ two-host functional receipt must be removed or declared: {RECEIPT_PATH}"
            )
    else:
        relative = manifest.get("piq_two_host_functional_receipt")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 2
        ):
            producer_errors.append(
                "manifest.piq_two_host_functional_receipt is supported only by schema 2"
            )
        if relative != RECEIPT_PATH:
            producer_errors.append(
                f"manifest.piq_two_host_functional_receipt must equal {RECEIPT_PATH!r}"
            )
        elif receipt_path.is_symlink() or not receipt_path.is_file():
            producer_errors.append(
                f"missing nonsymlink PIQ two-host functional receipt: {RECEIPT_PATH}"
            )
        else:
            data = _load_json(
                receipt_path, "PIQ two-host functional receipt", producer_errors
            )
            producer_errors.extend(validate_two_host_functional_receipt(package, data))

    if __package__:
        from .check_piq_two_host_owner_authority import (
            validate_declared_owner_authority,
        )
    else:
        from check_piq_two_host_owner_authority import validate_declared_owner_authority

    owner_errors, owner_valid = validate_declared_owner_authority(
        package, manifest, data, admissions=admissions
    )
    errors = [*producer_errors, *owner_errors]
    if (
        isinstance(data, dict)
        and data.get("authority") == "retained_producer_assertion"
    ):
        if producer_errors or not owner_valid:
            errors.append(READINESS_BLOCK)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    usage = f"usage: python3 {Path(__file__).name} PACKAGE"
    if "-h" in argv or "--help" in argv:
        print(usage)
        print("Check retained two-host assertions and the reviewed admission tuple.")
        print("Does not independently establish execution, physical hosts or traffic.")
        return 0
    if len(argv) != 1:
        print(usage, file=sys.stderr)
        return 2
    package = Path(argv[0]).resolve()
    errors: list[str] = []
    manifest = _load_json(package / "manifest.json", "manifest", errors)
    if (
        isinstance(manifest, dict)
        and "piq_two_host_functional_receipt" not in manifest
        and "piq_two_host_owner_authority_receipt" not in manifest
        and not (package / RECEIPT_PATH).exists()
        and not (package / RECEIPT_PATH).is_symlink()
    ):
        if __package__:
            from .check_piq_two_host_owner_authority import OWNER_RECEIPT_PATH
        else:
            from check_piq_two_host_owner_authority import OWNER_RECEIPT_PATH
        owner_path = package / OWNER_RECEIPT_PATH
        if not owner_path.exists() and not owner_path.is_symlink():
            print(f"NOT_DECLARED {package}", file=sys.stderr)
            return 1
    errors.extend(validate_declared_two_host_functional(package, manifest))
    readiness_blocked = READINESS_BLOCK in errors
    errors = [error for error in errors if error != READINESS_BLOCK]
    if errors:
        print(f"FAIL {package} ({len(errors)} problems)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if readiness_blocked:
        suffix = (
            "producer-assertion consistency only. Package readiness remains blocked"
        )
    elif (
        isinstance(manifest, dict)
        and "piq_two_host_owner_authority_receipt" in manifest
    ):
        suffix = (
            "retained owner-attested assertions and reviewed admission tuple only. "
            "Does not independently establish execution, physical hosts or traffic. "
            "Admission is source review, not cryptographic authentication. "
            "Scientific qualification is not established"
        )
    else:
        suffix = "validated"
    print(f"PASS {package} ({suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
