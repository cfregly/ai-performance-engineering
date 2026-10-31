#!/usr/bin/env python3
"""Validate the retained diagnostic finding that connects a problem to PIQ evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


DIAGNOSTIC_FINDING_PATH = "evidence/diagnostic-finding.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LAYERS = (
    "storage_and_input",
    "process_cpu_and_memory",
    "accelerator",
    "communication",
    "inference_or_training_runtime",
    "application_or_job",
)
LAYER_STATES = {"OBSERVED", "UNAVAILABLE", "NOT_APPLICABLE"}


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_list(
    value: Any, label: str, errors: list[str], *, nonempty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        errors.append(f"{label} must be a {qualifier}list of nonempty strings")
        return []
    if not all(nonempty_string(item) for item in value):
        errors.append(f"{label} must contain only nonempty strings")
        return []
    return value


def required_string(
    parent: Any, field: str, label: str, errors: list[str]
) -> str | None:
    if not isinstance(parent, dict):
        errors.append(f"{label} must be an object")
        return None
    value = parent.get(field)
    if not nonempty_string(value):
        errors.append(f"{label}.{field} must be a nonempty string")
        return None
    return value


def safe_relative(value: Any) -> bool:
    if not nonempty_string(value) or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_ref(package: Path, value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object with path and sha256")
        return
    relative = value.get("path")
    if not safe_relative(relative) or not str(relative).startswith("evidence/"):
        errors.append(f"{label}.path must be a safe package-relative evidence/ path")
        return
    target = package / relative
    try:
        target.resolve().relative_to(package.resolve())
    except ValueError:
        errors.append(f"{label}.path escapes the package")
        return
    current = package
    package_symlink = False
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            package_symlink = True
            break
    if package_symlink:
        errors.append(f"{label}.path must not use symlinks: {relative}")
    elif not target.is_file():
        errors.append(f"missing {label} file: {relative}")
    elif target.stat().st_size == 0:
        errors.append(f"{label} file is empty: {relative}")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex digest")
    elif target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        errors.append(f"{label}.sha256 does not match {relative}")


def ref_list(
    package: Path, value: Any, label: str, errors: list[str], *, nonempty: bool = False
) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        errors.append(f"{label} must be a {qualifier}list of evidence references")
        return
    for index, item in enumerate(value):
        validate_ref(package, item, f"{label}[{index}]", errors)


def validate_diagnostic_finding(package: Path, data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["diagnostic finding must be a JSON object"]
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        errors.append("diagnostic finding schema_version must equal 1")
    if data.get("kind") != "piq_diagnostic_finding":
        errors.append("diagnostic finding kind must equal 'piq_diagnostic_finding'")
    if not nonempty_string(data.get("example_id")):
        errors.append("diagnostic finding example_id must be a nonempty string")

    problem = data.get("problem")
    for field in (
        "user_visible_symptom",
        "reader_decision",
        "application_or_job_boundary",
    ):
        required_string(problem, field, "diagnostic finding problem", errors)

    detection = data.get("piq_detection")
    affected_layers: list[str] = []
    for field in ("detectable_issue", "observed_signal", "decision_enabled"):
        required_string(detection, field, "diagnostic finding piq_detection", errors)
    if isinstance(detection, dict):
        affected_layers = string_list(
            detection.get("affected_layers"),
            "diagnostic finding piq_detection.affected_layers",
            errors,
            nonempty=True,
        )
        if affected_layers:
            unknown_layers = sorted(set(affected_layers) - set(LAYERS))
            if unknown_layers:
                errors.append(
                    "diagnostic finding piq_detection.affected_layers contains unknown "
                    f"layers: {', '.join(unknown_layers)}"
                )
            if len(affected_layers) != len(set(affected_layers)):
                errors.append(
                    "diagnostic finding piq_detection.affected_layers must not contain duplicates"
                )
            if "application_or_job" not in affected_layers or not any(
                layer != "application_or_job" for layer in affected_layers
            ):
                errors.append(
                    "diagnostic finding piq_detection.affected_layers must include "
                    "application_or_job and at least one supporting layer"
                )
        ref_list(
            package,
            detection.get("evidence_refs"),
            "diagnostic finding piq_detection.evidence_refs",
            errors,
            nonempty=True,
        )

    prediction = data.get("predicted_symptom")
    required_string(
        prediction, "hypothesis", "diagnostic finding predicted_symptom", errors
    )
    if isinstance(prediction, dict):
        string_list(
            prediction.get("expected_observations"),
            "diagnostic finding predicted_symptom.expected_observations",
            errors,
            nonempty=True,
        )
        ref_list(
            package,
            prediction.get("evidence_refs"),
            "diagnostic finding predicted_symptom.evidence_refs",
            errors,
            nonempty=True,
        )

    competing = data.get("competing_explanation")
    required_string(
        competing, "hypothesis", "diagnostic finding competing_explanation", errors
    )
    if isinstance(competing, dict):
        string_list(
            competing.get("distinguishing_observations"),
            "diagnostic finding competing_explanation.distinguishing_observations",
            errors,
            nonempty=True,
        )
        ref_list(
            package,
            competing.get("evidence_refs"),
            "diagnostic finding competing_explanation.evidence_refs",
            errors,
            nonempty=True,
        )

    change = data.get("fixed_change")
    for field in ("reference_identity", "candidate_identity", "sole_intended_change"):
        required_string(change, field, "diagnostic finding fixed_change", errors)
    if isinstance(change, dict):
        string_list(
            change.get("held_inputs_and_settings"),
            "diagnostic finding fixed_change.held_inputs_and_settings",
            errors,
            nonempty=True,
        )
        string_list(
            change.get("unavoidable_differences"),
            "diagnostic finding fixed_change.unavoidable_differences",
            errors,
        )

    outcome = data.get("complete_outcome")
    for field in (
        "start_event",
        "stop_event",
        "setup_accounting",
        "cleanup_accounting",
        "failed_work_accounting",
    ):
        required_string(outcome, field, "diagnostic finding complete_outcome", errors)
    if isinstance(outcome, dict):
        string_list(
            outcome.get("excluded_costs"),
            "diagnostic finding complete_outcome.excluded_costs",
            errors,
        )

    correctness = data.get("correctness")
    for field in ("independent_oracle", "acceptance_rule", "covered_work"):
        required_string(correctness, field, "diagnostic finding correctness", errors)
    if isinstance(correctness, dict):
        if correctness.get("result") != "PASS":
            errors.append("diagnostic finding correctness.result must equal 'PASS'")
        validate_ref(
            package,
            correctness.get("evidence"),
            "diagnostic finding correctness.evidence",
            errors,
        )

    layers = data.get("layer_evidence")
    observed_layers = 0
    if not isinstance(layers, dict):
        errors.append("diagnostic finding layer_evidence must be an object")
    else:
        if set(layers) != set(LAYERS):
            errors.append(
                f"diagnostic finding layer_evidence must contain exactly: {', '.join(LAYERS)}"
            )
        for name in LAYERS:
            layer = layers.get(name)
            label = f"diagnostic finding layer_evidence.{name}"
            if not isinstance(layer, dict):
                errors.append(f"{label} must be an object")
                continue
            state = layer.get("state")
            if state not in LAYER_STATES:
                errors.append(
                    f"{label}.state must be OBSERVED, UNAVAILABLE, or NOT_APPLICABLE"
                )
                continue
            observations = string_list(
                layer.get("observations"), f"{label}.observations", errors
            )
            gaps = string_list(layer.get("gaps"), f"{label}.gaps", errors)
            evidence_refs = layer.get("evidence_refs")
            ref_list(package, evidence_refs, f"{label}.evidence_refs", errors)
            if state == "OBSERVED":
                observed_layers += 1
                if not observations:
                    errors.append(f"{label} OBSERVED state requires observations")
                if not isinstance(evidence_refs, list) or not evidence_refs:
                    errors.append(f"{label} OBSERVED state requires evidence_refs")
            elif not gaps:
                errors.append(
                    f"{label} {state} state requires a concrete gap or reason"
                )
        application = layers.get("application_or_job")
        if not isinstance(application, dict) or application.get("state") != "OBSERVED":
            errors.append(
                "diagnostic finding application_or_job layer must be OBSERVED"
            )
        if observed_layers < 2:
            errors.append(
                "diagnostic finding must observe the application/job layer and at least one supporting layer"
            )
        for name in affected_layers:
            if name in LAYERS and (
                not isinstance(layers.get(name), dict)
                or layers[name].get("state") != "OBSERVED"
            ):
                errors.append(
                    "diagnostic finding piq_detection affected layer "
                    f"{name} must have OBSERVED layer evidence"
                )

    custody = data.get("piq_custody")
    if not isinstance(custody, dict):
        errors.append("diagnostic finding piq_custody must be an object")
    else:
        required_string(
            custody,
            "source_runtime_input_identities",
            "diagnostic finding piq_custody",
            errors,
        )
        string_list(
            custody.get("run_ids"),
            "diagnostic finding piq_custody.run_ids",
            errors,
            nonempty=True,
        )
        string_list(
            custody.get("attempt_ids"),
            "diagnostic finding piq_custody.attempt_ids",
            errors,
            nonempty=True,
        )
        ref_list(
            package,
            custody.get("artifact_refs"),
            "diagnostic finding piq_custody.artifact_refs",
            errors,
            nonempty=True,
        )
        for field in ("readback", "export", "cleanup", "qualification_receipt"):
            validate_ref(
                package,
                custody.get(field),
                f"diagnostic finding piq_custody.{field}",
                errors,
            )
        report = custody.get("report")
        if report is not None:
            validate_ref(
                package, report, "diagnostic finding piq_custody.report", errors
            )

    disposition = data.get("disposition")
    if not isinstance(disposition, dict):
        errors.append("diagnostic finding disposition must be an object")
    else:
        if disposition.get("value") != "SUPPORTED":
            errors.append(
                "package-ready diagnostic finding disposition.value must equal 'SUPPORTED'"
            )
        string_list(
            disposition.get("supported_observations"),
            "diagnostic finding disposition.supported_observations",
            errors,
            nonempty=True,
        )
        string_list(
            disposition.get("remaining_unknowns"),
            "diagnostic finding disposition.remaining_unknowns",
            errors,
        )
        ref_list(
            package,
            disposition.get("evidence_refs"),
            "diagnostic finding disposition.evidence_refs",
            errors,
            nonempty=True,
        )

    tradeoff = data.get("tradeoff")
    for field in (
        "measured_benefit",
        "applicability",
        "simpler_option",
        "when_to_use_or_skip",
        "next_measurement_or_action",
    ):
        required_string(tradeoff, field, "diagnostic finding tradeoff", errors)
    if isinstance(tradeoff, dict):
        string_list(
            tradeoff.get("added_cost_or_responsibility"),
            "diagnostic finding tradeoff.added_cost_or_responsibility",
            errors,
            nonempty=True,
        )
    return errors


def validate_package_bindings(
    package: Path, manifest: dict[str, Any], finding: dict[str, Any], errors: list[str]
) -> None:
    """Bind the diagnostic interpretation to the package's qualified PIQ evidence."""
    if manifest.get("content_state") != "measured":
        errors.append(
            "declared diagnostic finding requires manifest.content_state 'measured'"
        )

    custody = finding.get("piq_custody")
    if not isinstance(custody, dict):
        return
    qualification = custody.get("qualification_receipt")
    qualification_path = (
        qualification.get("path") if isinstance(qualification, dict) else None
    )
    if qualification_path != manifest.get("measurement_receipt"):
        errors.append(
            "diagnostic finding piq_custody.qualification_receipt.path must equal "
            "manifest.measurement_receipt"
        )

    envelope_relative = manifest.get("piq_experiment_evidence")
    if (
        not safe_relative(envelope_relative)
        or Path(str(envelope_relative)).name != "envelope.json"
    ):
        errors.append(
            "declared diagnostic finding requires manifest.piq_experiment_evidence to name an exact envelope.json"
        )
        return
    artifacts = custody.get("artifact_refs")
    envelope_ref = None
    if isinstance(artifacts, list):
        envelope_ref = next(
            (
                item
                for item in artifacts
                if isinstance(item, dict) and item.get("path") == envelope_relative
            ),
            None,
        )
    if envelope_ref is None:
        errors.append(
            "diagnostic finding piq_custody.artifact_refs must include the exact "
            "manifest.piq_experiment_evidence envelope"
        )
        return

    envelope_path = package / str(envelope_relative)
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            f"cannot read diagnostic finding PIQ envelope {envelope_relative}: {exc}"
        )
        return
    attempts = envelope.get("attempts") if isinstance(envelope, dict) else None
    if (
        not isinstance(attempts, list)
        or not attempts
        or not all(isinstance(item, dict) for item in attempts)
    ):
        errors.append(
            "diagnostic finding PIQ envelope attempts must be a nonempty list of objects"
        )
        return
    expected_attempt_ids = [item.get("id") for item in attempts]
    expected_run_ids: list[Any] = []
    for item in attempts:
        run_id = item.get("run_id")
        if run_id not in expected_run_ids:
            expected_run_ids.append(run_id)
    if custody.get("attempt_ids") != expected_attempt_ids:
        errors.append(
            "diagnostic finding piq_custody.attempt_ids must exactly match the declared PIQ envelope order"
        )
    if custody.get("run_ids") != expected_run_ids:
        errors.append(
            "diagnostic finding piq_custody.run_ids must exactly match the declared PIQ envelope first-occurrence order"
        )


def validate_declared_diagnostic_finding(package: Path, manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return errors
    required = manifest.get("diagnostic_finding_required")
    relative = manifest.get("diagnostic_finding")
    piq_backed = manifest.get("piq_experiment_evidence") is not None
    if required is None and relative is None and not piq_backed:
        return errors
    if required is not True:
        errors.append(
            "manifest diagnostic_finding declaration must set diagnostic_finding_required to true"
        )
    if relative != DIAGNOSTIC_FINDING_PATH:
        errors.append(
            f"manifest.diagnostic_finding must equal {DIAGNOSTIC_FINDING_PATH!r}"
        )
        return errors
    path = package / relative
    if not path.is_file():
        errors.append(f"missing diagnostic finding: {relative}")
        return errors
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid diagnostic finding {path}: {exc}")
        return errors
    errors.extend(validate_diagnostic_finding(package, data))
    if isinstance(data, dict):
        validate_package_bindings(package, manifest, data, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "usage: python3 tools/check_diagnostic_finding.py PACKAGE", file=sys.stderr
        )
        return 2
    package = Path(argv[0]).resolve()
    manifest_path = package / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"FAIL {package}: invalid manifest: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(f"FAIL {package}: manifest must be a JSON object", file=sys.stderr)
        return 1
    errors = validate_declared_diagnostic_finding(package, manifest)
    if errors:
        print(f"FAIL {package} ({len(errors)} problems)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"PASS {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
