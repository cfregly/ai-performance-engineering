"""Read-only dashboard projection for optimization campaign evidence."""

from __future__ import annotations

import hmac
from pathlib import Path
from statistics import median
from typing import Any

from core.optimization.campaign import (
    CampaignConfig,
    CampaignWorkspace,
    ExperimentRecord,
    active_beam,
    budget_status,
    current_incumbent,
    evaluate_experiment,
    promotion_frontier,
    record_artifact_integrity_errors,
    resolve_workspace_artifact,
    sha256_file,
)


def _improvement_pct(control: float, candidate: float, direction: str) -> float:
    if direction == "lower":
        return (control - candidate) / control * 100.0
    return (candidate - control) / control * 100.0


def _artifact_summary(
    workspace_root: Path,
    record: ExperimentRecord,
    artifact: str,
    *,
    role: str,
) -> dict[str, Any]:
    digest = record.artifact_sha256.get(artifact)
    downloadable = False
    try:
        resolved = _resolve_artifact_path(workspace_root, artifact)
    except (FileNotFoundError, ValueError):
        resolved = None
    if resolved is not None and digest:
        downloadable = hmac.compare_digest(sha256_file(resolved), digest)
    return {
        "role": role,
        "path": artifact,
        "sha256": digest,
        "downloadable": downloadable,
    }


def _resolve_artifact_path(workspace_root: Path, artifact: str) -> Path:
    return resolve_workspace_artifact(workspace_root, artifact)


def resolve_campaign_artifact(workspace_root: Path, artifact: str) -> Path:
    """Resolve one ledger-declared, hash-valid artifact inside a campaign root."""

    workspace = CampaignWorkspace(workspace_root)
    config = workspace.load_config()
    records = workspace.ledger.latest()
    current_incumbent(config, records, workspace_root=workspace.root)
    matching_records: list[ExperimentRecord] = []
    matching_digests: set[str] = set()
    for record in records:
        declared = {*record.raw_artifacts, *record.profile_artifacts}
        if artifact in declared:
            digest = record.artifact_sha256.get(artifact)
            if digest:
                matching_digests.add(digest)
                matching_records.append(record)
    if not matching_digests:
        raise ValueError("artifact is not a hashed reference in the campaign ledger")
    if len(matching_digests) != 1:
        raise ValueError("artifact has conflicting hashes in the campaign ledger")
    for record in matching_records:
        integrity_errors = record_artifact_integrity_errors(
            record,
            workspace.root,
            require_diff_artifact=bool(record.measurements)
            and (config.require_git_provenance or config.require_candidate_diff),
            require_record_binding=bool(record.measurements),
        )
        if integrity_errors:
            raise ValueError(
                f"experiment {record.experiment_id} fails evidence integrity: "
                + ", ".join(integrity_errors)
            )
    path = _resolve_artifact_path(workspace.root, artifact)
    expected_digest = next(iter(matching_digests))
    if not hmac.compare_digest(sha256_file(path), expected_digest):
        raise ValueError("campaign artifact hash does not match the ledger")
    return path


def _case_rows(
    config: CampaignConfig,
    record: ExperimentRecord,
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    case_improvements = decision.get("case_improvements_pct") or {}
    confidence_intervals = decision.get("case_improvement_ci_pct") or {}
    rows: list[dict[str, Any]] = []
    for case_id, measurement in sorted(record.measurements.items()):
        control_median = float(median(measurement.control))
        candidate_median = float(median(measurement.candidate))
        improvement = case_improvements.get(case_id)
        if improvement is None:
            improvement = _improvement_pct(
                control_median,
                candidate_median,
                config.direction,
            )
        confidence_interval = confidence_intervals.get(case_id)
        frozen = case_id in config.frozen_cases
        point_violation = frozen and improvement < -config.max_case_regression_pct
        confidence_violation = bool(
            frozen
            and decision.get("confidence_required")
            and isinstance(confidence_interval, list)
            and confidence_interval
            and confidence_interval[0] < -config.max_case_regression_pct
        )
        rows.append(
            {
                "case_id": case_id,
                "primary": case_id in config.primary_cases,
                "frozen": frozen,
                "control_median": control_median,
                "candidate_median": candidate_median,
                "improvement_pct": improvement,
                "improvement_ci_pct": confidence_interval,
                "control_trials": len(measurement.control),
                "candidate_trials": len(measurement.candidate),
                "frozen_case_violation": point_violation or confidence_violation,
            }
        )
    return rows


def _record_summary(
    workspace_root: Path,
    config: CampaignConfig,
    record: ExperimentRecord,
    *,
    incumbent_id: str | None,
) -> dict[str, Any]:
    gate = evaluate_experiment(config, record).to_dict()
    integrity_errors = record_artifact_integrity_errors(
        record,
        workspace_root,
        require_diff_artifact=bool(record.measurements)
        and (config.require_git_provenance or config.require_candidate_diff),
        require_record_binding=bool(record.measurements),
    )
    if integrity_errors:
        gate["mechanical_decision"] = gate["decision"]
        gate["decision"] = "reject"
        gate["reasons"] = [
            *list(gate.get("reasons") or []),
            *(f"evidence integrity: {error}" for error in integrity_errors),
        ]
    artifacts = [
        _artifact_summary(workspace_root, record, artifact, role="raw")
        for artifact in record.raw_artifacts
    ]
    artifacts.extend(
        _artifact_summary(workspace_root, record, artifact, role="profile")
        for artifact in record.profile_artifacts
    )
    provenance_fields = (
        "repo_root",
        "git_branch",
        "git_commit",
        "control_commit",
        "candidate_commit",
        "diff_artifact",
        "diff_sha256",
        "diff_bytes",
        "evidence_adapter",
        "evidence_manifest_sha256",
        "record_binding_artifact",
        "record_binding_sha256",
    )
    return {
        "experiment_id": record.experiment_id,
        "parent_id": record.parent_id,
        "beam": record.beam,
        "hypothesis": record.hypothesis,
        "status": record.status,
        "correctness": record.correctness,
        "primary_case": record.primary_case,
        "changed_surface": list(record.changed_surface),
        "mechanism": record.mechanism,
        "outcome": record.outcome,
        "next_step": record.next_step,
        "recorded_at": record.recorded_at,
        "revision": record.revision,
        "is_incumbent": record.experiment_id == incumbent_id,
        "evidence_integrity": {
            "valid": not integrity_errors,
            "errors": integrity_errors,
        },
        "gate": gate,
        "cases": _case_rows(config, record, gate),
        "artifacts": artifacts,
        "provenance": {
            field_name: record.provenance.get(field_name)
            for field_name in provenance_fields
            if record.provenance.get(field_name) is not None
        },
    }


def build_campaign_dashboard(workspace_root: Path) -> dict[str, Any]:
    """Build the compact campaign status used by the dashboard and reviewers."""

    workspace = CampaignWorkspace(Path(workspace_root).expanduser().resolve())
    config = workspace.load_config()
    records = workspace.ledger.latest()
    promoted_records = [record for record in records if record.status == "promoted"]
    incumbent_state = current_incumbent(config, records, workspace_root=workspace.root)
    incumbent_id = incumbent_state["experiment_id"]
    summaries = [
        _record_summary(
            workspace.root,
            config,
            record,
            incumbent_id=incumbent_id,
        )
        for record in records
    ]
    summary_by_id = {row["experiment_id"]: row for row in summaries}
    measured = [row for row in summaries if row["cases"]]
    frontier = [
        summary_by_id[record.experiment_id]
        for record, _decision in promotion_frontier(config, records)
        if summary_by_id[record.experiment_id]["evidence_integrity"]["valid"]
    ]
    beam = [
        summary_by_id[record.experiment_id]
        for record in active_beam(config, records)
        if summary_by_id[record.experiment_id]["evidence_integrity"]["valid"]
    ]
    return {
        "workspace": str(workspace.root),
        "config": config.to_dict(),
        "budget": budget_status(config, records),
        "counts": {
            "experiments": len(records),
            "measured": len(measured),
            "promoted": len(promoted_records),
            "parked": sum(record.status == "parked" for record in records),
            "rejected": sum(record.status == "rejected" for record in records),
            "crashed": sum(record.status == "crashed" for record in records),
        },
        "incumbent": {
            **incumbent_state,
            "experiment": summary_by_id.get(incumbent_id) if incumbent_id else None,
        },
        "latest_measured": measured[-1] if measured else None,
        "active_beam": beam,
        "frontier": frontier,
        "experiments": summaries,
    }
