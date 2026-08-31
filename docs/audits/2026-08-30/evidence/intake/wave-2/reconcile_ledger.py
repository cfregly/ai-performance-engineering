#!/usr/bin/env python3
"""Reconcile the immutable Wave 2 inventory into the mutable audit ledger."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re


AUDIT_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_PATH = AUDIT_ROOT / "wave-2-source.json"
LEDGER_PATH = AUDIT_ROOT / "remediation-ledger.json"
RECEIPT_PATH = Path("evidence/intake/wave-2/receipt.json")


PACKAGE_DEFINITIONS = {
    "W2P01": "Early chapters (ch01-ch05)",
    "W2P02": "CUDA and systems chapters (ch06-ch11)",
    "W2P03": "Graphs, compilation, and numerics (ch12-ch14)",
    "W2P04": "Inference and capstone chapters (ch15-ch20)",
    "W2P05": "Labs",
    "W2P06": "MCP, dashboard, and monitoring",
    "W2P07": "Core framework",
    "W2P08": "Scripts, documentation, and residuals",
}


def package_for(location: str) -> str:
    path = location.rsplit(":", 1)[0]
    chapter = re.match(r"code/ch(\d{2})/", path)
    if chapter:
        number = int(chapter.group(1))
        if number <= 5:
            return "W2P01"
        if number <= 11:
            return "W2P02"
        if number <= 14:
            return "W2P03"
        return "W2P04"
    if path.startswith("code/labs/"):
        return "W2P05"
    if path.startswith(("code/mcp/", "code/dashboard/", "code/monitoring/")):
        return "W2P06"
    if path.startswith("code/core/"):
        return "W2P07"
    if path.startswith(("code/scripts/", "docs/", ".github/")):
        return "W2P08"
    raise ValueError(f"No Wave 2 package mapping for {location}")


def main() -> int:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()

    manifest = ledger["source_manifest"]["wave_2"]
    history = manifest.get("artifact_check_history", [])
    last_check = manifest.get("last_artifact_check")
    ledger["source_manifest"]["wave_2"] = {
        "status": "received",
        "url": inventory["source_url"],
        "reviewed_revision": inventory["reviewed_revision"],
        "reported_counts": inventory["open_finding_counts"],
        "source_capture": inventory["source_capture"],
        "parsed_inventory": "wave-2-source.json",
        "source_sha256": inventory["source_sha256"],
        "intake_receipt": str(RECEIPT_PATH),
        "required_for_goal_completion": True,
        "source_scope_note": inventory["source_scope_note"],
        "public_artifact_state": (
            "The public URL still served its older 128-finding Wave 1 frame during "
            "intake; the complete user-supplied attachment is authoritative."
        ),
        "previous_last_artifact_check": last_check,
        "artifact_check_history": history,
    }

    verdict_by_id = {
        row["id"]: row for row in inventory["wave_1_remediation_verdicts"]
    }
    assert set(verdict_by_id) == {row["id"] for row in ledger["findings"]}
    status_by_verdict = {
        "fixed": "source_fixed",
        "needs_runtime": "awaiting_runtime",
        "partial": "in_progress",
        "obsolete": "already_fixed_with_evidence",
    }
    runtime_required = 0
    for finding in ledger["findings"]:
        verdict = verdict_by_id[finding["id"]]
        prior_status = finding.get(
            "pre_external_verdict_status", finding["status"]
        )
        finding["pre_external_verdict_status"] = prior_status
        requires_runtime = (
            prior_status == "awaiting_runtime"
            or verdict["verdict"] == "needs_runtime"
        )
        runtime_required += int(requires_runtime)
        finding["external_remediation_verdict"] = {
            "verdict": verdict["verdict"],
            "source": "wave-2-source.json",
            "source_line": verdict["source_line"],
            "reviewed_revision": inventory["reviewed_revision"],
            "review_kind": "read_only_independent_source_review",
            "runtime_proof": False,
        }
        finding["runtime_acceptance"] = {
            "required_by_local_acceptance_plan": requires_runtime,
            "state": "pending" if requires_runtime else "not_required_by_current_ledger",
            "note": (
                "The external source verdict does not replace applicable build, "
                "GPU, distributed, sanitizer, or performance gates."
            ),
        }
        if "post_external_remediation" not in finding:
            finding["status"] = status_by_verdict[verdict["verdict"]]

    existing_wave_2 = {
        row["id"]: row for row in ledger.get("wave_2_findings", [])
    }
    mutable_wave_2_fields = {
        "triaged_severity",
        "status",
        "owner",
        "source_revalidation",
        "reproducer",
        "accepted_fix_design",
        "changed_files",
        "fix_revision",
        "verification",
        "evidence",
        "independent_review",
        "disposition_reason",
    }
    wave_2_findings = []
    for source_finding in inventory["open_findings"]:
        package = package_for(str(source_finding["original_location"]))
        row = {
            **source_finding,
            "wave": 2,
            "triaged_severity": None,
            "primary_package": package,
            "status": "untriaged",
            "owner": None,
            "source_revalidation": {
                "state": "received_from_user_report",
                "reviewed_revision": inventory["reviewed_revision"],
                "current_revision_at_intake": "cf48c8481df0de847abc1569fd3be3f33218f351",
            },
            "reproducer": None,
            "accepted_fix_design": None,
            "changed_files": [],
            "fix_revision": None,
            "verification": {
                "source_and_cpu": "pending",
                "build": "pending",
                "gpu_correctness": "pending_if_applicable",
                "performance": "pending_if_applicable",
                "documentation": "pending_if_applicable",
            },
            "evidence": ["wave-2-source.json", str(RECEIPT_PATH)],
            "independent_review": None,
            "disposition_reason": None,
        }
        prior = existing_wave_2.get(row["id"], {})
        for field in mutable_wave_2_fields:
            if field in prior:
                row[field] = prior[field]
        wave_2_findings.append(row)
    assert len(wave_2_findings) == 141
    assert len({row["id"] for row in wave_2_findings}) == 141
    ledger["wave_2_findings"] = wave_2_findings

    package_members: dict[str, list[str]] = {
        package_id: [] for package_id in PACKAGE_DEFINITIONS
    }
    for finding in wave_2_findings:
        package_members[finding["primary_package"]].append(finding["id"])
    ledger["wave_2_packages"] = [
        {
            "id": package_id,
            "title": title,
            "finding_ids": package_members[package_id],
            "status": "in_progress",
            "acceptance_document": "../../../AUDIT_REMEDIATION_PLAN.md",
        }
        for package_id, title in PACKAGE_DEFINITIONS.items()
    ]
    assert sum(len(row["finding_ids"]) for row in ledger["wave_2_packages"]) == 141

    external_counts = Counter(
        row["external_remediation_verdict"]["verdict"]
        for row in ledger["findings"]
    )
    ledger["wave_2_intake_checkpoint"] = {
        "checkpoint_type": "WAVE_2_USER_ATTACHMENT_INTAKE",
        "status": "RECEIVED_AND_PARSED__REMEDIATION_IN_PROGRESS",
        "at": now,
        "reviewed_revision": inventory["reviewed_revision"],
        "open_finding_counts": inventory["open_finding_counts"],
        "unique_files": len(
            {
                row["original_location"].rsplit(":", 1)[0]
                for row in wave_2_findings
            }
        ),
        "wave_1_external_verdict_counts": dict(sorted(external_counts.items())),
        "wave_1_local_runtime_acceptance_required": runtime_required,
        "source": inventory["source_capture"],
        "parsed_inventory": "wave-2-source.json",
        "receipt": str(RECEIPT_PATH),
        "public_artifact_mismatch": True,
        "goal_complete": False,
    }
    ledger["wave_2_intake_note"] = (
        "The former wave_2_coverage_warning field is the Wave 1 report's historical "
        "unreviewed-area footer. The supplied Wave 2 report now covers that follow-up "
        "and adds a merged 141-item open inventory."
    )
    ledger["updated_at"] = now

    LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "wave_2_findings": len(wave_2_findings),
                "wave_2_packages": {
                    row["id"]: len(row["finding_ids"])
                    for row in ledger["wave_2_packages"]
                },
                "wave_1_external_verdicts": external_counts,
                "wave_1_runtime_acceptance_required": runtime_required,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
