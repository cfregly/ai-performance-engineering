#!/usr/bin/env python3
"""Build the immutable Wave 2 inventory from the user-supplied report text."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


AUDIT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = AUDIT_ROOT / "wave-2-source.txt"
OUTPUT = AUDIT_ROOT / "wave-2-source.json"
SEVERITIES = {"critical", "high", "medium", "low"}
VERDICTS = {"fixed", "needs runtime", "partial", "obsolete", "not fixed", "regressed"}
LOCATION = re.compile(r"^(?:code|docs|\.github)/.+:\d+$")
W1_ID = re.compile(r"^W1-\d{3}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    remediation_start = lines.index("Wave-1 remediation ledger — independently verified")

    open_findings: list[dict[str, object]] = []
    for index in range(remediation_start - 2):
        if (
            lines[index + 2] in SEVERITIES
            and LOCATION.fullmatch(lines[index])
        ):
            finding_id = f"W2-{len(open_findings) + 1:03d}"
            residual = re.search(r"Residual of (W1-\d{3})", lines[index + 1])
            open_findings.append(
                {
                    "id": finding_id,
                    "source_record_index": len(open_findings),
                    "source_line": index + 1,
                    "original_severity": lines[index + 2],
                    "original_location": lines[index],
                    "original_title": lines[index + 1],
                    "source_group": "merged_wave_2_closure_critic_tail_and_wave_1_residual_inventory",
                    "related_wave_1": [residual.group(1)] if residual else [],
                }
            )

    remediation_verdicts: list[dict[str, object]] = []
    for index in range(remediation_start + 1, len(lines) - 3):
        if lines[index] in VERDICTS and W1_ID.fullmatch(lines[index + 1]):
            remediation_verdicts.append(
                {
                    "id": lines[index + 1],
                    "verdict": lines[index].replace(" ", "_"),
                    "source_line": index + 1,
                    "location": lines[index + 2],
                    "title": lines[index + 3],
                }
            )

    severity_counts = Counter(row["original_severity"] for row in open_findings)
    verdict_counts = Counter(row["verdict"] for row in remediation_verdicts)
    assert len(open_findings) == 141
    assert severity_counts == {"critical": 3, "high": 31, "medium": 62, "low": 45}
    assert len(remediation_verdicts) == 128
    assert len({row["id"] for row in remediation_verdicts}) == 128
    assert verdict_counts == {
        "fixed": 120,
        "needs_runtime": 5,
        "partial": 2,
        "obsolete": 1,
    }

    output = {
        "schema_version": 1,
        "source_kind": "user_supplied_full_report_text",
        "source_url": "https://claude.ai/code/artifact/9a311b78-5dac-4ea7-909f-e52993ab8e0e",
        "source_capture": "wave-2-source.txt",
        "source_sha256": sha256(SOURCE),
        "report_updated_date": "2026-08-31",
        "reviewed_revision": "cf48c8481",
        "wave_1_revision": "b57e4c6a9",
        "wave_1_fix_revision": "f49aae73f",
        "open_finding_counts": {
            "total": len(open_findings),
            **dict(sorted(severity_counts.items())),
        },
        "open_findings": open_findings,
        "wave_1_remediation_verdict_counts": dict(sorted(verdict_counts.items())),
        "wave_1_remediation_verdicts": remediation_verdicts,
        "source_scope_note": (
            "The report's 141-item open inventory merges Wave 2, the closure critic, "
            "the tail pass, and two explicitly linked Wave 1 residuals. Original order "
            "and wording are preserved; implementation work may deduplicate overlaps."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"open": severity_counts, "verdicts": verdict_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
