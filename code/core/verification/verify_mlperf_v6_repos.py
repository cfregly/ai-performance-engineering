#!/usr/bin/env python3
"""Verify vendored MLPerf v6 source trees under third_party/."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class VendoredRepoCheck:
    name: str
    path: Path
    expected_label: str
    expected_ref: str
    expected_repo: str


def verify_repo(check: VendoredRepoCheck) -> list[str]:
    issues: list[str] = []
    if not check.path.exists():
        return [f"{check.name}: missing directory {check.path}"]

    readme = check.path / "README.md"
    if not readme.exists():
        issues.append(f"{check.name}: missing README.md")
    else:
        content = readme.read_text(encoding="utf-8", errors="replace")
        if check.expected_label not in content:
            issues.append(f"{check.name}: README.md missing expected label '{check.expected_label}'")

    metadata_path = check.path / "VENDORED_FROM.json"
    if not metadata_path.exists():
        issues.append(f"{check.name}: missing VENDORED_FROM.json")
        return issues

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"{check.name}: invalid VENDORED_FROM.json ({exc})")
        return issues

    if payload.get("repo") != check.expected_repo:
        issues.append(
            f"{check.name}: repo mismatch (expected {check.expected_repo}, got {payload.get('repo')})"
        )
    if payload.get("requested_ref") != check.expected_ref:
        issues.append(
            f"{check.name}: ref mismatch (expected {check.expected_ref}, got {payload.get('requested_ref')})"
        )
    resolved_commit = payload.get("resolved_commit")
    if not isinstance(resolved_commit, str) or not FULL_GIT_SHA_RE.fullmatch(resolved_commit):
        issues.append(f"{check.name}: resolved_commit is not a full git SHA")
    if payload.get("expected_label") != check.expected_label:
        issues.append(
            f"{check.name}: label mismatch (expected {check.expected_label}, got {payload.get('expected_label')})"
        )

    return issues


def build_checks(project_root: Path) -> list[VendoredRepoCheck]:
    setup_text = (project_root / "setup.sh").read_text(encoding="utf-8")

    def setup_default(var_name: str) -> str:
        prefix = f'{var_name}="${{{var_name}:-'
        for raw_line in setup_text.splitlines():
            line = raw_line.strip()
            if line.startswith(prefix) and line.endswith('}"'):
                return line[len(prefix):-2]
        raise RuntimeError(f"Could not find default for {var_name} in setup.sh")

    third_party = project_root / "third_party"
    return [
        VendoredRepoCheck(
            name="mlperf_inference",
            path=third_party / "mlperf_inference",
            expected_label="MLPerf Inference v6.0",
            expected_ref=setup_default("MLPERF_INFERENCE_GIT_REF"),
            expected_repo=setup_default("MLPERF_INFERENCE_REPO_URL"),
        ),
        VendoredRepoCheck(
            name="mlperf_training",
            path=third_party / "mlperf_training",
            expected_label="MLPerf Training v6.0",
            expected_ref=setup_default("MLPERF_TRAINING_GIT_REF"),
            expected_repo=setup_default("MLPERF_TRAINING_REPO_URL"),
        ),
    ]


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    issues: list[str] = []
    for check in build_checks(project_root):
        issues.extend(verify_repo(check))

    if issues:
        print("MLPerf v6 vendor verification failed:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("MLPerf v6 vendor verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
