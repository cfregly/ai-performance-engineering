from __future__ import annotations

import json
from pathlib import Path

from core.verification.verify_mlperf_v6_repos import VendoredRepoCheck, verify_repo


def _dummy_sha(char: str) -> str:
    return char * 40


def _write_repo(
    root: Path,
    *,
    expected_label: str,
    repo: str,
    commit: str,
    requested_ref: str = "master",
) -> VendoredRepoCheck:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(f"# Placeholder\n\n## {expected_label}\n", encoding="utf-8")
    (root / "VENDORED_FROM.json").write_text(
        json.dumps(
            {
                "name": root.name,
                "repo": repo,
                "requested_ref": requested_ref,
                "resolved_commit": commit,
                "committed_at": "2026-04-16T00:00:00Z",
                "expected_label": expected_label,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return VendoredRepoCheck(
        name=root.name,
        path=root,
        expected_label=expected_label,
        expected_ref=requested_ref,
        expected_repo=repo,
    )


def test_verify_repo_accepts_expected_vendor_snapshot(tmp_path: Path) -> None:
    check = _write_repo(
        tmp_path / "mlperf_inference",
        expected_label="MLPerf Inference v6.0",
        repo="https://github.com/mlcommons/inference.git",
        commit=_dummy_sha("a"),
    )

    assert verify_repo(check) == []


def test_verify_repo_reports_label_mismatch(tmp_path: Path) -> None:
    check = _write_repo(
        tmp_path / "mlperf_training",
        expected_label="MLPerf Training v5.1",
        repo="https://github.com/mlcommons/training.git",
        commit=_dummy_sha("b"),
    )
    check.expected_label = "MLPerf Training v6.0"

    issues = verify_repo(check)
    assert issues == [
        "mlperf_training: README.md missing expected label 'MLPerf Training v6.0'",
        "mlperf_training: label mismatch (expected MLPerf Training v6.0, got MLPerf Training v5.1)",
    ]
