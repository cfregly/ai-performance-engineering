from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.analysis.regressions import compare_suite_summaries
from core.scripts.benchmarks.merge_tier1_history import merge_tier1_history_evidence
from core.scripts.benchmarks.promote_tier1_history import promote_tier1_history_anchor

RUN_ID = "tier1_candidate"
COMMIT = "a" * 40
EVIDENCE_NAME = "tier1-evidence-123-1"
EVIDENCE_DIGEST = f"sha256:{'b' * 64}"


def _write_candidate(
    root: Path,
    *,
    with_prior_anchor: bool = False,
    suite_version: int = 1,
    suppressed_regressions: list[dict[str, object]] | None = None,
    summary_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    history_root = root / "history"
    evidence_root = root / "evidence"
    run_dir = history_root / RUN_ID
    evidence_run_dir = evidence_root / RUN_ID
    run_dir.mkdir(parents=True)
    (evidence_run_dir / "results").mkdir(parents=True)

    suite_config = root / "tier1.yaml"
    suite_config.write_text(
        "\n".join(
            [
                "suite_name: tier1",
                f"version: {suite_version}",
                "targets:",
                "  - key: demo",
                "    target: ch01:demo",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "run_id": RUN_ID,
        "suite_name": "tier1",
        "suite_version": suite_version,
        "generated_at": "2026-08-16T00:00:00Z",
        "source_git_commit": COMMIT,
        "source_manifest_git_commit": COMMIT,
        "source_git_dirty": False,
        "source_manifest_json": f"{RUN_ID}/manifest.json",
        "source_result_json": f"{RUN_ID}/results/benchmark_test_results.json",
        "source_markdown_report": f"{RUN_ID}/report.md",
        "evidence_artifact_name": EVIDENCE_NAME,
        "targets": [
            {
                "key": "demo",
                "target": "ch01:demo",
                "status": "succeeded",
                "optimization_goal": "performance",
                "baseline_time_ms": 2.0,
                "best_speedup": 2.0,
                "best_optimized_time_ms": 1.0,
                "artifacts": {},
            }
        ],
        "summary": {
            "target_count": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "missing": 0,
            "avg_speedup": 2.0,
            "median_speedup": 2.0,
            "geomean_speedup": 2.0,
            "representative_speedup": 2.0,
            "max_speedup": 2.0,
        },
    }
    if summary_overrides:
        summary.update(summary_overrides)
    candidate_entry = {
        "run_id": RUN_ID,
        "generated_at": summary["generated_at"],
        "summary_path": f"{RUN_ID}/summary.json",
        "regression_summary_path": f"{RUN_ID}/regression_summary.md",
        "regression_json_path": f"{RUN_ID}/regression_summary.json",
        "trend_snapshot_path": f"{RUN_ID}/trend_snapshot.json",
        "avg_speedup": 2.0,
        "median_speedup": 2.0,
        "geomean_speedup": 2.0,
        "representative_speedup": 2.0,
        "max_speedup": 2.0,
        "succeeded": 1,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "run_accepted": False,
        "baseline_eligible": False,
    }
    runs: list[dict[str, object]] = []
    prior_summary: dict[str, object] | None = None
    if with_prior_anchor:
        prior_dir = history_root / "tier1_anchor"
        prior_dir.mkdir()
        prior_targets = [dict(target) for target in summary["targets"]]
        prior_targets[0]["best_speedup"] = 2.2
        prior_targets[0]["best_optimized_time_ms"] = 2.0 / 2.2
        prior_summary = {
            **summary,
            "run_id": "tier1_anchor",
            "targets": prior_targets,
            "summary": {
                **summary["summary"],
                "avg_speedup": 2.2,
                "median_speedup": 2.2,
                "geomean_speedup": 2.2,
                "representative_speedup": 2.2,
                "max_speedup": 2.2,
            },
        }
        (prior_dir / "summary.json").write_text(json.dumps(prior_summary), encoding="utf-8")
        runs.append(
            {
                "run_id": "tier1_anchor",
                "summary_path": "tier1_anchor/summary.json",
                "run_accepted": True,
                "baseline_eligible": True,
                "baseline_acceptance": "clean",
            }
        )
    comparison = compare_suite_summaries(summary, prior_summary)
    comparison.update(
        {
            "suppressed_regressions": suppressed_regressions or [],
            "rechecks": [],
            "warnings": [],
        }
    )
    runs.append(candidate_entry)
    index = {
        "suite_name": "tier1",
        "suite_version": suite_version,
        "history_root": ".",
        "runs": runs,
    }
    (history_root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "regression_summary.md").write_text("# Review\n", encoding="utf-8")
    (run_dir / "regression_summary.json").write_text(json.dumps(comparison), encoding="utf-8")
    (run_dir / "trend_snapshot.json").write_text(
        json.dumps({"run_count": int(with_prior_anchor)}), encoding="utf-8"
    )
    (evidence_run_dir / "manifest.json").write_text(
        json.dumps({"run_id": RUN_ID, "git": {"commit": COMMIT, "dirty": False}}),
        encoding="utf-8",
    )
    (evidence_run_dir / "results" / "benchmark_test_results.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "timestamp": summary["generated_at"],
                "results": [
                    {
                        "chapter": "ch01",
                        "benchmarks": [
                            {
                                "example": "demo",
                                "status": "succeeded",
                                "optimization_goal": "performance",
                                "baseline_time_ms": 2.0,
                                "best_speedup": 2.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (evidence_run_dir / "report.md").write_text("# Evidence\n", encoding="utf-8")
    return suite_config, evidence_root


def _write_live_anchor(
    live_root: Path,
    candidate_root: Path,
    *,
    suite_name: str = "tier1",
    suite_version: int = 1,
) -> None:
    run_id = f"tier1_v{suite_version}_anchor"
    run_dir = live_root / run_id
    run_dir.mkdir(parents=True)
    summary = json.loads((candidate_root / RUN_ID / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "run_id": run_id,
            "suite_name": suite_name,
            "suite_version": suite_version,
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    index = {
        "suite_name": suite_name,
        "suite_version": suite_version,
        "history_root": ".",
        "runs": [
            {
                "run_id": run_id,
                "summary_path": f"{run_id}/summary.json",
                "run_accepted": True,
                "baseline_eligible": True,
                "baseline_acceptance": "clean",
            }
        ],
    }
    (live_root / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _promote(root: Path, suite_config: Path, evidence_root: Path, **kwargs):
    options = {
        "history_root": root / "history",
        "evidence_root": evidence_root,
        "suite_config": suite_config,
        "run_id": RUN_ID,
        "requester": "dispatcher",
        "note": "Reviewed the immutable evidence package",
        "workflow_run": "https://github.example/owner/repo/actions/runs/123",
        "expected_git_commit": COMMIT,
        "expected_evidence_artifact": EVIDENCE_NAME,
        "expected_evidence_digest": EVIDENCE_DIGEST,
    }
    options.update(kwargs)
    return promote_tier1_history_anchor(**options)


@pytest.mark.parametrize("with_prior_anchor", [False, True])
def test_promote_tier1_history_ratifies_exact_candidate(
    tmp_path: Path,
    with_prior_anchor: bool,
) -> None:
    suite_config, evidence_root = _write_candidate(
        tmp_path,
        with_prior_anchor=with_prior_anchor,
    )

    result = _promote(
        tmp_path,
        suite_config,
        evidence_root,
        allow_bootstrap=not with_prior_anchor,
    )

    assert result["success"] is True
    index = json.loads((tmp_path / "history" / "index.json").read_text(encoding="utf-8"))
    entry = index["runs"][-1]
    assert entry["run_accepted"] is True
    assert entry["baseline_eligible"] is True
    assert entry["baseline_acceptance"] == "accept_history_anchor"
    assert entry["baseline_acceptance_actor"] == "dispatcher"
    assert entry["baseline_acceptance_actor_role"] == "requester"
    assert entry["baseline_evidence_digest"] == EVIDENCE_DIGEST
    trend = json.loads(
        (tmp_path / "history" / RUN_ID / "trend_snapshot.json").read_text(encoding="utf-8")
    )
    assert trend["latest_run_id"] == RUN_ID


def test_promote_tier1_history_normalizes_raw_upload_artifact_digest(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)

    result = _promote(
        tmp_path,
        suite_config,
        evidence_root,
        expected_evidence_digest="b" * 64,
        allow_bootstrap=True,
    )

    assert result["success"] is True
    index = json.loads((tmp_path / "history" / "index.json").read_text(encoding="utf-8"))
    assert index["runs"][-1]["baseline_evidence_digest"] == EVIDENCE_DIGEST


def test_confirmed_regression_requires_protected_explicit_promotion(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    candidate_root = tmp_path / "history"
    candidate_index = json.loads((candidate_root / "index.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (candidate_root / RUN_ID / "regression_summary.json").read_text(encoding="utf-8")
    )
    assert comparison["regressions"]
    assert comparison["suppressed_regressions"] == []

    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    shutil.copytree(candidate_root / "tier1_anchor", canonical_root / "tier1_anchor")
    (canonical_root / "index.json").write_text(
        json.dumps({**candidate_index, "runs": candidate_index["runs"][:-1]}),
        encoding="utf-8",
    )

    normal_output = tmp_path / "normal-publication"
    merge_tier1_history_evidence(
        candidate_history_root=candidate_root,
        canonical_history_root=canonical_root,
        output_history_root=normal_output,
        run_id=RUN_ID,
    )
    normal_index = json.loads((normal_output / "index.json").read_text(encoding="utf-8"))
    normal_entry = normal_index["runs"][-1]
    assert normal_entry["run_accepted"] is False
    assert normal_entry["baseline_eligible"] is False
    assert "baseline_acceptance" not in normal_entry

    override_reason = "Accept the confirmed latency regression after evidence review"
    protected_output = tmp_path / "protected-promotion"
    result = _promote(
        tmp_path,
        suite_config,
        evidence_root,
        canonical_history_root=canonical_root,
        output_history_root=protected_output,
        note=override_reason,
    )

    assert result["accepted_regression_count"] == len(comparison["regressions"])
    protected_index = json.loads((protected_output / "index.json").read_text(encoding="utf-8"))
    protected_entry = protected_index["runs"][-1]
    assert protected_entry["run_accepted"] is True
    assert protected_entry["baseline_eligible"] is True
    assert protected_entry["baseline_acceptance"] == "accept_history_anchor"
    assert protected_entry["baseline_acceptance_actor"] == "dispatcher"
    assert protected_entry["baseline_acceptance_note"] == override_reason
    assert protected_entry["baseline_acceptance_workflow_run"] == (
        "https://github.example/owner/repo/actions/runs/123"
    )


@pytest.mark.parametrize(
    "workflow_run",
    [
        "not-a-url",
        "http://github.example/owner/repo/actions/runs/123",
        "https://github.example/owner/repo/actions/runs/not-a-number",
        "https://github.example/owner/repo/actions/runs/123?token=secret",
    ],
)
def test_promote_tier1_history_requires_exact_workflow_run_url(
    tmp_path: Path,
    workflow_run: str,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match="exact HTTPS GitHub Actions workflow run URL"):
        _promote(
            tmp_path,
            suite_config,
            evidence_root,
            workflow_run=workflow_run,
            allow_bootstrap=True,
        )

    assert _tree_bytes(tmp_path / "history") == before


@pytest.mark.parametrize(
    ("summary_overrides", "suppressed_regressions", "match"),
    [
        ({}, [{"target": "ch01:demo"}], "cleared only by a recheck"),
        ({"source_git_dirty": True}, [], "clean manifest-bound Git provenance"),
        ({"source_manifest_git_commit": "b" * 40}, [], "clean manifest-bound Git provenance"),
        (
            {"targets": [], "summary": {"target_count": 1, "succeeded": 1}},
            [],
            "target set does not match",
        ),
    ],
)
def test_promote_tier1_history_rejects_untrusted_candidate_without_mutation(
    tmp_path: Path,
    summary_overrides: dict[str, object],
    suppressed_regressions: list[dict[str, object]],
    match: str,
) -> None:
    suite_config, evidence_root = _write_candidate(
        tmp_path,
        summary_overrides=summary_overrides,
        suppressed_regressions=suppressed_regressions,
    )
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match=match):
        _promote(
            tmp_path,
            suite_config,
            evidence_root,
            allow_bootstrap=True,
        )

    assert _tree_bytes(tmp_path / "history") == before


def test_promote_tier1_history_rejects_missing_evidence_file(tmp_path: Path) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    (evidence_root / RUN_ID / "report.md").unlink()
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match="evidence artifact is incomplete"):
        _promote(tmp_path, suite_config, evidence_root, allow_bootstrap=True)

    assert _tree_bytes(tmp_path / "history") == before


def test_promote_tier1_history_rejects_wrong_prior_anchor_binding(tmp_path: Path) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    regression_path = tmp_path / "history" / RUN_ID / "regression_summary.json"
    comparison = json.loads(regression_path.read_text(encoding="utf-8"))
    comparison["baseline_run_id"] = "not_the_anchor"
    regression_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="prior canonical anchor"):
        _promote(tmp_path, suite_config, evidence_root)


def test_promote_tier1_history_rejects_tampered_comparison_without_mutation(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    regression_path = tmp_path / "history" / RUN_ID / "regression_summary.json"
    comparison = json.loads(regression_path.read_text(encoding="utf-8"))
    comparison["regressions"] = []
    regression_path.write_text(json.dumps(comparison), encoding="utf-8")
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match="does not match the canonical summaries"):
        _promote(tmp_path, suite_config, evidence_root)

    assert _tree_bytes(tmp_path / "history") == before


def test_promote_tier1_history_merges_live_evidence_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    candidate_root = tmp_path / "history"
    live_root = tmp_path / "live"
    live_root.mkdir()
    shutil.copytree(candidate_root / "tier1_anchor", live_root / "tier1_anchor")
    candidate_index = json.loads((candidate_root / "index.json").read_text(encoding="utf-8"))
    evidence_run_id = "later_evidence"
    evidence_dir = live_root / evidence_run_id
    evidence_dir.mkdir()
    evidence_summary = json.loads(
        (candidate_root / "tier1_anchor" / "summary.json").read_text(encoding="utf-8")
    )
    evidence_summary["run_id"] = evidence_run_id
    (evidence_dir / "summary.json").write_text(json.dumps(evidence_summary), encoding="utf-8")
    live_index = {
        **candidate_index,
        "runs": [
            candidate_index["runs"][0],
            {
                "run_id": evidence_run_id,
                "summary_path": f"{evidence_run_id}/summary.json",
                "run_accepted": True,
                "baseline_eligible": False,
            },
        ],
    }
    (live_root / "index.json").write_text(json.dumps(live_index), encoding="utf-8")
    candidate_before = _tree_bytes(candidate_root)
    live_before = _tree_bytes(live_root)
    output_root = tmp_path / "ratified"

    result = _promote(
        tmp_path,
        suite_config,
        evidence_root,
        canonical_history_root=live_root,
        output_history_root=output_root,
    )

    assert result["success"] is True
    published_index = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert [entry["run_id"] for entry in published_index["runs"]] == [
        "tier1_anchor",
        evidence_run_id,
        RUN_ID,
    ]
    assert published_index["runs"][-1]["baseline_eligible"] is True
    assert _tree_bytes(candidate_root) == candidate_before
    assert _tree_bytes(live_root) == live_before


def test_promote_tier1_history_bootstraps_new_suite_version_over_valid_prior_version(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, suite_version=2)
    candidate_root = tmp_path / "history"
    live_root = tmp_path / "live"
    _write_live_anchor(live_root, candidate_root, suite_version=1)
    candidate_before = _tree_bytes(candidate_root)
    live_before = _tree_bytes(live_root)
    output_root = tmp_path / "ratified"

    result = _promote(
        tmp_path,
        suite_config,
        evidence_root,
        canonical_history_root=live_root,
        output_history_root=output_root,
        allow_bootstrap=True,
    )

    assert result["success"] is True
    published_index = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert published_index["suite_name"] == "tier1"
    assert published_index["suite_version"] == 2
    assert [entry["run_id"] for entry in published_index["runs"]] == [RUN_ID]
    assert not (output_root / "tier1_v1_anchor").exists()
    assert _tree_bytes(candidate_root) == candidate_before
    assert _tree_bytes(live_root) == live_before


def test_promote_tier1_history_rejects_live_version_mismatch_without_bootstrap(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, suite_version=2)
    candidate_root = tmp_path / "history"
    live_root = tmp_path / "live"
    _write_live_anchor(live_root, candidate_root, suite_version=1)
    candidate_before = _tree_bytes(candidate_root)
    live_before = _tree_bytes(live_root)
    output_root = tmp_path / "ratified"

    with pytest.raises(ValueError, match="does not match the checked-out suite config"):
        _promote(
            tmp_path,
            suite_config,
            evidence_root,
            canonical_history_root=live_root,
            output_history_root=output_root,
        )

    assert not output_root.exists()
    assert _tree_bytes(candidate_root) == candidate_before
    assert _tree_bytes(live_root) == live_before


def test_promote_tier1_history_rejects_live_suite_name_mismatch_during_bootstrap(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, suite_version=2)
    candidate_root = tmp_path / "history"
    live_root = tmp_path / "live"
    _write_live_anchor(
        live_root,
        candidate_root,
        suite_name="other_suite",
        suite_version=2,
    )
    output_root = tmp_path / "ratified"

    with pytest.raises(ValueError, match="does not match the checked-out suite config"):
        _promote(
            tmp_path,
            suite_config,
            evidence_root,
            canonical_history_root=live_root,
            output_history_root=output_root,
            allow_bootstrap=True,
        )

    assert not output_root.exists()


def test_promote_tier1_history_rejects_stale_live_anchor(tmp_path: Path) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    candidate_root = tmp_path / "history"
    live_root = tmp_path / "live"
    shutil.copytree(candidate_root, live_root)
    live_index_path = live_root / "index.json"
    live_index = json.loads(live_index_path.read_text(encoding="utf-8"))
    live_index["runs"] = live_index["runs"][:-1]
    newer_run_id = "newer_anchor"
    shutil.copytree(live_root / "tier1_anchor", live_root / newer_run_id)
    newer_summary_path = live_root / newer_run_id / "summary.json"
    newer_summary = json.loads(newer_summary_path.read_text(encoding="utf-8"))
    newer_summary["run_id"] = newer_run_id
    newer_summary_path.write_text(json.dumps(newer_summary), encoding="utf-8")
    live_index["runs"].append(
        {
            "run_id": newer_run_id,
            "summary_path": f"{newer_run_id}/summary.json",
            "run_accepted": True,
            "baseline_eligible": True,
            "baseline_acceptance": "clean",
        }
    )
    live_index_path.write_text(json.dumps(live_index), encoding="utf-8")

    with pytest.raises(ValueError, match="prior canonical anchor"):
        _promote(
            tmp_path,
            suite_config,
            evidence_root,
            canonical_history_root=live_root,
            output_history_root=tmp_path / "ratified",
        )


def test_merge_tier1_history_evidence_preserves_live_anchor(tmp_path: Path) -> None:
    _write_candidate(tmp_path, with_prior_anchor=True)
    candidate_root = tmp_path / "history"
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    shutil.copytree(candidate_root / "tier1_anchor", canonical_root / "tier1_anchor")
    candidate_index = json.loads((candidate_root / "index.json").read_text(encoding="utf-8"))
    canonical_index = {**candidate_index, "runs": candidate_index["runs"][:-1]}
    (canonical_root / "index.json").write_text(json.dumps(canonical_index), encoding="utf-8")
    output_root = tmp_path / "published"

    result = merge_tier1_history_evidence(
        candidate_history_root=candidate_root,
        canonical_history_root=canonical_root,
        output_history_root=output_root,
        run_id=RUN_ID,
    )

    assert result == {
        "success": True,
        "run_id": RUN_ID,
        "stale_baseline": False,
        "history_root": ".",
    }
    published = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert [entry["run_id"] for entry in published["runs"]] == [
        "tier1_anchor",
        RUN_ID,
    ]
    assert published["runs"][0]["baseline_eligible"] is True
    assert published["runs"][1]["baseline_eligible"] is False


def test_merge_tier1_history_evidence_rejects_suite_version_mismatch(
    tmp_path: Path,
) -> None:
    _write_candidate(tmp_path, suite_version=2)
    candidate_root = tmp_path / "history"
    canonical_root = tmp_path / "canonical"
    _write_live_anchor(canonical_root, candidate_root, suite_version=1)
    candidate_before = _tree_bytes(candidate_root)
    canonical_before = _tree_bytes(canonical_root)
    output_root = tmp_path / "published"

    with pytest.raises(ValueError, match="suite identities do not match"):
        merge_tier1_history_evidence(
            candidate_history_root=candidate_root,
            canonical_history_root=canonical_root,
            output_history_root=output_root,
            run_id=RUN_ID,
        )

    assert not output_root.exists()
    assert _tree_bytes(candidate_root) == candidate_before
    assert _tree_bytes(canonical_root) == canonical_before


def test_merge_keeps_newer_promotion_latest_when_older_producer_finishes(
    tmp_path: Path,
) -> None:
    _write_candidate(tmp_path, with_prior_anchor=True)
    candidate_root = tmp_path / "history"
    candidate_index_path = candidate_root / "index.json"
    candidate_index = json.loads(candidate_index_path.read_text(encoding="utf-8"))
    candidate_index["runs"][-1]["run_accepted"] = True
    candidate_index_path.write_text(json.dumps(candidate_index), encoding="utf-8")

    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    shutil.copytree(candidate_root / "tier1_anchor", canonical_root / "tier1_anchor")
    newer_run_id = "newer_anchor"
    shutil.copytree(candidate_root / "tier1_anchor", canonical_root / newer_run_id)
    newer_summary_path = canonical_root / newer_run_id / "summary.json"
    newer_summary = json.loads(newer_summary_path.read_text(encoding="utf-8"))
    newer_summary["run_id"] = newer_run_id
    newer_summary["generated_at"] = "2026-08-17T00:00:00Z"
    newer_summary_path.write_text(json.dumps(newer_summary), encoding="utf-8")
    (canonical_root / newer_run_id / "trend_snapshot.json").write_text(
        json.dumps({"latest_evidence_run_id": newer_run_id}),
        encoding="utf-8",
    )
    canonical_index = {
        **candidate_index,
        "runs": [
            candidate_index["runs"][0],
            {
                "run_id": newer_run_id,
                "generated_at": newer_summary["generated_at"],
                "summary_path": f"{newer_run_id}/summary.json",
                "trend_snapshot_path": f"{newer_run_id}/trend_snapshot.json",
                "run_accepted": True,
                "baseline_eligible": True,
                "baseline_acceptance": "clean",
            },
        ],
    }
    (canonical_root / "index.json").write_text(json.dumps(canonical_index), encoding="utf-8")
    output_root = tmp_path / "published"

    result = merge_tier1_history_evidence(
        candidate_history_root=candidate_root,
        canonical_history_root=canonical_root,
        output_history_root=output_root,
        run_id=RUN_ID,
    )

    assert result["stale_baseline"] is True
    published = json.loads((output_root / "index.json").read_text(encoding="utf-8"))
    assert [entry["run_id"] for entry in published["runs"]] == [
        "tier1_anchor",
        RUN_ID,
        newer_run_id,
    ]
    merged_entry = next(entry for entry in published["runs"] if entry["run_id"] == RUN_ID)
    assert merged_entry["run_accepted"] is False
    assert merged_entry["baseline_eligible"] is False
    trend = json.loads(
        (output_root / newer_run_id / "trend_snapshot.json").read_text(encoding="utf-8")
    )
    assert trend["latest_run_id"] == newer_run_id
    assert trend["latest_evidence_run_id"] == newer_run_id
    assert [row["run_id"] for row in trend["evidence_history"]] == [
        "tier1_anchor",
        RUN_ID,
        newer_run_id,
    ]


def test_promote_tier1_history_rejects_nonlatest_run(tmp_path: Path) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    index_path = tmp_path / "history" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["runs"].append({"run_id": "newer_candidate"})
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="newest immutable candidate"):
        _promote(tmp_path, suite_config, evidence_root, allow_bootstrap=True)


@pytest.mark.parametrize("payload_name", ["manifest.json", "results/benchmark_test_results.json"])
def test_promote_tier1_history_rejects_evidence_run_identity_mismatch(
    tmp_path: Path,
    payload_name: str,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    path = evidence_root / RUN_ID / payload_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "different_run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match="run id"):
        _promote(tmp_path, suite_config, evidence_root, allow_bootstrap=True)

    assert _tree_bytes(tmp_path / "history") == before


def test_promote_tier1_history_rejects_primary_reference_outside_candidate_run(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    other_dir = evidence_root / "other_run"
    other_dir.mkdir()
    (other_dir / "manifest.json").write_text(
        json.dumps({"run_id": RUN_ID, "git": {"commit": COMMIT, "dirty": False}}),
        encoding="utf-8",
    )
    summary_path = tmp_path / "history" / RUN_ID / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_manifest_json"] = "other_run/manifest.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="not bound to its run id"):
        _promote(tmp_path, suite_config, evidence_root, allow_bootstrap=True)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["results"][0]["benchmarks"][0].update(status="failed"),
            "immutable result",
        ),
        (
            lambda payload: payload["results"][0]["benchmarks"][0].update(best_speedup=3.0),
            "immutable result",
        ),
        (
            lambda payload: payload["results"][0]["benchmarks"][0].update(best_speedup=True),
            "invalid best_speedup",
        ),
    ],
)
def test_promote_tier1_history_rejects_summary_result_mismatch_without_mutation(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path)
    result_path = evidence_root / RUN_ID / "results" / "benchmark_test_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutator(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    before = _tree_bytes(tmp_path / "history")

    with pytest.raises(ValueError, match=match):
        _promote(tmp_path, suite_config, evidence_root, allow_bootstrap=True)

    assert _tree_bytes(tmp_path / "history") == before


def test_promote_tier1_history_uses_newest_same_suite_prior_anchor(
    tmp_path: Path,
) -> None:
    suite_config, evidence_root = _write_candidate(tmp_path, with_prior_anchor=True)
    history_root = tmp_path / "history"
    index_path = history_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    foreign_run_id = "foreign_anchor"
    foreign_dir = history_root / foreign_run_id
    foreign_dir.mkdir()
    foreign_summary = json.loads(
        (history_root / "tier1_anchor" / "summary.json").read_text(encoding="utf-8")
    )
    foreign_summary.update({"run_id": foreign_run_id, "suite_name": "other_suite"})
    (foreign_dir / "summary.json").write_text(json.dumps(foreign_summary), encoding="utf-8")
    index["runs"].insert(
        -1,
        {
            "run_id": foreign_run_id,
            "summary_path": f"{foreign_run_id}/summary.json",
            "baseline_eligible": True,
        },
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = _promote(tmp_path, suite_config, evidence_root)

    assert result["success"] is True
