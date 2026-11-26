from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_promotable_run(cluster_root: Path, run_id: str, label: str) -> Path:
    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    raw = run_dir / "raw" / f"{run_id}_suite"
    figures = run_dir / "figures"
    structured.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", {"run_id": run_id})
    _write_json(
        structured / f"{run_id}_suite_steps.json",
        [
            {"name": "preflight_services", "exit_code": 0, "start_time": "2026-03-05T21:08:24+00:00"},
            {"name": "discovery", "exit_code": 0, "start_time": "2026-03-05T21:08:25+00:00"},
            {"name": "hang_triage_bundle", "exit_code": 0, "start_time": "2026-03-05T21:08:28+00:00"},
            {"name": "connectivity_probe", "exit_code": 0, "start_time": "2026-03-05T21:08:29+00:00"},
            {"name": "nccl_env_sensitivity", "exit_code": 0, "start_time": "2026-03-05T21:08:47+00:00"},
            {"name": "vllm_serve_sweep", "exit_code": 0, "start_time": "2026-03-05T21:09:53+00:00"},
            {"name": "validate_required_artifacts", "exit_code": 0, "start_time": "2026-03-05T21:17:25+00:00"},
            {"name": "manifest_refresh", "exit_code": 0, "start_time": "2026-03-05T21:17:26+00:00"},
        ],
    )
    _write_json(structured / f"{run_id}_{label}_meta.json", {"commands": {"nvidia_smi_l": {"stdout": "GPU 0: Test GPU"}}})
    _write_json(structured / f"{run_id}_{label}_hang_triage_readiness.json", {"status": "ok"})
    _write_json(
        structured / f"{run_id}_torchrun_connectivity_probe.json",
        {"status": "ok", "world_size": 1, "ranks": [{"barrier_ms": [0.1], "payload_probe": {"algbw_gbps": 111.0}}]},
    )
    _write_json(structured / f"{run_id}_nccl_env_sensitivity.json", {"status": "ok", "failure_count": 0})
    _write_json(structured / f"{run_id}_node1_nccl.json", {"results": [{"algbw_gbps": 2222.0, "size_bytes": 67108864}]})
    _write_json(structured / f"{run_id}_preflight_services.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_node_parity_summary.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_fio.json", {"status": "ok"})
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text(
        "concurrency,total_token_throughput,mean_ttft_ms,p99_ttft_ms,p99_tpot_ms\n1,100.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    (structured / f"{run_id}_{label}_vllm_serve_sweep.jsonl").write_text("{}\n", encoding="utf-8")
    (raw / "suite.log").write_text("ok\n", encoding="utf-8")
    (figures / f"{run_id}_cluster_story_dashboard.png").write_text("png\n", encoding="utf-8")
    return run_dir


def _write_v2_manifest(run_dir: Path, run_id: str) -> dict:
    files = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "manifest_version": 2,
        "run_id": run_id,
        "finalized": True,
        "success": True,
        "files": files,
        "summary": {
            "file_count": len(files),
            "sha256": {path: hashlib.sha256((run_dir / path).read_bytes()).hexdigest() for path in files},
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize(
    "candidate",
    [
        "missing_manifest", "wrong_run_id", "missing_steps", "failed_step", "changed_artifact",
        "missing_inventory", "missing_hashes", "untracked_file", "failed_git_probe",
        "unsupported_version", "invalid_report",
    ],
)
def test_failed_promotion_preserves_publication_and_run_history(
    tmp_path: Path, candidate: str
) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-06_localhost_candidate"
    prior_id = "2026-03-05_localhost_published"
    prior_run = _seed_promotable_run(cluster_root, prior_id, "localhost")
    current = cluster_root / "published" / "current"
    _write_json(current / "manifest.json", {"run_id": prior_id})
    _write_json(current / "structured" / "marker.json", {"previous": True})
    report = cluster_root / "field-report.md"
    report.write_text("prior report\n", encoding="utf-8")
    run_dir = _seed_promotable_run(cluster_root, run_id, "localhost")
    if candidate == "missing_manifest":
        (run_dir / "manifest.json").unlink()
    elif candidate == "wrong_run_id":
        _write_json(run_dir / "manifest.json", {"run_id": prior_id})
    elif candidate == "missing_steps":
        (run_dir / "structured" / f"{run_id}_suite_steps.json").unlink()
    elif candidate == "failed_step":
        _write_json(run_dir / "structured" / f"{run_id}_suite_steps.json", [
            {"name": "discovery", "exit_code": 0},
            {"name": "discovery", "exit_code": 1},
        ])
    elif candidate == "changed_artifact":
        steps_path = f"structured/{run_id}_suite_steps.json"
        original_hash = hashlib.sha256((run_dir / steps_path).read_bytes()).hexdigest()
        _write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "files": [steps_path],
            "summary": {"sha256": {steps_path: original_hash}},
        })
        with (run_dir / steps_path).open("a", encoding="utf-8") as changed:
            changed.write("\n")
    elif candidate in {"missing_inventory", "missing_hashes", "untracked_file", "failed_git_probe", "unsupported_version"}:
        manifest = _write_v2_manifest(run_dir, run_id)
        if candidate == "missing_inventory":
            del manifest["files"]
        elif candidate == "missing_hashes":
            del manifest["summary"]["sha256"]
        elif candidate == "failed_git_probe":
            manifest["git"] = {"probe_ok": False}
        elif candidate == "unsupported_version":
            manifest["manifest_version"] = 3
        else:
            (run_dir / "raw" / "extra.log").write_text("unbound evidence\n", encoding="utf-8")
        _write_json(run_dir / "manifest.json", manifest)

    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"),
        "--repo-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--cleanup",
    ]
    if candidate != "invalid_report":
        command.extend(["--skip-render-localhost-report", "--skip-validate-localhost-report"])
    proc = subprocess.run(command, check=False, capture_output=True, text=True)

    assert json.loads((current / "manifest.json").read_text())["run_id"] == prior_id
    assert (current / "structured" / "marker.json").is_file()
    assert report.read_text(encoding="utf-8") == "prior report\n"
    assert (prior_run / "manifest.json").is_file()
    assert proc.returncode != 0


def test_report_install_failure_restores_previous_publication(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    current = cluster_root / "published" / "current"
    _write_json(current / "manifest.json", {"run_id": "prior"})
    report = cluster_root / "field-report-localhost.md"
    report.write_text("previous report\n", encoding="utf-8")
    (cluster_root / "unavailable").write_text("ordinary file\n", encoding="utf-8")
    run_id = "2026-03-06_localhost_candidate"
    _seed_promotable_run(cluster_root, run_id, "localhost")
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"

    proc = subprocess.run(
        [
            sys.executable, str(script), "--repo-root", str(tmp_path), "--run-id", run_id,
            "--publish-notes-path", "cluster/unavailable/notes.md", "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["steps"]["render_localhost_report"]["success"]
    assert payload["published"] is False
    assert json.loads((current / "manifest.json").read_text()) == {"run_id": "prior"}
    assert report.read_text(encoding="utf-8") == "previous report\n"
    assert not list(current.parent.glob(".promotion-*"))


def test_next_promotion_recovers_interrupted_install(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    current = cluster_root / "published" / "current"
    _write_json(current / "manifest.json", {"run_id": "prior"})
    report = cluster_root / "field-report-localhost.md"
    report.write_text("previous report\n", encoding="utf-8")
    run_id = "2026-03-06_localhost_candidate"
    _seed_promotable_run(cluster_root, run_id, "localhost")
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"
    crash_after_first_report = """
import os
import runpy
import sys
sys.argv = sys.argv[1:]
main = runpy.run_path(sys.argv[0])["main"]
original = main.__globals__["_replace_report"]
def replace_and_exit(source, destination):
    original(source, destination)
    os._exit(73)
main.__globals__["_replace_report"] = replace_and_exit
main()
"""
    options = ["--repo-root", str(tmp_path), "--skip-validate-localhost-report"]
    crashed = subprocess.run(
        [sys.executable, "-c", crash_after_first_report, str(script), *options, "--run-id", run_id],
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed.returncode == 73, crashed.stdout or crashed.stderr
    assert json.loads((current / "manifest.json").read_text()) == {"run_id": run_id}

    abandoned = current.parent / ".promotion-abandoned"
    abandoned.mkdir()
    (abandoned / ".owned-stage").touch()
    (abandoned / "staged-log").write_text("partial copy\n", encoding="utf-8")
    unrelated = current.parent / ".promotion-owner-work"
    unrelated.mkdir()
    retried = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path), "--recover"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert retried.returncode == 0, retried.stdout or retried.stderr
    assert json.loads((current / "manifest.json").read_text()) == {"run_id": "prior"}
    assert report.read_text(encoding="utf-8") == "previous report\n"
    assert list(current.parent.glob(".promotion-*")) == [unrelated]


def test_promotion_excludes_only_mutable_legacy_watcher_receipts(tmp_path: Path) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-06_localhost_legacy_watcher"
    run = _seed_promotable_run(cluster_root, run_id, "localhost")
    legacy_status = f"raw/{run_id}_postrun_promote_watch_status.json"
    raw_status = run / legacy_status
    _write_json(raw_status, {"run_id": run_id, "status": "watching"})
    source_manifest = _write_v2_manifest(run, run_id)
    source_hash = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    _write_json(raw_status, {"run_id": run_id, "status": "promoted"})
    legacy_log = f"raw/{run_id}_promote.log"
    (run / legacy_log).write_text("watch finished\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"

    proc = subprocess.run(
        [
            sys.executable, str(script), "--repo-root", str(tmp_path), "--run-id", run_id,
            "--skip-render-localhost-report", "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout or proc.stderr
    published = cluster_root / "published" / "current"
    manifest = json.loads((published / "manifest.json").read_text())
    assert manifest["publication_projection"] == {
        "source_manifest_sha256": source_hash,
        "excluded_mutable_operational_files": sorted([legacy_status, legacy_log]),
    }
    assert set(manifest["files"]) == set(source_manifest["files"]) - {legacy_status}
    assert not (published / legacy_status).exists()
    assert not (published / legacy_log).exists()
    assert json.loads(raw_status.read_text())["status"] == "promoted"
    assert json.loads((run / "manifest.json").read_text()) == source_manifest


def test_field_report_validator_rejects_a_malformed_url_and_checks_other_links(tmp_path: Path) -> None:
    cluster = tmp_path / "cluster"
    docs = cluster / "docs"
    docs.mkdir(parents=True)
    report = cluster / "field-report.md"
    report.write_text("Canonical run: `example`.\n\n[Broken URL](http://[x)\n\n[Missing](missing.json)\n")
    (cluster / "field-report-notes.md").write_text("Canonical run: `example`.\n")
    (docs / "field-report-template.md").write_text("# Template\n")
    (docs / "advanced-runbook.md").write_text("# Runbook\n")
    _write_json(cluster / "published" / "current" / "manifest.json", {"run_id": "example"})
    validator = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "validate_field_report_requirements.sh"

    result = subprocess.run(
        ["bash", str(validator), "--repo-root", str(tmp_path), "--package-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "target=http://[x resolved=invalid URL" in result.stdout
    assert "target=missing.json" in result.stdout


def test_field_report_validator_fails_when_hygiene_check_cannot_read_report(tmp_path: Path) -> None:
    cluster = Path(__file__).resolve().parents[1] / "cluster"
    report = tmp_path / "field-report.md"
    report.write_bytes((cluster / "field-report.md").read_bytes() + b"\n\xff\n")

    result = subprocess.run(
        [
            "bash", str(cluster / "scripts/validate_field_report_requirements.sh"),
            "--report", str(report), "--report-link-base", str(cluster),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Package validation: failures=0" in result.stdout
    assert "FAIL: stale-artifact hygiene check failed" in result.stdout


def test_promote_run_materializes_published_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-06_localhost_common_answer_fast_r4"
    _seed_promotable_run(cluster_root, run_id, "localhost")

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--skip-render-localhost-report",
            "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)

    assert payload["success"] is True
    assert (cluster_root / "published" / "current" / "manifest.json").exists()
    assert (cluster_root / "published" / "current" / "structured" / f"{run_id}_suite_steps.json").exists()
    assert (cluster_root / "published" / "current" / "raw" / f"{run_id}_suite" / "suite.log").exists()
    assert (cluster_root / "published" / "current" / "figures" / f"{run_id}_cluster_story_dashboard.png").exists()


@pytest.mark.parametrize(
    "old_run_link", [None, "[Prior evidence]({path})", "<img src='{path}'>", "[Prior][asset]\n\n[asset]: <{path}>"]
)
def test_promotion_applies_cleanup_plan_after_installing_package(
    tmp_path: Path, monkeypatch, old_run_link: str | None
) -> None:
    cluster_root = tmp_path / "cluster"
    run_id = "2026-03-06_localhost_current"
    obsolete_id = "2026-03-05_localhost_obsolete"
    keep_id = "2026-03-04_localhost_comparison"
    kept_run = _seed_promotable_run(cluster_root, keep_id, "localhost")
    obsolete_run = _seed_promotable_run(cluster_root, obsolete_id, "localhost")
    accepted_run = _seed_promotable_run(cluster_root, run_id, "localhost")
    current = cluster_root / "published" / "current"
    _write_json(current / "manifest.json", {"run_id": obsolete_id})
    if old_run_link:
        (cluster_root / "custom-report.md").write_text(
            old_run_link.format(path=f"runs/{obsolete_id}/manifest.json") + "\n", encoding="utf-8"
        )
    monkeypatch.setenv("AISP_CLUSTER_SUITE_QUEUE_LOCK_PATH", str(tmp_path / "suite.lock"))
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"

    proc = subprocess.run(
        [
            sys.executable, str(script), "--repo-root", str(tmp_path), "--run-id", run_id,
            "--skip-render-localhost-report", "--skip-validate-localhost-report", "--cleanup",
            "--allow-run-id", keep_id, "--publish-report-path", "cluster/custom-report.md",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == int(bool(old_run_link)), proc.stdout or proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["published"] is True
    assert payload["steps"]["cleanup_plan"]["success"] is not bool(old_run_link)
    assert json.loads((current / "manifest.json").read_text())["run_id"] == run_id
    assert accepted_run.is_dir()
    assert kept_run.is_dir()
    if old_run_link:
        assert "Cleanup would remove evidence linked" in payload["steps"]["cleanup_plan"]["stderr"]
        assert "cleanup" not in payload["steps"]
        assert obsolete_run.exists()
        registry = current.parent / ".report-paths.json"
        assert str(cluster_root / "custom-report.md") in json.loads(registry.read_text())["reports"]
        cleanup = script.with_name("cleanup_run_artifacts.sh")
        preview = subprocess.run(
            [
                "bash", str(cleanup), "--repo-root", str(tmp_path), "--canonical-run-id", run_id,
                "--allow-run-id", keep_id, "--plan-file", str(tmp_path / "standalone-plan.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert preview.returncode == 1
        assert "custom-report.md" in preview.stderr
        assert not (tmp_path / "standalone-plan.json").exists()
    else:
        assert payload["steps"]["cleanup"]["success"]
        assert not obsolete_run.exists()


def test_promote_run_renders_published_localhost_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-06_localhost_common_answer_fast_r4"
    run_dir = _seed_promotable_run(cluster_root, run_id, "localhost")
    report_relpath = "reports/field-report-localhost.md"
    (run_dir / "reports").mkdir()
    (run_dir / report_relpath).write_text("immutable run-local report\n", encoding="utf-8")
    _write_v2_manifest(run_dir, run_id)
    (cluster_root / "docs").mkdir(parents=True, exist_ok=True)
    (cluster_root / "docs" / "advanced-runbook.md").write_text("# runbook\n", encoding="utf-8")
    (cluster_root / "docs" / "field-report-template.md").write_text("# template\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "promote_run.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--publish-report-path",
            "cluster/guides/field-report-localhost.md",
            "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    report_path = cluster_root / "guides" / "field-report-localhost.md"
    notes_path = cluster_root / "field-report-localhost-notes.md"
    top_level_report = cluster_root / "field-report.md"
    top_level_notes = cluster_root / "field-report-notes.md"
    assert report_path.exists()
    assert notes_path.exists()
    assert top_level_report.exists()
    assert top_level_notes.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert f"../published/current/structured/{run_id}_suite_steps.json" in report_text
    published_reports = cluster_root / "published" / "current" / "reports"
    assert (published_reports / "field-report-localhost.md").read_text() == "immutable run-local report\n"
    publication_report = (published_reports / "publication" / "field-report-localhost.md").read_text()
    assert f"../../structured/{run_id}_suite_steps.json" in publication_report
    published = published_reports.parent
    manifest = json.loads((published / "manifest.json").read_text())
    assert "reports/publication/field-report-localhost.md" in manifest["publication_projection"]["rendered_reports"]
    assert set(manifest["files"]) == {
        str(path.relative_to(published))
        for path in published.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    for name, digest in manifest["summary"]["sha256"].items():
        assert hashlib.sha256((published / name).read_bytes()).hexdigest() == digest
    assert (cluster_root / "published" / "current" / "reports" / "field-report-localhost-notes.md").exists()
    assert (cluster_root / "published" / "current" / "reports" / "field-report.md").exists()
    assert (cluster_root / "published" / "current" / "reports" / "field-report-notes.md").exists()
    assert f"(published/current/structured/{run_id}_suite_steps.json)" in top_level_report.read_text()
    assert top_level_notes.read_text(encoding="utf-8") == notes_path.read_text(encoding="utf-8")
