from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_empty_canonical_run_leaves_package_destination_absent(tmp_path: Path) -> None:
    run_id = "2026-03-06_localhost_empty"
    run_dir = tmp_path / "cluster" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    output = tmp_path / "package"
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "build_canonical_package.py"

    proc = subprocess.run(
        [
            sys.executable, str(script), "--repo-root", str(tmp_path),
            "--canonical-run-id", run_id, "--output-dir", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "No canonical artifacts" in proc.stderr
    assert not output.exists()


def test_package_canonical_reader_snapshot_without_live_run(tmp_path: Path) -> None:
    run_id = "historical_canary"
    published = tmp_path / "cluster" / "published" / "current"
    artifact = published / "structured" / "summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    name = str(artifact.relative_to(published))
    manifest = {
        "run_id": run_id,
        "manifest_mode": "historical_reader_snapshot",
        "files": [name],
        "summary": {"file_count": 1, "sha256": {name: hashlib.sha256(artifact.read_bytes()).hexdigest()}},
    }
    (published / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "build_canonical_package.py"
    command = [sys.executable, str(script), "--repo-root", str(tmp_path), "--canonical-run-id", run_id]

    output = tmp_path / "package"
    proc = subprocess.run([*command, "--output-dir", str(output)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout or proc.stderr
    packaged = output / "runs" / run_id
    assert (packaged / name).read_bytes() == artifact.read_bytes()
    assert json.loads((packaged / "manifest.json").read_text()) == manifest
    assert "published snapshot" in (output / "README.md").read_text()

    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    rejected = tmp_path / "rejected"
    proc = subprocess.run([*command, "--output-dir", str(rejected)], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "hash mismatch" in proc.stderr
    assert not rejected.exists()


def test_build_canonical_package_materializes_live_runs_and_tracks_missing_historical(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    (cluster_root / "runs").mkdir(parents=True)

    canonical_run = "2026-03-05_localhost_modern_profile_r24_full20b"
    comparison_run = "2026-03-04_localhost_modern_profile_r22_fastcanon"
    historical_run = "2026-02-10_full_suite_e2e_wire_qf_mon"

    canonical_run_dir = cluster_root / "runs" / canonical_run
    (canonical_run_dir / "structured").mkdir(parents=True)
    (canonical_run_dir / "raw" / f"{canonical_run}_suite").mkdir(parents=True)
    (canonical_run_dir / "figures").mkdir(parents=True)
    (canonical_run_dir / "reports").mkdir(parents=True)
    (canonical_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (canonical_run_dir / "structured" / f"{canonical_run}_suite_steps.json").write_text("[]", encoding="utf-8")
    (canonical_run_dir / "figures" / f"{canonical_run}_cluster_story_dashboard.png").write_text("png", encoding="utf-8")

    comparison_run_dir = cluster_root / "runs" / comparison_run
    (comparison_run_dir / "structured").mkdir(parents=True)
    (comparison_run_dir / "raw").mkdir(parents=True)
    (comparison_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (comparison_run_dir / "raw" / f"{comparison_run}_suite.log").write_text("log", encoding="utf-8")

    (cluster_root / "field-report-localhost.md").write_text(f"Canonical run: `{canonical_run}`\n", encoding="utf-8")
    (cluster_root / "field-report-localhost-notes.md").write_text(f"Baseline: `{comparison_run}`\n", encoding="utf-8")
    (cluster_root / "multi_node_harden_and_verify_next_steps.md").write_text("placeholder\n", encoding="utf-8")
    (cluster_root / "field-report.md").write_text(
        f"Historical run: `{historical_run}`\nresults/structured/{historical_run}_manifest.json\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "build_canonical_package.py"
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(repo_root),
        "--canonical-run-id",
        canonical_run,
        "--comparison-run-id",
        comparison_run,
        "--historical-run-id",
        historical_run,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    package_root = cluster_root / "canonical_package"
    manifest = json.loads((package_root / "package_manifest.json").read_text(encoding="utf-8"))
    runs = {entry["run_id"]: entry for entry in manifest["runs"]}

    assert runs[canonical_run]["present"] is True
    assert runs[canonical_run]["layout"] == "run_dir"
    assert runs[canonical_run]["counts"]["structured"] >= 1
    assert runs[comparison_run]["present"] is True
    assert runs[comparison_run]["counts"]["raw"] == 1
    assert runs[historical_run]["present"] is False
    assert manifest["historical_markdown_references"][historical_run]["file_count"] >= 1

    assert (package_root / "runs" / canonical_run / "manifest.json").exists()
    assert (package_root / "field-report-localhost.md").exists()
    assert (package_root / "field-report-localhost-notes.md").exists()
    assert (package_root / "historical-multinode-reference.md").exists()
    assert (package_root / "cleanup_keep_run_ids.txt").read_text(encoding="utf-8").splitlines() == [
        canonical_run,
        comparison_run,
        historical_run,
    ]
