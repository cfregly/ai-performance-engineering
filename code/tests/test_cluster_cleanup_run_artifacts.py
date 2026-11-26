"""Exercise cluster cleanup through its shell entry point in isolated repositories."""

import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "cluster/scripts/cleanup_run_artifacts.sh"
CANONICAL = "2026-01-01_localhost_r2"
STALE = "2026-01-01_localhost"
BASELINE = "2026-01-01_localhost_r3"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run(root: Path, *args: str, canonical: str = CANONICAL) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--repo-root", str(root), "--canonical-run-id", canonical, *args],
        env={
            **os.environ,
            "AISP_CLUSTER_SUITE_QUEUE_LOCK_PATH": str(
                root / "artifacts/parallel_runs/queue.runner.lock"
            ),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path / f"cluster/runs/{CANONICAL}/manifest.json", json.dumps({"run_id": CANONICAL}))
    _write(tmp_path / f"cluster/runs/{STALE}/raw/data.log", "old run\n")
    return tmp_path


def test_rejects_missing_or_mismatched_canonical_manifest(root: Path) -> None:
    plan = root / "cleanup.json"
    rejected = _run(root, "--plan-file", str(plan), canonical=CANONICAL + "_typo")
    assert rejected.returncode != 0
    assert "no matching run or published manifest" in rejected.stderr
    assert not plan.exists()

    _write(root / f"cluster/runs/{CANONICAL}/manifest.json", json.dumps({"run_id": STALE}))
    rejected = _run(root, "--plan-file", str(plan))
    assert rejected.returncode != 0
    assert "does not match" in rejected.stderr
    assert not plan.exists()
    assert (root / f"cluster/runs/{STALE}/raw/data.log").is_file()


@pytest.mark.parametrize(
    "manifest",
    [
        "cluster/published/current/manifest.json",
        f"cluster/results/structured/{CANONICAL}_manifest.json",
    ],
)
def test_accepts_published_and_legacy_canonical_manifests(tmp_path: Path, manifest: str) -> None:
    _write(tmp_path / manifest, json.dumps({"run_id": CANONICAL}))
    old = _write(tmp_path / f"cluster/runs/{STALE}/raw/data.log", "old run\n")
    preview = _run(tmp_path)
    assert preview.returncode == 0, preview.stderr
    assert f"STALE_PATH {old.parent.parent.resolve()}" in preview.stdout
    assert old.is_file()


def test_apply_requires_saved_plan_and_keeps_canonical_and_allowlisted_artifacts(
    root: Path,
) -> None:
    plan = root / "cleanup.json"
    allow = ("--allow-run-id", BASELINE)
    current = _write(root / f"cluster/results/structured/{CANONICAL}_result.json", "{}")
    baseline = _write(root / f"cluster/results/raw/{BASELINE}_launch.log", "baseline\n")
    stale = _write(root / f"cluster/results/raw/{STALE}_launch.log", "stale\n")
    older = root / f"cluster/runs/{STALE}"
    other_published = "2026-02-02_published"
    published = _write(root / f"cluster/runs/{other_published}/raw/data.log", "published\n")
    _write(
        root / "cluster/published/current/manifest.json", json.dumps({"run_id": other_published})
    )

    rejected = _run(root, "--apply", *allow)
    assert rejected.returncode != 0
    assert "--apply requires --plan-file" in rejected.stderr
    preview = _run(root, "--plan-file", str(plan), *allow)
    assert preview.returncode == 0, preview.stderr
    reviewed = plan.read_bytes()
    assert {target["path"] for target in json.loads(reviewed)["targets"]} == {
        str(stale.relative_to(root)),
        str(older.relative_to(root)),
    }
    assert _run(root, "--plan-file", str(plan), *allow).returncode != 0
    assert plan.read_bytes() == reviewed

    rejected = _run(root, "--apply", "--plan-file", str(plan))
    assert rejected.returncode != 0
    assert stale.is_file()
    applied = _run(root, "--apply", "--plan-file", str(plan), *allow)
    assert applied.returncode == 0, applied.stderr
    assert "Deleted 2 stale artifact paths." in applied.stdout
    assert all(path.is_file() for path in (current, baseline, published))
    assert (root / f"cluster/runs/{CANONICAL}/manifest.json").is_file()
    assert not stale.exists()
    assert not older.exists()


@pytest.mark.parametrize("change", ["add_path", "replace_target", "edit_nested_file", "edit_plan"])
def test_apply_rejects_drift_before_deleting(root: Path, change: str) -> None:
    plan = root / "cleanup.json"
    preview = _run(root, "--plan-file", str(plan))
    assert preview.returncode == 0, preview.stderr
    stale = root / f"cluster/runs/{STALE}"
    if change == "add_path":
        _write(root / "cluster/runs/2026-01-02_new/raw/data.log", "new\n")
    elif change == "replace_target":
        stale.rename(root / "old")
        _write(stale / "raw/data.log", "replacement\n")
    elif change == "edit_nested_file":
        _write(stale / "raw/data.log", "edited\n")
    else:
        reviewed = json.loads(plan.read_text())
        reviewed["targets"] = []
        _write(plan, json.dumps(reviewed))
    rejected = _run(root, "--apply", "--plan-file", str(plan))
    assert rejected.returncode != 0
    assert "Cleanup plan differs" in rejected.stderr
    assert (stale / "raw/data.log").is_file()


def test_apply_rejects_symlink_target(root: Path) -> None:
    plan = root / "cleanup.json"
    preview = _run(root, "--plan-file", str(plan))
    assert preview.returncode == 0, preview.stderr
    stale = root / f"cluster/runs/{STALE}"
    stale.rename(root / "old")
    canonical = root / f"cluster/runs/{CANONICAL}"
    stale.symlink_to(canonical, target_is_directory=True)

    rejected = _run(root, "--apply", "--plan-file", str(plan))
    assert rejected.returncode != 0
    assert "symlink" in rejected.stderr
    assert stale.is_symlink()
    assert (canonical / "manifest.json").is_file()


def test_apply_refuses_while_suite_queue_lock_is_held(root: Path) -> None:
    plan = root / "cleanup.json"
    preview = _run(root, "--plan-file", str(plan))
    assert preview.returncode == 0, preview.stderr
    lock = _write(root / "artifacts/parallel_runs/queue.runner.lock", "")
    with lock.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rejected = _run(root, "--apply", "--plan-file", str(plan))
    assert rejected.returncode != 0
    assert "Queue lock is held" in rejected.stderr
    assert (root / f"cluster/runs/{STALE}/raw/data.log").is_file()
