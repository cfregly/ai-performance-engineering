from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import core.cluster.runner as cluster_runner


def test_watch_cluster_run_for_promotion_launches_detached_watcher(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-07_localhost_modern_profile_r28_full20b"
    run_dir = cluster_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    legacy_status = run_dir / "raw" / f"{run_id}_postrun_promote_watch_status.json"
    legacy_status.write_text('{"status": "published"}\n', encoding="utf-8")

    monkeypatch.setattr(cluster_runner, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(cluster_runner, "_cluster_root", lambda: cluster_root)
    monkeypatch.setattr(
        cluster_runner,
        "_cluster_run_dir",
        lambda rid: run_dir if rid == run_id else cluster_root / "runs" / rid,
    )

    captured: Dict[str, Any] = {}

    class DummyProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None, start_new_session=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout_name"] = getattr(stdout, "name", None)
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        return DummyProc(43210)

    monkeypatch.setattr(cluster_runner.subprocess, "Popen", fake_popen)

    result = cluster_runner.watch_cluster_run_for_promotion(
        run_id=run_id,
        pid=12345,
        cleanup=True,
        allow_run_ids=["2026-03-05_localhost_modern_profile_r24_full20b"],
        poll_interval_seconds=5.0,
    )

    assert result["success"] is True
    assert result["watcher_pid"] == 43210
    assert "--run-id" in captured["cmd"]
    assert "--pid" in captured["cmd"]
    assert "--cleanup" in captured["cmd"]
    assert "--allow-run-id" in captured["cmd"]
    assert "--repo-root" in captured["cmd"]
    assert str(repo_root.resolve()) in captured["cmd"]
    assert captured["cwd"] == str(repo_root)
    assert captured["start_new_session"] is True
    launch_log = run_dir / "progress" / f"{run_id}_postrun_promote_watch.launch.log"
    assert captured["stdout_name"] == str(launch_log)
    assert result["launch_log_path"] == str(launch_log)
    assert result["watch_status_path"] == str(run_dir / "progress" / legacy_status.name)
    assert result["legacy_watch_status_path"] == str(legacy_status)
    assert legacy_status.read_text(encoding="utf-8") == '{"status": "published"}\n'
