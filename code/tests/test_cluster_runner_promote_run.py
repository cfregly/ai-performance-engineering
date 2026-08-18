from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import core.cluster.runner as cluster_runner


def test_promote_cluster_run_builds_expected_command(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cmd(cmd, *, cwd=None, timeout_seconds=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": '{"success": true, "steps": {"publish_package_sync": {"success": true}}}',
            "stderr": "",
            "duration_ms": 12,
        }

    monkeypatch.setattr(cluster_runner, "_run_cmd", fake_run_cmd)

    result = cluster_runner.promote_cluster_run(
        run_id="2026-03-06_localhost_common_answer_fast_r4",
        label="localhost",
        allow_run_ids=["2026-03-04_localhost_modern_profile_r22_fastcanon"],
        cleanup=True,
    )

    assert result["success"] is True
    assert result["run_id"] == "2026-03-06_localhost_common_answer_fast_r4"
    assert "--run-id" in captured["cmd"]
    assert "--cleanup" in captured["cmd"]
    assert "--allow-run-id" in captured["cmd"]
    assert "--skip-flat-sync" not in captured["cmd"]
    assert result["published_localhost_report_path"].endswith("cluster/field-report-localhost.md")
    assert result["published_root"].endswith("cluster/published/current")
    assert "repo_root" in result


def test_promote_cluster_run_passes_repo_root_to_command(monkeypatch) -> None:
    captured: Dict[str, Any] = {}
    expected_root = Path("/tmp/fake_repo").resolve()

    def fake_run_cmd(cmd, *, cwd=None, timeout_seconds=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": '{"success": true}',
            "stderr": "",
            "duration_ms": 1,
        }

    monkeypatch.setattr(cluster_runner, "_run_cmd", fake_run_cmd)

    cluster_runner.promote_cluster_run(
        run_id="rid",
        skip_render_localhost_report=True,
        skip_validate_localhost_report=True,
        repo_root="/tmp/fake_repo",
    )

    assert "--repo-root" in captured["cmd"]
    repo_root_index = captured["cmd"].index("--repo-root") + 1
    assert captured["cmd"][repo_root_index] == str(expected_root)
    assert captured["cwd"] == expected_root
