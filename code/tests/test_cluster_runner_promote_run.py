from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict

from cluster.scripts import promotion_process
import core.cluster.runner as cluster_runner


def test_promote_cluster_run_builds_expected_command(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cmd(cmd, *, cwd=None, timeout_seconds=None, promotion=False):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["timeout_seconds"] = timeout_seconds
        captured["promotion"] = promotion
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
    assert captured["promotion"] is True
    assert result["published_localhost_report_path"].endswith("cluster/field-report-localhost.md")
    assert result["published_root"].endswith("cluster/published/current")
    assert "repo_root" in result


def test_promote_cluster_run_passes_repo_root_to_command(monkeypatch) -> None:
    captured: Dict[str, Any] = {}
    expected_root = Path("/tmp/fake_repo").resolve()

    def fake_run_cmd(cmd, *, cwd=None, timeout_seconds=None, promotion=False):  # type: ignore[no-untyped-def]
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


def test_promote_reports_launch_failure_without_exception_text(monkeypatch, caplog) -> None:
    sensitive = "private launch error detail"

    def fail(*args, **kwargs):
        raise FileNotFoundError(sensitive)

    monkeypatch.setattr(cluster_runner, "run_promotion_process", fail)
    payload = cluster_runner.promote_cluster_run(run_id="sample")

    assert payload["success"] is False
    assert payload["error_type"] == "FileNotFoundError"
    assert "executable or working directory was not found" in payload["error"]
    assert sensitive not in json.dumps(payload)
    assert "in fail" in caplog.text
    assert sensitive not in caplog.text


def test_field_report_validation_timeout_decodes_partial_output(monkeypatch) -> None:
    def time_out(cmd, *, timeout, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout, output=b"partial report\xff", stderr=b"validator warning")

    monkeypatch.setattr(cluster_runner.subprocess, "run", time_out)
    payload = cluster_runner.validate_field_report_requirements()

    assert payload["success"] is False
    assert payload["error"] == "timeout after 120s"
    assert payload["stdout"] == "partial report\ufffd"
    assert payload["stderr"] == "validator warning"
    json.dumps(payload)


def test_promotion_timeout_signals_only_its_owned_process_group(monkeypatch) -> None:
    events = []

    class OwnedProcess:
        pid = 12345

        def __init__(self, command, *, start_new_session, **kwargs):
            self.command = command
            assert start_new_session is True

        def __enter__(self):
            return self

        def communicate(self, *, timeout):
            raise subprocess.TimeoutExpired(self.command, timeout, output=b'{"published": true}')

        def __exit__(self, *exc):
            events.append("closed")

    monkeypatch.setattr(promotion_process.subprocess, "Popen", OwnedProcess)
    monkeypatch.setattr(promotion_process.os, "killpg", lambda pid, signum: events.append((pid, signum)))

    payload = cluster_runner.promote_cluster_run(run_id="sample", cleanup=True)
    assert payload["success"] is False
    assert payload["error"] == "timeout after 300s"
    assert payload["timed_out"] is True
    assert payload["published"] is None
    assert payload["stdout"] == '{"published": true}'
    assert events == [(OwnedProcess.pid, signal.SIGKILL), "closed"]
