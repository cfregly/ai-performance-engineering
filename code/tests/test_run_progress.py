from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import mcp.mcp_server as mcp_server
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness import run_benchmarks


def test_progress_recorder_writes_payload(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_001", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="baseline_timing",
            phase_index=1,
            total_phases=2,
            step="ch01:demo",
        )
    )
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_001"
    assert data["current"]["phase"] == "baseline_timing"
    assert data["current"]["step"] == "ch01:demo"
    assert data["current"]["run_id"] == "run_001"


def test_progress_recorder_uses_unique_temp_files_for_shared_progress_path(
    tmp_path: Path, monkeypatch
) -> None:
    progress_path = tmp_path / "progress" / "run_progress.json"
    recorder_a = ProgressRecorder(run_id="run_shared", progress_path=progress_path)
    recorder_b = ProgressRecorder(run_id="run_shared", progress_path=progress_path)

    replace_calls: list[tuple[str, str]] = []
    original_replace = Path.replace

    def _record_replace(self: Path, target: Path) -> Path:
        replace_calls.append((self.name, target.name))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _record_replace)

    recorder_a.emit(
        ProgressEvent(
            phase="baseline_timing",
            phase_index=1,
            total_phases=2,
            step="ch01:demo_a",
        )
    )
    recorder_b.emit(
        ProgressEvent(
            phase="optimized_timing",
            phase_index=2,
            total_phases=2,
            step="ch01:demo_b",
        )
    )

    assert len(replace_calls) == 2
    assert replace_calls[0][0] != replace_calls[1][0]
    assert all(target_name == "run_progress.json" for _, target_name in replace_calls)
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert data["current"]["step"] == "ch01:demo_b"
    assert not list(progress_path.parent.glob("*.tmp"))


def test_progress_recorder_surfaces_existing_history_load_failures(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress" / "run_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("{not-json", encoding="utf-8")

    recorder = ProgressRecorder(run_id="run_bad", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="run",
            phase_index=1,
            total_phases=2,
            step="resume",
        )
    )

    data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "load_warning" in data
    assert "Failed to load existing progress history" in data["load_warning"]


def test_job_status_includes_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000000"
    progress_path = run_dir / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_002", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="optimized_timing",
            phase_index=2,
            total_phases=2,
            step="ch01:demo",
        )
    )

    store = mcp_server.JOB_STORE

    def runner():
        return {"ok": True}

    ticket = store.queue_job(
        "run_benchmarks",
        runner,
        run_metadata={
            "run_id": "run_002",
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
        },
    )
    job_id = ticket["job_id"]
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["progress"]["phase"] == "optimized_timing"
        assert payload["progress"]["run_id"] == "run_002"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_surfaces_corrupt_progress_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000099"
    progress_path = run_dir / "progress" / "run_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("{not-json", encoding="utf-8")

    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_099",
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["progress"] is None
        assert "progress_warning" in payload
        assert "Failed to read progress payload" in payload["progress_warning"]
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_emit_progress_safe_warns_once_on_emit_failure(capsys) -> None:
    class BrokenRecorder:
        progress_path = "/tmp/bad-progress.json"

        def emit(self, event) -> None:
            raise RuntimeError("emit boom")

    original = set(mcp_server._PROGRESS_EMIT_WARNED_PATHS)
    mcp_server._PROGRESS_EMIT_WARNED_PATHS.clear()
    try:
        event = ProgressEvent(phase="analysis", phase_index=1, total_phases=1, step="broken")
        mcp_server._emit_progress_safe(BrokenRecorder(), event)
        mcp_server._emit_progress_safe(BrokenRecorder(), event)
        lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["event"] == "progress_emit_warning"
        assert "emit boom" in payload["message"]
    finally:
        mcp_server._PROGRESS_EMIT_WARNED_PATHS.clear()
        mcp_server._PROGRESS_EMIT_WARNED_PATHS.update(original)


def test_ensure_queue_artifacts_warns_on_init_failure(monkeypatch, capsys, tmp_path: Path) -> None:
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("not a directory", encoding="utf-8")

    original = set(mcp_server._QUEUE_ARTIFACT_WARNED_KEYS)
    mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.clear()
    try:
        monkeypatch.setattr(mcp_server, "_QUEUE_DIR", blocked_path)
        monkeypatch.setattr(mcp_server, "_QUEUE_SCRIPT_PATH", blocked_path / "bench_queue.sh")
        monkeypatch.setattr(mcp_server, "_QUEUE_LOG_PATH", blocked_path / "queue.log")

        assert mcp_server._ensure_queue_artifacts() is False
        lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["event"] == "queue_artifact_warning"
        assert payload["kind"] == "queue_dir_init"
        assert payload["path"] == str(blocked_path)
    finally:
        mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.clear()
        mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.update(original)


def test_queue_manifest_and_log_warnings_are_one_time(monkeypatch, capsys, tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    bad_manifest_path = queue_dir / "manifest_dir"
    bad_manifest_path.mkdir()
    bad_log_path = queue_dir / "log_dir"
    bad_log_path.mkdir()

    original = set(mcp_server._QUEUE_ARTIFACT_WARNED_KEYS)
    mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.clear()
    try:
        monkeypatch.setattr(mcp_server, "_ensure_queue_artifacts", lambda: True)
        monkeypatch.setattr(mcp_server, "_QUEUE_SCRIPT_PATH", bad_manifest_path)
        monkeypatch.setattr(mcp_server, "_QUEUE_LOG_PATH", bad_log_path)

        mcp_server._append_queue_script({"tool": "run_benchmarks"})
        mcp_server._append_queue_script({"tool": "run_benchmarks"})
        mcp_server._log_queue_event("first")
        mcp_server._log_queue_event("second")

        lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
        assert len(lines) == 2
        payloads = [json.loads(line) for line in lines]
        assert {payload["kind"] for payload in payloads} == {"queue_script_append", "queue_log_append"}
        for payload in payloads:
            assert payload["event"] == "queue_artifact_warning"
    finally:
        mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.clear()
        mcp_server._QUEUE_ARTIFACT_WARNED_KEYS.update(original)


def test_run_with_queue_records_start_and_returns_artifact_paths(
    monkeypatch, tmp_path: Path
) -> None:
    queue_dir = tmp_path / "queue"
    queue_script = queue_dir / "bench_queue.sh"
    queue_log = queue_dir / "queue.log"

    monkeypatch.setattr(mcp_server, "_QUEUE_DIR", queue_dir)
    monkeypatch.setattr(mcp_server, "_QUEUE_SCRIPT_PATH", queue_script)
    monkeypatch.setattr(mcp_server, "_QUEUE_LOG_PATH", queue_log)
    monkeypatch.setattr(mcp_server, "_wait_for_idle", lambda: True)

    def _wait_for_stop(stop_event, _overlap_event, **_kwargs) -> None:
        stop_event.wait(timeout=1.0)

    monkeypatch.setattr(mcp_server, "_monitor_overlap", _wait_for_stop)

    result = mcp_server._run_with_queue(
        "run_benchmarks",
        lambda: {"success": True, "returncode": 0},
        queue_label="bench run ch10:atomic_reduction",
        queue_payload={"targets": ["ch10:atomic_reduction"]},
    )

    assert result["queue"]["queue_log"] == str(queue_log)
    assert result["queue"]["queue_script"] == str(queue_script)
    log_text = queue_log.read_text(encoding="utf-8")
    assert "RUN_START" in log_text
    assert "tool=run_benchmarks" in log_text
    assert "RUN_END" in log_text


def test_progress_phases_include_llm() -> None:
    phases = run_benchmarks.PROGRESS_PHASES
    for key in (
        "llm_analysis",
        "llm_patch_apply",
        "llm_patch_rebenchmark",
        "llm_patch_verify",
        "llm_explain",
    ):
        assert key in phases


def test_global_progress_percent_uses_run_offset_instead_of_resetting() -> None:
    corrected = run_benchmarks._compute_global_progress_percent(
        completed_benchmarks=0,
        total_benchmarks=276,
        phase_index=4,
        total_phases=run_benchmarks.PROGRESS_TOTAL_PHASES,
        benchmark_offset=236,
    )
    local_reset = ((0 + ((4 - 1) / run_benchmarks.PROGRESS_TOTAL_PHASES)) / 14) * 100.0

    assert corrected is not None
    assert corrected > 80.0
    assert local_reset < 2.0


def test_job_status_running_without_artifacts_reports_effective_queued(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "missing_run"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_missing",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "queued"
        assert payload["reported_status"] == "running"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_queued_with_progress_reports_effective_running(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "20250101_000003"
    progress_path = run_dir / "progress" / "run_progress.json"
    recorder = ProgressRecorder(run_id="run_003", progress_path=progress_path)
    recorder.emit(
        ProgressEvent(
            phase="preflight",
            phase_index=1,
            total_phases=4,
            step="ch10:atomic_reduction",
        )
    )

    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "queued",
            "submitted_at": time.time(),
            "run_id": "run_003",
            "run_dir": str(run_dir),
            "progress_path": str(progress_path),
            "note": "Waiting for MCP queue runner.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "running"
        assert payload["reported_status"] == "queued"
        assert payload["progress"]["phase"] == "preflight"
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_running_via_queue_runner_keeps_running_without_run_dir(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "pending_run_dir"
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_pending",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
            "note": "Running via MCP queue runner.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "running"
        assert "reported_status" not in payload
    finally:
        with store._lock:
            store._store.pop(job_id, None)


def test_job_status_running_without_progress_with_existing_run_dir_reports_queued(tmp_path: Path) -> None:
    store = mcp_server.JOB_STORE
    job_id = f"run_benchmarks-{uuid.uuid4().hex[:8]}"
    run_dir = tmp_path / "artifacts" / "existing_run_dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    with store._lock:
        store._store[job_id] = {
            "job_id": job_id,
            "tool": "run_benchmarks",
            "status": "running",
            "submitted_at": time.time(),
            "run_id": "run_existing_dir",
            "run_dir": str(run_dir),
            "progress_path": str(run_dir / "progress" / "run_progress.json"),
            "note": "Poll job status to track completion.",
        }
    try:
        payload = mcp_server.tool_job_status({"job_id": job_id})
        assert payload["status"] == "queued"
        assert payload["reported_status"] == "running"
    finally:
        with store._lock:
            store._store.pop(job_id, None)
