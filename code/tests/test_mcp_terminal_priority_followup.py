"""Exercise terminal stdio signals and admission after restarting the MCP server."""

import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import mcp.mcp_server as server
from core.jobs import JobStore
from core.utils import cooperative_subprocess as owned


def _request(number, tool, arguments):
    return {
        "jsonrpc": "2.0",
        "id": number,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


@pytest.mark.parametrize(
    "phase", ["redaction", "tool_json", "rpc_json", "event_json", "emission", "all"]
)
def test_sigint_during_terminal_encoding_updates_response_and_event(
    phase, tmp_path, monkeypatch, capsys
):
    inner = {
        "stdout": "nested stdout",
        "stderr": "nested stderr",
        "returncode": 1,
        "execution": {"process_group_id": 71, "completion_scope": "cleanup_incomplete"},
    }
    nested_event = io.StringIO()
    owned.emit_cleanup_incomplete_event(inner, nonce="child-nonce", stream=nested_event)
    outer = {
        "stdout": "outer stdout",
        "stderr": nested_event.getvalue(),
        "returncode": 1,
        "execution": {
            "process_group_id": 72,
            "completion_scope": "process_group_quiescent",
            "cleanup_event_nonce": "child-nonce",
        },
    }
    error = SystemExit(1)
    error.outer_process_result = outer
    instance = server.MCPServer()

    def fail(_arguments):
        raise error

    monkeypatch.setattr(server, "_QUEUE_DIR", tmp_path)
    monkeypatch.setattr(server, "JOB_STORE", JobStore())
    monkeypatch.setitem(server.HANDLERS, "terminal_fixture", fail)
    request = _request(29, "terminal_fixture", {})
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))

    redact = server.redact_sensitive_data
    encode = json.dumps
    fired = []

    def signal_once(stage):
        if (phase == stage or phase == "all") and stage not in fired:
            fired.append(stage)
            signal.raise_signal(signal.SIGINT)

    def redact_and_interrupt(value, **kwargs):
        result = redact(value, **kwargs)
        if "cleanup_incomplete" in value:
            signal_once("redaction")
        return result

    def encode_and_interrupt(value, **kwargs):
        line = encode(value, **kwargs)
        if isinstance(value, dict):
            if value.get("tool") == "terminal_fixture":
                signal_once("tool_json")
            if value.get("id") == 29:
                signal_once("rpc_json")
            if value.get("nonce") == "parent-nonce":
                signal_once("event_json")
        return line

    emit = instance._emit_stdio_json

    def interrupt_on_emission(payload, *, failure_context):
        signal_once("emission")
        return emit(payload, failure_context=failure_context)

    monkeypatch.setattr(instance, "_emit_stdio_json", interrupt_on_emission)
    monkeypatch.setattr(server, "redact_sensitive_data", redact_and_interrupt)
    monkeypatch.setattr(json, "dumps", encode_and_interrupt)
    assert instance.run_stdio(cleanup_event_nonce="parent-nonce") == 130
    expected_count = 5 if phase == "all" else 1
    assert len(fired) == expected_count
    streams = capsys.readouterr()
    response = json.loads(streams.out)
    assert response["id"] == 29 and response["result"]["isError"]
    failed = json.loads(response["result"]["content"][0]["text"])["result"]
    assert failed["stdout"] == outer["stdout"]
    assert failed["stderr"] == outer["stderr"]
    assert failed["cancelled"] is True
    assert failed["cancellation"]["process_group_id"] == 72
    assert failed["cancellation"]["interrupt_count"] == expected_count
    nested = failed["cleanup_incomplete"]
    assert nested["stdout"] == inner["stdout"]
    assert nested["stderr"] == inner["stderr"]
    assert nested["execution"] == inner["execution"]
    assert nested["cancellation"]["process_group_id"] == 71
    assert nested["cancellation"]["interrupt_count"] == expected_count
    assert (
        owned.parse_cleanup_incomplete_event(streams.err, expected_nonce="parent-nonce") == nested
    )


def test_restarted_stdio_reports_unverifiable_cleanup_and_serves_job_status(tmp_path):
    code_root = Path(__file__).resolve().parents[1]
    barrier = {
        "event": "aisp_cleanup_unverifiable",
        "error": "Incomplete cleanup event requires a receipt object",
        "child_result": {"stdout": "prior child stdout", "stderr": "invalid child event"},
    }
    path = tmp_path / "cleanup_incomplete.json"
    path.write_text(json.dumps(barrier), encoding="utf-8")
    requests = [
        _request(1, "gpu_topology_matrix", {"include_context": False}),
        _request(2, "job_status", {"job_id": "missing-job", "include_context": False}),
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path\nimport sys\nimport mcp.mcp_server as m\n"
            "m._QUEUE_DIR = Path(sys.argv.pop(1))\nraise SystemExit(m.main())",
            str(tmp_path),
            "--serve",
        ],
        cwd=code_root,
        env={**os.environ, "PYTHONPATH": str(code_root)},
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    blocked, status = map(json.loads, result.stdout.splitlines())
    assert blocked["id"] == 1 and blocked["result"]["isError"]
    refused = json.loads(blocked["result"]["content"][0]["text"])["result"]
    assert refused["admission_blocked"] is True
    assert refused["blocking_cleanup"] == barrier
    assert refused["cleanup_barrier_path"] == str(path)
    assert "stdout" not in refused and "cancelled" not in refused
    assert status["id"] == 2
    assert json.loads(status["result"]["content"][0]["text"])["result"]["status"] == "not_found"


@pytest.mark.parametrize("record", ["in_memory", "unreadable", "symlink"])
def test_handler_exception_cannot_replace_admission_refusal(tmp_path, monkeypatch, record):
    monkeypatch.setattr(server, "_QUEUE_DIR", tmp_path)
    monkeypatch.setattr(server, "JOB_STORE", JobStore())
    monkeypatch.setattr(server, "_UNVERIFIABLE_CLEANUP", {})
    path = server._cleanup_barrier_path()
    if record == "in_memory":
        server._remember_unverifiable_cleanup({"returncode": 1}, ValueError("missing receipt"))
        expected = json.loads(path.read_text())
    elif record == "symlink":
        target = tmp_path / "outside-record"
        target.write_text("untrusted target content")
        path.symlink_to(target)
        expected = {"error": "Cleanup barrier is a symbolic link"}
    else:
        path.write_text("{")
        with pytest.raises(json.JSONDecodeError) as caught:
            json.loads("{")
        expected = {"error": str(caught.value)}

    def wrapped(_arguments):
        try:
            server._assert_owned_work_admission()
        except Exception as exc:
            raise RuntimeError("handler-specific failure") from exc
        pytest.fail("Admission allowed owned work")

    monkeypatch.setitem(server.HANDLERS, "wrapped", wrapped)
    result = server.MCPServer().call_tool("wrapped", {})
    assert result.is_error
    refused = json.loads(result.content[0]["text"])["result"]
    assert refused["admission_blocked"] is True
    assert refused["blocking_cleanup"] == expected
