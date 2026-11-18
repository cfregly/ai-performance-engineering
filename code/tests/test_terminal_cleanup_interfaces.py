"""Retain terminal CPU cleanup failures without admitting later owned work."""

from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import typer

from core.jobs import JobStore
from core.utils import cooperative_subprocess as owned
import mcp.mcp_server as server


CODE_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires owned POSIX process support")
CPU_FAULT = """
import sys
import core.utils.cooperative_subprocess as owned
original_communicate = owned.communicate_cooperatively
original_query = owned._live_process_group_members
observed = []
def observe(process, **kwargs):
    observed.append(process)
    return original_communicate(process, **kwargs)
def denied(group):
    if observed and observed[-1].stdout.closed and observed[-1].stderr.closed:
        raise PermissionError('CPU fixture final group observation denied')
    return original_query(group)
def cpu_failure():
    owned.communicate_cooperatively = observe
    owned._live_process_group_members = denied
    return owned.run_cooperative_process([sys.executable, '-c',
        "import sys; print('CPU child stdout'); print('CPU child stderr', file=sys.stderr)"])
"""


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    store = JobStore()
    queue = tmp_path / "queue"
    monkeypatch.setattr(server, "JOB_STORE", store)
    monkeypatch.setattr(server, "_QUEUE_DIR", queue)
    monkeypatch.setattr(server, "_QUEUE_LOG_PATH", queue / "queue.log")
    monkeypatch.setattr(server, "_QUEUE_SCRIPT_PATH", queue / "queue.sh")
    monkeypatch.setattr(server, "_UNVERIFIABLE_CLEANUP", {})
    yield
    store._executor.shutdown(wait=True)


def _receipt(operator: bool = False) -> dict:
    result = {"stdout": "CPU fixture stdout", "stderr": "CPU fixture stderr", "returncode": 0,
              "execution": {"process_group_id": 123, "completion_scope": "cleanup_incomplete"}}
    if operator:
        result["cancellation"] = {"reason": "interrupt", "operator_interrupt_observed": True}
    return result


def _malformed_nested_interrupt(nonce: str) -> owned.CooperativeProcessInterrupted:
    malformed = json.dumps({
        "event": "aisp_cleanup_incomplete",
        "nonce": nonce,
        "receipt": None,
    })
    return owned.CooperativeProcessInterrupted({
        "stdout": "outer stdout",
        "stderr": malformed + "\n",
        "returncode": 130,
        "execution": {
            "process_group_id": 456,
            "completion_scope": "process_group_quiescent",
            "cleanup_event_nonce": nonce,
        },
    })


def test_stdio_malformed_authenticated_event_persists_unverifiable_barrier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = _malformed_nested_interrupt("malformed-stdio-event-nonce")
    instance = server.MCPServer()
    monkeypatch.setattr(instance, "handle_message", lambda _message: (_ for _ in ()).throw(error))
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"),
    )

    assert instance.run_stdio() == 130
    responses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(responses) == 1
    failed = json.loads(responses[0]["result"]["content"][0]["text"])["result"]
    assert "cleanup_protocol_error" in failed
    barrier = json.loads(server._cleanup_barrier_path().read_text())
    assert barrier["event"] == "aisp_cleanup_unverifiable"
    with pytest.raises(RuntimeError, match="unverifiable cleanup"):
        server._assert_owned_work_admission()


def test_background_malformed_authenticated_event_finishes_error_job_and_blocks_work() -> None:
    error = _malformed_nested_interrupt("malformed-background-event-nonce")
    ticket = server._queue_job("cpu-malformed-cleanup", lambda: (_ for _ in ()).throw(error))
    server.JOB_STORE._executor.shutdown(wait=True)

    record = server.JOB_STORE.get_status(ticket["job_id"])
    assert record["status"] == "error"
    assert record["finished_at"] is not None
    assert "cleanup_protocol_error" in record["result"]
    barrier = json.loads(server._cleanup_barrier_path().read_text())
    assert barrier["event"] == "aisp_cleanup_unverifiable"
    with pytest.raises(RuntimeError, match="unverifiable cleanup"):
        server._assert_owned_work_admission()


@pytest.mark.parametrize("module", ["cli.aisp", "core.benchmark.bench_commands"])
@pytest.mark.parametrize("operator", [False, True])
def test_cli_main_emits_one_terminal_event_and_keeps_interrupt_priority(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: str, operator: bool,
) -> None:
    cli = importlib.import_module(module)
    receipt = _receipt(operator)
    error = owned.CooperativeProcessInterrupted({"stdout": "outer", "stderr": "outer", "execution": {
        "completion_scope": "process_group_quiescent"}}) if operator else owned.CooperativeCleanupIncomplete(receipt)
    if operator:
        error.cleanup_incomplete_receipt = receipt
    app = typer.Typer()

    @app.command()
    def fail() -> None:
        raise error

    monkeypatch.setattr(cli, "app", app)
    monkeypatch.setattr(sys, "argv", ["cpu-terminal-fixture"])
    nonce = "cpu-cli-terminal-event-nonce"
    monkeypatch.setenv("AISP_CLEANUP_EVENT_NONCE", nonce)
    assert cli.main() == (130 if operator else 1)
    stderr = capsys.readouterr().err
    assert owned.parse_cleanup_incomplete_event(stderr, expected_nonce=nonce) == receipt
    events = [line for line in stderr.splitlines() if line.startswith('{"event": "aisp_cleanup_incomplete"')]
    assert len(events) == 1


@pytest.mark.parametrize("launcher", ["finite", "cooperative", "diagnostic"])
def test_real_cli_terminal_event_crosses_owned_mcp_launchers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, launcher: str,
) -> None:
    fixture = tmp_path / "cpu_cli_failure.py"
    fixture.write_text(CPU_FAULT + "\nimport core.benchmark.bench_commands as cli\ncli.app = cpu_failure\nraise SystemExit(cli.main())\n")
    # Replace only the external command with a CPU failure driver. It invokes
    # the real CLI main, helper and stderr protocol without a GPU benchmark.
    if launcher == "cooperative":
        run = server.run_cooperative_process
        monkeypatch.setattr(server, "run_cooperative_process", lambda command, **kwargs: run(
            [sys.executable, str(fixture)], **kwargs,
        ))
    else:
        import core.diagnostics.process as diagnostic
        run = diagnostic.run_owned_process
        monkeypatch.setattr(diagnostic, "run_owned_process", lambda command, **kwargs: run(
            [sys.executable, str(fixture)], **kwargs,
        ))
    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        if launcher == "diagnostic":
            server._run_diagnostic_tools_cli("cpu-fixture")
        else:
            server._run_bench_cli(
                ["run"],
                timeout=None if launcher == "cooperative" else 17,
                uncapped=launcher == "cooperative",
            )
    receipt = caught.value.result
    assert "CPU child stdout" in receipt["stdout"]
    assert "CPU child stderr" in receipt["stderr"]
    assert receipt["execution"]["completion_scope"] == "cleanup_incomplete"
    outer = caught.value.outer_process_result
    assert owned.parse_cleanup_incomplete_event(
        outer["stderr"],
        expected_nonce=outer["execution"]["cleanup_event_nonce"],
    ) == receipt
    assert server._cleanup_barrier_path().is_file()


def test_diagnostic_launcher_rejects_work_after_incomplete_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.diagnostics.process as diagnostic

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic process launched after incomplete cleanup")

    server._remember_terminal_cleanup(owned.CooperativeCleanupIncomplete(_receipt()))
    monkeypatch.setattr(diagnostic, "run_owned_process", forbidden)
    with pytest.raises(owned.CooperativeCleanupIncomplete):
        server._run_diagnostic_tools_cli("cpu-fixture")


def test_child_operator_event_and_trace_interrupt_remain_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt(True)
    nonce = "cpu-child-operator-event-nonce"
    stream = io.StringIO()
    owned.emit_cleanup_incomplete_event(receipt, nonce=nonce, stream=stream)
    monkeypatch.setattr(server, "run_cooperative_process", lambda command, **kwargs: {
        "returncode": 130,
        "stdout": "outer stdout",
        "stderr": stream.getvalue(),
        "execution": {
            "process_group_id": 456,
            "completion_scope": "process_group_quiescent",
            "cleanup_event_nonce": nonce,
        },
    })
    with pytest.raises(owned.CooperativeProcessInterrupted) as caught:
        server._run_bench_cli(["run"], timeout=None)
    assert caught.value.result["stdout"] == "outer stdout"
    assert caught.value.cleanup_incomplete_receipt == receipt
    monkeypatch.setattr(server, "JOB_STORE", JobStore())
    monkeypatch.setattr(server, "_QUEUE_DIR", server._QUEUE_DIR / "trace")
    original = owned.CooperativeProcessInterrupted(receipt)

    def interrupted(*args, **kwargs):
        raise original

    monkeypatch.setattr(server, "run_cooperative_process", interrupted)
    with pytest.raises(owned.CooperativeProcessInterrupted) as trace:
        server.tool_tools_trace_replay({"args": ["--help"], "include_context": False})
    assert trace.value is original
    server.JOB_STORE._executor.shutdown(wait=True)


def test_real_stdio_emits_terminal_response_then_stops_before_next_request(tmp_path: Path) -> None:
    probe = tmp_path / "next-request-ran"
    hook = CPU_FAULT + """
import os
from pathlib import Path
import mcp.mcp_server as server
server._QUEUE_DIR = Path(os.environ['CPU_QUEUE'])
@server.register_tool('cpu_cleanup_failure', 'CPU failure fixture', {'type':'object','properties':{}})
def failure(params):
    return cpu_failure()
@server.register_tool('cpu_probe', 'CPU admission probe', {'type':'object','properties':{}})
def next_request(params):
    Path(os.environ['CPU_PROBE']).write_text('ran')
    return {'value':'probe'}
"""
    (tmp_path / "sitecustomize.py").write_text(hook)
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(CODE_ROOT),
               CPU_QUEUE=str(tmp_path / "shared-queue"), CPU_PROBE=str(probe))
    nonce = "cpu-stdio-terminal-event-nonce"
    env["AISP_CLEANUP_EVENT_NONCE"] = nonce
    requests = "".join(json.dumps({"jsonrpc":"2.0","id":number,"method":"tools/call",
                                 "params":{"name":name,"arguments":{}}}) + "\n"
                       for number, name in ((1,"cpu_cleanup_failure"),(2,"cpu_probe")))
    result = subprocess.run([sys.executable, "-c", "from mcp.mcp_server import main\nraise SystemExit(main())\n", "--serve"],
                            cwd=CODE_ROOT, env=env, input=requests, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 1
    assert responses[0]["id"] == 1 and responses[0]["result"]["isError"] is True
    failed = json.loads(responses[0]["result"]["content"][0]["text"])["result"]
    assert "CPU child stdout" in failed["stdout"]
    assert "CPU child stderr" in failed["stderr"]
    assert failed["cleanup_incomplete"]["execution"]["completion_scope"] == "cleanup_incomplete"
    assert owned.parse_cleanup_incomplete_event(
        result.stderr,
        expected_nonce=nonce,
    ) == failed["cleanup_incomplete"]
    assert not probe.exists()
    assert (tmp_path / "shared-queue" / "cleanup_incomplete.json").is_file()


def test_background_fatal_is_terminal_and_blocks_already_queued_work(monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(max_workers=1)
    monkeypatch.setattr(server, "JOB_STORE", store)
    entered = threading.Event()
    release = threading.Event()
    later = threading.Event()
    namespace = {}
    exec(CPU_FAULT, namespace)
    original_communicate = owned.communicate_cooperatively
    original_query = owned._live_process_group_members

    def first():
        entered.set()
        release.wait()
        return namespace["cpu_failure"]()

    def forbidden():
        later.set()
        raise AssertionError("Owned work ran after terminal cleanup failure")

    try:
        first_ticket = server._queue_job("cpu-fatal", first)
        entered.wait()
        second_ticket = server._queue_job("cpu-later", forbidden)
        release.set()
        store._executor.shutdown(wait=True)
        for ticket in (first_ticket, second_ticket):
            record = store.get_status(ticket["job_id"])
            assert record["status"] == "error"
            assert record["finished_at"] is not None
            assert record["result"]["cleanup_incomplete"]["execution"]["completion_scope"] == "cleanup_incomplete"
        assert not later.is_set()
        assert store.incomplete_cleanup_receipt() is not None
        assert server._cleanup_barrier_path().is_file()
    finally:
        release.set()
        store._executor.shutdown(wait=True)
        owned.communicate_cooperatively = original_communicate
        owned._live_process_group_members = original_query
