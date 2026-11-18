"""Stop trace replay when a nested CLI reports incomplete cleanup."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mcp.mcp_server as server
from core.jobs import JobStore
from core.utils import cooperative_subprocess as owned
from labs.trace_replay.runner import capture_plan

CODE_ROOT = Path(__file__).resolve().parents[1]


def _capture_plan(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "native-operator-capture-plan.v1",
                "events": [
                    {
                        "event_id": f"event-{index}",
                        "operation": "prefix_scan",
                        "parameters": {"numel": 17},
                        "seed": 900 + index,
                        "source": {
                            "kind": "generated_by_recipe",
                            "description": "CPU cleanup barrier fixture",
                        },
                    }
                    for index in range(2)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _cleanup_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("event") == "aisp_cleanup_incomplete":
            events.append(payload)
    return events


def _write_cleanup_fixture(
    tmp_path: Path,
    receipt: dict,
) -> tuple[Path, Path]:
    launches = tmp_path / "launches.jsonl"
    fixture = tmp_path / "nested_cleanup_fixture.py"
    fixture.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from core.utils.cooperative_subprocess import emit_cleanup_incomplete_event\n"
        "receipt = json.loads(os.environ['AISP_REPLAY_TEST_RECEIPT'])\n"
        "nonce = os.environ['AISP_CLEANUP_EVENT_NONCE']\n"
        "with Path(os.environ['AISP_REPLAY_TEST_LAUNCHES']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps({'event_id': sys.argv[1], 'nonce': nonce}) + '\\n')\n"
        "sys.stdout.write(receipt['stdout'])\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write(receipt['stderr'])\n"
        "sys.stderr.flush()\n"
        "if os.environ.get('AISP_REPLAY_TEST_MALFORMED') == '1':\n"
        "    print(json.dumps({'event': 'aisp_cleanup_incomplete', 'nonce': nonce, 'receipt': None}), file=sys.stderr, flush=True)\n"
        "else:\n"
        "    emit_cleanup_incomplete_event(receipt, nonce=nonce)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    return fixture, launches


def _cpu_fixture_runner(fixture: Path):
    def run(command, *, cwd, env):
        child_env, nonce = owned.prepare_cleanup_event_env(env)
        event_id = command[command.index("--run-id") + 1]
        completed = subprocess.run(
            [sys.executable, str(fixture), event_id],
            cwd=cwd,
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "command": "CPU trace-replay fixture",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout_seconds": None,
            "timeout_hit": False,
            "cancelled": False,
            "execution": {
                "process_id": 5151,
                "process_group_id": 5151,
                "start_new_session": True,
                "cleanup_event_nonce": nonce,
                "live_process_group_members": [],
                "completion_scope": "process_group_quiescent",
            },
        }

    return run


def _owned_result_with_cleanup_event(receipt: dict) -> dict:
    nonce = "trace-replay-cleanup-event-nonce"
    stream = io.StringIO()
    owned.emit_cleanup_incomplete_event(receipt, nonce=nonce, stream=stream)
    return {
        "command": "CPU trace-replay fixture",
        "returncode": 1,
        "stdout": "outer stdout\n",
        "stderr": "outer stderr\n" + stream.getvalue(),
        "timeout_seconds": None,
        "timeout_hit": False,
        "cancelled": False,
        "execution": {
            "process_id": 5151,
            "process_group_id": 5151,
            "start_new_session": True,
            "cleanup_event_nonce": nonce,
            "live_process_group_members": [],
            "completion_scope": "process_group_quiescent",
        },
    }


def test_trace_tool_consumes_wrapper_nonce_and_reemits_exact_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from labs.trace_replay import trace_replay_tool

    receipt = {
        "stdout": "trace tool nested stdout\n",
        "stderr": "trace tool nested stderr\n",
        "execution": {
            "process_group_id": 2468,
            "completion_scope": "cleanup_incomplete",
        },
    }
    nonce = "trace-tool-wrapper-nonce"
    monkeypatch.setenv("AISP_CLEANUP_EVENT_NONCE", nonce)

    def fail_replay(*args, **kwargs):
        assert "AISP_CLEANUP_EVENT_NONCE" not in os.environ
        raise owned.CooperativeCleanupIncomplete(receipt)

    monkeypatch.setattr(trace_replay_tool, "replay_bundle", fail_replay)
    assert trace_replay_tool.main(
        [
            "replay",
            "--bundle",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / "replay"),
        ]
    ) == 1

    stderr = capsys.readouterr().err
    assert owned.parse_cleanup_incomplete_event(stderr, expected_nonce=nonce) == receipt
    assert "AISP_CLEANUP_EVENT_NONCE" not in os.environ


def test_owned_cli_receipt_materializes_exact_raw_stream_bytes(tmp_path: Path) -> None:
    from labs.trace_replay.runner import _persist_owned_cli_result

    event_root = tmp_path / "event"
    event_root.mkdir()
    raw_stdout = b"raw CPU stdout\xff\n"
    result = {
        "stdout": "raw CPU stdout\ufffd\n",
        "stderr": "exact CPU stderr\n",
        "returncode": 1,
        "raw_streams_base64": {
            "stdout": base64.b64encode(raw_stdout).decode("ascii"),
        },
        "execution": {
            "process_group_id": 5151,
            "completion_scope": "process_group_quiescent",
        },
    }
    row = {}

    _persist_owned_cli_result(event_root, row, result)

    assert (event_root / "stdout.log").read_bytes() == raw_stdout
    assert (event_root / "stderr.log").read_bytes() == b"exact CPU stderr\n"
    assert json.loads(
        (event_root / "process-receipt.json").read_text(encoding="utf-8")
    ) == result
    assert row["execution"] == result["execution"]


def test_owned_cli_cleanup_barrier_wins_when_evidence_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.trace_replay import runner

    cleanup_receipt = {
        "stdout": "nested stdout\n",
        "stderr": "nested stderr\n",
        "execution": {
            "process_group_id": 4242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    outer_result = {
        "stdout": "outer stdout\n",
        "stderr": "outer stderr\n",
        "returncode": 1,
        "execution": {
            "process_group_id": 5151,
            "completion_scope": "process_group_quiescent",
        },
    }
    persistence_error = OSError("evidence volume is full")
    monkeypatch.setattr(
        runner,
        "run_cooperative_process",
        lambda *args, **kwargs: outer_result,
    )
    monkeypatch.setattr(
        runner,
        "authenticated_cleanup_receipt_or_raise",
        lambda result: cleanup_receipt,
    )

    def fail_persistence(*args, **kwargs) -> None:
        raise persistence_error

    monkeypatch.setattr(runner, "_persist_owned_cli_result", fail_persistence)

    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        runner.run_owned_cli(["fixture"], {}, tmp_path, {})

    assert caught.value.result == cleanup_receipt
    assert caught.value.__cause__ is persistence_error
    assert caught.value.evidence_persistence_error is persistence_error
    assert caught.value.outer_process_result == outer_result


def test_owned_cli_plain_post_handoff_interrupt_keeps_operator_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.trace_replay import runner

    cleanup_receipt = {
        "stdout": "nested stdout\n",
        "stderr": "nested stderr\n",
        "execution": {
            "process_group_id": 4242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    outer_result = _owned_result_with_cleanup_event(cleanup_receipt)
    outer_result["cancelled"] = True
    outer_result["cancellation"] = {
        "signal": "SIGINT",
        "process_group_id": 5151,
        "reason": "interrupt",
        "signal_forwarded": False,
        "signal_not_forwarded_reason": "process_group_already_quiescent",
        "operator_interrupt_observed": True,
        "interrupt_count": 1,
        "repeated_interrupts": 0,
        "status": "cleanup_completed",
    }
    interrupt = KeyboardInterrupt("interrupt after completed ownership handoff")
    setattr(interrupt, "cooperative_process_result", outer_result)  # noqa: B010

    def interrupt_after_handoff(*args, **kwargs):
        raise interrupt

    monkeypatch.setattr(runner, "run_cooperative_process", interrupt_after_handoff)
    event_root = tmp_path / "event"
    event_root.mkdir()
    row = {}

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_owned_cli(["fixture"], {}, event_root, row)

    assert caught.value is interrupt
    assert owned.operator_interrupt_from_error(caught.value)
    assert owned.cleanup_incomplete_receipt_from_error(caught.value) == cleanup_receipt
    assert caught.value.outer_process_result is outer_result
    assert json.loads(
        (event_root / "process-receipt.json").read_text(encoding="utf-8")
    ) == outer_result
    assert (event_root / "stdout.log").read_text(encoding="utf-8") == outer_result["stdout"]
    assert (event_root / "stderr.log").read_text(encoding="utf-8") == outer_result["stderr"]
    assert row["execution"] == outer_result["execution"]


def test_owned_cli_persists_before_repeated_interrupts_during_cleanup_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.trace_replay import runner

    cleanup_receipt = {
        "stdout": "nested stdout\n",
        "stderr": "nested stderr\n",
        "execution": {
            "process_group_id": 4242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    outer_result = _owned_result_with_cleanup_event(cleanup_receipt)
    monkeypatch.setattr(
        runner,
        "run_cooperative_process",
        lambda *args, **kwargs: outer_result,
    )
    event_root = tmp_path / "event"
    event_root.mkdir()
    row = {}
    first_interrupt = KeyboardInterrupt("first interrupt during cleanup parsing")
    interrupts = [
        first_interrupt,
        KeyboardInterrupt("second interrupt during cleanup parsing"),
    ]
    actual_parser = runner.authenticated_cleanup_receipt_or_raise
    parse_calls = 0

    def interrupt_cleanup_parsing(result):
        nonlocal parse_calls
        parse_calls += 1
        assert (event_root / "stdout.log").is_file()
        assert (event_root / "stderr.log").is_file()
        assert (event_root / "process-receipt.json").is_file()
        if interrupts:
            raise interrupts.pop(0)
        return actual_parser(result)

    monkeypatch.setattr(
        runner,
        "authenticated_cleanup_receipt_or_raise",
        interrupt_cleanup_parsing,
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_owned_cli(["fixture"], {}, event_root, row)

    assert caught.value is first_interrupt
    assert parse_calls == 3
    assert owned.operator_interrupt_from_error(caught.value)
    assert owned.cleanup_incomplete_receipt_from_error(caught.value) == cleanup_receipt
    assert outer_result["cancellation"]["interrupt_count"] == 2
    assert outer_result["cancellation"]["repeated_interrupts"] == 1
    assert outer_result["cancellation"]["status"] == "cleanup_completed"
    persisted = json.loads(
        (event_root / "process-receipt.json").read_text(encoding="utf-8")
    )
    assert persisted["stdout"] == outer_result["stdout"]
    assert persisted["stderr"] == outer_result["stderr"]
    assert row["execution"] == outer_result["execution"]


def test_replay_stops_before_next_event_and_propagates_exact_inner_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.trace_replay import runner

    bundle = capture_plan(
        _capture_plan(tmp_path / "plan.json"),
        tmp_path / "capture",
        device="cpu",
    )
    expected_receipt = {
        "stdout": "CPU inner stdout before cleanup barrier\n",
        "stderr": "CPU inner stderr before cleanup barrier\n",
        "returncode": 1,
        "execution": {
            "process_group_id": 424242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    fixture, launches = _write_cleanup_fixture(tmp_path, expected_receipt)
    replay_output = tmp_path / "replay"
    parent_nonce = "parent-token-must-not-reach-child"
    monkeypatch.setenv("AISP_CLEANUP_EVENT_NONCE", parent_nonce)
    monkeypatch.setenv("AISP_REPLAY_TEST_FIXTURE", str(fixture))
    monkeypatch.setenv("AISP_REPLAY_TEST_LAUNCHES", str(launches))
    monkeypatch.setenv("AISP_REPLAY_TEST_RECEIPT", json.dumps(expected_receipt))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "run_cooperative_process", _cpu_fixture_runner(fixture))
    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        runner.replay_bundle(bundle.root, replay_output, "portable")

    assert caught.value.result == expected_receipt
    launched = [json.loads(line) for line in launches.read_text(encoding="utf-8").splitlines()]
    assert [item["event_id"] for item in launched] == ["event-0"]
    assert launched[0]["nonce"] != parent_nonce
    assert not (replay_output / "event-1").exists()
    assert (replay_output / "event-0" / "stdout.log").read_text(
        encoding="utf-8"
    ) == expected_receipt["stdout"]
    stderr_log = (replay_output / "event-0" / "stderr.log").read_text(encoding="utf-8")
    assert stderr_log.startswith(expected_receipt["stderr"])
    child_events = _cleanup_events(stderr_log)
    assert len(child_events) == 1
    assert child_events[0]["receipt"] == expected_receipt
    assert child_events[0]["nonce"] == launched[0]["nonce"]


def test_malformed_authenticated_cleanup_is_terminal_before_next_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from labs.trace_replay import runner

    bundle = capture_plan(
        _capture_plan(tmp_path / "plan.json"),
        tmp_path / "capture",
        device="cpu",
    )
    fixture_receipt = {
        "stdout": "CPU malformed-event stdout\n",
        "stderr": "CPU malformed-event stderr\n",
        "returncode": 1,
        "execution": {
            "process_group_id": 424242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    fixture, launches = _write_cleanup_fixture(tmp_path, fixture_receipt)
    replay_output = tmp_path / "replay"
    monkeypatch.setenv("AISP_REPLAY_TEST_LAUNCHES", str(launches))
    monkeypatch.setenv("AISP_REPLAY_TEST_RECEIPT", json.dumps(fixture_receipt))
    monkeypatch.setenv("AISP_REPLAY_TEST_MALFORMED", "1")
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "run_cooperative_process", _cpu_fixture_runner(fixture))

    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        runner.replay_bundle(bundle.root, replay_output, "portable")

    receipt = caught.value.result
    assert receipt["execution"]["completion_scope"] == "cleanup_incomplete"
    assert receipt["execution"]["live_process_group_members"] is None
    assert receipt["execution"]["cleanup_protocol_scope"] == "authenticated_nested_event"
    assert receipt["execution"]["outer_completion_scope"] == "process_group_quiescent"
    assert receipt["outer_execution"]["completion_scope"] == "process_group_quiescent"
    assert "requires a receipt object" in receipt["cleanup_protocol_error"]
    assert receipt["stdout"] == fixture_receipt["stdout"]
    assert receipt["stderr"].startswith(fixture_receipt["stderr"])
    launched = [json.loads(line) for line in launches.read_text(encoding="utf-8").splitlines()]
    assert [item["event_id"] for item in launched] == ["event-0"]
    assert not (replay_output / "event-1").exists()
    outer = caught.value.outer_process_result
    assert json.loads(
        (replay_output / "event-0" / "process-receipt.json").read_text(encoding="utf-8")
    ) == outer
    assert (replay_output / "event-0" / "stdout.log").read_text(
        encoding="utf-8"
    ) == outer["stdout"]
    assert (replay_output / "event-0" / "stderr.log").read_text(
        encoding="utf-8"
    ) == outer["stderr"]


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "waitid"),
    reason="requires owned POSIX process support",
)
@pytest.mark.parametrize("malformed", [False, True], ids=["valid", "malformed"])
def test_nested_cleanup_event_stops_replay_and_persists_mcp_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,
) -> None:
    bundle = capture_plan(
        _capture_plan(tmp_path / "plan.json"),
        tmp_path / "capture",
        device="cpu",
    )
    expected_receipt = {
        "stdout": "CPU inner stdout before cleanup barrier\n",
        "stderr": "CPU inner stderr before cleanup barrier\n",
        "returncode": 1,
        "execution": {
            "process_group_id": 424242,
            "completion_scope": "cleanup_incomplete",
        },
    }
    fixture, launches = _write_cleanup_fixture(tmp_path, expected_receipt)
    (tmp_path / "sitecustomize.py").write_text(
        "import os, sys\n"
        "from labs.trace_replay import runner\n"
        "original_popen = runner.subprocess.Popen\n"
        "runner.torch.cuda.is_available = lambda: True\n"
        "def cpu_replay_popen(command, *args, **kwargs):\n"
        "    values = list(command) if isinstance(command, (list, tuple)) else []\n"
        "    if values[1:5] == ['-m', 'cli.aisp', 'bench', 'run']:\n"
        "        event_id = values[values.index('--run-id') + 1]\n"
        "        command = [sys.executable, os.environ['AISP_REPLAY_TEST_FIXTURE'], event_id]\n"
        "    return original_popen(command, *args, **kwargs)\n"
        "runner.subprocess.Popen = cpu_replay_popen\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + str(CODE_ROOT)
    environment["AISP_REPLAY_TEST_FIXTURE"] = str(fixture)
    environment["AISP_REPLAY_TEST_LAUNCHES"] = str(launches)
    environment["AISP_REPLAY_TEST_RECEIPT"] = json.dumps(expected_receipt)
    if malformed:
        environment["AISP_REPLAY_TEST_MALFORMED"] = "1"
    replay_output = tmp_path / "replay"
    queue = tmp_path / "queue"
    store = JobStore()
    monkeypatch.setattr(server, "JOB_STORE", store)
    monkeypatch.setattr(server, "_QUEUE_DIR", queue)
    monkeypatch.setattr(server, "_QUEUE_LOG_PATH", queue / "queue.log")
    monkeypatch.setattr(server, "_QUEUE_SCRIPT_PATH", queue / "queue.sh")
    monkeypatch.setattr(server, "_UNVERIFIABLE_CLEANUP", {})
    monkeypatch.setattr(server, "_subprocess_env", lambda: dict(environment))
    try:
        with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
            server.tool_tools_trace_replay(
                {
                    "args": [
                        "replay",
                        "--bundle",
                        str(bundle.root),
                        "--output-dir",
                        str(replay_output),
                        "--validity-profile",
                        "portable",
                    ],
                    "include_context": False,
                }
            )
    finally:
        store._executor.shutdown(wait=True)

    barrier_receipt = caught.value.result
    if malformed:
        assert barrier_receipt["execution"]["completion_scope"] == "cleanup_incomplete"
        assert barrier_receipt["execution"]["cleanup_protocol_scope"] == (
            "authenticated_nested_event"
        )
        assert "requires a receipt object" in barrier_receipt["cleanup_protocol_error"]
    else:
        assert barrier_receipt == expected_receipt
    barrier = json.loads(server._cleanup_barrier_path().read_text(encoding="utf-8"))
    assert barrier == {"event": "aisp_cleanup_incomplete", "receipt": barrier_receipt}
    launched = [json.loads(line) for line in launches.read_text(encoding="utf-8").splitlines()]
    assert [item["event_id"] for item in launched] == ["event-0"]
    assert not (replay_output / "event-1").exists()

    stdout_log = (replay_output / "event-0" / "stdout.log").read_text(encoding="utf-8")
    stderr_log = (replay_output / "event-0" / "stderr.log").read_text(encoding="utf-8")
    assert stdout_log == expected_receipt["stdout"]
    assert stderr_log.startswith(expected_receipt["stderr"])
    child_events = _cleanup_events(stderr_log)
    assert len(child_events) == 1
    assert child_events[0]["receipt"] == (None if malformed else expected_receipt)
    child_nonce = launched[0]["nonce"]
    assert child_events[0]["nonce"] == child_nonce

    outer = caught.value.outer_process_result
    outer_events = _cleanup_events(outer["stderr"])
    outer_nonce = outer["execution"]["cleanup_event_nonce"]
    assert len(outer_events) == 2
    assert all(event["receipt"] == barrier_receipt for event in outer_events)
    wrapper_nonces = {event["nonce"] for event in outer_events}
    assert outer_nonce in wrapper_nonces
    assert len(wrapper_nonces | {child_nonce}) == 3
