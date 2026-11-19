"""Check cleanup priority where the CLI receives and forwards tool output."""

from __future__ import annotations

import base64
import io
import json
import os
from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.tools import tools_commands as cli
from core.utils import cooperative_subprocess as owned

NONCE = "tool-cleanup-priority-fixture"


def _nested_receipt() -> dict:
    return {
        "stdout": "inner output",
        "stderr": "inner diagnostics",
        "returncode": 1,
        "execution": {"completion_scope": "cleanup_incomplete", "process_group_id": 543},
    }


def _outer_receipt(receipt: object, *, nonce: str = NONCE) -> dict:
    event = {"event": "aisp_cleanup_incomplete", "nonce": nonce, "receipt": receipt}
    return {
        "stdout": "outer output",
        "stderr": json.dumps(event) + "\n",
        "returncode": 3,
        "execution": {
            "completion_scope": "process_group_quiescent",
            "process_group_id": 321,
            "live_process_group_members": [],
            "cleanup_event_nonce": NONCE,
        },
    }


@pytest.mark.parametrize("nonce", ["different-launch", "different-launch-é"])
def test_untrusted_malformed_event_is_ignored(nonce: str) -> None:
    assert owned.authenticated_cleanup_receipt_or_raise(_outer_receipt(None, nonce=nonce)) is None


def test_receipt_lookup_preserves_direct_and_nested_cleanup() -> None:
    inner = _nested_receipt()
    assert owned.authenticated_cleanup_receipt_or_raise(inner) is inner
    outer = _outer_receipt(inner)
    outer["stderr"] += "1" * 5000 + "\n"
    assert owned.authenticated_cleanup_receipt_or_raise(outer) == inner


def test_tool_cleanup_precedes_inexact_capture(monkeypatch, capsys) -> None:
    inner = _nested_receipt()
    outer = _outer_receipt(inner)
    outer["execution"]["capture_errors"] = ["invalid byte in stdout"]
    outer["raw_streams_base64"] = {"stdout": "/w=="}
    before = deepcopy(outer)
    monkeypatch.setattr(owned, "run_cooperative_process", Mock(return_value=outer))

    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        cli._run_tool("trace-replay", [])

    assert caught.value.result == inner
    assert caught.value.outer_process_result is outer
    assert outer == before
    assert caught.value.capture_evidence_error["indicators"] == [
        "execution.capture_errors",
        "raw_streams_base64",
    ]
    assert isinstance(caught.value.__cause__, cli.ToolCaptureEvidenceError)
    assert caught.value.__cause__.result is outer
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("interrupted", [False, True])
def test_malformed_authenticated_event_retains_outer_evidence(monkeypatch, interrupted) -> None:
    outer = _outer_receipt(None)
    before = deepcopy(outer)
    original = owned.CooperativeProcessInterrupted(outer)
    run = Mock(side_effect=original) if interrupted else Mock(return_value=outer)
    monkeypatch.setattr(owned, "run_cooperative_process", run)

    expected = (
        owned.CooperativeProcessInterrupted if interrupted else owned.CooperativeCleanupIncomplete
    )
    with pytest.raises(expected) as caught:
        cli._run_tool("trace-replay", [])

    if interrupted:
        assert caught.value is original
    barrier = owned.cleanup_incomplete_receipt_from_error(caught.value)
    assert barrier["execution"]["completion_scope"] == "cleanup_incomplete"
    assert barrier["execution"]["outer_completion_scope"] == "process_group_quiescent"
    assert barrier["execution"]["live_process_group_members"] is None
    assert barrier["execution"]["process_group_id"] == outer["execution"]["process_group_id"]
    assert barrier["outer_execution"] == outer["execution"]
    assert "requires a receipt object" in barrier["cleanup_protocol_error"]
    assert caught.value.cleanup_protocol_error == barrier["cleanup_protocol_error"]
    assert caught.value.outer_process_result is outer
    assert outer == before


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("failed_stream", ["stdout", "stderr"])
def test_tool_preserves_cleanup_when_a_stream_cannot_be_forwarded(
    monkeypatch, interrupted, failed_stream
) -> None:
    inner = _nested_receipt()
    outer = _outer_receipt(inner)
    original = owned.CooperativeProcessInterrupted(outer)
    if interrupted:
        outer["execution"]["capture_limit"] = {"streams_closed_before_eof": ["stdout"]}
    before = deepcopy(outer)
    run = Mock(side_effect=original) if interrupted else Mock(return_value=outer)
    monkeypatch.setattr(owned, "run_cooperative_process", run)

    broken = BrokenPipeError(f"{failed_stream} reader exited")
    other_name = "stderr" if failed_stream == "stdout" else "stdout"
    other_stream = io.StringIO()
    with monkeypatch.context() as streams:
        streams.setattr(cli.sys, failed_stream, Mock(write=Mock(side_effect=broken)))
        streams.setattr(cli.sys, other_name, other_stream)
        expected = (
            owned.CooperativeProcessInterrupted
            if interrupted
            else owned.CooperativeCleanupIncomplete
        )
        with pytest.raises(expected) as caught:
            cli._run_tool("trace-replay", [])

    if interrupted:
        assert caught.value is original
        assert original.cleanup_incomplete_receipt == inner
        assert original.capture_evidence_error["indicators"] == ["execution.capture_limit"]
    assert owned.cleanup_incomplete_receipt_from_error(caught.value) == inner
    assert caught.value.outer_process_result is outer
    assert outer == before
    assert caught.value.stream_forward_errors == {failed_stream: broken}
    assert caught.value.__cause__ is broken
    assert other_stream.getvalue() == outer[other_name]


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")),
    reason="requires process-group ownership",
)
def test_cli_retains_authenticated_event_and_raw_bytes_from_a_child(tmp_path, monkeypatch) -> None:
    inner = _nested_receipt()
    script = tmp_path / "cleanup_tool.py"
    script.write_text(
        "import os\n"
        "from core.utils.cooperative_subprocess import (\n"
        "    consume_cleanup_event_nonce, emit_cleanup_incomplete_event,\n"
        ")\n"
        "os.write(1, bytes([255]))\n"
        f"emit_cleanup_incomplete_event({inner!r}, nonce=consume_cleanup_event_nonce())\n",
        encoding="utf-8",
    )
    name = "cleanup-protocol-fixture"
    monkeypatch.setitem(cli.TOOLS, name, cli.ToolSpec(name, script, "", cooperative_process=True))

    with pytest.raises(owned.CooperativeCleanupIncomplete) as caught:
        cli._run_tool(name, [])

    assert caught.value.result == inner
    outer = caught.value.outer_process_result
    assert outer["execution"]["completion_scope"] == "process_group_quiescent"
    assert base64.b64decode(outer["raw_streams_base64"]["stdout"]) == bytes([255])
    assert "raw_streams_base64" in caught.value.capture_evidence_error["indicators"]
