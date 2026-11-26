"""Focused regressions for MCP transport and credential redaction."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import core.cluster
from core.api import handlers
from core.api.redaction import redact_sensitive_data
from mcp import mcp_server
from mcp.mcp_client import RobustMCPClient


def _call_registered_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = asyncio.run(
        mcp_server.MCPServer().handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    )
    assert response is not None
    return response["result"]


def test_mcp_payload_redacts_nested_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "q7!"
    monkeypatch.setitem(
        mcp_server.HANDLERS,
        "review_secret_redaction",
        lambda _arguments: {
            "success": True,
            "command": ["cluster-eval", "--nmx-token", secret],
            "echo": f"request used {secret}",
            "shell": f"cluster-eval --nmx-token='{secret}'",
        },
    )
    monkeypatch.setitem(
        mcp_server.TOOLS,
        "review_secret_redaction",
        mcp_server.ToolDefinition(
            name="review_secret_redaction",
            description="Review redaction behavior.",
            input_schema={
                "type": "object",
                "properties": {
                    "nmx_token": {"type": "string"},
                    "max_tokens": {"type": "integer"},
                },
            },
        ),
    )
    arguments = {
        "nmx_token": secret,
        "nested": {"client_secret": secret, "safe": "visible"},
        "tool_args": ["--authorization", secret, "--label=visible"],
        "max_tokens": 128,
    }

    result = mcp_server.MCPServer().call_tool("review_secret_redaction", arguments)
    payload = json.loads(result.content[0]["text"])
    serialized = json.dumps(payload)

    assert secret not in serialized
    assert payload["arguments"]["nmx_token"] == "<redacted>"
    assert payload["arguments"]["nested"] == {
        "client_secret": "<redacted>",
        "safe": "visible",
    }
    assert payload["arguments"]["tool_args"] == [
        "--authorization",
        "<redacted>",
        "--label=visible",
    ]
    assert payload["arguments"]["max_tokens"] == 128
    assert payload["input_schema"]["properties"]["nmx_token"] == {"type": "string"}


def test_short_credential_redaction_preserves_adjacent_nonsensitive_text() -> None:
    payload = redact_sensitive_data(
        {"message": "token x rejected; xylophone remains"},
        source={"nmx_token": "x"},
    )

    assert payload == {"message": "token <redacted> rejected; xylophone remains"}


def test_long_credential_redaction_covers_concatenated_text() -> None:
    payload = redact_sensitive_data(
        {"message": "prefixknown-secret-valuesuffix"},
        source={"client_secret": "known-secret-value"},
    )

    assert payload == {"message": "prefix<redacted>suffix"}


def test_mcp_tool_exception_does_not_expose_internal_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    detail = "unexpected-internal-detail"
    monkeypatch.setattr(mcp_server, "TOOLS", {})
    monkeypatch.setattr(mcp_server, "HANDLERS", {})
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    monkeypatch.setenv("AISP_MCP_DEBUG", "1")

    @mcp_server.register_tool("review_exception", "Check the MCP error response.")
    def fail(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(detail)

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "review_exception", "arguments": {}},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    asyncio.run(mcp_server.MCPServer().run_stdio())

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["id"] == request["id"]
    assert response["result"]["isError"] is True
    payload = json.loads(response["result"]["content"][0]["text"])

    assert detail not in captured.out
    assert "traceback" not in captured.out.lower()
    assert payload["result"] == {
        "error": "Tool execution failed.",
        "error_type": "RuntimeError",
        "success": False,
    }
    assert "MCP tool review_exception failed with RuntimeError." in captured.err
    assert "in fail" in captured.err
    assert detail not in captured.err


def test_mcp_cluster_promotion_preserves_public_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    response = asyncio.run(
        mcp_server.MCPServer().handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cluster_promote_run", "arguments": {}},
            }
        )
    )

    assert response is not None
    assert response["result"]["isError"] is True
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["result"] == {
        "error": "run_id is required",
        "error_type": "value_error",
        "success": False,
    }


@pytest.mark.parametrize(
    ("tool", "handler"),
    [
        ("cluster_eval_suite", "run_cluster_eval_suite"),
        ("cluster_common_eval", "run_cluster_common_eval"),
        ("cluster_fabric_eval", "run_cluster_fabric_eval"),
        ("cluster_nmx_partition_lab", "build_cluster_nmx_partition_lab"),
        ("cluster_build_canonical_package", "build_canonical_package"),
        ("cluster_promote_run", "promote_cluster_run"),
        ("cluster_watch_promote", "watch_cluster_run_for_promotion"),
        ("cluster_validate_field_report", "validate_field_report_requirements"),
    ],
)
def test_cluster_wrappers_use_generic_error_for_unexpected_failures(
    tool: str, handler: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    detail = "internal-backend-failure-detail"

    def fail(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(detail)

    monkeypatch.setattr(core.cluster, handler, fail)
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    result = _call_registered_tool(tool, {"run_id": "test_run", "pid": 42})

    assert result["isError"] is True
    assert detail not in json.dumps(result)
    payload = json.loads(result["content"][0]["text"])
    assert payload["result"] == {
        "error": "Tool execution failed.",
        "error_type": "RuntimeError",
        "success": False,
    }
    assert f"MCP tool {tool} failed with RuntimeError." in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        (
            "cluster_promote_run",
            {"run_id": "test_run", "timeout_seconds": "invalid"},
            "timeout_seconds must be an integer",
        ),
        (
            "cluster_watch_promote",
            {"run_id": "test_run", "pid": "invalid"},
            "pid must be an integer",
        ),
    ],
)
def test_cluster_wrappers_keep_invalid_parameter_messages(
    tool: str, arguments: dict[str, Any], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    result = _call_registered_tool(tool, arguments)
    assert result["isError"] is True
    assert json.loads(result["content"][0]["text"])["result"]["error"] == message


def test_registered_queue_timeout_keeps_public_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mcp_server.fcntl is None:
        pytest.skip("Queue runner locking requires fcntl")
    monkeypatch.setattr(mcp_server, "TOOLS", {})
    monkeypatch.setattr(mcp_server, "HANDLERS", {})
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    monkeypatch.setattr(mcp_server, "_QUEUE_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "_QUEUE_SCRIPT_PATH", tmp_path / "queue.sh")
    monkeypatch.setattr(mcp_server, "_QUEUE_LOG_PATH", tmp_path / "queue.log")
    monkeypatch.setenv("AISP_MCP_QUEUE_RUNNER_LOCK_TIMEOUT_SEC", "1")

    @mcp_server.register_tool("queue_probe", "Check the MCP queue error response.")
    def queued(_arguments: dict[str, Any]) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            pytest.fail("A timed-out queue request must never reach its callback")

        return mcp_server._run_with_queue("queue_probe", run)

    with (tmp_path / "queue.runner.lock").open("a+", encoding="utf-8") as lock:
        mcp_server.fcntl.flock(lock.fileno(), mcp_server.fcntl.LOCK_EX)
        result = _call_registered_tool("queue_probe", {})

    assert result["isError"] is True
    message = json.loads(result["content"][0]["text"])["result"]["error"]
    assert message.startswith("Queue runner lock timeout after 1s (tool=queue_probe,")


def test_malformed_benchmark_export_keeps_public_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    malformed = tmp_path / "benchmark_test_results.json"
    malformed.write_text("{", encoding="utf-8")
    output = tmp_path / "export.csv"
    result = _call_registered_tool(
        "benchmark_export", {"data_file": str(malformed), "output": str(output)}
    )

    assert result["isError"] is True
    message = json.loads(result["content"][0]["text"])["result"]["error"]
    assert f"Failed to read benchmark results {malformed}" in message
    assert not output.exists()


@pytest.mark.parametrize(
    ("format_name", "expected_row"),
    [
        ("csv", "ch_test:legacy_case,1.20,,python"),
        ("markdown", "| ch_test:legacy_case | 1.20x | N/A | python |"),
    ],
)
def test_mcp_exports_legacy_speedup_with_unavailable_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_name: str, expected_row: str
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    data = tmp_path / "benchmark_test_results.json"
    data.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "ch_test",
                        "benchmarks": [{"example": "legacy_case", "best_speedup": 1.2}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / f"export.{format_name}"
    arguments = {"data_file": str(data), "output": str(output), "format": format_name}
    result = _call_registered_tool("benchmark_export", arguments)

    assert result["isError"] is False
    assert expected_row in output.read_text(encoding="utf-8")

    triage = _call_registered_tool("benchmark_triage", {"data_file": str(data)})
    assert triage["isError"] is False
    payload = json.loads(triage["content"][0]["text"])["result"]
    assert payload["improvements"][0]["baseline_ms"] is None
    assert not any(item["tool"] == "profile_nsys" for item in payload["recommended_tools"])


def test_mcp_tools_list_declares_cluster_promotion_side_effects() -> None:
    response = asyncio.run(
        mcp_server.MCPServer().handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    )

    assert response is not None
    descriptions = {tool["name"]: tool["description"] for tool in response["result"]["tools"]}
    promotion = descriptions["cluster_promote_run"]
    watcher = descriptions["cluster_watch_promote"]
    assert "Replaces cluster/published/current" in promotion
    assert "applies a revalidated cleanup plan after publication" in promotion
    assert "Eligibility follows the retained run IDs regardless of age." in promotion
    assert "steps.recover_publication" in promotion
    assert "success, published" in promotion
    assert "A later cleanup failure can leave `success` false." in promotion
    assert "Starts a detached process and writes watcher logs and status files" in watcher
    assert "recovers interrupted publications before installing" in watcher
    assert "read-only snapshot" not in promotion
    assert "read-only snapshot" not in watcher


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("benchmark_data", {"page": 0}, "page must be >= 1"),
        ("benchmark_data", {"sort_dir": "sideways"}, "sort_dir must be 'asc' or 'desc'"),
        ("benchmark_opportunities", {"top": -1}, "top must be >= 0"),
        ("benchmark_compare", {}, "baseline is required"),
    ],
)
def test_benchmark_tools_keep_public_argument_errors(
    tool: str, arguments: dict[str, Any], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    result = _call_registered_tool(tool, arguments)

    assert result["isError"] is True
    error = json.loads(result["content"][0]["text"])["result"]
    assert error["error"] == message
    assert error["error_type"] == "value_error"


def test_benchmark_compare_keeps_public_missing_and_malformed_file_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"results": []}', encoding="utf-8")
    arguments = {"baseline": str(baseline), "candidate": str(candidate)}
    missing = _call_registered_tool("benchmark_compare", arguments)
    assert missing["isError"] is True
    error = json.loads(missing["content"][0]["text"])["result"]
    assert error["error"] == f"baseline not found: {baseline}"
    assert error["error_type"] == "file_not_found"

    baseline.write_text("{", encoding="utf-8")
    malformed = _call_registered_tool("benchmark_compare", arguments)
    assert malformed["isError"] is True
    error = json.loads(malformed["content"][0]["text"])["result"]
    assert error["error"] == f"Failed to parse JSON: {baseline}"
    assert error["error_type"] == "value_error"


def test_benchmark_overview_keeps_unexpected_processing_errors_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> None:
        raise ValueError("unexpected-processing-detail")

    monkeypatch.setattr(handlers, "get_engine", fail)
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    result = _call_registered_tool("benchmark_overview", {})

    assert result["isError"] is True
    assert "unexpected-processing-detail" not in json.dumps(result)
    error = json.loads(result["content"][0]["text"])["result"]
    assert error["error"] == "Tool execution failed."
    assert error["error_type"] == "ValueError"


def test_cluster_validator_accepts_package_only_staged_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    candidate = tmp_path / "staged"
    candidate.mkdir()
    (candidate / "manifest.json").write_text('{"run_id": "stage_fixture"}', encoding="utf-8")
    arguments: dict[str, Any] = {"published_root": str(candidate), "package_only": True}
    for item in ("report", "notes", "template", "runbook"):
        path = tmp_path / f"{item}.md"
        path.write_text("Canonical run: `stage_fixture`\n", encoding="utf-8")
        arguments[item] = str(path)

    result = _call_registered_tool("cluster_validate_field_report", arguments)
    payload = json.loads(result["content"][0]["text"])["result"]

    assert result["isError"] is True
    assert (
        "published manifest, report, and notes identify the same run (stage_fixture)"
        in payload["stdout"]
    )
    assert "Stale-artifact hygiene: not requested" in payload["stdout"]


def test_handle_message_rejects_non_object_without_crashing() -> None:
    response = asyncio.run(mcp_server.MCPServer().handle_message([]))  # type: ignore[arg-type]

    assert response == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32600, "message": "Invalid Request: message must be an object"},
    }


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "status", "arguments": []},
        },
    ],
)
def test_handle_message_rejects_invalid_mcp_params(message: dict[str, Any]) -> None:
    response = asyncio.run(mcp_server.MCPServer().handle_message(message))

    assert response is not None
    assert response["error"]["code"] == -32602


def test_stdio_emits_parse_error_without_echoing_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "malformed-secret-value"
    monkeypatch.setattr(sys, "stdin", io.StringIO(f'{{"secret":"{secret}"\n'))

    asyncio.run(mcp_server.MCPServer().run_stdio())

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert response["id"] is None
    assert response["error"] == {"code": -32700, "message": "Parse error"}
    assert secret not in captured.err


def test_mcp_client_cleans_up_when_initialize_is_rejected() -> None:
    rejecting_server = """
import json
import sys

request = json.loads(sys.stdin.readline())
print(json.dumps({
    "jsonrpc": "2.0",
    "id": request["id"],
    "error": {"code": -32000, "message": "initialization rejected"},
}), flush=True)
"""
    client = RobustMCPClient(
        [sys.executable, "-u", "-c", rejecting_server],
        timeout=1.0,
    )

    with pytest.raises(RuntimeError, match="MCP initialization failed"):
        client.start()

    assert client._running is False
    assert client._process is not None
    assert client._process.poll() is not None


def test_mcp_client_fails_fast_when_server_exits_before_initialize() -> None:
    client = RobustMCPClient(
        [sys.executable, "-c", "raise SystemExit(7)"],
        timeout=2.0,
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="MCP initialization failed"):
        client.start()

    assert time.monotonic() - started < 1.0
    assert client._running is False
