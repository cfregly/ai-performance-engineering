"""Focused regressions for dashboard and MCP transport boundaries."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from typing import Any

import pytest

from core.api.redaction import redact_sensitive_data
from core.api.response import build_response
from dashboard.api import server as dashboard_server
from mcp import mcp_server
from mcp.mcp_client import RobustMCPClient
from tests.http_client import asgi_request


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_gpu_stream_rejects_unsafe_intervals(interval: str) -> None:
    pytest.importorskip("fastapi")

    response = asgi_request(
        dashboard_server.fastapi_app,
        "GET",
        "/api/gpu/stream",
        params={"interval": interval, "max_events": "1"},
    )

    assert response.status_code == 422
    assert "interval must be finite and at least" in response.json()["detail"]


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


def test_http_envelope_redacts_credentials_from_arguments_and_results() -> None:
    secret = "short"
    arguments = {
        "nmx_token": secret,
        "max_tokens": 64,
    }
    result = {
        "success": True,
        "command": ["cluster-eval", "--nmx-token", secret],
        "message": f"used credential {secret}",
        "shell": f"cluster-eval --nmx-token='{secret}'",
        "tokens_per_second": 123,
    }

    payload = build_response("cluster.common_eval", arguments, result, 1)
    serialized = json.dumps(payload)

    assert secret not in serialized
    assert payload["arguments"] == {"nmx_token": "<redacted>", "max_tokens": 64}
    assert payload["result"] == {
        "success": True,
        "command": ["cluster-eval", "--nmx-token", "<redacted>"],
        "message": "used credential <redacted>",
        "shell": "cluster-eval --nmx-token='<redacted>'",
        "tokens_per_second": 123,
    }


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
) -> None:
    secret = "exception-secret-value"

    def fail(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(secret)

    monkeypatch.setitem(mcp_server.HANDLERS, "review_exception", fail)

    result = mcp_server.MCPServer().call_tool("review_exception", {})
    payload = json.loads(result.content[0]["text"])
    serialized = json.dumps(payload)

    assert secret not in serialized
    assert "traceback" not in serialized.lower()
    assert payload["result"] == {
        "error": "Tool execution failed.",
        "error_type": "RuntimeError",
        "success": False,
    }


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
