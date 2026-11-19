"""Run the operator replay MCP entrypoint through the actual repository CLI."""

import json
import os

import mcp.mcp_server as server
import pytest


OWNED_PROCESS_SUPPORTED = all(
    hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")
)


@pytest.mark.skipif(
    not OWNED_PROCESS_SUPPORTED,
    reason="owned subprocess cleanup requires waitid with WNOWAIT",
)
def test_mcp_replay_help_runs_real_cli():
    result = server.tool_tools_trace_replay({"args": ["--help"], "include_context": False})
    assert result["returncode"] == 0, result
    assert "capture" in result["stdout"] and "replay" in result["stdout"]
    assert result["timeout_seconds"] is None


def test_mcp_replay_rejects_invalid_arguments():
    result = server.tool_tools_trace_replay({"args": "capture", "include_context": False})
    assert result["success"] is False
    assert "list of strings" in result["error"]


@pytest.mark.skipif(
    not OWNED_PROCESS_SUPPORTED,
    reason="owned subprocess cleanup requires waitid with WNOWAIT",
)
def test_mcp_replay_captures_and_validates_real_cpu_inputs(tmp_path):
    from labs.trace_replay.schema import PLAN_SCHEMA, load_bundle

    plan = tmp_path / "capture-plan.json"
    bundle = tmp_path / "captured"
    plan.write_text(json.dumps({
        "schema_version": PLAN_SCHEMA,
        "events": [{
            "event_id": "softmax",
            "operation": "online_softmax",
            "parameters": {"rows": 2, "cols": 17},
            "seed": 314,
            "source": {"kind": "generated_by_recipe", "description": "CPU reference integration fixture"},
        }],
    }))
    result = server.tool_tools_trace_replay({
        "args": ["capture", "--plan", str(plan), "--output-dir", str(bundle), "--device", "cpu"],
        "include_context": False,
    })
    assert result["returncode"] == 0, result
    captured = load_bundle(bundle)
    assert captured.events[0]["status"] == "completed"
    assert captured.events[0]["source"]["kind"] == "generated_by_recipe"
    validated = server.tool_tools_trace_replay({
        "args": ["validate", "--bundle", str(bundle)], "include_context": False,
    })
    assert validated["returncode"] == 0, validated
