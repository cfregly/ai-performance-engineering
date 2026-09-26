"""Run the diagnostic CLI and MCP paths and preserve fabric evidence boundaries."""

import json
import subprocess
import sys

import pytest

from cluster.fabric.evaluator import load_timed_diagnostic_evidence
from core.tools.tools_commands import DIAGNOSTIC_TOOLS, TOOLS


def test_network_calculation_through_real_cli(tmp_path):
    process = subprocess.run([sys.executable, "-m", "cli.aisp", "tools", "network-diagnose", "--", "bdp",
                              "--bandwidth-gbps", "10", "--rtt-ms", "200", "--run-dir", str(tmp_path)],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path / "structured/network-diagnose.json").read_text())
    assert report["bdp"]["bdp_bytes"] == 250_000_000


def test_registered_modules_exist_and_real_help_works():
    for name in DIAGNOSTIC_TOOLS:
        assert TOOLS[name].script_path.exists(), name
        process = subprocess.run([sys.executable, "-m", "cli.aisp", "tools", name, "--", "--help"],
                                 capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, (name, process.stderr)
        assert "usage:" in process.stdout.lower(), name


def test_mcp_dispatch_and_validation(tmp_path):
    from mcp.mcp_server import TOOLS as MCP_TOOLS
    from mcp.mcp_server import tool_tools_diagnostics
    assert set(MCP_TOOLS["tools_diagnostics"].input_schema["properties"]["tool"]["enum"]) == set(DIAGNOSTIC_TOOLS)
    result = tool_tools_diagnostics({"tool": "network-diagnose", "args": ["bdp", "--bandwidth-gbps", "10",
             "--rtt-ms", "200", "--run-dir", str(tmp_path)]})
    assert result["returncode"] == 0, result
    assert (tmp_path / "structured/network-diagnose.json").exists()
    assert "error" in tool_tools_diagnostics({"tool": "not-a-tool"})
    assert "error" in tool_tools_diagnostics({"tool": "network-diagnose", "timeout_seconds": 0})


def test_fabric_eval_attaches_only_matching_run_and_keeps_unknown_unknown(tmp_path):
    report = load_timed_diagnostic_evidence("example", tmp_path)
    assert report["status"] == "unavailable"
    assert report["affects_canonical_completeness"] is False
    structured = tmp_path / "structured"
    structured.mkdir()
    signal = {"metric": "pause_frames", "role": "counter", "semantics": "delta", "value": 2,
              "unit": "frames", "start_unix_s": 1, "end_unix_s": 2,
              "clock_domain": "collector-a", "scope": {"path": "rail-a"}}
    path = structured / "example_fabric_counter_deltas.json"
    path.write_text(json.dumps({"run_id": "example", "signals": [signal]}))
    report = load_timed_diagnostic_evidence("example", tmp_path)
    assert report["status"] == "partial"
    assert not report["correlations"]
    path.write_text(json.dumps({"run_id": "example", "status": "error", "signals": [signal]}))
    with pytest.raises(ValueError, match="Failed"):
        load_timed_diagnostic_evidence("example", tmp_path)
    path.write_text(json.dumps({"run_id": "other", "signals": [signal]}))
    with pytest.raises(ValueError, match="run ID"):
        load_timed_diagnostic_evidence("example", tmp_path)
