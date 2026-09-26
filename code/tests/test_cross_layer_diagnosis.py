"""Time and scope must align before a diagnostic association is reported."""

import json
import subprocess
import sys

import pytest

from core.analysis.cross_layer_diagnosis import SCHEMA, artifact_signals, correlate


def row(role, **kwargs):
    return {"metric": "slow_request" if role == "symptom" else "pfc_pause_frames",
            "role": role, "semantics": "delta", "unit": "count", "value": 1,
            "start_unix_s": 100, "end_unix_s": 102, "clock_domain": "capture-a",
            "scope": {"path": "rail-0", "workload_id": "example-run"}, **kwargs}


def test_association_requires_positive_delta_and_overlap():
    result = correlate([row("symptom"), row("counter"), row("counter", value=0)])
    assert len(result["correlations"]) == 1
    assert result["correlations"][0]["overlap_lower_bound_s"] == 2
    assert result["valid_for_performance_claim"] is False


def test_transport_measurements_are_context_not_congestion_claims():
    result = correlate([row("symptom"), row("measurement", metric="host_rdma_rate", unit="Gbps", value=40)])
    assert not result["correlations"]
    assert len(result["contextual_measurements"]) == 1
    assert result["contextual_measurements"][0]["measurement"]["value"] == 40
    assert not correlate([row("symptom"), row("measurement", clock_domain="other")])["contextual_measurements"]


def test_clocks_and_scope_do_not_silently_match():
    assert not correlate([row("symptom"), row("counter", clock_domain="capture-b")])["correlations"]
    result = correlate([row("symptom"), row("counter", clock_domain="capture-b")], max_clock_skew_s=0.1)
    assert result["correlations"][0]["overlap_lower_bound_s"] == pytest.approx(1.9)
    assert not correlate([row("symptom"), row("counter", scope={"path": "other-rail"})])["correlations"]
    assert not correlate([row("symptom"), row("counter", start_unix_s=102, end_unix_s=104)])["correlations"]
    assert not correlate([row("symptom", scope={"interface": "eth0"}),
                          row("counter", scope={"interface": "eth0"})])["correlations"]


def test_unknown_units_and_cumulative_counters_are_rejected():
    with pytest.raises(ValueError):
        correlate([row("counter", semantics="cumulative")])
    with pytest.raises(ValueError):
        correlate([row("symptom", end_unix_s=99)])
    with pytest.raises(ValueError):
        correlate([row("symptom", value=float("nan"))])
    with pytest.raises(ValueError):
        artifact_signals({"status": "healthy"})
    assert correlate([])["status"] == "unavailable"
    with pytest.raises(ValueError, match="Failed"):
        artifact_signals({"signals": [row("counter")], "status": "rejected"})


def test_malformed_signal_list_cli_returns_clear_error(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"schema": SCHEMA, "signals": None}))
    result = subprocess.run([sys.executable, "-m", "core.analysis.cross_layer_diagnosis",
                             "--input", str(source), "--run-dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.stderr


def test_real_cli_retains_both_source_artifacts(tmp_path):
    inputs = []
    for role in ("symptom", "counter"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"schema": SCHEMA, "signals": [row(role)]}))
        inputs.extend(["--input", str(path)])
    out = tmp_path / "out"
    process = subprocess.run([sys.executable, "-m", "core.analysis.cross_layer_diagnosis", *inputs, "--run-dir", str(out)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((out / "structured/cross-layer-diagnose.json").read_text())
    assert len(report["signals"]) == 2
    assert len({s["source_sha256"] for s in report["signals"]}) == 2
    assert report["status"] == "ok"
