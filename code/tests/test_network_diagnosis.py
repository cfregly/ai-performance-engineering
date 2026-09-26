"""Exercise network calculations and retained capture analysis without hardware claims."""

import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from ch03.network_diagnosis_tool import SCHEMA, analyze, bdp_analysis, main, parse_packets, parse_ss
from core.diagnostics.evidence import capture, write_bundle


def command(stdout):
    return {"status": "ok", "stdout": stdout, "stderr": "", "start_unix_s": 1}


def test_bdp_units_and_window_ceiling():
    result = bdp_analysis(10, 200, 16 * 1024**2)
    assert result["bdp_bytes"] == 250_000_000
    assert result["window_limit_gbps"] == pytest.approx(0.67108864)
    for value in (-1, 0, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            bdp_analysis(value, 200)


def test_ss_preserves_each_socket_and_window_units():
    raw = "ESTAB 0 1024 192.0.2.1:34000 192.0.2.2:5201\n cubic rtt:20/2 mss:1448 cwnd:10 snd_wnd:16384 retrans:0/3 bytes_retrans:4096\n"
    row = parse_ss(raw)[0]
    assert row["congestion_window_bytes"] == 14480
    assert row["rtt_ms"] == 20
    assert row["retransmissions_total"] == 3


def test_route_and_neighbor_explain_gateway_without_inventing_switch_state():
    data = {"schema": SCHEMA, "target": "198.51.100.2", "commands": {
        "route": command(json.dumps([{"dev": "eth0", "gateway": "192.0.2.254", "prefsrc": "192.0.2.1"}])),
        "neighbors": command(json.dumps([{"dst": "192.0.2.254", "dev": "eth0", "state": ["REACHABLE"], "lladdr": "02:00:00:00:00:01"}]))}}
    report = analyze(data)
    assert report["status"] == "ok"
    assert report["packet_paths"][0]["next_hop"] == "192.0.2.254"
    assert report["packet_paths"][0]["destination"] == "198.51.100.2"
    assert not report["findings"]


def test_mtu_timeout_is_inconclusive_not_mismatch():
    data = {"schema": SCHEMA, "commands": {"mtu_9000": {"status": "timeout", "stdout": "", "stderr": ""}}}
    report = analyze(data)
    assert report["mtu_probes"][0]["status"] == "inconclusive"
    assert report["status"] == "unavailable"


def test_packet_export_preserves_reset_provenance_and_mtu_signal():
    fields = "frame.time_epoch\tip.src\tip.dst\ttcp.flags.syn\ttcp.flags.ack\ttcp.flags.reset\ticmp.type\ticmp.code\ticmp.mtu\n"
    fields += "1\t192.0.2.1\t192.0.2.2\t1\t0\t0\t\t\t\n"
    fields += "2\t192.0.2.2\t192.0.2.1\t0\t1\t1\t\t\t\n"
    fields += "3\t192.0.2.254\t192.0.2.1\t\t\t\t3\t4\t1400\n"
    result = parse_packets(fields)
    assert result["syn"] == 1
    assert result["resets"][0]["source"] == "192.0.2.2"
    assert result["packet_too_big"][0]["advertised_mtu"] == "1400"
    with pytest.raises(ValueError):
        parse_packets("wrong\theader\n1\t2\n")


def test_iperf_receiver_rate_and_sampling(tmp_path):
    raw = {"schema": SCHEMA, "target": "192.0.2.2", "network_plane": "management", "commands": {
        "iperf_1_forward": command(json.dumps({"end": {"sum_received": {"bits_per_second": 1e9}, "sum_sent": {"retransmits": 2}}})),
        "socket_sample_iperf_1_forward_0": command("ESTAB 0 0 192.0.2.1:4000 192.0.2.2:5201\n rtt:2/1 mss:1448 cwnd:20\n")}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(raw))
    out = tmp_path / "analysis"
    process = subprocess.run([sys.executable, "-m", "ch03.network_diagnosis_tool", "analyze", "--input", str(source), "--run-dir", str(out)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((out / "structured/network-diagnose.json").read_text())
    assert report["throughput"][0]["receiver_gbps"] == 1
    assert report["throughput"][0]["network_plane"] == "management"
    assert report["sockets"][0]["sample_unix_s"] == 1
    assert report["valid_for_performance_claim"] is False
    assert json.loads((out / "manifest.json").read_text())["tools"]["network-diagnose"]["artifacts"]
    assert main(["analyze", "--input", str(source), "--run-dir", str(out)]) == 1


def test_packet_walk_counts_neighbor_resolution_and_echo():
    fields = "frame.time_epoch\ttcp.flags.syn\ttcp.flags.reset\tarp.opcode\tarp.src.proto_ipv4\tarp.dst.proto_ipv4\tarp.src.hw_mac\teth.dst\ticmp.type\ticmpv6.type\n"
    fields += "1\t\t\t1\t192.0.2.1\t192.0.2.2\t02:00:00:00:00:01\tff:ff:ff:ff:ff:ff\t\t\n"
    fields += "2\t\t\t2\t192.0.2.2\t192.0.2.1\t02:00:00:00:00:02\t02:00:00:00:00:01\t\t\n"
    fields += "3\t\t\t\t\t\t\t\t8\t\n"
    fields += "4\t\t\t\t\t\t\t\t0\t\n"
    report = parse_packets(fields)
    assert report["arp_requests"] == report["arp_replies"] == 1
    assert report["echo_requests"] == report["echo_replies"] == 1
    assert report["arp_events"][0]["ethernet_destination"] == "ff:ff:ff:ff:ff:ff"


def test_network_export_joins_cross_layer_without_inventing_capture_identity():
    from core.analysis.cross_layer_diagnosis import artifact_signals, correlate
    snapshot = {"schema": SCHEMA, "target": "192.0.2.2", "commands": {
        "ping": {**command("rtt min/avg/max/mdev = 1/2/3/0.5 ms"), "duration_s": 2}}}
    assert not analyze(snapshot)["signals"]
    snapshot["collector"] = {"host": "test-host", "clock_domain": "test-clock"}
    signals = artifact_signals(analyze(snapshot))
    symptom = {**signals[0], "metric": "late_request", "role": "symptom", "value": 1, "unit": "count"}
    report = correlate([*signals, symptom])
    assert len(report["contextual_measurements"]) == 1
    assert not report["correlations"]


def test_capture_real_success_failure_and_timeout():
    assert capture([sys.executable, "-c", "print('actual subprocess')"])["stdout"].strip() == "actual subprocess"
    assert capture([sys.executable, "-c", "raise SystemExit(3)"])["returncode"] == 3
    assert capture([sys.executable, "-c", "import time; time.sleep(1)"], timeout=0.01)["status"] == "timeout"
    assert capture(["/no/such/aisp-test-executable"])["status"] == "unavailable"


def test_no_measurements_cannot_pass_and_malformed_data_fails():
    assert analyze({"schema": SCHEMA, "commands": {}})["status"] == "unavailable"
    with pytest.raises(ValueError):
        analyze({"schema": SCHEMA, "commands": {"route": command("not json")}})
    assert analyze({"schema": SCHEMA, "packet_fields": "frame.time_epoch\ttcp.flags.syn\ttcp.flags.reset\n"})["status"] == "unavailable"


def test_malformed_capture_cli_returns_clear_error(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"schema": SCHEMA, "commands": []}))
    result = subprocess.run([sys.executable, "-m", "ch03.network_diagnosis_tool", "analyze",
                             "--input", str(source), "--run-dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.stderr


def test_bundle_retains_multiple_tools(tmp_path):
    for name in ("first", "second"):
        write_bundle(tmp_path, name, {}, {"status": "unavailable", "findings": []})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["tools"]) == {"first", "second"}


@pytest.mark.parametrize("same_tool", [False, True])
def test_concurrent_bundle_processes_keep_manifest_and_refuse_overwrite(tmp_path, same_tool):
    script = "from pathlib import Path; import sys; from core.diagnostics.evidence import write_bundle; write_bundle(Path(sys.argv[1]), sys.argv[2], {}, {'status': 'unavailable'})"
    names = ["same" if same_tool else f"tool-{index}" for index in range(6)]
    def launch(name):
        return subprocess.run([sys.executable, "-c", script, str(tmp_path), name], capture_output=True, text=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(launch, names))
    assert sum(result.returncode == 0 for result in results) == (1 if same_tool else 6)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["tools"]) == set(names)
    for name in set(names):
        assert json.loads((tmp_path / "structured" / f"{name}.json").read_text())["status"] == "unavailable"
