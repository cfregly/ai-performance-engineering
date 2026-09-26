from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cluster.fabric.diagnostics import (
    _default_runner,
    analyze_snapshot_pair,
    parse_cumulus_counter_tables,
    parse_numeric_fields,
)
from core.analysis.cross_layer_diagnosis import artifact_signals, correlate

CODE_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _synthetic_snapshot(
    timestamp: float, counters: list[dict[str, object]], phase: str
) -> dict[str, object]:
    return {
        "fixture_kind": "synthetic_parser_fixture",
        "schema_version": "fabric-diagnostics.v1",
        "artifact_type": "fabric_snapshot",
        "run_id": "synthetic-counter-run",
        "phase": phase,
        "captured_at": f"synthetic-{timestamp}",
        "captured_epoch_s": timestamp,
        "clock_domain": "synthetic_fixture_clock",
        "counters": counters,
    }


def test_parse_numeric_fields_preserves_zero_counter() -> None:
    parsed = parse_numeric_fields(
        "PFC Pause Frames: 0\nECN Marked Packets = 12\nQueue Occupancy: 4 packets\n"
    )
    assert parsed["pfc_pause_frames"]["value"] == 0
    assert parsed["ecn_marked_packets"]["value"] == 12
    assert parsed["queue_occupancy"]["value"] == 4


def test_parse_numeric_fields_rejects_duplicate_normalized_keys() -> None:
    with pytest.raises(ValueError, match="duplicate normalized field 'pfc_pause_frames'"):
        parse_numeric_fields("Priority 3\nPFC Pause Frames: 1\nPriority 4\nPFC Pause Frames: 2\n")


def test_parse_cumulus_counter_tables_qualifies_rows_and_directions() -> None:
    fixture = """
Ingress Buffer Statistics
=========================
priority-group  rx-frames  rx-buffer-discards
--------------  ---------  ------------------
0               10         0 Bytes
1               20         2 Bytes

Qos Port Statistics
===================
Counter             Receive  Transmit
------------------  -------  --------
ecn-marked-packets  n/a      7
pause-frames        3        4
"""

    fields = parse_cumulus_counter_tables(fixture)

    assert fields["priority_group_0_rx_frames"]["value"] == 10
    assert fields["priority_group_1_rx_buffer_discards"] == {
        "value": 2,
        "unit": "Bytes",
        "source_line": "1               20         2 Bytes",
    }
    assert fields["ecn_marked_packets_transmit"]["value"] == 7
    assert fields["pause_frames_receive"]["value"] == 3


def test_local_timeout_terminates_owned_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "term-observed"
    command = f"trap 'printf term > {marker}' TERM; sleep 30 & child=$!; wait $child"

    result = _default_runner(command, timeout=1)

    assert result["status"] == "error"
    assert result["timed_out"] is True
    assert result["termination"]["owned_process_group"] is True
    assert result["termination"]["term_sent"] is True
    assert marker.read_text() == "term"


def test_analyze_snapshot_pair_reports_rates_resets_and_signal_scope() -> None:
    before = _synthetic_snapshot(
        100.0,
        [
            {
                "id": "roce|leaf|swp1|pfc_pause_frames",
                "family": "roce",
                "host": None,
                "endpoint_alias_hash": hashlib.sha256(b"leaf").hexdigest()[:12],
                "scope": "swp1",
                "name": "pfc_pause_frames",
                "kind": "counter",
                "value": 0,
                "unit": None,
                "evidence": ["raw/synthetic-before.json"],
            },
            {
                "id": "infiniband|mgmt|mlx5_0@11:1|port_rcv_errors",
                "family": "infiniband",
                "host": "mgmt",
                "scope": "mlx5_0@11:1",
                "name": "port_rcv_errors",
                "kind": "counter",
                "value": 9,
                "unit": None,
                "evidence": ["raw/synthetic-before.json"],
            },
        ],
        "before",
    )
    after = _synthetic_snapshot(
        104.0,
        [
            {
                "id": "roce|leaf|swp1|pfc_pause_frames",
                "family": "roce",
                "host": None,
                "endpoint_alias_hash": hashlib.sha256(b"leaf").hexdigest()[:12],
                "scope": "swp1",
                "name": "pfc_pause_frames",
                "kind": "counter",
                "value": 8,
                "unit": None,
                "evidence": ["raw/synthetic-after.json"],
            },
            {
                "id": "infiniband|mgmt|mlx5_0@11:1|port_rcv_errors",
                "family": "infiniband",
                "host": "mgmt",
                "scope": "mlx5_0@11:1",
                "name": "port_rcv_errors",
                "kind": "counter",
                "value": 2,
                "unit": None,
                "evidence": ["raw/synthetic-after.json"],
            },
        ],
        "after",
    )

    result = analyze_snapshot_pair(before, after)

    assert result["status"] == "partial"
    pfc = next(row for row in result["counters"] if row["name"] == "pfc_pause_frames")
    assert pfc["delta"] == 8
    assert pfc["rate_per_second"] == 2
    assert pfc["start_unix_s"] == 100
    assert pfc["end_unix_s"] == 104
    reset = next(row for row in result["counters"] if row["name"] == "port_rcv_errors")
    assert reset["status"] == "reset_or_wrap"
    assert reset["delta"] is None
    signal = result["signals"][0]
    assert signal["kind"] == "pfc"
    assert signal["role"] == "counter"
    assert signal["semantics"] == "delta"
    assert signal["value"] == 8
    assert signal["unit"] == "count"
    assert signal["rate_per_second"] == 2
    assert signal["rate_unit"] == "count/s"
    assert signal["clock_domain"] == "synthetic_fixture_clock"
    leaf_hash = hashlib.sha256(b"leaf").hexdigest()[:12]
    assert signal["scope"] == {
        "endpoint_alias_hash": leaf_hash,
        "interface": "swp1",
    }
    assert "kernel_hostname_verified" not in signal["scope"]
    assert signal["scope_identity"]["kernel_hostname_verified"] is False
    assert signal["evidence"] == ["raw/synthetic-before.json", "raw/synthetic-after.json"]


def test_analyze_snapshot_pair_rejects_different_collector_clocks() -> None:
    before = _synthetic_snapshot(1.0, [], "before")
    after = _synthetic_snapshot(2.0, [], "after")
    after["clock_domain"] = "different_synthetic_clock"

    result = analyze_snapshot_pair(before, after)

    assert result["status"] == "invalid"
    assert result["signals"] == []
    assert "same collector clock domain" in result["reason"]


def test_analyze_snapshot_pair_keeps_uint64_delta_exact_and_rejects_nan() -> None:
    counter_id = "infiniband|mgmt|11:1|port_xmit_data"
    before = _synthetic_snapshot(
        1.0,
        [
            {
                "id": counter_id,
                "family": "infiniband",
                "host": "mgmt",
                "scope": "11:1",
                "name": "port_xmit_data",
                "kind": "counter",
                "value": 18_446_744_073_709_551_610,
                "unit": None,
            }
        ],
        "before",
    )
    after = _synthetic_snapshot(
        2.0,
        [
            {
                "id": counter_id,
                "family": "infiniband",
                "host": "mgmt",
                "scope": "11:1",
                "name": "port_xmit_data",
                "kind": "counter",
                "value": 18_446_744_073_709_551_614,
                "unit": None,
            }
        ],
        "after",
    )
    exact = analyze_snapshot_pair(before, after)
    assert exact["counters"][0]["delta"] == 4

    after["counters"][0]["value"] = float("nan")
    invalid = analyze_snapshot_pair(before, after)
    assert invalid["counters"][0]["status"] == "invalid_numeric"
    assert invalid["signals"] == []


def test_path_balance_reports_unequal_rates_without_causal_claim() -> None:
    before = _synthetic_snapshot(
        1.0,
        [
            {
                "id": "roce|leaf|swp1|tx_bytes",
                "family": "roce",
                "host": "leaf",
                "scope": "swp1",
                "name": "tx_bytes",
                "kind": "counter",
                "value": 100,
                "unit": "bytes",
            },
            {
                "id": "roce|leaf|swp2|tx_bytes",
                "family": "roce",
                "host": "leaf",
                "scope": "swp2",
                "name": "tx_bytes",
                "kind": "counter",
                "value": 100,
                "unit": "bytes",
            },
        ],
        "before",
    )
    after = _synthetic_snapshot(
        3.0,
        [
            {
                "id": "roce|leaf|swp1|tx_bytes",
                "family": "roce",
                "host": "leaf",
                "scope": "swp1",
                "name": "tx_bytes",
                "kind": "counter",
                "value": 300,
                "unit": "bytes",
            },
            {
                "id": "roce|leaf|swp2|tx_bytes",
                "family": "roce",
                "host": "leaf",
                "scope": "swp2",
                "name": "tx_bytes",
                "kind": "counter",
                "value": 500,
                "unit": "bytes",
            },
        ],
        "after",
    )

    result = analyze_snapshot_pair(before, after)

    assert result["path_balance"][0]["observation"] == "unequal_rates_observed"
    assert result["path_balance"][0]["resource_count"] == 2
    assert "do not prove an ECMP collision" in result["path_balance"][0]["interpretation"]


def test_gauge_signal_is_measurement_and_not_a_counter_association() -> None:
    counter_id = "roce|leaf|swp1|pfc_queue_occupancy"
    before = _synthetic_snapshot(
        1.0,
        [
            {
                "id": counter_id,
                "family": "roce",
                "host": "synthetic-host",
                "scope": "swp1",
                "name": "pfc_queue_occupancy",
                "kind": "gauge",
                "value": 1,
                "unit": "packets",
            }
        ],
        "before",
    )
    after = _synthetic_snapshot(
        2.0,
        [
            {
                "id": counter_id,
                "family": "roce",
                "host": "synthetic-host",
                "scope": "swp1",
                "name": "pfc_queue_occupancy",
                "kind": "gauge",
                "value": 5,
                "unit": "packets",
            }
        ],
        "after",
    )
    result = analyze_snapshot_pair(before, after)
    gauge = result["signals"][0]
    symptom = {
        "metric": "synthetic_latency",
        "value": 1,
        "unit": "ms",
        "start_unix_s": 1.0,
        "end_unix_s": 2.0,
        "clock_domain": "synthetic_fixture_clock",
        "scope": {"host": "synthetic-host"},
        "role": "symptom",
    }

    assert gauge["role"] == "measurement"
    assert "semantics" not in gauge
    assert gauge["value"] == 5
    assert correlate([gauge, symptom])["correlations"] == []


def test_counter_delta_composes_with_cross_layer_application_signal() -> None:
    counter_id = "roce|alias|swp1|pfc_pause_frames"
    before = _synthetic_snapshot(
        20.0,
        [
            {
                "id": counter_id,
                "family": "roce",
                "host": None,
                "endpoint_alias_hash": "synthetic-leaf",
                "scope": "swp1",
                "name": "pfc_pause_frames",
                "kind": "counter",
                "value": 0,
                "unit": "frames",
            }
        ],
        "before",
    )
    after = _synthetic_snapshot(
        24.0,
        [
            {
                "id": counter_id,
                "family": "roce",
                "host": None,
                "endpoint_alias_hash": "synthetic-leaf",
                "scope": "swp1",
                "name": "pfc_pause_frames",
                "kind": "counter",
                "value": 4,
                "unit": "frames",
            }
        ],
        "after",
    )
    delta_artifact = analyze_snapshot_pair(before, after)
    application_artifact = {
        "schema": "aisp.diagnostic-signals.v1",
        "run_id": "synthetic-counter-run",
        "signals": [
            {
                "metric": "application_step_latency_regression",
                "value": 5.0,
                "unit": "ms",
                "start_unix_s": 21.0,
                "end_unix_s": 23.0,
                "clock_domain": "synthetic_fixture_clock",
                "scope": {
                    "endpoint_alias_hash": "synthetic-leaf",
                    "interface": "swp1",
                    "workload_id": "synthetic-workload",
                },
                "role": "symptom",
            }
        ],
    }

    assert (
        delta_artifact["schema"] == application_artifact["schema"] == "aisp.diagnostic-signals.v1"
    )
    assert delta_artifact["run_id"] == application_artifact["run_id"]
    signals = artifact_signals(delta_artifact) + artifact_signals(application_artifact)
    report = correlate(signals)

    assert report["status"] == "ok"
    assert len(report["correlations"]) == 1
    assert report["correlations"][0]["counter"]["metric"] == "pfc_pause_frames"


def test_retained_snapshot_cli_writes_structured_report_and_manifest(tmp_path: Path) -> None:
    before = _synthetic_snapshot(
        10.0,
        [
            {
                "id": "roce|leaf|swp1|ecn_marked_packets",
                "family": "roce",
                "host": "leaf",
                "scope": "swp1",
                "name": "ecn_marked_packets",
                "kind": "counter",
                "value": 0,
                "unit": None,
                "evidence": ["raw/synthetic-before.json"],
            }
        ],
        "before",
    )
    after = _synthetic_snapshot(
        12.0,
        [
            {
                "id": "roce|leaf|swp1|ecn_marked_packets",
                "family": "roce",
                "host": "leaf",
                "scope": "swp1",
                "name": "ecn_marked_packets",
                "kind": "counter",
                "value": 0,
                "unit": None,
                "evidence": ["raw/synthetic-after.json"],
            }
        ],
        "after",
    )
    before_path = tmp_path / "synthetic-before.json"
    after_path = tmp_path / "synthetic-after.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "manifest.json",
        {"manifest_version": 2, "sentinel": "preserve-me", "tools": {"existing": {"status": "ok"}}},
    )
    _write_json(before_path, before)
    _write_json(after_path, after)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.diagnostics",
            "--run-id",
            "retained-test",
            "--run-dir",
            str(run_dir),
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--workload-id",
            "serving-profile-a",
            "--require-evidence",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(
        (run_dir / "structured" / "retained-test_fabric_diagnostics.json").read_text()
    )
    assert payload["collection_mode"] == "retained_snapshot_analysis"
    assert payload["delta"]["run_id"] == "retained-test"
    assert payload["delta"]["declared_workload_id"] == "serving-profile-a"
    assert payload["delta"]["signals"][0]["scope"]["workload_id"] == "serving-profile-a"
    assert payload["delta"]["signals"][0]["value"] == 0
    assert (run_dir / "reports" / "retained-test_fabric_diagnostics.md").is_file()
    assert (run_dir / "structured" / "retained-test_fabric_diagnostics_manifest.json").is_file()
    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert run_manifest["sentinel"] == "preserve-me"
    assert run_manifest["tools"]["existing"] == {"status": "ok"}
    assert run_manifest["tools"]["fabric_diagnostics"]["manifest_fragment"].endswith(
        "_fabric_diagnostics_manifest.json"
    )


def test_concurrent_manifest_merges_preserve_every_tool(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    kinds = [f"concurrent_tool_{index}" for index in range(6)]
    script = """
import sys
from pathlib import Path
from cluster.fabric.diagnostics import _write_manifest_fragment

run_dir = Path(sys.argv[1])
kind = sys.argv[2]
artifact = run_dir / f"{kind}.txt"
artifact.write_text(kind)
_write_manifest_fragment("concurrent-run", run_dir, [artifact], kind=kind)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(run_dir), kind],
            cwd=CODE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for kind in kinds
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(kinds).issubset(manifest["tools"])


def test_live_cli_records_real_unsupported_commands_without_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.diagnostics",
            "--run-id",
            "unsupported-test",
            "--run-dir",
            str(run_dir),
            "--family",
            "roce",
            "--cumulus-hosts",
            "localhost",
            "--switch-interfaces",
            "swp1",
            "--interval-seconds",
            "0",
            "--timeout-seconds",
            "5",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    payload = json.loads(
        (run_dir / "structured" / "unsupported-test_fabric_diagnostics.json").read_text()
    )
    assert payload["status"] == "unsupported"
    assert payload["before"]["commands"]
    assert all(record["status"] != "ok" for record in payload["before"]["commands"])
    assert list((run_dir / "raw").glob("*.json"))
