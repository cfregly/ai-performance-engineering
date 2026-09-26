from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from cluster.fabric.transport import (
    TransportConfig,
    _host_gdr_comparison,
    _measurement_signal,
    parse_iperf_output,
    parse_local_p2p_output,
    parse_nccl_output,
    parse_perftest_output,
)
from core.analysis.cross_layer_diagnosis import artifact_signals

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_transport_parser_fixtures_extract_metrics() -> None:
    synthetic_perftest_fixture = """
    #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
    8388608    100             190.20             188.75               0.0028
    """
    synthetic_nccl_fixture = """
    NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:2/RoCE
    # size count type redop root time algbw busbw errors
    8388608 2097152 float sum -1 82.30 101.93 191.12 0
    """
    synthetic_iperf_fixture = json.dumps(
        {
            "fixture_kind": "synthetic_parser_fixture",
            "end": {"sum_received": {"bits_per_second": 9_500_000_000, "seconds": 10}},
        }
    )

    perftest = parse_perftest_output(synthetic_perftest_fixture)
    perftest_json = parse_perftest_output(
        json.dumps(
            {
                "test_info": {"size": 8_388_608, "iterations": 100},
                "perftest_results": {"BW_average[Gb/sec]": 188.75},
            }
        )
    )
    nccl = parse_nccl_output(synthetic_nccl_fixture)
    iperf = parse_iperf_output(synthetic_iperf_fixture)

    assert perftest is not None and perftest["average_gbps"] == 188.75
    assert perftest_json is not None and perftest_json["message_bytes"] == 8_388_608
    assert nccl is not None and nccl["peak_busbw_gbps"] == 191.12
    assert nccl["runtime_transport_evidence"] == {
        "hcas": ["mlx5_0", "mlx5_1"],
        "ib_ports": [1, 2],
    }
    assert iperf is not None and iperf["gbps"] == 9.5


def test_synthetic_local_p2p_parser_retains_topology_and_peer_access() -> None:
    fixture = """
__AISP_TOPOLOGY_MATRIX_BEGIN__
        GPU0 GPU1 CPU Affinity
GPU0   X    NV18 0-3
GPU1   NV18 X    0-3
__AISP_TOPOLOGY_MATRIX_END__
__AISP_P2P_ACCESS_BEGIN__
        GPU0 GPU1
GPU0   X    OK
GPU1   OK   X
__AISP_P2P_ACCESS_END__
GPU 0 → GPU 1: 812.50 GB/s
"""

    parsed = parse_local_p2p_output(fixture)

    assert parsed is not None
    assert parsed["mean_bandwidth_gbps"] == 812.5
    assert parsed["unit"] == "GB/s"
    assert parsed["topology_links"]["0-1"] == "NV18"
    assert parsed["peer_read_access"]["0-1"] is True


def test_native_nvidia_smi_underlined_matrix_headers() -> None:
    capture = (
        "__AISP_TOPOLOGY_MATRIX_BEGIN__\n"
        "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
        "GPU0\t X \tNV18\t0-59\t0\t\tN/A\n"
        "GPU1\tNV18\t X \t0-59\t0\t\tN/A\n"
        "__AISP_TOPOLOGY_MATRIX_END__\n"
        "__AISP_P2P_ACCESS_BEGIN__\n"
        " \t\x1b[4mGPU0\tGPU1\t\x1b[0m\n"
        " GPU0\tX\tOK\t\n"
        " GPU1\tOK\tX\t\n"
        "__AISP_P2P_ACCESS_END__\n"
        "GPU 0 → GPU 1: 450.65 GB/s\n"
    )
    parsed = parse_local_p2p_output(capture)
    assert parsed is not None
    assert parsed["topology_links"] == {"0-1": "NV18", "1-0": "NV18"}
    assert parsed["peer_read_access"] == {"0-1": True, "1-0": True}


def test_synthetic_transport_measurement_signal_composes_with_cross_layer(tmp_path: Path) -> None:
    config = TransportConfig(
        run_id="synthetic-transport-run",
        run_dir=tmp_path,
        cases=("host_rdma",),
        server_address="synthetic-peer",
        hca="synthetic-hca",
        rail="synthetic-rail",
        ib_port=1,
        host_rdma_control_port=18515,
        gdr_control_port=18516,
        payload_bytes=8_388_608,
        iterations=100,
        queue_pairs=1,
        direction="write",
        gpu_id=None,
        cuda_mem_type=None,
        use_dmabuf=False,
        workload_id="serving-profile-a",
        local_p2p_command=None,
        nccl_command=None,
        iperf_seconds=10,
        iperf_parallel=1,
        timeout_seconds=30,
    )
    signals = _measurement_signal(
        config,
        "host_rdma",
        "synthetic command",
        {"average_gbps": 100.0, "unit": "Gb/s"},
        start_unix_s=10.0,
        end_unix_s=12.0,
        evidence="raw/synthetic.json",
    )

    normalized = artifact_signals(
        {
            "schema": "aisp.diagnostic-signals.v1",
            "run_id": "synthetic-transport-run",
            "signals": signals,
        }
    )
    assert normalized[0]["role"] == "measurement"
    assert normalized[0]["unit"] == "Gb/s"
    assert normalized[0]["scope"]["workload_id"] == "serving-profile-a"
    assert normalized[0]["scope"]["path"].startswith("configured:rdma:")


def test_synthetic_parsers_reject_nonfinite_and_unrecognized_rows() -> None:
    assert parse_perftest_output("warning counters 1 2 3 4") is None
    assert parse_nccl_output("1 2 3 4 5 6 7 8") is None
    assert parse_perftest_output(json.dumps({"results": {"BW_average": float("nan")}})) is None
    assert (
        parse_nccl_output(
            json.dumps(
                {
                    "results": [
                        {
                            "size_bytes": 8,
                            "time_us": 1,
                            "algbw_gbps": float("nan"),
                            "busbw_gbps": 1,
                        }
                    ]
                }
            )
        )
        is None
    )
    assert (
        parse_iperf_output(json.dumps({"end": {"sum_received": {"bits_per_second": float("inf")}}}))
        is None
    )


def test_synthetic_host_gdr_ratio_is_descriptive_only() -> None:
    identity = {
        "peer": "synthetic-peer",
        "hca": "synthetic-hca",
        "rail": "synthetic-rail",
        "ib_port": 1,
        "payload_bytes": 8_388_608,
        "direction": "write",
        "iterations": 100,
        "queue_pairs": 1,
    }
    comparison = _host_gdr_comparison(
        [
            {
                "case": "host_rdma",
                "status": "ok",
                "declared_identity": identity,
                "metric": {"message_bytes": 8_388_608, "average_gbps": 100.0},
            },
            {
                "case": "gdr",
                "status": "ok",
                "declared_identity": identity,
                "metric": {"message_bytes": 8_388_608, "average_gbps": 80.0},
            },
        ]
    )

    assert comparison["status"] == "descriptive"
    assert comparison["gdr_to_host_rate_ratio"] == 0.8
    assert comparison["causal_interpretation_eligible"] is False
    assert comparison["path_verification"] == "unverified"
    assert comparison["correctness_verification"] == "not_collected"


def test_cli_marks_missing_case_inputs_unsupported(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.transport",
            "--run-id",
            "missing-inputs",
            "--run-dir",
            str(run_dir),
            "--cases",
            "local_p2p,host_rdma,gdr,nccl,iperf",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    payload = json.loads(
        (run_dir / "structured" / "missing-inputs_transport_diagnostics.json").read_text()
    )
    assert payload["status"] == "unsupported"
    assert all(case["status"] == "unsupported" for case in payload["cases"])
    assert all(case["metric"] is None for case in payload["cases"])
    assert (run_dir / "reports" / "missing-inputs_transport_diagnostics.md").is_file()
    assert (run_dir / "structured" / "missing-inputs_transport_diagnostics_manifest.json").is_file()


def test_real_subprocess_with_unparseable_nccl_output_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    script = "print('not a benchmark metric')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.transport",
            "--run-id",
            "invalid-output",
            "--run-dir",
            str(run_dir),
            "--cases",
            "nccl",
            "--nccl-command",
            command,
            "--timeout-seconds",
            "10",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(
        (run_dir / "structured" / "invalid-output_transport_diagnostics.json").read_text()
    )
    assert payload["status"] == "error"
    assert payload["cases"][0]["status"] == "invalid"
    assert payload["cases"][0]["metric"] is None
    raw = json.loads((run_dir / "raw" / "invalid-output_transport_nccl.json").read_text())
    assert "not a benchmark metric" in raw["stdout"]


def test_real_subprocess_nccl_payload_mismatch_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    script = (
        "print('# synthetic parser fixture\\n'"
        "      '# size count type redop root time algbw busbw errors\\n'"
        "      '4194304 1048576 float sum -1 100.0 41.94 78.0 0')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.transport",
            "--run-id",
            "payload-mismatch",
            "--run-dir",
            str(run_dir),
            "--cases",
            "nccl",
            "--payload-bytes",
            "8388608",
            "--nccl-command",
            command,
            "--timeout-seconds",
            "10",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(
        (run_dir / "structured" / "payload-mismatch_transport_diagnostics.json").read_text()
    )
    case = payload["cases"][0]
    assert case["status"] == "invalid"
    assert case["dimension_validation"]["payload"] == "mismatch"
    assert case["rate_comparison_eligible"] is False
    assert case["matched_path_eligible"] is False


def test_real_missing_perftest_never_reports_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cluster.fabric.transport",
            "--run-id",
            "real-perftest-probe",
            "--run-dir",
            str(run_dir),
            "--cases",
            "host_rdma",
            "--server-address",
            "127.0.0.1",
            "--hca",
            "definitely_not_an_hca",
            "--timeout-seconds",
            "5",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(
        (run_dir / "structured" / "real-perftest-probe_transport_diagnostics.json").read_text()
    )
    case = payload["cases"][0]
    assert case["status"] in {"unsupported", "error"}
    assert proc.returncode == (2 if case["status"] == "unsupported" else 1)
    assert case["metric"] is None
