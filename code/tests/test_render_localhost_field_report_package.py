from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def _load_renderer_module():
    script_path = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "render_localhost_field_report_package.py"
    spec = importlib.util.spec_from_file_location("render_localhost_field_report_package", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_minimal_cluster_root(cluster_root: Path, run_id: str, label: str) -> None:
    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    figures = run_dir / "figures"
    structured.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", {"run_id": run_id})
    _write_json(
        structured / f"{run_id}_suite_steps.json",
        [
            {"name": "preflight_services", "exit_code": 0, "start_time": "2026-03-05T21:08:24+00:00"},
            {"name": "discovery", "exit_code": 0, "start_time": "2026-03-05T21:08:25+00:00"},
            {"name": "hang_triage_bundle", "exit_code": 0, "start_time": "2026-03-05T21:08:28+00:00"},
            {"name": "connectivity_probe", "exit_code": 0, "start_time": "2026-03-05T21:08:29+00:00"},
            {"name": "nccl_env_sensitivity", "exit_code": 0, "start_time": "2026-03-05T21:08:47+00:00"},
            {"name": "vllm_serve_sweep", "exit_code": 0, "start_time": "2026-03-05T21:09:53+00:00"},
            {"name": "validate_required_artifacts", "exit_code": 0, "start_time": "2026-03-05T21:17:25+00:00"},
            {"name": "manifest_refresh", "exit_code": 0, "start_time": "2026-03-05T21:17:26+00:00"},
        ],
    )
    _write_json(
        structured / f"{run_id}_{label}_meta.json",
        {"commands": {"nvidia_smi_l": {"stdout": "GPU 0: Test GPU"}}},
    )
    _write_json(structured / f"{run_id}_{label}_hang_triage_readiness.json", {"status": "ok"})
    _write_json(
        structured / f"{run_id}_torchrun_connectivity_probe.json",
        {
            "status": "ok",
            "world_size": 1,
            "ranks": [{"barrier_ms": [0.08, 0.07], "payload_probe": {"algbw_gbps": 120.288}}],
        },
    )
    _write_json(
        structured / f"{run_id}_nccl_env_sensitivity.json",
        {"status": "ok", "failure_count": 0, "baseline_peak_busbw_gbps": 0.0},
    )
    _write_json(
        structured / f"{run_id}_node1_nccl.json",
        {"results": [{"algbw_gbps": 2246.8, "size_bytes": 67108864}]},
    )
    _write_json(structured / f"{run_id}_preflight_services.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_meta_nvlink_topology.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_node_parity_summary.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_fio.json", {"status": "ok"})

    vllm_csv = structured / f"{run_id}_{label}_vllm_serve_sweep.csv"
    vllm_csv.write_text(
        "\n".join(
            [
                "concurrency,total_token_throughput,mean_ttft_ms,p99_ttft_ms,p99_tpot_ms",
                "1,405.615,72.198,88.385,9.128",
                "2,921.609,31.225,35.526,6.354",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (structured / f"{run_id}_{label}_vllm_serve_sweep.jsonl").write_text("{}\n", encoding="utf-8")


def test_render_localhost_report_handles_missing_operator_checks(tmp_path: Path) -> None:
    module = _load_renderer_module()
    run_id = "2026-03-05_localhost_common_eval_r1"
    label = "localhost"
    cluster_root = tmp_path / "cluster"
    _seed_minimal_cluster_root(cluster_root, run_id, label)

    args = Namespace(run_id=run_id, label=label, root=cluster_root, run_dir=cluster_root / "runs" / run_id)

    report = module.render_report(args)
    notes = module.render_notes(args)

    assert "Operator checks | not run in this preset" in report
    assert "operator-friction and monitoring artifacts were not requested in this preset run." in report
    assert "Example localhost fabric eval" in report
    assert "--run-id <new_run_id>" in report
    assert "Operator checks are optional and skipped in this preset" in notes
    assert "quick_friction | not run" in notes


def test_render_localhost_report_labels_only_recorded_monitoring_categories(tmp_path: Path) -> None:
    module = _load_renderer_module()
    run_id = "2026-03-16_localhost_monitoring"
    label = "localhost"
    cluster_root = tmp_path / "cluster"
    _seed_minimal_cluster_root(cluster_root, run_id, label)
    run_dir = cluster_root / "runs" / run_id
    _write_json(
        run_dir / "structured" / f"{run_id}_{label}_monitoring_expectations.json",
        {
            "status": "ok",
            "categories": {
                "gpu_telemetry": {"status": "ok", "checks": [{"name": "nvidia_nvlink", "status": "ok"}]},
                "system_signals": {"status": "not_requested", "checks": []},
            },
        },
    )
    args = Namespace(run_id=run_id, label=label, root=cluster_root, run_dir=run_dir)
    report = module.render_report(args)

    assert "| Monitoring mode | Recorded categories: gpu_telemetry |" in report
    assert "system_signals=not_requested" in report


def test_render_localhost_report_includes_fabric_section_when_present(tmp_path: Path) -> None:
    module = _load_renderer_module()
    run_id = "2026-03-16_localhost_fabric"
    label = "localhost"
    cluster_root = tmp_path / "cluster"
    _seed_minimal_cluster_root(cluster_root, run_id, label)

    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    _write_json(structured / f"{run_id}_fabric_command_catalog.json", {"entries": [{"id": "command_1"}]})
    _write_json(
        structured / f"{run_id}_fabric_capability_matrix.json",
        {
            "families": {
                "nvlink": {"present": True},
                "infiniband": {"present": True, "hcas": ["mlx5_0", "mlx5_1"]},
                "spectrum-x": {"present": True},
            }
        },
    )
    _write_json(
        structured / f"{run_id}_fabric_verification.json",
        {
            "families": {
                "nvlink": {
                    "completeness": "runtime_verified",
                    "control_plane": {
                        "nmx": {
                            "topology": {
                                "chassis_count": 1,
                                "chassis_serial_numbers": ["SN12345678901"],
                                "compute_node_count": 4,
                                "gpu_count": 8,
                                "switch_asic_count": 4,
                                "switch_tray_count": 2,
                                "port_count": 64,
                                "ports": {
                                    "gpu_facing_ports": 32,
                                    "switch_facing_ports": 32,
                                    "gpu_facing_ports_have_base_lid": True,
                                    "switch_facing_ports_have_base_lid": False,
                                    "expected_formula": "2 * gpu_count(8) * switch_asic_count(4) = 64",
                                    "matches_expected_formula": True,
                                },
                                "sample_compute_node": {
                                    "id": "node-1",
                                    "name": "node-1",
                                    "system_uid": "sys-node-1",
                                    "gpu_id_count": 2,
                                    "location": {"chassis_id": 1, "slot_id": 1, "host_id": 1, "tray_index": 0},
                                },
                                "sample_gpu": {
                                    "device_id": 1,
                                    "domain_uuid": "domain-1",
                                    "partition_id": 32766,
                                    "health": "HEALTHY",
                                    "port_count": 18,
                                    "location": {"chassis_id": 1, "slot_id": 1, "host_id": 1, "tray_index": 0},
                                },
                                "sample_switch": {
                                    "device_id": 1,
                                    "health": "HEALTHY",
                                    "port_count": 72,
                                    "location": {"chassis_id": 1, "slot_id": 10, "host_id": 1, "tray_index": 20},
                                },
                                "sample_chassis": {
                                    "name": "GB200-Rack-1",
                                    "health": "HEALTHY",
                                    "location": {"chassis_id": 1},
                                },
                                "scenario_answers": {
                                    "can_support_two_concurrent_4gpu_workloads": True,
                                    "node_gpu_grouping_field": "GpuIDList",
                                    "switch_asic_distinguishing_field": "DeviceID",
                                    "switch_tray_grouping_fields": [
                                        "LocationInfo.ChassisID",
                                        "LocationInfo.SlotID",
                                        "LocationInfo.HostID",
                                        "LocationInfo.TrayIndex",
                                    ],
                                    "sample_switch_tray": {
                                        "location": {"chassis_id": 1, "slot_id": 10, "host_id": 1, "tray_index": 20},
                                        "switch_asic_count": 2,
                                        "switch_device_ids": ["1", "2"],
                                    },
                                    "team_alpha_candidate": {
                                        "nodes": [{"node": "node-1"}, {"node": "node-2"}],
                                        "gpu_locations": ["1.1.1.1", "1.1.1.2", "1.2.1.1", "1.2.1.2"],
                                    },
                                    "team_beta_candidate": {
                                        "nodes": [{"node": "node-3"}, {"node": "node-4"}],
                                        "gpu_locations": ["1.3.1.1", "1.3.1.2", "1.4.1.1", "1.4.1.2"],
                                    },
                                },
                            },
                            "partitions": {
                                "partition_count": 2,
                                "default_partition": {"name": "Default", "member_count": 64},
                                "unassigned_gpu_count": 8,
                                "unassigned_gpu_locations": ["1.5.1.1", "1.5.1.2"],
                                "scenario_answers": {
                                    "ready_for_new_partition_create": True,
                                    "default_partition_present": True,
                                    "default_partition_member_count": 64,
                                    "operation_poll_path": "https://nmx.example/nmx/v1/operations/<operation-id>",
                                },
                            },
                            "telemetry": {
                                "metrics_endpoint": "https://nmx.example/nmx/v1/metrics",
                                "switch_temperature_series": 18,
                                "tx_throughput_series": 2592,
                                "rx_throughput_series": 2592,
                                "physical_error_series": 2592,
                                "cable_temperature_series": 2592,
                                "cable_rx_power_series": 2592,
                                "cable_tx_power_series": 2592,
                            },
                        }
                    },
                },
                "infiniband": {
                    "completeness": "full_stack_verified",
                    "runtime": {
                        "status": "runtime_verified",
                        "evidence": {
                            "single_nccl_peak_algbw_gbps": 2400.0,
                            "multi_nccl_peak_algbw_gbps": 480.0,
                            "alltoall_peak_algbw_gbps": 310.0,
                            "world_size": 2,
                        },
                    },
                    "scenario_summary": {
                        "capacity_and_path_visibility_ready": True,
                        "routing_and_counter_verification_ready": True,
                        "runtime_correlation_ready": True,
                        "visible_hca_count": 2,
                        "visible_host_count": 8,
                        "visible_switch_count": 4,
                        "linkinfo_visible": True,
                        "subnet_discovery_visible": True,
                        "saquery_visible": True,
                        "ibdiagnet_visible": True,
                        "routing_checks_ok": ["ibaddr", "ibtracert", "ibroute", "perfquery", "ibdiagnet"],
                        "multi_to_single_nccl_ratio": 0.2,
                        "world_size": 2,
                        "runtime_interpretation": "Multi-node NCCL scales below 0.70x of single-node; inspect routing, congestion, subnet state, and HCA/interface binding.",
                    },
                },
                "spectrum-x": {
                    "completeness": "full_stack_verified",
                    "runtime": {
                        "status": "runtime_verified",
                        "evidence": {
                            "single_nccl_peak_algbw_gbps": 2400.0,
                            "multi_nccl_peak_algbw_gbps": 600.0,
                            "alltoall_peak_algbw_gbps": 340.0,
                            "world_size": 2,
                        },
                    },
                    "scenario_summary": {
                        "fabric_readiness_ready": True,
                        "runtime_correlation_ready": True,
                        "switch_count_targeted": 2,
                        "switches_targeted": ["leaf01", "leaf02"],
                        "adaptive_routing_visible": True,
                        "roce_qos_visible": True,
                        "bgp_neighbor_state_visible": True,
                        "bgp_summary_visible": True,
                        "bgp_route_visibility": True,
                        "multi_to_single_nccl_ratio": 0.25,
                        "world_size": 2,
                        "runtime_interpretation": "Runtime artifacts show the Ethernet/RDMA path is alive; correlate any throughput or tail-latency knees with NVUE and BGP evidence.",
                    },
                },
            }
        },
    )
    _write_json(
        structured / f"{run_id}_fabric_ai_correlation.json",
        {"findings": ["NVLink runtime evidence peaks at 2400.0 GB/s algbw on the single-node NCCL path."]},
    )
    _write_json(
        structured / f"{run_id}_fabric_scorecard.json",
        {
            "status": "partial",
            "completeness": "runtime_verified",
            "summary": {
                "configured_management_planes": 0,
                "runtime_verified_families": 1,
                "full_stack_verified_families": 0,
            },
            "families": {
                "nvlink": {
                    "present": True,
                    "completeness": "runtime_verified",
                    "management_plane_configured": False,
                    "link_health": "runtime_verified",
                    "routing_correctness": "unknown",
                    "ai_workload_impact": "NVLink runtime evidence peaks at 2400.0 GB/s algbw on the single-node NCCL path.",
                    "hcas": [],
                },
                "infiniband": {
                    "present": True,
                    "completeness": "full_stack_verified",
                    "management_plane_configured": True,
                    "link_health": "runtime_verified",
                    "routing_correctness": "pass",
                    "ai_workload_impact": "Multi-node NCCL algbw peaks at 480.0 GB/s with a multi-to-single ratio of 0.20.",
                    "hcas": ["mlx5_0", "mlx5_1"],
                },
                "spectrum-x": {
                    "present": True,
                    "completeness": "full_stack_verified",
                    "management_plane_configured": True,
                    "link_health": "runtime_verified",
                    "routing_correctness": "pass",
                    "ai_workload_impact": "Spectrum-X / RoCE runtime verification is tied to the same multi-node NCCL and connectivity artifacts used for InfiniBand.",
                },
            },
        },
    )

    args = Namespace(run_id=run_id, label=label, root=cluster_root, run_dir=run_dir)
    report = module.render_report(args)
    notes = module.render_notes(args)

    assert "## Fabric Evaluation" in report
    assert "Fabric scorecard | Stored status" in report
    assert "NVLink runtime evidence peaks at 2400.0 GB/s algbw" in report
    assert "NMX Topology Scenario" in report
    assert "InfiniBand Scenario" in report
    assert "Spectrum-X / RoCE Scenario" in report
    assert "Chassis serials" in report
    assert "Node/GPU grouping field" in report
    assert "Switch tray grouping fields" in report
    assert "Port formula check" in report
    assert "Visible HCAs" in report
    assert "Routing checks passed" in report
    assert "Switches targeted" in report
    assert "RoCE QoS visible" in report
    assert "Sample compute-node fields" in report
    assert "Sample GPU fields" in report
    assert "Sample switch-tray grouping" in report
    assert "Supports Alpha+Beta 4-GPU split" in report
    assert "NMX Partition Scenario" in report
    assert "Default partition present" in report
    assert "python -m cli.aisp cluster nmx-partition-lab --nmx-url <nmx-base>" in report
    assert "NMX Telemetry Scenario" in report
    assert "Operation poll path | `<nmx-base>/operations/<operation-id>`" in report
    assert "Metrics endpoint | `<nmx-base>/metrics`" in report
    assert "python -m cli.aisp cluster fabric-eval --run-id " in report
    assert "python -m cli.aisp cluster fabric-eval --run-id " in notes
    assert "--nmx-url https://<your-nmx-host>" in report
    assert "--nmx-url https://<your-nmx-host>" in notes
    assert "nvlink.nvacademy.dev" not in report
    assert "common-eval --preset core-system" not in report
    assert "common-eval --preset core-system" not in notes
    assert "## Fabric Evaluation" in notes
