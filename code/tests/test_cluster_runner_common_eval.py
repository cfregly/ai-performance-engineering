from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import core.cluster.runner as cluster_runner


def test_run_cluster_common_eval_common_answer_fast_composes_expected_flags(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_common_eval(
        preset="common-answer-fast",
        run_id="2026-03-06_eval_fast",
        hosts=["localhost"],
        labels=["localhost"],
        extra_args=["--skip-render-localhost-report"],
    )

    assert result["success"] is True
    assert result["preset"] == "common-answer-fast"
    assert "fast answer bundle" in result["preset_description"].lower()
    assert "vllm_request_rate_sweep" in result["artifact_roles"]
    assert captured["extra_args"] == [
        "--skip-quick-friction",
        "--skip-monitoring-expectations",
        "--disable-fp4",
        "--health-suite",
        "off",
        "--skip-vllm-multinode",
        "--model",
        "openai/gpt-oss-20b",
        "--tp",
        "1",
        "--isl",
        "512",
        "--osl",
        "128",
        "--concurrency-range",
        "8 16 32",
        "--run-vllm-request-rate-sweep",
        "--vllm-request-rate-range",
        "1 2 4",
        "--vllm-request-rate-max-concurrency",
        "32",
        "--vllm-request-rate-num-prompts",
        "128",
        "--fio-runtime",
        "15",
        "--run-nvbandwidth",
        "--nvbandwidth-quick",
        "--skip-render-localhost-report",
    ]


def test_run_cluster_common_eval_core_system_composes_expected_flags(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_common_eval(
        preset="core-system",
        run_id="2026-03-05_eval_core",
        hosts=["localhost"],
        labels=["localhost"],
        coverage_baseline_run_id="baseline_r1",
        extra_args=["--foo"],
    )

    assert result["success"] is True
    assert result["preset"] == "core-system"
    assert "nvbandwidth" in result["preset_description"].lower()
    assert "vllm_request_rate_sweep" in result["artifact_roles"]
    assert captured["mode"] == "full"
    assert captured["hosts"] == ["localhost"]
    assert captured["labels"] == ["localhost"]
    assert captured["extra_args"] == [
        "--run-vllm-request-rate-sweep",
        "--run-nvbandwidth",
        "--coverage-baseline-run-id",
        "baseline_r1",
        "--foo",
    ]


def test_run_cluster_common_eval_rejects_unknown_preset() -> None:
    result = cluster_runner.run_cluster_common_eval(preset="not-a-preset", hosts=["localhost"])
    assert result["success"] is False
    assert "Unknown preset" in result["error"]


def test_run_cluster_common_eval_fabric_systems_composes_expected_flags(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_common_eval(
        preset="fabric-systems",
        run_id="2026-03-16_fabric_eval",
        hosts=["localhost"],
        labels=["localhost"],
        nmx_url="https://nmx.example",
        nmx_token="secret-token",
        ib_mgmt_host="ib-mgmt.example",
        ib_mgmt_user="ibadmin",
        ib_mgmt_ssh_key="/tmp/ib-key",
        cumulus_hosts=["leaf01", "leaf02"],
        cumulus_user="cumulus",
        cumulus_ssh_key="/tmp/cumulus-key",
    )

    assert result["success"] is True
    assert result["preset"] == "fabric-systems"
    assert "fabric" in result["preset_description"].lower()
    assert "fabric_scorecard" in result["artifact_roles"]
    assert captured["extra_args"] == [
        "--modern-llm-profile",
        "--no-strict-canonical-completeness",
        "--run-fabric-eval",
        "--nmx-url",
        "https://nmx.example",
        "--nmx-token",
        "secret-token",
        "--ib-mgmt-host",
        "ib-mgmt.example",
        "--ib-mgmt-user",
        "ibadmin",
        "--ib-mgmt-ssh-key",
        "/tmp/ib-key",
        "--cumulus-hosts",
        "leaf01,leaf02",
        "--cumulus-user",
        "cumulus",
        "--cumulus-ssh-key",
        "/tmp/cumulus-key",
    ]


def test_run_cluster_fabric_eval_readiness_only_reports_emitted_artifacts(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_eval_suite(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "command": ["fake"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_eval_suite", fake_run_cluster_eval_suite)

    result = cluster_runner.run_cluster_fabric_eval(
        run_id="2026-03-16_fabric_readiness",
        hosts=["node-a", "node-b"],
        labels=["node-a", "node-b"],
        extra_args=["--multinode-readiness-check-only"],
    )

    assert result["success"] is True
    assert result["artifact_roles"] == ["multinode_readiness", "manifest"]
    assert captured["extra_args"] == [
        "--modern-llm-profile",
        "--no-strict-canonical-completeness",
        "--run-fabric-eval",
        "--multinode-readiness-check-only",
    ]


def test_run_cluster_fabric_eval_adds_management_plane_flag(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_cluster_common_eval(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs.get("run_id"), "artifact_roles": ["fabric_scorecard"]}

    monkeypatch.setattr(cluster_runner, "run_cluster_common_eval", fake_run_cluster_common_eval)

    result = cluster_runner.run_cluster_fabric_eval(
        run_id="2026-03-16_fabric_eval",
        hosts=["localhost"],
        nmx_url="https://nmx.example",
        nmx_token="secret-token",
        ib_mgmt_host="ib-mgmt.example",
        ib_mgmt_user="ibadmin",
        ib_mgmt_ssh_key="/tmp/ib-key",
        cumulus_hosts=["leaf01", "leaf02"],
        cumulus_user="cumulus",
        cumulus_ssh_key="/tmp/cumulus-key",
        require_management_plane=True,
        extra_args=["--skip-render-localhost-report"],
    )

    assert result["success"] is True
    assert result["entrypoint"] == "cluster.fabric-eval"
    assert result["require_management_plane"] is True
    assert captured["preset"] == "fabric-systems"
    assert captured["nmx_url"] == "https://nmx.example"
    assert captured["nmx_token"] == "secret-token"
    assert captured["ib_mgmt_host"] == "ib-mgmt.example"
    assert captured["ib_mgmt_user"] == "ibadmin"
    assert captured["ib_mgmt_ssh_key"] == "/tmp/ib-key"
    assert captured["cumulus_hosts"] == ["leaf01", "leaf02"]
    assert captured["cumulus_user"] == "cumulus"
    assert captured["cumulus_ssh_key"] == "/tmp/cumulus-key"
    assert captured["extra_args"] == ["--skip-render-localhost-report", "--require-management-plane"]


def test_cluster_suite_planned_steps_readiness_only_reports_emitted_steps() -> None:
    planned_steps = cluster_runner._cluster_suite_planned_steps(
        run_id="2026-03-16_fabric_readiness",
        hosts=["node-a", "node-b"],
        labels=["node-a", "node-b"],
        primary_label="node-a",
        extra_args=[
            "--modern-llm-profile",
            "--run-fabric-eval",
            "--multinode-readiness-check-only",
        ],
        coverage_baseline_run_id=None,
        oob_if="eth0",
        socket_ifname="eth0",
    )

    assert planned_steps == ["multinode_readiness", "manifest_refresh"]


def test_cluster_suite_progress_current_uses_suite_steps_and_inflight_log(tmp_path: Path) -> None:
    suite_steps_path = tmp_path / "structured" / "2026-03-26_suite_steps.json"
    suite_log_dir = tmp_path / "raw" / "2026-03-26_suite"
    suite_steps_path.parent.mkdir(parents=True, exist_ok=True)
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_steps_path.write_text(
        json.dumps(
            [
                {"name": "bootstrap_nodes", "exit_code": 0},
                {"name": "preflight_services", "exit_code": 0},
            ]
        ),
        encoding="utf-8",
    )
    (suite_log_dir / "vllm_serve_sweep.log").write_text("running\n", encoding="utf-8")

    payload = cluster_runner._cluster_suite_progress_current(
        run_id="2026-03-26",
        planned_steps=[
            "bootstrap_nodes",
            "preflight_services",
            "discovery",
            "vllm_serve_sweep",
        ],
        suite_steps_path=suite_steps_path,
        suite_log_dir=suite_log_dir,
    )

    assert payload["step"] == "vllm_serve_sweep"
    assert payload["step_detail"] == "completed 2/4 suite steps"
    assert payload["percent_complete"] == 50.0
    assert payload["metrics"]["current_step"] == "vllm_serve_sweep"


def test_cluster_suite_progress_state_disables_request_rate_for_localhost_fabric_canary() -> None:
    state = cluster_runner._cluster_suite_progress_state(
        hosts=["localhost"],
        labels=["localhost"],
        primary_label="localhost",
        extra_args=["--run-fabric-eval", "--modern-llm-profile"],
        coverage_baseline_run_id=None,
        oob_if=None,
        socket_ifname=None,
    )

    assert state["run_fabric_eval"] is True
    assert state["modern_llm_profile"] is True
    assert state["is_localhost_package"] is True
    assert state["run_vllm_request_rate_sweep"] is False


def test_build_cluster_nmx_partition_lab_wraps_payload(monkeypatch) -> None:
    def fake_build_nmx_partition_lab_payload(**kwargs: Any) -> Dict[str, Any]:
        assert kwargs["nmx_url"] == "https://nmx.example"
        assert kwargs["nmx_token"] == "secret-token"
        assert kwargs["alpha_name"] == "AlphaPartition"
        return {
            "status": "ok",
            "collection_mode": "nmx_partition_lab",
            "lab_only": True,
            "commands": {"inspect_partitions": "curl -k https://nmx.example/nmx/v1/partitions | jq"},
        }

    monkeypatch.setattr("cluster.fabric.build_nmx_partition_lab_payload", fake_build_nmx_partition_lab_payload)

    result = cluster_runner.build_cluster_nmx_partition_lab(
        nmx_url="https://nmx.example",
        nmx_token="secret-token",
        alpha_name="AlphaPartition",
        beta_name="BetaPartition",
    )

    assert result["success"] is True
    assert result["entrypoint"] == "cluster.nmx-partition-lab"
    assert result["lab_only"] is True
    assert "inspect_partitions" in result["commands"]
