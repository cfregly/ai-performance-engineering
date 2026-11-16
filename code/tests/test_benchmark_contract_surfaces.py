from __future__ import annotations

import yaml

from core.benchmark.contracts_surface import get_benchmark_contracts_summary, render_benchmark_run_yaml
from core.tools.tools_commands import TOOLS
from mcp.mcp_server import HANDLERS, TOOLS as MCP_TOOLS


def test_benchmark_contracts_summary_exposes_expected_surfaces() -> None:
    summary = get_benchmark_contracts_summary()

    assert summary["schema_version"] == "2026-03-13"
    assert summary["surface_order"] == [
        "methodology",
        "warehouse",
        "workload_spec",
        "benchmark_run",
        "warehouse_contract",
        "kubernetes_service",
        "benchmark_run_crd",
    ]
    assert summary["available"] is True
    assert summary["surface_count"] == len(summary["surface_order"])
    assert summary["missing_surface_count"] == 0
    assert summary["interfaces"]["cli"] == "python -m cli.aisp tools benchmark-contracts"
    assert summary["interfaces"]["mcp_tool"] == "benchmark_contracts"
    assert [entry["id"] for entry in summary["interface_entries"]] == ["cli", "mcp"]
    assert summary["generator"]["defaults"]["name"] == "publication-inference-stack-b200"
    assert summary["generator"]["defaults"]["comparisonVariable"] == "runtime_version"
    assert summary["generator"]["preview_yaml"].startswith("apiVersion: benchmarking.aisp.dev/v1alpha1")
    assert [entry["id"] for entry in summary["generator"]["render_interface_entries"]] == ["cli", "mcp"]

    contracts = summary["contracts"]
    assert contracts["warehouse"]["exists"] is True
    assert contracts["benchmark_run"]["summary"]["has_observability"] is True
    assert contracts["benchmark_run"]["summary"]["has_sinks"] is True


def test_benchmark_contracts_are_exposed_in_cli_and_mcp() -> None:
    assert "benchmark-contracts" in TOOLS
    assert "benchmark-run-render" in TOOLS
    for name in ("benchmark_contracts", "render_benchmark_run"):
        assert name in MCP_TOOLS
        assert callable(HANDLERS[name])


def test_benchmark_run_renderer_tracks_template_lockstep() -> None:
    rendered = render_benchmark_run_yaml(
        {
            "name": "realism-mixed-stack",
            "benchmarkClass": "realism_grade",
            "workloadType": "mixed",
            "schedulerPath": "slinky-kueue-topology-aware",
            "cadence": "nightly",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "precision": "fp8",
            "batchingPolicy": "static",
            "concurrencyModel": "open_loop",
            "comparisonVariable": "scheduler_path",
        }
    )
    payload = yaml.safe_load(rendered["rendered_yaml"])

    assert payload["metadata"]["name"] == "realism-mixed-stack"
    assert payload["spec"]["intent"]["benchmarkClass"] == "realism_grade"
    assert payload["spec"]["intent"]["workloadType"] == "mixed"
    assert payload["spec"]["intent"]["schedulerPath"] == "slinky-kueue-topology-aware"
    assert payload["spec"]["workload"]["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["spec"]["metrics"]["training"]["enabled"] is True
    assert payload["spec"]["metrics"]["inference"]["enabled"] is True
    assert payload["spec"]["executionPolicy"]["realismGrade"]["multiTenantScenarios"] is True
