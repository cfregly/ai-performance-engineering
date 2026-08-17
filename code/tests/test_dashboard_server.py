import json
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import core.engine as engine_module
from core.api.handlers import ai_execute, ai_tools
from core.api.registry import get_dashboard_mcp_tools
from core.engine import get_engine, reset_engine
from core.optimization.campaign import CampaignConfig, CampaignWorkspace
from core.perf_core import PerfCore
from dashboard.api import server
from tests.http_client import asgi_request


class _TestPerfCore(PerfCore):
    def __init__(self, *, history_root, data_file=None, bench_root=None):
        super().__init__(data_file=data_file, bench_root=bench_root)
        self._test_history_root = history_root

    def _tier1_history_root(self):
        return self._test_history_root


def test_configure_engine_uses_data_file(sample_benchmark_results_file):
    reset_engine()
    server._configure_engine(sample_benchmark_results_file)
    result = get_engine().benchmark.data()

    assert result["summary"]["total_benchmarks"] == 1
    assert result["benchmarks"][0]["name"] == "example_a"

    reset_engine()


def test_dashboard_cli_has_serve_command():
    runner = CliRunner()
    result = runner.invoke(server.cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start the dashboard API server." in result.output


def test_dashboard_http_benchmark_overview_route_returns_success_envelope(
    sample_benchmark_results_file,
):
    pytest.importorskip("fastapi")

    reset_engine()
    server._configure_engine(sample_benchmark_results_file)
    response = asgi_request(server.fastapi_app, "GET", "/api/benchmark/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "benchmark.overview"
    assert payload["status"] == "ok"
    assert payload["success"] is True
    assert payload["result"]["summary"]["total"] == 1
    reset_engine()


def test_dashboard_http_compare_route_returns_error_envelope() -> None:
    pytest.importorskip("fastapi")

    response = asgi_request(server.fastapi_app, "GET", "/api/benchmark/compare")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "benchmark.compare"
    assert payload["status"] == "error"
    assert payload["success"] is False
    assert payload["error_type"] == "value_error"
    assert "baseline is required" in payload["error"]


def test_dashboard_cors_allows_only_configured_ui_origins() -> None:
    pytest.importorskip("fastapi")
    allowed_origin = server._allowed_ui_origins()[0]

    allowed = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/benchmark/compare",
        headers={"Origin": allowed_origin},
    )
    denied = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/benchmark/compare",
        headers={"Origin": "https://evil.example"},
    )

    assert allowed.headers["access-control-allow-origin"] == allowed_origin
    assert "access-control-allow-origin" not in denied.headers
    with pytest.raises(ValueError, match="invalid dashboard UI origin"):
        server._allowed_ui_origins("*")


def test_campaign_api_restricts_workspaces_to_configured_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    campaign_root = tmp_path / "campaigns"
    campaign_root.mkdir()
    allowed_workspace = campaign_root / "allowed"
    CampaignWorkspace.initialize(
        allowed_workspace,
        CampaignConfig(
            objective="Verify campaign API boundaries.",
            primary_metric="latency_ms",
            initial_control_commit="a" * 40,
            primary_cases=["common"],
            frozen_cases=["common"],
            workload_spec="workload.json",
            workload_sha256="b" * 64,
            environment_spec="environment.json",
            environment_sha256="c" * 64,
        ),
    )
    outside_workspace = tmp_path / "outside"
    outside_workspace.mkdir()
    monkeypatch.setenv(server.CAMPAIGN_ROOT_ENV, str(campaign_root))
    server._configure_campaign_root(None)

    allowed = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/optimization/campaign",
        params={"workspace": "allowed"},
    )
    denied = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/optimization/campaign",
        params={"workspace": str(outside_workspace)},
    )
    denied_traversal = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/optimization/campaign",
        params={"workspace": "../outside"},
    )
    denied_artifact = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/optimization/campaign/artifact",
        params={"workspace": str(outside_workspace), "artifact": "secret.txt"},
    )

    assert allowed.status_code == 200
    assert allowed.json()["status"] == "ok"
    assert denied.status_code == 403
    assert denied_traversal.status_code == 403
    assert denied_artifact.status_code == 403


def test_campaign_api_requires_an_explicit_server_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.delenv(server.CAMPAIGN_ROOT_ENV, raising=False)
    server._configure_campaign_root(None)

    response = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/optimization/campaign",
        params={"workspace": "campaign"},
    )

    assert response.status_code == 503
    assert "Campaign API is disabled" in response.json()["detail"]


def test_serve_uses_the_app_with_its_configured_campaign_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    fake_uvicorn = SimpleNamespace(run=lambda app, **kwargs: calls.append((app, kwargs)))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    campaign_root = tmp_path / "campaigns"
    campaign_root.mkdir()

    try:
        server.serve_dashboard(
            campaign_root=campaign_root,
            open_browser=False,
        )

        assert calls[0][0] is server.fastapi_app
        assert server._configured_campaign_root() == campaign_root.resolve()
    finally:
        server._configure_campaign_root(None)


def test_ai_tools_only_lists_dashboard_mcp_subset() -> None:
    payload = ai_tools({})
    tool_names = {tool["name"] for tool in payload["tools"]}

    assert tool_names == set(get_dashboard_mcp_tools())


def test_ai_execute_rejects_tools_outside_dashboard_subset() -> None:
    with pytest.raises(ValueError, match="not exposed by the dashboard API"):
        ai_execute({"tool": "recommend", "params": {}})


def test_engine_exposes_tier1_history_and_trends(tmp_path, sample_benchmark_results_file):
    history_root = tmp_path / "artifacts" / "history" / "tier1"
    run_dir = history_root / "20260309_010000_tier1_local"
    run_dir.mkdir(parents=True)

    summary_path = run_dir / "summary.json"
    regression_path = run_dir / "regression_summary.json"
    trend_path = run_dir / "trend_snapshot.json"
    index_path = history_root / "index.json"

    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "run_id": "20260309_010000_tier1_local",
                "generated_at": "2026-03-09T01:00:00",
                "targets": [
                    {
                        "key": "flashattention4_alibi",
                        "target": "labs/flashattention4:flashattention4_alibi",
                        "category": "attention",
                        "status": "succeeded",
                        "best_speedup": 12.5,
                        "artifacts": {
                            "baseline_nsys_rep": "artifacts/runs/demo/profiles/flash.nsys-rep",
                        },
                    }
                ],
                "summary": {
                    "target_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                    "avg_speedup": 12.5,
                    "median_speedup": 12.5,
                    "geomean_speedup": 12.5,
                    "representative_speedup": 12.5,
                    "max_speedup": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    regression_path.write_text(
        json.dumps(
            {
                "baseline_run_id": "20260308_225441_tier1_manual",
                "current_run_id": "20260309_010000_tier1_local",
                "regressions": [],
                "improvements": [{"key": "flashattention4_alibi", "delta_pct": 4.2}],
                "new_targets": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "run_count": 1,
                "latest_run_id": "20260309_010000_tier1_local",
                "best_speedup_seen": 12.5,
                "history": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "run_accepted": True,
                        "baseline_eligible": True,
                        "generated_at": "2026-03-09T01:00:00",
                        "avg_speedup": 12.5,
                        "median_speedup": 12.5,
                        "geomean_speedup": 12.5,
                        "representative_speedup": 12.5,
                        "max_speedup": 12.5,
                        "succeeded": 1,
                        "failed": 0,
                        "skipped": 0,
                        "missing": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "history_root": ".",
                "runs": [
                        {
                            "run_id": "20260309_010000_tier1_local",
                            "run_accepted": True,
                            "baseline_eligible": True,
                            "summary_path": "20260309_010000_tier1_local/summary.json",
                        "regression_summary_path": (
                            "20260309_010000_tier1_local/regression_summary.md"
                        ),
                        "regression_json_path": (
                            "20260309_010000_tier1_local/regression_summary.json"
                        ),
                        "trend_snapshot_path": ("20260309_010000_tier1_local/trend_snapshot.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reset_engine()
    engine_module._handler_instance = _TestPerfCore(
        history_root=history_root,
        data_file=sample_benchmark_results_file,
        bench_root=tmp_path,
    )
    history = get_engine().benchmark.tier1_history()
    trends = get_engine().benchmark.tier1_trends()
    target_history = get_engine().benchmark.tier1_target_history(key="flashattention4_alibi")

    assert history["total_runs"] == 1
    assert history["latest_run_id"] == "20260309_010000_tier1_local"
    assert history["latest"]["run"]["representative_speedup"] == 12.5
    assert history["latest"]["improvements"][0]["key"] == "flashattention4_alibi"
    assert history["latest"]["run"]["regression_summary_json_path"] == (
        "20260309_010000_tier1_local/regression_summary.json"
    )
    assert trends["latest_run_id"] == "20260309_010000_tier1_local"
    assert trends["best_speedup_seen"] == 12.5
    assert target_history["selected_key"] == "flashattention4_alibi"
    assert target_history["run_count"] == 1
    assert target_history["history"][0]["target"] == "labs/flashattention4:flashattention4_alibi"
    assert target_history["history"][0]["best_speedup"] == 12.5
    assert target_history["history"][0]["artifacts"]["baseline_nsys_rep"] == (
        "artifacts/runs/demo/profiles/flash.nsys-rep"
    )

    reset_engine()
