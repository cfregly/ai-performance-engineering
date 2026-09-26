"""Fail-closed P/D telemetry integrity checks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from labs.serving_comparison.schema import ArmProfile, ConfigError, TelemetrySource
from labs.serving_comparison.telemetry import (
    TelemetrySnapshot,
    _parse_prometheus_metric,
    _parse_prometheus_metric_with_evidence,
    capture_pd_telemetry,
    compute_pd_delta,
)


def test_prometheus_parser_ignores_nonfinite_unselected_metrics() -> None:
    text = "unrelated_metric NaN\nselected_metric 3\n"

    assert _parse_prometheus_metric(text, "selected_metric", "selected") == 3.0
    with pytest.raises(ConfigError, match="not finite"):
        _parse_prometheus_metric(text, "unrelated_metric", "selected")


def test_lazy_counter_zero_requires_exact_counter_family_metadata() -> None:
    selector = {
        "names": ["native_bootstrap_failures_total", "native_transfer_failures_total"],
        "reduce": "sum",
        "missing_value": 0,
    }
    text = "\n".join(
        [
            "# TYPE native_bootstrap_failures_total counter",
            "# TYPE native_transfer_failures_total counter",
            "unrelated_metric NaN",
        ]
    )

    value, evidence = _parse_prometheus_metric_with_evidence(text, selector, "failures")

    assert value == 0
    assert evidence == {
        "names": ["native_bootstrap_failures_total", "native_transfer_failures_total"],
        "labels": {},
        "reduce": "sum",
        "scale": 1.0,
        "offset": 0.0,
        "missing_value": 0.0,
        "missing_names": [
            "native_bootstrap_failures_total",
            "native_transfer_failures_total",
        ],
        "matched_sample_count": 0,
        "declared_types": {
            "native_bootstrap_failures_total": "counter",
            "native_transfer_failures_total": "counter",
        },
    }

    for invalid_text in (
        "",
        "# TYPE native_bootstrap_failures_total counter\n",
        "# TYPE native_bootstrap_failures_total counter\n"
        "# TYPE native_transfer_failures_total gauge\n",
    ):
        with pytest.raises(ConfigError, match="exact metric families"):
            _parse_prometheus_metric(invalid_text, selector, "failures")

    filtered_selector = {**selector, "labels": {"worker": "missing"}}
    with pytest.raises(ConfigError, match="invalid Prometheus selector"):
        _parse_prometheus_metric(text, filtered_selector, "failures")


def test_lazy_counter_positive_sample_remains_observed() -> None:
    selector = {
        "name": "native_transfer_failures_total",
        "reduce": "sum",
        "missing_value": 0,
    }
    text = (
        "# TYPE native_transfer_failures_total counter\n"
        'native_transfer_failures_total{worker="0"} 1\n'
    )

    assert _parse_prometheus_metric(text, selector, "failures") == 1


def _snapshot(
    *,
    transfer_requests: float,
    transfer_failures: float,
    transfer_time_seconds: float,
    prefill_requests: float,
    decode_requests: float,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        captured_monotonic_s=0.0,
        captured_unix_s=0.0,
        values={
            "kv_transfer_requests_total": transfer_requests,
            "kv_transfer_failures_total": transfer_failures,
            "kv_transfer_time_seconds_total": transfer_time_seconds,
            "prefill_requests_total": prefill_requests,
            "decode_requests_total": decode_requests,
        },
        raw_by_source={},
    )


@pytest.mark.parametrize(
    ("transfer_requests", "transfer_failures"),
    [(2.0, 2.0), (3.0, 1.0)],
    ids=["all-failed", "partially-failed"],
)
def test_pd_delta_rejects_any_failed_transfer(
    transfer_requests: float, transfer_failures: float
) -> None:
    before = _snapshot(
        transfer_requests=10.0,
        transfer_failures=4.0,
        transfer_time_seconds=2.0,
        prefill_requests=10.0,
        decode_requests=10.0,
    )
    after = _snapshot(
        transfer_requests=10.0 + transfer_requests,
        transfer_failures=4.0 + transfer_failures,
        transfer_time_seconds=2.25,
        prefill_requests=10.0 + transfer_requests,
        decode_requests=10.0 + transfer_requests,
    )

    with pytest.raises(ConfigError, match="failed KV transfers"):
        compute_pd_delta(before, after)


def test_pd_delta_preserves_counter_and_unit_semantics() -> None:
    before = _snapshot(
        transfer_requests=7.0,
        transfer_failures=2.0,
        transfer_time_seconds=0.25,
        prefill_requests=8.0,
        decode_requests=6.0,
    )
    after = _snapshot(
        transfer_requests=10.0,
        transfer_failures=2.0,
        transfer_time_seconds=0.375,
        prefill_requests=11.0,
        decode_requests=9.0,
    )

    assert compute_pd_delta(before, after) == {
        "decode_requests_total": 3.0,
        "kv_transfer_failures_total": 0.0,
        "kv_transfer_requests_total": 3.0,
        "kv_transfer_time_seconds_total": 0.125,
        "prefill_requests_total": 3.0,
    }

    reset = _snapshot(
        transfer_requests=10.0,
        transfer_failures=1.0,
        transfer_time_seconds=0.375,
        prefill_requests=11.0,
        decode_requests=9.0,
    )
    with pytest.raises(ConfigError, match="counter decreased: kv_transfer_failures_total"):
        compute_pd_delta(after, reset)


class _HangingMetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        time.sleep(2.0)

    def log_message(self, message_format: str, *args: object) -> None:
        return


class _MetricsServer(ThreadingHTTPServer):
    daemon_threads = True


@contextmanager
def _hanging_metrics_url() -> Iterator[str]:
    server = _MetricsServer(("127.0.0.1", 0), _HangingMetricsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/metrics"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _pd_arm(metrics_url: str) -> ArmProfile:
    return ArmProfile(
        arm_id="sglang-prefill-decode",
        engine="sglang",
        architecture="prefill_decode",
        endpoint="http://127.0.0.1:1/v1/completions",
        api_key_env=None,
        runtime_version="test",
        runtime_build_id="test",
        gpu_ids=("0", "1"),
        identity_probes=(),
        lifecycle={},
        telemetry_sources=(
            TelemetrySource(
                source_id="native",
                format="standard_json",
                url=metrics_url,
                metrics={
                    "kv_transfer_requests_total": "metrics.transfer_requests",
                    "kv_transfer_failures_total": "metrics.transfer_failures",
                    "kv_transfer_time_seconds_total": "metrics.transfer_time_seconds",
                    "prefill_requests_total": "metrics.prefill_requests",
                    "decode_requests_total": "metrics.decode_requests",
                },
            ),
        ),
        pd_provenance_path=Path("provenance.json"),
        pd_provenance={"connector": {"name": "NIXL"}},
    )


def test_hanging_metrics_endpoint_obeys_request_timeout() -> None:
    async def capture(url: str) -> None:
        timeout = httpx.Timeout(connect=1.0, read=None, write=1.0, pool=1.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            await capture_pd_telemetry(client, _pd_arm(url), request_timeout_s=0.1)

    with _hanging_metrics_url() as url:
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(capture(url))
        assert time.monotonic() - started < 1.0


def test_raw_metrics_are_retained_before_selector_rejection() -> None:
    raw = "# TYPE unrelated_metric gauge\nunrelated_metric 1\n"
    captured: dict[str, str] = {}
    arm = replace(
        _pd_arm("http://metrics.test/metrics"),
        telemetry_sources=(
            TelemetrySource(
                source_id="native",
                format="prometheus",
                url="http://metrics.test/metrics",
                metrics={
                    "kv_transfer_failures_total": {
                        "name": "missing_failures_total",
                        "missing_value": 0,
                    }
                },
            ),
        ),
    )

    async def capture() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=raw))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ConfigError, match="exact metric families"):
                await capture_pd_telemetry(
                    client,
                    arm,
                    raw_sink=lambda source_id, text: captured.__setitem__(source_id, text),
                )

    asyncio.run(capture())
    assert captured == {"native": raw}
