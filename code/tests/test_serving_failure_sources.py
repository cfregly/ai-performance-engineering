"""Read both native failure endpoints and reject failures or resets per source."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from labs.serving_comparison.schema import ArmProfile, ConfigError, TelemetrySource, load_profile
from labs.serving_comparison.telemetry import capture_pd_telemetry, compute_pd_delta


@contextmanager
def metrics_endpoints():
    values = {
        "prefill": {
            "bootstrap_failures_total": [({}, 5)],
            "transfer_failures_total": [({}, 0)],
        },
        "decode": {
            "bootstrap_failures_total": [({}, 0)],
            "transfer_failures_total": [({}, 0)],
        },
        "activity": 10,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            role = self.path.strip("/")
            lines = [f"activity_total {values['activity']}"]
            for metric_name, samples in values[role].items():
                lines.append(f"# TYPE {metric_name} counter")
                for labels, value in samples:
                    label_text = ""
                    if labels:
                        label_text = (
                            "{"
                            + ",".join(
                                f'{key}="{label_value}"' for key, label_value in labels.items()
                            )
                            + "}"
                        )
                    lines.append(f"{metric_name}{label_text} {value}")
            payload = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", values
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def pd_arm(base):
    failures = {"names": ["bootstrap_failures_total", "transfer_failures_total"], "reduce": "sum"}
    prefill_metrics = {
        name: "activity_total"
        for name in (
            "kv_transfer_requests_total",
            "kv_transfer_time_seconds_total",
            "prefill_requests_total",
        )
    }
    prefill_metrics["kv_transfer_failures_total"] = failures
    return ArmProfile(
        arm_id="sglang-prefill-decode",
        engine="sglang",
        architecture="prefill_decode",
        endpoint=base,
        api_key_env=None,
        runtime_version="test",
        runtime_build_id="test",
        gpu_ids=("0", "1"),
        identity_probes=(),
        lifecycle={},
        telemetry_sources=(
            TelemetrySource("prefill", "prometheus", base + "/prefill", prefill_metrics),
            TelemetrySource(
                "decode",
                "prometheus",
                base + "/decode",
                {
                    "decode_requests_total": "activity_total",
                    "kv_transfer_failures_total": failures,
                },
            ),
        ),
        pd_provenance_path=None,
        pd_provenance={"connector": {"name": "NIXL"}},
    )


def test_both_failure_sources_are_observed():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            before = await capture_pd_telemetry(client, arm)
            assert before.values["kv_transfer_failures_total"] == 5
            assert before.failure_counters_by_source == {"prefill": 5, "decode": 0}
            assert set(before.failure_series_by_source) == {"prefill", "decode"}
            values["activity"] += 1
            clean = await capture_pd_telemetry(client, arm)
            assert compute_pd_delta(before, clean)["kv_transfer_failures_total"] == 0

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_each_failure_source_increment_rejects(role):
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            before = await capture_pd_telemetry(client, arm)
            current = values[role]["bootstrap_failures_total"][0][1]
            values[role]["bootstrap_failures_total"] = [({}, current + 1)]
            values["activity"] += 1
            after = await capture_pd_telemetry(client, arm)
            assert set(after.raw_by_source) == {"prefill", "decode"}
            with pytest.raises(ConfigError, match=f"failed KV transfers at {role}"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_source_reset_cannot_be_hidden_by_another_sources_increase():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            before = await capture_pd_telemetry(client, arm)
            values["activity"] += 1
            clean = await capture_pd_telemetry(client, arm)
            incomplete = replace(clean, failure_counters_by_source={"prefill": 5})
            with pytest.raises(ConfigError, match="sources changed"):
                compute_pd_delta(before, incomplete)

            values["prefill"]["bootstrap_failures_total"] = [({}, 4)]
            values["decode"]["bootstrap_failures_total"] = [({}, 1)]
            after = await capture_pd_telemetry(client, arm)
            assert (
                after.values["kv_transfer_failures_total"]
                == before.values["kv_transfer_failures_total"]
            )
            with pytest.raises(ConfigError, match="counter decreased: prefill"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_same_source_metric_family_reset_cannot_hide_failure_increase():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            before = await capture_pd_telemetry(client, arm)
            values["prefill"]["bootstrap_failures_total"] = [({}, 4)]
            values["prefill"]["transfer_failures_total"] = [({}, 1)]
            values["activity"] += 1
            after = await capture_pd_telemetry(client, arm)
            assert after.failure_counters_by_source == before.failure_counters_by_source
            with pytest.raises(ConfigError, match="counter decreased: prefill KV failure series"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_labeled_series_reset_cannot_hide_another_series_increase():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            values["prefill"]["bootstrap_failures_total"] = [
                ({"worker": "a"}, 5),
                ({"worker": "b"}, 0),
            ]
            before = await capture_pd_telemetry(client, arm)
            values["prefill"]["bootstrap_failures_total"] = [
                ({"worker": "a"}, 4),
                ({"worker": "b"}, 1),
            ]
            values["activity"] += 1
            after = await capture_pd_telemetry(client, arm)
            assert after.failure_counters_by_source == before.failure_counters_by_source
            with pytest.raises(ConfigError, match="counter decreased: prefill KV failure series"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_partially_missing_failure_family_rejects_capture():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            del values["prefill"]["transfer_failures_total"]
            with pytest.raises(ConfigError, match="missing required Prometheus samples"):
                await capture_pd_telemetry(client, pd_arm(base))

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_failure_series_disappearance_rejects_interval():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            values["prefill"]["bootstrap_failures_total"] = [
                ({"worker": "a"}, 5),
                ({"worker": "b"}, 0),
            ]
            before = await capture_pd_telemetry(client, arm)
            values["prefill"]["bootstrap_failures_total"] = [({"worker": "a"}, 5)]
            values["activity"] += 1
            after = await capture_pd_telemetry(client, arm)
            with pytest.raises(ConfigError, match="series disappeared at prefill"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_duplicate_failure_series_rejects_capture():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            values["prefill"]["bootstrap_failures_total"] = [
                ({"worker": "a"}, 5),
                ({"worker": "a"}, 5),
            ]
            with pytest.raises(ConfigError, match="duplicate failure counter series"):
                await capture_pd_telemetry(client, pd_arm(base))

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


def test_new_nonzero_failure_series_rejects_interval():
    async def exercise(base, values):
        async with httpx.AsyncClient(trust_env=False) as client:
            arm = pd_arm(base)
            before = await capture_pd_telemetry(client, arm)
            values["prefill"]["transfer_failures_total"].append(({"worker": "new"}, 1))
            values["activity"] += 1
            after = await capture_pd_telemetry(client, arm)
            with pytest.raises(ConfigError, match="new nonzero failure series at prefill"):
                compute_pd_delta(before, after)

    with metrics_endpoints() as (base, values):
        asyncio.run(exercise(base, values))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_native_profile_requires_failures_from_both_roles(tmp_path, role):
    examples = Path(__file__).resolve().parents[1] / "labs/serving_comparison/examples"
    document = json.loads((examples / "profile.engine.template.json").read_text())
    for arm in document["arms"]:
        arm["lifecycle"]["working_directory"] = str(examples.parents[3])
        if "pd_provenance" in arm:
            arm["pd_provenance"] = str(examples / arm["pd_provenance"])
        if arm["engine"] == "sglang" and arm["architecture"] == "prefill_decode":
            for source in arm["telemetry_sources"]:
                if source["source_id"] == role + "-native":
                    del source["metrics"]["kv_transfer_failures_total"]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match=f"failure counters from {role}"):
        load_profile(profile)
