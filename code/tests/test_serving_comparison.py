"""Real-socket protocol and evidence gates for the serving comparison tool."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from labs.serving_comparison import lifecycle as serving_lifecycle
from labs.serving_comparison import protocol as serving_protocol
from labs.serving_comparison.lifecycle import _terminate_process
from labs.serving_comparison.runner import run_comparison
from labs.serving_comparison.schema import ConfigError, load_profile

FIXTURE_SERVER = r"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--engine", required=True)
parser.add_argument("--architecture", required=True)
parser.add_argument("--runtime-version", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--connector")
parser.add_argument("--multi-token", action="store_true")
parser.add_argument("--identity-delay-s", type=float, default=0.0)
parser.add_argument("--hang-identity-after-ready", action="store_true")
args = parser.parse_args()
started = time.monotonic()
identity_requests = {"/version": 0, "/v1/models": 0}

values = {
    "kv_transfer_bytes_total": 0.0,
    "kv_transfer_requests_total": 0.0,
    "kv_transfer_failures_total": 0.0,
    "kv_transfer_time_seconds_total": 0.0,
    "prefill_requests_total": 0.0,
    "decode_requests_total": 0.0,
    "queue_age_seconds": 0.001,
    "prefill_pool_idle_fraction": 0.25,
    "decode_pool_idle_fraction": 0.5,
}
text_prompt_ids = {
    "fixture completed": [1, 2, 3],
    "fixture failed": [98],
    "fixture cancelled": [99],
    "fixture mismatch": [7],
}

class Server(ThreadingHTTPServer):
    allow_reuse_address = True

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *message_args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/hang":
            time.sleep(10)
            self.send_json({"ready": False}, 503)
        elif self.path == "/health":
            self.send_json({"ready": True})
        elif self.path == "/version":
            identity_requests[self.path] += 1
            if args.hang_identity_after_ready and identity_requests[self.path] > 1:
                time.sleep(10)
            if time.monotonic() - started < args.identity_delay_s:
                self.send_json({"ready": False}, 503)
            else:
                self.send_json({"version": args.runtime_version})
        elif self.path == "/v1/models":
            identity_requests[self.path] += 1
            if args.hang_identity_after_ready and identity_requests[self.path] > 1:
                time.sleep(10)
            if time.monotonic() - started < args.identity_delay_s:
                self.send_json({"ready": False}, 503)
            else:
                self.send_json({"data": [{"id": args.model}]})
        elif self.path == "/metrics-json" and args.architecture == "prefill_decode":
            self.send_json({
                "schema_version": "serving-comparison.telemetry.v1",
                "engine": args.engine,
                "architecture": args.architecture,
                "connector": args.connector,
                "values": values,
            })
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        wire_prompt = request["prompt"]
        prompt = text_prompt_ids[wire_prompt] if isinstance(wire_prompt, str) else wire_prompt
        if args.architecture == "prefill_decode":
            values["kv_transfer_bytes_total"] += 256
            values["kv_transfer_requests_total"] += 1
            values["kv_transfer_time_seconds_total"] += 0.002
            values["prefill_requests_total"] += 1
            values["decode_requests_total"] += 1
        if prompt[0] == 98:
            self.send_json({"error": "planned fixture failure"}, 503)
            return
        if prompt[0] == 99:
            time.sleep(0.25)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        events = []
        if args.multi_token:
            events.append({
                "choices": [{
                    "index": 0,
                    "text": "ab",
                    "token_ids": [101, 102],
                    "prompt_token_ids": prompt,
                    "finish_reason": "length",
                }]
            })
        else:
            events.extend([
                {"choices": [{
                    "index": 0,
                    "text": "a",
                    "token_ids": [101],
                    "prompt_token_ids": prompt,
                    "finish_reason": None,
                }]},
                {"choices": [{
                    "index": 0,
                    "text": "b",
                    "token_ids": [102],
                    "finish_reason": "length",
                }]},
            ])
        events.append({
            "choices": [],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": 2,
                "total_tokens": len(prompt) + 2,
            },
        })
        try:
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.002)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

Server(("127.0.0.1", args.port), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_inputs(
    tmp_path: Path,
    *,
    multi_token: bool = False,
    identity_delay_s: float = 0.0,
    hang_identity_after_ready: bool = False,
    prompt_transport: str = "token_ids",
    prompt_mismatch: bool = False,
) -> tuple[Path, Path]:
    port = _free_port()
    server = tmp_path / "fixture_server.py"
    server.write_text(FIXTURE_SERVER, encoding="utf-8")
    model = "fixture/model"
    version = "fixture-1.0"
    gpu_ids = ["fixture-gpu-0", "fixture-gpu-1"]
    arms = []
    for engine in ("vllm", "sglang"):
        for architecture in ("monolithic", "prefill_decode"):
            arm_id = f"{engine}-{architecture}"
            connector = "NixlConnector" if engine == "vllm" else "NIXL"
            command = [
                sys.executable,
                str(server),
                "--port",
                str(port),
                "--engine",
                engine,
                "--architecture",
                architecture,
                "--runtime-version",
                version,
                "--model",
                model,
            ]
            if architecture == "prefill_decode":
                command.extend(["--connector", connector])
            if multi_token:
                command.append("--multi-token")
            if identity_delay_s:
                command.extend(["--identity-delay-s", str(identity_delay_s)])
            if hang_identity_after_ready:
                command.append("--hang-identity-after-ready")
            arm = {
                "arm_id": arm_id,
                "engine": engine,
                "architecture": architecture,
                "endpoint": f"http://127.0.0.1:{port}/v1/completions",
                "runtime": {"version": version, "build_id": "fixture-build"},
                "gpu_ids": gpu_ids,
                "identity": {
                    "probes": [
                        {
                            "url": f"http://127.0.0.1:{port}/v1/models",
                            "assertions": {"data.0.id": model},
                            "proves": ["model"],
                        },
                        {
                            "url": f"http://127.0.0.1:{port}/version",
                            "assertions": {"version": version},
                            "proves": ["runtime_version"],
                        },
                    ]
                },
                "lifecycle": {
                    "mode": "local_process",
                    "start_command": command,
                    "working_directory": ".",
                    "ready_urls": [
                        f"http://127.0.0.1:{port}/health",
                        f"http://127.0.0.1:{port}/v1/models",
                        f"http://127.0.0.1:{port}/version",
                    ],
                    "timeout_s": 5,
                    "shutdown_timeout_s": 2,
                },
                "telemetry_sources": [],
            }
            if architecture == "prefill_decode":
                provenance_path = tmp_path / f"{arm_id}-provenance.json"
                provenance_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "serving-comparison.pd-provenance.v1",
                            "engine": engine,
                            "architecture": "prefill_decode",
                            "request_path": "kv_handoff",
                            "connector": {"name": connector, "backend": "nixl"},
                            "proxy": {
                                "implementation": "aisp_vllm_pd_proxy"
                                if engine == "vllm"
                                else "sglang_model_gateway"
                            },
                            "runtime": {"version": version, "build_id": "fixture-build"},
                            "workload": {
                                "model": model,
                                "tokenizer": model,
                                "precision": "fixture",
                            },
                            "gpu_pools": {
                                "prefill": [gpu_ids[0]],
                                "decode": [gpu_ids[1]],
                            },
                            "endpoints": {
                                "prefill": f"http://127.0.0.1:{port}/prefill",
                                "decode": f"http://127.0.0.1:{port}/decode",
                                "router": f"http://127.0.0.1:{port}/v1/completions",
                            },
                            "manifest_digest": "sha256:" + "0" * 64,
                        }
                    ),
                    encoding="utf-8",
                )
                metrics = {
                    name: f"values.{name}"
                    for name in (
                        "kv_transfer_bytes_total",
                        "kv_transfer_requests_total",
                        "kv_transfer_failures_total",
                        "kv_transfer_time_seconds_total",
                        "prefill_requests_total",
                        "decode_requests_total",
                        "queue_age_seconds",
                        "prefill_pool_idle_fraction",
                        "decode_pool_idle_fraction",
                    )
                }
                arm.update(
                    {
                        "pd_provenance": provenance_path.name,
                        "telemetry_sources": [
                            {
                                "source_id": "fixture",
                                "format": "standard_json",
                                "url": f"http://127.0.0.1:{port}/metrics-json",
                                "metrics": metrics,
                            }
                        ],
                        "diagnostic_support": {
                            name: {"status": "measured"}
                            for name in (
                                "kv_transfer_bytes_total",
                                "queue_age_seconds",
                                "prefill_pool_idle_fraction",
                                "decode_pool_idle_fraction",
                            )
                        },
                    }
                )
            arms.append(arm)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "serving-comparison.profile.v1",
                "mode": "protocol_fixture",
                "workload": {
                    "workload_id": "fixture-workload",
                    "model": model,
                    "tokenizer": model,
                    "precision": "fixture",
                    "seed": 7,
                    "prompt_transport": prompt_transport,
                    "admission_max_concurrency": 2,
                },
                "correctness": {"policy": "per_request_golden", "max_token_mismatches": 0},
                "hardware": {
                    "gpu_budget": 2,
                    "gpus": [{"id": gpu_id} for gpu_id in gpu_ids],
                    "clocks_locked": False,
                },
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "trace.jsonl"
    requests = [
        {
            "schema_version": "serving-comparison.request-trace.v1",
            "request_id": "completed",
            "arrival_ms": 0,
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 2,
            "deadline_ms": 500,
            "expected_status": "completed",
            "expected_output_token_ids": [101, 102],
        },
        {
            "schema_version": "serving-comparison.request-trace.v1",
            "request_id": "failed",
            "arrival_ms": 5,
            "prompt_token_ids": [98],
            "max_tokens": 2,
            "deadline_ms": 500,
            "expected_status": "failed",
        },
        {
            "schema_version": "serving-comparison.request-trace.v1",
            "request_id": "cancelled",
            "arrival_ms": 10,
            "prompt_token_ids": [99],
            "max_tokens": 2,
            "deadline_ms": 500,
            "cancel_after_ms": 40,
            "expected_status": "cancelled",
        },
    ]
    if prompt_transport == "text_with_token_id_attestation":
        requests[0]["prompt_text"] = "fixture mismatch" if prompt_mismatch else "fixture completed"
        requests[1]["prompt_text"] = "fixture failed"
        requests[2]["prompt_text"] = "fixture cancelled"
    trace.write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
    return profile, trace


def test_protocol_fixture_runs_full_matrix_without_performance_claim(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path)
    output = tmp_path / "output"
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repository / "code") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "tools",
            "serving-compare",
            "--",
            "--profile",
            str(profile),
            "--trace",
            str(trace),
            "--output-dir",
            str(output),
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--ttft-ms",
            "1000",
            "--tpot-ms",
            "1000",
            "--max-itl-ms",
            "1000",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert result["status"] == "protocol_validated"
    assert result["comparison_valid"] is True
    assert result["valid_for_performance_claim"] is False
    assert result["publication_ready"] is False
    assert len(result["iterations"]) == 8
    assert result["execution_order"][0]["arm_ids"] == list(
        reversed(result["execution_order"][1]["arm_ids"])
    )
    statuses = {
        row["request_id"]: row["status"]
        for row in map(
            json.loads,
            (output / result["iterations"][0]["request_trace_artifact"])
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    assert statuses == {"completed": "completed", "failed": "failed", "cancelled": "cancelled"}
    assert all(
        signal["clock_domain"].startswith("collector_wall_clock:")
        and signal["scope"]["host"] == signal["clock_domain"].split(":", 1)[1]
        and signal["scope"]
        == {
            "host": signal["scope"]["host"],
            "workload_id": "fixture-workload",
        }
        and "source_artifact" in signal["evidence"]
        for signal in result["signals"]
    )
    pd_iterations = [
        item for item in result["iterations"] if item["architecture"] == "prefill_decode"
    ]
    assert all(
        item["pd_telemetry_delta"]["kv_transfer_requests_total"] > 0 for item in pd_iterations
    )
    assert all(
        item["diagnostics"]["kv_transfer_bytes_total"]["status"] == "measured"
        for item in pd_iterations
    )


def test_multi_token_sse_event_rejects_per_token_latency_claim(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path, multi_token=True)
    result = run_comparison(
        profile_path=profile,
        trace_path=trace,
        output_dir=tmp_path / "multi-output",
        repeats=2,
        warmups=0,
        ttft_ms=1000,
        tpot_ms=1000,
        max_itl_ms=1000,
    )

    assert result["status"] == "rejected"
    assert result["comparison_valid"] is False
    request = result["iterations"][0]["serving_trace"]["requests"][0]
    assert request["max_token_ids_per_sse_event"] == 2
    assert request["token_timing_scope"] == "observed_chunk_event_only"
    assert request["tpot_s"] is None
    assert any("TPOT and ITL are unmeasured" in reason for reason in result["rejections"])


def test_text_transport_rejects_nonroundtripping_prompt_ids(tmp_path: Path) -> None:
    profile, trace = _write_inputs(
        tmp_path,
        prompt_transport="text_with_token_id_attestation",
        prompt_mismatch=True,
    )

    result = run_comparison(
        profile_path=profile,
        trace_path=trace,
        output_dir=tmp_path / "text-mismatch-output",
        repeats=2,
        warmups=1,
        ttft_ms=1000,
        tpot_ms=1000,
        max_itl_ms=1000,
    )

    manifest = json.loads(
        (tmp_path / "text-mismatch-output/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["workload"]["prompt_transport"] == "text_with_token_id_attestation"
    assert result["status"] == "rejected"
    assert "expected completed, got failed" in result["rejections"][0]
    rejected_trace = (
        tmp_path
        / "text-mismatch-output/iterations/repeat-00/vllm-monolithic/warmup-00-requests.jsonl"
    )
    rows = [json.loads(line) for line in rejected_trace.read_text().splitlines()]
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "prompt_token_ids_missing_or_mismatched"


@pytest.mark.parametrize(
    "echoed_prompt_token_ids",
    [None, [9, 9]],
    ids=["cancel-before-first-sse", "mismatched-echo"],
)
def test_text_transport_rejects_cancel_without_exact_prompt_attestation(
    monkeypatch: pytest.MonkeyPatch,
    echoed_prompt_token_ids: list[int] | None,
) -> None:
    async def stalled_stream(
        _client: object,
        _arm: object,
        _profile: object,
        _request: object,
        _scheduled: float,
        _admitted: float,
        progress: dict[str, object],
    ) -> None:
        progress["echoed_prompt_token_ids"] = echoed_prompt_token_ids
        await asyncio.sleep(1)

    monkeypatch.setattr(serving_protocol, "_stream_one", stalled_stream)
    request = SimpleNamespace(
        request_id="cancelled",
        arrival_ms=0,
        prompt_token_ids=(1, 2),
        cancel_after_ms=10,
        deadline_ms=1000,
        expected_status="cancelled",
    )

    observation = asyncio.run(
        serving_protocol._run_scheduled(
            object(),
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(prompt_transport="text_with_token_id_attestation"),
            request,  # type: ignore[arg-type]
            time.monotonic(),
            asyncio.Semaphore(1),
        )
    )

    assert observation.status == "failed"
    assert observation.error == "prompt_token_ids_missing_or_mismatched_before_cancel"
    assert observation.cancellation == {
        "mechanism": "client_stream_close",
        "server_cancel_confirmation": False,
        "trigger_ms": 10,
    }


def test_text_transport_accepts_cancel_after_exact_prompt_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stalled_stream(
        _client: object,
        _arm: object,
        _profile: object,
        request: object,
        _scheduled: float,
        _admitted: float,
        progress: dict[str, object],
    ) -> None:
        progress["echoed_prompt_token_ids"] = list(request.prompt_token_ids)  # type: ignore[attr-defined]
        await asyncio.sleep(1)

    monkeypatch.setattr(serving_protocol, "_stream_one", stalled_stream)
    request = SimpleNamespace(
        request_id="cancelled",
        arrival_ms=0,
        prompt_token_ids=(1, 2),
        cancel_after_ms=10,
        deadline_ms=1000,
        expected_status="cancelled",
    )

    observation = asyncio.run(
        serving_protocol._run_scheduled(
            object(),
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(prompt_transport="text_with_token_id_attestation"),
            request,  # type: ignore[arg-type]
            time.monotonic(),
            asyncio.Semaphore(1),
        )
    )

    assert observation.status == "cancelled"
    assert observation.error == "client_cancelled_stream"
    assert observation.echoed_prompt_token_ids == [1, 2]


def test_early_health_waits_for_readiness_and_complete_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, trace = _write_inputs(tmp_path, identity_delay_s=0.2)
    original_active_allocation = serving_lifecycle._active_allocation
    allocation_attempts = 0

    def delayed_allocation(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal allocation_attempts
        allocation_attempts += 1
        if allocation_attempts == 1:
            raise serving_lifecycle._AllocationPendingError("fixed fleet is still starting")
        return original_active_allocation(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(serving_lifecycle, "_active_allocation", delayed_allocation)

    result = run_comparison(
        profile_path=profile,
        trace_path=trace,
        output_dir=tmp_path / "delayed-readiness-output",
        repeats=2,
        warmups=0,
        ttft_ms=1000,
        tpot_ms=1000,
        max_itl_ms=1000,
    )

    assert result["status"] == "protocol_validated"
    assert result["comparison_valid"] is True
    assert allocation_attempts == 9


def test_readiness_request_cannot_outlive_startup_deadline(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    arm = raw["arms"][0]
    port = arm["lifecycle"]["ready_urls"][0].split(":")[2].split("/")[0]
    arm["lifecycle"]["ready_urls"] = [f"http://127.0.0.1:{port}/hang"]
    arm["lifecycle"]["timeout_s"] = 0.3
    profile.write_text(json.dumps(raw), encoding="utf-8")

    started = time.monotonic()
    result = run_comparison(
        profile_path=profile,
        trace_path=trace,
        output_dir=tmp_path / "hanging-readiness-output",
        repeats=2,
        warmups=0,
        ttft_ms=1000,
        tpot_ms=1000,
        max_itl_ms=1000,
    )

    assert time.monotonic() - started < 2
    assert result["status"] == "rejected"
    assert "did not become ready" in result["rejections"][0]
    cleanup = (
        tmp_path
        / "hanging-readiness-output/iterations/repeat-00/vllm-monolithic/startup-cleanup.json"
    )
    assert json.loads(cleanup.read_text(encoding="utf-8"))["cleanup_complete"] is True


def test_identity_request_has_bounded_control_plane_timeout(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path, hang_identity_after_ready=True)

    started = time.monotonic()
    result = run_comparison(
        profile_path=profile,
        trace_path=trace,
        output_dir=tmp_path / "hanging-identity-output",
        repeats=2,
        warmups=0,
        ttft_ms=1000,
        tpot_ms=1000,
        max_itl_ms=1000,
        connect_timeout_s=0.2,
    )

    assert time.monotonic() - started < 2
    assert result["status"] == "rejected"


def test_application_clock_evidence_rejects_declared_only_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = SimpleNamespace(
        gpu_ids=("0", "1"),
        application_clocks_mhz={"0": 1965, "1": 1965},
    )
    monkeypatch.setattr(
        serving_lifecycle,
        "_gpu_uuid_map",
        lambda: {"0": "gpu-a", "1": "gpu-b", "gpu-a": "gpu-a", "gpu-b": "gpu-b"},
    )
    monkeypatch.setattr(
        serving_lifecycle,
        "_run_nvidia_smi",
        lambda _args: "gpu-a, 1965\ngpu-b, 1900\n",
    )

    with pytest.raises(ConfigError, match="application clock mismatch"):
        serving_lifecycle._application_clock_evidence(profile)


def test_stop_arm_closes_logs_and_checks_idle_after_termination_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle_check_called = False

    async def fail_termination(*_args: object) -> None:
        raise ConfigError("forced termination failure")

    def record_idle_check(_profile: object) -> dict[str, object]:
        nonlocal idle_check_called
        idle_check_called = True
        return {"processes": []}

    monkeypatch.setattr(serving_lifecycle, "_terminate_process", fail_termination)
    monkeypatch.setattr(serving_lifecycle, "require_fleet_idle", record_idle_check)
    stdout_handle = (tmp_path / "server.stdout.log").open("wb")
    stderr_handle = (tmp_path / "server.stderr.log").open("wb")
    active = serving_lifecycle.ActiveProcess(
        process=SimpleNamespace(pid=1234, returncode=None),  # type: ignore[arg-type]
        process_group_id=1234,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        allocation_evidence={},
    )
    arm = SimpleNamespace(arm_id="fixture-arm", lifecycle={"shutdown_timeout_s": 0.1})

    with pytest.raises(ConfigError, match="process termination failed: forced"):
        asyncio.run(
            serving_lifecycle.stop_arm(
                active,
                arm,
                SimpleNamespace(),  # type: ignore[arg-type]
            )
        )

    assert stdout_handle.closed
    assert stderr_handle.closed
    assert idle_check_called is True


def test_profile_rejects_whole_request_routing_as_pd(tmp_path: Path) -> None:
    profile, _ = _write_inputs(tmp_path)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    pd_arm = next(arm for arm in raw["arms"] if arm["architecture"] == "prefill_decode")
    provenance = tmp_path / pd_arm["pd_provenance"]
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["request_path"] = "whole_request_routing"
    provenance.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ConfigError, match="Whole-request routing is not P/D"):
        load_profile(profile)


def test_profile_rejects_missing_required_pd_metric(tmp_path: Path) -> None:
    profile, _ = _write_inputs(tmp_path)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    pd_arm = next(arm for arm in raw["arms"] if arm["architecture"] == "prefill_decode")
    del pd_arm["telemetry_sources"][0]["metrics"]["kv_transfer_time_seconds_total"]
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="lacks required live P/D metrics"):
        load_profile(profile)


def test_termination_kills_owned_group_after_root_exits(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    launcher = tmp_path / "launch_child.py"
    launcher.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import os",
                "from pathlib import Path",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                f"Path({str(child_pid_path)!r}).write_text(str(child.pid))",
                "os._exit(0)",
            ]
        ),
        encoding="utf-8",
    )

    async def exercise() -> int:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(launcher), start_new_session=True
        )
        process_group_id = process.pid
        await process.wait()
        deadline = time.monotonic() + 2
        while not child_pid_path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        await _terminate_process(process, process_group_id, 1)
        return child_pid

    child_pid = asyncio.run(exercise())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("owned child process remained alive after group termination")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    manifest = output / "manifest.json"
    summary = output / "summary.json"
    manifest.write_bytes(b"old manifest bytes\n")
    summary.write_bytes(b"old summary bytes\n")

    with pytest.raises(ConfigError, match="already exists"):
        run_comparison(
            profile_path=profile,
            trace_path=trace,
            output_dir=output,
            repeats=2,
            warmups=0,
            ttft_ms=1000,
            tpot_ms=1000,
            max_itl_ms=1000,
        )

    assert manifest.read_bytes() == b"old manifest bytes\n"
    assert summary.read_bytes() == b"old summary bytes\n"


def test_nonfinite_threshold_is_rejected_before_output(tmp_path: Path) -> None:
    profile, trace = _write_inputs(tmp_path)
    output = tmp_path / "nan-output"

    with pytest.raises(ConfigError, match="finite"):
        run_comparison(
            profile_path=profile,
            trace_path=trace,
            output_dir=output,
            repeats=2,
            warmups=0,
            ttft_ms=float("nan"),
            tpot_ms=1000,
            max_itl_ms=1000,
        )

    assert not output.exists()


def test_identity_claim_must_bind_to_pinned_value(tmp_path: Path) -> None:
    profile, _ = _write_inputs(tmp_path)
    raw = json.loads(profile.read_text(encoding="utf-8"))
    probe = raw["arms"][0]["identity"]["probes"][0]
    probe["assertions"] = {"unrelated": ["value"]}
    profile.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="not bound to its pinned assertion value"):
        load_profile(profile)


def test_vllm_pd_proxy_performs_real_handoff_and_fails_closed(tmp_path: Path) -> None:
    state: dict[str, object] = {
        "prefill_mode": "ok",
        "prefill_payloads": [],
        "decode_payloads": [],
    }

    class UpstreamServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

        def __init__(self, address: tuple[str, int], role: str):
            super().__init__(address, Handler)
            self.role = role

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, message_format: str, *args: object) -> None:
            return

        def send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_json({"ready": True})
            elif self.path == "/version":
                self.send_json({"version": "fixture"})
            elif self.path == "/v1/models":
                self.send_json({"data": [{"id": "fixture/model"}]})
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            role = self.server.role  # type: ignore[attr-defined]
            if role == "prefill":
                state["prefill_payloads"].append(payload)  # type: ignore[union-attr]
                if state["prefill_mode"] == "failed":
                    self.send_json({"error": "planned"}, 500)
                elif state["prefill_mode"] == "missing":
                    self.send_json({"choices": []})
                else:
                    self.send_json(
                        {
                            "choices": [],
                            "kv_transfer_params": {
                                "do_remote_prefill": True,
                                "do_remote_decode": False,
                                "remote_engine_id": "fixture-prefill",
                                "remote_block_ids": [3, 4],
                            },
                        }
                    )
                return
            state["decode_payloads"].append(payload)  # type: ignore[union-attr]
            body = (
                b'data: {"choices":[{"index":0,"text":"ok","token_ids":[201],'
                b'"prompt_token_ids":[1,2],"finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":2,'
                b'"completion_tokens":1,"total_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    prefill_port = _free_port()
    decode_port = _free_port()
    proxy_port = _free_port()
    prefill_server = UpstreamServer(("127.0.0.1", prefill_port), "prefill")
    decode_server = UpstreamServer(("127.0.0.1", decode_port), "decode")
    threads = [
        threading.Thread(target=prefill_server.serve_forever, daemon=True),
        threading.Thread(target=decode_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "code")
    proxy = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "labs.serving_comparison.vllm_pd_proxy",
            "--port",
            str(proxy_port),
            "--prefill-url",
            f"http://127.0.0.1:{prefill_port}",
            "--decode-url",
            f"http://127.0.0.1:{decode_port}",
            "--timeout-s",
            "2",
        ],
        cwd=repository,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def post(prompt: str | list[int] = "fixture prompt") -> tuple[int, bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/completions",
            data=json.dumps(
                {
                    "model": "fixture/model",
                    "prompt": prompt,
                    "max_tokens": 7,
                    "temperature": 0,
                    "stream": True,
                    "return_token_ids": True,
                    "stream_options": {"include_usage": True},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, response.read()

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{proxy_port}/health", timeout=0.2
                ) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            pytest.fail("vLLM P/D proxy did not become ready")

        status, body = post()
        assert status == 200
        assert b"data: [DONE]" in body
        prefill_payload = state["prefill_payloads"][0]  # type: ignore[index]
        decode_payload = state["decode_payloads"][0]  # type: ignore[index]
        assert prefill_payload["stream"] is False
        assert prefill_payload["max_tokens"] == 1
        assert prefill_payload["min_tokens"] == 1
        assert prefill_payload["kv_transfer_params"] == {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        assert prefill_payload["prompt"] == "fixture prompt"
        assert decode_payload["prompt"] == "fixture prompt"
        assert decode_payload["max_tokens"] == 7
        assert decode_payload["kv_transfer_params"]["do_remote_prefill"] is True

        status, body = post([1, 2])
        assert status == 200
        assert b"data: [DONE]" in body
        assert state["prefill_payloads"][-1]["prompt"] == [1, 2]  # type: ignore[index]
        assert state["decode_payloads"][-1]["prompt"] == [1, 2]  # type: ignore[index]

        for mode in ("missing", "failed"):
            state["prefill_mode"] = mode
            started = time.monotonic()
            with pytest.raises(urllib.error.HTTPError) as error:
                post()
            assert error.value.code == 502
            assert time.monotonic() - started < 1
    finally:
        proxy.terminate()
        proxy.wait(timeout=5)
        prefill_server.shutdown()
        decode_server.shutdown()
        prefill_server.server_close()
        decode_server.server_close()


def test_shipped_engine_profile_and_launch_digests_are_valid() -> None:
    repository = Path(__file__).resolve().parents[2]
    examples = repository / "code/labs/serving_comparison/examples"
    profile = load_profile(examples / "profile.engine.template.json")

    assert len(profile.arms) == 4
    assert profile.model == "Qwen/Qwen3-8B"
    assert profile.prompt_transport == "text_with_token_id_attestation"
    assert all(Path(arm.lifecycle["working_directory"]) == repository for arm in profile.arms)
    for engine in ("vllm", "sglang"):
        launch = examples / f"{engine}-pd-launch.json"
        provenance = json.loads(
            (examples / f"{engine}-pd-provenance.json").read_text(encoding="utf-8")
        )
        assert (
            provenance["manifest_digest"]
            == "sha256:" + hashlib.sha256(launch.read_bytes()).hexdigest()
        )

    arms = {arm.arm_id: arm for arm in profile.arms}
    assert arms["vllm-monolithic"].runtime_version == "0.26.0"
    assert "--max-num-queued-reqs" not in arms["vllm-monolithic"].lifecycle["start_command"]
    assert arms["sglang-monolithic"].runtime_version == "0.5.20"
    sglang_pd = arms["sglang-prefill-decode"]
    assert "http://127.0.0.1:8200/readiness" in sglang_pd.lifecycle["ready_urls"]
    assert {probe["url"] for probe in sglang_pd.identity_probes} >= {
        "http://127.0.0.1:8201/v1/models",
        "http://127.0.0.1:8202/v1/models",
    }
    sources = {source.source_id: source for source in sglang_pd.telemetry_sources}
    assert "kv_transfer_requests_total" in sources["prefill-native"].metrics
    assert "kv_transfer_requests_total" not in sources["decode-native"].metrics
    assert sources["prefill-native"].metrics["kv_transfer_failures_total"]["missing_value"] == 0
    assert sources["prefill-native"].metrics["kv_transfer_bytes_total"] == {
        "name": "sglang:kv_transfer_total_mb_sum",
        "reduce": "sum",
        "scale": 1048576,
    }
    launch = json.loads((examples / "sglang-pd-launch.json").read_text(encoding="utf-8"))
    engines = launch["processes"][:2]
    assert all(
        process["environment"]["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
        for process in engines
    )
    assert "--pd-disaggregation" in launch["processes"][2]["command"]
    vllm_launch = json.loads((examples / "vllm-pd-launch.json").read_text(encoding="utf-8"))
    connector_configs = [
        json.loads(process["command"][process["command"].index("--kv-transfer-config") + 1])
        for process in vllm_launch["processes"][:2]
    ]
    assert [config["kv_role"] for config in connector_configs] == [
        "kv_producer",
        "kv_consumer",
    ]
    trace = [
        json.loads(line)
        for line in (examples / "request_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cancelled = next(row for row in trace if row["expected_status"] == "cancelled")
    assert cancelled["cancel_after_ms"] == 2500
    assert cancelled["max_tokens"] == 2048
