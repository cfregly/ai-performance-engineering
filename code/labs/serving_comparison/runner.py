"""Run the fixed-fleet vLLM and SGLang serving comparison matrix."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.analysis.serving_trace import summarize

from . import protocol
from .lifecycle import ActiveProcess, start_arm, stop_arm
from .schema import (
    OPTIONAL_PD_DIAGNOSTICS,
    RESULT_SCHEMA,
    ArmProfile,
    ComparisonProfile,
    ConfigError,
    RequestSpec,
    load_profile,
    load_trace,
)
from .telemetry import TelemetrySnapshot, capture_pd_telemetry, compute_pd_delta


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _profile_manifest(profile: ComparisonProfile, trace_path: Path) -> dict[str, Any]:
    collector_host = hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]

    def safe_environment(environment: dict[str, str]) -> dict[str, str]:
        sensitive = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")
        return {
            name: "<redacted>" if any(marker in name.upper() for marker in sensitive) else value
            for name, value in environment.items()
        }

    return {
        "schema_version": "serving-comparison.manifest.v1",
        "created_unix_s": time.time(),
        "git_commit": _git_commit(),
        "mode": profile.mode,
        "collector": {
            "host": collector_host,
            "clock_domain": f"collector_wall_clock:{collector_host}",
        },
        "source_files": {
            "profile": {
                "path": str(profile.source_path),
                "sha256": _sha256(profile.source_path),
            },
            "request_trace": {
                "path": str(trace_path.resolve()),
                "sha256": _sha256(trace_path.resolve()),
            },
        },
        "workload": {
            "workload_id": profile.workload_id,
            "model": profile.model,
            "tokenizer": profile.tokenizer,
            "precision": profile.precision,
            "seed": profile.seed,
            "prompt_transport": profile.prompt_transport,
            "sampling": {"temperature": 0, "top_p": 1, "n": 1},
            "admission_max_concurrency": profile.admission_max_concurrency,
        },
        "correctness": {
            "policy": profile.correctness_policy,
            "max_token_mismatches": profile.max_token_mismatches,
        },
        "hardware": {
            "gpu_budget": profile.gpu_budget,
            "gpu_ids": list(profile.gpu_ids),
            "clocks_locked": profile.clocks_locked,
            "application_clocks_mhz": profile.application_clocks_mhz,
        },
        "arms": [
            {
                "arm_id": arm.arm_id,
                "engine": arm.engine,
                "architecture": arm.architecture,
                "endpoint": arm.endpoint,
                "runtime_version": arm.runtime_version,
                "runtime_build_id": arm.runtime_build_id,
                "gpu_ids": list(arm.gpu_ids),
                "lifecycle_mode": arm.lifecycle["mode"],
                "start_command": arm.lifecycle["start_command"],
                "working_directory": arm.lifecycle["working_directory"],
                "environment": safe_environment(arm.lifecycle["environment"]),
                "env_passthrough": arm.lifecycle["env_passthrough"],
                "pd_provenance": None
                if arm.pd_provenance_path is None
                else {
                    "path": str(arm.pd_provenance_path),
                    "sha256": _sha256(arm.pd_provenance_path),
                    "request_path": arm.pd_provenance["request_path"],
                    "connector": arm.pd_provenance["connector"],
                    "proxy": arm.pd_provenance["proxy"],
                    "manifest_digest": arm.pd_provenance["manifest_digest"],
                    "diagnostic_support": arm.pd_provenance["diagnostic_support"],
                },
            }
            for arm in profile.arms
        ],
        "timing_semantics": {
            "clock_domain": "load_generator_monotonic_mapped_to_unix",
            "token_timestamp": "client_observed_sse_event",
            "multiple_token_ids_in_one_event": "same_observed_timestamp",
            "text_only_chunks": "never converted to token timestamps",
        },
        "lifecycle_boundary": (
            "Each arm is activated alone. Controller evidence must show every other arm inactive, "
            "the exact GPU fleet assigned, and the fleet idle after deactivation."
        ),
    }


def _dotted_value(document: Any, path: str) -> Any:
    current = document
    for component in path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit() and int(component) < len(current):
            current = current[int(component)]
        else:
            raise ConfigError(f"identity response does not contain {path}")
    return current


async def _verify_identity(
    client: Any, arm: ArmProfile, control_timeout_s: float
) -> list[dict[str, Any]]:
    headers: dict[str, str] = {}
    if arm.api_key_env:
        value = os.environ.get(arm.api_key_env)
        if not value:
            raise ConfigError(f"required API key environment variable is unset: {arm.api_key_env}")
        headers["Authorization"] = f"Bearer {value}"
    evidence: list[dict[str, Any]] = []
    for probe in arm.identity_probes:
        response = await client.get(probe["url"], headers=headers, timeout=control_timeout_s)
        response.raise_for_status()
        try:
            document = response.json()
        except ValueError as exc:
            raise ConfigError(f"{arm.arm_id} identity probe did not return JSON") from exc
        for path, expected in probe["assertions"].items():
            if _dotted_value(document, path) != expected:
                raise ConfigError(f"{arm.arm_id} identity mismatch for {path}")
        evidence.append(
            {
                "url": probe["url"],
                "assertions": probe["assertions"],
                "proves": probe["proves"],
                "response_sha256": hashlib.sha256(response.content).hexdigest(),
            }
        )
    return evidence


def _status_rejections(
    requests: tuple[RequestSpec, ...], observations: tuple[protocol.RequestObservation, ...]
) -> list[str]:
    expected = {request.request_id: request.expected_status for request in requests}
    return [
        f"{observation.request_id}: expected {expected[observation.request_id]}, got {observation.status}"
        for observation in observations
        if observation.status != expected[observation.request_id]
    ]


def _token_mismatch_count(left: list[int], right: list[int]) -> int:
    overlap = min(len(left), len(right))
    return sum(left[index] != right[index] for index in range(overlap)) + abs(
        len(left) - len(right)
    )


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metric_summary(trace_summary: dict[str, Any]) -> dict[str, Any]:
    ttft = [row["ttft_s"] for row in trace_summary["requests"] if row["ttft_s"] is not None]
    tpot = [row["tpot_s"] for row in trace_summary["requests"] if row["tpot_s"] is not None]
    return {
        "goodput_tokens_per_second": trace_summary["goodput_tokens_per_second"],
        "goodput_requests_per_second": trace_summary["goodput_requests_per_second"],
        "slo_attainment_fraction": trace_summary["slo_attainment_fraction"],
        "completed_requests": trace_summary["completed_requests"],
        "failed_requests": trace_summary["failed_requests"],
        "cancelled_requests": trace_summary["cancelled_requests"],
        "ttft_p50_s": _percentile(ttft, 0.5),
        "ttft_p95_s": _percentile(ttft, 0.95),
        "tpot_p50_s": _percentile(tpot, 0.5),
        "tpot_p95_s": _percentile(tpot, 0.95),
    }


def _serving_summary(
    replay: protocol.ReplayResult,
    *,
    max_ttft_s: float,
    max_tpot_s: float,
    max_itl_s: float | None,
) -> tuple[dict[str, Any], list[str]]:
    summary = summarize(
        [item.to_serving_trace_record() for item in replay.observations],
        max_ttft_s=max_ttft_s,
        max_tpot_s=max_tpot_s,
        max_itl_s=max_itl_s,
    )
    observations = {item.request_id: item for item in replay.observations}
    timing_rejections: list[str] = []
    for row in summary["requests"]:
        observation = observations[row["request_id"]]
        maximum = max(
            (int(event.get("token_id_count", 0)) for event in observation.events), default=0
        )
        row["max_token_ids_per_sse_event"] = maximum
        if maximum <= 1:
            row["token_timing_scope"] = "observed_per_token_event"
            continue
        row["token_timing_scope"] = "observed_chunk_event_only"
        row["tpot_s"] = None
        row["max_itl_s"] = None
        row["meets_slo"] = False
        timing_rejections.append(
            f"{row['request_id']}: an SSE event contained {maximum} token ids, so TPOT and ITL are unmeasured"
        )
    if timing_rejections:
        good = [row for row in summary["requests"] if row["meets_slo"]]
        elapsed = summary["observation_seconds"]
        summary["slo_attainment_fraction"] = len(good) / len(summary["requests"])
        summary["goodput_requests_per_second"] = len(good) / elapsed
        summary["goodput_tokens_per_second"] = sum(row["emitted_tokens"] for row in good) / elapsed
    return summary, timing_rejections


def _snapshot_payload(snapshot: TelemetrySnapshot) -> dict[str, Any]:
    return {
        "captured_monotonic_s": snapshot.captured_monotonic_s,
        "captured_unix_s": snapshot.captured_unix_s,
        "values": snapshot.values,
        "selector_evidence_by_source": snapshot.selector_evidence_by_source,
    }


def _diagnostic_result(
    arm: ArmProfile, before: TelemetrySnapshot, after: TelemetrySnapshot
) -> dict[str, Any]:
    support = arm.pd_provenance["diagnostic_support"] if arm.pd_provenance else {}
    result: dict[str, Any] = {
        "kv_transfer": {
            "status": "measured",
            "metrics": [
                "kv_transfer_requests_total",
                "kv_transfer_failures_total",
                "kv_transfer_time_seconds_total",
            ],
        },
    }
    for diagnostic in sorted(OPTIONAL_PD_DIAGNOSTICS):
        declaration = support[diagnostic]
        if declaration["status"] == "measured":
            value = after.values[diagnostic]
            if diagnostic == "kv_transfer_bytes_total":
                value -= before.values[diagnostic]
                if value <= 0:
                    raise ConfigError("measured KV transfer bytes did not increase")
            result[diagnostic] = {"status": "measured", "value": value}
        else:
            result[diagnostic] = {
                "status": "unsupported",
                "reason": declaration["reason"],
            }
    if all(
        result[name]["status"] == "measured"
        for name in ("prefill_pool_idle_fraction", "decode_pool_idle_fraction")
    ):
        result["pool_idle_imbalance"] = {
            "status": "measured",
            "value": abs(
                result["prefill_pool_idle_fraction"]["value"]
                - result["decode_pool_idle_fraction"]["value"]
            ),
        }
    else:
        result["pool_idle_imbalance"] = {
            "status": "unsupported",
            "reason": "both pool idle fractions are required",
        }
    return result


def _iteration_signals(
    *,
    arm: ArmProfile,
    repeat_index: int,
    profile: ComparisonProfile,
    replay: protocol.ReplayResult,
    trace_summary: dict[str, Any],
    trace_artifact: Path,
    pd_delta: dict[str, float] | None,
    diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    start_unix = replay.started_unix_s
    end_unix = replay.finished_monotonic_s + replay.wall_offset_s
    path = str(trace_artifact)
    collector_host = hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]
    clock_domain = f"collector_wall_clock:{collector_host}"
    scope = {"host": collector_host, "workload_id": profile.workload_id}
    artifact_evidence = {"source_artifact": path}
    signals = [
        {
            "metric": "serving.goodput_tokens_per_second",
            "value": trace_summary["goodput_tokens_per_second"],
            "unit": "tokens_per_second",
            "start_unix_s": start_unix,
            "end_unix_s": end_unix,
            "clock_domain": clock_domain,
            "scope": scope,
            "evidence": artifact_evidence,
            "role": "measurement",
        },
        {
            "metric": "serving.slo_attainment_fraction",
            "value": trace_summary["slo_attainment_fraction"],
            "unit": "fraction",
            "start_unix_s": start_unix,
            "end_unix_s": end_unix,
            "clock_domain": clock_domain,
            "scope": scope,
            "evidence": artifact_evidence,
            "role": "measurement",
        },
    ]
    observation_by_id = {item.request_id: item for item in replay.observations}
    for row in trace_summary["requests"]:
        if row["meets_slo"]:
            continue
        observation = observation_by_id[row["request_id"]]
        signals.append(
            {
                "metric": "serving.request.slo_violation",
                "value": 1,
                "unit": "count",
                "start_unix_s": observation.scheduled_arrival_s + replay.wall_offset_s,
                "end_unix_s": observation.finish_s + replay.wall_offset_s,
                "clock_domain": clock_domain,
                "scope": scope,
                "evidence": {**artifact_evidence, "request_id": observation.request_id},
                "role": "symptom",
            }
        )
    if pd_delta is not None:
        units = {
            "kv_transfer_bytes_total": "bytes",
            "kv_transfer_requests_total": "count",
            "kv_transfer_failures_total": "count",
            "kv_transfer_time_seconds_total": "seconds",
            "prefill_requests_total": "count",
            "decode_requests_total": "count",
        }
        for metric, value in sorted(pd_delta.items()):
            signals.append(
                {
                    "metric": f"serving.pd.{metric}",
                    "value": value,
                    "unit": units[metric],
                    "start_unix_s": start_unix,
                    "end_unix_s": end_unix,
                    "clock_domain": clock_domain,
                    "scope": scope,
                    "evidence": artifact_evidence,
                    "role": "measurement",
                }
            )
    if diagnostics is not None:
        diagnostic_units = {
            "kv_transfer_bytes_total": "bytes",
            "queue_age_seconds": "seconds",
            "prefill_pool_idle_fraction": "fraction",
            "decode_pool_idle_fraction": "fraction",
            "pool_idle_imbalance": "fraction",
        }
        for metric, unit in diagnostic_units.items():
            diagnostic_evidence = diagnostics[metric]
            if diagnostic_evidence["status"] != "measured":
                continue
            signals.append(
                {
                    "metric": f"serving.pd.{metric}",
                    "value": diagnostic_evidence["value"],
                    "unit": unit,
                    "start_unix_s": start_unix,
                    "end_unix_s": end_unix,
                    "clock_domain": clock_domain,
                    "scope": scope,
                    "evidence": artifact_evidence,
                    "role": "measurement",
                }
            )
    return signals


async def _run_replay(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    requests: tuple[RequestSpec, ...],
    artifact_path: Path,
) -> protocol.ReplayResult:
    replay = await protocol.replay_trace(client, arm, profile, requests)
    _write_jsonl(
        artifact_path,
        [item.to_dict(replay.wall_offset_s) for item in replay.observations],
    )
    rejections = _status_rejections(requests, replay.observations)
    if rejections:
        raise ConfigError(f"{arm.arm_id} request status mismatch: {rejections}")
    return replay


async def _run_arm_iteration(
    *,
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    requests: tuple[RequestSpec, ...],
    run_id: str,
    repeat_index: int,
    warmups: int,
    output_dir: Path,
    slo: dict[str, float | None],
    control_timeout_s: float,
) -> dict[str, Any]:
    active: ActiveProcess | None = None
    primary_error: BaseException | None = None
    try:
        relative_dir = Path("iterations") / f"repeat-{repeat_index:02d}" / arm.arm_id
        trace_relative = relative_dir / "requests.jsonl"
        trace_path = output_dir / trace_relative
        active = await start_arm(client, arm, profile, output_dir / relative_dir)
        allocation = active.allocation_evidence
        identity = await _verify_identity(client, arm, control_timeout_s)
        warmup_results: list[dict[str, Any]] = []
        for warmup_index in range(warmups):
            warmup_relative = relative_dir / f"warmup-{warmup_index:02d}-requests.jsonl"
            replay = await _run_replay(
                client,
                arm,
                profile,
                requests,
                output_dir / warmup_relative,
            )
            warmup_results.append(
                {
                    "warmup_index": warmup_index,
                    "duration_seconds": replay.finished_monotonic_s - replay.started_monotonic_s,
                    "statuses": {item.request_id: item.status for item in replay.observations},
                    "request_trace_artifact": str(warmup_relative),
                }
            )

        telemetry_dir = output_dir / relative_dir / "telemetry"

        def retain_raw_telemetry(phase: str) -> Callable[[str, str], None]:
            def retain(source_id: str, text: str) -> None:
                telemetry_dir.mkdir(parents=True, exist_ok=True)
                (telemetry_dir / f"{phase}-{source_id}.txt").write_text(text, encoding="utf-8")

            return retain

        before: TelemetrySnapshot | None = None
        if arm.architecture == "prefill_decode":
            before = await capture_pd_telemetry(
                client,
                arm,
                request_timeout_s=control_timeout_s,
                raw_sink=retain_raw_telemetry("before"),
            )
        replay = await _run_replay(client, arm, profile, requests, trace_path)
        after: TelemetrySnapshot | None = None
        pd_delta: dict[str, float] | None = None
        diagnostics: dict[str, Any] | None = None
        if arm.architecture == "prefill_decode":
            after = await capture_pd_telemetry(
                client,
                arm,
                request_timeout_s=control_timeout_s,
                raw_sink=retain_raw_telemetry("after"),
            )
            pd_delta = compute_pd_delta(before, after)  # type: ignore[arg-type]
            diagnostics = _diagnostic_result(arm, before, after)  # type: ignore[arg-type]

        if before is not None and after is not None:
            _write_json(telemetry_dir / "before.json", _snapshot_payload(before))
            _write_json(telemetry_dir / "after.json", _snapshot_payload(after))
            for source_id, text in before.raw_by_source.items():
                (telemetry_dir / f"before-{source_id}.txt").write_text(text, encoding="utf-8")
            for source_id, text in after.raw_by_source.items():
                (telemetry_dir / f"after-{source_id}.txt").write_text(text, encoding="utf-8")

        trace_summary, timing_rejections = _serving_summary(
            replay,
            max_ttft_s=float(slo["max_ttft_s"]),
            max_tpot_s=float(slo["max_tpot_s"]),
            max_itl_s=slo["max_itl_s"],
        )
        metrics = _metric_summary(trace_summary)
        summary_relative = relative_dir / "summary.json"
        signals = _iteration_signals(
            arm=arm,
            repeat_index=repeat_index,
            profile=profile,
            replay=replay,
            trace_summary=trace_summary,
            trace_artifact=trace_relative,
            pd_delta=pd_delta,
            diagnostics=diagnostics,
        )
        result = {
            "arm_id": arm.arm_id,
            "engine": arm.engine,
            "architecture": arm.architecture,
            "prompt_transport": profile.prompt_transport,
            "repeat_index": repeat_index,
            "allocation_evidence": allocation,
            "pd_launch_binding": allocation.get("pd_launch_binding"),
            "identity": identity,
            "warmups": warmup_results,
            "request_trace_artifact": str(trace_relative),
            "metrics": metrics,
            "serving_trace": trace_summary,
            "pd_telemetry_delta": pd_delta,
            "diagnostics": diagnostics,
            "signals": signals,
            "rejections": timing_rejections,
        }
        _write_json(output_dir / summary_relative, result)
        result["summary_artifact"] = str(summary_relative)
        return result
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if active is not None:
            try:
                released = await stop_arm(active, arm, profile)
                _write_json(
                    output_dir
                    / "iterations"
                    / f"repeat-{repeat_index:02d}"
                    / arm.arm_id
                    / "released-allocation.json",
                    released,
                )
            except Exception as cleanup_error:
                if primary_error is not None:
                    raise ConfigError(
                        f"{arm.arm_id} failed and its fixed GPU fleet release also failed: "
                        f"{cleanup_error}"
                    ) from primary_error
                raise


def _correctness_rejections(
    profile: ComparisonProfile,
    requests: tuple[RequestSpec, ...],
    iterations: list[dict[str, Any]],
) -> list[str]:
    completed_ids = {
        request.request_id for request in requests if request.expected_status == "completed"
    }
    outputs: dict[tuple[str, int, str], list[int]] = {}
    for iteration in iterations:
        trace_path = Path(iteration["_output_dir"]) / iteration["request_trace_artifact"]
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["request_id"] in completed_ids:
                outputs[(iteration["arm_id"], iteration["repeat_index"], row["request_id"])] = row[
                    "output_token_ids"
                ]
    rejections: list[str] = []
    if profile.correctness_policy == "per_request_golden":
        golden = {
            request.request_id: request.expected_output_token_ids
            for request in requests
            if request.request_id in completed_ids
        }
        missing = sorted(request_id for request_id, tokens in golden.items() if tokens is None)
        if missing:
            return [f"per_request_golden lacks expected tokens for {missing}"]
        for (arm_id, repeat_index, request_id), actual in outputs.items():
            mismatch = _token_mismatch_count(actual, list(golden[request_id] or ()))
            if mismatch > profile.max_token_mismatches:
                rejections.append(
                    f"{arm_id} repeat {repeat_index} request {request_id} has {mismatch} "
                    "token mismatches from its golden output"
                )
    else:
        reference_arm = next(
            arm.arm_id
            for arm in profile.arms
            if arm.engine == "vllm" and arm.architecture == "monolithic"
        )
        for request_id in completed_ids:
            reference = outputs[(reference_arm, 0, request_id)]
            for (arm_id, repeat_index, candidate_id), actual in outputs.items():
                if candidate_id != request_id:
                    continue
                mismatch = _token_mismatch_count(actual, reference)
                if mismatch > profile.max_token_mismatches:
                    rejections.append(
                        f"{arm_id} repeat {repeat_index} request {request_id} has {mismatch} "
                        "token mismatches from the vLLM monolithic reference"
                    )
    return rejections


def _aggregate(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for iteration in iterations:
        by_arm.setdefault(iteration["arm_id"], []).append(iteration["metrics"])
    per_arm: dict[str, Any] = {}
    for arm_id, metrics in by_arm.items():
        aggregate: dict[str, Any] = {"repeats": len(metrics)}
        for key in (
            "goodput_tokens_per_second",
            "goodput_requests_per_second",
            "slo_attainment_fraction",
            "ttft_p50_s",
            "ttft_p95_s",
            "tpot_p50_s",
            "tpot_p95_s",
        ):
            values = [float(item[key]) for item in metrics if item[key] is not None]
            aggregate[f"median_{key}"] = statistics.median(values) if values else None
        per_arm[arm_id] = aggregate
    effects: dict[str, Any] = {}
    for engine in ("vllm", "sglang"):
        mono = next(
            item["arm_id"]
            for item in iterations
            if item["engine"] == engine and item["architecture"] == "monolithic"
        )
        pd = next(
            item["arm_id"]
            for item in iterations
            if item["engine"] == engine and item["architecture"] == "prefill_decode"
        )
        mono_goodput = per_arm[mono]["median_goodput_tokens_per_second"]
        pd_goodput = per_arm[pd]["median_goodput_tokens_per_second"]
        effects[engine] = {
            "monolithic_arm_id": mono,
            "prefill_decode_arm_id": pd,
            "goodput_ratio_pd_over_monolithic": None
            if mono_goodput == 0
            else pd_goodput / mono_goodput,
            "slo_attainment_delta_pd_minus_monolithic": (
                per_arm[pd]["median_slo_attainment_fraction"]
                - per_arm[mono]["median_slo_attainment_fraction"]
            ),
        }
    return {"per_arm": per_arm, "architecture_effects": effects}


async def _execute(
    *,
    profile: ComparisonProfile,
    requests: tuple[RequestSpec, ...],
    trace_path: Path,
    output_dir: Path,
    repeats: int,
    warmups: int,
    slo: dict[str, float | None],
    connect_timeout_s: float,
    run_id: str,
) -> dict[str, Any]:
    if repeats < 2:
        raise ConfigError("repeats must be at least 2 for forward and reverse interleaving")
    if warmups < 0:
        raise ConfigError("warmups must be nonnegative")
    if output_dir.exists():
        raise ConfigError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = _profile_manifest(profile, trace_path)
    manifest["run_id"] = run_id
    manifest["repeats"] = repeats
    manifest["warmups"] = warmups
    manifest["slos"] = slo
    _write_json(output_dir / "manifest.json", manifest)

    httpx_module = protocol.require_httpx()
    timeout = httpx_module.Timeout(
        connect=connect_timeout_s,
        read=None,
        write=max(connect_timeout_s, 30.0),
        pool=max(connect_timeout_s, 30.0),
    )
    order = list(profile.arms)
    execution_order: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    start = time.monotonic()
    async with httpx_module.AsyncClient(timeout=timeout, trust_env=False) as client:
        for repeat_index in range(repeats):
            current_order = order if repeat_index % 2 == 0 else list(reversed(order))
            execution_order.append(
                {"repeat_index": repeat_index, "arm_ids": [arm.arm_id for arm in current_order]}
            )
            for arm in current_order:
                iteration = await _run_arm_iteration(
                    client=client,
                    arm=arm,
                    profile=profile,
                    requests=requests,
                    run_id=run_id,
                    repeat_index=repeat_index,
                    warmups=warmups,
                    output_dir=output_dir,
                    slo=slo,
                    control_timeout_s=connect_timeout_s,
                )
                iteration["_output_dir"] = str(output_dir)
                iterations.append(iteration)

    rejections = _correctness_rejections(profile, requests, iterations)
    rejections.extend(
        f"{iteration['arm_id']} repeat {iteration['repeat_index']}: {reason}"
        for iteration in iterations
        for reason in iteration["rejections"]
    )
    comparison_valid = not rejections
    signals = [signal for iteration in iterations for signal in iteration["signals"]]
    public_iterations = [
        {key: value for key, value in iteration.items() if key != "_output_dir"}
        for iteration in iterations
    ]
    status = (
        "protocol_validated"
        if comparison_valid and profile.mode == "protocol_fixture"
        else "complete"
        if comparison_valid
        else "rejected"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_id": run_id,
        "status": status,
        "comparison_valid": comparison_valid,
        "valid_for_performance_claim": False,
        "publication_ready": False,
        "mode": profile.mode,
        "manifest_artifact": "manifest.json",
        "duration_seconds": time.monotonic() - start,
        "execution_order": execution_order,
        "comparison": _aggregate(public_iterations),
        "iterations": public_iterations,
        "signals": signals,
        "rejections": rejections,
        "limitations": [
            "SSE timestamps are client observations. They are not server token production times.",
            "Token ids in one SSE event share one observed timestamp.",
            "Client stream closure records attempted cancellation without server confirmation.",
            "Publication requires separate canonical environment and profiler evidence review.",
        ],
    }
    _write_json(output_dir / "summary.json", result)
    return result


def run_comparison(
    *,
    profile_path: Path,
    trace_path: Path,
    output_dir: Path,
    repeats: int,
    warmups: int,
    ttft_ms: float,
    tpot_ms: float,
    max_itl_ms: float | None,
    connect_timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Validate inputs, replay every arm, and write a self-contained result tree."""
    if output_dir.exists():
        raise ConfigError(f"output directory already exists: {output_dir}")
    profile = load_profile(profile_path)
    requests = load_trace(trace_path)
    missing_prompt_text = [
        request.request_id for request in requests if request.prompt_text is None
    ]
    unexpected_prompt_text = [
        request.request_id for request in requests if request.prompt_text is not None
    ]
    if profile.prompt_transport == "text_with_token_id_attestation" and missing_prompt_text:
        raise ConfigError(
            "text_with_token_id_attestation requires prompt_text for every request: "
            f"{missing_prompt_text}"
        )
    if profile.prompt_transport == "token_ids" and unexpected_prompt_text:
        raise ConfigError(
            f"token_ids transport forbids prompt_text fields: {unexpected_prompt_text}"
        )
    for name, value in {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "connect_timeout_s": connect_timeout_s,
    }.items():
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"{name} must be finite and nonnegative")
    if connect_timeout_s <= 0:
        raise ConfigError("connect_timeout_s must be positive")
    if max_itl_ms is not None and (not math.isfinite(max_itl_ms) or max_itl_ms < 0):
        raise ConfigError("max_itl_ms must be finite and nonnegative")
    slo = {
        "max_ttft_s": ttft_ms / 1000.0,
        "max_tpot_s": tpot_ms / 1000.0,
        "max_itl_s": None if max_itl_ms is None else max_itl_ms / 1000.0,
    }
    run_id = f"serving-comparison-{uuid.uuid4()}"
    try:
        return asyncio.run(
            _execute(
                profile=profile,
                requests=requests,
                trace_path=trace_path,
                output_dir=output_dir,
                repeats=repeats,
                warmups=warmups,
                slo=slo,
                connect_timeout_s=connect_timeout_s,
                run_id=run_id,
            )
        )
    except (ConfigError, OSError, protocol.require_httpx().HTTPError) as exc:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            raise
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            raise
        result = {
            "schema_version": RESULT_SCHEMA,
            "run_id": manifest["run_id"],
            "status": "rejected",
            "comparison_valid": False,
            "valid_for_performance_claim": False,
            "publication_ready": False,
            "mode": profile.mode,
            "manifest_artifact": "manifest.json",
            "duration_seconds": None,
            "execution_order": [],
            "comparison": None,
            "iterations": [],
            "signals": [],
            "rejections": [f"{type(exc).__name__}: {exc}"],
            "limitations": [
                "The matrix stopped at the first hard evidence or execution failure.",
                "No performance conclusion can be drawn from this rejected run.",
            ],
        }
        _write_json(output_dir / "summary.json", result)
        return result
