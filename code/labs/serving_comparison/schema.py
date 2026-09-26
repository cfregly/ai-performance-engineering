"""Strict input schemas for the serving comparison tool."""

from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROFILE_SCHEMA = "serving-comparison.profile.v1"
TRACE_SCHEMA = "serving-comparison.request-trace.v1"
PD_PROVENANCE_SCHEMA = "serving-comparison.pd-provenance.v1"
TELEMETRY_SCHEMA = "serving-comparison.telemetry.v1"
RESULT_SCHEMA = "serving-comparison.result.v1"

ENGINES = {"vllm", "sglang"}
ARCHITECTURES = {"monolithic", "prefill_decode"}
MODES = {"engine", "protocol_fixture"}
REQUIRED_PD_METRICS = {
    "kv_transfer_requests_total",
    "kv_transfer_failures_total",
    "kv_transfer_time_seconds_total",
    "prefill_requests_total",
    "decode_requests_total",
}
OPTIONAL_PD_DIAGNOSTICS = {
    "kv_transfer_bytes_total",
    "queue_age_seconds",
    "prefill_pool_idle_fraction",
    "decode_pool_idle_fraction",
}


class ConfigError(ValueError):
    """An input cannot support the requested comparison."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{location} must be a list")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a nonempty string")
    return value.strip()


def _number(value: Any, location: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{location} must be a finite number")
    if positive and value <= 0:
        raise ConfigError(f"{location} must be positive")
    return float(value)


def _integer(value: Any, location: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ConfigError(f"{location} must be an integer")
    if positive and value <= 0:
        raise ConfigError(f"{location} must be positive")
    return value


def _read_json(path: Path, location: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"{location} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{location} is not valid JSON: {path}:{exc.lineno}: {exc.msg}") from exc
    return _object(raw, location)


def _validate_url(url: str, location: str, mode: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{location} must be an http or https URL")
    host = parsed.hostname.lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    is_loopback = host in {"localhost", "localhost.localdomain"} or bool(
        address and address.is_loopback
    )
    if mode == "protocol_fixture" and not is_loopback:
        raise ConfigError(f"{location} must be loopback in protocol_fixture mode")
    private_name = "." not in host or host.endswith(
        (".internal", ".local", ".svc", ".cluster.local")
    )
    private_address = bool(
        address and (address.is_private or address.is_loopback or address.is_link_local)
    )
    if mode == "engine" and not (private_name or private_address):
        raise ConfigError(f"{location} must use a private serving endpoint")
    return url


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    deadline_ms: float
    cancel_after_ms: float | None
    expected_status: str
    expected_output_token_ids: tuple[int, ...] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TelemetrySource:
    source_id: str
    format: str
    url: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ArmProfile:
    arm_id: str
    engine: str
    architecture: str
    endpoint: str
    api_key_env: str | None
    runtime_version: str
    runtime_build_id: str
    gpu_ids: tuple[str, ...]
    identity_probes: tuple[dict[str, Any], ...]
    lifecycle: dict[str, Any]
    telemetry_sources: tuple[TelemetrySource, ...]
    pd_provenance_path: Path | None
    pd_provenance: dict[str, Any] | None


@dataclass(frozen=True)
class ComparisonProfile:
    source_path: Path
    mode: str
    workload_id: str
    model: str
    tokenizer: str
    precision: str
    seed: int
    gpu_budget: int
    gpu_ids: tuple[str, ...]
    clocks_locked: bool
    application_clocks_mhz: dict[str, int]
    admission_max_concurrency: int
    correctness_policy: str
    max_token_mismatches: int
    arms: tuple[ArmProfile, ...]


def _validate_pd_provenance(
    raw: dict[str, Any],
    *,
    arm: dict[str, Any],
    workload: dict[str, Any],
    fleet_gpu_ids: set[str],
    mode: str,
    location: str,
) -> None:
    if raw.get("schema_version") != PD_PROVENANCE_SCHEMA:
        raise ConfigError(f"{location}.schema_version must be {PD_PROVENANCE_SCHEMA}")
    engine = arm["engine"]
    if raw.get("engine") != engine or raw.get("architecture") != "prefill_decode":
        raise ConfigError(
            f"{location} must identify the arm's engine and prefill_decode architecture"
        )
    if raw.get("request_path") != "kv_handoff":
        raise ConfigError(
            f"{location}.request_path must be kv_handoff. Whole-request routing is not P/D"
        )

    connector = _object(raw.get("connector"), f"{location}.connector")
    connector_name = _text(connector.get("name"), f"{location}.connector.name")
    backend = _text(connector.get("backend"), f"{location}.connector.backend").lower()
    supported: dict[str, dict[str, set[str]]] = {
        "vllm": {
            "NixlConnector": {"nixl"},
            "MooncakeConnector": {"mooncake"},
            "LMCacheConnectorV1": {"nixl", "mooncake"},
        },
        "sglang": {"NIXL": {"nixl"}, "Mooncake": {"mooncake"}},
    }
    if connector_name not in supported[engine] or backend not in supported[engine][connector_name]:
        raise ConfigError(f"{location}.connector does not name a supported {engine} P/D connector")

    proxy = _object(raw.get("proxy"), f"{location}.proxy")
    implementation = _text(proxy.get("implementation"), f"{location}.proxy.implementation")
    allowed_proxy = {
        "vllm": {
            "aisp_vllm_pd_proxy",
            "vllm_disagg_proxy",
            "vllm_disagg_proxy_multiturn",
        },
        "sglang": {"sglang_model_gateway", "sglang_mini_lb"},
    }
    if implementation not in allowed_proxy[engine]:
        raise ConfigError(f"{location}.proxy.implementation is not supported for {engine}")

    runtime = _object(raw.get("runtime"), f"{location}.runtime")
    if runtime.get("version") != arm["runtime"]["version"]:
        raise ConfigError(f"{location}.runtime.version does not match the arm")
    if runtime.get("build_id") != arm["runtime"]["build_id"]:
        raise ConfigError(f"{location}.runtime.build_id does not match the arm")
    recorded_workload = _object(raw.get("workload"), f"{location}.workload")
    for key in ("model", "tokenizer", "precision"):
        if recorded_workload.get(key) != workload[key]:
            raise ConfigError(f"{location}.workload.{key} does not match the profile")

    pools = _object(raw.get("gpu_pools"), f"{location}.gpu_pools")
    prefill = {
        _text(item, f"{location}.gpu_pools.prefill")
        for item in _list(pools.get("prefill"), f"{location}.gpu_pools.prefill")
    }
    decode = {
        _text(item, f"{location}.gpu_pools.decode")
        for item in _list(pools.get("decode"), f"{location}.gpu_pools.decode")
    }
    if not prefill or not decode or prefill & decode:
        raise ConfigError(
            f"{location}.gpu_pools must contain nonempty, disjoint prefill and decode pools"
        )
    if prefill | decode != fleet_gpu_ids:
        raise ConfigError(f"{location}.gpu_pools must cover the fixed GPU fleet exactly")

    endpoints = _object(raw.get("endpoints"), f"{location}.endpoints")
    for key in ("prefill", "decode", "router"):
        _validate_url(
            _text(endpoints.get(key), f"{location}.endpoints.{key}"),
            f"{location}.endpoints.{key}",
            mode,
        )
    digest = _text(raw.get("manifest_digest"), f"{location}.manifest_digest")
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ConfigError(f"{location}.manifest_digest must be sha256 followed by 64 hex digits")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ConfigError(f"{location}.manifest_digest must contain lowercase hexadecimal") from exc
    if digest.lower() != digest:
        raise ConfigError(f"{location}.manifest_digest must contain lowercase hexadecimal")


def load_profile(path: Path) -> ComparisonProfile:
    """Load and validate one exact vLLM and SGLang architecture matrix."""
    path = path.resolve()
    raw = _read_json(path, "profile")
    if raw.get("schema_version") != PROFILE_SCHEMA:
        raise ConfigError(f"profile.schema_version must be {PROFILE_SCHEMA}")
    mode = _text(raw.get("mode"), "profile.mode")
    if mode not in MODES:
        raise ConfigError(f"profile.mode must be one of {sorted(MODES)}")
    workload = _object(raw.get("workload"), "profile.workload")
    workload_id = _text(workload.get("workload_id"), "profile.workload.workload_id")
    model = _text(workload.get("model"), "profile.workload.model")
    tokenizer = _text(workload.get("tokenizer"), "profile.workload.tokenizer")
    precision = _text(workload.get("precision"), "profile.workload.precision")
    seed = _integer(workload.get("seed"), "profile.workload.seed")
    admission_max_concurrency = _integer(
        workload.get("admission_max_concurrency"),
        "profile.workload.admission_max_concurrency",
        positive=True,
    )
    correctness = _object(raw.get("correctness"), "profile.correctness")
    correctness_policy = _text(correctness.get("policy"), "profile.correctness.policy")
    if correctness_policy not in {"exact_token_ids", "per_request_golden"}:
        raise ConfigError(
            "profile.correctness.policy must be exact_token_ids or per_request_golden"
        )
    max_token_mismatches = _integer(
        correctness.get("max_token_mismatches", 0),
        "profile.correctness.max_token_mismatches",
    )
    if max_token_mismatches < 0:
        raise ConfigError("profile.correctness.max_token_mismatches must be nonnegative")

    hardware = _object(raw.get("hardware"), "profile.hardware")
    gpu_budget = _integer(hardware.get("gpu_budget"), "profile.hardware.gpu_budget", positive=True)
    gpu_items = _list(hardware.get("gpus"), "profile.hardware.gpus")
    gpu_ids: list[str] = []
    application_clocks: dict[str, int] = {}
    for index, item in enumerate(gpu_items):
        gpu = _object(item, f"profile.hardware.gpus[{index}]")
        gpu_id = _text(gpu.get("id"), f"profile.hardware.gpus[{index}].id")
        if gpu_id in gpu_ids:
            raise ConfigError("profile.hardware.gpus contains duplicate ids")
        gpu_ids.append(gpu_id)
        if "application_clock_mhz" in gpu:
            application_clocks[gpu_id] = _integer(
                gpu["application_clock_mhz"],
                f"profile.hardware.gpus[{index}].application_clock_mhz",
                positive=True,
            )
    if len(gpu_ids) != gpu_budget:
        raise ConfigError("profile.hardware.gpu_budget must equal the number of GPU ids")
    clocks_locked = hardware.get("clocks_locked")
    if type(clocks_locked) is not bool:
        raise ConfigError("profile.hardware.clocks_locked must be a boolean")
    if mode == "engine" and (not clocks_locked or set(application_clocks) != set(gpu_ids)):
        raise ConfigError("engine mode requires locked application clocks for every GPU")

    arms_raw = _list(raw.get("arms"), "profile.arms")
    expected_matrix = {
        (engine, architecture) for engine in ENGINES for architecture in ARCHITECTURES
    }
    actual_matrix: set[tuple[str, str]] = set()
    arms: list[ArmProfile] = []
    arm_ids: set[str] = set()
    for index, item in enumerate(arms_raw):
        location = f"profile.arms[{index}]"
        arm = _object(item, location)
        arm_id = _text(arm.get("arm_id"), f"{location}.arm_id")
        engine = _text(arm.get("engine"), f"{location}.engine")
        architecture = _text(arm.get("architecture"), f"{location}.architecture")
        if engine not in ENGINES or architecture not in ARCHITECTURES:
            raise ConfigError(f"{location} has an unsupported engine or architecture")
        if arm_id in arm_ids or (engine, architecture) in actual_matrix:
            raise ConfigError("profile.arms contains a duplicate id or matrix cell")
        arm_ids.add(arm_id)
        actual_matrix.add((engine, architecture))
        endpoint = _validate_url(
            _text(arm.get("endpoint"), f"{location}.endpoint"), f"{location}.endpoint", mode
        )
        runtime = _object(arm.get("runtime"), f"{location}.runtime")
        runtime_version = _text(runtime.get("version"), f"{location}.runtime.version")
        runtime_build_id = _text(runtime.get("build_id"), f"{location}.runtime.build_id")
        arm_gpu_ids = tuple(
            _text(item, f"{location}.gpu_ids")
            for item in _list(arm.get("gpu_ids"), f"{location}.gpu_ids")
        )
        if set(arm_gpu_ids) != set(gpu_ids) or len(arm_gpu_ids) != len(gpu_ids):
            raise ConfigError(f"{location}.gpu_ids must equal the fixed GPU fleet")
        identity = _object(arm.get("identity"), f"{location}.identity")
        probes_raw = _list(identity.get("probes"), f"{location}.identity.probes")
        if not probes_raw:
            raise ConfigError(f"{location}.identity.probes must not be empty")
        identity_probes: list[dict[str, Any]] = []
        proven: set[str] = set()
        for probe_index, probe_raw in enumerate(probes_raw):
            probe_location = f"{location}.identity.probes[{probe_index}]"
            probe = _object(probe_raw, probe_location)
            assertions = _object(probe.get("assertions"), f"{probe_location}.assertions")
            if not assertions:
                raise ConfigError(f"{probe_location}.assertions must not be empty")
            proves = _list(probe.get("proves"), f"{probe_location}.proves")
            if any(item not in {"model", "runtime_version"} for item in proves):
                raise ConfigError(f"{probe_location}.proves has an unsupported identity claim")
            claim_values = {"model": model, "runtime_version": runtime_version}
            for claim in proves:
                if claim_values[claim] not in assertions.values():
                    raise ConfigError(
                        f"{probe_location}.proves.{claim} is not bound to its pinned assertion value"
                    )
            proven.update(proves)
            identity_probes.append(
                {
                    "url": _validate_url(
                        _text(probe.get("url"), f"{probe_location}.url"),
                        f"{probe_location}.url",
                        mode,
                    ),
                    "assertions": assertions,
                    "proves": proves,
                }
            )
        if proven != {"model", "runtime_version"}:
            raise ConfigError(
                f"{location}.identity.probes must prove the live model and runtime_version"
            )

        lifecycle = _object(arm.get("lifecycle"), f"{location}.lifecycle")
        if lifecycle.get("mode") != "local_process":
            raise ConfigError(f"{location}.lifecycle.mode must be local_process")
        command = _list(lifecycle.get("start_command"), f"{location}.lifecycle.start_command")
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ConfigError(f"{location}.lifecycle.start_command must be a nonempty argv list")
        environment = _object(lifecycle.get("environment", {}), f"{location}.lifecycle.environment")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ConfigError(f"{location}.lifecycle.environment must map strings to strings")
        passthrough = _list(
            lifecycle.get("env_passthrough", []), f"{location}.lifecycle.env_passthrough"
        )
        if any(not isinstance(name, str) or not name for name in passthrough):
            raise ConfigError(f"{location}.lifecycle.env_passthrough must contain variable names")
        if mode == "engine" and environment.get("CUDA_VISIBLE_DEVICES") != ",".join(gpu_ids):
            raise ConfigError(
                f"{location}.lifecycle.environment.CUDA_VISIBLE_DEVICES must list the fixed GPU fleet"
            )
        working_directory = lifecycle.get("working_directory", ".")
        if not isinstance(working_directory, str) or not working_directory:
            raise ConfigError(f"{location}.lifecycle.working_directory must be a path string")
        normalized_lifecycle: dict[str, Any] = {
            "mode": "local_process",
            "start_command": command,
            "environment": environment,
            "env_passthrough": passthrough,
            "working_directory": str((path.parent / working_directory).resolve()),
            "ready_url": _validate_url(
                _text(lifecycle.get("ready_url"), f"{location}.lifecycle.ready_url"),
                f"{location}.lifecycle.ready_url",
                mode,
            ),
            "gpu_evidence": "nvidia_smi" if mode == "engine" else "process_only",
        }
        normalized_lifecycle["timeout_s"] = _number(
            lifecycle.get("timeout_s", 120),
            f"{location}.lifecycle.timeout_s",
            positive=True,
        )
        normalized_lifecycle["shutdown_timeout_s"] = _number(
            lifecycle.get("shutdown_timeout_s", 30),
            f"{location}.lifecycle.shutdown_timeout_s",
            positive=True,
        )

        sources: list[TelemetrySource] = []
        covered_metrics: set[str] = set()
        for source_index, source_item in enumerate(
            _list(arm.get("telemetry_sources", []), f"{location}.telemetry_sources")
        ):
            source_location = f"{location}.telemetry_sources[{source_index}]"
            source = _object(source_item, source_location)
            source_format = _text(source.get("format"), f"{source_location}.format")
            if source_format not in {"prometheus", "standard_json"}:
                raise ConfigError(f"{source_location}.format is unsupported")
            metrics = _object(source.get("metrics"), f"{source_location}.metrics")
            unknown = set(metrics) - (REQUIRED_PD_METRICS | OPTIONAL_PD_DIAGNOSTICS)
            duplicate = set(metrics) & covered_metrics
            if unknown or duplicate:
                raise ConfigError(f"{source_location}.metrics has unknown or duplicate semantics")
            covered_metrics.update(metrics)
            sources.append(
                TelemetrySource(
                    source_id=_text(source.get("source_id"), f"{source_location}.source_id"),
                    format=source_format,
                    url=_validate_url(
                        _text(source.get("url"), f"{source_location}.url"),
                        f"{source_location}.url",
                        mode,
                    ),
                    metrics=metrics,
                )
            )

        provenance_path: Path | None = None
        provenance: dict[str, Any] | None = None
        if architecture == "prefill_decode":
            missing = sorted(REQUIRED_PD_METRICS - covered_metrics)
            if missing:
                raise ConfigError(f"{location} lacks required live P/D metrics: {missing}")
            provenance_ref = _text(arm.get("pd_provenance"), f"{location}.pd_provenance")
            provenance_path = (path.parent / provenance_ref).resolve()
            provenance = _read_json(provenance_path, f"{location}.pd_provenance")
            _validate_pd_provenance(
                provenance,
                arm=arm,
                workload=workload,
                fleet_gpu_ids=set(gpu_ids),
                mode=mode,
                location=f"{location}.pd_provenance",
            )
            diagnostic_support = _object(
                arm.get("diagnostic_support"), f"{location}.diagnostic_support"
            )
            if set(diagnostic_support) != OPTIONAL_PD_DIAGNOSTICS:
                raise ConfigError(
                    f"{location}.diagnostic_support must label every optional P/D diagnostic"
                )
            for diagnostic, declaration_raw in diagnostic_support.items():
                declaration = _object(
                    declaration_raw, f"{location}.diagnostic_support.{diagnostic}"
                )
                status = declaration.get("status")
                if status == "measured":
                    if diagnostic not in covered_metrics:
                        raise ConfigError(
                            f"{location}.{diagnostic} is measured but has no telemetry selector"
                        )
                elif status == "unsupported":
                    _text(
                        declaration.get("reason"),
                        f"{location}.diagnostic_support.{diagnostic}.reason",
                    )
                    if diagnostic in covered_metrics:
                        raise ConfigError(
                            f"{location}.{diagnostic} is unsupported but has a telemetry selector"
                        )
                else:
                    raise ConfigError(
                        f"{location}.diagnostic_support.{diagnostic}.status must be measured or unsupported"
                    )
            provenance["diagnostic_support"] = diagnostic_support
        elif sources or arm.get("pd_provenance") is not None:
            raise ConfigError(f"{location} is monolithic and cannot declare P/D evidence")

        api_key_env = arm.get("api_key_env")
        if api_key_env is not None:
            api_key_env = _text(api_key_env, f"{location}.api_key_env")
        arms.append(
            ArmProfile(
                arm_id=arm_id,
                engine=engine,
                architecture=architecture,
                endpoint=endpoint,
                api_key_env=api_key_env,
                runtime_version=runtime_version,
                runtime_build_id=runtime_build_id,
                gpu_ids=arm_gpu_ids,
                identity_probes=tuple(identity_probes),
                lifecycle=normalized_lifecycle,
                telemetry_sources=tuple(sources),
                pd_provenance_path=provenance_path,
                pd_provenance=provenance,
            )
        )

    if actual_matrix != expected_matrix or len(arms) != 4:
        raise ConfigError(
            "profile.arms must contain exactly the vLLM and SGLang mono and P/D matrix"
        )
    for engine in ENGINES:
        engine_arms = [arm for arm in arms if arm.engine == engine]
        versions = {(arm.runtime_version, arm.runtime_build_id) for arm in engine_arms}
        if len(versions) != 1:
            raise ConfigError(f"{engine} monolithic and P/D arms must use the same runtime build")
    return ComparisonProfile(
        source_path=path,
        mode=mode,
        workload_id=workload_id,
        model=model,
        tokenizer=tokenizer,
        precision=precision,
        seed=seed,
        gpu_budget=gpu_budget,
        gpu_ids=tuple(gpu_ids),
        clocks_locked=clocks_locked,
        application_clocks_mhz=application_clocks,
        admission_max_concurrency=admission_max_concurrency,
        correctness_policy=correctness_policy,
        max_token_mismatches=max_token_mismatches,
        arms=tuple(arms),
    )


def load_trace(path: Path) -> tuple[RequestSpec, ...]:
    """Load a deterministic token-id request trace from JSONL."""
    requests: list[RequestSpec] = []
    seen: set[str] = set()
    last_arrival = -1.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            raw = _object(json.loads(line), location)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{location}: invalid JSON: {exc.msg}") from exc
        if raw.get("schema_version") != TRACE_SCHEMA:
            raise ConfigError(f"{location}.schema_version must be {TRACE_SCHEMA}")
        request_id = _text(raw.get("request_id"), f"{location}.request_id")
        if request_id in seen:
            raise ConfigError(f"{location}.request_id is duplicated")
        seen.add(request_id)
        arrival_ms = _number(raw.get("arrival_ms"), f"{location}.arrival_ms")
        if arrival_ms < 0 or arrival_ms < last_arrival:
            raise ConfigError(f"{location}.arrival_ms must be nonnegative and ordered")
        last_arrival = arrival_ms
        prompt = _list(raw.get("prompt_token_ids"), f"{location}.prompt_token_ids")
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise ConfigError(f"{location}.prompt_token_ids must be nonempty nonnegative integers")
        max_tokens = _integer(raw.get("max_tokens"), f"{location}.max_tokens", positive=True)
        deadline_ms = _number(raw.get("deadline_ms"), f"{location}.deadline_ms", positive=True)
        cancel_after = raw.get("cancel_after_ms")
        if cancel_after is not None:
            cancel_after = _number(cancel_after, f"{location}.cancel_after_ms", positive=True)
            if cancel_after >= deadline_ms:
                raise ConfigError(f"{location}.cancel_after_ms must be less than deadline_ms")
        expected_status = raw.get("expected_status", "completed")
        if expected_status not in {"completed", "failed", "cancelled"}:
            raise ConfigError(f"{location}.expected_status is invalid")
        if expected_status == "cancelled" and cancel_after is None:
            raise ConfigError(f"{location} expects cancellation but has no cancel_after_ms")
        expected_tokens_raw = raw.get("expected_output_token_ids")
        expected_tokens: tuple[int, ...] | None = None
        if expected_tokens_raw is not None:
            expected_list = _list(expected_tokens_raw, f"{location}.expected_output_token_ids")
            if any(type(token) is not int or token < 0 for token in expected_list):
                raise ConfigError(
                    f"{location}.expected_output_token_ids must be nonnegative integers"
                )
            expected_tokens = tuple(expected_list)
        metadata = _object(raw.get("metadata", {}), f"{location}.metadata")
        requests.append(
            RequestSpec(
                request_id=request_id,
                arrival_ms=arrival_ms,
                prompt_token_ids=tuple(prompt),
                max_tokens=max_tokens,
                deadline_ms=deadline_ms,
                cancel_after_ms=cancel_after,
                expected_status=expected_status,
                expected_output_token_ids=expected_tokens,
                metadata=metadata,
            )
        )
    if not requests:
        raise ConfigError("request trace is empty")
    return tuple(requests)
