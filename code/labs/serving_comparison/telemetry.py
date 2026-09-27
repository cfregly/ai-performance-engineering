"""Capture and validate live P/D phase and KV transfer telemetry."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    REQUIRED_PD_METRICS,
    TELEMETRY_SCHEMA,
    ArmProfile,
    ConfigError,
)

_PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>[^\s]+)(?:\s+\d+)?$"
)
_PROMETHEUS_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_PROMETHEUS_TYPE = re.compile(r"^# TYPE (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*) (?P<type>[a-z]+)$")

# Streaming requests intentionally allow unbounded reads. Telemetry endpoints do not.
DEFAULT_TELEMETRY_REQUEST_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class TelemetrySnapshot:
    captured_monotonic_s: float
    captured_unix_s: float
    values: dict[str, float]
    raw_by_source: dict[str, str]
    selector_evidence_by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_counters_by_source: dict[str, float] = field(default_factory=dict)
    failure_series_by_source: dict[str, dict[str, float]] = field(default_factory=dict)


def _finite_nonnegative(value: Any, location: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"{location} must be a finite nonnegative number")
    return float(value)


def _dotted_get(document: dict[str, Any], path: str, location: str) -> Any:
    current: Any = document
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ConfigError(f"{location} does not contain {path}")
        current = current[component]
    return current


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    position = 0
    for match in _PROMETHEUS_LABEL.finditer(raw):
        between = raw[position : match.start()].strip()
        if between not in {"", ","}:
            raise ConfigError("Prometheus labels contain unsupported syntax")
        labels[match.group(1)] = bytes(match.group(2), "utf-8").decode("unicode_escape")
        position = match.end()
    if raw[position:].strip() not in {"", ","}:
        raise ConfigError("Prometheus labels contain unsupported syntax")
    return labels


def _prometheus_samples(
    text: str, selected_names: set[str]
) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE.match(line)
        if not match:
            raise ConfigError(f"Prometheus telemetry line {line_number} is unsupported")
        sample_name = match.group("name")
        if sample_name not in selected_names:
            continue
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise ConfigError(f"Prometheus telemetry line {line_number} has a bad value") from exc
        if not math.isfinite(value):
            raise ConfigError(f"Prometheus telemetry line {line_number} is not finite")
        samples.append((sample_name, _parse_labels(match.group("labels")), value))
    return samples


def _prometheus_types(text: str, selected_names: set[str]) -> dict[str, str]:
    types: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("# TYPE "):
            continue
        match = _PROMETHEUS_TYPE.match(line)
        if match is None or match.group("name") not in selected_names:
            continue
        name = match.group("name")
        metric_type = match.group("type")
        previous = types.get(name)
        if previous is not None and previous != metric_type:
            raise ConfigError(f"Prometheus metric {name} declares conflicting types")
        types[name] = metric_type
    return types


def _parse_prometheus_metric_with_evidence(
    text: str, selector: Any, location: str
) -> tuple[float, dict[str, Any]]:
    if isinstance(selector, str):
        names = [selector]
        labels: dict[str, str] = {}
        reduce = "one"
        scale = 1.0
        offset = 0.0
        missing_value: float | None = None
    else:
        selector_object = selector if isinstance(selector, dict) else None
        if selector_object is None:
            raise ConfigError(f"{location} must be a metric name or selector object")
        raw_names = selector_object.get("names", selector_object.get("name"))
        names = raw_names if isinstance(raw_names, list) else [raw_names]
        labels = selector_object.get("labels", {})
        reduce = selector_object.get("reduce", "one")
        scale = selector_object.get("scale", 1.0)
        offset = selector_object.get("offset", 0.0)
        raw_missing_value = selector_object.get("missing_value")
        missing_value = (
            float(raw_missing_value) if type(raw_missing_value) in (int, float) else None
        )
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or not isinstance(labels, dict)
            or reduce not in {"one", "sum"}
            or type(scale) not in (int, float)
            or type(offset) not in (int, float)
            or not math.isfinite(scale)
            or not math.isfinite(offset)
            or (raw_missing_value is not None and bool(labels))
            or (
                raw_missing_value is not None
                and (
                    type(raw_missing_value) not in (int, float)
                    or not math.isfinite(raw_missing_value)
                    or raw_missing_value != 0
                )
            )
        ):
            raise ConfigError(f"{location} has an invalid Prometheus selector")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()
        ):
            raise ConfigError(f"{location}.labels must map strings to strings")
    selected_names = set(names)
    types = _prometheus_types(text, selected_names)
    samples = _prometheus_samples(text, selected_names)
    matches_by_name = {
        name: [
            value
            for sample_name, sample_labels, value in samples
            if sample_name == name
            and all(sample_labels.get(key) == value for key, value in labels.items())
        ]
        for name in names
    }
    missing_names = [name for name, matches in matches_by_name.items() if not matches]
    if missing_names and missing_value is None and len(missing_names) != len(names):
        raise ConfigError(f"{location} is missing required Prometheus samples: {missing_names}")
    if missing_value is not None:
        unsupported = [name for name in missing_names if types.get(name) != "counter"]
        if unsupported:
            raise ConfigError(
                f"{location} can use missing_value only when exact metric families "
                f"declare # TYPE counter: {unsupported}"
            )
    matches = [value for name in names for value in matches_by_name[name]]
    matched_sample_count = len(matches)
    if not matches:
        if missing_value is None:
            raise ConfigError(f"{location} selected no Prometheus samples")
        raw_value = missing_value
    elif reduce == "one" and len(matches) != 1:
        raise ConfigError(f"{location} selected {len(matches)} samples, expected exactly one")
    else:
        raw_value = matches[0] if reduce == "one" else sum(matches)
    value = _finite_nonnegative(raw_value * scale + offset, location)
    return value, {
        "names": names,
        "labels": labels,
        "reduce": reduce,
        "scale": float(scale),
        "offset": float(offset),
        "missing_value": missing_value,
        "missing_names": missing_names,
        "matched_sample_count": matched_sample_count,
        "declared_types": {name: types.get(name) for name in names},
    }


def _parse_prometheus_metric(text: str, selector: Any, location: str) -> float:
    value, _evidence = _parse_prometheus_metric_with_evidence(text, selector, location)
    return value


def _failure_series(text: str, evidence: dict[str, Any], location: str) -> dict[str, float]:
    """Keep each selected counter's name and labels before any reduction."""
    result: dict[str, float] = {}
    for name, labels, value in _prometheus_samples(text, set(evidence["names"])):
        if not all(labels.get(key) == expected for key, expected in evidence["labels"].items()):
            continue
        key = json.dumps([name, sorted(labels.items())], separators=(",", ":"))
        if key in result:
            raise ConfigError(f"{location} contains a duplicate failure counter series: {key}")
        result[key] = _finite_nonnegative(value, f"{location}.{key}")
    return result


def _parse_standard_json(
    text: str, arm: ArmProfile, metrics: dict[str, Any], location: str
) -> dict[str, float]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{location} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{location} must be a JSON object")
    expected_connector = arm.pd_provenance["connector"]["name"] if arm.pd_provenance else None
    expected_header = {
        "schema_version": TELEMETRY_SCHEMA,
        "engine": arm.engine,
        "architecture": "prefill_decode",
        "connector": expected_connector,
    }
    for key, value in expected_header.items():
        if raw.get(key) != value:
            raise ConfigError(f"{location}.{key} does not match P/D provenance")
    values: dict[str, float] = {}
    for semantic, selector in metrics.items():
        if not isinstance(selector, str) or not selector:
            raise ConfigError(f"{location}.metrics.{semantic} must be a dotted JSON path")
        values[semantic] = _finite_nonnegative(
            _dotted_get(raw, selector, location), f"{location}.{selector}"
        )
    return values


async def capture_pd_telemetry(
    client: Any,
    arm: ArmProfile,
    *,
    request_timeout_s: float = DEFAULT_TELEMETRY_REQUEST_TIMEOUT_S,
    raw_sink: Callable[[str, str], None] | None = None,
) -> TelemetrySnapshot:
    """Capture every required counter from the configured live endpoints."""
    if arm.architecture != "prefill_decode":
        raise ConfigError(f"{arm.arm_id} is not a P/D arm")
    if (
        type(request_timeout_s) not in (int, float)
        or not math.isfinite(request_timeout_s)
        or request_timeout_s <= 0
    ):
        raise ConfigError("telemetry request timeout must be a finite positive number")
    values: dict[str, float] = {}
    raw_by_source: dict[str, str] = {}
    selector_evidence_by_source: dict[str, dict[str, Any]] = {}
    failure_counters_by_source: dict[str, float] = {}
    failure_series_by_source: dict[str, dict[str, float]] = {}
    for source in arm.telemetry_sources:
        if source.source_id in raw_by_source:
            raise ConfigError(f"{arm.arm_id} has duplicate telemetry source ids")
        response = await client.get(source.url, timeout=request_timeout_s)
        response.raise_for_status()
        text = response.text
        raw_by_source[source.source_id] = text
        if raw_sink is not None:
            raw_sink(source.source_id, text)
        location = f"{arm.arm_id}.telemetry.{source.source_id}"
        if source.format == "standard_json":
            parsed = _parse_standard_json(text, arm, source.metrics, location)
            selector_evidence_by_source[source.source_id] = {}
            if "kv_transfer_failures_total" in parsed:
                failure_series_by_source[source.source_id] = {
                    source.metrics["kv_transfer_failures_total"]: parsed[
                        "kv_transfer_failures_total"
                    ]
                }
        else:
            parsed = {}
            selector_evidence: dict[str, Any] = {}
            for semantic, selector in source.metrics.items():
                value, evidence = _parse_prometheus_metric_with_evidence(
                    text, selector, f"{location}.{semantic}"
                )
                parsed[semantic] = value
                selector_evidence[semantic] = evidence
                if semantic == "kv_transfer_failures_total":
                    failure_series_by_source[source.source_id] = _failure_series(
                        text, evidence, location
                    )
            selector_evidence_by_source[source.source_id] = selector_evidence
        overlap = (set(values) & set(parsed)) - {"kv_transfer_failures_total"}
        if overlap:
            raise ConfigError(f"{location} duplicated telemetry semantics: {sorted(overlap)}")
        for semantic, value in parsed.items():
            if semantic == "kv_transfer_failures_total":
                failure_counters_by_source[source.source_id] = value
                values[semantic] = values.get(semantic, 0.0) + value
            else:
                values[semantic] = value
    missing = REQUIRED_PD_METRICS - set(values)
    if missing:
        raise ConfigError(f"{arm.arm_id} live telemetry is missing {sorted(missing)}")
    return TelemetrySnapshot(
        captured_monotonic_s=time.monotonic(),
        captured_unix_s=time.time(),
        values=values,
        raw_by_source=raw_by_source,
        selector_evidence_by_source=selector_evidence_by_source,
        failure_counters_by_source=failure_counters_by_source,
        failure_series_by_source=failure_series_by_source,
    )


def compute_pd_delta(before: TelemetrySnapshot, after: TelemetrySnapshot) -> dict[str, float]:
    """Compute a strict interval delta and require observed phase handoff activity."""
    if before.failure_series_by_source.keys() != after.failure_series_by_source.keys():
        raise ConfigError("P/D failure counter series sources changed during the interval")
    for source_id, previous_series in before.failure_series_by_source.items():
        current_series = after.failure_series_by_source[source_id]
        if previous_series.keys() - current_series.keys():
            raise ConfigError(f"P/D failure counter series disappeared at {source_id}")
        for series_id, current in current_series.items():
            if series_id not in previous_series:
                if current > 0:
                    raise ConfigError(
                        f"P/D interval has a new nonzero failure series at {source_id}"
                    )
                continue
            change = current - previous_series[series_id]
            if change < 0:
                raise ConfigError(
                    f"P/D telemetry counter decreased: {source_id} KV failure series {series_id}"
                )
            if change > 0:
                raise ConfigError(
                    f"P/D request interval contains {change:g} failed KV transfers at "
                    f"{source_id}, series {series_id}"
                )
    if before.failure_counters_by_source.keys() != after.failure_counters_by_source.keys():
        raise ConfigError("P/D failure telemetry sources changed during the interval")
    for source_id, previous in before.failure_counters_by_source.items():
        change = after.failure_counters_by_source[source_id] - previous
        if change < 0:
            raise ConfigError(f"P/D telemetry counter decreased: {source_id} KV failures")
        if change > 0:
            raise ConfigError(
                f"P/D request interval contains {change:g} failed KV transfers at {source_id}"
            )
    missing = (REQUIRED_PD_METRICS - set(before.values)) | (REQUIRED_PD_METRICS - set(after.values))
    if missing:
        raise ConfigError(f"P/D telemetry snapshots are missing {sorted(missing)}")
    delta: dict[str, float] = {}
    for metric in sorted(REQUIRED_PD_METRICS):
        change = after.values[metric] - before.values[metric]
        if change < 0:
            raise ConfigError(f"P/D telemetry counter decreased: {metric}")
        delta[metric] = change
    transfer_failures = delta["kv_transfer_failures_total"]
    if transfer_failures > 0:
        raise ConfigError(
            f"P/D request interval contains {transfer_failures:g} failed KV transfers"
        )
    required_activity = REQUIRED_PD_METRICS - {"kv_transfer_failures_total"}
    inactive = sorted(metric for metric in required_activity if delta[metric] <= 0)
    if inactive:
        raise ConfigError(f"P/D request interval lacks required live activity: {inactive}")
    return delta
