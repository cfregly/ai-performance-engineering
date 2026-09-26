"""Capture and validate live P/D phase and KV transfer telemetry."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TelemetrySnapshot:
    captured_monotonic_s: float
    captured_unix_s: float
    values: dict[str, float]
    raw_by_source: dict[str, str]


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


def _prometheus_samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE.match(line)
        if not match:
            raise ConfigError(f"Prometheus telemetry line {line_number} is unsupported")
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise ConfigError(f"Prometheus telemetry line {line_number} has a bad value") from exc
        if not math.isfinite(value):
            raise ConfigError(f"Prometheus telemetry line {line_number} is not finite")
        samples.append((match.group("name"), _parse_labels(match.group("labels")), value))
    return samples


def _parse_prometheus_metric(text: str, selector: Any, location: str) -> float:
    if isinstance(selector, str):
        names = [selector]
        labels: dict[str, str] = {}
        reduce = "one"
        scale = 1.0
        offset = 0.0
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
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or not isinstance(labels, dict)
            or reduce not in {"one", "sum"}
            or type(scale) not in (int, float)
            or type(offset) not in (int, float)
            or not math.isfinite(scale)
            or not math.isfinite(offset)
        ):
            raise ConfigError(f"{location} has an invalid Prometheus selector")
        if any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()
        ):
            raise ConfigError(f"{location}.labels must map strings to strings")
    matches = [
        value
        for sample_name, sample_labels, value in _prometheus_samples(text)
        if sample_name in names
        and all(sample_labels.get(key) == value for key, value in labels.items())
    ]
    if not matches:
        raise ConfigError(f"{location} selected no Prometheus samples")
    if reduce == "one" and len(matches) != 1:
        raise ConfigError(f"{location} selected {len(matches)} samples, expected exactly one")
    raw_value = matches[0] if reduce == "one" else sum(matches)
    return _finite_nonnegative(raw_value * scale + offset, location)


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


async def capture_pd_telemetry(client: Any, arm: ArmProfile) -> TelemetrySnapshot:
    """Capture every required counter from the configured live endpoints."""
    if arm.architecture != "prefill_decode":
        raise ConfigError(f"{arm.arm_id} is not a P/D arm")
    values: dict[str, float] = {}
    raw_by_source: dict[str, str] = {}
    for source in arm.telemetry_sources:
        response = await client.get(source.url)
        response.raise_for_status()
        text = response.text
        raw_by_source[source.source_id] = text
        location = f"{arm.arm_id}.telemetry.{source.source_id}"
        if source.format == "standard_json":
            parsed = _parse_standard_json(text, arm, source.metrics, location)
        else:
            parsed = {
                semantic: _parse_prometheus_metric(text, selector, f"{location}.{semantic}")
                for semantic, selector in source.metrics.items()
            }
        overlap = set(values) & set(parsed)
        if overlap:
            raise ConfigError(f"{location} duplicated telemetry semantics: {sorted(overlap)}")
        values.update(parsed)
    missing = REQUIRED_PD_METRICS - set(values)
    if missing:
        raise ConfigError(f"{arm.arm_id} live telemetry is missing {sorted(missing)}")
    return TelemetrySnapshot(
        captured_monotonic_s=time.monotonic(),
        captured_unix_s=time.time(),
        values=values,
        raw_by_source=raw_by_source,
    )


def compute_pd_delta(before: TelemetrySnapshot, after: TelemetrySnapshot) -> dict[str, float]:
    """Compute a strict interval delta and require observed phase handoff activity."""
    missing = (REQUIRED_PD_METRICS - set(before.values)) | (REQUIRED_PD_METRICS - set(after.values))
    if missing:
        raise ConfigError(f"P/D telemetry snapshots are missing {sorted(missing)}")
    delta: dict[str, float] = {}
    for metric in sorted(REQUIRED_PD_METRICS):
        change = after.values[metric] - before.values[metric]
        if change < 0:
            raise ConfigError(f"P/D telemetry counter decreased: {metric}")
        delta[metric] = change
    required_activity = REQUIRED_PD_METRICS - {"kv_transfer_failures_total"}
    inactive = sorted(metric for metric in required_activity if delta[metric] <= 0)
    if inactive:
        raise ConfigError(f"P/D request interval lacks required live activity: {inactive}")
    return delta
