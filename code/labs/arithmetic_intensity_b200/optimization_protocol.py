"""Pure helpers for the bounded B200 kernel optimization comparison."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

SOFTWARE_POWER_CAP_THROTTLE_MASK = 4
REQUIRED_NOMINAL_GAP_TELEMETRY_FIELDS = (
    "gpu_index",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "applications_clock_sm_mhz",
    "applications_clock_memory_mhz",
    "power_draw_w",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "performance_state",
    "throttle_reasons",
)
_NOMINAL_GAP_SUMMARY_FIELDS = (
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "applications_clock_sm_mhz",
    "applications_clock_memory_mhz",
    "power_draw_w",
    "temperature_gpu_c",
    "utilization_gpu_pct",
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile values must be nonempty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) for value in samples):
        raise ValueError("distribution samples must be nonempty and finite")
    return {
        "count": len(samples),
        "min": min(samples),
        "p10": _percentile(samples, 0.10),
        "p50": statistics.median(samples),
        "p90": _percentile(samples, 0.90),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def summarize_nominal_gap_telemetry(
    samples: Sequence[Mapping[str, Any]],
    *,
    window_start_monotonic_s: float,
    window_end_monotonic_s: float,
    minimum_samples: int,
    expected_gpu_index: int,
    requested_interval_s: float,
    maximum_gap_s: float,
) -> dict[str, Any]:
    """Validate NVML query intervals wholly inside one host timing envelope."""
    if (
        not math.isfinite(window_start_monotonic_s)
        or not math.isfinite(window_end_monotonic_s)
        or window_end_monotonic_s <= window_start_monotonic_s
    ):
        raise ValueError("telemetry window must have finite increasing bounds")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int):
        raise ValueError("minimum_samples must be an integer")
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be at least two")
    if isinstance(expected_gpu_index, bool) or not isinstance(expected_gpu_index, int):
        raise ValueError("expected_gpu_index must be an integer")
    if expected_gpu_index < 0:
        raise ValueError("expected_gpu_index must be nonnegative")
    for name, value in (
        ("requested_interval_s", requested_interval_s),
        ("maximum_gap_s", maximum_gap_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_gap_s < requested_interval_s:
        raise ValueError("maximum_gap_s must be at least requested_interval_s")

    in_window = []
    for raw_sample in samples:
        sample = dict(raw_sample)
        query_start = sample.get("query_start_monotonic_s")
        query_end = sample.get("query_end_monotonic_s")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (query_start, query_end)
        ):
            raise RuntimeError("telemetry sample lacks numeric query interval bounds")
        query_start_value = float(query_start)
        query_end_value = float(query_end)
        if (
            not math.isfinite(query_start_value)
            or not math.isfinite(query_end_value)
            or query_end_value < query_start_value
        ):
            raise RuntimeError("telemetry sample query interval must be finite and ordered")
        if (
            window_start_monotonic_s <= query_start_value
            and query_end_value <= window_end_monotonic_s
        ):
            in_window.append(sample)

    if len(in_window) < minimum_samples:
        raise RuntimeError(
            "insufficient in-window telemetry samples: "
            f"required {minimum_samples}, found {len(in_window)}"
        )
    query_midpoints = [
        0.5 * (float(sample["query_start_monotonic_s"]) + float(sample["query_end_monotonic_s"]))
        for sample in in_window
    ]
    if any(
        right <= left for left, right in zip(query_midpoints, query_midpoints[1:], strict=False)
    ):
        raise RuntimeError("in-window telemetry query midpoints must be strictly increasing")

    for sample_index, sample in enumerate(in_window):
        for timestamp_field in (
            "query_start_timestamp_utc",
            "query_end_timestamp_utc",
            "telemetry_timestamp_utc",
        ):
            value = sample.get(timestamp_field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"telemetry sample {sample_index} lacks required field {timestamp_field}"
                )
        for field in REQUIRED_NOMINAL_GAP_TELEMETRY_FIELDS:
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError(
                    f"telemetry sample {sample_index} lacks numeric required field {field}"
                )
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"telemetry sample {sample_index} has non-finite required field {field}"
                )
        if int(float(sample["gpu_index"])) != expected_gpu_index:
            raise RuntimeError(
                f"telemetry sample {sample_index} reports the wrong logical GPU index"
            )
        throttle_value = float(sample["throttle_reasons"])
        if throttle_value < 0 or not throttle_value.is_integer():
            raise RuntimeError(f"telemetry sample {sample_index} has invalid throttle_reasons")
        for field in (
            "graphics_clock_mhz",
            "memory_clock_mhz",
            "applications_clock_sm_mhz",
            "applications_clock_memory_mhz",
            "power_draw_w",
        ):
            if float(sample[field]) <= 0:
                raise RuntimeError(
                    f"telemetry sample {sample_index} has nonpositive required field {field}"
                )
        utilization = float(sample["utilization_gpu_pct"])
        if not 0.0 <= utilization <= 100.0:
            raise RuntimeError(f"telemetry sample {sample_index} has invalid utilization_gpu_pct")

    query_latencies = [
        float(sample["query_end_monotonic_s"]) - float(sample["query_start_monotonic_s"])
        for sample in in_window
    ]
    midpoint_gaps = [
        right - left for left, right in zip(query_midpoints, query_midpoints[1:], strict=False)
    ]
    leading_boundary_gap_s = (
        float(in_window[0]["query_start_monotonic_s"]) - window_start_monotonic_s
    )
    trailing_boundary_gap_s = window_end_monotonic_s - float(in_window[-1]["query_end_monotonic_s"])
    if leading_boundary_gap_s > maximum_gap_s or trailing_boundary_gap_s > maximum_gap_s:
        raise RuntimeError(
            "telemetry cadence exceeded maximum boundary gap: "
            f"allowed {maximum_gap_s:.6f}s, leading {leading_boundary_gap_s:.6f}s, "
            f"trailing {trailing_boundary_gap_s:.6f}s"
        )
    if midpoint_gaps and max(midpoint_gaps) > maximum_gap_s:
        raise RuntimeError(
            "telemetry cadence exceeded maximum gap: "
            f"allowed {maximum_gap_s:.6f}s, observed {max(midpoint_gaps):.6f}s"
        )
    estimated_missed_samples = sum(
        max(0, math.floor(gap / requested_interval_s + 0.5) - 1) for gap in midpoint_gaps
    )

    throttle_masks = [int(float(sample["throttle_reasons"])) for sample in in_window]
    power_cap_count = sum(bool(mask & SOFTWARE_POWER_CAP_THROTTLE_MASK) for mask in throttle_masks)
    metric_summaries = {
        field: _distribution_summary([float(sample[field]) for sample in in_window])
        for field in _NOMINAL_GAP_SUMMARY_FIELDS
    }
    clock_values = [float(sample["graphics_clock_mhz"]) for sample in in_window]
    power_values = [float(sample["power_draw_w"]) for sample in in_window]
    midpoint = len(in_window) // 2
    first_slice = slice(0, midpoint)
    second_slice = slice(midpoint, len(in_window))

    def _relative_change(second: Sequence[float], first: Sequence[float]) -> float:
        first_median = statistics.median(first)
        if first_median == 0:
            raise RuntimeError("telemetry trend denominator must be nonzero")
        return (statistics.median(second) / first_median - 1.0) * 100.0

    clock_half_change_pct = _relative_change(clock_values[second_slice], clock_values[first_slice])
    power_half_change_pct = _relative_change(power_values[second_slice], power_values[first_slice])
    clock_span_pct = (
        (
            float(metric_summaries["graphics_clock_mhz"]["p90"])
            - float(metric_summaries["graphics_clock_mhz"]["p10"])
        )
        / float(metric_summaries["graphics_clock_mhz"]["p50"])
        * 100.0
    )
    power_span_pct = (
        (
            float(metric_summaries["power_draw_w"]["p90"])
            - float(metric_summaries["power_draw_w"]["p10"])
        )
        / float(metric_summaries["power_draw_w"]["p50"])
        * 100.0
    )
    no_trend_threshold_pct = 3.0
    span_threshold_pct = 10.0

    return {
        "window_start_monotonic_s": float(window_start_monotonic_s),
        "window_end_monotonic_s": float(window_end_monotonic_s),
        "window_duration_s": float(window_end_monotonic_s - window_start_monotonic_s),
        "sample_inclusion_rule": (
            "the complete NVML query interval lies inside the host timing envelope"
        ),
        "minimum_required_samples": minimum_samples,
        "sample_count": len(in_window),
        "samples": in_window,
        "cadence": {
            "requested_interval_s": float(requested_interval_s),
            "maximum_allowed_gap_s": float(maximum_gap_s),
            "query_latency_s": _distribution_summary(query_latencies),
            "midpoint_gap_s": (_distribution_summary(midpoint_gaps) if midpoint_gaps else None),
            "boundary_gap_s": {
                "window_start_to_first_query_start": leading_boundary_gap_s,
                "last_query_end_to_window_end": trailing_boundary_gap_s,
                "maximum_observed": max(
                    leading_boundary_gap_s,
                    trailing_boundary_gap_s,
                ),
                "acceptance_gate": True,
                "boundary_definition": "complete query interval bounds",
            },
            "estimated_missed_samples": estimated_missed_samples,
            "estimated_missed_samples_role": (
                "descriptive_nearest_requested_cadence_not_an_acceptance_gate"
            ),
        },
        "metrics": metric_summaries,
        "steady_state_diagnostics": {
            "method": "first-half versus second-half median and p10-to-p90 span",
            "graphics_clock_first_to_second_half_change_pct": clock_half_change_pct,
            "power_first_to_second_half_change_pct": power_half_change_pct,
            "graphics_clock_p10_to_p90_span_pct_of_p50": clock_span_pct,
            "power_p10_to_p90_span_pct_of_p50": power_span_pct,
            "declared_no_trend_threshold_pct": no_trend_threshold_pct,
            "declared_p10_to_p90_span_threshold_pct": span_threshold_pct,
            "passes_declared_steady_state_heuristic": (
                abs(clock_half_change_pct) <= no_trend_threshold_pct
                and abs(power_half_change_pct) <= no_trend_threshold_pct
                and clock_span_pct <= span_threshold_pct
                and power_span_pct <= span_threshold_pct
            ),
            "heuristic_is_a_measurement_acceptance_gate": True,
            "software_power_cap_is_diagnostic_not_a_stability_gate": True,
        },
        "throttle_reasons": {
            "software_power_cap_mask": SOFTWARE_POWER_CAP_THROTTLE_MASK,
            "software_power_cap_active_samples": power_cap_count,
            "software_power_cap_active_fraction": power_cap_count / len(in_window),
            "observed_bitmasks": sorted(set(throttle_masks)),
        },
    }


def observer_overhead_summary(
    sampler_off_median_ms: Sequence[float],
    sampler_on_median_ms: Sequence[float],
    *,
    limit_pct: float,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired sampler overhead without treating it as a speedup result."""
    off = [float(value) for value in sampler_off_median_ms]
    on = [float(value) for value in sampler_on_median_ms]
    if len(off) != len(on):
        raise ValueError("sampler calibration arms must have the same round count")
    if len(off) < 2:
        raise ValueError("sampler calibration requires at least two paired rounds")
    if not math.isfinite(limit_pct) or limit_pct < 0:
        raise ValueError("observer overhead limit must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in [*off, *on]):
        raise ValueError("sampler calibration durations must be finite and positive")
    paired_overhead = [
        (on_value / off_value - 1.0) * 100.0 for off_value, on_value in zip(off, on, strict=True)
    ]
    interval = bootstrap_median_ci(paired_overhead, seed=seed)
    every_observed_pair_within_limit = all(
        abs(overhead_pct) <= limit_pct for overhead_pct in paired_overhead
    )
    return {
        "statistic": "median of paired sampler-on versus sampler-off duration changes",
        "paired_sampler_overhead_pct": paired_overhead,
        "median_sampler_overhead_pct": statistics.median(paired_overhead),
        "bootstrap_95_ci_pct": interval,
        "bootstrap_interval_role": "descriptive_only_not_an_equivalence_proof",
        "observed_run_heuristic_only": True,
        "observer_overhead_limit_pct": float(limit_pct),
        "every_observed_pair_within_limit": every_observed_pair_within_limit,
        "within_observer_overhead_limit": every_observed_pair_within_limit,
        "acceptance_rule": (
            "every observed paired duration change is within the symmetric "
            "observer-overhead limit"
        ),
    }


def integrate_power_samples_over_window(
    samples: Sequence[Mapping[str, Any]],
    *,
    window_start_epoch_s: float,
    window_end_epoch_s: float,
) -> dict[str, Any]:
    """Integrate PowerSampler points over an explicitly bounded host phase."""
    if (
        not math.isfinite(window_start_epoch_s)
        or not math.isfinite(window_end_epoch_s)
        or window_end_epoch_s <= window_start_epoch_s
    ):
        raise ValueError("power integration window must have finite increasing bounds")
    points: list[tuple[float, float]] = []
    for index, raw_sample in enumerate(samples):
        timestamp = raw_sample.get("timestamp_epoch_s")
        total_power = raw_sample.get("total_power_w")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (timestamp, total_power)
        ):
            raise RuntimeError(f"power sample {index} lacks numeric time or power")
        timestamp_value = float(timestamp)
        power_value = float(total_power)
        if not math.isfinite(timestamp_value) or not math.isfinite(power_value) or power_value < 0:
            raise RuntimeError(f"power sample {index} contains an invalid value")
        points.append((timestamp_value, power_value))
    if len(points) < 2:
        raise RuntimeError("power integration requires at least two samples")
    if any(right[0] <= left[0] for left, right in zip(points, points[1:], strict=False)):
        raise RuntimeError("power sample timestamps must be strictly increasing")
    if points[0][0] > window_start_epoch_s or points[-1][0] < window_end_epoch_s:
        raise RuntimeError("power samples do not bracket the requested integration window")

    energy_joules = 0.0
    contributing_segments = 0
    for (left_time, left_power), (right_time, right_power) in zip(points, points[1:], strict=False):
        segment_start = max(left_time, window_start_epoch_s)
        segment_end = min(right_time, window_end_epoch_s)
        if segment_end <= segment_start:
            continue
        duration = right_time - left_time
        start_fraction = (segment_start - left_time) / duration
        end_fraction = (segment_end - left_time) / duration
        start_power = left_power + (right_power - left_power) * start_fraction
        end_power = left_power + (right_power - left_power) * end_fraction
        energy_joules += 0.5 * (start_power + end_power) * (segment_end - segment_start)
        contributing_segments += 1
    if contributing_segments == 0:
        raise RuntimeError("no power-sample segment intersects the integration window")
    phase_duration_s = window_end_epoch_s - window_start_epoch_s
    return {
        "window_start_epoch_s": float(window_start_epoch_s),
        "window_end_epoch_s": float(window_end_epoch_s),
        "phase_duration_s": phase_duration_s,
        "energy_joules": energy_joules,
        "time_weighted_power_w": energy_joules / phase_duration_s,
        "contributing_segments": contributing_segments,
        "method": "linear interpolation and trapezoidal integration over the host phase",
        "limitation": (
            "PowerSampler timestamps precede their NVML reads and provide coarse host-phase "
            "energy rather than exact CUDA-event energy"
        ),
    }


def select_algorithm_index(
    algorithms: Sequence[Mapping[str, Any]],
    expected_signature: str,
) -> int:
    """Resolve one exact cuBLASLt configuration and fail if it is unavailable."""
    if not expected_signature or expected_signature.startswith("UNSET_"):
        raise RuntimeError("The selected cuBLASLt algorithm signature is not frozen")
    matches = [
        int(candidate["heuristic_index"])
        for candidate in algorithms
        if candidate.get("signature") == expected_signature
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one supported cuBLASLt algorithm with signature "
            f"{expected_signature!r}, found {len(matches)}"
        )
    return matches[0]


def alternating_pair_order(window_index: int) -> tuple[str, str]:
    """Return AB/BA order for one independent confirmation window."""
    if window_index < 0:
        raise ValueError("window_index must be nonnegative")
    return (
        ("baseline", "candidate")
        if window_index % 2 == 0
        else (
            "candidate",
            "baseline",
        )
    )


def percent_change(candidate: float, baseline: float) -> float:
    """Return candidate throughput change relative to the equivalent baseline."""
    if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("throughputs must be finite and baseline must be positive")
    return (candidate / baseline - 1.0) * 100.0


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int | str]:
    """Return a deterministic percentile bootstrap interval for a median."""
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) for value in samples):
        raise ValueError("bootstrap samples must be nonempty and finite")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(samples, k=len(samples))) for _ in range(resamples)
    ]
    medians.sort()
    return {
        "method": "deterministic_percentile_bootstrap_of_window_median",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "low": medians[math.floor(0.025 * (resamples - 1))],
        "high": medians[math.ceil(0.975 * (resamples - 1))],
    }


def paired_window_summary(
    baseline_window_medians: Sequence[float],
    candidate_window_medians: Sequence[float],
    *,
    seed: int,
) -> dict[str, Any]:
    """Summarize paired window changes without discarding any window."""
    if len(baseline_window_medians) != len(candidate_window_medians):
        raise ValueError("paired arms must have the same window count")
    if len(baseline_window_medians) < 5:
        raise ValueError("paired confirmation requires at least five windows")
    changes = [
        percent_change(candidate, baseline)
        for baseline, candidate in zip(
            baseline_window_medians,
            candidate_window_medians,
            strict=True,
        )
    ]
    interval = bootstrap_median_ci(changes, seed=seed)
    if float(interval["low"]) > 0.0:
        verdict = "improved"
    elif float(interval["high"]) < 0.0:
        verdict = "regressed"
    else:
        verdict = "no_clear_improvement"
    return {
        "window_candidate_vs_baseline_pct": changes,
        "candidate_vs_baseline_pct": statistics.median(changes),
        "bootstrap_95_ci_pct": interval,
        "verdict": verdict,
    }


def copy_launch_candidates(
    *,
    threads: Sequence[int] = (128, 256, 512),
    blocks_per_sm: Sequence[int] = (4, 8, 16),
) -> list[dict[str, int | str]]:
    """Build the fixed native-copy candidate matrix."""
    return [
        {
            "candidate_id": f"copy_cg_unroll4_t{thread_count}_bsm{block_count}",
            "threads": thread_count,
            "blocks_per_sm": block_count,
        }
        for thread_count in threads
        for block_count in blocks_per_sm
    ]
