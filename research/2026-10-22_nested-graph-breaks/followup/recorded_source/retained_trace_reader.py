#!/usr/bin/env python3
"""Pure-standard-library reader for retained nested-graph Chrome traces.

This corrects a reader bug without changing or recapturing source evidence.
Only exact CPU ``user_annotation`` complete events named
``nested_graphs_compiled_invocation`` are invocation boundaries. CUDA API
events are bound by CPU process/thread plus full interval containment. GPU
kernels are bound first by correlation to an assigned CUDA API event, then by
unambiguous full timestamp containment when no usable correlation exists.

Signed boundary deltas are retained. Invalid geometry is excluded from the
aggregate rather than clamped. No host/GPU causal fraction is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "nested-graph-breaks-retained-trace-reader/v2"
MARKER_NAME = "nested_graphs_compiled_invocation"
DEFAULT_EXPECTED_INVOCATIONS = 5
DEFAULT_MAX_TRACE_BYTES = 512 * 1024 * 1024


class TraceReaderError(ValueError):
    """The retained trace cannot be read under the declared contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _category_tokens(value: Any) -> set[str]:
    return {
        token.strip().lower().replace(" ", "_")
        for token in str(value).split(",")
        if token.strip()
    }


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _interval(event: Mapping[str, Any], event_index: int) -> tuple[float, float]:
    if event.get("ph") != "X":
        raise TraceReaderError(f"event {event_index} is not a complete ph='X' event")
    timestamp = event.get("ts")
    duration = event.get("dur")
    if not _is_number(timestamp) or not _is_number(duration):
        raise TraceReaderError(f"event {event_index} lacks finite ts/dur")
    start = float(timestamp)
    duration_value = float(duration)
    if duration_value < 0.0:
        raise TraceReaderError(f"event {event_index} has negative duration")
    return start, start + duration_value


def _thread_identity(event: Mapping[str, Any]) -> tuple[Any, Any] | None:
    pid = event.get("pid")
    tid = event.get("tid")
    if pid is None or tid is None:
        return None
    if isinstance(pid, (dict, list, set)) or isinstance(tid, (dict, list, set)):
        return None
    return pid, tid


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _extract_correlation(event: Mapping[str, Any]) -> str | None:
    containers: list[Mapping[str, Any]] = [event]
    args = event.get("args")
    if isinstance(args, Mapping):
        containers.append(args)
    for container in containers:
        for key, value in container.items():
            if _normalized_key(key) in {"correlation", "correlationid"}:
                if value is None or isinstance(value, (dict, list, set)):
                    continue
                return str(value)
    return None


def _is_exact_cpu_marker(event: Mapping[str, Any]) -> bool:
    if event.get("ph") != "X" or event.get("name") != MARKER_NAME:
        return False
    tokens = _category_tokens(event.get("cat", ""))
    if "user_annotation" not in tokens:
        return False
    if any("gpu" in token or "cuda" in token for token in tokens):
        return False
    args = event.get("args")
    if isinstance(args, Mapping):
        for key, value in args.items():
            if _normalized_key(key) in {"devicetype", "device"}:
                lowered = str(value).lower()
                if "gpu" in lowered or "cuda" in lowered:
                    return False
    return _thread_identity(event) is not None


def _is_cuda_api(event: Mapping[str, Any]) -> bool:
    if event.get("ph") != "X":
        return False
    tokens = _category_tokens(event.get("cat", ""))
    return bool(tokens & {"cuda_runtime", "cuda_driver"})


def _is_kernel(event: Mapping[str, Any]) -> bool:
    if event.get("ph") != "X":
        return False
    return any("kernel" in token for token in _category_tokens(event.get("cat", "")))


def _is_graph_launch(event: Mapping[str, Any]) -> bool:
    return "cudagraphlaunch" in "".join(
        str(event.get("name", "")).lower().split()
    )


def _event_brief(event: Mapping[str, Any], event_index: int) -> dict[str, Any]:
    return {
        "event_index": event_index,
        "name": str(event.get("name", "")),
        "category": str(event.get("cat", "")),
        "phase": event.get("ph"),
        "pid": event.get("pid"),
        "tid": event.get("tid"),
        "ts": event.get("ts"),
        "dur": event.get("dur"),
        "correlation": _extract_correlation(event),
    }


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _load_trace(path: Path, max_bytes: int) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise TraceReaderError(f"trace must be a regular non-symlink file: {path}")
    stat_result = path.stat()
    if stat_result.st_size <= 0:
        raise TraceReaderError("trace is empty")
    if stat_result.st_size > max_bytes:
        raise TraceReaderError(
            f"trace is {stat_result.st_size} bytes, above limit {max_bytes}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TraceReaderError("trace root must be a JSON object")
    raw_events = payload.get("traceEvents")
    if not isinstance(raw_events, list):
        raise TraceReaderError("trace root lacks a traceEvents list")
    events = [event for event in raw_events if isinstance(event, Mapping)]
    return events, {
        "path": str(path),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
        "raw_event_count": len(raw_events),
        "mapping_event_count": len(events),
    }


def _select_markers(
    events: Sequence[Mapping[str, Any]], expected: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    named_events: list[tuple[int, Mapping[str, Any]]] = []
    cpu_candidates: list[tuple[int, Mapping[str, Any]]] = []
    excluded_named: list[dict[str, Any]] = []
    malformed_cpu: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        if event.get("name") != MARKER_NAME:
            continue
        named_events.append((event_index, event))
        if not _is_exact_cpu_marker(event):
            excluded_named.append(_event_brief(event, event_index))
            continue
        try:
            _interval(event, event_index)
        except TraceReaderError as exc:
            item = _event_brief(event, event_index)
            item["reason"] = str(exc)
            malformed_cpu.append(item)
            continue
        cpu_candidates.append((event_index, event))

    unique: list[dict[str, Any]] = []
    seen: dict[tuple[Any, Any, float, float, str], int] = {}
    duplicates: list[dict[str, Any]] = []
    for event_index, event in cpu_candidates:
        start, end = _interval(event, event_index)
        identity = _thread_identity(event)
        assert identity is not None
        key = (identity[0], identity[1], start, end, MARKER_NAME)
        if key in seen:
            duplicate = _event_brief(event, event_index)
            duplicate["duplicate_of_event_index"] = seen[key]
            duplicates.append(duplicate)
            continue
        seen[key] = event_index
        unique.append(
            {
                "event_index": event_index,
                "event": event,
                "pid": identity[0],
                "tid": identity[1],
                "start_us": start,
                "end_us": end,
                "duration_us": end - start,
            }
        )

    unique.sort(key=lambda item: (item["start_us"], item["end_us"]))
    overlaps: list[dict[str, Any]] = []
    for previous, current in zip(unique, unique[1:]):
        if current["start_us"] < previous["end_us"]:
            overlaps.append(
                {
                    "left_event_index": previous["event_index"],
                    "right_event_index": current["event_index"],
                    "left_end_us": previous["end_us"],
                    "right_start_us": current["start_us"],
                    "overlap_us": previous["end_us"] - current["start_us"],
                }
            )
    thread_ids = sorted(
        {(str(item["pid"]), str(item["tid"])) for item in unique}
    )
    validation = {
        "expected_cpu_invocations": expected,
        "same_name_event_count": len(named_events),
        "selected_unique_cpu_invocations": len(unique),
        "excluded_non_cpu_or_non_complete_same_name_events": excluded_named,
        "malformed_cpu_marker_candidates": malformed_cpu,
        "duplicate_cpu_marker_events": duplicates,
        "overlapping_cpu_intervals": overlaps,
        "cpu_thread_identities": [
            {"pid": pid, "tid": tid} for pid, tid in thread_ids
        ],
        "count_matches": len(unique) == expected,
        "all_unique": not duplicates,
        "nonoverlapping": not overlaps,
        "single_cpu_thread": len(thread_ids) == 1,
        "malformed_count_zero": not malformed_cpu,
    }
    validation["pass"] = all(
        (
            validation["count_matches"],
            validation["all_unique"],
            validation["nonoverlapping"],
            validation["single_cpu_thread"],
            validation["malformed_count_zero"],
        )
    )
    return unique, validation


def _full_containment(start: float, end: float, marker: Mapping[str, Any]) -> bool:
    return start >= marker["start_us"] and end <= marker["end_us"]


def _partial_overlap(start: float, end: float, marker: Mapping[str, Any]) -> bool:
    return start < marker["end_us"] and end > marker["start_us"]


def _boundary_record(
    *,
    event: Mapping[str, Any],
    event_index: int,
    marker: Mapping[str, Any],
    assignment: str,
) -> dict[str, Any]:
    start, end = _interval(event, event_index)
    return {
        "event": _event_brief(event, event_index),
        "invocation_index": marker["invocation_index"],
        "assignment": assignment,
        "start_from_cpu_marker_start_us": start - marker["start_us"],
        "end_from_cpu_marker_end_us": end - marker["end_us"],
        "cpu_tail_after_event_end_us": marker["end_us"] - end,
        "fully_contained": _full_containment(start, end, marker),
        "interpretation": (
            "Signed deltas are retained. A positive end_from_cpu_marker_end_us "
            "means the event ended after the CPU marker. It is a boundary/clock "
            "issue, not negative GPU time."
        ),
    }


def read_events(
    events: Sequence[Mapping[str, Any]], *, expected_invocations: int
) -> dict[str, Any]:
    if expected_invocations <= 0:
        raise TraceReaderError("expected_invocations must be positive")
    markers, marker_validation = _select_markers(events, expected_invocations)
    for invocation_index, marker in enumerate(markers, 1):
        marker["invocation_index"] = invocation_index
        marker["apis"] = []
        marker["kernels"] = []
        marker["boundary_violations"] = []

    api_records: list[tuple[int, Mapping[str, Any], float, float]] = []
    malformed_api: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if not _is_cuda_api(event):
            continue
        try:
            start, end = _interval(event, event_index)
        except TraceReaderError as exc:
            item = _event_brief(event, event_index)
            item["reason"] = str(exc)
            malformed_api.append(item)
            continue
        api_records.append((event_index, event, start, end))

    unassigned_apis: list[dict[str, Any]] = []
    api_boundary_violations: list[dict[str, Any]] = []
    correlation_to_markers: dict[str, set[int]] = {}
    for event_index, event, start, end in api_records:
        identity = _thread_identity(event)
        same_thread = [
            marker
            for marker in markers
            if identity == (marker["pid"], marker["tid"])
        ]
        contained = [
            marker
            for marker in same_thread
            if _full_containment(start, end, marker)
        ]
        if len(contained) == 1:
            marker = contained[0]
            record = {
                "event_index": event_index,
                "event": event,
                "start_us": start,
                "end_us": end,
                "duration_us": end - start,
                "correlation": _extract_correlation(event),
            }
            marker["apis"].append(record)
            correlation = record["correlation"]
            if correlation is not None:
                correlation_to_markers.setdefault(correlation, set()).add(
                    marker["invocation_index"]
                )
            continue
        partial = [
            marker
            for marker in same_thread
            if _partial_overlap(start, end, marker)
        ]
        if partial:
            for marker in partial:
                violation = _boundary_record(
                    event=event,
                    event_index=event_index,
                    marker=marker,
                    assignment="cuda_api_thread_and_partial_interval",
                )
                marker["boundary_violations"].append(violation)
                api_boundary_violations.append(violation)
            continue
        unassigned = _event_brief(event, event_index)
        unassigned["reason"] = (
            "no exact CPU marker on the same pid/tid fully contains this CUDA API span"
        )
        unassigned_apis.append(unassigned)

    kernel_records: list[tuple[int, Mapping[str, Any], float, float]] = []
    malformed_kernels: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if not _is_kernel(event):
            continue
        try:
            start, end = _interval(event, event_index)
        except TraceReaderError as exc:
            item = _event_brief(event, event_index)
            item["reason"] = str(exc)
            malformed_kernels.append(item)
            continue
        kernel_records.append((event_index, event, start, end))

    unassigned_kernels: list[dict[str, Any]] = []
    kernel_boundary_violations: list[dict[str, Any]] = []
    ambiguous_correlations: list[dict[str, Any]] = []
    markers_by_index = {marker["invocation_index"]: marker for marker in markers}
    for event_index, event, start, end in kernel_records:
        correlation = _extract_correlation(event)
        assigned_marker: dict[str, Any] | None = None
        assignment = None
        if correlation is not None and correlation in correlation_to_markers:
            candidate_indices = sorted(correlation_to_markers[correlation])
            if len(candidate_indices) == 1:
                assigned_marker = markers_by_index[candidate_indices[0]]
                assignment = "runtime_correlation"
            else:
                ambiguous = _event_brief(event, event_index)
                ambiguous["candidate_invocation_indices"] = candidate_indices
                ambiguous_correlations.append(ambiguous)
                continue
        if assigned_marker is None:
            contained = [
                marker for marker in markers if _full_containment(start, end, marker)
            ]
            if len(contained) == 1:
                assigned_marker = contained[0]
                assignment = "explicit_timestamp_containment"
            elif len(contained) > 1:
                ambiguous = _event_brief(event, event_index)
                ambiguous["candidate_invocation_indices"] = [
                    marker["invocation_index"] for marker in contained
                ]
                ambiguous["reason"] = "ambiguous timestamp containment"
                ambiguous_correlations.append(ambiguous)
                continue
        if assigned_marker is None:
            partial = [
                marker for marker in markers if _partial_overlap(start, end, marker)
            ]
            if partial:
                for marker in partial:
                    violation = _boundary_record(
                        event=event,
                        event_index=event_index,
                        marker=marker,
                        assignment="kernel_timestamp_partial_interval",
                    )
                    marker["boundary_violations"].append(violation)
                    kernel_boundary_violations.append(violation)
                continue
            unassigned = _event_brief(event, event_index)
            unassigned["reason"] = (
                "no assigned CUDA API correlation and no unique CPU-marker containment"
            )
            unassigned_kernels.append(unassigned)
            continue

        kernel_record = {
            "event_index": event_index,
            "event": event,
            "start_us": start,
            "end_us": end,
            "duration_us": end - start,
            "correlation": correlation,
            "assignment": assignment,
            "fully_contained": _full_containment(start, end, assigned_marker),
        }
        assigned_marker["kernels"].append(kernel_record)
        if not kernel_record["fully_contained"]:
            violation = _boundary_record(
                event=event,
                event_index=event_index,
                marker=assigned_marker,
                assignment=str(assignment),
            )
            assigned_marker["boundary_violations"].append(violation)
            kernel_boundary_violations.append(violation)

    invocation_results: list[dict[str, Any]] = []
    valid_aggregate_rows: list[dict[str, Any]] = []
    for marker in markers:
        api_intervals = [
            (record["start_us"], record["end_us"]) for record in marker["apis"]
        ]
        kernel_intervals = [
            (record["start_us"], record["end_us"])
            for record in marker["kernels"]
        ]
        first_kernel_start = min(
            (interval[0] for interval in kernel_intervals), default=None
        )
        last_kernel_end = max(
            (interval[1] for interval in kernel_intervals), default=None
        )
        signed_tail = (
            None
            if last_kernel_end is None
            else marker["end_us"] - last_kernel_end
        )
        geometry_valid = bool(
            marker_validation["pass"]
            and marker["kernels"]
            and not marker["boundary_violations"]
            and signed_tail is not None
            and signed_tail >= 0.0
        )
        row = {
            "invocation_index": marker["invocation_index"],
            "cpu_marker": {
                "event_index": marker["event_index"],
                "pid": marker["pid"],
                "tid": marker["tid"],
                "start_us": marker["start_us"],
                "end_us": marker["end_us"],
                "duration_us": marker["duration_us"],
            },
            "cuda_api_count": len(marker["apis"]),
            "cuda_api_event_indices": [
                record["event_index"] for record in marker["apis"]
            ],
            "cuda_graph_launch_count": sum(
                1 for record in marker["apis"] if _is_graph_launch(record["event"])
            ),
            "kernel_count": len(marker["kernels"]),
            "kernel_assignments": [
                {
                    "event_index": record["event_index"],
                    "correlation": record["correlation"],
                    "assignment": record["assignment"],
                    "fully_contained": record["fully_contained"],
                    "start_from_cpu_marker_start_us": (
                        record["start_us"] - marker["start_us"]
                    ),
                    "end_from_cpu_marker_end_us": (
                        record["end_us"] - marker["end_us"]
                    ),
                }
                for record in marker["kernels"]
            ],
            "cuda_api_union_us_unclamped": _union_duration(api_intervals),
            "kernel_active_union_us_unclamped": _union_duration(kernel_intervals),
            "first_kernel_start_from_cpu_marker_us": (
                None
                if first_kernel_start is None
                else first_kernel_start - marker["start_us"]
            ),
            "gpu_kernel_span_us_unclamped": (
                None
                if first_kernel_start is None or last_kernel_end is None
                else last_kernel_end - first_kernel_start
            ),
            "signed_cpu_tail_after_last_kernel_us": signed_tail,
            "boundary_violations": marker["boundary_violations"],
            "geometry_valid_for_aggregate": geometry_valid,
            "geometry_interpretation": (
                "A negative signed_cpu_tail_after_last_kernel_us is retained as "
                "a clock/boundary violation. It is not negative GPU time."
            ),
        }
        invocation_results.append(row)
        if geometry_valid:
            valid_aggregate_rows.append(row)

    aggregate = {
        "included_invocation_indices": [
            row["invocation_index"] for row in valid_aggregate_rows
        ],
        "excluded_invocation_indices": [
            row["invocation_index"]
            for row in invocation_results
            if not row["geometry_valid_for_aggregate"]
        ],
        "included_count": len(valid_aggregate_rows),
        "total_cuda_api_count": sum(
            row["cuda_api_count"] for row in valid_aggregate_rows
        ),
        "total_cuda_graph_launch_count": sum(
            row["cuda_graph_launch_count"] for row in valid_aggregate_rows
        ),
        "total_kernel_count": sum(
            row["kernel_count"] for row in valid_aggregate_rows
        ),
        "sum_cpu_marker_duration_us": sum(
            row["cpu_marker"]["duration_us"] for row in valid_aggregate_rows
        ),
        "sum_cuda_api_union_us_unclamped": sum(
            row["cuda_api_union_us_unclamped"] for row in valid_aggregate_rows
        ),
        "sum_kernel_active_union_us_unclamped": sum(
            row["kernel_active_union_us_unclamped"]
            for row in valid_aggregate_rows
        ),
        "interpretation": (
            "Only invocations with valid CPU boundaries, at least one kernel, "
            "and no boundary violation are included. These sums are descriptive "
            "trace geometry and are not a causal fraction."
        ),
    }
    boundary_count = len(api_boundary_violations) + len(kernel_boundary_violations)
    if not marker_validation["pass"]:
        status = "INVALID_CPU_MARKERS"
    elif boundary_count or aggregate["excluded_invocation_indices"]:
        status = "PASS_WITH_BOUNDARY_EXCLUSIONS"
    else:
        status = "PASS"
    return {
        "schema": SCHEMA,
        "status": status,
        "marker_validation": marker_validation,
        "assignment_diagnostics": {
            "malformed_cuda_api_events": malformed_api,
            "unassigned_cuda_api_events": unassigned_apis,
            "cuda_api_boundary_violations": api_boundary_violations,
            "malformed_kernel_events": malformed_kernels,
            "unassigned_kernel_events": unassigned_kernels,
            "ambiguous_kernel_correlations_or_containment": ambiguous_correlations,
            "kernel_boundary_violations": kernel_boundary_violations,
        },
        "invocations": invocation_results,
        "aggregate_valid_geometry_only": aggregate,
        "limits": [
            "This reader does not alter, normalize, or recapture the source trace.",
            "GPU annotation duplicates are not CPU invocation boundaries.",
            "Boundary violations retain signed deltas and are never clamped.",
            "Invalid invocation geometry is excluded from aggregate interpretation.",
            "No host-versus-GPU causal percentage is calculated.",
            "Profile geometry is not a replacement for acceptance timing.",
        ],
    }


def read_trace(
    path: Path, *, expected_invocations: int, max_bytes: int
) -> dict[str, Any]:
    events, source = _load_trace(path, max_bytes)
    result = read_events(events, expected_invocations=expected_invocations)
    result["source_trace"] = source
    return result


def _run_self_tests() -> int:
    start_dir = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        str(start_dir), pattern="test_retained_trace_reader.py"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser(
        "parse", help="Read one retained Chrome trace without modifying it."
    )
    parse_parser.add_argument("--trace", type=Path, required=True)
    parse_parser.add_argument("--out", type=Path, required=True)
    parse_parser.add_argument(
        "--expected-invocations", type=int, default=DEFAULT_EXPECTED_INVOCATIONS
    )
    parse_parser.add_argument(
        "--max-trace-bytes", type=int, default=DEFAULT_MAX_TRACE_BYTES
    )

    subparsers.add_parser(
        "self-test", help="Run only non-performance synthetic fixture tests."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        return _run_self_tests()
    if args.expected_invocations <= 0:
        raise TraceReaderError("--expected-invocations must be positive")
    if args.max_trace_bytes <= 0:
        raise TraceReaderError("--max-trace-bytes must be positive")
    trace_path = args.trace.resolve()
    output_path = args.out.resolve()
    if output_path == trace_path:
        raise TraceReaderError("output path must differ from source trace")
    if output_path.exists():
        raise TraceReaderError(f"output path already exists: {output_path}")
    result = read_trace(
        trace_path,
        expected_invocations=args.expected_invocations,
        max_bytes=args.max_trace_bytes,
    )
    _atomic_json(output_path, result)
    print(output_path)
    return 0 if result["marker_validation"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
