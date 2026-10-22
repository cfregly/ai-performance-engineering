#!/usr/bin/env python3
"""Fail-closed untimed CUDA profile capture for the nested-graphs benchmark.

The measured benchmark warms the compiled callable under ``torch.no_grad()``.
TorchDynamo guards on grad mode, so profiling must use the same context or the
profiled call can compile and capture a different variant inside Kineto.

CUDA event timings in this module only show that the selected stream advanced
between two events.  They never substitute for a named CUDA Graph launch in
the exported trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional


DEFAULT_INVOCATIONS = 5
DEFAULT_RECORD_LABEL = "nested_graphs_compiled_invocation"


def _normalized_category_tokens(category: Any) -> set[str]:
    return {
        token.strip().lower().replace(" ", "_")
        for token in str(category).split(",")
        if token.strip()
    }


def _is_cuda_api_event(event: Mapping[str, Any]) -> bool:
    """Accept only CUDA runtime/driver callbacks, never CPU annotations."""

    categories = _normalized_category_tokens(event.get("cat", ""))
    return bool(categories & {"cuda_runtime", "cuda_driver"})


def _read_trace_events(trace_path: Path) -> List[Mapping[str, Any]]:
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    if not isinstance(trace, dict):
        raise ValueError("Chrome trace root must be a JSON object")
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents list")
    return [event for event in events if isinstance(event, Mapping)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_counter_snapshot(snapshot: Mapping[str, int]) -> Dict[str, int]:
    if not isinstance(snapshot, Mapping):
        raise TypeError("compiler counter snapshot must be a mapping")
    validated: Dict[str, int] = {}
    for key, value in snapshot.items():
        if not isinstance(key, str):
            raise TypeError("compiler counter snapshot keys must be strings")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"compiler counter snapshot value for {key!r} must be an integer"
            )
        validated[key] = value
    return {key: validated[key] for key in sorted(validated)}


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> Dict[str, int]:
    keys = set(before) | set(after)
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
        if int(after.get(key, 0)) != int(before.get(key, 0))
    }


def _update_invocation_evidence(
    result: Dict[str, Any],
    *,
    requested: int,
    completed: int,
    cpu_call_durations_ns: List[int],
    cuda_event_elapsed_ms: List[float],
    cuda_synchronize_completed: bool,
) -> bool:
    cpu_evidence_complete = bool(
        len(cpu_call_durations_ns) == requested
        and all(isinstance(value, int) and value > 0 for value in cpu_call_durations_ns)
    )
    cuda_event_evidence_complete = bool(
        len(cuda_event_elapsed_ms) == requested
        and all(
            math.isfinite(value) and value > 0.0
            for value in cuda_event_elapsed_ms
        )
    )
    complete = bool(
        completed == requested
        and cuda_synchronize_completed
        and cpu_evidence_complete
        and cuda_event_evidence_complete
    )
    result.update(
        {
            "invocation_count_requested": requested,
            "invocation_count_completed": completed,
            "cpu_call_durations_ns": list(cpu_call_durations_ns),
            "cpu_call_evidence_complete": cpu_evidence_complete,
            "cuda_event_elapsed_ms": list(cuda_event_elapsed_ms),
            "cuda_event_evidence_complete": cuda_event_evidence_complete,
            "cuda_synchronize_completed": cuda_synchronize_completed,
            "invocation_evidence_complete": complete,
        }
    )
    return complete


def capture_untimed_profile(
    torch: Any,
    compiled: Any,
    x: Any,
    out_dir: Path,
    *,
    invocations: int = DEFAULT_INVOCATIONS,
    record_label: str = DEFAULT_RECORD_LABEL,
    counter_snapshot_fn: Optional[Callable[[], Mapping[str, int]]] = None,
) -> Dict[str, Any]:
    """Capture an untimed CPU+CUDA trace using the measured call context.

    Success requires all of the following:

    * every call completed under ``torch.no_grad()``;
    * positive CPU and CUDA-event timing evidence for every call;
    * no compiler-counter delta, when a snapshot callback is supplied;
    * a CUDA runtime/driver event named ``cudaGraphLaunch*``; and
    * at least one event whose category contains ``kernel``.

    CPU annotations and CUDA-event timings are retained as diagnostic evidence,
    but neither is counted as proof of a CUDA Graph launch.
    """

    out_dir = Path(out_dir)
    trace_path = out_dir / "profile.chrome.json"
    result: Dict[str, Any] = {
        "requested": True,
        "required_scope": "seed 0 nested_break_depth8 only",
        "trace_path": str(trace_path),
        "asserted_cuda_graph_capture_count": False,
        "cudagraph_step_marker_used": False,
        "invocation_policy": (
            "same compiled callable and output-release policy as measured calls; "
            "profiler wraps untimed calls only"
        ),
        "invocation_context": {
            "grad_mode": "disabled via torch.no_grad",
            "matches_timed_workload": True,
        },
        "output_lifetime": "released before each subsequent invocation",
        "event_evidence_role": (
            "CPU and CUDA-event timings corroborate completed stream work only; "
            "they do not prove CUDA Graph capture or replay"
        ),
        "launch_event_policy": (
            "named cudaGraphLaunch event in cuda_runtime or cuda_driver category; "
            "CPU/user annotations never count"
        ),
        "compiler_counter_audit": {
            "requested": counter_snapshot_fn is not None,
            "role": (
                "pre/post profile audit for recompilation; counter stability "
                "does not prove CUDA Graph capture or replay"
            ),
        },
        "evidence_complete": False,
    }

    completed = 0
    cpu_call_durations_ns: List[int] = []
    cuda_event_elapsed_ms: List[float] = []
    cuda_synchronize_completed = False
    counter_before: Dict[str, int] = {}
    counter_after: Dict[str, int] = {}
    counter_delta: Dict[str, int] = {}
    counter_audit_complete = counter_snapshot_fn is None
    counter_stable = counter_snapshot_fn is None

    try:
        if not isinstance(invocations, int) or isinstance(invocations, bool):
            raise TypeError("invocations must be an integer")
        if invocations <= 0:
            raise ValueError("invocations must be positive")
        if not isinstance(record_label, str) or not record_label:
            raise ValueError("record_label must be a nonempty string")

        out_dir.mkdir(parents=True, exist_ok=True)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        supported_activities_fn = getattr(
            torch.profiler, "supported_activities", None
        )
        if not callable(supported_activities_fn):
            raise RuntimeError("torch.profiler.supported_activities is unavailable")
        supported_activities = set(supported_activities_fn())
        cuda_activity_supported = (
            torch.profiler.ProfilerActivity.CUDA in supported_activities
        )
        result["cuda_profiler_activity_supported"] = cuda_activity_supported
        if not cuda_activity_supported:
            raise RuntimeError(
                "CUDA profiler activity is unavailable; Chrome CUDA evidence "
                "cannot be retained"
            )

        event_pairs = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(invocations)
        ]

        with torch.no_grad():
            grad_enabled = bool(torch.is_grad_enabled())
            result["invocation_context"]["grad_enabled_inside_capture"] = grad_enabled
            if grad_enabled:
                raise RuntimeError("torch.no_grad did not disable grad mode")

            if counter_snapshot_fn is not None:
                counter_before = _validated_counter_snapshot(counter_snapshot_fn())

            with torch.profiler.profile(activities=activities) as profiler:
                for start_event, end_event in event_pairs:
                    started_ns = time.perf_counter_ns()
                    with torch.profiler.record_function(record_label):
                        start_event.record()
                        profile_output = compiled(x)
                        end_event.record()
                    finished_ns = time.perf_counter_ns()
                    del profile_output
                    cpu_call_durations_ns.append(finished_ns - started_ns)
                    completed += 1
                torch.cuda.synchronize()
                cuda_synchronize_completed = True

            if counter_snapshot_fn is not None:
                counter_after = _validated_counter_snapshot(counter_snapshot_fn())
                counter_delta = _counter_delta(counter_before, counter_after)
                counter_audit_complete = True
                counter_stable = not counter_delta

        result["compiler_counter_audit"].update(
            {
                "complete": counter_audit_complete,
                "before_profile": counter_before,
                "after_profile": counter_after,
                "delta": counter_delta,
                "stable": counter_stable,
            }
        )

        for start_event, end_event in event_pairs:
            cuda_event_elapsed_ms.append(float(start_event.elapsed_time(end_event)))

        invocation_evidence_complete = _update_invocation_evidence(
            result,
            requested=invocations,
            completed=completed,
            cpu_call_durations_ns=cpu_call_durations_ns,
            cuda_event_elapsed_ms=cuda_event_elapsed_ms,
            cuda_synchronize_completed=cuda_synchronize_completed,
        )

        profiler.export_chrome_trace(str(trace_path))
        events = _read_trace_events(trace_path)
        graph_launch_names: List[str] = []
        kernel_names: List[str] = []
        for event in events:
            name = str(event.get("name", ""))
            category = str(event.get("cat", ""))
            normalized_name = "".join(name.lower().split())
            if "cudagraphlaunch" in normalized_name and _is_cuda_api_event(event):
                graph_launch_names.append(name)
            if "kernel" in category.lower():
                kernel_names.append(name)

        trace_markers_complete = bool(graph_launch_names and kernel_names)
        result.update(
            {
                "trace_sha256": _sha256_file(trace_path),
                "trace_bytes": trace_path.stat().st_size,
                "trace_event_count": len(events),
                "cuda_graph_launch_event_count": len(graph_launch_names),
                "cuda_graph_launch_event_names": sorted(set(graph_launch_names))[:50],
                "cuda_kernel_event_count": len(kernel_names),
                "cuda_kernel_event_names": sorted(set(kernel_names))[:50],
                "trace_markers_complete": trace_markers_complete,
            }
        )

        counter_gate_complete = bool(counter_audit_complete and counter_stable)
        evidence_complete = bool(
            trace_markers_complete
            and invocation_evidence_complete
            and counter_gate_complete
        )
        result["evidence_complete"] = evidence_complete
        if not trace_markers_complete:
            result.update(
                {
                    "status": "failed_missing_required_markers",
                    "error": (
                        "Chrome trace lacks a CUDA runtime/driver "
                        "cudaGraphLaunch event or CUDA kernel marker; no "
                        "capture-count or replay claim is made"
                    ),
                }
            )
        elif not invocation_evidence_complete:
            result.update(
                {
                    "status": "failed_incomplete_invocation_evidence",
                    "error": (
                        "Chrome markers were present but per-invocation CPU/CUDA "
                        "event evidence was incomplete; no replay claim is made"
                    ),
                }
            )
        elif not counter_gate_complete:
            result.update(
                {
                    "status": "failed_profile_recompiled",
                    "error": (
                        "Compiler counters changed during the untimed profile; "
                        "the trace does not establish replay of the warmed variant"
                    ),
                }
            )
        else:
            result["status"] = "captured"
    except Exception as exc:
        valid_requested_count = (
            invocations
            if isinstance(invocations, int)
            and not isinstance(invocations, bool)
            and invocations > 0
            else 0
        )
        _update_invocation_evidence(
            result,
            requested=valid_requested_count,
            completed=completed,
            cpu_call_durations_ns=cpu_call_durations_ns,
            cuda_event_elapsed_ms=cuda_event_elapsed_ms,
            cuda_synchronize_completed=cuda_synchronize_completed,
        )
        result["compiler_counter_audit"].update(
            {
                "complete": counter_audit_complete,
                "before_profile": counter_before,
                "after_profile": counter_after,
                "delta": counter_delta,
                "stable": counter_stable,
            }
        )
        result.update(
            {
                "status": "failed",
                "evidence_complete": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return result


__all__ = ["capture_untimed_profile"]
