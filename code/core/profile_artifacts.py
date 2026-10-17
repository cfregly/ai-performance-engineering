"""
Shared helpers for exporting benchmark data and loading profiling artifacts.

These utilities are reused by the dashboard handler, CLI, and MCP layers to
avoid duplicating file-system scans and CSV/report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

CODE_ROOT = Path(__file__).resolve().parent.parent


def _add_warning(payload: Dict[str, Any], message: str) -> None:
    warnings = payload.setdefault("warnings", [])
    if message not in warnings:
        warnings.append(message)


def _read_json_file(path: Path, *, label: str) -> Tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Failed to read {label} JSON from {path}: {exc}"


def _read_json_object(path: Path, *, label: str) -> Tuple[Dict[str, Any] | None, str | None]:
    payload, warning = _read_json_file(path, label=label)
    if warning is not None:
        return None, warning
    if not isinstance(payload, dict):
        return None, (
            f"Failed to read {label} JSON from {path}: expected JSON object, "
            f"got {type(payload).__name__}"
        )
    return payload, None


# =============================================================================
# Benchmark exports
# =============================================================================

def export_benchmarks_csv(data: Dict[str, Any]) -> str:
    """Export benchmark results to CSV format."""
    benchmarks = data.get("benchmarks", [])

    lines = [
        "Benchmark,Baseline Time (ms),Optimized Time (ms),Speedup,Chapter/Lab,Status,Pair Role,Chapter Alignment,Story Note"
    ]

    for b in benchmarks:
        name = b.get("name", "Unknown")
        baseline = b.get("baseline_time_ms", 0)
        optimized = b.get("optimized_time_ms", 0)
        speedup = b.get("speedup", 1.0)
        chapter = b.get("chapter", "")
        status = "✓ Optimized" if speedup > 1.1 else "⚠ Needs Work"
        pair_role = b.get("pair_role", "") or ""
        chapter_alignment = b.get("chapter_alignment", "") or ""
        story_note = str(b.get("story_note", "") or "").replace('"', '""')

        lines.append(
            f'"{name}",{baseline:.4f},{optimized:.4f},{speedup:.2f},"{chapter}","{status}",'
            f'"{pair_role}","{chapter_alignment}","{story_note}"'
        )

    return "\n".join(lines)


def export_detailed_csv(data: Dict[str, Any]) -> str:
    """Export detailed benchmark results including all metrics."""
    benchmarks = data.get("benchmarks", [])

    lines = [
        "Benchmark,Chapter,Baseline Time (ms),Optimized Time (ms),Speedup,"
        "Baseline Memory (MB),Optimized Memory (MB),Memory Reduction (%),"
        "Techniques Applied,LLM Patches,Patch Success Rate,Pair Role,Chapter Alignment,Story Note"
    ]

    for b in benchmarks:
        name = b.get("name", "Unknown")
        chapter = b.get("chapter", "")
        baseline = b.get("baseline_time_ms") or 0
        optimized = b.get("optimized_time_ms") or 0
        speedup = b.get("speedup", 1.0)
        baseline_mem = b.get("baseline_memory_mb") or 0
        optimized_mem = b.get("optimized_memory_mb") or 0
        mem_reduction = (
            (baseline_mem - optimized_mem) / baseline_mem * 100 if baseline_mem > 0 else 0
        )
        techniques = b.get("techniques", [])
        techniques_str = "; ".join(techniques) if techniques else ""
        llm_patches = b.get("llm_patches_applied", 0)
        patch_success = b.get("patch_success_rate", 0)
        pair_role = b.get("pair_role", "") or ""
        chapter_alignment = b.get("chapter_alignment", "") or ""
        story_note = str(b.get("story_note", "") or "").replace('"', '""')

        lines.append(
            f'"{name}","{chapter}",{baseline:.4f},{optimized:.4f},{speedup:.2f},'
            f"{baseline_mem:.2f},{optimized_mem:.2f},{mem_reduction:.1f},"
            f'"{techniques_str}",{llm_patches},{patch_success:.0f},"{pair_role}","{chapter_alignment}","{story_note}"'
        )

    return "\n".join(lines)


# =============================================================================
# Profiling artifact loaders
# =============================================================================
def _iter_profile_dirs(code_root: Path) -> List[Path]:
    runs_root = code_root / "artifacts" / "runs"
    profile_dirs: List[Path] = []
    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            profiles_dir = run_dir / "profiles"
            if profiles_dir.exists():
                profile_dirs.append(profiles_dir)
    return profile_dirs


# The flame graph sums complete ("ph": "X") events longer than this.
FLAME_MIN_EVENT_US = 10
# Each category keeps only its longest event names.
FLAME_TOP_EVENTS_PER_CATEGORY = 20


def chrome_trace_events(payload: Any, path: Path) -> List[Any] | None:
    """Return the event list of a Chrome trace payload, or None for any other JSON.

    A Chrome trace is an object with a traceEvents list, or a list that holds
    at least one event object with a "ph" field. Profile directories also hold
    manifests and summaries as JSON, and those return None.
    """
    if isinstance(payload, dict):
        if "traceEvents" not in payload:
            return None
        events = payload["traceEvents"]
        if not isinstance(events, list):
            raise ValueError(f"traceEvents in {path} is a {type(events).__name__}, not a list")
        return events
    if isinstance(payload, list) and any(isinstance(event, dict) and "ph" in event for event in payload):
        return payload
    return None


def _is_json_lines(text: str) -> bool:
    """Return whether text is JSON Lines: two or more lines that each hold one JSON object."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            return False
        if not isinstance(record, dict):
            return False
    return True


def find_newest_chrome_trace(code_root: Path) -> Dict[str, Any]:
    """Find the newest JSON file under artifacts/runs/*/profiles that holds a Chrome trace.

    Profile directories also hold manifests, summaries and metadata as JSON,
    so each file is parsed, newest first, until one holds trace events (see
    chrome_trace_events). A JSON Lines file, two or more lines that each hold
    one JSON object, is not a Chrome trace. It is skipped and listed in
    skipped_json_lines. HTA capture saves `nsys export --type=json` output in
    profile directories with a .json suffix, so such an export never stops the
    search, whether it is one JSON document or JSON Lines. Any other file that
    does not parse stops the search with error_type trace_read_error, so a
    broken trace is reported instead of passed over.

    Returns {"trace_path", "events", "skipped_json_lines"} when a trace is
    found. Otherwise returns success False with error, error_type
    (trace_read_error or no_chrome_trace) and skipped_json_lines.
    """
    runs_root = code_root / "artifacts" / "runs"
    candidates = sorted(
        (path for path in runs_root.glob("*/profiles/**/*.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    skipped_json_lines: List[str] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                if not _is_json_lines(text):
                    raise
                skipped_json_lines.append(str(path))
                continue
            events = chrome_trace_events(payload, path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return {
                "success": False,
                "error": f"Could not check {path} for Chrome trace events: {exc}",
                "error_type": "trace_read_error",
                "unreadable_path": str(path),
                "skipped_json_lines": skipped_json_lines,
            }
        if events is not None:
            return {"trace_path": path, "events": events, "skipped_json_lines": skipped_json_lines}
    json_lines_note = (
        f" {len(skipped_json_lines)} of them are JSON Lines files."
        if skipped_json_lines
        else ""
    )
    return {
        "success": False,
        "error": (
            f"No Chrome trace found under {runs_root}/*/profiles. None of the "
            f"{len(candidates)} JSON file(s) there holds a traceEvents list or an array of trace events."
            f"{json_lines_note}"
        ),
        "error_type": "no_chrome_trace",
        "searched_root": str(runs_root),
        "skipped_json_lines": skipped_json_lines,
    }


def flame_graph_from_trace_events(events: List[Any], trace_path: Path) -> Dict[str, Any]:
    """Sum the durations of complete trace events by category and event name."""
    category_times: Dict[str, Dict[str, float]] = {}
    ignored_events = 0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("dur", 0), (int, float)):
            ignored_events += 1
            continue
        if event.get("ph") == "X" and event.get("dur", 0) > FLAME_MIN_EVENT_US:
            names = category_times.setdefault(event.get("cat", "other"), {})
            name = event.get("name", "unknown")
            names[name] = names.get(name, 0) + event["dur"]
    children = [
        {
            "name": category,
            "value": sum(names.values()),
            "children": [
                {"name": name, "value": value, "children": []}
                for name, value in sorted(names.items(), key=lambda item: -item[1])[
                    :FLAME_TOP_EVENTS_PER_CATEGORY
                ]
            ],
        }
        for category, names in category_times.items()
    ]
    flame_data: Dict[str, Any] = {
        "name": "GPU Execution",
        "value": sum(child["value"] for child in children),
        "children": children,
        "trace_path": str(trace_path),
        "trace_file": trace_path.name,
    }
    if ignored_events:
        _add_warning(flame_data, f"Ignored {ignored_events} malformed trace event(s) in {trace_path}")
    return flame_data


def _record_trace_selection_failure(payload: Dict[str, Any], selection: Dict[str, Any]) -> None:
    """Report a failed find_newest_chrome_trace in a dashboard payload."""
    if selection["error_type"] == "no_chrome_trace":
        payload["message"] = f"{selection['error']} Run benchmarks with profiling enabled."
        return
    payload["trace_path"] = selection["unreadable_path"]
    payload["error"] = selection["error"]
    _add_warning(payload, selection["error"])


def load_flame_graph_data(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load flame graph data from the newest Chrome trace under artifacts/runs/*/profiles."""
    selection = find_newest_chrome_trace(code_root)
    if selection.get("error"):
        flame_data: Dict[str, Any] = {"name": "GPU Execution", "value": 0, "children": []}
        _record_trace_selection_failure(flame_data, selection)
        return flame_data
    flame_data = flame_graph_from_trace_events(selection["events"], selection["trace_path"])
    if not selection["events"]:
        _add_warning(flame_data, f"Chrome trace {selection['trace_path']} has no events")
    return flame_data


def load_memory_timeline(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load memory usage timeline data."""
    memory_data: Dict[str, Any] = {
        "timeline": [],
        "peak_mb": 0,
        "summary": {
            "total_allocated_mb": 0,
            "peak_allocated_mb": 0,
            "num_allocations": 0,
        },
        "has_real_data": False,
    }

    profile_dirs = _iter_profile_dirs(code_root)

    memory_files: List[Path] = []
    for profile_dir in profile_dirs:
        if profile_dir.exists():
            memory_files.extend(profile_dir.glob("**/*memory*.json"))
            memory_files.extend(profile_dir.glob("**/*memory*.pickle"))

    if memory_files:
        memory_files = sorted(memory_files, key=lambda f: f.stat().st_mtime, reverse=True)
        memory_path = memory_files[0]
        memory_data["artifact_path"] = str(memory_path)
        try:
            if memory_path.suffix == ".json":
                data, warning = _read_json_object(memory_path, label="memory timeline")
                if warning is not None:
                    raise ValueError(warning)
                if data is None:
                    raise ValueError(f"memory timeline reader returned no payload for {memory_path}")
                memory_data.update(data)
                memory_data["has_real_data"] = True
            else:
                raise ValueError(
                    f"Unsupported memory timeline artifact format at {memory_path}: "
                    f"expected .json, got {memory_path.suffix or '<none>'}"
                )
        except Exception as e:
            memory_data["error"] = str(e)
            _add_warning(memory_data, str(e))
    else:
        memory_data["message"] = "No memory profile data found. Run with: python -m torch.profiler"

    return memory_data


def load_cpu_gpu_timeline(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load CPU/GPU parallel timeline data from the newest Chrome trace."""
    timeline_data: Dict[str, Any] = {
        "cpu": [],
        "gpu": [],
        "streams": {},
        "summary": {
            "total_time_ms": 0,
            "cpu_time_ms": 0,
            "gpu_time_ms": 0,
            "overlap_ms": 0,
        },
    }

    selection = find_newest_chrome_trace(code_root)
    if selection.get("error"):
        _record_trace_selection_failure(timeline_data, selection)
    else:
        trace_path = selection["trace_path"]
        events = selection["events"]
        timeline_data["trace_path"] = str(trace_path)
        if not events:
            _add_warning(timeline_data, f"Chrome trace {trace_path} has no events")
        try:
            ignored_events = 0

            min_ts = float("inf")
            max_ts = 0

            for event in events[:500]:  # Limit for performance
                if not isinstance(event, dict):
                    ignored_events += 1
                    continue
                if event.get("ph") != "X":
                    continue

                ts = event.get("ts", 0)
                dur = event.get("dur", 0)
                name = event.get("name", "")
                cat = event.get("cat", "").lower()

                min_ts = min(min_ts, ts)
                max_ts = max(max_ts, ts + dur)

                event_data = {
                    "name": name[:50],
                    "start_ms": ts / 1000,
                    "duration_ms": dur / 1000,
                    # Attempt to propagate source info if present in trace
                    "file": event.get("file") or event.get("args", {}).get("file"),
                    "line": event.get("line") or event.get("args", {}).get("line"),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                    "cat": event.get("cat"),
                }

                if "cuda" in cat or "kernel" in cat:
                    timeline_data["gpu"].append(event_data)
                else:
                    timeline_data["cpu"].append(event_data)

            if max_ts > min_ts:
                timeline_data["summary"]["total_time_ms"] = (max_ts - min_ts) / 1000
            if ignored_events:
                _add_warning(
                    timeline_data,
                    f"Ignored {ignored_events} malformed trace event(s) in {trace_path}",
                )

        except Exception as e:
            timeline_data["error"] = str(e)
            _add_warning(timeline_data, str(e))

    return timeline_data


def load_kernel_breakdown(flame_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Get detailed kernel timing breakdown from flame graph data."""
    kernel_data: Dict[str, Any] = {
        "kernels": [],
        "summary": {
            "total_kernels": 0,
            "total_time_us": 0,
            "avg_kernel_time_us": 0,
        },
        "by_type": {},
    }

    flame = flame_data or load_flame_graph_data()

    all_kernels: List[Dict[str, Any]] = []
    for category in flame.get("children", []):
        cat_name = category.get("name", "other")
        # PyTorch Chrome traces use the ``kernel`` category for GPU kernels.
        # CPU operators, runtime calls and memory transfers are also complete
        # trace events, but reporting them as kernels produces false GPU-time
        # findings in analyze_bottlenecks.
        category_tokens = {
            token.strip().lower()
            for token in str(cat_name).replace(";", ",").split(",")
            if token.strip()
        }
        if not category_tokens.intersection({"kernel", "gpu_kernel", "cuda_kernel"}):
            continue
        for kernel in category.get("children", []):
            all_kernels.append(
                {
                    "name": kernel["name"],
                    "time_us": kernel["value"],
                    "category": cat_name,
                }
            )

            if cat_name not in kernel_data["by_type"]:
                kernel_data["by_type"][cat_name] = 0
            kernel_data["by_type"][cat_name] += kernel["value"]

    kernel_data["kernels"] = sorted(all_kernels, key=lambda k: -k["time_us"])[:50]
    kernel_data["summary"]["total_kernels"] = len(all_kernels)
    kernel_data["summary"]["total_time_us"] = sum(k["time_us"] for k in all_kernels)
    if all_kernels:
        kernel_data["summary"]["avg_kernel_time_us"] = (
            kernel_data["summary"]["total_time_us"] / len(all_kernels)
        )

    return kernel_data


def load_hta_analysis(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load HTA (Holistic Trace Analysis) results."""
    hta_data: Dict[str, Any] = {
        "temporal_breakdown": {
            "compute_pct": 70,
            "idle_pct": 15,
            "communication_pct": 10,
            "memory_pct": 5,
        },
        "top_kernels": [],
        "recommendations": [],
        "bottlenecks": [],
    }

    hta_files = list(code_root.glob("**/hta_report*.json"))
    hta_files.extend(code_root.glob("**/*hta_report*.json"))
    hta_files.extend(code_root.glob("**/hta_analysis*.json"))
    hta_files.extend(code_root.glob("**/*hta_analysis*.json"))

    if hta_files:
        hta_path = sorted(hta_files, key=lambda f: f.stat().st_mtime)[-1]
        hta_data["artifact_path"] = str(hta_path)
        try:
            data, warning = _read_json_object(hta_path, label="HTA analysis")
            if warning is not None:
                raise ValueError(warning)
            if data is None:
                raise ValueError(f"HTA analysis reader returned no payload for {hta_path}")
            hta_data.update(data)
        except Exception as exc:
            hta_data["error"] = str(exc)
            _add_warning(hta_data, str(exc))
    else:
        hta_data["message"] = "No HTA analysis found. Run HTA tooling to generate reports."

    return hta_data


def load_torch_profiler(code_root: Path = CODE_ROOT) -> Dict[str, Any]:
    """Load the latest torch.profiler summary + metadata."""
    data: Dict[str, Any] = {"top_ops": []}
    summary_files = list(code_root.glob("**/torch_profile_summary.json"))
    if not summary_files:
        data["message"] = "No torch.profiler captures found. Run with 'aisp profile torch <script.py>'."
        return data

    summary_path = max(summary_files, key=lambda f: f.stat().st_mtime)
    data["summary_path"] = str(summary_path)
    summary: Dict[str, Any] = {}
    try:
        summary, warning = _read_json_object(summary_path, label="torch profiler summary")
        if warning is not None:
            raise ValueError(warning)
        if summary is None:
            raise ValueError(f"torch profiler summary reader returned no payload for {summary_path}")
        data.update(summary)
    except Exception as exc:
        data["error"] = str(exc)
        _add_warning(data, str(exc))
    meta_path = summary_path.parent / "metadata.json"
    if meta_path.exists():
        data["metadata_path"] = str(meta_path)
        try:
            metadata_payload, warning = _read_json_object(meta_path, label="torch profiler metadata")
            if warning is not None:
                raise ValueError(warning)
            if metadata_payload is None:
                raise ValueError(f"torch profiler metadata reader returned no payload for {meta_path}")
            data["metadata"] = metadata_payload
        except Exception as exc:
            data.setdefault("metadata_error", str(exc))
            _add_warning(data, str(exc))
    trace_path = summary_path.parent / "trace.json"
    if not trace_path.exists():
        alt = summary_path.parent / f"chrome_trace_{summary.get('mode', 'full') if isinstance(summary, dict) else 'full'}.json"
        trace_path = alt if alt.exists() else trace_path
    if trace_path.exists():
        data["trace_path"] = str(trace_path)
    return data
