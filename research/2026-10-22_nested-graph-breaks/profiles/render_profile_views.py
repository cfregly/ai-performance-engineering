#!/usr/bin/env python3
"""Render exact retained Nsys spans and verbatim single-kernel NCU values.

The Nsys selector is bound to the benchmark source protocol. It selects the
first synchronized latency invocation after 10 cold calls and 25 warmups. The
first compiled call has no replay, so 34 replay-bearing calls precede the
target. False contributes 16 graph launches per call and True contributes 2.

SQLite inputs are opened read-only. The renderer fails if the graph-launch
ordinal, synchronization boundaries, launch count, or kernel attribution is
not proven. It does not calculate a CPU/GPU fraction or a performance ratio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCHEMA = "nested-graph-breaks-profile-views/v1"
SOURCE_COLD_CALLS = 10
SOURCE_WARMUP_CALLS = 25
SOURCE_LATENCY_TRIALS = 100
SOURCE_THROUGHPUT_BLOCKS = 10
SOURCE_THROUGHPUT_CALLS_PER_BLOCK = 50
SOURCE_FINAL_CORRECTNESS_CALLS = 1
PRIOR_REPLAY_INVOCATIONS = SOURCE_COLD_CALLS - 1 + SOURCE_WARMUP_CALLS
TOTAL_REPLAY_INVOCATIONS = (
    SOURCE_COLD_CALLS
    - 1
    + SOURCE_WARMUP_CALLS
    + SOURCE_LATENCY_TRIALS
    + SOURCE_THROUGHPUT_BLOCKS * SOURCE_THROUGHPUT_CALLS_PER_BLOCK
    + SOURCE_FINAL_CORRECTNESS_CALLS
)
TARGET_COMPILED_CALL_ORDINAL = SOURCE_COLD_CALLS + SOURCE_WARMUP_CALLS + 1
GRAPH_CALLS_PER_INVOCATION = {"false": 16, "true": 2}

NCU_METRICS = (
    "gpu__time_duration.avg",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "profiler__replayer_passes",
)


class RenderError(RuntimeError):
    """The retained input cannot prove the requested view."""


@dataclass(frozen=True)
class RuntimeEvent:
    start: int
    end: int
    correlation_id: int | None
    global_tid: int | None
    name: str


@dataclass(frozen=True)
class KernelEvent:
    start: int
    end: int
    correlation_id: int | None
    stream_id: int
    name: str


@dataclass(frozen=True)
class GraphTraceEvent:
    start: int
    end: int
    correlation_id: int | None
    stream_id: int
    graph_id: int
    graph_exec_id: int


@dataclass(frozen=True)
class InvocationSelection:
    arm: str
    graph_calls_per_invocation: int
    prior_replay_invocations: int
    target_compiled_call_ordinal: int
    graph_launch_ordinal_start: int
    graph_launch_ordinal_end: int
    pre_sync: RuntimeEvent
    post_sync: RuntimeEvent
    graph_launches: tuple[RuntimeEvent, ...]
    graph_executions: tuple[GraphTraceEvent, ...]
    runtime_events: tuple[RuntimeEvent, ...]
    kernels: tuple[KernelEvent, ...]


@dataclass(frozen=True)
class NcuPage:
    path: str
    kernel_name: str
    device_name: str
    units: Mapping[str, str]
    values: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path) -> Path:
    if path.is_symlink():
        raise RenderError(f"input must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RenderError(f"input is not a regular file: {resolved}")
    return resolved


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_member_for(path: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    members = manifest.get("members")
    if not isinstance(members, list) and isinstance(manifest.get("files"), list):
        members = [dict(item, local_path=item.get("path")) for item in manifest["files"]]
    if not isinstance(members, list):
        raise RenderError("manifest lacks a members or files list")
    exact = [
        item
        for item in members
        if isinstance(item, Mapping)
        and isinstance(item.get("local_path"), str)
        and Path(str(item["local_path"])).resolve() == path
    ]
    if len(exact) == 1:
        return exact[0]
    by_name = [
        item
        for item in members
        if isinstance(item, Mapping)
        and isinstance(item.get("local_path"), str)
        and Path(str(item["local_path"])).name == path.name
    ]
    if len(by_name) != 1:
        raise RenderError(
            f"transport manifest does not identify one member for {path.name}"
        )
    return by_name[0]


def verify_manifest_input(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _require_regular_file(path)
    member = _manifest_member_for(path, manifest)
    expected_sha256 = member.get("sha256")
    expected_bytes = member.get("bytes")
    observed_sha256 = _sha256_file(path)
    observed_bytes = path.stat().st_size
    if expected_sha256 != observed_sha256:
        raise RenderError(
            f"transport hash mismatch for {path.name}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    if expected_bytes != observed_bytes:
        raise RenderError(
            f"transport size mismatch for {path.name}: "
            f"{observed_bytes} != {expected_bytes}"
        )
    return {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "archive_member": member.get("archive_member"),
    }


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = path.as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _require_columns(
    connection: sqlite3.Connection, table: str, required: set[str]
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    columns = {str(row[1]) for row in rows}
    missing = sorted(required - columns)
    if missing:
        raise RenderError(f"{table} lacks required columns: {missing}")


def _read_nsys_events(
    path: Path,
) -> tuple[list[RuntimeEvent], list[KernelEvent], list[GraphTraceEvent]]:
    path = _require_regular_file(path)
    connection = _open_sqlite_read_only(path)
    try:
        _require_columns(connection, "StringIds", {"id", "value"})
        _require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            {"start", "end", "correlationId", "globalTid", "nameId"},
        )
        _require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            {"start", "end", "correlationId", "streamId", "shortName"},
        )
        _require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
            {
                "start",
                "end",
                "correlationId",
                "streamId",
                "graphId",
                "graphExecId",
            },
        )
        runtime_rows = connection.execute(
            """
            SELECT r.start, r.end, r.correlationId, r.globalTid, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id = r.nameId
            ORDER BY r.start, r.end, r.correlationId
            """
        ).fetchall()
        kernel_rows = connection.execute(
            """
            SELECT k.start, k.end, k.correlationId, k.streamId, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            ORDER BY k.start, k.end, k.correlationId
            """
        ).fetchall()
        graph_rows = connection.execute(
            """
            SELECT start, end, correlationId, streamId, graphId, graphExecId
            FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
            ORDER BY start, end, correlationId
            """
        ).fetchall()
    finally:
        connection.close()

    runtime = [
        RuntimeEvent(
            start=int(row["start"]),
            end=int(row["end"]),
            correlation_id=(
                None if row["correlationId"] is None else int(row["correlationId"])
            ),
            global_tid=None if row["globalTid"] is None else int(row["globalTid"]),
            name=str(row["name"]),
        )
        for row in runtime_rows
    ]
    kernels = [
        KernelEvent(
            start=int(row["start"]),
            end=int(row["end"]),
            correlation_id=(
                None if row["correlationId"] is None else int(row["correlationId"])
            ),
            stream_id=int(row["streamId"]),
            name=str(row["name"]),
        )
        for row in kernel_rows
    ]
    graph_executions = [
        GraphTraceEvent(
            start=int(row["start"]),
            end=int(row["end"]),
            correlation_id=(
                None if row["correlationId"] is None else int(row["correlationId"])
            ),
            stream_id=int(row["streamId"]),
            graph_id=int(row["graphId"]),
            graph_exec_id=int(row["graphExecId"]),
        )
        for row in graph_rows
    ]
    if any(event.end < event.start for event in runtime):
        raise RenderError("runtime table contains a negative event duration")
    if any(event.end < event.start for event in kernels):
        raise RenderError("kernel table contains a negative event duration")
    if any(event.end < event.start for event in graph_executions):
        raise RenderError("graph trace table contains a negative event duration")
    return runtime, kernels, graph_executions


def _is_graph_launch(name: str) -> bool:
    return name.startswith("cudaGraphLaunch")


def _is_device_sync(name: str) -> bool:
    return name.startswith("cudaDeviceSynchronize")


def _is_kernel_launch_api(name: str) -> bool:
    return name.startswith("cudaLaunchKernel")


def select_first_post_warmup_invocation(
    path: Path, *, arm: str
) -> InvocationSelection:
    if arm not in GRAPH_CALLS_PER_INVOCATION:
        raise RenderError(f"unknown arm: {arm}")
    runtime, kernels, graph_executions = _read_nsys_events(path)
    per_invocation = GRAPH_CALLS_PER_INVOCATION[arm]
    graph_launches = [event for event in runtime if _is_graph_launch(event.name)]
    expected_total = TOTAL_REPLAY_INVOCATIONS * per_invocation
    if len(graph_launches) != expected_total:
        raise RenderError(
            f"{arm} graph-launch total does not prove the source protocol: "
            f"{len(graph_launches)} != {expected_total}"
        )

    offset = PRIOR_REPLAY_INVOCATIONS * per_invocation
    target = tuple(graph_launches[offset : offset + per_invocation])
    if len(target) != per_invocation:
        raise RenderError(f"{arm} target graph-launch slice is incomplete")
    thread_ids = {event.global_tid for event in target}
    if len(thread_ids) != 1 or None in thread_ids:
        raise RenderError(f"{arm} target graph launches lack one CPU thread identity")
    target_tid = next(iter(thread_ids))

    syncs = [
        event
        for event in runtime
        if _is_device_sync(event.name) and event.global_tid == target_tid
    ]
    pre_candidates = [event for event in syncs if event.end <= target[0].start]
    post_candidates = [event for event in syncs if event.start >= target[-1].end]
    if not pre_candidates or not post_candidates:
        raise RenderError(f"{arm} target lacks surrounding cudaDeviceSynchronize calls")
    pre_sync = max(pre_candidates, key=lambda event: (event.end, event.start))
    post_sync = min(post_candidates, key=lambda event: (event.start, event.end))
    if pre_sync.end > target[0].start or post_sync.start < target[-1].end:
        raise RenderError(f"{arm} synchronization boundaries overlap target launches")

    window_graphs = tuple(
        event
        for event in graph_launches
        if event.global_tid == target_tid
        and event.start >= pre_sync.end
        and event.end <= post_sync.start
    )
    if window_graphs != target:
        raise RenderError(
            f"{arm} synchronization window does not isolate the source ordinal"
        )

    target_graph_correlations = {event.correlation_id for event in target}
    selected_graph_executions = tuple(
        event
        for event in graph_executions
        if event.correlation_id in target_graph_correlations
    )
    if len(selected_graph_executions) != per_invocation:
        raise RenderError(
            f"{arm} CUDA Graph execution count does not match graph launches: "
            f"{len(selected_graph_executions)} != {per_invocation}"
        )
    execution_correlations = {
        event.correlation_id for event in selected_graph_executions
    }
    if execution_correlations != target_graph_correlations:
        raise RenderError(f"{arm} CUDA Graph execution correlations are incomplete")

    window_runtime = tuple(
        event
        for event in runtime
        if event.global_tid == target_tid
        and event.start < post_sync.end
        and event.end > pre_sync.start
    )
    window_kernels = tuple(
        event
        for event in kernels
        if event.start >= pre_sync.end and event.end <= post_sync.start
    )
    if len(window_kernels) != per_invocation:
        raise RenderError(
            f"{arm} target kernel count does not match graph calls: "
            f"{len(window_kernels)} != {per_invocation}"
        )
    launch_correlations = {
        event.correlation_id
        for event in window_runtime
        if _is_kernel_launch_api(event.name)
    }
    missing_correlations = sorted(
        {
            event.correlation_id
            for event in window_kernels
            if event.correlation_id not in launch_correlations
        },
        key=lambda value: -1 if value is None else value,
    )
    if missing_correlations:
        raise RenderError(
            f"{arm} kernels lack matching cudaLaunchKernel APIs: {missing_correlations}"
        )

    return InvocationSelection(
        arm=arm,
        graph_calls_per_invocation=per_invocation,
        prior_replay_invocations=PRIOR_REPLAY_INVOCATIONS,
        target_compiled_call_ordinal=TARGET_COMPILED_CALL_ORDINAL,
        graph_launch_ordinal_start=offset + 1,
        graph_launch_ordinal_end=offset + per_invocation,
        pre_sync=pre_sync,
        post_sync=post_sync,
        graph_launches=target,
        graph_executions=selected_graph_executions,
        runtime_events=window_runtime,
        kernels=window_kernels,
    )


def read_ncu_page(path: Path) -> NcuPage:
    path = _require_regular_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 3:
        raise RenderError(
            f"NCU wide page must have header, units, and one data row: {path}"
        )
    header, units_row, values_row = rows
    if not header or len(header) != len(set(header)):
        raise RenderError(f"NCU header is empty or contains duplicates: {path}")
    if len(units_row) != len(header) or len(values_row) != len(header):
        raise RenderError(f"NCU wide page row width mismatch: {path}")
    index = {name: position for position, name in enumerate(header)}
    required = {"Kernel Name", "device__attribute_display_name", *NCU_METRICS}
    missing = sorted(required - set(index))
    if missing:
        raise RenderError(f"NCU wide page lacks required columns: {missing}")
    return NcuPage(
        path=str(path),
        kernel_name=values_row[index["Kernel Name"]],
        device_name=values_row[index["device__attribute_display_name"]],
        units={name: units_row[index[name]] for name in NCU_METRICS},
        values={name: values_row[index[name]] for name in NCU_METRICS},
    )


def _relative_us(timestamp: int, origin: int) -> float:
    return (timestamp - origin) / 1000.0


def _duration_us(start: int, end: int) -> float:
    duration = (end - start) / 1000.0
    if not math.isfinite(duration) or duration < 0.0:
        raise RenderError("event duration is not a finite nonnegative value")
    return duration


def render_nsys_timeline(
    false_selection: InvocationSelection,
    true_selection: InvocationSelection,
    output: Path,
) -> None:
    selections = (false_selection, true_selection)
    durations = [
        _duration_us(item.pre_sync.start, item.post_sync.end) for item in selections
    ]
    shared_limit = max(durations)
    if shared_limit <= 0.0:
        raise RenderError("selected Nsys windows have no positive duration")

    colors = {
        "sync": "#7C3AED",
        "kernel_api": "#2563EB",
        "graph": "#F59E0B",
        "other": "#94A3B8",
        "gpu_kernel": "#10B981",
        "gpu_graph": "#E11D48",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for axis, selection, label in zip(
        axes, selections, ("nested_graph_breaks=False", "nested_graph_breaks=True")
    ):
        origin = selection.pre_sync.start
        lanes = {
            "CUDA sync": 4.2,
            "cudaLaunchKernel": 3.2,
            "cudaGraphLaunch": 2.2,
            "multi_tensor_apply_kernel": 1.2,
            "GPU CUDA Graph execution": 0.2,
        }
        for event in selection.runtime_events:
            if _is_device_sync(event.name):
                lane, color = lanes["CUDA sync"], colors["sync"]
            elif _is_kernel_launch_api(event.name):
                lane, color = lanes["cudaLaunchKernel"], colors["kernel_api"]
            elif _is_graph_launch(event.name):
                lane, color = lanes["cudaGraphLaunch"], colors["graph"]
            else:
                lane, color = lanes["cudaLaunchKernel"], colors["other"]
            axis.broken_barh(
                [
                    (
                        _relative_us(event.start, origin),
                        _duration_us(event.start, event.end),
                    )
                ],
                (lane, 0.55),
                facecolors=color,
                edgecolors=color,
                linewidth=0.6,
            )
        for event in selection.kernels:
            axis.broken_barh(
                [
                    (
                        _relative_us(event.start, origin),
                        _duration_us(event.start, event.end),
                    )
                ],
                (lanes["multi_tensor_apply_kernel"], 0.55),
                facecolors=colors["gpu_kernel"],
                edgecolors=colors["gpu_kernel"],
                linewidth=0.6,
            )
        for event in selection.graph_executions:
            axis.broken_barh(
                [
                    (
                        _relative_us(event.start, origin),
                        _duration_us(event.start, event.end),
                    )
                ],
                (lanes["GPU CUDA Graph execution"], 0.55),
                facecolors=colors["gpu_graph"],
                edgecolors=colors["gpu_graph"],
                linewidth=0.6,
            )
        axis.set_ylim(0.0, 5.1)
        axis.set_yticks([4.475, 3.475, 2.475, 1.475, 0.475])
        axis.set_yticklabels(list(lanes))
        axis.grid(axis="x", color="#CBD5E1", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_title(
            f"{label}  |  graph-launch ordinals "
            f"{selection.graph_launch_ordinal_start}–"
            f"{selection.graph_launch_ordinal_end}  |  "
            f"{len(selection.graph_launches)} launches / executions, "
            f"{len(selection.kernels)} standalone kernels",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
    axes[-1].set_xlim(0.0, shared_limit)
    axes[-1].set_xlabel("Time from pre-invocation cudaDeviceSynchronize start (µs)")
    fig.suptitle(
        "First post-warmup synchronized invocation",
        fontsize=16,
        fontweight="bold",
        x=0.08,
        y=0.985,
        ha="left",
    )
    legend = [
        Patch(color=colors["sync"], label="cudaDeviceSynchronize"),
        Patch(color=colors["kernel_api"], label="cudaLaunchKernel"),
        Patch(color=colors["graph"], label="cudaGraphLaunch"),
        Patch(color=colors["other"], label="other CUDA API"),
        Patch(color=colors["gpu_kernel"], label="multi_tensor_apply_kernel"),
        Patch(color=colors["gpu_graph"], label="GPU CUDA Graph execution"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.945),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.text(
        0.08,
        0.015,
        "Rendered from retained Nsys SQLite in graph-level trace mode. Shared "
        "time scale and exact spans. No clipping, smoothing, interpolation, or "
        "CPU/GPU causal fraction.",
        fontsize=9.5,
        color="#334155",
    )
    fig.tight_layout(rect=(0.06, 0.055, 0.99, 0.88))
    fig.savefig(output, dpi=180, format="png", facecolor="white")
    plt.close(fig)


def render_ncu_table(false_page: NcuPage, true_page: NcuPage, output: Path) -> None:
    if false_page.kernel_name != true_page.kernel_name:
        raise RenderError("NCU pages describe different kernel names")
    if false_page.device_name != true_page.device_name:
        raise RenderError("NCU pages describe different device names")
    for metric in NCU_METRICS:
        if false_page.units[metric] != true_page.units[metric]:
            raise RenderError(f"NCU unit mismatch for {metric}")

    rows = [
        [
            metric,
            false_page.units[metric],
            false_page.values[metric],
            true_page.values[metric],
        ]
        for metric in NCU_METRICS
    ]
    fig, axis = plt.subplots(figsize=(14, 7.2))
    axis.axis("off")
    fig.suptitle(
        "Nsight Compute: one retained kernel record per arm",
        fontsize=16,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    axis.text(
        0.0,
        0.965,
        f"Kernel: {false_page.kernel_name}   |   Device: {false_page.device_name}",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#334155",
        va="top",
    )
    table = axis.table(
        cellText=rows,
        colLabels=("Raw CSV column", "Unit", "False", "True"),
        colWidths=(0.55, 0.14, 0.155, 0.155),
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=(0.0, 0.11, 1.0, 0.79),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        if row > 0 and column >= 2:
            cell.get_text().set_fontfamily("DejaVu Sans Mono")
    axis.text(
        0.0,
        0.035,
        "Values and units are copied verbatim from the two retained wide raw-page "
        "CSV rows. No ratio or aggregate is calculated.",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#334155",
        va="bottom",
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.93))
    fig.savefig(output, dpi=180, format="png", facecolor="white")
    plt.close(fig)


def _event_record(
    event: RuntimeEvent | KernelEvent | GraphTraceEvent,
) -> dict[str, Any]:
    return asdict(event)


def _selection_record(selection: InvocationSelection) -> dict[str, Any]:
    return {
        "arm": selection.arm,
        "source_protocol": {
            "cold_calls": SOURCE_COLD_CALLS,
            "warmup_calls": SOURCE_WARMUP_CALLS,
            "latency_trials": SOURCE_LATENCY_TRIALS,
            "throughput_calls": (
                SOURCE_THROUGHPUT_BLOCKS * SOURCE_THROUGHPUT_CALLS_PER_BLOCK
            ),
            "first_compiled_call_has_replay": False,
        },
        "prior_replay_invocations": selection.prior_replay_invocations,
        "target_compiled_call_ordinal": selection.target_compiled_call_ordinal,
        "graph_calls_per_invocation": selection.graph_calls_per_invocation,
        "graph_launch_ordinal_start": selection.graph_launch_ordinal_start,
        "graph_launch_ordinal_end": selection.graph_launch_ordinal_end,
        "pre_sync": _event_record(selection.pre_sync),
        "post_sync": _event_record(selection.post_sync),
        "graph_launch_count": len(selection.graph_launches),
        "graph_execution_count": len(selection.graph_executions),
        "kernel_count": len(selection.kernels),
        "runtime_event_count": len(selection.runtime_events),
        "graph_launches": [_event_record(event) for event in selection.graph_launches],
        "graph_executions": [
            _event_record(event) for event in selection.graph_executions
        ],
        "kernels": [_event_record(event) for event in selection.kernels],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-sqlite", type=Path, required=True)
    parser.add_argument("--true-sqlite", type=Path, required=True)
    parser.add_argument("--false-ncu-csv", type=Path, required=True)
    parser.add_argument("--true-ncu-csv", type=Path, required=True)
    parser.add_argument("--transport-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise RenderError("--output-dir must be an existing regular directory")
    output_paths = {
        "nsys_timeline": output_dir / "nsys-first-post-warmup-invocation.png",
        "ncu_table": output_dir / "ncu-single-kernel-metrics.png",
        "manifest": output_dir / "profile-view-manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise RenderError(f"refusing to overwrite existing outputs: {existing}")

    manifest_path = _require_regular_file(args.transport_manifest)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise RenderError("transport manifest root must be a JSON object")
    input_paths = {
        "false_sqlite": _require_regular_file(args.false_sqlite),
        "true_sqlite": _require_regular_file(args.true_sqlite),
        "false_ncu_csv": _require_regular_file(args.false_ncu_csv),
        "true_ncu_csv": _require_regular_file(args.true_ncu_csv),
    }
    input_records = {
        name: verify_manifest_input(path, manifest)
        for name, path in input_paths.items()
    }

    false_selection = select_first_post_warmup_invocation(
        input_paths["false_sqlite"], arm="false"
    )
    true_selection = select_first_post_warmup_invocation(
        input_paths["true_sqlite"], arm="true"
    )
    false_page = read_ncu_page(input_paths["false_ncu_csv"])
    true_page = read_ncu_page(input_paths["true_ncu_csv"])

    render_nsys_timeline(
        false_selection, true_selection, output_paths["nsys_timeline"]
    )
    render_ncu_table(false_page, true_page, output_paths["ncu_table"])
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "inputs": input_records,
        "transport_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "selection": {
            "false": _selection_record(false_selection),
            "true": _selection_record(true_selection),
        },
        "ncu_scope": {
            "rows_per_arm": 1,
            "false_kernel_name": false_page.kernel_name,
            "true_kernel_name": true_page.kernel_name,
            "metrics": list(NCU_METRICS),
            "values_are_verbatim_strings": True,
            "ratios_calculated": False,
        },
        "outputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in output_paths.items()
            if name != "manifest"
        },
        "limits": [
            "Rendered from retained Nsys SQLite, not a native GUI screenshot.",
            "Graph-level trace spans are shown without reconstructed node-level work.",
            "Standalone multi_tensor_apply_kernel rows are separate from graph executions.",
            "The Nsys panels use one shared time scale and exact event spans.",
            "No clipping, smoothing, interpolation, or causal CPU/GPU fraction is used.",
            "NCU values and units are copied from one retained raw-page row per arm.",
            "No benchmark, profiler, export, or GPU work was run.",
        ],
    }
    _atomic_json(output_paths["manifest"], payload)
    print(output_paths["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
