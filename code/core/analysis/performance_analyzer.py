from __future__ import annotations

import json
import time
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Find the repo root
CODE_ROOT = Path(__file__).parent.parent.parent
from core.benchmark.expectations import SPEEDUP_TOLERANCE
from core.discovery import get_bench_roots


def read_benchmark_json(path: Path) -> dict:
    """Read an explicitly selected local benchmark artifact."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark JSON file not found: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to read benchmark results {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in benchmark results {path}")
    return data


def iter_benchmark_rows(
    data: dict, source: Path, *, errors: Optional[List[dict]] = None
) -> Iterator[tuple[str, dict]]:
    """Validate raw chapter records, retaining valid siblings when scanning."""
    def invalid(message: str) -> None:
        if errors is None:
            raise ValueError(message)
        errors.append({"source_file": str(source), "error": message})

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' list in benchmark results {source}")
    for index, chapter in enumerate(results):
        if not isinstance(chapter, dict) or not isinstance(chapter.get("chapter"), str) or not chapter["chapter"].strip():
            invalid(f"Invalid chapter at results[{index}] in {source}")
            continue
        benchmarks = chapter.get("benchmarks")
        if not isinstance(benchmarks, list):
            invalid(f"Expected 'benchmarks' list for {chapter['chapter']} in {source}")
            continue
        for row_index, bench in enumerate(benchmarks):
            if (
                isinstance(bench, dict)
                and "example" not in bench
                and isinstance(bench.get("name"), str)
                and data.get("run_id") is None
                and not requires_variant_evidence(bench)
            ):
                bench = {**bench, "example": bench["name"]}
            if not isinstance(bench, dict) or not isinstance(bench.get("example"), str) or not bench["example"].strip():
                invalid(f"Invalid benchmark at results[{index}].benchmarks[{row_index}] in {source}")
                continue
            yield chapter["chapter"], bench


def is_finite_measurement(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def has_success_status(status: Any) -> bool:
    return isinstance(status, str) and status.strip().lower() in {"succeeded", "success", "ok"}


def _positive_measurement(value: Any) -> bool:
    return is_finite_measurement(value) and value > 0


def measured_optimizations(bench: dict) -> List[dict]:
    """Return successful variants whose recorded speedup agrees with measured timings."""
    baseline_time = bench.get("baseline_time_ms")
    optimizations = bench.get("optimizations")
    if not isinstance(optimizations, list) or not _positive_measurement(baseline_time):
        return []

    legacy_normalized = (
        "name" in bench
        and "example" not in bench
        and not requires_variant_evidence(bench)
        and has_success_status(bench.get("status"))
    )
    measured = []
    for opt in optimizations:
        if not isinstance(opt, dict):
            continue
        if not has_success_status(opt.get("status")) and not (legacy_normalized and "status" not in opt):
            continue
        time_ms = opt.get("time_ms")
        speedup = opt.get("speedup")
        if not _positive_measurement(time_ms) or not _positive_measurement(speedup):
            continue
        if abs(speedup - baseline_time / time_ms) <= SPEEDUP_TOLERANCE:
            measured.append(opt)
    return measured


def requires_variant_evidence(bench: dict) -> bool:
    """Identify current results with a run ID or a harness timing receipt."""
    return bench.get("run_id") is not None or "baseline_timing_configuration" in bench


def _recorded_speedup(bench: dict) -> Any:
    return bench.get("speedup", bench.get("best_speedup"))


def _benchmark_rejection_reason(bench: dict, measured: List[dict]) -> Optional[str]:
    current = requires_variant_evidence(bench)
    if current and "status" not in bench:
        return "Current benchmark result is missing its final status"
    if "status" in bench and not has_success_status(bench["status"]):
        return f"Benchmark status {bench['status']!r} is ineligible"
    if "status" not in bench and (bench.get("error") or bench.get("skip_reason")):
        return "Historical result has an error or skip reason"
    expectation = bench.get("expectation")
    if bench.get("required_profiler_failures") or (
        isinstance(expectation, dict) and expectation.get("required_profiler_failures")
    ):
        return "Required profiler captures failed"

    speedup = _recorded_speedup(bench)
    if not _positive_measurement(speedup):
        return "No positive measured speedup"
    if "optimizations" not in bench:
        return "Current benchmark result is missing its optimization records" if current else None
    if not measured:
        return "No successful optimization with measured timings and speedup"
    best_observed = max(opt["speedup"] for opt in measured)
    if abs(speedup - best_observed) > SPEEDUP_TOLERANCE:
        return "Recorded best speedup disagrees with the best successful optimization"
    return None


def qualified_speedup(bench: dict) -> Optional[float]:
    """Return a supported speedup from a raw or normalized benchmark row."""
    if _benchmark_rejection_reason(bench, measured_optimizations(bench)) is not None:
        return None
    return _recorded_speedup(bench)


def is_qualified_benchmark(bench: dict) -> bool:
    return qualified_speedup(bench) is not None


def _story_note(story_metadata: Dict[str, Any]) -> str:
    """Build a concise audit/report note from structured story metadata."""
    if not story_metadata:
        return ""

    parts: List[str] = []
    pair_role = str(story_metadata.get("pair_role") or "").strip().lower()
    chapter_alignment = str(story_metadata.get("chapter_alignment") or "").strip().lower()
    comparison_reason = str(story_metadata.get("comparison_reason") or "").strip()
    chapter_native_targets = story_metadata.get("chapter_native_targets") or []

    if pair_role == "comparison" and chapter_alignment == "supplementary":
        parts.append("Supplementary comparison pair.")
    elif pair_role:
        parts.append(f"{pair_role.title()} pair.")

    if comparison_reason:
        parts.append(comparison_reason.rstrip(".") + ".")

    if isinstance(chapter_native_targets, list):
        targets = [str(target).strip() for target in chapter_native_targets if str(target).strip()]
        if targets:
            parts.append(f"Chapter-native targets: {', '.join(targets)}.")

    return " ".join(part for part in parts if part).strip()


def _normalize_benchmark(
    chapter: str, bench: dict, *, source: Path, run_id: Optional[str], timestamp: str
) -> dict:
    name = bench["example"]
    bench_type = bench.get("type") or "python"
    if not isinstance(bench_type, str):
        raise ValueError(f"Invalid type for {chapter}:{name} in {source}")
    numbers = {
        key: bench.get(key)
        for key in ("baseline_time_ms", "best_speedup", "baseline_memory_mb", "best_memory_savings_pct")
    }
    for key, value in numbers.items():
        if value is not None and not is_finite_measurement(value):
            raise ValueError(f"Invalid {key} for {chapter}:{name} in {source}")

    optimizations = bench.get("optimizations", [])
    if not isinstance(optimizations, list) or any(not isinstance(opt, dict) for opt in optimizations):
        raise ValueError(f"Expected optimization objects for {chapter}:{name} in {source}")
    for opt in optimizations:
        for key in ("speedup", "time_ms", "memory_mb"):
            value = opt.get(key)
            if value is not None and not is_finite_measurement(value):
                raise ValueError(f"Invalid optimization {key} for {chapter}:{name} in {source}")

    baseline_time = numbers["baseline_time_ms"]
    speedup = numbers["best_speedup"]
    memory_savings = numbers["best_memory_savings_pct"] or 0
    baseline_memory = numbers["baseline_memory_mb"]
    eligible_optimizations = measured_optimizations(bench)
    best_optimization = max(eligible_optimizations, key=lambda opt: opt["speedup"], default={})
    optimized_memory = min(
        (opt["memory_mb"] for opt in eligible_optimizations if opt.get("memory_mb")),
        default=None,
    )
    if optimized_memory is None and baseline_memory and memory_savings:
        optimized_memory = baseline_memory * (1 - memory_savings / 100)

    baseline_story = bench.get("baseline_story_metadata")
    if not isinstance(baseline_story, dict):
        baseline_story = {}
    optimized_story = best_optimization.get("story_metadata")
    if not isinstance(optimized_story, dict):
        optimized_story = {}
    pair_story = {**{k: v for k, v in optimized_story.items() if v is not None}, **baseline_story}
    gpu_metrics = bench.get("baseline_gpu_metrics") or {}
    if not isinstance(gpu_metrics, dict):
        raise ValueError(f"Invalid baseline_gpu_metrics for {chapter}:{name} in {source}")

    row = {
        "name": name,
        "chapter": chapter,
        "type": bench_type,
        "baseline_time_ms": baseline_time,
        "optimized_time_ms": None,
        "speedup": speedup,
        "baseline_memory_mb": baseline_memory,
        "optimized_memory_mb": optimized_memory,
        "memory_savings_pct": memory_savings,
        "optimization_goal": bench.get("optimization_goal", "speed"),
        "gpu_temp": gpu_metrics.get("temperature_gpu_c"),
        "gpu_power": gpu_metrics.get("power_draw_w"),
        "gpu_util": gpu_metrics.get("utilization_gpu_pct"),
        "error": bench.get("error"),
        "p75_ms": bench.get("baseline_p75_ms"),
        "baseline_story_metadata": baseline_story,
        "best_optimization_story_metadata": optimized_story,
        "story_metadata": pair_story,
        "pair_role": pair_story.get("pair_role"),
        "chapter_alignment": pair_story.get("chapter_alignment"),
        "chapter_native_exemplar": pair_story.get("chapter_native_exemplar"),
        "comparison_reason": pair_story.get("comparison_reason"),
        "chapter_native_targets": pair_story.get("chapter_native_targets"),
        "story_note": _story_note(pair_story),
        "source_file": str(source),
        "source_timestamp": timestamp,
        "run_id": run_id,
    }
    if "status" in bench:
        if not isinstance(bench["status"], str):
            raise ValueError(f"Invalid status for {chapter}:{name} in {source}")
        row["status"] = bench["status"].strip().lower()
    reason = _benchmark_rejection_reason({**bench, **row}, eligible_optimizations)
    if reason is None:
        if "status" not in row:
            row["status_source"] = "legacy_unrecorded"
        row["status"] = "succeeded"
        if best_optimization:
            row["optimized_time_ms"] = best_optimization["time_ms"]
        else:
            row["measurement_source"] = "legacy_speedup_only"
    else:
        row["status"] = row.get("status") or "unknown"
        if has_success_status(row["status"]):
            row["status"] = "unqualified"
        row["error"] = row["error"] or reason

    if "optimizations" in bench:
        row["optimizations"] = []
        for opt in optimizations:
            opt_story = opt.get("story_metadata")
            if not isinstance(opt_story, dict):
                opt_story = {}
            row["optimizations"].append({
                "technique": opt.get("technique", ""),
                "speedup": opt.get("speedup"),
                "time_ms": opt.get("time_ms"),
                "file": opt.get("file", ""),
                "status": opt.get("status"),
                "story_metadata": opt_story,
                "story_note": _story_note(opt_story),
            })
    return row


def _source_timestamp(data: dict, path: Path) -> tuple[float, str]:
    timestamp = data.get("timestamp")
    if timestamp is None or timestamp == "":
        modified = path.stat().st_mtime
        return modified, datetime.fromtimestamp(modified).isoformat()
    if not isinstance(timestamp, str):
        raise ValueError(f"Invalid timestamp in benchmark results {path}")
    try:
        iso_timestamp = f"{timestamp[:-1]}+00:00" if timestamp.endswith("Z") else timestamp
        return datetime.fromisoformat(iso_timestamp).timestamp(), timestamp
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in benchmark results {path}: {timestamp!r}") from exc


def load_benchmark_data(data_file: Optional[Path] = None, bench_roots: Optional[List[Path]] = None) -> dict:
    """Select each target's latest qualified result and retain rejected rows separately."""
    if data_file is not None:
        paths = [Path(data_file)]
    else:
        if bench_roots is None:
            bench_roots = get_bench_roots(repo_root=CODE_ROOT)
        paths = []
        for root in bench_roots:
            paths.extend(sorted((root / "artifacts").rglob("benchmark_test_results.json")))
            default_path = root / "benchmark_test_results.json"
            if default_path.exists():
                paths.append(default_path)

    selected: Dict[tuple, tuple[float, dict]] = {}
    latest_rejected: Dict[tuple, tuple[float, dict]] = {}
    diagnostics = []
    ingest_errors: List[dict] = []
    latest = (float("-inf"), "")
    for path in dict.fromkeys(paths):
        try:
            data = read_benchmark_json(path)
            timestamp = _source_timestamp(data, path)
            run_id = data.get("run_id")
            if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
                raise ValueError(f"Invalid run_id in benchmark results {path}")
            for chapter, bench in iter_benchmark_rows(
                data, path, errors=ingest_errors if data_file is None else None
            ):
                try:
                    row = _normalize_benchmark(
                        chapter, bench, source=path, run_id=run_id, timestamp=timestamp[1]
                    )
                except ValueError as exc:
                    if data_file is not None:
                        raise
                    ingest_errors.append({"source_file": str(path), "error": str(exc)})
                    continue

                key = (row["chapter"], row["name"], row["type"])
                if is_qualified_benchmark(row):
                    destination = selected
                else:
                    diagnostics.append(row)
                    destination = latest_rejected
                previous = destination.get(key)
                if previous is None or timestamp[0] >= previous[0]:
                    destination[key] = (timestamp[0], row)
        except (OSError, ValueError) as exc:
            if data_file is not None:
                raise
            ingest_errors.append({"source_file": str(path), "error": str(exc)})
            continue

        latest = max(latest, timestamp)

    latest_outcomes = {key: row for key, (_, row) in selected.items()}
    for key, (timestamp, rejected) in latest_rejected.items():
        qualified = selected.get(key)
        if qualified is None:
            latest_outcomes[key] = rejected
        elif timestamp > qualified[0]:
            latest_outcomes[key] = rejected
            qualified[1]["newer_diagnostic"] = {
                name: rejected[name] for name in ("status", "error", "run_id", "source_file", "source_timestamp")
            }

    outcomes = list(latest_outcomes.values())
    benchmarks = sorted((row for _, row in selected.values()), key=lambda row: row["speedup"], reverse=True)
    speedups = [row["speedup"] for row in benchmarks]
    memory = [row for row in benchmarks if row["optimization_goal"] == "memory"]
    failed_regression = sum(row["status"] == "failed_regression" for row in outcomes)
    return {
        "timestamp": latest[1] or time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": benchmarks,
        "diagnostics": diagnostics,
        "latest_outcomes": outcomes,
        "ingest_errors": ingest_errors,
        "aggregated": True,
        "source_count": len(benchmarks),
        "summary": {
            "total_benchmarks": len(outcomes),
            "avg_speedup": sum(speedups) / len(speedups) if speedups else 0,
            "max_speedup": max(speedups, default=0),
            "min_speedup": min(speedups, default=0),
            "successful": sum(is_qualified_benchmark(row) for row in outcomes),
            "failed": sum(row["status"].startswith("failed") for row in outcomes) - failed_regression,
            "failed_regression": failed_regression,
            "skipped": sum(row["status"].startswith("skipped") for row in outcomes),
            "qualified": len(benchmarks),
            "diagnostic_rows": len(diagnostics),
            "memory_optimizations": len(memory),
            "speed_optimizations": len(benchmarks) - len(memory),
            "best_memory_savings_pct": max((row["memory_savings_pct"] for row in memory), default=0),
        },
    }


class PerformanceAnalyzer:
    """Stateless benchmark analysis utilities shared by CLI and server."""

    def __init__(self, data_loader: Optional[Callable[[], dict]] = None):
        self._data_loader = data_loader or load_benchmark_data

    def _load_data(self) -> dict:
        return self._data_loader() or {}

    def _qualified_benchmarks(self) -> List[dict]:
        return [row for row in self._load_data().get("benchmarks", []) if is_qualified_benchmark(row)]

    def get_categorized_leaderboards(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        speed_benchmarks = [b for b in benchmarks if b.get("optimization_goal", "speed") == "speed"]
        memory_benchmarks = [b for b in benchmarks if b.get("optimization_goal") == "memory"]
        throughput_benchmarks = [b for b in benchmarks if b.get("optimization_goal") == "throughput"]

        speed_benchmarks.sort(key=lambda x: x.get("speedup", 0), reverse=True)
        memory_benchmarks.sort(key=lambda x: x.get("memory_savings_pct", 0) or 0, reverse=True)
        throughput_benchmarks.sort(key=lambda x: x.get("speedup", 0), reverse=True)

        def format_benchmark(b, category):
            if category == "memory":
                return {
                    "rank": 0,
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "primary_metric": f"-{b.get('memory_savings_pct', 0):.0f}%",
                    "secondary_metric": f"{b.get('speedup', 1):.2f}x speed",
                    "value": b.get("memory_savings_pct", 0) or 0,
                }
            return {
                "rank": 0,
                "name": f"{b.get('chapter')}:{b.get('name')}",
                "primary_metric": f"{b.get('speedup', 1):.2f}x",
                "secondary_metric": f"{b.get('memory_savings_pct', 0):.0f}% mem" if b.get("memory_savings_pct") else "",
                "value": b.get("speedup", 1),
            }

        speed_board = [format_benchmark(b, "speed") for b in speed_benchmarks[:10]]
        for i, b in enumerate(speed_board):
            b["rank"] = i + 1

        memory_board = [format_benchmark(b, "memory") for b in memory_benchmarks[:10]]
        for i, b in enumerate(memory_board):
            b["rank"] = i + 1

        throughput_board = [format_benchmark(b, "throughput") for b in throughput_benchmarks[:10]]
        for i, b in enumerate(throughput_board):
            b["rank"] = i + 1

        return {
            "leaderboards": {
                "speed": {
                    "title": "🚀 Speed Champions",
                    "metric_label": "Speedup",
                    "entries": speed_board,
                    "count": len(speed_benchmarks),
                },
                "memory": {
                    "title": "💾 Memory Champions",
                    "metric_label": "Memory Saved",
                    "entries": memory_board,
                    "count": len(memory_benchmarks),
                },
                "throughput": {
                    "title": "📈 Throughput Champions",
                    "metric_label": "Throughput",
                    "entries": throughput_board,
                    "count": len(throughput_benchmarks),
                },
            },
            "summary": {
                "speed_count": len(speed_benchmarks),
                "memory_count": len(memory_benchmarks),
                "throughput_count": len(throughput_benchmarks),
            },
        }

    def get_pareto_frontier(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        points = []
        for b in benchmarks:
            speedup = b.get("speedup", 1.0)
            memory_savings = b.get("memory_savings_pct", 0) or 0
            points.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": speedup,
                    "memory_savings": memory_savings,
                    "goal": b.get("optimization_goal", "speed"),
                    "is_pareto": False,
                }
            )

        pareto_points: List[Dict[str, Any]] = []
        for i, p in enumerate(points):
            is_dominated = False
            for j, q in enumerate(points):
                if i == j:
                    continue
                if q["speedup"] >= p["speedup"] and q["memory_savings"] >= p["memory_savings"]:
                    if q["speedup"] > p["speedup"] or q["memory_savings"] > p["memory_savings"]:
                        is_dominated = True
                        break
            if not is_dominated:
                p["is_pareto"] = True
                pareto_points.append(p)

        return {
            "pareto_frontier": pareto_points,
            "all_points": points,
            "pareto_count": len(pareto_points),
            "total_count": len(points),
        }

    def get_tradeoff_analysis(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        tradeoffs: List[Dict[str, Any]] = []
        for b in benchmarks:
            speedup = b.get("speedup", 1.0)
            memory_savings = b.get("memory_savings_pct", 0) or 0
            goal = b.get("optimization_goal", "speed")

            if goal == "memory" and memory_savings > 0:
                speed_cost = max(0, 1 - speedup)
                cost_per_percent = (speed_cost * 100) / memory_savings if memory_savings > 0 else 0
                efficiency = memory_savings / (speed_cost * 100 + 1)
                benefit_label = f"-{memory_savings:.0f}% memory"
                cost_label = f"{speed_cost*100:.0f}% slower" if speed_cost > 0 else "No speed loss"
            else:
                speed_gain = max(0, speedup - 1)
                memory_cost = max(0, -memory_savings) if memory_savings else 0
                efficiency = speedup / (memory_cost + 1)
                benefit_label = f"{speedup:.2f}x faster"
                cost_label = f"+{memory_cost:.0f}% memory" if memory_cost > 0 else "No memory increase"
                cost_per_percent = memory_cost / (speed_gain * 100 + 1) if speed_gain > 0 else 0

            tradeoffs.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "goal": goal,
                    "speedup": speedup,
                    "memory_savings": memory_savings,
                    "efficiency_score": round(efficiency, 2),
                    "cost_per_percent": round(cost_per_percent, 3),
                    "benefit": benefit_label,
                    "cost": cost_label,
                }
            )

        tradeoffs.sort(key=lambda x: x["efficiency_score"], reverse=True)

        return {
            "tradeoffs": tradeoffs,
            "best_efficiency": tradeoffs[0] if tradeoffs else None,
            "memory_specialists": [t for t in tradeoffs if t["goal"] == "memory"][:5],
            "speed_specialists": [t for t in tradeoffs if t["goal"] == "speed"][:5],
        }

    def get_constraint_recommendations(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        scenarios: List[Dict[str, Any]] = []

        memory_opts = sorted(
            [b for b in benchmarks if (b.get("memory_savings_pct") or 0) > 10],
            key=lambda x: x.get("memory_savings_pct", 0),
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Memory Constrained",
                "description": "Need to fit larger models/batches in limited VRAM",
                "icon": "💾",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"-{b.get('memory_savings_pct', 0):.0f}% memory",
                        "tradeoff": f"{b.get('speedup', 1):.2f}x speed",
                    }
                    for b in memory_opts[:3]
                ],
            }
        )

        speed_opts = sorted(
            [b for b in benchmarks if b.get("speedup", 1) > 1.5],
            key=lambda x: x.get("speedup", 1),
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Latency Critical",
                "description": "Need lowest possible inference/training time",
                "icon": "⚡",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"{b.get('speedup', 1):.2f}x faster",
                        "tradeoff": f"{b.get('memory_savings_pct', 0):.0f}% mem change"
                        if b.get("memory_savings_pct")
                        else "Minimal memory impact",
                    }
                    for b in speed_opts[:3]
                ],
            }
        )

        balanced = sorted(
            [b for b in benchmarks if b.get("speedup", 1) >= 1.0 and (b.get("memory_savings_pct", 0) or 0) >= 0],
            key=lambda x: x.get("speedup", 1) + (x.get("memory_savings_pct", 0) or 0) / 50,
            reverse=True,
        )
        scenarios.append(
            {
                "name": "Balanced",
                "description": "Want improvements without significant trade-offs",
                "icon": "⚖️",
                "recommendations": [
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "benefit": f"{b.get('speedup', 1):.2f}x speed",
                        "tradeoff": f"{b.get('memory_savings_pct', 0):.0f}% mem"
                        if b.get("memory_savings_pct")
                        else "Neutral",
                    }
                    for b in balanced[:3]
                ],
            }
        )

        training_opts = [b for b in benchmarks if "training" in b.get("name", "").lower()]
        if training_opts:
            scenarios.append(
                {
                    "name": "Large Model Training",
                    "description": "Training models that barely fit in memory",
                    "icon": "🏋️",
                    "recommendations": [
                        {
                            "name": f"{b.get('chapter')}:{b.get('name')}",
                            "benefit": f"-{b.get('memory_savings_pct', 0):.0f}% memory"
                            if b.get("memory_savings_pct", 0) > 0
                            else f"{b.get('speedup', 1):.2f}x speed",
                            "tradeoff": b.get("optimization_goal", "speed"),
                        }
                        for b in training_opts[:3]
                    ],
                }
            )

        return {"scenarios": scenarios, "total_benchmarks": len(benchmarks)}

    def get_whatif_recommendations(self, params: dict) -> dict:
        benchmarks = self._qualified_benchmarks()

        max_vram_gb = float(params.get("vram", [999999])[0])
        max_latency_ms = float(params.get("latency", [999999])[0])
        min_throughput = float(params.get("throughput", [0])[0])
        max_memory_gb = float(params.get("memory_budget", [999999])[0])

        matching: List[Dict[str, Any]] = []
        for b in benchmarks:
            opt_mem_gb = (b.get("optimized_memory_mb") or b.get("baseline_memory_mb") or 0) / 1024
            if opt_mem_gb > max_memory_gb or opt_mem_gb > max_vram_gb:
                continue

            opt_latency_ms = b.get("optimized_time_ms")
            if max_latency_ms < 999999 and (
                not _positive_measurement(opt_latency_ms) or opt_latency_ms > max_latency_ms
            ):
                continue

            opt_throughput = (b.get("throughput") or b.get("speedup", 1)) * 1000
            if opt_throughput < min_throughput:
                continue

            matching.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": b.get("speedup", 1.0),
                    "memory_savings_pct": b.get("memory_savings_pct", 0),
                    "latency_ms": opt_latency_ms,
                    "throughput": opt_throughput,
                }
            )

        matching.sort(key=lambda x: (x["speedup"], x["memory_savings_pct"]), reverse=True)

        best_for_speed = matching[0] if matching else None
        best_for_memory = (
            max(matching, key=lambda x: x["memory_savings_pct"])
            if matching and any(m.get("memory_savings_pct") for m in matching)
            else None
        )

        return {
            "recommendations": matching[:10],
            "matching_count": len(matching),
            "total_benchmarks": len(benchmarks),
            "constraints": {
                "vram_gb": max_vram_gb if max_vram_gb < 999999 else None,
                "latency_ms": max_latency_ms if max_latency_ms < 999999 else None,
                "min_throughput": min_throughput if min_throughput > 0 else None,
                "memory_budget_gb": max_memory_gb if max_memory_gb < 999999 else None,
            },
            "best_for_speed": best_for_speed,
            "best_for_memory": best_for_memory,
        }

    def get_optimization_stacking(self) -> dict:
        optimization_categories = {
            "memory_format": ["quantization", "fp8", "fp4", "int8", "bf16"],
            "attention": ["flash_attention", "flex_attention", "sdpa", "paged_attention"],
            "parallelism": ["tensor_parallel", "pipeline_parallel", "data_parallel", "fsdp"],
            "caching": ["kv_cache", "cuda_graphs", "compile"],
            "memory_saving": ["checkpointing", "activation_checkpoint", "gradient_checkpoint"],
            "communication": ["nccl", "nvlink", "nvshmem"],
        }

        compatible_pairs = [
            ("flash_attention", "quantization", "High", "5-10x speed + 2-4x memory reduction"),
            ("cuda_graphs", "tensor_cores", "High", "Reduced launch overhead + fast math"),
            ("kv_cache", "quantization", "High", "Compressed cache for longer sequences"),
            ("flash_attention", "tensor_parallel", "High", "Scale attention across GPUs"),
            ("compile", "cuda_graphs", "Medium", "Compiled graphs for best performance"),
            ("fsdp", "bf16", "High", "Sharded training with mixed precision"),
            ("checkpointing", "flash_attention", "Medium", "Memory savings stack"),
        ]

        incompatible_pairs = [
            ("cuda_graphs", "dynamic_shapes", "Graphs require static shapes"),
            ("checkpointing", "cuda_graphs", "Checkpointing needs dynamic execution"),
            ("eager_mode", "compile", "Mutually exclusive execution modes"),
        ]

        return {
            "categories": optimization_categories,
            "compatible_combinations": [
                {"opt1": p[0], "opt2": p[1], "synergy": p[2], "benefit": p[3]} for p in compatible_pairs
            ],
            "incompatible_combinations": [{"opt1": p[0], "opt2": p[1], "reason": p[2]} for p in incompatible_pairs],
            "recommended_stacks": [
                {
                    "name": "Maximum Speed (Inference)",
                    "stack": ["flash_attention", "cuda_graphs", "fp8_quantization", "tensor_cores"],
                    "expected_benefit": "10-50x speedup",
                },
                {
                    "name": "Memory Efficient Training",
                    "stack": ["gradient_checkpointing", "fsdp", "bf16", "flash_attention"],
                    "expected_benefit": "2-4x larger models",
                },
                {
                    "name": "Balanced (Speed + Memory)",
                    "stack": ["flash_attention", "kv_cache_quantization", "compile"],
                    "expected_benefit": "3-5x speed, 30-50% memory reduction",
                },
            ],
        }

    def get_power_efficiency(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        efficiency_data: List[Dict[str, Any]] = []
        for b in benchmarks:
            power_w = b.get("gpu_power") or 0
            speedup = b.get("speedup", 1)
            time_ms = b.get("optimized_time_ms")

            if power_w > 0 and _positive_measurement(time_ms):
                ops_per_watt = speedup / (power_w * time_ms / 1000)
                efficiency_data.append(
                    {
                        "name": f"{b.get('chapter')}:{b.get('name')}",
                        "speedup": speedup,
                        "power_w": power_w,
                        "time_ms": time_ms,
                        "ops_per_watt": round(ops_per_watt, 4),
                        "energy_j": round(power_w * time_ms / 1000, 2),
                    }
                )

        efficiency_data.sort(key=lambda x: x["ops_per_watt"], reverse=True)

        return {
            "efficiency_rankings": efficiency_data[:15],
            "most_efficient": efficiency_data[0] if efficiency_data else None,
            "least_efficient": efficiency_data[-1] if efficiency_data else None,
            "avg_power_w": sum(e["power_w"] for e in efficiency_data) / len(efficiency_data) if efficiency_data else 0,
            "total_benchmarks_with_power": len(efficiency_data),
        }

    def get_scaling_analysis(self) -> dict:
        benchmarks = self._qualified_benchmarks()

        scaling_insights = []
        categories: Dict[str, List[Dict[str, Any]]] = {
            "attention": [],
            "matmul": [],
            "memory": [],
            "training": [],
        }

        for b in benchmarks:
            name = b.get("name", "").lower()
            speedup = b.get("speedup", 1)

            if "attention" in name or "flash" in name or "sdpa" in name:
                categories["attention"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "matmul" in name or "gemm" in name:
                categories["matmul"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "memory" in name or "kv" in name or "cache" in name:
                categories["memory"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})
            if "training" in name or "grad" in name:
                categories["training"].append({"name": f"{b.get('chapter')}:{b.get('name')}", "speedup": speedup})

        scaling_recommendations = [
            {
                "factor": "Sequence Length",
                "insight": "Flash Attention scales O(n) vs O(n²) - critical for long sequences",
                "recommendation": "Use Flash Attention for seq_len > 512",
            },
            {
                "factor": "Batch Size",
                "insight": "CUDA Graphs amortize launch overhead - better at larger batches",
                "recommendation": "Use CUDA Graphs for batch_size > 8",
            },
            {
                "factor": "Model Size",
                "insight": "Tensor Parallelism scales linearly with GPU count",
                "recommendation": "Use TP for models > single GPU memory",
            },
            {
                "factor": "Hidden Dimension",
                "insight": "Tensor Cores are most efficient at multiples of 8/16",
                "recommendation": "Pad dimensions to multiples of 16 for Tensor Core utilization",
            },
        ]

        return {
            "categories": {k: sorted(v, key=lambda x: x["speedup"], reverse=True)[:5] for k, v in categories.items()},
            "scaling_recommendations": scaling_recommendations,
            "key_insight": "Optimization impact increases with workload size - small benchmarks may underestimate real-world gains",
        }

    def get_cost_analysis(self, gpu: str = None, custom_rate: float = None) -> dict:
        benchmarks = self._qualified_benchmarks()

        gpu_pricing = {
            "B200": {"hourly": 5.00, "name": "NVIDIA B200", "tflops": 2250},
            "H100": {"hourly": 3.50, "name": "NVIDIA H100", "tflops": 1979},
            "A100": {"hourly": 2.00, "name": "NVIDIA A100", "tflops": 312},
            "L40S": {"hourly": 1.50, "name": "NVIDIA L40S", "tflops": 362},
            "A10G": {"hourly": 1.00, "name": "NVIDIA A10G", "tflops": 125},
            "T4": {"hourly": 0.50, "name": "NVIDIA T4", "tflops": 65},
        }

        selected_gpu = gpu or "B200"
        if custom_rate is not None:
            hourly_rate = custom_rate
        elif selected_gpu in gpu_pricing:
            hourly_rate = gpu_pricing[selected_gpu]["hourly"]
        else:
            hourly_rate = gpu_pricing["B200"]["hourly"]

        cost_analysis: List[Dict[str, Any]] = []
        for b in benchmarks:
            speedup = b.get("speedup", 1)
            if speedup <= 1:
                continue

            baseline_time_ms = b.get("baseline_time_ms")
            optimized_time_ms = b.get("optimized_time_ms")
            if not _positive_measurement(baseline_time_ms) or not _positive_measurement(optimized_time_ms):
                continue

            baseline_ops_per_hour = 3600 * 1000 / baseline_time_ms
            optimized_ops_per_hour = 3600 * 1000 / optimized_time_ms

            baseline_cost_per_m = (hourly_rate / baseline_ops_per_hour) * 1_000_000
            optimized_cost_per_m = (hourly_rate / optimized_ops_per_hour) * 1_000_000
            savings_per_m = baseline_cost_per_m - optimized_cost_per_m
            savings_pct = (savings_per_m / baseline_cost_per_m) * 100 if baseline_cost_per_m > 0 else 0

            cost_analysis.append(
                {
                    "name": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": speedup,
                    "baseline_cost_per_m": round(baseline_cost_per_m, 4),
                    "optimized_cost_per_m": round(optimized_cost_per_m, 4),
                    "savings_per_m": round(savings_per_m, 4),
                    "savings_pct": round(savings_pct, 1),
                }
            )

        cost_analysis.sort(key=lambda x: x["savings_pct"], reverse=True)

        return {
            "gpu_pricing": gpu_pricing,
            "assumed_gpu": selected_gpu,
            "hourly_rate": hourly_rate,
            "available_gpus": list(gpu_pricing.keys()),
            "cost_rankings": cost_analysis[:15],
            "highest_savings": cost_analysis[0] if cost_analysis else None,
            "total_potential_savings": f"Up to {cost_analysis[0]['savings_pct']:.0f}% cost reduction"
            if cost_analysis
            else "N/A",
        }
