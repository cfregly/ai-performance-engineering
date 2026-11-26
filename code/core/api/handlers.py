"""Shared benchmark handlers used by Engine and MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.analysis.performance_analyzer import load_benchmark_data
from core.engine import get_engine
from core.jobs import JobStore


class BenchmarkRequestError(ValueError):
    """An invalid benchmark argument or explicitly selected input file."""


class BenchmarkInputNotFoundError(FileNotFoundError):
    """An explicitly selected benchmark input file does not exist."""


def _require_param(params: Dict[str, Any], name: str) -> Any:
    value = params.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise BenchmarkRequestError(f"{name} is required")
    return value


def _parse_int_param(
    params: Dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    raw = params.get(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkRequestError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise BenchmarkRequestError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise BenchmarkRequestError(f"{name} must be <= {maximum}")
    return value


def _parse_float_param(
    params: Dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    raw = params.get(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkRequestError(f"{name} must be numeric") from exc
    if minimum is not None and value < minimum:
        raise BenchmarkRequestError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise BenchmarkRequestError(f"{name} must be <= {maximum}")
    return value


def _parse_optional_int_param(
    params: Dict[str, Any],
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    raw = params.get(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _parse_str_param(params: Dict[str, Any], name: str) -> Optional[str]:
    value = params.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_bool_param(params: Dict[str, Any], name: str, default: bool = False) -> bool:
    value = params.get(name)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BenchmarkRequestError(f"{name} must be boolean")


def _parse_path_param(params: Dict[str, Any], name: str) -> Path:
    raw = _require_param(params, name)
    path = Path(str(raw)).expanduser()
    if not path.exists():
        raise BenchmarkInputNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise BenchmarkRequestError(f"{name} must be a file: {path}")
    return path.resolve()


def _parse_list_param(params: Dict[str, Any], name: str) -> Optional[List[str]]:
    value = params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or None


def _normalize_status(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"failed_regression", "regression"}:
        return "failed"
    if lowered in {"success", "succeeded", "ok"}:
        return "succeeded"
    if lowered in {"skip", "skipped"}:
        return "skipped"
    if lowered in {"failed", "error"}:
        return "failed"
    return lowered


def _status_bucket(value: Optional[str]) -> Optional[str]:
    normalized = _normalize_status(value)
    if normalized is None:
        return None
    if normalized.startswith("failed_"):
        return "failed"
    return normalized


def _build_summary(
    benchmarks: Sequence[Dict[str, Any]], raw_summary: Dict[str, Any]
) -> Dict[str, Any]:
    total = len(benchmarks)
    succeeded = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "succeeded")
    failed = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "failed")
    skipped = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "skipped")
    avg_speedup = float(raw_summary.get("avg_speedup", 0) or 0)
    max_speedup = float(raw_summary.get("max_speedup", 0) or 0)
    min_speedup = float(raw_summary.get("min_speedup", 0) or 0)
    if not avg_speedup or not max_speedup or not min_speedup:
        speedups = [
            float(b["speedup"])
            for b in benchmarks
            if b.get("speedup") is not None and _status_bucket(b.get("status")) == "succeeded"
        ]
        if speedups:
            avg_speedup = avg_speedup or sum(speedups) / len(speedups)
            max_speedup = max_speedup or max(speedups)
            min_speedup = min_speedup or min(speedups)
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "avg_speedup": avg_speedup,
        "max_speedup": max_speedup,
        "min_speedup": min_speedup,
    }


def _normalize_benchmark_record(bench: Dict[str, Any]) -> Dict[str, Any]:
    status = _normalize_status(bench.get("status"))
    if status is None:
        status = str(bench.get("status", "unknown")).lower() or "unknown"
    normalized = dict(bench)
    normalized["status"] = status
    return normalized


def _sort_benchmarks(
    benchmarks: List[Dict[str, Any]],
    sort_field: str,
    sort_dir: str,
) -> List[Dict[str, Any]]:
    numeric_fields = {"speedup", "baseline_time_ms", "optimized_time_ms"}
    reverse = sort_dir == "desc"
    if sort_field not in {
        "name",
        "chapter",
        "speedup",
        "baseline_time_ms",
        "optimized_time_ms",
        "status",
    }:
        raise BenchmarkRequestError(f"Unsupported sort_field: {sort_field}")

    def _key(item: Dict[str, Any]) -> Tuple:
        value = item.get(sort_field)
        if sort_field in numeric_fields:
            return (float(value or 0),)
        if value is None:
            return ("",)
        return (str(value).lower(),)

    return sorted(benchmarks, key=_key, reverse=reverse)


def _filter_benchmarks(
    benchmarks: Iterable[Dict[str, Any]],
    *,
    search: Optional[str] = None,
    status_filters: Optional[List[str]] = None,
    chapter_filters: Optional[List[str]] = None,
    benchmark_name: Optional[str] = None,
    optimization_goal: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized_status = {_status_bucket(value) for value in status_filters or []}
    normalized_status.discard(None)
    normalized_chapters = {value.lower() for value in chapter_filters or []}
    target_name = benchmark_name.lower() if benchmark_name else None
    search_value = search.lower() if search else None
    goal = optimization_goal.lower() if optimization_goal else None

    results: List[Dict[str, Any]] = []
    for bench in benchmarks:
        status = _status_bucket(bench.get("status"))
        chapter = str(bench.get("chapter", "")).lower()
        name = str(bench.get("name", "")).lower()
        if normalized_status and status not in normalized_status:
            continue
        if normalized_chapters and chapter not in normalized_chapters:
            continue
        if target_name and name != target_name:
            continue
        if goal and str(bench.get("optimization_goal", "")).lower() != goal:
            continue
        if search_value and search_value not in name and search_value not in chapter:
            continue
        results.append(bench)
    return results


def _paginate(items: Sequence[Dict[str, Any]], page: int, page_size: int) -> List[Dict[str, Any]]:
    start = (page - 1) * page_size
    end = start + page_size
    return list(items[start:end])


def _load_run_file(path: Path) -> Dict[str, Any]:
    try:
        data = load_benchmark_data(data_file=path)
    except FileNotFoundError as exc:
        raise BenchmarkInputNotFoundError(str(exc)) from exc
    except ValueError as exc:
        if isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError)):
            raise BenchmarkRequestError(f"Failed to parse JSON: {path}") from exc
        raise BenchmarkRequestError(str(exc)) from exc

    def comparison_record(row: dict) -> dict:
        admitted = row["status"] == "succeeded"
        record = {
            "key": f"{row['chapter']}:{row['name']}",
            "chapter": row["chapter"],
            "name": row["name"],
            "type": row["type"],
            "status": row["status"],
            "baseline_time_ms": row["baseline_time_ms"],
            "optimized_time_ms": row["optimized_time_ms"],
            "speedup": row["speedup"] if admitted else None,
            "optimization_goal": row["optimization_goal"],
            "baseline_memory_mb": row["baseline_memory_mb"],
            "optimized_memory_mb": row["optimized_memory_mb"],
            "memory_savings_pct": row["memory_savings_pct"],
            "error": row["error"],
        }
        if not admitted:
            record["recorded_speedup"] = row["speedup"]
            record["source_file"] = row["source_file"]
            record["run_id"] = row["run_id"]
        return record

    outcomes = [comparison_record(row) for row in data["latest_outcomes"]]
    return {
        "path": str(path),
        "timestamp": data["timestamp"],
        "outcomes": outcomes,
        "diagnostics": [comparison_record(row) for row in data["diagnostics"]],
        "summary": _build_summary(outcomes, {}),
    }


def benchmark_data(params: Dict[str, Any]) -> Dict[str, Any]:
    page = _parse_int_param(params, "page", 1, minimum=1)
    page_size = _parse_int_param(params, "page_size", 50, minimum=1, maximum=500)
    search = _parse_str_param(params, "search")
    sort_field = _parse_str_param(params, "sort_field") or "speedup"
    sort_dir = (_parse_str_param(params, "sort_dir") or "desc").lower()
    if sort_dir not in {"asc", "desc"}:
        raise BenchmarkRequestError("sort_dir must be 'asc' or 'desc'")
    status_filters = _parse_list_param(params, "status")
    chapter_filters = _parse_list_param(params, "chapter")
    benchmark_name = _parse_str_param(params, "benchmark")
    optimization_goal = _parse_str_param(params, "optimization_goal")

    data = get_engine().benchmark.data()
    benchmarks = [_normalize_benchmark_record(b) for b in data.get("benchmarks", [])]
    diagnostics = [_normalize_benchmark_record(b) for b in data.get("diagnostics", [])]
    outcomes = data.get("latest_outcomes", benchmarks + diagnostics)
    raw_summary = data.get("summary", {}) or {}

    include_diagnostics = any(
        _status_bucket(status) != "succeeded" for status in status_filters or []
    )
    filtered = _filter_benchmarks(
        benchmarks + diagnostics if include_diagnostics else benchmarks,
        search=search,
        status_filters=status_filters,
        chapter_filters=chapter_filters,
        benchmark_name=benchmark_name,
        optimization_goal=optimization_goal,
    )
    sorted_items = _sort_benchmarks(filtered, sort_field, sort_dir)
    paged = _paginate(sorted_items, page, page_size)

    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size if page_size else 1
    return {
        "timestamp": data.get("timestamp"),
        "ingest_errors": data.get("ingest_errors", []),
        "summary": _build_summary(outcomes, raw_summary),
        "benchmarks": paged,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
        },
        "filters": {
            "search": search,
            "status": status_filters or [],
            "chapter": chapter_filters or [],
            "benchmark": benchmark_name,
            "optimization_goal": optimization_goal,
            "sort_field": sort_field,
            "sort_dir": sort_dir,
        },
    }


def benchmark_opportunities(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.analysis.optimization_opportunities import (
        analyze_opportunity_file,
        apply_novelty_validation_feedback,
        apply_run_queue_feedback,
        discover_benchmark_target_catalog,
        normalize_candidates,
        rank_opportunities,
        summarize_run_queue_root,
    )

    top = _parse_int_param(params, "top", 10, minimum=0, maximum=500)
    top_n = None if top <= 0 else top
    min_speedup = _parse_float_param(params, "min_speedup", 1.10, minimum=0.0)
    target_speedup = _parse_float_param(params, "target_speedup", 1.50, minimum=0.0)
    min_memory_savings_pct = _parse_float_param(
        params,
        "min_memory_savings_pct",
        10.0,
        minimum=0.0,
        maximum=100.0,
    )
    slow_baseline_ms = _parse_float_param(params, "slow_baseline_ms", 100.0, minimum=0.0)
    include_discovered_targets = _parse_bool_param(params, "include_discovered_targets", False)

    data_file = _parse_str_param(params, "data_file")
    catalog_file = _parse_str_param(params, "catalog_file")
    run_queue_root = _parse_str_param(params, "run_queue_root")
    novelty_queue_root = _parse_str_param(params, "novelty_queue_root")
    bench_root = _parse_str_param(params, "bench_root")
    catalog_path: Optional[Path] = None
    if catalog_file:
        catalog_path = _parse_path_param(params, "catalog_file")

    if data_file:
        path = _parse_path_param(params, "data_file")
        discovered_targets = None
        if include_discovered_targets:
            if bench_root:
                discovered_targets = discover_benchmark_target_catalog(
                    Path(bench_root).expanduser()
                ).get("targets", [])
            else:
                discovered_targets = get_engine().benchmark.targets().get("targets", [])
        result = analyze_opportunity_file(
            path.resolve(),
            catalog_path=catalog_path.resolve() if catalog_path else None,
            target_catalog=discovered_targets,
            top_n=top_n,
            min_speedup=min_speedup,
            target_speedup=target_speedup,
            min_memory_savings_pct=min_memory_savings_pct,
            slow_baseline_ms=slow_baseline_ms,
        )
        if run_queue_root:
            result = apply_run_queue_feedback(
                result, summarize_run_queue_root(Path(run_queue_root).expanduser())
            )
        if novelty_queue_root:
            result = apply_novelty_validation_feedback(
                result, summarize_run_queue_root(Path(novelty_queue_root).expanduser())
            )
        return result

    data = get_engine().benchmark.data()
    if catalog_path is not None:
        catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        data = dict(data)
        data["target_catalog"] = catalog_payload
    if include_discovered_targets:
        data = dict(data)
        if bench_root:
            data["available_targets"] = discover_benchmark_target_catalog(
                Path(bench_root).expanduser()
            ).get("targets", [])
        else:
            data["available_targets"] = get_engine().benchmark.targets().get("targets", [])
    result = rank_opportunities(
        normalize_candidates(data),
        top_n=top_n,
        min_speedup=min_speedup,
        target_speedup=target_speedup,
        min_memory_savings_pct=min_memory_savings_pct,
        slow_baseline_ms=slow_baseline_ms,
    )
    result["source"] = "engine benchmark data"
    if catalog_path is not None:
        result["target_catalog_source"] = str(catalog_path.resolve())
    if include_discovered_targets:
        result["discovered_target_source"] = (
            str(Path(bench_root).expanduser().resolve())
            if bench_root
            else "engine benchmark targets"
        )
    if run_queue_root:
        result = apply_run_queue_feedback(
            result, summarize_run_queue_root(Path(run_queue_root).expanduser())
        )
    if novelty_queue_root:
        result = apply_novelty_validation_feedback(
            result, summarize_run_queue_root(Path(novelty_queue_root).expanduser())
        )
    return result


def benchmark_overview(_: Dict[str, Any]) -> Dict[str, Any]:
    data = get_engine().benchmark.data()
    benchmarks = [_normalize_benchmark_record(b) for b in data.get("benchmarks", [])]
    diagnostics = [_normalize_benchmark_record(b) for b in data.get("diagnostics", [])]
    raw_summary = data.get("summary", {}) or {}
    outcomes = data.get("latest_outcomes", benchmarks + diagnostics)
    summary = _build_summary(outcomes, raw_summary)

    status_counts = {
        "succeeded": summary["succeeded"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
    }

    succeeded = [b for b in benchmarks if _normalize_status(b.get("status")) == "succeeded"]
    top_speedups = sorted(
        succeeded,
        key=lambda b: float(b.get("speedup", 0) or 0),
        reverse=True,
    )[:20]

    chapter_stats: Dict[str, Dict[str, Any]] = {}
    for bench in outcomes:
        chapter = str(bench.get("chapter", "unknown"))
        entry = chapter_stats.setdefault(
            chapter,
            {
                "chapter": chapter,
                "count": 0,
                "succeeded": 0,
                "avg_speedup": 0.0,
                "max_speedup": 0.0,
            },
        )
        entry["count"] += 1
        speedup = float(bench.get("speedup", 0) or 0)
        if _status_bucket(bench.get("status")) == "succeeded":
            entry["succeeded"] += 1
            entry["avg_speedup"] += speedup
            entry["max_speedup"] = max(entry["max_speedup"], speedup)

    chapter_summary = []
    for entry in chapter_stats.values():
        succeeded_count = entry["succeeded"]
        avg_speedup = entry["avg_speedup"] / succeeded_count if succeeded_count else 0.0
        chapter_summary.append(
            {
                "chapter": entry["chapter"],
                "count": entry["count"],
                "succeeded": succeeded_count,
                "avg_speedup": avg_speedup,
                "max_speedup": entry["max_speedup"],
            }
        )

    chapter_summary.sort(key=lambda x: (x["avg_speedup"], x["max_speedup"]), reverse=True)

    return {
        "timestamp": data.get("timestamp"),
        "ingest_errors": data.get("ingest_errors", []),
        "summary": summary,
        "status_counts": status_counts,
        "top_speedups": top_speedups,
        "chapter_stats": chapter_summary,
    }


def benchmark_history(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.history()


def benchmark_trends(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.trends()


def benchmark_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    baseline_path = _parse_path_param(params, "baseline")
    candidate_path = _parse_path_param(params, "candidate")
    top = _parse_int_param(params, "top", 10, minimum=1, maximum=100)
    return compare_benchmark_files(baseline_path, candidate_path, top=top)


def compare_benchmark_files(baseline_path: Path, candidate_path: Path, *, top: int) -> Dict[str, Any]:
    """Compare qualified results and report all rejected rows by input file."""
    baseline = _load_run_file(baseline_path)
    candidate = _load_run_file(candidate_path)

    base_map = {(b["chapter"], b["name"], b["type"]): b for b in baseline["outcomes"]}
    cand_map = {(b["chapter"], b["name"], b["type"]): b for b in candidate["outcomes"]}

    common_keys = sorted(set(base_map.keys()) & set(cand_map.keys()))
    added_keys = sorted(set(cand_map.keys()) - set(base_map.keys()))
    removed_keys = sorted(set(base_map.keys()) - set(cand_map.keys()))

    deltas: List[Dict[str, Any]] = []
    status_transitions: Dict[str, int] = {}
    for key in common_keys:
        base = base_map[key]
        cand = cand_map[key]
        status_changed = base["status"] != cand["status"]
        if status_changed:
            transition = f"{base['status']}->{cand['status']}"
            status_transitions[transition] = status_transitions.get(transition, 0) + 1

        base_speedup = base["speedup"]
        cand_speedup = cand["speedup"]
        if base_speedup is None or cand_speedup is None:
            continue
        delta = cand_speedup - base_speedup
        delta_pct = (delta / base_speedup) * 100.0
        deltas.append(
            {
                "key": base["key"],
                "chapter": base["chapter"],
                "name": base["name"],
                "type": base["type"],
                "baseline_speedup": base_speedup,
                "candidate_speedup": cand_speedup,
                "delta": delta,
                "delta_pct": delta_pct,
                "baseline_status": base["status"],
                "candidate_status": cand["status"],
                "status_changed": status_changed,
                "baseline_time_ms": base.get("baseline_time_ms"),
                "candidate_time_ms": cand.get("baseline_time_ms"),
                "baseline_optimized_time_ms": base.get("optimized_time_ms"),
                "candidate_optimized_time_ms": cand.get("optimized_time_ms"),
            }
        )

    regressions = sorted((d for d in deltas if d["delta"] < 0), key=lambda d: d["delta"])[:top]
    improvements = sorted(
        (d for d in deltas if d["delta"] > 0), key=lambda d: d["delta"], reverse=True
    )[:top]

    added = [cand_map[key] for key in added_keys]
    removed = [base_map[key] for key in removed_keys]

    return {
        "baseline": {
            "path": baseline["path"],
            "timestamp": baseline.get("timestamp"),
            "summary": baseline["summary"],
        },
        "candidate": {
            "path": candidate["path"],
            "timestamp": candidate.get("timestamp"),
            "summary": candidate["summary"],
        },
        "overlap": {
            "common": len(common_keys),
            "comparable": len(deltas),
            "added": len(added),
            "removed": len(removed),
            "baseline_total": len(base_map),
            "candidate_total": len(cand_map),
        },
        "deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "added_benchmarks": added,
        "removed_benchmarks": removed,
        "status_transitions": status_transitions,
        "diagnostics": {"baseline": baseline["diagnostics"], "candidate": candidate["diagnostics"]},
    }


def benchmark_e2e_sweep(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.benchmark.e2e_sweep import (
        attach_benchmark_e2e_status_hints,
        build_benchmark_e2e_progress_surface_hint,
        build_benchmark_e2e_status_actions,
        e2e_progress_path,
        e2e_run_dir,
        resolve_e2e_run_id,
        run_benchmark_e2e_sweep,
    )

    run_async = bool(params.get("async", False))
    dry_run = bool(params.get("dry_run", False))
    run_id = _parse_str_param(params, "run_id")
    bench_root_raw = _parse_str_param(params, "bench_root")
    kwargs: Dict[str, Any] = {
        "run_tier1": bool(params.get("run_tier1", True)),
        "run_full_sweep": bool(params.get("run_full_sweep", False)),
        "run_cluster": bool(params.get("run_cluster", True)),
        "run_fabric": bool(params.get("run_fabric", True)),
        "cluster_preset": str(params.get("cluster_preset", "common-answer-fast")),
        "hosts": _parse_list_param(params, "hosts"),
        "labels": _parse_list_param(params, "labels"),
        "ssh_user": _parse_str_param(params, "ssh_user"),
        "ssh_key": _parse_str_param(params, "ssh_key"),
        "bench_root": Path(bench_root_raw).resolve() if bench_root_raw else None,
        "profile_type": str(params.get("profile", "minimal")),
        "suite_timeout": _parse_optional_int_param(params, "suite_timeout", minimum=0),
        "full_sweep_suite_timeout": _parse_optional_int_param(
            params, "full_sweep_suite_timeout", minimum=0
        ),
        "timeout_seconds": _parse_optional_int_param(params, "timeout_seconds", minimum=1),
        "artifacts_dir": _parse_str_param(params, "artifacts_dir"),
        "validity_profile": str(params.get("validity_profile", "strict")),
        "single_gpu": bool(params.get("single_gpu", False)),
        "iterations": _parse_optional_int_param(params, "iterations", minimum=1),
        "warmup": _parse_optional_int_param(params, "warmup", minimum=0),
        "gpu_sm_clock_mhz": _parse_optional_int_param(params, "gpu_sm_clock_mhz", minimum=1),
        "gpu_mem_clock_mhz": _parse_optional_int_param(params, "gpu_mem_clock_mhz", minimum=1),
        "accept_regressions": bool(params.get("accept_regressions", False)),
        "update_expectations": bool(params.get("update_expectations", False)),
        "allow_mixed_provenance": bool(params.get("allow_mixed_provenance", False)),
        "allow_portable_expectations_update": bool(
            params.get("allow_portable_expectations_update", False)
        ),
        "auto_resume": bool(params.get("auto_resume", True)),
        "max_auto_resumes": _parse_int_param(params, "max_auto_resumes", 3, minimum=0, maximum=100),
        "watch_poll_interval_seconds": _parse_int_param(
            params, "watch_poll_interval_seconds", 15, minimum=1, maximum=3600
        ),
        "run_id": run_id,
        "resume": bool(params.get("resume", False)),
        "dry_run": dry_run,
    }

    if run_async and not dry_run:
        resolved_run_id = resolve_e2e_run_id(run_id)
        kwargs["run_id"] = resolved_run_id
        run_dir = e2e_run_dir(resolved_run_id)
        status_actions = build_benchmark_e2e_status_actions(resolved_run_id)
        ticket = JobStore.get().queue_job(
            "benchmark_e2e_sweep",
            lambda: run_benchmark_e2e_sweep(**kwargs),
            arguments=params,
            run_metadata={
                "run_id": resolved_run_id,
                "run_dir": str(run_dir),
                "progress_path": str(e2e_progress_path(run_dir)),
            },
        )
        ticket["run_id"] = resolved_run_id
        ticket["run_dir"] = str(run_dir)
        ticket["progress_path"] = str(e2e_progress_path(run_dir))
        ticket["actions"] = status_actions
        ticket["preferred_progress_source"] = build_benchmark_e2e_progress_surface_hint(
            resolved_run_id
        )
        return ticket

    result = run_benchmark_e2e_sweep(**kwargs)
    return attach_benchmark_e2e_status_hints(result, _parse_str_param(result, "run_id"))


def benchmark_e2e_status(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.benchmark.e2e_sweep import inspect_benchmark_e2e_sweep_run

    recent_events_limit = _parse_int_param(params, "recent_events", 10, minimum=0, maximum=200)
    bench_root_raw = _parse_str_param(params, "repo_root")
    return inspect_benchmark_e2e_sweep_run(
        run_id=_parse_str_param(params, "run_id"),
        repo_root=Path(bench_root_raw).resolve() if bench_root_raw else None,
        artifacts_dir=_parse_str_param(params, "artifacts_dir"),
        recent_events_limit=recent_events_limit,
    )


def benchmark_e2e_watch(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.benchmark.e2e_sweep import resolve_latest_e2e_run_id, watch_benchmark_e2e_sweep_run

    repo_root_raw = _parse_str_param(params, "repo_root")
    repo_root = Path(repo_root_raw).resolve() if repo_root_raw else None
    run_id = _parse_str_param(params, "run_id") or resolve_latest_e2e_run_id(repo_root=repo_root)
    if not run_id:
        return {"success": False, "error": "No e2e runs found to watch."}
    return watch_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=repo_root,
        poll_interval_seconds=_parse_int_param(
            params, "poll_interval_seconds", 15, minimum=1, maximum=3600
        ),
        max_auto_resumes=_parse_int_param(params, "max_auto_resumes", 3, minimum=0, maximum=100),
    )
