"""API handlers for the dashboard FastAPI server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.engine import get_engine
from core.jobs import JobStore


def _require_param(params: Dict[str, Any], name: str) -> Any:
    value = params.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} is required")
    return value


def gpu_info(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().gpu.info()


def gpu_topology(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().gpu.topology()


def gpu_power(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().gpu.power()


def gpu_bandwidth(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().gpu.bandwidth()


def system_software(_: Dict[str, Any]) -> Dict[str, Any]:
    result = get_engine().system.software()
    if isinstance(result, dict):
        result = dict(result)
        result.setdefault("python_version", result.get("python"))
        result.setdefault("pytorch_version", result.get("pytorch"))
        result.setdefault("cuda_version", result.get("cuda_runtime"))
        result.setdefault("triton_version", result.get("triton"))
    return result


def system_dependencies(_: Dict[str, Any]) -> Dict[str, Any]:
    result = get_engine().system.dependencies()
    if isinstance(result, dict):
        result = dict(result)
        result.setdefault("missing", result.get("issues", []))
        result.setdefault("outdated", result.get("warnings", []))
    return result


def system_context(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().system.context()


def system_capabilities(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().system.capabilities()


def system_clock_lock_check(params: Dict[str, Any]) -> Dict[str, Any]:
    sm_clock_mhz = _parse_optional_int_param(params, "sm_clock_mhz", minimum=1)
    mem_clock_mhz = _parse_optional_int_param(params, "mem_clock_mhz", minimum=1)
    devices_raw = _parse_list_param(params, "devices")
    devices: Optional[List[int]] = None
    if devices_raw is not None:
        devices = []
        for item in devices_raw:
            try:
                devices.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError("devices must be a comma-separated list of integers") from exc
    return get_engine().system.clock_lock_check(
        sm_clock_mhz=sm_clock_mhz,
        mem_clock_mhz=mem_clock_mhz,
        devices=devices,
    )


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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
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
    raise ValueError(f"{name} must be boolean")


def _parse_path_param(params: Dict[str, Any], name: str) -> Path:
    raw = _require_param(params, name)
    path = Path(str(raw)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")
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
            float(b.get("speedup", 0) or 0) for b in benchmarks if b.get("speedup") is not None
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
        raise ValueError(f"Unsupported sort_field: {sort_field}")

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


def _to_float(value: Any, name: str, *, allow_none: bool) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _load_run_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {path}") from exc

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Run file missing results list: {path}")

    benchmarks: List[Dict[str, Any]] = []
    for chapter_block in results:
        if not isinstance(chapter_block, dict):
            raise ValueError(f"Invalid chapter entry in {path}")
        chapter = chapter_block.get("chapter")
        if not chapter:
            raise ValueError(f"Missing chapter in {path}")
        bench_list = chapter_block.get("benchmarks")
        if not isinstance(bench_list, list):
            raise ValueError(f"Missing benchmarks list for {chapter} in {path}")
        for bench in bench_list:
            if not isinstance(bench, dict):
                raise ValueError(f"Invalid benchmark entry in {path}")
            if "example" in bench:
                name = bench.get("example")
            else:
                name = bench.get("name")
            if not name:
                raise ValueError(f"Benchmark missing name in {path}")
            status_raw = bench.get("status")
            if status_raw is None:
                raise ValueError(f"Benchmark missing status in {path}")
            status = _normalize_status(str(status_raw))
            if status is None:
                raise ValueError(f"Unsupported status '{status_raw}' in {path}")
            baseline_time = _to_float(
                bench.get("baseline_time_ms"), "baseline_time_ms", allow_none=True
            )
            speedup = _to_float(bench.get("best_speedup"), "best_speedup", allow_none=False)
            if status == "succeeded" and baseline_time is None:
                raise ValueError(f"Benchmark {chapter}:{name} missing timing metrics in {path}")
            optimized_time = None
            if baseline_time is not None and speedup is not None and speedup > 0:
                optimized_time = baseline_time / speedup
            record = {
                "key": f"{chapter}:{name}",
                "chapter": chapter,
                "name": name,
                "status": status,
                "baseline_time_ms": baseline_time,
                "optimized_time_ms": optimized_time,
                "speedup": speedup,
                "optimization_goal": bench.get("optimization_goal"),
                "baseline_memory_mb": bench.get("baseline_memory_mb"),
                "optimized_memory_mb": bench.get("optimized_memory_mb"),
                "memory_savings_pct": bench.get("memory_savings_pct"),
                "error": bench.get("error"),
            }
            benchmarks.append(record)

    summary = _build_summary(benchmarks, {})
    return {
        "path": str(path),
        "timestamp": data.get("timestamp"),
        "benchmarks": benchmarks,
        "summary": summary,
    }


def benchmark_data(params: Dict[str, Any]) -> Dict[str, Any]:
    data = get_engine().benchmark.data()
    benchmarks = [_normalize_benchmark_record(b) for b in data.get("benchmarks", [])]
    raw_summary = data.get("summary", {}) or {}

    page = _parse_int_param(params, "page", 1, minimum=1)
    page_size = _parse_int_param(params, "page_size", 50, minimum=1, maximum=500)
    search = _parse_str_param(params, "search")
    sort_field = _parse_str_param(params, "sort_field") or "speedup"
    sort_dir = (_parse_str_param(params, "sort_dir") or "desc").lower()
    if sort_dir not in {"asc", "desc"}:
        raise ValueError("sort_dir must be 'asc' or 'desc'")
    status_filters = _parse_list_param(params, "status")
    chapter_filters = _parse_list_param(params, "chapter")
    benchmark_name = _parse_str_param(params, "benchmark")
    optimization_goal = _parse_str_param(params, "optimization_goal")

    filtered = _filter_benchmarks(
        benchmarks,
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
        "summary": _build_summary(benchmarks, raw_summary),
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
        catalog_path = Path(catalog_file).expanduser()
        if not catalog_path.exists():
            raise FileNotFoundError(f"catalog_file not found: {catalog_path}")
        if not catalog_path.is_file():
            raise ValueError(f"catalog_file must be a file: {catalog_path}")

    if data_file:
        path = Path(data_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"data_file not found: {path}")
        if not path.is_file():
            raise ValueError(f"data_file must be a file: {path}")
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


def benchmark_contracts(_: Dict[str, Any]) -> Dict[str, Any]:
    from core.benchmark.contracts_surface import get_benchmark_contracts_summary

    return get_benchmark_contracts_summary()


def benchmark_contracts_render_run(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.benchmark.contracts_surface import render_benchmark_run_yaml

    return render_benchmark_run_yaml(params)


def benchmark_overview(_: Dict[str, Any]) -> Dict[str, Any]:
    data = get_engine().benchmark.data()
    benchmarks = [_normalize_benchmark_record(b) for b in data.get("benchmarks", [])]
    raw_summary = data.get("summary", {}) or {}
    summary = _build_summary(benchmarks, raw_summary)

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
    for bench in benchmarks:
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
        "summary": summary,
        "status_counts": status_counts,
        "top_speedups": top_speedups,
        "chapter_stats": chapter_summary,
    }


def benchmark_history(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.history()


def benchmark_trends(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.trends()


def benchmark_tier1_history(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.tier1_history()


def benchmark_tier1_trends(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.tier1_trends()


def benchmark_tier1_target_history(params: Dict[str, Any]) -> Dict[str, Any]:
    key = str(params.get("key") or "").strip() or None
    target = str(params.get("target") or "").strip() or None
    return get_engine().benchmark.tier1_target_history(key=key, target=target)


def benchmark_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    baseline_path = _parse_path_param(params, "baseline")
    candidate_path = _parse_path_param(params, "candidate")
    top = _parse_int_param(params, "top", 10, minimum=1, maximum=100)

    baseline = _load_run_file(baseline_path)
    candidate = _load_run_file(candidate_path)

    base_map = {b["key"]: b for b in baseline["benchmarks"]}
    cand_map = {b["key"]: b for b in candidate["benchmarks"]}

    common_keys = sorted(set(base_map.keys()) & set(cand_map.keys()))
    added_keys = sorted(set(cand_map.keys()) - set(base_map.keys()))
    removed_keys = sorted(set(base_map.keys()) - set(cand_map.keys()))

    deltas: List[Dict[str, Any]] = []
    status_transitions: Dict[str, int] = {}
    for key in common_keys:
        base = base_map[key]
        cand = cand_map[key]
        base_speedup = float(base.get("speedup") or 0)
        cand_speedup = float(cand.get("speedup") or 0)
        delta = cand_speedup - base_speedup
        delta_pct = None
        if base_speedup > 0:
            delta_pct = (delta / base_speedup) * 100.0
        status_changed = base["status"] != cand["status"]
        if status_changed:
            transition = f"{base['status']}->{cand['status']}"
            status_transitions[transition] = status_transitions.get(transition, 0) + 1
        deltas.append(
            {
                "key": key,
                "chapter": base["chapter"],
                "name": base["name"],
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
    }


def benchmark_targets(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().benchmark.targets()


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


def profile_flame(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.flame_graph()


def profile_kernels(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.kernels()


def profile_memory(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.memory_timeline()


def profile_hta(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.hta()


def profile_compile(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.compile_analysis()


def profile_roofline(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.roofline()


def profile_list(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.list_profiles()


def profile_ncu_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    report_path = str(_require_param(params, "report_path"))
    top_k = _parse_int_param(params, "top_k", 10, minimum=1, maximum=200)
    timeout_seconds = _parse_int_param(params, "timeout_seconds", 60, minimum=1, maximum=3600)
    metrics = _parse_list_param(params, "metrics")
    return get_engine().profile.ncu_summary(
        report_path=report_path,
        top_k=top_k,
        metrics=metrics,
        timeout_seconds=timeout_seconds,
    )


def profile_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    chapter = str(_require_param(params, "chapter"))
    return get_engine().profile.compare(chapter)


def profile_recommendations(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().profile.recommendations()


def ai_status(_: Dict[str, Any]) -> Dict[str, Any]:
    return get_engine().ai.status()


def ai_ask(params: Dict[str, Any]) -> Dict[str, Any]:
    question = str(_require_param(params, "question"))
    return get_engine().ai.ask(question)


def ai_explain(params: Dict[str, Any]) -> Dict[str, Any]:
    concept = str(_require_param(params, "concept"))
    return get_engine().ai.explain(concept)


def ai_tools(_: Dict[str, Any]) -> Dict[str, Any]:
    from core.api.registry import get_dashboard_mcp_tools
    from mcp.mcp_server import TOOLS

    allowed_tools = set(get_dashboard_mcp_tools())
    categories = {
        "gpu": [],
        "system": [],
        "analysis": [],
        "optimization": [],
        "distributed": [],
        "cluster": [],
        "inference": [],
        "ai": [],
        "profiling": [],
        "benchmarks": [],
        "exports": [],
        "other": [],
    }
    category_map = {
        "gpu": ["gpu_info", "gpu_bandwidth", "gpu_topology", "gpu_power"],
        "system": [
            "system_software",
            "system_dependencies",
            "system_context",
            "system_capabilities",
            "clock_lock_check",
            "system_parameters",
            "system_container",
            "system_cpu_memory",
            "system_env",
            "system_network",
            "system_full",
            "context_summary",
            "context_full",
            "status",
            "triage",
        ],
        "analysis": [
            "analyze_bottlenecks",
            "analyze_pareto",
            "analyze_scaling",
            "analyze_stacking",
            "analyze_whatif",
            "analyze_comm_overlap",
            "analyze_memory_patterns",
            "analyze_dataloader",
            "analyze_energy",
        ],
        "optimization": ["optimize", "recommend", "optimize_roi", "optimize_techniques"],
        "distributed": ["distributed_plan", "distributed_nccl", "cluster_slurm", "launch_plan"],
        "cluster": [
            "cluster_eval_suite",
            "cluster_common_eval",
            "cluster_build_canonical_package",
            "cluster_validate_field_report",
        ],
        "inference": [
            "inference_vllm",
            "inference_quantization",
            "inference_deploy",
            "inference_estimate",
        ],
        "ai": ["ask", "explain", "ai_status", "ai_troubleshoot", "suggest_tools"],
        "profiling": [
            "profile_flame",
            "profile_memory",
            "profile_kernels",
            "profile_roofline",
            "profile_nsys",
            "profile_ncu",
            "profile_torch",
            "profile_hta",
            "nsys_summary",
            "compare_nsys",
            "compare_ncu",
            "profile_list_profiles",
            "profile_compile_analysis",
            "ncu_summary",
        ],
        "benchmarks": [
            "benchmark_contracts",
            "run_benchmarks",
            "benchmark_e2e_sweep",
            "benchmark_targets",
            "list_chapters",
            "benchmark_report",
            "benchmark_export",
            "benchmark_compare_runs",
            "benchmark_triage",
            "benchmark_variants",
            "benchmark_deep_dive_compare",
            "benchmark_explore",
            "benchmark_llm_patch_loop",
            "benchmark_data",
            "benchmark_overview",
            "benchmark_opportunities",
            "benchmark_history",
            "benchmark_trends",
            "benchmark_compare",
        ],
        "exports": ["export_csv", "export_pdf", "export_html"],
    }
    tool_to_category: Dict[str, str] = {}
    for category, tools in category_map.items():
        for tool in tools:
            tool_to_category[tool] = category

    tools_list: List[Dict[str, Any]] = []
    for name, tool_def in TOOLS.items():
        if name not in allowed_tools:
            continue
        category = tool_to_category.get(name, "other")
        tools_list.append(
            {
                "name": name,
                "description": tool_def.description,
                "category": category,
                "schema": tool_def.input_schema,
            }
        )
        categories[category].append(name)

    return {
        "tools": tools_list,
        "categories": {key: value for key, value in categories.items() if value},
        "count": len(tools_list),
        "available": True,
    }


def ai_execute(params: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = str(_require_param(params, "tool"))
    tool_params = params.get("params") or {}
    if not isinstance(tool_params, dict):
        raise ValueError("params must be an object")

    from core.api.registry import get_dashboard_mcp_tools
    from mcp.mcp_server import HANDLERS

    if tool_name not in set(get_dashboard_mcp_tools()):
        raise ValueError(f"Tool '{tool_name}' is not exposed by the dashboard API")
    if tool_name not in HANDLERS:
        raise ValueError(f"Tool '{tool_name}' not found")
    result = HANDLERS[tool_name](tool_params)
    success = True
    if isinstance(result, dict):
        if result.get("success") is False or result.get("error"):
            success = False
    return {"success": success, "tool": tool_name, "result": result}


def cluster_eval_suite(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_eval_suite

    mode = str(params.get("mode", "smoke"))
    run_id = _parse_str_param(params, "run_id")
    hosts = _parse_list_param(params, "hosts")
    labels = _parse_list_param(params, "labels")
    ssh_user = _parse_str_param(params, "ssh_user")
    ssh_key = _parse_str_param(params, "ssh_key")
    oob_if = _parse_str_param(params, "oob_if")
    socket_ifname = _parse_str_param(params, "socket_ifname")
    nccl_ib_hca = _parse_str_param(params, "nccl_ib_hca")
    nmx_url = _parse_str_param(params, "nmx_url")
    primary_label = _parse_str_param(params, "primary_label")
    extra_args = _parse_list_param(params, "extra_args")
    timeout_seconds = _parse_optional_int_param(params, "timeout_seconds", minimum=1)

    return run_cluster_eval_suite(
        mode=mode,
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        primary_label=primary_label,
        extra_args=extra_args,
        timeout_seconds=timeout_seconds,
    )


def cluster_validate_field_report(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import validate_field_report_requirements

    report = _parse_str_param(params, "report")
    notes = _parse_str_param(params, "notes")
    template = _parse_str_param(params, "template")
    runbook = _parse_str_param(params, "runbook")
    canonical_run_id = _parse_str_param(params, "canonical_run_id")
    allow_run_id = _parse_list_param(params, "allow_run_id")
    timeout_seconds = _parse_int_param(params, "timeout_seconds", 120, minimum=1, maximum=3600)

    return validate_field_report_requirements(
        report=report,
        notes=notes,
        template=template,
        runbook=runbook,
        canonical_run_id=canonical_run_id,
        allow_run_id=allow_run_id,
        timeout_seconds=timeout_seconds,
    )


def cluster_build_canonical_package(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import build_canonical_package

    canonical_run_id = _parse_str_param(params, "canonical_run_id")
    if not canonical_run_id:
        raise ValueError("canonical_run_id is required")
    comparison_run_ids = _parse_list_param(params, "comparison_run_ids")
    historical_run_ids = _parse_list_param(params, "historical_run_ids")
    output_dir = _parse_str_param(params, "output_dir")
    timeout_seconds = _parse_int_param(params, "timeout_seconds", 300, minimum=1, maximum=3600)

    return build_canonical_package(
        canonical_run_id=canonical_run_id,
        comparison_run_ids=comparison_run_ids,
        historical_run_ids=historical_run_ids,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )


def cluster_promote_run(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import promote_cluster_run

    run_id = _parse_str_param(params, "run_id")
    if not run_id:
        raise ValueError("run_id is required")
    label = _parse_str_param(params, "label") or "localhost"
    allow_run_ids = _parse_list_param(params, "allow_run_ids")
    publish_report_path = _parse_str_param(params, "publish_report_path")
    publish_notes_path = _parse_str_param(params, "publish_notes_path")
    skip_render_localhost_report = bool(params.get("skip_render_localhost_report", False))
    skip_validate_localhost_report = bool(params.get("skip_validate_localhost_report", False))
    cleanup = bool(params.get("cleanup", False))
    timeout_seconds = _parse_int_param(params, "timeout_seconds", 300, minimum=1, maximum=3600)
    repo_root = _parse_str_param(params, "repo_root")

    return promote_cluster_run(
        run_id=run_id,
        label=label,
        allow_run_ids=allow_run_ids,
        publish_report_path=publish_report_path,
        publish_notes_path=publish_notes_path,
        skip_render_localhost_report=skip_render_localhost_report,
        skip_validate_localhost_report=skip_validate_localhost_report,
        cleanup=cleanup,
        timeout_seconds=timeout_seconds,
        repo_root=repo_root,
    )


def cluster_watch_promote(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import watch_cluster_run_for_promotion

    run_id = _parse_str_param(params, "run_id")
    if not run_id:
        raise ValueError("run_id is required")
    pid = int(params.get("pid") or 0)
    label = _parse_str_param(params, "label") or "localhost"
    allow_run_ids = _parse_list_param(params, "allow_run_ids")
    publish_report_path = _parse_str_param(params, "publish_report_path")
    publish_notes_path = _parse_str_param(params, "publish_notes_path")
    repo_root = _parse_str_param(params, "repo_root")
    skip_render_localhost_report = bool(params.get("skip_render_localhost_report", False))
    skip_validate_localhost_report = bool(params.get("skip_validate_localhost_report", False))
    cleanup = bool(params.get("cleanup", False))
    poll_interval_seconds = float(params.get("poll_interval_seconds", 30.0) or 30.0)

    return watch_cluster_run_for_promotion(
        run_id=run_id,
        pid=pid,
        label=label,
        allow_run_ids=allow_run_ids,
        publish_report_path=publish_report_path,
        publish_notes_path=publish_notes_path,
        skip_render_localhost_report=skip_render_localhost_report,
        skip_validate_localhost_report=skip_validate_localhost_report,
        cleanup=cleanup,
        poll_interval_seconds=poll_interval_seconds,
        repo_root=repo_root,
    )


def cluster_common_eval(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_common_eval

    preset = str(params.get("preset", "core-system"))
    run_id = _parse_str_param(params, "run_id")
    hosts = _parse_list_param(params, "hosts")
    labels = _parse_list_param(params, "labels")
    ssh_user = _parse_str_param(params, "ssh_user")
    ssh_key = _parse_str_param(params, "ssh_key")
    oob_if = _parse_str_param(params, "oob_if")
    socket_ifname = _parse_str_param(params, "socket_ifname")
    nccl_ib_hca = _parse_str_param(params, "nccl_ib_hca")
    nmx_url = _parse_str_param(params, "nmx_url")
    nmx_token = _parse_str_param(params, "nmx_token")
    ib_mgmt_host = _parse_str_param(params, "ib_mgmt_host")
    ib_mgmt_user = _parse_str_param(params, "ib_mgmt_user")
    ib_mgmt_ssh_key = _parse_str_param(params, "ib_mgmt_ssh_key")
    cumulus_hosts = _parse_list_param(params, "cumulus_hosts")
    cumulus_user = _parse_str_param(params, "cumulus_user")
    cumulus_ssh_key = _parse_str_param(params, "cumulus_ssh_key")
    primary_label = _parse_str_param(params, "primary_label")
    coverage_baseline_run_id = _parse_str_param(params, "coverage_baseline_run_id")
    extra_args = _parse_list_param(params, "extra_args")
    timeout_seconds = _parse_optional_int_param(params, "timeout_seconds", minimum=1)

    return run_cluster_common_eval(
        preset=preset,
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        nmx_url=nmx_url,
        nmx_token=nmx_token,
        ib_mgmt_host=ib_mgmt_host,
        ib_mgmt_user=ib_mgmt_user,
        ib_mgmt_ssh_key=ib_mgmt_ssh_key,
        cumulus_hosts=cumulus_hosts,
        cumulus_user=cumulus_user,
        cumulus_ssh_key=cumulus_ssh_key,
        primary_label=primary_label,
        coverage_baseline_run_id=coverage_baseline_run_id,
        extra_args=extra_args,
        timeout_seconds=timeout_seconds,
    )


def cluster_fabric_eval(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_fabric_eval

    run_id = _parse_str_param(params, "run_id")
    hosts = _parse_list_param(params, "hosts")
    labels = _parse_list_param(params, "labels")
    ssh_user = _parse_str_param(params, "ssh_user")
    ssh_key = _parse_str_param(params, "ssh_key")
    oob_if = _parse_str_param(params, "oob_if")
    socket_ifname = _parse_str_param(params, "socket_ifname")
    nccl_ib_hca = _parse_str_param(params, "nccl_ib_hca")
    nmx_url = _parse_str_param(params, "nmx_url")
    nmx_token = _parse_str_param(params, "nmx_token")
    ib_mgmt_host = _parse_str_param(params, "ib_mgmt_host")
    ib_mgmt_user = _parse_str_param(params, "ib_mgmt_user")
    ib_mgmt_ssh_key = _parse_str_param(params, "ib_mgmt_ssh_key")
    cumulus_hosts = _parse_list_param(params, "cumulus_hosts")
    cumulus_user = _parse_str_param(params, "cumulus_user")
    cumulus_ssh_key = _parse_str_param(params, "cumulus_ssh_key")
    primary_label = _parse_str_param(params, "primary_label")
    coverage_baseline_run_id = _parse_str_param(params, "coverage_baseline_run_id")
    extra_args = _parse_list_param(params, "extra_args")
    timeout_seconds = _parse_optional_int_param(params, "timeout_seconds", minimum=1)
    require_management_plane = bool(params.get("require_management_plane", False))

    return run_cluster_fabric_eval(
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        nmx_url=nmx_url,
        nmx_token=nmx_token,
        ib_mgmt_host=ib_mgmt_host,
        ib_mgmt_user=ib_mgmt_user,
        ib_mgmt_ssh_key=ib_mgmt_ssh_key,
        cumulus_hosts=cumulus_hosts,
        cumulus_user=cumulus_user,
        cumulus_ssh_key=cumulus_ssh_key,
        primary_label=primary_label,
        coverage_baseline_run_id=coverage_baseline_run_id,
        extra_args=extra_args,
        timeout_seconds=timeout_seconds,
        require_management_plane=require_management_plane,
    )


def cluster_nmx_partition_lab(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import build_cluster_nmx_partition_lab

    nmx_url = _parse_str_param(params, "nmx_url")
    nmx_token = _parse_str_param(params, "nmx_token")
    alpha_name = _parse_str_param(params, "alpha_name") or "AlphaPartition"
    beta_name = _parse_str_param(params, "beta_name") or "BetaPartition"
    alpha_size = int(params.get("alpha_size", 4) or 4)
    beta_size = int(params.get("beta_size", 4) or 4)

    return build_cluster_nmx_partition_lab(
        nmx_url=nmx_url,
        nmx_token=nmx_token,
        alpha_name=alpha_name,
        beta_name=beta_name,
        alpha_size=alpha_size,
        beta_size=beta_size,
    )


def job_status(params: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(_require_param(params, "job_id"))
    record = JobStore.get().get_status(job_id)
    if not record:
        return {
            "job_id": job_id,
            "status": "not_found",
            "success": False,
            "note": "No job with this id; ensure async=true was used on the original call.",
        }
    if "success" not in record:
        record["success"] = record.get("status") not in {"error", "not_found"}
    return record
