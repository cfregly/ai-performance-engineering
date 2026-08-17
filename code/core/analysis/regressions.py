"""Regression summaries for canonical benchmark suites."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _target_map(summary: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not summary:
        return {}
    return {
        str(target.get("target")): target
        for target in summary.get("targets", [])
        if target.get("target")
    }


def _normalized_optimization_goal(*targets: Dict[str, Any]) -> str:
    for target in targets:
        raw_goal = str(target.get("optimization_goal") or "").strip().lower()
        if not raw_goal:
            continue
        if raw_goal == "performance":
            return "speed"
        return raw_goal
    return "speed"


def _optimized_memory_mb(target: Dict[str, Any]) -> float:
    optimized = float(target.get("best_optimized_memory_mb", 0.0) or 0.0)
    if optimized > 0.0:
        return optimized
    baseline = float(target.get("baseline_memory_mb", 0.0) or 0.0)
    savings = float(target.get("best_memory_savings_pct", 0.0) or 0.0)
    if baseline <= 0.0:
        return 0.0
    return baseline * (1.0 - savings / 100.0)


def compare_suite_summaries(
    current: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    *,
    speedup_regression_threshold_pct: float = 5.0,
    memory_regression_threshold_points: float = 5.0,
    optimized_memory_regression_threshold_pct: float = 5.0,
    min_optimized_time_delta_ms: float = 0.05,
) -> Dict[str, Any]:
    if baseline is None:
        return {
            "baseline_run_id": None,
            "current_run_id": current.get("run_id"),
            "regressions": [],
            "improvements": [],
            "anchor_declines": [],
            "new_targets": list(current.get("targets", [])),
            "missing_targets": [],
        }

    current_map = _target_map(current)
    baseline_map = _target_map(baseline)
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    anchor_declines: List[Dict[str, Any]] = []

    for target_name, current_target in current_map.items():
        previous = baseline_map.get(target_name)
        if previous is None:
            continue
        optimization_goal = _normalized_optimization_goal(current_target, previous)

        current_status = str(current_target.get("status", "unknown"))
        previous_status = str(previous.get("status", "unknown"))
        if previous_status == "succeeded" and current_status != "succeeded":
            regressions.append(
                {
                    "target": target_name,
                    "reason": "status",
                    "before": previous_status,
                    "after": current_status,
                }
            )
            continue
        if previous_status != "succeeded" and current_status == "succeeded":
            improvements.append(
                {
                    "target": target_name,
                    "reason": "status",
                    "before": previous_status,
                    "after": current_status,
                }
            )

        current_speedup = float(current_target.get("best_speedup", 0.0) or 0.0)
        previous_speedup = float(previous.get("best_speedup", 0.0) or 0.0)
        speedup_regressed = False
        speedup_improved = False
        if optimization_goal != "memory" and current_speedup > 0 and previous_speedup > 0:
            delta_pct = ((current_speedup - previous_speedup) / previous_speedup) * 100.0
            current_optimized_ms = float(current_target.get("best_optimized_time_ms", 0.0) or 0.0)
            previous_optimized_ms = float(previous.get("best_optimized_time_ms", 0.0) or 0.0)
            optimized_time_delta_ms = current_optimized_ms - previous_optimized_ms
            payload = {
                "target": target_name,
                "reason": "speedup",
                "before": previous_speedup,
                "after": current_speedup,
                "delta_pct": delta_pct,
                "optimized_time_before_ms": previous_optimized_ms or None,
                "optimized_time_after_ms": current_optimized_ms or None,
                "optimized_time_delta_ms": optimized_time_delta_ms
                if previous_optimized_ms > 0 and current_optimized_ms > 0
                else None,
            }
            significant_optimized_time_change = (
                previous_optimized_ms <= 0.0
                or current_optimized_ms <= 0.0
                or abs(optimized_time_delta_ms) >= min_optimized_time_delta_ms
            )
            if current_speedup < previous_speedup:
                anchor_declines.append(payload)
            if delta_pct <= -speedup_regression_threshold_pct and significant_optimized_time_change:
                regressions.append(payload)
                speedup_regressed = True
            elif (
                delta_pct >= speedup_regression_threshold_pct and significant_optimized_time_change
            ):
                improvements.append(payload)
                speedup_improved = True

        if optimization_goal != "memory":
            current_optimized_ms = float(
                current_target.get("best_optimized_time_ms", 0.0) or 0.0
            )
            previous_optimized_ms = float(previous.get("best_optimized_time_ms", 0.0) or 0.0)
            if current_optimized_ms > 0.0 and previous_optimized_ms > 0.0:
                optimized_time_delta_ms = current_optimized_ms - previous_optimized_ms
                optimized_time_delta_pct = (
                    optimized_time_delta_ms / previous_optimized_ms
                ) * 100.0
                payload = {
                    "target": target_name,
                    "reason": "optimized_latency",
                    "before": previous_optimized_ms,
                    "after": current_optimized_ms,
                    "delta_pct": optimized_time_delta_pct,
                    "optimized_time_delta_ms": optimized_time_delta_ms,
                }
                significant_optimized_time_change = (
                    abs(optimized_time_delta_ms) >= min_optimized_time_delta_ms
                )
                if current_optimized_ms > previous_optimized_ms:
                    anchor_declines.append(payload)
                if (
                    optimized_time_delta_pct >= speedup_regression_threshold_pct
                    and significant_optimized_time_change
                    and not speedup_regressed
                ):
                    regressions.append(payload)
                elif (
                    optimized_time_delta_pct <= -speedup_regression_threshold_pct
                    and significant_optimized_time_change
                    and not speedup_improved
                ):
                    improvements.append(payload)

        if optimization_goal == "memory":
            current_memory = float(current_target.get("best_memory_savings_pct", 0.0) or 0.0)
            previous_memory = float(previous.get("best_memory_savings_pct", 0.0) or 0.0)
            memory_delta = current_memory - previous_memory
            memory_regressed = False
            memory_improved = False
            payload = {
                "target": target_name,
                "reason": "memory_savings",
                "before": previous_memory,
                "after": current_memory,
                "delta_points": memory_delta,
            }
            if current_memory < previous_memory:
                anchor_declines.append(payload)
            if memory_delta <= -memory_regression_threshold_points:
                regressions.append(payload)
                memory_regressed = True
            elif memory_delta >= memory_regression_threshold_points:
                improvements.append(payload)
                memory_improved = True

            current_optimized_memory = _optimized_memory_mb(current_target)
            previous_optimized_memory = _optimized_memory_mb(previous)
            if current_optimized_memory > 0.0 and previous_optimized_memory > 0.0:
                optimized_memory_delta = current_optimized_memory - previous_optimized_memory
                optimized_memory_delta_pct = (
                    optimized_memory_delta / previous_optimized_memory
                ) * 100.0
                payload = {
                    "target": target_name,
                    "reason": "optimized_memory",
                    "before": previous_optimized_memory,
                    "after": current_optimized_memory,
                    "delta_pct": optimized_memory_delta_pct,
                    "optimized_memory_delta_mb": optimized_memory_delta,
                }
                if current_optimized_memory > previous_optimized_memory:
                    anchor_declines.append(payload)
                if (
                    optimized_memory_delta_pct >= optimized_memory_regression_threshold_pct
                    and not memory_regressed
                ):
                    regressions.append(payload)
                elif (
                    optimized_memory_delta_pct <= -optimized_memory_regression_threshold_pct
                    and not memory_improved
                ):
                    improvements.append(payload)

    return {
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.get("run_id"),
        "regressions": regressions,
        "improvements": improvements,
        "anchor_declines": anchor_declines,
        "new_targets": [
            current_map[name] for name in sorted(current_map.keys() - baseline_map.keys())
        ],
        "missing_targets": [
            baseline_map[name] for name in sorted(baseline_map.keys() - current_map.keys())
        ],
    }


def render_regression_summary(
    current: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]] = None,
) -> str:
    comparison = comparison or compare_suite_summaries(current, baseline)
    lines = [
        "# Tier-1 Regression Summary",
        "",
        f"- Current run: `{current.get('run_id')}`",
        f"- Baseline run: `{comparison.get('baseline_run_id') or 'none'}`",
        "",
        "## Summary",
        "",
        f"- Regressions: {len(comparison.get('regressions', []))}",
        f"- Improvements: {len(comparison.get('improvements', []))}",
        f"- Suppressed regressions after recheck: {len(comparison.get('suppressed_regressions', []))}",
        f"- Baseline advancement holds: {len(comparison.get('anchor_declines', []))}",
        f"- New targets: {len(comparison.get('new_targets', []))}",
        f"- Missing targets: {len(comparison.get('missing_targets', []))}",
        "",
    ]

    if comparison.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in comparison["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    def _render_rows(title: str, rows: List[Dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Target | Reason | Before | After | Delta |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            delta = row.get("delta_pct")
            if delta is None:
                delta = row.get("delta_points")
            delta_text = "" if delta is None else f"{float(delta):+.2f}"
            before = row.get("before", "")
            after = row.get("after", "")
            if isinstance(before, float):
                before = f"{before:.3f}"
            if isinstance(after, float):
                after = f"{after:.3f}"
            lines.append(
                f"| `{row.get('target')}` | {row.get('reason')} | {before} | {after} | {delta_text} |"
            )
        lines.append("")

    if comparison.get("regressions"):
        _render_rows("Regressions", comparison["regressions"])
    if comparison.get("improvements"):
        _render_rows("Improvements", comparison["improvements"])

    if comparison.get("suppressed_regressions"):
        lines.extend(
            [
                "## Suppressed Regressions After Recheck",
                "",
                "| Target | Initial delta (%) | Recheck speedup | Recheck optimized ms | Recheck run |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in comparison["suppressed_regressions"]:
            delta_text = f"{float(row.get('delta_pct', 0.0)):+.2f}"
            recheck_speedup = row.get("recheck_speedup")
            recheck_speedup_text = (
                "" if recheck_speedup is None else f"{float(recheck_speedup):.3f}"
            )
            recheck_time = row.get("recheck_optimized_time_ms")
            recheck_time_text = "" if recheck_time is None else f"{float(recheck_time):.3f}"
            recheck_run = row.get("recheck_run_id", "")
            lines.append(
                f"| `{row.get('target')}` | {delta_text} | {recheck_speedup_text} | {recheck_time_text} | `{recheck_run}` |"
            )
        lines.append("")

    if comparison.get("new_targets"):
        lines.extend(["## New Targets", ""])
        for target in comparison["new_targets"]:
            lines.append(f"- `{target.get('target')}` ({target.get('category')})")
        lines.append("")

    if comparison.get("missing_targets"):
        lines.extend(["## Missing Targets", ""])
        for target in comparison["missing_targets"]:
            lines.append(f"- `{target.get('target')}` ({target.get('category')})")
        lines.append("")

    if baseline is None:
        lines.extend(
            [
                "## Notes",
                "",
                "No previous eligible canonical tier-1 summary was available. This run becomes "
                "an anchor only if its complete outcome passes baseline eligibility.",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
