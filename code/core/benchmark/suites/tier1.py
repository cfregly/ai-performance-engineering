"""Tier-1 canonical benchmark suite definition and artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_git_provenance(
    manifest_path: Optional[Path],
) -> tuple[Optional[str], Optional[bool], Optional[str]]:
    manifest_commit: Optional[str] = None
    manifest_dirty: Optional[bool] = None
    if manifest_path is not None:
        try:
            manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest_git = manifest_payload.get("git") if isinstance(manifest_payload, dict) else None
            if isinstance(manifest_git, dict):
                raw_manifest_commit = str(manifest_git.get("commit") or "").strip()
                if re.fullmatch(r"[0-9a-fA-F]{40}", raw_manifest_commit):
                    manifest_commit = raw_manifest_commit.lower()
                if isinstance(manifest_git.get("dirty"), bool):
                    manifest_dirty = manifest_git["dirty"]
        except Exception:
            pass

    environment_commit = str(os.environ.get("GITHUB_SHA") or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", environment_commit):
        source_commit = environment_commit.lower()
        return source_commit, manifest_dirty, manifest_commit
    try:
        from core.benchmark.run_manifest import get_git_info

        git_info = get_git_info()
        local_commit = str(git_info.get("commit") or "").strip()
        local_dirty = git_info.get("dirty")
        if manifest_dirty is None and isinstance(local_dirty, bool):
            manifest_dirty = local_dirty
    except Exception:
        return None, manifest_dirty, manifest_commit
    source_commit = (
        local_commit.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", local_commit) else None
    )
    return source_commit, manifest_dirty, manifest_commit


@dataclass(frozen=True)
class Tier1Target:
    key: str
    target: str
    category: str
    rationale: str
    profile: str = "minimal"


@dataclass(frozen=True)
class Tier1SuiteDefinition:
    name: str
    version: int
    description: str
    history_root: str
    default_profile: str
    default_output_format: str
    targets: List[Tier1Target]

    def target_strings(self) -> List[str]:
        return [target.target for target in self.targets]

    def by_target(self) -> Dict[str, Tier1Target]:
        return {target.target: target for target in self.targets}


def default_tier1_config_path(repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root or _repo_root())
    return root / "configs" / "benchmark_suites" / "tier1.yaml"


def _coerce_target(payload: Dict[str, Any]) -> Tier1Target:
    return Tier1Target(
        key=str(payload["key"]),
        target=str(payload["target"]),
        category=str(payload["category"]),
        rationale=str(payload.get("rationale", "")).strip(),
        profile=str(payload.get("profile", "minimal")).strip() or "minimal",
    )


def load_tier1_suite(config_path: Optional[Path] = None) -> Tier1SuiteDefinition:
    path = Path(config_path or default_tier1_config_path()).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Tier1SuiteDefinition(
        name=str(data.get("suite_name", "tier1")),
        version=int(data.get("version", 1)),
        description=str(data.get("description", "")).strip(),
        history_root=str(data.get("history_root", "artifacts/history/tier1")).strip(),
        default_profile=str(data.get("default_profile", "minimal")).strip() or "minimal",
        default_output_format=str(data.get("default_output_format", "both")).strip() or "both",
        targets=[_coerce_target(entry) for entry in data.get("targets", [])],
    )


def _load_json_object(
    path: Path,
    *,
    label: str,
    required: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = f"Failed to read {label} ({type(exc).__name__})"
        if required:
            raise ValueError(message) from exc
        return None, message

    if not isinstance(payload, dict):
        message = f"Expected JSON object in {label}, got {type(payload).__name__}"
        if required:
            raise ValueError(message)
        return None, message

    return payload, None


def _chapter_key_from_target(target: str) -> str:
    chapter = target.split(":", 1)[0].strip()
    return chapter.replace("/", "_").replace("-", "_")


def _example_from_target(target: str) -> Optional[str]:
    if ":" not in target:
        return None
    return target.split(":", 1)[1].strip() or None


def _find_best_optimization_name(benchmark: Dict[str, Any]) -> Optional[str]:
    best_speedup = float(benchmark.get("best_speedup", 0.0) or 0.0)
    best_name = benchmark.get("best_optimization")
    if best_name:
        return str(best_name)
    optimizations = benchmark.get("optimizations", []) or []
    if not optimizations:
        return None
    best_entry = max(
        optimizations,
        key=lambda entry: float(entry.get("speedup", 0.0) or 0.0),
    )
    if math.isclose(
        float(best_entry.get("speedup", 0.0) or 0.0), best_speedup, rel_tol=1e-6, abs_tol=1e-6
    ):
        return str(best_entry.get("technique") or best_entry.get("file") or "")
    return str(best_entry.get("technique") or best_entry.get("file") or "")


def _portable_relative_path(path: Path) -> Optional[str]:
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = tuple(part for part in path.parts if part not in ("", "."))
    return Path(*parts).as_posix() if parts else None


def _portable_evidence_path(value: Any, *, run_id: str) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if path.is_absolute():
        parts = path.parts
        matching_indexes = [index for index, part in enumerate(parts) if part == run_id]
        if not matching_indexes:
            return None
        return _portable_relative_path(Path(*parts[matching_indexes[-1] :]))

    parts = tuple(part for part in path.parts if part not in ("", "."))
    if ".." in parts:
        return None
    if run_id in parts:
        parts = parts[parts.index(run_id) :]
    else:
        parts = (run_id, *parts)
    return _portable_relative_path(Path(*parts)) if parts else None


def _portable_repo_source_path(value: Any) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.is_absolute():
        return _portable_relative_path(path)
    try:
        return path.resolve().relative_to(_repo_root().resolve()).as_posix()
    except ValueError:
        return None


def _artifact_refs(benchmark: Dict[str, Any], *, run_id: str) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for key, value in benchmark.items():
        if not isinstance(value, str):
            continue
        if key.endswith(("_rep", "_trace", "_json")) and value:
            portable = _portable_evidence_path(value, run_id=run_id)
            if portable:
                refs[key] = portable
    return refs


def _geometric_mean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _optional_finite_metric(value: Any, *, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Tier-1 benchmark returned a non-numeric {label}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Tier-1 benchmark returned a non-finite {label}")
    return parsed


def _summary_is_complete(summary: Dict[str, Any]) -> bool:
    counts = summary.get("summary")
    if not isinstance(counts, dict):
        return False
    try:
        target_count = int(counts.get("target_count", 0) or 0)
        succeeded = int(counts.get("succeeded", 0) or 0)
        failed = int(counts.get("failed", 0) or 0)
        skipped = int(counts.get("skipped", 0) or 0)
        missing = int(counts.get("missing", 0) or 0)
    except (TypeError, ValueError):
        return False
    targets = summary.get("targets")
    return (
        target_count > 0
        and succeeded == target_count
        and failed == 0
        and skipped == 0
        and missing == 0
        and isinstance(targets, list)
        and len(targets) == target_count
        and all(
            isinstance(target, dict) and target.get("status") == "succeeded" for target in targets
        )
    )


def _summary_has_baseline_metrics(summary: Dict[str, Any]) -> bool:
    if not _summary_is_complete(summary):
        return False
    targets = summary.get("targets")
    if not isinstance(targets, list) or not targets:
        return False
    for target in targets:
        if not isinstance(target, dict) or target.get("status") != "succeeded":
            return False
        try:
            baseline_time = float(target.get("baseline_time_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(baseline_time) or baseline_time <= 0.0:
            return False
        goal = str(target.get("optimization_goal") or "performance").strip().lower()
        metric_name = "best_memory_savings_pct" if goal == "memory" else "best_speedup"
        try:
            metric = float(target.get(metric_name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(metric) or (goal != "memory" and metric <= 0.0):
            return False
        if goal == "memory":
            try:
                optimized_memory = float(target.get("best_optimized_memory_mb", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(optimized_memory) or optimized_memory <= 0.0:
                return False
        else:
            try:
                optimized_time = float(target.get("best_optimized_time_ms", 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(optimized_time) or optimized_time <= 0.0:
                return False
    return True


def _tier1_run_accepted(
    summary: Dict[str, Any],
    comparison: Dict[str, Any],
    *,
    history_warnings: Iterable[str],
    accept_comparison: bool,
) -> bool:
    source_git_commit = str(summary.get("source_git_commit") or "")
    source_manifest_git_commit = str(summary.get("source_manifest_git_commit") or "")
    return (
        _summary_has_baseline_metrics(summary)
        and re.fullmatch(r"[0-9a-fA-F]{40}", source_git_commit) is not None
        and source_manifest_git_commit == source_git_commit
        and summary.get("source_git_dirty") is False
        and not list(history_warnings)
        and (
            accept_comparison
            or (not comparison.get("regressions") and not comparison.get("missing_targets"))
        )
    )


def _tier1_baseline_eligible(
    *,
    run_accepted: bool,
    comparison: Dict[str, Any],
    accept_comparison: bool,
    allow_initial_anchor: bool = True,
) -> bool:
    if not run_accepted:
        return False
    if accept_comparison:
        return not comparison.get("suppressed_regressions")
    return (
        allow_initial_anchor
        and comparison.get("baseline_run_id") is None
        and not comparison.get("suppressed_regressions")
        and not comparison.get("anchor_declines")
    )


def _tier1_baseline_acceptance(
    *,
    baseline_eligible: bool,
    accept_history_anchor: bool,
) -> Optional[str]:
    if not baseline_eligible:
        return None
    if accept_history_anchor:
        return "accept_history_anchor"
    return "clean"


def _single_target_suite(suite: Tier1SuiteDefinition, target: Tier1Target) -> Tier1SuiteDefinition:
    return Tier1SuiteDefinition(
        name=suite.name,
        version=suite.version,
        description=suite.description,
        history_root=suite.history_root,
        default_profile=suite.default_profile,
        default_output_format=suite.default_output_format,
        targets=[target],
    )


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "target"


def _confirm_speedup_regressions(
    *,
    comparison: Dict[str, Any],
    current_summary: Dict[str, Any],
    previous_summary: Optional[Dict[str, Any]],
    suite: Tier1SuiteDefinition,
    suite_run_dir: Path,
    bench_root: Optional[Path],
    execution_run_id: str,
    profile_type: str,
    output_format: str,
    suite_timeout: Optional[int],
    timeout_multiplier: float,
    validity_profile: str,
    allow_portable_expectations_update: bool,
    reproducible: bool,
    cold_start: bool,
    force_synchronize: bool,
    iterations: Optional[int],
    warmup: Optional[int],
    gpu_sm_clock_mhz: Optional[int],
    gpu_mem_clock_mhz: Optional[int],
    artifacts_dir: Optional[str],
    log_level: str,
    log_file: Optional[str],
    single_gpu: bool,
    accept_regressions: bool,
    update_expectations: bool,
    allow_mixed_provenance: bool,
    ncu_metric_set: str,
    ncu_replay_mode: Optional[str],
    pm_sampling_interval: Optional[int],
    nsys_timeout_seconds: Optional[int],
    ncu_timeout_seconds: Optional[int],
    launch_via: str,
    nproc_per_node: Optional[int],
    nnodes: Optional[str],
    rdzv_backend: Optional[str],
    rdzv_endpoint: Optional[str],
    torchrun_env: Optional[List[str]],
    target_extra_args: Optional[List[str]],
    verify_input: bool,
    verify_output: bool,
    llm_analysis: bool,
    force_llm: bool,
    llm_provider: Optional[str],
    apply_llm_patches: bool,
    rebenchmark_llm_patches: bool,
    patch_strategy: str,
    llm_patch_retries: int,
    use_llm_cache: bool,
    llm_explain: bool,
) -> Dict[str, Any]:
    from core.analysis.regressions import compare_suite_summaries
    from core.benchmark.bench_commands import _execute_benchmarks

    if previous_summary is None:
        comparison["rechecks"] = []
        comparison["suppressed_regressions"] = []
        return comparison

    regressions = list(comparison.get("regressions", []))
    confirmed: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    rechecks: List[Dict[str, Any]] = []
    target_map = suite.by_target()

    for regression in regressions:
        if regression.get("reason") not in {"optimized_latency", "speedup"}:
            confirmed.append(regression)
            continue
        target_name = str(regression.get("target") or "")
        target = target_map.get(target_name)
        if target is None:
            confirmed.append(regression)
            continue

        recheck_run_id = f"{execution_run_id}__recheck__{_sanitize_component(target.key)}"
        recheck_execution = _execute_benchmarks(
            targets=[target.target],
            bench_root=bench_root,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=suite_timeout,
            timeout_multiplier=timeout_multiplier,
            validity_profile=validity_profile,
            allow_portable_expectations_update=allow_portable_expectations_update,
            reproducible=reproducible,
            cold_start=cold_start,
            force_synchronize=force_synchronize,
            iterations=iterations,
            warmup=warmup,
            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
            artifacts_dir=artifacts_dir,
            run_id=recheck_run_id,
            log_level=log_level,
            log_file=log_file,
            single_gpu=single_gpu,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            allow_mixed_provenance=allow_mixed_provenance,
            ncu_metric_set=ncu_metric_set,
            ncu_replay_mode=ncu_replay_mode,
            pm_sampling_interval=pm_sampling_interval,
            nsys_timeout_seconds=nsys_timeout_seconds,
            ncu_timeout_seconds=ncu_timeout_seconds,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
            verify_input=verify_input,
            verify_output=verify_output,
            llm_analysis=llm_analysis,
            force_llm=force_llm,
            llm_provider=llm_provider,
            apply_llm_patches=apply_llm_patches,
            rebenchmark_llm_patches=rebenchmark_llm_patches,
            patch_strategy=patch_strategy,
            llm_patch_retries=llm_patch_retries,
            use_llm_cache=use_llm_cache,
            llm_explain=llm_explain,
            exit_on_failure=False,
        )

        recheck_summary = build_tier1_suite_summary(
            Path(recheck_execution["output_json"]),
            _single_target_suite(suite, target),
            run_id=recheck_execution["run_id"],
            manifest_path=Path(recheck_execution["manifest_path"])
            if recheck_execution.get("manifest_path")
            else None,
            report_path=Path(recheck_execution["output_markdown"])
            if recheck_execution.get("output_markdown")
            else None,
        )
        recheck_target = recheck_summary["targets"][0]
        recheck_comparison = compare_suite_summaries(recheck_summary, previous_summary)
        recheck_regressions = [
            row
            for row in recheck_comparison.get("regressions", [])
            if row.get("target") == target_name
        ]
        recheck_succeeded = recheck_target.get("status") == "succeeded"
        try:
            recheck_speedup = float(recheck_target.get("best_speedup", 0.0) or 0.0)
            recheck_optimized_time_ms = float(
                recheck_target.get("best_optimized_time_ms", 0.0) or 0.0
            )
            recheck_metrics_valid = (
                math.isfinite(recheck_speedup)
                and recheck_speedup > 0.0
                and math.isfinite(recheck_optimized_time_ms)
                and recheck_optimized_time_ms > 0.0
            )
        except (TypeError, ValueError):
            recheck_metrics_valid = False
        recheck_record = {
            "target": target_name,
            "recheck_run_id": recheck_execution["run_id"],
            "recheck_output_json": _portable_evidence_path(
                recheck_execution["output_json"],
                run_id=recheck_execution["run_id"],
            ),
            "recheck_summary": recheck_target,
            "recheck_regressions": recheck_regressions,
            "recheck_metrics_valid": recheck_metrics_valid,
            "confirmed_regression": (
                not recheck_succeeded or not recheck_metrics_valid or bool(recheck_regressions)
            ),
        }
        rechecks.append(recheck_record)

        if recheck_record["confirmed_regression"]:
            confirmed.append(regression)
        else:
            suppressed.append(
                {
                    **regression,
                    "suppression_reason": "recheck_not_regressed",
                    "recheck_run_id": recheck_execution["run_id"],
                    "recheck_speedup": recheck_target.get("best_speedup"),
                    "recheck_optimized_time_ms": recheck_target.get("best_optimized_time_ms"),
                }
            )

    comparison["regressions"] = confirmed
    comparison["suppressed_regressions"] = suppressed
    comparison["rechecks"] = rechecks

    recheck_path = suite_run_dir / "regression_rechecks.json"
    recheck_path.parent.mkdir(parents=True, exist_ok=True)
    recheck_path.write_text(
        json.dumps(rechecks, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    comparison["regression_rechecks_path"] = str(recheck_path.relative_to(suite_run_dir.parent))
    return comparison


def build_tier1_suite_summary(
    result_json_path: Path,
    suite: Tier1SuiteDefinition,
    *,
    run_id: str,
    manifest_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    evidence_artifact_name: Optional[str] = None,
) -> Dict[str, Any]:
    payload, _ = _load_json_object(
        Path(result_json_path),
        label="tier-1 benchmark result JSON",
        required=True,
    )
    if payload is None:
        raise ValueError(
            f"tier-1 benchmark result JSON reader returned no payload for {result_json_path}"
        )
    target_map = suite.by_target()

    chapter_results = {
        (chapter.get("chapter"), bench.get("example")): bench
        for chapter in payload.get("results", [])
        for bench in chapter.get("benchmarks", []) or []
    }

    targets: List[Dict[str, Any]] = []
    for target in suite.targets:
        chapter_key = _chapter_key_from_target(target.target)
        example = _example_from_target(target.target)
        bench = chapter_results.get((chapter_key, example))
        if bench is None:
            targets.append(
                {
                    "key": target.key,
                    "target": target.target,
                    "category": target.category,
                    "status": "missing",
                    "rationale": target.rationale,
                }
            )
            continue

        baseline_time_ms = _optional_finite_metric(
            bench.get("baseline_time_ms"),
            label=f"baseline_time_ms for {target.target}",
        )
        best_speedup = _optional_finite_metric(
            bench.get("best_speedup"),
            label=f"best_speedup for {target.target}",
        )
        baseline_memory_mb = _optional_finite_metric(
            bench.get("baseline_memory_mb"),
            label=f"baseline_memory_mb for {target.target}",
        )
        best_memory_savings_pct = _optional_finite_metric(
            bench.get("best_memory_savings_pct"),
            label=f"best_memory_savings_pct for {target.target}",
        )
        baseline_p75_ms = _optional_finite_metric(
            bench.get("baseline_p75_ms"),
            label=f"baseline_p75_ms for {target.target}",
        )

        targets.append(
            {
                "key": target.key,
                "target": target.target,
                "category": target.category,
                "rationale": target.rationale,
                "status": bench.get("status", "unknown"),
                "baseline_time_ms": baseline_time_ms,
                "best_speedup": best_speedup,
                "best_optimized_time_ms": (
                    baseline_time_ms / best_speedup
                    if baseline_time_ms is not None
                    and best_speedup is not None
                    and baseline_time_ms > 0.0
                    and best_speedup > 0.0
                    else None
                ),
                "best_optimization": _find_best_optimization_name(bench),
                "optimization_goal": bench.get("optimization_goal"),
                "baseline_memory_mb": baseline_memory_mb,
                "best_memory_savings_pct": best_memory_savings_pct,
                "best_optimized_memory_mb": (
                    baseline_memory_mb * (1.0 - best_memory_savings_pct / 100.0)
                    if baseline_memory_mb is not None
                    and best_memory_savings_pct is not None
                    and baseline_memory_mb > 0.0
                    else None
                ),
                "baseline_p75_ms": baseline_p75_ms,
                "baseline_file": _portable_repo_source_path(bench.get("baseline_file")),
                "artifacts": _artifact_refs(bench, run_id=run_id),
            }
        )

    speedups = [
        float(target.get("best_speedup", 0.0) or 0.0)
        for target in targets
        if target.get("status") == "succeeded"
    ]
    succeeded = sum(1 for target in targets if target.get("status") == "succeeded")
    failed = sum(1 for target in targets if str(target.get("status", "")).startswith("failed"))
    skipped = sum(1 for target in targets if str(target.get("status", "")).startswith("skipped"))
    missing = sum(1 for target in targets if target.get("status") == "missing")
    source_git_commit, source_git_dirty, source_manifest_git_commit = (
        _source_git_provenance(manifest_path)
    )

    return {
        "suite_name": suite.name,
        "suite_version": suite.version,
        "description": suite.description,
        "run_id": run_id,
        "generated_at": payload.get("timestamp"),
        "source_result_json": _portable_evidence_path(result_json_path, run_id=run_id),
        "source_manifest_json": _portable_evidence_path(manifest_path, run_id=run_id),
        "source_markdown_report": _portable_evidence_path(report_path, run_id=run_id),
        "source_git_commit": source_git_commit,
        "source_git_dirty": source_git_dirty,
        "source_manifest_git_commit": source_manifest_git_commit,
        "evidence_artifact_name": evidence_artifact_name,
        "targets": targets,
        "summary": {
            "target_count": len(targets),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "missing": missing,
            "avg_speedup": (sum(speedups) / len(speedups)) if speedups else 0.0,
            "median_speedup": statistics.median(speedups) if speedups else 0.0,
            "geomean_speedup": _geometric_mean(speedups),
            "representative_speedup": _geometric_mean(speedups),
            "max_speedup": max(speedups) if speedups else 0.0,
        },
    }


def run_tier1_suite(
    *,
    config_path: Optional[Path] = None,
    history_root: Optional[Path] = None,
    bench_root: Optional[Path] = None,
    profile_type: Optional[str] = None,
    output_format: Optional[str] = None,
    suite_timeout: Optional[int] = 14400,
    timeout_multiplier: float = 3.0,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    reproducible: bool = False,
    cold_start: bool = False,
    force_synchronize: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    gpu_sm_clock_mhz: Optional[int] = None,
    gpu_mem_clock_mhz: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    single_gpu: bool = False,
    accept_history_anchor: bool = False,
    acceptance_note: Optional[str] = None,
    history_anchor_candidate: bool = False,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    pm_sampling_interval: Optional[int] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    torchrun_env: Optional[List[str]] = None,
    target_extra_args: Optional[List[str]] = None,
    verify_input: bool = True,
    verify_output: bool = True,
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
) -> Dict[str, Any]:
    from core.analysis.history_index import (
        build_updated_history_index,
        load_history_index_with_warnings,
        resolve_history_entry_path,
        update_history_index,
    )
    from core.analysis.regressions import compare_suite_summaries, render_regression_summary
    from core.analysis.trends import build_trend_snapshot
    from core.benchmark.artifact_manager import default_artifacts_root
    from core.benchmark.bench_commands import _execute_benchmarks, _validate_run_id

    suite = load_tier1_suite(config_path)
    if accept_history_anchor:
        raise ValueError(
            "Tier-1 history anchors must be ratified from immutable post-benchmark evidence"
        )
    if acceptance_note:
        raise ValueError("Tier-1 acceptance notes are valid only during post-benchmark ratification")
    history_root_path = Path(history_root or (_repo_root() / suite.history_root)).resolve()
    validated_run_id = _validate_run_id(run_id)
    preflight_index, preflight_warnings = load_history_index_with_warnings(
        history_root_path / "index.json"
    )
    if preflight_warnings:
        raise ValueError(
            "Refusing to run Tier-1 with invalid canonical history: "
            + " | ".join(preflight_warnings)
        )
    if validated_run_id is not None:
        planned_run_dir = history_root_path / validated_run_id
        active_bench_root = Path(bench_root).resolve() if bench_root else _repo_root()
        artifact_base = (
            Path(artifacts_dir) if artifacts_dir else default_artifacts_root(active_bench_root)
        )
        planned_evidence_dir = artifact_base.resolve() / validated_run_id
        if planned_run_dir.is_symlink() or planned_run_dir.exists() or any(
            isinstance(entry, dict) and entry.get("run_id") == validated_run_id
            for entry in preflight_index.get("runs", [])
        ):
            raise ValueError(
                f"Refusing to overwrite existing Tier-1 history run {validated_run_id!r}"
            )
        if planned_evidence_dir.is_symlink() or planned_evidence_dir.exists():
            raise ValueError(
                f"Refusing to overwrite existing Tier-1 evidence run {validated_run_id!r}"
            )
    execution = _execute_benchmarks(
        targets=suite.target_strings(),
        bench_root=bench_root,
        output_format=output_format or suite.default_output_format,
        profile_type=profile_type or suite.default_profile,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        validity_profile=validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        reproducible=reproducible,
        cold_start=cold_start,
        force_synchronize=force_synchronize,
        iterations=iterations,
        warmup=warmup,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
        artifacts_dir=artifacts_dir,
        run_id=validated_run_id,
        log_level=log_level,
        log_file=log_file,
        single_gpu=single_gpu,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        pm_sampling_interval=pm_sampling_interval,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        torchrun_env=torchrun_env,
        target_extra_args=target_extra_args,
        verify_input=verify_input,
        verify_output=verify_output,
        llm_analysis=llm_analysis,
        force_llm=force_llm,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
        exit_on_failure=False,
    )

    execution_run_id = _validate_run_id(str(execution["run_id"]))
    if execution_run_id is None:
        raise ValueError("Tier-1 benchmark execution returned no run id")
    suite_run_dir = history_root_path / execution_run_id
    if suite_run_dir.is_symlink() or suite_run_dir.exists() or any(
        isinstance(entry, dict) and entry.get("run_id") == execution_run_id
        for entry in preflight_index.get("runs", [])
    ):
        raise ValueError(f"Refusing to overwrite existing Tier-1 history run {execution_run_id!r}")
    suite_run_dir.mkdir(parents=True, exist_ok=False)

    summary = build_tier1_suite_summary(
        Path(execution["output_json"]),
        suite,
        run_id=execution["run_id"],
        manifest_path=Path(execution["manifest_path"]) if execution.get("manifest_path") else None,
        report_path=Path(execution["output_markdown"])
        if execution.get("output_markdown")
        else None,
        evidence_artifact_name=os.environ.get("AISP_TIER1_EVIDENCE_ARTIFACT_NAME"),
    )
    summary_path = suite_run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    index_path = history_root_path / "index.json"
    history_warnings: List[str] = []
    previous_index, index_warnings = load_history_index_with_warnings(index_path)
    history_warnings.extend(index_warnings)
    previous_summary = None
    suite_identity_reset = False
    for entry in reversed(previous_index.get("runs", [])):
        entry_run_id = str(entry.get("run_id") or "").strip() or None
        if entry_run_id == execution["run_id"]:
            continue
        eligibility = entry.get("baseline_eligible")
        if eligibility is False:
            continue
        previous_summary_path_raw = str(entry.get("summary_path") or "").strip()
        if not previous_summary_path_raw:
            history_warnings.append(
                f"Tier-1 history entry {entry_run_id or '<unknown>'} has no summary_path"
            )
            continue
        try:
            previous_summary_path = resolve_history_entry_path(
                history_root_path,
                previous_summary_path_raw,
                run_id=entry_run_id,
            )
        except ValueError as exc:
            history_warnings.append(str(exc))
            continue
        if previous_summary_path is None:
            continue
        if not previous_summary_path.exists():
            history_warnings.append(
                f"Indexed previous tier-1 summary for {entry_run_id or '<unknown>'} is missing"
            )
            continue
        candidate_summary, previous_summary_warning = _load_json_object(
            previous_summary_path,
            label="previous tier-1 summary",
        )
        if previous_summary_warning:
            history_warnings.append(previous_summary_warning)
            continue
        if candidate_summary is not None and (
            candidate_summary.get("suite_name") != suite.name
            or candidate_summary.get("suite_version") != suite.version
        ):
            if history_anchor_candidate:
                suite_identity_reset = True
            else:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} has a different suite identity"
                )
            continue
        if candidate_summary is None or not _summary_has_baseline_metrics(candidate_summary):
            if entry.get("baseline_eligible") is True:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} is missing "
                    "complete finite metrics"
                )
            continue

        regression_path_raw = entry.get("regression_json_path") or entry.get(
            "regression_summary_json_path"
        )
        if not regression_path_raw:
            history_warnings.append(
                f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} has no regression JSON"
            )
            continue
        try:
            regression_path = resolve_history_entry_path(
                history_root_path,
                regression_path_raw,
                run_id=entry_run_id,
            )
        except ValueError as exc:
            history_warnings.append(str(exc))
            continue
        if regression_path is None or not regression_path.exists():
            history_warnings.append(
                "Indexed previous tier-1 regression summary for "
                f"{entry_run_id or '<unknown>'} is missing"
            )
            continue
        candidate_comparison, comparison_warning = _load_json_object(
            regression_path,
            label="previous tier-1 regression summary",
        )
        if comparison_warning:
            history_warnings.append(comparison_warning)
            continue
        if candidate_comparison is None or not all(
            isinstance(candidate_comparison.get(key, []), list)
            for key in (
                "anchor_declines",
                "missing_targets",
                "regressions",
                "suppressed_regressions",
            )
        ):
            history_warnings.append(
                f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} has malformed results"
            )
            continue
        if candidate_comparison.get("suppressed_regressions"):
            if eligibility is True or entry.get("run_accepted") is True:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} contains "
                    "suppressed regressions and cannot be an anchor"
                )
            continue
        comparison_has_anchor_changes = any(
            candidate_comparison.get(key)
            for key in (
                "anchor_declines",
                "missing_targets",
                "regressions",
                "suppressed_regressions",
            )
        )
        acceptance = entry.get("baseline_acceptance")
        explicit_acceptance = acceptance in {
            "accept_history_anchor",
            "accept_regressions",
            "update_expectations",
        }
        blocking_comparison_changes = bool(
            candidate_comparison.get("regressions")
            or candidate_comparison.get("missing_targets")
        )
        entry_run_accepted = entry.get("run_accepted")
        if entry_run_accepted is False:
            continue
        if blocking_comparison_changes and not explicit_acceptance:
            if eligibility is True or entry_run_accepted is True:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} has invalid acceptance"
                )
            continue
        inferred_run_accepted = not blocking_comparison_changes or explicit_acceptance
        if entry_run_accepted is None:
            entry_run_accepted = inferred_run_accepted
        if eligibility is None:
            if not (
                entry_run_accepted
                and not comparison_has_anchor_changes
                and candidate_comparison.get("baseline_run_id") is None
            ):
                continue
        elif eligibility is True:
            if not entry_run_accepted:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} was not accepted"
                )
                continue
            if comparison_has_anchor_changes and not explicit_acceptance:
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} lacks acceptance"
                )
                continue
            if (
                candidate_comparison.get("baseline_run_id") is not None
                and not explicit_acceptance
            ):
                history_warnings.append(
                    f"Indexed Tier-1 baseline {entry_run_id or '<unknown>'} was not ratified"
                )
                continue

        previous_summary = candidate_summary
        break

    if previous_index.get("runs") and previous_summary is None and not (
        history_anchor_candidate and suite_identity_reset and not history_warnings
    ):
        history_warnings.append("Tier-1 history contains runs but no eligible prior baseline")

    regression_summary_path = suite_run_dir / "regression_summary.md"
    comparison = compare_suite_summaries(summary, previous_summary)
    if history_warnings:
        comparison.setdefault("warnings", []).extend(history_warnings)
    comparison = _confirm_speedup_regressions(
        comparison=comparison,
        current_summary=summary,
        previous_summary=previous_summary,
        suite=suite,
        suite_run_dir=suite_run_dir,
        bench_root=bench_root,
        execution_run_id=execution["run_id"],
        profile_type=profile_type or suite.default_profile,
        output_format=output_format or suite.default_output_format,
        suite_timeout=suite_timeout,
        timeout_multiplier=timeout_multiplier,
        validity_profile=validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        reproducible=reproducible,
        cold_start=cold_start,
        force_synchronize=force_synchronize,
        iterations=iterations,
        warmup=warmup,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
        artifacts_dir=artifacts_dir,
        log_level=log_level,
        log_file=log_file,
        single_gpu=single_gpu,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        pm_sampling_interval=pm_sampling_interval,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        torchrun_env=torchrun_env,
        target_extra_args=target_extra_args,
        verify_input=verify_input,
        verify_output=verify_output,
        llm_analysis=llm_analysis,
        force_llm=force_llm,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
    )
    regression_summary_path.write_text(
        render_regression_summary(summary, previous_summary, comparison),
        encoding="utf-8",
    )
    regression_json_path = suite_run_dir / "regression_summary.json"
    regression_json_path.write_text(
        json.dumps(comparison, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    accepted_comparison = False
    run_accepted = _tier1_run_accepted(
        summary,
        comparison,
        history_warnings=history_warnings,
        accept_comparison=accepted_comparison,
    )
    baseline_eligible = _tier1_baseline_eligible(
        run_accepted=run_accepted,
        comparison=comparison,
        accept_comparison=accepted_comparison,
        allow_initial_anchor=not history_anchor_candidate,
    )
    baseline_acceptance = _tier1_baseline_acceptance(
        baseline_eligible=baseline_eligible,
        accept_history_anchor=False,
    )

    try:
        updated_index = build_updated_history_index(
            history_root=history_root_path,
            suite=suite,
            summary=summary,
            summary_path=summary_path,
            regression_summary_path=regression_summary_path,
            regression_json_path=regression_json_path,
            run_accepted=run_accepted,
            baseline_eligible=baseline_eligible,
            baseline_acceptance=baseline_acceptance,
            baseline_acceptance_actor=None,
            baseline_acceptance_note=None,
            baseline_acceptance_workflow_run=None,
        )
    except ValueError as exc:
        history_warnings.append(str(exc))
        updated_index = previous_index
    run_warnings = list(history_warnings)
    run_warnings.extend(updated_index.get("warnings", []))

    trend_snapshot = build_trend_snapshot(updated_index)
    trend_snapshot_path = suite_run_dir / "trend_snapshot.json"
    trend_snapshot_path.write_text(
        json.dumps(trend_snapshot, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    try:
        updated_index = update_history_index(
            history_root=history_root_path,
            suite=suite,
            summary=summary,
            summary_path=summary_path,
            regression_summary_path=regression_summary_path,
            regression_json_path=regression_json_path,
            trend_snapshot_path=trend_snapshot_path,
            run_accepted=run_accepted,
            baseline_eligible=baseline_eligible,
            baseline_acceptance=baseline_acceptance,
            baseline_acceptance_actor=None,
            baseline_acceptance_note=None,
            baseline_acceptance_workflow_run=None,
        )
    except ValueError as exc:
        history_warnings.append(str(exc))
        run_warnings.append(str(exc))
    run_warnings.extend(updated_index.get("warnings", []))

    return {
        "suite": suite,
        "execution": execution,
        "summary": summary,
        "summary_path": summary_path,
        "regression_summary_path": regression_summary_path,
        "regression_json_path": regression_json_path,
        "trend_snapshot_path": trend_snapshot_path,
        "index": updated_index,
        "history_root": history_root_path,
        "comparison": comparison,
        "history_integrity_failed": bool(history_warnings),
        "run_accepted": run_accepted,
        "baseline_eligible": baseline_eligible,
        "baseline_acceptance": baseline_acceptance,
        "warnings": list(dict.fromkeys(run_warnings)),
    }
