#!/usr/bin/env python3
"""Manage a full repo-wide portable rerun queue with selective expectation refresh."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.expectations import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
    ExpectationEntry,
    ExpectationsStore,
    RunProvenance,
    detect_expectation_key,
)
from core.benchmark.informational_benchmarks import is_informational_example
from core.discovery import chapter_slug, discover_all_chapters, discover_benchmarks, get_bench_roots
from core.harness.validity_checks import detect_execution_environment

TARGET_RENAMES = {
    "ch03:docker": "ch03:pinned_prefetch_mlp",
    "ch03:kubernetes": "ch03:double_buffered_batch_provisioning",
    "ch06:gemm_ilp": "ch06:elementwise_ilp",
    "ch06:launch_bounds_gmem": "ch06:launch_bounds_cuda",
    "ch07:float8_vector": "ch07:float4_vector",
    "ch04:nvlink": "ch04:pcie_staging",
    "ch14:flex_attention": "ch14:attention_eager_sdpa",
    "ch14:cutlass": "ch14:cublas_vs_cutlass",
    "ch14:model_compile_bf16": "ch14:model_compile_reduced_precision",
    "ch15:disaggregated_inference": "ch15:single_gpu_kv_handoff",
    "ch16:paged_attention": "ch16:dense_attention_flash",
    "ch16:paged_attention_blackwell": "ch16:dense_attention_flash_blackwell_variant",
    "ch18:tiny_gemm_pdl": "ch18:tiny_gemm_fused",
    "ch20:multiple_unoptimized": "ch20:bf16_mlp",
}

EXAMPLE_RENAMES = {
    old.split(":", 1)[1]: new.split(":", 1)[1]
    for old, new in TARGET_RENAMES.items()
}

FILE_RENAMES = {
    "baseline_docker.py": "baseline_pinned_prefetch_mlp.py",
    "optimized_docker.py": "optimized_pinned_prefetch_mlp.py",
    "baseline_kubernetes.py": "baseline_double_buffered_batch_provisioning.py",
    "optimized_kubernetes.py": "optimized_double_buffered_batch_provisioning.py",
    "baseline_gemm_ilp.py": "baseline_elementwise_ilp.py",
    "optimized_gemm_ilp.py": "optimized_elementwise_ilp.py",
    "baseline_launch_bounds_gmem.py": "baseline_launch_bounds_cuda.py",
    "optimized_launch_bounds_gmem.py": "optimized_launch_bounds_cuda.py",
    "baseline_float8_vector.py": "baseline_float4_vector.py",
    "optimized_float8_vector.py": "optimized_float4_vector.py",
    "baseline_nvlink.py": "baseline_pcie_staging.py",
    "optimized_nvlink.py": "optimized_pcie_staging.py",
    "baseline_flex_attention.py": "baseline_attention_eager_sdpa.py",
    "optimized_flex_attention.py": "optimized_attention_eager_sdpa.py",
    "baseline_cutlass.py": "baseline_cublas_vs_cutlass.py",
    "optimized_cutlass.py": "optimized_cublas_vs_cutlass.py",
    "baseline_model_compile_bf16.py": "baseline_model_compile_reduced_precision.py",
    "optimized_model_compile_bf16.py": "optimized_model_compile_reduced_precision.py",
    "baseline_disaggregated_inference.py": "baseline_single_gpu_kv_handoff.py",
    "optimized_disaggregated_inference.py": "optimized_single_gpu_kv_handoff.py",
    "baseline_paged_attention.py": "baseline_dense_attention_flash.py",
    "optimized_paged_attention.py": "optimized_dense_attention_flash.py",
    "optimized_paged_attention_blackwell.py": "optimized_dense_attention_flash_blackwell_variant.py",
    "baseline_tiny_gemm_pdl.py": "baseline_tiny_gemm_fused.py",
    "optimized_tiny_gemm_pdl.py": "optimized_tiny_gemm_fused.py",
    "baseline_multiple_unoptimized.py": "baseline_bf16_mlp.py",
    "optimized_multiple_unoptimized.py": "optimized_bf16_mlp.py",
}

TECHNIQUE_RENAMES = {
    old.replace(".py", ""): new.replace(".py", "")
    for old, new in FILE_RENAMES.items()
}

EXPECTED_UNSUPPORTED_PORTABLE_REASON = "expected_unsupported_portable_single_gpu"
EXPECTED_UNSUPPORTED_RUNTIME_REASON = "expected_unsupported_runtime_capability"
EXPECTED_UNSUPPORTED_PORTABLE_TOKENS = (
    "requires multiple gpus",
    "requires at least 2 gpus",
    "requires >=2 gpus",
    "requires >= 2 gpus",
    "insufficient gpus available",
)
EXPECTED_UNSUPPORTED_RUNTIME_TOKENS = (
    "missing batched_reduce_scatter_hook required for optimized zero-2",
    "requires grace-blackwell coherent memory support",
    "requires torchrun/distributed launch context",
    "requires nvshmem or symmetricmemory support",
    "requires nvshmem or symmetric memory support",
    "requires symmetricmemory support",
    "requires symmetric memory support",
    "requires usable cufile/gds support",
    "requires usable tensor-map/tma support",
    "requires symmetric-memory pipeline support",
    "requires grace cpu support",
    "requires sm100+ blackwell-class hardware",
    "requires cuda 12.0+ and sm90+",
    "algorithm unavailable on this driver/toolchain",
    "requires a native cublaslt heuristic for this exact benchmark",
)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, payloads: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _slugify_target(target: str, *, max_len: int = 96) -> str:
    normalized = []
    for ch in target.lower():
        normalized.append(ch if ch.isalnum() else "_")
    slug = "".join(normalized).strip("_")
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(digest) - 1
    return f"{slug[:keep].rstrip('_')}_{digest}"


def _relative_tolerance(expected: float) -> float:
    return max(abs(expected) * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def _git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    value = (proc.stdout or "").strip()
    return value or "unknown"


def _discover_targets(repo_root: Path) -> List[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "core.benchmark.bench_commands", "list-targets"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _discover_targets_for_chapter(repo_root: Path, chapter: str) -> List[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "core.benchmark.bench_commands", "list-targets", "--chapter", chapter],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _run_command(cmd: List[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def _is_pid_running(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _default_run_root(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "runs" / f"{_utc_now().strftime('%Y%m%d')}_full_virtualized_repo_rerun"


def _default_queue_root(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "parallel_runs" / f"{_utc_now().strftime('%Y%m%d')}_full_virtualized_repo_rerun"


def _queue_paths(queue_root: Path) -> Dict[str, Path]:
    return {
        "state": queue_root / "state.json",
        "lock": queue_root / "state.lock",
        "queue_log": queue_root / "queue.log",
        "worker_log": queue_root / "worker.log",
        "problems": queue_root / "problem_queue.jsonl",
        "unsupported": queue_root / "expected_unsupported.jsonl",
        "writes": queue_root / "expectation_writes.jsonl",
        "targets": queue_root / "targets.txt",
    }


def _log_message(log_path: Path, message: str) -> None:
    line = f"[{_utc_now_iso()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    payload.setdefault("target_records", {})
    payload.setdefault("pending_targets", [])
    payload.setdefault("discovered_targets", [])
    payload.setdefault("stop_requested", False)
    payload.setdefault("active_target", None)
    payload.setdefault("worker_pid", None)
    payload.setdefault("written_expectation_total", 0)
    payload.setdefault("queued_problem_total", 0)
    payload.setdefault("expected_unsupported_total", 0)
    return _canonicalize_state(payload)


def _canonicalize_target(target: Optional[str]) -> Optional[str]:
    if not target:
        return target
    return TARGET_RENAMES.get(str(target), str(target))


def _canonicalize_identity_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(payload)
    target = _canonicalize_target(updated.get("target"))
    if target is not None:
        updated["target"] = target
    example = updated.get("example")
    if example is not None:
        updated["example"] = EXAMPLE_RENAMES.get(str(example), str(example))
    best_file = updated.get("best_optimization_file")
    if best_file is not None:
        updated["best_optimization_file"] = FILE_RENAMES.get(str(best_file), str(best_file))
    best_technique = updated.get("best_optimization_technique")
    if best_technique is not None:
        updated["best_optimization_technique"] = TECHNIQUE_RENAMES.get(str(best_technique), str(best_technique))
    return updated


def _canonicalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(state)
    updated["allow_portable_expectations_update"] = bool(updated.get("allow_portable_expectations_update", False))
    updated["discovered_targets"] = [_canonicalize_target(target) for target in updated.get("discovered_targets", [])]
    updated["pending_targets"] = [_canonicalize_target(target) for target in updated.get("pending_targets", [])]
    updated["active_target"] = _canonicalize_target(updated.get("active_target"))
    target_records = updated.get("target_records") or {}
    canonical_records: Dict[str, Any] = {}
    for target, record in target_records.items():
        canonical_target = _canonicalize_target(target)
        next_record = dict(record)
        next_record["target"] = canonical_target
        normalized_benchmarks = [
            _normalize_target_benchmark_entry(bench)
            for bench in list(next_record.get("benchmarks") or [])
        ]
        next_record["benchmarks"] = normalized_benchmarks
        queued_problem_count = 0
        expected_unsupported_count = 0
        written_expectation_count = 0
        for bench in normalized_benchmarks:
            if _is_expected_unsupported_entry(bench):
                expected_unsupported_count += 1
            elif str(bench.get("action") or "").strip().lower() == "written":
                written_expectation_count += 1
            elif list(bench.get("queue_reasons") or []):
                queued_problem_count += 1
        next_record["queued_problem_count"] = queued_problem_count
        next_record["expected_unsupported_count"] = expected_unsupported_count
        next_record["written_expectation_count"] = written_expectation_count
        canonical_records[str(canonical_target)] = next_record
    updated["target_records"] = canonical_records
    return updated


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = _utc_now_iso()
    _atomic_write_json(state_path, state)


def _normalize_target_benchmark_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = _canonicalize_identity_fields(dict(payload))
    benchmark_status = str(updated.get("benchmark_status") or updated.get("status") or "").strip()
    if benchmark_status:
        updated["benchmark_status"] = benchmark_status
    unsupported_reason, unsupported_detail = _expected_unsupported_reason_for_summary(updated)
    if unsupported_reason:
        updated["classification"] = unsupported_reason
        updated["queue_reasons"] = [unsupported_reason]
        if unsupported_detail:
            updated["error"] = unsupported_detail
    return updated


def _is_expected_unsupported_entry(payload: Dict[str, Any]) -> bool:
    classification = str(payload.get("classification") or "").strip()
    if classification in {EXPECTED_UNSUPPORTED_PORTABLE_REASON, EXPECTED_UNSUPPORTED_RUNTIME_REASON}:
        return True
    queue_reasons = set(str(reason).strip() for reason in (payload.get("queue_reasons") or []))
    return bool(queue_reasons & {EXPECTED_UNSUPPORTED_PORTABLE_REASON, EXPECTED_UNSUPPORTED_RUNTIME_REASON})


def _expected_unsupported_from_optimization_skips(bench: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    optimizations = list(bench.get("optimizations") or [])
    if not optimizations or any(str(opt.get("status") or "").strip().lower() == "succeeded" for opt in optimizations):
        return None, None
    matched_reasons: List[str] = []
    matched_details: List[str] = []
    for opt in optimizations:
        if str(opt.get("status") or "").strip().lower() != "skipped":
            return None, None
        detail = str(opt.get("error") or opt.get("skip_reason") or "").strip()
        reason = _expected_unsupported_portable_reason({"status": "skipped", "error": detail})
        if not reason:
            return None, None
        matched_reasons.append(reason)
        if detail:
            matched_details.append(detail)
    if EXPECTED_UNSUPPORTED_RUNTIME_REASON in matched_reasons:
        return EXPECTED_UNSUPPORTED_RUNTIME_REASON, matched_details[0] if matched_details else None
    if EXPECTED_UNSUPPORTED_PORTABLE_REASON in matched_reasons:
        return EXPECTED_UNSUPPORTED_PORTABLE_REASON, matched_details[0] if matched_details else None
    return None, None


def _load_matching_benchmark_from_results_json(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    results_json = payload.get("results_json")
    example = str(payload.get("example") or "").strip()
    if not results_json or not example:
        return None
    path = Path(str(results_json))
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for chapter_result in data.get("results") or []:
        for bench in chapter_result.get("benchmarks") or []:
            if str(bench.get("example") or "").strip() == example:
                return bench
    return None


def _expected_unsupported_reason_for_summary(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    benchmark_status = str(payload.get("benchmark_status") or payload.get("status") or "").strip()
    detail = str(payload.get("error") or "").strip()
    top_level_reason = _expected_unsupported_portable_reason({"status": benchmark_status, "error": detail})
    if top_level_reason:
        return top_level_reason, detail or None
    queue_reasons = set(str(reason).strip() for reason in (payload.get("queue_reasons") or []))
    if "missing_successful_optimization" not in queue_reasons:
        return None, None
    bench = _load_matching_benchmark_from_results_json(payload)
    if not bench:
        return None, None
    return _expected_unsupported_from_optimization_skips(bench)


def _record_sort_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("finished_at") or record.get("started_at") or ""),
        str(record.get("target") or ""),
        str(record.get("run_id") or ""),
    )


def _rewrite_queue_ledgers_from_state(paths: Dict[str, Path], state: Dict[str, Any]) -> None:
    problems: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    target_records = state.get("target_records") or {}
    records = sorted(target_records.values(), key=_record_sort_key)
    for record in records:
        for bench in list(record.get("benchmarks") or []):
            normalized = _normalize_target_benchmark_entry(bench)
            if _is_expected_unsupported_entry(normalized):
                unsupported.append(normalized)
            elif list(normalized.get("queue_reasons") or []):
                problems.append(normalized)
    _write_jsonl_atomic(paths["problems"], problems)
    _write_jsonl_atomic(paths["unsupported"], unsupported)


_WORKER_LOG_LINE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\] finished target (?P<target>[^:]+:[^:]+): "
    r"rc=(?P<rc>\d+) written_expectations=(?P<written>\d+) queued_problems=(?P<queued>\d+)$"
)


def _backfill_written_expectation_total(worker_log_path: Path, state: Dict[str, Any]) -> None:
    if not worker_log_path.exists():
        return
    total = 0
    try:
        for line in worker_log_path.read_text(encoding="utf-8").splitlines():
            match = _WORKER_LOG_LINE.match(line.strip())
            if not match:
                continue
            total += int(match.group("written"))
    except OSError:
        return
    state["written_expectation_total"] = max(int(state.get("written_expectation_total", 0)), total)


def _persist_state(paths: Dict[str, Path], state: Dict[str, Any]) -> None:
    _finalize_state_counts(state)
    _backfill_written_expectation_total(paths["worker_log"], state)
    _save_state(paths["state"], state)
    _rewrite_queue_ledgers_from_state(paths, state)


def _lock_queue(queue_root: Path):
    queue_root.mkdir(parents=True, exist_ok=True)
    lock_path = _queue_paths(queue_root)["lock"]
    lock_handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    return lock_handle


def _unlock_queue(lock_handle) -> None:
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock_handle.close()


def _initialize_state(
    *,
    run_root: Path,
    queue_root: Path,
    profile: str,
    suite_timeout: int,
    gpu_sm_clock_mhz: int,
    gpu_mem_clock_mhz: int,
    allow_portable_expectations_update: bool,
) -> Dict[str, Any]:
    return {
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "profile": profile,
        "suite_timeout": suite_timeout,
        "targets_total": 0,
        "hardware_key": detect_expectation_key(),
        "tolerance_relative": RELATIVE_TOLERANCE,
        "tolerance_absolute": ABSOLUTE_TOLERANCE,
        "validity_profile": "portable",
        "allow_portable_expectations_update": bool(allow_portable_expectations_update),
        "gpu_sm_clock_mhz": gpu_sm_clock_mhz,
        "gpu_mem_clock_mhz": gpu_mem_clock_mhz,
        "run_root": str(run_root),
        "queue_root": str(queue_root),
        "discovered_targets": [],
        "pending_targets": [],
        "target_records": {},
        "stop_requested": False,
        "active_target": None,
        "active_target_started_at": None,
        "worker_pid": None,
        "worker_started_at": None,
        "worker_command": None,
        "written_expectation_total": 0,
        "queued_problem_total": 0,
        "expected_unsupported_total": 0,
    }


def _finalize_state_counts(state: Dict[str, Any]) -> None:
    records = state.get("target_records", {})
    state["written_expectation_total"] = sum(int(record.get("written_expectation_count", 0)) for record in records.values())
    state["queued_problem_total"] = sum(int(record.get("queued_problem_count", 0)) for record in records.values())
    state["expected_unsupported_total"] = sum(int(record.get("expected_unsupported_count", 0)) for record in records.values())


def _resolve_targets(repo_root: Path, explicit_targets: List[str], max_targets: Optional[int]) -> List[str]:
    targets = list(explicit_targets) if explicit_targets else _discover_targets(repo_root)
    if max_targets is not None:
        targets = targets[:max_targets]
    return targets


def _git_updated_paths(repo_root: Path) -> List[Path]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    paths: List[Path] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        path_text = line[3:] if len(line) > 3 else ""
        if " -> " in path_text:
            _, _, path_text = path_text.partition(" -> ")
        path_text = path_text.strip()
        if not path_text:
            continue
        paths.append((repo_root / path_text).resolve(strict=False))
    return paths


def _find_chapter_dir_for_path(repo_root: Path, candidate: Path) -> Optional[Path]:
    resolved = candidate.resolve(strict=False)
    best_match: Optional[Path] = None
    best_len = -1
    for chapter_dir in discover_all_chapters(repo_root, bench_roots=get_bench_roots(repo_root=repo_root)):
        chapter_resolved = chapter_dir.resolve()
        try:
            resolved.relative_to(chapter_resolved)
        except ValueError:
            continue
        if len(chapter_resolved.parts) > best_len:
            best_len = len(chapter_resolved.parts)
            best_match = chapter_resolved
    return best_match


def _targets_for_changed_path(
    repo_root: Path,
    candidate: Path,
    *,
    chapter_target_cache: Dict[str, List[str]],
    all_targets_cache: Dict[str, List[str]],
) -> List[str]:
    chapter_dir = _find_chapter_dir_for_path(repo_root, candidate)
    if chapter_dir is None:
        for global_root in ("core", "cli", "scripts"):
            try:
                candidate.relative_to(repo_root / global_root)
            except ValueError:
                continue
            if "all" not in all_targets_cache:
                all_targets_cache["all"] = _discover_targets(repo_root)
            return list(all_targets_cache["all"])
        return []

    bench_root = get_bench_roots(repo_root=repo_root)[0]
    chapter_name = chapter_slug(chapter_dir, repo_root, bench_root=bench_root)
    if chapter_name not in chapter_target_cache:
        chapter_target_cache[chapter_name] = _discover_targets_for_chapter(repo_root, chapter_name)
    chapter_targets = chapter_target_cache[chapter_name]
    chapter_target_set = set(chapter_targets)
    resolved = candidate.resolve(strict=False)

    direct_matches: List[str] = []
    for baseline_path, optimized_paths, example_name in discover_benchmarks(chapter_dir, warn_missing=False):
        pair_paths = [baseline_path.resolve()] + [opt.resolve() for opt in optimized_paths]
        if resolved in pair_paths:
            direct_matches.append(f"{chapter_name}:{example_name}")
    if direct_matches:
        return sorted(set(direct_matches))

    stem = resolved.stem
    for prefix in ("baseline_", "optimized_"):
        if not stem.startswith(prefix):
            continue
        example_name = stem[len(prefix) :]
        candidate_target = f"{chapter_name}:{example_name}"
        if candidate_target in chapter_target_set:
            return [candidate_target]
        parts = example_name.split("_")
        while len(parts) > 1:
            parts.pop()
            candidate_target = f"{chapter_name}:{'_'.join(parts)}"
            if candidate_target in chapter_target_set:
                return [candidate_target]

    return list(chapter_targets)


def _resolve_updated_targets(repo_root: Path, updated_paths: List[Path], max_targets: Optional[int]) -> List[str]:
    chapter_target_cache: Dict[str, List[str]] = {}
    all_targets_cache: Dict[str, List[str]] = {}
    resolved_targets: List[str] = []
    seen = set()
    for candidate in updated_paths:
        for target in _targets_for_changed_path(
            repo_root,
            candidate,
            chapter_target_cache=chapter_target_cache,
            all_targets_cache=all_targets_cache,
        ):
            if target in seen:
                continue
            seen.add(target)
            resolved_targets.append(target)
            if max_targets is not None and len(resolved_targets) >= max_targets:
                return resolved_targets
    return resolved_targets


def _enqueue_targets(state: Dict[str, Any], targets: List[str], *, force_rerun: bool) -> int:
    pending_targets = state.setdefault("pending_targets", [])
    target_records = state.setdefault("target_records", {})
    discovered_targets = state.setdefault("discovered_targets", [])
    active_target = state.get("active_target")
    added = 0
    for target in targets:
        if target not in discovered_targets:
            discovered_targets.append(target)
        if not force_rerun:
            if target in pending_targets or target == active_target or target in target_records:
                continue
            pending_targets.append(target)
            added += 1
            continue
        # Forced reruns may queue active or completed targets, but each target
        # still appears at most once in the pending queue.
        if target in pending_targets:
            continue
        pending_targets.append(target)
        added += 1
    state["targets_total"] = len(discovered_targets)
    return added


def _load_manifest_record(manifest_path: Path, target: str, best_file: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not manifest_path.exists():
        return None, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    records = payload.get("manifests") or []
    baseline_record = None
    optimized_record = None
    for record in records:
        if record.get("target_label") != target:
            continue
        if record.get("variant") == "baseline" and baseline_record is None:
            baseline_record = record
        if best_file and record.get("file") == best_file:
            optimized_record = record
    if optimized_record is None:
        for record in records:
            if record.get("target_label") != target:
                continue
            if record.get("variant") != "baseline":
                optimized_record = record
                break
    return baseline_record, optimized_record


def _best_optimization(bench: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    optimizations = [opt for opt in (bench.get("optimizations") or []) if opt.get("status") == "succeeded"]
    if not optimizations:
        return None
    goal = str(bench.get("optimization_goal") or "speed").strip().lower()
    if goal == "memory":
        baseline_memory = bench.get("baseline_memory_mb")

        def _memory_score(opt: Dict[str, Any]) -> float:
            optimized_memory = opt.get("memory_mb")
            if baseline_memory is None or optimized_memory in (None, 0):
                return float("-inf")
            return float(baseline_memory) / float(optimized_memory)

        return max(optimizations, key=_memory_score)
    return max(optimizations, key=lambda opt: float(opt.get("speedup") or 0.0))


def _build_entry(
    *,
    bench: Dict[str, Any],
    best_opt: Dict[str, Any],
    hardware_key: str,
    profile_name: str,
    git_commit: str,
    execution_environment: str,
    dmi_product_name: Optional[str],
    validity_profile: str,
    iterations: int,
    warmup_iterations: int,
) -> ExpectationEntry:
    throughput_baseline = bench.get("baseline_throughput")
    throughput_optimized = best_opt.get("throughput")
    return ExpectationEntry(
        example=str(bench.get("example") or "unknown"),
        type=str(bench.get("type") or "python"),
        optimization_goal=str(bench.get("optimization_goal") or "speed"),
        baseline_time_ms=float(bench.get("baseline_time_ms") or 0.0),
        best_optimized_time_ms=float(best_opt.get("time_ms") or 0.0),
        provenance=RunProvenance(
            git_commit=git_commit,
            hardware_key=hardware_key,
            profile_name=profile_name,
            timestamp=_utc_now_iso(),
            iterations=int(iterations),
            warmup_iterations=int(warmup_iterations),
            execution_environment=execution_environment,
            validity_profile=validity_profile,
            dmi_product_name=dmi_product_name,
        ),
        baseline_memory_mb=bench.get("baseline_memory_mb"),
        best_optimized_memory_mb=best_opt.get("memory_mb"),
        baseline_p75_ms=bench.get("baseline_p75_ms"),
        best_optimized_p75_ms=best_opt.get("p75_ms"),
        baseline_throughput=throughput_baseline if isinstance(throughput_baseline, dict) else None,
        best_optimized_throughput=throughput_optimized if isinstance(throughput_optimized, dict) else None,
        custom_metrics=bench.get("baseline_custom_metrics"),
        best_optimization_name=best_opt.get("technique") or best_opt.get("file"),
        best_optimization_file=best_opt.get("file"),
        best_optimization_technique=best_opt.get("technique"),
    )


def _score(entry: ExpectationEntry) -> float:
    goal = str(entry.optimization_goal or "speed").strip().lower()
    if goal == "memory":
        ratio = entry.best_memory_savings_ratio
        return float(ratio) if ratio is not None else 0.0
    return float(entry.best_speedup)


def _expectation_example_key(bench: Dict[str, Any]) -> str:
    example_name = str(bench.get("example") or "unknown").strip() or "unknown"
    bench_type = str(bench.get("type") or "python").strip().lower() or "python"
    if bench_type == "python":
        return example_name
    suffix = f"_{bench_type}"
    if example_name.lower().endswith(suffix):
        return example_name
    return f"{example_name}{suffix}"


def _write_eligible_expectation(
    *,
    repo_root: Path,
    target: str,
    bench: Dict[str, Any],
    entry: ExpectationEntry,
    hardware_key: str,
    allow_portable_expectations_update: bool,
) -> Tuple[str, List[str], Dict[str, Any]]:
    target_root = target.split(":", 1)[0]
    chapter_dir = repo_root / target_root
    store = ExpectationsStore(
        chapter_dir,
        hardware_key,
        accept_regressions=False,
        allow_mixed_provenance=True,
    )
    example_key = _expectation_example_key(bench)
    existing_entry = store.get_entry(example_key)
    new_score = _score(entry)
    details: Dict[str, Any] = {
        "expectation_path": str(store.path),
        "new_score": new_score,
        "goal": str(entry.optimization_goal or "speed"),
        "allow_portable_expectations_update": bool(allow_portable_expectations_update),
    }
    reasons: List[str] = []
    if not allow_portable_expectations_update:
        reasons.append("portable_expectation_write_not_allowed")
        return "queued", reasons, details
    if existing_entry is None:
        reasons.append("missing_expectation")
    else:
        old_score = _score(existing_entry)
        tol = _relative_tolerance(old_score)
        details["old_score"] = old_score
        details["score_tolerance"] = tol
        details["score_delta"] = new_score - old_score
        details["within_tolerance"] = abs(new_score - old_score) <= tol
        if abs(new_score - old_score) > tol:
            reasons.append("out_of_tolerance")
    if new_score < 1.0:
        reasons.append("non_speedup")
    if reasons:
        return "queued", reasons, details
    update_result = store.update_entry(example_key, entry)
    details["update_result"] = update_result.to_dict()
    if update_result.status == "rejected":
        reasons.append("expectation_write_rejected")
        return "queued", reasons, details
    store.save()
    return "written", [], details


def _expected_unsupported_portable_reason(bench: Dict[str, Any]) -> Optional[str]:
    if str(bench.get("status") or "").strip().lower() != "skipped":
        return None
    error_text = str(bench.get("error") or "").strip().lower()
    if not error_text:
        return None
    if any(token in error_text for token in EXPECTED_UNSUPPORTED_PORTABLE_TOKENS):
        return EXPECTED_UNSUPPORTED_PORTABLE_REASON
    if any(token in error_text for token in EXPECTED_UNSUPPORTED_RUNTIME_TOKENS):
        return EXPECTED_UNSUPPORTED_RUNTIME_REASON
    return None


def _is_informational_benchmark(chapter_name: str, bench: Dict[str, Any]) -> bool:
    example = str(bench.get("example") or "").strip()
    if not example:
        return False
    return is_informational_example(str(chapter_name).strip(), example)


def _iter_benchmarks(results_json: Path) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    for chapter_result in payload.get("results") or []:
        for bench in chapter_result.get("benchmarks") or []:
            yield str(chapter_result.get("chapter") or "unknown"), chapter_result, bench


def _summarize_problem(
    *,
    target: str,
    run_id: str,
    artifact_root: Path,
    results_json: Optional[Path],
    manifest_path: Optional[Path],
    return_code: int,
    bench: Optional[Dict[str, Any]],
    reasons: List[str],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "target": target,
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "results_json": str(results_json) if results_json else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "return_code": return_code,
        "queue_reasons": reasons,
    }
    if bench:
        payload.update(
            {
                "example": bench.get("example"),
                "benchmark_status": bench.get("status"),
                "optimization_goal": bench.get("optimization_goal"),
                "best_speedup": bench.get("best_speedup"),
                "baseline_time_ms": bench.get("baseline_time_ms"),
                "error": bench.get("error"),
            }
        )
    payload.update(extra)
    return _canonicalize_identity_fields(payload)


def _parse_problem_targets(problem_path: Path, reasons: List[str]) -> List[str]:
    if not problem_path.exists():
        return []
    wanted = set(reasons)
    targets: List[str] = []
    seen = set()
    for line in problem_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        queue_reasons = set(payload.get("queue_reasons") or [])
        if wanted and not (queue_reasons & wanted):
            continue
        target = _canonicalize_target(payload.get("target"))
        if not target or target in seen:
            continue
        seen.add(target)
        targets.append(str(target))
    return targets


def _spawn_worker(repo_root: Path, queue_root: Path, run_root: Path, worker_log: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-worker",
        "--queue-root",
        str(queue_root),
        "--run-root",
        str(run_root),
    ]
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    handle = worker_log.open("a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_queue_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--run-root", type=Path, default=None, help="Base directory for per-target run artifacts.")
        subparser.add_argument("--queue-root", type=Path, default=None, help="Directory for queue state/log files.")
        subparser.add_argument(
            "--allow-portable-expectations-update",
            action="store_true",
            help="Allow this portable rerun queue to write expectation files when eligible.",
        )

    def _add_target_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--target", dest="targets", action="append", default=[], help="Specific target to queue (repeatable).")
        subparser.add_argument(
            "--updated-path",
            dest="updated_paths",
            action="append",
            default=[],
            help="Queue targets affected by the given modified file or directory path (repeatable).",
        )
        subparser.add_argument(
            "--updated-from-git",
            action="store_true",
            help="Queue targets inferred from modified and untracked git paths.",
        )
        subparser.add_argument("--max-targets", type=int, default=None, help="Optional cap for discovered targets.")

    start_parser = subparsers.add_parser("start", help="Initialize/enqueue targets and launch the background worker.")
    _add_queue_args(start_parser)
    _add_target_args(start_parser)
    start_parser.add_argument(
        "--profile",
        default=None,
        choices=["none", "minimal", "deep_dive", "roofline"],
        help="Profiling preset for each target run. Existing queues retain their saved preset when omitted.",
    )
    start_parser.add_argument("--suite-timeout", type=int, default=0, help="Suite timeout per target in seconds. Use 0 to disable.")
    start_parser.add_argument("--gpu-sm-clock-mhz", type=int, default=1965, help="SM application clock in MHz.")
    start_parser.add_argument("--gpu-mem-clock-mhz", type=int, default=3996, help="Memory application clock in MHz.")
    start_parser.add_argument("--force-rerun", action="store_true", help="Requeue targets even if they have already completed.")
    start_parser.add_argument("--newly-discovered", action="store_true", help="Queue targets that exist in the repo but are not yet tracked in the current state.")

    worker_parser = subparsers.add_parser("run-worker", help="Internal worker entrypoint.")
    _add_queue_args(worker_parser)

    enqueue_parser = subparsers.add_parser("enqueue", help="Add targets to the pending queue without starting a worker.")
    _add_queue_args(enqueue_parser)
    _add_target_args(enqueue_parser)
    enqueue_parser.add_argument("--force-rerun", action="store_true", help="Requeue targets even if they have already completed.")
    enqueue_parser.add_argument("--newly-discovered", action="store_true", help="Queue targets that exist in the repo but are not yet tracked in the current state.")

    retry_parser = subparsers.add_parser("retry-problems", help="Requeue targets from the problem ledger.")
    _add_queue_args(retry_parser)
    retry_parser.add_argument("--reason", action="append", default=[], help="Queue only problems containing the given reason (repeatable).")

    status_parser = subparsers.add_parser("status", help="Show queue/worker status.")
    _add_queue_args(status_parser)

    stop_parser = subparsers.add_parser("stop", help="Request a graceful stop after the current target finishes.")
    _add_queue_args(stop_parser)

    return parser.parse_args()


def _queue_from_args(repo_root: Path, state: Dict[str, Any], args: argparse.Namespace) -> int:
    explicit_targets = list(getattr(args, "targets", []) or [])
    targets: List[str] = []
    if explicit_targets:
        targets.extend(_resolve_targets(repo_root, explicit_targets, args.max_targets))
    updated_paths = [(repo_root / raw_path).resolve(strict=False) for raw_path in list(getattr(args, "updated_paths", []) or [])]
    if getattr(args, "updated_from_git", False):
        updated_paths.extend(_git_updated_paths(repo_root))
    if updated_paths:
        targets.extend(_resolve_updated_targets(repo_root, updated_paths, args.max_targets))
    if getattr(args, "newly_discovered", False):
        current_targets = _discover_targets(repo_root)
        tracked = set(state.get("discovered_targets", [])) | set(state.get("target_records", {}).keys()) | set(state.get("pending_targets", []))
        if state.get("active_target"):
            tracked.add(str(state["active_target"]))
        targets.extend([target for target in current_targets if target not in tracked])
        if args.max_targets is not None:
            targets = targets[: args.max_targets]
    if not targets and not state.get("discovered_targets") and args.command == "start":
        targets.extend(_resolve_targets(repo_root, [], args.max_targets))
    return _enqueue_targets(state, list(dict.fromkeys(targets)), force_rerun=bool(getattr(args, "force_rerun", False)))


def _run_target(
    *,
    repo_root: Path,
    queue_root: Path,
    run_root: Path,
    target: str,
    state: Dict[str, Any],
    execution_environment: str,
    dmi_product_name: Optional[str],
    git_commit: str,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    started_at = _utc_now_iso()
    slug = _slugify_target(target)
    run_id = f"{_utc_now().strftime('%Y%m%d_%H%M%S')}__portable_repo_rerun__{slug}"
    artifact_root = run_root / run_id
    target_log_path = queue_root / "target_logs" / f"{slug}.log"
    cmd = [
        sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "run",
        "--targets",
        target,
        "--profile",
        str(state.get("profile") or "none"),
        "--format",
        "json",
        "--validity-profile",
        "portable",
        "--suite-timeout",
        str(int(state.get("suite_timeout") or 0)),
        "--artifacts-dir",
        str(run_root),
        "--run-id",
        run_id,
        "--gpu-sm-clock-mhz",
        str(int(state.get("gpu_sm_clock_mhz") or 1965)),
        "--gpu-mem-clock-mhz",
        str(int(state.get("gpu_mem_clock_mhz") or 3996)),
        "--log-level",
        "INFO",
    ]
    _log_message(paths["queue_log"], f"starting target: {target} run_id={run_id}")
    return_code = _run_command(cmd, cwd=repo_root, log_path=target_log_path)
    results_json = artifact_root / "results" / "benchmark_test_results.json"
    manifest_path = artifact_root / "manifest.json"
    target_record: Dict[str, Any] = {
        "target": target,
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "results_json": str(results_json) if results_json.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "log_path": str(target_log_path),
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "benchmarks": [],
        "queued_problem_count": 0,
        "expected_unsupported_count": 0,
        "written_expectation_count": 0,
    }
    hardware_key = str(state.get("hardware_key") or detect_expectation_key())

    if not results_json.exists():
        problem = _summarize_problem(
            target=target,
            run_id=run_id,
            artifact_root=artifact_root,
            results_json=None,
            manifest_path=manifest_path if manifest_path.exists() else None,
            return_code=return_code,
            bench=None,
            reasons=["missing_results_json"],
            extra={"target_log_path": str(target_log_path)},
        )
        _append_jsonl(paths["problems"], problem)
        target_record["benchmarks"].append(problem)
        target_record["queued_problem_count"] = 1
        return target_record

    for chapter_name, _chapter_result, bench in _iter_benchmarks(results_json):
        bench_reasons: List[str] = []
        bench_extra: Dict[str, Any] = {
            "chapter": chapter_name,
            "target_log_path": str(target_log_path),
        }
        if _is_informational_benchmark(chapter_name, bench):
            target_record["benchmarks"].append(
                _canonicalize_identity_fields(
                    {
                        "target": target,
                        "example": bench.get("example"),
                        "benchmark_status": bench.get("status"),
                        "optimization_goal": bench.get("optimization_goal"),
                        "classification": "informational_noncanonical",
                    }
                )
            )
            continue
        unsupported_reason = _expected_unsupported_portable_reason(bench)
        if unsupported_reason:
            unsupported_payload = _summarize_problem(
                target=target,
                run_id=run_id,
                artifact_root=artifact_root,
                results_json=results_json,
                manifest_path=manifest_path if manifest_path.exists() else None,
                return_code=return_code,
                bench=bench,
                reasons=[unsupported_reason],
                extra={"classification": unsupported_reason, **bench_extra},
            )
            _append_jsonl(paths["unsupported"], unsupported_payload)
            target_record["benchmarks"].append(unsupported_payload)
            target_record["expected_unsupported_count"] += 1
            continue
        optimization_unsupported_reason, optimization_unsupported_detail = _expected_unsupported_from_optimization_skips(bench)
        if optimization_unsupported_reason:
            unsupported_payload = _summarize_problem(
                target=target,
                run_id=run_id,
                artifact_root=artifact_root,
                results_json=results_json,
                manifest_path=manifest_path if manifest_path.exists() else None,
                return_code=return_code,
                bench=bench,
                reasons=[optimization_unsupported_reason],
                extra={
                    "classification": optimization_unsupported_reason,
                    "error": optimization_unsupported_detail,
                    **bench_extra,
                },
            )
            _append_jsonl(paths["unsupported"], unsupported_payload)
            target_record["benchmarks"].append(unsupported_payload)
            target_record["expected_unsupported_count"] += 1
            continue

        if bench.get("status") != "succeeded":
            bench_reasons.append(str(bench.get("status") or "failed"))
        best_opt = _best_optimization(bench)
        if best_opt is None:
            bench_reasons.append("missing_successful_optimization")

        if best_opt is not None and not bench_reasons:
            baseline_record, optimized_record = _load_manifest_record(
                manifest_path,
                target,
                str(best_opt.get("file")) if best_opt.get("file") else None,
            )
            baseline_manifest = (baseline_record or {}).get("manifest") or {}
            config = baseline_manifest.get("config") or {}
            git_info = baseline_manifest.get("git") or {}
            git_value = str(git_info.get("commit") or git_commit or "unknown")
            validity_profile = str(config.get("validity_profile") or "portable")
            entry = _build_entry(
                bench=bench,
                best_opt=best_opt,
                hardware_key=hardware_key,
                profile_name=str(state.get("profile") or "none"),
                git_commit=git_value,
                execution_environment=execution_environment,
                dmi_product_name=dmi_product_name,
                validity_profile=validity_profile,
                iterations=int(config.get("iterations") or 0),
                warmup_iterations=int(config.get("warmup") or 0),
            )
            action, reasons, details = _write_eligible_expectation(
                repo_root=repo_root,
                target=target,
                bench=bench,
                entry=entry,
                hardware_key=hardware_key,
                allow_portable_expectations_update=bool(state.get("allow_portable_expectations_update", False)),
            )
            bench_extra.update(details)
            bench_extra["best_optimization_file"] = best_opt.get("file")
            bench_extra["best_optimization_technique"] = best_opt.get("technique")
            bench_extra["best_optimized_time_ms"] = best_opt.get("time_ms")
            if baseline_manifest:
                hardware = baseline_manifest.get("hardware") or {}
                bench_extra["gpu_app_clock_mhz"] = hardware.get("gpu_app_clock_mhz")
                bench_extra["memory_app_clock_mhz"] = hardware.get("memory_app_clock_mhz")
                bench_extra["manifest_validity_profile"] = config.get("validity_profile")
                bench_extra["manifest_iterations"] = config.get("iterations")
                bench_extra["manifest_warmup"] = config.get("warmup")
                bench_extra["manifest_git_commit"] = git_value
            if optimized_record:
                bench_extra["optimized_run_id"] = optimized_record.get("run_id")

            if action == "written":
                write_payload = _summarize_problem(
                    target=target,
                    run_id=run_id,
                    artifact_root=artifact_root,
                    results_json=results_json,
                    manifest_path=manifest_path if manifest_path.exists() else None,
                    return_code=return_code,
                    bench=bench,
                    reasons=[],
                    extra={"action": "written", **bench_extra},
                )
                _append_jsonl(paths["writes"], write_payload)
                target_record["benchmarks"].append(write_payload)
                target_record["written_expectation_count"] += 1
            else:
                bench_reasons.extend(reasons)

        if bench_reasons:
            problem = _summarize_problem(
                target=target,
                run_id=run_id,
                artifact_root=artifact_root,
                results_json=results_json,
                manifest_path=manifest_path if manifest_path.exists() else None,
                return_code=return_code,
                bench=bench,
                reasons=bench_reasons,
                extra=bench_extra,
            )
            _append_jsonl(paths["problems"], problem)
            target_record["benchmarks"].append(problem)
            target_record["queued_problem_count"] += 1

    _log_message(
        paths["queue_log"],
        f"finished target {target}: rc={return_code} "
        f"written_expectations={target_record['written_expectation_count']} "
        f"queued_problems={target_record['queued_problem_count']}",
    )
    return target_record


def _resolve_start_profile(saved_profile: Optional[str], requested_profile: Optional[str]) -> str:
    """Use an explicit profile override, otherwise preserve the queue setting."""

    return requested_profile or saved_profile or "none"


def _command_start(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or _default_run_root(repo_root)).resolve()
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    queue_root.mkdir(parents=True, exist_ok=True)
    paths = _queue_paths(queue_root)

    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            state = _initialize_state(
                run_root=run_root,
                queue_root=queue_root,
                profile=_resolve_start_profile(None, args.profile),
                suite_timeout=args.suite_timeout,
                gpu_sm_clock_mhz=args.gpu_sm_clock_mhz,
                gpu_mem_clock_mhz=args.gpu_mem_clock_mhz,
                allow_portable_expectations_update=args.allow_portable_expectations_update,
            )
        else:
            state["run_root"] = str(run_root)
            state["queue_root"] = str(queue_root)
            state["profile"] = _resolve_start_profile(state.get("profile"), args.profile)
            state["suite_timeout"] = args.suite_timeout
            state["gpu_sm_clock_mhz"] = args.gpu_sm_clock_mhz
            state["gpu_mem_clock_mhz"] = args.gpu_mem_clock_mhz
        state["allow_portable_expectations_update"] = bool(args.allow_portable_expectations_update)

        added = _queue_from_args(repo_root, state, args)
        state["stop_requested"] = False
        _persist_state(paths, state)
        paths["targets"].write_text("".join(f"{target}\n" for target in state.get("discovered_targets", [])), encoding="utf-8")

        worker_pid = int(state.get("worker_pid") or 0)
        pending_count = len(state.get("pending_targets", []))
        if _is_pid_running(worker_pid):
            worker_started_pid = worker_pid
            worker_already_running = True
        else:
            proc = _spawn_worker(repo_root, queue_root, run_root, paths["worker_log"])
            worker_started_pid = int(proc.pid)
            worker_already_running = False
            state["worker_pid"] = worker_started_pid
            state["worker_started_at"] = _utc_now_iso()
            state["worker_command"] = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run-worker",
                "--queue-root",
                str(queue_root),
                "--run-root",
                str(run_root),
            ]
            _persist_state(paths, state)
    finally:
        _unlock_queue(lock_handle)

    if worker_already_running:
        _log_message(paths["queue_log"], f"worker already running pid={worker_started_pid} pending={pending_count}")
        print(f"worker already running: pid={worker_started_pid}")
        return 0

    _log_message(
        paths["queue_log"],
        f"worker started pid={worker_started_pid} added_targets={added} pending={pending_count}",
    )
    print(f"worker_started pid={worker_started_pid} queue_root={queue_root}")
    return 0


def _command_enqueue(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or _default_run_root(repo_root)).resolve()
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    paths = _queue_paths(queue_root)
    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            state = _initialize_state(
                run_root=run_root,
                queue_root=queue_root,
                profile="none",
                suite_timeout=0,
                gpu_sm_clock_mhz=1965,
                gpu_mem_clock_mhz=3996,
                allow_portable_expectations_update=args.allow_portable_expectations_update,
            )
        state["allow_portable_expectations_update"] = bool(args.allow_portable_expectations_update)
        added = _queue_from_args(repo_root, state, args)
        _persist_state(paths, state)
        paths["targets"].write_text("".join(f"{target}\n" for target in state.get("discovered_targets", [])), encoding="utf-8")
        pending_count = len(state.get("pending_targets", []))
    finally:
        _unlock_queue(lock_handle)
    _log_message(paths["queue_log"], f"enqueue added_targets={added} pending={pending_count}")
    print(f"enqueued={added} pending={pending_count}")
    return 0


def _command_retry_problems(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or _default_run_root(repo_root)).resolve()
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    paths = _queue_paths(queue_root)
    targets = _parse_problem_targets(paths["problems"], args.reason)
    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            state = _initialize_state(
                run_root=run_root,
                queue_root=queue_root,
                profile="none",
                suite_timeout=0,
                gpu_sm_clock_mhz=1965,
                gpu_mem_clock_mhz=3996,
                allow_portable_expectations_update=args.allow_portable_expectations_update,
            )
        state["allow_portable_expectations_update"] = bool(args.allow_portable_expectations_update)
        added = _enqueue_targets(state, targets, force_rerun=True)
        _persist_state(paths, state)
        pending_count = len(state.get("pending_targets", []))
    finally:
        _unlock_queue(lock_handle)
    _log_message(paths["queue_log"], f"retry-problems added_targets={added} pending={pending_count}")
    print(f"requeued={added} pending={pending_count}")
    return 0


def _command_status(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    paths = _queue_paths(queue_root)
    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            print("queue_state=missing")
            return 0
        _persist_state(paths, state)
        worker_pid = int(state.get("worker_pid") or 0)
        payload = {
            "queue_root": str(queue_root),
            "worker_pid": worker_pid or None,
            "worker_running": _is_pid_running(worker_pid),
            "stop_requested": bool(state.get("stop_requested")),
            "active_target": state.get("active_target"),
            "pending_targets": len(state.get("pending_targets", [])),
            "completed_targets": len(state.get("target_records", {})),
            "written_expectation_total": state.get("written_expectation_total", 0),
            "queued_problem_total": state.get("queued_problem_total", 0),
            "expected_unsupported_total": state.get("expected_unsupported_total", 0),
            "profile": state.get("profile"),
            "suite_timeout": state.get("suite_timeout"),
            "hardware_key": state.get("hardware_key"),
            "allow_portable_expectations_update": bool(state.get("allow_portable_expectations_update", False)),
            "updated_at": state.get("updated_at"),
        }
    finally:
        _unlock_queue(lock_handle)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _command_stop(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    paths = _queue_paths(queue_root)
    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            print("queue_state=missing")
            return 0
        state["stop_requested"] = True
        _persist_state(paths, state)
    finally:
        _unlock_queue(lock_handle)
    _log_message(paths["queue_log"], "graceful stop requested")
    print("stop_requested=true")
    return 0


def _command_run_worker(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = (args.run_root or _default_run_root(repo_root)).resolve()
    queue_root = (args.queue_root or _default_queue_root(repo_root)).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    paths = _queue_paths(queue_root)
    lock_handle = _lock_queue(queue_root)
    try:
        state = _load_state(paths["state"])
        if not state:
            state = _initialize_state(
                run_root=run_root,
                queue_root=queue_root,
                profile="none",
                suite_timeout=0,
                gpu_sm_clock_mhz=1965,
                gpu_mem_clock_mhz=3996,
                allow_portable_expectations_update=False,
            )
        state["run_root"] = str(run_root)
        state["queue_root"] = str(queue_root)
        state["worker_pid"] = os.getpid()
        state["worker_started_at"] = state.get("worker_started_at") or _utc_now_iso()
        _persist_state(paths, state)
    finally:
        _unlock_queue(lock_handle)
    _log_message(paths["queue_log"], f"worker loop entered pid={os.getpid()}")

    env_info = detect_execution_environment()
    execution_environment = env_info.kind
    dmi_product_name = getattr(env_info, "dmi_product_name", None)
    git_commit = _git_commit(repo_root)

    while True:
        lock_handle = _lock_queue(queue_root)
        try:
            state = _load_state(paths["state"])
            if bool(state.get("stop_requested")):
                state["active_target"] = None
                state["active_target_started_at"] = None
                state["worker_pid"] = None
                _persist_state(paths, state)
                target = None
                exit_reason = "graceful"
            else:
                pending_targets = list(state.get("pending_targets", []))
                if not pending_targets:
                    state["active_target"] = None
                    state["active_target_started_at"] = None
                    state["worker_pid"] = None
                    _persist_state(paths, state)
                    target = None
                    exit_reason = "empty"
                else:
                    target = pending_targets.pop(0)
                    state["pending_targets"] = pending_targets
                    state["active_target"] = target
                    state["active_target_started_at"] = _utc_now_iso()
                    _persist_state(paths, state)
                    exit_reason = None
        finally:
            _unlock_queue(lock_handle)

        if exit_reason == "graceful":
            _log_message(paths["queue_log"], "worker exiting after graceful stop request")
            return 0
        if exit_reason == "empty":
            _log_message(paths["queue_log"], "worker exiting; queue is empty")
            return 0

        target_record = _run_target(
            repo_root=repo_root,
            queue_root=queue_root,
            run_root=run_root,
            target=target,
            state=state,
            execution_environment=execution_environment,
            dmi_product_name=dmi_product_name,
            git_commit=git_commit,
            paths=paths,
        )

        lock_handle = _lock_queue(queue_root)
        try:
            latest_state = _load_state(paths["state"])
            latest_state.setdefault("target_records", {})[target] = target_record
            latest_state["active_target"] = None
            latest_state["active_target_started_at"] = None
            _persist_state(paths, latest_state)
        finally:
            _unlock_queue(lock_handle)


def main() -> int:
    args = _parse_args()
    if args.command == "start":
        return _command_start(args)
    if args.command == "enqueue":
        return _command_enqueue(args)
    if args.command == "retry-problems":
        return _command_retry_problems(args)
    if args.command == "status":
        return _command_status(args)
    if args.command == "stop":
        return _command_stop(args)
    if args.command == "run-worker":
        return _command_run_worker(args)
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
