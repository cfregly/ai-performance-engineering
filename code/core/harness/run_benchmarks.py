#!/usr/bin/env python3
"""Run every single benchmark/example and summarize results.

This script:
1. Discovers all baseline/optimized pairs across all chapters
2. Runs actual benchmarks using BenchmarkHarness
3. Collects performance metrics (speedup, latency, throughput, etc.)
4. Generates a comprehensive summary report

Usage:
    python -m core.harness.run_benchmarks [--targets chX chY:example] [--format json|markdown|both]
"""

import sys
import os
import atexit
from pathlib import Path
import json
import shutil
import argparse
import shlex
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Iterator, Sequence
from datetime import datetime
from collections import defaultdict
import statistics
import math
import difflib
from dataclasses import dataclass, fields, replace
import threading
from contextlib import ExitStack, contextmanager
import copy
 
# Force NVCC line info so Nsight/torch traces carry file/line metadata
os.environ["NVCCFLAGS"] = f"-lineinfo {os.environ.get('NVCCFLAGS', '')}".strip()

repo_root = Path(__file__).resolve().parents[2]

from core.harness.arch_config import configure_optimizations as _configure_arch_optimizations
from core.benchmark.artifact_manager import default_artifacts_root, build_run_id, slugify

_configure_arch_optimizations()

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

from core.env import apply_env_defaults, dump_environment_and_capabilities

apply_env_defaults()

import torch
import subprocess
import time
import tempfile
import signal
from core.harness.hardware_capabilities import detect_capabilities
from core.utils.chapter_compare_template import (
    discover_benchmarks,
    load_benchmark,
    get_last_load_error,
)
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkMode,
    BenchmarkConfig,
    TorchrunLaunchSpec,
    _lookup_target_extra_args,
)
from core.harness.validity_profile import (
    VALIDITY_PROFILE_CHOICES,
    VALIDITY_PROFILE_HELP_TEXT,
    PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
    normalize_validity_profile,
)
from core.harness.validity_checks import detect_execution_environment, _collect_process_lineage_pids
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness.triton_cache_utils import reset_triton_runtime_cache
from core.benchmark.defaults import BenchmarkDefaults, set_defaults, get_defaults
from core.benchmark.informational_benchmarks import INFORMATIONAL_BENCHMARKS, is_informational_example
from core.benchmark.run_manifest import get_gpu_state
from core.benchmark.run_manifest import reset_gpu_state, get_git_info
from core.profiling.gpu_telemetry import format_gpu_telemetry, query_gpu_telemetry
from core.harness.serving_stack import get_serving_stack_pins
from core.profiling.profiler_config import (
    build_profiler_config_from_benchmark,
    resolve_ncu_metrics,
)
from core.profiling.profiler_wrapper import (
    render_ncu_python_profile_wrapper,
    render_nsys_python_profile_wrapper,
    render_torch_python_profile_wrapper,
    temporary_python_profile_wrapper,
)
try:
    from core.benchmark.cuda_binary_benchmark import detect_supported_arch
except ImportError:  # pragma: no cover - optional dependency during docs builds
    detect_supported_arch = None  # type: ignore[assignment]
from core.benchmark.timing_parser import parse_kernel_time_ms
from core.analysis.llm_patch_promotion import promote_best_llm_patch
from core.benchmark.expectations import (
    ExpectationsStore,
    ExpectationEntry,
    RunProvenance,
    METRIC_DIRECTIONS,
    detect_expectation_key,
    select_best_optimization,
    compute_speedup,
)
from core.discovery import chapter_slug, resolve_target_chapters, is_cuda_binary_benchmark_file
from core.utils.python_entrypoints import build_python_entry_command, build_repo_python_env

# Import verification system for mandatory correctness checks
try:
    from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
    from core.benchmark.verification import (
        EnforcementPhase,
        get_enforcement_phase,
        QuarantineReason,
        VerifyResult,
        coerce_input_signature,
        get_signature_equivalence_spec,
        signature_workload_dict,
    )
    from core.benchmark.quarantine import QuarantineManager
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    VerifyRunner = None  # type: ignore
    VerifyConfig = None  # type: ignore
    QuarantineManager = None  # type: ignore

# Import logger
try:
    from core.utils.logger import get_logger, setup_logging
    logger = get_logger(__name__)
    LOGGER_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGGER_AVAILABLE = False
    def setup_logging(*args, **kwargs):
        pass

# Check if torch.profiler is available at module level
TORCH_PROFILER_AVAILABLE = hasattr(torch, 'profiler') and hasattr(torch.profiler, 'profile')

# Generous timeout so deep NCU profiling can finish (pulled from benchmark defaults)
NCU_TIMEOUT_SECONDS = get_defaults().ncu_timeout_seconds
_NCU_DRIVER_RESOURCE_RETRY_ATTEMPTS = 2
_NCU_DRIVER_RESOURCE_RETRY_DELAY_SECONDS = 5.0
_NCU_DRIVER_RESOURCE_UNAVAILABLE_MARKERS = (
    "driver resource was unavailable",
    "ensure that no other tool",
)


def _run_repo_python_module(
    module_name: str,
    *argv: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a repo Python module with repo-root package resolution."""
    env = build_repo_python_env(repo_root, base_env=os.environ.copy())
    cmd = build_python_entry_command(module_name=module_name, argv=list(argv))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
        cwd=str(repo_root),
        env=env,
    )


def _read_profile_log_text(path: Path) -> str:
    """Read a profiler stdout/stderr log if it exists."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def _is_retryable_ncu_driver_resource_error(
    *,
    stdout_log: Path,
    stderr_log: Path,
) -> bool:
    """Return True when Nsight Compute failed due to a transient resource conflict."""
    combined = "\n".join(
        part
        for part in (
            _read_profile_log_text(stdout_log),
            _read_profile_log_text(stderr_log),
        )
        if part
    ).lower()
    return all(marker in combined for marker in _NCU_DRIVER_RESOURCE_UNAVAILABLE_MARKERS)


PROGRESS_PHASES = {
    "discovery": 1,
    "baseline_timing": 2,
    "baseline_nsys": 3,
    "baseline_ncu": 4,
    "baseline_torch": 5,
    "optimized_timing": 6,
    "optimized_nsys": 7,
    "optimized_ncu": 8,
    "optimized_torch": 9,
    "verification": 10,
    "expectations": 11,
    "llm_analysis": 12,
    "llm_patch_apply": 13,
    "llm_patch_rebenchmark": 14,
    "llm_patch_verify": 15,
    "llm_explain": 16,
    "complete": 17,
}


def _query_gpu_telemetry_for_profile(
    validity_profile: str,
    *,
    device_index: Optional[int] = None,
    force_refresh: bool = False,
) -> Optional[Dict[str, Optional[float | str]]]:
    """Query GPU telemetry with strict/portable validity semantics."""
    try:
        return query_gpu_telemetry(device_index=device_index, force_refresh=force_refresh)
    except Exception:
        if str(validity_profile).strip().lower() == "portable":
            logger.warning(
                "Portable validity profile: GPU telemetry field unavailable on this hardware; continuing.",
                exc_info=True,
            )
            return None
        raise


def _set_profiler_status(statuses: Dict[str, str], profiler: str, status: str) -> None:
    """Track profiler outcomes for later validity enforcement."""
    statuses[profiler] = status


_GENERIC_BENCHMARK_FAILURE_ERROR = "Baseline or optimization failed"
_PROFILE_FAILURE_DETAILS: Dict[str, Optional[str]] = {
    "nsys": None,
    "ncu": None,
    "torch": None,
}


def _normalize_failure_detail(detail: Optional[str], *, max_chars: int = 600) -> Optional[str]:
    """Normalize failure detail text for stable structured output."""
    if detail is None:
        return None
    cleaned = " ".join(str(detail).strip().split())
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3] + "..."
    return cleaned


def _set_profile_failure_detail(profiler: str, detail: Optional[str]) -> None:
    """Persist the most recent profiler failure detail for the active capture."""
    _PROFILE_FAILURE_DETAILS[profiler] = _normalize_failure_detail(detail)


def _get_profile_failure_detail(profiler: str) -> Optional[str]:
    """Return the most recent profiler failure detail for the given profiler."""
    return _PROFILE_FAILURE_DETAILS.get(profiler)


def _record_profiler_error(
    errors: Dict[str, str],
    profiler: str,
    detail: Optional[str],
) -> Optional[str]:
    """Persist a structured profiler error when detail is available."""
    normalized = _normalize_failure_detail(detail)
    if normalized:
        errors[profiler] = normalized
    else:
        errors.pop(profiler, None)
    return normalized


def _read_profile_failure_detail(*paths: Path) -> Optional[str]:
    """Read and condense profiler stdout/stderr log detail."""
    parts: List[str] = []
    for path in paths:
        text = _normalize_failure_detail(_read_profile_log_text(path))
        if text and text not in parts:
            parts.append(text)
    if not parts:
        return None
    return " | ".join(parts)


def _collect_required_profiler_failure_details(
    result_entry: Dict[str, Any],
    best_opt: Optional[Dict[str, Any]],
    *,
    profiling_requested: bool,
) -> Dict[str, str]:
    """Return structured detail for failed/skipped required profilers."""
    if not profiling_requested:
        return {}

    details: Dict[str, str] = {}
    baseline_statuses = result_entry.get("baseline_profiler_statuses") or {}
    baseline_errors = result_entry.get("baseline_profiler_errors") or {}
    for profiler, status in sorted(baseline_statuses.items()):
        if status == "succeeded":
            continue
        detail = _normalize_failure_detail(baseline_errors.get(profiler))
        if detail:
            details[f"baseline:{profiler}"] = detail

    if best_opt and isinstance(best_opt, dict):
        optimized_statuses = best_opt.get("optimized_profiler_statuses") or {}
        optimized_errors = best_opt.get("optimized_profiler_errors") or {}
        for profiler, status in sorted(optimized_statuses.items()):
            if status == "succeeded":
                continue
            detail = _normalize_failure_detail(optimized_errors.get(profiler))
            if detail:
                details[f"optimized:{profiler}"] = detail

    return details


def _collect_required_profiler_failures(
    result_entry: Dict[str, Any],
    best_opt: Optional[Dict[str, Any]],
    *,
    profiling_requested: bool,
) -> List[str]:
    """Return required profiler failures for a benchmark pair.

    Profiling-enabled runs are only trustworthy when the requested profilers
    actually succeed. Treat skipped or failed profiler outcomes as invalid.
    """
    if not profiling_requested:
        return []

    failures: List[str] = []
    baseline_statuses = result_entry.get("baseline_profiler_statuses") or {}
    for profiler, status in sorted(baseline_statuses.items()):
        if status != "succeeded":
            failures.append(f"baseline:{profiler}:{status}")

    if best_opt and isinstance(best_opt, dict):
        optimized_statuses = best_opt.get("optimized_profiler_statuses") or {}
        for profiler, status in sorted(optimized_statuses.items()):
            if status != "succeeded":
                failures.append(f"optimized:{profiler}:{status}")

    return failures


def _format_required_profiler_failure(
    failures: List[str],
    failure_details: Optional[Dict[str, str]] = None,
) -> str:
    detail = ", ".join(failures)
    if not failure_details:
        return f"Required profilers did not succeed: {detail}"

    detail_parts: List[str] = []
    for failure in failures:
        failure_key = ":".join(str(failure).split(":", 2)[:2])
        failure_detail = _normalize_failure_detail(failure_details.get(failure_key))
        if failure_detail:
            detail_parts.append(f"{failure_key}: {failure_detail}")
    if not detail_parts:
        return f"Required profilers did not succeed: {detail}"
    return f"Required profilers did not succeed: {detail}. Details: {'; '.join(detail_parts)}"


def _classify_failure_message(message: Optional[str]) -> Optional[str]:
    """Classify a benchmark failure without changing the public status schema."""
    normalized = str(message or "").strip().lower()
    if not normalized:
        return None
    if "environment invalid:" in normalized:
        return "environment_invalid"
    if "setup pre-computation detected:" in normalized:
        return "setup_precomputation"
    if "required profilers did not succeed:" in normalized:
        return "profiler_failed"
    if "no report artifact was produced" in normalized:
        return "profiler_missing_artifact"
    if "no optimizations passed verification" in normalized:
        return "verification_failed"
    if "below required" in normalized and "threshold for speed-goal benchmark" in normalized:
        return "no_speedup"
    return "runtime_error"


def _attach_failure_metadata(result_entry: Dict[str, Any]) -> None:
    """Promote precise child failure detail into the parent benchmark result."""
    for field in (
        "optimized_profiler_statuses",
        "optimized_profiler_errors",
        "optimized_profiler_metrics",
    ):
        if result_entry.get(field):
            continue
        for opt in result_entry.get("optimizations") or []:
            value = opt.get(field)
            if not value:
                continue
            result_entry[field] = dict(value) if isinstance(value, dict) else value
            break

    status = str(result_entry.get("status") or "")
    if not status.startswith("failed"):
        return

    failure_details: List[Dict[str, Any]] = []
    top_error = _normalize_failure_detail(result_entry.get("error"))
    if top_error and top_error != _GENERIC_BENCHMARK_FAILURE_ERROR:
        failure_details.append(
            {
                "scope": "benchmark",
                "status": status,
                "error": top_error,
            }
        )

    child_failures: List[tuple[str, str]] = []
    for opt in result_entry.get("optimizations") or []:
        opt_status = str(opt.get("status") or "")
        if not opt_status.startswith("failed"):
            continue
        child_error = _normalize_failure_detail(
            opt.get("error") or opt.get("failure_reason") or opt.get("reason")
        )
        if not child_error:
            continue
        technique = str(opt.get("technique") or opt.get("file") or "optimized")
        child_failures.append((technique, child_error))
        failure_details.append(
            {
                "scope": "optimization",
                "technique": technique,
                "status": opt_status,
                "error": child_error,
            }
        )

    if (not top_error or top_error == _GENERIC_BENCHMARK_FAILURE_ERROR) and child_failures:
        if len(child_failures) == 1:
            result_entry["error"] = child_failures[0][1]
        else:
            rendered = [f"{technique}: {error}" for technique, error in child_failures]
            result_entry["error"] = "; ".join(rendered[:3])
            if len(rendered) > 3:
                result_entry["error"] += f"; +{len(rendered) - 3} more"

    final_error = _normalize_failure_detail(result_entry.get("error"))
    if final_error:
        result_entry["failure_class"] = _classify_failure_message(final_error)
    if failure_details:
        result_entry["failure_details"] = failure_details


def _append_benchmark_result(
    benchmark_results: List[Dict[str, Any]],
    result_entry: Dict[str, Any],
) -> None:
    """Attach normalized failure metadata before persisting a benchmark row."""
    _attach_failure_metadata(result_entry)
    benchmark_results.append(result_entry)



PROGRESS_TOTAL_PHASES = max(PROGRESS_PHASES.values())
_SERVING_STACK_PINS = get_serving_stack_pins()


def _is_cuda_wrapper(path: Path) -> bool:
    """Best-effort check for Python benchmarks that wrap CUDA binaries."""
    try:
        return is_cuda_binary_benchmark_file(path)
    except Exception:
        return False


def _discover_chapter_benchmark_pairs(
    chapter_dir: Path,
    *,
    only_examples: Optional[List[str]] = None,
    only_cuda: bool = False,
    only_python: bool = False,
) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]], Optional[Set[str]], int, int]:
    """Return chapter benchmark pairs using the same filtering as execution."""

    python_pairs = discover_benchmarks(chapter_dir)
    example_filters = None
    if only_examples:
        example_filters = {name.strip() for name in only_examples if name.strip()}
        if example_filters:
            python_pairs = [pair for pair in python_pairs if pair[2] in example_filters]

    suppressed_alias_pairs = 0
    if not example_filters:
        before_pairs = len(python_pairs)
        python_pairs = [
            pair
            for pair in python_pairs
            if pair[2] == pair[0].stem.replace("baseline_", "", 1)
        ]
        suppressed_alias_pairs = before_pairs - len(python_pairs)

    python_pairs, suppressed_variant_opts = _canonicalize_optimized_variants_for_full_sweep(
        python_pairs,
    )

    if only_cuda or only_python:
        cuda_wrapped_pairs = [pair for pair in python_pairs if _is_cuda_wrapper(pair[0])]
        if only_cuda:
            python_pairs = cuda_wrapped_pairs
        elif only_python:
            python_pairs = [pair for pair in python_pairs if pair not in cuda_wrapped_pairs]

    cuda_pairs = discover_cuda_benchmarks(chapter_dir)
    if example_filters:
        cuda_pairs = [pair for pair in cuda_pairs if pair[2] in example_filters]
    if only_python:
        cuda_pairs = []

    return (
        python_pairs,
        cuda_pairs,
        example_filters,
        suppressed_alias_pairs,
        suppressed_variant_opts,
    )


def _compute_global_progress_percent(
    *,
    completed_benchmarks: int,
    total_benchmarks: int,
    phase_index: int,
    total_phases: int,
    benchmark_offset: int = 0,
) -> Optional[float]:
    """Compute monotonic run-global progress for a benchmark phase."""

    if total_benchmarks <= 0:
        return None
    phase_progress = 0.0
    if phase_index > 0 and total_phases > 0:
        phase_progress = (phase_index - 1) / total_phases
    return ((benchmark_offset + completed_benchmarks + phase_progress) / total_benchmarks) * 100.0

# Import metric extraction utilities
try:
    from core.analysis.metric_extractor import (
        extract_from_ncu_report,
        extract_from_nsys_report,
    )
except ImportError:
    # Fallback if metric extractor not available
    def extract_from_ncu_report(path: Path) -> Dict[str, float]:
        return {}
    def extract_from_nsys_report(path: Path) -> Dict[str, float]:
        return {}


def reset_gpu_via_script(reason: str) -> None:
    """Invoke the GPU reset helper with the provided reason.

    Override the default script path by setting `AISP_GPU_RESET_SCRIPT`.
    """
    env_path = os.environ.get("AISP_GPU_RESET_SCRIPT")
    if env_path:
        reset_script = Path(env_path)
        if not reset_script.is_absolute():
            reset_script = (repo_root / reset_script).resolve()
    else:
        reset_script = Path(__file__).resolve().parents[1] / "scripts" / "reset_gpu.py"
    if not reset_script.exists():
        raise FileNotFoundError(
            f"GPU reset script not found at {reset_script}\n"
            f"Expected: {reset_script.resolve()}\n"
            f"This is a configuration error - the script must exist."
        )
    # Let errors propagate - no silent failures
    subprocess.run(
        [sys.executable, str(reset_script), "--reason", reason],
        check=True,
        timeout=180,
    )


def maybe_reset_gpu_for_error(error_message: str, context: str) -> None:
    """Optionally reset the GPU when the error message indicates a hang/timeout.

    Disabled by default because the reset helper terminates all GPU compute
    processes, which can include this harness runner. Enable explicitly by
    setting `AISP_AUTO_RESET_GPU_ON_TIMEOUT=1`.
    """
    normalized = error_message.strip().upper()
    if "TIMEOUT" not in normalized and "HANG" not in normalized:
        return
    flag = os.environ.get("AISP_AUTO_RESET_GPU_ON_TIMEOUT", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    reset_gpu_via_script(f"{context}: {error_message.splitlines()[0]}")


def extract_from_pytorch_trace(trace_path: Path) -> Dict[str, float]:
    """Extract metrics from PyTorch Chrome trace JSON file.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        Dictionary of extracted metrics
    """
    if not trace_path.exists():
        return {}
    
    metrics = {}
    
    try:
        with open(trace_path, 'r') as f:
            trace_data = json.load(f)
        
        # Chrome trace format: {"traceEvents": [...], "displayTimeUnit": "ms"}
        if isinstance(trace_data, dict) and "traceEvents" in trace_data:
            events = trace_data["traceEvents"]
            
            # Sum CUDA kernel times
            cuda_time_us = 0
            cpu_time_us = 0
            cuda_kernels = 0
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                # Look for CUDA kernel events
                if event.get("cat") == "cuda_runtime" or "cuda" in event.get("name", "").lower():
                    dur = event.get("dur", 0)  # Duration in microseconds
                    if dur > 0:
                        cuda_time_us += dur
                        cuda_kernels += 1
                
                # Look for CPU events
                if event.get("cat") == "cpu_op" or "cpu" in event.get("cat", "").lower():
                    dur = event.get("dur", 0)
                    if dur > 0:
                        cpu_time_us += dur
            
            if cuda_time_us > 0:
                metrics["pytorch_cuda_time_us"] = cuda_time_us
                metrics["pytorch_cuda_time_ms"] = cuda_time_us / 1000.0
            if cpu_time_us > 0:
                metrics["pytorch_cpu_time_us"] = cpu_time_us
                metrics["pytorch_cpu_time_ms"] = cpu_time_us / 1000.0
            if cuda_kernels > 0:
                metrics["pytorch_cuda_kernels"] = float(cuda_kernels)
                
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        logger.debug("Failed to parse profiler trace %s: %s", trace_path, exc)
    
    return metrics


# Examples that demonstrate techniques but may not show speedup (educational demos, analysis tools)
# These are valuable for showing HOW to implement something, even if not faster for this workload
# Note: The following legacy paths were previously under tools/ and are now in monitoring/ or core/ subpackages:
# - ch02/uma_memory_reporting -> labs/uma_memory/ (UMA reporting diagnostics)
# - speculative_decode/spec_config_sweep -> ch18/speculative_decode/ (shared helpers only)
# - occupancy_tuning/proton_* harness wrappers live in labs/occupancy_tuning; shared Triton schedules remain in core/profiling/occupancy_tuning/

def format_time_ms(time_ms: float) -> str:
    """Format time in milliseconds with adaptive precision.
    
    For very small values (< 1ms), use more decimal places to show actual timing.
    For larger values, use 2 decimal places.
    Handles zero and negative values appropriately.
    """
    if time_ms <= 0.0:
        return f"{time_ms:.2f}"
    elif time_ms < 0.001:
        return f"{time_ms:.6f}"  # microseconds precision
    elif time_ms < 0.01:
        return f"{time_ms:.5f}"
    elif time_ms < 0.1:
        return f"{time_ms:.4f}"
    elif time_ms < 1.0:
        return f"{time_ms:.3f}"
    else:
        return f"{time_ms:.2f}"


def format_throughput_summary(throughput_obj: Optional[Any]) -> str:
    """Pretty print throughput metrics for logs."""
    if throughput_obj is None:
        return ""
    parts: List[str] = []
    requests = getattr(throughput_obj, "requests_per_s", None)
    tokens = getattr(throughput_obj, "tokens_per_s", None)
    samples = getattr(throughput_obj, "samples_per_s", None)
    latency_ms = getattr(throughput_obj, "latency_ms", None)
    goodput = getattr(throughput_obj, "goodput", None)
    if requests:
        parts.append(f"{requests:,.2f} req/s")
    if tokens:
        parts.append(f"{tokens:,.2f} tokens/s")
    if samples and samples != tokens:
        parts.append(f"{samples:,.2f} samples/s")
    if latency_ms:
        parts.append(f"{latency_ms:.3f} ms/iter")
    if goodput is not None:
        parts.append(f"goodput={goodput:.2%}")
    return ", ".join(parts)


def serialize_throughput(throughput_obj: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert ThroughputStats into a JSON-serializable dict."""
    if throughput_obj is None:
        return None
    if hasattr(throughput_obj, "model_dump"):
        return throughput_obj.model_dump()
    if hasattr(throughput_obj, "__dict__"):
        return dict(throughput_obj.__dict__)
    return None


EXPECTATION_THROUGHPUT_FIELDS: Tuple[str, ...] = (
    "requests_per_s",
    "tokens_per_s",
    "samples_per_s",
    "goodput",
    "latency_ms",
)


def expectation_example_key(example_name: str, bench_type: str) -> str:
    bench_type = (bench_type or "python").lower()
    if bench_type == "python":
        return example_name
    return f"{example_name}_{bench_type}"


def find_best_optimization_entry(optimizations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best_entry: Optional[Dict[str, Any]] = None
    best_speed = float("-inf")
    for opt in optimizations or []:
        if opt.get("status") != "succeeded":
            continue
        speed = float(opt.get("speedup") or 0.0)
        if speed > best_speed:
            best_entry = opt
            best_speed = speed
    return best_entry


def _resolve_report_root(bench_root: Optional[Path]) -> Path:
    return bench_root.resolve() if bench_root else repo_root


def _format_rel_link(target: Path, base_dir: Path) -> str:
    try:
        return target.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        try:
            return os.path.relpath(str(target.resolve()), str(base_dir.resolve()))
        except Exception:
            return str(target)


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _safe_read_text_with_warning(
    path: Path,
    *,
    label: str,
    warnings_list: Optional[List[str]] = None,
) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        warning = f"Failed to read {label} from {path}: {exc}"
        logger.warning(warning)
        if warnings_list is not None and warning not in warnings_list:
            warnings_list.append(warning)
        return None


def _emit_run_benchmark_warning(
    message: str,
    *,
    exc: Optional[BaseException] = None,
    warnings_list: Optional[List[str]] = None,
    logger_obj: Any = None,
) -> str:
    warning = f"{message}: {exc}" if exc is not None else message
    target_logger = logger_obj if logger_obj is not None else logger
    target_logger.warning(warning)
    if warnings_list is not None and warning not in warnings_list:
        warnings_list.append(warning)
    return warning


def _append_profile_warning(log_path: Path, message: str) -> None:
    """Persist profiler warnings to logs and surface them through the logger."""
    if LOGGER_AVAILABLE:
        logger.warning("  %s", message)
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception as exc:
        if LOGGER_AVAILABLE:
            logger.warning("  Failed to append profiler warning to %s: %s", log_path, exc)


def _truncate_text(text: str, max_lines: int = 80, max_chars: int = 4000) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    truncated = "\n".join(lines)
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars] + "\n... (truncated)"
    return truncated


def _extract_markdown_section(text: str, header: str) -> Optional[str]:
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start_idx = idx + 1
            break
    if start_idx is None:
        return None
    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        if lines[idx].startswith("## ") and lines[idx].strip() != header:
            end_idx = idx
            break
    section = "\n".join(lines[start_idx:end_idx]).strip()
    return section or None


def _to_blockquote(text: str) -> str:
    return "\n".join(f"> {line}".rstrip() if line.strip() else ">" for line in text.splitlines())


def _generate_diff(
    original_text: str,
    patched_text: str,
    *,
    from_label: str,
    to_label: str,
    max_lines: int = 200,
) -> Optional[str]:
    diff_lines = list(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
        )
    )
    if not diff_lines:
        return None
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... (diff truncated)\n"]
    return "".join(diff_lines)


def _resolve_source_file(bench: Dict[str, Any], chapter_dir: Path) -> Optional[Path]:
    best_opt = find_best_optimization_entry(bench.get("optimizations", []))
    candidates = [
        best_opt.get("file") if best_opt else None,
        bench.get("baseline_file"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = chapter_dir / candidate
        if path.exists():
            return path
    return None


def _benchmark_has_llm_data(bench: Dict[str, Any]) -> bool:
    return bool(bench.get("llm_analysis") or bench.get("llm_patches") or bench.get("best_llm_patch"))


def start_progress_watchdog(
    logger,
    chapter_name: str,
    warn_after: float = 300.0,
    ping_every: float = 90.0,
):
    """Spawn a background watchdog that emits progress heartbeats and hang warnings."""
    state = {
        "last_progress": time.time(),
        "last_note": "initializing chapter",
        "warned": False,
        "active": True,
    }
    stop_event = threading.Event()
    ping_every = max(30.0, ping_every)
    warn_after = max(ping_every * 2.0, warn_after)

    def _phase_warn_after_seconds(note: str) -> float:
        note_l = (note or "").lower()
        if "ncu profiling" in note_l:
            return max(warn_after, 1800.0)
        if "nsys profiling" in note_l or "torch profiling" in note_l:
            return max(warn_after, 900.0)
        return warn_after

    def heartbeat() -> None:
        while not stop_event.wait(ping_every):
            if not state["active"]:
                break
            elapsed = time.time() - state["last_progress"]
            warn_after_for_phase = _phase_warn_after_seconds(state["last_note"])
            if elapsed >= warn_after_for_phase:
                if not state["warned"]:
                    logger.warning(
                        "    ⏱️ No benchmark progress in %s for %.0fs (last completed: %s)",
                        chapter_name,
                        elapsed,
                        state["last_note"],
                    )
                    state["warned"] = True
            else:
                logger.info(
                    "    …%s still running (last completed: %s, %.0fs ago)",
                    chapter_name,
                    state["last_note"],
                    elapsed,
                )

    thread = threading.Thread(
        target=heartbeat,
        name=f"{chapter_name}_progress_watchdog",
        daemon=True,
    )
    thread.start()

    def record(note: str) -> None:
        gap = time.time() - state["last_progress"]
        state["last_progress"] = time.time()
        state["last_note"] = note
        if state["warned"]:
            logger.info(
                "    ✅ Progress resumed after %.0fs (now at %s)",
                gap,
                note,
            )
            state["warned"] = False

    def stop() -> None:
        state["active"] = False
        stop_event.set()
        thread.join(timeout=1.0)

    return record, stop


def _capture_metric(metrics: Dict[str, float], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    if key in METRIC_DIRECTIONS:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value_f):
            return
        metrics[key] = value_f


def _capture_payload(metrics: Dict[str, float], prefix: str, payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not isinstance(payload, dict):
        return
    for field in EXPECTATION_THROUGHPUT_FIELDS:
        metric_key = f"{prefix}.{field}"
        if metric_key not in METRIC_DIRECTIONS:
            continue
        value = payload.get(field)
        if isinstance(value, (int, float)):
            value_f = float(value)
            if math.isfinite(value_f):
                metrics[metric_key] = value_f


def _capture_custom_metrics(metrics: Dict[str, float], prefix: str, payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not isinstance(payload, dict):
        return
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}"
        if metric_key not in METRIC_DIRECTIONS:
            continue
        if isinstance(value, (int, float)):
            value_f = float(value)
            if math.isfinite(value_f):
                metrics[metric_key] = value_f


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _coerce_positive_float(value: Any) -> Optional[float]:
    value_f = _coerce_finite_float(value)
    if value_f is None or value_f <= 0:
        return None
    return value_f


_MIN_SPEEDUP_FOR_SUCCESS = 1.05


def _resolve_min_speedup_for_success(result_entry: Dict[str, Any]) -> float:
    configured = _coerce_finite_float(result_entry.get("minimum_required_speedup"))
    if configured is None or configured <= 0:
        return _MIN_SPEEDUP_FOR_SUCCESS
    return configured


def _should_fail_no_speedup(result_entry: Dict[str, Any]) -> bool:
    optimization_goal = str(result_entry.get("optimization_goal") or "speed").strip().lower()
    if optimization_goal != "speed":
        return False
    best_speedup = _coerce_finite_float(result_entry.get("best_speedup"))
    if best_speedup is None:
        return False
    return best_speedup < _resolve_min_speedup_for_success(result_entry)


def _format_failed_no_speedup(result_entry: Dict[str, Any]) -> str:
    best_speedup = _coerce_finite_float(result_entry.get("best_speedup"))
    rendered_speedup = f"{best_speedup:.2f}x" if best_speedup is not None else "n/a"
    required_speedup = _resolve_min_speedup_for_success(result_entry)
    return (
        f"Best speedup {rendered_speedup} below required "
        f"{required_speedup:.2f}x threshold for speed-goal benchmark"
    )


def _collect_all_skipped_optimization_reason(optimizations: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not optimizations:
        return None
    reasons: List[str] = []
    for optimization in optimizations:
        if str(optimization.get("status") or "").strip().lower() != "skipped":
            return None
        raw_reason = str(
            optimization.get("skip_reason")
            or optimization.get("error")
            or ""
        ).strip()
        if not raw_reason:
            return None
        normalized_reason = raw_reason
        if normalized_reason.startswith("HARDWARE/SOFTWARE LIMITATION:"):
            normalized_reason = normalized_reason.split(":", 1)[1].strip()
        elif normalized_reason.upper().startswith("SKIPPED:"):
            normalized_reason = normalized_reason.split(":", 1)[1].strip()
        if not normalized_reason:
            return None
        if normalized_reason not in reasons:
            reasons.append(normalized_reason)
    if not reasons:
        return None
    if len(reasons) == 1:
        return reasons[0]
    return "; ".join(reasons)


def collect_expectation_metrics(result_entry: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    """Collect metrics for expectation tracking.

    Uses select_best_optimization() as single source of truth for selecting
    the best optimization. Speedup is derived from timing values, not stored
    independently.
    """
    metrics: Dict[str, float] = {}
    optimization_goal = (result_entry.get("optimization_goal") or "speed").strip().lower()

    if optimization_goal == "memory":
        baseline_memory = result_entry.get("baseline_memory_mb")
        _capture_metric(metrics, "baseline_memory_mb", baseline_memory)

        best_opt = select_best_optimization(result_entry.get("optimizations", []), goal="memory")
        if best_opt:
            optimized_memory = best_opt.get("memory_mb")
            _capture_metric(metrics, "best_optimized_memory_mb", optimized_memory)
            baseline_mem_f = _coerce_positive_float(baseline_memory)
            optimized_mem_f = _coerce_positive_float(optimized_memory)
            if baseline_mem_f is not None and optimized_mem_f is not None:
                _capture_metric(metrics, "best_memory_savings_ratio", baseline_mem_f / optimized_mem_f)
                _capture_metric(
                    metrics,
                    "best_memory_savings_pct",
                    ((baseline_mem_f - optimized_mem_f) / baseline_mem_f) * 100.0,
                )
        return metrics, best_opt

    # Capture baseline metrics
    baseline_time = result_entry.get("baseline_time_ms")
    _capture_metric(metrics, "baseline_time_ms", baseline_time)
    _capture_metric(metrics, "baseline_p75_ms", result_entry.get("baseline_p75_ms"))
    _capture_metric(metrics, "baseline_p90_ms", result_entry.get("baseline_p90_ms"))

    baseline_throughput = result_entry.get("baseline_throughput")
    _capture_payload(metrics, "baseline_throughput", baseline_throughput)

    baseline_custom = result_entry.get("baseline_custom_metrics")
    _capture_custom_metrics(metrics, "baseline_custom", baseline_custom)

    # Use single source of truth for selecting best optimization
    best_opt = select_best_optimization(result_entry.get("optimizations", []), goal="speed")
    if best_opt:
        optimized_time = best_opt.get("time_ms")
        _capture_metric(metrics, "best_optimized_time_ms", optimized_time)
        _capture_metric(metrics, "best_optimized_p75_ms", best_opt.get("p75_ms"))
        _capture_metric(metrics, "best_optimized_p90_ms", best_opt.get("p90_ms"))
        _capture_payload(metrics, "best_optimized_throughput", best_opt.get("throughput"))
        _capture_custom_metrics(metrics, "best_optimized_custom", best_opt.get("custom_metrics"))

        # Derive speedup from timing values (not from stored value)
        baseline_time_f = _coerce_positive_float(baseline_time)
        optimized_time_f = _coerce_positive_float(optimized_time)
        if baseline_time_f is not None and optimized_time_f is not None:
            derived_speedup = compute_speedup(baseline_time_f, optimized_time_f)
            if math.isfinite(derived_speedup) and derived_speedup > 0:
                _capture_metric(metrics, "best_speedup", derived_speedup)
                _capture_metric(metrics, "best_optimized_speedup", derived_speedup)
        else:
            # Fall back to stored speedup if timing not available
            stored_speedup = _coerce_positive_float(best_opt.get("speedup"))
            if stored_speedup is not None:
                _capture_metric(metrics, "best_speedup", stored_speedup)
                _capture_metric(metrics, "best_optimized_speedup", stored_speedup)
    else:
        # No successful optimization - use result_entry's best_speedup (likely 1.0)
        best_speedup = _coerce_positive_float(result_entry.get("best_speedup"))
        if best_speedup is not None:
            _capture_metric(metrics, "best_speedup", best_speedup)

    return metrics, best_opt


def build_expectation_metadata(
    result_entry: Dict[str, Any],
    best_opt: Optional[Dict[str, Any]],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    """Build metadata for expectation tracking.

    Ensures metadata speedup matches metrics speedup by deriving from timing
    values rather than using stored speedup.
    """
    metadata: Dict[str, Any] = {
        "example": result_entry.get("example"),
        "type": result_entry.get("type", "python"),
        "optimization_goal": result_entry.get("optimization_goal", "speed"),
    }
    required_speedup = _coerce_finite_float(result_entry.get("minimum_required_speedup"))
    if required_speedup is not None and required_speedup > 0:
        metadata["minimum_required_speedup"] = required_speedup
    if git_commit:
        metadata["git_commit"] = git_commit
    if best_opt:
        metadata["best_optimization"] = best_opt.get("technique") or best_opt.get("file")
        metadata["best_optimization_file"] = best_opt.get("file")
        metadata["best_optimization_time_ms"] = best_opt.get("time_ms")
        if (result_entry.get("optimization_goal") or "speed").strip().lower() == "memory":
            metadata["best_optimization_memory_mb"] = best_opt.get("memory_mb")
            baseline_memory = result_entry.get("baseline_memory_mb")
            optimized_memory = best_opt.get("memory_mb")
            baseline_mem_f = _coerce_positive_float(baseline_memory)
            optimized_mem_f = _coerce_positive_float(optimized_memory)
            if baseline_mem_f is not None and optimized_mem_f is not None:
                metadata["best_memory_savings_ratio"] = baseline_mem_f / optimized_mem_f
                metadata["best_memory_savings_pct"] = ((baseline_mem_f - optimized_mem_f) / baseline_mem_f) * 100.0
            return metadata

        # Derive speedup from timing values for consistency with metrics
        baseline_time = result_entry.get("baseline_time_ms")
        optimized_time = best_opt.get("time_ms")
        baseline_time_f = _coerce_positive_float(baseline_time)
        optimized_time_f = _coerce_positive_float(optimized_time)
        if baseline_time_f is not None and optimized_time_f is not None:
            derived = compute_speedup(baseline_time_f, optimized_time_f)
            if math.isfinite(derived) and derived > 0:
                metadata["best_optimization_speedup"] = derived
        else:
            stored_speedup = _coerce_positive_float(best_opt.get("speedup"))
            if stored_speedup is not None:
                metadata["best_optimization_speedup"] = stored_speedup
    return metadata


def _format_metric_value(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)):
        return str(value) if value is not None else "n/a"
    if math.isnan(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.5f}"


def _format_profiler_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _format_metric_value(float(value))
    return str(value)


def log_profiler_metrics_table(
    logger,
    metrics: Dict[str, Dict[str, Any]],
    indent: str = "",
) -> None:
    rows: List[Dict[str, str]] = []
    for profiler_name, profiler_metrics in sorted(metrics.items()):
        for metric_key, value in sorted(profiler_metrics.items()):
            rows.append(
                {
                    "Profiler": profiler_name,
                    "Metric": metric_key,
                    "Value": _format_profiler_value(value),
                }
            )
    if not rows:
        logger.info(f"{indent}(no profiler metrics)")
        return
    headers = ["Profiler", "Metric", "Value"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    logger.info(f"{indent}{header_line}")
    logger.info(f"{indent}{divider}")
    for row in rows:
        line = " | ".join(row.get(header, "").ljust(widths[header]) for header in headers)
        logger.info(f"{indent}{line}")


def log_expectation_evaluation(
    logger,
    evaluation: Optional[Any],
    repo_root: Path,
) -> None:
    """Log expectation evaluation results with enhanced regression display.

    Shows:
    - Metric comparison table with status indicators
    - Visual indicators for regressions (⚠️) and improvements (🚀)
    - Actual speedup values (never clamped)
    """
    if evaluation is None:
        return
    rel_path = None
    if evaluation.expectation_path:
        rel_path = _repo_relative_path(evaluation.expectation_path, repo_root)
    header = f"    Expectations [{evaluation.hardware_key}]"
    if rel_path:
        header += f": {rel_path}"
    logger.info(header)
    comparisons = evaluation.comparisons or []
    if not comparisons:
        logger.info("      (no expectation comparisons)")
        return

    headers = ["Metric", "Observed", "Expected", "Delta", "Δ%", "Status"]
    rows: List[Dict[str, str]] = []
    for comp in comparisons:
        delta_pct = comp.get("delta_pct")
        pct_str = "n/a"
        if delta_pct is not None and not math.isinf(delta_pct):
            pct_str = f"{delta_pct:+.2f}%"
        elif delta_pct is not None and math.isinf(delta_pct):
            pct_str = "+inf%" if delta_pct > 0 else "-inf%"

        # Enhanced status display with visual indicators
        status = comp.get("status", "")
        if status == "regressed":
            status = "⚠️ regressed"
        elif status == "improved":
            status = "🚀 improved"
        elif status == "met":
            status = "✓ met"

        rows.append(
            {
                "Metric": comp.get("metric", ""),
                "Observed": _format_metric_value(comp.get("observed")),
                "Expected": _format_metric_value(comp.get("expected")),
                "Delta": _format_metric_value(comp.get("delta")),
                "Δ%": pct_str,
                "Status": status,
            }
        )
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row.get(header, "")))
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    logger.info(f"      {header_line}")
    logger.info(f"      {divider}")
    for row in rows:
        line = " | ".join(row.get(header, "").ljust(widths[header]) for header in headers)
        logger.info(f"      {line}")

    # Show regression summary if any regressions detected
    if evaluation.regressed:
        regression_count = len(evaluation.regressions)
        logger.warning(f"      ⚠️ {regression_count} metric(s) regressed from expected values")


def log_expectation_delta(
    logger,
    *,
    example_key: str,
    goal: str,
    old_entry: Optional[ExpectationEntry],
    new_entry: ExpectationEntry,
    update_result: Optional[Any],
    event_logger: Optional["BenchmarkEventLogger"] = None,
    chapter: Optional[str] = None,
) -> None:
    """Log primary expectation delta for live monitoring during long runs."""
    goal_norm = (goal or "speed").strip().lower()
    new_score = new_entry.primary_improvement
    old_score = old_entry.primary_improvement if old_entry else None
    old_provenance = old_entry.provenance.to_dict() if old_entry else None
    new_provenance = new_entry.provenance.to_dict()
    update_message = update_result.message if update_result else None
    validation_issues = (
        [issue.to_dict() for issue in update_result.validation_issues]
        if update_result
        else []
    )
    validation_issue_types = [issue["issue_type"] for issue in validation_issues]
    provenance_mismatch_fields: list[str] = []
    for issue in validation_issues:
        if issue.get("issue_type") != "provenance_mismatch":
            continue
        stored_value = issue.get("stored_value")
        expected_value = issue.get("expected_value")
        if isinstance(stored_value, dict) and isinstance(expected_value, dict):
            for field_name in sorted(set(stored_value) | set(expected_value)):
                if stored_value.get(field_name) != expected_value.get(field_name):
                    provenance_mismatch_fields.append(field_name)
        break
    if not provenance_mismatch_fields and old_entry is not None:
        provenance_mismatch_fields = new_entry.provenance.mismatch_fields(old_entry.provenance)
    delta = None
    delta_pct = None
    if old_score is not None:
        delta = new_score - old_score
        if old_score != 0:
            delta_pct = (delta / old_score) * 100
    status = update_result.status if update_result else "unknown"
    logger.info(
        "    Expectations delta: example=%s goal=%s status=%s old_score=%s new_score=%.3f delta=%s delta_pct=%s",
        example_key,
        goal_norm,
        status,
        f"{old_score:.3f}" if old_score is not None else "none",
        new_score,
        f"{delta:.3f}" if delta is not None else "n/a",
        f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a",
    )
    if goal_norm == "memory":
        logger.info(
            "    Expectations metrics: baseline_memory_mb=%s optimized_memory_mb=%s memory_savings_ratio=%.3f",
            f"{new_entry.baseline_memory_mb:.3f}" if new_entry.baseline_memory_mb is not None else "n/a",
            f"{new_entry.best_optimized_memory_mb:.3f}" if new_entry.best_optimized_memory_mb is not None else "n/a",
            new_score,
        )
    else:
        logger.info(
            "    Expectations metrics: baseline_time_ms=%.3f optimized_time_ms=%.3f speedup=%.3f",
            new_entry.baseline_time_ms,
            new_entry.best_optimized_time_ms,
            new_score,
        )
    if event_logger:
        event_logger.emit(
            "expectation_update",
            chapter=chapter,
            example=example_key,
            goal=goal_norm,
            status=status,
            update_message=update_message,
            validation_issue_types=validation_issue_types,
            validation_issues=validation_issues,
            old_provenance=old_provenance,
            new_provenance=new_provenance,
            provenance_mismatch_fields=provenance_mismatch_fields,
            old_score=old_score,
            new_score=new_score,
            delta=delta,
            delta_pct=delta_pct,
            baseline_time_ms=new_entry.baseline_time_ms,
            optimized_time_ms=new_entry.best_optimized_time_ms,
            baseline_memory_mb=new_entry.baseline_memory_mb,
            optimized_memory_mb=new_entry.best_optimized_memory_mb,
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)


class BenchmarkEventLogger:
    """Append-only JSONL event logger for benchmark runs."""

    def __init__(self, path: Path, run_id: str, logger: Any) -> None:
        self.path = path
        self.run_id = run_id
        self.logger = logger
        self.seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")

    def emit(self, event_type: str, **payload: Any) -> None:
        self.seq += 1
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "run_id": self.run_id,
            "seq": self.seq,
        }
        event.update(payload)
        line = json.dumps(event, default=_json_default)
        self._fh.write(line + "\n")
        self._fh.flush()
        # Log a log-file friendly version for humans and dashboards.
        self.logger.info("EVENT %s %s", event_type, json.dumps(payload, default=_json_default))

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception as exc:
            _emit_run_benchmark_warning(
                f"Failed to close benchmark event log {self.path}",
                exc=exc,
                logger_obj=self.logger,
            )


def emit_event(
    event_logger: Optional["BenchmarkEventLogger"],
    logger: Any,
    event_type: str,
    **payload: Any,
) -> None:
    if event_logger:
        event_logger.emit(event_type, **payload)
        return
    logger.info("EVENT %s %s", event_type, json.dumps(payload, default=_json_default))


def reset_cuda_state(*, allow_cuda_context: bool = True):
    """Reset CUDA state to prevent cascading failures.
    
    Performs thorough cleanup:
    - Garbage collection to release Python references
    - Empty CUDA cache to free GPU memory
    - Synchronize all CUDA streams
    - Reset peak memory stats
    - Clear torch.compile caches
    - Reset CUDA graph memory pool
    - Clear TMA descriptor caches (critical for TMA kernel stability)
    """
    import gc
    
    # Force garbage collection first to release Python references to CUDA tensors
    gc.collect()

    if allow_cuda_context:
        try:
            if torch.cuda.is_available():
                # Synchronize first to complete pending operations
                torch.cuda.synchronize()
                
                # Empty cache to free all unreferenced memory
                torch.cuda.empty_cache()
                
                # Reset memory tracking stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Trim CUDA graph memory pool - critical for TMA kernel stability
                # This prevents stale graph state from affecting TMA tensor map encoding
                try:
                    # This is the correct API to release CUDA graph memory
                    if hasattr(torch.cuda, 'graph_pool_trim'):
                        torch.cuda.graph_pool_trim()
                except Exception as exc:
                    _emit_run_benchmark_warning("Failed to reset CUDA graph pool", exc=exc)
                
                # CRITICAL: Reset CUDA random number generator state
                # CUDA graphs capture the RNG offset, which causes "Offset increment 
                # outside graph capture" errors when subsequent benchmarks use torch.randn
                try:
                    device_idx = torch.cuda.current_device()
                    gen = torch.cuda.default_generators[device_idx]
                    # set_offset(0) properly resets the graph capture state
                    # manual_seed alone is not sufficient
                    gen.set_offset(0)
                    gen.manual_seed(torch.initial_seed() % (2**63))
                except Exception as exc:
                    _emit_run_benchmark_warning("Failed to reset CUDA RNG state", exc=exc)
                
                # Reset default CUDA stream to clear any pending operations
                try:
                    default_stream = torch.cuda.current_stream()
                    default_stream.synchronize()
                except Exception as exc:
                    _emit_run_benchmark_warning("Failed to synchronize default CUDA stream", exc=exc)
                
                # Another sync to ensure cleanup is complete
                torch.cuda.synchronize()
        except RuntimeError:
            pass  # CUDA not initialized or error
    
    # Clear torch.compile caches to prevent stale compiled code
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_reset = getattr(dynamo, "reset", None)
    if callable(dynamo_reset):
        try:
            dynamo_reset()
        except Exception as exc:
            _emit_run_benchmark_warning("Failed to reset torch._dynamo state", exc=exc)

    # Clear torch inductor caches as well
    inductor = getattr(torch, "_inductor", None)
    cudagraph_trees = getattr(inductor, "cudagraph_trees", None)
    reset_cudagraph_trees = getattr(cudagraph_trees, "reset_cudagraph_trees", None)
    if callable(reset_cudagraph_trees):
        try:
            reset_cudagraph_trees()
        except Exception as exc:
            _emit_run_benchmark_warning("Failed to reset torch._inductor cudagraph trees", exc=exc)

    reset_triton_runtime_cache(
        lambda message, exc: _emit_run_benchmark_warning(message, exc=exc)
    )
    
    # Second GC pass to clean up any CUDA objects freed above
    gc.collect()


def _reset_parent_execution_state(
    *,
    launch_via: Any = "python",
    cold_start: bool = False,
    include_gpu_state: bool = False,
) -> None:
    """Reset parent-process state between benchmark phases and chapters."""
    allow_cuda_context = str(launch_via).strip().lower() == "torchrun"
    reset_cuda_state(allow_cuda_context=allow_cuda_context)
    if include_gpu_state and allow_cuda_context:
        reset_gpu_state()
    if cold_start and allow_cuda_context:
        reset_gpu_state()


def clean_build_directories(chapter_dir: Path) -> None:
    """Clean build directories to prevent stale lock issues.

    IMPORTANT: Do not aggressively delete lock files.
    PyTorch CUDA extension builds use a file-baton lock (a plain `lock` file).
    Unlinking that lock while a build is in progress can crash the builder on
    release and break subsequent benchmark runs.
    """
    skip_raw = os.environ.get("AISP_SKIP_BUILD_CLEAN")
    clean_raw = os.environ.get("AISP_CLEAN_BUILD_DIRS")
    skip_enabled = False
    if skip_raw is not None and str(skip_raw).strip().lower() in {"1", "true", "yes", "on", "skip"}:
        skip_enabled = True
    if clean_raw is not None and str(clean_raw).strip().lower() in {"0", "false", "no", "off"}:
        skip_enabled = True
    if skip_enabled:
        logger.info("Skipping build directory cleanup (AISP_SKIP_BUILD_CLEAN/AISP_CLEAN_BUILD_DIRS override).")
        return

    try:
        from core.utils.build_utils import ensure_clean_build_directory
    except ImportError:
        return

    # Be conservative: treat only very old locks as stale.
    # CUDA extension compiles can take minutes; a too-small threshold will kill
    # legitimate in-progress builds if another run overlaps.
    max_lock_age_seconds = 300

    # Clean chapter-local build roots (and their direct children, which are
    # commonly extension build directories like build/<ext_name>/).
    build_root = chapter_dir / "build"
    if build_root.exists():
        candidates: list[Path] = [build_root]
        try:
            candidates.extend([p for p in build_root.iterdir() if p.is_dir()])
        except Exception as exc:
            _emit_run_benchmark_warning(
                f"Failed to inspect build directory children under {build_root}",
                exc=exc,
            )

        for build_dir in candidates:
            try:
                ensure_clean_build_directory(build_dir, max_lock_age_seconds=max_lock_age_seconds)
            except Exception as exc:
                logger.warning("Failed to clean build directory %s: %s", build_dir, exc)

    # Clean torch extensions cache for this chapter.
    torch_ext_dir = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            Path.home() / ".cache" / "torch_extensions",
        )
    )
    chapter_name = chapter_dir.name
    for ext_dir in torch_ext_dir.glob(f"py*/{chapter_name}*"):
        if not ext_dir.is_dir():
            continue
        try:
            ensure_clean_build_directory(ext_dir, max_lock_age_seconds=max_lock_age_seconds)
        except Exception as exc:
            logger.warning("Failed to clean torch extensions directory %s: %s", ext_dir, exc)

    # Also clean torch inductor cache locks (best-effort).
    inductor_cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor"))
    if inductor_cache.exists():
        try:
            ensure_clean_build_directory(inductor_cache, max_lock_age_seconds=max_lock_age_seconds)
        except Exception as exc:
            logger.warning("Failed to clean torch inductor cache %s: %s", inductor_cache, exc)


def is_distributed_benchmark(file_path: Path) -> bool:
    """Check if a benchmark file contains distributed operations.
    
    This function detects distributed benchmarks by looking for:
    - torch.distributed imports and usage
    - DistributedDataParallel (DDP)
    - NCCL backend usage
    - Environment variables like WORLD_SIZE, RANK
    - Multi-GPU communication patterns
    """
    name_lower = file_path.name.lower()
    # Fast path: examples explicitly suffixed as multi-GPU should be skipped
    # up-front on 1-GPU systems instead of failing at runtime.
    if any(token in name_lower for token in ("_multigpu", "multi_gpu", "multi-gpu")):
        return True

    try:
        content = file_path.read_text()
        content_lower = content.lower()

        # Some benchmarks use torchrun or distributed primitives as a 1-process
        # control surface while explicitly declaring that multiple GPUs are not
        # required. Respect that contract instead of skipping them up-front on
        # 1-GPU hosts.
        has_single_gpu_override = bool(
            re.search(r"multi_gpu_required\s*=\s*False", content)
        )
        if has_single_gpu_override:
            return False
        
        # Check for distributed imports
        has_dist_import = any(pattern in content for pattern in [
            'import torch.distributed',
            'from torch.distributed',
            'torch.distributed as dist',
        ])
        
        # Check for distributed operations
        has_dist_ops = any(pattern in content for pattern in [
            'dist.init_process_group',
            'torch.distributed.init_process_group',
            'torch.nn.parallel.DistributedDataParallel',
            'DistributedDataParallel(',
            'DDP(',
        ])
        
        # Check for NCCL backend (strong indicator of multi-GPU)
        has_nccl = any(pattern in content for pattern in [
            "backend='nccl'",
            'backend="nccl"',
            'backend = "nccl"',
            'backend = \'nccl\'',
        ])
        
        # Check for distributed environment variables (but not just setup code)
        # Only count if it's actually used, not just set
        has_world_size = 'WORLD_SIZE' in content and ('os.environ' in content or 'getenv' in content)
        has_rank = 'RANK' in content and ('os.environ' in content or 'getenv' in content)
        
        # Some examples gate their execution behind helper functions such as
        # skip_if_insufficient_gpus() even if they do not import torch.distributed.
        # Treat those as multi-GPU benchmarks so we can skip them up-front.
        multi_gpu_markers = (
            "skip_if_insufficient_gpus",
            "requires_multiple_gpus",
            "requires_multi_gpu",
            "MultiGPUBenchmark",
            "MIN_GPUS_REQUIRED",
        )
        has_explicit_multi_gpu_guard = any(marker in content for marker in multi_gpu_markers)

        explicit_gpu_checks = (
            "device_count() < 2",
            "device_count()<2",
            "requires >=2 gpu",
            "requires >=2 gpus",
            "requires >= 2 gpu",
            "requires >= 2 gpus",
            "requires multiple gpus",
        )
        has_explicit_gpu_check = any(token in content_lower for token in explicit_gpu_checks)
        
        # Torchrun is a strong indicator of multi-process / multi-GPU execution even if
        # distributed init is abstracted behind helpers imported from other modules.
        has_torchrun_launch = "LaunchVia.TORCHRUN" in content or "TorchrunLaunchSpec" in content

        # A benchmark is distributed if it has distributed imports AND operations,
        # OR if it explicitly uses NCCL backend, OR if it contains explicit
        # multi-GPU guard helpers, OR if it launches via torchrun.
        return (
            (has_dist_import and has_dist_ops)
            or has_nccl
            or (has_world_size and has_rank and has_dist_ops)
            or has_explicit_multi_gpu_guard
            or has_explicit_gpu_check
            or has_torchrun_launch
        )
    except Exception:
        return False


def check_hardware_limitation(error_msg: str) -> Optional[str]:
    """Check if error is due to hardware/software limitation and return skip reason.
    
    Only skips for TRUE hardware/software limitations that cannot be fixed in code:
    - Triton SM 12.1 bug (sm_121a issue)
    - Missing Triton features (e.g., DSMEM/TMA APIs not available in installed version)
    
    For other issues, we should fix them instead of skipping:
    - CUTLASS: Verify it's actually unavailable before skipping
    - CUDA extensions: Should be pre-compiled, not skipped
    - torch.compile timeouts: Should reduce model size, not skip
    - Device-side asserts: Already handled with reset_cuda_state()
    """
    error_lower = error_msg.lower()
    
    # FAIL FAST markers - keep serving-stack mismatches as hard errors.
    if 'fail fast:' in error_lower:
        # Serving runtime/ABI/version mismatches must fail hard so strict sweeps
        # do not silently pass with skipped serving examples.
        if (
            "serving stack mismatch" in error_lower
            or "vllm abi mismatch" in error_lower
            or "vllm import failed" in error_lower
            or "vllm._c failed to import" in error_lower
            or "requires vllm execution" in error_lower
        ):
            return None
        # Extract the reason after "FAIL FAST:"
        idx = error_lower.find('fail fast:')
        reason = error_msg[idx + len('fail fast:'):].strip()
        # Truncate at first period or newline for cleaner display
        if '.' in reason:
            reason = reason[:reason.index('.') + 1]
        return f"SKIPPED (software limitation): {reason}"
    
    # Treat explicit SKIPPED markers as hardware limitations.
    if 'skipped:' in error_lower:
        reason = error_msg[error_lower.find('skipped:') + len('skipped:'):].strip()
        return reason or "Benchmark reported SKIPPED"
    
    # Device-side assert cascades - these should be prevented by reset_cuda_state()
    # But if they still occur, it's a transient state issue, not a hardware limitation
    if 'device-side assert' in error_lower or 'cudaerrorassert' in error_lower:
        # Don't skip - reset should handle this. Return None to let it fail normally.
        return None
    
    # CUDA pipeline API unavailable (older GPUs)
    if 'cuda pipeline api unavailable' in error_lower:
        return "CUDA Pipeline API not supported on this GPU"
    if 'requires compute capability >= 8.0' in error_lower and 'pipeline' in error_lower:
        return "CUDA Pipeline API requires compute capability >= 8.0"
    
    if 'distributed shared memory unavailable' in error_lower:
        return "Distributed shared memory (DSMEM) not enabled on this hardware/driver"
    if 'cuda 13+ required for cluster dsmem support' in error_lower:
        return "Distributed shared memory (DSMEM) not supported by this toolkit/runtime"
    if 'thread block clusters unavailable' in error_lower or 'cluster target block not present' in error_lower:
        return "Thread block clusters unavailable on this driver/toolkit (needs CUDA 13.1+ or compute-sanitizer)"
    
    # Triton version limitations - missing features in installed Triton version
    if 'triton' in error_lower and ('missing required' in error_lower or 'is missing' in error_lower):
        # Extract feature names if possible
        if 'dsmem' in error_lower or 'tma' in error_lower or 'cluster' in error_lower:
            return f"Triton version missing Blackwell features (DSMEM/TMA/cluster). Requires newer Triton."
        return "Triton version missing required features"
    if 'distributed benchmark requires multiple gpus' in error_lower:
        if 'SKIPPED:' in error_msg:
            return error_msg.split('SKIPPED:', 1)[1].strip()
        return "Distributed benchmark requires multiple GPUs (insufficient GPUs available)"
    if (
        'world size mismatch' in error_lower
        and 'visible gpu' in error_lower
        and ('requires >= 2' in error_lower or 'requires exactly 2' in error_lower)
    ):
        return "Distributed benchmark requires multiple GPUs (insufficient GPUs available)"

    # Serving/inference dependency mismatches are hard failures.
    # They indicate benchmark environment drift, not optional behavior.
    if (
        "undefined symbol" in error_lower
        and ("vllm/_c.abi3.so" in error_lower or "vllm._c" in error_lower)
    ) or "c10_cuda_check_implementation" in error_lower:
        return None
    if (
        "vllm required for this benchmark" in error_lower
        or "no module named 'vllm'" in error_lower
    ):
        return None
    if "no module named 'flashinfer'" in error_lower:
        return None
    if (
        "cutlass python package (cutlass_library) is missing" in error_lower
        or "install nvidia-cutlass-dsl" in error_lower
    ):
        return "CUTLASS Python package (nvidia-cutlass-dsl / cutlass_library) is not installed"
    if (
        "no suitable algorithm found for nvfp4 gemm" in error_lower
        or "fp4 not supported at all on this system" in error_lower
    ):
        return "cuBLASLt NVFP4 algorithm unavailable on this driver/toolchain"
    
    # Segmentation faults - these should be prevented by pre-compilation
    # If they still occur after pre-compilation, it's a real issue, not a limitation
    if 'segmentation fault' in error_lower or 'segfault' in error_lower or 'sigsegv' in error_lower:
        # Don't skip - extensions should be pre-compiled
        return None
    
    # CUTLASS backend - verify it's actually unavailable before skipping
    if 'cutlass' in error_lower and ('attributeerror' in error_lower or 'loweringexception' in error_lower):
        # Check if CUTLASS is actually installed
        try:
            import cutlass
            import importlib_metadata
            try:
                version = importlib_metadata.version("nvidia-cutlass-dsl")
                # CUTLASS is installed - this might be a configuration issue, not unavailability
                # Don't skip - let it fail with clear error message
                return None
            except importlib_metadata.PackageNotFoundError:
                # CUTLASS package not found - might be truly unavailable
                pass
        except ImportError:
            # CUTLASS not installed - might be truly unavailable
            pass
        # Only skip if we're sure CUTLASS is not available
        # For now, don't skip - let the fallback logic handle it
        return None
    
    # CUDA extension failures - should be pre-compiled, not skipped
    if 'cuda extension' in error_lower or 'failed to load/compile' in error_lower:
        # Don't skip - extensions should be pre-compiled before running tests
        return None
    
    # TF32 API mixing - this is a code issue, not a hardware limitation
    if 'mix of the legacy and new apis' in error_lower or 'allow_tf32_new' in error_lower:
        # Don't skip - this should be fixed in arch_config.py
        return None
    
    return None


def discover_cuda_benchmarks(chapter_dir: Path) -> List[Tuple[Path, List[Path], str]]:
    """Wrapper-only mode: CUDA benchmarks are executed via Python wrappers.

    Direct .cu discovery is intentionally disabled so that CUDA binaries run
    only through CudaBinaryBenchmark wrappers (baseline_*.py/optimized_*.py).
    """
    return []


def cuda_binary_requires_multi_gpu(path: Path) -> bool:
    """Best-effort heuristic to detect CUDA binaries that require multi-GPU hardware."""
    name = path.stem.lower()
    multi_gpu_tokens = ("nvlink", "multigpu", "multi_gpu", "multi-gpu", "distributed")
    return any(token in name for token in multi_gpu_tokens)


def determine_cuda_skip_reason(
    cu_file: Path,
    chapter_dir: Path,
    build_success: bool,
    build_warning: Optional[str],
) -> str:
    """Return a best-effort skip reason when a CUDA executable is unavailable."""
    name = cu_file.stem.lower()
    if not build_success:
        detail = build_warning or "CUDA Makefile build failed"
        return f"SKIPPED: CUDA executables were not built ({detail})"
    
    # Implementation-only translation units are wrapped by *_host.cu or *_static.cu files.
    wrapper_candidates = [
        chapter_dir / f"{cu_file.stem}_host.cu",
        chapter_dir / f"{cu_file.stem}_static.cu",
        chapter_dir / f"{cu_file.stem}_host_sm121",
        chapter_dir / f"{cu_file.stem}_static_sm121",
    ]
    if any(candidate.exists() for candidate in wrapper_candidates):
        return (
            f"SKIPPED: {cu_file.name} is included by a host wrapper and is not built as a standalone binary on this platform"
        )
    
    if "tcgen05" in name:
        return (
            "SKIPPED: tcgen05 kernels require Tensor Memory Accelerator instructions that "
            "are unavailable in this CUDA 13.0 toolchain"
        )
    # Check actual hardware capabilities for DSMEM/cluster benchmarks
    if "dsmem" in name or "cluster" in name:
        try:
            from core.harness.hardware_capabilities import detect_capabilities
            cap = detect_capabilities()
            if cap and cap.cluster.supports_clusters and cap.cluster.has_dsmem:
                # Hardware supports it - this is a build/compile issue, not hardware
                return (
                    f"SKIPPED: CUDA executable not built (hardware supports DSMEM/clusters, "
                    f"but binary compilation failed - check Makefile or NVCC errors)"
                )
            # Hardware doesn't support it
            reason = cap.cluster.notes if cap and cap.cluster.notes else "Hardware does not support clusters/DSMEM"
            return f"SKIPPED: {reason}"
        except Exception:
            # Fallback if capability detection fails
            return (
                "SKIPPED: Could not verify cluster/DSMEM support - capability detection failed"
            )
    if "pipeline" in name and "warp_specialized" in name:
        try:
            from core.harness.hardware_capabilities import detect_capabilities
            cap = detect_capabilities()
            if cap and cap.cluster.supports_clusters:
                return (
                    "SKIPPED: Warp specialized pipeline binary not built (hardware supports clusters)"
                )
            return (
                "SKIPPED: Warp specialization cluster pipelines require thread block cluster hardware support"
            )
        except Exception:
            return (
                "SKIPPED: Warp specialization cluster pipelines require thread block cluster hardware support"
            )
    if "dynamic_parallelism" in name and "host" not in name:
        return (
            "SKIPPED: This dynamic parallelism driver is compiled only via the *_host.cu wrapper"
        )
    
    return (
        "SKIPPED: CUDA executable not available on this architecture (implementation-only or omitted from Makefile)"
    )


def find_cuda_executable(cu_file: Path, chapter_dir: Path) -> Optional[Path]:
    """Find the compiled executable for a CUDA source file.
    
    Looks for executables with SM suffixes (e.g., baseline_gemm_sm121) or without suffix.
    Prioritizes the current GPU's compute capability.
    
    Args:
        cu_file: Path to .cu source file
        chapter_dir: Path to chapter directory (for Makefile detection)
        
    Returns:
        Path to executable if found, None otherwise
    """
    base_name = cu_file.stem
    
    # Handle special naming conventions where driver.cu produces different executable name
    # E.g., optimized_warp_specialized_two_pipelines_driver.cu -> optimized_warp_specialized_two_pipelines_multistream
    # Also handle source files with suffixes like _gmem.cu that produce executables without that suffix
    driver_to_executable = {
        "optimized_warp_specialized_two_pipelines_driver": "optimized_warp_specialized_two_pipelines_multistream",
        "baseline_warp_specialized_two_pipelines_driver": "baseline_warp_specialized_two_pipelines_multistream",
    }
    if base_name in driver_to_executable:
        base_name = driver_to_executable[base_name]
    
    # Detect current GPU's SM version and prioritize it
    def _current_sm_suffix() -> str:
        cap = detect_capabilities()
        if cap is not None:
            return f"_sm{cap.compute_capability.replace('.', '')}"
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"_sm{major}{minor}"
        raise RuntimeError("CUDA device unavailable; cannot choose CUDA executable suffix.")

    current_sm = _current_sm_suffix()
    
    # Check SM suffixes with current GPU's SM first
    suffixes = [current_sm, "_sm100", "_sm103", "_sm121", "_sm90", "_sm89", "_sm86", ""]
    # Remove duplicates while preserving order
    seen = set()
    unique_suffixes = []
    for s in suffixes:
        if s not in seen:
            seen.add(s)
            unique_suffixes.append(s)
    
    for suffix in unique_suffixes:
        executable = chapter_dir / f"{base_name}{suffix}"
        if executable.exists() and os.access(executable, os.X_OK):
            return executable
    
    return None


@dataclass
class CudaBenchmarkResult:
    """Statistical results from CUDA executable benchmarking.
    
    Contains both kernel timing (parsed from stdout) and process timing
    (wall-clock time including startup/init) for diagnostics.
    """
    # Kernel timing (parsed from stdout) - this is the primary metric
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    percentiles: Dict[float, float]  # e.g., {25.0: 1.23, 50.0: 1.45, ...}
    iterations: int
    warmup_iterations: int
    skip_reason: Optional[str] = None
    # Process timing (wall-clock) - for diagnostics only
    process_mean_ms: Optional[float] = None
    process_median_ms: Optional[float] = None
    process_min_ms: Optional[float] = None
    process_max_ms: Optional[float] = None


def benchmark_cuda_executable(executable: Path, iterations: int = 3, warmup: int = 1, timeout: int = 30) -> Optional[CudaBenchmarkResult]:
    """Benchmark a CUDA executable and return statistical results.
    
    Parses kernel timing from stdout (e.g., "2.3074 ms") instead of measuring
    wall-clock time, which would include process startup and CUDA driver init.
    
    Uses the shared timing parser from core.benchmark.timing_parser for
    consistent behavior with CudaBinaryBenchmark.
    
    Args:
        executable: Path to CUDA executable
        iterations: Number of benchmark iterations
        warmup: Number of warmup runs (default: 1 to absorb CUDA driver init)
        timeout: Timeout per run in seconds (default: 30 seconds to prevent hangs)
        
    Returns:
        CudaBenchmarkResult with statistical measures (kernel time from stdout
        as primary metric, process wall-clock time as diagnostic), or None if failed
    """
    kernel_times_ms = []  # Parsed from stdout (primary metric)
    process_times_ms = []  # Wall-clock time (for diagnostics)
    skip_reason: Optional[str] = None
    SKIP_EXIT_CODES = {3}
    
    def decode_message(stdout: bytes, stderr: bytes, returncode: int) -> str:
        for stream in (stderr, stdout):
            if stream:
                text = stream.decode('utf-8', errors='ignore').strip()
                if text:
                    return text.splitlines()[0]
        return f"{executable.name} exited with code {returncode}"
    
    def safe_kill_process(process):
        """Safely kill a process and its children."""
        try:
            # Try to kill process group if setsid was used
            if hasattr(os, 'setsid'):
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        process.wait()
                except (ProcessLookupError, OSError, AttributeError):
                    # Fallback to killing just the process
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            else:
                # No setsid support, just kill the process
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except (ProcessLookupError, OSError):
            # Process already terminated
            pass
    
    # Warmup runs (executables already perform their own warmup/averaging internally)
    for _ in range(warmup):
        try:
            # Run executable with timeout protection
            process = subprocess.Popen(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group if available
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode in SKIP_EXIT_CODES:
                    reason = decode_message(stdout, stderr, process.returncode)
                    skip_reason = reason or "Hardware/software limitation detected"
                    logger.warning(f"CUDA executable {executable.name} reported SKIPPED during warmup: {skip_reason}")
                    return CudaBenchmarkResult(
                        mean_ms=0.0,
                        median_ms=0.0,
                        std_ms=0.0,
                        min_ms=0.0,
                        max_ms=0.0,
                        percentiles={},
                        iterations=0,
                        warmup_iterations=warmup,
                        skip_reason=skip_reason,
                    )
                if process.returncode != 0:
                    # Executable failed, skip warmup
                    continue
            except subprocess.TimeoutExpired:
                # Timeout occurred - kill the process
                safe_kill_process(process)
                logger.warning(f"CUDA executable {executable.name} timed out during warmup (>{timeout}s)")
                reset_gpu_via_script(f"{executable.name} warmup timeout")
                return None
        except Exception as e:
            # If process creation failed, return None
            logger.warning(f"Failed to run CUDA executable {executable.name}: {e}")
            return None
    
    # Benchmark runs
    # Ensure CUDA libraries are in library path
    env = os.environ.copy()
    cuda_lib_paths = [
        '/usr/lib/aarch64-linux-gnu',  # libcuda.so.1 location
        '/usr/local/cuda-13.0/lib64',
        '/usr/local/cuda-13.0/targets/sbsa-linux/lib',
    ]
    existing_ld_path = env.get('LD_LIBRARY_PATH', '')
    new_ld_path = ':'.join(cuda_lib_paths + ([existing_ld_path] if existing_ld_path else []))
    env['LD_LIBRARY_PATH'] = new_ld_path
    
    for i in range(iterations):
        try:
            start = time.perf_counter()
            # Run executable with timeout protection
            process = subprocess.Popen(
                [str(executable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group if available
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                end = time.perf_counter()
                wall_clock_ms = (end - start) * 1000.0
                
                if process.returncode == 0:
                    # Track process time for diagnostics
                    process_times_ms.append(wall_clock_ms)
                    
                    # Parse kernel timing from stdout
                    stdout_text = stdout.decode('utf-8', errors='ignore')
                    parsed_time_ms = parse_kernel_time_ms(stdout_text)
                    
                    if parsed_time_ms is None:
                        logger.error(
                            f"CUDA executable {executable.name}: could not parse timing from stdout. "
                            f"stdout: {stdout_text[:500]}"
                        )
                        return None
                    
                    kernel_times_ms.append(parsed_time_ms)
                elif process.returncode in SKIP_EXIT_CODES:
                    skip_reason = decode_message(stdout, stderr, process.returncode)
                    logger.warning(
                        f"CUDA executable {executable.name} reported SKIPPED on iteration {i+1}: {skip_reason}"
                    )
                    break
                else:
                    # Executable failed, log but continue
                    logger.warning(f"CUDA executable {executable.name} failed with return code {process.returncode} on iteration {i+1}")
                    if stderr:
                        logger.warning(f"  stderr: {stderr.decode('utf-8', errors='ignore')[:200]}")
            except subprocess.TimeoutExpired:
                # Timeout occurred - kill the process
                safe_kill_process(process)
                logger.warning(f"CUDA executable {executable.name} timed out on iteration {i+1} (>{timeout}s)")
                reset_gpu_via_script(f"{executable.name} measurement timeout")
                return None
        except Exception as e:
            # If process creation failed, log and return None
            logger.warning(f"Failed to run CUDA executable {executable.name} on iteration {i+1}: {e}")
            return None
    
    if skip_reason:
        return CudaBenchmarkResult(
            mean_ms=0.0,
            median_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            percentiles={},
            iterations=0,
            warmup_iterations=warmup,
            skip_reason=skip_reason,
        )
    
    if not kernel_times_ms:
        return None
    
    # Compute statistics for kernel times (primary metric)
    sorted_kernel_times = sorted(kernel_times_ms)
    n = len(sorted_kernel_times)
    
    # Compute percentiles (same as BenchmarkHarness)
    # Use float keys to match how they're accessed (99.0, 75.0, etc.)
    percentiles_to_compute = [25.0, 50.0, 75.0, 90.0, 99.0]
    percentiles_dict = {}
    for p in percentiles_to_compute:
        idx = int((p / 100.0) * (n - 1))
        idx = min(idx, n - 1)
        percentiles_dict[p] = sorted_kernel_times[idx]
    
    # Compute process time statistics (for diagnostics)
    process_mean = statistics.mean(process_times_ms) if process_times_ms else None
    process_median = statistics.median(process_times_ms) if process_times_ms else None
    process_min = min(process_times_ms) if process_times_ms else None
    process_max = max(process_times_ms) if process_times_ms else None
    
    return CudaBenchmarkResult(
        mean_ms=statistics.mean(kernel_times_ms),
        median_ms=statistics.median(kernel_times_ms),
        std_ms=statistics.stdev(kernel_times_ms) if n > 1 else 0.0,
        min_ms=min(kernel_times_ms),
        max_ms=max(kernel_times_ms),
        percentiles=percentiles_dict,
        iterations=n,
        warmup_iterations=warmup,
        process_mean_ms=process_mean,
        process_median_ms=process_median,
        process_min_ms=process_min,
        process_max_ms=process_max,
    )


def check_nsys_available() -> bool:
    """Check if nsys is available on the system."""
    try:
        result = subprocess.run(
            ["nsys", "--version"],
            capture_output=True,
            timeout=5,
            check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ncu_available() -> bool:
    """Check if ncu (NVIDIA Compute Profiler) is available on the system."""
    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            timeout=5,
            check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _is_torchrun_launch(config: Optional[BenchmarkConfig]) -> bool:
    if config is None:
        return False
    launch_via = getattr(config, "launch_via", None)
    if launch_via is None:
        return False
    if hasattr(launch_via, "value"):
        launch_via = launch_via.value
    return str(launch_via).lower() == "torchrun"


def _pick_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _select_single_gpu_visible() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
        return tokens[0] if tokens else "0"
    return "0"


def _resolve_expected_seed(config: Optional[BenchmarkConfig]) -> int:
    if config is not None and getattr(config, "seed", None) is not None:
        return int(config.seed)
    return int(torch.initial_seed())


def _resolve_expected_cuda_seed(config: Optional[BenchmarkConfig]) -> int:
    if config is not None and getattr(config, "seed", None) is not None:
        return int(config.seed)
    return int(torch.cuda.initial_seed())


def _should_profile_launch_torchrun_direct(config: BenchmarkConfig) -> bool:
    """Bypass the external torchrun launcher for single-process profile runs.

    Nsight tools are more reliable when they attach directly to the wrapper
    process instead of a short-lived torchrun parent that immediately spawns a
    single worker and exits after rendezvous setup.
    """
    nproc_per_node = int(getattr(config, "nproc_per_node", None) or torch.cuda.device_count() or 1)
    nnodes = int(getattr(config, "nnodes", None) or 1)
    return nproc_per_node == 1 and nnodes == 1


def _command_uses_external_torchrun(command: Sequence[str]) -> bool:
    if not command:
        return False
    head = os.path.basename(str(command[0]))
    if head == "torchrun":
        return True
    return (
        len(command) >= 3
        and str(command[0]) == sys.executable
        and str(command[1]) == "-m"
        and str(command[2]) == "torch.distributed.run"
    )


def _command_is_direct_torchrun_wrapper(command: Sequence[str]) -> bool:
    return (
        len(command) >= 3
        and str(command[0]) == sys.executable
        and str(command[1]) == "-m"
        and str(command[2]) == "core.harness.torchrun_wrapper"
    )


def _resolve_target_override_argv(config: Optional[BenchmarkConfig]) -> Optional[List[str]]:
    if config is None:
        return None
    target_label = getattr(config, "target_label", None)
    if not target_label:
        return None
    target_overrides = _lookup_target_extra_args(
        (getattr(config, "target_extra_args", {}) or {}),
        target_label,
    )
    if not target_overrides:
        return None
    if isinstance(target_overrides, str):
        return shlex.split(target_overrides)
    return list(target_overrides)


def _is_missing_nsys_artifact_error(last_error: Optional[str]) -> bool:
    if not last_error:
        return False
    normalized = str(last_error).lower()
    return "no report artifact was produced" in normalized


def _retry_nsys_in_clean_helper(
    *,
    output_dir: Path,
    output_name: str,
    target_command: Sequence[str],
    trace_forks: bool,
    profile_preset: str,
    full_timeline: bool,
    timeout: Optional[float],
    wait_mode: str,
    env: Dict[str, str],
) -> Optional[Path]:
    _set_profile_failure_detail("nsys", None)
    owner_run_id = str(os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", "")).strip()
    owner_pid = str(os.environ.get("AISP_BENCHMARK_OWNER_PID", "")).strip()
    payload = {
        "output_dir": str(output_dir),
        "output_name": output_name,
        "command": list(target_command),
        "trace_forks": bool(trace_forks),
        "preset": profile_preset,
        "full_timeline": bool(full_timeline),
        "timeout_seconds": float(timeout) if timeout and timeout > 0 else None,
        "wait_mode": wait_mode,
        "extra_env": env,
        "sanitize_python_startup": True,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as payload_handle:
        payload_handle.write(json.dumps(payload))
        payload_path = Path(payload_handle.name)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as result_handle:
        result_path = Path(result_handle.name)
    helper_env = build_repo_python_env(repo_root, base_env=os.environ.copy())
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "core.profiling.nsys_capture_helper",
                *(
                    ["--aisp-owner-run-id", owner_run_id]
                    if owner_run_id
                    else []
                ),
                *(
                    ["--aisp-owner-pid", owner_pid]
                    if owner_pid
                    else []
                ),
                "--payload",
                str(payload_path),
                "--result",
                str(result_path),
            ],
            cwd=str(repo_root),
            env=helper_env,
            capture_output=True,
            text=True,
            timeout=(float(timeout) if timeout and timeout > 0 else 120.0) + 60.0,
            check=False,
        )
        if proc.returncode != 0 and LOGGER_AVAILABLE:
            logger.warning(
                "  Clean helper NSYS retry exited with code %s. stdout=%s stderr=%s",
                proc.returncode,
                (proc.stdout or "").strip()[-400:],
                (proc.stderr or "").strip()[-400:],
            )
        if not result_path.exists():
            helper_detail = _normalize_failure_detail(
                " | ".join(
                    part
                    for part in ((proc.stderr or "").strip(), (proc.stdout or "").strip())
                    if part and str(part).strip()
                )
            )
            _set_profile_failure_detail("nsys", helper_detail)
            return None
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        report = result_payload.get("report")
        failure_detail = _normalize_failure_detail(result_payload.get("last_error"))
        if not failure_detail:
            failure_detail = _normalize_failure_detail(
                " | ".join(
                    part
                    for part in ((proc.stderr or "").strip(), (proc.stdout or "").strip())
                    if part and str(part).strip()
                )
            )
        if not report:
            _set_profile_failure_detail("nsys", failure_detail)
            return None
        report_path = Path(report)
        if report_path.exists():
            _set_profile_failure_detail("nsys", None)
            return report_path
        _set_profile_failure_detail("nsys", failure_detail)
        return None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        _set_profile_failure_detail("nsys", str(exc))
        if LOGGER_AVAILABLE:
            logger.warning("  Clean helper NSYS retry failed: %s", exc, exc_info=True)
        return None
    finally:
        payload_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def _build_torchrun_profile_command(
    config: BenchmarkConfig,
    wrapper_script_path: Optional[str] = None,
    *,
    spec: Optional[TorchrunLaunchSpec] = None,
) -> Tuple[List[str], Dict[str, str]]:
    if spec is None and not wrapper_script_path:
        raise ValueError("Torchrun profile launch requires wrapper_script_path or spec")
    nproc_per_node = getattr(config, "nproc_per_node", None) or torch.cuda.device_count() or 1
    nnodes = int(getattr(config, "nnodes", None) or 1)
    use_direct_wrapper = _should_profile_launch_torchrun_direct(config)
    if use_direct_wrapper:
        torchrun_cmd = [sys.executable]
    else:
        torchrun_path = shutil.which("torchrun")
        if torchrun_path:
            torchrun_cmd = [
                torchrun_path,
                "--nproc_per_node",
                str(nproc_per_node),
            ]
        else:
            torchrun_cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(nproc_per_node),
            ]
        if getattr(config, "nnodes", None):
            torchrun_cmd.extend(["--nnodes", str(config.nnodes)])

        rdzv_endpoint = getattr(config, "rdzv_endpoint", None)
        if not rdzv_endpoint:
            rdzv_endpoint = f"127.0.0.1:{_pick_free_port()}"
        rdzv_backend = getattr(config, "rdzv_backend", None) or "c10d"
        torchrun_cmd.extend(["--rdzv_backend", str(rdzv_backend), "--rdzv_endpoint", str(rdzv_endpoint)])

    expected_seed = _resolve_expected_seed(config)
    if spec is None:
        wrapper_args: List[str] = [
            "--aisp-target-script",
            str(wrapper_script_path),
            "--aisp-expected-torch-seed",
            str(expected_seed),
        ]
        script_args: List[str] = []
        spec_env: Dict[str, str] = {}
    else:
        script_path = Path(spec.script_path).resolve() if spec.script_path is not None else None
        module_name = spec.module_name
        if bool(script_path) == bool(module_name):
            raise RuntimeError("Torchrun profiling spec requires exactly one of script_path or module_name.")
        wrapper_args = ["--aisp-expected-torch-seed", str(expected_seed)]
        if script_path is not None:
            wrapper_args[0:0] = ["--aisp-target-script", str(script_path)]
        else:
            wrapper_args[0:0] = ["--aisp-target-module", str(module_name)]

        def _config_args_from_map() -> List[str]:
            args: List[str] = []
            for key, flag in spec.config_arg_map.items():
                if not flag or not hasattr(config, key):
                    continue
                value = getattr(config, key)
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.extend([flag, str(value)])
            return args

        extra_args: List[str] = []
        target_label = getattr(config, "target_label", None)
        if target_label:
            target_overrides = _lookup_target_extra_args(
                (getattr(config, "target_extra_args", {}) or {}),
                target_label,
            )
            if target_overrides:
                if isinstance(target_overrides, str):
                    extra_args.extend(shlex.split(target_overrides))
                else:
                    extra_args.extend(list(target_overrides))
        script_args = list(spec.script_args)
        script_args.extend(_config_args_from_map())
        script_args.extend(extra_args)
        spec_env = dict(spec.env)

    if getattr(config, "deterministic", False):
        wrapper_args.append("--aisp-deterministic")
    if torch.cuda.is_available():
        wrapper_args.extend(["--aisp-expected-cuda-seed", str(_resolve_expected_cuda_seed(config))])

    if use_direct_wrapper:
        torchrun_cmd.extend(["-m", "core.harness.torchrun_wrapper", *wrapper_args, *script_args])
    else:
        torchrun_cmd.extend(["-m", "core.harness.torchrun_wrapper", *wrapper_args])
        if spec is not None:
            torchrun_cmd.extend(script_args)

    env = build_repo_python_env(repo_root, base_env=os.environ.copy())
    if getattr(config, "single_gpu", False):
        env["CUDA_VISIBLE_DEVICES"] = _select_single_gpu_visible()
    if getattr(config, "lock_gpu_clocks", False) and torch.cuda.is_available():
        env["AISP_LOCK_GPU_CLOCKS"] = "1"
        if getattr(config, "gpu_sm_clock_mhz", None) is not None:
            env["AISP_GPU_SM_CLOCK_MHZ"] = str(config.gpu_sm_clock_mhz)
        if getattr(config, "gpu_mem_clock_mhz", None) is not None:
            env["AISP_GPU_MEM_CLOCK_MHZ"] = str(config.gpu_mem_clock_mhz)
        env["AISP_RAMP_GPU_CLOCKS"] = "1"
    if spec is not None:
        env.update(spec_env)
    if use_direct_wrapper:
        env.setdefault("RANK", "0")
        env.setdefault("LOCAL_RANK", "0")
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("LOCAL_WORLD_SIZE", "1")
        env.setdefault("GROUP_RANK", "0")
        env.setdefault("ROLE_RANK", "0")
        env.setdefault("ROLE_NAME", "default")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")

    return torchrun_cmd, env


def _resolve_profile_torchrun_spec(
    benchmark: Any,
    *,
    profiler: str,
    config: Optional[BenchmarkConfig],
    output_path: Optional[Path] = None,
) -> Optional[TorchrunLaunchSpec]:
    getter = getattr(benchmark, "get_profile_torchrun_spec", None)
    if callable(getter):
        spec = getter(profiler=profiler, config=config, output_path=output_path)
        if spec is not None:
            return spec
    if profiler in {"nsys", "ncu"}:
        base_getter = getattr(benchmark, "get_torchrun_spec", None)
        if callable(base_getter):
            return base_getter(config)
    return None


def _harden_profile_env(
    base_env: Optional[Dict[str, str]],
    repo_root: Path,
    chapter_dir: Optional[Path] = None,
) -> Dict[str, str]:
    startup_stub_dir = Path(tempfile.gettempdir()) / "aisp_profile_python_startup"
    startup_stub_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in {
        "sitecustomize.py": (
            '"""AISP profiling startup shim.\n'
            "\n"
            "Overrides host-level sitecustomize side effects during profiler launches.\n"
            '"""\n'
        ),
        "usercustomize.py": (
            '"""AISP profiling startup shim.\n'
            "\n"
            "Shadows incompatible host-level usercustomize hooks during profiler launches.\n"
            '"""\n'
        ),
    }.items():
        stub_path = startup_stub_dir / filename
        if not stub_path.exists() or stub_path.read_text() != contents:
            stub_path.write_text(contents)

    env = dict(base_env or os.environ.copy())
    owner_run_id = str(os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", "")).strip()
    owner_pid = str(os.environ.get("AISP_BENCHMARK_OWNER_PID", "")).strip()
    if owner_run_id:
        env.setdefault("AISP_BENCHMARK_OWNER_RUN_ID", owner_run_id)
    if owner_pid:
        env.setdefault("AISP_BENCHMARK_OWNER_PID", owner_pid)
    force_no_user_site = str(env.get("AISP_PROFILE_NO_USER_SITE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Default to user-site enabled so profiling subprocesses resolve the same
    # pinned serving stack as timing runs (torch/vllm/msgspec/etc.).
    include_user_site = True
    include_user_site_raw = str(env.get("AISP_PROFILE_INCLUDE_USER_SITE", "")).strip().lower()
    if include_user_site_raw:
        include_user_site = include_user_site_raw in {"1", "true", "yes", "on"}
    if force_no_user_site:
        include_user_site = False
    pythonpath_entries = [str(startup_stub_dir), str(repo_root)]
    if chapter_dir is not None:
        pythonpath_entries.append(str(chapter_dir))
    user_site: Optional[str] = None
    try:
        import site
        user_site = site.getusersitepackages()
        # Match default interpreter import precedence used by benchmark timing paths:
        # user-site packages precede dist-packages unless explicitly disabled.
        if include_user_site and user_site:
            pythonpath_entries.append(user_site)
        for site_path in site.getsitepackages():
            if site_path:
                pythonpath_entries.append(site_path)
    except Exception as exc:
        _emit_run_benchmark_warning(
            "Failed to discover Python site-packages while hardening profiler environment",
            exc=exc,
        )
    existing = env.get("PYTHONPATH")
    if existing:
        for entry in existing.split(os.pathsep):
            if not entry:
                continue
            if (
                not include_user_site
                and user_site
                and os.path.normpath(entry) == os.path.normpath(user_site)
            ):
                continue
            pythonpath_entries.append(entry)
    deduped: List[str] = []
    seen = set()
    for entry in pythonpath_entries:
        if not entry or entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    if include_user_site:
        env.pop("PYTHONNOUSERSITE", None)
    else:
        env["PYTHONNOUSERSITE"] = "1"
    # Avoid expensive/fragile addr2line symbolization in profiler subprocesses.
    # PyTorch itself recommends this when Module.cpp symbolization warnings appear.
    env.setdefault("TORCH_DISABLE_ADDR2LINE", "1")
    return env


@contextmanager
def _temporary_python_profile_launch(
    wrapper_source: str,
    *,
    chapter_dir: Path,
    repo_root: Path,
    config: Optional[BenchmarkConfig],
    benchmark: Any,
    allow_torchrun: bool = True,
) -> Iterator[Tuple[Path, List[str], Dict[str, str], bool]]:
    with temporary_python_profile_wrapper(wrapper_source) as wrapper_path:
        use_torchrun = bool(allow_torchrun and _is_torchrun_launch(config))
        if use_torchrun and config is not None:
            command, base_env = _build_torchrun_profile_command(config, str(wrapper_path))
        else:
            command = [sys.executable, str(wrapper_path)]
            base_env = None
        env = _apply_profile_env_overrides(
            _harden_profile_env(base_env, repo_root=repo_root, chapter_dir=chapter_dir),
            config=config,
            benchmark=benchmark,
        )
        yield wrapper_path, command, env, use_torchrun


def _apply_profile_env_overrides(
    env: Optional[Dict[str, str]],
    *,
    config: Optional[BenchmarkConfig] = None,
    benchmark: Optional[Any] = None,
) -> Dict[str, str]:
    """Apply benchmark-local profiler env overrides after generic hardening."""
    merged = dict(env or {})
    for source in (config, benchmark):
        if source is None:
            continue
        overrides = getattr(source, "profile_env_overrides", None)
        if not overrides:
            continue
        merged.update({str(key): str(value) for key, value in overrides.items()})
    return merged


def _profile_bench_dir(profile_root: Path, chapter_id: str) -> Path:
    safe_chapter = slugify(str(chapter_id).replace("/", "_").replace("\\", "_"))
    return profile_root / "bench" / safe_chapter


def _profile_example_dir(profile_root: Path, chapter_id: str, example_name: str) -> Path:
    safe_example = slugify(example_name)
    return _profile_bench_dir(profile_root, chapter_id) / safe_example


def _profile_pair_dir(
    profile_root: Path,
    chapter_id: str,
    example_name: str,
    pair_key: str,
) -> Path:
    safe_pair = slugify(pair_key or "default")
    return _profile_example_dir(profile_root, chapter_id, example_name) / f"pair__{safe_pair}"


def _repo_relative_path(path: Path | str, repo_root: Path) -> str:
    candidate = Path(path)
    root = repo_root.resolve()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return str(candidate)


def _resolve_profile_output_dir(output_dir: Path | str) -> Path:
    return Path(output_dir).resolve()


def profile_python_benchmark(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    config: Optional[BenchmarkConfig] = None,
    variant: str = "baseline",
    output_stem: Optional[str] = None,
) -> Optional[Path]:
    """Profile a Python benchmark using nsys by wrapping benchmark execution.
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save nsys-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated nsys-rep file, or None if failed
    """
    if not check_nsys_available():
        return None
    _set_profile_failure_detail("nsys", None)

    output_dir = _resolve_profile_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = output_stem or benchmark_path.stem
    
    bench_config = config
    if bench_config is None and hasattr(benchmark, "get_config"):
        bench_config = benchmark.get_config()
    profiling_view = bench_config.profiling if bench_config else None
    validity_view = bench_config.validity if bench_config else None
    timing_view = bench_config.timing if bench_config else None
    nvtx_includes = profiling_view.nsys_nvtx_include if profiling_view else None
    validity_profile = validity_view.validity_profile if validity_view else "strict"
    lock_gpu_clocks_flag = validity_view.lock_gpu_clocks if validity_view else False
    if validity_profile != "strict":
        lock_gpu_clocks_flag = False
    gpu_sm_clock_mhz = validity_view.gpu_sm_clock_mhz if validity_view else None
    gpu_mem_clock_mhz = validity_view.gpu_mem_clock_mhz if validity_view else None
    target_label = getattr(bench_config, "target_label", None) if bench_config else None
    target_override_argv = _resolve_target_override_argv(bench_config)
    # chapter_dir points to e.g. <repo>/ch10 or <repo>/labs/<lab>; use global
    # repository root for package imports like `labs.*` and `core.*`.
    repo_root = Path(__file__).resolve().parents[2]

    try:
        from core.profiling.nsight_automation import NsightAutomation
        torchrun_profile_spec = None
        target_command: List[str]
        env: Dict[str, str]
        use_torchrun = bool(_is_torchrun_launch(bench_config))
        with ExitStack() as stack:
            if use_torchrun:
                torchrun_profile_spec = _resolve_profile_torchrun_spec(
                    benchmark,
                    profiler="nsys",
                    config=bench_config,
                )
            if torchrun_profile_spec is not None and bench_config is not None:
                target_command, base_env = _build_torchrun_profile_command(
                    bench_config,
                    spec=torchrun_profile_spec,
                )
                env = _apply_profile_env_overrides(
                    _harden_profile_env(base_env, repo_root=repo_root, chapter_dir=chapter_dir),
                    config=bench_config,
                    benchmark=benchmark,
                )
            else:
                wrapper_source = render_nsys_python_profile_wrapper(
                    benchmark_path=benchmark_path,
                    nvtx_includes=nvtx_includes,
                    target_label=target_label,
                    target_override_argv=target_override_argv,
                    validity_profile=validity_profile,
                    lock_gpu_clocks_flag=lock_gpu_clocks_flag,
                    gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                    gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                )
                _wrapper_path, target_command, env, use_torchrun = stack.enter_context(
                    _temporary_python_profile_launch(
                        wrapper_source,
                        chapter_dir=chapter_dir,
                        repo_root=repo_root,
                        config=bench_config,
                        benchmark=benchmark,
                    )
                )
            profile_preset, full_timeline = _resolve_nsys_profile_mode(bench_config)
            trace_forks = _command_uses_external_torchrun(target_command)
            wait_mode = "all" if trace_forks else "primary"
            timeout = timing_view.timeout_for("nsys") if timing_view else None
            timeout = timeout or 120
            automation = NsightAutomation(output_dir)

            def _run_nsys_capture() -> Optional[Path]:
                report = automation.profile_nsys(
                    command=target_command,
                    output_name=f"{benchmark_name}__{variant}",
                    trace_cuda=True,
                    trace_nvtx=True,
                    trace_osrt=True,
                    full_timeline=full_timeline,
                    trace_forks=trace_forks,
                    preset=profile_preset,
                    timeout_seconds=float(timeout) if timeout and timeout > 0 else None,
                    wait_mode=wait_mode,
                    finalize_grace_seconds=20.0,
                    force_lineinfo=True,
                    extra_env=env,
                    sanitize_python_startup=True,
                )
                if report and Path(report).exists():
                    return Path(report)
                return None

            nsys_report = _run_nsys_capture()
            missing_artifact = _is_missing_nsys_artifact_error(getattr(automation, "last_error", None))
            if (
                nsys_report is None
                and not trace_forks
                and missing_artifact
            ):
                if LOGGER_AVAILABLE:
                    logger.warning(
                        "  Retrying Nsight Systems capture once for %s (%s) after Nsight exited successfully but did not materialize a report artifact on the single-process capture path.",
                        benchmark_name,
                        variant,
                    )
                time.sleep(1.0)
                nsys_report = _run_nsys_capture()
                if nsys_report is None:
                    if LOGGER_AVAILABLE:
                        logger.warning(
                            "  Retrying Nsight Systems capture from a clean helper process for %s (%s) after repeated missing artifacts on the single-process capture path.",
                            benchmark_name,
                            variant,
                        )
                    nsys_report = _retry_nsys_in_clean_helper(
                        output_dir=output_dir,
                        output_name=f"{benchmark_name}__{variant}",
                        target_command=target_command,
                        trace_forks=trace_forks,
                        profile_preset=profile_preset,
                        full_timeline=full_timeline,
                        timeout=float(timeout) if timeout and timeout > 0 else None,
                        wait_mode=wait_mode,
                        env=env,
                    )
            if nsys_report and Path(nsys_report).exists():
                _set_profile_failure_detail("nsys", None)
                return Path(nsys_report)
            _set_profile_failure_detail("nsys", getattr(automation, "last_error", None))
            return None
    except Exception as exc:
        _set_profile_failure_detail("nsys", str(exc))
        if LOGGER_AVAILABLE:
            logger.warning(
                "  Nsight Systems profiling failed for %s (%s): %s",
                benchmark_name,
                variant,
                exc,
                exc_info=True,
            )
        return None


def _resolve_nsys_profile_mode(config: Optional[BenchmarkConfig]) -> tuple[str, bool]:
    preset = str(getattr(config, "profile_type", None) or "minimal").lower()
    profile_preset = str(getattr(config, "nsys_preset_override", None) or "").strip().lower()
    if not profile_preset:
        profile_preset = "light" if preset == "minimal" else "full"
    return profile_preset, profile_preset == "full"


def profile_cuda_executable(
    executable: Path,
    chapter_dir: Path,
    output_dir: Path,
    config: Optional[BenchmarkConfig] = None,
    benchmark: Optional[Any] = None,
    variant: str = "baseline",
    output_stem: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> Optional[Path]:
    """Profile a CUDA executable using nsys.
    
    Args:
        executable: Path to CUDA executable
        chapter_dir: Path to chapter directory
        output_dir: Directory to save nsys-rep file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated nsys-rep file, or None if failed
    """
    if not check_nsys_available():
        return None
    _set_profile_failure_detail("nsys", None)

    output_dir = _resolve_profile_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_name = output_stem or executable.stem
    
    try:
        from core.profiling.nsight_automation import NsightAutomation

        automation = NsightAutomation(output_dir)
        target_env = _apply_profile_env_overrides(
            _harden_profile_env(
                None,
                repo_root=Path(__file__).resolve().parents[2],
                chapter_dir=chapter_dir,
            ),
            config=config,
            benchmark=benchmark,
        )

        def _run_nsys_capture() -> Optional[Path]:
            report = automation.profile_nsys(
                command=[str(executable)],
                output_name=f"{exec_name}__{variant}",
                trace_cuda=True,
                trace_nvtx=True,
                trace_osrt=True,
                full_timeline=False,
                trace_forks=False,
                preset="light",
                timeout_seconds=float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else 120.0,
                wait_mode="primary",
                finalize_grace_seconds=20.0,
                force_lineinfo=True,
                extra_env=target_env,
                sanitize_python_startup=True,
            )
            if report and Path(report).exists():
                return Path(report)
            return None

        nsys_report = _run_nsys_capture()
        missing_artifact = _is_missing_nsys_artifact_error(getattr(automation, "last_error", None))
        if nsys_report is None and missing_artifact:
            if LOGGER_AVAILABLE:
                logger.warning(
                    "  Retrying Nsight Systems capture once for %s (%s) after Nsight exited successfully but did not materialize a report artifact on the direct CUDA executable capture path.",
                    exec_name,
                    variant,
                )
            time.sleep(1.0)
            nsys_report = _run_nsys_capture()
            if nsys_report is None:
                if LOGGER_AVAILABLE:
                    logger.warning(
                        "  Retrying Nsight Systems capture from a clean helper process for %s (%s) after repeated missing artifacts on the direct CUDA executable capture path.",
                        exec_name,
                        variant,
                    )
                nsys_report = _retry_nsys_in_clean_helper(
                    output_dir=output_dir,
                    output_name=f"{exec_name}__{variant}",
                    target_command=[str(executable)],
                    trace_forks=False,
                    profile_preset="light",
                    full_timeline=False,
                    timeout=float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else 120.0,
                    wait_mode="primary",
                    env=target_env,
                )
        if nsys_report and Path(nsys_report).exists():
            _set_profile_failure_detail("nsys", None)
            return Path(nsys_report)
        _set_profile_failure_detail("nsys", getattr(automation, "last_error", None))
        return None
    except (subprocess.TimeoutExpired, Exception) as exc:
        _set_profile_failure_detail("nsys", str(exc))
        return None


def _terminate_process_group(process: subprocess.Popen, reason: str, timeout_seconds: Optional[float] = None) -> None:
    """Best-effort kill of a process group (and children) started with start_new_session."""
    cleanup_errors: List[str] = []
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)
    except (ProcessLookupError, OSError, AttributeError) as exc:
        cleanup_errors.append(f"process-group cleanup unavailable: {exc}")
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception as terminate_exc:
            cleanup_errors.append(f"process.terminate() fallback failed: {terminate_exc}")
            try:
                process.kill()
            except Exception as kill_exc:
                cleanup_errors.append(f"process.kill() fallback failed: {kill_exc}")
    if LOGGER_AVAILABLE:
        if timeout_seconds is not None:
            logger.warning("  NCU profiling timed out after %.1fs (%s); killed process group", timeout_seconds, reason)
        else:
            logger.warning("  NCU profiling cleanup triggered (%s); killed process group", reason)
        for detail in cleanup_errors:
            logger.warning("  Profiler cleanup detail (%s): %s", reason, detail)


def _collect_process_group_members(
    process_group_id: int,
    *,
    proc_root: Path = Path("/proc"),
) -> List[int]:
    members: List[int] = []
    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return members

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit():
            continue
        try:
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
        except Exception:
            continue
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            continue
        tail = stat_text[close_paren + 1 :].strip().split()
        if len(tail) < 3:
            continue
        try:
            pid = int(proc_dir.name)
            group_id = int(tail[2])
        except ValueError:
            continue
        if group_id == int(process_group_id):
            members.append(pid)
    members.sort()
    return members


def _reap_process_group_members(
    process_group_id: Optional[int],
    *,
    reason: str,
    preserve_pids: Optional[Set[int]] = None,
    grace_seconds: float = 2.0,
    proc_root: Path = Path("/proc"),
) -> int:
    if process_group_id is None or int(process_group_id) <= 0:
        return 0
    preserved = {int(pid) for pid in (preserve_pids or set()) if int(pid) > 0}
    members = [
        pid
        for pid in _collect_process_group_members(int(process_group_id), proc_root=proc_root)
        if pid not in preserved and pid != os.getpid()
    ]
    if not members:
        return 0

    for pid in members:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    deadline = time.monotonic() + max(0.0, grace_seconds)
    remaining = members
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in remaining if (proc_root / str(pid)).exists()]

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    if LOGGER_AVAILABLE:
        logger.warning(
            "  Reaped %d lingering profiler process-group member(s) after %s",
            len(members),
            reason,
        )
    return len(members)


@dataclass
class _ProfileSubprocessResult:
    process: subprocess.Popen[str]
    stdout_log: Path
    stderr_log: Path
    timed_out: bool
    failure_warning: Optional[str]


def _run_profile_subprocess(
    *,
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_seconds: float,
    log_base: Path,
    terminate_reason: str,
    capture_output: bool,
    timeout_collect_error_message: str,
    wait_error_message: str,
) -> _ProfileSubprocessResult:
    stdout_log = log_base.with_suffix(".stdout.log")
    stderr_log = log_base.with_suffix(".stderr.log")
    process_group_id: Optional[int] = None
    popen_kwargs = dict(
        cwd=str(cwd),
        text=True,
        start_new_session=True,
        env=env,
    )
    if capture_output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        process_group_id = process.pid
    else:
        with stdout_log.open("w") as stdout_handle, stderr_log.open("w") as stderr_handle:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                **popen_kwargs,
            )
            process_group_id = process.pid

    stdout_text = ""
    stderr_text = ""
    timed_out = False
    failure_warning: Optional[str] = None
    try:
        if capture_output:
            stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
        else:
            process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process, terminate_reason, timeout_seconds=timeout_seconds)
        try:
            if capture_output:
                stdout_text, stderr_text = process.communicate(timeout=2)
            else:
                process.wait(timeout=2)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            failure_warning = f"{timeout_collect_error_message}: {exc}"
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        failure_warning = f"{wait_error_message}: {exc}"
        _terminate_process_group(process, terminate_reason)

    if capture_output:
        if stdout_text:
            stdout_log.write_text(stdout_text)
        if stderr_text:
            stderr_log.write_text(stderr_text)
    if not timed_out and failure_warning is None:
        _reap_process_group_members(
            process_group_id,
            reason=terminate_reason,
            preserve_pids={process.pid},
        )
    log_base.with_suffix(".command.json").write_text(json.dumps({"command": command}, indent=2))
    return _ProfileSubprocessResult(
        process=process,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        timed_out=timed_out,
        failure_warning=failure_warning,
    )


def _collect_descendant_pids(root_pid: int, *, proc_root: Path = Path("/proc")) -> List[int]:
    """Collect descendant PIDs for the current run process."""
    parent_to_children: Dict[int, Set[int]] = {}
    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return []

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit():
            continue
        try:
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
        except Exception:
            continue
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            continue
        tail = stat_text[close_paren + 1 :].strip().split()
        if len(tail) < 2:
            continue
        try:
            pid = int(proc_dir.name)
            ppid = int(tail[1])
        except ValueError:
            continue
        parent_to_children.setdefault(ppid, set()).add(pid)

    descendants: List[int] = []
    stack = [int(root_pid)]
    seen = {int(root_pid)}
    while stack:
        parent = stack.pop()
        for child in parent_to_children.get(parent, set()):
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            stack.append(child)
    return descendants


def _safe_read_proc_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


def _read_proc_environ(pid: int, *, proc_root: Path = Path("/proc")) -> Dict[str, str]:
    env: Dict[str, str] = {}
    raw = _safe_read_proc_bytes(proc_root / str(int(pid)) / "environ")
    if not raw:
        return env
    for entry in raw.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        try:
            env[key.decode("utf-8", errors="ignore")] = value.decode("utf-8", errors="ignore")
        except Exception:
            continue
    return env


def _read_proc_cmdline(pid: int, *, proc_root: Path = Path("/proc")) -> str:
    raw = _safe_read_proc_bytes(proc_root / str(int(pid)) / "cmdline")
    if not raw:
        return ""
    return " ".join(part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part)


def _read_proc_cmdline_arg_value(
    pid: int,
    flag: str,
    *,
    proc_root: Path = Path("/proc"),
) -> Optional[str]:
    cmdline = _read_proc_cmdline(pid, proc_root=proc_root)
    if not cmdline:
        return None
    tokens = [token for token in shlex.split(cmdline) if token]
    prefix = f"{flag}="
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _read_proc_owner_marker(
    pid: int,
    *,
    env_key: str,
    cmdline_flag: str,
    proc_root: Path = Path("/proc"),
) -> Optional[str]:
    env = _read_proc_environ(pid, proc_root=proc_root)
    value = str(env.get(env_key, "")).strip()
    if value:
        return value
    cmdline_value = _read_proc_cmdline_arg_value(
        pid,
        cmdline_flag,
        proc_root=proc_root,
    )
    if cmdline_value is None:
        return None
    stripped = str(cmdline_value).strip()
    return stripped or None


def _read_proc_parent_pid(pid: int, *, proc_root: Path = Path("/proc")) -> Optional[int]:
    try:
        stat_text = (proc_root / str(int(pid)) / "stat").read_text(encoding="utf-8")
    except Exception:
        return None
    close_paren = stat_text.rfind(")")
    if close_paren < 0:
        return None
    tail = stat_text[close_paren + 1 :].strip().split()
    if len(tail) < 2:
        return None
    try:
        return int(tail[1])
    except ValueError:
        return None


def _pid_exists(pid: Optional[int], *, proc_root: Path = Path("/proc")) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    return (proc_root / str(int(pid))).exists()


def _collect_current_run_benchmark_orphan_pids(
    *,
    current_run_id: str,
    current_owner_pid: int,
    repo_root: Path,
    proc_root: Path = Path("/proc"),
) -> List[int]:
    """Find detached benchmark-owned processes from the current run."""
    run_id = str(current_run_id).strip()
    owner_pid = int(current_owner_pid)
    owner_pid_marker = str(owner_pid)
    if not run_id or owner_pid <= 0:
        return []

    related_pids = set(_collect_descendant_pids(owner_pid, proc_root=proc_root))
    related_pids.update(_collect_process_lineage_pids(owner_pid, proc_root=proc_root))
    related_pids.add(owner_pid)

    repo_root_str = str(repo_root.resolve())
    current_orphans: List[int] = []
    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return current_orphans

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid in related_pids:
            continue

        env = _read_proc_environ(pid, proc_root=proc_root)
        owner_run_id = _read_proc_owner_marker(
            pid,
            env_key="AISP_BENCHMARK_OWNER_RUN_ID",
            cmdline_flag="--aisp-owner-run-id",
            proc_root=proc_root,
        )
        if owner_run_id != run_id:
            continue
        owner_pid_value = _read_proc_owner_marker(
            pid,
            env_key="AISP_BENCHMARK_OWNER_PID",
            cmdline_flag="--aisp-owner-pid",
            proc_root=proc_root,
        )
        if owner_pid_value and owner_pid_value != owner_pid_marker:
            continue

        cmdline = _read_proc_cmdline(pid, proc_root=proc_root)
        env_context = " ".join(
            value
            for value in (
                env.get("PWD", ""),
                env.get("PYTHONPATH", ""),
            )
            if value
        )
        if repo_root_str not in cmdline and repo_root_str not in env_context:
            continue
        current_orphans.append(pid)

    current_orphans.sort()
    return current_orphans


def _collect_stale_benchmark_orphan_pids(
    *,
    current_run_id: str,
    repo_root: Path,
    proc_root: Path = Path("/proc"),
) -> List[int]:
    """Find leaked benchmark-owned children from older runs that became orphaned."""
    repo_root_str = str(repo_root.resolve())
    stale_pids: List[int] = []
    try:
        proc_entries = list(proc_root.iterdir())
    except Exception:
        return stale_pids

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        env = _read_proc_environ(pid, proc_root=proc_root)
        owner_run_id = _read_proc_owner_marker(
            pid,
            env_key="AISP_BENCHMARK_OWNER_RUN_ID",
            cmdline_flag="--aisp-owner-run-id",
            proc_root=proc_root,
        ) or ""
        if not owner_run_id or owner_run_id == str(current_run_id).strip():
            continue

        parent_pid = _read_proc_parent_pid(pid, proc_root=proc_root)
        if parent_pid not in (1, None) and _pid_exists(parent_pid, proc_root=proc_root):
            continue

        cmdline = _read_proc_cmdline(pid, proc_root=proc_root)
        env_context = " ".join(
            value
            for value in (
                env.get("PWD", ""),
                env.get("PYTHONPATH", ""),
            )
            if value
        )
        if repo_root_str not in cmdline and repo_root_str not in env_context:
            continue
        stale_pids.append(pid)

    stale_pids.sort()
    return stale_pids


def _reap_stale_benchmark_orphans(
    reason: str,
    *,
    current_run_id: str,
    repo_root: Path,
    grace_seconds: float = 2.0,
) -> int:
    """Kill orphaned benchmark-owned processes left behind by older interrupted runs."""
    stale_pids = _collect_stale_benchmark_orphan_pids(
        current_run_id=current_run_id,
        repo_root=repo_root,
    )
    if not stale_pids:
        return 0

    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    deadline = time.monotonic() + max(0.0, grace_seconds)
    remaining = stale_pids
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in remaining if Path(f"/proc/{pid}").exists()]

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    if LOGGER_AVAILABLE:
        logger.warning(
            "  Reaped %d orphaned benchmark process(es) from older run(s) before %s",
            len(stale_pids),
            reason,
        )
    return len(stale_pids)


def _reap_current_run_benchmark_orphans(
    reason: str,
    *,
    current_run_id: str,
    current_owner_pid: int,
    repo_root: Path,
    grace_seconds: float = 2.0,
) -> int:
    """Kill detached benchmark-owned processes from the current run."""
    current_orphans = _collect_current_run_benchmark_orphan_pids(
        current_run_id=current_run_id,
        current_owner_pid=current_owner_pid,
        repo_root=repo_root,
    )
    if not current_orphans:
        return 0

    for pid in current_orphans:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    deadline = time.monotonic() + max(0.0, grace_seconds)
    remaining = current_orphans
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in remaining if Path(f"/proc/{pid}").exists()]

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    if LOGGER_AVAILABLE:
        logger.warning(
            "  Reaped %d detached benchmark process(es) from current run before %s",
            len(current_orphans),
            reason,
        )
    return len(current_orphans)


def _reap_run_descendants(
    reason: str,
    *,
    grace_seconds: float = 2.0,
    proc_root: Path = Path("/proc"),
) -> None:
    """Best-effort cleanup of leaked descendant processes owned by this run."""
    descendants = _collect_descendant_pids(os.getpid(), proc_root=proc_root)
    if not descendants:
        return

    for pid in descendants:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    deadline = time.monotonic() + max(0.0, grace_seconds)
    remaining = descendants
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in remaining if (proc_root / str(pid)).exists()]

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            continue

    if LOGGER_AVAILABLE and descendants:
        logger.warning(
            "  Reaped %d leaked descendant process(es) after %s",
            len(descendants),
            reason,
        )


def _reap_benchmark_process_leftovers(
    reason: str,
    *,
    current_run_id: str,
    current_owner_pid: int,
    repo_root: Path,
) -> None:
    """Clear benchmark-owned leftovers before the next benchmark phase starts."""
    _reap_run_descendants(reason)
    _reap_current_run_benchmark_orphans(
        reason,
        current_run_id=current_run_id,
        current_owner_pid=current_owner_pid,
        repo_root=repo_root,
    )
    _reap_stale_benchmark_orphans(
        reason,
        current_run_id=current_run_id,
        repo_root=repo_root,
    )


def profile_python_benchmark_ncu(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    config: Optional[BenchmarkConfig],
    variant: str = "baseline",
    output_stem: Optional[str] = None,
) -> Optional[Path]:
    """Profile a Python benchmark using ncu (NVIDIA Compute Profiler).
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save ncu-rep file
        config: BenchmarkConfig used for profiling settings
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated ncu-rep file, or None if failed
    """
    if not check_ncu_available():
        return None
    _set_profile_failure_detail("ncu", None)

    output_dir = _resolve_profile_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on benchmark name
    benchmark_name = output_stem or benchmark_path.stem
    ncu_output = output_dir / f"{benchmark_name}__{variant}.ncu-rep"
    log_base = output_dir / f"{benchmark_name}__{variant}__ncu"
    
    if config is None:
        config = BenchmarkConfig()
    config = _apply_preferred_ncu_profile_overrides(config, benchmark)

    profiling_view = config.profiling
    validity_view = config.validity
    timing_view = config.timing
    profiler_config = build_profiler_config_from_benchmark(config)
    configured_nvtx_includes = profiling_view.nsys_nvtx_include or profiler_config.nvtx_includes
    nvtx_includes = list(configured_nvtx_includes or [])
    # Keep the emitted wrapper NVTX range aligned with configured include filters;
    # otherwise NCU can connect/disconnect without capturing any kernels.
    profile_nvtx_label = "compute_kernel:profile"
    if nvtx_includes:
        first_include = str(nvtx_includes[0]).strip()
        if first_include:
            profile_nvtx_label = first_include.rstrip("/")
    try:
        from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
        is_cuda_binary = isinstance(benchmark, CudaBinaryBenchmark)
    except Exception:
        is_cuda_binary = False

    # Keep NCU NVTX includes opt-in only. A default include filter can miss
    # kernels for workloads where work is launched outside the wrapper NVTX
    # context (for example CUDA Graph replay or subprocess-driven execution).
    ncu_nvtx_includes = nvtx_includes or None
    if bool(getattr(benchmark, "disable_ncu_nvtx_filter", False)):
        ncu_nvtx_includes = None
    # chapter_dir points to e.g. <repo>/ch10 or <repo>/labs/<lab>; use global
    # repository root for package imports like `labs.*` and `core.*`.
    repo_root = Path(__file__).resolve().parents[2]
    chapter_num = None
    chapter_name = chapter_dir.name
    if chapter_name.startswith("ch") and chapter_name[2:].isdigit():
        chapter_num = int(chapter_name[2:])
    metric_set = str(profiling_view.ncu_metric_set or "auto").lower()
    # "auto" follows the active profiling preset:
    # - minimal -> MINIMAL_METRICS (basic set)
    # - roofline -> ROOFLINE_METRICS
    # - deep_dive -> chapter-specific metrics (when available)
    if metric_set == "auto":
        preset = str(profiling_view.profile_type or "minimal").lower()
        if preset in {"minimal", "roofline"}:
            metrics_override = resolve_ncu_metrics(preset, chapter=None)
        elif preset == "deep_dive":
            metrics_override = resolve_ncu_metrics("auto", chapter=chapter_num)
        else:
            metrics_override = resolve_ncu_metrics("deep_dive", chapter=None)
    else:
        metrics_override = resolve_ncu_metrics(metric_set, chapter=chapter_num)
    validity_profile = validity_view.validity_profile
    lock_gpu_clocks_flag = validity_view.lock_gpu_clocks
    if validity_profile != "strict":
        lock_gpu_clocks_flag = False
    gpu_sm_clock_mhz = validity_view.gpu_sm_clock_mhz
    gpu_mem_clock_mhz = validity_view.gpu_mem_clock_mhz
    target_label = getattr(config, "target_label", None)
    target_override_argv = _resolve_target_override_argv(config)

    try:
        use_torchrun = bool(_is_torchrun_launch(config))
        wrapper_path: Optional[Path] = None
        torchrun_profile_spec = None
        has_target_launch_spec = False
        with ExitStack() as stack:
            if use_torchrun:
                torchrun_profile_spec = _resolve_profile_torchrun_spec(
                    benchmark,
                    profiler="ncu",
                    config=config,
                )
            if torchrun_profile_spec is not None:
                has_target_launch_spec = True
                target_command, base_env = _build_torchrun_profile_command(
                    config,
                    spec=torchrun_profile_spec,
                )
                env = _apply_profile_env_overrides(
                    _harden_profile_env(base_env, repo_root=repo_root, chapter_dir=chapter_dir),
                    config=config,
                    benchmark=benchmark,
                )
                use_torchrun = _command_uses_external_torchrun(target_command)
            else:
                wrapper_source = render_ncu_python_profile_wrapper(
                    benchmark_path=benchmark_path,
                    configured_nvtx_includes=configured_nvtx_includes,
                    target_label=target_label,
                    target_override_argv=target_override_argv,
                    profile_type=profiling_view.profile_type,
                    ncu_metric_set=profiling_view.ncu_metric_set,
                    pm_sampling_interval=profiling_view.pm_sampling_interval,
                    ncu_replay_mode=profiling_view.ncu_replay_mode,
                    validity_profile=validity_profile,
                    lock_gpu_clocks_flag=lock_gpu_clocks_flag,
                    gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                    gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                    profile_nvtx_label=profile_nvtx_label,
                )
                wrapper_path, target_command, env, use_torchrun = stack.enter_context(
                    _temporary_python_profile_launch(
                        wrapper_source,
                        chapter_dir=chapter_dir,
                        repo_root=repo_root,
                        config=config,
                        benchmark=benchmark,
                    )
                )

            if has_target_launch_spec:
                ncu_command = profiler_config.get_ncu_command_for_target(
                    str(ncu_output.with_suffix("")),
                    target_command,
                    metrics=metrics_override,
                    nvtx_includes=ncu_nvtx_includes,
                )
            else:
                if wrapper_path is None:
                    raise RuntimeError("Wrapper path required for non-torchrun NCU profiling")
                ncu_command = profiler_config.get_ncu_command(
                    str(ncu_output.with_suffix("")),
                    str(wrapper_path),
                    python_executable=sys.executable,
                    metrics=metrics_override,
                    nvtx_includes=ncu_nvtx_includes,
                )
            ncu_env_overrides = getattr(benchmark, "ncu_env_overrides", None)
            if ncu_env_overrides:
                env.update({str(key): str(value) for key, value in ncu_env_overrides.items()})
            ncu_command.insert(1, "--force-overwrite")

            # ncu profiling timeout: align with BenchmarkConfig.ncu_timeout_seconds
            # ncu is slower than nsys and needs more time for metric collection
            ncu_timeout_seconds = timing_view.timeout_for("ncu") or NCU_TIMEOUT_SECONDS
            for attempt in range(1, _NCU_DRIVER_RESOURCE_RETRY_ATTEMPTS + 1):
                run_result = _run_profile_subprocess(
                    command=ncu_command,
                    cwd=chapter_dir,
                    env=env,
                    timeout_seconds=float(ncu_timeout_seconds),
                    log_base=log_base,
                    terminate_reason=f"{benchmark_name}_{variant}",
                    capture_output=True,
                    timeout_collect_error_message=(
                        f"Timed-out NCU profiling for {benchmark_name} ({variant}) could not collect trailing stdout/stderr"
                    ),
                    wait_error_message=f"NCU profiling communicate failed for {benchmark_name} ({variant})",
                )
                if run_result.failure_warning:
                    _append_profile_warning(run_result.stderr_log, run_result.failure_warning)
                    _set_profile_failure_detail(
                        "ncu",
                        _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log)
                        or run_result.failure_warning,
                    )
                    return None

                # Check if file exists (ncu may create file even with non-zero exit code)
                if not run_result.timed_out:
                    if ncu_output.exists():
                        _set_profile_failure_detail("ncu", None)
                        return ncu_output
                    alt_path = output_dir / f"{benchmark_name}__{variant}.ncu-rep"
                    if alt_path.exists():
                        _set_profile_failure_detail("ncu", None)
                        return alt_path
                    for ncu_file in output_dir.glob(f"{benchmark_name}__{variant}*.ncu-rep"):
                        _set_profile_failure_detail("ncu", None)
                        return ncu_file
                    if (
                        run_result.process.returncode not in (0, None)
                        and attempt < _NCU_DRIVER_RESOURCE_RETRY_ATTEMPTS
                        and _is_retryable_ncu_driver_resource_error(
                            stdout_log=run_result.stdout_log,
                            stderr_log=run_result.stderr_log,
                        )
                    ):
                        time.sleep(_NCU_DRIVER_RESOURCE_RETRY_DELAY_SECONDS)
                        continue
                    if run_result.process.returncode not in (0, None):
                        _append_profile_warning(
                            run_result.stderr_log,
                            f"NCU profiling exited with code {run_result.process.returncode} for {benchmark_name} ({variant}) without producing a report.",
                        )
                else:
                    _append_profile_warning(
                        run_result.stderr_log,
                        f"NCU profiling timed out after {ncu_timeout_seconds}s for {benchmark_name} ({variant}).",
                    )
                _set_profile_failure_detail(
                    "ncu",
                    _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log),
                )
                return None
        return None
    except (subprocess.SubprocessError, OSError) as exc:
        _set_profile_failure_detail("ncu", str(exc))
        if LOGGER_AVAILABLE:
            logger.warning(
                "  Failed to launch NCU profiling for %s (%s): %s",
                benchmark_name,
                variant,
                exc,
                exc_info=True,
            )
        return None


def _apply_preferred_ncu_profile_overrides(config: BenchmarkConfig, benchmark: Any) -> BenchmarkConfig:
    """Apply benchmark-local Nsight Compute overrides for harness-managed profiles."""
    replacements: Dict[str, Any] = {}

    preferred_replay = getattr(benchmark, "preferred_ncu_replay_mode", None)
    if preferred_replay:
        replacements["ncu_replay_mode"] = str(preferred_replay)
        replacements["ncu_replay_mode_override"] = True

    preferred_metric_set = getattr(benchmark, "preferred_ncu_metric_set", None)
    if preferred_metric_set:
        replacements["ncu_metric_set"] = str(preferred_metric_set)

    if not replacements:
        return config
    return replace(config, **replacements)


def profile_cuda_executable_ncu(
    executable: Path,
    chapter_dir: Path,
    output_dir: Path,
    config: BenchmarkConfig,
    benchmark: Optional[Any] = None,
    variant: str = "baseline",
    output_stem: Optional[str] = None,
) -> Optional[Path]:
    """Profile a CUDA executable using ncu (NVIDIA Compute Profiler).
    
    Args:
        executable: Path to CUDA executable
        chapter_dir: Path to chapter directory
        output_dir: Directory to save ncu-rep file
        config: BenchmarkConfig used for profiling settings
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated ncu-rep file, or None if failed
    """
    if not check_ncu_available():
        return None
    _set_profile_failure_detail("ncu", None)

    output_dir = _resolve_profile_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on executable name
    exec_name = output_stem or executable.stem
    ncu_output = output_dir / f"{exec_name}__{variant}.ncu-rep"
    log_base = output_dir / f"{exec_name}__{variant}__ncu"
    
    profiler_config = build_profiler_config_from_benchmark(config)
    chapter_num = None
    chapter_name = chapter_dir.name
    if chapter_name.startswith("ch") and chapter_name[2:].isdigit():
        chapter_num = int(chapter_name[2:])
    metric_set = str(getattr(config, "ncu_metric_set", "auto") or "auto").lower()
    # Keep CLI's "auto" behavior aligned with the profiling preset (see above).
    if metric_set == "auto":
        preset = str(getattr(config, "profile_type", None) or "minimal").lower()
        if preset in {"minimal", "roofline"}:
            metrics_override = resolve_ncu_metrics(preset, chapter=None)
        elif preset == "deep_dive":
            metrics_override = resolve_ncu_metrics("auto", chapter=chapter_num)
        else:
            metrics_override = resolve_ncu_metrics("deep_dive", chapter=None)
    else:
        metrics_override = resolve_ncu_metrics(metric_set, chapter=chapter_num)
    ncu_command = profiler_config.get_ncu_command_for_target(
        str(ncu_output.with_suffix("")),
        [str(executable)],
        metrics=metrics_override,
        nvtx_includes=profiler_config.nvtx_includes,
    )
    ncu_command.insert(1, "--force-overwrite")
    
    try:
        env = _apply_profile_env_overrides(
            _harden_profile_env(
                None,
                repo_root=Path(__file__).resolve().parents[2],
                chapter_dir=chapter_dir,
            ),
            config=config,
            benchmark=benchmark,
        )
        # ncu profiling timeout: align with BenchmarkConfig.ncu_timeout_seconds
        # ncu is slower than nsys and needs more time for metric collection
        ncu_timeout_seconds = config.get_effective_timeout("ncu") or NCU_TIMEOUT_SECONDS
        for attempt in range(1, _NCU_DRIVER_RESOURCE_RETRY_ATTEMPTS + 1):
            run_result = _run_profile_subprocess(
                command=ncu_command,
                cwd=chapter_dir,
                env=env,
                timeout_seconds=float(ncu_timeout_seconds),
                log_base=log_base,
                terminate_reason=f"{exec_name}__{variant}",
                capture_output=True,
                timeout_collect_error_message=(
                    f"Timed-out NCU profiling for executable {exec_name} ({variant}) could not collect trailing stdout/stderr"
                ),
                wait_error_message=f"NCU profiling communicate failed for executable {exec_name} ({variant})",
            )
            if run_result.failure_warning:
                _append_profile_warning(run_result.stderr_log, run_result.failure_warning)
                _set_profile_failure_detail(
                    "ncu",
                    _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log)
                    or run_result.failure_warning,
                )
                return None

            # Check if file exists (ncu may create file even with non-zero exit code)
            if not run_result.timed_out:
                if ncu_output.exists():
                    _set_profile_failure_detail("ncu", None)
                    return ncu_output
                alt_path = output_dir / f"{exec_name}__{variant}.ncu-rep"
                if alt_path.exists():
                    _set_profile_failure_detail("ncu", None)
                    return alt_path
                for ncu_file in output_dir.glob(f"{exec_name}__{variant}*.ncu-rep"):
                    _set_profile_failure_detail("ncu", None)
                    return ncu_file
                if (
                    run_result.process.returncode not in (0, None)
                    and attempt < _NCU_DRIVER_RESOURCE_RETRY_ATTEMPTS
                    and _is_retryable_ncu_driver_resource_error(
                        stdout_log=run_result.stdout_log,
                        stderr_log=run_result.stderr_log,
                    )
                ):
                    time.sleep(_NCU_DRIVER_RESOURCE_RETRY_DELAY_SECONDS)
                    continue
                if run_result.process.returncode not in (0, None):
                    _append_profile_warning(
                        run_result.stderr_log,
                        f"NCU profiling exited with code {run_result.process.returncode} for executable {exec_name} ({variant}) without producing a report.",
                    )
            else:
                _append_profile_warning(
                    run_result.stderr_log,
                    f"NCU profiling timed out after {ncu_timeout_seconds}s for executable {exec_name} ({variant}).",
                )
            _set_profile_failure_detail(
                "ncu",
                _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log),
            )
            return None
        return None
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        _set_profile_failure_detail("ncu", str(exc))
        if LOGGER_AVAILABLE:
            logger.warning(
                "  Failed to launch NCU profiling for executable %s (%s): %s",
                exec_name,
                variant,
                exc,
                exc_info=True,
            )
        return None


def profile_python_benchmark_torch(
    benchmark: Any,  # Benchmark instance
    benchmark_path: Path,
    chapter_dir: Path,
    output_dir: Path,
    variant: str = "baseline",
    output_stem: Optional[str] = None,
) -> Optional[Path]:
    """Profile a Python benchmark using PyTorch profiler.
    
    Args:
        benchmark: Benchmark instance (already loaded)
        benchmark_path: Path to Python benchmark file (for naming)
        chapter_dir: Path to chapter directory
        output_dir: Directory to save torch trace file
        variant: 'baseline' or 'optimized' for naming
        
    Returns:
        Path to generated torch trace JSON file, or None if failed
    """
    if not TORCH_PROFILER_AVAILABLE:
        return None
    _set_profile_failure_detail("torch", None)

    output_dir = _resolve_profile_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on benchmark name
    benchmark_name = output_stem or benchmark_path.stem
    torch_output = output_dir / f"{benchmark_name}__{variant}_torch_trace.json"
    log_base = output_dir / f"{benchmark_name}__{variant}__torch"
    existing_cfg = getattr(benchmark, "_config", None)
    if existing_cfg is None and hasattr(benchmark, "get_config"):
        existing_cfg = benchmark.get_config()
    validity_view = existing_cfg.validity if existing_cfg else None
    timing_view = existing_cfg.timing if existing_cfg else None
    validity_profile = validity_view.validity_profile if validity_view else "strict"
    if validity_profile not in {"strict", "portable"}:
        validity_profile = "strict"
    lock_gpu_clocks_flag = validity_view.lock_gpu_clocks if validity_view else True
    if validity_profile != "strict":
        lock_gpu_clocks_flag = False
    profile_timeout_seconds = 900
    maybe_timeout = timing_view.timeout_for("torch") if timing_view else None
    if maybe_timeout:
        profile_timeout_seconds = int(maybe_timeout)
    target_label = getattr(existing_cfg, "target_label", None) if existing_cfg else None
    target_override_argv = _resolve_target_override_argv(existing_cfg)

    # Run torch profiling in an isolated subprocess so a profiler-side hang
    # cannot wedge the parent benchmark sweep.
    repo_root = Path(__file__).resolve().parents[2]
    torchrun_profile_spec = None
    if existing_cfg is not None and _is_torchrun_launch(existing_cfg):
        torchrun_profile_spec = _resolve_profile_torchrun_spec(
            benchmark,
            profiler="torch",
            config=existing_cfg,
            output_path=torch_output,
        )
    with ExitStack() as stack:
        if torchrun_profile_spec is not None and existing_cfg is not None:
            cmd, base_env = _build_torchrun_profile_command(existing_cfg, spec=torchrun_profile_spec)
            env = _apply_profile_env_overrides(
                _harden_profile_env(base_env, repo_root=repo_root, chapter_dir=chapter_dir),
                config=existing_cfg,
                benchmark=benchmark,
            )
        else:
            wrapper_source = render_torch_python_profile_wrapper(
                benchmark_path=benchmark_path,
                torch_output=torch_output,
                target_label=target_label,
                target_override_argv=target_override_argv,
                validity_profile=validity_profile,
                lock_gpu_clocks_flag=lock_gpu_clocks_flag,
                gpu_sm_clock_mhz=validity_view.gpu_sm_clock_mhz if validity_view else None,
                gpu_mem_clock_mhz=validity_view.gpu_mem_clock_mhz if validity_view else None,
            )
            _wrapper_path, cmd, env, _use_torchrun = stack.enter_context(
                _temporary_python_profile_launch(
                    wrapper_source,
                    chapter_dir=chapter_dir,
                    repo_root=repo_root,
                    config=existing_cfg,
                    benchmark=benchmark,
                    allow_torchrun=False,
                )
            )

        run_result = _run_profile_subprocess(
            command=cmd,
            cwd=chapter_dir,
            env=env,
            timeout_seconds=float(profile_timeout_seconds),
            log_base=log_base,
            terminate_reason=f"{benchmark_name}__{variant}__torch",
            capture_output=False,
            timeout_collect_error_message=(
                f"Timed-out torch profiler for {benchmark_name} ({variant}) could not confirm process exit"
            ),
            wait_error_message=f"Torch profiler wait failed for {benchmark_name} ({variant})",
        )

    if run_result.timed_out:
        with run_result.stderr_log.open("a") as handle:
            handle.write(f"\ntorch profiler timed out after {profile_timeout_seconds}s\n")
        _set_profile_failure_detail(
            "torch",
            _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log),
        )
        return None
    if run_result.failure_warning:
        _append_profile_warning(run_result.stderr_log, run_result.failure_warning)
        _set_profile_failure_detail(
            "torch",
            _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log)
            or run_result.failure_warning,
        )
        return None
    if run_result.process.returncode != 0:
        _append_profile_warning(
            run_result.stderr_log,
            f"Torch profiler exited with code {run_result.process.returncode} for {benchmark_name} ({variant}). See {run_result.stderr_log} for details.",
        )
        _set_profile_failure_detail(
            "torch",
            _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log),
        )
        return None
    if not torch_output.exists():
        _append_profile_warning(
            run_result.stderr_log,
            f"Torch profiler completed without producing expected trace {torch_output} for {benchmark_name} ({variant}).",
        )
        _set_profile_failure_detail(
            "torch",
            _read_profile_failure_detail(run_result.stdout_log, run_result.stderr_log),
        )
        return None
    _set_profile_failure_detail("torch", None)
    return torch_output


def ensure_cuda_executables_built(chapter_dir: Path) -> Tuple[bool, Optional[str]]:
    """Try to build CUDA executables if Makefile exists.
    
    Uses auto-detection to build for the correct GPU architecture (sm_121, sm_103, or sm_100).
    The Makefile will auto-detect the architecture unless ARCH is explicitly set.
    
    Args:
        chapter_dir: Path to chapter directory
        
    Returns:
        Tuple of (success flag, optional failure reason)
    """
    makefile = chapter_dir / "Makefile"
    if not makefile.exists():
        return True, None  # No Makefile, assume executables are pre-built or don't exist
    
    env = os.environ.copy()
    make_desc = "default ARCH"
    if detect_supported_arch is not None:
        try:
            arch = detect_supported_arch()
            if arch:
                env["ARCH"] = arch
                make_desc = f"ARCH={arch}"
        except Exception as exc:
            logger.warning(f"  WARNING: Unable to auto-detect CUDA arch for {chapter_dir.name}: {exc}")
    
    try:
        # Clean build directory before building to prevent stale lock issues
        build_dir = chapter_dir / "build"
        if build_dir.exists():
            try:
                from core.utils.build_utils import ensure_clean_build_directory
                ensure_clean_build_directory(build_dir)
            except ImportError:
                logger.warning("  WARNING: build_utils unavailable; skipping stale build directory cleanup")
            except Exception as exc:
                logger.warning(f"  WARNING: Failed to clean build directory {build_dir}: {exc}")
        
        logger.info(f"  Building CUDA executables ({make_desc})...")
        # Explicitly set ARCH so Makefiles consistently target the active GPU
        result = subprocess.run(
            ["make", "-B", "-C", str(chapter_dir), "all"],
            capture_output=True,
            timeout=300,  # Increased timeout for CUDA JIT compilation (can take 60+ seconds)
            check=False,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.warning(f"  WARNING: Make build failed (exit code {result.returncode})")
            if result.stderr:
                logger.warning(f"  Build stderr: {result.stderr[:500]}")
            failure_snippet = (result.stderr or "").strip().splitlines()
            reason = failure_snippet[0] if failure_snippet else f"Make exited with code {result.returncode}"
            return False, reason
        return True, None
    except subprocess.TimeoutExpired:
        # Make timed out - compilation takes too long
        logger.warning(f"  WARNING: Make build timed out after 300s - compilation may be too slow or hanging")
        return False, "Make build timed out after 300s"
    except Exception as e:
        logger.warning(f"  WARNING: Make build exception: {e}")
        return False, str(e)


def _compute_locked_fields(
    *,
    base_config: BenchmarkConfig,
    cli_iterations_provided: bool,
    cli_warmup_provided: bool,
    cli_ncu_replay_mode_provided: bool = False,
    cli_nsys_timeout_provided: bool = False,
    cli_ncu_timeout_provided: bool = False,
    enable_profiling: bool,
) -> Set[str]:
    """Compute run-level config fields that benchmarks may not override."""
    locked_fields: Set[str] = set()
    if cli_iterations_provided:
        locked_fields.add("iterations")
    if cli_warmup_provided:
        locked_fields.add("warmup")
    if cli_ncu_replay_mode_provided:
        locked_fields.add("ncu_replay_mode")
        locked_fields.add("ncu_replay_mode_override")
    if cli_nsys_timeout_provided:
        locked_fields.add("nsys_timeout_seconds")
    if cli_ncu_timeout_provided:
        locked_fields.add("ncu_timeout_seconds")

    runner_locked_when_true: Set[str] = {"enable_memory_tracking", "detect_setup_precomputation"}
    if enable_profiling:
        runner_locked_when_true.update({"enable_profiling", "enable_nsys", "enable_ncu", "enable_nvtx", "profile_type"})
    for field_name in runner_locked_when_true:
        if getattr(base_config, field_name, None):
            locked_fields.add(field_name)

    return locked_fields


@contextmanager
def _sanitized_distributed_env():
    """Temporarily clear torchrun env vars to avoid cross-benchmark contamination."""
    keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
    )
    saved = {}
    for key in keys:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ[key] = value


def _merge_benchmark_config(
    *,
    base_config: BenchmarkConfig,
    benchmark_obj: Any,
    defaults_obj: Optional[Any],
    locked_fields: Set[str],
) -> BenchmarkConfig:
    """Merge benchmark-provided config with run-level config, enforcing invariants."""
    merged = copy.deepcopy(base_config)

    bench_config = getattr(benchmark_obj, "get_config", None)
    if callable(bench_config):
        with _sanitized_distributed_env():
            override = bench_config()
    else:
        override = None
    if override:
        for field in fields(BenchmarkConfig):
            value = getattr(override, field.name, None)
            if value is None:
                continue
            if field.name in locked_fields:
                continue

            if field.name == "launch_via":
                base_value = getattr(merged, field.name, None)
                default_value = getattr(defaults_obj, field.name, None) if defaults_obj else None

                def _normalize_launch(val):
                    if val is None:
                        return None
                    if hasattr(val, "value"):
                        return str(getattr(val, "value")).lower()
                    return str(val).lower()

                base_norm = _normalize_launch(base_value)
                default_norm = _normalize_launch(default_value)
                value_norm = _normalize_launch(value)
                # Preserve CLI-provided launcher when the benchmark config only supplies the default
                if (
                    base_norm is not None
                    and default_norm is not None
                    and base_norm != default_norm
                    and value_norm == default_norm
                ):
                    continue
            if field.name == "ncu_metric_set":
                base_value = getattr(merged, field.name, None)
                default_value = getattr(defaults_obj, field.name, None) if defaults_obj else None
                if base_value is not None and default_value is not None:
                    base_norm = str(base_value).lower()
                    default_norm = str(default_value).lower()
                    value_norm = str(value).lower()
                    # Preserve CLI-provided metric set when benchmark config only supplies the default.
                    if base_norm != default_norm and value_norm == default_norm:
                        continue

            if field.name == "target_extra_args":
                if value:
                    merged.target_extra_args = {
                        **(getattr(merged, "target_extra_args", {}) or {}),
                        **value,
                    }
                continue

            if field.name == "env_passthrough" and not value:
                continue

            setattr(merged, field.name, copy.deepcopy(value))

    merged._sync_execution_mode()
    merged._sync_launch_via()

    # Explicit invariants: benchmarks must not override run-level policy knobs.
    merged.timeout_multiplier = getattr(base_config, "timeout_multiplier", merged.timeout_multiplier)
    merged.enforce_environment_validation = getattr(
        base_config,
        "enforce_environment_validation",
        merged.enforce_environment_validation,
    )
    merged.allow_virtualization = getattr(
        base_config,
        "allow_virtualization",
        getattr(merged, "allow_virtualization", False),
    )
    merged.allow_foreign_gpu_processes = getattr(
        base_config,
        "allow_foreign_gpu_processes",
        getattr(merged, "allow_foreign_gpu_processes", False),
    )
    merged.validity_profile = getattr(
        base_config,
        "validity_profile",
        getattr(merged, "validity_profile", "strict"),
    )

    return merged


def _canonicalize_optimized_variants_for_full_sweep(
    python_pairs: List[Tuple[Path, List[Path], str]],
) -> Tuple[List[Tuple[Path, List[Path], str]], int]:
    """Prefer one canonical optimized file per baseline when available.

    Discovery may return many optimized variants for a single baseline. For full chapter
    sweeps, we prefer `optimized_<example>.py` when present and keep alias-targeted pairs
    untouched.
    """

    suppressed = 0
    canonical_pairs: List[Tuple[Path, List[Path], str]] = []
    for baseline_path, optimized_paths, example_name in python_pairs:
        canonical_name = baseline_path.stem.replace("baseline_", "", 1)
        # Keep alias-targeted entries unchanged.
        if example_name != canonical_name:
            canonical_pairs.append((baseline_path, optimized_paths, example_name))
            continue

        canonical_opt = baseline_path.parent / f"optimized_{canonical_name}{baseline_path.suffix}"
        if canonical_opt in optimized_paths:
            removed = max(0, len(optimized_paths) - 1)
            suppressed += removed
            canonical_pairs.append((baseline_path, [canonical_opt], example_name))
        else:
            # Fallback to discovered variants when no canonical optimized file exists.
            canonical_pairs.append((baseline_path, optimized_paths, example_name))

    return canonical_pairs, suppressed


def _resolve_expectation_validation_policy(
    *,
    validity_profile: str,
    update_expectations: bool,
    accept_regressions: bool,
    allow_mixed_provenance: bool,
    allow_portable_expectations_update: bool,
) -> Tuple[bool, bool]:
    """Return (validation_enabled, writes_enabled) for expectation handling."""
    normalized_profile = normalize_validity_profile(
        str(validity_profile).strip().lower(),
        field_name="validity_profile",
    )
    validation_enabled = (
        normalized_profile == "strict" or bool(allow_portable_expectations_update)
    )
    write_requested = bool(
        update_expectations or accept_regressions or allow_mixed_provenance
    )
    writes_enabled = validation_enabled and write_requested
    return validation_enabled, writes_enabled


def _allow_mixed_provenance_for_expectation_writes(
    *,
    update_expectations: bool,
    allow_mixed_provenance: bool,
) -> bool:
    """Return whether expectation writes may override provenance mismatches.

    Explicit expectation refreshes from the current host are deliberate
    mixed-provenance writes. This includes virtualized/non-canonical hosts; the
    refreshed entry still records full provenance so downstream analysis can see
    that the source run was virtualized.
    """

    return bool(allow_mixed_provenance or update_expectations)


def _test_chapter_impl(
    chapter_dir: Path,
    enable_profiling: bool = False,
    profile_type: str = "none",
    profile_output_root: Optional[Path] = None,
    timeout_multiplier: float = 3.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    single_gpu: bool = False,
    enforce_environment_validation: bool = True,
    allow_virtualization: bool = False,
    allow_foreign_gpu_processes: bool = False,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    only_examples: Optional[List[str]] = None,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    pm_sampling_interval: Optional[int] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    force_synchronize: bool = False,
    graph_capture_ratio_threshold: Optional[float] = None,
    graph_capture_memory_threshold_mb: Optional[float] = None,
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    env_passthrough: Optional[List[str]] = None,
    target_extra_args: Optional[Dict[str, List[str]]] = None,
    subprocess_stderr_dir: Optional[Path] = None,
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    only_cuda: bool = False,
    only_python: bool = False,
    # LLM analysis and patching options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
    progress_recorder: Optional[ProgressRecorder] = None,
    progress_completed_benchmarks: int = 0,
    progress_total_benchmarks: Optional[int] = None,
    event_logger: Optional["BenchmarkEventLogger"] = None,
    fail_on_no_benchmarks: bool = False,
) -> Dict[str, Any]:
    """Test all benchmarks in a chapter and return results.
    
    Args:
        chapter_dir: Path to chapter directory
        enable_profiling: If True, generate profiling files (nsys, ncu, PyTorch) alongside benchmarks
        profile_output_root: Base directory for profiling artifacts (default: artifacts/runs/<run_id>/profiles)
        timeout_multiplier: Multiply all timeouts by this factor (e.g., 2.0 = double all timeouts)
        reproducible: If True, set all seeds to 42 and force deterministic algorithms (slower fallbacks; ops without deterministic support may fail)
        cold_start: If True, perform additional GPU state cleanup (gc.collect()) between benchmarks for cold start measurements. CUDA state is always reset by default.
        iterations: Number of benchmark iterations (defaults to 20 if not provided)
        warmup: Number of warmup iterations (defaults to 5 if not provided)
        only_examples: List of example names to run (e.g., ['moe', 'cutlass']). If None, runs all examples.
        launch_via: Launcher to use ('python' or 'torchrun')
        nproc_per_node: torchrun --nproc_per_node value
        nnodes: torchrun --nnodes value
        rdzv_backend: torchrun rendezvous backend
        rdzv_endpoint: torchrun rendezvous endpoint
        env_passthrough: Environment variables to pass through to subprocess launches
        target_extra_args: Optional per-target arg overrides (target -> list of CLI args)
    """
    logger.info("launch_via arg=%s nproc_per_node=%s nnodes=%s", launch_via, nproc_per_node, nnodes)
    debug_signature_verify = str(os.environ.get("AISP_DEBUG_SIGNATURE_VERIFY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    validity_profile = str(validity_profile).strip().lower()
    validity_profile = normalize_validity_profile(validity_profile, field_name="validity_profile")
    allow_virtualization = validity_profile == "portable"
    expectation_validation_enabled, expectation_writes_enabled = _resolve_expectation_validation_policy(
        validity_profile=validity_profile,
        update_expectations=bool(update_expectations),
        accept_regressions=bool(accept_regressions),
        allow_mixed_provenance=bool(allow_mixed_provenance),
        allow_portable_expectations_update=bool(allow_portable_expectations_update),
    )
    if validity_profile != "strict" and not expectation_validation_enabled:
        logger.warning(
            "Portable validity profile active: expectation validation and file updates are disabled. "
            "Set --allow-portable-expectations-update "
            "(allow_portable_expectations_update=True) to enable them."
        )

    if single_gpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
            os.environ["CUDA_VISIBLE_DEVICES"] = tokens[0] if tokens else "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dump_environment_and_capabilities()

    chapter_id = chapter_slug(chapter_dir, repo_root)
    chapter_name = chapter_id.replace("/", "_")
    emit_event(
        event_logger,
        logger,
        "chapter_start",
        chapter=chapter_name,
        chapter_dir=str(chapter_dir),
        profile_type=profile_type,
        enable_profiling=enable_profiling,
        only_examples=only_examples,
        timeout_multiplier=timeout_multiplier,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        ncu_metric_set=ncu_metric_set,
        enforce_environment_validation=enforce_environment_validation,
        validity_profile=validity_profile,
        allow_virtualization=allow_virtualization,
        allow_foreign_gpu_processes=allow_foreign_gpu_processes,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
    )

    # Set up profiling output directory if profiling is enabled
    profiling_output_dir = None
    if enable_profiling:
        if profile_output_root is None:
            profile_run_id = build_run_id(
                "bench-profile",
                f"chapter-{chapter_name}",
                base_dir=default_artifacts_root(repo_root),
            )
            profile_output_root = default_artifacts_root(repo_root) / profile_run_id / "profiles"
        profiling_output_dir = _profile_bench_dir(profile_output_root, chapter_id)
        profiling_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which profilers are available
        nsys_avail = check_nsys_available()
        ncu_avail = check_ncu_available()
        # Use module-level check to avoid local variable shadowing issue
        torch_avail = TORCH_PROFILER_AVAILABLE
        
        profilers = []
        if nsys_avail:
            profilers.append("nsys")
        if ncu_avail:
            profilers.append("ncu")
        if torch_avail:
            profilers.append("PyTorch")
        
        if profilers:
            logger.info(f"  Profiling enabled: {', '.join(profilers)} profiling files will be saved to {profiling_output_dir}")
            emit_event(
                event_logger,
                logger,
                "profiling_enabled",
                chapter=chapter_name,
                profilers=profilers,
                profiling_output_dir=str(profiling_output_dir),
            )
        else:
            logger.warning(f"  Profiling requested but no profilers available - skipping profiling")
            enable_profiling = False
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {chapter_name.upper()}")
    logger.info(f"{'='*80}")

    expectation_hardware_key = detect_expectation_key()
    allow_mixed_provenance_for_writes = _allow_mixed_provenance_for_expectation_writes(
        update_expectations=bool(update_expectations),
        allow_mixed_provenance=bool(allow_mixed_provenance),
    )

    expectations_store = ExpectationsStore(
        chapter_dir,
        expectation_hardware_key,
        accept_regressions=accept_regressions or update_expectations,
        allow_mixed_provenance=allow_mixed_provenance_for_writes,
    )
    expectation_path = _repo_relative_path(expectations_store.path, repo_root)
    logger.info(f"  Expectations key: {expectation_hardware_key} (file: {expectation_path})")
    emit_event(
        event_logger,
        logger,
        "expectations_context",
        chapter=chapter_name,
        hardware_key=expectation_hardware_key,
        expectation_file=str(expectation_path),
    )
    git_commit = None
    try:
        git_commit = get_git_info().get("commit")
    except Exception as exc:
        if LOGGER_AVAILABLE:
            logger.warning("  Failed to collect git commit provenance for %s: %s", chapter_name, exc, exc_info=True)
        git_commit = None
    execution_environment = detect_execution_environment()

    if not torch.cuda.is_available():
        emit_event(
            event_logger,
            logger,
            "chapter_skip",
            chapter=chapter_name,
            reason="CUDA not available",
        )
        return {
            'chapter': chapter_name,
            'status': 'skipped',
            'reason': 'CUDA not available',
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'successful': 0,
                'failed': 0,
                'total_speedup': 0.0,
                'average_speedup': 0.0,
            }
        }

    def _reset_parent_execution_state(*, include_gpu_state: bool = False) -> None:
        allow_cuda_context = str(launch_via).strip().lower() == "torchrun"
        reset_cuda_state(allow_cuda_context=allow_cuda_context)
        if include_gpu_state and allow_cuda_context:
            reset_gpu_state()
        if cold_start and allow_cuda_context:
            reset_gpu_state()
    
    # Clean build directories to prevent stale lock issues (before any GPU operations)
    logger.info(f"  Cleaning build directories...")
    clean_build_directories(chapter_dir)
    
    # Reset CUDA state at start of chapter (always, to prevent cascading failures)
    logger.info(f"  Resetting GPU state...")
    _reset_parent_execution_state(include_gpu_state=True)
    
    # Ensure PyTorch inductor cache directory exists to prevent C++ compilation errors
    # (This is also done in env_defaults.py, but we ensure it here as well for safety)
    # Use absolute path to avoid working directory issues
    inductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", ".torch_inductor")
    if inductor_cache_dir:
        # Convert relative paths to absolute paths
        if not os.path.isabs(inductor_cache_dir):
            inductor_cache_dir = str(Path.cwd() / inductor_cache_dir)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
        inductor_cache_path = Path(inductor_cache_dir)
        try:
            inductor_cache_path.mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "od").mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "tk").mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            if LOGGER_AVAILABLE:
                logger.warning(
                    "  Failed to create TORCHINDUCTOR_CACHE_DIR at %s: %s. Continuing with existing environment defaults.",
                    inductor_cache_path,
                    exc,
                    exc_info=True,
                )
    
    # Discover benchmark pairs using the same filters as the outer run planner.
    logger.info(f"  Discovering Python benchmarks...")
    if only_examples:
        logger.info(f"  Requested examples: {only_examples}")
    (
        python_pairs,
        cuda_pairs,
        example_filters,
        suppressed_pairs,
        suppressed_variant_opts,
    ) = _discover_chapter_benchmark_pairs(
        chapter_dir,
        only_examples=only_examples,
        only_cuda=only_cuda,
        only_python=only_python,
    )
    if example_filters:
        logger.info(f"  Filtered to {len(python_pairs)} example(s): {', '.join(sorted(example_filters))}")
    elif suppressed_pairs:
        logger.info(
            "  Suppressed %d alias benchmark pair(s) for full chapter run.",
            suppressed_pairs,
        )
    if suppressed_variant_opts:
        logger.info(
            "  Canonicalized %d optimization variant(s) for %s full sweep.",
            suppressed_variant_opts,
            chapter_id,
        )
    logger.info(f"  Found {len(python_pairs)} Python benchmark pair(s)")

    # Discover CUDA benchmarks and ensure executables are built
    logger.info(f"  Discovering CUDA benchmarks...")
    cuda_build_ok = True
    cuda_build_warning = None
    if cuda_pairs:
        logger.info(f"  Found {len(cuda_pairs)} CUDA benchmark pair(s), ensuring executables are built...")
        cuda_build_ok, cuda_build_warning = ensure_cuda_executables_built(chapter_dir)
    
    total_benchmarks = len(python_pairs) + len(cuda_pairs)
    logger.info(f"  Benchmark counts -> python: {len(python_pairs)}, cuda: {len(cuda_pairs)}, total: {total_benchmarks}")
    if not total_benchmarks:
        if fail_on_no_benchmarks:
            return {
                'chapter': chapter_name,
                'status': 'failed_no_benchmarks',
                'reason': 'No baseline/optimized pairs found for explicitly requested target',
                'benchmarks': [],
                'summary': {
                    'total_benchmarks': 0,
                    'successful': 0,
                    'failed': 1,
                    'failed_generic': 1,
                }
            }
        return {
            'chapter': chapter_name,
            'status': 'no_benchmarks',
            'reason': 'No baseline/optimized pairs found',
            'benchmarks': [],
            'summary': {
                'total_benchmarks': 0,
                'successful': 0,
                'failed': 0,
            }
        }
    
    watchdog_record = None
    stop_watchdog = None
    if total_benchmarks:
        watchdog_record, stop_watchdog = start_progress_watchdog(
            logger,
            chapter_name,
        )

    # Create harness for Python benchmarks with explicit timeout to prevent hangs
    cli_iterations_provided = iterations is not None
    cli_warmup_provided = warmup is not None

    if iterations is None:
        iterations = 20
    if warmup is None:
        warmup = 5
    
    try:
        from core.benchmark.defaults import get_defaults as _get_defaults  # type: ignore
        _defaults_obj = _get_defaults()
    except Exception as exc:
        if LOGGER_AVAILABLE:
            logger.warning("  Failed to load benchmark defaults object: %s", exc, exc_info=True)
        _defaults_obj = None

    measurement_timeout_default = getattr(_defaults_obj, "measurement_timeout_seconds", 1200) if _defaults_obj else 1200
    setup_timeout_default = getattr(_defaults_obj, "setup_timeout_seconds", 300) if _defaults_obj else 300
    # BenchmarkHarness has an internal profiling runner (enable_profiling/enable_nsys/enable_ncu),
    # but this script already performs explicit per-variant nsys+ncu captures after timing.
    # Keep a single profiling path to avoid duplicate captures and accidental "full" NCU runs.
    harness_internal_profiling = False
    config_kwargs: Dict[str, Any] = dict(
        iterations=iterations,
        warmup=warmup,
        measurement_timeout_seconds=measurement_timeout_default,
        setup_timeout_seconds=setup_timeout_default,
        timeout_multiplier=timeout_multiplier,  # Apply timeout multiplier from CLI
        enable_memory_tracking=True,  # Enable memory metrics display
        enable_profiling=harness_internal_profiling,
        enable_nsys=harness_internal_profiling,
        enable_ncu=harness_internal_profiling,
        single_gpu=single_gpu,
        seed=42 if reproducible else None,  # Set seed for reproducibility
        deterministic=reproducible,  # Enable deterministic algorithms for reproducibility
        enforce_environment_validation=enforce_environment_validation,
        validity_profile=validity_profile,
        allow_virtualization=allow_virtualization,
        allow_foreign_gpu_processes=allow_foreign_gpu_processes,
        force_synchronize=force_synchronize,
        ncu_metric_set=ncu_metric_set,
        profile_type=profile_type if enable_profiling else "none",
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        env_passthrough=env_passthrough or None,
        target_extra_args=target_extra_args or {},
    )
    if nsys_timeout_seconds is not None:
        config_kwargs["nsys_timeout_seconds"] = int(nsys_timeout_seconds)
    if ncu_timeout_seconds is not None:
        config_kwargs["ncu_timeout_seconds"] = int(ncu_timeout_seconds)
    if ncu_replay_mode is not None:
        config_kwargs["ncu_replay_mode"] = ncu_replay_mode
        config_kwargs["ncu_replay_mode_override"] = True
    if pm_sampling_interval is not None:
        config_kwargs["pm_sampling_interval"] = pm_sampling_interval
    elif _defaults_obj is not None:
        config_kwargs["pm_sampling_interval"] = getattr(_defaults_obj, "pm_sampling_interval", None)
    if graph_capture_ratio_threshold is not None:
        config_kwargs["graph_capture_cheat_ratio_threshold"] = graph_capture_ratio_threshold
    if graph_capture_memory_threshold_mb is not None:
        config_kwargs["graph_capture_memory_threshold_mb"] = graph_capture_memory_threshold_mb
    elif _defaults_obj is not None:
        config_kwargs["graph_capture_memory_threshold_mb"] = getattr(_defaults_obj, "graph_capture_memory_threshold_mb", None)
    # Note: graph_capture thresholds use BenchmarkDefaults values
    # To customize, add graph_capture_ratio_threshold/graph_capture_memory_threshold_mb
    # as parameters to _test_chapter_impl and pass from CLI
    base_config = BenchmarkConfig(**config_kwargs)
    logger.info("base_config launch_via=%s", base_config.launch_via)
    if profiling_output_dir:
        base_config.profiling_output_dir = str(profiling_output_dir)
    if subprocess_stderr_dir:
        base_config.subprocess_stderr_dir = str(subprocess_stderr_dir)

    locked_fields = _compute_locked_fields(
        base_config=base_config,
        cli_iterations_provided=cli_iterations_provided,
        cli_warmup_provided=cli_warmup_provided,
        cli_ncu_replay_mode_provided=ncu_replay_mode is not None,
        cli_nsys_timeout_provided=nsys_timeout_seconds is not None,
        cli_ncu_timeout_provided=ncu_timeout_seconds is not None,
        enable_profiling=enable_profiling,
    )

    def _resolve_required_world_size(benchmark_obj: Any, cfg: BenchmarkConfig) -> tuple[int, bool]:
        required = getattr(cfg, "required_world_size", None)
        if required is None:
            required = getattr(benchmark_obj, "required_world_size", None)
        if required is not None:
            required = int(required)
            if required <= 0:
                raise ValueError(f"required_world_size must be positive, got {required}")
            return required, True
        multi_gpu_required = bool(
            getattr(cfg, "multi_gpu_required", False) or getattr(benchmark_obj, "multi_gpu_required", False)
        )
        if multi_gpu_required:
            return 2, False
        return 1, True

    def _is_multi_gpu_benchmark(benchmark_obj: Any) -> bool:
        try:
            import inspect

            source_path = inspect.getsourcefile(benchmark_obj.__class__)
        except Exception:
            source_path = None
        if not source_path:
            return False
        try:
            return is_distributed_benchmark(Path(source_path))
        except Exception:
            return False

    def _run_with_config(benchmark_obj, run_id: str, target_label: Optional[str] = None):
        merged = _merge_benchmark_config(
            base_config=base_config,
            benchmark_obj=benchmark_obj,
            defaults_obj=_defaults_obj,
            locked_fields=locked_fields,
        )
        if getattr(merged, "use_subprocess", False):
            required, exact = _resolve_required_world_size(benchmark_obj, merged)
            if exact and required == 1 and not _is_multi_gpu_benchmark(benchmark_obj):
                merged.single_gpu = True
        if target_label and getattr(merged, "target_label", None) is None:
            merged.target_label = target_label
        logger.info("merged config launch_via=%s execution_mode=%s", merged.launch_via, merged.execution_mode)
        local_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=merged)
        return local_harness.benchmark_with_manifest(benchmark_obj, run_id=run_id), merged

    # ---------------------------------------------------------------------
    # Post-timing verification helpers (no re-execution)
    # ---------------------------------------------------------------------
    perf_compare_runner = VerifyRunner() if VERIFICATION_AVAILABLE else None

    def _get_perf_output(bench: Any):
        if hasattr(bench, "_subprocess_verify_output"):
            out = getattr(bench, "_subprocess_verify_output")
            if out is None:
                raise RuntimeError("Missing subprocess verify_output")
            return out
        return bench.get_verify_output()

    def _get_perf_tolerance(bench: Any) -> tuple[float, float]:
        if hasattr(bench, "_subprocess_output_tolerance"):
            tol = getattr(bench, "_subprocess_output_tolerance")
            if tol is None:
                raise RuntimeError("Missing subprocess output_tolerance")
            return tol
        return bench.get_output_tolerance()

    def _get_perf_signature(bench: Any):
        if hasattr(bench, "_subprocess_input_signature"):
            sig = getattr(bench, "_subprocess_input_signature")
            if sig is None:
                raise RuntimeError("Missing subprocess input_signature")
            return sig
        return bench.get_input_signature()

    def _diff_paths(a: Any, b: Any, prefix: str = "", out: Optional[List[str]] = None) -> List[str]:
        if out is None:
            out = []
        if len(out) >= 64:
            return out
        if isinstance(a, dict) and isinstance(b, dict):
            keys = sorted(set(a.keys()) | set(b.keys()))
            for k in keys:
                if len(out) >= 64:
                    break
                if k not in a:
                    out.append(f"{prefix}{k} (missing baseline)")
                    continue
                if k not in b:
                    out.append(f"{prefix}{k} (missing optimized)")
                    continue
                _diff_paths(a.get(k), b.get(k), prefix=f"{prefix}{k}.", out=out)
            return out
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{prefix}len {len(a)} != {len(b)}")
            for idx in range(min(len(a), len(b))):
                if len(out) >= 64:
                    break
                _diff_paths(a[idx], b[idx], prefix=f"{prefix}[{idx}].", out=out)
            return out
        if a != b:
            out.append(prefix[:-1] if prefix.endswith(".") else prefix or "<root>")
        return out
    
    benchmark_results = []
    manifest_entries: List[Dict[str, Any]] = []
    successful = 0
    failed_error = 0
    failed_regression = 0
    failed_no_speedup = 0
    skipped_hw = 0
    skipped_distributed = 0
    informational_skipped = 0
    speedups = []
    # Check GPU count for distributed benchmark detection
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    done_count = 0
    progress_ok = True
    progress_total = progress_total_benchmarks or total_benchmarks

    def emit_progress(
        phase: str,
        *,
        step: str,
        step_detail: Optional[str] = None,
        artifacts: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        percent_override: Optional[float] = None,
    ) -> None:
        nonlocal progress_ok
        if watchdog_record:
            # Keep watchdog phase-aware so long profiler phases don't trigger false hang warnings.
            watchdog_record(step_detail or step or phase)
        if not progress_recorder or not progress_ok:
            return
        phase_index = PROGRESS_PHASES.get(phase, 0)
        percent_complete = percent_override
        if percent_complete is None and progress_total:
            percent_complete = _compute_global_progress_percent(
                completed_benchmarks=done_count,
                total_benchmarks=progress_total,
                phase_index=phase_index,
                total_phases=PROGRESS_TOTAL_PHASES,
                benchmark_offset=progress_completed_benchmarks,
            )
        event = ProgressEvent(
            phase=phase,
            phase_index=phase_index,
            total_phases=PROGRESS_TOTAL_PHASES,
            step=step,
            step_detail=step_detail,
            percent_complete=percent_complete,
            artifacts=artifacts or [],
            metrics=metrics or {},
        )
        try:
            progress_recorder.emit(event)
        except Exception as exc:
            progress_ok = False
            logger.warning("Progress updates disabled: %s", exc)

    emit_progress(
        "discovery",
        step=f"{chapter_name}:discovery",
        step_detail=f"python={len(python_pairs)} cuda={len(cuda_pairs)} total={total_benchmarks}",
    )

    def _record_manifest(
        run: Any,
        *,
        variant: str,
        file_name: str,
        target_label: str,
        technique: Optional[str] = None,
    ) -> None:
        manifest = getattr(run, "manifest", None)
        if manifest is None:
            raise RuntimeError(f"Missing manifest for {target_label} ({variant})")
        manifest_entries.append({
            "run_id": getattr(run, "run_id", None),
            "timestamp": getattr(run, "timestamp", None),
            "target_label": target_label,
            "variant": variant,
            "file": file_name,
            "technique": technique,
            "manifest": manifest.model_dump(mode="json"),
        })

    def mark_progress(example_label: str) -> None:
        nonlocal done_count
        done_count += 1
        if progress_total:
            percent = ((progress_completed_benchmarks + done_count) / progress_total) * 100.0
        else:
            percent = None
        step_detail = f"completed {done_count}/{total_benchmarks}"
        if progress_total and progress_total != total_benchmarks:
            step_detail = (
                f"{step_detail} (global {progress_completed_benchmarks + done_count}/{progress_total})"
            )
        emit_progress(
            "complete",
            step=f"{chapter_name}:{example_label}",
            step_detail=step_detail,
            percent_override=percent,
        )
        if watchdog_record:
            watchdog_record(f"{chapter_name}:{example_label} ({done_count}/{total_benchmarks})")

    from contextlib import ExitStack

    with ExitStack() as cleanup_stack:
        if stop_watchdog:
            cleanup_stack.callback(stop_watchdog)

        # Process Python benchmarks
        for baseline_path, optimized_paths, example_name in python_pairs:
            logger.info(f"\n  Example: {example_name}")
            logger.info(f"    Baseline: {baseline_path.name}")
            example_type = "cuda" if _is_cuda_wrapper(baseline_path) else "python"
            emit_event(
                event_logger,
                logger,
                "example_start",
                chapter=chapter_name,
                example=example_name,
                example_type=example_type,
                baseline_file=baseline_path.name,
                optimized_files=[p.name for p in optimized_paths],
            )
        
            if is_informational_example(chapter_name, example_name):
                informational_skipped += 1
                logger.info("    ℹ️ Informational systems demo - documented for reference, not benchmarked.")
                emit_event(
                    event_logger,
                    logger,
                    "example_skip",
                    chapter=chapter_name,
                    example=example_name,
                    example_type=example_type,
                    reason="informational_demo",
                )
                mark_progress(example_name)
                continue

            _reap_benchmark_process_leftovers(
                f"example_start:{chapter_name}:{example_name}",
                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                current_owner_pid=os.getpid(),
                repo_root=repo_root,
            )
        
            result_entry = {
                'example': example_name,
                'type': example_type,
                'baseline_file': baseline_path.name,
                'baseline_time_ms': None,
                'baseline_throughput': None,
                'baseline_memory_mb': None,  # Peak memory for baseline
                'baseline_profiler_statuses': {},
                'optimizations': [],
                'best_speedup': 1.0,
                'best_memory_savings_pct': 0.0,  # Memory reduction percentage
                'optimization_goal': 'speed',  # Primary goal: speed, memory, throughput
                'minimum_required_speedup': None,
                'status': 'failed_error',
                'error': None,
            }

            baseline_signature = None
            baseline_equivalence = None
            baseline_verify_output = None
            baseline_verify_tolerance = None
        
            # Check if this is a distributed benchmark and we have only 1 GPU
            is_distributed = is_distributed_benchmark(baseline_path)
            if is_distributed and num_gpus == 1:
                skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
                logger.warning(f"    WARNING: {skip_reason}")
                result_entry['status'] = 'skipped'
                result_entry['error'] = skip_reason
                result_entry['skip_reason'] = skip_reason
                _append_benchmark_result(benchmark_results, result_entry)
                skipped_distributed += 1  # Count as skipped, not successful
                emit_event(
                    event_logger,
                    logger,
                    "example_skip",
                    chapter=chapter_name,
                    example=example_name,
                    example_type=example_type,
                    reason=skip_reason,
                )
                mark_progress(example_name)
                continue
        
            # Reset CUDA state before each benchmark pair (always, to prevent cascading failures)
            _reset_parent_execution_state()
            
            # Load and run baseline
            baseline_benchmark = load_benchmark(baseline_path)
            if baseline_benchmark is None:
                load_error = get_last_load_error() or ""
                skip_reason = check_hardware_limitation(load_error)
                if skip_reason:
                    result_entry['status'] = 'skipped'
                    result_entry['error'] = f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}'
                    result_entry['skip_reason'] = skip_reason
                    _append_benchmark_result(benchmark_results, result_entry)
                    skipped_hw += 1
                    emit_event(
                        event_logger,
                        logger,
                        "example_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        reason=skip_reason,
                        )
                else:
                    detail = load_error.strip().splitlines()[0] if load_error.strip() else ""
                    msg = "Failed to load baseline"
                    if detail:
                        msg = f"{msg}: {detail}"
                    result_entry['error'] = msg
                    _append_benchmark_result(benchmark_results, result_entry)
                    failed_error += 1
                    emit_event(
                        event_logger,
                        logger,
                        "example_error",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        error=msg,
                    )
                _reset_parent_execution_state()  # Reset after failure or skip
                mark_progress(example_name)
                continue

            try:
                _reap_benchmark_process_leftovers(
                    f"phase_start:{chapter_name}:{example_name}:baseline_timing",
                    current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                    current_owner_pid=os.getpid(),
                    repo_root=repo_root,
                )
                # Use benchmark_with_manifest for reproducibility
                run_id = f"{chapter_name}_{example_name}_baseline"
                baseline_phase_start = time.perf_counter()
                emit_event(
                    event_logger,
                    logger,
                    "phase_start",
                    chapter=chapter_name,
                    example=example_name,
                    example_type=example_type,
                    phase="baseline_timing",
                    variant="baseline",
                    target=baseline_path.name,
                )
                emit_progress(
                    "baseline_timing",
                    step=f"{chapter_name}:{example_name}",
                    step_detail="baseline timing",
                )
                baseline_run, baseline_config = _run_with_config(
                    baseline_benchmark,
                    run_id=run_id,
                    target_label=f"{chapter_name}:{example_name}",
                )
                baseline_result = baseline_run.result
                baseline_errors = list(getattr(baseline_result, "errors", None) or [])
                if baseline_errors:
                    skip_reason = None
                    for msg in baseline_errors:
                        upper = msg.upper()
                        if "SKIPPED" not in upper:
                            continue
                        if "SKIPPED:" in msg:
                            skip_reason = msg.split("SKIPPED:", 1)[1].strip()
                        else:
                            idx = upper.find("SKIPPED")
                            skip_reason = msg[idx:].strip() if idx != -1 else msg.strip()
                        break

                    error_message = baseline_errors[0].strip() if baseline_errors else "Benchmark harness reported errors"
                    if not skip_reason:
                        skip_reason = check_hardware_limitation(error_message)
                    if skip_reason:
                        logger.warning(f"    WARNING: SKIPPED: {skip_reason}")
                        result_entry["status"] = "skipped"
                        result_entry["error"] = f"SKIPPED: {skip_reason}"
                        result_entry["skip_reason"] = skip_reason
                        _append_benchmark_result(benchmark_results, result_entry)
                        skipped_hw += 1
                        emit_event(
                            event_logger,
                            logger,
                            "phase_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            phase="baseline_timing",
                            variant="baseline",
                            status="skipped",
                            duration_s=time.perf_counter() - baseline_phase_start,
                            error=skip_reason,
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "example_skip",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            reason=skip_reason,
                        )
                    else:
                        logger.error(f"    Baseline FAILED: {error_message}")
                        result_entry["status"] = "failed_error"
                        result_entry["error"] = error_message
                        _append_benchmark_result(benchmark_results, result_entry)
                        failed_error += 1
                        maybe_reset_gpu_for_error(error_message, f"{chapter_name}:{example_name}:baseline")
                        emit_event(
                            event_logger,
                            logger,
                            "phase_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            phase="baseline_timing",
                            variant="baseline",
                            status="failed",
                            duration_s=time.perf_counter() - baseline_phase_start,
                            error=error_message,
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "example_error",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            error=error_message,
                        )

                    _reset_parent_execution_state()
                    mark_progress(example_name)
                    continue
                _record_manifest(
                    baseline_run,
                    variant="baseline",
                    file_name=baseline_path.name,
                    target_label=f"{chapter_name}:{example_name}",
                )
                baseline_timing = baseline_result.timing
                baseline_memory = baseline_result.memory
                baseline_custom_metrics = getattr(baseline_result, "custom_metrics", None) or {}
                baseline_story_metadata = getattr(baseline_result, "story_metadata", None) or {}
                if not baseline_custom_metrics:
                    getter = getattr(baseline_benchmark, "get_custom_metrics", None)
                    if callable(getter):
                        try:
                            metrics = getter()
                            if isinstance(metrics, dict):
                                baseline_custom_metrics = metrics
                        except Exception:
                            baseline_custom_metrics = {}
                if not baseline_story_metadata:
                    getter = getattr(baseline_benchmark, "get_story_metadata", None)
                    if callable(getter):
                        try:
                            payload = getter()
                            if isinstance(payload, dict):
                                baseline_story_metadata = payload
                        except Exception:
                            baseline_story_metadata = {}
                baseline_time = baseline_timing.mean_ms if baseline_timing else 0.0
                result_entry['baseline_time_ms'] = baseline_time
                if baseline_custom_metrics:
                    result_entry['baseline_custom_metrics'] = baseline_custom_metrics
                if baseline_story_metadata:
                    result_entry['baseline_story_metadata'] = baseline_story_metadata
                
                # Capture baseline memory
                if baseline_memory and baseline_memory.peak_mb:
                    result_entry['baseline_memory_mb'] = baseline_memory.peak_mb
                
                # Enhanced baseline metrics display with emojis and formatting
                logger.info(f"    Baseline: {format_time_ms(baseline_time)} ms")
                if baseline_timing:
                    logger.info(f"      📊 Timing Stats: median={format_time_ms(baseline_timing.median_ms)}ms, "
                          f"min={format_time_ms(baseline_timing.min_ms)}ms, max={format_time_ms(baseline_timing.max_ms)}ms, "
                          f"std={format_time_ms(baseline_timing.std_ms)}ms")
                if baseline_memory and baseline_memory.peak_mb:
                    mem_str = f"      💾 Memory: peak={baseline_memory.peak_mb:.2f}MB"
                    if baseline_memory.allocated_mb:
                        mem_str += f", allocated={baseline_memory.allocated_mb:.2f}MB"
                    logger.info(mem_str)
                if baseline_timing and baseline_timing.percentiles:
                    p99 = baseline_timing.percentiles.get(99.0, 0)
                    p75 = baseline_timing.percentiles.get(75.0, 0)
                    p50 = baseline_timing.percentiles.get(50.0, baseline_timing.median_ms if baseline_timing else 0)
                    logger.info(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                    result_entry['baseline_percentiles'] = dict(baseline_timing.percentiles)
                    if p75 is not None:
                        result_entry['baseline_p75_ms'] = p75
                    p90 = baseline_timing.p90_ms or baseline_timing.percentiles.get(90.0)
                    if p90 is not None:
                        result_entry['baseline_p90_ms'] = p90
                baseline_throughput = baseline_result.throughput
                throughput_summary = format_throughput_summary(baseline_throughput)
                if throughput_summary:
                    logger.info(f"      ⚡ Throughput: {throughput_summary}")
                serialized_throughput = serialize_throughput(baseline_throughput)
                if serialized_throughput:
                    result_entry['baseline_throughput'] = serialized_throughput
                baseline_gpu_metrics = getattr(baseline_result, "gpu_metrics", None)
                if baseline_gpu_metrics:
                    result_entry['baseline_gpu_metrics'] = baseline_gpu_metrics
                    logger.info(f"      🌡️ GPU Telemetry: {format_gpu_telemetry(baseline_gpu_metrics)}")
                if "scenario_total_phase_ms" in baseline_custom_metrics:
                    logger.info(
                        f"      📐 Scenario phase sum: "
                        f"{baseline_custom_metrics['scenario_total_phase_ms']:.3f} ms"
                    )
                compile_error = baseline_custom_metrics.get("torch_compile_error")
                used_compile = baseline_custom_metrics.get("used_torch_compile")
                if compile_error:
                    logger.warning(f"      ⚠️ torch.compile fallback: {compile_error}")
                elif used_compile:
                    logger.info("      🚀 torch.compile enabled (reduce-overhead)")
                emit_event(
                    event_logger,
                    logger,
                    "phase_end",
                    chapter=chapter_name,
                    example=example_name,
                    example_type=example_type,
                    phase="baseline_timing",
                    variant="baseline",
                    status="succeeded",
                    duration_s=time.perf_counter() - baseline_phase_start,
                )
                emit_event(
                    event_logger,
                    logger,
                    "baseline_result",
                    chapter=chapter_name,
                    example=example_name,
                    example_type=example_type,
                    time_ms=baseline_time,
                    median_ms=baseline_timing.median_ms if baseline_timing else None,
                    min_ms=baseline_timing.min_ms if baseline_timing else None,
                    max_ms=baseline_timing.max_ms if baseline_timing else None,
                    std_ms=baseline_timing.std_ms if baseline_timing else None,
                    percentiles=baseline_timing.percentiles if baseline_timing else None,
                    p75_ms=result_entry.get("baseline_p75_ms"),
                    p90_ms=result_entry.get("baseline_p90_ms"),
                    throughput=serialized_throughput,
                    memory_mb=baseline_memory.peak_mb if baseline_memory else None,
                    gpu_metrics=baseline_gpu_metrics,
                    custom_metrics=baseline_custom_metrics,
                )

                # Capture baseline verification artifacts from the timing run (no re-execution).
                if verify_input or verify_output:
                    if not VERIFICATION_AVAILABLE:
                        result_entry["status"] = "failed_verification"
                        result_entry["error"] = "Verification system unavailable; cannot validate benchmark correctness"
                        _append_benchmark_result(benchmark_results, result_entry)
                        failed_error += 1
                        emit_event(
                            event_logger,
                            logger,
                            "example_error",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            error=result_entry["error"],
                        )
                        mark_progress(example_name)
                        _reset_parent_execution_state()
                        continue
                    try:
                        if verify_input:
                            baseline_signature = coerce_input_signature(_get_perf_signature(baseline_benchmark))
                            baseline_equivalence = get_signature_equivalence_spec(baseline_benchmark)
                        if verify_output:
                            baseline_verify_output = _get_perf_output(baseline_benchmark)
                            baseline_verify_tolerance = _get_perf_tolerance(baseline_benchmark)
                    except Exception as exc:
                        logger.error("    ✗ BASELINE VERIFICATION SETUP FAILED: %s", exc)
                        result_entry["status"] = "failed_verification"
                        result_entry["error"] = f"Baseline verification artifacts missing: {exc}"
                        _append_benchmark_result(benchmark_results, result_entry)
                        failed_error += 1
                        emit_event(
                            event_logger,
                            logger,
                            "example_error",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            error=result_entry["error"],
                        )
                        mark_progress(example_name)
                        _reset_parent_execution_state()
                        continue

                # Profile baseline if profiling is enabled (nsys, ncu, PyTorch)
                baseline_profile_paths: Dict[str, Optional[Path]] = {}
                example_profile_root: Optional[Path] = None
                example_profile_stem = slugify(example_name)
                if enable_profiling and profiling_output_dir:
                    example_profile_root = profiling_output_dir / example_profile_stem
                    baseline_profile_dir = example_profile_root / "baseline"
                    baseline_profile_dir.mkdir(parents=True, exist_ok=True)

                if enable_profiling and profiling_output_dir:
                    _reap_benchmark_process_leftovers(
                        f"phase_start:{chapter_name}:{example_name}:baseline_profiling",
                        current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                        current_owner_pid=os.getpid(),
                        repo_root=repo_root,
                    )
                    logger.info(f"    Profiling baseline...")
                    profiler_results = []
                    baseline_profiler_statuses: Dict[str, str] = {}
                    baseline_profiler_errors: Dict[str, str] = {}
                    baseline_metrics = {}
                    
                    # nsys profiling
                    if check_nsys_available():
                        emit_progress(
                            "baseline_nsys",
                            step=f"{chapter_name}:{example_name}",
                            step_detail="nsys profiling (baseline)",
                        )
                        logger.info(f"      nsys...")
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_start",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="nsys",
                            timeout_seconds=baseline_config.get_effective_timeout("nsys"),
                        )
                        nsys_path = profile_python_benchmark(
                            baseline_benchmark,
                            baseline_path,
                            chapter_dir,
                            baseline_profile_dir,
                            baseline_config,
                            variant="baseline",
                            output_stem=example_profile_stem,
                        )
                        nsys_error = _get_profile_failure_detail("nsys")
                        if nsys_path:
                            result_entry['baseline_nsys_rep'] = _repo_relative_path(nsys_path, repo_root)
                            profiler_results.append("nsys✓")
                            _set_profiler_status(baseline_profiler_statuses, "nsys", "succeeded")
                            baseline_profile_paths["nsys"] = nsys_path
                            # Extract metrics
                            nsys_metrics = extract_from_nsys_report(nsys_path)
                            if nsys_metrics:
                                baseline_metrics['nsys'] = nsys_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="nsys",
                                status="succeeded",
                                output_path=str(nsys_path),
                                metrics=nsys_metrics,
                            )
                        else:
                            profiler_results.append("nsys✗")
                            _set_profiler_status(baseline_profiler_statuses, "nsys", "failed")
                            nsys_error = _record_profiler_error(
                                baseline_profiler_errors,
                                "nsys",
                                nsys_error,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="nsys",
                                status="failed",
                                output_path=None,
                                metrics=None,
                                error=nsys_error,
                            )
                    else:
                        profiler_results.append("nsys-")
                        _set_profiler_status(baseline_profiler_statuses, "nsys", "skipped")
                        nsys_error = _record_profiler_error(
                            baseline_profiler_errors,
                            "nsys",
                            "Nsight Systems unavailable on current host.",
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="nsys",
                            status="skipped",
                            output_path=None,
                            metrics=None,
                            error=nsys_error,
                        )
                    
                    # ncu profiling
                    if check_ncu_available():
                        emit_progress(
                            "baseline_ncu",
                            step=f"{chapter_name}:{example_name}",
                            step_detail="ncu profiling (baseline)",
                        )
                        logger.info(f"ncu...")
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_start",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="ncu",
                            timeout_seconds=baseline_config.get_effective_timeout("ncu"),
                        )
                        ncu_path = profile_python_benchmark_ncu(
                            baseline_benchmark,
                            baseline_path,
                            chapter_dir,
                            baseline_profile_dir,
                            baseline_config,
                            variant="baseline",
                            output_stem=example_profile_stem,
                        )
                        ncu_error = _get_profile_failure_detail("ncu")
                        if ncu_path:
                            result_entry['baseline_ncu_rep'] = _repo_relative_path(ncu_path, repo_root)
                            profiler_results.append("ncu✓")
                            _set_profiler_status(baseline_profiler_statuses, "ncu", "succeeded")
                            baseline_profile_paths["ncu"] = ncu_path
                            # Extract metrics
                            ncu_metrics = extract_from_ncu_report(ncu_path)
                            if ncu_metrics:
                                baseline_metrics['ncu'] = ncu_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="ncu",
                                status="succeeded",
                                output_path=str(ncu_path),
                                metrics=ncu_metrics,
                            )
                        else:
                            profiler_results.append("ncu✗")
                            _set_profiler_status(baseline_profiler_statuses, "ncu", "failed")
                            ncu_error = _record_profiler_error(
                                baseline_profiler_errors,
                                "ncu",
                                ncu_error,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="ncu",
                                status="failed",
                                output_path=None,
                                metrics=None,
                                error=ncu_error,
                            )
                    else:
                        profiler_results.append("ncu-")
                        _set_profiler_status(baseline_profiler_statuses, "ncu", "skipped")
                        ncu_error = _record_profiler_error(
                            baseline_profiler_errors,
                            "ncu",
                            "Nsight Compute unavailable on current host.",
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="ncu",
                            status="skipped",
                            output_path=None,
                            metrics=None,
                            error=ncu_error,
                        )
                    
                    # PyTorch profiler
                    if TORCH_PROFILER_AVAILABLE:
                        emit_progress(
                            "baseline_torch",
                            step=f"{chapter_name}:{example_name}",
                            step_detail="torch profiling (baseline)",
                        )
                        logger.info(f"PyTorch...")
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_start",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="torch",
                            timeout_seconds=baseline_config.get_effective_timeout("torch"),
                        )
                        torch_path = profile_python_benchmark_torch(
                            baseline_benchmark,
                            baseline_path,
                            chapter_dir,
                            baseline_profile_dir,
                            variant="baseline",
                            output_stem=example_profile_stem,
                        )
                        torch_error = _get_profile_failure_detail("torch")
                        if torch_path:
                            result_entry['baseline_torch_trace'] = _repo_relative_path(torch_path, repo_root)
                            profiler_results.append("torch✓")
                            _set_profiler_status(baseline_profiler_statuses, "torch", "succeeded")
                            baseline_profile_paths["torch"] = torch_path
                            # Extract metrics
                            torch_metrics = extract_from_pytorch_trace(torch_path)
                            if torch_metrics:
                                baseline_metrics['torch'] = torch_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="torch",
                                status="succeeded",
                                output_path=str(torch_path),
                                metrics=torch_metrics,
                            )
                        else:
                            profiler_results.append("torch✗")
                            _set_profiler_status(baseline_profiler_statuses, "torch", "failed")
                            torch_error = _record_profiler_error(
                                baseline_profiler_errors,
                                "torch",
                                torch_error,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="baseline",
                                profiler="torch",
                                status="failed",
                                output_path=None,
                                metrics=None,
                                error=torch_error,
                            )
                    else:
                        profiler_results.append("torch-")
                        _set_profiler_status(baseline_profiler_statuses, "torch", "skipped")
                        torch_error = _record_profiler_error(
                            baseline_profiler_errors,
                            "torch",
                            "PyTorch profiler unavailable on current host.",
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            profiler="torch",
                            status="skipped",
                            output_path=None,
                            metrics=None,
                            error=torch_error,
                        )
                    
                    logger.info(f" ({', '.join(profiler_results)})")
                    result_entry["baseline_profiler_statuses"] = dict(baseline_profiler_statuses)
                    if baseline_profiler_errors:
                        result_entry["baseline_profiler_errors"] = dict(baseline_profiler_errors)
                    _reap_benchmark_process_leftovers(
                        f"{chapter_name}:{example_name}:baseline_profiling",
                        current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                        current_owner_pid=os.getpid(),
                        repo_root=repo_root,
                    )
                    
                    # Display extracted metrics
                    if baseline_metrics:
                        logger.info(f"      📈 Profiler Metrics:")
                        log_profiler_metrics_table(logger, baseline_metrics, indent="        ")
                        result_entry['baseline_profiler_metrics'] = baseline_metrics
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_summary",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            variant="baseline",
                            metrics=baseline_metrics,
                        )
            except Exception as e:
                error_str = str(e)
                skip_reason = check_hardware_limitation(error_str)
                
                if skip_reason:
                    result_entry['status'] = 'skipped'
                    result_entry['error'] = f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}'
                    result_entry['skip_reason'] = skip_reason
                    logger.warning(f"    WARNING: SKIPPED: {skip_reason}")
                    skipped_hw += 1
                    emit_event(
                        event_logger,
                        logger,
                        "example_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        reason=skip_reason,
                    )
                    if "baseline_phase_start" in locals():
                        emit_event(
                            event_logger,
                            logger,
                            "phase_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            phase="baseline_timing",
                            variant="baseline",
                            status="skipped",
                            duration_s=time.perf_counter() - baseline_phase_start,
                            error=skip_reason,
                        )
                else:
                    result_entry['error'] = f'Baseline execution failed: {error_str}'
                    failed_error += 1
                    maybe_reset_gpu_for_error(error_str, f"{chapter_name}:{example_name}:baseline")
                    emit_event(
                        event_logger,
                        logger,
                        "example_error",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        error=result_entry['error'],
                    )
                    if "baseline_phase_start" in locals():
                        emit_event(
                            event_logger,
                            logger,
                            "phase_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            phase="baseline_timing",
                            variant="baseline",
                            status="failed",
                            duration_s=time.perf_counter() - baseline_phase_start,
                            error=result_entry['error'],
                        )
                
                _append_benchmark_result(benchmark_results, result_entry)
                _reset_parent_execution_state()  # Reset after failure
                mark_progress(example_name)
                continue
            
            # When running the full benchmark suite with --profile minimal, profiling every
            # optimization variant can make runs prohibitively slow. Collect candidates
            # during timing, then profile only the best optimization after selection.
            optimized_profile_candidates: Dict[str, Dict[str, Any]] = {}

            # Test each optimization
            for optimized_path in optimized_paths:
                opt_name = optimized_path.name
                technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
                if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                    technique = 'default'
                
                # Check if optimized benchmark is distributed and we have only 1 GPU
                is_opt_distributed = is_distributed_benchmark(optimized_path)
                if is_opt_distributed and num_gpus == 1:
                    skip_reason = f"SKIPPED: Distributed benchmark requires multiple GPUs (found {num_gpus} GPU)"
                    logger.warning(f"    WARNING: {opt_name}: {skip_reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'error': skip_reason,
                    })
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        optimization_file=opt_name,
                        technique=technique,
                        reason=skip_reason,
                    )
                    continue
                
                optimized_benchmark = load_benchmark(optimized_path)
                
                # Capture optimization goal from the OPTIMIZED benchmark (not baseline)
                try:
                    if optimized_benchmark is not None:
                        opt_goal = optimized_benchmark.get_optimization_goal()
                        result_entry['optimization_goal'] = opt_goal
                        min_speedup = optimized_benchmark.get_min_speedup_for_success()
                        min_speedup_f = _coerce_finite_float(min_speedup)
                        if min_speedup_f is not None and min_speedup_f > 0:
                            result_entry['minimum_required_speedup'] = min_speedup_f
                except AttributeError:
                    pass  # Old benchmarks without get_optimization_goal()
                
                if optimized_benchmark is None:
                    load_error = get_last_load_error() or ""
                    skip_reason = check_hardware_limitation(load_error)
                    if skip_reason:
                        logger.warning(f"    Testing: {opt_name}... SKIPPED: {skip_reason}")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'skipped',
                            'error': f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}',
                            'skip_reason': skip_reason,
                        })
                        skipped_hw += 1
                        emit_event(
                            event_logger,
                            logger,
                            "optimization_skip",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            optimization_file=opt_name,
                            technique=technique,
                            reason=skip_reason,
                        )
                    else:
                        detail = load_error.strip().splitlines()[0] if load_error.strip() else ""
                        msg = "Failed to load"
                        if detail:
                            msg = f"{msg}: {detail}"
                        logger.error(f"    Testing: {opt_name}... FAILED (load)")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'failed_error',
                            'error': msg,
                        })
                        failed_error += 1
                        emit_event(
                            event_logger,
                            logger,
                            "optimization_error",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            optimization_file=opt_name,
                            technique=technique,
                            error=msg,
                        )
                    continue
                
                # NOTE: Verification now happens AFTER timing runs complete (see below)
                # This avoids running benchmarks twice - once for verification, once for timing
                
                try:
                    # Reset CUDA state before each optimized benchmark (always, to prevent cascading failures)
                    _reset_parent_execution_state()
                    _reap_benchmark_process_leftovers(
                        f"phase_start:{chapter_name}:{example_name}:{technique}:optimized_timing",
                        current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                        current_owner_pid=os.getpid(),
                        repo_root=repo_root,
                    )
                    
                    # Use benchmark_with_manifest for reproducibility
                    opt_run_id = f"{chapter_name}_{example_name}_optimized_{technique}"
                    opt_phase_start = time.perf_counter()
                    emit_event(
                        event_logger,
                        logger,
                        "phase_start",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        phase="optimized_timing",
                        variant="optimized",
                        technique=technique,
                        target=opt_name,
                    )
                    emit_progress(
                        "optimized_timing",
                        step=f"{chapter_name}:{example_name}",
                        step_detail=f"optimized timing ({technique})",
                    )
                    optimized_run, optimized_config = _run_with_config(
                        optimized_benchmark,
                        run_id=opt_run_id,
                        target_label=f"{chapter_name}:{example_name}",
                    )
                    optimized_result = optimized_run.result
                    optimized_errors = list(getattr(optimized_result, "errors", None) or [])
                    if optimized_errors:
                        skip_reason = None
                        for msg in optimized_errors:
                            upper = msg.upper()
                            if "SKIPPED" not in upper:
                                continue
                            if "SKIPPED:" in msg:
                                skip_reason = msg.split("SKIPPED:", 1)[1].strip()
                            else:
                                idx = upper.find("SKIPPED")
                                skip_reason = msg[idx:].strip() if idx != -1 else msg.strip()
                            break

                        error_message = optimized_errors[0].strip() if optimized_errors else "Benchmark harness reported errors"
                        if not skip_reason:
                            skip_reason = check_hardware_limitation(error_message)
                        if skip_reason:
                            logger.warning(f"    Testing: {opt_name}... SKIPPED: {skip_reason}")
                            result_entry["optimizations"].append({
                                "file": opt_name,
                                "technique": technique,
                                "status": "skipped",
                                "error": f"SKIPPED: {skip_reason}",
                                "skip_reason": skip_reason,
                            })
                            skipped_hw += 1
                            emit_event(
                                event_logger,
                                logger,
                                "phase_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                phase="optimized_timing",
                                variant="optimized",
                                technique=technique,
                                status="skipped",
                                duration_s=time.perf_counter() - opt_phase_start,
                                error=skip_reason,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "optimization_skip",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                optimization_file=opt_name,
                                technique=technique,
                                reason=skip_reason,
                            )
                        else:
                            logger.error(f"    Testing: {opt_name}... FAILED ({error_message})")
                            result_entry["optimizations"].append({
                                "file": opt_name,
                                "technique": technique,
                                "status": "failed_error",
                                "error": error_message,
                            })
                            failed_error += 1
                            maybe_reset_gpu_for_error(error_message, f"{chapter_name}:{example_name}:{opt_name}")
                            emit_event(
                                event_logger,
                                logger,
                                "phase_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                phase="optimized_timing",
                                variant="optimized",
                                technique=technique,
                                status="failed",
                                duration_s=time.perf_counter() - opt_phase_start,
                                error=error_message,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "optimization_error",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                optimization_file=opt_name,
                                technique=technique,
                                error=error_message,
                            )

                        _reset_parent_execution_state()
                        continue
                    _record_manifest(
                        optimized_run,
                        variant="optimized",
                        file_name=opt_name,
                        target_label=f"{chapter_name}:{example_name}",
                        technique=technique,
                    )
                    optimized_timing = optimized_result.timing
                    optimized_memory = optimized_result.memory
                    optimized_custom_metrics = getattr(optimized_result, "custom_metrics", None) or {}
                    optimized_story_metadata = getattr(optimized_result, "story_metadata", None) or {}
                    if not optimized_custom_metrics:
                        getter = getattr(optimized_benchmark, "get_custom_metrics", None)
                        if callable(getter):
                            try:
                                metrics = getter()
                                if isinstance(metrics, dict):
                                    optimized_custom_metrics = metrics
                            except Exception:
                                optimized_custom_metrics = {}
                    if not optimized_story_metadata:
                        getter = getattr(optimized_benchmark, "get_story_metadata", None)
                        if callable(getter):
                            try:
                                payload = getter()
                                if isinstance(payload, dict):
                                    optimized_story_metadata = payload
                            except Exception:
                                optimized_story_metadata = {}
                    optimized_time = optimized_timing.mean_ms if optimized_timing else 0.0
                    # Speedup is always derived from timing values (schema v2 integrity)
                    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                    emit_event(
                        event_logger,
                        logger,
                        "phase_end",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        phase="optimized_timing",
                        variant="optimized",
                        technique=technique,
                        status="succeeded",
                        duration_s=time.perf_counter() - opt_phase_start,
                    )

                    # Track scenario speedup separately in custom_metrics (not replacing timing speedup)
                    scenario_speedup = None
                    b_phase = (result_entry.get('baseline_custom_metrics') or {}).get("scenario_total_phase_ms")
                    o_phase = optimized_custom_metrics.get("scenario_total_phase_ms")
                    if b_phase and o_phase and o_phase > 0:
                        scenario_speedup = b_phase / o_phase
                        # Store as custom metric, don't override timing-based speedup
                        optimized_custom_metrics["custom_speedup"] = scenario_speedup
                    
                    # Enhanced metrics display with emojis and formatting
                    emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
                    logger.info(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")
                    
                    if optimized_timing:
                        logger.info(f"        📊 Timing: median={format_time_ms(optimized_timing.median_ms)}ms, "
                              f"min={format_time_ms(optimized_timing.min_ms)}ms, max={format_time_ms(optimized_timing.max_ms)}ms, "
                              f"std={format_time_ms(optimized_timing.std_ms)}ms")
                    
                    if optimized_memory and optimized_memory.peak_mb:
                        mem_change = ""
                        if baseline_memory and baseline_memory.peak_mb:
                            diff_mb = optimized_memory.peak_mb - baseline_memory.peak_mb
                            pct_change = (diff_mb / baseline_memory.peak_mb) * 100 if baseline_memory.peak_mb > 0 else 0
                            sign = "+" if diff_mb >= 0 else ""
                            mem_change = f" ({sign}{diff_mb:.2f}MB, {sign}{pct_change:.1f}%)"
                        
                        mem_str = f"        💾 Memory: peak={optimized_memory.peak_mb:.2f}MB{mem_change}"
                        logger.info(mem_str)
                        if optimized_memory.allocated_mb:
                            logger.info(f"                 allocated={optimized_memory.allocated_mb:.2f}MB")
                    
                    optimized_throughput = optimized_result.throughput
                    throughput_summary = format_throughput_summary(optimized_throughput)
                    throughput_payload = serialize_throughput(optimized_throughput)
                    if throughput_summary:
                        logger.info(f"        ⚡ Throughput: {throughput_summary}")
                    
                    if "scenario_total_phase_ms" in optimized_custom_metrics:
                        logger.info(
                            f"        📐 Scenario phase sum: "
                            f"{optimized_custom_metrics['scenario_total_phase_ms']:.3f} ms"
                        )
                    if scenario_speedup is not None:
                        logger.info(f"        📊 Scenario phase-sum speedup: {scenario_speedup:.2f}x")
                    opt_compile_error = optimized_custom_metrics.get("torch_compile_error")
                    opt_used_compile = optimized_custom_metrics.get("used_torch_compile")
                    if opt_compile_error:
                        logger.warning(f"        ⚠️ torch.compile fallback: {opt_compile_error}")
                    elif opt_used_compile:
                        logger.info("        🚀 torch.compile enabled (reduce-overhead)")
                    
                    opt_p75 = None
                    opt_p90 = None
                    if optimized_timing and optimized_timing.percentiles:
                        p99 = optimized_timing.percentiles.get(99.0, 0)
                        p75 = optimized_timing.percentiles.get(75.0, 0)
                        p50 = optimized_timing.percentiles.get(50.0, optimized_timing.median_ms if optimized_timing else 0)
                        opt_p75 = p75
                        opt_p90 = optimized_timing.p90_ms or optimized_timing.percentiles.get(90.0)
                        p99_speedup = ""
                        if baseline_timing and baseline_timing.percentiles and 99.0 in baseline_timing.percentiles:
                            p99_baseline = baseline_timing.percentiles[99.0]
                            if p99_baseline > 0:
                                p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                        logger.info(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                    
                    opt_gpu_metrics = getattr(optimized_result, "gpu_metrics", None)
                    if opt_gpu_metrics:
                        logger.info(f"        🌡️ GPU Telemetry: {format_gpu_telemetry(opt_gpu_metrics)}")
                    
                    # Visual speedup bar (always show for consistency)
                    bar_length = 40
                    if speedup > 1.0:
                        # Improvement: fill bar proportionally to speedup
                        filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        logger.info(f"        [{bar}] {speedup:.2f}x speedup")
                    elif speedup < 1.0:
                        # Regression: show how much slower (distance from 1.0)
                        regress_ratio = (1.0 - speedup)  # e.g., 0.93x = 0.07 (7% slower)
                        # Normalize: 0.5x (50% slower) = full bar, scale linearly
                        filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        logger.info(f"        [{bar}] {speedup:.2f}x slowdown")
                    else:
                        # No change
                        bar = "░" * bar_length
                        logger.info(f"        [{bar}] {speedup:.2f}x (no change)")
                    
                    opt_result = {
                        'file': opt_name,
                        'technique': technique,
                        'status': 'succeeded',
                        'time_ms': optimized_time,
                        'speedup': speedup,
                    }

                    # POST-TIMING VERIFICATION: validate workload equivalence + outputs using the timing-run artifacts.
                    if verify_input:
                        emit_progress(
                            "verification",
                            step=f"{chapter_name}:{example_name}",
                            step_detail="input verification",
                        )
                        try:
                            if baseline_signature is None:
                                raise RuntimeError("Baseline input signature missing")
                            optimized_signature = coerce_input_signature(_get_perf_signature(optimized_benchmark))
                            optimized_equivalence = get_signature_equivalence_spec(optimized_benchmark)
                            if baseline_equivalence != optimized_equivalence:
                                raise RuntimeError(
                                    "Signature equivalence mismatch: "
                                    f"baseline={baseline_equivalence} optimized={optimized_equivalence}"
                                )
                            workload_equiv = baseline_equivalence
                            baseline_workload = signature_workload_dict(baseline_signature, equivalence=workload_equiv)
                            optimized_workload = signature_workload_dict(optimized_signature, equivalence=workload_equiv)
                            mismatches = _diff_paths(baseline_workload, optimized_workload)
                            opt_result["input_verification"] = {
                                "passed": len(mismatches) == 0,
                                "mismatches": mismatches,
                                "equivalence": (
                                    {
                                        "group": workload_equiv.group,
                                        "ignore_fields": list(workload_equiv.ignore_fields),
                                    }
                                    if workload_equiv is not None
                                    else None
                                ),
                            }
                            if debug_signature_verify:
                                opt_result["input_verification"]["baseline_signature"] = baseline_signature.to_dict()
                                opt_result["input_verification"]["optimized_signature"] = optimized_signature.to_dict()
                                opt_result["input_verification"]["baseline_workload"] = baseline_workload
                                opt_result["input_verification"]["optimized_workload"] = optimized_workload
                            if mismatches:
                                raise RuntimeError(f"Input signature mismatch: {mismatches[0]}")
                        except Exception as exc:
                            opt_result["status"] = "failed_verification"
                            opt_result["error"] = f"Input verification failed: {exc}"

                    if verify_output and opt_result.get("status") == "succeeded":
                        emit_progress(
                            "verification",
                            step=f"{chapter_name}:{example_name}",
                            step_detail="output verification",
                        )
                        try:
                            if perf_compare_runner is None:
                                raise RuntimeError("Verification system unavailable")
                            if baseline_verify_output is None or baseline_verify_tolerance is None:
                                raise RuntimeError("Baseline verify_output/tolerance missing")
                            optimized_verify_output = _get_perf_output(optimized_benchmark)
                            comparison = perf_compare_runner.compare_perf_outputs(
                                baseline_verify_output,
                                optimized_verify_output,
                                baseline_verify_tolerance,
                            )
                            opt_result["verification"] = {
                                "passed": comparison.passed,
                                "max_diff": comparison.max_diff,
                                "location": comparison.location,
                                "rtol": baseline_verify_tolerance[0],
                                "atol": baseline_verify_tolerance[1],
                            }
                            if not comparison.passed:
                                reason = "Output mismatch"
                                if comparison.max_diff is not None:
                                    reason = f"Output mismatch (max_diff={comparison.max_diff:.6f})"
                                raise RuntimeError(reason)
                        except Exception as exc:
                            opt_result["status"] = "failed_verification"
                            opt_result["error"] = f"Output verification failed: {exc}"
                    
                    # Add memory metrics
                    if optimized_memory and optimized_memory.peak_mb:
                        opt_result['memory_mb'] = optimized_memory.peak_mb
                        # Calculate memory savings percentage
                        if baseline_memory and baseline_memory.peak_mb and baseline_memory.peak_mb > 0:
                            memory_savings_pct = ((baseline_memory.peak_mb - optimized_memory.peak_mb) 
                                                   / baseline_memory.peak_mb) * 100
                            opt_result['memory_savings_pct'] = memory_savings_pct
                            # Track best memory savings
                            if memory_savings_pct > result_entry.get('best_memory_savings_pct', 0):
                                result_entry['best_memory_savings_pct'] = memory_savings_pct
                    
                    if opt_p75 is not None:
                        opt_result['p75_ms'] = opt_p75
                    if opt_p90 is not None:
                        opt_result['p90_ms'] = opt_p90
                    if opt_gpu_metrics:
                        opt_result['gpu_metrics'] = opt_gpu_metrics
                    if optimized_custom_metrics:
                        opt_result['custom_metrics'] = optimized_custom_metrics
                    if optimized_story_metadata:
                        opt_result['story_metadata'] = optimized_story_metadata
                    if scenario_speedup is not None:
                        opt_result['scenario_speedup'] = scenario_speedup
                    if throughput_payload:
                        opt_result['throughput'] = throughput_payload
                    
                    # Profile optimized if profiling is enabled (nsys, ncu, PyTorch)
                    if (
                        enable_profiling
                        and profiling_output_dir
                        and str(profile_type).lower() == "minimal"
                        and opt_result.get("status") == "succeeded"
                    ):
                        optimized_profile_candidates[technique] = {
                            "benchmark": optimized_benchmark,
                            "path": optimized_path,
                            "config": optimized_config,
                            "result": opt_result,
                        }

                    if (
                        enable_profiling
                        and profiling_output_dir
                        and str(profile_type).lower() != "minimal"
                        and opt_result.get("status") == "succeeded"
                    ):
                        _reap_benchmark_process_leftovers(
                            f"phase_start:{chapter_name}:{example_name}:{technique}:optimized_profiling",
                            current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                            current_owner_pid=os.getpid(),
                            repo_root=repo_root,
                        )
                        pair_dir = _profile_pair_dir(
                            profile_output_root, chapter_id, example_name, technique
                        )
                        pair_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"\n    Profiling optimized...")
                        profiler_results = []
                        optimized_profiler_statuses: Dict[str, str] = {}
                        optimized_profiler_errors: Dict[str, str] = {}
                        optimized_metrics = {}
                        
                        # nsys profiling
                        if check_nsys_available():
                            emit_progress(
                                "optimized_nsys",
                                step=f"{chapter_name}:{example_name}",
                                step_detail=f"nsys profiling (optimized {technique})",
                            )
                            logger.info(f"      nsys...")
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_start",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="nsys",
                                technique=technique,
                                timeout_seconds=optimized_config.get_effective_timeout("nsys"),
                            )
                            nsys_path = profile_python_benchmark(
                                optimized_benchmark,
                                optimized_path,
                                chapter_dir,
                                pair_dir,
                                optimized_config,
                                variant="optimized",
                                output_stem=example_profile_stem,
                            )
                            nsys_error = _get_profile_failure_detail("nsys")
                            if nsys_path:
                                opt_result['optimized_nsys_rep'] = _repo_relative_path(nsys_path, repo_root)
                                profiler_results.append("nsys✓")
                                _set_profiler_status(optimized_profiler_statuses, "nsys", "succeeded")
                                # Extract metrics
                                nsys_metrics = extract_from_nsys_report(nsys_path)
                                if nsys_metrics:
                                    optimized_metrics['nsys'] = nsys_metrics
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=technique,
                                    status="succeeded",
                                    output_path=str(nsys_path),
                                    metrics=nsys_metrics,
                                )
                            else:
                                profiler_results.append("nsys✗")
                                _set_profiler_status(optimized_profiler_statuses, "nsys", "failed")
                                nsys_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "nsys",
                                    nsys_error,
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=technique,
                                    status="failed",
                                    output_path=None,
                                    metrics=None,
                                    error=nsys_error,
                                )
                        else:
                            profiler_results.append("nsys-")
                            _set_profiler_status(optimized_profiler_statuses, "nsys", "skipped")
                            nsys_error = _record_profiler_error(
                                optimized_profiler_errors,
                                "nsys",
                                "Nsight Systems unavailable on current host.",
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="nsys",
                                technique=technique,
                                status="skipped",
                                output_path=None,
                                metrics=None,
                                error=nsys_error,
                            )
                        
                        # ncu profiling
                        if check_ncu_available():
                            emit_progress(
                                "optimized_ncu",
                                step=f"{chapter_name}:{example_name}",
                                step_detail=f"ncu profiling (optimized {technique})",
                            )
                            logger.info(f"ncu...")
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_start",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="ncu",
                                technique=technique,
                                timeout_seconds=optimized_config.get_effective_timeout("ncu"),
                            )
                            ncu_path = profile_python_benchmark_ncu(
                                optimized_benchmark,
                                optimized_path,
                                chapter_dir,
                                pair_dir,
                                optimized_config,
                                variant="optimized",
                                output_stem=example_profile_stem,
                            )
                            ncu_error = _get_profile_failure_detail("ncu")
                            if ncu_path:
                                opt_result['optimized_ncu_rep'] = _repo_relative_path(ncu_path, repo_root)
                                profiler_results.append("ncu✓")
                                _set_profiler_status(optimized_profiler_statuses, "ncu", "succeeded")
                                # Extract metrics
                                ncu_metrics = extract_from_ncu_report(ncu_path)
                                if ncu_metrics:
                                    optimized_metrics['ncu'] = ncu_metrics
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=technique,
                                    status="succeeded",
                                    output_path=str(ncu_path),
                                    metrics=ncu_metrics,
                                )
                            else:
                                profiler_results.append("ncu✗")
                                _set_profiler_status(optimized_profiler_statuses, "ncu", "failed")
                                ncu_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "ncu",
                                    ncu_error,
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=technique,
                                    status="failed",
                                    output_path=None,
                                    metrics=None,
                                    error=ncu_error,
                                )
                        else:
                            profiler_results.append("ncu-")
                            _set_profiler_status(optimized_profiler_statuses, "ncu", "skipped")
                            ncu_error = _record_profiler_error(
                                optimized_profiler_errors,
                                "ncu",
                                "Nsight Compute unavailable on current host.",
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="ncu",
                                technique=technique,
                                status="skipped",
                                output_path=None,
                                metrics=None,
                                error=ncu_error,
                            )
                        
                        # PyTorch profiler
                        if TORCH_PROFILER_AVAILABLE:
                            emit_progress(
                                "optimized_torch",
                                step=f"{chapter_name}:{example_name}",
                                step_detail=f"torch profiling (optimized {technique})",
                            )
                            logger.info(f"PyTorch...")
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_start",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="torch",
                                technique=technique,
                                timeout_seconds=optimized_config.get_effective_timeout("torch"),
                            )
                            torch_path = profile_python_benchmark_torch(
                                optimized_benchmark,
                                optimized_path,
                                chapter_dir,
                                pair_dir,
                                variant="optimized",
                                output_stem=example_profile_stem,
                            )
                            torch_error = _get_profile_failure_detail("torch")
                            if torch_path:
                                opt_result['optimized_torch_trace'] = _repo_relative_path(torch_path, repo_root)
                                profiler_results.append("torch✓")
                                _set_profiler_status(optimized_profiler_statuses, "torch", "succeeded")
                                # Extract metrics
                                torch_metrics = extract_from_pytorch_trace(torch_path)
                                if torch_metrics:
                                    optimized_metrics['torch'] = torch_metrics
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="torch",
                                    technique=technique,
                                    status="succeeded",
                                    output_path=str(torch_path),
                                    metrics=torch_metrics,
                                )
                            else:
                                profiler_results.append("torch✗")
                                _set_profiler_status(optimized_profiler_statuses, "torch", "failed")
                                torch_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "torch",
                                    torch_error,
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="torch",
                                    technique=technique,
                                    status="failed",
                                    output_path=None,
                                    metrics=None,
                                    error=torch_error,
                                )
                        else:
                            profiler_results.append("torch-")
                            _set_profiler_status(optimized_profiler_statuses, "torch", "skipped")
                            torch_error = _record_profiler_error(
                                optimized_profiler_errors,
                                "torch",
                                "PyTorch profiler unavailable on current host.",
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                profiler="torch",
                                technique=technique,
                                status="skipped",
                                output_path=None,
                                metrics=None,
                                error=torch_error,
                            )
                        
                        logger.info(f" ({', '.join(profiler_results)})")
                        opt_result["optimized_profiler_statuses"] = dict(optimized_profiler_statuses)
                        if optimized_profiler_errors:
                            opt_result["optimized_profiler_errors"] = dict(optimized_profiler_errors)
                        _reap_benchmark_process_leftovers(
                            f"{chapter_name}:{example_name}:{technique}:optimized_profiling",
                            current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                            current_owner_pid=os.getpid(),
                            repo_root=repo_root,
                        )
                        
                        # Display extracted metrics
                        if optimized_metrics:
                            logger.info(f"        📈 Profiler Metrics:")
                            log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
                            opt_result['optimized_profiler_metrics'] = optimized_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_summary",
                                chapter=chapter_name,
                                example=example_name,
                                example_type=example_type,
                                variant="optimized",
                                technique=technique,
                                metrics=optimized_metrics,
                            )
                    
                    result_entry['optimizations'].append(opt_result)
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_result",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        optimization_file=opt_name,
                        technique=technique,
                        status=opt_result.get("status"),
                        time_ms=opt_result.get("time_ms"),
                        speedup=opt_result.get("speedup"),
                        memory_mb=opt_result.get("memory_mb"),
                        memory_savings_pct=opt_result.get("memory_savings_pct"),
                        p75_ms=opt_result.get("p75_ms"),
                        p90_ms=opt_result.get("p90_ms"),
                        throughput=opt_result.get("throughput"),
                        gpu_metrics=opt_result.get("gpu_metrics"),
                        custom_metrics=opt_result.get("custom_metrics"),
                        profiler_metrics=opt_result.get("optimized_profiler_metrics"),
                        verification=opt_result.get("verification"),
                        input_verification=opt_result.get("input_verification"),
                    )
                    
                    if opt_result.get("status") == "succeeded" and speedup > result_entry['best_speedup']:
                        result_entry['best_speedup'] = speedup
                        speedups.append(speedup)
                    
                except Exception as e:
                    # Get comprehensive error information with timeout protection
                    def safe_get_error_str(exc, timeout_sec=1):
                        """Safely get error string with timeout to prevent hangs."""
                        error_parts = {"type": type(exc).__name__, "str": None, "repr": None}
                        
                        def get_str():
                            try:
                                error_parts["str"] = str(exc)
                            except Exception as str_exc:
                                error_parts["str"] = (
                                    f"<str() failed: {type(str_exc).__name__}: {str_exc}>"
                                )

                        def get_repr():
                            try:
                                error_parts["repr"] = repr(exc)
                            except Exception as repr_exc:
                                error_parts["repr"] = (
                                    f"<repr() failed: {type(repr_exc).__name__}: {repr_exc}>"
                                )
                        
                        # Try to get string representation with timeout
                        import threading
                        t1 = threading.Thread(target=get_str, daemon=True)
                        t2 = threading.Thread(target=get_repr, daemon=True)
                        t1.start()
                        t2.start()
                        t1.join(timeout=timeout_sec)
                        t2.join(timeout=timeout_sec)
                        
                        # Use best available representation
                        if error_parts["str"]:
                            return error_parts["str"]
                        elif error_parts["repr"]:
                            return error_parts["repr"]
                        else:
                            return error_parts["type"]
                    
                    error_str = safe_get_error_str(e)
                    error_full = f"{type(e).__name__}: {error_str}" if error_str else type(e).__name__
                    
                    # If error string is suspiciously short or empty, try to get more info
                    if not error_str or len(error_str.strip()) < 3:
                        import traceback
                        try:
                            tb_lines = traceback.format_exception_only(type(e), e)
                            if tb_lines:
                                error_full = tb_lines[-1].strip()
                                error_str = error_full
                        except Exception:
                            # If even traceback fails, use minimal info
                            error_full = f"{type(e).__name__}: (error message unavailable)"
                    
                    skip_reason = check_hardware_limitation(error_full)
                    
                    if skip_reason:
                        logger.warning(f"    Testing: {opt_name}... WARNING: SKIPPED: {skip_reason}")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'skipped',
                            'error': f'HARDWARE/SOFTWARE LIMITATION: {skip_reason}',
                            'skip_reason': skip_reason,
                        })
                        skipped_hw += 1
                    else:
                        # Format error message: show full error but truncate if extremely long
                        if len(error_full) > 200:
                            # Try to truncate at word boundary for very long errors
                            truncated = error_full[:197]
                            last_space = truncated.rfind(' ')
                            if last_space > 150:
                                truncated = truncated[:last_space]
                            truncated += "..."
                            logger.error(f"    Testing: {opt_name}... FAILED ({truncated})")
                            logger.error(f"        Full error: {error_full}")
                        else:
                            logger.error(f"    Testing: {opt_name}... FAILED ({error_full})")
                        result_entry['optimizations'].append({
                            'file': opt_name,
                            'technique': technique,
                            'status': 'failed_error',
                            'error': error_full,  # Store full error with type
                        })
                        maybe_reset_gpu_for_error(error_full, f"{chapter_name}:{example_name}:{opt_name}")
                    if "opt_phase_start" in locals():
                        emit_event(
                            event_logger,
                            logger,
                            "phase_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type=example_type,
                            phase="optimized_timing",
                            variant="optimized",
                            technique=technique,
                            status="skipped" if skip_reason else "failed",
                            duration_s=time.perf_counter() - opt_phase_start,
                            error=skip_reason or error_full,
                        )
                    
                    _reset_parent_execution_state()  # Reset after failure
            
            if result_entry['status'] == 'skipped':
                _append_benchmark_result(benchmark_results, result_entry)
                continue
    
            baseline_ok = result_entry.get('baseline_time_ms') is not None
            optimizations = result_entry.get('optimizations', [])
            has_success = any(opt.get('status') == 'succeeded' for opt in optimizations)
            all_skipped_opt = bool(optimizations) and all(opt.get('status') == 'skipped' for opt in optimizations)
            any_failed_verification = any(opt.get('status') == 'failed_verification' for opt in optimizations)
            any_failed_error_opt = any(opt.get('status') == 'failed_error' for opt in optimizations)
    
            update_result = None
            if baseline_ok and has_success:
                example_key = expectation_example_key(result_entry['example'], result_entry.get('type', 'python'))
                optimization_goal = (result_entry.get("optimization_goal") or "speed").strip().lower()
                best_opt = select_best_optimization(result_entry.get("optimizations", []), goal=optimization_goal)
                if (
                    enable_profiling
                    and profiling_output_dir
                    and str(profile_type).lower() == "minimal"
                    and best_opt
                    and isinstance(best_opt, dict)
                    and best_opt.get("status") == "succeeded"
                ):
                    best_key = str(best_opt.get("technique") or best_opt.get("file") or "")
                    cand = optimized_profile_candidates.get(best_key)
                    if cand:
                        try:
                            _reap_benchmark_process_leftovers(
                                f"phase_start:{chapter_name}:{example_name}:{best_key}:optimized_profiling",
                                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                                current_owner_pid=os.getpid(),
                                repo_root=repo_root,
                            )
                            pair_dir = _profile_pair_dir(
                                profile_output_root, chapter_id, example_name, best_key
                            )
                            pair_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"\n    Profiling optimized (best only: {best_key})...")
                            profiler_results = []
                            optimized_profiler_statuses: Dict[str, str] = {}
                            optimized_profiler_errors: Dict[str, str] = {}
                            optimized_metrics = {}

                            optimized_benchmark = cand.get("benchmark")
                            optimized_path = cand.get("path")
                            optimized_config = cand.get("config")

                            # nsys profiling
                            if check_nsys_available():
                                emit_progress(
                                    "optimized_nsys",
                                    step=f"{chapter_name}:{example_name}",
                                    step_detail=f"nsys profiling (optimized {best_key})",
                                )
                                logger.info("      nsys...")
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_start",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=best_key,
                                    timeout_seconds=optimized_config.get_effective_timeout("nsys") if optimized_config else None,
                                )
                                nsys_path = profile_python_benchmark(
                                    optimized_benchmark,
                                    optimized_path,
                                    chapter_dir,
                                    pair_dir,
                                    optimized_config,
                                    variant="optimized",
                                    output_stem=example_profile_stem,
                                )
                                nsys_error = _get_profile_failure_detail("nsys")
                                if nsys_path:
                                    best_opt["optimized_nsys_rep"] = _repo_relative_path(nsys_path, repo_root)
                                    profiler_results.append("nsys✓")
                                    _set_profiler_status(optimized_profiler_statuses, "nsys", "succeeded")
                                    nsys_metrics = extract_from_nsys_report(nsys_path)
                                    if nsys_metrics:
                                        optimized_metrics["nsys"] = nsys_metrics
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="nsys",
                                        technique=best_key,
                                        status="succeeded",
                                        output_path=str(nsys_path),
                                        metrics=nsys_metrics,
                                    )
                                else:
                                    profiler_results.append("nsys✗")
                                    _set_profiler_status(optimized_profiler_statuses, "nsys", "failed")
                                    nsys_error = _record_profiler_error(
                                        optimized_profiler_errors,
                                        "nsys",
                                        nsys_error,
                                    )
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="nsys",
                                        technique=best_key,
                                        status="failed",
                                        output_path=None,
                                        metrics=None,
                                        error=nsys_error,
                                    )
                            else:
                                profiler_results.append("nsys-")
                                _set_profiler_status(optimized_profiler_statuses, "nsys", "skipped")
                                nsys_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "nsys",
                                    "Nsight Systems unavailable on current host.",
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=best_key,
                                    status="skipped",
                                    output_path=None,
                                    metrics=None,
                                    error=nsys_error,
                                )

                            # ncu profiling
                            if check_ncu_available():
                                emit_progress(
                                    "optimized_ncu",
                                    step=f"{chapter_name}:{example_name}",
                                    step_detail=f"ncu profiling (optimized {best_key})",
                                )
                                logger.info("ncu...")
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_start",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=best_key,
                                    timeout_seconds=optimized_config.get_effective_timeout("ncu") if optimized_config else None,
                                )
                                ncu_path = profile_python_benchmark_ncu(
                                    optimized_benchmark,
                                    optimized_path,
                                    chapter_dir,
                                    pair_dir,
                                    optimized_config,
                                    variant="optimized",
                                    output_stem=example_profile_stem,
                                )
                                ncu_error = _get_profile_failure_detail("ncu")
                                if ncu_path:
                                    best_opt["optimized_ncu_rep"] = _repo_relative_path(ncu_path, repo_root)
                                    profiler_results.append("ncu✓")
                                    _set_profiler_status(optimized_profiler_statuses, "ncu", "succeeded")
                                    ncu_metrics = extract_from_ncu_report(ncu_path)
                                    if ncu_metrics:
                                        optimized_metrics["ncu"] = ncu_metrics
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="ncu",
                                        technique=best_key,
                                        status="succeeded",
                                        output_path=str(ncu_path),
                                        metrics=ncu_metrics,
                                    )
                                else:
                                    profiler_results.append("ncu✗")
                                    _set_profiler_status(optimized_profiler_statuses, "ncu", "failed")
                                    ncu_error = _record_profiler_error(
                                        optimized_profiler_errors,
                                        "ncu",
                                        ncu_error,
                                    )
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="ncu",
                                        technique=best_key,
                                        status="failed",
                                        output_path=None,
                                        metrics=None,
                                        error=ncu_error,
                                    )
                            else:
                                profiler_results.append("ncu-")
                                _set_profiler_status(optimized_profiler_statuses, "ncu", "skipped")
                                ncu_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "ncu",
                                    "Nsight Compute unavailable on current host.",
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=best_key,
                                    status="skipped",
                                    output_path=None,
                                    metrics=None,
                                    error=ncu_error,
                                )

                            # PyTorch profiler
                            if TORCH_PROFILER_AVAILABLE:
                                emit_progress(
                                    "optimized_torch",
                                    step=f"{chapter_name}:{example_name}",
                                    step_detail=f"torch profiling (optimized {best_key})",
                                )
                                logger.info("PyTorch...")
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_start",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="torch",
                                    technique=best_key,
                                    timeout_seconds=optimized_config.get_effective_timeout("torch") if optimized_config else None,
                                )
                                torch_path = profile_python_benchmark_torch(
                                    optimized_benchmark,
                                    optimized_path,
                                    chapter_dir,
                                    pair_dir,
                                    variant="optimized",
                                    output_stem=example_profile_stem,
                                )
                                torch_error = _get_profile_failure_detail("torch")
                                if torch_path:
                                    best_opt["optimized_torch_trace"] = _repo_relative_path(torch_path, repo_root)
                                    profiler_results.append("torch✓")
                                    _set_profiler_status(optimized_profiler_statuses, "torch", "succeeded")
                                    torch_metrics = extract_from_pytorch_trace(torch_path)
                                    if torch_metrics:
                                        optimized_metrics["torch"] = torch_metrics
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="torch",
                                        technique=best_key,
                                        status="succeeded",
                                        output_path=str(torch_path),
                                        metrics=torch_metrics,
                                    )
                                else:
                                    profiler_results.append("torch✗")
                                    _set_profiler_status(optimized_profiler_statuses, "torch", "failed")
                                    torch_error = _record_profiler_error(
                                        optimized_profiler_errors,
                                        "torch",
                                        torch_error,
                                    )
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type=example_type,
                                        variant="optimized",
                                        profiler="torch",
                                        technique=best_key,
                                        status="failed",
                                        output_path=None,
                                        metrics=None,
                                        error=torch_error,
                                    )
                            else:
                                profiler_results.append("torch-")
                                _set_profiler_status(optimized_profiler_statuses, "torch", "skipped")
                                torch_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "torch",
                                    "PyTorch profiler unavailable on current host.",
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    profiler="torch",
                                    technique=best_key,
                                    status="skipped",
                                    output_path=None,
                                    metrics=None,
                                    error=torch_error,
                                )

                            logger.info(f" ({', '.join(profiler_results)})")
                            best_opt["optimized_profiler_statuses"] = dict(optimized_profiler_statuses)
                            if optimized_profiler_errors:
                                best_opt["optimized_profiler_errors"] = dict(optimized_profiler_errors)
                            _reap_benchmark_process_leftovers(
                                f"{chapter_name}:{example_name}:{best_key}:optimized_profiling",
                                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                                current_owner_pid=os.getpid(),
                                repo_root=repo_root,
                            )
                            if optimized_metrics:
                                logger.info("        📈 Profiler Metrics:")
                                log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
                                best_opt["optimized_profiler_metrics"] = optimized_metrics
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_summary",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type=example_type,
                                    variant="optimized",
                                    technique=best_key,
                                    metrics=optimized_metrics,
                                )
                        except Exception:
                            logger.warning("    WARNING: Best-only profiling failed", exc_info=True)
                if best_opt:
                    emit_event(
                        event_logger,
                        logger,
                        "best_optimization",
                        chapter=chapter_name,
                        example=example_name,
                        example_type=example_type,
                        optimization_goal=optimization_goal,
                        technique=best_opt.get("technique") or best_opt.get("file"),
                        optimization_file=best_opt.get("file"),
                        speedup=best_opt.get("speedup"),
                        time_ms=best_opt.get("time_ms"),
                        memory_mb=best_opt.get("memory_mb"),
                        throughput=best_opt.get("throughput"),
                    )
                    if verify_input and isinstance(best_opt.get("input_verification"), dict):
                        result_entry["input_verification"] = best_opt.get("input_verification")
                    if verify_output and isinstance(best_opt.get("verification"), dict):
                        result_entry["verification"] = best_opt.get("verification")

                if best_opt:
                    emit_progress(
                        "expectations",
                        step=f"{chapter_name}:{example_name}",
                        step_detail="expectations update",
                    )
                    old_entry = expectations_store.get_entry(example_key)
                    provenance = RunProvenance(
                        git_commit=git_commit or "unknown",
                        hardware_key=expectation_hardware_key,
                        profile_name=profile_type,
                        timestamp=datetime.now().isoformat(),
                        iterations=int(iterations),
                        warmup_iterations=int(warmup),
                        execution_environment=execution_environment.kind,
                        validity_profile=validity_profile,
                        dmi_product_name=execution_environment.dmi_product_name,
                    )
                    entry = ExpectationEntry(
                        example=result_entry.get("example", example_key),
                        type=result_entry.get("type", "python"),
                        optimization_goal=result_entry.get("optimization_goal", "speed"),
                        baseline_time_ms=float(result_entry.get("baseline_time_ms") or 0.0),
                        best_optimized_time_ms=float(best_opt.get("time_ms") or 0.0),
                        provenance=provenance,
                        baseline_memory_mb=result_entry.get("baseline_memory_mb"),
                        best_optimized_memory_mb=best_opt.get("memory_mb"),
                        baseline_p75_ms=result_entry.get("baseline_p75_ms"),
                        baseline_p90_ms=result_entry.get("baseline_p90_ms"),
                        best_optimized_p75_ms=best_opt.get("p75_ms"),
                        best_optimized_p90_ms=best_opt.get("p90_ms"),
                        baseline_throughput=result_entry.get("baseline_throughput"),
                        best_optimized_throughput=best_opt.get("throughput"),
                        best_optimization_name=best_opt.get("technique") or best_opt.get("file"),
                        best_optimization_file=best_opt.get("file"),
                        best_optimization_technique=best_opt.get("technique"),
                    )
                    update_result = None
                    if expectation_validation_enabled:
                        active_expectations_store = expectations_store
                        if not expectation_writes_enabled:
                            active_expectations_store = ExpectationsStore(
                                chapter_dir,
                                expectation_hardware_key,
                                accept_regressions=accept_regressions or update_expectations,
                                allow_mixed_provenance=allow_mixed_provenance_for_writes,
                            )
                        update_result = active_expectations_store.update_entry(example_key, entry)
                        try:
                            result_entry["expectation"] = update_result.to_dict()
                        except Exception:
                            result_entry["expectation"] = {
                                "status": update_result.status,
                                "message": update_result.message,
                                "validation_issues": [issue.to_dict() for issue in update_result.validation_issues],
                            }
                        if expectation_writes_enabled:
                            logger.info("    Expectations: %s", update_result.message)
                        else:
                            result_entry["expectation"]["persisted"] = False
                            result_entry["expectation"]["message"] = (
                                f"{update_result.message} (preview only; rerun with "
                                "--update-expectations/--accept-regressions/--allow-mixed-provenance "
                                "to write the file.)"
                            )
                            logger.info("    Expectations (preview): %s", update_result.message)
                        log_expectation_delta(
                            logger,
                            example_key=example_key,
                            goal=optimization_goal,
                            old_entry=old_entry,
                            new_entry=entry,
                            update_result=update_result,
                            event_logger=event_logger,
                            chapter=chapter_name,
                        )
                    else:
                        result_entry["expectation"] = {
                            "status": "skipped",
                            "message": (
                                "Expectation validation is disabled in portable validity profile. "
                                "Enable --allow-portable-expectations-update "
                                "(allow_portable_expectations_update=True) to validate and write expectation files."
                            ),
                            "validation_issues": [],
                        }

                is_rejected_regression = bool(
                    update_result
                    and update_result.status == "rejected"
                    and any(issue.issue_type == "regression" for issue in update_result.validation_issues)
                )
                profiler_failures = _collect_required_profiler_failures(
                    result_entry,
                    best_opt,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                profiler_failure_details = _collect_required_profiler_failure_details(
                    result_entry,
                    best_opt,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                if profiler_failures:
                    result_entry["status"] = "failed_profiler"
                    result_entry["error"] = _format_required_profiler_failure(
                        profiler_failures,
                        failure_details=profiler_failure_details,
                    )
                    logger.warning("    WARNING: %s", result_entry["error"])
                    failed_error += 1
                elif is_rejected_regression:
                    regression_metrics = None
                    if best_opt and isinstance(best_opt, dict):
                        regression_metrics = best_opt.get("gpu_metrics")
                    if not regression_metrics:
                        regression_metrics = result_entry.get("baseline_gpu_metrics")
                    if regression_metrics:
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(regression_metrics))
                        temp = regression_metrics.get("temperature_gpu_c")
                        if temp is not None and temp >= 85:
                            logger.warning("    ⚠️ GPU temperature %.1f°C exceeds recommended threshold; consider cooling or resetting before re-running.", temp)
                    else:
                        live_metrics = _query_gpu_telemetry_for_profile(validity_profile)
                        logger.warning("    🌡️ GPU telemetry during regression: %s", format_gpu_telemetry(live_metrics))
                    result_entry['status'] = 'failed_regression'
                    result_entry["error"] = update_result.message if update_result else "Expectation regression detected"
                    failed_regression += 1
                elif _should_fail_no_speedup(result_entry):
                    result_entry["status"] = "failed_no_speedup"
                    result_entry["error"] = _format_failed_no_speedup(result_entry)
                    failed_no_speedup += 1
                else:
                    result_entry['status'] = 'succeeded'
                    successful += 1
            elif baseline_ok and (all_skipped_opt or not optimizations):
                profiler_failures = _collect_required_profiler_failures(
                    result_entry,
                    None,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                profiler_failure_details = _collect_required_profiler_failure_details(
                    result_entry,
                    None,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                if profiler_failures:
                    result_entry["status"] = "failed_profiler"
                    result_entry["error"] = _format_required_profiler_failure(
                        profiler_failures,
                        failure_details=profiler_failure_details,
                    )
                    logger.warning("    WARNING: %s", result_entry["error"])
                    failed_error += 1
                elif all_skipped_opt:
                    skip_reason = _collect_all_skipped_optimization_reason(optimizations)
                    if skip_reason:
                        result_entry["status"] = "skipped"
                        result_entry["error"] = f"SKIPPED: {skip_reason}"
                        result_entry["skip_reason"] = skip_reason
                    elif _should_fail_no_speedup(result_entry):
                        result_entry["status"] = "failed_no_speedup"
                        result_entry["error"] = _format_failed_no_speedup(result_entry)
                        failed_no_speedup += 1
                    else:
                        result_entry['status'] = 'succeeded'
                        successful += 1
                elif _should_fail_no_speedup(result_entry):
                    result_entry["status"] = "failed_no_speedup"
                    result_entry["error"] = _format_failed_no_speedup(result_entry)
                    failed_no_speedup += 1
                else:
                    result_entry['status'] = 'succeeded'
                    successful += 1
            elif baseline_ok and (not has_success) and any_failed_verification and (not any_failed_error_opt):
                result_entry['status'] = 'failed_verification'
                result_entry['error'] = result_entry.get('error') or 'No optimizations passed verification'
                failed_error += 1
            else:
                result_entry['status'] = 'failed_error'
                if not result_entry.get('error'):
                    result_entry['error'] = _GENERIC_BENCHMARK_FAILURE_ERROR
                failed_error += 1
            
            emit_event(
                event_logger,
                logger,
                "example_end",
                chapter=chapter_name,
                example=example_name,
                example_type=example_type,
                status=result_entry.get("status"),
                best_speedup=result_entry.get("best_speedup"),
                best_memory_savings_pct=result_entry.get("best_memory_savings_pct"),
                optimization_goal=result_entry.get("optimization_goal"),
                error=result_entry.get("error"),
            )
            _append_benchmark_result(benchmark_results, result_entry)
            mark_progress(example_name)
            
            # Reset CUDA state after each benchmark pair (always, to ensure clean state)
            _reset_parent_execution_state()
        
        # Process CUDA benchmarks
        for baseline_cu_path, optimized_cu_paths, example_name in cuda_pairs:
            logger.info(f"\n  Example (CUDA): {example_name}")
            emit_event(
                event_logger,
                logger,
                "example_start",
                chapter=chapter_name,
                example=example_name,
                example_type="cuda",
                baseline_file=baseline_cu_path.name,
                optimized_files=[p.name for p in optimized_cu_paths],
            )

            if is_informational_example(chapter_name, example_name):
                informational_skipped += 1
                logger.info("    ℹ️ Informational systems demo - documented for reference, not benchmarked.")
                emit_event(
                    event_logger,
                    logger,
                    "example_skip",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    reason="informational_demo",
                )
                mark_progress(example_name)
                continue

            _reap_benchmark_process_leftovers(
                f"example_start:{chapter_name}:{example_name}:cuda",
                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                current_owner_pid=os.getpid(),
                repo_root=repo_root,
            )

            result_entry = {
                'example': example_name,
                'baseline_file': baseline_cu_path.name,
                'type': 'cuda',
                'baseline_time_ms': None,
                'baseline_throughput': None,
                'baseline_profiler_statuses': {},
                'optimizations': [],
                'best_speedup': 1.0,
                'status': 'failed_error',
                'error': None,
            }

            # Find baseline executable
            baseline_executable = find_cuda_executable(baseline_cu_path, chapter_dir)
            if baseline_executable is None:
                reason = determine_cuda_skip_reason(
                    baseline_cu_path, chapter_dir, cuda_build_ok, cuda_build_warning
                )
                logger.warning(
                    f"    Baseline executable {baseline_cu_path.name} SKIPPED: {reason}"
                )
                result_entry['status'] = 'skipped'
                result_entry['error'] = reason
                result_entry['skip_reason'] = reason
                _append_benchmark_result(benchmark_results, result_entry)
                skipped_hw += 1
                emit_event(
                    event_logger,
                    logger,
                    "example_skip",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    reason=reason,
                )
                mark_progress(example_name)
                _reset_parent_execution_state()  # Reset after skip to keep state clean
                continue

            # Benchmark baseline with explicit timeout
            # Note: Some CUDA benchmarks can take multiple seconds per run, so allow longer timeouts
            cuda_iterations = 3
            cuda_warmup = 0
            cuda_timeout = 30
            logger.info(
                f"    Running baseline executable {baseline_executable.name} "
                f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
            )
            _reap_benchmark_process_leftovers(
                f"phase_start:{chapter_name}:{example_name}:cuda_baseline_timing",
                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                current_owner_pid=os.getpid(),
                repo_root=repo_root,
            )
            baseline_phase_start = time.perf_counter()
            emit_event(
                event_logger,
                logger,
                "phase_start",
                chapter=chapter_name,
                example=example_name,
                example_type="cuda",
                phase="baseline_timing",
                variant="baseline",
                target=baseline_executable.name,
            )
            emit_progress(
                "baseline_timing",
                step=f"{chapter_name}:{example_name}",
                step_detail="cuda baseline timing",
            )
            baseline_result = benchmark_cuda_executable(
                baseline_executable,
                iterations=cuda_iterations,
                warmup=cuda_warmup,
                timeout=cuda_timeout,
            )
            if baseline_result is None:
                result_entry['error'] = f'Baseline execution failed or timed out ({cuda_timeout}s timeout)'
                _append_benchmark_result(benchmark_results, result_entry)
                failed_error += 1
                emit_event(
                    event_logger,
                    logger,
                    "phase_end",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    phase="baseline_timing",
                    variant="baseline",
                    status="failed",
                    duration_s=time.perf_counter() - baseline_phase_start,
                    error=result_entry['error'],
                )
                mark_progress(example_name)
                _reset_parent_execution_state()  # Reset after failure
                continue
            if baseline_result.skip_reason:
                reason = baseline_result.skip_reason
                logger.warning(f"    Baseline executable {baseline_executable.name} SKIPPED: {reason}")
                result_entry['status'] = 'skipped'
                result_entry['error'] = reason
                result_entry['skip_reason'] = reason
                _append_benchmark_result(benchmark_results, result_entry)
                skipped_hw += 1
                emit_event(
                    event_logger,
                    logger,
                    "example_skip",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    reason=reason,
                )
                emit_event(
                    event_logger,
                    logger,
                    "phase_end",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    phase="baseline_timing",
                    variant="baseline",
                    status="skipped",
                    duration_s=time.perf_counter() - baseline_phase_start,
                    error=reason,
                )
                mark_progress(example_name)
                _reset_parent_execution_state()
                continue

            baseline_time = baseline_result.mean_ms
            result_entry['baseline_time_ms'] = baseline_time

            # Enhanced baseline metrics display with emojis and formatting (same as Python)
            logger.info(f"    Baseline: {format_time_ms(baseline_time)} ms")
            logger.info(
                f"      📊 Timing Stats: median={format_time_ms(baseline_result.median_ms)}ms, "
                f"min={format_time_ms(baseline_result.min_ms)}ms, max={format_time_ms(baseline_result.max_ms)}ms, "
                f"std={format_time_ms(baseline_result.std_ms)}ms"
            )
            if baseline_result.percentiles:
                p99 = baseline_result.percentiles.get(99.0, 0)
                p75 = baseline_result.percentiles.get(75.0, 0)
                p50 = baseline_result.percentiles.get(50.0, baseline_result.median_ms)
                logger.info(f"      📈 Percentiles: p99={format_time_ms(p99)}ms, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")
                result_entry['baseline_percentiles'] = dict(baseline_result.percentiles)
                if p75 is not None:
                    result_entry['baseline_p75_ms'] = p75
                p90 = baseline_result.percentiles.get(90.0)
                if p90 is not None:
                    result_entry['baseline_p90_ms'] = p90

            baseline_gpu_metrics = getattr(baseline_result, "gpu_metrics", None)
            if not baseline_gpu_metrics:
                baseline_gpu_metrics = _query_gpu_telemetry_for_profile(validity_profile)
            if baseline_gpu_metrics:
                result_entry['baseline_gpu_metrics'] = baseline_gpu_metrics
                logger.info(f"      🌡️ GPU Telemetry: {format_gpu_telemetry(baseline_gpu_metrics)}")
            emit_event(
                event_logger,
                logger,
                "phase_end",
                chapter=chapter_name,
                example=example_name,
                example_type="cuda",
                phase="baseline_timing",
                variant="baseline",
                status="succeeded",
                duration_s=time.perf_counter() - baseline_phase_start,
            )
            emit_event(
                event_logger,
                logger,
                "baseline_result",
                chapter=chapter_name,
                example=example_name,
                example_type="cuda",
                time_ms=baseline_time,
                median_ms=baseline_result.median_ms,
                min_ms=baseline_result.min_ms,
                max_ms=baseline_result.max_ms,
                std_ms=baseline_result.std_ms,
                percentiles=baseline_result.percentiles,
                p75_ms=result_entry.get("baseline_p75_ms"),
                p90_ms=result_entry.get("baseline_p90_ms"),
                gpu_metrics=baseline_gpu_metrics,
            )

            # Profile baseline if profiling is enabled (nsys, ncu)
            baseline_profile_paths: Dict[str, Optional[Path]] = {}
            example_profile_root: Optional[Path] = None
            example_profile_stem = slugify(example_name)
            if enable_profiling and profiling_output_dir:
                example_profile_root = profiling_output_dir / example_profile_stem
                baseline_profile_dir = example_profile_root / "baseline"
                baseline_profile_dir.mkdir(parents=True, exist_ok=True)

            if enable_profiling and profiling_output_dir:
                _reap_benchmark_process_leftovers(
                    f"phase_start:{chapter_name}:{example_name}:cuda_baseline_profiling",
                    current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                    current_owner_pid=os.getpid(),
                    repo_root=repo_root,
                )
                logger.info(f"    Profiling baseline...")
                profiler_results = []
                baseline_profiler_statuses: Dict[str, str] = {}
                baseline_profiler_errors: Dict[str, str] = {}
                baseline_metrics = {}

                # nsys profiling
                if check_nsys_available():
                    emit_progress(
                        "baseline_nsys",
                        step=f"{chapter_name}:{example_name}",
                        step_detail="nsys profiling (cuda baseline)",
                    )
                    logger.info(f"      nsys...")
                    emit_event(
                        event_logger,
                        logger,
                        "profiler_start",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        variant="baseline",
                        profiler="nsys",
                        timeout_seconds=baseline_config.get_effective_timeout("nsys"),
                    )
                    nsys_path = profile_cuda_executable(
                        baseline_executable,
                        chapter_dir,
                        baseline_profile_dir,
                        config=baseline_config,
                        benchmark=baseline_benchmark,
                        variant="baseline",
                        output_stem=example_profile_stem,
                        timeout_seconds=baseline_config.get_effective_timeout("nsys"),
                    )
                    nsys_error = _get_profile_failure_detail("nsys")
                    if nsys_path:
                        result_entry['baseline_nsys_rep'] = _repo_relative_path(nsys_path, repo_root)
                        profiler_results.append("nsys✓")
                        _set_profiler_status(baseline_profiler_statuses, "nsys", "succeeded")
                        baseline_profile_paths["nsys"] = nsys_path
                        # Extract metrics
                        nsys_metrics = extract_from_nsys_report(nsys_path)
                        if nsys_metrics:
                            baseline_metrics['nsys'] = nsys_metrics
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="baseline",
                            profiler="nsys",
                            status="succeeded",
                            output_path=str(nsys_path),
                            metrics=nsys_metrics,
                        )
                    else:
                        profiler_results.append("nsys✗")
                        _set_profiler_status(baseline_profiler_statuses, "nsys", "failed")
                        nsys_error = _record_profiler_error(
                            baseline_profiler_errors,
                            "nsys",
                            nsys_error,
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="baseline",
                            profiler="nsys",
                            status="failed",
                            output_path=None,
                            metrics=None,
                            error=nsys_error,
                        )
                else:
                    profiler_results.append("nsys-")
                    _set_profiler_status(baseline_profiler_statuses, "nsys", "skipped")
                    nsys_error = _record_profiler_error(
                        baseline_profiler_errors,
                        "nsys",
                        "Nsight Systems unavailable on current host.",
                    )
                    emit_event(
                        event_logger,
                        logger,
                        "profiler_end",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        variant="baseline",
                        profiler="nsys",
                        status="skipped",
                        output_path=None,
                        metrics=None,
                        error=nsys_error,
                    )

                # ncu profiling
                if check_ncu_available():
                    emit_progress(
                        "baseline_ncu",
                        step=f"{chapter_name}:{example_name}",
                        step_detail="ncu profiling (cuda baseline)",
                    )
                    logger.info(f"      ncu...")
                    emit_event(
                        event_logger,
                        logger,
                        "profiler_start",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        variant="baseline",
                        profiler="ncu",
                        timeout_seconds=baseline_config.get_effective_timeout("ncu"),
                    )
                    ncu_path = profile_cuda_executable_ncu(
                        baseline_executable,
                        chapter_dir,
                        baseline_profile_dir,
                        baseline_config,
                        benchmark=baseline_benchmark,
                        variant="baseline",
                        output_stem=example_profile_stem,
                    )
                    ncu_error = _get_profile_failure_detail("ncu")
                    if ncu_path:
                        result_entry['baseline_ncu_rep'] = _repo_relative_path(ncu_path, repo_root)
                        profiler_results.append("ncu✓")
                        _set_profiler_status(baseline_profiler_statuses, "ncu", "succeeded")
                        baseline_profile_paths["ncu"] = ncu_path
                        # Extract metrics
                        ncu_metrics = extract_from_ncu_report(ncu_path)
                        if ncu_metrics:
                            baseline_metrics['ncu'] = ncu_metrics
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="baseline",
                            profiler="ncu",
                            status="succeeded",
                            output_path=str(ncu_path),
                            metrics=ncu_metrics,
                        )
                    else:
                        profiler_results.append("ncu✗")
                        _set_profiler_status(baseline_profiler_statuses, "ncu", "failed")
                        ncu_error = _record_profiler_error(
                            baseline_profiler_errors,
                            "ncu",
                            ncu_error,
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="baseline",
                            profiler="ncu",
                            status="failed",
                            output_path=None,
                            metrics=None,
                            error=ncu_error,
                        )
                else:
                    profiler_results.append("ncu-")
                    _set_profiler_status(baseline_profiler_statuses, "ncu", "skipped")
                    ncu_error = _record_profiler_error(
                        baseline_profiler_errors,
                        "ncu",
                        "Nsight Compute unavailable on current host.",
                    )
                    emit_event(
                        event_logger,
                        logger,
                        "profiler_end",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        variant="baseline",
                        profiler="ncu",
                        status="skipped",
                        output_path=None,
                        metrics=None,
                        error=ncu_error,
                    )

                logger.info(f" ({', '.join(profiler_results)})")
                result_entry["baseline_profiler_statuses"] = dict(baseline_profiler_statuses)
                if baseline_profiler_errors:
                    result_entry["baseline_profiler_errors"] = dict(baseline_profiler_errors)
                _reap_benchmark_process_leftovers(
                    f"{chapter_name}:{example_name}:cuda_baseline_profiling",
                    current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                    current_owner_pid=os.getpid(),
                    repo_root=repo_root,
                )

                # Display extracted metrics
                if baseline_metrics:
                    logger.info("      📈 Profiler Metrics:")
                    if 'nsys' in baseline_metrics:
                        for key, value in baseline_metrics['nsys'].items():
                            logger.info(f"        nsys.{key}: {value:.2f}")
                    if 'ncu' in baseline_metrics:
                        for key, value in baseline_metrics['ncu'].items():
                            logger.info(f"        ncu.{key}: {value:.2f}")
                    result_entry['baseline_profiler_metrics'] = baseline_metrics
                    emit_event(
                        event_logger,
                        logger,
                        "profiler_summary",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        variant="baseline",
                        metrics=baseline_metrics,
                    )

            # Defer optimized profiling for --profile minimal to avoid profiling every
            # variant; profile only the best optimization after selection.
            cuda_optimized_profile_candidates: Dict[str, Dict[str, Any]] = {}

            # Test each optimization
            for optimized_cu_path in optimized_cu_paths:
                opt_name = optimized_cu_path.name
                technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.cu', '')
                if technique == opt_name.replace('optimized_', '').replace('.cu', ''):
                    technique = 'default'

                if num_gpus < 2 and cuda_binary_requires_multi_gpu(optimized_cu_path):
                    reason = (
                        f"SKIPPED: {opt_name} requires >=2 GPUs (e.g., NVLink/C2C) but only {num_gpus} GPU present"
                    )
                    logger.warning(f"    WARNING: {reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'skip_reason': reason,
                    })
                    skipped_hw += 1
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        optimization_file=opt_name,
                        technique=technique,
                        reason=reason,
                    )
                    continue

                optimized_executable = find_cuda_executable(optimized_cu_path, chapter_dir)
                if optimized_executable is None:
                    reason = determine_cuda_skip_reason(
                        optimized_cu_path, chapter_dir, cuda_build_ok, cuda_build_warning
                    )
                    logger.warning(f"    Testing: {opt_name}... SKIPPED: {reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'skip_reason': reason,
                        'error': reason,
                    })
                    skipped_hw += 1
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        optimization_file=opt_name,
                        technique=technique,
                        reason=reason,
                    )
                    continue

                logger.info(
                    f"    Running {opt_name} "
                    f"(runs={cuda_iterations}, timeout={cuda_timeout}s per run)"
                )
                _reap_benchmark_process_leftovers(
                    f"phase_start:{chapter_name}:{example_name}:{technique}:cuda_optimized_timing",
                    current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                    current_owner_pid=os.getpid(),
                    repo_root=repo_root,
                )
                opt_phase_start = time.perf_counter()
                emit_event(
                    event_logger,
                    logger,
                    "phase_start",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    phase="optimized_timing",
                    variant="optimized",
                    technique=technique,
                    target=opt_name,
                )
                emit_progress(
                    "optimized_timing",
                    step=f"{chapter_name}:{example_name}",
                    step_detail=f"cuda optimized timing ({technique})",
                )
                optimized_result = benchmark_cuda_executable(
                    optimized_executable,
                    iterations=cuda_iterations,
                    warmup=cuda_warmup,
                    timeout=cuda_timeout,
                )
                if optimized_result is None:
                    logger.error(f"    Testing: {opt_name}... FAILED (execution or timeout)")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'failed_error',
                        'error': f'Execution failed or timed out ({cuda_timeout}s timeout)',
                    })
                    failed_error += 1
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_error",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        optimization_file=opt_name,
                        technique=technique,
                        error=f'Execution failed or timed out ({cuda_timeout}s timeout)',
                    )
                    emit_event(
                        event_logger,
                        logger,
                        "phase_end",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        phase="optimized_timing",
                        variant="optimized",
                        technique=technique,
                        status="failed",
                        duration_s=time.perf_counter() - opt_phase_start,
                        error=f'Execution failed or timed out ({cuda_timeout}s timeout)',
                    )
                    continue
                if optimized_result.skip_reason:
                    reason = optimized_result.skip_reason
                    logger.warning(f"    Testing: {opt_name}... WARNING: SKIPPED: {reason}")
                    result_entry['optimizations'].append({
                        'file': opt_name,
                        'technique': technique,
                        'status': 'skipped',
                        'skip_reason': reason,
                        'error': f'HARDWARE/SOFTWARE LIMITATION: {reason}',
                    })
                    skipped_hw += 1
                    emit_event(
                        event_logger,
                        logger,
                        "optimization_skip",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        optimization_file=opt_name,
                        technique=technique,
                        reason=reason,
                    )
                    emit_event(
                        event_logger,
                        logger,
                        "phase_end",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        phase="optimized_timing",
                        variant="optimized",
                        technique=technique,
                        status="skipped",
                        duration_s=time.perf_counter() - opt_phase_start,
                        error=reason,
                    )
                    continue

                optimized_time = optimized_result.mean_ms
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                emit_event(
                    event_logger,
                    logger,
                    "phase_end",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    phase="optimized_timing",
                    variant="optimized",
                    technique=technique,
                    status="succeeded",
                    duration_s=time.perf_counter() - opt_phase_start,
                )

                # Enhanced metrics display with emojis and formatting (same as Python)
                emoji = "🚀" if speedup > 1.0 else "⚠️" if speedup < 1.0 else "="
                logger.info(f"    Testing: {opt_name}... {format_time_ms(optimized_time)} ms ({speedup:.2f}x) {emoji}")

                logger.info(
                    f"        📊 Timing: median={format_time_ms(optimized_result.median_ms)}ms, "
                    f"min={format_time_ms(optimized_result.min_ms)}ms, max={format_time_ms(optimized_result.max_ms)}ms, "
                    f"std={format_time_ms(optimized_result.std_ms)}ms"
                )

                opt_p75 = None
                opt_p90 = None
                if optimized_result.percentiles:
                    p99 = optimized_result.percentiles.get(99.0, 0)
                    p75 = optimized_result.percentiles.get(75.0, 0)
                    p50 = optimized_result.percentiles.get(50.0, optimized_result.median_ms)
                    opt_p75 = p75
                    opt_p90 = optimized_result.percentiles.get(90.0)
                    p99_speedup = ""
                    if baseline_result.percentiles and 99.0 in baseline_result.percentiles:
                        p99_baseline = baseline_result.percentiles[99.0]
                        if p99_baseline > 0:
                            p99_speedup = f" ({p99_baseline/p99:.2f}x)" if p99 > 0 else ""
                    logger.info(f"        📈 Percentiles: p99={format_time_ms(p99)}ms{p99_speedup}, p75={format_time_ms(p75)}ms, p50={format_time_ms(p50)}ms")

                # Visual speedup bar (always show for consistency, same as Python)
                bar_length = 40
                if speedup > 1.0:
                    # Improvement: fill bar proportionally to speedup
                    filled = min(int((speedup - 1.0) / max(speedup, 10.0) * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    logger.info(f"        [{bar}] {speedup:.2f}x speedup")
                elif speedup < 1.0:
                    # Regression: show how much slower (distance from 1.0)
                    regress_ratio = (1.0 - speedup)
                    filled = min(int(regress_ratio / 0.5 * bar_length), bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    logger.info(f"        [{bar}] {speedup:.2f}x slowdown")
                else:
                    # No change
                    bar = "░" * bar_length
                    logger.info(f"        [{bar}] {speedup:.2f}x (no change)")

                opt_result = {
                    'file': opt_name,
                    'technique': technique,
                    'status': 'succeeded',
                    'time_ms': optimized_time,
                    'speedup': speedup,
                }
                if opt_p75 is not None:
                    opt_result['p75_ms'] = opt_p75
                if opt_p90 is not None:
                    opt_result['p90_ms'] = opt_p90
                cuda_opt_gpu_metrics = getattr(optimized_result, "gpu_metrics", None)
                if not cuda_opt_gpu_metrics:
                    cuda_opt_gpu_metrics = _query_gpu_telemetry_for_profile(validity_profile)
                if cuda_opt_gpu_metrics:
                    opt_result['gpu_metrics'] = cuda_opt_gpu_metrics
                    logger.info(f"        🌡️ GPU Telemetry: {format_gpu_telemetry(cuda_opt_gpu_metrics)}")

                # Profile optimized if profiling is enabled (nsys, ncu)
                if (
                    enable_profiling
                    and profiling_output_dir
                    and str(profile_type).lower() == "minimal"
                    and opt_result.get("status") == "succeeded"
                ):
                    cuda_optimized_profile_candidates[technique] = {
                        "executable": optimized_executable,
                        "config": optimized_config,
                        "result": opt_result,
                    }

                if (
                    enable_profiling
                    and profiling_output_dir
                    and str(profile_type).lower() != "minimal"
                    and opt_result.get("status") == "succeeded"
                ):
                    _reap_benchmark_process_leftovers(
                        f"phase_start:{chapter_name}:{example_name}:{technique}:cuda_optimized_profiling",
                        current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                        current_owner_pid=os.getpid(),
                        repo_root=repo_root,
                    )
                    pair_dir = _profile_pair_dir(
                        profile_output_root, chapter_id, example_name, technique
                    )
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"\n    Profiling optimized...")
                    profiler_results = []
                    optimized_profiler_statuses: Dict[str, str] = {}
                    optimized_profiler_errors: Dict[str, str] = {}
                    optimized_metrics = {}

                    # nsys profiling
                    if check_nsys_available():
                        emit_progress(
                            "optimized_nsys",
                            step=f"{chapter_name}:{example_name}",
                            step_detail=f"nsys profiling (cuda optimized {technique})",
                        )
                        logger.info(f"      nsys...")
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_start",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="optimized",
                            profiler="nsys",
                            technique=technique,
                            timeout_seconds=optimized_config.get_effective_timeout("nsys"),
                        )
                        nsys_path = profile_cuda_executable(
                            optimized_executable,
                            chapter_dir,
                            pair_dir,
                            config=optimized_config,
                            benchmark=optimized_benchmark,
                            variant="optimized",
                            output_stem=example_profile_stem,
                            timeout_seconds=optimized_config.get_effective_timeout("nsys"),
                        )
                        nsys_error = _get_profile_failure_detail("nsys")
                        if nsys_path:
                            opt_result['optimized_nsys_rep'] = _repo_relative_path(nsys_path, repo_root)
                            profiler_results.append("nsys✓")
                            _set_profiler_status(optimized_profiler_statuses, "nsys", "succeeded")
                            # Extract metrics
                            nsys_metrics = extract_from_nsys_report(nsys_path)
                            if nsys_metrics:
                                optimized_metrics['nsys'] = nsys_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type="cuda",
                                variant="optimized",
                                profiler="nsys",
                                technique=technique,
                                status="succeeded",
                                output_path=str(nsys_path),
                                metrics=nsys_metrics,
                            )
                        else:
                            profiler_results.append("nsys✗")
                            _set_profiler_status(optimized_profiler_statuses, "nsys", "failed")
                            nsys_error = _record_profiler_error(
                                optimized_profiler_errors,
                                "nsys",
                                nsys_error,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type="cuda",
                                variant="optimized",
                                profiler="nsys",
                                technique=technique,
                                status="failed",
                                output_path=None,
                                metrics=None,
                                error=nsys_error,
                            )
                    else:
                        profiler_results.append("nsys-")
                        _set_profiler_status(optimized_profiler_statuses, "nsys", "skipped")
                        nsys_error = _record_profiler_error(
                            optimized_profiler_errors,
                            "nsys",
                            "Nsight Systems unavailable on current host.",
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="optimized",
                            profiler="nsys",
                            technique=technique,
                            status="skipped",
                            output_path=None,
                            metrics=None,
                            error=nsys_error,
                        )

                    # ncu profiling
                    if check_ncu_available():
                        emit_progress(
                            "optimized_ncu",
                            step=f"{chapter_name}:{example_name}",
                            step_detail=f"ncu profiling (cuda optimized {technique})",
                        )
                        logger.info("ncu...")
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_start",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="optimized",
                            profiler="ncu",
                            technique=technique,
                            timeout_seconds=optimized_config.get_effective_timeout("ncu"),
                        )
                        ncu_path = profile_cuda_executable_ncu(
                            optimized_executable,
                            chapter_dir,
                            pair_dir,
                            optimized_config,
                            benchmark=optimized_benchmark,
                            variant="optimized",
                            output_stem=example_profile_stem,
                        )
                        ncu_error = _get_profile_failure_detail("ncu")
                        if ncu_path:
                            opt_result['optimized_ncu_rep'] = _repo_relative_path(ncu_path, repo_root)
                            profiler_results.append("ncu✓")
                            _set_profiler_status(optimized_profiler_statuses, "ncu", "succeeded")
                            # Extract metrics
                            ncu_metrics = extract_from_ncu_report(ncu_path)
                            if ncu_metrics:
                                optimized_metrics['ncu'] = ncu_metrics
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type="cuda",
                                variant="optimized",
                                profiler="ncu",
                                technique=technique,
                                status="succeeded",
                                output_path=str(ncu_path),
                                metrics=ncu_metrics,
                            )
                        else:
                            profiler_results.append("ncu✗")
                            _set_profiler_status(optimized_profiler_statuses, "ncu", "failed")
                            ncu_error = _record_profiler_error(
                                optimized_profiler_errors,
                                "ncu",
                                ncu_error,
                            )
                            emit_event(
                                event_logger,
                                logger,
                                "profiler_end",
                                chapter=chapter_name,
                                example=example_name,
                                example_type="cuda",
                                variant="optimized",
                                profiler="ncu",
                                technique=technique,
                                status="failed",
                                output_path=None,
                                metrics=None,
                                error=ncu_error,
                            )
                    else:
                        profiler_results.append("ncu-")
                        _set_profiler_status(optimized_profiler_statuses, "ncu", "skipped")
                        ncu_error = _record_profiler_error(
                            optimized_profiler_errors,
                            "ncu",
                            "Nsight Compute unavailable on current host.",
                        )
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_end",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="optimized",
                            profiler="ncu",
                            technique=technique,
                            status="skipped",
                            output_path=None,
                            metrics=None,
                            error=ncu_error,
                        )

                    logger.info(f" ({', '.join(profiler_results)})")
                    opt_result["optimized_profiler_statuses"] = dict(optimized_profiler_statuses)
                    if optimized_profiler_errors:
                        opt_result["optimized_profiler_errors"] = dict(optimized_profiler_errors)
                    _reap_benchmark_process_leftovers(
                        f"{chapter_name}:{example_name}:{technique}:cuda_optimized_profiling",
                        current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                        current_owner_pid=os.getpid(),
                        repo_root=repo_root,
                    )

                    # Display extracted metrics
                    if optimized_metrics:
                        logger.info("        📈 Profiler Metrics:")
                        log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
                        opt_result['optimized_profiler_metrics'] = optimized_metrics
                        emit_event(
                            event_logger,
                            logger,
                            "profiler_summary",
                            chapter=chapter_name,
                            example=example_name,
                            example_type="cuda",
                            variant="optimized",
                            technique=technique,
                            metrics=optimized_metrics,
                        )

                result_entry['optimizations'].append(opt_result)
                emit_event(
                    event_logger,
                    logger,
                    "optimization_result",
                    chapter=chapter_name,
                    example=example_name,
                    example_type="cuda",
                    optimization_file=opt_name,
                    technique=technique,
                    status=opt_result.get("status"),
                    time_ms=opt_result.get("time_ms"),
                    speedup=opt_result.get("speedup"),
                    p75_ms=opt_result.get("p75_ms"),
                    p90_ms=opt_result.get("p90_ms"),
                    gpu_metrics=opt_result.get("gpu_metrics"),
                    profiler_metrics=opt_result.get("optimized_profiler_metrics"),
                )

                if speedup > result_entry['best_speedup']:
                    result_entry['best_speedup'] = speedup
                    speedups.append(speedup)

            if result_entry['best_speedup'] > 1.0:
                logger.info(f"    Best speedup: {result_entry['best_speedup']:.2f}x")
            if result_entry['status'] == 'skipped':
                _append_benchmark_result(benchmark_results, result_entry)
                mark_progress(example_name)
                continue

            optimizations = result_entry.get('optimizations', [])
            has_success = any(opt.get('status') == 'succeeded' for opt in optimizations)
            all_skipped_opt = bool(optimizations) and all(opt.get('status') == 'skipped' for opt in optimizations)
            baseline_ok = result_entry.get('baseline_time_ms') is not None

            update_result = None
            if baseline_ok and has_success:
                example_key = expectation_example_key(result_entry["example"], result_entry.get("type", "cuda"))
                optimization_goal = (result_entry.get("optimization_goal") or "speed").strip().lower()
                best_opt = select_best_optimization(result_entry.get("optimizations", []), goal=optimization_goal)
                if (
                    enable_profiling
                    and profiling_output_dir
                    and str(profile_type).lower() == "minimal"
                    and best_opt
                    and isinstance(best_opt, dict)
                    and best_opt.get("status") == "succeeded"
                ):
                    best_key = str(best_opt.get("technique") or best_opt.get("file") or "")
                    cand = cuda_optimized_profile_candidates.get(best_key)
                    if cand:
                        try:
                            _reap_benchmark_process_leftovers(
                                f"phase_start:{chapter_name}:{example_name}:{best_key}:cuda_optimized_profiling",
                                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                                current_owner_pid=os.getpid(),
                                repo_root=repo_root,
                            )
                            pair_dir = _profile_pair_dir(
                                profile_output_root, chapter_id, example_name, best_key
                            )
                            pair_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"\n    Profiling optimized (best only: {best_key})...")
                            profiler_results = []
                            optimized_profiler_statuses: Dict[str, str] = {}
                            optimized_profiler_errors: Dict[str, str] = {}
                            optimized_metrics = {}

                            optimized_executable = cand.get("executable")
                            optimized_config = cand.get("config") or base_config

                            if check_nsys_available():
                                emit_progress(
                                    "optimized_nsys",
                                    step=f"{chapter_name}:{example_name}",
                                    step_detail=f"nsys profiling (cuda optimized {best_key})",
                                )
                                logger.info("      nsys...")
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_start",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type="cuda",
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=best_key,
                                    timeout_seconds=optimized_config.get_effective_timeout("nsys") if optimized_config else None,
                                )
                                nsys_path = profile_cuda_executable(
                                    optimized_executable,
                                    chapter_dir,
                                    pair_dir,
                                    config=optimized_config,
                                    benchmark=optimized_benchmark,
                                    variant="optimized",
                                    output_stem=example_profile_stem,
                                    timeout_seconds=optimized_config.get_effective_timeout("nsys") if optimized_config else None,
                                )
                                nsys_error = _get_profile_failure_detail("nsys")
                                if nsys_path:
                                    best_opt["optimized_nsys_rep"] = _repo_relative_path(nsys_path, repo_root)
                                    profiler_results.append("nsys✓")
                                    _set_profiler_status(optimized_profiler_statuses, "nsys", "succeeded")
                                    nsys_metrics = extract_from_nsys_report(nsys_path)
                                    if nsys_metrics:
                                        optimized_metrics["nsys"] = nsys_metrics
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type="cuda",
                                        variant="optimized",
                                        profiler="nsys",
                                        technique=best_key,
                                        status="succeeded",
                                        output_path=str(nsys_path),
                                        metrics=nsys_metrics,
                                    )
                                else:
                                    profiler_results.append("nsys✗")
                                    _set_profiler_status(optimized_profiler_statuses, "nsys", "failed")
                                    nsys_error = _record_profiler_error(
                                        optimized_profiler_errors,
                                        "nsys",
                                        nsys_error,
                                    )
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type="cuda",
                                        variant="optimized",
                                        profiler="nsys",
                                        technique=best_key,
                                        status="failed",
                                        output_path=None,
                                        metrics=None,
                                        error=nsys_error,
                                    )
                            else:
                                profiler_results.append("nsys-")
                                _set_profiler_status(optimized_profiler_statuses, "nsys", "skipped")
                                nsys_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "nsys",
                                    "Nsight Systems unavailable on current host.",
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type="cuda",
                                    variant="optimized",
                                    profiler="nsys",
                                    technique=best_key,
                                    status="skipped",
                                    output_path=None,
                                    metrics=None,
                                    error=nsys_error,
                                )

                            if check_ncu_available():
                                emit_progress(
                                    "optimized_ncu",
                                    step=f"{chapter_name}:{example_name}",
                                    step_detail=f"ncu profiling (cuda optimized {best_key})",
                                )
                                logger.info("ncu...")
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_start",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type="cuda",
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=best_key,
                                    timeout_seconds=optimized_config.get_effective_timeout("ncu") if optimized_config else None,
                                )
                                ncu_path = profile_cuda_executable_ncu(
                                    optimized_executable,
                                    chapter_dir,
                                    pair_dir,
                                    optimized_config,
                                    benchmark=optimized_benchmark,
                                    variant="optimized",
                                    output_stem=example_profile_stem,
                                )
                                ncu_error = _get_profile_failure_detail("ncu")
                                if ncu_path:
                                    best_opt["optimized_ncu_rep"] = _repo_relative_path(ncu_path, repo_root)
                                    profiler_results.append("ncu✓")
                                    _set_profiler_status(optimized_profiler_statuses, "ncu", "succeeded")
                                    ncu_metrics = extract_from_ncu_report(ncu_path)
                                    if ncu_metrics:
                                        optimized_metrics["ncu"] = ncu_metrics
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type="cuda",
                                        variant="optimized",
                                        profiler="ncu",
                                        technique=best_key,
                                        status="succeeded",
                                        output_path=str(ncu_path),
                                        metrics=ncu_metrics,
                                    )
                                else:
                                    profiler_results.append("ncu✗")
                                    _set_profiler_status(optimized_profiler_statuses, "ncu", "failed")
                                    ncu_error = _record_profiler_error(
                                        optimized_profiler_errors,
                                        "ncu",
                                        ncu_error,
                                    )
                                    emit_event(
                                        event_logger,
                                        logger,
                                        "profiler_end",
                                        chapter=chapter_name,
                                        example=example_name,
                                        example_type="cuda",
                                        variant="optimized",
                                        profiler="ncu",
                                        technique=best_key,
                                        status="failed",
                                        output_path=None,
                                        metrics=None,
                                        error=ncu_error,
                                    )
                            else:
                                profiler_results.append("ncu-")
                                _set_profiler_status(optimized_profiler_statuses, "ncu", "skipped")
                                ncu_error = _record_profiler_error(
                                    optimized_profiler_errors,
                                    "ncu",
                                    "Nsight Compute unavailable on current host.",
                                )
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_end",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type="cuda",
                                    variant="optimized",
                                    profiler="ncu",
                                    technique=best_key,
                                    status="skipped",
                                    output_path=None,
                                    metrics=None,
                                    error=ncu_error,
                                )

                            logger.info(f" ({', '.join(profiler_results)})")
                            best_opt["optimized_profiler_statuses"] = dict(optimized_profiler_statuses)
                            if optimized_profiler_errors:
                                best_opt["optimized_profiler_errors"] = dict(optimized_profiler_errors)
                            _reap_benchmark_process_leftovers(
                                f"{chapter_name}:{example_name}:{best_key}:cuda_optimized_profiling",
                                current_run_id=os.environ.get("AISP_BENCHMARK_OWNER_RUN_ID", ""),
                                current_owner_pid=os.getpid(),
                                repo_root=repo_root,
                            )
                            if optimized_metrics:
                                logger.info("        📈 Profiler Metrics:")
                                log_profiler_metrics_table(logger, optimized_metrics, indent="          ")
                                best_opt["optimized_profiler_metrics"] = optimized_metrics
                                emit_event(
                                    event_logger,
                                    logger,
                                    "profiler_summary",
                                    chapter=chapter_name,
                                    example=example_name,
                                    example_type="cuda",
                                    variant="optimized",
                                    technique=best_key,
                                    metrics=optimized_metrics,
                                )
                        except Exception:
                            logger.warning("    WARNING: Best-only profiling failed", exc_info=True)
                if best_opt:
                    emit_event(
                        event_logger,
                        logger,
                        "best_optimization",
                        chapter=chapter_name,
                        example=example_name,
                        example_type="cuda",
                        optimization_goal=optimization_goal,
                        technique=best_opt.get("technique") or best_opt.get("file"),
                        optimization_file=best_opt.get("file"),
                        speedup=best_opt.get("speedup"),
                        time_ms=best_opt.get("time_ms"),
                    )
                    emit_progress(
                        "expectations",
                        step=f"{chapter_name}:{example_name}",
                        step_detail="expectations update",
                    )
                    old_entry = expectations_store.get_entry(example_key)
                    provenance = RunProvenance(
                        git_commit=git_commit or "unknown",
                        hardware_key=expectation_hardware_key,
                        profile_name=profile_type,
                        timestamp=datetime.now().isoformat(),
                        iterations=int(iterations),
                        warmup_iterations=int(warmup),
                        execution_environment=execution_environment.kind,
                        validity_profile=validity_profile,
                        dmi_product_name=execution_environment.dmi_product_name,
                    )
                    entry = ExpectationEntry(
                        example=result_entry.get("example", example_key),
                        type=result_entry.get("type", "cuda"),
                        optimization_goal=result_entry.get("optimization_goal", "speed"),
                        baseline_time_ms=float(result_entry.get("baseline_time_ms") or 0.0),
                        best_optimized_time_ms=float(best_opt.get("time_ms") or 0.0),
                        provenance=provenance,
                        baseline_memory_mb=result_entry.get("baseline_memory_mb"),
                        best_optimized_memory_mb=best_opt.get("memory_mb"),
                        baseline_p75_ms=result_entry.get("baseline_p75_ms"),
                        baseline_p90_ms=result_entry.get("baseline_p90_ms"),
                        best_optimized_p75_ms=best_opt.get("p75_ms"),
                        best_optimized_p90_ms=best_opt.get("p90_ms"),
                        baseline_throughput=result_entry.get("baseline_throughput"),
                        best_optimized_throughput=best_opt.get("throughput"),
                        best_optimization_name=best_opt.get("technique") or best_opt.get("file"),
                        best_optimization_file=best_opt.get("file"),
                        best_optimization_technique=best_opt.get("technique"),
                    )
                    update_result = None
                    if expectation_validation_enabled:
                        active_expectations_store = expectations_store
                        if not expectation_writes_enabled:
                            active_expectations_store = ExpectationsStore(
                                chapter_dir,
                                expectation_hardware_key,
                                accept_regressions=accept_regressions or update_expectations,
                                allow_mixed_provenance=allow_mixed_provenance_for_writes,
                            )
                        update_result = active_expectations_store.update_entry(example_key, entry)
                        try:
                            result_entry["expectation"] = update_result.to_dict()
                        except Exception:
                            result_entry["expectation"] = {
                                "status": update_result.status,
                                "message": update_result.message,
                                "validation_issues": [issue.to_dict() for issue in update_result.validation_issues],
                            }
                        if expectation_writes_enabled:
                            logger.info("    Expectations: %s", update_result.message)
                        else:
                            result_entry["expectation"]["persisted"] = False
                            result_entry["expectation"]["message"] = (
                                f"{update_result.message} (preview only; rerun with "
                                "--update-expectations/--accept-regressions/--allow-mixed-provenance "
                                "to write the file.)"
                            )
                            logger.info("    Expectations (preview): %s", update_result.message)
                        log_expectation_delta(
                            logger,
                            example_key=example_key,
                            goal=optimization_goal,
                            old_entry=old_entry,
                            new_entry=entry,
                            update_result=update_result,
                            event_logger=event_logger,
                            chapter=chapter_name,
                        )
                    else:
                        result_entry["expectation"] = {
                            "status": "skipped",
                            "message": (
                                "Expectation validation is disabled in portable validity profile. "
                                "Enable --allow-portable-expectations-update "
                                "(allow_portable_expectations_update=True) to validate and write expectation files."
                            ),
                            "validation_issues": [],
                        }

                is_rejected_regression = bool(
                    update_result
                    and update_result.status == "rejected"
                    and any(issue.issue_type == "regression" for issue in update_result.validation_issues)
                )
                profiler_failures = _collect_required_profiler_failures(
                    result_entry,
                    best_opt,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                profiler_failure_details = _collect_required_profiler_failure_details(
                    result_entry,
                    best_opt,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                if profiler_failures:
                    result_entry["status"] = "failed_profiler"
                    result_entry["error"] = _format_required_profiler_failure(
                        profiler_failures,
                        failure_details=profiler_failure_details,
                    )
                    logger.warning("    WARNING: %s", result_entry["error"])
                    failed_error += 1
                elif is_rejected_regression:
                    regression_metrics = None
                    if best_opt and isinstance(best_opt, dict):
                        regression_metrics = best_opt.get("gpu_metrics")
                    if not regression_metrics:
                        regression_metrics = result_entry.get("baseline_gpu_metrics")
                    if regression_metrics:
                        logger.warning(
                            "    🌡️ GPU telemetry during regression: %s",
                            format_gpu_telemetry(regression_metrics),
                        )
                        temp = regression_metrics.get("temperature_gpu_c")
                        if temp is not None and temp >= 85:
                            logger.warning(
                                "    ⚠️ GPU temperature %.1f°C exceeds recommended threshold; consider cooling or resetting before re-running.",
                                temp,
                            )
                    else:
                        live_metrics = _query_gpu_telemetry_for_profile(validity_profile)
                        logger.warning(
                            "    🌡️ GPU telemetry during regression: %s",
                            format_gpu_telemetry(live_metrics),
                        )
                    result_entry["status"] = "failed_regression"
                    result_entry["error"] = (
                        update_result.message if update_result else "Expectation regression detected"
                    )
                    failed_regression += 1
                elif _should_fail_no_speedup(result_entry):
                    result_entry["status"] = "failed_no_speedup"
                    result_entry["error"] = _format_failed_no_speedup(result_entry)
                    failed_no_speedup += 1
                else:
                    result_entry["status"] = "succeeded"
                    successful += 1
            elif baseline_ok and (all_skipped_opt or not optimizations):
                profiler_failures = _collect_required_profiler_failures(
                    result_entry,
                    None,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                profiler_failure_details = _collect_required_profiler_failure_details(
                    result_entry,
                    None,
                    profiling_requested=bool(enable_profiling and profiling_output_dir),
                )
                if profiler_failures:
                    result_entry["status"] = "failed_profiler"
                    result_entry["error"] = _format_required_profiler_failure(
                        profiler_failures,
                        failure_details=profiler_failure_details,
                    )
                    logger.warning("    WARNING: %s", result_entry["error"])
                    failed_error += 1
                elif all_skipped_opt:
                    skip_reason = _collect_all_skipped_optimization_reason(optimizations)
                    if skip_reason:
                        result_entry["status"] = "skipped"
                        result_entry["error"] = f"SKIPPED: {skip_reason}"
                        result_entry["skip_reason"] = skip_reason
                    elif _should_fail_no_speedup(result_entry):
                        result_entry["status"] = "failed_no_speedup"
                        result_entry["error"] = _format_failed_no_speedup(result_entry)
                        failed_no_speedup += 1
                    else:
                        result_entry['status'] = 'succeeded'
                        successful += 1
                elif _should_fail_no_speedup(result_entry):
                    result_entry["status"] = "failed_no_speedup"
                    result_entry["error"] = _format_failed_no_speedup(result_entry)
                    failed_no_speedup += 1
                else:
                    result_entry['status'] = 'succeeded'
                    successful += 1
            else:
                result_entry['status'] = 'failed_error'
                if not result_entry.get('error'):
                    result_entry['error'] = _GENERIC_BENCHMARK_FAILURE_ERROR
                failed_error += 1

            emit_event(
                event_logger,
                logger,
                "example_end",
                chapter=chapter_name,
                example=example_name,
                example_type="cuda",
                status=result_entry.get("status"),
                best_speedup=result_entry.get("best_speedup"),
                optimization_goal=result_entry.get("optimization_goal"),
                error=result_entry.get("error"),
            )
            _append_benchmark_result(benchmark_results, result_entry)
            mark_progress(example_name)

    logger.info(f"  Recorded benchmark entries: {len(benchmark_results)}")
    expectations_store.save()

    # LLM Analysis and Patching
    llm_patch_metrics = {
        'total_analyzed': 0,
        'patches_extracted': 0,
        'patches_applied': 0,
        'patches_failed': 0,
        'patches_rebenchmarked': 0,
        'patches_refined': 0,  # Successfully refined after initial failure
        'best_patches_selected': 0,  # Number of "best" patches identified
        'total_speedup_improvement': 0.0,  # Sum of speedups from best patches
        'patches_verified': 0,  # Patches that passed output verification
        'patches_verification_failed': 0,  # Patches with verification errors
        'failures': [],  # List of {example, reason}
    }
    
    if llm_analysis:
        logger.info("  Running LLM-powered analysis...")
        def _patch_label(patch: Dict[str, Any]) -> str:
            name = patch.get("variant_name")
            if name:
                return str(name)
            patched_file = patch.get("patched_file") or ""
            if patched_file:
                return Path(patched_file).name
            return "patch"

        for bench_result in benchmark_results:
            # Run LLM analysis for benchmarks that need optimization
            # Default: <1.1x speedup, but --force-llm runs on ALL benchmarks
            best_speedup = bench_result.get('best_speedup', 1.0)
            needs_analysis = force_llm or best_speedup < 1.1
            if bench_result.get('status') in ('succeeded', 'failed_regression', 'failed_no_speedup') and needs_analysis:
                emit_progress(
                    "llm_analysis",
                    step=f"{chapter_name}:{bench_result['example']}",
                    step_detail="analysis start",
                )
                llm_result = _run_llm_analysis_for_benchmark(
                    bench_result,
                    profiling_output_dir,
                    chapter_dir,
                    llm_provider=llm_provider,
                    use_cache=use_llm_cache,
                )
                if llm_result:
                    bench_result['llm_analysis'] = llm_result
                    llm_patch_metrics['total_analyzed'] += 1
                    logger.info(f"    ✓ {bench_result['example']}: LLM analysis ({llm_result.get('latency_seconds', 0):.1f}s)")
                    emit_progress(
                        "llm_analysis",
                        step=f"{chapter_name}:{bench_result['example']}",
                        step_detail="analysis complete",
                    )
                    
                    # Apply patches if enabled
                    if apply_llm_patches and llm_result.get('md_path'):
                        emit_progress(
                            "llm_patch_apply",
                            step=f"{chapter_name}:{bench_result['example']}",
                            step_detail="patch apply start",
                        )
                        patch_results = _apply_llm_patches_for_benchmark(
                            bench_result,
                            llm_result,
                            chapter_dir,
                            profiling_output_dir,
                            patch_strategy=patch_strategy,
                            llm_provider=llm_provider,
                            max_refinement_attempts=llm_patch_retries,
                        )
                        if patch_results:
                            bench_result['llm_patches'] = patch_results
                            successful_patches = [p for p in patch_results if p.get('success')]
                            failed_patches = [p for p in patch_results if not p.get('success')]
                            
                            llm_patch_metrics['patches_extracted'] += len(patch_results)
                            llm_patch_metrics['patches_applied'] += len(successful_patches)
                            llm_patch_metrics['patches_failed'] += len(failed_patches)
                            
                            # Log failures with reasons
                            for fp in failed_patches:
                                llm_patch_metrics['failures'].append({
                                    'example': bench_result['example'],
                                    'reason': fp.get('error', fp.get('failure_reason', 'Unknown')),
                                })

                            for patch in patch_results:
                                emit_progress(
                                    "llm_patch_apply",
                                    step=f"{chapter_name}:{bench_result['example']}",
                                    step_detail=f"patch {_patch_label(patch)}: {'ok' if patch.get('success') else 'failed'}",
                                )
                            
                            logger.info(f"    📝 {bench_result['example']}: Applied {len(successful_patches)}/{len(patch_results)} patches")
                            
                            # Re-benchmark if enabled
                            if rebenchmark_llm_patches and successful_patches:
                                benchmarkable = [p for p in successful_patches if p.get('can_benchmark', True)]
                                baseline_time = bench_result.get('baseline_time_ms', 0)
                                
                                # Load original optimized code for potential refinement
                                original_optimized_code = None
                                # Find optimized file using same logic as _apply_llm_patches_for_benchmark
                                example_name = bench_result.get('example', '')
                                optimizations = bench_result.get('optimizations', [])
                                
                                best_opt = None
                                best_speedup = 0.0
                                for opt in optimizations:
                                    if opt.get('status') == 'succeeded' and opt.get('speedup', 0) > best_speedup:
                                        best_speedup = opt.get('speedup', 0)
                                        best_opt = opt
                                
                                optimized_file = chapter_dir / f"optimized_{example_name}.py"
                                if not optimized_file.exists():
                                    optimized_file = chapter_dir / f"baseline_{example_name}.py"
                                if best_opt and best_opt.get('file'):
                                    optimized_file = chapter_dir / best_opt['file']
                                
                                if optimized_file.exists() and optimized_file.is_file():
                                    original_optimized_code = _safe_read_text_with_warning(
                                        optimized_file,
                                        label=f"original optimized source code for {example_name}",
                                    )
                                
                                for patch in benchmarkable:
                                    patch_name = _patch_label(patch)
                                    emit_progress(
                                        "llm_patch_rebenchmark",
                                        step=f"{chapter_name}:{bench_result['example']}",
                                        step_detail=f"rebenchmark start {patch_name}",
                                    )
                                    rebench_result = _rebenchmark_patched_variant(
                                        patch['patched_file'],
                                        iterations=iterations or 3,
                                        warmup=warmup or 1,
                                        enable_profiling=enable_profiling,
                                        profile_type=profile_type,
                                        profile_output_dir=profiling_output_dir / "llm_patches" if profiling_output_dir else None,
                                    )
                                    patch['rebenchmark_result'] = rebench_result
                                    emit_progress(
                                        "llm_patch_rebenchmark",
                                        step=f"{chapter_name}:{bench_result['example']}",
                                        step_detail=f"rebenchmark {'ok' if rebench_result.get('success') else 'failed'} {patch_name}",
                                    )
                                    
                                    if rebench_result.get('success'):
                                        llm_patch_metrics['patches_rebenchmarked'] += 1
                                        # Calculate actual speedup
                                        patch_time = rebench_result.get('median_ms')
                                        if patch_time and baseline_time > 0:
                                            patch['actual_speedup'] = baseline_time / patch_time
                                            logger.info(f"      ✓ {patch.get('variant_name', 'patch')}: {patch_time:.3f}ms ({patch['actual_speedup']:.2f}x vs baseline)")
                                        
                                        # Auto-verify patched output matches original
                                        if optimized_file.exists() and optimized_file.is_file():
                                            verify_result = _verify_patched_benchmark(
                                                str(optimized_file),
                                                patch['patched_file'],
                                            )
                                            patch['verification'] = verify_result
                                            emit_progress(
                                                "llm_patch_verify",
                                                step=f"{chapter_name}:{bench_result['example']}",
                                                step_detail=f"verify {'ok' if verify_result.get('verified') else 'failed'} {patch_name}",
                                            )
                                            if verify_result.get('verified'):
                                                llm_patch_metrics['patches_verified'] += 1
                                                logger.info(f"      ✓ Verified: output matches original")
                                            elif verify_result.get('errors'):
                                                llm_patch_metrics['patches_verification_failed'] += 1
                                                logger.warning(f"      ⚠ Verification: {verify_result['errors'][0]}")
                                    else:
                                        # Rebenchmark failed - try iterative refinement
                                        error_info = rebench_result
                                        logger.warning(f"      ✗ {patch.get('variant_name', 'patch')} failed: {error_info.get('error_type')}")
                                        
                                        # Try refinement (up to llm_patch_retries attempts)
                                        if original_optimized_code:
                                            patched_file_path = Path(patch['patched_file'])
                                            patched_code = (
                                                _safe_read_text_with_warning(
                                                    patched_file_path,
                                                    label=f"patched source code for {patch_name}",
                                                )
                                                if patched_file_path.exists()
                                                else None
                                            )
                                            if patched_code:
                                                for attempt in range(llm_patch_retries):
                                                    logger.info(f"      🔄 Refinement attempt {attempt + 1}/{llm_patch_retries}...")
                                                    refined_code = _refine_patch_with_llm(
                                                        original_optimized_code,
                                                        patched_code,
                                                        error_info,
                                                        bench_result,
                                                        chapter_dir,
                                                        llm_provider=llm_provider,
                                                    )
                                                    if refined_code:
                                                        # Save refined code
                                                        refined_path = Path(patch['patched_file']).with_suffix('.refined.py')
                                                        refined_path.write_text(refined_code)
                                                        
                                                        # Try rebenchmark again
                                                        emit_progress(
                                                            "llm_patch_rebenchmark",
                                                            step=f"{chapter_name}:{bench_result['example']}",
                                                            step_detail=f"refine rebenchmark attempt {attempt + 1} {patch_name}",
                                                        )
                                                        refined_result = _rebenchmark_patched_variant(
                                                            str(refined_path),
                                                            iterations=iterations or 3,
                                                            warmup=warmup or 1,
                                                            enable_profiling=enable_profiling,
                                                            profile_type=profile_type,
                                                            profile_output_dir=profiling_output_dir / "llm_patches" if profiling_output_dir else None,
                                                        )
                                                        emit_progress(
                                                            "llm_patch_rebenchmark",
                                                            step=f"{chapter_name}:{bench_result['example']}",
                                                            step_detail=f"refine rebenchmark {'ok' if refined_result.get('success') else 'failed'} {patch_name}",
                                                        )
                                                        
                                                        if refined_result.get('success'):
                                                            patch['rebenchmark_result'] = refined_result
                                                            patch['refined'] = True
                                                            patch['refinement_attempts'] = attempt + 1
                                                            patch['patched_file'] = str(refined_path)
                                                            llm_patch_metrics['patches_rebenchmarked'] += 1
                                                            
                                                            patch_time = refined_result.get('median_ms')
                                                            if patch_time and baseline_time > 0:
                                                                patch['actual_speedup'] = baseline_time / patch_time
                                                                logger.info(f"      ✓ Refined {patch.get('variant_name', 'patch')}: {patch_time:.3f}ms ({patch['actual_speedup']:.2f}x)")
                                                            
                                                            # Verify refined patch
                                                            if optimized_file.exists() and optimized_file.is_file():
                                                                verify_result = _verify_patched_benchmark(
                                                                    str(optimized_file),
                                                                    str(refined_path),
                                                                )
                                                                patch['verification'] = verify_result
                                                                emit_progress(
                                                                    "llm_patch_verify",
                                                                    step=f"{chapter_name}:{bench_result['example']}",
                                                                    step_detail=f"verify {'ok' if verify_result.get('verified') else 'failed'} {patch_name}",
                                                                )
                                                                if verify_result.get('verified'):
                                                                    llm_patch_metrics['patches_verified'] += 1
                                                                    logger.info(f"      ✓ Verified: output matches original")
                                                                elif verify_result.get('errors'):
                                                                    llm_patch_metrics['patches_verification_failed'] += 1
                                                                    logger.warning(f"      ⚠ Verification: {verify_result['errors'][0]}")
                                                            break
                                                        else:
                                                            # Update error info for next attempt
                                                            error_info = refined_result
                                                            patched_code = refined_code
                                
                                # Track refined patches
                                refined_count = sum(1 for p in benchmarkable if p.get('refined'))
                                if refined_count > 0:
                                    llm_patch_metrics['patches_refined'] += refined_count
                                
                                # Auto-select best patch
                                best_patch = _select_best_patch(benchmarkable, baseline_time)
                                if best_patch:
                                    bench_result['best_llm_patch'] = {
                                        'variant_name': best_patch.get('variant_name'),
                                        'patched_file': best_patch.get('patched_file'),
                                        'actual_speedup': best_patch.get('actual_speedup'),
                                        'median_ms': best_patch.get('rebenchmark_result', {}).get('median_ms'),
                                        'refined': best_patch.get('refined', False),
                                    }
                                    llm_patch_metrics['best_patches_selected'] += 1
                                    if best_patch.get('actual_speedup'):
                                        llm_patch_metrics['total_speedup_improvement'] += best_patch['actual_speedup']
                                    
                                    # Generate educational explanation if enabled
                                    if llm_explain and original_optimized_code:
                                        patched_file_path = Path(best_patch.get('patched_file', ''))
                                        if patched_file_path.exists():
                                            patched_code = _safe_read_text_with_warning(
                                                patched_file_path,
                                                label=f"best patched source code for {_patch_label(best_patch)}",
                                            )
                                            if patched_code is None:
                                                continue
                                            logger.info(f"    📚 Generating educational explanation...")
                                            emit_progress(
                                                "llm_explain",
                                                step=f"{chapter_name}:{bench_result['example']}",
                                                step_detail=f"explain start {_patch_label(best_patch)}",
                                            )
                                            explanation = _explain_best_patch_with_llm(
                                                best_patch,
                                                bench_result,
                                                original_optimized_code,
                                                patched_code,
                                                chapter_dir,
                                                llm_provider=llm_provider,
                                            )
                                            if explanation:
                                                bench_result['best_llm_patch']['explanation'] = explanation
                                                logger.info(f"    📚 Explanation: {explanation.get('technique_name', 'Unknown')}")
                                                emit_progress(
                                                    "llm_explain",
                                                    step=f"{chapter_name}:{bench_result['example']}",
                                                    step_detail=f"explain complete {_patch_label(best_patch)}",
                                                )
                                                
                                                # Save explanation to file
                                                explain_dir = chapter_dir / "llm_explanations"
                                                explain_dir.mkdir(exist_ok=True)
                                                explain_file = explain_dir / f"explanation_{bench_result['example']}.md"
                                                _save_explanation_markdown(explanation, bench_result, explain_file)
                                            else:
                                                emit_progress(
                                                    "llm_explain",
                                                    step=f"{chapter_name}:{bench_result['example']}",
                                                    step_detail=f"explain failed {_patch_label(best_patch)}",
                                                )

                                    promoted_file = promote_best_llm_patch(
                                        best_patch,
                                        bench_result,
                                        chapter_dir,
                                    )
                                    if promoted_file:
                                        bench_result['best_llm_patch']['promoted_file'] = promoted_file
                else:
                    emit_progress(
                        "llm_analysis",
                        step=f"{chapter_name}:{bench_result['example']}",
                        step_detail="analysis failed",
                    )

    # Calculate summary statistics
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    max_speedup = max(speedups) if speedups else 1.0
    min_speedup = min(speedups) if speedups else 1.0

    # Final status counts must be derived from finalized benchmark entries.
    # Per-optimization counters can over/under-count failure classes.
    final_status_counts: Dict[str, int] = {}
    for bench in benchmark_results:
        raw_status = str(bench.get("status", "unknown") or "unknown").strip().lower() or "unknown"
        final_status_counts[raw_status] = final_status_counts.get(raw_status, 0) + 1

    successful_final = final_status_counts.get("succeeded", 0)
    failed_error_final = final_status_counts.get("failed_error", 0)
    failed_verification_final = final_status_counts.get("failed_verification", 0)
    failed_regression_final = final_status_counts.get("failed_regression", 0)
    failed_no_speedup_final = final_status_counts.get("failed_no_speedup", 0)
    failed_profiler_final = final_status_counts.get("failed_profiler", 0)
    failed_plain_final = final_status_counts.get("failed", 0)
    total_failed = sum(
        count for status, count in final_status_counts.items() if status == "failed" or status.startswith("failed_")
    )
    failed_other_final = max(
        0,
        total_failed
        - failed_error_final
        - failed_verification_final
        - failed_regression_final
        - failed_no_speedup_final
        - failed_profiler_final
        - failed_plain_final,
    )
    total_skipped = final_status_counts.get("skipped", 0)

    logger.info("\n" + "-" * 80)
    logger.info(f"{chapter_name.upper()} SUMMARY")
    logger.info(
        f"Benchmarks: {len(benchmark_results)} | Succeeded: {successful_final} | "
        f"Failed: {total_failed} (errors={failed_error_final}, verification={failed_verification_final}, "
        f"regressions={failed_regression_final}, no_speedup={failed_no_speedup_final}, "
        f"profiler={failed_profiler_final}, generic={failed_plain_final}, "
        f"other={failed_other_final}) | "
        f"Skipped: {total_skipped} (HW: {skipped_hw}, Dist: {skipped_distributed}) | "
        f"Informational: {informational_skipped}"
    )
    if speedups:
        logger.info(f"Speedups collected: {len(speedups)} | Avg: {avg_speedup:.2f}x | Best: {max_speedup:.2f}x | Worst: {min_speedup:.2f}x")
    else:
        logger.info("No successful optimizations exceeded baseline performance")
    logger.info("-" * 80)
    emit_event(
        event_logger,
        logger,
        "chapter_end",
        chapter=chapter_name,
        status="completed",
        total_benchmarks=len(benchmark_results),
        successful=successful_final,
        failed=total_failed,
        failed_error=failed_error_final,
        failed_verification=failed_verification_final,
        failed_regression=failed_regression_final,
        failed_no_speedup=failed_no_speedup_final,
        failed_profiler=failed_profiler_final,
        failed_generic=failed_plain_final,
        failed_other=failed_other_final,
        skipped_hardware=skipped_hw,
        skipped_distributed=skipped_distributed,
        informational=informational_skipped,
        average_speedup=avg_speedup,
        max_speedup=max_speedup,
        min_speedup=min_speedup,
    )
    
    result = {
        'chapter': chapter_name,
        'status': 'completed',
        'benchmarks': benchmark_results,
        'manifests': manifest_entries,
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'successful': successful_final,
            'failed': total_failed,
            'failed_error': failed_error_final,
            'failed_verification': failed_verification_final,
            'failed_regression': failed_regression_final,
            'failed_no_speedup': failed_no_speedup_final,
            'failed_generic': failed_plain_final,
            'failed_other': failed_other_final,
            'skipped_hardware': skipped_hw,
            'skipped_distributed': skipped_distributed,
            'total_skipped': total_skipped,
            'total_speedups': len(speedups),
            'average_speedup': avg_speedup,
            'max_speedup': max_speedup,
            'min_speedup': min_speedup,
            'informational': informational_skipped,
        }
    }
    
    # Add LLM patch metrics if analysis was run
    if llm_analysis:
        result['llm_patch_metrics'] = llm_patch_metrics
        if llm_patch_metrics['total_analyzed'] > 0:
            logger.info(f"  LLM Analysis: {llm_patch_metrics['total_analyzed']} examples analyzed")
            logger.info(f"  Patches: {llm_patch_metrics['patches_applied']}/{llm_patch_metrics['patches_extracted']} applied, {llm_patch_metrics['patches_failed']} failed")
            if llm_patch_metrics['patches_rebenchmarked'] > 0:
                logger.info(f"  Rebenchmarked: {llm_patch_metrics['patches_rebenchmarked']} patches")
            if llm_patch_metrics['patches_refined'] > 0:
                logger.info(f"  Refined (after failure): {llm_patch_metrics['patches_refined']} patches")
            if llm_patch_metrics['best_patches_selected'] > 0:
                avg_improvement = llm_patch_metrics['total_speedup_improvement'] / llm_patch_metrics['best_patches_selected']
                logger.info(f"  🏆 Best patches selected: {llm_patch_metrics['best_patches_selected']} (avg {avg_improvement:.2f}x speedup)")
            if llm_patch_metrics['failures']:
                logger.info(f"  Failures:")
                for f in llm_patch_metrics['failures'][:5]:  # Show first 5
                    logger.info(f"    - {f['example']}: {f['reason'][:80]}")
    
    return result


def _compute_cache_key(baseline_code: Optional[str], optimized_code: Optional[str], speedup: float) -> str:
    """Compute a cache key based on source code content and speedup."""
    import hashlib
    content = f"{baseline_code or ''}{optimized_code or ''}{speedup:.4f}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _run_llm_analysis_for_benchmark(
    benchmark_result: Dict[str, Any],
    profiling_output_dir: Optional[Path],
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """Run LLM analysis on a benchmark that needs optimization.
    
    Args:
        use_cache: If True, check for cached analysis before running LLM.
    """
    from core.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    example_name = benchmark_result.get('example', '')
    
    # Build diff report from benchmark data
    baseline_time = benchmark_result.get('baseline_time_ms', 0)
    best_opt = None
    best_speedup = 0.0
    for opt in benchmark_result.get('optimizations', []):
        if opt.get('speedup', 0) > best_speedup:
            best_speedup = opt.get('speedup', 0)
            best_opt = opt
    
    optimized_time = best_opt.get('time_ms', baseline_time) if best_opt else baseline_time
    
    # Load source code
    baseline_code = None
    optimized_code = None
    source_warnings: List[str] = []
    
    for ext in ['.py', '.cu']:
        baseline_file = chapter_dir / f"baseline_{example_name}{ext}"
        if baseline_file.exists():
            baseline_code = _safe_read_text_with_warning(
                baseline_file,
                label=f"baseline source code for {example_name}",
                warnings_list=source_warnings,
            )
            if baseline_code is not None:
                break
    
    if best_opt and best_opt.get('file'):
        opt_file = chapter_dir / best_opt['file']
        if opt_file.exists():
            optimized_code = _safe_read_text_with_warning(
                opt_file,
                label=f"optimized source code for {example_name}",
                warnings_list=source_warnings,
            )
    
    # Setup output directory
    output_dir = chapter_dir / "llm_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_md_path = output_dir / f"llm_analysis_{example_name}.md"
    cache_key_path = output_dir / f".cache_key_{example_name}"
    
    # Check cache
    cache_key = _compute_cache_key(baseline_code, optimized_code, best_speedup)
    if use_cache and llm_md_path.exists() and cache_key_path.exists():
        cached_text = _safe_read_text_with_warning(
            cache_key_path,
            label=f"LLM cache key for {example_name}",
            warnings_list=source_warnings,
        )
        cached_key = cached_text.strip() if cached_text is not None else None
        if cached_key == cache_key:
            logger.info(f"    📦 Using cached LLM analysis for {example_name}")
            return {
                'md_path': str(llm_md_path),
                'provider': 'cached',
                'model': 'cached',
                'latency_seconds': 0.0,
                'cached': True,
                'warnings': source_warnings,
            }
    
    diff_report = {
        "overall_speedup": best_speedup if best_speedup > 0 else 1.0,
        "baseline_total_time_ms": baseline_time,
        "optimized_total_time_ms": optimized_time,
        "example_name": example_name,
        "status": benchmark_result.get('status', 'unknown'),
    }
    
    # Run analysis
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    env_ctx = collect_environment_context()
    
    result = analyzer.analyze_differential(
        diff_report,
        baseline_code=baseline_code,
        optimized_code=optimized_code,
        environment=env_ctx,
    )
    result.warnings.extend(w for w in source_warnings if w not in result.warnings)
    
    # Save output and cache key
    llm_md_path.write_text(result.to_markdown())
    cache_key_path.write_text(cache_key)
    
    return {
        'md_path': str(llm_md_path),
        'provider': result.provider,
        'model': result.model,
        'latency_seconds': result.latency_seconds,
        'warnings': source_warnings,
    }


def _apply_llm_patches_for_benchmark(
    benchmark_result: Dict[str, Any],
    llm_result: Dict[str, Any],
    chapter_dir: Path,
    profiling_output_dir: Optional[Path],
    patch_strategy: str = "ast",
    llm_provider: Optional[str] = None,
    max_refinement_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Apply LLM-suggested patches to create new optimized variants.
    
    If a patch fails with a syntax error, it will be sent back to the LLM
    for refinement (up to max_refinement_attempts times).
    """
    from core.analysis.llm_patch_applier import LLMPatchApplier
    
    md_path = llm_result.get('md_path')
    if not md_path or not Path(md_path).exists():
        return []

    warnings_list = llm_result.setdefault('warnings', [])
    llm_response = _safe_read_text_with_warning(
        Path(md_path),
        label=f"LLM analysis markdown for {benchmark_result.get('example', 'unknown')}",
        warnings_list=warnings_list,
    )
    if llm_response is None:
        return []
    
    applier = LLMPatchApplier(strategy=patch_strategy, dry_run=False, validate_syntax=True)
    patches = applier.extract_patches(llm_response)
    
    if not patches:
        return []
    
    # Find source file
    example_name = benchmark_result.get('example', '')
    optimizations = benchmark_result.get('optimizations', [])
    
    best_opt = None
    best_speedup = 0.0
    for opt in optimizations:
        if opt.get('status') == 'succeeded' and opt.get('speedup', 0) > best_speedup:
            best_speedup = opt.get('speedup', 0)
            best_opt = opt
    
    source_file = chapter_dir / f"optimized_{example_name}.py"
    if not source_file.exists():
        source_file = chapter_dir / f"baseline_{example_name}.py"
    if best_opt and best_opt.get('file'):
        source_file = chapter_dir / best_opt['file']
    
    if not source_file.exists():
        return []
    
    if profiling_output_dir:
        output_dir = profiling_output_dir / "llm_patches"
    else:
        chapter_id = chapter_slug(chapter_dir, repo_root)
        output_dir = default_artifacts_root(repo_root) / "llm_patches" / chapter_id / example_name
    output_dir.mkdir(parents=True, exist_ok=True)

    original_code = _safe_read_text_with_warning(
        source_file,
        label=f"source code for {example_name}",
        warnings_list=warnings_list,
    ) or ""
    results = applier.apply_patches(patches, source_file, output_dir)
    
    # Serialize results, with refinement for failures
    serializable = []
    for i, r in enumerate(results):
        if r.success:
            variant_name = getattr(r.patch, 'variant_name', '') if r.patch else ''
            serializable.append({
                'success': True,
                'patched_file': str(r.patched_file) if r.patched_file else None,
                'variant_name': variant_name,
                'description': getattr(r.patch, 'description', '') if r.patch else '',
                'expected_speedup': getattr(r.patch, 'expected_speedup', '') if r.patch else '',
                'validation_errors': r.validation_errors,
                'can_benchmark': not bool(r.validation_errors),
            })
        else:
            # Try refinement for failed patches (syntax errors, etc.)
            variant_name = getattr(r.patch, 'variant_name', f'patch_{i}') if r.patch else f'patch_{i}'
            patch_code = getattr(r.patch, 'code', '') if r.patch else ''
            error_msg = r.error or 'Unknown error'
            
            refined_successfully = False
            for attempt in range(max_refinement_attempts):
                if not patch_code:
                    break
                    
                logger.info(f"      🔄 Refining {variant_name} (attempt {attempt + 1}/{max_refinement_attempts})...")
                
                # Send to LLM for refinement
                refined_code = _refine_patch_with_llm(
                    original_code,
                    patch_code,
                    {'error': error_msg, 'error_type': 'syntax_error'},
                    benchmark_result,
                    chapter_dir,
                    llm_provider=llm_provider,
                )
                
                if refined_code:
                    # Try to apply the refined patch
                    refined_path = output_dir / f"optimized_{example_name}_{variant_name}_refined.py"
                    try:
                        refined_path.write_text(refined_code)
                        # Validate syntax
                        compile(refined_code, str(refined_path), 'exec')
                        logger.info(f"      ✓ {variant_name} refined successfully")
                        serializable.append({
                            'success': True,
                            'patched_file': str(refined_path),
                            'variant_name': f"{variant_name}_refined",
                            'description': getattr(r.patch, 'description', '') if r.patch else '',
                            'expected_speedup': getattr(r.patch, 'expected_speedup', '') if r.patch else '',
                            'validation_errors': [],
                            'can_benchmark': True,
                            'refined': True,
                            'refinement_attempts': attempt + 1,
                        })
                        refined_successfully = True
                        break
                    except SyntaxError as e:
                        error_msg = str(e)
                        patch_code = refined_code
                        logger.warning(f"      ✗ Refinement still has syntax error: {e}")
                else:
                    break
            
            if not refined_successfully:
                serializable.append({
                    'success': False,
                    'error': r.error,
                    'failure_reason': r.error,
                    'can_benchmark': False,
                    'variant_name': variant_name,
                })
                logger.warning(f"      ✗ Patch failed: {r.error}")
    
    return serializable


def _rebenchmark_patched_variant(
    patched_file: str,
    iterations: int = 3,
    warmup: int = 1,
    enable_profiling: bool = False,
    profile_type: str = "none",
    profile_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Re-benchmark a patched variant file.
    
    Returns:
        Dict with keys:
            - success: bool
            - time_ms, median_ms, min_ms, iterations (if success)
            - error: str (if failure)
            - error_type: str (if failure, e.g., 'import_error', 'runtime_error', 'cuda_error')
            - profile_path: str (if profiling enabled)
    """
    import importlib.util
    import traceback
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkConfig
    
    path = Path(patched_file)
    if not path.exists():
        return {'success': False, 'error': f"File not found: {patched_file}", 'error_type': 'file_not_found'}
    
    # Try to load the module
    try:
        spec = importlib.util.spec_from_file_location("patched_module", path)
        if not spec or not spec.loader:
            return {'success': False, 'error': f"Could not load module spec: {patched_file}", 'error_type': 'import_error'}
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as e:
        return {'success': False, 'error': f"Syntax error: {e}", 'error_type': 'syntax_error', 'patched_file': patched_file}
    except Exception as e:
        return {'success': False, 'error': f"Import error: {e}", 'error_type': 'import_error', 'patched_file': patched_file}
    
    # Find benchmark class (exclude BaseBenchmark itself)
    from core.harness.benchmark_harness import BaseBenchmark
    benchmark_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseBenchmark) and obj is not BaseBenchmark:
            benchmark_class = obj
            break
    
    if not benchmark_class:
        return {'success': False, 'error': f"No benchmark class found in: {patched_file}", 'error_type': 'class_not_found', 'patched_file': patched_file}
    
    # Try to run the benchmark
    try:
        config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            use_subprocess=False,
        )
        
        benchmark = benchmark_class()
        harness = BenchmarkHarness(config=config)
        
        result = harness.benchmark(benchmark)
        
        # Extract timing from result
        timing = getattr(result, 'timing', None)
        
        response = {
            'success': True,
            'time_ms': timing.median_ms if timing else None,
            'median_ms': timing.median_ms if timing else None,
            'min_ms': timing.min_ms if timing else None,
            'iterations': timing.iterations if timing else iterations,
            'patched_file': patched_file,
        }
        
        # Profile the patched variant if requested.
        if enable_profiling and profile_type != "none":
            patch_name = Path(patched_file).stem
            profile_env = _harden_profile_env(None, repo_root, path.parent)

            if profile_output_dir:
                profile_output_dir.mkdir(parents=True, exist_ok=True)

            profile_preset = str(profile_type).lower()
            config = replace(config, profile_type=profile_preset)
            profiler_config = build_profiler_config_from_benchmark(config)

            if check_nsys_available():
                try:
                    import subprocess
                    nsys_output = (
                        profile_output_dir / f"{patch_name}_nsys.nsys-rep"
                        if profile_output_dir
                        else Path(f"{patch_name}_nsys.nsys-rep")
                    )
                    cmd = profiler_config.get_nsys_command(
                        str(nsys_output.with_suffix("")),
                        patched_file,
                        python_executable=sys.executable,
                    )
                    subprocess.run(cmd, capture_output=True, timeout=120, env=profile_env)
                    if nsys_output.exists():
                        response["nsys_profile"] = str(nsys_output)
                except Exception as e:
                    logger.warning(f"Failed to profile patch with nsys: {e}")

            if check_ncu_available():
                try:
                    import subprocess
                    ncu_output = (
                        profile_output_dir / f"{patch_name}_ncu.ncu-rep"
                        if profile_output_dir
                        else Path(f"{patch_name}_ncu.ncu-rep")
                    )
                    cmd = profiler_config.get_ncu_command(
                        str(ncu_output.with_suffix("")),
                        patched_file,
                        python_executable=sys.executable,
                    )
                    cmd.insert(1, "--force-overwrite")
                    subprocess.run(cmd, capture_output=True, timeout=300, env=profile_env)
                    if ncu_output.exists():
                        response["ncu_profile"] = str(ncu_output)
                except Exception as e:
                    logger.warning(f"Failed to profile patch with ncu: {e}")
        
        return response
    except Exception as e:
        error_str = str(e)
        error_type = 'runtime_error'
        if 'CUDA' in error_str or 'cuda' in error_str:
            error_type = 'cuda_error'
        elif 'AttributeError' in str(type(e).__name__):
            error_type = 'attribute_error'
        
        return {
            'success': False,
            'error': f"{type(e).__name__}: {error_str}",
            'error_type': error_type,
            'traceback': traceback.format_exc()[-1000:],  # Last 1000 chars of traceback
            'patched_file': patched_file,
        }


def _verify_inputs_match(
    baseline_benchmark,
    optimized_benchmark,
    baseline_path: str,
    optimized_path: str,
) -> Dict[str, Any]:
    """Verify that baseline and optimized benchmarks have equivalent workloads.
    
    This is critical for benchmark validity: comparing performance of different
    workloads is meaningless. Without input verification, an "optimized" benchmark
    could simply be doing less work.
    
    Args:
        baseline_benchmark: Instantiated baseline benchmark
        optimized_benchmark: Instantiated optimized benchmark
        baseline_path: Path to baseline file (for error messages)
        optimized_path: Path to optimized file (for error messages)
        
    Returns:
        Dict with keys:
            - equivalent: bool (True if inputs match)
            - verification_type: str (e.g., 'input_signature', 'skipped')
            - mismatches: List[str] (description of any mismatches)
            - baseline_signature: Dict (baseline input signature)
            - optimized_signature: Dict (optimized input signature)
    """
    result = {
        'equivalent': False,
        'verification_type': 'input_signature',
        'mismatches': [],
        'baseline_signature': {},
        'optimized_signature': {},
    }
    
    # Check if benchmarks opt out of input verification
    baseline_skip = getattr(baseline_benchmark, 'skip_input_verification', lambda: False)()
    optimized_skip = getattr(optimized_benchmark, 'skip_input_verification', lambda: False)()
    
    if baseline_skip or optimized_skip:
        result['verification_type'] = 'skipped'
        result['equivalent'] = False  # STRICT: Skip flags are NON-COMPLIANT - must verify
        result['quarantine_reason'] = 'skip_flag_present'
        result['mismatches'].append(f"VERIFICATION REQUIRED: benchmark {'baseline' if baseline_skip else 'optimized'} has skip flag - remove flag and implement proper verification")
        return result
    
    # Get input signatures from both benchmarks
    baseline_sig_fn = getattr(baseline_benchmark, 'get_input_signature', None)
    optimized_sig_fn = getattr(optimized_benchmark, 'get_input_signature', None)
    
    if baseline_sig_fn and callable(baseline_sig_fn):
        try:
            result['baseline_signature'] = baseline_sig_fn() or {}
        except Exception as e:
            result['mismatches'].append(f"Failed to get baseline signature: {e}")
    
    if optimized_sig_fn and callable(optimized_sig_fn):
        try:
            result['optimized_signature'] = optimized_sig_fn() or {}
        except Exception as e:
            result['mismatches'].append(f"Failed to get optimized signature: {e}")
    
    baseline_sig = result['baseline_signature']
    optimized_sig = result['optimized_signature']
    
    # If neither has a signature, we can't verify - this is a FAILURE not a pass
    if not baseline_sig and not optimized_sig:
        result['verification_type'] = 'no_signature'
        result['equivalent'] = False  # STRICT: Cannot verify without signature - FAIL
        result['quarantine_reason'] = 'missing_input_signature'
        result['mismatches'].append("VERIFICATION REQUIRED: Neither benchmark provides input signature - implement get_input_signature()")
        return result
    
    # Compare signatures - exclude keys that are expected to differ between baseline/optimized
    # binary_name: CUDA binaries have different names (baseline_X vs optimized_X)
    # technique: optimization technique description varies
    # file_path: file paths are always different
    EXCLUDED_KEYS = {'binary_name', 'technique', 'file_path', 'name', 'friendly_name'}
    
    all_keys = set(baseline_sig.keys()) | set(optimized_sig.keys())
    all_keys -= EXCLUDED_KEYS
    
    for key in all_keys:
        baseline_val = baseline_sig.get(key)
        optimized_val = optimized_sig.get(key)
        
        if baseline_val is None and optimized_val is not None:
            result['mismatches'].append(f"{key}: baseline missing, optimized={optimized_val}")
        elif baseline_val is not None and optimized_val is None:
            result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized missing")
        elif baseline_val != optimized_val:
            # For numeric values, allow small tolerance
            if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
                if abs(baseline_val - optimized_val) > 1e-6 * max(abs(baseline_val), abs(optimized_val), 1):
                    result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized={optimized_val}")
            else:
                result['mismatches'].append(f"{key}: baseline={baseline_val}, optimized={optimized_val}")
    
    result['equivalent'] = len(result['mismatches']) == 0
    return result


# Global verification runner instance (lazily initialized)
_verify_runner: Optional['VerifyRunner'] = None


def _get_verify_runner() -> Optional['VerifyRunner']:
    """Get the global verification runner instance.
    
    Returns None if verification system is not available.
    """
    global _verify_runner
    if not VERIFICATION_AVAILABLE:
        return None
    if _verify_runner is None:
        _verify_runner = VerifyRunner()
    return _verify_runner


def _run_full_verification_suite(
    baseline_benchmark,
    optimized_benchmark,
    baseline_path: str,
    optimized_path: str,
    enforce: bool = True,
) -> Dict[str, Any]:
    """Run FULL verification suite including anti-hacking checks.
    
    This is the main verification entry point that runs ALL checks from the
    benchmark-verification-enforcement spec:
    - Input signature matching
    - Output comparison with dtype-aware tolerances
    - Fresh-input check (detect output caching with different seeds)
    - Jitter check (detect hardcoded outputs)
    - Workload invariant enforcement
    - Quarantine enforcement
    
    Args:
        baseline_benchmark: Instantiated baseline benchmark
        optimized_benchmark: Instantiated optimized benchmark  
        baseline_path: Path to baseline file
        optimized_path: Path to optimized file
        enforce: If True, block perf on verification failure
        
    Returns:
        Dict with verification results:
            - passed: bool
            - verification_type: str ('full_suite', 'legacy', 'skipped')
            - reason: str (if failed)
            - details: Dict (full verification details)
            - block_perf: bool (if perf should be blocked)
            - quarantine_reason: str (if quarantined)
    """
    import traceback
    
    result = {
        'passed': False,
        'verification_type': 'full_suite',
        'reason': None,
        'details': {},
        'block_perf': False,
        'quarantine_reason': None,
    }
    
    runner = _get_verify_runner()
    if runner is None:
        logger.error("Full verification suite not available - blocking performance execution")
        result['verification_type'] = 'unavailable'
        result['passed'] = False
        result['reason'] = "Full verification suite unavailable; perf is blocked until verification is installed"
        result['block_perf'] = True
        result['quarantine_reason'] = 'verification_unavailable'
        return result
    
    # Check enforcement phase
    try:
        phase = get_enforcement_phase()
    except Exception:
        phase = EnforcementPhase.DETECT if 'EnforcementPhase' in dir() else None
    
    # Check if benchmarks have skip flags (should quarantine)
    baseline_skip_flags = []
    optimized_skip_flags = []
    for flag in ['skip_output_check', 'skip_input_check', 'skip_verification']:
        if hasattr(baseline_benchmark, flag) and getattr(baseline_benchmark, flag):
            baseline_skip_flags.append(flag)
        if hasattr(optimized_benchmark, flag) and getattr(optimized_benchmark, flag):
            optimized_skip_flags.append(flag)
    
    if baseline_skip_flags or optimized_skip_flags:
        result['verification_type'] = 'skipped_with_flags'
        result['passed'] = False
        skip_info = []
        if baseline_skip_flags:
            skip_info.append(f"baseline: {baseline_skip_flags}")
        if optimized_skip_flags:
            skip_info.append(f"optimized: {optimized_skip_flags}")
        result['reason'] = f"Skip flags present ({', '.join(skip_info)}) - benchmarks with skip flags are non-compliant"
        result['quarantine_reason'] = 'skip_flag_present'
        
        # In GATE phase, block perf
        if phase == EnforcementPhase.GATE if phase else False:
            result['block_perf'] = True
        # In QUARANTINE phase, block from perf reports but don't fail CI
        elif phase == EnforcementPhase.QUARANTINE if phase else False:
            result['block_perf'] = True
        
        logger.warning(f"    ⚠ SKIP FLAGS DETECTED: {result['reason']}")
        return result
    
    # Run full verification suite
    try:
        config = VerifyConfig(
            seed=42,
            verbose=True,
        )
        
        verify_result = runner.verify_pair(baseline_benchmark, optimized_benchmark, config)
        
        result['passed'] = verify_result.passed
        result['details'] = {
            'signature_hash': verify_result.signature_hash,
            'baseline_checksum': verify_result.baseline_checksum,
            'optimized_checksum': verify_result.optimized_checksum,
            'seed_info': verify_result.seed_info,
        }
        
        if verify_result.comparison_details:
            result['details']['comparison'] = {
                'passed': verify_result.comparison_details.passed,
                'max_diff': verify_result.comparison_details.max_diff,
                'location': verify_result.comparison_details.location,
            }
        
        if verify_result.workload_delta:
            result['details']['workload_delta'] = verify_result.workload_delta
        
        if not verify_result.passed:
            result['reason'] = verify_result.reason
            
            # Determine quarantine reason
            reason_str = verify_result.reason.lower() if verify_result.reason else ''
            if 'signature' in reason_str:
                result['quarantine_reason'] = 'signature_mismatch'
            elif 'output' in reason_str or 'mismatch' in reason_str:
                result['quarantine_reason'] = 'output_mismatch'
            elif 'workload' in reason_str:
                result['quarantine_reason'] = 'workload_mismatch'
            elif 'jitter' in reason_str:
                result['quarantine_reason'] = 'jitter_fail'
            elif 'fresh' in reason_str or 'cache' in reason_str:
                result['quarantine_reason'] = 'cached_output_detected'
            elif 'compliance' in reason_str:
                if 'input_signature' in reason_str:
                    result['quarantine_reason'] = 'missing_input_signature'
                elif 'validate_result' in reason_str:
                    result['quarantine_reason'] = 'missing_validate_result'
                elif 'workload_metadata' in reason_str:
                    result['quarantine_reason'] = 'workload_metadata_missing'
                else:
                    result['quarantine_reason'] = 'non_compliant'
            else:
                result['quarantine_reason'] = 'verification_failed'
            
            # Quarantine the benchmark
            if runner.quarantine and baseline_path:
                runner.quarantine.quarantine(
                    baseline_path,
                    QuarantineReason(result['quarantine_reason']) if hasattr(QuarantineReason, result['quarantine_reason'].upper()) else QuarantineReason.MISSING_INPUT_SIGNATURE,
                    {'reason': verify_result.reason, 'optimized_path': optimized_path},
                )
            
            # Block perf based on enforcement phase
            if phase == EnforcementPhase.GATE if phase else False:
                result['block_perf'] = True
                logger.error(f"    ✗ FULL VERIFICATION FAILED (GATE mode): {verify_result.reason}")
                logger.error(f"      Perf measurement BLOCKED - speedup would be INVALID")
            elif phase == EnforcementPhase.QUARANTINE if phase else False:
                result['block_perf'] = True
                logger.warning(f"    ✗ FULL VERIFICATION FAILED (QUARANTINE mode): {verify_result.reason}")
                logger.warning(f"      Benchmark excluded from perf reports")
            else:
                # DETECT mode - just report
                logger.warning(f"    ⚠ FULL VERIFICATION FAILED (DETECT mode): {verify_result.reason}")
                logger.warning(f"      Perf will continue but results may be INVALID")
        else:
            # Verification passed!
            logger.info(f"    ✓ FULL VERIFICATION PASSED: signatures match, outputs match, anti-hacking checks passed")
            
            # Clear any existing quarantine
            if runner.quarantine and baseline_path:
                runner.quarantine.clear_quarantine(baseline_path)
        
        return result
        
    except Exception as e:
        logger.error(f"    ✗ VERIFICATION ERROR: {e}")
        result['passed'] = False
        result['reason'] = f"Verification exception: {e}"
        result['details']['exception'] = str(e)
        result['details']['traceback'] = traceback.format_exc()[-500:]
        
        # In GATE mode, verification errors should block perf
        if phase == EnforcementPhase.GATE if phase else False:
            result['block_perf'] = True
        
        return result


def _verify_patched_benchmark(
    original_file: str,
    patched_file: str,
    test_shape: tuple = (256, 256),
) -> Dict[str, Any]:
    """Verify that a patched benchmark produces the same output as the original.
    
    Uses the kernel verification tools to compare outputs.
    
    Args:
        original_file: Path to original optimized benchmark
        patched_file: Path to LLM-patched benchmark
        test_shape: Shape for test tensors
        
    Returns:
        Dict with keys:
            - verified: bool (True if outputs match)
            - verification_type: str (e.g., 'output_comparison', 'skipped')
            - errors: List[str] (if any verification errors)
            - details: Dict (additional info)
    """
    import importlib.util
    import torch
    import traceback
    from pathlib import Path
    
    result = {
        'verified': False,
        'verification_type': 'output_comparison',
        'errors': [],
        'details': {},
    }

    def _append_detail_warning(message: str, *, exc: Optional[BaseException] = None) -> str:
        warning = f"{message}: {exc}" if exc is not None else message
        warnings_list = result['details'].setdefault('warnings', [])
        if warning not in warnings_list:
            warnings_list.append(warning)
        return warning
    
    # Load both modules
    try:
        # Load original
        orig_path = Path(original_file)
        if not orig_path.exists():
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Original file not found: {original_file}")
            return result
        
        # Skip verification for CUDA files - they're not Python modules
        if orig_path.suffix == '.cu':
            result['verification_type'] = 'cuda_binary'
            result['verified'] = False  # STRICT: CUDA binaries need separate verification
            result['details']['reason'] = 'CUDA files require CudaBinaryBenchmark.get_verify_output() for checksum verification'
            result['quarantine_reason'] = 'cuda_no_verify_path'
            return result
        
        # Skip non-Python files
        if orig_path.suffix != '.py':
            result['verification_type'] = 'unsupported_file_type'
            result['verified'] = False  # STRICT: Cannot verify non-Python files without explicit handler
            result['details']['reason'] = f'Non-Python file ({orig_path.suffix}) - implement get_verify_output() for this file type'
            return result
            
        # Use unique module names to avoid collisions
        orig_module_name = f"_verify_orig_{orig_path.stem}_{id(result)}"
        orig_spec = importlib.util.spec_from_file_location(orig_module_name, orig_path)
        if orig_spec is None or orig_spec.loader is None:
            result['verification_type'] = 'module_load_failed'
            result['verified'] = False  # STRICT: Module load failure is verification failure
            result['details']['reason'] = f'Could not load module spec for {orig_path.name} - fix module or implement get_verify_output()'
            return result
        orig_module = importlib.util.module_from_spec(orig_spec)
        # Register module BEFORE exec_module - required for dataclasses and self-referential imports
        sys.modules[orig_module_name] = orig_module
        try:
            orig_spec.loader.exec_module(orig_module)
        finally:
            sys.modules.pop(orig_module_name, None)
        
        # Load patched
        patch_path = Path(patched_file)
        if not patch_path.exists():
            result['verification_type'] = 'skipped'
            result['errors'].append(f"Patched file not found: {patched_file}")
            return result
        
        # Skip non-Python files
        if patch_path.suffix != '.py':
            result['verification_type'] = 'unsupported_file_type'
            result['verified'] = False  # STRICT: Cannot verify non-Python files without explicit handler
            result['details']['reason'] = f'Non-Python file ({patch_path.suffix}) - implement get_verify_output() for this file type'
            return result
            
        patch_module_name = f"_verify_patch_{patch_path.stem}_{id(result)}"
        patch_spec = importlib.util.spec_from_file_location(patch_module_name, patch_path)
        if patch_spec is None or patch_spec.loader is None:
            result['verification_type'] = 'module_load_failed'
            result['verified'] = False  # STRICT: Module load failure is verification failure
            result['details']['reason'] = f'Could not load module spec for {patch_path.name} - fix module or implement get_verify_output()'
            return result
        patch_module = importlib.util.module_from_spec(patch_spec)
        # Register module BEFORE exec_module - required for dataclasses and self-referential imports
        sys.modules[patch_module_name] = patch_module
        try:
            patch_spec.loader.exec_module(patch_module)
        finally:
            sys.modules.pop(patch_module_name, None)
        
    except Exception as e:
        error_str = str(e)
        # Known compatibility issues during module loading - still requires resolution
        known_compat_issues = [
            "SymNodeVariable",  # torch.compile/dynamo issue with Triton
            "SymNode",          # Related symbolic shape issues  
            "SKIPPED:",         # Benchmark explicitly skipped
        ]
        if any(issue in error_str for issue in known_compat_issues):
            result['verification_type'] = 'compat_issue'
            result['verified'] = False  # STRICT: Compat issues need resolution, not bypass
            result['details']['reason'] = f'Known compatibility issue needs resolution: {error_str[:100]}'
            result['details']['compat_issue'] = next(i for i in known_compat_issues if i in error_str)
            return result
        result['errors'].append(f"Failed to load modules: {e}")
        return result
    
    # Find benchmark classes or instances via get_benchmark()
    from core.harness.benchmark_harness import BaseBenchmark
    
    def find_benchmark_class(module):
        """Find the benchmark class defined in the module, ignoring imported helpers."""
        candidates = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseBenchmark)
                and obj is not BaseBenchmark
            ):
                # Prefer classes defined in the module itself (not imported utilities)
                if getattr(obj, "__module__", "") == module.__name__:
                    candidates.append(obj)
        if candidates:
            return candidates[0]
        # Fallback: pick the first subclass that isn't one of the shared harness classes
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseBenchmark)
                and obj is not BaseBenchmark
                and not getattr(obj, "__module__", "").startswith("core.")
            ):
                return obj
        return None
    
    def get_benchmark_instance(module):
        """Try to get benchmark instance via get_benchmark() factory function."""
        if hasattr(module, 'get_benchmark'):
            try:
                return module.get_benchmark()
            except Exception:
                return None
        return None
    
    orig_class = find_benchmark_class(orig_module)
    patch_class = find_benchmark_class(patch_module)
    
    # If no class found, try get_benchmark() factory function (for wrapper modules)
    orig_instance = None
    patch_instance = None
    if not orig_class:
        orig_instance = get_benchmark_instance(orig_module)
        if orig_instance is not None:
            # Check skip flags on the instance - skip flags are NON-COMPLIANT
            orig_skip = getattr(orig_instance, 'skip_output_verification', lambda: False)()
            if not orig_skip:
                orig_skip = getattr(orig_instance, 'skip_output_check', False)
            if orig_skip:
                result['verification_type'] = 'skip_flag_present'
                result['verified'] = False  # STRICT: Skip flags are non-compliant
                result['quarantine_reason'] = 'skip_flag_present'
                result['details']['reason'] = 'VERIFICATION REQUIRED: Remove skip_output_check flag and implement get_verify_output()'
                return result
    
    if not patch_class:
        patch_instance = get_benchmark_instance(patch_module)
        if patch_instance is not None:
            patch_skip = getattr(patch_instance, 'skip_output_verification', lambda: False)()
            if not patch_skip:
                patch_skip = getattr(patch_instance, 'skip_output_check', False)
            if patch_skip:
                result['verification_type'] = 'skip_flag_present'
                result['verified'] = False  # STRICT: Skip flags are non-compliant
                result['quarantine_reason'] = 'skip_flag_present'
                result['details']['reason'] = 'VERIFICATION REQUIRED: Remove skip_output_check flag and implement get_verify_output()'
                return result
    
    if not orig_class and not orig_instance:
        result['verification_type'] = 'skipped'
        result['errors'].append("Could not find benchmark class or get_benchmark() in original")
        return result
    if not patch_class and not patch_instance:
        result['verification_type'] = 'skipped'
        result['errors'].append("Could not find benchmark class or get_benchmark() in patched")
        return result
    
    # Run both benchmarks with same seed and compare outputs
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Helper to instantiate benchmark, handling various signatures
        def instantiate_benchmark(cls, file_path):
            import inspect
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:]  # Skip 'self'
            
            # Check if all params have defaults
            required = [p for p in params if p.default is inspect.Parameter.empty]
            if not required:
                return cls()
            
            # Try to provide common required args
            kwargs = {}
            for p in required:
                if p.name == 'chapter_dir':
                    kwargs['chapter_dir'] = Path(file_path).parent
                elif p.name == 'binary_name':
                    kwargs['binary_name'] = Path(file_path).stem
                elif p.name == 'friendly_name':
                    kwargs['friendly_name'] = Path(file_path).stem.replace('_', ' ')
                else:
                    # Unknown required param - can't instantiate
                    return None
            return cls(**kwargs)
        
        # Run original - try instantiation first, fall back to get_benchmark()
        orig_benchmark = None
        if orig_class:
            orig_benchmark = instantiate_benchmark(orig_class, original_file)
        if orig_benchmark is None:
            # Fall back to get_benchmark() if class instantiation fails
            orig_benchmark = orig_instance or get_benchmark_instance(orig_module)
        if orig_benchmark is None:
            result['verification_type'] = 'skipped'
            class_name = orig_class.__name__ if orig_class else "unknown"
            result['errors'].append(f"Cannot instantiate {class_name} - unknown required args")
            return result
        
        # Check if either benchmark opts out of output verification - skip flags are NON-COMPLIANT
        orig_skip = getattr(orig_benchmark, 'skip_output_verification', lambda: False)()
        if not orig_skip:
            # Also check the attribute directly (some benchmarks use skip_output_check)
            orig_skip = getattr(orig_benchmark, 'skip_output_check', False)
        
        if orig_skip:
            result['verification_type'] = 'skip_flag_present'
            result['verified'] = False  # STRICT: Skip flags are non-compliant
            result['quarantine_reason'] = 'skip_flag_present'
            result['details']['reason'] = 'VERIFICATION REQUIRED: Remove skip_output_check flag and implement get_verify_output()'
            return result
            
        orig_benchmark.setup()
        orig_benchmark.benchmark_fn()
        try:
            orig_benchmark.capture_verification_payload()
        except Exception as e:
            result['verification_type'] = 'verification_payload_error'
            result['verified'] = False
            result['errors'].append(f"capture_verification_payload() failed for original: {e}")
            result['details']['reason'] = (
                'VERIFICATION REQUIRED: capture_verification_payload() failed for original benchmark'
            )
            result['details']['capture_phase'] = 'original'
            result['details']['capture_error'] = str(e)
            result['details']['capture_traceback'] = traceback.format_exc()[-500:]
            result['quarantine_reason'] = 'capture_verification_payload_error'
            try:
                orig_benchmark.teardown()
            except Exception as cleanup_exc:
                _append_detail_warning(
                    "Original benchmark teardown failed after capture_verification_payload() error",
                    exc=cleanup_exc,
                )
            return result
        # Prefer get_verify_output() method if available (consistent with FULL VERIFICATION)
        if hasattr(orig_benchmark, 'get_verify_output'):
            try:
                orig_output = orig_benchmark.get_verify_output()
            except Exception as exc:
                _append_detail_warning(
                    "Original benchmark get_verify_output() failed during patched verification",
                    exc=exc,
                )
                orig_output = None
        else:
            orig_output = getattr(orig_benchmark, 'output', None)
            if orig_output is None:
                # Try common attribute names (C is used by add benchmarks)
                for attr in ['result', 'y', 'out', 'output_tensor', 'C']:
                    orig_output = getattr(orig_benchmark, attr, None)
                    if orig_output is not None:
                        break
        
        # Reset seed and run patched
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Try instantiation first, fall back to get_benchmark()
        patch_benchmark = None
        if patch_class:
            patch_benchmark = instantiate_benchmark(patch_class, patched_file)
        if patch_benchmark is None:
            # Fall back to get_benchmark() if class instantiation fails
            patch_benchmark = patch_instance or get_benchmark_instance(patch_module)
        if patch_benchmark is None:
            result['verification_type'] = 'skipped'
            class_name = patch_class.__name__ if patch_class else "unknown"
            result['errors'].append(f"Cannot instantiate {class_name} - unknown required args")
            return result
        
        # Check if patched benchmark also opts out - skip flags are NON-COMPLIANT
        patch_skip = getattr(patch_benchmark, 'skip_output_verification', lambda: False)()
        if not patch_skip:
            patch_skip = getattr(patch_benchmark, 'skip_output_check', False)
        
        if patch_skip:
            result['verification_type'] = 'skip_flag_present'
            result['verified'] = False  # STRICT: Skip flags are non-compliant
            result['quarantine_reason'] = 'skip_flag_present'
            result['details']['reason'] = 'VERIFICATION REQUIRED: Remove skip_output_check flag and implement get_verify_output()'
            return result
            
        patch_benchmark.setup()
        patch_benchmark.benchmark_fn()
        try:
            patch_benchmark.capture_verification_payload()
        except Exception as e:
            result['verification_type'] = 'verification_payload_error'
            result['verified'] = False
            result['errors'].append(f"capture_verification_payload() failed for patched: {e}")
            result['details']['reason'] = (
                'VERIFICATION REQUIRED: capture_verification_payload() failed for patched benchmark'
            )
            result['details']['capture_phase'] = 'patched'
            result['details']['capture_error'] = str(e)
            result['details']['capture_traceback'] = traceback.format_exc()[-500:]
            result['quarantine_reason'] = 'capture_verification_payload_error'
            try:
                patch_benchmark.teardown()
            except Exception as cleanup_exc:
                _append_detail_warning(
                    "Patched benchmark teardown failed after capture_verification_payload() error",
                    exc=cleanup_exc,
                )
            try:
                orig_benchmark.teardown()
            except Exception as cleanup_exc:
                _append_detail_warning(
                    "Original benchmark teardown failed during patched capture_verification_payload() cleanup",
                    exc=cleanup_exc,
                )
            return result
        # Prefer get_verify_output() method if available (consistent with FULL VERIFICATION)
        if hasattr(patch_benchmark, 'get_verify_output'):
            try:
                patch_output = patch_benchmark.get_verify_output()
            except Exception as exc:
                _append_detail_warning(
                    "Patched benchmark get_verify_output() failed during patched verification",
                    exc=exc,
                )
                patch_output = None
        else:
            patch_output = getattr(patch_benchmark, 'output', None)
            if patch_output is None:
                # Try common attribute names (C is used by add benchmarks)
                for attr in ['result', 'y', 'out', 'output_tensor', 'C']:
                    patch_output = getattr(patch_benchmark, attr, None)
                    if patch_output is not None:
                        break
        
        # Compare outputs - STRICT: No output means verification FAILS
        if orig_output is None or patch_output is None:
            result['verification_type'] = 'no_output'
            which_missing = 'both' if (orig_output is None and patch_output is None) else ('original' if orig_output is None else 'patched')
            result['details']['reason'] = f'VERIFICATION REQUIRED: No output tensor found ({which_missing}) - implement get_verify_output()'
            result['details']['missing_output'] = which_missing
            result['verified'] = False  # STRICT: Cannot verify without outputs - FAIL
            result['quarantine_reason'] = 'missing_verify_output'
        elif isinstance(orig_output, torch.Tensor) and isinstance(patch_output, torch.Tensor):
            if orig_output.shape != patch_output.shape:
                result['errors'].append(f"Shape mismatch: {orig_output.shape} vs {patch_output.shape}")
            else:
                # Check if benchmarks specify custom tolerance (for precision comparison benchmarks)
                custom_tol = None
                for bm in [orig_benchmark, patch_benchmark]:
                    if hasattr(bm, 'get_output_tolerance'):
                        custom_tol = bm.get_output_tolerance()
                        if custom_tol:
                            break
                
                dtype = None
                if custom_tol:
                    rtol, atol = custom_tol
                elif orig_output is not None:
                    # Dtype-aware tolerances - reasonable for CUDA kernels
                    # CUDA operations have inherent non-determinism due to parallel execution order,
                    # different reduction tree structures, and fused multiply-add instructions.
                    # These tolerances are set to catch real bugs while allowing normal numerical variation.
                    dtype = orig_output.dtype
                    if dtype == torch.float32:
                        # FP32: 1e-3 relative, 1e-3 absolute (CUDA parallel reduction has ~1e-3 variance)
                        rtol, atol = 1e-3, 1e-3
                    elif dtype == torch.float16:
                        # FP16: 1e-2 relative/absolute (limited precision)
                        rtol, atol = 1e-2, 1e-2
                    elif dtype == torch.bfloat16:
                        # BF16: 1e-2 relative/absolute (7 mantissa bits = ~1% precision)
                        rtol, atol = 1e-2, 1e-2
                    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        # FP8: 5e-2 relative/absolute (very limited precision)
                        rtol, atol = 5e-2, 5e-2
                    else:
                        # Integer types: exact match
                        rtol, atol = 0, 0
                else:
                    result['details']['reason'] = 'VERIFICATION REQUIRED: Missing output from baseline - implement get_verify_output()'
                    result['status'] = 'failed'
                    result['verified'] = False  # STRICT: Missing output is verification failure
                    result['equivalent'] = False
                    result['quarantine_reason'] = 'missing_verify_output'
                    return result
                
                result['details']['dtype'] = str(dtype) if dtype is not None else 'unknown'
                result['details']['rtol'] = rtol
                result['details']['atol'] = atol
                
                max_diff = (orig_output.float() - patch_output.float()).abs().max().item()
                result['details']['max_diff'] = max_diff
                
                if torch.allclose(orig_output.float(), patch_output.float(), rtol=rtol, atol=atol):
                    result['verified'] = True
                else:
                    result['errors'].append(f"Output mismatch: max diff = {max_diff:.6f} (rtol={rtol}, atol={atol})")
        else:
            result['verification_type'] = 'non_tensor_output'
            orig_type = type(orig_output).__name__ if orig_output else 'None'
            patch_type = type(patch_output).__name__ if patch_output else 'None'
            result['details']['reason'] = f'VERIFICATION REQUIRED: Non-tensor outputs (orig={orig_type}, patch={patch_type}) - implement get_verify_output() to return tensor'
            result['verified'] = False  # STRICT: Non-tensor outputs need explicit handling
        
        # Cleanup
        orig_benchmark.teardown()
        patch_benchmark.teardown()
        
    except Exception as e:
        error_str = str(e)
        # Known PyTorch/Triton compatibility issues - still need resolution, not bypass
        known_compat_issues = [
            "SymNodeVariable",  # torch.compile/dynamo issue with Triton kernels
            "SymNode",          # Related symbolic shape issues
            "SKIPPED:",         # Benchmark explicitly skipped
        ]
        if any(issue in error_str for issue in known_compat_issues):
            result['verification_type'] = 'compat_issue'
            result['verified'] = False  # STRICT: Compat issues need resolution
            issue_type = next((i for i in known_compat_issues if i in error_str), 'unknown')
            result['details']['reason'] = f'Compatibility issue needs resolution ({issue_type}): {error_str[:100]}'
            result['details']['compat_issue'] = issue_type
        else:
            result['errors'].append(f"Verification error: {e}")
    
    return result


def _refine_patch_with_llm(
    original_code: str,
    patched_code: str,
    error_info: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
) -> Optional[str]:
    """Send a failed patch back to the LLM for refinement.
    
    Returns:
        New patched code if LLM provides a fix, None otherwise.
    """
    from core.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    environment = collect_environment_context()
    
    error_type = error_info.get('error_type', 'unknown')
    error_msg = error_info.get('error', 'Unknown error')
    traceback_str = error_info.get('traceback', '')
    
    # Build refinement prompt
    prompt = f"""## Patch Refinement Request

Your previous code patch failed during execution. Please analyze the error and provide a corrected version.

### Error Information
- **Error Type**: {error_type}
- **Error Message**: {error_msg}
- **Traceback** (last 1000 chars):
```
{traceback_str}
```

### Original Code (before your patch)
```python
{original_code[:8000]}
```

### Your Previous Patch (that failed)
```python
{patched_code[:8000]}
```

### Environment
- GPU: {environment.gpu_name} ({environment.gpu_arch})
- CUDA: {environment.cuda_version}
- PyTorch: {environment.pytorch_version}

### Instructions
Please provide a CORRECTED version of the patch that fixes the error. Common issues:
- **CUDA Graph errors**: Ensure all operations are captured correctly, avoid stream capture violations
- **AttributeError**: Make sure all instance attributes are defined in __init__
- **RuntimeError**: Check tensor shapes and device placement

Respond with the COMPLETE corrected code in a ```python code block.
"""
    
    try:
        response_tuple = analyzer._call_llm(prompt)
        if not response_tuple:
            return None
        
        # _call_llm returns (text, tokens)
        response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        
        # Extract the code block
        import re
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        return None
    except Exception as e:
        logger.warning(f"LLM refinement failed: {e}")
        return None


def _select_best_patch(
    patches: List[Dict[str, Any]],
    baseline_time_ms: float,
) -> Optional[Dict[str, Any]]:
    """Select the best patch based on rebenchmark results.
    
    Returns the patch with the best speedup, or None if no patches succeeded.
    """
    successful = [p for p in patches if p.get('rebenchmark_result', {}).get('success')]
    
    if not successful:
        return None
    
    # Calculate speedup for each patch
    for p in successful:
        rebench = p['rebenchmark_result']
        patch_time = rebench.get('median_ms')
        if patch_time and baseline_time_ms > 0:
            p['actual_speedup'] = baseline_time_ms / patch_time
        else:
            p['actual_speedup'] = 0
    
    # Sort by speedup descending
    successful.sort(key=lambda x: x.get('actual_speedup', 0), reverse=True)
    
    best = successful[0]
    logger.info(f"    🏆 Best patch: {best.get('variant_name', 'unknown')} with {best.get('actual_speedup', 0):.2f}x speedup")
    
    return best


def _save_explanation_markdown(
    explanation: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save the educational explanation as a markdown file."""
    example_name = benchmark_result.get('example', 'unknown')
    speedup = benchmark_result.get('best_llm_patch', {}).get('actual_speedup', 1.0)
    variant_name = benchmark_result.get('best_llm_patch', {}).get('variant_name', 'unknown')
    
    with open(output_path, 'w') as f:
        f.write(f"# 📚 Optimization Explanation: {example_name}\n\n")
        f.write(f"**Technique:** {explanation.get('technique_name', 'Unknown')}\n")
        f.write(f"**Variant:** {variant_name}\n")
        f.write(f"**Speedup Achieved:** {speedup:.2f}x\n\n")
        
        f.write("## What Changed?\n\n")
        f.write(f"{explanation.get('explanation', 'No explanation available.')}\n\n")
        
        f.write("## Why It Works\n\n")
        f.write(f"{explanation.get('why_it_works', 'No explanation available.')}\n\n")
        
        if explanation.get('key_concepts'):
            f.write("## Key Concepts to Understand\n\n")
            for concept in explanation.get('key_concepts', []):
                f.write(f"- {concept}\n")
            f.write("\n")
        
        if explanation.get('performance_impact'):
            f.write("## Performance Impact\n\n")
            perf = explanation['performance_impact']
            if perf.get('memory_bandwidth'):
                f.write(f"- **Memory Bandwidth:** {perf['memory_bandwidth']}\n")
            if perf.get('compute_utilization'):
                f.write(f"- **Compute Utilization:** {perf['compute_utilization']}\n")
            if perf.get('latency'):
                f.write(f"- **Latency:** {perf['latency']}\n")
            f.write("\n")
        
        if explanation.get('when_to_use'):
            f.write("## When to Use This Technique\n\n")
            f.write(f"{explanation['when_to_use']}\n\n")
        
        if explanation.get('when_not_to_use'):
            f.write("## When NOT to Use This Technique\n\n")
            f.write(f"{explanation['when_not_to_use']}\n\n")
        
        if explanation.get('further_reading'):
            f.write("## Further Reading\n\n")
            for topic in explanation.get('further_reading', []):
                f.write(f"- {topic}\n")
            f.write("\n")
        
        f.write("---\n")
        f.write("*Generated by LLM-powered benchmark analysis*\n")
    
    logger.info(f"      📄 Saved explanation to: {output_path}")


def _explain_best_patch_with_llm(
    best_patch: Dict[str, Any],
    benchmark_result: Dict[str, Any],
    original_code: str,
    patched_code: str,
    chapter_dir: Path,
    llm_provider: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate educational explanation for why the best patch works.
    
    Returns a dictionary with:
        - explanation: Plain-language explanation of the optimization
        - technique_name: Name of the optimization technique
        - technique_description: Educational description of the technique
        - why_it_works: Specific explanation for this use case
        - key_concepts: List of key concepts to understand
        - further_reading: Suggested topics for learning more
    """
    from core.analysis.llm_profile_analyzer import LLMProfileAnalyzer, collect_environment_context
    
    analyzer = LLMProfileAnalyzer(provider=llm_provider)
    environment = collect_environment_context()
    
    variant_name = best_patch.get('variant_name', 'unknown')
    actual_speedup = best_patch.get('actual_speedup', 1.0)
    baseline_time = benchmark_result.get('baseline_time_ms', 0)
    patch_time = best_patch.get('rebenchmark_result', {}).get('median_ms', 0)
    example_name = benchmark_result.get('example', 'unknown')
    
    prompt = f"""## Educational Explanation Request

You selected a code optimization that achieved a {actual_speedup:.2f}x speedup. Please explain this optimization in an educational way.

### Context
- **Benchmark**: {example_name}
- **Variant Name**: {variant_name}
- **Baseline Time**: {baseline_time:.3f}ms
- **Optimized Time**: {patch_time:.3f}ms
- **Speedup**: {actual_speedup:.2f}x
- **GPU**: {environment.gpu_name} ({environment.gpu_arch})
- **CUDA**: {environment.cuda_version}

### Original Code (before optimization)
```python
{original_code[:6000]}
```

### Optimized Code (your best patch)
```python
{patched_code[:6000]}
```

### Instructions

Provide an educational explanation in JSON format:

```json
{{
  "technique_name": "Name of the optimization technique (e.g., 'Memory Coalescing', 'Kernel Fusion', 'Stream Parallelism')",
  "explanation": "A 2-3 sentence plain-language explanation of what changed and why it's faster",
  "why_it_works": "Technical explanation of why this optimization works on this specific GPU architecture ({environment.gpu_arch})",
  "key_concepts": [
    "Concept 1: Brief explanation",
    "Concept 2: Brief explanation",
    "Concept 3: Brief explanation"
  ],
  "performance_impact": {{
    "memory_bandwidth": "How does this affect memory bandwidth utilization?",
    "compute_utilization": "How does this affect GPU compute utilization?",
    "latency": "How does this affect latency?"
  }},
  "when_to_use": "When should developers apply this optimization technique?",
  "when_not_to_use": "When might this optimization be counterproductive?",
  "further_reading": [
    "Topic 1 to learn more about",
    "Topic 2 to learn more about"
  ]
}}
```

Focus on being educational - help the user understand not just WHAT changed, but WHY it's faster and HOW they can apply similar optimizations in their own code.
"""
    
    try:
        response_tuple = analyzer._call_llm(prompt)
        if not response_tuple:
            return None
        
        # _call_llm returns (text, tokens)
        response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple
        
        # Extract JSON from response
        import re
        import json
        json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
        if json_match:
            explanation_data = json.loads(json_match.group(1))
            explanation_data['raw_response'] = response
            return explanation_data
        
        # Fallback: return raw response
        return {
            'explanation': response,
            'technique_name': variant_name,
            'raw_response': response,
        }
    except Exception as e:
        logger.warning(f"LLM explanation failed: {e}")
        return None


def test_chapter(
    chapter_dir: Path,
    enable_profiling: bool = False,
    profile_type: str = "none",
    profile_output_root: Optional[Path] = None,
    timeout_multiplier: float = 3.0,
    reproducible: bool = False,
    cold_start: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    single_gpu: bool = False,
    enforce_environment_validation: bool = True,
    allow_virtualization: bool = False,
    allow_foreign_gpu_processes: bool = False,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    only_examples: Optional[List[str]] = None,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    pm_sampling_interval: Optional[int] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    force_synchronize: bool = False,
    graph_capture_ratio_threshold: Optional[float] = None,
    graph_capture_memory_threshold_mb: Optional[float] = None,
    launch_via: str = "python",
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    env_passthrough: Optional[List[str]] = None,
    target_extra_args: Optional[Dict[str, List[str]]] = None,
    subprocess_stderr_dir: Optional[Path] = None,
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    only_cuda: bool = False,
    only_python: bool = False,
    # LLM analysis and patching options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
    progress_recorder: Optional[ProgressRecorder] = None,
    progress_completed_benchmarks: int = 0,
    progress_total_benchmarks: Optional[int] = None,
    event_logger: Optional["BenchmarkEventLogger"] = None,
    fail_on_no_benchmarks: bool = False,
) -> Dict[str, Any]:
    return _test_chapter_impl(
        chapter_dir,
        enable_profiling=enable_profiling,
        profile_type=profile_type,
        profile_output_root=profile_output_root,
        timeout_multiplier=timeout_multiplier,
        reproducible=reproducible,
        cold_start=cold_start,
        iterations=iterations,
        warmup=warmup,
        single_gpu=single_gpu,
        enforce_environment_validation=enforce_environment_validation,
        allow_virtualization=allow_virtualization,
        allow_foreign_gpu_processes=allow_foreign_gpu_processes,
        validity_profile=validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        graph_capture_ratio_threshold=graph_capture_ratio_threshold,
        graph_capture_memory_threshold_mb=graph_capture_memory_threshold_mb,
        only_examples=only_examples,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        pm_sampling_interval=pm_sampling_interval,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        force_synchronize=force_synchronize,
        launch_via=launch_via,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        env_passthrough=env_passthrough,
        target_extra_args=target_extra_args,
        subprocess_stderr_dir=subprocess_stderr_dir,
        verify_input=verify_input,
        verify_output=verify_output,
        force_llm=force_llm,
        only_cuda=only_cuda,
        only_python=only_python,
        llm_analysis=llm_analysis,
        llm_provider=llm_provider,
        apply_llm_patches=apply_llm_patches,
        rebenchmark_llm_patches=rebenchmark_llm_patches,
        patch_strategy=patch_strategy,
        llm_patch_retries=llm_patch_retries,
        use_llm_cache=use_llm_cache,
        llm_explain=llm_explain,
        progress_recorder=progress_recorder,
        progress_completed_benchmarks=progress_completed_benchmarks,
        progress_total_benchmarks=progress_total_benchmarks,
        event_logger=event_logger,
        fail_on_no_benchmarks=fail_on_no_benchmarks,
    )


def generate_markdown_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    *,
    bench_root: Optional[Path] = None,
) -> None:
    """Generate markdown summary report."""
    report_root = _resolve_report_root(bench_root)
    report_dir = output_path.parent
    has_llm_details = any(
        _benchmark_has_llm_data(bench)
        for result in results
        for bench in result.get("benchmarks", [])
    )
    with open(output_path, 'w') as f:
        f.write("# Benchmark Test Results Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        total_chapters = len(results)
        completed = sum(1 for r in results if r['status'] == 'completed')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        no_benchmarks = sum(1 for r in results if r['status'] == 'no_benchmarks')
        failed_chapters = sum(1 for r in results if str(r.get('status', '')).startswith('failed'))
        
        total_benchmarks = sum(r['summary']['total_benchmarks'] for r in results)
        total_successful = sum(r['summary']['successful'] for r in results)
        total_failed = sum(r['summary']['failed'] for r in results)
        
        all_speedups = []
        for r in results:
            if r['status'] == 'completed':
                for bench in r['benchmarks']:
                    if bench['status'] == 'succeeded':
                        all_speedups.append(bench['best_speedup'])
        
        avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 1.0
        
        f.write("## Overall Summary\n\n")
        f.write(f"- **Chapters tested:** {completed}/{total_chapters}\n")
        f.write(f"- **Chapters skipped:** {skipped} (CUDA unavailable)\n")
        f.write(f"- **Chapters failed:** {failed_chapters}\n")
        f.write(f"- **Chapters with no benchmarks:** {no_benchmarks}\n")
        total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in results)
        total_informational = sum(r['summary'].get('informational', 0) for r in results)
        total_regressions = sum(r['summary'].get('failed_regression', 0) for r in results)
        total_failed_no_speedup = sum(r['summary'].get('failed_no_speedup', 0) for r in results)
        total_failed_errors = sum(r['summary'].get('failed_error', 0) for r in results)
        total_failed_verification = sum(r['summary'].get('failed_verification', 0) for r in results)
        total_failed_generic = sum(r['summary'].get('failed_generic', 0) for r in results)
        total_failed_other = sum(r['summary'].get('failed_other', 0) for r in results)
        
        f.write(f"- **Total benchmarks:** {total_benchmarks}\n")
        f.write(f"- **Successful:** {total_successful}\n")
        f.write(f"- **Failed:** {total_failed}\n")
        f.write(f"  - Errors: {total_failed_errors}\n")
        f.write(f"  - Verification: {total_failed_verification}\n")
        f.write(f"  - Regressions: {total_regressions}\n")
        f.write(f"  - No speedup (<{_MIN_SPEEDUP_FOR_SUCCESS:.2f}x, speed goal): {total_failed_no_speedup}\n")
        f.write(f"  - Generic: {total_failed_generic}\n")
        f.write(f"  - Other failed_*: {total_failed_other}\n")
        f.write(f"- **Informational (not benchmarked):** {total_informational}\n")
        if total_skipped_hw > 0:
            f.write(f"- **WARNING: Skipped (hardware/software limitations):** {total_skipped_hw}\n")
        if all_speedups:
            f.write(f"- **Average speedup:** {avg_speedup:.2f}x\n")
            f.write(f"- **Best speedup:** {max(all_speedups):.2f}x\n")
            f.write(f"- **Worst speedup:** {min(all_speedups):.2f}x\n")
        f.write("\n")
        
        # LLM Patch Metrics (if any)
        total_llm_analyzed = sum(r.get('llm_patch_metrics', {}).get('total_analyzed', 0) for r in results)
        if total_llm_analyzed > 0:
            f.write("## LLM Analysis & Patching Summary\n\n")
            total_patches_extracted = sum(r.get('llm_patch_metrics', {}).get('patches_extracted', 0) for r in results)
            total_patches_applied = sum(r.get('llm_patch_metrics', {}).get('patches_applied', 0) for r in results)
            total_patches_failed = sum(r.get('llm_patch_metrics', {}).get('patches_failed', 0) for r in results)
            total_patches_rebenchmarked = sum(r.get('llm_patch_metrics', {}).get('patches_rebenchmarked', 0) for r in results)
            
            f.write(f"- **Benchmarks analyzed:** {total_llm_analyzed}\n")
            f.write(f"- **Patches extracted:** {total_patches_extracted}\n")
            f.write(f"- **Patches applied:** {total_patches_applied}\n")
            f.write(f"- **Patches failed:** {total_patches_failed}\n")
            if total_patches_rebenchmarked > 0:
                f.write(f"- **Patches re-benchmarked:** {total_patches_rebenchmarked}\n")
            
            # Collect all failures
            all_failures = []
            for r in results:
                all_failures.extend(r.get('llm_patch_metrics', {}).get('failures', []))
            
            if all_failures:
                f.write(f"\n### Patch Failures ({len(all_failures)})\n\n")
                f.write("| Example | Failure Reason |\n")
                f.write("|---------|----------------|\n")
                for failure in all_failures[:20]:  # Show first 20
                    reason = failure.get('reason', 'Unknown')[:100]
                    reason = reason.replace('|', '\\|').replace('\n', ' ')
                    f.write(f"| {failure.get('example', 'unknown')} | {reason} |\n")
                if len(all_failures) > 20:
                    f.write(f"\n*...and {len(all_failures) - 20} more failures*\n")
            f.write("\n")

        if has_llm_details:
            f.write("## LLM Transparency (Highlighted)\n\n")
            f.write("- **LLM analysis excerpts** embedded per benchmark, with links to full analysis.\n")
            f.write("- **Patch diffs** shown against the source file used for patching.\n")
            f.write("- **Re-benchmark + verification outcomes** included for each patch.\n")
            f.write("- **Best-patch explanations** linked when available.\n\n")
        
        # Per-chapter summary table
        f.write("## Per-Chapter Summary\n\n")
        f.write("| Chapter | Status | Benchmarks | Successful | Failed | Avg Speedup | Max Speedup |\n")
        f.write("|---------|--------|------------|------------|--------|-------------|-------------|\n")
        
        for r in sorted(results, key=lambda x: x['chapter']):
            status_emoji = {
                'completed': 'PASS',
                'skipped': 'SKIP',
                'no_benchmarks': 'WARN',
                'failed_no_benchmarks': 'FAIL',
            }.get(r['status'], 'UNKNOWN')
            
            summary = r['summary']
            avg_sp = summary.get('average_speedup', 0.0)
            max_sp = summary.get('max_speedup', 0.0)
            
            f.write(f"| {r['chapter']} | {status_emoji} | {summary['total_benchmarks']} | "
                   f"{summary['successful']} | {summary['failed']} | "
                   f"{avg_sp:.2f}x | {max_sp:.2f}x |\n")
        
        f.write("\n")
        
        # Detailed results per chapter
        f.write("## Detailed Results\n\n")
        for r in sorted(results, key=lambda x: x['chapter']):
            if r['status'] != 'completed':
                continue
            
            f.write(f"### {r['chapter'].upper()}\n\n")
            
            for bench in r['benchmarks']:
                bench_type = bench.get('type', 'python')
                f.write(f"**{bench['example']}**")
                if bench_type == 'cuda':
                    f.write(" *(CUDA)*")
                f.write("\n")
                f.write(f"- Baseline: `{bench['baseline_file']}`")
                if bench['baseline_time_ms']:
                    f.write(f" ({bench['baseline_time_ms']:.2f} ms)")
                profiler_links = []
                if bench.get('baseline_nsys_rep'):
                    profiler_links.append(f"[nsys](./{bench['baseline_nsys_rep']})")
                if bench.get('baseline_ncu_rep'):
                    profiler_links.append(f"[ncu](./{bench['baseline_ncu_rep']})")
                if bench.get('baseline_torch_trace'):
                    profiler_links.append(f"[torch](./{bench['baseline_torch_trace']})")
                if profiler_links:
                    f.write(f" | {' | '.join(profiler_links)}")
                f.write("\n")
                
                bench_status = bench.get('status')
                if bench_status == 'failed_error':
                    f.write(f"- Failed: {bench.get('error', 'Unknown error')}\n")
                elif bench_status == 'failed_verification':
                    f.write(f"- Verification failed: {bench.get('error', 'Correctness validation failed')}\n")
                elif bench_status == 'failed_regression':
                    f.write(f"- Regression: {bench.get('error', 'Expectation regression detected')}\n")
                elif bench_status == 'failed_no_speedup':
                    f.write(f"- No speedup: {bench.get('error', _format_failed_no_speedup(bench))}\n")
                elif bench_status == 'skipped':
                    f.write(f"- WARNING: **SKIPPED**: {bench.get('skip_reason', bench.get('error', 'Hardware/software limitation'))}\n")
                else:
                    for opt in bench['optimizations']:
                        if opt['status'] == 'succeeded':
                            f.write(f"- `{opt['file']}`: {opt['time_ms']:.2f} ms ({opt['speedup']:.2f}x speedup)")
                            profiler_links = []
                            if opt.get('optimized_nsys_rep'):
                                profiler_links.append(f"[nsys](./{opt['optimized_nsys_rep']})")
                            if opt.get('optimized_ncu_rep'):
                                profiler_links.append(f"[ncu](./{opt['optimized_ncu_rep']})")
                            if opt.get('optimized_torch_trace'):
                                profiler_links.append(f"[torch](./{opt['optimized_torch_trace']})")
                            if profiler_links:
                                f.write(f" | {' | '.join(profiler_links)}")
                            f.write("\n")
                        elif opt['status'] == 'skipped':
                            f.write(f"- `{opt['file']}`: WARNING: **SKIPPED** - {opt.get('skip_reason', opt.get('error', 'Hardware/software limitation'))}\n")
                        else:
                            f.write(f"- `{opt['file']}`: {opt.get('error', 'Failed')}\n")
                    
                    if bench['best_speedup'] > 1.0:
                        f.write(f"- Best speedup: {bench['best_speedup']:.2f}x\n")

                if _benchmark_has_llm_data(bench):
                    chapter_dir = report_root / r["chapter"]
                    f.write("\n#### LLM Analysis & Patch Diffs\n\n")

                    llm_result = bench.get("llm_analysis") or {}
                    llm_path = llm_result.get("md_path")
                    llm_md = Path(llm_path) if llm_path else None
                    if llm_md and llm_md.exists():
                        llm_link = _format_rel_link(llm_md, report_dir)
                        provider = llm_result.get("provider", "unknown")
                        model = llm_result.get("model", "unknown")
                        latency = llm_result.get("latency_seconds")
                        cached = llm_result.get("cached", False)
                        latency_str = f"{latency:.1f}s" if isinstance(latency, (int, float)) else "unknown"
                        f.write(f"- **LLM analysis:** [{llm_md.name}]({llm_link})\n")
                        f.write(f"  - Provider: {provider} | Model: {model} | Latency: {latency_str} | Cached: {cached}\n")
                        llm_warnings = llm_result.get("warnings") or []
                        if llm_warnings:
                            f.write("  - Warnings:\n")
                            for warning in llm_warnings:
                                f.write(f"    - {warning}\n")

                        llm_text = _safe_read_text(llm_md)
                        if llm_text:
                            why = _extract_markdown_section(llm_text, "## Why Is It Faster?")
                            root = _extract_markdown_section(llm_text, "## Root Cause Analysis")
                            suggested = _extract_markdown_section(llm_text, "## Suggested Code Changes")
                            missed = _extract_markdown_section(llm_text, "## Missed Optimization Opportunities")
                            excerpt_parts = []
                            for title, section in (
                                ("Why Is It Faster?", why),
                                ("Root Cause Analysis", root),
                                ("Suggested Code Changes", suggested),
                                ("Missed Optimization Opportunities", missed),
                            ):
                                if section:
                                    excerpt_parts.append(f"**{title}**\n\n{section}")
                            if not excerpt_parts:
                                excerpt_parts = [_truncate_text(llm_text)]
                            excerpt = "\n\n".join(excerpt_parts)
                            excerpt = _truncate_text(excerpt)
                            f.write("\n**LLM analysis excerpt:**\n\n")
                            f.write(_to_blockquote(excerpt))
                            f.write("\n\n")
                    else:
                        f.write("- **LLM analysis:** Not available in results\n\n")

                    explanation_path = chapter_dir / "llm_explanations" / f"explanation_{bench['example']}.md"
                    if explanation_path.exists():
                        explanation_link = _format_rel_link(explanation_path, report_dir)
                        f.write(f"- **Best-patch explanation:** [{explanation_path.name}]({explanation_link})\n")

                    best_patch = bench.get("best_llm_patch")
                    if best_patch:
                        best_variant = best_patch.get("variant_name", "unknown")
                        best_speedup = best_patch.get("actual_speedup")
                        best_speedup_str = f"{best_speedup:.2f}x" if isinstance(best_speedup, (int, float)) else "unknown"
                        f.write(f"- **Best LLM patch selected:** `{best_variant}` ({best_speedup_str} vs baseline)\n")
                        promoted = best_patch.get("promoted_file")
                        if promoted:
                            promoted_path = Path(promoted)
                            promoted_link = _format_rel_link(promoted_path, report_dir)
                            f.write(f"  - Promoted file: [{promoted_path.name}]({promoted_link})\n")

                    patches = bench.get("llm_patches") or []
                    source_file = _resolve_source_file(bench, chapter_dir)
                    patch_diff_warnings: List[str] = []
                    source_text = (
                        _safe_read_text_with_warning(
                            source_file,
                            label=f"patch diff source file for {bench['example']}",
                            warnings_list=patch_diff_warnings,
                        )
                        if source_file
                        else None
                    )
                    if source_file:
                        source_link = _format_rel_link(source_file, report_dir)
                        f.write(f"- **Patch diff base:** [{source_file.name}]({source_link})\n")
                    if patches:
                        f.write("\n**Patch outcomes:**\n\n")
                        f.write("| Variant | Expected | Actual | Median (ms) | Verified | Patch File |\n")
                        f.write("|---------|----------|--------|-------------|----------|------------|\n")
                        for patch in patches:
                            variant = patch.get("variant_name", "unknown")
                            expected = patch.get("expected_speedup", "unknown")
                            actual = patch.get("actual_speedup")
                            actual_str = f"{actual:.2f}x" if isinstance(actual, (int, float)) else "unknown"
                            rebench = patch.get("rebenchmark_result", {})
                            median = rebench.get("median_ms")
                            median_str = format_time_ms(median) if isinstance(median, (int, float)) else "n/a"
                            verification = patch.get("verification", {})
                            verified = verification.get("verified")
                            verified_str = "yes" if verified else ("no" if verified is not None else "n/a")
                            patched_file = patch.get("patched_file")
                            patch_link = "n/a"
                            if patched_file:
                                patched_path = Path(patched_file)
                                patch_link = f"[{patched_path.name}]({_format_rel_link(patched_path, report_dir)})"
                            f.write(f"| {variant} | {expected} | {actual_str} | {median_str} | {verified_str} | {patch_link} |\n")

                        f.write("\n")
                        for patch in patches:
                            patched_file = patch.get("patched_file")
                            if not patched_file:
                                continue
                            patched_path = Path(patched_file)
                            if not patched_path.exists() or not source_text:
                                continue
                            patched_text = _safe_read_text_with_warning(
                                patched_path,
                                label=f"patched variant file for {patch.get('variant_name', patched_path.name)}",
                                warnings_list=patch_diff_warnings,
                            )
                            if not patched_text:
                                continue
                            diff = _generate_diff(
                                source_text,
                                patched_text,
                                from_label=str(source_file.name if source_file else "source"),
                                to_label=str(patched_path.name),
                            )
                            if not diff:
                                continue
                            variant = patch.get("variant_name", patched_path.name)
                            f.write("<details>\n")
                            f.write(f"<summary>Patch diff: {variant}</summary>\n\n")
                            f.write("```diff\n")
                            f.write(diff)
                            f.write("\n```\n")
                            f.write("</details>\n\n")

                        failed_patches = [p for p in patches if not p.get("success")]
                        if failed_patches:
                            f.write("**Failed patches:**\n\n")
                            for patch in failed_patches:
                                variant = patch.get("variant_name", "unknown")
                                reason = patch.get("error") or patch.get("failure_reason") or "Unknown error"
                                f.write(f"- `{variant}`: {reason}\n")
                            f.write("\n")
                        if patch_diff_warnings:
                            f.write("**Patch diff warnings:**\n\n")
                            for warning in patch_diff_warnings:
                                f.write(f"- {warning}\n")
                            f.write("\n")
                
                f.write("\n")
            
            f.write("\n")


def _extract_cli_flag_value(argv: List[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 < len(argv):
                return argv[idx + 1]
            return None
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _resolve_target_override_path(
    target_extra_args: Dict[str, List[str]],
    *,
    chapter_name: str,
    flag: str,
) -> Optional[Path]:
    for target_key, argv in target_extra_args.items():
        chapter_token = target_key.split(":", 1)[0].replace("\\", "/")
        normalized = chapter_token.replace("/", "_")
        if normalized != chapter_name:
            continue
        value = _extract_cli_flag_value(list(argv), flag)
        if value and value.strip():
            return Path(value.strip()).expanduser()
    return None


def _phi35_default_model_path() -> Path:
    return repo_root / "phi-3.5-moe" / "original"


def _phi35_default_engine_candidates() -> List[Path]:
    return [
        repo_root / "phi-3.5-moe" / "trtllm_engine_tp1_fp16",
        repo_root / "phi-3.5-moe" / "trtllm_engine",
    ]


def _phi35_engine_has_assets(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if (path / "config.json").exists():
        return True
    if any(path.glob("rank*.engine")):
        return True
    if any(path.glob("*.engine")):
        return True
    if any(path.glob("*.plan")):
        return True
    return False


def _resolve_phi35_model_path(model_override: Optional[Path]) -> Path:
    if model_override is not None:
        return model_override
    env_model = os.environ.get("AISP_PHI35_MOE_MODEL_PATH", "").strip()
    if env_model:
        return Path(env_model).expanduser()
    return _phi35_default_model_path()


def _resolve_phi35_engine_path(engine_override: Optional[Path]) -> Path:
    if engine_override is not None:
        return engine_override
    env_engine = os.environ.get("AISP_PHI35_MOE_ENGINE_PATH", "").strip()
    if env_engine:
        return Path(env_engine).expanduser()
    for candidate in _phi35_default_engine_candidates():
        if _phi35_engine_has_assets(candidate):
            return candidate
    return _phi35_default_engine_candidates()[0]


def _preflight_target_coverage_and_assets(
    chapter_dirs: List[Path],
    chapter_filters: Dict[str, Set[str]],
    *,
    only_cuda: bool,
    only_python: bool,
    target_extra_args: Dict[str, List[str]],
) -> List[str]:
    issues: List[str] = []

    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, repo_root)
        chapter_name = chapter_id.replace("/", "_")
        example_filters = chapter_filters.get(chapter_id)

        python_pairs = discover_benchmarks(chapter_dir, warn_missing=False)
        if example_filters:
            python_pairs = [pair for pair in python_pairs if pair[2] in example_filters]
        if not example_filters:
            python_pairs = [
                pair
                for pair in python_pairs
                if pair[2] == pair[0].stem.replace("baseline_", "", 1)
            ]
        python_pairs, _ = _canonicalize_optimized_variants_for_full_sweep(
            python_pairs,
        )
        if only_cuda or only_python:
            cuda_wrapped_pairs = [pair for pair in python_pairs if _is_cuda_wrapper(pair[0])]
            if only_cuda:
                python_pairs = cuda_wrapped_pairs
            elif only_python:
                python_pairs = [pair for pair in python_pairs if pair not in cuda_wrapped_pairs]

        cuda_pairs = discover_cuda_benchmarks(chapter_dir)
        if example_filters:
            cuda_pairs = [pair for pair in cuda_pairs if pair[2] in example_filters]
        if only_python:
            cuda_pairs = []

        total = len(python_pairs) + len(cuda_pairs)
        if total == 0:
            examples = ", ".join(sorted(example_filters)) if example_filters else "<all>"
            issues.append(
                f"{chapter_name}: no runnable benchmark pairs discovered "
                f"(requested examples={examples}, only_cuda={only_cuda}, only_python={only_python})."
            )

        if chapter_name == "labs_trtllm_phi_3_5_moe" and not only_cuda:
            model_override = _resolve_target_override_path(
                target_extra_args,
                chapter_name=chapter_name,
                flag="--model-path",
            )
            engine_override = _resolve_target_override_path(
                target_extra_args,
                chapter_name=chapter_name,
                flag="--engine-path",
            )
            model_path = _resolve_phi35_model_path(model_override)
            if not model_path.exists():
                issues.append(
                    "labs_trtllm_phi_3_5_moe: missing model assets at "
                    f"{model_path}. Remediation: set AISP_PHI35_MOE_MODEL_PATH or pass "
                    "--target-extra-arg labs/trtllm_phi_3_5_moe:optimized_trtllm_phi_3_5_moe=\"--model-path /path/to/model\". "
                    f"Canonical repo default: {_phi35_default_model_path()}."
                )

            engine_path = _resolve_phi35_engine_path(engine_override)
            if not _phi35_engine_has_assets(engine_path):
                defaults_hint = ", ".join(str(p) for p in _phi35_default_engine_candidates())
                issues.append(
                    "labs_trtllm_phi_3_5_moe: missing TensorRT-LLM engine artifacts at "
                    f"{engine_path}. Remediation: set AISP_PHI35_MOE_ENGINE_PATH or pass "
                    "--target-extra-arg labs/trtllm_phi_3_5_moe:optimized_trtllm_phi_3_5_moe=\"--engine-path /path/to/engine_dir_or_plan\". "
                    f"Canonical repo defaults: {defaults_hint}."
                )

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Test all benchmarks and generate summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        help=("Space-separated list of targets. "
              "Use 'ch03' to test an entire chapter or 'ch03:resnet_50' "
              "to run baseline_resnet_50 and optimized_resnet_50. "
              "Omit this flag (or pass 'all') to run every chapter.")
    )
    parser.add_argument(
        '--bench-root',
        type=Path,
        default=None,
        help="Root directory to scan for benchmarks (defaults to repo root)."
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=repo_root / 'benchmark_test_results.json',
        help='Output file path (default: benchmark_test_results.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--profile',
        choices=['none', 'minimal', 'deep_dive', 'roofline'],
        default='minimal',
        help='Profiling preset: minimal (default), none, deep_dive, or roofline. Non-none enables nsys/ncu/PyTorch profiling.'
    )
    parser.add_argument(
        '--reproducible',
        action='store_true',
        help='Force deterministic seeds/algorithms for reproducible comparisons (slower fallbacks; ops without deterministic support may fail).'
    )
    parser.add_argument(
        '--cold-start',
        action='store_true',
        help='Reset CUDA/GPU state aggressively between benchmarks to emulate cold-start runs.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Override iteration count for Python benchmarks (default: 20 unless the benchmark defines its own).'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=None,
        help='Override warmup iteration count for Python benchmarks (default: 5 unless the benchmark defines its own).'
    )
    parser.add_argument(
        '--accept-regressions',
        action='store_true',
        help='Update expectation files when regressions are detected instead of failing the run.'
    )
    parser.add_argument(
        '--update-expectations',
        action='store_true',
        help='Force-write observed metrics into expectation files (overrides regressions and provenance checks).'
    )
    parser.add_argument(
        '--allow-mixed-provenance',
        action='store_true',
        help='Allow expectation updates when provenance differs (commit/hardware/profile mismatch) without forcing updates. Does NOT accept regressions (use --accept-regressions or --update-expectations).'
    )
    parser.add_argument(
        '--launch-via',
        choices=['python', 'torchrun'],
        default='python',
        help='Launcher to use for benchmarks (python or torchrun).'
    )
    parser.add_argument(
        '--nproc-per-node',
        type=int,
        default=None,
        help='torchrun --nproc_per_node value.'
    )
    parser.add_argument(
        '--nnodes',
        type=str,
        default=None,
        help='torchrun --nnodes value.'
    )
    parser.add_argument(
        '--rdzv-backend',
        type=str,
        default=None,
        help='torchrun rendezvous backend (default: c10d when nnodes is set).'
    )
    parser.add_argument(
        '--rdzv-endpoint',
        type=str,
        default=None,
        help='torchrun rendezvous endpoint (host:port).'
    )
    parser.add_argument(
        '--torchrun-env',
        action='append',
        default=None,
        help='Environment variables to forward into torchrun launches (repeatable).'
    )
    parser.add_argument(
        '--target-extra-arg',
        action='append',
        default=None,
        help='Per-target extra args, format: target="--flag value" (repeatable).'
    )
    parser.add_argument(
        '--only-cuda',
        action='store_true',
        help='Run only CUDA binary benchmarks (Python wrappers).'
    )
    parser.add_argument(
        '--only-python',
        action='store_true',
        help='Run only Python benchmarks (skip CUDA binary wrappers).'
    )
    parser.add_argument(
        '--force-sync',
        action='store_true',
        help='Force a device-wide synchronize immediately after benchmark_fn() (opt-in safeguard).'
    )
    parser.add_argument(
        '--timeout-multiplier',
        type=float,
        default=3.0,
        help='Multiply all benchmark timeouts by this factor (e.g., 2.0 doubles every timeout).'
    )
    parser.add_argument(
        '--validity-profile',
        choices=list(VALIDITY_PROFILE_CHOICES),
        default=VALIDITY_PROFILE_CHOICES[0],
        help=VALIDITY_PROFILE_HELP_TEXT,
    )
    parser.add_argument(
        '--allow-portable-expectations-update',
        action='store_true',
        help=PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
    )
    parser.add_argument(
        '--allow-foreign-gpu-processes',
        action='store_true',
        help='Warn instead of fail when unrelated CUDA compute processes are active on the benchmark GPU. Use only on shared hosts when strict isolation is impossible.',
    )
    parser.add_argument(
        '--nsys-timeout-seconds',
        type=int,
        default=None,
        help='Override Nsight Systems timeout in seconds (default from BenchmarkDefaults).'
    )
    parser.add_argument(
        '--ncu-timeout-seconds',
        type=int,
        default=None,
        help='Override Nsight Compute timeout in seconds (default from BenchmarkDefaults).'
    )
    parser.add_argument(
        '--ncu-metric-set',
        choices=['auto', 'minimal', 'deep_dive', 'roofline'],
        default='auto',
        help='Nsight Compute metric preset (auto/minimal/deep_dive/roofline). Auto follows the profile preset.'
    )
    parser.add_argument(
        '--pm-sampling-interval',
        type=int,
        default=None,
        help='Nsight Compute pm-sampling-interval (cycles between samples). Optional; leave unset to skip the flag.'
    )
    parser.add_argument(
        '--graph-capture-ratio-threshold',
        type=float,
        default=None,
        help='Max allowed capture/replay time ratio before flagging graph capture cheat (default from BenchmarkDefaults).'
    )
    parser.add_argument(
        '--graph-capture-memory-threshold-mb',
        type=float,
        default=None,
        help='Memory allocated during graph capture above this threshold (MB) is considered suspicious (default from BenchmarkDefaults).'
    )
    
    args = parser.parse_args()
    logger.info(
        "Benchmark validity profile (--validity-profile): %s",
        (
            "strict (fail-fast, full validity enforcement)"
            if args.validity_profile == "strict"
            else "portable (compatibility mode; some strict checks are relaxed)"
        ),
    )
    if (
        args.validity_profile == "portable"
        and not bool(getattr(args, "allow_portable_expectations_update", False))
        and (
            bool(getattr(args, "update_expectations", False))
            or bool(getattr(args, "accept_regressions", False))
            or bool(getattr(args, "allow_mixed_provenance", False))
        )
    ):
        logger.error(
            "Portable validity profile does not write expectations unless "
            "--allow-portable-expectations-update is set."
        )
        sys.exit(1)
    # Keep "auto" aligned with the profile preset so `--profile minimal` stays fast.
    if getattr(args, "ncu_metric_set", None) == "auto" and getattr(args, "profile", None) in {
        "minimal",
        "deep_dive",
        "roofline",
    }:
        args.ncu_metric_set = args.profile
    active_bench_root = Path(args.bench_root).resolve() if args.bench_root else repo_root
    if args.output == repo_root / 'benchmark_test_results.json' and args.bench_root:
        args.output = active_bench_root / 'benchmark_test_results.json'

    # Refresh benchmark defaults.
    defaults = BenchmarkDefaults()
    set_defaults(defaults)
    extra_arg_map: Dict[str, List[str]] = {}
    for entry in args.target_extra_arg or []:
        target, sep, payload = entry.partition("=")
        if not sep or not target or not payload:
            continue
        extra_arg_map[target.strip()] = shlex.split(payload)
    
    logger.info("=" * 80)
    logger.info("TESTING ALL BENCHMARKS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Target override: {args.targets}")
    logger.info(f"Bench root: {active_bench_root}")
    event_run_id = build_run_id("bench-main", base_dir=default_artifacts_root(active_bench_root))
    os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] = event_run_id
    os.environ["AISP_BENCHMARK_OWNER_PID"] = str(os.getpid())
    atexit.register(_reap_run_descendants, "run_exit")
    _reap_benchmark_process_leftovers(
        "run_start",
        current_run_id=event_run_id,
        current_owner_pid=os.getpid(),
        repo_root=repo_root,
    )
    event_log_path = args.output.parent / "benchmark_events.jsonl"
    event_logger = BenchmarkEventLogger(event_log_path, event_run_id, logger)
    defaults = get_defaults() or BenchmarkDefaults()
    gpu_state = get_gpu_state(allow_telemetry_failures=args.validity_profile == "portable")
    emit_event(
        event_logger,
        logger,
        "run_start",
        targets=args.targets or ["all"],
        profile_type=args.profile,
        ncu_metric_set=args.ncu_metric_set,
        timeout_multiplier=args.timeout_multiplier,
        nsys_timeout_seconds=args.nsys_timeout_seconds,
        ncu_timeout_seconds=args.ncu_timeout_seconds,
        validity_profile=args.validity_profile,
        allow_foreign_gpu_processes=bool(args.allow_foreign_gpu_processes),
        update_expectations=args.update_expectations,
        allow_portable_expectations_update=bool(args.allow_portable_expectations_update),
        allow_mixed_provenance=args.allow_mixed_provenance,
        output_json=str(args.output),
        events_file=str(event_log_path),
        gpu_clock_lock_enabled=defaults.lock_gpu_clocks,
        gpu_sm_clock_mhz=defaults.gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=defaults.gpu_mem_clock_mhz,
        gpu_app_clock_mhz=gpu_state.get("gpu_app_clock_mhz"),
        memory_app_clock_mhz=gpu_state.get("memory_app_clock_mhz"),
        gpu_clock_mhz=gpu_state.get("gpu_clock_mhz"),
        memory_clock_mhz=gpu_state.get("memory_clock_mhz"),
    )

    # Determine chapters to test and fail fast on unresolved target requirements.
    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(args.targets, bench_root=active_bench_root)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"ERROR: {exc}")
        _reap_run_descendants("target_resolution_failure")
        event_logger.close()
        sys.exit(1)
    preflight_issues = _preflight_target_coverage_and_assets(
        chapter_dirs,
        chapter_filters,
        only_cuda=bool(args.only_cuda),
        only_python=bool(args.only_python),
        target_extra_args=extra_arg_map,
    )
    if preflight_issues:
        for issue in preflight_issues:
            logger.error("PREFLIGHT FAILED: %s", issue)
        emit_event(
            event_logger,
            logger,
            "run_end",
            preflight_failed=True,
            issues=preflight_issues,
        )
        _reap_run_descendants("preflight_failure")
        event_logger.close()
        return 1

    dump_environment_and_capabilities()
    logger.info("")
    
    # Dump hardware capabilities at start - MUST succeed
    logger.info("Dumping hardware capabilities...")
    dump_caps_path = repo_root / "core" / "scripts" / "utilities" / "dump_hardware_capabilities.py"
    if not dump_caps_path.exists():
        raise FileNotFoundError(
            f"Hardware capabilities script not found: {dump_caps_path}\n"
            f"Expected: {dump_caps_path.resolve()}\n"
            f"This is a critical configuration error."
        )
    result = _run_repo_python_module(
        "core.scripts.utilities.dump_hardware_capabilities",
        "--fast",
        timeout=15,
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    
    # Pre-compile CUDA extensions before running benchmarks - MUST succeed
    logger.info("Pre-compiling CUDA extensions...")
    precompile_path = repo_root / "core" / "scripts" / "utilities" / "precompile_cuda_extensions.py"
    if not precompile_path.exists():
        raise FileNotFoundError(
            f"Pre-compilation script not found: {precompile_path}\n"
            f"Expected: {precompile_path.resolve()}\n"
            f"This is a critical configuration error."
        )
    result = _run_repo_python_module(
        "core.scripts.utilities.precompile_cuda_extensions",
        timeout=300,
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    logger.info("")

    profile_output_root = None
    if args.profile != "none":
        profile_run_id = build_run_id(
            "bench-profile",
            f"profile-{args.profile}",
            base_dir=default_artifacts_root(active_bench_root),
        )
        profile_output_root = default_artifacts_root(active_bench_root) / profile_run_id / "profiles"

    chapter_progress_totals: Dict[Path, int] = {}
    global_total_benchmarks = 0
    for chapter_dir in chapter_dirs:
        if not chapter_dir.exists():
            continue
        example_filters = chapter_filters.get(chapter_slug(chapter_dir, repo_root))
        only_examples = sorted(example_filters) if example_filters else None
        planned_python_pairs, planned_cuda_pairs, _, _, _ = _discover_chapter_benchmark_pairs(
            chapter_dir,
            only_examples=only_examples,
            only_cuda=bool(args.only_cuda),
            only_python=bool(args.only_python),
        )
        planned_total = len(planned_python_pairs) + len(planned_cuda_pairs)
        chapter_progress_totals[chapter_dir.resolve()] = planned_total
        global_total_benchmarks += planned_total
    logger.info(
        "Progress planning: %d benchmark(s) across %d chapter target(s)",
        global_total_benchmarks,
        len(chapter_progress_totals),
    )
    
    # Test all chapters
    all_results = []
    completed_benchmarks = 0
    explicit_targets = bool(
        args.targets and not any(str(token).strip().lower() == "all" for token in args.targets)
    )
    for chapter_idx, chapter_dir in enumerate(chapter_dirs):
        if not chapter_dir.exists():
            continue

        # GPU reset and cleanup between chapters to prevent state corruption
        if chapter_idx > 0:
            logger.info(f"\n  Resetting GPU state before {chapter_dir.name}...")
            _reset_parent_execution_state(include_gpu_state=True)
        _reap_benchmark_process_leftovers(
            f"chapter_start:{chapter_dir.name}",
            current_run_id=event_run_id,
            current_owner_pid=os.getpid(),
            repo_root=repo_root,
        )
        
        # Clean build directories to prevent stale lock issues
        build_dir = chapter_dir / "build"
        if build_dir.exists():
            try:
                from core.utils.build_utils import ensure_clean_build_directory
                ensure_clean_build_directory(build_dir)
            except ImportError as exc:
                logger.warning(
                    "  build_utils unavailable during final chapter cleanup for %s: %s",
                    chapter_dir.name,
                    exc,
                )
            except Exception as e:
                logger.warning(f"  Failed to clean build directory: {e}")

        example_filters = chapter_filters.get(chapter_slug(chapter_dir, repo_root))
        only_examples = sorted(example_filters) if example_filters else None
        chapter_progress_total = chapter_progress_totals.get(chapter_dir.resolve(), 0)
        result = test_chapter(
            chapter_dir,
            enable_profiling=args.profile != 'none',
            profile_type=args.profile,
            profile_output_root=profile_output_root,
            timeout_multiplier=args.timeout_multiplier,
            reproducible=args.reproducible,
            cold_start=args.cold_start,
            iterations=args.iterations,
            warmup=args.warmup,
            only_examples=only_examples,
            validity_profile=args.validity_profile,
            allow_foreign_gpu_processes=bool(args.allow_foreign_gpu_processes),
            allow_portable_expectations_update=bool(args.allow_portable_expectations_update),
            accept_regressions=args.accept_regressions if hasattr(args, "accept_regressions") else False,
            update_expectations=args.update_expectations if hasattr(args, "update_expectations") else False,
            allow_mixed_provenance=args.allow_mixed_provenance if hasattr(args, "allow_mixed_provenance") else False,
            ncu_metric_set=args.ncu_metric_set,
            pm_sampling_interval=args.pm_sampling_interval,
            nsys_timeout_seconds=args.nsys_timeout_seconds,
            ncu_timeout_seconds=args.ncu_timeout_seconds,
            graph_capture_ratio_threshold=args.graph_capture_ratio_threshold,
            graph_capture_memory_threshold_mb=args.graph_capture_memory_threshold_mb,
            launch_via=args.launch_via,
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            rdzv_backend=args.rdzv_backend,
            rdzv_endpoint=args.rdzv_endpoint,
            env_passthrough=args.torchrun_env,
            target_extra_args=extra_arg_map,
            only_cuda=bool(args.only_cuda),
            only_python=bool(args.only_python),
            force_synchronize=bool(args.force_sync),
            progress_completed_benchmarks=completed_benchmarks,
            progress_total_benchmarks=global_total_benchmarks,
            event_logger=event_logger,
            fail_on_no_benchmarks=explicit_targets,
        )
        all_results.append(result)
        completed_benchmarks += chapter_progress_total
        if args.format in ['json', 'both']:
            with open(args.output, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                }, f, indent=2)
            logger.info(f"\nJSON results checkpoint saved to: {args.output}")
    
    # Save results
    output_json = args.output
    output_md = args.output.with_suffix('.md')
    
    if args.format in ['json', 'both']:
        with open(output_json, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)
        logger.info(f"\nJSON results saved to: {output_json}")
    
    if args.format in ['markdown', 'both']:
        generate_markdown_report(all_results, output_md, bench_root=active_bench_root)
        logger.info(f"Markdown report saved to: {output_md}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    total_benchmarks = sum(r['summary']['total_benchmarks'] for r in all_results)
    total_successful = sum(r['summary']['successful'] for r in all_results)
    total_failed = sum(r['summary']['failed'] for r in all_results)
    total_failed_errors = sum(r['summary'].get('failed_error', 0) for r in all_results)
    total_failed_verification = sum(r['summary'].get('failed_verification', 0) for r in all_results)
    total_failed_regressions = sum(r['summary'].get('failed_regression', 0) for r in all_results)
    total_failed_no_speedup = sum(r['summary'].get('failed_no_speedup', 0) for r in all_results)
    total_failed_generic = sum(r['summary'].get('failed_generic', 0) for r in all_results)
    total_failed_other = sum(r['summary'].get('failed_other', 0) for r in all_results)
    total_skipped_hw = sum(r['summary'].get('skipped_hardware', 0) for r in all_results)
    total_informational = sum(r['summary'].get('informational', 0) for r in all_results)
    
    logger.info(f"Total benchmarks tested: {total_benchmarks}")
    logger.info(f"Succeeded: {total_successful}")
    logger.info(
        f"Failed: {total_failed} (errors={total_failed_errors}, verification={total_failed_verification}, "
        f"regressions={total_failed_regressions}, no_speedup={total_failed_no_speedup}, "
        f"generic={total_failed_generic}, other={total_failed_other})"
    )
    logger.info(f"Informational (not benchmarked): {total_informational}")
    emit_event(
        event_logger,
        logger,
        "run_end",
        total_benchmarks=total_benchmarks,
        total_successful=total_successful,
        total_failed=total_failed,
        total_failed_errors=total_failed_errors,
        total_failed_verification=total_failed_verification,
        total_failed_regressions=total_failed_regressions,
        total_failed_no_speedup=total_failed_no_speedup,
        total_failed_generic=total_failed_generic,
        total_failed_other=total_failed_other,
        total_skipped_hardware=total_skipped_hw,
        total_informational=total_informational,
        output_json=str(output_json),
        output_markdown=str(output_md) if args.format in ["markdown", "both"] else None,
    )
    event_logger.close()
    if total_skipped_hw > 0:
        logger.warning(f"WARNING: Skipped (hardware/software limitations): {total_skipped_hw}")
    
    if total_benchmarks > 0:
        success_rate = (total_successful / total_benchmarks) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if total_skipped_hw > 0:
        logger.warning(f"\nWARNING: HARDWARE/SOFTWARE LIMITATIONS DETECTED:")
        logger.warning(f"   {total_skipped_hw} benchmarks skipped due to known limitations")
        logger.warning(f"   (e.g., Triton SM 12.1 support, device-side assert cascades)")
        logger.warning(f"   See detailed report for specific skip reasons")
    
    # Calculate overall speedup statistics
    all_speedups = []
    for r in all_results:
        if r['status'] == 'completed':
            for bench in r['benchmarks']:
                if bench['status'] == 'succeeded' and bench['best_speedup'] > 1.0:
                    all_speedups.append(bench['best_speedup'])
    
    if all_speedups:
        logger.info(f"\nSpeedup Statistics:")
        logger.info(f"  Average: {sum(all_speedups)/len(all_speedups):.2f}x")
        logger.info(f"  Best: {max(all_speedups):.2f}x")
        logger.info(f"  Worst: {min(all_speedups):.2f}x")

    _reap_run_descendants("run_end")
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
