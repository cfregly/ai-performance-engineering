"""Benchmark commands mounted by aisp (`aisp bench ...`)."""

from __future__ import annotations

import json
import threading
import shlex
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import typer
    from typer import Option

    TYPER_AVAILABLE = True
except ImportError:  # pragma: no cover - Typer is optional for docs builds
    TYPER_AVAILABLE = False
    typer = None  # type: ignore
    Option = None  # type: ignore
    Argument = None  # type: ignore
    Context = None  # type: ignore

# Ensure nvcc emits line tables for all benchmark builds
import os  # noqa: E402

os.environ["NVCCFLAGS"] = f"-lineinfo {os.environ.get('NVCCFLAGS', '')}".strip()

from core.analysis.optimization_opportunities import (
    analyze_opportunity_file,
    apply_novelty_validation_feedback,
    apply_run_queue_feedback,
    discover_benchmark_target_catalog,
    normalize_candidates,
    rank_opportunities,
    render_novelty_next_wave_shell,
    render_novelty_validation_shell,
    render_opportunities_markdown,
    render_run_queue_shell,
    summarize_run_queue_root,
)
from core.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    load_benchmark_data as load_benchmark_results,
)
from core.plugins.loader import load_plugin_apps
from core.utils.warning_filters import (
    suppress_known_cuda_capability_warnings,
    warn_optional_component_unavailable,
)

# Load plugins (if installed) to allow capability registration
try:
    load_plugin_apps()
except Exception as exc:
    warn_optional_component_unavailable(
        "plugin app loader",
        exc,
        impact="plugin-provided benchmark capabilities will not be registered",
        context="benchmark CLI startup",
    )

# LLM features are always available - everything is in this single package
LLM_CAPABLE = True


def _expand_multi_value_option(option_names: List[str]) -> None:
    """Allow passing `--option value1 value2` by rewriting argv."""
    argv = sys.argv
    if not set(option_names).intersection(argv):
        return
    new_argv = [argv[0]]
    i = 1
    option_set = set(option_names)
    while i < len(argv):
        token = argv[i]
        if token in option_set:
            option = token
            i += 1
            consumed = False
            while i < len(argv) and not argv[i].startswith("-"):
                new_argv.append(option)
                new_argv.append(argv[i])
                i += 1
                consumed = True
            if not consumed:
                new_argv.append(option)
            continue
        new_argv.append(token)
        i += 1
    sys.argv = new_argv


_expand_multi_value_option(["--targets", "-t", "--hosts", "--labels"])

from core.env import apply_env_defaults, dump_environment_and_capabilities
from core.utils.logger import setup_logging, get_logger
from core.benchmark.artifact_manager import (
    ArtifactManager,
    default_artifacts_root,
    build_bench_run_label,
    build_run_id,
)
from dataclasses import replace

from core.benchmark.defaults import BenchmarkDefaults, get_defaults, set_defaults
from core.benchmark.run_manifest import get_gpu_state
from core.harness.validity_profile import (
    VALIDITY_PROFILE_CHOICES,
    VALIDITY_PROFILE_HELP_TEXT,
    PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
    normalize_validity_profile,
)
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.profiling import profiler_config as profiler_config_mod
from core.discovery import (
    chapter_slug,
    discover_all_chapters,
    resolve_target_chapters,
    discover_benchmarks,
)

apply_env_defaults()


def _select_single_gpu_visible(existing: Optional[str]) -> str:
    if existing:
        tokens = [tok.strip() for tok in existing.split(",") if tok.strip()]
        if tokens:
            return tokens[0]
    return "0"


def _get_analyzer(data_file: Optional[Path] = None) -> PerformanceAnalyzer:
    """Shared analysis helper for CLI commands."""
    loader = (lambda: load_benchmark_results(data_file)) if data_file else load_benchmark_results
    return PerformanceAnalyzer(loader)


def _discover_target_catalog(bench_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    return list(discover_benchmark_target_catalog(bench_root).get("targets", []))


def _validate_output_format(fmt: str | None) -> str:
    if fmt is None:
        return "both"
    normalized = fmt.strip().lower()
    valid = {"json", "markdown", "both"}
    if normalized not in valid:
        message = "Output format must be one of json, markdown, or both"
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _tier1_result_failure_count(result: Dict[str, Any]) -> int:
    summary_payload = result.get("summary")
    if isinstance(summary_payload, dict):
        summary_counts = summary_payload.get("summary")
        if isinstance(summary_counts, dict):
            try:
                return int(summary_counts.get("failed", 0) or 0)
            except (TypeError, ValueError):
                pass
    execution = result.get("execution")
    if isinstance(execution, dict):
        try:
            return int(execution.get("total_failed", 0) or 0)
        except (TypeError, ValueError):
            pass
    try:
        return int(result.get("total_failed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _validate_ncu_metric_set(metric_set: str) -> str:
    normalized = metric_set.strip().lower()
    valid = {"auto", "deep_dive", "roofline", "minimal"}
    if normalized not in valid:
        message = (
            f"Invalid Nsight Compute metric set '{metric_set}'. "
            "Choose from 'auto', 'deep_dive', 'roofline', or 'minimal'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    if normalized != "auto":
        profiler_config_mod.set_default_profiler_metric_set(normalized)
    return normalized


def _validate_ncu_replay_mode(mode: str | None) -> Optional[str]:
    if mode is None:
        return None
    normalized = mode.strip().lower()
    valid = {"kernel", "application"}
    if normalized not in valid:
        message = (
            f"Invalid Nsight Compute replay mode '{mode}'. Choose from 'kernel' or 'application'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _validate_profile_type(profile: str | None) -> str:
    if profile is None:
        return "none"
    normalized = profile.strip().lower()
    valid = {"none", "minimal", "deep_dive", "roofline"}
    if normalized not in valid:
        message = (
            f"Invalid profile choice '{profile}'. "
            "Choose from 'none', 'minimal', 'deep_dive', or 'roofline'."
        )
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _validate_validity_profile(profile: str | None) -> str:
    if profile is None:
        return VALIDITY_PROFILE_CHOICES[0]
    try:
        return normalize_validity_profile(profile, field_name="--validity-profile")
    except ValueError as exc:
        message = str(exc)
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)


def _validate_deep_dive_mode(mode: str | None) -> str:
    if mode is None:
        return "auto"
    normalized = mode.strip().lower()
    valid = {"auto", "always", "never"}
    if normalized not in valid:
        message = f"Invalid deep-dive mode '{mode}'. Choose from 'auto', 'always', or 'never'."
        if TYPER_AVAILABLE and typer is not None:
            raise typer.BadParameter(message)
        raise ValueError(message)
    return normalized


def _parse_target_extra_args(entries: Optional[List[str]]) -> Dict[str, List[str]]:
    """Parse --target-extra-arg entries of the form target="--flag value"."""
    parsed: Dict[str, List[str]] = {}
    for entry in entries or []:
        target, sep, args = entry.partition("=")
        if not sep or not target or not args:
            continue
        parsed[target.strip()] = shlex.split(args)
    return parsed


def _apply_suite_timeout(seconds: Optional[int]):
    """Install a scoped SIGALRM timeout and return a restore callback."""
    sigalrm = getattr(signal, "SIGALRM", None)
    if sigalrm is None:
        return lambda: None

    previous_handler = signal.getsignal(sigalrm)
    previous_delay = 0.0
    previous_interval = 0.0
    use_itimer = all(hasattr(signal, attr) for attr in ("getitimer", "setitimer", "ITIMER_REAL"))
    if use_itimer:
        try:
            previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
        except Exception:
            use_itimer = False
    elif hasattr(signal, "alarm"):
        previous_delay = float(signal.alarm(0))

    started_at = time.monotonic()

    def _clear_alarm() -> None:
        try:
            if use_itimer:
                signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
            elif hasattr(signal, "alarm"):
                signal.alarm(0)
        except Exception:
            pass

    def _restore() -> None:
        elapsed = max(0.0, time.monotonic() - started_at)
        _clear_alarm()
        try:
            signal.signal(sigalrm, previous_handler)
        except Exception:
            return
        remaining_delay = max(0.0, previous_delay - elapsed)
        try:
            if use_itimer:
                if previous_delay > 0.0 or previous_interval > 0.0:
                    signal.setitimer(signal.ITIMER_REAL, remaining_delay, previous_interval)
            elif hasattr(signal, "alarm") and previous_delay > 0.0:
                signal.alarm(max(0, int(remaining_delay)))
        except Exception:
            pass

    _clear_alarm()
    if seconds is None or seconds <= 0:
        return _restore

    def _on_timeout(signum, frame):
        raise TimeoutError(f"Benchmark suite exceeded timeout of {seconds} seconds")

    signal.signal(sigalrm, _on_timeout)
    if use_itimer:
        signal.setitimer(signal.ITIMER_REAL, float(seconds), 0.0)
    else:
        signal.alarm(int(seconds))
    return _restore


def _read_progress_payload_for_heartbeat(
    progress_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort progress read for heartbeat telemetry without hiding corruption."""
    try:
        if not progress_path.exists():
            return None, None
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to read progress payload from {progress_path}: {exc}"
    if not isinstance(payload, dict):
        return None, (
            f"Failed to read progress payload from {progress_path}: "
            f"expected JSON object, got {type(payload).__name__}"
        )
    return payload, None


def _load_expectations_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to load expectations file {path}: {exc}") from exc


def _expectation_example_key(example_name: str, bench_type: str) -> str:
    bench_type_norm = (bench_type or "python").strip().lower()
    if bench_type_norm == "python":
        return example_name
    suffix = f"_{bench_type_norm}"
    if example_name.lower().endswith(suffix):
        return example_name
    return f"{example_name}_{bench_type_norm}"


def _file_uses_torchrun(path: Path) -> bool:
    content = path.read_text(encoding="utf-8")
    markers = ("TorchrunLaunchSpec", "get_torchrun_spec", "LaunchVia.TORCHRUN")
    return any(marker in content for marker in markers)


def _collect_multi_gpu_examples(chapter_dir: Path) -> Dict[str, bool]:
    multi_gpu: Dict[str, bool] = {}
    pairs = discover_benchmarks(chapter_dir, validate=False, warn_missing=False)

    try:
        from core.harness.run_benchmarks import cuda_binary_requires_multi_gpu
        from core.discovery import is_cuda_binary_benchmark_file
    except Exception as exc:
        raise RuntimeError(f"Failed to import benchmark helpers: {exc}") from exc

    for baseline_path, optimized_paths, example_name in pairs:
        is_cuda_wrapper = is_cuda_binary_benchmark_file(baseline_path)
        if is_cuda_wrapper:
            uses_multi_gpu = cuda_binary_requires_multi_gpu(baseline_path)
            for opt_path in optimized_paths:
                uses_multi_gpu = uses_multi_gpu or cuda_binary_requires_multi_gpu(opt_path)
            example_key = _expectation_example_key(example_name, "cuda")
        else:
            try:
                uses_torchrun = _file_uses_torchrun(baseline_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to read {baseline_path}: {exc}") from exc
            for opt_path in optimized_paths:
                try:
                    uses_torchrun = uses_torchrun or _file_uses_torchrun(opt_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to read {opt_path}: {exc}") from exc
            uses_multi_gpu = uses_torchrun
            example_key = _expectation_example_key(example_name, "python")

        if example_key in multi_gpu:
            multi_gpu[example_key] = multi_gpu[example_key] or uses_multi_gpu
        else:
            multi_gpu[example_key] = uses_multi_gpu
    return multi_gpu


def _collect_expectations(
    hardware_key: str,
    repo_root: Path,
) -> List[Tuple[Path, Dict[str, Any], Dict[str, bool]]]:
    files: List[Tuple[Path, Dict[str, Any], Dict[str, bool]]] = []
    for chapter_dir in discover_all_chapters(repo_root):
        exp_path = chapter_dir / f"expectations_{hardware_key}.json"
        if not exp_path.exists():
            continue
        data = _load_expectations_file(exp_path)
        multi_gpu = _collect_multi_gpu_examples(chapter_dir)
        files.append((exp_path, data, multi_gpu))
    return files


# Import architecture optimizations early
try:
    import arch_config  # noqa: F401
except ImportError as exc:
    warn_optional_component_unavailable(
        "arch_config",
        exc,
        impact="architecture-specific defaults were not applied",
        context="benchmark CLI startup",
    )

# Import benchmark functionality
BENCHMARK_IMPORT_ERROR: Optional[str] = None
try:
    with suppress_known_cuda_capability_warnings(context="benchmark CLI torch import"):
        import torch  # noqa: F401
        from core.utils.chapter_compare_template import discover_benchmarks
        from core.harness.run_benchmarks import (
            test_chapter,
            generate_markdown_report,
            BenchmarkEventLogger,
            emit_event,
            _preflight_target_coverage_and_assets,
        )

    BENCHMARK_AVAILABLE = True
except ImportError as exc:
    BENCHMARK_AVAILABLE = False
    BENCHMARK_IMPORT_ERROR = str(exc)
    warn_optional_component_unavailable(
        "benchmark runtime imports",
        exc,
        impact="benchmark execution commands will not be available until imports succeed",
        context="benchmark CLI startup",
    )

    # Test functions come from run_benchmarks; if that import failed above, this is False.
TEST_FUNCTIONS_AVAILABLE = BENCHMARK_AVAILABLE

# Typer app setup
if TYPER_AVAILABLE:
    app = typer.Typer(
        name="benchmark",
        help="Unified benchmark execution and management CLI",
        add_completion=False,
    )
else:
    app = None


def _execute_benchmarks(
    targets: Optional[List[str]] = None,
    bench_root: Optional[Path] = None,
    output_format: str = "both",
    profile_type: str = "minimal",
    suite_timeout: Optional[int] = 14400,
    timeout_multiplier: float = 3.0,
    validity_profile: str = "strict",
    allow_foreign_gpu_processes: bool = False,
    allow_portable_expectations_update: bool = False,
    reproducible: bool = False,
    cold_start: bool = False,
    force_synchronize: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    force_pipeline: bool = False,
    gpu_sm_clock_mhz: Optional[int] = None,
    gpu_mem_clock_mhz: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    single_gpu: bool = False,
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
    # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
    verify_input: bool = True,
    verify_output: bool = True,
    only_cuda: bool = False,
    only_python: bool = False,
    # LLM options
    llm_analysis: bool = False,
    force_llm: bool = False,
    llm_provider: Optional[str] = None,
    apply_llm_patches: bool = False,
    rebenchmark_llm_patches: bool = False,
    patch_strategy: str = "ast",
    llm_patch_retries: int = 2,
    use_llm_cache: bool = True,
    llm_explain: bool = False,
    exit_on_failure: bool = True,
) -> Dict[str, Any]:
    """Execute selected benchmarks with optional profiling."""
    validity_profile = _validate_validity_profile(validity_profile)
    portable_mode = validity_profile == "portable"
    allow_virtualization = portable_mode
    if portable_mode and not allow_portable_expectations_update:
        if update_expectations or accept_regressions or allow_mixed_provenance:
            requested_writes: List[str] = []
            if update_expectations:
                requested_writes.append("--update-expectations")
            if accept_regressions:
                requested_writes.append("--accept-regressions")
            if allow_mixed_provenance:
                requested_writes.append("--allow-mixed-provenance")
            requested_summary = ", ".join(requested_writes)
            print(
                "Invalid flag combination: "
                f"{requested_summary} requested with --validity-profile portable. "
                "Portable validity profile disables expectation writes by default. "
                "Add --allow-portable-expectations-update to enable writes in portable mode, "
                "or use --validity-profile strict.",
                flush=True,
            )
            sys.exit(1)

    parsed_extra_args = _parse_target_extra_args(target_extra_args)
    repo_root = Path(__file__).resolve().parents[2]
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root

    if single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = _select_single_gpu_visible(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        )

    try:
        from core.harness.cuda_capabilities import set_force_pipeline

        set_force_pipeline(force_pipeline)
    except ImportError as exc:
        if force_pipeline:
            raise RuntimeError(
                "Requested force_pipeline=True but core.harness.cuda_capabilities could not be imported. "
                f"Install or fix the runtime so the override can be applied: {exc}"
            ) from exc
        warnings.warn(
            "core.harness.cuda_capabilities is unavailable; force_pipeline overrides cannot be applied. "
            f"Continuing without override: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    artifact_base = (
        Path(artifacts_dir) if artifacts_dir else default_artifacts_root(active_bench_root)
    )
    run_label = None
    if run_id is None:
        run_label = build_bench_run_label(targets or [], profile_type)
        run_id = build_run_id("bench", run_label, base_dir=artifact_base)
    artifact_manager = ArtifactManager(
        base_dir=artifact_base,
        run_id=run_id,
        run_kind="bench",
        run_label=run_label,
    )
    # Propagate a stable run marker before any helper subprocesses launch so
    # same-run profiler/isolated children can be attributed correctly.
    os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] = artifact_manager.run_id
    os.environ["AISP_BENCHMARK_OWNER_PID"] = str(os.getpid())
    if log_file is None:
        log_file = artifact_manager.get_log_path()
    output_json = artifact_manager.get_result_path("benchmark_test_results.json")
    output_md = artifact_manager.get_report_path("benchmark_test_results.md")

    setup_logging(level=log_level, log_file=log_file, log_format="json", use_rich=True)
    logger = get_logger(__name__)
    if validity_profile == "strict":
        logger.info(
            "Benchmark validity profile (--validity-profile): strict "
            "(fail-fast, full validity enforcement)."
        )
    else:
        logger.info(
            "Benchmark validity profile (--validity-profile): portable "
            "(compatibility mode for virtualized/limited hosts; some strict checks are relaxed)."
        )
        if not allow_portable_expectations_update:
            logger.info(
                "Portable mode default: expectation writes are disabled unless "
                "--allow-portable-expectations-update is provided."
            )
    event_logger = BenchmarkEventLogger(
        artifact_manager.get_log_path("benchmark_events.jsonl"),
        artifact_manager.run_id,
        logger,
    )

    if not BENCHMARK_AVAILABLE or not TEST_FUNCTIONS_AVAILABLE:
        detail = f" Import error: {BENCHMARK_IMPORT_ERROR}" if BENCHMARK_IMPORT_ERROR else ""
        logger.error(
            "Benchmark dependencies missing (torch/benchmark_harness or test functions).%s",
            detail,
        )
        if exit_on_failure:
            sys.exit(1)
        event_logger.close()
        return {
            "run_id": artifact_manager.run_id,
            "artifact_root": str(artifact_manager.run_dir),
            "output_json": str(output_json),
            "output_markdown": (str(output_md) if output_format in ["markdown", "both"] else None),
            "manifest_path": None,
            "bench_root": str(active_bench_root),
            "total_failed": 0,
            "total_successful": 0,
            "total_skipped": 0,
            "results": [],
            "error": (
                "Benchmark dependencies missing"
                if not BENCHMARK_IMPORT_ERROR
                else f"Benchmark dependencies missing: {BENCHMARK_IMPORT_ERROR}"
            ),
        }

    try:
        dump_environment_and_capabilities()
    except Exception as exc:
        logger.error(f"Skipping environment/capabilities dump due to error: {exc}")

    if only_cuda and only_python:
        logger.error("Cannot combine --only-cuda and --only-python.")
        sys.exit(1)

    try:
        chapter_dirs, chapter_filters = resolve_target_chapters(
            targets, bench_root=active_bench_root
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        sys.exit(1)
    explicit_targets = bool(targets) and not all(
        str(target).strip().lower() == "all" for target in targets
    )
    preflight_issues = _preflight_target_coverage_and_assets(
        chapter_dirs,
        chapter_filters,
        only_cuda=bool(only_cuda),
        only_python=bool(only_python),
        target_extra_args=parsed_extra_args,
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
        event_logger.close()
        sys.exit(1)

    def _classify_target_dir(path: Path) -> str:
        try:
            rel = path.resolve().relative_to(repo_root.resolve())
        except Exception:
            rel = Path(path.name)
        if rel.parts and rel.parts[0].startswith("ch") and rel.parts[0][2:].isdigit():
            return "chapter"
        if rel.parts and rel.parts[0] == "labs":
            return "lab"
        return "other"

    chapter_count = sum(1 for p in chapter_dirs if _classify_target_dir(p) == "chapter")
    lab_count = sum(1 for p in chapter_dirs if _classify_target_dir(p) == "lab")
    other_count = len(chapter_dirs) - chapter_count - lab_count
    if other_count:
        logger.info(
            f"FOUND {chapter_count} chapter(s), {lab_count} lab(s), {other_count} other target(s)"
        )
    else:
        logger.info(f"FOUND {chapter_count} chapter(s), {lab_count} lab(s)")

    enable_profiling = (profile_type or "none").lower() != "none"

    restore_suite_timeout = _apply_suite_timeout(suite_timeout)
    try:
        progress_recorder = ProgressRecorder(
            run_id=artifact_manager.run_id,
            progress_path=artifact_manager.progress_dir / "run_progress.json",
        )
        if progress_recorder.load_warning:
            logger.warning(progress_recorder.load_warning)
            emit_event(
                event_logger,
                logger,
                "progress_history_unavailable",
                warning=progress_recorder.load_warning,
                progress_path=str(progress_recorder.progress_path),
            )
        progress_recorder.emit(
            ProgressEvent(
                phase="run",
                phase_index=1,
                total_phases=2,
                step="start",
                step_detail=f"targets={targets or ['all']}",
                percent_complete=0.0,
            )
        )
        defaults = get_defaults() or BenchmarkDefaults()
        # Optional per-run GPU clock overrides. These become defaults for all BenchmarkConfig
        # instances created during this run (including in subprocess isolation).
        clock_overrides = {}
        if gpu_sm_clock_mhz is not None:
            clock_overrides["gpu_sm_clock_mhz"] = int(gpu_sm_clock_mhz)
        if gpu_mem_clock_mhz is not None:
            clock_overrides["gpu_mem_clock_mhz"] = int(gpu_mem_clock_mhz)
        if clock_overrides:
            defaults = replace(defaults, **clock_overrides)
            set_defaults(defaults)
        gpu_state = get_gpu_state(allow_telemetry_failures=portable_mode)
        emit_event(
            event_logger,
            logger,
            "run_start",
            targets=targets or ["all"],
            profile_type=profile_type,
            ncu_metric_set=ncu_metric_set,
            ncu_replay_mode=ncu_replay_mode,
            timeout_multiplier=timeout_multiplier,
            nsys_timeout_seconds=nsys_timeout_seconds,
            ncu_timeout_seconds=ncu_timeout_seconds,
            validity_profile=validity_profile,
            allow_virtualization=allow_virtualization,
            allow_foreign_gpu_processes=allow_foreign_gpu_processes,
            update_expectations=update_expectations,
            allow_portable_expectations_update=allow_portable_expectations_update,
            allow_mixed_provenance=allow_mixed_provenance,
            artifacts_dir=str(artifact_manager.run_dir),
            log_file=str(log_file),
            events_file=str(artifact_manager.get_log_path("benchmark_events.jsonl")),
            gpu_clock_lock_enabled=defaults.lock_gpu_clocks,
            gpu_sm_clock_mhz=defaults.gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=defaults.gpu_mem_clock_mhz,
            gpu_app_clock_mhz=gpu_state.get("gpu_app_clock_mhz"),
            memory_app_clock_mhz=gpu_state.get("memory_app_clock_mhz"),
            gpu_clock_mhz=gpu_state.get("gpu_clock_mhz"),
            memory_clock_mhz=gpu_state.get("memory_clock_mhz"),
        )
        heartbeat_stop = threading.Event()
        stale_progress_seconds = max(60, int(os.environ.get("AISP_STALE_PROGRESS_SECONDS", "900")))
        stale_phases = {
            "baseline_nsys",
            "optimized_nsys",
            "baseline_ncu",
            "optimized_ncu",
            "baseline_torch",
            "optimized_torch",
        }
        # Long high-iteration timing phases can legitimately run for extended periods
        # without progress timestamp updates. Keep timing stale-abort as explicit opt-in.
        if os.environ.get("AISP_STALE_WATCH_TIMING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            stale_phases.update({"baseline_timing", "optimized_timing"})

        def _snapshot_process_table() -> Dict[int, Tuple[int, str]]:
            try:
                proc = subprocess.run(
                    ["ps", "-eo", "pid=,ppid=,cmd="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            except Exception:
                return {}
            rows: Dict[int, Tuple[int, str]] = {}
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 2)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except ValueError:
                    continue
                cmd = parts[2] if len(parts) > 2 else ""
                rows[pid] = (ppid, cmd)
            return rows

        def _descendants_of(
            root_pid: int, table: Dict[int, Tuple[int, str]]
        ) -> List[Tuple[int, str]]:
            children: Dict[int, List[int]] = {}
            for pid, (ppid, _cmd) in table.items():
                children.setdefault(ppid, []).append(pid)
            out: List[Tuple[int, str]] = []
            queue: List[int] = [root_pid]
            seen: set[int] = set()
            while queue:
                current = queue.pop(0)
                for child in children.get(current, []):
                    if child in seen:
                        continue
                    seen.add(child)
                    cmd = table.get(child, (0, ""))[1]
                    out.append((child, cmd))
                    queue.append(child)
            return out

        def _is_profiler_process(cmd: str) -> bool:
            lower = cmd.lower()
            return any(
                token in lower
                for token in (
                    " ncu ",
                    "nsight-compute",
                    "/ncu",
                    " nsys ",
                    "nsight-systems",
                    "qdstrmimporter",
                    "nsys-tee",
                )
            )

        def _is_build_process(cmd: str) -> bool:
            lower = cmd.lower()
            return any(
                token in lower
                for token in (
                    " nvcc ",
                    "/nvcc",
                    " ptxas ",
                    "/ptxas",
                    " ninja ",
                    "/ninja",
                    " gcc ",
                    " g++ ",
                    " cc1plus ",
                    " cicc ",
                    " fatbinary ",
                    " ld ",
                    " cmake ",
                )
            )

        def _terminate_stalled_children() -> Tuple[List[int], List[int], List[int]]:
            table = _snapshot_process_table()
            descendants = _descendants_of(os.getpid(), table)
            profiler_pids = [pid for pid, cmd in descendants if _is_profiler_process(cmd)]
            build_pids = [pid for pid, cmd in descendants if _is_build_process(cmd)]
            # Do not kill while profiling/build toolchains are actively running.
            # These phases can legitimately run for extended periods without
            # benchmark progress updates.
            if profiler_pids or build_pids:
                return [], profiler_pids, build_pids
            kill_targets: List[int] = []
            for pid, cmd in descendants:
                lower = cmd.lower()
                if "isolated_runner.py" in lower or "core.harness.isolated_runner" in lower:
                    kill_targets.append(pid)
            killed: List[int] = []
            for pid in kill_targets:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                    killed.append(pid)
                except Exception:
                    continue
            if killed:
                time.sleep(1.0)
                for pid in killed:
                    if not Path(f"/proc/{pid}").exists():
                        continue
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        continue
            return killed, profiler_pids, build_pids

        stale_state: Dict[str, Any] = {
            "fingerprint": None,
            "since": time.time(),
            "last_logged_sec": 0.0,
            "abort_triggered": False,
        }
        progress_read_state: Dict[str, Any] = {
            "warning": None,
            "last_logged_sec": 0.0,
        }

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.is_set():
                payload, progress_warning = _read_progress_payload_for_heartbeat(
                    progress_recorder.progress_path
                )
                now = time.time()
                if progress_warning:
                    if (
                        progress_read_state["warning"] != progress_warning
                        or now - float(progress_read_state["last_logged_sec"]) >= 30.0
                    ):
                        progress_read_state["warning"] = progress_warning
                        progress_read_state["last_logged_sec"] = now
                        emit_event(
                            event_logger,
                            logger,
                            "run_progress_unreadable",
                            warning=progress_warning,
                            progress_path=str(progress_recorder.progress_path),
                            run_id=artifact_manager.run_id,
                        )
                        logger.warning(progress_warning)
                else:
                    progress_read_state["warning"] = None
                current = payload.get("current") if isinstance(payload, dict) else None
                if isinstance(current, dict):
                    phase = str(current.get("phase") or "")
                    fingerprint = json.dumps(
                        {
                            "phase": phase,
                            "step": current.get("step"),
                            "step_detail": current.get("step_detail"),
                            "timestamp": current.get("timestamp"),
                        },
                        sort_keys=True,
                        default=str,
                    )
                    if stale_state["fingerprint"] != fingerprint:
                        stale_state["fingerprint"] = fingerprint
                        stale_state["since"] = now
                        stale_state["abort_triggered"] = False
                    stagnant_for = now - float(stale_state["since"])
                    if phase in stale_phases and stagnant_for >= stale_progress_seconds:
                        if now - float(stale_state["last_logged_sec"]) >= 30.0:
                            stale_state["last_logged_sec"] = now
                            emit_event(
                                event_logger,
                                logger,
                                "run_stale_progress",
                                phase=phase,
                                stagnant_seconds=stagnant_for,
                                stale_threshold_seconds=stale_progress_seconds,
                                step=current.get("step"),
                                step_detail=current.get("step_detail"),
                            )
                        if not stale_state["abort_triggered"]:
                            killed_pids, profiler_pids, build_pids = _terminate_stalled_children()
                            if profiler_pids or build_pids:
                                stale_state["since"] = now
                                emit_event(
                                    event_logger,
                                    logger,
                                    "run_stale_deferred",
                                    phase=phase,
                                    stagnant_seconds=stagnant_for,
                                    stale_threshold_seconds=stale_progress_seconds,
                                    profiler_pids=profiler_pids,
                                    build_pids=build_pids,
                                    run_id=artifact_manager.run_id,
                                )
                                logger.error(
                                    "Stale progress detected for %.0fs in phase=%s, but active worker pids were found; defer auto-abort (profiler=%s, build=%s).",
                                    stagnant_for,
                                    phase,
                                    profiler_pids,
                                    build_pids,
                                )
                                continue
                            stale_state["abort_triggered"] = True
                            emit_event(
                                event_logger,
                                logger,
                                "run_stale_abort",
                                phase=phase,
                                stagnant_seconds=stagnant_for,
                                stale_threshold_seconds=stale_progress_seconds,
                                killed_pids=killed_pids,
                                profiler_pids=profiler_pids,
                                build_pids=build_pids,
                                run_id=artifact_manager.run_id,
                            )
                            if killed_pids:
                                logger.error(
                                    "Stale progress detected for %.0fs in phase=%s; terminated child benchmark pids=%s.",
                                    stagnant_for,
                                    phase,
                                    killed_pids,
                                )
                            else:
                                logger.error(
                                    "Stale progress detected for %.0fs in phase=%s with no killable child; terminating run process.",
                                    stagnant_for,
                                    phase,
                                )
                                try:
                                    os.kill(os.getpid(), signal.SIGTERM)
                                except Exception:
                                    os._exit(143)
                emit_event(
                    event_logger,
                    logger,
                    "run_heartbeat",
                    progress=current,
                    progress_path=str(progress_recorder.progress_path),
                    progress_warning=progress_warning,
                )
                heartbeat_stop.wait(60.0)

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        all_results = []
        output_json = artifact_manager.get_result_path("benchmark_test_results.json")
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
            example_filters = chapter_filters.get(chapter_id)
            only_examples = sorted(example_filters) if example_filters else None
            result = test_chapter(
                chapter_dir=chapter_dir,
                enable_profiling=enable_profiling,
                profile_type=profile_type if enable_profiling else "none",
                profile_output_root=artifact_manager.profiles_dir,
                timeout_multiplier=timeout_multiplier,
                reproducible=reproducible,
                cold_start=cold_start,
                force_synchronize=force_synchronize,
                iterations=iterations,
                warmup=warmup,
                single_gpu=single_gpu,
                enforce_environment_validation=True,
                allow_virtualization=allow_virtualization,
                allow_foreign_gpu_processes=allow_foreign_gpu_processes,
                validity_profile=validity_profile,
                only_examples=only_examples,
                accept_regressions=accept_regressions,
                update_expectations=update_expectations,
                allow_portable_expectations_update=allow_portable_expectations_update,
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
                env_passthrough=torchrun_env,
                target_extra_args=parsed_extra_args,
                subprocess_stderr_dir=artifact_manager.logs_dir,
                only_cuda=only_cuda,
                only_python=only_python,
                progress_recorder=progress_recorder,
                # Verification - both enabled by default for valid benchmark comparisons
                verify_input=verify_input,
                verify_output=verify_output,
                # LLM options
                llm_analysis=llm_analysis or force_llm,
                force_llm=force_llm,
                llm_provider=llm_provider,
                apply_llm_patches=apply_llm_patches,
                rebenchmark_llm_patches=rebenchmark_llm_patches,
                patch_strategy=patch_strategy,
                llm_patch_retries=llm_patch_retries,
                use_llm_cache=use_llm_cache,
                llm_explain=llm_explain,
                fail_on_no_benchmarks=explicit_targets,
                event_logger=event_logger,
            )
            all_results.append(result)
            if output_format in ["json", "both"]:
                with open(output_json, "w") as f:
                    json.dump(
                        {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": all_results},
                        f,
                        indent=2,
                    )
                logger.info(f"JSON results checkpoint saved to: {output_json}")

        progress_recorder.emit(
            ProgressEvent(
                phase="run",
                phase_index=2,
                total_phases=2,
                step="complete",
                percent_complete=100.0,
            )
        )

        manifests = []
        for result in all_results:
            manifests.extend(result.get("manifests", []))

        if manifests:
            with open(artifact_manager.manifest_path, "w") as f:
                json.dump(
                    {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "manifests": manifests},
                    f,
                    indent=2,
                )
            logger.info(f"Manifest saved to: {artifact_manager.manifest_path}")

        if output_format in ["json", "both"]:
            with open(output_json, "w") as f:
                json.dump(
                    {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": all_results},
                    f,
                    indent=2,
                )
            logger.info(f"JSON results saved to: {output_json}")
        if output_format in ["markdown", "both"]:
            generate_markdown_report(all_results, output_md, bench_root=active_bench_root)
            logger.info(f"Markdown report saved to: {output_md}")

        def _failed_count(summary: Dict[str, Any]) -> int:
            summary_failed = int(summary.get("failed", 0) or 0)
            derived_failed = sum(
                int(summary.get(key, 0) or 0)
                for key in (
                    "failed_error",
                    "failed_verification",
                    "failed_regression",
                    "failed_no_speedup",
                    "failed_generic",
                    "failed_other",
                )
            )
            return max(summary_failed, derived_failed)

        total_failed = sum(_failed_count(r.get("summary", {})) for r in all_results)
        total_successful = sum(r.get("summary", {}).get("successful", 0) for r in all_results)
        total_skipped = sum(r.get("summary", {}).get("total_skipped", 0) for r in all_results)
        emit_event(
            event_logger,
            logger,
            "run_end",
            total_chapters=len(all_results),
            total_successful=total_successful,
            total_failed=total_failed,
            total_skipped=total_skipped,
            output_json=str(output_json),
            output_markdown=str(output_md) if output_format in ["markdown", "both"] else None,
        )
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5.0)
        event_logger.close()
        if total_failed > 0 and exit_on_failure:
            sys.exit(1)
        return {
            "run_id": artifact_manager.run_id,
            "artifact_root": str(artifact_manager.run_dir),
            "output_json": str(output_json),
            "output_markdown": (str(output_md) if output_format in ["markdown", "both"] else None),
            "manifest_path": (str(artifact_manager.manifest_path) if manifests else None),
            "bench_root": str(active_bench_root),
            "total_failed": total_failed,
            "total_successful": total_successful,
            "total_skipped": total_skipped,
            "results": all_results,
        }
    finally:
        restore_suite_timeout()


if TYPER_AVAILABLE:

    @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    def run(
        ctx: typer.Context,
        targets: Optional[List[str]] = Option(
            None,
            "--targets",
            "-t",
            help="Chapter(s) or chapter:example pairs to run. Repeat the flag for multiple targets. Omit or use 'all' for every chapter.",
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
        output_format: str = Option(
            "both",
            "--format",
            "-f",
            help="Output format: 'json', 'markdown', or 'both'",
            callback=_validate_output_format,
        ),
        profile_type: str = Option(
            "minimal",
            "--profile",
            "-p",
            help="Profiling preset: minimal (default), none, deep_dive, or roofline. Non-'none' enables nsys/ncu/PyTorch profiling.",
            callback=_validate_profile_type,
        ),
        suite_timeout: Optional[int] = Option(
            14400,
            "--suite-timeout",
            help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)",
        ),
        timeout_seconds: Optional[int] = Option(
            None, "--timeout-seconds", help="Override suite timeout in seconds (0 disables timeout)"
        ),
        timeout_multiplier: float = Option(
            3.0,
            "--timeout-multiplier",
            help="Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts)",
        ),
        validity_profile: str = Option(
            "strict",
            "--validity-profile",
            help=VALIDITY_PROFILE_HELP_TEXT,
            callback=_validate_validity_profile,
        ),
        allow_foreign_gpu_processes: bool = Option(
            False,
            "--allow-foreign-gpu-processes",
            help="Warn instead of fail when unrelated CUDA compute processes are active on the benchmark GPU. Use only on shared hosts when strict isolation is impossible.",
        ),
        reproducible: bool = Option(
            False,
            "--reproducible",
            help="Enable reproducible mode: set all seeds to 42 and force deterministic algorithms (uses slower fallbacks; ops without deterministic support may error).",
        ),
        cold_start: bool = Option(
            False,
            "--cold-start",
            help="Reset GPU state between benchmarks for cold start measurements",
        ),
        force_sync: bool = Option(
            False,
            "--force-sync",
            help="Force a device-wide synchronize immediately after benchmark_fn() (opt-in safeguard).",
        ),
        iterations: Optional[int] = Option(
            None, "--iterations", help="Number of benchmark iterations (default: chapter-specific)"
        ),
        warmup: Optional[int] = Option(
            None, "--warmup", help="Number of warmup iterations (default: chapter-specific)"
        ),
        force_pipeline: bool = Option(
            False,
            "--force-pipeline",
            help="Force enable CUDA Pipeline API even on compute capability 12.0+ (may cause instability on Blackwell GPUs)",
        ),
        gpu_sm_clock_mhz: Optional[int] = Option(
            None,
            "--gpu-sm-clock-mhz",
            help=(
                "Lock the SM application clock (MHz) for this run (requires clock locking to be enabled). "
                "Recommended for SOL comparisons, e.g. --gpu-sm-clock-mhz 1500 on B200/B200."
            ),
        ),
        gpu_mem_clock_mhz: Optional[int] = Option(
            None,
            "--gpu-mem-clock-mhz",
            help=(
                "Lock the GPU memory (HBM) application clock (MHz) for this run (requires clock locking to be enabled). "
                "If omitted, the harness will lock memory to the max supported clock."
            ),
        ),
        artifacts_dir: Optional[str] = Option(
            None,
            "--artifacts-dir",
            help="Base directory for run artifacts (default: ./artifacts/runs).",
        ),
        run_id: Optional[str] = Option(
            None,
            "--run-id",
            help=(
                "Run ID for artifact directory (default: <timestamp>__bench__profile-<type>__targets-<...>)."
            ),
        ),
        log_level: str = Option(
            "INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"
        ),
        log_file: Optional[str] = Option(
            None,
            "--log-file",
            help="Path to log file (default: artifacts/runs/<run_id>/logs/benchmark.log)",
        ),
        single_gpu: bool = Option(
            False,
            "--single-gpu",
            help="Force single-GPU visibility (sets CUDA_VISIBLE_DEVICES=0 for this run).",
        ),
        ncu_metric_set: str = Option(
            "minimal",
            "--ncu-metric-set",
            help="Nsight Compute metric preset: auto, minimal, deep_dive, or roofline. Default is minimal for safer, lower-overhead profiling.",
            callback=_validate_ncu_metric_set,
        ),
        ncu_replay_mode: Optional[str] = Option(
            None,
            "--ncu-replay-mode",
            help="Nsight Compute replay mode: kernel or application. If omitted, benchmark-specific config or harness defaults apply.",
            callback=_validate_ncu_replay_mode,
        ),
        nsys_timeout_seconds: Optional[int] = Option(
            None,
            "--nsys-timeout-seconds",
            help="Override Nsight Systems timeout in seconds (default from BenchmarkDefaults).",
        ),
        ncu_timeout_seconds: Optional[int] = Option(
            None,
            "--ncu-timeout-seconds",
            help="Override Nsight Compute timeout in seconds (default from BenchmarkDefaults).",
        ),
        pm_sampling_interval: Optional[int] = Option(
            None,
            "--pm-sampling-interval",
            help="Nsight Compute pm-sampling-interval (cycles between samples). Optional: set to reduce overhead; defaults to unset.",
        ),
        only_cuda: bool = Option(
            False, "--only-cuda", help="Run only CUDA binary benchmarks (Python wrappers)."
        ),
        only_python: bool = Option(
            False, "--only-python", help="Run only Python benchmarks (skip CUDA binary wrappers)."
        ),
        accept_regressions: bool = Option(
            False,
            "--accept-regressions",
            help="Update expectation files when improvements are detected instead of flagging regressions.",
        ),
        update_expectations: bool = Option(
            False,
            "--update-expectations",
            help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.",
        ),
        allow_portable_expectations_update: bool = Option(
            False,
            "--allow-portable-expectations-update",
            help=PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
        ),
        allow_mixed_provenance: bool = Option(
            False,
            "--allow-mixed-provenance",
            help="Allow expectation updates when provenance differs (commit/hardware/profile mismatch) without forcing updates. Does NOT accept regressions (use --accept-regressions or --update-expectations).",
        ),
        launch_via: str = Option(
            "python", "--launch-via", help="Launcher to use for benchmarks: python or torchrun."
        ),
        nproc_per_node: Optional[int] = Option(
            None, "--nproc-per-node", help="torchrun --nproc_per_node value."
        ),
        nnodes: Optional[str] = Option(None, "--nnodes", help="torchrun --nnodes value."),
        rdzv_backend: Optional[str] = Option(
            None,
            "--rdzv-backend",
            help="torchrun rendezvous backend (defaults to c10d when nnodes is set).",
        ),
        rdzv_endpoint: Optional[str] = Option(
            None, "--rdzv-endpoint", help="torchrun rendezvous endpoint (host:port)."
        ),
        torchrun_env: Optional[List[str]] = Option(
            None,
            "--torchrun-env",
            help="Environment variables to forward into torchrun launches (repeatable).",
        ),
        target_extra_args: Optional[List[str]] = Option(
            None,
            "--target-extra-arg",
            help='Per-target extra args, format: target="--flag value". Repeatable.',
        ),
        # LLM analysis and patching options
        llm_analysis: bool = Option(
            False,
            "--llm-analysis",
            help="Enable LLM-powered analysis for benchmarks with <1.1x speedup. Requires API keys in .env.local",
        ),
        force_llm: bool = Option(
            False,
            "--force-llm",
            help="Force LLM analysis on ALL benchmarks regardless of speedup. Use to try improving even good results.",
        ),
        llm_provider: Optional[str] = Option(
            None,
            "--llm-provider",
            help="LLM provider: 'anthropic' or 'openai'. Defaults to env LLM_PROVIDER.",
        ),
        apply_llm_patches: bool = Option(
            False,
            "--apply-llm-patches",
            help="Apply LLM-suggested patches to create new optimized variants. Requires --llm-analysis.",
        ),
        rebenchmark_llm_patches: bool = Option(
            False,
            "--rebenchmark-llm-patches",
            help="Re-benchmark LLM-patched variants. Requires --apply-llm-patches.",
        ),
        patch_strategy: str = Option(
            "ast",
            "--patch-strategy",
            help="Patch strategy: 'ast' (default, AST-based) or 'fuzzy' (text matching).",
        ),
        llm_patch_retries: int = Option(
            2,
            "--llm-patch-retries",
            help="Max retry attempts when LLM patch fails (syntax/runtime errors). Default: 2",
        ),
        no_llm_cache: bool = Option(
            False,
            "--no-llm-cache",
            help="Disable LLM analysis caching (always re-run LLM even if cached results exist).",
        ),
        llm_explain: bool = Option(
            False,
            "--llm-explain",
            help="Generate educational explanations for best patches (why it works, optimization techniques used). Requires --rebenchmark-llm-patches.",
        ),
        # Verification - BOTH enabled by default; without verification, benchmarks are meaningless
        skip_input_verify: bool = Option(
            False,
            "--skip-input-verify",
            help="Skip input equivalence verification. WARNING: Without this check, benchmark comparisons may be invalid (different workloads).",
        ),
        skip_output_verify: bool = Option(
            False,
            "--skip-output-verify",
            help="Skip output correctness verification. WARNING: Without this check, optimizations may produce incorrect results.",
        ),
        skip_verify: bool = Option(
            False,
            "--skip-verify",
            help="Skip BOTH input and output verification. Equivalent to --skip-input-verify --skip-output-verify.",
        ),
        verify_phase: str = Option(
            "gate",
            "--verify-phase",
            help="Verification enforcement phase: 'detect' (report only), 'quarantine' (exclude non-compliant from reports), 'gate' (default, fail on verification failure)",
        ),
        precheck_only: bool = Option(
            False,
            "--precheck-only",
            help="Validate targets and print planned command without running.",
        ),
        dry_run: bool = Option(
            False, "--dry-run", help="Describe planned execution without running benchmarks."
        ),
    ):
        """Run benchmarks - discover, run, and summarize results.

        CUDA binaries execute only via Python wrappers (CudaBinaryBenchmark).
        """
        # LLM features are always available - no capability check needed
        repo_root = Path(__file__).resolve().parents[2]

        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root

        combined_targets: List[str] = []
        for arg in list(targets) if targets else []:
            if arg:
                combined_targets.append(arg)
        for extra in ctx.args or []:
            if not extra:
                continue
            # Drop stray values that belong to other options when Click defers parsing.
            if extra.lower() in {
                "none",
                "minimal",
                "deep_dive",
                "roofline",
                "json",
                "markdown",
                "both",
            }:
                continue
            combined_targets.append(extra)
        # Final cleanup: drop any falsy or duplicate entries
        combined_targets = [t for t in combined_targets if t]
        # Deduplicate to avoid running the same target multiple times when
        # Click/Typer shuffles positional args.
        combined_targets = list(dict.fromkeys(combined_targets))
        effective_timeout = timeout_seconds if timeout_seconds is not None else suite_timeout
        # Set verification enforcement phase
        if verify_phase:
            os.environ["VERIFY_ENFORCEMENT_PHASE"] = verify_phase.lower()

        if precheck_only or dry_run:
            plan = {
                "precheck_only": precheck_only,
                "dry_run": dry_run,
                "targets": combined_targets or ["all"],
                "bench_root": str(active_bench_root),
                "profile_type": profile_type,
                "output_format": output_format,
                "suite_timeout": effective_timeout,
                "verify_phase": verify_phase,
                "single_gpu": single_gpu,
                "validity_profile": validity_profile,
                "allow_foreign_gpu_processes": allow_foreign_gpu_processes,
                "allow_mixed_provenance": allow_mixed_provenance,
                "allow_portable_expectations_update": allow_portable_expectations_update,
                "force_sync": force_sync,
                "gpu_sm_clock_mhz": gpu_sm_clock_mhz,
                "gpu_mem_clock_mhz": gpu_mem_clock_mhz,
                "nsys_timeout_seconds": nsys_timeout_seconds,
                "ncu_timeout_seconds": ncu_timeout_seconds,
            }
            typer.echo(json.dumps(plan, indent=2))
            raise typer.Exit(code=0)
        _execute_benchmarks(
            targets=combined_targets or None,
            bench_root=active_bench_root,
            output_format=output_format,
            profile_type=profile_type,
            suite_timeout=effective_timeout,
            timeout_multiplier=timeout_multiplier,
            validity_profile=validity_profile,
            allow_foreign_gpu_processes=allow_foreign_gpu_processes,
            allow_portable_expectations_update=allow_portable_expectations_update,
            reproducible=reproducible,
            cold_start=cold_start,
            force_synchronize=force_sync,
            iterations=iterations,
            warmup=warmup,
            force_pipeline=force_pipeline,
            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            log_level=log_level,
            log_file=log_file,
            single_gpu=single_gpu,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            allow_mixed_provenance=allow_mixed_provenance,
            ncu_metric_set=ncu_metric_set,
            ncu_replay_mode=ncu_replay_mode,
            nsys_timeout_seconds=nsys_timeout_seconds,
            ncu_timeout_seconds=ncu_timeout_seconds,
            only_cuda=only_cuda,
            only_python=only_python,
            launch_via=launch_via,
            nproc_per_node=nproc_per_node,
            nnodes=nnodes,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint,
            torchrun_env=torchrun_env,
            target_extra_args=target_extra_args,
            # Verification - both enabled by default for valid benchmark comparisons
            verify_input=not (skip_verify or skip_input_verify),
            verify_output=not (skip_verify or skip_output_verify),
            # LLM options
            llm_analysis=llm_analysis or force_llm,
            force_llm=force_llm,
            llm_provider=llm_provider,
            apply_llm_patches=apply_llm_patches,
            rebenchmark_llm_patches=rebenchmark_llm_patches,
            patch_strategy=patch_strategy,
            llm_patch_retries=llm_patch_retries,
            use_llm_cache=not no_llm_cache,
            llm_explain=llm_explain,
        )

    @app.command("run-tier1")
    def run_tier1(
        config: Optional[Path] = Option(
            None,
            "--config",
            help="Path to tier-1 suite YAML (defaults to configs/benchmark_suites/tier1.yaml).",
        ),
        history_root: Optional[Path] = Option(
            None,
            "--history-root",
            help="Directory to store tier-1 history artifacts (default: artifacts/history/tier1).",
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
        output_format: Optional[str] = Option(
            None,
            "--format",
            "-f",
            help="Output format override: json, markdown, or both.",
            callback=lambda value: _validate_output_format(value) if value is not None else None,
        ),
        profile_type: Optional[str] = Option(
            None,
            "--profile",
            "-p",
            help="Profile override for the suite (defaults to tier-1 config).",
            callback=lambda value: _validate_profile_type(value) if value is not None else None,
        ),
        suite_timeout: Optional[int] = Option(
            14400,
            "--suite-timeout",
            help="Suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled)",
        ),
        timeout_seconds: Optional[int] = Option(
            None, "--timeout-seconds", help="Override suite timeout in seconds (0 disables timeout)"
        ),
        timeout_multiplier: float = Option(
            3.0, "--timeout-multiplier", help="Multiply all benchmark timeouts by this factor."
        ),
        validity_profile: str = Option(
            "strict",
            "--validity-profile",
            help=VALIDITY_PROFILE_HELP_TEXT,
            callback=_validate_validity_profile,
        ),
        reproducible: bool = Option(False, "--reproducible", help="Enable reproducible mode."),
        cold_start: bool = Option(
            False,
            "--cold-start",
            help="Reset GPU state between benchmarks for cold-start measurements.",
        ),
        force_sync: bool = Option(
            False,
            "--force-sync",
            help="Force a device-wide synchronize immediately after benchmark_fn().",
        ),
        iterations: Optional[int] = Option(
            None, "--iterations", help="Override benchmark iterations."
        ),
        warmup: Optional[int] = Option(
            None, "--warmup", help="Override benchmark warmup iterations."
        ),
        gpu_sm_clock_mhz: Optional[int] = Option(
            None, "--gpu-sm-clock-mhz", help="Lock the SM application clock (MHz) for this run."
        ),
        gpu_mem_clock_mhz: Optional[int] = Option(
            None,
            "--gpu-mem-clock-mhz",
            help="Lock the GPU memory application clock (MHz) for this run.",
        ),
        artifacts_dir: Optional[str] = Option(
            None, "--artifacts-dir", help="Base directory for benchmark run artifacts."
        ),
        run_id: Optional[str] = Option(None, "--run-id", help="Explicit run id."),
        log_level: str = Option(
            "INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"
        ),
        log_file: Optional[str] = Option(None, "--log-file", help="Path to log file."),
        single_gpu: bool = Option(False, "--single-gpu", help="Force single-GPU visibility."),
        accept_regressions: bool = Option(
            False,
            "--accept-regressions",
            help="Update expectation files when improvements are detected instead of flagging regressions.",
        ),
        update_expectations: bool = Option(
            False,
            "--update-expectations",
            help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.",
        ),
        allow_mixed_provenance: bool = Option(
            False,
            "--allow-mixed-provenance",
            help="Allow expectation updates when provenance differs (commit/hardware/profile mismatch) without forcing updates. Does NOT accept regressions (use --accept-regressions or --update-expectations).",
        ),
        ncu_metric_set: str = Option(
            "minimal",
            "--ncu-metric-set",
            help="Nsight Compute metric preset.",
            callback=_validate_ncu_metric_set,
        ),
        ncu_replay_mode: Optional[str] = Option(
            None,
            "--ncu-replay-mode",
            help="Nsight Compute replay mode: kernel or application. If omitted, suite or benchmark-specific config applies.",
            callback=_validate_ncu_replay_mode,
        ),
        nsys_timeout_seconds: Optional[int] = Option(
            None, "--nsys-timeout-seconds", help="Override Nsight Systems timeout in seconds."
        ),
        ncu_timeout_seconds: Optional[int] = Option(
            None, "--ncu-timeout-seconds", help="Override Nsight Compute timeout in seconds."
        ),
        allow_portable_expectations_update: bool = Option(
            False,
            "--allow-portable-expectations-update",
            help=PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
        ),
    ):
        """Run the canonical tier-1 suite and write summary/history artifacts."""
        from core.benchmark.suites.tier1 import run_tier1_suite as _run_tier1_suite

        effective_timeout = timeout_seconds if timeout_seconds is not None else suite_timeout
        result = _run_tier1_suite(
            config_path=config,
            history_root=history_root,
            bench_root=bench_root,
            profile_type=profile_type,
            output_format=output_format,
            suite_timeout=effective_timeout,
            timeout_multiplier=timeout_multiplier,
            validity_profile=validity_profile,
            allow_portable_expectations_update=allow_portable_expectations_update,
            reproducible=reproducible,
            cold_start=cold_start,
            force_synchronize=force_sync,
            iterations=iterations,
            warmup=warmup,
            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            log_level=log_level,
            log_file=log_file,
            single_gpu=single_gpu,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            allow_mixed_provenance=allow_mixed_provenance,
            ncu_metric_set=ncu_metric_set,
            ncu_replay_mode=ncu_replay_mode,
            nsys_timeout_seconds=nsys_timeout_seconds,
            ncu_timeout_seconds=ncu_timeout_seconds,
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": result["execution"]["run_id"],
                    "summary_path": str(result["summary_path"]),
                    "regression_summary_path": str(result["regression_summary_path"]),
                    "trend_snapshot_path": str(result["trend_snapshot_path"]),
                    "history_root": str(result["history_root"]),
                },
                indent=2,
            )
        )
        if _tier1_result_failure_count(result) > 0:
            raise typer.Exit(code=1)

    @app.command("run-e2e")
    def run_e2e(
        run_tier1: bool = Option(
            True, "--run-tier1/--no-run-tier1", help="Run the canonical tier-1 suite stage."
        ),
        run_full_sweep: bool = Option(
            False,
            "--run-full-sweep/--no-run-full-sweep",
            help="Run the heavier discovered full benchmark sweep.",
        ),
        run_cluster: bool = Option(
            True, "--run-cluster/--no-run-cluster", help="Run the cluster common-eval stage."
        ),
        run_fabric: bool = Option(
            True, "--run-fabric/--no-run-fabric", help="Run the dedicated fabric-eval stage."
        ),
        cluster_preset: str = Option(
            "common-answer-fast", "--cluster-preset", help="Cluster common-eval preset to run."
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
        hosts: Optional[List[str]] = Option(
            None,
            "--hosts",
            help="Cluster host list. Repeat the flag for multiple hosts; defaults to localhost.",
        ),
        labels: Optional[List[str]] = Option(
            None, "--labels", help="Optional labels matching --hosts."
        ),
        ssh_user: Optional[str] = Option(
            None, "--ssh-user", help="SSH user for non-local cluster runs."
        ),
        ssh_key: Optional[str] = Option(
            None, "--ssh-key", help="SSH key path for non-local cluster runs."
        ),
        profile_type: str = Option(
            "minimal",
            "--profile",
            "-p",
            help="Profiling preset: minimal (default), none, deep_dive, or roofline.",
            callback=_validate_profile_type,
        ),
        suite_timeout: Optional[int] = Option(
            14400,
            "--suite-timeout",
            help="Benchmark suite timeout in seconds (default: 14400 = 4 hours, 0 = disabled).",
        ),
        full_sweep_suite_timeout: Optional[int] = Option(
            0,
            "--full-sweep-suite-timeout",
            help="Aggregate suite timeout applied to each full-sweep benchmark bucket. Default 0 disables the bucket-wide watchdog while preserving per-benchmark/profiler timeouts.",
        ),
        timeout_seconds: Optional[int] = Option(
            None, "--timeout-seconds", help="Optional cluster stage timeout in seconds."
        ),
        artifacts_dir: Optional[str] = Option(
            None, "--artifacts-dir", help="Base directory for benchmark run artifacts."
        ),
        run_id: Optional[str] = Option(
            None, "--run-id", help="Explicit top-level e2e sweep run id."
        ),
        validity_profile: str = Option(
            "strict",
            "--validity-profile",
            help=VALIDITY_PROFILE_HELP_TEXT,
            callback=_validate_validity_profile,
        ),
        single_gpu: bool = Option(
            False, "--single-gpu", help="Force single-GPU visibility for benchmark stages."
        ),
        iterations: Optional[int] = Option(
            None, "--iterations", help="Override benchmark iterations."
        ),
        warmup: Optional[int] = Option(
            None, "--warmup", help="Override benchmark warmup iterations."
        ),
        gpu_sm_clock_mhz: Optional[int] = Option(
            None,
            "--gpu-sm-clock-mhz",
            help="Lock the SM application clock (MHz) for benchmark stages.",
        ),
        gpu_mem_clock_mhz: Optional[int] = Option(
            None,
            "--gpu-mem-clock-mhz",
            help="Lock the GPU memory application clock (MHz) for benchmark stages.",
        ),
        accept_regressions: bool = Option(
            False,
            "--accept-regressions",
            help="Update expectation files when improvements are detected instead of flagging regressions.",
        ),
        update_expectations: bool = Option(
            False,
            "--update-expectations",
            help="Force-write observed metrics into expectation files (overrides regressions). Useful for refreshing baselines on new hardware.",
        ),
        allow_mixed_provenance: bool = Option(
            False,
            "--allow-mixed-provenance",
            help="Allow expectation updates when provenance differs (commit/hardware/profile mismatch) without forcing updates. Does NOT accept regressions (use --accept-regressions or --update-expectations).",
        ),
        allow_portable_expectations_update: bool = Option(
            False,
            "--allow-portable-expectations-update",
            help=PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
        ),
        auto_resume: bool = Option(
            True,
            "--auto-resume/--no-auto-resume",
            help="Launch a detached watcher that auto-resumes stale or aborted e2e runs using the stored contract.",
        ),
        max_auto_resumes: int = Option(
            3,
            "--max-auto-resumes",
            help="Maximum detached auto-resume attempts before the watcher gives up.",
        ),
        watch_poll_interval_seconds: int = Option(
            15, "--watch-poll-interval-seconds", help="Detached watcher poll interval in seconds."
        ),
        resume: bool = Option(
            False,
            "--resume",
            help="Resume an aborted e2e run. Requires --run-id and preserves prior stage artifacts.",
        ),
        dry_run: bool = Option(
            False, "--dry-run", help="Describe planned execution without running stages."
        ),
    ):
        """Run tier1, optional full sweep, cluster common eval, and optional fabric eval in one orchestrated flow."""
        from core.benchmark.e2e_sweep import run_benchmark_e2e_sweep

        result = run_benchmark_e2e_sweep(
            run_tier1=run_tier1,
            run_full_sweep=run_full_sweep,
            run_cluster=run_cluster,
            run_fabric=run_fabric,
            cluster_preset=cluster_preset,
            hosts=hosts,
            labels=labels,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            bench_root=bench_root,
            profile_type=profile_type,
            suite_timeout=suite_timeout,
            full_sweep_suite_timeout=full_sweep_suite_timeout,
            timeout_seconds=timeout_seconds,
            validity_profile=validity_profile,
            allow_portable_expectations_update=allow_portable_expectations_update,
            iterations=iterations,
            warmup=warmup,
            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
            artifacts_dir=artifacts_dir,
            single_gpu=single_gpu,
            accept_regressions=accept_regressions,
            update_expectations=update_expectations,
            allow_mixed_provenance=allow_mixed_provenance,
            auto_resume=auto_resume,
            max_auto_resumes=max_auto_resumes,
            watch_poll_interval_seconds=watch_poll_interval_seconds,
            run_id=run_id,
            resume=resume,
            dry_run=dry_run,
        )
        typer.echo(json.dumps(result, indent=2))
        if result.get("success") is False:
            raise typer.Exit(code=1)

    @app.command("run-e2e-status")
    def run_e2e_status(
        run_id: Optional[str] = Option(
            None, "--run-id", help="Explicit e2e run id to inspect; defaults to the latest run."
        ),
        artifacts_dir: Optional[str] = Option(
            None, "--artifacts-dir", help="Base directory for benchmark run artifacts."
        ),
        recent_events: int = Option(
            10, "--recent-events", help="How many recent top-level and child events to include."
        ),
        watch: bool = Option(
            False,
            "--watch",
            help="Continuously print status snapshots until the run reaches a terminal state.",
        ),
        poll_interval_seconds: int = Option(
            15,
            "--poll-interval-seconds",
            help="Polling interval in seconds when --watch is enabled.",
        ),
        as_json: bool = Option(
            False, "--json", help="Emit structured JSON instead of the concise text view."
        ),
    ):
        from core.benchmark.e2e_sweep import (
            inspect_benchmark_e2e_sweep_run,
            render_benchmark_e2e_status_text,
        )

        last_rendered: Optional[str] = None
        while True:
            result = inspect_benchmark_e2e_sweep_run(
                run_id=run_id,
                artifacts_dir=artifacts_dir,
                recent_events_limit=recent_events,
            )
            rendered = (
                json.dumps(result, indent=2)
                if as_json
                else render_benchmark_e2e_status_text(result)
            )
            if rendered != last_rendered:
                typer.echo(rendered)
                last_rendered = rendered
                if watch:
                    typer.echo("")
            if not watch:
                if result.get("success") is False:
                    raise typer.Exit(code=1)
                return
            if str(result.get("inferred_state") or "") in {"completed", "aborted_terminal"}:
                if result.get("success") is False:
                    raise typer.Exit(code=1)
                return
            time.sleep(max(1, poll_interval_seconds))

    @app.command("watch-e2e")
    def watch_e2e(
        run_id: Optional[str] = Option(
            None, "--run-id", help="Explicit e2e run id to supervise; defaults to the latest run."
        ),
        poll_interval_seconds: int = Option(
            15, "--poll-interval-seconds", help="Detached watcher poll interval in seconds."
        ),
        max_auto_resumes: int = Option(
            3,
            "--max-auto-resumes",
            help="Maximum detached auto-resume attempts before the watcher gives up.",
        ),
    ):
        from core.benchmark.e2e_sweep import (
            resolve_latest_e2e_run_id,
            watch_benchmark_e2e_sweep_run,
        )

        resolved_run_id = run_id or resolve_latest_e2e_run_id()
        if not resolved_run_id:
            typer.echo(
                json.dumps({"success": False, "error": "No e2e runs found to watch."}, indent=2)
            )
            raise typer.Exit(code=1)
        result = watch_benchmark_e2e_sweep_run(
            run_id=resolved_run_id,
            poll_interval_seconds=poll_interval_seconds,
            max_auto_resumes=max_auto_resumes,
        )
        typer.echo(json.dumps(result, indent=2))
        if result.get("success") is False:
            raise typer.Exit(code=1)

    @app.command("explore")
    def explore(
        path: Path = Option(
            ...,
            "--path",
            "-p",
            help=(
                "Baseline wrapper path (baseline_*.py) or baseline_*.cu. "
                "If the wrapper is missing, the tool will attempt to auto-generate one. "
                "The tool then copies the wrapper and runs variants on the copy."
            ),
        ),
        copy_tag: str = Option(
            "mcp_copy",
            "--copy-tag",
            help="Suffix tag for copied baseline files (default: mcp_copy).",
        ),
        copy_cu: bool = Option(
            True, "--copy-cu/--no-copy-cu", help="Copy matching baseline_*.cu file if present."
        ),
        max_variants: int = Option(
            3, "--max-variants", help="Max number of LLM variants to report (clamped to 1-3)."
        ),
        deep_dive: str = Option(
            "auto",
            "--deep-dive",
            help="Deep-dive mode: auto, always, or never.",
            callback=_validate_deep_dive_mode,
        ),
        deep_dive_speedup_threshold: float = Option(
            1.05,
            "--deep-dive-speedup-threshold",
            help="Speedup threshold below which deep_dive is triggered in auto mode.",
        ),
        artifacts_dir: Optional[str] = Option(
            None,
            "--artifacts-dir",
            help="Base directory for run artifacts (default: ./artifacts/runs).",
        ),
        run_id: Optional[str] = Option(
            None, "--run-id", help="Run ID for artifacts (default: <timestamp>__explore__<label>)."
        ),
        iterations: Optional[int] = Option(
            None, "--iterations", help="Override benchmark iterations for minimal profiling run."
        ),
        warmup: Optional[int] = Option(
            None, "--warmup", help="Override warmup iterations for minimal profiling run."
        ),
        timeout_seconds: Optional[int] = Option(
            0,
            "--timeout-seconds",
            help="Max runtime for minimal profiling run (0 disables timeout).",
        ),
        deep_dive_iterations: int = Option(
            1, "--deep-dive-iterations", help="Override iterations for deep_dive run (default: 1)."
        ),
        deep_dive_warmup: int = Option(
            5,
            "--deep-dive-warmup",
            help="Override warmup iterations for deep_dive run (default: 5).",
        ),
        deep_dive_timeout_seconds: Optional[int] = Option(
            0,
            "--deep-dive-timeout-seconds",
            help="Max runtime for deep_dive run (0 disables timeout).",
        ),
        update_expectations: bool = Option(
            True,
            "--update-expectations/--no-update-expectations",
            help="Force-write observed metrics into expectation files (recommended).",
        ),
        validity_profile: str = Option(
            "strict",
            "--validity-profile",
            help=VALIDITY_PROFILE_HELP_TEXT,
            callback=_validate_validity_profile,
        ),
        allow_portable_expectations_update: bool = Option(
            False,
            "--allow-portable-expectations-update",
            help=PORTABLE_EXPECTATIONS_UPDATE_HELP_TEXT,
        ),
        async_run: bool = Option(
            False, "--async", help="Run in background and return job_id; poll with job_status."
        ),
    ):
        """Copy a baseline benchmark, run LLM variants with profiling, and compare utilization."""
        from mcp.mcp_server import tool_benchmark_explore

        params: Dict[str, Any] = {
            "path": str(path),
            "copy_tag": copy_tag,
            "copy_cu": copy_cu,
            "max_variants": max_variants,
            "deep_dive": deep_dive,
            "deep_dive_speedup_threshold": deep_dive_speedup_threshold,
            "artifacts_dir": artifacts_dir,
            "run_id": run_id,
            "iterations": iterations,
            "warmup": warmup,
            "timeout_seconds": timeout_seconds,
            "deep_dive_iterations": deep_dive_iterations,
            "deep_dive_warmup": deep_dive_warmup,
            "deep_dive_timeout_seconds": deep_dive_timeout_seconds,
            "update_expectations": update_expectations,
            "validity_profile": validity_profile,
            "allow_portable_expectations_update": allow_portable_expectations_update,
            "async": async_run,
        }
        result = tool_benchmark_explore(params)
        typer.echo(json.dumps(result, indent=2))
        if result.get("success") is False:
            raise typer.Exit(code=1)

    @app.command("verify")
    def verify(
        targets: Optional[List[str]] = Option(
            None,
            "--targets",
            "-t",
            help="Chapter(s) or chapter:example pairs to verify. Repeat the flag for multiple targets. Omit or use 'all' for every chapter.",
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
        verify_phase: str = Option(
            "gate",
            "--verify-phase",
            "-p",
            help="Verification enforcement phase: 'detect' (report only), 'quarantine' (exclude non-compliant), 'gate' (default, strict enforcement)",
        ),
        skip_jitter: bool = Option(
            False,
            "--skip-jitter",
            help="Skip jitter check (output changes when inputs are perturbed)",
        ),
        skip_fresh_input: bool = Option(
            False,
            "--skip-fresh-input",
            help="Skip fresh-input check (different seeds produce different outputs)",
        ),
        skip_workload: bool = Option(
            False,
            "--skip-workload",
            help="Skip workload invariant check (bytes/tokens/ops per iteration)",
        ),
        json_output: bool = Option(False, "--json", help="Output results as JSON"),
        verbose: bool = Option(
            False, "--verbose", "-v", help="Verbose output with detailed comparison info"
        ),
        clear_cache: bool = Option(
            False, "--clear-cache", help="Clear golden output cache before verification"
        ),
    ):
        """Verify benchmark pairs for correctness without measuring performance.

        Runs verification checks on baseline/optimized benchmark pairs:
        - Input signature matching (same batch size, dtypes, shapes)
        - Output correctness (optimized produces same outputs as baseline)
        - Jitter check (outputs change when inputs are perturbed)
        - Fresh-input check (different seeds produce different outputs)
        - Workload invariant check (same bytes/tokens/ops per iteration)

        Examples:
            aisp bench verify                     # Verify all chapters
            aisp bench verify -t ch11            # Verify chapter 11
            aisp bench verify -t ch11:streams    # Verify specific example
            aisp bench verify --json             # JSON output for CI
        """
        from core.benchmark.verify_runner import VerifyRunner, VerifyConfig
        from core.benchmark.verification import (
            EnforcementPhase,
            QuarantineReason,
            get_enforcement_phase,
        )
        from core.benchmark.quarantine import QuarantineManager
        from core.discovery import chapter_slug, discover_benchmarks
        import importlib.util
        import hashlib

        repo_root = Path(__file__).resolve().parents[2]
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root

        # Set enforcement phase
        os.environ["VERIFY_ENFORCEMENT_PHASE"] = verify_phase.lower()
        phase = get_enforcement_phase()

        # Initialize verification components
        cache_dir = active_bench_root / "artifacts" / "verify_cache"
        quarantine_mgr = QuarantineManager(cache_dir=cache_dir)

        if clear_cache:
            import shutil

            golden_cache = cache_dir / "golden_outputs"
            if golden_cache.exists():
                shutil.rmtree(golden_cache)
                typer.echo(f"🗑️  Cleared golden output cache: {golden_cache}")

        verify_config = VerifyConfig(
            skip_jitter_check=skip_jitter,
            skip_fresh_input_check=skip_fresh_input,
            skip_workload_check=skip_workload,
            seed=42,
        )
        verify_runner = VerifyRunner(
            cache_dir=cache_dir / "golden_outputs",
            quarantine_manager=quarantine_mgr,
        )

        # Resolve targets (chapters + per-chapter example filters)
        effective_targets = targets if targets else ["all"]
        chapter_dirs, chapter_filters = resolve_target_chapters(
            effective_targets,
            bench_root=active_bench_root,
            repo_root=active_bench_root,
        )

        results = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0

        def _load_benchmark_module(path: Path):
            """Load a Python benchmark module by path."""
            mod_id = hashlib.md5(str(path).encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(f"aisp_verify_{mod_id}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        def _load_benchmark(path: Path):
            mod = _load_benchmark_module(path)
            get_benchmark = getattr(mod, "get_benchmark", None)
            if not callable(get_benchmark):
                raise AttributeError(f"{path.name} is missing get_benchmark()")
            return get_benchmark()

        def _format_reason(reason: object) -> str:
            if reason is None:
                return "unknown"
            value = getattr(reason, "value", None)
            if value is not None:
                return str(value)
            return str(reason)

        def _to_quarantine_reason(reason: object) -> Optional[QuarantineReason]:
            if isinstance(reason, QuarantineReason):
                return reason
            value = getattr(reason, "value", None)
            if isinstance(value, str):
                try:
                    return QuarantineReason(value)
                except Exception:
                    return None
            if isinstance(reason, str):
                try:
                    return QuarantineReason(reason)
                except Exception:
                    return None
            return None

        def _dedup_paths(paths: List[Path]) -> List[Path]:
            seen = set()
            out: List[Path] = []
            for p in paths:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            return out

        for chapter_dir in chapter_dirs:
            chapter_slug_name = chapter_slug(
                chapter_dir, active_bench_root, bench_root=active_bench_root
            )
            allowed_examples = chapter_filters.get(chapter_slug_name, set())

            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"📋 Verifying {chapter_slug_name}")
            typer.echo(f"{'=' * 60}")

            try:
                pairs = discover_benchmarks(chapter_dir)
            except Exception as e:
                typer.echo(f"  ⚠️  Could not find benchmark pairs: {e}")
                skipped_count += 1
                results.append(
                    {
                        "chapter": chapter_slug_name,
                        "status": "skipped",
                        "reason": str(e),
                    }
                )
                continue

            if not pairs:
                typer.echo(f"  ⏭️  No benchmark pairs found")
                skipped_count += 1
                results.append(
                    {
                        "chapter": chapter_slug_name,
                        "status": "skipped",
                        "reason": "No benchmark pairs found",
                    }
                )
                continue

            # De-dupe alias pairs: discover_benchmarks() yields both canonical entries (one baseline to N optimized)
            # and per-variant alias entries. For verification we run the baseline once per file and validate only
            # the selected optimized variants.
            grouped: Dict[Path, Dict[str, object]] = {}
            for baseline_path, optimized_paths, example_name in pairs:
                base_example = baseline_path.stem.replace("baseline_", "", 1)
                group = grouped.setdefault(
                    baseline_path,
                    {
                        "base_example": base_example,
                        "optimized_paths": [],
                        "alias_map": {},
                    },
                )
                if example_name == base_example:
                    group["optimized_paths"] = optimized_paths
                else:
                    alias_map = group.get("alias_map", {})
                    if isinstance(alias_map, dict) and optimized_paths:
                        alias_map[example_name] = optimized_paths[0]

            for baseline_path in sorted(grouped.keys(), key=lambda p: p.as_posix()):
                group = grouped[baseline_path]
                base_example = str(
                    group.get("base_example", baseline_path.stem.replace("baseline_", "", 1))
                )
                canonical_opts = list(group.get("optimized_paths", []))  # type: ignore[list-item]
                if not canonical_opts:
                    continue

                selected_opts: List[Path]
                if not allowed_examples:
                    selected_opts = canonical_opts
                elif base_example in allowed_examples:
                    selected_opts = canonical_opts
                else:
                    alias_map = group.get("alias_map", {})
                    selected_opts = []
                    if isinstance(alias_map, dict):
                        for example in sorted(allowed_examples):
                            opt_path = alias_map.get(example)
                            if isinstance(opt_path, Path):
                                selected_opts.append(opt_path)

                selected_opts = _dedup_paths(selected_opts)
                if not selected_opts:
                    continue

                try:
                    baseline = _load_benchmark(baseline_path)
                except Exception as e:
                    load_reason = str(e)
                    load_skipped = load_reason.startswith("SKIPPED:")
                    for optimized_path in selected_opts:
                        pair_name = (
                            f"{base_example}/{optimized_path.stem.replace('optimized_', '')}"
                        )
                        typer.echo(f"\n  🔍 {pair_name}:")
                        typer.echo(f"      Baseline:  {baseline_path.name}")
                        typer.echo(f"      Optimized: {optimized_path.name}")
                        if load_skipped:
                            typer.echo(f"      ⏭️  SKIPPED: {load_reason}")
                            skipped_count += 1
                        else:
                            typer.echo(f"      ❌ ERROR: Failed to load baseline benchmark: {e}")
                            failed_count += 1
                        results.append(
                            {
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": "skipped" if load_skipped else "error",
                                "reason": load_reason
                                if load_skipped
                                else f"Failed to load baseline benchmark: {e}",
                            }
                        )
                    continue

                baseline_result = verify_runner.verify_baseline(baseline, config=verify_config)
                baseline_reason = _format_reason(baseline_result.reason)
                baseline_skipped = (not baseline_result.passed) and baseline_reason.startswith(
                    "SKIPPED:"
                )

                if not baseline_result.passed and not baseline_skipped:
                    if phase in (EnforcementPhase.QUARANTINE, EnforcementPhase.GATE):
                        qr = _to_quarantine_reason(baseline_result.reason)
                        if qr is not None:
                            quarantine_mgr.quarantine(str(baseline_path), qr)
                elif baseline_result.passed:
                    quarantine_mgr.clear_quarantine(str(baseline_path))

                for optimized_path in selected_opts:
                    pair_name = f"{base_example}/{optimized_path.stem.replace('optimized_', '')}"
                    typer.echo(f"\n  🔍 {pair_name}:")
                    typer.echo(f"      Baseline:  {baseline_path.name}")
                    typer.echo(f"      Optimized: {optimized_path.name}")

                    try:
                        if not baseline_result.passed:
                            status = "skipped" if baseline_skipped else "failed"
                            if baseline_skipped:
                                typer.echo(f"      ⏭️  SKIPPED: {baseline_reason}")
                                skipped_count += 1
                            else:
                                typer.echo(f"      ❌ FAILED: {baseline_reason}")
                                failed_count += 1
                            results.append(
                                {
                                    "chapter": chapter_slug_name,
                                    "pair": pair_name,
                                    "status": status,
                                    "reason": baseline_reason,
                                }
                            )
                            continue

                        try:
                            optimized = _load_benchmark(optimized_path)
                        except Exception as e:
                            load_reason = str(e)
                            load_skipped = load_reason.startswith("SKIPPED:")
                            if load_skipped:
                                typer.echo(f"      ⏭️  SKIPPED: {load_reason}")
                                skipped_count += 1
                            else:
                                typer.echo(
                                    f"      ❌ ERROR: Failed to load optimized benchmark: {e}"
                                )
                                failed_count += 1
                            results.append(
                                {
                                    "chapter": chapter_slug_name,
                                    "pair": pair_name,
                                    "status": "skipped" if load_skipped else "error",
                                    "reason": load_reason
                                    if load_skipped
                                    else f"Failed to load optimized benchmark: {e}",
                                }
                            )
                            continue

                        # Validate timing config matches (anti-gaming)
                        if not verify_config.skip_timing_validation:
                            timing_valid, timing_error = verify_runner._validate_timing_config(
                                baseline, optimized
                            )  # noqa: SLF001
                            if not timing_valid:
                                typer.echo(
                                    f"      ❌ FAILED: {QuarantineReason.TIMING_CONFIG_MISMATCH.value}"
                                )
                                if verbose and timing_error:
                                    typer.echo(f"         Details: {timing_error}")
                                failed_count += 1
                                results.append(
                                    {
                                        "chapter": chapter_slug_name,
                                        "pair": pair_name,
                                        "status": "failed",
                                        "reason": QuarantineReason.TIMING_CONFIG_MISMATCH.value,
                                        "details": timing_error,
                                    }
                                )
                                if phase in (EnforcementPhase.QUARANTINE, EnforcementPhase.GATE):
                                    quarantine_mgr.quarantine(
                                        str(optimized_path), QuarantineReason.TIMING_CONFIG_MISMATCH
                                    )
                                continue

                        # Run verification (baseline golden output already cached)
                        result = verify_runner.verify_optimized(optimized, config=verify_config)

                        if result.passed:
                            quarantine_mgr.clear_quarantine(str(optimized_path))
                            typer.echo(f"      ✅ PASSED")
                            passed_count += 1
                            results.append(
                                {
                                    "chapter": chapter_slug_name,
                                    "pair": pair_name,
                                    "status": "passed",
                                }
                            )
                        else:
                            reason_str = _format_reason(result.reason)
                            is_skipped = reason_str.startswith("SKIPPED:")
                            if is_skipped:
                                typer.echo(f"      ⏭️  SKIPPED: {reason_str}")
                                skipped_count += 1
                            else:
                                typer.echo(f"      ❌ FAILED: {reason_str}")
                                failed_count += 1
                            if verbose and result.details:
                                typer.echo(f"         Details: {result.details}")
                            results.append(
                                {
                                    "chapter": chapter_slug_name,
                                    "pair": pair_name,
                                    "status": "skipped" if is_skipped else "failed",
                                    "reason": reason_str,
                                    "details": result.details,
                                }
                            )

                            # Quarantine if in quarantine or gate phase
                            if (not is_skipped) and phase in (
                                EnforcementPhase.QUARANTINE,
                                EnforcementPhase.GATE,
                            ):
                                qr = _to_quarantine_reason(result.reason)
                                if qr is not None:
                                    quarantine_mgr.quarantine(str(optimized_path), qr)

                    except Exception as e:
                        typer.echo(f"      ❌ ERROR: {e}")
                        if verbose:
                            import traceback

                            typer.echo(f"         {traceback.format_exc()}")
                        failed_count += 1
                        results.append(
                            {
                                "chapter": chapter_slug_name,
                                "pair": pair_name,
                                "status": "error",
                                "reason": str(e),
                            }
                        )

        # Summary
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"📊 VERIFICATION SUMMARY")
        typer.echo(f"{'=' * 60}")
        typer.echo(f"  ✅ Passed:  {passed_count}")
        typer.echo(f"  ❌ Failed:  {failed_count}")
        typer.echo(f"  ⏭️  Skipped: {skipped_count}")
        typer.echo(f"  📈 Total:   {passed_count + failed_count + skipped_count}")

        if failed_count > 0:
            typer.echo(f"\n  ⚠️  Phase: {phase.value}")
            if phase == EnforcementPhase.GATE:
                typer.echo(
                    f"  🚫 Gate mode: non-compliant benchmarks will block performance measurement"
                )
            elif phase == EnforcementPhase.QUARANTINE:
                typer.echo(f"  🏷️  Quarantine mode: non-compliant benchmarks excluded from reports")
            else:
                typer.echo(f"  ℹ️  Detect mode: issues reported but not enforced")

        if json_output:
            output = {
                "summary": {
                    "passed": passed_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "phase": phase.value,
                },
                "results": results,
                "quarantine": {k: v.to_dict() for k, v in quarantine_mgr.get_all_records().items()}
                if hasattr(quarantine_mgr, "get_all_records")
                else {},
            }
            typer.echo("\n" + json.dumps(output, indent=2, default=str))

        # Exit with error if failed and in gate mode
        if failed_count > 0 and phase == EnforcementPhase.GATE:
            raise typer.Exit(code=1)

    @app.command("list-targets")
    def list_targets(
        chapter: Optional[str] = Option(
            None,
            "--chapter",
            "-c",
            help="Limit output to a single chapter (e.g., ch15 or labs/blackwell_matmul).",
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
    ):
        """List available benchmark targets in chapter:example format."""
        repo_root = Path(__file__).resolve().parents[2]
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
        if chapter:
            chapter_dirs, _ = resolve_target_chapters([chapter], bench_root=active_bench_root)
        else:
            chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])

        if not chapter_dirs:
            typer.echo("No chapter directories found.")
            raise typer.Exit(code=1)

        any_targets = False
        for chapter_dir in chapter_dirs:
            chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
            pairs = discover_benchmarks(chapter_dir)
            if not pairs:
                continue
            any_targets = True
            for _, _, example in sorted(pairs, key=lambda entry: entry[2]):
                typer.echo(f"{chapter_id}:{example}")

        if not any_targets:
            typer.echo("No benchmark targets discovered.")

    @app.command("list-chapters")
    def list_chapters(
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan for benchmarks (defaults to repo root).",
        ),
    ):
        """List all discoverable chapters and labs."""
        repo_root = Path(__file__).resolve().parents[2]
        active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
        chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])
        for chapter_dir in chapter_dirs:
            typer.echo(chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root))

    @app.command("analyze")
    def analyze(
        show_leaderboards: bool = Option(
            True, "--leaderboards/--no-leaderboards", help="Show separate speed/memory leaderboards"
        ),
        show_pareto: bool = Option(
            True, "--pareto/--no-pareto", help="Show Pareto-optimal benchmarks"
        ),
        show_tradeoffs: bool = Option(
            True, "--tradeoffs/--no-tradeoffs", help="Show cost-benefit trade-off analysis"
        ),
        show_recommendations: bool = Option(
            True,
            "--recommendations/--no-recommendations",
            help="Show constraint-based recommendations",
        ),
        show_chart: bool = Option(
            True, "--chart/--no-chart", help="Show ASCII trade-off scatter chart"
        ),
        top_n: int = Option(5, "--top", "-n", help="Number of entries to show per category"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Analyze benchmark results: Pareto frontier, trade-offs, and recommendations."""
        analyzer = _get_analyzer(data_file)

        results = {}

        if show_leaderboards:
            results["leaderboards"] = analyzer.get_categorized_leaderboards()
        if show_pareto:
            results["pareto"] = analyzer.get_pareto_frontier()
        if show_tradeoffs:
            results["tradeoffs"] = analyzer.get_tradeoff_analysis()
        if show_recommendations:
            results["recommendations"] = analyzer.get_constraint_recommendations()

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        # Pretty print results
        typer.echo("\n" + "=" * 70)
        typer.echo("📊 MULTI-METRIC BENCHMARK ANALYSIS")
        typer.echo("=" * 70)

        if show_leaderboards and "leaderboards" in results:
            boards = results["leaderboards"].get("leaderboards", {})

            # Speed leaderboard
            speed = boards.get("speed", {})
            typer.echo(f"\n🚀 SPEED CHAMPIONS ({speed.get('count', 0)} benchmarks)")
            typer.echo("-" * 50)
            for e in speed.get("entries", [])[:top_n]:
                rank_icon = (
                    "🥇"
                    if e["rank"] == 1
                    else "🥈"
                    if e["rank"] == 2
                    else "🥉"
                    if e["rank"] == 3
                    else f"#{e['rank']}"
                )
                typer.echo(f"  {rank_icon} {e['name']}: {e['primary_metric']}")

            # Memory leaderboard
            memory = boards.get("memory", {})
            if memory.get("entries"):
                typer.echo(f"\n💾 MEMORY CHAMPIONS ({memory.get('count', 0)} benchmarks)")
                typer.echo("-" * 50)
                for e in memory.get("entries", [])[:top_n]:
                    rank_icon = (
                        "🥇"
                        if e["rank"] == 1
                        else "🥈"
                        if e["rank"] == 2
                        else "🥉"
                        if e["rank"] == 3
                        else f"#{e['rank']}"
                    )
                    typer.echo(
                        f"  {rank_icon} {e['name']}: {e['primary_metric']} ({e['secondary_metric']})"
                    )

        if show_pareto and "pareto" in results:
            pareto = results["pareto"]
            typer.echo(
                f"\n⭐ PARETO-OPTIMAL BENCHMARKS ({pareto.get('pareto_count', 0)} / {pareto.get('total_count', 0)})"
            )
            typer.echo("-" * 50)
            typer.echo("  (No other benchmark is better on ALL metrics)")
            for p in pareto.get("pareto_frontier", [])[:top_n]:
                mem_str = f"-{p['memory_savings']:.0f}% mem" if p["memory_savings"] > 0 else "N/A"
                typer.echo(f"  ⭐ {p['name']}")
                typer.echo(f"      Speed: {p['speedup']:.2f}x | Memory: {mem_str}")

        if show_tradeoffs and "tradeoffs" in results:
            tradeoffs = results["tradeoffs"]
            typer.echo(f"\n⚡ EFFICIENCY RANKINGS (Cost-Benefit Analysis)")
            typer.echo("-" * 50)

            # Memory specialists
            mem_specs = tradeoffs.get("memory_specialists", [])
            if mem_specs:
                typer.echo("  💾 Memory Efficiency:")
                for t in mem_specs[:3]:
                    typer.echo(f"      {t['name']}: {t['benefit']} ({t['cost']})")

            # Speed specialists
            speed_specs = tradeoffs.get("speed_specialists", [])
            if speed_specs:
                typer.echo("  🚀 Speed Efficiency (top 3):")
                for t in speed_specs[:3]:
                    typer.echo(f"      {t['name']}: {t['benefit']} (eff={t['efficiency_score']})")

        if show_recommendations and "recommendations" in results:
            recs = results["recommendations"]
            typer.echo(f"\n🎯 RECOMMENDATIONS BY USE CASE")
            typer.echo("-" * 50)
            for scenario in recs.get("scenarios", []):
                typer.echo(f"\n  {scenario['icon']} {scenario['name']}")
                typer.echo(f"     {scenario['description']}")
                for r in scenario.get("recommendations", [])[:2]:
                    typer.echo(f"       → {r['name']}: {r['benefit']}")

        if show_chart and "pareto" in results:
            render_ascii_scatter_chart(results["pareto"])

        typer.echo("\n" + "=" * 70)
        typer.echo("💡 Tip: Use --json for machine-readable output")
        typer.echo("=" * 70 + "\n")

    def render_ascii_scatter_chart(pareto_data: dict, width: int = 60, height: int = 20):
        """Render an ASCII scatter chart of speed vs memory trade-offs."""
        all_points = pareto_data.get("all_points", [])
        pareto_points = pareto_data.get("pareto_frontier", [])
        pareto_names = set(p["name"] for p in pareto_points)

        if not all_points:
            return

        typer.echo(f"\n📈 SPEED vs MEMORY TRADE-OFF CHART")
        typer.echo("-" * 70)

        # Filter to reasonable range for visualization (log scale for speed)
        import math

        # Get ranges
        speedups = [p["speedup"] for p in all_points if p["speedup"] > 0]
        mem_savings = [p["memory_savings"] for p in all_points]

        if not speedups:
            typer.echo("  No data to display")
            return

        # Use log scale for speedup (clamped)
        min_speedup = max(0.1, min(speedups))
        max_speedup = min(1000, max(speedups))  # Cap at 1000x for visualization
        min_mem = min(mem_savings) if mem_savings else 0
        max_mem = max(mem_savings) if mem_savings else 100

        # Ensure some range
        if max_mem <= min_mem:
            max_mem = min_mem + 10

        # Create grid
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for p in all_points:
            speedup = max(min_speedup, min(max_speedup, p["speedup"]))
            mem = p["memory_savings"]

            # Log scale for x (speedup)
            if speedup > 0:
                x = int(
                    (math.log10(speedup) - math.log10(min_speedup))
                    / (math.log10(max_speedup) - math.log10(min_speedup) + 0.001)
                    * (width - 1)
                )
            else:
                x = 0

            # Linear scale for y (memory savings)
            y = int((mem - min_mem) / (max_mem - min_mem + 0.001) * (height - 1))

            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            y = height - 1 - y  # Flip y axis

            # Mark point
            if p["name"] in pareto_names:
                grid[y][x] = "★"  # Pareto optimal
            elif grid[y][x] == " ":
                grid[y][x] = "·"  # Regular point
            elif grid[y][x] == "·":
                grid[y][x] = "○"  # Multiple points

        # Draw chart
        typer.echo(f"  Memory")
        typer.echo(f"  Savings")
        typer.echo(f"  {max_mem:>5.0f}% ┌" + "─" * width + "┐")

        for i, row in enumerate(grid):
            if i == 0 or i == height - 1 or i == height // 2:
                label = f"{max_mem - (max_mem - min_mem) * i / (height - 1):>5.0f}%"
            else:
                label = "      "
            typer.echo(f"  {label} │{''.join(row)}│")

        typer.echo(f"  {min_mem:>5.0f}% └" + "─" * width + "┘")

        # X-axis labels
        typer.echo(
            f"         {min_speedup:<10.1f}x" + " " * (width - 25) + f"{max_speedup:>10.1f}x"
        )
        typer.echo(f"                              Speedup (log scale) →")

        # Legend
        typer.echo(f"\n  Legend: ★ = Pareto optimal  · = Regular  ○ = Multiple points")

    @app.command("summary")
    def summary(
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Show a quick summary of benchmark results."""
        data = load_benchmark_results(data_file)
        summary = data.get("summary", {})
        benchmarks = data.get("benchmarks", [])
        payload = {
            "total": summary.get("total_benchmarks", len(benchmarks)),
            "avg_speedup": summary.get("avg_speedup", 0),
            "max_speedup": summary.get("max_speedup", 0),
            "successful": summary.get("successful"),
            "failed": summary.get("failed"),
            "memory_optimizations": summary.get("memory_optimizations"),
            "speed_optimizations": summary.get("speed_optimizations"),
            "timestamp": data.get("timestamp"),
        }

        if json_output:
            typer.echo(json.dumps(payload, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("📜 BENCHMARK SUMMARY")
        typer.echo("=" * 60)
        typer.echo(f"Total benchmarks: {payload['total']}")
        typer.echo(f"Average speedup:  {payload['avg_speedup']:.2f}x")
        typer.echo(f"Max speedup:      {payload['max_speedup']:.2f}x")
        if payload.get("successful") is not None:
            typer.echo(
                f"Successful:       {payload['successful']} | Failed: {payload.get('failed', 0)}"
            )
        if payload.get("memory_optimizations") is not None:
            typer.echo(
                f"Memory-focused:   {payload['memory_optimizations']} | Speed-focused: {payload.get('speed_optimizations', 0)}"
            )
        typer.echo(f"Timestamp:        {payload.get('timestamp', 'N/A')}")

    @app.command("opportunities")
    def opportunities(
        data_file: Optional[Path] = Option(
            None,
            "--data-file",
            "-d",
            help="Path to benchmark_test_results.json, analyzer JSON, or tier-1 summary.json",
        ),
        catalog_file: Optional[Path] = Option(
            None,
            "--catalog-file",
            help="Optional JSON catalog of runnable targets used to surface unmeasured frontier probes",
        ),
        include_discovered_targets: bool = Option(
            False,
            "--include-discovered-targets",
            help="Also compare evidence against targets discovered from the benchmark tree",
        ),
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Root directory to scan when --include-discovered-targets is set",
        ),
        top_n: int = Option(
            10, "--top", "-n", help="Number of opportunities to return; use 0 for all"
        ),
        min_speedup: float = Option(
            1.10, "--min-speedup", help="Speedup below this is treated as a flat optimization"
        ),
        target_speedup: float = Option(
            1.50,
            "--target-speedup",
            help="Speedup below this is treated as compound-optimization headroom",
        ),
        min_memory_savings_pct: float = Option(
            10.0,
            "--min-memory-savings-pct",
            help="Memory-focused targets below this savings threshold are prioritized",
        ),
        slow_baseline_ms: float = Option(
            100.0,
            "--slow-baseline-ms",
            help="Baseline runtime threshold that increases opportunity priority",
        ),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        output_json: Optional[Path] = Option(
            None, "--output-json", help="Optional path to write the JSON result"
        ),
        output_md: Optional[Path] = Option(
            None, "--output-md", help="Optional path to write the Markdown report"
        ),
        output_run_queue_sh: Optional[Path] = Option(
            None,
            "--output-run-queue-sh",
            help="Optional path to write an executable shell runbook for the selected run queue",
        ),
        output_novelty_validation_sh: Optional[Path] = Option(
            None,
            "--output-novelty-validation-sh",
            help="Optional path to write an executable shell runbook for the top novelty validation leads",
        ),
        output_novelty_next_wave_sh: Optional[Path] = Option(
            None,
            "--output-novelty-next-wave-sh",
            help="Optional path to write an executable shell checklist for novelty next-wave actions",
        ),
        run_queue_root: Optional[Path] = Option(
            None,
            "--run-queue-root",
            help="Optional artifact root from a prior opportunity run queue; overlays resume and promotion status",
        ),
        novelty_queue_root: Optional[Path] = Option(
            None,
            "--novelty-queue-root",
            help="Optional artifact root from a prior novelty validation queue; overlays resume and review status",
        ),
    ):
        """Rank benchmark evidence by next optimization opportunity.

        This complements `bench triage`: triage focuses on pass/fail and regressions,
        while opportunities ranks the next experiments that can produce novel wins.
        """
        normalized_top_n = None if top_n <= 0 else top_n
        discovered_targets = (
            _discover_target_catalog(bench_root) if include_discovered_targets else None
        )
        if data_file:
            result = analyze_opportunity_file(
                data_file,
                catalog_path=catalog_file,
                target_catalog=discovered_targets,
                top_n=normalized_top_n,
                min_speedup=min_speedup,
                target_speedup=target_speedup,
                min_memory_savings_pct=min_memory_savings_pct,
                slow_baseline_ms=slow_baseline_ms,
            )
            if discovered_targets is not None:
                result["discovered_target_source"] = str(
                    Path(bench_root).resolve()
                    if bench_root
                    else Path(__file__).resolve().parents[2]
                )
        else:
            data = load_benchmark_results(None)
            if catalog_file:
                data = dict(data)
                data["target_catalog"] = json.loads(catalog_file.read_text(encoding="utf-8"))
            if discovered_targets is not None:
                data = dict(data)
                data["available_targets"] = discovered_targets
            candidates = normalize_candidates(data)
            result = rank_opportunities(
                candidates,
                top_n=normalized_top_n,
                min_speedup=min_speedup,
                target_speedup=target_speedup,
                min_memory_savings_pct=min_memory_savings_pct,
                slow_baseline_ms=slow_baseline_ms,
            )
            result["source"] = "discovered benchmark artifacts"
            if catalog_file:
                result["target_catalog_source"] = str(catalog_file)
            if discovered_targets is not None:
                result["discovered_target_source"] = str(
                    Path(bench_root).resolve()
                    if bench_root
                    else Path(__file__).resolve().parents[2]
                )

        if run_queue_root:
            result = apply_run_queue_feedback(result, summarize_run_queue_root(run_queue_root))
        if novelty_queue_root:
            result = apply_novelty_validation_feedback(
                result, summarize_run_queue_root(novelty_queue_root)
            )

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        markdown = render_opportunities_markdown(result)
        if output_md:
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(markdown + "\n", encoding="utf-8")
        if output_run_queue_sh:
            output_run_queue_sh.parent.mkdir(parents=True, exist_ok=True)
            output_run_queue_sh.write_text(render_run_queue_shell(result) + "\n", encoding="utf-8")
            output_run_queue_sh.chmod(0o755)
        if output_novelty_validation_sh:
            output_novelty_validation_sh.parent.mkdir(parents=True, exist_ok=True)
            output_novelty_validation_sh.write_text(
                render_novelty_validation_shell(result) + "\n", encoding="utf-8"
            )
            output_novelty_validation_sh.chmod(0o755)
        if output_novelty_next_wave_sh:
            output_novelty_next_wave_sh.parent.mkdir(parents=True, exist_ok=True)
            output_novelty_next_wave_sh.write_text(
                render_novelty_next_wave_shell(result) + "\n", encoding="utf-8"
            )
            output_novelty_next_wave_sh.chmod(0o755)

        if not result.get("opportunities"):
            if json_output:
                typer.echo(json.dumps(result, indent=2))
            else:
                typer.echo(
                    "No benchmark opportunities found. Pass --data-file or run benchmarks first."
                )
            return

        if json_output:
            typer.echo(json.dumps(result, indent=2))
            return

        typer.echo(markdown)

    @app.command("opportunity-catalog")
    def opportunity_catalog(
        bench_root: Optional[Path] = Option(
            None,
            "--bench-root",
            "-r",
            help="Benchmark tree root to scan; defaults to the repository code root",
        ),
        output_json: Optional[Path] = Option(
            None, "--output-json", help="Optional path to write the mined catalog JSON"
        ),
        json_output: bool = Option(False, "--json", help="Output the full catalog as JSON"),
        top_n: int = Option(
            20, "--top", "-n", help="Number of catalog targets to preview in text mode"
        ),
    ):
        """Mine runnable benchmark pairs into a frontier target catalog."""
        catalog = discover_benchmark_target_catalog(bench_root)
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        if json_output:
            typer.echo(json.dumps(catalog, indent=2))
            return

        typer.echo(f"Catalog source: {catalog['source']}")
        typer.echo(f"Targets: {catalog['target_count']}")
        typer.echo(f"Motifs: {catalog.get('motif_counts', {})}")
        typer.echo(f"Signals: {catalog.get('signal_counts', {})}")
        for entry in list(catalog.get("targets", []))[: max(0, top_n)]:
            signals = [
                str(match.get("signal")) for match in entry.get("frontier_signal_matches", [])[:3]
            ]
            typer.echo(
                f"- {entry.get('target')} [{entry.get('category')}] "
                f"signals={','.join(signals) if signals else 'none'}"
            )

    @app.command("opportunity-run-summary")
    def opportunity_run_summary(
        run_queue_root: Path = Option(
            ...,
            "--run-queue-root",
            "-r",
            help="Root directory produced by an opportunity run queue script",
        ),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Summarize evidence produced by an opportunity run queue."""
        summary = summarize_run_queue_root(run_queue_root)
        if json_output:
            typer.echo(json.dumps(summary, indent=2))
            return

        typer.echo(f"Run queue root: {summary['root']}")
        typer.echo(f"Jobs: {summary['job_count']}")
        typer.echo(f"Status counts: {summary['status_counts']}")
        promotion = summary.get("promotion_summary", {})
        typer.echo(
            "Promotions: "
            f"{promotion.get('claim_allowed_count', 0)} approved / "
            f"{promotion.get('promotion_job_count', 0)} total"
        )
        for action in summary.get("next_actions", []):
            typer.echo(f"- {action}")

    @app.command("triage")
    def triage(
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
        baseline_file: Optional[Path] = Option(
            None, "--baseline", "-b", help="Optional baseline for regression detection"
        ),
        auto_deep_dive: bool = Option(
            False,
            "--auto-deep-dive",
            help="Automatically run deep_dive profiling for top regressions.",
        ),
        deep_dive_top_n: int = Option(
            3,
            "--deep-dive-top-n",
            help="Number of regressed benchmarks to deep-dive when --auto-deep-dive is set.",
        ),
        deep_dive_iterations: int = Option(
            1, "--deep-dive-iterations", help="Iterations for deep-dive reruns."
        ),
        deep_dive_warmup: int = Option(
            5, "--deep-dive-warmup", help="Warmup iterations for deep-dive reruns."
        ),
        deep_dive_timeout_seconds: int = Option(
            900, "--deep-dive-timeout-seconds", help="Timeout for each deep-dive rerun."
        ),
        deep_dive_artifacts_dir: Optional[Path] = Option(
            None,
            "--deep-dive-artifacts-dir",
            help="Optional artifacts directory for deep-dive reruns.",
        ),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        top_n: int = Option(10, "--top", "-n", help="Number of items to show"),
    ):
        """Post-benchmark triage: analyze results and get actionable recommendations.

        Use after running benchmarks to:
        - Identify regressions and improvements
        - Get specific tool recommendations
        - Plan next optimization steps
        """
        data = load_benchmark_results(data_file)
        benchmarks = data.get("benchmarks", [])

        if not benchmarks:
            typer.echo("❌ No benchmark results found. Run `aisp bench run` first.")
            raise typer.Exit(1)

        def _status_bucket(status: object) -> str:
            normalized = str(status or "").strip().lower()
            if normalized in {"success", "ok"}:
                normalized = "succeeded"
            elif normalized in {"skip"}:
                normalized = "skipped"
            elif normalized in {"error"}:
                normalized = "failed"
            if normalized.startswith("failed_"):
                return "failed"
            if normalized in {"succeeded", "failed", "skipped"}:
                return normalized
            return normalized or "unknown"

        # Analyze
        total = len(benchmarks)
        passed = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "succeeded")
        failed = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "failed")
        skipped = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "skipped")
        avg_speedup = sum(b.get("speedup", 1.0) for b in benchmarks) / total if total > 0 else 0

        regressions = sorted(
            [
                b
                for b in benchmarks
                if _status_bucket(b.get("status")) == "succeeded" and b.get("speedup", 1.0) < 0.95
            ],
            key=lambda x: x.get("speedup", 1.0),
        )[:top_n]

        improvements = sorted(
            [
                b
                for b in benchmarks
                if _status_bucket(b.get("status")) == "succeeded" and b.get("speedup", 1.0) > 1.05
            ],
            key=lambda x: x.get("speedup", 1.0),
            reverse=True,
        )[:top_n]

        slow_kernels = [b for b in benchmarks if b.get("baseline_time_ms", 0) > 100]

        # Build recommendations
        recommendations = []
        if regressions:
            recommendations.append(
                {
                    "tool": "aisp profile bottleneck",
                    "reason": f"Identify root cause of {len(regressions)} regression(s)",
                    "priority": "high",
                }
            )
        if slow_kernels:
            recommendations.append(
                {
                    "tool": "aisp profile nsys",
                    "reason": f"Profile {len(slow_kernels)} slow benchmark(s) (>100ms)",
                    "priority": "medium",
                }
            )
        if improvements:
            recommendations.append(
                {
                    "tool": "aisp bench report --output report.html",
                    "reason": f"Document {len(improvements)} improvement(s)",
                    "priority": "low",
                }
            )
        recommendations.append(
            {
                "tool": "aisp bench compare-runs",
                "reason": "Compare with previous baseline",
                "priority": "medium",
            }
        )

        result = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "pass_rate": f"{(passed / total) * 100:.1f}%" if total > 0 else "N/A",
                "avg_speedup": round(avg_speedup, 2),
            },
            "regressions": [
                {
                    "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": round(b.get("speedup", 0), 2),
                    "baseline_ms": round(b.get("baseline_time_ms", 0), 1),
                }
                for b in regressions
            ],
            "improvements": [
                {
                    "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                    "speedup": round(b.get("speedup", 0), 2),
                    "baseline_ms": round(b.get("baseline_time_ms", 0), 1),
                }
                for b in improvements
            ],
            "recommendations": recommendations,
            "deep_dive_runs": [],
        }

        if auto_deep_dive and regressions:
            if deep_dive_top_n <= 0:
                if TYPER_AVAILABLE and typer is not None:
                    raise typer.BadParameter(
                        "--deep-dive-top-n must be >= 1 when --auto-deep-dive is enabled."
                    )
                raise ValueError("--deep-dive-top-n must be >= 1 when --auto-deep-dive is enabled.")
            selected = regressions[:deep_dive_top_n]
            if not json_output:
                typer.echo(f"\n🧪 AUTO DEEP-DIVE ({len(selected)} regression target(s)):")
            for bench in selected:
                chapter = str(bench.get("chapter", "")).strip()
                name = str(bench.get("name", "")).strip()
                if not chapter or not name:
                    result["deep_dive_runs"].append(
                        {
                            "target": "",
                            "status": "skipped",
                            "reason": "missing_chapter_or_name",
                        }
                    )
                    continue

                target = f"{chapter}:{name}"
                cmd: List[str] = [
                    sys.executable,
                    "-m",
                    "cli.aisp",
                    "bench",
                    "run",
                    "--targets",
                    target,
                    "--profile",
                    "deep_dive",
                    "--iterations",
                    str(deep_dive_iterations),
                    "--warmup",
                    str(deep_dive_warmup),
                    "--timeout-seconds",
                    str(deep_dive_timeout_seconds),
                ]
                if deep_dive_artifacts_dir:
                    cmd.extend(["--artifacts-dir", str(deep_dive_artifacts_dir)])

                if not json_output:
                    typer.echo(f"   • {target}")
                    typer.echo(f"     cmd: {' '.join(shlex.quote(token) for token in cmd)}")

                t0 = time.time()
                if json_output:
                    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
                else:
                    proc = subprocess.run(cmd, check=False)
                elapsed_s = time.time() - t0
                run_status = "ok" if proc.returncode == 0 else "failed"
                run_entry: Dict[str, Any] = {
                    "target": target,
                    "status": run_status,
                    "returncode": int(proc.returncode),
                    "elapsed_s": round(elapsed_s, 3),
                    "command": cmd,
                }
                if json_output:
                    # Preserve machine-readable triage output while retaining useful command diagnostics.
                    run_entry["stdout"] = (proc.stdout or "")[-4000:]
                    run_entry["stderr"] = (proc.stderr or "")[-4000:]
                result["deep_dive_runs"].append(run_entry)
                if not json_output:
                    typer.echo(f"     rc={proc.returncode} elapsed={elapsed_s:.1f}s")

        if json_output:
            typer.echo(json.dumps(result, indent=2))
            return

        # Pretty print
        typer.echo("\n" + "=" * 60)
        typer.echo("🔍 BENCHMARK TRIAGE")
        typer.echo("=" * 60)

        s = result["summary"]
        typer.echo(
            f"\n📊 Summary: {s['total']} benchmarks | {s['pass_rate']} pass rate | {s['avg_speedup']:.2f}x avg speedup"
        )

        if result["regressions"]:
            typer.echo(f"\n⚠️  REGRESSIONS ({len(result['regressions'])} found):")
            for r in result["regressions"]:
                typer.echo(
                    f"   • {r['benchmark']}: {r['speedup']:.2f}x ({r['baseline_ms']:.1f}ms baseline)"
                )
        else:
            typer.echo("\n✅ No regressions detected!")

        if result["improvements"]:
            typer.echo(f"\n🚀 TOP IMPROVEMENTS ({len(result['improvements'])} found):")
            for i in result["improvements"]:
                typer.echo(
                    f"   • {i['benchmark']}: {i['speedup']:.2f}x ({i['baseline_ms']:.1f}ms baseline)"
                )

        typer.echo(f"\n💡 RECOMMENDED NEXT STEPS:")
        for i, rec in enumerate(result["recommendations"], 1):
            priority_icon = (
                "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
            )
            typer.echo(f"   {i}. {priority_icon} {rec['tool']}")
            typer.echo(f"      └─ {rec['reason']}")

        if result["deep_dive_runs"]:
            ok = sum(1 for r in result["deep_dive_runs"] if r.get("status") == "ok")
            fail = sum(1 for r in result["deep_dive_runs"] if r.get("status") == "failed")
            skipped_dd = sum(1 for r in result["deep_dive_runs"] if r.get("status") == "skipped")
            typer.echo(
                f"\n🧾 DEEP-DIVE SUMMARY: {ok} succeeded, {fail} failed, {skipped_dd} skipped"
            )
            for entry in result["deep_dive_runs"]:
                target = entry.get("target") or "<unknown>"
                status = entry.get("status")
                rc = entry.get("returncode")
                elapsed = entry.get("elapsed_s")
                if status == "skipped":
                    typer.echo(f"   • {target}: skipped ({entry.get('reason', 'unknown')})")
                else:
                    typer.echo(f"   • {target}: {status} rc={rc} elapsed={elapsed}s")

        typer.echo()

    @app.command("expectations")
    def expectations(
        hardware: str = Option("b200", "--hardware", "-H", help="Hardware key (e.g., b200, h100)"),
        min_speedup: float = Option(1.05, "--min-speedup", help="Minimum improvement threshold"),
        goal: str = Option(
            "any", "--goal", help="Filter by optimization goal: speed, memory, or any"
        ),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        show_all: bool = Option(False, "--all", help="Show all entries (not just below threshold)"),
        multi_gpu_only: bool = Option(
            False, "--multi-gpu-only", help="Only show multi-GPU benchmarks"
        ),
        single_gpu_only: bool = Option(
            False, "--single-gpu-only", help="Only show single-GPU benchmarks"
        ),
        strict: bool = Option(False, "--strict", help="Exit non-zero if missing required metrics"),
    ):
        """Report expectation entries that miss a target improvement threshold."""
        goal_norm = goal.strip().lower()
        if goal_norm not in {"any", "speed", "memory"}:
            raise typer.BadParameter("Goal must be one of: any, speed, memory")
        if min_speedup <= 0:
            raise typer.BadParameter("min_speedup must be positive")
        if multi_gpu_only and single_gpu_only:
            raise typer.BadParameter(
                "Choose either --multi-gpu-only or --single-gpu-only, not both"
            )

        repo_root = Path(__file__).resolve().parents[2]
        files = _collect_expectations(hardware, repo_root)
        if not files:
            typer.echo(f"No expectations_{hardware}.json files found.")
            raise typer.Exit(code=1)

        entries: List[Dict[str, Any]] = []
        missing: List[Dict[str, Any]] = []
        total_entries = 0
        for exp_path, data, multi_map in files:
            examples = data.get("examples", {})
            if not isinstance(examples, dict):
                continue
            for example_name, entry in examples.items():
                if not isinstance(entry, dict):
                    continue
                metadata = entry.get("metadata", {}) or {}
                entry_goal = (metadata.get("optimization_goal") or "speed").strip().lower()
                if goal_norm != "any" and entry_goal != goal_norm:
                    continue
                entry_type = (entry.get("type") or "python").strip().lower()
                entry_example = entry.get("example") or example_name
                example_key = _expectation_example_key(entry_example, entry_type)
                if example_key not in multi_map:
                    continue
                total_entries += 1
                metrics = entry.get("metrics", {}) or {}
                multi_gpu = multi_map.get(example_key)
                if multi_gpu_only and not multi_gpu:
                    continue
                if single_gpu_only and multi_gpu:
                    continue

                baseline_ms = metrics.get("baseline_time_ms")
                optimized_ms = metrics.get("best_optimized_time_ms")
                baseline_mb = metrics.get("baseline_memory_mb")
                optimized_mb = metrics.get("best_optimized_memory_mb")
                improvement = None
                metric_name = None

                if entry_goal == "memory":
                    metric_name = "best_memory_savings_ratio"
                    improvement = metrics.get(metric_name)
                else:
                    metric_name = "best_speedup"
                    improvement = metrics.get(metric_name) or metrics.get("best_optimized_speedup")

                if improvement is None:
                    missing.append(
                        {
                            "path": str(exp_path.relative_to(repo_root)),
                            "example": example_name,
                            "goal": entry_goal,
                            "metric": metric_name,
                            "multi_gpu": multi_gpu,
                        }
                    )
                    continue

                try:
                    improvement_f = float(improvement)
                except (TypeError, ValueError):
                    missing.append(
                        {
                            "path": str(exp_path.relative_to(repo_root)),
                            "example": example_name,
                            "goal": entry_goal,
                            "metric": metric_name,
                            "multi_gpu": multi_gpu,
                        }
                    )
                    continue

                if show_all or improvement_f < min_speedup:
                    entries.append(
                        {
                            "path": str(exp_path.relative_to(repo_root)),
                            "example": example_name,
                            "goal": entry_goal,
                            "improvement": improvement_f,
                            "multi_gpu": multi_gpu,
                            "baseline_time_ms": baseline_ms,
                            "optimized_time_ms": optimized_ms,
                            "baseline_memory_mb": baseline_mb,
                            "optimized_memory_mb": optimized_mb,
                            "is_regression": metrics.get("is_regression"),
                            "type": entry.get("type"),
                        }
                    )

        entries.sort(key=lambda e: e["improvement"])
        summary = {
            "hardware": hardware,
            "files_scanned": len(files),
            "entries_scanned": total_entries,
            "entries_reported": len(entries),
            "missing_metrics": len(missing),
            "min_speedup": min_speedup,
            "goal_filter": goal_norm,
            "multi_gpu_only": multi_gpu_only,
            "single_gpu_only": single_gpu_only,
        }

        if json_output:
            typer.echo(
                json.dumps({"summary": summary, "entries": entries, "missing": missing}, indent=2)
            )
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo("📌 EXPECTATION THRESHOLD REPORT")
            typer.echo("=" * 60)
            for key, value in summary.items():
                typer.echo(f"{key.replace('_', ' ').title():<20} {value}")
            if not entries:
                typer.echo("\nNo entries matched the filters.")
            else:
                typer.echo("\nEntries:")
                for entry in entries:
                    baseline = entry["baseline_time_ms"]
                    optimized = entry["optimized_time_ms"]
                    baseline_mb = entry["baseline_memory_mb"]
                    optimized_mb = entry["optimized_memory_mb"]
                    if entry["goal"] == "memory":
                        detail = f"memory {baseline_mb}->{optimized_mb} MB"
                    else:
                        detail = f"time {baseline}->{optimized} ms"
                    typer.echo(
                        f"  {entry['improvement']:.4f}x  {entry['goal']:<6}  "
                        f"{entry['path']}::{entry['example']}  "
                        f"multi_gpu={entry['multi_gpu']}  {detail}"
                    )
            if missing:
                typer.echo("\nMissing metrics:")
                for item in missing:
                    typer.echo(
                        f"  {item['path']}::{item['example']} "
                        f"(goal={item['goal']}, metric={item['metric']}, multi_gpu={item['multi_gpu']})"
                    )

        if strict and missing:
            raise typer.Exit(code=1)

    @app.command("whatif")
    def whatif(
        vram: Optional[float] = Option(None, "--vram", "-v", help="Max VRAM in GB (e.g., 24)"),
        latency: Optional[float] = Option(
            None, "--latency", "-l", help="Max latency in ms (e.g., 50)"
        ),
        memory: Optional[float] = Option(None, "--memory", "-m", help="Max memory budget in GB"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """What-If Solver: Find optimizations that fit your constraints."""
        analyzer = _get_analyzer(data_file)
        params = {}
        if vram:
            params["vram"] = [str(vram)]
        if latency:
            params["latency"] = [str(latency)]
        if memory:
            params["memory_budget"] = [str(memory)]

        results = analyzer.get_whatif_recommendations(params)

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("🔍 WHAT-IF CONSTRAINT SOLVER")
        typer.echo("=" * 60)

        constraints = results.get("constraints", {})
        active_constraints = [f"{k}={v}" for k, v in constraints.items() if v]
        if active_constraints:
            typer.echo(f"\nConstraints: {', '.join(active_constraints)}")
        else:
            typer.echo("\nNo constraints specified - showing all optimizations")

        typer.echo(
            f"Matching: {results.get('matching_count', 0)} / {results.get('total_benchmarks', 0)} benchmarks"
        )

        if results.get("best_for_speed"):
            typer.echo(
                f"\n🚀 Best for Speed: {results['best_for_speed']['name']} ({results['best_for_speed']['speedup']:.2f}x)"
            )
        if results.get("best_for_memory"):
            mem = results["best_for_memory"]
            if mem["memory_savings_pct"]:
                typer.echo(f"💾 Best for Memory: {mem['name']} (-{mem['memory_savings_pct']:.0f}%)")

        typer.echo(f"\n📋 Top Recommendations:")
        typer.echo("-" * 60)
        for i, r in enumerate(results.get("recommendations", [])[:8], 1):
            mem_str = f"-{r['memory_savings_pct']:.0f}% mem" if r["memory_savings_pct"] else ""
            typer.echo(f"  {i}. {r['name']}: {r['speedup']:.2f}x {mem_str}")

    @app.command("stacking")
    def stacking(
        top_n: int = Option(5, "--top", "-n", help="Number of items to show per section"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Show which optimizations combine well together."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_optimization_stacking()

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("🔗 OPTIMIZATION STACKING GUIDE")
        typer.echo("=" * 60)

        typer.echo("\n✅ COMPATIBLE COMBINATIONS:")
        typer.echo("-" * 60)
        for c in results.get("compatible_combinations", [])[:top_n]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}")
            typer.echo(f"    Synergy: {c['synergy']} | {c['benefit']}")

        typer.echo("\n❌ INCOMPATIBLE COMBINATIONS:")
        typer.echo("-" * 60)
        for c in results.get("incompatible_combinations", [])[:top_n]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}")
            typer.echo(f"    Reason: {c['reason']}")

        typer.echo("\n🎯 RECOMMENDED STACKS:")
        typer.echo("-" * 60)
        for s in results.get("recommended_stacks", [])[:top_n]:
            typer.echo(f"  {s['name']}:")
            typer.echo(f"    Stack: {' → '.join(s['stack'])}")
            typer.echo(f"    Expected: {s['expected_benefit']}")

    @app.command("power")
    def power(
        top_n: int = Option(10, "--top", "-n", help="Number of entries to show"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Analyze power efficiency (ops/watt) of benchmarks."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_power_efficiency()

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("⚡ POWER EFFICIENCY ANALYSIS")
        typer.echo("=" * 60)

        typer.echo(f"\nBenchmarks with power data: {results.get('total_benchmarks_with_power', 0)}")
        typer.echo(f"Average power: {results.get('avg_power_w', 0):.1f}W")

        if results.get("most_efficient"):
            e = results["most_efficient"]
            typer.echo(f"\n🏆 Most Efficient: {e['name']}")
            typer.echo(
                f"   {e['ops_per_watt']:.4f} ops/watt | {e['power_w']:.0f}W | {e['speedup']:.2f}x"
            )

        typer.echo(f"\n📊 Efficiency Rankings (ops/watt):")
        typer.echo("-" * 60)
        for i, e in enumerate(results.get("efficiency_rankings", [])[:top_n], 1):
            typer.echo(f"  {i:2}. {e['name'][:40]:<40} {e['ops_per_watt']:.4f}")

    @app.command("cost")
    def cost(
        top_n: int = Option(10, "--top", "-n", help="Number of entries to show"),
        gpu: str = Option(
            "B200", "--gpu", "-g", help="GPU type (B200, H100, A100, L40S, A10G, T4)"
        ),
        rate: Optional[float] = Option(None, "--rate", "-r", help="Custom hourly rate in $/hr"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Calculate cost savings ($/token) for optimizations."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_cost_analysis(gpu=gpu, custom_rate=rate)

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("💰 COST ANALYSIS")
        typer.echo("=" * 60)

        typer.echo(
            f"\nAssuming: {results.get('assumed_gpu', 'B200')} @ ${results.get('hourly_rate', 0):.2f}/hr"
        )

        if results.get("highest_savings"):
            h = results["highest_savings"]
            typer.echo(f"\n🏆 Highest Savings: {h['name']}")
            typer.echo(
                f"   ${h['baseline_cost_per_m']:.4f} → ${h['optimized_cost_per_m']:.4f} per 1M ops"
            )
            typer.echo(f"   Savings: ${h['savings_per_m']:.4f}/1M ops ({h['savings_pct']:.0f}%)")

        typer.echo(f"\n📊 Cost Savings Rankings:")
        typer.echo("-" * 60)
        typer.echo(f"  {'Benchmark':<40} {'Savings':>10}")
        typer.echo("-" * 60)
        for c in results.get("cost_rankings", [])[:top_n]:
            typer.echo(f"  {c['name'][:40]:<40} {c['savings_pct']:>8.0f}%")

    @app.command("scaling")
    def scaling(
        json_output: bool = Option(False, "--json", help="Output as JSON"),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Analyze how optimizations scale with workload size."""
        analyzer = _get_analyzer(data_file)
        results = analyzer.get_scaling_analysis()

        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return

        typer.echo("\n" + "=" * 60)
        typer.echo("📈 SCALING ANALYSIS")
        typer.echo("=" * 60)

        typer.echo(f"\n💡 Key Insight: {results.get('key_insight', '')}")

        typer.echo("\n📊 Top Optimizations by Category:")
        typer.echo("-" * 60)
        for cat, benchmarks in results.get("categories", {}).items():
            if benchmarks:
                typer.echo(f"\n  {cat.upper()}:")
                for b in benchmarks[:3]:
                    typer.echo(f"    • {b['name']}: {b['speedup']:.2f}x")

        typer.echo("\n🎯 Scaling Recommendations:")
        typer.echo("-" * 60)
        for r in results.get("scaling_recommendations", []):
            typer.echo(f"\n  {r['factor']}:")
            typer.echo(f"    {r['insight']}")
            typer.echo(f"    → {r['recommendation']}")

    @app.command("report")
    def report(
        data_file: Optional[str] = Option(
            None, "--data-file", "-d", help="Path or URL to benchmark_test_results.json"
        ),
        output: Path = Option(
            Path("report.pdf"), "--output", "-o", help="Output file (.pdf or .html)"
        ),
        format: str = Option("pdf", "--format", "-f", help="pdf or html"),
        title: str = Option("GPU Performance Report", "--title", help="Report title"),
        author: str = Option("AI Performance Engineering", "--author", help="Author/owner"),
    ):
        """Generate PDF/HTML report from benchmark results."""
        try:
            from core.analysis.reporting.generator import generate_report, ReportConfig
        except Exception as exc:  # pragma: no cover - optional dependency
            typer.echo(f"Report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)

        input_path = data_file or "benchmark_test_results.json"
        cfg = ReportConfig(title=title, author=author)
        path = generate_report(input_path, str(output), format=format, config=cfg)
        typer.echo(f"✅ Report generated: {path}")

    @app.command("verify-report")
    def verify_report(
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
        output: Path = Option(
            Path("verification_report.html"), "--output", "-o", help="Output file (.html or .json)"
        ),
        format: str = Option("html", "--format", "-f", help="html or json"),
        gpu: str = Option(
            "", "--gpu", "-g", help="GPU name for theoretical peak (e.g., H100, B200)"
        ),
        title: str = Option("Benchmark Verification Report", "--title", help="Report title"),
        quarantine: Optional[Path] = Option(
            None, "--quarantine", "-q", help="Path to quarantine.json"
        ),
    ):
        """Generate verification report with anti-cheat status and theoretical peaks."""
        try:
            from core.analysis.reporting.verification_report import generate_verification_report
        except ImportError as exc:
            typer.echo(f"Verification report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)

        input_path = str(data_file) if data_file else "benchmark_test_results.json"
        fmt = format.lower()
        if fmt not in ("html", "json"):
            typer.echo("Format must be html or json", err=True)
            raise typer.Exit(code=1)

        try:
            path = generate_verification_report(
                input_path,
                str(output),
                format=fmt,
                gpu_name=gpu,
                title=title,
                quarantine_path=quarantine,
            )
            typer.echo(f"✅ Verification report generated: {path}")
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    @app.command("theoretical-peak")
    def theoretical_peak(
        gpu: str = Option(..., "--gpu", "-g", help="GPU name (e.g., H100, B200, A100)"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Show theoretical peak performance for a GPU."""
        try:
            from core.analysis.reporting.verification_report import (
                get_theoretical_peak,
                GPU_THEORETICAL_PEAKS,
            )
        except ImportError as exc:
            typer.echo(f"Theoretical peak lookup unavailable: {exc}", err=True)
            raise typer.Exit(code=1)

        if gpu.lower() == "list":
            typer.echo("Available GPUs:")
            for key in sorted(GPU_THEORETICAL_PEAKS.keys()):
                p = GPU_THEORETICAL_PEAKS[key]
                typer.echo(f"  {key}: {p.gpu_name} ({p.architecture})")
            return

        peak = get_theoretical_peak(gpu)
        if not peak:
            typer.echo(f"GPU '{gpu}' not found. Use --gpu list to see available GPUs.", err=True)
            raise typer.Exit(code=1)

        if json_output:
            import json as json_module

            typer.echo(json_module.dumps(peak.to_dict(), indent=2))
        else:
            typer.echo(f"\n🎯 Theoretical Peak: {peak.gpu_name}")
            typer.echo(f"   Architecture: {peak.architecture}")
            typer.echo(f"   SMs: {peak.sm_count}")
            typer.echo("")
            typer.echo("   Compute:")
            typer.echo(f"     FP32:  {peak.fp32_tflops:,.1f} TFLOPS")
            typer.echo(f"     FP16:  {peak.fp16_tflops:,.1f} TFLOPS")
            typer.echo(f"     BF16:  {peak.bf16_tflops:,.1f} TFLOPS")
            if peak.fp8_tflops > 0:
                typer.echo(f"     FP8:   {peak.fp8_tflops:,.1f} TFLOPS")
            typer.echo(f"     INT8:  {peak.int8_tops:,.1f} TOPS")
            typer.echo("")
            typer.echo("   Memory:")
            typer.echo(f"     Bandwidth: {peak.memory_bandwidth_gbps:,.0f} GB/s")
            typer.echo(f"     Size:      {peak.memory_size_gb:.0f} GB")
            typer.echo(f"     L2 Cache BW: ~{peak.l2_cache_bandwidth_gbps:,.0f} GB/s")
            typer.echo("")
            typer.echo(f"   TDP: {peak.tdp_watts:.0f}W")

    @app.command("quarantine-report")
    def quarantine_report(
        format: str = Option("text", "--format", "-f", help="text, markdown, or json"),
        output: Optional[Path] = Option(None, "--output", "-o", help="Output file"),
        quarantine_file: Optional[Path] = Option(
            None, "--quarantine-file", help="Path to quarantine.json"
        ),
    ):
        """Generate report of quarantined benchmarks."""
        try:
            from core.scripts.ci.generate_quarantine_report import (
                generate_text_report,
                generate_markdown_report,
                generate_json_report,
            )
            from core.benchmark.quarantine import QuarantineManager
        except ImportError as exc:
            typer.echo(f"Quarantine report generation unavailable: {exc}", err=True)
            raise typer.Exit(code=1)

        # Initialize quarantine manager
        if quarantine_file:
            cache_dir = quarantine_file.parent
        else:
            cache_dir = Path("artifacts/verify_cache")

        manager = QuarantineManager(cache_dir=cache_dir)

        # Generate report
        if format == "text":
            report = generate_text_report(manager)
        elif format == "markdown":
            report = generate_markdown_report(manager)
        elif format == "json":
            report = generate_json_report(manager)
        else:
            typer.echo("Format must be text, markdown, or json", err=True)
            raise typer.Exit(code=1)

        # Output
        if output:
            output.write_text(report)
            typer.echo(f"✅ Report written to {output}")
        else:
            typer.echo(report)

    @app.command("export")
    def export(
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
        format: str = Option("csv", "--format", "-f", help="csv, markdown, or json"),
        output: Optional[Path] = Option(None, "--output", "-o", help="Output file path"),
    ):
        """Export benchmark results to csv/markdown/json."""
        analyzer = _get_analyzer(data_file)
        data = analyzer._load_data()  # type: ignore[attr-defined]  # lightweight export
        fmt = format.lower()
        out_path = output or Path(f"export.{fmt}")
        benchmarks = data.get("benchmarks", [])

        if fmt == "json":
            out_path.write_text(json.dumps(data, indent=2))
        elif fmt == "markdown":
            lines = ["| Benchmark | Speedup | Baseline (ms) | Type |", "|---|---|---|---|"]
            for b in benchmarks:
                lines.append(
                    f"| {b.get('chapter')}:{b.get('name')} | {b.get('speedup', 0):.2f}x | {b.get('baseline_time_ms', 0):.3f} | {b.get('type', 'python')} |"
                )
            out_path.write_text("\n".join(lines))
        elif fmt == "csv":
            import csv

            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["benchmark", "speedup", "baseline_ms", "type"])
                for b in benchmarks:
                    writer.writerow(
                        [
                            f"{b.get('chapter')}:{b.get('name')}",
                            f"{b.get('speedup', 0):.2f}",
                            f"{b.get('baseline_time_ms', 0):.3f}",
                            b.get("type", "python"),
                        ]
                    )
        else:
            typer.echo("Format must be one of: csv, markdown, json", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"✅ Exported to {out_path}")

    @app.command("compare-runs")
    def compare_runs(
        baseline: Path = Option(
            ..., "--baseline", "-b", help="Baseline benchmark_test_results.json"
        ),
        candidate: Path = Option(
            ..., "--candidate", "-c", help="Candidate benchmark_test_results.json"
        ),
        top: int = Option(10, "--top", "-n", help="Top regressions/improvements to show"),
    ):
        """Diff two benchmark JSON files and show speedup deltas."""

        def _load(path: Path) -> dict:
            with open(path) as f:
                return json.load(f)

        base = _load(baseline)
        cand = _load(candidate)

        def _flatten(blob: dict) -> dict:
            flat = {}
            for chapter in blob.get("results", []):
                chap = chapter.get("chapter", "unknown")
                for bench in chapter.get("benchmarks", []):
                    key = f"{chap}:{bench.get('example', bench.get('name', 'unknown'))}"
                    flat[key] = bench
            return flat

        bflat = _flatten(base)
        cflat = _flatten(cand)
        deltas = []
        for key, c in cflat.items():
            b = bflat.get(key)
            if not b:
                continue
            delta = (c.get("best_speedup", 0) or 0) - (b.get("best_speedup", 0) or 0)
            deltas.append(
                (key, delta, b.get("best_speedup", 0) or 0, c.get("best_speedup", 0) or 0)
            )

        deltas.sort(key=lambda x: x[1])
        regressions = [d for d in deltas if d[1] < 0][:top]
        improvements = sorted([d for d in deltas if d[1] > 0], key=lambda x: -x[1])[:top]

        typer.echo("\n🚨 Regressions")
        for name, delta, b, c in regressions:
            typer.echo(f"  {name}: {b:.2f}x → {c:.2f}x (Δ {delta:.2f})")

        typer.echo("\n✅ Improvements")
        for name, delta, b, c in improvements:
            typer.echo(f"  {name}: {b:.2f}x → {c:.2f}x (Δ +{delta:.2f})")

        typer.echo("\nℹ️ Use --top to adjust output; full data available in JSON files.")

    @app.command("tui")
    def tui(
        simple: bool = Option(
            False, "--simple", "-s", help="Use simple menu instead of curses TUI"
        ),
        data_file: Optional[Path] = Option(
            None, "--data-file", "-d", help="Path to benchmark_test_results.json"
        ),
    ):
        """Interactive Terminal UI for benchmark analysis."""
        if simple:
            _run_basic_menu(data_file)
            return

        try:
            from cli.tui import run_tui

            run_tui(data_file)
        except KeyboardInterrupt:
            typer.echo("\nExiting...")
        except ImportError as e:
            typer.echo(f"TUI requires curses (standard on Unix): {e}")
            typer.echo("Falling back to basic menu...")
            _run_basic_menu(data_file)
        except Exception as e:
            typer.echo(f"TUI error: {e}")
            typer.echo("Falling back to basic menu...")
            _run_basic_menu(data_file)

    def _run_basic_menu(data_file: Optional[Path] = None):
        """Simple menu-based interface when rich TUI isn't available."""
        analyzer = _get_analyzer(data_file)

        while True:
            typer.echo("\n" + "=" * 50)
            typer.echo("📊 BENCHMARK ANALYSIS MENU")
            typer.echo("=" * 50)
            typer.echo("  1. Leaderboards (speed vs memory)")
            typer.echo("  2. Pareto Frontier")
            typer.echo("  3. What-If Solver")
            typer.echo("  4. Optimization Stacking")
            typer.echo("  5. Power Efficiency")
            typer.echo("  6. Cost Analysis")
            typer.echo("  7. Scaling Analysis")
            typer.echo("  8. Trade-off Chart (ASCII)")
            typer.echo("  q. Quit")
            typer.echo("-" * 50)

            choice = input("Select option: ").strip().lower()

            if choice == "q":
                break
            elif choice == "1":
                data = analyzer.get_categorized_leaderboards()
                _print_leaderboards(data)
            elif choice == "2":
                data = analyzer.get_pareto_frontier()
                _print_pareto(data)
            elif choice == "3":
                vram = input("Max VRAM (GB, or Enter to skip): ").strip()
                latency = input("Max latency (ms, or Enter to skip): ").strip()
                params = {}
                if vram:
                    params["vram"] = [vram]
                if latency:
                    params["latency"] = [latency]
                data = analyzer.get_whatif_recommendations(params)
                _print_whatif(data)
            elif choice == "4":
                data = analyzer.get_optimization_stacking()
                _print_stacking(data)
            elif choice == "5":
                data = analyzer.get_power_efficiency()
                _print_power(data)
            elif choice == "6":
                data = analyzer.get_cost_analysis()
                _print_cost(data)
            elif choice == "7":
                data = analyzer.get_scaling_analysis()
                _print_scaling(data)
            elif choice == "8":
                data = analyzer.get_pareto_frontier()
                render_ascii_scatter_chart(data)

    def _run_interactive_tui():
        """Rich interactive TUI using curses or similar."""
        # For now, fall back to basic menu
        _run_basic_menu()

    def _print_leaderboards(data):
        boards = data.get("leaderboards", {})
        for name, board in boards.items():
            typer.echo(f"\n{board.get('title', name)}:")
            for e in board.get("entries", [])[:5]:
                typer.echo(f"  #{e['rank']} {e['name']}: {e['primary_metric']}")

    def _print_pareto(data):
        typer.echo(
            f"\n⭐ Pareto Optimal: {data.get('pareto_count', 0)} / {data.get('total_count', 0)}"
        )
        for p in data.get("pareto_frontier", [])[:5]:
            typer.echo(f"  ⭐ {p['name']}: {p['speedup']:.2f}x, {p['memory_savings']:.0f}% mem")

    def _print_whatif(data):
        typer.echo(f"\nMatching: {data.get('matching_count', 0)} benchmarks")
        for r in data.get("recommendations", [])[:5]:
            typer.echo(f"  • {r['name']}: {r['speedup']:.2f}x")

    def _print_stacking(data):
        typer.echo("\n✅ Compatible:")
        for c in data.get("compatible_combinations", [])[:3]:
            typer.echo(f"  {c['opt1']} + {c['opt2']}: {c['benefit']}")

    def _print_power(data):
        typer.echo(f"\nBenchmarks with power data: {data.get('total_benchmarks_with_power', 0)}")
        for e in data.get("efficiency_rankings", [])[:5]:
            typer.echo(f"  {e['name']}: {e['ops_per_watt']:.4f} ops/W")

    def _print_cost(data):
        typer.echo(f"\nAssuming ${data.get('hourly_rate', 0):.2f}/hr")
        for c in data.get("cost_rankings", [])[:5]:
            typer.echo(f"  {c['name']}: {c['savings_pct']:.0f}% savings")

    def _print_scaling(data):
        for r in data.get("scaling_recommendations", [])[:3]:
            typer.echo(f"\n{r['factor']}: {r['recommendation']}")

    @app.command("audit")
    def audit(
        chapter: Optional[str] = Option(
            None, "--chapter", "-c", help="Specific chapter to audit (e.g., ch10)"
        ),
        lab: Optional[str] = Option(
            None, "--lab", "-l", help="Specific lab to audit (e.g., decode_optimization)"
        ),
        all_targets: bool = Option(False, "--all", "-a", help="Audit all chapters and labs"),
        verbose: bool = Option(False, "--verbose", "-v", help="Show detailed output"),
        json_output: bool = Option(False, "--json", help="Output as JSON"),
    ):
        """Audit benchmark verification compliance.

        Uses Python introspection to correctly detect inherited methods
        (unlike grep-based approaches). This gives accurate coverage stats.

        Example:
            aisp bench audit --all          # Audit everything
            aisp bench audit -c ch10        # Audit specific chapter
            aisp bench audit -l decode_optimization  # Audit specific lab
        """
        import json as json_lib
        from core.scripts.audit_verification_compliance import (
            audit_directory,
            print_summary,
            load_benchmark_class,
            check_compliance,
        )

        code_dir = Path(__file__).parent.parent.parent

        def _count_audit_results(results: dict[str, dict[str, object]]) -> tuple[int, int, int]:
            compliant = sum(1 for result in results.values() if result.get("status") == "compliant")
            needs_work = sum(
                1 for result in results.values() if result.get("status") == "needs_work"
            )
            errors = sum(1 for result in results.values() if result.get("status") == "error")
            return compliant, needs_work, errors

        total_compliant = 0
        total_needs_work = 0
        total_errors = 0
        all_results = {}

        # Audit chapters
        if chapter:
            chapters = [chapter]
        elif all_targets or not lab:
            chapters = [f"ch{i:02d}" for i in range(1, 21)]
        else:
            chapters = []

        for ch in chapters:
            chapter_dir = code_dir / ch
            if not chapter_dir.exists():
                continue

            results = audit_directory(chapter_dir)
            if results:
                c, n, e = _count_audit_results(results)
                total_compliant += c
                total_needs_work += n
                total_errors += e
                if json_output:
                    all_results[ch] = results
                else:
                    print_summary(results, f"{ch.upper()}")

        # Audit labs
        if lab:
            labs = [lab]
        elif all_targets or not chapter:
            labs_dir = code_dir / "labs"
            if labs_dir.exists():
                labs = [d.name for d in labs_dir.iterdir() if d.is_dir()]
            else:
                labs = []
        else:
            labs = []

        for lb in sorted(labs):
            lab_dir = code_dir / "labs" / lb
            if not lab_dir.exists():
                continue

            results = audit_directory(lab_dir)
            if results:
                c, n, e = _count_audit_results(results)
                total_compliant += c
                total_needs_work += n
                total_errors += e
                if json_output:
                    all_results[f"lab:{lb}"] = results
                else:
                    print_summary(results, f"LAB: {lb}")

        # Output
        if json_output:
            typer.echo(
                json_lib.dumps(
                    {
                        "results": all_results,
                        "summary": {
                            "compliant": total_compliant,
                            "needs_work": total_needs_work,
                            "errors": total_errors,
                            "total": total_compliant + total_needs_work + total_errors,
                            "coverage_pct": round(
                                100 * total_compliant / max(1, total_compliant + total_needs_work),
                                1,
                            ),
                        },
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(f"\n{'=' * 60}")
            typer.echo("GRAND TOTAL")
            typer.echo(f"{'=' * 60}")
            typer.echo(f"✅ Compliant: {total_compliant}")
            typer.echo(f"⚠️  Needs work: {total_needs_work}")
            typer.echo(f"❌ Errors: {total_errors}")
            typer.echo(f"Total: {total_compliant + total_needs_work + total_errors}")

            coverage = (total_compliant / max(1, total_compliant + total_needs_work)) * 100
            typer.echo(f"\n📊 Coverage: {coverage:.1f}%")


def main():
    """Entry point for CLI."""
    if not TYPER_AVAILABLE:
        print("ERROR: typer is required for CLI. Install with: pip install typer")
        sys.exit(1)

    if app is None:
        print("ERROR: CLI not available")
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
