"""Standard template and utilities for chapter compare.py modules.

All chapters should use these functions to ensure consistency:
- discover_benchmarks() - Find baseline/optimized pairs
- load_benchmark() - Load BaseBenchmark instances from files
- create_profile_template() - Standard profile() function structure

All compare.py modules must:
1. Import BenchmarkHarness, BaseBenchmark, BenchmarkMode, BenchmarkConfig
2. Use discover_benchmarks() to find pairs
3. Use load_benchmark() to instantiate benchmarks
4. Run via harness.benchmark(benchmark_instance)
5. Return standardized format: {"metrics": {...}}
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
import threading
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable, cast

from core.utils.warning_filters import suppress_known_cuda_capability_warnings

with suppress_known_cuda_capability_warnings(context="chapter_compare_template torch import"):
    import torch

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkMode,
    BenchmarkConfig,
)
from core.harness.validity_profile import (
    VALIDITY_PROFILE_CHOICES,
    VALIDITY_PROFILE_HELP_TEXT,
)
from core.discovery import discover_benchmarks
from core.utils.python_entrypoints import temporary_sys_path

# Import logger
try:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

__all__ = ['discover_benchmarks', 'load_benchmark', 'create_standard_metrics', 'profile_template', 'get_last_load_error']

_LAST_LOAD_ERROR: Optional[str] = None


@dataclass(frozen=True)
class _CompareCliArgs:
    validity_profile: Optional[str] = None
    allow_foreign_gpu_processes: bool = False
    profile: str = "none"
    examples: Optional[List[str]] = None


def load_benchmark(module_path: Path, timeout_seconds: int = 120) -> Optional[BaseBenchmark]:
    """Load benchmark from module by calling get_benchmark() function.
    
    Uses threading timeout to prevent hangs during module import or get_benchmark() calls.
    
    Args:
        module_path: Path to Python file with BaseBenchmark implementation
        timeout_seconds: Maximum time to wait for module load (default: 15 seconds)
        
    Returns:
        BaseBenchmark instance or None if loading fails or times out
    """
    global _LAST_LOAD_ERROR
    result: Dict[str, Any] = {"benchmark": None, "error": None, "done": False}
    
    def load_internal():
        try:
            # Add repo root (directory containing the benchmark package) only while loading.
            repo_root = module_path.resolve()
            while repo_root.parent != repo_root:
                if (repo_root / "core" / "common").exists():
                    break
                repo_root = repo_root.parent
            else:
                repo_root = module_path.parent.parent
            with temporary_sys_path(repo_root):
                try:
                    rel_path = module_path.resolve().relative_to(repo_root).with_suffix("")
                    module_name = ".".join(rel_path.parts)
                except ValueError:
                    module_name = module_path.stem

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    result["error"] = "Failed to create module spec"
                    return

                module = importlib.util.module_from_spec(spec)

                # Register module in sys.modules BEFORE exec_module() - this is critical!
                # Dataclasses and other introspection tools need the module to be registered
                # during class definition, otherwise sys.modules.get(cls.__module__) returns None
                # and causes "'NoneType' object has no attribute '__dict__'" errors.
                simple_name = module_path.stem
                if module_name and module_name not in sys.modules:
                    sys.modules[module_name] = module
                if simple_name not in sys.modules:
                    sys.modules[simple_name] = module

                spec.loader.exec_module(module)

                if hasattr(module, 'get_benchmark'):
                    benchmark = module.get_benchmark()
                    if (
                        benchmark is not None
                        and benchmark.__class__.__module__ != module.__name__
                        and not getattr(benchmark, "_module_file_override", None)
                    ):
                        attach_benchmark_metadata(benchmark, str(module_path.resolve()))
                    result["benchmark"] = benchmark
                else:
                    result["error"] = "Module does not have get_benchmark() function"
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True
    
    # Run load in a thread with timeout to prevent hangs
    thread = threading.Thread(target=load_internal, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result["done"]:
        logger.warning(f"Failed to load {module_path.name}: TIMEOUT (exceeded {timeout_seconds}s)")
        _LAST_LOAD_ERROR = f"TIMEOUT (exceeded {timeout_seconds}s)"
        return None
    
    if result["error"]:
        _LAST_LOAD_ERROR = result["error"]
        logger.warning(f"Failed to load {module_path.name}: {result['error']}")
        return None
    
    _LAST_LOAD_ERROR = None
    return cast(Optional[BaseBenchmark], result["benchmark"])


def get_last_load_error() -> Optional[str]:
    return _LAST_LOAD_ERROR


def _is_compare_cli_invocation(argv0: Optional[str] = None) -> bool:
    if argv0 is None:
        argv0 = sys.argv[0] if sys.argv else ""
    return Path(str(argv0)).name == "compare.py"


def _parse_compare_cli_args(
    argv: Optional[List[str]] = None,
    *,
    argv0: Optional[str] = None,
) -> _CompareCliArgs:
    """Parse compare.py CLI args.

    Parsing is only applied when the active entrypoint is compare.py. This keeps
    programmatic imports (tests/harness flows) isolated from CLI-only validation.
    """
    if not _is_compare_cli_invocation(argv0=argv0):
        return _CompareCliArgs()

    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0] if argv0 is None else argv0).name,
        description="Run chapter baseline/optimized comparison using the benchmark harness.",
    )
    parser.add_argument(
        "--validity-profile",
        choices=list(VALIDITY_PROFILE_CHOICES),
        default=None,
        help=VALIDITY_PROFILE_HELP_TEXT,
    )
    parser.add_argument(
        "--allow-foreign-gpu-processes",
        action="store_true",
        help=(
            "Warn instead of fail when unrelated CUDA compute processes are active on the "
            "benchmark GPU. Use only on shared hosts where strict isolation is impossible."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["none", "minimal", "deep_dive", "roofline"],
        default="none",
        help="Profiling preset for compare.py runs (default: none).",
    )
    parser.add_argument(
        "--examples",
        action="append",
        default=None,
        help=(
            "Optional example name filter (repeatable, supports comma-separated values). "
            "Only matching baseline/optimized pairs are executed."
        ),
    )
    parsed = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    raw_examples = cast(Optional[List[str]], parsed.examples)
    normalized_examples: List[str] = []
    for raw in raw_examples or []:
        for token in str(raw).split(","):
            example = token.strip()
            if example:
                normalized_examples.append(example)
    return _CompareCliArgs(
        validity_profile=cast(Optional[str], parsed.validity_profile),
        allow_foreign_gpu_processes=bool(parsed.allow_foreign_gpu_processes),
        profile=cast(str, parsed.profile),
        examples=normalized_examples or None,
    )


def create_standard_metrics(
    chapter: str,
    all_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Create standardized metrics dictionary from collected results.
    
    Ensures all chapters return consistent metrics format.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch01', 'ch16')
        all_metrics: Dictionary of collected metrics (will be modified in place)
        default_tokens_per_s: Default throughput if not calculated
        default_requests_per_s: Default request rate if not calculated
        default_goodput: Default efficiency metric if not calculated
        default_latency_s: Default latency if not calculated
        
    Returns:
        Standardized metrics dictionary
    """
    # Ensure chapter is set
    all_metrics['chapter'] = chapter
    
    # Calculate speedups from collected metrics
    speedups = [
        v for k, v in all_metrics.items() 
        if k.endswith('_speedup') and isinstance(v, (int, float)) and v > 0
    ]
    
    if speedups:
        all_metrics['speedup'] = max(speedups)
        all_metrics['average_speedup'] = sum(speedups) / len(speedups)
    else:
        # Default if no speedups found
        all_metrics['speedup'] = 1.0
        all_metrics['average_speedup'] = 1.0
    
    return all_metrics


def profile_template(
    chapter: str,
    chapter_dir: Path,
    harness_config: Optional[BenchmarkConfig] = None,
    custom_metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Template profile() function for chapter compare.py modules.
    
    Standard implementation that all chapters should use or adapt.
    
    Args:
        chapter: Chapter identifier (e.g., 'ch01', 'ch16')
        chapter_dir: Path to chapter directory
        harness_config: Optional BenchmarkConfig override (default: iterations=20, warmup=5)
        custom_metrics_callback: Optional function to add custom metrics: f(all_metrics) -> None
        
    Returns:
        Standardized format: {"metrics": {...}}
    """
    logger.info("=" * 70)
    logger.info(f"Chapter {chapter.upper()}: Comparing Implementations")
    logger.info("=" * 70)
    
    chapter_tokens: List[float] = []
    chapter_requests: List[float] = []
    chapter_latencies_ms: List[float] = []
    chapter_goodputs: List[float] = []

    def record_throughput(prefix: str, throughput_obj: Optional[Any], metrics: Dict[str, Any]) -> None:
        if not throughput_obj:
            return
        tokens = getattr(throughput_obj, "tokens_per_s", None)
        requests = getattr(throughput_obj, "requests_per_s", None)
        samples = getattr(throughput_obj, "samples_per_s", None)
        bytes_per_s = getattr(throughput_obj, "bytes_per_s", None)
        custom_unit = getattr(throughput_obj, "custom_unit_per_s", None)
        custom_unit_name = getattr(throughput_obj, "custom_unit_name", None)
        latency_ms = getattr(throughput_obj, "latency_ms", None)
        goodput = getattr(throughput_obj, "goodput", None)

        if tokens is not None:
            metrics[f"{prefix}_tokens_per_s"] = tokens
            chapter_tokens.append(tokens)
        if requests is not None:
            metrics[f"{prefix}_requests_per_s"] = requests
            chapter_requests.append(requests)
        if samples is not None:
            metrics[f"{prefix}_samples_per_s"] = samples
        if bytes_per_s is not None:
            metrics[f"{prefix}_bytes_per_s"] = bytes_per_s
        if custom_unit is not None:
            key = f"{prefix}_custom_unit_per_s"
            metrics[key] = custom_unit
            if custom_unit_name:
                metrics[f"{prefix}_custom_unit_name"] = custom_unit_name
        if latency_ms is not None:
            metrics[f"{prefix}_latency_ms"] = latency_ms
            metrics[f"{prefix}_latency_s"] = latency_ms / 1000.0
            chapter_latencies_ms.append(latency_ms)
        if goodput is not None:
            metrics[f"{prefix}_goodput"] = goodput
            chapter_goodputs.append(goodput)

    def compact_ms(value: Optional[float]) -> str:
        if value is None:
            return "-"
        abs_val = abs(value)
        if abs_val >= 100:
            text = f"{value:.1f}"
        elif abs_val >= 10:
            text = f"{value:.2f}"
        else:
            text = f"{value:.3f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def format_rate_short(throughput_obj: Optional[Any]) -> str:
        if not throughput_obj:
            return "-"
        metrics = [
            ("requests_per_s", "req/s"),
            ("tokens_per_s", "tok/s"),
            ("samples_per_s", "samples/s"),
            ("bytes_per_s", "B/s"),
        ]
        for attr, label in metrics:
            value = getattr(throughput_obj, attr, None)
            if value:
                abs_val = abs(value)
                if abs_val >= 1000:
                    number = f"{value:,.0f}"
                elif abs_val >= 100:
                    number = f"{value:.1f}"
                else:
                    number = f"{value:.2f}"
                return f"{number} {label}"
        latency_ms = getattr(throughput_obj, "latency_ms", None)
        if latency_ms:
            return f"{compact_ms(latency_ms)} ms/iter"
        return "-"

    def format_memory_short(memory_obj: Optional[Any]) -> str:
        if not memory_obj:
            return "-"
        peak_mb = getattr(memory_obj, "peak_mb", None)
        if peak_mb is None:
            return "-"
        if peak_mb >= 1000:
            text = f"{peak_mb:.0f}"
        elif peak_mb >= 100:
            text = f"{peak_mb:.1f}"
        else:
            text = f"{peak_mb:.2f}"
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def build_perf_row(
        label: str,
        mean_ms: Optional[float],
        timing_obj: Optional[Any],
        throughput_obj: Optional[Any],
        speedup_value: float,
        memory_obj: Optional[Any],
    ) -> Dict[str, str]:
        percentiles = getattr(timing_obj, "percentiles", None) if timing_obj else None
        p99 = None
        if percentiles and isinstance(percentiles, dict):
            p99 = percentiles.get(99.0)
        median = getattr(timing_obj, "median_ms", None) if timing_obj else None
        mean_val = mean_ms if timing_obj else None
        speedup_str = "-" if speedup_value <= 0 else f"{speedup_value:.2f}x"
        return {
            "Variant": label,
            "Mean(ms)": compact_ms(mean_val),
            "Median": compact_ms(median),
            "p99": compact_ms(p99),
            "Rate": format_rate_short(throughput_obj),
            "Speedup": speedup_str,
            "Mem(MB)": format_memory_short(memory_obj),
        }

    def render_example_table(example_name: str, rows: List[Dict[str, str]]) -> None:
        if not rows:
            return
        headers = ["Variant", "Mean(ms)", "Median", "p99", "Rate", "Speedup", "Mem(MB)"]
        widths = {header: len(header) for header in headers}
        for row in rows:
            for header in headers:
                widths[header] = max(widths[header], len(row.get(header, "")))
        header_line = " | ".join(header.ljust(widths[header]) for header in headers)
        divider = "-+-".join("-" * widths[header] for header in headers)
        lines = [header_line, divider]
        for row in rows:
            line = " | ".join(row.get(header, "").ljust(widths[header]) for header in headers)
            lines.append(line)
        table = "\n".join(lines)
        logger.info(f"\nExample: {example_name}\n{table}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - skipping")
        return {
            "metrics": {
                'chapter': chapter,
                'cuda_unavailable': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    # Initialize CUDA context early to prevent cuBLAS warnings
    # This ensures the context is ready before any operations
    try:
        torch.cuda.init()
        # Create a dummy tensor to force context initialization
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"CUDA context initialization warning (non-fatal): {e}")
    
    cli_args = _parse_compare_cli_args()
    pairs = discover_benchmarks(chapter_dir)

    if cli_args.examples:
        include_examples = set(cli_args.examples)
        filtered_pairs = [pair for pair in pairs if pair[2] in include_examples]
        missing_examples = sorted(include_examples.difference({pair[2] for pair in filtered_pairs}))
        if missing_examples:
            logger.warning(
                "compare.py --examples ignored unknown example(s): %s",
                ", ".join(missing_examples),
            )
        pairs = filtered_pairs
    
    if not pairs:
        logger.warning("No baseline/optimized pairs found")
        logger.info("Tip: Create baseline_*.py and optimized_*.py files")
        logger.info("    Each file must implement BaseBenchmark with get_benchmark() function")
        return {
            "metrics": {
                'chapter': chapter,
                'no_pairs_found': True,
                'speedup': 1.0,
                'latency_s': 0.0,
                'tokens_per_s': 0.0,
                'requests_per_s': 0.0,
                'goodput': 0.0,
            }
        }
    
    logger.info(f"Found {len(pairs)} example(s) with optimization(s):\n")
    
    # Create harness with default or custom config
    # Enable memory tracking by default to capture all metrics
    # Increase timeouts to prevent premature failures
    from dataclasses import replace
    if harness_config is None:
        config = BenchmarkConfig(
            iterations=20, 
            warmup=5, 
            enable_memory_tracking=True,
            measurement_timeout_seconds=30,  # Increased from 15 to prevent timeouts
            setup_timeout_seconds=60,  # Increased for slow setups
        )
    else:
        # Enable memory tracking by default unless explicitly disabled
        # Increase timeouts if not explicitly set
        if harness_config.enable_memory_tracking is False:
            # Respect explicit False
            config = harness_config
        else:
            # Default to True for comprehensive metrics
            config = replace(harness_config, enable_memory_tracking=True)
        
        # Ensure reasonable timeouts
        if config.measurement_timeout_seconds < 30:
            config.measurement_timeout_seconds = 30
        if config.setup_timeout_seconds is not None and config.setup_timeout_seconds < 60:
            config.setup_timeout_seconds = 60

    # Allow chapter compare entrypoints to accept the same validity profile knob.
    # Rebuild the config so derived fields (e.g., allow_virtualization) remain consistent.
    cli_validity_profile = cli_args.validity_profile
    if cli_validity_profile is not None and cli_validity_profile != config.validity_profile:
        logger.info(
            "Applying compare.py CLI override: --validity-profile %s",
            cli_validity_profile,
        )
        config = replace(config, validity_profile=cli_validity_profile)
    if cli_args.allow_foreign_gpu_processes and not config.allow_foreign_gpu_processes:
        logger.info("Applying compare.py CLI override: --allow-foreign-gpu-processes")
        config = replace(config, allow_foreign_gpu_processes=True)
    if cli_args.profile != "none":
        logger.info("Applying compare.py CLI profiling preset: --profile %s", cli_args.profile)
        config = replace(
            config,
            enable_profiling=True,
            enable_nsys=True,
            enable_ncu=True,
            profile_mode=cli_args.profile,
            profile_type=cli_args.profile,
        )
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    
    all_metrics: Dict[str, Any] = {
        'chapter': chapter,
    }
    
    # Collect all results for summary
    summary_data: List[Dict[str, Any]] = []
    
    for baseline_path, optimized_paths, example_name in pairs:
        example_rows: List[Dict[str, str]] = []
        baseline_benchmark = load_benchmark(baseline_path)
        if baseline_benchmark is None:
            logger.error(f"Baseline failed to load (missing get_benchmark() function?): {baseline_path.name}")
            continue
        
        try:
            baseline_result = harness.benchmark(baseline_benchmark)
            baseline_timing = baseline_result.timing
            baseline_time = baseline_timing.mean_ms if baseline_timing else 0.0
            baseline_throughput = getattr(baseline_result, "throughput", None)
            record_throughput(f"{example_name}_baseline", baseline_throughput, all_metrics)
            baseline_mean_for_row = baseline_time if baseline_timing else None
            example_rows.append(
                build_perf_row(
                    baseline_path.name,
                    baseline_mean_for_row,
                    baseline_timing,
                    baseline_throughput,
                    1.0,
                    baseline_result.memory,
                )
            )
        except Exception as e:
            error_msg = str(e)
            # Check for skip warnings
            if "SKIPPED" in error_msg or "SKIP" in error_msg.upper() or "WARNING: SKIPPED" in error_msg:
                logger.warning(f"{error_msg}")
            else:
                logger.error(f"Baseline failed to run: {error_msg}")
            continue
        
        best_speedup = 1.0
        best_optimized = None
        optimized_results = []
        
        for optimized_path in optimized_paths:
            opt_name = optimized_path.name
            # Extract technique name: optimized_moe_sparse.py -> sparse
            technique = opt_name.replace(f'optimized_{example_name}_', '').replace('.py', '')
            if technique == opt_name.replace('optimized_', '').replace('.py', ''):
                technique = 'default'
            
            optimized_benchmark = load_benchmark(optimized_path)
            if optimized_benchmark is None:
                logger.error(f"Failed to load: {opt_name}")
                continue
            
            try:
                optimized_result = harness.benchmark(optimized_benchmark)
                optimized_timing = optimized_result.timing
                optimized_time = optimized_timing.mean_ms if optimized_timing else 0.0
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                optimized_throughput = getattr(optimized_result, "throughput", None)
                record_throughput(f"{example_name}_{technique}", optimized_throughput, all_metrics)
                
                optimized_results.append({
                    'name': opt_name,
                    'time': optimized_time,
                    'speedup': speedup,
                    'technique': technique
                })
                
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_optimized = optimized_path
                
                # Store per-technique metrics
                all_metrics[f"{example_name}_{technique}_baseline_time"] = baseline_time
                all_metrics[f"{example_name}_{technique}_optimized_time"] = optimized_time
                all_metrics[f"{example_name}_{technique}_speedup"] = speedup

                optimized_mean_for_row = optimized_time if optimized_timing else None
                example_rows.append(
                    build_perf_row(
                        opt_name,
                        optimized_mean_for_row,
                        optimized_timing,
                        optimized_throughput,
                        speedup,
                        optimized_result.memory,
                    )
                )
                
            except Exception as e:
                error_msg = str(e)
                # Check for skip warnings
                if "SKIPPED" in error_msg or "SKIP" in error_msg.upper():
                    logger.warning(f"{error_msg}")
                else:
                    logger.error(f"Failed: {error_msg}")
                continue
        
        render_example_table(example_name, example_rows)

        # Summary for this example
        if optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': best_speedup,
                'best_name': best_optimized.name if best_optimized else None,
                'num_optimizations': len(optimized_results)
            })
            all_metrics[f"{example_name}_best_speedup"] = best_speedup
        
        if not optimized_results:
            summary_data.append({
                'example': example_name,
                'baseline': baseline_time,
                'best_speedup': 1.0,
                'best_name': None,
                'num_optimizations': 0
            })
    
    if chapter_requests:
        all_metrics['requests_per_s'] = max(chapter_requests)
    if chapter_tokens:
        all_metrics['tokens_per_s'] = max(chapter_tokens)
    if chapter_latencies_ms:
        best_latency = min(chapter_latencies_ms)
        all_metrics['latency_ms'] = best_latency
        all_metrics['latency_s'] = best_latency / 1000.0
    if chapter_goodputs:
        all_metrics['goodput'] = max(chapter_goodputs)

    # Apply custom metrics callback if provided
    if custom_metrics_callback is not None:
        custom_metrics_callback(all_metrics)
    
    # Standardize metrics format
    all_metrics = create_standard_metrics(chapter, all_metrics)
    
    # Print snazzy summary
    if summary_data:
        print("\n" + "=" * 80)
        print("📊 SUMMARY - Performance Improvements")
        print("=" * 80)
        
        # Sort by speedup (best first)
        def get_speedup(x: Dict[str, Any]) -> float:
            val = x.get('best_speedup', 1.0)
            return float(val) if isinstance(val, (int, float)) else 1.0
        
        summary_data.sort(key=get_speedup, reverse=True)
        
        for idx, item in enumerate(summary_data, 1):
            example_name = str(item.get('example', ''))
            baseline_val = item.get('baseline', 0.0)
            baseline = float(baseline_val) if isinstance(baseline_val, (int, float)) else 0.0
            speedup_val = item.get('best_speedup', 1.0)
            speedup = float(speedup_val) if isinstance(speedup_val, (int, float)) else 1.0
            best_name_val = item.get('best_name')
            best_name = str(best_name_val) if best_name_val is not None else None
            num_opts_val = item.get('num_optimizations', 0)
            num_opts = int(num_opts_val) if isinstance(num_opts_val, (int, float)) else 0
            
            # Status emoji
            if speedup >= 2.0:
                status = "🔥"
            elif speedup >= 1.5:
                status = "✨"
            elif speedup >= 1.2:
                status = "👍"
            elif speedup >= 1.0:
                status = "✅"
            else:
                status = "⚠️"
            
            logger.info(f"\n  {idx}. {example_name} {status}")
            logger.info(f"     Baseline: {baseline:.2f} ms")
            
            if best_name and speedup > 1.0:
                improvement_pct = (1 - 1/speedup) * 100
                logger.info(f"     🏆 Best: {best_name}")
                logger.info(f"     📈 Improvement: {speedup:.2f}x faster ({improvement_pct:.1f}% reduction)")
                
                # ASCII bar showing improvement
                bar_width = 50
                filled = int(min(bar_width, (speedup - 1.0) / 5.0 * bar_width))
                bar = "█" * filled + "░" * (bar_width - filled)
                logger.info(f"     {bar}")
            elif num_opts == 0:
                logger.warning(f"     ⚠️  No successful optimizations")
            else:
                logger.warning(f"     ⚠️  Optimization did not improve performance")
            
            logger.info(f"     📦 {num_opts} optimization(s) tested")
        
        # Overall stats
        def get_speedup_for_filter(s: Dict[str, Any]) -> bool:
            val = s.get('best_speedup', 1.0)
            return float(val) > 1.0 if isinstance(val, (int, float)) else False
        
        def get_speedup_for_sum(s: Dict[str, Any]) -> float:
            val = s.get('best_speedup', 1.0)
            return float(val) if isinstance(val, (int, float)) else 1.0
        
        successful = [s for s in summary_data if get_speedup_for_filter(s)]
        avg_speedup = sum(get_speedup_for_sum(s) for s in successful) / len(successful) if successful else 0
        best_overall = max(summary_data, key=get_speedup_for_sum)
        
        logger.info("\n" + "-" * 80)
        logger.info(f"📊 Overall Stats:")
        logger.info(f"   • Examples tested: {len(summary_data)}")
        logger.info(f"   • Successful optimizations: {len(successful)}")
        if successful:
            logger.info(f"   • Average speedup: {avg_speedup:.2f}x")
            logger.info(f"   • Best improvement: {best_overall['example']} ({best_overall['best_speedup']:.2f}x)")
        logger.info("=" * 80)
    
    logger.info("")
    
    return {
        "metrics": all_metrics
    }


__all__ = [
    'discover_benchmarks',
    'load_benchmark',
    'create_standard_metrics',
    'profile_template',
]
