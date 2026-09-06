"""Utilities for comparing benchmark results and detecting regressions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable

logger = logging.getLogger(__name__)
# GPU benchmarks typically have 10-30% variance between runs.
DEFAULT_THRESHOLD_PCT = 25.0

_NCU_RANGE_DURATION_METRIC_PATHS = frozenset(
    {
        "range_time_ms",
        "ncu_range_time_ms",
        "profiler_metrics.ncu.range_time_ms",
        "profiler_metrics.ncu.ncu_range_time_ms",
        "profiler_metrics.ncu.raw.range_time_ms",
        "profiler_metrics.ncu.raw.ncu_range_time_ms",
    }
)

_chapter_metrics_loader: Optional[Callable[[str], Dict[str, Dict[str, Any]]]] = None
_target_metadata_loader: Optional[Callable[[], Dict[str, Any]]] = None
_target_metadata_warning_logged = False


def _is_ncu_range_duration_metric(metric_name: str) -> bool:
    """Return whether a metric is descriptive NCU selected-range duration."""
    return metric_name in _NCU_RANGE_DURATION_METRIC_PATHS

try:
    from core.benchmark.performance_targets import (
        get_chapter_metrics as _chapter_metrics_loader,
        get_targets_metadata as _target_metadata_loader,
    )
except (ImportError, AttributeError, TypeError):
    pass


def get_chapter_metrics(chapter: str):
    """Load chapter metrics from performance_targets, if available."""
    if _chapter_metrics_loader is None:
        raise ImportError("performance_targets.get_chapter_metrics is unavailable")
    return _chapter_metrics_loader(chapter)


class MetricDirection(Enum):
    """Direction for metric comparison - whether higher or lower is better."""
    LOWER_IS_BETTER = "lower"  # e.g., latency, time, memory usage
    HIGHER_IS_BETTER = "higher"  # e.g., throughput, bandwidth, efficiency


@dataclass
class MetricComparison:
    """Result of comparing a single metric between baseline and optimized."""
    metric_name: str
    display_name: str
    baseline_value: float
    optimized_value: float
    ratio: float  # optimized/baseline for higher_is_better, baseline/optimized for lower_is_better
    direction: MetricDirection
    unit: str = ""
    regression: bool = False
    regression_pct: Optional[float] = None
    improvement_pct: Optional[float] = None
    significant_change: bool = False
    
    def __post_init__(self):
        """Calculate ratio based on direction."""
        baseline = self.baseline_value if self.baseline_value is not None else 0.0
        optimized = self.optimized_value if self.optimized_value is not None else 0.0
        
        if self.direction == MetricDirection.LOWER_IS_BETTER:
            # For lower-is-better: ratio = baseline/optimized (like speedup)
            if optimized <= 0:
                self.ratio = float("inf") if baseline > 0 else 1.0
            elif baseline <= 0:
                self.ratio = float("inf")
            else:
                self.ratio = baseline / optimized
        else:
            # For higher-is-better: ratio = optimized/baseline
            if baseline <= 0:
                self.ratio = float("inf") if optimized > 0 else 1.0
            else:
                self.ratio = optimized / baseline


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs (legacy - timing only)."""
    baseline_mean_ms: float
    optimized_mean_ms: float
    speedup: float
    regression: bool
    regression_pct: Optional[float] = None
    improvement_pct: Optional[float] = None


@dataclass
class ComprehensiveComparison:
    """Comprehensive comparison of all metrics between baseline and optimized."""
    timing_comparison: ComparisonResult
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    
    def get_all_comparisons(self) -> List[MetricComparison]:
        """Get all metric comparisons.
        
        Returns only metrics from metric_comparisons list. Timing comparison
        is available via the timing_comparison attribute for backward compatibility.
        
        Note: Timing metrics appear in metric_comparisons when include_timing=True
        in compare_all_metrics(). This method no longer synthesizes a timing metric.
        """
        return self.metric_comparisons.copy()


def compare_results(
    baseline_result,
    optimized_result,
    regression_threshold_pct: Optional[float] = None,
    improvement_threshold_pct: Optional[float] = None
) -> ComparisonResult:
    """Compare baseline and optimized benchmark results.
    
    Args:
        baseline_result: BenchmarkResult (Pydantic) from baseline run
        optimized_result: BenchmarkResult (Pydantic) from optimized run
        regression_threshold_pct: Percentage degradation to consider a regression (None -> default 25%)
        improvement_threshold_pct: Percentage improvement to consider significant (None -> default 25%)
        
    Returns:
        ComparisonResult with speedup and regression detection
    """
    regression_threshold = (
        regression_threshold_pct if regression_threshold_pct is not None else DEFAULT_THRESHOLD_PCT
    )
    improvement_threshold = (
        improvement_threshold_pct if improvement_threshold_pct is not None else DEFAULT_THRESHOLD_PCT
    )
    
    # Access timing stats from Pydantic model
    baseline_mean = baseline_result.timing.mean_ms if baseline_result.timing else 0.0
    optimized_mean = optimized_result.timing.mean_ms if optimized_result.timing else 0.0
    
    if optimized_mean <= 0:
        speedup = 0.0
    else:
        speedup = baseline_mean / optimized_mean
    
    # Detect regression: optimized is slower by threshold
    regression = False
    regression_pct = None
    if speedup < 1.0:
        regression_pct = (1.0 - speedup) * 100
        regression = regression_pct >= regression_threshold
    
    # Detect improvement: optimized is faster by threshold
    improvement_pct = None
    if speedup > 1.0:
        improvement_pct = (speedup - 1.0) * 100
    
    return ComparisonResult(
        baseline_mean_ms=baseline_mean,
        optimized_mean_ms=optimized_mean,
        speedup=speedup,
        regression=regression,
        regression_pct=regression_pct,
        improvement_pct=improvement_pct if improvement_pct and improvement_pct >= improvement_threshold else None,
    )


def detect_regressions(
    comparisons: List[ComparisonResult],
    regression_threshold_pct: float = 5.0
) -> List[ComparisonResult]:
    """Detect regressions from a list of comparisons.
    
    Args:
        comparisons: List of ComparisonResult objects
        regression_threshold_pct: Percentage degradation to consider a regression
        
    Returns:
        List of ComparisonResult objects that represent regressions
    """
    return [c for c in comparisons if c.regression]


# Metric configuration: maps metric paths to (display_name, direction, unit, regression_threshold, improvement_threshold)
# For backward compatibility, if only one threshold is provided, it's used for both directions.
METRIC_CONFIG: Dict[str, Tuple[str, MetricDirection, str, float, float]] = {
    # Timing metrics (lower is better)
    "timing.mean_ms": ("Mean Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.median_ms": ("Median Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.min_ms": ("Min Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.max_ms": ("Max Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.std_ms": ("Std Dev Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 10.0, 10.0),
    "timing.p99_ms": ("P99 Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.p95_ms": ("P95 Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "timing.p90_ms": ("P90 Execution Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    
    # Memory metrics (lower is better for usage)
    "memory.peak_mb": ("Peak Memory", MetricDirection.LOWER_IS_BETTER, "MB", 10.0, 10.0),
    "memory.allocated_mb": ("Allocated Memory", MetricDirection.LOWER_IS_BETTER, "MB", 10.0, 10.0),
    "memory.reserved_mb": ("Reserved Memory", MetricDirection.LOWER_IS_BETTER, "MB", 10.0, 10.0),
    
    # Profiler metrics - NCU
    "profiler_metrics.ncu.kernel_time_ms": ("NCU Kernel Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "profiler_metrics.ncu.sm_throughput_pct": ("NCU SM Throughput", MetricDirection.HIGHER_IS_BETTER, "%", 5.0, 5.0),
    "profiler_metrics.ncu.dram_throughput_pct": ("NCU DRAM Throughput", MetricDirection.HIGHER_IS_BETTER, "%", 5.0, 5.0),
    "profiler_metrics.ncu.l2_throughput_pct": ("NCU L2 Throughput", MetricDirection.HIGHER_IS_BETTER, "%", 5.0, 5.0),
    "profiler_metrics.ncu.occupancy_pct": ("NCU Occupancy", MetricDirection.HIGHER_IS_BETTER, "%", 5.0, 5.0),
    
    # Profiler metrics - NSYS
    "profiler_metrics.nsys.total_gpu_time_ms": ("NSYS Total GPU Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    
    # Profiler metrics - Proton (TorchInductor/Triton)
    "profiler_metrics.proton.kernel_count": ("Proton Kernel Count", MetricDirection.LOWER_IS_BETTER, "kernels", 5.0, 5.0),
    "profiler_metrics.proton.max_regs_per_thread": ("Proton Max Regs/Thread", MetricDirection.LOWER_IS_BETTER, "regs", 5.0, 5.0),
    "profiler_metrics.proton.max_shared_mem_bytes": ("Proton Max Shared Mem", MetricDirection.LOWER_IS_BETTER, "bytes", 5.0, 5.0),
    "profiler_metrics.proton.max_blocks_per_sm": ("Proton Max Blocks/SM", MetricDirection.HIGHER_IS_BETTER, "blocks", 5.0, 5.0),
    "profiler_metrics.proton.max_time_ms": ("Proton Max Kernel Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "profiler_metrics.proton.occupancy_limited_kernel_count": ("Proton Occupancy-Limited Kernels", MetricDirection.LOWER_IS_BETTER, "kernels", 5.0, 5.0),
    
    # Profiler metrics - PyTorch
    "profiler_metrics.torch.total_time_ms": ("PyTorch Total Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "profiler_metrics.torch.cuda_time_ms": ("PyTorch CUDA Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "profiler_metrics.torch.cpu_time_ms": ("PyTorch CPU Time", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "profiler_metrics.torch.memory_allocated_mb": ("PyTorch Memory Allocated", MetricDirection.LOWER_IS_BETTER, "MB", 10.0, 10.0),
    
    # Inference timing metrics (TTFT/TPOT)
    "inference_timing.ttft_mean_ms": ("TTFT Mean", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.ttft_p50_ms": ("TTFT P50", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.ttft_p90_ms": ("TTFT P90", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.ttft_p95_ms": ("TTFT P95", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.ttft_p99_ms": ("TTFT P99", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.tpot_mean_ms": ("TPOT Mean", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.tpot_p50_ms": ("TPOT P50", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.tpot_p90_ms": ("TPOT P90", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.tpot_p95_ms": ("TPOT P95", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
    "inference_timing.tpot_p99_ms": ("TPOT P99", MetricDirection.LOWER_IS_BETTER, "ms", 5.0, 5.0),
}


def _format_percentile_metric_name(prefix: str, percentile: Any, suffix: str = "_ms") -> str:
    """Normalize percentile key names to avoid duplicates like p99_vs_p99.0."""
    pct_value: Optional[float] = None
    if isinstance(percentile, (int, float)):
        pct_value = float(percentile)
    elif isinstance(percentile, str):
        cleaned = percentile.strip().lower()
        if cleaned.startswith("p"):
            cleaned = cleaned[1:]
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            pct_value = float(cleaned)
        except ValueError:
            pct_value = None
    if pct_value is not None and math.isfinite(pct_value):
        if float(pct_value).is_integer():
            pct_str = str(int(pct_value))
        else:
            pct_str = f"{pct_value}".rstrip("0").rstrip(".")
    else:
        pct_str = str(percentile).strip()
    pct_str = pct_str.replace(".", "_").replace(" ", "")
    return f"{prefix}p{pct_str}{suffix}"


def extract_metrics(benchmark_result, include_raw_metrics: bool = False) -> Dict[str, float]:
    """Extract all numeric metrics from a BenchmarkResult object.
    
    Args:
        benchmark_result: BenchmarkResult (Pydantic) object
        include_raw_metrics: If True, include raw profiler metrics (default: False)
        
    Returns:
        Dictionary mapping metric paths to values
    """
    metrics = {}
    
    # Extract timing metrics
    if benchmark_result.timing:
        timing = benchmark_result.timing
        metrics["timing.mean_ms"] = timing.mean_ms
        metrics["timing.median_ms"] = timing.median_ms
        metrics["timing.min_ms"] = timing.min_ms
        metrics["timing.max_ms"] = timing.max_ms
        metrics["timing.std_ms"] = timing.std_ms
        
        if timing.p99_ms is not None:
            metrics["timing.p99_ms"] = timing.p99_ms
        if timing.p95_ms is not None:
            metrics["timing.p95_ms"] = timing.p95_ms
        if timing.p90_ms is not None:
            metrics["timing.p90_ms"] = timing.p90_ms
        
        # Extract percentiles from dict if available
        # Only add if metric name doesn't already exist (prevents duplicates)
        if timing.percentiles:
            for pct, value in timing.percentiles.items():
                metric_name = _format_percentile_metric_name("timing.", pct, "_ms")
                if metric_name not in metrics:
                    metrics[metric_name] = value
    
    # Extract memory metrics
    if benchmark_result.memory:
        memory = benchmark_result.memory
        if memory.peak_mb is not None:
            metrics["memory.peak_mb"] = memory.peak_mb
        if memory.allocated_mb is not None:
            metrics["memory.allocated_mb"] = memory.allocated_mb
        if memory.reserved_mb is not None:
            metrics["memory.reserved_mb"] = memory.reserved_mb
    
    # Extract inference timing metrics
    if benchmark_result.inference_timing:
        inference = benchmark_result.inference_timing
        metrics["inference_timing.ttft_mean_ms"] = inference.ttft_mean_ms
        if inference.ttft_p50_ms is not None:
            metrics["inference_timing.ttft_p50_ms"] = inference.ttft_p50_ms
        if inference.ttft_p90_ms is not None:
            metrics["inference_timing.ttft_p90_ms"] = inference.ttft_p90_ms
        if inference.ttft_p95_ms is not None:
            metrics["inference_timing.ttft_p95_ms"] = inference.ttft_p95_ms
        if inference.ttft_p99_ms is not None:
            metrics["inference_timing.ttft_p99_ms"] = inference.ttft_p99_ms
        
        metrics["inference_timing.tpot_mean_ms"] = inference.tpot_mean_ms
        if inference.tpot_p50_ms is not None:
            metrics["inference_timing.tpot_p50_ms"] = inference.tpot_p50_ms
        if inference.tpot_p90_ms is not None:
            metrics["inference_timing.tpot_p90_ms"] = inference.tpot_p90_ms
        if inference.tpot_p95_ms is not None:
            metrics["inference_timing.tpot_p95_ms"] = inference.tpot_p95_ms
        if inference.tpot_p99_ms is not None:
            metrics["inference_timing.tpot_p99_ms"] = inference.tpot_p99_ms
        
        # Extract percentiles from dict if available
        if inference.ttft_percentiles:
            for pct, value in inference.ttft_percentiles.items():
                metric_name = _format_percentile_metric_name("inference_timing.ttft_", pct, "_ms")
                if metric_name not in metrics:
                    metrics[metric_name] = value
        
        if inference.tpot_percentiles:
            for pct, value in inference.tpot_percentiles.items():
                metric_name = _format_percentile_metric_name("inference_timing.tpot_", pct, "_ms")
                if metric_name not in metrics:
                    metrics[metric_name] = value
    
    # Extract profiler metrics
    if benchmark_result.profiler_metrics:
        prof = benchmark_result.profiler_metrics
        
        # NCU metrics
        if prof.ncu:
            ncu = prof.ncu
            if ncu.kernel_time_ms is not None:
                metrics["profiler_metrics.ncu.kernel_time_ms"] = ncu.kernel_time_ms
            # An NCU selected-range duration is descriptive capture metadata. It
            # is intentionally not a benchmark timing or per-kernel comparison.
            if ncu.sm_throughput_pct is not None:
                metrics["profiler_metrics.ncu.sm_throughput_pct"] = ncu.sm_throughput_pct
            if ncu.dram_throughput_pct is not None:
                metrics["profiler_metrics.ncu.dram_throughput_pct"] = ncu.dram_throughput_pct
            if ncu.l2_throughput_pct is not None:
                metrics["profiler_metrics.ncu.l2_throughput_pct"] = ncu.l2_throughput_pct
            if ncu.occupancy_pct is not None:
                metrics["profiler_metrics.ncu.occupancy_pct"] = ncu.occupancy_pct
            
            # Add raw metrics only if explicitly requested
            if include_raw_metrics:
                for key, value in ncu.raw_metrics.items():
                    metric_name = f"profiler_metrics.ncu.raw.{key}"
                    if _is_ncu_range_duration_metric(metric_name):
                        continue
                    # Avoid isinstance to prevent TypeGuardedType serialization issues
                    try:
                        metrics[metric_name] = float(value)
                    except (TypeError, ValueError):
                        pass
        
        # NSYS metrics
        if prof.nsys:
            nsys = prof.nsys
            if nsys.total_gpu_time_ms is not None:
                metrics["profiler_metrics.nsys.total_gpu_time_ms"] = nsys.total_gpu_time_ms
            
            # Add raw metrics only if explicitly requested
            # Note: Bandwidth metrics (memory_throughput_gb_per_s) are stored in raw_metrics,
            # so include_raw_metrics=True is required for chapter bandwidth targets
            if include_raw_metrics:
                for key, value in nsys.raw_metrics.items():
                    # Avoid isinstance to prevent TypeGuardedType serialization issues
                    try:
                        metrics[f"profiler_metrics.nsys.raw.{key}"] = float(value)
                    except (TypeError, ValueError):
                        pass
        
        # Proton metrics
        if getattr(prof, "proton", None):
            proton = prof.proton
            if proton.kernel_count is not None:
                metrics["profiler_metrics.proton.kernel_count"] = proton.kernel_count
            if proton.summary_stats:
                for key, value in proton.summary_stats.items():
                    try:
                        metrics[f"profiler_metrics.proton.{key}"] = float(value)
                    except (TypeError, ValueError):
                        continue
            if proton.occupancy_limited_kernels:
                metrics["profiler_metrics.proton.occupancy_limited_kernel_count"] = len(proton.occupancy_limited_kernels)
                if include_raw_metrics:
                    metrics["profiler_metrics.proton.occupancy_limited_kernel_names"] = ",".join(proton.occupancy_limited_kernels)
            if include_raw_metrics and proton.kernel_summaries:
                for idx, kernel in enumerate(proton.kernel_summaries[:3]):
                    name = kernel.get("name") or f"kernel_{idx}"
                    for field in ("regs_per_thread", "shared_mem_bytes", "blocks_per_sm", "occupancy_pct", "time_ms"):
                        try:
                            value = kernel.get(field)
                            metrics[f"profiler_metrics.proton.kernels.{name}.{field}"] = float(value)
                        except (TypeError, ValueError, AttributeError):
                            continue
        
        # PyTorch metrics
        if prof.torch:
            torch_metrics = prof.torch
            if torch_metrics.total_time_ms is not None:
                metrics["profiler_metrics.torch.total_time_ms"] = torch_metrics.total_time_ms
            if torch_metrics.cuda_time_ms is not None:
                metrics["profiler_metrics.torch.cuda_time_ms"] = torch_metrics.cuda_time_ms
            if torch_metrics.cpu_time_ms is not None:
                metrics["profiler_metrics.torch.cpu_time_ms"] = torch_metrics.cpu_time_ms
            if torch_metrics.memory_allocated_mb is not None:
                metrics["profiler_metrics.torch.memory_allocated_mb"] = torch_metrics.memory_allocated_mb
            
            # Add raw metrics only if explicitly requested
            if include_raw_metrics:
                for key, value in torch_metrics.raw_metrics.items():
                    # Avoid isinstance to prevent TypeGuardedType serialization issues
                    try:
                        metrics[f"profiler_metrics.torch.raw.{key}"] = float(value)
                    except (TypeError, ValueError):
                        pass
    
    return metrics


def get_chapter_metric_config(chapter: str) -> Dict[str, Tuple[str, MetricDirection, str, float, float]]:
    """Get chapter-specific metric configuration from performance_targets.py.
    
    Maps chapter metric names from performance_targets.py to BenchmarkResult metric paths.
    Derived speedup targets are represented by the timing comparison's speedup,
    not by reinterpreting raw milliseconds. Absolute target values are not
    regression percentages; use the metric's configured percentage thresholds.
    Handles unknown metrics gracefully (skips with warning) and uses metadata when available.
    
    Args:
        chapter: Chapter identifier (e.g., "ch07")
        
    Returns:
        Dictionary mapping metric paths to METRIC_CONFIG tuples
    """
    global _target_metadata_warning_logged
    if not _target_metadata_warning_logged and _target_metadata_loader is not None:
        try:
            target_metadata = _target_metadata_loader()
        except Exception:
            target_metadata = {}
        warnings = target_metadata.get("warnings") or []
        if warnings:
            logger.warning("Performance target metadata warnings: %s", "; ".join(str(w) for w in warnings))
            _target_metadata_warning_logged = True

    try:
        chapter_metrics = get_chapter_metrics(chapter)
    except (ImportError, TypeError, AttributeError) as exc:
        # If import fails or module has errors, return empty dict gracefully
        logger.warning(
            "Could not load chapter metrics for %s, skipping chapter-specific config: %s",
            chapter,
            exc,
        )
        return {}
    
    if not chapter_metrics:
        return {}
    
    # Warn if include_raw_metrics might be needed for some metrics
    raw_metric_keys = [
        "shared_memory_throughput_percent",
        "tensor_core_utilization_percent",
        "stream_overlap_percent",
        # Bandwidth metrics use NSYS raw metrics
        "nvlink_bandwidth_gbs",
        "hbm3e_bandwidth_tbs",
        "allreduce_bandwidth_gbs",
        "p2p_bandwidth_gbs",
    ]
    # Metrics that require NSYS (not just raw NCU)
    nsys_metric_keys = [
        "nvlink_bandwidth_gbs",
        "hbm3e_bandwidth_tbs",
        "allreduce_bandwidth_gbs",
        "p2p_bandwidth_gbs",
    ]
    # Metrics that are TBD (not yet implemented)
    tbd_metric_keys = [
        "nvswitch_bandwidth_tbs",
        "global_memory_load_efficiency_percent",
    ]
    
    needs_raw = [k for k in chapter_metrics.keys() if k in raw_metric_keys]
    needs_nsys = [k for k in chapter_metrics.keys() if k in nsys_metric_keys]
    has_tbd = [k for k in chapter_metrics.keys() if k in tbd_metric_keys]
    
    if needs_raw:
        # Separate bandwidth metrics (NSYS) from other raw metrics (NCU)
        bandwidth_metrics = [k for k in needs_raw if k in nsys_metric_keys]
        other_raw_metrics = [k for k in needs_raw if k not in nsys_metric_keys]
        
        if bandwidth_metrics:
            logger.warning(
                f"Chapter {chapter} bandwidth metrics {bandwidth_metrics} require NSYS raw metrics. "
                f"Set include_raw_metrics=True in compare_all_metrics() and ensure NSYS profiling is enabled."
            )
        if other_raw_metrics:
            logger.warning(
                f"Chapter {chapter} metrics {other_raw_metrics} require raw profiler metrics. "
                f"Set include_raw_metrics=True in compare_all_metrics() to enable these metrics."
            )
    if needs_nsys and not any(k in needs_raw for k in needs_nsys):
        # Only log if not already covered by raw metrics warning
        logger.info(
            f"Chapter {chapter} metrics {needs_nsys} use NSYS metrics. "
            f"Ensure NSYS profiling is enabled for accurate measurements."
        )
    if has_tbd:
        logger.warning(
            f"Chapter {chapter} metrics {has_tbd} are not yet implemented (counter names TBD). "
            f"These metrics will be skipped until counter names are confirmed."
        )
    
    config = {}
    
    # Mapping from chapter metric names to BenchmarkResult paths
    # This maps the metric names used in performance_targets.py to actual metric paths
    # Note: Metrics requiring raw_metrics (e.g., NSYS bandwidth metrics, NCU raw counters)
    # will only be available when include_raw_metrics=True
    # NSYS bandwidth metrics (memory_throughput_gb_per_s) are stored in nsys.raw_metrics,
    # so include_raw_metrics=True is required for chapter bandwidth targets
    metric_mappings = {
        # Bandwidth metrics -> profiler metrics
        # NVLink bandwidth: Use NSYS Memory Throughput GB/s (actual bandwidth)
        # NSYS parser creates keys like "nsys_memory_throughput_gb_per_s" -> stored in raw_metrics as "memory_throughput_gb_per_s"
        "nvlink_bandwidth_gbs": ("profiler_metrics.nsys.raw.memory_throughput_gb_per_s", "NVLink Bandwidth", MetricDirection.HIGHER_IS_BETTER, "GB/s"),
        # HBM3e bandwidth: Use NSYS Memory Throughput GB/s (actual bandwidth), not NCU percentage
        # Note: Target in performance_targets.py is now in GB/s to match NSYS output
        "hbm3e_bandwidth_tbs": ("profiler_metrics.nsys.raw.memory_throughput_gb_per_s", "HBM3e Bandwidth", MetricDirection.HIGHER_IS_BETTER, "GB/s"),
        # NVSwitch bandwidth: Counter name TBD - skip mapping until confirmed
        # "nvswitch_bandwidth_tbs": ("profiler_metrics.ncu.raw.nvswitch_bandwidth", "NVSwitch Bandwidth", MetricDirection.HIGHER_IS_BETTER, "TB/s"),
        # AllReduce/P2P bandwidth: Use NSYS Memory Throughput GB/s
        "allreduce_bandwidth_gbs": ("profiler_metrics.nsys.raw.memory_throughput_gb_per_s", "AllReduce Bandwidth", MetricDirection.HIGHER_IS_BETTER, "GB/s"),
        "p2p_bandwidth_gbs": ("profiler_metrics.nsys.raw.memory_throughput_gb_per_s", "P2P Bandwidth", MetricDirection.HIGHER_IS_BETTER, "GB/s"),
        
        # TFLOPS metrics -> profiler metrics
        "tflops": ("profiler_metrics.ncu.sm_throughput_pct", "TFLOPS", MetricDirection.HIGHER_IS_BETTER, "TFLOPS"),
        "fp8_tflops": ("profiler_metrics.ncu.sm_throughput_pct", "FP8 TFLOPS", MetricDirection.HIGHER_IS_BETTER, "TFLOPS"),
        "fp16_tflops": ("profiler_metrics.ncu.sm_throughput_pct", "FP16 TFLOPS", MetricDirection.HIGHER_IS_BETTER, "TFLOPS"),
        
        # Latency metrics -> timing metrics
        "latency_ms": ("timing.mean_ms", "Latency", MetricDirection.LOWER_IS_BETTER, "ms"),
        "latency_us": ("timing.mean_ms", "Latency", MetricDirection.LOWER_IS_BETTER, "μs"),
        "small_message_latency_us": ("timing.mean_ms", "Small Message Latency", MetricDirection.LOWER_IS_BETTER, "μs"),
        
        # Utilization/Occupancy -> profiler metrics
        "utilization_percent": ("profiler_metrics.ncu.sm_throughput_pct", "Utilization", MetricDirection.HIGHER_IS_BETTER, "%"),
        "occupancy_percent": ("profiler_metrics.ncu.occupancy_pct", "Occupancy", MetricDirection.HIGHER_IS_BETTER, "%"),
        "achieved_occupancy_percent": ("profiler_metrics.ncu.occupancy_pct", "Achieved Occupancy", MetricDirection.HIGHER_IS_BETTER, "%"),
        "tensor_core_utilization_percent": ("profiler_metrics.ncu.raw.tensor_core_utilization", "Tensor Core Utilization", MetricDirection.HIGHER_IS_BETTER, "%"),
        
        # CH7 Memory Access Pattern metrics
        # DRAM throughput: uses existing dram_throughput_pct field (percentage of peak)
        "dram_throughput_percent": ("profiler_metrics.ncu.dram_throughput_pct", "DRAM Throughput", MetricDirection.HIGHER_IS_BETTER, "%"),
        # Global Memory Load Efficiency: Counter name TBD - skip mapping until confirmed
        # Actual NCU counter name needs to be verified (may be derived metric)
        # "global_memory_load_efficiency_percent": ("profiler_metrics.ncu.raw.<TBD>", "Global Memory Load Efficiency", MetricDirection.HIGHER_IS_BETTER, "%"),
        # Shared memory throughput: Use actual NCU counter name
        "shared_memory_throughput_percent": ("profiler_metrics.ncu.raw.l1tex__throughput.avg.pct_of_peak_sustained_active", "Shared Memory Throughput", MetricDirection.HIGHER_IS_BETTER, "%"),
        
        # Memory metrics
        "memory_reduction_percent": ("memory.peak_mb", "Memory Reduction", MetricDirection.LOWER_IS_BETTER, "%"),
        "quantization_memory_reduction_percent": ("memory.peak_mb", "Quantization Memory Reduction", MetricDirection.LOWER_IS_BETTER, "%"),
        
        # CH4 Distributed Networking metrics
        "scaling_efficiency_percent": ("profiler_metrics.ncu.sm_throughput_pct", "Scaling Efficiency", MetricDirection.HIGHER_IS_BETTER, "%"),
        "topology_efficiency_percent": ("profiler_metrics.ncu.sm_throughput_pct", "Topology Efficiency", MetricDirection.HIGHER_IS_BETTER, "%"),
        
        # CH11 Streams & Concurrency
        "stream_overlap_percent": ("profiler_metrics.ncu.raw.stream_overlap", "Stream Overlap", MetricDirection.HIGHER_IS_BETTER, "%"),
        
        # CH17 Dynamic Routing metrics
        "routing_overhead_ms": ("timing.mean_ms", "Routing Overhead", MetricDirection.LOWER_IS_BETTER, "ms"),
        "load_balance_variance": ("timing.std_ms", "Load Balance Variance", MetricDirection.LOWER_IS_BETTER, ""),
        # TTFT/TPOT: Use InferenceTimingStats model
        "ttft_p99_ms": ("inference_timing.ttft_p99_ms", "TTFT P99", MetricDirection.LOWER_IS_BETTER, "ms"),
        "ttft_p99_mlperf_ms": ("inference_timing.ttft_p99_ms", "TTFT P99 (MLPerf)", MetricDirection.LOWER_IS_BETTER, "ms"),
        "tpot_p99_ms": ("inference_timing.tpot_p99_ms", "TPOT P99", MetricDirection.LOWER_IS_BETTER, "ms"),
    }
    # These targets compare runs; they are not measurements stored in one result.
    # compare_results() already calculates their timing ratio in its speedup field.
    derived_speedup_metrics = {
        "speedup",
        "stream_speedup",
        "fp8_speedup",
        "torch_compile_speedup_small",
        "torch_compile_speedup_large",
        "compiled_autograd_speedup",
        "flex_attention_speedup",
        "fp8_training_speedup",
        "adaptive_parallelism_speedup",
        "dynamic_precision_speedup",
        "dynamic_quantized_cache_speedup",
        "deepseek_l2_speedup",
    }
    
    for metric_name, metric_def in chapter_metrics.items():
        # Skip unimplemented counters and ratios already supplied by the comparison.
        if metric_name in tbd_metric_keys or metric_name in derived_speedup_metrics:
            continue
        
        # Try to find mapping
        if metric_name in metric_mappings:
            metric_path, display_name, default_direction, default_unit = metric_mappings[metric_name]
            
            # Check if this metric requires raw_metrics (NSYS or NCU)
            requires_raw = (
                metric_name in raw_metric_keys or
                ".raw." in metric_path
            )
            
            # Store requirement info in config for later validation
            # (The warning above already covers this, but we track it here for completeness)
        else:
            # Unknown metric - skip with warning
            logger.warning(f"Unknown chapter metric '{metric_name}' for {chapter}, skipping")
            continue
        
        # Get direction from metadata if available, otherwise use default
        direction = default_direction
        # Avoid isinstance to prevent TypeGuardedType serialization issues in mypy cache
        # Treat metric_def as dict-like without type narrowing
        metric_dict: Any = metric_def if metric_def else {}
        if "lower_is_better" in metric_dict:
            direction = MetricDirection.LOWER_IS_BETTER if metric_dict["lower_is_better"] else MetricDirection.HIGHER_IS_BETTER
        
        # Get unit from metadata if available, otherwise use default
        unit = default_unit
        if "unit" in metric_dict:
            unit = metric_dict["unit"]
        
        # Targets/minima are absolute values in the target's own units, evaluated
        # separately by performance_targets.compute_status(). Reuse existing
        # percentage thresholds; chapter-only raw counters use the global default.
        base_config = METRIC_CONFIG.get(metric_path)
        regression_threshold = base_config[3] if base_config else DEFAULT_THRESHOLD_PCT
        improvement_threshold = base_config[4] if base_config else DEFAULT_THRESHOLD_PCT
        
        if metric_path in config:
            logger.info(
                "Chapter %s metric '%s' maps to %s which is already configured; keeping existing metadata",
                chapter,
                metric_name,
                metric_path,
            )
            continue
        
        config[metric_path] = (display_name, direction, unit, regression_threshold, improvement_threshold)
    
    return config


def compare_metric(
    metric_name: str,
    baseline_value: float,
    optimized_value: float,
    regression_threshold_pct: Optional[float] = None,
    improvement_threshold_pct: Optional[float] = None,
    metric_config: Optional[Dict[str, Tuple[str, MetricDirection, str, float, float]]] = None
) -> Optional[MetricComparison]:
    """Compare a single metric between baseline and optimized.
    
    Args:
        metric_name: Full metric path (e.g., "timing.mean_ms")
        baseline_value: Baseline metric value
        optimized_value: Optimized metric value
        regression_threshold_pct: Percentage change to consider regression
        improvement_threshold_pct: Percentage change to consider improvement
        metric_config: Optional metric configuration dict (defaults to METRIC_CONFIG)
        
    Returns:
        MetricComparison object or None if metric not configured
    """
    if metric_config is None:
        metric_config = METRIC_CONFIG

    if _is_ncu_range_duration_metric(metric_name):
        return None
    
    regression_threshold = regression_threshold_pct
    improvement_threshold = improvement_threshold_pct
    
    if metric_name not in metric_config:
        # Try to infer direction from metric name
        display_name = metric_name.replace("_", " ").title()
        direction = MetricDirection.LOWER_IS_BETTER
        unit = ""
        
        # Heuristics for direction (extended keyword lists)
        # Prefer chapter config if metric is in chapter config (handled by caller)
        higher_keywords = ["throughput", "bandwidth", "occupancy", "efficiency", "tflops", "tokens_per_s", "requests_per_s",
                          "inst_executed", "instructions", "sectors", "bytes", "tokens", "requests", "count", "sum"]
        lower_keywords = ["time", "latency", "ms", "memory", "mb", "conflicts", "stalls", "overhead", "waste", "reduction"]
        
        name_lower = metric_name.lower()
        higher_match = any(keyword in name_lower for keyword in higher_keywords)
        lower_match = any(keyword in name_lower for keyword in lower_keywords)
        if higher_match and not lower_match:
            direction = MetricDirection.HIGHER_IS_BETTER
        elif lower_match:
            direction = MetricDirection.LOWER_IS_BETTER
        
        # Heuristics for unit
        if "ms" in metric_name.lower() or "time" in metric_name.lower():
            unit = "ms"
        elif "mb" in metric_name.lower() or "memory" in metric_name.lower():
            unit = "MB"
        elif "pct" in metric_name.lower() or "percent" in metric_name.lower() or "%" in metric_name.lower():
            unit = "%"
    else:
        config_tuple = metric_config[metric_name]
        # Handle backward compatibility: if tuple has 4 elements, duplicate threshold for both directions
        if len(config_tuple) == 4:
            display_name, direction, unit, threshold = config_tuple
            if regression_threshold is None:
                regression_threshold = threshold
            if improvement_threshold is None:
                improvement_threshold = threshold  # Use same threshold for both
        else:
            # New format: 5 elements (display_name, direction, unit, regression_threshold, improvement_threshold)
            display_name, direction, unit, cfg_regression_threshold, cfg_improvement_threshold = config_tuple
            if regression_threshold is None:
                regression_threshold = cfg_regression_threshold
            if improvement_threshold is None:
                improvement_threshold = cfg_improvement_threshold
    
    if regression_threshold is None:
        regression_threshold = DEFAULT_THRESHOLD_PCT
    if improvement_threshold is None:
        improvement_threshold = DEFAULT_THRESHOLD_PCT
    
    # Create comparison
    comparison = MetricComparison(
        metric_name=metric_name,
        display_name=display_name,
        baseline_value=baseline_value,
        optimized_value=optimized_value,
        ratio=0.0,  # Will be calculated in __post_init__
        direction=direction,
        unit=unit
    )
    
    # Calculate change percentage using delta-based calculation (not ratio-based)
    # CRITICAL: Handle zero/None baselines to prevent divide-by-zero crashes
    if baseline_value is None or baseline_value == 0:
        # Cannot compute percentage when baseline is zero or None
        comparison.improvement_pct = None
        comparison.regression_pct = None
        comparison.significant_change = False
        return comparison
    
    if direction == MetricDirection.LOWER_IS_BETTER:
        # For lower-is-better: improvement means optimized < baseline
        # Delta-based: improvement_pct = (baseline - optimized) / baseline * 100
        if optimized_value < baseline_value:
            improvement_pct = ((baseline_value - optimized_value) / baseline_value) * 100
            comparison.improvement_pct = improvement_pct
            comparison.significant_change = improvement_pct >= improvement_threshold
        elif optimized_value > baseline_value:
            regression_pct = ((optimized_value - baseline_value) / baseline_value) * 100
            comparison.regression_pct = regression_pct
            comparison.regression = regression_pct >= regression_threshold
            comparison.significant_change = regression_pct >= regression_threshold
    else:
        # For higher-is-better: improvement means optimized > baseline
        # Delta-based: improvement_pct = (optimized - baseline) / baseline * 100
        if optimized_value > baseline_value:
            improvement_pct = ((optimized_value - baseline_value) / baseline_value) * 100
            comparison.improvement_pct = improvement_pct
            comparison.significant_change = improvement_pct >= improvement_threshold
        elif optimized_value < baseline_value:
            regression_pct = ((baseline_value - optimized_value) / baseline_value) * 100
            comparison.regression_pct = regression_pct
            comparison.regression = regression_pct >= regression_threshold
            comparison.significant_change = regression_pct >= regression_threshold
    
    return comparison


def compare_all_metrics(
    baseline_result,
    optimized_result,
    regression_threshold_pct: Optional[float] = None,
    improvement_threshold_pct: Optional[float] = None,
    include_timing: bool = True,
    include_raw_metrics: bool = False,
    chapter: Optional[str] = None
) -> ComprehensiveComparison:
    """Compare all metrics between baseline and optimized results.
    
    Args:
        baseline_result: Baseline BenchmarkResult
        optimized_result: Optimized BenchmarkResult
        regression_threshold_pct: Threshold for detecting regressions. When None, use metric config or default 25%.
        improvement_threshold_pct: Threshold for detecting improvements. When None, use metric config or default 25%.
        include_timing: Include timing metrics (default: True)
        include_raw_metrics: Include raw profiler metrics (NCU/NSYS raw counters).
            Required for chapter bandwidth metrics (nvlink_bandwidth_gbs, hbm3e_bandwidth_tbs, etc.)
            and other raw metrics (shared_memory_throughput_percent, etc.)
        chapter: Chapter identifier (e.g., "ch07") to enable chapter-specific metrics (default: None)
    
    Returns:
        ComprehensiveComparison with all metric comparisons
    """
    # Merge chapter config if provided
    merged_config = METRIC_CONFIG.copy()
    if chapter:
        chapter_config = get_chapter_metric_config(chapter)
        # Runtime check: warn if chapter config requires raw_metrics but they're not enabled
        if chapter_config and not include_raw_metrics:
            # Check if any chapter metrics require raw_metrics (have .raw. in path)
            chapter_raw_paths = [path for path in chapter_config.keys() if ".raw." in path]
            if chapter_raw_paths:
                logger.warning(
                    f"Chapter {chapter} has {len(chapter_raw_paths)} metric(s) requiring raw profiler data "
                    f"(paths contain '.raw.'), but include_raw_metrics=False. "
                    f"Set include_raw_metrics=True to enable these metrics. "
                    f"Note: NSYS bandwidth metrics (e.g., nvlink_bandwidth_gbs, hbm3e_bandwidth_tbs) "
                    f"require both include_raw_metrics=True and NSYS profiling enabled."
                )
        # The result schema owns a known path's meaning, units and thresholds.
        # Chapter aliases can add raw counters, but must not reinterpret an
        # existing measurement (for example milliseconds as a higher-is-better ratio).
        for metric_path, config in chapter_config.items():
            merged_config.setdefault(metric_path, config)
    
    # Get timing comparison (legacy)
    timing_comparison = compare_results(
        baseline_result,
        optimized_result,
        regression_threshold_pct=regression_threshold_pct,
        improvement_threshold_pct=improvement_threshold_pct
    )
    
    # Extract all metrics
    baseline_metrics = extract_metrics(baseline_result, include_raw_metrics=include_raw_metrics)
    optimized_metrics = extract_metrics(optimized_result, include_raw_metrics=include_raw_metrics)
    
    # Find common metrics
    common_metrics = set(baseline_metrics.keys()) & set(optimized_metrics.keys())
    
    # Compare each metric
    metric_comparisons = []
    for metric_name in sorted(common_metrics):
        baseline_value = baseline_metrics[metric_name]
        optimized_value = optimized_metrics[metric_name]
        
        # Use merged config for this metric comparison
        comparison = compare_metric(
            metric_name,
            baseline_value,
            optimized_value,
            regression_threshold_pct=regression_threshold_pct,
            improvement_threshold_pct=improvement_threshold_pct,
            metric_config=merged_config
        )
        
        if comparison:
            metric_comparisons.append(comparison)
    
    # If include_timing is True, ensure timing.mean_ms is in metric_comparisons
    # (it should already be there from the loop above, but this ensures it's included)
    if include_timing and "timing.mean_ms" in common_metrics:
        # Verify timing.mean_ms is already in metric_comparisons
        timing_in_comparisons = any(c.metric_name == "timing.mean_ms" for c in metric_comparisons)
        if not timing_in_comparisons:
            # Add it manually if somehow missing
            baseline_timing = baseline_metrics["timing.mean_ms"]
            optimized_timing = optimized_metrics["timing.mean_ms"]
            timing_comparison_metric = compare_metric(
                "timing.mean_ms",
                baseline_timing,
                optimized_timing,
                regression_threshold_pct=regression_threshold_pct,
                improvement_threshold_pct=improvement_threshold_pct,
                metric_config=merged_config
            )
            if timing_comparison_metric:
                metric_comparisons.insert(0, timing_comparison_metric)  # Insert at beginning
    
    return ComprehensiveComparison(
        timing_comparison=timing_comparison,
        metric_comparisons=metric_comparisons
    )


def format_comparison(comparison: ComparisonResult, name: str = "Benchmark") -> str:
    """Format a comparison result as a human-readable string.
    
    Args:
        comparison: ComparisonResult to format
        name: Name of the benchmark
        
    Returns:
        Formatted string
    """
    lines = [
        f"{name}:",
        f"  Baseline: {comparison.baseline_mean_ms:.3f} ms",
        f"  Optimized: {comparison.optimized_mean_ms:.3f} ms",
        f"  Speedup: {comparison.speedup:.2f}x",
    ]
    
    if comparison.regression:
        lines.append(f"  ⚠ REGRESSION: {comparison.regression_pct:.1f}% slower")
    elif comparison.improvement_pct:
        lines.append(f"  ✓ Improvement: {comparison.improvement_pct:.1f}% faster")
    else:
        lines.append("  → No significant change")
    
    return "\n".join(lines)


def format_metric_comparison_table(
    comprehensive_comparison: ComprehensiveComparison,
    name: str = "Benchmark",
    show_only_significant: bool = False,
    max_width: int = 100
) -> str:
    """Format comprehensive comparison as a table showing all metrics.
    
    Args:
        comprehensive_comparison: ComprehensiveComparison object
        name: Name of the benchmark
        show_only_significant: If True, only show metrics with significant changes
        max_width: Maximum width for table formatting
        
    Returns:
        Formatted table string
    """
    all_comparisons = comprehensive_comparison.get_all_comparisons()
    
    if show_only_significant:
        all_comparisons = [c for c in all_comparisons if c.significant_change]
    
    if not all_comparisons:
        return f"{name}: No metrics to compare"
    
    lines = []
    lines.append(f"\n{'='*max_width}")
    lines.append(f"METRIC COMPARISON: {name}")
    lines.append(f"{'='*max_width}")
    
    # Table header
    header = f"{'Metric':<35} {'Baseline':>15} {'Optimized':>15} {'Ratio':>10} {'Status':>12}"
    lines.append(header)
    lines.append("-" * max_width)
    
    # Group metrics by category
    timing_metrics = []
    memory_metrics = []
    profiler_metrics = []
    other_metrics = []
    
    for comp in all_comparisons:
        if comp.metric_name.startswith("timing."):
            timing_metrics.append(comp)
        elif comp.metric_name.startswith("memory."):
            memory_metrics.append(comp)
        elif comp.metric_name.startswith("profiler_metrics."):
            profiler_metrics.append(comp)
        else:
            other_metrics.append(comp)
    
    # Format each category
    def format_metric_row(comp: MetricComparison) -> str:
        """Format a single metric row."""
        unit_str = f" {comp.unit}" if comp.unit else ""
        
        baseline_str = f"{comp.baseline_value:>14.3f}{unit_str}"
        optimized_str = f"{comp.optimized_value:>14.3f}{unit_str}"
        
        # Format ratio
        if comp.ratio == float('inf'):
            ratio_str = "∞"
        elif comp.ratio == 0.0:
            ratio_str = "0.00x"
        else:
            ratio_str = f"{comp.ratio:.2f}x"
        
        # Status indicator
        if comp.regression:
            status = "⚠️ REGRESS"
        elif comp.improvement_pct:
            status = "✅ IMPROVE"
        else:
            status = "→"
        
        return f"{comp.display_name:<35} {baseline_str} {optimized_str} {ratio_str:>10} {status:>12}"
    
    # Print timing metrics
    if timing_metrics:
        lines.append("\n📊 TIMING METRICS:")
        for comp in timing_metrics:
            lines.append(format_metric_row(comp))
    
    # Print memory metrics
    if memory_metrics:
        lines.append("\n💾 MEMORY METRICS:")
        for comp in memory_metrics:
            lines.append(format_metric_row(comp))
    
    # Print profiler metrics
    if profiler_metrics:
        lines.append("\n🔬 PROFILER METRICS:")
        for comp in profiler_metrics:
            lines.append(format_metric_row(comp))
    
    # Print other metrics
    if other_metrics:
        lines.append("\n📈 OTHER METRICS:")
        for comp in other_metrics:
            lines.append(format_metric_row(comp))
    
    lines.append("-" * max_width)
    
    # Summary
    improvements = [c for c in all_comparisons if c.improvement_pct]
    regressions = [c for c in all_comparisons if c.regression]
    
    lines.append(f"\nSummary: {len(improvements)} improvements, {len(regressions)} regressions")
    
    if improvements:
        best = max(improvements, key=lambda x: x.ratio)
        lines.append(f"Best improvement: {best.display_name} ({best.ratio:.2f}x)")
    
    if regressions:
        worst = min(regressions, key=lambda x: x.ratio)
        lines.append(f"Worst regression: {worst.display_name} ({worst.ratio:.2f}x)")
    
    lines.append("=" * max_width)
    
    return "\n".join(lines)


def format_metric_comparison_summary(
    comprehensive_comparison: ComprehensiveComparison,
    name: str = "Benchmark"
) -> str:
    """Format a concise summary of metric comparisons.
    
    Args:
        comprehensive_comparison: ComprehensiveComparison object
        name: Name of the benchmark
        
    Returns:
        Formatted summary string
    """
    all_comparisons = comprehensive_comparison.get_all_comparisons()
    
    improvements = [c for c in all_comparisons if c.improvement_pct]
    regressions = [c for c in all_comparisons if c.regression]
    
    lines = [f"\n{name} - Metric Comparison Summary:"]
    
    if improvements:
        lines.append(f"  ✅ Improvements ({len(improvements)}):")
        for comp in sorted(improvements, key=lambda x: x.ratio, reverse=True)[:5]:  # Top 5
            lines.append(f"    • {comp.display_name}: {comp.ratio:.2f}x ({comp.improvement_pct:.1f}% better)")
    
    if regressions:
        lines.append(f"  ⚠️  Regressions ({len(regressions)}):")
        for comp in sorted(regressions, key=lambda x: x.ratio)[:5]:  # Worst 5
            lines.append(f"    • {comp.display_name}: {comp.ratio:.2f}x ({comp.regression_pct:.1f}% worse)")
    
    if not improvements and not regressions:
        lines.append("  → No significant changes detected")
    
    return "\n".join(lines)


def compare_and_display_all_metrics(
    baseline_result,
    optimized_result,
    name: str = "Benchmark",
    regression_threshold_pct: float = 5.0,
    improvement_threshold_pct: float = 5.0,
    format_style: str = "table",  # "table", "summary", or "both"
    show_only_significant: bool = False,
    include_raw_metrics: bool = False,
    chapter: Optional[str] = None
) -> ComprehensiveComparison:
    """Convenience function to compare all metrics and display results.
    
    This is the main entry point for comparing all metrics. It:
    1. Extracts all metrics from both results
    2. Compares them comprehensively
    3. Displays formatted output
    
    Args:
        baseline_result: BenchmarkResult (Pydantic) from baseline run
        optimized_result: BenchmarkResult (Pydantic) from optimized run
        name: Name of the benchmark for display
        regression_threshold_pct: Percentage degradation to consider a regression (default: 5%)
        improvement_threshold_pct: Percentage improvement to consider significant (default: 5%)
        format_style: Output format - "table", "summary", or "both" (default: "table")
        show_only_significant: If True, only show metrics with significant changes (default: False)
        include_raw_metrics: If True, include raw profiler metrics (default: False)
        chapter: Chapter identifier (e.g., "ch07") to enable chapter-specific metrics (default: None)
        
    Returns:
        ComprehensiveComparison object
        
    Example:
        >>> from core.benchmark.comparison import compare_and_display_all_metrics
        >>> comprehensive = compare_and_display_all_metrics(
        ...     baseline_result=baseline_benchmark_result,
        ...     optimized_result=optimized_benchmark_result,
        ...     name="My Benchmark",
        ...     format_style="both",
        ...     include_raw_metrics=True,
        ...     chapter="ch07"
        ... )
    """
    # Perform comprehensive comparison
    comprehensive = compare_all_metrics(
        baseline_result,
        optimized_result,
        regression_threshold_pct=regression_threshold_pct,
        improvement_threshold_pct=improvement_threshold_pct,
        include_raw_metrics=include_raw_metrics,
        chapter=chapter
    )
    
    # Display results based on format_style
    if format_style in ("table", "both"):
        print(format_metric_comparison_table(
            comprehensive,
            name=name,
            show_only_significant=show_only_significant
        ))
    
    if format_style in ("summary", "both"):
        print(format_metric_comparison_summary(comprehensive, name=name))
    
    return comprehensive
