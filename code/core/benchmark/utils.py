"""benchmark_utils.py - Shared benchmarking utilities for PyTorch examples."""

import time
import torch
import statistics
from typing import Callable, List, Optional, Dict, Any

# Import logger
try:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def scalar_tensor_to_float(value: torch.Tensor) -> float:
    """Read a scalar tensor through the standard detached CPU-list path."""
    return float(value.detach().cpu().tolist())


def warmup_cuda(func: Callable, iterations: int = 10) -> None:
    """Warmup function to stabilize GPU clocks and caches."""
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()


def benchmark_function(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark a function with proper warmup and synchronization.
    
    Args:
        func: Function to benchmark
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        device: CUDA device (if None, will detect)
    
    Returns:
        Dictionary with timing statistics (mean, std, min, max, median)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_cuda = device.type == "cuda"
    
    # Warmup
    for _ in range(warmup):
        func()
    if is_cuda:
        torch.cuda.synchronize(device)
    
    # Benchmark
    times: List[float] = []
    for _ in range(iterations):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        func()
        if is_cuda:
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
    }


def compare_implementations(
    baseline: Callable,
    optimized: Callable,
    name: str = "Comparison",
    iterations: int = 100,
    warmup: int = 10
) -> None:
    """
    Compare baseline vs optimized implementations and print results.
    
    Args:
        baseline: Baseline function
        optimized: Optimized function
        name: Name for the comparison
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"{name}")
    logger.info(f"{'='*70}")
    
    logger.info("\nBaseline:")
    baseline_stats = benchmark_function(baseline, iterations, warmup)
    logger.info(f"  Mean: {baseline_stats['mean_ms']:.3f} ms")
    logger.info(f"  Std:  {baseline_stats['std_ms']:.3f} ms")
    logger.info(f"  Min:  {baseline_stats['min_ms']:.3f} ms")
    logger.info(f"  Max:  {baseline_stats['max_ms']:.3f} ms")
    
    logger.info("\nOptimized:")
    optimized_stats = benchmark_function(optimized, iterations, warmup)
    logger.info(f"  Mean: {optimized_stats['mean_ms']:.3f} ms")
    logger.info(f"  Std:  {optimized_stats['std_ms']:.3f} ms")
    logger.info(f"  Min:  {optimized_stats['min_ms']:.3f} ms")
    logger.info(f"  Max:  {optimized_stats['max_ms']:.3f} ms")
    
    speedup = baseline_stats['mean_ms'] / optimized_stats['mean_ms']
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    logger.info(f"{'='*70}\n")


def calculate_bandwidth_gbs(bytes_transferred: int, time_ms: float) -> float:
    """Calculate memory bandwidth in GB/s."""
    return (bytes_transferred / (1024**3)) / (time_ms / 1000)


def calculate_tflops(flops: int, time_ms: float) -> float:
    """Calculate TFLOPS (trillion floating-point operations per second)."""
    return (flops / 1e12) / (time_ms / 1000)


def print_gpu_info(device: int = 0) -> None:
    """Print GPU information for the specified device using structured logging."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    
    prop = torch.cuda.get_device_properties(device)
    logger.info(f"\nGPU {device}: {prop.name}")
    logger.info(f"  Compute capability: {prop.major}.{prop.minor}")
    logger.info(f"  Total memory: {prop.total_memory / (1024**3):.2f} GB")
    logger.info(f"  Multi-processors: {prop.multi_processor_count}")
    logger.info(f"  CUDA cores: ~{prop.multi_processor_count * 128}")  # Approximate
    logger.info("")


def format_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format comparison results as a markdown table.
    
    Args:
        results: Dictionary mapping implementation name to stats dict
    
    Returns:
        Formatted markdown table string
    """
    lines = []
    lines.append("| Implementation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    lines.append("|---------------|-----------|----------|----------|----------|")
    
    for name, stats in results.items():
        lines.append(
            f"| {name:13s} | {stats['mean_ms']:9.3f} | "
            f"{stats['std_ms']:8.3f} | {stats['min_ms']:8.3f} | "
            f"{stats['max_ms']:8.3f} |"
        )
    
    return "\n".join(lines)


def warn_benchmark_scaling(
    scaling_type: str,
    original_values: Dict[str, Any],
    scaled_values: Dict[str, Any],
    impact_description: Optional[str] = None,
    recommendation: Optional[str] = None
) -> None:
    """
    Print a standardized warning when benchmark parameters are automatically scaled.
    
    This function provides consistent warning messages across all benchmarks when
    automatic scaling occurs to prevent OOM or other resource constraints. The warnings
    inform users that scaling may affect measured speedup of optimizations.
    
    Args:
        scaling_type: Type of scaling (e.g., "Model size", "Workload size", "Batch size")
        original_values: Dictionary of original parameter values (e.g., {"layers": 48, "d_model": 8192})
        scaled_values: Dictionary of scaled parameter values (e.g., {"layers": 32, "d_model": 6144})
        impact_description: Optional custom description of how scaling affects benchmarks
        recommendation: Optional recommendation for accurate benchmarks (e.g., "use GPUs with >=80GB memory")
    
    Example:
        warn_benchmark_scaling(
            scaling_type="Model size",
            original_values={"layers": 48, "d_model": 8192, "d_ff": 32768},
            scaled_values={"layers": 32, "d_model": 6144, "d_ff": 24576},
            impact_description="Smaller models may compile faster (reducing timeout benefit)",
            recommendation="For accurate production benchmarks, use GPUs with >=80GB memory"
        )
    """
    # Check if any scaling actually occurred
    any_scaled = False
    for key in original_values:
        if key in scaled_values:
            orig_val = original_values[key]
            scaled_val = scaled_values[key]
            # Handle numeric comparisons
            if isinstance(orig_val, (int, float)) and isinstance(scaled_val, (int, float)):
                if scaled_val < orig_val:
                    any_scaled = True
                    break
            # Handle string/other comparisons
            elif scaled_val != orig_val:
                any_scaled = True
                break
    
    if not any_scaled:
        return  # No scaling occurred, no warning needed
    
    # Build warning message
    lines = [
        "\n" + "!" * 80,
        f"⚠️  WARNING: {scaling_type} has been REDUCED to prevent OOM (Out of Memory) kills",
        "!" * 80,
        "   Original values:",
    ]
    for key, value in original_values.items():
        lines.append(f"      {key}: {value}")
    
    lines.append("   Using scaled values:")
    for key, value in scaled_values.items():
        if key in original_values:
            orig_val = original_values[key]
            if isinstance(orig_val, (int, float)) and isinstance(value, (int, float)):
                if value < orig_val:
                    lines.append(f"      {key}: {value} (reduced from {orig_val})")
                else:
                    lines.append(f"      {key}: {value}")
            else:
                lines.append(f"      {key}: {value}")
        else:
            lines.append(f"      {key}: {value}")
    
    # Impact description
    lines.append("\n   ⚠️  IMPORTANT: This scaling may affect measured speedup of optimizations!")
    if impact_description:
        lines.append(f"      - {impact_description}")
    else:
        lines.append("      - Smaller workloads may not fully demonstrate optimization benefits")
        lines.append("      - Speedup ratios may differ from production-scale workloads")
    
    if recommendation:
        lines.append(f"      - {recommendation}")
    else:
        lines.append("      - For accurate production benchmarks, use larger GPUs or reduce batch sizes manually")
    
    lines.append("!" * 80 + "\n")
    logger.warning("\n".join(lines))
