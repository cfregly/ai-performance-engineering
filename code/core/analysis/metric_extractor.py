#!/usr/bin/env python3
"""
Unified metric extractor for all profiler outputs.
Extracts performance metrics from test outputs, NCU reports, Nsys reports,
PyTorch profiler outputs, and benchmark JSON files.
"""

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Any

# Regex patterns for extracting metrics from text outputs
PATTERNS = {
    # Bandwidth patterns
    "bandwidth_gbs": r"[Bb]andwidth[:\s]+(\d+\.?\d*)\s*GB/s",
    "bandwidth_tbs": r"[Bb]andwidth[:\s]+(\d+\.?\d*)\s*TB/s",
    "nvlink_bandwidth": r"NVLink.*?[Bb]andwidth[:\s]+(\d+\.?\d*)\s*GB/s",
    "hbm_bandwidth": r"HBM.*?[Bb]andwidth[:\s]+(\d+\.?\d*)\s*[GT]B/s",
    
    # FLOPS patterns
    "tflops": r"(\d+\.?\d*)\s*TFLOPS",
    "fp8_tflops": r"FP8.*?(\d+\.?\d*)\s*TFLOPS",
    "fp16_tflops": r"FP16.*?(\d+\.?\d*)\s*TFLOPS",
    
    # Speedup patterns
    "speedup": r"[Ss]peedup[:\s]+(\d+\.?\d*)x",
    "speedup_alt": r"(\d+\.?\d*)x.*?speedup",
    "stream_speedup": r"Stream(?: overlap)? speedup.*?(\d+\.?\d*)x",
    "stream_overlap_percent": r"Stream overlap percent[:\s]+(\d+\.?\d*)\s*%",
    "fp8_speedup": r"FP8 Speedup.*?(\d+\.?\d*)x",
    "fp8_compiled_speedup": r"FP8 Compiled Speedup.*?(\d+\.?\d*)x",
    "tensor_core_utilization_percent": r"Tensor core utilization percent[:\s]+(\d+\.?\d*)\s*%",
    
    # Utilization/efficiency patterns
    "utilization_percent": r"[Uu]tilization[:\s]+(\d+\.?\d*)\s*%",
    "efficiency_percent": r"[Ee]fficiency[:\s]+(\d+\.?\d*)\s*%",
    "occupancy_percent": r"[Oo]ccupancy[:\s]+(\d+\.?\d*)\s*%",
    
    # Latency patterns
    "latency_ms": r"[Ll]atency[:\s]+(\d+\.?\d*)\s*ms",
    "latency_us": r"[Ll]atency[:\s]+(\d+\.?\d*)\s*[uμ]s",
    
    # Overhead/reduction patterns
    "overhead_reduction": r"[Oo]verhead.*?[Rr]eduction[:\s]+(\d+\.?\d*)\s*%",
    "launch_overhead": r"[Ll]aunch.*?[Oo]verhead[:\s]+(\d+\.?\d*)\s*%",
}

BLACKWELL_REQUIRED_CHAPTERS = {"ch02", "ch07", "ch10"}
NON_BLACKWELL_MARKERS = (
    "⚠ Not Blackwell",
    "⚠ Not a Blackwell",
    "NVLink-C2C not available",
    "WARNING: This code is optimized for Blackwell (CC 10.0)",
    "WARNING: This code is optimized for Blackwell",
)


def _supports_blackwell_features(text: str) -> bool:
    """Heuristically determine if the captured output comes from a Blackwell GPU."""
    # Explicit warning markers take precedence
    if any(marker in text for marker in NON_BLACKWELL_MARKERS):
        return False

    # Parse compute capability when available
    cc_match = re.search(r"Compute Capability[:\s]+([0-9]+\.[0-9]+)", text)
    if cc_match:
        compute_capability = cc_match.group(1).strip()
        if compute_capability.startswith("10."):
            return True
        # Grace-Blackwell (12.x) or older architectures are not yet tuned
        return False

    # Fallback: look for explicit GPU names
    if re.search(r"NVIDIA\s+B[23]00", text, re.IGNORECASE):
        return True

    # Conservative default: assume unsupported unless we can prove otherwise
    return False


def extract_from_text(text: str, patterns: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """
    Extract metrics from text using regex patterns.
    
    Args:
        text: Input text to parse
        patterns: Optional custom patterns, defaults to PATTERNS
    
    Returns:
        Dictionary of extracted metrics
    """
    if patterns is None:
        patterns = PATTERNS
    
    metrics = {}
    for metric_name, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            try:
                value = float(match.group(1))
                # Store first match, or overwrite with more specific matches
                if metric_name not in metrics or "alt" not in metric_name:
                    metrics[metric_name] = value
            except (ValueError, IndexError):
                continue
    
    return metrics


def extract_from_test_output(txt_path: Path) -> Dict[str, float]:
    """
    Extract metrics from test output text files.
    
    Args:
        txt_path: Path to test output .txt file
    
    Returns:
        Dictionary of extracted metrics with chapter prefix
    """
    if not txt_path.exists():
        return {}
    
    try:
        text = txt_path.read_text()
    except Exception:
        return {}
    
    # Extract chapter from filename (e.g., ch02_nvlink.txt -> ch02)
    chapter = ""
    filename = txt_path.stem
    if filename.startswith("ch"):
        chapter = filename.split("_")[0]
    
    metrics = extract_from_text(text)
    
    # Many hardware-tuned metrics only make sense on Blackwell (SM 10.x). If we
    # can tell from the log that we're running on a different architecture, mark
    # the metrics as unavailable by returning an empty dict. The analyzer will
    # surface these as "MISSING" instead of false failures.
    if chapter in BLACKWELL_REQUIRED_CHAPTERS and not _supports_blackwell_features(text):
        return {}

    # Add chapter prefix to metric names
    if chapter:
        metrics = {f"{chapter}_{k}": v for k, v in metrics.items()}
    
    return metrics


def extract_from_benchmark_json(json_path: Path) -> Dict[str, float]:
    """
    Extract metrics from benchmark JSON files.
    
    Args:
        json_path: Path to benchmark_peak_results_*.json file
    
    Returns:
        Dictionary of extracted metrics
    """
    if not json_path.exists():
        return {}
    
    try:
        with json_path.open() as f:
            data = json.load(f)
    except Exception:
        return {}
    
    metrics = {}
    
    # Extract HBM memory bandwidth (previously hbm3e, now just hbm)
    if "hbm" in data:
        hbm_data = data["hbm"]
        if "peak_bandwidth_tbs" in hbm_data:
            metrics["hbm_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
            # Also keep old name for compatibility
            metrics["hbm3e_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
        if "peak_utilization_percent" in hbm_data:
            metrics["hbm_utilization_percent"] = hbm_data["peak_utilization_percent"]
            metrics["hbm3e_utilization_percent"] = hbm_data["peak_utilization_percent"]
    elif "hbm3e" in data:
        # Handle legacy hbm3e key
        hbm_data = data["hbm3e"]
        if "peak_bandwidth_tbs" in hbm_data:
            metrics["hbm_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
            metrics["hbm3e_bandwidth_tbs"] = hbm_data["peak_bandwidth_tbs"]
        if "peak_utilization_percent" in hbm_data:
            metrics["hbm_utilization_percent"] = hbm_data["peak_utilization_percent"]
            metrics["hbm3e_utilization_percent"] = hbm_data["peak_utilization_percent"]
    
    # Extract FP4 compute
    if "fp4_compute" in data:
        fp4_data = data["fp4_compute"]
        if "peak_tflops" in fp4_data:
            metrics["fp4_compute_tflops"] = fp4_data["peak_tflops"]
    
    # Extract FP6 compute
    if "fp6_compute" in data:
        fp6_data = data["fp6_compute"]
        if "peak_tflops" in fp6_data:
            metrics["fp6_compute_tflops"] = fp6_data["peak_tflops"]
    
    # Extract FP8 compute
    if "fp8_compute" in data:
        fp8_data = data["fp8_compute"]
        if "peak_tflops" in fp8_data:
            metrics["fp8_compute_tflops"] = fp8_data["peak_tflops"]
    
    # Extract FP16 compute
    if "fp16_compute" in data:
        fp16_data = data["fp16_compute"]
        if "peak_tflops" in fp16_data:
            metrics["fp16_compute_tflops"] = fp16_data["peak_tflops"]
    
    # Extract BF16 compute
    if "bf16_compute" in data:
        bf16_data = data["bf16_compute"]
        if "peak_tflops" in bf16_data:
            metrics["bf16_compute_tflops"] = bf16_data["peak_tflops"]
    
    # Extract torch.compile speedup
    if "torch_compile" in data:
        compile_data = data["torch_compile"]
        if "speedup" in compile_data:
            metrics["torch_compile_speedup"] = compile_data["speedup"]
    
    # Extract L2 cache bandwidth
    if "l2_cache" in data:
        l2_data = data["l2_cache"]
        if "peak_bandwidth_gbs" in l2_data:
            metrics["l2_cache_bandwidth_gbs"] = l2_data["peak_bandwidth_gbs"]
        if "l2_cache_size_mb" in l2_data:
            metrics["l2_cache_size_mb"] = l2_data["l2_cache_size_mb"]
    
    # Extract shared memory info
    if "shared_memory" in data:
        sm_data = data["shared_memory"]
        if "shared_memory_per_sm_kb" in sm_data:
            metrics["shared_memory_per_sm_kb"] = sm_data["shared_memory_per_sm_kb"]
        if "total_shared_memory_mb" in sm_data:
            metrics["total_shared_memory_mb"] = sm_data["total_shared_memory_mb"]
    
    # Extract NVLink bandwidth
    if "nvlink" in data:
        nvlink_data = data["nvlink"]
        if "peak_bandwidth_gbs" in nvlink_data:
            metrics["nvlink_bandwidth_gbs"] = nvlink_data["peak_bandwidth_gbs"]
        if "gpu_count" in nvlink_data:
            metrics["nvlink_gpu_count"] = nvlink_data["gpu_count"]
    
    # Extract GPU hardware info
    if "gpu_hardware" in data:
        hw_data = data["gpu_hardware"]
        # Store key hardware characteristics
        if "num_sms" in hw_data:
            metrics["gpu_num_sms"] = hw_data["num_sms"]
        if "l2_cache_size_kb" in hw_data:
            metrics["gpu_l2_cache_size_kb"] = hw_data["l2_cache_size_kb"]
        if "max_threads_per_block" in hw_data:
            metrics["gpu_max_threads_per_block"] = hw_data["max_threads_per_block"]
    
    return metrics


def extract_from_ncu_report(ncu_rep: Path) -> Dict[str, float]:
    """
    Extract metrics from NCU report files.
    
    Args:
        ncu_rep: Path to .ncu-rep file
    
    Returns:
        Dictionary of extracted metrics
    """
    if not ncu_rep.exists():
        return {}

    # Prefer the unified metrics extractor so we stay consistent across CLI/MCP/dashboard.
    try:
        from core.profiling.metrics_extractor import extract_ncu_metrics
    except Exception:
        return {}

    try:
        metrics_obj = extract_ncu_metrics(ncu_rep)
    except Exception:
        return {}

    metrics: Dict[str, float] = {}
    if metrics_obj.kernel_time_ms is not None:
        metrics["kernel_time_ms"] = metrics_obj.kernel_time_ms
    if metrics_obj.range_time_ms is not None:
        metrics["range_time_ms"] = metrics_obj.range_time_ms
    if metrics_obj.sm_throughput_pct is not None:
        metrics["sm_throughput_percent"] = metrics_obj.sm_throughput_pct
    if metrics_obj.dram_throughput_pct is not None:
        metrics["dram_throughput_percent"] = metrics_obj.dram_throughput_pct
    if metrics_obj.l2_throughput_pct is not None:
        metrics["l2_throughput_percent"] = metrics_obj.l2_throughput_pct
    if metrics_obj.occupancy_pct is not None:
        metrics["occupancy"] = metrics_obj.occupancy_pct
    return metrics


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse NCU CSV output for key metrics."""
    metrics = {}
    
    # Look for key metrics in CSV
    patterns = {
        "sm_throughput_percent": r"sm__throughput.*?,(\d+\.?\d*)",
        "dram_throughput_percent": r"dram__throughput.*?,(\d+\.?\d*)",
        "tensor_core_utilization": r"tensor.*?active.*?,(\d+\.?\d*)",
        "occupancy": r"achieved.*?occupancy.*?,(\d+\.?\d*)",
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, csv_text, re.IGNORECASE)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except (ValueError, IndexError):
                pass
    
    return metrics


def extract_from_nsys_report(nsys_rep: Path) -> Dict[str, float]:
    """
    Extract metrics from Nsys report files.
    
    Args:
        nsys_rep: Path to .nsys-rep or .sqlite file
    
    Returns:
        Dictionary of extracted metrics
    """
    if not nsys_rep.exists():
        return {}

    try:
        from core.profiling.metrics_extractor import extract_nsys_metrics
    except Exception:
        return {}

    try:
        metrics_obj = extract_nsys_metrics(nsys_rep)
    except Exception:
        return {}

    metrics: Dict[str, float] = {}
    if metrics_obj.total_gpu_time_ms is not None:
        metrics["total_gpu_time_ms"] = metrics_obj.total_gpu_time_ms
    return metrics


def _parse_nsys_csv(csv_text: str) -> Dict[str, float]:
    """Parse Nsys CSV output for timing metrics."""
    metrics = {}
    
    # Extract total GPU time
    match = re.search(r"Total GPU Time.*?,(\d+\.?\d*)", csv_text, re.IGNORECASE)
    if match:
        try:
            metrics["total_gpu_time_ms"] = float(match.group(1))
        except (ValueError, IndexError):
            pass
    
    return metrics


def extract_from_pytorch_profile(profile_dir: Path) -> Dict[str, float]:
    """
    Extract metrics from PyTorch profiler output directory.
    
    Args:
        profile_dir: Path to PyTorch profiler output directory
    
    Returns:
        Dictionary of extracted metrics
    """
    if not profile_dir.exists() or not profile_dir.is_dir():
        return {}
    
    metrics = {}
    
    # Check for metadata.json
    metadata_path = profile_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
            
            if "duration_seconds" in metadata:
                metrics["duration_seconds"] = metadata["duration_seconds"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # Metadata file corrupt or missing
    
    # Check for key_averages JSON files
    for json_file in profile_dir.glob("key_averages_*.json"):
        try:
            with json_file.open() as f:
                data = json.load(f)
            
            # Sum CUDA time
            total_cuda_time = sum(
                op.get("cuda_time_total", 0) for op in data if isinstance(op, dict)
            )
            if total_cuda_time > 0:
                metrics["pytorch_cuda_time_us"] = total_cuda_time
            
            # Sum CPU time
            total_cpu_time = sum(
                op.get("cpu_time_total", 0) for op in data if isinstance(op, dict)
            )
            if total_cpu_time > 0:
                metrics["pytorch_cpu_time_us"] = total_cpu_time
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass  # JSON file corrupt or unexpected format
    
    # Check for summary text files
    for txt_file in profile_dir.glob("summary_*.txt"):
        try:
            text = txt_file.read_text()
            text_metrics = extract_from_text(text)
            metrics.update(text_metrics)
        except OSError:
            pass  # File read error
    
    return metrics


def discover_and_extract_all(directory: Path) -> Dict[str, Any]:
    """
    Discover all profiler outputs in a directory and extract metrics.
    
    Args:
        directory: Root directory to search
    
    Returns:
        Dictionary with extracted metrics organized by source
    """
    results: Dict[str, Any] = {
        "test_outputs": {},
        "benchmark": {},
        "ncu": {},
        "nsys": {},
        "pytorch": {},
    }
    
    if not directory.exists():
        return results
    
    # Extract from test output files
    for txt_file in directory.glob("*.txt"):
        if txt_file.stem not in ["SUMMARY", "summary"]:
            metrics = extract_from_test_output(txt_file)
            results["test_outputs"][txt_file.name] = metrics
    
    # Extract from benchmark JSON (support both lowercase and uppercase patterns)
    for json_file in directory.glob("benchmark_peak_results_*.json"):
        metrics = extract_from_benchmark_json(json_file)
        results["benchmark"][json_file.name] = metrics
    # Also check for old uppercase pattern for backwards compatibility
    for json_file in directory.glob("BENCHMARK_PEAK_RESULTS_*.json"):
        if json_file.name not in results["benchmark"]:
            metrics = extract_from_benchmark_json(json_file)
            results["benchmark"][json_file.name] = metrics
    
    # Extract from NCU reports
    for ncu_file in directory.rglob("*.ncu-rep"):
        metrics = extract_from_ncu_report(ncu_file)
        results["ncu"][ncu_file.name] = metrics
    
    # Extract from Nsys reports
    for nsys_file in directory.rglob("*.nsys-rep"):
        metrics = extract_from_nsys_report(nsys_file)
        results["nsys"][nsys_file.name] = metrics

    for sqlite_file in directory.rglob("*.sqlite"):
        if "nsys" in sqlite_file.name or "nsight" in sqlite_file.name.lower():
            metrics = extract_from_nsys_report(sqlite_file)
            results["nsys"][sqlite_file.name] = metrics

    # Include pre-extracted Nsight Systems metrics from harness CSV if present
    csv_path = directory / "nsys_metrics.csv"
    if csv_path.exists():
        try:
            with csv_path.open() as fh:
                reader = csv.DictReader(fh)
                csv_metrics = {}
                for row in reader:
                    metric_name = row.get("metric")
                    value = row.get("value")
                    section = row.get("section", "")
                    if not metric_name or value is None:
                        continue
                    key = f"{section}:{metric_name}" if section else metric_name
                    try:
                        csv_metrics[key] = float(value)
                    except (TypeError, ValueError):
                        continue
                if csv_metrics:
                    results["nsys"][csv_path.name] = csv_metrics
        except (OSError, csv.Error):
            pass  # CSV read error
    
    # Extract from PyTorch profiler outputs
    for pytorch_dir in directory.rglob("pytorch*"):
        if pytorch_dir.is_dir():
            for subdir in pytorch_dir.iterdir():
                if subdir.is_dir():
                    metrics = extract_from_pytorch_profile(subdir)
                    results["pytorch"][subdir.name] = metrics
    
    return results


def flatten_metrics(nested_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten nested results into a single metrics dictionary.
    
    Args:
        nested_results: Results from discover_and_extract_all
    
    Returns:
        Flattened metrics dictionary
    """
    flattened = {}
    
    for source_type, sources in nested_results.items():
        for source_name, metrics in sources.items():
            for metric_name, value in metrics.items():
                # Avoid duplicates by prefixing with source if needed
                key = metric_name
                if key in flattened:
                    key = f"{source_type}_{metric_name}"
                flattened[key] = value
    
    # Alias common metrics that feed multiple chapters
    if "ch07_bandwidth_tbs" in flattened and "ch02_hbm3e_bandwidth_tbs" not in flattened:
        flattened["ch02_hbm3e_bandwidth_tbs"] = flattened["ch07_bandwidth_tbs"]

    return flattened
