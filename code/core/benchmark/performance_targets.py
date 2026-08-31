#!/usr/bin/env python3
"""
Performance targets and thresholds for all chapters.
Extracted from README.md Performance Targets Summary.

Peak performance values are loaded from benchmark_peak_results_*.json files
if available (created during setup.sh). If not found, falls back to hardcoded
baseline values.
"""

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import json
import copy
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from core.benchmark.metrics import compute_copy_bandwidth_metrics

# Hints for Nsight Compute counter names when collecting bandwidth/efficiency.
# These are informational; metrics remain numeric (min/target/unit).
_NCU_COUNTER_HINTS = {
    "nvswitch_bandwidth_tbs": "nvlsrx__throughput.avg.pct_of_peak_sustained_elapsed",
    "global_memory_load_efficiency_percent": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_cycle_active",
}

# Default baseline targets (used as fallback if benchmark_peak hasn't run)
# Each chapter can have "description" (str) and "metrics" (dict) keys.
TargetsDict = Dict[str, Dict[str, Any]]

_DEFAULT_TARGETS: TargetsDict = {
    "overall": {
        # Same fallback policy expressed as read+write traffic (formerly payload-only).
        # This arithmetic correction does not certify a measured hardware peak.
        "hbm3e_bandwidth_tbs": {"min": 6.0, "target": 7.0, "unit": "TB/s", "realistic_max": 8.0},
        "fp16_compute_tflops": {"min": 1000, "target": 2000, "unit": "TFLOPS", "realistic_max": 1300},
        "torch_compile_speedup_small": {"min": 1.1, "target": 1.2, "unit": "x"},
        "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
        "flex_attention_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
        "deepseek_l2_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
    },
    "ch01": {
        "description": "Performance Basics",
        "metrics": {}
    },
    "ch02": {
        "description": "Hardware Overview",
        "metrics": {
            # NVLink-C2C bandwidth: ~900 GB/s between Grace and Blackwell (book/ch02.md:41-47)
            "nvlink_bandwidth_gbs": {"min": 650, "target": 725, "unit": "GB/s"},
            # HBM3e bandwidth: ~8 TB/s per GPU (book/ch02.md:194-199)
            # Note: NSYS reports in GB/s, so targets converted (5.5 TB/s = 5500 GB/s)
            "hbm3e_bandwidth_tbs": {"min": 5000, "target": 5500, "unit": "GB/s"},
            # NVLink 5 aggregate: 1.8 TB/s per GPU for NVSwitch fabric (book/ch02.md:392-399)
            "nvswitch_bandwidth_tbs": {"min": 1500, "target": 1800, "unit": "GB/s", "counter_hint": _NCU_COUNTER_HINTS["nvswitch_bandwidth_tbs"]},
        }
    },
    "ch03": {
        "description": "System Setup",
        "metrics": {}
    },
    "ch04": {
        "description": "Distributed Networking (includes ch04:nixl_tier_handoff memory-tier analogue)",
        "metrics": {
            # NCCL all-reduce: 100 GB/s on 800 Gb/s InfiniBand (book/ch04.md:760-779)
            "allreduce_bandwidth_gbs": {"min": 700, "target": 800, "unit": "GB/s"},
            # P2P bandwidth: NVLink 5 provides 900 GB/s per direction (book/ch04.md:392-399)
            "p2p_bandwidth_gbs": {"min": 800, "target": 900, "unit": "GB/s"},
            # Small message latency: single-digit microseconds (book/ch04.md:551-552)
            "small_message_latency_us": {"min": 0, "target": 2, "unit": "μs", "lower_is_better": True},
            # Scaling efficiency: near-linear scaling goal (book/ch04.md:494-504)
            "scaling_efficiency_percent": {"min": 85, "target": 95, "unit": "%"},
            # Topology-aware efficiency: hierarchical NCCL patterns (book/ch04.md:1088-1105)
            "topology_efficiency_percent": {"min": 90, "target": 95, "unit": "%"},
        }
    },
    "ch05": {
        "description": "Storage & I/O",
        "metrics": {}
    },
    "ch06": {
        "description": "CUDA Kernels",
        "metrics": {}
    },
    "ch07": {
        "description": "Memory Access Patterns",
        "metrics": {
            # DRAM throughput: 25% → 90% improvement shown (book/ch07.md:633-643, Table 7-2)
            "dram_throughput_percent": {"min": 80, "target": 90, "unit": "%"},
            # Global Memory Load Efficiency: 28% → 97% improvement (book/ch07.md:633-643, Table 7-2)
            "global_memory_load_efficiency_percent": {"min": 85, "target": 97, "unit": "%", "counter_hint": _NCU_COUNTER_HINTS["global_memory_load_efficiency_percent"]},
            # Achieved occupancy: 42% → 89% improvement (book/ch07.md:993-997, Table 7-4)
            "achieved_occupancy_percent": {"min": 75, "target": 89, "unit": "%"},
            # Shared memory throughput: 52% → 100% improvement (book/ch07.md:1395, Table 7-5)
            "shared_memory_throughput_percent": {"min": 90, "target": 100, "unit": "%"},
        }
    },
    "ch08": {
        "description": "Occupancy & ILP",
        "metrics": {}
    },
    "ch09": {
        "description": "Kernel Fusion",
        "metrics": {}
    },
    "ch10": {
        "description": "Tensor Cores",
        "metrics": {
            # FP8 TFLOPS target (book/ch10.md:75-78)
            "fp8_tflops": {"min": 500, "target": 550, "unit": "TFLOPS"},
            # FP16 TFLOPS target (book/ch10.md:75-78)
            "fp16_tflops": {"min": 500, "target": 550, "unit": "TFLOPS"},
            # Tensor Core utilization: FlashAttention-3 achieves ~75% of peak FP16 FLOPs (book/ch10.md:119-123)
            "tensor_core_utilization_percent": {"min": 60, "target": 75, "unit": "%"},
        }
    },
    "ch11": {
        "description": "Streams & Concurrency",
        "metrics": {
            "stream_speedup": {"min": 1.5, "target": 1.7, "unit": "x"},
            "stream_overlap_percent": {"min": 30, "target": 35, "unit": "%"},
        }
    },
    "ch12": {
        "description": "CUDA Graphs",
        "metrics": {}
    },
    "ch13": {
        "description": "PyTorch Profiling",
        "metrics": {
            "compiled_autograd_speedup": {"min": 1.1, "target": 1.3, "unit": "x"},
        }
    },
    "ch14": {
        "description": "Compiler & Triton",
        "metrics": {
            "torch_compile_speedup_large": {"min": 1.0, "target": 1.3, "unit": "x"},
        }
    },
    "ch15": {
        "description": "Disaggregated Inference",
        # No explicit performance targets documented in book/ch15.md
        "metrics": {}
    },
    "ch16": {
        "description": "Inference Optimization",
        "metrics": {
            # GPTQ/AWQ: ~2x inference speedups documented (book/ch16.md:1370-1394)
            "speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            # FP8 speedup: maintain performance with lower precision
            "fp8_speedup": {"min": 0.90, "target": 1.00, "unit": "x"},
            # Quantization memory reduction: 4x footprint reduction = 75% savings (book/ch16.md:1370-1394)
            "quantization_memory_reduction_percent": {"min": 60, "target": 75, "unit": "%"},
        }
    },
    "ch17": {
        "description": "Disaggregated Prefill/Decode and Routing",
        "metrics": {
            # Routing overhead: target < 1.0 ms (book/ch17.md:119)
            "routing_overhead_ms": {"min": 0, "target": 1.0, "unit": "ms", "lower_is_better": True},
            # Load balance variance: target < 0.1 (book/ch17.md:120)
            "load_balance_variance": {"min": 0, "target": 0.1, "unit": "", "lower_is_better": True},
            # TTFT p99 SLO: < 200-300 ms for interactive services (book/ch17.md:81-88)
            "ttft_p99_ms": {"min": 0, "target": 300, "unit": "ms", "lower_is_better": True},
            # TTFT p99 MLPerf: 450 ms for Llama2 70B (book/ch17.md:86-87)
            "ttft_p99_mlperf_ms": {"min": 0, "target": 450, "unit": "ms", "lower_is_better": True},
            # TPOT p99 MLPerf: 40 ms for Llama2 70B (book/ch17.md:87)
            "tpot_p99_ms": {"min": 0, "target": 40, "unit": "ms", "lower_is_better": True},
        }
    },
    "ch18": {
        "description": "Attention Optimization and Decode Kernels",
        "metrics": {}
    },
    "ch19": {
        "description": "Dynamic and Adaptive Inference",
        "metrics": {
            "adaptive_parallelism_speedup": {"min": 1.05, "target": 1.25, "unit": "x"},
            "dynamic_precision_speedup": {"min": 1.05, "target": 1.20, "unit": "x"},
            "dynamic_quantized_cache_speedup": {"min": 1.05, "target": 1.15, "unit": "x"},
            "fp8_training_speedup": {"min": 1.5, "target": 2.0, "unit": "x"},
            "memory_reduction_percent": {"min": 30, "target": 50, "unit": "%"},
        }
    },
    "ch20": {
        "description": "AI-Assisted Performance Optimization",
        "metrics": {}
    },
}


def _load_peak_benchmark_results(
    search_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Load peak performance results from benchmark_peak_results_*.json."""
    if search_dir is None:
        # Try to find the project root (assume we're in benchmark/)
        current = Path(__file__).parent
        # Go up to repository root (core/benchmark -> core -> repo)
        search_dir = current.parent.parent.parent
    
    # Find the most recent benchmark results file (support both uppercase and lowercase)
    json_files = list(search_dir.glob("benchmark_peak_results_*.json"))
    if not json_files:
        # Fallback to old uppercase pattern for backwards compatibility
        json_files = list(search_dir.glob("BENCHMARK_PEAK_RESULTS_*.json"))
    if not json_files:
        return None, None, None
    
    # Sort by modification time, get most recent
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_file = json_files[0]

    try:
        with open(latest_file, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            if not isinstance(data, dict):
                return (
                    None,
                    f"Failed to read peak benchmark results from {latest_file}: expected JSON object, got {type(data).__name__}",
                    str(latest_file),
                )
            return data, None, str(latest_file)
    except Exception as exc:
        return None, f"Failed to read peak benchmark results from {latest_file}: {exc}", str(latest_file)


def _get_peak_values(search_dir: Optional[Path] = None) -> Tuple[Dict[str, float], List[str], Optional[str], str]:
    """Get measured peak values from benchmark results."""
    benchmark_data, warning, artifact_path = _load_peak_benchmark_results(search_dir)
    warnings: List[str] = []
    if warning:
        warnings.append(warning)
    if not benchmark_data:
        source = "defaults" if artifact_path is None else "defaults_due_to_peak_results_warning"
        return {}, warnings, artifact_path, source

    peak_values = {}
    
    # Extract HBM memory bandwidth (previously hbm3e, now hbm)
    hbm_data = benchmark_data.get("hbm") or benchmark_data.get("hbm3e")  # Support both
    if hbm_data:
        try:
            if not isinstance(hbm_data, dict) or (
                hbm_data.get("accounting_version") != 2
                or hbm_data.get("byte_accounting") != "read_plus_write"
                or hbm_data.get("bandwidth_unit") != "decimal_GB_per_s"
            ):
                raise ValueError("missing read+write accounting version 2 provenance")
            recomputed = compute_copy_bandwidth_metrics(
                hbm_data["payload_bytes"], hbm_data["iterations"], hbm_data["elapsed_ms"]
            )
            for field in ("bytes_per_iteration", "total_bytes", "peak_bandwidth_gbs", "peak_bandwidth_tbs"):
                value = hbm_data[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid {field}")
                if not math.isclose(value, recomputed[field], rel_tol=1e-9, abs_tol=0.0):
                    raise ValueError(f"{field} disagrees with bytes/timing provenance")
            peak_values["hbm_bandwidth_tbs"] = recomputed["peak_bandwidth_tbs"]
            peak_values["hbm3e_bandwidth_tbs"] = recomputed["peak_bandwidth_tbs"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            warnings.append(
                f"Rejected HBM target from {artifact_path}: {exc}. Legacy payload-only artifacts "
                "are preserved but require a new measurement or explicit reviewed arithmetic migration."
            )
    
    # Extract FP4 compute
    if "fp4_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp4_compute"]:
        peak_values["fp4_compute_tflops"] = benchmark_data["fp4_compute"]["peak_tflops"]
    
    # Extract FP6 compute
    if "fp6_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp6_compute"]:
        peak_values["fp6_compute_tflops"] = benchmark_data["fp6_compute"]["peak_tflops"]
    
    # Extract FP8 compute
    if "fp8_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp8_compute"]:
        peak_values["fp8_compute_tflops"] = benchmark_data["fp8_compute"]["peak_tflops"]
    
    # Extract FP16 compute
    if "fp16_compute" in benchmark_data and "peak_tflops" in benchmark_data["fp16_compute"]:
        peak_values["fp16_compute_tflops"] = benchmark_data["fp16_compute"]["peak_tflops"]
    
    # Extract BF16 compute
    if "bf16_compute" in benchmark_data and "peak_tflops" in benchmark_data["bf16_compute"]:
        peak_values["bf16_compute_tflops"] = benchmark_data["bf16_compute"]["peak_tflops"]
    
    # Extract torch.compile speedup
    if "torch_compile" in benchmark_data and "speedup" in benchmark_data["torch_compile"]:
        peak_values["torch_compile_speedup"] = benchmark_data["torch_compile"]["speedup"]

    source = "measured_peak_results"
    if warnings:
        source = "partial_peak_results_with_warnings" if peak_values else "defaults_due_to_peak_results_warning"
    return peak_values, warnings, artifact_path, source


def _get_metrics_section(chapter_entry: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return the metrics sub-dictionary for a chapter if it exists."""
    metrics_section = chapter_entry.get("metrics")
    if isinstance(metrics_section, dict):
        return metrics_section
    return None


def _ensure_metrics_section(chapter_entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Ensure a chapter entry has a metrics dict and return it."""
    metrics_section = _get_metrics_section(chapter_entry)
    if metrics_section is None:
        metrics_section = {}
        chapter_entry["metrics"] = metrics_section
    return metrics_section


def _build_targets(
    search_dir: Optional[Path] = None,
) -> Tuple[TargetsDict, List[str], Optional[str], str]:
    """Build TARGETS dict with measured peak values if available."""
    targets = copy.deepcopy(_DEFAULT_TARGETS)
    peak_values, warnings, artifact_path, source = _get_peak_values(search_dir)

    if not peak_values:
        return targets, warnings, artifact_path, source
    
    # Update overall targets with measured peak values
    # Use peak as target, and set min to 85% of peak
    if "hbm_bandwidth_tbs" in peak_values or "hbm3e_bandwidth_tbs" in peak_values:
        peak_tbs = peak_values.get("hbm_bandwidth_tbs") or peak_values.get("hbm3e_bandwidth_tbs")
        if peak_tbs is not None:
            # Convert TB/s to GB/s to match NSYS output format and chapter target units
            peak_gbs = peak_tbs * 1000.0
            
            targets["overall"]["hbm3e_bandwidth_tbs"] = {
                "min": peak_gbs * 0.85,
                "target": peak_gbs,
                "unit": "GB/s",
                "realistic_max": peak_gbs * 1.1,  # Allow 10% overhead
            }
            # Also update ch02 target (now in GB/s to match NSYS)
            ch02_entry = targets.get("ch02")
            if isinstance(ch02_entry, dict):
                ch02_metrics = _ensure_metrics_section(ch02_entry)
                ch02_metrics["hbm3e_bandwidth_tbs"] = {
                    "min": peak_gbs * 0.85,
                    "target": peak_gbs,
                    "unit": "GB/s",
                }
    
    if "fp4_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp4_compute_tflops"]
        if peak_tflops is not None:
            ch10_entry = targets.get("ch10")
            if isinstance(ch10_entry, dict):
                ch10_metrics = _ensure_metrics_section(ch10_entry)
                ch10_metrics["fp4_tflops"] = {
                    "min": peak_tflops * 0.85,
                    "target": peak_tflops,
                    "unit": "TFLOPS",
                }
    
    if "fp6_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp6_compute_tflops"]
        if peak_tflops is not None:
            ch10_entry = targets.get("ch10")
            if isinstance(ch10_entry, dict):
                ch10_metrics = _ensure_metrics_section(ch10_entry)
                ch10_metrics["fp6_tflops"] = {
                    "min": peak_tflops * 0.85,
                    "target": peak_tflops,
                    "unit": "TFLOPS",
                }
    
    if "fp16_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp16_compute_tflops"]
        if peak_tflops is not None:
            targets["overall"]["fp16_compute_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
                "realistic_max": peak_tflops * 1.1,
            }
            ch10_entry = targets.get("ch10")
            if isinstance(ch10_entry, dict):
                ch10_metrics = _ensure_metrics_section(ch10_entry)
                ch10_metrics["fp16_tflops"] = {
                    "min": peak_tflops * 0.85,
                    "target": peak_tflops,
                    "unit": "TFLOPS",
                }
    
    if "bf16_compute_tflops" in peak_values:
        peak_tflops = peak_values["bf16_compute_tflops"]
        if peak_tflops is not None:
            targets["overall"]["bf16_compute_tflops"] = {
                "min": peak_tflops * 0.85,
                "target": peak_tflops,
                "unit": "TFLOPS",
                "realistic_max": peak_tflops * 1.1,
            }
            ch10_entry = targets.get("ch10")
            if isinstance(ch10_entry, dict):
                ch10_metrics = _ensure_metrics_section(ch10_entry)
                ch10_metrics["bf16_tflops"] = {
                    "min": peak_tflops * 0.85,
                    "target": peak_tflops,
                    "unit": "TFLOPS",
                }
    
    if "fp8_compute_tflops" in peak_values:
        peak_tflops = peak_values["fp8_compute_tflops"]
        if peak_tflops is not None:
            ch10_entry = targets.get("ch10")
            if isinstance(ch10_entry, dict):
                ch10_metrics = _ensure_metrics_section(ch10_entry)
                ch10_metrics["fp8_tflops"] = {
                    "min": peak_tflops * 0.85,
                    "target": peak_tflops,
                    "unit": "TFLOPS",
                }
    
    if "torch_compile_speedup" in peak_values:
        peak_speedup = peak_values["torch_compile_speedup"]
        if peak_speedup is not None:
            targets["overall"]["torch_compile_speedup_small"] = {
                "min": peak_speedup * 0.85,
                "target": peak_speedup,
                "unit": "x",
            }
            targets["overall"]["torch_compile_speedup_large"] = {
                "min": peak_speedup * 0.85,
                "target": peak_speedup,
                "unit": "x",
            }
            ch14_entry = targets.get("ch14")
            if isinstance(ch14_entry, dict):
                ch14_metrics = _ensure_metrics_section(ch14_entry)
                ch14_metrics["torch_compile_speedup_large"] = {
                    "min": peak_speedup * 0.85,
                    "target": peak_speedup,
                    "unit": "x",
                }
    
    return targets, warnings, artifact_path, source


# Build TARGETS with measured peak values (if available)
TARGETS, TARGETS_BUILD_WARNINGS, TARGETS_ARTIFACT_PATH, TARGETS_SOURCE = _build_targets()


def get_targets_metadata() -> Dict[str, Any]:
    """Return provenance and warning metadata for target construction."""
    return {
        "source": TARGETS_SOURCE,
        "artifact_path": TARGETS_ARTIFACT_PATH,
        "warnings": list(TARGETS_BUILD_WARNINGS),
    }


def _read_only_metrics(chapter: str) -> Dict[str, Dict[str, Any]]:
    """Helper to access a chapter's metrics as a plain dict."""
    chapter_lower = chapter.lower()
    chapter_entry = TARGETS.get(chapter_lower)
    if not isinstance(chapter_entry, dict):
        return {}
    if chapter_lower == "overall":
        return {
            key: value
            for key, value in chapter_entry.items()
            if key != "description" and isinstance(value, dict)
        }
    metrics_section = _get_metrics_section(chapter_entry)
    return dict(metrics_section) if metrics_section else {}


def get_target(chapter: str, metric: str) -> Dict[str, Any]:
    """Get target definition for a specific chapter/metric."""
    metrics_section = _read_only_metrics(chapter)
    result = metrics_section.get(metric)
    return dict(result) if isinstance(result, dict) else {}


def get_all_chapters() -> List[str]:
    """Get list of all chapters with targets."""
    return [chapter for chapter in TARGETS.keys() if chapter != "overall"]


def get_chapter_description(chapter: str) -> str:
    """Get chapter description."""
    chapter_lower = chapter.lower()
    if chapter_lower not in TARGETS:
        return ""
    if chapter_lower == "overall":
        return "Overall System Performance"
    desc = TARGETS[chapter_lower].get("description", "")
    # desc can be str or other types, but we always return str
    return str(desc) if desc else ""


def get_chapter_metrics(chapter: str) -> Dict[str, Dict[str, Any]]:
    """Get all metrics for a chapter."""
    return _read_only_metrics(chapter)


def compute_status(actual: float, target_def: Dict[str, Any]) -> str:
    """
    Compute status (PASS/WARN/FAIL) based on actual value and target.
    
    Args:
        actual: Measured value
        target_def: Target definition with 'min', 'target', 'lower_is_better' fields
    
    Returns:
        Status string: "PASS", "WARN", or "FAIL"
    """
    if not target_def or "target" not in target_def:
        return "UNKNOWN"
    
    target_value = target_def["target"]
    min_value = target_def.get("min", target_value * 0.85)
    lower_is_better = target_def.get("lower_is_better", False)
    
    if lower_is_better:
        # For metrics where lower is better (latency, overhead)
        if actual <= target_value:
            return "PASS"
        elif actual <= min_value or actual <= target_value * 1.15:
            return "WARN"
        else:
            return "FAIL"
    else:
        # For metrics where higher is better (throughput, speedup)
        if actual >= target_value or actual >= min_value:
            return "PASS"
        elif actual >= target_value * 0.85:
            return "WARN"
        else:
            return "FAIL"


def format_value(value: float, unit: str) -> str:
    """Format a value with its unit."""
    if unit == "x":
        return f"{value:.2f}x"
    elif unit == "%":
        return f"{value:.1f}%"
    elif "FLOPS" in unit:
        return f"{value:.0f} {unit}"
    elif unit in ["GB/s", "TB/s"]:
        return f"{value:.2f} {unit}"
    elif unit in ["ms", "μs"]:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.2f} {unit}" if unit else f"{value:.2f}"
