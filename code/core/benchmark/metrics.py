"""Benchmark metrics helpers for domain-specific performance analysis.

This module provides easy-to-use helpers for computing performance metrics
that help understand WHY optimizations work and HOW to improve performance.

Each chapter has specific metrics that matter. Use these helpers in your
benchmark's get_custom_metrics() method.

Usage:
    from core.benchmark.metrics import (
        compute_memory_metrics,
        compute_compute_metrics,
        compute_stream_metrics,
        ...
    )
    
    def get_custom_metrics(self) -> Optional[dict]:
        return compute_memory_metrics(
            bytes_transferred=self.N * 4,
            elapsed_ms=self._last_elapsed_ms,
        )
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Dict, Optional, Any


# =============================================================================
# Hardware Specifications (Blackwell B200 defaults)
# =============================================================================

@dataclass(frozen=True)
class HardwareSpecs:
    """Per-GPU theoretical peaks; Tensor Core rates use dense arithmetic.

    Link rates are aggregate one-way payload bandwidth per GPU, not per link.
    These profiles describe named SKUs, not measured performance guarantees.
    """
    name: str
    hbm_bandwidth_gbps: float      # HBM3e bandwidth in GB/s
    pcie_bandwidth_gbps: float     # PCIe Gen5 x16 bandwidth in GB/s
    nvlink_bandwidth_gbps: float   # Per-GPU one-way NVLink aggregate, decimal GB/s
    fp32_tflops: float             # Peak FP32 TFLOPS
    fp16_tflops: float             # Peak FP16 TFLOPS
    fp8_tflops: float              # Dense FP8 Tensor Core TFLOPS
    tensor_tflops: float           # Dense FP16/BF16 Tensor Core TFLOPS
    num_sms: int                   # Number of Streaming Multiprocessors
    shared_mem_per_sm_kb: float    # Shared memory per SM in KB
    profile_source: str = "published_static_profile"


# NVIDIA HGX B200 table: eight GPUs, FP8=72 PF/s sparse,
# FP16/BF16=36 PF/s sparse, FP32=600 TF/s. Divide by eight and
# by two for sparse Tensor Core figures only. This is not GB200 NVL72.
# https://www.nvidia.com/en-us/data-center/hgx/
BLACKWELL_B200 = HardwareSpecs(
    name="NVIDIA B200",
    hbm_bandwidth_gbps=8000.0,     # ~8 TB/s HBM3e
    pcie_bandwidth_gbps=64.0,      # PCIe Gen5 x16
    nvlink_bandwidth_gbps=900.0,   # NVLink5 one-way aggregate per GPU
    fp32_tflops=75.0,              # Non-tensor FP32
    fp16_tflops=150.0,             # Packed non-tensor FP16 (two operations per FP32 lane)
    fp8_tflops=4500.0,             # Dense FP8 Tensor Core
    tensor_tflops=2250.0,          # Dense FP16/BF16 Tensor Core
    num_sms=148,
    shared_mem_per_sm_kb=228.0,
)

HOPPER_H100 = HardwareSpecs(
    name="NVIDIA H100",
    hbm_bandwidth_gbps=3350.0,
    pcie_bandwidth_gbps=64.0,
    nvlink_bandwidth_gbps=450.0,
    fp32_tflops=67.0,
    fp16_tflops=134.0,
    fp8_tflops=1979.0,             # H100 SXM dense; sparse is 3958 TF/s
    tensor_tflops=989.0,           # H100 SXM dense, rounded; sparse is 1979 TF/s
    num_sms=132,
    shared_mem_per_sm_kb=228.0,
)

# Default to Blackwell
DEFAULT_SPECS = replace(
    BLACKWELL_B200,
    name="NVIDIA B200 (assumed static profile; no CUDA detection)",
    profile_source="assumed_static_no_cuda",
)


def compute_copy_bandwidth_metrics(
    payload_bytes: int,
    iterations: int,
    elapsed_ms: float,
    *,
    traffic: str = "read_plus_write",
) -> Dict[str, Any]:
    """Convert copy measurements to decimal GB/s with explicit byte semantics.

    Local memory copies count one read and one write. A one-way interconnect
    transfer counts the payload once, not both endpoints' memory accesses.
    See NVIDIA CUDA Best Practices Guide, Effective Bandwidth Calculation.
    """
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes <= 0:
        raise ValueError("payload_bytes must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)) or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be finite and positive")
    if traffic not in {"read_plus_write", "one_way_payload"}:
        raise ValueError(f"Unsupported copy byte accounting: {traffic}")
    multiplier = 2 if traffic == "read_plus_write" else 1
    bytes_per_iteration = payload_bytes * multiplier
    total_bytes = bytes_per_iteration * iterations
    bandwidth_gbs = total_bytes / (elapsed_ms / 1000.0) / 1e9
    return {
        "peak_bandwidth_gbs": bandwidth_gbs,
        "peak_bandwidth_tbs": bandwidth_gbs / 1000.0,
        "byte_accounting": traffic,
        "bandwidth_unit": "decimal_GB_per_s",
        "accounting_version": 2,
        "payload_bytes": payload_bytes,
        "bytes_per_iteration": bytes_per_iteration,
        "total_bytes": total_bytes,
        "iterations": iterations,
        "elapsed_ms": elapsed_ms,
    }


def hardware_specs_for_device(name: str, capability: tuple[int, int]) -> HardwareSpecs:
    """Match named supported SKUs; compute capability alone cannot identify peaks."""
    upper_name = name.upper()
    if capability == (10, 0) and re.search(r"\bB200\b", upper_name):
        return BLACKWELL_B200
    if capability == (9, 0) and re.search(r"\bH100\b", upper_name) and (
        "SXM" in upper_name or "HBM3" in upper_name
    ) and "PCIE" not in upper_name and "NVL" not in upper_name:
        return HOPPER_H100
    raise ValueError(
        f"No validated static peak profile for CUDA device {name!r} at {capability[0]}.{capability[1]}; "
        "supply explicit HardwareSpecs for this SKU. Compute capability does not identify its peaks."
    )


def detect_hardware_specs() -> HardwareSpecs:
    """Select a named device profile, or a labeled assumed B200 profile without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return hardware_specs_for_device(props.name, (props.major, props.minor))
    except ImportError:
        pass
    return DEFAULT_SPECS


# =============================================================================
# Chapter 2: Memory Transfer Metrics
# =============================================================================

def compute_memory_transfer_metrics(
    bytes_transferred: float,
    elapsed_ms: Optional[float],
    transfer_type: str = "pcie",  # "pcie", "nvlink", "hbm"
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for memory transfer benchmarks (ch02).
    
    Args:
        bytes_transferred: Number of bytes moved
        elapsed_ms: Time elapsed in milliseconds
        transfer_type: Type of transfer ("pcie", "nvlink", "hbm")
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with bandwidth metrics and efficiency percentages
    """
    specs = specs or detect_hardware_specs()
    # Get theoretical peak for transfer type
    peak_map = {
        "pcie": specs.pcie_bandwidth_gbps,
        "nvlink": specs.nvlink_bandwidth_gbps,
        "hbm": specs.hbm_bandwidth_gbps,
    }
    theoretical_peak = peak_map.get(transfer_type, specs.pcie_bandwidth_gbps)

    metrics = {
        "transfer.bytes": bytes_transferred,
        "transfer.theoretical_peak_gbps": theoretical_peak,
        "transfer.type": 0.0 if transfer_type == "pcie" else (1.0 if transfer_type == "nvlink" else 2.0),
    }
    if elapsed_ms is None:
        return metrics

    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    achieved_gbps = (bytes_transferred / 1e9) / elapsed_s
    efficiency = (achieved_gbps / theoretical_peak) * 100.0 if theoretical_peak > 0 else 0.0
    metrics["transfer.achieved_gbps"] = achieved_gbps
    metrics["transfer.efficiency_pct"] = min(efficiency, 100.0)
    return metrics


# =============================================================================
# Chapter 6: Kernel Fundamentals Metrics
# =============================================================================

def compute_kernel_fundamentals_metrics(
    num_elements: int,
    num_iterations: int = 1,
    expected_bank_conflicts_per_warp: float = 0.0,
    expected_divergent_branches: float = 0.0,
) -> Dict[str, float]:
    """Compute metrics for kernel fundamentals benchmarks (ch06).
    
    Args:
        num_elements: Number of elements processed
        num_iterations: Number of kernel iterations
        expected_bank_conflicts_per_warp: Expected bank conflicts (0 = none, 32 = worst)
        expected_divergent_branches: Expected divergent branches per warp
    
    Returns:
        Dict with kernel characteristic metrics
    """
    return {
        "kernel.elements": float(num_elements),
        "kernel.iterations": float(num_iterations),
        "kernel.expected_bank_conflicts_per_warp": expected_bank_conflicts_per_warp,
        "kernel.expected_divergent_branches": expected_divergent_branches,
        # 32-way bank conflict = worst case
        "kernel.bank_conflict_severity": expected_bank_conflicts_per_warp / 32.0,
    }


# =============================================================================
# Chapter 7: Memory Access Pattern Metrics
# =============================================================================

def compute_memory_access_metrics(
    bytes_requested: float,
    bytes_actually_transferred: float,
    num_transactions: int,
    optimal_transactions: int,
) -> Dict[str, float]:
    """Compute metrics for memory access pattern benchmarks (ch07).
    
    Args:
        bytes_requested: Bytes actually needed by the kernel
        bytes_actually_transferred: Bytes moved over the bus (includes waste)
        num_transactions: Actual memory transactions issued
        optimal_transactions: Theoretical minimum transactions needed
    
    Returns:
        Dict with coalescing and efficiency metrics
    """
    efficiency = (bytes_requested / bytes_actually_transferred) * 100.0 if bytes_actually_transferred > 0 else 0.0
    transaction_efficiency = (optimal_transactions / num_transactions) * 100.0 if num_transactions > 0 else 0.0
    
    return {
        "memory.bytes_requested": bytes_requested,
        "memory.bytes_transferred": bytes_actually_transferred,
        "memory.efficiency_pct": min(efficiency, 100.0),
        "memory.transactions_actual": float(num_transactions),
        "memory.transactions_optimal": float(optimal_transactions),
        "memory.transaction_efficiency_pct": min(transaction_efficiency, 100.0),
        # Coalescing: 100% = perfect, <50% = severe coalescing issues
        "memory.coalescing_quality": min(efficiency, 100.0),
    }


# =============================================================================
# Chapter 8: Optimization Technique Metrics
# =============================================================================

def compute_optimization_metrics(
    baseline_ms: Optional[float],
    optimized_ms: Optional[float],
    technique: str,
    registers_per_thread: int = 0,
    shared_mem_bytes: int = 0,
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for optimization technique benchmarks (ch08).
    
    Args:
        baseline_ms: Baseline execution time
        optimized_ms: Optimized execution time
        technique: Name of optimization technique
        registers_per_thread: Registers used per thread
        shared_mem_bytes: Shared memory used per block
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with optimization effectiveness metrics
    """
    specs = specs or detect_hardware_specs()
    # Estimate occupancy impact
    max_regs_per_sm = 65536  # Typical for modern GPUs
    max_shared_per_sm = specs.shared_mem_per_sm_kb * 1024
    
    # Simple occupancy estimate (actual calculation is more complex)
    reg_limited_blocks = max_regs_per_sm // (registers_per_thread * 256) if registers_per_thread > 0 else 32
    smem_limited_blocks = int(max_shared_per_sm / shared_mem_bytes) if shared_mem_bytes > 0 else 32
    estimated_blocks_per_sm = min(reg_limited_blocks, smem_limited_blocks, 32)
    
    metrics = {
        "optimization.registers_per_thread": float(registers_per_thread),
        "optimization.shared_mem_bytes": float(shared_mem_bytes),
        "optimization.estimated_blocks_per_sm": float(estimated_blocks_per_sm),
    }
    if baseline_ms is not None:
        metrics["optimization.baseline_ms"] = float(baseline_ms)
    if optimized_ms is not None:
        metrics["optimization.optimized_ms"] = float(optimized_ms)
    if baseline_ms is not None and optimized_ms is not None:
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
        improvement_pct = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
        metrics["optimization.speedup"] = speedup
        metrics["optimization.improvement_pct"] = improvement_pct
    return metrics


# =============================================================================
# Chapter 9: Compute-Bound Metrics (Roofline)
# =============================================================================

def compute_roofline_metrics(
    total_flops: float,
    total_bytes: float,
    elapsed_ms: Optional[float],
    precision: str = "fp16",  # "fp32", "fp16", "fp8", "tensor"
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute roofline analysis metrics for compute-bound benchmarks (ch09).
    
    Args:
        total_flops: Total floating-point operations
        total_bytes: Total bytes moved to/from memory
        elapsed_ms: Execution time in milliseconds
        precision: Precision type for peak calculation
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with roofline position and efficiency metrics
    """
    specs = specs or detect_hardware_specs()

    # Arithmetic intensity (FLOPS per byte)
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0
    
    # Peak performance for precision type
    peak_map = {
        "fp32": specs.fp32_tflops,
        "fp16": specs.fp16_tflops,
        "fp8": specs.fp8_tflops,
        "tensor": specs.tensor_tflops,
    }
    peak_tflops = peak_map.get(precision, specs.fp16_tflops)
    
    # Ridge point: where memory and compute rooflines meet
    # ridge_point = peak_tflops / (memory_bandwidth_TB_per_s)
    ridge_point = (peak_tflops * 1000.0) / specs.hbm_bandwidth_gbps
    
    # Classification
    is_compute_bound = arithmetic_intensity > ridge_point

    metrics = {
        "roofline.arithmetic_intensity": arithmetic_intensity,
        "roofline.ridge_point": ridge_point,
        "roofline.peak_tflops": peak_tflops,
        "roofline.peak_gbps": specs.hbm_bandwidth_gbps,
        "roofline.is_compute_bound": 1.0 if is_compute_bound else 0.0,
    }
    if elapsed_ms is None:
        return metrics

    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    achieved_tflops = (total_flops / 1e12) / elapsed_s
    achieved_gbps = (total_bytes / 1e9) / elapsed_s
    memory_ceiling_tflops = (specs.hbm_bandwidth_gbps / 1000.0) * arithmetic_intensity
    if is_compute_bound:
        efficiency = (achieved_tflops / peak_tflops) * 100.0
    else:
        efficiency = (achieved_gbps / specs.hbm_bandwidth_gbps) * 100.0

    metrics["roofline.achieved_tflops"] = achieved_tflops
    metrics["roofline.achieved_gbps"] = achieved_gbps
    metrics["roofline.memory_ceiling_tflops"] = memory_ceiling_tflops
    metrics["roofline.efficiency_pct"] = min(efficiency, 100.0)
    return metrics


def compute_gemm_metrics(
    m: int,
    n: int,
    k: int,
    *,
    elapsed_ms: Optional[float] = None,
    precision: str = "fp32",
    bytes_per_element: int = 4,
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute GEMM workload metrics and optional roofline metrics.

    Args:
        m: GEMM rows in the output matrix.
        n: GEMM columns in the output matrix.
        k: Reduction dimension.
        elapsed_ms: Optional measured latency for achieved-performance metrics.
        precision: Precision/backend used for the GEMM roofline classification.
        bytes_per_element: Storage bytes per matrix element.
        specs: Hardware specs (auto-detected if None).

    Returns:
        Dict with GEMM shape, FLOP/byte totals, and roofline metrics when timing is available.
    """
    total_flops = 2.0 * float(m) * float(n) * float(k)
    total_bytes = float((m * k) + (k * n) + (m * n)) * float(bytes_per_element)
    metrics = {
        "gemm.m": float(m),
        "gemm.n": float(n),
        "gemm.k": float(k),
        "gemm.total_flops": total_flops,
        "gemm.total_bytes": total_bytes,
        "gemm.bytes_per_element": float(bytes_per_element),
    }
    if elapsed_ms is None:
        return metrics
    return {
        **metrics,
        **compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=elapsed_ms,
            precision=precision,
            specs=specs,
        ),
    }


def compute_bandwidth_metrics(
    total_bytes: float,
    elapsed_ms: Optional[float],
    *,
    bandwidth_type: str = "hbm",
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute achieved bandwidth for kernels that only expose byte traffic."""
    metrics = {
        "bandwidth.total_bytes": float(total_bytes),
    }
    if elapsed_ms is None:
        return metrics

    specs = specs or detect_hardware_specs()
    elapsed_s = max(float(elapsed_ms) / 1000.0, 1e-9)
    achieved_gbps = (float(total_bytes) / 1e9) / elapsed_s
    peak_map = {
        "pcie": specs.pcie_bandwidth_gbps,
        "nvlink": specs.nvlink_bandwidth_gbps,
        "hbm": specs.hbm_bandwidth_gbps,
    }
    theoretical_peak = peak_map.get(str(bandwidth_type).lower(), specs.hbm_bandwidth_gbps)
    efficiency = (achieved_gbps / theoretical_peak) * 100.0 if theoretical_peak > 0 else 0.0
    return {
        **metrics,
        "bandwidth.elapsed_ms": float(elapsed_ms),
        "bandwidth.achieved_gbps": achieved_gbps,
        "bandwidth.theoretical_peak_gbps": theoretical_peak,
        "bandwidth.efficiency_pct": min(efficiency, 100.0),
    }


def compute_reduction_metrics(
    num_elements: int,
    elapsed_ms: Optional[float],
    *,
    bytes_per_element: int = 4,
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute reduction workload size plus optional achieved bandwidth."""
    total_bytes = float(num_elements) * float(bytes_per_element)
    return {
        "reduction.num_elements": float(num_elements),
        "reduction.bytes_per_element": float(bytes_per_element),
        "reduction.total_bytes": total_bytes,
        **compute_bandwidth_metrics(
            total_bytes=total_bytes,
            elapsed_ms=elapsed_ms,
            bandwidth_type="hbm",
            specs=specs,
        ),
    }


# =============================================================================
# Chapter 11: CUDA Stream Metrics
# =============================================================================

def compute_stream_metrics(
    sequential_time_ms: Optional[float],
    overlapped_time_ms: Optional[float],
    num_streams: int,
    num_operations: int,
) -> Dict[str, float]:
    """Compute metrics for CUDA stream benchmarks (ch11).
    
    Args:
        sequential_time_ms: Time when running sequentially
        overlapped_time_ms: Time with stream overlap
        num_streams: Number of CUDA streams used
        num_operations: Number of independent operations
    
    Returns:
        Dict with stream overlap efficiency metrics
    """
    theoretical_speedup = num_operations  # Perfect parallelism
    metrics = {
        "stream.num_streams": float(num_streams),
        "stream.num_operations": float(num_operations),
        "stream.theoretical_speedup": float(theoretical_speedup),
    }
    if sequential_time_ms is not None:
        metrics["stream.sequential_ms"] = float(sequential_time_ms)
    if overlapped_time_ms is not None:
        metrics["stream.overlapped_ms"] = float(overlapped_time_ms)
    if sequential_time_ms is not None and overlapped_time_ms is not None:
        time_saved_ms = sequential_time_ms - overlapped_time_ms
        overlap_efficiency = (time_saved_ms / sequential_time_ms) * 100.0 if sequential_time_ms > 0 else 0.0
        actual_speedup = sequential_time_ms / overlapped_time_ms if overlapped_time_ms > 0 else 1.0
        parallelism_efficiency = (actual_speedup / theoretical_speedup) * 100.0 if theoretical_speedup > 0 else 0.0
        metrics["stream.time_saved_ms"] = time_saved_ms
        metrics["stream.overlap_efficiency_pct"] = overlap_efficiency
        metrics["stream.actual_speedup"] = actual_speedup
        metrics["stream.parallelism_efficiency_pct"] = min(parallelism_efficiency, 100.0)
    return metrics


# =============================================================================
# Chapter 12: CUDA Graph Metrics
# =============================================================================

def compute_graph_metrics(
    baseline_launch_overhead_us: Optional[float],
    graph_launch_overhead_us: Optional[float],
    num_nodes: int,
    num_iterations: int,
) -> Dict[str, float]:
    """Compute metrics for CUDA graph benchmarks (ch12).
    
    Args:
        baseline_launch_overhead_us: Per-launch overhead without graphs (microseconds)
        graph_launch_overhead_us: Per-launch overhead with graphs (microseconds)
        num_nodes: Number of nodes in the graph
        num_iterations: Number of graph replays
    
    Returns:
        Dict with graph optimization metrics
    """
    metrics = {
        "graph.num_nodes": float(num_nodes),
        "graph.num_iterations": float(num_iterations),
    }
    if baseline_launch_overhead_us is not None:
        metrics["graph.baseline_launch_us"] = float(baseline_launch_overhead_us)
    if graph_launch_overhead_us is not None:
        metrics["graph.graph_launch_us"] = float(graph_launch_overhead_us)
    if baseline_launch_overhead_us is not None and graph_launch_overhead_us is not None:
        overhead_reduction_us = baseline_launch_overhead_us - graph_launch_overhead_us
        overhead_reduction_pct = (
            (overhead_reduction_us / baseline_launch_overhead_us) * 100.0
            if baseline_launch_overhead_us > 0
            else 0.0
        )
        metrics["graph.overhead_reduction_us"] = overhead_reduction_us
        metrics["graph.overhead_reduction_pct"] = overhead_reduction_pct
        metrics["graph.total_overhead_saved_us"] = overhead_reduction_us * num_iterations
    return metrics


# =============================================================================
# Chapter 13+: Precision and Training Metrics
# =============================================================================

def compute_precision_metrics(
    fp32_time_ms: Optional[float],
    reduced_precision_time_ms: Optional[float],
    precision_type: str,  # "fp32", "fp16", "bf16", "fp8", "fp4", "int8", "int4"
    accuracy_delta: float = 0.0,  # Accuracy loss (if measured)
) -> Dict[str, float]:
    """Compute metrics for precision optimization benchmarks (ch13, ch19).
    
    Args:
        fp32_time_ms: Baseline FP32 execution time
        reduced_precision_time_ms: Reduced precision execution time
        precision_type: Precision mode for this path. Memory reduction factors
            are interpreted relative to FP32 storage.
        accuracy_delta: Change in accuracy (negative = loss)
    
    Returns:
        Dict with precision tradeoff metrics
    """
    # Memory reduction factors
    memory_reduction = {
        "fp32": 1.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 4.0,
        "int4": 8.0,
        "fp8": 4.0,
        "fp4": 8.0,
    }
    reduction_factor = memory_reduction.get(precision_type, 1.0)
    
    metrics = {
        "precision.memory_reduction_factor": reduction_factor,
        "precision.accuracy_delta": accuracy_delta,
    }
    if fp32_time_ms is not None:
        metrics["precision.fp32_ms"] = float(fp32_time_ms)
    if reduced_precision_time_ms is not None:
        metrics["precision.reduced_ms"] = float(reduced_precision_time_ms)
    if fp32_time_ms is not None and reduced_precision_time_ms is not None:
        speedup = fp32_time_ms / reduced_precision_time_ms if reduced_precision_time_ms > 0 else 0.0
        theoretical_speedup = reduction_factor  # Simplified: assumes memory-bound
        speedup_efficiency = (speedup / theoretical_speedup) * 100.0 if theoretical_speedup > 0 else 0.0
        metrics["precision.speedup"] = speedup
        metrics["precision.theoretical_speedup"] = theoretical_speedup
        metrics["precision.speedup_efficiency_pct"] = speedup_efficiency
    return metrics


# =============================================================================
# Chapter 15+: Inference Metrics
# =============================================================================

def compute_inference_metrics(
    ttft_ms: Optional[float],
    tpot_ms: Optional[float],
    total_tokens: int,
    total_requests: int,
    batch_size: int,
    max_batch_size: int,
) -> Dict[str, float]:
    """Compute metrics for inference benchmarks (ch15-ch18).
    
    Args:
        ttft_ms: Time to first token (milliseconds)
        tpot_ms: Time per output token (milliseconds)
        total_tokens: Total tokens generated
        total_requests: Total requests processed
        batch_size: Actual batch size used
        max_batch_size: Maximum supported batch size
    
    Returns:
        Dict with inference performance metrics
    """
    batch_utilization = (batch_size / max_batch_size) * 100.0 if max_batch_size > 0 else 0.0
    metrics = {
        "inference.total_tokens": float(total_tokens),
        "inference.total_requests": float(total_requests),
        "inference.batch_size": float(batch_size),
        "inference.max_batch_size": float(max_batch_size),
        "inference.batch_utilization_pct": batch_utilization,
    }
    if ttft_ms is not None:
        metrics["inference.ttft_ms"] = float(ttft_ms)
    if tpot_ms is not None:
        metrics["inference.tpot_ms"] = float(tpot_ms)
    if ttft_ms is not None and tpot_ms is not None and total_tokens > 0:
        metrics["inference.tokens_per_second"] = (
            total_tokens / (ttft_ms + tpot_ms * total_tokens)
        ) * 1000.0
    return metrics


# =============================================================================
# Chapter 18: Speculative Decoding Metrics
# =============================================================================

def compute_speculative_decoding_metrics(
    draft_tokens: int,
    accepted_tokens: Optional[int],
    draft_time_ms: Optional[float],
    verify_time_ms: Optional[float],
    num_rounds: Optional[int],
) -> Dict[str, float]:
    """Compute metrics for speculative decoding benchmarks (ch18).
    
    Args:
        draft_tokens: Total draft tokens generated
        accepted_tokens: Total tokens accepted after verification
        draft_time_ms: Total time spent in draft phase
        verify_time_ms: Total time spent in verification phase
        num_rounds: Number of draft-verify rounds
    
    Returns:
        Dict with speculative decoding efficiency metrics
    """
    metrics = {
        "speculative.draft_tokens": float(draft_tokens),
    }
    if accepted_tokens is not None:
        metrics["speculative.accepted_tokens"] = float(accepted_tokens)
        rejected_tokens = draft_tokens - accepted_tokens
        acceptance_rate = (accepted_tokens / draft_tokens) * 100.0 if draft_tokens > 0 else 0.0
        waste_pct = (rejected_tokens / draft_tokens) * 100.0 if draft_tokens > 0 else 0.0
        metrics["speculative.rejected_tokens"] = float(rejected_tokens)
        metrics["speculative.acceptance_rate_pct"] = acceptance_rate
        metrics["speculative.waste_pct"] = waste_pct
    if draft_time_ms is not None:
        metrics["speculative.draft_time_ms"] = float(draft_time_ms)
    if verify_time_ms is not None:
        metrics["speculative.verify_time_ms"] = float(verify_time_ms)
    if num_rounds is not None:
        metrics["speculative.num_rounds"] = float(num_rounds)
    if accepted_tokens is not None and num_rounds is not None and num_rounds > 0:
        metrics["speculative.avg_accepted_per_round"] = accepted_tokens / num_rounds
    if draft_time_ms is not None and verify_time_ms is not None and verify_time_ms > 0:
        metrics["speculative.draft_verify_ratio"] = draft_time_ms / verify_time_ms
    return metrics


# =============================================================================
# Chapter 1: Environment Setup Metrics
# =============================================================================

def compute_environment_metrics(
    gpu_count: int,
    gpu_memory_gb: float,
    cuda_version: str = "",
    driver_version: str = "",
    pytorch_version: str = "",
) -> Dict[str, float]:
    """Compute metrics for environment setup benchmarks (ch01).
    
    Args:
        gpu_count: Number of GPUs detected
        gpu_memory_gb: Total GPU memory in GB
        cuda_version: CUDA version string
        driver_version: Driver version string
        pytorch_version: PyTorch version string
    
    Returns:
        Dict with environment configuration metrics
    """
    return {
        "env.gpu_count": float(gpu_count),
        "env.gpu_memory_gb": gpu_memory_gb,
        "env.cuda_major": float(cuda_version.split('.')[0]) if cuda_version else 0.0,
        "env.is_multi_gpu": 1.0 if gpu_count > 1 else 0.0,
        "env.total_memory_gb": gpu_memory_gb * gpu_count,
    }


# =============================================================================
# Chapter 3: System Configuration Metrics
# =============================================================================

def compute_system_config_metrics(
    numa_nodes: int,
    cpu_cores: int,
    memory_channels: int = 0,
    pcie_lanes: int = 0,
    nvlink_connections: int = 0,
) -> Dict[str, float]:
    """Compute metrics for system configuration benchmarks (ch03).
    
    Args:
        numa_nodes: Number of NUMA nodes (0 when topology is unavailable)
        cpu_cores: Total CPU cores
        memory_channels: Memory channels per socket
        pcie_lanes: PCIe lanes available
        nvlink_connections: NVLink connections between GPUs
    
    Returns:
        Dict with system topology metrics
    """
    return {
        "system.numa_nodes": float(numa_nodes),
        "system.cpu_cores": float(cpu_cores),
        "system.memory_channels": float(memory_channels),
        "system.pcie_lanes": float(pcie_lanes),
        "system.nvlink_connections": float(nvlink_connections),
        "system.cores_per_numa": float(cpu_cores / numa_nodes) if numa_nodes > 0 else 0.0,
    }


# =============================================================================
# Chapter 4: Distributed Communication Metrics
# =============================================================================

def compute_distributed_metrics(
    world_size: int,
    bytes_transferred: float,
    elapsed_ms: Optional[float],
    collective_type: str = "allreduce",  # "allreduce", "allgather", "reduce_scatter", "p2p"
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for distributed communication benchmarks (ch04).
    
    Args:
        world_size: Number of processes/GPUs
        bytes_transferred: Bytes moved in the collective
        elapsed_ms: Time elapsed in milliseconds
        collective_type: Type of collective operation
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with distributed communication metrics
    """
    specs = specs or detect_hardware_specs()

    # Theoretical peak (NVLink for multi-GPU)
    theoretical_peak = specs.nvlink_bandwidth_gbps * (world_size - 1)  # Ring topology

    metrics = {
        "distributed.world_size": float(world_size),
        "distributed.bytes_transferred": bytes_transferred,
        "distributed.theoretical_peak_gbps": theoretical_peak,
        "distributed.collective_type": 0.0 if collective_type == "allreduce" else 1.0,
    }
    if elapsed_ms is None:
        return metrics

    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    achieved_gbps = (bytes_transferred / 1e9) / elapsed_s
    efficiency = (achieved_gbps / theoretical_peak) * 100.0 if theoretical_peak > 0 else 0.0
    algo_bw_factor = {
        "allreduce": 2.0 * (world_size - 1) / world_size,
        "allgather": (world_size - 1) / world_size,
        "reduce_scatter": (world_size - 1) / world_size,
        "p2p": 1.0,
    }
    factor = algo_bw_factor.get(collective_type, 1.0)
    metrics["distributed.achieved_gbps"] = achieved_gbps
    metrics["distributed.algo_bandwidth_gbps"] = achieved_gbps * factor
    metrics["distributed.efficiency_pct"] = min(efficiency, 100.0)
    return metrics


# =============================================================================
# Chapter 5: Storage I/O Metrics
# =============================================================================

def compute_storage_io_metrics(
    bytes_read: float,
    bytes_written: float,
    read_time_ms: float,
    write_time_ms: float,
    num_files: int = 1,
    is_async: bool = False,
) -> Dict[str, float]:
    """Compute metrics for storage I/O benchmarks (ch05).
    
    Args:
        bytes_read: Total bytes read from storage
        bytes_written: Total bytes written to storage
        read_time_ms: Time spent reading
        write_time_ms: Time spent writing
        num_files: Number of files accessed
        is_async: Whether async I/O was used
    
    Returns:
        Dict with storage I/O performance metrics
    """
    read_time_s = max(read_time_ms / 1000.0, 1e-9)
    write_time_s = max(write_time_ms / 1000.0, 1e-9)
    
    read_gbps = (bytes_read / 1e9) / read_time_s if bytes_read > 0 else 0.0
    write_gbps = (bytes_written / 1e9) / write_time_s if bytes_written > 0 else 0.0
    
    total_bytes = bytes_read + bytes_written
    total_time_s = read_time_s + write_time_s
    total_gbps = (total_bytes / 1e9) / total_time_s if total_bytes > 0 else 0.0
    
    return {
        "storage.bytes_read": bytes_read,
        "storage.bytes_written": bytes_written,
        "storage.read_gbps": read_gbps,
        "storage.write_gbps": write_gbps,
        "storage.total_gbps": total_gbps,
        "storage.num_files": float(num_files),
        "storage.is_async": 1.0 if is_async else 0.0,
        "storage.read_write_ratio": bytes_read / bytes_written if bytes_written > 0 else 0.0,
    }


# =============================================================================
# Chapter 10: Pipeline Metrics
# =============================================================================

def compute_pipeline_metrics(
    num_stages: int,
    stage_times_ms: list,
    bubble_time_ms: float = 0.0,
    microbatches: int = 1,
) -> Dict[str, float]:
    """Compute metrics for pipeline parallelism benchmarks (ch10).
    
    Args:
        num_stages: Number of pipeline stages
        stage_times_ms: List of execution times per stage
        bubble_time_ms: Time spent in pipeline bubbles
        microbatches: Number of microbatches
    
    Returns:
        Dict with pipeline efficiency metrics
    """
    if not stage_times_ms:
        stage_times_ms = [1.0]
    
    total_stage_time = sum(stage_times_ms)
    max_stage_time = max(stage_times_ms)
    min_stage_time = min(stage_times_ms)
    avg_stage_time = total_stage_time / len(stage_times_ms)
    
    # Pipeline efficiency: ideal vs actual
    ideal_time = max_stage_time * (num_stages + microbatches - 1)
    actual_time = total_stage_time + bubble_time_ms
    efficiency = (ideal_time / actual_time) * 100.0 if actual_time > 0 else 0.0
    
    # Load imbalance
    imbalance = (max_stage_time / min_stage_time) if min_stage_time > 0 else 1.0
    
    # Bubble fraction
    bubble_fraction = bubble_time_ms / actual_time if actual_time > 0 else 0.0
    
    return {
        "pipeline.num_stages": float(num_stages),
        "pipeline.microbatches": float(microbatches),
        "pipeline.max_stage_ms": max_stage_time,
        "pipeline.min_stage_ms": min_stage_time,
        "pipeline.avg_stage_ms": avg_stage_time,
        "pipeline.bubble_time_ms": bubble_time_ms,
        "pipeline.bubble_fraction": bubble_fraction,
        "pipeline.load_imbalance": imbalance,
        "pipeline.efficiency_pct": min(efficiency, 100.0),
    }


# =============================================================================
# Chapter 14: Triton Kernel Metrics
# =============================================================================

def compute_triton_metrics(
    num_elements: int,
    elapsed_ms: Optional[float],
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 2,
    bytes_transferred: float = 0.0,
    specs: Optional[HardwareSpecs] = None,
) -> Dict[str, float]:
    """Compute metrics for Triton kernel benchmarks (ch14).
    
    Args:
        num_elements: Number of elements processed
        elapsed_ms: Execution time in milliseconds
        block_size: Triton block size
        num_warps: Number of warps per block
        num_stages: Software pipeline stages
        bytes_transferred: Bytes moved to/from memory
        specs: Hardware specs (auto-detected if None)
    
    Returns:
        Dict with Triton kernel performance metrics
    """
    specs = specs or detect_hardware_specs()

    # Occupancy estimate
    threads_per_block = num_warps * 32
    blocks_per_sm_estimate = min(32, 2048 // threads_per_block)

    metrics = {
        "triton.num_elements": float(num_elements),
        "triton.block_size": float(block_size),
        "triton.num_warps": float(num_warps),
        "triton.num_stages": float(num_stages),
        "triton.threads_per_block": float(threads_per_block),
        "triton.blocks_per_sm_estimate": float(blocks_per_sm_estimate),
    }
    if elapsed_ms is None:
        return metrics

    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    elements_per_second = num_elements / elapsed_s
    achieved_gbps = (bytes_transferred / 1e9) / elapsed_s if bytes_transferred > 0 else 0.0
    bandwidth_efficiency = (achieved_gbps / specs.hbm_bandwidth_gbps) * 100.0 if achieved_gbps > 0 else 0.0
    metrics["triton.elements_per_second"] = elements_per_second
    metrics["triton.achieved_gbps"] = achieved_gbps
    metrics["triton.bandwidth_efficiency_pct"] = bandwidth_efficiency
    return metrics


# =============================================================================
# Chapter 20: AI-Assisted Optimization Metrics
# =============================================================================

def compute_ai_optimization_metrics(
    original_time_ms: Optional[float],
    ai_optimized_time_ms: Optional[float],
    suggestions_applied: Optional[int],
    suggestions_total: Optional[int],
    code_changes: int = 0,
) -> Dict[str, float]:
    """Compute metrics for AI-assisted optimization benchmarks (ch20).
    
    Args:
        original_time_ms: Original execution time
        ai_optimized_time_ms: Time after AI-suggested optimizations
        suggestions_applied: Number of AI suggestions applied
        suggestions_total: Total AI suggestions generated
        code_changes: Number of code changes made
    
    Returns:
        Dict with AI optimization effectiveness metrics
    """
    metrics = {
        "ai_opt.code_changes": float(code_changes),
    }
    if original_time_ms is not None:
        metrics["ai_opt.original_ms"] = float(original_time_ms)
    if ai_optimized_time_ms is not None:
        metrics["ai_opt.optimized_ms"] = float(ai_optimized_time_ms)
    if original_time_ms is not None and ai_optimized_time_ms is not None:
        speedup = original_time_ms / ai_optimized_time_ms if ai_optimized_time_ms > 0 else 0.0
        improvement_pct = (
            ((original_time_ms - ai_optimized_time_ms) / original_time_ms) * 100.0
            if original_time_ms > 0
            else 0.0
        )
        metrics["ai_opt.speedup"] = speedup
        metrics["ai_opt.improvement_pct"] = improvement_pct
        metrics["ai_opt.improvement_per_change"] = (
            improvement_pct / code_changes if code_changes > 0 else 0.0
        )
    if suggestions_applied is not None:
        metrics["ai_opt.suggestions_applied"] = float(suggestions_applied)
    if suggestions_total is not None:
        metrics["ai_opt.suggestions_total"] = float(suggestions_total)
    if (
        suggestions_applied is not None
        and suggestions_total is not None
        and suggestions_total > 0
    ):
        metrics["ai_opt.acceptance_rate_pct"] = (suggestions_applied / suggestions_total) * 100.0
    return metrics


# =============================================================================
# MoE (Mixture of Experts) Metrics
# =============================================================================

def compute_moe_metrics(
    num_experts: int,
    active_experts: int,
    tokens_per_expert: list,
    routing_time_ms: float,
    expert_compute_time_ms: float,
    load_balance_loss: float = 0.0,
) -> Dict[str, float]:
    """Compute metrics for MoE (Mixture of Experts) benchmarks.
    
    Args:
        num_experts: Total number of experts
        active_experts: Number of experts activated per token (top-k)
        tokens_per_expert: List of token counts per expert
        routing_time_ms: Time spent in routing/gating
        expert_compute_time_ms: Time spent in expert computation
        load_balance_loss: Auxiliary load balancing loss
    
    Returns:
        Dict with MoE efficiency metrics
    """
    if not tokens_per_expert:
        tokens_per_expert = [1]
    
    total_tokens = sum(tokens_per_expert)
    max_tokens = max(tokens_per_expert)
    min_tokens = min(tokens_per_expert)
    avg_tokens = total_tokens / len(tokens_per_expert)
    
    # Load imbalance: 1.0 = perfect balance
    load_imbalance = max_tokens / avg_tokens if avg_tokens > 0 else 1.0
    
    # Expert utilization
    experts_used = sum(1 for t in tokens_per_expert if t > 0)
    utilization = (experts_used / num_experts) * 100.0 if num_experts > 0 else 0.0
    
    # Routing overhead
    total_time = routing_time_ms + expert_compute_time_ms
    routing_overhead = (routing_time_ms / total_time) * 100.0 if total_time > 0 else 0.0
    
    return {
        "moe.num_experts": float(num_experts),
        "moe.active_experts": float(active_experts),
        "moe.total_tokens": float(total_tokens),
        "moe.max_tokens_per_expert": float(max_tokens),
        "moe.min_tokens_per_expert": float(min_tokens),
        "moe.load_imbalance": load_imbalance,
        "moe.expert_utilization_pct": utilization,
        "moe.routing_time_ms": routing_time_ms,
        "moe.expert_compute_time_ms": expert_compute_time_ms,
        "moe.routing_overhead_pct": routing_overhead,
        "moe.load_balance_loss": load_balance_loss,
    }


# =============================================================================
# Generic Helper
# =============================================================================

def compute_speedup_metrics(
    baseline_ms: Optional[float],
    optimized_ms: Optional[float],
    name: str = "",
) -> Dict[str, float]:
    """Compute basic speedup metrics between baseline and optimized.
    
    Args:
        baseline_ms: Baseline execution time
        optimized_ms: Optimized execution time
        name: Optional name prefix for metrics
    
    Returns:
        Dict with speedup and improvement metrics
    """
    prefix = f"{name}." if name else ""
    metrics: Dict[str, float] = {}
    if baseline_ms is not None:
        metrics[f"{prefix}baseline_ms"] = float(baseline_ms)
    if optimized_ms is not None:
        metrics[f"{prefix}optimized_ms"] = float(optimized_ms)
    if baseline_ms is not None and optimized_ms is not None:
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
        improvement_pct = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else 0.0
        metrics[f"{prefix}speedup"] = speedup
        metrics[f"{prefix}improvement_pct"] = improvement_pct
    return metrics


# =============================================================================
# Validation Helper
# =============================================================================

def validate_metrics(metrics: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Validate that metrics dict is well-formed and contains meaningful data.
    
    Args:
        metrics: The metrics dict to validate
    
    Returns:
        Dict with validation results:
        - valid: bool
        - issues: list of issues found
        - stats: dict with statistics about the metrics
    """
    issues = []
    
    if metrics is None:
        return {"valid": False, "issues": ["Metrics is None"], "stats": {}}
    
    if not isinstance(metrics, dict):
        return {"valid": False, "issues": [f"Metrics is {type(metrics).__name__}, expected dict"], "stats": {}}
    
    if len(metrics) == 0:
        return {"valid": False, "issues": ["Metrics dict is empty"], "stats": {}}
    
    # Check for common issues
    all_zeros = all(v == 0.0 for v in metrics.values() if isinstance(v, (int, float)))
    if all_zeros and len(metrics) > 1:
        issues.append("All metric values are zero")
    
    # Check for NaN/Inf
    for key, value in metrics.items():
        if isinstance(value, float):
            if value != value:  # NaN check
                issues.append(f"{key} is NaN")
            elif value == float('inf') or value == float('-inf'):
                issues.append(f"{key} is infinite")
    
    # Check naming convention
    for key in metrics.keys():
        if '.' not in key:
            issues.append(f"Key '{key}' doesn't follow 'category.metric' naming convention")
    
    stats = {
        "num_metrics": len(metrics),
        "num_zero": sum(1 for v in metrics.values() if v == 0.0),
        "num_negative": sum(1 for v in metrics.values() if isinstance(v, (int, float)) and v < 0),
    }
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "stats": stats,
    }
