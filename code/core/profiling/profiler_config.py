"""Centralized profiler configuration.

Single source of truth for profiler command generation and metric sets.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from core.profiling.nvtx_helper import canonicalize_nvtx_name


# =============================================================================
# Metric sets for different profiling scenarios
# Updated with comprehensive stall metrics, tensor core breakdown, and
# register pressure detection for Hopper/Blackwell architectures
# =============================================================================

ROOFLINE_METRICS = [
    # Kernel timing
    "gpu__time_duration.avg",
    # Compute throughput (SM)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - DRAM (HBM)
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - L2 cache
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy - active warps
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # SM cycles active (different perspective from warp active)
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
]

# Complete set of warp stall reasons - critical for understanding "why slow"
WARP_STALL_METRICS = [
    # Primary stall reasons (most common bottlenecks)
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",           # Barrier sync
    "smsp__warp_issue_stalled_dependency_per_warp_active.pct",        # Data dependencies
    "smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct",   # Memory pipeline full
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",   # Waiting for L2/DRAM
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",  # Waiting for L1/shared
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", # Math pipe congestion
    "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct",          # L2/FB miss pressure
    # Extended stall reasons (often overlooked but critical)
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",              # Wait for exec resources
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",      # Memory I/O congestion
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",      # Texture cache pressure
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",             # Pipeline draining
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",       # Local/global throttle
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",    # Instruction fetch stalls
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",          # Intentionally yielded
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",            # Memory barrier stalls
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",      # Scheduler didn't select
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",    # Dispatch unit stalls
    "smsp__warp_issue_stalled_misc_per_warp_active.pct",              # Miscellaneous stalls
]

# Register pressure and local memory spill detection
REGISTER_PRESSURE_METRICS = [
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",   # Local mem loads (SPILLS!)
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",   # Local mem stores (SPILLS!)
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",     # Spill load bytes
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",     # Spill store bytes
]

# Tensor Core metrics broken down by precision
TENSOR_CORE_METRICS = [
    # Overview
    "sm__inst_executed_pipe_tensor.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    # FP16 Tensor Core (HMMA)
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    # INT8 Tensor Core (IMMA)
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_imma.sum",
    # FP8 (Hopper/Blackwell SM 9.0+)
    "sm__inst_executed_pipe_fp8.sum",
    # FP64 MMA
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
]

# Cache and atomic metrics
CACHE_METRICS = [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_rate.pct",
    "lts__t_sectors_op_read_hit_rate.pct",
    "lts__t_sectors_op_write_hit_rate.pct",
    "lts__t_sector_op_atom_hit_rate.pct",  # Atomic hit rate in L2
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
]

# Atomic operation metrics
ATOMIC_METRICS = [
    "lts__t_sectors_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "smsp__sass_thread_inst_executed_op_atom_pred_on.sum",
]

DEEP_DIVE_METRICS = [
    # All roofline metrics
    *ROOFLINE_METRICS,
    # Memory throughput counters (sectors + bytes) for bandwidth + reuse analysis
    "dram__sectors_read.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    # Instruction mix (separated adds/muls/ffma + dtype split)
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    # Tensor Core utilization
    *TENSOR_CORE_METRICS,
    # IPC metric
    "sm__sass_inst_executed_per_cycle.avg",
    # Occupancy proxy already in roofline (sm__warps_active...), keep explicit time
    "gpu__time_duration.sum",
    # Cache and bank conflicts
    *CACHE_METRICS,
    # Register pressure and spills
    *REGISTER_PRESSURE_METRICS,
]

MINIMAL_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

# Comprehensive metrics set - everything for deep analysis
COMPREHENSIVE_METRICS = [
    *DEEP_DIVE_METRICS,
    *WARP_STALL_METRICS,
    *ATOMIC_METRICS,
    # Shared memory detailed
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_shared_op_st.sum",
    # Occupancy limiters
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "sm__ctas_launched.sum",
    # GPC cycles for scaling
    "gpc__cycles_elapsed.max",
]

# =============================================================================
# Chapter-Specific Metric Sets
# =============================================================================
# Use these for targeted profiling of specific optimization techniques.
# Each set focuses on the metrics most relevant to that chapter's topic.
# Updated with comprehensive stall reasons, tensor core breakdown, and
# register pressure detection.

# Ch6: Kernel Fundamentals - Bank conflicts, warp divergence, occupancy
CH6_KERNEL_METRICS = [
    *ROOFLINE_METRICS,
    # Bank conflicts (critical for Ch6)
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Warp divergence
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    # Occupancy details
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_per_block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    *REGISTER_PRESSURE_METRICS,
]

# Ch7: Memory Access - Coalescing, vectorization, cache behavior
CH7_MEMORY_METRICS = [
    *ROOFLINE_METRICS,
    # Coalescing efficiency
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    # L1 cache behavior
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_rate.pct",
    # L2 cache behavior
    "lts__t_sectors_op_read_hit_rate.pct",
    "lts__t_sectors_op_write_hit_rate.pct",
    "lts__t_sector_op_atom_hit_rate.pct",  # Atomic hit rate
    # Transaction counts
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    # DRAM bytes
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # Memory stalls
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
]

# Ch8: Optimization Techniques - COMPLETE stall analysis, ILP, register pressure
CH8_OPTIMIZATION_METRICS = [
    *DEEP_DIVE_METRICS,
    # ALL stall reasons (THE KEY to understanding "why faster")
    *WARP_STALL_METRICS,
    # ILP metrics
    "smsp__inst_executed_per_warp.ratio",
    "sm__sass_inst_executed_per_cycle.avg",
]

# Ch9: Compute-Bound - Tensor Core utilization by precision, arithmetic intensity
CH9_COMPUTE_METRICS = [
    *ROOFLINE_METRICS,
    # ALL Tensor Core metrics (FP8/FP16/INT8/FP64)
    *TENSOR_CORE_METRICS,
    # FMA operations
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # Instruction throughput
    "sm__inst_executed.avg.per_cycle_elapsed",
    "sm__sass_inst_executed_per_cycle.avg",
    # Math pipe stalls
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
]

# Ch10: Pipelining - Pipeline utilization, async copy, barrier stalls
CH10_PIPELINE_METRICS = [
    *ROOFLINE_METRICS,
    # Async copy metrics
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    # Pipeline stalls - complete set
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    # SM utilization
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__ctas_active.avg.pct_of_peak_sustained_active",
    # MIO throttle (memory I/O congestion)
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
]

# Ch11: Streams - Overlap metrics (use nsys primarily)
CH11_STREAM_METRICS = [
    *MINIMAL_METRICS,
    # Basic timing for overlap analysis
    "gpu__time_duration.sum",
    "gpu__time_duration.avg",
    "gpc__cycles_elapsed.max",  # For scaling analysis
]

# Ch12: CUDA Graphs - Launch overhead metrics
CH12_GRAPH_METRICS = [
    *MINIMAL_METRICS,
    # Launch timing
    "gpu__time_duration.avg",
    "gpu__time_duration.sum",
    "sm__ctas_launched.sum",
]

# Ch13-14: PyTorch/Triton - General performance with tensor core breakdown
CH13_PYTORCH_METRICS = [
    *ROOFLINE_METRICS,
    # ALL Tensor Core metrics for FP8/FP16/INT8
    *TENSOR_CORE_METRICS,
    # Memory efficiency
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # Register spills (common in compiled code)
    *REGISTER_PRESSURE_METRICS,
]

# Ch15: Attention Optimization - Texture cache, memory patterns
CH15_ATTENTION_METRICS = [
    *ROOFLINE_METRICS,
    *TENSOR_CORE_METRICS,
    # Texture throttle (relevant for attention)
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    # Memory stalls
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct",
    # Cache behavior
    *CACHE_METRICS,
]

# Ch16: Distributed Training - NVLink, atomics, barriers
CH16_DISTRIBUTED_METRICS = [
    *ROOFLINE_METRICS,
    # Barrier stalls (critical for distributed)
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    # Atomics
    *ATOMIC_METRICS,
    # Memory throughput
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]

# Ch17: Blackwell/Hopper specific metrics
CH17_BLACKWELL_METRICS = [
    *COMPREHENSIVE_METRICS,  # Everything for Blackwell analysis
]

# Mapping from chapter number to metric set
CHAPTER_METRICS = {
    6: CH6_KERNEL_METRICS,
    7: CH7_MEMORY_METRICS,
    8: CH8_OPTIMIZATION_METRICS,
    9: CH9_COMPUTE_METRICS,
    10: CH10_PIPELINE_METRICS,
    11: CH11_STREAM_METRICS,
    12: CH12_GRAPH_METRICS,
    13: CH13_PYTORCH_METRICS,
    14: CH13_PYTORCH_METRICS,  # Same as Ch13
    15: CH15_ATTENTION_METRICS,
    16: CH16_DISTRIBUTED_METRICS,
    17: CH17_BLACKWELL_METRICS,
}

# Convenience mapping for metric set names
METRIC_SET_MAP = {
    "minimal": MINIMAL_METRICS,
    "roofline": ROOFLINE_METRICS,
    "deep_dive": DEEP_DIVE_METRICS,
    "comprehensive": COMPREHENSIVE_METRICS,
    "stalls": WARP_STALL_METRICS,
    "tensor_core": TENSOR_CORE_METRICS,
    "cache": CACHE_METRICS,
    "atomics": ATOMIC_METRICS,
    "register_pressure": REGISTER_PRESSURE_METRICS,
}


def get_chapter_metrics(chapter: int) -> List[str]:
    """Get the recommended ncu metrics for a specific chapter.
    
    Args:
        chapter: Chapter number (1-20)
    
    Returns:
        List of ncu metric names appropriate for that chapter's topic
    """
    return CHAPTER_METRICS.get(chapter, ROOFLINE_METRICS)


def resolve_ncu_metrics(metric_set: str, chapter: Optional[int] = None) -> List[str]:
    """Resolve an NCU metrics list from a metric set and optional chapter."""
    metric_set_norm = str(metric_set or "deep_dive").lower()
    if metric_set_norm == "auto":
        if chapter is not None:
            return get_chapter_metrics(chapter)
        return DEEP_DIVE_METRICS
    if metric_set_norm == "roofline":
        return ROOFLINE_METRICS
    if metric_set_norm == "minimal":
        return MINIMAL_METRICS
    if metric_set_norm == "deep_dive":
        return DEEP_DIVE_METRICS
    return DEEP_DIVE_METRICS


NCU_SET_BY_METRIC = {
    "deep_dive": "full",
    "roofline": "roofline",
    "minimal": "basic",
    "comprehensive": "full",
    "stalls": "full",
    "tensor_core": "full",
}

NCU_REPLAY_MODES = ("kernel", "application", "app-range")


def validate_ncu_replay_mode(mode: str) -> str:
    """Validate the replay modes supported by repository report consumers."""
    normalized = str(mode).strip().lower()
    if normalized not in NCU_REPLAY_MODES:
        raise ValueError(
            f"Invalid Nsight Compute replay mode '{mode}'. "
            f"Choose from {', '.join(NCU_REPLAY_MODES)}."
        )
    return normalized


def validate_ncu_app_range_capture(
    *,
    nvtx_includes: Sequence[str],
    metrics: Sequence[str],
    kernel_filter: Optional[str] = None,
    launch_skip: Optional[int] = None,
    launch_count: Optional[int] = None,
    sampling_interval: Optional[int] = None,
    profile_from_start: Optional[str] = None,
) -> None:
    """Keep app-range captures within the validated full-range metric contract."""
    if len(nvtx_includes) != 1 or not nvtx_includes[0].strip():
        raise ValueError("app-range requires exactly one explicit NVTX start/stop range.")
    if len(metrics) != len(MINIMAL_METRICS) or set(metrics) != set(MINIMAL_METRICS):
        raise ValueError("app-range requires the five minimal Nsight Compute metrics.")
    if kernel_filter or launch_skip is not None or launch_count is not None or sampling_interval is not None:
        raise ValueError("app-range requires the complete NVTX range without kernel, launch, or sampling limits.")
    if profile_from_start is not None and str(profile_from_start).strip().lower() != "on":
        raise ValueError("app-range requires profiling from the start of the selected NVTX range.")


@dataclass
class ProfilerConfig:
    """Configuration for profiler execution."""
    
    metric_set: str = "minimal"  # "roofline", "deep_dive", or "minimal"
    preset: str = "minimal"  # "minimal" (low overhead) or "deep_dive"
    nvtx_includes: Optional[List[str]] = None
    nsys_trace_types: str = "cuda,nvtx,osrt,cublas,cudnn"
    nsys_trace_types_minimal: str = "cuda,nvtx,osrt"
    nsys_sample_cpu: bool = True
    nsys_python_sampling: bool = True
    nsys_python_sampling_frequency: int = 1000
    nsys_cudabacktrace: bool = True
    nsys_stats: bool = True
    nsys_backtrace: str = "none"
    ncu_replay_mode: str = "kernel"  # See NCU_REPLAY_MODES.
    honor_replay_mode_in_minimal: bool = False
    pm_sampling_interval: Optional[int] = None
    
    def get_nsys_command(
        self, 
        output_path: str, 
        python_script: str,
        python_executable: Optional[str] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate nsys command with configured flags.
        
        Args:
            output_path: Path for nsys output (without .nsys-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
            nvtx_includes: Optional NVTX range filters to include
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        
        preset = (self.preset or "minimal").lower()
        nvtx_filters = list(dict.fromkeys(nvtx_includes or self.nvtx_includes or []))
        trace_types = (
            self.nsys_trace_types_minimal
            if preset == "minimal"
            else self.nsys_trace_types
        )

        cmd = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-o",
            output_path,
            "-t", trace_types,
        ]
        
        if self.nsys_sample_cpu:
            cmd.extend(["-s", "cpu"])
        
        if preset != "minimal" and self.nsys_python_sampling:
            cmd.extend([
                "--python-sampling=true",
                f"--python-sampling-frequency={self.nsys_python_sampling_frequency}",
            ])
        
        if preset != "minimal" and self.nsys_cudabacktrace:
            cmd.append("--cudabacktrace=true")
        
        if preset == "minimal":
            cmd.extend(["--backtrace", self.nsys_backtrace])
        if self.nsys_stats:
            cmd.append("--stats=true")

        # Nsight Systems does not support `--nvtx-include` (that's an Nsight Compute flag).
        # We still trace NVTX ranges, but do not filter by range/domain here.
        
        cmd.append(python_executable or sys.executable)
        cmd.append(python_script)
        
        return cmd
    
    def get_ncu_command(
        self,
        output_path: str,
        python_script: str,
        python_executable: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate ncu command with configured metrics.
        
        Args:
            output_path: Path for ncu output (without .ncu-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
            metrics: Optional list of metrics (defaults to metric_set)
            nvtx_includes: Optional NVTX range filters to include
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        return self.get_ncu_command_for_target(
            output_path,
            [python_executable or sys.executable, python_script],
            metrics=metrics,
            nvtx_includes=nvtx_includes,
        )

    def get_ncu_command_for_target(
        self,
        output_path: str,
        target_command: Sequence[str],
        metrics: Optional[List[str]] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generate ncu command for an arbitrary target command."""
        preset = (self.preset or "minimal").lower()
        metric_set = self.metric_set or "deep_dive"
        metric_set_norm = str(metric_set).lower()
        if metrics is None:
            # "auto" should follow the active profiling preset when the caller did not
            # supply an explicit metrics list.
            if metric_set_norm == "auto":
                if preset == "minimal":
                    metrics = MINIMAL_METRICS
                elif preset == "roofline":
                    metrics = ROOFLINE_METRICS
                else:
                    metrics = DEEP_DIVE_METRICS
            elif metric_set_norm == "roofline":
                metrics = ROOFLINE_METRICS
            elif metric_set_norm == "minimal":
                metrics = MINIMAL_METRICS
            else:
                metrics = DEEP_DIVE_METRICS
        if metric_set_norm == "auto":
            minimal_allow = set(MINIMAL_METRICS)
            minimal_allow.update(
                {
                    "gpu__time_duration.sum",
                    "gpu__time_duration.avg",
                    "gpc__cycles_elapsed.max",
                    "sm__ctas_launched.sum",
                }
            )
            metrics_set = set(metrics or [])
            if metrics_set and metrics_set.issubset(minimal_allow):
                ncu_set = "basic"
            elif metrics_set and metrics_set.issubset(set(ROOFLINE_METRICS)):
                ncu_set = "roofline"
            else:
                ncu_set = "full"
        else:
            ncu_set = NCU_SET_BY_METRIC.get(metric_set_norm, "full")

        cmd = [
            "ncu",
            # We manage GPU clocks at the harness level when possible. On many
            # managed clusters, Nsight Compute cannot lock clocks (permission
            # denied) which causes noisy stderr and can end profiling early.
            "--clock-control",
            "none",
            "--set", ncu_set,
            "--metrics", ",".join(metrics),
        ]

        nvtx_filters = list(dict.fromkeys(nvtx_includes or self.nvtx_includes or []))
        minimal_capture = preset == "minimal" or metric_set_norm == "minimal"
        # NVSHMEM's existing benchmark-local range replay is not a public CLI
        # choice. Preserve its command path while validating public modes.
        requested_replay_mode = str(self.ncu_replay_mode).strip().lower()
        if requested_replay_mode != "range":
            requested_replay_mode = validate_ncu_replay_mode(requested_replay_mode)
        if requested_replay_mode == "app-range":
            validate_ncu_app_range_capture(
                nvtx_includes=nvtx_filters,
                metrics=metrics,
                sampling_interval=self.pm_sampling_interval,
            )
        if minimal_capture:
            replay_mode = (
                requested_replay_mode
                if self.honor_replay_mode_in_minimal or requested_replay_mode == "app-range"
                else "kernel"
            )
            cmd.extend([
                "--replay-mode", replay_mode,
            ])
            if self.pm_sampling_interval:
                cmd.extend(["--pm-sampling-interval", str(self.pm_sampling_interval)])
            cmd.extend(["--target-processes", "all"])
            if not nvtx_filters:
                # Unscoped callers retain the bounded legacy behavior. A filtered
                # benchmark invocation must keep every kernel in the measured range;
                # limiting it to one launch silently drops multi-kernel workloads.
                cmd.extend(["--launch-count", "1"])
        else:
            cmd.extend(["--replay-mode", requested_replay_mode])

        if nvtx_filters:
            cmd.append("--nvtx")
        for tag in nvtx_filters:
            cmd.extend(["--nvtx-include", tag])

        cmd.extend([
            "-o", output_path,
            *target_command,
        ])

        return cmd
    
    def get_proton_command(
        self,
        output_path: str,
        python_script: str,
        python_executable: Optional[str] = None,
    ) -> List[str]:
        """Generate Proton CLI invocation.
        
        Falls back to python -m torch._inductor.tools.proton if the standalone
        `proton` binary is not on PATH.
        """
        import sys
        
        output_arg = output_path if Path(output_path).suffix else f"{output_path}.json"
        if shutil.which("proton"):
            return [
                "proton",
                "profile",
                "--output", output_arg,
                "--python-script", python_script,
            ]
        return [
            python_executable or sys.executable,
            "-m",
            "torch._inductor.tools.proton",
            "profile",
            "--output", output_arg,
            "--python-script", python_script,
        ]
    
    def get_torch_profiler_config(self) -> dict:
        """Get PyTorch profiler configuration.
        
        Returns:
            Dictionary of kwargs for torch.profiler.profile()
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity
        except Exception:
            return {}

        preset = (self.preset or "minimal").lower()
        if preset == "minimal":
            activities = [ProfilerActivity.CUDA]
            record_shapes = False
            profile_memory = False
            with_stack = False
            with_flops = False
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)
            experimental_config = None
            if hasattr(torch._C, "_profiler"):
                try:
                    experimental_config = torch._C._profiler._ExperimentalConfig(
                        verbose=False,
                        profiler_measure_per_kernel=False,
                        use_cuda=True,
                    )
                except Exception:
                    experimental_config = None
        else:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            record_shapes = True
            profile_memory = True
            with_stack = True
            with_flops = True
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1)
            experimental_config = None

        cfg = {
            "activities": activities,
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "with_flops": with_flops,
            "schedule": schedule,
        }
        if experimental_config is not None:
            cfg["experimental_config"] = experimental_config
        return cfg


# Default profiler config instance
_DEFAULT_METRIC_SET = "minimal"
DEFAULT_PROFILER_CONFIG = ProfilerConfig(metric_set=_DEFAULT_METRIC_SET)


def set_default_profiler_metric_set(metric_set: str) -> None:
    """Update the global profiler configuration with a new metric set."""
    normalized = metric_set.strip().lower()
    if normalized not in {"deep_dive", "roofline", "minimal"}:
        raise ValueError(
            f"Unsupported metric set '{metric_set}'. "
            "Choose from 'deep_dive', 'roofline', or 'minimal'."
        )
    global _DEFAULT_METRIC_SET, DEFAULT_PROFILER_CONFIG
    _DEFAULT_METRIC_SET = normalized
    DEFAULT_PROFILER_CONFIG = ProfilerConfig(metric_set=_DEFAULT_METRIC_SET)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _discover_nvtx_includes_from_source(module: Any) -> List[str]:
    if module is None:
        return []
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return []
    try:
        text = Path(module_file).read_text()
    except Exception:
        return []
    matches = re.findall(r'(?:nvtx_range|_nvtx_range)\(\s*[\'"]([^\'"]+)[\'"]', text)
    return [canonicalize_nvtx_name(m) for m in matches if m]


def discover_nvtx_includes(
    benchmark_module: Any,
    benchmark_class: Optional[str] = None,
    explicit: Optional[Sequence[str]] = None,
    limit: int = 6,
) -> List[str]:
    """Suggest NVTX filters for nsys/ncu based on benchmark source."""
    includes: List[str] = []
    if benchmark_class:
        includes.append(canonicalize_nvtx_name(benchmark_class))
    includes.extend(explicit or [])
    includes.extend(_discover_nvtx_includes_from_source(benchmark_module))
    
    priority_keywords = (
        "prefill",
        "decode",
        "router",
        "moe",
        "expert",
        "kv",
        "warp",
        "graph",
        "nvlink",
        "nccl",
        "p2p",
    )
    prioritized = [
        name for name in includes
        if any(key in name for key in priority_keywords)
    ]
    ordered = _dedupe_preserve_order(prioritized or includes)
    return ordered[:limit] if limit else ordered


def build_profiler_config_from_benchmark(
    config: Any,
    benchmark_module: Any = None,
    benchmark_class: Optional[str] = None,
) -> ProfilerConfig:
    """Create a ProfilerConfig tuned to the benchmark + harness defaults."""
    preset = str(getattr(config, "profile_type", None) or "minimal").lower()
    metric_set = getattr(config, "ncu_metric_set", None)
    if metric_set is None:
        metric_set = preset if preset in {"minimal", "roofline", "deep_dive"} else "deep_dive"
    else:
        metric_set = str(metric_set)
    sampling_interval = getattr(config, "pm_sampling_interval", None)
    replay_mode = getattr(config, "ncu_replay_mode", None)
    replay_mode_override = bool(getattr(config, "ncu_replay_mode_override", False))
    explicit_includes = getattr(config, "nsys_nvtx_include", None)
    # Fail-fast policy: do not auto-infer NVTX include filters from source code.
    # If the caller wants NVTX filtering, they must explicitly pass
    # BenchmarkConfig.nsys_nvtx_include (MCP/CLI can surface this as a parameter).
    nvtx_includes = list(explicit_includes or [])
    replay_mode_value = str(replay_mode).lower() if replay_mode else "application"
    honor_replay_mode_in_minimal = replay_mode_override or (replay_mode_value != "application")
    return ProfilerConfig(
        metric_set=str(metric_set),
        preset=preset,
        nvtx_includes=nvtx_includes or None,
        pm_sampling_interval=sampling_interval,
        ncu_replay_mode=replay_mode_value,
        honor_replay_mode_in_minimal=honor_replay_mode_in_minimal,
    )
