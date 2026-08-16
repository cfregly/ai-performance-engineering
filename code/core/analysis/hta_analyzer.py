#!/usr/bin/env python3
"""
HTA (Holistic Trace Analysis) Integration for Performance Engineering.

This module provides integration with Meta's HTA library for analyzing
PyTorch profiler traces and Nsight Systems exports. It extracts:
- Critical path analysis
- Idle time breakdown
- Communication/computation overlap metrics
- Kernel execution patterns
- Memory transfer analysis

Usage:
    from core.analysis.hta_analyzer import HTAAnalyzer
    
    analyzer = HTAAnalyzer()
    results = analyzer.analyze_trace("path/to/trace.json")
    comparison = analyzer.compare_traces("baseline.json", "optimized.json")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import HTA - it's an optional dependency
HTA_AVAILABLE = False
try:
    from hta.trace_analysis import TraceAnalysis
    HTA_AVAILABLE = True
except ImportError:
    logger.debug("HTA (hta-lib) not installed. Install with: pip install hta-lib")
    TraceAnalysis = None  # type: ignore


@dataclass
class KernelStats:
    """Statistics for a single kernel."""
    name: str
    count: int
    total_time_us: float
    avg_time_us: float
    min_time_us: float
    max_time_us: float
    std_dev_us: float
    percentage_of_total: float


@dataclass
class CriticalPathSegment:
    """A segment in the critical path."""
    operation: str
    duration_us: float
    category: str  # "compute", "communication", "memory", "idle"
    device: str
    start_time_us: float
    end_time_us: float


@dataclass
class IdleTimeBreakdown:
    """Breakdown of idle time by cause."""
    total_idle_us: float
    kernel_wait_us: float  # GPU waiting for kernel launch
    memory_wait_us: float  # Waiting for memory transfers
    sync_wait_us: float  # Waiting for synchronization
    other_idle_us: float
    idle_percentage: float


@dataclass
class CommCompOverlap:
    """Communication and computation overlap analysis."""
    total_compute_us: float
    total_comm_us: float
    overlapped_us: float
    overlap_percentage: float
    compute_exposed_us: float  # Compute not overlapped
    comm_exposed_us: float  # Communication not overlapped


@dataclass
class HTATraceAnalysis:
    """Complete HTA analysis results for a single trace."""
    trace_path: str
    total_time_us: float
    
    # Kernel statistics
    top_kernels: List[KernelStats]
    kernel_count: int
    unique_kernel_count: int
    
    # Critical path
    critical_path: List[CriticalPathSegment]
    critical_path_length_us: float
    critical_path_compute_pct: float
    critical_path_comm_pct: float
    critical_path_idle_pct: float
    
    # Idle time analysis
    idle_breakdown: IdleTimeBreakdown
    
    # Communication/computation overlap
    comm_comp_overlap: CommCompOverlap
    
    # Memory analysis
    peak_memory_mb: float
    memory_copy_time_us: float
    memory_copy_count: int
    
    # Raw data for further analysis
    raw_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HTAComparison:
    """Comparison between baseline and optimized traces."""
    baseline_analysis: HTATraceAnalysis
    optimized_analysis: HTATraceAnalysis
    
    # Speedup metrics
    total_speedup: float
    critical_path_speedup: float
    
    # Idle time improvement
    idle_reduction_pct: float
    
    # Overlap improvement
    overlap_improvement_pct: float
    
    # Kernel-level changes
    kernel_speedups: Dict[str, float]  # kernel_name -> speedup
    eliminated_kernels: List[str]
    new_kernels: List[str]
    
    # Recommendations
    insights: List[str]


class HTAAnalyzer:
    """
    Holistic Trace Analyzer for PyTorch profiler and Nsight traces.
    
    Provides critical path analysis, idle time breakdown, and
    communication/computation overlap metrics.
    """
    
    def __init__(self):
        self._hta_available = HTA_AVAILABLE
    
    @property
    def is_available(self) -> bool:
        """Check if HTA library is available."""
        return self._hta_available
    
    def analyze_trace(
        self,
        trace_path: Path,
        top_k: int = 10,
    ) -> Optional[HTATraceAnalysis]:
        """
        Analyze a single trace file.
        
        Args:
            trace_path: Path to PyTorch trace JSON or Nsight trace
            top_k: Number of top kernels to include
        
        Returns:
            HTATraceAnalysis with complete analysis, or None if analysis fails
        """
        trace_path = Path(trace_path)
        if not trace_path.exists():
            logger.error(f"Trace file not found: {trace_path}")
            return None
        
        # Try HTA first if available
        if self._hta_available:
            return self._analyze_with_hta(trace_path, top_k)
        
        # Fallback to manual parsing
        return self._analyze_manually(trace_path, top_k)
    
    def _analyze_with_hta(
        self,
        trace_path: Path,
        top_k: int,
    ) -> Optional[HTATraceAnalysis]:
        """Analyze using HTA library."""
        try:
            # HTA expects a directory containing trace files
            trace_dir = trace_path.parent if trace_path.is_file() else trace_path
            
            analyzer = TraceAnalysis(trace_dir=str(trace_dir))
            
            # Get kernel statistics
            kernel_breakdown = analyzer.get_gpu_kernel_breakdown()
            idle_time_data = analyzer.get_idle_time_breakdown()
            comm_comp_data = analyzer.get_comm_comp_overlap()
            memory_stats = analyzer.get_memory_copy_summary()
            
            # Extract top kernels
            top_kernels = self._extract_top_kernels(kernel_breakdown, top_k)
            
            # Build critical path (HTA provides this)
            critical_path = self._extract_critical_path(analyzer)
            
            # Build idle breakdown
            idle_breakdown = self._build_idle_breakdown(idle_time_data)
            
            # Build comm/comp overlap
            comm_comp = self._build_comm_comp_overlap(comm_comp_data)
            
            # Calculate totals
            total_time_us = sum(k.total_time_us for k in top_kernels)
            critical_path_length = sum(seg.duration_us for seg in critical_path)
            
            return HTATraceAnalysis(
                trace_path=str(trace_path),
                total_time_us=total_time_us,
                top_kernels=top_kernels,
                kernel_count=len(kernel_breakdown),
                unique_kernel_count=len(set(k.name for k in top_kernels)),
                critical_path=critical_path,
                critical_path_length_us=critical_path_length,
                critical_path_compute_pct=self._calc_path_category_pct(critical_path, "compute"),
                critical_path_comm_pct=self._calc_path_category_pct(critical_path, "communication"),
                critical_path_idle_pct=self._calc_path_category_pct(critical_path, "idle"),
                idle_breakdown=idle_breakdown,
                comm_comp_overlap=comm_comp,
                peak_memory_mb=memory_stats.get("peak_memory_mb", 0.0),
                memory_copy_time_us=memory_stats.get("total_copy_time_us", 0.0),
                memory_copy_count=memory_stats.get("copy_count", 0),
                raw_stats={
                    "kernel_breakdown": kernel_breakdown,
                    "idle_time": idle_time_data,
                    "comm_comp": comm_comp_data,
                    "memory": memory_stats,
                },
            )
        except Exception as e:
            logger.warning(f"HTA analysis failed: {e}. Falling back to manual parsing.")
            return self._analyze_manually(trace_path, top_k)
    
    def _analyze_manually(
        self,
        trace_path: Path,
        top_k: int,
    ) -> Optional[HTATraceAnalysis]:
        """
        Manual trace analysis when HTA is not available.
        Supports PyTorch profiler JSON format.
        """
        try:
            with trace_path.open() as f:
                trace_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse trace file: {e}")
            return None
        
        # Handle different trace formats
        events = []
        if isinstance(trace_data, list):
            events = trace_data
        elif isinstance(trace_data, dict):
            events = trace_data.get("traceEvents", [])
            if not events:
                events = trace_data.get("events", [])
        
        if not events:
            logger.error("No trace events found in file")
            return None
        
        # Extract kernel events
        kernel_events = self._extract_kernel_events(events)
        
        # Build kernel statistics
        kernel_stats = self._build_kernel_stats(kernel_events, top_k)
        
        # Build simple critical path (sequential execution)
        critical_path = self._build_simple_critical_path(kernel_events)
        
        # Calculate idle time (gaps between kernels)
        idle_breakdown = self._calculate_idle_time(events)
        
        # Estimate comm/comp overlap
        comm_comp = self._estimate_comm_comp_overlap(events)
        
        # Memory analysis
        memory_info = self._analyze_memory_events(events)
        
        total_time_us = sum(k.total_time_us for k in kernel_stats)
        critical_path_length = sum(seg.duration_us for seg in critical_path)
        
        return HTATraceAnalysis(
            trace_path=str(trace_path),
            total_time_us=total_time_us,
            top_kernels=kernel_stats,
            kernel_count=len(kernel_events),
            unique_kernel_count=len(set(e.get("name", "") for e in kernel_events)),
            critical_path=critical_path,
            critical_path_length_us=critical_path_length,
            critical_path_compute_pct=self._calc_path_category_pct(critical_path, "compute"),
            critical_path_comm_pct=self._calc_path_category_pct(critical_path, "communication"),
            critical_path_idle_pct=self._calc_path_category_pct(critical_path, "idle"),
            idle_breakdown=idle_breakdown,
            comm_comp_overlap=comm_comp,
            peak_memory_mb=memory_info.get("peak_mb", 0.0),
            memory_copy_time_us=memory_info.get("copy_time_us", 0.0),
            memory_copy_count=memory_info.get("copy_count", 0),
            raw_stats={"events_count": len(events)},
        )
    
    def _extract_kernel_events(self, events: List[Dict]) -> List[Dict]:
        """Extract GPU kernel events from trace."""
        kernel_events = []
        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            
            # Identify kernel events
            is_kernel = (
                cat in ("kernel", "cuda_runtime", "gpu_kernel", "cuda") or
                "kernel" in name.lower() or
                event.get("ph") == "X" and event.get("dur", 0) > 0
            )
            
            if is_kernel and event.get("dur", 0) > 0:
                kernel_events.append(event)
        
        return kernel_events
    
    def _build_kernel_stats(
        self,
        kernel_events: List[Dict],
        top_k: int,
    ) -> List[KernelStats]:
        """Build kernel statistics from events."""
        # Group by kernel name
        kernel_times: Dict[str, List[float]] = {}
        for event in kernel_events:
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)  # Duration in microseconds
            kernel_times.setdefault(name, []).append(dur)
        
        # Calculate statistics
        stats = []
        total_time = sum(sum(times) for times in kernel_times.values())
        
        for name, times in kernel_times.items():
            import statistics as stat_lib
            
            avg_time = stat_lib.mean(times)
            std_dev = stat_lib.stdev(times) if len(times) > 1 else 0.0
            
            stats.append(KernelStats(
                name=name,
                count=len(times),
                total_time_us=sum(times),
                avg_time_us=avg_time,
                min_time_us=min(times),
                max_time_us=max(times),
                std_dev_us=std_dev,
                percentage_of_total=(sum(times) / total_time * 100) if total_time > 0 else 0,
            ))
        
        # Sort by total time and return top_k
        stats.sort(key=lambda k: k.total_time_us, reverse=True)
        return stats[:top_k]
    
    def _build_simple_critical_path(
        self,
        kernel_events: List[Dict],
    ) -> List[CriticalPathSegment]:
        """Build a simple critical path from kernel events."""
        if not kernel_events:
            return []
        
        # Sort events by start time
        sorted_events = sorted(kernel_events, key=lambda e: e.get("ts", 0))
        
        segments = []
        for event in sorted_events[:50]:  # Limit to avoid huge paths
            name = event.get("name", "unknown")
            dur = event.get("dur", 0)
            ts = event.get("ts", 0)
            
            # Categorize based on name
            category = "compute"
            if any(x in name.lower() for x in ["memcpy", "copy", "transfer"]):
                category = "memory"
            elif any(x in name.lower() for x in ["nccl", "allreduce", "broadcast", "comm"]):
                category = "communication"
            
            segments.append(CriticalPathSegment(
                operation=name,
                duration_us=dur,
                category=category,
                device="GPU",
                start_time_us=ts,
                end_time_us=ts + dur,
            ))
        
        return segments
    
    def _calculate_idle_time(self, events: List[Dict]) -> IdleTimeBreakdown:
        """Calculate idle time from gaps between events."""
        if not events:
            return IdleTimeBreakdown(0, 0, 0, 0, 0, 0)
        
        # Get GPU events sorted by time
        gpu_events = [e for e in events if e.get("dur", 0) > 0]
        if not gpu_events:
            return IdleTimeBreakdown(0, 0, 0, 0, 0, 0)
        
        gpu_events.sort(key=lambda e: e.get("ts", 0))
        
        total_idle = 0.0
        kernel_wait = 0.0
        memory_wait = 0.0
        sync_wait = 0.0
        
        for i in range(1, len(gpu_events)):
            prev_end = gpu_events[i-1].get("ts", 0) + gpu_events[i-1].get("dur", 0)
            curr_start = gpu_events[i].get("ts", 0)
            gap = curr_start - prev_end
            
            if gap > 0:
                total_idle += gap
                
                # Categorize based on next event type
                next_name = gpu_events[i].get("name", "").lower()
                if "sync" in next_name or "wait" in next_name:
                    sync_wait += gap
                elif "memcpy" in next_name or "copy" in next_name:
                    memory_wait += gap
                else:
                    kernel_wait += gap
        
        total_time = sum(e.get("dur", 0) for e in gpu_events) + total_idle
        idle_pct = (total_idle / total_time * 100) if total_time > 0 else 0
        
        return IdleTimeBreakdown(
            total_idle_us=total_idle,
            kernel_wait_us=kernel_wait,
            memory_wait_us=memory_wait,
            sync_wait_us=sync_wait,
            other_idle_us=total_idle - kernel_wait - memory_wait - sync_wait,
            idle_percentage=idle_pct,
        )
    
    def _estimate_comm_comp_overlap(self, events: List[Dict]) -> CommCompOverlap:
        """Estimate communication/computation overlap."""
        compute_events = []
        comm_events = []
        
        for event in events:
            if event.get("dur", 0) <= 0:
                continue
            
            name = event.get("name", "").lower()
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            
            if any(x in name for x in ["nccl", "allreduce", "broadcast", "reduce_scatter", "all_gather"]):
                comm_events.append((ts, ts + dur))
            elif "kernel" in name or event.get("cat", "") in ("kernel", "cuda"):
                compute_events.append((ts, ts + dur))
        
        total_compute = sum(end - start for start, end in compute_events)
        total_comm = sum(end - start for start, end in comm_events)
        
        # Calculate overlap (simplified)
        overlapped = 0.0
        for c_start, c_end in compute_events:
            for m_start, m_end in comm_events:
                overlap_start = max(c_start, m_start)
                overlap_end = min(c_end, m_end)
                if overlap_end > overlap_start:
                    overlapped += overlap_end - overlap_start
        
        overlap_pct = (overlapped / max(total_comm, 1)) * 100 if total_comm > 0 else 0
        
        return CommCompOverlap(
            total_compute_us=total_compute,
            total_comm_us=total_comm,
            overlapped_us=overlapped,
            overlap_percentage=overlap_pct,
            compute_exposed_us=total_compute - overlapped,
            comm_exposed_us=total_comm - overlapped,
        )
    
    def _analyze_memory_events(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze memory-related events."""
        copy_events = []
        peak_memory = 0.0
        
        for event in events:
            name = event.get("name", "").lower()
            
            if "memcpy" in name or "copy" in name:
                copy_events.append(event.get("dur", 0))
            
            # Look for memory allocation events
            args = event.get("args", {})
            if "bytes" in args:
                mem_mb = args["bytes"] / (1024 * 1024)
                peak_memory = max(peak_memory, mem_mb)
        
        return {
            "peak_mb": peak_memory,
            "copy_time_us": sum(copy_events),
            "copy_count": len(copy_events),
        }
    
    def _extract_top_kernels(
        self,
        kernel_breakdown: Any,
        top_k: int,
    ) -> List[KernelStats]:
        """Extract top kernels from HTA breakdown."""
        # HTA returns different formats depending on version
        stats = []
        if hasattr(kernel_breakdown, "iterrows"):
            # DataFrame format
            for _, row in kernel_breakdown.head(top_k).iterrows():
                stats.append(KernelStats(
                    name=row.get("kernel_name", row.get("name", "unknown")),
                    count=int(row.get("count", 1)),
                    total_time_us=float(row.get("total_time", 0)),
                    avg_time_us=float(row.get("avg_time", 0)),
                    min_time_us=float(row.get("min_time", 0)),
                    max_time_us=float(row.get("max_time", 0)),
                    std_dev_us=float(row.get("std_time", 0)),
                    percentage_of_total=float(row.get("percentage", 0)),
                ))
        return stats
    
    def _extract_critical_path(self, analyzer: Any) -> List[CriticalPathSegment]:
        """Extract critical path from HTA analyzer."""
        try:
            cp_data = analyzer.get_critical_path_summary()
            segments = []
            for entry in cp_data:
                segments.append(CriticalPathSegment(
                    operation=entry.get("name", "unknown"),
                    duration_us=entry.get("duration", 0),
                    category=entry.get("category", "compute"),
                    device=entry.get("device", "GPU"),
                    start_time_us=entry.get("start", 0),
                    end_time_us=entry.get("end", 0),
                ))
            return segments
        except Exception:
            return []
    
    def _build_idle_breakdown(self, idle_data: Any) -> IdleTimeBreakdown:
        """Build idle breakdown from HTA data."""
        if isinstance(idle_data, dict):
            return IdleTimeBreakdown(
                total_idle_us=idle_data.get("total_idle", 0),
                kernel_wait_us=idle_data.get("kernel_wait", 0),
                memory_wait_us=idle_data.get("memory_wait", 0),
                sync_wait_us=idle_data.get("sync_wait", 0),
                other_idle_us=idle_data.get("other", 0),
                idle_percentage=idle_data.get("idle_percentage", 0),
            )
        return IdleTimeBreakdown(0, 0, 0, 0, 0, 0)
    
    def _build_comm_comp_overlap(self, data: Any) -> CommCompOverlap:
        """Build comm/comp overlap from HTA data."""
        if isinstance(data, dict):
            return CommCompOverlap(
                total_compute_us=data.get("compute_time", 0),
                total_comm_us=data.get("comm_time", 0),
                overlapped_us=data.get("overlapped", 0),
                overlap_percentage=data.get("overlap_percentage", 0),
                compute_exposed_us=data.get("compute_exposed", 0),
                comm_exposed_us=data.get("comm_exposed", 0),
            )
        return CommCompOverlap(0, 0, 0, 0, 0, 0)
    
    def _calc_path_category_pct(
        self,
        path: List[CriticalPathSegment],
        category: str,
    ) -> float:
        """Calculate percentage of critical path for a category."""
        if not path:
            return 0.0
        total = sum(seg.duration_us for seg in path)
        category_total = sum(seg.duration_us for seg in path if seg.category == category)
        return (category_total / total * 100) if total > 0 else 0.0
    
    def compare_traces(
        self,
        baseline_path: Path,
        optimized_path: Path,
        top_k: int = 10,
    ) -> Optional[HTAComparison]:
        """
        Compare two traces and generate insights.
        
        Args:
            baseline_path: Path to baseline trace
            optimized_path: Path to optimized trace
            top_k: Number of top kernels to analyze
        
        Returns:
            HTAComparison with full analysis, or None on failure
        """
        baseline = self.analyze_trace(baseline_path, top_k)
        optimized = self.analyze_trace(optimized_path, top_k)
        
        if baseline is None or optimized is None:
            logger.error("Failed to analyze one or both traces")
            return None
        
        # Calculate speedups
        total_speedup = (
            baseline.total_time_us / optimized.total_time_us
            if optimized.total_time_us > 0 else 1.0
        )
        critical_path_speedup = (
            baseline.critical_path_length_us / optimized.critical_path_length_us
            if optimized.critical_path_length_us > 0 else 1.0
        )
        
        # Idle time improvement
        idle_reduction = (
            (baseline.idle_breakdown.idle_percentage - optimized.idle_breakdown.idle_percentage)
            if baseline.idle_breakdown.idle_percentage > 0 else 0.0
        )
        
        # Overlap improvement
        overlap_improvement = (
            optimized.comm_comp_overlap.overlap_percentage -
            baseline.comm_comp_overlap.overlap_percentage
        )
        
        # Kernel-level speedups
        baseline_kernels = {k.name: k.total_time_us for k in baseline.top_kernels}
        optimized_kernels = {k.name: k.total_time_us for k in optimized.top_kernels}
        
        kernel_speedups = {}
        for name, b_time in baseline_kernels.items():
            if name in optimized_kernels and optimized_kernels[name] > 0:
                kernel_speedups[name] = b_time / optimized_kernels[name]
        
        eliminated = [k for k in baseline_kernels if k not in optimized_kernels]
        new_kernels = [k for k in optimized_kernels if k not in baseline_kernels]
        
        # Generate insights
        insights = self._generate_comparison_insights(
            baseline, optimized, total_speedup, idle_reduction, overlap_improvement
        )
        
        return HTAComparison(
            baseline_analysis=baseline,
            optimized_analysis=optimized,
            total_speedup=total_speedup,
            critical_path_speedup=critical_path_speedup,
            idle_reduction_pct=idle_reduction,
            overlap_improvement_pct=overlap_improvement,
            kernel_speedups=kernel_speedups,
            eliminated_kernels=eliminated,
            new_kernels=new_kernels,
            insights=insights,
        )
    
    def _generate_comparison_insights(
        self,
        baseline: HTATraceAnalysis,
        optimized: HTATraceAnalysis,
        total_speedup: float,
        idle_reduction: float,
        overlap_improvement: float,
    ) -> List[str]:
        """Generate actionable insights from comparison."""
        insights = []
        
        # Overall speedup
        if total_speedup > 1.5:
            insights.append(f"✅ Excellent overall speedup of {total_speedup:.2f}x")
        elif total_speedup > 1.1:
            insights.append(f"👍 Good improvement of {total_speedup:.2f}x")
        elif total_speedup < 0.95:
            insights.append(f"⚠️ Regression detected: {total_speedup:.2f}x (optimized is slower)")
        
        # Idle time
        if idle_reduction > 10:
            insights.append(f"✅ Reduced idle time by {idle_reduction:.1f}%")
        elif baseline.idle_breakdown.idle_percentage > 20:
            insights.append(
                f"⚠️ High idle time ({baseline.idle_breakdown.idle_percentage:.1f}% -> "
                f"{optimized.idle_breakdown.idle_percentage:.1f}%). Consider kernel fusion or pipelining."
            )
        
        # Communication overlap
        if overlap_improvement > 10:
            insights.append(f"✅ Improved communication overlap by {overlap_improvement:.1f}%")
        elif optimized.comm_comp_overlap.overlap_percentage < 50 and optimized.comm_comp_overlap.total_comm_us > 0:
            insights.append(
                f"⚠️ Low communication overlap ({optimized.comm_comp_overlap.overlap_percentage:.1f}%). "
                "Consider overlapping AllReduce with backward pass."
            )
        
        # Critical path
        cp_compute_change = optimized.critical_path_compute_pct - baseline.critical_path_compute_pct
        if cp_compute_change > 10:
            insights.append(
                f"✅ Critical path is more compute-dominated (+{cp_compute_change:.1f}% compute)"
            )
        
        # Memory analysis
        if baseline.memory_copy_time_us > 0:
            mem_speedup = baseline.memory_copy_time_us / max(optimized.memory_copy_time_us, 1)
            if mem_speedup > 1.5:
                insights.append(f"✅ Memory transfer time reduced by {mem_speedup:.1f}x")
            elif mem_speedup < 0.8:
                insights.append("⚠️ Memory transfer overhead increased")
        
        # Kernel count change
        if baseline.kernel_count > optimized.kernel_count * 1.2:
            fusion_pct = (1 - optimized.kernel_count / baseline.kernel_count) * 100
            insights.append(f"✅ Kernel fusion detected ({fusion_pct:.0f}% fewer kernels)")
        
        if not insights:
            insights.append("No significant changes detected. Consider more aggressive optimizations.")
        
        return insights
    
    def generate_markdown_report(
        self,
        analysis: HTATraceAnalysis,
    ) -> str:
        """Generate markdown report from HTA analysis."""
        lines = []
        lines.append("# HTA Trace Analysis Report")
        lines.append("")
        lines.append(f"**Trace:** `{analysis.trace_path}`")
        lines.append(f"**Total Time:** {analysis.total_time_us / 1000:.2f} ms")
        lines.append(f"**Kernel Count:** {analysis.kernel_count} ({analysis.unique_kernel_count} unique)")
        lines.append("")
        
        # Top Kernels
        lines.append("## Top Kernels")
        lines.append("")
        lines.append("| Kernel | Count | Total (ms) | Avg (µs) | % of Total |")
        lines.append("|--------|-------|------------|----------|------------|")
        for k in analysis.top_kernels[:10]:
            lines.append(
                f"| {k.name[:40]} | {k.count} | {k.total_time_us/1000:.3f} | "
                f"{k.avg_time_us:.1f} | {k.percentage_of_total:.1f}% |"
            )
        lines.append("")
        
        # Critical Path
        lines.append("## Critical Path Analysis")
        lines.append("")
        lines.append(f"- **Length:** {analysis.critical_path_length_us / 1000:.2f} ms")
        lines.append(f"- **Compute:** {analysis.critical_path_compute_pct:.1f}%")
        lines.append(f"- **Communication:** {analysis.critical_path_comm_pct:.1f}%")
        lines.append(f"- **Idle:** {analysis.critical_path_idle_pct:.1f}%")
        lines.append("")
        
        # Idle Time
        lines.append("## Idle Time Breakdown")
        lines.append("")
        idle = analysis.idle_breakdown
        lines.append(f"- **Total Idle:** {idle.total_idle_us / 1000:.2f} ms ({idle.idle_percentage:.1f}%)")
        lines.append(f"- **Kernel Wait:** {idle.kernel_wait_us / 1000:.2f} ms")
        lines.append(f"- **Memory Wait:** {idle.memory_wait_us / 1000:.2f} ms")
        lines.append(f"- **Sync Wait:** {idle.sync_wait_us / 1000:.2f} ms")
        lines.append("")
        
        # Comm/Comp Overlap
        cc = analysis.comm_comp_overlap
        if cc.total_comm_us > 0:
            lines.append("## Communication/Computation Overlap")
            lines.append("")
            lines.append(f"- **Total Compute:** {cc.total_compute_us / 1000:.2f} ms")
            lines.append(f"- **Total Communication:** {cc.total_comm_us / 1000:.2f} ms")
            lines.append(f"- **Overlapped:** {cc.overlapped_us / 1000:.2f} ms ({cc.overlap_percentage:.1f}%)")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_comparison_markdown(
        self,
        comparison: HTAComparison,
    ) -> str:
        """Generate markdown report from comparison."""
        lines = []
        lines.append("# HTA Trace Comparison Report")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Baseline | Optimized | Change |")
        lines.append("|--------|----------|-----------|--------|")
        lines.append(
            f"| Total Time (ms) | {comparison.baseline_analysis.total_time_us/1000:.2f} | "
            f"{comparison.optimized_analysis.total_time_us/1000:.2f} | "
            f"**{comparison.total_speedup:.2f}x** |"
        )
        lines.append(
            f"| Critical Path (ms) | {comparison.baseline_analysis.critical_path_length_us/1000:.2f} | "
            f"{comparison.optimized_analysis.critical_path_length_us/1000:.2f} | "
            f"{comparison.critical_path_speedup:.2f}x |"
        )
        lines.append(
            f"| Idle % | {comparison.baseline_analysis.idle_breakdown.idle_percentage:.1f}% | "
            f"{comparison.optimized_analysis.idle_breakdown.idle_percentage:.1f}% | "
            f"{comparison.idle_reduction_pct:+.1f}% |"
        )
        if comparison.baseline_analysis.comm_comp_overlap.total_comm_us > 0:
            lines.append(
                f"| Comm/Comp Overlap | {comparison.baseline_analysis.comm_comp_overlap.overlap_percentage:.1f}% | "
                f"{comparison.optimized_analysis.comm_comp_overlap.overlap_percentage:.1f}% | "
                f"{comparison.overlap_improvement_pct:+.1f}% |"
            )
        lines.append("")
        
        # Insights
        lines.append("## Insights")
        lines.append("")
        for insight in comparison.insights:
            lines.append(f"- {insight}")
        lines.append("")
        
        # Kernel Changes
        if comparison.kernel_speedups:
            lines.append("## Kernel-Level Speedups")
            lines.append("")
            lines.append("| Kernel | Speedup |")
            lines.append("|--------|---------|")
            for name, speedup in sorted(
                comparison.kernel_speedups.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                lines.append(f"| {name[:40]} | {speedup:.2f}x |")
            lines.append("")
        
        if comparison.eliminated_kernels:
            lines.append("### Eliminated Kernels")
            lines.append("")
            for k in comparison.eliminated_kernels[:5]:
                lines.append(f"- {k}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """CLI entry point for HTA analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HTA Trace Analyzer")
    parser.add_argument(
        "--trace",
        type=Path,
        help="Path to trace file for single-trace analysis",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline trace for comparison",
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        help="Path to optimized trace for comparison",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top kernels to include (default: 10)",
    )
    
    args = parser.parse_args()
    
    analyzer = HTAAnalyzer()
    
    if not analyzer.is_available:
        logger.warning("HTA library not available. Using fallback parser.")
    
    if args.baseline and args.optimized:
        # Comparison mode
        comparison = analyzer.compare_traces(args.baseline, args.optimized, args.top_k)
        if comparison is None:
            print("Failed to compare traces", file=sys.stderr)
            return 1
        
        markdown = analyzer.generate_comparison_markdown(comparison)
        
        if args.output_md:
            args.output_md.write_text(markdown)
            print(f"Markdown report written to: {args.output_md}")
        else:
            print(markdown)
        
        if args.output_json:
            # Serialize comparison to JSON
            import dataclasses
            
            def to_dict(obj):
                if dataclasses.is_dataclass(obj):
                    return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
                elif isinstance(obj, list):
                    return [to_dict(v) for v in obj]
                elif isinstance(obj, dict):
                    return {k: to_dict(v) for k, v in obj.items()}
                return obj
            
            args.output_json.write_text(json.dumps(to_dict(comparison), indent=2))
            print(f"JSON report written to: {args.output_json}")
    
    elif args.trace:
        # Single trace mode
        analysis = analyzer.analyze_trace(args.trace, args.top_k)
        if analysis is None:
            print("Failed to analyze trace", file=sys.stderr)
            return 1
        
        markdown = analyzer.generate_markdown_report(analysis)
        
        if args.output_md:
            args.output_md.write_text(markdown)
            print(f"Markdown report written to: {args.output_md}")
        else:
            print(markdown)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


