#!/usr/bin/env python3
"""
Differential Profile Analyzer - Compare baseline vs optimized deep profiling reports.

Takes two deep_profiling_report.py JSON outputs and generates:
1. Binding shift analysis (memory-bound <-> compute-bound transitions)
2. Improvement attribution (what contributed to speedup)
3. Kernel-level speedup breakdown
4. Remaining bottlenecks and next optimization steps

Usage:
    python -m core.analysis.differential_profile_analyzer \\
        --baseline artifacts/runs/analysis/baseline_deep_profile.json \\
        --optimized artifacts/runs/analysis/optimized_deep_profile.json \\
        --output-json artifacts/runs/analysis/differential_analysis.json \\
        --output-md artifacts/runs/analysis/differential_analysis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class KernelDiff:
    """Per-kernel differential analysis."""
    name: str
    baseline_time_ms: Optional[float]
    optimized_time_ms: Optional[float]
    speedup: float
    
    # Roofline position change
    baseline_binding: str  # "memory-bound", "compute-bound", "tmem-bound"
    optimized_binding: str
    binding_changed: bool
    
    # Key metric changes (deltas as percentages)
    sm_util_delta: Optional[float]
    dram_util_delta: Optional[float]
    occupancy_delta: Optional[float]
    l2_hit_delta: Optional[float]
    arithmetic_intensity_delta: Optional[float]
    
    # Interpretation
    primary_improvement: str
    

@dataclass
class ImprovementAttribution:
    """Attribution of speedup to different factors."""
    reduced_memory_traffic: float  # 0.0 - 1.0 fraction
    improved_compute_utilization: float
    better_cache_behavior: float
    kernel_fusion_overlap: float
    other: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "reduced_memory_traffic": self.reduced_memory_traffic,
            "improved_compute_utilization": self.improved_compute_utilization,
            "better_cache_behavior": self.better_cache_behavior,
            "kernel_fusion_overlap": self.kernel_fusion_overlap,
            "other": self.other,
        }


@dataclass
class DifferentialReport:
    """Complete differential analysis between baseline and optimized."""
    baseline_path: str
    optimized_path: str
    
    # Overall metrics
    total_baseline_time_ms: float
    total_optimized_time_ms: float
    overall_speedup: float
    
    # Binding analysis
    baseline_dominant_binding: str
    optimized_dominant_binding: str
    binding_shift: Optional[str]  # e.g., "memory-bound -> compute-bound"
    
    # Per-kernel analysis
    kernel_diffs: List[KernelDiff]
    
    # Improvement breakdown
    improvement_attribution: ImprovementAttribution
    
    # Actionable insights
    key_improvements: List[str]
    remaining_bottlenecks: List[str]
    next_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "optimized_path": self.optimized_path,
            "total_baseline_time_ms": self.total_baseline_time_ms,
            "total_optimized_time_ms": self.total_optimized_time_ms,
            "overall_speedup": self.overall_speedup,
            "baseline_dominant_binding": self.baseline_dominant_binding,
            "optimized_dominant_binding": self.optimized_dominant_binding,
            "binding_shift": self.binding_shift,
            "kernel_diffs": [
                {
                    "name": kd.name,
                    "baseline_time_ms": kd.baseline_time_ms,
                    "optimized_time_ms": kd.optimized_time_ms,
                    "speedup": kd.speedup,
                    "baseline_binding": kd.baseline_binding,
                    "optimized_binding": kd.optimized_binding,
                    "binding_changed": kd.binding_changed,
                    "sm_util_delta": kd.sm_util_delta,
                    "dram_util_delta": kd.dram_util_delta,
                    "occupancy_delta": kd.occupancy_delta,
                    "l2_hit_delta": kd.l2_hit_delta,
                    "arithmetic_intensity_delta": kd.arithmetic_intensity_delta,
                    "primary_improvement": kd.primary_improvement,
                }
                for kd in self.kernel_diffs
            ],
            "improvement_attribution": self.improvement_attribution.to_dict(),
            "key_improvements": self.key_improvements,
            "remaining_bottlenecks": self.remaining_bottlenecks,
            "next_steps": self.next_steps,
        }


def _safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dict."""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is default:
            return default
    return data


def _determine_binding(advisory: Dict) -> str:
    """Determine binding type from advisory."""
    roofline = advisory.get("roofline", {})
    if roofline:
        if roofline.get("is_tmem_bound"):
            return "tmem-bound"
        elif roofline.get("is_memory_bound"):
            return "memory-bound"
        elif roofline.get("is_compute_bound"):
            return "compute-bound"
        binding = roofline.get("binding", "")
        if binding:
            return f"{binding}-bound"
    return "unknown"


def _compute_improvement_attribution(
    baseline_advisories: List[Dict],
    optimized_advisories: List[Dict],
    kernel_diffs: List[KernelDiff],
) -> ImprovementAttribution:
    """Attribute speedup to different factors based on metric changes."""
    # Aggregate metric changes
    total_sm_delta = 0.0
    total_dram_delta = 0.0
    total_l2_delta = 0.0
    total_ai_delta = 0.0
    count = 0
    
    for kd in kernel_diffs:
        if kd.speedup > 1.0:  # Only count improved kernels
            if kd.sm_util_delta is not None:
                total_sm_delta += kd.sm_util_delta
            if kd.dram_util_delta is not None:
                total_dram_delta += kd.dram_util_delta
            if kd.l2_hit_delta is not None:
                total_l2_delta += kd.l2_hit_delta
            if kd.arithmetic_intensity_delta is not None:
                total_ai_delta += kd.arithmetic_intensity_delta
            count += 1
    
    if count == 0:
        return ImprovementAttribution(0.0, 0.0, 0.0, 0.0, 1.0)
    
    # Normalize deltas
    avg_sm_delta = total_sm_delta / count
    avg_dram_delta = total_dram_delta / count
    avg_l2_delta = total_l2_delta / count
    avg_ai_delta = total_ai_delta / count
    
    # Heuristic attribution based on metric changes
    # Positive SM delta + reduced DRAM = better compute utilization
    # Negative DRAM delta + positive AI = reduced memory traffic
    # Positive L2 delta = better cache behavior
    
    compute_score = max(0, avg_sm_delta) * 0.01  # Scale to 0-1
    memory_score = max(0, -avg_dram_delta) * 0.01 + max(0, avg_ai_delta) * 0.001
    cache_score = max(0, avg_l2_delta) * 0.01
    
    # Check for kernel count reduction (fusion)
    baseline_kernel_count = len(baseline_advisories)
    optimized_kernel_count = len(optimized_advisories)
    fusion_score = max(0, (baseline_kernel_count - optimized_kernel_count) / max(baseline_kernel_count, 1)) * 0.5
    
    # Normalize to sum to 1.0
    total_score = compute_score + memory_score + cache_score + fusion_score
    if total_score < 0.01:
        return ImprovementAttribution(0.0, 0.0, 0.0, 0.0, 1.0)
    
    other_score = max(0, 1.0 - total_score)
    total_score += other_score
    
    return ImprovementAttribution(
        reduced_memory_traffic=memory_score / total_score,
        improved_compute_utilization=compute_score / total_score,
        better_cache_behavior=cache_score / total_score,
        kernel_fusion_overlap=fusion_score / total_score,
        other=other_score / total_score,
    )


def _generate_key_improvements(kernel_diffs: List[KernelDiff], binding_shift: Optional[str]) -> List[str]:
    """Generate list of key improvements."""
    improvements = []
    
    if binding_shift:
        improvements.append(f"Shifted from {binding_shift}")
    
    # Find most improved kernels
    sorted_diffs = sorted(kernel_diffs, key=lambda kd: kd.speedup, reverse=True)
    for kd in sorted_diffs[:3]:
        if kd.speedup > 1.1:
            improvements.append(f"Kernel '{kd.name}' improved {kd.speedup:.2f}x: {kd.primary_improvement}")
    
    # Note binding changes
    binding_changes = [kd for kd in kernel_diffs if kd.binding_changed]
    if binding_changes:
        improvements.append(f"{len(binding_changes)} kernel(s) changed binding type")
    
    return improvements if improvements else ["Minimal improvements detected"]


def _generate_remaining_bottlenecks(optimized_advisories: List[Dict]) -> List[str]:
    """Identify remaining bottlenecks from optimized profile."""
    bottlenecks = []
    
    for advisory in optimized_advisories:
        roofline = advisory.get("roofline", {})
        kernel = advisory.get("kernel", "unknown")
        
        # Check for low utilization
        sm_util = advisory.get("sm_util_pct")
        if sm_util is not None and sm_util < 50:
            bottlenecks.append(f"'{kernel}': SM utilization at {sm_util:.1f}% (target >50%)")
        
        dram_util = advisory.get("dram_util_pct")
        if dram_util is not None and dram_util > 80:
            bottlenecks.append(f"'{kernel}': HBM saturated at {dram_util:.1f}% (memory-bound)")
        
        occupancy = advisory.get("occupancy_pct")
        if occupancy is not None and occupancy < 40:
            bottlenecks.append(f"'{kernel}': Low occupancy at {occupancy:.1f}%")
        
        l2_hit = advisory.get("l2_hit_pct")
        if l2_hit is not None and l2_hit < 50:
            bottlenecks.append(f"'{kernel}': L2 hit rate only {l2_hit:.1f}%")
    
    return bottlenecks[:5] if bottlenecks else ["No major bottlenecks identified"]


def _generate_next_steps(
    optimized_advisories: List[Dict],
    kernel_diffs: List[KernelDiff],
    overall_speedup: float,
) -> List[str]:
    """Generate actionable next steps."""
    steps = []
    
    # Check dominant binding in optimized
    memory_bound_count = sum(1 for kd in kernel_diffs if "memory" in kd.optimized_binding)
    compute_bound_count = sum(1 for kd in kernel_diffs if "compute" in kd.optimized_binding)
    
    if memory_bound_count > compute_bound_count:
        steps.append("Focus on increasing arithmetic intensity: operator fusion, data reuse, TMA prefetching")
    elif compute_bound_count > memory_bound_count:
        steps.append("Focus on compute efficiency: occupancy tuning, ILP optimization, register pressure")
    
    # Collect recommendations from advisories
    seen_recommendations = set()
    for advisory in optimized_advisories:
        for rec in advisory.get("recommendations", [])[:2]:
            if rec not in seen_recommendations:
                seen_recommendations.add(rec)
                steps.append(rec)
    
    # Add speedup-dependent suggestions
    if overall_speedup < 1.5:
        steps.append("Consider algorithmic changes or precision reduction for larger gains")
    elif overall_speedup >= 2.0:
        steps.append("Good progress! Profile again after implementing remaining optimizations")
    
    return steps[:5] if steps else ["Profile with NCU source correlation for finer-grained analysis"]


def analyze_differential(
    baseline_json: Path,
    optimized_json: Path,
) -> DifferentialReport:
    """
    Compare two deep profiling JSON reports and generate differential analysis.
    
    Args:
        baseline_json: Path to baseline deep_profiling_report.py JSON output
        optimized_json: Path to optimized deep_profiling_report.py JSON output
    
    Returns:
        DifferentialReport with complete analysis
    """
    # Load JSON files
    with baseline_json.open() as f:
        baseline_data = json.load(f)
    with optimized_json.open() as f:
        optimized_data = json.load(f)
    
    baseline_advisories = baseline_data.get("advisories", [])
    optimized_advisories = optimized_data.get("advisories", [])
    
    # Match kernels between baseline and optimized
    baseline_by_name = {adv.get("kernel", ""): adv for adv in baseline_advisories}
    optimized_by_name = {adv.get("kernel", ""): adv for adv in optimized_advisories}
    
    all_kernel_names = set(baseline_by_name.keys()) | set(optimized_by_name.keys())
    
    # Compute kernel-level diffs
    kernel_diffs: List[KernelDiff] = []
    total_baseline_time = 0.0
    total_optimized_time = 0.0
    
    for kernel_name in all_kernel_names:
        if not kernel_name:
            continue
            
        baseline_adv = baseline_by_name.get(kernel_name, {})
        optimized_adv = optimized_by_name.get(kernel_name, {})
        
        b_time = baseline_adv.get("duration_ms")
        o_time = optimized_adv.get("duration_ms")
        
        if b_time is not None:
            total_baseline_time += b_time
        if o_time is not None:
            total_optimized_time += o_time
        
        # Calculate speedup
        if b_time and o_time and o_time > 0:
            speedup = b_time / o_time
        elif b_time and not o_time:
            speedup = float('inf')  # Kernel eliminated
        elif o_time and not b_time:
            speedup = 0.0  # New kernel added
        else:
            speedup = 1.0
        
        # Determine bindings
        b_binding = _determine_binding(baseline_adv) if baseline_adv else "unknown"
        o_binding = _determine_binding(optimized_adv) if optimized_adv else "unknown"
        binding_changed = b_binding != o_binding and b_binding != "unknown" and o_binding != "unknown"
        
        # Compute metric deltas
        b_sm = baseline_adv.get("sm_util_pct")
        o_sm = optimized_adv.get("sm_util_pct")
        sm_delta = (o_sm - b_sm) if (b_sm is not None and o_sm is not None) else None
        
        b_dram = baseline_adv.get("dram_util_pct")
        o_dram = optimized_adv.get("dram_util_pct")
        dram_delta = (o_dram - b_dram) if (b_dram is not None and o_dram is not None) else None
        
        b_occ = baseline_adv.get("occupancy_pct")
        o_occ = optimized_adv.get("occupancy_pct")
        occ_delta = (o_occ - b_occ) if (b_occ is not None and o_occ is not None) else None
        
        b_l2 = baseline_adv.get("l2_hit_pct")
        o_l2 = optimized_adv.get("l2_hit_pct")
        l2_delta = (o_l2 - b_l2) if (b_l2 is not None and o_l2 is not None) else None
        
        b_roof = baseline_adv.get("roofline") or {}
        o_roof = optimized_adv.get("roofline") or {}
        b_ai = b_roof.get("arithmetic_intensity") if b_roof else None
        o_ai = o_roof.get("arithmetic_intensity") if o_roof else None
        ai_delta = (o_ai - b_ai) if (b_ai is not None and o_ai is not None) else None
        
        # Determine primary improvement
        primary_improvement = "no change"
        if speedup > 1.05:
            if sm_delta and sm_delta > 10:
                primary_improvement = "improved compute utilization"
            elif dram_delta and dram_delta < -10:
                primary_improvement = "reduced memory traffic"
            elif ai_delta and ai_delta > 1:
                primary_improvement = "increased arithmetic intensity"
            elif l2_delta and l2_delta > 10:
                primary_improvement = "better cache utilization"
            elif binding_changed:
                primary_improvement = f"binding shift ({b_binding} → {o_binding})"
            else:
                primary_improvement = "general optimization"
        elif speedup < 0.95:
            primary_improvement = "regression detected"
        
        kernel_diffs.append(KernelDiff(
            name=kernel_name,
            baseline_time_ms=b_time,
            optimized_time_ms=o_time,
            speedup=speedup,
            baseline_binding=b_binding,
            optimized_binding=o_binding,
            binding_changed=binding_changed,
            sm_util_delta=sm_delta,
            dram_util_delta=dram_delta,
            occupancy_delta=occ_delta,
            l2_hit_delta=l2_delta,
            arithmetic_intensity_delta=ai_delta,
            primary_improvement=primary_improvement,
        ))
    
    # Sort by speedup (most improved first)
    kernel_diffs.sort(key=lambda kd: kd.speedup, reverse=True)
    
    # Overall speedup
    overall_speedup = total_baseline_time / total_optimized_time if total_optimized_time > 0 else 1.0
    
    # Determine dominant binding
    baseline_stats = baseline_data.get("stats", {})
    optimized_stats = optimized_data.get("stats", {})
    
    b_memory_bound = baseline_stats.get("memory_bound_kernels", [])
    b_compute_bound = baseline_stats.get("compute_bound_kernels", [])
    o_memory_bound = optimized_stats.get("memory_bound_kernels", [])
    o_compute_bound = optimized_stats.get("compute_bound_kernels", [])
    
    baseline_dominant = "memory-bound" if len(b_memory_bound) >= len(b_compute_bound) else "compute-bound"
    optimized_dominant = "memory-bound" if len(o_memory_bound) >= len(o_compute_bound) else "compute-bound"
    
    binding_shift = None
    if baseline_dominant != optimized_dominant:
        binding_shift = f"{baseline_dominant} → {optimized_dominant}"
    
    # Compute improvement attribution
    attribution = _compute_improvement_attribution(baseline_advisories, optimized_advisories, kernel_diffs)
    
    # Generate insights
    key_improvements = _generate_key_improvements(kernel_diffs, binding_shift)
    remaining_bottlenecks = _generate_remaining_bottlenecks(optimized_advisories)
    next_steps = _generate_next_steps(optimized_advisories, kernel_diffs, overall_speedup)
    
    return DifferentialReport(
        baseline_path=str(baseline_json),
        optimized_path=str(optimized_json),
        total_baseline_time_ms=total_baseline_time,
        total_optimized_time_ms=total_optimized_time,
        overall_speedup=overall_speedup,
        baseline_dominant_binding=baseline_dominant,
        optimized_dominant_binding=optimized_dominant,
        binding_shift=binding_shift,
        kernel_diffs=kernel_diffs,
        improvement_attribution=attribution,
        key_improvements=key_improvements,
        remaining_bottlenecks=remaining_bottlenecks,
        next_steps=next_steps,
    )


def _generate_why_faster_summary(report: DifferentialReport) -> List[str]:
    """Generate a clear explanation of why the optimized version is faster."""
    lines = []
    lines.append("## Why Is It Faster?")
    lines.append("")
    
    time_saved_ms = report.total_baseline_time_ms - report.total_optimized_time_ms
    if time_saved_ms <= 0:
        lines.append("⚠️ **No improvement detected** - optimized version is not faster.")
        return lines
    
    lines.append(f"**Total time saved:** {time_saved_ms:.2f}ms ({report.overall_speedup:.2f}x speedup)")
    lines.append("")
    
    # Analyze top contributing factors
    factors = []
    
    # Check for binding shift (often the biggest factor)
    if report.binding_shift:
        factors.append(f"🔄 **Binding shift:** {report.binding_shift} - this is often the primary optimization")
    
    # Analyze kernel-level contributions
    kernel_contributions = []
    for kd in report.kernel_diffs:
        if kd.baseline_time_ms and kd.optimized_time_ms and kd.speedup > 1.05:
            time_saved = kd.baseline_time_ms - kd.optimized_time_ms
            pct_of_total = (time_saved / time_saved_ms * 100) if time_saved_ms > 0 else 0
            if pct_of_total > 5:  # Only show significant contributors
                kernel_contributions.append({
                    'name': kd.name,
                    'time_saved': time_saved,
                    'pct': pct_of_total,
                    'reason': kd.primary_improvement,
                    'binding_changed': kd.binding_changed,
                    'baseline_binding': kd.baseline_binding,
                    'optimized_binding': kd.optimized_binding,
                })
    
    # Sort by contribution
    kernel_contributions.sort(key=lambda x: x['pct'], reverse=True)
    
    if kernel_contributions:
        lines.append("**Breakdown by kernel:**")
        lines.append("")
        lines.append("| Kernel | Time Saved | % of Total | Why |")
        lines.append("|--------|------------|------------|-----|")
        for kc in kernel_contributions[:5]:
            why = kc['reason']
            if kc['binding_changed']:
                why = f"{kc['baseline_binding']} → {kc['optimized_binding']}"
            lines.append(f"| {kc['name'][:25]} | {kc['time_saved']:.2f}ms | {kc['pct']:.0f}% | {why} |")
        lines.append("")
    
    # Summarize the improvement attribution
    attr = report.improvement_attribution
    attr_factors = []
    if attr.reduced_memory_traffic > 0.1:
        attr_factors.append(f"**Reduced memory traffic** ({attr.reduced_memory_traffic*100:.0f}%): Less data moved between GPU and memory")
    if attr.improved_compute_utilization > 0.1:
        attr_factors.append(f"**Better compute utilization** ({attr.improved_compute_utilization*100:.0f}%): More SMs active, better parallelism")
    if attr.better_cache_behavior > 0.1:
        attr_factors.append(f"**Improved caching** ({attr.better_cache_behavior*100:.0f}%): Higher L2 hit rates, less HBM traffic")
    if attr.kernel_fusion_overlap > 0.1:
        attr_factors.append(f"**Kernel fusion/overlap** ({attr.kernel_fusion_overlap*100:.0f}%): Fewer kernel launches, better pipelining")
    
    if attr_factors:
        lines.append("**Primary optimization factors:**")
        lines.append("")
        for factor in attr_factors:
            lines.append(f"- {factor}")
        lines.append("")
    
    return lines


def _generate_how_to_improve_further(report: DifferentialReport) -> List[str]:
    """Generate specific, actionable recommendations for further optimization."""
    lines = []
    lines.append("## How to Improve Further")
    lines.append("")
    
    # Identify the current bottleneck
    memory_bound_kernels = [kd for kd in report.kernel_diffs if "memory" in (kd.optimized_binding or "")]
    compute_bound_kernels = [kd for kd in report.kernel_diffs if "compute" in (kd.optimized_binding or "")]
    
    # Find the slowest remaining kernel
    slowest = None
    for kd in report.kernel_diffs:
        if kd.optimized_time_ms and kd.optimized_time_ms > 0.1:
            if slowest is None or kd.optimized_time_ms > slowest.optimized_time_ms:
                slowest = kd
    
    if slowest:
        lines.append(f"### 🎯 Focus Area: `{slowest.name}`")
        lines.append("")
        lines.append(f"This kernel takes **{slowest.optimized_time_ms:.2f}ms** and is **{slowest.optimized_binding}**.")
        lines.append("")
        
        if "memory" in (slowest.optimized_binding or ""):
            lines.append("**Recommended optimizations for memory-bound kernels:**")
            lines.append("")
            lines.append("1. **Increase arithmetic intensity** - Do more computation per byte loaded")
            lines.append("   - Fuse with adjacent operators to reuse data in registers/shared memory")
            lines.append("   - Use operator fusion (torch.compile, custom CUDA kernels)")
            lines.append("")
            lines.append("2. **Reduce memory traffic**")
            lines.append("   - Use lower precision (FP16/BF16/FP8) to halve memory bandwidth")
            lines.append("   - Implement tiling to fit working set in L2 cache")
            lines.append("   - Use TMA (Tensor Memory Accelerator) for async prefetching")
            lines.append("")
            lines.append("3. **Improve memory access patterns**")
            lines.append("   - Ensure coalesced memory access (consecutive threads access consecutive addresses)")
            lines.append("   - Align data to 128-byte boundaries")
            lines.append("")
        elif "compute" in (slowest.optimized_binding or ""):
            lines.append("**Recommended optimizations for compute-bound kernels:**")
            lines.append("")
            lines.append("1. **Improve SM utilization**")
            lines.append("   - Tune block size for better occupancy")
            lines.append("   - Reduce register pressure to allow more concurrent warps")
            lines.append("")
            lines.append("2. **Use Tensor Cores** (if not already)")
            lines.append("   - Ensure matrix dimensions are multiples of 16 (FP16) or 8 (TF32)")
            lines.append("   - Use torch.compile with mode='max-autotune'")
            lines.append("")
            lines.append("3. **Increase instruction-level parallelism**")
            lines.append("   - Unroll loops to expose more independent operations")
            lines.append("   - Interleave compute with memory operations")
            lines.append("")
    
    # Add specific bottleneck warnings
    if report.remaining_bottlenecks:
        lines.append("### ⚠️ Remaining Bottlenecks")
        lines.append("")
        for bn in report.remaining_bottlenecks[:3]:
            lines.append(f"- {bn}")
        lines.append("")
    
    return lines


def generate_markdown_report(report: DifferentialReport) -> str:
    """Generate human-readable markdown from differential report."""
    lines = []
    
    lines.append("# Differential Profile Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Baseline | Optimized | Change |")
    lines.append("|--------|----------|-----------|--------|")
    lines.append(f"| Total Time (ms) | {report.total_baseline_time_ms:.3f} | {report.total_optimized_time_ms:.3f} | **{report.overall_speedup:.2f}x faster** |")
    lines.append(f"| Dominant Binding | {report.baseline_dominant_binding} | {report.optimized_dominant_binding} | {report.binding_shift or 'unchanged'} |")
    lines.append("")
    
    # Add the new "Why Faster" section
    lines.extend(_generate_why_faster_summary(report))
    
    # Add the new "How to Improve Further" section
    lines.extend(_generate_how_to_improve_further(report))
    
    # Improvement attribution
    lines.append("## Improvement Attribution")
    lines.append("")
    attr = report.improvement_attribution
    total = attr.reduced_memory_traffic + attr.improved_compute_utilization + attr.better_cache_behavior + attr.kernel_fusion_overlap + attr.other
    if total > 0:
        lines.append("| Factor | Contribution |")
        lines.append("|--------|--------------|")
        if attr.reduced_memory_traffic > 0.05:
            lines.append(f"| Reduced Memory Traffic | {attr.reduced_memory_traffic*100:.0f}% |")
        if attr.improved_compute_utilization > 0.05:
            lines.append(f"| Improved Compute Utilization | {attr.improved_compute_utilization*100:.0f}% |")
        if attr.better_cache_behavior > 0.05:
            lines.append(f"| Better Cache Behavior | {attr.better_cache_behavior*100:.0f}% |")
        if attr.kernel_fusion_overlap > 0.05:
            lines.append(f"| Kernel Fusion/Overlap | {attr.kernel_fusion_overlap*100:.0f}% |")
        if attr.other > 0.05:
            lines.append(f"| Other | {attr.other*100:.0f}% |")
    lines.append("")
    
    # Key improvements
    lines.append("## Key Improvements")
    lines.append("")
    for imp in report.key_improvements:
        lines.append(f"- ✅ {imp}")
    lines.append("")
    
    # Kernel breakdown (top 5)
    lines.append("## Kernel-Level Analysis")
    lines.append("")
    lines.append("| Kernel | Baseline (ms) | Optimized (ms) | Speedup | Binding | Primary Improvement |")
    lines.append("|--------|---------------|----------------|---------|---------|---------------------|")
    for kd in report.kernel_diffs[:10]:
        b_time = f"{kd.baseline_time_ms:.3f}" if kd.baseline_time_ms else "N/A"
        o_time = f"{kd.optimized_time_ms:.3f}" if kd.optimized_time_ms else "N/A"
        speedup_str = f"{kd.speedup:.2f}x" if kd.speedup != float('inf') else "eliminated"
        binding_str = f"{kd.baseline_binding} → {kd.optimized_binding}" if kd.binding_changed else kd.optimized_binding
        lines.append(f"| {kd.name[:30]} | {b_time} | {o_time} | {speedup_str} | {binding_str} | {kd.primary_improvement} |")
    lines.append("")
    
    # Remaining bottlenecks
    lines.append("## Remaining Bottlenecks")
    lines.append("")
    for bottleneck in report.remaining_bottlenecks:
        lines.append(f"- ⚠️ {bottleneck}")
    lines.append("")
    
    # Next steps
    lines.append("## Recommended Next Steps")
    lines.append("")
    for i, step in enumerate(report.next_steps, 1):
        lines.append(f"{i}. {step}")
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Differential Profile Analyzer")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline deep_profiling_report JSON output",
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        required=True,
        help="Path to optimized deep_profiling_report JSON output",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Output path for Markdown report",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print Markdown report to stdout",
    )
    
    args = parser.parse_args()
    
    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        return 1
    if not args.optimized.exists():
        print(f"Error: Optimized file not found: {args.optimized}", file=sys.stderr)
        return 1
    
    # Run analysis
    report = analyze_differential(args.baseline, args.optimized)
    
    # Output JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"JSON report written to: {args.output_json}")
    
    # Output Markdown
    markdown = generate_markdown_report(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
        print(f"Markdown report written to: {args.output_md}")
    
    if args.print or (not args.output_json and not args.output_md):
        print(markdown)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
