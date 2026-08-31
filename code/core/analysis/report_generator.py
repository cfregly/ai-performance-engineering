#!/usr/bin/env python3
"""
Markdown report generator for performance analysis results.
Generates reports similar to docs/performance_baseline.md format.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.benchmark.performance_targets import (
    TARGETS,
    get_chapter_description,
    get_chapter_metrics,
    get_all_chapters,
    compute_status,
    format_value,
    get_targets_metadata,
)


class PerformanceReport:
    """Generate markdown performance analysis reports."""
    
    def __init__(self):
        self.chapters_analyzed: List[str] = []
        self.metrics_by_chapter: Dict[str, Dict[str, Any]] = {}
        self.summary_stats = {
            "total_metrics": 0,
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "missing_count": 0,
        }
    
    def add_chapter_metrics(self, chapter: str, metrics: Dict[str, float]):
        """
        Add metrics for a chapter.
        
        Args:
            chapter: Chapter identifier (e.g., "ch02")
            metrics: Dictionary of metric_name -> value
        """
        if chapter not in self.chapters_analyzed:
            self.chapters_analyzed.append(chapter)
        
        # Compare against targets
        chapter_targets = get_chapter_metrics(chapter)
        results = {}
        
        for metric_name, target_def in chapter_targets.items():
            actual_value = metrics.get(metric_name)
            
            if actual_value is not None:
                status = compute_status(actual_value, target_def)
                results[metric_name] = {
                    "actual": actual_value,
                    "target": target_def,
                    "status": status,
                }
                
                # Update summary stats
                self.summary_stats["total_metrics"] += 1
                if status == "PASS":
                    self.summary_stats["pass_count"] += 1
                elif status == "WARN":
                    self.summary_stats["warn_count"] += 1
                elif status == "FAIL":
                    self.summary_stats["fail_count"] += 1
            else:
                # Metric not found
                results[metric_name] = {
                    "actual": None,
                    "target": target_def,
                    "status": "MISSING",
                }
                self.summary_stats["total_metrics"] += 1
                self.summary_stats["missing_count"] += 1
        
        self.metrics_by_chapter[chapter] = results
    
    def generate_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown formatted report
        """
        lines = []
        
        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# Performance Analysis Report")
        lines.append(f"")
        lines.append(f"**Generated:** {timestamp}")
        lines.append(f"")

        target_metadata = get_targets_metadata()
        target_source = target_metadata.get("source")
        target_artifact = target_metadata.get("artifact_path")
        target_warnings = target_metadata.get("warnings") or []
        if target_source:
            lines.append(f"**Target Source:** {target_source}")
        if target_artifact:
            lines.append(f"**Target Artifact:** `{target_artifact}`")
        if target_warnings:
            lines.append("")
            lines.append("**Target Provenance Warnings:**")
            for warning in target_warnings:
                lines.append(f"- {warning}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        stats = self.summary_stats
        lines.append(f"- **Total metrics analyzed:** {stats['total_metrics']}")
        lines.append(f"- **PASS:** {stats['pass_count']} [OK]")
        lines.append(f"- **WARN:** {stats['warn_count']} WARNING:")
        lines.append(f"- **FAIL:** {stats['fail_count']} ERROR:")
        lines.append(f"- **MISSING:** {stats['missing_count']} ❓")
        lines.append("")
        
        # Detailed breakdown per chapter
        lines.append("## Detailed Results by Chapter")
        lines.append("")
        
        for chapter in sorted(self.chapters_analyzed):
            desc = get_chapter_description(chapter)
            results = self.metrics_by_chapter.get(chapter, {})
            
            if not results:
                continue
            
            lines.append(f"### {chapter.upper()}: {desc}")
            lines.append("")
            
            # Count statuses for chapter summary
            statuses = [r["status"] for r in results.values()]
            fail_c = statuses.count("FAIL")
            warn_c = statuses.count("WARN")
            pass_c = statuses.count("PASS")
            
            if fail_c > 0:
                lines.append("**Overall Status:** ERROR: FAIL")
            elif warn_c > 0:
                lines.append("**Overall Status:** WARNING: WARN")
            else:
                lines.append("**Overall Status:** [OK] PASS")
            lines.append("")
            
            # Metrics table
            lines.append("| Metric | Target | Actual | Status |")
            lines.append("|--------|--------|--------|--------|")
            
            for metric_name, result in sorted(results.items()):
                target_def = result["target"]
                actual = result["actual"]
                status = result["status"]
                
                # Format target
                target_str = format_value(target_def["target"], target_def["unit"])
                
                # Format actual
                if actual is not None:
                    actual_str = format_value(actual, target_def["unit"])
                else:
                    actual_str = "N/A"
                
                # Status icon
                status_icon = {
                    "PASS": "[OK] PASS",
                    "WARN": "WARNING: WARN",
                    "FAIL": "ERROR: FAIL",
                    "MISSING": "❓ MISSING",
                }.get(status, status)
                
                # Format metric name
                metric_display = metric_name.replace("_", " ").title()
                
                lines.append(f"| {metric_display} | {target_str} | {actual_str} | {status_icon} |")
            
            lines.append("")
        
        # Recommendations
        if stats["fail_count"] > 0 or stats["warn_count"] > 0:
            lines.append("## Recommendations")
            lines.append("")
            
            for chapter in sorted(self.chapters_analyzed):
                results = self.metrics_by_chapter.get(chapter, {})
                
                failed = [m for m, r in results.items() if r["status"] == "FAIL"]
                warned = [m for m, r in results.items() if r["status"] == "WARN"]
                
                if failed or warned:
                    desc = get_chapter_description(chapter)
                    lines.append(f"### {chapter.upper()}: {desc}")
                    lines.append("")
                    
                    if failed:
                        lines.append("**Failed metrics:**")
                        for metric in failed:
                            lines.append(f"- `{metric}`: Below minimum threshold")
                        lines.append("")
                    
                    if warned:
                        lines.append("**Warning metrics:**")
                        for metric in warned:
                            lines.append(f"- `{metric}`: Below target but above minimum")
                        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by automated performance analysis system*")
        lines.append("")
        
        return "\n".join(lines)
    
    def write_report(self, output_path: Path):
        """
        Write report to file.
        
        Args:
            output_path: Path to output markdown file
        """
        content = self.generate_markdown()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)


def generate_report_from_metrics(
    metrics: Dict[str, float],
    output_path: Optional[Path] = None,
) -> str:
    """
    High-level function to generate a report from flat metrics dictionary.
    
    Args:
        metrics: Dictionary of metric_name -> value
        output_path: Optional path to write report
    
    Returns:
        Markdown formatted report
    """
    report = PerformanceReport()
    
    def normalize(name: str) -> str:
        return name.replace("_", "").lower()
    
    def match_metric(metric: str, targets: Dict[str, Any], *, allow_partial: bool) -> tuple[str | None, bool]:
        """Return (target_metric, is_exact) if matched."""
        metric_norm = normalize(metric)
        
        # Exact name match first
        if metric in targets:
            return metric, True
        
        # Normalized equality
        for target in targets:
            if metric_norm == normalize(target):
                return target, True
        
        if not allow_partial:
            return None, False
        
        # Metric contains target (metric is more specific)
        for target in targets:
            if normalize(target) in metric_norm:
                return target, False
        
        # Target contains metric (metric is more general)
        for target in targets:
            if metric_norm and metric_norm in normalize(target):
                return target, False
        
        return None, False
    
    # Group metrics by chapter
    metrics_by_chapter: Dict[str, Dict[str, tuple[float, bool]]] = {}
    
    for raw_name, value in metrics.items():
        metric_name = raw_name
        chapter = None
        
        # Strip common prefixes
        for prefix in ["benchmark_", "test_outputs_", "ncu_", "nsys_", "pytorch_"]:
            if metric_name.startswith(prefix):
                metric_name = metric_name[len(prefix):]
                break
        
        # Extract chapter prefix if present
        if metric_name.startswith("ch") and "_" in metric_name:
            prefix, rest = metric_name.split("_", 1)
            chapter_number = prefix.removeprefix("ch")
            if chapter_number.isdigit() and 1 <= int(chapter_number) <= 20:
                chapter = f"ch{int(chapter_number):02d}"
                metric_name = rest
        
        matched_chapter = None
        matched_metric = None
        is_exact = False
        
        if chapter:
            target_metric, exact = match_metric(metric_name, get_chapter_metrics(chapter), allow_partial=True)
            if target_metric:
                matched_chapter = chapter
                matched_metric = target_metric
                is_exact = exact
        else:
            # Metrics without explicit chapter prefix only map to overall targets
            if "overall" in TARGETS and metric_name in TARGETS["overall"]:
                matched_chapter = "overall"
                matched_metric = metric_name
                is_exact = True
        
        if matched_chapter and matched_metric:
            chapter_metrics = metrics_by_chapter.setdefault(matched_chapter, {})
            existing = chapter_metrics.get(matched_metric)
            if existing:
                _, existing_exact = existing
                # Prefer exact matches over partial ones
                if existing_exact and not is_exact:
                    continue
            chapter_metrics[matched_metric] = (value, is_exact)
    
    # Add metrics to report (strip auxiliary data)
    for chapter, ch_metrics in metrics_by_chapter.items():
        cleaned = {metric: val for metric, (val, _) in ch_metrics.items()}
        report.add_chapter_metrics(chapter, cleaned)
    
    # Generate and optionally write report
    markdown = report.generate_markdown()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    
    return markdown
