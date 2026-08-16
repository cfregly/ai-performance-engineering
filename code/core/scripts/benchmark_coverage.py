#!/usr/bin/env python3
"""Generate a benchmark coverage report.

This script analyzes the codebase and generates a comprehensive report of:
- Benchmark file counts per chapter
- Baseline/optimized pairs
- Method implementation status
- Metric helper usage
- Lab coverage

Usage:
    python core/scripts/benchmark_coverage.py              # Full report
    python core/scripts/benchmark_coverage.py --json       # JSON output
    python core/scripts/benchmark_coverage.py --markdown   # Markdown output
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.discovery import is_generated_benchmark_copy


@dataclass
class ChapterStats:
    """Statistics for a single chapter."""
    chapter: int
    baseline_count: int = 0
    optimized_count: int = 0
    cuda_count: int = 0
    paired_count: int = 0  # Files with both baseline and optimized
    has_get_benchmark: int = 0
    has_get_custom_metrics: int = 0
    uses_helper: int = 0
    has_validate_result: int = 0
    has_nvtx_range: int = 0
    topic: str = ""
    
    @property
    def total_python(self) -> int:
        return self.baseline_count + self.optimized_count
    
    @property
    def completeness_pct(self) -> float:
        if self.total_python == 0:
            return 0.0
        required = self.has_get_benchmark + self.has_get_custom_metrics
        return (required / (self.total_python * 2)) * 100


@dataclass
class LabStats:
    """Statistics for a single lab."""
    name: str
    file_count: int = 0
    has_readme: bool = False
    baseline_count: int = 0
    optimized_count: int = 0


@dataclass
class CoverageReport:
    """Full coverage report."""
    total_files: int = 0
    total_chapters: int = 0
    total_labs: int = 0
    chapters: Dict[int, ChapterStats] = field(default_factory=dict)
    labs: Dict[str, LabStats] = field(default_factory=dict)
    metric_helper_usage: Dict[str, int] = field(default_factory=dict)
    missing_methods: List[str] = field(default_factory=list)


# Chapter topics
CHAPTER_TOPICS = {
    1: "Introduction & Benchmarking",
    2: "Hardware (Grace-Blackwell)",
    3: "OS/Docker/Kubernetes",
    4: "Distributed Computing",
    5: "Storage I/O",
    6: "GPU Architecture Basics",
    7: "Memory Access Patterns",
    8: "Occupancy & Optimization",
    9: "Arithmetic Intensity",
    10: "Intra-Kernel Pipelining",
    11: "CUDA Streams",
    12: "CUDA Graphs",
    13: "PyTorch Profiling",
    14: "torch.compile & Triton",
    15: "MoE & KV Management",
    16: "Production Inference",
    17: "vLLM/SGLang",
    18: "Advanced Decode",
    19: "Dynamic Inference",
    20: "AI-Assisted Optimization",
}


def analyze_file(filepath: Path) -> Dict[str, bool | str | None]:
    """Analyze a Python file for benchmark patterns."""
    try:
        content = filepath.read_text()
    except Exception:
        return {}
    
    result: Dict[str, bool | str | None] = {
        "has_get_benchmark": "def get_benchmark(" in content,
        "has_get_custom_metrics": "def get_custom_metrics(" in content,
        "has_validate_result": "def validate_result(" in content,
        "has_nvtx_range": "_nvtx_range" in content or "nvtx.range" in content,
        "uses_helper": bool(re.search(r'compute_\w+_metrics\(', content)),
        "helper_name": None,
    }

    # Find which helper is used
    match = re.search(r'(compute_\w+_metrics)\(', content)
    if match:
        return {**result, "helper_name": match.group(1)}
    
    return result


def analyze_chapter(root: Path, chapter: int) -> ChapterStats:
    """Analyze a single chapter."""
    ch_dir = root / f"ch{chapter}"
    if not ch_dir.exists():
        return ChapterStats(chapter=chapter)
    
    stats = ChapterStats(chapter=chapter, topic=CHAPTER_TOPICS.get(chapter, ""))
    
    # Find all baseline/optimized files
    baselines = set()
    optimized = set()
    
    for f in ch_dir.glob("baseline_*.py"):
        if f.is_file():
            baselines.add(f.stem.replace("baseline_", ""))
            stats.baseline_count += 1
            
            analysis = analyze_file(f)
            if analysis.get("has_get_benchmark"):
                stats.has_get_benchmark += 1
            if analysis.get("has_get_custom_metrics"):
                stats.has_get_custom_metrics += 1
            if analysis.get("uses_helper"):
                stats.uses_helper += 1
            if analysis.get("has_validate_result"):
                stats.has_validate_result += 1
            if analysis.get("has_nvtx_range"):
                stats.has_nvtx_range += 1
    
    for f in ch_dir.glob("optimized_*.py"):
        if f.is_file():
            optimized.add(f.stem.replace("optimized_", ""))
            stats.optimized_count += 1
            
            analysis = analyze_file(f)
            if analysis.get("has_get_benchmark"):
                stats.has_get_benchmark += 1
            if analysis.get("has_get_custom_metrics"):
                stats.has_get_custom_metrics += 1
            if analysis.get("uses_helper"):
                stats.uses_helper += 1
            if analysis.get("has_validate_result"):
                stats.has_validate_result += 1
            if analysis.get("has_nvtx_range"):
                stats.has_nvtx_range += 1
    
    # Count CUDA files
    stats.cuda_count = len([path for path in ch_dir.glob("*.cu") if not is_generated_benchmark_copy(path)])
    
    # Count paired files
    stats.paired_count = len(baselines & optimized)
    
    return stats


def analyze_lab(root: Path, lab_name: str) -> LabStats:
    """Analyze a single lab."""
    lab_dir = root / "labs" / lab_name
    if not lab_dir.exists():
        return LabStats(name=lab_name)
    
    stats = LabStats(name=lab_name)
    stats.has_readme = (lab_dir / "README.md").exists()
    stats.file_count = len(list(lab_dir.glob("*.py")))
    stats.baseline_count = len(list(lab_dir.glob("baseline_*.py")))
    stats.optimized_count = len(list(lab_dir.glob("optimized_*.py")))
    
    return stats


def generate_report(root: Path) -> CoverageReport:
    """Generate full coverage report."""
    report = CoverageReport()
    
    # Analyze chapters
    for ch in range(1, 21):
        stats = analyze_chapter(root, ch)
        report.chapters[ch] = stats
        report.total_files += stats.total_python
    
    report.total_chapters = len([s for s in report.chapters.values() if s.total_python > 0])
    
    # Analyze labs
    labs_dir = root / "labs"
    if labs_dir.exists():
        for lab_dir in labs_dir.iterdir():
            if lab_dir.is_dir() and not lab_dir.name.startswith('_'):
                stats = analyze_lab(root, lab_dir.name)
                report.labs[lab_dir.name] = stats
    
    report.total_labs = len(report.labs)
    
    # Count helper usage
    for ch_stats in report.chapters.values():
        report.metric_helper_usage[f"ch{ch_stats.chapter}"] = ch_stats.uses_helper
    
    return report


def print_text_report(report: CoverageReport) -> None:
    """Print human-readable report."""
    print("=" * 80)
    print("BENCHMARK COVERAGE REPORT")
    print("=" * 80)
    
    print(f"\nOverview:")
    print(f"  Total Python files: {report.total_files}")
    print(f"  Chapters with content: {report.total_chapters}")
    print(f"  Labs: {report.total_labs}")
    
    print(f"\n{'=' * 80}")
    print("CHAPTER BREAKDOWN")
    print("=" * 80)
    print(f"{'Ch':<4} {'Topic':<30} {'Base':>5} {'Opt':>5} {'Pair':>5} {'CUDA':>5} {'Metrics':>8}")
    print("-" * 80)
    
    for ch in sorted(report.chapters.keys()):
        stats = report.chapters[ch]
        if stats.total_python == 0:
            continue
        
        metrics_pct = (stats.has_get_custom_metrics / stats.total_python * 100) if stats.total_python > 0 else 0
        
        print(f"{ch:<4} {stats.topic[:30]:<30} {stats.baseline_count:>5} {stats.optimized_count:>5} "
              f"{stats.paired_count:>5} {stats.cuda_count:>5} {metrics_pct:>7.0f}%")
    
    print("-" * 80)
    totals = {
        "baseline": sum(s.baseline_count for s in report.chapters.values()),
        "optimized": sum(s.optimized_count for s in report.chapters.values()),
        "paired": sum(s.paired_count for s in report.chapters.values()),
        "cuda": sum(s.cuda_count for s in report.chapters.values()),
        "metrics": sum(s.has_get_custom_metrics for s in report.chapters.values()),
    }
    total_py = totals["baseline"] + totals["optimized"]
    metrics_pct = (totals["metrics"] / total_py * 100) if total_py > 0 else 0
    print(f"{'TOTAL':<4} {'':<30} {totals['baseline']:>5} {totals['optimized']:>5} "
          f"{totals['paired']:>5} {totals['cuda']:>5} {metrics_pct:>7.0f}%")
    
    print(f"\n{'=' * 80}")
    print("LABS")
    print("=" * 80)
    print(f"{'Lab':<35} {'Files':>6} {'README':>8} {'Base':>5} {'Opt':>5}")
    print("-" * 80)
    
    for name in sorted(report.labs.keys()):
        stats = report.labs[name]
        readme = "✅" if stats.has_readme else "❌"
        print(f"{name:<35} {stats.file_count:>6} {readme:>8} {stats.baseline_count:>5} {stats.optimized_count:>5}")
    
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find chapters with low metric coverage
    low_metrics = [
        (ch, stats) for ch, stats in report.chapters.items()
        if stats.total_python > 0 and stats.has_get_custom_metrics < stats.total_python * 0.8
    ]
    if low_metrics:
        print("\n📊 Low metric coverage (< 80%):")
        for ch, stats in low_metrics:
            pct = stats.has_get_custom_metrics / stats.total_python * 100
            print(f"  ch{ch}: {pct:.0f}% ({stats.has_get_custom_metrics}/{stats.total_python})")
    
    # Find labs without README
    no_readme = [name for name, stats in report.labs.items() if not stats.has_readme]
    if no_readme:
        print("\n📝 Labs without README:")
        for name in no_readme:
            print(f"  {name}")
    
    # Find unpaired files
    unpaired_chapters = [
        (ch, stats) for ch, stats in report.chapters.items()
        if stats.paired_count < min(stats.baseline_count, stats.optimized_count)
    ]
    if unpaired_chapters:
        print("\n🔗 Chapters with unpaired baseline/optimized:")
        for ch, stats in unpaired_chapters[:5]:
            print(f"  ch{ch}: {stats.baseline_count} baseline, {stats.optimized_count} optimized, {stats.paired_count} paired")


def print_json_report(report: CoverageReport) -> None:
    """Print JSON report."""
    # Convert to dict
    data = {
        "summary": {
            "total_files": report.total_files,
            "total_chapters": report.total_chapters,
            "total_labs": report.total_labs,
        },
        "chapters": {
            str(ch): {
                "topic": stats.topic,
                "baseline_count": stats.baseline_count,
                "optimized_count": stats.optimized_count,
                "cuda_count": stats.cuda_count,
                "paired_count": stats.paired_count,
                "has_get_custom_metrics": stats.has_get_custom_metrics,
                "uses_helper": stats.uses_helper,
            }
            for ch, stats in report.chapters.items()
        },
        "labs": {
            name: {
                "file_count": stats.file_count,
                "has_readme": stats.has_readme,
                "baseline_count": stats.baseline_count,
                "optimized_count": stats.optimized_count,
            }
            for name, stats in report.labs.items()
        },
    }
    print(json.dumps(data, indent=2))


def print_markdown_report(report: CoverageReport) -> None:
    """Print Markdown report."""
    print("# Benchmark Coverage Report\n")
    
    print("## Summary\n")
    print(f"- **Total Python files**: {report.total_files}")
    print(f"- **Chapters with content**: {report.total_chapters}")
    print(f"- **Labs**: {report.total_labs}\n")
    
    print("## Chapter Breakdown\n")
    print("| Ch | Topic | Baseline | Optimized | Paired | CUDA | Metrics |")
    print("|:---|:------|-------:|-------:|-------:|-----:|--------:|")
    
    for ch in sorted(report.chapters.keys()):
        stats = report.chapters[ch]
        if stats.total_python == 0:
            continue
        
        metrics_pct = (stats.has_get_custom_metrics / stats.total_python * 100) if stats.total_python > 0 else 0
        
        print(f"| {ch} | {stats.topic} | {stats.baseline_count} | {stats.optimized_count} | "
              f"{stats.paired_count} | {stats.cuda_count} | {metrics_pct:.0f}% |")
    
    print("\n## Labs\n")
    print("| Lab | Files | README | Baseline | Optimized |")
    print("|:----|------:|:------:|-------:|-------:|")
    
    for name in sorted(report.labs.keys()):
        stats = report.labs[name]
        readme = "✅" if stats.has_readme else "❌"
        print(f"| {name} | {stats.file_count} | {readme} | {stats.baseline_count} | {stats.optimized_count} |")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark coverage report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[2]
    report = generate_report(root)
    
    if args.json:
        print_json_report(report)
    elif args.markdown:
        print_markdown_report(report)
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
