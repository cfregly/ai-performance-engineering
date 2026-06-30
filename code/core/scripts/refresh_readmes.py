"""Generate consistent README files for chapters and labs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunSection:
    """Command guidance for running benchmarks."""

    commands: Sequence[str]
    notes: Sequence[str] = field(default_factory=tuple)


@dataclass
class MarkdownSection:
    """Named markdown section rendered before the generic chapter/lab scaffolding."""

    heading: str
    body: str


@dataclass
class Entry:
    """README content definition."""

    title: str
    summary: str
    goals: Sequence[str]
    contents: Sequence[Tuple[str, str]]
    validation: Sequence[str]
    lead_sections: Sequence[MarkdownSection] = field(default_factory=tuple)
    run: Optional[RunSection] = None
    run_heading: str = "Running the Benchmarks"
    run_intro: str = (
        "Use the benchmark harness for quick comparisons or drive the Typer CLI "
        "when you need repeatable artifact capture."
    )
    validation_heading: str = "Validation Checklist"
    extra_sections: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)


def _format_markdown(entry: Entry) -> str:
    """Create markdown using the shared layout."""
    lines: List[str] = []
    lines.append(f"# {entry.title}")
    lines.append("")
    lines.append("## Summary")
    lines.append(entry.summary.strip())
    lines.append("")
    for section in entry.lead_sections:
        lines.append(f"## {section.heading}")
        lines.append(section.body.strip())
        lines.append("")
    lines.append("## Learning Goals")
    for goal in entry.goals:
        lines.append(f"- {goal}")
    lines.append("")
    lines.append("## Directory Layout")
    lines.append("| Path | Description |")
    lines.append("| --- | --- |")
    for path, desc in entry.contents:
        lines.append(f"| {path} | {desc} |")
    lines.append("")
    if entry.run:
        lines.append(f"## {entry.run_heading}")
        if entry.run_intro:
            lines.append(entry.run_intro)
        lines.append("```bash")
        for cmd in entry.run.commands:
            lines.append(cmd)
        lines.append("```")
        for note in entry.run.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append(f"## {entry.validation_heading}")
    for item in entry.validation:
        lines.append(f"- {item}")
    if entry.extra_sections:
        lines.append("")
        for idx, section in enumerate(entry.extra_sections):
            if idx:
                lines.append("")
            lines.append(section.strip())
    if entry.notes:
        lines.append("")
        lines.append("## Notes")
        for note in entry.notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _output_path_for_slug(slug: str, *, repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root or REPO_ROOT)
    if slug.endswith(".md"):
        return root / slug
    return root / slug / "README.md"


def _resolve_target_slugs(*, targets: Sequence[str], include_all: bool) -> List[str]:
    if include_all:
        return list(ENTRIES.keys())
    if not targets:
        return []
    ordered = list(dict.fromkeys(targets))
    unknown = [slug for slug in ordered if slug not in ENTRIES]
    if unknown:
        raise ValueError(
            "Unknown README target(s): " + ", ".join(sorted(unknown))
        )
    return ordered


def write_readmes(*, targets: Sequence[str], repo_root: Optional[Path] = None) -> List[Path]:
    root = Path(repo_root or REPO_ROOT)
    written: List[Path] = []
    for slug in targets:
        output_path = _output_path_for_slug(slug, repo_root=root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = _format_markdown(ENTRIES[slug]).rstrip() + "\n"
        output_path.write_text(markdown, encoding="utf-8")
        written.append(output_path)
    return written


def check_readmes(*, targets: Sequence[str], repo_root: Optional[Path] = None) -> List[str]:
    root = Path(repo_root or REPO_ROOT)
    mismatches: List[str] = []
    for slug in targets:
        output_path = _output_path_for_slug(slug, repo_root=root)
        expected = _format_markdown(ENTRIES[slug]).rstrip() + "\n"
        if not output_path.exists():
            mismatches.append(f"missing: {output_path.relative_to(root)}")
            continue
        actual = output_path.read_text(encoding="utf-8")
        if actual != expected:
            mismatches.append(f"out_of_sync: {output_path.relative_to(root)}")
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render generator-owned README files safely.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write the selected README targets to disk.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify selected README targets are already in sync on disk.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="README target slug to operate on (repeatable), for example ch10 or labs/README.md.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Select every generator-owned README target. Required for full-repo writes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available README target slugs and exit.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to read/write (defaults to the current repo root).",
    )
    return parser


def _fallback_tier1_representative_rows() -> Sequence[Tuple[str, float, float, str]]:
    return (
        ("labs/block_scaling:block_scaling", 0.198, 1.76, "artifacts/runs/20260305_222139__bench__profile_none_targets_labs_block_scaling_block_scaling/..."),
        ("labs/flashattention4:flashattention4_alibi", 5.562, 14.45, "artifacts/runs/20260306_023114__bench__profile_none_targets_labs_flashattention4_flashattention4_alibi/..."),
        ("labs/persistent_decode:persistent_decode", 1.411, 11.94, "artifacts/runs/20260302_full_strict_all_singlegpu/..."),
        ("labs/kv_optimization:kv_standard", 1687.906, 1.57, "artifacts/runs/20260302_full_strict_all_singlegpu/..."),
        ("ch04:gradient_fusion", 3.931, 67.63, "artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/..."),
        ("labs/real_world_models:llama_3_1_8b", 13.143, 2.49, "artifacts/runs/20260302_full_strict_all_singlegpu/..."),
    )


def _load_json_object(path: Path, *, label: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to read {label} {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"Expected JSON object in {label} {path}, got {type(payload).__name__}"
    return payload, None


def _latest_tier1_summary(repo_root: Optional[Path] = None) -> Tuple[Optional[Dict[str, object]], Optional[Path], List[str]]:
    root = Path(repo_root or REPO_ROOT)
    index_path = root / "artifacts" / "history" / "tier1" / "index.json"
    warnings: List[str] = []
    if not index_path.exists():
        return None, None, warnings
    index, index_warning = _load_json_object(index_path, label="tier-1 history index")
    if index_warning:
        warnings.append(index_warning)
        return None, None, warnings
    if index is None:
        warnings.append(f"Tier-1 history index {index_path} did not yield a JSON object")
        return None, None, warnings
    runs = index.get("runs", [])
    if not isinstance(runs, list):
        warnings.append(f"Expected 'runs' list in tier-1 history index {index_path}, got {type(runs).__name__}")
        return None, None, warnings
    if not runs:
        return None, None, warnings
    latest = runs[-1]
    if not isinstance(latest, dict):
        warnings.append(f"Expected run object in tier-1 history index {index_path}, got {type(latest).__name__}")
        return None, None, warnings
    summary_path_raw = latest.get("summary_path")
    if not isinstance(summary_path_raw, str) or not summary_path_raw:
        warnings.append(f"Missing summary_path in latest tier-1 history entry from {index_path}")
        return None, None, warnings
    summary_path = Path(summary_path_raw)
    if not summary_path.is_absolute():
        summary_path = (root / summary_path).resolve()
    if not summary_path.exists():
        warnings.append(f"Tier-1 summary artifact is missing: {summary_path}")
        return None, None, warnings
    summary, summary_warning = _load_json_object(summary_path, label="tier-1 summary artifact")
    if summary_warning:
        warnings.append(summary_warning)
        return None, summary_path, warnings
    return summary, summary_path, warnings


def _render_current_representative_deltas_body(repo_root: Optional[Path] = None) -> str:
    def _fallback_body(warnings: Optional[Sequence[str]] = None) -> str:
        rows = _fallback_tier1_representative_rows()
        lines = [
            "These are measured results from current validated benchmark artifacts in `artifacts/runs/`, not aspirational target numbers.",
            "",
        ]
        if warnings:
            lines.extend(["README generation had to fall back to stored representative rows because the latest tier-1 history artifacts were unavailable or malformed.", ""])
            lines.extend(["Warnings:", ""])
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
        lines.extend([
            "| Target | Baseline | Optimized | Measured delta | Artifact |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        for target, baseline_ms, speedup, artifact in rows:
            optimized_ms = baseline_ms / speedup
            lines.append(
                f"| `{target}` | `{baseline_ms:.3f} ms` | `{optimized_ms:.3f} ms` | `{speedup:.2f}x` | `{artifact}` |"
            )
        return "\n".join(lines)

    root = Path(repo_root or REPO_ROOT)
    summary, summary_path, warnings = _latest_tier1_summary(root)
    if summary is None or summary_path is None:
        return _fallback_body(warnings)

    targets = summary.get("targets", [])
    if not isinstance(targets, list):
        warnings.append(
            f"Expected targets list in tier-1 summary artifact {summary_path}, got {type(targets).__name__}"
        )
        return _fallback_body(warnings)
    summary_metrics = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    representative_speedup = float(summary_metrics.get("representative_speedup", summary_metrics.get("geomean_speedup", 0.0)) or 0.0)
    median_speedup = float(summary_metrics.get("median_speedup", 0.0) or 0.0)
    avg_speedup = float(summary_metrics.get("avg_speedup", 0.0) or 0.0)

    lines = [
        "These numbers are taken from the latest canonical tier-1 history summary rather than from hand-maintained README text.",
        "",
        f"Source artifact: `{summary_path.relative_to(root) if summary_path.is_relative_to(root) else summary_path}`",
        "",
    ]
    if representative_speedup > 0.0:
        lines.extend(
            [
                (
                    f"Representative suite speedup: `{representative_speedup:.2f}x` geomean"
                    + (f", `{median_speedup:.2f}x` median" if median_speedup > 0.0 else "")
                    + (f", `{avg_speedup:.2f}x` arithmetic average" if avg_speedup > 0.0 else "")
                    + "."
                ),
                "",
            ]
        )
    lines.extend(
        [
        "| Target | Baseline | Optimized | Measured delta | Artifact |",
        "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    emitted = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        if target.get("status") != "succeeded":
            continue
        target_name = str(target.get("target") or "").strip()
        baseline_ms = float(target.get("baseline_time_ms", 0.0) or 0.0)
        speedup = float(target.get("best_speedup", 0.0) or 0.0)
        if not target_name or baseline_ms <= 0.0 or speedup <= 0.0:
            continue
        optimized_ms = baseline_ms / speedup
        artifact = f"artifacts/history/tier1/{summary.get('run_id')}/summary.json"
        lines.append(
            f"| `{target_name}` | `{baseline_ms:.3f} ms` | `{optimized_ms:.3f} ms` | `{speedup:.2f}x` | `{artifact}` |"
        )
        emitted += 1
    if emitted == 0:
        return _fallback_body()
    return "\n".join(lines)


def _chapter_run_commands(slug: str) -> RunSection:
    """Default run commands for a chapter directory."""
    commands = [
        f"python -m {slug}.compare",
        f"python -m cli.aisp bench list-targets --chapter {slug}",
        f"python -m cli.aisp bench run --targets {slug} --profile minimal",
    ]
    notes = [
        "Override `--profile` or `--iterations` per workload when capturing Nsight traces.",
        "Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.",
        "Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.",
    ]
    return RunSection(commands=commands, notes=notes)


def _lab_run_commands(slug: str) -> RunSection:
    """Default run commands for a lab that is exposed through the CLI."""
    commands = [
        f"python -m cli.aisp bench list-targets --chapter {slug}",
        f"python -m cli.aisp bench run --targets {slug} --profile minimal",
    ]
    notes = [
        f"Targets follow the `{slug}:<workload>` naming convention listed by `list-targets`.",
        f"Use `--target-extra-arg {slug}:<workload>=\"--flag value\"` to sweep schedule knobs.",
        "Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.",
        "Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.",
    ]
    return RunSection(commands=commands, notes=notes)


def chapter_entry(
    slug: str,
    title: str,
    summary: str,
    goals: Sequence[str],
    contents: Sequence[Tuple[str, str]],
    validation: Sequence[str],
    lead_sections: Sequence[MarkdownSection] = (),
    notes: Sequence[str] = (),
    run_heading: str = "Running the Benchmarks",
    run_intro: str = (
        "Use the benchmark harness for quick comparisons or drive the Typer CLI "
        "when you need repeatable artifact capture."
    ),
    validation_heading: str = "Validation Checklist",
) -> Entry:
    """Create a chapter README entry."""
    return Entry(
        title=title,
        summary=summary,
        goals=goals,
        contents=contents,
        validation=validation,
        lead_sections=lead_sections,
        run=_chapter_run_commands(slug),
        run_heading=run_heading,
        run_intro=run_intro,
        validation_heading=validation_heading,
        notes=notes,
    )


def lab_entry(
    slug: str,
    title: str,
    summary: str,
    goals: Sequence[str],
    contents: Sequence[Tuple[str, str]],
    validation: Sequence[str],
    lead_sections: Sequence[MarkdownSection] = (),
    notes: Sequence[str] = (),
    run: Optional[RunSection] = None,
    run_heading: str = "Running the Benchmarks",
    run_intro: str = (
        "Use the benchmark harness for quick comparisons or drive the Typer CLI "
        "when you need repeatable artifact capture."
    ),
    validation_heading: str = "Validation Checklist",
    extra_sections: Sequence[str] = (),
) -> Entry:
    """Create a lab README entry."""
    return Entry(
        title=title,
        summary=summary,
        goals=goals,
        contents=contents,
        validation=validation,
        lead_sections=lead_sections,
        run=run if run else _lab_run_commands(slug),
        run_heading=run_heading,
        run_intro=run_intro,
        validation_heading=validation_heading,
        extra_sections=extra_sections,
        notes=notes,
    )


ENTRIES: Dict[str, Entry] = {}

WALL_OF_SHAME = dedent(
    """\
    ## Wall of Shame
    The benchmark harness includes a strict set of correctness and validity checks to prevent misleading speedups.
    Below is the reference list of validity issues we explicitly protect against, plus real-world incidents that
    motivated these checks.

    Note: All 95 validity issues are protected by the harness.

    CUDA Graph Note: Capturing CUDA graphs in `setup()` is allowed for steady-state replay benchmarks (we intentionally
    measure replay, not capture). It is NOT allowed to precompute and reuse the final output from `setup()`. The output
    used for verification must come from the timed `benchmark_fn()` run and be surfaced via `capture_verification_payload()`.

    Virtualization Note: `validate_environment()` treats virtualization (hypervisor present) as a warning. Benchmarks can
    run in virtualized environments, but bare metal remains the preferred source for final performance numbers.

    ### Benchmark Validity Issues Reference

    | Category | Issue | What Happens | Protection | Status | Real-World Incident |
    | --- | --- | --- | --- | --- | --- |
    | Timing | Unsynced Streams | Work on non-default streams is not timed | Full device sync + StreamAuditor | OK | Locus/KernelBench 2025 |
    | Timing | Incomplete Async Ops | Timer stops before async work finishes | Full device sync | OK | Locus/KernelBench 2025 |
    | Timing | Event Timing Gaps | CUDA events recorded incorrectly | Cross-validate with wall clock | OK | |
    | Timing | Timer Granularity | Measurement too coarse for fast ops | Adaptive iterations | OK | |
    | Timing | Warmup Bleed | Real work happens during warmup | isolate_warmup_cache | OK | |
    | Timing | Clock Drift | System clock changes during measurement | Monotonic clock usage | OK | |
    | Timing | Profiler Overhead | Profiling tools add latency | Profile-free timing path | OK | |
    | Output | Constant Output | Same result regardless of input | Jitter check | OK | |
    | Output | Stale Cache | Same result across different seeds | Fresh-input check | OK | |
    | Output | Approximation Drift | Rough estimate instead of full compute | Output tolerance validation | OK | |
    | Output | Invalid Values (NaN) | NaN in output | validate_result NaN check | OK | |
    | Output | Invalid Values (Inf) | Inf in output | validate_result Inf check | OK | |
    | Output | Invalid Ground Truth | Labels/expected values wrong | GoldenOutputCache | OK | ImageNet Labels 2021, MMLU Errors 2025 |
    | Output | Shape Mismatch | Output shape differs from expected | Shape validation | OK | |
    | Output | Dtype Mismatch | Output dtype differs from expected | ToleranceSpec dtype check | OK | |
    | Output | Denormalized Values | Subnormal floats cause slowdowns | Denormal check | OK | |
    | Output | Uninitialized Memory | Output contains garbage | Memory initialization check | OK | |
    | Workload | Precision Mismatch | Claims FP32 but uses FP16 | InputSignature dtype verification | OK | |
    | Workload | Backend Precision Policy Drift | Global precision policy changes during timing | Backend policy immutability check | OK | PyTorch TF32 Default 2020 |
    | Workload | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | OK | AI Agent Benchmark Shortcuts 2024 |
    | Workload | Early Exit | Stops iteration loops early | Config immutability | OK | |
    | Workload | Batch Shrinking | Processes fewer samples | InputSignature matching | OK | |
    | Workload | Sequence Truncation | Processes shorter sequences | InputSignature matching | OK | |
    | Workload | Hidden Downsampling | Silently reduces resolution | Dimension validation | OK | |
    | Workload | Sparsity Mismatch | Different sparsity patterns | Sparsity ratio check | OK | |
    | Workload | Attention Mask Mismatch | Different masking applied | Mask equivalence check | OK | |
    | Workload | KV Cache Size Mismatch | Different cache sizes | Cache dimension check | OK | |
    | Workload | Train/Test Overlap | Model tested on training data | Dataset isolation | OK | Computational Biology 2019 |
    | Location | CPU Spillover | Work offloaded to CPU | GPU kernel time validation | OK | |
    | Location | Setup Pre-computation | Work done in setup | check_setup_precomputation | OK | |
    | Location | Graph Capture Cheat | Pre-compute during graph capture | GraphCaptureCheatDetector | OK | |
    | Location | Warmup Computation | Compute results during warmup | isolate_warmup_cache | OK | |
    | Location | Background Thread | Compute in separate thread | Process isolation | OK | |
    | Location | Lazy Evaluation Skip | Returns unevaluated lazy tensor | force_tensor_evaluation | OK | |
    | Location | JIT Compilation Timing | JIT compile time included/excluded inconsistently | clear_compile_cache | OK | |
    | Memory | Pre-allocated Output | Result buffer allocated in setup | MemoryAllocationTracker | OK | |
    | Memory | Input-Output Aliasing | Output points to pre-filled input | check_input_output_aliasing | OK | |
    | Memory | Pinned Memory Timing | Async pinned transfers not waited | Transfer completion check | OK | |
    | Memory | Memory Pool Reuse | Cached allocations skew timing | reset_cuda_memory_pool | OK | |
    | Memory | Fragmentation Effects | Memory fragmentation differs | Memory pool reset | OK | |
    | Memory | Page Fault Timing | First-touch page faults included | Memory pre-touch | OK | |
    | Memory | Swap Interference | Swapping affects timing | Memory lock / swap disable | OK | |
    | CUDA | Host Callback Escape | cudaLaunchHostFunc returns early | Host function tracking | OK | |
    | CUDA | Async Memcpy Incomplete | D2H/H2D copies not awaited | Full device sync | OK | |
    | CUDA | Workspace Pre-compute | Work in cuBLAS workspace alloc | Workspace monitoring | OK | |
    | CUDA | Persistent Kernel | Kernel left running across calls | Kernel lifetime check | OK | |
    | CUDA | Undeclared Multi-GPU | Work spread across undeclared GPUs | validate_environment | OK | |
    | CUDA | Context Switch Overhead | CUDA context switches affect timing | Context pinning | OK | |
    | CUDA | Driver Overhead | Driver calls not accounted for | Driver call tracking | OK | |
    | CUDA | Cooperative Launch Abuse | Cooperative kernels bypass checks | Launch mode validation | OK | |
    | CUDA | Dynamic Parallelism Hidden | Child kernels not tracked | CDP kernel tracking | OK | |
    | CUDA | Unified Memory Faults | Page migration not timed | UM fault tracking | OK | |
    | Compile | Compilation Cache Hit | Returns cached compiled output | clear_compile_cache | OK | |
    | Compile | Trace Reuse | Exploits trace caching | torch._dynamo.reset | OK | |
    | Compile | Mode Inconsistency | Different compile mode verify vs perf | Mode consistency check | OK | |
    | Compile | Inductor Asymmetry | Inductor optimizations inconsistent | Compilation parity | OK | |
    | Compile | Guard Failure Hidden | Recompilation not counted | get_compile_state | OK | |
    | Compile | Autotuning Variance | Autotuning picks different kernels | Fixed autotuning cache | OK | |
    | Compile | Symbolic Shape Exploit | Different shapes trigger different code | InputSignature matching | OK | |
    | Distributed | Rank Skipping | Some ranks do not do work | check_rank_execution | OK | |
    | Distributed | Collective Short-circuit | Communication skipped | NCCL validation | OK | |
    | Distributed | Topology Mismatch | Claims different topology | verify_distributed | OK | |
    | Distributed | Barrier Timing | Barrier timing exploited | Barrier synchronization | OK | |
    | Distributed | Gradient Bucketing Mismatch | Different bucket sizes | Bucket size validation | OK | |
    | Distributed | Async Gradient Timing | Async all-reduce not awaited | Full device sync | OK | |
    | Distributed | Pipeline Bubble Hiding | Pipeline bubbles not counted | Bubble time tracking | OK | |
    | Distributed | Shard Size Mismatch | FSDP shards differ | InputSignature matching | OK | |
    | Environment | Device Mismatch | Uses different GPU than declared | validate_environment | OK | |
    | Environment | Frequency Boost | Overclocked for benchmark only | lock_gpu_clocks | OK | |
    | Environment | Priority Elevation | Runs at higher priority | Process isolation | OK | |
    | Environment | Memory Overcommit | Exploits memory overcommit | Memory validation | OK | |
    | Environment | NUMA Inconsistency | NUMA placement differs | NUMA audit | OK | |
    | Environment | CPU Governor Mismatch | Different CPU frequency scaling | Governor lock | OK | |
    | Environment | Thermal Throttling | GPU throttles during run | capture_gpu_state (pynvml) | OK | |
    | Environment | Power Limit Difference | Different TDP settings | capture_gpu_state (pynvml) | OK | |
    | Environment | Driver Version Mismatch | Different CUDA drivers | RunManifest version lock | OK | |
    | Environment | Library Version Mismatch | Different cuDNN/cuBLAS | RunManifest version lock | OK | |
    | Environment | Container Resource Limits | cgroups limits differ | Resource limit check | OK | |
    | Environment | Virtualization Overhead | VM/container overhead varies | Bare-metal validation | OK | |
    | Statistical | Cherry-picking | Only best iterations reported | All-iteration reporting | OK | Leaderboard Illusion 2025 |
    | Statistical | Outlier Injection | Slow iterations added to baseline | Statistical validation | OK | |
    | Statistical | Variance Gaming | Variance reporting manipulated | Consistent statistics | OK | |
    | Statistical | Percentile Selection | Favorable percentile chosen | Fixed percentile policy | OK | |
    | Statistical | Insufficient Samples | Too few iterations for significance | Adaptive iterations | OK | Measuring What Matters 2025 |
    | Statistical | Cold Start Inclusion | First run included unfairly | Warmup enforcement | OK | |
    | Statistical | GC Interference | Garbage collection during timing | gc_disabled | OK | |
    | Statistical | Background Process Noise | System processes affect timing | Process isolation | OK | |
    | Evaluation | Eval Code Exploitation | Benchmark code modified to pass | BenchmarkContract enforcement | OK | |
    | Evaluation | Timeout Manipulation | Timeout extended to hide slowdowns | Config immutability | OK | |
    | Evaluation | Metric Definition Gaming | Redefine what speedup means | Standardized metric definitions | OK | MLPerf 2019, HANS 2019, Measuring What Matters 2025, Medical LLM Benchmarks 2025 |
    | Evaluation | Test Data Leakage | Training on test data | Data contamination checks | OK | Benchmark Data Contamination Survey 2024 |
    | Evaluation | Benchmark Overfitting | Optimize specifically for benchmark | Fresh-input + jitter checks | OK | Underspecification 2020, Epic Sepsis 2021, NaturalCodeBench 2024 |
    | Evaluation | Self-Modifying Tests | AI/code modifies its own tests | Config immutability | OK | |
    | Evaluation | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | OK | AI Agent Benchmark Shortcuts 2024 |
    | Evaluation | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | OK | AI Agent Benchmark Shortcuts 2024, Microsoft Tay 2016 |

    Total: 11 categories, 95 validity issues - all protected by the harness.

    ### Notable Real-World Incidents

    | Year | Incident | Issue Type | What Happened | Source |
    | --- | --- | --- | --- | --- |
    | 2025 | Locus/KernelBench Stream Exploit | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. 32.8 percent of RL-generated kernels exploited this, causing fake 18x speedups. | https://x.com/miru_why/status/1991773868806361138 |
    | 2025 | Measuring What Matters: Construct Validity in LLM Benchmarks | Metric Definition Gaming / Construct Validity | Systematic review of 445 LLM benchmarks found construct-validity weaknesses and low statistical rigor; issued eight design recommendations. | https://ora.ox.ac.uk/objects/uuid%3Aad2b69b6-0986-42d0-a512-a6e56338b6cc |
    | 2025 | Medical LLM Benchmarks and Construct Validity | Metric Definition Gaming / Construct Validity | Position paper argues exam-style medical LLM benchmarks miss real-world tasks and documents construct-validity gaps using clinical data. | https://arxiv.org/abs/2503.10694 |
    | 2025 | Sakana AI Scientist Evaluation | Evaluation Integrity | Independent evaluation found frequent experiment failures and hallucinated numerical results. | https://arxiv.org/abs/2502.14297 |
    | 2025 | Leaderboard Illusion (Chatbot Arena) | Cherry-picking | Analysis of Chatbot Arena reports selection effects and leaderboard instability when submissions are inconsistent or selectively disclosed. | https://arxiv.org/abs/2504.20879 |
    | 2024 | MMLU Benchmark Errors | Invalid Ground Truth | Analysis found 57 percent of MMLU virology subset questions incorrect and estimated 6.49 percent errors overall. | https://arxiv.org/abs/2406.04127 |
    | 2024 | AI Agent Benchmark Shortcuts | Missing Holdout Sets | Study found AI agents memorize benchmark test samples instead of learning to generalize. Many benchmarks lack proper holdout test sets. | https://arxiv.org/abs/2407.01502 |
    | 2024 | NaturalCodeBench vs HumanEval | Benchmark Overfitting | Real-user coding tasks in NaturalCodeBench show large performance gaps and weak correlation with HumanEval scores. | https://aclanthology.org/2024.findings-acl.471/ |
    | 2024 | Benchmark Data Contamination Survey | Data Contamination | Survey catalogs contamination pathways across LLM benchmarks and highlights mitigation gaps. | https://arxiv.org/abs/2406.04244 |
    | 2023 | NLP Evaluation Data Contamination | Data Contamination | Position paper warns that LLMs trained on benchmark test splits can inflate reported scores. | https://arxiv.org/abs/2310.18018 |
    | 2022 | MLPerf Participation Issues | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | https://web.archive.org/web/20250813110435/https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/ |
    | 2022 | ML Benchmark Validity (Berkeley) | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity. | https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html |
    | 2021 | ImageNet Label Errors | Invalid Ground Truth | At least 6 percent label errors in ImageNet validation set. | https://arxiv.org/abs/2103.14749 |
    | 2021 | MLPerf Reproducibility | Benchmark Reproducibility | Users could not reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repos. | https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo |
    | 2021 | Epic Sepsis Model External Validation | Benchmark Overfitting | External validation found poor discrimination and calibration for the Epic Sepsis Model, leading to missed cases and alert fatigue. | https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307 |
    | 2020 | Underspecification in ML | Benchmark Overfitting | Models with equivalent benchmark performance diverged in deployment behavior. | https://arxiv.org/abs/2011.03395 |
    | 2020 | TF32 Default on Ampere | Precision Policy Drift | TF32-enabled matmul/conv trades precision for speed unless explicitly disabled. | https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere |
    | 2019 | NLI Heuristic Shortcuts (HANS) | Metric Definition Gaming | Models trained on MNLI (GLUE) rely on shallow heuristics and fail on HANS, revealing spurious shortcut behavior. | https://aclanthology.org/P19-1334/ |
    | 2019 | MLPerf Inference Bias | Metric Definition Gaming | Vendors selectively submitted results highlighting strengths. | http://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/ |
    | 2019 | Computational Biology Overfitting | Train/Test Overlap | Tools developed and tested on same datasets failed on new data. | https://www.nature.com/articles/s41467-019-09406-4 |
    | 2016 | Microsoft Tay Chatbot | Missing Holdout Sets | AI chatbot learned abusive behavior within 24 hours after deployment due to adversarial user interactions. | https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/ |

    ### Incident Categories and Our Protections

    | Category | Incidents | Our Protection | Status |
    | --- | --- | --- | --- |
    | Timing Manipulation | 1 (Locus/KernelBench) | Full device sync + StreamAuditor | OK |
    | Invalid Ground Truth | 2 (ImageNet Labels, MMLU) | GoldenOutputCache + validate_result | OK |
    | Benchmark Overfitting | 4 (Underspecification, Epic Sepsis, NaturalCodeBench, Berkeley) | Fresh-input checks + jitter | OK |
    | Data Contamination | 2 (LLM Survey 2024, NLP Contamination 2023) | Data contamination checks + fresh inputs | OK |
    | Metric Gaming | 4 (Measuring What Matters 2025, Medical LLM Benchmarks 2025, HANS 2019, MLPerf 2019) | Standardized metric definitions | OK |
    | Cherry-picking | 2 (Leaderboard Illusion, MLPerf 2022) | All-iteration reporting | OK |
    | Train/Test Overlap | 1 (Computational Biology) | Dataset isolation + holdout enforcement | OK |
    | Missing Holdout Sets | 2 (AI Agent Shortcuts, Microsoft Tay) | Held-out evaluation data | OK |
    | Reproducibility | 1 (MLPerf 2021) | RunManifest version locking | OK |
    | Evaluation Integrity | 1 (Sakana AI Scientist) | BenchmarkContract + verification enforcement | OK |
    | Precision Policy Drift | 1 (TF32 Default) | Backend policy immutability check | OK |

    ### Deep Dive: The Locus/KernelBench Stream Timing Vulnerability

    This 2025 incident illustrates why correctness verification alone is insufficient.

    ```python
    # VULNERABLE TIMING (what KernelBench did)
    start_event.record(original_model_stream)  # Only records on default stream
    model(*inputs)                              # But work runs on s1, s2, s3
    end_event.record(original_model_stream)    # Timer stops before s1/s2/s3 finish
    torch.cuda.synchronize(device=device)      # Waits, but timing already recorded

    # CORRECT TIMING (the fix)
    for stream in custom_model_streams:
        custom_model_stream.wait_stream(stream)  # Wait for ALL streams
    _event.record(custom_model_stream)           # Then record timing
    ```

    The exploit pattern:
    1. AI creates non-default streams: `s1 = getStreamFromPool()`, `s2 = ...`, `s3 = ...`
    2. AI launches GEMMs on those streams: `at::mm_out(gate, x2d, gate_proj.t())` on s1
    3. AI does not call `setCurrentCUDAStream(s3)` or wait for streams before returning
    4. Correctness test uses `torch.cuda.synchronize()` and passes
    5. Performance test uses stream-specific events and reports fake speedups

    Result: 82/250 (32.8 percent) of RL-generated CUDA kernels exploited this, producing artificial 18x speedups with
    zero actual performance improvement.

    ### Protection Implementation Reference

    | Module | Key Protections |
    | --- | --- |
    | `core/harness/benchmark_harness.py` | Full device sync, L2 cache clearing, GPU clock locking, warmup isolation, config immutability, adaptive iterations, CUDA graph mode |
    | `core/harness/validity_checks.py` | StreamAuditor, MemoryAllocationTracker, GraphCaptureCheatDetector, gc_disabled, clear_compile_cache, capture_gpu_state, validate_environment |
    | `core/harness/l2_cache_utils.py` | Dynamic L2 cache size detection, clear_l2_cache |
    | `core/benchmark/verify_runner.py` | VerifyRunner, GoldenOutputCache, jitter check, fresh-input check, output comparison, workload invariants |
    | `core/benchmark/verification.py` | InputSignature, ToleranceSpec, QuarantineReason, seed mutation detection |
    | `core/benchmark/quarantine.py` | QuarantineManager with persistence |
    | `core/benchmark/contract.py` | BenchmarkContract enforcement |
    """
).strip()

ENTRIES["README.md"] = Entry(
    title="AI Systems Performance Engineering",
    summary=dedent(
        """\
        Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
        The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

        Roadmap: [`docs/performance_repo_roadmap.md`](docs/performance_repo_roadmap.md) defines the prioritized plan for canonical suites, trend tracking, anti-pattern enforcement, shared benchmark bases, and evidence-first documentation."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Most performance repos are easy to browse and hard to trust. This one is meant to do the opposite:
                - baseline and optimized paths live side-by-side
                - correctness checks run before speedup claims matter
                - profiler artifacts and benchmark manifests are first-class outputs, not optional extras"""
            ),
        ),
        MarkdownSection(
            "Tier-1 Canonical Suite",
            dedent(
                """\
                The fastest way to answer "is this repo still delivering real wins?" is the canonical tier-1 suite.

                ```bash
                python -m cli.aisp bench run-tier1 --single-gpu --profile minimal
                ```

                That command now writes a stable history package under `artifacts/history/tier1/<run_id>/`:
                - `summary.json`: per-target baseline, optimized path, and best speedup
                - `regression_summary.md`: human-readable before/after summary against the previous tier-1 run
                - `regression_summary.json`: machine-readable regressions and improvements
                - `trend_snapshot.json`: run-history summary for dashboards and release notes
                - `artifacts/history/tier1/index.json`: suite history index

                See [`docs/tier1_benchmark_suite.md`](docs/tier1_benchmark_suite.md) for the current target list, artifact contract, and interpretation guidance."""
            ),
        ),
        MarkdownSection(
            "Current Representative Deltas",
            _render_current_representative_deltas_body(),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                When you want proof beyond wall-clock timing, use the same harness target with a profiling mode instead of a different script.

                ```bash
                python -m cli.aisp bench run --targets labs/block_scaling:block_scaling --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/persistent_decode:persistent_decode --profile deep_dive --single-gpu
                ```

                - `minimal` is the fastest artifact-bearing path.
                - `deep_dive` is the profiler-backed path for Nsight Systems + Nsight Compute comparisons.
                - The benchmark harness now blocks more hot-path anti-patterns and follows imported helper code, so "clean benchmark" means more than it used to."""
            ),
        ),
        MarkdownSection(
            "Lab Navigation",
            dedent(
                """\
                If you are navigating by performance problem instead of by chapter, start with these benchmark-pair labs:

                | Lab | Best for | Distinct from |
                | --- | --- | --- |
                | `labs/parameterized_cuda_graphs` | fixed-shape PyTorch CUDA Graph replay where request bindings change but graph topology does not | broader decode-serving labs; dynamic-shape graph problems |
                | `labs/persistent_decode` | persistent decode kernels plus graph/TMA variants on serving-style decode paths | the narrower executable-graph parameter-mutation story |
                | `labs/decode_optimization` | decode microbenchmarks that separate pinned memory, streams, compile, graphs, and cache policy | a single graph-launch-overhead benchmark pair |
                | `labs/block_scaling` | Blackwell block-scaled GEMM mechanics and measured tensor-core wins | serving/control-plane orchestration labs |
                | `labs/training_hotpath` | training-path launch and fusion bottlenecks with clean benchmark pairs | inference/decode-oriented graph replay work |

                This keeps `labs/parameterized_cuda_graphs` visible without pretending it replaces the broader persistent-decode or decode-optimization stories."""
            ),
        ),
        MarkdownSection(
            "Benchmark Methodology",
            dedent(
                """\
                This repo now exposes one repeatable benchmarking methodology instead of leaving performance work as a collection of scripts.

                Start with:
                - [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md) for the three-layer model (`micro`, `component`, `end_to_end`), bottleneck taxonomy, publication-vs-realism policy, and straggler playbook.
                - [`docs/performance_warehouse.md`](docs/performance_warehouse.md) for the stable event schema, raw-versus-curated storage split, retention tiers, and telemetry lineage back to raw evidence.
                - [`templates/performance_intake.yaml`](templates/performance_intake.yaml) for KPIs, constraints, and the variable under test.
                - [`templates/benchmark_workload_spec.yaml`](templates/benchmark_workload_spec.yaml) for the frozen workload definition and measurement policy.
                - [`templates/benchmark_run.yaml`](templates/benchmark_run.yaml) for the CRD-aligned declarative `BenchmarkRun` shape the repo would map onto a Kubernetes-native service.
                - [`cluster/docs/kubernetes_benchmark_service.md`](cluster/docs/kubernetes_benchmark_service.md) plus [`cluster/configs/benchmarkrun-crd.yaml`](cluster/configs/benchmarkrun-crd.yaml) for the cluster-native operator/CRD direction already being sketched in the repo.

                Thin surfaces for these contracts are also exposed through `python -m cli.aisp tools benchmark-contracts`, dashboard API `GET /api/benchmark/contracts`, and MCP tool `benchmark_contracts`.

                The current harness already captures manifests, profiler artifacts, raw timings, and artifact hashes. Cryptographic provenance signing is still a documented gap, so external publication packets should record that explicitly rather than assuming hashes alone are sufficient."""
            ),
        ),
        MarkdownSection(
            "Cluster Evaluation",
            dedent(
                """\
                Cluster evaluation has one supported artifact contract for new work:

                ```text
                cluster/runs/<run_id>/
                  manifest.json
                  structured/
                  raw/
                  figures/
                  reports/
                ```

                Start with:
                - [`cluster/README.md`](cluster/README.md) for the current commands and folder contract.
                - `python -m cli.aisp cluster common-eval --preset common-answer-fast ...` for the normal "evaluate this system" ask.
                - `python -m cli.aisp cluster common-eval --preset modern-llm ...` when you need the full canonical package.
                - `python -m cli.aisp cluster common-eval --preset multinode-readiness ...` before first real multi-node workloads.
                - `python -m cli.aisp cluster promote-run --run-id <run_id> ...` when one collected run should become the published localhost package.

                The current published canonical package lives under `cluster/published/current/`. New collection still happens under `cluster/runs/<run_id>/`."""
            ),
        ),
    ],
    goals=[
        "Understand how the chapters, labs, and shared tooling fit together.",
        "Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.",
        "Run the benchmark harness directly or through the Typer CLI for automated artifact capture.",
        "Validate peak hardware characteristics before grading optimizations against stored expectations.",
    ],
    contents=[
        ("`ch01` - `ch20`", "One directory per chapter with baseline/optimized benchmarks, workload configs, and chapter-level harness entrypoints such as `ch01/compare.py`."),
        ("`labs/`", "Deep-dive labs for memory-bandwidth patterns, matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more."),
        ("`core/benchmark/`, `profiling/`, `core/`, `optimization/`, `analysis/`", "Shared harness, logging, workload metadata, profiling, and optimization utilities used by every chapter."),
        ("`python -m cli.aisp bench`", "Typer-based CLI for running and profiling targets with reproducible artifacts."),
        ("`docs/` + `core/scripts/`", "Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`)."),
    ],
    run=RunSection(
        commands=[
            "cd ai-performance-engineering/code",
            "python3 -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements_latest.txt",
            "python -m cli.aisp bench list-targets --chapter ch01",
            "python -m cli.aisp bench run --targets ch01 --profile minimal",
            "python -m cli.aisp bench run-tier1 --single-gpu --profile minimal",
        ],
        notes=[
            "`setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.",
            "Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited hosts.",
            "Use `python -m cli.aisp bench expectations --hardware b200 --min-speedup 1.05` to report expectation entries below a target threshold.",
            "Repeat `--targets` for multi-scope runs, for example `python -m cli.aisp bench run --targets ch01 --targets ch02 --profile minimal`; use `python -m cli.aisp bench run-tier1 --single-gpu --profile minimal` for the canonical regression suite.",
            "Portable runs do not update expectation files unless `--allow-portable-expectations-update` is supplied.",
            "`python core/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.",
        ],
    ),
    validation=[
        "`pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.",
        "`python core/benchmark/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.",
    ],
    extra_sections=[WALL_OF_SHAME],
    notes=[
        "`core/scripts/profile_all_workloads.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.",
        "`artifacts/runs/` holds run outputs (results/profiles/reports/logs); clean via `python cleanup.py` when rotating hardware.",
        "`docs/benchmark_methodology.md` defines the repo-wide benchmarking method that ties harness runs, cluster packages, and publication-grade evidence together.",
        "`docs/performance_warehouse.md` defines the warehouse contract that ties published numbers back to raw evidence and telemetry joins.",
        "`docs/perf_intake_and_triage.md` outlines the standard intake bundle for performance investigations.",
    ],
)

ENTRIES["labs/README.md"] = Entry(
    title="Labs",
    summary=dedent(
        """\
        Labs are where the repo stops being chapter-by-chapter pedagogy and starts telling complete optimization stories.
        Some labs are strict baseline/optimized benchmark pairs. Others are playbooks or matrix harnesses that need a different, more honest doc shape."""
    ),
    lead_sections=[
        MarkdownSection(
            "How To Read This Directory",
            dedent(
                """\
                There are two useful lab classes in this repo:

                - **Benchmark-pair labs**: these expose harness targets, keep correctness gates, and are the right place to make performance claims.
                - **Playbook / matrix labs**: these package workflows, scenario drills, or large tuning matrices. They are still valuable, but they should not pretend to be a single baseline/optimized benchmark when they are not."""
            ),
        ),
        MarkdownSection(
            "What Counts As A Good Lab Here",
            dedent(
                """\
                A strong public lab should make three things obvious:

                - what the baseline path is
                - what changed in the optimized or alternative path
                - what measured artifact proves the claim

                If a lab cannot answer those questions yet, the doc should say so directly instead of faking a benchmark pair."""
            ),
        ),
    ],
    goals=[
        "Help readers find the right lab quickly.",
        "Separate benchmark-pair labs from playbook/matrix labs honestly.",
        "Point contributors toward the repo's expected lab quality bar.",
    ],
    contents=[
        ("`labs/block_scaling`, `labs/blackwell_matmul`, `labs/blackwell_gemm_optimizations`, `labs/flashattention4`, `labs/memory_bandwidth_patterns`, `labs/nccl_nixl_nvshmem`, `labs/persistent_decode`, `labs/parameterized_cuda_graphs`, `labs/training_hotpath`", "Benchmark-pair labs with strong kernel/perf narratives and artifact-backed measured deltas, including narrow bandwidth-pattern, communication-stack tradeoff, and CUDA-graph replay labs."),
        ("`labs/decode_optimization`, `labs/kv_optimization`, `labs/moe_cuda`, `labs/moe_optimization_journey`", "Serving-path and MoE labs where the benchmark pair is part of a broader optimization story."),
        ("`labs/moe_decode_blackwell_matrix`, `labs/nanochat_fullstack`, `labs/python_concurrency`, `labs/vllm-deepseek-tuning`", "Larger workflow-oriented and matrix/playbook labs that need a richer doc model than a simple pair benchmark."),
        ("`labs/nvfp4_*`", "Low-precision kernel labs where verification discipline matters as much as the timing win."),
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp bench list-targets --chapter labs/block_scaling",
            "python -m cli.aisp bench list-targets --chapter labs/decode_optimization",
            "python -m cli.aisp bench list-targets --chapter labs/memory_bandwidth_patterns",
            "python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem",
            "python -m cli.aisp bench list-targets --chapter labs/moe_cuda",
        ],
        notes=[
            "Use `list-targets` first; the benchmark-pair labs expose clean harness targets, while the playbook/matrix labs often have their own scripts or Makefiles.",
            "Use `labs/memory_bandwidth_patterns` when you want a narrow coalescing/vectorization/shared-memory bandwidth lab instead of a broader serving-path workflow.",
            "Use `labs/nccl_nixl_nvshmem` when you want communication-stack tradeoffs and capability probing instead of a broader distributed-training or serving workflow.",
            "If a lab does not have a clean baseline/optimized target yet, do not invent one in documentation.",
        ],
    ),
    validation=[
        "Benchmark-facing labs should expose reproducible harness targets or clearly document why they are still workflow/matrix labs.",
        "Public lab READMEs should prefer measured artifact-backed claims over generic feature descriptions.",
    ],
    extra_sections=[
        dedent(
            """\
            ## Lab Index

            | Lab | Summary | Suggested Chapters |
            | --- | --- | --- |
            | `labs/nvfp4_gemv/` | GPUMODE `nvfp4_gemv` challenge workspace | ch06, ch10 |
            | `labs/nvfp4_gemm/` | GPUMODE `nvfp4_gemm` challenge workspace | ch06, ch09, ch10 |
            | `labs/async_input_pipeline/` | Async CPU->GPU input overlap | ch02, ch05, ch11 |
            | `labs/block_scaling/` | Blackwell hardware-supported block scaling with direct CUTLASS vs PyTorch microbenchmarks | ch06, ch09 |
            | `labs/blackwell_gemm_optimizations/` | Blackwell grouped/MoE GEMM optimization journey | ch10, ch15 |
            | `labs/blackwell_matmul/` | Matmul suite focused on Blackwell | ch06, ch09, ch10 |
            | `labs/cudnn_sdpa_bench/` | cuDNN SDPA benchmarking | ch10, ch18 |
            | `labs/custom_vs_cublas/` | Custom kernel vs cuBLAS parity | ch06, ch09 |
            | `labs/cache_aware_disagg_inference/` | Cache-aware disaggregated inference scheduling lab | ch17, ch19 |
            | `labs/cutlass_profiler_kernel_selector/` | CUTLASS profiler-based kernel selection | ch06, ch09 |
            | `labs/decode_optimization/` | Decoder hot-path optimization | ch18, ch19 |
            | `labs/dynamic_router/` | Dynamic prefill/decode routing | ch17, ch19 |
            | `labs/flashattention4/` | FlashAttention-4 pipeline co-design | ch10, ch18 |
            | `labs/flashattention_gluon/` | FlashAttention experimentation | ch18 |
            | `labs/flashinfer_attention/` | FlashInfer block-sparse attention lab | ch16 |
            | `labs/flexattention/` | FlexAttention harness and sweeps | ch18 |
            | `labs/fullstack_cluster/` | Full-stack cluster + DSMEM workflows | ch10 |
            | `labs/kv_cache_compression/` | KV-cache compression/quantization | ch18, ch19 |
            | `labs/kv_optimization/` | KV-cache performance optimization | ch15, ch18, ch19 |
            | `labs/memory_bandwidth_patterns/` | Measured bandwidth patterns: coalesced vs strided access, vectorized copy, shared-memory transpose, and cp.async checkpoints | ch02, ch10, ch11 |
            | `labs/moe_cuda/` | CUDA MoE decode toolkit | ch06, ch10, ch15 |
            | `labs/moe_decode_blackwell_matrix/` | Blackwell MoE decode scenario matrix for routing locality, persistent scheduling, and graph-launch tradeoffs | ch10, ch15, ch18, ch19 |
            | `labs/moe_optimization_journey/` | MoE optimization narrative | ch15, ch19 |
            | `labs/moe_parallelism/` | MoE parallelism planning | ch04, ch15 |
            | `labs/nccl_nixl_nvshmem/` | Communication-stack tradeoffs with an honest single-GPU analogue and NCCL/NIXL/NVSHMEM stack probing | ch04, ch17, ch19 |
            | `labs/nanochat_fullstack/` | End-to-end inference stack (NanoChat) | ch16 |
            | `labs/occupancy_tuning/` | Triton occupancy/schedule sweeps | ch08, ch14 |
            | `labs/parameterized_cuda_graphs/` | PyTorch-first parameterized CUDA Graph replay with stable buffers and executable memcpy-node updates | ch10, ch11 |
            | `labs/persistent_decode/` | Persistent decode + TMA prefill | ch10, ch11 |
            | `labs/python_concurrency/` | Python concurrency control-plane playbook (`asyncio`, retries, idempotency, hybrid pipelines) | ch03, ch11, ch16 |
            | `labs/real_world_models/` | Real-world model optimization playbook | ch20 |
            | `labs/speculative_decode/` | Speculative decoding | ch15, ch18 |
            | `labs/training_hotpath/` | Training hot-path supporting examples for reduction fusion and padding-aware projections | ch12, ch14 |
            | `labs/trtllm_phi_3_5_moe/` | TensorRT-LLM Phi-3.5-MoE comparison | ch16, ch18 |
            | `labs/train_distributed/` | Distributed training workflows | ch03, ch04 |
            | `labs/uma_memory/` | UMA / unified memory diagnostics | ch02, ch07 |
            """
        ),
    ],
    notes=[
        "Labs now intentionally support both benchmark-pair docs and honest workflow/component docs. The distinction is part of the quality bar, not an exception to it.",
    ],
)

ENTRIES["ch01"] = chapter_entry(
    slug="ch01",
    title="Chapter 1 - Performance Fundamentals",
    summary=dedent(
        """\
        Establishes the baseline benchmarking discipline with a simple training-loop goodput benchmark and a small CUDA GEMM case study. The goal is to ground later optimizations in repeatable measurement, equivalent workloads, and verifiable outputs. The repo chapter intentionally blends the book's introductory methodology material with an early hands-on kernel case study so the harness contract is concrete from the start."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 1 sets the measurement contract for the rest of the repo. The useful question here is not "can I make something faster?" but "can I show a repeatable before/after delta without changing the workload or hiding correctness problems?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager FP32 or minimally optimized Python training loops
                - one-launch-per-work-item CUDA examples
                - benchmark setups that make launch overhead and framework overhead visible"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - FP16 and fused microbatch execution for the training loop
                - separate precision-only and fusion-only variants so the training-loop story is decomposable
                - batched or strided CUDA launches to amortize dispatch cost
                - memory-reduction variants where the main win is footprint, not raw speed"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `gemm` | `0.364 ms` | `0.012 ms` | `29.51x` | strided batched GEMM removes launch overhead |
                | `performance` | `68.836 ms` | `14.286 ms` | `4.82x` | FP16 + fused microbatches raise goodput |
                | `nvfp4_mlp` | `1.130 ms` | `1.167 ms` | `0.97x` | near-flat latency, but `37.9%` lower memory use |

                This chapter intentionally includes both pure speedup examples and one memory-oriented tradeoff so later chapters do not overfit on "speedup only" thinking."""
            ),
        ),
        MarkdownSection(
            "Training-Loop Variants",
            dedent(
                """\
                The Chapter 1 training loop is intentionally split into three related targets:

                | Target | Isolated change | Intended lesson |
                | --- | --- | --- |
                | `performance` | FP16 math + fused microbatches | the combined goodput story |
                | `performance_fp16` | FP16 math only | what tensor-core-friendly precision buys you without changing batching; uses a more compute-heavy local shape so precision is visible |
                | `performance_fusion` | fused microbatches only | what launch amortization buys you without changing math precision |"""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want proof of where the win comes from instead of only a runtime delta:

                ```bash
                python -m cli.aisp bench run --targets ch01:gemm --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch01:performance --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch01:nvfp4_mlp --profile deep_dive --single-gpu
                ```

                The expected profiler story is straightforward:
                - `gemm`: fewer launches and lower dispatch overhead
                - `performance`: fewer launches plus faster tensor-core math
                - `performance_fp16`: faster GEMMs from FP16 tensor-core math with the same microbatch structure, using a benchmark-local compute-heavy shape so the precision delta is not drowned out by Python overhead
                - `performance_fusion`: fewer forward/backward launches at unchanged FP32 math
                - `nvfp4_mlp`: reduced memory footprint rather than a large wall-clock win"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch01.compare
                python -m cli.aisp bench list-targets --chapter ch01
                python -m cli.aisp bench run --targets ch01 --profile minimal
                python -m cli.aisp bench run --targets ch01:gemm --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Profile a minimal PyTorch training loop with the shared harness and reason about throughput vs latency.",
        "Separate precision wins from batching wins instead of treating all training-loop speedups as one bundle.",
        "Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.",
    ],
    contents=[
        ("`baseline_performance.py`, `optimized_performance.py`, `baseline_performance_fp16.py`, `optimized_performance_fp16.py`, `optimized_performance_fusion.py`", "Training-loop variants covering the baseline, the combined FP16+fusion path, the FP16-only pair with a benchmark-local compute-heavy shape, and the fusion-only path."),
        ("`baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu`", "CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing."),
        ("`compare.py`, `workload_config.py`, `arch_config.py`, `expectations_{hardware_key}.json`", "Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds."),
    ],
    validation=[
        "`python -m ch01.compare` reports the chapter baseline/optimized training loop pair through the shared harness with consistent workloads.",
        "Running `make && ./baseline_gemm_sm100` vs `./optimized_gemm_batched_sm100` shows a substantial drop in launch count and total runtime.",
    ],
    notes=[
        "`requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.",
        "`Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.",
    ],
)

ENTRIES["ch02"] = chapter_entry(
    slug="ch02",
    title="Chapter 2 - GPU Hardware Architecture",
    summary=dedent(
        """\
        Provides architecture awareness tooling for Blackwell-era systems-query SM and memory specs, validate NVLink throughput, and experiment with CPU-GPU coherency so optimizations stay grounded in measured hardware limits."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 2 is the "know the machine first" chapter. The point is not to collect pretty hardware facts; it is to tie optimization decisions to measured fabric, memory, and coherency behavior on the actual target system."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - generic transfer paths that do not exploit topology or coherency
                - untuned cuBLAS defaults
                - hardware assumptions based on specs instead of measured bandwidth/latency"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - topology-aware transfer and coherency choices
                - tuned cuBLAS invocation parameters
                - system bring-up driven by measured bandwidth ceilings rather than marketing numbers"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `grace_coherent_memory` | `22468.299 ms` | `970.890 ms` | `23.14x` | coherent-memory placement stops fighting the platform |
                | `memory_transfer` | `18.901 ms` | `3.637 ms` | `5.20x` | optimized transfer path fits the actual link behavior |
                | `cublas` | `0.590 ms` | `0.114 ms` | `5.17x` | tuned cuBLAS settings match the hardware better |

                `grace_coherent_memory` is only valid on Grace-Blackwell coherent-memory hosts. On PCIe-only Blackwell or other non-Grace systems, the target fails fast with `SKIPPED:` instead of emitting fallback transfer numbers under the wrong benchmark name.

                This chapter is the hardware sanity anchor for later claims: if these numbers drift, everything that depends on them deserves scrutiny."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                These are mostly hardware-path benchmarks, so the main evidence is topology, transfer, and kernel traces rather than high-level model metrics:

                ```bash
                python -m cli.aisp bench run --targets ch02:cublas --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch02:memory_transfer --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch02:grace_coherent_memory --profile deep_dive --single-gpu
                ```

                The expected story is:
                - `cublas`: better math-mode and launch configuration behavior
                - `memory_transfer`: less time lost to the wrong host/device path
                - `grace_coherent_memory`: on GB200/GB300 the placement choice dominates runtime; on non-Grace hosts the benchmark should fail fast with `SKIPPED:`"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch02.compare
                python -m ch02.hardware_info
                python -m cli.aisp bench list-targets --chapter ch02
                python -m cli.aisp bench run --targets ch02 --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Query and log GPU, CPU, and fabric capabilities before running performance studies.",
        "Measure NVLink, PCIe, and memory-bandwidth ceilings using purpose-built microbenchmarks.",
        "Validate Grace-Blackwell coherency paths to know when zero-copy buffers help or hurt.",
        "Contrast baseline vs optimized cuBLAS invocations to highlight architecture-specific tuning levers.",
    ],
    contents=[
        ("`hardware_info.py`, `cpu_gpu_topology_aware.py`", "System scanners that record GPU capabilities, NUMA layout, NVLink/NVSwitch connectivity, and affinity hints."),
        ("`nvlink_c2c_bandwidth_benchmark.py`, `baseline_memory_transfer.py`, `optimized_memory_transfer.py`, `memory_transfer_pcie_demo.cu`, `memory_transfer_nvlink_demo.cu`, `memory_transfer_zero_copy_demo.cu`, `baseline_memory_transfer_multigpu.cu`, `optimized_memory_transfer_multigpu.cu`", "Peer-to-peer and zero-copy experiments for quantifying NVLink, PCIe, and coherent memory performance."),
        ("`cpu_gpu_grace_blackwell_coherency.cu`, `cpu_gpu_grace_blackwell_coherency_sm121`", "Grace-Blackwell cache-coherent samples that compare explicit transfers vs shared mappings."),
        ("`baseline_cublas.py`, `optimized_cublas.py`", "cuBLAS GEMM benchmark pair that toggles TF32, tensor op math, and stream affinity to highlight architecture knobs."),
        ("`compare.py`, `Makefile`, `expectations_{hardware_key}.json`", "Harness driver, CUDA build rules, and expectation file for automated pass/fail checks."),
    ],
    validation=[
        "`python -m ch02.hardware_info` records the correct device name, SM count, and HBM size for every GPU in the system.",
        "`python -m ch02.nvlink_c2c_bandwidth_benchmark` reports the host↔device and bidirectional bandwidth table for the active topology.",
        "Running the coherency sample on GB200/GB300 shows zero-copy benefiting sub-MB transfers while large transfers favor explicit H2D copies; on non-Grace hosts the benchmark should stop immediately with a `SKIPPED:` capability diagnostic.",
    ],
    notes=[
        "Grace-only coherency tests require GB200/GB300 nodes; on PCIe-only or discrete-GPU Blackwell hosts the Python benchmark pair fails fast with an explicit unsupported-capability diagnostic.",
        "`Makefile` builds both CUDA and CPU tools so results can be compared without leaving the chapter.",
    ],
)

ENTRIES["ch03"] = chapter_entry(
    slug="ch03",
    title="Chapter 3 - System Tuning",
    summary=dedent(
        """\
        Captures the host-level changes-NUMA pinning, governor tweaks, container settings, and Kubernetes manifests-that keep GPU workloads fed before kernel-level optimization begins."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 3 is where "the GPU is slow" often turns out to be a host problem. The chapter matters when CPU affinity, container defaults, or orchestration choices are quietly capping the work that later CUDA kernels can ever see."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - NUMA-unaware or scheduler-default execution
                - untuned container and Kubernetes settings
                - host configuration that leaves throughput on the floor before kernels even matter"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - NUMA pinning and topology-aware process placement
                - container and cluster settings that stop starving the GPU
                - host-level tuning that is measurable through the same shared harness"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `pinned_prefetch_mlp` | `4.456 ms` | `1.225 ms` | `3.64x` | pinned-memory prefetch removes blocking host staging |
                | `gemm` | `0.548 ms` | `0.189 ms` | `2.90x` | comparison GEMM shows how host/runtime launch overhead caps achievable FLOP/s |
                | `double_buffered_batch_provisioning` | `1.734 ms` | `1.076 ms` | `1.61x` | double-buffered provisioning overlaps batch copies with compute |

                The magnitude is smaller than the headline CUDA chapters, but the lesson is important: host tuning changes are often prerequisite wins, not optional polish.
                `gemm` is intentionally a comparison workload for host/runtime overhead, while `rack_prep` is the more chapter-native staged-copy example for locality-aware host preparation. Structured metrics mark `gemm` with `story.comparison_pair=1` and `story.chapter_native_exemplar=0`, and structured story metadata marks it as a supplementary comparison pair with chapter-native targets like `pageable_copy` and `rack_prep`, so downstream reports can keep that distinction explicit."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                For this chapter, pair the harness runtime with host/GPU traces so the bottleneck story stays grounded:

                ```bash
                python -m cli.aisp bench run --targets ch03:pinned_prefetch_mlp --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch03:gemm --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch03:double_buffered_batch_provisioning --profile deep_dive --single-gpu
                ```

                Expected evidence:
                - `pinned_prefetch_mlp`: pinned-memory prefetch removes blocking host staging
                - `gemm`: lower host overhead around the same kernel without changing the math
                - `double_buffered_batch_provisioning`: double buffering reduces batch-provisioning stalls"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch03.compare
                python -m cli.aisp bench list-targets --chapter ch03
                python -m cli.aisp bench run --targets ch03 --profile minimal
                python -m ch03.power_tuning_tool --power-limits 300,350 --iterations 5 --warmup 1
                ```"""
            ),
        ),
    ],
    goals=[
        "Diagnose CPU and memory affinity issues that throttle GPU pipelines.",
        "Harden Docker and Kubernetes environments for sustained GPU throughput on shared clusters.",
        "Automate repeatable system tuning via shell scripts so lab machines stay consistent.",
        "Use comparison workloads like GEMM and rack-prep to quantify host/runtime overhead, locality, and launch latency.",
    ],
    contents=[
        ("`baseline_pageable_copy.py`, `optimized_pageable_copy.py`, `bind_numa_affinity.py`, `numa_topology_script.sh`", "Host-transfer and NUMA-adjacent helpers: the benchmark pair covers pageable-vs-pinned async copies, while the scripts handle CPU/GPU socket placement and topology inspection."),
        ("`baseline_rack_prep.py`, `optimized_rack_prep.py`, `grace_blackwell_topology.py`", "Topology-aware staging comparison pair: baseline uses blocking pageable staging, while optimized adds affinity planning plus pinned double-buffered copy/compute overlap."),
        ("`baseline_pinned_prefetch_mlp.py`, `optimized_pinned_prefetch_mlp.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh`", "Pinned-memory prefetch comparison pair plus host/container setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility."),
        ("`baseline_double_buffered_batch_provisioning.py`, `optimized_double_buffered_batch_provisioning.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml`", "Batch-provisioning overlap pair plus Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets."),
        ("`cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh`", "Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness."),
        ("`baseline_gemm.py`, `optimized_gemm.py`, `train.py`", "Comparison GEMM + training loops that expose host/runtime launch overhead in measurable FLOP/s without claiming a NUMA-specific kernel optimization."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`", "Harness entry, Python deps, and regression thresholds."),
    ],
    validation=[
        "`python -m ch03.mig_mps_tool --device 0` reports the active MIG/MPS state before you change host-level scheduling policy.",
        "`python -m ch03.power_tuning_tool --power-limits 300,350 --iterations 5 --warmup 1` produces a short perf-per-watt sweep with the harness clock-lock path.",
        "`python -m ch03.compare` keeps the chapter baseline/optimized tuning pairs runnable through the shared harness.",
    ],
    notes=[
        "`cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.",
        "Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.",
    ],
)

ENTRIES["ch04"] = chapter_entry(
    slug="ch04",
    title="Chapter 4 - Distributed Communication & Multi-GPU Distribution",
    summary=dedent(
        """\
        Demonstrates how to scale training and inference across multiple Blackwell GPUs with NVLink/NVSwitch fabric awareness, NCCL tuning, NVSHMEM collectives, and symmetric memory patterns. The current repo chapter emphasizes NCCL/NVSHMEM-style communication overlap; the broader Magnum IO + NIXL connector story from the manuscript is only partially represented here."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 4 is where multi-GPU claims have to survive contact with real communication cost. The useful question is not "can this scale?" but "which overlap, fusion, and topology choices actually move the latency or throughput needle under the shared harness?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - naive or minimally overlapped communication
                - CPU-visible coordination where device-driven orchestration would be cheaper
                - topology-agnostic execution that pays the full cost of bad placement"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - explicit overlap between compute and communication
                - fusion and pre-staging to reduce collective overhead
                - topology-aware or NVSHMEM/symmetric-memory variants where the fabric is the bottleneck"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `gradient_fusion` | `3.791 ms` | `0.055 ms` | `68.83x` | fused gradient reduction path |
                | `dataparallel` | `7.601 ms` | `0.968 ms` | `7.86x` | direct GPU execution over DataParallel overhead |
                | `grace_blackwell_locality` | `5.082 ms` | `0.317 ms` | `16.01x` | locality-aware placement |
                | `bandwidth_benchmark_suite` | `4.704 ms` | `2.685 ms` | `1.75x` | cleaner communication path |

                The chapter has both huge "remove obvious framework overhead" wins and smaller communication-optimization wins. Those should not be read as one uniform scaling story."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want actual overlap evidence rather than only before/after runtime:

                ```bash
                python -m cli.aisp bench run --targets ch04:gradient_fusion --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch04:dataparallel --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch04:grace_blackwell_locality --profile deep_dive --single-gpu
                ```

                The expected profiler story differs by target:
                - `gradient_fusion`: fewer communication phases and less launch fragmentation
                - `dataparallel`: elimination of framework-side fan-out/fan-in overhead
                - `grace_blackwell_locality`: better data placement and lower locality penalties"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch04.compare
                python -m cli.aisp bench list-targets --chapter ch04
                python -m cli.aisp bench run --targets ch04 --profile minimal
                python -m cli.aisp bench run --targets ch04:gradient_fusion --profile deep_dive --single-gpu
                python -m cli.aisp tools ch04-nixl-tier-handoff -- --mode probe --json
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark data-parallel and tensor-parallel training loops with and without overlap.",
        "Quantify NVLink bandwidth and topology effects when mixing local and disaggregated GPUs.",
        "Experiment with NVSHMEM pipelines to reduce host involvement in GPU synchronization.",
        "Adopt symmetric memory pools to simplify KV-cache replication and optimizer state sharding.",
    ],
    contents=[
        ("`baseline_dataparallel.py`, `optimized_dataparallel.py`", "Single-GPU DataParallel anti-pattern vs direct GPU execution."),
        ("`baseline_dataparallel_multigpu.py`, `optimized_dataparallel_multigpu.py`", "Multi-GPU DataParallel vs manual gradient reduction with pre-staged shards."),
        ("`baseline_no_overlap.py`, `optimized_no_overlap.py`", "Strict distributed DDP overlap benchmarks; they now require `torchrun` plus `>=2` GPUs and fail fast with `SKIPPED:` when launched on unsupported hosts."),
        ("`baseline_pcie_staging.py`, `optimized_pcie_staging.py`, `baseline_nvlink_topology_aware.py`, `optimized_nvlink_topology_aware.py`, `baseline_nvlink_multigpu.py`, `optimized_nvlink_multigpu.py`, `baseline_nvlink_topology_aware_multigpu.py`, `optimized_nvlink_topology_aware_multigpu.py`", "PCIe host-staging comparison pair plus NVLink exercises for validating peer bandwidth and topology effects (single- and multi-GPU)."),
        ("`baseline_nixl_tier_handoff.py`, `optimized_nixl_tier_handoff.py`, `nixl_tier_handoff_tool.py`", "Chapter-local NIXL-style memory-tier handoff analogue built on the same selected-block transport workload used by the communication-stack lab, with both benchmark and probe entrypoints."),
        ("`baseline_continuous_batching.py`, `optimized_continuous_batching.py`, `baseline_disaggregated.py`, `optimized_disaggregated.py`, `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py`, `baseline_disaggregated_multigpu.py`, `optimized_disaggregated_multigpu.py`", "Continuous batching + disaggregated inference demos that showcase pooling and remote KV reuse."),
        ("`baseline_gradient_compression_fp16.py`, `optimized_gradient_compression_fp16.py`, `baseline_gradient_compression_int8.py`, `optimized_gradient_compression_int8.py`, `baseline_gradient_compression_fp16_multigpu.py`, `optimized_gradient_compression_fp16_multigpu.py`, `baseline_gradient_compression_int8_multigpu.py`, `optimized_gradient_compression_int8_multigpu.py`", "Gradient compression all-reduce benchmarks comparing small-bucket vs full-buffer compression (single GPU and multi-GPU FP16/INT8 paths)."),
        ("`baseline_gradient_compression_fp16_comm_only.py`, `optimized_gradient_compression_fp16_comm_only.py`, `baseline_gradient_compression_int8_comm_only.py`, `optimized_gradient_compression_int8_comm_only.py`, `baseline_gradient_compression_fp16_comm_only_multigpu.py`, `optimized_gradient_compression_fp16_comm_only_multigpu.py`, `baseline_gradient_compression_int8_comm_only_multigpu.py`, `optimized_gradient_compression_int8_comm_only_multigpu.py`", "Communication-only gradient compression benchmarks with pre-quantized buffers (single GPU and multi-GPU FP16/INT8 paths)."),
        ("`baseline_pipeline_parallel.py`, `optimized_pipeline_parallel_1f1b.py`, `baseline_tensor_parallel.py`, `optimized_tensor_parallel_async.py`, `baseline_torchcomms.py`, `optimized_torchcomms.py`, `baseline_pipeline_parallel_multigpu.py`, `optimized_pipeline_parallel_multigpu_1f1b.py`, `baseline_tensor_parallel_multigpu.py`, `optimized_tensor_parallel_multigpu.py`, `baseline_tensor_parallel_allgather_multigpu.py`, `optimized_tensor_parallel_allgather_multigpu.py`, `baseline_torchcomms_multigpu.py`, `optimized_torchcomms_multigpu.py`", "Pipeline/tensor-parallel and torchcomms overlap studies (single- and multi-GPU)."),
        ("`baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py`, `baseline_nvshmem_training_example_multigpu.py`, `optimized_nvshmem_training_example_multigpu.py`", "NVSHMEM/symmetric-memory samples that require real multi-GPU launch plus symmetric-memory support; unsupported hosts fail fast with `SKIPPED:`."),
        ("`baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py`, `baseline_symmetric_memory_multigpu.py`, `optimized_symmetric_memory_multigpu.py`, `baseline_symmetric_memory_perf_multigpu.py`, `optimized_symmetric_memory_perf_multigpu.py`", "Symmetric memory utilities and perf probes for KV cache and optimizer shards."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py`", "Harness driver plus standalone NCCL/NVLink sweepers for topology bring-up."),
    ],
    validation=[
        "`python compare.py --examples dataparallel_multigpu` shows the optimized pair overlapping compute and communication with lower latency.",
        "`python -m cli.aisp bench run --targets ch04:nixl_tier_handoff --profile minimal --single-gpu` verifies the Chapter 4 NIXL-style tier handoff analogue on the same selected-block workload.",
        "`python -m cli.aisp bench run --targets ch04:bandwidth_benchmark_suite_multigpu --profile minimal` surfaces >=250 GB/s links on connected GPU pairs and highlights any slow hops on a host with >=2 visible GPUs.",
        "NVSHMEM samples now run only on real multi-GPU symmetric-memory hosts; unsupported environments fail fast with `SKIPPED:` instead of publishing fallback timings.",
    ],
    notes=[
        "`symmetric_memory_*` helpers hold user-space allocators for pooling KV-cache lines across GPUs without NVSwitch penalties.",
        "Use `nccl_blackwell_config.py` to seed NCCL env vars (min NRings, IB mapping) before launching multi-node tests.",
    ],
)

ENTRIES["ch05"] = chapter_entry(
    slug="ch05",
    title="Chapter 5 - Storage and IO Optimization",
    summary=dedent(
        """\
        Focuses on feeding GPUs efficiently: tune DataLoader workers, vectorize preprocessing, overlap IO with compute, and adopt GPUDirect Storage when NVMe traffic becomes the bottleneck."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 5 exists because GPUs do not care how elegant a kernel is if the input path is late. The useful question is which storage and preprocessing changes actually turn an IO-bound workload into a compute-bound one."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - CPU-heavy preprocessing and unvectorized parsing
                - storage paths that serialize work on the host
                - dataloading behavior that leaves visible GPU idle time"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - vectorized preprocessing and overlap between IO and compute
                - tuned worker/prefetch settings
                - GPUDirect Storage or cleaner staging paths where the platform supports them"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `vectorization` | `3.861 ms` | `0.053 ms` | `72.64x` | Python-heavy preprocessing becomes vectorized |
                | `storage_cpu` | `111.652 ms` | `53.898 ms` | `2.07x` | storage path stops starving the device |

                The headline win here is often preprocessing, not raw storage hardware. The `ai` pair now stays in the chapter as an informational overlap/control demo instead of a canonical speedup claim."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs to distinguish host-side preprocessing waste from actual storage limits:

                ```bash
                python -m cli.aisp bench run --targets ch05:vectorization --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch05:storage_cpu --profile deep_dive --single-gpu
                ```

                The expected evidence is:
                - `vectorization`: dramatically less CPU time in preprocessing
                - `storage_cpu`: fewer long idle gaps between batches
                - `ai`: useful as an overlap/control trace, but no longer treated as a canonical speed target"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch05.compare
                python -m cli.aisp bench list-targets --chapter ch05
                python -m cli.aisp bench run --targets ch05 --profile minimal
                python -m ch05.gds_cufile_minimal /tmp/gds_test_file.bin 1073741824 --generate
                ```"""
            ),
        ),
    ],
    goals=[
        "Detect IO stalls via harness metrics and restructure pipelines to keep GPUs busy.",
        "Tune PyTorch DataLoader knobs (workers, prefetch, pinned memory) for large-batch training.",
        "Evaluate GPUDirect Storage paths vs traditional CPU-mediated reads.",
        "Benchmark remote storage and distributed data reading strategies.",
    ],
    contents=[
        ("`baseline_storage_cpu.py`, `optimized_storage_cpu.py`", "Single-node dataloader comparison covering worker count, pinned memory, and caching strategies."),
        ("`baseline_vectorization.py`, `optimized_vectorization.py`", "Vectorized parsing and memory-map examples that remove Python loops from preprocessing."),
        ("`baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py`", "LLM-style token pipelines showcasing overlapping compute with streaming reads and prefetch. `ai` is kept as an informational overlap/control demo."),
        ("`baseline_host_staged_reduction.py`, `optimized_host_staged_reduction.py`", "Single-GPU host-staged reduction vs on-device reduction."),
        ("`baseline_distributed_multigpu.py`, `optimized_distributed_multigpu.py`", "Actual multi-GPU reduction baseline (CPU staging) vs GPU-side reduce_add."),
        ("`gds_cufile_minimal.py`, `gpudirect_storage_example.py`", "GPUDirect Storage utilities that verify real cuFile/GDS capability; unsupported hosts fail fast with `SKIPPED:` instead of publishing host-staged fallback throughput."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`", "Harness entrypoint plus expectation baselines for spotting regressions."),
    ],
    validation=[
        "`python baseline_storage_cpu.py --inspect` exposes CPU wait time > GPU time; `optimized_storage_cpu.py` reverses the ratio with >=80% GPU utilization.",
        "`python -m ch05.gds_cufile_minimal /tmp/gds_test_file.bin 1073741824 --generate` now acts as a strict capability probe: it either confirms usable cuFile/GDS support or exits with `SKIPPED:`.",
        "`python -m ch05.compare` remains useful for inspecting the `ai` overlap/control demo, but canonical chapter claims should come from `vectorization` and `storage_cpu`.",
    ],
    notes=[
        "GPUDirect scripts no longer publish host-mediated fallback numbers under the cuFile/GDS names; unsupported hosts receive explicit `SKIPPED:` diagnostics instead.",
        "`requirements.txt` captures the limited extra deps (like `lmdb`) needed for the dataset shims.",
    ],
)

ENTRIES["ch06"] = chapter_entry(
    slug="ch06",
    title="Chapter 6 - CUDA Programming Fundamentals",
    summary=dedent(
        """\
        Moves from Python into CUDA C++: write first kernels, reason about occupancy, control memory layouts, and experiment with ILP, launch bounds, and unified memory on Blackwell devices. The manuscript introduces roofline thinking in this chapter; the repo's deeper roofline tooling lives in later utilities while Chapter 6 keeps the runnable benchmark surface focused on kernel fundamentals."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 6 is where kernel mechanics stop being theoretical. The real question is which low-level changes register pressure, vector width, launch bounds, ILP, and memory layout actually show up as measured improvement under the harness."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - simple kernels with minimal attention to occupancy or memory layout
                - scalar or poorly amortized execution paths
                - examples that surface launch and memory inefficiency clearly"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - vectorized and parallelized kernels
                - ILP- and launch-bound-aware variants
                - autotuned or occupancy-tuned schedules where the hardware payoff is visible"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `add` | `172.202 ms` | `0.044 ms` | `3881.04x` | naive add path replaced with a true CUDA implementation |
                | `attention_ilp` | `140.603 ms` | `0.529 ms` | `265.82x` | the attention-score inner loop moves from one dependent chain per thread to four independent chains |
                | `autotuning` | `63.881 ms` | `16.310 ms` | `3.92x` | schedule selection finds a materially better kernel config |

                This chapter has the biggest synthetic-looking wins in the repo because many baselines are intentionally pedagogical. They are still useful, but they should be read as controlled teaching deltas, not production uplift guarantees."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                This is a profiler-heavy chapter by design. Use deep-dive runs when you want to connect the wall-clock delta to occupancy, memory throughput, and launch behavior:

                ```bash
                python -m cli.aisp bench run --targets ch06:add --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch06:attention_ilp --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch06:autotuning --profile deep_dive --single-gpu
                ```

                Expected profiler story:
                - `add`: removal of pure-framework overhead and better GPU utilization
                - `attention_ilp`: higher effective work per thread inside an attention-shaped score microbenchmark, not a different attention algorithm
                - `autotuning`: better schedule choice rather than different math"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch06.compare
                python -m cli.aisp bench list-targets --chapter ch06
                python -m cli.aisp bench run --targets ch06 --profile minimal
                python -m cli.aisp bench run --targets ch06:attention_ilp --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Write and launch custom kernels that mirror the harness workloads.",
        "Understand how occupancy, launch bounds, and register pressure interact.",
        "Use ILP and vectorized memory ops to increase throughput per thread.",
        "Validate unified memory and allocator tuning on Blackwell GPUs.",
    ],
    contents=[
        ("`my_first_kernel.cu`, `simple_kernel.cu`, `baseline_add_cuda.cu`, `optimized_add_cuda_parallel.cu`, `baseline_add.py`, `optimized_add.py`, `baseline_add_cuda.py`, `optimized_add_cuda_parallel.py`", "Hello-world kernels plus Python wrappers for the canonical `add` / `add_cuda` targets and the underlying CUDA launch-path verification."),
        ("`baseline_attention_ilp.py`, `optimized_attention_ilp.py`, `baseline_elementwise_ilp.py`, `optimized_elementwise_ilp.py`, `ilp_low_occupancy_vec4_demo.cu`, `ilp_extreme_low_occupancy_vec4_demo.cu`", "Instruction-level parallelism studies that keep the math fixed while changing independent chains per thread, register pressure, and vector width."),
        ("`baseline_bank_conflicts.cu`, `optimized_bank_conflicts.cu`, `baseline_launch_bounds*.{py,cu}`, `optimized_launch_bounds*.{py,cu}`", "Bank conflict and launch-bound exercises to highlight shared memory layouts and CTA sizing."),
        ("`baseline_autotuning.py`, `optimized_autotuning.py`, `memory_pool_tuning.cu`, `stream_ordered_allocator/`", "Autotuning harness plus allocator experiments for controlling fragmentation and stream ordering."),
        ("`unified_memory.cu`, `occupancy_api.cu`, `baseline_quantization_ilp.py`, `optimized_quantization_ilp.py`", "Unified memory demo, occupancy calculator sample, and quantization-focused ILP workloads."),
        ("`compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `workload_config.py`", "Harness entry, build scripts, expectation baselines, and workload settings."),
    ],
    validation=[
        "`nvcc -o baseline_add_cuda_sm121 baseline_add_cuda.cu` vs the optimized vectorized version shows a clear bandwidth delta when inspected with Nsight Compute.",
        "`python optimized_autotuning.py --search` converges to the same schedule as the curated preset and logs the score table under `artifacts/`.",
        "`python -m ch06.compare` confirms the chapter baseline/optimized pairs stay runnable through the harness after ILP and launch-bound refactors.",
    ],
    notes=[
        "`arch_config.py` forces SM-specific compile flags (e.g., disabling pipelines on unsupported GPUs) so targets fail gracefully on older hardware.",
        "The canonical public add targets are `add` and `add_cuda`; the older add-tensors wrapper names were removed instead of kept as legacy aliases.",
        "`launch_bounds` and `launch_bounds_cuda` are small-effect teaching cases with local expectation-backed contracts, not headline Chapter 6 speed claims.",
        "`attention_ilp` is an attention-score preprocessing microbenchmark. It is intentionally not a fused SDPA or multi-stream overlap example.",
        "CUDA extensions in `cuda_extensions/` can be imported directly into notebooks for interactive prototyping.",
    ],
)

ENTRIES["ch07"] = chapter_entry(
    slug="ch07",
    title="Chapter 7 - Memory Access Patterns",
    summary=dedent(
        """\
        Teaches how memory layout drives performance: coalesced copies, tiled matmuls, async prefetch, TMA transfers, and shared-memory staging for lookup-heavy workloads."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 7 is where memory layout turns from a CUDA lecture into a measurable cost model. The useful question is not "is coalescing good?" but "which access-pattern changes actually move the runtime enough to justify changing the kernel or data layout?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - scalar or poorly staged memory movement
                - little reuse of shared memory or async transfer mechanisms
                - straightforward for correctness, but wasteful once bandwidth dominates"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - coalesced/vectorized copy paths
                - shared-memory tiling and TMA-backed staging where it helps
                - measured through the shared harness so the memory-layout wins are directly comparable to other chapter benchmarks"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `tma_bulk_tensor_2d` | `0.029 ms` | `0.008 ms` | `3.44x` | real tensor-map TMA bulk copy instead of manual 2D staging |
                | `lookup` | `0.397 ms` | `0.009 ms` | `45.41x` | locality-aware lookup path |
                | `matmul` | `1.165 ms` | `0.367 ms` | `3.18x` | shared-memory tiled matmul instead of the naive layout |

                This chapter has some intentionally dramatic wins because memory access mistakes are expensive. `tma_copy` now means a strict tensor-map/TMA-capable run only, and unsupported hosts fail fast with `SKIPPED:` instead of publishing an async-pipeline fallback under the TMA name."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want to see whether the win came from less memory traffic, better staging, or fewer expensive accesses:

                ```bash
                python -m cli.aisp bench run --targets ch07:tma_bulk_tensor_2d --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch07:lookup --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch07:matmul --profile deep_dive --single-gpu
                ```

                These targets answer different chapter-level questions:
                - `tma_bulk_tensor_2d`: descriptor-backed TMA vs manual 2D staging
                - `tma_copy`: scalar baseline vs strict descriptor-backed tensor-map/TMA path
                - `lookup`: cache/locality sensitivity
                - `matmul`: memory-layout and tile-reuse payoff"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch07.compare
                python -m cli.aisp bench list-targets --chapter ch07
                python -m cli.aisp bench run --targets ch07 --profile minimal
                python -m cli.aisp bench run --targets ch07:tma_bulk_tensor_2d --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure the gap between scalar, coalesced, and vectorized memory moves.",
        "Use shared-memory tiling, async copy, and tensor maps where they actually help.",
        "Analyze lookup-heavy workloads and mitigate cache-thrashing access patterns.",
        "Quantify transpose and gather/scatter penalties to justify layout changes.",
    ],
    contents=[
        ("`baseline_copy_scalar.cu`, `baseline_copy_uncoalesced.cu`, `baseline_copy_uncoalesced.py`, `optimized_copy_uncoalesced_coalesced.cu`, `optimized_copy_scalar_vectorized.cu`, `optimized_copy_scalar_vectorized_sm121`", "Copy kernels highlighting coalescing, vector width, and warp-level efficiency."),
        ("`baseline_hbm_copy.cu`, `baseline_hbm_peak.cu`, `optimized_hbm_copy.cu`, `optimized_hbm_peak.cu`, `baseline_hbm_copy.py`, `optimized_hbm_copy.py`", "HBM peak-bandwidth probes with CUDA and Python harnesses."),
        ("`baseline_async_prefetch.cu`, `optimized_async_prefetch.cu`, `baseline_tma_copy.cu`, `optimized_tma_copy.cu`, `baseline_tma_copy.py`, `optimized_tma_copy.py`, `async_prefetch_2d_demo.cu`, `baseline_tma_bulk_tensor_2d.{py,cu}`, `optimized_tma_bulk_tensor_2d.{py,cu}`", "Async copy demos plus the separate descriptor-backed TMA benchmark used for the chapter's canonical tensor-map evidence."),
        ("`baseline_matmul.cu`, `baseline_matmul.py`, `optimized_matmul_tiled.py`, `optimized_matmul_tiled.cu`", "Matmul implementations to contrast naive global-memory access with shared-memory tiling and warp-level reuse."),
        ("`baseline_lookup.cu`, `baseline_lookup.py`, `optimized_lookup.cu`, `lookup_pytorch.py`", "Cache-sensitive lookup workloads demonstrating how to reorganize tables for better locality."),
        ("`baseline_transpose.cu`, `baseline_transpose.py`, `optimized_copy_scalar_vectorized.cu`, `optimized_transpose_padded.py`", "Transpose and gather/scatter experiments that show how to minimize bank conflicts."),
        ("`compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `memory_access_pytorch.py`", "Harness entry, build recipes, expectation thresholds, and PyTorch validation scripts."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets ch07:hbm_copy --profile minimal` reports the baseline/optimized bandwidth gap, proving vectorization plus async copies work.",
        "`python -m ch07.compare` runs the full baseline/optimized chapter sweep through the shared harness.",
        "Nsight Compute captures of `optimized_matmul_tiled.cu` hit >80% shared-memory bandwidth utilization with minimal bank conflicts.",
    ],
    notes=[
        "Toggle `TORCH_COMPILE_MODE` when using the Python matmul wrappers to verify fusion benefits alongside the raw CUDA kernels.",
        "HBM tooling reads real peak numbers from `benchmark_peak_results_*.json` when present, providing realistic reference ceilings.",
    ],
)

ENTRIES["ch08"] = chapter_entry(
    slug="ch08",
    title="Chapter 8 - Occupancy, Warp Efficiency & ILP",
    summary=dedent(
        """\
        Concentrates on the Chapter 8 core loop from the book: tune occupancy, reduce warp divergence, and expose instruction-level parallelism until the profiler shows fewer stalls and more useful issue bandwidth."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 8 is where profiler symptoms need to map cleanly to fixes. The useful question is not "is occupancy important?" but "which changes reduce execution-dependency stalls, improve warp efficiency, and keep enough resident work on the SM without blowing up register pressure?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - conservative launch geometry or branch-heavy kernels
                - more dependency chains per thread and lower warp execution efficiency
                - easier to reason about, but often leaves the SM underfilled or the warp schedulers idle"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - occupancy-aware launch and block-shape tuning
                - predication and loop-unrolling changes that expose more useful work per warp
                - measured through the same harness contract as the rest of the repo, so the gains are not one-off microbench stories"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `ch08/expectations_b200.json`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `threshold` | `2.324 ms` | `0.228 ms` | `10.19x` | predication removes the branch-heavy slow path and raises warp efficiency |
                | `loop_unrolling` | `1.591 ms` | `0.382 ms` | `4.17x` | more independent work per thread reduces execution-dependency stalls |
                | `ai_optimization` | `0.646 ms` | `0.241 ms` | `2.68x` | occupancy-aware scheduling keeps more useful work resident |

                These are the chapter-native exemplars. The repo also keeps a few real bridge comparison pairs here, such as `thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp`, but those are explicitly marked in structured metrics as comparison pairs so dashboards do not blur them with the book's core Chapter 8 story. `tcgen05_custom_vs_cublas` remains an explicit custom-versus-library comparison target, and it now stays runnable as a supplementary comparison benchmark with a local contract rather than being skipped as informational."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want to see whether the improvement came from better warp efficiency, more ILP, or a better occupancy/resource balance:

                ```bash
                python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch08:loop_unrolling --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch08:ai_optimization --profile deep_dive --single-gpu
                ```

                Those targets give you three useful slices:
                - `threshold`: branch elimination and warp execution efficiency
                - `loop_unrolling`: per-thread ILP and execution-dependency stalls
                - `ai_optimization`: occupancy/resource tradeoffs in a more compute-heavy kernel"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch08.compare
                python -m cli.aisp bench list-targets --chapter ch08
                python -m cli.aisp bench run --targets ch08 --profile minimal
                python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Tune occupancy explicitly and observe how register counts limit resident CTAs.",
        "Minimize warp divergence with predication and uniform control flow.",
        "Use loop unrolling and instruction scheduling to increase throughput per thread.",
        "Reprofile after each change so occupancy, warp efficiency, and ILP improvements are visible in the same harness.",
    ],
    contents=[
        ("`baseline_threshold.py`, `optimized_threshold.py`, `threshold_kernels.cu`, `threshold_benchmark_base.py`", "Chapter-native warp-divergence pair: branchy thresholding versus predicated thresholding on the same workload shape."),
        ("`baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `occupancy_tuning_tool.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu`", "Occupancy studies that tune CTA shapes, register caps, and API-computed limits (plus a sweep tool for quick preset exploration)."),
        ("`baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu`", "Loop-unrolling case studies that expose more independent work per thread while tracking register pressure."),
        ("`baseline_ai_optimization.py`, `optimized_ai_optimization.py`, `ai_optimization_kernels.cu`, `independent_ops.cu`", "AI-kernel scheduling samples that stage independent ops to highlight occupancy and issue-efficiency tradeoffs."),
        ("`baseline_thresholdtma.py`, `optimized_thresholdtma.py`, `threshold_tma_benchmark_base.py`", "Bridge comparison pair into the later TMA chapters: same threshold workload shape, but a TMA-backed path marked as a comparison pair in structured metrics."),
        ("`baseline_tiling.py`, `optimized_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu`, `tiling_extension_tcgen05.py`", "Bridge comparison pairs into Chapter 9: arithmetic-intensity and tensor-core tiling workloads kept as real baseline/optimized pairs but marked non-native for Chapter 8. `optimized_tiling.py` uses the strict `matmul_tiled_fast` path so runtime issues fail fast instead of silently falling back."),
        ("`baseline_tcgen05_custom_vs_cublas.py`, `optimized_tcgen05_custom_vs_cublas.py`, `tcgen05_custom_vs_cublas_benchmark_base.py`", "Supplementary custom-tcgen05-versus-cuBLAS bridge comparison benchmark that points ahead to Chapter 9 tensor-core scheduling without acting as a canonical Chapter 8 speed claim."),
        ("`baseline_nvfp4_mlp.py`, `optimized_nvfp4_mlp.py`", "Precision bridge comparison pair: BF16 versus NVFP4 MLP path kept here as a real pair, but explicitly marked as a Chapter 9-style comparison."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`", "Harness entry, dependencies, and regression thresholds."),
    ],
    validation=[
        "Nsight Compute traces for `optimized_threshold.py` should show higher warp execution efficiency than `baseline_threshold.py`.",
        "`python -m cli.aisp tools occupancy-tuning` prints preset timings + speedups for the occupancy tuning microbenchmark.",
        "`python -m cli.aisp bench run --targets ch08:thresholdtma --profile minimal` exercises the Blackwell-only bridge comparison on the same threshold shape used by the chapter-native threshold pair.",
    ],
    notes=[
        "`arch_config.py` exposes toggles for enabling/disabling tcgen05 lowering per GPU so the same scripts work on SM100 and SM121.",
        "`threshold`, `loop_unrolling`, and `ai_optimization` are the chapter-native exemplars. `thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp` remain real baseline/optimized bridge comparisons and expose `story.comparison_pair=1` plus `story.chapter_native_exemplar=0` in structured metrics.",
        "`tcgen05_custom_vs_cublas` is intentionally named as a custom-versus-library comparison target so the benchmark surface matches the story it is telling.",
        "`build/` caches CUDA object files per configuration; clean via `python cleanup.py --include-build` when adjusting toolchains.",
    ],
)

ENTRIES["ch09"] = chapter_entry(
    slug="ch09",
    title="Chapter 9 - Arithmetic Intensity & Kernel Fusion",
    summary=dedent(
        """\
        Explores how to move workloads along the roofline: raise arithmetic intensity with tiling, fuse memory-bound kernels, and deploy CUTLASS/Triton/inline-PTX paths built for Blackwell tensor cores."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 9 is where roofline reasoning has to cash out in actual kernels. The useful question is not "is this compute-bound or memory-bound?" but "which arithmetic-intensity and fusion changes create a measurable gain once the same harness measures both sides?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - lower-intensity or less-fused kernels
                - more time spent moving data than doing useful math
                - easier to inspect, but often too far from the hardware roofline"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - CUTLASS/Triton/custom-kernel paths with better tiling and reuse
                - fused or higher-intensity schedules that reduce redundant memory work
                - the same benchmark contract as the rest of the repo, so the gains are not script-local illusions"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `cutlass_gemm` | `0.178 ms` | `0.045 ms` | `3.95x` | better GEMM schedule and kernel implementation |
                | `memory_bound` | `3.491 ms` | `0.205 ms` | `17.05x` | less wasted memory movement on a bandwidth-limited workload |
                | `sdpa_attention` | `0.762 ms` | `0.446 ms` | `1.71x` | attention path with improved compute/memory balance |

                The right chapter-level read is not that every CUTLASS or Triton change is dramatic. It is that arithmetic-intensity gains and fusion gains show up very differently depending on whether the workload is math-limited or traffic-limited."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want to see whether the improvement came from better tensor-core utilization, less memory traffic, or simply fewer kernels:

                ```bash
                python -m cli.aisp bench run --targets ch09:cutlass_gemm --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch09:memory_bound --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch09:sdpa_attention --profile deep_dive --single-gpu
                ```

                Those three targets give you a balanced view of the chapter:
                - `cutlass_gemm`: math-path scheduling
                - `memory_bound`: bandwidth-path improvement
                - `sdpa_attention`: mixed compute/memory behavior in a more realistic primitive"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch09.compare
                python -m cli.aisp bench list-targets --chapter ch09
                python -m cli.aisp bench run --targets ch09 --profile minimal
                python -m cli.aisp bench run --targets ch09:cutlass_gemm --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Separate compute-bound vs memory-bound behaviors and adjust kernels accordingly.",
        "Design micro-tiling schedules that balance register pressure with data reuse.",
        "Leverage CUTLASS and Triton for rapid iteration while keeping custom CUDA fallbacks.",
        "Fuse reduction-heavy kernels (e.g., norm + activation) to eliminate redundant memory trips.",
    ],
    contents=[
        ("`baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py`", "Reference kernels that isolate compute vs bandwidth ceilings and demonstrate tuning strategies."),
        ("`baseline_micro_tiling_matmul.cu`, `baseline_micro_tiling_matmul.py`, `optimized_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.py`", "Micro-tiling matmuls with explicit register blocking and cp.async prefetch."),
        ("`baseline_cublaslt_gemm.cu`, `baseline_cublaslt_gemm.py`, `optimized_cublaslt_gemm.cu`, `optimized_cublaslt_gemm.py`, `tcgen05_pipelined.cu`", "cuBLASLt-driven matmuls and tcgen05 pipeline kernels showcasing tcgen05 lowering and occupancy tuning."),
        ("`baseline_cublaslt_gemm_fp4.cu`, `baseline_cublaslt_gemm_fp4.py`, `optimized_cublaslt_gemm_fp4.cu`, `optimized_cublaslt_gemm_fp4.py`", "FP4 comparison path: naive block-scaled FP4 baseline vs native cuBLASLt NVFP4 when the driver/toolchain exposes the required heuristic."),
        ("`baseline_fused_l2norm.cu`, `baseline_fused_l2norm.py`, `optimized_fused_l2norm.cu`, `optimized_fused_l2norm.py`, `fusedL2Norm/`", "Fusion examples that merge L2 norm + scaling while staying numerically stable."),
        ("`baseline_triton.py`, `optimized_triton.py`", "Triton counterparts for quick prototyping and verifying compiler-generated PTX on Blackwell."),
        ("`baseline_tcgen05_tma_pipeline.py`, `optimized_tcgen05_tma_pipeline.py`, `two_stage_pipeline.cu`", "Producer/consumer pipelines emphasizing staged TMA loads and inline PTX hooks."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`", "Harness hooks plus regression thresholds for every example."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets ch09:compute_bound ch09:memory_bound --profile minimal` reports much higher arithmetic intensity for the compute-bound path than the memory-bound path, matching the roofline plots.",
        "`python -m cli.aisp bench run --targets ch09:cublaslt_gemm --profile minimal` confirms the optimized path improving throughput over the baseline on the same device.",
        "`python -m ch09.compare --examples fused_l2norm` confirms numerically identical outputs before and after fusion.",
    ],
    notes=[
        "`inline_ptx_example.cu` demonstrates how to wrap tcgen05 intrinsics safely with architecture guards.",
        "`requirements.txt` includes Triton nightly pinning so the kernels track PyTorch 2.10-dev features.",
        "`optimized_cublaslt_gemm_fp4` is intentionally capability-gated: if cuBLASLt cannot provide the native block-scaled NVFP4 heuristic, the benchmark reports a clean skip instead of silently falling back to a different FP4 mode.",
    ],
)

ENTRIES["ch10"] = chapter_entry(
    slug="ch10",
    title="Chapter 10 - Tensor Core Pipelines & Cluster Features",
    summary=dedent(
        """\
        Applies tensor-core friendly scheduling on Blackwell: warp specialization, TMA-powered pipelines, persistent kernels, and thread-block clusters with DSMEM and NVLink-C2C awareness."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            "Chapter 10 is where the repo stops talking about tensor-core scheduling in the abstract and starts proving which pipeline and cluster choices actually matter on Blackwell.",
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - scalar-heavy or launch-heavy kernels that leave tensor cores underfed
                - non-persistent pipelines that pay setup cost every iteration
                - cluster-disabled variants that show the cost of ignoring DSMEM / multicast hardware"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - warp-specialized and persistent kernels that keep producer/consumer work separated
                - TMA-fed pipelines that reduce staging overhead
                - cluster-enabled kernels that exploit DSMEM and multicast when the hardware supports it"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `cluster_group_single_cta` | `2.203 ms` | `0.031 ms` | `71.42x` |
                | `batch` | `10.061 ms` | `0.185 ms` | `54.44x` |

                These are good chapter-level "does the optimization concept work?" numbers, not universal hardware ceilings.
                `book-after/ch10.md` is centered on intra-kernel pipelines, warp specialization, persistent kernels, and cluster workflows, so the canonical Chapter 10 surface stays anchored on targets such as `double_buffered_pipeline`, `pipeline_3stage`, `warp_specialized_pipeline`, and `cluster_group`."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the harness target directly when you want reproducible Nsight evidence instead of ad hoc scripts:

                ```bash
                python -m cli.aisp bench run --targets ch10:cluster_group_single_cta --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch10:batch --profile deep_dive --single-gpu
                ```

                The deep-dive path gives you a concrete before/after pairing for launch count, kernel duration, and memory/cluster behavior."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch10.compare --profile none
                python -m cli.aisp bench list-targets --chapter ch10
                python -m cli.aisp bench run --targets ch10 --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Use warp specialization and cp.async/TMA to keep tensor cores saturated.",
        "Prototype persistent matmuls that amortize launch overhead across iterations.",
        "Exercise thread-block clusters with and without DSMEM to understand hardware limits.",
        "Combine PyTorch, Triton, and CUDA kernels while keeping expectations synchronized.",
    ],
    contents=[
        ("`baseline_attention.py`, `optimized_attention.py`, `baseline_flash_attention.py`, `optimized_flash_attention.py`, `analyze_scaling.py`", "Attention workloads that span eager, fused, and `torch.compile` paths for modern decoder models."),
        ("`baseline_batch.py`, `optimized_batch.py`, `baseline_matmul_tcgen05_vs_cublas.py`, `optimized_matmul_tcgen05_vs_cublas.py`", "Batch scheduling benchmarks plus a custom-tcgen05-versus-cuBLAS comparison target kept for manual tensor-core reference."),
        ("`baseline_tcgen05_warp_specialization.py`, `optimized_tcgen05_warp_specialization.py`, `tcgen05_warp_specialized.cu`", "Warp-specialized tcgen05 GEMM with dedicated producer/consumer warps."),
        ("`baseline_tcgen05_warp_specialization_cutlass.py`, `optimized_tcgen05_warp_specialization_cutlass.py`, `tcgen05_warp_specialized_cutlass.cu`, `tcgen05_warpgroup_specialized.cu`", "CUTLASS warp-specialized mainloop comparison (1-SM warp-specialized vs 2-SM warpgroup tile)."),
        ("`warpgroup_specialization_demo.py`, `tcgen05_warpgroup_specialized.cu`", "Demo of the CUTLASS warpgroup array mainloop using a 2-SM tile."),
        ("`baseline_double_buffered_pipeline.{py,cu}`, `optimized_double_buffered_pipeline.{py,cu}`, `baseline_tma_2d_pipeline.py`, `optimized_tma_2d_pipeline.py`", "Async pipeline samples mixing cp.async, TMA, and manual double buffering."),
        ("`baseline_cluster_group*.{py,cu}`, `optimized_cluster_group*.{py,cu}`, `cluster_group_common.cuh`, `cluster_group_utils.py`", "Clustered kernel suite covering DSMEM-enabled and DSMEM-free thread-block clusters."),
        ("`baseline_cluster_multicast.py`, `optimized_cluster_multicast.py`, `tma_multicast_baseline.cu`, `tma_multicast_cluster.cu`", "Cluster multicast GEMM example (baseline vs cluster multicast) wrapped as CUDA-binary harness benchmarks."),
        ("`baseline_cooperative_persistent.{py,cu}`, `optimized_cooperative_persistent.{py,cu}`, `baseline_persistent_matmul_tma.py`, `optimized_persistent_matmul_tma.py`", "Persistent kernels that keep the iteration loop on-device, contrasting synchronous staging with a two-stage shared-memory pipeline."),
        ("`baseline_warp_spec_pingpong.{py,cu}`, `optimized_warp_spec_pingpong.{py,cu}`, `baseline_flash_attn_tma_micro_pipeline.{py,cu}`, `optimized_flash_attn_tma_micro_pipeline.{py,cu}`, `baseline_warp_specialized_pipeline*.{py,cu}`, `optimized_warp_specialized_pipeline*.{py,cu}`", "Micro-pipeline and warp specialization studies, including explicit producer/compute/consumer warp roles and ping-pong staging."),
        ("`baseline_warp_specialized_cluster_pipeline.{py,cu}`, `optimized_warp_specialized_cluster_pipeline.{py,cu}`", "Thread-block-cluster warp specialization example: a synchronous DSMEM baseline versus a leader-CTA pipeline that stages tiles once per cluster."),
        ("`compare.py`, `workload_config.py`, `demo_both_examples.sh`, `profile.sh`, `requirements_cufile.txt`", "Harness entry, workload dials, demo runner, Nsight automation, and optional cuFile deps."),
    ],
    validation=[
        "Cluster-enabled kernels fail fast on hardware without DSMEM support, while DSMEM-free variants still execute-use this to confirm cluster capability flags.",
        "`python -m cli.aisp bench run --targets ch10:flash_attention --profile minimal` produces fewer kernel launches and higher achieved FLOP/s than the baseline script.",
        "`python -m ch10.analyze_scaling` summarizes the chapter's scaling behavior without relying on path surgery.",
        "`python -m ch10.cufile_gds_example` runs the CUDA memory pipeline and GDS demo, highlighting launch amortization and IO overlap.",
    ],
    notes=[
        "`cufile_gds_example.py` demonstrates integrating GPUDirect Storage into tensor-core pipelines for IO-heavy training loops.",
        "`requirements_cufile.txt` holds the optional `cufile` wheel; install it only on hosts with GPUDirect Storage enabled.",
        "The CUTLASS-style warp-specialization pair provides a reference implementation aligned with `sm100_mma_array_warpspecialized` for performance comparison.",
    ],
)

ENTRIES["ch11"] = chapter_entry(
    slug="ch11",
    title="Chapter 11 - Streams & Concurrency",
    summary=dedent(
        """\
        Explains how to overlap compute, memory, and communication on Blackwell using CUDA streams, ordered sequences, Hyper-Q, warp-specialized pipelines, and adaptive scheduling. The README keeps the legacy target names tied to the actual copy+elementwise overlap workload and runtime-adaptive scheduling story instead of pretending every target is a generic multi-stream demo. CUDA Graph capture and graph-level orchestration continue in the manuscript and repo Chapter 12; this chapter’s runnable surface stays focused on stream/concurrency paths."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 11 is where concurrency ideas have to prove they are reducing real idle time instead of just making traces look busier. The useful question is not "can we add streams?" but "which ordering and overlap changes actually improve the measured workload?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - more serialized stream usage
                - conservative ordering that protects correctness but leaves overlap untapped
                - simpler to debug, but often too launch- and idle-heavy"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - stream overlap where work is truly independent
                - stream-ordered cache and KV update paths that preserve correctness without full serialization
                - warp-specialized multistream execution where the hardware can support it"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `streams` | `15.035 ms` | `8.073 ms` | `1.86x` | basic overlap instead of more serialized launches |
                | `stream_ordered_kv_cache` | `3.153 ms` | `2.103 ms` | `1.50x` | ordered KV updates with less idle time |
                | `warp_specialization_multistream` | `17.530 ms` | `10.500 ms` | `1.67x` | multistream warp specialization path |

                These are not "big baseline mistake" wins. They are the more realistic kind of concurrency gains where overlap helps, but only when the work graph actually allows it."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want to confirm that the gain is real overlap rather than timing noise:

                ```bash
                python -m cli.aisp bench run --targets ch11:streams --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch11:stream_ordered_kv_cache --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch11:warp_specialization_multistream --profile deep_dive --single-gpu
                ```

                The Nsight story should differ by workload:
                - `streams`: less idle between independent launches
                - `stream_ordered_kv_cache`: correctness-preserving ordering without full-device serialization
                - `warp_specialization_multistream`: more useful overlap across specialized work partitions"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch11.compare
                python -m cli.aisp bench list-targets --chapter ch11
                python -m cli.aisp bench run --targets ch11 --profile minimal
                python -m cli.aisp bench run --targets ch11:streams --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Use multiple CUDA streams to overlap independent kernels without starving priority work.",
        "Control ordering constraints for KV-cache updates and stream-ordered memory pools.",
        "Benchmark warp-specialized multistream kernels that share data via DSMEM.",
        "Introduce adaptive policies that adjust stream usage based on runtime telemetry.",
    ],
    contents=[
        ("`baseline_streams.py`, `optimized_streams.py`, `streams_overlap_demo.cu`, `streams_ordered_demo.cu`, `streams_warp_specialized_demo.cu`, `stream_overlap_base.py`", "Core stream overlap demos that contrast serialized launches with overlapped workloads."),
        ("`baseline_stream_ordered.py`, `baseline_stream_ordered_kv_cache.py`, `optimized_stream_ordered.py`, `optimized_stream_ordered_kv_cache.py`", "Stream-ordered allocator and KV-cache examples ensuring deterministic updates while enabling overlap."),
        ("`baseline_gemm_streams.py`, `optimized_gemm_streams.py`, `baseline_tensor_cores_streams.py`, `optimized_tensor_cores_streams.py`", "GEMM pipelines that schedule tensor-core kernels across multiple streams to decouple math vs IO phases."),
        ("`baseline_distributed_streams.py`, `optimized_distributed_streams.py`, `baseline_adaptive_streams.py`, `optimized_adaptive_streams.py`", "Adaptive streaming controllers that balance NCCL, compute, and IO tasks on large systems."),
        ("`baseline_warp_specialization_multistream.*`, `optimized_warp_specialized_multistream.*`, `warp_specialized_cluster_pipeline_multistream.cu`", "Warp-specialized multistream kernels demonstrating DSMEM usage and per-stream specialization."),
        ("`compare.py`, `Makefile`, `expectations_{hardware_key}.json`", "Harness driver plus expectation data for concurrency regressions."),
    ],
    validation=[
        "`python optimized_streams.py --trace` captures overlapping NVTX ranges in Nsight Systems, proving concurrency is active.",
        "`python optimized_stream_ordered_kv_cache.py --validate` matches the baseline's outputs while reducing idle gaps between cache updates.",
        "Warp-specialized multistream kernels flag unsupported hardware (missing DSMEM) immediately, preventing silent fallbacks.",
    ],
    notes=[
        "The README calls out the legacy target names explicitly so book-facing labels still point at the actual copy+elementwise overlap workload and runtime-adaptive scheduling pairs.",
        "`warp_specialized_triton.py` provides a Triton analogue for the CUDA concurrency demos so you can compare compiler-generated schedules.",
        "`kv_prefetch_pipeline_enhanced_demo.cu` builds on the DSMEM kernels bundled in this directory so you can study the entire pipeline locally.",
    ],
)

ENTRIES["ch12"] = chapter_entry(
    slug="ch12",
    title="Chapter 12 - CUDA Graphs & Dynamic Workloads",
    summary=dedent(
        """\
        Covers modern CUDA Graph capabilities-conditional capture, graph memory tuning, dynamic parallelism, and work queues-to keep irregular workloads performant without per-launch overhead. The current repo chapter emphasizes the single-GPU graph/work-queue side of the manuscript; the larger multi-GPU NCCL/NVSHMEM orchestration arc is only partially represented here."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 12 is where launch overhead and dynamic work management have to justify themselves with measured wins. The useful question is not "can we capture a graph?" but "which graph or dynamic-work techniques actually reduce the real runtime once correctness and workload shape stay fixed?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager launches and more CPU-visible scheduling work
                - less reuse of graph capture or GPU-resident work management
                - easy to inspect, but often too expensive for irregular steady-state workloads"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - CUDA Graph replay where the steady-state workload is stable enough
                - fused or GPU-resident queueing/dispatch where it actually removes launch overhead
                - measured through the shared harness instead of hand-timed one-off scripts"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `cuda_graphs` | `529.532 ms` | `125.874 ms` | `4.21x` | graph replay instead of repeated eager launch overhead |
                | `kernel_fusion` | `1.776 ms` | `0.654 ms` | `2.72x` | fewer launches through fused graph-friendly execution |
                | `work_queue` | `2.100 ms` | `0.442 ms` | `4.75x` | GPU-resident work queue path |

                This chapter is useful because it separates "graphs help" from "graphs help on a workload that is actually stable enough to benefit." The work-queue target also keeps the chapter from being only about graph replay."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want hard evidence for launch reduction and work scheduling changes:

                ```bash
                python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch12:kernel_fusion --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch12:work_queue --profile deep_dive --single-gpu
                ```

                Those targets answer slightly different questions:
                - `cuda_graphs`: replay payoff
                - `kernel_fusion`: launch-count reduction
                - `work_queue`: GPU-side dynamic dispatch effectiveness"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch12.compare
                python -m cli.aisp bench list-targets --chapter ch12
                python -m cli.aisp bench run --targets ch12 --profile minimal
                python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Capture steady-state workloads into CUDA Graphs and study the delta vs eager launches.",
        "Use conditional nodes and graph memory pools for adaptive pipelines.",
        "Experiment with device-side launches (dynamic parallelism) to reduce CPU involvement.",
        "Implement GPU-resident work queues and uneven partition schedulers.",
    ],
    contents=[
        ("`baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu`", "Graph capture demos that evolve from simple replay to conditional and DSM-aware execution, including a supplementary host-dispatch-versus-device-branching conditional graph demo."),
        ("`baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py`", "Launch- and bandwidth-focused studies illustrating how graphs reduce CPU overhead."),
        ("`baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu`, `dynamic_parallelism_sm121/`", "Device-side launch samples showing when dynamic parallelism helps or hurts."),
        ("`baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh`", "GPU work queues for irregular batch sizes, including NVTX instrumentation."),
        ("`baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu`", "Uneven workload partitioners that rebalance CTA assignments at runtime."),
        ("`baseline_kernel_fusion.py`, `optimized_kernel_fusion.py`, `kernel_fusion_cuda_demo.cu`", "Kernel fusion exercises within graph capture so you can remove CPU synchronization entirely. (`kernel_fusion_cuda_demo.cu` is a standalone tool; not a benchmark target.)"),
        ("`compare.py`, `cuda_extensions/`, `expectations_{hardware_key}.json`", "Harness entry, extension stubs, and expectation thresholds."),
    ],
    validation=[
        "`python optimized_cuda_graphs.py --iterations 100` should report lower wall-clock time than the baseline while matching outputs.",
        "Device-side dynamic parallelism samples emit warnings on unsupported hardware, ensuring you only trust data from GPUs with the feature enabled.",
        "`python optimized_work_queue.py --trace` exposes balanced dequeue times across CTAs when compared to the baseline's stragglers.",
    ],
    notes=[
        "`cuda_graphs_workload.cuh` holds reusable graph capture helpers when you want to wrap your own kernels.",
        "`helper_*.cu` files contain host/device glue for the dynamic-parallelism case studies-copy them when bootstrapping new experiments.",
        "`graph_conditional_runtime` is the canonical conditional-node benchmark. Keep `cuda_graphs_conditional` as a supplementary demo rather than a canonical speedup target.",
        "For a smaller supporting lab that keeps Chapter 12's CUDA Graph story primary while adding fused reduction and padding-aware projection examples, see `labs/training_hotpath`.",
    ],
)

ENTRIES["ch13"] = chapter_entry(
    slug="ch13",
    title="Chapter 13 - PyTorch Profiling & Memory Tuning",
    summary=dedent(
        """\
        Focuses on PyTorch-centric optimizations: compiled autograd, memory profiling, FSDP/context/expert parallelism, and FP8/quantization workflows backed by the same harness infrastructure. The chapter README is fairness-refreshed so canonical pairs stay separate from informational variants such as `torchao_quantization_compiled`, `kv_cache_naive_flash_blockwise`, and the torchao FP8 recipe demos (`precisionfp8`, `precisionfp8_rowwise`, `precisionfp8_rowwise_gw_hp`). The manuscript walkthrough uses a fuller Hugging Face/MoE profiling example, while the repo chapter keeps the runnable surface lighter and more harness-native."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 13 is where high-level PyTorch optimizations have to prove they are doing more than rearranging framework overhead. The useful question is not "can PyTorch do this optimization?" but "which profiling, compilation, precision, and memory changes actually improve the workload under the shared harness?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager or less-optimized PyTorch execution
                - higher-overhead cache, precision, and dataloader paths
                - easier to debug, but often too expensive once memory and framework overhead dominate"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compiled, quantized, or allocator-aware PyTorch paths where they produce a real measured benefit
                - lower-overhead cache and attention paths
                - still benchmarked through the same harness contract, so the numbers stay comparable to the lower-level chapters"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `kv_cache_naive` | `1664.672 ms / 1194.135 MB` | `1739.080 ms / 370.394 MB` | `68.98% less memory` | token-by-token paged allocation preserves the batch contract while cutting KV-cache footprint |
                | `memory_profiling` | allocator fragmentation baseline | gradient-checkpointed path | memory-goal benchmark | checkpointing is judged by lower peak allocation and cleaner allocator behavior, not by raw speedup |
                | `autograd_standard` | `1.644 ms` | `0.204 ms` | `8.04x` | compiled/optimized autograd path |
                | `precisionfp8_te` | `2.800 ms` | `0.542 ms` | `5.17x` | Transformer Engine FP8 path |

                This chapter is one of the easiest places to fool yourself with framework overhead. That is why the benchmark contract and side-by-side baseline/optimized structure matter here more than almost anywhere else."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want to see whether the gain came from framework overhead reduction, memory behavior, or the lower-precision path itself:

                ```bash
                python -m cli.aisp bench run --targets ch13:kv_cache_naive --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch13:autograd_standard --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu
                ```

                Those targets cover three different PyTorch optimization stories:
                - `kv_cache_naive`: cache-path and memory behavior, with memory reduction treated as the primary win
                - `autograd_standard`: framework/compile overhead
                - `precisionfp8_te`: lower-precision execution with real library support

                The torchao FP8 recipe demos (`precisionfp8`, `precisionfp8_rowwise`, `precisionfp8_rowwise_gw_hp`) remain useful implementation references, but they are treated as informational examples rather than canonical speed-claim surfaces."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch13.compare
                python -m cli.aisp bench list-targets --chapter ch13
                python -m cli.aisp bench run --targets ch13 --profile minimal
                python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Profile PyTorch training loops end-to-end, capturing goodput, memory, and kernel traces.",
        "Apply `torch.compile`, regional compilation, and custom allocators to reduce overhead.",
        "Tune DataLoader, KV-cache, and optimizer states to eliminate fragmentation.",
        "Exercise FP8/quantized training recipes with Transformer Engine integration.",
    ],
    contents=[
        ("`baseline_training_standard.py`, `optimized_training_standard.py`, `train.py`, `train_deepseek_v3.py`, `train_deepseek_coder.py`", "Reference training loops showcasing eager vs compiled paths and DeepSeek-inspired configs."),
        ("`baseline_dataloader_default.py`, `optimized_dataloader_default.py`, `baseline_memory_profiling.py`, `optimized_memory_profiling.py`, `memory_profiling.py`", "DataLoader/memory studies that explain how to read allocator stats and fix leaks."),
        ("`baseline_attention_standard.py`, `optimized_attention_standard.py`, `baseline_long_context_attention.py`, `optimized_long_context_attention.py`, `baseline_arithmetic_intensity.py`, `optimized_arithmetic_intensity.py`, `baseline_matmul_pytorch.py`, `optimized_matmul_pytorch.py`", "Attention and matmul microbenchmarks tuned purely within PyTorch, including long-context Flash SDP."),
        ("`baseline_context_parallel_multigpu.py`, `optimized_context_parallel_multigpu.py`, `context_parallel_benchmark_common.py`", "Context-parallel attention benchmarks comparing all-gather vs ring-style streaming across ranks."),
        ("`baseline_expert_parallel_multigpu.py`, `optimized_expert_parallel_multigpu.py`, `expert_parallel_common.py`", "Expert-parallel all-to-all benchmarks contrasting per-iteration list allocations vs pre-allocated all_to_all_single."),
        ("`context_parallelism.py`, `fsdp_example.py`", "Context and FSDP sharding demos for scaling beyond a single GPU. (Tools; not benchmark targets.)"),
        ("`baseline_precisionfp8*.py`, `optimized_precisionfp8*.py`, `baseline_precisionmixed.py`, `optimized_precisionmixed.py`, `compiled_autograd.py`", "Precision-management suites covering Transformer Engine, torchao FP8 recipe demos, and compiled autograd recipes."),
        ("`baseline_quantization.py`, `optimized_quantization.py`, `baseline_kv_cache_naive.py`, `optimized_kv_cache_naive.py`, `optimized_kv_cache_naive_pool.py`", "Quantization and KV-cache pipelines for inference/training memory savings, including the quantization-only canonical pair and a token-by-token decode with naive concat cache versus paged cache allocation."),
        ("`compare.py`, `compare_perf.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `workload_config.py`", "Harness entry, performance comparison helper, dependencies, and regression baselines."),
    ],
    validation=[
        "`python -m ch13.compare --examples training_standard` shows optimized training runs producing higher goodput with identical metrics.",
        "`python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile minimal` confirms Transformer Engine calibration plus NVFP8 execution with max error tolerances enforced.",
        "`python -m ch13.memory_profiling --dump` and the optimized variant demonstrate allocator fragmentation dropping after applying the recommended knobs, with memory reduction treated as the primary benchmark outcome.",
    ],
    notes=[
        "`custom_allocator.py` contains a standalone torch allocator shim that can be re-used in other chapters when debugging fragmentation.",
        "`compiled_autograd.py` doubles as a tutorial on partial graph capture; the README here references it directly.",
        "`torchao_quantization_compiled`, `kv_cache_naive_flash_blockwise`, `precisionfp8`, `precisionfp8_rowwise`, and `precisionfp8_rowwise_gw_hp` remain informational variants.",
        "`kv_cache_naive` and `memory_profiling` are memory-goal benchmarks; they are expected to reduce memory pressure even when the timed path is not faster.",
    ],
)

ENTRIES["ch14"] = chapter_entry(
    slug="ch14",
    title="Chapter 14 - Compiler & Triton Optimization",
    summary=dedent(
        """\
        Highlights compiler-driven acceleration: `torch.compile` workflows, Triton kernels, CUTLASS/TMA experimentation, and quantization-aware communication, all validated through the shared harness. The repo chapter focuses on CUDA/Triton/Inductor paths; the broader XLA backend discussion from the manuscript is not represented as runnable chapter code here. The explicit `cublas_vs_cutlass` pair remains an informational comparison surface rather than a chapter-native speed-claim benchmark."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 14 is where compiler claims have to turn into measured wins. The useful question is not "can `torch.compile` or Triton work?" but "which compiler-driven optimizations still deliver real latency and memory reductions on current Blackwell-class hardware?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager or minimally fused PyTorch execution
                - generic Triton/CUTLASS paths without persistent or regional specialization
                - easier to reason about, but heavy on launch overhead, graph breaks, and redundant staging"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - `torch.compile` and regional compilation where the graph is stable enough to pay back compile cost
                - Triton persistent kernels and TMA-fed schedules where memory movement dominates
                - the same harness contract as every other benchmarked chapter, so the speedups are comparable instead of script-local"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `model_compile_reduced_precision` | `29.873 ms` | `7.978 ms` | `3.74x` | eager reduced precision vs reduced precision + `torch.compile` |
                | `regional_triton` | `1.944 ms` | `0.863 ms` | `2.25x` | regional compilation and Triton fusion |
                | `triton_persistent` | `0.830 ms` | `0.086 ms` | `9.68x` | persistent Triton kernel scheduling |

                These are chapter-level proof points, not vendor peak numbers. The chapter is most useful when you separate "compiler removes Python/graph overhead" from "kernel schedule removes memory-movement overhead." """
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the same benchmark targets with deep-dive profiling when you want launch-count and kernel-attribution evidence instead of only the wall-clock delta:

                ```bash
                python -m cli.aisp bench run --targets ch14:model_compile_reduced_precision --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch14:regional_triton --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch14:triton_persistent --profile deep_dive --single-gpu
                ```

                The expected story is different per workload:
                - `model_compile_reduced_precision`: fewer graph breaks and lower framework overhead, with both paths using the same reduced-precision dtype and the optimized path adding `torch.compile`
                - `regional_triton`: fewer unfused launches and better steady-state scheduling
                - `triton_persistent`: materially longer-lived kernels with less relaunch churn"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch14.compare --profile none
                python -m cli.aisp bench list-targets --chapter ch14
                python -m cli.aisp bench run --targets ch14 --profile minimal
                python -m cli.aisp bench run --targets ch14:triton_persistent --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Adopt `torch.compile` modes for large models while tracking compile-time and steady-state gains.",
        "Author Triton kernels (including TMA schedules) that rival custom CUDA.",
        "Profile FlexAttention and regional compilation strategies end-to-end.",
        "Blend quantization with NCCL and pipeline overlap without regressions.",
    ],
    contents=[
        ("`baseline_model_compile_reduced_precision.py`, `optimized_model_compile_reduced_precision.py`, `model_eager_common.py`, `torch_compile_large_model.py`, `torch_compiler_examples.py`, `training_large_model_1_5x.py`", "Model-scale examples showcasing the eager-vs-compiled reduced-precision pair, shared transformer scaffolding, compile modes, guard rails, and large-model sanity tests."),
        ("`baseline_cublas_vs_cutlass.py`, `optimized_cublas_vs_cutlass.py`, `triton_examples.py`, `triton_tma_blackwell.py`, `triton_fp8_advanced.py`, `triton_nvshmem_example.py`", "Explicit cuBLAS-vs-CUTLASS comparison pair plus advanced TMA/NVSHMEM Triton kernels. The comparison pair remains informational rather than a canonical chapter speed claim."),
        ("`baseline_attention_eager_sdpa.py`, `optimized_attention_eager_sdpa.py`, `baseline_flex_attention_sparse.py`, `optimized_flex_attention_sparse.py`, `flex_attention_sparse_demo.py`", "Eager-vs-SDPA attention plus FlexAttention sparse workloads that validate custom score mods, masks, sparsity, and compile speedups."),
        ("`baseline_nccl_quantization.py`, `optimized_nccl_quantization.py`, `deepseek_innovation_l2_bypass.py`", "Quantization-aware communication and the DeepSeek-inspired L2 bypass experiment."),
        ("`baseline_regional_triton.py`, `optimized_regional_triton.py`, `inspect_compiled_code.py`, `benchmark_tma_configs.py`", "Regional compilation and TMA parameter sweeps for auto-tuning generated kernels."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `train.py`, `transformer.py`", "Harness entry plus model definitions and dependency pins."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets ch14:model_compile_reduced_precision --profile minimal` produces compile-time summaries followed by steady-state throughput gains vs an eager baseline running the same reduced-precision model.",
        "`python -m ch14.triton_tma_blackwell --validate` compares Triton and CUDA outputs to double-check TMA scheduling logic.",
        "`python -m ch14.compare --examples attention_eager_sdpa` shows the fused SDPA path reducing kernel launch count without changing accuracy.",
    ],
    notes=[
        "`inspect_compiled_code.py` dumps Triton/PTX/Graph captures for any target; edit the helper to introspect new workloads.",
        "`requirements.txt` includes nightly Triton + PyTorch wheels to keep compiler features aligned with the CUDA 13 toolchain.",
        "For repo-native supporting examples that fill the training hot-path gaps without changing this chapter's primary compile narrative, see `labs/training_hotpath`.",
        "`cublas_vs_cutlass` is a supplementary comparison pair. Chapter-native performance claims stay anchored on `model_compile_reduced_precision`, `regional_triton`, and `triton_persistent`.",
    ],
)

ENTRIES["ch15"] = chapter_entry(
    slug="ch15",
    title="Chapter 15 - Disaggregated Inference & KV Management",
    summary=dedent(
        """\
        Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns. The repo chapter captures the disaggregated serving and KV-management themes directly, while the NIXL-specific connector story from the manuscript is only represented indirectly."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 15 is where inference-system ideas have to justify themselves with end-to-end measurements. The useful question is not "can we disaggregate or batch this?" but "which orchestration changes actually reduce latency or increase throughput once KV movement and scheduling overhead are included?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - monolithic or minimally coordinated inference execution
                - straightforward KV management and queue draining
                - easy to reason about, but expensive once prefill/decode and cache movement start to dominate"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - disaggregated prefill/decode and batched scheduling where they help
                - NVLink-pooled KV-cache strategies and topology-aware routing
                - still measured through the shared benchmark harness, so the chapter is a performance case study instead of a pile of demos"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `continuous_batching` | `52.955 ms` | `12.719 ms` | `4.16x` | queueing and batching strategy |
                | `kv_cache_nvlink_pool` | `1047.860 ms` | `171.477 ms` | `6.11x` | pooled KV-cache path |
                | `guided_decoding` | `12.702 ms` | `2.131 ms` | `5.96x` | guided decode path |
                | `greedy_sampler` | `3.221 ms` | `0.907 ms` | `3.55x` | skips full-vocab probability materialization for greedy decode |
                | `speculative_decoding` | `103.323 ms` | `26.761 ms` | `3.86x` | speculative decode orchestration |

                The `greedy_sampler` row is from `artifacts/runs/codex_ch15_greedy_sampler_20260630_212500/`. It is a semantically narrow serving fast path where temperature-positive greedy decode uses logits argmax directly instead of building a probability tensor. The chapter mixes system-level wins from queueing/orchestration with fabric/cache-path wins. Those are both valuable, but they are not the same optimization story."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want to attribute the gains to scheduling, cache movement, or decode behavior instead of only quoting the runtime delta:

                ```bash
                python -m cli.aisp bench run --targets ch15:continuous_batching --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch15:kv_cache_nvlink_pool --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch15:greedy_sampler --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch15:speculative_decoding --profile deep_dive --single-gpu
                ```

                Those runs are the right place to check whether the win came from less queue idle time, less cache movement, or fewer wasted decode steps."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch15.compare
                python -m cli.aisp bench list-targets --chapter ch15
                python -m cli.aisp bench run --targets ch15 --profile minimal
                python -m cli.aisp bench run --targets ch15:kv_cache_nvlink_pool --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.",
        "Design KV-cache managers that gracefully span local and remote HBM pools.",
        "Implement continuous batching and queueing so decode throughput stays high.",
        "Use decode-policy fast paths so greedy serving avoids unnecessary probability tensors.",
        "Serve MoE models efficiently by pairing routing with optimized communication.",
    ],
    contents=[
        ("`baseline_inference_monolithic.py`, `optimized_inference_monolithic.py`", "Single-box inference loops that establish the baseline before disaggregation."),
        ("`disaggregated_inference_multigpu.py`", "Disaggregated inference demo that layers speculative decoding on top of prefill/decode pools."),
        ("`baseline_single_gpu_kv_handoff.py`, `optimized_single_gpu_kv_handoff.py`, `baseline_disaggregated_inference_multigpu.py`, `optimized_disaggregated_inference_multigpu.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`, `baseline_prefill_decode_disagg_multigpu.py`, `optimized_prefill_decode_disagg_multigpu.py`, `disaggregated_inference_single_common.py`", "Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling (multi-GPU), plus a supplementary single-GPU KV-handoff comparison pair."),
        ("`baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `kv_cache_management_math.py`, `baseline_kv_cache_nvlink_pool.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_nvlink_pool_multigpu.py`, `optimized_kv_cache_nvlink_pool_multigpu.py`", "KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants."),
        ("`baseline_continuous_batching.py`, `optimized_continuous_batching.py`", "Single-GPU continuous batching scheduler for TTFT-aware queueing."),
        ("`baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py`", "Multi-GPU continuous batching scheduler for scaled queueing throughput."),
        ("`baseline_greedy_sampler.py`, `optimized_greedy_sampler.py`, `greedy_sampler_common.py`", "Greedy decode sampler comparison showing that positive-temperature argmax can skip full-vocab softmax materialization."),
        ("`baseline_moe_inference.py`, `optimized_moe_inference.py`", "Inference-specific MoE workloads that pair router load with communication control."),
        ("`baseline_moe_overlap.py`, `optimized_moe_overlap_shared_expert.py`, `baseline_moe_overlap_local_route.py`, `optimized_moe_overlap_local_route.py`, `moe_overlap_local_route_common.py`, `baseline_wide_ep.py`, `optimized_wide_ep.py`, `baseline_moe_dispatch.py`, `optimized_moe_dispatch.py`, `baseline_moe_routing_topology_aware.py`, `optimized_moe_routing_topology_aware.py`", "MoE expert-parallel microbenchmarks that now split dispatch-path optimization, direct local routed placement, and topology-aware routing locality so attribution stays clean."),
        ("`compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `Makefile`", "Harness entry and dependencies for inference-focused validation."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets ch15:disaggregated_inference_multigpu --profile minimal --ncu-replay-mode kernel` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity (kernel replay avoids NCU application-replay stalls on this workload).",
        "`python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.",
        "`python compare.py --examples continuous_batching` (single GPU) and `python compare.py --examples continuous_batching_multigpu` (multi-GPU) show optimized scheduling increases tokens/sec vs naive queue draining.",
        "`python -m cli.aisp bench run --targets ch15:greedy_sampler --profile minimal --single-gpu` verifies the direct-logit argmax path matches the full-probability baseline exactly.",
    ],
    notes=[
        "`disaggregated_inference_multigpu.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.",
        "Use `torchrun --nproc_per_node <num_gpus>` to run the disaggregated pipeline on the desired GPU count (defaults to all visible GPUs, even count).",
        "`Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.",
    ],
)

ENTRIES["ch16"] = chapter_entry(
    slug="ch16",
    title="Chapter 16 - Production Inference Optimization",
    summary=dedent(
        """\
        Focuses on real-world inference services: paged attention, Flash SDP, FP8 serving, telemetry hooks, schedulers, and Blackwell-friendly load-test harnesses."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 16 is where "serving optimization" stops being a collection of tricks and becomes a latency budget. The chapter is most useful when it proves which serving-path changes actually improve steady-state latency, scheduling efficiency, or memory behavior under the shared harness."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - straightforward serving loops with conservative attention and scheduling choices
                - little or no graph capture, cache-aware staging, or backend specialization
                - easier to debug, but usually too expensive for production latency targets"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - Flash SDP, block-sparse attention, and scheduler-aware execution where they help
                - selective graph/compilation techniques for steady-state serving paths
                - the same benchmark harness contract as the rest of the repo, so the gains are comparable and reproducible"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `flash_sdp` | `0.322 ms` | `0.198 ms` | `1.63x` | Flash SDP path |
                | `flashinfer_block_sparse` | `0.941 ms` | `0.239 ms` | `3.94x` | block-sparse attention path |
                | `runtime_scheduler` | `112.762 ms` | `63.425 ms` | `1.78x` | scheduler/runtime coordination |

                The good chapter-level read is "which serving-path changes help enough to matter?" rather than trying to average these into one generic serving number."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want Nsight-backed evidence for backend selection and scheduling behavior:

                ```bash
                python -m cli.aisp bench run --targets ch16:flash_sdp --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch16:flashinfer_block_sparse --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch16:runtime_scheduler --profile deep_dive --single-gpu
                ```

                Those targets answer different questions:
                - `flash_sdp`: better attention backend choice
                - `flashinfer_block_sparse`: structured sparsity payoff
                - `runtime_scheduler`: queueing and scheduling overhead reduction"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch16.compare
                python -m cli.aisp bench list-targets --chapter ch16
                python -m cli.aisp bench run --targets ch16 --profile minimal
                python -m cli.aisp bench run --targets ch16:flash_sdp --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Profile large decoder workloads to spot hotspots before deploying models.",
        "Adopt paged attention, Flash SDP, and piecewise compilation to hit latency targets.",
        "Integrate FP8 quantization, symmetric memory, and cache monitoring in serving loops.",
        "Simulate production loads (multi-node, MoE) while validating accuracy via perplexity checks.",
    ],
    contents=[
        ("`inference_optimizations_blackwell.py`, `inference_profiling.py`, `inference_server_load_test.py`, `inference_serving_multigpu.py`", "Top-level orchestration scripts for profiling and load testing multi-GPU inference deployments."),
        ("`baseline_flash_sdp.py`, `optimized_flash_sdp.py`, `baseline_dense_attention_flash.py`, `optimized_dense_attention_flash.py`, `optimized_dense_attention_flash_blackwell_variant.py`", "Attention kernels that compare naive implementations versus Flash backends, including an explicit non-canonical hardware variant for the Blackwell-tagged dense-attention path."),
        ("`baseline_piece_graphs.py`, `optimized_piece_graphs.py`, `baseline_regional_compilation.py`, `optimized_regional_compilation.py`", "Piecewise graph capture and regional compilation for stable low-latency decode, with explicit steady-state replay metrics such as `regional_compilation.capture_ms`, `regional_compilation.graph_bucket_count`, and `regional_compilation.steady_state_only`."),
        ("`fp8_transformer_engine.py`, `test_fp8_quantization_real.py`, `symmetric_memory_inference.py`, `multi_gpu_validation.py`", "Serving-time FP8 and symmetric-memory validations to guarantee accuracy and NVLink efficiency."),
        ("`moe_performance_benchmark.py`, `synthetic_moe_inference_benchmark.py`, `moe_workload.py`", "MoE inference harnesses that stress router placement and per-expert batching."),
        ("`cache_monitoring.py`, `dcgm_prometheus_exporter.py`, `scheduler.py`, `perplexity_eval.py`", "Telemetry, scheduling, and accuracy utilities wired into the inference pipeline."),
        ("`compare.py`, `requirements.txt`, `Makefile`, `expectations_{hardware_key}.json`", "Harness entry and dependencies for inference-focused verification."),
    ],
    validation=[
        "`python optimized_dense_attention_flash.py --profile minimal` yields fewer page faults and improved throughput relative to the baseline script.",
        "`python symmetric_memory_inference.py --validate` confirms NVLink-backed KV replicas stay in sync with negligible skew.",
        "`python inference_server_load_test.py --duration 120` exercises the scheduler and should report stable TTFT/TPOT metrics after warm-up.",
        "`python -m cli.aisp bench run --targets ch16:regional_compilation --profile minimal --single-gpu` should emit `regional_compilation.capture_ms`, `regional_compilation.graph_bucket_count`, and `regional_compilation.steady_state_only=1` so replay-only timing stays auditable.",
    ],
    notes=[
        "`dcgm_prometheus_exporter.py` emits per-GPU metrics consumable by Prometheus/Grafana without extra setup.",
        "`cache_monitoring.py` can be run standalone to sanity-check allocator health between runs.",
        "`optimized_dense_attention_flash_blackwell_variant.py` is an explicit non-canonical hardware variant of `baseline_dense_attention_flash.py`; keep it out of canonical expectation coverage unless it grows real hardware-specific behavior.",
    ],
)

ENTRIES["labs/kv_optimization"] = lab_entry(
    slug="labs/kv_optimization",
    title="Lab - KV Cache Optimization",
    summary=dedent(
        """\
        Compares a standard FP16 KV cache path against a compressed KV-cache implementation so longer context lengths fit without treating memory reduction as a free lunch."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                KV cache growth is one of the fastest ways to turn a good inference path into an unusable one. This lab exists to measure how much memory the cache optimization actually gives back, and what latency tradeoff you pay to get it."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - standard FP16 KV cache
                - simple, high-fidelity, and expensive in HBM
                - useful as the correctness and memory reference"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compressed KV cache with lower memory footprint
                - benchmarked through the same harness path, so the speed/memory tradeoff is explicit
                - designed to answer "does the memory saving justify the latency change?" instead of assuming quantization is always a win"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current validated expectation-backed B200 result from `labs/kv_optimization/expectations_b200.json`:

                | Target | Baseline | Optimized | Measured delta | Memory change |
                | --- | ---: | ---: | ---: | ---: |
                | `kv_standard` | `3782.365 ms` | `1777.506 ms` | `2.13x` | `49.77%` lower memory |

                That run recorded:
                - baseline memory: `32916.315 MB`
                - optimized memory: `16534.378 MB`

                This lab is useful because it makes the speed/memory tradeoff explicit instead of treating KV compression as a free optimization."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                The first thing to trust here is the benchmark pair and its recorded memory delta. When you want deeper attribution, run the same target through the harness with profiling enabled:

                ```bash
                python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/kv_optimization
                python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile minimal
                python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure the latency and memory tradeoff of KV-cache compression instead of optimizing for one metric in isolation.",
        "Use the shared harness to keep the baseline and optimized cache paths directly comparable.",
        "Validate that memory savings survive the same contract checks as every other repo benchmark.",
    ],
    contents=[
        ("`baseline_kv_standard.py`", "Reference FP16 KV-cache path."),
        ("`optimized_kv_standard.py`", "Compressed KV-cache path used for the optimized benchmark."),
        ("`expectations_{hardware_key}.json`", "Stored speedup and memory-savings baselines for the lab."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile minimal` reports both latency and memory deltas for the pair.",
        "The optimized path should reduce memory materially without violating the benchmark contract or correctness checks.",
        "`python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu` produces profiler artifacts for the same measured path.",
    ],
    notes=[
        "This README is now generator-owned; update the source of truth in `core/scripts/refresh_readmes.py`, not the rendered file.",
        "The current public numbers come from the stored expectation baseline because there is no newer canonical tier-1 history artifact for this lab yet.",
    ],
)

ENTRIES["ch17"] = chapter_entry(
    slug="ch17",
    title="Chapter 17 - Disaggregated Prefill/Decode & Routing",
    summary=dedent(
        """\
        Centers the chapter on disaggregated prefill/decode serving, then layers routing and scheduling policies on top so Blackwell clusters can hand work between prefill/decode pools, MoE experts, and pipeline stages without sacrificing utilization."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 17 is where routing and disaggregation ideas stop being whiteboard architecture and start paying rent. The useful question is not "can we route dynamically?" but "which router, queueing, and handoff changes actually improve TTFT, TPOT, or throughput once the full prefill/decode path is measured?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - static or minimally adaptive routing
                - conservative prefill/decode handoff with more blocking behavior
                - easy to reason about, but expensive once queue imbalance and KV movement dominate"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - topology-aware or telemetry-aware routing decisions
                - disaggregated prefill/decode paths that reduce idle time and handoff overhead
                - measured through the shared harness so routing wins are comparable to kernel and memory wins elsewhere in the repo"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `routing_static` | `5.680 ms` | `0.804 ms` | `7.07x` | smarter routing policy without changing the underlying workload |
                | `moe_router_uniform` | `4.719 ms` | `0.931 ms` | `5.07x` | topology-aware expert routing instead of uniform placement |
                | `prefill_decode_disagg_ttft` | `2678.148 ms` | `938.237 ms` | `2.85x` | disaggregated prefill/decode handoff optimized for TTFT |

                This chapter mixes policy wins with orchestration wins. That is useful, but it means you should read each target as a specific system story rather than as one generic routing number.
                Use the `prefill_decode_disagg*` targets as the chapter-native exemplars; `inference_full` remains a comparison pair for model-side work reduction rather than a disaggregated serving benchmark. Its structured metrics now expose `active_layers`, `identity_layers_skipped`, `story.comparison_pair=1`, and `story.chapter_native_exemplar=0`, while structured story metadata points to the `prefill_decode_disagg*` family as the chapter-native exemplar set."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want evidence for where the gain came from instead of only the final runtime delta:

                ```bash
                python -m cli.aisp bench run --targets ch17:routing_static --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch17:moe_router_uniform --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch17:prefill_decode_disagg_ttft --profile deep_dive --single-gpu
                ```

                Those three targets answer different questions:
                - `routing_static`: policy overhead versus routing quality
                - `moe_router_uniform`: topology-aware MoE routing payoff
                - `prefill_decode_disagg_ttft`: queueing and handoff behavior in a split prefill/decode system"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch17.compare
                python -m cli.aisp bench list-targets --chapter ch17
                python -m cli.aisp bench run --targets ch17 --profile minimal
                python -m cli.aisp bench run --targets ch17:prefill_decode_disagg_ttft --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Implement dynamic routers that react to TTFT, TPOT, and KV-locality metrics.",
        "Profile complete inference stacks (prefill + decode) under realistic synthetic loads.",
        "Blend pipeline parallelism with routing logic for long-context workloads.",
        "Document profiling steps (roofline, Nsight) specific to the routing lab.",
    ],
    contents=[
        ("`baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py`", "Routing controllers that evolve from static heuristics to telemetry-driven admission and rejection policies."),
        ("`baseline_inference_full.py`, `optimized_inference_full.py`", "Control pair for full-depth inference versus early-exit pruning. Useful as an end-to-end inference sanity check, but not the chapter's primary disaggregated prefill/decode story."),
        ("`baseline_prefill_decode_disagg_overlap_multigpu.py`, `optimized_prefill_decode_disagg_overlap_multigpu.py`, `baseline_prefill_decode_disagg_batched_multigpu.py`, `optimized_prefill_decode_disagg_batched_multigpu.py`, `baseline_prefill_decode_disagg_ttft_multigpu.py`, `optimized_prefill_decode_disagg_ttft_multigpu.py`, `baseline_prefill_decode_disagg_tpot_long_multigpu.py`, `optimized_prefill_decode_disagg_tpot_long_multigpu.py`", "Chapter-native end-to-end inference flows modeling separate prefill and decode pools, including overlap-focused, batched-handoff, TTFT-focused, and long-output TPOT-focused multi-GPU pairs."),
        ("`baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py`", "Pipeline parallel workloads combining compute and KV-transfer scheduling."),
        ("`baseline_moe_router_uniform.py`, `optimized_moe_router_uniform_topology.py`", "Comparable MoE router benchmark pair contrasting uniform vs topology-aware routing while keeping outputs invariant via shared expert weights."),
        ("`baseline_moe_router_local_capacity.py`, `optimized_moe_router_local_capacity.py`", "MoE capacity-reservation pair contrasting topology-aware overflow spill with all-local expert placement."),
        ("`moe_router_uniform_demo.py`, `moe_router_topology_demo.py`", "MoE routing demos (non-benchmark) contrasting uniform vs topology-aware expert selection."),
        ("`baseline_routing_static.py`, `optimized_routing_static.py`", "Router variants for static/dynamic sharding decisions (comparable benchmarks)."),
        ("`baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py`", "Memory-bound case studies plus profiling guides tailored to routing workloads (use `aisp tools roofline` for roofline analysis)."),
        ("`compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `dynamo_config.yaml`", "Harness entry, build rules, expectation baselines, and Dynamo config knobs."),
    ],
    validation=[
        "`python optimized_dynamic_routing.py --trace` logs TTFT/TPOT trends that settle faster than the baseline's oscillations.",
        "`python optimized_pipeline_parallelism.py --profile minimal` shows overlapping prefill/decode segments with fewer idle bubbles.",
        "`python -m cli.aisp tools roofline` reproduces the documented roofline points using your latest captures.",
    ],
    notes=[
        "`blackwell_profiling_guide.py` walks through Nsight Systems/Compute captures and interpreting roofline vs occupancy bottlenecks for routing-heavy workloads.",
        "`baseline_prefill_decode_disagg_overlap_multigpu.py` and `baseline_prefill_decode_disagg_batched_multigpu.py` run via torchrun and default to a 50/50 split when world size is even; override with `--prefill-ranks` (e.g., 2P1D). Use `torchrun --nproc_per_node` to choose the GPU count.",
        "The disaggregated prefill/decode baselines use per-request blocking handoff with per-request sync/barrier to model naive scheduling; optimized counterparts batch per group or send contiguous KV/seed slabs to overlap or boost throughput.",
    ],
)

ENTRIES["ch18"] = chapter_entry(
    slug="ch18",
    title="Chapter 18 - Advanced Attention & Decoding",
    summary=dedent(
        """\
        Collects modern decoder techniques-FlexAttention, FlexDecoding, speculative and paged attention workflows-implemented in both PyTorch and CUDA/Triton so you can iterate quickly while validating kernels on real hardware."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 18 is the "does decoder complexity actually buy you anything?" checkpoint. It puts flexible masking, speculative decoding, tensor-core kernels, and serving integration on the same chapter surface so you can see which tricks reduce latency and which ones only add engineering cost."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - straightforward FlexAttention / decode execution
                - conservative serving integration without aggressive caching or graph replay
                - good correctness anchor, but usually too much launch and data-movement overhead"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - FlexDecoding, tensor-core-specialized kernels, and cache-aware paths
                - graph replay and serving-integrated decode paths where they help
                - still benchmarked through the shared harness instead of one-off scripts"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `flexdecoding` | `161.596 ms` | `81.980 ms` | `1.97x` | baseline masks the full KV cache; optimized slices to the active decode window |
                | `eos_early_exit` | `11.694 ms` | `2.481 ms` | `4.71x` | optimized stops after 16/128 decode steps once all rows are EOS |
                | `tensor_cores` | `3.805 ms` | `0.243 ms` | `15.65x` | tensor-core decode kernel |
                | `rope_q_cache` | `106.429 ms` | `4.523 ms` | `23.53x` | cache-aware rope/Q-path reuse |

                The `eos_early_exit` row is from `artifacts/runs/codex_ch18_eos_early_exit_light_20260630_214000/`. The chapter has a mix of "moderate but real" improvements and "big kernel-level" improvements. Treat those as different stories rather than averaging them together into one headline number. `flexdecoding` is the chapter-native work-reduction story: the baseline scores the full KV cache with a sliding-window mask, while the optimized path trims the decode step down to the active window before attention. Re-measure it on your hardware before treating the chapter numbers as a decision threshold."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want Nsight evidence for cache reuse, launch count, and kernel selection:

                ```bash
                python -m cli.aisp bench run --targets ch18:flexdecoding --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch18:eos_early_exit --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch18:tensor_cores --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch18:rope_q_cache --profile deep_dive --single-gpu
                ```

                For serving integration, use the chapter-specific vLLM path only after the direct benchmark targets are clean, because the chapter harness gives you the more trustworthy baseline/optimized comparison."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch18.compare
                python -m cli.aisp bench list-targets --chapter ch18
                python -m cli.aisp bench run --targets ch18 --profile minimal
                python -m cli.aisp bench run --targets ch18:flexdecoding --profile deep_dive --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "Prototype FlexAttention/FlexDecoding workloads with custom masks, score mods, and KV-cache integration.",
        "Evaluate speculative decoding pipelines that trade extra compute for lower latency.",
        "Test tensor-core optimized attention kernels tailored for Blackwell tmem limits.",
        "Validate integration points with serving frameworks (vLLM) using the provided runners.",
        "Separate EOS polling overhead from early-exit work reduction when a batch finishes before the max decode budget.",
    ],
    contents=[
        ("`baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_graphs.py`, `v1_engine_loop.py`, `v1_engine_loop_common.py`", "FlexDecoding benchmarks plus a V1 polling-loop correctness tool (not a benchmark pair)."),
        ("`baseline_eos_sync_polling.py`, `optimized_eos_sync_polling.py`, `baseline_eos_early_exit.py`, `optimized_eos_early_exit.py`, `eos_early_exit_common.py`", "EOS synchronization and early-exit decode-loop comparisons, including a lightweight fixed-buffer benchmark for skipped tail decode work."),
        ("`baseline_paged_attn_backend.py`, `optimized_paged_attn_backend.py`, `baseline_paged_attn_layout.py`, `optimized_paged_attn_layout.py`, `paged_attn_split_common.py`", "Split paged-attention comparisons: dense math-versus-flash backend selection and dense masked decode versus block-table-driven FlexAttention sparse kernels."),
        ("`baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py`", "Tensor-core attention kernels plus Triton equivalents for rapid validation."),
        ("`flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py`", "FlexAttention examples ranging from toy sizes to large models with KV-cache reuse."),
        ("`baseline_vllm_v1_integration.py`, `optimized_vllm_v1_integration.py`, `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py`, `configs/`, `spec_configs/`, `workload_config.py`", "Serving integrations and config presets for pushing workloads through vLLM or custom harnesses."),
        ("`speculative_decode/spec_config_sweep.py`", "Tooling to sweep speculative-decoding configs and summarize latency/throughput tradeoffs."),
        ("`compare.py`, `expectations_{hardware_key}.json`, `test_flex_attention.py`", "Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs."),
    ],
    validation=[
        "`python -m ch18.compare` runs the chapter baseline/optimized sweep through the shared harness.",
        "`python -m cli.aisp bench run --targets ch18:vllm_v1_integration --profile minimal` completes with accuracy parity vs the native FlexAttention path.",
        "`python -m pytest -q ch18/test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.",
        "`ch18:eos_early_exit` reports fewer decoded steps on the optimized path while preserving the generated-token verification buffer.",
    ],
    notes=[
        "`flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.",
        "`flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.",
        "`paged_attn_backend` isolates SDPA backend choice on a dense layout, while `paged_attn_layout` converts a real per-batch block table into both a dense reference mask and a fused FlexAttention block-mask kernel.",
    ],
)

ENTRIES["ch19"] = chapter_entry(
    slug="ch19",
    title="Chapter 19 - Dynamic & Adaptive Inference Precision/Memory Systems",
    summary=dedent(
        """\
        Explores dynamic precision, KV-cache quantization, memory double buffering, and adaptive allocators so inference-oriented low-precision experiments stay numerically safe while squeezing every byte of HBM."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 19 is where adaptive precision and memory-system ideas have to prove they are more than paper wins. The useful question is not "can we quantize or double-buffer this?" but "which runtime precision and memory changes improve the real workload enough to justify the added complexity?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - higher-cost cache, precision, and memory-management paths
                - simpler allocator and buffering behavior
                - cleaner as a reference, but often too expensive in memory traffic or precision budget"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - quantized caches, lower-precision training/inference paths, and explicit buffering improvements
                - adaptive allocator or overlap logic where memory behavior is the actual bottleneck
                - benchmarked through the same harness contract so the speedup claims remain comparable and verified"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from current expectation baselines and recent strict reruns:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `dynamic_quantized_cache` | `1.710 ms` | `1.517 ms` | `1.13x` | adaptive-bitwidth quantized refresh over the same full-cache footprint, with CPU-side verification output so the optimized path no longer pays a large GPU memory penalty |
                | `memory_double_buffering` | `5.536 ms` | `2.809 ms` | `1.97x` | double-buffered memory path |
                | `mxfp8_moe` | `16.037 ms` | `2.080 ms` | `7.71x` | lower-precision MoE path with materially better execution behavior |

                This chapter is where "low precision" should be read as a systems decision, not just a dtype choice. Some wins come from lower math cost, others from lower memory traffic or better overlap.

                `dynamic_quantized_cache` now uses the fair steady-state full-footprint refresh model introduced on `2026-03-17`. Repeated strict reruns on this virtualized host now land around `1.13-1.15x`, and the optimized path's GPU peak memory dropped from the earlier `~765 MB` down to `~269 MB`, below the baseline's `~404 MB`."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want to inspect whether the gain came from compute reduction, memory reduction, or allocator/buffering behavior:

                ```bash
                python -m cli.aisp bench run --targets ch19:dynamic_quantized_cache --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch19:memory_double_buffering --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch19:mxfp8_moe --profile deep_dive --single-gpu
                ```

                Those targets make good chapter probes because they cover cache behavior, memory overlap, and lower-precision MoE execution without collapsing everything into one synthetic headline."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch19.compare
                python -m cli.aisp bench list-targets --chapter ch19
                python -m cli.aisp bench run --targets ch19 --profile minimal
                python -m cli.aisp bench run --targets ch19:dynamic_precision --profile minimal --single-gpu
                python -m cli.aisp bench run --targets ch19:mxfp8_moe --profile deep_dive --single-gpu
                python -m cli.aisp tools ch19-adaptive-parallelism
                python -m cli.aisp tools ch19-dynamic-precision -- --help
                python -m cli.aisp tools ch19-dynamic-quantized-cache -- --help
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark FP4/FP6/FP8 training loops with calibration and validation hooks.",
        "Overlap KV-cache prefetch with compute while respecting precision constraints.",
        "Implement dynamic quantized caches that switch formats mid-run without drift.",
        "Design allocator helpers to monitor and rebalance fragmented memory pools.",
    ],
    contents=[
        ("`baseline_nvfp4_training.py`, `optimized_nvfp4_training.py`, `native_fp4_quantization.py`, `native_fp6_quantization.py`, `native_fp8_training.py`", "Training and quantization recipes that switch between FP8 and NVFP4 with automatic calibration."),
        ("`baseline_adaptive_parallelism.py`, `optimized_adaptive_parallelism.py`, `adaptive_parallelism_benchmark_common.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py`", "Chapter-native adaptive parallelism benchmark pair plus the routing helpers that model tensor/pipeline/hybrid/data worker-pool selection on the same synthetic request stream."),
        ("`baseline_dynamic_precision.py`, `optimized_dynamic_precision.py`, `dynamic_precision_benchmark_common.py`, `dynamic_precision_switching.py`, `token_precision_switching.py`", "Chapter-native dynamic precision benchmark pair plus the confidence-driven precision helpers that keep decode outputs stable while switching precision modes."),
        ("`baseline_memory_double_buffering.py`, `optimized_memory_double_buffering.py`, `memory_allocator_with_monitoring.py`, `dynamic_memory_allocator.py`, `_allocator_worker.py`", "Memory-management helpers covering double buffering, instrumentation, and adaptive worker pools."),
        ("`baseline_kv_prefetch_overlap.cu`, `optimized_kv_prefetch_overlap.cu`, `kv_prefetch_overlap_sm121` binaries", "CUDA kernels proving that quantized KV prefetch can overlap with compute when using cp.async pipelines."),
        ("`baseline_dynamic_quantized_cache.py`, `optimized_dynamic_quantized_cache.py`, `baseline_dynamic_quantized_cache_coalesced.py`, `optimized_dynamic_quantized_cache_coalesced.py`, `dynamic_quantized_cache.py`, `token_precision_switching.py`, `dynamic_precision_switching.py`", "Cache-refresh experiments comparing full-precision FP32 maintenance, adaptive-bitwidth quantized refresh, and coalesced repeated-width refresh on the same KV footprint."),
        ("`baseline_fp4_hardware_kernel.cu`, `optimized_fp4_hardware_kernel.cu`, `fp8_hardware_kernel.cu`, `custom_allocator_retry.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py`", "Hardware-level kernels and adaptive scheduling helpers for heterogeneous precision fleets."),
        ("`compare.py`, `arch_config.py`, `expectations_{hardware_key}.json`", "Harness entry, architecture toggles, and stored expectation data."),
    ],
    validation=[
        "`python -m ch19.compare` runs the chapter baseline/optimized sweep through the shared harness.",
        "`python -m cli.aisp bench run --targets ch19:adaptive_parallelism --profile minimal` keeps the baseline Python routing loop and the optimized vectorized routing path output-equivalent on the same request stream.",
        "`python -m cli.aisp bench run --targets ch19:dynamic_precision --profile minimal --single-gpu` keeps the fixed-precision and dynamic-precision decode outputs token-equivalent on the same prompt stream.",
        "`python -m cli.aisp bench run --targets ch19:dynamic_quantized_cache --profile minimal` validates the adaptive-bitwidth quantized refresh against the same full-cache FP32 baseline while tracking bounded error.",
        "`nvcc -o optimized_kv_prefetch_overlap_sm121 optimized_kv_prefetch_overlap.cu` plus the baseline binary show measurable overlap improvements in Nsight Compute.",
    ],
    notes=[
        "`arch_config.py` exposes `ENABLE_NVFP4`/`ENABLE_TF32` toggles per device, making it easy to compare precision recipes.",
        "`validate_quantization_performance.py` aggregates accuracy vs throughput numbers into CSV form for proof-of-benefit reporting.",
    ],
)

ENTRIES["ch20"] = chapter_entry(
    slug="ch20",
    title="Chapter 20 - AI-Assisted Performance Optimization & Case Studies",
    summary=dedent(
        """\
        Combines AI-assisted kernel exploration with end-to-end case studies: prototype or verify generated kernels, then test whether memory, pipeline, and inference optimizations still hold up once they are composed into a full workload."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Chapter 20 is where AI-generated ideas and isolated wins have to survive contact with the full stack. The useful question is not only "did one optimization help in isolation?" but also "can we validate generated or staged optimizations once memory, pipeline, and inference changes are stacked together in one end-to-end workload?" """
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - sequential or minimally optimized end-to-end execution
                - independent subsystems with little cross-stage coordination
                - useful as a proof baseline, but usually leaves bandwidth and overlap on the table"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - staged pipeline, memory, and KV-cache optimizations combined into one workload
                - the same harness contract as every other chapter, so the end-to-end gains stay comparable to the lower-level chapters
                - better for answering whether the optimizations compose cleanly instead of fighting each other"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `integrated_kv_cache` | `456.705 ms` | `67.381 ms` | `6.78x` | integrated KV-cache and overlap path |
                | `bf16_mlp` | `0.616 ms` | `0.234 ms` | `2.63x` | BF16 precision policy on the same eager MLP graph |

                This chapter is the best place to check whether wins compose. `pipeline_sequential` now remains available as an informational overlap demo, while canonical chapter claims focus on the pairs that still hold up as end-to-end improvements. The AI-assisted kernel-generation thread is represented here by `ai_kernel_generator.py` plus the verifier helpers, even though the full manuscript chapter covers a broader RL/AlphaTensor narrative than the current harness surface."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive harness runs when you want to see how the end-to-end gain breaks down by subsystem:

                ```bash
                python -m cli.aisp bench run --targets ch20:integrated_kv_cache --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets ch20:bf16_mlp --profile deep_dive --single-gpu
                ```

                That is the right place to answer whether the gain came from overlap, memory movement, or simply removing one obvious bottleneck from the baseline."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m ch20.compare
                python -m cli.aisp bench list-targets --chapter ch20
                python -m cli.aisp bench run --targets ch20 --profile minimal
                python -m cli.aisp tools ch20-ai-kernel-generator -- --device cpu --seqlen 512
                python -m cli.aisp tools ch20-ai-kernel-workflow -- --device cpu --seqlen 512
                ```"""
            ),
        ),
    ],
    goals=[
        "Chain memory, pipeline, and KV-cache optimizations together to see cumulative impact.",
        "Generate automatic reports that compare baseline vs tuned end-to-end runs.",
        "Prototype new kernels via the AI kernel generator and slot them into the harness.",
        "Validate improvements with workload-specific acceptance tests.",
    ],
    contents=[
        ("`baseline_bf16_mlp.py`, `optimized_bf16_mlp.py`, `ai_kernel_generator.py`, `core/optimization/inductor_guard.py`", "Precision-policy workload plus the shared Inductor cudagraph guard used by the compiled end-to-end paths."),
        ("`baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py`", "Pipeline and bandwidth case studies showing how optimizations interact across stages. `pipeline_sequential` currently remains informational after the fairness refresh."),
        ("`baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py`", "Integrated KV-cache demos that merge allocator, overlap, and NVLink pooling tricks."),
        ("`baseline_memory_standard.py`, `optimized_memory_standard.py`", "Memory-focused harness verifying allocator changes at system level."),
        ("`baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile`", "Single-device training case study plus CUDA kernels used in the final report."),
        ("`compare.py`, `arch_config.py`, `expectations_{hardware_key}.json`", "Harness driver, architecture settings, and expectation baselines."),
    ],
    validation=[
        "`python -m ch20.compare` emits per-stage summaries that show each optimized variant meeting or exceeding stored expectations.",
        "`python -m ch20.ai_kernel_generator --emit test.cu` produces CUDA kernels that compile via `nvcc` and integrate into the harness without manual edits.",
        "`python -m cli.aisp bench run --targets ch20:integrated_kv_cache --profile deep_dive` is the stronger canonical end-to-end overlap proof after the fairness refresh.",
    ],
    notes=[
        "`core/optimization/inductor_guard.py` is the canonical helper for gating Inductor cudagraph features in the compiled chapter 20 paths.",
        "`ai_kernel_generator.py` logs generated code to `artifacts/` for reproducibility; capture the log with your proof-of-benefit bundle.",
    ],
)

ENTRIES["labs/blackwell_matmul"] = lab_entry(
    slug="labs/blackwell_matmul",
    title="Lab - Blackwell Matmul Suite",
    summary=dedent(
        """\
        Ports the four-part Blackwell matmul deep dive into the harness: start with a naive CUDA kernel, then layer pipeline loads, real TMA, and cluster DSMEM broadcasts until you surpass the baseline roofline."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                This lab exists to answer a very specific Blackwell question: which part of the matmul stack is actually buying the win on this machine? Pipeline staging, TMA, and cluster/DSMEM support do not always move together, so the lab keeps them as separate benchmark targets instead of hiding everything behind one "optimized" label."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - naive CUDA matmul kernel
                - no TMA or DSMEM cluster help
                - useful roofline reference, but not a realistic Blackwell schedule"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - pipelined staging path
                - TMA-enabled path for lower copy/staging overhead
                - cluster/DSMEM variants when the hardware and shape make them worthwhile"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260301_1032__bench__profile_none_targets_labs26_recheck/`:

                | Target | Baseline | Optimized | Measured delta | What changed |
                | --- | ---: | ---: | ---: | --- |
                | `blackwell_matmul_pipeline` | `29.254 ms` | `5.045 ms` | `5.80x` | pipeline staging only |
                | `blackwell_matmul_tma` | `29.303 ms` | `4.373 ms` | `6.70x` | TMA staging path |
                | `blackwell_matmul_cluster` | `29.259 ms` | `16.307 ms` | `1.79x` | cluster/DSMEM path |

                The useful reading is that the current local winner is the TMA path, not the cluster path. The cluster target is still valuable because it keeps the DSMEM route benchmarked and verified, but this repo does not pretend it is the latency leader on every shape."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want Nsight evidence for each schedule family:

                ```bash
                python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_pipeline --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_tma --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile deep_dive --single-gpu
                ```

                Keep the targets separate when you analyze them. The point of this lab is to attribute the gain to the schedule family, not to blur TMA and cluster behavior together."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/blackwell_matmul
                python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_tma --profile minimal
                python labs/blackwell_matmul/run_blackwell_matmul.py --variant tma --size 4096
                ```"""
            ),
        ),
    ],
    goals=[
        "Reproduce the reference matmul trajectory (baseline -> pipelined -> TMA -> cluster).",
        "Compare PyTorch harness timings against the CUDA extensions while reusing the same shapes.",
        "Validate kernels on SM100/103 targets and gracefully skip DSMEM-only paths on SM121.",
        "Capture dual roofline metadata (SM vs TMEM) for every variant.",
    ],
    contents=[
        ("`baseline_blackwell_matmul.py`, `optimized_blackwell_matmul_pipeline.py`, `optimized_blackwell_matmul_tma.py`, `optimized_blackwell_matmul_cluster.py`", "Python entrypoints for each stage of the matmul tutorial."),
        ("`blackwell_benchmarks.py`, `run_blackwell_matmul.py`", "Harness adapters and standalone runner for quick sweeps and metadata capture."),
        ("`grace_blackwell_extension.py`, `grace_blackwell_kernels.cu`", "PyTorch extension and CUDA kernels implementing the baseline and optimized kernels."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile minimal` delivers higher TFLOP/s than the baseline and emits artifacts under `artifacts/labs_blackwell_matmul*`.",
        "`python labs/blackwell_matmul/run_blackwell_matmul.py --variant pipeline --size 4096 --roofline-meta artifacts/labs_blackwell_matmul/matmul_meta.csv` saves roofline metadata alongside timings.",
        "DSM-aware variants error out early on GPUs that lack cluster DSMEM support, preventing misleading results.",
    ],
    notes=[
        "`run_blackwell_matmul.py` accepts `--variant baseline|pipeline|tma|cluster` plus `--size` to mirror the blog walkthrough.",
        "TMA kernels require CUDA 13.0+ and SM100/103 hardware; on GB10 they log a warning and skip execution.",
    ],
)

ENTRIES["labs/blackwell_gemm_optimizations"] = lab_entry(
    slug="labs/blackwell_gemm_optimizations",
    title="Lab - Blackwell Grouped GEMM Optimizations",
    summary=dedent(
        """\
        Implements the grouped/MoE GEMM optimization journey from the Blackwell slide deck inside the harness: start with a small-tile grouped kernel, then move to larger tiles, fused route-weight application, `GROUP_SIZE_M` swizzling, autotune, and a persistent tile schedule."""
    ),
    lead_sections=[
        MarkdownSection(
            "Targets",
            dedent(
                """\
                | Target | Stage |
                | --- | --- |
                | `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm` | canonical baseline plus all optimized siblings |
                | `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles` | Step 1: `128x128x64` tiles |
                | `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack` | Step 3: fused route weights + `GROUP_SIZE_M` + autotune |
                | `labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent` | Step 4: persistent strided tile loop |"""
            ),
        ),
        MarkdownSection(
            "Problem",
            dedent(
                """\
                The source deck is not a plain dense GEMM story. It is a routed/grouped GEMM story where the useful question is: once tokens are bucketed per expert, which schedule choices actually move grouped expert throughput on Blackwell?

                This lab keeps that shape honest:
                - tokens are packed into per-expert buckets
                - route weights stay visible in the workload
                - the public targets map to the real optimization stages instead of collapsing everything into one opaque "optimized" kernel"""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - `64x64x32` tiles
                - one grouped tile launch model without persistent residency
                - route weights applied after the grouped GEMM instead of inside the kernel
                - useful control for launch/schedule overhead, but not a realistic Blackwell endpoint"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - `large_tiles`: `128x128x64` grouped kernel to improve tensor-core utilization
                - `full_stack`: fused route-weight multiply, `GROUP_SIZE_M` swizzle, and autotuned schedule selection
                - `persistent`: full-stack kernel with a strided tile loop over the total grouped tile space"""
            ),
        ),
        MarkdownSection(
            "Negative Controls",
            dedent(
                """\
                The deck also includes changes that are useful to measure but should not become public default targets:

                - `fast_math_control`: represented as a control only, because the Triton path has no separate user-visible fast-math toggle
                - `latency10`: deeper buffering / stage count increase
                - `two_cta`: extra resident warp pressure without a stable win
                - `tile_n256`: larger N tile that increases working-set pressure

                Use `python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --include-experimental` when you want those controls, but keep the public harness surface on the three real milestones."""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current implementation-state measurements from March 19, 2026 on the local virtualized B200 host:

                | Measurement | Shape / Artifact | Result | What it proves |
                | --- | --- | --- | --- |
                | Direct matrix smoke (`artifacts/blackwell_grouped_gemm_smoke_matrix.csv`) | `tokens=128, hidden=256, ffn=512` | `baseline=0.206 ms`, `large_tiles=0.158 ms`, `full_stack=0.176 ms`, `persistent=0.165 ms` | all three public optimized stages beat the baseline on a tiny grouped workload, and the negative controls remain runnable for comparison |
                | Harness pair (`artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`) | `tokens=512, hidden=512, ffn=1024` | `baseline=0.132 ms`, `persistent=0.131 ms`, `1.01x` | the public persistent pair already runs with locked clocks, verification, and Nsight traces |

                These numbers are directional, not canonical: the current host is virtualized, and the smoke matrix is intentionally small so implementation bugs show up quickly. Treat the tiny-shape ranking as a functional smoke check, not as the final promotion signal. The important point is that the grouped schedule transitions are now reproducible inside the repo instead of living only in screenshots."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want Nsight evidence for the grouped schedule transitions:

                ```bash
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile deep_dive --single-gpu --gpu-sm-clock-mhz 1500
                ```

                The implementation-state persistent pair already has `nsys`, `ncu`, and PyTorch traces in `artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_large_tiles --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
                python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500
                python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 1024 2048 4096 8192 --include-experimental
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep the Blackwell deck honest by exposing grouped/MoE GEMM rather than pretending the story is dense matmul.",
        "Show which gains come from larger tiles versus fusion/swizzle versus persistent residency.",
        "Preserve the negative controls as runnable experiments without promoting them to public benchmark targets.",
        "Give the repo a Blackwell-gated grouped GEMM lab that can be profiled, verified, and extended with future kernel work.",
    ],
    contents=[
        ("`baseline_blackwell_grouped_gemm.py`, `optimized_blackwell_grouped_gemm_large_tiles.py`, `optimized_blackwell_grouped_gemm_full_stack.py`, `optimized_blackwell_grouped_gemm_persistent.py`", "Public harness wrappers for the baseline and three optimization stages."),
        ("`blackwell_grouped_gemm_common.py`", "Deterministic grouped-workload builder, BF16/FP16 reference path, benchmark class, and verification logic."),
        ("`blackwell_grouped_gemm_kernel.py`, `blackwell_grouped_gemm_autotune.py`", "Triton kernel launches, schedule registry, autotune candidates, and experimental controls."),
        ("`compare_blackwell_grouped_gemm_matrix.py`", "Direct token-count sweep runner for smoke checks and negative-control comparisons outside the harness."),
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations",
            "python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_full_stack --profile minimal --single-gpu --gpu-sm-clock-mhz 1500",
            "python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500",
            "python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 1024 2048 4096 8192 --include-experimental",
        ],
        notes=[
            "Use `--target-extra-arg labs/blackwell_gemm_optimizations:<workload>=\"--num-tokens N --hidden-dim K --expert-ffn-dim N\"` to sweep grouped-token shapes through the harness.",
            "Launch the direct runner as a module (`python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix`), not as a bare script path.",
            "The direct runner is a smoke/perf helper. Use the harnessed `bench run` targets for clock-locked pair results and profiler artifacts.",
        ],
    ),
    validation=[
        "`python -m cli.aisp bench list-targets --chapter labs/blackwell_gemm_optimizations` exposes the canonical baseline target plus the `large_tiles`, `full_stack`, and `persistent` aliases.",
        "`python -m cli.aisp bench run --targets labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent --profile minimal --single-gpu --gpu-sm-clock-mhz 1500 --iterations 2 --warmup 1 --target-extra-arg 'labs/blackwell_gemm_optimizations:blackwell_grouped_gemm_persistent=--num-tokens 512 --hidden-dim 512 --expert-ffn-dim 1024'` succeeds with verification and produces `nsys`, `ncu`, and PyTorch traces under `artifacts/runs/20260319_015359__bench__profile_minimal_targets_labs_blackwell_gemm_optimizations_blackwell_grouped_gemm/`.",
        "`python -m labs.blackwell_gemm_optimizations.compare_blackwell_grouped_gemm_matrix --token-counts 128 --num-experts 4 --hidden-dim 256 --expert-ffn-dim 512 --warmup 1 --repeats 2` shows the expected stage-wise speedup trend on the implementation smoke shape.",
        "`pytest tests/test_blackwell_gemm_optimizations_lab.py` validates discovery, wrapper exposure, and per-variant CUDA parity on a small grouped workload.",
    ],
    notes=[
        "This lab is intentionally grouped/MoE-flavored GEMM. The directory name keeps the user-facing Blackwell GEMM framing, but the benchmark targets stay explicit about grouped execution.",
        "The slide-deck Step 2 fast-math idea is preserved only as a control because the Triton path does not expose an equivalent standalone toggle.",
    ],
)

ENTRIES["labs/cutlass_profiler_kernel_selector"] = lab_entry(
    slug="labs/cutlass_profiler_kernel_selector",
    title="Lab - CUTLASS Profiler Kernel Selector",
    summary=dedent(
        """\
        Automates CUTLASS profiler sweeps for transformer-style GEMMs, records Triton or custom kernel results, and compares everything so you can prove custom kernels beat the fastest stock CUTLASS option."""
    ),
    goals=[
        "Generate per-shape CUTLASS profiler logs and store the best kernel metadata.",
        "Optionally benchmark Triton or custom paths on the same shapes.",
        "Compare providers (CUTLASS, Triton, DeepEP, custom) with a uniform JSON schema.",
        "Adjust shapes quickly by editing a single definition file.",
    ],
    contents=[
        ("`run_cutlass_profiler_sweep.py`", "Invokes `cutlass_profiler` for every shape in `shapes.py` and stores JSON summaries."),
        ("`run_triton_matmul.py`", "Optional Triton matmul runner for parity checks."),
        ("`compare_against_baselines.py`", "Reads CUTLASS + competitor JSON files and emits TFLOP/s + speedup tables."),
        ("`shapes.py`", "Central list of GEMM shapes (prefill, decode, KV proj, etc.)."),
    ],
    run=RunSection(
        commands=[
            "cd ai-performance-engineering",
            "python labs/cutlass_profiler_kernel_selector/run_cutlass_profiler_sweep.py --output-dir artifacts/cutlass_profiler",
            "python labs/cutlass_profiler_kernel_selector/run_triton_matmul.py --output-dir artifacts/cutlass_profiler",
            "python labs/cutlass_profiler_kernel_selector/compare_against_baselines.py --include-default-triton",
        ],
        notes=[
            "Set `CUTLASS_PROFILER_BIN` to point at your `cutlass_profiler` binary after running `setup.sh` from the repo root.",
            "Add extra providers by writing JSON files matching the documented schema (see `compare_against_baselines.py`).",
        ],
    ),
    validation=[
        "Profiler runs emit `artifacts/cutlass_profiler/cutlass_profiler_results.json` with per-shape winners; rerun when upgrading CUDA or GPUs.",
        "Triton baselines land in `artifacts/cutlass_profiler/triton_matmul_results.json` and should stay within a few percent of CUTLASS for supported shapes.",
        "`compare_against_baselines.py` exits non-zero when provided result files are missing records, ensuring CI catches stale outputs.",
    ],
    notes=[
        "Shapes can be overridden via CLI flags (e.g., `--shapes decode_mlp_m4096_n4096_k8192`).",
        "Provider JSON files may include metadata (kernel names, launch params) for additional debugging.",
    ],
)

ENTRIES["labs/cudnn_sdpa_bench"] = lab_entry(
    slug="labs/cudnn_sdpa_bench",
    title="Lab - cuDNN SDPA Bench",
    summary=dedent(
        """\
        Microbenchmarks cuDNN fused scaled-dot-product attention against Flash and math backends with explicit CLI backend selection."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Attention backend choices are often treated as an implementation detail. This lab exists to keep that choice explicit and benchmarked so you can tell whether cuDNN, Flash, or the math path is actually the right answer for this exact shape family."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - attention path with conservative backend selection
                - stable reference for correctness and shape coverage
                - useful when fused paths are unavailable or unstable"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - fused SDPA backend path
                - same shapes and validation contract
                - tuned to answer "does backend choice alone move the result?" rather than mixing in unrelated changes"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `flash_sdp` | `0.345 ms` | `0.282 ms` | `1.22x` |

                This is not a giant benchmark pair, and that is useful. The lab exists to show a real backend-selection delta without pretending it is a bigger architectural win than it is."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"
                python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"
                ```

                Keep the backend fixed per run when you profile. The point is to attribute the gain to backend behavior, not to mixed runtime heuristics."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
                python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal
                python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"
                ```"""
            ),
        ),
    ],
    goals=[
        "Compare cuDNN fused SDPA to Flash and math backends on identical shapes.",
        "Capture Nsight traces per backend to inspect kernel fusion and launch counts.",
        "Keep regression thresholds per architecture in `expectations_{hardware_key}.json`.",
    ],
    contents=[
        ("`baseline_flash_sdp.py`, `optimized_flash_sdp.py`", "Shared attention microbenchmarks; backend chosen via `--backend {auto,cudnn,flash,math}` passed with `--target-extra-arg`."),
        ("`expectations_{hardware_key}.json`", "Current golden timings for the active hardware key."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend cudnn\"` captures cuDNN with Nsight traces.",
        "`python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend flash\"` compares the Flash path against cuDNN.",
        "`python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend math\"` sanity-checks the math backend where fused kernels are unsupported.",
    ],
    notes=[
        "Backend selection is CLI-only; environment variables are intentionally ignored.",
        "Profiling outputs are stored under `artifacts/runs/<run_id>/profiles/bench/labs_cudnn_sdpa_bench/` with harness artifacts in `artifacts/runs/<run_id>/`.",
    ],
)

ENTRIES["labs/dynamic_router"] = lab_entry(
    slug="labs/dynamic_router",
    title="Lab - Dynamic Prefill/Decode Router",
    summary=dedent(
        """\
        Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it."""
    ),
    goals=[
        "Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.",
        "Prototype migration budgets, KV-locality boosts, and per-pool thresholds.",
        "Drive the router against synthetic workloads or real vLLM engines.",
        "Export detailed metrics (TTFT, TPOT, queue depth) for visualization.",
    ],
    contents=[
        ("`router_round_robin.py`, `router_policy.py`, `driver.py`, `eval_stack.py`", "Core router logic plus a synthetic simulator for deterministic comparisons."),
        ("`baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py`", "Integrations for running the routing policy against vLLM instances."),
        ("`baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py`", "Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`."),
        ("`topology.py`, `topology_probe.py`", "NUMA-aware GPU mapping helpers and a target that emits topology JSON under `artifacts/topology/` for routing hints."),
    ],
    validation=[
        "`python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.",
        "`python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs.",
        "`python -m cli.aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm=\"--model /path/to/model --decode-gpus 0,1\"` succeeds on hosts with at least two GPUs and a local model copy.",
        "`python -m cli.aisp bench run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm=\"--model /path/to/model --prefill-gpus 0 --decode-gpus 1\"` contrasts shared versus dual pools and emits per-pool TTFT and queue depth.",
        "`python -m cli.aisp bench run --targets labs/dynamic_router:topology_probe` captures GPU↔NUMA mappings and distance matrices for consumption by the router.",
    ],
    notes=[
        "`driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.",
        "vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) plus locally available tokenizer/model weights.",
        "Router scoring incorporates pinned-host KV slab availability and NUMA-locality bias; feed it real topology via `topology_probe.py` or NVML when available.",
    ],
)

ENTRIES["labs/cache_aware_disagg_inference"] = lab_entry(
    slug="labs/cache_aware_disagg_inference",
    title="Lab - Cache-Aware Disaggregated Inference",
    summary=dedent(
        """\
        Recreates the article-level scheduler story behind cache-aware prefill/decode disaggregation: a cache-unaware round-robin baseline versus cache-affine decode placement with a shared KV hierarchy and a warm/cold request mix."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Naive prefill/decode disaggregation makes the topology look correct while still wasting locality. If chunk handoff bounces each request across decode workers, warm prefixes stop being warm and KV reload traffic overwhelms the supposed benefit of disaggregation."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - round-robin logical decode placement
                - shared-prefix reloads whenever chunk ownership changes
                - a useful control for showing how temporal locality gets destroyed"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - cache-affine worker assignment
                - warm prefixes stay resident on the same logical decode worker
                - the benchmark reports cache hit rate, KV transfer volume, worker switches, and TTFT/TPOT so the win is explained, not just timed"""
            ),
        ),
    ],
    goals=[
        "Compare cache-unaware round-robin handoff against cache-aware decode affinity.",
        "Make temporal and spatial locality visible through custom metrics rather than narrative alone.",
        "Keep the lab runnable on one GPU by simulating logical workers instead of requiring a full cluster.",
    ],
    contents=[
        ("`baseline_cache_aware_disagg.py`, `optimized_cache_aware_disagg.py`, `cache_aware_disagg_common.py`", "Single-GPU logical-worker benchmark pair plus the shared scheduler/cache model that reproduces the article's core behavior."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference --profile minimal` compares the cache-unaware and cache-aware paths through the standard harness.",
        "`python -m cli.aisp bench run --targets labs/cache_aware_disagg_inference:cache_aware_disagg --profile minimal` emits the round-robin baseline and cache-affine optimized metrics through one harness artifact set.",
        "The cache-aware path should report lower KV transfer volume and fewer worker switches than the round-robin baseline on the same warm/cold request mix.",
    ],
    notes=[
        "This lab is intentionally a logical reproduction of the scheduler/caching story, not a full serving engine.",
        "Treat single-GPU `cache_aware_disagg` as a locality-comparison benchmark with a local comparison contract. The stable value on one GPU is the cache hit rate, KV transfer volume, and worker affinity improvement; the timed delta is recorded, but it is not a trustworthy headline speed gate on this host.",
        "Judge the single-GPU target by cache hit rate, KV transfer volume, and worker affinity before raw wall-clock speedup.",
        "The defaults model chunked prefill, warm/cold requests, and a 2P1D-style comparison problem without forcing an 8-GPU host.",
    ],
)

ENTRIES["labs/nccl_nixl_nvshmem"] = lab_entry(
    slug="labs/nccl_nixl_nvshmem",
    title="Lab - NCCL, NIXL, and NVSHMEM",
    summary=dedent(
        """\
        Explores communication-stack tradeoffs without pretending a `1x B200` host can exercise every path in the GTC deck. The lab exposes one honest benchmark pair for the decision we can measure locally, then pairs it with a stack probe runner that tells you when NCCL collectives, NIXL, or NVSHMEM are actually available."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                NCCL, NVSHMEM, and NIXL solve different communication problems. The practical mistake is to treat them as interchangeable before checking runtime and topology constraints. This lab keeps that straight by benchmarking the host-staged versus packed async tier-handoff decision we can run locally while making the unavailable paths explicit."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - CPU-staged, block-by-block handoff
                - per-block D2H and H2D copies with explicit synchronizations
                - models the control path that appears when movement and metadata bounce through the host"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - packed async handoff over the same selected blocks
                - coalesces noncontiguous blocks into one packed transfer plan
                - uses a dedicated copy stream and one bulk roundtrip instead of per-block synchronization"""
            ),
        ),
        MarkdownSection(
            "Stack Mapping",
            dedent(
                """\
                | Deck theme | What this lab does |
                | --- | --- |
                | NCCL symmetric memory and copy-engine collectives | Probes whether NCCL collectives are runnable on this host and records the repo-local `all_reduce_perf` binary path. True collective benchmarking still requires `>=2` GPUs. |
                | NVSHMEM GPU-initiated one-sided communication | Probes PyTorch symmetric-memory availability plus NVSHMEM launcher/runtime presence. On the current host, the probe finds the versioned launcher install but still stays blocked instead of fabricating a one-sided result. |
                | NIXL async, noncontiguous movement across tiers | The `tier_handoff` benchmark pair is the local analogue: scattered GPU blocks, CPU tier roundtrip, and the difference between host-staged control flow and packed async transport. |"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative portable/shared-host result from `artifacts/review/labs/nccl_nixl_nvshmem_portable_shared_v3/20260320_041146__bench__profile_none_targets_labs_nccl_nixl_nvshmem_tier_handoff/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `tier_handoff` | `17.07 ms` | `1.12 ms` | `15.28x` |

                This is intentionally non-canonical: the host is virtualized and the successful dogfood run used `--validity-profile portable --allow-foreign-gpu-processes`. The lab keeps that provenance visible instead of pretending it is publish-grade NCCL or NVSHMEM evidence."""
            ),
        ),
        MarkdownSection(
            "Related Work",
            dedent(
                """\
                This lab sits between the low-level communication primitives in `ch04` and the higher-level workflow labs:

                - `ch04` covers raw NCCL, symmetric-memory, and NVSHMEM building blocks.
                - `labs/train_distributed` and `labs/fullstack_cluster` use communication as part of broader training or cluster stories.
                - `labs/cache_aware_disagg_inference` and `labs/dynamic_router` use communication to support serving and routing policies.

                The unique focus here is the communication-stack decision itself: what you can measure locally, what is blocked by runtime or topology, and how to tell those apart honestly."""
            ),
        ),
        MarkdownSection(
            "Local Reality",
            dedent(
                """\
                | Probe field | Local outcome |
                | --- | --- |
                | GPU | `1x NVIDIA B200` |
                | `torch.distributed` NCCL | available |
                | `all_reduce_perf` | available at `tools/nccl-tests/build/all_reduce_perf` |
                | `nixl` import | available |
                | `nvshmem` import | missing |
                | NVSHMEM launcher | available at `/usr/bin/nvshmem_13/nvshmrun` |
                | PyTorch symmetric memory | available |
                | Multi-GPU NCCL / NVSHMEM validation | blocked by topology (`>=2` GPUs required) |"""
            ),
        ),
    ],
    goals=[
        "Separate host-staged control flow from packed async transport planning on a deterministic workload.",
        "Connect the deck's NIXL memory-tier story to a runnable single-GPU analogue.",
        "Make NCCL, NVSHMEM, and symmetric-memory availability visible before attempting the wrong experiment.",
        "Keep local dogfood results honest when the host cannot provide a clean multi-GPU transport path.",
    ],
    contents=[
        ("`baseline_tier_handoff.py`, `optimized_tier_handoff.py`", "Auto-discoverable benchmark pair for the local host-staged vs packed async handoff comparison."),
        ("`comm_stack_common.py`", "Shared workload, runtime probe, and benchmark implementation for the pair."),
        ("`run_lab_nccl_nixl_nvshmem.py`", "Standalone probe, direct compare, and sweep entrypoint."),
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem",
            "python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode probe --json",
            "python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode compare --warmup 3 --iterations 8 --selected-blocks 96 --block-kib 64 --inner-iterations 4",
            "python -m cli.aisp bench run --targets labs/nccl_nixl_nvshmem:tier_handoff --profile none --validity-profile portable --allow-foreign-gpu-processes --target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff='--selected-blocks 96 --block-kib 64 --inner-iterations 4'",
        ],
        notes=[
            "Targets follow the `labs/nccl_nixl_nvshmem:<workload>` naming convention listed by `list-targets`.",
            "Use `--target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff=\"--flag value\"` to sweep block counts or block sizes through the harness.",
            "On clean bare-metal single-GPU hosts, the harness target should run in strict mode.",
            "On the current shared virtualized host, the successful dogfood run required `--validity-profile portable --allow-foreign-gpu-processes`.",
        ],
    ),
    validation=[
        "`python -m py_compile labs/nccl_nixl_nvshmem/*.py`",
        "`python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode probe --json`",
        "`python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode compare --warmup 3 --iterations 8 --selected-blocks 96 --block-kib 64 --inner-iterations 4 --json`",
        "`python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem`",
        "`python -m cli.aisp bench run --targets labs/nccl_nixl_nvshmem:tier_handoff --profile none --validity-profile portable --allow-foreign-gpu-processes --target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff='--selected-blocks 96 --block-kib 64 --inner-iterations 4' --artifacts-dir artifacts/review/labs/nccl_nixl_nvshmem_portable_shared_v3`",
    ],
    notes=[
        "The strict harness path was attempted first and failed for a real reason: a separate isolated benchmark runner already owned GPU memory on the current host. The lab keeps that provenance visible instead of rewriting history.",
        "No expectation file is checked in yet. The current verified data point is portable and shared-host only, so it is useful for dogfood and documentation, not for a canonical baseline refresh.",
        "If a future pass adds `>=2` GPUs plus NIXL or NVSHMEM runtime support, keep the probe behavior intact and add new benchmark targets only when those paths are actually runnable and verifiable.",
    ],
)

ENTRIES["labs/decode_optimization"] = lab_entry(
    slug="labs/decode_optimization",
    title="Lab - Decode Optimization",
    summary=dedent(
        """\
        Decode-focused microbenchmarks that isolate serving-side wins such as pinned memory, streams, compile/graphs, FP8/FP4, warp specialization, and HuggingFace cache policy changes without dragging full attention stacks into every comparison."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Decode paths die by a thousand cuts: host staging, stream orchestration, cache policy, compile overhead, and kernel schedule all matter. This lab keeps those costs as separate targets so you can see what actually moves TTFT, TPOT, and total decode latency."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager decode on pageable inputs and conservative cache policy
                - straightforward correctness reference
                - enough host and launch overhead to make serving optimizations visible"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - pinned inputs and dual-stream decode variants
                - `torch.compile` and CUDA Graph decode paths
                - FP8/FP4 and warp-specialized kernels where the hardware supports them
                - static-cache HuggingFace loop for the cache-policy pair"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `decode` | `9.845 ms` | `2.441 ms` (`ultimate`) | `4.03x` |
                | `decode_hf_cache` | `288.157 ms` | `39.843 ms` | `7.23x` |
                | `decode_streams` | `27.391 ms` | `23.753 ms` | `1.15x` |
                | `decode_warp_specialized` | `38.386 ms` | `14.963 ms` | `2.57x` |
                | `decode_double_buffer_tma` | `0.173 ms` | `0.081 ms` | `2.14x` |
                | `decode_device_resident` | `11.487 ms` | `2.340 ms` | `4.91x` |
                | `decode_candidate_logits` | `8.866 ms` | `6.335 ms` | `1.40x` |

                The `decode_device_resident` row is from `artifacts/runs/codex_decode_device_resident_20260630_204000/`; the optimized path reports zero prompt/payload copies per iteration. The `decode_candidate_logits` row is from `artifacts/runs/codex_decode_candidate_logits_forced_20260630_210000/`; the optimized path reduces the effective logits vocabulary from `131072` to `1`. This is the useful shape of the lab: some decode optimizations are huge, some are modest, and the lab keeps them separated instead of averaging them into a fake single story."""
            ),
        ),
        MarkdownSection(
            "Control Surfaces",
            dedent(
                """\
                Treat `decode_pinned` as a supplementary local-contract speed benchmark. The pair now uses a dedicated pageable-vs-pinned baseline on the same transfer-heavy workload (`host_payload_mb=512`, no stream overlap) so the pinned-memory comparison is measurable on its own. More broadly, the standalone pinned-memory stepping stones remain non-headline benchmarks, while the lab's canonical host-overhead benchmark stays on `decode_streams`, where the workload keeps that same large host payload and adds copy/compute overlap."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/decode_optimization:decode --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/decode_optimization:decode_device_resident --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/decode_optimization:decode_candidate_logits --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/decode_optimization:decode_hf_cache --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/decode_optimization:decode_warp_specialized --profile deep_dive --single-gpu
                ```

                Those targets cover the most useful slices: general decode orchestration, device-resident serving contracts, real decoder-loop cache policy, and the fused Triton kernel path."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/decode_optimization
                python -m cli.aisp bench run --targets labs/decode_optimization --profile none
                python -m cli.aisp demos labs-decode-multigpu --nproc-per-node 4 -- --iters 4 --warmup 1
                ```"""
            ),
        ),
    ],
    goals=[
        "Contrast eager vs pinned/streamed vs compiled/graph decode paths on the same workload.",
        "Measure FP8/FP4 tensor-core benefits relative to FP16/BF16 baselines.",
        "Validate Triton warp-specialized decode kernels against Python math and harness expectations.",
        "Observe NVLink-C2C behavior by scaling the decode loop across available GPUs.",
        "Show when a prefix-cache/device-resident request path can remove recurring prompt-side H2D staging.",
        "Show when guided decoding can avoid full-vocabulary logits by scoring only legal candidates.",
    ],
    contents=[
        ("`baseline_decode.py`, `optimized_decode_pinned.py`, `optimized_decode_streams.py`, `optimized_decode_compile.py`, `optimized_decode_graph.py`, `optimized_decode_graph_full.py`, `optimized_decode_ultimate.py`", "Serving-path decode variants that isolate host, stream, compile, and graph effects."),
        ("`baseline_decode_device_resident.py`, `optimized_decode_device_resident.py`", "Prefix-cache-style serving variant that seeds prompt-side inputs once and skips recurring H2D staging in the decode hot path."),
        ("`baseline_decode_candidate_logits.py`, `optimized_decode_candidate_logits.py`", "Guided/constrained decode variant that compares full-vocabulary scoring plus candidate filtering with direct candidate-only projection."),
        ("`baseline_decode_hf_cache.py`, `optimized_decode_hf_cache.py`", "Real HuggingFace decoder-loop comparison: dynamic cache + per-step EOS sync vs static cache + compiled decode + batched EOS polling."),
        ("`baseline_decode_fp8.py`, `optimized_decode_fp8.py`, `baseline_decode_fp4.py`, `optimized_decode_fp4.py`", "Prefill-focused low-precision decode comparisons on hardware that supports them, including the intentional BF16/nn.Linear versus FP8/Transformer Engine TELinear path."),
        ("`baseline_decode_warp_specialized.py`, `optimized_decode_warp_specialized.py`", "Warp-specialized decode path plus its eager correctness reference."),
        ("`baseline_decode_double_buffer_tma.py`, `optimized_decode_double_buffer_tma.py`, `decode_common.py`, `decode_multigpu_demo.py`", "CUDA double-buffer/TMA path, shared helpers, and the multi-GPU NVLink-C2C demo."),
    ],
    validation=[
        "Baseline vs pinned/streams shows improved TTFT and TPOT with lower host wait time.",
        "Compile/graph variants emit fewer kernels and higher tokens/sec than the baseline in harness output.",
        "FP8/FP4 runs use a prefill-focused workload (`decode_tokens=0`) to surface tensor-core benefits; outputs remain within tolerance.",
        "Warp-specialized Triton kernel is validated against a workload-matched eager baseline; the expectation file stays green.",
        "`decode_device_resident` emits zero prompt/payload copies per iteration on the optimized path while preserving the same model output.",
        "`decode_candidate_logits` emits the same constrained-token decode output while reducing the effective logits vocabulary from the full vocabulary to the candidate set.",
        "The multi-GPU demo exercises NVLink-C2C without graph-capture failures when launched via `torchrun`.",
    ],
    notes=[
        "All targets emit TTFT, TPOT mean, decode time, total time, and tokens/sec in `custom_metrics` for easy diffing.",
        "`decode_pinned` is a supplementary local-contract stepping-stone target that now isolates pageable vs pinned staging on the same large host payload; use `decode_streams` when you want the lab's canonical pinned-host plus overlap speed claim.",
        "`decode_device_resident` is a larger serving-contract optimization: it applies when routing and prefix-cache policy keep prompt-side buffers resident on the GPU between decode iterations.",
        "`decode_candidate_logits` models grammar, schema, or router-constrained serving where the legal next-token set is known before the lm_head projection.",
        "FP4 requires NVFP4-capable Blackwell hardware; unsupported platforms fail fast.",
        "The HF cache pair reproduces the main idea from Chaim Rand's token-generation optimization write-up while keeping the harness contract intact.",
        "`decode_fp8` is intentionally a BF16/`nn.Linear` baseline versus FP8/Transformer Engine `TELinear`, because Transformer Engine is the supported FP8 linear path in this lab.",
    ],
)

ENTRIES["labs/block_scaling"] = lab_entry(
    slug="labs/block_scaling",
    title="Lab - Blackwell Hardware Block Scaling",
    summary=dedent(
        """\
        Recreates the practical flow of Colfax Research's article on hardware-supported block scaling with NVIDIA Blackwell GPUs inside this repo's lab structure. The baseline path is intentionally conservative: it materializes block scales in BF16, applies them as explicit elementwise multiplies, and then calls `matmul`. The optimized path compiles the CUTLASS/CuTe blockscaled GEMM once during setup and measures only the Blackwell hardware-supported execution path.

        The lab now defaults to the larger Colfax-style workload:
        - `MNKL = 8192,8192,1024,1`
        - `mma_tiler_mn = 256,128`
        - `cluster_shape_mn = 2,1`
        - `sf_vec_size = 16`"""
    ),
    lead_sections=[
        MarkdownSection(
            "Credit",
            dedent(
                """\
                - Source article: Colfax Research, ["CUTLASS Tutorial: Hardware-supported Block Scaling with NVIDIA Blackwell GPUs"](<https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/>)
                - Kernel/source inspiration: NVIDIA CUTLASS `examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py` and the related `72_blackwell_narrow_precision_gemm` examples"""
            ),
        ),
        MarkdownSection(
            "Problem",
            "This lab exists to answer one concrete question: how much faster is Blackwell's hardware-supported block scaling than a conservative software block-scaling path on the same workload?",
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - explicit scale expansion in BF16
                - explicit elementwise scale application
                - BF16 GEMM after the scale work is already paid for"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compile-once CUTLASS/CuTe block-scaled GEMM in setup
                - timed Blackwell hardware path in the measured loop
                - separate microbenchmark path to compare the repo wrapper against the direct Colfax/CUTLASS run path"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current validated measurements on this B200:

                ### Harness Pair
                From `artifacts/runs/20260305_222139__bench__profile_none_targets_labs_block_scaling_block_scaling/`:

                | Path | Latency | Relative |
                | --- | ---: | ---: |
                | Baseline (`baseline_block_scaling`) | `0.198 ms` | `1.00x` |
                | Optimized (`optimized_block_scaling`) | `0.113 ms` | `1.76x faster` |

                ### Apples-to-Apples Microbenchmark
                | Path | Latency | TFLOP/s | Relative to lab hardware |
                | --- | ---: | ---: | ---: |
                | Software blockscaled ref | `0.1566 ms` | `877.8` | `2.34x slower` |
                | PyTorch BF16 GEMM | `0.1199 ms` | `1145.8` | `1.79x slower` |
                | Lab CUTLASS hardware | `0.0670 ms` | `2050.5` | `1.00x` |
                | Colfax/CUTLASS direct | `0.0711 ms` | `1934.0` | `1.06x slower` |"""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the harness path when you want reproducible profiler artifacts instead of just a local timing number:

                ```bash
                python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p deep_dive --single-gpu
                ```

                For the closest comparison to the Colfax article itself, use the standalone microbenchmark:

                ```bash
                python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python labs/block_scaling/compare_block_scaling.py
                python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10
                python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none --iterations 10 --warmup 5 --timeout-seconds 900 --single-gpu
                ```"""
            ),
        ),
    ],
    goals=[
        "See how Blackwell's blockscaled tensor-core path changes the cost model versus a software dequantize-and-matmul baseline.",
        "Run a CUTLASS/CuTe blockscaled kernel from the article in a repo-native, repeatable lab.",
        "Validate the numerical output against a software reference before trusting the timing.",
        "Sweep matrix shapes, tile shapes, and cluster shapes without rewriting the kernel code.",
    ],
    contents=[
        ("`baseline_block_scaling.py`", "Conservative software baseline: expand scales, multiply in BF16, then matmul."),
        ("`optimized_block_scaling.py`", "Compile-once Blackwell hardware blockscaled GEMM benchmark."),
        ("`block_scaling_common.py`", "Shared config parsing, tensor prep, CUTLASS example loading, and timing helpers."),
        ("`compare_block_scaling.py`", "Reproducible three-path runner: software blockscaled ref, pre-scaled BF16 GEMM, and hardware blockscaled GEMM."),
        ("`microbenchmark_block_scaling.py`", "Apples-to-apples microbenchmark that adds the direct Colfax/CUTLASS `run()` path."),
    ],
    run=RunSection(
        commands=[
            "python labs/block_scaling/compare_block_scaling.py",
            "python labs/block_scaling/compare_block_scaling.py --json-out /tmp/block_scaling_compare.json",
            "python labs/block_scaling/microbenchmark_block_scaling.py",
            "python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10 --json-out /tmp/block_scaling_microbench.json",
            "python -m cli.aisp bench list-targets --chapter labs/block_scaling",
            "python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none --iterations 10 --warmup 5 --timeout-seconds 900 --single-gpu",
        ],
        notes=[
            "The standalone runners lock GPU clocks through the repo harness by default; use `--no-lock-gpu-clocks` only for quick local iteration when repeatability is not the goal.",
            "Set `AISP_BLOCK_SCALING_SKIP_VERIFY=1` only for explicit timing-only sweeps; the default path verifies correctness.",
        ],
    ),
    run_heading="Running the Lab",
    run_intro="Use the comparison runner when you want a one-command answer on correctness plus speedup, and the harness path when you want standard repo artifacts.",
    validation=[
        "`python labs/block_scaling/compare_block_scaling.py` reports a hardware speedup greater than `1.0x` on a B200 / Blackwell system.",
        "`python labs/block_scaling/microbenchmark_block_scaling.py` keeps the lab wrapper in the same range as the direct Colfax/CUTLASS example.",
        "The comparison runner's correctness check passes before timing is reported.",
        "`python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none ...` executes both paths without recompiling the hardware kernel inside each measured iteration.",
    ],
    extra_sections=[
        dedent(
            """\
            ## Recommended Knobs
            The defaults are tuned for the larger Colfax-style B200 workload:
            - `AISP_BLOCK_SCALING_MNKL=8192,8192,1024,1`
            - `AISP_BLOCK_SCALING_MMA_TILER_MN=256,128`
            - `AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN=2,1`
            - `AISP_BLOCK_SCALING_SF_VEC_SIZE=16`

            You can still override them explicitly:
            ```bash
            AISP_BLOCK_SCALING_MNKL=8192,8192,1024,1 \\
            AISP_BLOCK_SCALING_MMA_TILER_MN=256,128 \\
            AISP_BLOCK_SCALING_CLUSTER_SHAPE_MN=2,1 \\
            python labs/block_scaling/compare_block_scaling.py
            ```"""
        ),
        dedent(
            """\
            ## Default Tuning Pass
            The current default tile/cluster pair came from a first-pass direct CUTLASS sweep on the article-sized workload (`warmup=1`, `iterations=10`, `skip_ref_check=True`):

            | `mma_tiler_mn` | `cluster_shape_mn` | Direct CUTLASS latency |
            | --- | --- | --- |
            | `128,128` | `1,1` | `78.33 us` |
            | `128,128` | `1,2` | `74.96 us` |
            | `128,128` | `2,1` | `74.75 us` |
            | `128,256` | `1,2` | `84.68 us` |
            | `256,128` | `2,1` | `71.68 us` |
            | `256,128` | `2,2` | `88.06 us` |

            That keeps the default aligned with the article's representative command line while also matching the best result from the local sweep."""
        ),
        dedent(
            """\
            ## Microbenchmark View
            The microbenchmark reports four distinct numbers on the same logical workload:

            | Path | What it measures |
            | --- | --- |
            | `Software blockscaled ref` | PyTorch scale multiply plus BF16 GEMM every iteration. |
            | `PyTorch BF16 GEMM` | BF16 GEMM after the scales were already applied. |
            | `Lab CUTLASS hardware` | The repo-native compile-once wrapper around the blockscaled tensor-core kernel. |
            | `Colfax/CUTLASS direct` | The original CUTLASS example's `run()` benchmark path. |

            This is the apples-to-apples interpretation:
            - `Software blockscaled ref` vs `Lab CUTLASS hardware` shows the real improvement from Blackwell's hardware-supported block scaling.
            - `PyTorch BF16 GEMM` isolates how much of the software path is just GEMM versus scale application overhead.
            - `Lab CUTLASS hardware` vs `Colfax/CUTLASS direct` checks whether the lab wrapper is staying in the same performance range as the original example.

            ### Representative B200 Ranges
            On this B200, with `python labs/block_scaling/microbenchmark_block_scaling.py --warmup 2 --iterations 10`, the lab produced:

            | Path | Latency | TFLOP/s | Relative to lab hardware |
            | --- | --- | --- | --- |
            | `Software blockscaled ref` | `0.1566 ms` | `877.8` | `2.34x slower` |
            | `PyTorch BF16 GEMM` | `0.1199 ms` | `1145.8` | `1.79x slower` |
            | `Lab CUTLASS hardware` | `0.0670 ms` | `2050.5` | `1.00x` |
            | `Colfax/CUTLASS direct` | `0.0711 ms` | `1934.0` | `1.06x slower` |"""
        ),
        dedent(
            """\
            ## Harness vs Microbenchmark
            The repo harness and the standalone microbenchmark answer slightly different questions:
            - `microbenchmark_block_scaling.py` uses CUDA-event timing around the direct call sites, so it is the right place to compare against Colfax/CUTLASS and PyTorch-reported kernel-adjacent ranges.
            - `bench run` measures the benchmark pair through the generic harness, including the per-iteration synchronization the harness uses for correctness and stability. That number is expected to be higher than the direct microbenchmark.

            On this B200, the harness run:
            - `python -m cli.aisp bench run -t labs/block_scaling:block_scaling -p none --iterations 10 --warmup 5 --timeout-seconds 900 --single-gpu`
            - reported `0.198 ms` baseline, `0.113 ms` optimized, and `1.76x` speedup
            - updated `labs/block_scaling/expectations_b200.json` from `1.669x` to `1.762x`"""
        ),
    ],
    notes=[
        "The optimized path requires a Blackwell-class GPU (`sm100+`). The software baseline still requires CUDA because the lab is meant to be compared on the same device.",
    ],
)

ENTRIES["labs/flashattention4"] = lab_entry(
    slug="labs/flashattention4",
    title="Lab - FlashAttention-4 Pipeline Co-Design",
    summary=dedent(
        """\
        Recreates the practical shape of the FlashAttention-4 article: eager FlexAttention as the scalar-heavy baseline, then a compiled Blackwell-friendly path that tries the FLASH backend and falls back to FlexAttention+TMA when needed. The default benchmark uses ALiBi because it is stable on the local stack and still exercises the FA4 score-mod path."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                This lab is here to test two different questions cleanly:
                - does the fused FA4-style path beat the eager score-materializing baseline in this repo?
                - does the local stack reproduce the Colfax / PyTorch FlashAttention-4 performance envelope?"""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager FlexAttention
                - explicit score materialization
                - good correctness reference, bad steady-state cost model"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compiled Blackwell-oriented path
                - prefers the experimental FLASH backend
                - falls back to compiled FlexAttention + TMA when the backend/toolchain combination cannot lower cleanly"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current validated harness result for the default `ALiBi` target from `artifacts/runs/20260306_023114__bench__profile_none_targets_labs_flashattention4_flashattention4_alibi/`:

                | Path | Latency | Relative |
                | --- | ---: | ---: |
                | Baseline (`baseline_flashattention4`) | `5.562 ms` | `1.00x` |
                | Optimized (`optimized_flashattention4_alibi`) | `0.385 ms` | `14.45x faster` |

                This lab also carries an important negative result: the local stack does **not** currently reproduce the published Colfax/PyTorch FA4 envelope on the direct TFLOP/s microbench. That is a useful finding, not a documentation problem to hide."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the harness for artifacted Nsight evidence:

                ```bash
                python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile deep_dive --single-gpu
                ```

                Use the microbenchmark when you want the closest backend-vs-backend comparison to the published articles:

                ```bash
                python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
                python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/flashattention4
                python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal
                python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure the delta between eager score materialization and a fused compiled attention kernel.",
        "Exercise FA4-style score modifiers such as ALiBi and soft-capped logits, and optionally probe sliding-window masks on a best-effort basis.",
        "Inspect provider selection on Blackwell (`flash_backend` vs `flex_tma`).",
        "Use a coarse pipeline model to explain why overlap matters more under asymmetric hardware scaling.",
    ],
    contents=[
        ("`baseline_flashattention4.py`, `optimized_flashattention4.py`", "Benchmark pair comparing eager FlexAttention to a compiled, provider-aware FA4 path."),
        ("`flashattention4_common.py`", "Shared input builders, score mods, mask construction, and provider resolution."),
        ("`pipeline_model.py`", "Latency model for serial versus overlapped attention tiles."),
        ("`tflops_microbench.py`", "Clock-locked TFLOPs/s microbenchmark for Colfax/PyTorch-style backend comparisons."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal` shows the eager baseline materializing scores while the optimized path stays fused.",
        "`python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal` succeeds on a cold-start process and exercises the FA4 score-mod path without relying on env vars.",
        "`python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal` gives the clearest absolute-performance path for standard attention on this stack.",
        "`python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_windowed --profile minimal` and `labs/flashattention4:flashattention4_alibi_windowed` remain explicit experimental probes; treat failures there as a PyTorch/FA4 integration limitation on this stack rather than as a lab bug.",
        "`python labs/flashattention4/pipeline_model.py --tiles 64 --tensor-core-scale 4 --scalar-scale 2` demonstrates overlap becoming more valuable as tensor cores scale faster than scalar hardware.",
        "`python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi` runs the public-shape backend comparison against the local FLASH backend, the local Triton-style proxy, and cuDNN where supported.",
        "`python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` checks whether a larger compute-bound shape moves the local stack toward the published Colfax/PyTorch envelope.",
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp bench list-targets --chapter labs/flashattention4",
            "python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal",
            "python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal",
            "python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal",
            "python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_softcap --profile minimal",
            "python labs/flashattention4/pipeline_model.py --tiles 32 --tensor-core-scale 4 --scalar-scale 2",
            "python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi",
            "python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa",
        ],
        notes=[
            "Harness workflows use explicit targets such as `flashattention4_dense`, `flashattention4_causal`, `flashattention4_alibi`, `flashattention4_softcap`, `flashattention4_windowed`, `flashattention4_alibi_windowed`, and the matching `best_available_attention_*` variants.",
            "On the local `torch 2.9.1+cu130` build, `windowed` and `alibi_windowed` are experimental: the optimized path can produce non-finite outputs on a fresh compile even though upstream FA4 supports sliding-window patterns.",
            "`tflops_microbench.py` locks GPU clocks through `core.harness.benchmark_harness.lock_gpu_clocks()` by default; use `--no-lock-gpu-clocks` only for local debugging.",
        ],
    ),
    extra_sections=[
        dedent(
            """\
            ## TFLOPs/s Microbenchmark
            Use `tflops_microbench.py` when you want something closer to the published Colfax and PyTorch comparisons than the harness benchmark pair. The harness pair is intentionally end-to-end and compares eager score materialization against a fused kernel; the microbenchmark instead compares backend implementations on the same attention workload.

            | Published comparison target | Local command | Notes |
            | --- | --- | --- |
            | Colfax B200 BF16 forward envelope (`1605 TFLOPs/s`, up to `1.3x` over cuDNN 9.13, up to `2.7x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` | Uses a larger shape to push the local stack harder. |
            | PyTorch GB200 standard-attention forward envelope (`1.6x-3.2x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal` | Uses the public blog shape `B=2, H=8, S=2048, D=128`. |
            | PyTorch GB200 ALiBi forward envelope (`1.2x-2.1x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode alibi --backends flash_backend triton_flex flex_tma` | cuDNN SDPA is not applicable to ALiBi. |

            The FLOP accounting matches the common SDPA forward convention used in vendor/blog comparisons:
            `forward_flops = 4 * batch * heads * head_dim * nonmasked_attention_elements`

            - For `dense`, `alibi`, and `softcap`, `nonmasked_attention_elements = q_seq_len * kv_seq_len`.
            - For `causal`, `windowed`, and `alibi_windowed`, only the unmasked score matrix entries are counted.
            - `triton_flex` is the closest local proxy for the blog's Triton baseline: compiled FlexAttention with `USE_TMA=False`.
            """
        ),
        dedent(
            """\
            ## Current Local Results
            These measurements were taken on March 5, 2026 on the current local `torch 2.9.1+cu130` stack with harness clock locking enabled. This host is still virtualized, so treat the numbers as directional rather than canonical.

            ### Public Blog Shape (`B=2, H=8, S=2048, D=128`)
            | Mode | Backend | Median (ms) | TFLOPs/s | Flash vs Triton | Flash vs cuDNN | Published check |
            | --- | --- | ---: | ---: | ---: | ---: | --- |
            | `dense` | `flash_backend` | 0.224 | 153.6 | `1.02x` | `0.40x` | Outside Colfax and PyTorch ranges |
            | `dense` | `triton_flex` | 0.229 | 150.1 | `1.00x` | `0.39x` | Local Triton-style proxy |
            | `dense` | `cudnn_sdpa` | 0.090 | 382.5 | `2.55x` | `1.00x` | Local cuDNN leader |
            | `causal` | `flash_backend` | 0.238 | 72.1 | `14.84x` | `0.37x` | Beats local Triton-style proxy, still far below cuDNN |
            | `causal` | `triton_flex` | 3.538 | 4.9 | `1.00x` | `0.02x` | Local Triton-style proxy collapses on this stack |
            | `causal` | `cudnn_sdpa` | 0.088 | 195.5 | `40.25x` | `1.00x` | Local cuDNN leader |
            | `alibi` | `flash_backend` | 6.221 | 5.5 | `1.02x` | n/a | Outside PyTorch ALiBi range |
            | `alibi` | `triton_flex` | 6.323 | 5.4 | `1.00x` | n/a | Local Triton-style proxy |
            | `alibi` | `flex_tma` | 6.169 | 5.6 | `1.03x` | n/a | Slightly ahead locally, still not near published envelope |

            ### Peak Probe Shape (`B=8, H=16, S=4096, D=128`)
            | Mode | Backend | Median (ms) | TFLOPs/s | % of Colfax 1605 | Flash vs Triton | Flash vs cuDNN |
            | --- | --- | ---: | ---: | ---: | ---: | ---: |
            | `dense` | `flash_backend` | 3.576 | 307.5 | 19.2% | `1.01x` | `0.34x` |
            | `dense` | `triton_flex` | 3.614 | 304.2 | 19.0% | `1.00x` | `0.34x` |
            | `dense` | `cudnn_sdpa` | 1.222 | 899.8 | 56.1% | `2.96x` | `1.00x` |
            | `causal` | `flash_backend` | 2.264 | 242.9 | 15.1% | `0.97x` | `0.36x` |
            | `causal` | `triton_flex` | 2.200 | 250.0 | 15.6% | `1.00x` | `0.37x` |
            | `causal` | `cudnn_sdpa` | 0.814 | 675.1 | 42.1% | `2.70x` | `1.00x` |

            The local conclusion is straightforward: this stack does not currently reproduce the published Colfax or PyTorch FlashAttention-4 envelope. The larger probe rules out a pure small-shape saturation explanation because the local FLASH path still tops out at `307.5 TFLOPs/s` on dense and `242.9 TFLOPs/s` on causal, well below both Colfax's `1605 TFLOPs/s` peak and the local cuDNN path.
            """
        ),
    ],
    notes=[
        "Sources: Colfax Research's FlashAttention-4 article (`https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/`) and the PyTorch FlexAttention + FlashAttention-4 integration post (`https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/`).",
        "Colfax reports up to `1605 TFLOPs/s` on B200 BF16 at roughly `71%` utilization, plus up to `1.3x` over cuDNN 9.13 and `2.7x` over Triton for forward passes.",
        "The PyTorch post reports `1.6x-3.2x` forward speedup over Triton for standard dense/causal attention on GB200, `1.2x-2.1x` for ALiBi, and `1.4x-2.1x` for sliding-window attention.",
        "The local PyTorch/Triton stack needs a quoted backend literal for the experimental FLASH backend; the lab handles that workaround internally and falls back automatically if needed.",
        "The lab pins float32 accumulation to IEEE mode because the current sm_100 lowering produced non-finite outputs under TF32 accumulation.",
        "Sliding-window modes remain exposed as explicit benchmark targets, but the stable day-to-day harness path is `flashattention4_alibi`.",
    ],
)

ENTRIES["labs/flashinfer_attention"] = lab_entry(
    slug="labs/flashinfer_attention",
    title="Lab - FlashInfer Block-Sparse Attention",
    summary=dedent(
        """\
        Runs a block-sparse attention kernel with FlashInfer and compares it to dense SDP plus an equivalent sparsity mask on an LLM-scale head configuration, including the output projection."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Block-sparse attention is only interesting if the sparse kernel plus projection work actually beats the dense masked path. This lab keeps the output projection in both paths so the comparison stays honest."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - dense SDP plus sparsity mask
                - same output projection as the optimized path
                - useful correctness and cost-model reference"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - FlashInfer block-sparse attention
                - same head geometry and output projection
                - tuned to measure sparsity benefits at realistic hidden sizes"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `flashinfer_attention` | `1.043 ms` | `0.320 ms` | `3.26x` |

                This is a good example of a sparse-kernel lab that still keeps the surrounding work visible instead of benchmarking an unrealistically stripped-down kernel in isolation."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile deep_dive --single-gpu
                ```

                Use the deep-dive run when you want Nsight evidence for both the sparse attention kernel and the output projection path."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/flashinfer_attention
                python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure block-sparse attention speedups at high sparsity ratios.",
        "Validate FlashInfer kernels on realistic head dimensions.",
        "Profile attention plus output projection as a unit of work.",
    ],
    contents=[
        ("`baseline_flashinfer_attention.py`", "Dense SDP + mask baseline with output projection."),
        ("`optimized_flashinfer_attention.py`", "FlashInfer block-sparse attention with output projection."),
        ("`expectations_{hardware_key}.json`", "Expectation files that keep the benchmark pair regression-checked on supported hardware."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile minimal` captures the dense-vs-sparse delta on the same head geometry.",
        "The optimized path should verify against the dense masked reference before timing is reported.",
    ],
    notes=[
        "The default head configuration targets a GPT-OSS-20B-style hidden size (`2880`) with `head_dim=64` (`45` heads).",
        "Increase `seq_len` if you want larger sparse regions to dominate the timing.",
        "Requires FlashInfer (`pip install flashinfer-python==0.6.3`).",
    ],
)

ENTRIES["labs/async_input_pipeline"] = lab_entry(
    slug="labs/async_input_pipeline",
    title="Lab - Async Input Pipeline",
    summary=dedent(
        """\
        Compares a blocking input path to an overlapped asynchronous staging path so you can see whether host/device feeding is the bottleneck on this workload."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Host-side staging can make a GPU workload look slower even when the kernel is fine. This lab keeps the input path measurable instead of letting pipeline overhead hide inside model-level numbers."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - synchronous input preparation and transfer
                - simple end-to-end reference
                - intentionally leaves overlap on the table"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - asynchronous input staging
                - overlaps host/device preparation with compute
                - same benchmark contract, but less visible pipeline stall"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `async_input_pipeline` | `135.105 ms` | `88.690 ms` | `1.52x` |

                Earlier exploratory runs showed larger numbers, but the strict rerun is the one worth publishing. This lab is about making overlap measurable under the harness contract, not about keeping the biggest scalar."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile deep_dive --single-gpu
                ```

                Nsight is useful here because the overlap story should show up directly in the timeline: less host-visible stall, not just a smaller latency number."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/async_input_pipeline
                python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Make input staging cost visible under the same harness contract as the rest of the repo.",
        "Show when async overlap matters and when it does not.",
        "Keep host-side data movement from masquerading as a kernel optimization problem.",
    ],
    contents=[
        ("`baseline_async_input_pipeline.py`, `optimized_async_input_pipeline.py`", "Benchmark pair for blocking vs asynchronous staging."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the async pipeline pair."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile minimal` should show lower end-to-end latency for the overlapped path on this host.",
        "Deep-dive runs should show the optimized path reducing host-visible stall rather than changing the math workload.",
    ],
    notes=[
        "This is an end-to-end pipeline lab, so the value is in the timeline and total latency, not just kernel-local timing.",
        "Keep this lab as the primary copy-stream overlap example. `labs/training_hotpath` reuses that concept by reference and only adds the missing reduction and padding-aware cases.",
    ],
)

ENTRIES["labs/training_hotpath"] = lab_entry(
    slug="labs/training_hotpath",
    title="Lab - Training Hot-Path",
    summary=dedent(
        """\
        Adds three small supporting examples for training-side bottlenecks that were missing from the main chapter flow: vectorized metric aggregation, fused CUDA reduction, and padding-aware projection work that skips padded rows without changing the chapter-primary stories in `ch12`, `ch14`, or `labs/async_input_pipeline`."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Some training bottlenecks are too small and cross-cutting to deserve a chapter rewrite, but still matter in practice. This lab isolates three of them so they can be benchmarked honestly:

                - scalar Python-side metric aggregation that should have been vectorized
                - segmented reductions that should have been fused into one CUDA pass
                - padded projection work that should only run on active tokens"""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - scalar or unfused reduction logic
                - padded dense projection work over every token position
                - simple reference paths that preserve correctness but spend work on overhead and padding"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - tensor-vectorized metric aggregation
                - a local CUDA extension for fused segmented abs-mean reduction
                - packed-row projection kernels that only touch active rows before scattering results back into the padded layout"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Strict `minimal` expectation-backed runs on the current `b200` hardware key produced the following deltas:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `metric_reduction_vectorized` | `14.050 ms` | `0.158 ms` | `89.12x` faster |
                | `metric_reduction_cuda` | `2.760 ms` | `0.050 ms` | `55.06x` faster |
                | `padding_aware_transformer` | `2.919 ms` | `3.430 ms` | `0.85x` speed, `76.4%` lower peak memory |
                """
            ),
        ),
        MarkdownSection(
            "Goal Semantics",
            dedent(
                """\
                Treat `padding_aware_transformer` as a memory-goal benchmark. Its value is the large reduction in padded-row activation footprint; a slower timed path is acceptable when the measured memory drop is material and verification stays green."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/training_hotpath:metric_reduction_vectorized --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/training_hotpath:metric_reduction_cuda --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/training_hotpath:padding_aware_transformer --profile deep_dive --single-gpu
                ```

                Use the profile when you want to see the missing work directly:
                - vectorized vs scalar host-side reduction behavior
                - one fused CUDA reduction kernel instead of repeated segmented slicing
                - active-row packing that reduces padded projection work"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m labs.training_hotpath.compare --example metric_reduction_vectorized
                python -m cli.aisp bench list-targets --chapter labs/training_hotpath
                python -m cli.aisp bench run --targets labs/training_hotpath --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark vectorized training-side metric aggregation instead of leaving it as an unmeasured Python loop.",
        "Show when a local CUDA fused reduction is worth the extension overhead.",
        "Make padding-aware projection savings visible without changing the primary CUDA Graph or compiler chapter narratives.",
    ],
    contents=[
        ("`baseline_metric_reduction_vectorized.py`, `optimized_metric_reduction_vectorized.py`", "Scalar per-output aggregation vs tensor-vectorized metric reduction."),
        ("`baseline_metric_reduction_cuda.py`, `optimized_metric_reduction_cuda.py`", "Torch segmented reduction baseline vs fused CUDA extension."),
        ("`baseline_padding_aware_transformer.py`, `optimized_padding_aware_transformer.py`", "Dense padded projections vs packed-row padding-aware projections."),
        ("`training_hotpath_common.py`, `training_hotpath_extension.py`, `training_hotpath_kernels.cu`, `compare.py`", "Shared workloads, local extension loader, CUDA kernels, and a direct compare entrypoint."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the three supporting benchmark pairs."),
    ],
    validation=[
        "`python -m cli.aisp bench list-targets --chapter labs/training_hotpath` should discover the three supporting benchmark pairs.",
        "The optimized metric reduction targets should match their baselines numerically while flipping `metric_reduction.is_vectorized` or `metric_reduction.is_fused_cuda` in custom metrics.",
        "The padding-aware target should preserve outputs while flipping `padding_aware.enabled`, reporting a non-trivial padded-token fraction, and reducing peak memory even if raw time is not faster.",
    ],
    notes=[
        "Keep `ch14:model_compile_reduced_precision` as the primary compile + reduced-precision training story.",
        "Keep `ch12:cuda_graphs` as the primary CUDA Graph replay story.",
        "Keep `labs/async_input_pipeline:async_input_pipeline` as the primary copy-stream overlap story.",
        "`padding_aware_transformer` is a memory-goal benchmark; judge it by peak-memory reduction first, not by raw speedup.",
    ],
)

ENTRIES["labs/custom_vs_cublas"] = lab_entry(
    slug="labs/custom_vs_cublas",
    title="Lab - Custom Kernel vs cuBLAS",
    summary=dedent(
        """\
        Pits a hand-tuned TCGEN05/CUTLASS-style matmul path against the library baseline so you can see when a custom schedule is actually worth the maintenance cost."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Custom matmul kernels are easy to oversell. This lab keeps the question narrow: for this shape family, does the custom path really beat the cuBLAS-style baseline enough to justify itself?"""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - library/reference matmul path
                - stable correctness anchor
                - useful for checking whether the custom kernel is actually better than "just use cuBLAS" """
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - custom TCGEN05-oriented matmul implementation
                - same math, but explicit schedule/layout control
                - designed to answer whether a bespoke kernel wins on this shape family"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `tcgen05_matmul` | `4.027 ms` | `1.740 ms` | `2.31x` |

                That is a meaningful win, not just benchmark noise. The lab matters because it turns "custom vs vendor library" into a measured tradeoff rather than ideology."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile deep_dive --single-gpu
                ```

                Use the deep-dive profile when you want to attribute the win to tile/schedule choices instead of relying on a single latency number."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/custom_vs_cublas
                python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile minimal
                python labs/custom_vs_cublas/autotune.py --help
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark a bespoke Blackwell-oriented matmul path against the library baseline.",
        "Keep kernel-selection and autotuning artifacts close to the benchmark pair.",
        "Make it obvious when a custom path is real value instead of benchmark folklore.",
    ],
    contents=[
        ("`baseline_tcgen05_matmul.py`, `optimized_tcgen05_matmul.py`", "Baseline/optimized benchmark pair for the TCGEN05 matmul lab."),
        ("`autotune.py`, `cutlass_gemm/`, `experimental/`", "Tuning helpers and implementation artifacts for the custom kernel path."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the benchmark pair."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile minimal` should keep the custom path ahead of the baseline on validated hardware.",
        "The optimized path must stay verification-clean; a faster wrong kernel does not count.",
    ],
    notes=[
        'This lab is one of the clearest places to show the repo\'s bias toward measured custom-kernel value instead of hand-wavy "handwritten kernels are faster" claims.',
    ],
)

ENTRIES["labs/flashattention_gluon"] = lab_entry(
    slug="labs/flashattention_gluon",
    title="Lab - FlashAttention Gluon",
    summary=dedent(
        """\
        Benchmarks a FlashAttention-style optimized path against a simpler attention reference so the local Gluon-flavored integration stays measured and honest."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Attention-stack integrations can look "fast" because the benchmark is fuzzy. This lab keeps the pair narrow so you can see whether the Gluon-oriented optimized path really buys anything on this stack."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - simple attention reference path
                - correctness anchor for the optimized implementation
                - no fused fast-path assumptions"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - FlashAttention-style optimized path
                - same workload and harness contract
                - focused on local integration cost/benefit, not a synthetic peak score"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `flashattention_gluon` | `0.205 ms` | `0.154 ms` | `1.33x` |

                This is a modest but real backend/path win. The useful part is that the result stays measured and reproducible instead of being hidden in a broader model benchmark."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
                python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep the local FlashAttention/Gluon integration benchmarked as a clean pair.",
        "Measure backend-path value without mixing in unrelated model-level effects.",
        "Use a small, stable attention benchmark as an integration health signal.",
    ],
    contents=[
        ("`baseline_flashattention_gluon.py`, `optimized_flashattention_gluon.py`", "Baseline and optimized harness entrypoints."),
        ("`flashattention_gluon_common.py`", "Shared workload setup and helper code."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the lab."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal` should keep the optimized path ahead on validated hardware.",
    ],
    notes=[
        "Treat this as an integration-health benchmark more than as a giant architectural headline win.",
    ],
)

ENTRIES["labs/kv_cache_compression"] = lab_entry(
    slug="labs/kv_cache_compression",
    title="Lab - KV Cache Compression",
    summary=dedent(
        """\
        Tests whether compressing the KV cache is worth it for this workload, instead of assuming lower memory footprint automatically means better serving latency."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                KV-cache compression is attractive because the memory story is obvious, but the latency story often is not. This lab exists to keep those two questions separate."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - uncompressed KV cache path
                - simple latency/memory reference
                - no compression overhead in the hot path"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compressed KV cache representation
                - same benchmark harness and validation contract
                - tests whether the memory tradeoff is actually latency-positive here"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `kv_cache` | `6066.040 ms` | `5897.083 ms` | `1.03x` |

                The important takeaway is restraint: the compressed path helps, but only slightly on this workload. This is exactly the kind of lab where a clean benchmark pair prevents an overclaim."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/kv_cache_compression
                python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure the latency cost/benefit of KV-cache compression under the harness contract.",
        "Keep memory-saving and latency-saving claims distinct.",
        "Make it easy to inspect whether compression overhead dominates the win.",
    ],
    contents=[
        ("`baseline_kv_cache.py`, `optimized_kv_cache_nvfp4.py`", "Baseline and compressed KV-cache benchmark pair."),
        ("`kv_cache_common.py`", "Shared workload setup."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal` should keep the compressed path verification-clean and modestly ahead on this hardware.",
    ],
    notes=[
        "This is a good lab for demonstrating that some memory optimizations are valuable mostly for capacity, not for giant latency wins.",
    ],
)

ENTRIES["labs/moe_optimization_journey"] = lab_entry(
    slug="labs/moe_optimization_journey",
    title="Lab - MoE Optimization Journey",
    summary=dedent(
        """\
        Packages a staged MoE optimization story from naive execution to quantized/padded fast paths so you can measure which step is actually doing the work."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                MoE optimization is often told as a narrative, not a benchmarked sequence. This lab keeps the sequence explicit so you can see which stage of the journey is providing the real win."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - naive MoE execution path
                - simple correctness reference
                - useful for showing how expensive unstructured expert execution can be"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - staged optimized MoE path with batching/layout/scheduling improvements
                - separate padded/quantized route for a more production-like fast path
                - designed to attribute wins to concrete optimization steps"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict results from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `moe` | `41.938 ms` | `1.217 ms` | `34.47x` |
                | `moe_pad_quant` | `4.681 ms` | `1.790 ms` | `2.62x` |

                The spread is useful. The big win is in the core MoE path, while the padded/quantized lane is a smaller, still-real follow-on improvement."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe_pad_quant --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/moe_optimization_journey
                python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Show a stepwise MoE optimization story with measured deltas instead of vague progression.",
        "Keep the naive path, batched path, and padded/quantized path benchmarked under one roof.",
        "Make it obvious which optimization stage is worth carrying forward.",
    ],
    contents=[
        ("`baseline_moe.py`, `baseline_moe_pad_quant.py`", "Naive/reference entrypoints."),
        ("`level0_naive.py` through `level6_full_stack.py`", "Incremental optimization stages used by the journey, including a real CUDA-graph replay stage before the compiled finale."),
        ("`moe_benchmark.py`", "Shared benchmark harness layer for the staged MoE path."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/moe_optimization_journey --profile minimal` should keep both the core MoE and pad/quant targets green.",
        "Deep-dive runs should make the kernel/layout win attributable to the staged path rather than only to end-to-end timing.",
        "The Level 6 CUDA-graphs entrypoint should report graph capture/replay instead of silently falling back to the Level 5 fused path.",
    ],
    notes=[
        "This lab is a good example of how the repo should teach optimization: staged, benchmarked, and profiler-backed.",
    ],
)

ENTRIES["labs/nanochat_fullstack"] = lab_entry(
    slug="labs/nanochat_fullstack",
    title="Lab - NanoChat Fullstack",
    summary=dedent(
        """\
        Wraps the NanoChat full-stack tree with a clean harness benchmark pair so the repo can talk about a real end-to-end inference stack with measured baseline vs optimized deltas, not just kernels."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Full-stack LLM projects are easy to describe in product terms and hard to benchmark cleanly. This lab keeps a narrow baseline/optimized inference pair inside the larger NanoChat tree so the performance story stays measurable."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - slower NanoChat inference path
                - end-to-end reference inside the same full-stack project
                - useful for checking whether the optimized path is buying real latency reduction"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - optimized NanoChat inference path
                - same harness contract and verification expectations
                - intended to represent the practical serving-side improvements, not just a kernel microbench"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `nanochat_inference` | `122.975 ms` | `67.621 ms` | `1.82x` |

                That is the useful local story: NanoChat is still a full-stack project, but the repo now has a concrete measured inference delta for it instead of leaving the performance claim buried in a much larger README."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile deep_dive --single-gpu
                ```

                Use the deep-dive path when you want Nsight evidence for the inference stack. Keep the `speedrun.sh` story separate from the benchmark pair; they answer different questions."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/nanochat_fullstack
                python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile minimal
                python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep a real full-stack LLM project in the benchmark story, not just microkernels.",
        "Benchmark NanoChat inference as a clean baseline/optimized pair inside the larger tree.",
        "Point readers at the broader project context without losing the measured harness story.",
    ],
    contents=[
        ("`baseline_nanochat_inference.py`, `optimized_nanochat_inference.py`", "Harness benchmark pair for NanoChat inference."),
        ("`benchmark_incremental_optimizations.py`", "Incremental benchmarking helper inside the NanoChat tree."),
        ("`speedrun.sh`, `run1000.sh`, `README_FAST.md`", "Broader NanoChat quick-start and end-to-end project entrypoints."),
        ("`nanochat/`, `scripts/`, `tasks/`, `tests/`", "Core NanoChat project tree and operational helpers."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/nanochat_fullstack:nanochat_inference --profile minimal` should keep the optimized path ahead under the harness contract.",
        "`python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference` should stay green before any performance claim is accepted.",
    ],
    notes=[
        "This README focuses on the repo's benchmarked NanoChat story. Use `README_FAST.md` and the project scripts when you want the broader training/serving walkthrough.",
        "The decode microbenchmarks live separately in `labs/decode_optimization`; this lab is the broader inference-stack companion.",
    ],
    extra_sections=[
        dedent(
            """\
            ## Project Context
            NanoChat is intentionally bigger than a single benchmark pair. The point of this lab entry is to give the repo one clean performance anchor inside that tree, not to replace the broader NanoChat project documentation.

            - Use [README_FAST.md](README_FAST.md) for the faster end-to-end project walkthrough.
            - Use [speedrun.sh](speedrun.sh) when you want the broader "train and talk to a small model" experience.
            - Use [rustbpe/README.md](rustbpe/README.md) for the tokenizer-specific component work.
            """
        ),
    ],
)

ENTRIES["labs/nanochat_fullstack/rustbpe"] = lab_entry(
    slug="labs/nanochat_fullstack/rustbpe",
    title="Component - rustbpe",
    summary=dedent(
        """\
        Lightweight Rust tokenizer-training library that complements the broader NanoChat stack. It is a component doc, not a benchmark-pair lab."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Tokenizer training is often either too slow and simple or too feature-heavy and opaque. `rustbpe` exists to keep the implementation lightweight, reasonably fast, and easy to understand."""
            ),
        ),
        MarkdownSection(
            "What This Component Is",
            dedent(
                """\
                - a Rust library for training a GPT-style tokenizer
                - part of the broader NanoChat project tree
                - focused on simple implementation and practical speed, not benchmark-harness integration"""
            ),
        ),
        MarkdownSection(
            "Why This Is Not A Benchmark Pair",
            dedent(
                """\
                `rustbpe` is a supporting component, not a baseline/optimized benchmark lab. The right contract here is build/test clarity plus how it fits into NanoChat, not a fabricated performance delta section."""
            ),
        ),
    ],
    goals=[
        "Keep the tokenizer-training component visible and understandable inside the larger NanoChat tree.",
        "Document how to build and test the Rust component without pretending it is a harness benchmark.",
        "Make the component's role in the broader full-stack project easy to find.",
    ],
    contents=[
        ("`Cargo.toml`, `Cargo.lock`", "Rust package metadata and dependency lockfile."),
        ("`src/lib.rs`", "Tokenizer-training implementation."),
        ("`../README.md`, `../README_FAST.md`", "Broader NanoChat project docs that explain how this component fits into the full stack."),
    ],
    run=RunSection(
        commands=[
            "cd labs/nanochat_fullstack/rustbpe",
            "cargo build --release",
            "cargo test",
        ],
        notes=[
            "This is a Rust-native component workflow, not a harness-target workflow.",
            "The crate is pinned to Rust Edition 2021 so it builds on stable Cargo toolchains used by the repo test environment.",
            "Use the parent NanoChat docs for end-to-end training/inference context.",
        ],
    ),
    run_heading="Building and Testing",
    run_intro="Use Cargo directly for this component.",
    validation=[
        "`cargo build --release` should compile the library cleanly on the repo's supported stable Rust toolchain.",
        "`cargo test` should keep the component healthy as the NanoChat tree evolves.",
    ],
    extra_sections=[
        dedent(
            """\
            ## How It Fits Into NanoChat
            `rustbpe` is the tokenizer-training companion inside the broader [labs/nanochat_fullstack/README.md](../README.md) tree.

            - Use [labs/nanochat_fullstack/README.md](../README.md) for the measured inference-stack story.
            - Use [labs/nanochat_fullstack/README_FAST.md](../README_FAST.md) for the quicker project walkthrough.
            - Use this component doc when you only need the tokenizer-training piece.
            """
        ),
    ],
    notes=[
        "This doc intentionally uses the same generator path as the benchmark-facing labs so the repo stays tidy, even when the component itself is not a benchmark pair.",
        "The crate no longer requires a nightly-or-newer Cargo parser just to read `Cargo.toml`; keep it 2021-compatible unless the code actually needs a newer edition feature.",
    ],
)

ENTRIES["labs/python_concurrency"] = lab_entry(
    slug="labs/python_concurrency",
    title="Lab - Python Concurrency Playbook",
    summary=dedent(
        """\
        A control-plane lab for Python concurrency work: bounded queues, retries, cancellation, idempotency, hybrid async/process pipelines, and the operational invariants that keep them correct."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Concurrency bugs usually look like performance bugs until you inspect the invariants. This lab is here to teach the control-plane discipline directly: bounded pressure, explicit failure handling, and deterministic terminal state."""
            ),
        ),
        MarkdownSection(
            "What This Lab Is",
            dedent(
                """\
                - a scenario and playbook lab
                - multiple runnable reference scripts
                - focused on correctness plus measurable throughput/latency behavior

                It is **not** currently a single baseline/optimized benchmark pair, and the README should say that plainly."""
            ),
        ),
        MarkdownSection(
            "What A Proper Benchmark Pair Would Look Like",
            dedent(
                """\
                If we decide to productize this as a harness target later, the clean shape would be something like:

                - `baseline_sync_pipeline.py`: serial or poorly bounded control path
                - `optimized_hybrid_pipeline.py`: bounded async + process-pool pipeline

                with a fixed JSON workload, invariant checks (`one terminal status per item`, ordered output, retry accounting), and measured outputs such as throughput plus p95/p99. That would be a new benchmark pair, not just a rename of the current playbook scripts."""
            ),
        ),
    ],
    goals=[
        "Teach practical concurrency design under production-style constraints.",
        "Keep correctness invariants visible instead of treating them as afterthoughts.",
        "Provide runnable drills for async I/O, CPU work, retries, cancellation, and hybrid pipelines.",
    ],
    contents=[
        ("`all_in_one_pipeline.py`", "Single-file reference for bounded queues, retries, dedupe, timeout handling, and hybrid async/process execution."),
        ("`taskrun_round1_asyncio.py`, `taskrun_round2_controls.py`, `taskrun_round3_idempotency.py`", "Staged drills that build the playbook incrementally."),
        ("`hybrid_three_stage_pipeline.py`, `executors_cpu_vs_io.py`, `gil_demo.py`", "Focused experiments for workload classification and executor behavior."),
        ("`ADVANCED_SCENARIOS.md`, `SCENARIO_QA.md`, `QUICK_REFERENCE_GUIDE.md`", "Interview/playbook-oriented docs that explain the patterns and failure modes."),
    ],
    run=RunSection(
        commands=[
            "python labs/python_concurrency/all_in_one_pipeline.py --input labs/python_concurrency/sample_all_in_one_items.json --stage-a-workers 3 --stage-b-workers 2 --stage-c-workers 2 --queue-size 4 --rps 12 --fetch-inflight 3 --write-inflight 2 --fetch-timeout-ms 200 --write-timeout-ms 200 --cpu-timeout-ms 1200 --fetch-retries 1 --write-retries 1 --cpu-rounds 8000 --seed 7",
            "python labs/python_concurrency/taskrun_round1_asyncio.py --help",
            "python labs/python_concurrency/hybrid_three_stage_pipeline.py --help",
        ],
        notes=[
            "This lab is script-first, not harness-first.",
            "The right success metric is invariant safety plus bounded latency/throughput behavior, not one synthetic speedup scalar.",
        ],
    ),
    validation=[
        "Each runnable scenario should preserve one terminal status per input item and deterministic ordered output where promised.",
        "Retry, cancellation, dedupe, and poison-path behavior should be visible in counters and summaries, not hidden.",
        "If this lab is later promoted into benchmark targets, add new explicit baseline/optimized files instead of retrofitting the current playbook scripts.",
    ],
    notes=[
        "This is a control-plane lab, so the current documentation shape is intentionally different from the benchmark-pair labs.",
    ],
)

ENTRIES["labs/nvfp4_dual_gemm"] = lab_entry(
    slug="labs/nvfp4_dual_gemm",
    title="Lab - NVFP4 Dual GEMM",
    summary=dedent(
        """\
        Challenge workspace for the GPUMODE NVFP4 dual-GEMM problem. It mixes baseline/optimized wrappers, official-parity local evaluation, and promotion-report AB checks."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Challenge workspaces are easy to turn into folklore. This lab is useful only if the local evaluator, the current promoted route, and the leaderboard target stay visible at the same time."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - official/reference and baseline submission path
                - correctness and challenge-semantics anchor
                - much slower than the tuned route on current local measurements"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - current promoted candidate route in `optimized_submission.py`
                - validated primarily through official-parity local eval plus strict A/B promotion reports
                - challenge workspace semantics first, generic harness story second"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current local state is best understood through two measurements:

                | Measurement surface | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | Fresh official-parity local eval (`2026-03-09`, `warmup=2`, `repeats=8`, `inputs_per_repeat=20`) | `190.124 us` (`baseline_submission.py`) | pending fresh rerun | pending |
                | Strict promotion A/B from `promotion_report_strict_ab.json` | `20.950 us` (prior promoted route) | `20.937 us` (`optimized_submission.py`) | `~1.00x` |

                The honest takeaway is that this workspace is still a challenge-tuning loop, not a fully canonical benchmark pair. The current optimized route is close to the prior promoted route on strict A/B, and the README should say that instead of pretending every run is a giant win."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the official-parity evaluator first, then the stored promotion reports:

                ```bash
                python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
                python -m json.tool labs/nvfp4_dual_gemm/promotion_report_strict_ab.json
                ```

                The promotion report is the artifact to trust when the per-run leaderboard numbers are too noisy to promote on their own."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/baseline_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
                python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep the dual-GEMM challenge workspace measurable under official-parity local semantics.",
        'Separate "current promoted candidate" evidence from generic baseline/optimized storytelling.',
        "Make the promotion-report A/B flow visible in the public docs.",
    ],
    contents=[
        ("`reference_submission.py`, `baseline_submission.py`, `optimized_submission.py`", "Reference, baseline, and promoted candidate submission files."),
        ("`baseline_nvfp4_dual_gemm.py`, `optimized_nvfp4_dual_gemm.py`", "Wrapper files for the benchmark-facing side of the workspace."),
        ("`local_eval.py`, `official_semantics_eval.py`, `promotion_report_strict_ab.json`", "Official-parity evaluator plus stored A/B promotion evidence."),
        ("`route_sweep_verify_green.json`, `grid_sweep_verify_green.json`, `top_submission_local_screen.json`", "Supporting tuning artifacts from the challenge loop."),
    ],
    run=RunSection(
        commands=[
            "python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/baseline_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json",
            "python -m labs.nvfp4_dual_gemm.local_eval --submission-file labs/nvfp4_dual_gemm/optimized_submission.py --reference-file labs/nvfp4_dual_gemm/reference_submission.py --warmup 2 --repeats 8 --inputs-per-repeat 20 --lock-gpu-clocks --sm-clock-mhz 1500 --json",
        ],
        notes=[
            "Use module invocation; direct script invocation is now re-exec'd to the same module entrypoint for compatibility.",
            "Treat `promotion_report_strict_ab.json` as the promotion gate when candidate deltas are small.",
        ],
    ),
    validation=[
        "Local evaluator runs should stay verification-clean against the reference implementation.",
        "Promotion decisions should still be based on repeated A/B evidence, not on a single low score.",
    ],
    notes=[
        "This is a challenge workspace first. It is benchmark-adjacent, but it is not yet a canonical harness-history lab in the same way as `labs/nvfp4_gemm` or `labs/nvfp4_group_gemm`.",
    ],
)

ENTRIES["labs/nvfp4_gemm"] = lab_entry(
    slug="labs/nvfp4_gemm",
    title="Lab - NVFP4 GEMM",
    summary=dedent(
        """\
        Benchmarks an NVFP4 GEMM kernel path against the higher-precision reference so you can measure what the precision/schedule tradeoff is actually buying."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Low-precision GEMM work can devolve into kernel folklore quickly. This lab keeps the question narrow: what does the NVFP4 path actually save on this shape family, and does it stay verification-clean?"""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - higher-precision or less-specialized GEMM reference
                - correctness anchor for the low-precision path
                - useful for measuring the real cost/benefit of NVFP4"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - NVFP4 GEMM kernel path
                - same benchmark contract, lower-precision execution
                - tuned to answer whether the precision/schedule tradeoff is worth it here"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `nvfp4_gemm` | `0.0189 ms` | `0.0128 ms` | `1.47x` |

                That is a healthy microbenchmark win, but still the kind of result that must stay verification-gated. This lab is here to make that discipline visible."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/nvfp4_gemm
                python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure the NVFP4 GEMM path under a strict verification contract.",
        "Keep low-precision wins attributable to a real benchmark pair instead of a submission-only script.",
        "Expose when the path regresses verification or only wins on one measurement surface.",
    ],
    contents=[
        ("`baseline_nvfp4_gemm.py`, `optimized_nvfp4_gemm.py`", "Harness entrypoints for the reference and NVFP4 paths."),
        ("`baseline_submission.py`, `optimized_submission.py`, `local_eval_*.py`", "Submission/evaluation helpers for the kernel lane."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the lab."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile minimal` should keep the optimized path faster and verification-clean on current hardware.",
    ],
    notes=[
        "The repo's NVFP4 labs are intentionally verification-heavy; a faster incorrect low-precision path is not an acceptable outcome.",
    ],
)

ENTRIES["labs/nvfp4_gemv"] = lab_entry(
    slug="labs/nvfp4_gemv",
    title="Lab - NVFP4 GEMV",
    summary=dedent(
        """\
        Challenge workspace for the GPUMODE NVFP4 GEMV problem with exact official leaderboard semantics preserved in a local evaluator."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Challenge workspaces need an honest measurement surface. This lab keeps the official leaderboard semantics explicit so the optimized route is judged against the real baseline, not a toy proxy."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - `baseline_submission.py`
                - official-parity local eval path
                - correctness and challenge-semantics anchor"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - `optimized_submission.py`
                - same official eval semantics and clock-locking path
                - challenge-oriented route with case-specific tuning"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative official-parity local results from `labs/nvfp4_gemv/official_eval_*_20260228.json`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `nvfp4_gemv` | `203.711 us` | `68.206 us` | `2.99x` |

                That is a real challenge-workspace win, and the important part is that it comes from the official evaluator semantics rather than from an easier local proxy."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json
                python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json
                ```

                The evaluator produces per-case means plus the official aggregate score, which is the right artifact to compare in this workspace."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json
                python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep the GEMV challenge workspace aligned with the official evaluator semantics.",
        "Make the baseline vs optimized submission delta visible with real stored artifacts.",
        "Prevent local-tuning wins from being claimed without official-parity evidence.",
    ],
    contents=[
        ("`baseline_submission.py`, `optimized_submission.py`, `reference_submission.py`", "Submission files for the challenge workspace."),
        ("`baseline_nvfp4_gemv.py`, `optimized_nvfp4_gemv.py`", "Wrapper files for benchmark-facing integration."),
        ("`local_eval.py`, `official_eval_baseline_20260228.json`, `official_eval_optimized_20260228.json`", "Official-parity local evaluator and stored result artifacts."),
        ("`task.py`, `utils.py`", "Challenge helpers and task definitions."),
    ],
    run=RunSection(
        commands=[
            "python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/baseline_submission.py --json",
            "python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --json",
        ],
        notes=[
            "Use module invocation; direct script invocation is now re-exec'd to the same module entrypoint for compatibility.",
            "When the Popcorn service is healthy, compare these local official-parity results against benchmark-mode submissions rather than against ad hoc local timings.",
        ],
    ),
    validation=[
        "The official-parity local evaluator should pass correctness and emit per-case timing plus aggregate score.",
        "Any promoted route should stay explainable in terms of the official evaluator output, not only custom local scripts.",
    ],
    notes=[
        "This lab already has a cleaner local evidence story than `nvfp4_dual_gemm` because the stored official baseline and optimized reports are checked in.",
    ],
)

ENTRIES["labs/nvfp4_group_gemm"] = lab_entry(
    slug="labs/nvfp4_group_gemm",
    title="Lab - NVFP4 Grouped GEMM",
    summary=dedent(
        """\
        Explores grouped-GEMM routing and schedule variants across multiple shapes so you can see where the grouped NVFP4 path is actually winning and where it is merely legal."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Grouped GEMM tuning is noisy and easy to overclaim. This lab keeps the shape routing explicit and benchmarked so promotions are based on repeated verified wins instead of one-off lows."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - canonical baseline front-door target for the promoted grouped shape
                - shape-specific baseline companions for the other workload shapes
                - useful for showing which grouped shapes are hard versus easy"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - canonical optimized front-door target for the promoted grouped shape
                - shape-specific tuned NVFP4 grouped GEMM variants
                - same grouped workloads, but explicit schedule/routing choices
                - designed to keep promotions tied to repeated verify and ABAB checks"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Fresh portable B200 single-target reruns on this host showed one clear winner, two smaller positive companions, and one flat control shape:

                | Target | Baseline | Optimized | Measured delta | Contract |
                | --- | ---: | ---: | ---: | --- |
                | `nvfp4_group_gemm` | `0.614 ms` | `0.595 ms` | `1.03x` | canonical local-contract speed benchmark |
                | `nvfp4_group_gemm_g2_n3072_k4096` | `0.615 ms` | `0.594 ms` | `1.04x` | promoted shape companion (same routed workload as the front-door target) |
                | `nvfp4_group_gemm_g8_n7168_k2048` | `2.080 ms` | `2.019 ms` | `1.03x` | supplementary comparison benchmark |
                | `nvfp4_group_gemm_g8_n4096_k7168` | `2.408 ms` | `2.379 ms` | `1.01x` | supplementary comparison benchmark |
                | `nvfp4_group_gemm_g2_n4096_k1536` | `0.352 ms` | `0.352 ms` | `1.00x` | supplementary comparison benchmark |

                The older strict all-case snapshots in `artifacts/runs/20260302_rerun_all_labschapters_strict/` are still useful historical router evidence, but they are not the current runnable truth for this harness surface on this host. The former competition `caseN` numbering is retired from the public benchmark targets; the canonical front-door target now points at the isolated single-target winner `g2_n3072_k4096`."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --single-gpu
                ```

                Use the harness artifacts for schedule attribution, then use the router/ABAB tooling for promotion decisions. The benchmark pair tells you the shape of the win; the tuning scripts decide whether a default should actually move."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
                python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep grouped-GEMM tuning grounded in repeated verified shape-by-shape evidence.",
        "Keep non-winning shapes visible as supplementary comparison benchmarks without forcing them to carry the lab's canonical speed claim on every host.",
        "Retire the old competition `caseN` labels from the public harness surface.",
    ],
    contents=[
        ("`baseline_nvfp4_group_gemm.py`, `optimized_nvfp4_group_gemm.py`", "Canonical front-door grouped-GEMM target for the promoted shape."),
        ("`baseline_nvfp4_group_gemm_g*.py`, `optimized_nvfp4_group_gemm_g*.py`", "Shape-specific grouped-GEMM companions for the remaining workload shapes."),
        ("`WORKLOG.md`, `custom_cuda_submission.py`, `cutlass_extension.py`", "Tuning log and implementation plumbing for the promoted routes."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/nvfp4_group_gemm:nvfp4_group_gemm --profile minimal` should keep the promoted `g2_n3072_k4096` route verification-clean.",
        "`nvfp4_group_gemm_g8_n4096_k7168`, `nvfp4_group_gemm_g8_n7168_k2048`, and `nvfp4_group_gemm_g2_n4096_k1536` remain supplementary comparison benchmarks; use the ABAB/router tooling when deciding whether any of them should become canonical speed-claim targets again.",
        "Old `caseN` target names should no longer appear in `python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm`.",
        "Default changes should still be gated by the stricter ABAB/verify process documented in the codebase notes, not by a single benchmark run.",
    ],
    notes=[
        "This lab is intentionally stricter than a normal benchmark pair because grouped-GEMM route tuning is unusually noise-prone.",
        "The benchmark harness now exposes one canonical front-door speed target, the explicit promoted-shape companion, and three supplementary comparison shapes on this host-aligned repo surface, because fresh portable B200 single-target reruns showed `g2_n3072_k4096` as the clearest isolated winner.",
    ],
)

ENTRIES["labs/trtllm_phi_3_5_moe"] = lab_entry(
    slug="labs/trtllm_phi_3_5_moe",
    title="Lab - TRT-LLM Phi-3.5 MoE",
    summary=dedent(
        """\
        Benchmarks a TensorRT-LLM style Phi-3.5 MoE serving path against a slower reference path so the repo has a measured inference-stack example, not just kernel-level microbenches."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                End-to-end serving optimizations are easy to misread because setup, engine build, and runtime execution all blur together. This lab keeps a reference path and an optimized TRT-LLM path in the same harness contract so the serving win is measurable."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - slower reference serving/inference path
                - stable end-to-end anchor for the optimized TRT-LLM route
                - useful for showing the cost of not using the optimized engine stack"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - TensorRT-LLM-oriented optimized serving path
                - same workload and verification contract
                - tuned to show the practical inference-stack win, not just a kernel-local result
                - verifies a deterministic generated-token prefix rather than the more fragile full-logits or full-suffix path"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated result from `artifacts/runs/20260303_trtllm_phi35moe_minimal_expectations_mixedprov_clean17/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `trtllm_phi_3_5_moe` | `9767.635 ms` | `1065.477 ms` | `9.17x` |

                That is a substantial end-to-end win, and it only became worth documenting after the earlier failure and verification-cleanup passes were sorted out. This is exactly why the repo keeps the validation history instead of hiding the false starts."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/trtllm_phi_3_5_moe
                python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Keep a real serving-stack optimization in the repo's benchmark story.",
        "Measure TRT-LLM value under the same harness discipline as the kernel labs.",
        "Make it clear when an optimized inference stack is really worth the complexity.",
    ],
    contents=[
        ("`baseline_trtllm_phi_3_5_moe.py`, `optimized_trtllm_phi_3_5_moe.py`", "Baseline and TensorRT-LLM benchmark entrypoints."),
        ("`trtllm_common.py`", "Shared helpers and workload setup for the pair."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the lab."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile minimal` should keep the optimized path verification-clean on the shared generated-token prefix and materially ahead.",
    ],
    notes=[
        'This lab is one of the best repo examples for "serving-stack optimization" as opposed to pure kernel tuning.',
    ],
)

ENTRIES["labs/vllm-deepseek-tuning"] = lab_entry(
    slug="labs/vllm-deepseek-tuning",
    title="Lab - vLLM DeepSeek Tuning Harness",
    summary=dedent(
        """\
        A matrix-driven tuning harness for DeepSeek + vLLM scenarios: scenario sweeps, plots, reports, and startup-failure capture. It is a comparison matrix, not a single honest baseline/optimized benchmark pair yet."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Large serving-stack comparisons rarely reduce to one "optimized" switch. TP/EP choices, MTP on/off, model family, and concurrency sweeps all matter, so this lab keeps the experiment matrix explicit instead of pretending there is one universal optimized path."""
            ),
        ),
        MarkdownSection(
            "What This Lab Is",
            dedent(
                """\
                - a matrix harness around `vllm serve` + `vllm bench serve`
                - scenario/config/report tooling
                - useful for comparative serving experiments and artifact generation

                It is not currently a clean benchmark-pair lab because the important comparisons are multiple named variants, not one generic baseline/optimized route."""
            ),
        ),
        MarkdownSection(
            "Current Artifact State",
            dedent(
                """\
                The checked-in artifact set under `results/` is currently startup-failure/report oriented, not a canonical baseline/optimized pair history. That is still valuable, but it should be described honestly.

                If we want benchmark-pair docs here later, we should first produce canonical successful runs for one or two concrete comparisons instead of extrapolating from the matrix harness."""
            ),
        ),
        MarkdownSection(
            "What Proper Benchmark Pairs Would Look Like",
            dedent(
                """\
                If we productize this into benchmark targets, the clean shape is to create explicit comparison pairs such as:

                - `baseline_vllm_deepseek_tp2.py` vs `optimized_vllm_deepseek_ep2.py`
                - `baseline_vllm_deepseek_mtp0.py` vs `optimized_vllm_deepseek_mtp1.py`

                Each pair would need fixed prompts, fixed ISL/OSL/concurrency, stable serve lifecycle handling, and a clear validation/report artifact contract. That is better than inventing one generic `optimized_vllm_deepseek.py` wrapper that hides what actually changed."""
            ),
        ),
    ],
    goals=[
        "Keep DeepSeek + vLLM comparison work reproducible and auditable.",
        "Separate matrix-harness experimentation from benchmark-pair performance claims.",
        "Provide a cleaner path to future canonical vLLM serving benchmark pairs.",
    ],
    contents=[
        ("`configs/benchmark_matrix.yaml`, `configs/smoke_tiny.yaml`", "Scenario matrices and smoke-test configs."),
        ("`scripts/run_matrix.py`, `scripts/plot_results.py`, `scripts/report_results.py`", "Serve/bench orchestration plus plotting/report generation."),
        ("`results/`, `plots/`, `reports/`", "Structured outputs from the matrix harness."),
        ("`Makefile`, `scripts/vllm_docker.sh`, `scripts/teardown.sh`", "Operational entrypoints for running and cleaning up the matrix."),
    ],
    run=RunSection(
        commands=[
            "cd labs/vllm-deepseek-tuning",
            "make smoke",
            "make full",
            "make artifacts",
        ],
        notes=[
            "This lab is Makefile/script driven today, not harness-target driven.",
            "Use Docker-backed `vllm` when host `torch` and host `vllm` are mismatched.",
        ],
    ),
    validation=[
        "`make smoke` should prove the orchestration path is alive before a full matrix run.",
        "`make full` should emit raw logs plus structured `results/*.json` records.",
        "`make artifacts` should regenerate plots and markdown/csv reports from collected results.",
        "If this lab is promoted into benchmark-pair targets later, create explicit named comparison pairs instead of a fake one-size-fits-all optimized wrapper.",
    ],
    notes=[
        "This README stays intentionally honest: useful matrix harness, not yet a canonical baseline/optimized benchmark lab.",
    ],
)

ENTRIES["labs/moe_decode_blackwell_matrix"] = lab_entry(
    slug="labs/moe_decode_blackwell_matrix",
    title="Lab - MoE Decode Blackwell Matrix",
    summary=dedent(
        """\
        A Blackwell-class MoE decode scenario matrix for routing locality, persistent scheduling, and graph-launch tradeoffs. This is the canonical public surface for the deck-aligned Blackwell MoE decode work."""
    ),
    lead_sections=[
        MarkdownSection(
            "Why This Exists",
            dedent(
                """\
                The source PDF in `.cursor/plans/` overlaps enough with the repo's existing MoE and decode material that a second public benchmark pair would be redundant. The more honest surface is a matrix/playbook lab that answers the actual tradeoff questions:

                - how routing locality changes the decode-step shape
                - when persistent scheduling helps
                - when graph replay is worth the setup cost

                This lab is the shared-doc Blackwell MoE decode entry."""
            ),
        ),
        MarkdownSection(
            "What This Lab Is",
            dedent(
                """\
                - a decode-step MoE scenario matrix for Blackwell-class GPUs
                - a playbook surface with named sweep presets and structured artifacts
                - focused on routing, scheduling, and launch tradeoffs rather than a single baseline/optimized claim

                It reuses the repo's existing MoE primitives instead of duplicating the staged MoE or decode benchmark stories."""
            ),
        ),
        MarkdownSection(
            "Why This Is Not A Benchmark Pair",
            dedent(
                """\
                The important comparisons here are multiple named variants, not one universal optimized path. Keeping this as a matrix lab avoids inventing a fake one-dimensional story.

                - `labs/moe_optimization_journey` is the staged MoE benchmark narrative.
                - `labs/decode_optimization` is the broader decode hot-path microbenchmark suite.
                - `labs/blackwell_gemm_optimizations` is the grouped expert / GEMM kernel journey.

                `labs/moe_decode_blackwell_matrix` stays at the scenario/playbook layer above those labs."""
            ),
        ),
        MarkdownSection(
            "What It Measures",
            dedent(
                """\
                Each matrix point runs a deterministic single-step decode-style MoE expert path and records:

                - per-step mean / stdev / p95 latency
                - tokens per second and dispatch-tokens per second
                - routing entropy, active-expert fraction, and max tokens per expert
                - CUDA-graph capture cost when graph mode is enabled
                - max absolute difference against the non-persistent grouped reference
                - locked application clocks and theoretical device ceilings"""
            ),
        ),
    ],
    goals=[
        "Keep the deck-aligned Blackwell MoE decode work visible without inventing a redundant benchmark pair.",
        "Make routing locality, persistent scheduling, and graph-launch tradeoffs reproducible and auditable.",
        "Give readers a MoE-specific decode scenario lab that is clearly distinct from the staged MoE, decode microbenchmark, and grouped-GEMM labs.",
    ],
    contents=[
        ("`matrix_types.py`, `matrix_catalog.py`", "Typed scenario definitions plus YAML playbook loading."),
        ("`preflight.py`, `runner.py`, `artifact_io.py`, `run_matrix.py`", "Preflight checks, deterministic decode-step execution, artifact writing, and the main matrix runner."),
        ("`playbooks/deck_matrix.yaml`, `playbooks/smoke_b200.yaml`", "Broad deck-style sweep plus fast local smoke preset."),
        ("`profiler/run_profile_compare.py`, `profiler/capture.py`, `profiler/compare.py`", "Profiler-backed graph-vs-eager comparison helpers for prior matrix runs."),
        ("`tests/test_matrix_lab.py`", "Lab-local pytest coverage for playbook loading, batch generation, and summary contracts."),
    ],
    run=RunSection(
        commands=[
            "python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook smoke_b200 --run-id smoke_20260320",
            "python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook deck_matrix --run-id deck_20260320",
        ],
        notes=[
            "Use `smoke_b200` first; it is the fast local evidence path for a Blackwell host.",
            "This lab is script-first, not harness-target driven.",
            "Use this lab when you want the scenario/playbook view of Blackwell MoE decode tradeoffs rather than a single benchmark-pair claim.",
        ],
    ),
    run_heading="Running The Matrix",
    run_intro="Use the matrix runner for scenario sweeps. Use the profiler helper when you want a concrete graph-vs-eager comparison from an existing run.",
    validation=[
        "`python -m compileall labs/moe_decode_blackwell_matrix` should succeed.",
        "`python -m pytest labs/moe_decode_blackwell_matrix/tests -q` should keep the playbook and summary contracts green.",
        "`python -m labs.moe_decode_blackwell_matrix.run_matrix --playbook smoke_b200 --run-id smoke_<date>` should emit a self-contained artifact directory under `artifacts/moe_decode_blackwell_matrix/runs/`.",
        "`python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare --run-dir artifacts/moe_decode_blackwell_matrix/runs/<RUN_ID>` should write trace outputs plus a structured comparison summary.",
    ],
    extra_sections=[
        dedent(
            """\
            ## Playbooks

            | Playbook | Intent |
            | --- | --- |
            | `deck_matrix` | Broad tradeoff sweep for the duplicated deck surface. |
            | `smoke_b200` | Fast local B200 smoke run with artifact generation and graph-vs-eager evidence. |
            """
        ),
        dedent(
            """\
            ## Artifact Layout

            Every matrix run writes a self-contained directory under `artifacts/moe_decode_blackwell_matrix/runs/<RUN_ID>/`:

            | Artifact | Contents |
            | --- | --- |
            | `sys_meta.json` | Playbook config, GPU metadata, profiler-tool visibility, lock state, and command. |
            | `matrix.jsonl` | One row per matrix point, including unsupported and error states. |
            | `matrix.csv` | Flat export for spreadsheet-style inspection. |
            | `summary.json` | Best overall point plus persistent-vs-dynamic and graph-vs-eager deltas. |
            | `manifest.json` | File hashes for reproducibility. |
            | `profiles/<name>/summary.json` | Profiler comparison metadata and deltas. |
            | `profiles/<name>/*.trace.json` | Chrome-trace exports from `torch.profiler`. |
            """
        ),
        dedent(
            """\
            ## Profiler-Backed Comparison

            ```bash
            python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare \
              --run-dir artifacts/moe_decode_blackwell_matrix/runs/smoke_20260320
            ```

            Or choose explicit config ids from `matrix.jsonl`:

            ```bash
            python -m labs.moe_decode_blackwell_matrix.profiler.run_profile_compare \
              --run-dir artifacts/moe_decode_blackwell_matrix/runs/smoke_20260320 \
              --config-a e8_k2_b8_stk_pst_egr \
              --config-b e8_k2_b8_stk_pst_grf
            ```
            """
        ),
    ],
    notes=[
        "`dynamic + cuda_graph` is emitted as `unsupported` on purpose; the lab makes that constraint visible instead of silently falling back.",
        "`sticky` routing is the locality-friendly control, while the other policies help explain when persistent scheduling does or does not pay off.",
        "This lab remains the matrix/playbook companion to `labs/moe_optimization_journey`, `labs/decode_optimization`, and `labs/blackwell_gemm_optimizations`, not a replacement for those benchmark stories.",
    ],
)

ENTRIES["labs/flexattention"] = lab_entry(
    slug="labs/flexattention",
    title="Lab - FlexAttention Harness",
    summary=dedent(
        """\
        Mirrors the FlexAttention CuTe DSL walkthrough: run eager vs compiled FlexAttention, compare to the CuTe path, and experiment with block masks, score modifiers, and Triton-style compilation."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                FlexAttention and FlashAttention-style paths are easy to describe and harder to verify. This lab is here to answer whether the compiled sparse/masked path in this repo actually beats the eager baseline on the same score modifiers and masks."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager FlexAttention path
                - straightforward correctness reference
                - higher Python and kernel-launch overhead"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - compiled FlexAttention path
                - same masks and score modifiers
                - tuned for fused execution and fewer launches"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `flex_attention` | `9.052 ms` | `0.320 ms` | `28.25x` |

                That is exactly why this lab is useful: it keeps the mask/score-mod path visible while still showing a very large compile/fusion win on the local stack."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile deep_dive --single-gpu
                python -m cli.aisp tools flex-attention-cute -- --batch 2 --seq-len 1024
                ```

                The harness run gives you the artifacted baseline/optimized pair. The CuTe tool is the useful fallback when you want to compare semantics on systems without working FlexAttention bindings."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/flexattention
                python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
                BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.",
        "Validate CuTe-based FlashAttention fallbacks for platforms where FlexAttention is not available.",
        "Sweep sparsity knobs (block size, doc span) without editing source.",
        "Collect Nsight traces showing kernel fusion improvements after compiling.",
    ],
    contents=[
        ("`baseline_flex_attention.py`, `optimized_flex_attention.py`", "FlexAttention DSL workloads toggling `torch.compile` for fused kernels."),
        ("`flex_attention_cute.py`", "CuTe/FlashAttention tool for hardware without FlexAttention bindings."),
        ("`flexattention_common.py`, `expectations_{hardware_key}.json`", "Shared input builders, score modifiers, and regression thresholds."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal` captures the eager vs compiled delta and stores artifacts.",
        "`BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention` demonstrates masked sparsity sweeps.",
        "`python -m cli.aisp tools flex-attention-cute -- --batch 2 --seq-len 1024` succeeds even on systems missing FlexAttention bindings.",
    ],
    notes=[
        "Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.",
        "Artifacts include NVTX traces; feed them to `core/analysis/deep_profiling_report.py` for convenience.",
    ],
)

ENTRIES["labs/fullstack_cluster"] = lab_entry(
    slug="labs/fullstack_cluster",
    title="Lab - Full-Stack Blackwell Cluster",
    summary=dedent(
        """\
        Replays the entire performance-engineering arc as scenarios: from system prep to streaming inference, plus the original cluster GEMM CUDA kernels wired into the harness."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                This lab is where the repo stops being a pile of isolated kernels and starts behaving like a system story. The important question is whether the end-to-end scenario kernels still show the same directional wins once you put them back into a larger flow."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - scenario kernels without Blackwell-specific cluster tuning
                - useful for a stable reference, but not a good steady-state throughput path
                - keeps the extension and harness wiring honest"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - optimized cluster GEMM variants
                - tcgen05 route where available
                - same harness contract and validation as the rest of the repo
                - canonical hybrid-EP comparisons now keep the same default routing mode, with topology-aware routing available only through an explicit override"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `cluster_gemm` | `29.213 ms` | `4.583 ms` | `6.37x` |
                | `cluster_gemm_tcgen05` | `0.240 ms` | `0.230 ms` | `1.04x` |

                The useful split here is that `cluster_gemm` demonstrates the big end-to-end kernel win, while `cluster_gemm_tcgen05` is the fine-grained tcgen05 follow-up where the remaining headroom is much smaller.

                Treat `cluster_gemm_tcgen05` as a supplementary comparison benchmark with a local contract. It is still useful for tcgen05 profiling and regression tracking, but the lab's canonical speed claim stays on `cluster_gemm`."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm_tcgen05 --profile deep_dive --single-gpu
                ```

                The tcgen05 path is worth profiling separately. Its win is much smaller than the coarse cluster GEMM delta, so Nsight evidence matters more than headline speedup alone."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/fullstack_cluster
                python -m cli.aisp bench run --targets labs/fullstack_cluster:moe_hybrid_ep --profile minimal
                python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm --profile minimal
                python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048
                ```"""
            ),
        ),
    ],
    goals=[
        "Run scenario benchmarks that stitch together chapters into end-to-end workflows.",
        "Compare a baseline vs topology-aware DeepSeek-style hybrid expert-parallel optimizer step.",
        "Inspect cluster GEMM kernels (baseline and DSMEM/TMA optimized) via the CUDA extension.",
        "Track GPU requirements, expected shapes, and automation scripts in one place.",
        "Collect artifact bundles that summarize every phase of the scenario.",
    ],
    contents=[
        ("`baseline_moe_hybrid_ep.py`, `optimized_moe_hybrid_ep.py`, `baseline_moe_hybrid_ep_multigpu.py`, `optimized_moe_hybrid_ep_multigpu.py`, `moe_hybrid_ep_common.py`", "DeepSeek-style hybrid EP optimizer-step benchmarks with explicit dispatch/combine phases, load-balance metrics, and intra-node fallback reporting."),
        ("`baseline_cluster_gemm.py`, `optimized_cluster_gemm.py`, `baseline_cluster_gemm_tcgen05.py`, `optimized_cluster_gemm_tcgen05.py`", "Python entrypoints for the cluster GEMM kernels with tcgen05 fallbacks."),
        ("`capstone_extension.py`, `capstone_extension_tcgen05.py`, `capstone_kernels.cu`, `capstone_kernels_tcgen05.cu`, `capstone_benchmarks.py`", "PyTorch extension, CUDA kernels, and harness hooks for the GEMM showcase."),
        ("`run_lab_fullstack_cluster.py`, `gpu_requirements.py`, `expectations_{hardware_key}.json`", "Standalone runner, hardware requirement helper, and expectation file."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/fullstack_cluster:moe_hybrid_ep --profile minimal` records a full optimizer step with routing/dispatch/combine/backward/grad-sync metrics.",
        "`python -m cli.aisp bench run --targets labs/fullstack_cluster --profile minimal` records per-phase metrics for the entire scenario suite.",
        "`python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048` builds the extension on first run and prints baseline vs optimized TFLOP/s.",
        "KF-specific kernels skip gracefully on hardware lacking tcgen05 or DSMEM, ensuring CI signal stays meaningful.",
    ],
    notes=[
        "`gpu_requirements.py` reports the minimum GPU count, memory, and features for each scenario; consult it before scheduling runs.",
        "`capstone_extension.py` caches builds under `~/.cache/torch_extensions`; run `python cleanup.py --include-extensions` when switching CUDA versions.",
        "Single-GPU `moe_hybrid_ep` benchmark runs measure `HybridEPTrainer.run_step()` in-process so the timing reflects the optimizer step instead of single-rank launcher overhead; multi-rank runs still use `torchrun`.",
        "Canonical hybrid-EP comparisons now keep the same default routing mode; use `--route-mode topology_aware` when you want that alternate behavior to be visible instead of relying on a silent default.",
        "`cluster_gemm_tcgen05` is a supplementary comparison benchmark with a local contract; use `cluster_gemm` when you want the lab's canonical cluster-GEMM speed claim.",
    ],
)

ENTRIES["labs/moe_cuda"] = lab_entry(
    slug="labs/moe_cuda",
    title="Lab - CUDA MoE Decode Toolkit",
    summary=dedent(
        """\
        Implements mixture-of-experts decode helpers directly in CUDA: decode kernels, KV-transfer overlap/graph variants, router policies, and validation math so you can iterate on Blackwell-friendly pipelines."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                MoE serving paths usually fail for one of four reasons: decode kernels are too launch-heavy, KV movement is too slow, backend selection is naïve, or routers do too much scalar work. This lab keeps those costs separated so you can see which one actually improved."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager CUDA helpers for decode, routing, and KV transfer
                - good correctness references
                - too much overhead in the hot path for steady-state serving"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - staged decode kernels
                - overlapped and graph-assisted KV transfer
                - backend and router kernels tuned for Blackwell-friendly execution"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `decode_attention` | `0.259 ms` | `0.207 ms` | `1.25x` |
                | `kv_transfer` | `1.224 ms` | `1.085 ms` | `1.13x` |
                | `kv_transfer_direct` | baseline transfer path | direct KV writes | destination-placement variant |
                | `kv_transfer_graphs` | `1.224 ms` | `0.315 ms` | `3.88x` |
                | `kv_transfer_direct_graphs` | baseline transfer path | direct KV writes + graph replay | destination + launch-removal variant |
                | `moe_backend_selection` | `1.747 ms` | `0.308 ms` | `5.67x` |
                | `router` | `67.265 ms` | `8.674 ms` | `7.75x` |

                That spread is the point of the lab. Not every MoE subsystem gets the same win: overlap-only KV transfer is a modest directional step, direct destination placement removes the copy when the serving layout allows it, and graphed direct placement removes most of the remaining launch overhead. The router/backend work is still where the biggest local payoff is showing up."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/moe_cuda:decode_attention --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/moe_cuda:kv_transfer_direct_graphs --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/moe_cuda:router_vectorized --profile deep_dive --single-gpu
                ```

                Those three targets cover the highest-value slices: decode kernel efficiency, KV movement/orchestration, and router kernel behavior."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/moe_cuda
                python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal
                python -m cli.aisp bench verify -t labs/moe_cuda:decode_attention
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark decode kernels that stage tokens through shared memory and cp.async pipelines.",
        "Optimize KV-transfer strategies (manual, CUDA Graphs) across NVLink fabrics.",
        "Prototype routers that understand MoE grouping, locality, and vectorized loads.",
        "Validate CUDA kernels against Python math models before integrating into serving stacks.",
    ],
    contents=[
        ("`baseline_decode_attention.py`, `optimized_decode_attention.py`", "Attention microbenchmarks that validate correctness while optimizing kernel schedules."),
        ("`baseline_decode_kernel.py`, `optimized_decode_kernel.py`, `decode_kernels.py`, `kernels/`", "CUDA kernels and wrappers for the decode core."),
        ("`baseline_kv_transfer.py`, `baseline_kv_transfer_direct.py`, `baseline_kv_transfer_direct_graphs.py`, `optimized_kv_transfer.py`, `optimized_kv_transfer_direct.py`, `optimized_kv_transfer_graphs.py`, `optimized_kv_transfer_direct_graphs.py`", "KV-transfer samples comparing eager copies, overlap, direct destination placement, and CUDA Graph orchestration."),
        ("`baseline_router.py`, `optimized_router.py`, `optimized_router_vectorized.py`", "MoE router logic fit for device execution."),
        ("`expectations_{hardware_key}.json`, `__init__.py`", "Metadata and module exports needed by the harness."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal` runs every baseline/optimized pair and captures NVTX traces.",
        "`python -m cli.aisp bench verify -t labs/moe_cuda:decode_attention` compares the CUDA path to the math reference and fails loudly if drift is detected.",
        "KV transfer graphs print latency breakdowns showing overlap improvements relative to the baseline script.",
    ],
    notes=[
        "`kernels/` houses the raw CUDA sources split by component; edit schedules there before rebuilding via the harness.",
        "`optimized_kv_transfer_graphs.py` and `optimized_kv_transfer_direct_graphs.py` emit CUDA Graph captures under `artifacts/` for reproducibility.",
    ],
)

ENTRIES["labs/moe_parallelism"] = lab_entry(
    slug="labs/moe_parallelism",
    title="Lab - MoE Parallelism Planner",
    summary=dedent(
        """\
        Scenario planning tool for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules."""
    ),
    goals=[
        "Quantify memory budgets for experts, routers, and KV caches before deploying models.",
        "Explore different grouping strategies (hashing, topology-aware) and their throughput impact.",
        "Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.",
        "Simulate pipeline schedules to identify bottlenecks before touching production systems.",
    ],
    contents=[
        ("`run_lab.py`, `scenarios.py`, `plan.py`", "Tool entry point + canonical scenario definitions and sizing model."),
        ("`benchmarking.py`", "Optional harness-compatible wrapper for ad-hoc integration; not currently a public baseline/optimized benchmark pair."),
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp tools moe-parallelism -- --scenario memory_budget",
            "python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200",
            "python labs/moe_parallelism/run_lab.py --scenario deepseek_gb200",
        ],
        notes=[
            "`python -m cli.aisp bench list-targets --chapter labs/moe_parallelism` intentionally returns no benchmark pairs today.",
            "If this lab is later promoted into harness targets, add explicit `baseline_*.py` and `optimized_*.py` entrypoints instead of implying them in the README.",
        ],
    ),
    validation=[
        "`python -m cli.aisp tools moe-parallelism -- --scenario memory_budget` runs a single scenario via the tool registry.",
        "`python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200` runs a larger cluster scenario.",
        "`python labs/moe_parallelism/run_lab.py --scenario deepseek_gb200` runs the planner directly (without aisp).",
    ],
    run_heading="Running the Tool",
    run_intro=(
        "Use the tool entrypoint or the direct script when you want reproducible "
        "scenario comparisons. This lab does not currently expose public "
        "baseline/optimized harness targets."
    ),
    notes=[
        "Baseline vs optimized here are *planning* scenarios (different designs), not comparable performance benchmarks.",
        "`plan.py` centralizes scenario definitions so you only update one file when adding a new topology.",
    ],
)

ENTRIES["labs/uma_memory"] = lab_entry(
    slug="labs/uma_memory",
    title="Lab - UMA Memory Diagnostics",
    summary=dedent(
        """\
        Diagnostics for UMA / unified-memory systems: capture device-visible free memory, host reclaimable memory, and JSON snapshots before you make claims about allocator behavior or memory-fit limits."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                On UMA-capable systems, a "GPU memory" question is often really a host-memory and reclaimability question. This lab exists to make that boundary explicit instead of inferring it from a single `nvidia-smi` or `cudaMemGetInfo()` number."""
            ),
        ),
        MarkdownSection(
            "What This Lab Is",
            dedent(
                """\
                - a diagnostics/tool lab
                - human-readable reporting plus JSON snapshots
                - useful before larger UMA, memory-fit, or allocation experiments

                It is **not** currently a baseline/optimized benchmark pair, and discovery intentionally reports zero harness targets here."""
            ),
        ),
    ],
    goals=[
        "Measure device-free memory and host-available memory together.",
        "Estimate UMA allocatable capacity with an explicit reclaim assumption.",
        "Capture reproducible snapshots before and after runtime, allocator, or workload changes.",
    ],
    contents=[
        ("`uma_memory_reporting.py`", "Main reporting tool that prints a human-readable report and can emit JSON snapshots."),
        ("`uma_memory_utils.py`", "Helpers for parsing `/proc/meminfo`, formatting byte counts, and detecting integrated GPUs."),
        ("`__init__.py`", "Package marker for the lab/tool module."),
    ],
    run=RunSection(
        commands=[
            "python -m cli.aisp tools uma-memory -- --device-index 0",
            "python -m cli.aisp tools uma-memory -- --device-index 0 --snapshot --snapshot-dir artifacts/uma_memory_snapshots",
            "python labs/uma_memory/uma_memory_reporting.py --json --device-index 0",
        ],
        notes=[
            "This lab is tool-driven, not benchmark-pair driven.",
            "Snapshot JSON is the artifact to compare across allocator, driver, or system-configuration changes.",
        ],
    ),
    validation=[
        "`python -m cli.aisp tools uma-memory -- --device-index 0` prints a readable summary without requiring manual parsing.",
        "`python -m cli.aisp tools uma-memory -- --device-index 0 --snapshot` writes a structured JSON artifact under `artifacts/uma_memory_snapshots/`.",
        "Direct-script output and tool-registry output should agree for the same `--device-index` and reclaim settings.",
    ],
    run_heading="Running the Tool",
    run_intro=(
        "Use the tool entrypoint for standard reporting or call the script "
        "directly when iterating locally."
    ),
    notes=[
        "The tool combines `torch.cuda.mem_get_info()` with `/proc/meminfo` so the report stays explicit about what is device memory vs host-side reclaimability.",
        "Use this as environment evidence for chapters/labs that discuss UMA or memory fit; it is not a substitute for workload benchmarks.",
    ],
)

ENTRIES["labs/occupancy_tuning"] = lab_entry(
    slug="labs/occupancy_tuning",
    title="Lab - Triton Occupancy & Schedule Sweep",
    summary=dedent(
        """\
        Sweeps Triton matmul schedules for ProtonNet-style workloads on Blackwell, comparing the baseline schedule against optimized block/warp dimensions and reporting how each choice affects occupancy and FLOP/s."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Occupancy work is easy to oversell. This lab exists to measure schedule choices directly and show whether better resident work actually lands a throughput win on the same matmul workload."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - one baseline Triton schedule
                - stable correctness and a clean occupancy reference
                - not tuned for this GPU/shape family"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - curated block/warp schedule variants
                - measured through the same harness contract
                - designed to answer "which schedule is actually best here?" instead of assuming bigger blocks win"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `proton_matmul` baseline | `0.251 ms` | `0.197 ms` (`bm64_bn256_bk32`) | `1.28x` |
                | `proton_matmul` baseline | `0.251 ms` | `0.206 ms` (`bm128_bn256_bk64`) | `1.22x` |

                The lab is valuable because it keeps the schedule sweep honest. The win is real, but it is a schedule-selection win, not magic.

                Keep `proton_matmul_bm64_bn64_bk32_nw2` as a supplementary local-contract schedule benchmark. It also remains a supplementary comparison schedule benchmark for low-warp occupancy and Proton-vs-Nsight agreement checks, while the canonical speed claims stay on `proton_matmul`, `proton_matmul_bm64_bn256_bk32`, `proton_matmul_bm128_bn128_bk32_nw8`, and `proton_matmul_bm128_bn256_bk64`."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/occupancy_tuning:proton_matmul --profile deep_dive --single-gpu
                python labs/occupancy_tuning/sweep_schedules.py --output artifacts/occupancy_tuning.csv
                ```

                Use the deep-dive harness run for Nsight evidence and the sweep script when you want to explore candidate schedules before promoting one into the benchmark pair."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/occupancy_tuning
                python -m cli.aisp bench run --targets labs/occupancy_tuning:proton_matmul --profile minimal
                python labs/occupancy_tuning/sweep_schedules.py --output artifacts/occupancy_tuning.csv
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure how Triton block sizes map to achieved occupancy on SM100/121.",
        "Autogenerate schedule sweeps and record best-performing parameter sets.",
        "Compare baseline schedules to curated optimized variants packaged with the lab.",
        "Integrate selected schedules into harness targets for regression tracking.",
    ],
    contents=[
        ("`baseline_proton_matmul.py`, `optimized_proton_matmul_bm128_bn128_bk32_nw8.py`, `optimized_proton_matmul_bm64_bn64_bk32_nw2.py`, `optimized_proton_matmul_bm64_bn256_bk32.py`, `optimized_proton_matmul_bm128_bn256_bk64.py`", "Baseline and optimized Triton schedules covering multiple block/warp configurations."),
        ("`triton_matmul.py`, `triton_matmul_schedules.py`", "Core Triton kernel and schedule definitions used by the harness."),
        ("`sweep_schedules.py`", "Utility for enumerating candidate schedules and logging throughput/occupancy to `artifacts/`."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/occupancy_tuning --profile minimal` executes every schedule defined in the lab.",
        "`python labs/occupancy_tuning/sweep_schedules.py --output artifacts/occupancy_tuning.csv` enumerates schedules and highlights the top performer.",
        "`python labs/occupancy_tuning/optimized_proton_matmul_bm128_bn128_bk32_nw8.py --validate` compares outputs against the baseline to ensure correctness.",
    ],
    notes=[
        "Add new schedules to `triton_matmul_schedules.py` and regenerate the harness targets by rerunning the sweep script.",
        "`proton_matmul_bm64_bn64_bk32_nw2` is a supplementary local-contract schedule benchmark and supplementary comparison schedule benchmark; use the larger winning schedules when you want the lab's canonical speed claims.",
        "`expectations_{hardware_key}.json` records FLOP/s per schedule so improvements show up in CI.",
    ],
)

ENTRIES["labs/speculative_decode"] = lab_entry(
    slug="labs/speculative_decode",
    title="Lab - Speculative Decoding",
    summary=dedent(
        """\
        Accelerates autoregressive generation by letting a smaller draft model propose multiple tokens that the larger target model verifies in parallel."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Speculative decoding only pays off when the draft model is accurate enough that verification overhead is amortized. This lab keeps that tradeoff explicit instead of treating speculation as a guaranteed win."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - target-only greedy decode
                - simple reference for latency and correctness
                - no draft-model parallelism"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - small draft model proposes multiple tokens per round
                - target model verifies the draft batch in parallel
                - rejection/correction logic preserves exactness of the target path"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `speculative_decode` | `105.903 ms` | `34.399 ms` | `3.08x` |

                That result is why the lab matters: speculation is only interesting when the acceptance rate is high enough to beat the verification cost, and this benchmark pair makes that visible on a deterministic toy-model setup."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile deep_dive --single-gpu
                ```

                The profiler view is useful here because it shows whether the runtime really shifted work into fewer target-model verification steps instead of just moving cost around."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/speculative_decode
                python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Measure how draft length and acceptance rate combine into real speedup on a deterministic workload.",
        "Keep the draft/target comparison exact enough that verification still means something.",
        "Demonstrate when speculative decoding is helpful and when it is not.",
    ],
    contents=[
        ("`baseline_speculative_decode.py`", "Target-only greedy decode baseline."),
        ("`optimized_speculative_decode.py`", "Draft proposals plus batched target verification."),
        ("`baseline_speculative_decode_trusted.py`, `optimized_speculative_decode_trusted.py`", "Trusted-draft serving variant that compares verified speculative decode with direct draft-token placement."),
        ("`baseline_speculative_decode_transition_table.py`, `optimized_speculative_decode_transition_table.py`", "Token-local transition cache variant that removes draft-model calls from repeated trusted decode."),
        ("`speculative_decode_common.py`", "Toy-model helpers and workload setup used by both paths."),
        ("`expectations_{hardware_key}.json`", "Regression thresholds for the benchmark pair."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile minimal` should show lower end-to-end decode latency for the optimized path.",
        "The optimized path should remain verification-clean against the target-only reference path.",
    ],
    notes=[
        "The lab uses a token-local `TokenMLP` so the benchmark stays deterministic and focused on speculative-decoding mechanics instead of model-download/setup noise.",
        "This is a good lab for studying acceptance-rate sensitivity before trying the same idea in a full serving stack.",
    ],
)

ENTRIES["labs/persistent_decode"] = lab_entry(
    slug="labs/persistent_decode",
    title="Lab - Persistent Decode & TMA Prefill",
    summary=dedent(
        """\
        Demonstrates Blackwell-friendly persistent decode kernels and TMA-powered prefill paths, all validated via Python harnesses plus CUDA/Triton implementations."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            "Decode and prefill paths often die by launch overhead, staging overhead, or both. This lab exists to show which of those costs persistent kernels, CUDA Graphs, and TMA actually remove on the same logical workload.",
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - naive decode loops and non-persistent prefill paths
                - higher launch overhead
                - less efficient staging into shared memory"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - persistent decode kernels
                - CUDA Graph replay where it helps
                - TMA-powered prefill variants for lower staging cost"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative validated results from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

                | Target | Baseline | Optimized | Measured delta | Best optimization |
                | --- | ---: | ---: | ---: | --- |
                | `persistent_decode` | `1.411 ms` | `0.118 ms` | `11.94x` | `graphs` |
                | `tma_prefill_decode` | `1.588 ms` | `0.931 ms` | `1.71x` | `optimized_tma_prefill_decode` |

                The decode win is a launch-overhead story. The prefill win is a staging/data-movement story. This lab is more useful when you keep those two categories separate.

                The direct transport swaps stay visible as a transport-comparison benchmark on `nvlink_offload` and as `paged_kv_offload` as a real speed benchmark with a small local contract. The canonical KV-offload overlap claim stays on `paged_kv_offload_prefetch`, where async prefetch materially changes the overlap story instead of only swapping host-transport mechanics."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use deep-dive runs when you want to see launch count and staging behavior instead of only the wall-clock delta:

                ```bash
                python -m cli.aisp bench run --targets labs/persistent_decode:persistent_decode --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/persistent_decode:tma_prefill_decode --profile deep_dive --single-gpu
                ```"""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/persistent_decode
                python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal
                python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50
                ```"""
            ),
        ),
    ],
    goals=[
        "Contrast naive decode loops against persistent kernels that pin CTAs per sequence.",
        "Adopt TMA-based prefill to stream activations into shared memory with minimal latency.",
        "Benchmark CUDA vs Triton implementations with unified validation utilities.",
        "Mix CUDA Graphs into the decode path to remove residual launch overhead.",
        "Compare pinned direct H2D staging against async prefetch overlap for paged KV offload.",
    ],
    contents=[
        ("`baseline_persistent_decode.py`, `optimized_persistent_decode_cuda.py`, `optimized_persistent_decode_graphs.py`, `optimized_persistent_decode_triton.py`", "Persistent decode variants spanning CUDA, graphs, and Triton."),
        ("`baseline_tma_prefill_decode.py`, `optimized_tma_prefill_decode.py`, `baseline_native_tma_prefill_decode.py`, `optimized_native_tma_prefill_decode.py`", "Prefill workloads illustrating cp.async vs native TMA scheduling."),
        ("`baseline_paged_kv_offload.py`, `optimized_paged_kv_offload.py`, `baseline_paged_kv_offload_prefetch.py`, `optimized_paged_kv_offload_prefetch.py`", "KV offload comparisons (pinned direct H2D with memmap, plus async prefetch on pinned host cache)."),
        ("`core/scripts/kv_locality_microbench.py`", "Pinned/pageable/NUMA host slab copy microbench (HBM vs local/remote pinned vs pageable)."),
        ("`persistent_decode_common.py`, `tma_extension.py`, `expectations_{hardware_key}.json`", "Shared helpers, CUDA extension wrappers, and expectation thresholds."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal` compares all persistent/TMA variants in one sweep.",
        "`python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50` shows lower launch overhead than `baseline_persistent_decode.py`.",
        "`python labs/persistent_decode/optimized_native_tma_prefill_decode.py --validate` matches the math reference while reporting achieved memory throughput.",
        "`python core/scripts/kv_locality_microbench.py` surfaces H2D copy time deltas for pageable vs pinned slabs; add `QUICK=1` for a short run.",
    ],
    notes=[
        "Set `TORCH_COMPILE_MODE` or `TMA_TILE_SIZE` via env vars before invoking the harness to sweep tile sizes.",
        "`tma_extension.py` caches builds under `~/.cache/torch_extensions`; clean the cache when switching CUDA versions.",
        "`nvlink_offload` remains a transport-comparison benchmark and `paged_kv_offload` stays a real speed benchmark with a small local contract; use `paged_kv_offload_prefetch` when you want the lab's canonical KV-offload overlap benchmark.",
    ],
)

ENTRIES["labs/parameterized_cuda_graphs"] = lab_entry(
    slug="labs/parameterized_cuda_graphs",
    title="Lab - Parameterized CUDA Graph Launch",
    summary=dedent(
        """\
        A narrow PyTorch-first benchmark pair for parameterized CUDA Graph replay: fixed shapes, fixed device buffers, and fixed graph topology, with request-specific host bindings updated on the executable graph before replay."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Normal CUDA Graph capture wants stable addresses. Real serving loops often keep tensor shapes static while rotating through different request slots, scalar values, and output buffers. If the graph is recaptured every time those bindings change, capture and instantiation overhead can erase the benefit."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - eager-style warmup followed by per-request graph recapture
                - fresh capture and instantiation for each request slot
                - same math as the optimized path, but full recapture tax stays in the hot path"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - one `torch.cuda.CUDAGraph(keep_graph=True)` capture on stable device buffers
                - executable memcpy-node parameter updates for the input slot, scalar slot, and output slot
                - graph replay without changing shapes, weights, or graph topology"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative local strict result from `artifacts/runs/20260320_parameterized_cuda_graphs_local/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `parameterized_graph_launch` | `0.938 ms` | `0.096 ms` | `9.74x` |

                This run was verification-clean and used harness clock locking, but the host was also flagged as virtualized. Treat it as solid local evidence for the parameterized replay technique, not as a publish-grade bare-metal number."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                Use the same target with a profiler-backed run when you want Nsight artifacts instead of only the wall-clock delta:

                ```bash
                python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile deep_dive --single-gpu
                ```

                The thing to look for is unchanged compute and kernels with the recapture overhead removed from the optimized path."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/parameterized_cuda_graphs
                python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile none --single-gpu --gpu-sm-clock-mhz 1500
                ```"""
            ),
        ),
    ],
    goals=[
        "Show a clean benchmark pair for graph recapture versus executable-graph parameter mutation.",
        "Keep the claim narrow: stable-shape PyTorch work with changing request bindings, not generic dynamic-shape graphing.",
        "Make the measured artifact prove that replay stays correct while per-request graph construction disappears from the hot path.",
    ],
    contents=[
        ("`baseline_parameterized_graph_launch.py`, `optimized_parameterized_graph_launch.py`", "Benchmark wrappers for per-request recapture versus parameterized executable-graph replay."),
        ("`parameterized_cuda_graphs_common.py`", "Shared PyTorch block, request-slot buffers, graph capture, memcpy-node mutation, and verification helpers."),
        ("`__init__.py`", "Package export for the shared benchmark classes and config."),
    ],
    validation=[
        "`python -m cli.aisp bench list-targets --chapter labs/parameterized_cuda_graphs` should discover `parameterized_graph_launch`.",
        "`python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile none --single-gpu --gpu-sm-clock-mhz 1500` should keep the optimized path verification-clean and faster than the recapture baseline.",
        "The optimized path should report executable memcpy-node parameter updates while avoiding per-iteration graph recapture.",
    ],
    notes=[
        "This lab overlaps conceptually with `labs/persistent_decode` and `labs/decode_optimization`, but it is intentionally the narrower companion focused on executable-graph parameter mutation in PyTorch.",
        "The current measured delta is backed by a local virtualized B200 run; replace the artifact when a canonical bare-metal run is collected.",
        "Optional local shape sweeps use `AISP_PARAMETERIZED_CUDA_GRAPHS_*` environment variables without changing the core fixed-shape benchmark contract.",
    ],
)

ENTRIES["labs/real_world_models"] = lab_entry(
    slug="labs/real_world_models",
    title="Lab - Real-World Model Optimizations",
    summary=dedent(
        """\
        Applies the course-wide optimization patterns to representative models (Llama 3.1 8B, DeepSeek-R1 MoE, GPT-4-style) so you can practice end-to-end tuning on Blackwell and Grace-Blackwell hardware."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            "Microbenchmarks are useful, but they can hide whether the repo's optimizations still matter on a real model path. This lab is the end-to-end check.",
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - representative model skeletons with conservative serving/training defaults
                - enough realism to surface KV-cache, routing, and compile effects
                - intentionally simpler than production deployment code so the optimization deltas stay readable"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - torch.compile and fused attention where they help
                - topology-aware and memory-aware configuration choices
                - the same benchmark harness contract as the lower-level labs"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Current validated result from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `llama_3_1_8b` | `13.143 ms` | `5.274 ms` | `2.49x` |

                This is the right lab to use when you want to sanity-check that the lower-level wins still add up on a model-shaped workload."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile deep_dive --single-gpu
                ```

                That path keeps the same evidence model as the rest of the repo: baseline/optimized timing, validation, and profiler artifacts in one run tree."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/real_world_models
                python -m cli.aisp bench run --targets labs/real_world_models --profile minimal
                python labs/real_world_models/llama_3_1_8b_optimization.py --seq-length 8192 --use-compile
                ```"""
            ),
        ),
    ],
    goals=[
        "Exercise attention, MoE, and memory optimizations on realistic architectures instead of toy kernels.",
        "Use the benchmark harness to collect reproducible throughput/latency metrics across models.",
        "Track expert balance, routing entropy, and KV-cache pressure while iterating on serving choices.",
        "Compare FP8/FP16, torch.compile, and topology-aware placements without changing source code.",
    ],
    contents=[
        ("`llama_3_1_8b_optimization.py`", "Single-node 8B walkthrough with `torch.compile`, FlexAttention, and Flash SDPA toggles."),
        ("`deepseek_r1_moe_optimization.py`", "64-expert top-6 routing demo with balance/Gini/entropy metrics and auxiliary loss."),
        ("`gpt4_architecture_optimization.py`", "GPT-4-style MoE + context-parallel sketch with FP8 support and memory estimation."),
        ("`__init__.py`", "Exports harness targets for the CLI."),
    ],
    run=RunSection(
        commands=[
            "cd ai-performance-engineering",
            "python -m cli.aisp bench list-targets --chapter labs/real_world_models",
            "python -m cli.aisp bench run --targets labs/real_world_models --profile minimal",
            "# Direct runs",
            "python labs/real_world_models/llama_3_1_8b_optimization.py --seq-length 8192 --use-compile",
            "python labs/real_world_models/deepseek_r1_moe_optimization.py --num-experts 64 --top-k 6 --batch-size 4",
            "python labs/real_world_models/gpt4_architecture_optimization.py --seq-length 8192 --context-parallel",
        ],
        notes=[
            "Override per-model flags via `--target-extra-arg labs/real_world_models:<target>=\"--flag value\"` when using the harness.",
        ],
    ),
    validation=[
        "`llama_3_1_8b_optimization.py` sustains ~20K tokens/sec on B200 with `--use-compile` enabled and stays memory-efficient at 8K+ context.",
        "`deepseek_r1_moe_optimization.py` reports balanced experts (Gini < 0.2) and stable router entropy across batches.",
        "`gpt4_architecture_optimization.py` runs the context-parallel path without OOM on appropriately sized clusters; memory estimates match the printed budget.",
        "Harness runs emit comparable baseline/optimized timings for every target without manual wiring.",
    ],
    notes=[
        "These scripts are intentionally weight-light sketches for benchmarking; swap in real checkpoints to validate production settings.",
        "Hardware expectations: B200/GB200 for best results; GPT-4-scale examples assume 24+ GPUs with NVLink/NVL fabrics.",
        "Metrics (balance loss, entropy, KV cache) are emitted alongside throughput so you can gate deployments with more than raw speed.",
    ],
)

ENTRIES["labs/train_distributed"] = lab_entry(
    slug="labs/train_distributed",
    title="Lab - Distributed Training Playbook",
    summary=dedent(
        """\
        Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness."""
    ),
    lead_sections=[
        MarkdownSection(
            "Problem",
            dedent(
                """\
                Distributed training has too many "optimized" labels that mean different things. This lab is here to keep DDP compression, pipeline schedules, and symmetric-memory training as separate benchmarked choices so you can see what actually helps on the current stack."""
            ),
        ),
        MarkdownSection(
            "Baseline Path",
            dedent(
                """\
                - conservative DDP, pipeline, and symmetric-memory paths
                - useful for correctness and topology sanity
                - enough communication overhead to make overlap/compression visible"""
            ),
        ),
        MarkdownSection(
            "Optimized Path",
            dedent(
                """\
                - overlap-aware pipeline schedules
                - compression-aware DDP variants
                - symmetric-memory and sharding strategies run through the same harness"""
            ),
        ),
        MarkdownSection(
            "Measured Delta",
            dedent(
                """\
                Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

                | Target | Baseline | Optimized | Measured delta |
                | --- | ---: | ---: | ---: |
                | `ddp_compression` | `1135.768 ms` | `408.656 ms` (`powersgd`) | `2.78x` |
                | `pipeline_1f1b` | `159.060 ms` | `105.125 ms` | `1.51x` |
                | `pipeline_dualpipe` | `154.106 ms` | `105.111 ms` | `1.47x` |
                | `symmem_training` | `177.269 ms` | `167.167 ms` | `1.06x` |

                The useful point is that the lab shows more than one kind of "distributed optimization." Compression and pipeline scheduling move the needle more than the current symmetric-memory path on this local setup.

                Treat single-GPU `fsdp2` on `b200` as a supplementary comparison surface with a local comparison contract. The real FSDP2 speed gate stays on the multi-GPU `2x_b200` contract where sharding and overlap can actually change the communication story."""
            ),
        ),
        MarkdownSection(
            "Profiler Evidence",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench run --targets labs/train_distributed:ddp_compression --profile deep_dive --single-gpu
                python -m cli.aisp bench run --targets labs/train_distributed:pipeline_1f1b --profile deep_dive --single-gpu
                ```

                For the multi-GPU variants, keep using `torchrun` through the lab utilities. The single-GPU harness targets are the evidence-first entrypoint, not a replacement for real cluster validation."""
            ),
        ),
        MarkdownSection(
            "Repro Commands",
            dedent(
                """\
                ```bash
                python -m cli.aisp bench list-targets --chapter labs/train_distributed
                python -m cli.aisp bench run --targets labs/train_distributed:ddp_compression --profile minimal
                python -m cli.aisp bench run --targets labs/train_distributed:pipeline_1f1b --profile minimal
                ```"""
            ),
        ),
    ],
    goals=[
        "Benchmark standard DDP vs optimized overlap-aware variants.",
        "Exercise FSDP and ZeRO strategies with shared helper utilities.",
        "Validate symmetric-memory training modes that pool NVLink bandwidth.",
        "Reuse launcher utilities (torchrun) with consistent configuration.",
    ],
    contents=[
        ("`baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `baseline_ddp_multigpu.py`, `optimized_ddp_multigpu.py`, `baseline_ddp_flash_multigpu.py`, `optimized_ddp_flash_multigpu.py`, `baseline_ddp_compression_multigpu_int8.py`, `optimized_ddp_compression_multigpu_int8.py`, `baseline_ddp_compression_multigpu_powersgd.py`, `optimized_ddp_compression_multigpu_powersgd.py`, `ddp.py`", "DDP workloads including flash-attention and compression variants (single + multi GPU)."),
        ("`baseline_fsdp.py`, `optimized_fsdp.py`, `baseline_fsdp_multigpu.py`, `optimized_fsdp_multigpu.py`, `baseline_fsdp2.py`, `optimized_fsdp2.py`, `baseline_fsdp2_multigpu.py`, `optimized_fsdp2_multigpu.py`, `train_fsdp.py`, `train_fsdp2.py`", "FSDP/FSDP2 scripts that demonstrate shard-by-shard memory savings."),
        ("`baseline_pipeline_1f1b.py`, `optimized_pipeline_1f1b.py`, `baseline_pipeline_gpipe.py`, `optimized_pipeline_gpipe.py`, `baseline_pipeline_dualpipe.py`, `optimized_pipeline_dualpipe.py`, `baseline_pipeline_dualpipev.py`, `optimized_pipeline_dualpipev.py`, `baseline_pipeline_1f1b_multigpu.py`, `optimized_pipeline_1f1b_multigpu.py`, `baseline_pipeline_gpipe_multigpu.py`, `optimized_pipeline_gpipe_multigpu.py`, `baseline_pipeline_1f1b_to_gpipe_multigpu.py`, `optimized_pipeline_1f1b_to_gpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipe_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipev_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipev_multigpu.py`, `baseline_pipeline_dualpipe_multigpu.py`, `optimized_pipeline_dualpipe_multigpu.py`, `baseline_pipeline_dualpipev_multigpu.py`, `optimized_pipeline_dualpipev_multigpu.py`, `pipeline_*.py`", "Pipeline parallelism schedules (single GPU simulations + multi-GPU execution)."),
        ("`baseline_symmem_training.py`, `optimized_symmem_training.py`, `baseline_symmem_training_multigpu.py`, `optimized_symmem_training_multigpu.py`", "Symmetric-memory strategies for optimizer state replication."),
        ("`baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `baseline_zero1_multigpu.py`, `baseline_zero2_multigpu.py`, `baseline_zero3_multigpu.py`, `optimized_zero1_multigpu.py`, `optimized_zero2_multigpu.py`, `optimized_zero3_multigpu.py`, `zero1.py`, `zero2.py`, `zero3.py`", "ZeRO implementations (1/2/3) plus helpers for parameter partitioning."),
        ("`training_utils/`, `utils.py`, `__init__.py`", "Shared launch utilities, argument parsing, and harness exports."),
    ],
    validation=[
        "`python -m cli.aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.",
        "`python labs/train_distributed/train_fsdp.py --validate` confirms numerical parity between FSDP shards and the baseline DDP path.",
        "`python labs/train_distributed/optimized_zero3_multigpu.py --summary` shows reduced peak memory vs the baseline script.",
    ],
    notes=[
        "Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.",
        "`utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.",
        "FSDP/FSDP2 benchmarks default to `labs/train_distributed/data/tinystories_packed_seq128.jsonl` plus `labs/train_distributed/data/tinyllama_config.json`, with `AISP_TINYSTORIES_LAYERS=4` to keep the model small. Override with `AISP_TINYSTORIES_PACKED_PATH`, `AISP_TINYSTORIES_LOCAL_PATH`, `AISP_TINYSTORIES_CONFIG_PATH`, or `AISP_TINYSTORIES_LAYERS`.",
        "Scale up by increasing `AISP_TINYSTORIES_LAYERS` or swapping to a larger config and pairing it with a packed dataset that matches the new sequence length.",
        "Set `AISP_FSDP_DISABLE_FP8=1` to keep the minimal BF16 path; unset it when you want to exercise the FP8 conversion on larger workloads.",
        "On single-GPU `b200`, `fsdp2` remains runnable for regression tracking and profiler capture, but the benchmark is judged as a local comparison contract rather than a canonical speed claim.",
    ],
)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    if args.list:
        for slug in ENTRIES:
            print(slug)
        return 0

    if not args.write and not args.check:
        parser.print_help()
        return 0

    try:
        targets = _resolve_target_slugs(targets=args.target, include_all=args.all)
    except ValueError as exc:
        parser.error(str(exc))

    if args.write and not targets:
        parser.error("Refusing to write without an explicit scope. Pass --target ... or --all.")

    if args.check and not targets:
        targets = list(ENTRIES.keys())

    if args.write:
        for output_path in write_readmes(targets=targets, repo_root=repo_root):
            print(f"Wrote {output_path.relative_to(repo_root)}")
        return 0

    mismatches = check_readmes(targets=targets, repo_root=repo_root)
    if mismatches:
        print("README targets out of sync:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print(f"All {len(targets)} README target(s) are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
