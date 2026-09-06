"""CLI commands for profile analysis and comparison."""

from __future__ import annotations

import json
import html
import math
import shlex
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List
import typer

from core.profile_insights import generate_flamegraph_comparison
from core.perf_core_base import PerformanceCoreBase


def _get_core() -> PerformanceCoreBase:
    """Get an instance of PerformanceCoreBase for profile operations."""
    return PerformanceCoreBase()


def _prepare_profile_run(output_dir_opt: Optional[str], tool_key: str, label: str) -> tuple[Path, Path]:
    from core.benchmark.artifact_manager import (
        ArtifactManager,
        build_run_id,
        default_artifacts_root,
        slugify,
    )

    base_dir = Path(output_dir_opt) if output_dir_opt else default_artifacts_root(Path.cwd())
    run_label = label or "run"
    run_id = build_run_id(f"profile-{tool_key}", run_label, base_dir=base_dir)
    artifacts = ArtifactManager(base_dir=base_dir, run_id=run_id, run_kind=f"profile-{tool_key}", run_label=run_label)
    profile_root = artifacts.profiles_dir / "tools" / slugify(tool_key) / slugify(run_label)
    profile_root.mkdir(parents=True, exist_ok=True)
    return artifacts.run_dir, profile_root


def compare_profiles(args) -> None:
    """Compare baseline vs optimized profiles and generate flame graph visualization."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    
    # Extract arguments from namespace
    chapter = getattr(args, 'chapter', None)
    output = getattr(args, 'output', 'comparison_flamegraph.html')
    json_out = getattr(args, 'json_out', None)
    pair_key = getattr(args, 'pair', None)
    include_ncu_details = bool(getattr(args, 'include_ncu_details', False))
    
    if not chapter:
        # List available profile pairs
        core = _get_core()
        pairs = core.list_deep_profile_pairs()
        
        if not pairs.get("pairs"):
            console.print("[yellow]No profile pairs found.[/yellow]")
            console.print("Run profiling with nsys/ncu on baseline and optimized code first:")
            console.print("  [cyan]aisp profile nsys baseline_code.py[/cyan]")
            console.print("  [cyan]aisp profile nsys optimized_code.py[/cyan]")
            return
        
        table = Table(title="Available Profile Pairs for Comparison")
        table.add_column("Chapter", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("NSYS", style="yellow")
        table.add_column("NCU", style="yellow")
        
        for pair in pairs["pairs"]:
            table.add_row(
                pair.get("chapter", "unknown"),
                pair.get("type", "unknown"),
                "✓" if pair.get("has_nsys") else "✗",
                "✓" if pair.get("has_ncu") else "✗",
            )
        
        console.print(table)
        console.print(f"\nFound [green]{pairs['count']}[/green] profile pairs.")
        console.print("\nTo compare a specific chapter:")
        console.print("  [cyan]aisp profile compare <chapter-name>[/cyan]")
        return
    
    # Generate comparison for specified chapter
    core = _get_core()
    profile_dir = core._find_profile_directory(chapter)
    
    if not profile_dir:
        console.print(f"[red]Profile directory not found for: {chapter}[/red]")
        return
    
    console.print(f"[cyan]Analyzing profiles in: {profile_dir}[/cyan]")
    
    result = generate_flamegraph_comparison(profile_dir, pair_key=pair_key)
    
    if not result:
        console.print("[red]No baseline/optimized nsys profiles found.[/red]")
        return
    
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        if result.get("candidates"):
            console.print(f"[yellow]Available pairs:[/yellow] {result['candidates']}")
        return
    
    # NEW: Get metric-level analysis (improvements, regressions, bottleneck shifts)
    metric_comparison = core.compare_profiles(
        chapter,
        pair_key=pair_key,
        include_ncu_details=include_ncu_details,
    )
    if metric_comparison and metric_comparison.get("error"):
        console.print(f"[red]Error: {metric_comparison['error']}[/red]")
        if metric_comparison.get("candidates"):
            console.print(f"[yellow]Available pairs:[/yellow] {metric_comparison['candidates']}")
        return

    # Display summary
    baseline = result.get("baseline", {})
    optimized = result.get("optimized", {})
    metrics = result.get("metrics", {})
    speedup = result.get("speedup", 1.0)
    
    console.print(Panel.fit(
        f"[bold green]⚡ {speedup}x Speedup[/bold green]\n\n"
        f"Baseline: {baseline.get('total_time_ms', 0):.2f} ms\n"
        f"Optimized: {optimized.get('total_time_ms', 0):.2f} ms\n\n"
        f"Sync Calls: {metrics.get('baseline_sync_calls', 0)} → {metrics.get('optimized_sync_calls', 0)} "
        f"([green]-{metrics.get('sync_reduction_pct', 0):.1f}%[/green])\n"
        f"Device Syncs: {metrics.get('baseline_device_sync', 0)} → {metrics.get('optimized_device_sync', 0)} "
        f"([green]-{metrics.get('device_sync_reduction_pct', 0):.1f}%[/green])\n"
        f"Wait Events: {metrics.get('optimized_wait_events', 0)} (lightweight coordination)",
        title="🔥 Flame Graph Comparison",
        border_style="green",
    ))
    
    if result.get("insight"):
        console.print(f"\n[yellow]Insight:[/yellow] {result['insight']}")
    
    if metric_comparison and not metric_comparison.get("error"):
        metric_analysis = metric_comparison.get("metric_analysis")
        if metric_analysis:
            console.print("\n[bold cyan]📊 Metric Analysis[/bold cyan]")
            
            if metric_analysis.get("bottleneck_shift"):
                console.print(f"[yellow]Bottleneck Shift:[/yellow] {metric_analysis['bottleneck_shift']}")
            
            if metric_analysis.get("key_improvements"):
                console.print("\n[green]Key Improvements:[/green]")
                for imp in metric_analysis["key_improvements"][:5]:
                    console.print(f"  ✅ {imp}")
            
            if metric_analysis.get("regressions"):
                console.print("\n[red]Regressions (investigate):[/red]")
                for reg in metric_analysis["regressions"][:3]:
                    console.print(f"  ⚠️  {reg['metric']}: {reg['baseline']:.1f} → {reg['optimized']:.1f} ({reg['change_pct']:+.1f}%)")
            
            if metric_analysis.get("remaining_issues"):
                console.print("\n[yellow]Remaining Optimization Opportunities:[/yellow]")
                for issue in metric_analysis["remaining_issues"][:3]:
                    console.print(f"  → {issue}")
            
            # Add to result for JSON output
            result["metric_analysis"] = metric_analysis
    
    # Generate HTML flame graph
    html_content = _generate_comparison_html(result, chapter)
    output_path = Path(output)
    output_path.write_text(html_content)
    console.print(f"\n[green]✓ Flame graph saved to: {output_path}[/green]")
    
    # Optionally save JSON
    if json_out:
        json_path = Path(json_out)
        json_path.write_text(json.dumps(result, indent=2))
        console.print(f"[green]✓ JSON data saved to: {json_path}[/green]")


def _generate_comparison_html(data: Dict[str, Any], chapter: str) -> str:
    """Generate an interactive HTML flame graph comparison."""
    baseline = data.get("baseline", {})
    optimized = data.get("optimized", {})
    metrics = data.get("metrics", {})
    speedup = data.get("speedup", 1.0)
    insight = data.get("insight", "")
    
    def bar_html(bars: list, is_kernel: bool = False) -> str:
        if not bars:
            return '<div class="bar other" style="width:100%">No data</div>'
        
        type_colors = {
            'sync': '#ef4444',
            'malloc': '#8b5cf6',
            'launch': '#a78bfa',
            'memcpy': '#fbbf24',
            'wait': '#34d399',
            'other': '#6b7280',
        }
        kernel_color = '#22d3d3'
        
        html_parts = []
        for bar in bars:
            color = kernel_color if is_kernel else type_colors.get(bar.get('type', 'other'), '#6b7280')
            width = max(bar.get('time_pct', 0), 3)
            name = bar.get('name', 'unknown')[:25]
            pct = bar.get('time_pct', 0)
            html_parts.append(
                f'<div class="bar" style="width:{width}%;background:{color}" '
                f'title="{bar.get("name", "")}: {pct:.1f}%">{name if pct > 8 else f"{pct:.0f}%"}</div>'
            )
        return ''.join(html_parts)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flame Graph Comparison - {chapter}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2rem; 
            margin-bottom: 1.5rem;
            background: linear-gradient(90deg, #22d3ee, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .speedup-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            border-radius: 9999px;
            font-size: 1.25rem;
            font-weight: bold;
            color: #0f172a;
            margin-bottom: 2rem;
        }}
        .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
        .panel {{
            background: rgba(255,255,255,0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .panel.baseline {{ border-left: 4px solid #ef4444; }}
        .panel.optimized {{ border-left: 4px solid #10b981; }}
        .panel h2 {{ 
            font-size: 1.1rem; 
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .panel h2 .time {{
            font-family: monospace;
            font-size: 0.9rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
        }}
        .panel.baseline h2 .time {{ background: rgba(239,68,68,0.2); color: #fca5a5; }}
        .panel.optimized h2 .time {{ background: rgba(16,185,129,0.2); color: #6ee7b7; }}
        .section-title {{ 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            letter-spacing: 0.1em;
            color: rgba(255,255,255,0.5);
            margin: 1rem 0 0.5rem;
        }}
        .flame-row {{
            display: flex;
            gap: 2px;
            height: 2rem;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 0.5rem;
            font-size: 0.7rem;
            font-weight: 500;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: filter 0.2s;
            cursor: pointer;
        }}
        .bar:hover {{ filter: brightness(1.2); }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric {{
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .metric-label {{ font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-bottom: 0.5rem; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; }}
        .metric-change {{ font-size: 0.8rem; color: #10b981; }}
        .insight {{
            background: linear-gradient(90deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1));
            border: 1px solid rgba(16,185,129,0.3);
            border-radius: 0.75rem;
            padding: 1rem 1.5rem;
        }}
        .insight-title {{ color: #6ee7b7; font-weight: 600; margin-bottom: 0.5rem; }}
        .legend {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .legend-color {{
            width: 1rem;
            height: 0.75rem;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Flame Graph Comparison: {chapter}</h1>
        
        <div class="speedup-badge">⚡ {speedup}x Speedup</div>
        
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#ef4444"></div>Sync</div>
            <div class="legend-item"><div class="legend-color" style="background:#8b5cf6"></div>Malloc/Free</div>
            <div class="legend-item"><div class="legend-color" style="background:#a78bfa"></div>Launch</div>
            <div class="legend-item"><div class="legend-color" style="background:#fbbf24"></div>Memcpy</div>
            <div class="legend-item"><div class="legend-color" style="background:#34d399"></div>Wait</div>
            <div class="legend-item"><div class="legend-color" style="background:#22d3d3"></div>Kernel</div>
        </div>
        
        <div class="panels">
            <div class="panel baseline">
                <h2>
                    Baseline (Sequential)
                    <span class="time">{baseline.get('total_time_ms', 0):.2f} ms</span>
                </h2>
                <div class="section-title">CUDA API Distribution</div>
                <div class="flame-row">{bar_html(baseline.get('api_bars', []))}</div>
                <div class="section-title">Kernel Distribution</div>
                <div class="flame-row">{bar_html(baseline.get('kernel_bars', []), True)}</div>
            </div>
            <div class="panel optimized">
                <h2>
                    Optimized (Pipelined)
                    <span class="time">{optimized.get('total_time_ms', 0):.2f} ms</span>
                </h2>
                <div class="section-title">CUDA API Distribution</div>
                <div class="flame-row">{bar_html(optimized.get('api_bars', []))}</div>
                <div class="section-title">Kernel Distribution</div>
                <div class="flame-row">{bar_html(optimized.get('kernel_bars', []), True)}</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Sync Calls</div>
                <div class="metric-value">{metrics.get('baseline_sync_calls', 0)} → {metrics.get('optimized_sync_calls', 0)}</div>
                <div class="metric-change">-{metrics.get('sync_reduction_pct', 0):.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Device Syncs</div>
                <div class="metric-value">{metrics.get('baseline_device_sync', 0)} → {metrics.get('optimized_device_sync', 0)}</div>
                <div class="metric-change">-{metrics.get('device_sync_reduction_pct', 0):.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Wait Events</div>
                <div class="metric-value">{metrics.get('optimized_wait_events', 0)}</div>
                <div class="metric-change">Lightweight sync</div>
            </div>
            <div class="metric">
                <div class="metric-label">Time Saved</div>
                <div class="metric-value">{(baseline.get('total_time_ms', 0) - optimized.get('total_time_ms', 0)):.1f} ms</div>
                <div class="metric-change">{((1 - optimized.get('total_time_ms', 1) / max(baseline.get('total_time_ms', 1), 0.001)) * 100):.1f}% faster</div>
            </div>
        </div>
        
        {"" if not insight else f'''<div class="insight">
            <div class="insight-title">💡 Optimization Insight</div>
            <p>{insight}</p>
        </div>'''}
    </div>
</body>
</html>'''


def _flame_html(data: Dict[str, Any]) -> str:
    """Offline SVG duration view; profile names are data, never HTML/script."""
    rectangles = []
    max_depth = 0

    def draw(node, x, width, depth):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        name = html.escape(str(node['name']))
        value = float(node['value'])
        rectangles.append(
            f'<g><title>{name}: {value:.3f} us</title>'
            f'<rect x="{x:.4f}" y="{depth * 28}" width="{width:.4f}" height="26" '
            f'fill="hsl({25 + depth * 37},75%,65%)" stroke="white"/>'
            f'<text x="{x + 3:.4f}" y="{depth * 28 + 18}">{name[:80]}</text></g>'
        )
        child_x = x
        for child in node.get('children', []):
            child_width = width * float(child['value']) / value if value else 0
            if child_width > 0:
                draw(child, child_x, child_width, depth + 1)
                child_x += child_width

    draw(data, 0, 1200, 0)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Trace duration flame graph</title>'
        '<style>body{font-family:system-ui;margin:24px}svg{width:100%;height:auto}text{font-size:11px;pointer-events:none}</style>'
        '</head><body><h1>Trace duration flame graph</h1>'
        '<p>Widths show summed complete-event durations, grouped by trace category. '
        'Overlapping CPU/GPU events may double-count elapsed time; this is not a wall-time timeline or reconstructed call stack. '
        'Hover a rectangle for its full name and duration.</p>'
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Trace duration flame graph" '
        f'viewBox="0 0 1200 {(max_depth + 1) * 28}">' + ''.join(rectangles) + '</svg></body></html>'
    )


def flame(args) -> int:
    """Export a real Chrome trace as offline HTML/SVG or flame graph JSON."""
    from rich.console import Console
    from core.profiling.flame_graph import FlameGraphGenerator
    console = Console()
    try:
        filename = getattr(args, 'file', None)
        if not filename:
            raise ValueError("A Chrome trace JSON file is required; export one with torch.profiler")
        source = Path(filename)
        if not source.is_file():
            raise ValueError(f"Profile file not found: {source}")
        if source.suffix.lower() != '.json':
            raise ValueError("Only Chrome trace JSON is supported; export binary Nsight reports to JSON first")
        output = Path(getattr(args, 'output', None) or 'flame.html')
        if output.suffix.lower() not in ('.html', '.json'):
            raise ValueError("Flame graph output must end in .html or .json")
        if output.resolve() == source.resolve():
            raise ValueError("Output must differ from the input trace")
        # Preserve invalid durations for explicit validation instead of hiding
        # them behind a minimum-duration filter.
        generator = FlameGraphGenerator(min_duration_us=float('-inf'))
        data = generator.from_chrome_trace(source, root_name='Trace events (summed duration)')
        if data.get('error'):
            raise ValueError(data['error'])

        def validate(node):
            value = float(node['value'])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Trace contains non-finite or negative event durations")
            for child in node.get('children', []):
                validate(child)

        validate(data)
        if data['value'] <= 0 or not data.get('children'):
            raise ValueError("Trace has no positive complete-event durations to visualize")
        for warning in data.get('warnings', []):
            console.print(f"Warning: {warning}", markup=False)
        content = json.dumps(data, indent=2, allow_nan=False) if output.suffix.lower() == '.json' else _flame_html(data)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding='utf-8')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        console.print(f"Flame graph failed: {exc}", style='red', markup=False)
        return 1
    console.print(f"Flame graph saved to: {output}", style='green', markup=False)
    return 0


def memory(args) -> int:
    """Memory timeline analysis."""
    from rich.console import Console
    console = Console()
    console.print("Memory timeline conversion is not implemented by this command; no output was written. Use the dashboard for existing memory timeline artifacts.", style='red')
    return 1


def kernels(args) -> int:
    """Kernel breakdown analysis."""
    from rich.console import Console
    console = Console()
    console.print("Standalone kernel breakdown is not implemented by this command. Use 'aisp profile compare' with collected baseline/optimized profiles.", style='red')
    return 1


def hta(args) -> None:
    """Holistic Trace Analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Use 'aisp profile hta-capture' to collect a new HTA trace or view highlights in the dashboard.[/yellow]")


def torch_profiler(args) -> int:
    """Run torch.profiler capture with NVTX + lineinfo defaults."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.torch_profiler import TorchProfilerAutomation

    console = Console()
    script_path = Path(getattr(args, "script", ""))
    if not script_path.exists():
        console.print(f"[red]Script not found:[/red] {script_path}")
        return 1

    output_dir_opt = getattr(args, "output_dir", None)
    mode = getattr(args, "mode", "full")
    output_name = getattr(args, "output_name", None) or script_path.stem
    _, output_root = _prepare_profile_run(output_dir_opt, "torch", output_name)
    nvtx_label = getattr(args, "nvtx_label", "aisp_torch_profile")
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    use_nvtx = bool(getattr(args, "use_nvtx", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)
    script_args: List[str] = getattr(args, "script_args", None) or []

    profiler = TorchProfilerAutomation(output_root)
    console.print(f"[cyan]Running torch.profiler ({mode}) on {script_path}[/cyan]")
    result = profiler.profile(
        script=script_path,
        output_name=output_name,
        mode=mode,
        script_args=script_args,
        force_lineinfo=force_lineinfo,
        timeout_seconds=timeout_seconds,
        nvtx_label=nvtx_label,
        use_nvtx=use_nvtx,
    )

    if not result.get("success"):
        console.print(f"[red]Profiler failed:[/red] {result.get('error', 'unknown error')}")
        return 1

    panel_lines = [
        f"[green]✓[/green] Trace: {result.get('trace_path') or 'n/a'}",
        f"Summary: {result.get('summary') and 'torch_profile_summary.json' or 'n/a'}",
        f"Metadata: {result.get('metadata') and 'metadata.json' or 'n/a'}",
        f"NVTX label: {result.get('nvtx_label')}",
        f"Force lineinfo: {result.get('force_lineinfo')}",
    ]
    for warning in result.get("warnings", []):
        panel_lines.append(f"[yellow]Warning:[/yellow] {warning}")
    console.print(Panel.fit("\n".join(panel_lines), title="torch.profiler capture", border_style="green"))
    return 0


def hta_capture(args) -> int:
    """Run an HTA-friendly nsys capture and analyze it."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.hta_capture import HTACaptureAutomation

    console = Console()
    command_str = getattr(args, "command", "")
    command_list = getattr(args, "command_list", None) or (shlex.split(command_str) if command_str else [])
    if not command_list:
        console.print("[red]Command is required (e.g., python train.py --arg foo)[/red]")
        return 1

    output_dir_opt = getattr(args, "output_dir", None)
    output_name = getattr(args, "output_name", "hta_capture")
    _, output_root = _prepare_profile_run(output_dir_opt, "hta", output_name)
    preset = getattr(args, "preset", "full")
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)

    capture = HTACaptureAutomation(output_root)
    console.print(f"[cyan]Running HTA capture via nsys ({preset})[/cyan]")
    result = capture.capture(
        command=command_list,
        output_name=output_name,
        preset=preset,
        force_lineinfo=force_lineinfo,
        timeout_seconds=timeout_seconds,
    )

    if not result.get("success"):
        console.print(f"[red]HTA capture failed:[/red] {result.get('error', 'unknown error')}")
        return 1

    lines = [
        f"[green]✓[/green] NSYS report: {result.get('nsys_output')}",
        f"Trace JSON: {result.get('trace_path')}",
        f"HTA report: {result.get('hta_report')}",
        f"Force lineinfo: {result.get('force_lineinfo')}",
    ]
    console.print(Panel.fit("\n".join(lines), title="HTA capture", border_style="green"))
    return 0


def nsys(args) -> int:
    """Run Nsight Systems profiling on an arbitrary command."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.nsight_automation import NsightAutomation

    console = Console()
    command_str = getattr(args, "command", "")
    command_list = getattr(args, "command_list", None) or (shlex.split(command_str) if command_str else [])
    if not command_list:
        console.print("[red]Command is required (example: \"python train.py --batch 8\").[/red]")
        return 1

    output_dir_opt = getattr(args, "output_dir", None)
    output_name = getattr(args, "output_name", None) or "profile_nsys"
    _, output_root = _prepare_profile_run(output_dir_opt, "nsys", output_name)

    trace_cuda = bool(getattr(args, "trace_cuda", True))
    trace_nvtx = bool(getattr(args, "trace_nvtx", True))
    trace_osrt = bool(getattr(args, "trace_osrt", True))
    full_timeline = bool(getattr(args, "full_timeline", False))
    trace_forks = bool(getattr(args, "trace_forks", False))
    preset = str(getattr(args, "preset", "light") or "light").strip().lower()
    wait_mode = str(getattr(args, "wait_mode", "primary") or "primary").strip().lower()
    finalize_grace_seconds = float(getattr(args, "finalize_grace_seconds", 20.0) or 20.0)
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    sanitize_python_startup = bool(getattr(args, "sanitize_python_startup", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)

    automation = NsightAutomation(output_root)
    if not automation.nsys_available:
        console.print("[red]Nsight Systems (nsys) is not installed or not on PATH.[/red]")
        return 1

    console.print(f"[cyan]Running Nsight Systems ({preset})[/cyan]")
    try:
        output = automation.profile_nsys(
            command=command_list,
            output_name=output_name,
            trace_cuda=trace_cuda,
            trace_nvtx=trace_nvtx,
            trace_osrt=trace_osrt,
            full_timeline=full_timeline,
            trace_forks=trace_forks,
            preset=preset,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            wait_mode=wait_mode,
            finalize_grace_seconds=finalize_grace_seconds,
            sanitize_python_startup=sanitize_python_startup,
        )
    except ValueError as exc:
        console.print(f"[red]NSYS configuration error:[/red] {exc}")
        return 1

    if not output:
        console.print(f"[red]NSYS profiling failed:[/red] {automation.last_error or 'unknown error'}")
        return 1

    last_run = automation.last_run or {}
    lines = [
        f"[green]✓[/green] NSYS report: {output}",
        f"Preset: {preset}",
        f"Wait mode: {wait_mode}",
        f"Trace forks: {bool(trace_forks)}",
        f"Timeout hit: {bool(last_run.get('timeout_hit', False))}",
        f"Force lineinfo: {force_lineinfo}",
        f"Sanitize python startup: {sanitize_python_startup}",
    ]
    console.print(Panel.fit("\n".join(lines), title="Nsight Systems capture", border_style="green"))
    return 0


def ncu(args) -> None:
    """Run Nsight Compute profiling on a command or Python script."""
    from rich.console import Console
    from rich.panel import Panel
    from core.profiling.nsight_automation import NsightAutomation
    from core.profiling.profiler_config import (
        MINIMAL_METRICS,
        validate_ncu_app_range_capture,
        validate_ncu_replay_mode,
    )

    console = Console()

    command_str = getattr(args, "command", "")
    command_list = getattr(args, "command_list", None) or (shlex.split(command_str) if command_str else [])
    script_path = Path(getattr(args, "script", "")) if getattr(args, "script", None) else None
    script_args: List[str] = getattr(args, "script_args", None) or []
    kernel_filter = getattr(args, "kernel_filter", None) or getattr(args, "kernel", None)
    kernel_name_base = getattr(args, "kernel_name_base", None)
    profile_from_start = getattr(args, "profile_from_start", None)
    nvtx_include_arg = getattr(args, "nvtx_include", None) or []
    if isinstance(nvtx_include_arg, str):
        nvtx_includes: List[str] = [v.strip() for v in nvtx_include_arg.split(",") if v.strip()]
    else:
        nvtx_includes = [str(v).strip() for v in nvtx_include_arg if str(v).strip()]

    if not command_list:
        if not script_path:
            console.print("[red]Provide --command or a script path.[/red]")
            return 1
        if not script_path.exists():
            console.print(f"[red]Script not found:[/red] {script_path}")
            return 1
        command_list = [sys.executable, str(script_path), *script_args]

    output_dir_opt = getattr(args, "output_dir", None)
    output_name = getattr(args, "output_name", None) or (script_path.stem if script_path else "profile_ncu")
    _, output_root = _prepare_profile_run(output_dir_opt, "ncu", output_name)

    workload_type = getattr(args, "workload_type", "memory_bound")
    metric_set = getattr(args, "metric_set", "full")
    replay_mode = getattr(args, "replay_mode", "application")
    launch_skip = getattr(args, "launch_skip", None)
    launch_count = getattr(args, "launch_count", None)
    sampling_interval = getattr(args, "pm_sampling_interval", None)
    force_lineinfo = bool(getattr(args, "force_lineinfo", True))
    timeout_seconds = getattr(args, "timeout_seconds", None)

    automation = NsightAutomation(output_root)
    console.print(f"[cyan]Running Nsight Compute ({metric_set}, {workload_type})[/cyan]")
    try:
        replay_mode = validate_ncu_replay_mode(replay_mode)
        if replay_mode == "app-range":
            if str(metric_set).strip().lower() not in {"minimal", "basic"}:
                raise ValueError("app-range requires metric_set='minimal' or 'basic'.")
            validate_ncu_app_range_capture(
                nvtx_includes=nvtx_includes,
                metrics=MINIMAL_METRICS,
                kernel_filter=kernel_filter,
                launch_skip=launch_skip,
                launch_count=launch_count,
                sampling_interval=sampling_interval,
                profile_from_start=profile_from_start,
            )
        output = automation.profile_ncu(
            command=command_list,
            output_name=output_name,
            workload_type=workload_type,
            kernel_filter=kernel_filter,
            kernel_name_base=kernel_name_base,
            nvtx_includes=nvtx_includes,
            profile_from_start=profile_from_start,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
            sampling_interval=sampling_interval,
            metric_set=metric_set,
            launch_skip=launch_skip,
            launch_count=launch_count,
            replay_mode=replay_mode,
        )
    except ValueError as exc:
        console.print(f"[red]NCU configuration error:[/red] {exc}")
        return 1

    if not output:
        console.print(f"[red]NCU profiling failed:[/red] {automation.last_error or 'unknown error'}")
        return 1

    last_run = automation.last_run or {}
    launch_skip_used = last_run.get("launch_skip", launch_skip)
    launch_count_used = last_run.get("launch_count", launch_count)
    replay_mode_used = last_run.get("replay_mode", replay_mode)
    metric_set_resolved = last_run.get("metric_set_resolved", metric_set)
    lines = [
        f"[green]✓[/green] NCU report: {output}",
        f"Workload type: {workload_type}",
        f"Metric set: {metric_set} (resolved: {metric_set_resolved})",
        f"Replay mode: {replay_mode_used}",
        f"Kernel filter: {kernel_filter or 'none'}",
        f"Kernel name base: {kernel_name_base or 'default'}",
        f"NVTX includes: {', '.join(nvtx_includes) if nvtx_includes else 'none'}",
        f"Profile from start: {profile_from_start or 'default'}",
        f"Launch skip: {launch_skip_used if launch_skip_used is not None else 'none'}",
        f"Launch count: {launch_count_used if launch_count_used is not None else 'none'}",
        f"Force lineinfo: {force_lineinfo}",
    ]
    if last_run.get("ncu_capture"):
        lines.append("Report scope: complete selected NVTX range")
        lines.append(f"Capture evidence: {Path(output).with_suffix('.capture.json')}")
    console.print(Panel.fit("\n".join(lines), title="Nsight Compute capture", border_style="green"))
    return 0


def warp_divergence(args) -> None:
    """Warp divergence analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Warp divergence analysis[/yellow]")


def bank_conflicts(args) -> None:
    """Bank conflict analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Bank conflict analysis[/yellow]")


def occupancy(args) -> None:
    """Occupancy analysis."""
    from rich.console import Console
    console = Console()
    console.print("[yellow]Occupancy analysis[/yellow]")
