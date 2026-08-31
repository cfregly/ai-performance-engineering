#!/usr/bin/env python3
"""
🚀 aisp - AI Systems Performance CLI

Single entry point for the AI Performance Engineering system.
Defaults: no args → launch the interactive benchmark analysis menu.

ARCHITECTURE:
    This CLI uses the unified PerformanceEngine (10 domains) as its backend.
    Commands are organized to match the engine's domain model:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  CLI Command Groups       →  Engine Domains                          │
    │──────────────────────────────────────────────────────────────────────│
    │  aisp gpu [info|topology|power|bandwidth|features]  → gpu            │
    │  aisp system [status|software|deps|capabilities|context|parameters|container|cpu-memory|env|network] → system │
    │  aisp profile [nsys|ncu|compare|flame|hta]          → profile        │
    │  aisp optimize [recommend|roi|techniques]           → optimize       │
    │  aisp distributed [plan|nccl|topology|slurm]        → distributed    │
    │  aisp inference [vllm|quantize|deploy|serve|estimate] → inference     │
    │  aisp ai [ask|explain|troubleshoot]                 → ai             │
    │  aisp benchmark [memory|pcie|tc|speed|diagnostics]  → benchmark      │
    │  aisp bench [run|targets|compare|report|export]     → benchmark      │
    └──────────────────────────────────────────────────────────────────────┘

DOMAIN MODEL (10 domains in core/engine.py):
    1. gpu        - Hardware info, topology, power, bandwidth
    2. system     - Software stack, dependencies, capabilities
    3. profile    - nsys/ncu profiling, flame graphs, HTA
    4. analyze    - Bottlenecks, pareto, scaling, memory patterns
    5. optimize   - Recommendations, ROI, techniques
    6. distributed- Parallelism planning, NCCL, FSDP
    7. inference  - vLLM config, quantization, deployment
    8. benchmark  - Run & track benchmarks, history
    9. ai         - Ask questions, explain concepts, LLM analysis
   10. export     - CSV, PDF, HTML reports

QUICK START:
    aisp                    # Launch interactive benchmark analysis menu
    aisp system status      # System status overview
    aisp gpu info           # GPU information
    aisp ai ask "Why slow?" # Ask AI about performance
    aisp bench run -t ch07  # Run benchmarks
    
EXAMPLES:
    # Get optimization recommendations for 70B model on 4 GPUs
    aisp optimize recommend --model-size 70 --gpus 4
    
    # Profile with Nsight Systems
    aisp profile nsys python train.py
    
    # Plan distributed training
    aisp distributed plan --model-size 70 --gpus 16 --nodes 2
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
from pathlib import Path
from enum import Enum
from types import SimpleNamespace
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

if __package__ in {None, ""}:
    env = dict(os.environ)
    pythonpath = [str(REPO_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.extend(part for part in existing.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "cli.aisp", *sys.argv[1:]], env=env)
    )

from core.plugins.loader import load_plugin_apps
from core.utils.dotenv import load_repo_dotenv

try:
    import typer
except ImportError:  # pragma: no cover - Typer required for the CLI
    typer = None  # type: ignore

# =============================================================================
# ENV LOADING (single implementation)
# =============================================================================

def load_env() -> None:
    """Load .env and .env.local files."""
    load_repo_dotenv(REPO_ROOT)

load_env()


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    return str(value)


def emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=_json_default))


# =============================================================================
# Choice enums (Typer-compatible)
# =============================================================================


class ProfilePreset(str, Enum):
    none = "none"
    minimal = "minimal"
    deep_dive = "deep_dive"
    roofline = "roofline"


class BottleneckMode(str, Enum):
    profile = "profile"
    llm = "llm"
    both = "both"


class GoalChoice(str, Enum):
    throughput = "throughput"
    latency = "latency"
    memory = "memory"


class SpeedTestChoice(str, Enum):
    all = "all"
    gemm = "gemm"
    attention = "attention"
    memory = "memory"


# =============================================================================
# Helpers
# =============================================================================

def _typer_required() -> None:
    print("Typer is required for aisp. Install with: pip install typer")
    sys.exit(1)


def _build_args(ctx: typer.Context, **kwargs) -> SimpleNamespace:
    """Inject common flags into a SimpleNamespace for legacy command functions."""
    verbose = False
    json_output = False
    if ctx and ctx.obj:
        verbose = bool(ctx.obj.get("verbose", False))
        json_output = bool(ctx.obj.get("json_output", False))
    normalized = {
        key: (value.value if isinstance(value, Enum) else value) for key, value in kwargs.items()
    }
    return SimpleNamespace(verbose=verbose, json=json_output, **normalized)


def _run(func, ctx: typer.Context, **kwargs) -> None:
    """Execute legacy command functions and honor their integer exit codes."""
    args = _build_args(ctx, **kwargs)
    try:
        result = func(args)
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    code = 0 if result is None else int(result)
    raise typer.Exit(code=code)


def _run_module(module: str, args: List[str]) -> None:
    """Execute a module as a subprocess and propagate its exit code."""
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd)
    raise typer.Exit(code=result.returncode)


def _rewrite_help_aliases(argv: List[str]) -> List[str]:
    """Support `aisp help` and `aisp <category> help` as help aliases."""
    if len(argv) > 1 and argv[1] == "help":
        return [argv[0], "help", *argv[2:]]
    if len(argv) > 2 and argv[2] == "help":
        return [argv[0], "help", argv[1], *argv[3:]]
    return argv


# =============================================================================
# Typer app + subcommands
# =============================================================================

if typer:
    app = typer.Typer(
        name="aisp",
        help="AI Systems Performance CLI - Unified 10-Domain API",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    # ==========================================================================
    # 10-DOMAIN COMMAND GROUPS (matching engine.py domains)
    # ==========================================================================
    
    # Domain 1: GPU - Hardware info, topology, power, bandwidth
    gpu_app = typer.Typer(help="GPU hardware: info, topology, power, bandwidth")
    
    # Domain 2: System - Software stack, dependencies, capabilities
    system_app = typer.Typer(help="System: software, deps, capabilities, env, parameters")
    
    # Domain 3: Profile - nsys/ncu profiling, flame graphs, HTA
    profile_app = typer.Typer(help="Profiling: nsys, ncu, torch, flame graphs, HTA")
    
    # Domain 4: Analyze - Bottlenecks, pareto, scaling, what-if
    analyze_app = typer.Typer(help="Analysis: bottlenecks, pareto, scaling, what-if")
    
    # Domain 5: Optimize - Recommendations, ROI, techniques
    optimize_app = typer.Typer(help="Optimization: recommendations, ROI, techniques")
    
    # Domain 6: Distributed - Parallelism, NCCL, FSDP
    distributed_app = typer.Typer(help="Distributed: parallelism planning, NCCL, FSDP")
    
    # Domain 7: Inference - vLLM, quantization, deployment
    inference_app = typer.Typer(help="Inference: vLLM config, quantization, deployment")
    
    # Domain 8: Benchmark - Microbench diagnostics
    benchmark_app = typer.Typer(help="Benchmarks: microbench diagnostics, speed tests")
    
    # Domain 9: AI - Ask questions, explain concepts
    ai_app = typer.Typer(help="AI: ask questions, explain concepts, suggest tools")
    
    # Domain 10: Export - CSV, PDF, HTML reports
    export_app = typer.Typer(help="Export: CSV, PDF, HTML reports")

    # Extra group: Cluster - preset evals, raw suite, canonical packaging, and validation helpers
    cluster_app = typer.Typer(help="Cluster: common eval presets, raw suite, canonical packaging, field report validation")
    
# =============================================================================
# Root callback
# =============================================================================

if typer:

    def _run_benchmark_menu(data_file: Optional[Path] = None) -> None:
        """Use the same working analysis menu as ``aisp bench tui --simple``."""
        try:
            from core.benchmark.bench_commands import _run_basic_menu

            _run_basic_menu(data_file)
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nExiting benchmark analysis menu.")
        except Exception as exc:
            typer.echo(f"Benchmark menu error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    @app.callback(invoke_without_command=True)
    def _root(
        ctx: typer.Context,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON when supported"),
        dynamo_logs: bool = typer.Option(
            False,
            "--dynamo-logs/--no-dynamo-logs",
            help="Enable TorchDynamo/TORCH_LOGS output (disabled by default for cleaner CLI)",
        ),
    ) -> None:
        """Default: launch the benchmark analysis menu when no subcommand is provided."""
        ctx.obj = {"verbose": verbose, "json_output": json_output}
        if not dynamo_logs:
            for key in ["TORCH_LOGS", "TORCH_COMPILE_DEBUG"]:
                os.environ.pop(key, None)
        if ctx.invoked_subcommand is None:
            _run_benchmark_menu()
            raise typer.Exit()


# =============================================================================
# Domain 1: GPU - Hardware info, topology, power, bandwidth
# =============================================================================

if typer:

    @gpu_app.command("info", help="GPU information: name, memory, temperature, utilization")
    def gpu_info_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().gpu.info()
        if ctx.obj and ctx.obj.get("json_output"):
            emit_json(result)
        else:
            typer.echo(f"\n🖥️  GPU Info")
            for i, gpu in enumerate(result.get("gpus", [result])):
                name = gpu.get("name", "Unknown")
                mem = gpu.get("memory_total_gb", 0)
                temp = gpu.get("temperature_c", 0)
                util = gpu.get("utilization_pct", 0)
                typer.echo(f"  GPU {i}: {name}")
                typer.echo(f"    Memory: {mem:.1f} GB, Temp: {temp}°C, Util: {util}%")
        raise typer.Exit(0)

    @gpu_app.command("topology", help="GPU topology: NVLink, PCIe, P2P matrix")
    def gpu_topology_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().gpu.topology()
        emit_json(result)
        raise typer.Exit(0)

    @gpu_app.command("power", help="Power draw, limits, thermal status")
    def gpu_power_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.power()
        emit_json(result)
        raise typer.Exit(0)

    @gpu_app.command("bandwidth", help="Run GPU memory bandwidth test")
    def gpu_bandwidth_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        typer.echo("Running GPU bandwidth test...")
        result = get_engine().gpu.bandwidth()
        emit_json(result)
        raise typer.Exit(0)

    @gpu_app.command("features", help="Hardware features (TMA, tensor cores, FP8)")
    def gpu_features(ctx: typer.Context) -> None:
        """Display detailed hardware capabilities and features."""
        from core.harness.hardware_capabilities import detect_capabilities

        cap = detect_capabilities()
        if cap is None:
            typer.echo("❌ CUDA not available on this system.")
            raise typer.Exit(code=1)

        # Header
        typer.echo(f"\n🖥️  GPU: {cap.device_name}")
        typer.echo(f"   Architecture: {cap.name} ({cap.sm_version})")
        typer.echo(f"   Compute Capability: {cap.compute_capability}")
        typer.echo(f"   Memory: {cap.total_memory_gb:.1f} GB", nl=False)
        if cap.memory_bandwidth_tbps:
            typer.echo(f" @ {cap.memory_bandwidth_tbps:.2f} TB/s")
        else:
            typer.echo()
        typer.echo(f"   SMs: {cap.num_sms}, Warp Size: {cap.warp_size}")
        typer.echo()

        # Features table
        typer.echo("┌─────────────────────┬────────────────────────────────────┐")
        typer.echo("│ Feature             │ Status                             │")
        typer.echo("├─────────────────────┼────────────────────────────────────┤")

        # Tensor Cores
        tc_status = cap.tensor_cores if cap.tensor_cores else "None"
        typer.echo(f"│ Tensor Cores        │ {tc_status:<34} │")

        # TMA
        if cap.tma_supported:
            tma_status = f"✓ (1D:{cap.tma_limits.max_1d_elements}, 2D:{cap.tma_limits.max_2d_width}×{cap.tma_limits.max_2d_height})"
            if not cap.tma_compiler_supported:
                tma_status += " [compiler N/A]"
        else:
            tma_status = "✗"
        typer.echo(f"│ TMA                 │ {tma_status:<34} │")

        # TMEM (Tensor Memory for MMA accumulators - SM100+ Blackwell)
        tmem_status = "✓" if cap.architecture.lower() in ["blackwell", "blackwell_ultra"] else "✗"
        typer.echo(f"│ TMEM (MMA accum)    │ {tmem_status:<34} │")

        # DSMEM (Distributed Shared Memory - cluster feature)
        if cap.cluster.supports_clusters:
            dsmem_status = f"✓ (max cluster: {cap.cluster.max_cluster_size})"
            if cap.cluster.has_dsmem:
                dsmem_status = f"✓ (cluster: {cap.cluster.max_cluster_size})"
        else:
            dsmem_status = "✗"
        typer.echo(f"│ DSMEM (clusters)    │ {dsmem_status:<34} │")

        # NVLink
        nvlink_status = "✓" if cap.nvlink_c2c else "✗"
        typer.echo(f"│ NVLink C2C          │ {nvlink_status:<34} │")

        # Grace Coherence
        grace_status = "✓" if cap.grace_coherence else "✗"
        typer.echo(f"│ Grace Coherence     │ {grace_status:<34} │")

        # Features from list
        fp8 = "✓" if "FP8" in cap.features or cap.architecture.lower() in ["hopper", "blackwell"] else "✗"
        typer.echo(f"│ FP8                 │ {fp8:<34} │")

        hbm3e = "✓ HBM3e" if "HBM3e" in cap.features else "✗"
        typer.echo(f"│ HBM3e Memory        │ {hbm3e:<34} │")

        # ILP-related info
        typer.echo("├─────────────────────┼────────────────────────────────────┤")
        typer.echo(f"│ Max Threads/Block   │ {cap.max_threads_per_block:<34} │")
        typer.echo(f"│ Max Threads/SM      │ {cap.max_threads_per_sm:<34} │")
        smem_kb = cap.max_shared_mem_per_block / 1024
        typer.echo(f"│ Shared Mem/Block    │ {smem_kb:.0f} KB{'':<28} │")
        if cap.l2_cache_kb:
            typer.echo(f"│ L2 Cache            │ {cap.l2_cache_kb:.0f} KB{'':<28} │")

        typer.echo("└─────────────────────┴────────────────────────────────────┘")

        # Driver/Runtime info
        if cap.driver_version or cap.cuda_runtime_version:
            typer.echo()
            if cap.driver_version:
                typer.echo(f"   Driver: {cap.driver_version}")
            if cap.cuda_runtime_version:
                typer.echo(f"   CUDA Runtime: {cap.cuda_runtime_version}")

        # Notes
        if cap.notes:
            typer.echo("\n📝 Notes:")
            for note in cap.notes:
                typer.echo(f"   • {note}")

        typer.echo()
        raise typer.Exit(0)

# =============================================================================
# Domain 2: System - Software stack, dependencies, capabilities  
# =============================================================================

if typer:

    @system_app.command("status", help="System status overview")
    def system_status_cmd(ctx: typer.Context) -> None:
        from cli.commands import system as system_cmds
        _run(system_cmds.system_status, ctx)

    @system_app.command("software", help="Software versions: PyTorch, CUDA, Python")
    def system_software_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.software()
        if ctx.obj and ctx.obj.get("json_output"):
            emit_json(result)
        else:
            typer.echo("\n📦 Software Stack")
            for key, val in result.items():
                typer.echo(f"  {key}: {val}")
        raise typer.Exit(0)

    @system_app.command("deps", help="Check dependency health")
    def system_deps_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.dependencies()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("capabilities", help="Hardware capabilities (TMA, FP8, etc)")
    def system_capabilities_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.capabilities()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("context", help="Full system context for AI analysis")
    def system_context_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.context()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("parameters", help="Kernel/system parameters (swappiness, dirty ratios)")
    def system_parameters_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.parameters()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("container", help="Container/cgroup limits")
    def system_container_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.container()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("cpu-memory", help="CPU/NUMA/cache hierarchy snapshot")
    def system_cpu_memory_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.cpu_memory()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("env", help="Environment variables and paths")
    def system_env_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.env()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command("network", help="Network + InfiniBand status (structured)")
    def system_network_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().system.network()
        emit_json(result)
        raise typer.Exit(0)

    @system_app.command(
        "clock-lock-check",
        help="Validate harness GPU clock locking works (required for canonical benchmarks)",
    )
    def system_clock_lock_check(
        ctx: typer.Context,
        sm_clock_mhz: Optional[int] = typer.Option(None, "--sm-clock-mhz", help="Target SM clock in MHz (default: max)"),
        mem_clock_mhz: Optional[int] = typer.Option(None, "--mem-clock-mhz", help="Target memory clock in MHz (default: max)"),
    ) -> None:
        from core.engine import get_engine

        result = get_engine().system.clock_lock_check(sm_clock_mhz=sm_clock_mhz, mem_clock_mhz=mem_clock_mhz)
        emit_json(result)
        raise typer.Exit(0)

# =============================================================================
# Domain 4: Analyze - Bottlenecks, pareto, scaling, what-if
# =============================================================================

if typer:

    @analyze_app.command("bottlenecks", help="Identify performance bottlenecks")
    def analyze_bottlenecks_cmd(
        ctx: typer.Context,
        mode: BottleneckMode = typer.Option(BottleneckMode.both, "--mode", "-m", help="Analysis mode")
    ) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.bottlenecks(mode=mode.value)
        emit_json(result)
        raise typer.Exit(0)

    @analyze_app.command("pareto", help="Pareto frontier analysis")
    def analyze_pareto_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.pareto()
        emit_json(result)
        raise typer.Exit(0)

    @analyze_app.command("scaling", help="Scaling analysis with GPU count")
    def analyze_scaling_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.scaling()
        emit_json(result)
        raise typer.Exit(0)

    @analyze_app.command("whatif", help="What-if constraint analysis")
    def analyze_whatif_cmd(
        ctx: typer.Context,
        max_vram: Optional[float] = typer.Option(None, "--max-vram", help="Max VRAM in GB"),
        max_latency: Optional[float] = typer.Option(None, "--max-latency", help="Max latency in ms"),
    ) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.whatif(max_vram_gb=max_vram, max_latency_ms=max_latency)
        emit_json(result)
        raise typer.Exit(0)

    @analyze_app.command("stacking", help="Optimization stacking compatibility")
    def analyze_stacking_cmd(ctx: typer.Context) -> None:
        from core.engine import get_engine
        result = get_engine().analyze.stacking()
        emit_json(result)
        raise typer.Exit(0)

# =============================================================================
# Category: ai
# =============================================================================

if typer and ai_app is not None:

    @ai_app.command("ask", help="Ask a performance question (LLM-powered)")
    def ai_ask(
        ctx: typer.Context,
        question: Optional[List[str]] = typer.Argument(None, help="Question to ask"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.ask_question, ctx, question=question, no_book=no_book)

    @ai_app.command("explain", help="Explain a concept with book + LLM")
    def ai_explain(
        ctx: typer.Context,
        concept: Optional[str] = typer.Argument(None, help="Concept to explain"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.explain_concept, ctx, concept=concept, no_book=no_book)

    @ai_app.command("troubleshoot", help="Diagnose and fix issues")
    def ai_troubleshoot(
        ctx: typer.Context,
        issue: Optional[List[str]] = typer.Argument(None, help="Issue description"),
        no_book: bool = typer.Option(False, "--no-book", help="Skip book citations"),
    ) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.troubleshoot, ctx, issue=issue, no_book=no_book)

    @ai_app.command("status", help="Check LLM backend status")
    def ai_status(ctx: typer.Context) -> None:
        from cli.commands import ai_assistant
        _run(ai_assistant.llm_status, ctx)


# =============================================================================
# Category: profile (merged with analyze)
# =============================================================================

if typer and profile_app is not None:

    # --- Capture commands ---
    
    @profile_app.command("nsys", help="Run Nsight Systems capture")
    def profile_nsys(
        ctx: typer.Context,
        command: str = typer.Argument(..., help='Command to profile (quote it), e.g., "python train.py"'),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for output"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Base artifacts directory (default: ./artifacts/runs)"),
        preset: str = typer.Option("light", "--preset", help="Preset: light or full"),
        trace_cuda: bool = typer.Option(True, "--trace-cuda/--no-trace-cuda", help="Trace CUDA API calls"),
        trace_nvtx: bool = typer.Option(True, "--trace-nvtx/--no-trace-nvtx", help="Trace NVTX ranges"),
        trace_osrt: bool = typer.Option(True, "--trace-osrt/--no-trace-osrt", help="Trace OS runtime"),
        full_timeline: bool = typer.Option(False, "--full-timeline", help="Include cuda-hw/cublas/cudnn traces"),
        trace_forks: bool = typer.Option(False, "--trace-forks", help="Trace child processes before exec"),
        wait_mode: str = typer.Option("primary", "--wait-mode", help="NSYS wait mode: primary or all"),
        finalize_grace_seconds: float = typer.Option(
            20.0,
            "--finalize-grace-seconds",
            help="Grace window after timeout SIGINT for report finalization",
        ),
        force_lineinfo: bool = typer.Option(
            True,
            "--force-lineinfo/--no-force-lineinfo",
            help="Force -lineinfo for source mapping",
        ),
        sanitize_python_startup: bool = typer.Option(
            True,
            "--sanitize-python-startup/--no-sanitize-python-startup",
            help="Prefix a safe sitecustomize shim for profiling subprocesses",
        ),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
    ) -> None:
        """Run Nsight Systems profiling on a command."""
        from cli.commands import profiling
        _run(
            profiling.nsys,
            ctx,
            command=command,
            output_name=output_name,
            output_dir=str(output_dir) if output_dir else None,
            preset=preset,
            trace_cuda=trace_cuda,
            trace_nvtx=trace_nvtx,
            trace_osrt=trace_osrt,
            full_timeline=full_timeline,
            trace_forks=trace_forks,
            wait_mode=wait_mode,
            finalize_grace_seconds=finalize_grace_seconds,
            force_lineinfo=force_lineinfo,
            sanitize_python_startup=sanitize_python_startup,
            timeout_seconds=timeout_seconds,
        )

    @profile_app.command("ncu", help="Run Nsight Compute capture")
    def profile_ncu(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Python script to profile (optional when using --command)"),
        command: Optional[str] = typer.Option(None, "--command", "-c", help='Command to profile (quote it), e.g., "python train.py --batch 8"'),
        script_args: List[str] = typer.Option([], "--arg", "-a", help="Arguments forwarded to the script", show_default=False),
        kernel: Optional[str] = typer.Option(None, "--kernel", help="Specific kernel filter (regex)"),
        kernel_filter: Optional[str] = typer.Option(None, "--kernel-filter", help="Specific kernel filter (regex)"),
        kernel_name_base: Optional[str] = typer.Option(
            None,
            "--kernel-name-base",
            help="NCU kernel name base for filter matching (e.g., function, demangled)",
        ),
        nvtx_include: List[str] = typer.Option(
            [],
            "--nvtx-include",
            help="NCU NVTX include filter (repeatable); useful with --profile-from-start off",
            show_default=False,
        ),
        profile_from_start: Optional[str] = typer.Option(
            None,
            "--profile-from-start",
            help="NCU profiling gate: on/off (set off to capture only after cudaProfilerStart)",
        ),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Output name"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Base artifacts directory (default: artifacts/runs)"),
        workload_type: str = typer.Option(
            "memory_bound",
            "--workload-type",
            help="Workload metrics: memory_bound, compute_bound, tensor_core (used when --metric-set=full)",
            show_default=True,
        ),
        metric_set: str = typer.Option(
            "full",
            "--metric-set",
            help="NCU set: full, roofline, minimal, speed-of-light, basic",
            show_default=True,
        ),
        replay_mode: str = typer.Option(
            "application",
            "--replay-mode",
            help="NCU replay mode: application (all launches) or kernel (one instance per kernel)",
            show_default=True,
        ),
        launch_skip: Optional[int] = typer.Option(None, "--launch-skip", help="Kernel launches to skip before profiling"),
        launch_count: Optional[int] = typer.Option(None, "--launch-count", help="Kernel launches to profile"),
        pm_sampling_interval: Optional[int] = typer.Option(None, "--pm-sampling-interval", help="Nsight Compute pm-sampling-interval (cycles)"),
        force_lineinfo: bool = typer.Option(True, "--force-lineinfo/--no-force-lineinfo", help="Force -lineinfo for source mapping"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.ncu,
            ctx,
            script=str(script) if script else None,
            command=command,
            script_args=script_args,
            kernel=kernel,
            kernel_filter=kernel_filter,
            kernel_name_base=kernel_name_base,
            nvtx_include=nvtx_include,
            profile_from_start=profile_from_start,
            output_name=output_name,
            output_dir=str(output_dir) if output_dir else None,
            workload_type=workload_type,
            metric_set=metric_set,
            replay_mode=replay_mode,
            launch_skip=launch_skip,
            launch_count=launch_count,
            pm_sampling_interval=pm_sampling_interval,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
        )

    @profile_app.command("list-profiles", help="List available baseline/optimized profile pairs")
    def profile_list_profiles(ctx: typer.Context) -> None:
        from core.engine import get_engine

        result = get_engine().profile.list_profiles()
        emit_json(result)
        raise typer.Exit(0)

    @profile_app.command("compile-analysis", help="Summarize torch.compile graphs/breaks from the latest benchmark data")
    def profile_compile_analysis(ctx: typer.Context) -> None:
        from core.engine import get_engine

        result = get_engine().profile.compile_analysis()
        emit_json(result)
        raise typer.Exit(0)

    @profile_app.command("ncu-summary", help="Summarize an NCU report (top kernels + key metrics)")
    def profile_ncu_summary(
        ctx: typer.Context,
        report_path: Path = typer.Argument(..., help="Path to .ncu-rep or exported NCU raw CSV"),
        top_k: int = typer.Option(10, "--top-k", help="Number of kernels to include (sorted by total time when available)"),
        timeout_seconds: int = typer.Option(60, "--timeout", help="Timeout for `ncu --import` when report_path is .ncu-rep"),
    ) -> None:
        from core.engine import get_engine

        result = get_engine().profile.ncu_summary(
            str(report_path),
            top_k=top_k,
            metrics=None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @profile_app.command("torch", help="Run torch.profiler capture (Chrome trace + summary)")
    def profile_torch_profiler(
        ctx: typer.Context,
        script: Path = typer.Argument(..., help="Python script to profile"),
        mode: str = typer.Option("full", "--mode", help="Profiler preset (full/memory/flops/modules/blackwell)"),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for capture folder"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Base artifacts directory (default: artifacts/runs)"),
        nvtx_label: str = typer.Option("aisp_torch_profile", "--nvtx-label", help="NVTX/record_function label for correlation"),
        no_nvtx: bool = typer.Option(False, "--no-nvtx", help="Disable NVTX range emission"),
        force_lineinfo: bool = typer.Option(True, "--force-lineinfo/--no-force-lineinfo", help="Force -lineinfo for better source mapping"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
        script_args: List[str] = typer.Option([], "--arg", "-a", help="Arguments forwarded to the script", show_default=False),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.torch_profiler,
            ctx,
            script=str(script),
            mode=mode,
            output_name=output_name,
            output_dir=str(output_dir) if output_dir else None,
            nvtx_label=nvtx_label,
            use_nvtx=not no_nvtx,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
            script_args=script_args,
        )

    @profile_app.command("hta-capture", help="Launch HTA capture (nsys + HTAAnalyzer)")
    def profile_hta_capture(
        ctx: typer.Context,
        command: str = typer.Argument(..., help='Command to profile (quote it), e.g., "python train.py --batch 8"'),
        preset: str = typer.Option("full", "--preset", help="Nsight preset (full|light)", show_default=True),
        output_name: Optional[str] = typer.Option(None, "--output-name", "-o", help="Base name for capture"),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory for nsys + HTA outputs"),
        force_lineinfo: bool = typer.Option(True, "--force-lineinfo/--no-force-lineinfo", help="Force -lineinfo for source mapping"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.hta_capture,
            ctx,
            command=command,
            preset=preset,
            output_name=output_name or "hta_capture",
            output_dir=str(output_dir) if output_dir else None,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
        )

    # --- View commands ---

    @profile_app.command("flame", help="Generate flame graph")
    def profile_flame(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: str = typer.Option("flame.html", "--output", "-o", help="Output file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.flame, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("memory", help="Memory timeline analysis")
    def profile_memory(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional path to write JSON timeline"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.memory, ctx, file=str(file) if file else None, output=output)

    @profile_app.command("kernels", help="Kernel breakdown analysis")
    def profile_kernels(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.kernels, ctx, file=str(file) if file else None)

    @profile_app.command("hta", help="Holistic Trace Analysis")
    def profile_hta(
        ctx: typer.Context,
        file: Optional[Path] = typer.Argument(None, help="Profile file"),
    ) -> None:
        from cli.commands import profiling
        _run(profiling.hta, ctx, file=str(file) if file else None)

    # --- Compare commands ---

    @profile_app.command("compare", help="Compare baseline vs optimized profiles (flame graph)")
    def profile_compare(
        ctx: typer.Context,
        chapter: Optional[str] = typer.Argument(None, help="Chapter name or path to profile directory"),
        output: str = typer.Option("comparison_flamegraph.html", "--output", "-o", help="Output HTML file"),
        json_out: Optional[str] = typer.Option(None, "--json", "-j", help="Also output JSON data"),
        pair: Optional[str] = typer.Option(None, "--pair", help="Substring or key to select a specific profile pair"),
        include_ncu_details: bool = typer.Option(
            False,
            "--include-ncu-details",
            help="Include NCU source/disassembly details in the comparison (slower).",
        ),
    ) -> None:
        from cli.commands import profiling
        _run(
            profiling.compare_profiles,
            ctx,
            chapter=chapter,
            output=output,
            json_out=json_out,
            pair=pair,
            include_ncu_details=include_ncu_details,
        )

    @profile_app.command("diff", help="Differential analysis between baseline/optimized")
    def profile_diff(
        ctx: typer.Context,
        baseline: Optional[Path] = typer.Argument(None, help="Baseline deep profile JSON"),
        optimized: Optional[Path] = typer.Argument(None, help="Optimized deep profile JSON"),
    ) -> None:
        from cli.commands import analysis
        _run(analysis.diff_analysis, ctx, baseline=str(baseline) if baseline else None, optimized=str(optimized) if optimized else None)

    # --- Analysis commands (merged from analyze) ---

    @profile_app.command("run", help="Run profiling on a benchmark/script")
    def profile_run(
        ctx: typer.Context,
        targets: Optional[List[str]] = typer.Option(
            None, "--targets", "-t", help="Chapter(s) or chapter:example pairs to profile"
        ),
        profile: ProfilePreset = typer.Option(
            ProfilePreset.minimal,
            "--profile",
            "-p",
            help="Benchmark CLI profile preset",
            show_choices=True,
        ),
    ) -> None:
        from cli.commands import analysis
        _run(analysis.run_profile, ctx, targets=targets, profile=profile)

    @profile_app.command("roofline", help="Roofline model analysis")
    def profile_roofline(ctx: typer.Context) -> None:
        from cli.commands import analysis
        _run(analysis.roofline, ctx)

    @profile_app.command("bottleneck", help="Identify performance bottlenecks")
    def profile_bottleneck(
        ctx: typer.Context,
        mode: BottleneckMode = typer.Option(
            BottleneckMode.both,
            "--mode",
            help="Use profile-only, LLM-only, or combined analysis",
            show_choices=True,
        ),
        limit: int = typer.Option(5, "--limit", help="Limit number of bottlenecks shown"),
    ) -> None:
        from cli.commands import analysis
        _run(analysis.bottleneck, ctx, mode=mode, limit=limit)

    # --- Advanced analysis commands ---

    @profile_app.command("warp-divergence", help="Warp divergence analysis")
    def profile_warp_divergence(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.warp_divergence, ctx)

    @profile_app.command("bank-conflicts", help="Bank conflict analysis")
    def profile_bank_conflicts(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.bank_conflicts, ctx)

    @profile_app.command("occupancy", help="Occupancy analysis")
    def profile_occupancy(ctx: typer.Context) -> None:
        from cli.commands import profiling
        _run(profiling.occupancy, ctx)

    @profile_app.command("ilp", help="Instruction-level parallelism analysis")
    def profile_ilp(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze ILP metrics from NCU report or run fresh capture."""
        typer.echo("\n📊 ILP (Instruction-Level Parallelism) Analysis\n")
        
        if report and report.exists():
            # Parse existing NCU report for ILP metrics
            typer.echo(f"Analyzing: {report}")
            # Would use ncu --import to extract metrics
            typer.echo("\nKey ILP Metrics:")
            typer.echo("  • Issued IPC: Instructions per cycle issued")
            typer.echo("  • Executed IPC: Instructions per cycle executed")
            typer.echo("  • Warp cycles per issued instruction")
            typer.echo("  • Pipeline stall reasons")
            typer.echo("\nRun `aisp profile stalls` for detailed stall breakdown.")
        else:
            typer.echo("Usage: aisp profile ilp <ncu-report.ncu-rep>")
            typer.echo("\nTo capture ILP metrics:")
            typer.echo("  ncu --set full --metrics \\")
            typer.echo("    sm__inst_executed_pipe_*,sm__warps_active,sm__inst_issued \\")
            typer.echo("    -o ilp_report python your_script.py")

    @profile_app.command("stalls", help="Pipeline stall analysis")
    def profile_stalls(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze pipeline stall reasons from NCU report."""
        typer.echo("\n⏱️  Pipeline Stall Analysis\n")
        
        typer.echo("Stall Categories:")
        typer.echo("  • Memory Dependency: Waiting for memory operations")
        typer.echo("  • Execution Dependency: Waiting for ALU/FPU results")
        typer.echo("  • Synchronization: Barrier or atomic waits")
        typer.echo("  • Texture: Waiting for texture fetch")
        typer.echo("  • Pipe Busy: Execution units occupied")
        typer.echo("  • Not Selected: Warp not scheduled (low priority)")
        typer.echo()
        
        if report and report.exists():
            typer.echo(f"Analyzing: {report}")
            # Would parse NCU report for stall metrics
        else:
            typer.echo("To capture stall metrics:")
            typer.echo("  ncu --set full --section WarpStateStats \\")
            typer.echo("    -o stall_report python your_script.py")

    @profile_app.command("warmup", help="Warmup/JIT audit")
    def profile_warmup(
        ctx: typer.Context,
        script: Optional[Path] = typer.Argument(None, help="Script to analyze"),
        iterations: int = typer.Option(10, "--iterations", help="Iterations"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.warmup_audit, ctx, script=str(script) if script else None, iterations=iterations)

    @profile_app.command("registers", help="Register pressure and spilling analysis")
    def profile_registers(
        ctx: typer.Context,
        report: Optional[Path] = typer.Argument(None, help="NCU report file (.ncu-rep)"),
    ) -> None:
        """Analyze register usage and spilling from NCU report."""
        typer.echo("\n📊 Register Pressure Analysis\n")
        
        typer.echo("Key Metrics:")
        typer.echo("  • Registers per thread: Target <64 for good occupancy")
        typer.echo("  • Register spills: Local memory accesses (slow)")
        typer.echo("  • Occupancy impact: Fewer registers = more warps")
        typer.echo()
        
        if report and report.exists():
            typer.echo(f"Analyzing: {report}")
            typer.echo("\nTo extract register metrics from NCU:")
            typer.echo("  ncu --query-metrics-mode all | grep -i register")
        else:
            typer.echo("Usage: aisp profile registers <ncu-report.ncu-rep>")
            typer.echo("\nTo capture register metrics:")
            typer.echo("  ncu --set full --section LaunchStats \\")
            typer.echo("    -o register_report python your_script.py")
            typer.echo("\nKey NCU metrics:")
            typer.echo("  • launch__registers_per_thread")
            typer.echo("  • launch__registers_per_thread_allocated")
            typer.echo("  • lts__t_sectors_srcunit_tex_op_read (spill loads)")
            typer.echo("  • lts__t_sectors_srcunit_tex_op_write (spill stores)")

    @profile_app.command("coalescing", help="Memory access coalescing analysis")
    def profile_coalescing(
        ctx: typer.Context,
        stride: int = typer.Option(1, "--stride", help="Access stride in elements"),
        element_size: int = typer.Option(4, "--element-size", help="Element size in bytes"),
    ) -> None:
        """Analyze memory access patterns for coalescing efficiency."""
        typer.echo("\n🔄 Memory Coalescing Analysis\n")
        
        # Calculate coalescing efficiency
        warp_size = 32
        cache_line = 128  # L1/L2 cache line size
        
        access_span = stride * element_size * warp_size
        transactions = (access_span + cache_line - 1) // cache_line
        ideal_transactions = (warp_size * element_size + cache_line - 1) // cache_line
        efficiency = (ideal_transactions / transactions) * 100 if transactions > 0 else 0
        
        typer.echo(f"Configuration:")
        typer.echo(f"  Stride: {stride} elements")
        typer.echo(f"  Element size: {element_size} bytes")
        typer.echo(f"  Warp size: {warp_size} threads")
        typer.echo()
        typer.echo(f"Memory Access Pattern:")
        typer.echo(f"  Access span per warp: {access_span} bytes")
        typer.echo(f"  Cache transactions: {transactions} (ideal: {ideal_transactions})")
        typer.echo(f"  Coalescing efficiency: {efficiency:.1f}%")
        typer.echo()
        
        if efficiency < 50:
            typer.echo("⚠️  Poor coalescing! Consider:")
            typer.echo("  • Transpose data for contiguous access")
            typer.echo("  • Use Structure of Arrays (SoA) instead of AoS")
            typer.echo("  • Pad arrays to avoid bank conflicts")
        elif efficiency < 80:
            typer.echo("⚡ Moderate coalescing. Room for improvement.")
        else:
            typer.echo("✅ Good coalescing efficiency!")

    @profile_app.command("comm-overlap", help="Communication/compute overlap analysis")
    def profile_comm_overlap(
        ctx: typer.Context,
        model: str = typer.Option("llama-3.1-70b", "--model", "-m", help="Model name"),
    ) -> None:
        """Analyze communication/computation overlap opportunities."""
        from core.perf_core_base import PerformanceCore
        
        core = PerformanceCore()
        result = core.get_comm_overlap_analysis(model)
        
        if result.get("success"):
            typer.echo(f"\n🔄 Communication Overlap Analysis: {result.get('model', model)}\n")
            
            inputs = result.get("inputs", {})
            typer.echo(f"Configuration:")
            typer.echo(f"  Model: {inputs.get('params_b', '?')}B params")
            typer.echo(f"  Layout: TP={inputs.get('tp')}, PP={inputs.get('pp')}, DP={inputs.get('dp')}")
            typer.echo(f"  GPUs: {inputs.get('gpus')}")
            typer.echo()
            
            overlap = result.get("overlap_analysis", {})
            if overlap:
                typer.echo(f"Overlap Opportunities:")
                for key, value in overlap.items():
                    if isinstance(value, (int, float)):
                        typer.echo(f"  {key}: {value}")
                    elif isinstance(value, dict):
                        typer.echo(f"  {key}:")
                        for k, v in value.items():
                            typer.echo(f"    {k}: {v}")
            
            if result.get("warning"):
                typer.echo(f"\n⚠️  {result['warning']}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    @profile_app.command("energy", help="Energy efficiency analysis")
    def profile_energy(ctx: typer.Context) -> None:
        """Analyze GPU energy efficiency and power characteristics."""
        from core.perf_core_base import PerformanceCore
        
        core = PerformanceCore()
        result = core.get_energy_analysis()
        
        if result.get("success"):
            typer.echo("\n⚡ Energy Efficiency Analysis\n")
            
            current = result.get("current_state", {})
            if current:
                typer.echo(f"Current Power State:")
                typer.echo(f"  Power draw: {current.get('power_w', '?')} W")
                typer.echo(f"  Temperature: {current.get('temp_c', '?')}°C")
                typer.echo(f"  GPU utilization: {current.get('gpu_util_pct', '?')}%")
                typer.echo(f"  Memory utilization: {current.get('mem_util_pct', '?')}%")
            
            efficiency = result.get("efficiency_metrics", {})
            if efficiency:
                typer.echo(f"\nEfficiency Metrics:")
                for key, value in efficiency.items():
                    typer.echo(f"  {key}: {value}")
            
            recommendations = result.get("recommendations", [])
            if recommendations:
                typer.echo(f"\n💡 Recommendations:")
                for rec in recommendations:
                    typer.echo(f"  • {rec}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    @profile_app.command(
        "data-loading",
        help="Read-only DataLoader settings and CPU/GPU locality observations",
    )
    def profile_data_loading(ctx: typer.Context) -> None:
        """Report configured DataLoader recommendations and observed locality."""
        from core.perf_core_base import PerformanceCoreBase
        
        core = PerformanceCoreBase()
        result = core.get_data_loading_analysis()
        
        if result.get("success"):
            typer.echo("\n📦 Data Loading Analysis (read-only)\n")
            typer.echo(f"Schema: {result.get('schema_version', 'unknown')}")

            settings = result.get("dataloader_kwargs", {})
            if settings:
                typer.echo("\nRecommended DataLoader Settings:")
                for key, value in settings.items():
                    typer.echo(f"  {key}: {value}")

            cpu = result.get("cpu", {})
            typer.echo("\nObserved CPU:")
            typer.echo(f"  architecture: {cpu.get('cpu_arch') or 'unknown'}")
            typer.echo(f"  model: {cpu.get('cpu_model') or 'unknown'}")
            typer.echo(f"  Grace status: {cpu.get('grace_status', 'unknown')}")

            mapping = result.get("gpu_numa_mapping", {})
            gpu_id = result.get("gpu_id")
            typer.echo("\nObserved GPU/NUMA Locality:")
            typer.echo(f"  GPU: {gpu_id if gpu_id is not None else 'unknown'}")
            typer.echo(f"  mapping status: {mapping.get('status', 'unknown')}")
            numa_node = mapping.get("numa_node")
            typer.echo(f"  NUMA node: {numa_node if numa_node is not None else 'unknown'}")
            typer.echo(f"  provenance: {mapping.get('provenance', 'unknown')}")

            affinity = result.get("current_cpu_affinity")
            typer.echo("\nObserved Process Affinity:")
            typer.echo(f"  status: {result.get('current_affinity_status', 'unknown')}")
            typer.echo(f"  CPU IDs: {affinity if affinity is not None else 'unknown'}")
            typer.echo(f"  affinity applied: {result.get('affinity_applied', False)}")
            typer.echo(
                f"  provenance: {result.get('current_affinity_provenance', 'unknown')}"
            )

            notes = result.get("notes", [])
            if notes:
                typer.echo("\nNotes:")
                for note in notes:
                    typer.echo(f"  • {note}")
        else:
            typer.echo(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            raise typer.Exit(code=1)


# =============================================================================
# Category: optimize
# =============================================================================

if typer and optimize_app is not None:

    @optimize_app.command("recommend", help="Get optimization recommendations")
    def optimize_recommend(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        goal: GoalChoice = typer.Option(
            GoalChoice.throughput,
            "--goal",
            help="Optimization goal",
            show_choices=True,
        ),
    ) -> None:
        from cli.commands import optimization
        _run(optimization.recommend, ctx, model_size=model_size, gpus=gpus, goal=goal)

    @optimize_app.command("auto", help="Auto-optimization with LLM guidance")
    def optimize_auto(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.auto_optimize, ctx)

    @optimize_app.command("whatif", help="What-if analysis for optimizations")
    def optimize_whatif(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.whatif, ctx)

    @optimize_app.command("stacking", help="Compound optimization stacking")
    def optimize_stacking(ctx: typer.Context) -> None:
        from cli.commands import optimization
        _run(optimization.stacking, ctx)

    @optimize_app.command("playbook", help="Pre-built optimization playbooks")
    def optimize_playbook(
        ctx: typer.Context,
        action: str = typer.Argument("list", help='Playbook name or "list"'),
    ) -> None:
        from cli.commands import optimization
        _run(optimization.playbook, ctx, action=action)


# =============================================================================
# Category: distributed
# =============================================================================

if typer and distributed_app is not None:

    @distributed_app.command("plan", help="Plan parallelism strategy (TP/PP/DP)")
    def distributed_plan(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.plan_parallelism, ctx, model_size=model_size, gpus=gpus, nodes=nodes)

    @distributed_app.command("topology", help="Analyze GPU topology")
    def distributed_topology(ctx: typer.Context) -> None:
        from cli.commands import distributed
        _run(distributed.topology, ctx)

    @distributed_app.command("nccl", help="NCCL tuning recommendations")
    def distributed_nccl(
        ctx: typer.Context,
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        diagnose: bool = typer.Option(False, "--diagnose", help="Include diagnostic checks"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.nccl_tuning, ctx, nodes=nodes, gpus=gpus, diagnose=diagnose)

    @distributed_app.command("zero", help="ZeRO/FSDP configuration")
    def distributed_zero(
        ctx: typer.Context,
        model_size: float = typer.Option(70.0, "--model-size", help="Model size in billions"),
        gpus: int = typer.Option(8, "--gpus", help="Number of GPUs"),
    ) -> None:
        from cli.commands import distributed
        _run(distributed.zero_config, ctx, model_size=model_size, gpus=gpus)

    @distributed_app.command("launch-plan", help="Generate torchrun launch plan (dry-run/save).")
    def distributed_launch_plan(
        ctx: typer.Context,
        model_params: int = typer.Option(70, "--model-params", help="Model size in billions"),
        nodes: int = typer.Option(1, "--nodes", help="Nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        tp: int = typer.Option(1, "--tp", help="Tensor parallel degree"),
        pp: int = typer.Option(1, "--pp", help="Pipeline parallel degree"),
        dp: int = typer.Option(1, "--dp", help="Data parallel degree"),
        batch_size: int = typer.Option(1, "--batch-size", help="Global batch size"),
        script: str = typer.Option("train.py", "--script", help="Entry script to launch"),
        extra_args: Optional[str] = typer.Option(None, "--extra-args", help="Extra args to append to command"),
        save_plan: Optional[Path] = typer.Option(None, "--save-plan", help="Optional path to save plan JSON"),
    ) -> None:
        from core.optimization.parallelism_planner.launch_plan import generate_launch_plan

        try:
            plan = generate_launch_plan(
                model_params=model_params,
                nodes=nodes,
                gpus_per_node=gpus,
                tp=tp,
                pp=pp,
                dp=dp,
                batch_size=batch_size,
                script=script,
                extra_args=extra_args,
            )
        except Exception as exc:
            typer.echo(f"Failed to build launch plan: {exc}", err=True)
            raise typer.Exit(code=1)

        typer.echo("\n🧭 Launch Plan")
        typer.echo(f"Model params: {plan.model_params}B")
        typer.echo(f"Layout: TP={plan.tp} PP={plan.pp} DP={plan.dp} on {plan.nodes}x{plan.gpus_per_node}")
        typer.echo(f"Command:\n  {plan.command}")
        if save_plan:
            save_plan.write_text(plan.to_json())
            typer.echo(f"\n💾 Saved plan to {save_plan}")

    @distributed_app.command("slurm", help="Generate SLURM job script")
    def distributed_slurm(
        ctx: typer.Context,
        model: str = typer.Option("7b", "--model", help="Model name or size"),
        nodes: int = typer.Option(1, "--nodes", help="Number of nodes"),
        gpus: int = typer.Option(8, "--gpus", help="GPUs per node"),
        framework: str = typer.Option("pytorch", "--framework", help="Framework name"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output file"),
    ) -> None:
        from core.engine import get_engine

        result = get_engine().distributed.slurm(model=model, nodes=nodes, gpus=gpus, framework=framework)
        if not isinstance(result, dict) or not result.get("success"):
            message = result.get("error") if isinstance(result, dict) else "Unknown failure"
            typer.echo(f"Failed to generate SLURM script: {message}", err=True)
            raise typer.Exit(code=1)
        script = result.get("slurm_script")
        if not script:
            typer.echo("SLURM script was empty.", err=True)
            raise typer.Exit(code=1)
        if output:
            output.write_text(script)
            typer.echo(f"✅ Wrote SLURM script to {output}")
        else:
            typer.echo(script)


# =============================================================================
# Category: inference
# =============================================================================

if typer and inference_app is not None:

    @inference_app.command("vllm", help="vLLM configuration and optimization")
    def inference_vllm(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(None, "--model", help="Model name"),
        model_size: Optional[float] = typer.Option(
            None, "--model-size", help="Model size in billions (required)"
        ),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        gpu_memory_gb: float = typer.Option(80.0, "--gpu-memory-gb", help="VRAM per GPU (GB)"),
        target: GoalChoice = typer.Option(
            GoalChoice.throughput,
            "--target",
            help="Optimization target",
            show_choices=True,
        ),
        max_seq_length: int = typer.Option(8192, "--max-seq-length", help="Max sequence length"),
        quantization: Optional[str] = typer.Option(None, "--quantization", help="Quantization (awq/gptq/fp8/int8)"),
        compare: bool = typer.Option(False, "--compare", help="Compare inference engines instead of config"),
    ) -> None:
        from cli.commands import inference
        _run(
            inference.vllm_config,
            ctx,
            model=model,
            model_size=model_size,
            gpus=gpus,
            gpu_memory_gb=gpu_memory_gb,
            target=target,
            max_seq_length=max_seq_length,
            quantization=quantization,
            compare=compare,
        )

    @inference_app.command("quantize", help="Quantization recommendations")
    def inference_quantize(
        ctx: typer.Context,
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions"),
        target_memory: Optional[float] = typer.Option(
            None, "--target-memory", help="Target memory per GPU (GB)"
        ),
    ) -> None:
        from cli.commands import inference
        _run(inference.quantize, ctx, model_size=model_size, target_memory=target_memory)

    @inference_app.command("deploy", help="Deployment configuration (explicit model size required)")
    def inference_deploy(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions (required)"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        gpu_memory_gb: float = typer.Option(80.0, "--gpu-memory-gb", help="VRAM per GPU (GB)"),
        goal: str = typer.Option("throughput", "--goal", help="Optimization goal (throughput/latency/memory)"),
        max_seq_length: int = typer.Option(8192, "--max-seq-length", help="Max sequence length"),
    ) -> None:
        from cli.commands import inference
        _run(
            inference.deploy_config,
            ctx,
            model=model,
            model_size=model_size,
            gpus=gpus,
            gpu_memory_gb=gpu_memory_gb,
            goal=goal,
            max_seq_length=max_seq_length,
        )

    @inference_app.command("serve", help="Generate (and optionally run) inference server command")
    def inference_serve(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions (required)"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        gpu_memory_gb: float = typer.Option(80.0, "--gpu-memory-gb", help="VRAM per GPU (GB)"),
        goal: str = typer.Option("throughput", "--goal", help="Optimization goal (throughput/latency/memory)"),
        max_seq_length: int = typer.Option(8192, "--max-seq-length", help="Max sequence length"),
        run: bool = typer.Option(False, "--run", help="Execute the launch command"),
    ) -> None:
        from cli.commands import inference
        _run(
            inference.serve,
            ctx,
            model=model,
            model_size=model_size,
            gpus=gpus,
            gpu_memory_gb=gpu_memory_gb,
            goal=goal,
            max_seq_length=max_seq_length,
            run=run,
        )

    @inference_app.command("estimate", help="Estimate inference throughput/latency")
    def inference_estimate(
        ctx: typer.Context,
        model: str = typer.Option("meta-llama/Llama-2-70b-hf", "--model", help="Model name"),
        model_size: Optional[float] = typer.Option(None, "--model-size", help="Model size in billions (required)"),
        gpus: int = typer.Option(1, "--gpus", help="Number of GPUs"),
        gpu_memory_gb: float = typer.Option(80.0, "--gpu-memory-gb", help="VRAM per GPU (GB)"),
        goal: str = typer.Option("throughput", "--goal", help="Optimization goal (throughput/latency/memory)"),
        max_seq_length: int = typer.Option(8192, "--max-seq-length", help="Max sequence length"),
    ) -> None:
        from cli.commands import inference
        _run(
            inference.estimate,
            ctx,
            model=model,
            model_size=model_size,
            gpus=gpus,
            gpu_memory_gb=gpu_memory_gb,
            goal=goal,
            max_seq_length=max_seq_length,
        )


# =============================================================================
# Category: benchmark (hardware microbench)
# =============================================================================

if typer:

    @benchmark_app.command("memory", help="GPU VRAM bandwidth test")
    def hw_memory(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Buffer size (MB)"),
        stride: int = typer.Option(128, "--stride", help="Stride (bytes)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.mem_hierarchy, ctx, size_mb=size_mb, stride=stride)

    @benchmark_app.command("cache", help="Memory hierarchy stride test (cache behavior)")
    def hw_cache(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Buffer size (MB)"),
        stride: int = typer.Option(128, "--stride", help="Stride (bytes)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.mem_hierarchy, ctx, size_mb=size_mb, stride=stride)

    @benchmark_app.command("roofline", help="Stride sweep ASCII roofline for memory")
    def hw_roofline(
        ctx: typer.Context,
        size_mb: int = typer.Option(32, "--size-mb", help="Buffer size (MB)"),
        strides: Optional[List[int]] = typer.Option(None, "--stride", help="Stride values (repeatable)"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.mem_roofline, ctx, size_mb=size_mb, strides=strides)

    @benchmark_app.command("pcie", help="PCIe H2D/D2H bandwidth")
    def hw_pcie(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        iters: int = typer.Option(10, "--iters", "-i", help="Iterations"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.pcie, ctx, size_mb=size_mb, iters=iters)

    @benchmark_app.command("tc", help="Tensor core throughput (TFLOPS)")
    def hw_tc(
        ctx: typer.Context,
        size: int = typer.Option(4096, "--size", help="Matrix size"),
        precision: str = typer.Option("fp16", "--precision", help="Precision (fp16/fp32/bf16/fp8)"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.tensor_core, ctx, size=size, precision=precision)

    @benchmark_app.command("sfu", help="SFU (Special Function Units) benchmark")
    def hw_sfu(
        ctx: typer.Context,
        elements: int = typer.Option(64 * 1024 * 1024, "--elements", help="Number of elements"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.sfu, ctx, elements=elements)

    @benchmark_app.command("disk", help="Disk I/O benchmark (sequential)")
    def hw_disk(
        ctx: typer.Context,
        file_size_mb: int = typer.Option(256, "--size-mb", "-s", help="File size (MB)"),
        block_size_kb: int = typer.Option(1024, "--block-kb", "-b", help="Block size (KB)"),
        tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temporary directory"),
    ) -> None:
        from core.diagnostics import microbench
        _run(
            microbench.disk,
            ctx,
            file_size_mb=file_size_mb,
            block_size_kb=block_size_kb,
            tmp_dir=str(tmp_dir) if tmp_dir else None,
        )

    @benchmark_app.command("tcp", help="TCP loopback throughput")
    def hw_tcp(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        port: int = typer.Option(5789, "--port", help="Port to use"),
    ) -> None:
        from core.diagnostics import microbench
        _run(microbench.loopback, ctx, size_mb=size_mb, port=port)

    @benchmark_app.command("ib", help="InfiniBand bandwidth test")
    def hw_ib(
        ctx: typer.Context,
        size_mb: int = typer.Option(64, "--size-mb", "-s", help="Transfer size (MB)"),
        duration: int = typer.Option(5, "--duration", "-d", help="Test duration (seconds)"),
    ) -> None:
        """Run InfiniBand bandwidth test using ib_write_bw or NCCL."""
        import shutil
        
        typer.echo("\n📡 InfiniBand Bandwidth Test\n")
        
        ib_write_bw = shutil.which("ib_write_bw")
        if ib_write_bw:
            typer.echo("Using: ib_write_bw (perftest)")
            typer.echo(f"Size: {size_mb} MB, Duration: {duration}s")
            typer.echo("\nTo run manually between two nodes:")
            typer.echo("  Server: ib_write_bw -d mlx5_0")
            typer.echo("  Client: ib_write_bw -d mlx5_0 <server_ip>")
        else:
            typer.echo("ib_write_bw not found. Install perftest package:")
            typer.echo("  apt install perftest  # or yum install perftest")
            typer.echo("\nAlternative: Use NCCL tests")
            typer.echo("  all_reduce_perf -b 8M -e 256M -g 8")

    @benchmark_app.command("nccl", help="NCCL collective bandwidth (all_reduce, etc.)")
    def hw_nccl(
        ctx: typer.Context,
        collective: str = typer.Option("all_reduce", "--collective", "-c", help="Collective type"),
        min_bytes: str = typer.Option("8M", "--min", help="Min message size"),
        max_bytes: str = typer.Option("256M", "--max", help="Max message size"),
        gpus: int = typer.Option(8, "--gpus", "-g", help="Number of GPUs"),
    ) -> None:
        """Run NCCL collective bandwidth test."""
        import shutil
        
        typer.echo("\n🔄 NCCL Collective Bandwidth Test\n")
        
        # Map collective name to binary
        collective_bins = {
            "all_reduce": "all_reduce_perf",
            "all_gather": "all_gather_perf", 
            "reduce_scatter": "reduce_scatter_perf",
            "broadcast": "broadcast_perf",
            "reduce": "reduce_perf",
            "alltoall": "alltoall_perf",
        }
        
        bin_name = collective_bins.get(collective, f"{collective}_perf")
        bin_path = shutil.which(bin_name)
        
        if bin_path:
            cmd = [bin_path, "-b", min_bytes, "-e", max_bytes, "-g", str(gpus)]
            typer.echo(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            raise typer.Exit(code=result.returncode)
        else:
            typer.echo(f"{bin_name} not found. Install nccl-tests:")
            typer.echo("  git clone https://github.com/NVIDIA/nccl-tests")
            typer.echo("  cd nccl-tests && make MPI=1")
            typer.echo(f"\nManual command:")
            typer.echo(f"  {bin_name} -b {min_bytes} -e {max_bytes} -g {gpus}")

    @benchmark_app.command("p2p", help="GPU-to-GPU P2P/NVLink bandwidth")
    def hw_p2p(
        ctx: typer.Context,
        size_mb: int = typer.Option(256, "--size-mb", "-s", help="Transfer size (MB)"),
        bidirectional: bool = typer.Option(False, "--bidir", "-b", help="Bidirectional test"),
    ) -> None:
        """Test GPU-to-GPU peer-to-peer bandwidth over NVLink/PCIe."""
        try:
            import torch
        except ImportError:
            typer.echo("❌ PyTorch not available")
            raise typer.Exit(code=1)
        
        if not torch.cuda.is_available():
            typer.echo("❌ CUDA not available")
            raise typer.Exit(code=1)
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            typer.echo("❌ P2P test requires at least 2 GPUs")
            raise typer.Exit(code=1)
        
        typer.echo(f"\n🔗 GPU P2P Bandwidth Test ({num_gpus} GPUs)\n")
        typer.echo(f"Transfer size: {size_mb} MB")
        typer.echo(f"Mode: {'Bidirectional' if bidirectional else 'Unidirectional'}\n")
        
        size_bytes = size_mb * 1024 * 1024
        
        # Test P2P between all pairs
        typer.echo("GPU Pair    │ P2P Enabled │ Bandwidth (GB/s)")
        typer.echo("────────────┼─────────────┼─────────────────")
        
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                
                can_access = torch.cuda.can_device_access_peer(i, j)
                
                # Allocate tensors
                with torch.cuda.device(i):
                    src = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{i}")
                with torch.cuda.device(j):
                    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{j}")
                
                # Warmup
                dst.copy_(src)
                torch.cuda.synchronize()
                
                # Benchmark
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                iters = 10
                start.record()
                for _ in range(iters):
                    dst.copy_(src)
                end.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start.elapsed_time(end)
                bw_gbps = (size_bytes * iters / (elapsed_ms / 1000)) / 1e9
                
                p2p_str = "✓" if can_access else "✗"
                typer.echo(f"GPU {i}→{j}    │ {p2p_str:^11} │ {bw_gbps:>8.2f}")
        
        typer.echo()

    @benchmark_app.command("speed", help="Run speed tests (GEMM, memory, attention)")
    def hw_speed(
        ctx: typer.Context,
        type: SpeedTestChoice = typer.Option(
            SpeedTestChoice.all,
            "--type",
            help="Speed test type",
            show_choices=True,
        ),
        gemm_size: int = typer.Option(512, "--gemm-size", help="Square GEMM size (e.g., 512 => 512x512)"),
        precision: str = typer.Option("fp16", "--precision", help="Precision for GEMM (fp16/bf16/tf32/fp32/fp8)"),
        mem_size_mb: int = typer.Option(16, "--mem-size-mb", help="Memory test size (MB)"),
        mem_stride: int = typer.Option(128, "--mem-stride", help="Memory test stride (bytes)"),
    ) -> None:
        from cli.commands import tests as test_cmds
        _run(
            test_cmds.speedtest,
            ctx,
            type=type,
            gemm_size=gemm_size,
            precision=precision,
            mem_size_mb=mem_size_mb,
            mem_stride=mem_stride,
        )

    @benchmark_app.command("diagnostics", help="System diagnostics")
    def hw_diagnostics(ctx: typer.Context) -> None:
        from cli.commands import tests as test_cmds
        _run(test_cmds.diagnostics, ctx)


# =============================================================================
# Extra group: cluster
# =============================================================================

if typer and cluster_app is not None:

    @cluster_app.command("eval-suite", help="Run the cluster eval suite (full) or a fast local smoke run")
    def cluster_eval_suite_cmd(
        ctx: typer.Context,
        mode: str = typer.Option("smoke", "--mode", help="smoke (local metadata+manifest) or full (cluster suite)"),
        run_id: Optional[str] = typer.Option(None, "--run-id", help="RUN_ID prefix (default: YYYY-MM-DD)"),
        hosts: Optional[str] = typer.Option(None, "--hosts", help="Comma-separated host list (required for --mode full)"),
        labels: Optional[str] = typer.Option(None, "--labels", help="Comma-separated labels (optional; must match hosts count)"),
        ssh_user: Optional[str] = typer.Option(None, "--ssh-user", help="SSH user (full mode)"),
        ssh_key: Optional[Path] = typer.Option(None, "--ssh-key", help="SSH key path (full mode)"),
        oob_if: Optional[str] = typer.Option(None, "--oob-if", help="Out-of-band interface (full mode, multi-node)"),
        socket_ifname: Optional[str] = typer.Option(None, "--socket-ifname", help="NCCL socket interface (full mode)"),
        nccl_ib_hca: Optional[str] = typer.Option(None, "--nccl-ib-hca", help="NCCL_IB_HCA allowlist (full mode)"),
        primary_label: Optional[str] = typer.Option(None, "--primary-label", help="Label for single-node/local steps"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
        extra_arg: List[str] = typer.Option([], "--extra-arg", help="Extra args appended to run_cluster_eval_suite.sh", show_default=False),
    ) -> None:
        from core.cluster import run_cluster_eval_suite

        host_list = [h.strip() for h in (hosts or "").split(",") if h.strip()] if hosts else None
        label_list = [l.strip() for l in (labels or "").split(",") if l.strip()] if labels else None

        result = run_cluster_eval_suite(
            mode=mode,
            run_id=run_id,
            hosts=host_list,
            labels=label_list,
            ssh_user=ssh_user,
            ssh_key=str(ssh_key) if ssh_key else None,
            oob_if=oob_if,
            socket_ifname=socket_ifname,
            nccl_ib_hca=nccl_ib_hca,
            primary_label=primary_label,
            extra_args=extra_arg or None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("common-eval", help="Run a preset system-evaluation bundle for common benchmark questions")
    def cluster_common_eval_cmd(
        ctx: typer.Context,
        preset: str = typer.Option("core-system", "--preset", help="Preset: common-answer-fast, core-system, modern-llm, fabric-systems, or multinode-readiness"),
        run_id: Optional[str] = typer.Option(None, "--run-id", help="RUN_ID prefix (default: YYYY-MM-DD)"),
        hosts: str = typer.Option(..., "--hosts", help="Comma-separated host list"),
        labels: Optional[str] = typer.Option(None, "--labels", help="Comma-separated labels (optional; must match hosts count)"),
        ssh_user: Optional[str] = typer.Option(None, "--ssh-user", help="SSH user"),
        ssh_key: Optional[Path] = typer.Option(None, "--ssh-key", help="SSH key path"),
        oob_if: Optional[str] = typer.Option(None, "--oob-if", help="Out-of-band interface (required for multi-node readiness and network-heavy presets)"),
        socket_ifname: Optional[str] = typer.Option(None, "--socket-ifname", help="NCCL socket interface"),
        nccl_ib_hca: Optional[str] = typer.Option(None, "--nccl-ib-hca", help="NCCL_IB_HCA allowlist"),
        nmx_url: Optional[str] = typer.Option(None, "--nmx-url", help="NMX base URL for the target NVLink management plane"),
        nmx_token: Optional[str] = typer.Option(None, "--nmx-token", help="Optional NMX bearer token"),
        ib_mgmt_host: Optional[str] = typer.Option(None, "--ib-mgmt-host", help="InfiniBand management host for IB CLI checks"),
        ib_mgmt_user: Optional[str] = typer.Option(None, "--ib-mgmt-user", help="SSH user for the InfiniBand management host"),
        ib_mgmt_ssh_key: Optional[Path] = typer.Option(None, "--ib-mgmt-ssh-key", help="SSH key path for the InfiniBand management host"),
        cumulus_hosts: Optional[str] = typer.Option(None, "--cumulus-hosts", help="Comma-separated Cumulus/Spectrum-X switch hosts"),
        cumulus_user: Optional[str] = typer.Option(None, "--cumulus-user", help="SSH user for Cumulus/Spectrum-X switches"),
        cumulus_ssh_key: Optional[Path] = typer.Option(None, "--cumulus-ssh-key", help="SSH key path for Cumulus/Spectrum-X switches"),
        primary_label: Optional[str] = typer.Option(None, "--primary-label", help="Label for single-node/local steps"),
        coverage_baseline_run_id: Optional[str] = typer.Option(None, "--coverage-baseline-run-id", help="Optional baseline run id for coverage delta output"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
        extra_arg: List[str] = typer.Option([], "--extra-arg", help="Extra args appended after the preset flags", show_default=False),
    ) -> None:
        from core.cluster import run_cluster_common_eval

        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
        label_list = [l.strip() for l in (labels or "").split(",") if l.strip()] if labels else None

        result = run_cluster_common_eval(
            preset=preset,
            run_id=run_id,
            hosts=host_list,
            labels=label_list,
            ssh_user=ssh_user,
            ssh_key=str(ssh_key) if ssh_key else None,
            oob_if=oob_if,
            socket_ifname=socket_ifname,
            nccl_ib_hca=nccl_ib_hca,
            nmx_url=nmx_url,
            nmx_token=nmx_token,
            ib_mgmt_host=ib_mgmt_host,
            ib_mgmt_user=ib_mgmt_user,
            ib_mgmt_ssh_key=str(ib_mgmt_ssh_key) if ib_mgmt_ssh_key else None,
            cumulus_hosts=[h.strip() for h in (cumulus_hosts or "").split(",") if h.strip()] if cumulus_hosts else None,
            cumulus_user=cumulus_user,
            cumulus_ssh_key=str(cumulus_ssh_key) if cumulus_ssh_key else None,
            primary_label=primary_label,
            coverage_baseline_run_id=coverage_baseline_run_id,
            extra_args=extra_arg or None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("fabric-eval", help="Run the canonical AI fabric evaluation bundle")
    def cluster_fabric_eval_cmd(
        ctx: typer.Context,
        run_id: Optional[str] = typer.Option(None, "--run-id", help="RUN_ID prefix (default: YYYY-MM-DD)"),
        hosts: str = typer.Option(..., "--hosts", help="Comma-separated host list"),
        labels: Optional[str] = typer.Option(None, "--labels", help="Comma-separated labels (optional; must match hosts count)"),
        ssh_user: Optional[str] = typer.Option(None, "--ssh-user", help="SSH user"),
        ssh_key: Optional[Path] = typer.Option(None, "--ssh-key", help="SSH key path"),
        oob_if: Optional[str] = typer.Option(None, "--oob-if", help="Out-of-band interface"),
        socket_ifname: Optional[str] = typer.Option(None, "--socket-ifname", help="NCCL socket interface"),
        nccl_ib_hca: Optional[str] = typer.Option(None, "--nccl-ib-hca", help="NCCL_IB_HCA allowlist"),
        nmx_url: Optional[str] = typer.Option(None, "--nmx-url", help="NMX base URL for the target NVLink management plane"),
        nmx_token: Optional[str] = typer.Option(None, "--nmx-token", help="Optional NMX bearer token"),
        ib_mgmt_host: Optional[str] = typer.Option(None, "--ib-mgmt-host", help="InfiniBand management host for IB CLI checks"),
        ib_mgmt_user: Optional[str] = typer.Option(None, "--ib-mgmt-user", help="SSH user for the InfiniBand management host"),
        ib_mgmt_ssh_key: Optional[Path] = typer.Option(None, "--ib-mgmt-ssh-key", help="SSH key path for the InfiniBand management host"),
        cumulus_hosts: Optional[str] = typer.Option(None, "--cumulus-hosts", help="Comma-separated Cumulus/Spectrum-X switch hosts"),
        cumulus_user: Optional[str] = typer.Option(None, "--cumulus-user", help="SSH user for Cumulus/Spectrum-X switches"),
        cumulus_ssh_key: Optional[Path] = typer.Option(None, "--cumulus-ssh-key", help="SSH key path for Cumulus/Spectrum-X switches"),
        primary_label: Optional[str] = typer.Option(None, "--primary-label", help="Label for single-node/local steps"),
        coverage_baseline_run_id: Optional[str] = typer.Option(None, "--coverage-baseline-run-id", help="Optional baseline run id for coverage delta output"),
        require_management_plane: bool = typer.Option(False, "--require-management-plane", help="Fail publish-grade completeness when management-plane endpoints are missing"),
        timeout_seconds: Optional[int] = typer.Option(None, "--timeout", help="Optional timeout in seconds"),
        extra_arg: List[str] = typer.Option([], "--extra-arg", help="Extra args appended after the fabric preset flags", show_default=False),
    ) -> None:
        from core.cluster import run_cluster_fabric_eval

        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
        label_list = [l.strip() for l in (labels or "").split(",") if l.strip()] if labels else None

        result = run_cluster_fabric_eval(
            run_id=run_id,
            hosts=host_list,
            labels=label_list,
            ssh_user=ssh_user,
            ssh_key=str(ssh_key) if ssh_key else None,
            oob_if=oob_if,
            socket_ifname=socket_ifname,
            nccl_ib_hca=nccl_ib_hca,
            nmx_url=nmx_url,
            nmx_token=nmx_token,
            ib_mgmt_host=ib_mgmt_host,
            ib_mgmt_user=ib_mgmt_user,
            ib_mgmt_ssh_key=str(ib_mgmt_ssh_key) if ib_mgmt_ssh_key else None,
            cumulus_hosts=[h.strip() for h in (cumulus_hosts or "").split(",") if h.strip()] if cumulus_hosts else None,
            cumulus_user=cumulus_user,
            cumulus_ssh_key=str(cumulus_ssh_key) if cumulus_ssh_key else None,
            primary_label=primary_label,
            coverage_baseline_run_id=coverage_baseline_run_id,
            require_management_plane=require_management_plane,
            extra_args=extra_arg or None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("nmx-partition-lab", help="Build a lab-only NVLink/NMX partition workflow guide from live inventory")
    def cluster_nmx_partition_lab_cmd(
        ctx: typer.Context,
        nmx_url: Optional[str] = typer.Option(None, "--nmx-url", help="NMX base URL for the target fabric"),
        nmx_token: Optional[str] = typer.Option(None, "--nmx-token", help="Optional NMX bearer token"),
        alpha_name: str = typer.Option("AlphaPartition", "--alpha-name", help="Alpha partition name"),
        beta_name: str = typer.Option("BetaPartition", "--beta-name", help="Beta partition name"),
        alpha_size: int = typer.Option(4, "--alpha-size", help="Alpha partition GPU count"),
        beta_size: int = typer.Option(4, "--beta-size", help="Beta partition GPU count"),
    ) -> None:
        from core.cluster import build_cluster_nmx_partition_lab

        result = build_cluster_nmx_partition_lab(
            nmx_url=nmx_url,
            nmx_token=nmx_token,
            alpha_name=alpha_name,
            beta_name=beta_name,
            alpha_size=alpha_size,
            beta_size=beta_size,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("validate-field-report", help="Validate cluster field report requirements + artifact hygiene")
    def cluster_validate_field_report_cmd(
        ctx: typer.Context,
        canonical_run_id: Optional[str] = typer.Option(None, "--canonical-run-id", help="Expected canonical run id (optional)"),
        report: Optional[Path] = typer.Option(None, "--report", help="Path to field-report.md"),
        notes: Optional[Path] = typer.Option(None, "--notes", help="Path to field-report-notes.md"),
        template: Optional[Path] = typer.Option(None, "--template", help="Path to field-report-template.md"),
        runbook: Optional[Path] = typer.Option(None, "--runbook", help="Path to advanced-runbook.md"),
        allow_run_id: List[str] = typer.Option([], "--allow-run-id", help="Additional run id allowed for hygiene checks", show_default=False),
        timeout_seconds: int = typer.Option(120, "--timeout", help="Validator timeout in seconds"),
    ) -> None:
        from core.cluster import validate_field_report_requirements

        result = validate_field_report_requirements(
            report=str(report) if report else None,
            notes=str(notes) if notes else None,
            template=str(template) if template else None,
            runbook=str(runbook) if runbook else None,
            canonical_run_id=canonical_run_id,
            allow_run_id=allow_run_id or None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("build-canonical-package", help="Materialize a clean canonical cluster package from selected runs")
    def cluster_build_canonical_package_cmd(
        ctx: typer.Context,
        canonical_run_id: str = typer.Option(..., "--canonical-run-id", help="Primary canonical run id to package"),
        comparison_run_id: List[str] = typer.Option([], "--comparison-run-id", help="Additional comparison/baseline run ids", show_default=False),
        historical_run_id: List[str] = typer.Option([], "--historical-run-id", help="Historical run ids to preserve as references or package when present", show_default=False),
        output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Package output directory (must not already contain files)"),
        timeout_seconds: int = typer.Option(300, "--timeout", help="Builder timeout in seconds"),
    ) -> None:
        from core.cluster import build_canonical_package

        result = build_canonical_package(
            canonical_run_id=canonical_run_id,
            comparison_run_ids=comparison_run_id or None,
            historical_run_ids=historical_run_id or None,
            output_dir=str(output_dir) if output_dir else None,
            timeout_seconds=timeout_seconds,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("promote-run", help="Promote one run-local cluster result tree into the published cluster package")
    def cluster_promote_run_cmd(
        ctx: typer.Context,
        run_id: str = typer.Option(..., "--run-id", help="Run id under cluster/runs/<run_id>"),
        label: str = typer.Option("localhost", "--label", help="Host label for localhost report rendering"),
        allow_run_id: List[str] = typer.Option([], "--allow-run-id", help="Additional run ids to retain during cleanup/validation hygiene checks", show_default=False),
        publish_report_path: Optional[Path] = typer.Option(None, "--publish-report-path", help="Published localhost report path"),
        publish_notes_path: Optional[Path] = typer.Option(None, "--publish-notes-path", help="Published localhost notes path"),
        repo_root: Optional[Path] = typer.Option(
            None,
            "--repo-root",
            help="Override repository root (default: this checkout). Use for tests or alternate trees.",
        ),
        skip_render_localhost_report: bool = typer.Option(False, "--skip-render-localhost-report", help="Skip rendering run-local + published localhost report markdown"),
        skip_validate_localhost_report: bool = typer.Option(False, "--skip-validate-localhost-report", help="Skip localhost report validation after promotion"),
        cleanup: bool = typer.Option(False, "--cleanup", help="Run cleanup_run_artifacts.sh after promotion using this run_id as canonical"),
        timeout_seconds: int = typer.Option(300, "--timeout", help="Promotion timeout in seconds"),
    ) -> None:
        from core.cluster import promote_cluster_run

        result = promote_cluster_run(
            run_id=run_id,
            label=label,
            allow_run_ids=allow_run_id or None,
            publish_report_path=str(publish_report_path) if publish_report_path else None,
            publish_notes_path=str(publish_notes_path) if publish_notes_path else None,
            skip_render_localhost_report=skip_render_localhost_report,
            skip_validate_localhost_report=skip_validate_localhost_report,
            cleanup=cleanup,
            timeout_seconds=timeout_seconds,
            repo_root=str(repo_root) if repo_root else None,
        )
        emit_json(result)
        raise typer.Exit(0)

    @cluster_app.command("watch-promote", help="Attach a detached post-run watcher that promotes a completed cluster run")
    def cluster_watch_promote_cmd(
        ctx: typer.Context,
        run_id: str = typer.Option(..., "--run-id", help="Run id under cluster/runs/<run_id>"),
        pid: int = typer.Option(..., "--pid", help="PID to watch before evaluating + promoting the run"),
        label: str = typer.Option("localhost", "--label", help="Host label for localhost report rendering"),
        allow_run_id: List[str] = typer.Option([], "--allow-run-id", help="Additional run ids to retain during cleanup/validation hygiene checks", show_default=False),
        publish_report_path: Optional[Path] = typer.Option(None, "--publish-report-path", help="Published localhost report path"),
        publish_notes_path: Optional[Path] = typer.Option(None, "--publish-notes-path", help="Published localhost notes path"),
        repo_root: Optional[Path] = typer.Option(
            None,
            "--repo-root",
            help="Override repository root (default: this checkout). Use for tests or alternate trees.",
        ),
        skip_render_localhost_report: bool = typer.Option(False, "--skip-render-localhost-report", help="Skip rendering run-local + published localhost report markdown"),
        skip_validate_localhost_report: bool = typer.Option(False, "--skip-validate-localhost-report", help="Skip localhost report validation after promotion"),
        cleanup: bool = typer.Option(False, "--cleanup", help="Run cleanup_run_artifacts.sh after promotion using this run_id as canonical"),
        poll_interval_seconds: float = typer.Option(30.0, "--poll-interval-seconds", help="How often to check whether the watched PID has exited"),
    ) -> None:
        from core.cluster import watch_cluster_run_for_promotion

        result = watch_cluster_run_for_promotion(
            run_id=run_id,
            pid=pid,
            label=label,
            allow_run_ids=allow_run_id or None,
            publish_report_path=str(publish_report_path) if publish_report_path else None,
            publish_notes_path=str(publish_notes_path) if publish_notes_path else None,
            skip_render_localhost_report=skip_render_localhost_report,
            skip_validate_localhost_report=skip_validate_localhost_report,
            cleanup=cleanup,
            poll_interval_seconds=poll_interval_seconds,
            repo_root=str(repo_root) if repo_root else None,
        )
        emit_json(result)
        raise typer.Exit(0)


# =============================================================================
# Top-level commands (bench, version, help)
# =============================================================================

if typer:

    # Bench integration (core)
    try:
        from core.benchmark import bench_commands
        BENCH_APP = bench_commands.app if getattr(bench_commands, "TYPER_AVAILABLE", False) else None
    except ImportError:
        BENCH_APP = None

    # Tools integration (core)
    try:
        from core.tools import tools_commands
        TOOLS_APP = tools_commands.app if getattr(tools_commands, "TYPER_AVAILABLE", False) else None
    except ImportError:
        TOOLS_APP = None

    # Demos integration (core)
    try:
        from core.demos import demos_commands
        DEMOS_APP = demos_commands.app if getattr(demos_commands, "TYPER_AVAILABLE", False) else None
    except ImportError:
        DEMOS_APP = None

    plugin_apps = load_plugin_apps()

    if TOOLS_APP:
        app.add_typer(TOOLS_APP, name="tools", help="Run non-benchmark tools and utilities")
    else:  # pragma: no cover - tools missing during docs builds

        @app.command("tools", help="Run non-benchmark tools and utilities")
        def tools_stub() -> None:
            typer.echo("Tools CLI unavailable (typer not installed or import failed).")
            raise typer.Exit(code=1)

    if DEMOS_APP:
        app.add_typer(DEMOS_APP, name="demos", help="Run demos and runnable examples (non-benchmark)")
    else:  # pragma: no cover - demos missing during docs builds

        @app.command("demos", help="Run demos and runnable examples (non-benchmark)")
        def demos_stub() -> None:
            typer.echo("Demos CLI unavailable (typer not installed or import failed).")
            raise typer.Exit(code=1)

    if BENCH_APP:
        app.add_typer(BENCH_APP, name="bench", help="Run and profile benchmarks")
    else:  # pragma: no cover - bench missing during docs builds

        @app.command("bench", help="Run and profile benchmarks")
        def bench_stub() -> None:
            typer.echo("Bench CLI unavailable (typer not installed or import failed).")
            raise typer.Exit(code=1)

    # Top-level utility commands
    @app.command("tui", help="Launch interactive benchmark analysis menu")
    def tui_command(
        ctx: typer.Context,
        data_file: Optional[Path] = typer.Option(None, "--data-file", "-d", help="Path to benchmark_test_results.json"),
    ) -> None:
        _run_benchmark_menu(data_file)

    @app.command("dashboard", help="Launch web dashboard")
    def dashboard_command(
        ctx: typer.Context,
        port: int = typer.Option(6970, "--port", "-p", help="Port to run the server on"),
        data: Optional[Path] = typer.Option(None, "--data", "-d", help="Path to benchmark results JSON file"),
        no_browser: bool = typer.Option(False, "--no-browser", help="Do not open browser automatically"),
    ) -> None:
        from dashboard.api.server import serve_dashboard
        serve_dashboard(port=port, data_file=data, open_browser=not no_browser)

    @app.command("mcp", help="Start MCP server for AI chat integration")
    def mcp_command(
        ctx: typer.Context,
        list_tools: bool = typer.Option(False, "--list", help="List available tools"),
        test: Optional[str] = typer.Option(None, "--test", help="Test a specific tool"),
        serve: bool = typer.Option(False, "--serve", help="Start MCP server (stdio)"),
    ) -> None:
        import mcp.mcp_server as server

        old_argv = sys.argv
        new_argv = [old_argv[0]]
        if list_tools:
            new_argv.append("--list")
        if test:
            new_argv.extend(["--test", test])
        if serve:
            new_argv.append("--serve")
        sys.argv = new_argv
        try:
            result = server.main()
            raise typer.Exit(code=0 if result is None else int(result))
        finally:
            sys.argv = old_argv

    @app.command("version", help="Show version information")
    def version_command() -> None:
        typer.echo("aisp - AI System Performance CLI (Typer)")
        typer.echo(f"Repository: {REPO_ROOT}")

    @app.command("help", help="Show help (alias for --help)")
    def help_command(topic: Optional[List[str]] = typer.Argument(None, help="Command or category to show help for")) -> None:
        click_cmd = typer.main.get_command(app)
        args = ["--help"] if not topic else [*topic, "--help"]
        try:
            click_cmd.main(args=args, prog_name="aisp", standalone_mode=False)
        except SystemExit as exc:
            raise typer.Exit(code=exc.code or 0)

    # Load optional plugin Typer apps via entry points (aipe.plugins)
    for plugin in plugin_apps:
        try:
            app.add_typer(plugin.app, name=plugin.name, help=plugin.description or None)
        except Exception:
            continue


# =============================================================================
# Attach sub-apps (10 domains)
# =============================================================================

if typer:
    # Primary domain apps (the 10-domain model)
    app.add_typer(gpu_app, name="gpu", help="GPU: info, topology, power, bandwidth")
    app.add_typer(system_app, name="system", help="System: software, deps, capabilities")
    app.add_typer(profile_app, name="profile", help="Profile: nsys, ncu, flame, HTA")
    app.add_typer(analyze_app, name="analyze", help="Analyze: bottlenecks, pareto, scaling")
    app.add_typer(optimize_app, name="optimize", help="Optimize: recommend, ROI, techniques")
    app.add_typer(distributed_app, name="distributed", help="Distributed: plan, NCCL, FSDP")
    app.add_typer(inference_app, name="inference", help="Inference: vLLM, quantization")
    app.add_typer(benchmark_app, name="benchmark", help="Benchmark: microbench diagnostics, speed tests")
    app.add_typer(ai_app, name="ai", help="AI: ask, explain, suggest")
    app.add_typer(export_app, name="export", help="Export: CSV, PDF, HTML")
    app.add_typer(cluster_app, name="cluster", help="Cluster: common eval presets + raw suite + canonical packaging + field report validation")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> int:
    if typer is None:
        _typer_required()
    rewritten = _rewrite_help_aliases(sys.argv)
    sys.argv = rewritten
    try:
        app()
    except SystemExit as exc:  # Typer raises SystemExit
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
