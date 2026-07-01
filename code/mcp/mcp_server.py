#!/usr/bin/env python3
"""
🚀 MCP Server for AI Systems Performance

Exposes the unified PerformanceEngine as MCP tools for AI chat integration.
Full-featured tool catalog (see --list for the authoritative tool count).

Usage:
    # Start the MCP server
    python -m mcp.mcp_server

    # Or use the aisp command
    aisp mcp serve

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  AI Chat Client                                                         │
    │  ↓                                                                      │
    │  MCP Server (this file) - tool catalog (see --list for count)           │
    │  ↓                                                                      │
    │  PerformanceEngine (core/engine.py) - 10 unified domains                │
    │  ├── gpu         : gpu_info, gpu_topology, gpu_power     │
    │  ├── system      : system_software, system_env, system_* │
    │  ├── profile     : profile_nsys, profile_ncu, profile_*  │
    │  ├── analyze     : analyze_bottlenecks, analyze_pareto, ...   │
    │  ├── optimize    : optimize, recommend, optimize_roi, ...     │
    │  ├── distributed : distributed_plan, distributed_nccl         │
    │  ├── inference   : inference_vllm, inference_deploy, ...      │
    │  ├── benchmark   : run_benchmarks, benchmark_data, ...        │
    │  ├── ai          : ask, explain, ai_troubleshoot         │
    │  └── export      : export_csv, export_pdf, export_html   │
    └─────────────────────────────────────────────────────────────────────────┘

TOOL NAMING: {domain}_{operation}

QUICK START (4 tools):
    triage        - START HERE: status + context in one call
    status        - Quick health check (GPU/software/AI)
    suggest_tools - Get tool recommendations for your task
    job_status    - Poll async job completion

DOMAIN TOOLS (organized by 10-domain model):

    GPU (5 tools):
        gpu_info, gpu_bandwidth, gpu_topology, gpu_power,
        gpu_topology_matrix

    System (10 tools):
        system_software, system_dependencies, system_context,
        system_capabilities, system_parameters, system_container,
        system_cpu_memory, system_env, system_network,
        system_full

    Profile (8 tools):
        profile_nsys, profile_ncu, profile_torch,
        profile_hta, profile_flame, profile_memory,
        profile_kernels, profile_compare

    Analyze (5 tools):
        analyze_bottlenecks, analyze_pareto, analyze_scaling,
        analyze_stacking, analyze_whatif

    Optimize (4 tools):
        optimize, recommend, optimize_roi, optimize_techniques

    Distributed (3 tools):
        distributed_plan, distributed_nccl, cluster_slurm

    Inference (4 tools):
        inference_vllm, inference_quantization,
        inference_deploy, inference_estimate

    Benchmark (20 tools):
        run_benchmarks, benchmark_e2e_sweep, benchmark_e2e_status, benchmark_e2e_watch,
        list_chapters, benchmark_targets, benchmark_report,
        benchmark_export, benchmark_compare_runs, benchmark_triage,
        benchmark_data, benchmark_overview, benchmark_opportunities, benchmark_history,
        benchmark_trends, benchmark_compare,
        benchmark_variants, benchmark_deep_dive_compare,
        benchmark_explore,
        benchmark_llm_patch_loop

    AI (4 tools):
        ask, explain, ai_status, ai_troubleshoot

    Export (3 tools):
        export_csv, export_pdf, export_html

    HuggingFace (1 tool):
        hf (search/trending/download via action param)

BENCHMARKS VS DIAGNOSTICS:
    - `run_benchmarks` runs harness benchmarks (comparative `baseline_*.py` vs
      `optimized_*.py`) and includes the full validity protections.
    - `hw_*` tools run diagnostic microbenchmarks for quick hardware sanity
      checks and intentionally bypass the benchmark harness protections. Do not
      use them to claim baseline-vs-optimized speedups.

HARDWARE MICRO-BENCHMARKS (10 tools):
    hw_speed, hw_roofline, hw_disk, hw_pcie,
    hw_cache, hw_tc, hw_ib, hw_nccl, hw_p2p,
    hw_network

WORKFLOW EXAMPLES:
    New session:     triage → recommend → specific tools
    Slow training:   analyze_bottlenecks → profile_nsys → fix
    Multi-GPU:       gpu_topology → distributed_plan → distributed_nccl
    Inference:       inference_quantization → inference_vllm → inference_deploy
    Benchmarks:      run_benchmarks(async=true) → job_status → benchmark_triage
"""

import asyncio
import json
import os
import shlex
import sys
import subprocess
import traceback
import time
import threading
import uuid
import re
import contextlib
from collections import defaultdict, deque
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, NotRequired

try:
    import fcntl  # POSIX-only; optional for portability.
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None  # type: ignore[assignment]


# Ensure repository root is on sys.path for imports (e.g., analysis.advanced_analysis).
# In some launch modes mcp_server can be invoked from outside the repo checkout,
# so discover the real workspace root defensively instead of trusting __file__ only.
def _discover_code_root() -> Path:
    candidates: List[Path] = []
    env_root = os.environ.get("AISP_CODE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(__file__).resolve().parent.parent)
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    for candidate in candidates:
        try:
            root = candidate.resolve()
        except Exception:
            continue
        if (
            (root / "core").exists()
            and (root / "mcp").exists()
            and (root / "pyproject.toml").exists()
        ):
            return root
        if (root / ".git").exists() and (root / "core").exists():
            return root
    return Path(__file__).resolve().parent.parent


CODE_ROOT = _discover_code_root()

from core.analysis.tool_router import DEFAULT_SUGGEST_RULES, suggest_tools_auto
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.jobs import JobStore
from core.utils.dotenv import load_repo_dotenv


def _load_mcp_env() -> None:
    """Load .env and .env.local so MCP has access to API keys."""
    load_repo_dotenv(CODE_ROOT)


_load_mcp_env()


# MCP Protocol Types
@dataclass
class ToolDefinition:
    """MCP Tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolResult:
    """MCP Tool result."""

    content: List[Dict[str, Any]]
    is_error: bool = False


# =============================================================================
# RESULT TYPE DEFINITIONS (TypedDict for type safety)
# =============================================================================


class ToolResultBase(TypedDict, total=False):
    """Base shape for all tool results."""

    success: bool
    error: NotRequired[str]
    context: NotRequired[Dict[str, Any]]
    context_error: NotRequired[str]


class BenchmarkExportResult(ToolResultBase):
    """Result shape for benchmark export operations."""

    output: str
    format: str
    benchmarks_written: int


class JobStatusResult(ToolResultBase):
    """Result shape for job status queries."""

    job_id: str
    status: str
    tool: NotRequired[str]
    submitted_at: NotRequired[float]
    finished_at: NotRequired[float]
    duration_ms: NotRequired[int]
    result: NotRequired[Any]
    note: NotRequired[str]


class SuggestionResult(ToolResultBase):
    """Result shape for tool suggestions."""

    suggestions: List[Dict[str, Any]]
    count: int


class ContextResult(ToolResultBase):
    """Result shape for context provider tools."""

    context: Dict[str, Any]


# =============================================================================
# CONTEXT PARAMS SCHEMA (DRY helper for tool definitions)
# =============================================================================

_CONTEXT_PARAMS_SCHEMA: Dict[str, Any] = {
    "include_context": {
        "type": "boolean",
        "description": "Include full system context in the response",
        "default": False,
    },
    "context_level": {
        "type": "string",
        "description": "Context level: summary or full",
        "enum": ["summary", "full"],
        "default": "summary",
    },
}

# =============================================================================
# MINIMAL PARAM NORMALIZATION - Only for commonly confused values
# MCP Convention: LLM reads schema enums; we only fix the most common mistakes
# =============================================================================

# Only the most problematic aliases that cause real errors
_COMMON_ALIASES: Dict[str, Dict[str, str]] = {
    "profile": {"full": "deep_dive", "deep": "deep_dive"},
    "preset": {"deep": "full", "complete": "full"},
    "format": {"md": "markdown"},
    "collective": {"allreduce": "all_reduce", "allgather": "all_gather"},
}


def normalize_param(param_name: str, value: Any, default: Any = None) -> Any:
    """Normalize common parameter aliases. Returns original if no alias found."""
    if value is None:
        return default
    str_value = str(value).lower().strip()
    aliases = _COMMON_ALIASES.get(param_name, {})
    return aliases.get(str_value, str_value if str_value else default)


def with_context_params(props: Dict[str, Any]) -> Dict[str, Any]:
    """Merge tool-specific properties with standard context params."""
    return {**props, **_CONTEXT_PARAMS_SCHEMA}


def extract_context_opts(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Extract include_context and context_level from params."""
    return (bool(params.get("include_context", False)), params.get("context_level", "summary"))


# =============================================================================
# DESCRIPTION ENRICHMENT HELPERS
# =============================================================================

_OUTPUT_ENVELOPE_SUMMARY = (
    "JSON envelope with tool/status/timestamp/duration_ms, sanitized arguments + details, "
    "result + preview + metadata, context_summary, guidance.next_steps; content is emitted as a "
    "text (JSON string) entry."
)

# Explicit overrides for tools that have notable runtime/side effects.
_EXPECTATION_OVERRIDES: Dict[str, str] = {
    "run_benchmarks": "Runs the bench CLI; can take minutes and writes artifacts/logs under artifacts/runs/<run_id>/ by default. Run IDs are self-describing (timestamp + run kind + targets). Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. Also runs triage + HTML report generation unless auto_analyze/auto_report are disabled. Use artifacts_dir to control output location. For full baseline-vs-optimized profiling + diffs, use profile='deep_dive' or benchmark_deep_dive_compare.",
    "benchmark_e2e_sweep": "Runs the tier1 release-confidence suite, optional discovered full sweep, cluster common-eval, and optional fabric eval in one orchestrated flow. Writes a top-level package under artifacts/e2e_runs/<run_id>/, auto-arms a detached watcher by default, and uses async job tickets when requested.",
    "benchmark_e2e_status": "Reads the e2e run package and returns one normalized status snapshot: liveness, current child progress, recent events, watcher metadata, ledgers, and resume/watch commands.",
    "benchmark_e2e_watch": "Launches the detached e2e watcher that auto-resumes stale or aborted runs using the stored contract and writes watcher_status.json into the run package.",
    "benchmark_deep_dive_compare": "Runs bench with profile='deep_dive' (slow) and then compares baseline vs optimized profiles (nsys+ncu). Emits side-by-side JSON + narrative and writes a self-describing run directory under output_dir (default: artifacts/runs).",
    "benchmark_report": "Generates a report from existing benchmark JSON; writes PDF/HTML to the chosen output.",
    "benchmark_export": "Exports existing benchmark JSON to csv/markdown/json; writes to the chosen output file.",
    "benchmark_compare_runs": "Diffs two benchmark JSON files; CPU-bound and quick, writes only if an output is specified.",
    "profile_nsys": "Calls Nsight Systems; requires nsys installed and writes .nsys-rep under artifacts/runs/<run_id>/profiles/tools/<tool>/<label>/ by default. Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. Slow/interactive; run status or triage first. Default preset is light; set preset=full for richer traces.",
    "profile_ncu": "Calls Nsight Compute; requires ncu installed and writes .ncu-rep under artifacts/runs/<run_id>/profiles/tools/<tool>/<label>/ by default. Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. Slow/interactive; run status or triage first. Defaults to memory_bound metric set; opt into heavier modes explicitly.",
    "profile_compare": "Generates flame graph comparison + side-by-side Nsight Systems/Compute JSON report; parses NSYS reports and may traverse multiple files; allow extra runtime.",
    "hw_speed": "Runs GPU/host micro-benchmarks; stresses hardware briefly. Run status first; supports precheck_only/dry_run/timeout_seconds.",
    "hw_roofline": "Runs roofline micro-benchmark; stresses memory subsystem briefly. Run status first; supports precheck_only/dry_run/timeout_seconds.",
    "hw_disk": "Runs disk I/O hardware benchmark; writes temporary files to tmp_dir. Supports precheck_only/dry_run/timeout_seconds.",
    "hw_pcie": "Runs PCIe hardware benchmark; exercises host↔GPU transfers. Run status first; supports precheck_only/dry_run/timeout_seconds.",
    "hw_cache": "Runs memory hierarchy hardware benchmark; exercises GPU cache. Run status first; supports precheck_only/dry_run/timeout_seconds.",
    "hw_tc": "Runs tensor core hardware benchmark; exercises GPU math units. Run status first; supports precheck_only/dry_run/timeout_seconds.",
    "tools_compare_precision": "Runs aisp tools compare-precision; may run evaluation workloads and write reports depending on args.",
    "tools_dump_hw": "Runs aisp tools dump-hw; can be slow unless --fast is set.",
    "tools_probe_hw": "Runs aisp tools probe-hw; probes hardware capabilities and writes artifacts/hardware_capabilities.json.",
}

_SELECTION_HINTS: Dict[str, str] = {
    "triage": (
        "First call for combined status + context; use status for a faster health-only check; "
        "use suggest_tools when you only need tool recommendations."
    ),
    "status": ("Health-only snapshot; use triage for context or system_full for full inventory."),
    "suggest_tools": ("Intent-to-tool mapping; use ask for answers or triage for system snapshot."),
    "run_benchmarks": (
        "Run benchmarks; use benchmark_targets to discover targets and "
        "benchmark_deep_dive_compare for one-shot run+profile+diff."
    ),
    "optimize": (
        "Shortcut for benchmark_variants using a benchmark file path or target; "
        "runs quick LLM variants by default."
    ),
    "benchmark_deep_dive_compare": (
        "One-shot run+profile+diff; use run_benchmarks if you only want results without deep profiling."
    ),
    "benchmark_triage": ("Analyze a single run; use benchmark_compare_runs to compare two runs."),
    "benchmark_opportunities": (
        "Rank what to optimize next; use benchmark_triage for pass/fail triage or benchmark_compare_runs for before/after diffs."
    ),
    "benchmark_compare_runs": (
        "Compare baseline vs candidate runs; use benchmark_triage for single-run analysis."
    ),
    "benchmark_report": (
        "Human-readable PDF/HTML report; use benchmark_export for CSV/markdown/json data exports."
    ),
    "benchmark_export": (
        "Raw data export (CSV/markdown/json); use benchmark_report for PDF/HTML reports."
    ),
    "profile_nsys": (
        "Timeline/API tracing; use profile_ncu for kernel metrics or profile_torch for PyTorch ops."
    ),
    "profile_ncu": (
        "Kernel metrics; use profile_nsys for timeline or profile_torch for PyTorch ops."
    ),
    "profile_torch": (
        "PyTorch operator breakdown; use profile_nsys for timeline or profile_ncu for kernel metrics."
    ),
    "profile_compare": (
        "Narrative + flamegraph comparison; use compare_nsys/ncu for raw metric diffs."
    ),
    "compare_nsys": (
        "Timeline comparison; use compare_ncu for kernel metrics and profile_compare for narrative+flamegraph."
    ),
    "compare_ncu": (
        "Kernel metric comparison; use compare_nsys for timeline and profile_compare for narrative+flamegraph."
    ),
    "ask": ("Conceptual performance Q&A; use explain to interpret existing tool outputs."),
    "explain": ("Interpret tool outputs/results; use ask for general performance questions."),
    "export_csv": (
        "Inline CSV payload; use benchmark_export for file output and benchmark_report for PDF/HTML."
    ),
    "export_html": (
        "Inline HTML payload; use benchmark_report for file output or export_pdf for PDF."
    ),
    "export_pdf": (
        "Inline PDF payload; use benchmark_report for file output or export_html for HTML."
    ),
}


def _property_implies_output(prop: str) -> bool:
    """Detect property names that imply writing to disk."""
    return (
        prop == "output"
        or prop.startswith("output_")
        or prop.endswith("_output")
        or prop == "output_dir"
        or prop == "path"
        or prop.startswith("path_")
        or prop.endswith("_path")
        or prop == "file"
        or prop.startswith("file_")
        or prop.endswith("_file")
        or prop == "dir"
        or prop.startswith("dir_")
        or prop.endswith("_dir")
        or prop == "report"
        or prop.startswith("report_")
        or prop.endswith("_report")
    )


def _repr_default(value: Any) -> str:
    """Stringify a default value for inline descriptions."""
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _format_inputs_from_schema(schema: Optional[Dict[str, Any]]) -> str:
    """Create a compact inline summary of inputs from JSON schema."""
    schema = schema or {}
    props = schema.get("properties") or {}
    if not props:
        return "none"
    required = set(schema.get("required") or [])
    parts: List[str] = []
    for name, meta in props.items():
        param_type = meta.get("type") or "any"
        if isinstance(param_type, list):
            param_type = "/".join(map(str, param_type))
        details = [param_type, "required" if name in required else "optional"]
        if "default" in meta:
            details.append(f"default={_repr_default(meta.get('default'))}")
        desc = meta.get("description")
        detail_str = ", ".join(details)
        text = f"{name} ({detail_str})"
        if desc:
            text += f" - {desc}"
        parts.append(text)
    return "; ".join(parts)


def _expectations_from_name_and_schema(name: str, schema: Optional[Dict[str, Any]]) -> str:
    """Build expectation hints (runtime, side effects, context toggles)."""
    name_key = name.lower()
    props: Dict[str, Any] = schema.get("properties") if schema else {}
    notes: List[str] = []

    override = _EXPECTATION_OVERRIDES.get(name)
    if override:
        notes.append(override)
    elif "benchmark" in name_key:
        notes.append("Runs benchmarks; can be long-running and GPU-intensive.")
    elif "profile" in name_key:
        notes.append("Runs profiling; may be slower and produce trace files.")
    elif name_key.startswith("hw_") or name_key.startswith("test_"):
        notes.append("Runs hardware benchmarks; may briefly stress hardware.")
    else:
        notes.append("Typically fast, read-only snapshot.")

    if props and any(_property_implies_output(prop) for prop in props):
        notes.append(
            "May write files when output/path parameters are set (creates directories when needed)."
        )
    if props and "include_context" in props:
        notes.append("Use include_context/context_level to attach system snapshot to the response.")
    return " ".join(notes)


def _enrich_description(name: str, description: str, schema: Optional[Dict[str, Any]]) -> str:
    """Combine base description with derived inputs/outputs/expectations."""
    inputs_text = _format_inputs_from_schema(schema)
    expectations = _expectations_from_name_and_schema(name, schema)
    selection_hint = _SELECTION_HINTS.get(name)
    parts = [description.strip()]
    if selection_hint:
        parts.append(f"Selection: {selection_hint}")
    parts.extend(
        [
            f"Inputs: {inputs_text}.",
            f"Outputs: {_OUTPUT_ENVELOPE_SUMMARY}",
            f"Expectations: {expectations}",
        ]
    )
    return " ".join(parts)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOLS: Dict[str, ToolDefinition] = {}
HANDLERS: Dict[str, callable] = {}


def register_tool(name: str, description: str, schema: Dict[str, Any] = None):
    """Decorator to register an MCP tool."""

    def decorator(func):
        tool_schema = schema or {"type": "object", "properties": {}}
        enriched_description = _enrich_description(name, description, tool_schema)
        TOOLS[name] = ToolDefinition(
            name=name, description=enriched_description, input_schema=tool_schema
        )

        def wrapper(*args, **kwargs):
            """Lenient entry point that tolerates missing params and unexpected kwargs."""
            # Allow callers to pass params dict directly or expanded kwargs.
            if args and not kwargs:
                call_params = args[0] if isinstance(args[0], dict) else {}
            elif "params" in kwargs and len(kwargs) == 1:
                call_params = kwargs.get("params") or {}
            else:
                # Treat any expanded kwargs as params dict for convenience.
                call_params = kwargs or {}

            try:
                return func(call_params)
            except Exception as exc:  # pragma: no cover - defensive
                # Return normalized error with success=False for consistency
                return {"error": str(exc), "success": False}

        HANDLERS[name] = wrapper
        return func

    return decorator


_CONTEXT_CACHE: Dict[str, Any] = {"summary": None, "full": None}
_CONTEXT_TS: Dict[str, float] = {"summary": 0.0, "full": 0.0}
_CONTEXT_TTL_SECONDS = 60.0
_PREVIEW_MAX_LENGTH = int(os.environ.get("AISP_MCP_PREVIEW_LIMIT", "200000") or "200000")
_PREVIEW_MAX_ITEMS = int(os.environ.get("AISP_MCP_PREVIEW_ITEMS", "2000") or "2000")


def _subprocess_env() -> Dict[str, str]:
    """Build an environment with repo root on PYTHONPATH for child processes."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{CODE_ROOT}:{existing}" if existing else str(CODE_ROOT)
    return env


def _ensure_dir(path: Path) -> None:
    """Create parent directory for a file path (no-op if already exists)."""
    try:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # best-effort; let downstream errors surface
        return None


def _build_context(level: str):
    """Build context at the requested level."""
    from core.perf_core import get_core
    from core.engine import get_engine

    engine = get_engine()

    if level == "summary":
        # Keep summary light: GPU + software snapshot.
        return {
            "gpu": engine.gpu.info(),
            "software": engine.system.software(),
            "dependencies": engine.system.dependencies(),
        }

    # Full context can be heavier; delegate to engine.
    return engine.system.context()


def get_cached_context(level: str) -> Any:
    """Fetch cached context or rebuild if stale."""
    now = time.time()
    level = "full" if level == "full" else "summary"
    if (
        _CONTEXT_CACHE.get(level) is None
        or (now - _CONTEXT_TS.get(level, 0)) > _CONTEXT_TTL_SECONDS
    ):
        _CONTEXT_CACHE[level] = _build_context(level)
        _CONTEXT_TS[level] = now
    return _CONTEXT_CACHE[level]


def attach_context_if_requested(result: Any, include_context: bool, context_level: str = "summary"):
    """Optionally attach system context to the tool result."""
    if not include_context:
        return result

    try:
        context = get_cached_context(context_level)
    except Exception as e:
        # Preserve the original result and surface the context error
        if isinstance(result, dict):
            return {**result, "context_error": str(e)}
        return {"result": result, "context_error": str(e)}

    if isinstance(result, dict):
        # Avoid clobbering any existing 'context' key
        if "context" in result:
            return {**result, "_context": context}
        return {**result, "context": context}

    return {"result": result, "context": context}


def make_error(
    msg: str, include_context: bool = False, context_level: str = "summary", **extra: Any
) -> Dict[str, Any]:
    """Build standardized error response with optional context attachment.

    Args:
        msg: The error message.
        include_context: Whether to attach system context.
        context_level: Level of context to attach ("summary" or "full").
        **extra: Additional fields to include in the error response.

    Returns:
        A normalized error dict with success=False and optional context.
    """
    error_type = extra.pop("error_type", None)
    if error_type is None:
        lowered = msg.lower()
        if "required" in lowered or "must be" in lowered or "invalid" in lowered:
            error_type = "value_error"
        elif "not found" in lowered:
            error_type = "file_not_found"
        else:
            error_type = "runtime_error"
    result: Dict[str, Any] = {"error": msg, "error_type": error_type, "success": False, **extra}
    return attach_context_if_requested(result, include_context, context_level)


def ensure_result(
    result: Dict[str, Any], include_context: bool, context_level: str
) -> Dict[str, Any]:
    """Normalize result and optionally attach context - single exit point for handlers.

    Args:
        result: The raw result dict from a tool handler.
        include_context: Whether to attach system context.
        context_level: Level of context to attach.

    Returns:
        Normalized result with success flag and optional context.
    """
    normalized = dict(result)
    if "success" not in normalized:
        normalized["success"] = not bool(normalized.get("error"))
    if normalized.get("error") and "error_type" not in normalized:
        message = str(normalized["error"]).lower()
        if "required" in message or "must be" in message or "invalid" in message:
            normalized["error_type"] = "value_error"
        elif "not found" in message:
            normalized["error_type"] = "file_not_found"
        else:
            normalized["error_type"] = "runtime_error"
    return attach_context_if_requested(normalized, include_context, context_level)


_BENCH_CLI_TIMEOUT = 900  # generous default; keeps CLI invocations from hanging forever


def _run_cli(args: List[str], timeout: Optional[int] = _BENCH_CLI_TIMEOUT) -> Dict[str, Any]:
    """Invoke aisp CLI directly (without bench prefix) and return stdout/stderr/exit code."""
    cmd = [sys.executable, "-m", "cli.aisp", *args]
    env = _subprocess_env()
    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout is None or timeout <= 0 else timeout,
            env=env,
            cwd=str(CODE_ROOT),
        )
        result = {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout_seconds": timeout if timeout and timeout > 0 else None,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
        }
        return result
    except subprocess.TimeoutExpired as te:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": te.stdout.decode() if te.stdout else "",
            "stderr": te.stderr.decode() if te.stderr else "",
            "timeout_seconds": timeout,
            "timeout_hit": True,
            "duration_seconds": round(time.time() - started_at, 2),
            "error": f"Timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "timeout_seconds": timeout if timeout and timeout > 0 else None,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
            "error": str(exc),
        }


def _run_tools_cli(
    tool: str,
    tool_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = _BENCH_CLI_TIMEOUT,
) -> Dict[str, Any]:
    """Invoke `aisp tools <tool> -- <args...>` and return stdout/stderr/exit code."""
    args: List[str] = ["tools", tool]
    if tool_args:
        args.append("--")
        args.extend(tool_args)
    result = _run_cli(args, timeout=timeout_seconds)
    if isinstance(result, dict):
        returncode = int(result.get("returncode", 0) or 0)
        if returncode != 0 and not result.get("error"):
            result["error"] = (
                result.get("stderr")
                or result.get("stdout")
                or f"aisp tools {tool} failed with code {returncode}"
            )
    return result


def _run_bench_cli(args: List[str], timeout: Optional[int] = _BENCH_CLI_TIMEOUT) -> Dict[str, Any]:
    """Invoke bench CLI and return stdout/stderr/exit code."""
    cmd = [sys.executable, "-m", "core.benchmark.bench_commands", *args]
    env = _subprocess_env()
    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout is None or timeout <= 0 else timeout,
            env=env,
            cwd=str(CODE_ROOT),
        )
        result = {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout_seconds": timeout if timeout and timeout > 0 else None,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
        }
        if proc.returncode != 0 and not result.get("error"):
            result["error"] = (
                proc.stderr or proc.stdout or f"bench CLI failed with code {proc.returncode}"
            )
        return result
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(cmd),
            "error": f"bench CLI timed out after {timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout_seconds": timeout,
            "timeout_hit": True,
            "duration_seconds": round(time.time() - started_at, 2),
        }


_BENCH_RESULTS_MARKER = "JSON results saved to:"


def _is_test_mode() -> bool:
    return os.getenv("AISP_TEST_MODE") == "1" or "PYTEST_CURRENT_TEST" in os.environ


def _extract_bench_results_json(result: Dict[str, Any]) -> Optional[Path]:
    """Best-effort extract the benchmark_test_results.json path from bench CLI output."""
    if not isinstance(result, dict):
        return None
    for stream_key in ("stdout", "stderr"):
        text = result.get(stream_key) or ""
        if not isinstance(text, str):
            continue
        if _BENCH_RESULTS_MARKER not in text:
            continue

        # Rich logging can wrap long paths across lines; capture the marker tail and
        # stitch the following non-empty lines until we reconstruct the .json path.
        tail = text.rsplit(_BENCH_RESULTS_MARKER, 1)[1]
        fragments: List[str] = []
        candidate: Optional[str] = None
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fragments.append(stripped)
            stitched = "".join(fragments).strip().strip("\"'")
            if "benchmark_test_results" not in stitched:
                # Keep collecting until we reach the filename portion.
                continue
            if stitched.endswith(".json"):
                candidate = stitched
                break
            if ".json" in stitched:
                candidate = stitched.split(".json", 1)[0] + ".json"
                break
            if len(stitched) > 4096:
                break

        if not candidate:
            continue

        path = Path(candidate)
        if not path.is_absolute():
            path = (CODE_ROOT / path).resolve()
        return path
    return None


def _resolve_artifact_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (CODE_ROOT / path).resolve()
    return path


def _resolve_benchmark_target_from_path(path_value: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a benchmark file path to a unique chapter:example target."""
    if not path_value:
        return None, "path is required"
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (CODE_ROOT / path).resolve()
    if not path.exists():
        return None, f"Benchmark path not found: {path}"
    if path.is_dir():
        return None, f"Path must be a benchmark file, not a directory: {path}"

    try:
        from core.discovery import (
            discover_all_chapters,
            discover_benchmarks,
            chapter_slug,
            get_bench_roots,
        )
    except Exception as exc:
        return None, f"Benchmark discovery import failed: {exc}"

    if path.suffix == ".cu":
        return None, (
            "CUDA benchmarks run via Python wrappers only. "
            "Provide the baseline_/optimized_ .py wrapper path instead of the .cu file."
        )

    roots = get_bench_roots(repo_root=CODE_ROOT)
    bench_root = roots[0] if roots else CODE_ROOT
    candidates: List[str] = []

    for chapter_dir in discover_all_chapters(CODE_ROOT, bench_roots=roots):
        try:
            python_pairs = discover_benchmarks(chapter_dir)
        except Exception:
            python_pairs = []
        for baseline, optimized_list, example_name in python_pairs:
            if path == baseline.resolve() or any(path == opt.resolve() for opt in optimized_list):
                slug = chapter_slug(chapter_dir, CODE_ROOT, bench_root=bench_root)
                candidates.append(f"{slug}:{example_name}")

    unique = sorted(set(candidates))
    if not unique:
        return None, f"No benchmark target found for path: {path}"
    if len(unique) > 1:
        return None, f"Benchmark path maps to multiple targets: {', '.join(unique)}"
    return unique[0], None


def _file_entry(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        entry["exists"] = True
        entry["size_bytes"] = stat.st_size
        entry["size_mb"] = round(stat.st_size / (1024 * 1024), 3)
    except FileNotFoundError:
        entry["exists"] = False
    return entry


def _collect_dir_files(dir_path: Path) -> List[Dict[str, Any]]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.iterdir() if p.is_file()]
    return [_file_entry(p) for p in sorted(files)]


def _collect_dir_files_recursive(dir_path: Path) -> List[Dict[str, Any]]:
    if not dir_path.exists():
        return []
    files = [p for p in dir_path.rglob("*") if p.is_file()]
    return [_file_entry(p) for p in sorted(files)]


def _profile_prefix(path: Path) -> str:
    name = path.name
    for suffix in (".nsys-rep", ".ncu-rep"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _collect_profile_files(paths: List[Path]) -> List[Path]:
    files: set[Path] = set()
    for path in paths:
        prefix = _profile_prefix(path)
        for candidate in path.parent.glob(f"{prefix}*"):
            if candidate.is_file():
                files.add(candidate)
    return sorted(files)


def _summarize_bench_artifacts(results_json: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    try:
        payload = json.loads(results_json.read_text())
    except Exception as exc:
        summary["artifact_summary_error"] = f"Failed to read results JSON: {exc}"
        return summary

    results = payload.get("results")
    if not isinstance(results, list):
        summary["artifact_summary_error"] = "Unexpected benchmark results schema"
        return summary

    run_dir = results_json.parent.parent
    artifact_links = {
        "run_dir": str(run_dir),
        "manifest": _file_entry(run_dir / "manifest.json"),
        "results": _collect_dir_files(run_dir / "results"),
        "reports": _collect_dir_files(run_dir / "reports"),
        "logs": _collect_dir_files(run_dir / "logs"),
        "progress": _collect_dir_files(run_dir / "progress"),
        "profiles": _collect_dir_files_recursive(run_dir / "profiles"),
    }

    profiling_summary: List[Dict[str, Any]] = []
    profile_files: set[Path] = set()

    for result in results:
        chapter = result.get("chapter")
        benchmarks = result.get("benchmarks", [])
        if not isinstance(benchmarks, list):
            continue
        for bench in benchmarks:
            example = bench.get("example") or bench.get("name") or bench.get("benchmark")
            if chapter and example:
                benchmark_id = f"{chapter}:{example}"
            else:
                benchmark_id = example or chapter or "unknown"

            baseline_nsys = _resolve_artifact_path(bench.get("baseline_nsys_rep"))
            baseline_ncu = _resolve_artifact_path(bench.get("baseline_ncu_rep"))
            baseline_paths = [p for p in (baseline_nsys, baseline_ncu) if p]
            baseline_files = _collect_profile_files(baseline_paths) if baseline_paths else []
            profile_files.update(baseline_files)

            baseline_entry = {
                "nsys_rep": str(baseline_nsys) if baseline_nsys else None,
                "ncu_rep": str(baseline_ncu) if baseline_ncu else None,
                "metrics": bench.get("baseline_profiler_metrics", {}),
                "files": [_file_entry(p) for p in baseline_files],
            }

            optimizations: List[Dict[str, Any]] = []
            for opt in bench.get("optimizations", []) or []:
                opt_nsys = _resolve_artifact_path(opt.get("optimized_nsys_rep"))
                opt_ncu = _resolve_artifact_path(opt.get("optimized_ncu_rep"))
                opt_paths = [p for p in (opt_nsys, opt_ncu) if p]
                opt_files = _collect_profile_files(opt_paths) if opt_paths else []
                profile_files.update(opt_files)
                optimizations.append(
                    {
                        "label": opt.get("file")
                        or opt.get("technique")
                        or opt.get("name")
                        or "optimized",
                        "status": opt.get("status"),
                        "nsys_rep": str(opt_nsys) if opt_nsys else None,
                        "ncu_rep": str(opt_ncu) if opt_ncu else None,
                        "metrics": opt.get("optimized_profiler_metrics", {}),
                        "files": [_file_entry(p) for p in opt_files],
                    }
                )

            profiling_summary.append(
                {
                    "benchmark": benchmark_id,
                    "baseline": baseline_entry,
                    "optimizations": optimizations,
                }
            )

    if profile_files:
        artifact_links["profiling_files"] = [_file_entry(p) for p in sorted(profile_files)]

    summary["artifact_links"] = artifact_links
    summary["profiling_summary"] = profiling_summary
    return summary


def _attach_bench_artifact_paths(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach {results_json, run_dir} to a bench CLI result when discoverable."""
    results_json = _extract_bench_results_json(result)
    if not results_json and isinstance(result, dict):
        run_dir = result.get("run_dir")
        if run_dir:
            candidate = Path(str(run_dir)) / "results" / "benchmark_test_results.json"
            if candidate.exists():
                results_json = candidate
    if not results_json:
        return result

    run_dir: Optional[Path] = None
    try:
        # Standard artifact layout: <artifacts_dir>/<run_id>/results/benchmark_test_results.json
        run_dir = results_json.parent.parent
    except Exception:
        run_dir = None

    enriched = dict(result)
    enriched["results_json"] = str(results_json)
    if run_dir:
        enriched["run_dir"] = str(run_dir)
        enriched["run_id"] = run_dir.name
    summary = _summarize_bench_artifacts(results_json)
    if summary:
        enriched.update(summary)
    return enriched


def _default_benchmark_report_path(result: Dict[str, Any], fmt: str) -> Optional[Path]:
    run_dir = result.get("run_dir")
    if run_dir:
        return Path(str(run_dir)) / "reports" / f"benchmark_report.{fmt}"
    results_json = result.get("results_json")
    if results_json:
        try:
            return Path(str(results_json)).parent.parent / "reports" / f"benchmark_report.{fmt}"
        except Exception:
            return None
    return None


def _parse_event_message(
    message: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Parse benchmark event lines encoded as `EVENT <name> <json-payload>`."""
    if not isinstance(message, str):
        return None, None, "event message is not a string"
    if not message.startswith("EVENT "):
        return None, None, "message is not an EVENT payload"
    event_body = message[len("EVENT ") :]
    if " " not in event_body:
        return None, None, "event payload is missing a JSON body"
    event_name, payload_text = event_body.split(" ", 1)
    try:
        payload = json.loads(payload_text)
    except Exception as exc:
        return event_name, None, f"failed to parse EVENT payload JSON: {exc}"
    if not isinstance(payload, dict):
        return event_name, None, f"expected EVENT payload object, got {type(payload).__name__}"
    return event_name, payload, None


def _extract_speedup_attribution(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Derive non-LLM speedup attribution from benchmark EVENT logs."""
    run_dir = result.get("run_dir")
    if not run_dir and result.get("results_json"):
        try:
            run_dir = str(Path(str(result["results_json"])).parent.parent)
        except Exception:
            run_dir = None
    if not run_dir:
        return None

    log_path = Path(str(run_dir)) / "logs" / "benchmark.log"
    if not log_path.exists():
        return None

    metric_fields = (
        "kernel_time_ms",
        "sm_throughput_percent",
        "dram_throughput_percent",
        "l2_throughput_percent",
        "occupancy",
    )
    baseline_ncu: Dict[Tuple[str, str], Dict[str, float]] = {}
    optimized_ncu: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    optimizations: List[Dict[str, Any]] = []
    warning_samples: List[str] = []
    warning_counts = {
        "malformed_json_lines": 0,
        "malformed_event_payloads": 0,
    }

    for line_no, raw_line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        try:
            line_obj = json.loads(raw_line)
        except Exception as exc:
            warning_counts["malformed_json_lines"] += 1
            if len(warning_samples) < 5:
                warning_samples.append(f"line {line_no}: failed to parse benchmark log JSON: {exc}")
            continue
        event_name, payload, parse_warning = _parse_event_message(line_obj.get("message"))
        if parse_warning and event_name:
            warning_counts["malformed_event_payloads"] += 1
            if len(warning_samples) < 5:
                warning_samples.append(f"line {line_no}: {parse_warning}")
        if not event_name or not payload:
            continue

        chapter = str(payload.get("chapter") or "").strip()
        example = str(payload.get("example") or "").strip()
        if not chapter or not example:
            continue

        if event_name == "profiler_end":
            profiler = str(payload.get("profiler") or "").strip().lower()
            status = str(payload.get("status") or "").strip().lower()
            if profiler != "ncu" or status != "succeeded":
                continue
            metrics_obj = payload.get("metrics")
            if not isinstance(metrics_obj, dict):
                continue
            metrics: Dict[str, float] = {}
            for field_name in metric_fields:
                value = metrics_obj.get(field_name)
                if isinstance(value, (int, float)):
                    metrics[field_name] = float(value)
            if not metrics:
                continue
            variant = str(payload.get("variant") or "").strip().lower()
            if variant == "baseline":
                baseline_ncu[(chapter, example)] = metrics
            elif variant == "optimized":
                technique = str(payload.get("technique") or "").strip()
                if technique:
                    optimized_ncu[(chapter, example, technique)] = metrics
            continue

        if event_name == "optimization_result":
            optimizations.append(payload)

    if not optimizations and not any(warning_counts.values()):
        return None

    items: List[Dict[str, Any]] = []
    for payload in optimizations:
        chapter = str(payload.get("chapter") or "").strip()
        example = str(payload.get("example") or "").strip()
        technique = str(payload.get("technique") or "").strip()
        if not chapter or not example:
            continue

        baseline = baseline_ncu.get((chapter, example))
        optimized = optimized_ncu.get((chapter, example, technique)) if technique else None
        delta: Dict[str, Dict[str, float]] = {}
        if baseline and optimized:
            for field_name in metric_fields:
                base_val = baseline.get(field_name)
                opt_val = optimized.get(field_name)
                if base_val is None or opt_val is None:
                    continue
                delta_entry: Dict[str, float] = {
                    "baseline": base_val,
                    "optimized": opt_val,
                    "absolute_delta": opt_val - base_val,
                }
                if base_val != 0.0:
                    delta_entry["percent_delta"] = ((opt_val - base_val) / base_val) * 100.0
                delta[field_name] = delta_entry

        notes: List[str] = []
        if baseline and optimized:
            confidence = "high"
            if not delta:
                notes.append(
                    "Baseline and optimized NCU runs succeeded, but no overlapping attribution metrics were present."
                )
        elif baseline and not optimized:
            confidence = "partial"
            notes.append(
                "Baseline NCU metrics are available, but optimized NCU metrics were not captured."
            )
        elif optimized and not baseline:
            confidence = "partial"
            notes.append(
                "Optimized NCU metrics are available, but baseline NCU metrics were not captured."
            )
        else:
            confidence = "none"
            notes.append(
                "Neither baseline nor optimized NCU metrics were captured for this optimization."
            )

        items.append(
            {
                "target": f"{chapter}:{example}",
                "technique": technique or payload.get("optimization_file"),
                "status": payload.get("status"),
                "speedup": payload.get("speedup"),
                "time_ms": payload.get("time_ms"),
                "ncu": {
                    "baseline": baseline,
                    "optimized": optimized,
                    "delta": delta if delta else None,
                },
                "confidence": confidence,
                "notes": notes,
            }
        )

    if not items and not any(warning_counts.values()):
        return None

    response = {
        "source": str(log_path),
        "items": items,
    }
    if any(warning_counts.values()):
        response["warnings"] = warning_samples
        response["warning_counts"] = warning_counts
    return response


def _maybe_run_post_benchmark_steps(
    result: Dict[str, Any],
    *,
    auto_analyze: bool,
    auto_report: bool,
    report_format: str,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    if int(result.get("returncode", 0) or 0) != 0 or result.get("error"):
        return result
    results_json = result.get("results_json")
    if not results_json:
        return result

    include_context = False
    context_level = "summary"
    updated = dict(result)
    if auto_analyze:
        updated["triage"] = tool_benchmark_triage(
            {
                "data_file": str(results_json),
                "include_context": include_context,
                "context_level": context_level,
            }
        )
    if auto_report:
        fmt = normalize_param("format", report_format, "html")
        if fmt not in {"pdf", "html"}:
            fmt = "html"
        report_path = _default_benchmark_report_path(updated, fmt)
        report_output = str(report_path) if report_path else f"benchmark_report.{fmt}"
        updated["report"] = tool_benchmark_report(
            {
                "data_file": str(results_json),
                "output": report_output,
                "format": fmt,
                "include_context": include_context,
                "context_level": context_level,
            }
        )
    attribution = _extract_speedup_attribution(updated)
    if attribution:
        updated["speedup_attribution"] = attribution
    return updated


def _default_run_id(kind: str, label: str, base_dir: Optional[Path]) -> str:
    from core.benchmark.artifact_manager import build_run_id

    return build_run_id(kind, label, base_dir=base_dir)


def _resolve_run_dir(artifacts_dir: Optional[str], run_id: str) -> Path:
    base_dir = Path(artifacts_dir) if artifacts_dir else CODE_ROOT / "artifacts" / "runs"
    if not base_dir.is_absolute():
        base_dir = (CODE_ROOT / base_dir).resolve()
    return base_dir / run_id


def _progress_path_for_run(run_dir: Optional[Path], run_id: Optional[str]) -> Optional[Path]:
    if not run_dir or not run_id:
        return None
    return run_dir / "progress" / "run_progress.json"


def _progress_path_in_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / "progress" / f"run_{run_id}.json"


def _prepare_profile_run(
    output_dir_param: Optional[str],
    run_id_param: Optional[str],
    tool_key: str,
    label: str,
) -> tuple[Path, Path, str]:
    """Resolve standard run + profile directories for profiling tools."""
    from core.benchmark.artifact_manager import ArtifactManager, slugify

    base_dir = Path(output_dir_param) if output_dir_param else CODE_ROOT / "artifacts" / "runs"
    if not base_dir.is_absolute():
        base_dir = (CODE_ROOT / base_dir).resolve()
    run_label = label or "run"
    run_id = run_id_param or _default_run_id(f"profile-{tool_key}", run_label, base_dir)
    artifact_manager = ArtifactManager(
        base_dir=base_dir, run_id=run_id, run_kind=f"profile-{tool_key}", run_label=run_label
    )
    profile_dir = artifact_manager.profiles_dir / "tools" / slugify(tool_key) / slugify(run_label)
    profile_dir.mkdir(parents=True, exist_ok=True)
    return artifact_manager.run_dir, profile_dir, run_id


@contextlib.contextmanager
def _profile_workdir_env(workdir: Path):
    """Force profiling helpers to execute from a stable repository root cwd."""
    prev = os.environ.get("AISP_PROFILE_WORKDIR")
    os.environ["AISP_PROFILE_WORKDIR"] = str(workdir)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AISP_PROFILE_WORKDIR", None)
        else:
            os.environ["AISP_PROFILE_WORKDIR"] = prev


def _normalize_profile_command(command: Any) -> tuple[List[str], Optional[str]]:
    """Normalize profiling command argv with repo-root script path resolution."""
    if isinstance(command, str):
        try:
            command_list = shlex.split(command)
        except Exception:
            command_list = [command]
    elif isinstance(command, list):
        command_list = [str(part) for part in command]
    else:
        return [], "command must be a string or argv list"

    if not command_list:
        return [], None

    normalized = list(command_list)

    def _resolve_repo_script(script_arg: str) -> tuple[str, Optional[str]]:
        script_path = Path(script_arg)
        if script_path.is_absolute():
            if not script_path.exists():
                return script_arg, f"command script not found: {script_arg}"
            return script_arg, None

        repo_candidate = (CODE_ROOT / script_path).resolve()
        if repo_candidate.exists():
            return str(repo_candidate), None

        cwd_candidate = (Path.cwd() / script_path).resolve()
        if cwd_candidate.exists():
            return str(cwd_candidate), None

        return (
            script_arg,
            f"command script not found; checked repo root and cwd: {script_arg}",
        )

    exe_name = Path(normalized[0]).name.lower()
    if exe_name.startswith("python") and len(normalized) > 1:
        second = normalized[1]
        if second.endswith(".py"):
            normalized_second, error = _resolve_repo_script(second)
            normalized[1] = normalized_second
            if error:
                return normalized, error
    elif normalized[0].endswith(".py"):
        normalized_first, error = _resolve_repo_script(normalized[0])
        normalized[0] = normalized_first
        if error:
            return normalized, error

    return normalized, None


_PROGRESS_EMIT_WARNED_PATHS: set[str] = set()


def _read_progress_payload(progress_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not progress_path:
        return None
    path = Path(progress_path)
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception as exc:
        return {
            "current": None,
            "history": [],
            "load_warning": f"Failed to read progress payload from {path}: {exc}",
        }


def _emit_progress_safe(recorder: Optional[ProgressRecorder], event: ProgressEvent) -> None:
    if recorder is None:
        return
    try:
        recorder.emit(event)
    except Exception as exc:
        try:
            recorder_key = str(
                getattr(recorder, "progress_path", getattr(recorder, "run_id", "unknown"))
            )
        except Exception:
            recorder_key = "unknown"
        if recorder_key in _PROGRESS_EMIT_WARNED_PATHS:
            return
        _PROGRESS_EMIT_WARNED_PATHS.add(recorder_key)
        try:
            print(
                json.dumps(
                    {
                        "event": "progress_emit_warning",
                        "progress_path": recorder_key,
                        "message": f"Failed to emit progress update: {exc}",
                    }
                ),
                file=sys.stderr,
            )
        except Exception as fallback_exc:
            sys.__stderr__.write(
                f"progress_emit_warning_fallback path={recorder_key} error={fallback_exc}\n"
            )


def _trim_value(
    value: Any, max_length: int = _PREVIEW_MAX_LENGTH, max_items: int = _PREVIEW_MAX_ITEMS
) -> Any:
    """Lightly trim values so MCP responses stay readable without hiding detail (very high limits)."""
    if isinstance(value, str):
        if len(value) > max_length:
            return f"{value[:max_length]}... (truncated {len(value) - max_length} chars)"
        return value
    if isinstance(value, list):
        if len(value) > max_items:
            return value[:max_items] + [f"... ({len(value) - max_items} more)"]
        return value
    if isinstance(value, dict):
        if len(value) > max_items:
            keys = list(value.keys())[:max_items]
            trimmed = {k: value[k] for k in keys}
            trimmed["_truncated"] = f"{len(value) - max_items} more entries"
            return trimmed
        return value
    return value


def _sanitize_arguments(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a safe snapshot of arguments without huge payloads."""
    if not arguments:
        return {}
    return {k: _trim_value(v) for k, v in arguments.items()}


def _argument_details(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Provide extra metadata about arguments to aid chat UX."""
    if not arguments:
        return {}
    details: Dict[str, Any] = {}
    for k, v in arguments.items():
        info: Dict[str, Any] = {"type": type(v).__name__}
        try:
            info["length"] = len(v)  # type: ignore[arg-type]
        except TypeError:
            info["length"] = None
        info["preview"] = _trim_value(v)
        details[k] = info
    return details


def _result_metadata(result: Any) -> Dict[str, Any]:
    """Summarize the shape/type of the result for quick inspection."""
    meta: Dict[str, Any] = {"type": type(result).__name__}
    try:
        if isinstance(result, dict):
            meta["keys"] = list(result.keys())
            meta["size"] = len(result)
        elif isinstance(result, list):
            meta["size"] = len(result)
        elif isinstance(result, (str, bytes)):
            meta["length"] = len(result)
    except TypeError as exc:
        meta["metadata_warning"] = f"length unavailable: {exc}"
    return meta


def _cuda_precheck() -> Dict[str, Any]:
    """Lightweight CUDA/GPU availability snapshot for precheck-only flows."""
    try:
        import torch
    except ImportError as exc:
        return {
            "ok": False,
            "torch_available": False,
            "cuda_available": False,
            "reason": f"torch not available: {exc}",
        }

    cuda_ok = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "ok": cuda_ok,
        "torch_available": True,
        "cuda_available": cuda_ok,
        "device_count": torch.cuda.device_count() if cuda_ok else 0,
    }
    if not cuda_ok:
        info["reason"] = "CUDA not available"
    return info


def _looks_like_error(result: Any, had_exception: bool = False) -> bool:
    """Heuristic to flag errors so callers can react quickly."""
    if had_exception:
        return True
    if isinstance(result, dict):
        if result.get("error"):
            return True
        if result.get("success") is False:
            return True
    return False


def _normalize_result(result: Any) -> Any:
    """Ensure results consistently expose a success flag when possible."""
    if isinstance(result, dict):
        normalized = dict(result)
        if "success" not in normalized:
            normalized["success"] = not bool(normalized.get("error"))
        if normalized.get("error") and "error_type" not in normalized:
            message = str(normalized["error"]).lower()
            if "required" in message or "must be" in message or "invalid" in message:
                normalized["error_type"] = "value_error"
            elif "not found" in message:
                normalized["error_type"] = "file_not_found"
            else:
                normalized["error_type"] = "runtime_error"
        return normalized
    return result


JOB_STORE = JobStore.get()


def _cleanup_job_store(now: Optional[float] = None) -> None:
    """Evict old job records to keep long-running MCP sessions bounded."""
    JOB_STORE.cleanup(now)


def _queue_job(
    tool_name: str,
    runner: Callable[[], Any],
    arguments: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a task in the background and return a ticket for polling."""
    return JOB_STORE.queue_job(
        tool_name,
        runner,
        arguments=_sanitize_arguments(arguments),
        run_metadata=run_metadata,
        job_id=job_id,
    )


# =============================================================================
# BENCH/PROFILE QUEUE RUNNER (serialized execution under artifacts/parallel_runs)
# =============================================================================

_QUEUE_DIR = CODE_ROOT / "artifacts" / "parallel_runs"
_QUEUE_SCRIPT_PATH = _QUEUE_DIR / "bench_queue.sh"
_QUEUE_LOG_PATH = _QUEUE_DIR / "queue.log"
_QUEUE_LOCK = threading.Lock()
_QUEUE_LOG_LOCK = threading.Lock()
_QUEUE_WARN_LOCK = threading.Lock()
_QUEUE_MAX_OVERLAP_RETRIES_ENV = "AISP_MCP_QUEUE_MAX_OVERLAP_RETRIES"
_QUEUE_IDLE_TIMEOUT_ENV = "AISP_MCP_QUEUE_IDLE_TIMEOUT_SEC"
_QUEUE_RUNNER_LOCK_TIMEOUT_ENV = "AISP_MCP_QUEUE_RUNNER_LOCK_TIMEOUT_SEC"
_QUEUE_ARTIFACT_WARNED_KEYS: set[str] = set()


def _queue_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _emit_queue_warning_once(kind: str, path: Optional[Path], message: str) -> None:
    path_text = str(path) if path is not None else ""
    warning_key = f"{kind}:{path_text}"
    with _QUEUE_WARN_LOCK:
        if warning_key in _QUEUE_ARTIFACT_WARNED_KEYS:
            return
        _QUEUE_ARTIFACT_WARNED_KEYS.add(warning_key)
    payload = {
        "event": "queue_artifact_warning",
        "kind": kind,
        "message": message,
    }
    if path_text:
        payload["path"] = path_text
    try:
        print(json.dumps(payload), file=sys.stderr)
    except Exception as exc:
        sys.__stderr__.write(
            f"queue_artifact_warning_fallback kind={kind} path={path_text} error={exc}\n"
        )


def _queue_max_overlap_retries() -> int:
    raw = os.environ.get(_QUEUE_MAX_OVERLAP_RETRIES_ENV, "2")
    try:
        value = int(raw)
    except Exception:
        value = 2
    return max(0, value)


def _queue_idle_timeout_seconds() -> int:
    raw = os.environ.get(_QUEUE_IDLE_TIMEOUT_ENV, "600")
    try:
        value = int(raw)
    except Exception:
        value = 600
    return max(0, value)


def _queue_runner_lock_timeout_seconds() -> int:
    raw = os.environ.get(_QUEUE_RUNNER_LOCK_TIMEOUT_ENV, "1800")
    try:
        value = int(raw)
    except Exception:
        value = 1800
    return max(0, value)


def _ensure_queue_artifacts() -> bool:
    lock_path = _QUEUE_DIR / "queue.runner.lock"
    try:
        _QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _emit_queue_warning_once(
            "queue_dir_init",
            _QUEUE_DIR,
            f"Failed to initialize queue artifact directory {_QUEUE_DIR}: {exc}",
        )
        return False
    if not _QUEUE_SCRIPT_PATH.exists():
        header = (
            "#!/usr/bin/env bash\n"
            "# MCP queue runner manifest (append-only).\n"
            "# Entries are recorded automatically by MCP to serialize bench/profiling runs.\n"
        )
        try:
            _QUEUE_SCRIPT_PATH.write_text(header, encoding="utf-8")
        except Exception as exc:
            _emit_queue_warning_once(
                "queue_script_init",
                _QUEUE_SCRIPT_PATH,
                f"Failed to initialize queue script manifest {_QUEUE_SCRIPT_PATH}: {exc}",
            )
            return False
    try:
        lock_path.touch(exist_ok=True)
    except Exception as exc:
        _emit_queue_warning_once(
            "queue_lock_init",
            lock_path,
            f"Failed to initialize queue runner lock {lock_path}: {exc}",
        )
        return False
    return True


@contextlib.contextmanager
def _queue_runner_lock(queue_id: str, tool_name: str, queue_label: Optional[str]):
    """
    Cross-process queue lock so multiple MCP server processes cannot run
    bench/profiling payloads concurrently.
    """
    if fcntl is None:
        yield
        return
    if not _ensure_queue_artifacts():
        raise RuntimeError(
            "Queue runner unavailable: failed to initialize artifacts/parallel_runs."
        )

    lock_path = _QUEUE_DIR / "queue.runner.lock"
    timeout = _queue_runner_lock_timeout_seconds()
    started = time.monotonic()
    wait_logged = False
    lock_handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if not wait_logged:
                    _log_queue_event(
                        f"RUNNER_LOCK_WAIT id={queue_id} tool={tool_name} label={queue_label} timeout_s={timeout}"
                    )
                    wait_logged = True
                if timeout > 0 and (time.monotonic() - started) >= timeout:
                    raise RuntimeError(
                        f"Queue runner lock timeout after {timeout}s "
                        f"(tool={tool_name}, id={queue_id}, label={queue_label})"
                    )
                time.sleep(1.0)
        payload = {
            "queue_id": queue_id,
            "tool": tool_name,
            "label": queue_label,
            "pid": os.getpid(),
            "ts": _queue_timestamp(),
        }
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(json.dumps(payload, default=str) + "\n")
        lock_handle.flush()
        _log_queue_event(f"RUNNER_LOCK_ACQUIRED id={queue_id} tool={tool_name} label={queue_label}")
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                _log_queue_event(
                    f"RUNNER_LOCK_RELEASE_WARN id={queue_id} tool={tool_name} label={queue_label} error={exc}"
                )
            _log_queue_event(
                f"RUNNER_LOCK_RELEASED id={queue_id} tool={tool_name} label={queue_label}"
            )
        try:
            lock_handle.close()
        except OSError as exc:
            _log_queue_event(
                f"RUNNER_LOCK_CLOSE_WARN id={queue_id} tool={tool_name} label={queue_label} error={exc}"
            )


def _append_queue_script(entry: Dict[str, Any]) -> None:
    if not _ensure_queue_artifacts():
        return
    try:
        payload = json.dumps(entry, default=str)
    except Exception:
        payload = str(entry)
    line = f"# QUEUED {_queue_timestamp()} {payload}\n"
    try:
        with _QUEUE_SCRIPT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception as exc:
        _emit_queue_warning_once(
            "queue_script_append",
            _QUEUE_SCRIPT_PATH,
            f"Failed to append queue manifest entry to {_QUEUE_SCRIPT_PATH}: {exc}",
        )


def _log_queue_event(message: str) -> None:
    if not _ensure_queue_artifacts():
        return
    line = f"[{_queue_timestamp()}] {message}\n"
    try:
        with _QUEUE_LOG_LOCK:
            with _QUEUE_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception as exc:
        _emit_queue_warning_once(
            "queue_log_append",
            _QUEUE_LOG_PATH,
            f"Failed to append queue log entry to {_QUEUE_LOG_PATH}: {exc}",
        )


def _snapshot_processes() -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,command"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    lines = result.stdout.splitlines()
    records: List[Dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        try:
            pid_val = int(pid)
            ppid_val = int(ppid)
        except ValueError:
            continue
        records.append({"pid": pid_val, "ppid": ppid_val, "cmd": cmd})
    return records


def _descendant_pids(root_pid: int, records: List[Dict[str, Any]]) -> set[int]:
    children_map: Dict[int, List[int]] = defaultdict(list)
    for rec in records:
        children_map[rec["ppid"]].append(rec["pid"])
    descendants = set()
    queue = deque([root_pid])
    while queue:
        current = queue.popleft()
        if current in descendants:
            continue
        descendants.add(current)
        for child in children_map.get(current, []):
            if child not in descendants:
                queue.append(child)
    return descendants


def _is_active_run_command(cmd: str) -> bool:
    tokens = cmd.split()
    if not tokens:
        return False
    if "core.benchmark.bench_commands" in cmd and " run " in f" {cmd} ":
        return True
    if "bench run" in cmd and (
        "cli.aisp" in cmd or "aisp.py" in cmd or "python -m cli.aisp" in cmd
    ):
        return True
    if "bench" in tokens and "run" in tokens:
        exe = Path(tokens[0]).name
        if exe == "aisp":
            return True
    if "torchrun_wrapper.py" in cmd or "core.harness.torchrun_wrapper" in cmd:
        return True
    exe = Path(tokens[0]).name
    if exe == "nsys" and "profile" in tokens:
        return True
    if exe == "ncu":
        return True
    return False


def _active_run_processes(
    records: List[Dict[str, Any]], ignore_pids: set[int]
) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    for rec in records:
        if rec["pid"] in ignore_pids:
            continue
        cmd = rec["cmd"]
        if (
            "parallel_runs" in cmd
            or "bench_queue.sh" in cmd
            or "queue.log" in cmd
            or "queued_cmd." in cmd
            or "queued_wrapper." in cmd
            or "run_queue.sh" in cmd
        ):
            continue
        if _is_active_run_command(cmd):
            active.append(rec)
    return active


def _wait_for_idle(poll_seconds: int = 5, timeout_seconds: Optional[int] = None) -> bool:
    logged = False
    timeout = (
        _queue_idle_timeout_seconds() if timeout_seconds is None else max(0, int(timeout_seconds))
    )
    start = time.monotonic()
    while True:
        records = _snapshot_processes()
        ignore_pids = _descendant_pids(os.getpid(), records)
        active = _active_run_processes(records, ignore_pids)
        if not active:
            if logged:
                _log_queue_event("System idle: no active bench/profiling processes.")
            return True
        if not logged:
            _log_queue_event(f"Waiting for idle; active processes={len(active)}")
            logged = True
        if timeout > 0 and (time.monotonic() - start) >= timeout:
            _log_queue_event(
                f"WAIT_IDLE_TIMEOUT timeout_s={timeout} active_processes={len(active)} proceeding_anyway=true"
            )
            return False
        time.sleep(poll_seconds)


def _monitor_overlap(
    stop_event: threading.Event,
    overlap_event: threading.Event,
    *,
    baseline_ignore_pids: Optional[set[int]] = None,
    poll_seconds: int = 5,
) -> None:
    baseline_roots = set(int(pid) for pid in (baseline_ignore_pids or set()))
    while not stop_event.is_set():
        records = _snapshot_processes()
        ignore_pids = _descendant_pids(os.getpid(), records)
        for root in baseline_roots:
            ignore_pids.update(_descendant_pids(root, records))
        active = _active_run_processes(records, ignore_pids)
        if active:
            overlap_event.set()
        time.sleep(poll_seconds)


def _extract_exit_code(result: Any) -> Optional[int]:
    if isinstance(result, dict):
        if "returncode" in result:
            try:
                return int(result.get("returncode"))
            except (TypeError, ValueError):
                return None
        if "success" in result:
            return 0 if result.get("success") else 1
    return None


def _attach_queue_metadata(
    result: Any,
    retries: int,
    overlap_detected: bool,
    idle_wait_timed_out: bool = False,
) -> Any:
    if not isinstance(result, dict):
        return result
    queue_info = {
        "queue_log": str(_QUEUE_LOG_PATH),
        "queue_script": str(_QUEUE_SCRIPT_PATH),
        "queue_retries": retries,
        "overlap_detected": overlap_detected,
        "idle_wait_timed_out": idle_wait_timed_out,
    }
    result.setdefault("queue", {}).update(queue_info)
    return result


def _run_with_queue(
    tool_name: str,
    runner: Callable[[], Any],
    *,
    queue_label: Optional[str] = None,
    queue_payload: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> Any:
    if not _ensure_queue_artifacts():
        raise RuntimeError(
            "Queue runner unavailable: failed to initialize artifacts/parallel_runs."
        )
    queue_id = job_id or f"{tool_name}-{uuid.uuid4().hex[:8]}"
    entry = {
        "queue_id": queue_id,
        "tool": tool_name,
        "label": queue_label,
        "payload": queue_payload or {},
    }
    _append_queue_script(entry)
    if job_id:
        JOB_STORE.update_job(job_id, status="queued", note="Waiting for MCP queue runner.")
    with _QUEUE_LOCK:
        with _queue_runner_lock(queue_id, tool_name, queue_label):
            if job_id:
                JOB_STORE.update_job(job_id, status="running", note="Running via MCP queue runner.")
            retries = 0
            overlap_detected = False
            idle_wait_timed_out = False
            max_overlap_retries = _queue_max_overlap_retries()
            while True:
                idle_now = _wait_for_idle()
                if not idle_now:
                    idle_wait_timed_out = True
                    records = _snapshot_processes()
                    ignore_pids = _descendant_pids(os.getpid(), records)
                    active = _active_run_processes(records, ignore_pids)
                    timeout_s = _queue_idle_timeout_seconds()
                    _log_queue_event(
                        f"WAIT_IDLE_ABORT id={queue_id} tool={tool_name} label={queue_label} "
                        f"timeout_s={timeout_s} active_processes={len(active)}"
                    )
                    failure = {
                        "success": False,
                        "error": (
                            f"Queue idle wait timed out after {timeout_s}s; refusing to run while other "
                            "bench/profiling processes are active."
                        ),
                        "active_process_count": len(active),
                        "active_processes": [rec.get("cmd", "") for rec in active[:5]],
                    }
                    return _attach_queue_metadata(
                        failure, retries, overlap_detected, idle_wait_timed_out
                    )
                _log_queue_event(
                    f"RUN_START id={queue_id} tool={tool_name} label={queue_label} attempt={retries + 1}"
                )
                start_ts = time.time()
                stop_event = threading.Event()
                overlap_event = threading.Event()
                baseline_overlap_roots: set[int] = set()
                if idle_wait_timed_out:
                    # If we had to bypass idle-wait timeout, ignore those pre-existing
                    # active runs when deciding overlap for this attempt.
                    baseline_records = _snapshot_processes()
                    baseline_ignore = _descendant_pids(os.getpid(), baseline_records)
                    baseline_active = _active_run_processes(baseline_records, baseline_ignore)
                    baseline_overlap_roots = {int(rec["pid"]) for rec in baseline_active}
                monitor = threading.Thread(
                    target=_monitor_overlap,
                    args=(stop_event, overlap_event),
                    kwargs={"baseline_ignore_pids": baseline_overlap_roots},
                    daemon=True,
                )
                monitor.start()
                try:
                    result = runner()
                except Exception as exc:
                    stop_event.set()
                    monitor.join()
                    _log_queue_event(
                        f"RUN_END id={queue_id} tool={tool_name} label={queue_label} exit_code=1 error={exc}"
                    )
                    raise
                stop_event.set()
                monitor.join()
                duration = time.time() - start_ts
                overlap = overlap_event.is_set()
                if overlap:
                    overlap_detected = True
                exit_code = _extract_exit_code(result)
                _log_queue_event(
                    f"RUN_END id={queue_id} tool={tool_name} label={queue_label} "
                    f"exit_code={exit_code} overlap={overlap} duration_s={duration:.2f}"
                )
                if overlap:
                    # Avoid endless requeue loops on persistent overlap/noise.
                    # Only requeue on an explicit successful run (exit_code == 0).
                    # Any non-zero or unknown exit should fail loudly and stop retries.
                    if exit_code != 0:
                        _log_queue_event(
                            f"OVERLAP_NOT_REQUEUED id={queue_id} tool={tool_name} "
                            f"label={queue_label} reason=non_success_exit exit_code={exit_code}"
                        )
                        result = _attach_queue_metadata(
                            result, retries, overlap_detected, idle_wait_timed_out
                        )
                        if isinstance(result, dict):
                            result.setdefault("queue", {})
                            result["queue"]["overlap_retry_limit"] = max_overlap_retries
                            result["queue"]["overlap_retry_exhausted"] = False
                        return result
                    retries += 1
                    if retries > max_overlap_retries:
                        _log_queue_event(
                            f"OVERLAP_RETRY_EXHAUSTED id={queue_id} tool={tool_name} "
                            f"label={queue_label} retries={retries - 1} limit={max_overlap_retries}"
                        )
                        result = _attach_queue_metadata(
                            result, retries - 1, overlap_detected, idle_wait_timed_out
                        )
                        if isinstance(result, dict):
                            result.setdefault("queue", {})
                            result["queue"]["overlap_retry_limit"] = max_overlap_retries
                            result["queue"]["overlap_retry_exhausted"] = True
                        return result
                    _log_queue_event(
                        f"REQUEUE id={queue_id} tool={tool_name} label={queue_label} reason=overlap"
                    )
                    continue
                result = _attach_queue_metadata(
                    result, retries, overlap_detected, idle_wait_timed_out
                )
                if isinstance(result, dict):
                    result.setdefault("queue", {})
                    result["queue"]["overlap_retry_limit"] = max_overlap_retries
                    result["queue"]["overlap_retry_exhausted"] = False
                return result


def _context_snapshot() -> Dict[str, Any]:
    """Provide cached context (summary + full) so callers have rich environment data."""
    try:
        ctx_summary = get_cached_context("summary")
        ctx_full = get_cached_context("full")
        summary_age = max(0, time.time() - _CONTEXT_TS.get("summary", 0.0))
        full_age = max(0, time.time() - _CONTEXT_TS.get("full", 0.0))
        return {
            "summary": _trim_value(ctx_summary),
            "summary_age_seconds": int(summary_age),
            "full": _trim_value(ctx_full),
            "full_age_seconds": int(full_age),
        }
    except Exception as e:
        return {"context_error": f"context snapshot failed: {e}"}


def _default_next_steps(tool_name: str, status_is_error: bool) -> List[str]:
    """Actionable nudges to keep the caller moving."""
    steps: List[str] = [
        "If you need environment details, set include_context=true on this tool or call context_summary.",
        "If you're unsure what to run next, call suggest_tools with a short intent.",
    ]
    if status_is_error:
        steps.insert(0, "Call status to check GPU/software health after this failure.")
    elif tool_name not in {"triage", "context_summary", "context_full"}:
        steps.insert(0, "Use triage first if you still need a quick status snapshot.")
    return steps


def _build_enriched_tool_payload(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    result: Any,
    duration_ms: int,
    had_exception: bool = False,
    server_info: Optional[Dict[str, Any]] = None,
    tool_meta: Optional[ToolDefinition] = None,
) -> Dict[str, Any]:
    """Wrap raw tool output with metadata/context so MCP callers get richer responses."""
    status_is_error = _looks_like_error(result, had_exception)
    payload: Dict[str, Any] = {
        "tool": tool_name,
        "status": "error" if status_is_error else "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
        "arguments": _sanitize_arguments(arguments),
        "arguments_details": _argument_details(arguments),
        "result": result,
        "result_preview": _trim_value(
            result, max_length=_PREVIEW_MAX_LENGTH, max_items=_PREVIEW_MAX_ITEMS
        ),
        "result_metadata": _result_metadata(result),
    }
    if server_info:
        payload["server"] = server_info
    if tool_meta:
        payload["tool_description"] = tool_meta.description
        payload["input_schema"] = tool_meta.input_schema

    # Always provide a lightweight context snapshot for continuity.
    payload["context_summary"] = _context_snapshot()

    payload["guidance"] = {"next_steps": _default_next_steps(tool_name, status_is_error)}
    return payload


def _content_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build MCP content entries for structured consumption.
    Returns text (JSON string) to align with MCP content type expectations (text/image/audio/resource).
    """
    try:
        text_payload = json.dumps(payload, indent=2, default=str)
    except Exception:
        text_payload = str(payload)

    return [{"type": "text", "text": text_payload}]


# =============================================================================
# GPU TOOLS
# =============================================================================


@register_tool(
    "gpu_info",
    "Tags: gpu, info, snapshot, health-check, inventory, nvidia-smi. "
    "Get GPU hardware snapshot: name, architecture, VRAM (total/used/free), temperature, power draw, utilization %. "
    "Returns: {gpus: [{name, memory_total_gb, memory_used_gb, temperature_c, power_w, utilization_pct}], count}. "
    "⚡ FAST (~1s). USE FIRST when: Starting any performance investigation, verifying hardware before profiling. "
    "USE INSTEAD OF: Running nvidia-smi manually in terminal. "
    'Example: "Show GPU names, memory, temps" or "What GPUs do I have?" or "Check VRAM before loading model". '
    "WORKFLOW: gpu_info → status → recommend → specific optimization tools. "
    "NOT FOR: Feature detection (info_features), topology (gpu_topology), power throttling (gpu_power).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_gpu_info(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU information."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.info()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "gpu_bandwidth",
    "Tags: bandwidth, memory, hbm, nvlink, throughput, spec-check. "
    "Run GPU memory bandwidth test measuring actual vs theoretical HBM bandwidth. "
    "Returns: {bandwidth_gbps, theoretical_gbps, efficiency_pct, test_size_mb}. "
    "USE when: Validating memory throughput matches GPU spec, diagnosing memory-bound kernels, checking for HBM/PCIe issues. "
    'Example: "Check H100 bandwidth vs spec" or "Why is my memory-bound kernel slow?". '
    "NOT FOR: PCIe H2D/D2H bandwidth (use hw_pcie), GPU-to-GPU P2P (use hw_p2p). 🕐 MEDIUM (~10s). WORKFLOW: gpu_info → gpu_bandwidth → diagnose memory-bound.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_gpu_bandwidth(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run GPU bandwidth test."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.bandwidth_test()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "gpu_topology",
    "Tags: topology, nvlink, pcie, multi-gpu, interconnect, p2p, numa. "
    "Get multi-GPU topology: NVLink/PCIe connections, NUMA affinity, P2P capability matrix. "
    "Returns: {gpu_count, connections: [{src, dst, type, bandwidth_gbps}], numa_nodes, p2p_matrix}. "
    "USE when: Planning tensor/pipeline parallelism, debugging P2P transfer issues, optimizing GPU placement. "
    'Example: "Show NVLink/PCIe layout on 8x GPU server" or "Which GPUs have NVLink?". '
    "NOT FOR: Raw topology matrix output (use gpu_topology_matrix for nvidia-smi topo -m). ⚡ FAST (~2s). WORKFLOW: gpu_topology → distributed_plan → distributed_nccl.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_gpu_topology(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU topology."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().gpu.topology()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "gpu_power",
    "Tags: power, thermal, throttling, headroom, tdp, temperature. "
    "Get GPU power and thermal status: current power draw, power limit, temperature, throttling state. "
    "Returns: {gpus: [{power_w, power_limit_w, headroom_w, temperature_c, throttling, fan_pct}]}. "
    "USE when: Checking for thermal/power throttling, verifying TDP headroom before heavy workloads. "
    'Example: "Are GPUs power-throttling right now?" or "How much thermal headroom do I have?". '
    "NOT FOR: General GPU info (use gpu_info), sustained power monitoring over time. ⚡ FAST (~1s). WORKFLOW: gpu_power → if throttling → reduce workload.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_gpu_power(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU power info."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.power()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# SYSTEM TOOLS
# =============================================================================


@register_tool(
    "system_software",
    "Tags: software, versions, pytorch, cuda, python, driver, stack. "
    "Get software stack versions: PyTorch, CUDA toolkit, cuDNN, Python, NVIDIA driver. "
    "Returns: {pytorch_version, cuda_version, cudnn_version, python_version, driver_version, transformers_version}. "
    "USE when: Filing bug reports, checking compatibility, reproducing issues, verifying install. "
    'Example: "What PyTorch and CUDA versions are installed?" or "Is my CUDA version compatible with FlashAttention?". '
    "NOT FOR: Checking if dependencies import correctly (use system_dependencies). ⚡ FAST (~1s). WORKFLOW: system_software → check → verify compatibility.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_software(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get software information."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.software()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_dependencies",
    "Tags: deps, health, import, missing, broken, install, torch.cuda. "
    "Check health of ML/AI dependencies: torch, triton, flash-attn, transformers, vllm, etc. "
    "Returns: {dependencies: [{name, installed, importable, version, error}], healthy_count, broken_count}. "
    "USE when: Diagnosing import errors, checking if optional dependencies are available, debugging install issues. "
    'Example: "Why does torch.cuda fail to import?" or "Is flash-attn installed correctly?". '
    "NOT FOR: Version numbers only (use system_software), general system health (use status). ⚡ FAST (~2s). WORKFLOW: system_dependencies → fix broken → retry.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check dependency health."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.dependencies()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_parameters",
    "Tags: kernel, sysctl, tuning, swappiness, dirty_ratio, numa, io, networking. "
    "Inspect kernel parameters that commonly affect performance (swappiness, dirty ratios, NUMA balancing, net buffers). "
    "Returns: {parameters: [{name, path, current, recommended, description, needs_tuning}], quick_tune_commands}. "
    "USE when: Validating host tuning before benchmarks or diagnosing IO/NUMA slowdowns. "
    'Example: "Check system tuning parameters" or "Show swappiness and dirty ratios". '
    "NOT FOR: Software versions (use system_software). ⚡ FAST (~1s). WORKFLOW: system_parameters → apply quick_tune_commands.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect kernel parameters that impact performance."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.parameters()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_container",
    "Tags: container, cgroup, limits, quota, memory, cpu, docker, k8s. "
    "Inspect container/cgroup limits (CPU quota, memory limit, cgroup version). "
    "Returns: {in_container, container_type, cgroup_version, cpu_limit, memory_limit_gb, recommendations}. "
    "USE when: Diagnosing throttling in containers or Kubernetes. "
    'Example: "Am I CPU throttled in this container?" or "Show cgroup limits". '
    "NOT FOR: Hardware capabilities (use system_capabilities). ⚡ FAST (~1s).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_container(params: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect container/cgroup limits."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.container()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_cpu_memory",
    "Tags: cpu, numa, cache, memory, tlb, topology, host-bound. "
    "Analyze CPU/memory hierarchy (NUMA nodes, cache sizes, memory stats). "
    "Returns: {cpu, cache_hierarchy, memory, numa, tlb, recommendations}. "
    "USE when: Diagnosing host-bound workloads or dataloader bottlenecks. "
    'Example: "Show NUMA layout" or "Check CPU cache hierarchy". '
    "NOT FOR: GPU topology (use gpu_topology). ⚡ FAST (~1s).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_cpu_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze CPU/memory hierarchy."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.cpu_memory()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_env",
    "Tags: environment, vars, cuda, nccl, torch, paths. "
    "Snapshot key environment variables and working directory. "
    "Returns: {cwd, reported_keys, environment}. "
    "USE when: Debugging env-dependent behavior or verifying CUDA/NCCL settings. "
    'Example: "Show CUDA/NCCL env vars". '
    "NOT FOR: Full system context (use system_context). ⚡ FAST (~1s).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_env(params: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot environment variables relevant to performance."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.env()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_network",
    "Tags: network, ib, infiniband, rdma, gpudirect, nccl, interfaces. "
    "Inspect network interfaces, InfiniBand status, and GPUDirect/NCCL env hints. "
    "Returns: {interfaces, infiniband, gpudirect_rdma, nccl_env}. "
    "USE when: Debugging multi-node networking or NCCL connectivity issues. "
    'Example: "Check InfiniBand status" or "Show NCCL network env". '
    "NOT FOR: NCCL collective tuning (use distributed_nccl). ⚡ FAST (~1s).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_network(params: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect network/InfiniBand status."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.network()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "clock_lock_check",
    "Tags: clocks, lock, benchmark-validity, harness, nvidia-smi, sudo. "
    "Validate GPU clock locking works via the benchmark harness (required for canonical benchmark/profiling runs). "
    "Returns: {success, gpu_count, results:[{device, physical_index, locked, before/during/after clocks, error?}]}. "
    "⚡ FAST (~1-5s). USE when: You want to verify the host supports clock locking before running benchmarks. "
    "NOT FOR: Manually locking clocks; this tool uses the repo harness and resets clocks on exit.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "sm_clock_mhz": {
                    "type": "integer",
                    "description": "Target SM clock in MHz (default: max)",
                },
                "mem_clock_mhz": {
                    "type": "integer",
                    "description": "Target memory clock in MHz (default: max)",
                },
                "devices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional list of CUDA device indices to check (default: all visible GPUs)",
                },
            }
        ),
    },
)
def tool_clock_lock_check(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate harness clock locking works."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    devices_raw = params.get("devices")
    devices: Optional[List[int]] = None
    if isinstance(devices_raw, list):
        try:
            devices = [int(x) for x in devices_raw]
        except Exception:
            devices = None

    try:
        result = get_engine().system.clock_lock_check(
            sm_clock_mhz=params.get("sm_clock_mhz"),
            mem_clock_mhz=params.get("mem_clock_mhz"),
            devices=devices,
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"clock lock check failed: {e}", include_context, context_level)


@register_tool(
    "run_benchmarks",
    "Tags: benchmarks, run, profiling, performance-test, chapters, labs, validation. "
    "Run benchmarks via the bench CLI with optional profiling and LLM analysis. "
    "MCP serializes bench/profiling runs via a hidden queue under artifacts/parallel_runs to prevent overlap. "
    "Returns: {stdout, stderr, returncode, duration_seconds, results_json (best-effort), run_dir (best-effort), speedup_attribution (best-effort), suggested_next_steps}. "
    "⚠️ SLOW: 2-30+ minutes depending on targets. ALWAYS run status first! "
    "USE when: Validating optimizations, generating benchmark data for comparison. "
    'Example: "Run ch07 benchmarks" or "Benchmark attention examples". '
    "SAFE WORKFLOW: "
    "1. status → verify GPU/CUDA ready "
    "2. list_chapters → see what's available "
    "3. run_benchmarks(targets=['ch07'], dry_run=true) → preview command "
    "4. run_benchmarks(targets=['ch07'], async=true) → run in background "
    "5. job_status(job_id=...) → poll until complete "
    "6. benchmark_triage → analyze results and get recommendations. "
    "DEEP DIVE WORKFLOW (manual): "
    "1) benchmark_targets → pick target like 'ch10:atomic_reduction' "
    "2) run_benchmarks(targets=['ch10:atomic_reduction'], profile='deep_dive', artifacts_dir='artifacts/runs') "
    "3) benchmark_triage(data_file=results_json from step 2) "
    "4) profile_compare / compare_nsys / compare_ncu (point at a profiles_dir that contains baseline+optimized .nsys-rep/.ncu-rep). "
    "DEEP DIVE WORKFLOW (one-shot): use benchmark_deep_dive_compare for run+profile+diff in one call. "
    "WORKFLOW: status → list_chapters → run_benchmarks → benchmark_triage. "
    "By default, MCP runs post-benchmark triage + HTML report generation for richer detail, and emits non-LLM speedup attribution from profiler metrics when available. "
    "Disable with auto_analyze=false or auto_report=false if you only want raw results. "
    "NOT FOR: Quick GPU health (use hw_speed first).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Benchmark targets as chapter or chapter:example (or lab paths like labs/<lab>:example); "
                        "discover via benchmark_targets or list_chapters. "
                        "Examples: ['ch07'] or ['ch10:atomic_reduction']."
                    ),
                },
                "profile": {
                    "type": "string",
                    "description": "Profiling preset: none (no profiling), minimal (basic), deep_dive (full nsys/ncu profiling), or roofline",
                    "enum": ["none", "minimal", "deep_dive", "roofline"],
                    "default": "minimal",
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: ./artifacts/runs).",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for artifacts (default: <timestamp>__bench__profile-<type>__targets-<...>)",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override benchmark iterations (all targets).",
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations (all targets).",
                },
                "force_sync": {
                    "type": "boolean",
                    "description": "Force a device-wide synchronize immediately after benchmark_fn() (opt-in safeguard).",
                    "default": False,
                },
                "gpu_sm_clock_mhz": {
                    "type": "integer",
                    "description": (
                        "Lock the SM application clock (MHz) for this run (recommended for SOL comparisons). "
                        "Example: 1500 for B200/B200. Requires clock locking to be enabled in the harness."
                    ),
                },
                "gpu_mem_clock_mhz": {
                    "type": "integer",
                    "description": (
                        "Lock the GPU memory (HBM) application clock (MHz) for this run. "
                        "If omitted, the harness will lock memory to the max supported clock."
                    ),
                },
                "llm_analysis": {
                    "type": "boolean",
                    "description": "Enable LLM-powered analysis for benchmarks with <1.1x speedup. DISABLED BY DEFAULT (false) to avoid API costs. Only set to true when user explicitly requests: 'with LLM analysis', 'use AI insights', 'analyze with AI', 'get AI recommendations', or similar phrases. If user doesn't mention LLM/AI analysis, leave this false.",
                    "default": False,
                },
                "apply_patches": {"type": "boolean"},
                "force_llm": {
                    "type": "boolean",
                    "description": "Force LLM analysis on all benchmarks regardless of speedup (costs API credits).",
                    "default": False,
                },
                "rebenchmark_llm_patches": {
                    "type": "boolean",
                    "description": "Re-benchmark LLM-patched variants (requires apply_patches=true).",
                    "default": False,
                },
                "llm_explain": {
                    "type": "boolean",
                    "description": "Generate LLM explanations for best patches (requires rebenchmark_llm_patches=true).",
                    "default": False,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites and planned command without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the bench run without executing it (alias: estimate_only)",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning with partial output; set 0/null for no timeout",
                    "default": 900,
                },
                "timeout_multiplier": {
                    "type": "number",
                    "description": "Multiply all benchmark timeouts by this factor (e.g., 2.0 = double all timeouts).",
                    "default": 3.0,
                },
                "nsys_timeout_seconds": {
                    "type": "integer",
                    "description": "Override Nsight Systems timeout in seconds (default from BenchmarkDefaults).",
                    "default": None,
                },
                "ncu_timeout_seconds": {
                    "type": "integer",
                    "description": "Override Nsight Compute timeout in seconds (default from BenchmarkDefaults).",
                    "default": None,
                },
                "ncu_metric_set": {
                    "type": "string",
                    "description": (
                        "Nsight Compute metric preset: auto, minimal, deep_dive, or roofline. "
                        "Default is minimal for safer profiling."
                    ),
                    "enum": ["auto", "minimal", "deep_dive", "roofline"],
                    "default": "minimal",
                },
                "ncu_replay_mode": {
                    "type": "string",
                    "description": (
                        "Nsight Compute replay mode: kernel or application. "
                        "Default is kernel for safer profiling on dynamic workloads."
                    ),
                    "enum": ["kernel", "application"],
                    "default": "kernel",
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
                "accept_regressions": {
                    "type": "boolean",
                    "description": (
                        "Update expectation files when improvements are detected instead of flagging regressions."
                    ),
                    "default": False,
                },
                "allow_mixed_provenance": {
                    "type": "boolean",
                    "description": (
                        "Allow expectation updates when provenance differs (commit/hardware/profile mismatch) without "
                        "forcing updates. Does NOT accept regressions (use accept_regressions/update_expectations)."
                    ),
                    "default": False,
                },
                "update_expectations": {
                    "type": "boolean",
                    "description": (
                        "Force-write observed metrics into expectation files (overrides regressions). "
                        "Useful for refreshing baselines on new hardware."
                    ),
                    "default": False,
                },
                "only_cuda": {
                    "type": "boolean",
                    "description": "Run only CUDA binary benchmarks (Python wrappers).",
                    "default": False,
                },
                "only_python": {
                    "type": "boolean",
                    "description": "Run only Python benchmarks (skip CUDA binary wrappers).",
                    "default": False,
                },
                "auto_analyze": {
                    "type": "boolean",
                    "description": "Automatically run benchmark_triage after a successful run.",
                    "default": True,
                },
                "auto_report": {
                    "type": "boolean",
                    "description": "Automatically generate a benchmark report after a successful run.",
                    "default": True,
                },
                "report_format": {
                    "type": "string",
                    "description": "Report format used when auto_report=true (html or pdf).",
                    "enum": ["html", "pdf"],
                    "default": "html",
                },
            }
        ),
        "required": ["targets"],
    },
)
def tool_run_benchmarks(params: Dict[str, Any]) -> Dict[str, Any]:
    targets = params.get("targets") or []
    for raw_target in targets:
        target_text = str(raw_target).strip()
        if not target_text:
            return {
                "error": "Empty target provided. Targets must be chapter or chapter:example strings.",
                "success": False,
            }
        if target_text.startswith("-"):
            return {
                "error": (
                    f"Invalid benchmark target '{target_text}'. "
                    "Targets cannot begin with '-' (likely malformed CLI flags in target list)."
                ),
                "success": False,
            }

    # Normalize profile - uses _COMMON_ALIASES for most common mistakes
    raw_profile = params.get("profile") or "minimal"
    profile = normalize_param("profile", raw_profile, "minimal")
    artifacts_dir = params.get("artifacts_dir")
    if artifacts_dir:
        artifacts_dir = str(_resolve_artifact_path(str(artifacts_dir)))
    run_id_param = params.get("run_id")
    run_id = run_id_param.strip() if isinstance(run_id_param, str) else run_id_param
    if not run_id:
        from core.benchmark.artifact_manager import build_bench_run_label

        base_dir = Path(artifacts_dir) if artifacts_dir else CODE_ROOT / "artifacts" / "runs"
        if not base_dir.is_absolute():
            base_dir = (CODE_ROOT / base_dir).resolve()
        run_label = build_bench_run_label(targets, profile)
        run_id = _default_run_id("bench", run_label, base_dir)
    iterations_param = params.get("iterations")
    warmup_param = params.get("warmup")
    force_sync = bool(params.get("force_sync", False))
    gpu_sm_clock_mhz = params.get("gpu_sm_clock_mhz")
    gpu_mem_clock_mhz = params.get("gpu_mem_clock_mhz")
    timeout_multiplier = params.get("timeout_multiplier")
    nsys_timeout_seconds = params.get("nsys_timeout_seconds")
    ncu_timeout_seconds = params.get("ncu_timeout_seconds")
    ncu_metric_set = params.get("ncu_metric_set", "minimal")
    ncu_replay_mode = params.get("ncu_replay_mode", "kernel")
    raw_validity_profile = params.get("validity_profile", "strict")
    validity_profile = normalize_param("validity_profile", raw_validity_profile, "strict")
    if validity_profile not in {"strict", "portable"}:
        return {
            "error": (
                f"Invalid benchmark validity profile '{raw_validity_profile}' for validity_profile. "
                "Valid options: strict, portable."
            ),
            "success": False,
        }
    allow_portable_expectations_update = bool(
        params.get("allow_portable_expectations_update", False)
    )
    accept_regressions = bool(params.get("accept_regressions", False))
    allow_mixed_provenance = bool(params.get("allow_mixed_provenance", False))
    update_expectations = bool(params.get("update_expectations", False))
    only_python = bool(params.get("only_python", False))
    only_cuda = bool(params.get("only_cuda", False))
    auto_analyze = bool(params.get("auto_analyze", True))
    auto_report = bool(params.get("auto_report", True))
    test_mode = _is_test_mode()
    if test_mode:
        if "auto_analyze" not in params:
            auto_analyze = True
        if "auto_report" not in params:
            auto_report = False
        if "profile" not in params or profile in {"minimal", "deep_dive", "roofline"}:
            profile = "none"
        if "only_python" not in params and "only_cuda" not in params:
            only_python = True
        if "validity_profile" not in params:
            validity_profile = "portable"
    report_format = params.get("report_format", "html")

    # Validate profile value
    valid_profiles = ["none", "minimal", "deep_dive", "roofline"]
    if profile not in valid_profiles:
        return {
            "error": f"Invalid profile '{raw_profile}'. Valid options: {', '.join(valid_profiles)}",
            "hint": "Use 'deep_dive' for full nsys/ncu profiling",
            "success": False,
        }
    # LLM analysis disabled by default to avoid unexpected API costs
    # Enable explicitly with llm_analysis=true when you want AI-powered insights
    llm_analysis = params.get("llm_analysis", False)
    apply_patches = params.get("apply_patches", False)
    force_llm = bool(params.get("force_llm", False))
    rebenchmark_llm_patches = bool(params.get("rebenchmark_llm_patches", False))
    llm_explain = bool(params.get("llm_explain", False))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async", False))
    queue_job_id_raw = params.get("_queue_job_id")
    queue_job_id = str(queue_job_id_raw).strip() if queue_job_id_raw else None
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    if test_mode and (timeout_seconds is None or timeout_seconds <= 0):
        timeout_seconds = 600
    include_context, context_level = extract_context_opts(params)

    if (
        validity_profile == "portable"
        and not allow_portable_expectations_update
        and (update_expectations or accept_regressions or allow_mixed_provenance)
    ):
        return {
            "error": (
                "Portable validity profile does not write expectations unless "
                "allow_portable_expectations_update=true is set."
            ),
            "success": False,
        }

    if apply_patches and not (llm_analysis or force_llm):
        return {
            "error": "apply_patches requires llm_analysis=true or force_llm=true",
            "hint": "Set llm_analysis or force_llm to enable LLM analysis before patching.",
            "success": False,
        }
    if rebenchmark_llm_patches and not apply_patches:
        return {
            "error": "rebenchmark_llm_patches requires apply_patches=true",
            "hint": "Set apply_patches=true to generate patches before rebenchmarking.",
            "success": False,
        }
    if llm_explain and not rebenchmark_llm_patches:
        return {
            "error": "llm_explain requires rebenchmark_llm_patches=true",
            "hint": "Set rebenchmark_llm_patches=true to benchmark patches before explaining.",
            "success": False,
        }

    # Return async ticket immediately; execute preflight + bench in background.
    if run_async and not precheck_only and not dry_run:
        run_dir = _resolve_run_dir(artifacts_dir, str(run_id)) if run_id else None
        progress_path = _progress_path_for_run(run_dir, str(run_id)) if run_dir else None
        sync_params = dict(params)
        sync_params["async"] = False
        if run_id and not sync_params.get("run_id"):
            sync_params["run_id"] = str(run_id)
        if profile and not sync_params.get("profile"):
            sync_params["profile"] = profile
        if artifacts_dir and not sync_params.get("artifacts_dir"):
            sync_params["artifacts_dir"] = str(artifacts_dir)

        job_id = f"run_benchmarks-{uuid.uuid4().hex[:10]}"
        sync_params["_queue_job_id"] = job_id
        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else None,
            "progress_path": str(progress_path) if progress_path else None,
        }
        queued = _queue_job(
            "run_benchmarks",
            lambda: tool_run_benchmarks(sync_params),
            params,
            run_metadata=run_metadata,
            job_id=job_id,
        )
        queued["targets"] = targets
        queued["queue_log"] = str(_QUEUE_LOG_PATH)
        queued["queue_script"] = str(_QUEUE_SCRIPT_PATH)
        queued["note"] = (
            "Background benchmark accepted; poll with job_status using job_id. "
            "Preflight and benchmark execution run inside the queued job."
        )
        return queued

    llm_requested = bool(
        llm_analysis or force_llm or apply_patches or rebenchmark_llm_patches or llm_explain
    )
    if llm_requested:
        from core.llm import get_llm_status

        llm_status = get_llm_status()
        if os.getenv("LLM_ANALYSIS_ENABLED", "true").lower() == "false":
            return {
                "error": "LLM analysis disabled via LLM_ANALYSIS_ENABLED=false.",
                "llm_status": llm_status,
                "success": False,
            }
        if not llm_status.get("available"):
            return {
                "error": "LLM analysis requested but no LLM backend is configured.",
                "llm_status": llm_status,
                "hint": llm_status.get("warning")
                or "Set an LLM API key/base URL in .env/.env.local.",
                "success": False,
            }

    cuda_check = _cuda_precheck()

    args: List[str] = ["run", "--profile", profile]
    args.extend(["--validity-profile", str(validity_profile)])
    if artifacts_dir:
        args.extend(["--artifacts-dir", str(artifacts_dir)])
    if run_id:
        args.extend(["--run-id", str(run_id)])
    if iterations_param is not None:
        args.extend(["--iterations", str(int(iterations_param))])
    if warmup_param is not None:
        args.extend(["--warmup", str(int(warmup_param))])
    if force_sync:
        args.append("--force-sync")
    if gpu_sm_clock_mhz is not None:
        args.extend(["--gpu-sm-clock-mhz", str(int(gpu_sm_clock_mhz))])
    if gpu_mem_clock_mhz is not None:
        args.extend(["--gpu-mem-clock-mhz", str(int(gpu_mem_clock_mhz))])
    if timeout_multiplier is not None:
        args.extend(["--timeout-multiplier", str(float(timeout_multiplier))])
    if nsys_timeout_seconds is not None:
        args.extend(["--nsys-timeout-seconds", str(int(nsys_timeout_seconds))])
    if ncu_timeout_seconds is not None:
        args.extend(["--ncu-timeout-seconds", str(int(ncu_timeout_seconds))])
    if ncu_metric_set is not None:
        args.extend(["--ncu-metric-set", str(ncu_metric_set)])
    if ncu_replay_mode is not None:
        args.extend(["--ncu-replay-mode", str(ncu_replay_mode)])
    if allow_portable_expectations_update:
        args.append("--allow-portable-expectations-update")
    if accept_regressions:
        args.append("--accept-regressions")
    if allow_mixed_provenance:
        args.append("--allow-mixed-provenance")
    if update_expectations:
        args.append("--update-expectations")
    if only_python:
        args.append("--only-python")
    if only_cuda:
        args.append("--only-cuda")
    for t in targets:
        args.extend(["-t", t])
    # Add LLM options only if explicitly enabled (costs API credits)
    if llm_analysis:
        args.append("--llm-analysis")
    if force_llm:
        args.append("--force-llm")
    if apply_patches:
        args.append("--apply-llm-patches")
    if rebenchmark_llm_patches:
        args.append("--rebenchmark-llm-patches")
    if llm_explain:
        args.append("--llm-explain")

    run_dir = _resolve_run_dir(artifacts_dir, str(run_id)) if run_id else None
    progress_path = _progress_path_for_run(run_dir, str(run_id)) if run_dir else None

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned_args": args,
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else None,
            "progress_path": str(progress_path) if progress_path else None,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned_args": args,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "command": " ".join([sys.executable, "-m", "cli.aisp", "bench", *args]),
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir else None,
            "progress_path": str(progress_path) if progress_path else None,
            "note": "Set dry_run=false to execute; use async=true for background execution.",
        }

    def _execute_bench_cli_only() -> Dict[str, Any]:
        return _attach_bench_artifact_paths(
            _run_bench_cli(
                args, timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
            )
        )

    def _execute_benchmarks(queue_job_id: Optional[str] = None):
        queue_label = f"bench run {' '.join(targets)}" if targets else "bench run"
        queue_payload = {"targets": targets, "profile": profile, "run_id": run_id}
        base_result = _run_with_queue(
            "run_benchmarks",
            _execute_bench_cli_only,
            queue_label=queue_label,
            queue_payload=queue_payload,
            job_id=queue_job_id,
        )
        return _maybe_run_post_benchmark_steps(
            base_result,
            auto_analyze=auto_analyze,
            auto_report=auto_report,
            report_format=report_format,
        )

    result = _execute_benchmarks(queue_job_id=queue_job_id)
    # Add suggested next steps to help users continue their workflow
    result["suggested_next_steps"] = _benchmark_next_steps(result)
    result["run_id"] = run_id
    if run_dir:
        result["run_dir"] = str(run_dir)
    if progress_path:
        result["progress_path"] = str(progress_path)
    return attach_context_if_requested(result, include_context, context_level)


def _benchmark_next_steps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable next steps after benchmark completion."""
    steps = []
    returncode = result.get("returncode", 0)

    if returncode == 0:
        # Success path
        steps.append(
            {
                "tool": "benchmark_triage",
                "reason": "Analyze benchmark results and get optimization recommendations",
                "priority": "high",
            }
        )
        steps.append(
            {
                "tool": "benchmark_report",
                "reason": "Generate shareable PDF/HTML report",
                "params": {"format": "html"},
            }
        )
        steps.append(
            {
                "tool": "benchmark_compare_runs",
                "reason": "Compare with previous baseline if available",
                "note": "Requires baseline benchmark_test_results.json",
            }
        )
        steps.append(
            {"tool": "analyze_pareto", "reason": "Find optimal throughput/latency/memory tradeoffs"}
        )
    else:
        # Failure path
        steps.append(
            {
                "tool": "status",
                "reason": "Check system health after benchmark failure",
                "priority": "high",
            }
        )
        steps.append(
            {
                "tool": "system_dependencies",
                "reason": "Verify all dependencies are correctly installed",
            }
        )
        steps.append(
            {"tool": "analyze_bottlenecks", "reason": "Identify what might be causing issues"}
        )

    return steps


@register_tool(
    "benchmark_variants",
    "Tags: benchmark, llm, variants, optimize, profiling. "
    "Shortcut to profile and generate optimized variants via LLM: runs benchmarks with profile='minimal', "
    "forces LLM analysis, applies patches, and rebenchmarks patched variants by default. "
    "Returns the same outputs as run_benchmarks. "
    "USE when: You want to quickly profile and generate/test/benchmark new optimized variants for a target. "
    "Example: targets=['ch11:warp_specialization_multistream'].",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Benchmark targets as chapter or chapter:example; "
                        "discover via benchmark_targets or list_chapters."
                    ),
                },
                "profile": {
                    "type": "string",
                    "description": "Profiling preset: none (no profiling), minimal (basic), deep_dive (full nsys/ncu profiling), or roofline",
                    "enum": ["none", "minimal", "deep_dive", "roofline"],
                    "default": "minimal",
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Base directory for artifacts (bench creates a self-describing run dir underneath).",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for artifacts (default: <timestamp>__bench__profile-<type>__targets-<...>)",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override benchmark iterations (all targets).",
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations (all targets).",
                },
                "llm_analysis": {
                    "type": "boolean",
                    "description": "Enable LLM-powered analysis (default true for this shortcut).",
                    "default": True,
                },
                "force_llm": {
                    "type": "boolean",
                    "description": "Force LLM analysis on all benchmarks regardless of speedup (default true for this shortcut).",
                    "default": True,
                },
                "apply_patches": {
                    "type": "boolean",
                    "description": "Apply LLM-suggested patches to create new optimized variants (default true for this shortcut).",
                    "default": True,
                },
                "rebenchmark_llm_patches": {
                    "type": "boolean",
                    "description": "Re-benchmark LLM-patched variants (default true for this shortcut).",
                    "default": True,
                },
                "llm_explain": {
                    "type": "boolean",
                    "description": "Generate LLM explanations for best patches (requires rebenchmark_llm_patches=true).",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning with partial output; set 0/null for no timeout",
                    "default": 900,
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
            }
        ),
        "required": ["targets"],
    },
)
def tool_benchmark_variants(params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(params)
    merged.setdefault("profile", "minimal")
    merged.setdefault("llm_analysis", True)
    merged.setdefault("force_llm", True)
    merged.setdefault("apply_patches", True)
    merged.setdefault("rebenchmark_llm_patches", True)
    return tool_run_benchmarks(merged)


@register_tool(
    "benchmark_deep_dive_compare",
    "Tags: benchmark, deep_dive, compare, baseline, optimized, nsys, ncu, torch, one-shot, workflow. "
    "ONE-SHOT deep-dive workflow: run benchmarks with profile='deep_dive' AND return structured diffs from Nsight Systems + Nsight Compute (+ any available profiler artifacts). "
    "Also generates a side-by-side JSON report + narrative by default. "
    "Writes outputs under artifacts/runs/<run_id>/ by default and returns {run_dir, results_json, analysis_json} plus per-benchmark profiles_dir + side_by_side_report + followup_tool_calls for chaining. "
    "Selection rule: for each example, compares baseline vs the best succeeded optimization by speedup (ties break arbitrarily); surfaces the chosen optimized file in the output. "
    "Defaults: iterations=1, warmup=5 to keep deep profiling fast; override if you need more stable timing stats. "
    "USE when: You want the common chain 'bench run → deep_dive profile → compare nsys+ncu' in one tool call. "
    "Example: targets=['ch10:atomic_reduction'], output_dir='artifacts/runs'. "
    "Follow-ups: you can re-run the comparisons later by calling profile_compare / compare_nsys / compare_ncu with the returned profiles_dir.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Benchmark targets to run (chapter or chapter:example). Prefer a single example pair for clean diffs. "
                        "Discover targets via benchmark_targets or list_chapters. Provide either targets or path."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Benchmark file path (baseline_*.py or optimized_*.py). "
                        "Resolves to a single target. Provide either path or targets."
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs).",
                    "default": "artifacts/runs",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for artifacts (default: <timestamp>__deep-dive__profile-deep_dive__targets-<...>)",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override benchmark iterations (default 1 for profiling).",
                    "default": 1,
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations (default 5).",
                    "default": 5,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime for the full run+analysis; set 0/null for no timeout.",
                    "default": 0,
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
            }
        ),
        "required": [],
    },
)
def tool_benchmark_deep_dive_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run deep_dive benchmarks and produce LLM-friendly baseline-vs-optimized profiler diffs."""
    import shutil

    from core import profile_insights

    include_context, context_level = extract_context_opts(params)
    targets = params.get("targets") or []
    path = params.get("path")
    output_dir = params.get("output_dir") or "artifacts/runs"
    run_id_param = params.get("run_id")
    run_id = run_id_param.strip() if isinstance(run_id_param, str) else run_id_param
    if not run_id:
        from core.benchmark.artifact_manager import build_bench_run_label

        base_dir = Path(output_dir)
        if not base_dir.is_absolute():
            base_dir = (CODE_ROOT / base_dir).resolve()
        run_label = build_bench_run_label(targets, "deep_dive")
        run_id = _default_run_id("deep-dive", run_label, base_dir)
    iterations = params.get("iterations", 1)
    warmup = params.get("warmup", 5)
    raw_validity_profile = params.get("validity_profile", "strict")
    validity_profile = normalize_param("validity_profile", raw_validity_profile, "strict")
    if validity_profile not in {"strict", "portable"}:
        return make_error(
            f"Invalid benchmark validity profile '{raw_validity_profile}' for validity_profile. "
            "Valid options: strict, portable.",
            include_context,
            context_level,
        )
    if _is_test_mode() and "validity_profile" not in params:
        validity_profile = "portable"
    allow_portable_expectations_update = bool(
        params.get("allow_portable_expectations_update", False)
    )
    run_async = bool(params.get("async", False))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    if _is_test_mode() and (timeout_seconds is None or timeout_seconds <= 0):
        timeout_seconds = 600
    run_dir = _resolve_run_dir(output_dir, str(run_id))
    progress_path = _progress_path_for_run(run_dir, str(run_id))

    if path is not None and not isinstance(path, str):
        return make_error("path must be a string.", include_context, context_level)
    if path and targets:
        return make_error(
            "Provide either path or targets, not both.", include_context, context_level
        )
    if path:
        path_value = path.strip()
        if not path_value:
            return make_error("path cannot be empty.", include_context, context_level)
        if path_value.startswith("@"):
            path_value = path_value[1:]
        resolved_target, err = _resolve_benchmark_target_from_path(path_value)
        if err:
            return make_error(err, include_context, context_level)
        targets = [resolved_target]
    if not targets:
        return make_error("targets or path is required", include_context, context_level)

    # Return async ticket immediately; run bench + deep-dive analysis in background.
    if run_async:
        sync_params = dict(params)
        sync_params["async"] = False
        if run_id and not sync_params.get("run_id"):
            sync_params["run_id"] = str(run_id)
        if output_dir and not sync_params.get("output_dir"):
            sync_params["output_dir"] = output_dir
        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "progress_path": str(progress_path) if progress_path else None,
        }
        queued = _queue_job(
            "benchmark_deep_dive_compare",
            lambda: tool_benchmark_deep_dive_compare(sync_params),
            params,
            run_metadata=run_metadata,
        )
        queued["output_dir"] = output_dir
        queued["run_id"] = run_id
        queued["targets"] = targets
        queued["note"] = (
            "Background deep-dive accepted; poll with job_status using job_id. "
            "Bench execution and profile analysis run inside the queued job."
        )
        return queued

    def _run_and_analyze() -> Dict[str, Any]:
        test_mode = _is_test_mode()
        profile_mode = "none" if test_mode else "deep_dive"
        bench_iterations = iterations
        bench_warmup = warmup
        bench_timeout = timeout_seconds
        if test_mode:
            bench_iterations = min(bench_iterations, 1)
            bench_warmup = min(bench_warmup, 1)
            if bench_timeout is None or bench_timeout <= 0 or bench_timeout > 300:
                bench_timeout = 300
        # Run bench with deep_dive profiling into output_dir/<timestamp>/...
        bench_params = {
            "targets": targets,
            "profile": profile_mode,
            "artifacts_dir": output_dir,
            "run_id": run_id,
            "iterations": bench_iterations,
            "warmup": bench_warmup,
            # Explicitly disable LLM analysis; caller can run it separately if desired.
            "llm_analysis": False,
            "apply_patches": False,
            "timeout_seconds": bench_timeout,
            "only_python": bool(test_mode),
        }
        effective_validity_profile = validity_profile
        if test_mode and "validity_profile" not in params:
            effective_validity_profile = "portable"
        bench_params["validity_profile"] = effective_validity_profile
        if allow_portable_expectations_update:
            bench_params["allow_portable_expectations_update"] = True

        bench_result = tool_run_benchmarks(bench_params)
        bench_result = _attach_bench_artifact_paths(bench_result)
        if not isinstance(bench_result, dict):
            return {
                "error": "bench run failed to return a result payload",
                "bench_result": bench_result,
                "success": False,
            }

        results_json = bench_result.get("results_json")
        run_dir = bench_result.get("run_dir")

        if bench_result.get("returncode", 0) != 0:
            return {
                "error": "bench run failed",
                "bench_result": {
                    k: v for k, v in bench_result.items() if k not in {"stdout", "stderr"}
                },
                "results_json": results_json,
                "run_dir": run_dir,
                "success": False,
            }

        if not results_json or not run_dir:
            return {
                "error": "bench run succeeded but results_json/run_dir could not be discovered from output",
                "bench_result": {
                    k: v for k, v in bench_result.items() if k not in {"stdout", "stderr"}
                },
                "success": False,
            }

        results_path = Path(str(results_json))
        run_dir_path = Path(str(run_dir))
        analysis_path = run_dir_path / "reports" / "deep_dive_compare.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_progress = ProgressRecorder(
            run_id=str(run_id),
            progress_path=run_dir_path / "progress" / "run_progress.json",
        )
        _emit_progress_safe(
            analysis_progress,
            ProgressEvent(
                phase="analysis",
                phase_index=1,
                total_phases=2,
                step="deep_dive_compare",
                step_detail="analysis start",
                percent_complete=0.0,
            ),
        )

        raw = json.loads(results_path.read_text())
        chapters = raw.get("results", []) if isinstance(raw, dict) else []

        benchmark_analyses: List[Dict[str, Any]] = []
        for chapter_entry in chapters:
            if not isinstance(chapter_entry, dict):
                continue
            chapter = chapter_entry.get("chapter", "")
            for bench in chapter_entry.get("benchmarks", []) or []:
                if not isinstance(bench, dict):
                    continue
                example = bench.get("example", "")
                optimizations = bench.get("optimizations", []) or []

                # Select best succeeded optimization by speedup.
                succeeded = [
                    o
                    for o in optimizations
                    if isinstance(o, dict) and o.get("status") == "succeeded"
                ]
                if not succeeded:
                    benchmark_analyses.append(
                        {
                            "chapter": chapter,
                            "example": example,
                            "status": "no_succeeded_optimized_variant",
                            "note": "No succeeded optimizations; no baseline-vs-optimized profile diff available.",
                        }
                    )
                    continue

                def _speedup(o: Dict[str, Any]) -> float:
                    try:
                        return float(o.get("speedup", 0) or 0)
                    except Exception:
                        return 0.0

                best_opt = max(succeeded, key=_speedup)

                baseline_paths = {
                    "nsys": bench.get("baseline_nsys_rep"),
                    "ncu": bench.get("baseline_ncu_rep"),
                    "torch": bench.get("baseline_torch_trace"),
                }
                optimized_paths = {
                    "nsys": best_opt.get("optimized_nsys_rep"),
                    "ncu": best_opt.get("optimized_ncu_rep"),
                    "torch": best_opt.get("optimized_torch_trace"),
                }

                def _copy_profile(
                    rel_path: Optional[str],
                    dst_dir: Path,
                    dest_stem: Optional[str] = None,
                ) -> Optional[str]:
                    if not rel_path:
                        return None
                    src = Path(rel_path)
                    if not src.is_absolute():
                        src = CODE_ROOT / src
                    if not src.exists():
                        return None
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    if dest_stem:
                        suffix = "".join(src.suffixes)
                        dst = dst_dir / f"{dest_stem}{suffix}"
                    else:
                        dst = dst_dir / src.name
                    shutil.copy2(src, dst)
                    return str(dst)

                from core.benchmark.artifact_manager import slugify

                safe_chapter = slugify(str(chapter)) if chapter else "unknown_chapter"
                safe_example = slugify(str(example)) if example else "unknown_example"
                technique_label = best_opt.get("technique") or "default"
                safe_pair = slugify(str(technique_label))
                profiles_dir = (
                    run_dir_path
                    / "profiles"
                    / "bench"
                    / safe_chapter
                    / safe_example
                    / f"pair__{safe_pair}"
                )

                copied = {
                    "baseline_nsys_rep": _copy_profile(
                        baseline_paths["nsys"], profiles_dir, dest_stem=f"{safe_example}__baseline"
                    ),
                    "optimized_nsys_rep": _copy_profile(
                        optimized_paths["nsys"],
                        profiles_dir,
                        dest_stem=f"{safe_example}__optimized",
                    ),
                    "baseline_ncu_rep": _copy_profile(
                        baseline_paths["ncu"], profiles_dir, dest_stem=f"{safe_example}__baseline"
                    ),
                    "optimized_ncu_rep": _copy_profile(
                        optimized_paths["ncu"], profiles_dir, dest_stem=f"{safe_example}__optimized"
                    ),
                    "baseline_torch_trace": _copy_profile(
                        baseline_paths["torch"], profiles_dir, dest_stem=f"{safe_example}__baseline"
                    ),
                    "optimized_torch_trace": _copy_profile(
                        optimized_paths["torch"],
                        profiles_dir,
                        dest_stem=f"{safe_example}__optimized",
                    ),
                }

                # Run comparisons on the per-benchmark profiles_dir (contains only this pair).
                nsys_comparison = (
                    profile_insights.compare_nsys_files(profiles_dir)
                    if profiles_dir.exists()
                    else None
                )
                ncu_comparison = (
                    profile_insights.compare_ncu_files(profiles_dir)
                    if profiles_dir.exists()
                    else None
                )
                profile_compare = (
                    profile_insights.generate_flamegraph_comparison(profiles_dir)
                    if profiles_dir.exists()
                    else None
                )
                side_by_side_report = None
                side_by_side_error = None
                if profiles_dir.exists():
                    side_by_side_report = profile_insights.generate_side_by_side_report(
                        profiles_dir,
                        ncu_comparison=ncu_comparison,
                    )
                    if not side_by_side_report.get("success"):
                        side_by_side_error = side_by_side_report.get("error", "side_by_side_failed")

                followup_tool_calls = [
                    {"tool": "profile_compare", "params": {"profiles_dir": str(profiles_dir)}},
                    {"tool": "compare_nsys", "params": {"profiles_dir": str(profiles_dir)}},
                    {"tool": "compare_ncu", "params": {"profiles_dir": str(profiles_dir)}},
                ]

                benchmark_analyses.append(
                    {
                        "chapter": chapter,
                        "example": example,
                        "baseline_time_ms": bench.get("baseline_time_ms"),
                        "optimized_time_ms": best_opt.get("time_ms"),
                        "speedup": best_opt.get("speedup"),
                        "selected_optimized": {
                            "file": best_opt.get("file"),
                            "technique": best_opt.get("technique"),
                        },
                        "profiles_dir": str(profiles_dir),
                        "copied_profiles": copied,
                        "nsys_comparison": nsys_comparison,
                        "ncu_comparison": ncu_comparison,
                        "profile_compare": profile_compare,
                        "side_by_side_report": side_by_side_report,
                        "side_by_side_error": side_by_side_error,
                        "followup_tool_calls": followup_tool_calls,
                    }
                )

        analysis = {
            "run_dir": str(run_dir_path),
            "results_json": str(results_path),
            "targets": targets,
            "profile": profile_mode,
            "benchmarks": benchmark_analyses,
        }
        analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
        _emit_progress_safe(
            analysis_progress,
            ProgressEvent(
                phase="analysis",
                phase_index=2,
                total_phases=2,
                step="deep_dive_compare",
                step_detail="analysis complete",
                percent_complete=100.0,
            ),
        )

        return {
            "run_dir": str(run_dir_path),
            "results_json": str(results_path),
            "analysis_json": str(analysis_path),
            "benchmarks": benchmark_analyses,
            "run_id": run_id,
            "progress_path": str(run_dir_path / "progress" / "run_progress.json"),
            "success": True,
        }

    result = _run_and_analyze()
    result["run_id"] = run_id
    if progress_path:
        result["progress_path"] = str(progress_path)
    return attach_context_if_requested(result, include_context, context_level)


def _parse_speedup_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _resolve_profile_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (CODE_ROOT / path).resolve()
    return path


def _extract_profile_metrics(
    ncu_path: Optional[Path],
    nsys_path: Optional[Path],
) -> Tuple[Dict[str, float], List[str]]:
    metrics: Dict[str, float] = {}
    errors: List[str] = []
    try:
        from core.profiling.metrics_extractor import extract_ncu_metrics, extract_nsys_metrics
    except Exception as exc:
        return metrics, [f"metrics_extractor_unavailable: {exc}"]

    if nsys_path and nsys_path.exists():
        try:
            nsys_metrics = extract_nsys_metrics(nsys_path)
            if hasattr(nsys_metrics, "to_dict"):
                metrics.update(nsys_metrics.to_dict())
        except Exception as exc:
            errors.append(f"nsys_metrics_error: {exc}")
    elif nsys_path:
        errors.append(f"nsys_profile_missing: {nsys_path}")

    if ncu_path and ncu_path.exists():
        try:
            ncu_metrics = extract_ncu_metrics(ncu_path)
            if hasattr(ncu_metrics, "to_dict"):
                metrics.update(ncu_metrics.to_dict())
        except Exception as exc:
            errors.append(f"ncu_metrics_error: {exc}")
    elif ncu_path:
        errors.append(f"ncu_profile_missing: {ncu_path}")

    return metrics, errors


def _summarize_variant_profiles(
    bench_entry: Dict[str, Any],
    *,
    max_variants: int = 3,
) -> Dict[str, Any]:
    max_variants = max(1, min(int(max_variants), 3))
    baseline_time = float(bench_entry.get("baseline_time_ms") or 0.0)
    baseline_nsys = _resolve_profile_path(bench_entry.get("baseline_nsys_rep"))
    baseline_ncu = _resolve_profile_path(bench_entry.get("baseline_ncu_rep"))
    baseline_metrics, baseline_errors = _extract_profile_metrics(baseline_ncu, baseline_nsys)

    baseline_entry = {
        "name": "baseline",
        "time_ms": baseline_time,
        "speedup": 1.0,
        "profiles": {
            "nsys": str(baseline_nsys) if baseline_nsys else None,
            "ncu": str(baseline_ncu) if baseline_ncu else None,
        },
        "metrics": baseline_metrics,
        "metric_errors": baseline_errors,
    }

    variants: List[Dict[str, Any]] = []
    patch_entries = bench_entry.get("llm_patches") or []
    for patch in patch_entries:
        if not isinstance(patch, dict):
            continue
        rebench = patch.get("rebenchmark_result") or {}
        if not isinstance(rebench, dict) or not rebench.get("success"):
            continue
        time_ms = rebench.get("median_ms") or rebench.get("time_ms")
        if time_ms is None:
            continue
        actual_speedup = _parse_speedup_value(patch.get("actual_speedup"))
        if actual_speedup is None and baseline_time > 0:
            try:
                actual_speedup = float(baseline_time) / float(time_ms)
            except Exception:
                actual_speedup = None
        expected_speedup = _parse_speedup_value(patch.get("expected_speedup"))

        nsys_profile = _resolve_profile_path(rebench.get("nsys_profile"))
        ncu_profile = _resolve_profile_path(rebench.get("ncu_profile"))
        metrics, metric_errors = _extract_profile_metrics(ncu_profile, nsys_profile)

        variants.append(
            {
                "name": patch.get("variant_name")
                or Path(str(rebench.get("patched_file") or "")).stem,
                "description": patch.get("description"),
                "expected_speedup": expected_speedup,
                "actual_speedup": actual_speedup,
                "time_ms": float(time_ms),
                "profiles": {
                    "nsys": str(nsys_profile) if nsys_profile else None,
                    "ncu": str(ncu_profile) if ncu_profile else None,
                },
                "metrics": metrics,
                "metric_errors": metric_errors,
                "patched_file": rebench.get("patched_file"),
            }
        )

    variants.sort(key=lambda v: v.get("actual_speedup") or 0.0, reverse=True)
    selected_variants = variants[:max_variants]
    best_variant = selected_variants[0] if selected_variants else None

    return {
        "baseline": baseline_entry,
        "variants": selected_variants,
        "variant_count": len(variants),
        "best_variant": best_variant,
    }


def _summarize_utilization_deltas(
    baseline_metrics: Dict[str, float],
    variant_metrics: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    keys = [
        "ncu_sm_throughput_pct",
        "ncu_dram_throughput_pct",
        "ncu_l2_throughput_pct",
        "ncu_occupancy_pct",
        "ncu_kernel_time_ms",
        "nsys_total_gpu_time_ms",
    ]
    deltas: Dict[str, Dict[str, float]] = {}
    for key in keys:
        base_val = baseline_metrics.get(key)
        var_val = variant_metrics.get(key)
        if base_val is None or var_val is None:
            continue
        delta = var_val - base_val
        delta_pct = (delta / base_val * 100.0) if base_val else 0.0
        deltas[key] = {
            "baseline": float(base_val),
            "variant": float(var_val),
            "delta": delta,
            "delta_pct": delta_pct,
        }
    return deltas


def _should_run_deep_dive(
    summary: Dict[str, Any],
    *,
    mode: str,
    speedup_threshold: float,
    require_nsys: bool,
    require_ncu: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    normalized = (mode or "auto").strip().lower()
    if normalized == "always":
        return True, ["forced"]
    if normalized == "never":
        return False, ["disabled"]

    best_variant = summary.get("best_variant")
    if not best_variant:
        reasons.append("no_variants")
    else:
        best_speedup = _parse_speedup_value(best_variant.get("actual_speedup")) or 0.0
        if speedup_threshold and best_speedup < speedup_threshold:
            reasons.append(f"speedup_below_{speedup_threshold:.2f}x")

        if require_nsys:
            if not summary.get("baseline", {}).get("profiles", {}).get("nsys"):
                reasons.append("baseline_missing_nsys")
            if not best_variant.get("profiles", {}).get("nsys"):
                reasons.append("variant_missing_nsys")
        if require_ncu:
            if not summary.get("baseline", {}).get("profiles", {}).get("ncu"):
                reasons.append("baseline_missing_ncu")
            if not best_variant.get("profiles", {}).get("ncu"):
                reasons.append("variant_missing_ncu")

    return bool(reasons), reasons


def _copy_baseline_benchmark(
    baseline_path: Path,
    tag: str,
    *,
    copy_cu: bool = True,
) -> Dict[str, Any]:
    import shutil
    from core.benchmark.artifact_manager import slugify

    if baseline_path.suffix != ".py" or not baseline_path.name.startswith("baseline_"):
        raise ValueError("baseline path must be a baseline_*.py wrapper")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline path not found: {baseline_path}")

    chapter_dir = baseline_path.parent
    example_name = baseline_path.stem.replace("baseline_", "")
    tag_slug = slugify(tag or "mcp_copy")
    base_candidate = f"{example_name}_{tag_slug}" if tag_slug else f"{example_name}_copy"

    candidate = base_candidate
    counter = 2
    while (chapter_dir / f"baseline_{candidate}.py").exists() or (
        chapter_dir / f"baseline_{candidate}.cu"
    ).exists():
        candidate = f"{base_candidate}_{counter}"
        counter += 1

    new_example_name = candidate
    new_baseline_py = chapter_dir / f"baseline_{new_example_name}.py"
    shutil.copy2(baseline_path, new_baseline_py)

    new_cu_path = None
    original_cu = chapter_dir / f"baseline_{example_name}.cu"
    if copy_cu and original_cu.exists():
        new_cu_path = chapter_dir / f"baseline_{new_example_name}.cu"
        shutil.copy2(original_cu, new_cu_path)

    content = new_baseline_py.read_text(encoding="utf-8")
    new_binary_name = f"baseline_{new_example_name}"
    binary_pattern = re.compile(r"(binary_name\s*=\s*)(['\"])([^'\"]+)(['\"])")
    content, replaced = binary_pattern.subn(rf"\1\2{new_binary_name}\4", content, count=1)

    friendly_pattern = re.compile(r"(friendly_name\s*=\s*)(['\"])([^'\"]+)(['\"])")
    if friendly_pattern.search(content):

        def _friendly_repl(match: re.Match) -> str:
            label = match.group(3)
            if "MCP Copy" in label:
                return match.group(0)
            return f"{match.group(1)}{match.group(2)}{label} (MCP Copy){match.group(4)}"

        content = friendly_pattern.sub(_friendly_repl, content, count=1)

    if replaced:
        new_baseline_py.write_text(content, encoding="utf-8")

    return {
        "chapter_dir": str(chapter_dir),
        "example_name": example_name,
        "new_example_name": new_example_name,
        "baseline_py": str(new_baseline_py),
        "baseline_cu": str(new_cu_path) if new_cu_path else None,
        "binary_name": new_binary_name if replaced else None,
    }


def _evaluate_cu_int_expression(expr: str, constants: Dict[str, int]) -> Optional[int]:
    import ast

    cleaned = expr.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned)
    cleaned = cleaned.split("//", 1)[0].strip()
    cleaned = re.sub(r"sizeof\s*\(\s*float\s*\)", "4", cleaned)
    cleaned = re.sub(r"sizeof\s*\(\s*double\s*\)", "8", cleaned)
    cleaned = re.sub(r"sizeof\s*\(\s*(?:half|__half)\s*\)", "2", cleaned)
    cleaned = re.sub(r"sizeof\s*\(\s*(?:__nv_bfloat16|nv_bfloat16)\s*\)", "2", cleaned)
    cleaned = re.sub(r"([0-9]+)(?:ULL|UL|LL|U|L)", r"\1", cleaned)

    try:
        node = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return int(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = _eval(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.Name):
            if n.id in constants:
                return int(constants[n.id])
            raise ValueError(f"unknown constant {n.id}")
        if isinstance(n, ast.BinOp) and isinstance(
            n.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.LShift, ast.RShift, ast.Mod),
        ):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, (ast.Div, ast.FloorDiv)):
                return left // right
            if isinstance(n.op, ast.LShift):
                return left << right
            if isinstance(n.op, ast.RShift):
                return left >> right
            if isinstance(n.op, ast.Mod):
                return left % right
        raise ValueError("unsupported expression")

    try:
        return _eval(node)
    except Exception:
        return None


def _extract_workload_params_from_cu(cu_path: Path) -> Dict[str, int]:
    text = cu_path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = [line.split("//", 1)[0] for line in text.splitlines()]

    const_pattern = re.compile(
        r"\b(?:static\s+)?(?:constexpr|const)\s+(?:unsigned\s+)?"
        r"(?:long\s+long|long|int|size_t|std::size_t|uint32_t|uint64_t|int32_t|int64_t)"
        r"\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);"
    )

    constants: Dict[str, int] = {}
    for line in lines:
        match = const_pattern.search(line)
        if not match:
            continue
        name = match.group(1)
        expr = match.group(2)
        value = _evaluate_cu_int_expression(expr, constants)
        if value is None:
            continue
        constants[name] = int(value)

    if not constants:
        return {}

    selected: Dict[str, int] = {}
    selector = re.compile(
        r"(batch|batches|seq|tile|block|thread|warp|element|size|dim|head|hidden|"
        r"iter|num|m|n|k)",
        re.I,
    )
    for name, value in constants.items():
        if selector.search(name):
            selected[name] = int(value)

    return selected or constants


def _infer_dtype_from_cu(cu_path: Path) -> str:
    text = cu_path.read_text(encoding="utf-8")
    if re.search(r"\\b(__nv_bfloat16|nv_bfloat16)\\b", text):
        return "bfloat16"
    if re.search(r"\\b(__half|half)\\b", text):
        return "float16"
    if re.search(r"\\bdouble\\b", text):
        return "float64"
    return "float32"


def _camelize_identifier(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    return "".join(part.capitalize() for part in parts if part)


def _generate_baseline_wrapper_from_cu(cu_path: Path) -> Path:
    chapter_dir = cu_path.parent
    example_name = cu_path.stem.replace("baseline_", "")
    wrapper_path = chapter_dir / f"baseline_{example_name}.py"

    workload_params = _extract_workload_params_from_cu(cu_path)
    if not workload_params:
        raise ValueError(
            "Unable to infer workload parameters from CUDA source. "
            "Create a baseline_*.py wrapper with explicit workload_params."
        )
    workload_params["dtype"] = _infer_dtype_from_cu(cu_path)

    class_name = f"Baseline{_camelize_identifier(example_name)}Benchmark"
    friendly_title = example_name.replace("_", " ").strip().title()
    binary_name = f"baseline_{example_name}"
    params_lines = ",\n                ".join(
        f"{key!r}: {value!r}" for key, value in sorted(workload_params.items())
    )
    params_block = "{\n                " + params_lines + "\n            }"

    wrapper = f'''"""Auto-generated Python harness wrapper for {cu_path.name}."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class {class_name}(CudaBinaryBenchmark):
    """Auto-generated wrapper for {cu_path.name}."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="{binary_name}",
            friendly_name="Baseline {friendly_title} (Auto)",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={params_block},
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return {class_name}()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
'''
    wrapper_path.write_text(wrapper, encoding="utf-8")
    return wrapper_path


def _resolve_baseline_wrapper_path(path: Path) -> Path:
    if path.suffix == ".py":
        if not path.name.startswith("baseline_"):
            raise ValueError("path must be a baseline_*.py wrapper (or a matching baseline_*.cu).")
        return path
    if path.suffix == ".cu":
        if not path.name.startswith("baseline_"):
            raise ValueError("path must be a baseline_*.cu file (or a baseline_*.py wrapper).")
        wrapper = path.with_suffix(".py")
        if wrapper.exists():
            return wrapper
        return _generate_baseline_wrapper_from_cu(path)
    raise ValueError("path must be a baseline_*.py wrapper or baseline_*.cu file.")


@register_tool(
    "benchmark_explore",
    "Tags: benchmark, variants, llm, profiling, compare, workflow. "
    "Copy a baseline_*.py (or baseline_*.cu; auto-generates wrapper if missing), run minimal profiling with LLM patch variants, "
    "compare resource utilization across variants, and optionally run deep_dive profiling when minimal "
    "results are inconclusive. "
    "USE when: You want an end-to-end baseline→variants→compare workflow for a specific benchmark file. "
    "NOT FOR: General recommendations (use recommend) or pure profiling (use profile_*). "
    "Outputs progress via run_progress.json; poll with job_status if async.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "path": {
                    "type": "string",
                    "description": (
                        "Baseline wrapper path (baseline_*.py) or baseline_*.cu. "
                        "If the wrapper is missing, the tool will attempt to auto-generate one. "
                        "The tool then copies the wrapper and runs variants on the copy."
                    ),
                },
                "copy_tag": {
                    "type": "string",
                    "description": "Suffix tag for copied baseline files (default: mcp_copy).",
                    "default": "mcp_copy",
                },
                "copy_cu": {
                    "type": "boolean",
                    "description": "Copy matching baseline_*.cu file if present.",
                    "default": True,
                },
                "max_variants": {
                    "type": "integer",
                    "description": "Max number of LLM variants to report (clamped to 1-3).",
                    "default": 3,
                },
                "deep_dive": {
                    "type": "string",
                    "description": "Deep-dive mode: auto (default), always, or never.",
                    "enum": ["auto", "always", "never"],
                    "default": "auto",
                },
                "deep_dive_speedup_threshold": {
                    "type": "number",
                    "description": "Speedup threshold below which deep_dive is triggered in auto mode.",
                    "default": 1.05,
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: ./artifacts/runs).",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for artifacts (default: <timestamp>__explore__<label>).",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override benchmark iterations for minimal profiling run.",
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations for minimal profiling run.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime for minimal profiling run (0/null = no timeout).",
                    "default": 0,
                },
                "deep_dive_iterations": {
                    "type": "integer",
                    "description": "Override iterations for deep_dive run (default: 1).",
                    "default": 1,
                },
                "deep_dive_warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations for deep_dive run (default: 5).",
                    "default": 5,
                },
                "deep_dive_timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime for deep_dive run (0/null = no timeout).",
                    "default": 0,
                },
                "update_expectations": {
                    "type": "boolean",
                    "description": "Force-write observed metrics into expectation files (recommended).",
                    "default": True,
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status.",
                    "default": False,
                },
            }
        ),
        "required": ["path"],
    },
)
def tool_benchmark_explore(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    path = params.get("path")
    if not path or not isinstance(path, str):
        return make_error("path is required and must be a string.", include_context, context_level)

    artifacts_dir = params.get("artifacts_dir") or "artifacts/runs"
    base_dir = Path(artifacts_dir)
    if not base_dir.is_absolute():
        base_dir = (CODE_ROOT / base_dir).resolve()

    run_id_param = params.get("run_id")
    run_id = run_id_param.strip() if isinstance(run_id_param, str) else run_id_param
    if not run_id:
        label = f"explore-{Path(path).stem}"
        run_id = _default_run_id("explore", label, base_dir)
    raw_validity_profile = params.get("validity_profile", "strict")
    validity_profile = normalize_param("validity_profile", raw_validity_profile, "strict")
    if validity_profile not in {"strict", "portable"}:
        return make_error(
            f"Invalid benchmark validity profile '{raw_validity_profile}' for validity_profile. "
            "Valid options: strict, portable.",
            include_context,
            context_level,
        )
    if _is_test_mode() and "validity_profile" not in params:
        validity_profile = "portable"
    allow_portable_expectations_update = bool(
        params.get("allow_portable_expectations_update", False)
    )

    progress_path = _progress_path_in_dir(base_dir, str(run_id))
    progress_recorder = ProgressRecorder(run_id=str(run_id), progress_path=progress_path)

    def _emit(step: str, detail: str, percent: float) -> None:
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="explore",
                phase_index=1,
                total_phases=1,
                step=step,
                step_detail=detail,
                percent_complete=percent,
            ),
        )

    def _run() -> Dict[str, Any]:
        from core.discovery import chapter_slug, get_bench_roots
        from core.harness.run_benchmarks import check_nsys_available, check_ncu_available

        test_mode = _is_test_mode()
        effective_validity_profile = validity_profile
        if test_mode and "validity_profile" not in params:
            effective_validity_profile = "portable"

        _emit("copy", "starting copy", 2.0)
        baseline_path = Path(path)
        if not baseline_path.is_absolute():
            baseline_path = (CODE_ROOT / baseline_path).resolve()
        try:
            baseline_path = _resolve_baseline_wrapper_path(baseline_path)
        except Exception as exc:
            return {"success": False, "error": f"path_invalid: {exc}"}
        try:
            copy_info = _copy_baseline_benchmark(
                baseline_path,
                params.get("copy_tag") or "mcp_copy",
                copy_cu=bool(params.get("copy_cu", True)),
            )
        except Exception as exc:
            return {"success": False, "error": f"copy_failed: {exc}"}

        chapter_dir = Path(copy_info["chapter_dir"])
        new_example = copy_info["new_example_name"]
        roots = get_bench_roots(repo_root=CODE_ROOT)
        bench_root = roots[0] if roots else CODE_ROOT
        target = f"{chapter_slug(chapter_dir, CODE_ROOT, bench_root=bench_root)}:{new_example}"

        _emit("minimal", "running minimal profiling + LLM patches", 10.0)
        minimal_run_id = f"{run_id}__minimal"
        minimal_params = {
            "targets": [target],
            "profile": "minimal",
            "artifacts_dir": str(base_dir),
            "run_id": minimal_run_id,
            "iterations": params.get("iterations"),
            "warmup": params.get("warmup"),
            "timeout_seconds": params.get("timeout_seconds"),
            "llm_analysis": True,
            "force_llm": True,
            "apply_patches": True,
            "rebenchmark_llm_patches": True,
            "llm_explain": False,
            "validity_profile": effective_validity_profile,
            "update_expectations": bool(params.get("update_expectations", True)),
        }
        if allow_portable_expectations_update:
            minimal_params["allow_portable_expectations_update"] = True
        minimal_result = tool_run_benchmarks(minimal_params)
        minimal_result = _normalize_result(minimal_result)
        if minimal_result.get("success") is False:
            minimal_result.setdefault("error", "minimal_run_failed")
            return {
                "success": False,
                "error": minimal_result.get("error", "minimal_run_failed"),
                "minimal": minimal_result,
                "copy": copy_info,
                "target": target,
                "run_id": run_id,
                "progress_path": str(progress_path),
            }

        results_json = minimal_result.get("results_json")
        if not results_json:
            return {
                "success": False,
                "error": "minimal_run_missing_results_json",
                "minimal": minimal_result,
                "copy": copy_info,
                "target": target,
                "run_id": run_id,
                "progress_path": str(progress_path),
            }

        results_path = _resolve_artifact_path(results_json)
        if not results_path or not results_path.exists():
            return {
                "success": False,
                "error": f"results_json_not_found: {results_json}",
                "minimal": minimal_result,
                "copy": copy_info,
                "target": target,
                "run_id": run_id,
                "progress_path": str(progress_path),
            }

        data = json.loads(results_path.read_text())
        bench_entry = None
        for chapter_entry in data.get("results", []) or []:
            for bench in chapter_entry.get("benchmarks", []) or []:
                if bench.get("example") == new_example:
                    bench_entry = bench
                    break
            if bench_entry:
                break

        if not bench_entry:
            return {
                "success": False,
                "error": f"example_not_found_in_results: {new_example}",
                "minimal": minimal_result,
                "copy": copy_info,
                "target": target,
                "run_id": run_id,
                "progress_path": str(progress_path),
            }

        _emit("analyze", "summarizing minimal results", 55.0)
        summary = _summarize_variant_profiles(
            bench_entry,
            max_variants=params.get("max_variants", 3),
        )

        # Compute utilization deltas vs baseline for reported variants.
        baseline_metrics = summary.get("baseline", {}).get("metrics", {}) or {}
        for variant in summary.get("variants", []):
            variant_metrics = variant.get("metrics", {}) or {}
            variant["utilization_deltas"] = _summarize_utilization_deltas(
                baseline_metrics,
                variant_metrics,
            )

        require_nsys = check_nsys_available()
        require_ncu = check_ncu_available()
        deep_dive_needed, deep_dive_reasons = _should_run_deep_dive(
            summary,
            mode=params.get("deep_dive", "auto"),
            speedup_threshold=float(params.get("deep_dive_speedup_threshold", 1.05) or 0.0),
            require_nsys=require_nsys,
            require_ncu=require_ncu,
        )

        deep_dive_result = None
        deep_dive_summary = None
        if deep_dive_needed:
            _emit("deep_dive", "running deep_dive profiling + LLM patches", 70.0)
            deep_run_id = f"{run_id}__deep_dive"
            deep_params = {
                "targets": [target],
                "profile": "deep_dive",
                "artifacts_dir": str(base_dir),
                "run_id": deep_run_id,
                "iterations": params.get("deep_dive_iterations", 1),
                "warmup": params.get("deep_dive_warmup", 5),
                "timeout_seconds": params.get("deep_dive_timeout_seconds"),
                "llm_analysis": True,
                "force_llm": True,
                "apply_patches": True,
                "rebenchmark_llm_patches": True,
                "llm_explain": False,
                "validity_profile": effective_validity_profile,
                "update_expectations": bool(params.get("update_expectations", True)),
            }
            if allow_portable_expectations_update:
                deep_params["allow_portable_expectations_update"] = True
            deep_dive_result = tool_run_benchmarks(deep_params)
            deep_dive_result = _normalize_result(deep_dive_result)
            if deep_dive_result.get("success") is False:
                deep_dive_result.setdefault("error", "deep_dive_failed")
            else:
                deep_results_json = deep_dive_result.get("results_json")
                deep_path = _resolve_artifact_path(deep_results_json) if deep_results_json else None
                if deep_path and deep_path.exists():
                    deep_data = json.loads(deep_path.read_text())
                    deep_entry = None
                    for chapter_entry in deep_data.get("results", []) or []:
                        for bench in chapter_entry.get("benchmarks", []) or []:
                            if bench.get("example") == new_example:
                                deep_entry = bench
                                break
                        if deep_entry:
                            break
                    if deep_entry:
                        deep_dive_summary = _summarize_variant_profiles(
                            deep_entry,
                            max_variants=params.get("max_variants", 3),
                        )
                        baseline_metrics = (
                            deep_dive_summary.get("baseline", {}).get("metrics", {}) or {}
                        )
                        for variant in deep_dive_summary.get("variants", []):
                            variant_metrics = variant.get("metrics", {}) or {}
                            variant["utilization_deltas"] = _summarize_utilization_deltas(
                                baseline_metrics,
                                variant_metrics,
                            )

        _emit("complete", "variant study complete", 100.0)

        return {
            "success": True,
            "run_id": run_id,
            "target": target,
            "copy": copy_info,
            "minimal": {
                "run_id": minimal_run_id,
                "results_json": str(results_path),
                "summary": summary,
            },
            "deep_dive": {
                "requested": bool(deep_dive_needed),
                "reasons": deep_dive_reasons,
                "result": deep_dive_result,
                "summary": deep_dive_summary,
            },
            "progress_path": str(progress_path),
        }

    run_async = bool(params.get("async", False))
    if run_async:
        job_id = f"benchmark_explore-{uuid.uuid4().hex[:10]}"
        run_metadata = {
            "run_id": run_id,
            "progress_path": str(progress_path),
            "target_path": path,
        }
        queued = _queue_job(
            "benchmark_explore", _run, params, run_metadata=run_metadata, job_id=job_id
        )
        queued["run_id"] = run_id
        queued["progress_path"] = str(progress_path)
        queued["note"] = "Background variant study started; poll with job_status using job_id."
        return queued

    result = _run()
    return attach_context_if_requested(result, include_context, context_level)


def _summarize_metric_deltas(metrics: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for metric in metrics:
        delta_pct = metric.get("delta_pct")
        delta = metric.get("delta")
        score = None
        if isinstance(delta_pct, (int, float)):
            score = abs(delta_pct)
        elif isinstance(delta, (int, float)):
            score = abs(delta)
        if score is None:
            continue
        ranked.append((score, metric))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:limit]]


def _summarize_ncu_comparison(
    ncu_comparison: Optional[Dict[str, Any]], limit: int = 5
) -> Dict[str, Any]:
    if not ncu_comparison:
        return {}
    if "metrics" in ncu_comparison:
        return {
            "top_metrics": _summarize_metric_deltas(
                ncu_comparison.get("metrics") or [], limit=limit
            )
        }
    kernel_rows = ncu_comparison.get("kernel_comparison") or []
    if not kernel_rows:
        return {}
    top_kernel = kernel_rows[0]
    metrics = top_kernel.get("metrics") or {}
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for name, payload in metrics.items():
        ratio = payload.get("ratio")
        delta = payload.get("delta")
        score = None
        if isinstance(ratio, (int, float)):
            score = abs(ratio - 1.0)
        elif isinstance(delta, (int, float)):
            score = abs(delta)
        if score is None:
            continue
        ranked.append((score, {"name": name, **payload}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return {
        "kernel": top_kernel.get("kernel"),
        "top_metrics": [item[1] for item in ranked[:limit]],
    }


def _extract_promoted_targets(results_json: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(results_json.read_text())
    except Exception:
        return []
    chapters = raw.get("results", []) if isinstance(raw, dict) else []
    promoted: List[Dict[str, Any]] = []
    for chapter_entry in chapters:
        if not isinstance(chapter_entry, dict):
            continue
        chapter = chapter_entry.get("chapter", "")
        for bench in chapter_entry.get("benchmarks", []) or []:
            if not isinstance(bench, dict):
                continue
            best_patch = bench.get("best_llm_patch") or {}
            promoted_file = best_patch.get("promoted_file")
            if not promoted_file:
                continue
            promoted_path = Path(str(promoted_file))
            alias = promoted_path.stem
            if alias.startswith("optimized_"):
                alias = alias.replace("optimized_", "", 1)
            promoted.append(
                {
                    "chapter": chapter,
                    "example": bench.get("example", ""),
                    "alias": alias,
                    "promoted_file": promoted_file,
                    "variant_name": best_patch.get("variant_name"),
                    "actual_speedup": best_patch.get("actual_speedup"),
                }
            )
    return promoted


@register_tool(
    "benchmark_llm_patch_loop",
    "Tags: benchmark, llm, patches, deep_dive, compare, baseline, optimized, one-shot. "
    "Run the full LLM patch loop: deep-dive profile baseline/optimized, force LLM analysis, "
    "apply patches, rebenchmark, generate explanation, promote best patch, then run a clean "
    "baseline-vs-patch deep-dive compare and summarize nsys/ncu deltas. "
    "Returns: {run_dir, results_json, promoted_targets, compare_runs, summary}. "
    "USE when: You want the full loop in one call without chaining multiple tools. "
    "Example: targets=['ch12:kernel_fusion'].",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Benchmark targets (chapter or chapter:example). Prefer a single example.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs).",
                    "default": "artifacts/runs",
                },
                "compare_output_dir": {
                    "type": "string",
                    "description": "Base directory for compare-run artifacts (default: artifacts/runs).",
                    "default": "artifacts/runs",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override iterations for the LLM patch run (default 1 for profiling).",
                    "default": 1,
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations for the LLM patch run (default 5).",
                    "default": 5,
                },
                "compare_iterations": {
                    "type": "integer",
                    "description": "Override iterations for the compare run (default 1).",
                    "default": 1,
                },
                "compare_warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations for the compare run (default 5).",
                    "default": 5,
                },
                "force_llm": {
                    "type": "boolean",
                    "description": "Force LLM analysis on all benchmarks regardless of speedup (costs API credits).",
                    "default": True,
                },
                "llm_explain": {
                    "type": "boolean",
                    "description": "Generate LLM explanations for best patches.",
                    "default": True,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime for the full loop; set 0/null for no timeout.",
                    "default": 0,
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
            }
        ),
        "required": ["targets"],
    },
)
def tool_benchmark_llm_patch_loop(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    targets = params.get("targets") or []
    if not targets:
        return make_error("targets is required", include_context, context_level)

    output_dir = params.get("output_dir") or "artifacts/runs"
    compare_output_dir = params.get("compare_output_dir") or "artifacts/runs"
    iterations = params.get("iterations", 1)
    warmup = params.get("warmup", 5)
    compare_iterations = params.get("compare_iterations", 1)
    compare_warmup = params.get("compare_warmup", 5)
    force_llm = bool(params.get("force_llm", True))
    llm_explain = bool(params.get("llm_explain", True))
    raw_validity_profile = params.get("validity_profile", "strict")
    validity_profile = normalize_param("validity_profile", raw_validity_profile, "strict")
    if validity_profile not in {"strict", "portable"}:
        return make_error(
            f"Invalid benchmark validity profile '{raw_validity_profile}' for validity_profile. "
            "Valid options: strict, portable.",
            include_context,
            context_level,
        )
    if _is_test_mode() and "validity_profile" not in params:
        validity_profile = "portable"
    allow_portable_expectations_update = bool(
        params.get("allow_portable_expectations_update", False)
    )
    run_async = bool(params.get("async", False))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

    def _run_loop() -> Dict[str, Any]:
        bench_params = {
            "targets": targets,
            "profile": "deep_dive",
            "artifacts_dir": output_dir,
            "iterations": iterations,
            "warmup": warmup,
            "llm_analysis": True,
            "force_llm": force_llm,
            "apply_patches": True,
            "rebenchmark_llm_patches": True,
            "llm_explain": llm_explain,
            "validity_profile": validity_profile,
            "timeout_seconds": timeout_seconds,
        }
        if allow_portable_expectations_update:
            bench_params["allow_portable_expectations_update"] = True

        bench_result = tool_run_benchmarks(bench_params)
        bench_result = _attach_bench_artifact_paths(bench_result)

        results_json = bench_result.get("results_json")
        run_dir = bench_result.get("run_dir")

        if bench_result.get("returncode", 0) != 0:
            return {
                "error": "llm patch run failed",
                "bench_result": {
                    k: v for k, v in bench_result.items() if k not in {"stdout", "stderr"}
                },
                "results_json": results_json,
                "run_dir": run_dir,
                "success": False,
            }

        if not results_json or not run_dir:
            return {
                "error": "bench run succeeded but results_json/run_dir could not be discovered",
                "bench_result": {
                    k: v for k, v in bench_result.items() if k not in {"stdout", "stderr"}
                },
                "success": False,
            }

        promoted_targets = _extract_promoted_targets(Path(str(results_json)))
        if not promoted_targets:
            return {
                "error": "no promoted LLM patches found; ensure rebenchmark_llm_patches succeeded",
                "run_dir": run_dir,
                "results_json": results_json,
                "success": False,
            }

        compare_runs: List[Dict[str, Any]] = []
        for promoted in promoted_targets:
            alias_target = f"{promoted.get('chapter')}:{promoted.get('alias')}"
            compare_params = {
                "targets": [alias_target],
                "output_dir": compare_output_dir,
                "iterations": compare_iterations,
                "warmup": compare_warmup,
                "validity_profile": validity_profile,
                "timeout_seconds": timeout_seconds,
            }
            if allow_portable_expectations_update:
                compare_params["allow_portable_expectations_update"] = True
            compare_result = tool_benchmark_deep_dive_compare(compare_params)
            if not isinstance(compare_result, dict):
                compare_runs.append(
                    {
                        "target": alias_target,
                        "promoted_file": promoted.get("promoted_file"),
                        "compare_result": compare_result,
                        "summary": {},
                        "error": "compare_result_missing",
                    }
                )
                continue

            summaries: Dict[str, Any] = {}
            benchmarks = compare_result.get("benchmarks") or []
            if benchmarks:
                entry = benchmarks[0]
                nsys_summary = _summarize_metric_deltas(
                    entry.get("nsys_comparison", {}).get("metrics", []) or []
                )
                ncu_summary = _summarize_ncu_comparison(entry.get("ncu_comparison"))
                summaries = {
                    "nsys_top_deltas": nsys_summary,
                    "ncu_top_deltas": ncu_summary,
                }

            compare_runs.append(
                {
                    "target": alias_target,
                    "promoted_file": promoted.get("promoted_file"),
                    "compare_result": compare_result,
                    "summary": summaries,
                }
            )

        summary = {
            "targets": targets,
            "promoted_targets": promoted_targets,
            "compare_targets": [c.get("target") for c in compare_runs],
        }

        return {
            "run_dir": run_dir,
            "results_json": results_json,
            "promoted_targets": promoted_targets,
            "compare_runs": compare_runs,
            "summary": summary,
            "success": True,
        }

    if run_async:
        queued = _queue_job("benchmark_llm_patch_loop", _run_loop, params)
        queued["targets"] = targets
        queued["note"] = "Background LLM patch loop started; poll with job_status using job_id."
        return queued

    result = _run_loop()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_context",
    "Tags: context, environment, inventory, full-dump, comprehensive. "
    "Get comprehensive system context: GPU info + software stack + hardware capabilities combined. "
    "Returns: {gpu, software, dependencies, capabilities} - all system info in one call. "
    "USE when: Need complete environment dump for analysis, sharing system state with others. "
    'Example: "Provide full context for LLM analysis" or "Dump entire system state". '
    "PREFER triage or context_summary for quick checks; this is heavier. 🕐 SLOW (2-30+ min). NOT FOR: Quick GPU health (use hw_speed). ⚡ FAST (~2s). WORKFLOW: system_dependencies → fix broken → retry.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_context(params: Dict[str, Any]) -> ContextResult:
    """Get full system context."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result: ContextResult = {"success": True, "context": get_engine().system.context()}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "system_capabilities",
    "Tags: capabilities, features, supported, compute-capability. "
    "Get hardware capabilities summary: compute capability, tensor cores, supported precisions. "
    "Returns: {compute_capability, sm_version, tensor_cores, supported_dtypes, max_shared_mem}. "
    "USE when: Checking if a feature (FP8, TF32, etc.) is supported, planning which optimizations to apply. "
    'Example: "What features does my GPU support?" or "Can I use FP8 on this hardware?". '
    "PREFER info_features for detailed capability breakdown with TMA/cluster info. ⚡ FAST (~1s). WORKFLOW: system_capabilities → check features → recommend. NOT FOR: Version info (use system_software).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_capabilities(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get hardware capabilities."""
    from core.perf_core import get_core
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().system.capabilities()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_data",
    "Tags: benchmark, data, table, pagination, filters, dashboard. "
    "Fetch benchmark results with filtering/sorting/pagination (dashboard data view). "
    "Returns: {timestamp, summary, benchmarks, pagination, filters}. "
    "USE when: Building tables or filtering benchmark results. "
    'Example: "List failed benchmarks" or "Show top speedups". '
    "NOT FOR: Comparing two runs (use benchmark_compare). ⚡ FAST (~1s).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "page": {"type": "integer", "description": "Page number (1-based)", "default": 1},
                "page_size": {"type": "integer", "description": "Page size (1-500)", "default": 50},
                "search": {"type": "string", "description": "Search substring for chapter/name"},
                "sort_field": {
                    "type": "string",
                    "description": "Sort field",
                    "enum": [
                        "name",
                        "chapter",
                        "speedup",
                        "baseline_time_ms",
                        "optimized_time_ms",
                        "status",
                    ],
                    "default": "speedup",
                },
                "sort_dir": {
                    "type": "string",
                    "description": "Sort direction",
                    "enum": ["asc", "desc"],
                    "default": "desc",
                },
                "status": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by status (comma-separated string also accepted).",
                },
                "chapter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by chapter (comma-separated string also accepted).",
                },
                "benchmark": {"type": "string", "description": "Exact benchmark name filter"},
                "optimization_goal": {
                    "type": "string",
                    "description": "Filter by optimization goal (performance/memory)",
                },
            }
        ),
    },
)
def tool_benchmark_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch benchmark data with filters."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_data(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_overview",
    "Tags: benchmark, overview, summary, dashboard. "
    "Summarize the latest benchmark results (status counts, top speedups, per-chapter stats). "
    "Returns: {summary, status_counts, top_speedups, chapter_stats}. "
    "USE when: High-level dashboard summary. "
    'Example: "Give me the latest benchmark summary". ⚡ FAST (~1s).',
    {"type": "object", "properties": with_context_params({})},
)
def tool_benchmark_overview(params: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize latest benchmark results."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_overview(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_opportunities",
    "Tags: benchmark, opportunities, next-steps, optimization, radar, evidence. "
    "Rank benchmark evidence by next optimization opportunity. "
    "Returns: {summary, opportunities: [{target, priority, opportunity_type, score, evidence, recommended_experiments, next_command, experiment_blueprints, source_delta_terms, optimization_primitives}], frontier_discovery_map with diversity_queue and experiment_blueprints, innovation_hypotheses, source_transfer_map with reusable source-mined patterns and recipient targets, compound_primitive_hypotheses with source-backed multi-primitive experiments, novelty_primitive_pair_synthesis_plan with source-backed untried primitive pairs outside the fixed compound catalog, coverage_gap_map with negative-space signal/primitive/compound leads, cross_lane_bridge_map with multi-signal experiment bridges, novelty_queue with one ranked action list, novelty_experiment_playbooks with variant ladders/metrics/guardrails, novelty_mutation_plan with one-variable mutation candidates, novelty_mutation_budget_plan with information-gain mutation selection and deferral unlocks, novelty_budget_plan with risk-adjusted expected-value selection, mitigation steps, and backlog deferral reasons, novelty_decision_frontier with quick-proof/high-upside/de-risk/Pareto/deferred-unlock lanes, novelty_falsification_plan with null hypotheses/disproof checks/claim boundaries, novelty_ablation_plan with negative controls for isolating selected claims, novelty_reproducibility_plan with repeat counts/stability metrics/variance gates, novelty_instrumentation_plan with profiler tools/preflight checks/launch environment/artifact evidence, novelty_artifact_contract_plan with required control/candidate/profile/review files and package manifests, novelty_claim_packet_plan with bounded claim sections and blocked overclaims, optional novelty_evidence_audit_plan for supplied validation queue roots, optional novelty_recovery_plan with repair/rerun actions for failed jobs and missing artifacts, optional novelty_adaptive_decision_plan with per-lead continue/recover/review/backup decisions, optional novelty_learning_plan with rerank guidance and risk updates from validation outcomes, optional novelty_next_wave_plan with ordered recover/continue/review/mutation/backup/harvest-follow-up/learning campaign actions, optional novelty_harvest_plan with approved-and-audited pattern seeds and bounded follow-up experiments, novelty_validation_plan with control/candidate/profile/review jobs, risk-specific evidence gates, falsification evidence gates, ablation evidence gates, reproducibility evidence gates, instrumentation evidence gates, artifact-contract evidence gates, claim-packet evidence gates, and optional resume state, experiment_matrix, portfolio_plan, promotion_gates, run_queue with dispatch_groups and job evidence criteria}. CLI can also mine a source-derived frontier catalog via bench opportunity-catalog, write executable runbooks via --output-run-queue-sh, --output-novelty-validation-sh, and --output-novelty-next-wave-sh, summarize evidence markers via bench opportunity-run-summary, and overlay prior queue roots with run_queue_root or novelty_queue_root. "
    "USE when: Deciding what target deserves the next deep-dive profile or optimization experiment. "
    "Accepts raw benchmark_test_results.json, transformed analyzer JSON, or tier-1 summary.json via data_file. "
    "WORKFLOW: run_benchmarks or run-tier1 → benchmark_opportunities → run the emitted next_command for the top target. "
    "PREFER benchmark_triage for pass/fail triage and benchmark_compare_runs for before/after comparisons. ⚡ FAST (~1s).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "data_file": {
                    "type": "string",
                    "description": "Path to benchmark_test_results.json, analyzer JSON, or tier-1 summary.json. Defaults to current engine benchmark data.",
                },
                "catalog_file": {
                    "type": "string",
                    "description": "Optional JSON catalog of runnable targets. Targets absent from evidence are ranked as novel_frontier_probe opportunities.",
                },
                "include_discovered_targets": {
                    "type": "boolean",
                    "description": "Also compare evidence against discovered targets and rank unmeasured targets as frontier probes. With bench_root, discovery is source-mined into a richer catalog.",
                    "default": False,
                },
                "bench_root": {
                    "type": "string",
                    "description": "Optional benchmark tree root to source-mine when include_discovered_targets is true.",
                },
                "run_queue_root": {
                    "type": "string",
                    "description": "Optional artifact root from a prior opportunity run queue. Overlays DONE, MANUAL_REVIEW_REQUIRED, APPROVED, and log diagnostic status to produce a resume plan and update promotion gates.",
                },
                "novelty_queue_root": {
                    "type": "string",
                    "description": "Optional artifact root from a prior novelty validation queue. Overlays lead-level DONE, MANUAL_REVIEW_REQUIRED, APPROVED, and log diagnostic status onto the novelty queue, budget plan, and validation resume plan.",
                },
                "top": {
                    "type": "integer",
                    "description": "Number of opportunities to return; use 0 for all",
                    "default": 10,
                },
                "min_speedup": {
                    "type": "number",
                    "description": "Speedup below this is treated as flat",
                    "default": 1.10,
                },
                "target_speedup": {
                    "type": "number",
                    "description": "Speedup below this is treated as compound-optimization headroom",
                    "default": 1.50,
                },
                "min_memory_savings_pct": {
                    "type": "number",
                    "description": "Memory-focused targets below this savings threshold are prioritized",
                    "default": 10.0,
                },
                "slow_baseline_ms": {
                    "type": "number",
                    "description": "Baseline runtime threshold that increases opportunity priority",
                    "default": 100.0,
                },
            }
        ),
    },
)
def tool_benchmark_opportunities(params: Dict[str, Any]) -> Dict[str, Any]:
    """Rank benchmark evidence by next optimization opportunity."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_opportunities(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_history",
    "Tags: benchmark, history, runs, timeline. "
    "List historical benchmark runs with summary stats. "
    "Returns: {total_runs, latest, runs: [{date, avg_speedup, max_speedup, benchmark_count, ...}]}. "
    "USE when: Building a history page or selecting runs to compare. "
    'Example: "List recent benchmark runs". ⚡ FAST (~1s).',
    {"type": "object", "properties": with_context_params({})},
)
def tool_benchmark_history(params: Dict[str, Any]) -> Dict[str, Any]:
    """List historical benchmark runs."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_history(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_trends",
    "Tags: benchmark, trends, speedup, history, dashboard. "
    "Compute performance trends over time (avg/max speedup by run). "
    "Returns: {trend_points, best_ever}. "
    "USE when: Charting benchmark trends. "
    'Example: "Show benchmark speedup trends". ⚡ FAST (~1s).',
    {"type": "object", "properties": with_context_params({})},
)
def tool_benchmark_trends(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute benchmark trends."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_trends(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_compare",
    "Tags: benchmark, compare, diff, regression, improvement. "
    "Compare two benchmark run JSON files (baseline vs candidate). "
    "Returns: {regressions, improvements, unchanged, summary}. "
    "USE when: Diffing results from two runs. "
    'Example: "Compare two benchmark result files". ⚡ FAST (~1s).',
    {
        "type": "object",
        "properties": with_context_params(
            {
                "baseline": {
                    "type": "string",
                    "description": "Path to baseline benchmark_test_results.json",
                },
                "candidate": {
                    "type": "string",
                    "description": "Path to candidate benchmark_test_results.json",
                },
                "top": {
                    "type": "integer",
                    "description": "Show top N regressions/improvements",
                    "default": 10,
                },
            }
        ),
        "required": ["baseline", "candidate"],
    },
)
def tool_benchmark_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two benchmark runs."""
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_compare(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_targets",
    "Tags: benchmarks, targets, list, chapters, examples, discovery. "
    "List benchmark targets in chapter:example format (e.g., 'ch07:flash_attention'). "
    "Returns: {targets: [{chapter, example, path}], count} or filtered by chapter. "
    "USE when: Finding exact target names to pass to run_benchmarks. "
    'Example: "List targets for ch07" or "What examples are in the attention chapter?". '
    "PREFER list_chapters to see all chapters first. ⚡ FAST (~1s). WORKFLOW: benchmark_targets → run_benchmarks. NOT FOR: Running benchmarks (use run_benchmarks).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "chapter": {
                    "type": "string",
                    "description": "Optional chapter or lab slug to filter (e.g., 'ch07', 'labs/decode')",
                },
            }
        ),
    },
)
def tool_benchmark_targets(params: Dict[str, Any]) -> Dict[str, Any]:
    """List benchmark targets."""
    include_context, context_level = extract_context_opts(params)
    args: List[str] = ["list-targets"]
    chapter = params.get("chapter")
    if chapter:
        args.extend(["--chapter", chapter])
    result = _run_bench_cli(args)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_e2e_sweep",
    "Tags: benchmark, e2e, tier1, cluster, fabric, orchestration, release-gate. "
    "Run the tier1 release-confidence suite, optional discovered full sweep, cluster common-eval, "
    "and optional fabric eval in one orchestrated flow. "
    "Returns: {success, overall_status, run_id, stages, manifest_path, summary_path, actions, preferred_progress_source}. "
    "USE when: You want one top-level end-to-end benchmark package instead of composing tier1, sweep, and cluster runs manually. "
    "Writes top-level artifacts under artifacts/e2e_runs/<run_id>/. "
    "Supports async job tickets via async=true. "
    "For live progress, prefer benchmark_e2e_status over generic job_status because it joins child progress, watcher state, recent events, and PID liveness.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_tier1": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run the canonical tier-1 suite stage",
                },
                "run_full_sweep": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run the heavier discovered full benchmark sweep",
                },
                "run_cluster": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run the cluster common-eval stage",
                },
                "run_fabric": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run the dedicated fabric-eval stage",
                },
                "cluster_preset": {
                    "type": "string",
                    "enum": [
                        "common-answer-fast",
                        "core-system",
                        "modern-llm",
                        "fabric-systems",
                        "multinode-readiness",
                    ],
                    "default": "common-answer-fast",
                    "description": "Cluster common-eval preset to run",
                },
                "hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cluster host list; defaults to localhost",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels matching hosts",
                },
                "ssh_user": {
                    "type": "string",
                    "description": "SSH user for non-local cluster runs",
                },
                "ssh_key": {
                    "type": "string",
                    "description": "SSH key path for non-local cluster runs",
                },
                "bench_root": {
                    "type": "string",
                    "description": "Benchmark discovery root (defaults to repo root)",
                },
                "profile": {
                    "type": "string",
                    "enum": ["none", "minimal", "deep_dive", "roofline"],
                    "default": "minimal",
                    "description": "Profiling preset for benchmark stages",
                },
                "suite_timeout": {
                    "type": "integer",
                    "description": "Benchmark suite timeout in seconds",
                    "default": 14400,
                },
                "full_sweep_suite_timeout": {
                    "type": "integer",
                    "description": "Aggregate suite timeout applied to each full-sweep bucket; default 0 disables the bucket-wide watchdog",
                    "default": 0,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional cluster stage timeout in seconds",
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Base directory for benchmark run artifacts",
                },
                "run_id": {"type": "string", "description": "Explicit top-level e2e sweep run id"},
                "validity_profile": {
                    "type": "string",
                    "enum": ["strict", "portable"],
                    "default": "strict",
                    "description": "Benchmark validity profile",
                },
                "single_gpu": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force single-GPU visibility for benchmark stages",
                },
                "iterations": {"type": "integer", "description": "Override benchmark iterations"},
                "warmup": {
                    "type": "integer",
                    "description": "Override benchmark warmup iterations",
                },
                "gpu_sm_clock_mhz": {
                    "type": "integer",
                    "description": "Lock the SM application clock (MHz) for benchmark stages",
                },
                "gpu_mem_clock_mhz": {
                    "type": "integer",
                    "description": "Lock the GPU memory application clock (MHz) for benchmark stages",
                },
                "accept_regressions": {
                    "type": "boolean",
                    "default": False,
                    "description": "Update expectations when improvements are detected instead of flagging regressions",
                },
                "update_expectations": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force-write observed metrics into expectation files",
                },
                "allow_mixed_provenance": {
                    "type": "boolean",
                    "default": False,
                    "description": "Allow expectation writes when provenance differs without forcing updates",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "default": False,
                    "description": "Allow expectation writes while validity_profile=portable",
                },
                "auto_resume": {
                    "type": "boolean",
                    "default": True,
                    "description": "Launch a detached watcher that auto-resumes stale or aborted e2e runs using the stored contract",
                },
                "max_auto_resumes": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum detached auto-resume attempts before the watcher gives up",
                },
                "watch_poll_interval_seconds": {
                    "type": "integer",
                    "default": 15,
                    "description": "Detached watcher poll interval in seconds",
                },
                "resume": {
                    "type": "boolean",
                    "default": False,
                    "description": "Resume an aborted e2e run. Requires run_id and preserves prior stage artifacts.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Describe planned execution without running stages",
                },
                "async": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run in background and return a job ticket",
                },
            }
        ),
    },
)
def tool_benchmark_e2e_sweep(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_e2e_sweep(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_e2e_status",
    "Tags: benchmark, e2e, status, watch, resume, progress, watchdog, liveness. "
    "Inspect an end-to-end e2e run package and return one normalized status snapshot. "
    "Returns: {run_state, overall_status, inferred_state, progress_source, current, liveness, watcher, ledgers, actions}. "
    "⚡ FAST (~1s). USE when: You want the authoritative source of truth for a live or stalled run instead of manually joining summary/checkpoint/progress/events files. "
    'Example: "Status for run-e2e" or "Is my e2e run stale?". '
    "WORKFLOW: benchmark_e2e_sweep(async=true) → benchmark_e2e_status → benchmark_e2e_watch if needed. "
    "Prefer this tool whenever an e2e run is active; raw summary/checkpoint files may lag during long child stages. "
    "Selection: Read-only e2e run inspection; use benchmark_e2e_watch to arm detached auto-resume.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_id": {
                    "type": "string",
                    "description": "Explicit e2e run id to inspect; defaults to the latest run.",
                },
                "repo_root": {
                    "type": "string",
                    "description": "Optional repository root override.",
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Optional benchmark artifacts root override for child benchmark runs.",
                },
                "recent_events": {
                    "type": "integer",
                    "default": 10,
                    "description": "How many recent top-level and child events to include.",
                },
            }
        ),
    },
)
def tool_benchmark_e2e_status(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_e2e_status(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_e2e_watch",
    "Tags: benchmark, e2e, watch, watchdog, autoresume, detached, recovery. "
    "Launch the detached e2e watcher that auto-resumes stale or aborted runs using the stored contract. "
    "Returns: {success, watcher_pid, watch_status_path, launch_log_path}. "
    "⚡ FAST to launch (~ms); the watcher continues running in the background. USE when: You want detached supervision for a long e2e run or need to re-arm auto-resume on an existing run. "
    "WORKFLOW: benchmark_e2e_sweep → benchmark_e2e_watch → benchmark_e2e_status. "
    "Selection: Detached supervision; use benchmark_e2e_status for read-only inspection.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_id": {
                    "type": "string",
                    "description": "Explicit e2e run id to supervise; defaults to the latest run.",
                },
                "repo_root": {
                    "type": "string",
                    "description": "Optional repository root override.",
                },
                "poll_interval_seconds": {
                    "type": "integer",
                    "default": 15,
                    "description": "Detached watcher poll interval in seconds.",
                },
                "max_auto_resumes": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum auto-resume attempts before the watcher gives up.",
                },
            }
        ),
    },
)
def tool_benchmark_e2e_watch(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.api import handlers

    include_context, context_level = extract_context_opts(params)
    result = handlers.benchmark_e2e_watch(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_report",
    "Tags: report, pdf, html, export, visualization, share, document. "
    "Generate PDF/HTML report from benchmark results for sharing and documentation. "
    "Returns: {output_path, format, success} and writes report file. "
    "⚡ FAST (~5s). USE AFTER: run_benchmarks or benchmark_triage. "
    'Example: "Generate HTML report" or "Create PDF performance summary". '
    "FORMATS: "
    "• html: Interactive, best for sharing/web "
    "• pdf: Static, best for formal documentation. "
    "WORKFLOW: run_benchmarks → benchmark_triage → benchmark_report(format='html'). "
    "REQUIRES: benchmark_test_results.json from run_benchmarks.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "data_file": {
                    "type": "string",
                    "description": (
                        "Path to benchmark_test_results.json from run_benchmarks "
                        "(typically artifacts/runs/<run_id>/benchmark_test_results.json; defaults to latest in artifacts/runs/)."
                    ),
                },
                "output": {
                    "type": "string",
                    "description": "Output file path (.pdf or .html); extension should match format.",
                    "default": "report.pdf",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: pdf or html",
                    "enum": ["pdf", "html"],
                    "default": "pdf",
                },
                "title": {"type": "string", "description": "Report title (optional)"},
                "author": {"type": "string", "description": "Report author (optional)"},
            }
        ),
    },
)
def tool_benchmark_report(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    args = ["report"]
    data_file = params.get("data_file")
    if data_file:
        if not Path(data_file).exists():
            return make_error(
                f"data_file not found: {data_file}",
                include_context,
                context_level,
                data_file=data_file,
            )
        args.extend(["--data-file", data_file])
    if params.get("output"):
        args.extend(["--output", params["output"]])
        try:
            _ensure_dir(Path(params["output"]))
        except Exception as exc:
            return make_error(
                f"Failed to prepare output path for benchmark report {params['output']}: {exc}",
                include_context,
                context_level,
                output=params["output"],
            )
    if params.get("format"):
        args.extend(["--format", params["format"]])
    if params.get("title"):
        args.extend(["--title", params["title"]])
    if params.get("author"):
        args.extend(["--author", params["author"]])
    result = _run_bench_cli(args)
    if isinstance(result, dict) and "success" not in result:
        returncode = result.get("returncode", 0)
        result["success"] = returncode == 0 and not result.get("error")
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_export",
    "Tags: export, csv, markdown, json, data, share, spreadsheet. "
    "Export benchmark results to CSV/Markdown/JSON format for further analysis. "
    "Returns: {output_path, format, benchmarks_written, success}. "
    "USE when: Importing results into spreadsheets, documentation, or other tools. "
    'Example: "Export benchmarks to CSV" or "Convert results to markdown table". '
    "REQUIRES: Run run_benchmarks first to generate benchmark_test_results.json. ⚡ FAST (~2s). WORKFLOW: run_benchmarks → benchmark_export. NOT FOR: Reports (use benchmark_report).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "data_file": {
                    "type": "string",
                    "description": (
                        "Path to benchmark_test_results.json from run_benchmarks "
                        "(typically artifacts/runs/<run_id>/benchmark_test_results.json; defaults to latest in artifacts/runs/)."
                    ),
                },
                "format": {
                    "type": "string",
                    "description": "Output format: csv, markdown, or json",
                    "enum": ["csv", "markdown", "json"],
                    "default": "csv",
                },
                "output": {
                    "type": "string",
                    "description": "Output file path (defaults to benchmark_export.<format> if omitted).",
                },
            }
        ),
    },
)
def tool_benchmark_export(params: Dict[str, Any]) -> BenchmarkExportResult:
    """Export benchmark results without spawning the bench CLI."""
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data

    include_context, context_level = extract_context_opts(params)
    fmt = normalize_param("format", params.get("format"), "csv")
    data_file = params.get("data_file")
    output = params.get("output")

    valid_formats = {"csv", "markdown", "json"}
    if fmt not in valid_formats:
        return make_error(
            f"format must be one of {sorted(valid_formats)}", include_context, context_level
        )

    data_path = Path(data_file) if data_file else None
    if data_path and not data_path.exists():
        return make_error(
            f"data_file not found: {data_path}",
            include_context,
            context_level,
            data_file=str(data_path),
        )
    output_path = Path(output) if output else Path(f"benchmark_export.{fmt}")
    _ensure_dir(output_path)

    analyzer = PerformanceAnalyzer(lambda: load_benchmark_data(data_path))
    data = analyzer._load_data() or {}
    benchmarks = data.get("benchmarks", [])

    try:
        if fmt == "json":
            output_path.write_text(json.dumps(data, indent=2))
        elif fmt == "markdown":
            lines = ["| Benchmark | Speedup | Baseline (ms) | Type |", "|---|---|---|---|"]
            for b in benchmarks:
                lines.append(
                    f"| {b.get('chapter')}:{b.get('name')} | {b.get('speedup', 0):.2f}x | "
                    f"{b.get('baseline_time_ms', 0):.3f} | {b.get('type', 'python')} |"
                )
            output_path.write_text("\n".join(lines))
        else:  # csv
            import csv

            with output_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["benchmark", "speedup", "baseline_ms", "type"])
                for b in benchmarks:
                    writer.writerow(
                        [
                            f"{b.get('chapter')}:{b.get('name')}",
                            f"{b.get('speedup', 0):.2f}",
                            f"{b.get('baseline_time_ms', 0):.3f}",
                            b.get("type", "python"),
                        ]
                    )
    except Exception as exc:
        return make_error(
            f"failed to export benchmarks: {exc}",
            include_context,
            context_level,
            output=str(output_path),
        )

    result: BenchmarkExportResult = {
        "output": str(output_path),
        "format": fmt,
        "benchmarks_written": len(benchmarks),
        "success": True,
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_compare_runs",
    "Tags: compare, diff, regression, improvement, delta, a-b-test, before-after. "
    "Compare two benchmark runs showing speedup deltas, regressions, and improvements. "
    "Returns: {regressions: [...], improvements: [...], unchanged: [...], summary}. "
    "⚡ FAST (~1s). USE when: Comparing before/after optimization, detecting regressions. "
    'Example: "Compare baseline vs optimized" or "Show top 10 regressions". '
    "REQUIRES: Two benchmark_test_results.json files from separate runs. "
    "WORKFLOW: "
    "1. run_benchmarks (baseline) → save results "
    "2. Make optimizations "
    "3. run_benchmarks (candidate) → save results "
    "4. benchmark_compare_runs(baseline=..., candidate=...) "
    "5. If regressions: analyze_bottlenecks on affected benchmarks.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "baseline": {
                    "type": "string",
                    "description": "Path to baseline benchmark_test_results.json (e.g., artifacts/runs/<run_id>/benchmark_test_results.json)",
                },
                "candidate": {
                    "type": "string",
                    "description": "Path to candidate benchmark_test_results.json (e.g., artifacts/runs/<run_id>/benchmark_test_results.json)",
                },
                "top": {
                    "type": "integer",
                    "description": "Show top N regressions/improvements",
                    "default": 10,
                },
            }
        ),
        "required": ["baseline", "candidate"],
    },
)
def tool_benchmark_compare_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    baseline = params.get("baseline")
    candidate = params.get("candidate")
    if not baseline or not candidate:
        return make_error(
            "baseline and candidate benchmark files are required", include_context, context_level
        )
    if baseline and not Path(baseline).exists():
        return make_error(
            f"baseline file not found: {baseline}",
            include_context,
            context_level,
            baseline=baseline,
        )
    if candidate and not Path(candidate).exists():
        return make_error(
            f"candidate file not found: {candidate}",
            include_context,
            context_level,
            candidate=candidate,
        )

    args = [
        "compare-runs",
        "--baseline",
        baseline,
        "--candidate",
        candidate,
        "--top",
        str(params.get("top", 10)),
    ]
    result = _run_bench_cli(args)
    if isinstance(result, dict) and "success" not in result:
        returncode = result.get("returncode", 0)
        result["success"] = returncode == 0 and not result.get("error")
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "benchmark_triage",
    "Tags: benchmark, triage, analysis, recommendations, next-steps, post-benchmark, actionable. "
    "🔍 POST-BENCHMARK ANALYSIS: Analyze benchmark results and get actionable recommendations. "
    "Returns: {summary, regressions, improvements, top_issues, recommended_tools, optimization_plan}. "
    "⚡ FAST (~2s). USE AFTER: run_benchmarks completes successfully. "
    'Example: "Analyze my benchmark results" or "What should I optimize based on benchmarks?". '
    "PROVIDES: "
    "• Summary of all benchmark results (pass/fail/speedup) "
    "• Identification of regressions and improvements "
    "• Specific tool recommendations based on findings "
    "• Prioritized optimization plan. "
    "WORKFLOW: run_benchmarks → benchmark_triage → implement recommendations → re-benchmark.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "data_file": {
                    "type": "string",
                    "description": "Path to benchmark_test_results.json from run_benchmarks (defaults to latest in artifacts/).",
                },
                "baseline_file": {
                    "type": "string",
                    "description": "Optional baseline to compare against for regression detection",
                },
            }
        ),
    },
)
def tool_benchmark_triage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and provide actionable recommendations."""
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data

    include_context, context_level = extract_context_opts(params)
    data_file = params.get("data_file")
    baseline_file = params.get("baseline_file")

    # Find latest benchmark results if not specified
    data_path = Path(data_file) if data_file else None
    if not data_path:
        # Search common locations
        search_paths = [
            Path("artifacts/benchmark_test_results.json"),
            Path("benchmark_test_results.json"),
            Path("artifacts/benchmarks/benchmark_test_results.json"),
        ]
        for p in search_paths:
            if p.exists():
                data_path = p
                break

    if not data_path or not data_path.exists():
        return make_error(
            "No benchmark results found. Run run_benchmarks first.",
            include_context,
            context_level,
            searched_paths=[str(p) for p in search_paths] if not data_file else None,
            hint="Specify data_file parameter or run run_benchmarks to generate results.",
        )

    try:
        data = load_benchmark_data(data_path)
    except Exception as exc:
        return make_error(f"Failed to load benchmark data: {exc}", include_context, context_level)

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return make_error(
            "Benchmark file contains no results",
            include_context,
            context_level,
            data_file=str(data_path),
        )

    def _status_bucket(status: Any) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"success", "ok"}:
            normalized = "succeeded"
        elif normalized in {"skip"}:
            normalized = "skipped"
        elif normalized in {"error"}:
            normalized = "failed"
        if normalized.startswith("failed_"):
            return "failed"
        if normalized in {"succeeded", "failed", "skipped"}:
            return normalized
        return normalized or "unknown"

    status_counts_raw: Dict[str, int] = {}
    for b in benchmarks:
        raw = str(b.get("status", "unknown")).strip().lower() or "unknown"
        status_counts_raw[raw] = status_counts_raw.get(raw, 0) + 1

    # Analyze results
    total = len(benchmarks)
    passed = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "succeeded")
    failed = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "failed")
    skipped = sum(1 for b in benchmarks if _status_bucket(b.get("status")) == "skipped")

    # Find regressions and improvements
    regressions = sorted(
        [
            b
            for b in benchmarks
            if _status_bucket(b.get("status")) == "succeeded" and b.get("speedup", 1.0) < 0.95
        ],
        key=lambda x: x.get("speedup", 1.0),
    )[:10]

    improvements = sorted(
        [
            b
            for b in benchmarks
            if _status_bucket(b.get("status")) == "succeeded" and b.get("speedup", 1.0) > 1.05
        ],
        key=lambda x: x.get("speedup", 1.0),
        reverse=True,
    )[:10]

    # Identify patterns for recommendations
    slow_kernels = [b for b in benchmarks if b.get("baseline_time_ms", 0) > 100]
    memory_issues = [
        b
        for b in benchmarks
        if "memory" in b.get("name", "").lower() or "oom" in str(b.get("error", "")).lower()
    ]
    attention_benchmarks = [b for b in benchmarks if "attention" in b.get("name", "").lower()]

    # Build recommendations
    recommended_tools = []
    optimization_plan = []

    if regressions:
        recommended_tools.append(
            {
                "tool": "analyze_bottlenecks",
                "reason": f"Identify root cause of {len(regressions)} regression(s)",
                "priority": "high",
            }
        )
        optimization_plan.append(
            {
                "step": 1,
                "action": "Investigate regressions",
                "details": f"Top regression: {regressions[0].get('chapter')}:{regressions[0].get('name')} ({regressions[0].get('speedup', 0):.2f}x)",
            }
        )

    if slow_kernels:
        recommended_tools.append(
            {
                "tool": "profile_nsys",
                "reason": f"Profile {len(slow_kernels)} slow benchmark(s) (>100ms baseline)",
                "params": {"preset": "light"},
            }
        )
        optimization_plan.append(
            {
                "step": 2,
                "action": "Profile slow operations",
                "details": f"Slowest: {slow_kernels[0].get('chapter')}:{slow_kernels[0].get('name')} ({slow_kernels[0].get('baseline_time_ms', 0):.1f}ms)",
            }
        )

    if attention_benchmarks:
        avg_attention_speedup = sum(b.get("speedup", 1.0) for b in attention_benchmarks) / len(
            attention_benchmarks
        )
        if avg_attention_speedup < 1.5:
            recommended_tools.append(
                {
                    "tool": "explain",
                    "reason": "Learn about FlashAttention optimization",
                    "params": {"concept": "flash-attention"},
                }
            )

    if improvements:
        recommended_tools.append(
            {
                "tool": "benchmark_report",
                "reason": f"Document {len(improvements)} improvement(s) in shareable report",
                "params": {"format": "html"},
            }
        )

    # Always suggest comparison if we have results
    recommended_tools.append(
        {
            "tool": "benchmark_compare_runs",
            "reason": "Compare with previous baseline for trend analysis",
            "note": "Save current results as baseline for future comparisons",
        }
    )

    # Add general optimization recommendations
    recommended_tools.append(
        {
            "tool": "recommend",
            "reason": "Get optimization playbook based on your hardware and goals",
        }
    )

    result = {
        "success": True,
        "data_file": str(data_path),
        "summary": {
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "failure_classes": {
                k: v
                for k, v in status_counts_raw.items()
                if k.startswith("failed_") or k == "failed"
            },
            "pass_rate": f"{(passed / total) * 100:.1f}%" if total > 0 else "N/A",
            "avg_speedup": sum(b.get("speedup", 1.0) for b in benchmarks) / total
            if total > 0
            else 0,
        },
        "regressions": [
            {
                "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                "speedup": b.get("speedup", 0),
                "baseline_ms": b.get("baseline_time_ms", 0),
            }
            for b in regressions
        ],
        "improvements": [
            {
                "benchmark": f"{b.get('chapter')}:{b.get('name')}",
                "speedup": b.get("speedup", 0),
                "baseline_ms": b.get("baseline_time_ms", 0),
            }
            for b in improvements
        ],
        "top_issues": [],
        "recommended_tools": recommended_tools,
        "optimization_plan": optimization_plan,
        "next_steps_summary": (
            f"Found {len(regressions)} regression(s) and {len(improvements)} improvement(s). "
            f"{'Focus on fixing regressions first.' if regressions else 'Results look good!'} "
            f"Use recommended_tools for specific actions."
        ),
    }

    # Add top issues
    if regressions:
        result["top_issues"].append(
            {
                "type": "regression",
                "count": len(regressions),
                "severity": "high"
                if any(b.get("speedup", 1.0) < 0.5 for b in regressions)
                else "medium",
            }
        )
    if slow_kernels:
        result["top_issues"].append(
            {"type": "slow_operations", "count": len(slow_kernels), "severity": "medium"}
        )

    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "analyze_bottlenecks",
    "Tags: bottleneck, slow, latency, utilization, diagnosis, why-slow, root-cause, debug. "
    "Identify performance bottlenecks: memory-bound, compute-bound, communication-bound, host-bound. "
    "Returns: {bottleneck_type, confidence, profile_data, llm_analysis, recommendations, availability}. "
    "⚡ FAST (~2-5s). USE FIRST when: Workload is slow and you don't know why. "
    'Example: "Why is my 7B model slow on 4xH100?" or "What\'s the bottleneck at batch 32, seq 4k?". '
    "mode='both' (default) combines profiling data with LLM analysis for best results. "
    "WORKFLOW by bottleneck_type: "
    "• memory-bound → profile_memory, analyze_whatif(max_vram_gb=X) "
    "• compute-bound → profile_kernels, hw_tc "
    "• communication-bound → distributed_nccl, hw_nccl "
    "• host-bound → cpu_memory_analysis, data_loading NOT FOR: Kernel metrics (use profile_ncu).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "analysis_type": {
                    "type": "string",
                    "description": "Focus area: bottleneck (general), memory, or compute",
                    "enum": ["bottleneck", "memory", "compute"],
                    "default": "bottleneck",
                },
                "mode": {
                    "type": "string",
                    "description": "Analysis mode: profile (data only), llm (AI analysis only), both (combined)",
                    "enum": ["profile", "llm", "both"],
                    "default": "both",
                },
            }
        ),
    },
)
def tool_analyze_bottlenecks(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze bottlenecks."""
    from core.engine import get_engine

    analysis_type = normalize_param("analysis_type", params.get("analysis_type"), "bottleneck")
    mode = normalize_param("mode", params.get("mode"), "both")
    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.bottlenecks(
        "profile" if mode == "profile" else "llm" if mode == "llm" else analysis_type
    )
    profile_available = bool(
        isinstance(result, dict)
        and isinstance(result.get("profile"), dict)
        and not result["profile"].get("error")
        and result["profile"].get("bottlenecks")
    )
    llm_available = bool(
        isinstance(result, dict)
        and isinstance(result.get("llm"), dict)
        and result["llm"].get("llm_response")
        and not result["llm"].get("error")
    )
    annotated = {
        **result,
        "availability": {
            "profile": profile_available,
            "llm": llm_available,
        },
    }
    return attach_context_if_requested(annotated, include_context, context_level)


@register_tool(
    "analyze_pareto",
    "Tags: pareto, tradeoff, throughput, latency, memory, frontier, optimal, comparison, choose. "
    "Find Pareto-optimal configurations: best throughput/latency/memory tradeoffs. "
    "Returns: {pareto_frontier: [{config, throughput, latency_ms, memory_gb}], dominated_configs, analysis}. "
    "⚡ FAST (~2s). USE AFTER: run_benchmarks with multiple configurations. "
    'Example: "Show Pareto frontier" or "What\'s the best throughput/latency tradeoff?". '
    "PARETO EXPLAINED: Points on the frontier are 'optimal' - you can't improve one metric without sacrificing another. "
    "WORKFLOW: "
    "1. run_benchmarks with varied batch_size/seq_len configs "
    "2. analyze_pareto → find optimal operating points "
    "3. analyze_whatif → check if constraints are met. "
    "REQUIRES: benchmark_test_results.json with multiple configurations.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_analyze_pareto(params: Dict[str, Any]) -> Dict[str, Any]:
    """Pareto analysis."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.pareto()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "analyze_scaling",
    "Tags: scaling, throughput, gpus, nodes, projection, extrapolation, multi-gpu. "
    "Analyze how performance scales with workload size, sequence length, batch size, or GPU count. "
    "Returns: {scaling_efficiency, projections: [{gpus, throughput, efficiency_pct}], bottleneck_at_scale}. "
    "⚡ FAST (~2s). USE when: Projecting performance to larger inputs, planning multi-GPU scaling. "
    'Example: "Predict throughput if I double sequence length" or "How does it scale from 2 to 4 GPUs?". '
    "WORKFLOW: gpu_topology → analyze_scaling → distributed_plan. "
    "ALSO USE: predict_scaling for specific GPU count predictions.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_analyze_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    """Scaling analysis."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.scaling()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "analyze_stacking",
    "Tags: stacking, combinations, techniques, compatibility, conflicts, compose. "
    "Analyze which optimization techniques work well together and which conflict. "
    "Returns: {compatible_stacks: [...], conflicts: [{technique1, technique2, reason}], recommended_order}. "
    "⚡ FAST (~2s). USE when: Planning to combine multiple optimizations, checking for conflicts. "
    'Example: "Can FlashAttention + torch.compile + CUDA graphs coexist?" or "What\'s the best optimization order?". '
    "WORKFLOW: recommend → analyze_stacking → apply compatible techniques. "
    "ALSO USE: optimize_techniques for full technique list, optimize_roi for prioritization.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_analyze_stacking(params: Dict[str, Any]) -> Dict[str, Any]:
    """Stacking analysis."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.stacking()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "analyze_whatif",
    "Tags: constraints, latency, vram, throughput, what-if, feasibility, target, sla, budget. "
    "What-if analysis: Find optimizations that meet your constraints (VRAM, latency, throughput). "
    "Returns: {feasible_configs: [...], recommended_optimizations, tradeoff_analysis}. "
    "⚡ FAST (~1s). USE when: Targeting specific SLA bounds, checking feasibility. "
    'Example: "Need <50ms latency with <24GB VRAM" or "Can I hit 2k tok/s?". '
    "CONSTRAINT EXAMPLES: "
    "• max_vram_gb=24: Fit on RTX 4090 / single A10G "
    "• max_latency_ms=50: Real-time chatbot SLA "
    "• min_throughput=1000: High-volume batch processing "
    "• Combine: max_vram_gb=48, max_latency_ms=100 (A6000 real-time). "
    "WORKFLOW: profile_memory → analyze_whatif → inference_quantization → verify.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "max_vram_gb": {
                    "type": "number",
                    "description": "Maximum VRAM budget in GB (e.g., 24 for single 3090)",
                },
                "max_latency_ms": {
                    "type": "number",
                    "description": "Maximum acceptable latency in milliseconds (e.g., 50ms SLA)",
                },
                "min_throughput": {
                    "type": "number",
                    "description": "Minimum required throughput in tokens/sec or samples/sec",
                },
            }
        ),
    },
)
def tool_analyze_whatif(params: Dict[str, Any]) -> Dict[str, Any]:
    """What-if analysis."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().analyze.whatif(params)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# OPTIMIZATION TOOLS
# =============================================================================


@register_tool(
    "optimize",
    "Tags: optimize, benchmark, llm, variants, workflow, shortcut. "
    "Resolve a benchmark file path or target and run quick LLM variants by default. "
    "Accepts either `path` (benchmark file path) or `target` (chapter:example). "
    "Fails fast if the path does not map to a unique benchmark target. "
    "Returns the same outputs as benchmark_variants. "
    "USE when: You have a benchmark file path or target and want quick LLM variant testing. "
    "NOT FOR: General recommendations (use recommend). "
    "Example: path='ch10/baseline_atomic_reduction.py' or target='ch10:atomic_reduction'.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "path": {
                    "type": "string",
                    "description": "Benchmark file path (baseline_*/optimized_* .py wrapper only).",
                },
                "target": {
                    "type": "string",
                    "description": "Benchmark target in chapter:example format.",
                },
                "profile": {
                    "type": "string",
                    "description": "Profiling preset: none (no profiling), minimal (basic), deep_dive (full nsys/ncu profiling), or roofline",
                    "enum": ["none", "minimal", "deep_dive", "roofline"],
                    "default": "minimal",
                },
                "artifacts_dir": {
                    "type": "string",
                    "description": "Base directory for artifacts (bench creates a self-describing run dir underneath).",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for artifacts (default: <timestamp>__bench__profile-<type>__targets-<...>)",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Override benchmark iterations (all targets).",
                },
                "warmup": {
                    "type": "integer",
                    "description": "Override warmup iterations (all targets).",
                },
                "llm_analysis": {
                    "type": "boolean",
                    "description": "Enable LLM-powered analysis (default true for this shortcut).",
                    "default": True,
                },
                "force_llm": {
                    "type": "boolean",
                    "description": "Force LLM analysis on all benchmarks regardless of speedup (default true for this shortcut).",
                    "default": True,
                },
                "apply_patches": {
                    "type": "boolean",
                    "description": "Apply LLM-suggested patches to create new optimized variants (default true for this shortcut).",
                    "default": True,
                },
                "rebenchmark_llm_patches": {
                    "type": "boolean",
                    "description": "Re-benchmark LLM-patched variants (default true for this shortcut).",
                    "default": True,
                },
                "llm_explain": {
                    "type": "boolean",
                    "description": "Generate LLM explanations for best patches (requires rebenchmark_llm_patches=true).",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning with partial output; set 0/null for no timeout",
                    "default": 900,
                },
                "validity_profile": {
                    "type": "string",
                    "description": (
                        "Benchmark validity profile (`--validity-profile`, MCP field `validity_profile`): strict (default; fail-fast with full validity checks) or portable "
                        "(explicit compatibility mode for hardware without full benchmark controls)."
                    ),
                    "enum": ["strict", "portable"],
                    "default": "strict",
                },
                "allow_portable_expectations_update": {
                    "type": "boolean",
                    "description": (
                        "In the portable validity profile, expectation writes are disabled by default. "
                        "Set this flag to allow expectation-file updates."
                    ),
                    "default": False,
                },
            }
        ),
    },
)
def tool_optimize(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    path = params.get("path")
    target = params.get("target")
    if path is not None and not isinstance(path, str):
        return make_error("path must be a string.", include_context, context_level)
    if target is not None and not isinstance(target, str):
        return make_error("target must be a string.", include_context, context_level)
    if path and target:
        return make_error(
            "Provide either path or target, not both.", include_context, context_level
        )
    if not path and not target:
        return make_error("path or target is required.", include_context, context_level)

    resolved_target = None
    if path:
        if not path.strip():
            return make_error("path cannot be empty.", include_context, context_level)
        resolved_target, err = _resolve_benchmark_target_from_path(str(path))
        if err:
            return make_error(err, include_context, context_level)
    else:
        resolved_target = target.strip()
        if not resolved_target:
            return make_error("target cannot be empty.", include_context, context_level)

    merged = dict(params)
    merged.pop("path", None)
    merged.pop("target", None)
    merged["targets"] = [resolved_target]
    return tool_benchmark_variants(merged)


@register_tool(
    "recommend",
    "Tags: recommend, playbook, throughput, latency, memory, optimization, guide, strategy, gameplan. "
    "Get prioritized optimization recommendations for your model configuration and goal. "
    "Returns: {recommendations: [{technique, priority, expected_speedup, effort}], playbook, warnings}. "
    "⚡ FAST (~1s). USE EARLY when: Starting optimization work, need a game plan. "
    'Example: "Recommend for 13B on 4xA100 focused on throughput" or "Low-latency 7B on single H100". '
    "GOALS explained: "
    "• throughput → maximize tokens/sec (batch processing, training) "
    "• latency → minimize TTFT (real-time inference, chatbots) "
    "• memory → reduce VRAM (fit larger models, longer sequences). "
    "WORKFLOW: triage → recommend → optimize_roi → implement techniques → run_benchmarks.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (7, 13, 70, etc.)",
                    "default": 7,
                },
                "gpus": {
                    "type": "integer",
                    "description": "Number of GPUs available (1, 4, 8, etc.)",
                    "default": 1,
                },
                "goal": {
                    "type": "string",
                    "description": "Primary optimization goal: throughput (tok/s), latency (TTFT), or memory (VRAM)",
                    "enum": ["throughput", "latency", "memory"],
                    "default": "throughput",
                },
            }
        ),
        "required": [],
    },
)
def tool_recommend(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommendations."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    goal = normalize_param("goal", params.get("goal"), "throughput")
    result = get_engine().optimize.recommend(
        model_size=params.get("model_size", 7), gpus=params.get("gpus", 1), goal=goal
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "optimize_roi",
    "Tags: ROI, prioritize, cost-benefit, effort, impact, ranking, efficiency. "
    "Calculate ROI (return on investment) for optimization techniques: expected gain vs implementation effort. "
    "Returns: {ranked_techniques: [{name, expected_speedup, effort_hours, roi_score}], quick_wins, high_impact}. "
    "USE when: Prioritizing optimization work, deciding what to implement first, limited engineering time. "
    'Example: "Which optimizations give best ROI?" or "Rank techniques by cost vs gain". '
    "ALSO USE: optimize_techniques for full technique details, recommend for goal-specific recs. ⚡ FAST (~1s). WORKFLOW: recommend → optimize_roi → prioritize.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_optimize_roi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimization ROI."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().optimize.roi()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "optimize_techniques",
    "Tags: techniques, list, options, catalog, encyclopedia, reference. "
    "Get catalog of all optimization techniques with details, requirements, and expected benefits. "
    "Returns: {techniques: [{name, category, description, requirements, expected_speedup, gotchas}], count}. "
    "USE when: Exploring what optimizations exist, learning about technique requirements, reference lookup. "
    'Example: "List all optimization techniques" or "What techniques exist for attention?". '
    "ALSO USE: optimize_roi for prioritization, analyze_stacking for compatibility. ⚡ FAST (~1s). WORKFLOW: optimize_techniques → choose → optimize_roi.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_optimize_techniques(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all optimization techniques."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().optimize.all_techniques()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# DISTRIBUTED TRAINING TOOLS
# =============================================================================


@register_tool(
    "distributed_plan",
    "Tags: distributed, dp, tp, pp, fsdp, parallelism, multi-gpu, multi-node, strategy, sharding. "
    "Plan parallelism strategy: recommend DP/TP/PP/FSDP layout for model size and GPU count. "
    "Returns: {recommended_layout: {tp, pp, dp}, memory_per_gpu_gb, communication_volume, rationale}. "
    "⚡ FAST (~1s). USE when: Setting up distributed training, choosing parallelism degrees. "
    'Example: "Plan 70B on 2 nodes x 4 GPUs" or "What TP/PP for 14B on 4 GPUs?". '
    "PARALLELISM explained: "
    "• TP (Tensor Parallel): Split layers across GPUs; needs NVLink; TP ≤ 8 typically "
    "• PP (Pipeline Parallel): Split model stages; good for multi-node "
    "• DP (Data Parallel): Replicate model; scale batch size "
    "• FSDP: Shard parameters + gradients; memory-efficient DP. "
    "WORKFLOW: distributed_plan → distributed_nccl → launch_plan → training.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (7, 13, 70, etc.)",
                    "default": 7,
                },
                "gpus": {
                    "type": "integer",
                    "description": "Total number of GPUs across all nodes",
                    "default": 8,
                },
                "nodes": {
                    "type": "integer",
                    "description": "Number of nodes (1 for single-node, 2+ for multi-node)",
                    "default": 1,
                },
            }
        ),
    },
)
def tool_distributed_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Plan parallelism."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().distributed.plan(
        model_size=params.get("model_size", 7),
        gpus=params.get("gpus", 8),
        nodes=params.get("nodes", 1),
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "distributed_nccl",
    "Tags: nccl, multi-node, collective, allreduce, environment, tuning, ib, rdma. "
    "Get NCCL tuning recommendations: environment variables, IB settings, collective algorithms. "
    "Returns: {env_vars: {NCCL_*: value}, algorithm_hints, ib_recommendations, debug_tips}. "
    "USE when: Tuning NCCL for multi-node training, debugging collective performance, IB/RDMA setup. "
    'Example: "NCCL settings for 2-node 4xH100" or "Tune NCCL for InfiniBand". '
    "ALSO USE: hw_nccl for NCCL bandwidth testing, system_network for IB status. ⚡ FAST (~1s). WORKFLOW: distributed_plan → distributed_nccl → apply env vars.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "nodes": {"type": "integer", "description": "Number of nodes", "default": 1},
                "gpus": {"type": "integer", "description": "GPUs per node", "default": 8},
            }
        ),
    },
)
def tool_distributed_nccl(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get NCCL tuning."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().distributed.nccl(nodes=params.get("nodes", 1), gpus=params.get("gpus", 8))
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# INFERENCE TOOLS
# =============================================================================


@register_tool(
    "inference_vllm",
    "Tags: vllm, inference, serving, deployment, config, batching, kv-cache, production. "
    "Generate optimized vLLM configuration for inference serving (explicit model size required). "
    "Returns: {vllm_config|engine_comparison, launch_command, tips}. "
    "⚡ FAST (~1s). USE when: Deploying vLLM server, optimizing inference serving. "
    'Example: "vLLM settings for 7B low latency on A100" or "High-throughput 70B vLLM config". '
    "TARGETS explained: "
    "• throughput: Large batches, high gpu_memory_utilization (~0.9), best for batch inference "
    "• latency: Small batches, lower memory util, continuous batching tuned for TTFT "
    "• memory: Maximize fit with conservative batching. "
    "WORKFLOW: gpu_info → inference_quantization → inference_vllm → inference_deploy. "
    "ALSO USE: inference_quantization for precision recommendations.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model": {
                    "type": "string",
                    "description": "Model name (e.g., 'meta-llama/Llama-3.1-70B')",
                    "default": "model",
                },
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (required).",
                },
                "gpus": {
                    "type": "integer",
                    "description": "Number of GPUs available",
                    "default": 1,
                },
                "gpu_memory_gb": {
                    "type": "number",
                    "description": "VRAM per GPU in GB",
                    "default": 80,
                },
                "target": {
                    "type": "string",
                    "description": "Optimization target: throughput, latency, or memory",
                    "enum": ["throughput", "latency", "memory"],
                    "default": "throughput",
                },
                "max_seq_length": {
                    "type": "integer",
                    "description": "Max sequence length for config sizing",
                    "default": 8192,
                },
                "quantization": {
                    "type": "string",
                    "description": "Optional quantization mode (awq/gptq/fp8/int8)",
                },
                "compare": {
                    "type": "boolean",
                    "description": "If true, return engine comparison instead of config",
                    "default": False,
                },
            }
        ),
        "required": ["model_size"],
    },
)
def tool_inference_vllm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate vLLM config."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    target = normalize_param("target", params.get("target"), "throughput")
    model_size = params.get("model_size")
    result = get_engine().inference.vllm_config(
        model=params.get("model", "model"),
        model_params_b=model_size,
        num_gpus=params.get("gpus", 1),
        gpu_memory_gb=params.get("gpu_memory_gb", 80),
        target=target,
        max_seq_length=params.get("max_seq_length", 8192),
        quantization=params.get("quantization"),
        compare=bool(params.get("compare", False)),
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "inference_deploy",
    "Tags: inference, deploy, serving, config, launch, throughput, latency. "
    "Generate inference deployment configuration (explicit model size required). "
    "Returns: {model, hardware, goal, engine, launch_command, report}. "
    "⚡ FAST (~1s). USE when: Planning inference deployments or generating launch commands. "
    'Example: "Deploy config for 70B on 4xA100" or "Get inference launch command". '
    "GOALS: throughput (batch), latency (TTFT), memory (fit). "
    "WORKFLOW: gpu_info → inference_quantization → inference_deploy. "
    "NOT FOR: vLLM-specific tuning (use inference_vllm).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model": {
                    "type": "string",
                    "description": "Model name (e.g., 'meta-llama/Llama-3.1-70B')",
                    "default": "model",
                },
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (required).",
                },
                "gpus": {
                    "type": "integer",
                    "description": "Number of GPUs available",
                    "default": 1,
                },
                "gpu_memory_gb": {
                    "type": "number",
                    "description": "VRAM per GPU in GB",
                    "default": 80,
                },
                "goal": {
                    "type": "string",
                    "description": "Optimization goal: throughput, latency, or memory",
                    "enum": ["throughput", "latency", "memory"],
                    "default": "throughput",
                },
                "target": {
                    "type": "string",
                    "description": "Alias for goal (throughput/latency/memory).",
                    "enum": ["throughput", "latency", "memory"],
                },
                "max_seq_length": {
                    "type": "integer",
                    "description": "Max sequence length for config sizing",
                    "default": 8192,
                },
            }
        ),
        "required": ["model_size"],
    },
)
def tool_inference_deploy(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate inference deployment config."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    payload = {
        "model": params.get("model", "model"),
        "model_params_b": params.get("model_size"),
        "model_size": params.get("model_size"),
        "num_gpus": params.get("gpus", 1),
        "gpu_memory_gb": params.get("gpu_memory_gb", 80),
        "goal": params.get("goal") or params.get("target") or "throughput",
        "max_seq_length": params.get("max_seq_length", 8192),
    }
    result = get_engine().inference.deploy(payload)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "inference_estimate",
    "Tags: inference, estimate, throughput, latency, memory, sizing. "
    "Estimate inference throughput/latency based on model + hardware (explicit model size required). "
    "Returns: {model, hardware, goal, estimate: {throughput_tps, latency_ms, memory_gb}, engine}. "
    "⚡ FAST (~1s). USE when: Quickly sizing deployments or comparing hardware options. "
    'Example: "Estimate latency for 13B on 1xH100". '
    "WORKFLOW: inference_deploy → inference_estimate. "
    "NOT FOR: Exact benchmarking (use run_benchmarks).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model": {
                    "type": "string",
                    "description": "Model name (e.g., 'meta-llama/Llama-3.1-70B')",
                    "default": "model",
                },
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (required).",
                },
                "gpus": {
                    "type": "integer",
                    "description": "Number of GPUs available",
                    "default": 1,
                },
                "gpu_memory_gb": {
                    "type": "number",
                    "description": "VRAM per GPU in GB",
                    "default": 80,
                },
                "goal": {
                    "type": "string",
                    "description": "Optimization goal: throughput, latency, or memory",
                    "enum": ["throughput", "latency", "memory"],
                    "default": "throughput",
                },
                "target": {
                    "type": "string",
                    "description": "Alias for goal (throughput/latency/memory).",
                    "enum": ["throughput", "latency", "memory"],
                },
                "max_seq_length": {
                    "type": "integer",
                    "description": "Max sequence length for config sizing",
                    "default": 8192,
                },
            }
        ),
        "required": ["model_size"],
    },
)
def tool_inference_estimate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate inference performance."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    payload = {
        "model": params.get("model", "model"),
        "model_params_b": params.get("model_size"),
        "model_size": params.get("model_size"),
        "num_gpus": params.get("gpus", 1),
        "gpu_memory_gb": params.get("gpu_memory_gb", 80),
        "goal": params.get("goal") or params.get("target") or "throughput",
        "max_seq_length": params.get("max_seq_length", 8192),
    }
    result = get_engine().inference.estimate(payload)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "inference_quantization",
    "Tags: quantization, fp8, int8, int4, awq, gptq, precision, compression, memory, bnb. "
    "Get quantization recommendations: precision format, method, expected accuracy/speedup tradeoffs. "
    "Returns: {recommended_format, alternatives: [{format, memory_reduction, speedup, accuracy_loss}], tips}. "
    "⚡ FAST (~1s). USE when: Choosing quantization format for inference. "
    'Example: "Should I use FP8 or INT8 for 70B inference?" or "Best quantization for 24GB VRAM?". '
    "FORMATS explained: "
    "• FP8 (E4M3/E5M2): Best quality/speed; Hopper+ only (H100/H200); ~50% memory reduction "
    "• INT8: Good quality/speed; Ampere+ (A100/RTX30xx+); ~50% memory reduction "
    "• INT4 (AWQ/GPTQ): Max compression; ~75% memory reduction; slight quality loss "
    "• NF4 (bitsandbytes): Easy setup; ~75% reduction; QLoRA-friendly. "
    "WORKFLOW: gpu_info (check arch) → inference_quantization → inference_vllm.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model_size": {
                    "type": "number",
                    "description": "Model size in billions of parameters (affects memory savings calculation)",
                },
            }
        ),
    },
)
def tool_inference_quantization(params: Dict[str, Any]) -> Dict[str, Any]:
    """Quantization recommendations."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().inference.quantization(model_size=params.get("model_size"))
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# AI/LLM TOOLS
# =============================================================================


@register_tool(
    "ask",
    "Tags: question, advice, why-slow, guidance, help, answer, book, citations, free-form. "
    "Ask a free-form performance question and get an answer with book citations. "
    "Returns: {answer, citations: [{chapter, section, relevance}], related_tools}. "
    "⚡ FAST (~2-5s). USE when: Need targeted advice, best practices, 'why' questions. "
    "GOOD QUESTIONS: "
    "• 'Is FlashAttention worth it on Llama-2 7B?' "
    "• 'Why is my attention kernel slow at seq_len=4k?' "
    "• 'Should I use torch.compile or CUDA graphs?' "
    "• 'What causes GPU memory fragmentation?'. "
    "REQUIRES: AI backend available (check with ai_status). "
    "VERSUS: explain (concept definitions), recommend (optimization playbooks), "
    "suggest_tools (which tool to use). WORKFLOW: ask for advice → specific tools for action. NOT FOR: Raw data (use domain tools).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "question": {
                    "type": "string",
                    "description": "Your performance question in natural language",
                },
            }
        ),
        "required": ["question"],
    },
)
def tool_ask(params: Dict[str, Any]) -> Dict[str, Any]:
    """Ask a performance question."""
    from core.engine import get_engine

    question = params.get("question", "")
    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.ask(question)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "explain",
    "Tags: explain, concept, definition, learn, understand, what-is, glossary. "
    "Explain a GPU/AI performance concept with clear definition and book citations. "
    "Returns: {explanation, key_points: [...], citations: [...], related_concepts}. "
    "USE when: Learning what a technique/concept is, understanding terminology, comparing concepts. "
    'Example: "Explain tensor parallelism vs pipeline parallelism" or "What is FlashAttention?". '
    "Good for: flash-attention, tensor-parallelism, FSDP, KV-cache, torch.compile, CUDA graphs. "
    "PREFER ask for 'why' or 'how' questions, optimize_techniques for technique catalog. ⚡ FAST (~3s). WORKFLOW: explain for concepts → ask for specific advice. NOT FOR: How-to questions (use ask).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "concept": {
                    "type": "string",
                    "description": "The concept to explain (e.g., 'flash-attention', 'tensor parallelism')",
                },
            }
        ),
        "required": ["concept"],
    },
)
def tool_explain(params: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a concept."""
    from core.engine import get_engine

    concept = params.get("concept", "")
    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.explain(concept)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "ai_troubleshoot",
    "Tags: troubleshoot, errors, distributed, nccl, oom, diagnosis, fixes. "
    "Diagnose common training/distributed errors and suggest fixes. "
    "Returns: {issues_found, issues: [{category, severity, title, description, symptoms, root_causes, solutions, code_fix, env_vars}]}. "
    "⚡ FAST (~2s). USE when: You have an error message or symptoms and need actionable fixes. "
    'Example: "Diagnose NCCL timeout" or "Why am I OOMing on 24GB?". '
    "WORKFLOW: ai_troubleshoot → apply fixes → re-run. "
    "NOT FOR: General performance advice (use ask).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "issue": {"type": "string", "description": "Error message or issue description"},
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of symptoms",
                },
                "config": {
                    "type": "object",
                    "description": "Optional configuration context (model/hardware/parallelism)",
                },
            }
        ),
        "required": ["issue"],
    },
)
def tool_ai_troubleshoot(params: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnose common training/distributed errors."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    issue = params.get("issue")
    if not issue:
        return make_error("issue is required", include_context, context_level)
    result = get_engine().ai.troubleshoot(
        issue=str(issue),
        symptoms=params.get("symptoms"),
        config=params.get("config"),
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "ai_status",
    "Tags: ai, llm, backend, connectivity, health, api-key. "
    "Check AI/LLM backend availability: connectivity, API key status, model availability. "
    "Returns: {available, backend_type, model, api_key_set, error_if_any}. "
    "⚡ FAST (<1s). USE when: Verifying LLM connectivity before ask/explain. "
    'Example: "Is the LLM backend reachable?" or "Why is ask failing?". '
    "WORKFLOW: ai_status → if available → ask/explain. "
    "NOT FOR: General system health (use status).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_ai_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check AI status."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().ai.status()
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# PROFILING TOOLS
# =============================================================================


@register_tool(
    "profile_flame",
    "Tags: profile, flame, hotspots, time, breakdown, visualization, call-stack. "
    "Get flame graph data showing execution time breakdown by function/operation. "
    "Returns: {flame_data, top_hotspots: [{function, time_pct, time_ms}], call_tree}. "
    "USE when: Identifying time hotspots, understanding where time is spent, visualizing call stacks. "
    'Example: "Show flame graph for my training loop" or "Where is time spent?". '
    "ALSO USE: profile_kernels for CUDA kernel breakdown, profile_nsys for full timeline. ⚡ FAST (~2s). WORKFLOW: profile_flame → hotspots → profile_kernels.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_flame(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get flame graph."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.flame_graph()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "profile_memory",
    "Tags: memory, timeline, spikes, leaks, oom, fragmentation, allocation, vram, cuda-oom. "
    "Get memory allocation timeline: VRAM usage over time, allocation spikes, potential leaks. "
    "Returns: {timeline: [{timestamp, allocated_gb, reserved_gb}], peak_usage, spikes, leak_suspects}. "
    "⚡ FAST (~2s). USE when: Debugging OOM, tracking memory spikes, finding leaks. "
    'Example: "Graph VRAM over time" or "Why am I running out of memory?" or "Find memory leak". '
    "COMMON OOM CAUSES: "
    "• Peak > VRAM: Reduce batch_size, use gradient checkpointing "
    "• Fragmentation: Use memory_efficient_attention, torch.cuda.empty_cache() "
    "• Leak: Check for growing tensor lists, unreleased intermediate tensors. "
    "WORKFLOW: profile_memory → analyze_whatif(max_vram_gb=X) → inference_quantization.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_memory(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory timeline."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.memory_timeline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "profile_kernels",
    "Tags: kernel, cuda, hotspots, gpu-time, breakdown, slow-kernels. "
    "Get CUDA kernel execution breakdown: time per kernel, launch counts, occupancy hints. "
    "Returns: {kernels: [{name, total_time_ms, call_count, avg_time_us, occupancy}], total_gpu_time}. "
    "USE when: Identifying slow CUDA kernels, analyzing GPU time distribution, finding optimization targets. "
    'Example: "Which CUDA kernels are slow?" or "Kernel breakdown for attention". '
    "ALSO USE: profile_ncu for detailed kernel metrics, profile_roofline for bound analysis. ⚡ FAST (~2s). WORKFLOW: profile_kernels → slow kernels → profile_ncu.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_kernels(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get kernel breakdown."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.kernel_breakdown()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "profile_roofline",
    "Tags: roofline, compute-bound, memory-bound, arithmetic-intensity, efficiency, bottleneck. "
    "Get roofline model analysis: compute vs memory bound positioning, arithmetic intensity, efficiency. "
    "Returns: {bound_type, arithmetic_intensity, achieved_flops, peak_flops, achieved_bandwidth, peak_bandwidth}. "
    "USE when: Determining if kernels are compute- or memory-bound, understanding optimization direction. "
    'Example: "Are my kernels memory-bound?" or "What\'s the arithmetic intensity of my workload?". '
    "Memory-bound → optimize memory access; Compute-bound → optimize math operations. ⚡ FAST (~2s). WORKFLOW: profile_roofline → if memory-bound → analyze_memory_patterns. NOT FOR: Running benchmarks (use hw_roofline).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get roofline data."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().profile.roofline()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "profile_compare",
    "Tags: compare, flamegraph, baseline, optimized, speedup, visualization, why-faster. "
    "Generate visual flame graph comparison showing WHY optimized code is faster. "
    "Returns: {speedup, cuda_api_comparison, kernel_breakdown, flame_diff, side_by_side_report, html_output (if requested)}. "
    "Also generates a side-by-side Nsight Systems + Nsight Compute JSON report + narrative by default. "
    "USE when: Understanding optimization impact visually, presenting before/after comparison. "
    'Example: "Compare baseline vs optimized streams profiles" or "Why is the optimized code faster?". '
    "Provide chapter (e.g., 'ch11') OR profiles_dir path (for example, benchmarks[].profiles_dir from benchmark_deep_dive_compare). "
    "If multiple pairs exist, provide pair to select one. Outputs interactive HTML if output_html set. "
    "Always returns nsys/ncu comparison metrics when profiles are captured; analyze metric deltas for regressions/improvements. "
    "🕐 MEDIUM (~5s). WORKFLOW: profile baseline → optimize → profile_compare. NOT FOR: Raw comparison (use compare_nsys/ncu).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "chapter": {
                    "type": "string",
                    "description": "Chapter name (e.g., 'ch11', 'ch11-streams-comparison') - will find profile dir automatically",
                },
                "profiles_dir": {
                    "type": "string",
                    "description": "Direct path to a profile pair dir or a parent profiles dir (alternative to chapter)",
                },
                "output_html": {
                    "type": "string",
                    "description": "Path to write interactive HTML comparison (optional, great for sharing)",
                    "default": None,
                },
                "pair": {
                    "type": "string",
                    "description": "Profile pair key to select when multiple exist",
                },
            }
        ),
    },
)
def tool_profile_compare(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate flame graph comparison between baseline and optimized profiles."""
    from pathlib import Path
    from core import profile_insights
    from core.perf_core_base import PerformanceCoreBase

    chapter = params.get("chapter")
    profiles_dir_param = params.get("profiles_dir")
    output_html = params.get("output_html")
    pair_key = params.get("pair")
    include_context, context_level = extract_context_opts(params)

    core = PerformanceCoreBase()

    # Resolve the profile directory
    if profiles_dir_param:
        profiles_dir = Path(profiles_dir_param)
    elif chapter:
        profiles_dir = core._find_profile_directory(chapter)
        if not profiles_dir:
            return {
                "error": f"Chapter not found: {chapter}",
                "hint": "Use profile_compare with profiles_dir parameter, or run 'aisp profile compare' to list available chapters",
            }
    else:
        # List available profile pairs
        pairs = core.list_deep_profile_pairs()
        return {
            "available_chapters": [p.get("chapter") for p in pairs.get("pairs", [])],
            "count": pairs.get("count", 0),
            "hint": "Provide chapter parameter to compare profiles. Example: profile_compare(chapter='ch11-streams-comparison')",
        }

    if not profiles_dir or not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)

    pair_health = profile_insights.assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    result = profile_insights.generate_flamegraph_comparison(profiles_dir, pair_key=pair_key)
    if result is None:
        nsys_comparison = profile_insights.compare_nsys_files(profiles_dir, pair_key=pair_key)
        ncu_comparison = profile_insights.compare_ncu_files(profiles_dir, pair_key=pair_key)
        if nsys_comparison is None and ncu_comparison is None:
            return attach_context_if_requested(
                {
                    "error": "No baseline/optimized nsys profiles found",
                    "profiles_dir": str(profiles_dir),
                    "hint": "Profile both baseline and optimized with: nsys profile --stats=true -o <name> python <script>.py",
                    "pair_health": pair_health,
                },
                include_context,
                context_level,
            )
        result = {
            "warning": "No baseline/optimized nsys profiles found for flamegraph comparison.",
            "profiles_dir": str(profiles_dir),
            "pair_health": pair_health,
        }
        if nsys_comparison is not None:
            result["nsys_comparison"] = nsys_comparison
        if ncu_comparison is not None:
            result["ncu_comparison"] = ncu_comparison
        side_by_side_report = profile_insights.generate_side_by_side_report(
            profiles_dir,
            pair_key=pair_key,
            ncu_comparison=ncu_comparison,
        )
        result["side_by_side_report"] = side_by_side_report
        if not side_by_side_report.get("success"):
            result["side_by_side_error"] = side_by_side_report.get("error", "side_by_side_failed")
        return attach_context_if_requested(result, include_context, context_level)

    if result.get("error"):
        result.setdefault("pair_health", pair_health)
        return attach_context_if_requested(result, include_context, context_level)

    nsys_comparison = profile_insights.compare_nsys_files(profiles_dir, pair_key=pair_key)
    if nsys_comparison is not None:
        result["nsys_comparison"] = nsys_comparison

    ncu_comparison = profile_insights.compare_ncu_files(profiles_dir, pair_key=pair_key)
    if ncu_comparison is not None:
        result["ncu_comparison"] = ncu_comparison

    # NEW: Also get metric-level analysis via compare_profiles()
    # This adds improvements/regressions analysis and bottleneck shift detection
    if chapter:
        try:
            metric_comparison = core.compare_profiles(chapter, pair_key=pair_key)
            if metric_comparison and not metric_comparison.get("error"):
                if "metric_analysis" in metric_comparison:
                    result["metric_analysis"] = metric_comparison["metric_analysis"]
                if "ncu_comparison" in metric_comparison and "ncu_comparison" not in result:
                    result["ncu_comparison"] = metric_comparison["ncu_comparison"]
                if "nsys_comparison" in metric_comparison and "nsys_comparison" not in result:
                    result["nsys_comparison"] = metric_comparison["nsys_comparison"]
                if "recommendations" in metric_comparison:
                    result["recommendations"] = metric_comparison["recommendations"]
        except Exception as exc:
            result.setdefault("warnings", []).append(
                f"Metric analysis unavailable for profile_compare chapter={chapter}: {exc}"
            )

    side_by_side_report = profile_insights.generate_side_by_side_report(
        profiles_dir,
        pair_key=pair_key,
        ncu_comparison=result.get("ncu_comparison"),
    )
    result["side_by_side_report"] = side_by_side_report
    if not side_by_side_report.get("success"):
        result["side_by_side_error"] = side_by_side_report.get("error", "side_by_side_failed")

    # Optionally generate HTML
    if output_html:
        from cli.commands.profiling import _generate_comparison_html

        html_content = _generate_comparison_html(result, chapter or profiles_dir.name)
        Path(output_html).write_text(html_content)
        result["html_output"] = output_html

    # Add chapter/directory info
    result["chapter"] = chapter
    result["profiles_dir"] = str(profiles_dir)
    result.setdefault("pair_health", pair_health)

    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "profile_nsys",
    "Tags: nsys, nsight-systems, profile, timeline, trace, cuda-api, nvtx, deep-dive. "
    "Run Nsight Systems profiling to capture GPU timeline, CUDA API calls, kernel launches. "
    "Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. "
    "Returns: {output_path, success, run_details, nsys_metrics} and writes .nsys-rep file. "
    "USE when: Need detailed timeline view, understanding kernel launch patterns, API overhead. "
    'Example: "Profile python train.py with nsys" or "Capture timeline for batch 32 inference". '
    "⚠️ SLOW: 1-10+ minutes depending on workload. ALWAYS use dry_run=true first to preview command. "
    "PRESETS: preset='light' (default) for quick/small traces, preset='full' for comprehensive data. "
    "Reliability knobs: wait_mode='primary' (default) and finalize_grace_seconds help finalize reports on timeout-prone captures. "
    "Startup isolation: sanitize_python_startup=true (default) prefixes a safe sitecustomize shim for profiler subprocesses. "
    "STREAM/OVERLAP STUDIES: set full_timeline=true and add NVTX ranges in the code so overlap is visible. "
    "WORKFLOW: status → profile_nsys(dry_run=true) → profile_nsys → nsys_summary → compare_nsys. "
    "COMPARE: compare_nsys auto-pairs baseline/optimized across subdirectories; pass pair if multiple pairs exist. "
    "FOR QUICK CHECKS: Use hw_speed or profile_kernels instead. NOT FOR: Kernel metrics (use profile_ncu). "
    "Analyze the returned nsys_metrics to explain timeline shifts and driver/API overhead changes.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command to profile (argv list), e.g., ['python', 'train.py', '--batch', '32']",
                },
                "output_name": {
                    "type": "string",
                    "description": "Label for the profile (used in profiles/tools/<tool>/<label>/ and output stem)",
                    "default": "mcp_nsys",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for the artifact directory (default: <timestamp>__profile-nsys__<label>)",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs)",
                    "default": "artifacts/runs",
                },
                "trace_cuda": {
                    "type": "boolean",
                    "default": True,
                    "description": "Trace CUDA API calls",
                },
                "trace_nvtx": {
                    "type": "boolean",
                    "default": True,
                    "description": "Trace NVTX ranges",
                },
                "trace_osrt": {
                    "type": "boolean",
                    "default": True,
                    "description": "Trace OS runtime",
                },
                "full_timeline": {
                    "type": "boolean",
                    "default": False,
                    "description": "Trace cuda-hw, cublas, cusolver, cusparse, cudnn (richer timelines)",
                },
                "trace_forks": {
                    "type": "boolean",
                    "default": False,
                    "description": "Trace child processes before exec",
                },
                "preset": {
                    "type": "string",
                    "description": "NSYS preset: light (default, smaller/faster) or full (adds cuda-hw/cublas/cusolver/cusparse/cudnn + fork tracing)",
                    "enum": ["light", "full"],
                    "default": "light",
                },
                "wait_mode": {
                    "type": "string",
                    "description": "NSYS --wait mode (primary or all). primary reduces finalize hangs on short runs.",
                    "enum": ["primary", "all"],
                    "default": "primary",
                },
                "finalize_grace_seconds": {
                    "type": "number",
                    "description": "Grace window after timeout SIGINT to allow NSYS to finalize a report.",
                    "default": 20.0,
                },
                "force_lineinfo": {
                    "type": "boolean",
                    "description": "Force -lineinfo via NVCC/TORCH_NVCC_FLAGS to improve source mapping",
                    "default": True,
                },
                "sanitize_python_startup": {
                    "type": "boolean",
                    "description": "Prefix a safe sitecustomize shim on PYTHONPATH to isolate profiler runs from host startup hooks",
                    "default": True,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites without running (nsys/ncu/cuda availability, output path)",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the capture without executing (alias: estimate_only)",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Return a job ticket and run capture in background; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 300,
                },
            }
        ),
        "required": ["command"],
    },
)
def tool_profile_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Systems profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command_raw = params.get("command") or []
    command, command_error = _normalize_profile_command(command_raw)
    if not command:
        return make_error("command is required", include_context, context_level)
    if command_error:
        return make_error(command_error, include_context, context_level)

    output_name = params.get("output_name", "mcp_nsys")
    run_dir, profile_dir, run_id = _prepare_profile_run(
        params.get("output_dir"),
        params.get("run_id"),
        "nsys",
        output_name,
    )
    progress_path = _progress_path_in_dir(run_dir, run_id)
    trace_cuda = bool(params.get("trace_cuda", True))
    trace_nvtx = bool(params.get("trace_nvtx", True))
    trace_osrt = bool(params.get("trace_osrt", True))
    full_timeline = bool(params.get("full_timeline", False))
    trace_forks = bool(params.get("trace_forks", False))
    preset = normalize_param("preset", params.get("preset"), "light")
    wait_mode = normalize_param("wait_mode", params.get("wait_mode"), "primary")
    if wait_mode not in {"primary", "all"}:
        return make_error("wait_mode must be 'primary' or 'all'", include_context, context_level)
    finalize_grace_seconds = float(params.get("finalize_grace_seconds", 20.0))
    force_lineinfo = bool(params.get("force_lineinfo", True))
    sanitize_python_startup = bool(params.get("sanitize_python_startup", True))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

    with _profile_workdir_env(CODE_ROOT):
        automation = NsightAutomation(profile_dir)
    cuda_check = _cuda_precheck()
    precheck = {
        "nsys_available": automation.nsys_available,
        "ncu_available": automation.ncu_available,
        "cuda": cuda_check,
        "run_dir": str(run_dir),
        "profiles_dir": str(profile_dir),
        "command_provided": bool(command),
    }

    if precheck_only:
        return {
            "precheck_only": True,
            **precheck,
            "note": "Prereq snapshot only; rerun without precheck_only to capture.",
        }

    if not precheck["command_provided"]:
        return make_error("command is required", include_context, context_level, **precheck)
    if not automation.nsys_available:
        return make_error(
            "nsys is not installed or not on PATH", include_context, context_level, **precheck
        )
    if not cuda_check.get("ok", True):
        return make_error(
            cuda_check.get("reason", "CUDA not available"),
            include_context,
            context_level,
            **precheck,
        )

    output_path = profile_dir / f"{output_name}.nsys-rep"
    if dry_run:
        return {
            "dry_run": True,
            **precheck,
            "preset": preset,
            "wait_mode": wait_mode,
            "finalize_grace_seconds": finalize_grace_seconds,
            "full_timeline": full_timeline or preset == "full",
            "force_lineinfo": force_lineinfo,
            "sanitize_python_startup": sanitize_python_startup,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "planned_output": str(output_path),
            "note": "Set dry_run=false to execute; use async=true to background the run. Default preset is light; set preset=full for richer traces.",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }

    def _execute_capture():
        progress_recorder = ProgressRecorder(run_id=run_id, progress_path=progress_path)
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=1,
                total_phases=2,
                step="nsys",
                step_detail="capture start",
                percent_complete=0.0,
            ),
        )
        with _profile_workdir_env(CODE_ROOT):
            auto = NsightAutomation(profile_dir)
            path = auto.profile_nsys(
                command=command,
                output_name=output_name,
                trace_cuda=trace_cuda,
                trace_nvtx=trace_nvtx,
                trace_osrt=trace_osrt,
                full_timeline=full_timeline,
                trace_forks=trace_forks,
                preset=preset,
                force_lineinfo=force_lineinfo,
                timeout_seconds=timeout_seconds
                if timeout_seconds and timeout_seconds > 0
                else None,
                wait_mode=wait_mode,
                finalize_grace_seconds=finalize_grace_seconds,
                sanitize_python_startup=sanitize_python_startup,
            )
        nsys_metrics: Dict[str, Any] = {}
        nsys_metrics_error = None
        if path:
            try:
                from core.profiling.metrics_extractor import extract_nsys_metrics

                metrics_obj = extract_nsys_metrics(Path(path))
                if hasattr(metrics_obj, "to_dict"):
                    nsys_metrics = metrics_obj.to_dict()
                elif hasattr(metrics_obj, "model_dump"):
                    nsys_metrics = metrics_obj.model_dump()
                else:
                    nsys_metrics = dict(metrics_obj)  # type: ignore[arg-type]
            except Exception as exc:
                nsys_metrics_error = str(exc)
        result = {
            "success": path is not None,
            "output": str(path) if path else None,
            "nsys_available": auto.nsys_available,
            "cwd": str(getattr(auto, "last_run", {}).get("cwd", profile_dir)),
            "preset": preset,
            "wait_mode": wait_mode,
            "finalize_grace_seconds": finalize_grace_seconds,
            "full_timeline": full_timeline or preset == "full",
            "force_lineinfo": force_lineinfo,
            "sanitize_python_startup": sanitize_python_startup,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "timeout_hit": bool(auto.last_run.get("timeout_hit"))
            if hasattr(auto, "last_run")
            else False,  # type: ignore[attr-defined]
            "warning": "NSYS full timeline enabled: captures may run slower and produce large traces; set preset=light to keep it small."
            if preset == "full" or full_timeline
            else "Using light NSYS preset for safer/faster capture.",
            "error": auto.last_error if path is None else None,
            "run_details": getattr(auto, "last_run", {}),  # type: ignore[attr-defined]
            "nsys_metrics": nsys_metrics,
            "suggestions": [
                "Use preset=full only for deep dives; keep light for routine runs.",
                "If disk space is low, set TMPDIR to a directory with >200MB free before capturing.",
                "If capture fails, try preset=light to reduce trace size.",
            ],
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        if nsys_metrics_error:
            result["nsys_metrics_error"] = nsys_metrics_error
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=2,
                total_phases=2,
                step="nsys",
                step_detail="capture complete",
                percent_complete=100.0,
                artifacts=[str(path)] if path else [],
            ),
        )
        return attach_context_if_requested(result, include_context, context_level)

    def _execute_capture_queued(queue_job_id: Optional[str] = None):
        queue_label = f"profile_nsys {output_name}"
        queue_payload = {"output_name": output_name, "run_id": run_id}
        return _run_with_queue(
            "profile_nsys",
            _execute_capture,
            queue_label=queue_label,
            queue_payload=queue_payload,
            job_id=queue_job_id,
        )

    if run_async:
        job_id = f"profile_nsys-{uuid.uuid4().hex[:10]}"

        def _execute_capture_with_job():
            return _execute_capture_queued(job_id)

        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        queued = _queue_job(
            "profile_nsys",
            _execute_capture_with_job,
            params,
            run_metadata=run_metadata,
            job_id=job_id,
        )
        queued["note"] = "Background capture started; poll with job_status using job_id."
        queued["preset"] = preset
        queued["run_id"] = run_id
        queued["queue_log"] = str(_QUEUE_LOG_PATH)
        queued["queue_script"] = str(_QUEUE_SCRIPT_PATH)
        return queued

    return _execute_capture_queued()


@register_tool(
    "profile_ncu",
    "Tags: ncu, nsight-compute, profile, kernel-metrics, occupancy, memory-throughput. "
    "Run Nsight Compute profiling to capture detailed per-kernel metrics (occupancy, throughput, etc.). "
    "Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. "
    "Returns: {output_path, success, run_details, ncu_metrics} and writes .ncu-rep file. "
    "USE when: Deep-diving into specific kernel performance, optimizing occupancy, memory access. "
    'Example: "Profile attention kernel with ncu" or "Get detailed metrics for matmul". '
    "⚠️ VERY SLOW: Replays kernels. Use kernel_filter to limit scope. Use dry_run=true first. "
    "metric_set selects the NCU --set; workload_type picks custom metrics (only when metric_set=full). "
    "workload_type: memory_bound (default, fast), compute_bound, tensor_core. "
    "🕐 SLOW (varies). WORKFLOW: profile_kernels → profile_ncu. NOT FOR: Timeline (use profile_nsys). "
    "DEFAULTS: metric_set='full' by default; use metric_set='minimal' (auto-resolves to speed-of-light/basic by Nsight version) for routine baseline/optimized compares; "
    "use metric_set='roofline' for bound analysis; use metric_set='full' for deep dives. "
    "COMPARE: compare_ncu auto-pairs baseline/optimized across subdirectories; pass pair if multiple pairs exist. "
    "Use launch_skip/launch_count to limit captures on many-launch benchmarks (e.g., 4096 batches). "
    "Analyze ncu_metrics to explain occupancy, throughput, and kernel-time shifts.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command to profile (argv list), e.g., ['python', 'train.py', '--batch', '32']",
                },
                "output_name": {
                    "type": "string",
                    "description": "Label for the profile (used in profiles/tools/<tool>/<label>/ and output stem)",
                    "default": "mcp_ncu",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for the artifact directory (default: <timestamp>__profile-ncu__<label>)",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs)",
                    "default": "artifacts/runs",
                },
                "workload_type": {
                    "type": "string",
                    "description": "Metric list selection for metric_set=full: memory_bound, compute_bound, tensor_core",
                    "enum": ["memory_bound", "compute_bound", "tensor_core"],
                    "default": "memory_bound",
                },
                "kernel_filter": {
                    "type": "string",
                    "description": "Optional kernel name filter (regex)",
                },
                "kernel_name_base": {
                    "type": "string",
                    "description": "Optional NCU kernel name base for filter matching (e.g., function, demangled).",
                },
                "nvtx_include": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional NVTX include filters (repeatable); useful with profile_from_start='off'.",
                    "default": [],
                },
                "profile_from_start": {
                    "type": "string",
                    "description": "NCU profiling gate: on/off. Use off to gate capture until cudaProfilerStart.",
                    "enum": ["on", "off"],
                    "default": None,
                },
                "force_lineinfo": {
                    "type": "boolean",
                    "description": "Force -lineinfo via NVCC/TORCH_NVCC_FLAGS to improve source mapping",
                    "default": True,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites without running (nsys/ncu/cuda availability, output path)",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the capture without executing (alias: estimate_only)",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Return a job ticket and run capture in background; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 300,
                },
                "pm_sampling_interval": {
                    "type": "integer",
                    "description": "Nsight Compute pm-sampling-interval (cycles). Increase to reduce overhead; omit for default.",
                    "default": None,
                },
                "metric_set": {
                    "type": "string",
                    "description": "NCU --set selection: full, roofline, minimal, speed-of-light, basic. workload_type is used only when metric_set=full.",
                    "enum": ["full", "speed-of-light", "roofline", "minimal", "basic"],
                    "default": "full",
                },
                "launch_skip": {
                    "type": "integer",
                    "description": "Number of kernel launches to skip before profiling (prevents timeout on many-launch benchmarks).",
                    "default": None,
                },
                "launch_count": {
                    "type": "integer",
                    "description": "Number of kernel launches to profile (None = all remaining).",
                    "default": None,
                },
                "replay_mode": {
                    "type": "string",
                    "description": "NCU replay mode: application (profile all launches) or kernel (profile one instance per kernel)",
                    "enum": ["application", "kernel"],
                    "default": "application",
                },
            }
        ),
        "required": ["command"],
    },
)
def tool_profile_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Nsight Compute profiling for an arbitrary command."""
    from pathlib import Path
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command_raw = params.get("command") or []
    command, command_error = _normalize_profile_command(command_raw)
    if not command:
        return make_error("command is required", include_context, context_level)
    if command_error:
        return make_error(command_error, include_context, context_level)

    output_name = params.get("output_name", "mcp_ncu")
    run_dir, profile_dir, run_id = _prepare_profile_run(
        params.get("output_dir"),
        params.get("run_id"),
        "ncu",
        output_name,
    )
    progress_path = _progress_path_in_dir(run_dir, run_id)
    workload_type = normalize_param("workload_type", params.get("workload_type"), "memory_bound")
    kernel_filter = params.get("kernel_filter")
    kernel_name_base = params.get("kernel_name_base")
    nvtx_include_raw = params.get("nvtx_include") or []
    if isinstance(nvtx_include_raw, str):
        nvtx_includes = [part.strip() for part in nvtx_include_raw.split(",") if part.strip()]
    else:
        nvtx_includes = [str(part).strip() for part in nvtx_include_raw if str(part).strip()]
    profile_from_start = params.get("profile_from_start")
    if profile_from_start is not None:
        profile_from_start = str(profile_from_start).strip().lower()
        if profile_from_start not in {"on", "off"}:
            return make_error(
                "profile_from_start must be 'on' or 'off'",
                include_context,
                context_level,
            )
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    sampling_param = (
        params.get("pm_sampling_interval")
        if "pm_sampling_interval" in params
        else params.get("sampling_interval")
    )
    sampling_interval = None if sampling_param in (None, "") else int(sampling_param)
    metric_set = params.get("metric_set", "full")
    launch_skip_param = params.get("launch_skip")
    launch_skip = None if launch_skip_param is None else int(launch_skip_param)
    launch_count_param = params.get("launch_count")
    launch_count = None if launch_count_param is None else int(launch_count_param)
    replay_mode = params.get("replay_mode", "application")
    effective_launch_skip = launch_skip
    effective_launch_count = launch_count
    if kernel_filter:
        if effective_launch_count is None:
            effective_launch_count = 1

    with _profile_workdir_env(CODE_ROOT):
        automation = NsightAutomation(profile_dir)
    cuda_check = _cuda_precheck()
    precheck = {
        "nsys_available": automation.nsys_available,
        "ncu_available": automation.ncu_available,
        "cuda": cuda_check,
        "run_dir": str(run_dir),
        "profiles_dir": str(profile_dir),
        "command_provided": bool(command),
        "pm_sampling_interval": sampling_interval,
    }

    if precheck_only:
        return {
            "precheck_only": True,
            **precheck,
            "note": "Prereq snapshot only; rerun without precheck_only to capture.",
        }

    if not precheck["command_provided"]:
        return make_error("command is required", include_context, context_level, **precheck)
    if not automation.ncu_available:
        return make_error(
            "ncu is not installed or not on PATH", include_context, context_level, **precheck
        )
    if not cuda_check.get("ok", True):
        return make_error(
            cuda_check.get("reason", "CUDA not available"),
            include_context,
            context_level,
            **precheck,
        )
    try:
        metric_set_resolved = automation._resolve_ncu_set(metric_set)  # type: ignore[attr-defined]
    except ValueError as exc:
        return make_error(str(exc), include_context, context_level, **precheck)
    except Exception:
        metric_set_resolved = None

    output_path = profile_dir / f"{output_name}.ncu-rep"
    if dry_run:
        return {
            "dry_run": True,
            **precheck,
            "workload_type": workload_type,
            "kernel_filter": kernel_filter,
            "kernel_name_base": kernel_name_base,
            "nvtx_include": nvtx_includes,
            "profile_from_start": profile_from_start,
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "pm_sampling_interval": sampling_interval,
            "metric_set": metric_set,
            "metric_set_resolved": metric_set_resolved,
            "launch_skip": effective_launch_skip,
            "launch_count": effective_launch_count,
            "replay_mode": replay_mode,
            "planned_output": str(output_path),
            "note": "Set dry_run=false to execute; use async=true to background the run.",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }

    def _execute_capture():
        progress_recorder = ProgressRecorder(run_id=run_id, progress_path=progress_path)
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=1,
                total_phases=2,
                step="ncu",
                step_detail="capture start",
                percent_complete=0.0,
            ),
        )
        with _profile_workdir_env(CODE_ROOT):
            auto = NsightAutomation(profile_dir)
            path = auto.profile_ncu(
                command=command,
                output_name=output_name,
                workload_type=workload_type,
                kernel_filter=kernel_filter,
                kernel_name_base=kernel_name_base,
                nvtx_includes=nvtx_includes,
                profile_from_start=profile_from_start,
                force_lineinfo=force_lineinfo,
                timeout_seconds=timeout_seconds
                if timeout_seconds and timeout_seconds > 0
                else None,
                sampling_interval=sampling_interval,
                metric_set=metric_set,
                launch_skip=launch_skip,
                launch_count=launch_count,
                replay_mode=replay_mode,
            )
        run_details = getattr(auto, "last_run", {})  # type: ignore[attr-defined]
        launch_skip_used = run_details.get("launch_skip", launch_skip)
        launch_count_used = run_details.get("launch_count", launch_count)
        replay_mode_used = run_details.get("replay_mode", replay_mode)
        metric_set_resolved_run = run_details.get("metric_set_resolved", metric_set_resolved)
        ncu_metrics: Dict[str, Any] = {}
        ncu_metrics_error = None
        if path:
            try:
                from core.profiling.metrics_extractor import extract_ncu_metrics

                metrics_obj = extract_ncu_metrics(Path(path))
                if hasattr(metrics_obj, "to_dict"):
                    ncu_metrics = metrics_obj.to_dict()
                elif hasattr(metrics_obj, "model_dump"):
                    ncu_metrics = metrics_obj.model_dump()
                else:
                    ncu_metrics = dict(metrics_obj)  # type: ignore[arg-type]
            except Exception as exc:
                ncu_metrics_error = str(exc)

        result = {
            "success": path is not None,
            "output": str(path) if path else None,
            "workload_type": workload_type,
            "ncu_available": auto.ncu_available,
            "cwd": str(run_details.get("cwd", profile_dir)),
            "force_lineinfo": force_lineinfo,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "pm_sampling_interval": sampling_interval,
            "metric_set": metric_set,
            "metric_set_resolved": metric_set_resolved_run,
            "launch_skip": launch_skip_used,
            "launch_count": launch_count_used,
            "replay_mode": replay_mode_used,
            "kernel_filter": kernel_filter,
            "kernel_name_base": kernel_name_base,
            "nvtx_include": nvtx_includes,
            "profile_from_start": profile_from_start,
            "timeout_hit": bool(auto.last_run.get("timeout_hit"))
            if hasattr(auto, "last_run")
            else False,  # type: ignore[attr-defined]
            "error": auto.last_error if path is None else None,
            "run_details": run_details,
            "ncu_metrics": ncu_metrics,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        if ncu_metrics_error:
            result["ncu_metrics_error"] = ncu_metrics_error
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=2,
                total_phases=2,
                step="ncu",
                step_detail="capture complete",
                percent_complete=100.0,
                artifacts=[str(path)] if path else [],
            ),
        )
        return attach_context_if_requested(result, include_context, context_level)

    def _execute_capture_queued(queue_job_id: Optional[str] = None):
        queue_label = f"profile_ncu {output_name}"
        queue_payload = {"output_name": output_name, "run_id": run_id}
        return _run_with_queue(
            "profile_ncu",
            _execute_capture,
            queue_label=queue_label,
            queue_payload=queue_payload,
            job_id=queue_job_id,
        )

    if run_async:
        job_id = f"profile_ncu-{uuid.uuid4().hex[:10]}"

        def _execute_capture_with_job():
            return _execute_capture_queued(job_id)

        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        queued = _queue_job(
            "profile_ncu",
            _execute_capture_with_job,
            params,
            run_metadata=run_metadata,
            job_id=job_id,
        )
        queued["note"] = "Background capture started; poll with job_status using job_id."
        queued["workload_type"] = workload_type
        queued["run_id"] = run_id
        queued["queue_log"] = str(_QUEUE_LOG_PATH)
        queued["queue_script"] = str(_QUEUE_SCRIPT_PATH)
        return queued

    return _execute_capture_queued()


@register_tool(
    "profile_torch",
    "Tags: torch, profiler, pytorch, chrome-trace, autograd, cpu-gpu. "
    "Run PyTorch torch.profiler to capture CPU/GPU activity with Chrome trace output. "
    "Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. "
    "Returns: {trace_path, summary, torch_metrics, success} and writes Chrome trace JSON + summary. "
    "USE when: Profiling PyTorch code specifically, understanding autograd overhead, CPU/GPU interplay. "
    'Example: "Profile my training script with torch.profiler" or "Get PyTorch trace for train.py". '
    "Output viewable in chrome://tracing or Perfetto. Emits NVTX for nsys correlation. 🕐 SLOW (varies). WORKFLOW: profile_torch → analyze operators → optimize. NOT FOR: Timeline (use profile_nsys). "
    "Analyze torch_metrics to identify CPU/GPU hotspots and autograd overhead.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "script": {"type": "string", "description": "Path to Python script to profile"},
                "script_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Args forwarded to the script",
                },
                "output_name": {
                    "type": "string",
                    "description": "Label for the capture (used in profiles/tools/<tool>/<label>/)",
                    "default": "mcp_torch",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for the artifact directory (default: <timestamp>__profile-torch__<label>)",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs)",
                    "default": "artifacts/runs",
                },
                "mode": {
                    "type": "string",
                    "description": "Profiler preset",
                    "enum": ["full", "memory", "flops", "modules", "blackwell"],
                    "default": "full",
                },
                "nvtx_label": {
                    "type": "string",
                    "description": "NVTX/record_function range label",
                    "default": "torch_profile",
                },
                "use_nvtx": {
                    "type": "boolean",
                    "description": "Emit NVTX range around the profiled run",
                    "default": True,
                },
                "force_lineinfo": {
                    "type": "boolean",
                    "description": "Force -lineinfo in NVCC/TORCH_NVCC_FLAGS for better source mapping",
                    "default": True,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prereqs without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the capture without executing (alias: estimate_only)",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Return a job ticket and run capture in background; poll with job_status",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 300,
                },
            }
        ),
        "required": ["script"],
    },
)
def tool_profile_torch(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run torch.profiler for a Python script and return summary + trace paths."""
    from pathlib import Path
    from core.profiling.torch_profiler import TorchProfilerAutomation

    include_context, context_level = extract_context_opts(params)
    script = params.get("script")
    if not script:
        return make_error("script is required", include_context, context_level)

    script_path = Path(script)
    output_name = params.get("output_name") or script_path.stem or "mcp_torch"
    run_dir, profile_dir, run_id = _prepare_profile_run(
        params.get("output_dir"),
        params.get("run_id"),
        "torch",
        output_name,
    )
    progress_path = _progress_path_in_dir(run_dir, run_id)
    mode = normalize_param("torch_mode", params.get("mode"), "full")
    script_args = params.get("script_args") or []
    if isinstance(script_args, str):
        import shlex as _shlex  # local import to avoid global side effects

        script_args = _shlex.split(script_args)
    force_lineinfo = bool(params.get("force_lineinfo", True))
    use_nvtx = bool(params.get("use_nvtx", True))
    nvtx_label = params.get("nvtx_label", "torch_profile")
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

    try:
        import torch  # noqa: F401

        torch_available = True
        cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]
        torch_error = None
    except Exception as exc:  # pragma: no cover - defensive
        torch_available = False
        cuda_available = False
        torch_error = str(exc)

    precheck = {
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "torch_error": torch_error,
        "run_dir": str(run_dir),
        "profiles_dir": str(profile_dir),
        "script_exists": script_path.exists(),
        "force_lineinfo": force_lineinfo,
        "nvtx_label": nvtx_label,
    }
    if precheck_only:
        return {"precheck_only": True, **precheck}
    if not script_path.exists():
        return make_error(f"script not found: {script}", include_context, context_level, **precheck)
    if dry_run:
        planned = profile_dir / f"{output_name}_<timestamp>"
        return {
            "dry_run": True,
            **precheck,
            "planned_output": str(planned),
            "timeout_seconds": timeout_seconds,
            "mode": mode,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }

    def _execute_capture():
        progress_recorder = ProgressRecorder(run_id=run_id, progress_path=progress_path)
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=1,
                total_phases=2,
                step="torch",
                step_detail="capture start",
                percent_complete=0.0,
            ),
        )
        runner = TorchProfilerAutomation(profile_dir)
        result = runner.profile(
            script=script_path,
            output_name=output_name,
            mode=mode,
            script_args=script_args,
            force_lineinfo=force_lineinfo,
            timeout_seconds=timeout_seconds,
            nvtx_label=nvtx_label,
            use_nvtx=use_nvtx,
        )
        result.update({"torch_available": torch_available, "cuda_available": cuda_available})
        result["torch_metrics"] = result.get("summary") or {}
        result["report"] = result.get("summary")
        if not result.get("success"):
            result.setdefault("error", runner.last_error or "torch profiler failed")
        result["run_id"] = run_id
        result["run_dir"] = str(run_dir)
        result["profiles_dir"] = str(profile_dir)
        result["progress_path"] = str(progress_path)
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=2,
                total_phases=2,
                step="torch",
                step_detail="capture complete",
                percent_complete=100.0,
                artifacts=[str(result.get("trace_path"))] if result.get("trace_path") else [],
            ),
        )
        return attach_context_if_requested(result, include_context, context_level)

    def _execute_capture_queued(queue_job_id: Optional[str] = None):
        queue_label = f"profile_torch {output_name}"
        queue_payload = {"output_name": output_name, "run_id": run_id}
        return _run_with_queue(
            "profile_torch",
            _execute_capture,
            queue_label=queue_label,
            queue_payload=queue_payload,
            job_id=queue_job_id,
        )

    if run_async:
        job_id = f"profile_torch-{uuid.uuid4().hex[:10]}"

        def _execute_capture_with_job():
            return _execute_capture_queued(job_id)

        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        queued = _queue_job(
            "profile_torch",
            _execute_capture_with_job,
            params,
            run_metadata=run_metadata,
            job_id=job_id,
        )
        queued["note"] = (
            "Background torch.profiler capture started; poll with job_status using job_id."
        )
        queued["mode"] = mode
        queued["nvtx_label"] = nvtx_label
        queued["run_id"] = run_id
        queued["queue_log"] = str(_QUEUE_LOG_PATH)
        queued["queue_script"] = str(_QUEUE_SCRIPT_PATH)
        return queued

    return _execute_capture_queued()


@register_tool(
    "profile_hta",
    "Tags: hta, holistic-trace, nsys, analysis, gpu-idle, timeline. "
    "Run Nsight Systems capture with HTA (Holistic Trace Analysis) for automated bottleneck detection. "
    "Serialized via the MCP queue runner under artifacts/parallel_runs to prevent overlap. "
    "Returns: {nsys_rep_path, trace_json_path, hta_report_path, analysis_summary, nsys_metrics}. "
    "USE when: Want automated analysis of trace data, finding GPU idle time, communication bottlenecks. "
    'Example: "Profile and analyze with HTA" or "Get holistic trace analysis for my script". '
    "Produces .nsys-rep + trace.json + hta_report.json with actionable insights. 🕐 MEDIUM (~30s). WORKFLOW: profile_torch → operators → optimize. NOT FOR: CUDA-level (use profile_nsys). "
    "Analyze nsys_metrics alongside analysis_summary to pinpoint idle gaps and API overhead.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command to profile (argv list)",
                },
                "output_name": {
                    "type": "string",
                    "description": "Label for the capture (used in profiles/tools/<tool>/<label>/)",
                    "default": "mcp_hta",
                },
                "run_id": {
                    "type": "string",
                    "description": "Run ID for the artifact directory (default: <timestamp>__profile-hta__<label>)",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for run artifacts (default: artifacts/runs)",
                    "default": "artifacts/runs",
                },
                "preset": {
                    "type": "string",
                    "description": "nsys preset",
                    "enum": ["light", "full"],
                    "default": "full",
                },
                "force_lineinfo": {
                    "type": "boolean",
                    "description": "Force -lineinfo for source/line mapping",
                    "default": True,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prereqs without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the capture without executing",
                    "default": False,
                },
                "async": {
                    "type": "boolean",
                    "description": "Run in background and return job_id",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for none",
                    "default": 300,
                },
            }
        ),
        "required": ["command"],
    },
)
def tool_profile_hta(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run nsys + HTA analysis."""
    from pathlib import Path
    from core.profiling.hta_capture import HTACaptureAutomation
    from core.profiling.nsight_automation import NsightAutomation

    include_context, context_level = extract_context_opts(params)
    command_raw = params.get("command") or []
    command_list, command_error = _normalize_profile_command(command_raw)
    if not command_list:
        return make_error("command is required", include_context, context_level)
    if command_error:
        return make_error(command_error, include_context, context_level)

    output_name = params.get("output_name", "mcp_hta")
    run_dir, profile_dir, run_id = _prepare_profile_run(
        params.get("output_dir"),
        params.get("run_id"),
        "hta",
        output_name,
    )
    progress_path = _progress_path_in_dir(run_dir, run_id)
    preset = normalize_param("preset", params.get("preset"), "full")
    force_lineinfo = bool(params.get("force_lineinfo", True))
    precheck_only = bool(params.get("precheck_only"))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    run_async = bool(params.get("async"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)

    with _profile_workdir_env(CODE_ROOT):
        nsight = NsightAutomation(profile_dir)
    try:
        import hta  # noqa: F401

        hta_available = True
    except Exception:
        hta_available = False

    precheck = {
        "nsys_available": nsight.nsys_available,
        "hta_available": hta_available,
        "run_dir": str(run_dir),
        "profiles_dir": str(profile_dir),
        "preset": preset,
        "command_provided": bool(command_list),
        "force_lineinfo": force_lineinfo,
    }
    if precheck_only:
        return {"precheck_only": True, **precheck}
    if not nsight.nsys_available:
        return make_error(
            "nsys is not installed or not on PATH", include_context, context_level, **precheck
        )
    if dry_run:
        base = profile_dir / f"{output_name}.nsys-rep"
        return {
            "dry_run": True,
            **precheck,
            "planned_output": str(base),
            "timeout_seconds": timeout_seconds,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }

    def _execute_capture():
        progress_recorder = ProgressRecorder(run_id=run_id, progress_path=progress_path)
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=1,
                total_phases=2,
                step="hta",
                step_detail="capture start",
                percent_complete=0.0,
            ),
        )
        with _profile_workdir_env(CODE_ROOT):
            runner = HTACaptureAutomation(profile_dir)
            result = runner.capture(
                command=command_list,
                output_name=output_name,
                preset=preset,
                force_lineinfo=force_lineinfo,
                timeout_seconds=timeout_seconds,
            )
        result.update({"hta_available": hta_available, "nsys_available": nsight.nsys_available})
        nsys_metrics: Dict[str, Any] = {}
        nsys_metrics_error = None
        nsys_rep_path = result.get("nsys_rep_path")
        if nsys_rep_path:
            try:
                from core.profiling.metrics_extractor import extract_nsys_metrics

                metrics_obj = extract_nsys_metrics(Path(nsys_rep_path))
                if hasattr(metrics_obj, "to_dict"):
                    nsys_metrics = metrics_obj.to_dict()
                elif hasattr(metrics_obj, "model_dump"):
                    nsys_metrics = metrics_obj.model_dump()
                else:
                    nsys_metrics = dict(metrics_obj)  # type: ignore[arg-type]
            except Exception as exc:
                nsys_metrics_error = str(exc)
        if not result.get("success"):
            result.setdefault("error", runner.last_error or "HTA capture failed")
        result["run_id"] = run_id
        result["run_dir"] = str(run_dir)
        result["profiles_dir"] = str(profile_dir)
        result["progress_path"] = str(progress_path)
        result["nsys_metrics"] = nsys_metrics
        if nsys_metrics_error:
            result["nsys_metrics_error"] = nsys_metrics_error
        _emit_progress_safe(
            progress_recorder,
            ProgressEvent(
                phase="capture",
                phase_index=2,
                total_phases=2,
                step="hta",
                step_detail="capture complete",
                percent_complete=100.0,
                artifacts=[str(result.get("nsys_rep_path"))] if result.get("nsys_rep_path") else [],
            ),
        )
        return attach_context_if_requested(result, include_context, context_level)

    def _execute_capture_queued(queue_job_id: Optional[str] = None):
        queue_label = f"profile_hta {output_name}"
        queue_payload = {"output_name": output_name, "run_id": run_id}
        return _run_with_queue(
            "profile_hta",
            _execute_capture,
            queue_label=queue_label,
            queue_payload=queue_payload,
            job_id=queue_job_id,
        )

    if run_async:
        job_id = f"profile_hta-{uuid.uuid4().hex[:10]}"

        def _execute_capture_with_job():
            return _execute_capture_queued(job_id)

        run_metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "profiles_dir": str(profile_dir),
            "progress_path": str(progress_path),
        }
        queued = _queue_job(
            "profile_hta",
            _execute_capture_with_job,
            params,
            run_metadata=run_metadata,
            job_id=job_id,
        )
        queued["note"] = "Background HTA capture started; poll with job_status using job_id."
        queued["preset"] = preset
        queued["run_id"] = run_id
        queued["queue_log"] = str(_QUEUE_LOG_PATH)
        queued["queue_script"] = str(_QUEUE_SCRIPT_PATH)
        return queued

    return _execute_capture_queued()


@register_tool(
    "export_csv",
    "Tags: export, csv, spreadsheet, data, share. "
    "Export benchmarks to CSV format for spreadsheet analysis or sharing. "
    "Returns: {csv: <csv_string>, detailed: bool}. "
    "USE when: Importing benchmark data into Excel/Sheets, sharing raw numbers. "
    'Example: "Export benchmarks to CSV" or "Get CSV of all results". '
    "detailed=true includes all metrics; false gives summary columns only. 🕐 SLOW (varies). WORKFLOW: run_benchmarks → benchmark_export or export_csv. NOT FOR: PDF/HTML reports (use benchmark_report).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "detailed": {
                    "type": "boolean",
                    "description": "Include all metrics (true) or summary only (false)",
                    "default": False,
                },
            }
        ),
    },
)
def tool_export_csv(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to CSV."""
    from core.engine import get_engine

    detailed = bool(params.get("detailed", False))
    include_context, context_level = extract_context_opts(params)
    export = get_engine().export.csv_detailed() if detailed else get_engine().export.csv()
    result = {"csv": export, "detailed": detailed}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "export_pdf",
    "Tags: export, pdf, report, document, share, print. "
    "Export benchmarks to PDF report format for printing or formal sharing. "
    "Returns: {pdf_base64: <base64_encoded_pdf>}. "
    "USE when: Creating printable reports, formal documentation, sharing with stakeholders. "
    'Example: "Generate PDF report" or "Create printable benchmark summary". '
    "PREFER benchmark_report for more control over report options. 🕐 MEDIUM (~5s). WORKFLOW: run_benchmarks → benchmark_report or export_pdf. NOT FOR: Raw data (use export_csv).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_export_pdf(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to PDF."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    pdf_bytes = get_engine().export.pdf()
    result = {"pdf_base64": pdf_bytes if isinstance(pdf_bytes, str) else str(pdf_bytes)}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "export_html",
    "Tags: export, html, interactive, web, share, visualization. "
    "Export benchmarks to interactive HTML report with charts and tables. "
    "Returns: {html: <html_string>}. "
    "USE when: Sharing interactive web-viewable reports, embedding in documentation. "
    'Example: "Generate HTML report" or "Create interactive benchmark visualization". '
    "PREFER benchmark_report for more control over report options. ⚡ FAST (~2s). WORKFLOW: run_benchmarks → benchmark_report or export_html. NOT FOR: Raw data (use export_csv).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_export_html(params: Dict[str, Any]) -> Dict[str, Any]:
    """Export benchmarks to HTML."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    html = get_engine().export.html()
    result = {"html": html}
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# TEST TOOLS
# =============================================================================


@register_tool(
    "hw_speed",
    "Tags: speed, benchmark, gemm, memory, attention, quick-test, sanity-check. "
    "Run quick GPU speed tests: GEMM throughput, memory bandwidth, attention kernel. "
    "Returns: {tests: [{name, latency_ms, throughput, result}]}. "
    "USE when: Quick sanity check of GPU performance, verifying hardware is working correctly. "
    'Example: "Quick benchmark GPU speed" or "Run GEMM and memory tests". '
    "type='all' runs everything; 'gemm'/'memory'/'attention' for specific tests. "
    "⚠️ Stresses GPU briefly. Use dry_run=true to preview what will run. 🕐 MEDIUM (~15s). WORKFLOW: status → hw_speed → verify GPU health. NOT FOR: Deep profiling (use profile_*).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "type": {
                    "type": "string",
                    "description": "Test selection: all, gemm, memory, or attention",
                    "default": "all",
                    "enum": ["all", "gemm", "memory", "attention"],
                },
                "gemm_size": {"type": "integer", "description": "GEMM size", "default": 512},
                "precision": {
                    "type": "string",
                    "description": "Precision (fp16/bf16/tf32/fp32/fp8)",
                    "enum": ["fp16", "bf16", "tf32", "fp32", "fp8"],
                    "default": "fp16",
                },
                "mem_size_mb": {
                    "type": "integer",
                    "description": "Memory test size MB",
                    "default": 16,
                },
                "mem_stride": {
                    "type": "integer",
                    "description": "Memory stride bytes",
                    "default": 128,
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites and planned command without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the bench invocation without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 300,
                },
            }
        ),
    },
)
def tool_hw_speed(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run speed tests without invoking the bench CLI."""
    from core.diagnostics import microbench

    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    cuda_check = _cuda_precheck()

    test_type = normalize_param("test_type", params.get("type"), "all")
    precision_val = normalize_param("precision", params.get("precision"), "fp16")
    planned = {
        "type": test_type,
        "gemm_size": params.get("gemm_size", 512),
        "precision": precision_val,
        "mem_size_mb": params.get("mem_size_mb", 16),
        "mem_stride": params.get("mem_stride", 128),
    }

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned": planned,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }

    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
        }

    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "planned": planned,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run status first.",
        }

    results: List[Dict[str, Any]] = []

    if test_type in ("all", "gemm"):
        res = microbench.tensor_core_bench(
            size=int(params.get("gemm_size", 512)),
            precision=precision_val,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        res["name"] = "tensor_core_gemm"
        results.append(res)

    if test_type in ("all", "memory"):
        res = microbench.mem_hierarchy_test(
            size_mb=int(params.get("mem_size_mb", 16)),
            stride=int(params.get("mem_stride", 128)),
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        res["name"] = "mem_hierarchy"
        results.append(res)

    if test_type in ("all", "attention"):
        attn: Dict[str, Any] = {"name": "attention_softmax"}
        try:
            import torch

            if torch.cuda.is_available():
                q = torch.randn(256, 256, device="cuda")
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.softmax(q, dim=-1)
                torch.cuda.synchronize()
                attn["latency_ms"] = (time.perf_counter() - start) * 1000
            else:
                attn["error"] = "CUDA not available"
        except Exception as exc:
            attn["error"] = str(exc)
        results.append(attn)

    result = {"tests": results}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_roofline",
    "Tags: roofline, stride, memory, bandwidth, sweep, cache. "
    "Run stride sweep to measure memory bandwidth at different access patterns (roofline data). "
    "Returns: {size_mb, rows: [(stride, bandwidth_gbps), ...]}. "
    "USE when: Understanding memory hierarchy performance, cache behavior, roofline positioning. "
    'Example: "Run roofline stride sweep" or "Measure bandwidth at different strides". '
    "Sweeps strides from 32 to 4096 bytes by default. ⚠️ Stresses memory subsystem. 🕐 MEDIUM (~20s). WORKFLOW: hw_roofline → profile_roofline. NOT FOR: Quick tests (use hw_speed).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size_mb": {"type": "integer", "description": "Buffer size MB", "default": 32},
                "strides": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Stride values",
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites and planned command without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the bench invocation without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 300,
                },
            }
        ),
    },
)
def tool_hw_roofline(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    size_mb = int(params.get("size_mb", 32))
    strides = [int(s) for s in params.get("strides") or []]
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    cuda_check = _cuda_precheck()

    planned = {"size_mb": size_mb, "strides": strides or None}

    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "planned": planned,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return {
            "error": cuda_check.get("reason", "CUDA not available"),
            "cuda": cuda_check,
            "planned": planned,
        }
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "planned": planned,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "Set dry_run=false to execute; run status first.",
        }

    sweep_strides = strides or [32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    for stride in sweep_strides:
        res = microbench.mem_hierarchy_test(
            size_mb=size_mb,
            stride=stride,
            timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        rows.append((stride, res.get("bandwidth_gbps")))

    result = {"size_mb": size_mb, "rows": rows}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_disk",
    "Tags: disk, io, storage, read, write, sequential, throughput. "
    "Run disk I/O benchmark measuring sequential read/write throughput. "
    "Returns: {read_mbps, write_mbps, file_size_mb, block_size_kb}. "
    "USE when: Checking if disk I/O is a bottleneck, verifying storage performance. "
    'Example: "Benchmark disk I/O" or "Is my storage fast enough for checkpointing?". '
    "Writes temp file to tmp_dir (or /tmp). ⚠️ Writes to disk. 🕐 MEDIUM (~10s). WORKFLOW: analyze_dataloader → if IO-bound → hw_disk. NOT FOR: GPU tests (use hw_speed).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "file_size_mb": {"type": "integer", "default": 256},
                "block_size_kb": {"type": "integer", "default": 1024},
                "tmp_dir": {"type": "string"},
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites and paths without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the disk test without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 120,
                },
            }
        ),
    },
)
def tool_test_disk(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    include_context, context_level = extract_context_opts(params)
    tmp_dir = params.get("tmp_dir")
    tmp_path = Path(tmp_dir) if tmp_dir else None
    precheck = {
        "tmp_dir": str(tmp_path) if tmp_path else None,
        "exists": tmp_path.exists() if tmp_path else True,
        "writable": os.access(tmp_path, os.W_OK) if tmp_path and tmp_path.exists() else None,
    }
    if precheck_only:
        return {
            "precheck_only": True,
            "disk": precheck,
            "note": "No data written; rerun without precheck_only to execute.",
        }
    if dry_run:
        return {
            "dry_run": True,
            "disk": precheck,
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
            "note": "No data written; rerun with dry_run=false to execute.",
        }
    if tmp_dir:
        try:
            _ensure_dir(Path(tmp_dir))
        except Exception:
            return make_error(
                f"failed to create tmp_dir: {tmp_dir}",
                include_context,
                context_level,
                tmp_dir=tmp_dir,
            )
    result = microbench.disk_io_test(
        file_size_mb=int(params.get("file_size_mb", 256)),
        block_size_kb=int(params.get("block_size_kb", 1024)),
        tmp_dir=tmp_dir,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_pcie",
    "Tags: pcie, h2d, d2h, host-to-device, bandwidth, transfer. "
    "Run PCIe bandwidth benchmark measuring Host-to-Device and Device-to-Host transfer speeds. "
    "Returns: {h2d_gbps, d2h_gbps, size_mb, iters}. "
    "USE when: Checking PCIe bandwidth, diagnosing data transfer bottlenecks. "
    'Example: "Test PCIe bandwidth" or "How fast is H2D transfer?". '
    "NOT FOR: GPU memory bandwidth (use gpu_bandwidth), GPU-to-GPU (use hw_p2p). 🕐 MEDIUM (~10s). WORKFLOW: hw_pcie → if slow → check PCIe gen/width.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size_mb": {"type": "integer", "default": 256},
                "iters": {"type": "integer", "default": 10},
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the PCIe test without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 120,
                },
            }
        ),
    },
)
def tool_test_pcie(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    include_context, context_level = extract_context_opts(params)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return make_error(
            cuda_check.get("reason", "CUDA not available"),
            include_context,
            context_level,
            cuda=cuda_check,
        )
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "iters": params.get("iters", 10)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.pcie_bandwidth_test(
        size_mb=int(params.get("size_mb", 256)),
        iters=int(params.get("iters", 10)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_cache",
    "Tags: cache, memory, hierarchy, l2, hbm, stride, bandwidth. "
    "Run GPU memory hierarchy test measuring bandwidth at specific stride pattern. "
    "Returns: {bandwidth_gbps, size_mb, stride, achieved_vs_peak_pct}. "
    "USE when: Understanding cache/memory hierarchy effects, optimizing memory access patterns. "
    'Example: "Test L2 cache effect" or "Measure bandwidth at 128-byte stride". '
    "ALSO USE: hw_roofline for full stride sweep. 🕐 MEDIUM (~15s). WORKFLOW: hw_cache → profile_roofline.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size_mb": {"type": "integer", "default": 256},
                "stride": {"type": "integer", "default": 128},
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the test without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 120,
                },
            }
        ),
    },
)
def tool_test_mem_hierarchy(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    include_context, context_level = extract_context_opts(params)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return make_error(
            cuda_check.get("reason", "CUDA not available"),
            include_context,
            context_level,
            cuda=cuda_check,
        )
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size_mb": params.get("size_mb", 256), "stride": params.get("stride", 128)},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.mem_hierarchy_test(
        size_mb=int(params.get("size_mb", 256)),
        stride=int(params.get("stride", 128)),
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_tc",
    "Tags: tensor-core, matmul, gemm, tflops, precision, fp16, bf16, tf32. "
    "Run Tensor Core throughput test measuring matmul performance at different precisions. "
    "Returns: {tflops, latency_ms, size, precision, efficiency_vs_peak_pct}. "
    "USE when: Verifying Tensor Core performance, comparing precision throughput. "
    'Example: "Test Tensor Core TFLOPS" or "Compare FP16 vs BF16 matmul speed". '
    "precision: fp16, bf16, tf32, fp32, fp8 (H100+ only). 🕐 MEDIUM (~15s). WORKFLOW: hw_tc → compare vs expected TFLOPS. NOT FOR: Memory tests (use gpu_bandwidth).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size": {"type": "integer", "default": 4096},
                "precision": {
                    "type": "string",
                    "enum": ["fp16", "bf16", "tf32", "fp32", "fp8"],
                    "default": "fp16",
                },
                "precheck_only": {
                    "type": "boolean",
                    "description": "Return prerequisites without running",
                    "default": False,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Describe the test without executing (alias: estimate_only)",
                    "default": False,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime before returning partial output; set 0/null for no timeout",
                    "default": 120,
                },
            }
        ),
    },
)
def tool_test_tensor_core(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.diagnostics import microbench

    include_context, context_level = extract_context_opts(params)
    precheck_only = bool(params.get("precheck_only", False))
    dry_run = bool(params.get("dry_run") or params.get("estimate_only"))
    timeout_param = params.get("timeout_seconds")
    timeout_seconds = None if timeout_param is None else int(timeout_param)
    precision = normalize_param("precision", params.get("precision"), "fp16")
    cuda_check = _cuda_precheck()
    if precheck_only:
        return {
            "precheck_only": True,
            "cuda": cuda_check,
            "note": "Run status or triage first, then rerun without precheck_only.",
        }
    if not cuda_check.get("ok", True):
        return make_error(
            cuda_check.get("reason", "CUDA not available"),
            include_context,
            context_level,
            cuda=cuda_check,
        )
    if dry_run:
        return {
            "dry_run": True,
            "cuda": cuda_check,
            "params": {"size": params.get("size", 4096), "precision": precision},
            "timeout_seconds": timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        }
    result = microbench.tensor_core_bench(
        size=int(params.get("size", 4096)),
        precision=precision,
        timeout_seconds=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_network",
    "Tags: network, throughput, interconnect, bandwidth, nic. "
    "Run network throughput tests to check NIC and interconnect performance. "
    "Returns: {throughput_gbps, latency_ms, interface_info}. "
    "USE when: Checking host network bandwidth, interconnect performance for distributed training. "
    'Example: "Test network bandwidth" or "Check interconnect speed between nodes". '
    "ALSO USE: system_network for InfiniBand status, hw_nccl for NCCL collectives. 🕐 MEDIUM (~15s). WORKFLOW: hw_network → if slow → check NIC config.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_test_network(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run network tests."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    result = get_engine().test.network()
    return attach_context_if_requested(result, include_context, context_level)


# NOTE: benchmark_targets is defined earlier in this file with bench CLI integration
# This duplicate registration was removed to avoid conflicts.

# =============================================================================
# ADVANCED SYSTEM ANALYSIS TOOLS
# =============================================================================


@register_tool(
    "system_full",
    "Tags: system, full, audit, cpu, memory, container, kernel, comprehensive. "
    "Full system analysis: CPU/memory hierarchy, kernel params, container limits, tuning recommendations. "
    "Returns: {cpu_info, memory_hierarchy, system_params, container_limits, recommendations}. "
    "🕐 MEDIUM (~3s). USE when: Deep environment auditing, diagnosing host-side bottlenecks. "
    'Example: "Full system audit" or "Check host-side performance issues". '
    "WORKFLOW: triage → system_full → apply recommendations. "
    "NOT FOR: Quick checks (use status), GPU-specific (use gpu_info).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_system_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """Full system analysis bundle."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_full_system_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"full system analysis failed: {e}", include_context, context_level)


@register_tool(
    "analyze_memory_patterns",
    "Tags: analyze, memory, warp, bank, coalescing, access, patterns, divergence, conflicts. "
    "Memory access pattern analysis: warp divergence, bank conflicts, memory coalescing. "
    "Returns: {warp_divergence, bank_conflicts, memory_access, recommendations}. "
    "⚡ FAST (~2s). USE when: Debugging memory-bound kernels, optimizing memory access. "
    "Params: analysis_type='all'|'warp'|'bank'|'access' for specific analysis. "
    'Example: "Check for warp divergence" or "Analyze bank conflicts". '
    "WORKFLOW: profile_roofline (check if memory-bound) → analyze_memory_patterns → optimize. "
    "NOT FOR: High-level bottlenecks (use analyze_bottlenecks first).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "analysis_type": {
                    "type": "string",
                    "enum": ["all", "warp", "bank", "access"],
                    "default": "all",
                    "description": "Type of memory analysis: warp divergence, bank conflicts, or memory access patterns",
                },
                "stride": {
                    "type": "integer",
                    "default": 1,
                    "description": "Memory stride for analysis",
                },
            }
        ),
    },
)
def tool_analyze_memory_patterns(params: Dict[str, Any]) -> Dict[str, Any]:
    """Memory access pattern analysis."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    analysis_type = normalize_param("analysis_type", params.get("analysis_type"), "all")
    stride = params.get("stride", 1)

    try:
        core = PerformanceCoreBase()
        result = {}

        if analysis_type in ("all", "warp"):
            result["warp_divergence"] = core.get_warp_divergence()
        if analysis_type in ("all", "bank"):
            result["bank_conflicts"] = core.get_bank_conflicts(stride=stride)
        if analysis_type in ("all", "access"):
            result["memory_access"] = core.get_memory_access_patterns(stride=stride)

        result["success"] = True
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"memory analysis failed: {e}", include_context, context_level)


@register_tool(
    "analyze_dataloader",
    "Tags: analyze, dataloader, data, loading, io, bottleneck, workers, prefetch. "
    "DataLoader bottleneck analysis: worker efficiency, prefetch, throughput. "
    "Returns: {throughput, worker_efficiency, recommendations}. "
    "⚡ FAST (~2s). USE when: Diagnosing data loading bottlenecks in training. "
    'Example: "Is data loading the bottleneck?" or "Check DataLoader efficiency". '
    "WORKFLOW: analyze_bottlenecks → if host-bound → analyze_dataloader → tune num_workers/prefetch. "
    "COMMON FIXES: Increase num_workers, enable pin_memory, use persistent_workers.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_analyze_dataloader(params: Dict[str, Any]) -> Dict[str, Any]:
    """DataLoader bottleneck analysis."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_data_loading_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"dataloader analysis failed: {e}", include_context, context_level)


@register_tool(
    "analyze_comm_overlap",
    "Tags: analyze, communication, compute, overlap, distributed, efficiency, allreduce. "
    "Communication/compute overlap analysis for distributed training. "
    "Returns: {overlap_efficiency, compute_time, comm_time, recommendations}. "
    "⚡ FAST (~2s). USE when: Optimizing distributed training efficiency. "
    'Example: "Is communication overlapping compute?" or "Check allreduce overlap". '
    "WORKFLOW: distributed_plan → analyze_comm_overlap → distributed_nccl. "
    "NOT FOR: Single-GPU training (no communication to overlap).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model": {
                    "type": "string",
                    "default": "llama-3.1-70b",
                    "description": "Model name for analysis",
                },
            }
        ),
    },
)
def tool_analyze_comm_overlap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Communication overlap analysis."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    model = params.get("model", "llama-3.1-70b")
    try:
        core = PerformanceCoreBase()
        result = core.get_comm_overlap_analysis(model=model)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"comm overlap analysis failed: {e}", include_context, context_level)


@register_tool(
    "cost_estimate",
    "Tags: cost, cloud, estimate, pricing, aws, gcp, azure, budget, billing. "
    "Cloud cost estimation for GPU fleets. "
    "Returns: {selection, cloud_comparison, recommendation, warnings}. "
    "⚡ FAST (~1s). USE when: Planning cloud deployments, comparing providers. "
    'Example: "Estimate 4xH100 monthly cost" or "Compare AWS vs GCP pricing". '
    "WORKFLOW: distributed_plan → cost_estimate → choose provider. "
    "ALSO USE: analyze_energy for power/efficiency considerations.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "gpu_type": {
                    "type": "string",
                    "default": "h100",
                    "description": "GPU type (h100, a100, etc.)",
                },
                "num_gpus": {"type": "integer", "default": 8, "description": "Number of GPUs"},
                "hours_per_day": {
                    "type": "number",
                    "default": 8,
                    "description": "Usage hours per day",
                },
            }
        ),
    },
)
def tool_cost_estimate(params: Dict[str, Any]) -> Dict[str, Any]:
    """Cloud cost estimation."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_cloud_cost_estimate(params)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"cost estimate failed: {e}", include_context, context_level)


@register_tool(
    "analyze_energy",
    "Tags: analyze, energy, power, efficiency, green, carbon, sustainability. "
    "Energy efficiency analysis: power consumption, efficiency metrics, green recommendations. "
    "Returns: {power_draw, efficiency_score, carbon_estimate, recommendations}. "
    "⚡ FAST (~2s). USE when: Optimizing for energy efficiency, reducing carbon footprint. "
    'Example: "What\'s my energy efficiency?" or "Estimate carbon footprint". '
    "WORKFLOW: gpu_power → analyze_energy → apply power-saving recommendations. "
    "ALSO USE: cost_estimate for cloud cost implications.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_analyze_energy(params: Dict[str, Any]) -> Dict[str, Any]:
    """Energy efficiency analysis."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_energy_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"energy analysis failed: {e}", include_context, context_level)


@register_tool(
    "launch_plan",
    "Tags: launch, torchrun, srun, distributed, command, script, multi-node. "
    "Generate launch commands for distributed training (torchrun, srun, etc.). "
    "Returns: {torchrun_cmd, srun_cmd, env_vars, tips}. "
    "⚡ FAST (~1s). USE when: Setting up distributed training launch. "
    'Example: "Generate torchrun command for 2 nodes" or "Slurm launch script". '
    "WORKFLOW: distributed_plan → launch_plan → run training. "
    "ALSO USE: cluster_slurm for full SLURM job scripts.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "nodes": {"type": "integer", "default": 1, "description": "Number of nodes"},
                "gpus_per_node": {"type": "integer", "default": 8, "description": "GPUs per node"},
                "script": {"type": "string", "description": "Training script path"},
            }
        ),
    },
)
def tool_launch_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate launch plan."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_launch_plan(params)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"launch plan failed: {e}", include_context, context_level)


@register_tool(
    "info_features",
    "Tags: info, features, capabilities, tma, clusters, hopper. "
    "GPU feature detection: TMA, thread block clusters, async copy, etc. "
    "Returns: {features: {tma, clusters, async_copy, ...}, compute_capability}. "
    "⚡ FAST (~1s). USE when: Checking advanced GPU feature support. WORKFLOW: info_features → choose optimizations. NOT FOR: Basic GPU info (use gpu_info).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_info_features(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPU feature detection."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        result = core.get_hardware_capabilities()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"feature detection failed: {e}", include_context, context_level)


@register_tool(
    "profile_list_profiles",
    "Tags: profiles, list, discovery, baseline, optimized, nsys, ncu. "
    "List available baseline/optimized profile pairs discovered under artifacts. "
    "Returns: {pairs: [...], count}. "
    "⚡ FAST (~1s). USE when: You want a machine-readable list of captured profile pairs for follow-up compare tools. "
    "WORKFLOW: profile_list_profiles → compare_nsys/compare_ncu/profile_compare.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_list_profiles(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    try:
        result = get_engine().profile.list_profiles()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"profile list failed: {e}", include_context, context_level)


@register_tool(
    "profile_compile_analysis",
    "Tags: torch.compile, inductor, dynamo, graphs, breaks, analysis. "
    "Summarize torch.compile behavior from the latest benchmark results (graphs, breaks, time in compile). "
    "Returns: {summary, graphs, breaks, ...} (best-effort; depends on captured compile artifacts). "
    "⚡ FAST (~1s). USE when: Investigating compile overhead or graph breaks affecting performance. "
    "WORKFLOW: profile_compile_analysis → profile_torch/profile_nsys for deeper attribution.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_profile_compile_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    try:
        result = get_engine().profile.compile_analysis()
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"compile analysis failed: {e}", include_context, context_level)


@register_tool(
    "ncu_summary",
    "Tags: ncu, nsight-compute, summary, kernels, top, hotspots. "
    "Summarize an existing Nsight Compute report, returning the top-N kernels with key utilization metrics. "
    "Returns: {kernels:[...], kernel_count, sort_by, total_time_sum_ms?}. "
    '⚡ FAST (~1-10s). USE when: You want a quick "what kernel should I tune next?" table from an .ncu-rep. '
    "WORKFLOW: ncu_summary → profile_ncu (scoped) → compare_ncu.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "report_path": {
                    "type": "string",
                    "description": "Path to .ncu-rep or exported NCU raw CSV",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of kernels to return",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit NCU metric columns to request/parse",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 60,
                    "description": "Timeout for `ncu --import` when report_path is .ncu-rep",
                },
            }
        ),
    },
)
def tool_ncu_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    report_path = params.get("report_path")
    if not report_path:
        return make_error("report_path is required", include_context, context_level)

    metrics = params.get("metrics")
    metrics_list: Optional[List[str]] = None
    if isinstance(metrics, list):
        metrics_list = [str(x) for x in metrics if str(x).strip()]

    try:
        result = get_engine().profile.ncu_summary(
            str(report_path),
            top_k=int(params.get("top_k", 10) or 10),
            metrics=metrics_list,
            timeout_seconds=int(params.get("timeout_seconds", 60) or 60),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"ncu summary failed: {e}", include_context, context_level)


@register_tool(
    "nsys_summary",
    "Tags: nsys, nsight, summary, quick, stats. "
    "Quick Nsight Systems summary stats without full profile capture. "
    "Returns: {metrics: [...], count, report_path}. "
    "⚡ FAST (~3s). USE when: Quick nsys stats without full profiling. WORKFLOW: profile_nsys → nsys_summary. NOT FOR: Full profiling (use profile_nsys).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "report_path": {
                    "type": "string",
                    "description": "Path to existing .nsys-rep file to summarize",
                },
            }
        ),
    },
)
def tool_nsys_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Quick nsys summary."""
    from core.perf_core_base import PerformanceCoreBase

    include_context, context_level = extract_context_opts(params)
    try:
        core = PerformanceCoreBase()
        report_path = params.get("report_path")
        result = core.summarize_nsys_report(report_path)
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nsys summary failed: {e}", include_context, context_level)


@register_tool(
    "predict_scaling",
    "Tags: predict, scaling, multi-gpu, efficiency, projection, planning. "
    "Predict performance scaling to more GPUs/larger batches. "
    "Returns: {current_perf, predicted_perf, scaling_efficiency, bottleneck_at_scale}. "
    "⚡ FAST (~2s). USE when: Planning scale-up, predicting multi-GPU performance. "
    'Example: "Predict 4-GPU scaling" or "How will throughput scale from 1 to 4 GPUs?". '
    "WORKFLOW: gpu_topology → predict_scaling → distributed_plan. "
    "NOTE: Scaling efficiency typically 80-90% for well-optimized workloads.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "target_gpus": {"type": "integer", "default": 8, "description": "Target GPU count"},
                "current_gpus": {
                    "type": "integer",
                    "default": 1,
                    "description": "Current GPU count",
                },
                "model_size": {"type": "number", "description": "Model size in billions"},
            }
        ),
    },
)
def tool_predict_scaling(params: Dict[str, Any]) -> Dict[str, Any]:
    """Predict scaling performance."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    try:
        engine = get_engine()
        # Use analyze.scaling with prediction
        result = engine.analyze.scaling()
        result["prediction"] = {
            "target_gpus": params.get("target_gpus", 8),
            "current_gpus": params.get("current_gpus", 1),
            "estimated_speedup": min(
                params.get("target_gpus", 8) / max(params.get("current_gpus", 1), 1) * 0.85, 7.5
            ),
            "note": "Scaling efficiency typically 80-90% for well-optimized workloads",
        }
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"scaling prediction failed: {e}", include_context, context_level)


@register_tool(
    "gpu_topology_matrix",
    "Tags: topology, nvlink, pcie, numa, raw, nvidia-smi, matrix. "
    "Get raw GPU/NUMA topology matrix directly from nvidia-smi topo -m. "
    "Returns: {stdout: <nvidia-smi topo -m output>, returncode}. "
    "⚡ FAST (~1s). USE when: Need exact nvidia-smi topology output format. "
    'Example: "Show raw nvidia-smi topo output" or "Get NVLink matrix raw". '
    "WORKFLOW: gpu_topology_matrix → parse manually if needed. "
    "NOT FOR: Parsed topology (use gpu_topology).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_gpu_topology_matrix(params: Dict[str, Any]) -> Dict[str, Any]:
    """Raw GPU topology matrix from nvidia-smi."""
    include_context, context_level = extract_context_opts(params)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5
        )
        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "success": proc.returncode == 0,
        }
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nvidia-smi topo failed: {e}", include_context, context_level)


@register_tool(
    "compare_nsys",
    "Tags: compare, nsys, nsight-systems, baseline, optimized, diff. "
    "Compare baseline vs optimized Nsight Systems reports. "
    "Returns: {metrics, baseline_file, optimized_file, ncu_comparison?, side_by_side_report}. "
    "If paired NCU profiles are present, also emits a side-by-side JSON report + narrative. "
    "🕐 MEDIUM (~5s). USE when: Comparing before/after nsys profiles. "
    "Tip: if you used benchmark_deep_dive_compare, pass benchmarks[].profiles_dir here. "
    "Auto-pairs baseline/optimized across subdirectories; if multiple pairs exist, provide pair to select one. "
    "Always returns ncu/nsys comparison metrics when profiles are captured; analyze metric deltas to explain speedups/regressions. "
    "WORKFLOW: profile_nsys → optimize → compare_nsys. NOT FOR: Kernel metrics (use compare_ncu).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "profiles_dir": {
                    "type": "string",
                    "description": "Directory with baseline/optimized .nsys-rep files (pair dir or a parent dir; use pair to select a sub-pair)",
                },
                "pair": {
                    "type": "string",
                    "description": "Profile pair key to select when multiple exist",
                },
            }
        ),
        "required": ["profiles_dir"],
    },
)
def tool_compare_nsys(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Systems profiles."""
    from pathlib import Path
    from core import profile_insights

    include_context, context_level = extract_context_opts(params)
    profiles_dir = Path(params.get("profiles_dir", ""))
    pair_key = params.get("pair")
    if not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)
    pair_health = profile_insights.assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    try:
        result = profile_insights.compare_nsys_files(profiles_dir, pair_key=pair_key)
        if result is None:
            result = {
                "error": "No comparable nsys files found",
                "success": False,
                "pair_health": pair_health,
            }
        else:
            result.setdefault("pair_health", pair_health)
        ncu_comparison = profile_insights.compare_ncu_files(profiles_dir, pair_key=pair_key)
        if ncu_comparison is not None:
            result["ncu_comparison"] = ncu_comparison
        side_by_side_report = profile_insights.generate_side_by_side_report(
            profiles_dir,
            pair_key=pair_key,
            ncu_comparison=result.get("ncu_comparison"),
        )
        result["side_by_side_report"] = side_by_side_report
        if not side_by_side_report.get("success"):
            result["side_by_side_error"] = side_by_side_report.get("error", "side_by_side_failed")
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(
            f"nsys comparison failed: {e}",
            include_context,
            context_level,
            pair_health=pair_health,
        )


@register_tool(
    "compare_ncu",
    "Tags: compare, ncu, nsight-compute, baseline, optimized, kernel-metrics. "
    "Compare baseline vs optimized Nsight Compute kernel metrics. "
    "Returns: {kernel_comparison | metrics, baseline_file, optimized_file, nsys_comparison?, side_by_side_report}. "
    "If paired NSYS profiles are present, also emits a side-by-side JSON report + narrative. "
    "🕐 MEDIUM (~5s). USE when: Deep-diving into kernel-level improvements. "
    "Tip: if you used benchmark_deep_dive_compare, pass benchmarks[].profiles_dir here. "
    "Auto-pairs baseline/optimized across subdirectories; if multiple pairs exist, provide pair to select one. "
    "Rank-only kernel symbol alignment is flagged as low-confidence for tuning and surfaced with an advisory warning. "
    "Always returns nsys/ncu comparison metrics when profiles are captured; analyze metric deltas to explain speedups/regressions. "
    "WORKFLOW: profile_ncu → optimize → compare_ncu. NOT FOR: Timeline comparison (use compare_nsys).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "profiles_dir": {
                    "type": "string",
                    "description": "Directory with baseline/optimized .ncu-rep files (pair dir or a parent dir; use pair to select a sub-pair)",
                },
                "pair": {
                    "type": "string",
                    "description": "Profile pair key to select when multiple exist",
                },
            }
        ),
        "required": ["profiles_dir"],
    },
)
def tool_compare_ncu(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare Nsight Compute profiles."""
    from pathlib import Path
    from core import profile_insights

    include_context, context_level = extract_context_opts(params)
    profiles_dir = Path(params.get("profiles_dir", ""))
    pair_key = params.get("pair")
    if not profiles_dir.exists():
        return make_error(f"profiles_dir not found: {profiles_dir}", include_context, context_level)
    pair_health = profile_insights.assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    try:
        result = profile_insights.compare_ncu_files(profiles_dir, pair_key=pair_key)
        if result is None:
            result = {
                "error": "No comparable ncu files found",
                "success": False,
                "pair_health": pair_health,
            }
        elif isinstance(result, dict):
            result.setdefault("pair_health", pair_health)
            alignment_valid = bool(result.get("alignment_valid_for_tuning", True))
            if not alignment_valid:
                result.setdefault("success", True)
                result["advisory_warning"] = (
                    "NCU kernel comparison alignment is low-confidence (rank-only/unmatched symbols). "
                    "Per-kernel deltas are advisory; rely on aggregate metrics and re-capture with tighter filters for decisive tuning."
                )
                result.setdefault(
                    "hint",
                    "Use profile_ncu with kernel_filter + nvtx_include and compare a concrete baseline/optimized pair directory.",
                )
        nsys_comparison = profile_insights.compare_nsys_files(profiles_dir, pair_key=pair_key)
        if nsys_comparison is not None:
            result["nsys_comparison"] = nsys_comparison
        side_by_side_report = profile_insights.generate_side_by_side_report(
            profiles_dir,
            pair_key=pair_key,
            ncu_comparison=result,
        )
        result["side_by_side_report"] = side_by_side_report
        if not side_by_side_report.get("success"):
            result["side_by_side_error"] = side_by_side_report.get("error", "side_by_side_failed")
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(
            f"ncu comparison failed: {e}",
            include_context,
            context_level,
            pair_health=pair_health,
        )


@register_tool(
    "list_chapters",
    "Tags: chapters, labs, list, discovery, book, curriculum. "
    "List all discoverable chapters and labs from the book curriculum. "
    "Returns: {chapters: [{name, path, description}], labs: [...]}. "
    "⚡ FAST (~1s). USE when: Exploring what content is available. WORKFLOW: list_chapters → benchmark_targets → run_benchmarks. NOT FOR: Running benchmarks (use run_benchmarks).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_list_chapters(params: Dict[str, Any]) -> Dict[str, Any]:
    """List available chapters and labs."""
    include_context, context_level = extract_context_opts(params)
    result = _run_bench_cli(["list-chapters"])
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "context_summary",
    "Tags: context, summary, quick, environment, snapshot. "
    "Get quick context summary: GPU + software snapshot. "
    "Returns: {gpu, software, dependencies}. "
    "⚡ FAST (~1s). USE when: Need lightweight context attachment. "
    'Example: "Quick system snapshot" or "Get context for LLM analysis". '
    "NOT FOR: Full details (use context_full or system_full).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_context_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get context summary."""
    include_context, context_level = extract_context_opts(params)
    result = {"success": True, "context": get_cached_context("summary")}
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "context_full",
    "Tags: context, full, comprehensive, environment, dump. "
    "Get full comprehensive context: complete system state. "
    "Returns: {gpu, software, dependencies, capabilities, system_params}. "
    "🕐 MEDIUM (~3s). USE when: Need complete environment dump. "
    'Example: "Full context for debugging" or "Complete system state". '
    "NOT FOR: Quick checks (use context_summary or status).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_context_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get full context."""
    include_context, context_level = extract_context_opts(params)
    result = {"success": True, "context": get_cached_context("full")}
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# HARDWARE MICRO-BENCHMARK TOOLS
# =============================================================================


@register_tool(
    "hw_ib",
    "Tags: infiniband, ib, bandwidth, rdma, multi-node, interconnect. "
    "Get InfiniBand bandwidth test instructions and check if ib_write_bw is available. "
    "Returns: {ib_write_bw_available, instructions: {server_cmd, client_cmd}, alternative}. "
    "USE when: Testing InfiniBand bandwidth, verifying multi-node interconnect performance. "
    'Example: "How do I test InfiniBand bandwidth?" or "Is IB working correctly?". '
    "Provides ib_write_bw commands; alternative is NCCL tests if perftest not installed. ⚡ FAST (~1s). WORKFLOW: hw_ib → hw_nccl. NOT FOR: Single-node (use hw_p2p).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size_mb": {
                    "type": "integer",
                    "description": "Transfer size in MB for test guidance",
                    "default": 64,
                },
            }
        ),
    },
)
def tool_hw_ib(params: Dict[str, Any]) -> Dict[str, Any]:
    """InfiniBand bandwidth test guidance."""
    import shutil

    include_context, context_level = extract_context_opts(params)

    ib_write_bw = shutil.which("ib_write_bw")
    result = {
        "success": True,
        "ib_write_bw_available": ib_write_bw is not None,
        "instructions": {
            "server": "ib_write_bw -d mlx5_0",
            "client": "ib_write_bw -d mlx5_0 <server_ip>",
            "install": "apt install perftest  # or yum install perftest",
        },
        "alternative": "Use NCCL tests: all_reduce_perf -b 8M -e 256M -g 8",
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_nccl",
    "Tags: nccl, collective, allreduce, bandwidth, multi-gpu, communication. "
    "Get NCCL collective bandwidth test command and check if nccl-tests is available. "
    "Returns: {tool_available, command, install_instructions, collectives_list}. "
    "USE when: Measuring collective communication bandwidth, benchmarking NCCL performance. "
    'Example: "Test NCCL all_reduce bandwidth" or "Benchmark 4-GPU allreduce". '
    "Collectives: all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall. ⚡ FAST (~1s). WORKFLOW: hw_nccl → tune NCCL env vars. NOT FOR: IB hardware (use hw_ib).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "collective": {
                    "type": "string",
                    "description": "Collective type: all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall",
                    "enum": [
                        "all_reduce",
                        "all_gather",
                        "reduce_scatter",
                        "broadcast",
                        "reduce",
                        "alltoall",
                    ],
                    "default": "all_reduce",
                },
                "min_bytes": {
                    "type": "string",
                    "description": "Minimum message size (e.g., '8M')",
                    "default": "8M",
                },
                "max_bytes": {
                    "type": "string",
                    "description": "Maximum message size (e.g., '256M')",
                    "default": "256M",
                },
                "gpus": {
                    "type": "integer",
                    "description": "Number of GPUs to test with",
                    "default": 8,
                },
            }
        ),
    },
)
def tool_hw_nccl(params: Dict[str, Any]) -> Dict[str, Any]:
    """NCCL collective bandwidth test guidance."""
    import shutil

    include_context, context_level = extract_context_opts(params)

    collective = normalize_param("collective", params.get("collective"), "all_reduce")
    bin_name = f"{collective}_perf"
    bin_path = shutil.which(bin_name)

    result = {
        "success": True,
        "tool_available": bin_path is not None,
        "command": f"{bin_name} -b {params.get('min_bytes', '8M')} -e {params.get('max_bytes', '256M')} -g {params.get('gpus', 8)}",
        "install": "git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make MPI=1",
        "collectives": [
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "broadcast",
            "reduce",
            "alltoall",
        ],
    }
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "hw_p2p",
    "Tags: p2p, nvlink, gpu-to-gpu, bandwidth, peer-to-peer, transfer. "
    "Run GPU-to-GPU P2P bandwidth test measuring NVLink or PCIe peer access speed. "
    "Returns: {results: [{src, dst, p2p_enabled, bandwidth_gbps}], gpu_count}. "
    "USE when: Verifying NVLink bandwidth, checking P2P connectivity, debugging tensor parallelism. "
    'Example: "Test GPU P2P bandwidth" or "Is NVLink working at full speed?". '
    "REQUIRES: At least 2 GPUs. Tests first GPU pair by default. 🕐 MEDIUM (~20s). WORKFLOW: gpu_topology → hw_p2p. NOT FOR: Host-device (use hw_pcie).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "size_mb": {
                    "type": "integer",
                    "description": "Transfer size in MB",
                    "default": 256,
                },
            }
        ),
    },
)
def tool_hw_p2p(params: Dict[str, Any]) -> Dict[str, Any]:
    """GPU P2P bandwidth test."""
    try:
        import torch

        include_context = bool(params.get("include_context", False))
        context_level = params.get("context_level", "summary")

        if not torch.cuda.is_available():
            return {"success": False, "error": "CUDA not available"}

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            return {
                "success": False,
                "error": "P2P test requires at least 2 GPUs",
                "gpu_count": num_gpus,
            }

        size_mb = params.get("size_mb", 256)
        size_bytes = size_mb * 1024 * 1024

        results = []
        for i in range(min(num_gpus, 2)):  # Test first pair only for speed
            for j in range(min(num_gpus, 2)):
                if i == j:
                    continue

                can_access = torch.cuda.can_device_access_peer(i, j)

                with torch.cuda.device(i):
                    src = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{i}")
                with torch.cuda.device(j):
                    dst = torch.empty(size_bytes // 4, dtype=torch.float32, device=f"cuda:{j}")

                dst.copy_(src)
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                iters = 5
                start.record()
                for _ in range(iters):
                    dst.copy_(src)
                end.record()
                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end)
                bw_gbps = (size_bytes * iters / (elapsed_ms / 1000)) / 1e9

                results.append(
                    {
                        "src": i,
                        "dst": j,
                        "p2p_enabled": can_access,
                        "bandwidth_gbps": round(bw_gbps, 2),
                    }
                )

        result = {"success": True, "gpu_count": num_gpus, "results": results}
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# CLUSTER TOOLS
# =============================================================================


@register_tool(
    "cluster_slurm",
    "Tags: slurm, batch, cluster, hpc, job-script, sbatch, multi-node. "
    "Generate SLURM job script for cluster submission with optimal settings. "
    "Returns: {script: <slurm_script_content>, filename_suggestion, notes}. "
    "USE when: Submitting training jobs to SLURM clusters, setting up multi-node runs. "
    'Example: "Create SLURM script for 2 nodes x 4 GPUs" or "Generate sbatch for 70B training". '
    "Includes: resource requests, NCCL env vars, torchrun launch command. ⚡ FAST (~1s). WORKFLOW: distributed_plan → cluster_slurm → submit. NOT FOR: torchrun only (use launch_plan).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "model": {
                    "type": "string",
                    "description": "Model size for resource estimation (e.g., '7b', '70b')",
                    "default": "7b",
                },
                "nodes": {
                    "type": "integer",
                    "description": "Number of nodes to request",
                    "default": 1,
                },
                "gpus": {"type": "integer", "description": "GPUs per node", "default": 8},
            }
        ),
    },
)
def tool_cluster_slurm(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SLURM script."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)
    # Engine exposes SLURM generation under the distributed domain.
    result = get_engine().distributed.slurm(
        model=params.get("model", "7b"),
        nodes=params.get("nodes", 1),
        gpus=params.get("gpus", 8),
        framework="pytorch",
    )
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "cluster_eval_suite",
    "Tags: cluster, eval, suite, field-report, discovery, reproducibility. "
    "Run the cluster field-report eval suite, or a fast local smoke run that captures metadata + manifest. "
    "Returns: {success, mode, run_id, stdout/stderr, paths}. "
    "🕐 MEDIUM (varies). USE when: You want a reproducible cluster evaluation bundle under `cluster/results/`. "
    "Defaults to mode='smoke' for safety; use mode='full' with hosts to run the full suite.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "mode": {
                    "type": "string",
                    "enum": ["smoke", "full"],
                    "default": "smoke",
                    "description": "smoke (local) or full (runs cluster suite)",
                },
                "run_id": {"type": "string", "description": "RUN_ID prefix (default: YYYY-MM-DD)"},
                "hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Host list (required for mode=full)",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels (must match hosts count)",
                },
                "ssh_user": {"type": "string", "description": "SSH user (full mode)"},
                "ssh_key": {"type": "string", "description": "SSH key path (full mode)"},
                "oob_if": {
                    "type": "string",
                    "description": "Out-of-band interface (full mode, multi-node)",
                },
                "socket_ifname": {
                    "type": "string",
                    "description": "NCCL socket interface (full mode)",
                },
                "nccl_ib_hca": {
                    "type": "string",
                    "description": "NCCL_IB_HCA allowlist (full mode)",
                },
                "primary_label": {
                    "type": "string",
                    "description": "Label for single-node/local steps",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra args appended to run_cluster_eval_suite.sh",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (0/null = no timeout)",
                },
            }
        ),
    },
)
def tool_cluster_eval_suite(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_eval_suite

    include_context, context_level = extract_context_opts(params)
    try:
        result = run_cluster_eval_suite(
            mode=str(params.get("mode", "smoke")),
            run_id=params.get("run_id"),
            hosts=params.get("hosts") if isinstance(params.get("hosts"), list) else None,
            labels=params.get("labels") if isinstance(params.get("labels"), list) else None,
            ssh_user=params.get("ssh_user"),
            ssh_key=params.get("ssh_key"),
            oob_if=params.get("oob_if"),
            socket_ifname=params.get("socket_ifname"),
            nccl_ib_hca=params.get("nccl_ib_hca"),
            primary_label=params.get("primary_label"),
            extra_args=params.get("extra_args")
            if isinstance(params.get("extra_args"), list)
            else None,
            timeout_seconds=params.get("timeout_seconds"),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"cluster eval suite failed: {e}", include_context, context_level)


@register_tool(
    "cluster_common_eval",
    "Tags: cluster, eval, preset, llm, system-validation, reproducibility. "
    "Run a preset benchmark bundle that answers the common system-evaluation questions people usually ask. "
    "Presets: common-answer-fast, core-system, modern-llm, fabric-systems, multinode-readiness. "
    "Returns: {success, preset, artifact_roles, run_id, stdout/stderr}. "
    "🕐 MEDIUM (varies). USE when: You want one obvious entrypoint for standard evaluation output instead of composing suite flags manually.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "preset": {
                    "type": "string",
                    "enum": [
                        "common-answer-fast",
                        "core-system",
                        "modern-llm",
                        "fabric-systems",
                        "multinode-readiness",
                    ],
                    "default": "core-system",
                    "description": "Preset evaluation bundle",
                },
                "run_id": {"type": "string", "description": "RUN_ID prefix (default: YYYY-MM-DD)"},
                "hosts": {"type": "array", "items": {"type": "string"}, "description": "Host list"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels (must match hosts count)",
                },
                "ssh_user": {"type": "string", "description": "SSH user"},
                "ssh_key": {"type": "string", "description": "SSH key path"},
                "oob_if": {
                    "type": "string",
                    "description": "Out-of-band interface for multi-node readiness and network-heavy presets",
                },
                "socket_ifname": {"type": "string", "description": "NCCL socket interface"},
                "nccl_ib_hca": {"type": "string", "description": "NCCL_IB_HCA allowlist"},
                "nmx_url": {
                    "type": "string",
                    "description": "NMX base URL for fabric/NVLink management-plane checks",
                },
                "nmx_token": {"type": "string", "description": "Optional NMX bearer token"},
                "ib_mgmt_host": {
                    "type": "string",
                    "description": "InfiniBand management host for IB CLI verification",
                },
                "ib_mgmt_user": {
                    "type": "string",
                    "description": "SSH user for the InfiniBand management host",
                },
                "ib_mgmt_ssh_key": {
                    "type": "string",
                    "description": "SSH key path for the InfiniBand management host",
                },
                "cumulus_hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cumulus/Spectrum-X switch hosts",
                },
                "cumulus_user": {
                    "type": "string",
                    "description": "SSH user for Cumulus/Spectrum-X switches",
                },
                "cumulus_ssh_key": {
                    "type": "string",
                    "description": "SSH key path for Cumulus/Spectrum-X switches",
                },
                "primary_label": {
                    "type": "string",
                    "description": "Label for single-node/local steps",
                },
                "coverage_baseline_run_id": {
                    "type": "string",
                    "description": "Optional baseline run id for coverage delta artifacts",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra args appended after preset flags",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (0/null = no timeout)",
                },
            }
        ),
    },
)
def tool_cluster_common_eval(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_common_eval

    include_context, context_level = extract_context_opts(params)
    try:
        result = run_cluster_common_eval(
            preset=str(params.get("preset", "core-system")),
            run_id=params.get("run_id"),
            hosts=params.get("hosts") if isinstance(params.get("hosts"), list) else None,
            labels=params.get("labels") if isinstance(params.get("labels"), list) else None,
            ssh_user=params.get("ssh_user"),
            ssh_key=params.get("ssh_key"),
            oob_if=params.get("oob_if"),
            socket_ifname=params.get("socket_ifname"),
            nccl_ib_hca=params.get("nccl_ib_hca"),
            nmx_url=params.get("nmx_url"),
            nmx_token=params.get("nmx_token"),
            ib_mgmt_host=params.get("ib_mgmt_host"),
            ib_mgmt_user=params.get("ib_mgmt_user"),
            ib_mgmt_ssh_key=params.get("ib_mgmt_ssh_key"),
            cumulus_hosts=params.get("cumulus_hosts")
            if isinstance(params.get("cumulus_hosts"), list)
            else None,
            cumulus_user=params.get("cumulus_user"),
            cumulus_ssh_key=params.get("cumulus_ssh_key"),
            primary_label=params.get("primary_label"),
            coverage_baseline_run_id=params.get("coverage_baseline_run_id"),
            extra_args=params.get("extra_args")
            if isinstance(params.get("extra_args"), list)
            else None,
            timeout_seconds=params.get("timeout_seconds"),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"common cluster eval failed: {e}", include_context, context_level)


@register_tool(
    "cluster_fabric_eval",
    "Tags: cluster, fabric, nvlink, infiniband, spectrum-x, roce, verification, reproducibility. "
    "Run the canonical AI fabric evaluation bundle with a structured capability matrix, verification ledger, "
    "AI-workload correlation, and fabric scorecard. "
    "Returns: {success, entrypoint, artifact_roles, run_id, stdout/stderr}. "
    "🕐 MEDIUM (varies). USE when: You want the go-to cluster run for NVLink, InfiniBand, and Spectrum-X / RoCE characterization.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_id": {"type": "string", "description": "RUN_ID prefix (default: YYYY-MM-DD)"},
                "hosts": {"type": "array", "items": {"type": "string"}, "description": "Host list"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels (must match hosts count)",
                },
                "ssh_user": {"type": "string", "description": "SSH user"},
                "ssh_key": {"type": "string", "description": "SSH key path"},
                "oob_if": {
                    "type": "string",
                    "description": "Out-of-band interface for multi-node fabric studies",
                },
                "socket_ifname": {"type": "string", "description": "NCCL socket interface"},
                "nccl_ib_hca": {"type": "string", "description": "NCCL_IB_HCA allowlist"},
                "nmx_url": {
                    "type": "string",
                    "description": "NMX base URL for NVLink management-plane checks",
                },
                "nmx_token": {"type": "string", "description": "Optional NMX bearer token"},
                "ib_mgmt_host": {
                    "type": "string",
                    "description": "InfiniBand management host for IB CLI verification",
                },
                "ib_mgmt_user": {
                    "type": "string",
                    "description": "SSH user for the InfiniBand management host",
                },
                "ib_mgmt_ssh_key": {
                    "type": "string",
                    "description": "SSH key path for the InfiniBand management host",
                },
                "cumulus_hosts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cumulus/Spectrum-X switch hosts",
                },
                "cumulus_user": {
                    "type": "string",
                    "description": "SSH user for Cumulus/Spectrum-X switches",
                },
                "cumulus_ssh_key": {
                    "type": "string",
                    "description": "SSH key path for Cumulus/Spectrum-X switches",
                },
                "primary_label": {
                    "type": "string",
                    "description": "Label for single-node/local steps",
                },
                "coverage_baseline_run_id": {
                    "type": "string",
                    "description": "Optional baseline run id for coverage delta artifacts",
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra args appended after the fabric preset flags",
                },
                "require_management_plane": {
                    "type": "boolean",
                    "default": False,
                    "description": "Require configured management-plane endpoints for publish-grade fabric completeness",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (0/null = no timeout)",
                },
            }
        ),
    },
)
def tool_cluster_fabric_eval(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import run_cluster_fabric_eval

    include_context, context_level = extract_context_opts(params)
    try:
        result = run_cluster_fabric_eval(
            run_id=params.get("run_id"),
            hosts=params.get("hosts") if isinstance(params.get("hosts"), list) else None,
            labels=params.get("labels") if isinstance(params.get("labels"), list) else None,
            ssh_user=params.get("ssh_user"),
            ssh_key=params.get("ssh_key"),
            oob_if=params.get("oob_if"),
            socket_ifname=params.get("socket_ifname"),
            nccl_ib_hca=params.get("nccl_ib_hca"),
            nmx_url=params.get("nmx_url"),
            nmx_token=params.get("nmx_token"),
            ib_mgmt_host=params.get("ib_mgmt_host"),
            ib_mgmt_user=params.get("ib_mgmt_user"),
            ib_mgmt_ssh_key=params.get("ib_mgmt_ssh_key"),
            cumulus_hosts=params.get("cumulus_hosts")
            if isinstance(params.get("cumulus_hosts"), list)
            else None,
            cumulus_user=params.get("cumulus_user"),
            cumulus_ssh_key=params.get("cumulus_ssh_key"),
            primary_label=params.get("primary_label"),
            coverage_baseline_run_id=params.get("coverage_baseline_run_id"),
            extra_args=params.get("extra_args")
            if isinstance(params.get("extra_args"), list)
            else None,
            require_management_plane=bool(params.get("require_management_plane", False)),
            timeout_seconds=params.get("timeout_seconds"),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"fabric cluster eval failed: {e}", include_context, context_level)


@register_tool(
    "cluster_nmx_partition_lab",
    "Tags: cluster, nmx, nvlink, partitioning, lab, management-plane, tenancy. "
    "Build a lab-only NVLink/NMX partition workflow guide from live inventory. "
    "Returns: {success, entrypoint, commands, topology, partitions, recommendations}. "
    "⚡ FAST (~1-5s). USE when: You want exact create/update/delete/poll commands and Alpha/Beta partition seed locations without mutating the fabric.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "nmx_url": {
                    "type": "string",
                    "description": "NMX base URL (for example https://<host> or https://<host>/nmx/v1)",
                },
                "nmx_token": {"type": "string", "description": "Optional NMX token"},
                "alpha_name": {
                    "type": "string",
                    "default": "AlphaPartition",
                    "description": "Alpha partition name",
                },
                "beta_name": {
                    "type": "string",
                    "default": "BetaPartition",
                    "description": "Beta partition name",
                },
                "alpha_size": {
                    "type": "integer",
                    "default": 4,
                    "description": "Alpha partition GPU count",
                },
                "beta_size": {
                    "type": "integer",
                    "default": 4,
                    "description": "Beta partition GPU count",
                },
            }
        ),
    },
)
def tool_cluster_nmx_partition_lab(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import build_cluster_nmx_partition_lab

    include_context, context_level = extract_context_opts(params)
    try:
        result = build_cluster_nmx_partition_lab(
            nmx_url=params.get("nmx_url"),
            nmx_token=params.get("nmx_token"),
            alpha_name=str(params.get("alpha_name") or "AlphaPartition"),
            beta_name=str(params.get("beta_name") or "BetaPartition"),
            alpha_size=int(params.get("alpha_size", 4) or 4),
            beta_size=int(params.get("beta_size", 4) or 4),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"nmx partition lab helper failed: {e}", include_context, context_level)


@register_tool(
    "cluster_build_canonical_package",
    "Tags: cluster, package, canonical, cleanup, artifacts, reproducibility. "
    "Materialize a clean canonical cluster package from one primary run plus optional comparison and historical runs. "
    "Returns: {success, output_dir, package_manifest_path, package_readme_path, cleanup_keep_run_ids_path}. "
    "⚡ FAST (~1-10s). USE when: You want one agent-friendly bundle rooted at a single output directory without deleting source artifacts.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "canonical_run_id": {
                    "type": "string",
                    "description": "Primary canonical run id to package",
                },
                "comparison_run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional comparison/baseline run ids",
                },
                "historical_run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Historical run ids to retain as references or include when present",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Package output directory (must not already contain files)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 300,
                    "description": "Builder timeout in seconds",
                },
            }
        ),
    },
)
def tool_cluster_build_canonical_package(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import build_canonical_package

    include_context, context_level = extract_context_opts(params)
    try:
        result = build_canonical_package(
            canonical_run_id=str(params.get("canonical_run_id") or ""),
            comparison_run_ids=params.get("comparison_run_ids")
            if isinstance(params.get("comparison_run_ids"), list)
            else None,
            historical_run_ids=params.get("historical_run_ids")
            if isinstance(params.get("historical_run_ids"), list)
            else None,
            output_dir=params.get("output_dir"),
            timeout_seconds=int(params.get("timeout_seconds", 300) or 300),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"canonical package build failed: {e}", include_context, context_level)


@register_tool(
    "cluster_promote_run",
    "Tags: cluster, publish, promote, canonical, localhost, reports. "
    "Promote one run-local cluster result tree into the published cluster package. "
    "Returns: {success, run_id, published_root, published_structured_dir, published_localhost_report_path, steps}. "
    "⚡ FAST (~1-30s). USE when: A run under `cluster/runs/<run_id>/` should become the published localhost package and materialized published artifact set.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_id": {"type": "string", "description": "Run id under cluster/runs/<run_id>"},
                "label": {
                    "type": "string",
                    "default": "localhost",
                    "description": "Host label for localhost report rendering",
                },
                "allow_run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional run ids to retain during cleanup/validation hygiene checks",
                },
                "publish_report_path": {
                    "type": "string",
                    "description": "Published localhost report path",
                },
                "publish_notes_path": {
                    "type": "string",
                    "description": "Published localhost notes path",
                },
                "skip_render_localhost_report": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip rendering run-local + published localhost report markdown",
                },
                "skip_validate_localhost_report": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip localhost report validation after promotion",
                },
                "cleanup": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run cleanup_run_artifacts.sh after promotion using this run_id as canonical",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 300,
                    "description": "Promotion timeout in seconds",
                },
                "repo_root": {
                    "type": "string",
                    "description": "Optional repository root override (tests/alternate trees); default is this checkout",
                },
            }
        ),
    },
)
def tool_cluster_promote_run(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import promote_cluster_run

    include_context, context_level = extract_context_opts(params)
    try:
        run_id = str(params.get("run_id") or "").strip()
        if not run_id:
            return make_error("run_id is required", include_context, context_level)
        rr = params.get("repo_root")
        repo_root_val = str(rr).strip() if isinstance(rr, str) and rr.strip() else None
        result = promote_cluster_run(
            run_id=run_id,
            label=str(params.get("label") or "localhost"),
            allow_run_ids=params.get("allow_run_ids")
            if isinstance(params.get("allow_run_ids"), list)
            else None,
            publish_report_path=params.get("publish_report_path"),
            publish_notes_path=params.get("publish_notes_path"),
            skip_render_localhost_report=bool(params.get("skip_render_localhost_report", False)),
            skip_validate_localhost_report=bool(
                params.get("skip_validate_localhost_report", False)
            ),
            cleanup=bool(params.get("cleanup", False)),
            timeout_seconds=int(params.get("timeout_seconds", 300) or 300),
            repo_root=repo_root_val,
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"promote run failed: {e}", include_context, context_level)


@register_tool(
    "cluster_watch_promote",
    "Tags: cluster, publish, promote, watch, background, localhost. "
    "Attach a detached watcher process that waits for a PID to exit, validates run artifacts, then promotes the run. "
    "Returns: {success, run_id, repo_root, watcher_pid, watch_command, watch_status_path, launch_log_path, ...}. "
    "⚡ FAST to launch (~ms); the watcher may run until the watched PID exits. "
    "USE when: a long-running cluster job should auto-promote after exit without blocking the launcher.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "run_id": {"type": "string", "description": "Run id under cluster/runs/<run_id>"},
                "pid": {
                    "type": "integer",
                    "description": "Host PID to wait on before evaluating artifacts + promoting",
                },
                "label": {
                    "type": "string",
                    "default": "localhost",
                    "description": "Host label for localhost report rendering",
                },
                "allow_run_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional run ids to retain during cleanup/validation hygiene checks",
                },
                "publish_report_path": {
                    "type": "string",
                    "description": "Published localhost report path",
                },
                "publish_notes_path": {
                    "type": "string",
                    "description": "Published localhost notes path",
                },
                "repo_root": {
                    "type": "string",
                    "description": "Optional repository root override (tests/alternate trees); default is this checkout",
                },
                "skip_render_localhost_report": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip rendering run-local + published localhost report markdown",
                },
                "skip_validate_localhost_report": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip localhost report validation after promotion",
                },
                "cleanup": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run cleanup_run_artifacts.sh after promotion using this run_id as canonical",
                },
                "poll_interval_seconds": {
                    "type": "number",
                    "default": 30.0,
                    "description": "How often to poll whether the watched PID is still alive",
                },
            }
        ),
    },
)
def tool_cluster_watch_promote(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import watch_cluster_run_for_promotion

    include_context, context_level = extract_context_opts(params)
    try:
        run_id = str(params.get("run_id") or "").strip()
        if not run_id:
            return make_error("run_id is required", include_context, context_level)
        pid_raw = params.get("pid")
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            return make_error("pid must be an integer", include_context, context_level)
        rr = params.get("repo_root")
        repo_root_val = str(rr).strip() if isinstance(rr, str) and rr.strip() else None
        result = watch_cluster_run_for_promotion(
            run_id=run_id,
            pid=pid,
            label=str(params.get("label") or "localhost"),
            allow_run_ids=params.get("allow_run_ids")
            if isinstance(params.get("allow_run_ids"), list)
            else None,
            publish_report_path=params.get("publish_report_path"),
            publish_notes_path=params.get("publish_notes_path"),
            skip_render_localhost_report=bool(params.get("skip_render_localhost_report", False)),
            skip_validate_localhost_report=bool(
                params.get("skip_validate_localhost_report", False)
            ),
            cleanup=bool(params.get("cleanup", False)),
            poll_interval_seconds=float(params.get("poll_interval_seconds", 30.0) or 30.0),
            repo_root=repo_root_val,
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"watch promote failed: {e}", include_context, context_level)


@register_tool(
    "cluster_validate_field_report",
    "Tags: cluster, report, validate, hygiene, artifacts, requirements. "
    "Validate `cluster/field-report.md` and companion notes/template/runbook plus artifact hygiene. "
    "Returns: {success, returncode, stdout, stderr}. "
    "⚡ FAST (~1-10s). USE when: You updated the field report and want to ensure required sections and canonical artifacts are consistent.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "report": {
                    "type": "string",
                    "description": "Path to field-report.md (default: cluster/field-report.md)",
                },
                "notes": {
                    "type": "string",
                    "description": "Path to field-report-notes.md (default: cluster/field-report-notes.md)",
                },
                "template": {
                    "type": "string",
                    "description": "Path to field-report-template.md (default: cluster/docs/field-report-template.md)",
                },
                "runbook": {
                    "type": "string",
                    "description": "Path to advanced-runbook.md (default: cluster/docs/advanced-runbook.md)",
                },
                "canonical_run_id": {
                    "type": "string",
                    "description": "Expected canonical run id (optional)",
                },
                "allow_run_id": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional run id(s) allowed for hygiene checks (repeatable)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "default": 120,
                    "description": "Validator timeout in seconds",
                },
            }
        ),
    },
)
def tool_cluster_validate_field_report(params: Dict[str, Any]) -> Dict[str, Any]:
    from core.cluster import validate_field_report_requirements

    include_context, context_level = extract_context_opts(params)
    try:
        allow = params.get("allow_run_id") if isinstance(params.get("allow_run_id"), list) else None
        result = validate_field_report_requirements(
            report=params.get("report"),
            notes=params.get("notes"),
            template=params.get("template"),
            runbook=params.get("runbook"),
            canonical_run_id=params.get("canonical_run_id"),
            allow_run_id=allow,
            timeout_seconds=int(params.get("timeout_seconds", 120) or 120),
        )
        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"field report validation failed: {e}", include_context, context_level)


@register_tool(
    "status",
    "Tags: status, health, quick-check, sanity, ready, first-call, prerequisite. "
    "🚀 QUICK STATUS CHECK: Fast snapshot of GPU, software, and AI backend health. "
    "Returns: {gpu_ok, software_ok, ai_backend_ok, warnings, summary, gpu_count, cuda_version}. "
    "⚡ VERY FAST (<1s). USE FIRST when: Starting any session, before slow operations. "
    'Example: "Quick status check" or "Is everything healthy?" or "Ready for profiling?". '
    "WORKFLOW: status → if issues → system_dependencies or gpu_info. "
    "NOT FOR: Full context (use triage), deep audit (use system_full).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get quick status."""
    from core.engine import get_engine

    include_context = bool(params.get("include_context", True))
    context_level = params.get("context_level", "summary")
    result = get_engine().status()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "triage",
    "Tags: triage, start, first, status, context, health, entry-point, begin, overview. "
    "🎯 START HERE: Quick triage = status check + context summary in one call. "
    "Returns: {status: {gpu_ok, software_ok, ai_backend_ok}, context: {gpu, software, dependencies}}. "
    "⚡ FAST (~1-2s). THE BEST FIRST CALL for any new performance investigation. "
    'Example: "Start with triage" or "Quick overview" or "What\'s my system like?". '
    "PROVIDES: GPU model/count/VRAM, CUDA/PyTorch versions, dependency health, warnings. "
    "WORKFLOW: triage → recommend OR analyze_bottlenecks → specific tools. "
    "VERSUS OTHER ENTRY POINTS: "
    "• triage: status + context (recommended) "
    "• status: status only (faster, less info) "
    "• suggest_tools: tool recommendations based on intent. NOT FOR: Deep system audit (use system_full).",
    {"type": "object", "properties": with_context_params({})},
)
def tool_triage(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return quick status plus context summary to guide next actions."""
    from core.engine import get_engine

    engine = get_engine()
    result = {
        "success": True,
        "status": engine.status(),
        "context": get_cached_context("summary"),
    }
    include_context, context_level = extract_context_opts(params)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "job_status",
    "Tags: job, status, poll, async, background, queue, progress. "
    "Check status of a background job started with async=true. "
    "Returns: {job_id, status: queued|running|completed|error, result (if completed), duration_ms, progress?}. "
    "⚡ FAST (<1s). USE when: Polling for completion of background jobs. "
    'Example: "Check job status" or "Is my benchmark done?" or "Poll nsys capture". '
    "STATUS VALUES: "
    "• 'queued' → Waiting for the MCP queue runner to start "
    "• 'running' → Job in progress, poll again in 10-30s "
    "• 'completed' → Done! Result in 'result' field "
    "• 'error' → Failed, check 'error' field for details. "
    "TOOLS SUPPORTING async=true: run_benchmarks, profile_nsys, profile_ncu, profile_torch, profile_hta. "
    "WORKFLOW: tool(async=true) → poll job_status(job_id) → [completed] benchmark_triage or nsys_summary.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned from tool call with async=true",
                },
            }
        ),
        "required": ["job_id"],
    },
)
def tool_job_status(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    job_id = params.get("job_id")
    if not job_id:
        return make_error("job_id is required", include_context, context_level)
    _cleanup_job_store()
    record = JOB_STORE.get_status(job_id)
    if not record:
        return {
            "job_id": job_id,
            "status": "not_found",
            "note": "No job with this id; ensure you passed async=true on the original call.",
        }
    # Copy so we don't mutate stored record when adding error decoration
    payload = dict(record)
    if payload.get("status") == "error" and "error" not in payload:
        result = payload.get("result") or {}
        if isinstance(result, dict) and result.get("error"):
            payload["error"] = result.get("error")
        else:
            payload["error"] = "Job failed"
    progress_path = None
    if payload.get("progress_path"):
        progress_path = Path(str(payload.get("progress_path")))
    else:
        run_dir = payload.get("run_dir")
        run_id = payload.get("run_id")
        if run_dir and run_id:
            progress_path = _progress_path_for_run(Path(str(run_dir)), str(run_id))
    progress_payload = _read_progress_payload(progress_path)
    if progress_payload:
        payload["progress"] = progress_payload.get("current")
        payload["progress_history"] = progress_payload.get("history", [])[-5:]
        if progress_payload.get("load_warning"):
            payload["progress_warning"] = progress_payload.get("load_warning")
        if progress_path:
            payload["progress_path"] = str(progress_path)
    reported_status = str(payload.get("status", "") or "")
    effective_status = reported_status
    run_dir_path: Optional[Path] = None
    run_dir_value = payload.get("run_dir")
    if run_dir_value:
        run_dir_path = Path(str(run_dir_value))
    run_dir_exists = bool(run_dir_path and run_dir_path.exists())
    has_progress = bool(progress_payload)
    note_text = str(payload.get("note", "")).lower()
    waiting_for_queue_runner = "waiting for mcp queue runner" in note_text
    running_via_queue_runner = "running via mcp queue runner" in note_text

    queue_managed_tools = {
        "run_benchmarks",
        "profile_nsys",
        "profile_ncu",
        "profile_torch",
        "profile_hta",
    }
    queue_managed_job = str(payload.get("tool", "") or "") in queue_managed_tools

    if reported_status == "running" and (
        waiting_for_queue_runner
        or (queue_managed_job and (not has_progress) and (not running_via_queue_runner))
        or ((not run_dir_exists and not has_progress) and (not running_via_queue_runner))
    ):
        effective_status = "queued"
    elif reported_status == "queued" and has_progress:
        effective_status = "running"

    if effective_status != reported_status:
        payload["reported_status"] = reported_status
        payload["status"] = effective_status
    if "success" not in payload:
        payload["success"] = payload.get("status") not in {"error", "not_found"}
    return attach_context_if_requested(payload, include_context, context_level)


# =============================================================================
# HUGGINGFACE TOOLS
# =============================================================================


@register_tool(
    "hf",
    "Tags: huggingface, hf, models, download, search, trending, hub. "
    "HuggingFace Hub operations: search models, get trending, download models. "
    "⚡ FAST (~2s). USE when: Finding models, downloading from HF Hub. "
    "Example: action='search', query='llama 2 7b' or action='trending', limit=5. "
    "WORKFLOW: hf(action='search') → hf(action='download'). "
    "NOT FOR: Model performance recommendations (use recommend).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "action": {
                    "type": "string",
                    "enum": ["search", "trending", "download"],
                    "default": "search",
                    "description": "Operation: search, trending, or download",
                },
                "query": {
                    "type": "string",
                    "description": "Search query or model name for download",
                },
                "limit": {"type": "integer", "default": 10, "description": "Max results to return"},
            }
        ),
    },
)
def tool_hf(params: Dict[str, Any]) -> Dict[str, Any]:
    """HuggingFace Hub operations."""
    from core.engine import get_engine

    include_context, context_level = extract_context_opts(params)

    action = normalize_param("action", params.get("action"), "search")
    query = params.get("query", "")
    limit = params.get("limit", 10)

    try:
        engine = get_engine()
        if action == "search":
            result = engine.hf.search(query, limit=limit)
        elif action == "trending":
            result = engine.hf.trending(limit=limit)
        elif action == "download":
            if not query:
                return make_error(
                    "query (model name) required for download", include_context, context_level
                )
            result = engine.hf.download(query)
        else:
            return make_error(f"Unknown action: {action}", include_context, context_level)

        return attach_context_if_requested(result, include_context, context_level)
    except Exception as e:
        return make_error(f"HuggingFace operation failed: {e}", include_context, context_level)


# =============================================================================
# TOOLS (NON-BENCHMARK UTILITIES)
# =============================================================================


def _extract_tools_cli_args(
    params: Dict[str, Any],
    include_context: bool,
    context_level: str,
    *,
    default_timeout_seconds: int,
) -> Tuple[Optional[List[str]], Optional[int], Optional[Dict[str, Any]]]:
    raw_args = params.get("args")
    if raw_args is None:
        tool_args: Optional[List[str]] = []
    elif isinstance(raw_args, list) and all(isinstance(a, str) for a in raw_args):
        tool_args = raw_args
    else:
        return (
            None,
            None,
            make_error("args must be a list of strings", include_context, context_level),
        )

    timeout_param = params.get("timeout_seconds")
    timeout_seconds: Optional[int]
    if timeout_param is None:
        timeout_seconds = default_timeout_seconds
    else:
        try:
            timeout_seconds = int(timeout_param)
        except Exception:
            return (
                None,
                None,
                make_error("timeout_seconds must be an integer", include_context, context_level),
            )

    return tool_args, timeout_seconds, None


@register_tool(
    "tools_kv_cache",
    "Tags: tools, kv-cache, memory, sizing, utility. "
    "Run the KV-cache size calculator (non-benchmark utility). "
    "Forwards args to `aisp tools kv-cache -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 60,
                },
            }
        ),
    },
)
def tool_tools_kv_cache(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=60,
    )
    if err:
        return err
    result = _run_tools_cli("kv-cache", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "tools_cost_per_token",
    "Tags: tools, cost, power, throughput, utility. "
    "Run the cost-per-token calculator (non-benchmark utility). "
    "Forwards args to `aisp tools cost-per-token -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 60,
                },
            }
        ),
    },
)
def tool_tools_cost_per_token(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=60,
    )
    if err:
        return err
    result = _run_tools_cli("cost-per-token", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "tools_compare_precision",
    "Tags: tools, precision, accuracy, fp16, bf16, fp8, utility. "
    "Run the precision/accuracy comparison tool (non-benchmark utility). "
    "Forwards args to `aisp tools compare-precision -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 300,
                },
            }
        ),
    },
)
def tool_tools_compare_precision(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=300,
    )
    if err:
        return err
    result = _run_tools_cli("compare-precision", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "tools_detect_cutlass",
    "Tags: tools, cutlass, environment, discovery, utility. "
    "Run CUTLASS environment detection (non-benchmark utility). "
    "Forwards args to `aisp tools detect-cutlass -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 30,
                },
            }
        ),
    },
)
def tool_tools_detect_cutlass(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=30,
    )
    if err:
        return err
    result = _run_tools_cli("detect-cutlass", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "tools_dump_hw",
    "Tags: tools, hardware, capabilities, report, utility. "
    "Dump comprehensive hardware capability report (non-benchmark utility). "
    "Forwards args to `aisp tools dump-hw -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 300,
                },
            }
        ),
    },
)
def tool_tools_dump_hw(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=300,
    )
    if err:
        return err
    result = _run_tools_cli("dump-hw", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "tools_probe_hw",
    "Tags: tools, hardware, capabilities, probe, cache, utility. "
    "Probe GPU capabilities dynamically and cache results (non-benchmark utility). "
    "Forwards args to `aisp tools probe-hw -- <args...>`.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the tool script.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout for the tool invocation.",
                    "default": 600,
                },
            }
        ),
    },
)
def tool_tools_probe_hw(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    tool_args, timeout_seconds, err = _extract_tools_cli_args(
        params,
        include_context,
        context_level,
        default_timeout_seconds=600,
    )
    if err:
        return err
    result = _run_tools_cli("probe-hw", tool_args, timeout_seconds=timeout_seconds)
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# UTILITY TOOLS
# =============================================================================


@register_tool(
    "benchmark_contracts",
    "Tags: benchmark, methodology, warehouse, docs, contract. "
    "Return the repo benchmark methodology, warehouse, and BenchmarkRun contract surfaces. "
    "Returns: {schema_version, contracts, interface_entries, generator}. "
    "⚡ FAST (<1s). USE when: You need the stable methodology/warehouse contract or want the CLI, dashboard, and MCP entrypoints in one place.",
    {"type": "object", "properties": with_context_params({})},
)
def tool_benchmark_contracts(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    from core.benchmark.contracts_surface import get_benchmark_contracts_summary

    result = get_benchmark_contracts_summary()
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "render_benchmark_run",
    "Tags: benchmark, generator, yaml, contract. "
    "Render BenchmarkRun YAML through the shared backend template renderer. "
    "Returns: {schema_version, template_path, applied_values, rendered_yaml}. "
    "⚡ FAST (<1s). USE when: You need a lockstep BenchmarkRun preview or want to script the same generator used by the dashboard.",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "name": {"type": "string", "description": "BenchmarkRun metadata.name override."},
                "benchmarkClass": {
                    "type": "string",
                    "enum": ["publication_grade", "realism_grade"],
                },
                "workloadType": {"type": "string", "enum": ["training", "inference", "mixed"]},
                "schedulerPath": {"type": "string", "description": "Scheduler path identifier."},
                "cadence": {"type": "string", "enum": ["canary", "nightly", "pre_release"]},
                "model": {"type": "string", "description": "Model identifier."},
                "precision": {"type": "string", "description": "Precision override."},
                "batchingPolicy": {"type": "string", "description": "Batching policy override."},
                "concurrencyModel": {
                    "type": "string",
                    "description": "Concurrency model override.",
                },
                "comparisonVariable": {
                    "type": "string",
                    "enum": [
                        "hardware_generation",
                        "runtime_version",
                        "scheduler_path",
                        "control_plane_path",
                        "driver_stack",
                        "network_topology",
                        "storage_stack",
                    ],
                },
            }
        ),
    },
)
def tool_render_benchmark_run(params: Dict[str, Any]) -> Dict[str, Any]:
    include_context, context_level = extract_context_opts(params)
    from core.benchmark.contracts_surface import render_benchmark_run_yaml

    overrides = {
        key: value
        for key, value in params.items()
        if key
        in {
            "name",
            "benchmarkClass",
            "workloadType",
            "schedulerPath",
            "cadence",
            "model",
            "precision",
            "batchingPolicy",
            "concurrencyModel",
            "comparisonVariable",
        }
    }
    result = render_benchmark_run_yaml(overrides)
    return attach_context_if_requested(result, include_context, context_level)


@register_tool(
    "suggest_tools",
    "Tags: suggest, recommend, navigation, intent, which-tool, discovery, help, lost. "
    "🧭 TOOL NAVIGATOR: Get ranked tool suggestions based on your intent or problem. "
    "Returns: {suggestions: [{tool, reason, score}], count}. "
    "⚡ FAST (<1s). USE when: Unsure which tool to use, have a problem description. "
    "EXAMPLE INTENTS → SUGGESTED TOOLS: "
    "• 'OOMing on 24GB' → profile_memory, analyze_whatif, inference_quantization "
    "• 'slow training' → analyze_bottlenecks, profile_nsys, recommend "
    "• 'multi-GPU setup' → distributed_plan, gpu_topology, launch_plan "
    "• 'vLLM latency' → inference_vllm, inference_quantization "
    "• 'deep dive baseline vs optimized (nsys+ncu)' → benchmark_deep_dive_compare "
    "• 'compare profiles' → compare_nsys, profile_compare. "
    "WORKFLOW: suggest_tools → use suggested tools. "
    "NOT FOR: Direct answers (use ask), getting started (use triage). "
    "Defaults to LLM-based routing; if no LLM backend is configured, "
    "automatically falls back to keyword heuristics with a WARNING. "
    "Set llm_routing=false to force heuristics (also returns a WARNING).",
    {
        "type": "object",
        "properties": with_context_params(
            {
                "query": {
                    "type": "string",
                    "description": "Your intent, problem, or question in natural language",
                },
                "llm_routing": {
                    "type": "boolean",
                    "description": "Use LLM-based intent routing instead of keyword heuristics (requires LLM backend).",
                    "default": True,
                },
                "max_suggestions": {
                    "type": "integer",
                    "description": "Maximum number of suggestions to return (default: no limit for heuristics, 6 for LLM).",
                },
            }
        ),
        "required": ["query"],
    },
)
def tool_suggest_tools(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ranked list of suggested tools given a query."""
    query = params.get("query") or ""
    include_context, context_level = extract_context_opts(params)
    llm_routing = params.get("llm_routing", True)
    max_suggestions = params.get("max_suggestions")

    if not isinstance(llm_routing, bool):
        return make_error("llm_routing must be a boolean.", include_context, context_level)
    if max_suggestions is not None and not isinstance(max_suggestions, int):
        return make_error("max_suggestions must be an integer.", include_context, context_level)
    if max_suggestions is not None and max_suggestions < 1:
        return make_error("max_suggestions must be >= 1.", include_context, context_level)

    tool_catalog = [{"tool": name, "description": tool.description} for name, tool in TOOLS.items()]

    try:
        routing_result = suggest_tools_auto(
            query,
            llm_routing=llm_routing,
            rules=DEFAULT_SUGGEST_RULES,
            tool_catalog=tool_catalog,
            max_suggestions=max_suggestions,
        )
    except ValueError as exc:
        return make_error(str(exc), include_context, context_level)

    result = {
        "suggestions": routing_result.get("suggestions", []),
        "count": len(routing_result.get("suggestions", [])),
        "success": True,
        "routing": routing_result.get("routing"),
        "llm_available": routing_result.get("llm_available"),
    }
    if routing_result.get("warning"):
        result["warning"] = routing_result["warning"]
    return attach_context_if_requested(result, include_context, context_level)


# =============================================================================
# MCP PROTOCOL IMPLEMENTATION
# =============================================================================


class MCPServer:
    """MCP Server for AI Systems Performance."""

    def __init__(self):
        self.name = "aisp"
        self.version = "2.0.0"
        # Track pending requests to help with debugging and prevent duplicate processing
        self._pending_requests: Dict[Any, Dict[str, Any]] = {}
        self._request_lock = threading.RLock()
        # Track request timestamps to detect stale requests
        self._request_timeouts: Dict[Any, float] = {}
        self._timeout_seconds = 300.0  # 5 minutes default timeout

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema}
            for tool in TOOLS.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Call a tool by name."""
        start_ts = time.time()
        tool_meta = TOOLS.get(name)
        server_info = {"name": self.name, "version": self.version}
        if name not in HANDLERS:
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                {"error": f"Unknown tool: {name}"},
                duration_ms=int((time.time() - start_ts) * 1000),
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(content=_content_from_payload(payload), is_error=True)

        try:
            raw_result = HANDLERS[name](arguments)
            result = _normalize_result(raw_result)
            duration_ms = int((time.time() - start_ts) * 1000)
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                result,
                duration_ms,
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(
                content=_content_from_payload(payload), is_error=payload.get("status") == "error"
            )
        except Exception as e:
            tb = traceback.format_exc()
            duration_ms = int((time.time() - start_ts) * 1000)
            payload = _build_enriched_tool_payload(
                name,
                arguments,
                _normalize_result({"error": str(e), "traceback": tb}),
                duration_ms=duration_ms,
                had_exception=True,
                server_info=server_info,
                tool_meta=tool_meta,
            )
            return ToolResult(content=_content_from_payload(payload), is_error=True)

    def _cleanup_stale_requests(self):
        """Remove requests that have timed out."""
        current_time = time.time()
        with self._request_lock:
            stale_ids = [
                req_id
                for req_id, timestamp in self._request_timeouts.items()
                if current_time - timestamp > self._timeout_seconds
            ]
            for req_id in stale_ids:
                self._pending_requests.pop(req_id, None)
                self._request_timeouts.pop(req_id, None)
                if os.environ.get("AISP_MCP_DEBUG"):
                    print(f"[DEBUG] Cleaned up stale request ID: {req_id}", file=sys.stderr)

    def _track_request(self, msg_id: Any, method: str, params: Dict[str, Any]) -> bool:
        """Track a request and return True if it's a duplicate."""
        if msg_id is None:
            return False

        with self._request_lock:
            # Clean up stale requests periodically
            if len(self._pending_requests) > 100:  # Threshold to avoid constant cleanup
                self._cleanup_stale_requests()

            # Check if this is a duplicate request
            if msg_id in self._pending_requests:
                existing = self._pending_requests[msg_id]
                if os.environ.get("AISP_MCP_DEBUG"):
                    print(
                        f"[DEBUG] Duplicate request detected - ID: {msg_id}, "
                        f"Method: {method}, Previous: {existing.get('method')}",
                        file=sys.stderr,
                    )
                # Update timestamp but don't process again
                self._request_timeouts[msg_id] = time.time()
                return True

            # Track new request
            self._pending_requests[msg_id] = {
                "method": method,
                "params": params,
                "received_at": time.time(),
            }
            self._request_timeouts[msg_id] = time.time()
            return False

    def _complete_request(self, msg_id: Any):
        """Mark a request as completed."""
        if msg_id is None:
            return
        with self._request_lock:
            self._pending_requests.pop(msg_id, None)
            self._request_timeouts.pop(msg_id, None)

    @staticmethod
    def _write_stdio_warning(message: str) -> None:
        """Emit a best-effort transport warning to stderr."""
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()

    def _emit_stdio_json(self, payload: Dict[str, Any], *, failure_context: str) -> bool:
        """Emit a JSON-RPC payload to stdout and report transport failures visibly."""
        try:
            print(json.dumps(payload), flush=True)
            return True
        except Exception as exc:
            self._write_stdio_warning(f"Failed to emit {failure_context}: {exc}")
            return False

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle an MCP message. Returns None for notifications (messages without an id)."""
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = msg_id is None
        params = message.get("params", {})

        # Validate message structure
        if not isinstance(message, dict):
            if msg_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: message must be an object",
                    },
                }
            return None

        # Ignore notifications (messages without an id); MCP clients may send these.
        if is_notification:
            return None

        # Track request and detect duplicates
        # Note: initialize is tracked but always processed (clears state on client restart)
        is_duplicate = self._track_request(msg_id, method, params)
        if is_duplicate and method != "initialize":
            # For duplicate requests (except initialize), return an error
            # Initialize should always be processed to reset server state
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": "Duplicate request detected. This request was already processed.",
                },
            }

        try:
            if method == "initialize":
                # Initialize clears old pending requests (from previous client session)
                # but we still process and complete this initialize request itself
                with self._request_lock:
                    # Save current initialize request info before clearing
                    current_req = self._pending_requests.get(msg_id, {})
                    current_timeout = self._request_timeouts.get(msg_id, time.time())
                    # Clear all pending requests (client restart scenario)
                    self._pending_requests.clear()
                    self._request_timeouts.clear()
                    # Re-add this initialize request so it can be properly completed
                    if msg_id is not None:
                        self._pending_requests[msg_id] = current_req or {
                            "method": method,
                            "params": params,
                            "received_at": time.time(),
                        }
                        self._request_timeouts[msg_id] = current_timeout

                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": self.name, "version": self.version},
                    },
                }
                # Don't complete here - will be completed after sending response
                return response

            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": self.get_tool_list()},
                }
                # Don't complete here - will be completed after sending response
                return response

            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = self.call_tool(tool_name, arguments)

                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": result.content, "isError": result.is_error},
                }
                # Don't complete here - will be completed after sending response
                return response

            else:
                if is_notification:
                    return None
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
                # Don't complete here - will be completed after sending response
                return response
        except Exception as e:
            # On error, we still need to complete the request
            # But only if we're going to send an error response
            # Re-raise to be handled by outer error handler
            raise

    async def run_stdio(self):
        """Run MCP server over stdio."""
        import sys

        print(f"AISP MCP Server v{self.version} - {len(TOOLS)} tools available", file=sys.stderr)

        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                # Skip empty lines
                line = line.strip()
                if not line:
                    continue

                message = json.loads(line)

                # Validate JSON-RPC version
                if message.get("jsonrpc") != "2.0":
                    msg_id = message.get("id")
                    if msg_id is not None:
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {
                                "code": -32600,
                                "message": "Invalid Request: jsonrpc must be '2.0'",
                            },
                        }
                        self._emit_stdio_json(
                            error_response,
                            failure_context=f"invalid request response for id {msg_id!r}",
                        )
                    continue

                response = await self.handle_message(message)
                if response is not None:
                    # Send response, then complete the request
                    # This ensures we only send responses for requests we're still tracking
                    response_id = response.get("id")
                    should_send = True
                    if response_id is not None:
                        with self._request_lock:
                            if response_id not in self._pending_requests:
                                # Request was cleaned up (timeout or client restart)
                                # Don't send response to avoid "unknown message ID" errors
                                if os.environ.get("AISP_MCP_DEBUG"):
                                    print(
                                        f"[DEBUG] Skipping response for cleaned-up request ID: {response_id}",
                                        file=sys.stderr,
                                    )
                                should_send = False

                    if should_send:
                        print(json.dumps(response), flush=True)
                        # Complete request after sending response
                        if response_id is not None:
                            self._complete_request(response_id)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", file=sys.stderr)
                line_preview = line[:200] if "line" in locals() else "<unavailable>"
                self._write_stdio_warning(
                    "Unable to recover request id from malformed JSON input; "
                    f"no parse error response emitted. line_preview={line_preview!r}"
                )
            except Exception as e:
                print(f"Error handling message: {e}", file=sys.stderr)
                import traceback

                print(traceback.format_exc(), file=sys.stderr)
                # Try to send error response
                try:
                    msg_id = message.get("id") if "message" in locals() else None
                    if msg_id is not None:
                        # Only send error response if request is still tracked
                        with self._request_lock:
                            if msg_id in self._pending_requests:
                                error_response = {
                                    "jsonrpc": "2.0",
                                    "id": msg_id,
                                    "error": {
                                        "code": -32603,
                                        "message": f"Internal error: {str(e)}",
                                    },
                                }
                                if self._emit_stdio_json(
                                    error_response,
                                    failure_context=f"internal error response for id {msg_id!r}",
                                ):
                                    # Complete request after sending error response
                                    self._complete_request(msg_id)
                except Exception as response_exc:
                    self._write_stdio_warning(
                        f"Failed to prepare internal error response for request {msg_id!r}: {response_exc}"
                    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AISP MCP Server")
    parser.add_argument("--list", action="store_true", help="List available tools")
    parser.add_argument("--test", type=str, help="Test a specific tool")
    parser.add_argument("--serve", action="store_true", help="Start MCP server (stdio)")
    args = parser.parse_args()

    if args.list:
        print(f"\n🚀 AISP MCP Tools ({len(TOOLS)} available):\n")
        for name, tool in sorted(TOOLS.items()):
            print(f"  {name}")
            print(f"    {tool.description}\n")
        return

    if args.test:
        tool_name = args.test
        if tool_name not in HANDLERS:
            print(f"Unknown tool: {tool_name}")
            return 1

        print(f"Testing {tool_name}...")
        result = HANDLERS[tool_name]({})
        print(json.dumps(result, indent=2, default=str))
        return

    if args.serve:
        server = MCPServer()
        asyncio.run(server.run_stdio())
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
