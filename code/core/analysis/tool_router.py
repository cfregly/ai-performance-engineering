"""Tool intent routing helpers (heuristic + LLM)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.llm import get_llm_status, llm_call

DEFAULT_SUGGEST_RULES: List[Dict[str, Any]] = [
    {
        "tool": "tools_diagnostics",
        "keywords": ["tcp window", "path mtu", "bandwidth delay", "pfc", "ecn", "collective diagnosis", "serving comparison", "cross-layer diagnosis"],
        "reason": "Run network, fabric, collective or serving diagnostics with retained evidence",
    },
    {
        "tool": "benchmark_deep_dive_compare",
        "keywords": [
            "deep_dive",
            "deep dive",
            "deep-dive",
            "profile and compare",
            "profile & compare",
            "profile compare",
            "profile+compare",
            "compare and profile",
            "compare & profile",
            "compare+profile",
            "profile baseline vs optimized",
            "compare baseline vs optimized",
            "compare baseline and optimized",
            "baseline optimized",
            "baseline/optimized",
            "baseline and optimized",
            "baseline and optimized version",
            "baseline vs optimized version",
            "optimized version",
            "baseline file",
            "optimized file",
            "baseline vs optimized file",
            "baseline vs optimized",
            "compare baseline",
            "compare optimized",
            "nsys+ncu",
            "nsys and ncu",
            "nsight systems and compute",
            "profile diff",
        ],
        "reason": "One-shot: run benchmark with deep_dive profiling and return baseline-vs-optimized diffs (nsys+ncu+torch)",
    },
    {
        "tool": "profile_compare",
        "keywords": ["compare profiles", "profile compare", "flamegraph compare", "flame graph compare"],
        "reason": "Compare profiles with flame graph narrative + metrics",
    },
    {
        "tool": "compare_nsys",
        "keywords": ["compare nsys", "nsys diff", "nsight systems compare", "timeline compare"],
        "reason": "Compare Nsight Systems reports",
    },
    {
        "tool": "compare_ncu",
        "keywords": ["compare ncu", "ncu diff", "nsight compute compare", "kernel compare"],
        "reason": "Compare Nsight Compute reports",
    },
    {
        "tool": "benchmark_compare_runs",
        "keywords": [
            "compare runs",
            "compare benchmark runs",
            "compare benchmarks",
            "benchmark runs",
            "bench runs",
            "benchmark diff",
            "diff results",
            "regressions",
            "improvements",
        ],
        "reason": "Diff two benchmark JSON runs",
    },
    {
        "tool": "benchmark_compare",
        "keywords": ["compare benchmark results", "benchmark compare", "compare results table"],
        "reason": "Compare benchmark results (dashboard-style diff)",
    },
    {
        "tool": "benchmark_triage",
        "keywords": ["benchmark analysis", "analyze results", "triage benchmarks", "triage results"],
        "reason": "Post-benchmark analysis and recommendations",
    },
    {
        "tool": "analyze_bottlenecks",
        "keywords": [
            "slow",
            "latency",
            "bottleneck",
            "utilization",
            "stall",
            "idle",
            "regression",
            "throughput drop",
            "analyze",
            "analysis",
            "diagnose",
            "why slow",
        ],
        "reason": "Diagnose bottlenecks for slow workload/latency issues",
    },
    {
        "tool": "analyze_comm_overlap",
        "keywords": ["comm overlap", "communication overlap", "allreduce overlap", "overlap compute"],
        "reason": "Analyze communication/compute overlap",
    },
    {
        "tool": "analyze_dataloader",
        "keywords": ["dataloader", "data loading", "input pipeline", "prefetch", "data loader"],
        "reason": "Find input pipeline and DataLoader bottlenecks",
    },
    {
        "tool": "analyze_energy",
        "keywords": ["energy", "power efficiency", "watts", "energy efficiency"],
        "reason": "Analyze power draw and energy efficiency",
    },
    {
        "tool": "analyze_memory_patterns",
        "keywords": ["memory pattern", "memory coalescing", "coalescing", "bank conflict", "warp divergence"],
        "reason": "Analyze memory access patterns and coalescing",
    },
    {
        "tool": "predict_scaling",
        "keywords": ["scaling", "scale up", "scale to", "multi-gpu scaling", "more gpus"],
        "reason": "Predict scaling to more GPUs or larger workloads",
    },
    {
        "tool": "hw_disk",
        "keywords": ["disk", "io", "storage"],
        "reason": "Disk I/O benchmark (sequential)",
    },
    {
        "tool": "hw_pcie",
        "keywords": ["pcie", "h2d", "d2h", "pci-e"],
        "reason": "PCIe H2D/D2H bandwidth benchmark",
    },
    {
        "tool": "hw_cache",
        "keywords": ["memory stride", "cache", "l2", "hbm"],
        "reason": "Stride/bandwidth test for memory hierarchy",
    },
    {
        "tool": "hw_tc",
        "keywords": ["tensor core", "tflops", "matmul"],
        "reason": "Tensor core throughput test",
    },
    {
        "tool": "hw_speed",
        "keywords": ["speed test", "quick speed", "gemm speed", "attention speed"],
        "reason": "Quick GPU speed tests (GEMM/memory/attention)",
    },
    {
        "tool": "hw_ib",
        "keywords": ["infiniband", "ib bandwidth", "rdma bandwidth"],
        "reason": "InfiniBand bandwidth test",
    },
    {
        "tool": "hw_nccl",
        "keywords": ["nccl bandwidth", "allreduce bandwidth", "collective bandwidth"],
        "reason": "NCCL collective bandwidth test",
    },
    {
        "tool": "hw_p2p",
        "keywords": ["p2p bandwidth", "nvlink bandwidth", "gpu p2p"],
        "reason": "GPU-to-GPU P2P bandwidth test",
    },
    {
        "tool": "hw_network",
        "keywords": ["network bandwidth", "nic throughput", "network test"],
        "reason": "Network throughput test",
    },
    {
        "tool": "profile_flame",
        "keywords": ["flame", "flame graph", "flamegraph", "hotspot", "hot spot", "call stack", "stack trace"],
        "reason": "Inspect time hotspots with flame graph",
    },
    # Profile kernels is reserved for perf hotspots; avoid matching install/import issues.
    {
        "tool": "profile_kernels",
        "keywords": [
            "kernel hotspot",
            "cuda hotspot",
            "ptx hotspot",
            "kernel time",
            "kernel breakdown",
            "kernel list",
            "top kernels",
            "kernel stats",
            "launch count",
            "kernel profiling",
        ],
        "reason": "Check CUDA kernel hotspots",
    },
    {
        "tool": "profile_roofline",
        "keywords": ["roofline", "compute bound", "memory bound", "arithmetic intensity"],
        "reason": "See compute vs memory bound positioning",
    },
    {
        "tool": "profile_torch",
        "keywords": [
            "torch profiler",
            "pytorch profiler",
            "operator breakdown",
            "op breakdown",
            "autograd",
            "torch.compile",
            "torch compile",
            "inductor",
            "dynamo",
            "graph break",
            "graph breaks",
        ],
        "reason": "Profile PyTorch operator breakdown",
    },
    {
        "tool": "profile_nsys",
        "keywords": ["profile", "profiling", "trace", "timeline", "nsys", "nsight systems", "systems trace", "cuda api", "overlap"],
        "reason": "Capture timeline with Nsight Systems",
    },
    {
        "tool": "profile_ncu",
        "keywords": [
            "ncu",
            "nsight compute",
            "compute profile",
            "kernel metrics",
            "kernel profile",
            "occupancy",
            "register",
            "register pressure",
            "smem",
            "shared memory",
            "warp",
            "ipc",
            "sm efficiency",
            "kernel tuning",
            "tune kernel",
            "kernel optimize",
            "ptx",
            "sass",
        ],
        "reason": "Capture kernel metrics with Nsight Compute",
    },
    {
        "tool": "profile_memory",
        "keywords": ["memory", "vram", "oom", "out of memory", "memory leak", "fragmentation", "allocation", "spike"],
        "reason": "See memory timeline and spikes",
    },
    {
        "tool": "profile_hta",
        "keywords": ["hta", "holistic trace", "trace analysis", "bottleneck trace"],
        "reason": "Run HTA analysis for timeline bottlenecks",
    },
    {
        "tool": "nsys_summary",
        "keywords": ["nsys summary", "summarize nsys", "nsys report summary"],
        "reason": "Summarize an existing Nsight Systems report",
    },
    {
        "tool": "gpu_bandwidth",
        "keywords": ["bandwidth", "p2p", "nvlink", "pci-e", "pci express"],
        "reason": "Check GPU memory/P2P bandwidth",
    },
    {
        "tool": "gpu_power",
        "keywords": ["power", "thermal", "throttle", "temperature", "temp"],
        "reason": "Check power/thermal headroom and throttling",
    },
    {
        "tool": "gpu_info",
        "keywords": ["gpu info", "name", "memory", "vram", "utilization", "compute capability"],
        "reason": "Get GPU inventory and basic telemetry",
    },
    {
        "tool": "gpu_topology_matrix",
        "keywords": ["topology matrix", "topo -m", "nvidia-smi topo", "topo matrix"],
        "reason": "Get raw GPU/NUMA topology matrix",
    },
    {
        "tool": "system_software",
        "keywords": ["pytorch", "cuda version", "driver", "software version", "cuDNN", "python version"],
        "reason": "Check software stack versions",
    },
    {
        "tool": "system_dependencies",
        "keywords": ["import error", "torch.cuda", "dependency", "missing library", "package check", "install issue"],
        "reason": "Check dependency health for install/import issues",
    },
    {
        "tool": "system_env",
        "keywords": ["env vars", "environment variables", "env", "paths", "cuda_home"],
        "reason": "Snapshot key environment variables and paths",
    },
    {
        "tool": "system_network",
        "keywords": ["network status", "ib status", "rdma status", "gpudirect", "infiniband status"],
        "reason": "Inspect network interfaces and InfiniBand status",
    },
    {
        "tool": "system_parameters",
        "keywords": ["sysctl", "kernel parameters", "swappiness", "dirty ratio", "numa balancing"],
        "reason": "Inspect kernel/system parameters",
    },
    {
        "tool": "system_container",
        "keywords": ["container", "cgroup", "limits", "quota"],
        "reason": "Inspect container/cgroup limits",
    },
    {
        "tool": "system_cpu_memory",
        "keywords": ["numa", "cpu memory", "cache size", "memory hierarchy", "hugepages"],
        "reason": "Analyze CPU/NUMA/memory hierarchy",
    },
    {
        "tool": "system_capabilities",
        "keywords": ["capabilities", "features", "tensor cores", "fp8", "tma", "bf16"],
        "reason": "Inspect hardware capabilities",
    },
    {
        "tool": "system_full",
        "keywords": ["full system", "system audit", "system analysis", "full inventory"],
        "reason": "Full system analysis with tuning recommendations",
    },
    {
        "tool": "context_summary",
        "keywords": ["context summary", "summary context", "system snapshot"],
        "reason": "Get a quick system context summary",
    },
    {
        "tool": "context_full",
        "keywords": ["full context", "full system context", "system dump"],
        "reason": "Get full system context",
    },
    {
        "tool": "tools_kv_cache",
        "keywords": ["kv cache", "kv-cache size", "kv cache size"],
        "reason": "Calculate KV-cache size",
    },
    {
        "tool": "tools_cost_per_token",
        "keywords": ["cost per token", "token cost", "cost estimate"],
        "reason": "Estimate cost per token",
    },
    {
        "tool": "tools_compare_precision",
        "keywords": ["compare precision", "precision comparison", "fp16 vs bf16", "accuracy comparison"],
        "reason": "Compare precision/accuracy tradeoffs",
    },
    {
        "tool": "tools_detect_cutlass",
        "keywords": ["detect cutlass", "cutlass setup", "cutlass environment"],
        "reason": "Detect CUTLASS environment",
    },
    {
        "tool": "tools_dump_hw",
        "keywords": ["dump hardware", "hardware report", "capability report"],
        "reason": "Dump hardware capability report",
    },
    {
        "tool": "tools_probe_hw",
        "keywords": ["probe hardware", "hardware probe", "capabilities probe"],
        "reason": "Probe GPU capabilities and cache results",
    },
    {
        "tool": "hf",
        "keywords": ["huggingface", "hf search", "trending models", "download model", "hugging face"],
        "reason": "HuggingFace Hub operations (search/trending/download)",
    },
    {
        "tool": "cost_estimate",
        "keywords": ["cost estimate", "cloud cost", "training cost", "inference cost", "pricing"],
        "reason": "Estimate cloud cost for workloads",
    },
    {
        "tool": "analyze_whatif",
        "keywords": ["vram", "memory", "limit", "constraint", "cap"],
        "reason": "What-if recommendations under VRAM/latency constraints",
    },
    {
        "tool": "optimize",
        "keywords": [
            "optimize file",
            "optimize path",
            "optimize benchmark",
            "optimize target",
            "optimize this file",
            "benchmark file",
            "benchmark target",
            "baseline_",
            "optimized_",
        ],
        "reason": "Run quick LLM variants for a benchmark file or target",
    },
    {
        "tool": "recommend",
        "keywords": [
            "optimize",
            "optimization",
            "tune",
            "tuning",
            "speed up",
            "speedup",
            "faster",
            "improve performance",
            "increase throughput",
            "reduce latency",
            "throughput",
            "latency target",
            "goal",
            "recommend",
            "playbook",
        ],
        "reason": "Get an optimization playbook for your goal",
    },
    {
        "tool": "benchmark_variants",
        "keywords": [
            "autotune",
            "auto-tune",
            "auto tune",
            "parameter sweep",
            "grid search",
            "sweep",
            "tile size",
            "block size",
            "kernel tuning",
        ],
        "reason": "Generate and benchmark optimized variants (LLM-assisted)",
    },
    {
        "tool": "benchmark_llm_patch_loop",
        "keywords": ["llm patch loop", "full patch loop", "auto optimize", "one shot optimize", "end-to-end optimize"],
        "reason": "Run the full LLM patch loop with deep-dive comparison",
    },
    {
        "tool": "optimize_roi",
        "keywords": ["roi", "cost benefit", "impact vs effort", "prioritize", "quick wins"],
        "reason": "Rank optimizations by ROI and effort",
    },
    {
        "tool": "optimize_techniques",
        "keywords": ["optimization techniques", "list optimizations", "techniques", "methods", "options"],
        "reason": "List available optimization techniques",
    },
    {
        "tool": "analyze_pareto",
        "keywords": ["compare", "tradeoff", "pareto"],
        "reason": "Compare throughput/latency/memory tradeoffs",
    },
    {
        "tool": "analyze_scaling",
        "keywords": ["scale", "scaling", "scale out", "scale up", "multi-gpu scaling", "strong scaling", "weak scaling"],
        "reason": "Analyze scaling behavior",
    },
    {
        "tool": "analyze_stacking",
        "keywords": ["stack optimizations", "combine optimizations", "optimization stacking", "compatibility"],
        "reason": "Check optimization compatibility and stacking order",
    },
    {
        "tool": "cluster_slurm",
        "keywords": ["slurm", "batch", "sbatch", "job script", "slurm script", "srun"],
        "reason": "Generate SLURM script for cluster runs",
    },
    {
        "tool": "distributed_plan",
        "keywords": ["distributed", "multi node", "tp", "pp", "dp", "fsdp"],
        "reason": "Plan DP/TP/PP strategy",
    },
    {
        "tool": "distributed_nccl",
        "keywords": ["nccl", "tune nccl", "collective", "allreduce", "all-gather", "rdma", "infiniband", "ib"],
        "reason": "Tune NCCL for multi-node",
    },
    {
        "tool": "launch_plan",
        "keywords": ["torchrun", "srun", "launch plan", "launch command", "run command"],
        "reason": "Generate torchrun/srun launch commands",
    },
    {
        "tool": "run_benchmarks",
        "keywords": ["benchmark", "benchmarks", "bench run", "run benchmark", "perf run"],
        "reason": "Run standard benchmarks with optional profiling",
    },
    {
        "tool": "inference_vllm",
        "keywords": ["vllm", "inference", "serving", "throughput", "latency"],
        "reason": "Generate vLLM config for throughput/latency",
    },
    {
        "tool": "inference_deploy",
        "keywords": ["deploy", "deployment", "serving config", "serve model"],
        "reason": "Generate inference deployment configuration",
    },
    {
        "tool": "inference_estimate",
        "keywords": ["estimate throughput", "estimate latency", "throughput estimate", "latency estimate"],
        "reason": "Estimate inference throughput/latency",
    },
    {
        "tool": "inference_quantization",
        "keywords": ["quant", "int8", "fp8", "fp4", "kv cache"],
        "reason": "Quantization guidance for inference",
    },
    {
        "tool": "benchmark_targets",
        "keywords": ["benchmark targets", "bench targets", "list benchmarks", "what can I run"],
        "reason": "List benchmark targets",
    },
    {
        "tool": "list_chapters",
        "keywords": ["list chapters", "chapters list", "chapters"],
        "reason": "List all benchmark chapters and labs",
    },
    {
        "tool": "benchmark_overview",
        "keywords": ["overview", "summary", "latest results", "latest benchmarks"],
        "reason": "Summarize latest benchmark results",
    },
    {
        "tool": "benchmark_history",
        "keywords": ["history", "past runs", "previous runs"],
        "reason": "List historical benchmark runs",
    },
    {
        "tool": "benchmark_trends",
        "keywords": ["trend", "trends", "over time", "performance trend"],
        "reason": "Compute performance trends over time",
    },
    {
        "tool": "benchmark_data",
        "keywords": ["benchmark data", "raw results", "results data", "table view"],
        "reason": "Fetch benchmark results with filtering/pagination",
    },
    {
        "tool": "triage",
        "keywords": ["triage", "start", "first", "status", "health", "quick check"],
        "reason": "Get status + summary context",
    },
    {
        "tool": "status",
        "keywords": ["status", "health", "ready", "check", "sanity"],
        "reason": "Quick status: GPU, software, AI backend",
    },
    {
        "tool": "job_status",
        "keywords": ["job status", "async status", "check job", "poll job"],
        "reason": "Check status of a background job",
    },
    {
        "tool": "ai_status",
        "keywords": ["ai status", "llm status", "api key", "model availability"],
        "reason": "Check AI/LLM backend availability",
    },
    {
        "tool": "ask",
        "keywords": ["question", "why", "how"],
        "reason": "Free-form performance question with citations",
    },
    {
        "tool": "explain",
        "keywords": ["what is", "explain", "concept"],
        "reason": "Explain a performance concept with citations",
    },
    {
        "tool": "ai_troubleshoot",
        "keywords": ["troubleshoot", "error", "failure", "stack trace", "timeout", "nccl timeout"],
        "reason": "Diagnose common training/distributed errors",
    },
    {
        "tool": "ask",
        "keywords": ["flash attention", "torch.compile", "compile", "cuda graphs", "why slow"],
        "reason": "Ask targeted performance questions (FlashAttn, torch.compile, CUDA Graphs, etc.)",
    },
    {
        "tool": "benchmark_targets",
        "keywords": ["list targets", "what benchmarks", "examples", "chapters"],
        "reason": "List available benchmark targets (chapter:example)",
    },
    {
        "tool": "benchmark_report",
        "keywords": ["report", "pdf", "html", "export report"],
        "reason": "Generate PDF/HTML benchmark report",
    },
    {
        "tool": "benchmark_export",
        "keywords": ["export", "csv", "markdown", "json"],
        "reason": "Export benchmark results",
    },
    {
        "tool": "export_csv",
        "keywords": ["export csv", "inline csv", "csv data", "return csv"],
        "reason": "Inline CSV export of benchmark data",
    },
    {
        "tool": "export_html",
        "keywords": ["export html", "inline html", "html data"],
        "reason": "Inline HTML export of benchmark data",
    },
    {
        "tool": "export_pdf",
        "keywords": ["export pdf", "inline pdf", "pdf data"],
        "reason": "Inline PDF export of benchmark data",
    },
    {
        "tool": "benchmark_compare_runs",
        "keywords": ["compare runs", "diff results", "regressions", "improvements"],
        "reason": "Diff two benchmark JSON runs",
    },
    {
        "tool": "hw_roofline",
        "keywords": ["stride", "roofline", "memory sweep"],
        "reason": "Quick stride sweep roofline for memory hierarchy",
    },
    {
        "tool": "gpu_topology",
        "keywords": ["topology", "nvlink", "pcie", "multi gpu"],
        "reason": "Inspect multi-GPU topology",
    },
]


def _score_rule(rule: Dict[str, Any], text: str) -> int:
    score = 0
    for kw in rule.get("keywords", []):
        if kw in text:
            score += 2 if " " in kw else 1
    return score


def _fallback_suggestions() -> List[Dict[str, Any]]:
    return [
        {"tool": "triage", "reason": "Start with triage to gather context"},
        {"tool": "analyze_bottlenecks", "reason": "Check for bottlenecks"},
        {"tool": "recommend", "reason": "Get optimization recommendations"},
    ]


def _normalize_max_suggestions(max_suggestions: Optional[int]) -> Optional[int]:
    if max_suggestions is None:
        return None
    if not isinstance(max_suggestions, int):
        raise ValueError("max_suggestions must be an integer.")
    if max_suggestions < 1:
        raise ValueError("max_suggestions must be >= 1.")
    return max_suggestions


def suggest_tools_heuristic(
    query: str,
    rules: Optional[List[Dict[str, Any]]] = None,
    max_suggestions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    text = (query or "").lower()
    active_rules = rules or DEFAULT_SUGGEST_RULES
    max_suggestions = _normalize_max_suggestions(max_suggestions)

    scored: List[tuple[int, Dict[str, Any]]] = []
    for rule in active_rules:
        score = _score_rule(rule, text)
        if score > 0:
            scored.append((score, rule))

    if not scored:
        suggestions = _fallback_suggestions()
        return suggestions[:max_suggestions] if max_suggestions else suggestions

    scored.sort(key=lambda item: item[0], reverse=True)

    seen = set()
    suggestions: List[Dict[str, Any]] = []
    for score, rule in scored:
        tool = rule.get("tool")
        if not tool or tool in seen:
            continue
        seen.add(tool)
        suggestions.append({"tool": tool, "reason": rule.get("reason", ""), "score": score})

    return suggestions[:max_suggestions] if max_suggestions else suggestions


def suggest_tools_llm(
    query: str,
    rules: Optional[List[Dict[str, Any]]] = None,
    tool_catalog: Optional[List[Dict[str, str]]] = None,
    max_suggestions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    max_suggestions = _normalize_max_suggestions(max_suggestions)
    status = get_llm_status()
    if not status.get("available", False):
        raise RuntimeError(
            "LLM routing requested but no LLM backend is configured. "
            "Set an API key/base URL in .env/.env.local or call ai_status."
        )

    active_rules = rules or DEFAULT_SUGGEST_RULES
    allowed_tools = set()
    catalog: List[Dict[str, Any]] = []

    if tool_catalog:
        for entry in tool_catalog:
            tool = entry.get("tool")
            if not tool or tool in allowed_tools:
                continue
            allowed_tools.add(tool)
            catalog.append(
                {
                    "tool": tool,
                    "description": entry.get("description", ""),
                }
            )
    else:
        for rule in active_rules:
            tool = rule.get("tool")
            if not tool or tool in allowed_tools:
                continue
            allowed_tools.add(tool)
            catalog.append(
                {
                    "tool": tool,
                    "description": rule.get("reason", ""),
                    "keywords": rule.get("keywords", []),
                }
            )

    limit = max_suggestions or 6
    system_prompt = (
        "You are a routing assistant. Choose the most relevant tools from the provided catalog. "
        "Return STRICT JSON: {\"suggestions\": [{\"tool\": \"...\", \"reason\": \"...\"}]}. "
        "Only use tool names from the catalog. Return at most the requested count."
    )
    user_prompt = json.dumps(
        {
            "query": query,
            "max_suggestions": limit,
            "tools": catalog,
        }
    )
    raw = llm_call(prompt=user_prompt, system=system_prompt, json_mode=True)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM routing failed: invalid JSON response ({exc}).") from exc

    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        raise RuntimeError("LLM routing failed: response missing 'suggestions' list.")

    results: List[Dict[str, Any]] = []
    seen = set()
    for idx, entry in enumerate(suggestions):
        if not isinstance(entry, dict):
            raise RuntimeError("LLM routing failed: suggestion entry is not an object.")
        tool = entry.get("tool")
        if not isinstance(tool, str):
            raise RuntimeError("LLM routing failed: suggestion tool must be a string.")
        tool = tool.strip()
        if tool not in allowed_tools:
            raise RuntimeError(f"LLM routing failed: unknown tool '{tool}'.")
        if tool in seen:
            continue
        seen.add(tool)
        reason = entry.get("reason") if isinstance(entry.get("reason"), str) else ""
        results.append({"tool": tool, "reason": reason, "score": 100 - idx})

    if not results:
        raise RuntimeError("LLM routing failed: no valid suggestions returned.")

    return results[:max_suggestions] if max_suggestions else results


def suggest_tools_auto(
    query: str,
    llm_routing: bool = True,
    rules: Optional[List[Dict[str, Any]]] = None,
    tool_catalog: Optional[List[Dict[str, str]]] = None,
    max_suggestions: Optional[int] = None,
) -> Dict[str, Any]:
    """Auto-route with LLM, fall back to heuristics with warning."""
    if llm_routing:
        status = get_llm_status()
        if not status.get("available", False):
            warning = (
                "WARNING: LLM routing requested but no LLM backend is configured. "
                "Falling back to keyword heuristics."
            )
            suggestions = suggest_tools_heuristic(
                query,
                rules,
                max_suggestions=max_suggestions,
            )
            return {
                "suggestions": suggestions,
                "routing": "heuristic",
                "warning": warning,
                "llm_available": False,
            }
        try:
            suggestions = suggest_tools_llm(
                query,
                rules,
                tool_catalog=tool_catalog,
                max_suggestions=max_suggestions,
            )
            return {
                "suggestions": suggestions,
                "routing": "llm",
                "llm_available": True,
            }
        except RuntimeError as exc:
            warning = (
                f"WARNING: LLM routing failed ({exc}). "
                "Falling back to keyword heuristics."
            )
            suggestions = suggest_tools_heuristic(
                query,
                rules,
                max_suggestions=max_suggestions,
            )
            return {
                "suggestions": suggestions,
                "routing": "heuristic",
                "warning": warning,
                "llm_available": True,
            }

    suggestions = suggest_tools_heuristic(
        query,
        rules,
        max_suggestions=max_suggestions,
    )
    return {
        "suggestions": suggestions,
        "routing": "heuristic",
        "warning": "WARNING: LLM routing disabled (llm_routing=false). Using keyword heuristics.",
        "llm_available": None,
    }
