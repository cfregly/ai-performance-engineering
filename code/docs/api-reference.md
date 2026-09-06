# AI Systems Performance - Unified API Reference

This document describes the unified 10-domain API that powers all interfaces (CLI, MCP, Dashboard, Python API).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Systems Performance                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Interfaces                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────────┐  ┌─────────────┐               │
│   │   CLI   │  │   MCP   │  │  Dashboard  │  │  Python API │               │
│   │  aisp   │  │  Tools  │  │   Web UI    │  │   Direct    │               │
│   └────┬────┘  └────┬────┘  └──────┬──────┘  └──────┬──────┘               │
│        │            │              │                │                       │
│        └────────────┴──────────────┴────────────────┘                       │
│                              │                                              │
│   ┌──────────────────────────▼──────────────────────────┐                   │
│   │           PerformanceEngine (core/engine.py)         │                   │
│   │                                                      │                   │
│   │   10 Domains:                                        │                   │
│   │   ┌─────┐ ┌──────┐ ┌───────┐ ┌───────┐ ┌────────┐   │                   │
│   │   │ gpu │ │system│ │profile│ │analyze│ │optimize│   │                   │
│   │   └─────┘ └──────┘ └───────┘ └───────┘ └────────┘   │                   │
│   │   ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌──┐ ┌────┐ │                   │
│   │   │distributed│ │inference│ │benchmark│ │ai│ │exp │ │                   │
│   │   └───────────┘ └─────────┘ └─────────┘ └──┘ └────┘ │                   │
│   └──────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Python API

```python
from core.engine import get_engine

engine = get_engine()

# GPU info
engine.gpu.info()
engine.gpu.bandwidth()

# Analysis
engine.analyze.bottlenecks()
engine.analyze.pareto()

# Optimization
engine.optimize.recommend(model_size=70, gpus=8)

# AI-powered
engine.ai.ask("Why is my attention kernel slow?")
```

### CLI

```bash
# System info
python -m cli.aisp system status
python -m cli.aisp gpu info

# Optimization recommendations
python -m cli.aisp optimize recommend --model-size 70 --gpus 8

# AI questions
python -m cli.aisp ai ask "Why is my kernel slow?"

# Profiling
python -m cli.aisp profile nsys python train.py
```

The command tables below use the installed-console shorthand `aisp ...`. When you are running directly from a repo checkout, the equivalent invocation is `python -m cli.aisp ...`.

### MCP Tools (for AI assistants)

```
gpu_info          - Get GPU hardware info
analyze_bottlenecks - Identify performance issues
optimize          - Run quick LLM benchmark variants from a target or file path
benchmark_explore - Copy a baseline benchmark, run LLM variants, compare utilization
recommend         - Get optimization recommendations
ask              - Ask performance questions
```

---

## Domain Reference

### 1. GPU Domain

Hardware information, topology, power management, and bandwidth testing.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `info()` | GPU name, memory, temperature, utilization | `aisp gpu info` | `gpu_info` |
| `topology()` | Multi-GPU topology, NVLink, P2P matrix | `aisp gpu topology` | `gpu_topology` |
| `power()` | Power draw, limits, thermal status | `aisp gpu power` | `gpu_power` |
| `bandwidth()` | Memory bandwidth test (HBM) | `aisp gpu bandwidth` | `gpu_bandwidth` |
| `topology_matrix()` | Raw `nvidia-smi topo -m` matrix | - | `gpu_topology_matrix` |

**Python API:**
```python
engine.gpu.info()           # GPU info dict
engine.gpu.topology()       # Topology matrix
engine.gpu.bandwidth()      # Bandwidth test results
engine.gpu.power()          # Power/thermal info
```

---

### 2. System Domain

Software stack, dependencies, and environment information.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `software()` | PyTorch, CUDA, Python versions | `aisp system software` | `system_software` |
| `dependencies()` | ML/AI dependency health | `aisp system deps` | `system_dependencies` |
| `capabilities()` | Hardware features (TMA, FP8, tensor cores) | `aisp system capabilities` | `system_capabilities` |
| `context()` | Full system context for AI analysis | `aisp system context` | `system_context` |
| `parameters()` | Kernel parameters affecting performance | `aisp system parameters` | `system_parameters` |
| `container()` | Container/cgroup limits | `aisp system container` | `system_container` |
| `cpu_memory()` | CPU/NUMA/cache hierarchy snapshot | `aisp system cpu-memory` | `system_cpu_memory` |
| `env()` | Environment variables + paths | `aisp system env` | `system_env` |
| `network()` | Network + InfiniBand status | `aisp system network` | `system_network` |
| `clock_lock_check(...)` | Validate harness GPU clock locking works | `aisp system clock-lock-check` | `clock_lock_check` |

**Python API:**
```python
engine.system.software()      # Software versions
engine.system.dependencies()  # Dependency health
engine.system.capabilities()  # Hardware capabilities
engine.system.context()       # Full context for AI
engine.system.parameters()    # Kernel/system parameters
engine.system.container()     # Container/cgroup limits
engine.system.cpu_memory()    # CPU/NUMA/cache snapshot
engine.system.env()           # Environment variables
engine.system.network()       # Network/IB status
engine.system.clock_lock_check()  # Validate clock locking works (benchmark validity gate)
```

**MCP-only system tools:** `system_full`, `context_summary`, `context_full`, `status`, `triage`.

---

### 3. Profile Domain

Profiling with Nsight Systems, Nsight Compute, and torch.profiler.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `nsys(command)` | Nsight Systems capture | `aisp profile nsys` | `profile_nsys` |
| `ncu(command)` | Nsight Compute capture | `aisp profile ncu` | `profile_ncu` |
| `torch(script)` | torch.profiler capture summary | `aisp profile torch` | `profile_torch` |
| `hta(command)` | HTA capture + analysis | `aisp profile hta-capture` | `profile_hta` |
| `flame_graph()` | Flame graph visualization data | `aisp profile flame` | `profile_flame` |
| `kernels()` | Kernel execution breakdown | `aisp profile kernels` | `profile_kernels` |
| `memory_timeline()` | Memory allocation timeline | `aisp profile memory` | `profile_memory` |
| `roofline()` | Roofline model data | `aisp profile roofline` | `profile_roofline` |
| `compare(chapter)` | Compare baseline vs optimized | `aisp profile compare` | `profile_compare` |
| `list_profiles()` | List available profile pairs | `aisp profile list-profiles` | `profile_list_profiles` |
| `compile_analysis()` | torch.compile analysis summary | `aisp profile compile-analysis` | `profile_compile_analysis` |
| `ncu_summary(report_path, ...)` | Top-N kernel summary from an NCU report | `aisp profile ncu-summary` | `ncu_summary` |

**MCP profiling captures include metrics JSON:** `profile_nsys` returns `nsys_metrics`, `profile_ncu` returns `ncu_metrics`, `profile_torch` returns `torch_metrics` (and `report` alias), and `profile_hta` includes `nsys_metrics`. Use these payloads to analyze regressions and bottleneck shifts.

**Targeted NCU capture (CLI + MCP):** `profile_ncu` supports kernel scoping (`kernel_filter`, optional `kernel_name_base`) plus NVTX gating (`nvtx_include`, `profile_from_start='off'`) to isolate specific kernels and avoid setup-noise captures. Captures now fail loudly when NCU profiles zero kernels or collects zero metrics, and `metric_set='minimal'` auto-resolves to `speed-of-light` or `basic` depending on Nsight Compute version. `compare_ncu` flags rank-only kernel symbol alignment as low-confidence for tuning (advisory), applies kernel-family alias matching, and returns a concrete `staged_pair_dir` for stable follow-up diffs.

**Pair-health + manifests (CLI + MCP):** profile comparisons now emit `pair_health` metadata (presence/absence of baseline/optimized NSYS/NCU pairs), and comparison staging writes `pair_manifest.json` so downstream automation can consume deterministic pair context without re-discovering files.

**NSYS timeout hardening (CLI + MCP + harness):** all NSYS entrypoints now route through `NsightAutomation.profile_nsys`, default to the safer `preset='light'`, support `wait_mode` (`primary`/`all`), and use a graceful timeout finalization window (`finalize_grace_seconds`) before hard termination. `profile_nsys` also supports `sanitize_python_startup=true` to prefix a safe `sitecustomize` shim for profiler subprocesses.

**Python API:**
```python
engine.profile.flame_graph()      # Flame graph data
engine.profile.kernels()          # Kernel breakdown
engine.profile.memory_timeline()  # Memory allocations
engine.profile.hta()              # HTA analysis
engine.profile.compare("ch11")    # Compare profiles
engine.profile.list_profiles()    # Discover available profile pairs
engine.profile.compile_analysis() # torch.compile graphs/breaks (best-effort)
engine.profile.ncu_summary("path/to/report.ncu-rep", top_k=10)  # Top kernels from NCU
```

---

## Cluster Tools (Non-Domain)

These are repo helpers for cluster evaluations under `cluster/` (field report workflows). They are intentionally separate from the unified 10-domain engine.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `run_cluster_eval_suite(...)` | Run cluster eval suite (full) or a safe local smoke run | `aisp cluster eval-suite` | `cluster_eval_suite` |
| `run_cluster_common_eval(...)` | Run a preset cluster evaluation bundle | `aisp cluster common-eval` | `cluster_common_eval` |
| `run_cluster_fabric_eval(...)` | Run the canonical fabric evaluation bundle | `aisp cluster fabric-eval` | `cluster_fabric_eval` |
| `build_canonical_package(...)` | Build a canonical comparison/history package | `aisp cluster build-canonical-package` | `cluster_build_canonical_package` |
| `promote_cluster_run(...)` | Promote `cluster/runs/<run_id>/` into `cluster/published/current` | `aisp cluster promote-run` (optional `--repo-root` for alternate trees) | `cluster_promote_run` |
| `watch_cluster_run_for_promotion(...)` | Detached watcher: wait on PID, then promote | `aisp cluster watch-promote` (optional `--repo-root`) | `cluster_watch_promote` |
| `validate_field_report_requirements(...)` | Validate `cluster/field-report*.md` requirements + artifact hygiene | `aisp cluster validate-field-report` | `cluster_validate_field_report` |

**HTTP:** `POST /api/cluster/promote-run` and `POST /api/cluster/watch-promote` accept the same JSON fields as the MCP tools (including optional `repo_root`). The watch-promote route is registered for programmatic use; it is not part of the dashboard’s restricted MCP tool subset.

---

### 4. Analyze Domain

Performance analysis, bottleneck detection, and what-if scenarios.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `bottlenecks(mode)` | Identify performance bottlenecks | `aisp analyze bottlenecks` | `analyze_bottlenecks` |
| `pareto()` | Pareto frontier (throughput vs latency vs memory) | `aisp analyze pareto` | `analyze_pareto` |
| `scaling()` | Scaling analysis with GPU count | `aisp analyze scaling` | `analyze_scaling` |
| `whatif(...)` | What-if constraint analysis | `aisp analyze whatif` | `analyze_whatif` |
| `stacking()` | Optimization stacking compatibility | `aisp analyze stacking` | `analyze_stacking` |
| `comm_overlap()` | Communication/compute overlap analysis | - | `analyze_comm_overlap` |
| `memory_patterns()` | Memory access pattern analysis | - | `analyze_memory_patterns` |
| `dataloader()` | DataLoader bottleneck analysis | - | `analyze_dataloader` |
| `energy()` | Energy efficiency analysis | - | `analyze_energy` |
| `predict_scaling()` | Predict scaling to more GPUs | - | `predict_scaling` |

**Python API:**
```python
engine.analyze.bottlenecks()                    # Bottleneck analysis
engine.analyze.bottlenecks(mode="llm")          # AI-powered analysis
engine.analyze.pareto()                         # Pareto frontier
engine.analyze.scaling()                        # Scaling analysis
engine.analyze.whatif(max_vram_gb=24)           # What-if scenarios
engine.analyze.stacking()                       # Technique compatibility
```

---

### 5. Optimize Domain

Optimization recommendations and technique analysis.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `recommend(model_size, gpus, goal)` | Get recommendations | `aisp optimize recommend` | `recommend` |
| `techniques()` | List all optimization techniques | `aisp optimize techniques` | `optimize_techniques` |
| `roi()` | Calculate optimization ROI | `aisp optimize roi` | `optimize_roi` |

**MCP-only optimize workflow:** `optimize` accepts `path` or `target` and runs quick LLM variants (benchmark_variants defaults).

**MCP-only deep-dive workflow:** `benchmark_deep_dive_compare` accepts `targets` or a benchmark `path` (baseline_*/optimized_*) and runs one-shot deep-dive profiling + compare.

**Python API:**
```python
engine.optimize.recommend(model_size=70, gpus=8)
engine.optimize.recommend(model_size=7, goal="memory")
engine.optimize.techniques()                     # All techniques
engine.optimize.roi()                            # ROI analysis
```

---

### 6. Distributed Domain

Distributed training: parallelism planning, NCCL tuning, FSDP configuration.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `plan(model_size, gpus, nodes)` | Plan parallelism strategy | `aisp distributed plan` | `distributed_plan` |
| `topology()` | Parallelism topology snapshot | `aisp distributed topology` | - |
| `nccl(nodes, gpus)` | NCCL tuning recommendations | `aisp distributed nccl` | `distributed_nccl` |
| `fsdp(model)` | FSDP configuration | `aisp distributed zero` | - |
| `tensor_parallel(model)` | Tensor parallelism config | - | - |
| `pipeline(model)` | Pipeline parallelism config | - | - |
| `launch_plan(...)` | Generate torchrun launch plan | `aisp distributed launch-plan` | `launch_plan` |
| `slurm(...)` | Generate SLURM script | `aisp distributed slurm` | `cluster_slurm` |
| `cost_estimate(...)` | Cloud cost estimation | - | `cost_estimate` |

**Python API:**
```python
engine.distributed.plan(model_size=70, gpus=16, nodes=2)
engine.distributed.topology()
engine.distributed.nccl(nodes=2, gpus=8)
engine.distributed.slurm(model="70b", nodes=4, gpus=8)
engine.distributed.cost_estimate(gpu_type="h100", num_gpus=4, hours_per_day=8)
```

---

### 7. Inference Domain

Inference optimization: vLLM configuration, quantization, deployment.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `vllm_config(model, target)` | Generate vLLM configuration (model size required) | `aisp inference vllm` | `inference_vllm` |
| `quantization(model_size)` | Quantization recommendations | `aisp inference quantize` | `inference_quantization` |
| `deploy(params)` | Deployment configuration | `aisp inference deploy` | `inference_deploy` |
| `estimate(params)` | Inference performance estimate | `aisp inference estimate` | `inference_estimate` |

**Python API:**
```python
engine.inference.vllm_config(model="llama-70b", model_params_b=70, target="throughput")
engine.inference.quantization(model_size=70)    # FP8, INT8, INT4 options
engine.inference.deploy({"model": "70b", "model_size": 70, "num_gpus": 4})
engine.inference.estimate({"model": "70b", "model_size": 70, "num_gpus": 4})
```

**CLI-only convenience:** `aisp inference serve` generates (and optionally runs) a launch command.

---

### 8. Benchmark Domain

Benchmark execution, history tracking, and result comparison.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `run(targets, profile)` | Run benchmarks | `aisp bench run` | `run_benchmarks` |
| `run_e2e(...)` | Run the all-stages tier1/full-sweep/cluster/fabric orchestration flow | `aisp bench run-e2e` | `benchmark_e2e_sweep` |
| `e2e_status(run_id)` | Inspect the authoritative normalized live-status surface for an e2e run | `aisp bench run-e2e-status` | `benchmark_e2e_status` |
| `e2e_watch(run_id)` | Arm the detached watcher that auto-resumes stale or aborted e2e runs | `aisp bench watch-e2e` | `benchmark_e2e_watch` |
| `targets()` | List benchmark targets | `aisp bench list-targets` | `benchmark_targets` |
| `explore(path, ...)` | Copy baseline, run LLM variants w/ profiling | `aisp bench explore` | `benchmark_explore` |
| `history()` | Historical benchmark runs | - | `benchmark_history` |
| `data()` | Load benchmark results (filtered/paged) | - | `benchmark_data` |
| `overview()` | Summary of latest results | - | `benchmark_overview` |
| `trends()` | Performance trends over time | - | `benchmark_trends` |
| `compare(params)` | Compare two benchmark runs (dashboard-style diff) | - | `benchmark_compare` |
| `compare_runs(baseline, candidate)` | Compare two benchmark runs (bench CLI diff) | `aisp bench compare-runs` | `benchmark_compare_runs` |
| `speed_test()` | Quick speed tests (diagnostic) | `aisp benchmark speed` | `hw_speed` |

**Explore workflow:** `aisp bench explore` / `benchmark_explore` copies a `baseline_*.py` (or a `baseline_*.cu` with auto-generated wrapper), runs minimal profiling with LLM patch variants, compares utilization deltas, and triggers deep_dive when minimal results are inconclusive.

**E2E status workflow:** prefer `aisp bench run-e2e-status` / `benchmark_e2e_status` for live progress. It joins top-level progress, child benchmark progress, recent events, watcher state, and process liveness. Raw `summary.json` / `checkpoint.json` can lag during long child stages; the run package now advertises this via `preferred_progress_source` and `actions`.

**Python API:**
```python
engine.benchmark.run(targets=["ch07", "ch11"], profile="minimal")
engine.benchmark.targets()                      # Available targets
engine.benchmark.history()                      # Historical runs
engine.benchmark.data()                         # Current results
engine.benchmark.overview()                     # Summary stats
engine.benchmark.speed_test()                   # Quick GEMM/attention test
```

**Runner notes:**
- CUDA benchmarks execute via Python wrappers (`CudaBinaryBenchmark`). Direct `.cu` runs are not supported.
- Ad-hoc script runs should use `benchmark_main` (supports `--iterations`, `--warmup`, and defaults to `--force-sync`; disable with `--no-force-sync` or `AISP_FORCE_SYNC=0`).
- Benchmark validity profile defaults to strict across CLI/MCP/dashboard (`--validity-profile strict` / `validity_profile=\"strict\"`).
- The portable profile relaxes clock locking, application-clock verification, GPU telemetry, and virtualization reporting. It labels the result as noncanonical. It does not bypass hard failures such as unsupported platforms, swap, CPU governor, cgroup limits, missing CUDA, or foreign GPU processes (`validity_profile=\"portable\"` in MCP).
- Portable validity profile does not update expectation files unless explicitly enabled (`--allow-portable-expectations-update` / `allow_portable_expectations_update=true`).
- Use `--force-sync` (CLI) / `force_sync=true` (MCP) to insert a device-wide synchronize after each `benchmark_fn()` when you need an extra safety net outside standard harness timing.
- Safer profiling defaults are `--ncu-metric-set minimal` and `--ncu-replay-mode kernel` (same defaults in MCP `run_benchmarks`).
- `--ncu-replay-mode app-range` captures one complete NVTX start/stop range using the five minimal metrics. It requires an explicit range and rejects kernel filters, launch limits and PM sampling. The KV-standard pair prefers this mode locally because per-kernel application replay repeats its full setup and decode workload many times. Each accepted range capture retains report SHA-256, report-carried replay mode/range identity, metric units and full selected-range policy. Its `ncu_range_time_ms` is a profiler measurement and does not represent benchmark latency or an individual kernel time. Timing, correctness checks and Nsight Systems remain separate requirements.
- MCP `run_benchmarks` emits `speedup_attribution` from captured profiler metrics when available, without requiring an LLM backend.
- Use `--only-cuda` / `only_cuda=true` to run only CUDA binary wrappers, or `--only-python` / `only_python=true` to skip them.

**Note:** `aisp benchmark ...` commands are diagnostic microbenchmarks (`hw_*` tools) and do not use the harness.

---

### 9. AI Domain

LLM-powered analysis, questions, and explanations.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `ask(question)` | Ask a performance question | `aisp ai ask` | `ask` |
| `explain(concept)` | Explain a concept | `aisp ai explain` | `explain` |
| `analyze_kernel(code)` | AI kernel analysis | - | - |
| `troubleshoot(issue)` | Diagnose errors & fixes | `aisp ai troubleshoot` | `ai_troubleshoot` |
| `suggest_tools(query)` | Suggest tools for a task | - | `suggest_tools` |
| `status()` | AI/LLM availability | `aisp ai status` | `ai_status` |

**Python API:**
```python
engine.ai.ask("Why is my attention kernel slow?")
engine.ai.ask("How do I fix CUDA OOM?", include_citations=True)
engine.ai.explain("flash-attention")
engine.ai.troubleshoot("NCCL timeout")
engine.ai.suggest_tools("I keep OOMing on 24GB VRAM")
engine.ai.suggest_tools("Profile and compare baseline vs optimized", llm_routing=True)
engine.ai.status()                              # Check LLM availability
```

**Routing options:** `suggest_tools` defaults to LLM-based routing. If no LLM backend is configured, it automatically falls back to keyword heuristics and includes a WARNING in the response. Set `llm_routing=false` to force heuristics (also returns a WARNING). Use `max_suggestions` to cap results.

---

### 10. Export Domain

Export reports in various formats.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `csv(detailed)` | Export to CSV | `aisp bench export --format csv` | `export_csv` |
| `pdf()` | Generate PDF report | `aisp bench report --format pdf` | `export_pdf` |
| `html()` | Generate HTML report | `aisp bench report --format html` | `export_html` |

**Python API:**
```python
csv_content = engine.export.csv()               # Basic CSV
csv_detailed = engine.export.csv(detailed=True) # Detailed CSV
pdf_bytes = engine.export.pdf()                 # PDF report
html_content = engine.export.html()             # HTML report
```

---

## Convenience Methods

The engine also provides top-level convenience methods:

```python
engine.status()           # Quick system status (GPU + software + AI)
engine.triage()           # Status + context + next steps
engine.ask(question)      # Shortcut to engine.ai.ask()
engine.recommend(...)     # Shortcut to engine.optimize.recommend()
engine.list_domains()     # List all domains and operations
```

---

## Interface Mapping Summary

| Engine Domain | CLI Command Group | MCP Tool Prefix | Dashboard API |
|---------------|-------------------|-----------------|---------------|
| `gpu` | `aisp gpu` | `gpu_*` | `/api/gpu/*` |
| `system` | `aisp system` | `system_*` | `/api/system/*` |
| `profile` | `aisp profile` | `profile_*` | `/api/profile/*` |
| `analyze` | `aisp analyze` | `analyze_*` | - (not exposed) |
| `optimize` | `aisp optimize` | `optimize`, `optimize_*`, `recommend` | - (not exposed) |
| `distributed` | `aisp distributed` | `distributed_*` | - (not exposed) |
| `inference` | `aisp inference` | `inference_*` | - (not exposed) |
| `benchmark` | `aisp bench` | `benchmark_*`, `run_benchmarks` | `/api/benchmark/*` |
| `ai` | `aisp ai` | `ask`, `explain`, `ai_*` | `/api/ai/*` |
| `export` | `aisp bench report/export` | `export_*` | - (not exposed) |

---

**Dashboard API note:** The dashboard backend intentionally exposes only the subset of endpoints used by the UI. See `core/api/registry.py` for the authoritative list.

## Error Handling

All operations return dictionaries with consistent structure:

```python
# Success
{
    "success": True,
    "data": {...},
    "timestamp": "2024-01-01T12:00:00Z"
}

# Error
{
    "success": False,
    "error": "Error message",
    "error_type": "AttributeError"
}
```

---

## Best Practices

### 1. Start with Triage

```python
# Get full context before diving in
context = engine.triage()
```

### 2. Use Domain-Specific Methods

```python
# Good: Clear domain separation
engine.analyze.bottlenecks()
engine.optimize.recommend(model_size=70)

# Avoid: Using internal methods directly
engine._handler.detect_bottlenecks()  # Don't do this
```

### 3. Combine AI with Data

```python
# Get hard data first
bottlenecks = engine.analyze.bottlenecks(mode="profile")

# Then ask AI for interpretation
explanation = engine.ai.ask(f"Explain these bottlenecks: {bottlenecks}")
```

### 4. Use Consistent Parameters

```python
# Parameters are consistent across interfaces
# CLI:       aisp optimize recommend --model-size 70 --gpus 8
# Python:    engine.optimize.recommend(model_size=70, gpus=8)
# MCP:       recommend with {"model_size": 70, "gpus": 8}
```
