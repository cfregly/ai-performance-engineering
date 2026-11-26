# AI Systems Performance - Unified API Reference

This document describes the unified 10-domain API used by the CLI, MCP tools, and Python API.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Systems Performance                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Interfaces                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────────┐                                 │
│   │   CLI   │  │   MCP   │  │  Python API │                                 │
│   │  aisp   │  │  Tools  │  │   Direct    │                                 │
│   └────┬────┘  └────┬────┘  └──────┬──────┘                                 │
│        │            │              │                                        │
│        └────────────┴──────────────┘                                        │
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
| `kernels()` | Kernel execution breakdown | Not implemented (`aisp profile kernels` exits 1) | `profile_kernels` |
| `memory_timeline()` | Memory allocation timeline | Not implemented (`aisp profile memory` exits 1) | `profile_memory` |
| `roofline()` | Roofline model data | `aisp profile roofline` | `profile_roofline` |
| `compare(chapter)` | Compare baseline vs optimized | `aisp profile compare` | `profile_compare` |
| `list_profiles()` | List available profile pairs | `aisp profile list-profiles` | `profile_list_profiles` |
| `compile_analysis()` | torch.compile analysis summary | `aisp profile compile-analysis` | `profile_compile_analysis` |
| `ncu_summary(report_path, ...)` | Top-N kernel summary from an NCU report | `aisp profile ncu-summary` | `ncu_summary` |

**MCP profiling captures include metrics JSON:** `profile_nsys` returns `nsys_metrics`, `profile_ncu` returns `ncu_metrics`, `profile_torch` returns `torch_metrics` (and `report` alias), and `profile_hta` includes `nsys_metrics`. Use these payloads to analyze regressions and bottleneck shifts.

**Targeted NCU capture (CLI + MCP):** `profile_ncu` supports kernel scoping (`kernel_filter`, optional `kernel_name_base`) plus NVTX gating (`nvtx_include`, `profile_from_start='off'`) to isolate specific kernels and avoid setup-noise captures. Captures fail loudly when NCU profiles zero kernels or collects zero metrics. `metric_set='minimal'` requests exactly the repository's five metrics without an additional NVIDIA section set, keeping replay overhead bounded; choose `basic` explicitly for NVIDIA's broader basic sections. `compare_ncu` flags rank-only kernel symbol alignment as low-confidence for tuning (advisory), applies kernel-family alias matching, and returns a concrete `staged_pair_dir` for stable follow-up diffs.

**Pair-health + manifests (CLI + MCP):** profile comparisons now emit `pair_health` metadata (presence/absence of baseline/optimized NSYS/NCU pairs), and comparison staging writes `pair_manifest.json` so downstream automation can consume deterministic pair context without re-discovering files.

**NSYS timeout hardening (CLI + MCP + harness):** all NSYS entrypoints now route through `NsightAutomation.profile_nsys`, default to the safer `preset='light'`, support `wait_mode` (`primary`/`all`), and use a graceful timeout finalization window (`finalize_grace_seconds`) before hard termination. `profile_nsys` also supports `sanitize_python_startup=true` to prefix a safe `sitecustomize` shim for profiler subprocesses.

The Python benchmark Nsys wrapper honors explicit `BenchmarkConfig.profiling_warmup`
and `profiling_iterations`: warmups run before capture, and measured calls run
inside the profile range. If unset, this wrapper retains one warmup and one
measured call. Counts must be integers, with nonnegative warmups and positive
measured iterations. Reused serving engines can therefore warm up completely
before a steady-state mechanism capture.

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

---

### 4. Analyze Domain

Performance analysis, bottleneck detection, and what-if scenarios.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `bottlenecks(mode, focus)` | Rule-based bottlenecks in the newest captured Chrome trace, with an optional LLM explanation | `aisp analyze bottlenecks` | `analyze_bottlenecks` |
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
engine.analyze.bottlenecks(mode="llm")          # LLM explanation of the trace findings
engine.analyze.bottlenecks(mode="llm", focus="memory")  # LLM explanation focused on memory
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
| `compare(params)` | Compare two benchmark runs | - | `benchmark_compare` |
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

`engine.benchmark.run()` currently returns a plan or a CLI recommendation. It
does not execute a benchmark. Use the CLI, MCP `run_benchmarks`, or the Python
`core.harness.run_benchmarks.test_chapter()` executor for actual execution.

**Benchmark deadlines:** MCP benchmark calls have three separate deadline layers.

| Field | Scope | Omitted or `null` | `0` | Positive integer |
|-------|-------|-------------------|-----|------------------|
| MCP `timeout_seconds` | The bench CLI process launched by the MCP tool | No outer deadline for a synchronous call | No outer deadline for a synchronous call | Exact outer deadline in seconds |
| MCP `suite_timeout` / CLI `--suite-timeout` | The complete benchmark suite inside the CLI process | Retain the CLI suite default | No suite deadline | Exact suite deadline in seconds |
| MCP `execution_timeout_seconds` / CLI `--execution-timeout-seconds` / Python `BenchmarkConfig` | Setup, warmup, measurement and legacy execution stages | Retain the merged stage defaults and timeout multipliers | No stage deadline when the execution path supports an owned subprocess | Exact stage deadline in seconds |

For synchronous `run_benchmarks`, omitting `timeout_seconds`, setting it to `0`,
or setting it to `null` removes only the outer MCP deadline. Suite, execution-stage,
profiler, compiler and benchmark-declared executable deadlines still apply. A
positive `timeout_seconds` limits the MCP-owned bench CLI process. It does not
replace the suite or execution-stage settings.

An asynchronous benchmark job must provide an explicit positive
`timeout_seconds`. The queued worker runs in a thread that cannot receive
operator cancellation. This rule also applies to `benchmark_variants`,
`optimize`, `benchmark_deep_dive_compare`, `benchmark_explore` and
`benchmark_llm_patch_loop`. Synchronous calls to these tools accept an omitted,
zero or `null` outer timeout. `benchmark_variants` and `optimize` forward the
field to `run_benchmarks`. `benchmark_deep_dive_compare` applies it to its
benchmark run. `benchmark_explore` applies `timeout_seconds` to the minimal run
and `deep_dive_timeout_seconds` to its optional deep-dive run.
An asynchronous `benchmark_explore` call with `deep_dive=auto` or
`deep_dive=always` must provide a positive value for both fields. The queued
worker may enter either benchmark stage and cannot receive operator cancellation.
`benchmark_llm_patch_loop` applies the value separately to the patch run and
each comparison run. These workflow fields are not cumulative deadlines for
later in-process analysis.

For execution stages, a positive `execution_timeout_seconds` sets the exact
setup, warmup, measurement and legacy execution deadlines after benchmark config
merging and isolated-child reconstruction. It is not multiplied again. These
values do not give every phase a fresh, independent budget. The enclosing worker
still uses the measurement limit and the existing runtime-provenance allowance.
Finite values beyond the platform's supported wait range are rejected before
launching a worker. The override does not replace internal compiler or executable
deadlines declared by a benchmark.

Setting `execution_timeout_seconds=0` selects uncapped execution through an owned
Python subprocess. Direct thread and torchrun execution reject that request
because they do not provide the same owned-process cancellation path. Select this
override at the calling interface. A benchmark's `get_config()` cannot enable
uncapped execution. A raw zero measurement or legacy deadline requires the
caller's explicit `execution_timeout_seconds=0`.

Uncapped execution stages also require `suite_timeout=0` and an explicit
`timeout_seconds=0` or `timeout_seconds=null`. An omitted outer timeout is
normally equivalent for a synchronous call, but this combination requires the
field to be present so the no-deadline choice is explicit. A finite enclosing
deadline is rejected before work starts because it could abandon the uncapped
worker. MCP background jobs reject uncapped execution because they have no
cancellation endpoint. Use the synchronous MCP call or CLI. The opt-in stale
timing watchdog also rejects uncapped execution because it can terminate work
independently.
On interruption, the owned process group receives SIGINT and the caller waits
for its cleanup without adding a forced deadline. The caller retains bytes it
has read from stdout and stderr. Output buffered by a child but never flushed is
outside this guarantee.
An unresponsive child can keep the caller waiting. Profiler deadlines and
correctness, environment and performance acceptance gates remain unchanged.
The verified isolated worker runs uncapped work on its main thread so an
interrupt unwinds through teardown without starting later iterations. When a
worker completes outside cancellation or coordinator-abort cleanup, helpers left
in its dedicated group receive SIGKILL before its process identity is released.
Cancellation and abort cleanup wait cooperatively. This does not limit execution
time. Cooperative cleanup requires POSIX process groups, `waitid` with `WNOWAIT`,
and the repository's `psutil` dependency. SIGINT must not be ignored, SIGCHLD
must use its default disposition, and the caller must be the sole reaper for the
worker. Unsupported signal dispositions are rejected before launch. The receipt's
cleanup scope covers only members of the worker's process group. A child launcher
remains responsible for waiting for and cleaning up peers that start another
session or process group. If an out-of-group descendant keeps a capture pipe open
after the owned group becomes quiescent, the coordinator drains currently
available bytes and closes its read end. The receipt records this boundary in
`execution.capture_limit`, including the affected streams and retained byte
counts. Later output from that out-of-group process is outside the receipt. If the
coordinator cannot establish cleanup, the sweep stops with the original error and
retained worker diagnostics. It does not start the next benchmark. Stock
`CudaBinaryBenchmark` implementations reject an explicit
`execution_timeout_seconds=0` because their build and executable paths have
independent deadlines and process ownership.
Both communication labs support uncapped execution through owned launchers.
Their harness commands require `execution_timeout_seconds=0` because a finite
outer worker timeout cannot safely clean up the MPI launcher's separate process
group. Their standalone native launchers remain available. Their build and launch
paths do not add a deadline. Profiler and verification policies remain independent.

**Runner notes:**
- CUDA benchmarks execute via Python wrappers (`CudaBinaryBenchmark`). Direct `.cu` runs are not supported.
- Ad-hoc script runs should use `benchmark_main` (supports `--iterations`, `--warmup`, and defaults to `--force-sync`; disable with `--no-force-sync` or `AISP_FORCE_SYNC=0`).
- Benchmark validity profile defaults to strict across CLI and MCP (`--validity-profile strict` / `validity_profile=\"strict\"`).
- The portable profile does not request GPU clock control, including persistence-mode or application-clock changes. It relaxes application-clock verification, GPU telemetry, and virtualization reporting, and labels the result as noncanonical. It does not bypass hard failures such as unsupported platforms, swap, CPU governor, cgroup limits, missing CUDA, or foreign GPU processes (`validity_profile=\"portable\"` in MCP).
- Portable validity profile does not update expectation files unless explicitly enabled (`--allow-portable-expectations-update` / `allow_portable_expectations_update=true`).
- Use `--force-sync` (CLI) / `force_sync=true` (MCP) to insert a device-wide synchronize after each `benchmark_fn()` when you need an extra safety net outside standard harness timing.
- Safer profiling defaults are `--ncu-metric-set minimal` and `--ncu-replay-mode kernel` (same defaults in MCP `run_benchmarks`).
- `--ncu-replay-mode app-range` captures one complete NVTX start/stop range using the five minimal metrics. It requires an explicit range and rejects kernel filters, launch limits and PM sampling. The command uses the explicit metric list without a section set, which would add extra metrics and replay passes. The KV-standard pair prefers this mode locally because per-kernel application replay repeats its full setup and decode workload many times. Each accepted range capture retains report SHA-256, report-carried replay mode/range identity, metric units and full selected-range policy. Its `ncu_range_time_ms` is a profiler measurement and does not represent benchmark latency or an individual kernel time. Timing, correctness checks and Nsight Systems remain separate requirements.
- MCP `run_benchmarks` emits `speedup_attribution` from captured profiler metrics when available, without requiring an LLM backend.
- Use `--only-cuda` / `only_cuda=true` to run only CUDA binary wrappers, or `--only-python` / `only_python=true` to skip them.

**Note:** `aisp benchmark ...` commands are diagnostic microbenchmarks (`hw_*` tools) and do not use the harness.

**Explicit execution audit:** from `code/`, run one fresh setup and callback outside
benchmark timing, then emit a JSON placement and destination-write receipt:

```bash
python -m core.harness.execution_audit ch05/optimized_vectorization.py \
  --expected-device cuda:0 --destination _output_buffer
```

Repeat `--destination ATTRIBUTE` for additional preallocated contiguous floating-point
or complex outputs. The audit poisons those exact tensors and rejects incomplete writes
or replaced destination identities. Placement checks cover PyTorch dispatcher-visible
operations on the current thread and require at least one operation touching the
expected device. No-op and host-only callbacks cannot pass a CUDA execution audit.
These checks do not inspect arbitrary extension internals,
other processes, or general uninitialized-memory provenance. An intentional host tensor
can be allowed for one exact operator with `--allow-host-tensor ATTRIBUTE=aten.operator.overload`.
This standalone audit produces correctness evidence, not performance measurements.

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

| Engine Domain | CLI Command Group | MCP Tool Prefix |
|---------------|-------------------|-----------------|
| `gpu` | `aisp gpu` | `gpu_*` |
| `system` | `aisp system` | `system_*` |
| `profile` | `aisp profile` | `profile_*` |
| `analyze` | `aisp analyze` | `analyze_*` |
| `optimize` | `aisp optimize` | `optimize`, `optimize_*`, `recommend` |
| `distributed` | `aisp distributed` | `distributed_*` |
| `inference` | `aisp inference` | `inference_*` |
| `benchmark` | `aisp bench` | `benchmark_*`, `run_benchmarks` |
| `ai` | `aisp ai` | `ask`, `explain`, `ai_*` |
| `export` | `aisp bench report/export` | `export_*` |

---

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

## Diagnostic tools

The network, fabric, transport, collective, serving and cross-layer tools
share the CLI registry in `core/tools/tools_commands.py`. See the
[diagnostic tool guide](diagnostic_tools.md) for runnable commands and artifact
contracts.

MCP exposes the same registry through `tools_diagnostics`:

```json
{"tool": "network-diagnose", "args": ["--help"], "timeout_seconds": 60}
```

`tool` is one of `network-diagnose`, `fabric-diagnose`, `transport-diagnose`,
`collective-diagnose`, `serving-compare`, `serving-prepare-runtime` or
`cross-layer-diagnose`. `args` is an
argument vector, defaulting to `--help`. The timeout defaults to 300 seconds and
must be between 1 and 86400 seconds. Results include the command, exit code,
stdout, stderr and timeout state. Paths are local to the MCP server. Unsupported
hardware and missing measurements are not successful benchmark results.

`serving-prepare-runtime` takes `--package-dir` and `--output-dir`. It copies the
supported SGLang package into a new directory and initializes its native P/D
failure counters. It rejects unknown collector source bytes and existing output
directories. It leaves the installed package unchanged and records both package
digests. See the [serving guide](../labs/serving_comparison/README.md) before use.

The CLI also provides `aisp tools multi-gpu-communication` for the native
[MPI, NCCL, and NVSHMEM Jacobi lab](../labs/multi_gpu_communication/README.md).
It accepts explicit backend, scheduling, registration, graph, and rank options.
This CLI lab is not in the MCP diagnostic-tool allowlist. Its direct native
timings are component diagnostics, while benchmark claims need the harness
verification and profiling gates.

`aisp tools nvshmem-device` runs the
[device NVSHMEM lab](../labs/nvshmem_device/README.md). Select scalar or persistent
Jacobi halos, a block signal exchange, a persistent signal exchange, or a
persistent atomic-counter exchange. The runner requires a current native build
receipt and verifies the complete mesh or received history with an independent
Python reference. It retains fresh JSON, logs, and float32 output. This CLI lab
does not add an MCP diagnostic-tool allowlist entry.

## Operator trace replay

`aisp tools trace-replay` captures complete operator input tensors, validates
retained bundles, and replays each event through the native benchmark pairs.
The initial operators are prefix scan, online softmax, and fused NVFP4
quantization. The [replay lab](../labs/trace_replay/README.md) documents the
capture plan, input provenance, and supported hardware.

```bash
python -m cli.aisp tools trace-replay -- capture --plan capture-plan.json --output-dir captured --device cpu
python -m cli.aisp tools trace-replay -- validate --bundle captured
python -m cli.aisp tools trace-replay -- replay --bundle captured --output-dir replay-results
```

CPU captures use the numerical references. Their manifests identify the capture
backend and distinguish generated recipes from observed application inputs.
GPU replay preserves full outputs and uses the existing correctness, clock,
runtime, and profiler checks. Each event receives its own result and failure
record. Event order is preserved, while request arrival pacing and stream overlap
remain outside this operator replay contract. These results do not establish
whole application latency, throughput, or output quality.

MCP exposes the same CLI through `tools_trace_replay`:

```json
{"args": ["validate", "--bundle", "captured"], "include_context": false}
```

`args` defaults to `--help` and is forwarded as an argument vector. Paths refer
to the MCP server filesystem. The tool adds no invocation deadline. Existing
benchmark policies still apply. Results include the command, exit code, stdout,
and stderr. When the launcher receives a keyboard interrupt, it forwards SIGINT
through the owned launch groups and waits for child cleanup without a deadline.
The MCP handler returns retained output with `cancelled: true` after that cleanup.
This does not add a cancellation endpoint for background MCP jobs.

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
