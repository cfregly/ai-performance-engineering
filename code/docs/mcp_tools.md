# MCP Tools Reference

The `aisp` MCP server exposes AI Systems Performance tools via JSON-RPC over stdio.

## Quick Start

```bash
# Start the MCP server (stdio)
python -m mcp.mcp_server --serve

# List tools (authoritative)
python -m mcp.mcp_server --list
```

## Security / Authentication

The MCP server is **JSON-RPC over stdio** and is intended for **local use** (editor integration, local automation, CI). It does not implement authentication because stdio does not expose a network listener.

If you wrap MCP for network exposure, deploy it behind an authenticated transport (for example: SSH tunnel, a reverse proxy with auth, or mTLS) and treat tool execution as privileged.

## Common Workflow: Deep-Dive Baseline vs Optimized Compare

One-shot (recommended): `benchmark_deep_dive_compare`

```json
{
  "targets": ["ch10:atomic_reduction"],
  "output_dir": "artifacts/runs",
  "async": true
}
```

This runs `bench run` with `profile=\"deep_dive\"`, writes outputs under a timestamped run dir, and returns:
- `run_dir`, `results_json`, `analysis_json`
- per-benchmark `profiles_dir` + `followup_tool_calls` for `profile_compare` / `compare_nsys` / `compare_ncu`

## Common Workflow: Standard Benchmark Run (Minimal Profile)

Default: `run_benchmarks` uses `profile="minimal"` unless you explicitly request `deep_dive`.
It also runs post-benchmark triage and generates an HTML report unless you set
`auto_analyze=false` or `auto_report=false`.

```json
{
  "targets": ["ch10:atomic_reduction"]
}
```

### Benchmark validity profile (shared with CLI/dashboard)

- `validity_profile` (benchmark validity profile): `strict` (default; fail-fast with full validity checks) or `portable` (compatibility mode for virtualized/limited hosts)
- `allow_portable_expectations_update`: optional override to allow expectation-file updates in `portable`

Example portable run:

```json
{
  "targets": ["ch10:atomic_reduction"],
  "profile": "minimal",
  "validity_profile": "portable",
  "allow_portable_expectations_update": true
}
```

## Tool Names

Tool names are the exact names returned by `tools/list` / `--list` (for example: `gpu_info`, not `gpu info`).

## Response Format

All tools return a single MCP `text` content entry containing a JSON envelope with:
- `tool`, `status`, `timestamp`, `duration_ms`
- `arguments` + `arguments_details`
- `result` + `result_preview` + `result_metadata`
- `context_summary` + `guidance.next_steps`

## Profiling Metrics Payloads

Profiling tools return structured JSON metrics in their result payloads:
- `profile_nsys` includes `result.nsys_metrics` (parsed Nsight Systems metrics)
- `profile_ncu` includes `result.ncu_metrics` (parsed Nsight Compute metrics)
- `profile_torch` includes `result.torch_metrics` (summary metrics; `result.report` is an alias)
- `profile_hta` includes `result.nsys_metrics` alongside `analysis_summary`

Compare tools also return metric diffs when profiles are present:
- `compare_nsys` returns `result.nsys_comparison` and `result.ncu_comparison` when captured
- `compare_ncu` returns `result.ncu_comparison` and `result.nsys_comparison` when captured
- `profile_compare` always attaches both comparisons when captured

Use these metric payloads to explain speedups/regressions and identify bottleneck shifts.

## Async Jobs

Some tools can return an async job ticket (`job_id`) that you can poll via `job_status`. Job records are kept in-memory with bounded retention:

- `AISP_MCP_JOB_TTL_SECONDS` (default: `3600`)
- `AISP_MCP_JOB_MAX_ENTRIES` (default: `1000`)
- `AISP_MCP_JOB_CLEANUP_INTERVAL_SECONDS` (default: `30`)

## `tools_*` (Non-benchmark Utilities)

These tools are intentionally **not** comparative benchmarks; they run utilities via `aisp tools <name>`.

- `tools_kv_cache`
- `tools_cost_per_token`
- `tools_compare_precision`
- `tools_detect_cutlass`
- `tools_dump_hw`
- `tools_probe_hw`

Each accepts:
- `args`: list of strings forwarded to the underlying utility script
- `timeout_seconds`: max runtime before returning
- `include_context` / `context_level`

Example call shape:

```json
{
  "args": ["--layers", "32", "--hidden", "4096", "--tokens", "4096", "--dtype", "fp16"]
}
```

<!-- BEGIN MCP TOOL LIST -->

## Tool Catalog (generated)

Generated from `mcp.mcp_server.TOOLS`. Run `python -m scripts.generate_mcp_docs` to refresh.

### GPU (5)
- `gpu_bandwidth`: Run GPU memory bandwidth test measuring actual vs theoretical HBM bandwidth
- `gpu_info`: Get GPU hardware snapshot: name, architecture, VRAM (total/used/free), temperature, power draw, utilization %
- `gpu_power`: Get GPU power and thermal status: current power draw, power limit, temperature, throttling state
- `gpu_topology`: Get multi-GPU topology: NVLink/PCIe connections, NUMA affinity, P2P capability matrix
- `gpu_topology_matrix`: Get raw GPU/NUMA topology matrix directly from nvidia-smi topo -m

### System (11)
- `clock_lock_check`: Validate GPU clock locking works via the benchmark harness (required for canonical benchmark/profiling runs)
- `system_capabilities`: Get hardware capabilities summary: compute capability, tensor cores, supported precisions
- `system_container`: Inspect container/cgroup limits (CPU quota, memory limit, cgroup version)
- `system_context`: Get comprehensive system context: GPU info + software stack + hardware capabilities combined
- `system_cpu_memory`: Analyze CPU/memory hierarchy (NUMA nodes, cache sizes, memory stats)
- `system_dependencies`: Check health of ML/AI dependencies: torch, triton, flash-attn, transformers, vllm, etc
- `system_env`: Snapshot key environment variables and working directory
- `system_full`: Full system analysis: CPU/memory hierarchy, kernel params, container limits, tuning recommendations
- `system_network`: Inspect network interfaces, InfiniBand status, and GPUDirect/NCCL env hints
- `system_parameters`: Inspect kernel parameters that commonly affect performance (swappiness, dirty ratios, NUMA balancing, net buffers)
- `system_software`: Get software stack versions: PyTorch, CUDA toolkit, cuDNN, Python, NVIDIA driver

### Profiling (15)
- `compare_ncu`: Compare baseline vs optimized Nsight Compute kernel metrics
- `compare_nsys`: Compare baseline vs optimized Nsight Systems reports
- `ncu_summary`: Summarize an existing Nsight Compute report, returning the top-N kernels with key utilization metrics
- `nsys_summary`: Quick Nsight Systems summary stats without full profile capture
- `profile_compare`: Generate visual flame graph comparison showing WHY optimized code is faster
- `profile_compile_analysis`: Summarize torch.compile behavior from the latest benchmark results (graphs, breaks, time in compile)
- `profile_flame`: Get flame graph data showing execution time breakdown by function/operation
- `profile_hta`: Run Nsight Systems capture with HTA (Holistic Trace Analysis) for automated bottleneck detection
- `profile_kernels`: Get CUDA kernel execution breakdown: time per kernel, launch counts, occupancy hints
- `profile_list_profiles`: List available baseline/optimized profile pairs discovered under artifacts
- `profile_memory`: Get memory allocation timeline: VRAM usage over time, allocation spikes, potential leaks
- `profile_ncu`: Run Nsight Compute profiling to capture detailed per-kernel metrics (occupancy, throughput, etc.)
- `profile_nsys`: Run Nsight Systems profiling to capture GPU timeline, CUDA API calls, kernel launches
- `profile_roofline`: Get roofline model analysis: compute vs memory bound positioning, arithmetic intensity, efficiency
- `profile_torch`: Run PyTorch torch.profiler to capture CPU/GPU activity with Chrome trace output

### Analyze (10)
- `analyze_bottlenecks`: Identify performance bottlenecks: memory-bound, compute-bound, communication-bound, host-bound
- `analyze_comm_overlap`: Communication/compute overlap analysis for distributed training
- `analyze_dataloader`: DataLoader bottleneck analysis: worker efficiency, prefetch, throughput
- `analyze_energy`: Energy efficiency analysis: power consumption, efficiency metrics, green recommendations
- `analyze_memory_patterns`: Memory access pattern analysis: warp divergence, bank conflicts, memory coalescing
- `analyze_pareto`: Find Pareto-optimal configurations: best throughput/latency/memory tradeoffs
- `analyze_scaling`: Analyze how performance scales with workload size, sequence length, batch size, or GPU count
- `analyze_stacking`: Analyze which optimization techniques work well together and which conflict
- `analyze_whatif`: What-if analysis: Find optimizations that meet your constraints (VRAM, latency, throughput)
- `predict_scaling`: Predict performance scaling to more GPUs/larger batches

### Optimize (4)
- `optimize`: Resolve a benchmark file path or target and run quick LLM variants by default
- `optimize_roi`: Calculate ROI (return on investment) for optimization techniques: expected gain vs implementation effort
- `optimize_techniques`: Get catalog of all optimization techniques with details, requirements, and expected benefits
- `recommend`: Get prioritized optimization recommendations for your model configuration and goal

### Distributed (3)
- `distributed_nccl`: Get NCCL tuning recommendations: environment variables, IB settings, collective algorithms
- `distributed_plan`: Plan parallelism strategy: recommend DP/TP/PP/FSDP layout for model size and GPU count
- `launch_plan`: Generate launch commands for distributed training (torchrun, srun, etc.)

### Inference (4)
- `inference_deploy`: Generate inference deployment configuration (explicit model size required)
- `inference_estimate`: Estimate inference throughput/latency based on model + hardware (explicit model size required)
- `inference_quantization`: Get quantization recommendations: precision format, method, expected accuracy/speedup tradeoffs
- `inference_vllm`: Generate optimized vLLM configuration for inference serving (explicit model size required)

### Benchmark (21)
- `benchmark_compare`: Compare two benchmark run JSON files (baseline vs candidate)
- `benchmark_compare_runs`: Compare two benchmark runs showing speedup deltas, regressions, and improvements
- `benchmark_contracts`: Return the repo benchmark methodology, warehouse, and BenchmarkRun contract surfaces
- `benchmark_data`: Fetch benchmark results with filtering/sorting/pagination (dashboard data view)
- `benchmark_deep_dive_compare`: ONE-SHOT deep-dive workflow: run benchmarks with profile='deep_dive' AND return structured diffs from Nsight Systems + Nsight Compute (+ any available profiler artifacts)
- `benchmark_e2e_status`: Inspect an end-to-end e2e run package and return one normalized status snapshot
- `benchmark_e2e_sweep`: Run the tier1 release-confidence suite, optional discovered full sweep, cluster common-eval, and optional fabric eval in one orchestrated flow
- `benchmark_e2e_watch`: Launch the detached e2e watcher that auto-resumes stale or aborted runs using the stored contract
- `benchmark_explore`: Copy a baseline_*.py (or baseline_*.cu, auto-generates wrapper if missing), run minimal profiling with LLM patch variants, compare resource utilization across variants, and optionally run deep_dive profiling when minimal results are inconclusive
- `benchmark_export`: Export benchmark results to CSV/Markdown/JSON format for further analysis
- `benchmark_history`: List historical benchmark runs with summary stats
- `benchmark_llm_patch_loop`: Run the full LLM patch loop: deep-dive profile baseline/optimized, force LLM analysis, apply patches, rebenchmark, generate explanation, promote best patch, then run a clean baseline-vs-patch deep-dive compare and summarize nsys/ncu deltas
- `benchmark_opportunities`: Rank benchmark evidence by next optimization opportunity
- `benchmark_overview`: Summarize the latest benchmark results (status counts, top speedups, per-chapter stats)
- `benchmark_report`: Generate PDF/HTML report from benchmark results for sharing and documentation
- `benchmark_targets`: List benchmark targets in chapter:example format (e.g., 'ch07:flash_attention')
- `benchmark_trends`: Compute performance trends over time (avg/max speedup by run)
- `benchmark_triage`: 🔍 POST-BENCHMARK ANALYSIS: Analyze benchmark results and get actionable recommendations
- `benchmark_variants`: Shortcut to profile and generate optimized variants via LLM: runs benchmarks with profile='minimal', forces LLM analysis, applies patches, and rebenchmarks patched variants by default
- `list_chapters`: List all discoverable chapters and labs from the book curriculum
- `run_benchmarks`: Run benchmarks via the bench CLI with optional profiling and LLM analysis

### AI (5)
- `ai_status`: Check AI/LLM backend availability: connectivity, API key status, model availability
- `ai_troubleshoot`: Diagnose common training/distributed errors and suggest fixes
- `ask`: Ask a free-form performance question and get an answer with book citations
- `explain`: Explain a GPU/AI performance concept with clear definition and book citations
- `suggest_tools`: 🧭 TOOL NAVIGATOR: Get ranked tool suggestions based on your intent or problem

### Export (3)
- `export_csv`: Export benchmarks to CSV format for spreadsheet analysis or sharing
- `export_html`: Export benchmarks to interactive HTML report with charts and tables
- `export_pdf`: Export benchmarks to PDF report format for printing or formal sharing

### Hardware (10)
- `hw_cache`: Run GPU memory hierarchy test measuring bandwidth at specific stride pattern
- `hw_disk`: Run disk I/O benchmark measuring sequential read/write throughput
- `hw_ib`: Get InfiniBand bandwidth test instructions and check if ib_write_bw is available
- `hw_nccl`: Get NCCL collective bandwidth test command and check if nccl-tests is available
- `hw_network`: Run network throughput tests to check NIC and interconnect performance
- `hw_p2p`: Run GPU-to-GPU P2P bandwidth test measuring NVLink or PCIe peer access speed
- `hw_pcie`: Run PCIe bandwidth benchmark measuring Host-to-Device and Device-to-Host transfer speeds
- `hw_roofline`: Run stride sweep to measure memory bandwidth at different access patterns (roofline data)
- `hw_speed`: Run quick GPU speed tests: GEMM throughput, memory bandwidth, attention kernel
- `hw_tc`: Run Tensor Core throughput test measuring matmul performance at different precisions

### HuggingFace (1)
- `hf`: HuggingFace Hub operations: search models, get trending, download models

### Cluster/Cost (10)
- `cluster_build_canonical_package`: Materialize a clean canonical cluster package from one primary run plus optional comparison and historical runs
- `cluster_common_eval`: Run a preset benchmark bundle that answers the common system-evaluation questions people usually ask
- `cluster_eval_suite`: Run the cluster field-report eval suite, or a fast local smoke run that captures metadata + manifest
- `cluster_fabric_eval`: Run the canonical AI fabric evaluation bundle with a structured capability matrix, verification ledger, AI-workload correlation, and fabric scorecard
- `cluster_nmx_partition_lab`: Build a lab-only NVLink/NMX partition workflow guide from live inventory
- `cluster_promote_run`: Promote one run-local cluster result tree into the published cluster package
- `cluster_slurm`: Generate SLURM job script for cluster submission with optimal settings
- `cluster_validate_field_report`: Validate `cluster/field-report.md` and companion notes/template/runbook plus artifact hygiene
- `cluster_watch_promote`: Attach a detached watcher process that waits for a PID to exit, validates run artifacts, then promotes the run
- `cost_estimate`: Cloud cost estimation for GPU fleets

### Tools (6)
- `tools_compare_precision`: Run the precision/accuracy comparison tool (non-benchmark utility)
- `tools_cost_per_token`: Run the cost-per-token calculator (non-benchmark utility)
- `tools_detect_cutlass`: Run CUTLASS environment detection (non-benchmark utility)
- `tools_dump_hw`: Dump comprehensive hardware capability report (non-benchmark utility)
- `tools_kv_cache`: Run the KV-cache size calculator (non-benchmark utility)
- `tools_probe_hw`: Probe GPU capabilities dynamically and cache results (non-benchmark utility)

### Utility (5)
- `context_full`: Get full comprehensive context: complete system state
- `context_summary`: Get quick context summary: GPU + software snapshot
- `job_status`: Check status of a background job started with async=true
- `status`: 🚀 QUICK STATUS CHECK: Fast snapshot of GPU, software, and AI backend health
- `triage`: 🎯 START HERE: Quick triage = status check + context summary in one call

### Other (2)
- `info_features`: GPU feature detection: TMA, thread block clusters, async copy, etc
- `render_benchmark_run`: Render BenchmarkRun YAML through the shared backend template renderer

<!-- END MCP TOOL LIST -->
