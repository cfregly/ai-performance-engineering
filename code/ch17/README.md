# Chapter 17 - Disaggregated Prefill/Decode & Routing

## Summary
Centers the chapter on disaggregated prefill/decode serving, then layers routing and scheduling policies on top so Blackwell clusters can hand work between prefill/decode pools, MoE experts, and pipeline stages without sacrificing utilization.

## Problem
Chapter 17 is where routing and disaggregation ideas stop being whiteboard architecture and start paying rent. The useful question is not "can we route dynamically?" but "which router, queueing, and handoff changes actually improve TTFT, TPOT, or throughput once the full prefill/decode path is measured?"

## Baseline Path
- static or minimally adaptive routing
- conservative prefill/decode handoff with more blocking behavior
- easy to reason about, but expensive once queue imbalance and KV movement dominate

## Optimized Path
- topology-aware or telemetry-aware routing decisions
- disaggregated prefill/decode paths that reduce idle time and handoff overhead
- measured through the shared harness so routing wins are comparable to kernel and memory wins elsewhere in the repo

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `routing_static` | `5.680 ms` | `0.804 ms` | `7.07x` | smarter routing policy without changing the underlying workload |
| `moe_router_uniform` | `4.719 ms` | `0.931 ms` | `5.07x` | topology-aware expert routing instead of uniform placement |
| `prefill_decode_disagg_ttft` | `2678.148 ms` | `938.237 ms` | `2.85x` | disaggregated prefill/decode handoff optimized for TTFT |

This chapter mixes policy wins with orchestration wins. That is useful, but it means you should read each target as a specific system story rather than as one generic routing number.
Use the `prefill_decode_disagg*` targets as the chapter-native exemplars; `inference_full` remains a comparison pair for model-side work reduction rather than a disaggregated serving benchmark. Its structured metrics now expose `active_layers`, `identity_layers_skipped`, `story.comparison_pair=1`, and `story.chapter_native_exemplar=0`, while structured story metadata points to the `prefill_decode_disagg*` family as the chapter-native exemplar set.

## Profiler Evidence
Use deep-dive harness runs when you want evidence for where the gain came from instead of only the final runtime delta:

```bash
python -m cli.aisp bench run --targets ch17:routing_static --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch17:moe_router_uniform --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch17:prefill_decode_disagg_ttft --profile deep_dive --single-gpu
```

Those three targets answer different questions:
- `routing_static`: policy overhead versus routing quality
- `moe_router_uniform`: topology-aware MoE routing payoff
- `prefill_decode_disagg_ttft`: queueing and handoff behavior in a split prefill/decode system

## Repro Commands
```bash
python -m ch17.compare
python -m cli.aisp bench list-targets --chapter ch17
python -m cli.aisp bench run --targets ch17 --profile minimal
python -m cli.aisp bench run --targets ch17:prefill_decode_disagg_ttft --profile deep_dive --single-gpu
```

## Learning Goals
- Implement dynamic routers that react to TTFT, TPOT, and KV-locality metrics.
- Profile complete inference stacks (prefill + decode) under realistic synthetic loads.
- Blend pipeline parallelism with routing logic for long-context workloads.
- Document profiling steps (roofline, Nsight) specific to the routing lab.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py` | Routing controllers that evolve from static heuristics to telemetry-driven admission and rejection policies. |
| `baseline_inference_full.py`, `optimized_inference_full.py` | Control pair for full-depth inference versus early-exit pruning. Useful as an end-to-end inference sanity check, but not the chapter's primary disaggregated prefill/decode story. |
| `baseline_prefill_decode_disagg_overlap_multigpu.py`, `optimized_prefill_decode_disagg_overlap_multigpu.py`, `baseline_prefill_decode_disagg_batched_multigpu.py`, `optimized_prefill_decode_disagg_batched_multigpu.py`, `baseline_prefill_decode_disagg_ttft_multigpu.py`, `optimized_prefill_decode_disagg_ttft_multigpu.py`, `baseline_prefill_decode_disagg_tpot_long_multigpu.py`, `optimized_prefill_decode_disagg_tpot_long_multigpu.py` | Chapter-native end-to-end inference flows modeling separate prefill and decode pools, including overlap-focused, batched-handoff, TTFT-focused, and long-output TPOT-focused multi-GPU pairs. |
| `baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py` | Pipeline parallel workloads combining compute and KV-transfer scheduling. |
| `baseline_moe_router_uniform.py`, `optimized_moe_router_uniform_topology.py` | Comparable MoE router benchmark pair contrasting uniform vs topology-aware routing while keeping outputs invariant via shared expert weights. |
| `baseline_moe_router_local_capacity.py`, `optimized_moe_router_local_capacity.py` | MoE capacity-reservation pair contrasting topology-aware overflow spill with all-local expert placement. |
| `moe_router_uniform_demo.py`, `moe_router_topology_demo.py` | MoE routing demos (non-benchmark) contrasting uniform vs topology-aware expert selection. |
| `baseline_routing_static.py`, `optimized_routing_static.py` | Router variants for static/dynamic sharding decisions (comparable benchmarks). |
| `baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py` | Memory-bound case studies plus profiling guides tailored to routing workloads (use `aisp tools roofline` for roofline analysis). |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `dynamo_config.yaml` | Harness entry, build rules, expectation baselines, and Dynamo config knobs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch17.compare
python -m cli.aisp bench list-targets --chapter ch17
python -m cli.aisp bench run --targets ch17 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_dynamic_routing.py --trace` logs TTFT/TPOT trends that settle faster than the baseline's oscillations.
- `python optimized_pipeline_parallelism.py --profile minimal` shows overlapping prefill/decode segments with fewer idle bubbles.
- `python -m cli.aisp tools roofline` reproduces the documented roofline points using your latest captures.

## Notes
- `blackwell_profiling_guide.py` walks through Nsight Systems/Compute captures and interpreting roofline vs occupancy bottlenecks for routing-heavy workloads.
- `baseline_prefill_decode_disagg_overlap_multigpu.py` and `baseline_prefill_decode_disagg_batched_multigpu.py` run via torchrun and default to a 50/50 split when world size is even; override with `--prefill-ranks` (e.g., 2P1D). Use `torchrun --nproc_per_node` to choose the GPU count.
- The disaggregated prefill/decode baselines use per-request blocking handoff with per-request sync/barrier to model naive scheduling; optimized counterparts batch per group or send contiguous KV/seed slabs to overlap or boost throughput.
