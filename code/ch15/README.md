# Chapter 15 - Disaggregated Inference & KV Management

## Summary
Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns. The repo chapter captures the disaggregated serving and KV-management themes directly, while the NIXL-specific connector story from the manuscript is only represented indirectly.

## Problem
Chapter 15 is where inference-system ideas have to justify themselves with end-to-end measurements. The useful question is not "can we disaggregate or batch this?" but "which orchestration changes actually reduce latency or increase throughput once KV movement and scheduling overhead are included?"

## Baseline Path
- monolithic or minimally coordinated inference execution
- straightforward KV management and queue draining
- easy to reason about, but expensive once prefill/decode and cache movement start to dominate

## Optimized Path
- disaggregated prefill/decode and batched scheduling where they help
- NVLink-pooled KV-cache strategies and topology-aware routing
- still measured through the shared benchmark harness, so the chapter is a performance case study instead of a pile of demos

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `continuous_batching` | `52.955 ms` | `12.719 ms` | `4.16x` | queueing and batching strategy |
| `kv_cache_nvlink_pool` | `1047.860 ms` | `171.477 ms` | `6.11x` | pooled KV-cache path |
| `guided_decoding` | `12.702 ms` | `2.131 ms` | `5.96x` | guided decode path |
| `greedy_sampler` | `3.221 ms` | `0.907 ms` | `3.55x` | skips full-vocab probability materialization for greedy decode |
| `speculative_decoding` | `103.323 ms` | `26.761 ms` | `3.86x` | speculative decode orchestration |

The `greedy_sampler` row is from `artifacts/runs/codex_ch15_greedy_sampler_20260630_212500/`. It is a semantically narrow serving fast path where temperature-positive greedy decode uses logits argmax directly instead of building a probability tensor. The chapter mixes system-level wins from queueing/orchestration with fabric/cache-path wins. Those are both valuable, but they are not the same optimization story.

## Profiler Evidence
Use deep-dive harness runs when you want to attribute the gains to scheduling, cache movement, or decode behavior instead of only quoting the runtime delta:

```bash
python -m cli.aisp bench run --targets ch15:continuous_batching --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch15:kv_cache_nvlink_pool --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch15:greedy_sampler --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch15:speculative_decoding --profile deep_dive --single-gpu
```

Those runs are the right place to check whether the win came from less queue idle time, less cache movement, or fewer wasted decode steps.

## Repro Commands
```bash
python -m ch15.compare
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
python -m cli.aisp bench run --targets ch15:kv_cache_nvlink_pool --profile deep_dive --single-gpu
```

## Learning Goals
- Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.
- Design KV-cache managers that gracefully span local and remote HBM pools.
- Implement continuous batching and queueing so decode throughput stays high.
- Use decode-policy fast paths so greedy serving avoids unnecessary probability tensors.
- Serve MoE models efficiently by pairing routing with optimized communication.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_inference_monolithic.py`, `optimized_inference_monolithic.py` | Single-box inference loops that establish the baseline before disaggregation. |
| `disaggregated_inference_multigpu.py` | Disaggregated inference demo that layers speculative decoding on top of prefill/decode pools. |
| `baseline_single_gpu_kv_handoff.py`, `optimized_single_gpu_kv_handoff.py`, `baseline_disaggregated_inference_multigpu.py`, `optimized_disaggregated_inference_multigpu.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`, `baseline_prefill_decode_disagg_multigpu.py`, `optimized_prefill_decode_disagg_multigpu.py`, `disaggregated_inference_single_common.py` | Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling (multi-GPU), plus a supplementary single-GPU KV-handoff comparison pair. |
| `baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `kv_cache_management_math.py`, `baseline_kv_cache_nvlink_pool.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_nvlink_pool_multigpu.py`, `optimized_kv_cache_nvlink_pool_multigpu.py` | KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` | Single-GPU continuous batching scheduler for TTFT-aware queueing. |
| `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py` | Multi-GPU continuous batching scheduler for scaled queueing throughput. |
| `baseline_greedy_sampler.py`, `optimized_greedy_sampler.py`, `greedy_sampler_common.py` | Greedy decode sampler comparison showing that positive-temperature argmax can skip full-vocab softmax materialization. |
| `baseline_moe_inference.py`, `optimized_moe_inference.py` | Inference-specific MoE workloads that pair router load with communication control. |
| `baseline_moe_overlap.py`, `optimized_moe_overlap_shared_expert.py`, `baseline_moe_overlap_local_route.py`, `optimized_moe_overlap_local_route.py`, `moe_overlap_local_route_common.py`, `baseline_wide_ep.py`, `optimized_wide_ep.py`, `baseline_moe_dispatch.py`, `optimized_moe_dispatch.py`, `baseline_moe_routing_topology_aware.py`, `optimized_moe_routing_topology_aware.py` | MoE expert-parallel microbenchmarks that now split dispatch-path optimization, direct local routed placement, and topology-aware routing locality so attribution stays clean. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `Makefile` | Harness entry and dependencies for inference-focused validation. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch15.compare
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m cli.aisp bench run --targets ch15:disaggregated_inference_multigpu --profile minimal --ncu-replay-mode kernel` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity (kernel replay avoids NCU application-replay stalls on this workload).
- `python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.
- `python compare.py --examples continuous_batching` (single GPU) and `python compare.py --examples continuous_batching_multigpu` (multi-GPU) show optimized scheduling increases tokens/sec vs naive queue draining.
- `python -m cli.aisp bench run --targets ch15:greedy_sampler --profile minimal --single-gpu` verifies the direct-logit argmax path matches the full-probability baseline exactly.

## Notes
- `disaggregated_inference_multigpu.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.
- Use `torchrun --nproc_per_node <num_gpus>` to run the disaggregated pipeline on the desired GPU count (defaults to all visible GPUs, even count).
- `Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.
