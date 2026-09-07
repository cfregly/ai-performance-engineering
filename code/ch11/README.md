# Chapter 11 - Streams & Concurrency

## Summary
Explains how to overlap compute, memory, and communication on Blackwell using CUDA streams, ordered sequences, Hyper-Q, warp-specialized pipelines, and adaptive scheduling. The README keeps the legacy target names tied to the actual copy+elementwise overlap workload and runtime-adaptive scheduling story instead of pretending every target is a generic multi-stream demo. CUDA Graph capture and graph-level orchestration continue in the manuscript and repo Chapter 12; this chapter’s runnable surface stays focused on stream/concurrency paths.

## Problem
Chapter 11 is where concurrency ideas have to prove they are reducing real idle time instead of just making traces look busier. The useful question is not "can we add streams?" but "which ordering and overlap changes actually improve the measured workload?"

## Baseline Path
- more serialized stream usage
- conservative ordering that protects correctness but leaves overlap untapped
- simpler to debug, but often too launch- and idle-heavy

## Optimized Path
- stream overlap where work is truly independent
- stream-ordered cache and KV update paths that preserve correctness without full serialization
- warp-specialized multistream execution where the hardware can support it

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `streams` | `15.035 ms` | `8.073 ms` | `1.86x` | basic overlap instead of more serialized launches |
| `stream_ordered_kv_cache` | `3.153 ms` | `2.103 ms` | `1.50x` | ordered KV updates with less idle time |
| `warp_specialization_multistream` | `17.530 ms` | `10.500 ms` | `1.67x` | multistream warp specialization path |

These are not "big baseline mistake" wins. They are the more realistic kind of concurrency gains where overlap helps, but only when the work graph actually allows it.

## Profiler Evidence
Use deep-dive runs when you want to confirm that the gain is real overlap rather than timing noise:

```bash
python -m cli.aisp bench run --targets ch11:streams --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch11:stream_ordered_kv_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch11:warp_specialization_multistream --profile deep_dive --single-gpu
```

The Nsight story should differ by workload:
- `streams`: less idle between independent launches
- `stream_ordered_kv_cache`: correctness-preserving ordering without full-device serialization
- `warp_specialization_multistream`: more useful overlap across specialized work partitions

## Repro Commands
```bash
python -m ch11.compare
python -m cli.aisp bench list-targets --chapter ch11
python -m cli.aisp bench run --targets ch11 --profile minimal
python -m cli.aisp bench run --targets ch11:streams --profile deep_dive --single-gpu
```

## Learning Goals
- Use multiple CUDA streams to overlap independent kernels without starving priority work.
- Control ordering constraints for KV-cache updates and stream-ordered memory pools.
- Benchmark warp-specialized multistream kernels that share data via DSMEM.
- Introduce adaptive policies that adjust stream usage based on runtime telemetry.
- Connect stream-level ideas to [FA4's intra-kernel producer/consumer pipelines](../labs/flashattention4/colfax_optimization_diaries.md), where per-slot phases and warp-local signals expose otherwise serialized work.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_streams.py`, `optimized_streams.py`, `streams_overlap_demo.cu`, `streams_ordered_demo.cu`, `streams_warp_specialized_demo.cu`, `stream_overlap_base.py` | Core stream overlap demos that contrast serialized launches with overlapped workloads. |
| `baseline_stream_ordered.py`, `baseline_stream_ordered_kv_cache.py`, `optimized_stream_ordered.py`, `optimized_stream_ordered_kv_cache.py` | Stream-ordered allocator and KV-cache examples ensuring deterministic updates while enabling overlap. |
| `baseline_gemm_streams.py`, `optimized_gemm_streams.py`, `baseline_tensor_cores_streams.py`, `optimized_tensor_cores_streams.py` | GEMM pipelines that schedule tensor-core kernels across multiple streams to decouple math vs IO phases. |
| `baseline_distributed_streams.py`, `optimized_distributed_streams.py`, `baseline_adaptive_streams.py`, `optimized_adaptive_streams.py` | Adaptive streaming controllers that balance NCCL, compute, and IO tasks on large systems. |
| `baseline_warp_specialization_multistream.*`, `optimized_warp_specialized_multistream.*`, `warp_specialized_cluster_pipeline_multistream.cu` | Warp-specialized multistream kernels demonstrating DSMEM usage and per-stream specialization. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json` | Harness driver plus expectation data for concurrency regressions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch11.compare
python -m cli.aisp bench list-targets --chapter ch11
python -m cli.aisp bench run --targets ch11 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_streams.py --trace` captures overlapping NVTX ranges in Nsight Systems, proving concurrency is active.
- `python optimized_stream_ordered_kv_cache.py --validate` matches the baseline's outputs while reducing idle gaps between cache updates.
- Warp-specialized multistream kernels flag unsupported hardware (missing DSMEM) immediately, preventing silent fallbacks.

## Notes
- The README calls out the legacy target names explicitly so book-facing labels still point at the actual copy+elementwise overlap workload and runtime-adaptive scheduling pairs.
- `warp_specialized_triton.py` provides a Triton analogue for the CUDA concurrency demos so you can compare compiler-generated schedules.
- `kv_prefetch_pipeline_enhanced_demo.cu` builds on the DSMEM kernels bundled in this directory so you can study the entire pipeline locally.
