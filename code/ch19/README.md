# Chapter 19 - Dynamic & Adaptive Inference Precision/Memory Systems

## Summary
Explores dynamic precision, KV-cache quantization, memory double buffering, and adaptive allocators so inference-oriented low-precision experiments stay numerically safe while squeezing every byte of HBM.

## Problem
Chapter 19 is where adaptive precision and memory-system ideas have to prove they are more than paper wins. The useful question is not "can we quantize or double-buffer this?" but "which runtime precision and memory changes improve the real workload enough to justify the added complexity?"

## Baseline Path
- higher-cost cache, precision, and memory-management paths
- simpler allocator and buffering behavior
- cleaner as a reference, but often too expensive in memory traffic or precision budget

## Optimized Path
- quantized caches, lower-precision training/inference paths, and explicit buffering improvements
- adaptive allocator or overlap logic where memory behavior is the actual bottleneck
- benchmarked through the same harness contract so the speedup claims remain comparable and verified

## Measured Delta
Representative validated results from current expectation baselines and recent strict reruns:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `dynamic_quantized_cache` | `1.710 ms` | `1.517 ms` | `1.13x` | adaptive-bitwidth quantized refresh over the same full-cache footprint, with CPU-side verification output so the optimized path no longer pays a large GPU memory penalty |
| `memory_double_buffering` | `5.536 ms` | `2.809 ms` | `1.97x` | double-buffered memory path |
| `mxfp8_moe` | `16.037 ms` | `2.080 ms` | `7.71x` | lower-precision MoE path with materially better execution behavior |

This chapter is where "low precision" should be read as a systems decision, not just a dtype choice. Some wins come from lower math cost, others from lower memory traffic or better overlap.

`dynamic_quantized_cache` now uses the fair steady-state full-footprint refresh model introduced on `2026-03-17`. Repeated strict reruns on this virtualized host now land around `1.13-1.15x`, and the optimized path's GPU peak memory dropped from the earlier `~765 MB` down to `~269 MB`, below the baseline's `~404 MB`.

## Profiler Evidence
Use deep-dive harness runs when you want to inspect whether the gain came from compute reduction, memory reduction, or allocator/buffering behavior:

```bash
python -m cli.aisp bench run --targets ch19:dynamic_quantized_cache --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch19:memory_double_buffering --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch19:mxfp8_moe --profile deep_dive --single-gpu
```

Those targets make good chapter probes because they cover cache behavior, memory overlap, and lower-precision MoE execution without collapsing everything into one synthetic headline.

## Repro Commands
```bash
python -m ch19.compare
python -m cli.aisp bench list-targets --chapter ch19
python -m cli.aisp bench run --targets ch19 --profile minimal
python -m cli.aisp bench run --targets ch19:dynamic_precision --profile minimal --single-gpu
python -m cli.aisp bench run --targets ch19:mxfp8_moe --profile deep_dive --single-gpu
python -m cli.aisp tools ch19-adaptive-parallelism
python -m cli.aisp tools ch19-dynamic-precision -- --help
python -m cli.aisp tools ch19-dynamic-quantized-cache -- --help
```

## Learning Goals
- Benchmark FP4/FP6/FP8 training loops with calibration and validation hooks.
- Overlap KV-cache prefetch with compute while respecting precision constraints.
- Implement dynamic quantized caches that switch formats mid-run without drift.
- Design allocator helpers to monitor and rebalance fragmented memory pools.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_training.py`, `optimized_nvfp4_training.py`, `native_fp4_quantization.py`, `native_fp6_quantization.py`, `native_fp8_training.py` | Training and quantization recipes that switch between FP8 and NVFP4 with automatic calibration. |
| `baseline_adaptive_parallelism.py`, `optimized_adaptive_parallelism.py`, `adaptive_parallelism_benchmark_common.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py` | Chapter-native adaptive parallelism benchmark pair plus the routing helpers that model tensor/pipeline/hybrid/data worker-pool selection on the same synthetic request stream. |
| `baseline_dynamic_precision.py`, `optimized_dynamic_precision.py`, `dynamic_precision_benchmark_common.py`, `dynamic_precision_switching.py`, `token_precision_switching.py` | Chapter-native dynamic precision benchmark pair plus the confidence-driven precision helpers that keep decode outputs stable while switching precision modes. |
| `baseline_memory_double_buffering.py`, `optimized_memory_double_buffering.py`, `memory_allocator_with_monitoring.py`, `dynamic_memory_allocator.py`, `_allocator_worker.py` | Memory-management helpers covering double buffering, instrumentation, and adaptive worker pools. |
| `baseline_kv_prefetch_overlap.cu`, `optimized_kv_prefetch_overlap.cu`, `kv_prefetch_overlap_sm121` binaries | CUDA kernels proving that quantized KV prefetch can overlap with compute when using cp.async pipelines. |
| `baseline_dynamic_quantized_cache.py`, `optimized_dynamic_quantized_cache.py`, `baseline_dynamic_quantized_cache_coalesced.py`, `optimized_dynamic_quantized_cache_coalesced.py`, `dynamic_quantized_cache.py`, `token_precision_switching.py`, `dynamic_precision_switching.py` | Cache-refresh experiments comparing full-precision FP32 maintenance, adaptive-bitwidth quantized refresh, and coalesced repeated-width refresh on the same KV footprint. |
| `baseline_fp4_hardware_kernel.cu`, `optimized_fp4_hardware_kernel.cu`, `fp8_hardware_kernel.cu`, `custom_allocator_retry.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py` | Hardware-level kernels and adaptive scheduling helpers for heterogeneous precision fleets. |
| `compare.py`, `arch_config.py`, `expectations_{hardware_key}.json` | Harness entry, architecture toggles, and stored expectation data. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch19.compare
python -m cli.aisp bench list-targets --chapter ch19
python -m cli.aisp bench run --targets ch19 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch19.compare` runs the chapter baseline/optimized sweep through the shared harness.
- `python -m cli.aisp bench run --targets ch19:adaptive_parallelism --profile minimal` keeps the baseline Python routing loop and the optimized vectorized routing path output-equivalent on the same request stream.
- `python -m cli.aisp bench run --targets ch19:dynamic_precision --profile minimal --single-gpu` keeps the fixed-precision and dynamic-precision decode outputs token-equivalent on the same prompt stream.
- `python -m cli.aisp bench run --targets ch19:dynamic_quantized_cache --profile minimal` validates the adaptive-bitwidth quantized refresh against the same full-cache FP32 baseline while tracking bounded error.
- `nvcc -o optimized_kv_prefetch_overlap_sm121 optimized_kv_prefetch_overlap.cu` plus the baseline binary show measurable overlap improvements in Nsight Compute.

## Notes
- `arch_config.py` exposes `ENABLE_NVFP4`/`ENABLE_TF32` toggles per device, making it easy to compare precision recipes.
- `validate_quantization_performance.py` aggregates accuracy vs throughput numbers into CSV form for proof-of-benefit reporting.
