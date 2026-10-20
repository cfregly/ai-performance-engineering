# Chapter 7 - Memory Access Patterns

## Summary
Teaches how memory layout drives performance: coalesced copies, tiled matmuls, async prefetch, TMA transfers, and table layout changes for lookup-heavy workloads.

## Problem
Chapter 7 is where memory layout turns from a CUDA lecture into a measurable cost model. The useful question is not "is coalescing good?" but "which access-pattern changes actually move the runtime enough to justify changing the kernel or data layout?"

## Baseline Path
- scalar or poorly staged memory movement
- little reuse of shared memory or async transfer mechanisms
- straightforward for correctness, but wasteful once bandwidth dominates

## Optimized Path
- coalesced/vectorized copy paths
- shared-memory tiling and TMA-backed staging where it helps
- measured through the shared harness so the memory-layout wins are directly comparable to other chapter benchmarks

## Historical Delta (not requalified)
Stored results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`. The sources for every target below changed after that run, so these rows are history, not current evidence. Rerun them on a B200 before quoting them.

| Target | Baseline | Optimized | Recorded delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `tma_bulk_tensor_2d` | `0.029 ms` | `0.008 ms` | `3.44x` | real tensor-map TMA bulk copy instead of manual 2D staging |
| `lookup` | `0.397 ms` | pending B200 rerun | pending | pretransposed path table, so each output reads its 64 values with coalesced loads |
| `matmul` | `1.165 ms` | `0.367 ms` | `3.18x` | shared-memory tiled matmul instead of the naive layout |

The stored `lookup` optimized time of `0.009 ms` (`45.41x`) is withdrawn. It timed an earlier kernel that read one precomputed sum per output. Commit `b39acf103` replaced that kernel with the current `optimized_lookup.cu`, which reads all 64 path values per output. One launch now moves 260 MiB: 256 MiB of path reads and a 4 MiB output write. The path table is far larger than L2, so at the B200's 8 TB/s HBM peak a launch takes at least `0.034 ms`. The old time cannot describe the current kernel.

Other source changes after that run include the following. `baseline_tma_bulk_tensor_2d.cu` now uses the optimized kernel's tile shape. `optimized_tma_bulk_tensor_2d.cu` fixed its TMA coordinate order and bounds its tail stores. `baseline_matmul.cu` changed its inner loop.

`tma_copy` now means a strict tensor-map/TMA-capable run only, and unsupported hosts fail fast with `SKIPPED:` instead of publishing an async-pipeline fallback under the TMA name.

## Profiler Evidence
Use deep-dive harness runs when you want to see whether the win came from less memory traffic, better staging, or fewer expensive accesses:

```bash
python -m cli.aisp bench run --targets ch07:tma_bulk_tensor_2d --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch07:lookup --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch07:matmul --profile deep_dive --single-gpu
```

These targets answer different chapter-level questions:
- `tma_bulk_tensor_2d`: descriptor-backed TMA vs manual 2D staging
- `tma_copy`: scalar global loads vs descriptor-backed 2D TMA staging for the same 64x64 tile-local transform on a 4096x4096 matrix. Audit W2-011 rewrote both binaries so they time this one transform. The rewrite has not yet been built or run on a B200, so the pair has no measured delta.
- `lookup`: cache/locality sensitivity
- `matmul`: memory-layout and tile-reuse payoff

## Repro Commands
```bash
python -m ch07.compare
python -m cli.aisp bench list-targets --chapter ch07
python -m cli.aisp bench run --targets ch07 --profile minimal
python -m cli.aisp bench run --targets ch07:tma_bulk_tensor_2d --profile deep_dive --single-gpu
```

## Learning Goals
- Measure the gap between scalar, coalesced, and vectorized memory moves.
- Use shared-memory tiling, async copy, and tensor maps where they actually help.
- Analyze lookup-heavy workloads and mitigate cache-thrashing access patterns.
- Quantify transpose and gather/scatter penalties to justify layout changes.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_copy_scalar.cu`, `baseline_copy_uncoalesced.cu`, `baseline_copy_uncoalesced.py`, `optimized_copy_uncoalesced_coalesced.cu`, `optimized_copy_scalar_vectorized.cu`, `optimized_copy_scalar_vectorized_sm121` | Copy kernels highlighting coalescing, vector width, and warp-level efficiency. |
| `baseline_hbm_copy.cu`, `baseline_hbm_peak.cu`, `optimized_hbm_copy.cu`, `optimized_hbm_peak.cu`, `baseline_hbm_copy.py`, `optimized_hbm_copy.py` | HBM peak-bandwidth probes with CUDA and Python harnesses. |
| `baseline_async_prefetch.cu`, `optimized_async_prefetch.cu`, `baseline_tma_copy.cu`, `optimized_tma_copy.cu`, `baseline_tma_copy.py`, `optimized_tma_copy.py`, `async_prefetch_2d_demo.cu`, `baseline_tma_bulk_tensor_2d.{py,cu}`, `optimized_tma_bulk_tensor_2d.{py,cu}` | Async copy demos plus the separate descriptor-backed TMA benchmark used for the chapter's canonical tensor-map evidence. |
| `baseline_matmul.cu`, `baseline_matmul.py`, `optimized_matmul_tiled.py`, `optimized_matmul_tiled.cu` | Matmul implementations to contrast naive global-memory access with shared-memory tiling and warp-level reuse. |
| `baseline_lookup.cu`, `baseline_lookup.py`, `optimized_lookup.cu`, `lookup_pytorch.py` | Cache-sensitive lookup workloads demonstrating how to reorganize tables for better locality. |
| `baseline_transpose.cu`, `baseline_transpose.py`, `optimized_copy_scalar_vectorized.cu`, `optimized_transpose_padded.py` | Transpose and gather/scatter experiments that show how to minimize bank conflicts. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json`, `memory_access_pytorch.py` | Harness entry, build recipes, expectation thresholds, and PyTorch validation scripts. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch07.compare
python -m cli.aisp bench list-targets --chapter ch07
python -m cli.aisp bench run --targets ch07 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m cli.aisp bench run --targets ch07:hbm_copy --profile minimal` reports the baseline/optimized bandwidth gap, proving vectorization plus async copies work.
- `python -m ch07.compare` runs the full baseline/optimized chapter sweep through the shared harness.
- Nsight Compute captures of `optimized_matmul_tiled.cu` hit >80% shared-memory bandwidth utilization with minimal bank conflicts.

## Notes
- Toggle `TORCH_COMPILE_MODE` when using the Python matmul wrappers to verify fusion benefits alongside the raw CUDA kernels.
- HBM tooling reads real peak numbers from `benchmark_peak_results_*.json` when present, providing realistic reference ceilings.
