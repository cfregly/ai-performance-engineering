# Lab - Persistent Decode & TMA Prefill

## Summary
Demonstrates Blackwell-friendly persistent decode kernels and TMA-powered prefill paths, with Python harnesses and CUDA/Triton implementations. Revised synchronization and full-output checks still require target GPU validation.

## Verification status after the audit
All four prefill/decode wrappers now capture every decode element and the complete prefill destination, with the prefill source included in the input signature. Both peers perform the same copy-only prefill workload. Independent decode checks and exact prefill-copy checks run outside timing; graph replay refreshes its inputs and side streams wait for caller input writes. CPU payload controls pass, but actual CUDA/TMA/graph runs, sanitizer checks and numerical-budget calibration remain pending. The inherited decode tolerance is not a newly calibrated accuracy policy.

## Problem
Decode and prefill paths often die by launch overhead, staging overhead, or both. This lab exists to show which of those costs persistent kernels, CUDA Graphs, and TMA actually remove on the same logical workload.

## Baseline Path
- naive decode loops and non-persistent prefill paths
- higher launch overhead
- less efficient staging into shared memory

## Optimized Path
- persistent decode kernels
- CUDA Graph replay where it helps
- TMA-powered prefill variants for lower staging cost

## Historical Delta (target revalidation pending)
Historical results preserved from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

| Target | Baseline | Optimized | Measured delta | Best optimization |
| --- | ---: | ---: | ---: | --- |
| `persistent_decode` | `1.411 ms` | `0.118 ms` | `11.94x` | `graphs` |
| `tma_prefill_decode` | `1.588 ms` | `0.931 ms` | `1.71x` | `optimized_tma_prefill_decode` |

These stored timings predate the complete-output checks and matching copy-only prefill workload. They do not establish a speedup for the repaired revision. Rerun correctness, numerical-budget and timing gates on the exact target before making a new performance claim.

The direct transport swaps stay visible as a transport-comparison benchmark on `nvlink_offload` and as `paged_kv_offload` as a real speed benchmark with a small local contract. The canonical KV-offload overlap claim stays on `paged_kv_offload_prefetch`, where async prefetch materially changes the overlap story instead of only swapping host-transport mechanics.

## Profiler Evidence
Use deep-dive runs when you want to see launch count and staging behavior instead of only the wall-clock delta:

```bash
python -m cli.aisp bench run --targets labs/persistent_decode:persistent_decode --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/persistent_decode:tma_prefill_decode --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/persistent_decode
python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal
python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50
```

## Learning Goals
- Contrast naive decode loops against persistent kernels that pin CTAs per sequence.
- Distinguish descriptor-based bulk TMA transfers from thread-scope cp.async/ordinary-copy scheduling.
- Benchmark CUDA vs Triton implementations with unified validation utilities.
- Mix CUDA Graphs into the decode path to remove residual launch overhead.
- Compare pinned direct H2D staging against async prefetch overlap for paged KV offload.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_persistent_decode.py`, `optimized_persistent_decode_cuda.py`, `optimized_persistent_decode_graphs.py`, `optimized_persistent_decode_triton.py` | Persistent decode variants spanning CUDA, graphs, and Triton. |
| `baseline_tma_prefill_decode.py`, `optimized_tma_prefill_decode.py`, `baseline_native_tma_prefill_decode.py`, `optimized_native_tma_prefill_decode.py` | Two distinct paths: `optimized_tma_prefill_decode.py` uses descriptor-based bulk tensor copies; the legacy `*_native_tma_*` pair uses 4-byte thread-scope CUDA pipeline copies, not TMA. |
| `baseline_paged_kv_offload.py`, `optimized_paged_kv_offload.py`, `baseline_paged_kv_offload_prefetch.py`, `optimized_paged_kv_offload_prefetch.py` | KV offload comparisons (pinned direct H2D with memmap, plus async prefetch on pinned host cache). |
| `core/scripts/kv_locality_microbench.py` | Pinned/pageable/NUMA host slab copy microbench (HBM vs local/remote pinned vs pageable). |
| `persistent_decode_common.py`, `tma_extension.py`, `expectations_{hardware_key}.json` | Shared helpers, CUDA extension wrappers, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/persistent_decode
python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal
```
- Targets follow the `labs/persistent_decode:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/persistent_decode:<workload>="--flag value"` to sweep schedule knobs.
- Run only on an allocated, authorized target after its correctness gates pass. Canonical GPU measurements require bare metal; an explicitly approved virtualized run must preserve clock/provenance evidence and remain noncanonical.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- Check every sequence and token, including `--tier large --num-programs 1` and `8`; grid-stride scheduling must not drop the final sequences.
- Run CUDA racecheck/synccheck for the shared reduction and repeated eager-versus-reference comparisons.
- Run `python -m pytest tests/test_audit_wave1_prefill_full_output.py -q` for complete payload controls and seven capability-gated actual CUDA cases. The CUDA cases change all inputs on a caller stream before each launch and cover full and piecewise graphs; CPU passes do not qualify GPU execution.
- For pinned staging, verify host-buffer reuse waits for DMA completion and device-buffer reuse waits for prior attention consumers; cover memmap, direct pinned views, both streams and host-worker modes.
- `python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal` compares all persistent/TMA variants in one sweep.
- `python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50` is a candidate measurement command; establish correctness and compare valid target timings before claiming lower launch overhead.
- The legacy `native_tma_prefill_decode` target measures thread-scope CUDA copy scheduling. Inspect generated SASS before claiming cp.async; it is never a native-TMA measurement.
- `python core/scripts/kv_locality_microbench.py` surfaces H2D copy time deltas for pageable vs pinned slabs; add `QUICK=1` for a short run.

## Notes
- `tma_extension.py` exposes `load_async_copy()`; `load_native_tma()` and the old target filenames remain compatibility aliases. No TMA performance or capability claim follows from those names.
- Historical timings and expectation files predate these correctness fixes; requalify them on the target.
- Set `TORCH_COMPILE_MODE` or `TMA_TILE_SIZE` via env vars before invoking the harness to sweep tile sizes.
- `tma_extension.py` caches builds under `~/.cache/torch_extensions`; clean the cache when switching CUDA versions.
- `nvlink_offload` remains a transport-comparison benchmark and `paged_kv_offload` stays a real speed benchmark with a small local contract; use `paged_kv_offload_prefetch` when you want the lab's canonical KV-offload overlap benchmark.
