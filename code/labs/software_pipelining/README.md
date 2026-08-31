# Lab - Software Pipelining

## Summary
Adds an explicit schedule-design lab for GPU software pipelining: a small runnable benchmark pair for serialized versus staged tiling, plus a deterministic analyzer that makes dependency classes, loop-carried constraints, anti-dependencies, and phase structure visible.

## Problem
The repo already had good implementation-side coverage for `cuda::pipeline`, TMA staging, warp specialization, and FA4-adjacent overlap. What it did not have was a compact place to answer the higher-level question: why is one pipeline schedule legal while another one is not?

## Baseline Path
- serialized tile loop
- same math as the optimized path
- one tile must finish loading before the next tile begins

## Optimized Path
- two-stage producer/consumer tile loop
- block-scoped `cuda::pipeline` overlap for next-tile staging
- same math and shape, but explicit software pipelining

## Measured Delta
No current speedup is claimed. The earlier shared-host result compared kernels with different load participation and was invalidated when the pair was repaired to differ only in serialized versus two-stage scheduling. Re-run the controlled pair on a clean canonical host before adding `expectations_{hardware_key}.json` or publishing a measured delta.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/software_pipelining:tile_pipeline --profile deep_dive --single-gpu
```

Use the deep-dive path when you want Nsight evidence that the optimized path actually reduces idle time instead of just shifting work around.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/software_pipelining
python -m cli.aisp bench run --targets labs/software_pipelining:tile_pipeline --profile minimal
python labs/software_pipelining/schedule_visualizer.py --example gemm_mainloop --format ascii
python labs/software_pipelining/schedule_visualizer.py --example fa_like_inner_loop --format json
```

## Learning Goals
- Compare a serialized tiled loop against a legal two-stage software pipeline on the same workload.
- Make same-iteration, loop-carried, and anti-dependency edges explicit instead of implicit.
- Show where prologue, steady-state, and epilogue phases come from.
- Keep the schedule model deterministic and explainable without introducing a solver.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_tile_pipeline.py`, `optimized_tile_pipeline.py` | Harness-discoverable benchmark pair for the serialized and pipelined tile loops. |
| `software_pipelining_common.py`, `software_pipelining_extension.py`, `software_pipelining_kernels.cu` | Shared benchmark wrapper, local extension loader, and CUDA kernels. |
| `pipeline_graph.py`, `schedule_visualizer.py` | CPU-only legality checker plus ASCII/JSON schedule rendering for deterministic examples. |
| `tests/test_pipeline_graph.py` | Unit tests for legality checking, JSON/ASCII rendering, and benchmark contract basics. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the visualizer directly when you only want the schedule explanation surface.
```bash
python -m cli.aisp bench list-targets --chapter labs/software_pipelining
python -m cli.aisp bench run --targets labs/software_pipelining:tile_pipeline --profile minimal
python labs/software_pipelining/schedule_visualizer.py --example gemm_mainloop --format ascii
```
- Targets follow the `labs/software_pipelining:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/software_pipelining:tile_pipeline="--length 8388608 --repeat-fmas 16"` to sweep the problem size or arithmetic intensity.
- The analyzer is CPU-only and should run even on hosts that cannot execute the benchmark.
- The benchmark pair fails fast with `SKIPPED:` on unsupported CUDA hardware/toolchains instead of silently degrading to a different runtime path.

## Validation Checklist
- `python labs/software_pipelining/schedule_visualizer.py --example fa_like_inner_loop --format json` emits a stable JSON payload with nodes, edges, schedule slots, and validation status.
- `python labs/software_pipelining/schedule_visualizer.py --example gemm_mainloop --format ascii` includes prologue, steady-state, and epilogue sections in the text view.
- `python -m cli.aisp bench run --targets labs/software_pipelining:tile_pipeline --profile minimal` shows the serialized and pipelined paths on the same math workload.

## Notes
- This lab is intentionally smaller than `labs/flashattention4`: the FA-like schedule lives in the analyzer so the schedule story stays legible.
- Related implementation-heavy labs:
  - `labs/flashattention4` for FA4-style fused attention behavior and provider selection
  - `labs/blackwell_matmul` for staged GEMM, TMA, and cluster comparisons on Blackwell
