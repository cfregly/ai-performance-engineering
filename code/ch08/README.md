# Chapter 8 - Occupancy, Warp Efficiency & ILP

## Summary
Concentrates on the Chapter 8 core loop from the book: tune occupancy, reduce warp divergence, and expose instruction-level parallelism until the profiler shows fewer stalls and more useful issue bandwidth.

## Problem
Chapter 8 is where profiler symptoms need to map cleanly to fixes. The useful question is not "is occupancy important?" but "which changes reduce execution-dependency stalls, improve warp efficiency, and keep enough resident work on the SM without blowing up register pressure?"

## Baseline Path
- conservative launch geometry or branch-heavy kernels
- more dependency chains per thread and lower warp execution efficiency
- easier to reason about, but often leaves the SM underfilled or the warp schedulers idle

## Optimized Path
- occupancy-aware launch and block-shape tuning
- predication and loop-unrolling changes that expose more useful work per warp
- measured through the same harness contract as the rest of the repo, so the gains are not one-off microbench stories

## Measured Delta
Representative validated results from `ch08/expectations_b200.json`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `threshold` | `2.324 ms` | `0.228 ms` | `10.19x` | predication removes the branch-heavy slow path and raises warp efficiency |
| `loop_unrolling` | `1.591 ms` | `0.382 ms` | `4.17x` | more independent work per thread reduces execution-dependency stalls |
| `ai_optimization` | `0.646 ms` | `0.241 ms` | `2.68x` | occupancy-aware scheduling keeps more useful work resident |

These are the chapter-native exemplars. The repo also keeps a few real bridge comparison pairs here, such as `thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp`, but those are explicitly marked in structured metrics as comparison pairs so dashboards do not blur them with the book's core Chapter 8 story. `tcgen05_custom_vs_cublas` remains an explicit custom-versus-library comparison target, and it now stays runnable as a supplementary comparison benchmark with a local contract rather than being skipped as informational.

## Profiler Evidence
Use deep-dive harness runs when you want to see whether the improvement came from better warp efficiency, more ILP, or a better occupancy/resource balance:

```bash
python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch08:loop_unrolling --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch08:ai_optimization --profile deep_dive --single-gpu
```

Those targets give you three useful slices:
- `threshold`: branch elimination and warp execution efficiency
- `loop_unrolling`: per-thread ILP and execution-dependency stalls
- `ai_optimization`: occupancy/resource tradeoffs in a more compute-heavy kernel

## Repro Commands
```bash
python -m ch08.compare
python -m cli.aisp bench list-targets --chapter ch08
python -m cli.aisp bench run --targets ch08 --profile minimal
python -m cli.aisp bench run --targets ch08:threshold --profile deep_dive --single-gpu
```

## Learning Goals
- Tune occupancy explicitly and observe how register counts limit resident CTAs.
- Minimize warp divergence with predication and uniform control flow.
- Use loop unrolling and instruction scheduling to increase throughput per thread.
- Reprofile after each change so occupancy, warp efficiency, and ILP improvements are visible in the same harness.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_threshold.py`, `optimized_threshold.py`, `threshold_kernels.cu`, `threshold_benchmark_base.py` | Chapter-native warp-divergence pair: branchy thresholding versus predicated thresholding on the same workload shape. |
| `baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `occupancy_tuning_tool.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu` | Occupancy studies that tune CTA shapes, register caps, and API-computed limits (plus a sweep tool for quick preset exploration). |
| `baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu` | Loop-unrolling case studies that expose more independent work per thread while tracking register pressure. |
| `baseline_ai_optimization.py`, `optimized_ai_optimization.py`, `ai_optimization_kernels.cu`, `independent_ops.cu` | AI-kernel scheduling samples that stage independent ops to highlight occupancy and issue-efficiency tradeoffs. |
| `baseline_thresholdtma.py`, `optimized_thresholdtma.py`, `threshold_tma_benchmark_base.py` | Legacy-named bridge comparison pair on the same threshold workload: the optimized path uses Blackwell-gated, per-thread async-copy staging in shared memory, not descriptor-based bulk TMA, and remains marked as a comparison pair in structured metrics. |
| `baseline_tiling.py`, `optimized_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu`, `tiling_extension_tcgen05.py` | Bridge comparison pairs into Chapter 9: arithmetic-intensity and tensor-core tiling workloads kept as real baseline/optimized pairs but marked non-native for Chapter 8. `optimized_tiling.py` uses the strict `matmul_tiled_fast` path so runtime issues fail fast instead of silently falling back. |
| `baseline_tcgen05_custom_vs_cublas.py`, `optimized_tcgen05_custom_vs_cublas.py`, `tcgen05_custom_vs_cublas_benchmark_base.py` | Supplementary custom-tcgen05-versus-cuBLAS bridge comparison benchmark that points ahead to Chapter 9 tensor-core scheduling without acting as a canonical Chapter 8 speed claim. |
| `baseline_nvfp4_mlp.py`, `optimized_nvfp4_mlp.py` | Precision bridge comparison pair: BF16 versus NVFP4 MLP path kept here as a real pair, but explicitly marked as a Chapter 9-style comparison. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entry, dependencies, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch08.compare
python -m cli.aisp bench list-targets --chapter ch08
python -m cli.aisp bench run --targets ch08 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- Nsight Compute traces for `optimized_threshold.py` should show higher warp execution efficiency than `baseline_threshold.py`.
- `python -m cli.aisp tools occupancy-tuning` prints preset timings + speedups for the occupancy tuning microbenchmark.
- `python -m cli.aisp bench run --targets ch08:thresholdtma --profile minimal` exercises the Blackwell-only bridge comparison on the same threshold shape used by the chapter-native threshold pair.

## Notes
- `arch_config.py` exposes toggles for enabling/disabling tcgen05 lowering per GPU so the same scripts work on SM100 and SM121.
- `threshold`, `loop_unrolling`, and `ai_optimization` are the chapter-native exemplars. `thresholdtma`, `tiling`, `tiling_tcgen05`, and `nvfp4_mlp` remain real baseline/optimized bridge comparisons and expose `story.comparison_pair=1` plus `story.chapter_native_exemplar=0` in structured metrics.
- `tcgen05_custom_vs_cublas` is intentionally named as a custom-versus-library comparison target so the benchmark surface matches the story it is telling.
- `build/` caches CUDA object files per configuration; clean via `python cleanup.py --include-build` when adjusting toolchains.
