# Lab - Training Hot-Path

## Summary
Adds three small supporting examples for training-side bottlenecks that were missing from the main chapter flow: vectorized metric aggregation, fused CUDA reduction, and padding-aware projection work that skips padded rows without changing the chapter-primary stories in `ch12`, `ch14`, or `labs/async_input_pipeline`.

## Problem
Some training bottlenecks are too small and cross-cutting to deserve a chapter rewrite, but still matter in practice. This lab isolates three of them so they can be benchmarked honestly:

- scalar Python-side metric aggregation that should have been vectorized
- segmented reductions that should have been fused into one CUDA pass
- padded projection work that should only run on active tokens

## Baseline Path
- scalar or unfused reduction logic
- padded dense projection work over every token position
- simple reference paths that preserve correctness but spend work on overhead and padding

## Optimized Path
- tensor-vectorized metric aggregation
- a local CUDA extension for fused segmented abs-mean reduction
- packed-row projection kernels that only touch active rows before scattering results back into the padded layout

## Measured Delta
Strict `minimal` expectation-backed runs on the current `b200` hardware key produced the following deltas:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `metric_reduction_vectorized` | `14.050 ms` | `0.158 ms` | `89.12x` faster |
| `metric_reduction_cuda` | `2.760 ms` | `0.050 ms` | `55.06x` faster |
| `padding_aware_transformer` | `2.919 ms` | `3.430 ms` | `0.85x` speed, `76.4%` lower peak memory |

## Goal Semantics
Treat `padding_aware_transformer` as a memory-goal benchmark. Its value is the large reduction in padded-row activation footprint; a slower timed path is acceptable when the measured memory drop is material and verification stays green.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/training_hotpath:metric_reduction_vectorized --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/training_hotpath:metric_reduction_cuda --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/training_hotpath:padding_aware_transformer --profile deep_dive --single-gpu
```

Use the profile when you want to see the missing work directly:
- vectorized vs scalar host-side reduction behavior
- one fused CUDA reduction kernel instead of repeated segmented slicing
- active-row packing that reduces padded projection work

## Repro Commands
```bash
python -m labs.training_hotpath.compare --example metric_reduction_vectorized
python -m cli.aisp bench list-targets --chapter labs/training_hotpath
python -m cli.aisp bench run --targets labs/training_hotpath --profile minimal
```

## Learning Goals
- Benchmark vectorized training-side metric aggregation instead of leaving it as an unmeasured Python loop.
- Show when a local CUDA fused reduction is worth the extension overhead.
- Make padding-aware projection savings visible without changing the primary CUDA Graph or compiler chapter narratives.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_metric_reduction_vectorized.py`, `optimized_metric_reduction_vectorized.py` | Scalar per-output aggregation vs tensor-vectorized metric reduction. |
| `baseline_metric_reduction_cuda.py`, `optimized_metric_reduction_cuda.py` | Torch segmented reduction baseline vs fused CUDA extension. |
| `baseline_padding_aware_transformer.py`, `optimized_padding_aware_transformer.py` | Dense padded projections vs packed-row padding-aware projections. |
| `training_hotpath_common.py`, `training_hotpath_extension.py`, `training_hotpath_kernels.cu`, `compare.py` | Shared workloads, local extension loader, CUDA kernels, and a direct compare entrypoint. |
| `expectations_{hardware_key}.json` | Regression thresholds for the three supporting benchmark pairs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/training_hotpath
python -m cli.aisp bench run --targets labs/training_hotpath --profile minimal
```
- Targets follow the `labs/training_hotpath:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/training_hotpath:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench list-targets --chapter labs/training_hotpath` should discover the three supporting benchmark pairs.
- The optimized metric reduction targets should match their baselines numerically while flipping `metric_reduction.is_vectorized` or `metric_reduction.is_fused_cuda` in custom metrics.
- The padding-aware target should preserve outputs while flipping `padding_aware.enabled`, reporting a non-trivial padded-token fraction, and reducing peak memory even if raw time is not faster.

## Notes
- Keep `ch14:model_compile_reduced_precision` as the primary compile + reduced-precision training story.
- Keep `ch12:cuda_graphs` as the primary CUDA Graph replay story.
- Keep `labs/async_input_pipeline:async_input_pipeline` as the primary copy-stream overlap story.
- `padding_aware_transformer` is a memory-goal benchmark; judge it by peak-memory reduction first, not by raw speedup.
