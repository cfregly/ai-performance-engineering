# Lab - NVFP4 Grouped GEMM

## Summary
Explores grouped-GEMM routing and schedule variants across multiple shapes so you can see where the grouped NVFP4 path is actually winning and where it is merely legal.

## Problem
Grouped GEMM tuning is noisy and easy to overclaim. This lab keeps the shape routing explicit and benchmarked so promotions are based on repeated verified wins instead of one-off lows.

## Baseline Path
- independent E2M1 unpacking and original-scale application followed by FP64 matmul, rounded to FP16 output
- all five baseline wrappers use `reference_math.py`; they do not import the custom kernel or reordered-scale packing
- unpacking and matmul are included in the measured baseline; this differs from the former conservative tcgen05 baseline
- canonical baseline front-door target for the promoted grouped shape
- shape-specific baseline companions for the other workload shapes
- useful for showing which grouped shapes are hard versus easy

## Optimized Path
- canonical optimized front-door target for the promoted grouped shape
- shape-specific tuned NVFP4 grouped GEMM variants
- same grouped workloads, but explicit schedule/routing choices
- designed to keep promotions tied to repeated verify and ABAB checks

## Historical Delta (not valid for the new baseline)
The table below records earlier custom-kernel-versus-custom-kernel timings. It does not establish independent numerical correctness and cannot be used as the speedup of the new unpack-and-FP64 baseline. Every logical request now has an independent full-output oracle; actual GPU validation and new timings remain pending:

| Target | Baseline | Optimized | Measured delta | Contract |
| --- | ---: | ---: | ---: | --- |
| `nvfp4_group_gemm` | `0.614 ms` | `0.595 ms` | `1.03x` | canonical local-contract speed benchmark |
| `nvfp4_group_gemm_g2_n3072_k4096` | `0.615 ms` | `0.594 ms` | `1.04x` | promoted shape companion (same routed workload as the front-door target) |
| `nvfp4_group_gemm_g8_n7168_k2048` | `2.080 ms` | `2.019 ms` | `1.03x` | supplementary comparison benchmark |
| `nvfp4_group_gemm_g8_n4096_k7168` | `2.408 ms` | `2.379 ms` | `1.01x` | supplementary comparison benchmark |
| `nvfp4_group_gemm_g2_n4096_k1536` | `0.352 ms` | `0.352 ms` | `1.00x` | supplementary comparison benchmark |

The older strict all-case snapshots in `artifacts/runs/20260302_rerun_all_labschapters_strict/` are still useful historical router evidence, but they are not the current runnable truth for this harness surface on this host. The former competition `caseN` numbering is retired from the public benchmark targets; the canonical front-door target now points at the isolated single-target winner `g2_n3072_k4096`.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile deep_dive --single-gpu
```

Use the harness artifacts for schedule attribution, then use the router/ABAB tooling for promotion decisions. The benchmark pair tells you the shape of the win; the tuning scripts decide whether a default should actually move.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```

## Learning Goals
- Keep grouped-GEMM tuning grounded in repeated verified shape-by-shape evidence.
- Keep non-winning shapes visible as supplementary comparison benchmarks without forcing them to carry the lab's canonical speed claim on every host.
- Retire the old competition `caseN` labels from the public harness surface.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_group_gemm.py`, `optimized_nvfp4_group_gemm.py` | Canonical front-door grouped-GEMM target for the promoted shape. |
| `baseline_nvfp4_group_gemm_g*.py`, `optimized_nvfp4_group_gemm_g*.py` | Shape-specific grouped-GEMM companions for the remaining workload shapes. |
| `WORKLOG.md`, `custom_cuda_submission.py`, `custom_cuda_group_gemm_kernel.cu` | Tuning history and custom CUDA implementation. |
| `nvfp4_group_gemm_common.py`, `nvfp4_group_gemm_inputs.py`, `reference_math.py` | Workload generation, all-request verification, and independent mathematical baseline. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```
- Targets follow the `labs/nvfp4_group_gemm:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/nvfp4_group_gemm:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- Validate every element of every logical request against the independent reference, including requests hidden by fused-call compression. Negative controls must reject zeros, corruption, NaNs and reference aliasing.
- Requested CUDA-graph capture failure is an error; explicitly set `AISP_NVFP4_GROUP_GEMM_CAPTURE_ITER_GRAPH=0` for an eager comparison.
- The CPU checks validate the reference and verifier contract only. Recompile and run all routes on SM100 before treating any old expectation as current.
- `python -m cli.aisp bench run --targets labs/nvfp4_group_gemm:nvfp4_group_gemm --profile minimal` must pass the new independent full-output check before any timing is accepted.
- `nvfp4_group_gemm_g8_n4096_k7168`, `nvfp4_group_gemm_g8_n7168_k2048`, and `nvfp4_group_gemm_g2_n4096_k1536` remain supplementary comparison benchmarks; use the ABAB/router tooling when deciding whether any of them should become canonical speed-claim targets again.
- Old `caseN` target names should no longer appear in `python -m cli.aisp bench list-targets --chapter labs/nvfp4_group_gemm`.
- Default changes should still be gated by the stricter ABAB/verify process documented in the codebase notes, not by a single benchmark run.

## Notes
- This lab is intentionally stricter than a normal benchmark pair because grouped-GEMM route tuning is unusually noise-prone.
- Target names remain stable for compatibility; no route is newly promoted by this CPU-only remediation.
