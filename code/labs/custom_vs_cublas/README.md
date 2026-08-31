# Lab - Custom Kernel vs cuBLAS

## Summary
Pits a hand-tuned TCGEN05/CUTLASS-style matmul path against the library baseline so you can see when a custom schedule is actually worth the maintenance cost.

## Problem
Custom matmul kernels are easy to oversell. This lab keeps the question narrow: for this shape family, does the custom path really beat the cuBLAS-style baseline enough to justify itself?

## Baseline Path
- library/reference matmul path
- stable correctness anchor
- useful for checking whether the custom kernel is actually better than "just use cuBLAS"

## Optimized Path
- custom TCGEN05-oriented matmul implementation
- same math, but explicit schedule/layout control
- designed to answer whether a bespoke kernel wins on this shape family

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `tcgen05_matmul` | `4.027 ms` | `1.740 ms` | `2.31x` |

That is a meaningful win, not just benchmark noise. The lab matters because it turns "custom vs vendor library" into a measured tradeoff rather than ideology.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile deep_dive --single-gpu
```

Use the deep-dive profile when you want to attribute the win to tile/schedule choices instead of relying on a single latency number.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/custom_vs_cublas
python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile minimal
python labs/custom_vs_cublas/autotune.py --help
```

## Learning Goals
- Benchmark a bespoke Blackwell-oriented matmul path against the library baseline.
- Keep kernel-selection and autotuning artifacts close to the benchmark pair.
- Make it obvious when a custom path is real value instead of benchmark folklore.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_tcgen05_matmul.py`, `optimized_tcgen05_matmul.py` | Baseline/optimized benchmark pair for the TCGEN05 matmul lab. |
| `autotune.py`, `cutlass_gemm/`, `experimental/` | Tuning helpers and implementation artifacts for the custom kernel path. |
| `expectations_{hardware_key}.json` | Regression thresholds for the benchmark pair. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/custom_vs_cublas
python -m cli.aisp bench run --targets labs/custom_vs_cublas --profile minimal
```
- Targets follow the `labs/custom_vs_cublas:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/custom_vs_cublas:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul --profile minimal` must establish full-output correctness on the exact supported target before its timings can be compared. Historical expectations do not qualify the repaired source or imply a speedup.
- The optimized path must stay verification-clean; a faster wrong kernel does not count.

## Notes
- This lab is one of the clearest places to show the repo's bias toward measured custom-kernel value instead of hand-wavy "handwritten kernels are faster" claims.
