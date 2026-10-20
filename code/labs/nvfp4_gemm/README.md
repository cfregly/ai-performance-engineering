# Lab - NVFP4 GEMM

## Summary
Benchmarks an NVFP4 GEMM kernel path against the higher-precision reference so you can measure what the precision/schedule tradeoff is actually buying.

## Problem
Low-precision GEMM work can devolve into kernel folklore quickly. This lab keeps the question narrow: what does the NVFP4 path actually save on this shape family, and does it stay verification-clean?

## Baseline Path
- higher-precision or less-specialized GEMM reference
- correctness anchor for the low-precision path
- useful for measuring the real cost/benefit of NVFP4

## Optimized Path
- NVFP4 GEMM kernel path
- same benchmark contract, lower-precision execution
- tuned to answer whether the precision/schedule tradeoff is worth it here

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `nvfp4_gemm` | `0.0189 ms` | `0.0128 ms` | `1.47x` |

That is a healthy microbenchmark win, but still the kind of result that must stay verification-gated. This lab is here to make that discipline visible.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_gemm
python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile minimal
```

## Learning Goals
- Measure the NVFP4 GEMM path under a strict verification contract.
- Keep low-precision wins attributable to a real benchmark pair instead of a submission-only script.
- Expose when the path regresses verification or only wins on one measurement surface.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_gemm.py`, `optimized_nvfp4_gemm.py` | Harness entrypoints for the reference and NVFP4 paths. |
| `baseline_submission.py`, `optimized_submission.py`, `local_eval_*.py` | Submission/evaluation helpers for the kernel lane. |
| `expectations_{hardware_key}.json` | Regression thresholds for the lab. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_gemm
python -m cli.aisp bench run --targets labs/nvfp4_gemm --profile minimal
```
- Targets follow the `labs/nvfp4_gemm:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/nvfp4_gemm:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/nvfp4_gemm:nvfp4_gemm --profile minimal` should keep the optimized path faster and verification-clean on current hardware.

## Notes
- The repo's NVFP4 labs are intentionally verification-heavy; a faster incorrect low-precision path is not an acceptable outcome.
