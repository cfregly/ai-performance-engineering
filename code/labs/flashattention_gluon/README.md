# Lab - Tiled Triton Attention (legacy Gluon target)

## Summary
Benchmarks ordinary tiled Triton attention with online softmax against an eager attention reference. Historical Gluon target/module names remain compatible; no Gluon DSL, warp specialization, or TMA implementation is claimed.

## Problem
Attention-stack integrations can look "fast" because the benchmark is fuzzy. This lab keeps the pair narrow so you can see whether the tiled Triton path improves this workload on the target stack.

## Baseline Path
- simple attention reference path
- correctness anchor for the optimized implementation
- no fused fast-path assumptions

## Optimized Path
- FlashAttention-style optimized path
- same workload and harness contract
- focused on local integration cost/benefit, not a synthetic peak score

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flashattention_gluon` | `0.205 ms` | `0.154 ms` | `1.33x` |

This is a historical divisible-length, noncausal workload result. It does not qualify the corrected tail masking, causal masking, padded head dimension, or current runtime. Fresh CUDA numerical and performance checks remain pending.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal
```

## Learning Goals
- Compare tiled ordinary-Triton attention with the eager reference using identical shapes and masks.
- Measure backend-path value without mixing in unrelated model-level effects.
- Use a small, stable attention benchmark as an integration health signal.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashattention_gluon.py`, `optimized_flashattention_gluon.py` | Baseline and optimized harness entrypoints. |
| `flashattention_gluon_common.py` | Shared workload setup and helper code. |
| `expectations_{hardware_key}.json` | Regression thresholds for the lab. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
python -m cli.aisp bench run --targets labs/flashattention_gluon --profile minimal
```
- Targets follow the `labs/flashattention_gluon:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/flashattention_gluon:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- Check nonmultiples of 64, negative scores, causal and noncausal modes, and non-power-of-two head dimensions against PyTorch SDPA. Invalid columns receive negative infinity before softmax.
- Nonzero dropout is explicitly unsupported; output buffers must not alias inputs.
- `python -m cli.aisp bench run --targets labs/flashattention_gluon:flashattention_gluon --profile minimal` must pass numerical checks before any new speedup is accepted.

## Notes
- Treat this as an integration-health benchmark more than as a giant architectural headline win.
