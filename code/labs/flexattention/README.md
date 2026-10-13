# Lab - FlexAttention Harness

## Summary
Mirrors the FlexAttention CuTe DSL walkthrough: run eager vs compiled FlexAttention, compare to the CuTe path, and experiment with block masks, score modifiers, and Triton-style compilation.

## Problem
FlexAttention and FlashAttention-style paths are easy to describe and harder to verify. This lab is here to answer whether the compiled sparse/masked path in this repo actually beats the eager baseline on the same score modifiers and masks.

## Baseline Path
- eager FlexAttention path
- straightforward correctness reference
- higher Python and kernel-launch overhead

## Optimized Path
- compiled FlexAttention path
- same masks and score modifiers
- tuned for fused execution and fewer launches

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_all_singlegpu/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flex_attention` | `9.052 ms` | `0.320 ms` | `28.25x` |

That is exactly why this lab is useful: it keeps the mask/score-mod path visible while still showing a very large compile/fusion win on the local stack.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile deep_dive --single-gpu
python -m cli.aisp tools flex-attention-cute -- --batch 2 --seq-len 1024
```

The harness run gives you the artifacted baseline/optimized pair. The CuTe tool is the useful fallback when you want to compare semantics on systems without working FlexAttention bindings.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flexattention
python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
```

## Learning Goals
- Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.
- Validate CuTe-based FlashAttention fallbacks for platforms where FlexAttention is not available.
- Sweep sparsity knobs (block size, doc span) without editing source.
- Collect Nsight traces showing kernel fusion improvements after compiling.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flex_attention.py`, `optimized_flex_attention.py` | FlexAttention DSL workloads toggling `torch.compile` for fused kernels. |
| `flex_attention_cute.py` | CuTe/FlashAttention tool for hardware without FlexAttention bindings. |
| `flexattention_common.py`, `expectations_{hardware_key}.json` | Shared input builders, score modifiers, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flexattention
python -m cli.aisp bench run --targets labs/flexattention --profile minimal
```
- Targets follow the `labs/flexattention:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/flexattention:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal` captures the eager vs compiled delta and stores artifacts.
- `BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention` demonstrates masked sparsity sweeps.
- `python -m cli.aisp tools flex-attention-cute -- --batch 2 --seq-len 1024` succeeds even on systems missing FlexAttention bindings.

## Notes
- Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.
- Artifacts include NVTX traces; feed them to `core/analysis/deep_profiling_report.py` for convenience.
