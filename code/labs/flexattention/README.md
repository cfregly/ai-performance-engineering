# Lab - FlexAttention Harness

## Summary
Runs eager versus compiled FlexAttention with shared masks and score modifiers. A separate FlashAttention-4 CuTe tool checks the public FA4 path in its own pinned environment.

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
```

The harness run gives you the artifacted baseline and optimized pair. Run the CuTe tool from the dedicated environment below as a separate correctness and timing diagnostic. Its result is not part of the harness comparison.

## Dedicated FlashAttention-4 CuTe environment
The repository's default environment intentionally keeps FlashAttention 2.8.3 for NanoChat and CUTLASS DSL 4.5.2 for the B200 and GB300 labs. FlashAttention 2 also ships legacy `flash_attn/cute` files that do not use the CUTLASS 4.5.2 API. Do not install FlashAttention-4 over that environment because both distributions own many of the same package paths.

Create a separate Python 3.12 CUDA 13 environment from `code/`. Do not run the repository-wide `setup.sh` in this environment. The dedicated requirements keep Torch 2.9.1+cu130, CUTLASS DSL 4.5.2, and the newest FA4 release declared compatible with that DSL version.

```bash
python3.12 -m venv .venv-flexattention-cute
source .venv-flexattention-cute/bin/activate
python -m pip install -r labs/flexattention/requirements_cute.txt

# Run these commands only on an allocated CUDA 13 B200 host.
python -m labs.flexattention.flex_attention_cute --batch 2 --seq-len 1024
python -m pytest -q tests/test_audit_wave1_attention_regressions.py -k real_cute_layout_and_attention
```

The tool checks the Torch, FlashAttention-4, CUTLASS DSL, and Quack pins. It rejects an environment that also contains the `flash-attn` distribution. It imports the supported `flash_attn.cute.flash_attn_func` API and fails before timing when the critical environment does not match.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flexattention
python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
```

## Learning Goals
- Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.
- Validate the public FlashAttention-4 CuTe forward path in an isolated environment.
- Sweep sparsity knobs (block size, doc span) without editing source.
- Collect Nsight traces showing kernel fusion improvements after compiling.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flex_attention.py`, `optimized_flex_attention.py` | FlexAttention DSL workloads toggling `torch.compile` for fused kernels. |
| `flex_attention_cute.py`, `requirements_cute.txt` | Public FlashAttention-4 CuTe tool and its isolated CUDA 13 dependency lock. |
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
- In the dedicated environment, `python -m pytest -q tests/test_audit_wave1_attention_regressions.py -k real_cute_layout_and_attention` checks the public FA4 result against an independent full-output reference on CUDA.

## Notes
- Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.
- Keep FlashAttention 2 in the default repository environment so NanoChat continues to use the `flash_attn_2_cuda` extension.
- Artifacts include NVTX traces. Feed them to `core/analysis/deep_profiling_report.py` for convenience.
