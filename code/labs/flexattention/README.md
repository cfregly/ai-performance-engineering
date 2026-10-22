# Lab - FlexAttention Harness

## Summary
Runs eager versus compiled FlexAttention with shared masks and score modifiers. A FlashAttention-4 CuTe tool checks the public FA4 path in the repository's current pinned environment.

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

The harness run gives you the artifacted baseline and optimized pair. Run the CuTe tool from the shared environment below as a separate correctness and timing diagnostic. Its result is not part of the harness comparison.

## Shared FlashAttention-4 CuTe environment
The current shared graph pins Torch 2.14.1+cu130, Triton 3.8.0,
`flash-attn-4[cu13]` 4.0.0b36 and CUTLASS DSL 4.8.0. The CuTe tool uses that
same graph. Its `requirements_cute.txt` includes the repository's complete
Linux x86-64 hash lock instead of resolving a second set of direct dependencies.

Activate the environment created by `setup.sh`, or create a fresh Python 3.12
environment and install the include file from `code/`. The nested lock supplies
the CUDA index, the repository's vendored DLPack adapter, and every artifact
hash. A successful install does not qualify the GPU path.

```bash
python3.12 -m venv .venv-torch2141
source .venv-torch2141/bin/activate
python -m pip install --require-hashes --no-deps                   -r labs/flexattention/requirements_cute.txt

# Run these commands only on an allocated CUDA 13 B200 host.
python -m labs.flexattention.flex_attention_cute --batch 2 --seq-len 1024
python -m pytest -q tests/test_audit_wave1_attention_regressions.py -k real_cute_layout_and_attention
```

The tool checks the Torch, Triton, FlashAttention-4, CUTLASS DSL, Quack,
TVM FFI and DLPack-adapter pins. It rejects an environment that also contains
the `flash-attn` distribution because FA2 and FA4 own the same `flash_attn`
namespace. It imports the supported `flash_attn.cute.flash_attn_func` API and
fails before timing when the critical environment does not match. FA2
2.8.3.post2 is built and validated only in the separate environment described
by `native_torch_extensions.json`.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flexattention
python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal
```

## Learning Goals
- Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.
- Validate the public FlashAttention-4 CuTe forward path in the shared environment.
- Sweep sparsity knobs (block size, doc span) without editing source.
- Collect Nsight traces showing kernel fusion improvements after compiling.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flex_attention.py`, `optimized_flex_attention.py` | FlexAttention DSL workloads toggling `torch.compile` for fused kernels. |
| `flex_attention_cute.py`, `requirements_cute.txt` | Public FlashAttention-4 CuTe tool and an include of the shared Torch 2.14.1 hash lock. |
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
- In the shared environment, `python -m pytest -q tests/test_audit_wave1_attention_regressions.py -k real_cute_layout_and_attention` checks the public FA4 result against an independent full-output reference on CUDA.

## Notes
- Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.
- Keep FlashAttention 2 in its separate validation environment. Installing it beside shared FA4 would overwrite their common Python and native namespace.
- Artifacts include NVTX traces. Feed them to `core/analysis/deep_profiling_report.py` for convenience.
