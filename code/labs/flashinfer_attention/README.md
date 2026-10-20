# Lab - FlashInfer Block-Sparse Attention

## Summary
Runs a block-sparse attention kernel with FlashInfer and compares it to dense SDP plus an equivalent sparsity mask on an LLM-scale head configuration, including the output projection.

## Problem
Block-sparse attention is only interesting if the sparse kernel plus projection work actually beats the dense masked path. This lab keeps the output projection in both paths so the comparison stays honest.

## Baseline Path
- dense SDP plus sparsity mask
- same output projection as the optimized path
- useful correctness and cost-model reference

## Optimized Path
- FlashInfer block-sparse attention
- same head geometry and output projection
- tuned to measure sparsity benefits at realistic hidden sizes

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flashinfer_attention` | `1.043 ms` | `0.320 ms` | `3.26x` |

This is a good example of a sparse-kernel lab that still keeps the surrounding work visible instead of benchmarking an unrealistically stripped-down kernel in isolation.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile deep_dive --single-gpu
```

Use the deep-dive run when you want Nsight evidence for both the sparse attention kernel and the output projection path.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flashinfer_attention
python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile minimal
```

## Learning Goals
- Measure block-sparse attention speedups at high sparsity ratios.
- Validate FlashInfer kernels on realistic head dimensions.
- Profile attention plus output projection as a unit of work.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashinfer_attention.py` | Dense SDP + mask baseline with output projection. |
| `optimized_flashinfer_attention.py` | FlashInfer block-sparse attention with output projection. |
| `expectations_{hardware_key}.json` | Expectation files that keep the benchmark pair regression-checked on supported hardware. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flashinfer_attention
python -m cli.aisp bench run --targets labs/flashinfer_attention --profile minimal
```
- Targets follow the `labs/flashinfer_attention:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/flashinfer_attention:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flashinfer_attention:flashinfer_attention --profile minimal` captures the dense-vs-sparse delta on the same head geometry.
- The optimized path should verify against the dense masked reference before timing is reported.

## Notes
- The default head configuration targets a GPT-OSS-20B-style hidden size (`2880`) with `head_dim=64` (`45` heads).
- Increase `seq_len` if you want larger sparse regions to dominate the timing.
- Requires FlashInfer (`pip install flashinfer-python==0.6.3`).
