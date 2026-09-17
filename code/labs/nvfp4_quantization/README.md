# Fused NVFP4 quantization

This lab extends the repository's NVFP4 material with three operation comparisons:
residual add plus RMSNorm plus quantization, standalone quantization, and SiLU
plus multiply plus quantization. Each has a materialized PyTorch baseline and
a fused Triton kernel. Six shapes are workload configurations of those pairs.

| Pair | Workload configuration | Logical output shape |
| --- | --- | --- |
| `add_rmsnorm` | `rmsnorm_2048` (default) | 128 × 2048 |
| `add_rmsnorm` | `rmsnorm_4096` | 128 × 4096 |
| `add_rmsnorm` | `rmsnorm_8192` | 128 × 8192 |
| `quantize` | `quantize_14336` (default) | 128 × 14336 |
| `silu_mul` | `silu_mul_7168` (default) | 8 × 256 × 7168 |
| `silu_mul` | `silu_mul_14336` | 8 × 256 × 14336 |

[benchmarks.py](benchmarks.py) owns `WORKLOADS` and workload selection;
[kernels.py](kernels.py) owns fusion; [reference.py](reference.py) owns the explicit
format reference. Scale blocking reuses the existing GEMM layout through
[core/utils/nvfp4_layout.py](../../core/utils/nvfp4_layout.py). The independent
grouped-GEMM numerical oracle is unchanged. Existing [NVFP4 GEMM](../nvfp4_gemm/README.md)
and [block scaling](../block_scaling/README.md) remain the matrix-compute comparisons.

## Numerical and replay contract

- Inputs and transform outputs are BF16; RMSNorm accumulates in FP32 with epsilon `1e-6`.
- E2M1 values occupy two nibbles per byte, even column low, ties to even.
- E4M3FN scales cover 16 values; dequantization is `E2M1 × block_scale / global_scale`.
- A positive per-batch global multiplier is explicit; default workloads use 1.
- SiLU/multiply inputs have twice the output width and use masked expert rows,
  128×4 scale tiles, and deterministic zero padding. Other operations use linear scales.
- RMSNorm returns an updated residual separately, preserving the input for equivalent replay.
- Verification compares every packed byte, scale byte, and residual element with
  an independently executed materialized path after timing.

## Run

From `code/` on Blackwell SM100:

```bash
python -m cli.aisp bench list-targets --chapter labs/nvfp4_quantization
python -m cli.aisp bench run --targets labs/nvfp4_quantization:add_rmsnorm --profile minimal
python -m cli.aisp bench run --targets labs/nvfp4_quantization:add_rmsnorm --target-extra-arg 'labs/nvfp4_quantization:add_rmsnorm=--workload rmsnorm_4096' --profile minimal
python -m cli.aisp bench run --targets labs/nvfp4_quantization:silu_mul --target-extra-arg 'labs/nvfp4_quantization:silu_mul=--workload silu_mul_14336' --profile minimal
python -m pytest tests/test_nvfp4_quantization.py tests/test_concept_integration.py -q
```

The override reaches both benchmark arms. Unknown or incompatible configurations
fail explicitly. CPU tests check format bytes, shared layout, padding, and input
contracts. CUDA tests cover ragged cases and all six shapes with changed inputs.
SM100 and Triton are required for the GPU pairs; compilation, full GPU correctness,
and performance remain target-host validation work.

## Workload source

Workload shapes follow the
[pinned public B200 task export](https://github.com/wafer-ai/wafer-data/blob/7be29c1ea2a6681009b6e62ea778cba1d97d4a82/kernel-arena/exports/waferbench-nvfp4-b200/index.json),
whose historical reference is FlashInfer `0.2.6.post1`. These kernels are local
implementations; upstream speedups are not local measurements.
