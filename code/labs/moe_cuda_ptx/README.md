# Lab - MoE CUDA PTX

## Summary
Benchmarks a routed top-2 SwiGLU MoE FFN with a conservative BF16 reference path and a staged optimized CUDA path. The lab covers routing, token packing, grouped expert compute, MXFP8-style quantization surfaces, and layer timing. Both forward-layer variants include route selection or packing, expert compute, and output combination in every measured call. Expert assignments and weights are supplied inputs; router-network computation is outside this contract.

## Problem
MoE optimization claims usually blur together three different costs:

- routing and token packing,
- grouped expert GEMM, and
- quantization / layout preparation for low-precision kernels.

This lab keeps those surfaces separate and states whether route preparation is inside or outside each timed surface.

## Baseline Path
- BF16 reference path
- explicit top-2 token-to-expert dispatch
- per-expert Python / eager grouped execution
- conservative quantization path with explicit reshape / transpose work

## Optimized Path
- grouped expert execution on pre-packed token buckets
- vectorized expert BMM path for forward and backward
- MXFP8-style activation quantization surface benchmarked separately via `moe_quant`
- forward CUDA path that times route packing, grouped compute, and output combination without quantization

## Targets
| Target | Path |
| --- | --- |
| `labs/moe_cuda_ptx:moe_quant` | BF16 -> MXFP8-style quantization surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_fwd` | Grouped expert forward FFN surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_bwd` | Grouped expert forward+backward FFN surface |
| `labs/moe_cuda_ptx:moe_layer` | Routed top-2 SwiGLU layer including per-call route packing and output combination |

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_cuda_ptx
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_layer \
  --target-extra-arg labs/moe_cuda_ptx:moe_layer="--num-tokens 257 --num-experts 8 --hidden-dim 512 --expert-ffn-dim 256 --capacity-factor 1.5 --mode forward --histogram balanced" \
  --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_fwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_bwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_quant --profile minimal
```

Layer forward accuracy policy:

`moe_layer --mode forward` does not quantize activations. Its packaged
`layer_accuracy_policy.json` uses schema 3 and records separate FP16 and BF16
limits plus exact validated workload cells. Full logical outputs are checked
against an independent unquantized reference built from original input,
weight, and routing snapshots. Missing rows, nonfinite results, shape changes,
and aliased reference storage fail regardless of the numerical limits.

The limits were fixed before held-out testing: relative L2 must be at most one
representable epsilon for the workload dtype, and maximum absolute error
normalized by the maximum reference magnitude must be at most two epsilons.
Calibration covered both layer implementations, both routing histograms,
multiple shapes, seeds, and input amplitudes, and an independent FP64
calculation. Held-out shapes and seeds passed without changing the limits.
A separate default-cell review used fixed limits, an independent FP64 logical
route calculation, distinct calibration and holdout seeds, and both layer
implementations. Zero, half-scale, dropped-row, nonfinite, aliased, and
truncated outputs were rejected. The policy defines an acceptance bound; it is
not a performance claim or a substitute for measured GPU evidence on a new
platform.

The reviewed domain contains the exact forward cells `65x48x32` with four
experts, `129x128x64` with four experts, and `257x512x256` with eight experts,
all with top-2 routing, capacity factor 1.5, both supported dtypes, and both
routing histograms. It also contains the exact default `32768x7168x2048`,
eight-expert, top-2, capacity-factor-1.25 cell for BF16 with balanced routing.
That dedicated review does not cover FP16, skewed routing, or nearby default
shapes. Dimensions cannot be mixed across entries.

`AISP_MOE_PTX_LAYER_ACCURACY_POLICY` remains available for explicit audit
experiments. An override must use the same schema 3 limits and exact-domain
structure and must pass the same structural checks.

The original `(rtol=0.05, atol=0.2)` exception falsely attributed drift to quantization. Even the restored `(0.02, 0.02)` pairwise budget can accept all zeros for small outputs; it is only an additional pairwise check after the scale-invariant reference policy. All other quant/grouped/backward verification surfaces retain their existing contracts and are not newly qualified by this change.

After obtaining the target GPU lease, collect a new diagnostic from the
repository root with `python docs/audits/2026-08-30/evidence/moe-ptx/calibrate_layer_accuracy.py --output /absolute/new/attempt.json`.
The tool records real baseline and CUDA-BMM errors for the chosen
shape/seed/routing, preserves failures, and never generates or changes the
acceptance limits. Route generation supports `top_k=2` only and rejects
unsupported overrides explicitly.

Useful debug overrides:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_layer \
  --target-extra-arg labs/moe_cuda_ptx:moe_layer="--num-tokens 257 --num-experts 8 --hidden-dim 512 --expert-ffn-dim 256 --capacity-factor 1.5 --mode forward --histogram skewed"
```

Backward verification shape:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_grouped_gemm_bwd \
  --target-extra-arg labs/moe_cuda_ptx:moe_grouped_gemm_bwd="--num-tokens 4096 --hidden-dim 2048 --expert-ffn-dim 1024 --mode fwd_bwd"
```

## Historical Status and Current Acceptance
- Historical implementation notes reported debug-shape `moe_quant` verification and a directional CUDA win. That is not a current source/host qualification.
- Historical implementation notes reported a debug-shape grouped-forward win after moving timing to the packed core. This receipt does not revalidate those results.
- Historical direct-runtime claims for grouped backward and layer do not establish current acceptance. The old layer forward tolerance accepted all-zero output on bounded real CPU fixtures, and its 32x32 payload omitted the last rows/channels. Current layer forward requires full-output reference checks and a reviewed accuracy policy; CUDA/runtime/performance gates remain pending.
- The PTX path is still an explicit Blackwell-gated scaffold. The current optimized backend is CUDA.

Debug-shape verification used during implementation:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_quant \
  --profile minimal \
  --target-extra-arg 'labs/moe_cuda_ptx:moe_quant=--num-tokens 2048 --hidden-dim 1024 --expert-ffn-dim 512 --mode forward'

python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_grouped_gemm_fwd \
  --profile minimal \
  --target-extra-arg 'labs/moe_cuda_ptx:moe_grouped_gemm_fwd=--num-tokens 2048 --hidden-dim 1024 --expert-ffn-dim 512 --mode forward'
```

## Learning Goals
- Keep routing local to the lab instead of hiding it behind another chapter.
- Compare grouped expert execution with eager per-expert masking while accounting for different routing preparation.
- Measure quantization work as a first-class MoE cost, not a hidden helper stage.
- Leave a clear upgrade path for a future Blackwell `tcgen05` PTX backend.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_moe_*.py`, `optimized_moe_*.py` | Thin benchmark wrappers discovered by the harness. |
| `moe_cuda_ptx_common.py` | Shared workload config, routing, grouped FFN paths, quantization, and benchmark class. |
| `moe_cuda_ptx_extension.py`, `moe_cuda_ptx_stub.cu` | PTX-backend scaffold and Blackwell gating for future milestone work. |
| `expectations_{hardware_key}.json` | Reserved for strict expectations once the lab settles. |

## Notes
- The current optimized path is the staged CUDA milestone, not the final PTX/tcgen05 backend.
- The PTX scaffold is intentionally explicit and Blackwell-gated so the future tcgen05 path can land without changing the benchmark interface.
