# Lab - MoE CUDA PTX

## Summary
Benchmarks a routed top-2 SwiGLU MoE FFN with a conservative BF16 reference path and a staged optimized CUDA path. The lab is built as a standalone MoE kernel story: routing, token packing, grouped expert compute, MXFP8-style quantization surfaces, and layer compute/combine timing all live here. The optimized forward layer pre-packs routes in setup; its timing is not an end-to-end routing measurement.

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
- forward CUDA path that prepares route packing during setup and times the packed grouped compute/combine path without quantization; this timing boundary must be kept explicit when comparing it with baseline route masking

## Targets
| Target | Path |
| --- | --- |
| `labs/moe_cuda_ptx:moe_quant` | BF16 -> MXFP8-style quantization surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_fwd` | Grouped expert forward FFN surface |
| `labs/moe_cuda_ptx:moe_grouped_gemm_bwd` | Grouped expert forward+backward FFN surface |
| `labs/moe_cuda_ptx:moe_layer` | Routed top-2 SwiGLU layer; CUDA forward pre-packs routes outside timing |

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_cuda_ptx
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_layer --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_fwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_grouped_gemm_bwd --profile minimal
python -m cli.aisp bench run --targets labs/moe_cuda_ptx:moe_quant --profile minimal
```

Layer forward accuracy policy:

`moe_layer --mode forward` does not quantize activations. It now requires `AISP_MOE_PTX_LAYER_ACCURACY_POLICY` naming a reviewed JSON policy with `schema_version: 1` and a `moe_layer_forward` object containing `relative_l2` and `normalized_max_abs`. Both limits must be finite and in `[0, 1)`; no default numerical threshold is supplied. Full logical outputs are checked against an independent unquantized BF16/FP16 reference using original input/weight/routing snapshots. Missing rows, nonfinite results and aliased reference storage fail regardless of pairwise tolerance. A configured policy is not measured GPU accuracy evidence.

The original `(rtol=0.05, atol=0.2)` exception falsely attributed drift to quantization. Even the restored `(0.02, 0.02)` pairwise budget can accept all zeros for small outputs; it is only an additional pairwise check after the scale-invariant reference policy. All other quant/grouped/backward verification surfaces retain their existing contracts and are not newly qualified by this change.

After obtaining the target GPU lease, collect diagnostics from the repository root with `python docs/audits/2026-08-30/evidence/moe-ptx/calibrate_layer_accuracy.py --output /absolute/new/attempt.json`. The tool records real baseline and CUDA-BMM errors for the chosen shape/seed/routing, preserves failures, and never generates an acceptance threshold. The full audit receipt includes the bounded CUDA test matrix. Route generation supports `top_k=2` only and now rejects unsupported overrides explicitly.

Useful debug overrides:
```bash
python -m cli.aisp bench run \
  --targets labs/moe_cuda_ptx:moe_layer \
  --target-extra-arg labs/moe_cuda_ptx:moe_layer="--num-tokens 4096 --hidden-dim 2048 --expert-ffn-dim 1024 --mode forward --histogram skewed"
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
