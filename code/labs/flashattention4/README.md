# Lab - FlashAttention-4 Pipeline Co-Design

## Summary
Recreates the practical shape of the FlashAttention-4 article: eager FlexAttention as the scalar-heavy baseline, then a compiled Blackwell-friendly path that tries the FLASH backend and falls back to FlexAttention+TMA when needed. The default benchmark uses ALiBi because it is stable on the local stack and still exercises the FA4 score-mod path.

## Colfax decode and backward kernel ablations
The [Colfax optimization diaries guide](colfax_optimization_diaries.md) extends
this lab with two direct upstream FA4 comparisons, linked from Chapters 10, 11,
and 18. These use separate, pinned PR revisions and do not use the provider
selection path described in the older forward experiments below.

| Harness target | `baseline_` path | `optimized_` path |
| --- | --- | --- |
| `flashattention4_decode` | Single S/P TMEM slot | Two-slot S/P ping-pong, overlapping next-tile QK with softmax |
| `flashattention4_backward` | Aliased P/S and dS/dP; compute-wide barriers | Dedicated P/dS storage plus warp-local synchronization at head dimension 64 |

Both arms replay one captured CUDA graph per iteration. JIT compilation, warmup,
and capture are setup costs. Backward times preprocessing, gradient kernels, and
postprocessing; its common forward output/LSE are prepared outside the timer.
Verification consumes the full replay output, including **all of dQ, dK, and dV**
for backward. Deterministic backward requires exact gradient equality between
arms; the GPU test also checks an independent high-precision reference.
`optimized_` identifies the candidate, not a measured speedup.

Use **two separate CUDA 13 / SM100 environments** with the repository harness
dependencies available. These experiments need CUTLASS DSL 4.6.2, whereas the
repository's default stack pins 4.5.2. Install one recipe per environment; installing
both in one environment replaces the first FA4 revision. From `code/`:

```bash
# In the dedicated decode environment:
python -m pip install -r labs/flashattention4/requirements_colfax_decode.txt
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_decode --profile deep_dive --single-gpu
AISP_TEST_COLFAX_KIND=decode python -m pytest -q tests/test_flashattention4_colfax.py

# In the dedicated backward environment:
python -m pip install -r labs/flashattention4/requirements_colfax_backward.txt
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_backward --profile deep_dive --single-gpu
AISP_TEST_COLFAX_KIND=backward python -m pytest -q tests/test_flashattention4_colfax.py
```

Missing CUDA, a non-SM100 GPU, or mismatched installed source produces
an explicit `SKIPPED:` diagnostic. Before importing CUDA code, the loader
verifies the exact VCS installation and all 52 upstream runtime Python files.
There is no substitute backend. The source
pins are in [colfax_upstream.json](colfax_upstream.json); benchmark workloads and
the performance hypothesis are in [colfax_workload_spec.yaml](colfax_workload_spec.yaml)
and [colfax_performance_intake.yaml](colfax_performance_intake.yaml).

These new pairs have **no local GPU measurements or expectations yet**. Run the
opt-in correctness tests and retain harness clock/provenance, interleaved repeat,
Nsight Systems, and Nsight Compute evidence before claiming a win. The older
results below apply only to their named forward/provider targets.

Local validation on 2026-09-07 (macOS, Python 3.12, CPU PyTorch 2.14.0):

- `python -m pytest -q tests/test_flashattention4_colfax.py`: **31 passed, 1 skipped**; the opt-in SM100 test was not enabled. The GPU test changes Q or dO on the same graph, compares against an independent reference, and checks restored-input replay.
- `python scripts/linting/check_benchmarks.py labs/flashattention4/baseline_flashattention4_decode.py labs/flashattention4/optimized_flashattention4_decode.py labs/flashattention4/baseline_flashattention4_backward.py labs/flashattention4/optimized_flashattention4_backward.py`: **0 errors, 0 warnings**.
- `python -m cli.aisp bench list-targets --chapter labs/flashattention4`: both new targets discovered.
- Ruff, syntax, documentation links, requirement/manifest consistency, and SHA256 checks against the pinned upstream sources passed. Direct setup returned the expected CUDA-required `SKIPPED:` diagnostic.

## Problem
This lab is here to test two different questions cleanly:
- does the fused FA4-style path beat the eager score-materializing baseline in this repo?
- does the local stack reproduce the Colfax / PyTorch FlashAttention-4 performance envelope?

## Baseline Path
- eager FlexAttention
- explicit score materialization
- good correctness reference, bad steady-state cost model

## Optimized Path
- compiled Blackwell-oriented path
- prefers the experimental FLASH backend
- falls back to compiled FlexAttention + TMA when the backend/toolchain combination cannot lower cleanly

## Latest recorded validation and historical results
The latest repository handoff, dated **2026-08-17** in `HANDOFF.md`, records the ALiBi target passing input/output verification but failing the 1.05x speed gate on B200:

| Path | Latency | Relative |
| --- | ---: | ---: |
| Baseline | `3.429079 ms` | `1.00x` |
| Optimized | `3.572818 ms` | `0.959769x` |

That run had a first-valid-provider selection defect. The source fix measures all correct compiled providers; B200 re-verification remains pending. The recorded result does not prove an inherent regression, but it does not support a current speedup claim.

The March 6, 2026 virtualized-host result (5.562 ms / 0.385 ms, 14.45x) in `artifacts/runs/20260306_023114__bench__profile_none_targets_labs_flashattention4_flashattention4_alibi/` is historical and superseded as the latest validation record. It is not the current accepted result. ALiBi and softcap FLOP accounting now uses their actual causal masks, so earlier dense-count TFLOP/s figures for those modes also require recomputation.

No new GPU measurement was performed for this audit repair, and reproducing the published Colfax/PyTorch envelope remains unverified.

## Profiler Evidence
Use the harness for artifacted Nsight evidence:

```bash
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile deep_dive --single-gpu
```

Use the microbenchmark when you want the closest backend-vs-backend comparison to the published articles:

```bash
python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention4
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal
python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
```

## Learning Goals
- Measure the delta between eager score materialization and a fused compiled attention kernel.
- Exercise FA4-style score modifiers such as ALiBi and soft-capped logits, and optionally probe sliding-window masks on a best-effort basis.
- Inspect provider selection on Blackwell (`flash_backend` vs `flex_tma`).
- Use a coarse pipeline model to explain why overlap matters more under asymmetric hardware scaling.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashattention4.py`, `optimized_flashattention4.py` | Benchmark pair comparing eager FlexAttention to a compiled, provider-aware FA4 path. |
| `flashattention4_common.py` | Shared input builders, score mods, mask construction, and provider resolution. |
| `pipeline_model.py` | Latency model for serial versus overlapped attention tiles. |
| `tflops_microbench.py` | Clock-locked TFLOPs/s microbenchmark for Colfax/PyTorch-style backend comparisons. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flashattention4
python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_softcap --profile minimal
python labs/flashattention4/pipeline_model.py --tiles 32 --tensor-core-scale 4 --scalar-scale 2
python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi
python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa
```
- Harness workflows use explicit targets such as `flashattention4_dense`, `flashattention4_causal`, `flashattention4_alibi`, `flashattention4_softcap`, `flashattention4_windowed`, `flashattention4_alibi_windowed`, and the matching `best_available_attention_*` variants.
- On the local `torch 2.9.1+cu130` build, `windowed` and `alibi_windowed` are experimental: the optimized path can produce non-finite outputs on a fresh compile even though upstream FA4 supports sliding-window patterns.
- `tflops_microbench.py` locks GPU clocks through `core.harness.benchmark_harness.lock_gpu_clocks()` by default; use `--no-lock-gpu-clocks` only for local debugging.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flashattention4 --profile minimal` shows the eager baseline materializing scores while the optimized path stays fused.
- `python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_alibi --profile minimal` succeeds on a cold-start process and exercises the FA4 score-mod path without relying on env vars.
- `python -m cli.aisp bench run --targets labs/flashattention4:best_available_attention_dense --profile minimal` gives the clearest absolute-performance path for standard attention on this stack.
- `python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_windowed --profile minimal` and `labs/flashattention4:flashattention4_alibi_windowed` remain explicit experimental probes; treat failures there as a PyTorch/FA4 integration limitation on this stack rather than as a lab bug.
- `python labs/flashattention4/pipeline_model.py --tiles 64 --tensor-core-scale 4 --scalar-scale 2` demonstrates overlap becoming more valuable as tensor cores scale faster than scalar hardware.
- `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal alibi` runs the public-shape backend comparison against the local FLASH backend, the local Triton-style proxy, and cuDNN where supported.
- `python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` checks whether a larger compute-bound shape moves the local stack toward the published Colfax/PyTorch envelope.

## TFLOPs/s Microbenchmark
Use `tflops_microbench.py` when you want something closer to the published Colfax and PyTorch comparisons than the harness benchmark pair. The harness pair is intentionally end-to-end and compares eager score materialization against a fused kernel; the microbenchmark instead compares backend implementations on the same attention workload.

| Published comparison target | Local command | Notes |
| --- | --- | --- |
| Colfax B200 BF16 forward envelope (`1605 TFLOPs/s`, up to `1.3x` over cuDNN 9.13, up to `2.7x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset peak_probe --mode dense causal --backends flash_backend triton_flex cudnn_sdpa` | Uses a larger shape to push the local stack harder. |
| PyTorch GB200 standard-attention forward envelope (`1.6x-3.2x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode dense causal` | Uses the public blog shape `B=2, H=8, S=2048, D=128`. |
| PyTorch GB200 ALiBi forward envelope (`1.2x-2.1x` over Triton) | `python labs/flashattention4/tflops_microbench.py --preset public_blog --mode alibi --backends flash_backend triton_flex flex_tma` | cuDNN SDPA is not applicable to ALiBi. |

The FLOP accounting matches the common SDPA forward convention used in vendor/blog comparisons:
`forward_flops = 4 * batch * heads * head_dim * nonmasked_attention_elements`

- For `dense`, `nonmasked_attention_elements = q_seq_len * kv_seq_len`.
- For `causal`, `alibi`, and `softcap`, count triangular causal attention pairs; for `windowed` and `alibi_windowed`, count the exact causal-window pairs. These are effective mathematical FLOPs, not measured hardware instructions.
- `triton_flex` is the closest local proxy for the blog's Triton baseline: compiled FlexAttention with `USE_TMA=False`.

## Historical Local Results (March 5, 2026)
These historical measurements were recorded on March 5, 2026 with `torch 2.9.1+cu130` and harness clock locking on a virtualized host. They are preserved as recorded, not requalified against the corrected code or evidence of the current host state. The historical ALiBi TFLOPs/s entries used a dense numerator for causal work and are not corrected throughput rates; corrected accounting requires the causal pair count.

### Public Blog Shape (`B=2, H=8, S=2048, D=128`)
| Mode | Backend | Median (ms) | TFLOPs/s | Flash vs Triton | Flash vs cuDNN | Published check |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `dense` | `flash_backend` | 0.224 | 153.6 | `1.02x` | `0.40x` | Outside Colfax and PyTorch ranges |
| `dense` | `triton_flex` | 0.229 | 150.1 | `1.00x` | `0.39x` | Local Triton-style proxy |
| `dense` | `cudnn_sdpa` | 0.090 | 382.5 | `2.55x` | `1.00x` | Local cuDNN leader |
| `causal` | `flash_backend` | 0.238 | 72.1 | `14.84x` | `0.37x` | Beats local Triton-style proxy, still far below cuDNN |
| `causal` | `triton_flex` | 3.538 | 4.9 | `1.00x` | `0.02x` | Local Triton-style proxy collapses on this stack |
| `causal` | `cudnn_sdpa` | 0.088 | 195.5 | `40.25x` | `1.00x` | Local cuDNN leader |
| `alibi` | `flash_backend` | 6.221 | 5.5 | `1.02x` | n/a | Outside PyTorch ALiBi range |
| `alibi` | `triton_flex` | 6.323 | 5.4 | `1.00x` | n/a | Local Triton-style proxy |
| `alibi` | `flex_tma` | 6.169 | 5.6 | `1.03x` | n/a | Slightly ahead locally, still not near published envelope |

### Peak Probe Shape (`B=8, H=16, S=4096, D=128`)
| Mode | Backend | Median (ms) | TFLOPs/s | % of Colfax 1605 | Flash vs Triton | Flash vs cuDNN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | `flash_backend` | 3.576 | 307.5 | 19.2% | `1.01x` | `0.34x` |
| `dense` | `triton_flex` | 3.614 | 304.2 | 19.0% | `1.00x` | `0.34x` |
| `dense` | `cudnn_sdpa` | 1.222 | 899.8 | 56.1% | `2.96x` | `1.00x` |
| `causal` | `flash_backend` | 2.264 | 242.9 | 15.1% | `0.97x` | `0.36x` |
| `causal` | `triton_flex` | 2.200 | 250.0 | 15.6% | `1.00x` | `0.37x` |
| `causal` | `cudnn_sdpa` | 0.814 | 675.1 | 42.1% | `2.70x` | `1.00x` |

That historical snapshot did not reproduce the published Colfax or PyTorch FlashAttention-4 envelope. Within that snapshot, the larger probe reported `307.5 TFLOPs/s` on dense and `242.9 TFLOPs/s` on causal, well below both Colfax's `1605 TFLOPs/s` peak and the local cuDNN path.

## Notes
- Sources: Colfax Research's FlashAttention-4 article (`https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/`) and the PyTorch FlexAttention + FlashAttention-4 integration post (`https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/`).
- For a smaller, schedule-first explanation surface, see `labs/software_pipelining`, which models same-iteration, loop-carried, and anti-dependency constraints without requiring a full FA4 kernel.
- Colfax reports up to `1605 TFLOPs/s` on B200 BF16 at roughly `71%` utilization, plus up to `1.3x` over cuDNN 9.13 and `2.7x` over Triton for forward passes.
- The PyTorch post reports `1.6x-3.2x` forward speedup over Triton for standard dense/causal attention on GB200, `1.2x-2.1x` for ALiBi, and `1.4x-2.1x` for sliding-window attention.
- The local PyTorch/Triton stack needs a quoted backend literal for the experimental FLASH backend; the lab handles that workaround internally and falls back automatically if needed.
- The lab pins float32 accumulation to IEEE mode because the current sm_100 lowering produced non-finite outputs under TF32 accumulation.
- Sliding-window modes remain exposed as explicit benchmark targets, but the stable day-to-day harness path is `flashattention4_alibi`.
