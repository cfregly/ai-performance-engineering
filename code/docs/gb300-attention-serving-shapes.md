# GB300 attention roofline at serving-class shapes (Front AT/AT2)

**Question.** B38 closed the attention quartet (fa4 lab 14 arms, gluon, flex, cudnn_sdpa)
as "overhead-bound micro-shapes, 0.06-3.4% of floors, no optimization EV". That close was
never tested where attention kernels actually fill the machine. This front rooflines
attention at serving-class shapes (bf16, D=128): prefill B=1 Hq=32 causal
S in {8192, 16384, 32768} (MHA and GQA H_kv=8), decode B=32 q_len=1 KV in {8192, 32768}.

**Verdict up front.** The B38 "no optimization EV" close HOLDS at serving shapes, but the
reason inverts. At micro shapes nothing was worth optimizing because everything was
overhead-bound. At serving shapes the best backend -- cuDNN SDPA via
`torch sdpa_kernel(CUDNN_ATTENTION)` -- is **already at the machine ceiling on both
rooflines** (prefill 98.8-108% of the B61 real compute SoL; decode 93.6-94.4% of HBM SoL),
so there is still no *kernel-engineering* EV. What appears instead is a large
**backend-routing** lever that is invisible at micro shapes: the serving-default kernels
(flashinfer auto/FA2-class) run at 21-24% prefill SoL -- 4.3-4.7x slower than the cuDNN
kernel at identical shapes -- and at 80-83% decode SoL (1.12-1.18x slower).

## The map

All cells: GPU 3 (shared, flock-gated, foreign-process check clean pre+post on every
cell), CUDA-graph-replay timing (median of 15, 3 rounds x 5 reps, arms interleaved
within each round), bf16, correctness cross-checked vs sdpa_cudnn before timing
(max_abs <= 0.0156 everywhere, bf16-scale). Same-session cuBLAS 8192^3 bf16 GEMM anchor
in every cell: 1953-2030 TFLOPS (<=4% session drift; SM clocks pinned 2070 MHz all cells).
Compute frame: %real-SoL vs 1900 TFLOPS (B61 cuBLAS asymptote). Memory frame: KV-read
GB/s vs 8000 GB/s nominal HBM. Stack: torch 2.12.0a0+nv26.05, cuDNN via torch SDPA,
flash_attn 2.7.4.post1, flashinfer 0.6.3, CUDA 13 / sm_103.

### Prefill, B=1 Hq=32 D=128 causal MHA (us | TFLOPS | %real-SoL)

| backend | S=8192 | S=16384 | S=32768 |
|---|---|---|---|
| **sdpa_cudnn** | **293.0 / 1877 / 98.8%** | **1098.2 / 2003 / 105.4%** | **4358.0 / 2018 / 106.2%** |
| flex_compiled (max-autotune) | 846.2 / 650 / 34.2% | 3031.2 / 726 / 38.2% | 11567.9 / 760 / 40.0% |
| flashinfer_prefill (auto) | 1377.8 / 399 / 21.0% | 5081.0 / 433 / 22.8% | 19417.1 / 453 / 23.8% |
| flash_attn2 (2.7.4) | 1385.0 / 397 / 20.9% | 5062.9 / 434 / 22.9% | 19339.6 / 455 / 23.9% |
| sdpa_flash | 1480.6 / 371 / 19.5% | 5430.6 / 405 / 21.3% | 20695.9 / 425 / 22.4% |
| sdpa_mem_efficient | 3339.7 / 165 / 8.7% | 12638.1 / 174 / 9.2% | 49149.8 / 179 / 9.4% |

(S=32768 flex measured in its own sub-cell with same-session cudnn 4272.6 us / 2059 TF
and anchor 2030 TF; all other S=32768 rows from the main sub-cell, anchor 1958 TF.)

### Prefill GQA, H_kv=8 (us | TFLOPS | %real-SoL)

| backend | S=8192 | S=16384 | S=32768 |
|---|---|---|---|
| **sdpa_cudnn** | **284.9 / 1930 / 101.6%** | **1071.8 / 2052 / 108.0%** | **4346.2 / 2024 / 106.5%** |
| flex_compiled | 842.2 / 653 / 34.4% | 3023.7 / 727 / 38.3% | (not run) |
| flashinfer_prefill | 1374.6 / 400 / 21.0% | 5034.4 / 437 / 23.0% | 19211.9 / 458 / 24.1% |
| flash_attn2 | 1395.1 / 394 / 20.7% | 5066.5 / 434 / 22.8% | 19345.0 / 455 / 23.9% |
| sdpa_flash | 1475.9 / 373 / 19.6% | 5439.6 / 404 / 21.3% | 20701.5 / 425 / 22.4% |
| sdpa_mem_efficient | no kernel (GQA unsupported) | no kernel | no kernel |

GQA timing == MHA timing per backend (within noise): prefill at these shapes is
compute-bound and causal FLOPs are independent of H_kv, so this is the expected result,
and it confirms no backend is accidentally penalized by `enable_gqa`.

### Decode, B=32 q_len=1 Hq=H_kv=32 D=128 (us | KV GB/s | %HBM-SoL)

| backend | KV=8192 | KV=32768 |
|---|---|---|
| **sdpa_cudnn** | **573.3 / 7491 / 93.6%** | **2274.4 / 7554 / 94.4%** |
| flex_compiled | 619.0 / 6938 / 86.7% | FAIL: max-autotune `NoValidChoicesError` (all choices fail to compile) |
| flashinfer_decode (paged, page=16) | 673.4 / 6378 / 79.7% | 2597.8 / 6613 / 82.7% |
| **proto: flashinfer cudnn_batch_decode** | (not run) | **2311 / 7433 / 92.9%** (bit-identical, graph-compatible) |
| flash_attn2 | 1240.4 / 3463 / 43.3% | 4924.8 / 3489 / 43.6% |
| sdpa_flash | 1328.5 / 3233 / 40.4% | 5276.1 / 3256 / 40.7% |
| sdpa_mem_efficient | 1595.4 / 2692 / 33.7% | 6317.4 / 2720 / 34.0% |

Decode frame disambiguation: at KV=32768 the cell does ~2.2 GFLOP against 17.2 GB of KV
reads -- compute is <0.1% of SoL, so the memory frame is the only meaningful one (compute
%SoL omitted). Note decode cuDNN sustains 7.49-7.55 TB/s -- well above the 5.64 TB/s the
memory_bandwidth_patterns lab achieves (B38), i.e. the attention decode kernel is one of
the strongest HBM streamers measured on this machine.

## Findings

1. **cuDNN SDPA is at ceiling everywhere it matters.** Prefill: 1877-2059 TFLOPS =
   98.8-108% of the B61 1900 TF reference and 96-103% of the same-session cuBLAS anchor.
   Decode: 93.6-94.4% HBM SoL. The causal-FLOP model (2*B*H*S^2*D) undercounts true work
   by only ~1/S, so >100% readings mean the cuDNN attention pipeline genuinely sustains
   slightly more tensor-core throughput than the 8192^3 cuBLAS GEMM -- the practical
   ceiling band on this part is ~1.95-2.06 PF and cuDNN attention sits inside it.
   There is no headroom for a custom attention kernel at serving shapes.

2. **The gap structure is backend routing, not kernel headroom.** At identical shapes
   the spread between best and serving-default is 4.3-4.7x for prefill (cudnn vs
   flashinfer-auto/FA2) and 1.12-1.18x for decode (cudnn vs flashinfer paged decode).
   flash_attn 2.7.4 and flashinfer-auto prefill are FA2-class kernels that plateau at
   ~400-458 TF on sm_103 -- they are not Blackwell-tuned. flex (max-autotune, B70 live)
   reaches 34-40% -- best torch-native non-cudnn option, still 2.6-2.9x off.

3. **Prototype LANDED (decode): route serving decode to cuDNN through flashinfer.**
   `flashinfer.cudnn_batch_decode_with_kv_cache` (paged KV page=16, block_tables,
   is_cuda_graph_compatible=True) runs at 2311 us / 7433 GB/s / 92.9% HBM SoL at
   KV=32768 -- 1.124x over flashinfer's default decode (2598 us / 82.7%), within 1.6% of
   the torch-sdpa-cudnn ceiling, and **bit-identical to the cuDNN reference (max_abs
   0.0)**. One workaround required: this cuDNN frontend demands **4-D seq-len tensors**
   (`(B,1,1,1)`, device-resident) -- the 1-D `(B,)` form in the flashinfer docstring is
   rejected with `KV sequence length tensor must be 4-dimensional`. Repro:
   `/tmp/frontAT2/proto_backend.py`, result `cells/proto_backend.json`.

4. **Named lever (prefill, not landed in <=3 attempts): wire serving prefill to the
   cuDNN attention engine.** The kernel exists (torch SDPA reaches it at 2 PF), but the
   serving-stack route is blocked in flashinfer 0.6.3 on this stack:
   `cudnn_batch_prefill_with_kv_cache` fails for the non-paged B=1 ragged layout with
   `CUDNN_ATTR_TENSOR_STRIDES ... BAD_PARAM` (1-D lens), then with 4-D lens it gets past
   shape validation but dies at plan finalization: `No valid execution plans built`
   (attempt evidence: `cells/proto_backend_attempt{1,2}.json`). Also measured:
   `backend="fa3"` ships sm90-only cubins (`no kernel image` on sm_103);
   `trtllm-gen`/`cutlass` are not valid single_prefill backends in 0.6.3. The lever is a
   flashinfer upgrade/fix (or a direct cudnn-frontend graph) -- EV up to ~4.3x on
   attention-prefill time for any flashinfer-routed serving stack at long context.

5. **Negatives worth keeping.** (a) flex max-autotune decode at KV=32768 fails to lower
   (`NoValidChoicesError`) -- flex cannot currently cover the long-KV decode cell at all.
   (b) flex `create_block_mask(_compile=True)` costs 82 s at S=16384 (one-build-per-
   process, B61 trap applies). (c) cuDNN graph-cache pollution: a failed
   `cudnn_batch_prefill` build poisons the `@graph_cache` for subsequent different-arg
   retries in the same process (`attn_scale with tensor and value cannot be set at the
   same time`) -- retry layout experiments in fresh processes.

6. **B38 verdict refined.** "No optimization EV" survives serving shapes for *kernel
   engineering*: the at-ceiling kernel already ships in cuDNN. But B38's micro-shape
   close could not see the backend-selection gap (at micro shapes all backends collapse
   into overhead); at serving shapes picking the wrong backend costs 4.3-4.7x on prefill.
   The actionable output is routing policy, not kernel work: **prefill -> torch
   SDPA/cuDNN (or fixed flashinfer-cudnn); decode -> flashinfer cudnn_batch_decode with
   the 4-D seq-len workaround.**

## Method / provenance

- Probe: `/tmp/frontAT2/probe_attn2.py` (continuation of frontAT's probe; standalone,
  committed lab pairs untouched). One (shape-set, backend-set) cell per process;
  per-cell JSON in `/tmp/frontAT2/cells/`; predecessor's single valid cell
  (`/tmp/frontAT/results/prefill_S8192.json`) reproduced within noise before reuse.
- GPU 3 shared with sibling PR1: fcntl flock on `/tmp/gpu3.lock` held only around the
  timed section, acquired via retry loop with attempt counter (acquired on attempt 1 in
  every cell); `nvidia-smi` compute-apps checked before and after every timed section --
  zero foreign processes in all reported cells.
- CUDA graphs: every reported arm timed as graph replay (3 inner iters per replay);
  graph-capture failures fall back to eager timing and are flagged -- none needed it.
- Thermal: arms interleaved within each round; same-session GEMM anchor per cell bounds
  drift at <=4%; SM clock 2070 MHz in every cell's pre/post probe.
- FA4 note: the repo's flashattention4 lab arms are compiled-flex / sdpa-flash / cudnn
  paths (all covered above); no FA3/FA4 kernel exists on this stack (flash_attn 2.7.4,
  no `flash_attn_interface`/`flash_attn.cute`), and the lab itself records that the
  published FA4 envelope does not reproduce locally.
- Raw cells (md5s in runbook/session log): prefill_S{8192,16384,32768}.json,
  prefill_S32768_flex.json, gqa_S{8192,16384,32768}.json, decode_S{8192,32768}.json,
  proto_backend.json (+ _attempt1/_attempt2 failure evidence).
