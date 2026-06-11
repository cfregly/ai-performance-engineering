# GB300: Vectorized fixed-capacity dispatch for `labs/moe_cuda:router_vectorized`

**Verdict: WIN — 16.5x over the previous optimized arm (8.29 ms -> 0.50 ms), verify PASS x3.**
The `GroupedTopKMoE.forward` 32-iteration Python expert loop with data-dependent boolean
masking (`flat_tokens[mask]`) made the benchmark's own `torch.cuda.CUDAGraph` capture in
`setup()` throw (every `mask` index triggers `aten::nonzero` -> DtoH sync, illegal during
capture), so `self.graph` silently fell back to `None` and the measured loop ran the eager
32-expert loop with 96 host syncs per iteration. Replacing the loop with the proven B40
sortless fixed-capacity dense dispatch makes every op static-shape; the existing capture
plumbing (unchanged) now succeeds and the whole forward replays as ONE CUDA graph.

- Date: 2026-06-11 | Pod: `<gb300-pod>` (<namespace>), GPU 2, GB300 (sm_103)
- Base: 2f7e30f9 + uncommitted GB300 fixes; only file changed:
  `labs/moe_cuda/optimized_router_vectorized.py` (optimized arm only; baseline untouched)
- Precedent: `docs/gb300-moe-router-override.md` (B40, same lever in
  labs/moe_optimization_journey, ~13x)

## Before / after (harness: `python -m cli.aisp bench run --targets labs/moe_cuda:router_vectorized --profile none --single-gpu`, GPU 2)

| run | baseline_ms | optimized_ms | speedup vs baseline | verify |
|---|---|---|---|---|
| BEFORE rep1 (20260611_062506) | 91.523 | 8.2889 | 11.04x | PASS |
| BEFORE rep2 (062637) | 90.431 | 8.4782 | 10.67x | PASS |
| mid (design B, naive cumsum) x3 (063046/063116/063143) | 88.7-91.7 | 2.559-2.563 | 34.6-35.9x | PASS x3 |
| AFTER rep1 (063710) | 135.581* | 0.5020 | 270.06x | PASS |
| AFTER rep2 (063741) | 97.577 | 0.5017 | 194.51x | PASS |
| AFTER rep3 (063810) | 96.241 | 0.5000 | 192.48x | PASS |
| AFTER rep4 (064409, exact shipped bytes after whitespace normalization) | 88.855 | 0.5007 | 177.46x | PASS |

*baseline (66 GB per-token weight-gather path) is noisy 88-136 ms across reps; the gate
metric below uses the stable optimized-arm times.

**Gate: new optimized arm vs old optimized arm = 8.289 / 0.501 = 16.5x** (required
>= 1.05x). Verification passed on all 3 AFTER reps (rtol 0.1, atol 1.0); numeric
cross-check vs the original grouped-loop forward was **bit-exact (max_abs_diff = 0.0)**
on 5 random bf16 inputs, both eager and via graph replay.

## Capture-collapse evidence (torch.profiler, 10 measured iters, benchmark class as-is)

| metric (per iteration) | BEFORE (old file) | AFTER (shipped) |
|---|---|---|
| graph state in setup() | capture FAILED -> eager | captured |
| cudaGraphLaunch | 0 | **1** |
| cudaLaunchKernel | 679 | **0** |
| aten::nonzero events | 288 (96 calls) | **0** |
| Memcpy DtoH (sync) | 96 | **0** |
| GPU kernel events | 840 | 33 (inside the one graph) |
| standalone wall ms/iter | 11.32 | **0.447** |

## Design that won (B40 Design B, adapted): sortless fixed-capacity dense dispatch

1. Rank-within-expert via one-hot cumsum — built directly in `[E, N]` layout
   (`flat_expert_ids.unsqueeze(0) == arange(E).unsqueeze(1)`) so the scan runs along the
   contiguous inner dim. The first cut used `one_hot(ids).cumsum(0)` ([N, E] outer-dim
   scan = only E=32 parallel lanes): that single int64 scan kernel was **2048 us of a
   2507 us replay**. The transposed scan is 24 us -> harness 2.56 ms dropped to 0.50 ms.
2. `index_copy_` routed tokens into a `[E * capacity (+1 trash row), H]` slot buffer;
   slot indices unique by construction; overflow (never happens with calibrated
   capacity) clamps to the trash row and is masked to zero on the way out.
3. Two batched GEMMs over all experts at once — `torch.baddbmm` folds the per-expert
   bias into the GEMM (the standalone broadcast adds were 151 us each); zero rows inert.
4. Gather back by the same slot indices (restores token order for free), scale by
   `prob * valid`, then the same `index_add_` accumulate as before.

Capacity is calibrated once eagerly in `setup()` BEFORE capture (`calibrate_capacity`:
bincount -> 2x observed max tokens/expert, rounded up to a multiple of 64, clamped to
N) -> **640** here (4096 tokens x top-2 over 32 experts, max load ~300). setup() also
warms up cuBLAS with 3 eager forwards before capture; the only host sync of the
optimized path lives in calibration.

## Exact diff (labs/moe_cuda/optimized_router_vectorized.py only)

- `GroupedTopKMoE.__init__`: added `self.capacity: Optional[int] = None`.
- Added `GroupedTopKMoE.calibrate_capacity()` (`@torch.no_grad`, eager-only host sync).
- `GroupedTopKMoE.forward`: replaced the `for expert in range(num_experts)` boolean-mask
  loop (nonzero/DtoH per expert) with the static-shape dispatch above. No nonzero, no
  `.item()`, no host sync in the hot path.
- `VectorizedRouterBenchmark.setup()`: added capacity calibration + 3 eager warmup
  forwards + sync immediately before the (unchanged) CUDA-graph capture block.
- Everything else (capture/replay plumbing, benchmark_fn, verification payload,
  tolerances, batch sizes, baseline file) untouched.

## Remaining headroom / next lever

Replay device time is 456 us vs ~250 us floor for the present kernel set: the top cost
is now 3 `direct_copy` kernels (130 us total — the two `baddbmm` bias materializations
at [32, 640, 2048] bf16 plus the `flat_tokens` expand-reshape), then the 2 nvjet GEMMs
(124 us, near-roofline for 2x86 GFLOP @ bf16) and eager GELU (44 us). Named next lever:
torch.compile (mode=default, sm_103 tcgen05 guard) over the new static-shape forward to
fuse bias+GELU into the GEMM epilogue and elide the expand copy — est. 456 -> ~300 us
device, i.e. harness ~0.35 ms (+~1.4x); riskier because the harness's manual
`torch.cuda.graph` capture must then wrap a compiled callable (B40 showed this works via
reduce-overhead, but here the benchmark owns the graph, so capture-vs-compile interplay
needs care).
