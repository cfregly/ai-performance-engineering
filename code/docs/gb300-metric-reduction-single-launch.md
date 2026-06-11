# GB300: single-launch fused metric_reduction (Front R, the B65/B50-named lever) + the 7 explicit-but-default stream fixes

**Verdict (Task 1): WIN, 1.112x pure-GPU warm — but the B50 estimate's mechanism was
WRONG.** The zeros-fill launch is gone (graphed fidelity count 2 → 1, the launch drop
is the proof) via a persistent replicated accumulator + atomic-ticket last-block fold:
graphed pure-GPU warm **9.048 → 8.134 us** (6/6 paired interleaved wins, zero band
overlap, gate >= 1.05x cleared at 1.112x), default frame **0.05738 → 0.05173 ms**
(1.109x, 6/6 paired wins, also non-overlapping — the frame win is mostly the dropped
`torch::zeros` HOST work, not the 1.25 us fill kernel). All 24 harness runs
verification PASS; numerics class unchanged and measurably TIGHTER (max-abs vs fp64
0.0029 vs incumbent's 0.0137 — fewer atomic partials at the smaller grid). The
estimate "4–5 us via float4, ~1.8–2x" is **REFUTED with mechanism**: the kernel is
latency/atomic-bound, not bandwidth-bound — float4 loads made it SLOWER (14.9 vs
7.9 us at the incumbent grid) and the single-launch tail costs ~5 us until the atomic
chains are replicated. Effective bandwidth 3.05–3.10 TB/s ≈ 38–39% HBM-SoL; the gap
to the 3.15 us single-read floor narrows 2.9x → 2.59x.

**Verdict (Task 2): all 7 explicit-but-default stream sites FIXED and validated.**
persistent_decode (2 sites) + memory_bandwidth_patterns (5 sites) now use
`getCurrentCUDAStream()` (B62/B65 style, eager-identical); B64 smoke (eager +
side-stream capture + kernel-count) PASSES on every site; both harness-runnable
targets PASS in/at band; `persistent_decode_cuda` is an `informational_demo` skip
(loud, by design — no benchmark pair exists for it).

Hardware: GB300 (cc 10.3, **152 SMs** — measured via the final grid, 304 = 2/SM),
GPU3 quiet (pre-rep foreign-process checks, flock'd active runs only, no idle holds),
pod `<gb300-pod>`. Pod repo md5-reconciled against local before editing (6/6
files matched HEAD `b77108e8`-era state) and after every sync (4/4 changed files
match). Evidence: `/tmp/frontR/` (r0–r13 tuples, `r3/r8/r9_*_ab.jsonl`,
`frontR_summary.json`, `ALL_DONE`). Harness flags: `--profile none --single-gpu`;
graphed mode via `AISP_BENCH_TIMING_MODE=graphed`.

## 1. What shipped (training_hotpath_kernels.cu)

One fused kernel per call, zero fill launches. `out` is `torch::empty`; blocks
atomic-fold into a **persistent, all-zero, K=8-replicated accumulator** (replica =
`blockIdx % 8`; scratch allocated+zeroed ONCE per device, outside capture — the warm
`setup()` call triggers it); the **last block to arrive** (atomic ticket,
threadFenceReduction ordering) sums the 8 replicas into `out` and re-zeroes
accumulator + ticket, restoring the invariant for the next launch — so CUDA-graph
replays are correct with no per-iteration state cost (verified: 4 replays x 3
captured iters, bit-stable vs fp64 reference within fp32 class).

- **Default variant `scalar`** (the measured winner): the incumbent's proven load/fold
  structure EXACTLY (column-per-thread coalesced loads, block row-striding, fp32 fmaf,
  3 atomicAdds/thread, no barriers in the load path) + the ticket tail. Grid
  **2 blocks/SM** (304), threads 256.
- Variant `vec4` (env `AISP_METRIC_REDUCTION_VARIANT=vec4`, kept for other
  parts/shapes): float4 loads, 4-deep software pipeline, smem block fold. REFUTED
  slower here (see §3).
- Kill switch `AISP_METRIC_REDUCTION_SINGLE_LAUNCH=0` restores the B50/B65 two-launch
  incumbent exactly (it remains in-file as the fallback for odd responders /
  misaligned inputs).
- Sweep knob `AISP_METRIC_REDUCTION_BLOCKS_PER_SM` (B48-style, default 2).
- NOTE: the persistent scratch is stream-serialized state; concurrent same-device
  calls from different streams are unsupported (lab + capture paths are single-stream).

## 2. Rep tables (final kernel, interleaved A/B, 6 reps/arm, all verification PASS)

### Graphed mode (decisive; capture-fidelity-audited; warm = exact whole-graph replay/N)

| rep | new warm us (1 kernel/iter) | incumbent warm us (2 kernels/iter) | new wins |
|---|---|---|---|
| 1 | 8.1299 | 9.0362 | yes |
| 2 | 8.1309 | 9.0410 | yes |
| 3 | 8.1427 | 9.0506 | yes |
| 4 | 8.1376 | 9.0566 | yes |
| 5 | 8.1286 | 9.0451 | yes |
| 6 | 8.1370 | 9.0550 | yes |
| **median** | **8.134** | **9.048** | **1.1124x** |

Bands do not overlap (new max 8.143 < inc min 9.036). The incumbent arm reproduces
the banked 9.02 us warm (B65 a2_02) and B62's 9.04 — same-session ruler confirmed.
Honest negative: **graphed COLD regresses 11.63 → 13.63 us** (L2-cleared replicas put
the atomic chains + ticket on DRAM latency at the smaller grid); the banked
comparison-grade figure is warm (B62/B65 caveat), and default mode (which L2-clears
every iter but pays the frame) still wins 6/6, so the cold regression is confined to
the cold-graph metric.

### Default mode (frame; band-bracketing)

| rep | new ms | incumbent ms |
|---|---|---|
| 1 | 0.05194 | 0.06028 |
| 2 | 0.05153 | 0.05849 |
| 3 | 0.05024 | 0.05572 |
| 4 | 0.04896 | 0.05726 |
| 5 | 0.05400 | 0.05673 |
| 6 | 0.05252 | 0.05750 |
| **median** | **0.05173** | **0.05738** (1.109x) |

Incumbent band 0.0557–0.0603 brackets the banked 0.0556–0.0578 (B50/B65; high tail
within the documented ~15% frame jitter). New-arm headline speedups 200–235x
(median ~222x vs banked 188.8x). The frame-mode delta (~5.6 us) exceeds the pure-GPU
delta (~0.9 us): the dropped `torch::zeros` saves host allocator+fill dispatch inside
the timed region — consistent with the B62 frame model (+3.2 us/launch + host
residual).

### Gates summary

| Gate | Result |
|---|---|
| default A/B >= 6 interleaved reps, band-bracketing | PASS (6/6 paired wins, banked band bracketed) |
| graphed pure-GPU warm >= 1.05x | **PASS, 1.112x** |
| capture-fidelity audit | PASS (eager 1 kernel == replayed 1; was 2) |
| verification (rtol/atol 1e-4, unweakened) | PASS 24/24 harness runs; direct contract check 8/8, min headroom 3.6x |
| shared-file regression metric_reduction_cuda | PASS, 0.05543 ms / 8.91x — inside banked 0.0497–0.0569 (B65) |

## 3. Mechanism: why the B50 estimate was wrong (and what actually paid)

kprobe = torch-profiler kernel mean over 200 eager calls; K = accumulator replicas.

| Step | Kernel us | Lesson |
|---|---|---|
| Incumbent (main + fill) | 7.891 + 1.246 = 9.14 | the baseline |
| First draft: float4 + smem fold + ticket, K=1, 4/SM | 14.90 | ncu: DRAM 18% vs incumbent 26%, both latency-bound; 5 grid-stride trips/thread starve the load pipeline |
| + 4-deep unrolled loads (8x512B in flight/warp) | 14.90 (unchanged) | load-MLP was NOT the binding constraint |
| scalar structure + ticket, K=1, 4/SM | 12.69 | the single-launch TAIL costs ~4.8 us: pre-ticket `__threadfence` waits out 592-deep same-address atomic chains |
| grid sweep, K=1 | best 10.43 (vec4 2/SM) | fewer blocks = shorter chains; tail shrinks with grid, loads don't care |
| **K=8 replicas** | scalar 2/SM **8.26**; 4/SM 8.33; vec4 1/SM 8.70 | chain depth /8 → tail ~0.4 us; THE fix |
| K=4 / K=16 / K=32 | 8.37 / 8.65 / 10.21 (scalar 2/SM) | K trades chain depth vs last-block fold+re-zero volume; K=8 is the GB300 optimum |

ncu (final winner, `--set full`, replay-inflated duration): 15.10 us under ncu, DRAM
throughput **21.1%** (1.67 TB/s under serialized replay), grid 304, 40 regs/thread,
occupancy 25.5% (grid-limited by design at 2/SM). Honest throughput basis (B50's own
basis, bytes/eager-duration): 25.166 MB / 8.252 us = **3.05 TB/s**; on the graphed
warm figure 25.166/8.134 = **3.09 TB/s ≈ 38.7% HBM-SoL** vs the 3.15 us single-read
floor → remaining gap **2.59x** (was 2.9x). The residual is NOT addressable by load
vectorization (refuted above): both kernels sit at ~67%/60% long-scoreboard stalls
with ~26 achieved warps/SM — a 12288-row x 1KB reduction at this size simply doesn't
generate enough latency-hiding parallelism before it ends; the floor would need a
shape change (CONTRACT-level) or a persistent-CTA + programmatic-dependent-launch
style overlap with the surrounding graph.

## 4. Task 2: the 7 explicit-but-default stream sites (B45-class #4, from B65 §3)

Fix: `at::cuda::/c10::cuda::getDefaultCUDAStream()` → `getCurrentCUDAStream()`
(B62/B65 precedent comment at each lab's first site; eager-identical, current ==
default outside capture).

| Site | Harness validation (default mode, 1 run) | B64 smoke (eager + side-stream capture + kernel-count) |
|---|---|---|
| `persistent_decode/persistent_decode_ext.cu:94` | `persistent_decode:persistent_decode_cuda` → **informational_demo skip** (loud, by design: optimized-only demo, no pair — recorded, not a refusal of the fix) | **PASS** (1 kernel/replay, finite output) |
| `persistent_decode/tma_extension.py:84` | `persistent_decode:native_tma_prefill_decode` → **0.7126 ms / 2.80x, verification PASS**; no banked GB300 band → **NEW REFERENCE** | **PASS** (1 kernel/replay, dst==src) |
| `memory_bandwidth_patterns/bandwidth_patterns_kernels.cu:187,205,232,245,261` (copy_scalar, copy_vectorized, copy_async_double_buffered, transpose_naive, transpose_tiled) | `memory_bandwidth_patterns:bandwidth_patterns` → **0.07213 ms / 3.05x, verification PASS** vs banked 0.0704 ms / 3.12x (gb300-sol-roofline-labs2.md, run 20260611_031322; +2.5%, inside frame jitter; baseline 0.2203 vs 0.2192 likewise) | **PASS x5** (1 kernel/replay each, exact-equal outputs) |

No graphed-harness requirement attempted beyond the smokes (the smoke IS the capture
proof prescribed by B65 §3). Repo grep post-fix: zero remaining
`getDefaultCUDAStream` call sites under `labs/` (comments only).

## 5. Files touched

- `labs/training_hotpath/training_hotpath_kernels.cu` — md5 `750507c2…` →
  `4d5342af0c31ecd44fb96b73454e61ac` (single-launch scalar+vec4 kernels, replicated
  scratch, host dispatch + knobs; incumbent kernel retained as fallback/kill-switch).
- `labs/persistent_decode/persistent_decode_ext.cu` → `31f71571…` (1-line + comment).
- `labs/persistent_decode/tma_extension.py` → `8b0a2110…` (1-line + comment).
- `labs/memory_bandwidth_patterns/bandwidth_patterns_kernels.cu` → `18d6c3f8…`
  (5 sites + one comment).
- `docs/gb300-metric-reduction-single-launch.md` — this report.

All four synced local == pod (md5-verified post-sync). docs/gb300-dual-figure-bank.md
§3's "remaining unfixed" list is now fully addressed.

## 6. Named next lever

**metric_reduction_vectorized is now ~overhead/latency-floor-bound at 8.13 us warm;
the remaining 2.59x to the 3.15 us HBM floor is NOT a load-side micro-opt** (float4 +
unroll refuted; tail already minimized at K=8). Highest-EV next: (1) **fold the
metric kernel into the harness-level graph with its consumer** so the 22 us frame
bracket amortizes across the lab (frame share is still ~73%) — i.e., the B62-named
harness-bracket lever, not a kernel lever; (2) if a kernel lever is demanded,
**cooperative one-wave persistent launch with cluster-wide reduction in DSMEM**
(skips the L2 atomic chains entirely; est. tail ~0; but the load phase floor ~7.9 us
stands, so EV is capped at ~3%— formulate before building). (3) For the cold-graph
metric only: warm the replica lines inside the graph (a 24 KB prefetch) if anyone
ever banks cold figures.
