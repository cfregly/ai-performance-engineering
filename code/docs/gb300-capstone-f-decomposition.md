# GB300 capstone tcgen05: F decomposed — the "fixed cost" was the t2r epilogue atom (Front Y)

**Date:** 2026-06-11 | **Pod:** <gb300-pod> (GPU 3) | **Base:** 2f7e30f9 (B53 state) + uncommitted GB300 fixes
**Verdict: WIN — 1.49x kernel (23.8 -> 16.0us at FP16 2048^3), bit-identical (torch.equal, CTA1 + CTA2), all 3 harness targets verify PASS x3.**

B52 named the lever: ~15us of FIXED per-CTA cost (F), 62-65% of the 23.9-24.2us
capstone kernel, with every scaling lever falsified (depth B48, bK B48,
B-multicast B53, wave-quant/stream-K B52). This front decomposed F with
in-kernel `%globaltimer` stamps before touching anything. The result kills the
folk theory list (launch, TMEM alloc, prologue, barrier init — all <0.7us
combined noise): **F is the epilogue, and 52% of it is one copy-atom choice.**
`SM100_TMEM_LOAD_32dp32b1x` issues 256 `tcgen05.ld` per thread, and ptxas
lowers EVERY one into a LEPC + CALL.ABS.NOINC + WARPSYNC convergence-helper
subroutine call. Widening the atom to `32dp32b32x` (8 loads/thread) deletes
7.2us of the kernel for one changed line, bit-identically.

## 1. F budget table (probe stamps, per-CTA medians, full grid 2048^3, 128 CTAs)

| # | Region (stamp pair) | incumbent 1x | shipped 32x | B52 candidate verdict |
|---|---|---|---|---|
| 1 | entry -> setup done (cute layout/partition init) | 0.22us | ~same | dead |
| 2 | tcgen05.alloc + sync | 0.19-0.22us | ~same | dead (alloc-lock lever pointless) |
| 3 | mbarrier init + sync | 0.13us | ~same | dead |
| 4 | barinit -> first TMA issue (producer w1) | 0.19us | ~same | dead |
| 5 | barinit -> first stage arrives (w0) | 0.64us | ~same | prologue fill ~= physics |
| 6 | mainloop k-linear (nk=32) | 9.38-9.47us | 9.44us | not F (u ~= 0.29us/k-tile in probe) |
| 7 | **epilogue t2r TMEM_LOAD** | **7.65-7.81us** | **0.42us** | **THE lever — fixed here** |
| 8 | **epilogue fp16 cvt + STG** | **3.68-3.78us** | **3.36us** | next lever (sec. 6) |
| 9 | final sync + TMEM free | 0.22us | ~same | dead |
|   | **in-kernel total** | **22.46us** | **14.72us** | probe build incl. stamp overhead |

Cross-checks that close the decomposition:
- slot-0 spread across all 128 CTAs: 0.22us — cluster-launch CTA rollout is NOT a cost;
  warp-parallel epilogue ALREADY in place (all 4 warps t2r+store); stores ALREADY
  vectorized (STG.E.128). Three more B52 candidates dead on arrival.
- Single-CTA (M=128,N=256,K=2048): total 20.93us, epilogue 7.78+3.42us — F is genuinely
  per-CTA, not a grid/contention effect.
- Full-grid K=256: total 14.21us, mainloop 1.31us, epilogue 7.87+3.58us unchanged —
  confirms F's K-independence (B52's "K=256 is 3.6%-SoL F-bound" exactly).
- Stamp F (~13.3us in-kernel fixed) + ~1.2us stamp-vs-event launch gap ~= B52's K-slope
  F = 14.9-15.5us. Budget closes; the epilogue is 77% of F.

## 2. Root cause (SASS)

`cuobjdump -sass` on the incumbent: 512 LDTM paired 1:1 with 512 LEPC,
512 CALL.ABS.NOINC and 520 WARPSYNC.ALL across the two kernels — every
`tcgen05.ld.sync.aligned.32x32b.x1.b32` is wrapped in a per-load
convergence-helper call (~60 cycles each). 256 calls/thread x 4 warps
serialize to 7.7us per CTA. The x32 form moves 32 columns per instruction:
8 loads/thread, helper amortized 32x, t2r collapses to 0.42us.

## 3. Width sweep (probe build, bit-checked)

| atom | t2r | cvt+store | in-kernel total | torch.equal vs 1x |
|---|---|---|---|---|
| 32dp32b1x (incumbent) | 7.65us | 3.68us | 22.46us | — |
| 32dp32b16x | 0.51us | 3.36us | 14.75us | PASS |
| **32dp32b32x (shipped)** | **0.42us** | **3.36us** | **14.72us** | **PASS** |
| 32dp32b128x | 0.99us | 4.03us | 15.97us | PASS (128-output asm serializes writeback) |

Bit-identity is structural, not luck: the atom only changes how many columns one
instruction moves; each thread still owns the same (row, all-256-cols) fragment,
so the per-element RNE convert and the STG.E.128 mapping are untouched.

## 4. Kernel A/B + ncu (winner)

Interleaved process-level A/B, 7 reps each, CUDA-graph x32 calls/rep, GPU 3:

- A (1x incumbent): med 23.81-23.90us — matches the B52 23.9-24.2us incumbent
- B (32x shipped): med 15.94-16.10us
- **1.49x kernel-level, zero overlap across 14 measurements.**
- FP16-SoL: ~19% -> ~28.5% (17.18 GFLOP / 16.0us = 1.07 PFLOP/s).

ncu (--set full, skip 5, count 2, locked clocks 1.90GHz): duration 19.1us;
`sm__pipe_tensor_cycles_active` 37.9% of elapsed vs 20.4% on the incumbent
lineage — the 1.49x carried straight into tensor-pipe duty cycle.
`dram__bytes_write` ~0 (D drains via L2). L2 write sectors 524,288 x 32B = 16MB
for 8MB of D = **50% write-sector efficiency** (row-strided stores) — sec. 6.

## 5. Verification + harness (gate: >=1.05x + verification PASS)

Bit-identity lineage: shipped repo build `torch.equal` vs incumbent-1x output —
**CTA1 PASS, CTA2 PASS** (B44 bit lineage holds; epilogue change is exact).
allclose vs fp32 torch.matmul ref (2e-2): PASS both variants.

Harness x3, all three shared-file targets (failed_verification=0 in all 9 runs):

| target | run1 | run2 | run3 |
|---|---|---|---|
| blackwell_matmul_tcgen05 (median / speedup) | 0.169ms / 90.4x | 0.103ms / 145.2x | 0.102ms / 147.5x |
| cluster_gemm_tcgen05 | 0.0896ms / 1.07x | 0.0880ms / 1.05x | 0.0816ms / 1.16x |
| cluster_gemm_tcgen05_cta2 | 0.0889ms / 1.15x | 0.0817ms / 1.13x | 0.0844ms / 1.30x |

Honest read: these wrappers are host-dispatch-bound (~85-100us/call), so the
7.9us kernel cut is inside host noise on the blackwell wrapper (run1 additionally
carries the post-edit extension rebuild; runs 2-3 settle at the B52-era
end-to-end level). The fullstack_cluster pair runs BOTH its baseline (cta1) and
optimized (cta2) on this same edited kernel and both moved ~10-18% end-to-end vs
the same-day B53 suite logs (0.100-0.104 -> 0.082-0.097ms). The bankable number
is the kernel-level 1.49x (graph A/B + stamps + ncu agree).

## 6. Diff (entire change) + md5

`code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu`
(b61fb99737f260ee59fc756a2b089eb7 -> 385d820f0b06239077a63a3ed90ca81a):

```diff
+// TMEM->register epilogue atom (B54/Y: F-decomposition). [comment block]
+#ifndef AISP_TCGEN05_T2R_ATOM
+#define AISP_TCGEN05_T2R_ATOM SM100_TMEM_LOAD_32dp32b32x
+#endif
...
-  auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
+  auto tiled_t2r_copy = make_tmem_copy(AISP_TCGEN05_T2R_ATOM{}, tCtAcc);
```

Probe artifacts (never the shipped path): /tmp/frontY/probe.cu (+`AISP_TCGEN05_PROBE`
stamps), t2r_one.log (width sweep), ab_run.log (interleaved A/B), ncu_w32.ncu-rep,
harness_after.log, out_w*.pt (bit refs).

## 7. Named next lever

Remaining F after this fix: the **3.36us cvt+store** (now ~21% of the 16us
kernel). It is quantified write-sector-bound: each warp's STG.E.128 spans 32
DIFFERENT D rows (t2r maps lane=row), so every 16B store occupies its own 32B
L2 sector — 2x write amplification measured (16MB sectors / 8MB data). Lever: a
smem-staged transpose epilogue (r2s, then warp-coalesced 128B s2g rows),
CUTLASS-style, reusing the (drained) A/B ring smem — worth ~1.5-2us (~10%).
Values and destinations unchanged => bit-identity preservable. After that the
kernel is mainloop-bound (9.4us, 59%, TMA delivery efficiency — B48/D5
territory) and F is spent.

## 8. Y2: smem-staged transpose epilogue — the write-sector half, harvested (1SM); honest 2SM gate-out

**Date:** 2026-06-11 | **Front:** Y2 | **Base:** B55 (385d820f0b06239077a63a3ed90ca81a)
**Verdict: WIN on the 1SM path — 1.054x kernel (16.01 -> 15.19us med-of-meds at FP16
2048^3), L2 write sectors exactly halved to the 100%-efficiency floor, bit-identical
(torch.equal, CTA1 + CTA2), all 3 harness targets verify PASS x3. The 2SM (CTA2) entry
is constexpr-gated to the B55 direct store: staging halves its sectors identically but
REGRESSES the kernel +0.27us — measured, mechanism below.**

Sec. 7 named the lever: B55's cvt+store is write-sector-bound (each warp's
STG.E.128 spans 32 different D rows; 524,288 x 32B = 16MB of L2 write sectors
for 8MB of D). The fix stages the converted fp16 tile through the drained A/B
smem ring: each thread r2s-stores its 512B row as 32 16B chunks, XOR-swizzled
(`chunk ^ (tid&7)`) so both the STS.128 and the LDS.128 are bank-conflict-free;
after one `__syncthreads()`, each warp drains whole rows — lane l stores chunk
l of ONE row, so every warp store instruction covers 512 contiguous bytes
(16 fully-written 32B sectors). The thread->row mapping is never assumed: each
thread stashes its gmem row pointer (`&tDgD(0)`) in a 1KB smem table and phase
2 stores through it. Ring reuse is safe: the epilogue runs strictly after the
final-k-tile mma-barrier drain (the tcgen05 commit orders ALL prior UMMA smem
reads, incl. the 2SM peer's, before the barrier flips) and every TMA fill was
consumed in-loop before that commit; the 65KB stage stops 127KB short of the
mbarrier words. Macro `AISP_TCGEN05_SMEM_EPILOGUE` (default 1, 1SM-only via
`if constexpr (kMmaCtas == 1)`); `=0` rebuilds the exact B55 store everywhere.

### Component (probe2 stamps, per-CTA medians, full grid 2048^3, 128 CTAs, CTA1)

| region | sm=0 (B55 path) | sm=1 (staged) |
|---|---|---|
| mainloop k-linear | 9.38us | 9.41us |
| t2r TMEM_LOAD | 0.42us | 0.42us |
| **cvt+store** | **3.39us** | **2.43us (-0.96us)** |
| sync + TMEM free | 0.26us | 0.19us |
| in-kernel total | 14.72us | 13.70us |

Per-warp slot-8 medians all 2.43us — the 32-pass s2g loop is perfectly
load-balanced across the 4 warps.

### Kernel A/B (interleaved process-level, CUDA-graph x32 calls/rep, GPU 3)

- CTA1, 7 reps each: A (direct) med 15.93-16.09us; B (staged) med
  15.12-15.26us — **1.054x, zero overlap across 14 measurements.**
- CTA2, 3 reps each: A med 15.45-15.53us; B med 15.72-15.78us — **staged
  REGRESSES +0.27us, zero overlap.** Shipped CTA2 keeps the direct store
  (post-gate sanity x3: med 15.41-15.51us, back at its control band; CTA1
  post-gate x3: med 15.17-15.21us, win retained).
- Shipped kernel ~30.0% FP16-SoL (17.18 GFLOP / 15.19us = 1.13 PFLOP/s), up
  from ~28.5%.

### ncu (--set full, skip 5, count 2; same session, both variants, both entries)

| metric | CTA1 sm=0 | CTA1 sm=1 | CTA2 sm=0 | CTA2 sm=1 (not shipped) |
|---|---|---|---|---|
| global store sectors | 524,288 (50% eff) | **262,144 (100%)** | 524,288 (50%) | 262,144 (100%) |
| global_st instructions | 16,384 | 16,384 | 16,384 | 16,384 |
| shared_st / shared_ld | 256 / 896 | 17,152 / 33,664 | 256 / 1,088 | 17,152 / 33,856 |
| duration under ncu | 18.8-19.2us | 18.3-18.4us | 18.8-18.9us | 18.9-19.1us |

SASS: staged store phase = 64 STS.128 + 64 LDS.128 + 64 LDS.64 (row-pointer
table) + 64 ST.E.128 across the two kernel instantiations; control = 64
STG.E.128 direct.

### Why CTA1 wins 0.96us but CTA2 loses 0.27us (honest accounting)

The projection (~1.5-2us) ignored two costs that don't vanish: the gmem write
itself has a ~1.2us floor at 100% efficiency (8MB through L2, single wave) and
the smem round-trip costs ~0.55us (64KB STS + 64KB LDS at 128B/cycle/SM) plus
a CTA-wide sync. Floor for a staged cvt+store is ~2.1-2.3us; measured 2.43us.
On CTA1 the sector waste WAS time on the critical path: removing 8MB of
half-filled sectors bought 0.96us against the 0.55us round-trip. On CTA2 the
identical sector halving bought nothing: its control already ran 0.55us faster
than CTA1's at identical sector counts (524,288), i.e. the 2SM pair's store
phase was not L2-write-time-bound, so the round-trip is a pure add (+0.27us).
The waste is recoverable only where it is actually the binding constraint.

### Verification + harness (gate: verification PASS)

- Final gated build `torch.equal` vs incumbent-1x bit-lineage refs: **CTA1
  PASS, CTA2 PASS** (same RNE convert per element; CTA1 changes only store
  order, CTA2 is byte-identical code path to B55).
- allclose vs fp32 torch.matmul ref (2e-2): PASS both.
- Harness x3 on the FINAL gated file, all three shared-file targets
  (failed_verification=0 in all 9 runs):

| target | run1 | run2 | run3 |
|---|---|---|---|
| blackwell_matmul_tcgen05 (median / speedup) | 0.163ms / 96.4x | 0.108ms / 144.8x | 0.104ms / 151.2x |
| cluster_gemm_tcgen05 | 0.0887ms / 1.15x | 0.0972ms / 1.04x | 0.0887ms / 1.12x |
| cluster_gemm_tcgen05_cta2 | 0.0881ms / 1.18x | 0.0908ms / 1.16x | 0.0868ms / 1.18x |

These wrappers stay host-dispatch-bound (~85-100us/call), so the ~0.8us kernel
cut sits inside host noise end-to-end; the bankable number is the kernel-level
graph A/B + stamps + ncu agreement above. run1 blackwell carries the post-edit
extension rebuild (same pattern as B55's run1).

(An earlier harness x3 on the staged-everywhere build also passed 9/9 —
/tmp/frontY2/harness_staged_all.log — the gate-out is a perf decision, not a
correctness one.)

### Diff (entire change) + md5

`code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu`
(385d820f0b06239077a63a3ed90ca81a -> c7e412cc81c2905c36464ae4bb681219):

```diff
+// smem-staged transpose store (B55 follow-up, Y2). [rationale + measured
+// numbers comment block, incl. the 2SM gate-out]
+#ifndef AISP_TCGEN05_SMEM_EPILOGUE
+#define AISP_TCGEN05_SMEM_EPILOGUE 1
+#endif
...
   for (int i = 0; i < size(tDrD); ++i) {
     tDrD(i) = static_cast<DType>(alpha * tDrAcc(i));
   }
-  copy(tDrD, tDgD);
+#if AISP_TCGEN05_SMEM_EPILOGUE
+  if constexpr (Variant::kMmaCtas == 1) {
+    // r2s swizzled 16B chunks into the drained ring + row-pointer stash,
+    // __syncthreads, then per-warp whole-row s2g (lane l = chunk l)
+    ... (35-line block; see kernel source)
+  } else {
+    copy(tDrD, tDgD);
+  }
+#else
+  copy(tDrD, tDgD);
+#endif
```

Y2 artifacts (never the shipped path): /tmp/frontY2/probe2.cu (+stamps,
regenerated from the shipped file by make_probe2.py), ab_run.log / ab_cta2.log
/ final_check.log (interleaved A/B), ncu_sm{0,1}.ncu-rep /
ncu_cta2_sm{0,1}.ncu-rep, harness_staged_all.log / harness_after.log,
capstone_kernels_tcgen05.cu.b55.bak (pristine B55).

## 9. Named next lever (post-Y2)

The CTA1 epilogue residual is ~0.3-0.5us of SM-side staging slack: a TMA
tensor store (`SM90_TMA_STORE` atom over mD; smem stage in the descriptor's
swizzle mode; one `cp.async.bulk.tensor.2d` per CTA + `tma_store_wait` before
exit) would delete the per-thread LDS+ST issue chain and make the drain async
— but it is bounded by the same ~1.2us L2-write floor, so it is NOT
breakthrough-shaped. The bigger fish is now unambiguous: the **9.4us mainloop**
(62% of the 15.2us kernel) against a ~5-6us tensor-pipe SoL at this grid.
B48/D5 already localized the residual to TMA delivery efficiency (bK64/SW128
box shape), and B52 falsified wave-quant/stream-K — so the next
breakthrough-shaped levers are mainloop-side: L2-resident B reuse across
M-tiles (persistent CTAs + tile scheduler so the 16x-duplicated B panel pulls
become L2 hits), or the 2SM ring deepened to cover TMA latency at the pair
level. The CTA2-vs-CTA1 store finding above is a free clue for that work: the
2SM pair's phase-staggering already hides store latency — the same staggering
argument applies to its TMA fetch stream.

# Y3: persistent CTAs / tile scheduler for L2 B-reuse — premise FALSIFIED in pre-check; mainloop diagnosed tensor-pipe-paced at the REAL (measured) fp16 ceiling; early-fill prologue shipped (CTA1 15.20 -> 14.98us)

Front Y3 owned the final named capstone lever: persistent CTAs + tile
scheduler so M-tiles sharing a B k-panel run consecutively/co-resident and
the 16x-duplicated B pulls become L2 hits. The mandated pre-check killed the
premise with data before a line of scheduler code was written, and the
mainloop-bound analysis that replaced it re-grounds the whole "~5-6us
tensor-pipe SoL floor" framing: the 9.4us mainloop is ALREADY ~87% of the
empirically achievable tensor rate on this part. The residual prologue slack
was harvested with an early-fill reorder (AISP_TCGEN05_EARLY_FILL).

## 1. Pre-check: B (and A, and D) are ALREADY fully L2-resident

ncu --set full on the B58 incumbent (CTA1, 2048^3, launches 6-7,
/tmp/frontY3/ncu_full_cta1.ncu-rep, kernel 18.2-18.4us under replay):

| metric | value | reading |
|---|---|---|
| dram__bytes_read.sum | 16.82 MB | = A(8MB) + B(8MB) cold fill + ~0.8MB metadata: the EXACT compulsory floor |
| dram__bytes_write.sum | 0 B | even D's 8MB stays in L2 past kernel end |
| derived lts lookup-miss-promoted | 0% | no capacity/conflict misses at all |
| l1tex xbar->L1 TMA-load bytes | 201.33 MB | = 128 CTAs x 32 k-tiles x 48KB, the full 16x-B / 8x-A duplication |
| same, rate | 11.0 TB/s = 29.4% of path peak (~37.5 TB/s) | aggregate delivery path nowhere near saturated |
| lts srcunit_tex read sectors | 3.26M = ~104MB | hardware in-flight merge already collapses ~1.93x of the duplication before L2 banks see it (B53 confirmed) |

Verdict on the named lever: with DRAM reads at the compulsory floor and 0%
misses, NO tile schedule, persistence shape, or grid swizzle can save a
single DRAM byte at this shape — the 126MB L2 swallows the entire 24MB
working set on first touch. The duplication is real but is absorbed at
11 TB/s on a 37.5 TB/s path. Banked WITHOUT implementation, per the
pre-check design. (Corollary: the B48/X CLUSTER_M=2 multicast non-win is now
explained — it could only halve L2 lookups the in-flight merge was already
halving.)

## 2. What actually binds the 9.4us mainloop: per-k-tile stamp split (probe3)

/tmp/frontY3/{make_probe3.py,probe3.cu,p3_driver.py}: %globaltimer stamps
per k-tile on consumer warp 0 (t0 wait-entry / t1 data-ready / t2 MMAs
issued+committed) and producer warp 1 (g0 gate-entry / g1 gate-clear / g2
TMA issued), 128 CTAs, one un-replayed launch (probed mainloop 9.98us vs
9.4us native, ~0.6us stamp overhead):

| segment (med across 128 CTAs) | value | reading |
|---|---|---|
| w0 sum tma_wait | 2.30us | ALL of it is fill-phase: tile0 waits ~860ns, steady-state wait ~0ns |
| w0 sum mma_issue+commit | 5.12us | consumer issue path, ~160ns/tile |
| w0 inter-iteration gap | ~2.31us | loop bookkeeping + stamps (~74ns/tile) |
| w1 sum gate_wait (ring full) | 6.14us | producer always AHEAD: delivery never the steady-state laggard |
| w1 sum tma_issue | 1.02us | TMA issue cost trivial — producer-issue-bound falsified |
| TMA issue -> barrier-flip latency | 768ns (p10 512 / p90 768) | fully hidden by the 4-stage ring in steady state |
| fill-phase arrival spacing | 256-288ns / 48KB stage | per-SM delivery path streams ~178GB/s after first arrival |
| steady-state per-tile period | ~290ns (timer granularity 32ns) | vs 256ns/tile pure-pipe time (below) |
| drain after last issue | 0.26us | pipe backlog at loop end ~1 tile: issue and execution are paced together |

## 3. The "5-6us SoL floor" is falsified: the real fp16 ceiling is ~1.9 PF

cuBLAS (torch.matmul, fp16/fp32acc) on the same GPU: 2048^3 = 11.47us
(1.50 PF); 4096^3 = 1.78 PF; 8192^3 = 1.87 PF (asymptote); 16384^3 = 1.71 PF.
The 3.77 PF chip rate implied by "30.0% SoL" (42ns per 128x256x16 UMMA) is
not approached by ANY dense fp16 kernel on this part. The practical per-SM
UMMA rate is ~64ns per 128x256x16 (16.4 TF/SM, 2.49 PF chip if all 152 SMs
worked): our mainloop moves 134.2 MFLOP/CTA in 9.4us = 14.3 TF/SM = 87% of
that — ABOVE cuBLAS's own best per-SM rate (1.87PF/152 = 12.3 TF/SM).
Steady state is tensor-pipe-paced (consumer never starves; ~290ns period vs
~256ns pipe time per 4-UMMA tile); the only recoverable mainloop slack is
the ~2.3us fill-phase ramp (768ns first arrival + ~270ns/stage stream-in),
of which only the issue START time is addressable — the ramp rate is a
per-SM delivery property. Kernel floor under bit-identity constraints
(no split-K; 128 tiles = 128 SMs): ~8.2us pipe + ~0.8us fill + 0.26us drain
+ ~2.8us epilogue ~= 12.1us. A <=12us breakthrough is NOT reachable at this
shape without breaking accumulation-order bit-identity (B52's falsified
territory); after the early-fill ship below (14.98us) the honest remaining
envelope is ~14.2-14.5us, all of it epilogue-side, none of it mainloop.

## 4. Shipped: AISP_TCGEN05_EARLY_FILL=1 (prologue overlap)

The one addressable piece: the plain 1SM branch now inits its mbarriers and
issues the first kStages TMA fills BEFORE the tcgen05 TMEM allocation, so
alloc + __syncthreads + accumulator setup overlap the 768ns first-arrival
TMA flight instead of preceding it. The producer loop starts at p=kStages
with identical stage/gate arithmetic; TMA issue order and per-tile UMMA
order are unchanged (bit-identity structural). 2SM and mcast branches keep
the incumbent order (alloc at branch top).

- A/B (process-interleaved paired CUDA-graph timing, one build per process
  — NB: loading two builds into one process ALIASES the kernel (identical
  mangled template symbol; the dynamic linker binds both modules to the
  first-loaded instance and the launch fails with cluster_launch invalid
  argument — same trap Y2's per-process ab2.py design dodged); 5 pairs,
  7 graph reps of 32 calls each):

| pair | EF0 med | EF1 med | delta |
|---|---|---|---|
| 1 | 15.189us | 14.935us | -0.254 |
| 2 | 15.204us | 15.050us | -0.154 |
| 3 | 15.114us | 14.928us | -0.186 |
| 4 | 15.200us | 15.134us | -0.066 |
| 5 | 15.204us | 14.980us | -0.224 |

med-of-meds 15.200 -> 14.980us (-0.22us, 1.015x); min-of-mins 14.967 ->
14.771us; EF1 wins ALL 5 pairs on med AND min, zero overlap of pairwise
medians. The B58-backup control build measured 15.193us med in the same
session (B58's shipped number), and the refactored file at EARLY_FILL=0
measured 15.189us — the deferred-callable refactor is perf-neutral at
default-off.
- Bit-identity (cross-process .pt compare): EF0 and EF1, CTA1 AND CTA2, all
  torch.equal vs the B58 backup build outputs. allclose vs fp32 matmul ref
  PASS. (CTA2's code path at EARLY_FILL=1 is order-identical to incumbent
  by construction.)
- ncu --set full on the shipped build (/tmp/frontY3/ncu_full_ef1.ncu-rep vs
  ncu_full_cta1.ncu-rep): byte-identical traffic — TMA-load 201.326592MB in
  both, DRAM read 16.82MB in both, lts tex-read sectors ~3.27M in both. The
  change is purely temporal (issue start time), as designed. Kernel time
  under 41-pass replay (18.2-18.4us) cannot resolve 0.22us; the bankable
  number is the interleaved graph A/B above.
- Harness x3 all three shared-file targets: see log
  /tmp/frontY3/harness_after.log (verification passed, 9/9).
- Diff: c7e412cc81c2905c36464ae4bb681219 -> 82e2a128eecc4e71e68b5eb7f3121706
  (`code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu`): +65/-9 lines:
  the AISP_TCGEN05_EARLY_FILL macro + comment, the alloc_tmem deferred
  callable, alloc_tmem() calls at the 2SM/mcast branch tops, and the plain
  1SM branch's p_fill early-fill block with the producer loop starting at
  p_fill. Y3 artifacts: /tmp/frontY3/{make_probe3.py,probe3.cu,p3_driver.py,
  ncu_full_cta1.ncu-rep,ncu_full_ef1.ncu-rep,raw_cta1.csv,raw_ef1.csv,
  cublas_sol.py,cublas_sol2.py,y3_one.py,y3_bitcmp.py,ab_run.log,
  ab_interleaved.log,harness_after.log,capstone_kernels_tcgen05.cu.bak.B58}.

## 5. Named next lever (post-Y3)

The mainloop is CLOSED (87% of the real pipe rate; the fill ramp's issue
start was harvested here, and the ramp RATE plus the steady-state period are
hardware properties at this stage size). The remaining kernel-level meat is
the EPILOGUE (~2.8us probed): Y2's named TMA tensor store (~0.3-0.5us) is
the only concrete residual. The structural unlock beyond that is shape- or
contract-level, not kernel-level: at 2048^3 with 128x256 atoms there are
only 128 tiles for 152 SMs and a hard accumulation-order contract; relaxing
either (larger problem, or a sanctioned non-bit-identical split-K mode)
is what would re-open >1us wins. Recommend the lab's expectation files
adopt the measured ~1.9 PF dense-fp16 ceiling for GB300 SoL accounting
(the B58 kernel was at ~58% REAL SoL, the Y3 ship is at ~59%; the "30.0%
SoL" framing against 3.77 PF overstated the remaining headroom by 2x).
