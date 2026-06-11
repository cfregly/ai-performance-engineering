# GB300 capstone tcgen05: F decomposed — the "fixed cost" was the t2r epilogue atom (Front Y)

**Date:** 2026-06-11 | **Pod:** aisp-gb300-runall (GPU 3) | **Base:** 2f7e30f9 (B53 state) + uncommitted GB300 fixes
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
