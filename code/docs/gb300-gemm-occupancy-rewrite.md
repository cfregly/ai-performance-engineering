# GB300 gemm_cluster Occupancy Rewrite (dual-CTA variant) — Fronts E + E2

## Verdict

**WIN (measured, verified) — config (256,2): 1311 TFLOPS best / 1222 median,
35.0% / 32.6% FP16-SoL, beats the incumbent cluster kernel in every
interleaved rep (median 1.163x) and lifts the harness contract 2.33x -> 2.63x
with verification PASSED.** Short of the 40-48% prediction (mechanism below).
Front E delivered the implementation without pod access; Front E2 (this doc,
2026-06-10/11) built, measured, ncu-grounded, and gated it on the GB300 pod
(aisp-gb300-runall, GPU 2, 8192x8192x8192 FP16, peak 3.75 PFLOPS).

The kernel (`tcgen05_dual_cta.cu`) ran correct on first build — zero .cu
fixes. rel_err = 0.0 vs torch.matmul on every config. No hangs.

## Config sweep (CUDA events, 50 iters, GPU 2)

| variant | us (run A) | us (run B) | TFLOPS (best) | %SoL (best) | correct |
|---|---|---|---|---|---|
| cuBLAS (target) | 604.6 | 605.1 | 1818.7 | 48.5% | yes |
| cluster (incumbent) | 949.8 | 962.2 | 1157.6 | 30.9% | yes |
| dual_cta n=128 s=3 | 1054.4 | 1108.9 | 1042.7 | 27.8% | yes |
| dual_cta n=128 s=2 | 909.1 | 949.4 | 1209.4 | 32.3% | yes |
| **dual_cta n=256 s=2** | **838.3** | 914.8 | **1311.5** | **35.0%** | yes |

Front E's predicted default (128,3) is a measured **LOSS** (-8% vs cluster):
halving tile N halves arithmetic intensity, and DRAM/L2 latency per delivered
FLOP eats the occupancy gain. The winning config keeps the incumbent's
128x256 tile (full AI) and pays for the second CTA with stages (4 -> 2):
2 CTAs x 2 stages beats 1 CTA x 4 stages — concurrency from a second
independent MMA+TMA stream is worth more than lookahead depth.

Run-to-run thermal drift is real (~5-9% on this node): interleaved A/B
(6 reps, alternating in one process) is the fair read:

```
cluster      min  952.4  median 1046.7  max 1080.1 us   (drifts hot)
dual(256,2)  min  885.5  median  899.7  max  917.2 us   (stable)
per-rep speedup: 1.076x .. 1.178x, median 1.163x — dual wins ALL 6 reps
```

## ncu evidence (256,2) — the occupancy mechanism landed

`--set full --launch-skip 5 --launch-count 2`, report at
`/tmp/frontE2/dual_cta_n256s2.ncu-rep` on the pod. Both profiled launches
agree:

| metric | incumbent (B35) | dual_cta (256,2) |
|---|---|---|
| **Block Limit Shared Mem** | 1 | **2** (98.4KB + 1KB driver per CTA) |
| Block Limit Registers | — | 2 (255 regs/thread, at the LB(128,2) cap) |
| Theoretical / Achieved Occupancy | — / 6.2% | 12.5% / **12.06%** (2 CTAs/SM resident) |
| Duration (locked clocks) | 904 us | **799.4 us** (= 1375 TFLOPS, 36.7% SoL) |
| Tensor pipe (of elapsed) | 58% SM busy | **61.9%** (highest-utilized pipe) |
| DRAM throughput | 33% | **21.8%** (1.73 TB/s — nowhere near the roof) |
| L2 throughput | — | 45.2% |
| Top stall | scoreboard-on-smem 46.5% | **long_scoreboard 28.5 of 38.9 warp-cycles/instr (~73%)** |

Mechanism of the shortfall vs the 40-48% prediction: occupancy doubled but
tensor-pipe duty rose only 58% -> 61.9%. The strict empty-barrier protocol
(correct, but each CTA now waits for tcgen05.commit before stage reuse)
costs per-CTA duty that the co-resident CTA only partially refills — with 2
stages each, both CTAs tend to starve at the same time. DRAM (21.8%) is NOT
the constraint; the kernel is still TMA-latency-bound (long_scoreboard
dominant). The fix is more in-flight loads per SM, not more CTAs:
multicast (halve B traffic) or 3+ stages at 2 CTAs (needs <75.8KB/CTA, i.e.
a smaller B stage or A/B stage split).

## Harness verify gates (all PASSED)

```
AISP_TCGEN05_VARIANT=dual_cta (loader defaults now 256,2):
  verification PASSED, speedup 2.6317x   (run 20260611_031840)
AISP_TCGEN05_VARIANT=dual_cta AISP_DUAL_TILE_N=256 AISP_DUAL_STAGES=2:
  verification PASSED, speedup 2.5578x   (run 20260611_030506)
default (cluster) regression check:
  verification PASSED, speedup 2.4447x   (run 20260611_030548; contract 2.329x intact)
```

## Deltas vs Front E's delivery (documented deviations)

- **Zero .cu changes.** Pipeline protocol (empty-barrier phase parity,
  expect_tx bytes, umma_arrive single-arrival) verified by tracing and by
  running: correct on first build for all three configs.
- `tcgen05_loader.py`: env defaults flipped to the measured winner —
  `AISP_DUAL_TILE_N` 128 -> **256**, `AISP_DUAL_STAGES` 3 -> **2**.
- `bench_dual_cta.py`: non-sweep default config (128,3) -> **(256,2)**.

Pod state preserved: base 2f7e30f9, all GB300 fixes untouched; originals of
the 3 overwritten files backed up at `/tmp/frontE2/`. Nothing committed.

## Named next lever

1. **2x1 cluster + TMA multicast of B** on top of the dual-CTA (256,2)
   footprint: halves B-side L2->SM traffic, directly attacks the dominant
   long_scoreboard stall while keeping 2-CTA-equivalent concurrency via the
   cluster pair.
2. **Persistent CTAs + tile-swizzled scheduler**: 4096 one-shot CTAs pay
   launch + 255-reg epilogue (256 fp32/thread at n=256, at the register cap)
   per tile; persistence amortizes both and enables L2-friendly tile order —
   i.e. converge on the CUTLASS sm100 warp-specialized collective shape one
   verified step at a time.
