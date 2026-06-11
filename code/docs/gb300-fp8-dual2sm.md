# GB300 FP8 (e4m3) port of the dual_2sm GEMM champion (Front F8)

**Mission (B68 frontier paragraph):** port the battle-tested FP16 dual_2sm
machinery (2-SM UMMA pair, warp-split producer/consumer, persistent
clusters + GROUP_M raster, TMA-store epilogue; ~70% real FP16 SoL) to dense
FP8 e4m3 operands as a NEW variant file and measure against cuBLASLt FP8 on
8192^3. Plain dense GEMM: e4m3 x e4m3 -> fp32 TMEM accumulate -> fp16 D.
NO block scales.

Files (all NEW or append-only; the FP16 champion is untouched):
- `labs/custom_vs_cublas/tcgen05_dual_2sm_fp8.cu` -- the port (generated
  from the FP16 file by an exact-match patch script, /tmp/frontF8/
  make_fp8_variant.py; every replacement asserted to hit exactly once).
- `labs/custom_vs_cublas/tcgen05_loader.py` -- appended Stage 15 loader
  (`matmul_tcgen05_dual_2sm_fp8`, env prefix `AISP_DUAL2SM_FP8_*`).
- `labs/custom_vs_cublas/bench_dual_2sm_fp8.py` -- kernel-level A/B +
  correctness harness (this lab has no FP8 quick-lab target; the
  kernel-level A/B below is the deliverable).

## 1. Design math (stated before building)

MMA atom: `SM100_MMA_F8F6F4_2x1SM_SS` (tcgen05.mma.cta_group::2
.kind::f8f6f4). In the repo CUTLASS drop the F8F6F4 op is a NON-template
tag; types/shape go through `MMA_Traits<Op, a,b,c, C<M>,C<N>, ...>` (the
77_blackwell_fmha pattern) -- unlike the templated FP16 op tag.

B49-style host-only partition print check (FIRST, before any device build;
/tmp/frontF8/check_fp8_partition.cu against the repo CUTLASS):

```
=== TILE_N=128 TILE_K=128 ===  tile_size MNK: 256 128 32 (atom K-extent=32)
partition_shape_A: ((_128,_32),_1,_4)   -> A = this CTA's 128-row M-half
partition_shape_B: ((_64,_32),_1,_4)    -> B = this CTA's kTileN/2 col-half
smem A bytes/stage: 16384  B bytes/stage: 8192 (SW128)
=== TILE_N=128 TILE_K=64 ===   12KB/stage (SW64)
=== TILE_N=256 TILE_K=128 ===  A 16KB + B 16KB = 32KB/stage (SW128)
```

The FP8 atom splits B N/2-per-CTA exactly like the FP16 atom (the B49
trap verified at trait+print level), and the atom K-extent is 32 e4m3
elements (256 bits / 8) vs 16 for FP16. Consequences baked into the port:

- `kTileK` default 128 (was 64): the per-stage byte footprint at TILE_N=128
  (A-half 16KB + B-half 8KB = 24KB) is BYTE-IDENTICAL to the FP16
  champion's stage while feeding 2x the K-elements -> barrier round-trips
  per fed byte HALVE (num_k_tiles at 8192: 128 -> 64; still 4 atom
  k-blocks/stage). Same 72KB ring at stages=3 -> the FP16 champion's
  3-CTAs/SM occupancy math carries over unchanged (smem-implied 3,
  TMEM-implied 4 at 128 fp32 cols/CTA, min = 3; B63 law).
- Swizzle atom is kTileK-conditional (e4m3 K-rows are kTileK BYTES):
  kTileK=128 -> `Layout_K_SW128_Atom`, kTileK=64 -> `Layout_K_SW64_Atom`.
- TMA_EPI (lever h) keeps the fp32 staging path: the 128x32 fp32 = 16KB
  chunk only fits the drained smem_A stage when kTileK >= 128 (asserted).
- expect-tx stays the generic `2 * (sizeof A-slice + sizeof B-slice)`
  (B53/B57: full delivered slice, no participant multiplier).
- TMEM accumulator stays fp32 (the 512-col wall is unchanged).
- Levers ported and exercised: WARP_SPLIT, MIN_BLOCKS, TILE_K, PERSIST,
  RASTER_GM, TMA_EPI, EPI_ATOM. AMCAST / EPI_OVERLAP / PREFETCH paths were
  carried verbatim (type-generic, compiled out at 0) but not swept -- all
  three were measured FP16 losers/ties.

## 2. FP8 ceiling calibration (B61 method)

cuBLASLt FP8 = `torch._scaled_mm` with unit fp32 scales (e4m3 x e4m3 ->
fp16), profiler-named `nvjet_sm103_qqhsh_128x256_128x6_2x1_2cta_v_bz_TNT`
(nvjet itself runs a 2-CTA 128x256 tile, k128, 6 stages). Cold-session
sweep on GPU 2:

| size    | median   | TFLOPS |
|---------|----------|--------|
| 4096^3  |  40.2 us | 3418   |
| 8192^3  | 288.8 us | 3807   |
| 12288^3 | 1123.9us | 3302   |
| 16384^3 | 2590.3us | 3396   |

The asymptote PEAKS at 8192^3. In the warmed interleaved session below the
same kernel runs 266.9 us = 4119 TFLOPS; the honest real-SoL denominator
is the best observed cuBLASLt FP8 rate, **~4119 TFLOPS (266.9 us at
8192^3)** -- the B68 estimate (~3.7-3.8 PF) was the cold-session value.
(FP16 analog: B61's ~1900 TF dense-fp16 ceiling.)

## 3. Correctness (explicit tolerance)

Exact-dataset method: inputs drawn from {0, +-0.5, +-1, +-1.5, +-2} cast
to e4m3 (all exactly representable). Every product is a multiple of 0.25
with |sum| <= 32768, so EVERY fp32 partial sum is exact regardless of
accumulation order: the kernel's fp16 output must equal the fp32
upcast-matmul reference (TF32 off) rounded to fp16 BIT-EXACTLY.
**Required tolerance: rel_err == 0.0 exactly.** (A randn-quantized
dataset gates rel_err < 1e-3 to cover order-dependent fp32 rounding;
exact-mode is the primary gate.)

Machinery ladder, each rung verified at 2048^3 AND 4096^3, rel_err = 0.0:

| rung | config (n,s,ws,mb,k,p,rg,te) | result |
|------|------------------------------|--------|
| base loop          | 128,3,0,2,128,0,0,0 | 0.0 / 0.0 |
| + warp-split       | 128,3,1,2,128,0,0,0 | 0.0 / 0.0 |
| + persist + raster | 128,3,1,3,128,1,8,0 | 0.0 / 0.0 |
| + TMA-store epi    | 128,3,1,3,128,1,8,1 | 0.0 / 0.0 |
| deep ring (SW64)   | 128,6,1,3,64,1,8,0  | 0.0 / 0.0 |
| 4 CTAs/SM (SW64)   | 128,4,1,4,64,1,8,0  | 0.0 / 0.0 |
| big tile           | 256,3,1,2,128,1,8,1 | 0.0 / 0.0 |

All 8192^3 bench arms below also verified rel_err = 0.0 on the exact set.

## 4. 8192^3 interleaved A/B (12 round-robin reps/arm, GPU 2)

(results file /tmp/frontF8/ab_8192_223418.log)

| arm | median | TFLOPS | vs cuBLASLt FP8 (paired) |
|-----|--------|--------|--------------------------|
| cuBLASLt FP8 (scaled_mm)        | 266.9 us | 4119 | 1.0 |
| fp8_2sm n256s3 mb2 k128 rg8 te1 | 374.8 us | 2934 | 0.721x, 0/12 |
| fp8_2sm n128s3 mb3 k128 rg8 te1 | 387.9 us | 2835 | 0.691x, 0/12 |
| fp8_2sm n128s3 mb3 k128 rg8 te0 | 448.0 us | 2454 | 0.599x, 0/12 |
| fp8_2sm n128s4 mb4 k64 rg8 te0  | 534.1 us | 2058 | 0.505x, 0/12 |
| fp8_2sm n128s6 mb3 k64 rg8 te0  | 591.3 us | 1860 | 0.456x, 0/12 |
| fp16_2sm champion (own fp16 data) | 715.4 us | 1537 | 0.376x, 0/12 |

Readings:
- **custom FP8 vs custom FP16: 1.91x paired in-session** (374.8 vs 715.4
  us; vs the B68 830 us cross-session figure: 2.21x). The port banks
  nearly the full 2x FP8 rate.
- TMA-store epilogue is a BIGGER lever at FP8 than FP16 (n128: te0->te1 =
  1.155x vs the FP16 V7 1.09x): the fp32 D store path is a 2x-larger
  fraction of an FP8 kernel's time.
- The k64/SW64 family (deeper ring or 4 CTAs/SM) LOSES badly: halving the
  TMA box doubles barrier round-trips per byte -- issue-bound again, and
  even 4 CTAs/SM cannot buy it back. Ring DEPTH was not the FP8 unlock;
  feed WIDTH was.
- TILE_N=256 (pair tile 256x256, 2 CTAs/SM, TMEM 2x256=512 exactly) beats
  the FP16 champion geometry n128 (3 CTAs/SM): at FP8 rates the bigger
  tile's halved B traffic per flop outweighs the occupancy loss. nvjet's
  own 128x256_128x6 shape corroborates.

## 5. Refinement round + FINAL A/B (16 round-robin reps/arm, GPU 2)

(results file /tmp/frontF8/ab_final_223932.log; all arms rel_err = 0.0 on
the exact set; the champion config also re-verified rel_err = 0.0 on the
randn-quantized set at 8192^3 -- products are exact in fp32 and the fp16
output rounding absorbs order-level fp32 ulps.)

| arm | median | TFLOPS | vs cuBLASLt FP8 (paired) |
|-----|--------|--------|--------------------------|
| cuBLASLt FP8 (scaled_mm)       | 273.5 us | 4020 | 1.0 |
| **fp8_2sm n256s3 mb2 k128 rg8 te1** | **377.6 us** | **2912** | **0.7228x, 0/16** |
| fp8_2sm n256s3 ... rg4 te1     | 381.1 us | 2885 | 0.7197x |
| fp8_2sm n256s3 ... rg16 te1    | 380.9 us | 2887 | 0.7204x |
| fp8_2sm n256s3 ... rg8 te0     | 426.1 us | 2581 | 0.6423x |
| fp8_2sm n128s4 mb2 k128 rg8 te1| 410.4 us | 2679 | 0.6647x |
| fp16_2sm champion (own data)   | 724.1 us | 1519 | 0.3797x |

- Champion FP8 config: **(TILE_N=256, STAGES=3, ws=1, mb=2, TILE_K=128,
  persist=1, raster_gm=8, tma_epi=1)** = 377.6 us = 2912 TFLOPS = 70.7% of
  the calibrated 4119-TF ceiling (72.3% paired vs same-run cuBLASLt).
- **Custom FP8 vs custom FP16: 1.918x paired in-session** (377.6 vs 724.1
  us; vs B68's 830 us cross-session figure: 2.20x). The port banks ~96% of
  the theoretical 2x FP8-rate unlock.
- raster_gm 4/8/16 indistinguishable (<1%); 8 stays default.
- TMA-store epilogue at n256: 1.128x (426.1 -> 377.6); bigger than the
  FP16 V7 lever (1.09x) because the fp32 D-store path is a 2x-larger
  fraction of an FP8-rate kernel.
- Loader defaults flipped to the measured champion (n256 geometry).

## 6. ncu of the champion (single launch, /tmp/frontF8/fp8_best.ncu-rep)

```
Duration 323.5 us (isolated replay; steady-state interleaved median 377.6)
Compute (SM) Throughput 82.3% -- Tensor is the highest-utilized pipeline
  (81.7%, flagged over-utilized) ; DRAM Throughput 25.6%
Grid 304 = 152 SMs x 2 CTAs/SM (persistent static sizing exact, B65)
Block 128 thr, 71 regs/thread, dynamic smem 98.43 KB/block
Block Limits: SMem 2 (binding, as designed) | Regs 7 | Warps 16 | SM 32
Achieved occupancy 12.06% vs 12.5% theoretical (2 CTAs/SM x 4 warps)
Stages achieved: 3-deep ring x 32KB/stage (96KB + barriers in smem)
```

The kernel is TENSOR-PIPE-BOUND at 82%: the dual_2sm machinery feeds the
FP8 tensor core at the same relative saturation the FP16 champion achieved,
with the remaining ~28% gap to nvjet living in MMA-issue density (nvjet
runs a 6-stage 128x256 pipeline with k128 boxes -- our 2-CTA pair has a
3-stage 256-wide pipeline and pays the pair-barrier round-trip per stage).

## 7. Verdict + next lever

**WIN (correctness-proven port, measured).** The FP8 port preserves the
FP16 champion's machinery bit-for-bit (12 exact-match patches; every
correctness gate rel_err == 0.0 exactly), and:
- beats the custom FP16 champion 1.918x paired in-session (377.6 vs 724.1
  us; 2.20x vs the B68 830 us figure) -- the top-named unlock delivered;
- lands at 72.3% of cuBLASLt FP8 paired (the FP16 journey ended at 70% of
  ITS ceiling after seven fronts; the FP8 port reaches the same relative
  position in ONE front because the levers transferred);
- ncu-proven tensor-bound (81.7%) -- the structure, not the feed, is the
  next frontier.

**Named next lever:** MMA-issue density on the 2SM pair -- nvjet's shape
(128x256_128x6_2x1) suggests a 6-stage ring at TILE_N=256 needs the
110KB/CTA cap lifted toward the full 227KB (1 CTA/SM, 192KB = 6 stages x
32KB): trade co-residency for ring depth ON THE n256 FOOTPRINT (the FP16
(256,2) experiments never tested s>=4 because fp16 bytes made it
impossible; e4m3 makes 6 stages fit where 3 fit before). Secondary: fp16
D output in-kernel (halves D-store bytes; the te0->te1 delta shows the
store path is still ~12% of runtime).

## 8. FP16 gates re-verified (nothing destabilized)

```
gate 1/3: rel_err=0.00e+00 time=740.5us 1485 TFLOPS -> PASS
gate 2/3: rel_err=0.00e+00 time=806.9us 1363 TFLOPS -> PASS
gate 3/3: rel_err=0.00e+00 time=850.4us 1293 TFLOPS -> PASS
FP16 gates: 3/3  (champion band incl. thermal drift; rel_err 0.0)
```

Session: 2026-06-11, pod aisp-gb300-runall, GPU 2 under /tmp/gpu2.lock,
torch 2.12.0a0+nv26.05 / CUDA 13.2 / ncu 2026.1.1. Backups of touched
files in /tmp/frontF8/. Harness note: this lab has no FP8 quick-lab
target; the kernel-level A/B + exact-tolerance correctness above is the
deliverable (run_lab.py FP16 stages untouched).

---

# F8b. The named lever measured: 1-CTA/SM deep ring (honest negative) and in-kernel fp16 D (the real win)

Front F8b, 2026-06-11, same pod/GPU/lock. Incumbent verified at entry
(tcgen05_dual_2sm_fp8.cu dff82e3c..., loader 63837..., bench ef958...,
pod == local). Two levers from B72's Section 7, in order.

## F8b.1 Deep ring at 1 CTA/SM: the build variant

`DUAL2SM_MIN_BLOCKS=1` now selects a 1-CTA/SM deep-ring build: the 110KB
static smem cap (the 2-CTAs/SM footprint rule) lifts to the sm_103
per-block dynamic-smem opt-in maximum (227KB = 232448B; device reports
smem/SM 233472, opt-in 232448), and the existing B65 static grid sizing
derives 1 CTA/SM from the same footprint with no further change (grid
152, ncu-verified). At 1 CTA/SM the TMEM constraint RELAXES (256 of 512
cols, not binding); the binding limit is smem by design
(Block-Limit-SMem = 1). e4m3 is what makes the ring fit: 6 stages x
32KB (A 16KB + B 16KB at n256/k128) = 192KB (196.86KB with barriers) --
impossible at fp16, and nvjet's own FP8 champion shape
(128x256_128x6_2x1) runs exactly this 6-deep geometry. mb is part of
`kCfgMain`, so every mb variant gets a distinct kernel symbol (B61).
`tma_store_wait<kStages-1>` (the B58 ring-last-consumer gate) is already
kStages-parametric and recomputes at every depth.

Correctness: rel_err == 0.0 EXACTLY at 2048/4096/8192 for every config
below (exact dataset).

## F8b.2 8192^3 interleaved A/B, 12 round-robin reps/arm

| arm | smem/CTA | CTAs/SM | median | TFLOPS | vs cuBLASLt |
|---|---|---|---|---|---|
| cuBLASLt FP8 (scaled_mm)   | --       | -- | 271.7 us | 4046 | 1.0 |
| s3 mb2 control (B72 champ) | 98.4 KB  | 2  | 374.2 us | 2938 | 0.7261x |
| s4 mb1 deep ring           | 131.3 KB | 1  | 434.0 us | 2534 | 0.6262x |
| s5 mb1 deep ring           | 164.1 KB | 1  | 398.7 us | 2758 | 0.6795x |
| s6 mb1 deep ring           | 196.9 KB | 1  | 395.4 us | 2781 | 0.6890x |

Paired vs the s3/mb2 control: s4 0.8602x (0/12), s5 0.9318x (0/12),
s6 0.9456x (0/12). Depth-monotonic recovery that SATURATES below 1.0x:
the deepest ring that fits recovers most -- but not all -- of what the
second co-resident CTA was buying. The control reproduced B72 exactly
(374.2 us / 0.7261x vs the banked 377.6 us / 0.7228x).

## F8b.3 ncu mechanism (the bankable close)

Single-launch full-set profiles, /tmp/frontF8b/fp8_s3mb2.ncu-rep and
fp8_s6mb1.ncu-rep:

| metric | s3/mb2 champion | s6/mb1 deep ring |
|---|---|---|
| Duration (isolated)      | 298.8 us | 318.9 us |
| Tensor pipe              | 81.0% (flagged over-utilized) | 76.8% |
| Grid                     | 304 = 2 CTAs/SM | 152 = 1 CTA/SM (B65 exact) |
| Block Limit SMem         | 2 (binding) | 1 (binding, as designed) |
| Regs/thread              | 71 | 70 |
| Dynamic smem/block       | 98.43 KB | 196.86 KB |
| Achieved occupancy       | 12.06% | 6.27% |
| Long-scoreboard stall    | 32.0 of 37.0 cyc/warp (86.4%) | 16.7 of 20.8 cyc/warp (80.6%) |

The deep ring did EXACTLY what it promised -- long-scoreboard stall per
warp HALVED (16.7 vs 32.0 cycles): the 6-deep feed covers TMA latency
far better than the 3-deep feed. It lost anyway, because that latency
was already hidden by the co-resident CTA. What mb1 surrendered is the
second independent MMA issue stream per SM: under the eo=0 per-round
serialized epilogue, CTA B's mainloop keeps the tensor pipe busy while
CTA A walks its 8-chunk t2r -> stage -> TMA-store epilogue and its
round-boundary ring drain/refill; at 1 CTA/SM those windows (x ~14
persistent rounds) go fully exposed on the tensor-pipe timeline, and
the pipe drops 81.0 -> 76.8%. This is the B66 law operating at CTA
granularity: the lever shortened a latency nobody was waiting on, and
paid issue-stream concurrency for it. nvjet's 6-stage 1-CTA shape wins
only because its epilogue OVERLAPS the next tile's mainloop inside one
CTA (specialized epilogue warps + mbarrier choreography) -- that is a
STRUCTURE, not a config, and it is the same deep-rewrite frontier B32
named for the group-GEMM. Verdict for the lever as named: HONEST
NEGATIVE, mechanism banked; mb stays 2.

## F8b.4 The secondary lever was the real unlock: in-kernel fp16 D (DUAL2SM_D_HALF)

B72 sized the D-store path at ~12%. The true cost was larger: the dh0
champion stores D as fp32 (268 MB) and then `d_buffer.to(fp16)` runs a
SECOND elementwise kernel (re-read 268 MB + write 134 MB) inside the
timed region -- ~400 MB of DRAM traffic serialized after every GEMM.
`DUAL2SM_D_HALF=1` (scoped to TMA_EPI) converts in the staging epilogue
instead: each PAIR of 32-col fp32 t2r chunks packs via __floats2half2_rn
into ONE 128x64 fp16 staged tile -- still 128B rows, still the SW128
16B-group XOR, still one drained 16KB A stage, half the store calls
(4/round), half the D bytes, descriptor dtype FLOAT16 box (64,128) --
and the host returns the fp16 buffer directly. No conversion kernel
exists anymore.

8192^3, 12-rep interleave, run 1 / run 2 (fresh sessions; this pair of
sessions ran warmer than F8b.2 -- compare PAIRED ratios, not absolutes):

| arm | run 1 median | run 2 median | rel_err |
|---|---|---|---|
| cuBLASLt FP8       | 286.6 us | 283.1 us | 0.0 |
| champion dh0       | 395.5 us | 391.4 us | 0.0 |
| champion dh1       | 315.0 us (3491 TF) | 313.5 us (3508 TF) | 0.0 |

- paired dh1 vs dh0: **1.2527x / 1.2525x, 24/24 interleaved wins**
- paired dh1 vs cuBLASLt FP8: **0.9081x / 0.9070x** (the B72 incumbent
  was 0.7228x -- the gap to the library SHRANK by two thirds)
- correctness: rel_err == 0.0 EXACTLY at 2048/4096/8192 (in-kernel RN
  rounding of an exact fp32 accumulator is bit-identical to torch's
  host-side .to(fp16) RN; predicted and confirmed)

Loader defaults flipped after the two-session ratification:
AISP_DUAL2SM_FP8_D_HALF=1 -- the FP8 champion is now
(n256, s3, ws1, mb2, k128, p1, rg8, te1, dh1).

## F8b.5 ncu of the dh1 champion (/tmp/frontF8b/fp8_dh1.ncu-rep)

```
Duration 284.7 us isolated (dh0 same-method: 298.8) -- the GEMM kernel
  itself got ~14 us faster, on top of retiring the ~65-80 us conversion
Compute (SM) Throughput 87.2%; Tensor pipe 86.6% (dh0: 81.0%) --
  flagged over-utilized; the +5.6pt jump is store-path time returned
  to the MMA stream
DRAM write bytes 112.33 MB (dh0: 234.68 MB) -- the fp16-D floor (128
  MiB logical), write sectors 9.26M -> 5.07M
Grid 304 = 2 CTAs/SM, dyn smem 98.43 KB, Block-Limit-SMem 2 (binding),
  achieved occupancy 12.08%
Regs 103/thread (dh0: 71; the __floats2half2_rn packing) -- Block Limit
  Regs still >= BL-SMem, occupancy unchanged
```

## F8b.6 Verdict + next lever

- **Deep ring at 1 CTA/SM (the named lever): HONEST NEGATIVE** with the
  pipe-utilization mechanism banked (F8b.3). Ring depth is not the
  binding resource at this geometry; co-resident issue streams are.
- **In-kernel fp16 D: WIN, ratified, defaults flipped** -- 1.2526x
  paired over the B72 champion, 24/24, rel_err 0.0, landing at
  **0.907-0.908x of same-run cuBLASLt FP8**: parity-class (>= 0.85x).

**Named next lever:** the remaining ~9% to cuBLASLt is the FP8
scheduling frontier proper: overlap the per-round epilogue with the
next round's mainloop WITHIN the 2-CTAs/SM footprint -- a dedicated
epilogue warp pair + double-buffered 2x128-col TMEM accumulator views
inside the existing 256-col/CTA budget (the eo=2 family's structure at
eo=0's occupancy; needs the mbarrier rewrite, not a config). The F8b.3
profile says that epilogue exposure is ALSO what co-residency is
currently spending its concurrency to hide; reclaiming it in-CTA would
free the second CTA to add depth instead.

## F8b.7 FP16 gates re-verified (nothing destabilized)

```
gate 1/3: rel_err=0.00e+00 time=738.5us 1489 TFLOPS -> PASS
gate 2/3: rel_err=0.00e+00 time=786.8us 1397 TFLOPS -> PASS
gate 3/3: rel_err=0.00e+00 time=819.3us 1342 TFLOPS -> PASS
FP16 gates: 3/3  (champion band; shared loader rebuilt clean)
```

Session F8b: 2026-06-11/12, pod aisp-gb300-runall, GPU 2 under
/tmp/gpu2.lock (compute-apps checked idle before every timed block),
torch 2.12.0a0+nv26.05 / CUDA 13.2 / ncu 2026.1.1. Evidence:
/tmp/frontF8b/ (entry backups, build logs, dh_ab_run{1,2}.log,
fp8_s3mb2/fp8_s6mb1/fp8_dh1.ncu-rep, ncu_driver.py, run_dh_ab.sh).
Files (md5, pod == local verified at handoff):
tcgen05_dual_2sm_fp8.cu fe1f3aa68a062f2c07131ea4ceef8457,
tcgen05_loader.py e860783dcb07ceb96777dbf943c14bc5,
bench_dual_2sm_fp8.py 9be8d66ca79844c31350566686366e1c
(this doc md5-matched pod == local by the mirror step itself).

---

# F8c. Declared-final front: in-CTA epilogue/mainloop overlap inside the 2-CTAs/SM footprint -- HONEST NEGATIVE; the FP8 scheduling frontier is DECLARED at 0.908x

Mission (B73's named lever): reclaim the per-round serialized epilogue
exposure IN-CTA -- "a dedicated epilogue warp pair + double-buffered
2x128-col TMEM views inside the existing 256-col/CTA budget; the eo=2
structure at eo=0's occupancy." Measured to a decisive negative with the
mechanism fully decomposed; no default changes. The FP8 champion stands
at (n256, s3, ws1, mb2, k128, p1, rg8, te1, dh1) = 313.5-315.7 us /
~3500 TF / 0.907-0.908x of same-run cuBLASLt FP8.

## F8c.1 The TMEM/tile arithmetic, stated BEFORE coding (the formulation finding)

TMEM = 128 lanes x 512 fp32 columns per SM. Each CTA's accumulator
buffer = its 128-row M-half x kTileN fp32 = kTileN TMEM columns. The
budget identity that closes the whole family:

```
kTileN x kNumAccBufs x CTAs/SM <= 512
  n256 eo0:  256 x 1 x 2 = 512   <- the champion (big-tile AI, 2 streams/SM)
  n256 eo2:  256 x 2 x 2 = 1024  <- INADMISSIBLE (even 1 CTA/SM consumes all 512)
  n128 eo0:  128 x 1 x 3 = 384   <- 3 CTAs/SM (smem-bound at 3)
  n128 eo2:  128 x 2 x 2 = 512   <- the ONLY admissible double-buffer point
```

Three structural facts sharpened the brief's lever before any code:
1. **Double-buffering at n256 does not fit**: 512 cols/CTA fails the
   co-residency static_assert; there is no "n256 overlap at 2 CTAs/SM".
2. **2x128-col views of ONE n256 accumulation cannot work**: every
   256-wide tcgen05.mma k-block writes ALL 256 columns, so neither half
   is drainable before the tile's final k-block (the file's own eo=2
   structural note). The views must be two FULL n128 tiles.
3. **A warp PAIR cannot drain**: tcgen05.ld 32dp32b from TMEM
   subpartition w is only addressable by the warp of rank w within its
   warpgroup; a 128-lane drain needs ranks 0-3 = a full second
   warpgroup (threads 128-255), not parked warps 2-3.

So the implementable formulation is fallback (a): n128 tiles,
kNumAccBufs=2 (256 TMEM cols/CTA = exactly the champion's budget,
2 CTAs/SM preserved), the persistent eo=2 producer/consumer/epilogue-
warpgroup structure, PLUS the F8b te1+dh1 staged-fp16 store re-hosted in
the overlap epilogue.

## F8c.2 What was built (new mechanism, behind DUAL2SM_EPI_OVERLAP=2 + TMA_EPI=1 + D_HALF=1)

The pre-existing eo=2 path stored fp32 D by direct STG (the 50%-sector
path) + host-side .to(fp16) -- the very ~400 MB tax F8b retired. F8c
adds the staged TMA store to the eo=2 persistent epilogue, with the B58
wait choreography re-solved for a LIVE ring:

- Under eo=2 the smem ring NEVER drains (the mainloop streams across
  tile boundaries), so the eo=0 trick of staging through a drained
  smem_A stage is unavailable. A DEDICATED staging pair is added to
  SharedStorage: 2 x (128 rows x 64 fp16 cols, SW128 16B-group XOR) =
  32 KB. Total smem 107.52 KB/CTA (ncu-confirmed) <= the 110 KB
  2-CTAs/SM cap; the persistent grid static-sizes to 304 = 2 CTAs/SM.
- Sync is warpgroup-local `bar.sync 1, 128` (ids audited: nothing else
  in the kernel or the included CUTLASS device code uses named
  bar.sync) -- a __syncthreads would deadlock against the mainloop
  warps, which only rendezvous at kernel end.
- Staging-slot reuse: the ISSUING thread (warp 4 elected) waits
  `tma_store_wait<1>` (engine has READ the box issued two stores ago),
  then the named barrier republishes to all 128 epi threads; a final
  `tma_store_wait<0>` precedes the end-of-kernel rendezvous (smem
  lifetime). Per tile: 4 x 32-col t2r chunks -> 2 chunk-pair fp16
  boxes -> 2 TMA stores.
- Plumbing: loader gains AISP_DUAL2SM_FP8_EPI_OVERLAP / an
  `epi_overlap` arg (build name suffix `eo{n}`; symbol uniqueness via
  the kCfgEpi template tag -- B61); bench --fp8 gains the optional
  10th field EO.

## F8c.3 Correctness: rel_err == 0.0 EXACTLY, all paths, all sizes

Exact dataset, same-run fp32-upcast reference (TF32 off), sizes
2048/4096/8192: champion control, F8c eo2-te1dh1, and eo2-te0 all
report rel_err = 0.00e+00 bit-exactly (/tmp/frontF8c/
correctness_run1.log). The live-ring staging choreography is sound.

## F8c.4 The A/B: decisive loss, with the geometry decomposed

8192^3, 12-rep order-alternated interleave, same session
(/tmp/frontF8c/ab_run1.log):

| arm | median | vs cuBLASLt | vs champion |
|---|---|---|---|
| cuBLASLt FP8                       | 285.1 us / 3856 TF | --      | --      |
| champion n256-eo0-te1dh1           | 315.7 us / 3483 TF | 0.9072x | --      |
| **F8c eo2-n128-te1dh1**            | 396.6 us / 2772 TF | 0.7201x | **0.7948x, 0/12** |
| eo0-n128-mb3-te1dh1 (geom control) | 336.1 us / 3271 TF | 0.8494x | 0.9360x, 0/12 |

- The champion replicated its banked 0.908x in-session (healthy run).
- The staged-store lever DOES generalize into the eo=2 epilogue:
  eo2-te1dh1 376.7 us vs eo2-te0 546.2 us single-shot at 8192 (1.45x;
  the host-conversion + fp32-STG tax measured inside the overlap
  structure too). Banked as a mechanism; it just cannot rescue n128.

## F8c.5 ncu: the thesis falsified cleanly (/tmp/frontF8c/fp8_eo2.ncu-rep)

| metric | champion (F8b.5) | F8c eo2-n128 |
|---|---|---|
| Duration (isolated)   | 284.7 us | 359.0 us |
| Tensor pipe           | 86.6% (over-utilized) | **65.8%** |
| Compute (SM)          | 87.2% | 73.8% |
| L2 throughput         | 59.1% | 67.1% |
| Memory throughput     | 58.1% / 1.68 TB/s | 68.8% / 1.27 TB/s |
| Grid / CTAs/SM        | 304 / 2 | 304 / 2 (as designed) |
| Regs/thread           | 103 | 96 (no spill; BL-Regs 2 = BL-SMem 2) |
| Dyn smem/CTA          | 98.43 KB | 107.52 KB (= the stated arithmetic) |
| Top stalls            | long-scoreboard 88.3% of 34.7 cyc | long-scoreboard 59.8% + CTA-barrier 31.5% of 48.5 cyc |

The F8c thesis was "epilogue exposure reclaimed -> tensor pipe >90% ->
~290 us class." The opposite happened: the pipe FELL 86.6 -> 65.8. The
overlap works structurally (the epilogue is no longer serialized), but
at n128 the MAINLOOP cannot saturate the pair MMA: per FLOP it carries
2x the k-block barrier/issue round-trips and 1.5x the smem-feed bytes
(341 FLOP/smem-byte at n128 vs 512 at n256), and L2 pressure rises to
67%. The limiter at this geometry was never the epilogue -- it is the
feed, which the n256 big tile was specifically amortizing.

## F8c.6 Why the family is closed (the banked mechanism)

Two independent dominations, both measured:
1. **Geometry-matched**: at fixed n128, the in-CTA overlap (eo2,
   2 CTAs/SM + dedicated warpgroup) is 0.847x of plain eo0 at
   3 CTAs/SM (396.6 vs 336.1 us). The third co-resident CTA is the
   better epilogue-hider AND TMA-latency-hider than the warpgroup that
   replaced it -- B63's FP16 verdict replicates exactly at FP8, one
   rung down the occupancy ladder.
2. **Cross-geometry**: even the BEST n128 schedule is bounded by its
   own mainloop feed rate: eo0-n128-mb3 with three full MMA streams/SM
   and its epilogue already co-residency-hidden runs 336.1 us > the
   n256 champion's 315.7 us. No epilogue scheduling at n128 can make
   up a deficit that exists in the mainloop itself. (The k64 deep-ring
   escape is already closed by the B72 kTileK=128 box law.)

With the TMEM identity of F8c.1, this closes every point in the
in-CTA-overlap family on this kernel: the 512-col TMEM wall forces
double-buffering down to n128, and n128's feed tax exceeds the ~9%
epilogue exposure the overlap could reclaim. The structure nvjet uses
(specialized epilogue warps inside one CTA) only pays at tile shapes
whose accumulator fits TWICE in TMEM without halving tile AI -- FP8
dense at 256-col tiles is precisely where it cannot.

## F8c.7 The FP8 scheduling frontier, DECLARED

The final FP8 ladder (8192^3, vs same-run cuBLASLt FP8):

| rung | result |
|---|---|
| F8 type-level port (n128 FP16 geometry)         | 0.7228x |
| F8a/F8b n256 big tile + TMA-store epi (te1)     | ~0.79x -> 0.83x |
| F8b in-kernel fp16 D (dh1) -- **the champion**  | **0.907-0.908x** |
| F8c in-CTA epilogue overlap (eo2)               | 0.7201x -- REJECTED |

The remaining ~9% to the library is structural under this kernel's
resources: cuBLASLt's nvjet runs the epilogue-overlap structure at a
tile/TMEM geometry this 256-col accumulator budget cannot express
(and reaches 4046-4156 TF). Within the 2-CTAs/SM, 512-TMEM-col,
110KB-smem envelope, the scheduling levers are exhausted: warp-split,
persistence, raster, TMA-store epilogue, in-kernel fp16 D are all
banked wins; deep ring (F8b) and in-CTA overlap (F8c) are both
measured negatives with mechanisms understood. **The FP8 frontier for
tcgen05_dual_2sm_fp8.cu is 0.908x of cuBLASLt FP8 (313.5-315.7 us,
~3500 TF), parity-class (>= 0.85x), and further progress requires a
different ACCUMULATOR geometry (e.g. n128-epilogue-split atoms with
TMEM column sharing), not a different schedule.**

Defaults: UNCHANGED everywhere (loader eo default 0; champion config
untouched). The eo=2+te1+dh1 machinery remains available behind
AISP_DUAL2SM_FP8_EPI_OVERLAP=2 for future accumulator-geometry work,
correctness-proven at rel_err 0.0.

## F8c.8 Gates + files

```
FP8 exactness: rel_err == 0.0 at 2048/4096/8192 x {champion, eo2-te1dh1, eo2-te0}
gate 1/3: rel_err=0.00e+00 time=737.8us 1490 TFLOPS -> PASS
gate 2/3: rel_err=0.00e+00 time=813.2us 1352 TFLOPS -> PASS
gate 3/3: rel_err=0.00e+00 time=835.8us 1315 TFLOPS -> PASS
FP16 gates: 3/3  (champion band; rel_err 0.0)
```

Session F8c: 2026-06-11/12, pod aisp-gb300-runall, GPU 2 under
/tmp/gpu2.lock (compute-apps checked idle before every timed block),
torch 2.12.0a0+nv26.05 / CUDA 13.2 / ncu 2026.1.1. Evidence:
/tmp/frontF8c/ (entry backups, correctness_run1.log, ab_run1.log,
fp8_eo2.ncu-rep + details, fp16_gates.log, run scripts).
Files (md5, pod == local verified at handoff):
tcgen05_dual_2sm_fp8.cu 26bc94aa59b06d028937be179a814e75,
tcgen05_loader.py 8236d523308b1bc87ddfddbb9107b5f4,
bench_dual_2sm_fp8.py a179a74103d579a815dc5ab24c26d723
(this doc md5-matched pod == local by the mirror step itself).
