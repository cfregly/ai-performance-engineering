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
