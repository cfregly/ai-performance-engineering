# GB300 NVFP4 dual_2sm port (Front N4b, 2026-06-12)

## Verdict
**SOTA-CLASS WIN (absolute) / correctness-proven; HONEST NEGATIVE vs cuBLASLt-NVFP4 relative.**
The block-scaled NVFP4 port of the dual_2sm machinery runs the full SF pipeline
(TMA -> smem -> UTCCP -> TMEM, block-scaled tcgen05 2SM MMA) **bit-exactly**
(rel_err == 0.0 on the exact dataset at 2048/4096/8192) and lands at
**219.8 us at 8192^3** — under the 313.5 us SOTA-CLASS bar (the FP8 champion's
absolute time at the same FLOP count; 1.43x faster) — but only **0.594x of
cuBLASLt NVFP4** (5003 vs 8427 TFLOPS). The deficit is structural (see
"Why 58%"), with the next lever named below.

## Recovery note (why N4b)
Predecessor Front N4 stalled mid-implementation and saved NOTHING usable:
/tmp/frontN4 held stale pre-B75 FP8 backups only (no NVFP4 partial, no design
notes). N4b restarted from the committed-spec base (verified md5s:
tcgen05_dual_2sm_fp8.cu 26bc94aa59b06d028937be179a814e75, tcgen05_loader.py
8236d523308b1bc87ddfddbb9107b5f4, bench_dual_2sm_fp8.py
a179a74103d579a815dc5ab24c26d723).

## Design (stated before building; B49 host print-checked)
- **Atom**: `SM100_MMA_MXF4_2x1SM_SS<e2m1, e2m1, f32, ue4m3, 256, 128, VS=16, K, K>`
  -> `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16`. Atom
  K-extent = 64 elements. (B46's "MMA-K=96" is the CuTeDSL SM103 op; this C++
  drop's dense MXF4 traits are K=64 — verified in the vendor headers and by
  host print: `partition_shape_A((256,256)) = ((_128,_64),_1,_4)`,
  `partition_shape_B((128,256)) = ((_64,_64),_1,_4)`.)
- **Tile/stages**: kTileM=256 (SM pair), kTileN=128, kTileK=256 (SW128 e2m1
  atom: K-rows are 128 BYTES), kStages=3. Stage = A 16KB + B 8KB + SFA 2KB +
  SFB 2KB = 28KB; smem/CTA 87,040 B (host print). num_k_tiles at 8192 = 32,
  4 atom k-blocks/stage.
- **SF gmem**: cuBLASLt 128x4-blocked layout via
  `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA/SFB` — identical to
  torch._scaled_mm's expected layout (`to_blocked`); proven by cuBLASLt
  returning rel_err == 0.0 on the same buffers the custom kernel consumes.
- **SF pipeline** (the vendor collective's exact scheme, re-hosted in the
  FP8 champion's persistent eo=0 mainloop):
  - SFA: per-CTA M-half slices, `SM100_TMA_2SM_LOAD` (completes at the
    leader's barrier like A/B).
  - SFB: BOTH CTAs need FULL kTileN scales, so each CTA TMA-issues HALF via
    `SM100_TMA_2SM_LOAD_MULTICAST` masked to the pair (mask from
    `create_tma_multicast_mask<1>` over the TiledMMA_SF cluster layout).
  - expect-tx (B53/B57 INCLUDING SF slices): `2 x (A-half + B-half +
    SFA-half + SFB-half-multicast-slice)`; the multicast slice counts once
    per destination pair (B53 law) — correct on first try (no hang).
  - UTCCP: `SM100_UTCCP_4x32dp128bit_2cta` via `make_utccp_copy` +
    `get_utccp_smem_desc_tensor`, issued by the LEADER consumer's elected
    thread right after the stage's full-wait, before the stage's MMAs.
    Single TMEM SF buffer; same-thread same-pipe in-order issue orders
    UTCCP(s+1) after MMA(s) (vendor-proven; no fence). The stage's
    `tcgen05.commit` (umma_arrive_multicast) tracks the UTCCP too, so the
    empty barrier also gates SF-smem reuse.
- **TMEM budget (B75, stated first)**: acc = 128 fp32 cols/CTA + SFA 4 cols +
  SFB 4 cols (k256: 4 k-blocks x 1 col; Dup4by1 handled inside tmem_sf_frg).
  `tcgen05.alloc` takes a POWER-OF-TWO column count -> 256-col alloc/CTA ->
  TMEM-implied **2 CTAs/SM** (admissible; enforced by a statically-folded
  device `__trap` check). **n256 is INADMISSIBLE for NVFP4**: acc 256 + SF
  needs a 512-col alloc = 1 CTA/SM — so the FP8 champion's n256 geometry
  cannot carry over; this port is n128-only.
- **Epilogue**: verbatim FP8 champion te1+dh1 (chunked 32-col t2r -> SW128
  staged smem -> one cp.async.bulk.tensor.2d per 128x64 fp16 box; in-kernel
  __float2half_rn). dh1 carried — dtype-independent of operands (bit-exact
  on the exact dataset, proving RN equivalence held again).
- **Scope cut** (stall-proofing): PERSIST=1 + eo=0 only. AMCAST, EPI_OVERLAP,
  PREFETCH, and non-persistent grids were NOT ported.

## Ceiling calibration (B61)
Warmed `torch._scaled_mm` NVFP4 (cuBLASLt nvjet, e2m1 x e2m1, ue4m3 SF_VEC=16,
fp16 out) at 8192^3: **127.5-127.7 us = 8607-8622 TFLOPS** — the doubled
(~8 PF-class) ceiling above FP8's ~3859 TF, confirmed.

## Measurements (GPU 1, exact dataset, rel_err == 0.0 everywhere)
12-rep interleaved medians at 8192^3 (run7):

| arm | us | TFLOPS | % calSoL | vs cuBLASLt |
|---|---|---|---|---|
| cuBLASLt NVFP4 (scaled_mm) | 130.5 | 8427 | 97.7% | 1.0 |
| **nvfp4_2sm n128s3mb2k256rg4te1dh1 (champion)** | **219.8** | **5003** | **58.0%** | **0.594x, 0/12** |
| nvfp4_2sm n128s3mb2k256rg8te1dh1 | 224.2 | 4904 | 56.9% | 0.582x |

Sweep (8-rep, run6): rg4 1.0205x and rg16 1.0179x over rg8 (8/8 each);
s2k256 0.767x (shallow ring starves); k128/SW64 te0 s4/s6 ~0.54x (doubled
barrier round-trips per fed byte + the te0 register-store epilogue).
Spread at 12 reps: champion [218.6, 221.3] us; cuBLASLt [129.3, 131.1] us.

Correctness ladder:
- exact (e2m1 values {0,±0.5..±2}, power-of-2 scales {0.5,1,2} -> exact fp32
  partials): rel_err == **0.0** at 2048, 4096, 8192 — custom AND cuBLASLt.
- scaled-random (per-16-group amax/6 e4m3 scales, nearest-e2m1, matching
  torch._scaled_mm quantization; stated gate < 1e-3): measured **0.0** at
  4096 for rg8 and rg4 (fp16 rounding absorbs fp32 ordering deltas).
- **Gates: FP8 champion rel_err 0.0 and FP16 champion rel_err 0.0 on the
  modified shared loader — PASS.**

## Why 58% (and not FP8's 91%)
The FP8 champion amortized the per-round serialized epilogue + ring
drain/refill over an n256 tile and 64 k-stages. NVFP4 forces n128 (TMEM pow2
alloc law) -> 2x the rounds, halves the k-stages per round (32), and the MMA
rate doubles — so the FIXED per-round cost (full ring drain, cold refill,
tmem_empty handshake, serialized epilogue) is paid 2x as often against
half-length compute. Raster (rg4/8/16) moves it only ~2%; ring depth beyond
s3 is smem-capped at k256 (s4 = 113KB > 110KB).

## Named next lever
**Port the F8c eo=2 cross-tile overlap epilogue** (dedicated epilogue
warpgroup + double TMEM accumulator buffers): at n128 the double buffer is
2x128 = 256 cols + SF 8 -> still the 256-col... it needs 512-col alloc
(2 CTAs/SM lost) — so the admissible variant is the **dual TMEM alloc**
(pow2 law workaround: separate 256-col acc-pair alloc + 32-col SF alloc, or
eo=2 with 2x128 acc in one 256-col alloc + SF in a separate 32-col alloc;
2x(256+32)=576 > 512 fails, 3x(128+32)=480 fits for eo=0) — i.e. **either**
(a) eo=2 with split SF alloc if the allocator packs 288/CTA x 2 (it does
not: 576 > 512 — eo=2 must drop to 1.78 CTAs/SM equivalent, inadmissible),
**or** (b) the cheaper structural fix: **multi-tile rounds** — walk T
consecutive n-tiles per round through the SAME smem ring with one epilogue
per T tiles (no extra TMEM), amortizing the round boundary T-fold. (b) is
the highest-EV unlock; (a)'s arithmetic is stated above as un-winnable.

## Traps hit this session (bankable)
1. **This CUTLASS drop's `tile_atom_to_shape_SFA/SFB` ALWAYS appends an L
   mode** (rank-3 (M,K,L) layout) — unlike cutlass-main's rank-3-problem
   branch. Slice L=0 on the host or every downstream local_tile/partition
   gains a phantom mode ("Mismatched Ranks" at the copy sites).
2. **Host-side tmem_ptr arithmetic is a gcc always_inline death**:
   `cute::rotr` (cute/pointer.hpp:311, tmem_ptr operator+) cannot inline as
   a host function at -O0 -> "inlining failed in call to always_inline".
   Keep ALL tmem fragment probing (find_tmem_tensor_col_offset etc.) on
   device; the FP8 file never touched tmem ptrs on host, which is why it
   never saw this.
3. **`ArrayEngine<e2m1>::begin()` is a subbyte_iterator**: the te1 staging
   path's `reinterpret_cast<uint4*>` / `cast_smem_ptr_to_uint` need
   `raw_pointer_cast(...)` around it.
4. The B53 multicast expect-tx law held verbatim for the SFB pair-multicast
   (one count per destination pair): first-try no-hang.
5. tcgen05.alloc pow2-column law is the binding occupancy constraint for
   block-scaled TMEM (acc+SF can't share the FP8 alloc count): state it in
   the TMEM-arithmetic-first pass (B75) before picking kTileN.

## Files (pod /work/ai-performance-engineering == local repo, md5-matched)
- code/labs/custom_vs_cublas/tcgen05_dual_2sm_nvfp4.cu (NEW)
- code/labs/custom_vs_cublas/bench_dual_2sm_nvfp4.py (NEW)
- code/labs/custom_vs_cublas/tcgen05_loader.py (MODIFIED: +nvfp4 loader,
  champion defaults n128 s3 mb2 k256 rg4 te1 dh1)
- docs/gb300-nvfp4-dual2sm.md (this doc)
- Pod logs: /tmp/frontN4b/{DESIGN.md,build*.log,run*.log}
