# GB300 capstone tcgen05 k-stage deepen: bK 64->32 + 8-stage ring (Front D5b)

**Date:** 2026-06-11 | **Pod:** <gb300-pod> (GPU 3) | **Base:** f4cab652 (B44) + uncommitted GB300 fixes
**Verdict: HONEST NEGATIVE — 0.873x (CTA1 24.11us -> 27.62us), bit-identical, control-isolated, dual-front replicated.**

The banked-next-lever from B44 — halve the k-tile (bK 64->32, SW64 swizzle atom,
~24KB/stage) so an 8-stage ring fits the same ~192KB smem budget and doubles
TMA-latency cover — was implemented, verified bit-identical, and measured as a
clean loss. The S=4@bK32 control arm ties the S=8@bK32 lever arm exactly
(27.60 vs 27.62us), which falsifies the lever's premise outright: **TMA latency
was already fully covered at B44's 4 stages; ring depth contributes nothing.
The entire 14.5% regression is the finer k-tile granularity itself.** The B44
config (bK=64, SW128, 4 stages) stays the incumbent.

This front (D5b) ran as the independent replication arm of a dual dispatch; the
primary D5 front reached the same verdict concurrently (see section 6).

## 1. What was built

`code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu` was parameterized so
every arm is one `-D` flag from a single source (no kernel-structure change —
the B44 4-stage warp-specialized producer/consumer protocol, mbarrier sizing,
TMA boxes, transaction bytes, trip counts and drain parity are all already
generic in `Stages`/layouts):

- `AISP_TCGEN05_KBLOCKS` (4 -> bK=64 + `UMMA::Layout_K_SW128_Atom`;
  2 -> bK=32 + `UMMA::Layout_K_SW64_Atom`) — plumbed as
  `TcgenVariantConfig::kKBlocks` into `bK = tile_size<2>(tiled_mma) * Int<kKBlocks>{}`
  and a `std::conditional_t` smem-atom pick (the atom's K extent must tile the
  k-tile's K extent exactly: 128B rows at bK=64 fp16, 64B rows at bK=32).
- `AISP_TCGEN05_STAGES` (ring depth; per-stage CTA1 smem 48KB at bK64,
  24KB at bK32 — 8 stages = same 192KB, inside the 227KB opt-in limit).
- `AISP_TCGEN05_SWEEP_BUILD` suppresses the global `TORCH_LIBRARY`
  registration so multiple configs coexist in one process for interleaved A/B.

Accumulation order is preserved by construction: k-blocks within a tile and the
tiles themselves walk K in strictly ascending order at the same instruction-K
granularity (bK64 = 4 k-blocks/tile x 32 tiles; bK32 = 2 x 64), so the UMMA
sequence — and the output bits — are independent of (kKBlocks, Stages).

## 2. Bit-identity gate (PASS, all arms)

Reference captured from the pristine B44 working copy (md5 `898fe01d...`,
verified == `/tmp/frontD5/capstone_kernels_tcgen05.cu.B44.bak` == local HEAD
f4cab652) with seed-42 2048x2048 fp16 inputs, including a 5-rep determinism
check (`/tmp/frontD5b/01_refcap_b44.*`, `ref_B44.pt`).

All three arms, both entry points, `torch.equal` vs that reference — **True**
(`/tmp/frontD5b/03_ab_kstage.stdout`, `ab_kstage_results.json`):

| arm | cta1 | cta2 |
|---|---|---|
| B44_bK64_S4 | True | True |
| D5_bK32_S8 | True | True |
| CTL_bK32_S4 | True | True |

## 3. Interleaved A/B timing (B37 method)

One process, three coexisting builds, arms rotated per rep (B44 -> D5 -> CTL),
9 reps x 300 CUDA-event-timed calls each, GPU 3, quiet window. Units: us/call.
Source: `/tmp/frontD5b/03_ab_kstage.stdout`.

**CTA1 (primary target `blackwell_matmul_tcgen05`):**

| arm | median | min | max | vs B44 |
|---|---|---|---|---|
| B44_bK64_S4 | **24.111** | 24.037 | 24.181 | 1.000x |
| D5_bK32_S8 | 27.615 | 27.573 | 27.691 | **0.873x** |
| CTL_bK32_S4 | 27.603 | 27.586 | 27.731 | 0.874x |

reps B44: 24.151 24.133 24.051 24.079 24.049 24.037 24.111 24.120 24.181
reps D5: 27.655 27.652 27.614 27.611 27.573 27.615 27.691 27.577 27.642
reps CTL: 27.731 27.639 27.622 27.586 27.603 27.591 27.601 27.588 27.626

**CTA2 (`cluster_gemm_tcgen05_cta2` kernel):**

| arm | median | min | max | vs B44 |
|---|---|---|---|---|
| B44_bK64_S4 | 24.215 | 24.191 | 24.228 | 1.000x |
| D5_bK32_S8 | 24.717 | 24.651 | 24.756 | 0.980x |
| CTL_bK32_S4 | 25.066 | 24.987 | 25.101 | 0.966x |

The B44 arm reproduces the banked B44 kernel frame (24.0us) within 0.5%. Rep
spreads are <=0.6% with zero overlap between arms — the 5-9% thermal drift on
this node is fully controlled by the interleave.

**Control-arm verdict:** S=8 vs S=4 at bK32 differ by 0.04% (CTA1). Ring depth
is a non-factor; the regression is tile-size-caused. (S=6 was not run by this
front — gate 5 makes it conditional on S=8 regressing vs S=4, which did not
happen; the primary front swept S=6 independently and reports the same ~27.5-
27.7us plateau.)

**Harness frames** (verification PASSED on every run):
- D5 state (defaults bK32/S8): 147.6us optimized frame, 106.19x
  (`/tmp/frontD5b/02_harness_d5_run1.results.json`).
- Final shipped state (defaults bK64/S4): 108.9us frame, 143.85x
  (`/tmp/frontD5b/07_harness_final_state.stdout`) — B44-class restored.
  Primary front's runs on the same state: 107.5us/146.0x and 112.8us/138.9x;
  `cluster_gemm_tcgen05` and `_cta2` verification True x2 each
  (`artifacts/runs/20260611_074533..074742*`).

## 4. Mechanism (ncu, B44 vs D5 arm, CTA1)

`--set full --launch-skip 5 --launch-count 2`, GPU 3. Reps:
`/tmp/frontD5b/ncu_b44_bk64_s4.ncu-rep`, `ncu_d5_bk32_s8.ncu-rep`; diff:
`/tmp/frontD5b/06_ncu_diff.stdout`; details: `ncu_*_details.txt`.

| metric (launch-avg) | B44 bK64/S4 | D5 bK32/S8 | delta |
|---|---|---|---|
| gpu__time_duration | 45.22us | 49.39us | +9.2% (replay-clamped; live +14.5%) |
| sm__cycles_active.avg | 67,766 | 75,639 | +11.6% |
| sm__mem_tensor_cycles_active (pct_of_peak_active) | 20.74% | 18.72% | -2.0pp (-9.7% rel) |
| smsp__inst_executed.sum | 2.182M | 2.642M | +21.1% |
| stall long_scoreboard / wait / short_scoreboard (per issue-active) | 3.47 / 2.29 / 0.96 | 3.64 / 2.17 / 0.67 | mix ~flat |
| smem ld wavefronts / bank conflicts | 1664 / 128 | 1664 / 8.5 | epilogue-only; negligible both |
| DRAM throughput (clean launch) | 371.6-376.3 GB/s | 346.5 GB/s | arithmetic of same bytes / longer time |

Verified mechanism: at bK32 the kernel issues **+21% instructions for the same
math** (64 k-tiles instead of 32: the consumer waits an mbarrier per 2 UMMAs
instead of per 4, the producer issues 2x TMA copies at half size, 2x commits),
and the tensor pipe's active duty drops 9.7% relative — the UMMA pipe goes
idle at k-tile boundaries more often, regardless of how many stages of data
sit ready in smem. The S-depth tie proves arrival latency was never the
binder at bK32; deeper staging has nothing to hide. Warp-stall mix and smem
behavior are essentially unchanged — the loss is structural cadence, not a
new stall class. (The primary front additionally attributes part of the loss
to lower per-request TMA efficiency of 64B-row SW64 boxes; plausible, but my
DRAM numbers are equally explained by constant bytes over longer time, so I
carry it as hypothesis, not finding.)

Caveat: launch 1 of my D5 ncu pass overlapped the primary front's harness runs
on GPU 3 (device-level Memory Throughput read 655.7 GB/s there); launch 2
(346.5 GB/s) and all kernel-scoped counters are consistent with the primary's
clean-window ncu (their 20.4%->18.4% tensor duty vs my 20.74%->18.72%).

## 5. Falsification statement

The B44-banked lever "bK 64->32 + SW64 atom halves per-stage smem, enabling an
8-stage ring that doubles TMA-latency cover" is **refuted at implementation
grade** on GB300 at 2048^3 FP16:

1. Implemented exactly as specified (SW64 atom, 8 stages, same warp-specialized
   pipeline), bit-identical outputs (sec 2).
2. The latency-cover premise is false: S=8 and S=4 at bK32 tie to 0.04%
   (sec 3) — the 4-stage/48KB B44 ring already hides all TMA latency.
3. The bK32 change itself costs 14.5% (CTA1) / 2.0% (CTA2) from doubled
   wait/commit cadence and +21% instruction issue at -9.7% tensor-pipe duty
   (sec 4). Both halves of the lever are individually negative or null; the
   combination cannot win. Do not revisit bK<64 for this kernel family at
   K-major fp16.

## 6. Dual-front note (validity)

A primary D5 front ran the same lever concurrently on this pod (evidence in
`/tmp/frontD5/`: own ref capture 07:21, timing 07:30, ncu 07:35-07:37,
ship-verify + working-copy update 07:38:20, final harness x6 07:45-07:48). My
session (D5b, evidence in `/tmp/frontD5b/`) is fully independent: own
reference, own builds, own interleaved timing, own ncu. The working-copy
source changed mid-session (f4bf058b -> aa013923, comments + `#ifndef`
defaults only); every measured arm in sections 2-4 pins
`-DAISP_TCGEN05_KBLOCKS/-DAISP_TCGEN05_STAGES` explicitly, so the compiled
kernels are invariant to that change. Timing windows did not overlap GPU work
(rep tightness + B44 reproduction at 24.11us confirm); the one contaminated
ncu launch is flagged in sec 4. Two independent measurements agree to <0.5% on
every shared number — B43-style replication, achieved accidentally.

## 7. Final tree state

- Working copy (LOCAL == POD, md5 `aa01392300ba8fa2ddeb4c64859bb06c`):
  the parameterized source with **defaults = the B44 incumbent (bK64/S4)** and
  the negative finding documented at the knobs. Shipped by the primary front;
  validated by my final harness run (verification PASSED, 143.85x, sec 3) —
  it compiles the B44 config by construction.
- The brief's restore-from-`.B44.bak` instruction was superseded by the
  primary front's deliberate placement of this state on both sides at
  07:38:20; overwriting it would have destroyed their banked work. Intent of
  the restore clause (no losing kernel left in the tree) is satisfied: the
  losing configs exist only behind non-default `-D` flags. Byte-exact B44
  remains available at `/tmp/frontD5/capstone_kernels_tcgen05.cu.B44.bak`
  (md5 `898fe01d...` == local HEAD f4cab652).
- This report (`code/docs/gb300-capstone-kstage-deepen.md`) is Front D5b's
  independent evidence; nothing was committed to git by this front.

## 8. Evidence index (pod, /tmp/frontD5b/)

- `01_refcap_b44.{cmd,stdout,stderr,exit}`, `ref_B44.pt`, `refcap_b44.py` —
  B44 reference + determinism.
- `02_harness_d5_run1.{cmd,stdout,stderr,exit,results.json}` — D5-state
  harness, verification PASSED, 106.19x.
- `03_ab_kstage.{cmd,stdout,stderr,exit}`, `ab_kstage.py`,
  `ab_kstage_results.json` — bit-identity + interleaved 3-arm timing.
- `04_ncu_b44.{cmd,stdout,stderr,exit}`, `ncu_b44_bk64_s4.ncu-rep`,
  `ncu_b44_raw.csv`, `ncu_b44_details.txt` — B44 ncu.
- `05_ncu_d5.{cmd,stdout,stderr,exit}`, `ncu_d5_bk32_s8.ncu-rep`,
  `ncu_d5_raw.csv`, `ncu_d5_details.txt` — D5 ncu.
- `06_ncu_diff.stdout`, `ncu_diff.py` — side-by-side metric diff.
- `07_harness_final_state.{cmd,stdout,stderr,exit}` — shipped-state
  validation, verification PASSED, 143.85x.
- `capstone_kernels_tcgen05.cu.D5` — archived working-copy source
  (md5 `aa013923...`, the shipped state). Note: harness run 02 built the
  pre-swap source (md5 `f4bf058b...`, defaults bK32/S8), whose bytes were
  overwritten by the primary front at 07:38:20 and are not archived; its
  compiled kernel config is exactly the D5 arm
  (`-DAISP_TCGEN05_KBLOCKS=2 -DAISP_TCGEN05_STAGES=8` on `aa013923`), which
  is archived, bit-identity-verified, and timed in sec 3. All timed/profiled
  arms pin `-D` flags and are reproducible from the archived source.

## 9. Next lever

**Wave-quantization fill.** The CTA1 kernel launches a (16,8) = 128-block grid
at 1 CTA/SM on a 152-SM GB300 (`torch.cuda.get_device_properties`:
`multi_processor_count=152`, this session): 24 SMs (15.8%) sit idle for the
whole kernel, and CTA2's 64x2 clusters have the same fill. A persistent /
stream-K schedule (or a finer N-tile trading per-tile efficiency for a second
wave) is the only structural >=10% headroom visible in this kernel's ncu
profile — tensor duty, stalls, and smem are otherwise flat between arms, and
the byte path is nowhere near DRAM-bound (4.7% DRAM throughput). EV up to
~15%, needs measurement.
