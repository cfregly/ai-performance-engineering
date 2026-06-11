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

---

# E5 verdict (2026-06-11): B-multicast (cluster_m=2) is an HONEST NEGATIVE — tie within noise, mechanism profiled

Measurement-only session (Front E5, GPU 2, pod `aisp-gb300-runall`). Files
reconciled: pod `tcgen05_dual_cta.cu` / `tcgen05_loader.py` /
`bench_dual_cta.py` md5-match the committed c045227b state (pod copies
backed up at `/tmp/frontE5/`). Sanity: plain dual (256,2,cm=1) measured
904.4us — inside the 840-915us window.

## Interleaved sweep (8 reps x (warmup=5, iters=20), round-robin, 8192^3 FP16)

| arm | median | TFLOPS | %SoL | reps [min..max] us |
|---|---|---|---|---|
| cuBLAS (target) | 602.1 us | 1826.1 | 48.7% | [588.1..607.0] |
| cluster (incumbent) | 1082.8 us | 1015.5 | 27.1% | [1012.3..1120.9] |
| dual_cta (256,2,cm=1) plain | 929.5 us | 1182.9 | 31.5% | [918.0..960.6] |
| **dual_cta (256,2,cm=2) mcast** | **919.0 us** | **1196.5** | **31.9%** | [889.7..940.4] |
| dual_cta (256,2,cm=4) | 949.7 us | 1157.7 | 30.9% | [932.1..972.1] |
| dual_cta (128,3,cm=2) | 1211.8 us | 907.4 | 24.2% | [1185.7..1257.6] |

Paired per-rep A/B (12 alternating reps, plain vs cm=2, cuBLAS drift
reference): **mcast/plain median ratio 1.0065 (mcast 0.65% SLOWER), mcast
wins 3/12 pairs**. Coldest rep reproduced the B37 bank: plain 843.2us vs
mcast 847.2us (cuBLAS drifted 597.8 -> 645.9us across the run = the known
5-9% thermal climb). E3's orphaned pre-lapse sweep (`/tmp/frontE3/
sweep_e3.log`) had shown cm=2 +2.6%; combined evidence: **a tie inside the
thermal-noise band. cm=2 does not clear the 35.0% plain-dual bank, and
cm=4 is strictly worse.** (256,3,*) does not build by design: 3x48KB
stages trip the `sizeof(SharedStorageT) <= 110KB` 2-CTA/SM static_assert.

## ncu mechanism: why multicast cannot win here (full set, skip 3, 1 launch)

| metric | plain (256,2,cm=1) | mcast (256,2,cm=2) |
|---|---|---|
| Duration | 789.2 us | 806.9 us |
| **long_scoreboard (warp-cycles/issue)** | **28.25** | **28.04 (-0.8%, unchanged)** |
| L2 sectors srcunit_tex | 278.9M | 275.5M (-1.2%) |
| DRAM throughput (of peak) | 12.5% | 12.5% |
| Tensor pipe (of elapsed) | 61.8% | 60.5% |
| Achieved warps_active | 12.09% (2 CTAs/SM) | 12.09% (2 CTAs/SM) |
| Block Limit smem / regs | 2 / 2 | 2 / 2 (clustering costs no occupancy) |

The premise of the lever was "halve B-side L2->SM traffic to relieve
long_scoreboard". The profile falsifies the premise, not the
implementation: the multicast IS active (cluster_dim 2, MULTICAST atom in
the kernel signature) and costs no occupancy, but L2 sector traffic drops
only 1.2% — the plain kernel's duplicate B reads were already absorbed by
L2 hits across cluster-adjacent CTAs — and DRAM sits at 12.5% of peak.
The stall is TMA **latency** serialization on the single producer/consumer
warp 0 (wait full_barrier -> 4 MMAs -> commit, 2 stages deep), which
multicast does not shorten; cluster_sync + multicast-commit fan-out add
back the little it saves. Bandwidth was never the binding resource at
(256,2): the lever is config-un-winnable, deepening it is pointless.

## Harness verify gates (3/3 PASSED, `verification passed: true`)

```
AISP_TCGEN05_VARIANT=dual_cta AISP_DUAL_CLUSTER_M=2 (multicast selected):
  verification PASSED, speedup 2.6703x   (run 20260611_061106)
AISP_TCGEN05_VARIANT=dual_cta (plain dual regression):
  verification PASSED, speedup 2.6710x   (run 20260611_061310, gate2)
default (cluster) regression:
  verification PASSED, speedup 2.4248x   (run gate3; 2.329x contract intact)
```

One transient first-attempt failure on gate 1 (harness TIMING
CROSS-VALIDATION: CUDA event 0.889ms vs wall 3.256ms, a wall-clock-noise
trip, not a kernel fault); immediate re-run passed clean. Logs:
`/tmp/frontE5/gate{1,2,3}.log`, sweep + paired A/B at
`/tmp/frontE5/{sweep_e5.log,ab_pair.log}`, ncu reports at
`/tmp/frontE5/ncu_{plain,mcast}.ncu-rep`.

## Defaults recommendation

**Keep the c045227b loader defaults — no revert.** The defaults are
(tile_n=256, stages=2, **cluster_m=1**): the committed default path is the
plain dual-CTA winner, re-confirmed best non-cuBLAS arm this session. The
cm=2 code stays as a measured-negative comparison arm (bench default
configs include it); it is correct, verify-clean, and costs nothing when
not selected.

## Named next lever

**2-SM UMMA pair (cta_group=2 / SM100_MMA_F16BF16_2x1SM_SS) on the
dual-CTA footprint.** ncu says the kernel is latency/issue-bound, not
bandwidth-bound (DRAM 12.5%, long_scoreboard 73% of stall at 8 active
warps/SM) — so spend the cluster on what GB300 actually accelerates:
fusing the CTA pair into one 256-wide MMA halves instruction/barrier
traffic per fed byte and is the cuBLAS-shaped path, vs. re-cutting
TMA traffic that L2 already dedups. Smem-feasible stage-deepening at
(256,2) is exhausted (stage 3 needs 144KB > 110KB cap), so the warp-
specialized producer + 2-SM MMA rewrite is the only remaining
structural lever toward the 48.7% cuBLAS ceiling.

---

# E4 verdict (2026-06-11): independent replication — cluster_m=2 B-multicast CONFIRMED HONEST NEGATIVE (tie within noise) + co-tenancy caveat

Front E4 ran the same measurement mission as E5 **concurrently and unaware**
(both fronts benched GPU 2 in the 06:04–06:13 UTC window: E5's sweep/ncu/
gates overlapped E4's sweep and focused A/Bs — exactly the same-GPU
co-tenancy this runbook bans). Detected post-hoc from `/tmp/frontE5/` mtimes
vs E4 artifact run-ids interleaving minute-by-minute. Everything marked
CLEAN below was re-measured after E5 finished, with GPU 2 verified idle
(`nvidia-smi -i 2`, no compute apps). The clean re-runs confirm the E5
verdict, so the contamination flipped no conclusion — but the overlapped-
window numbers in both sessions carry that caveat (E4's three overlapped
interleaves scattered cm2 +3.4% / tie / −2.2% — sign-flipping noise).

Reconciliation: pod `tcgen05_dual_cta.cu` / `tcgen05_loader.py` /
`bench_dual_cta.py` already md5-matched the committed c045227b state
(43b814c3 / 55d8f32c / 1a533beb) — no transfer, nothing overwritten.
Sanity: plain dual (256,2,cm=1) first read 893.5 us, inside the 840–915
window. All arms rel_err = 0.0 vs torch.matmul in every run.

## A/B — CLEAN (12 interleaved reps x (warmup=5, iters=20), quiet GPU 2)

| arm | median | TFLOPS | %SoL | reps [min..max] us |
|---|---|---|---|---|
| cuBLAS (target) | 609.4 us | 1804.2 | 48.1% | [599.8..632.8] |
| cluster (incumbent) | 1119.0 us | 982.6 | 26.2% | [980.9..1207.5] |
| dual_cta (256,2,cm=1) plain | **926.1 us** | 1187.3 | 31.7% | [884.1..970.3] |
| dual_cta (256,2,cm=2) mcast | 928.3 us | 1184.5 | 31.6% | [903.8..950.2] |

**Plain vs mcast: 0.24% apart, ranges overlap — a dead tie**, replicating
E5's paired read (mcast +0.65%) with independent code paths. Sweep arms
(6-rep interleave, overlapped window, directionally consistent with E5):
cm=4 950.2 us (loses ~2–5%), (128,3,cm=2) 1188.0 us (loses ~22%).
(256,3,*) does not build **by design** — 3x48KB stages trip the
`sizeof(SharedStorageT) <= 110KB` static_assert (tcgen05_dual_cta.cu:397).
Node state note: incumbent (1119 us) and plain (926 us) both sit well off
their B37 banks (904–962 / 838 best) on a quiet GPU while cuBLAS held
600–616 us all session — hot-node drift hits the long-running custom
kernels hardest; the interleaved relative ordering is the meaningful read.

## ncu — CLEAN (skip 4, 1 launch each, quiet GPU 2)

| metric | plain (256,2,cm=1) | mcast (256,2,cm=2) |
|---|---|---|
| Duration | 801.1 us | 808.0 us (+0.9%) |
| **long_scoreboard (warp-cycles/issued of total)** | **28.5 / 39.2 (72.7%)** | **28.0 / 38.7 (72.4%) — unchanged** |
| DRAM throughput (of peak) | 21.7% | 21.6% |
| L2 throughput | 45.1% | 44.5% |
| Compute (SM) throughput | 64.8% | 63.8% |
| Block Limit smem / regs | 2 / 2 | 2 / 2 |
| Achieved warps/SM (2 CTAs/SM) | 7.74 | 7.69 |

Same mechanism E5 profiled, independently reproduced: the multicast IS real
(`SM90_TMA_LOAD_MULTICAST` atom in the profiled signature, cluster size 2)
and 2 CTAs/SM co-residency survives clustering, but it buys nothing —
L2/DRAM traffic is essentially unchanged (L2 was already absorbing the
duplicate B reads) and the dominant long_scoreboard TMA-**latency** stall
(~73% of warp time) is untouched because the single producer/consumer
warp's wait→MMA→commit chain is serialized on latency, not starved of
bandwidth (DRAM 22%). The lever's premise is falsified at (256,2):
bandwidth was never the binding resource. E4 ncu durations (798–808 us
across 4 profiles incl. the overlapped pair) agree with E2's locked-clock
799.4 us bank, confirming ncu replay isolation kept profiles trustworthy.

## Harness verify gates (3/3 PASSED, `verification passed: true`)

```
AISP_TCGEN05_VARIANT=dual_cta AISP_DUAL_CLUSTER_M=2 (multicast):
  verification PASSED, speedup 2.6967x   (run 20260611_061340)
AISP_TCGEN05_VARIANT=dual_cta (plain regression):
  verification PASSED, speedup 2.6733x   (run 20260611_061425)
default (cluster) regression:
  verification PASSED, speedup 2.4150x   (run 20260611_061446; 2.329x contract intact)
```

E4 artifacts on the pod: `/tmp/e4_focus_ab.py`, `/tmp/e4_ncu_one.py`,
ncu reports `/tmp/e4_ncu_{plain,cm2}.ncu-rep` (overlapped window) and
`/tmp/e4_ncu_{plain,cm2}_clean.ncu-rep` (clean).

## Defaults recommendation (concurs with E5)

**Keep the c045227b defaults — no revert.** The committed default path is
already the plain winner (tile_n=256, stages=2, **cluster_m=1**); cm=2 is
opt-in only, correct, verify-clean, and now twice-measured neutral. Do not
flip the default to cm=2 (no win to take) and do not delete the mode (it
is the documented negative that closes this branch of the search tree).

## Named next lever (concurs with E5, plus one process lever)

1. **2-SM UMMA pair (cta_group=2 / SM100_MMA_F16BF16_2x1SM_SS) on the
   dual-CTA footprint** — the kernel is latency/issue-bound at 8 warps/SM;
   fuse the CTA pair into one 256-wide MMA instead of re-cutting TMA
   traffic that L2 already dedups. Stage-deepening at (256,2) is
   smem-impossible (static_assert), so this is the remaining structural
   path toward the ~48% cuBLAS ceiling.
2. **Process: single-GPU mutex for concurrent fronts.** E4+E5 silently
   co-measured GPU 2 for ~9 minutes; a `/tmp/gpu2.lock` flock (or
   per-front GPU assignment in the dispatch brief) costs nothing and
   protects every future A/B from sign-flipping contamination.

---

# U verdict (2026-06-11): 2-SM UMMA pair (cta_group::2) is a WIN — 2sm(128,3) beats plain dual 16/16 paired reps (median +4.6–6.6%), 33.5% SoL, shipped as a NEW selectable kernel

Implementation session (Front U, GPU 2, pod `aisp-gb300-runall`, 8192^3 FP16,
peak 3.75 PFLOPS). The B43/B44-named lever — fuse the dual-CTA pair into one
256-wide `SM100_MMA_F16BF16_2x1SM_SS` — landed as **`tcgen05_dual_cta_2sm.cu`**
(new file; the measured plain dual `tcgen05_dual_cta.cu` is byte-untouched,
md5 43b814c3). Correct on first build, rel_err = 0.0 vs torch.matmul on every
config and every run.

## Design facts (verified against CUTLASS 4.2.0 third_party + cute tutorial 04_mma_tma_2sm_sm100.cu)

- Cluster (2,1,1): blockIdx.x = (pair, v) interleaved; v = block_rank & 1;
  pair tile 256 x N (per-CTA output still 128 x N).
- Per-CTA operand residency: A = own 128-row M-half; **B = N/2 columns — the
  2x1SM atom SPLITS B across the pair** (ncu smem footprints prove it:
  64KB/CTA at (256,2), 72KB at (128,3) -> Block Limit Shared Mem 3, where
  full-N B would give 96KB -> limit 2; the mbarrier tx-byte accounting
  triple-confirms it — an over-expect would never complete. The tutorial's
  printed comments suggesting full-N B per CTA are stale). C = own 128 rows
  x N in TMEM (N cols/CTA). So per-CTA B traffic HALVES vs the plain dual.
- TMA: `SM100_TMA_2SM_LOAD` issued by BOTH CTAs into their own smem with
  their OWN `full_barrier[stage]` handle; hardware redirects every mbarrier
  arrival to the EVEN CTA's barrier (`Sm100MmaPeerBitMask`), so the leader's
  barrier counts pair bytes 2x(A-half + B-half). Only the leader runs
  `set_barrier_transaction_bytes` (peer completions before leader expect_tx
  are legal: mbarrier tx-count is transiently negative).
- Leader-only MMA (`gemm()` under `if (leader_cta)`, fma elects one lane);
  per-stage `umma_arrive_multicast_2x1SM` (tcgen05.commit.cta_group::2,
  mask 0b11) releases BOTH CTAs' `empty_barrier[stage]` (init count 1). The
  odd CTA is a pure TMA producer free-running ahead, bounded by empties —
  whole-CTA role assignment (B44 ncu-replay trap respected: `--set full`
  profiled clean, 41 passes, both launches).
- TMEM: `cute::TMEM::Allocator2Sm` pair-collective alloc of N cols (same
  base in both SMs' TMEM) + IMMEDIATE permit release -> co-resident CTAs of
  other pairs can allocate.
- Final multicast commit on `mma_barrier` + both-CTA empty-drain before
  epilogue (no in-flight remote arrival when any CTA exits).

## Interleaved A/B (8 reps x (warmup=5, iters=20), round-robin, 8192^3 FP16, GPU 2)

Run `ab_u_full.log` (11 arms, identical arms to the E5 sweep + 2sm):

| arm | median | TFLOPS | %SoL | reps [min..max] us |
|---|---|---|---|---|
| cuBLAS (target) | 629.5 us | 1746.7 | 46.6% | [595.1..638.4] |
| cluster (incumbent) | 1162.0 us | 946.2 | 25.2% | [985.2..1222.5] |
| dual_cta (256,2,cm=1) plain | 931.2 us | 1180.7 | 31.5% | [922.6..973.0] |
| dual_cta (256,2,cm=2) | 915.6 us | 1200.9 | 32.0% | [895.7..931.2] |
| **dual_2sm (128,3)** | **874.9 us** | **1256.8** | **33.5%** | [862.9..882.0] |
| dual_2sm (256,2) | 897.3 us | 1225.3 | 32.7% | [880.1..911.5] |
| dual_2sm (128,2) | 971.6 us | 1131.7 | 30.2% | [956.5..998.2] |

Paired per-rep vs same-session plain dual (256,2): **2sm(128,3) wins 16/16
reps across two independent 8-rep runs; median speedup 1.0458x (run 1,
`ab_u.log`) and 1.0656x (run 2, `ab_u_full.log`), range [1.0053..1.1032]**.
Two soak re-runs of the full 11-arm sweep reproduce: 875.3 us / 873.5 us
median (33.5/33.6% SoL), best-arm both times. NOT a breakthrough (>=44%):
33.5% vs the 46.6–48.2% same-session cuBLAS ceiling.

Note the inversion: n=128 is a measured LOSS in plain dual (1165 us) but the
WINNER in 2SM form — the MMA stays 256-wide (pair tile 256x128), so halving
N no longer halves per-instruction work; it halves the epilogue register
footprint instead, unlocking a third CTA per SM (below).

## ncu mechanism (--set full, --launch-skip 5 --launch-count 2, locked clocks)

| metric | plain (256,2) | **2sm (128,3)** | 2sm (256,2) |
|---|---|---|---|
| Duration | 789.8 us | 786.5 us | 795.7 us |
| Registers/thread | 255 | **152** | 255 |
| Block Limit regs / smem | 2 / 2 | **3 / 3** | 2 / 3 |
| Theoretical / Achieved occupancy | 12.5 / 12.12% | **18.75 / 18.30% (3 CTAs/SM)** | 12.5 / 12.23% |
| long_scoreboard (warp-cycles/issued) | 28.51 | **23.47 (-18%)** | 34.31 |
| Warp cycles/issued instr | 38.82 | **30.83** | 44.50 |
| Issued warp/scheduler | 0.05 | **0.09 (+80%)** | 0.04 |
| Executed IPC | 0.20 | 0.37 | 0.17 |
| Tensor pipe active (of elapsed) | 49.46% | **54.01%** | — |
| DRAM throughput | 1.75 TB/s (22.0%) | 2.17–2.27 TB/s (27.4%) | 1.76 TB/s |
| L2 read sectors (srcunit_tex) | 257.5M | 376.8M (A re-reads, n=128 tiling) | 242.0M |
| launch cluster size | — | 2 | 2 |

Mechanism of the win at (128,3): the N=128 epilogue holds 128 fp32/thread
(152 regs vs 255) and 72KB smem/CTA -> Block Limits 3/3 -> **three CTAs/SM**
(TMEM 3x128 = 384 of 512 cols; `__launch_bounds__(128,2)` is a minimum, not
a cap), on top of the structural 2SM gains (one MMA stream per SM pair,
halved per-CTA B bytes, pure-producer peer). Issue rate +80%, stall
cycles/instr -21%, tensor-pipe duty +4.6pts.

Mechanism of the (256,2) 2SM tie: the 256-col epilogue pins 255 regs ->
Block Limit Registers 2, and TMEM 2x256 = 512 is an exact fit -> same
2 CTAs/SM as plain; halved MMA issue alone does not move the duration
(long_scoreboard per-issued rises to 34.3 by denominator effect). The lever
pays through occupancy + traffic, not instruction count per se.

## One-off hang (flagged, unreproduced — treat 2SM sweeps with a watchdog)

The FIRST 8-rep full-sweep attempt hung in a warmup launch (~21 min,
py-spy: `torch.cuda.synchronize`, GPU pegged at 100%, single process on
GPU 2, no co-tenancy). Killed and chased: 4x40-launch isolated stress per
config PASS, two instrumented 8-rep interleaves PASS, two exact-command
soaks with on-hang autopsy armed PASS (`soak_run{1,2}.log`) — ~4,000
subsequent 2SM launches clean, rel_err 0.0 everywhere. Unproven suspicion:
transient TMEM pair-alloc (tcgen05.alloc.cta_group::2 needs a COMMON free
base across both SMs of a pair) slot-mismatch under cross-pair churn —
(256,2)'s 2x256 exact fit is the natural suspect; the (128,3) default has
4-slot headroom. Action: defaults avoid the exact-fit config, and any long
unattended 2SM sweep should run under `timeout -s KILL`.

## Harness verify gates (3/3 PASSED, `verification: {passed: true}`)

```
AISP_TCGEN05_VARIANT=dual_cta_2sm (loader defaults 128,3):
  verification PASSED, speedup 2.5539x   (run 20260611_081140)
AISP_TCGEN05_VARIANT=dual_cta (plain dual regression):
  verification PASSED, speedup 2.6213x   (run 20260611_081242)
default (cluster) regression:
  verification PASSED, speedup 2.3675x   (run 20260611_081307; 2.329x contract intact)
re-gate after final comment-only .cu doc fix (rebuild, same codegen):
  verification PASSED, speedup 2.52x     (run 20260611_082149; 4-rep A/B re-check
  879.1us 2sm(128,3) vs 916.7us plain — win intact, ab_u_rebuild.log)
```

## Files (pod, md5)

```
NEW      labs/custom_vs_cublas/tcgen05_dual_cta_2sm.cu       9727eb7dbfe5c8df5c35bcf9f037c152
CHANGED  labs/custom_vs_cublas/tcgen05_loader.py             d8295ae153d95f3e0943ca890158c110
         (adds _load_tcgen05_dual_cta_2sm_module / load_tcgen05_dual_cta_2sm_module /
          matmul_tcgen05_dual_cta_2sm; env AISP_DUAL2SM_TILE_N=128, AISP_DUAL2SM_STAGES=3)
CHANGED  labs/custom_vs_cublas/bench_dual_cta.py              70982a094bbe602fe494cacdfb7abf7e
         (adds dual_2sm arms: default [(128,3),(256,2)], sweep adds (128,2))
CHANGED  labs/custom_vs_cublas/optimized_tcgen05_matmul.py    15a9a1828b5046d51581f7b83021cbb4
         (adds AISP_TCGEN05_VARIANT=dual_cta_2sm branch)
UNTOUCHED labs/custom_vs_cublas/tcgen05_dual_cta.cu           43b814c3f1ba7b815d7a0195941c1a18
Originals backed up at /tmp/frontU/ (pod). Logs: /tmp/frontU/{ab_u.log,
ab_u_full.log,sweep_u2.log,soak_run1.log,soak_run2.log}; ncu reports:
/tmp/frontU/ncu_{plain_n256s2,2sm_n128s3,2sm_n256s2}.ncu-rep. Nothing committed.
```

## Named next lever

**Warp-specialized leader + A-multicast on the 2SM footprint.** ncu says
2sm(128,3) is STILL TMA-latency-bound (long_scoreboard 23.5 of 30.8 = 76%
of stalls; DRAM 27%): (a) the leader's warp 0 still serializes
empty-wait -> TMA-issue -> full-wait -> MMA -> commit; split producer and
consumer across two warps (whole-warp roles, replay-safe) to overlap the
leader's own TMA issue with its MMA stream; (b) n=128 tiling doubled A
re-reads (L2 srcunit_tex 376.8M vs 257.5M sectors) — a (2,2,1) cluster with
`SM100_TMA_2SM_LOAD_MULTICAST` on A across the N-mode halves that, and
unlike E5's falsified B-multicast premise, this targets traffic that is NOT
already L2-deduped (it shows up in the sector counts). Both are incremental
on tcgen05_dual_cta_2sm.cu, not a rewrite.

---

# V2 verdict (2026-06-11): warp-split leader is a WIN (12/12 paired, 1.073x hot-median, new default); A-multicast is an HONEST NEGATIVE with the traffic premise PROVEN TRUE but latency-coupling cost 9% — and its V-front deadlock was the B53 expect-tx trap, now FIXED

Continuation session (Front V2, GPU 2, pod `aisp-gb300-runall`, 8192^3 FP16).
Front V died at its turn limit waiting on the first correctness build of its
two B49 levers; this session recovered its in-flight state, finished both
levers, and root-caused + fixed a real deadlock in lever (b).

## Predecessor-state disposition: SALVAGED (full implementation, one fatal bug)

/tmp/frontV/backup matched the B49 reference md5s exactly
(tcgen05_dual_cta_2sm.cu 9727eb7d); the live tree carried a complete
flag-gated implementation of BOTH levers (DUAL2SM_WARP_SPLIT /
DUAL2SM_AMCAST, env AISP_DUAL2SM_WARP_SPLIT / AISP_DUAL2SM_AMCAST, default
base config untouched). Code review against CUTLASS 4.2.0
sm100_mma_warpspecialized.hpp + tutorial 04_mma_tma_2sm found the design
canonical (per-CTA multicast issue, per-coord create_tma_multicast_mask<2>,
empty-barrier count = kClusterN, commit mask 0xF) EXCEPT one line — and that
line was the exact trap B53 warned about:

```
WRONG (V-front): tma_bytes = 2 * (kClusterN * sizeof(tAsA_0) + sizeof(tBsB_0));
RIGHT (V2):      tma_bytes = 2 * (sizeof(tAsA_0) + sizeof(tBsB_0));
```

Under cta_group::2 multicast the leader full-barrier counts each issue's
slice ONCE per destination pair — NO participant multiplier (tutorial 04:
`size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))`). The
kClusterN multiplier over-expects by A-half/stage, the barrier never fires,
and the kernel hangs: that is precisely the 25-min GPU-pegged hang that ate
Front V's correctness run (reproduced down to a SINGLE (2,2,1) cluster at
256x256x2048, then fixed: rel_err 0.0 at every size after the one-line fix;
kClusterN=1 value is unchanged, so the incumbent is byte-identical).
All four variants now verify: (128,3) ws/amc = 00, 10, 01, 11 all
rel_err 0.00e+00 vs torch.matmul at 4096^3.

## Lever (a) warp-split leader — WIN, shipped as the new default

Whole-warp producer (warp 0, both CTAs: empty-wait + TMA) / consumer
(warp 1, leader CTA: full-wait + MMA + commit) split of the previously
serialized leader chain. B44 whole-warp-role trap respected; ncu --set full
replays clean.

Interleaved 8-rep round-robin (8 arms, `ab_full.log`, quiet GPU 2):

| arm | median | TFLOPS | %SoL |
|---|---|---|---|
| cuBLAS (target, same session) | 614.6 us | 1788.9 | 47.7% |
| cluster (incumbent) | 1141.0 us | 963.6 | 25.7% |
| dual_cta (256,2,cm=1) plain | 936.4 us | 1174.2 | 31.3% |
| dual_2sm (128,3) base | 892.3 us | 1232.2 | 32.9% |
| **dual_2sm (128,3) ws=1** | **875.3 us** | **1256.2** | **33.5%** |
| dual_2sm (128,3) amc=1 | 1147.4 us | 958.3 | 25.6% |
| dual_2sm (128,3) ws=1 amc=1 | 933.8 us | 1177.5 | 31.4% |

Paired per-rep (the thermally honest instrument; round-robin medians
understate because slow arms give the base cooling breaks): 10/10 wins
order-fixed (median 1.0784x) and **12/12 wins order-ALTERNATED with thermal
pre-soak, median speedup 1.0725x** (base_med 989.8us vs ws_med 920.7us,
same-session cuBLAS 672.6us hot). NOT a breakthrough (33.5% << 40%); a
solid incremental WIN on the same mechanism axis as B49 predicted.

ncu mechanism (--set full, skip 5, 2 launches, means):
duration 786.9 -> 755.4us (-4.0%); long_scoreboard/issued 23.65 -> 22.34
(-5.5%); tensor-pipe-active-of-elapsed 66.7% -> 70.9% (+4.2pts); IPC 0.374
-> 0.392; L2 srcunit_tex read sectors 375.0M -> 340.5M (-9%); DRAM read
1.53 -> 1.38GB. The free-running producer keeps all 3 stages in flight, so
fewer stalled cycles AND fewer L2 re-fetches. Occupancy unchanged
(18.75/18.2%, 3 CTAs/SM, 152 regs) — this lever pays purely through overlap.

## Lever (b) A-multicast (2,2,1) — HONEST NEGATIVE (premise TRUE, mechanism wrong axis)

B53-style pre-check, answered with this session's ncu: A's duplicate
loads are genuinely NOT hardware-merged in the (2,1,1) shape — L2
srcunit_tex 375.0M re-measured (B49: 376.8M), and the (2,2,1)
SM100_TMA_2SM_LOAD_MULTICAST on A really does eliminate them: 261.4M
sectors (-30%, right at plain-dual's 257.5M level). The traffic premise
was TRUE — unlike E5's falsified B-multicast — and the implementation is
correct (rel_err 0.0 everywhere after the expect-tx fix).

It still LOSES 9% (ncu 786.9 -> 857.8us; bench 892.3 -> 1147.4us in
round-robin, -22% there because the (2,2,1) shape also tanks under thermal
load): the kernel is LATENCY-bound, not traffic-bound (DRAM 28%), so
removing L2 sectors buys nothing, while the cross-pair coupling costs
plenty: every stage's smem reuse now waits on the SLOWER of two pairs'
MMA streams (empty barrier count 2, commits from both pair-leaders),
tensor-pipe duty drops 66.7% -> 55.6% (-11pts), long_scoreboard RISES
23.65 -> 26.84, and DRAM bytes read go UP 1.53 -> 1.83GB (the interleaved
4-CTA working set worsens L2 residency). Lock-step pipeline serialization
across pairs > traffic savings, mechanism closed. The flag stays in the
tree (correct, selectable, OFF by default) as the measured record.

## Harness verify gates (3/3 PASSED)

```
AISP_TCGEN05_VARIANT=dual_cta_2sm (loader defaults 128,3,ws=1):
  verification passed: true, 0 failed (run 20260611_164022, speedup 2.76x)
AISP_TCGEN05_VARIANT=dual_cta (plain dual regression):
  verification passed: true, 0 failed (run 20260611_164034)
default (cluster) regression:
  verification passed: true, 0 failed (run 20260611_164048)
(speedup figures 6.45x/11.94x on gates 2-3 are a throttled-baseline
artifact — harness logged baseline launches at SM=120MHz; pass/fail
gates are the signal, all green.)
```

## Files (pod, md5)

```
CHANGED  labs/custom_vs_cublas/tcgen05_dual_cta_2sm.cu   ac866efe2c96ed07b8c6db2a5536100d
         (V-front: DUAL2SM_WARP_SPLIT + DUAL2SM_AMCAST flag-gated levers;
          V2: expect-tx fix — drop kClusterN multiplier in tma_bytes)
CHANGED  labs/custom_vs_cublas/tcgen05_loader.py         da0bbda48279f0a8817a79ecc2da711a
         (V-front: warp_split/amcast plumbing; V2: AISP_DUAL2SM_WARP_SPLIT
          default flips to 1; base remains AISP_DUAL2SM_WARP_SPLIT=0)
CHANGED  labs/custom_vs_cublas/bench_dual_cta.py         5ad5ea81aa5ec2c22b61bb2673878384
         (V-front: --twosm N,S[,WS[,AMC]] explicit arms; unchanged by V2)
Backups: /tmp/frontV/backup (B49 originals), /tmp/frontV2/backup (V-front
in-flight state). Logs: /tmp/frontV2/{ab_ws,ab_full}.log, paired runs in
session transcript; ncu: /tmp/frontV2/ncu_{base,ws,amc}.ncu-rep. Nothing
committed.
```

## Named next lever

**Occupancy x pipeline-depth frontier under warp-split: tile_n=64.**
After ws=1, long_scoreboard is STILL the dominant stall (22.3 of 29.9 =
75%) and the producer is bounded by kStages=3 in-flight tiles (72KB/CTA at
n=128 caps smem at 3 CTAs/SM). n=64 halves both the epilogue register
footprint (below 152) and per-stage smem (~12KB/stage-CTA): stages 5-6 AND
a 4th CTA/SM (TMEM 4x64=256 of 512 cols) become simultaneously reachable —
deeper TMA in-flight to bury the latency plus more resident warps to absorb
it. Risk: 256x64 MMA efficiency; the B49 inversion (n=128 won in 2SM form
where it lost in plain) says small-N + 2SM is repeatedly underestimated.
Fallback axis: kTileK=128 (double the TMA box, halve barrier round-trips
per byte) at n=128 s=2.

---

# V3 verdict (2026-06-11): tile_n=64 is an HONEST NEGATIVE on every axis — the 4th (even 5th) CTA/SM is REACHED and still loses 30-38%; kTileK=128 fallback also negative (-15%); the B57 incumbent (128,3,ws=1) stands

Front V3 session (GPU 2, pod `aisp-gb300-runall`, 8192^3 FP16, base 2f7e30f9
+ uncommitted B57 state, incumbent md5s verified before work). B57's named
lever was tile_n=64 ("simultaneously opens a 4th CTA/SM and stages 5-6"),
with kTileK=128 as the fallback axis. Both were implemented flag-gated,
measured, and falsified. One forecast correction up front: B57 estimated
~12KB/stage-CTA at n=64, but the A-half is 16KB/stage at ANY tile_n (128
rows x 64 K x 2B; only the B-half halves to 4KB), so n=64 is 20KB/stage —
stages 5-6 and the 4th CTA are NOT simultaneously reachable. The session
measured the whole reachable frontier anyway.

## What was added (all flag-gated, defaults byte-equivalent to B57)

`tcgen05_dual_cta_2sm.cu` + `tcgen05_loader.py`:
- `DUAL2SM_STAGES` extended 2..6 (stage-array initializers for s=5,6).
- `DUAL2SM_MIN_BLOCKS` (env `AISP_DUAL2SM_MIN_BLOCKS`, default 2):
  `__launch_bounds__(128, MB)` hint; 4/5 cap regs at 128/102 for the
  4th/5th co-resident CTA.
- `DUAL2SM_TILE_K` (env `AISP_DUAL2SM_TILE_K`, default 64): K-extent per
  stage; 128 doubles the TMA box and halves barrier round-trips per byte
  (`bK = atom_K * Int<kTileK/16>`).
ptxas: every n=64 config compiles at 88 regs/thread (vs 152 at n=128), no
spills at any MIN_BLOCKS; (64,6) is statically excluded by the 110KB smem
guard (6 x 20KB = 120KB also exceeds 227KB/2 — it would be 1 CTA/SM,
strictly dominated).

## Config table (8192^3, 6-rep round-robin medians, GPU 2, all rel_err 0.00e+00)

| config (n,s,ws,amc,mb,k) | us | TFLOPS | %SoL | CTAs/SM (ncu Block Limits) | binding limit |
|---|---|---|---|---|---|
| cuBLAS same-session | 639.1 | 1720.3 | 45.9 | - | - |
| **(128,3,1,0,2,64) incumbent** | **863.7** | **1273.1** | **33.95** | **3 (regs 3 / smem 3)** | regs+smem tie |
| (128,2,1,0,2,128) k128 fallback | 1008.9 | 1089.8 | 29.1 | 2 (smem 2, 96KB/CTA) | smem |
| (64,2,1,0,5,64) | 1223.5 | 898.7 | 24.0 | **5** (regs 5 / smem 5, occ 30.3%) | smem+regs tie |
| (64,2,1,0,4,64) | 1238.8 | 887.5 | 23.7 | 5 (same 88-reg binary) | smem+regs |
| (64,5,1,0,2,64) | 1380.7 | 796.3 | 21.2 | 2 (smem, 100KB/CTA) | smem |
| (64,3,1,0,2,64) control | 1394.3 | 788.6 | 21.0 | 3 (smem 3, 60KB/CTA) | smem |
| (64,4,1,0,2,64) | 1444.6 | 761.1 | 20.3 | 2 (smem, 80KB/CTA) | smem |

Paired order-alternated A/B of the best challenger (k128) vs the incumbent
(8 reps, thermal pre-soak, cuBLAS probe each rep, `ab_k128.log`):
challenger wins 0/8, median speedup 0.8306x (17% slower hot) — inc
hot-paired median 891.4us vs chal 1073.3us, same-session cuBLAS 641.2us.
The n=64 arms were not paired-tested: their round-robin ranges
(1213-1466us) do not overlap the incumbent's (846-967us).

## ncu mechanism (--set full, skip 5, 2 launches, means)

| metric | inc (128,3) | (64,3) ctrl | (64,2,mb5) | (128,2,k128) |
|---|---|---|---|---|
| duration us | 748.8 | 1173.3 | 1075.2 | 862.7 |
| long_scoreboard /issue | 22.05 | 23.00 | **53.18** | 25.76 |
| warp latency /inst issued | 29.55 | 28.44 | 61.75 | 33.44 |
| tensor pipe % of elapsed | 71.6 | 40.6 | 44.2 | 57.7 |
| achieved occupancy % (warps/SM) | 18.2 (11.7) | 18.6 (11.9) | **30.3 (19.4)** | 12.1 (7.7) |
| L2 srcunit_tex read sectors | 344.6M | 579.9M | 560.3M | 341.9M |
| DRAM bytes read GB | 1.37 | 2.92 | 2.54 | 2.06 |
| IPC | 0.39 | 0.41 | 0.31 | 0.23 |

Three separate falsifications:
1. **tile_n=64 per se (control (64,3), same 3 CTAs/SM, same stages):**
   tensor-pipe duty craters 71.6 -> 40.6%. n=64 halves the FLOPs per
   barrier round-trip while the A-half feed stays 16KB/stage, and the
   grid's n-tiles double (8192 CTAs, waves 18.0 vs 9.0) so A re-reads
   DOUBLE through L2 (sectors +68%, DRAM read +114%, L2 hit 77.0 -> 73.5%).
   Arithmetic intensity per fed byte halves — the exact inverse of the
   lever's premise. The 32-col B slices also halve TMA box width (B48
   delivery-efficiency trap), compounding per-byte cost.
2. **Occupancy axis (64,2,mb4/mb5): the 4th AND 5th CTA/SM genuinely
   arrive** (Block Limits 5/5/16/32, theoretical 31.25%, achieved 30.3% —
   the first >3 CTA/SM residency in this kernel family) and the config is
   the best of the n=64 family (+9% vs (64,3)) — but it still loses 38% to
   the incumbent: long_scoreboard per issue MORE THAN DOUBLES (22.1 ->
   53.2) because 10 concurrent TMA streams/SM (5 CTAs x 2 stages) contend
   for the same L2/DRAM path with doubled per-byte traffic. More resident
   warps absorb latency only if the latency stays constant; here added
   residency CREATES the latency it was meant to hide.
3. **kTileK=128 fallback (128,2,k128):** barrier round-trips per byte
   genuinely halve (64 vs 128 k-iters) and per-issue stalls stay near
   incumbent (25.8), but 98.4KB/CTA smem caps residency at 2 CTAs/SM
   (occ 12.1%, IPC 0.23): losing the 3rd CTA costs more than halved sync
   saves. DRAM read rises 1.37 -> 2.06GB (poorer cross-CTA L2 timing, hit
   rate 77.0 -> 68.0%).

Conclusion: the (128,3) point sits at a genuine local optimum of the
(tile_n, stages, CTAs/SM, tile_k) frontier — every single-axis move from
it that fits the hardware budgets was now measured and loses. The
long_scoreboard stall at 22/29.6 issue-cycles (75%) is NOT an
occupancy-curable artifact; it is the latency floor of the
single-leader-MMA + dual-producer-TMA structure at 3 CTAs/SM.

## Harness verify gates (3/3 PASSED, `--profile none`)

```
AISP_TCGEN05_VARIANT=dual_cta_2sm (loader defaults 128,3,ws=1,mb=2,k=64):
  verification passed: true, 0 failed (ts 17:28:15, best_speedup 2.7974x)
AISP_TCGEN05_VARIANT=dual_cta (plain dual regression):
  verification passed: true, 0 failed (ts 17:28:35, best_speedup 2.7656x)
AISP_TCGEN05_VARIANT=cluster (default regression):
  verification passed: true, 0 failed (ts 17:28:52, best_speedup 2.4448x;
  2.329x contract intact)
```

Procedural note: a first gate pass ran at the harness's profile_minimal
DEFAULT and gate 3 (cluster) hung in the post-timing profiling phase
(harness self-reported "no progress for 328s"; python SIGKILLed, GPU 2
clean). Gates 1-2 passed in that mode too (gate-2 json: passed, 2.742x).
The quoted 3/3 above is the clean `--profile none` re-run, matching the
V2 gate procedure. Verdict snapshots:
`/tmp/frontV3/gate{1,2,3}_{2sm,dual,cluster}_results.json`.

## Files (pod, md5)

```
CHANGED  labs/custom_vs_cublas/tcgen05_dual_cta_2sm.cu  ae3f0a34730f0d79347ad6f807a75694
         (V3: DUAL2SM_STAGES 2..6, DUAL2SM_MIN_BLOCKS launch-bounds knob,
          DUAL2SM_TILE_K=64|128; defaults byte-equivalent to B57 — the
          incumbent rebuilt under V3 source reproduces 152 regs and the
          863.7us/33.95% SoL round-robin median)
CHANGED  labs/custom_vs_cublas/tcgen05_loader.py        83744a2d1e925186f69bfa9cb29c89df
         (V3: min_blocks/tile_k params + AISP_DUAL2SM_MIN_BLOCKS /
          AISP_DUAL2SM_TILE_K env plumbing; defaults UNCHANGED:
          128,3,ws=1,amc=0,mb=2,k=64 — the B57 incumbent stays the default)
UNCHANGED labs/custom_vs_cublas/bench_dual_cta.py       5ad5ea81aa5ec2c22b61bb2673878384
Backups: /tmp/frontV3/{tcgen05_dual_cta_2sm.cu,tcgen05_loader.py} (B57
originals, md5 ac86…/da0b…). Logs: /tmp/frontV3/{build_v3,sweep_v3,
ab_k128,gateN*}.log; ncu: /tmp/frontV3/ncu_{inc_n128s3,n64s3,n64s2mb5,
n128s2k128}.ncu-rep. Nothing committed.
```

## Named next lever

The occupancy/pipeline-shape configuration space on this kernel is now
EXHAUSTED (B49 tiles, B53/V2 multicast both axes, B57 warp-split, V3
tile_n=64/stages/min-blocks/tile_k — every frontier point measured). The
remaining 863.7 -> 639.1us gap to cuBLAS (33.9 -> 45.9% SoL) cannot come
from launch-shape knobs. The two structural candidates, in EV order:
1. **2-CTA output tiling with TMEM double-buffering (epilogue overlap):**
   the epilogue (TMEM->reg->gmem, ~all-thread) is serialized after the
   FULL k-loop per CTA; cuBLAS-class kernels split N into two TMEM
   halves and drain half while the MMA stream fills the other. TMEM
   2x128 of 512 cols leaves room at n=128 to double-buffer WITHIN the
   3-CTA footprint (3 x 256 = 768 > 512 fails; 2 x 256 fits at 2 CTAs —
   measure both).
2. **Persistent CTAs + tile rasterization (L2-locality scheduling):** the
   8192^3 grid re-reads A/B with hash-order CTA placement; a persistent
   grid with swizzled (m,n) walk is the standard cuBLAS L2-shaping tool
   and attacks the SAME DRAM/L2 numbers that just sank every V3 config —
   without touching the winning (128,3) shape.

# B-runbook V4 front (2026-06-11): TMEM double-buffered epilogue overlap (lever e) -- HONEST NEGATIVE

Mission: B60's named structural lever on `tcgen05_dual_cta_2sm.cu` -- drain
one TMEM accumulator buffer (t2r + store) while the MMA stream fills the
other, instead of the incumbent's fully-serialized post-k-loop epilogue.

## Which formulation the layout permits (stated before coding)

1. WITHIN-TILE k-tail split (drain N-half 1 while half 2's final MMAs run):
   structurally DEAD on this kernel. The 2x1SM atom
   (SM100_MMA_F16BF16_2x1SM_SS, 256 x TILE_N) computes BOTH N-halves of the
   pair tile per k-block instruction, and both halves are fed from the SAME
   smem stage ring -- the maximum k-stagger between halves is the in-flight
   stage count, so the overlap window is (kStages-1)/num_k_tiles = 2/128
   ~ 1.6% of the mainloop. Splitting the atom into two N/2 atoms does not
   change this (they would still share the stage ring).
2. CROSS-TILE TMEM double-buffering: the only real window, and the layout
   permits it cleanly at n=128: 2 buffers x 128 cols = 256 TMEM cols/CTA ->
   exactly 2 CTAs/SM (512-col TMEM); the incumbent's 3 CTAs/SM would need
   3 x 256 = 768 > 512 (the B60 TMEM math; impossible).
   One additional layout fact forces the implementation shape: the t2r
   drain is WARPGROUP-bound (TMEM subpartition w is only addressable by a
   warp of rank w within its warpgroup), so the drain cannot run on the
   parked warps 2-3 -- it needs a full SECOND warpgroup (256 threads/CTA).

## What was built (flag-gated; incumbent path untouched and default)

`DUAL2SM_EPI_OVERLAP=T` (0 = off/incumbent-byte-path; 2|4|8 = tiles per
CTA walked consecutively along n, grid_y / T):
- Warpgroup 0 = the B57 warp-split producer/consumer mainloop over the
  FLATTENED (tile, k) iteration space -- the smem stage ring crosses tile
  boundaries with NO per-tile prologue bubble. Warpgroup 1 (warps 4-7,
  BOTH CTAs) = dedicated epilogue draining TMEM buffer (t%2) + storing
  tile t while the consumer fills buffer ((t+1)%2).
- Per-buffer mma_barrier[2] (leader tcgen05.commit multicast to the pair);
  tmem_empty[2] on the leader (cluster-scope ClusterBarrier::arrive from
  both CTAs' epilogue warpgroups) gates buffer reuse at T>2; arrives are
  SKIPPED for the last kNumAccBufs tiles of the walk so no remote arrive
  can land after the target CTA's smem barriers die (B49/V2 lifetime trap).
- Chunked t2r drain (`DUAL2SM_EPI_ATOM`, default 32 cols -> 32-reg
  fragments): 256 threads x 2 CTAs/SM caps registers at 128/thread, so
  the incumbent's full-tile one-shot fragment does not fit. group_modes
  over the partition rest-modes makes the chunking atom-width-agnostic.

## A/B (paired, order-alternated, 8192^3, GPU 2, vs incumbent (128,3,ws=1))

ALL arms rel_err = 0.0 (incl. the T=4 buffer-reuse path: the cluster-scope
tmem_empty protocol is correctness-proven, not just the T=2 no-reuse case).

| challenger config      | inc median      | chal median     | paired sp | wins |
|------------------------|----------------:|----------------:|----------:|------|
| (128,3) e2 ea32        | 889.2us / 33.0% | 1144.1us / 25.6%|  0.7903   | 0/12 |
| (128,3) e4 ea32        |   (same-sess)   |    (same-sess)  |  0.7318   | 0/10 |
| (128,4) e2 ea32  BEST  | 929.2us / 31.6% |  980.3us / 29.9%|  0.9479   | 0/10 |
| (128,2) k128 e2 ea32   | 918.6us / 31.9% | 1034.1us / 28.4%|  0.8856   | 0/10 |
| (128,4) e2 ea16        | 924.8us / 31.7% |  978.0us / 30.0%|  0.9442   | 0/10 |

0/52 paired wins total. cuBLAS same-session 640-678us / 43-46% SoL (node
thermal drift; the paired order-alternated protocol absorbs it -- note the
incumbent itself swung 889->929us across sessions). Atom width 16 vs 32 is
flat (0.944 vs 0.948): the B55 re-sweep duty is done and the epilogue atom
is NOT the binder -- the drain is fully hidden, exactly as designed; the
loss is mainloop-side.

## Why it loses (ncu, --set full, skip 5 / count 2; vs V3 incumbent rep)

| metric                              | incumbent (128,3) | e2 (128,3) | e2 (128,4) |
|-------------------------------------|------------------:|-----------:|-----------:|
| duration (ncu clock)                |       748.8us     |  890-975us |  801-809us |
| tensor-pipe active (% of elapsed)   |         71.6%     |  49.8-53.8%|     66.9%  |
| long_scoreboard (cyc/issue)         |          22.0     |      28.5  |      27.6  |
| barrier stall (cyc/issue)           |          0.29     |      34.0  |      15.2  |
| warp latency (cyc/inst issued)      |          29.6     |      69.9  |      48.4  |
| occupancy limit (smem)              |     3 CTAs/SM     |  3 CTAs/SM |  2 CTAs/SM |
| warps_active (% of peak)            |         18.3%     |     34.3%  |     24.1%  |
| registers/thread                    |           152     |        46  |        46  |
| dram bytes read                     |        1.36GB     |    2.1GB   |    2.1GB   |
| waves (vs theoretical)              |          8.98     |      4.49  |      6.74  |

Three structural findings:
1. **The s=3 overlap pathology is a TMEM-vs-smem occupancy mismatch:** at
   72KB/CTA the hardware co-schedules a THIRD CTA per SM (smem limit 3),
   but TMEM (2x256 of 512 cols) only fits two -- the third spins inside
   tcgen05.alloc for a whole tile duration. ncu shows it directly:
   warps_active 34.3% (~2.75 CTAs resident, one useless), barrier stall
   0.29 -> 34.0 cyc/issue, tensor pipe 71.6 -> ~52%. LESSON (new, bankable):
   when TMEM binds occupancy, size the smem footprint to the TMEM-implied
   CTA count (s=4 = 96KB caps smem residency at exactly 2) or pay a
   spinning-allocator slot.
2. **At the clean 2-CTA point (s=4) the deficit is feed concurrency, not
   the epilogue:** tensor-pipe 66.9 vs 71.6%, long_scoreboard 27.6 vs 22.0
   -- one fewer independent TMA/MMA stream per SM exposes more TMA latency
   per warp. Registers are 46/thread (vs cap 128): no spill pressure; the
   drain itself is invisible in the A/B (atom width flat) -- the overlap
   mechanism WORKS, the occupancy it costs is simply worth more.
3. **The n-walk hurts L2:** DRAM read bytes rise 1.36 -> 2.1GB at equal L2
   sector counts -- 304 clusters walking 2-4 consecutive n-tiles reuse B
   worse across the SM array than 456 independent (m,n) CTAs. Walk length
   T=4 compounds it (0.73x): more serial tiles behind one producer stream,
   fewer concurrent distinct tiles for L2 sharing.

The decomposition: at 9 waves deep and 3 CTAs/SM, the incumbent's
serialized epilogue was ALREADY hidden by co-resident CTAs' mainloops --
the per-SM tensor pipe does not idle while one CTA drains, because two
other k-pipelines keep feeding it. Explicit intra-CTA overlap buys back
only what co-residency already provided, and pays 1/3 of the SM's
independent feed streams for it on a mainloop that is TMA-latency-bound
(B60: long_scoreboard 75% of issue-cycles).

## Verdict

HONEST NEGATIVE for lever e on this kernel shape: best overlap config
(128,4,e2) = 0.9479x, 0/52 paired wins across the 5-config space (stages,
tile_k, walk length, atom width all probed; rel_err 0.0 everywhere). The
lever is structurally priced out: TMEM double-buffering costs exactly the
third co-resident CTA whose mainloop was already hiding the epilogue for
free. It can only pay inside a kernel whose 2-CTA pair saturates the SM's
tensor pipe on its own (persistent cuBLAS-class mainloop) -- i.e. the
lever is DOWNSTREAM of the persistent-CTA rewrite, not parallel to it.

## Harness verify gates (3/3 PASSED, `--profile none`)

```
AISP_TCGEN05_VARIANT=dual_cta_2sm (loader defaults 128,3,ws=1,mb=2,k=64,eo=0):
  verification passed: true, 0 failed (ts 18:11:21, best_speedup 2.8196x)
AISP_TCGEN05_VARIANT=dual_cta (plain dual regression):
  verification passed: true, 0 failed (ts 18:11:38, best_speedup 2.7577x)
AISP_TCGEN05_VARIANT=cluster (default regression):
  verification passed: true, 0 failed (ts 18:11:54, best_speedup 2.4624x;
  2.329x contract intact)
```

Verdict snapshots: `/tmp/frontV4/gate{1,2,3}_{2sm,dual,cluster}_results.json`.

## Files (pod, md5)

```
CHANGED  labs/custom_vs_cublas/tcgen05_dual_cta_2sm.cu  39b52e587871389181a3ef74b24b5566
         (V4: DUAL2SM_EPI_OVERLAP=0|2|4|8 cross-tile TMEM double-buffer +
          second epilogue warpgroup + DUAL2SM_EPI_ATOM chunked t2r;
          defaults byte-equivalent path to B60 -- the incumbent rebuilt
          under V4 source reproduces 152 regs / 3 CTAs/SM / the same
          863-929us round-robin band and passes all gates)
CHANGED  labs/custom_vs_cublas/tcgen05_loader.py        1ea36d72cffadccf312f0122d68b646e
         (V4: epi_overlap/epi_atom params + AISP_DUAL2SM_EPI_OVERLAP /
          AISP_DUAL2SM_EPI_ATOM env plumbing; defaults UNCHANGED:
          128,3,ws=1,amc=0,mb=2,k=64,eo=0,ea=32 -- B57/B60 incumbent
          stays the default)
UNCHANGED labs/custom_vs_cublas/bench_dual_cta.py       5ad5ea81aa5ec2c22b61bb2673878384
Backups: /tmp/frontV4/{tcgen05_dual_cta_2sm.cu,tcgen05_loader.py} (B60
originals, md5 ae3f0a34/83744a2d). Logs: /tmp/frontV4/ab_{e2,e4,s4e2,
k128e2,s4e2a16}.log, gate{1,2,3}_*.log; ncu: /tmp/frontV4/
ncu_{e2_n128s3,e2_n128s4,inc_n128s3}.ncu-rep. A/B driver:
/tmp/frontV4/ab_v4.py. Nothing committed.
```

## Named next lever

**Persistent CTAs + tile rasterization (B60's candidate #2), now carrying
V4's building block:** a persistent grid of ~304 clusters (2 CTAs/SM,
TMEM 2x128-col buffers) with a SWIZZLED (m,n) tile walk. V4's epilogue
warpgroup + double-TMEM + cross-tile-pipeline machinery is exactly the
persistent kernel's inner structure and is now flag-gated, correctness-
proven (rel_err 0.0 incl. buffer reuse) and measured -- what failed here
is only the GRID SHAPE (consecutive-n walks with no rasterization destroy
L2 reuse, +0.74GB DRAM) and the feed-concurrency deficit (2 vs 3 streams).
The persistent rewrite must therefore ship BOTH: (a) an L2-aware swizzled
rasterization (the standard cuBLAS tool, attacks the V3 DRAM/L2 numbers
directly), and (b) deeper stages (s=4/96KB measured best; s=5/120KB fits
the full 227KB budget at 2 CTAs/SM if the 110KB static cap is lifted to
113KB) or B-multicast across persistent pairs to close the 2-stream gap.
V4's data says (b) alone recovers 16 of the 21 lost points -- the
remaining 5 must come from (a).
