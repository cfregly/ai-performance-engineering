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
