# GB300 capstone tcgen05 wave quantization: balanced k-redistribution (Front W)

**Date:** 2026-06-11 | **Pod:** <gb300-pod> (GPU 3) | **Base:** 222a8ac7 (B48-banked parameterized kernel, bK64/S4 incumbent) + uncommitted GB300 fixes
**Verdict: HONEST NEGATIVE — refuted by measured wave math before any build. Zero-implementation-overhead ceiling is 1.045–1.074x and the mandatory costs of the only legal mechanism exceed the entire margin above the 1.05x gate. No code was changed; the incumbent stands.**

The D5b-named lever — 128 CTAs on 152 SMs (84.2% fill, 15.8% idle) at the scored
2048^3 shape; close the fill gap — was attacked with cheap measured probes on the
unmodified kernel before committing to an implementation. The probes falsify the
premise quantitatively: **the kernel is not wave-bound, it is fixed-cost-bound.**
A ~15us per-CTA fixed cost (62–65% of the 23.9us kernel) is paid by every CTA
regardless of k-work, so redistributing the k-loop across idle SMs can recover at
most 5 of 32 k-tiles' time (~2.2% to 7.4% depending on fit), and the measured
+3.9–4.2% full-fill contention penalty plus the mandatory split/reduction costs
consume more than that. All three candidate mechanisms die; two die by
arithmetic alone.

## 1. Verified ground truth

- **SM count: 152** (torch `get_device_properties`, GPU 3, this session — not the
  148 of B200). 128-CTA grid fill = 128/152 = **84.21%**; D5b's "~15.8% idle" is
  exactly 24/152.
- Grid at 2048^3 with 128x256 tiles: 16 x 8 = **128 CTAs**, 32 k-tiles each
  (bK=64), TMEM allocator pins 1 CTA/SM, all CTAs resident at t=0 (single wave).
- Baseline reproduced before any work: harness
  `labs/blackwell_matmul:blackwell_matmul_tcgen05` = **138.95x, verification
  passed** (max_diff 0.125, rtol 1e-3 / atol 5.0), kernel-level 2048^3 median
  **23.90us** (matches B48's 24.0–24.1 within drift; B48's banked 151x harness
  number is within the documented 5–9% thermal band).
- Total balanced work: 128 tiles x 32 k-tiles = 4096 k-tile units over 152 SMs =
  ceil = **27 units/SM** vs 32 today — the absolute best redistribution.

## 2. Probe measurements (incumbent kernel, no modifications)

Interleaved shape rotation per cycle, CUDA-event timed (300 calls/shape/cycle),
GPU 3. Two clean replications: rep1 = 5 cycles, rep3 = 7 cycles (rep2 discarded —
see section 6). Source: `/tmp/frontW/wq/02_probe_results.json`,
`06_probe_results_rep3.json`.

| shape | CTAs | rep1 median (min/max) us | rep3 median (min/max) us | note |
|---|---|---|---|---|
| 2048x2048x2048 | 128 | 23.903 (23.886/24.128) | 23.941 (23.892/24.105) | scored shape, 718 TFLOPS |
| 2432x2048x2048 | 152 | 24.894 (24.890/25.143) | 24.871 (24.854/25.098) | full single wave, +18.75% work in +4.1% time |
| 2560x2048x2048 | 160 | 41.742 (41.730/41.777) | 41.759 (41.735/41.777) | 8-CTA second wave costs +70% — confirms 1 CTA/SM single-wave scheduling |
| 2048x2048x4096 | 128 | 32.324 | 32.275 | K-slope |
| 2048x2048x1024 | 128 | 19.431 | 19.442 | K-slope |
| 2048x2048x256 | 128 | 15.712 | 15.724 | K-slope: **137 TFLOPS — the kernel is F-bound** |
| 128x256x2048 | 1 | 20.443 | 20.444 | single CTA = 85% of full-grid time |
| 128x256x256 | 1 | 14.194 | 15.378 | per-tile cycle cost probe (noisiest shape) |

rep3 per-rep (base): 24.105 23.892 23.904 23.923 23.954 23.944 23.941
rep3 per-rep (152-CTA): 25.098 24.871 24.868 24.854 24.862 24.882 24.885

**Derived constants** (T = F + u * nk, nk = K/64; both fits, both reps):

| fit | u (us/k-tile) | F (us) | F share of 23.9us |
|---|---|---|---|
| least-squares (4 pts), rep1/rep3 | 0.275 / 0.274 | 14.87 / 14.90 | **62%** |
| K=2048→4096 endpoint, rep1/rep3 | 0.263 / 0.260 | 15.48 / 15.61 | **65%** |

**Full-fill contention** (152 active CTAs vs 128, identical per-CTA work,
interleaved within cycles so thermal drift cancels): **+4.15% (rep1), +3.89%
(rep3)** per-CTA inflation.

## 3. Wave-fill math: the ceiling (derived from measured constants)

Perfect balanced k-redistribution (ideal stream-K: every SM gets exactly 27 of
the 4096 k-tile units, zero implementation cost):

| scenario | rep1 ceiling | rep3 ceiling |
|---|---|---|
| zero overhead, zero contention (ls / endpoint fit) | 1.072 / 1.058 | 1.074 / 1.058 |
| + measured contention applied to k-loop only | 1.058 / 1.045 | 1.060 / 1.045 |
| + contention applied to F too (if clock/uncore-driven) | 1.030 / 1.016 | 1.034 / 1.018 |

The **zero-implementation-overhead** band is 1.016–1.074 with midpoint at the
gate. The mandatory implementation costs (section 4) are >= 1.5us against an
ideal of <= 22.9us, i.e. they consume 2–3x the entire margin above 1.05x in the
most favorable fit, and the gate is missed before overhead in the endpoint fit.

Amdahl form: only 8.4–8.5us of the 23.9us (35–38%) is k-redistributable; the
~15us F is per-CTA fixed cost (prologue fill + TMEM alloc + t2r epilogue +
launch turnaround) that redistribution cannot touch — and stream-K *adds* F-class
work (extra epilogues, refills).

## 4. Mechanism-by-mechanism falsification

1. **Tile re-shape (preferred mechanism): dead by divisibility.** Need a legal
   UMMA atom giving 128 < CTAs <= 152 at 2048x2048 output. M=128 atoms: CTAs =
   16 * 2048/N, so 2048/N must lie in (8, 9.5] — no divisor of 2048 does. M=64
   atoms: CTAs = 32 * 2048/N <= 152 requires N >= 432 > 256 (max atom N). The
   2SM 256x256 variant gives the same 128 CTAs. No legal shape exists.
2. **Persistent CTAs + tile scheduler: dead by arithmetic.** 128 tiles < 152
   SMs; every CTA is resident at t=0 and persistence reorders nothing. (The
   brief's own caveat, confirmed: at this shape persistence alone creates no
   work.)
3. **Deterministic k-split on a subset: dead by makespan.** All 128 CTAs start
   at t=0 and an unsplit tile takes F + 32u — the makespan. Splitting any
   subset's k-tails onto idle SMs changes nothing unless **every** tile sheds
   k-work. That forces >= 128 tile splits => >= 16.8MB fp32 partial writes +
   16.8MB reduction reads through L2 on the critical path, >= 147 extra
   pipeline-segment cycles (4-stage TMA refill + drain + partial epilogue per
   switch; the hot single-tile cycle measures 14–15us *with* launch, so an
   in-kernel switch is multiple us), plus a flag/spin protocol. B48's finding
   that doubling the k-cadence cost 14.5% is the same cost class. These
   overheads (>= 1.5us, realistically 2–4us) exceed the <= 0.6us of margin the
   most optimistic ceiling leaves above 1.05x — and the owner/tail and full
   stream-K variants were both modeled with measured constants before rejection.

**Shape scope:** every scored tcgen05 GEMM target (`blackwell_matmul_tcgen05`,
`cluster_gemm_tcgen05`, `cluster_gemm_tcgen05_cta2`) constructs size=2048 only.
2048^3 is THE scored shape; there is no scored larger-M/N shape that already
fills the GPU. (At M=2432 the same kernel hits 820 TFLOPS — the fill gap is
real, but it is not recoverable at the scored shape for this kernel because the
work that would fill it is F-dominated.)

## 5. ncu fill metrics (profile-grounding, locked clocks)

`--metrics` pass, `--launch-skip 5`, first profiled launch; ncu locks clocks and
flushes caches, so absolute times are inflated — used for structural ratios
only. Source: `/tmp/frontW/wq/03_ncu_*.stdout`, `04_ncu_tensor_*.stdout`,
summary in `03_ncu_summary.json`.

| metric | base 128 CTA | wave 152 CTA |
|---|---|---|
| launch__grid_size | 128 | 152 |
| sm__cycles_active.avg / .max (fill) | 70341/84614 = **83.1%** | 84035/85610 = **98.2%** |
| sm__pipe_tensor_cycles_active avg / max %peak | **15.38 / 18.26** (avg/max = 84.2% = fill) | **17.99 / 17.99** |
| busiest-SM active cycles | 84614 | 85610 (**+1.2%** at locked clocks) |
| lts__throughput %peak | 11.31 | 13.58 |
| dram__throughput %peak | 8.12 | 9.01 |

Reading: the 15.8% idle-SM diagnosis is confirmed (83.1% active-cycle fill,
tensor-duty dilution exactly equal to fill), and filling the GPU equalizes
per-SM tensor duty at ~18% — i.e. even at 100% fill the tensor pipes idle 82%
of the time. The wave-quant gap is real but sits on top of an issue/latency-
bound kernel whose per-CTA critical path *grows* under full fill (busiest-SM
+1.2% locked-clock, +3.9–4.2% free-clock hot). Single-CTA control:
`sm__cycles_active.avg = 631.95 = max/152` — exactly one SM active, proving no
foreign SM activity contaminated the profiled windows.

## 6. Ops incidents (B43-class, both contained)

**GPU squatter:** the first 7-cycle replication (rep2, `05_probe_rep2.*`) was
contaminated: an orphaned foreign `python3` (PID 975899, started 13:54:09Z,
parent 1) squatted GPU 3 mid-run — base read 50.7us and the 152-CTA shape
flipped bimodally 51/61us. Detected by the pre-run
`nvidia-smi -i 3 --query-compute-apps` probe captured in the evidence tuple;
rep2 discarded; rep3 re-run on a verified-clean GPU (pre/post checks
`06_precheck_gpu3.txt` / `06_postcheck_gpu3.txt`, both empty) and replicates
rep1 within 0.2% on every load-bearing shape. The ncu passes overlapped the
squatter's lifetime but are internally proven clean (see single-CTA control
above). Also noted: `/tmp/frontW/` contained another front's training_hotpath
scratch at session start — this front's evidence is namespaced under
`/tmp/frontW/wq/`.

**Mid-session working-tree edit by a sibling front:** the pod kernel changed
md5 `aa013923` -> `6857217e` at 13:50:32Z (Front X adding an opt-in
`AISP_TCGEN05_CTA1_CLUSTER_M` B-multicast lever; backup at
`/tmp/frontX/capstone_kernels_tcgen05.cu.b48bak` == the verified pre-edit
file). Build forensics (`/work/torch_extensions/wq_tcgen05_kb4_s4/`, ninja log):
rep1 ran an .so compiled from the PRE-edit source (~13:48); the loader rebuilt
at 13:57:31, so the ncu passes and rep3 ran the POST-edit source at default
flags. Containment is positive, not assumed: the seeded 2048^3 reference
outputs of the pre-edit and post-edit builds are **bit-equal**
(`torch.equal(ref_rep1, ref_rep3) = True`) and rep1/rep3 medians agree within
0.2% on every shape — Front X's default path is empirically byte-identical to
the incumbent, so every number in this report stands on one measurement basis.

## 7. Numerics class

No kernel change shipped — the incumbent's bit-identical B36/B39/B44/B48
lineage is untouched. (Had the k-split variant been built it would have broken
bit-identity by construction — two-part fp32 accumulation — which is why the
falsification was completed *before* building.) Determinism of the incumbent
re-verified this session: seed-42 reference, `torch.equal` across repeat calls =
True (3 checks in each probe run); reference saved at
`/tmp/frontW/wq/ref_B48_2048.pt` (+ `_rep3` copy).

## 8. Named next lever

**Attack F, not the wave.** F ~= 14.9–15.6us fixed per-CTA cost is 62–65% of the
scored kernel and caps every k-side optimization (at K=256 the kernel delivers
137 TFLOPS — 3.6% of FP16 SoL — purely F-bound). First step is a hot
decomposition of F into launch turnaround vs TMEM alloc/free vs 4-stage
prologue fill vs t2r epilogue (the epilogue uses the narrowest
`SM100_TMEM_LOAD_32dp32b1x` atom; a wider tmem-load atom or an epilogue/k-loop
overlap are the obvious candidates; single-CTA probes here give
launch+F_in-kernel ~= 13.3us and per-k-tile u_1cta ~= 0.223us as the starting
split). EV: F is ~64% of kernel time; even a 20% F cut is ~2x the entire
wave-quant ceiling.

## Evidence

`/tmp/frontW/wq/` on the pod: `01_baseline_repro.{cmd,stdout,stderr,exit}`,
`02_probe.{cmd,stdout,stderr,exit}` + `02_probe_results.json`,
`03_ncu_{base128,wave152,tile1k2048,tile1k256}.{cmd,stdout,stderr,exit}` +
`03_ncu_summary.json`, `04_ncu_tensor_{base128,wave152}.*`,
`05_probe_rep2.*` (contaminated, retained for the incident record),
`06_probe_rep3.*` + `06_probe_results_rep3.json` + GPU pre/post checks,
`07_verdict.json`, scripts `probe_wave.py`, `ncu_shape.py`, `verdict.py`,
references `ref_B48_2048*.pt`. Harness artifact:
`artifacts/runs/20260611_134542__bench__profile_none_targets_labs_blackwell_matmul_blackwell_matmul_tcgen05/`.
Working tree: this front touched ZERO repo files (verdict reached before
building). At session start local == pod for kernel
(`aa01392300ba8fa2ddeb4c64859bb06c`) and loader
(`e56aea061071c1da314bb0e34b380d9b`); at session end the loader is unchanged
and the pod kernel is `6857217e...` due to Front X's mid-session opt-in edit
(theirs to bank/sync — see section 6 for the proof it did not affect these
measurements).
