# moe_cuda router_vectorized: GELU into the GEMM1 template epilogue (Front M2)

## Verdict

**HONEST NEGATIVE — the fusion LANDS structurally (16 -> 15 kernels/replay, the
standalone GELU kernel disappears into `triton_tem_fused_baddbmm_gelu_*`,
bit-identical output) but buys only ~1.03x end-to-end, below the 1.05x gate.**
The B51 estimate (~1.2x) assumed the 69us standalone GELU was eliminable
overhead; measured, it is erf compute that is *conserved* under fusion — it
moves into the template epilogue (+58.6us on the GEMM1 template) instead of
disappearing. Fusion saves only the intermediate's memory round-trip (~10.6us).
Default stays B51; the fused path ships as opt-in `AISP_MOE_CUDA_GELU_EPILOGUE=1`.

- Date: 2026-06-11, pod `<gb300-pod>` (GB300 NVL72), GPU 3 only, flock lease.
- Torch `2.12.0a0+5aff3928d8.nv26.05`. Baseline arm: B51 (commit `a21f4781`, 0.3932 ms).
- B51 re-repro before any edit: **0.39083 ms**, verification passed, graph_breaks=0
  (`/tmp/frontM2/r01_b51_repro.*`) — within noise of the banked 0.3932.
- Evidence: `/tmp/frontM2/` on the pod (`.cmd/.stdout/.stderr/.exit` per run).

## The Inductor-level WHY (the bankable diagnosis)

Why GELU did NOT fuse at B51 (`/tmp/frontM2/r02_probe_v0.stderr`, fusion log of
an exact B51-config compile with caches disabled):

1. Inductor's template-epilogue fusion is benchmark-driven here
   (`benchmark_epilogue_fusion=True` default) and benchmarks only
   **`max_epilogue_benchmarked_choices = 1`** candidate
   (torch/_inductor/config.py:881 — no env knob).
2. That single fused candidate was
   `BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_stages=4, num_warps=8`, which
   fails Triton precompile on SM103 with a shared-memory overflow:
   `OutOfMemoryError: out of resource: triton_bmm Required: 262160 Hardware
   limit: 232448` (select_algorithm.py logs the exception; the config is
   infeasible for ANY use — the *unfused* autotune skipped it gracefully and
   picked `BLOCK_K=64, num_stages=3` at 0.1065 ms).
3. The failed benchmark scores the fusion as infinitely slow:
   `cannot fuse (benchmark): fusing OrderedSet(['buf7']) with
   OrderedSet(['buf8']) cause infx slowdown` (scheduler.py:4374) — fusion refused.

So at B51 the refusal was an **accident of an infeasible benchmark candidate**,
not a measured decision. With the benchmark widened
(`max_epilogue_benchmarked_choices=8`, probe v2), valid fused configs get
measured and Inductor refuses *honestly*: `cause 1.044x slowdown`
(`/tmp/frontM2/r03_probe_v2_epi8.stderr`) — the fused template is slower than
template + a well-tuned separate GELU. (Nuance: that comparison uses Inductor's
freshly tuned ~52us GELU; the B51 disk-cache artifact runs a worse 69us GELU
config, which is why forcing fusion still nets a small real win vs the
incumbent arm.)

## Approach chosen + why others rejected

Chosen: **(1) Inductor knob** — `torch._inductor.config.benchmark_epilogue_fusion
= False`, set in `setup()` only when `AISP_MOE_CUDA_GELU_EPILOGUE=1`. With the
(broken) benchmark gate off, the scheduler fuses heuristically and the fusion
materializes. Changing the config also changes the Inductor cache key, so the
arms never share compiled artifacts.

- (2) forward rewrite rejected: GELU is already immediately adjacent to the
  GEMM1 `baddbmm` with no expand/gather boundary between them; the refusal is
  not structural, so there is nothing to rewrite.
- (3) hand-fused baddbmm+GELU kernel rejected: probes bound the prize first —
  the erf cost (~50us, ALU/SFU-bound: memory SoL for the [32,640,2048] bf16
  pointwise is ~21us but the best standalone kernel runs 52us) is conserved in
  any composition, so even a perfect hand epilogue at nvjet GEMM throughput
  (~66us GEMM + fully hidden erf) caps near ~50us saved; the Triton-template
  reality measured (+58.6us epilogue cost) says erf does NOT hide behind the
  MMA pipeline at these shapes. Not worth a custom kernel for a sub-gate prize.

Knob variants probed (all `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, GPU us/call
over 5 profiled calls, compiled-no-graph; `/tmp/frontM2/r0{2,3,4}_probe_*`):

| probe | knobs | GPU us/call | kernels | GEMM1(+GELU?) | standalone GELU |
|---|---|---|---|---|---|
| v0 control (=B51 config) | — | 349.5 | 15 | 103.9 (no) | 69.4 |
| v1 | benchmark_epilogue_fusion=0 | 335.5 | 15 | **162.0 (fused)** | — |
| v2 | max_epilogue_benchmarked_choices=8 | 333.4 | 15 | 103.8 (no) | 52.1 |
| v3 | v1 + coordinate_descent_tuning | 337.8 | 15 | 162.1 (fused) | — |
| v4 | v1 + gemm_backends=TRITON | 336.4 | 14 | 161.8 (fused) | — |
| v5 | v1 + gelu approximate=tanh | 340.9 | 15 | 166.0 (fused) | — |

The fused GEMM1+GELU template is pinned at ~162us under every knob (CD tuning,
backend restriction, tanh approx) — the epilogue erf cost is irreducible at the
template's occupancy. v2 shows the counterfactual: a well-tuned *separate* GELU
(52us) makes unfused (156us pair) beat fused (162us), which is exactly the
1.044x Inductor measured.

## A/B vs B51 (harness, interleaved, 6 reps/arm, GPU 3, foreign-app check per rep)

REP_TABLE_PLACEHOLDER

## Kernel-count + per-kernel timing diff (5-replay torch profiler on the
harness-built CUDA graph, `/tmp/frontM2/r06_replay_arm{0,1}.stdout`)

| kernel | B51 arm (us/replay) | fused arm (us/replay) |
|---|---|---|
| `triton_tem_fused_baddbmm_slice_unsqueeze_view_3` (GEMM1) | 103.3 | — |
| `triton_poi_fused_gelu_4` | 69.2 | **— (eliminated)** |
| `triton_tem_fused_baddbmm_gelu_slice_unsqueeze_view_3` (GEMM1+GELU) | — | 161.9 |
| `nvjet_sm103_tst_128x160_64x8` (GEMM2) | 58.2 | 58.0 |
| `direct_copy` elementwise (gather-back) | 40.2 | 39.3 |
| `indexFuncLargeIndex` (combine `index_add_`) | 25.3 | 25.5 |
| 11 small kernels (router/topk/dispatch/softmax) | 48.9 | 50.4 |
| **total / kernels per replay** | **345.2 / 16** | **333.7 / 15** |

GPU delta: 11.5us/replay = 172.5 (GEMM1+GELU pair) - 161.9 (fused) ≈ exactly
the intermediate's HBM round-trip; the erf compute moved, unchanged, into the
epilogue (161.9 - 103.3 = 58.6us added to the template).

## Numerics class

**Bit-identical.** Both arms: `bitwise_equal=True`, `max_abs_diff=0.0` vs the
eager GroupedTopKMoE forward on the lab input (seed-42, fixed routing),
`/tmp/frontM2/r06_replay_arm{0,1}.stdout`. Inductor's epilogue codegen rounds
the fp32 accumulator to bf16 *before* applying GELU (matching the unfused
dataflow), so fusion does not reassociate here. Harness verification (vs eager
per-token baseline, rtol 0.1 / atol 1.0 — untouched) passed on every rep.

## Audit-rule proof metrics (exported every run)

`router_vectorized.{manual_graph_active, compile_enabled, compile_seconds,
dynamo_graph_breaks(=0), dynamo_unique_graphs(=1)}` as at B51, plus NEW:
`router_vectorized.kernels_per_replay` (16.0 incumbent / 15.0 fused arm,
counted by a 5-replay profiler pass in untimed `setup()`) and
`router_vectorized.gelu_epilogue_enabled`. The opt-in path FAILS LOUDLY if a
standalone `triton_poi*fused_gelu` kernel survives in the replay profile
(fusion silently refused -> RuntimeError in `setup()`).

## Files touched

- `code/labs/moe_cuda/optimized_router_vectorized.py` (local == pod, md5
  `MD5_PLACEHOLDER`): opt-in `AISP_MOE_CUDA_GELU_EPILOGUE` knob
  (default OFF = incumbent B51 compile config, byte-identical Inductor cache
  key), kernels-per-replay audit metric (both arms), loud-failure assert on the
  opt-in path. B51 kill switch `AISP_MOE_CUDA_COMPILE=0` untouched.
- Probe/driver scripts (evidence, not lab code): pod
  `/tmp/frontM2/{probe_fusion,replay_kernels}.py`, `/tmp/frontM2/ab_driver.sh`.

## Pod evidence

`/tmp/frontM2/`: `r01_b51_repro.*` (pre-edit repro), `r02_probe_v0.*` (control
probe + fusion log with the smem-OOM/infx refusal), `r03_probe_{v1_noepibench,
v2_epi8}.*`, `r04_probe_{v3_fuse_cd,v4_fuse_triton,v5_fuse_tanh}.*`,
`r05_smoke_{b51,gelu_epi}.*`, `r06_replay_arm{0,1}.*` (kernel tables +
numerics), `ab/rep01..rep06_arm{0,1}.*` + `ab/ab_results_parsed.json`
(authoritative A/B summary); harness run dirs under
`code/artifacts/runs/20260611_*_targets_moe_cuda_router_vectorized/`.

## Mechanism (one paragraph)

The standalone GELU kernel is not launch/memory overhead — it is ~50us of
erf evaluation over 41.9M elements (2.4-3.3x above the 21us memory SoL for the
167.8MB round-trip, i.e. transcendental-ALU-bound at the achieved occupancy).
Epilogue fusion can only delete the memory round-trip (10.6us measured); the
erf work re-serializes inside the GEMM template at no better throughput
(+58.6us), and the template itself (104us for 86 GFLOP = 22% of bf16 SoL;
nvjet does it at 65.9us) is too slow for the pair to come out ahead. Inductor's
benchmark-driven refusal is the *correct* call against a well-tuned separate
GELU; at B51 it was made for the wrong reason (infeasible candidate -> infx).

## Named next lever

**Recover nvjet for GEMM1 under compile.** The compiled graph's GEMM1 runs as
a Triton template at 103-162us, but eager nvjet does the same baddbmm in
65.9us (B45 profile). Inductor's autotune mis-measures the extern choice at
0.1403 ms (`r02_probe_v0.stderr` AUTOTUNE table) — 2.1x worse than nvjet's
real cost, plausibly a broadcast-bias (stride `[2048,0,1]`) benchmark artifact.
If extern baddbmm competes at its true 66us: 66 + 52 (tuned separate GELU,
the v2 pick) vs the incumbent 172.5 saves ~54us/replay -> ~0.34 ms -> ~1.15x.
Diagnose the extern-benchmark gap, or pin GEMM1 to ATEN via a local
`max_autotune_gemm_backends` scope and let GEMM2 keep its template.

## Final interleaved A/B (authoritative, parsed post-completion by the banking session)

Driver: /tmp/frontM2/ab_driver.sh — 6 reps/arm, alternating, flock-held GPU3,
per-rep foreign-process check (all 0). Parsed: /tmp/frontM2/parse_ab.py.

| arm | median ms | min | max | n | verified | kernels/replay |
|---|---|---|---|---|---|---|
| B51 (epilogue=0) | 0.40292 | 0.40258 | 0.40668 | 6 | 6/6 | 16 |
| fused (epilogue=1) | 0.39396 | 0.39293 | 0.39805 | 6 | 6/6 | 15 |

speedup_median = **1.0227x** (below the 1.05 gate; distributions non-overlapping
— every fused rep beat every B51 rep — so the improvement is real, just small).
graph_breaks=0 and unique_graphs=1 on every rep. Verdict above stands.
