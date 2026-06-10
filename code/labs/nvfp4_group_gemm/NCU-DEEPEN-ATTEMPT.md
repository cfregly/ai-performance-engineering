# Deepen-to-SoL attempt: lever 1 (smem/occupancy) on the custom NVFP4 grouped GEMM (GB300)

## Outcome (short)

No `.cu` change landed. Lever 1 (cut per-CTA shared memory to raise CTAs/SM) was implemented as a
compile-time experiment and MEASURED to regress, so per the guardrail (keep only a change that
verify-passes AND improves) it was not applied to the source. The kernel `.cu` is pristine
(sha256 `982cce72...`, byte-identical to the start). Verify still PASSES; the lab gate is unchanged
(0.922x, FAIL).

The reason is a refinement of the original diagnosis: the occupancy ceiling is NOT shared-memory
capped in practice, it is warp-specialization capped. Adding CTAs/SM does not raise achieved
occupancy, so the smem lever cannot convert to a SoL gain. The only lever that can is the deep
barrier/warp-spec rewrite (lever 2), which is load-bearing and out of safe single-session scope.

No win is claimed (no verify-passing ncu-measured SoL improvement was achieved).

## Before vs after (optimized config, UnrollN=2, g2_n3072_k4096, GB300, ncu --set full, boost clock)

| Metric | BEFORE (stages=2, shipped) | Lever 1 (stages=1) | Verdict |
| --- | ---: | ---: | --- |
| GPU kernel duration | 226.1 us | 280.6 us | regress +24% |
| %HBM-SoL (DRAM vs 8.0 TB/s) | 15.1% | 12.0% | regress |
| achieved occupancy | 6.24% | 6.22% | no change |
| CTAs/SM (waves) | 2 (2.96) | 4 (1.48) | 2x CTAs, no occ gain |
| theoretical occupancy | 12.5% | 25.0% | doubled (but unrealized) |
| barrier stall (cyc/issue) | 11.1 (46.6%) | 17.9 | worse |
| issue-active | 4.20% | 3.01% | worse |

Wall-clock (lab harness, cudagraph + L2-clear): optimized 0.3646 ms, unchanged (no `.cu` edit).

## Why lever 1 fails: achieved occupancy is warp-spec-capped, not smem-capped

The shipped kernel is shared-memory limited to 2 CTAs/SM: per-CTA static smem is 110.6 KB
(`sA` 16, `sB` 32, `sSFA` 2, `sSFB` 4 KB per stage, 2 stages), and the SM budget is ~233 KB.
Cutting to `PIPELINE_STAGES=1` halves that to ~56 KB/CTA, which the occupancy model rewards with
4 CTAs/SM and 25% theoretical occupancy.

But achieved occupancy did not move (6.24% -> 6.22%), and the kernel got slower. Direct evidence
that CTAs/SM is not the binding constraint, three configs all pinned at ~6.2% achieved:

| Config | CTAs/SM | theoretical occ | achieved occ |
| --- | ---: | ---: | ---: |
| UnrollN=2, stages=2 | 2 | 12.5% | 6.24% |
| UnrollN=2, stages=1 | 4 | 25.0% | 6.22% |
| UnrollN=1, stages=2 | 3 | 18.8% | 6.23% |

The binding constraint is the warp-specialized design: 128 threads = 4 warps/CTA, of which only the
TMA-producer warp and the consumer (thread0, which issues both the UTCCP scale-copies and the MMAs)
are active in the mainloop; the rest idle until the epilogue. So ~4 active warps/SM regardless of how
many CTAs are resident. Worse, `PIPELINE_STAGES=1` force-disables `WS_TMA_PRODUCER` (it requires
stages>1) and removes the double-buffer, so the consumer stalls harder at the per-k-tile barrier
(11.1 -> 17.9 cyc/issue) and DRAM throughput drops (15.1% -> 12.0%). The extra CTAs cannot
compensate because each CTA is individually more stalled.

Conclusion: the occupancy lever from the diagnosis does not convert here. Theoretical occupancy rose
2x, achieved occupancy and SoL did not.

## Why lever 2 (barrier rewrite) is out of safe scope

The 46.6% barrier stall is the per-k-tile `__syncthreads()` (kernel line ~3459). It is load-bearing:
with double-buffering, the dedicated TMA-producer warp fills `stage_next` while the consumer reads
`stage_cur`, and this CTA-wide barrier is what guarantees the consumer is finished with a stage
buffer before the producer reuses it next iteration. It is not a cp->mma ordering sync (thread0
issues both the UTCCP copies and the MMAs, so that ordering is already guaranteed by the tcgen05
pipeline), it is stage-buffer-lifetime coordination between two different warps.

Removing it races the producer against the consumer (a correctness break). Narrowing it to a named
barrier or an mbarrier-only producer/consumer handoff requires matching changes in both the consumer
mainloop and the separate producer-warp buffer-reuse logic, in a 6316-line megakernel with intricate
TMEM addressing and many interacting compile-time paths (CtaGroup 1/2, UnrollN 1/2, WS_TMA_PRODUCER,
WS_UNROLL2_MMA, WS_SPLIT_U0_SEGS, segment-parallel). That is the full lever-2 async-pipelining
rewrite, not a contained change, and it could not be validated verify-passing within this session.
Per the user's gate (attempt lever 2 only if lever 1 lands cleanly) it was not pursued to a source
edit.

## The real GB300 win (out of the .cu-only scope)

The lab shows the "optimized" config (UnrollN=2, B200-tuned) is 0.922x of the baseline config
(UnrollN=1): 0.3646 ms vs 0.3362 ms, verify PASS both. UnrollN=2 doubles `sB` (32 -> 64 KB/stage),
dropping it to 2 CTAs/SM vs UnrollN=1's 3, and adds the per-k-tile full-CTA barrier path that
UnrollN=1 can partly avoid. On GB300 the better config is UnrollN=1. That is a wrapper change
(`optimized_nvfp4_group_gemm.py`, `AISP_NVFP4_GROUP_GEMM_UNROLL_N=2 -> 1`), which the
modify-only-the-.cu constraint excludes here, but it is the highest-value, lowest-risk GB300 win and
is recommended as the next step. It also flips the lab gate.

## Status

- Kernel `.cu`: UNCHANGED (pristine, sha256 `982cce72...`).
- Verify: PASS (baseline; `.cu` byte-identical to the verified state).
- Lab gate (`labs/nvfp4_group_gemm:nvfp4_group_gemm`): FAIL, 0.922x (unchanged).
- Win claimed: NONE. Lever 1 measured-regressed; lever 2 is a deep rewrite out of safe scope.

## Method

ncu `--set full`, `--clock-control none`, `--kernel-name regex:nvfp4_group_gemm_tcgen05_kernel`,
`--launch-skip`/`--launch-count` scoping (skips the curand setup kernels), GPU 1 (free), reports
validated non-empty (41 passes). `PIPELINE_STAGES` swept via the compile-time knob
`AISP_NVFP4_GROUP_GEMM_PIPELINE_STAGES` (rebuilds the JIT extension), no source edit. Lab verify via
`cli.aisp bench run labs/nvfp4_group_gemm:nvfp4_group_gemm`. Scratch harnesses + ncu reports in the
pod under `/tmp/ncu_diag/`.
