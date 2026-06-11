# GB300 gemm_cluster Occupancy Rewrite (dual-CTA variant) — Front E

## Verdict

**BLOCKED-EXECUTION / IMPLEMENTATION DELIVERED.** The full higher-occupancy
variant (kernel + loader + harness wiring + A/B bench) is implemented and
ready to build, but this session had **no pod access**: `kubectl exec` and
`kubectl cp` are on the permission ask-list and every request was auto-denied
(only read-only `kubectl get/logs/describe` prefixes are allow-listed). No
build, bench, ncu, or harness verify was possible. All numbers below are
design-model predictions, clearly marked. Nothing was committed.

## Why the incumbent is capped at 1 CTA/SM (B35 diagnosis, measured)

`tcgen05_cluster.cu` (904us, 1207 TFLOPS, 32.4% FP16-SoL, occupancy 6.2%):

1. **TMEM**: allocates `Sm100TmemCapacityColumns` = **512 cols (all of TMEM)**
   for a 128x256 fp32 accumulator that only needs 256 cols, **and holds the
   tcgen05 allocation permit until kernel end** — a second CTA's
   `tcgen05.alloc` can never be served.
2. **SMEM**: 4 stages x 48KB = ~192KB of 227KB -> Block Limit Shared Mem = 1.
3. Result: one CTA must hide the full TMA round-trip alone; 4-stage lookahead
   x ~166ns/MMA-ktile vs ~400-500ns TMA RT -> tensor core starves
   (scoreboard-on-smem 46.5%, SM busy 58%).

## Design: `tcgen05_dual_cta.cu` (new file; existing kernels untouched)

One primary lever — restore SM-level concurrency (2 CTAs/SM) instead of
deepening one CTA:

| Resource | incumbent (cluster) | dual_cta (default n=128 s=3) | 2-CTA fit |
|---|---|---|---|
| MMA tile | 128x256 | **128x128** (`SM100_MMA_F16BF16_SS`, N=128 legal: N%8, ≤256) | — |
| TMEM acc | 512/512 cols + permit held | **128 cols** + `release_allocation_lock()` immediately after alloc | 2x128 ≤ 512 ✓ |
| SMEM | 4 x 48KB ≈ 192KB | **3 x 32KB ≈ 96.3KB** | 2x96.3 ≤ 227KB ✓ |
| launch | `__cluster_dims__(2,1)`, LB(128,1) | plain launch, **`__launch_bounds__(128, 2)`**, max-smem carveout | — |
| pipeline | no-wait (racy stage overwrite) | **proper empty-barrier wait** before stage reuse (`umma_arrive` = `tcgen05.commit`, verified internally elected in CUTLASS 4.3.0 `barrier.h` -> exactly 1 arrival/call, so phase tracking with arrive-count 1 is sound) | — |
| mainloop | all 4 warps spin every k-tile | **warp 0 only**; warps 1-3 park on `mma_barrier` until epilogue | — |

Mechanism: per-CTA the strict empty-wait costs some MMA-pipe duty
(~40-50% per CTA at 2-deep TMA lookahead), but the co-resident CTA fills the
gaps -> predicted ~80-90% tensor-pipe duty per SM vs measured 58%.
Cost: 128x128 tile halves arithmetic intensity (64 vs 85 FLOP/B of smem
traffic) -> DRAM busy predicted ~55-60% (from measured 33% at 1207 TFLOPS);
below the 8 TB/s roof, so latency — not bandwidth — remains the binding
constraint being fixed. **Prediction (unmeasured): 1.5-1.8 PFLOPS, 40-48%
SoL.** If DRAM saturates first, fallback config `n=256 s=2` keeps AI at 85
with the same 2-CTA footprint (one env var, no code change).

All CUTLASS API contracts were verified against the pinned CUTLASS 4.3.0
(8cd5bef4, fetched read-only from GitHub): `Allocator1Sm::allocate` accepts
any power-of-2 column count 32..512; `SM100_MMA_F16BF16_SS` and
`umma_arrive` are internally `elect_one_sync`-guarded; `fma` asserts
M∈{64,128}, N%8==0, N≤256.

## Files (all in working tree, uncommitted; pod copies still needed)

- `code/labs/custom_vs_cublas/tcgen05_dual_cta.cu` — NEW kernel
  (`gemm_dual_cta`); compile-time tunables `-DDUAL_TILE_N`, `-DDUAL_STAGES`;
  `static_assert` guards 2-CTA smem/TMEM fit.
- `code/labs/custom_vs_cublas/tcgen05_loader.py` — `_load_kernel` gains
  `extra_cuda_flags` (hash-keyed); new `load_tcgen05_dual_cta_module()` /
  `matmul_tcgen05_dual_cta()`; env `AISP_DUAL_TILE_N` (128), `AISP_DUAL_STAGES` (3).
- `code/labs/custom_vs_cublas/optimized_tcgen05_matmul.py` — variant switch
  `AISP_TCGEN05_VARIANT=dual_cta|cluster`; **default unchanged (cluster)** so
  the committed verify contract (2.329x) is untouched.
- `code/labs/custom_vs_cublas/run_lab.py` — stage 13 "+ 2 CTAs/SM (Dual-CTA)".
- `code/labs/custom_vs_cublas/bench_dual_cta.py` — NEW standalone kernel-only
  A/B bench (cuBLAS vs cluster vs dual_cta, `--sweep` for n/s configs,
  correctness vs torch.matmul).

## Exact run plan (for whoever has pod exec)

```bash
# copy the 5 files to the pod (kubectl cp), then:
kubectl exec aisp-gb300-runall -n hpc-verification -- bash -lc '
  export CUDA_VISIBLE_DEVICES=2
  cd /work/ai-performance-engineering/code
  python labs/custom_vs_cublas/bench_dual_cta.py --sweep'

# occupancy proof (Block Limit Shared Mem must read 2, achieved occ ~12.5%):
kubectl exec aisp-gb300-runall -n hpc-verification -- bash -lc '
  export CUDA_VISIBLE_DEVICES=2
  cd /work/ai-performance-engineering/code
  /usr/local/bin/ncu --set full --launch-skip 5 --launch-count 2 \
    --kernel-name regex:gemm_dual_cta --target-processes application-only \
    python labs/custom_vs_cublas/bench_dual_cta.py'

# harness verify gate (only flips to dual_cta when env is set):
kubectl exec aisp-gb300-runall -n hpc-verification -- bash -lc '
  export CUDA_VISIBLE_DEVICES=2 AISP_TCGEN05_VARIANT=dual_cta
  cd /work/ai-performance-engineering/code
  python -m cli.aisp bench run --targets labs/custom_vs_cublas:tcgen05_matmul \
    --profile none --single-gpu'
```

Accept/iterate criteria: ncu Block Limit Shared Mem == 2 and achieved
occupancy ~2x incumbent, else inspect TMEM alloc serialization; if
kernel-only beats 904us, run the verify gate; if DRAM% > ~85, switch to
`AISP_DUAL_TILE_N=256 AISP_DUAL_STAGES=2`. If 2 CTAs/SM are resident but the
win does not materialize, the measured mechanism (where the stall moved) is
the bankable negative.

## Named next lever

If dual-CTA banks: add **2x1 cluster + TMA multicast of B** on top of the
dual-CTA footprint (halves B-side L2->SM traffic, recovering the AI lost to
the 128-wide tile), then **persistent CTAs + tile-swizzled scheduler** to cut
the 4096-CTA launch/epilogue overhead — i.e. converge on the CUTLASS sm100
warp-specialized collective shape one verified step at a time.
