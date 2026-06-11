# GB300 SoL / Roofline Grounding — Labs Batch 2 (Front F2)

Date: 2026-06-11 (UTC) | Pod: <gb300-pod> (<namespace>) | GPU: 3 only (CUDA_VISIBLE_DEVICES=3)
Repo: /work/ai-performance-engineering/code @ 2f7e30f9 + uncommitted GB300 fixes
Peaks used: NVFP4 15.0 PFLOPS / FP8 7.5 / BF16 3.75 PFLOPS / HBM3e 8.0 TB/s
Harness: `python -m cli.aisp bench run --targets <lab> --profile none --single-gpu`, validity strict, 20 iters / 5 warmup unless noted.

## Headline

1. **block_scaling WIN (shipped)**: CUDA-graph capture+replay of the CuTeDSL blockscaled launch.
   Optimized arm 94.6us -> 81.1us (**1.168x vs prior optimized arm**, gate >=1.05x), verification PASSED
   (rtol 1e-2 / atol 1e-1), reproduced twice (81.05 / 81.48us). `block_scaling.graph_replay=1.0` in metrics.
2. The survey's "host-invoke dominance" hypothesis is **half right**: the per-iter Python->CuTeDSL invoke is
   only ~8us and fully hidden in back-to-back launch; what the harness wall actually contains is the kernel
   (44.8us, ncu) plus ~50us of *timing-frame* overhead (event bracket + full-device sync + per-iter L2-clear
   + Python). Graph replay reclaims the in-frame host/launch slice (~13.6us at the harness frame).
3. The kernel itself is at **20.5% NVFP4-compute SoL** (44.8us vs 9.2us floor) and the config space is
   **flat** (21 valid tiler/cluster combos within +-5% of default). Remaining headroom is an in-kernel
   rewrite, not config and not host.

## Fresh GB300 ranking table (all runs 2026-06-11, GPU 3, verification PASS unless noted)

Kernel-frame = back-to-back saturated CUDA-event/wall probe (200 iters) or ncu; wall = harness optimized arm.

| # | Lab | Harness base/opt (ms) | Kernel-frame | Binding floor | %SoL (kernel-frame) | %SoL (wall) | Class | Provenance |
|---|-----|----------------------|--------------|---------------|---------------------|-------------|-------|------------|
| 1 | memory_bandwidth_patterns | 0.2192 / 0.0704 (3.12x) | 47.6us | HBM 33.6us (268.4MB) | **70.6%** (5.64 TB/s) | 47.7% | AT-CEILING (cutoff) | run 20260611_031322 + probe |
| 2 | software_pipelining | 0.3200 / 0.1519 (2.11x) | 83.2us | HBM 50.3us (402.7MB) | **60.5%** (4.84 TB/s) | 33.1% | BELOW-CEILING, lever named | run 20260611_031343 + probe |
| 3 | block_scaling (AFTER) | 0.1826 / 0.0815 (2.24x) | 44.8us (ncu) | NVFP4 9.2us / HBM 18.0us | **20.5% compute / 40.2% HBM** | 11.4% | KERNEL-BOUND below ceiling; graph win shipped | runs 20260611_031115/031220, ncu /tmp/frontF2/block_scaling_kernel.ncu-rep |
| 4 | nvfp4_dual_gemm | 0.3704 / 0.0485 (7.64x) | 6.6us | HBM 2.0us / FP4 0.86us | **30.3% HBM / 13.0% FP4** | 4.1% | FRAME-DOMINATED measurement (iters=4) | run 20260611_031405 + probe |
| 5 | flashattention4 (14 arms) | best arm 3.8519 / 0.1330 (28.96x) | n/a | dense 4.6us / causal 2.3us (BF16) | n/a | 3.4% (best arm), <=0.15% others | OVERHEAD-BOUND | run 20260611_031745 |
| 6 | flashattention_gluon | 0.1971 / 0.1240 (1.59x) | n/a | BF16 1.15us (B2 H8 S1024 D64) | n/a | 0.9% | OVERHEAD-BOUND | run 20260611_031723 |
| 7 | flexattention | 12.3416 / 0.1519 (81.2x) | n/a | BF16 0.29us (doc-masked, span 256) | n/a | 0.2% | OVERHEAD-BOUND | run 20260611_031636 |
| 8 | cudnn_sdpa_bench | 0.2606 / 0.3018 (0.86x, no_speedup) | n/a | BF16 ~1.15us (B8 H8 S256 D64 + proj) | n/a | 0.4% | OVERHEAD-BOUND; optimized arm slower | run 20260611_031702 |

## Priority 1 — block_scaling (NVFP4xNVFP4->BF16, M=N=8192 K=1024, sf16, tiler 256x128, cluster (2,1))

### BEFORE (fresh, run 20260611_025858)
baseline 0.18655ms / optimized 0.09463ms / speedup 1.971x / verification PASSED.

### Host-vs-kernel split (probe /tmp/frontF2/probe_block_scaling.py, 200 iters)
- back-to-back wall: **43.8-44.0us/iter** (event span identical -> GPU saturated)
- host submit (Python->CuTeDSL compiled call): **7.9us/call** -> fully hidden back-to-back
- per-iter-sync frame: 58.5us wall; event-bracket p50 55.9us
- harness frame: 94.6us (adds per-iter L2-clear, full-device sync, event bracket, bookkeeping)
- ncu (--set full, locked clocks): kernel duration **44.8us**, grid (2,1,76)=152 CTAs (1 persistent wave,
  cluster(2,1)x76), 224 thr/CTA, theoretical occupancy 10.9% (smem-limited, 1 CTA/SM)

So wall = 44.8us kernel + ~50us frame; the *kernel* is 47% of the harness number. Survey's est. 9-18% SoL
at wall confirmed (9.7% compute-SoL at BEFORE wall); kernel-frame is 20.5% compute-SoL.

### ncu mechanism (kernel inefficiency, not host)
- Compute (SM) 51.1% | Memory 54.0% | DRAM 24.9% (1.98 TB/s) | L2 hit 82.8%
- tensor pipe (UMMA) active: **34.5%** of peak sustained-active -> the MMA is starved, not saturated
- warp stalls: ~58% of cycles = scoreboard dependency on shared/L1TEX operations (shallow-K mainloop:
  K=1024 -> short per-tile mainloop; prologue/epilogue + smem latency not amortized)
- IPC 0.58, issue slots 13%, "No Eligible" 85% -> latency-bound warp-specialized kernel

### Config sweep (step 4): FLAT — refuted as a lever
21 valid (tiler x cluster) combos compiled+timed (probe /tmp/frontF2/sweep_block_scaling.py):
best = tiler (256,256) cluster (2,1) @ **41.98us** vs default (256,128)x(2,1) @ 43.98us = 4.5% kernel-level
(~2% at wall) -> below gate. (128,x) tilers and (4,4) clusters all slower (46-60us). M=64-tile and N=64
tile variants rejected by can_implement.

### Shipped lever (step 3): CUDA-graph capture/replay — WIN 1.168x
- `labs/block_scaling/block_scaling_benchmarks.py`: `OptimizedBlockScalingBenchmarkBase._post_setup` now
  captures the compiled_gemm launch in a `torch.cuda.CUDAGraph` (side-stream capture; the capture stream's
  `CUstream` handle is passed to the CuTeDSL callable — stream is a runtime arg); `_run_problem` replays.
  Eager fallback + `AISP_BLOCK_SCALING_DISABLE_GRAPH=1` kill-switch; `teardown` drops the graph;
  custom metric `block_scaling.graph_replay` records which path ran.
- `labs/block_scaling/optimized_block_scaling.py`: leaf `_post_setup` calls `super()._post_setup()` first.
- Verify path untouched: `_build_verification_output`/`verify_close` still run `problem.run_hardware()`
  eagerly. Graph-replayed output checked bit-identical in probe (max_abs_err 0.0000 vs software ref).
- Cheat-detector: harness `GraphCaptureCheatDetector` only arms for harness-managed graphs
  (`config.enable_cuda_graph`); lab-managed capture in setup is explicitly allowed by the harness setup
  contract and precedented by labs/parameterized_cuda_graphs. Inputs are static device buffers (same as
  the eager path); every replay re-executes the full GEMM.
- AFTER: optimized **0.08105ms** (run 031115, speedup 2.566x) and **0.08148ms** (run 031220, 2.241x),
  verification PASSED both. Win vs BEFORE optimized arm: 0.09463/0.08105 = **1.168x**.
- Mechanism: in the harness frame the 8us CuTeDSL invoke + launch latency sit inside the timed event
  window; replay launches in ~2us. Per-iter-sync probe: eager event p50 55.9us -> graph 47.7us (-8.3us);
  harness frame reclaimed ~13.6us (94.6->81.0).
- Backups: /tmp/frontF2/{block_scaling_benchmarks.py.bak,optimized_block_scaling.py.bak,block_scaling_common.py.bak} (pod).

### Named next lever (NOT config, NOT host)
Kernel rewrite of the vendored sm103 example for shallow-K: deeper smem staging / larger K-stage count to
cover the ~58% scoreboard stalls, or split-K/stream-K to raise UMMA pipe active from 34.5%. Kernel headroom
44.8 -> 18.0us (HBM floor) = 2.5x. This is a CuTeDSL kernel-surgery item, est. multi-day; everything
cheaper is harvested.

## Priority 2 — fresh %SoL numbers

### memory_bandwidth_patterns (fp32 4096x8192 tiled transpose, 268.4MB/iter, floor 33.6us)
Harness: base 0.2192 / opt 0.0704ms, verify PASS. Kernel-frame: **47.6us/iter** back-to-back (host 2.3us)
= **5.64 TB/s = 70.6% HBM-SoL** — at the survey's 70% cutoff, i.e. effectively AT-CEILING for a
non-TMA fp32 kernel (GB300 pattern: memory micro-opts near-tie). Named lever per survey (float4 64x64
tiles / TMA bulk copy) NOT implemented per mandate; expected gain <=1.4x kernel-frame, less at wall.

### software_pipelining (fp32 2^25 elems, 402.7MB/iter, floor 50.3us)
Harness: base 0.3200 / opt 0.1519ms, verify PASS. Kernel-frame: **83.2us/iter** (host submit 22.3us/call,
hidden) = **4.84 TB/s = 60.5% HBM-SoL** (<70%). Named lever: TMA bulk-copy / deeper pipeline stages
(float4-style wide loads); NOT implemented per the GB300 memory-micro-opt near-tie pattern.

### nvfp4_dual_gemm (M=256 N=3072 K=4096; FP4 floor 0.86us, HBM floor 2.0us)
Harness (iterations=4): base 0.3704 / opt 0.0485ms, verify PASS.
Probe (200 iters, /tmp/frontF2/probe_nvfp4_dual_gemm.py):
- back-to-back: **6.68us/iter** (event span 6.58us -> GPU-saturated)
- host submit: 4.42us/call (hidden back-to-back); per-iter-sync frame: 17.5us
Split: harness wall 48.5us = **86% measurement frame / host**, kernel-frame 6.6us.
Kernel-frame %SoL: **30.3% of HBM floor / 13.0% of FP4 floor** -> ~3.3x kernel headroom remains, but the
harness number is dominated by iterations=4 + per-iter frame. Named lever (measurement): raise
`BenchmarkConfig(iterations=...)` from 4 to 50 in labs/nvfp4_dual_gemm/*_nvfp4_dual_gemm.py to amortize
the frame before any kernel claim. Named lever (kernel): mainloop latency on tiny-M (256) dual GEMM.

## Priority 3 — attention quartet closure (no optimization attempts)

| Lab | Fresh base/opt (ms) | Compute floor | %SoL (wall) | Verdict |
|-----|--------------------|---------------|-------------|---------|
| flashattention4 best_available_dense | 3.8519 / 0.1330 (28.96x) | 4.6us | 3.4% | OVERHEAD-BOUND |
| flashattention4 best_available_causal | 3.2212 / 0.1256 (25.64x) | 2.3us | 1.8% | OVERHEAD-BOUND |
| flashattention4 other 12 arms | 3.1-6.6 / 3.1-7.7 (11x no_speedup) | 2.3-4.6us | 0.06-0.15% | OVERHEAD-BOUND |
| flashattention_gluon | 0.1971 / 0.1240 | 1.15us | 0.9% | OVERHEAD-BOUND |
| flexattention | 12.3416 / 0.1519 | 0.29us (doc-masked) | 0.2% | OVERHEAD-BOUND |
| cudnn_sdpa_bench flash_sdp | 0.2606 / 0.3018 (0.86x no_speedup) | ~1.15us | 0.4% | OVERHEAD-BOUND |

flashattention4 detail (all verify PASS): the 3-7.7ms arms execute the reference/educational FA4 Python path,
i.e. abstraction overhead ~50x above the 0.13ms fused-SDPA arms, which are themselves ~30x above the roofline
floor at this micro-shape (B=2 H=8 S=2048 D=64).

All four labs run micro-shapes (B=2-8, S=256-2048, H=8, D=64) whose roofline floors are 0.3-4.6us while
walls are 0.12-7ms: the measured time is launch/framework overhead, not attention math. AT-CEILING claims
are not meaningful at these shapes; closing the quartet as OVERHEAD-BOUND. If these labs should ever
roofline, the lever is shape scale-up (S>=8k, B*H>=64), not kernel work.

## Provenance
- All runs: artifacts/runs/20260611_0*__bench__profile_none_targets_labs_* on the pod.
- Probes + ncu report + file backups: /tmp/frontF2/ on the pod.
- GPU 3 only; sibling fronts ran custom_vs_cublas on other GPUs concurrently (runs 030506/030548/031840);
  block_scaling AFTER numbers reproduced twice with consistent 81us optimized arm -> no contamination signal.
- Survey est. for block_scaling ("0.111ms on B200, host-invoke dominance suspected"): GB300 wall was 94.6us;
  host invoke is real but small (8us); the dominant non-kernel term is the harness timing frame; the
  kernel is 44.8us at 20.5% NVFP4 SoL.
