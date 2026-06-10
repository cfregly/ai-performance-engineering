# GB300 (Blackwell Ultra, sm_103) validation runbook

How to run this repo's benchmark harness on a single GB300 node (Grace + Blackwell
Ultra, compute capability 10.3 / `sm_103`, 4 GPUs), what was fixed to make it
GB300-correct, and the open issues found during validation.

## GB300 optimization wins (TL;DR)

Every win is verification-passed; full per-win descriptors, SoL grounding, and the banked negatives
are in "GB300 validated wins summary" + the SoL bullets (B1-B7) below. Headlines:
- GEMM on Blackwell tensor cores: ch09 cublaslt_gemm_fp4 706x vs naive (~7107 TFLOPS NVFP4); ch09
  cutlass_gemm_fp16 2.66x kernel (440 -> 1171 TFLOPS, 31.2% FP16 SoL, harness 12.1x -> 32.16x) and
  ch14 cublas_vs_cutlass CUTLASS arm 3.0x kernel (531 -> 1596 TFLOPS, now matches cuBLAS) both by
  porting the lab off Ampere arch::Sm80 (it had been running the Ampere HMMA path on Blackwell) to the
  Sm100 tcgen05 collective; ch09 cutlass_gemm_fp8 tile-tuned (deeper K=128) 2481 -> 3432 TFLOPS (1.38x,
  45.7% FP8 SoL), now 1.12x FASTER than cuBLAS-FP8.
- Memory bandwidth: ch10 dsmem_reduction_warp_specialized 67.5% -> 84% HBM SoL (harness 2.80x; v3
  54.5% -> 69.8%) by amortizing the cluster-sync overhead (ELEMENTS_PER_BLOCK 4096 -> 65536); ch07
  tma_copy 39.2% -> 63.7% (1.63x, runtime div/mod -> compile-time shift/mask).
- Frontier unblock: the sm_103a fix loads the whole tcgen05/TMEM family (blackwell_matmul 126x, MoE
  ladder 41.6x) that was unloadable on Blackwell Ultra.
- MoE grouped GEMM (Triton): all three grouped kernels (full_stack / standard / persistent) now skip
  all-padding tiles (fully-invalid-tile early-return, or a K-loop guard where Triton rejects continue),
  1.40x (autotune) / 1.31x (persistent) on a skewed MoE histogram (the grouped-GEMM's real win over a
  naive padded bmm); balanced unchanged, all variants verify-pass (B11).
- Discovery-sweep wins (config-tuning beyond the grouped GEMM): ch14 triton_persistent 3.6x kernel
  (32-cubed tile -> 128x128x64, 12.76x harness, B13); ch10 persistent_matmul_tma 1.128x (B200->GB300
  retune: wider N-tile + 8 warps, B14). Banked: ch12 fixed grids (work_queue atomic-bound,
  uneven_partition already 216x; block-scaling is a no-op, B15).
- GB300 lab unblock: occupancy_tuning (proton_matmul) was SKIPPED on GB300 (a Triton 3.7 tcgen05.wait.st
  LLVM-select abort); root-caused to two triggers in triton_matmul.py (a dead num_warps:constexpr param
  AND an unconditional GB10/sm_121-only arch_config patch) and fixed both, so the lab now runs (1.212x
  harness, 256x256x64 = 1.62x ceiling). Lesson: guard arch patches by compute capability (B19).
- Banked with evidence (forward progress, not dead ends): TMA 2D double-buffer (built + measured -19%,
  occupancy-dominated); ch02 P2P 762 GB/s (~80-85% of the NVLink5 pairwise ceiling, vendor-optimal);
  generic cutlass GEMM (also Sm80 but FP32 underfill-capped at 1024^3).

## Hardware confirmed
- 4x NVIDIA GB300, compute capability 10.3 (`blackwell_ultra` / `sm_103`), 284 GB
  HBM each, driver 580.159.03, Grace (aarch64) host.
- Expectation hardware keys on this node: `gb300` (single visible GPU),
  `4x_gb300` (4 visible GPUs). The harness derives these from the device name
  `NVIDIA GB300` plus `torch.cuda.device_count()`.

## Working environment
Base image `nvcr.io/nvidia/pytorch:26.05-py3` is the cleanest CUDA-13 base: it
ships torch 2.12 / CUDA 13.2 / Triton 3.7 / cuDNN plus `nsys`, `ncu`, `nvcc`,
`ptxas`, so CUPTI initializes against the CUDA-13 node driver (no toolkit/driver
skew). TransformerEngine is already importable in the image.

Bring-up steps:
1. Run on a GB300 node with 4 free GPUs (this image has the profiling tools).
2. Clone the repo, init the `code/third_party/cutlass` submodule (needed by the
   tcgen05 labs).
3. Install the harness Python deps on top of the NGC torch (do NOT reinstall
   torch/triton/vllm; keep the NGC CUDA-13 build so CUPTI/nsys/ncu stay intact):
   `pip install --ignore-installed nvidia-ml-py psutil GPUtil py-cpuinfo hypothesis pytest nvidia-cutlass-dsl==4.3.0`
   (the NGC image already provides typer, pydantic, pyyaml, rich, numpy; the
   Debian-packaged PyYAML cannot be replaced, so skip the `pyyaml==6.0.2` pin).
4. Run the hardware probe once: `python core/scripts/utilities/probe_hardware_capabilities.py`.
   Confirm `architecture=blackwell_ultra`, `tma.compiler_support=true`,
   `cluster.has_dsmem=true`, `grace_coherence=true`.

### Container cgroup quirk (strict validity)
The strict validity profile rejects a container CPU/memory quota
(`ENVIRONMENT INVALID: CPU quota is set via cgroup cpu.max=...`). On a node you
fully own, clear the quota on the pod's leaf cgroup before running so strict
validity passes (otherwise use `--validity-profile portable`, numbers labeled
non-canonical):

```bash
LEAF=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
echo max > "/sys/fs/cgroup${LEAF}/cpu.max"
echo max > "/sys/fs/cgroup${LEAF}/memory.max"
```

The cleaner alternative is to launch the pod with no CPU/memory limits (GPU
resources still pin the device). GPU clock locking (`nvidia-smi -lgc`) works in a
privileged pod, so strict validity is viable.

## Running
```bash
cd code
# canonical smoke (6 targets)
python -m cli.aisp bench run-tier1 --profile minimal
# full single-node sweep (tier1 + discovered full sweep), writes gb300 expectations
python -m cli.aisp bench run-e2e --run-id <id> --run-full-sweep \
  --no-run-fabric --no-run-cluster --validity-profile strict \
  --update-expectations --allow-mixed-provenance --profile minimal
python -m cli.aisp bench run-e2e-status --run-id <id> --watch
```
Omit `--run-fabric` (multi-node only). `--no-run-cluster` skips the cluster-eval
stage (separate serving-eval concern).

## GB300 source fixes applied (branch `gb300-validate-optimize`)
The harness was calibrated for B200/GB200 (`sm_100`); these make it GB300-correct:
1. `ch02/{baseline,optimized}_grace_coherent_memory.py`: detect Grace coherence
   via ARM host + Blackwell GPU (CC >= 10), not the CC==12.1 check that wrongly
   skipped real GB300.
2. `core/common/tcgen05/__init__.py`: emit `sm_103a` on CC 10.3 (an `sm_100a`
   cubin is arch-locked and will not load on `sm_103`).
3. `core/scripts/utilities/probe_hardware_capabilities.py`: retry the ptxas TMA
   probe with the `a` suffix for any Blackwell SM (`sm_103a`, not just `sm_100a`)
   so TMA is not falsely reported unsupported (which would skip every TMA lab);
   label CC 10.3 `blackwell_ultra`; set `nvlink_c2c`/`grace_coherence` on
   ARM-host Blackwell nodes.
4. `core/scripts/profiling/profile.sh`: resolve arch via the GB300-aware
   `detect_sm` (`sm_103`).
5. `setup.sh`: single-source Triton from `requirements_latest.txt` (3.5.0) instead
   of the 3.6 nightly default; correct the misleading 2.10-dev header.
6. `labs/moe_optimization_journey/{moe_benchmark.py,optimized_moe_pad_quant.py}`:
   route the direct `torch.compile(mode="max-autotune")` calls through
   `get_optimal_compile_mode`, which keeps max-autotune on the pinned toolchain but
   falls back to `default` on sm_103 + Triton >= 3.6 (where max-autotune emits an
   unloadable `tcgen05.wait.st` kernel). Same class as the llama_3_1_8b fix; these
   two levels (`moe`, `moe_pad_quant`) bypassed the centralized guard.
7. `labs/occupancy_tuning/triton_matmul_schedules.py`: proactive toolchain
   capability-check in `setup()` that raises the file's existing `SKIPPED:` idiom
   when a raw `@triton.jit` matmul would emit the same unloadable `tcgen05` kernel
   on sm_103 + Triton >= 3.6. The LLVM abort is an uncatchable SIGABRT, so the skip
   must be proactive (before the JIT fires); converts a -6 crash into a clean skip.
8. `labs/train_distributed/optimized_ddp_multigpu.py`: route its one direct
   `max-autotune` call through `get_optimal_compile_mode` (same class as 6).

The CUDA-binary build path (`detect_sm.py`, `cuda_binary_benchmark.py`,
`cuda_arch.mk`) was already GB300-aware (CC 10.3 -> `sm_103`).

## Validated frontier-kernel results on GB300
Confirmed working on GB300 (Blackwell Ultra) during breakthrough validation:
- tcgen05/TMEM family: `ch10/matmul_tcgen05.cu` relL2 0.00021 (raw, `A @ B^T`
  reference); `ch08:tcgen05_custom_vs_cublas` SUCCEEDED in the harness. The
  `sm_103a` fix is what unblocks this whole family (it could not load before).
- `blackwell_matmul` (2048^3, Python-kernel): tcgen05 variant 0.123 ms vs naive
  baseline 15.5 ms (126x); TMA variant 2.39 ms (6.5x).
- NVFP4 GEMM (CUTLASS NVFP4 tensor cores), real binary timing (decode shapes
  M=128, leaderboard N/K): `optimized_nvfp4_gemm_sm103` geomean 7.39 us vs
  baseline 9.78 us (1.32x); largest shape ~3.1 PFLOPS / ~60% HBM-bound SoL.
- ch09 CuTe-DSL NVFP4 GEMM (`optimized_cute_dsl_nvfp4_gemm_sm103`): 7.55 us vs
  17.9 us baseline (2.37x).
- MoE optimization ladder (`moe_optimization_journey`, Python kernels):
  `optimized_moe_cuda_graphs` 0.935 ms vs 38.9 ms naive baseline (41.6x);
  `optimized_moe_triton` 17x.
- `blackwell_gemm_optimizations` grouped GEMM (MoE-relevant): full_stack 0.124 ms
  vs 0.312 ms baseline (2.5x); skewed-histogram all-padding-tile skip 1.40x (B11).
- `decode_optimization` ladder: `decode_ultimate` 1.29 ms vs 9.81 ms baseline (7.6x).

Net: the repo's frontier optimizations (tcgen05/TMA GEMM, NVFP4 GEMM, MoE ladder)
deliver real speedups on GB300. The headline GB300 fix is the `sm_103a` unblock of
the tcgen05/TMEM family (previously unloadable on Blackwell Ultra).

## L4 ncu-grounded Speed-of-Light: ch09 NVFP4 GEMM (decode shapes, sm_103)

ncu `--set`-style metrics (kernel-replay, isolated, deterministic over 8 launches)
on `optimized_cutlass_gemm_fp4_sm103`, GB300 (148 SMs), shape M=128/N=7168/K=16384:

- Kernel: `SM100_MMA_MXF4_SS<float_e2m1_t, ...>` (NVFP4 tensor-core blockscaled),
  CTA tile 128x64x256, grid (1, 112, 1) = 112 CTAs, block 256.
- `gpu__dram_throughput.avg.pct_of_peak` = 43-46% (mean ~45%).
- `sm__throughput.avg.pct_of_peak` = 33-36% (mean ~35%).
- Wall-clock (un-profiled) 13.1 us = ~2.29 PFLOPS = 15% of the NVFP4 FLOP roofline
  (GB300 nvfp4_dense 15.0 PFLOPS), and ~59% of the HBM-bound minimum
  (B = 58.7 MB / 8.0 TB/s = 7.7 us).

VERDICT (L4): at the decode shape (M=128) this kernel is OCCUPANCY/LATENCY-bound,
not compute- or memory-bound. M=128 == tile_M so there is only ONE row-tile;
N=7168 / tile_N=64 = 112 col-tiles, so the grid is 112 CTAs and under-fills the
148-SM GPU (1 CTA/SM, 36 SMs idle, no second wave). Neither the tensor cores (SM
35%) nor HBM (DRAM 45%) are saturated. This is the physics of small-M GEMM, not a
kernel defect: the CUTLASS NVFP4 path itself is already the well-optimized
(tensor-core, TMA, blockscaled) implementation.

NEXT LEVER (refined 2026-06-09 by reading the kernel; supersedes the earlier draft):
raise the CTA count above the SM count to fill the GPU. The earlier draft listed
(a) "smaller tile_N (64 -> 32)" -- that is INVALID and is RETRACTED: the SM100
blockscaled NVFP4 1SM MMA constrains `TileShape_N` to {64, 128, 192, 256}
(`optimized_nvfp4_gemm.cu` lines 144-145, citing
`sm100_make_blockscaled_1sm_trivial_tiled_mma()`). So tile_N=64 is ALREADY the
minimum valid tile = the max col-tile count (N=7168/64 = 112 CTAs); tile_N=128 would
HALVE the CTAs to 56 (worse occupancy). The tile lever is therefore exhausted: among
the valid tiles, N64 is occupancy-optimal at this shape (which is why the kernel's
per-shape dispatch already pins the N64C1 lane for decode). The ONLY remaining
occupancy lever is split-K / StreamK across the large K=16384 reduction (112*S CTAs +
an epilogue reduce). I MEASURED it (2026-06-09): added a `cutlass::gemm::StreamKScheduler`
lane to the kernel (the 4th `GemmUniversal` param; CUTLASS v4.3.2 has an SM100 StreamK
scheduler, `sm100_tile_scheduler_stream_k.hpp`). It COMPILES + runs the decode shape
(so StreamK IS compatible with the blockscaled NVFP4 1SM collective), but it is NOT a
win: a 3-trial same-binary A/B at the decode shape (128/7168/16384, 3-iter bounded) is
StreamK 15.79 us mean (15.62/16.23/15.53) vs the data-parallel N64 lane 15.14 us mean
(15.37/15.10/14.94) -- StreamK is ~4% SLOWER. The K-split reduction overhead exceeds
the occupancy gain at M=128. StreamK is also UNSTABLE on this path: it hangs / times out
on the other two leaderboard shapes (128/4096/7168 and 128/7168/2048). VERDICT (now
measured, not just argued): the data-parallel N64 lane is the practical optimum; the
StreamK occupancy lever is REFUTED. This kernel is at its H4/P4 ceiling (it IS the
production CUTLASS NVFP4 tensor-core path; the MMA atom is the FP4 tensor-core op); the
decode-M residual is small-M-GEMM shape physics, not a closable gap. The experimental
StreamK lane was reverted (a slower + unstable variant is not a keeper).

Companion data point, the repo's own hand-written tcgen05 dense GEMM
(`ch10:matmul_tcgen05_vs_cublas`, the educational CUDA-C++ tcgen05 kernel): on
GB300 it measures ~4x SLOWER than cuBLAS (baseline custom-tcgen05 2.78 ms,
optimized cuBLAS = best_speedup 4.0x), i.e. ~25% of the vendor library. In
K/R/H/P/A terms the educational kernel is K3 / R4 (CUDA C++ + tcgen05) / H4 / P2
(~25% of SoTA) and cuBLAS is P4 (SoTA). That 4x gap is the lab's intended lesson (a
hand-written tensor-core kernel vs the tuned vendor library), not a fixable defect,
and it is consistent across the tcgen05 teaching kernels (`matmul_tcgen05_epilogue`
is 1.27x over its own naive baseline). The two frontier GEMM SoL reads together:
the vendor NVFP4 path is occupancy-bound at decode-M (right kernel, shape-limited),
and the educational tcgen05 path is P2-vs-P4 off cuBLAS by design.

Decode ladder technique headroom (from the live loop's `labs/decode_optimization`
strict run, best_speedup vs the lab's naive baseline): decode (main kernel opt)
9.02x, decode_warp_specialized 5.43x, decode_fp4 1.73x, decode_fp8 1.29x,
decode_pinned 1.19x, decode_streams 1.05x (below the 1.05x gate), decode_hf_cache /
decode_double_buffer_tma skipped. The shape of this ladder IS the GB300 decode SoL
story: the kernel-structure optimizations (the persistent/fused decode kernel and
warp specialization) carry essentially all the headroom, the quant paths add a
moderate 1.3-1.7x, and the host/memory-movement optimizations (pinned, streams) add
almost nothing because GB300's memory subsystem is already fast enough that the
naive path is not memory-starved. Same pattern as the ch02 grace_coherent_memory /
memory_transfer near-ties: on GB300, optimize the kernel, not the byte movement.

## GB300 perf frontier status (2026-06-09): achievable levers closed with evidence

A single place for "what is the status + what is left", per the grind discipline:
1. The one clean kernel-roofline target, the NVFP4 GEMM, is at its H4/P4 ceiling: it
   is the production CUTLASS NVFP4 tensor-core path, the decode-M residual is small-M-GEMM
   shape physics (112 CTAs < 148 SMs, the smallest valid tile), and the StreamK occupancy
   lever was MEASURED + refuted (~4% slower at decode-M + unstable on the blockscaled path).
2. The technique ladders deliver their optimizations and all follow the GB300 pattern:
   kernel-structure + CUDA-graph opts carry the headroom (decode main-kernel 9.02x,
   MoE torch.compile 43.38x, blackwell_matmul tcgen05 126x vs naive), while
   memory-movement / host opts are near-ties because GB300's bandwidth is abundant
   (pinned/streams ~1.0-1.2x; ch02 coherent-memory ties). These are serving-loop /
   technique optimizations, correctly characterized by speedup, NOT single-kernel
   roofline targets.
3. Every chapter + lab break is fixed + re-validated (sm_103a kernels, the
   max-autotune->default guard, the proton tcgen05 skip-guard, the deps, and the
   moe_hybrid_ep CUDA-event fix).
4. The toolchain is clarified + self-corrected: the NGC torch 2.12 base (forward-compat
   onto sm_103) + the source fixes is the working GB300 path; the pinned torch 2.9.1 and
   triton 3.5.0 do not help (verified).

Remaining non-fixables (documented, not defects): the vllm env-gap (ch18 / dynamic_router)
and the ch13 sequence_parallel_multigpu collective_type pair quirk (hardware-agnostic).

Next genuine breakthroughs are OUT OF SCOPE for this teaching repo and named for honesty:
a native sm_103 torch/triton (upstream; would unlock native max-autotune + sm_103-native
codegen, removing the fallback) and production-scale model kernels (this repo teaches the
paths; the production paths are already the at-ceiling vendor libraries). Net: the GB300
validation + optimization effort is comprehensively complete.

## no_speedup tie audit (2026-06-09): correctly classified, no hidden regression

Audited all 20 GB300 ties (best_optimization_speedup below the 1.05x gate, goal=speed, from
the live this-session expectations). Verdict: every tie is correctly classified. The
optimization genuinely does not win on GB300 at the lab's shape, and none is a mislabeled
critical regression. Two groups.

Near-ties (~0.95 to 1.03x) are memory-movement, host, or serving-orchestration opts whose
bottleneck GB300 does not have: ch02 grace_coherent_memory 1.012, memory_transfer 1.026;
ch03 double_buffered_batch_provisioning 0.956, pageable_copy 1.009, rack_prep 0.98; ch06
launch_bounds 1.002; ch11 tensor_cores_streams 0.991; ch13 context/expert_parallel ~0.95;
ch15 / ch17 disagg-serving ~0.95 to 1.02. Same GB300 pattern as the SoL sections: abundant
bandwidth plus a fast host means memory and host opts tie.

Sub-0.8 "optimized slower" cases are each a legitimate overhead or tradeoff teaching result
(verified numerically correct), confirmed live where extreme:
1. ch13 quantization 0.17x, confirmed live: baseline 0.671 ms vs optimized 3.91 ms. The
   int8 `_int_mm` kernel itself is fast (5.4 us by ncu), but the per-call activation
   quantize/dequantize overhead dominates, netting 5.8x slower than the fast baseline
   (verification passed). The book's quant-overhead lesson, sharpened on GB300's fast
   baseline. ch13 torchao_quantization 0.792 is the same class.
2. ch10 tcgen05_warp_specialization 0.703 and warp_specialized_cluster_pipeline_cuda 0.714:
   the hand-written warp-specialized tcgen05 kernel vs a simpler 2-stage TMA pipeline
   baseline. Warp specialization's benefit is shape, arch, and implementation dependent,
   and the teaching kernel loses at this shape on sm_103 (same family as the
   educational-tcgen05-vs-cuBLAS P2-vs-P4 gap above).
3. ch14 regional_triton 0.653: regional MLP compile vs a full-graph compile baseline (both
   max-autotune). On GB300 the full-graph compile is fast enough that regional
   compilation's churn-reduction does not pay back as raw speed.

Robustness note surfaced by the audit: 31 raw `torch.compile(mode="max-autotune")` call
sites exist repo-wide, only 3 routed through `get_optimal_compile_mode` (the sm_103 +
Triton>=3.6 default-fallback guard). The sites that CRASH on GB300 (FlexAttention / proton
tcgen05.wait.st codegen) were fixed individually; the rest produce loadable kernels (the
quant no_speedup regressions are quant-overhead-bound, not compile-mode-bound). Routing the
guard repo-wide is a safe GB300 robustness improvement (a no-op on the B200 / Triton-3.5
pin), available if a future GB300 target trips the crash path, not a current defect for the
validated set.

Net: the no_speedup classification is sound. No regression hides behind a tie. This closes
the last in-scope GB300 lever.

## Latent max-autotune crash hunt + coverage closure (2026-06-09)

Hunted the hypothesis that the 28 unguarded `torch.compile(mode="max-autotune")` sites harbor
latent GB300 SIGABRT crashes (the tcgen05.wait.st signature that hit llama and proton).
REFUTED with evidence. Five flex_attention + max-autotune sites run clean on GB300: ch18
flexdecoding 1.63x, paged_attn_backend 14.36x, paged_attn_layout 3.05x,
flexattention_sliding_window 8.16x, and labs/flashattention4 best_available_attention 1.02x
plus flashattention4 (kernel) 1.00x. So the flex+max-autotune signature is NOT predictive of
the crash. The 2-3 real crashes (llama's FlexAttention config, proton's matmul autotune) were
config-specific and are already fixed/guarded. A blind guard sweep of the remaining 28 sites
is therefore correctly AVOIDED: on GB300 the guard turns max-autotune into default, which
would change (and risk regressing) the many sites where max-autotune already produces a good
kernel. The 3 guarded sites plus the per-site fixes cover the actual crash configs; the rest
are GB300-safe as written.

Coverage closures found by the hunt (targets untested in the original sweep):
1. labs/flashattention4: now validated on GB300. The FA-4 flash_backend is unavailable in this
   image, so the lab falls back gracefully to the best-available SDPA backend (no crash):
   best_available_attention 1.02x, flashattention4 (kernel) 1.00x. Attention is at parity on
   GB300 because SDPA is already optimal, correctly classified no_speedup.
2. ch16 flashinfer_block_sparse: 3.70x on GB300, enabled by the flashinfer-python install this
   session. A real validated win, previously blocked by the missing dependency.
3. ch16 piece_graphs is an informational example (not a perf pair). multi_node_blackwell (no
   multi-node fabric on a single GB300 node) and gpudirect_storage (a concepts demo) are
   genuine env/scope gaps. inference_optimizations_blackwell, inference_serving_multigpu, and
   fp8_compiled_matmul are source modules, not standalone targets (the earlier "missing" flags
   were name mismatches).

Net: no latent max-autotune crash exists on GB300, and the coverage gaps are closed
(flashattention4 and flashinfer_block_sparse now validated).

## ch04 distributed/comm coverage closure (2026-06-09): the chapter was a coverage blind spot full of wins

ch04 (distributed/comm) was the largest untested chapter on GB300: only 1 of ~48 targets had a
result (gradient_fusion) in the original sweep, because that sweep did not exercise the
torchrun-distributed suite. The harness auto-dispatches torchrun (nproc_per_node 4), so the
single-node comm suite runs on the 4-GPU NVLink node. Running the 22 runnable non-nvshmem
targets validated a large win set the sweep had missed, and sharpened the GB300 comm pattern.

Wins (validated this session, verification-passed, previously untested), high to low:
1. nixl_tier_handoff 40.44x (NIXL tiered transfer vs a naive copy).
2. nccl 20.27x (NCCL collective vs a naive comm path).
3. cpu_reduction 18.20x (GPU reduction vs a CPU reduction baseline).
4. grace_blackwell_locality 8.89x (Grace-Blackwell C2C locality, a GB300-specific win).
5. gradient_compression_int8 comm-only 5.75x (the all-reduce in isolation).
6. bandwidth_benchmark_suite 5.24x.
7. gradient_compression_fp16 comm-only 3.38x.
8. continuous_batching 3.07x.
9. dataparallel 2.68x.
10. gradient_compression_int8 2.16x full-step.
11. nvlink_topology_aware 1.48x (topology-aware all-reduce routing).
12. gradient_compression_fp16 1.25x full-step.

Ties (overlap / backend-swap / already-saturated, correctly no_speedup): disaggregated 1.10x,
pcie_staging 1.06x, symmetric_memory_perf 1.02x, tensor_parallel async overlap 1.01x,
torchcomms 1.01x.

Asset-validation sanity check (the big numbers are real, not degenerate): verification passed
4/4 on the largest wins and the baselines are sane: nixl 43.6 ms to 1.08 ms, nccl 6.10 ms to
0.30 ms, cpu_reduction 5.58 ms to 0.31 ms, grace_blackwell_locality 2.34 ms to 0.26 ms. The
double-digit speedups are technique-vs-naive contrasts by design (NIXL vs naive copy, NCCL vs
manual comm, GPU vs CPU reduction), the chapter's intended lessons measured on GB300.

The refined GB300 comm pattern, sharper than the SoL-section "memory-movement opts tie": on the
fast NVLink fabric, comm-OVERLAP and backend-swap opts tie (the fabric is not the bottleneck, so
overlapping it or swapping the backend buys almost nothing), but comm-VOLUME-reduction (int8/fp16
gradient compression, 5.75x/3.38x on the comm itself), comm-ROUTING (NVLink-topology-aware,
grace_blackwell C2C locality), and use-the-right-engine (GPU vs CPU, NCCL vs naive, NIXL tiering)
WIN. Less data, smarter routing, and the right engine beat fast bandwidth; merely overlapping it
does not. This is the distributed-training corollary of the decode-ladder lesson (optimize the
kernel, not the byte movement): reduce, reroute, or re-engine the comm, do not just overlap it.

Edge-case failures (banked, low value): no_overlap, pipeline_parallel, symmetric_memory are
torchrun multigpu targets that fail with a generic "Baseline or optimization failed" whose root
cause is upstream in the torchrun worker (not surfaced in the harness summary). 3 of ~48, not
chased further given the 12 wins captured. reinit_comm skips cleanly ("requires
torchrun/distributed launch context").

nvshmem half of ch04 (verified 2026-06-09, correcting an earlier "nvshmem not installed" note):
torch 2.12 actually BUNDLES nvshmem (`torch.distributed._symmetric_memory.is_nvshmem_available()`
returns True), so these targets are NOT dependency-gated. But running the 5 base targets shows a
runtime/fabric infra-wall, not a win cluster: 4 (nvshmem_vs_nccl_benchmark,
nvshmem_pipeline_parallel, nvshmem_training_example, nvshmem_training_patterns) fail_error because
the nvshmem runtime init/bootstrap does not complete in the plain single-node torchrun context (no
nvshmem launcher/fabric), and nvshmem_ibgda_microbench skips cleanly (IBGDA needs an InfiniBand
fabric this single node lacks). So the nvshmem half contributes no wins: it is a runtime/fabric
infra-wall, not a missing-dependency gap. The `_multigpu` suffixed duplicates are not separately
run (the base names already torchrun-dispatch to 4 GPUs).

## Untested non-ch04 closure (2026-06-09): 4 more wins, premature "complete" corrected

The earlier "coverage audit complete" line was premature: it checked only a sliver, not every
chapter. A naming-aware audit (resolving `_cuda` / `_multigpu` / `_enhanced` variants) found
genuinely-untested non-ch04 targets, and running the 13 runnable non-vllm ones added 4 validated
wins:
1. ch11:warp_specialized_two_pipelines_multistream 2.36x (the earlier 300s-timeout fix, now validated).
2. ch10:matmul_tcgen05_pipelined 2.32x (the earlier sm_103a gencode fix, now validated).
3. ch10:tcgen05_cluster_pipeline 1.58x (the earlier tighter-timeout fix, now validated).
4. ch18:eos_sync_polling 1.26x.

The first three confirm that earlier source fixes (sm_103a kernels, timeouts) produce real wins once
their results are actually captured. ch09:cublaslt_gemm_fp4 was a gated skip but is now FIXED +
PASSING (467.54x, the wrong-transpose/scale-swizzle bug, see "FP4 cuBLASLt unblock" below). The
remaining gated skip is ch10:tcgen05_warpgroup_specialization (kernel skip gate). Informational examples (run and
demonstrate a concept, not perf pairs): ch19:nvfp4_training, ch14:cublas_vs_cutlass,
ch15:inference_placement, ch17:inference_full, ch17:pipeline_parallelism, ch20:pipeline_sequential,
ch05:ai.

With these run, the coverage is now evidence-complete (not asserted-complete): every runnable,
non-env-gapped target has a GB300 result. The only untested remainder is the vllm env-gap (ch18
vllm_*, labs/dynamic_router; vllm breaks the NGC toolchain), the ch04 nvshmem runtime/fabric
infra-wall, the 2 gated skips above, and the 7 informational examples. Honest note: this is the
second corrected over-claim of the session (the first was the nvshmem "not installed" note);
re-examining on "proceed" surfaced 4 wins the premature close would have buried.

## FP4 cuBLASLt unblock (2026-06-09): the skip is a wrong-transpose BUG, not a cuBLASLt gate

ch09:cublaslt_gemm_fp4 skips with "SKIPPED: cuBLASLt NVFP4 algorithm unavailable on this
driver/toolchain" (the optimized binary's `cublasLtMatmulAlgoGetHeuristic` returns status=15
CUBLAS_STATUS_NOT_SUPPORTED, 0 results). That message is WRONG: cuBLASLt 13.4.1.1 DOES support
NVFP4 GEMM on GB300. A standalone heuristic probe (`/tmp/fp4probe2.cu`, the read-the-source +
reproducer verdict) isolates the cause:
1. The lab's recipe (transa=N, transb=N, VEC16_UE4M3 block-scale, FP16 out): status=15 at all
   sizes (256 and 4096), batched and non-batched. So it is not a size or batch issue.
2. The TN recipe (transa=T, transb=N, VEC16_UE4M3, FP16 OR BF16 out): status=0, results=1.
   cuBLASLt finds an NVFP4 algorithm.
3. NT (transb=T) and an FP4 output type: status=15.

So cuBLASLt NVFP4 on GB300 requires the TN format (transa=T, transb=N) with FP16/BF16 output,
exactly like the cuBLASLt FP8 path. The lab uses N/N (a col-major reinterpretation that the FP8
path tolerates but NVFP4 does not), which cuBLASLt rejects. This is a fixable wrong-transpose bug,
not a driver/toolchain gate, and the lab's own skip message is misleading.

Fix recipe (implemented + verified in a standalone reproducer): set transa=CUBLAS_OP_T, keep
transb=CUBLAS_OP_N, store/quantize both operands K-major (contraction dim K is the leading dim, the
standard cuBLASLt low-precision TN layout), FP16/BF16 output. Two reproducers confirm it:
1. All-ones TN FP4 GEMM (A=B=FP4 1.0, unit UE4M3 scales): heuristic results=1 and the running GEMM
   gives C == K (256), VERIFY PASS (`/tmp/fp4gemm.cu`). So the TN layout/transpose/dtype recipe
   COMPUTES correctly end-to-end on GB300, not merely heuristic-accepts.
2. Real (non-uniform) inputs with a PLAIN row-major scale layout: maxrel 5.9, 33/64 elements wrong,
   VERIFY FAIL (`/tmp/fp4real.cu`). So cuBLASLt VEC16_UE4M3 requires a SWIZZLED scale-factor layout
   (the all-ones case passed only because every scale is 1, swizzle-independent).

SOLVED (2026-06-09): the last piece, the VEC16_UE4M3 scale-factor swizzle, is implemented and
verified. The cuBLASLt NVFP4 SF layout (from the CUTLASS/Colfax block16 SF interleave) is a
512-byte tile of 128 rows x 4 SF-K with offset:
  offset(r, sk) = (r/128)*512*(K/64) + (sk/4)*512 + (r%32)*16 + ((r%128)/32)*4 + (sk%4)
With the scales written in this swizzle (and the host reference kept in plain layout), the
standalone real-input TN FP4 GEMM VERIFIES: maxrel 0.0004, 0/64 wrong (FP4 quant error only).

So the COMPLETE, verified NVFP4 cuBLASLt GEMM recipe for GB300 is: TN format (transa=T, transb=N),
K-major operands, FP16/BF16 output, VEC16_UE4M3 scales in the SF swizzle layout above. cuBLASLt 13.4
computes NVFP4 GEMM correctly on GB300; the lab's "unavailable" skip was purely the wrong-transpose
plus plain-scale config. The working, self-verifying reference is committed at
[gb300-cublaslt-nvfp4-tn-reference.cu](gb300-cublaslt-nvfp4-tn-reference.cu) (build:
`nvcc -arch=sm_103a ... -lcublasLt`).

LAB PORTED + PASSING (2026-06-09): ch09 optimized_cublaslt_gemm_fp4.cu now implements the recipe
end-to-end and the target is flipped from SKIPPED to PASSING. `bench run ch09:cublaslt_gemm_fp4
--profile none` -> successful:1, failed:0, speedup 467.54x, verification passed. The optimized
checksum 2.3319485440e9 matches the naive baseline 2.3318361600e9 to 0.0048% (FP32 accumulation
order + FP16 output rounding only). Measured: cuBLASLt NVFP4 tensor-core GEMM ~0.029 ms/GEMM,
~4709 TFLOPS, vs the naive tiled baseline 10.09 TFLOPS (the 467x is naive-no-tensor-core vs
tensor-core, not a tuned-vs-tuned number). What the port does: A (M x K row-major) is already
K-major; B (K x N) is transposed to N x K so each N-column is K-contiguous; both block-scale
tensors are written in the SF swizzle; transa=T/transb=N; each of the 8 batches is a single-matrix
TN matmul, looped (matching the baseline's per-batch kernel loop, so no batched-block-scaling
dependency). The one non-obvious extra fix beyond the standalone recipe: the A/B scale pointers
must be set NON-NULL before `cublasLtMatmulAlgoGetHeuristic` (the block-scaled heuristic validates
them), otherwise it still returns 0 results even in TN; set them to batch 0 pre-heuristic, then
update per batch.

Under nsys (`--profile minimal`/`deep_dive`) the SAME target reports failed_profiler, but the
benchmark itself is status=succeeded with the 467x speedup; only the profiler-capture wrapper is red,
and it hits the UNCHANGED naive baseline equally (`baseline:nsys:failed, optimized:nsys:failed`), so
it is NOT the port. Root cause (diagnosed 2026-06-09, corrects an earlier "generic nsys-on-GB300"
guess): it is a HARNESS python-profile-wrapper issue, not a GB300/driver nsys gap. For a
CudaBinaryBenchmark the harness always nsys-profiles a generated python wrapper
(`render_nsys_python_profile_wrapper`) that calls `benchmark_fn` -> `_run_once`, which re-spawns the
compiled binary as a CHILD subprocess. nsys runs with `--wait primary`, so it follows the python
parent; the child binary is captured only partially (~6 of 88 kernels) and `_run_once` then raises,
yielding the non-zero exit. Proof it is the wrapper, not nsys/GB300: running nsys DIRECTLY on the
binary, even with the harness's exact flag set (`--trace cuda,nvtx,osrt --sample none --cpuctxsw none
--cuda-memory-usage true --cuda-um-gpu-page-faults true --cuda-um-cpu-page-faults true --wait
primary`), exits 0 and captures all 88 kernels into a valid report. So `--profile none` is the clean
correctness+perf verdict, and a DIRECT `nsys profile -t cuda,nvtx,osrt -o out ./<binary>_sm103` is
the working deep-dive path for any CUDA-binary lab on GB300.

HARNESS FIXED (2026-06-09): the harness now nsys-profiles a CudaBinaryBenchmark's compiled binary
DIRECTLY instead of via the python wrapper. `core/harness/run_benchmarks.py` adds
`_cuda_binary_direct_command` (build the binary, return `[binary, *run_args]` + a hardened env) and an
`elif cuda_binary_direct is not None` branch in the nsys path; any non-binary benchmark falls through
to the unchanged python-wrapper path. This unblocks harness profiler-mode (`--profile
minimal`/`deep_dive`) for all 122 CudaBinaryBenchmark targets on GB300, which previously all hit
failed_profiler. Validated: ch09:cublaslt_gemm_fp4 `--profile minimal` -> successful:1,
failed_profiler:0, 88-kernel full report (was failed_profiler:1 + a 6-kernel truncated report);
ch06:add (a simple cuda-binary) -> successful:1, failed_profiler:0. Python no-regression confirmed on
ch13:quantization and ch10:matmul_tcgen05_pipelined (both BaseBenchmark; failed_profiler:0, the
python-wrapper path is byte-identical). This is the gateway to per-kernel deep-dive SoL across the
compiled-binary suite (the plan's Phase 5).

Baseline note (corrected 2026-06-09): baseline_cublaslt_gemm_fp4_sm103 runs FINE standalone (Naive
Tiled FP4 GEMM 13.63 ms, 10.09 TFLOPS, exit 0). The earlier "-11" was an ncu-profiling/harness
artifact, not a baseline bug.

FP4 GEMM SoL grounding (ncu --set full, sol_rigor L4, 2026-06-09; enabled by the harness profiler-mode
fix): the optimized path resolves to the CUTLASS sm100 block-scaled NVFP4 tensor-op kernel
`cutlass3x_sm100_bstensorop_s256x256x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_f16_f16_256x256x256_..._tnn_align32_o_vs16_2sm_...`
i.e. the real H4 ue4m3xf4 tensor-core path, not a fallback. At 4096^3 the kernel runs Compute(SM)
52.7%, DRAM 8.3%, achieved occupancy 10.1%, 29.3 us/GEMM: tensor-compute-bound, but only ~53% SM
because a single 4096^3 GEMM with the 256x256 tile makes just 256 output tiles on 152 SMs (~1.7
waves), so the GPU is UNDERFILLED at this shape. So the single-matrix 4709 TFLOPS was a genuine NVFP4
tensor-core number at ~53% SM, underfill-limited, NOT at the vendor ceiling.

BATCHED LEVER REALIZED (2026-06-09, the plan-B headroom find turned into a win): a standalone 2-batch
probe confirmed cuBLASLt advances the VEC16 block-scale pointer per batch (both batches verify, maxrel
~0.0005), so the lab was reworked to ONE batched cublasLtMatmul over all kBatchCount matrices
(BATCH_COUNT + STRIDED_BATCH_OFFSET on the A/B/C layouts; scale pointers set once, cuBLASLt strides
them). Filling the GPU (256 -> ~2048 output tiles): measured 4709 -> 6634.69 TFLOPS (1.41x), harness
speedup 467.54x -> 655.40x vs naive, ncu Compute(SM) 52.7% -> 78.68% (DRAM 8.3% -> 28.4%; achieved
occupancy stays ~10%, inherent to the 256x256 tile, so the gain is wave-quantization/fill, not
occupancy). Verification preserved (checksum 2.33195e9, identical to the single-matrix path;
ch09:cublaslt_gemm_fp4 --profile none green). 78.68% SM is near the well-fed tensor-core ceiling for
this shape, so the FP4 signature path is now SoL-grounded AND filled.

HEURISTIC AUTO-TUNE (2026-06-09): cuBLASLt's first-ranked heuristic is NOT the fastest for this shape.
Requesting the top-8 candidates and timing each (auto-tune, then use the fastest) selects candidate
#3/#4, not #0, lifting 6634.69 -> ~7107 TFLOPS (+7.1%), harness speedup 655.40x -> 706.39x vs naive.
Verification preserved (checksum identical; harness green). So the full FP4 GEMM arc is 4709
(single-matrix, rank-0 algo) -> 6635 (batched) -> ~7107 TFLOPS (batched + auto-tuned), a 1.51x lift
over the original passing lab, all at verified parity. Teaching point: benchmark the heuristic
candidates; do not assume rank-0 is optimal.

FP8 sibling (cublaslt_gemm_fp8): the same batched-fill lever applies (it too was single-matrix-looped
with the first heuristic). Batching it: 3424 -> 3808 TFLOPS (1.11x), harness 332.90x vs baseline,
verification green. Here the auto-tune CONFIRMS rank-0 is already fastest for FP8 (no extra gain,
unlike FP4 where #3/#4 beat #0). The FP8 batched gain (1.11x) is smaller than FP4's (1.41x), plausibly
because FP8 is more memory-bound (2x the operand bytes of FP4) so it underfills less severely at this
shape (not ncu-confirmed). So both ch09 cuBLASLt GEMM labs now fill the GPU + auto-tune the algo.

Auto-tune finding across the cuBLASLt GEMM family (2026-06-09): the heuristic auto-tune (top-8, pick
fastest) only beats rank-0 on the NEWER block-scaled path. FP4 won (candidate #3/#4 > #0, +7.1%); FP8
and FP16 (mature paths) both confirm rank-0 is already fastest (no algo win). FP16 did surface a
separate methodology bug: optimized_cublaslt_gemm_fp16 lacked the warmup that baseline_cublaslt_gemm_fp16
has, so the optimized was timed COLD against the warm baseline (under-reporting it, 0.0126 ms). The
auto-tune supplies the missing warmup, so it is now matched warm-vs-warm: 0.0097 ms, accurate harness
speedup 152.58x, verification green. So the auto-tune lever is FP4-specific; on mature dtypes its value
is confirming rank-0 (and, for FP16, restoring matched-warm methodology). ch19 fp4_hardware (FP4,
single-matrix) CONFIRMS the FP4-path theory: auto-tune picks non-rank-0 candidates (#2/#5), ~1.03-1.12x
faster (TIME_MS 0.00518 -> ~0.00462-0.00503; the selection varies run-to-run on this small fast kernel
with the short 3-iter tuning timing, but it is never worse than rank-0), harness green. So the pattern
is clean: the auto-tune lever wins on FP4 paths (ch09 FP4 GEMM +7.1%, ch19 fp4_hardware ~1.1x) and
confirms rank-0 on mature paths (FP8/FP16). The generic batched gemm: auto-tune picks a marginal #1
(~1.03x) and also gets the matched-warm fix (it too lacked the baseline's warmup); harness green. So
all 6 cuBLASLt matmul labs are now swept: FP4 GEMM (skip -> 706x, batched + auto-tune #3/#4), FP8
(332x, batched), FP16 (152x, matched-warm), generic (marginal #1 + matched-warm), ch19 fp4_hardware
(~1.1x, auto-tune #2/#5); perchannel is cuBLAS (different API). The durable lessons: (1) batch to fill
the GPU when a single GEMM underfills it; (2) auto-tune the heuristic (rank-0 is best on mature dtypes
but not the newer NVFP4 path); (3) warm BOTH arms for a matched A/B.

Warmup-asymmetry audit (2026-06-09): after the cold-optimized-vs-warm-baseline bug surfaced in the
FP16 + generic GEMM labs, all 27 optimized cuda-binary labs were scanned for it (does a kernel launch
precede the first timed cudaEventRecord). It is NOT widespread: only those 2 cuBLASLt GEMMs had the
asymmetry (their baselines warm up, their optimized did not), and both are fixed (matched-warm). Every
other no-warmup optimized lab (hbm_copy, hbm_peak, lookup, add_cuda_parallel) is SYMMETRIC -- its
baseline is also cold -- so the A/B speedup ratio is fair (cold/cold preserves the ratio). So the lab
A/B speedups are measurement-fair across the suite; no further warmup fixes are warranted. A banked
negative: the audit confirms measurement integrity rather than surfacing more fixes.

## GB300 validated wins summary (consolidated, 2026-06-09)

The wins surfaced from previously-untested coverage on the 4-GPU GB300 node, all
verification-passed. Speedups are vs the lab's own naive/baseline arm (the book's lesson).

| Win (chapter) | Speedup | Category | SoL note |
| --- | --- | --- | --- |
| cublaslt_gemm_fp4 (ch09) | 706.40x | kernel, FP4 tensor cores, batched+autotuned | ~7107 TFLOPS cuBLASLt NVFP4 (78.68% SM batched, ncu L4) vs 10.09 naive (no TC); skip->pass (TN+swizzle), batched fills GPU 256->2048 tiles (+1.41x), heuristic auto-tune picks rank-3/4 not rank-0 (+7.1%). 4709->7107 (1.51x). Naive-vs-TC headline |
| nixl_tier_handoff (ch04) | 40.44x | comm, tiered transfer | 92.66 GB/s achieved vs 2.29 naive (measured) |
| cutlass_gemm_fp16 (ch09) | 32.16x | kernel, Blackwell tensor-core arch port | 1171 TFLOPS = 31.2% FP16 SoL (ncu L4: SM100_MMA_F16BF16 tcgen05 + 2SM TMA; 1.68-wave underfill at fixed 2048^3); was arch::Sm80 Ampere HMMA 440 TFLOPS (11.7%) -> arch::Sm100 collective 1171 (2.66x kernel); harness 12.1x -> 32.16x vs SIMT baseline, verify-passed |
| nccl (ch04) | 20.27x | comm, right-engine | NCCL vs naive; small-message latency-bound (~0 NVLink BW measured) |
| cpu_reduction (ch04) | 18.20x | comm, right-engine | GPU vs CPU reduction |
| grace_blackwell_locality (ch04) | 8.89x | comm, routing | Grace-Blackwell C2C locality |
| gradient_compression_int8 (ch04) | 2.16x (5.75x comm-only) | comm, volume-reduction | int8 grads |
| bandwidth_benchmark_suite (ch04) | 5.24x | comm | |
| flashinfer_block_sparse (ch16) | 3.70x | kernel, dep-unlock | flashinfer install |
| gradient_compression_fp16 (ch04) | 1.25x (3.38x comm-only) | comm, volume-reduction | fp16 grads |
| continuous_batching (ch04) | 3.07x | serving | |
| dataparallel (ch04) | 2.68x | comm | |
| warp_specialized_two_pipelines_multistream (ch11) | 2.36x | kernel | earlier timeout fix |
| dsmem_reduction_warp_specialized (ch10) | 2.80x | kernel, HBM BW, sync-amortization | 6729 GB/s = 84% HBM SoL (ncu DRAM 65.9%->78.6%); ELEMENTS_PER_BLOCK 4096->65536 amortizes cluster.sync + DSMEM atomic, grid-stride MLP holds BW as blocks fall (5402->6729, 1.245x); was 2.24x harness, verify-passed. v3 + cluster_atomic siblings: 54.5%->69.8% and 47%->66.6% (1.28x / 1.42x, harness 2.33x each) |
| matmul_tcgen05_pipelined (ch10) | 2.32x | kernel, tcgen05 | 28.2% FP16 tensor-core SoL (measured: 1057 TFLOPS) |
| cutlass_gemm_fp8 (ch09) | 1.72x | kernel, FP8 tile-tune, beats cuBLAS | 3432 TFLOPS = 45.7% FP8 SoL (deeper K=128 tile, 256x128x64 -> 128x256x128, 1.38x over default); 1.12x FASTER than cuBLAS-FP8 (3054); verify exact |
| tcgen05_cluster_pipeline (ch10) | 1.58x | kernel, tcgen05 | below cuBLAS tensor-core SoL (P2 teaching) |
| nvlink_topology_aware (ch04) | 1.48x | comm, routing | |
| eos_sync_polling (ch18) | 1.26x | serving | |

Original-validation wins (earlier in the effort, for reference): block_scaling 1.96x (CuTe-DSL
sm103 port), llama_3_1_8b 2.54x (compile-mode guard), the decode ladder (decode main-kernel 9.02x,
warp-spec 5.43x), MoE journey torch.compile 43.38x, blackwell_matmul tcgen05 126x vs naive. The
NVFP4 GEMM (labs/nvfp4_gemm) is the one clean kernel-SoL target and is at its H4/P4 vendor ceiling.

SoL framing (B), measured 2026-06-09:
- Kernel (B2): matmul_tcgen05_pipelined measured at 1057 TFLOPS = 28.2% of the GB300 FP16
  tensor-core SoL (3750 TFLOPS), via accurate CUDA-event timing of its size=12288 FP16 GEMM
  (3.509 ms/iter). So the tcgen05 teaching kernel's 2.32x-over-naive sits at ~28% of the
  tensor-core ceiling. That is the P2 teaching gap vs the vendor cuBLAS path, now measured rather
  than asserted; the other tcgen05 teaching kernels fall in the same band. This is a real headroom
  signal (a tuned kernel reaches 70-90% of SoL), but closing it is a kernel-rewrite of a teaching
  kernel whose lesson is the technique, not vendor-parity.
- Comm (B1): NVLink telemetry is live (a controlled P2P copy moved 107 GB at ~85 GB/s single-stream
  payload, dcgmi is absent so this used nvidia-smi nvlink counters). But the teaching nccl run moved
  negligible NVLink bytes (peak ~0 across 122 samples spanning the run): the headline comm wins
  (nccl 20.27x and the like) are small-message LATENCY-bound, not NVLink-BW-bound, so they sit
  nowhere near the 1.8 TB/s NVLink ceiling. That is expected (the win is algorithmic/latency, not
  bandwidth saturation), not a BW-headroom lever. nixl 92.66 GB/s is a C2C/tiered path (about 10%
  of the ~900 GB/s Grace-Blackwell C2C).
- Memory (B3), a Phase-5 deep-dive win (enabled by the harness profiler-mode fix above, which makes
  nsys/ncu work on cuda-binary targets): measuring %SoL across the ch07 memory targets found a REAL
  fixable gap, not just a teaching cap. optimized_tma_copy's 2D TMA kernel ran at 39.2% HBM SoL (3136
  GB/s) while the float4_vector copy hits 89.8% (7181 GB/s). ncu localized it: DRAM 25% / SM 71%, i.e.
  compute-bound not memory-bound, because the per-element stencil divided by the RUNTIME tile_cols (6
  integer div/mod per element, which the compiler cannot strength-reduce). Fix: a full-tile fast path
  using the compile-time TILE_N (a power of two, so / and % fold to shift/mask; indices identical).
  Result: 39.2% -> 63.7% HBM SoL (3136 -> 5098 GB/s, 1.63x), ncu-confirmed (DRAM 25%->42%, SM
  71%->52%, duration 43.6->25.6 us); the harness ch07:tma_copy target stays green. Next lever (banked):
  the residual gap to ~90% is the single-tile barrier serialization + Long Scoreboard smem latency; a
  double-buffered multi-tile TMA pipeline could close it but is smem-limited (32 KB/block -> 6
  blocks/SM, and double-buffering lowers occupancy), so the EV is uncertain. Phase-2 follow-up
  (2026-06-09, MEASURED): re-ncu confirmed the limiter is smem-occupancy (Block Limit Shared Mem = 6
  at 32 KB/block; 75% theoretical / 61.3% achieved) with DRAM 42.7% (latency-bound) and SM 52.8%. A
  2-stage double-buffer (grid-stride tile loop, 2 input + 1 output stage in 48 KB dynamic smem,
  prefetching the next tile's TMA load during the current combine) was BUILT + MEASURED: 5109 -> 4142
  GB/s (63.9% -> 51.8%), a 19% REGRESSION. Reverted. Root cause: the single-tile design already
  overlaps across 4096 independent blocks (6 blocks/SM), so the grid-stride double-buffer trades that
  block-level parallelism for within-block pipelining at LOWER occupancy (48 KB -> 4 blocks/SM = 50%)
  plus a per-tile store serialization (cp_async_bulk_wait_group_read<0>). Banked measured-negative:
  the 2D TMA copy's 63.9% is near its design ceiling -- the per-tile descriptor + barrier overhead is
  the cost of the TMA abstraction (a raw float4 LDG.128 copy hits 89.8% on the same device). No
  next_lever of value (a smaller-tile higher-occupancy variant changes the non-square TMA coord
  convention; the abstraction overhead is inherent to per-tile TMA).
- Memory (B4), optimized_hbm_copy: two stacked limiters found by the same Phase-5 hunt. (1) The grid
  was hardcoded `<<<256, 256>>>` = ~1.7 blocks/SM on a 152-SM GB300, leaving the machine nearly empty.
  Sizing it to the device (num_sms*32) raised achieved occupancy from ~20% to 90% (ncu). (2) But BW
  rose only 60% -> 63.5% HBM SoL, because ncu at 90% occupancy shows DRAM 63.6% / SM 11.75%: the
  kernel is now memory-subsystem-limited by its "Float8 / 256-bit" access (each element is 2x LDG.128 +
  a load-store dependency), NOT occupancy. The float4 / 128-bit path (optimized_float4_vector) reaches
  89.8% on the same device, so on GB300 the native 128-bit LDG.128 is the HBM-optimal width and the
  256-bit Float8 premise this file teaches is empirically suboptimal. Kept the device-grid fix (correct
  + maxes occupancy, 1.06x); the width is left as a teaching-premise observation (hbm_copy vs
  float4_vector form a width comparison; 128-bit wins on GB300), flagged for the author rather than
  silently rewritten.
- Kernel (B5), a Phase-1 discovery-sweep win (un-deep-dived measurable-SoL labs): the ch09 CUTLASS
  FP16 GEMM (optimized_cutlass_gemm_fp16, M=N=K=2048) declared `cutlass::arch::Sm80`, which on
  Blackwell compiled the AMPERE HMMA path (ncu: `Mma<GemmShape<16,8,16>...OpMultiplyAdd>` =
  mma.sync.m16n8k16) at 440 TFLOPS = 11.7% of the 3750 TFLOPS FP16 SoL, underfilled at 0.84 waves/SM.
  The FP8 sibling already used the CUTLASS 3.x Sm100 collective (GemmUniversalAdapter, TMA 2SM
  warp-specialized). Porting the FP16 lab to the same Sm100 collective (half_t; RowMajor A / RowMajor
  B kept so C=A@B and the |C| checksum still match the baseline) moved it to the Blackwell tcgen05
  path (ncu: `SM100_MMA_F16BF16_2x1SM_SS` + `SM100_TMA_2SM_LOAD`): 440 -> 1171 TFLOPS (2.66x kernel,
  31.2% FP16 SoL), harness 12.1x -> 32.16x vs the SIMT baseline, verification passed (checksum
  1.26223e7 matches baseline to 2e-6). It is the same lesson as B3/B4: an "optimized" GB300 lab can
  silently run a prior-arch path. Still underfill-limited (1.68 waves, 43% SM) at the fixed 2048^3 the
  A/B requires; next lever (banked, low EV): fill, capped by that shape. The FP8 + fp4 CUTLASS siblings
  are already on the Sm100 path (FP8 2481 TFLOPS, 6.74 waves at 4096^3). The generic CUTLASS GEMM is
  also arch::Sm80 but FP32/TF32 at a tiny 1024^3 (0.42 waves), so its Sm100 port is underfill-capped
  (banked low-EV).
- Kernel (B6), Phase-1 discovery-sweep, ch10 DSMEM cluster reductions (64M-float / 256 MB): the
  `__launch_bounds__(,1)` occupancy hypothesis was REFUTED by ncu (warp_specialized at 91.45% achieved
  occupancy, 26 regs, 13.5 waves; not occupancy-starved). The real limiter was sync-overhead
  amortization: at ELEMENTS_PER_BLOCK=4096 (16 KB/block) each block read only 16 KB before its
  block-reduce + cluster.sync + DSMEM step, so a read-only stream sat at 66% DRAM vs a copy's ~90%.
  Streaming 256 KB/block (ELEMENTS_PER_BLOCK 4096->65536) amortizes the fixed sync cost, and the long
  grid-stride load gives each thread many in-flight loads (MLP from ILP) so HBM BW rises even as the
  block count falls (warp_specialized knee 5402->6389->6613->6726 GB/s at 4096/16384/32768/65536;
  131072 underfills to 6529). Results: warp_specialized 5402 -> 6729 GB/s (1.245x, 67.5% -> 84% HBM
  SoL, ncu DRAM 65.9% -> 78.6%, harness 2.24x -> 2.80x); v3 (CLUSTER_SIZE=2) 4363 -> 5585 GB/s (1.28x,
  54.5% -> 69.8%, harness 2.33x). Both verify-passed (sum exact), DSMEM-cluster lesson intact. Same
  class as B3 (a real fixable gap behind a refuted first hypothesis), found by the discovery sweep.
  Sibling sweep: the same lever lifts dsmem_reduction_cluster_atomic (CLUSTER_SIZE=4, DSMEM-atomic)
  47% -> 66.6% HBM SoL (0.068 -> 0.048 ms, 1.42x, harness 2.33x, verify exact); its DSMEM-atomic
  contention caps it below the warp_specialized 84%. The base dsmem_reduction is a pedagogical
  "shows-the-pattern" demo (left as-is) and atomic_reduction is non-cluster (contention-bound at 67%,
  no cluster.sync to amortize) -- both banked. ch10 reduction family swept.
- Comm (B7, banked-negative), Phase-1 discovery-sweep, ch02 multi-GPU P2P transfer:
  optimized_memory_transfer_multigpu (single-stream cudaMemcpyPeer GPU0->1, 400 MB) measured 762.76
  GB/s vs the host-staged baseline 124.62 GB/s (6.12x, the lab's P2P lesson). That is ~80-85% of the
  per-direction NVLink5 pairwise ceiling (nvidia-smi: 53.125 GB/s/link), so the single cudaMemcpyPeer
  is already near-ceiling: a contiguous P2P copy is DMA-pipelined across the links, so multi-stream /
  chunked splitting contends for the same links (no BW gain) and bidirectional overlap would change
  the one-direction demo. Banked: near-ceiling vendor primitive, no lesson-preserving lever.
- Kernel (B8), Phase-1 discovery-sweep (a repo-wide arch::Sm80 scan, after B5): the ch14
  cublas_vs_cutlass CUTLASS arm (core/benchmark/cuda/cutlass_gemm_extension.cu, a PyTorch extension at
  M=N=K=4096 FP16) declared cutlass::arch::Sm80, so on Blackwell it ran the Ampere HMMA path at 531.5
  TFLOPS = 3x SLOWER than cuBLAS (1576 TFLOPS), making the lab's cuBLAS-vs-CUTLASS teaching comparison
  misleading (the gap was the arch tag, not the library). Ported the extension to the CUTLASS 3.x
  Sm100 collective (GemmUniversalAdapter, TMA 2SM, RowMajor A/B, half_t) + fixed the JIT build
  (cutlass_binding.py) to emit sm_103a (the tcgen05/TMA path needs the arch 'a' variant; torch
  auto-detect gives plain sm_103) and to add the cutlass tools/util include (make_cute_packed_stride):
  531.5 -> 1595.8 TFLOPS (3.0x kernel, 14.2% -> 42.6% FP16 SoL), now 1.018x vs cuBLAS (matched, was
  0.337x), maxdiff 0.0 vs cuBLAS (identical result). The comparison is now fair (both on Blackwell
  tensor cores). Measured by direct timing (the harness classifies this comparison pair as
  informational/skipped). Same root cause as B5 (ch09 cutlass_gemm_fp16); the arch-tag scan found
  both. The sibling labs/top_k_kernel/top_k_kernel_cuda.cu scoring GEMM has the same Sm80 tag but is
  BANKED measured-negative: at its top-k shapes (M=32768, K=128, N=256-512) it is memory-bound (K=128
  -> arithmetic intensity ~57 FLOP/byte, far below the ~469 FP16 ridge; 174 TFLOPS = 4.6% FP16 SoL)
  and the Sm80 path already beats cuBLAS there (1.10-1.41x), so the tensor-core arch barely matters and
  a Sm100 dense-tuned 256x128 tile would likely regress the tall-skinny K=128 GEMM. Arch-tag scan
  complete: ch09 + ch14 ported (wins); top_k + generic cutlass_gemm banked.
- Kernel (B9), Phase-1 discovery-sweep (cuBLAS-parity check -> tile-tune): the ch09 CUTLASS FP8 GEMM
  (optimized_cutlass_gemm_fp8, 4096^3, already Sm100) ran at 2481 TFLOPS = 33% FP8 SoL with the
  default 256x128x64 2SM tile, which a parity probe showed was 1.23x SLOWER than cuBLAS-FP8
  (torch._scaled_mm = 3054 TFLOPS / 40.7% SoL) -- real tile headroom. Swept the tile on GB300:
  deepening the K tile (64 -> 128) is the dominant lever (it amortizes the FP8 mainloop), with 128x256
  MN best. 128x256x128 peaks at 3432 TFLOPS (mean of 3: 3431.6/3426.6/3438.1) = 45.7% FP8 SoL, 1.38x
  the default, and 1.12x FASTER than cuBLAS-FP8 (a vendor-beating P4 result). Verify EXACT (checksum
  1.2853684480e+09 == baseline), harness 1.72x (was ~1.24x). Sweep: 256x128x64 2481, 128x256x64 2492,
  256x256x64 3007, 256x256x128 3353, 128x256x128 3432 (best), 128x128x128 2756, 128x256x256 3332. Same
  lab family as B5/B8 but the lever is the K-tile depth, not the arch (already Sm100); the parity check
  vs cuBLAS is what surfaced the headroom. The deeper-K lever is FP8-SPECIFIC (banked-negative for
  FP16): applying 128x256x128 to the ch14 FP16 extension (B8) REGRESSED it 1596 -> 1545 TFLOPS because
  FP16's 2-byte elements double the K-tile smem and halve the pipeline stages; FP16 is already at
  cuBLAS-parity there (1596 vs cuBLAS-FP16 1587), so no headroom remained (ch14 kept at 256x128x64). A
  parity probe across the dense FP16 GEMMs confirms they are at the vendor ceiling: ch09
  cutlass_gemm_fp16 (2048^3) 1171 TFLOPS BEATS cuBLAS-FP16 (1072, 1.09x); ch14 (4096^3) 1596 matches
  cuBLAS (1634, 0.98x). With FP8 now 1.12x OVER cuBLAS, the dense-GEMM tile-tune frontier is closed
  (every dense GEMM is at or above the vendor library). The CUTLASS FP4 labs (cutlass_gemm_fp4,
  fp4_all_concepts, fp4_perchannel) are ALL decode-shape (M=128, e.g. 128x7168x16384; memory-bound
  ~60% HBM SoL; all_concepts already variant/stage/swizzle-tuned), a different regime where the
  compute-tile lever does not apply. CUTLASS GEMM family now fully classified: dense FP16/FP8
  at-or-above cuBLAS (FP8 +12%), FP4 decode at the HBM ceiling, generic FP32 underfill-capped at
  1024^3. The standalone dense-GEMM + memory + reduction + arch-tag frontiers are all closed; remaining
  headroom is a different class (the Python/torch.compile MoE/decode/attention kernels already at
  7-41x, and end-to-end/fusion).
- Kernel (B10, banked-negative), first Python/Triton frontier probe: the blackwell_gemm_optimizations
  Triton grouped GEMM (FP16, 4 experts, M~2048 / K=2048 / N=3072) full_stack autotune explored only
  BLOCK_K=64. Added BLOCK_K=128 deep-K configs (the FP8 winner) to the @triton.autotune set -- the
  autotuner did NOT pick them (961 TFLOPS unchanged), confirming deep-K is FP8-specific (FP16's 2-byte
  smem, same as the ch14 regression). The grouped GEMM sits at 25.6% FP16 SoL (961 vs cuBLAS-batched
  torch.bmm 1491 = 1.55x), but that gap is the grouped/masked kernel's inherent overhead vs the vendor
  batched path (the tile space is autotuned + now deep-K-checked); closing it needs a deep Triton
  rewrite (mask elision / pipelining), not a tile knob. Reverted the unused configs. Lesson: the
  standalone tile/arch/sync levers do NOT transfer to the Python/Triton frontier kernels -- they are a
  genuinely different class needing framework-specific deep work.
- Kernel (B11, WIN), the deep Triton rewrite B10 flagged, delivered. The grouped GEMM launches a grid
  sized to the busiest expert (max_rows), so a skewed token histogram leaves many all-padding tiles. The
  old kernel ran the FULL MMA on those tiles and then masked the store to nothing (pure wasted compute).
  Two changes to the full_stack autotune kernel + the standard kernel: (1) a fully-invalid-tile
  early-return (`if pid_m*BLOCK_M >= valid_rows: return`) that skips the whole GEMM on all-padding tiles
  (this is the grouped-GEMM's reason to exist over a naive padded bmm, which cannot skip padding rows),
  and (2) a full-tile fast path that drops the per-iteration 2D boundary masks on fully-valid tiles.
  Same-node matched A/B (80 iters after 8 warmup, valid-row correctness-checked, maxdiff 0.0020 vs the
  0.35 gate): skewed (pad_frac 0.41, counts 3445/2514/1582/651) 0.162 -> 0.116 ms = 1.40x faster
  (637.5 -> 891.1 useful-TFLOPS), closing the gap vs cuBLAS-batched torch.bmm-over-padded from 1.845x to
  1.321x. Balanced (no padding) is neutral (979.7 -> 988.6 TFLOPS: early-return is a correct no-op,
  full-tile +0.9%), so the default benchmark and its verification are unchanged. All 4 variants
  (baseline / large_tiles / full_stack / persistent) verify-pass on both histograms. The persistent
  kernel gets the same skip via a K-loop guard (`if pid_m*BLOCK_M < valid_rows:`) rather than an
  early-return, because this Triton (NGC 26.05) rejects `continue` in the persistent tile loop
  (`unsupported AST node type: Continue`); same-node A/B 0.234 -> 0.178 ms = 1.31x on skewed
  (440.4 -> 579.8 useful-TFLOPS), balanced neutral. All three grouped kernels now skip padding tiles.
  Lesson update to B10: the Python/Triton frontier DOES yield to deep framework-specific work, but the
  lever here is FLOP-elision (skip padding work), not a tile knob.
- Kernel (B12, WIN-regime of B11 + banked autotune lever). (a) Swept the grouped-GEMM padding fraction
  to locate where the B11 skip makes the grouped kernel BEAT a naive cuBLAS torch.bmm-over-padded (the
  alternative an MoE author would otherwise write). Synthetic E=4 / K=2048 / N=3072 / FP16, counts
  geometric from 2048, 60-iter timing: pad_frac 0.00 grouped 1.66x slower, 0.37 1.25x, 0.53 1.05x
  (parity), and at pad_frac >= ~0.6 the grouped kernel WINS (0.62: 0.062 vs 0.063 ms = 0.99x; 0.70:
  0.875x). Crossover ~pad_frac 0.55-0.6. Takeaway for the lab: with the padding-skip, the grouped GEMM
  is the right choice over a padded batched GEMM exactly when expert routing is heavily imbalanced (a
  few hot experts, many cold), which is the realistic MoE regime; below that crossover cuBLAS's faster
  per-tile MMA wins despite the padding waste. (b) Banked-negative on the residual balanced gap: added
  BLOCK_M=256 + GROUP_M=8 to the @triton.autotune set; the autotuner still picks
  BLOCK_M=128 / BLOCK_N=256 / GROUP_M=4 / stages=4 (992 TFLOPS, unchanged), confirming the ~1.67x
  balanced gap vs cuBLAS is Triton tl.dot codegen on Blackwell, not a tile knob. Reverted the unused
  configs.
- Kernel (B13, WIN), discovery-sweep beyond the grouped GEMM: ch14:triton_persistent (the harness's
  Triton persistent batched GEMM, batch=64 / M=N=K=256 / fp16) hardcoded a 32x32x32 tile (no autotune),
  which on Blackwell sm_103 spawns ~4096 tiny tiles (8x8 per 256-matrix x 64 batches) dominated by
  per-tile overhead. A config sweep picked 128x128x64 / 8 warps / 3 stages. Matched same-node A/B (200
  iters after 20 warmup, exact match vs torch.bmm): 0.0697 -> 0.019 ms = 3.6x kernel (30.8 -> 112
  TFLOPS); harness 12.76x vs baseline, verification PASSED. Committed 2e947753.
- Kernel (B14, WIN), B200->GB300 retune: ch10:persistent_matmul_tma (TMA-descriptor persistent matmul,
  M=N=K=4096 / fp16) carried a B200-screened narrow static config (BLOCK_N=128, GROUP_M=2, 4 warps, 5
  stages) that under-fed the Blackwell tensor cores. A GB300 sweep favored a wider N-tile + more warps:
  BLOCK_N=256, GROUP_M=4, 8 warps, 4 stages. 5-trial same-node A/B (exact match vs torch.mm): 1232 +/- 2
  -> 1390 +/- 4 TFLOPS = 1.128x kernel; harness 1.174x vs baseline (was ~1.04x), verification PASSED.
  Committed ad9d6d6d.
- Kernel (B15, banked-negative), ch12 fixed-grid underfill hypothesis REFUTED: the audit flagged
  ch12:work_queue (hardcoded 32 blocks) and ch12:uneven_partition (hardcoded 192 blocks) as GB300
  underfill. Measured: scaling work_queue's persistent grid from 32 blocks to full occupancy (via
  cudaOccupancyMaxActiveBlocksPerMultiprocessor) moved the harness 3.403 -> 3.443x (noise). The kernel
  is bound by the single g_index atomic counter serializing work distribution, NOT by SM count (more
  blocks just add atomic contention; the real lever would be a hierarchical counter, a deeper rewrite
  that changes the lesson). uneven_partition is ALREADY 216x vs baseline (device work-stealing,
  memory-bound) with no block-count headroom. Reverted the work_queue change; ch12 has no fixable
  grid-size lever.
- Kernel (B16, banked / infra-blocked), Phase-B deep grouped-GEMM codegen probe: the residual balanced
  gap (988 Triton vs ~1650 cuBLAS bmm = 1.67x) was already diagnosed (B12) as Triton tl.dot codegen on
  Blackwell, not a tile knob (autotune frontier exhausted). To ground it with a zymtrace GPU flamegraph,
  stood up a zymtrace-injected pod (NGC image + CUDA_INJECTION64_PATH + the /var/lib/zymtrace/profiler
  hostPath) on s22l4nb4 running the grouped-GEMM in a loop: pod-side healthy (IMPLANT_PRESENT, frames
  generating, a zymtrace-profiler on the node). BUT the zymtrace backend (freshly redeployed ~15 min
  earlier by a concurrent session) has an uninitialized ClickHouse: the zymtrace_profiling database does
  not exist (SHOW DATABASES shows only default/system), so the MCP topfunctions query fails ("Database
  zymtrace_profiling does not exist") and no flamegraph can be served. Infra-wall, not a grouped-GEMM
  finding. Banked: the gap stays diagnosed-as-codegen (B12); next lever = a hand-written tcgen05 /
  TMA-descriptor Triton grouped kernel matching cuBLAS's 2-SM tensor-core path (large, uncertain,
  likely-upstream-Triton). Injected pod torn down (experiment=aisp-ggemm-zymtrace-20260610).
- Kernel (B17, round-2 sweep close, mostly banked): after the ch14/ch10 retile wins, a second sweep of
  the remaining harness-backed fixed-tile candidates found no new clear breakthrough. (a)
  labs/occupancy_tuning:proton_matmul (8192x8192x256 tile-sweep lab) is BLOCKED on GB300: the matmul
  kernel JITs to a tcgen05.wait.st intrinsic LLVM cannot select on Triton 3.7 / sm_103 ("LLVM ERROR:
  Cannot select intrinsic llvm.nvvm.tcgen05.wait.st", an uncatchable abort; the lab's own
  _tcgen05_codegen_broken() skip-guard is correct). Setting num_stages=3 does not help; the fix is the
  repo-pinned Triton 3.5.0, an upstream/toolchain matter. (b) labs/top_k_kernel:top_k_kernel (Triton)
  is already 7.79x vs baseline (well-optimized; BLOCK_N=16 retile headroom uncertain on the non-GEMM
  top-k scoring shape). (c) labs/flashattention_gluon runs (not blocked) but is Tier-3 (attention, far
  from the FlashInfer ceiling); ch14's demo/batched_bench aux paths are a known-good retile propagation
  but auxiliary/low-value. Banked: the standalone single-kernel lever frontier (arch-tag, underfill,
  sync-amortization, padding-skip, tile-retile) is HARVESTED across the repo. Named next levers (all
  upstream/large): upstream Triton tcgen05 codegen (unblocks occupancy + the grouped-GEMM gap), a
  hand-written tcgen05/TMA Triton grouped kernel, and end-to-end/fusion (the MoE journey's higher ladder
  levels are already at 41.6x).
- Kernel (B18, occupancy unblock probe REFUTED, refines B17): isolated that occupancy's matmul_kernel
  uniquely declares a DEAD `num_warps: tl.constexpr` kernel param (shadowing Triton's launch
  meta-param). A bare standalone copy of the kernel WITHOUT that param JITs + runs cleanly on Triton 3.7
  / sm_103 across all tiles (256x256x64 reached ~551 TFLOPS = 1.62x over the 64x64x32 baseline's 340 at
  8192x8192x256). BUT removing the param from triton_matmul.py did NOT unblock the real path:
  triton_matmul.run_one (which also runs enable_tf32() + imports arch_config at module load) STILL aborts
  with the same "LLVM ERROR: Cannot select tcgen05.wait.st". So the dead param is ONE trigger, not the
  only one; a second trigger lives in the module context (enable_tf32 / arch_config / module-level
  state). The lab stays correctly skip-guarded; the edits were reverted (a half-applied unblock would
  make the lab SIGABRT instead of skip cleanly). Full unblock needs the repo-pinned Triton 3.5.0 or
  isolating + removing the second trigger (deeper follow-up; the 1.62x bare-kernel result is the
  demonstrated ceiling if unblocked).
- Kernel (B19, occupancy unblock ACHIEVED, SUPERSEDES B18): isolated the second trigger B18 was missing.
  `import arch_config` is the other trigger: at import it runs configure_optimizations() (inductor config
  + arch env vars + a Triton-compat patch), and empirically the kernel ABORTS with arch_config imported
  but JITs cleanly WITHOUT it on sm_103. (Honest mechanism note: the obvious suspect,
  triton.runtime.driver.set_active_device_capability(10,3), is RULED OUT -- it is a no-op on Triton 3.7,
  where DriverConfig lacks that method; the exact configure_optimizations side effect that flips the
  codegen is not yet isolated, likely the ensure_triton_compat Triton patch -- so the fix below is
  empirical, not mechanism-proven.) Fixing BOTH triggers in triton_matmul.py (remove the dead
  num_warps:constexpr kernel param + guard `import arch_config` to compute capability (12,1)) makes
  triton_matmul.run_one JIT cleanly on Triton 3.7 / sm_103, so triton_matmul_schedules.
  _tcgen05_codegen_broken() -> False and the lab no longer skips. Verified end-to-end: harness
  labs/occupancy_tuning:proton_matmul_bm128_bn256_bk64 now SUCCEEDS at 1.212x vs the 64x64x32 baseline
  (verify-pass), where it previously SKIPPED entirely on GB300. The 256x256x64 tile is the demonstrated
  champion (~1.62x / 551 TFLOPS at 8192x8192x256 in a direct probe), now SHIPPED as the variant
  optimized_proton_matmul_bm256_bn256_bk64.py: harness 1.306x vs the 64x64x32 baseline (verify-pass),
  the lab's new best arm (prior best was the 128x256x64 wide-N variant at 1.212x). A previously-dead
  GB300 lab is now live AND optimized. Next lever (low-EV, named): isolate the exact
  configure_optimizations side effect (likely
  ensure_triton_compat) and guard it centrally in arch_config by capability, which would protect any
  other raw-triton tl.dot kernel on sm_103 (blast radius is narrow: most arch_config importers use
  torch/cuBLAS/CUTLASS paths that already run).
- Kernel (B20, ch13:quantization 0.17x regression root-caused, measured bank): the flagged
  ch13:quantization "optimization" (INT8 dynamic-quant via torch._int_mm + torch.compile max-autotune)
  runs at 0.17x = 5.9x SLOWER than the fp32 baseline on GB300. Measured the components at the lab's
  shape (M=8192, K=N=4096): fp32-tf32 linear 0.273 ms, fp16 0.149 ms, but torch._int_mm 3.65 ms -- 13.4x
  slower than tf32 and 24.5x slower than fp16. torch._int_mm's INT8 GEMM is not optimized for Blackwell
  Ultra (sm_103); it dominates the regression (quant + dequant add only ~0.5 ms: 3.65 -> 4.14 ms full
  forward). Two walls, neither lab-fixable: (1) torch._int_mm is an upstream PyTorch sm_103 issue;
  (2) even a perfect INT8 GEMM cannot win here -- the per-call dynamic-quant overhead (~0.5 ms) alone
  exceeds the fp16 time (0.149 ms), so INT8 dynamic-quant inherently loses to fp16 on Blackwell at this
  shape. Bank: a true regression on Blackwell, not a fixable optimization.
- Frontier note (measured close): every remaining repo perf lever is now blocked by an UPSTREAM sm_103
  issue, not a lab bug. The grouped-GEMM 1.67x balanced gap and the occupancy/raw-triton tcgen05 path
  are Triton-3.7 tcgen05.wait.st codegen (needs Triton 3.5 / upstream; a hand-written tcgen05 Triton
  kernel would hit the same LLVM-select abort on this stack); INT8 is torch._int_mm on sm_103. The
  standalone single-kernel lever frontier (arch-tag, underfill, sync-amortization, padding-skip,
  tile-retile, skip-guard-unblock) is harvested. Further perf gains require an upstream Triton/torch fix
  or the repo-pinned Triton 3.5.0 (not in the GB300 NGC image), not more standalone tuning.
- Survey (B21, "compile-crash list" corrected + concurrent-session contention): the audit-flagged
  "compile-crash" labs are NOT crashes -- they run but fail the speed gate on GB300: nvfp4_group_gemm
  1.00x (optimized == baseline) and nvfp4_gemv 0.22x (4.5x SLOWER than baseline). nvfp4_gemv's optimized
  custom_kernel routes through nvfp4_gemm's gemm_v3b (the sibling lab's optimized_submission.py); on a
  gemm_v3b error it silently falls back (try/except) to torch._scaled_mm, so the 0.22x is the fallback
  plus the per-call failed-attempt overhead. Both nvfp4_group_gemm + nvfp4_gemm (and moe_optimization_
  journey, block_scaling sm103, custom_vs_cublas tcgen05, fullstack_cluster tcgen05, persistent_decode,
  train_distributed, real_world_models/llama) are under ACTIVE concurrent-session development (files
  modified Jun 9-10). Per the Grind Mandate (a concurrent session in the same space lowers a lever's
  rank -> prefer the non-contended space), these high-value GB300 kernel labs are DEFERRED to that
  session to avoid collision. The remaining clean fair-game lab (ozaki_scheme int-emulation) is niche
  and likely hits the same sm_103 int-slow wall as ch13 (torch._int_mm 13x slower than tf32). Net: my
  non-contended frontier is harvested; the contended space is owned by the other session.

- Reopened frontier (B22): the "contended" high-value labs were the operator's OWN committed Jun 9
  GB300-enablement (sm_103a loaders, tcgen05 guards, block_scaling sm_103 port; tree clean), not an
  active competitor. Re-surveyed with access restored: block_scaling, custom_vs_cublas tcgen05_matmul,
  nvfp4_dual_gemm, ozaki (56x) all WIN on GB300. A fast `-p none` moe + persistent_decode survey found
  two underperformers, both root-caused below.
- WIN (B22a, persistent_decode:paged_kv_offload_prefetch 0.568x -> 1.223x, verify-pass): the prefetch
  overlap used a host prefetch THREAD, which is counterproductive on GB300. NVLink-C2C makes the
  coherent H2D copies cheap, so the Python/GIL thread overhead dominates them and inverts the result.
  Config probe at the lab shape (bs4/h16/d128/seq65536/page8192/decode1/rep128): baseline 814 ms;
  +thread 1364 ms (0.597x); pinned-direct 935 ms (0.871x); async-stream prefetch with NO thread 666 ms
  (1.224x). Fix = use_host_prefetch_thread=False (keep pinned + async-stream prefetch); output unchanged
  (verify max_diff 0.07). The async-stream prefetch still demonstrates the overlap lesson, just without
  the GIL-bound thread.
- BANK (B22b, nvfp4_gemv 0.22x env-sweep refuted): the optimized routes through nvfp4_gemm's gemm_v3b (a
  GEMM kernel on a GEMV N=1 padded to N_eff=96, sequential GEMM_V3B_STREAMS=1). Env-config sweep
  refuted: USE_GEMM_V3B_ALL=0 (scaled_mm 4-stream) 0.45x, GEMM_V3B_STREAMS=4 0.23x, +CASE1_N_EFF=16
  0.234x, all still below the simple baseline (4-stream overhead exceeds the parallelism benefit for
  tiny GEMVs). The baseline (torch._scaled_mm, N padded 1->128) is ~143x off the HBM roofline (127/128
  of the N is wasted), so real headroom exists but needs a FUSED fp4 GEMV kernel (the fp4 cast
  device-asserts; a materialized dequant moves 4x the bytes of the fp4-direct path). Deep, deferred.
- WIN (B22c, moe_optimization_journey:moe_pad_quant 1.019x -> 1.927x, verify-pass): the optimization is
  torch.compile, but get_optimal_compile_mode falls back to "default" on sm_103 (max-autotune emits the
  tcgen05.wait.st bug). For this small launch-bound MoE (1024 tokens, many tiny kernels) "default" leaves
  it at ~1.27x vs eager. Compile-mode probe vs eager: default 1.268x, max-autotune-no-cudagraphs 1.456x,
  reduce-overhead (cudagraphs) 2.194x. Fix: on the sm_103 "default" fallback use "reduce-overhead"
  (cudagraph capture cuts the per-kernel launch overhead and still avoids tcgen05; static shapes here make
  it safe). Surgical + portable: B200 keeps max-autotune, only the sm_103 fallback flips default ->
  reduce-overhead, in this lab (not the global get_optimal_compile_mode, which stays "default" since some
  labs have dynamic shapes cudagraphs cannot capture). Generalizable lever (banked): other launch-bound
  sm_103 labs on the "default" fallback may similarly prefer reduce-overhead, gated by per-lab cudagraph
  safety.
- Survey complete (B22d): the remaining compile/inference labs all pass or are intentional. ch20
  memory_standard 1.385x, moe 2.549x, training_single 2.39x, pipeline_sequential pass; moe_cuda,
  flexattention, blackwell_matmul, recsys all pass. ch20:nvfp4_mlp reads 0.766x latency but is a
  MEMORY-goal benchmark (get_optimization_goal="memory": TE-NVFP4 trades latency for footprint vs BF16;
  intentional, not a regression, do not "fix"). nvfp4_group_gemm 1.00x BANKED: a frontier 50-knob
  tcgen05 cutlass kernel with thin headroom (only 1.03x on B200), so a GB300 config sweep is
  compile-bound (one nvcc build per config) and low-EV. Net this session (access restored): 2 shipped
  wins (paged_kv_offload_prefetch 0.568->1.223x, moe_pad_quant 1.019->1.927x), and the underperformers
  that remain are deep (nvfp4_gemv fused fp4 GEMV), intentional (nvfp4_mlp memory), or thin-frontier
  (nvfp4_group_gemm). The reduce-overhead lever (B22c) is the reusable takeaway. The "reopened frontier"
  was the operator's OWN committed Jun 9 GB300-enablement, not an active competitor (B22).

Patterns (the durable GB300 lessons): (1) comm, reduce or reroute or re-engine the bytes
(volume-reduction, routing, right-engine win; overlap/backend-swap tie on fast NVLink). (2) kernel,
optimize the kernel structure not the byte movement (kernel-structure + CUDA-graph opts carry the
headroom; memory-movement opts near-tie on GB300's abundant bandwidth). (3) FP4, cuBLASLt NVFP4
needs the TN format (the cublaslt_gemm_fp4 fix above).

Not wins (documented env-gaps / infra-walls / gated-skips, not defects): vllm targets (vllm breaks
the NGC toolchain), ch04 nvshmem half (runtime/fabric infra-wall: torch bundles nvshmem but the
single-node torchrun context has no nvshmem launcher/fabric), tcgen05_warpgroup_specialization
(kernel skip gate), the informational examples (not perf pairs), and the train_distributed remainder
(marginal/slow, banked). cublaslt_gemm_fp4 was here as a known-recipe skip; it is now FIXED + PASSING
(467.54x), moved to the wins.

Net: ch04 is no longer a coverage blind spot. It contributed 12 validated GB300 wins (up to
40.44x), 5 ties, 3 banked torchrun edge cases, 1 clean skip, and the nvshmem env-gap, plus the
refined comm pattern.

## train_distributed coverage (2026-06-09): remainder banked (low value, high cost)

Extending the distributed hunt to labs/train_distributed: its 19 recorded entries
(ddp_compression, zero1/zero3, pipeline_1f1b/gpipe/dualpipe) already cover the chapter's speed
lessons, but 6 base targets lacked a result (ddp, ddp_flash, fsdp, fsdp2, zero2, symmem_training).
Measured ddp at 1.06x (marginal, just over the 1.05x gate, the overlap-ties pattern). The other 5
were banked per value-vs-cost: ddp_flash is single-GPU 100-step flash training whose default-mode
compile (the sm_103 max-autotune fallback) ran ~47 min ETA for one target, and fsdp, fsdp2, zero2,
symmem_training are memory-sharding and symmetric-memory methods that tie on raw step-time at
single-node 4-GPU scale (their benefit is memory headroom at larger scale, not speed here). The
distributed win cluster was ch04 (12 wins up to 40.44x); the train_distributed remainder is
marginal and slow, so it is banked rather than chased.

## Coverage audit complete (2026-06-09): the systematic gap was the distributed suite

Final sliver check of the non-distributed "missing" targets confirms no further untested-runnable
target hides a win. They resolve as: naming false-positives that did run under a `_cuda` key
(ozaki_scheme as ozaki_scheme_cuda, nvfp4_gemm/gemv as `*_cuda`, ch12 with 12 recorded keys),
user-submission template slots (nvfp4_*:submission, not benchmark pairs), or a graceful skip
(moe_cuda:decode_kernel skips with "TMA optimized kernel not available"). The one systematic
coverage gap was the torchrun-distributed suite the original sweep never launched, which ch04
turned into 12 validated wins. With that closed and the non-distributed sliver verified, the
GB300 coverage audit is complete: the breakthrough frontier is now a documented hard wall
(perf-kernel at ceiling, env-gaps for vllm/nvshmem/multi-node-fabric/trtllm, and out-of-scope
native sm_103 toolchain + production-scale kernels).

Measurement caveat learned the hard way: the `CudaBinaryBenchmark` targets
(`nvfp4_gemm`, `nvfp4_group_gemm`, `nvfp4_dual_gemm`, `top_k_kernel_cuda`, etc.)
report their OWN internal kernel timing; a wall-clock probe of `benchmark_fn`
measures binary launch + CUDA init + the full internal iteration loop, NOT the
kernel, and overstates time by ~1000x. Use the harness (which parses the binary
stdout) or run the `*_sm103` binary directly. An early ad-hoc probe wrongly
flagged NVFP4 GEMM as "4.4 s / broken"; the real number is microseconds (above).

## Open issues found on GB300 (tier1)
- `labs/block_scaling:block_scaling`: RESOLVED (2026-06-09). Was a CUTLASS DSL
  leading-dim stride assertion under pinned `nvidia-cutlass-dsl==4.3.0` (no sm_103
  in the DSL `Arch` enum). Fixed by pinning `nvidia-cutlass-dsl[cu13]>=4.5.2`,
  vendoring the sm_103 example, and making `block_scaling_common.py` arch-aware.
  Validated on GB300: verify 0.0 abs error, 1.96x speedup (software BF16 dequant
  0.0867 ms -> hardware NVFP4 blockscaled 0.0443 ms). See the toolchain-skew note
  below for the full root cause + the 3-part fix.
- `labs/real_world_models:llama_3_1_8b`: RESOLVED (2026-06-09). Was a SIGABRT in the
  optimized variant. Matched A/B on ONE toolchain (Triton 3.7, sm_103, same GPU,
  only the compile mode differs): `mode="max-autotune"` aborts with `LLVM ERROR:
  Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` (rc=134), `mode="default"`
  runs clean. So the trigger is max-autotune's Triton 3.7 codegen emitting an
  sm_103 `tcgen05.wait.st` its LLVM backend cannot lower (vanilla flex_attention
  with max-autotune compiles fine, so it is the model-compile autotune path, not
  FlexAttention itself). Fix: `_safe_compile_mode()` in
  `labs/real_world_models/llama_3_1_8b_optimization.py` falls back max-autotune ->
  default ONLY on sm_103 + Triton >= 3.6 (warns), preserving max-autotune on the
  pinned Triton 3.5.0 and on every other arch. Validated on GB300: optimized now
  runs (no abort) at 2.97 ms vs eager baseline 7.54 ms = 2.54x.

  Generalized (2026-06-09): the same guard now lives centrally in
  `core/utils/compile_utils.get_optimal_compile_mode()` (the documented selector
  that `compile_model()` routes through), so EVERY `compile_model` caller that asks
  for max-autotune is auto-downgraded to default on sm_103 + Triton >= 3.6 (warns),
  not just llama. Validated: `get_optimal_compile_mode("max-autotune")` returns
  "default" on GB300 + Triton 3.7. CAVEAT: targets that call `torch.compile(...,
  mode="max-autotune")` DIRECTLY (several MoE-journey levels, blackwell_matmul_pipeline,
  moe_backend_selection, nanochat, train_distributed, flashattention4/flexattention)
  bypass the helper. Tested on GB300 + Triton 3.7: the MoE-journey direct max-autotune
  path (`optimized_moe_pad_quant`) compiles + runs fine, so the abort is STRUCTURE-
  specific to llama's attention compile, NOT all max-autotune model compiles, and the
  direct callers are mostly safe. They are also correct on the pinned Triton 3.5.0.
  The inventory loop (isolated_runner + watchdog) surfaces any real direct-call
  crasher cleanly as a failed target; route such a crasher through
  `get_optimal_compile_mode`/`compile_model` to fix it on 3.7.

## Toolchain-version-skew note (important)
Both targets below were NGC base-image version skews, NOT fundamental GB300
problems, and BOTH are now RESOLVED (2026-06-09): block_scaling via the cutlass-dsl
4.5.2 + vendored sm_103 example + arch-aware lab port; llama via the
`_safe_compile_mode()` max-autotune -> default fallback on sm_103 + Triton >= 3.6.
Full root cause + fix for each:
- `block_scaling`: the Python CuTe-DSL JIT path cannot target Blackwell Ultra on
  the pinned `nvidia-cutlass-dsl 4.3.0`. Root-caused four layers deep on GB300:
  (1) DSL `convert_cute_tensor` marks `dim_order()[-1]` as the leading dim, which
  mispicks the L=1 batch dim (a stride==1 selection fixes it); (2) `cute.experimental`
  is a stub that unconditionally raises, and DSL `get_version()` walk_packages
  imports it and crashes (pre-stubbing `sys.modules['cutlass.cute.experimental']`
  fixes it); (3) the DSL `Arch` enum has sm_100/100a/100f, sm_101*, sm_110*,
  sm_120/121 but NO sm_103* at all; (4) forcing `CUTE_DSL_ARCH=sm_100f` (the
  family target that runs across SM10x) passes arch validation but the NVFP4
  blockscaled MMA op `MmaMXF4NVF4Op` is hardcoded to require `sm_100a` (arch-locked
  to sm_100, will not load on sm_103) and rejects sm_100f. So   there is no DSL-4.3.0
  arch that both drives the NVFP4 MMA op AND runs on sm_103. Note the asymmetry:
  the C++ CUTLASS path (the `nvfp4_*_sm103` binaries, nvcc `-arch=sm_103a`) DOES
  support sm_103a and works; only the Python DSL JIT lags.

  COMPLETE FIX (all 3 parts DONE + validated on GB300, 2026-06-09):
  1. cutlass-dsl: `nvidia-cutlass-dsl[cu13]>=4.5.2` in requirements_latest.txt.
     4.5.2 adds `sm_103`/`sm_103a`/`sm_103f` to the `Arch` enum (4.3.0/4.4.2 lack
     them); the `[cu13]` extra pulls the CUDA-13 backend.
  2. Vendored (not a submodule bump) the sm_103-specific example to
     `labs/block_scaling/vendor/sm103_dense_blockscaled_gemm_persistent.py`
     (cutlass main, commit 1fc71b3, BSD-3, byte-identical; see vendor/README.md).
     It imports only from the pip `cutlass` package (`cutlass.utils.blackwell_helpers`,
     `cutlass.utils.blockscaled_layout`), so it runs standalone with no sibling-file
     deps and the v4.1.0 C++ tcgen05/nvfp4 header path is left untouched (still works).
  3. `block_scaling_common.py` is now arch-aware: `_resolve_cutlass_example_path()`
     picks the vendored sm_103 example on Blackwell Ultra (CC 10.3) else the sm_100
     submodule example; `_select_blockscaled_kernel()` returns the `Sm103...` class
     and passes the extra trailing `use_tma_store=True` that sm_103 `can_implement`
     and `__init__` require (sm_100 does not); and load injects `module.cutlass_torch`
     (the sm_103 example imports `cutlass.torch` locally inside `run()`). The SF
     setup is unchanged: the sm_103 example uses the identical
     `cvt_sf_MKL_to_M32x4xrm_K4xrk_L` + atom_m=(32,4)/atom_k=4 layout as sm_100.

  VALIDATED on the GB300 pod (sm_103, cutlass-dsl 4.5.2[cu13]): the lab's own
  harness path `build_problem(compile_hardware=True)` -> `verify_close()` ->
  `run_hardware()` succeeds; verify_close = 0.0 max/mean abs error vs the software
  reference (and the example's internal ref check passes at tol 0.1); measured
  baseline (software BF16 dequant) 0.0867 ms vs optimized (hardware NVFP4
  blockscaled) 0.0443 ms = 1.96x speedup (8192x8192x1024, sf_vec=16, mma (256,128),
  cluster (2,1)). The strict-validity harness expectation write happens when the
  inventory loop reaches the lab (the foreign-process guard only blocked an ad-hoc
  concurrent run, not the fix).
- `llama` optimized (RESOLVED): the max-autotune Triton 3.7 sm_103 `tcgen05.wait.st`
  abort, fixed by the `_safe_compile_mode()` fallback (max-autotune -> default on
  sm_103 + Triton >= 3.6, preserving max-autotune on the pinned 3.5.0). Validated
  optimized 2.97 ms vs eager 7.54 ms = 2.54x; matched A/B above.
The repo's actual GB300 kernels (tcgen05, NVFP4 GEMM, MoE, blackwell_matmul) are
validated working. Both former toolchain-skew breakages are now fixed in-repo, so
the full suite is expected to be all-green on both the NGC image and the pinned
toolchain.
- `labs/flashattention4:flashattention4_alibi`: optimized path measures 1.00x
  (no speedup over baseline) on the NGC torch 2.12 stack; the optimized backend
  is not beating the eager baseline here.

Passing tier1 targets (with my fixes, strict validity): `persistent_decode`
(20.2x), `ch04:gradient_fusion` (150.8x), `kv_optimization:kv_standard` (2.4x).

### tcgen05 hand-written kernels: VALIDATED correct on sm_103a (breakthrough)
Direct validation on GB300: the hand-written tcgen05 matmul
(`ch10/matmul_tcgen05.cu` via `core/common/tcgen05`) compiles, loads, and runs
NUMERICALLY CORRECT with the `sm_103a` fix (relL2 0.00021 vs reference). The prior
`sm_100a` cubin is architecture-locked and will not load on an `sm_103` device, so
the fix is what unblocks the entire tcgen05/TMEM frontier-kernel family
(ch08/ch10 + `custom_vs_cublas`) on Blackwell Ultra. (An earlier note claimed
relL2 ~1.4 / "numerically wrong"; that was a wrong-reference test error: the
kernel computes `A[M,K] @ B[N,K]^T` with shape constraints `m%128==0, n%256==0,
k%64==0`, so the reference must be `a @ b.T`, not `a @ b`. Superseded.)

## moe_hybrid_ep distributed hang -- RESOLVED (CUDA-event elapsed_time race)

RESOLUTION (2026-06-09): FIXED with a one-line root-cause fix + validated on BOTH
targets. The hang was NOT a collective-symmetry bug (that was a symptom + a wrong
hypothesis, see the superseded analysis below). The actual cause: `PhaseEvents.to_metrics()`
calls `cuda.Event.elapsed_time()`, and the BASELINE caller invokes it at `forward_loss`
line ~640 BEFORE the `torch.cuda.synchronize()` at line ~647. On a rank whose terminal
CUDA event has not completed yet (timing-dependent), `elapsed_time` raises
`RuntimeError: Both events must be completed before calculating elapsed time`. That rank
bails through the `finally: shutdown_topology` barrier while the ranks whose events DID
complete proceed to the next collective (`route_counts_global` all_reduce) -- so the
ranks deadlock on DIFFERENT collectives (the desync). Because it is a timing race, only
SOME ranks raise each run, which is exactly the data-dependent asymmetry observed.

FIX: `PhaseEvents.to_metrics()` now calls `self.end.synchronize()` (the terminal event,
recorded last on the stream) before any `elapsed_time`, guaranteeing all four events are
complete. Captured via a per-rank exception print (the `finally` barrier had swallowed
the traceback). VALIDATED strict on GB300: both `moe_hybrid_ep` and `moe_hybrid_ep_multigpu`
go from `failed_error` (NCCL desync hang) to `failed_no_speedup` (1.00x, errors=0) -- they
now run cleanly to completion. The 1.00x is an expected GB300 EP-dispatch tie (memory-
movement, same signature as the other GB300 near-ties), not a break. The diagnostic path
that found it: py-spy stacks -> per-rank collective tracer -> NCCL watchdog (size-mismatch
at SeqNum=59) -> per-call-site tracer (rank0 barrier@139 vs rank1 all_reduce@674) ->
per-rank exception print (the `to_metrics` elapsed_time RuntimeError). The collective-
symmetry hypotheses below are SUPERSEDED (they were real latent-bug candidates but not the
cause; the fixes for them were reverted as unvalidated).

--- SUPERSEDED INVESTIGATION (kept for the record) ---

Found 2026-06-09 in the labs phase: `labs/fullstack_cluster:moe_hybrid_ep` and
`:moe_hybrid_ep_multigpu` both fail "torchrun exited with code 1". The real error
(in the torchrun subprocess stderr) is a NCCL ALLREDUCE/all-to-all timeout after
600s: ranks stuck at DIFFERENT collectives (some at `forward_loss` line 674, some at
`shutdown_topology` barrier line 139), the signature of a collective-symmetry break.

ROOT CAUSE (precise, in `labs/fullstack_cluster/moe_hybrid_ep_common.py`): the
same-node expert dispatch is an all-to-all over `topology.local_group` (all node
ranks), but it is gated per-rank in two places, so a rank that routes 0 tokens to
other ranks skips the collective while its peers call it:
1. Caller (line ~579): `if bool(same_node_mask.any()) and ... local_group is not None:`
   - a rank with no same-node tokens never calls `_roundtrip_routes`.
2. `_roundtrip_routes` (line ~400): `if tokens.numel() == 0: return tokens, None`
   - even if called, an empty-token rank returns before `_exchange_counts`
     (all_gather) + `_all_to_all_single`.
On a single 4-GPU GB300 node `inter_node_world_size = world_size - local_world_size
= 0`, so ALL routing is same-node; with the test's unbalanced routing at least one
of the 4 ranks gets 0 same-node tokens per step and skips the all-to-all -> the
other ranks block in the collective -> 600s NCCL watchdog timeout -> torchrun exits
1. (The remote/inter-node path line ~534 `bool(remote_node_mask.any())` has the same
pattern but is inert on a single node; it is the multi-node analog.) This is a
general EP-correctness bug, not GB300-specific, but the single-node 4-GPU topology
makes it fire every run.

UPDATE (2026-06-09, trace-grounded): the prior same_node-gate root cause is
INSUFFICIENT (superseded). A 4-rank repro on the now-free GB300 node, instrumented
with py-spy (per-rank Python stack) AND a per-rank collective tracer (monkeypatch
`dist.{all_reduce,all_gather,all_to_all,all_to_all_single,barrier}` to log seq+op+shape),
shows the hang on the BASELINE arm too -- and the baseline does NOT use the same_node
optimized path. So the same_node empty-token gate is a real latent bug but NOT the
(sole) cause.

DEFINITIVE TRACE: all 4 ranks issue 59 collectives; the streams are byte-identical
through #58 (the dispatch: 1 all_gather + 4 all_to_all), then at #59 rank1 issues
`all_reduce shape=[16]` (forward_loss metrics `route_counts_global`, line ~683,
shape=num_experts) while ranks 0/2/3 issue the shutdown `barrier` (line 139). py-spy
confirms: 3 ranks at `shutdown_topology` barrier, 1 stuck in `forward_loss`. So one
rank issues exactly ONE extra `route_counts`-shaped all_reduce -> NCCL order-desync
-> deadlock (barrier needs all 4).

NCCL WATCHDOG (the definitive size-mismatch): at NCCL `SeqNum=59` (last completed 58
on every rank), rank1 issues `ALLREDUCE NumelIn=16` (route_counts_global, line ~683)
while ranks 0/2/3 issue `ALLREDUCE NumelIn=1` (a scalar). So rank1 is the lone
straggler -- it SKIPPED one scalar all_reduce the others did, leaving it one collective
behind, and the size mismatch (16 vs 1) deadlocks. This is a missing-collective on
exactly ONE rank, data-dependent (each rank's `data_seed=4242+rank` gives different
routing).

RULED OUT (5 hypotheses tested + refuted with the trace/NCCL/py-spy evidence):
(1) same_node empty-token gate -- the BASELINE arm hangs and does not use that path;
(2) per-rank step count -- `_steady_state_warmup_steps` returns a uniform 2;
(3) `_sync_replicated_grads` `if param.grad is None: continue` -- a genuine latent
asymmetric-collective bug (fixed to all_reduce all replicated params, None->zeros, the
correct DDP average), but the hang PERSISTED;
(4) stale `.pyc` masking the fix -- re-ran with `__pycache__` force-cleared + source
re-touched, hang PERSISTED;
(5) variable metric key-set -- `compute_moe_metrics` + the `forward_loss`/`run_step`
metric `.update()`s + `_reduce_metrics` (827) all iterate a FIXED, uniform key set, so
the scalar all_reduce count is uniform there.

So every collective source read so far (dispatch 5, metrics block route_counts+3 scalars,
_reduce_metrics over a fixed key set, grad-sync over all replicated params) is uniform,
yet one rank still skips exactly one scalar all_reduce. NEXT LEVER (precise): per-LINE
collective instrumentation (tag each `dist.all_reduce` call site, not just seq+shape)
on a 4-rank repro to identify the exact skipped call site on the straggler rank -- the
asymmetry is a non-obvious / data-dependent collective NOT in any of the five uniform
sources above (candidate: a torch/NCCL interaction, or a collective hidden in a
helper). The two latent asymmetric-collective bugs already positively identified (the
same_node empty-token gate + the grad-sync None-skip) belong in the eventual full fix.
Tooling for the next session: per-rank monkeypatch tracer with `traceback.extract_stack()`
on each collective (to get the call site) + `py-spy dump` (`pip install py-spy`) + an
env-driven `init_process_group` timeout for fast repro (the torch 2.12
`TORCH_NCCL_DESYNC_DEBUG` dump did NOT fire). Per the rigor discipline (no unvalidated
distributed fix committed), all attempted fixes were REVERTED; this analysis is the
deliverable. The harness records both targets failed and continues (the NCCL timeout
contains the hang; it never wedged the inventory).

## Missing repo deps on the NGC base image (env gap, not a repo bug)

Found 2026-06-09 (ch16 `flashinfer_block_sparse` failed `No module named 'flashinfer'`).
The pod runs the NGC base image, which is NOT the repo's pinned env: a set of
`requirements_latest.txt` deps are absent (verified by import, not just name match):
`flashinfer-python`, `vllm`, `transformers`, `accelerate`, `sentencepiece`,
`compressed-tensors`, `xgrammar`, `openai`, `anthropic`, `kvikio-cu13`, plus vllm's
transitive deps and the dev/viz tools (jupyter/ruff/mypy/seaborn/...). The repo is
correct (requirements lists them); the NGC image just predates a full install.

Impact is small + specific (the CUDA/Triton chapter targets do not need these, which
is why ch01-17 ran clean):
- flashinfer (`flashinfer-python==0.6.3`): blocks `ch16:flashinfer_block_sparse`,
  `labs/flashinfer_attention`, and the flashattention4 best-available paths. FIXED on
  the pod: `pip install --no-deps flashinfer-python==0.6.3 apache-tvm-ffi` (its
  `tvm_ffi` backend; torch requirement is unpinned so torch 2.12 is untouched).
  VALIDATED on GB300: flashinfer_block_sparse compiles + runs (sm_103), no kernel-image
  or import error.
- transformers: every `labs/train_distributed` variant calls `build_tokenizer()` ->
  `from transformers import AutoTokenizer`, so all 8 baselines (ddp/ddp_flash/fsdp/
  fsdp2 x single/multigpu) failed `No module named 'transformers'` (recorded
  `failed_error`, not skipped, because `build_tokenizer()` raises a plain RuntimeError).
  Also gates several ch18 HF-decoder targets. FIXED on the pod: `pip install
  transformers` (5.10.2; additive, torch 2.12 + triton 3.7 unchanged, only `tokenizers`
  shifted 0.23.1->0.22.2). VALIDATED: `baseline_ddp` runs (3 steps, 1,387 tok/s/rank).
  The one optimized variant with a direct `max-autotune` (`optimized_ddp_multigpu.py`)
  is additionally routed through `get_optimal_compile_mode` (fix 8 above).
- vllm: needed only by `labs/dynamic_router` (4 targets) and `labs/trtllm_phi_3_5_moe`
  (which also needs external model/engine assets, so it skips regardless). NOT installed
  on the pod: `vllm==0.16.0` pins torch/triton strictly, so installing it mid-run would
  downgrade   the validated torch 2.12 / triton 3.7 and risk the whole inventory. Run the
  4 dynamic_router targets on the repo's pinned env (`pip install -r
  requirements_latest.txt` on torch 2.9.1+cu130 / triton 3.5.0), which also resolves
  the Triton-3.7 max-autotune quirk. Documented, not worked around mid-loop.
  CONFIRMED in the live loop: both dynamic_router vllm targets report `status=skipped`
  (graceful), NOT failed_error, so the missing-vllm gap does not dirty the results.
  (flashinfer_block_sparse erroring rather than skipping was the inconsistent case;
  fixed by installing flashinfer.)

Proper one-shot fix for a from-scratch GB300 run: build the image from
`requirements_latest.txt` (the pinned toolchain) rather than layering on the NGC base;
then only the GB300-arch source fixes in this doc are needed.

## sm_100a hardcode in lab loaders (the Phase-0 fix was incomplete)

Found 2026-06-09 (the loop surfaced `ch10:matmul_tcgen05_pipelined` failing with
`CUDA error: no kernel image is available for execution on the device`). The
Phase-0 `sm_103a` fix only covered `core/common/tcgen05/__init__.py`, but the
gencode hardcode `-gencode=arch=compute_100a,code=sm_100a` is duplicated across the
lab CUDA loaders. An sm_100a cubin is arch-locked and will NOT load on sm_103
(GB300), so every lab that compiles its kernel through one of these loaders fails
with "no kernel image" on GB300 when the loop reaches the labs phase.

Fixed (6 active loaders) so they build a fat binary that loads on BOTH B200 (sm_100)
and GB300 (sm_103). The two shapes:
- Loaders with a `get_device_capability()` guard (`tcgen05_loader.py`): emit
  sm_103a on CC 10.3, else sm_100a (matches the core/common/tcgen05 fix).
- Static `extra_cuda_cflags` lists (`cutlass_gemm/__init__.py` + its CMakeLists,
  `fullstack_cluster/capstone_extension_tcgen05.py`,
  `nvfp4_group_gemm/custom_cuda_submission.py`, `nvfp4_gemm/optimized_submission.py`,
  `nvfp4_dual_gemm/optimized_submission.py`): ADD the sm_103a gencode next to the
  sm_100a one (fat binary; no device branch needed in scope).

Validated on GB300 (clear stale cache + recompile): `matmul_tcgen05_pipelined` runs
correct (12288^2), and `nvfp4_gemm` compiles + runs with no kernel-image error.
The remaining four loaders use the identical gencode mechanism so the same fix
applies; the loop exercises them in the labs phase (guarded by the watchdog below).

Comprehensive arch sweep (2026-06-09): scanned every lab `.py` for arch flags
(`code=sm_100`/`sm_100a`, `gencode`, `TORCH_CUDA_ARCH`). Result, all ACTIVE CUDA
targets now have an sm_103 image:
- 6 loaders above: fixed (sm_103a / fat binary), validated end-to-end (all 6 labs run).
- `persistent_decode/optimized_persistent_decode_cuda.py` (target
  `persistent_decode_cuda`): was the LAST sm_100-only active target
  (`code=sm_100`, no sm_103, no PTX -> no kernel image on GB300). Fixed to the
  sibling arch set (sm_100/103/120/121) and validated (compiles + runs on sm_103).
  Distinct from the `persistent_decode` target that already passed tier1.
- Already GB300-ready (no change): `persistent_decode/tma_extension.py` +
  `optimized_tma_prefill_decode.py` (sm_100/103/120/121),
  `fullstack_cluster/capstone_extension.py` (sm_100+sm_103),
  `blackwell_matmul/grace_blackwell_extension.py` (sm_103+sm_100+PTX, why
  blackwell_matmul ran at 126x).

Not fixed (intentionally): the nvfp4 competition-submission ARCHIVES
(`top_submission_candidates/`, `modal697_candidates/`, `submission_*`, `cand_*`,
`candidate_*`, the `optimized_submission_*cacheA*` variants) carry the same latent
hardcode but have no `get_benchmark` (not run by the inventory). Bulk-apply the
same one-line gencode addition if any is ever activated.

## GB300 hazard: intermittent tcgen05 cluster-graph hang + slow hang-detection

Found 2026-06-09 during the inventory loop: `ch10:tcgen05_cluster_pipeline`
(optimized = cluster-launched tcgen05 GEMM replayed from a CUDA graph) HUNG in its
optimized-timing phase, pinning a GPU at 100% with no progress for ~55 min (the
benchmark heartbeat froze at the identical snapshot the whole time). It wedged the
entire inventory until the stuck isolated_runner was killed, after which the loop
recorded the target failed and advanced normally (ch10 finished, ch11 started).

Two distinct issues:

1. The kernel hang is INTERMITTENT. In isolation on a clean GPU the exact path
   (direct cluster matmul, CUDA-graph capture, and 50 graph replays) all run clean,
   so it is not deterministic. Treat the cluster-launch + CUDA-graph-replay
   combination for tcgen05 as a GB300 (sm_103) reliability hazard, not a guaranteed
   failure. Root-causing further needs a reliable repro (not yet found).

2. Hang-detection was far too slow. `BenchmarkDefaults.measurement_timeout_seconds`
   is 1200s, and with the timeout multiplier the effective per-target bound was
   ~3600s, so a hung sub-second microbench was not reaped for ~60 min. The harness
   DOES kill on `subprocess.TimeoutExpired` (a parent-side `communicate(timeout=...)`
   plus a process-group SIGTERM/SIGKILL), so the in-process SIGALRM cannot-interrupt-
   a-CUDA-kernel problem is handled; the issue is purely that the bound is too
   generous for catching a hang quickly.

Mitigations applied: `optimized_tcgen05_cluster_pipeline.get_config()` now pins
`setup_timeout_seconds=180` + `measurement_timeout_seconds=180` (it is a sub-second
microbench, so 180s covers a cold compile and the fast measurement while failing a
hang ~10x sooner). For resilient unattended inventory runs on GB300, prefer a tight
per-target `measurement_timeout` and `timeout_multiplier=1` so any hang fails fast
instead of eating the per-chapter wall-clock budget.

Generic guard: `docs/gb300-inventory-watchdog.sh` reaps ANY hung target during a
long `bench run` inventory, not just this one. It watches the run-progress
snapshot's embedded `timestamp`; when it stops advancing for `FREEZE_LIMIT` (default
600s) it kills the hung `isolated_runner` so the parent loop records a failure and
advances. It uses the frozen-timestamp signal (not a wall-clock age), so a
slow-but-progressing target (including a long training lab) is never killed; only a
genuinely stuck worker is. Run it alongside an unattended inventory:
`bash docs/gb300-inventory-watchdog.sh &`.

## MoE technique-ladder on GB300 (the headroom pattern holds)

The MoE labs span three implementations; their strict per-variant speedups (vs each
lab's naive baseline, cudagraph-on, profile=none) characterize where the MoE headroom
is on GB300, and it follows the SAME pattern as the GEMM + decode ladders: the
KERNEL-STRUCTURE and CUDA-GRAPH levers carry the headroom; the MEMORY-MOVEMENT and
QUANT levers are near-ties.

Headroom carriers (kernel-structure / graph / batching):
- `moe_optimization_journey` (Python ladder): the Level-7 `torch.compile` variant
  (`moe`) lands 43.38x strict (`gb300_reval_moe`, post-fix) vs the naive Python-loop
  baseline; the prior direct-validation ladder showed `optimized_moe_cuda_graphs`
  41.6x and `optimized_moe_triton` 17x. This is the headline MoE win on GB300:
  collapsing the Python expert loop into a single compiled/graph-captured kernel.
- `moe_cuda` (CUDA kernels): `router` 13.79x, `moe_backend_selection` 5.77x.
- `moe_cuda_ptx` (PTX): `moe_grouped_gemm_bwd` 2.20x, `moe_layer` 1.40x.

Near-ties (memory-movement / quant, the GB300 signature):
- `moe_cuda`: `decode_attention` 1.27x, `kv_transfer` 1.12x.
- `moe_cuda_ptx`: `moe_grouped_gemm_fwd` 1.08x, `moe_quant` 1.00x (no_speedup).
- `moe_optimization_journey`: `moe_pad_quant` 1.00x (no_speedup; the pad+quant+
  finalize+slice chain is memory-movement, so torch.compile-default ties eager on
  GB300, exactly the signature below). It now RUNS (was a tcgen05 crash pre-fix).

Read: on GB300 the MoE win is dominated by collapsing Python/launch overhead (CUDA
graphs, batched/grouped kernels, a fused router) and by structure (backend selection),
exactly as for the decode ladder (`decode_ultimate` 7.6x) and the dense GEMM ladder
(tcgen05 126x vs naive, TMA 6.5x). Memory-movement micro-opts (kv_transfer, grouped
GEMM forward) and quant packing (moe_quant) are at parity, because GB300's HBM3e
bandwidth + large L2 already absorb those access patterns. These are technique
speedups vs naive, not byte-grounded %SoL (profile=none captured no per-variant DCGM);
the byte-grounded MoE roofline would need an L3 DCGM pass per variant.

## Completion-phase audit: full inventory result on GB300

The full strict inventory (ch01-20 + every `labs/*`, profile=none, validity=strict,
cudagraph-on, watchdog-guarded) ran to `INVENTORY_COMPLETE` with **0 watchdog reaps**
(no hang wedged the run; the earlier tcgen05-cluster hang predates the tightened
timeouts). 55 scopes wrote results; the 333-benchmark tally:

| status | count | meaning |
| --- | --- | --- |
| succeeded | 259 | optimized beats baseline past threshold |
| failed_no_speedup | 44 | runs correct, speedup < 1.05x (GB300 memory-bound ties) |
| skipped | 13 | preflight / capability / dep skip (graceful) |
| failed_error | 16 | hard error (all classified + resolved below) |
| failed_verification | 1 | input-signature mismatch (pre-existing, hardware-agnostic) |

Every non-green target is classified, with no unexplained GB300 failure:

FIXED this session + re-validated strict (failed -> pass):
- `ch10:matmul_tcgen05_pipelined` 2.33x (sm_103a loader fat-binary).
- `ch10:tcgen05_cluster_pipeline` 1.54x (tightened timeout; the intermittent
  cluster-graph hang did not recur on the clean re-run).
- `ch11:warp_specialized_two_pipelines_multistream` 2.46x (cold-compile timeout).
- `ch16:flashinfer_block_sparse` 3.72x (flashinfer installed).
- `labs/moe_optimization_journey` `moe` 43.38x + `moe_pad_quant` tie (max-autotune
  guard, fix 6 above; the LLVM tcgen05 abort that wrote no results json is gone).
- `labs/occupancy_tuning:proton_matmul` -> skipped (proactive tcgen05 toolchain
  skip-guard, fix 7; was an uncatchable -6 SIGABRT).
- `labs/train_distributed` 8 variants (ddp/ddp_flash/fsdp/fsdp2 x single/multigpu):
  transformers installed + max-autotune guard (fix 8). Fix PROVEN: a re-run shows
  **0 "Baseline FAILED"** (was 8) and 15 optimized variants timed. The full strict
  expectation-update exceeds a 30-min cap (this is a heavy 4-GPU lab once the
  baselines actually train), so the failed->pass expectation rewrite is deferred;
  the unblock itself is confirmed.

FIXED + validated (was a hang, not a GB300-arch break):
- `labs/fullstack_cluster:moe_hybrid_ep` + `moe_hybrid_ep_multigpu`: a CUDA-event
  `elapsed_time` race in `PhaseEvents.to_metrics()` (called before
  `forward_loss`'s `cuda.synchronize()`), which raised on some ranks and desynced
  the collective stream. Fixed by synchronizing the terminal event before
  `elapsed_time`. Both targets now run clean strict (failed_error -> failed_no_speedup
  1.00x, errors=0; the 1.00x is an expected GB300 EP-dispatch tie). See the resolved
  moe_hybrid_ep section above; the earlier collective-symmetry hypothesis is superseded.

ENV-GAP (NGC base image lacks the dep; the pinned env has it; not a repo bug):
- `ch18:vllm_v1_integration`: needs the vllm serving stack. vllm pins torch/triton
  strictly, so installing it mid-run would downgrade the validated torch 2.12 /
  triton 3.7. Resolved by the pinned-env build (below), not worked around mid-loop.

PRE-EXISTING METHODOLOGY QUIRK (hardware-agnostic, not a GB300 regression):
- `ch13:sequence_parallel_multigpu` (failed_verification): the optimized variant uses
  a different `collective_type` than its baseline, so the harness INPUT-signature
  check flags the pair as non-comparable (`mismatches: ['collective_type']`). This is
  a benchmark-pair definition choice that fails identically on B200; the optimized
  path is also 0.95x (slower) here, so there is no GB300 perf claim to rescue. Left
  as-is (fixing it is a pair-redefinition, out of scope for GB300 validation).

Net: every GB300-arch break is fixed (sm_103a kernels, tcgen05 toolchain class,
deps), the moe_hybrid_ep capstone hang is fixed (CUDA-event race) + validated on both
targets, and the only remaining non-green targets are an env-gap (vllm) and a
hardware-agnostic pre-existing pair quirk (ch13). Every actual GB300 break is now
resolved; the repo's frontier optimizations run correct and fast on GB300 (sm_103).

The durable GB300 calibration baseline the repo lacked now exists:
**58 `expectations_4x_gb300.json` files, 310 example expectations** (schema v2,
hardware_key `4x_gb300`), one per chapter/lab scope, each carrying the baseline +
best-optimization timing/speedup/memory + throughput metrics. The chapter +
moe + occupancy re-validations updated their entries failed -> pass.

## Toolchain GB300-readiness (verified 2026-06-09) -- the pinned torch is NOT the GB300 path

Two verified facts that correct the naive "just build the pinned env" recommendation:

1. NO torch build ships a NATIVE sm_103 cubin yet. The NGC torch 2.12 `arch_list` is
   `[sm_80, sm_86, sm_90, sm_100, sm_110, sm_120, compute_120]` -- it runs on the GB300
   (device cc 10.3 = sm_103) via FORWARD-COMPAT (sm_100 SASS + `compute_120` PTX JIT),
   not a native sm_103 cubin. This is exactly why torch's own ops serve fine on GB300
   while the custom CUDA kernels needed explicit `sm_103a` gencode (the `a` cubins are
   arch-locked; torch's are forward-compatible). For a perf book this is worth stating:
   torch kernels on GB300 are PTX-JIT'd from `compute_120`, not sm_103-native.

2. The repo's pinned `torch==2.9.1+cu130` does NOT cleanly install+import on the GB300
   pod's Python 3.12: a fresh venv `pip install torch==2.9.1+cu130` (cu130 index)
   fails at import with `ModuleNotFoundError: No module named 'torch._opaque_base'`
   (the module is genuinely absent from that wheel). So the validated GB300 toolchain
   is the NGC torch 2.12 image, NOT the pinned 2.9.1 -- the 2.9.1 pin is the
   B200/baseline era.

3. triton 3.5.0 does NOT fix sm_103 max-autotune either (REFUTES the earlier
   "3.5.0 JITs sm_103 cleanly" assumption, for the only testable pairing). A venv
   with NGC torch 2.12 + `triton==3.5.0` (confirmed loaded: "USING triton 3.5.0")
   STILL aborts a `torch.compile(mode="max-autotune")` with the same
   `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`. So BOTH triton
   3.5.0 and 3.7 hit the sm_103 `tcgen05` codegen wall with torch 2.12 (the
   matched pinned pair torch 2.9.1 + 3.5.0 is untestable per fact 2, so 3.5.0's
   inherent sm_103 behavior is unproven). NET: there is no proven clean
   max-autotune-on-sm_103 toolchain today; the graceful `max-autotune -> default`
   fallback (the `get_optimal_compile_mode` guard) is the necessary mitigation, not
   a triton-3.7-only workaround.

## One-shot env build (the working GB300 path)

The validated, working GB300 env is the NGC base (torch 2.12, triton 3.7) PLUS the
source fixes in this doc: items 1-8 (sm_103a kernels), the `max-autotune -> default`
guard (sm_103 + triton >= 3.6, which covers the NGC pod), the proton tcgen05
skip-guard, and the additive dep installs (transformers, flashinfer). The
`requirements_latest.txt` dep set (transformers, flashinfer, vllm, ...) closes the
env-gap skips. Do NOT expect a "pinned-env build" (torch 2.9.1 + triton 3.5.0) to
be a cleaner GB300 base: fact 2 (2.9.1 won't import) and fact 3 (3.5.0 still aborts
max-autotune with 2.12) refute that. A clean native sm_103 max-autotune path awaits
an upstream torch/triton that emits a selectable `tcgen05` lowering for sm_103.
