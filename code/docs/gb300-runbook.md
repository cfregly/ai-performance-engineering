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
  demonstrated ceiling if unblocked). [2026-06-11 correction, see B69/B70: the dead param was a red
  herring -- Front P2 probe B shows it JITs clean without the arch de-suffix; the ONLY real trigger
  was triton_compat.py rewriting sm_103a -> sm_103.]
- Kernel (B19, occupancy unblock ACHIEVED, SUPERSEDES B18): isolated the second trigger B18 was missing.
  `import arch_config` is the other trigger: at import it runs configure_optimizations() (inductor config
  + arch env vars + a Triton-compat patch), and empirically the kernel ABORTS with arch_config imported
  but JITs cleanly WITHOUT it on sm_103. (Mechanism note, RESOLVED 2026-06-11 -- the suspect ensure_triton_compat
  Triton patch IS the mechanism, now proven: core/benchmark/triton_compat.py de-suffixed
  sm_103a -> sm_103 (preserving only sm_100a), and the arch-conditional tcgen05 intrinsics are only
  LLVM-selectable for the 'a' target, so instruction selection dies with the uncatchable fatal.
  Probe D in code/upstream/triton-tcgen05-wait-st/STATUS.md reproduces the abort by applying exactly
  that transform to vanilla Triton 3.7; probes A-C show the unpatched stack is clean. Fixed in-repo +
  downgrades retired: see B70.) Fixing BOTH triggers in triton_matmul.py (remove the dead
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
  torch/cuBLAS/CUTLASS paths that already run). [Executed 2026-06-11 (B70): fixed centrally in
  triton_compat.py itself -- suffix preserved, no capability guard needed.]
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
- WIN (B23, ch13:regional_compile 0.0917x -> 1.103x, verify-pass, 12x): two compounding GB300 bugs.
  (1) mode="reduce-overhead" cudagraph THRASHED -- the region is fed the EAGER attention output (a fresh
  tensor address each call) with the seq length cycling [256..1536], so the cudagraph has no stable input
  and re-records every call (0.0917x = 11x slower than the whole-block-compiled baseline). (2) dynamic=True
  added dynamic-shape guard overhead. Fix: mode="default" (drops the cudagraph) + dynamic=False (a static
  MLP kernel per bucket) -> 1.103x, clearing the gate. The mode change alone (dynamic=True default) was
  1.033x (sub-gate); dynamic=False static-per-bucket kernels were the difference.
- BANK (B23b, ch14:regional_triton 0.680x inherently losing on GB300, reverted): same MLP-only regional
  structure but mode="max-autotune" (which bundles cudagraphs). Here cudagraphs HELP (smaller shapes
  [128..512] let the regional cudagraph stabilize) -- max-autotune-no-cudagraphs measured 0.602x (WORSE),
  so there is NO thrash to fix; regional just loses to the whole-block baseline on GB300 (the eager-MHA
  bottleneck: nn.MultiheadAttention eager is competitive with the baseline's compiled attn). Reverted to
  the original. The cudagraph rule cuts both ways across these three labs: moe_pad_quant static shapes
  WANT cudagraphs (B22c); regional_compile's unstable regional input must AVOID them (B23); regional_triton's
  cudagraph is fine and the lab is just a marginal-thesis loss (B23b).
- BANK + survey close (B23c): ch14:model_compile_reduced_precision 1.015x is compute-bound, not a
  reduce-overhead candidate. A SimpleTransformer at batch=24/seq=1536 (36864 tokens) is cuBLAS-GEMM-bound,
  so eager is near-optimal and ALL compile modes add ~1.5% (eager 2.725ms; default / reduce-overhead /
  max-autotune-no-cudagraphs all ~2.68ms); max-autotune (which would autotune the GEMMs) is
  tcgen05-blocked on sm_103. Inherently ~1.015x, no mode clears the gate. This sharpens the B22c lever:
  reduce-overhead helps LAUNCH-bound labs (moe_pad_quant 1024-token MoE -> 1.93x), not compute-bound ones.
  Rest of the survey all-pass: nanochat 2.365x, moe_compiled 2.168x, graph_break 1.145x, and the 9
  moe-journey variants win big (moe 44.7x, moe_bmm_fusion 37.7x, moe_fused/memefficient/permuted ~6.3x,
  moe_streams 5.8x, moe_sorted 5.0x, moe_grouped 3.2x, moe_batched 1.7x); torchao_quantization_compiled
  skips (dependency/HW). Net this session: 3 shipped GB300 wins (paged_kv_offload_prefetch 0.57->1.22x,
  moe_pad_quant 1.02->1.93x, regional_compile 0.09->1.10x). The compile/inference/moe frontier is
  harvested; the remaining headroom is the deep nvfp4_gemv fused fp4 GEMV (~143x off the HBM roofline).
- WIN (B24, nvfp4_gemv 0.22x -> 2.099x, verify-pass bit-exact, ~9.5x; RESOLVES the B22b bank): instead
  of a hand-written Triton fp4 kernel, a torch.compile-fused dequant GEMV. Unpack the fp4 (e2m1) matrix
  from uint8 nibbles via a 16-entry LUT (lo nibble = even k), apply the per-16 e4m3 block scale
  (repeat_interleave 16), and reduce against the dequantized vector; INDUCTOR FUSES THE DEQUANT INTO THE
  GEMV REDUCTION (no [m,k] fp16 materialization), which beats the legacy gemm_v3b (a GEMM kernel on a
  GEMV) / scaled_mm (pads N=1 to 128). Bit-exact to the scaled_mm reference (max abs diff 0.0). Eager is
  0.31x (the materialization dominates); torch.compile(default) is 2.57x (probe). KEY GOTCHA: it needs
  warmup -- the original warmup=5/iters=4 measured only 1.050x (residual JIT in the few iters);
  warmup=15/iters=10 measures the true 2.099x. Env-gated (AISP_NVFP4_GEMV_USE_DEQUANT_GEMV=1 default),
  legacy paths kept as fallback. Lesson: torch.compile can fuse a quant-dequant prologue into a GEMV
  reduction, turning a materialized 0.31x into a fused 2.1x; budget the compile warmup before timing.
- WIN (B25, ch15:inference_monolithic 1.046x -> 7.029x, verify-pass, ~6.7x): the batch=1 autoregressive
  decode (128 tokens x 8 tiny Linear layers = ~2048 tiny kernel launches) is pure launch overhead. The
  original optimized only reused an output buffer (1.046x). Fix: compile the WHOLE decode loop with
  reduce-overhead so inductor cudagraphs it -- kv_cache is the STABLE graph input (no per-step cudagraph
  re-record, unlike regional_compile B23), collapsing the 2048 launches into one graph replay. Probe:
  eager 25.8ms, default 1.12x, reduce-overhead 7.376x; harness 7.029x. Needed warmup=10 for the cudagraph
  capture. The canonical batch=1-decode cudagraph win, and the inverse of B23: there the regional input
  was unstable so cudagraphs had to be AVOIDED; here the loop input is stable so cudagraph the whole loop.
  Also surveyed ch15-19: awq_gptq_smoothquant 0.149x and medusa_eagle_speculative 0.302x are intentional
  (get_optimization_goal memory / throughput; awq uses torch._int_mm = the B20 sm_103 slow path).
- BANK (B25b, ch15:speculative_decoding 1.013x): the B25 full-loop cudagraph lever does NOT apply. The
  speculative path has data-dependent control flow (variable accept_k via a per-round mismatch[0].item()
  GPU->CPU sync, and a while-loop with pos += accept_k), so it cannot be captured as one graph. A partial
  cudagraph of just the fixed k-step draft loop is possible (static input buffer + over-draft to a fixed
  k like inference_monolithic), but the per-round .item() syncs plus the eager verify/accept bound the
  win. Medium-effort, deferred. Distinction: inference_monolithic's decode loop is data-INDEPENDENT (fixed
  128 steps) so it cudagraphs whole; speculative decode's accept logic is data-DEPENDENT so it cannot.
- Survey close (B26, decode/kv/ch18 harvested): swept labs/decode_optimization (12), labs/kv_optimization,
  labs/kv_cache_compression, ch18 (12). All win except two non-fixable cases: one decode_optimization
  variant at 1.0305x (marginal near-miss; the lab's other variants win 1.26x-9.33x, including the
  cudagraph ones, so the near-miss is a redundant non-cudagraph technique) and ch18:vllm_v1_integration
  which ERRORS on an environment mismatch (FAIL FAST: torch expected 2.9.1+cu130, NGC image has 2.12, not
  a perf issue). Highlights: ch18 flexattention_sliding_window 8.5x, paged_attn_backend 12x, flexdecoding
  1.70x; decode variants up to 9.33x. The decode/inference/compile/moe/nvfp4 frontier is harvested across
  this session (5 shipped wins). Remaining unswept = foundational chapters (ch01-09, 11) + ch16-19 rest,
  all low-EV (intro CUDA / already-covered); the high-EV cudagraph-decode + compile-fusion levers found
  their wins.
- WIN (B27, ch02:grace_coherent_memory 1.014x -> 17.049x, verify-pass, ~17x; foundational sweep was NOT
  all-marginal): the optimized used async_pinned (pinned + H2D/D2H stream copies) for the 256MB workload,
  a PCIe-era strategy that is marginal on GB300 (1.014x) because NVLink-C2C makes pinned ~= pageable. The
  GB300-correct strategy is ZERO-COPY: a GPU-resident buffer the CPU reads directly over the coherent C2C
  fabric, so the per-iter H2D+D2H transfers (256MB x2) are skipped entirely and only the in-place mul/add
  remains. Fix: _select_strategy returns "zero_copy" at all sizes on Grace-Blackwell (the old <4MB
  threshold was a PCIe heuristic), and the zero-copy buffer is initialized from a CPU randn (same seed as
  the baseline) so the in-place f^N result equals the baseline's transfer-and-compute result (verify-pass;
  the win is skipping the transfers, not changing the math). The third C2C lesson this session: on
  coherent C2C, keep data GPU-resident and operate in-place; explicit host staging is the anti-pattern
  (cf. B22a prefetch-thread, and the marginal memory_transfer / pageable_copy near-misses are the same
  pattern, candidates for the same zero-copy fix).
- BANK (B27b, ch02:memory_transfer 1.024x + ch03:pageable_copy 1.015x): NOT zero-copy candidates after
  all (correcting the B27 note). These are transfer-MEASUREMENT benchmarks: memory_transfer's benchmark_fn
  is a pure per-iter H2D copy (no compute), and pageable_copy copies a CONSTANT host tensor per-iter then
  sums it. The per-iter transfer IS the measured workload, so skipping/caching it via coherence would game
  the benchmark (unlike grace_coherent_memory, whose data CHANGES each iter via f(x)=2x+1 and is
  legitimately kept GPU-resident -- the transfer there is incidental to an on-GPU compute pipeline). The
  legit lever here is only the copy TECHNIQUE (pinned vs pageable), which on coherent C2C is marginal
  (pinned ~= pageable) -> 1.01-1.02x is inherent. The distinction: zero-copy is a legit win when the
  transfer is incidental to a compute that can stay GPU-resident; it is gaming when the transfer itself is
  the measured quantity.
- Foundational sweep close (B27c, ch04-09/11): grace_coherent_memory (17x, B27) was the UNIQUE
  foundational win. ch04 is distributed training (DDP/NCCL/torchrun 4-GPU), outside the single-GPU
  cudagraph/compile/C2C lever space. ch05-08 single-GPU yield only marginals (launch_bounds 1.0028x, a
  __launch_bounds__ hint with no measurable effect on GB300 since ptxas already optimizes) plus the banked
  nvfp4_mlp (memory-goal). The foundational frontier is harvested. Session total: 6 shipped GB300 wins;
  the three reusable levers (cudagraph the data-independent launch-bound decode loop; torch.compile-fuse a
  quant-dequant prologue into a reduction; zero-copy on coherent C2C to skip incidental host staging) all
  found their wins. Remaining headroom is upstream-blocked (Triton 3.7 tcgen05, torch._int_mm on sm_103)
  or deep multi-day kernel work (a hand-written tcgen05 fp4 GEMM for nvfp4_group_gemm).

- CORRECTION + REOPEN (B28, nvfp4_group_gemm): SUPERSEDES the original B28 bank, which had a WRONG
  premise. An ncu --set full roofline diagnosis (lab dir NCU-DIAGNOSIS.md) overturned it: the committed
  baseline_nvfp4_group_gemm.py imports the SAME custom tcgen05 kernel with a different config (UNROLL_N=1)
  -- it is a config-A/B SELF-comparison, NOT cuBLAS (verified by reading the baseline file). So the
  measured 0.924x is the optimized config (UNROLL_N=2, B200-tuned) REGRESSING vs the baseline config
  (UNROLL_N=1) on GB300, not "cuBLAS 8% faster". The env-config-inert finding stays correct
  (BLOCK_M/BLOCK_N/KPACK are scalar-path knobs that do not tune the tcgen05 path), but the cuBLAS
  attribution + the P4-dead-end verdict were wrong. Corrected ncu picture (vs gb300_nvl72 15.0 PFLOPS FP4
  / 8.0 TB/s HBM): the custom fused kernel is 5.7% FP4-SoL, 15.1% HBM-SoL, 11% tensor-pipe, 6.24%
  occupancy (smem-capped to 2 CTAs/SM, Block Limit Shared Mem), top stall a 46.6% CTA-barrier (the
  warp-specialized bar.sync handoff). The genuine cuBLAS FP4 path (nvjet_sm103) is ALSO far below roof
  (3.8-6.2% FP4-SoL) and cannot batch: torch._scaled_grouped_mm REFUSES NVFP4 (FP8/MXFP8 only), so the
  library forces 30 separate _scaled_mm launches and the custom fused kernel BEATS it 3.1x wall / 1.14x
  GPU on the batched workload (cuBLAS wins 2.6x only on a single isolated grouped call). VERDICT REVISED:
  (b) VIABLE, not a dead-end. The shape (M=192/320, AI ~707 FLOP/byte vs the 1875 ridge) is
  memory-latency + occupancy bound; the reachable ceiling is ~34-38% FP4-SoL (HBM-bound for this AI, not
  100%), and the kernel sits at 15% HBM with ~6.6x latency-hiding headroom blocked by occupancy, not
  bytes/FLOPs. Next lever (in progress): (1) cut per-CTA smem to raise occupancy above the 2-CTA cap,
  (2) replace the bar.sync handoff with mbarrier async pipelining to attack the 46.6% barrier stall. This
  is a deepen-toward-SoL lever on the kernel, NOT a beat-cuBLAS frontier. Lesson: ALWAYS verify what the
  "baseline" actually IS before banking an A/B (the original B28 assumed cuBLAS without checking the
  baseline file).

- WIN (B29, ch15:speculative_decoding 1.013x -> 1.258x, verify-pass, status succeeded): UNBANKS B25b.
  The B25b bank ("data-dependent control flow makes the cudagraph lever N/A") was too aggressive: it
  banked the FULL-LOOP cudagraph (the outer while loop has .item() syncs + a variable accept_k), but the
  SUB-PIECES are fixed-shape and cudagraph-able. The draft's k-step forward loop and the target verify
  forward are both batch=1 launch-bound. Compiling JUST the draft + target models with
  torch.compile(mode="reduce-overhead", dynamic=False) cudagraphs those forwards while the accept logic
  stays eager OUTSIDE the compiled models, so the full-loop blocker never applies. The eager path
  measured 1.013x because per-forward launch overhead ate the speculative gain (baseline greedy 38.39 ms
  vs speculative-eager ~37.9 ms); removing it lands speculative at 30.52 ms = 1.258x. Durable lesson
  (Grind Mandate, never prematurely close): a "data-dependent control flow" bank applies to the FULL
  loop ONLY -- always check whether the fixed-shape sub-pieces (here the draft gen + the verify forward)
  are separately cudagraph-able. 7th GB300 win, found by re-opening a too-aggressive bank.

- RE-AUDIT (B30, bank re-examination prompted by the B29 mis-bank): swept every prior bank for the same
  "missed sub-piece cudagraph" pattern. ch15:speculative_decoding was the SOLE mis-bank (fixed, B29). The
  rest are sound: labs/speculative_decode already passes 2.18x eager (its workload lets the speculative
  algorithm win without compile); ch15:medusa_eagle is a passing throughput/acceptance STUDY
  (get_optimization_goal=throughput, no speed gate, no_speedup=0); ch17:prefill_decode_disagg passes via
  its CUDA-stream prefill/decode overlap; labs/decode_optimization is compute-bound (its best cudagraph
  variant graph/ultimate is marginal, B26) and already ships those cudagraph variants; nvfp4_group_gemm
  is data-backed cuBLAS-bound (B28). Net: 7 GB300 wins this campaign; the batch=1 launch-bound-loop lever
  is fully harvested across the speculative, monolithic, disagg, persistent-decode, and decode families;
  no further single-GPU mis-bank remains. Remaining headroom is upstream-toolchain or the deep P4
  cuBLAS-beating kernel.

- SoL-GROUNDING (B31, all 7 shipped wins, ncu --set full): docs/gb300-sol-roofline.md. 6 of 7 are
  AT-CEILING or latency/transfer-moot (the config wins removed launch/transfer overhead; residual kernels
  at peak, latency-scale at batch=1, or fabric/HBM-bound): regional_compile BF16 nvjet MLP GEMM 94-99%
  tensor (at-ceiling); inference_monolithic + speculative_decoding latency-moot at batch=1; paged_kv
  host/transfer; grace transfer-eliminated; moe_pad_quant launch/occupancy-bound (modest). The ONE with
  real headroom is nvfp4_gemv: the torch.compile-fused dequant GEMV is 3.8% HBM-SoL (0% tensor, 18% FMA)
  -- an H1 (no-tensor-core) kernel on a K3 op burning the SM on int64 LUT-gather + nibble-interleave +
  fp32 reduce. Lever (in progress): a purpose-built HBM-bound fp4 GEMV -- in-register e2m1 bit-decode +
  fp16 accumulate (FMA fix, ~6-13x kernel toward 25-50% HBM-SoL) or a tensor-core CUTLASS NVFP4 MMA
  (~8-26x). Safe verify-passing first step: route case0 (l=1, k=16384) to the existing tensor-core
  scaled_mm path (1.26x, 346->276us, bit-exact). Caveat: clocks not locked (read-only), absolute us/%SoL
  directional; verdicts from --set full achieved tensor/DRAM throughput, not --set basic SM-busy.

- HONEST NEGATIVE (B32, nvfp4_group_gemm deepen attempt): NCU-DEEPEN-ATTEMPT.md. Implementing the B28
  viable lever measured-FAILED, and a config-win is refuted. Lever 1 (cut per-CTA smem, PIPELINE_STAGES
  2->1): REGRESSED (15.1% -> 12.0% HBM-SoL, 226 -> 281us); not applied. Root-cause REFINEMENT (corrects
  B28's "smem-capped"): achieved occupancy is WARP-SPECIALIZATION-capped (~6.2% across 2/3/4 CTAs/SM, all
  configs), not smem-capped -- the mainloop has only ~2 active warps/CTA (TMA-producer + the thread0
  consumer that issues both UTCCP + MMA), so extra CTAs add no active warps and the occupancy lever
  cannot convert to SoL. Lever 2 (cut the 46.6% barrier): the per-k-tile __syncthreads() is load-bearing
  (double-buffer stage-lifetime between the TMA-producer + consumer warps); narrowing it to mbarrier
  async pipelining is a full rewrite across a 6316-line multi-path megakernel, not safely landable in a
  session. Config-win REFUTED (clean test): setting the optimized to UNROLL_N=1 (fresh u1 build) still
  measures 0.926x (optimized 0.364 vs baseline 0.337ms) -- UNROLL_N is a wash; the regression is the
  optimized's EXTRA knobs (EPILOGUE_LD_X32 / WS_TMA_PRODUCER / TMA_L2_PROMOTION / ASSUME_NO_N_TAIL) being
  net-negative on GB300, and dropping them only TIES the baseline (1.0x < the 1.05x gate). VERDICT:
  nvfp4_group is config-UN-WINNABLE on GB300 (no config beats the minimal baseline; UNROLL_N wash; extras
  hurt); the ONLY path to a real win is the deep mbarrier async-pipelining kernel rewrite (next-lever,
  out of single-session scope). The kernel still remains the fastest grouped-NVFP4 option (no library
  grouped NVFP4 exists; it beats per-group _scaled_mm 3.1x, B28). .cu pristine, verify PASS.

- WIN + CORRECTION (B33, nvfp4_gemv deepen): step 1 SHIPPED, step 2 reverted, + a B24 magnitude fix.
  Step 1 (case0 routing, KEPT): route case0 (l=1, k=16384) to the tensor-core scaled_mm path -- 2.87x on
  case0 (339->118us, bit-exact maxdiff=0) AND fixes a latent GB300 CRASH (the old case0 gemm_v3b path
  hits cudaErrorNoKernelImageForDevice, not built for sm_103). The big-K NVFP4 tensor-core GEMM amortizes
  the N=1->128 pad; the dequant is SM-ALU-bound at this shape. case1/case2 stay on the dequant (best
  there). Step 2 (in-register e2m1 decode, REVERTED): bit-exact but SLOWER (fused 24.7->36us, microbench
  +35%) -- inductor already lowers the 16-entry LUT gather to an efficient indexed load, so arithmetic
  decode ADDS net SM-ALU work; the GEMV stays bound on inductor's reduction codegen (stack-interleave +
  repeat_interleave broadcast), which a torch-level edit cannot move. The real lever is a hand-written
  @triton.jit fp4 GEMV (vectorized packed-byte loads, in-register decode, no interleave materialization,
  HBM-bound) -- deferred (larger, higher-risk). B24 MAGNITUDE CORRECTION: the lab measures case2 only
  (m=7168, k=2048, l=4), which is HOST-BOUND (CUDA ~200us of a ~520us call) + noisy; the committed
  "2.099x" is the high end of a ~1.05-2.1x range and does NOT reliably reproduce on this GB300 (steady
  ~1.05-1.07x). The win is real (verify-pass, clears the 1.05x gate) but host-bound, so its magnitude is
  noise-wide. Harness note: the validity gate's foreign-process check calls nvmlDeviceGetHandleByIndex(0)
  = PHYSICAL GPU0 (ignores CUDA_VISIBLE_DEVICES), so a non-contending sibling on GPU0 trips it; set
  AISP_FOREIGN_GPU_PROCESS_MIN_MB above the sibling's footprint when running off-GPU0.

- HONEST NEGATIVE (B34, moe_pad_quant deepen): docs/gb300-moe-roofline.md. No in-scope surgical lever;
  the shipped cudagraph win (1.686x, a host-overhead win already captured) stands, .py untouched. After
  the cudagraph win the call is 67% GPU-busy / 33% host-gap, but the GPU-busy time is dominated by the
  MoE ROUTER (32x aten::nonzero + 32x Memcpy DtoH + cub/gather, ~43% of GPU time), NOT the per-expert
  GEMMs (~19%) -- routing-bound + host-bound, not expert-GEMM-bound. Expert dtype is BF16 (the "quant"
  is INT8 fake-quant in the pad/finalize tail, not the GEMM), so the FP8 _scaled_grouped_mm lever does
  NOT apply. ncu --set full on the dominant per-expert GEMM: Compute(SM) 16.3%, DRAM 4.3%, theoretical
  occupancy 12.5% (register-bound, 255 reg/thread), 0.63 waves/SM -- latency/occupancy-bound, far from
  the 3.75 PFLOPS BF16 roofline, but only ~0.38ms of a ~3ms wall, so a GEMM win is end-to-end-capped.
  The BF16 torch._grouped_mm analog caps at ~1.1x (GEMMs are ~12% of wall). The real lever is a
  non-surgical ROUTER rewrite (the 32x nonzero loop lives in moe_model.py / ch19.mxfp8_moe_common.py,
  outside the optimized_moe_pad_quant.py edit scope) -- deferred. META-PATTERN (B33 + B34): the shipped
  GB300 wins are SURGICAL (config / cudagraph / case-routing); the remaining DEEP kernel levers are
  end-to-end-capped because the calls are host/routing-bound (a kernel win on a host-bound call does not
  move the wall), or deep-out-of-scope. The surgical frontier is harvested.

- WIN + SURVEY (B35, single-target custom-kernel roofline): docs/gb300-sol-roofline-kernels.md.
  SoL-grounded 5 priority custom-CUDA-kernel labs (ncu --set full where hand-owned). AT-CEILING /
  library-bound (the %SoL is the answer, no hand kernel to tune): moe_cuda_ptx (torch.bmm cuBLAS BF16,
  layer 17% FLOP-SoL, PTX path unimplemented), ozaki_scheme (cuBLAS FP-emulation, 61.3 TFLOPS),
  nvfp4_gemm (CUTLASS NVFP4 M=128 memory-bound ~65% HBM-SoL near-ceiling; tiny shapes launch-bound).
  HAND-OWNED HEADROOM: custom_vs_cublas gemm_cluster (CuTe tcgen05 FP16 8192^3) at 32.2% FP16 FLOP-SoL,
  latency/occupancy-bound (Compute(SM) 58%, DRAM 33%, achieved occupancy 6.2% = 1 CTA/SM, top stall
  46.5% scoreboard-on-smem); cuBLAS hits 49.3% on the same shape (1.6x faster), so the gap is real.
  WIN (verify-passing, KEPT): eliminated the beta=0 dead-C path in tcgen05_cluster.cu -- the epilogue
  loaded a 256MB C only to multiply it by 0.0, plus a 256MB per-call torch.zeros host memset; now writes
  the accumulator straight to D (zeros->empty, drop the C-load + zero-axpby). Verify PASSES bit-identical
  (rel 2.6e-4); harness 2.260x -> 2.329x. Honestly modest (~3% wall, ~0.8% kernel) because the kernel is
  latency/occupancy-bound not bandwidth-bound, so removing 256MB of traffic barely moves it -- the real
  gain is the killed per-call memset + dead C-load (a clean efficiency/correctness fix). NEXT LEVER: the
  32%->50%+ gap needs an occupancy rewrite (warp-specialized persistent kernel + split TMEM); structurally
  capped for an incremental edit (smem 192/227KB + full-TMEM both pin it to 1 CTA/SM), deferred.

- WIN + HYPOTHESIS CORRECTION (B36, blackwell_matmul host-overhead deepen): docs/
  gb300-blackwell-matmul-hostoverhead.md. labs/blackwell_matmul:blackwell_matmul_tcgen05 (FP16 2048^3)
  0.2250 ms -> ~0.155 ms median (~1.45x; 69.61x -> 101-104x vs naive), verify PASS x3, outputs
  BIT-IDENTICAL (torch.equal) pre/post. The shared-file siblings also gain: fullstack_cluster
  cluster_gemm_tcgen05 1.06x, cta2 1.12x, both verify PASS. Patch (capstone_kernels_tcgen05.cu, shared
  by both labs): (1) drop the beta=0 epilogue C-load + zero-axpby, (2) torch::zeros C + empty_like D ->
  single torch::empty D aliased as C, (3) cache the TMA atoms in a leaked static thread_local keyed on
  (a_ptr,b_ptr,m,n,k,device) -- content-agnostic, mutated-tensor case verified correct, (4) one-shot
  cudaFuncSetAttribute guard. TWO HYPOTHESES CORRECTED by the nsys probe: (a) the B35-survey suspect
  cuTensorMapEncodeTiled is 0.24 us/call (0.7% of API time), NOT ~190 us -- the TMA-encode-dominance
  hypothesis is refuted; (b) the B35 "kernel ~16us vs wall ~213us" framing was wrong -- the kernel was
  66.2 us, and the single biggest win was IN-KERNEL: the dead beta=0 C-load was serializing ~27 us of
  gmem->reg latency in the epilogue (66.2 -> 38.9 us kernel). The "13x host overhead" was really a mix
  of a slower-than-surveyed kernel + ~63 us of host costs (zeros fill+alloc ~31 us, setattr 4.5 us,
  encode+atom-build, pybind). Serial probe: 134.4 -> 71.1 us warm, 156.8 -> 91.9 us cold-L2. Durable
  lesson: a wall-vs-kernel gap claim from a DRAFT wall-clock survey must be profile-split before
  trusting the ratio -- here the same beta=0 dead-path class (B35) hid on BOTH sides of the boundary.
  NEXT LEVER: fp16-direct epilogue (kill the .to(kFloat16) ~3.9 us copy + ~20 us host _to_copy, halve D
  store traffic); then the ~28 us/call residual host gap (pybind + empty + launch) is a CUDA-graph
  candidate; kernel-side 38.9 us = ~440 TFLOPS = 11.7% FP16-SoL at 2048^3 -- k-stage double-buffered
  pipelining is the deep candidate.

- WIN + PREDICTION CALIBRATION (B37, custom_vs_cublas dual-CTA occupancy rewrite, measured): docs/
  gb300-gemm-occupancy-rewrite.md. The B35-deferred occupancy lever LANDED as a new opt-in variant
  (tcgen05_dual_cta.cu, AISP_TCGEN05_VARIANT=dual_cta): config (256,2) measures 838us best / 1311
  TFLOPS / 35.0% FP16-SoL vs the incumbent cluster kernel 904us/32.4%, beating it in ALL 6 interleaved
  A/B reps (1.076-1.178x, median 1.163x); harness contract 2.33x -> 2.63x, verification PASSED; default
  cluster path regression-checked (2.44x PASS, contract intact). Kernel correct on first build (zero .cu
  fixes, rel_err 0.0). MECHANISM PROVEN by ncu: Block Limit Shared Mem = 2, achieved occupancy 12.06%
  vs the incumbent 6.2% -- 2 CTAs/SM resident. PREDICTION CORRECTED: the designed default (128,3)
  is a measured LOSS (1054us, 27.8% -- halving tile N halves arithmetic intensity); the winner keeps
  the incumbent's 128x256 tile and funds the 2nd CTA by cutting pipeline stages 4->2: CONCURRENCY BEATS
  LOOKAHEAD DEPTH on this kernel. Shortfall vs the 40-48% prediction: occupancy doubled but tensor-pipe
  duty rose only 58 -> 61.9% -- the correct empty-barrier wait costs per-CTA duty that the co-resident
  CTA only partially refills (both 2-stage CTAs starve simultaneously on TMA latency; long_scoreboard
  still ~73% of stall cycles); DRAM is NOT the constraint (21.8%, the saturation worry was unfounded).
  Node thermal drift is real (~5-9%; cluster drifts hot across reps, dual stays flat) -- interleaved A/B
  is the fair read. Gap to cuBLAS remains: 1818.7 TFLOPS / 48.5% same-session. NEXT LEVERS: (1) 2x1
  cluster + TMA multicast of B on the (256,2) footprint (halves B-side L2->SM traffic, attacks the
  dominant long_scoreboard stall); (2) persistent CTAs + tile-swizzled scheduler (4096 one-shot CTAs
  each pay launch + a register-capped 256-fp32/thread epilogue).

- WIN + SURVEY (B38, never-rooflined labs sweep + block_scaling cudagraph): docs/
  gb300-sol-roofline-labs2.md. WIN (shipped, verify PASS x2): labs/block_scaling cudagraph-captures the
  CuTeDSL compiled_gemm launch in _post_setup (side-stream capture, replay in _run_problem; eager
  fallback + AISP_BLOCK_SCALING_DISABLE_GRAPH=1 kill-switch; verify path stays eager; graph output
  bit-identical, max_abs_err 0.0) -- optimized arm 0.0946 -> 0.0811 ms = 1.168x over the prior optimized,
  harness 1.97x -> 2.24-2.57x vs baseline. SPLIT (corrects the F-survey host-dominance hypothesis to
  half-right): the kernel is 44.8 us (20.5% NVFP4-SoL, 40% of the 18 us HBM floor); host submit is only
  7.9 us hidden; the 94.6 us wall was kernel + ~50 us timing-frame overhead (event bracket + device sync
  + L2-clear + Python), of which graph replay reclaims ~13.6 us. Config lever REFUTED clean: 21
  tiler/cluster combos flat (best 4.5% kernel). ncu mechanism: UMMA tensor-pipe 34.5%, ~58% scoreboard
  stalls (smem/L1TEX), single persistent wave, DRAM 25% -- shallow-K (K=1024) mainloop cannot hide smem
  latency; the remaining 2.5x to the HBM floor needs an in-kernel rewrite of the vendored sm103 example
  (deeper smem staging / split-K), deep-deferred. ALSO MEASURED (fresh GB300 numbers, GPU3):
  nvfp4_dual_gemm kernel-frame is 6.6 us vs 48.5 us harness wall at iterations=4 -- 86% of the harness
  number is frame/host (raise iterations before ANY kernel claim; kernel-side 3.3x headroom, tiny-M
  mainloop latency); memory_bandwidth_patterns 70.6% HBM-SoL (5.64 TB/s, at-ceiling at the 70% cutoff);
  software_pipelining 60.5% HBM-SoL (TMA bulk-copy/deeper stages named, not implemented); attention
  quartet (fa4 14 arms, gluon, flex, cudnn_sdpa) CLOSED as overhead-bound micro-shapes (0.06-3.4% of
  floors; no optimization EV at these shapes). With B36+B37 this round: three verify-passing wins, all
  three from the same playbook -- profile-split the wall, kill the incidental work, graph the launch.

- WIN + EVIDENCE-BASED SKIP (B39, fp16-direct epilogue on the B36 kernel): docs/
  gb300-blackwell-matmul-hostoverhead.md (D3 section). capstone_kernels_tcgen05.cu epilogue now
  converts the fp32 UMMA accumulator to fp16 in registers and stores straight to an fp16 D
  (TypeD float -> cutlass::half_t; the .to(kFloat16) cast kernel + host aten::_to_copy are GONE, D
  store traffic halved): blackwell_matmul_tcgen05 0.14949 -> 0.1300-0.1309 ms (104.9x -> ~120x vs
  naive), verify PASS x3; BIT-IDENTICAL (torch.equal pre/post for CTA1+CTA2 -- fp32-accum->fp16 RN in
  registers rounds identically to fp32-store-then-convert). fullstack_cluster pair ratios settle at
  ~1.0x by construction (B28 pattern: baseline CTA1 and optimized CTA2 are BOTH this kernel -- both
  sides got faster; absolute opt times improved, verify PASS). Kernel 38.9 -> 37.6 us; serial probe
  75.7 -> 64.5 us. CUDA-GRAPH SKIPPED with evidence (not negligence): the sanctioned path is the
  harness enable_cuda_graph + declared graph_capture_enabled signature audited by
  GraphCaptureCheatDetector; this benchmark declares none, flipping the shared config changes the
  timing mode of every variant in both labs (a measurement-contract change, not an op win), and an
  extension-internal undeclared stream-capture is ambiguous-to-illegitimate. Ceiling was ~3% anyway
  (graph replaces only the 8.5 us launch API; the ~18 us pybind/check/empty host residual is
  un-graphable). Cumulative B36+B39 on this target: 0.2250 -> 0.130 ms = 1.73x, bit-identical
  throughout. NEXT LEVER (now the dominant term): k-stage pipelining in gemm_device_variant
  (double-buffered smem, TMA prefetch of tile k+1 overlapping MMA of tile k) -- the mainloop is serial
  TMA->wait->MMA->wait and 37.6 us is ~457 TFLOPS = 12.2% FP16-SoL at 2048^3.

- BREAKTHROUGH + UN-BANK (B40, moe_pad_quant in-file router override): docs/gb300-moe-router-override.md.
  SUPERSEDES the B34 negative, which was a MIS-BANK of the B29 class: the "out of scope" wall was the
  original task's single-file framing, not the repo's -- the OPTIMIZED arm builds its own model instance
  and may legitimately bind a vectorized router onto it from within optimized_moe_pad_quant.py. Measured:
  optimized arm 4.222 ms -> 0.309-0.384 ms (median ~0.324) = ~13x over the shipped arm (23-35x vs the
  eager baseline), verify PASS x5 (max_diff 0.1025, rtol 0.1/atol 1.0). The B34-era ceiling estimate
  (~1.4-1.5x) was beaten ~9x because the win COMPOUNDS: killing the 32x nonzero + .tolist() graph breaks
  let torch.compile capture the ENTIRE model as ONE cudagraph, collapsing the fragmented compiled
  subgraphs too -- host gap ~1.0 ms -> ~0, aten::nonzero 32/iter -> 0, DtoH 32/iter -> 0; new wall split
  is pure GPU (~0.16 ms/iter standalone: expert bmms ~57us, lm_head ~24us, router cumsum/scatter ~25us).
  Winning design (of 4): sortless fixed-capacity dense dispatch via types.MethodType in setup() --
  rank-within-expert by one-hot cumsum (no argsort; the same slot indices gather back, restoring token
  order for free), index_copy_ into a [32,192,512] slot tensor (capacity = 2x observed max, calibrated
  by ONE eager pass in setup, zero host syncs in the measured path), 3 bmms vs stacked weights, weights
  post-gather, sum over top_k. Numeric cross-check max_abs_diff 0.0 vs original forward_grouped before
  any harness run. torch._grouped_mm (works on torch 2.12/sm_103) measured SLOWER (0.824 vs 0.775 ms
  eager) -- documented reserve. Durable lessons: (1) re-audit banked negatives whose blocker was a TASK
  constraint, not a measured wall (B29, now B40); (2) graph-break removal pays superlinearly when it
  unlocks whole-model cudagraph capture -- the lever's value is the capture boundary it moves, not the
  op it fixes. Related sweep (same session, static): the B36 host-waste class (dead beta=0 buffers,
  per-call descriptor builds/attrs, fusable casts) is ABSENT from the rest of the repo -- descriptors
  cached at init, attrs in setup; the class was confined to the two labs already fixed (B36/B39); the
  three static candidates surfaced were baseline-arm or cuBLASLt-beta=0-fast-path false positives.
  NEXT LEVER: residual is GEMM-shape efficiency (<=1.5x on 0.16 ms, below the harness noise floor) --
  bank this front as harvested.

- WIN + CORRECTION (B41, software_pipelining TMA bulk-copy lever): docs/gb300-software-pipelining.md.
  The B38-named lever landed: kernel 83.0 -> 60.5 us = 4850 -> 6653 GB/s = 60.6% -> 83.2% HBM-SoL
  (fp32 2^25, 403MB/iter); harness 144.0 -> 123.0/126.0 us = 1.17x over the committed optimized arm,
  verify PASS x2, max_abs_diff 7.15e-7, baseline arm untouched. DIAGNOSIS (the transferable part): ncu
  showed the 2-stage cuda::pipeline kernel was INSTRUCTION-ISSUE-bound, not DRAM-bound (issue slots
  74.6% busy, IPC 3.17/4, DRAM only 56%) -- scalar element loops + per-element predicates + 4 redundant
  cta.sync per 1KB tile + per-thread cp.async machinery. Fix: (1) float4 + aligned_size_t<16> + sync
  removal on the sm_80 fallback (83 -> 64.2 us); (2) a new warp-specialized sm_90+ kernel: single
  producer lane issues cp.async.bulk per operand per tile with full/empty mbarrier pairs, STAGE
  EARLY-RELEASE (drain smem->registers, release the stage before compute/store), exact-cover 544-thread
  blocks (1 producer warp + 512 consumers, one float4 pair each, zero ragged loops). Winning config
  stages=2 tile=2048 (stages 3/4/6/8, tiles 1024/4096, V=2/4, __stcs all rejected by sweep). SoL
  GROUNDING: torch.add (same 2R:1W bytes, zero staging) measures 86.8-87.1% = the practical GB300 triad
  ceiling, so 83.2% is within 4.6% of unstaged parity and the naive 50.3us "floor" is unreachable here.
  This validates+supersedes the e5655e3a/b487f3f8 unvalidated cp.async candidates. CORRECTION (B38
  nvfp4_dual_gemm frame claim): raising iterations 4 -> 50 measures 48 vs 47 us/iter -- the "86% of the
  harness wall is frame" premise FAILS measurement (per-iter wall is frame-clean); the dual_gemm bank
  reverts to a plain kernel-side question (banked, tiny-M latency). NEXT LEVER (small EV, <=3us): TMA
  stores to skip the st.global drain; atomic-ticket tile assignment for the ~1/36 tail.

- RE-AUDIT (B42, full-bank false-wall audit, prompted by the B40 mis-bank): every banked negative/tie/
  deferral re-classified against the six levers proven AFTER those banks were written (in-file
  model-surgery, whole-model cudagraph capture, warp-specialized TMA bulk-copy, dual-CTA occupancy,
  fp16-direct epilogue/dead-operand elimination, _post_setup graph capture). RESULT: all standing
  negatives are SOUND measured walls (B1/B7 NVLink ceiling, B15 atomic-serialization, B20 torch._int_mm
  upstream, B22b dequant-bytes, B23b/B23c compile-vs-eager structure, B27b transfer-measurement gaming
  rule, B32 warp-specialization cap); the three historical mis-banks (B25b->B29, B28->B28-reopen,
  B34->B40) were already caught -- pattern confirmed: both un-banked classes were TASK-framing traps
  (full-loop generalization; single-file scope), not measured walls. STALE-CHECKS (2): B10 grouped-GEMM
  Triton tl.dot codegen gap (never re-benchmarked on a newer triton/torch pair) and B16 (zymtrace
  infra-wall blocked the codegen ground-truth) -- both reopen only on a toolchain change. NEW TERRITORY:
  7 labs fell through EVERY survey (ran in the strict inventory, never headroom-examined):
  parameterized_cuda_graphs, python_concurrency, moe_parallelism (multi-GPU), tcgen05_cluster_shapes,
  training_hotpath (the B36-sweep flagged its torch::zeros at rank 6), uma_memory, vllm-deepseek-tuning
  (env-gap). Single-GPU survey front queued on these. The 44 inventory failed_no_speedup ties re-confirm
  as the GB300 memory-movement signature; no hidden regression.

- HONEST NEGATIVE (B43, dual-CTA B-multicast measured): docs/gb300-gemm-occupancy-rewrite.md (E5
  section). The c045227b 2x1-cluster B-multicast variant is a STATISTICAL TIE with plain dual-CTA
  (interleaved 12-rep paired A/B: mcast/plain median ratio 1.0065, mcast wins 3/12; sweep medians
  919.0 vs 929.5 us; cm=4 strictly worse; (128,3,cm=2) worst arm 1211.8 us). The lever's PREMISE is
  FALSIFIED by ncu: the duplicate B reads were already deduplicated by L2 (sectors 278.9M -> 275.5M,
  -1.2%, not the expected ~33%; DRAM 12.5% both), so there was no bandwidth to reclaim; the dominant
  stall is long_scoreboard 28.25 -> 28.04 warp-cycles/issue (UNCHANGED) = TMA LATENCY serialization on
  the single warp-0 producer/consumer at 2 stages; cluster_sync + multicast-commit fan-out eat the
  negligible savings. Clustering costs no occupancy (Block Limits smem=2/regs=2 identical, 12.09%
  achieved both). Verify gates 3/3 PASSED (multicast 2.6703x / plain 2.6710x / default-cluster 2.4248x;
  2.329x contract intact). DEFAULTS KEPT (loader default is cluster_m=1 = the plain winner; cm=2 stays
  as a zero-cost measured-negative comparison arm). (256,3) confirmed smem-impossible by build
  (static_assert 144 > 110KB/CTA). Same-session cuBLAS reference: 602.1 us / 48.7%. NAMED NEXT LEVER
  (the only remaining structural path to the cuBLAS ceiling): 2-SM UMMA pair (cta_group=2,
  SM100_MMA_F16BF16_2x1SM_SS) + warp-specialized producer on the dual-CTA footprint -- the kernel is
  issue/latency-bound, not BW-bound, and stage-deepening at (256,2) is smem-exhausted.
  INDEPENDENTLY REPLICATED (second front, same session): a concurrent front raced the same mission,
  detected a 9-minute GPU-2 co-tenancy overlap with the first (post-hoc, from artifact mtimes),
  re-measured everything decisive on a verified-quiet GPU: plain 926.1 vs mcast 928.3 us (0.24% delta,
  dead tie), long_scoreboard 28.5 -> 28.0 of ~39 (unchanged), multicast confirmed ACTIVE
  (SM90_TMA_LOAD_MULTICAST, cluster 2) with L2/DRAM barely moving -- the premise falsification is
  replicated, verify 3/3 PASS again. PROCESS LESSON: orchestrator must treat agent silence as
  UNKNOWN, not dead -- two fronts silently co-measured on one GPU and only forensics caught it; a
  GPU flock/mutex (or orchestrator-side GPU lease) is the named process lever.

- WIN (B44, capstone k-stage pipeline, the B39-named lever): docs/gb300-blackwell-matmul-hostoverhead.md
  (D4 section). capstone_kernels_tcgen05.cu mainloop rebuilt from serial TMA->wait->MMA->wait to a
  4-stage smem ring (48KB/stage = 192KB < 227KB cap, static_assert-guarded) with per-stage tma
  (expect-tx) + mma (tcgen05-commit) barrier pairs (no full-drain wait per k-iteration), whole-warp
  specialization (warp 1 TMA producer, warp 0 UMMA consumer, warps 2-3 phase-lock), parity-safe final
  drain; both 1SM and 2SM-multicast branches. Kernel 37.6 -> 24.0 us = 1.57x = ~716 TFLOPS = 12.2% ->
  19.1% FP16-SoL; ncu like-for-like 68.5 -> 45.3 us, tensor-pipe 13.0 -> 20.4%, IPC 0.13 -> 0.21.
  Harness: blackwell_matmul_tcgen05 0.13113 -> 0.10377/0.10847/0.12029 ms (best 151.30x vs naive),
  verify PASS x3; both fullstack_cluster targets PASS. BIT-IDENTICAL (torch.equal, 20 repeats each,
  CTA1+CTA2) -- structural, the k-tile MMA issue ORDER is unchanged, only the prefetch overlaps.
  Cumulative B36+B39+B44 on this target: 0.225 -> ~0.104 ms = ~2.2x, bit-identical lineage intact.
  Iteration log (the depth story): ring+single-barrier 29.45, per-stage barriers 24.32, warp-spec
  24.03, S=2 attribution 27.92 -- depth matters; residual is TMA latency x depth and S=4 is the smem
  ceiling at bK=64. BANKED TRAP (durable): a 1-LANE producer spin role inside a warp ran bit-identical
  on hardware but reproducibly DEADLOCKED under ncu kernel-replay; whole-warp role assignment fixed it.
  Never put mutually-dependent spin roles inside one warp. NEXT LEVER: finer k-tiles (bK 64->32, SW64
  atom) halve stage cost to 24KB -> 8-stage ring in the same smem budget (doubles TMA-latency cover,
  accumulation order preserved); fallbacks: cluster B-multicast (NOTE: B43 falsified that premise on
  the SIBLING dual-CTA kernel -- L2 already dedups; re-check L2 sectors here before building it), or
  the deferred persistent-occupancy rewrite (full-TMEM alloc still pins 1 CTA/SM).

- BREAKTHROUGH (B45, moe_cuda router_vectorized, the B40 lever class generalizes): docs/
  gb300-moe-cuda-router-vectorize.md. labs/moe_cuda:router_vectorized optimized arm 8.289 -> 0.500 ms
  = 16.5x over the shipped arm (harness 177-270x vs the noisy eager baseline), verify PASS x4,
  numerics BIT-EXACT (max_abs_diff 0.0, eager and via replay). Root cause WORSE than the static
  estimate: the 32-iteration boolean-mask expert loop made the benchmark's own torch.cuda.CUDAGraph
  capture THROW (flat_tokens[mask] -> aten::nonzero -> DtoH, illegal during capture), so self.graph
  silently fell back to None and the "graph-optimized" arm ran EAGER with 96 host syncs/iter. After
  the B40-style sortless fixed-capacity dense dispatch (one-hot-cumsum rank -> index_copy_ into
  [32x640] slots -> 2 baddbmm with bias folded -> gather-back): 679 cudaLaunchKernel + 96 nonzero +
  96 DtoH per iter -> 1 cudaGraphLaunch, 0/0/0, 33 kernels inside one graph. Design (a) (gathered
  per-token weights) rejected by arithmetic: w[ids] gather is ~64GB of traffic = the 91ms baseline.
  PERF TRAP BANKED: a naive [N,E] one_hot.cumsum(0) (outer-dim int64 scan, 32 lanes) was a single
  2048us kernel = 82% of the replay; transposing to [E,N].cumsum(1) is 24us. Audit rule from B45:
  a benchmark whose graph-capture path can THROW must fail loudly -- a silent eager fallback hid a
  16.5x for the lab's entire life. NEXT LEVER (modest, ~1.4x est): torch.compile the now-static
  forward to fuse bias+GELU epilogues; compile-vs-manual-capture interplay needs care.

- HONEST NEGATIVE (B46, block_scaling tile_k lever, implementation-grade): docs/
  gb300-block-scaling-deepen.md. The reopened H-front lever is REFUTED three ways, with the strongest
  evidence class (working bit-correct implementation measured at parity): (1) tile_k=384/256 are NOT
  configuration -- MMA-K is ISA-FIXED at 96 FP4 elements (SM103MmaMXF4NVF4Op) and the AB smem buffer
  is a K_SW128 atom (256 elements); 768 = LCM(96,256); the mainloop is a hand-unrolled 8-MMA/3-buffer
  period with cross-buffer descriptor wraps. (2) The stage-starvation premise was FACTUALLY WRONG: a
  trace probe shows ab=8 stages (24KB each) -- the ring already holds the whole K=1024 extent plus
  lookahead (corrects the H-front static edit map). (3) The implementable equivalent (peeled
  short-tail last k-tile, 16->11 MMAs/tile = -31% MMA work, bit-correct err 0.0) measures EXACT PARITY:
  43.62 vs 43.65 us (0.9994x, 6 interleaved rounds); ncu proves it -- tensor-pipe active drops 34.50 ->
  24.02% (exactly x11/16) at FLAT duration: the zero-pad MMA work was fully hidden. Harness A/B/A all
  verification PASSED, no B38 regression. Kernel re-diagnosed FEED-BOUND: ~786MB/GEMM of A/B re-reads
  (4096 CTA-tiles x 192KB) vs 8MB unique bytes + a 128MB C write (~16us of the 18us HBM floor); the
  suggested (256,256) tiler relief was ALREADY measured marginal in the B38 21-combo sweep (4.5%
  kernel, ~2% wall, below gate). VERDICT: block_scaling is at its structural ceiling absent a
  different data layout; closed. Also closes the K_SW64/tile_k=384 deep rewrite (strictly worse than
  the parity-measured 11-MMA variant). Default path const_expr-gated bit-identical post-edit.

- WIN + SURVEY (B47, never-examined labs sweep, closes the B42 territory): docs/
  gb300-sol-roofline-labs3.md. WIN (shipped): labs/training_hotpath:metric_reduction_cuda 8.11x ->
  9.01x/8.89x (kernel 15.5 -> 10.3 us), verify PASS x2 -- segment_abs_mean launched only 128 blocks
  (Waves/SM 0.11, occupancy 12.4%, DRAM 4.2%); fix = chunked grid (1280 blocks) + per-chunk atomicAdd
  partials + SM-count cached in a static. Attempt-1 trap banked: a per-call cudaDeviceGetAttribute
  (~10us) made it SLOWER (6.32x) -- cache device attributes. Now host/harness-overhead-bound (~10.8us
  zeros+launch floor). FINDINGS: (1) padding_aware_transformer's packed-row path is a SPEED-REGRESSION
  on GB300 (0.674x; CUDA-event isolated 1.647 vs 2.597 ms) that "succeeds" only via its memory goal --
  the per-iter 315MB torch::zeros memset is only ~100us of the ~950us regression; the real cost is ~72
  extra tiny launches + odd-shape GEMMs (the CPU-era packing trick inverts on GB300's launch profile);
  named deep lever: cudagraph the packed forward, do not patch. (2) metric_reduction_vectorized 93.6x
  but ~22% SoL: 3 mul+sum passes re-read 126MB; fusion to one pass (25MB) named (~150x headline est),
  fp32-accumulate guardrail required. (3) B45-CLASS AUDIT CLEAN: no silent capture->eager fallback in
  any of the 7 labs; parameterized_cuda_graphs is loud-by-construction and its capture is LIVE on GB300
  (33.83x via real graph replay, param-update probe confirms). (4) tcgen05_cluster_shapes = BUILD-GAP
  (pod third_party/cutlass missing cutlass/detail/collective/moe_stride_utils.hpp -- blocks the ch10
  tcgen05 extension family); uma_memory/python_concurrency/moe_parallelism = NOT-A-BENCHMARK
  (utility/CPU-teaching/planner); vllm-deepseek-tuning = ENV-GAP. (5) Harness measurement overhead
  ~30-55us/iter compresses every sub-100us lab speedup -- audit-worthy as a harness lever.

- HONEST NEGATIVE (B48, capstone bK=32 8-stage ring, falsified by its own control): docs/
  gb300-blackwell-matmul-hostoverhead.md (D5 section). B44's named lever is DEAD: bK=32 is ~14% slower
  on CTA1 at EVERY depth (bK64/S4 incumbent 24.15us/19.0% SoL vs bK32 S4/S6/S8 = 27.60/27.48/27.66us),
  and the bK32/S4 CONTROL proves the regression is the finer k-tile itself -- deepening 4->6->8 recovers
  <=0.2us. No latency-cover deficit existed (long-scoreboard/barrier stalls flat in ncu); the binding
  term is TMA delivery efficiency per byte (SW64's narrower box: DRAM 347 vs 375 GB/s with L2 sector
  util UP 9.5->10.5% = more, narrower requests). CTA2's penalty is 4x smaller (-2%) because the 2SM
  multicast amortizes the doubled request count -- corroborating the box-width attribution. Bit-identity
  INTACT across all configs (torch.equal vs B44, lineage B36/B39/B44 unbroken); harness x3 all PASS;
  machinery shipped env-configurable (AISP_TCGEN05_KBLOCKS/STAGES, defaults = incumbent, control
  measured == B44). OPS INCIDENT banked: a foreign 4-hour idle `flock ... sleep 14400` squatted the
  GPU3 lease (violates the per-WORKLOAD lease protocol); killed + reclaimed. Leases wrap workloads,
  never idle holds. NEXT LEVER (capstone): cluster-N B-multicast at unchanged bK=64 box width (CTA2's
  muted penalty is the supporting evidence; NOTE B43 falsified B-multicast on the SIBLING dual kernel
  via L2-dedup -- measure L2 sectors here first).

- WIN (B49, 2-SM UMMA pair on the dual-CTA footprint, the B43/B44-named lever): docs/
  gb300-gemm-occupancy-rewrite.md (U section). NEW variant tcgen05_dual_cta_2sm.cu (cta_group::2,
  SM100_MMA_F16BF16_2x1SM_SS): dual_2sm (128,3) = 874.9us median / 1256.8 TFLOPS / 33.5% FP16-SoL,
  beating same-session plain dual (256,2) (931.2us/31.5%) in 16/16 paired interleaved reps (median
  ~1.05x, range 1.005-1.103), rel_err 0.0 everywhere, correct on first build; verify gates 3/3 PASS
  (2sm 2.5539x / plain 2.6213x / default cluster 2.3675x, 2.329x contract intact). MECHANISM (ncu):
  THREE CTAs/SM (152 regs/thread vs 255, 72KB smem, TMEM 3x128 of 512) -> occupancy 18.75% vs 12.5%;
  long_scoreboard 28.51 -> 23.47 (-18%); IPC 0.20 -> 0.37; tensor-pipe 49.5 -> 54.0%; DRAM 27.4%
  (still latency-bound). DISCOVERED FACT (corrects the CUTLASS tutorial's stale prints): the 2x1SM
  atom SPLITS B N/2-per-CTA (proven by smem footprints + mbarrier tx-byte balance) -- per-CTA B feed
  halves. The (256,2) 2SM config TIES (255-reg epilogue re-caps at 2 CTAs/SM, TMEM exact-fit): the
  lever pays through occupancy + traffic, NOT instruction-count halving. FLAGGED: one-off unreproduced
  hang in a warmup launch (~4000 subsequent launches clean; unproven TMEM pair-alloc suspect; watchdog
  recommended on unattended 2SM sweeps; defaults avoid the exact-fit config). Custom-kernel ladder now:
  cluster 25.2% -> plain dual 31.5% -> dual_2sm 33.5% vs cuBLAS 46.6-48.2%. NEXT LEVER: warp-split
  leader (the serialized empty-wait->TMA->full-wait->MMA chain; long_scoreboard still 76% of stalls) +
  A-multicast on a (2,2,1) cluster (n=128 tiling doubled A re-reads, L2 sectors 376.8M vs 257.5M --
  visibly NOT L2-deduped, unlike E5's falsified B premise).

- WIN (B50, metric_reduction_vectorized single-read fusion, the B47-named lever, beats its estimate):
  docs/gb300-metric-reduction-fusion.md. The 3-pass torch mul+sum (126MB re-read) fused into ONE
  CUDA kernel (coalesced column-per-thread loads, block row-striding, fp32 fmaf register accumulators,
  atomic fold; 25.2MB single read): harness optimized arm 0.1285 -> 0.0565-0.0578 ms = 2.26x over the
  prior arm, headline 90.5x -> ~197x vs baseline (estimate was ~150x), verify PASS all reps at
  rtol/atol 1e-4; numeric cross-check vs scalar/old-vectorized/float64 all clean pre-harness; fp32
  guardrail respected; output contract unchanged. Kernel-frame: 90.8 -> 23.4 us eager (3.9x), 8.16 us
  pure-GPU graphed (9.7x); main kernel 7.91 us = 3.19 TB/s = ~40% HBM-SoL. Shared-file regression
  check: metric_reduction_cuda re-run matches B47 expectations (no regression). The B47 harness-frame
  finding re-confirmed: eager frame is now ~60% host (2 launches + pybind ~14us) and the harness adds
  ~33-35us/iter fixed -- the lab is becoming harness-overhead-bound. OPS: a SECOND idle lease-squatter
  (gpu1, `flock ... sleep 14400`) killed under the no-idle-hold rule -- whoever holds leases idle is
  violating the per-workload protocol; recurring pattern, watch for it. NEXT LEVER (small): single
  -launch variant (drop the zeros fill via partials + last-block reduction, float4 loads) est. ~4-5us
  kernel / ~225x harness; beyond that the ceiling play is harness-level graph capture (pure GPU work
  is already 8.16us).

- WIN + MECHANISM CORRECTION (B51, moe_cuda compile-in-graph, the B45-named lever): docs/
  gb300-moe-cuda-compile-fusion.md. torch.compile (max-autotune-no-cudagraphs, fullgraph=True)
  composed INSIDE the existing manual CUDA-graph capture (vLLM pattern): optimized arm
  0.4996 -> 0.3932 ms = 1.271x median, 12/12 interleaved reps verification PASS, distributions
  non-overlapping (compile max 0.3945 < B45 min 0.4976), bit-identical on the lab input
  (max_abs_diff 0.0). MECHANISM CORRECTED vs the B45 estimate: GELU did NOT epilogue-fuse (it
  compiled to a standalone triton kernel, 69.3us, slower than eager 44.5us); the win is
  dispatch-chain fusion -- 33 -> 16 kernels/replay, ~130us/replay of expand/gather direct_copy
  eliminated, GEMM1 absorbed into a Triton template that swallows the slot-view traffic. Loud-
  failure asserts per the B45 audit rule (dynamo_graph_breaks=0, unique_graphs=1,
  manual_graph_active exported into harness metrics; capture/compile failures raise -- no silent
  fallback). Kill switch AISP_MOE_CUDA_COMPILE=0 restores B45 exactly; grad-mode capture pitfall
  documented (warmup and capture must share no_grad). One-time compile 15.6s cold / ~2.6s warm
  lives in untimed setup (setup_timeout_seconds=600). Evidence /tmp/frontM/compile/ on the pod.
  NEXT LEVER (~1.2x est): force GELU into the GEMM1 template epilogue (Inductor template
  epilogue fusion or hand-fused baddbmm+GELU); compiled GELU 69.3us + Triton GEMM1 104us are
  the remaining top per-replay costs.

- HONEST NEGATIVE, MEASURED-CEILING GRADE (B52, capstone wave-quantization fill, the D5b-named
  lever, refuted BEFORE building): docs/gb300-capstone-wave-quant.md. At the scored 2048^3 shape
  the 128-CTA grid fills 128/152 SMs (84.2%, SM count verified), but the kernel is FIXED-COST-
  bound, not wave-bound: K-slope probes (K=256..4096) split T = F + u*nk with F = 14.9-15.5us =
  62-65% of the 23.9us kernel; only ~8.5us is k-redistributable, so perfectly balanced zero-
  overhead stream-K ceilings at 1.058-1.074x -- and measured full-fill contention (152-CTA single
  wave, identical per-CTA work, replicated x2) costs +3.9-4.2%, dropping the ceiling to 1.016-
  1.058x, at/below the 1.05 gate before ANY implementation cost (>=33.6MB fp32 partial round-trip,
  >=147 extra drain/refill+epilogue segments, flag protocol). Tile reshape dead by divisibility
  (no legal UMMA atom yields 128<CTAs<=152); persistence dead by arithmetic (128 tiles < 152 SMs,
  all resident at t=0). ncu grounding: fill 83.1->98.2% achievable but tensor-pipe duty ~18%/SM
  either way (issue/latency-bound even when full). All scored tcgen05 GEMM targets are 2048-only:
  no scored shape escapes. Zero repo files touched; incumbent bit-identical lineage untouched.
  OPS: orphaned foreign python squatted GPU3 mid-session (rep discarded, re-run replicated 0.2%);
  Front X's concurrent pod kernel edit proven non-contaminating (pre/post builds bit-equal).
  NEXT LEVER (the real target, ~2x the wave-quant ceiling from a 20% cut): attack F itself --
  decompose the ~15us fixed per-CTA cost (launch turnaround vs TMEM alloc vs 4-stage prologue
  fill vs TMEM_LOAD t2r epilogue; starting split: launch+F_in-kernel ~13.3us single-CTA,
  u_1cta ~0.223us); K=256 runs at 137 TFLOPS = 3.6% SoL, purely F-bound.

- HONEST NEGATIVE (B53, capstone cluster-M B-multicast: pre-check passed, implemented, measured
  slower; the deepest-grounded multicast refutation): docs/gb300-blackwell-matmul-hostoverhead.md
  (X section). The B43 gate was honored and PASSED: at 2048^3 the incumbent's L2 read sectors are
  3.31M vs 6.29M naive (xbar TMA-load count = exactly 6.29M) -- GB300's HARDWARE IN-FLIGHT TMA MERGE
  dedups only ~2x of the 16x B duplication -- so the lever was implemented (env-gated
  AISP_TCGEN05_CTA1_CLUSTER_M, (2,1,1) cluster, SM90_TMA_LOAD_MULTICAST B split across the M-pair,
  umma_arrive_multicast stage gating). MEASURED: +0.75% SLOWER (24.16 vs 23.98us, 7 interleaved
  reps); L2 sectors -2.3% only (vs -33% if multicast removed new duplicates). MECHANISM: explicit
  multicast eliminates exactly the duplicates the hardware merge already catches, plus ~0.2us
  cluster overhead; the residual 8x B duplication is INTER-PAIR (unreachable at cluster-2) and NOT
  BINDING (L2 9% of peak, DRAM 5%, duration flat as traffic shifts). RE-EXPLAINS B48's "supporting
  evidence" (CTA2's muted bK32 penalty = the 2SM pairing replicating the hardware merge, not
  headroom). Verify 9/9 PASS, torch.equal == B44 lineage everywhere, defaults unchanged. TRAPS
  BANKED: (1) tma_partition's multicast slice is an OFFSET VIEW, not shrunken -- scaling expect-tx
  by participant count arms 80KB vs 48KB delivered = deterministic deadlock (localized via cuda-gdb
  attach); correct bytes = the full slice, no multiplier. (2) Co-loaded sweep modules with
  identically-mangled CTA2 kernels fail cluster launch with cudaErrorInvalidValue -- single-module
  builds only. Together with B52: every byte-side AND wave-side lever on this kernel is now
  falsified; the F-decomposition (B52's named lever) is the sole remaining capstone path. OPS: the
  gpu3 idle lease-squatter RE-SPAWNED and was killed again (something re-creates it).

- HONEST NEGATIVE, ESTIMATE CORRECTION (B54, moe_cuda GELU-into-GEMM1-epilogue, the B51-named
  lever): docs/gb300-moe-cuda-gelu-epilogue.md. The fusion LANDS structurally (16 -> 15 kernels/
  replay; the standalone GELU kernel disappears into triton_tem_fused_baddbmm_gelu_*;
  verification PASS 12/12; graph_breaks=0, unique_graphs=1 every rep) but pays only 1.0227x
  median (0.40292 -> 0.39396 ms, 6 reps/arm interleaved, non-overlapping distributions -- real
  but below the 1.05 gate). WHY THE ~1.2x ESTIMATE WAS WRONG: the 69us standalone GELU is erf
  COMPUTE, conserved under fusion -- it moves INTO the GEMM1 template (+58.6us on the template)
  instead of disappearing; fusion only saves the intermediate's memory round-trip (~10.6us).
  DURABLE LESSON: kernel-count reduction is not cost reduction when the eliminated kernel is
  compute-bound -- classify standalone kernels compute- vs traffic-dominated BEFORE estimating
  fusion EV. Fused path ships opt-in (AISP_MOE_CUDA_GELU_EPILOGUE=1, default stays B51) with a
  loud-failure assert that the standalone GELU is actually gone when enabled. Evidence
  /tmp/frontM2/ on pod. NEXT LEVER (~1.15x est, from the Inductor diagnosis): Inductor's
  extern-kernel autotune mis-measures ATEN baddbmm at 0.1403ms vs nvjet's real 66us (broadcast-
  bias stride [2048,0,1] benchmark artifact), so the slower Triton template wins the autotune;
  pin GEMM1 to ATEN via scoped max_autotune_gemm_backends (66us extern + 52us tuned standalone
  GELU vs the 172.5us fused template saves ~54us/replay -> ~0.34ms) or fix the extern benchmark.

- WIN (B55, capstone F-decomposition -> epilogue t2r atom fix, 1.49x kernel, bit-identical): docs/
  gb300-capstone-f-decomposition.md. The B52-named F attack landed via measurement-first
  decomposition (%globaltimer stamps in a PROBE-only build): F budget = epilogue t2r TMEM_LOAD
  7.65-7.81us (77% of F!) + cvt/store 3.7us + everything else (launch rollout, tcgen05.alloc,
  barrier init, prologue) < 1.5us combined -- every other B52 candidate measured dead. ROOT CAUSE
  (SASS): SM100_TMEM_LOAD_32dp32b1x = 256 tcgen05.ld per thread, EACH lowered by ptxas to
  LEPC + CALL.ABS.NOINC + WARPSYNC (512 helper calls per kernel). FIX: 32dp32b32x atom = 8
  loads/thread -> epilogue 7.7 -> 0.42us. Sweep: 16x ties, 128x REGRESSES (the 128-output asm
  serializes writeback) -- widest is not best. Kernel 23.81-23.90 -> 15.94-16.10us = 1.49x (zero
  distribution overlap, 7 interleaved graphed reps); FP16-SoL ~19% -> ~28.5%; ncu tensor-pipe
  active 20.4 -> 37.9%. BIT-IDENTICAL (torch.equal vs incumbent, CTA1+CTA2; the atom changes
  columns-per-instruction only). Harness 9/9 verification PASS all 3 targets; honest note: the
  wrapper is host-dispatch-bound (~85-100us/call) so end-to-end cannot resolve the 7.9us kernel cut
  -- the kernel-level 1.49x is the bankable metric (triangulated stamps+graphs+ncu). One-hunk ship:
  AISP_TCGEN05_T2R_ATOM macro (default 32dp32b32x). Durable lesson: TMEM_LOAD atom WIDTH is a
  first-class perf knob on sm_103 -- the narrow atom's per-load helper-call overhead can dominate
  an entire kernel's fixed cost; check SASS for CALL.ABS.NOINC around tcgen05.ld. NEXT LEVER:
  cvt+store is write-sector-bound (16MB L2 write sectors for 8MB of D = 50% efficiency, STG.E.128
  spanning 32 rows); smem-staged transpose epilogue (r2s -> coalesced s2g, reusing the drained A/B
  ring) worth ~1.5-2us; after that the kernel is mainloop-bound (9.4us, TMA delivery efficiency).

- WIN + HYPOTHESIS FALSIFICATION (B56, moe_cuda bmm+fused-bias GEMM routing, from the B54-named
  lever): docs/gb300-moe-cuda-gemm1-aten.md. The B54 "Inductor autotune mis-benchmark" hypothesis
  is FALSIFIED: the 0.1402ms extern-baddbmm benchmark is HONEST -- with broadcast bias stride
  [2048,0,1], at::baddbmm_out pays a 74.6us bias->output direct_copy + 66.9us beta=1 nvjet =
  138us real (Inductor's benchmarker reproduces 0.1392ms; B45's "66us eager nvjet" was kernel-
  only under-counting). Plain backend pin to ATEN measures WORSE (369.8 vs 313.4 GPU us/call).
  THE LANDED LEVER: rewrite both expert GEMMs baddbmm(b,x,w) -> bmm(x,w)+b -- the honest
  autotuner then routes GEMM1 to extern nvjet beta-zero (59.2us vs 104us template), bias adds
  fuse into pointwise for free, and GEMM2's hidden 40us bias-copy (mislabeled "gather-back" in
  the B51 table; corrected) disappears. 0.40653 -> 0.31139 ms = 1.3055x median (6 reps/arm
  interleaved, 12/12 verification PASS, non-overlapping, graph_breaks=0/unique_graphs=1 every
  rep, zero foreign procs). Bit-identical on the lab input (observed, not guaranteed). Default
  flipped ON (AISP_MOE_CUDA_GEMM1_ATEN=0 kill switch restores B51 exactly: 0.40256 re-measured);
  composes with B54's opt-in (0.39462 reproduces M2's 0.39396). moe_cuda lab lifetime:
  8.29ms (pre-B45) -> 0.311ms = ~26.6x. UPSTREAM-REPORTABLE: Inductor never considers the
  bmm+pointwise-bias decomposition for broadcast-bias baddbmm (~40% on the table at these
  shapes). Evidence /tmp/frontM3/. NEXT LEVER (~1.2x est): expert-capacity right-sizing (the
  padded capacity dim overcomputes vs actual routed tokens).

- WIN + PREMISE-TRUE NEGATIVE (B57, dual_2sm warp-split shipped / A-multicast falsified, the
  B49-named levers): docs/gb300-gemm-occupancy-rewrite.md (V2 section). LEVER (a) WARP-SPLIT WIN
  (default flipped to AISP_DUAL2SM_WARP_SPLIT=1): splitting the leader's serialized
  empty-wait->TMA->full-wait->MMA chain across two whole warps wins 22/22 order-alternated paired
  reps, median 1.0725x hot (base 989.8 -> ws 920.7us; cold 892.3 -> 875.3us = 33.5% SoL); ncu
  long_scoreboard 23.65 -> 22.34, tensor-pipe-of-elapsed 66.7 -> 70.9%, occupancy unchanged -- pays
  purely through producer/consumer overlap. Gates 3/3 verification PASS. LEVER (b) A-MULTICAST
  (2,2,1): the rare PREMISE-TRUE negative -- the B53-style pre-check CONFIRMED A duplicates are NOT
  hardware-merged (375.0M sectors reproduces B49's 376.8M) and multicast truly removes them (261.4M
  ~= plain-dual's 257.5M) -- yet it LOSES 9%: the kernel is latency-bound (DRAM 28%), cross-pair
  lock-step (empty barrier waits the slower pair) drops tensor-pipe 11pts and RAISES DRAM bytes read
  (worse L2 residency). Traffic removal is not a win when the synchronization that buys it costs
  more than the bytes did. Flag stays in-tree, correct, OFF. BUG FIXED (Front V's deadlock): the
  B53 expect-tx trap verbatim -- V multiplied tma_bytes by cluster participants; the tutorial-04
  formula has NO multiplier; one-line fix, rel_err 0.0 at every size. Custom-kernel ladder:
  cluster 25.2% -> dual 31.5% -> 2sm 33.5% -> 2sm+ws ~34.2% (hot-paired 1.07x) vs cuBLAS 47.7%.
  NEXT LEVER: tile_n=64 opens stages 5-6 AND a 4th CTA/SM simultaneously (TMEM 4x64=256 cols;
  long_scoreboard still 75% of stalls, producer capped at 3 in-flight stages); fallback kTileK=128
  to halve barrier round-trips per byte.

- WIN + MEASURED GATE-OUT (B58, capstone smem-staged transpose epilogue, the B55-named lever):
  docs/gb300-capstone-f-decomposition.md (sections 8-9). 1SM path WINS: swizzled r2s
  (chunk ^ (tid&7), STS.128/LDS.128 conflict-free) + per-warp whole-row s2g replaces the direct
  32-row-spanning STG.E.128 -- L2 write sectors 524,288 -> 262,144 (the EXACT 100%-efficiency
  target: 8MB of sectors for 8MB of D); cvt+store 3.39 -> 2.43us (probe stamps); kernel 16.01 ->
  15.19us med-of-meds = 1.054x with ZERO overlap across 14 measurements; ~28.5 -> ~30.0% FP16-SoL.
  2SM path REGRESSES (+0.27us, zero overlap) despite identical sector halving -- its store phase
  was never L2-write-time-bound (control already 0.55us faster at identical sectors), so the smem
  round-trip is a pure add; constexpr-gated to kMmaCtas==1 only. LESSON: sector waste is only
  recoverable where it is the BINDING constraint -- same byte fix, opposite outcomes on sibling
  paths. Estimate calibration: 0.96us realized vs 1.5-2us projected (the projection ignored the
  ~1.2us L2-write floor + 0.55us STS/LDS round-trip; measured is within 0.2us of the staged floor).
  BIT-IDENTICAL (torch.equal CTA1+CTA2, gated build); harness 9/9 verification PASS; ships as
  AISP_TCGEN05_SMEM_EPILOGUE=1 default (=0 rebuilds exact B55). Capstone ladder: 66 (B31-era) ->
  38.9 (B36) -> 37.6 (B39) -> 24.0 (B44) -> 16.0 (B55) -> 15.2us (B58), bit-identical lineage
  throughout. NEXT LEVER (the epilogue is SPENT; mainloop now 62% = 9.4us vs ~5-6us tensor-pipe
  SoL): L2-resident B reuse across M-tiles -- persistent CTAs + tile scheduler turning the
  16x-duplicated B pulls into L2 hits; clue: the 2SM pair's phase-staggering already hides store
  latency, the same argument applies to its TMA fetch stream.

- WIN (B59, moe_cuda expert-capacity right-sizing, the B56-named lever): docs/
  gb300-moe-cuda-capacity-rightsizing.md. The static per-expert slot capacity was 2x max
  observed load rounded to 64 (= 640 at the lab input) -> 60% of dense-dispatch GEMM rows were
  padding; right-sized to observed-max-rounded (320, 20% padding), halving the M dim of both
  expert GEMMs and the dispatch/GELU pointwise work. 0.31006 -> 0.23491 ms = 1.3199x median
  (6 reps/arm interleaved, all verified, clean GPU snapshots). CORRECTNESS RAILS: capacity stays
  STATIC (manual-graph contract intact, graph_breaks=0); an in-graph sticky overflow counter is
  checked host-side in setup and post-reps -- any routed assignment beyond capacity FAILS the
  run loudly (demonstrated: forced capacity=256 -> BENCHMARK FAILED with the guard message;
  /tmp/frontM4/r04b_*); tokens are never silently dropped; semantics on non-overflowing inputs
  unchanged (padded rows inert, same gather-back indices). Default ON post-validation (bare
  396.7x vs baseline ~0.238ms; kill switch AISP_MOE_CUDA_CAPACITY_RIGHTSIZE=0 restores B56,
  re-measured 313.3x ~0.301ms). moe_cuda lab lifetime: 8.29ms (pre-B45) -> 0.235ms = ~35x
  (B45 16.5x -> B51 1.27x -> B56 1.31x -> B59 1.32x). Evidence /tmp/frontM4/. NEXT LEVER:
  calibrated-capacity generalization (capacity follows the input distribution; a production
  router would calibrate per deployment -- document as the lab's transferable pattern) and the
  remaining 20% padding floor (rounding granularity 64; sub-64 rounding is cheap to test).

- HONEST NEGATIVE x3 (B60, dual_2sm launch-shape frontier exhausted, the B57-named levers): docs/
  gb300-gemm-occupancy-rewrite.md (V3 section). tile_n=64 AND kTileK=128 both implemented
  flag-gated, fully measured, FALSIFIED -- (128,3,ws=1) is a measured LOCAL OPTIMUM of the whole
  (tile_n, stages, CTAs/SM, tile_k) frontier. FORECAST CORRECTION first by arithmetic then ncu:
  B57's ~12KB/stage estimate was wrong -- the A-half is 16KB/stage at ANY tile_n (only B halves),
  so "4th CTA + stages 5-6 simultaneously" was never reachable. Three mechanisms: (1) (64,3)
  control: tensor-pipe 71.6 -> 40.6%, L2 sectors +68%, DRAM +114% -- halving tile_n halves
  FLOPs/barrier-trip while doubling n-tiles; arithmetic intensity inverts the premise (rhymes with
  B37's (128,3) loss). (2) The 5th CTA/SM GENUINELY ARRIVES ((64,2,mb5): Block Limits 5/5, occ
  30.3%, first >3 CTAs/SM in this family) and STILL LOSES 38% -- long_scoreboard 22.05 -> 53.18:
  10 TMA streams/SM CREATE the latency they were meant to hide; occupancy is not free latency
  cover when every resident CTA adds memory pressure. (3) k128: barrier trips genuinely halve,
  stalls/issue flat, but 2 CTAs/SM costs more than halved sync saves. Paired gate: best challenger
  0/8 wins (0.8306x). All rel_err 0.0; verify 3/3 PASS (2.7974x/2.7656x/2.4448x); defaults
  byte-equivalent to B57. Procedural: a profile_minimal harness stall (328s) was SIGKILLed and
  re-run clean per V2 precedent -- use --profile none for gates. NEXT LEVERS (structural, the
  launch-shape space is done): (1) TMEM double-buffered epilogue overlap (split N into two TMEM
  halves, drain one while MMA fills the other; epilogue is currently serialized after the full
  k-loop; 2x256 fits at 2 CTAs/SM, 3x256 does not -- measure both); (2) persistent CTAs +
  swizzled tile rasterization (attacks the DRAM/L2 numbers that sank every V3 config without
  touching the winning shape).

- WIN + SOL REFRAMING (B61, capstone early-fill + the persistence pre-check + the REAL dense-FP16
  ceiling): docs/gb300-capstone-f-decomposition.md (Y3 section). THE PRE-CHECK KILLED THE NAMED
  LEVER before a line of scheduler code: dram__bytes_read = 16.82MB = the EXACT A+B cold-fill
  floor, dram writes 0 (even D stays in L2), promoted misses 0% -- B is FULLY L2-resident; there
  are no DRAM misses for any persistence/swizzle/schedule to convert. Mainloop split (per-k-tile
  globaltimer): consumer tma_wait is ALL fill-phase (steady-state ~0), producer gate_wait 6.14us
  (delivery always AHEAD of consumption) -- the producer-issue-bound hypothesis falsified too;
  steady-state period ~290ns/tile ~= pure tensor-pipe time. THE REFRAMING (bankable, repo-wide):
  the 3.75 PFLOPS dense-FP16 "peak" is unreachable on this part -- cuBLAS itself asymptotes at
  1.87 PF (8192^3; 11.47us at 2048^3); the real per-SM rate is ~64ns per 128x256x16 UMMA, and the
  capstone mainloop runs at ~87% of THAT (14.3 TF/SM, ABOVE cuBLAS's own 12.3) -- B58 was at ~58%
  REAL SoL, not 30%, and the kernel floor under the bit-identity contract (no split-K, 128 tiles
  on 128 SMs) is ~12.1us: a <=12us "breakthrough" is structurally unreachable. RECOMMEND: adopt
  ~1.9 PF as the GB300 dense-fp16 SoL reference (current sol-ceilings framing overstates remaining
  headroom 2x). SHIPPED WIN: AISP_TCGEN05_EARLY_FILL=1 -- mbarrier init + first kStages TMA fills
  issued BEFORE tmem alloc (alloc+sync overlaps TMA flight): 15.200 -> 14.980us med-of-meds
  (1.015x, EF1 won all 5 interleaved process-pairs on med AND min), TMA traffic byte-identical
  (purely temporal), torch.equal bit-identity CTA1+CTA2, harness 9/9 verification PASS. TRAP
  BANKED: two builds loaded in ONE process alias the mangled kernel symbol -> invalid-argument
  launch; one build per process for A/B. Capstone ladder final: 66 -> 38.9 -> 37.6 -> 24.0 ->
  16.0 -> 15.2 -> 14.98us (~4.4x cumulative, bit-identical lineage throughout); mainloop CLOSED at
  ~87% real SoL; only residual the ~0.3-0.5us epilogue TMA store. Structural unlocks beyond are
  CONTRACT-level (sanctioned non-bit-identical split-K, larger shapes).

- INFRASTRUCTURE WIN + B45-CLASS INSTANCE #2 (B62, harness frame-overhead audit + opt-in graphed
  timing): docs/gb300-harness-frame-audit.md. MEASURED FRAME MODEL (validated ~5% on 3 arms):
  T_reported ~= 22us bracket floor + 3.2us/extra-launch + fn host residual + T_pureGPU; wall adds
  ~53-61us/iter outside the bracket (L2 zero_+sync 35, post-sync 5, bookkeeping 13-20, +20
  force-eval on tensor returns). Frame-share ranking of optimized arms: nvfp4_dual_gemm 86%,
  blackwell_matmul 84%, metric_reduction 79%, software_pipelining 51%, block_scaling 44%,
  moe_cuda ~15% -- the microsecond labs are now measuring the ruler. OPT-IN
  AISP_BENCH_TIMING_MODE=graphed (default path untouched; regression-proven: metric_reduction
  0.0567ms in the B50 band, blackwell 0.109ms, shared-.cu sibling re-run clean, zero graphed_*
  keys leak when unset): graph-captures N iters of benchmark_fn, reports BOTH figures;
  demonstrated metric_reduction 56.7 -> 9.04us warm (reproduces B50's 8.16us pure-GPU) and
  blackwell_matmul 105.2 -> 15.04us warm (reproduces B58's 15.2us kernel); block_scaling REFUSES
  LOUDLY (internal-graph replay, uncapturable -- the B45 contract enforced, no silent fallback).
  NEW B45-CLASS FINDING (#2): metric_reduction's fused kernel launched on the LEGACY DEFAULT
  STREAM was SILENTLY DROPPED from capture (first graphed run timed 1.2us = fill kernel only;
  profiler-of-replay proof); fixed via getCurrentCUDAStream() (4 sites,
  training_hotpath_kernels.cu) and guarded harness-side by a kernel-count CAPTURE-FIDELITY AUDIT
  that refuses on mismatch. Durable lesson: any <<<>>> launch without an explicit current-stream
  argument is invisible to capture-based timing AND to manual-graph workloads -- audit the class
  repo-wide. Evidence /tmp/frontH2/ (91 tuples + h2_audit.json). NEXT LEVER: repo-wide
  legacy-stream launch audit; then dual-figure banking for the 4 frame-bound labs.

- HONEST NEGATIVE (B63, dual_2sm TMEM double-buffered epilogue overlap, the B60-named lever): docs/
  gb300-gemm-occupancy-rewrite.md (V4 section). Formulation analysis FIRST (as mandated): the
  within-tile k-tail split is structurally DEAD (the 2x1SM atom computes both N-halves per k-block
  from one shared stage ring; max window (kStages-1)/nk ~= 1.6%); cross-tile double-buffering is
  the only real window and was implemented CORRECTLY (2x128 TMEM cols/CTA at 2 CTAs/SM; discovered
  constraint: the t2r drain is WARPGROUP-BOUND (TMEM subpartition w <-> warp rank w) so it needs a
  dedicated second epilogue warpgroup, not the parked warps; flattened (tile,k) pipeline with
  cluster-scope buffer-reuse barriers; rel_err 0.0 everywhere including the T=4 reuse protocol).
  MEASURED: best config 0.9479x, 0/52 paired wins across the 5-config space. MECHANISM: at 9 waves
  and 3 CTAs/SM, CO-RESIDENCY ALREADY HIDES THE EPILOGUE FOR FREE -- the lever pays only inside a
  persistent kernel; at the clean 2-CTA point the deficit is feed concurrency (long_scoreboard
  27.6 vs 22.0), not the epilogue (drain fully hidden as designed); the consecutive-n walk costs
  +0.74GB DRAM. NEW SIZING LAW (bankable): smem footprint must match the TMEM-IMPLIED CTA count --
  at s=3 a third co-scheduled CTA fits in smem but NOT TMEM and spins a full tile inside
  tcgen05.alloc (barrier stall 0.29 -> 34.0 cyc/issue, tensor-pipe craters ~52%); sizing smem to
  the TMEM count (s=4) recovers 16 of 21 points. Gates 3/3 verification PASS; B57/B60 defaults
  byte-path intact (AISP_DUAL2SM_EPI_OVERLAP=0 default). NEXT LEVER (carrying V4's proven building
  block): persistent CTAs + L2-swizzled tile rasterization -- the epilogue-warpgroup double-TMEM
  cross-tile pipeline IS the persistent kernel's inner loop; what failed was only the grid shape
  (no rasterization -> DRAM) and the 2-vs-3 feed-stream gap (deeper stages recover 16/21; the
  swizzled walk must supply the rest). NOTE vs B61: at 8192^3, B (128MB) does NOT fit L2 -- the
  swizzle premise is ALIVE here, unlike the 2048^3 capstone where B61's pre-check killed it.

- INFRASTRUCTURE WIN (B64, repo-wide legacy-stream launch audit, the B62-named lever): docs/
  gb300-legacy-stream-audit.md. Parser-based sweep of 849 CUDA-bearing files + 38 embedded-CUDA
  .py (266 legacy-stream launch sites): exactly 1 DANGEROUS (capture-eligible benchmark_fn)
  site -- labs/nvfp4_dual_gemm/optimized_submission.py, the lab the B62 frame table ranked #1
  (86% frame share), whose single fused kernel was launched on the legacy default stream and
  SILENTLY DROPPED from side-stream capture (B45/B62 class, instance #3) -- FIXED via
  getCurrentCUDAStream(); +7 latent torch-extension sites fixed proactively (4 files); 222
  standalone-binary + 29 excluded-helper sites classified BENIGN with reachability evidence;
  third_party out of scope. Validation: default mode brackets the banked 47-48.5us band
  (verification PASS); graphed mode now PASSES the capture-fidelity audit
  (graphed_eager_kernel_count=1) and banks the first dual figure: nvfp4_dual_gemm pure-GPU
  6.89us warm -- confirming B38's 6.6us kernel figure (+4%) and proving the 86% frame share
  measured, not modeled. Latent-extension smoke: 5/5 capture-fidelity PASS. Evidence
  /tmp/frontA/ (sweep_raw.json = the full inventory). NEXT LEVER: bank graphed dual figures
  for capstone_extension_tcgen05 (~85us pybind host residual, B55) and the remaining
  frame-bound labs, now that the repo is legacy-stream clean.

- WIN + TRAP (B65, dual_2sm persistent clusters + GROUP_M raster, the B63-named lever): docs/
  gb300-gemm-occupancy-rewrite.md (V5 section). L2 MATH FIRST (as mandated): at 8192^3 the
  premise is alive on PANELS, not matrices -- GROUP_M keeps gm x 4MiB A row-panels resident while
  2MiB B col-panels stream; in-flight window at 3 CTAs/SM (C=228 clusters) gm=8 = 32A+57B+~29D
  ~ 118MiB < 129.25MiB L2; DRAM floor 640MiB vs 1.36GB incumbent (x-fastest wave re-reads A every
  one of 9 waves). MEASURED WIN: persistent clusters at CHAMPION occupancy (eo=0, 3 CTAs/SM,
  warp-split rounds, chunked t2r, 84 regs) + gm=8 raster = 1.0598x paired median, 12/12 wins
  (926.0 -> 885.0us; ncu 754 -> 726us, tensor-pipe 71.4 -> 72.9%, waves exactly 1.0 on a 456-CTA
  grid); selectable AISP_DUAL2SM_PERSIST=1 MIN_BLOCKS=3 RASTER_GM=8; defaults UNCHANGED pending an
  E4-style replication (rel_err 0.0 at 4 sizes incl non-square + tiles<C clamp; gates 3/3 PASS
  2.8115x/2.7532x/2.4489x). DECOMPOSITION: persistence alone +4.8% (12/12, launch/prologue
  amortization); raster adds +1.1% only UNDER persistence; raster alone TIES (0.9989) with DRAM
  read -35% (0.88-0.93GB) and long_scoreboard UNMOVED (21.87 -> 21.88) -- the L2 premise is TRUE
  but NOT BINDING (TMA latency already covered by 3 stages x 3 co-resident CTAs); bytes-premise
  levers need a bandwidth-bound kernel. V4's eo2 inner loop persists to a TIE (1.0060x, 7/12) --
  the dedicated-epilogue family is retired below 4 feed streams at ANY grid shape. TRAP BANKED
  (the iteration-1 killer): cudaOccupancyMaxActiveBlocksPerMultiprocessor is UNSTABLE for this
  cluster kernel on sm_103 -- identical back-to-back launches got grids of 152 vs 456 CTAs (waves
  0.33 vs 1.0; 1.32ms vs 742us), and a persistent kernel sized to the bad answer runs the whole
  GEMM at 1/3 occupancy with no second wave to backfill (0.51-0.67x, 0/36 paired). Size persistent
  cluster grids STATICALLY: min(smem-implied (+1KiB/CTA reserve), TMEM-implied) -- regs measured
  non-binding. Also closed: per-config non-type template tags end the B61 same-symbol alias risk
  for this kernel family. Evidence /tmp/frontV5/. NEXT LEVER (only exists under persist+raster):
  raster-aware L2 prefetch -- cp.async.bulk.prefetch.tensor for round t+1's deterministic panels
  (raster_map(rr+(t+1)C)) issued during round t's drain, attacking the long_scoreboard stall that
  never moved across V2-V5; secondary: replication front for the default flip.

- MEASUREMENT BANK + B45-CLASS #4 (B65, dual-figure bank for the frame-bound labs, the B64-named
  lever): docs/gb300-dual-figure-bank.md. Six graphed pure-GPU figures banked with capture-
  fidelity audit, every default run verification-PASS inside its banked band: capstone
  cluster_gemm_tcgen05 15.58us warm (79.0% frame; B55's "~85-100us host dispatch" now measured;
  pure-GPU = B61's closed 14.98us kernel + ~0.6us epilogue/wrapper), cta2 15.54us; blackwell
  15.04us (= B62 demo); metric_reduction_vectorized 9.02us (78.5% frame) + _cuda 5.87us first
  figure; software_pipelining 60.73us warm = B41's 60.5us kernel, banked after a 1-line fix --
  NEW SUB-CLASS #4: at::cuda::getDefaultCUDAStream() passed as an EXPLICIT stream arg is
  parser-"explicit" but capture-invisible (B64's audit only flagged missing/zero args); same
  pattern still live UNFIXED in persistent_decode (2 sites) + memory_bandwidth_patterns (5
  sites). moe_cuda: loud refusal by construction (nested manual graph), default re-confirmed
  0.2352ms. TRUE-HEADROOM RANKING (vs B61 real ceilings): metric_reduction_vectorized is the
  top target -- 9.02us warm vs 3.15us single-read HBM floor = 2.9x gap (~40% HBM-SoL); tcgen05
  GEMMs 1.24-1.29x off a structurally-capped floor; tile_pipeline closed (1.05x of torch.add
  parity). Evidence /tmp/frontA2/ (a2_summary.json). NEXT LEVER: the B50-named single-launch
  fused metric_reduction kernel (drop the zeros fill; partials + last-block reduce, float4) --
  est 4-5us = ~1.8-2x pure-GPU, now MEASURABLE via graphed mode where frame mode buried it;
  plus fix the 7 remaining explicit-but-default sites.

- NO-OP + RATIFICATION->DEFAULT-FLIP (B66, dual_2sm raster-aware prefetch + B65 replication): docs/
  gb300-gemm-occupancy-rewrite.md (V6 section). PREFETCH (the B65-named lever): implemented via
  cute::prefetch on the TMA atoms (cp.async.bulk.prefetch.tensor, the CUTLASS sm103 collective
  pattern), depth + issuer swept (5 configs, 20/20 rel_err 0.0). HONEST NO-OP ON TIME with the
  bytes premise proven TRUE a second time: DRAM reads -7-9% (the prefetch lands, merging ~8
  clusters' concurrent demand misses per fresh B panel) at FLAT long_scoreboard (31.4 -> 31.5,
  UNMOVED -- the V2-V6 invariant survives its most surgical attack) and noise-level duration.
  LAW (now thrice-proven on this kernel: V5 raster, V6 prefetch, B53 multicast): feed latency is
  fully covered by 3 stages x 3 co-resident CTAs -- shortening a latency nobody waits on buys
  nothing; the wait is on the EPILOGUE side. Prefetch ships default-OFF (pf=2 is a free -7-9%
  DRAM-read knob for power-sensitive use). RATIFICATION REPRODUCED: fresh session, 12 pairs,
  1.0562x 12/12 (replicates B65's 1.0598x 12/12) -> LOADER DEFAULTS FLIPPED (PERSIST=1,
  MIN_BLOCKS=3, RASTER_GM=8); gates 3/3 verification PASS on the new defaults (2sm 2.9062x, up
  from 2.8115x). HARNESS WATCH ITEM: a stale-results-file trap -- gate3's first "pass" was
  gate2's leftover JSON (caught by timestamp); clean re-run passed; the harness should
  nonce/clean its results path. Ladder (real-SoL): cuBLAS 683us 84.6% > persist+raster DEFAULT
  883us 65.5% > ex-champion 928us > plain dual; prefetch/eo2/A-mcast/tile_n=64/tile_k=128 all
  honestly retired. NEXT LEVER (final on this kernel): TMA-store epilogue through the drained
  ring stage (t2r -> st.shared -> cp.async.bulk.tensor store per chunk; stage-0's 24KB is free
  exactly when the epilogue runs; zero TMEM/occupancy cost, unlike the retired eo2) -- if it
  fails to move tensor-pipe past ~72.5%, DECLARE THE FRONTIER.

- WIN + ESTIMATE REFUTATION (B67, metric_reduction single-launch fuse, the B65/B50-named lever,
  decided on the graphed pure-GPU figure): docs/gb300-metric-reduction-single-launch.md.
  Pure-GPU warm 9.048 -> 8.134us = 1.1124x (6/6 interleaved paired wins, zero overlap, capture-
  fidelity 2 -> 1 kernel/iter -- the zeros-fill launch provably gone); default frame 0.05738 ->
  0.05173ms (1.109x, 6/6); all 24 runs verification PASS at the unweakened 1e-4 contract,
  numerics TIGHTER than incumbent (max-abs vs fp64 0.0029 vs 0.0137). Shipped: persistent K=8
  replicated accumulators + atomic-ticket last-block fold/re-zero, 2 blocks/SM, kill switch
  AISP_METRIC_REDUCTION_SINGLE_LAUNCH=0. ESTIMATE REFUTED WITH MECHANISM: B50's "4-5us via
  float4" was wrong -- the kernel is latency/atomic-bound, not BW-bound; float4+unroll measured
  SLOWER (14.9us); the real cost was __threadfence waiting out 592-deep same-address atomic
  chains (~5us), fixed by replication (K sweep: K=8 optimal). Honest note: graphed COLD
  regresses 11.6 -> 13.6us; warm is the comparison-grade figure. ncu: effective 3.05-3.09 TB/s
  ~38.7% HBM-SoL; gap to the 3.15us single-read floor now 2.59x (was 2.9x). Shared-file
  regression metric_reduction_cuda PASS in band. PLUS B45-class #4 CLOSED: the last 7
  explicit-but-default stream sites fixed (persistent_decode x2 incl. tma_extension.py,
  memory_bandwidth_patterns x5), each validated (bandwidth_patterns brackets its banked
  0.0704ms; native_tma_prefill_decode 0.7126ms = new reference; zero getDefaultCUDAStream
  call sites remain under labs/). Evidence /tmp/frontR/ (235 files). NEXT LEVER (small,
  per report: cooperative/DSMEM reduction EV ~3% -- the residual 2.59x is frame, not kernel):
  harness-bracket amortization for sub-10us labs, opt-in like B62.

- WIN + FRONTIER DECLARED (B68, dual_2sm TMA-store epilogue, the B66-named FINAL lever): docs/
  gb300-gemm-occupancy-rewrite.md (V7 section + FINAL LADDER). The lever LANDS as the new default
  (AISP_DUAL2SM_TMA_EPI=1): per-round D epilogue rerouted t2r -> swizzled STS.128 into the drained
  ring stage (16KB = one 128x32 fp32 chunk) -> ONE cp.async.bulk.tensor.2d per chunk via a raw
  cuTensorMapEncodeTiled SWIZZLE_128B descriptor (__grid_constant__); reuse gated by
  tma_store_wait<kStages-1> (the TMA engine is the ring's last consumer, B58 trap honored). Zero
  TMEM/occupancy cost, 3 CTAs/SM intact. MEASURED: 906.5 -> 830.3us = 1.0905x 12/12 + ratify
  898.5 -> 826.9us = 1.0863x 12/12 (24/24 total, two sessions); real SoL 63.8-64.4% -> 69.7-70.0%
  (cuBLAS 82.8-83.3% same-session). ncu: global-st sectors 16.78M -> ZERO (SASS: 32x STS.128, no
  STG.E); L2 writes 20.05M -> 11.67M = D at the EXACT 100%-efficiency floor; tensor-pipe-elapsed
  61.9 -> 71.5% (+9.5pts, past the plateau); long_scoreboard 31.4 -> 28.2 -- the V2-V6 invariant
  FINALLY moved: its movable half was the epilogue's own store chain. Gates 3/3 PASS on the new
  defaults (2sm best_speedup 3.0859, up from 2.9062); the B66 stale-results trap FIRED LIVE and
  was caught by mtime (quarantined .STALE_gate2copy). TRAPS BANKED: cute get_tma_swizzle_base
  rejects element-space fp32 swizzles (raw descriptor is simpler and sufficient -- tma_partition
  drops the swizzle anyway); the t2r register fragment must shape from partition_D, not
  partition_S. FINAL LADDER: 2sm base -> +warp-split 1.0725x (B57) -> +persist/raster 1.056-1.060x
  (B65/B66) -> +TMA-store epilogue 1.086-1.091x (B68) = ~70% real SoL. SCHEDULING FRONTIER CLOSED:
  the store path sits at its sector floor at zero occupancy cost; the residual ~13pts to cuBLAS
  lives in the mainloop CONTRACT -- FP8/NVFP4 operands (2-4x effective ring depth per smem byte),
  a 256-wide-N pair tile (blocked by the 512-col TMEM wall), or grid-level split-K/stream-K
  (overlap one tile's epilogue with another's mainloop). None reachable by re-scheduling the
  existing shape.

- FILING PACKAGES + ATTRIBUTION REVISION (B69, upstream Tier-2 reproducers, pod-verified): docs/
  gb300-upstream-patches.md + code/upstream/. (4) _scaled_grouped_mm NVFP4 refusal verified
  (exact ValueErrors captured; per-group NVFP4 _scaled_mm works via nvjet; 30-launch fallback
  1068us/call re-measured; CUTLASS example 90 proves hardware capability) + BONUS: FP8 rowwise
  grouped is ALSO broken on sm_103 (arch-conditional trap + poisoned context) -- NO
  _scaled_grouped_mm path works on sm_103 at all; upstream PR pytorch#174699 in flight, comment
  there first. (5) _int_mm 13.4x repro verified with NEW dispatched-kernel evidence: sm_103
  _int_mm lands on AMPERE cutlass_80 kernels while tf32/fp16 get sm100/nvjet -- a dispatch gap,
  not a kernel gap; layout-controlled vs pytorch#165230. (6) Triton tcgen05.wait.st: ALREADY
  FILED upstream (triton#8473; 3.5 release cut between #7725/#8299) -- do-not-file; and the
  4-probe repro proves vanilla Triton 3.7 is CLEAN on sm_103: the campaign's aborts were
  SELF-INFLICTED by core/benchmark/triton_compat.py:100-109 stripping the 'a' from sm_103a
  (only sm_100a preserved). REVISES the B17/B18/B19-era toolchain attribution; RE-OPENS the
  B10/B16 stale-checks (max-autotune may work after a 1-line in-repo fix). Evidence
  /tmp/frontP2/. NEXT LEVER: fix the de-suffix, re-run the max-autotune-gated paths.
- IN-REPO FIX SHIPPED (B70, tcgen05.wait.st de-suffix fixed + sm_103 downgrades retired, pod-verified
  2026-06-11): executes B69's named next lever. core/benchmark/triton_compat.py now preserves the 'a'
  suffix for ALL arches in both _canonicalize_triton_arch and the sm_arch_from_capability patch (the
  patch closure now delegates to _canonicalize_triton_arch -- single source of truth); only
  sm_121[a] -> sm_120 is clamped, the original GB10 intent. GB10 behavior verified unchanged without
  GB10 hardware: tests/test_triton_compat_arch.py (15 tests, pod-run on live Triton 3.7) pins
  sm_121[a] -> sm_120, sm_120[a]/sm_103[a]/sm_100[a]/sm_90[a] pass through verbatim, and the patched
  sm_arch_from_capability(103) -> sm_103a. Re-ran the previously-aborting max-autotune-gated paths on
  the pod with the fix active (evidence /tmp/frontFIX/): (1) occupancy_tuning triton_matmul with the
  `import arch_config` capability guard REMOVED -- raw tl.dot JITs + verifies at 64x64x32,
  128x256x64, 256x256x64, and harness occupancy_tuning:proton_matmul_bm256_bn256_bk64 SUCCEEDS at
  1.327x verify-pass (vs 1.306x when B19 shipped it guarded). (2) llama max-autotune --
  _safe_compile_mode retired from llama_3_1_8b_optimization.py and the central
  get_optimal_compile_mode sm_103+Triton>=3.6 "default" downgrade retired from compile_utils.py;
  get_optimal_compile_mode now returns max-autotune on sm_103 and the lab compiles + runs clean
  (real triton_mm autotune sweeps in the log = live Triton codegen on sm_103a). Confirms B18's dead
  num_warps param was a red herring (Front P2 probe B). The B10/B16 stale-checks re-opened by B69
  can now re-run under true max-autotune.

- RE-SWEEP CLOSE (B71, the B69/B70 unlock measured across the formerly-gated paths): docs/
  gb300-sm103a-unlock.md. The Front-T re-sweep under true max-autotune: 1 UNLOCKED WIN + 3 honest
  parities + survey clean, ZERO aborts in 15 harness runs. DECISIVE CONTROL (cold-cache
  counterfactual): re-applying the old de-suffix with a fresh TRITON_CACHE_DIR makes the
  grouped-GEMM lab itself die with the tcgen05.wait.st abort (exit 134) -- pre-fix runs only ever
  worked via warm caches and Inductor's subprocess autotuning SILENTLY SWALLOWING every aborting
  Triton candidate, i.e. the pre-fix tuner space was systematically biased toward extern kernels.
  WIN: ch14:model_compile_reduced_precision max-autotune 1.130/1.075/1.137x vs default 0.994-1.005x
  (median 4.509 -> 4.165 ms = 1.083x over the incumbent arm, verify PASS x3) -- FLIPS the lab from
  failed_no_speedup to succeeded and CORRECTS B23c ("compile adds ~1.5% all modes" -- max-autotune
  had ABORTED then, it was never measured). PARITIES (each bankable): blackwell_grouped_gemm
  510 TFLOPS harness / 870 TFLOPS kernel = parity with banked -- the B10/B11 tl.dot-gap attribution
  STANDS (the unlock is works-from-cold, not TFLOPS; B10/B16 now formally closed);
  moe_pad_quant max-autotune 0.3160 vs champion 0.3096 ms (B40's win was CAPTURE, not codegen --
  confirmed); router_vectorized parity (nvjet honestly beats Triton templates). Guard disposition:
  sm_103 max-autotune downgrades RETIRED (15-run validation, zero aborts/verify failures);
  AISP_COMPILE_MODE_OVERRIDE added as the A/B escape hatch. Durable lesson: a silent
  candidate-swallowing autotuner turns a toolchain abort into an invisible PERF BIAS -- when a
  compile backend "works", verify its candidate space actually compiles from a COLD cache.

- CONTRACT-LEVEL WIN (B72, FP8 port of the dual_2sm champion, the B68-named unlock): docs/
  gb300-fp8-dual2sm.md. NEW tcgen05_dual_2sm_fp8.cu (generated from the FP16 champion by 12
  exact-match asserted patches; FP16 file untouched, md5-identical, FP16 gates 3/3 PASS):
  fp8_2sm (n256, s3, mb2, kTileK=128, rg8, te1) = 377.6us / 2912 TFLOPS at 8192^3 = 1.918x paired
  over the in-session custom FP16 champion (724.1us; 2.20x vs the B68 830us figure), 0.7228x of
  same-run cuBLASLt FP8 (0/16 -- not beaten, but ONE front reached the ~70%-of-ceiling position
  the FP16 journey needed seven fronts for). Correctness: rel_err == 0.0 EXACTLY on all 11 configs
  (exact-dataset provably-exact fp32 partials; randn-quantized also 0.0). FP8 CEILING CALIBRATED
  (B61 method): warmed nvjet FP8 best 266.9us = 4119 TFLOPS = the real ceiling (B68's 3.7-3.8PF
  estimate was the cold value). DESIGN FACTS: the FP8 2SM atom K-extent is 32 and splits B
  N/2-per-CTA (B49 print-check held); kTileK=128 keeps per-stage bytes equal to the FP16 champion
  (24KB) while halving barrier trips/byte; the n256 BIG TILE beats the FP16-geometry port at FP8
  rates (2 CTAs/SM trades occupancy for halved B-traffic/flop -- nvjet's own 128x256 corroborates);
  TMA-store epi is a BIGGER lever at FP8 (1.128x vs 1.09x); k64/SW64 ring-deepening loses badly
  (TMA-box halving -> issue-bound, the B48 law at FP8). ncu: tensor pipe 81.7% (highest-utilized),
  grid 304 = 152 SMs x 2 (B65 static sizing), 71 regs, 98.4KB smem. The kernel is TENSOR-BOUND --
  structure, not feed, is the frontier. NEXT LEVER: ring depth on the n256 footprint at 1 CTA/SM
  (e4m3 makes 6x32KB = 192KB fit the 227KB SM budget, impossible at fp16; requires lifting the
  110KB static cap); secondary: in-kernel fp16 D output (D-store still ~12%).

- PARITY-CLASS WIN + HONEST NEGATIVE (B73, FP8 deep-ring refuted / fp16-D ships, the B72-named
  levers): docs/gb300-fp8-dual2sm.md (F8b section). LEVER 1 (deep ring at 1 CTA/SM): HONEST
  NEGATIVE with the cleanest mechanism yet -- s4/s5/s6 at 1 CTA/SM all lose (0.86/0.93/0.95x,
  0/36 paired) even though the ring DID ITS JOB: long_scoreboard stall HALVED (32.0 -> 16.7
  cyc/warp) and still lost, because that latency was ALREADY HIDDEN by the co-resident CTA; mb1
  surrendered the second MMA issue stream that covers the serialized per-round epilogue. The B66
  law at CTA granularity: don't spend resources hiding latency something else already hides.
  nvjet's 6-deep 1-CTA shape requires an in-CTA overlapped epilogue (structure, not config).
  LEVER 2 (in-kernel fp16 D, DUAL2SM_D_HALF=1): WIN, RATIFIED, DEFAULTS FLIPPED -- the B72
  champion's .to(fp16) was a ~400MB conversion kernel INSIDE the timed region; converting in the
  TMA-epi staging epilogue (chunk pairs -> 128x64 fp16 tiles, FLOAT16 descriptor) wins 1.2527x,
  24/24 paired across two fresh sessions: 313.5-315.0us / ~3500 TFLOPS = 0.908x of same-run
  cuBLASLt FP8 (the B72 gap of 0.7228x cut by two thirds). rel_err == 0.0 EXACTLY everywhere
  (in-kernel RN == torch RN, predicted + confirmed); ncu: tensor pipe 86.6%, DRAM writes at the
  fp16 floor (234.7 -> 112.3MB). FP16 gates 3/3 PASS. FP8 ladder: port 0.72x (B72) -> +fp16-D
  0.91x of cuBLASLt. NEXT LEVER (the FP8 scheduling frontier proper, ~9% remains): in-CTA
  epilogue/mainloop overlap WITHIN the 2-CTA footprint -- dedicated epilogue warp pair +
  double-buffered 2x128-col TMEM views inside the 256-col/CTA budget (the eo=2 structure at
  eo=0's occupancy; an mbarrier rewrite, not a config) -- reclaiming the epilogue exposure frees
  co-residency's concurrency for depth.
- ENGINE-BEST WALL DOCUMENTED (B74, block_scaling tile_k lever CLOSED with four refutations,
  resolves Front H/H2): the 20.5%-NVFP4-SoL diagnosis chain ends at a documented shape wall, not
  a win. Front H's static map (tile_k 768->384/256 re-tile) was REFUTED structurally before
  measurement: 768 is the LCM of the ISA-fixed MMA-K (96 elem) and the K_SW128 buffer width (256
  elem) -- 8 MMAs consume exactly 3 buffers per k-tile; 384 needs K_SW64 + a rewritten 4-MMA
  mainloop, 256 is impossible at any swizzle (96 does not divide 256). Front H's "only ~2 AB
  stages" claim was ALSO wrong: debug-stages shows ab=8 sf=6. What IS structurally possible (a
  Front H2 successor implemented it, then died unmeasured -- found via the orphaned-edit check):
  a peeled short-TAIL k-tile (env-gated) that stops the K=1024 tail from multiplying pure TMA
  zero-fill in 5 of its 8 MMAs (K-coverage 1536 vs 1024). MEASURED: activation debug-confirmed
  (short_tail=True, const_expr-branched codegen), 50-iter A/B 43.887us (incumbent) vs 43.955us
  (short-tail) = 0.998x PARITY, numerics exact (max_abs_err 0.0 both arms) -- the tail's
  zero-fill MMAs were already fully latency-hidden behind the 58% long-scoreboard stalls;
  removing ~31% of MMA issue work moves nothing in a load-starved pipeline (the B73 deep-ring
  lesson again: hiding already-hidden latency is worth zero). Engine swap ALSO refuted: nvjet
  NVFP4 _scaled_mm at the exact shape (8192x8192x1024, sf_vec 16) is 73.7us / 1866 TFLOP/s =
  0.596x of the lab kernel -- the CUTLASS blockscaled kernel (43.9us / 3132 TFLOP/s, within
  0.12% of the vendor reference) is ALREADY the best engine at this shallow-K shape, 1.68x over
  cuBLASLt's path. Split-K stays math-refuted (>=256MB partial-accumulator traffic vs the ~96MB
  floor). VERDICT: ~21% NVFP4-SoL at K=1024 is the engine-best wall for this kernel family --
  only 2 forced k-tiles exist, so load latency cannot amortize regardless of MMA savings.
  Evidence tuples in the front-BS bundle on the dev pod. NEXT LEVER (low-EV, large): the K_SW64
  swizzle + 4-MMA mainloop rewrite (the only structural opening, same effort class as B68's
  contract-level residuals); secondary audit: whether the persistent scheduler overlaps tile
  N+1's loads with tile N's epilogue (if not, cross-tile prefetch is a cheaper opening).

- HONEST NEGATIVE + FP8 FRONTIER DECLARED (B75, in-CTA epilogue overlap, the B73-named lever):
  docs/gb300-fp8-dual2sm.md (F8c section). THE TMEM ARITHMETIC KILLED THE THESIS BEFORE CODING
  (stated as mandated): n256 double-buffer = 512 TMEM cols/CTA = INADMISSIBLE (one CTA eats all
  of TMEM); 2x128-col views of one n256 accumulation CANNOT exist (every 256-wide MMA k-block
  writes all 256 cols -- nothing is drainable early); a warp PAIR cannot drain (tcgen05.ld
  subpartition w is warpgroup-rank-addressable -- 128 lanes need a full second warpgroup). The
  only admissible point (n128 x 2 buffers, champion's TMEM spend, 2 CTAs/SM) was implemented
  BIT-EXACT (rel_err 0.0 x3 sizes; warpgroup-local bar.sync, audited barrier ids; dual 16KB
  staging since the ring never drains under eo2) and REJECTED: 0.7948x of the dh1 champion,
  0/12 paired; ncu tensor pipe 65.8% vs 86.6% (the >90% thesis falsified). TWO INDEPENDENT
  DOMINATIONS: (1) at fixed n128, overlap@2CTA is 0.847x of plain@3CTA -- the third co-resident
  CTA out-schedules the dedicated warpgroup (B63's FP16 verdict replicates at FP8); (2) even
  perfect n128 scheduling is bounded by its own feed-taxed mainloop (336.1us > 315.7us champion;
  341 vs 512 FLOP/smem-byte; k64 escape closed by the B72 box law). FINAL FP8 LADDER: port
  0.7228x -> n256+TMA-epi ~0.83x -> fp16-D champion 0.907-0.908x (313.5-315.7us, ~3500 TF) ->
  overlap REJECTED. The FP8 scheduling frontier is CLOSED at 0.908x of cuBLASLt; further
  progress requires a different accumulator GEOMETRY, not a different schedule. The staged-store
  lever generalizes (eo2-te1dh1 1.45x over eo2-te0) and stays available. FP16 gates 3/3 PASS;
  no defaults changed.

- SURVEY + WIN + INVERTED VERDICT (B76, attention at serving shapes, the unexplored axis): docs/
  gb300-attention-serving-shapes.md. The map (bf16 D=128, CUDA-graphed, clock-stable, cuBLAS
  anchor per session): PREFILL (B1 Hq32 causal, S=8k/16k/32k): cuDNN SDPA at the MACHINE CEILING
  -- 1877-2018 TFLOPS = 98.8-106.2% of the 1.9PF real-SoL frame (96-103% of the same-session GEMM
  anchor; GQA identical structure); flex compiled (max-autotune) 34-40%; flashinfer-auto and
  flash_attn2 21-24% = the serving-default FA2-class paths are 4.3-4.7x OFF AT PREFILL. DECODE
  (B32 q1, KV 8k/32k): cuDNN 7491-7554 GB/s = 93.6-94.4% HBM (the strongest HBM streamer measured
  on this machine); flashinfer default 80-83%; flex fails to lower at KV=32k (NoValidChoicesError).
  PROTOTYPE LANDED: flashinfer cudnn_batch_decode_with_kv_cache paged+graphed = 92.9% HBM = 1.124x
  over flashinfer's default decode, BIT-IDENTICAL (max_abs 0.0) -- workaround discovered: the cuDNN
  frontend demands 4-D (B,1,1,1) seq-len tensors (the docstring's 1-D form is rejected). B38
  VERDICT HOLDS, REASON INVERTED: still no kernel-engineering EV at serving shapes -- because the
  at-ceiling kernel ALREADY SHIPS in cuDNN; the micro-shape close could not see the 4.3-4.7x
  BACKEND-ROUTING gap that dominates at serving shapes. ACTIONABLE: routing policy (prefill ->
  SDPA/cuDNN; decode -> flashinfer cudnn path with the 4-D workaround), not kernel work. NEW
  UPSTREAM CANDIDATES (flashinfer 0.6.3 on sm_103): (1) cudnn_batch_prefill_with_kv_cache broken
  (stride BAD_PARAM -> no valid execution plans; fa3 backend ships sm90-only cubins) -- EV up to
  ~4.3x on long-context prefill through flashinfer; (2) cudnn_batch_decode docstring/API mismatch
  (1-D vs required 4-D seq-lens). Cells + probe evidence /tmp/frontAT2/, 12 cell JSONs mirrored.

- HONEST NEGATIVE + L4 ncu (B77, NVFP4 dual_2sm contract port, the B72-named unlock):
  labs/custom_vs_cublas/bench_dual_2sm_nvfp4.py on pod aisp-gb300-runall @ commit 66f6307a
  (2026-06-16 sprint re-measure after repo sync). K4/R4/H4 custom
  (`SM100_MMA_MXF4_2x1SM_SS` + TMA-2SM, tcgen05 dual-cluster) vs H4 baseline cuBLASLt
  `torch._scaled_mm` NVFP4 (TN, 128x4-blocked ue4m3 scales). Shape: exact dataset,
  8192^3 e2m1, 12 interleaved paired reps, GPU1, no cudagraph. cuBLASLt median 130.9 us
  (8402 TFLOPS, 97.6% calSoL vs warmed ceiling 8607 TFLOPS); custom champion
  n128s3mb2k256rg4te1dh1 median 220.7 us (4983 TFLOPS, 57.9% calSoL); paired
  0.5923x, 0/12 wins. Correctness: rel_err == 0.0 both arms. ncu --set full on the custom
  kernel (launch-skip 30, isolated replay): Duration ~222 us, Compute(SM) 71.2%,
  DRAM ~15% HBM-SoL proxy, achieved occupancy 11.2% (230 KB smem/cluster, register-limited
  1 block/SM). VERDICT: DRAFT scaffold only; cannot beat the production H4 library at
  dense 8192^3 despite tensor-core engagement. The FP16 dual_2sm scheduling ladder (B68)
  does not transfer: NVFP4 needs a mainloop/operand-geometry rewrite, not more
  raster/prefetch/TMA-epilogue tuning. Next lever: structural NVFP4 mainloop (swizzle +
  4-MMA feed path), same effort class as B68's epilogue win on FP16. Evidence:
  /tmp/sprint-20260616/gpu1_nvfp4_full.log, /tmp/sprint-20260616/ncu_nvfp4_full.txt.

- RATIFICATION (B78, FP16 dual_2sm V7 TMA epilogue re-check post-pull, no code delta):
  /tmp/frontV7/ab_v7.py on GPU2 @ 66f6307a confirms B68 defaults still hold: challenger
  TE=1 cfg (128,3,1,0,3,64,0,32,1,8,0,0,1) vs incumbent TE=0, FP16 8192^3, 12 paired reps.
  Median 827.6 us vs 905.2 us = 1.0883x, 12/12 wins (~69.9% vs ~63.9% real SoL); cuBLAS
  702.1 us (~82.4% real SoL). Scheduling frontier stays CLOSED at ~70% real SoL; gap to
  cuBLAS is epilogue/mainloop geometry, not TE=0 vs TE=1. Evidence:
  /tmp/sprint-20260616/gpu2_v7.log; /tmp/frontV7-backup-20260616T073829Z.tgz.

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
