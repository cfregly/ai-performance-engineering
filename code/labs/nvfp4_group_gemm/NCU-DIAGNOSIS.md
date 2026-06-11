# NCU roofline diagnosis: custom tcgen05 NVFP4 grouped GEMM vs cuBLAS, shape g2_n3072_k4096 (GB300)

## Verdict (short)

VIABLE, but the premise is wrong and the headroom is not where it was assumed.

1. cuBLAS is NOT at the FP4 FLOP roofline on this shape. The genuine NVIDIA library FP4 kernel
   (`nvjet_sm103_...Avec16UE4M3_Bvec16UE4M3_TNT`) runs at 3.8 to 6.2% of the GB300 NVFP4 FLOP
   roofline and 11% of HBM. The shape (M = 192 / 320, tiny) is memory-latency and occupancy bound,
   far below every roofline. So this is not the "cuBLAS is at-ceiling, do not bother" dead-end.
2. The custom kernel already beats the only real library NVFP4 path at the batched workload.
   `torch._scaled_grouped_mm` REFUSES NVFP4 (FP8 / MXFP8 only), so the library forces 30 separate
   `_scaled_mm` launches for the 15-input grouped workload. The custom kernel fuses them into one
   launch and wins 3.1x wall-clock / 1.14x pure-GPU-time.
3. At a single isolated grouped call the custom kernel is 2.6x SLOWER than cuBLAS, and it has one
   specific, addressable, cuBLAS-absent bottleneck: a 46.6% CTA-barrier stall on top of 6% occupancy
   (shared-memory-capped at 12.5% theoretical). cuBLAS spends ~2% on barriers and is purely
   memory-latency (long-scoreboard) bound.

Net: do not BANK as a P4 at-ceiling dead-end. The reachable ceiling for this shape is the
memory-latency-bound HBM regime (about 34 to 38% of FP4 peak, capped by arithmetic intensity, not
100%), and the custom kernel sits at 15% HBM vs that. The concrete lever is occupancy + barrier
removal, not a faster MMA.

## Environment and provenance

- Node `<gb300-pod>`, GB300 (Blackwell Ultra, sm_103, cc 10.3), 152 SMs, 284 GB, driver
  580.159.03, CUDA 13.2. GPU 0 (idle, no foreign CUDA process). Driver = CUDA 13.x, so no CUPTI
  image/driver skew. ncu 2026.1.1.0, nvcc 13.2, torch 2.12.0a0+...nv26.05.
- Lab: `labs/nvfp4_group_gemm`, shape `COMPETITION_CASES[2]` = `g2_n3072_k4096`,
  M = (192, 320), N = 3072, K = 4096, g = 2.
- Scratch harnesses + ncu reports (not committed): `/tmp/ncu_diag/` in the pod
  (`harness_custom.py`, `harness_cublas.py`, `custom_fused.ncu-rep`, `custom_single.ncu-rep`,
  `cublas_m192.ncu-rep`, `cublas_m320.ncu-rep`).
- Ceilings from `perf-tune-report/configs/sol-ceilings.yaml` key `gb300_nvl72`:
  `nvfp4_dense_pflops = 15.0 PFLOPS`, `hbm3e_tbps = 8.0 TB/s`. Not inlined as guesses.

## K/R/H/P/A classification

| Dim | Custom kernel | cuBLAS baseline |
| --- | --- | --- |
| K (complexity) | K4 frontier (NVFP4 e2m1 block-scaled grouped GEMM, variable-M) | K4 (same op) |
| R (representation) | R4 instruction-level (hand-written CUDA C++ + inline PTX `tcgen05.mma.kind::mxf4nvf4.block_scale`, TMA, warp specialization) | R1 library (cuBLASLt `nvjet`) |
| H (hardware spec) | H4 (tcgen05 tensor cores, TMA, mbarrier, cluster) | H4 (tcgen05 nvjet, 2cta clusters) |
| P (perf vs SoTA) | measured below: 1.9 to 5.7% FP4-SoL | measured below: 3.8 to 6.2% FP4-SoL |
| A (automation) | A1 expert co-design (heavily hand-tuned, see lab WORKLOG) | A4 (autonomous library heuristic) |

This is the rare fair fight where the custom candidate and the baseline are BOTH K4 / H4. The custom
kernel is R4 (instruction-level) vs cuBLAS R1 (library). Per the kernel rubric, a candidate is only
structurally capped when it sits below the baseline's H level; here H matches, so the comparison is
legitimate and a win is not ruled out a priori.

## JIT build (custom kernel)

- Source `labs/nvfp4_group_gemm/custom_cuda_group_gemm_kernel.cu`, loaded via
  `load_custom_cuda_nvfp4_group_gemm()` -> `core.utils.extension_loader_template.load_cuda_extension`.
- nvcc flags: `--std=c++17 -O3 --use_fast_math -lineinfo`,
  `-gencode=arch=compute_100a,code=sm_100a`, `-gencode=arch=compute_100a,code=compute_100a`,
  `-gencode=arch=compute_103a,code=sm_103a`, `-gencode=arch=compute_103a,code=compute_103a`,
  `extra_ldflags=['-lcuda']` (Driver API for `cuTensorMapEncodeTiled`).
- Optimized `EXT_NAME = nvfp4_group_gemm_tcgen05_opt_u2_tp1_epi1_tm`; the extension name appends a
  compile-config hash so knob changes rebuild.
- Profiled `__global__` kernel: `void <unnamed>::nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>`
  (CLUSTER_DIM_X = 1, UNROLL_N = 2, BLOCK_N = 128, cta2 = false), 64 registers/thread, 128 threads/CTA.
- Confirms the workspace note: `BLOCK_M / BLOCK_N / KPACK_TILE` are scalar-path knobs and do not tune
  the tcgen05 path, which is why the env sweep was inert. The gap is structural, not a knob.
- Delivery for the profiled numbers: runtime JIT extension (not an infr image). The roofline shape is
  delivery-independent (same kernel, same shapes).

## Workload accounting and roofline ridge

Per grouped call (g = 2): group0 M = 192, group1 M = 320, both N = 3072, K = 4096.
- FLOPs = 2*M*N*K: group0 = 4.832 GFLOP, group1 = 8.053 GFLOP, g2 total = 12.885 GFLOP.
- Fused 15-input workload = 30 sub-problems = 193.27 GFLOP.
- FP4 roofline ridge = 15.0 PFLOPS / 8.0 TB/s = 1875 FLOP/byte.
- Measured arithmetic intensity (ncu bytes): custom fused 707 FLOP/byte, cuBLAS M=192 639 FLOP/byte.
  Both are well below the 1875 ridge, so this shape is on the MEMORY side of the roofline. The
  achievable ceiling is HBM-bound: at AI ~640 to 710, peak attainable is ~5.1 to 5.7 PFLOPS, i.e.
  about 34 to 38% of the FP4 FLOP peak even at 100% HBM. Chasing the 15 PFLOPS FP4 number on this
  shape is not physical.

## ncu --set full roofline (per kernel, --clock-control none, SM ~1.95 GHz)

Achieved TFLOPS = analytical FLOPs / ncu `gpu__time_duration.sum`. %FP4-SoL vs 15.0 PFLOPS,
%HBM-SoL vs 8.0 TB/s. Tensor pipe = `sm__pipe_tensor_cycles_active...pct_of_peak`.

| Kernel (config) | grid / waves | GPU dur | per-group | TFLOPS | %FP4-SoL | DRAM BW | %HBM-SoL | tensor-pipe | achieved occ | issue-active | top stall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Custom fused (30 sub) `tcgen05<1,2,128>` | 900 / 2.96 | 226.11 us | 7.54 us | 855 | 5.70% | 1.21 TB/s | 15.1% | 11.1% | 6.24% | 4.20% | barrier 46.6% |
| Custom single (g2) `tcgen05<1,2,128>` | 60 / 0.20 | 44.51 us | 22.26 us | 290 | 1.93% | 0.35 TB/s | 4.4% | 3.7% | 6.29% | 4.10% | barrier 46% |
| cuBLAS M=192 `nvjet 128x128 2x2 2cta` | 48 / 0.32 | 8.54 us | 8.54 us | 565 | 3.77% | 0.88 TB/s | 11.0% | 7.8% | 8.45% | 9.12% | long-scoreboard 58% |
| cuBLAS M=320 `nvjet 128x128 2x4 2cta` | 96 / 0.63 | 8.61 us | 8.61 us | 936 | 6.24% | 0.91 TB/s | 11.4% | 15.0% | 9.57% | 8.53% | long-scoreboard 55% |
| cuBLAS g2 (sum of two launches) | n/a | 17.15 us | 8.58 us | 751 | 5.01% | ~0.90 TB/s | ~11.2% | n/a | n/a | n/a | long-scoreboard |

Wall-clock (cuda-event, includes launch overhead), for context only:
- Custom fused, 1 launch / 30 sub-problems: 0.2235 ms = 7.45 us/group.
- Custom single (no fusion), 1 input: 0.0858 ms = 42.9 us/group (launch overhead dominates at N=1).
- cuBLAS 30x `_scaled_mm` (15 inputs x 2 groups): 0.701 ms = 23.4 us/group.

Top warp-stall reasons (cycles per issued instruction, ncu `smsp__average_warps_issue_stalled_*`):
- Custom (fused and single): barrier 11.1 (46.6% of the 23.8-cycle issue interval), long-scoreboard
  5.2 to 5.4, wait 2.9, lg-throttle 2.0. Barrier-sync dominated.
- cuBLAS (M=192 and M=320): long-scoreboard 9.5 to 9.7 (55 to 58%), no-instruction 1.8 to 2.6,
  wait 1.6, short-scoreboard 1.1 to 1.3, barrier ~0.3. Pure memory-latency bound, almost no barriers.

## Bottleneck diagnosis

Neither kernel is compute-bound or bandwidth-bound. Both are latency / occupancy bound because M is
tiny, so there is too little work per tile to fill 152 SMs or to hide HBM latency.

Custom kernel:
- Occupancy 6.24% achieved, 12.5% theoretical. The binding limit is shared memory (ncu Block Limit
  Shared Mem = 2 CTAs/SM); registers (64) and warps are not the cap. So only 2 CTAs (8 warps) fit per
  SM, and only ~4 warps are actually active.
- With so few warps and a warp-specialized producer/consumer/epilogue split, the scheduler has almost
  nothing to issue while a warp waits at a CTA `bar.sync`: 95.8% of cycles have no eligible warp,
  issue-active is 4.2%, and the barrier stall is 46.6% of the issue interval. This is the one thing
  cuBLAS does not pay.
- Fusion is the kernel's real strength and it works: the fused launch reaches 900 CTAs / 2.96 waves
  and 15.1% HBM, vs the single call at 60 CTAs / 0.20 waves and 4.4% HBM. More concurrency = more HBM
  utilization, exactly the right direction for a latency-bound shape.

cuBLAS kernel:
- `nvjet` is lean: 255 registers/thread, no CTA-barrier tax, software-pipelined, purely
  long-scoreboard (memory-latency) bound. But it cannot batch: each call is one tiny GEMM at
  0.32 to 0.63 waves, work-starved (cannot fill the GPU with a single M=192/320 GEMM), and there is no
  grouped NVFP4 API, so the 15-input workload costs 30 launches.

Why the stated "cuBLAS 8% faster (0.92x)" did not reproduce: the committed
`baseline_nvfp4_group_gemm.py` imports the SAME custom kernel with a different config (a config-A/B
self-comparison), not cuBLAS. The genuine library FP4 path measured here (`torch._scaled_mm` ->
`nvjet_sm103`) gives a different result: cuBLAS is ~3x slower than the fused custom kernel on the
batched workload, and ~2.6x faster only on a single isolated grouped call.

## Verdict: (b) VIABLE, with the headroom reframed

It is not (a). Option (a) requires cuBLAS to be near the FP4 roofline; it is at 3.8 to 6.2% FP4-SoL
and 11% HBM-SoL, nowhere near any ceiling. And the custom kernel already beats the only real library
NVFP4 option at the batched workload (no grouped NVFP4 kernel exists in cuBLAS).

It is (b): the custom kernel is well below the (memory-latency-bound) ceiling with a specific,
cuBLAS-absent, addressable bottleneck.

Concrete next levers, ranked, each with roofline evidence:
1. Raise occupancy by cutting per-CTA shared memory. Occupancy is smem-capped at 12.5% theoretical
   (Block Limit Shared Mem = 2). Fewer pipeline stages or a smaller tile that drops the smem-per-CTA
   below the 1/3-SM threshold would allow >=3 CTAs/SM, giving the scheduler more warps to hide both
   the barrier waits and the long-scoreboard (memory) latency. Evidence: fused vs single shows HBM
   rises 4.4% -> 15.1% purely from more concurrent CTAs; the DRAM-bound floor for the fused workload
   is ~34 us vs the measured 226 us, so there is ~6.6x of latency-hiding headroom to the HBM ceiling
   that is currently blocked by occupancy, not by bytes or FLOPs.
2. Remove the CTA-barrier handoff. The 46.6% barrier stall is the producer/consumer/epilogue
   `bar.sync`. cuBLAS pays ~2% here. Replacing the bar.sync stage handoff with mbarrier-only async
   pipelining (so warps never block at a CTA barrier) directly attacks the single largest stall
   bucket. This couples with lever 1: more resident warps make the remaining barrier waits cheaper.
3. For the single-grouped-call (non-batched) regime only: smaller tiles or split-K to exceed one
   wave (currently 0.20 waves / 60 CTAs). This is the regime where cuBLAS wins 2.6x; it matters only
   if the deployment ever issues one isolated g2 call rather than a stream.

Expected unlock (bounded, not a promise): the target is the memory-latency-bound HBM ceiling for this
AI (~34 to 38% of FP4 peak), not the 15 PFLOPS FP4 number. Closing occupancy + barrier should move the
fused workload from 15% toward the 20 to 30% HBM-SoL band and extend the existing batched-workload win
over cuBLAS; it will not make a single tiny g2 call hit the FLOP roofline, because the arithmetic
intensity forbids it.

Do not BANK. The right framing for leadership: for a grouped NVFP4 workload there is no library
kernel, the custom fused kernel is already the fastest available option, and it has a measured
occupancy + barrier bottleneck (not a FLOP wall) with ~6.6x of HBM headroom left.

## Methodology notes

- ncu hygiene: profiled from start with `--launch-skip N --launch-count 1`, scoped with
  `--kernel-name regex:...` so the curand input-generation kernels were excluded (an unscoped
  `--launch-skip` first landed on a `random_from_to_kernel`, the classic capture-hygiene trap). Env
  was set on the ncu process, never via an `env VAR=v` wrapper after `--`. Every report was validated
  non-empty (1.5 to 8.8 MB, 41 passes, kernel count > 0) before trusting any number.
- `--set full` (not `--set basic`): %FP4-SoL is achieved-TFLOPS-vs-FLOP-ceiling, and the bottleneck
  is read from achieved DRAM%, tensor-pipe%, occupancy, and warp-stall reasons, not from SM-busy%.
- cuBLAS path is the genuine FP4 tensor-core kernel (`Avec16UE4M3_Bvec16UE4M3`); scale values are
  synthetic (perf is layout/value-independent, and the correct scale tensor SHAPE selects the real
  nvjet kernel).
