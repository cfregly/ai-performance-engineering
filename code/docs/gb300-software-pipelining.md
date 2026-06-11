# GB300 software_pipelining lab: TMA bulk-copy pipeline (B38 lever)

**Verdict: WIN.** Kernel 83.0us -> 60.5us (1.37x kernel-level), 4850 -> 6653 GB/s
(60.6% -> 83.2% of the 8.0 TB/s HBM3e peak). Harness wall 144.0us -> 123.0/126.0us
(two reps), pair speedup 2.16x -> 2.60x/2.46x vs the untouched baseline arm,
**verification passed** on both reps (rtol 1e-4 / atol 2e-4 payload check plus the
5e-4 max-abs-diff reference check; max_abs_diff stayed 7.15e-7, bit-identical math).

- Hardware: NVIDIA GB300 (sm_103, 152 SMs, 276.6 GB HBM3e), GPU 3, pod `<gb300-pod>`
- Workload: fp32 triad-style tile transform, length 2^25, 403 MB moved per iteration
  (2 read streams + 1 write stream); bytes are the workload (B27b gaming check done:
  `benchmark_fn` runs the full-tensor kernel; bytes_per_iteration is registered metadata)
- Base commit: 2f7e30f9 + uncommitted GB300 fixes

## Why it was at 60.5% (ncu, before)

`optimized_tile_pipeline_kernel` (2-stage block-scoped `cuda::pipeline`, 1024-float
tiles, warp-chunked `memcpy_async`, scalar consumer loop):

| Metric | Before |
|---|---|
| Duration | 82.5-84.5 us |
| DRAM throughput | 55.6-56.9% |
| **SM issue-slots busy** | **74.3-74.6% (IPC 3.17/4)** |
| Achieved occupancy | 58.8% (75% theoretical, register-limited, 40 regs) |
| Executed insts/scheduler | 126K |

The kernel was **instruction-issue-bound, not DRAM-bound**: per 1024-element tile every
thread paid scalar shared-loads/global-stores with per-element bounds predicates, four
`cta.sync()` (all redundant with the pipeline's own barrier semantics), plus the
per-thread `cp.async` + `producer_acquire/commit/consumer_wait/release` machinery.

## What was done (3 shipped levers, in `software_pipelining_kernels.cu`)

1. **float4 + de-sync of the cuda::pipeline kernel** (kept as the sm_80..sm_89 fallback):
   128-bit vector consumer loop, `cuda::aligned_size_t<16>` on `memcpy_async`, removed
   the four redundant `cta.sync()` per tile. 83.0 -> 64.2us (78.4%). Regs 40 -> 32,
   occupancy 75 -> 100% theoretical, but SM issue still ~75% busy -> plateau.
2. **TMA bulk-copy warp-specialized pipeline** (new `optimized_tile_pipeline_tma_kernel`,
   dispatched on sm_90+): producer = warp 0 lane 0 issues one
   `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` per operand per
   tile against a tx-count mbarrier (`full_bar`); consumer warps spin on mbarrier parity,
   compute with float4, and signal a second mbarrier (`empty_bar`) for stage reuse.
   Per-thread copy/barrier instruction cost goes to ~zero (SM busy 75% -> 52-54%).
3. **Stage early-release + exact-cover consumers** (the two tuning wins inside the TMA
   kernel): consumers drain the staged tile into registers and arrive on `empty_bar`
   *before* compute/store, decoupling stage refill from the global-store drain; block
   shape 544 = 1 producer warp + 512 consumers so each consumer handles exactly one
   float4 pair per 2048-float tile (no ragged loop, no predicates).

Final config (swept): `kTmaStages=2`, `kTmaTileElems=2048` (8 KB/operand bulk copies),
`kTmaVecPerConsumer=1`, 544-thread blocks, 32.8 KB dynamic smem, grid = 456 (3
blocks/SM, waves = 1.0). Swept and rejected: stages 3/4/6/8 (deeper stages lose
blocks/warps and add wait phases: 75-79%), tiles 1024/4096, V=2/4, `__stcs`
streaming stores (82.5%), 4-stage cuda::pipeline (76%).

## After (ncu, final kernel)

| Metric | After |
|---|---|
| Duration (ncu replay) | 53.7-55.7 us |
| DRAM throughput | 80.8-83.3% |
| Compute (SM) throughput | 52-54% (was the 74% bottleneck) |
| Achieved occupancy | 77% (79.7% theoretical), 28 regs |
| Free-clock kernel-only | 60.5 us, 6653 GB/s, **83.2% of 8.0 TB/s** |

## SoL grounding: where the ceiling actually is

Calibration on the same GPU, same 403 MB 2R:1W mix: `torch.add` = 57.8-58.0us,
6946-6971 GB/s = **86.8-87.1%** — the practical triad ceiling on GB300 for a kernel
with *no* staging at all (pure `ld.global.v4 x2 / st.global.v4`). A 1R:1W `copy_`
lands lower (82.5%), i.e. write-mix efficiency, not implementation, sets these caps.
The staged-pipeline result (83.2%) is within 4.6% of the unstaged ceiling — that
residual is the price of the smem round-trip + mbarrier protocol the lab exists to
teach. The original "50.3us floor @ 8.0 TB/s" is not reachable by any kernel here.

## Exact diff

- `labs/software_pipelining/software_pipelining_kernels.cu`
  (md5 `a8cfcf000408c7053259f23dc0e99d7c`):
  - `optimized_tile_pipeline_kernel`: float4 consumer path, `aligned_size_t<16>`
    memcpy_async, redundant `cta.sync()`s removed (now the sm_80 fallback).
  - new: mbarrier PTX helpers (`mbar_init/arrive/arrive_expect_tx/wait_parity`,
    `tma_bulk_load`) guarded by `__CUDA_ARCH__ >= 900`, and
    `optimized_tile_pipeline_tma_kernel` (warp-specialized, 2-stage, early-release,
    exact-cover; partial tail tiles bypass smem and read global directly).
  - `launch_common`: sm_90+ dispatch to the TMA kernel with
    `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in and occupancy-derived grid.
- `labs/software_pipelining/optimized_tile_pipeline.py`
  (md5 `583457d4acaf4a935f3b2b2b4f9dca98`): notes string updated to describe the TMA
  design (op_name/label/stage-count metric unchanged; benchmark contract intact).
- Baseline kernel and all benchmark I/O/verification untouched.

## Named next lever

The remaining 4.6% gap to the `torch.add` ceiling is the smem round-trip itself; the
only way to claw it back while keeping the pipelined-staging story is **TMA stores**
(`cp.async.bulk.global.shared::cta` for the output tile + `commit_group/wait_group`
ring), which would also let consumers skip `st.global` drain stalls. Expected value
is small (<=3us) and it adds a third barrier protocol — flagged, not implemented.
A cheaper follow-up: ticket-based (atomic counter) tile assignment to remove the
~1/36 per-block tile-count raggedness at the kernel tail.
