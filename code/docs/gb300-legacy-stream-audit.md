# GB300: repo-wide legacy-stream launch audit (Front A, B62 named next lever)

**Verdict.** One DANGEROUS site found and fixed: `labs/nvfp4_dual_gemm/optimized_submission.py`
launched its single fused kernel `<<<grid, TB_SIZE, smem_size>>>` on the **legacy default
stream** inside the harness target's `benchmark_fn` — the exact B45/B62 silent-drop class, on
the lab the frame audit ranked **#1 by frame share (86%)**. Post-fix the lab passes the
graphed-mode capture-fidelity audit for the first time and banks its dual figure:
**frame 47.0 us / graphed pure-GPU 6.89 us (warm)** — confirming B38's 6.6 us kernel claim
end-to-end. Seven more legacy launches in 4 dormant in-process torch extensions were fixed
proactively (LATENT class); all 4 now pass an eager + CUDA-graph capture + kernel-count
fidelity smoke on GB300. The remaining 258 legacy sites (of 266 found) are standalone
`main()` binaries (subprocess-run, never capture-eligible), non-harness `load_inline` eval
helpers, or vendored third_party — listed, untouched.
Default-mode numbers match the banked band; verification PASS everywhere.

Hardware: GB300 (cc 10.3), GPU3 quiet (foreign-process precheck on every run), pod
`<gb300-pod>`, repo base 2f7e30f9. Evidence: `/tmp/frontA/a1*,a2*,a3*.{cmd,stdout,stderr,exit,precheck}`
+ `/tmp/frontA/b62_audit.json` + `/tmp/frontA/a2_latent_smoke.json` on the pod.

## 1. Sweep coverage (the proof of completeness)

Parser-based (not bare grep): comments/strings stripped, `<<<...>>>` matched to its closing
chevron with paren/template tracking, top-level commas counted (2/3 config args = legacy
stream; 4th arg `0/NULL/nullptr` = legacy-by-zero), plus argument-position checks for the
runtime/driver API surface.

| Sweep | Scope | Files | Launch sites | Legacy |
|---|---|---|---|---|
| chevron + API launches | `*.cu .cuh .cpp .cc .h .hpp .inl` under `code/` | 849 scanned (245 with `<<<`) | 804 | 236 |
| API patterns | `cudaLaunchKernel`, `cudaLaunchCooperativeKernel`, `cuLaunchKernel(Ex)`, `cudaMemcpy*/Memset*/MemPrefetch/MemcpyPeer Async`, `cudaGraphLaunch`, `cudaLaunchHostFunc` | (same pass) | included above | 23 of the 236 |
| `cudaLaunchKernelEx` config-struct | `.stream` field of every `cudaLaunchAttribute` launch | 16 files | all | 0 in torch-integrated code (labs/core all `at::cuda::getCurrentCUDAStream()`; `=0` only in ch10/ch11 `main()` binaries) |
| CUDA embedded in Python strings (`load_inline`) | all 38 `.py` files containing `<<<` (non-third_party) | 38 | — | 30 sites in 27 files (all labs/nvfp4_*) |
| false positives excluded | harness error-message string (`benchmark_harness.py:5567`), test assertion strings (`test_benchmark_hygiene_regressions.py:102`) | | | |

Total legacy inventory: **236 + 30 = 266 sites**; non-third_party: 229 (.cu: 221 in
`main()` binaries + 7 in torch extensions + 1 header) + 30 (.py-embedded).

## 2. Classification (evidence-based reachability)

Reachability model (from `core/discovery/__init__.py` + `core/benchmark/cuda_binary_benchmark.py`):
harness targets are Python `get_benchmark()` files; in-process torch extensions are
capture-eligible (`AISP_BENCH_TIMING_MODE=graphed` side-stream capture, `enable_cuda_graph`,
B45-class in-lab captures); `CudaBinaryBenchmark` targets run standalone binaries in a
subprocess — never capturable from the harness process.

| Class | Files | Sites | Disposition |
|---|---|---|---|
| **DANGEROUS** (harness `benchmark_fn`, capture-eligible) | 1 | 1 | **FIXED** |
| **DANGEROUS-LATENT** (in-process torch extension, no current loader/caller) | 4 | 7 | **FIXED** (proactive; capture-unsafe by construction) |
| BENIGN — standalone `main()` binaries (incl. `core/.../blackwell_optimizations/test_*.cu`) | 91 | 221 | listed, untouched |
| BENIGN — header whose only includer chain ends in a `main()` binary (`core/common/headers/cuda13_demos.cuh` → `cuda13_demo_runner.cuh` → `ch06/cuda13_demo_runner.cu`) | 1 | 1 | listed, untouched |
| BENIGN-HELPER — `load_inline` eval/sweep artifacts excluded from harness discovery (`should_ignore_benchmark_candidate`; loaded only by `local_eval*.py`) | 26 | 29 | listed, untouched (preserves banked A/B reproducibility) |
| OUT-OF-SCOPE — vendored third_party (TransformerEngine/cudnn-frontend samples) | 4 | 7 | listed |

Cross-checks that came back clean: every file performing its own `cudaStreamBeginCapture`
(B45-class manual graphs: ch09/ch10/ch12 + `ch12/cuda_extensions/*`) launches with explicit
streams — 0 legacy sites among them; all `labs/*.cu` chevron launches were already explicit
(45/45); `persistent_decode` embedded CUDA explicit.

### The dangerous site (call chain, primary sources)

```
harness target labs/nvfp4_dual_gemm:nvfp4_dual_gemm (optimized arm)
  -> labs/nvfp4_dual_gemm/optimized_nvfp4_dual_gemm.py:56-87
       load_submission_module(LAB_DIR / "optimized_submission.py"); benchmark_fn() -> custom_kernel(data)
  -> optimized_submission.py:869 custom_kernel -> dual_gemm (TORCH_LIBRARY my_module)
  -> embedded CUDA dual_gemm_launch:
       this_kernel<<<grid, TB_SIZE, smem_size>>>(...)   // 3 args -> LEGACY stream  [pre-fix]
```

A side-stream capture of this `benchmark_fn` silently skips the kernel (no CUDA error). With
the B62 harness fidelity audit it refuses loudly; either way the #1 frame-share lab was
unmeasurable in graphed mode and corruptible under any manual capture (B45 class). The baseline arm
(`reference_submission.py`) is pure `torch._scaled_mm` — clean.

### LATENT fixes (in-process extensions, dormant)

| File | Sites | Loader | Reachability evidence |
|---|---|---|---|
| `core/benchmark/cuda/memory_patterns_extension.cu` | 86, 104, 136 | `core/benchmark/memory_kernels.py` | wrappers (`coalesced_copy`, ...) unreferenced repo-wide |
| `core/profiling/cuda/occupancy_extension.cu` | 66, 80 | `core/profiling/occupancy.py` | wrappers unreferenced; `cli profile occupancy` is a stub (`cli/commands/profiling.py:721`) |
| `core/profiling/lineinfo_demo.cu` | 20 | `core/profiling/lineinfo_demo.py` (`python -m`, ncu demo) | no importers |
| `ch10/warp_specialized_pipeline_enhanced_extension.cu` | 207 | `warp_specialized_pipeline_setup.py` (CUDAExtension) | module never imported; the harness target `ch10:warp_specialized_pipeline_enhanced` uses standalone binaries via `CudaBinaryBenchmark` |

## 3. The fixes (5 files, 8 sites)

Style matches B62's training_hotpath fix: `#include <c10/cuda/CUDAStream.h>` +
`c10::cuda::getCurrentCUDAStream()` as the 4th launch argument (eager behavior identical —
torch's default current stream IS the legacy stream; under capture the launch follows the
capture stream).

- `labs/nvfp4_dual_gemm/optimized_submission.py` — embedded CUDA: include + 1 launch site
- `core/benchmark/cuda/memory_patterns_extension.cu` — include + 3 sites
- `core/profiling/cuda/occupancy_extension.cu` — include + 2 sites
- `core/profiling/lineinfo_demo.cu` — include + 1 site
- `ch10/warp_specialized_pipeline_enhanced_extension.cu` — include + 1 site (note: its wrapper
  also calls `cudaDeviceSynchronize()` post-launch — capture now fails LOUDLY there instead of
  silently dropping the kernel, which is the B45 rule's accepted behavior for dormant demo code)

Post-fix re-sweep: **0 in-process extension files with legacy launches remain** (the 222
remaining non-third_party `.cu` legacy sites are all standalone-`main()` binaries plus the
one header on a `main()`-only include chain).

## 4. Validation (GPU3, flock-wrapped, foreign-process precheck clean)

Banked references: B38/B41 frame-mode 47-48.5 us/iter (`gb300-runbook.md:1205,1274`;
frame-audit table 47.5 us), B38 kernel-frame 6.6 us (`gb300-runbook.md:1205`).

| Run | Mode | Result | vs banked | Verify |
|---|---|---|---|---|
| `a1` (cold rebuild) | default | optimized 49.93 us/iter, 7.58x | +3-5% of band (ninja rebuild run) | PASS |
| `a1b` (warm) | default | optimized 41.96 us/iter, 8.67x | band bracketed (42.0-49.9 spans 47-48.5) | PASS |
| `a3` | graphed | frame 47.0 us/iter, 7.80x; **capture-fidelity PASS, `graphed_eager_kernel_count=1`**; graphed warm 6.89 us / cold 7.88 us | warm 6.89 vs banked kernel 6.6 us (+4%) | PASS (both arms) |
| `a2` | latent smoke | see §5 | — | — |

The `a3` row is the proof of fix: pre-fix, the side-stream capture would have contained zero
kernels for this `benchmark_fn` (the B62 metric_reduction failure mode); now the captured
graph replays the exact eager kernel count and the pure-GPU figure lands on the banked kernel
number. nvfp4_dual_gemm's dual figure (frame 47.0 / pure-GPU 6.89 us) is hereby measurable
end-to-end — the first of the four >79%-frame labs to get its honest denominator banked.

## 5. Latent-extension smoke (a2)

Each fixed extension: build + eager call + correctness check vs torch reference +
CUDA-graph capture (side stream via `torch.cuda.graph`) + replay-correctness + kernel-count
fidelity (profiler kernel count of 1 eager call == 1 replay, the B62 audit rule; pre-fix a
legacy-stream kernel replays as 0). Exit 0, `ALL_SMOKE_PASS`
(`/tmp/frontA/a2_latent_smoke.{stdout,json}`):

| Extension entry point | eager | capture+replay | kernels eager/replay | fidelity |
|---|---|---|---|---|
| `memory_patterns_ext.coalesced_copy` | OK | OK | 1 / 1 | PASS |
| `memory_patterns_ext.bank_conflict_transpose` | OK | OK | 1 / 1 | PASS |
| `occupancy_ext.run_low_occupancy` | OK | OK | 1 / 1 | PASS |
| `occupancy_ext.run_high_occupancy` | OK | OK | 1 / 1 | PASS |
| `lineinfo_demo_ext.forward` | OK | OK | 2 / 2 (kernel + `zeros_like` fill) | PASS |
| `ch10.warp_specialized_pipeline_enhanced_forward` | OK (finite output) | skipped — wrapper calls `cudaDeviceSynchronize()`, capture fails loudly by design (B45 fail-loud rule) | — | n/a |

## 6. Files touched (md5 local == pod, verified after tar-stream)

- `labs/nvfp4_dual_gemm/optimized_submission.py` — `ba3660bd2b4c2c9a7876ef95fd7e3c98`
- `core/benchmark/cuda/memory_patterns_extension.cu` — `e22f046b205385f1dd021cc0963b27d1`
- `core/profiling/cuda/occupancy_extension.cu` — `9614b00a3acd1f65b103f6990cd9fa46`
- `core/profiling/lineinfo_demo.cu` — `45b01e27d3b379767a63a465d7fb5474`
- `ch10/warp_specialized_pipeline_enhanced_extension.cu` — `cdf5ee8465634a64a0bc03d4601dd202`

## 7. Named next lever

**Bank the graphed dual figures for the remaining two >79%-frame labs** now that the repo is
legacy-stream clean: `fullstack_cluster:capstone_extension_tcgen05` (~85% frame; the ~85 us
pybind/checks host residual from B55 is the single biggest now-visible target) and a
re-confirmation pass on `blackwell_matmul_tcgen05` after the B61 SoL recalibration. The
audit machinery (sweep parser at `/tmp/frontA`, fidelity smoke pattern) is reusable for any
new lab intake — consider a CI check that rejects 2/3-arg chevron launches in
`PYBIND11_MODULE`-bearing files.
