# P04 standalone feature and GEMM corrections

Source/host gates pass; CUDA compilation, extension import on the pinned CUDA stack, device execution, sanitizers and performance remain **HOLD**. No GPU or nvcc was available or used. These records cover W1-019/020/027/065/069/070/109/116/117 without changing the original 128-finding inventory.

- W1-019/020/109: the legacy `test_tma` target now identifies its ordinary scalar/shared-memory work; `test_all_features` identifies cluster synchronization, FP8 storage and scalar arithmetic. Both use legal 32x32 blocks, reject launch/runtime/nonfinite verification failures before publishing throughput, and label minimum-traffic rates as estimates. The multi-buffer arrays honor their template stage count. Initial writes are published before reads and every reused cluster buffer waits for all readers before overwrite. Cluster grids are padded to the fixed 2x2 shape; padded CTAs still participate.
- W1-027: the fused epilogue broadcasts global column bias through the same CuTe output partition before SiLU. Same-device/positive-size/floating-bias checks and noncontiguous conversion remain explicit. The non-fused register copy no longer depends on reading a zero-weight output fragment.
- W1-065: the real cluster attribute is queried under a CUDA runtime version guard; failed queries return false. TMA is identified as a Hopper-or-newer feature. The input device guard precedes capability queries. Related CUDA attribute/launch failures now propagate; an unused dynamic-shared-memory request for a static-storage TMA kernel was removed.
- W1-069/070/117: stage descriptions require explicit empty-barrier reuse, identify stages 6/11 as ordinary TMA cluster loads, and remove fixed speedup, roofline and gap-attribution claims. The already-repaired `tcgen05_cluster.cu` header was read and hashed, not edited. `run_lab.py` refuses backend substitution, compares every output element before and after timing, fails verification/CLI status when a selected backend fails, uses the actual stream for the native launcher and computes ideal FLOPs/byte from actual dimensions and input/output element sizes. For 4096 cubed, FP16 A/B/C gives 1365.333 FLOPs/byte and FP16 A/B plus FP32 C gives 1024; these figures do not establish a measured bottleneck.
- W1-116: the unreferenced, noncompiling `experimental/tcgen05_multicast.cu` was retired in favor of the existing `experimental/tcgen05_tma_multicast.cu`. The latter stayed frozen. Exact originals of the retired file and earlier feature/lab claims are retained under `originals/`; old claims are historical, not qualified results.

Adjacent source discoveries, separate from those nine original IDs:

1. The basic tcgen05 kernel allowed warp 0 to lap other parity-wait observers. An independent source review identified this; a CTA rendezvous now follows every MMA wait before either barrier can be reused. The reviewer rechecked this insertion and found no further actionable source concern in the three reviewed kernels.
2. The lab loader treated unknown/future capabilities and CC12 as sm100a, and its cache key omitted architecture and Torch ABI. It now allows only exact CC10.0/sm100a and CC10.3/sm103a, and separates cached modules by complete CUDA flags, Torch version, Torch CUDA version and C++ ABI. This does not claim full transitive toolchain cache provenance.
3. The feature Makefile masked executable failures with `|| true`. Its runner now stops with the real failure; a real temporary executable returning 23 exercises this path without CUDA.
4. The legacy lab runner replaced unavailable backends with library GEMM and reported their timing under the requested stage. This is now an explicit failed/unavailable result with no throughput. CPU controls exercise real exception propagation and prevent wrong output from reaching timing.

## Validation and retained attempts

`validation-receipts.json` and `source-manifest.json` contain exact commands, hashes and acceptance boundaries. Original red source controls are `before.txt` (10 failures). `after-initial.txt` (10 passes), `after-expanded.txt` (37 passes), `combined-host-tests.txt` (101 passes), loader red/green controls, the whitespace correction attempt, and the final combined run (113 passes in 12.61 seconds) are all retained. Diagnostic “graph mismatch” messages in passing combined tests are intentional negative controls from the prior P03 validator tests, not CUDA failures.

Two real preflight rounds are retained. Each round ran the standalone gate for SM100a and Hopper, the bias extension gate for SM100a, the genuine TMA extension gate for Hopper, and the lab verify CLI. All exited 3: missing nvcc or CUDA. These are HOLD records, not tests of device code. No dependency was installed, shared environment changed, or Git mutation performed.

## Prepared device gates

Run from the repository root in a compatible, separately allocated CUDA environment. Use a fresh output directory for every attempt. The supplied `CUDA_VISIBLE_DEVICES` must identify the allocated target; devices are not pooled.

```sh
python code/tests/cuda/run_feature_cuda_validation.py --arch sm_100a --output-dir /tmp/p04-features-sm100a-unique
python code/tests/cuda/run_feature_extensions_validation.py --case bias_silu --arch sm_100a --cutlass-include /path/to/pinned/cutlass/include --output-dir /tmp/p04-bias-sm100a-unique
python code/tests/cuda/run_feature_extensions_validation.py --case grace_tma --arch sm_90 --output-dir /tmp/p04-tma-sm90-unique
```

The standalone runner compiles the production translation units and invokes their actual kernels: 28 ordinary scalar and 21 FP8/cluster cases, each with five launches, full output comparison and output canaries. Seven asymmetric/tail shapes exercise short K, stage reuse, cluster padding and buffer counts 1/2/4. Fixed dyadic operands make exact FP32 comparison appropriate for this bounded input family only.

The extension runner builds/imports the production source in a separate process: 8 bias+SiLU comparisons, 4 non-fused comparisons and 9 invalid-input controls, or 5 genuine TMA comparisons and 3 invalid-input controls. Bias values vary by global column, and a zero input row isolates every column's epilogue. It covers FP16/FP32 bias, noncontiguous views, asymmetric whole-tile GEMMs, K up to 1088 and nondefault streams. SiLU comparisons allow FP16 rounding error (`atol=1e-4, rtol=1e-3`); separate CPU controls show that missing and shifted bias fail. The TMA cases cover ragged M/N and K tails with valid 16-byte tensor-map row strides. Multi-device rejection controls require an explicit `--multi-device-controls` flag and two separately allocated visible devices; otherwise they are reported HOLD.

Both runners require the exact requested compute capability, capture source/compiler/binary provenance and bounded child output, enforce 300-second compile/run timeouts including descendant cleanup, and default to actual memcheck/racecheck/synccheck. `--sanitizers none` reports `PASS_WITHOUT_SANITIZERS`, never sanitizer acceptance. Missing tooling/devices return HOLD. The bias implementation accepts only whole 128x256x64 tiles; unsupported tails are tested as rejections, not represented as supported.

Official references checked on 2026-08-30:

- [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html): Hopper introduced TMA and thread-block clusters.
- [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html): cluster dimensions constrain grid divisibility; synchronization must cover the threads sharing storage.
- [PTX barrier wait contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait): waits may observe only the current incomplete or immediately previous completed phase.
- [CUDA runtime attribute enumeration](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html): `cudaDevAttrClusterLaunch` is a real enum attribute, not a preprocessor feature macro.
- [NVIDIA compute-capability table](https://developer.nvidia.com/cuda/gpus) and [Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html): CC10.0/10.3 differ from CC12; architecture-specific `a` targets are not forward-compatible.

README generator and generated lab README changes are parent-owned and are not included in this source receipt.
