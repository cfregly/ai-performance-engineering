# Wave 1 P08 numerical contracts

Source and CPU checks cover W1-030/031/037/038/077/086/087/119/122/123. **No CUDA numerical, compiler, memory, or performance result is claimed.** This host has torch 2.8.0 CPU and no CUDA/nvcc. The revised KV and Ozaki workflows deliberately refuse accepted quantized results until a reviewed accuracy policy is supplied; policy configuration alone is not qualification.

| Finding | Source change and observed evidence | Remaining gate |
| --- | --- | --- |
| W1-030 | Full K/V reference and payload, true model/input signatures; zero/tail corruption/NaN/alias/shape negative controls exercise the production comparator. Removed `(1,10)` tolerance. | Target BF16-reference accuracy calibration for FP8 and NVFP4, policy review, full workload run. |
| W1-031 | Supported default recipe construction in the KV lab and chapter 19. Official TE 2.18 constructor and local vendored constructor executed on CPU. | Actual TE/CUDA setup/training. **The audit's TypeError claim was not reproduced**: pydantic 2.12.5 accepts but ignores the three unsupported arguments. |
| W1-037 | Independent cloned submission/reference inputs and a snapshot taken before reference execution. Actual lab checker rejects zero/corrupt/NaN/input-mutating/shared-workspace submissions. Old shared-C check accepts the zero negative control. | Official CUDA evaluator and all competition shapes. |
| W1-038 | Module-level `sys` import. Both real CLI entry modes and a real child process pass. The child is a local protocol fixture, not an official benchmark. | Official upstream evaluator on CUDA. |
| W1-077 | Metrics derive bytes/bits/ratio from allocated tensors; BF16 cache ratio is 1.0. Optimization goal and generated docs describe compute speed. | Live GPU allocator/latency measurements. |
| W1-086 | All five baselines use independent E2M1 decoding, original scales, FP64 matmul and FP16 output. All logical requests retain independent references, including fused-call compression. Actual production payload rejects corruption outside the returned first request. | Recompile and run every custom route on SM100, including all shapes, graph/eager modes and multiple seeds. |
| W1-087 | Actual CUDA-used host C++ comparator checks full arrays, finite values, overlap, relative L2 and normalized maximum error. CPU-compiled negative controls reject zeros, cancelling corruption, NaNs, aliasing and missing bounds. Measurement-only mode exits 2 without accepted timing/checksum. | CUDA compile, native/emulated runs, retained-bit/input-scale calibration and reviewed bounds. |
| W1-119 | Corrected MXFP8 claim to per-tensor delayed-scaling FP8. | Documentation/source correction complete; runtime remains subject to W1-030. |
| W1-122 | Bounded cache retains exact source identities and versions; versionless inference tensors are repacked. Real CPU packing tests cover mutation, distinct inputs, source lifetime, eviction and inference tensors. Original cache drops the source owner. | CUDA allocator reuse/concurrent-stream kernel validation and performance. No partial-content hash was introduced. |
| W1-123 | Replaced nonexistent CUTLASS-extension entry with actual CUDA/common/input/reference files; parent mirrored generators. | Documentation correction complete. |

Final verification from `code/`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_numerics_regressions.py tests/test_ozaki_scheme_lab.py
/opt/miniconda3/bin/python -m core.scripts.linting.check_benchmarks labs/kv_cache_compression labs/nvfp4_group_gemm labs/ozaki_scheme
```

Result: **41 passed, 1 skipped in 6.66 s**; the skipped case is the real CUDA custom grouped kernel. Static linter: 15 entrypoints, zero errors/warnings. Python syntax compilation: 35 files. The C++ test compiles and executes the same `accuracy.h` included by the CUDA binary; it does not compile the CUDA translation unit. The first diagnostic test run had one fixture keyword mistake (33 passed/1 failed/1 skipped); it is retained, corrected, and is not described as a product regression.

`reproduce_cpu.py` reads the original Git objects without modifying Git, reproduces CPU-visible old-verifier failures, and records source/lifetime facts. `original-reproductions.json` preserves the results. `official-te218-constructor.json` records the downloaded official source SHA256, constructor signature, successful default construction, and ignored legacy arguments. The unversioned vendored recipe differs from the official 2.18 source; neither was presented as an installed CUDA TE runtime.

Calibration-only CPU invocation failed clearly with exit 1 and produced no output file, as expected. Exact target commands and policy schemas are in the KV/Ozaki READMEs. The full GPU gate must record compiler/driver/CUDA/TE versions, source hashes, hardware/clocks, per-output errors, seeds, negative controls and repeated timings. Historical expectation JSON files were preserved unchanged and are not accepted evidence for the new baselines/verifiers.

Adjacent source prerequisites: KV setup resolves the CUDA device explicitly and leaves RNG seeding to the harness; master weights remain BF16 for the independent reference. Only delayed-scaling FP8 uses calibration history. Group graph-capture failures no longer silently become eager runs. Chapter 19 warmup docs now state that its warmup performs optimizer steps. The existing parent-owned `b200` Makefile recursion change was preserved; this slice only adds `accuracy.h` dependencies.

Primary sources checked: [Transformer Engine 2.18 recipe source](https://raw.githubusercontent.com/NVIDIA/TransformerEngine/v2.18/transformer_engine/common/recipe/__init__.py), [TE 2.18 common API](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/api/common.html), and [cuBLAS floating-point emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation). No paid model calls or private prompt transmission occurred.

The receipt records final source/evidence hashes. It excludes parent-owned README generators, shared harness changes, and later packages.
