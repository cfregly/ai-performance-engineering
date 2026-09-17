# Integration validation - 2026-09-16

The [machine-readable receipt](validation.json) records the B200 experiment,
native CUDA source hashes, and digests of the original result reports. The
vendored source remains byte-for-byte identical to the revision and 34 file
hashes in [upstream_manifest.json](upstream_manifest.json).

## B200 results

Both comparisons ran on one visible NVIDIA B200 (SM100), with PyTorch
2.9.1+cu130 and CUDA toolkit 13.0.88. The repository harness used strict validity,
20 requested iterations, 5 warmups, and the `deep_dive` profile. It locked SM
and memory clocks to 1965 and 3996 MHz and recorded application-clock telemetry.
Nsight Systems, Nsight Compute, and PyTorch profiling succeeded for both arms
of both comparisons. Input equivalence and full-output verification passed.

| Target | Baseline | Candidate | Ratio | Harness outcome |
| --- | ---: | ---: | ---: | --- |
| `int32_reduction` | 0.188336 ms | 0.180134 ms | 1.0455x | Below the required 1.05x speed threshold |
| `b200_epilogue` | 0.051687 ms | 0.048537 ms | 1.0649x | Passed the speed threshold in this run |

These are **exploratory, non-canonical measurements on a virtualized host**.
They are one fixed-resident-buffer harness run, not the blog's rotating-input,
alternating-round protocol. No expectation file or published speedup baseline
was created. Repeat interleaved measurements on the intended deployment hardware
before treating either ratio as a durable improvement.

The reduction is not marked as a passing optimization merely because it was
slightly faster. The 256-bit epilogue result applies to that isolated conversion
and store workload, not to a full B200 NVFP4 GEMM.

## Correctness and integration checks

The final focused suite passed **66 tests with 2 expected hardware skips** on
B200, and **63 tests with 5 expected hardware skips** on the local CPU host.

- Real B200 tests compare CUB and the safe reduction against an independent
  integer sum, on a nondefault CUDA stream.
- Real B200 epilogue tests compare every output element with PyTorch's FP16
  conversion, cover nondefault-stream ordering, and verify caller-owned RNG and
  setup/verification lifecycle behavior.
- CPU tests cover source integrity, changed/missing/extra source rejection,
  build-cache identity, explicit hardware/toolkit contracts, target discovery,
  output lifecycle, and selectable NVFP4 rungs.
- All four targets are discoverable through `aisp bench list-targets`.
- All 12 README-generator tests pass, and the lab index plus four chapter
  READMEs match their generator definitions.
- Repository-wide benchmark lint checked 944 entrypoints with zero errors and
  zero warnings. Focused Ruff, formatting, syntax, YAML, and local-link checks
  also passed.
- The H100 adapter compiled and linked for `sm_90a` with CUDA 13.0. No H100
  kernel was executed; numerical and performance qualification still needs H100.
- Every SM103 rung, r0-r9, compiled, linked, and imported together in one Python
  process with CUDA compiler 13.1.80 and `compute_103a,sm_103a`. This isolated
  compile-only check used PyTorch 2.9.1+cu130, cuBLAS 13.0.0.19 headers/library,
  and CUDA 13.1.80 runtime headers/library, with no GPU visible and no kernel launches. The production adapter still requires CUDA runtime and toolkit 13.1+
  on exact SM103 hardware. Its independent host oracle, poisoned-output, guard,
  determinism, and placement checks remain unqualified without B300/GB300.

## Failures caught and corrected

1. The original reduction resets a shared output from inside one CTA while other
   CTAs can already update it. The adaptation resets before the grid on the
   current stream and uses bounded int32 inputs.
2. The first B200 epilogue conversion swapped adjacent FP16 lanes. The full-output
   GPU oracle rejected it; the PTX operands were corrected and the test passed.
3. Publishing an uninitialized epilogue output during setup triggered the
   harness's precomputation check. Setup now retains only a private allocation;
   the public output is assigned by the timed invocation.
4. Verification payloads retained previous GPU inputs across teardown. Setup and
   teardown now release those payloads in every adapter.
5. PyTorch's extension flags disabled the BF16 conversion used by the original
   Hopper header. The H100 adapter explicitly restores that conversion when
   compiling, without modifying the pinned source.
6. The host's system Python reported conflicting installed-library versions.
   The harness correctly rejected that comparison; reported results use an
   isolated environment with unambiguous package metadata.
7. The final documentation audit caught links added only to generated READMEs.
   Their definitions now live in the README generator, so regeneration preserves
   the chapter mappings.
8. Compiling with CUDA 13.1 exposed an overly broad PyTorch include and a macro
   argument containing a template comma. The adapter now includes only the CUDA
   stream API it uses and gives the schedule shape a named local variable.
9. Importing multiple compiled NVFP4 rungs exposed a Pybind global-type collision.
   Each module now registers its context type locally. The all-rung import check
   passed, and the opt-in SM103 test also imports r0 after setting up r9.

## Reproduce

From `code/`, run the tests on the allocated GPU and then the two B200 targets
serially using the [README commands](README.md#run-with-the-repository-harness):

```bash
python -m pytest tests/test_fast_cu_*.py -q -ra
python -m core.scripts.linting.check_benchmarks labs/fast_cu --fail-on-warnings
```

For actual SM103 validation, explicitly enable its GPU test on B300/GB300 with
CUDA 13.1+:

```bash
AISP_RUN_FAST_CU_NVFP4_GPU_TEST=1 python -m pytest tests/test_fast_cu_nvfp4.py -q -ra
```

The native default is r9; construct `FastCuNvfp4KernelBenchmark(rung=N)` to
select another snapshot. Every rung must pass its own target-host checks before
its results can be used. Retained upstream logs and article figures are source
evidence, not measurements from this integration.
