# Lab - fast.cu kernel optimization studies

This lab brings [Pranjal Shankhdhar's fast.cu](https://github.com/pranjalssh/fast.cu)
into the repository: all twelve Hopper GEMM snapshots, four reduction variants,
and all ten GB300 NVFP4 snapshots live in [upstream/](upstream/). Repository-native
`baseline_*` / `optimized_*` pairs add explicit hardware gates, current-stream
execution, caller-owned random inputs, and full output verification.

The [optimization guide](optimization_guide.md) maps the examples and lessons to
existing chapters and labs. The [source manifest](upstream_manifest.json) pins
revision `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b` and the SHA256 of every upstream
file. The upstream MIT license is preserved in [upstream/LICENSE](upstream/LICENSE).

## Native benchmark pairs

| Target in `labs/fast_cu` | Baseline | Optimized | Required GPU/toolkit |
| --- | --- | --- | --- |
| `h100_bf16_gemm` | cuBLAS BF16 GEMM | fast.cu Hopper kernel 12 | H100, SM90, CUDA 12+ |
| `int32_reduction` | CUB sum | Corrected vectorized warp reduction | H100 SM90/CUDA 12+ or B200 SM100/CUDA 12.8+ |
| `b200_epilogue` | Two 128-bit stores per thread | One 256-bit store per thread | B200, SM100, CUDA 12.9+ |
| `nvfp4_gemm` | cuBLASLt NVFP4 GEMM | fast.cu r9 NVFP4 GEMM | B300/GB300, SM103, CUDA 13.1+ |

The NVFP4 kernel uses architecture-specific SM103 instructions and is **not a
B200 kernel**. Changing the compiler target to SM100 does not port its K=96 MMA
pipeline. The B200 epilogue is a separate experiment applying the article's wide
store technique to the same FP32-to-FP16 conversion workload in both arms. It
does not stand in for a complete NVFP4 GEMM.

## Run with the repository harness

Run from `code/` in the normal CUDA-enabled repository environment (PyTorch,
matching local CUDA toolkit, Ninja, Nsight Compute and Nsight Systems):

```bash
python -m cli.aisp bench list-targets --chapter labs/fast_cu

# B200: architecture-compatible adaptations
python -m cli.aisp bench run --targets labs/fast_cu:int32_reduction --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/fast_cu:b200_epilogue --profile deep_dive --single-gpu

# H100
python -m cli.aisp bench run --targets labs/fast_cu:h100_bf16_gemm --profile deep_dive --single-gpu

# B300/GB300 with CUDA 13.1+
python -m cli.aisp bench run --targets labs/fast_cu:nvfp4_gemm --profile deep_dive --single-gpu
```

Select an allocated GPU using `CUDA_VISIBLE_DEVICES` before launching. Unsupported
hardware/toolkits produce explicit `SKIPPED:` diagnostics; no precision or
architecture fallback is substituted. Extension compilation and preparation run
in setup, outside timed execution. The source tree is checked before compilation,
and build cache names include all upstream file digests and adapter sources.

Use the repository's strict validity profile and clock-lock/profiler machinery for
performance evidence. The original upstream Makefile and shell scripts are retained
as source evidence; their direct `sudo nvidia-smi`/`ncu` helpers are not the lab's
execution interface. Do not build inside the pinned `upstream/` directory: generated
files intentionally fail the exact file-set integrity check.

## Correctness and measurement boundaries

- Both arms use the same logical inputs, precision, shape, and full timed output.
  Preparation, JIT, workspace allocation, and reference checks are separate from
  the steady-state hot path.
- The reduction adaptation moves output initialization ahead of the grid on the
  same stream, preventing the original cross-block reset race. Its bounded input
  distribution avoids signed 32-bit overflow.
- The Hopper adapter normalizes the kernel's physical transposed output into a
  logical matrix view for verification. A successful upstream process exit is not
  treated as evidence of correctness.
- The NVFP4 native pair uses a single set of resident buffers. The article's
  headline instead rotates input buffers and alternates five cooled-down timing
  rounds. Those are different workload/cache protocols; this lab does not import
  the article's speedup as an expectation.
- [Validation](validation.md) records what has actually run and what remains
  hardware-gated. Source preservation and CPU tests do not prove GPU correctness.

Focused checks, from `code/`:

```bash
python -m pytest tests/test_fast_cu_*.py -q
python -m core.scripts.linting.check_benchmarks labs/fast_cu --fail-on-warnings
```

See [performance_intake.yaml](performance_intake.yaml) and
[workload_spec.yaml](workload_spec.yaml) for cost hypotheses and fixed workloads.
