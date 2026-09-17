# fast.cu examples

This lab brings [Pranjal Shankhdhar's fast.cu](https://github.com/pranjalssh/fast.cu)
into the repository: all twelve Hopper GEMM snapshots, four reduction variants,
and all ten GB300 NVFP4 snapshots live in [upstream/](upstream/). The
`baseline_*` / `optimized_*` pairs check hardware support, use the caller's CUDA
stream and random inputs, and verify every output.

The [optimization guide](optimization_guide.md) maps the examples and lessons to
existing chapters and labs. The [source manifest](upstream_manifest.json) pins
base revision `2dfe5e26aecfd9e5f27bf9d5837deea01acda24b`, the SHA256 of every
included file, and three fixes to the older H100 reduction and scheduler examples.
The newer `gb300/nvfp4` r0-r9 kernels are unchanged. A separate
[B200 K64 port](nvfp4_sm100.cuh) adapts the r5 GEMM to SM100.
The upstream MIT license is preserved in [upstream/LICENSE](upstream/LICENSE).

## Benchmark pairs

| Target in `labs/fast_cu` | Baseline | Optimized | Required GPU/toolkit |
| --- | --- | --- | --- |
| `h100_bf16_gemm` | cuBLAS BF16 GEMM | fast.cu Hopper kernel 12 | H100, SM90, CUDA 12+ |
| `int32_reduction` | CUB sum | Corrected vectorized warp reduction | H100 SM90/CUDA 12+ or B200 SM100/CUDA 12.8+ |
| `b200_epilogue` | Two 128-bit stores per thread | One 256-bit store per thread | B200, SM100, CUDA 12.9+ |
| `nvfp4_gemm` | cuBLASLt NVFP4 GEMM | fast.cu r9 NVFP4 GEMM | B300/GB300, SM103, CUDA 13.1+ |
| `nvfp4_sm100` | cuBLASLt NVFP4 GEMM | K64 port of fast.cu r5 | B200, SM100, CUDA 13.0+ |

The original NVFP4 kernels require SM103 K96 instructions. The B200 port uses
K64 instructions and matching scale loading for the complete GEMM. It retains
the two-CTA pipeline, separate operand and scale queues, two accumulator buffers,
and 256-bit FP16 stores. See the [B200 results](sm100_validation.md).

The `b200_epilogue` pair remains a separate experiment that measures store width
in the FP32-to-FP16 conversion step.

## Run with the repository harness

Run from `code/` in the normal CUDA-enabled repository environment (PyTorch,
matching local CUDA toolkit, Ninja, Nsight Compute and Nsight Systems):

```bash
python -m cli.aisp bench list-targets --chapter labs/fast_cu

# B200
python -m cli.aisp bench run --targets labs/fast_cu:nvfp4_sm100 --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/fast_cu:int32_reduction --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/fast_cu:b200_epilogue --profile deep_dive --single-gpu

# H100
python -m cli.aisp bench run --targets labs/fast_cu:h100_bf16_gemm --profile deep_dive --single-gpu

# B300/GB300 with CUDA 13.1+
python -m cli.aisp bench run --targets labs/fast_cu:nvfp4_gemm --profile deep_dive --single-gpu
```

Select an allocated GPU using `CUDA_VISIBLE_DEVICES` before launching. Unsupported
hardware or toolkits return `SKIPPED:`. The examples compile and prepare buffers
before timing. Builds check source hashes and include them in the cache key.

Use the commands above for performance measurements. The harness applies the
repository's strict validity checks, locks clocks, and collects profiles. The
upstream Makefile and shell scripts are included for reference. Build their
examples in a separate copy because generated files under `upstream/` will fail
the source integrity check.

## What is checked

- Both variants use the same logical inputs, precision, shape, and output size.
  Compilation, buffer allocation, and correctness checks run outside timing.
- The reduction resets output before each grid on the same stream. This prevents
  one block from erasing another block's sum. Inputs keep the sum within int32 range.
- The Hopper adapter reads the kernel's transposed output as a logical matrix
  view and checks every element.
- The NVFP4 pair reuses one set of GPU buffers. The article rotates inputs,
  alternates kernel order, and cools down between five timing rounds. Those
  differences prevent a direct comparison with the article's speedup.
- The B200 port is checked against an independent CPU calculation on small and
  partial tiles, with output guards, poisoned buffers, and repeated runs. The
  full 8192³ result also matches cuBLASLt.
- The unchanged SM103 NVFP4 r0-r9 kernels compiled, linked, and imported with CUDA
  13.1.80 for `sm_103a`. No SM103 GPU was available, so runtime correctness and
  performance remain untested.
- [Validation](validation.md) records the tests and measurements completed so far.

Focused checks, from `code/`:

```bash
python -m pytest tests/test_fast_cu_*.py -q
python -m core.scripts.linting.check_benchmarks labs/fast_cu --fail-on-warnings
```

See [performance_intake.yaml](performance_intake.yaml) and
[workload_spec.yaml](workload_spec.yaml) for estimated costs and workload definitions.

The B200 port has its own [workload](nvfp4_sm100_workload_spec.yaml) and
[cost model](nvfp4_sm100_performance_intake.yaml).
