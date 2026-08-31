# Lab - Ozaki Scheme on B200

## Summary
This lab benchmarks NVIDIA's CUDA 13 FP64 fixed-point emulation path for dense GEMM on Blackwell and frames it explicitly as an Ozaki-style lab instead of a generic cuBLAS timing demo.

The implemented runtime paths are:

- native FP64 GEMM as the control path
- Ozaki-I style dynamic retained-bit control through cuBLAS FP64 emulation
- Ozaki-I style fixed retained-bit control through cuBLAS FP64 emulation

The lab includes narrative-check drivers, but their optimized runs require an explicitly reviewed accuracy policy. The following claims remain target-validation gates:

- controllable accuracy
- adaptive retained-bit behavior
- reproducibility across repeated runs

## Coverage Against The Ozaki Narrative

| Slide theme | In this lab | Status |
| --- | --- | --- |
| Age of low-precision matrix engines / "why now?" | Motivation, measured B200 result, CUDA 13 emulation control surface | Implemented |
| Ozaki-I slicing-based FP64 emulation | Dynamic + fixed Ozaki-I style GEMM paths on top of cuBLAS FP64 emulation | Implemented |
| CUDA 13 opt-in FP64 emulation | Explicit cuBLAS-handle API path plus documented env-var equivalent | Implemented |
| Controllable accuracy | `narrative_checks.py --section accuracy` sweeps fixed retained-bit counts and reports time/error | Implemented |
| Adaptive behavior | `narrative_checks.py --section adaptive` sweeps input scale and reports dynamic retained bits | Implemented |
| Reproducibility | `narrative_checks.py --section reproducibility` checks checksum, retained bits, and emulation-use stability | Implemented |
| Ozaki-II / Chinese Remainder Theorem | Context, papers, and code references only | Context only |
| Linear-algebra applications beyond GEMM | Documented as downstream consumers of accurate matrix multiplication | Context only |
| Disadvantages / tradeoffs | Explicit limitations section | Implemented |
| Papers and open-source code | Curated reference section | Implemented |

The key scope boundary is that this lab is still a GEMM-focused Ozaki-I lab. It does not claim to be a full Ozaki-II modular-arithmetic implementation.

## Historical B200 Result: accuracy not qualified
These earlier numbers predate the full-array verifier. The old signed-checksum comparison used `rtol=1e-2, atol=1e-2` for outputs around `1e-5`; even zero output could pass. These timings are retained as history, not current accepted speedups:

| Variant | Time (ms) | Speedup vs native | Retained bits | Max abs error | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native FP64 | `6.714` | `1.00x` | `-1` | `0` | `0` |
| Ozaki dynamic | `1.130` | `5.94x` | `4` | `3.0e-06` | `0` |
| Ozaki fixed | `0.842` | `7.98x` | `12` | `0` | `0` |

The dynamic path's reported `3e-6` maximum error is material relative to this small output scale. The old tolerance does not establish FP64-equivalent accuracy.

## Accuracy gate: target calibration pending

The CUDA executable now checks every element against a separately allocated native FP64 reference before returning an accepted timing. The host comparator in `accuracy.h` rejects non-finite values and aliased reference storage and reports relative L2 and maximum absolute error normalized by the largest reference magnitude. Both accuracy limits must be explicitly configured, finite, and in `[0,1)`. The Python checksum comparison is secondary and defaults to exact equality; it cannot replace the full-array check.

No numerical threshold has been calibrated or approved here. Without limits, optimized binaries fail before GPU allocation. Python wrappers and runners accept `AISP_OZAKI_ACCURACY_POLICY`, a JSON file with `schema_version: 1` and separate `dynamic`/`fixed` objects containing `relative_l2`, `normalized_max_abs`, `checksum_rtol`, and `checksum_atol`. The first three fields must be finite and in `[0,1)`; the last must be finite and nonnegative. A configured policy is not measured accuracy evidence.

After compiling on the target, collect errors without accepting a benchmark result (binary suffix depends on the target):

```bash
make -C labs/ozaki_scheme all
labs/ozaki_scheme/optimized_ozaki_scheme_dynamic_sm100 --m 4096 --n 4096 --k 4096 --seed 2026 --input-scale 0.001 --dynamic-max-bits 16 --dynamic-offset -56 --accuracy-measure-only
labs/ozaki_scheme/optimized_ozaki_scheme_fixed_sm100 --m 4096 --n 4096 --k 4096 --seed 2026 --input-scale 0.001 --fixed-bits 12 --accuracy-measure-only
```

Measurement-only runs exit **2** with `ACCURACY_STATUS: MEASUREMENT_ONLY_NOT_ACCEPTED`; they omit `TIME_MS`, TFLOPS and the verifier checksum. Sweep seeds, retained-bit settings, and input scales before selecting workload-specific bounds. Repeated controlled GPU runs, numerical calibration and performance acceptance remain open gates.

## Why This Lab Exists
The motivating story from the slides is that low-precision tensor-core hardware keeps getting faster while native FP64 throughput improves much more slowly, so accurate FP64-equivalent matrix multiplication increasingly wants an emulation story instead of a brute-force FP64 story.

That is exactly the lane this lab exercises:

- use the fast low-precision matrix engine indirectly
- keep a native FP64 control path beside it
- show when the emulation path is both faster and numerically acceptable

Relevant references:

- NVIDIA blog: <https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/>
- cuBLAS floating-point emulation docs: <https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation>

## Ozaki-I In This Lab
The implemented runtime path is the Ozaki-I side of the narrative: slice-like retained-bit control feeding tensor-core-backed emulation for FP64 GEMM.

In practical repo terms:

- `optimized_ozaki_scheme_dynamic.py` is the adaptive Ozaki-I style path
- `optimized_ozaki_scheme_fixed.py` is the fixed-budget Ozaki-I style path
- both use `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`
- both fail fast if cuBLAS falls back to native FP64 or full-array errors exceed explicitly configured limits

The default scenario keeps the original `4096 x 4096 x 4096` GEMM size but narrows the operand range with `input_scale=1e-3`. That puts the dynamic controller into a regime where it can materially reduce the retained-bit budget instead of behaving like an accuracy-only fallback.

## Ozaki-II Context
The screenshots also cover Ozaki-II, Chinese Remainder Theorem reconstruction, modular arithmetic, and the newer modular-GEMM papers.

This lab does not implement Ozaki-II today. Instead, it makes that scope boundary explicit and keeps Ozaki-II in the lab as:

- conceptual context for the broader Ozaki family
- paper/code references for follow-on work
- a reminder that "Ozaki lab" here currently means "Ozaki-I style cuBLAS emulation lab", not "all Ozaki-family algorithms are implemented"

## Controllable Accuracy
The fixed retained-bit path is the direct "controllable accuracy" hook in this lab. You can sweep the retained-bit budget and watch the speed/error tradeoff move.

```bash
python labs/ozaki_scheme/narrative_checks.py --section accuracy
python labs/ozaki_scheme/narrative_checks.py --section accuracy --fixed-bits-list 8,10,12,14
```

That script reports:

- fixed retained-bit count
- time
- speedup vs native FP64
- max absolute error
- mean absolute error

This makes the slide claim concrete instead of leaving it as a diagram.

## Adaptive Behavior
The dynamic variant is the practical "adaptive" part of this lab. It does not prove the full Ozaki-I reuse theorem from the slides, but it does let you observe the retained-bit budget responding to the input distribution rather than staying pinned.

```bash
python labs/ozaki_scheme/narrative_checks.py --section adaptive
python labs/ozaki_scheme/narrative_checks.py --section adaptive --input-scales 1e-1,1e-2,1e-3,1e-4,1e-5
```

The adaptive section sweeps input scale and reports:

- input scale
- dynamic retained bits selected by cuBLAS
- time
- speedup vs native FP64
- max absolute error

## Reproducibility
The slides call reproducibility out as a first-class property. The lab now checks that claim directly for repeated runs on the same host.

```bash
python labs/ozaki_scheme/narrative_checks.py --section reproducibility
python labs/ozaki_scheme/narrative_checks.py --section reproducibility --variant fixed --repeats 5
```

The reproducibility section checks:

- `RESULT_CHECKSUM`
- retained-bit stability
- emulation-used stability

Accepted runs emit `RESULT_CHECKSUM` for repeated-run stability. Measurement-only calibration deliberately emits no accepted checksum or timing.

## CUDA 13 Control Surface
The screenshots mention the environment-variable path from CUDA 13:

```bash
export CUBLAS_EMULATE_DOUBLE_PRECISION=1
export CUBLAS_EMULATION_STRATEGY=performant
```

This lab uses the explicit cuBLAS handle APIs because benchmark claims need to pin the emulation behavior instead of depending on ambient shell state.

The equivalent lab-side control is:

- `--emulation-strategy eager`
- `--emulation-strategy performant`
- `--emulation-strategy default`

Examples:

```bash
python labs/ozaki_scheme/run_lab.py --emulation-strategy eager
python labs/ozaki_scheme/run_lab.py --emulation-strategy performant
python labs/ozaki_scheme/narrative_checks.py --section accuracy --emulation-strategy default
```

Why the harness targets pin `eager` today:

- benchmark labels should not silently no-op
- `eager` forces the emulation path instead of letting the library decide to stay native
- the optimized variants already fail if emulation is not actually used

## Applications and Limits
The Ozaki slides naturally lead from GEMM to linear algebra workloads such as LU, Cholesky, QR, and eigensolvers. That is directionally correct, but this lab still measures GEMM only.

| Topic | Lab stance |
| --- | --- |
| Dense GEMM | Directly measured |
| SYMM / SYRK / SYR2K / TRMM / TRSM | Not benchmarked here; mentioned as adjacent level-3 BLAS consumers |
| Linear algebra factorization / eigensolvers | Not benchmarked here; treated as downstream application context |
| Mixed-precision solver workflows | Not benchmarked here |

The lab is useful precisely because it stays honest about that boundary. It proves the matrix-multiplication building block first.

## Disadvantages
The disadvantages from the slides are real and apply here too:

- performance depends strongly on problem size and arithmetic intensity
- large input-magnitude variation can push the dynamic path toward higher retained-bit budgets
- the fixed path can be very fast but only when the retained-bit choice still satisfies the accuracy goal
- the emulation story is compelling for GEMM-dominated workloads; it is much weaker when matrix multiplication is not the dominant cost

This is why the lab keeps native FP64 beside the emulated variants instead of assuming emulation is always the right answer.

## Benchmark Structure
The lab exposes one baseline and two optimized variants:

| Path | Role |
| --- | --- |
| `baseline_ozaki_scheme.py` | Native FP64 accuracy and performance anchor |
| `optimized_ozaki_scheme_dynamic.py` | Ozaki-I style dynamic retained-bit control |
| `optimized_ozaki_scheme_fixed.py` | Ozaki-I style fixed retained-bit control |

## Direct Runner
Use the direct lab runner when you want a single three-way table:

```bash
python labs/ozaki_scheme/run_lab.py
python labs/ozaki_scheme/run_lab.py --m 6144 --n 6144 --k 6144 --iters 6
python labs/ozaki_scheme/run_lab.py --emulation-strategy performant
```

## Narrative Runner
Use the narrative runner when you want the lab to answer the slide-driven questions directly:

```bash
python labs/ozaki_scheme/narrative_checks.py --section all
python labs/ozaki_scheme/narrative_checks.py --section accuracy
python labs/ozaki_scheme/narrative_checks.py --section adaptive
python labs/ozaki_scheme/narrative_checks.py --section reproducibility
```

## Harness Targets
Use the benchmark harness when you want verification, expectations, and profiler integration:

```bash
python -m cli.aisp bench list-targets --chapter labs/ozaki_scheme
python -m cli.aisp bench run --targets labs/ozaki_scheme --profile minimal
python -m cli.aisp bench run --targets labs/ozaki_scheme:ozaki_scheme_dynamic --profile minimal
python -m cli.aisp bench run --targets labs/ozaki_scheme:ozaki_scheme_fixed --profile minimal
```

The canonical expectation file for this host is `labs/ozaki_scheme/expectations_b200.json`, generated from a strict `--profile none --update-expectations` run.

## Exact Repro
These are the commands behind the numbers quoted above:

```bash
python -m cli.aisp bench run --targets labs/ozaki_scheme --profile none --update-expectations
python -m cli.aisp bench run --targets labs/ozaki_scheme --profile minimal
python -m cli.aisp bench run --targets labs/ozaki_scheme:ozaki_scheme_dynamic --profile minimal
python labs/ozaki_scheme/run_lab.py --skip-build
python labs/ozaki_scheme/narrative_checks.py --section all --skip-build
```

The timing table in this README comes from run `20260318_183903__bench__profile_none_targets_labs_ozaki_scheme`.

The generic `--profile minimal` run captures full-lab profiling for the best optimized path (`fixed`) under run `20260318_184813__bench__profile_minimal_targets_labs_ozaki_scheme`.

The dynamic-path explanation below comes from the dedicated profiled pair run `20260318_185200__bench__profile_minimal_targets_labs_ozaki_scheme_ozaki_scheme_dynamic`.

## Why Dynamic Wins Here
The profiled `ozaki_scheme_dynamic` pair on the tuned default scenario points to a reduction in emulation work, not a bandwidth-driven win:

- strict timing run (`20260318_183903__bench__profile_none_targets_labs_ozaki_scheme`) measured native FP64 at `6.714 ms` and dynamic Ozaki at `1.130 ms`
- profiled dynamic pair run (`20260318_185200__bench__profile_minimal_targets_labs_ozaki_scheme_ozaki_scheme_dynamic`) kept the same verification-clean result while showing baseline `nsys total_gpu_time_ms ~= 258.4` versus dynamic `~= 157.0`
- the minimal NCU pass did not expose a dramatically different bandwidth signature, and the sampled kernel counters stayed close run-to-run

The strongest signal is therefore the large end-to-end `nsys` reduction, together with the stable numerical result: the dynamic controller appears to be collapsing the amount of fixed-point emulation work needed for this low-range input distribution, rather than benefiting from a memory-system optimization.

## Papers and Code
The screenshots cite a broader Ozaki literature trail than the original README did. These are the most relevant anchors for this lab.

### Ozaki-I papers

- D. Mukunoki, K. Ozaki, T. Ogita, T. Imamura, "DGEMM Using Tensor Cores, and Its Accurate and Reproducible Versions," LNCS 12151, 2020
- H. Ootomo, K. Ozaki, R. Yokota, "DGEMM on Integer Matrix Multiplication Unit," IJHPCA 38, 2024
- Y. Uchino, K. Ozaki, T. Imamura, "Performance Enhancement of the Ozaki Scheme on Integer Matrix Multiplication Unit," IJHPCA 39:3, 2025
- D. Mukunoki, "DGEMM using FP64 Arithmetic Emulation and FP8 Tensor Cores with Ozaki Scheme," SCA / HPCAsia Workshops 2026

### Ozaki-II papers

- K. Ozaki, Y. Uchino, T. Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique," arXiv:2504.08009, 2025
- Y. Uchino, K. Ozaki, T. Imamura, "High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8 Matrix Engines," SC '25 Workshops, 2025
- Y. Uchino, Q. Ma, T. Imamura, K. Ozaki, P. L. Gutsche, "Emulation of Complex Matrix Multiplication based on the Chinese Remainder Theorem," arXiv:2512.08321, 2025
- Y. Uchino, K. Ozaki, T. Imamura, "Error Analysis of Matrix Multiplication Emulation Using Ozaki-II Scheme," arXiv:2602.02549, 2026

### Open-source code references

- Mukunoki / OzBLAS: <https://github.com/mukunoki/ozblas>
- Ootomo / ozIMMU: <https://github.com/enp1s0/ozIMMU>
- Uchino / accelerator_for_ozIMMU: <https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU>
- NVIDIA / cuBLASDx: <https://developer.nvidia.com/cublasdx-downloads>

## What The Binary Emits
Each CUDA binary prints:

- `TIME_MS`
- `TFLOPS`
- `RETAINED_BITS`
- `EMULATION_USED`
- `EMULATION_STRATEGY`
- `MAX_ABS_ERROR`
- `MEAN_ABS_ERROR`
- `RESULT_CHECKSUM`
- `VERIFY_CHECKSUM` in verify builds

That keeps the harness integration thin while still surfacing the Ozaki-specific knobs that matter.

## Validation Goal
The optimized variants only count if both conditions hold:

- `EMULATION_USED: 1`
- numerical error remains small relative to the native FP64 reference

If cuBLAS floating-point emulation falls back to native FP64, the optimized binary exits non-zero.
