# Division by six: name the numerical contract

My advice is to choose the reference and acceptable error before replacing a divide with a stored reciprocal. Matching a reference's bits can matter even when a tolerance check passes. A tolerance contract can be enough when the application has justified that error; it leaves that decision with the application owner.

The original scientific example and historical B200 measurements are identified in the [source provenance](../evidence/provenance.json). This is a maintained successor, with corrected cohort labels, signed-zero direction classification and failure handling. These maintained bytes did not produce the historical results. The original executed source and results remain unchanged in the source archive. Historical reports do not embed a source hash. No new Chris-run GPU result or speedup is claimed here.

## Scope and source results

The whole-binade census is **finite |x| >= 2^-123**, exponent fields 4–254 for both signs: 4,211,081,216 inputs. It is narrower than all inputs with a normal quotient. Inputs in field 3 with significand at least 1.5 also have normal quotients. The original report calls the census “inputs whose quotient is a normal float”; read that historical label with this correction.

The exact residue prediction for reciprocal multiplication is 2,796,202 differences per binade, out of 8,388,608 significands: approximately one third, each one representable step away from zero. Across the census this is 1,403,693,404. Below this census, underflow, signed zeros and flush-to-zero need separate treatment. Normal inputs below the census number 50,331,648; the subset with subnormal correctly rounded quotients numbers 41,943,040.

The [first source B200 count summary](../evidence/antje-b200-summary.md) and [second source B200 count summary](../evidence/antje-b200-run2-summary.md) contain the source-reported counts. The runtime was driver 580.173.02, CUDA 13.0, PyTorch 2.13.0 and Triton 3.7.1. In the same source run, the reciprocal-multiply paths passed default `torch.allclose` for every finite input (Table 2). That describes a uniform census of bit patterns, not the distribution of a model's values. Direction labels outside the census in those historical reports predate the signed-zero fix.

I omit the original timing and profiler narrative here: no native Nsight Compute report accompanied that historical evidence. Equal instruction counts alone would not prove equal outputs or identical machine code.

## What the maintained lab checks

- [CUDA census](src/divide_not_reciprocal/count.cu): all 2^32 patterns, four variants in default and fast-math builds. The reference remains inline `div.rn.f32`. Counts, direction and emitted PTX/SASS are separate observations.
- [Triton and PyTorch division](src/divide_not_reciprocal/triton_div.py): six forms against a tensor-divisor reference, in fixed 2^27-element chunks. The separate `tl.div_rn` row corroborates agreement with that reference on covered inputs; agreement between GPU paths does not independently prove IEEE correctness. A source-level precision flag does not establish the compiler's behavior; the `ieee_rounding` regression assertion is specific to Triton 3.7.1.
- [Hardsigmoid](src/divide_not_reciprocal/hardsigmoid.py): four expressions against `F.hardsigmoid`. The distribution of the rounded shifted operand is different from a uniform input-bit census.
- Optional [component timing and profiling](src/divide_not_reciprocal/timing.py): seven cases, three Triton forms each choosing among seven launch shapes, then CUDA-event rotations. Each CUDA-event interval starts on a drained stream and includes host dispatch. Triton writes to a preallocated output; eager cases allocate outputs. These are component measurements, not end-to-end serving results. Case names describe configured expressions, not verified PTX or SASS. No current lowering claim follows from those names or from an instruction count.

## Setup

Use a Linux NVIDIA CUDA environment with an already installed compatible Python (3.11+), CUDA-enabled PyTorch and Triton. The source run's versions above are a historical reference, not a promise that other versions emit the same code. The CUDA census also requires `nvcc`, `cuobjdump` and a compatible host C++ compiler. `nvcc` is resolved from PATH or `/usr/local/cuda/bin/nvcc`; `cuobjdump` is resolved from PATH or beside that compiler. Set `CXX` to one compiler executable if needed. Preflight records those resolved tools and passes that host compiler to nvcc.

The fixed count chunks contain 2^27 float32 values (512 MiB of input) plus several output, reference and temporary tensors. Allow several GiB of available GPU memory; the exact peak and small-GPU compatibility have not been established. The timing stage allocates 2^28-element tensors and sweeps launch shapes. `--quick` does **not** reduce these tensor sizes or skip the sweep: it counts two chunks per Python experiment, still runs the full CUDA census, shortens the timing rotations to 5 warm-up and 20 timed iterations, and omits ncu. `--skip timing` also omits ncu because the profile uses the timing sweep's selected shapes. To omit only profiling, use `--skip ncu`. For a smaller diagnostic, explicitly skip timing and profiling.

For profiling, install Nsight Compute beforehand and arrange GPU performance-counter permission through the machine owner. The runner does not install tools or change permissions. `pytest` is needed for tests; NumPy is needed only for GPU reference tests. CPU tensor fixtures additionally require PyTorch. Pure derivation tests require neither PyTorch nor a GPU.

## Profiler capture contract

Profiling is a separate observation from the CUDA-event timing protocol. The existing 16,384-element warm call and 2^26-element call are both placed inside one case-specific NVTX push/pop range. Allocations, input filling and scalar operands are prepared outside that range. The command captures all kernels inside it; it does not use a kernel filter or launch skip/count that could hide an extra launch. The [NVIDIA CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering) documents the push/pop range filter. No clock change is requested (`--clock-control none`).

The parser expects raw-page wide CSV: a metric-name header, a units row and one data row per launch. It requires exactly two launches from the recorded probe process, in warm/large order, with the supported case kernel family, equal block shapes and a 4,096-fold x-grid ratio. Triton must also match the actual sweep-selected BLOCK and warp count. This pair binds the large input without assuming ATen's elements per thread. A backend that uses different kernel families, geometry or multiple kernels fails closed and needs investigation; no compatibility is asserted for an untested stack. Only the large launch's counters are normalized by 2^26 elements.

A durable child manifest records the actual tensor sizes, case, process, selected shapes, NVTX range and completed calls. Missing or incomplete manifests, warm-only/extra launches, wrong units, malformed numbers, mismatched geometry, denied counters and failed native imports mark the requested stage FAILED, leave the run PARTIAL and make the command exit nonzero. The raw duration unit must be `nsecond`, the instruction unit `inst` and both DRAM units `byte`; duration is converted to seconds only after validation. `sm__inst_executed.sum` counts warp instructions, as distinguished from thread instructions in the [NVIDIA metric reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference). The per-32-element normalization is not an instruction count per CUDA thread. DRAM traffic is observed rather than replaced by a presumed input-byte count.

The capture and native-report import both explicitly disable ambient config files and kernel-name simplification, and request raw CSV, metric names, base units and demangled printed kernel names. Capture explicitly uses cache control `all` (flush caches before replay passes) and clock control `none`. This records the counter protocol; it is not the timing protocol or an unprofiled cache-state measurement. The [console/file options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#console-output) specify that an explicit page prints console results even when a report is exported. The two decodings must agree on both launch identities and all metrics. Native `.ncu-rep`, capture/import stdout and stderr, and the child manifest are retained with hashes. Tool stdout/stderr are written as exact bytes before strict UTF-8 decoding, including on decode or parse failure.

The CPU CSV fixtures are handwritten synthetic examples of this supported schema, not verbatim official examples or real captures. No target-host raw CSV, native report, NVTX execution or emitted-code check has validated this maintained profiler path yet. Its first real capture and import must be retained before claiming tool compatibility. The fixed timing protocol remains 20/100 warm-up/timed rotations, a 3/10 launch-shape sweep, and sorted p10/p90 samples at zero-based indices `int(0.1*n)` and `int(0.9*n)-1`. Failures during row construction retain completed raw timing samples.

## Run without replacing source evidence

Activate the prepared environment, change to this directory and expose the package on `PYTHONPATH`. These commands do not install or download dependencies:

```sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/test_derivation.py
python -m pytest tests/test_cpu.py tests/test_runner.py tests/test_report.py
python -m divide_not_reciprocal --quick --skip timing,ncu
```

The default output is a **new** timestamp-and-random-suffix directory under `${XDG_STATE_HOME:-$HOME/.local/state}/divide-not-reciprocal/runs/`, outside this package. To choose a location and identifier, provide a fresh name:

```sh
python -m divide_not_reciprocal --results /absolute/path/to/my-runs --name my-new-count-run --skip timing,ncu
python -m divide_not_reciprocal --results /absolute/path/to/my-runs --name my-new-full-run
```

Each identifier reserves its own directory atomically. An existing identifier is refused; there is no overwrite option. Successful requested stages produce COMPLETE output. A handled failure exits nonzero and attempts PARTIAL JSON/Markdown with its stage, full error and completed work. JSON is saved before Markdown; a rendering failure can leave JSON only, and a storage failure can prevent retention. A no-CUDA or missing-tool failure happens before experiment work. An unwritable output directory fails before experiment work and may prevent any file from being saved.

Counts are checkpointed after each CUDA variant and each completed Python chunk. Timing samples are kept on handled failures without inserting disk writes into timed rotations. A crash, power loss or forced process kill can lose the current uncheckpointed chunk/timing stage; a disk failure can prevent the final checkpoint. COMPLETE describes execution of the requested stages, not scientific qualification or publication approval. JSON records the maintained implementation's hashes, requested/skipped stages, environment and samples.

## Validation limits

[Pure tests](tests/test_derivation.py) check the residue rule and require the product to be the exact next float when it differs. [CPU fixtures](tests/test_cpu.py) cover signed zeros, subnormal direction, NaNs, infinities, bit wrapping and input contracts. [Runner fixtures](tests/test_runner.py) exercise collision refusal, checkpoints, compiler errors, metadata failures and malformed/denied profiler output without CUDA. [Populated report fixtures](tests/test_report.py) exercise all five tables with synthetic values and preserve completed JSON when Markdown rendering fails. [GPU tests](tests/test_gpu.py) still require an assigned NVIDIA execution owner and window. A CPU pass does not establish CUDA compilation, FTZ behavior, emitted PTX/SASS, reference agreement, GPU memory fit or profiler compatibility. This maintained successor needs fresh GPU evidence for those claims.


For an assigned-host GPU test receipt, set `DNR_REQUIRE_GPU_TESTS=1` and retain
`pytest tests/test_gpu.py --junitxml=/absolute/new/path/gpu-tests.xml` output.
Dependencies are checked per test: the CUDA census needs nvcc, cuobjdump and a
host compiler; only the Triton tests need Triton. Required mode turns unavailable
dependencies into failures. JUnit records each test's runtime versions, inherited
device mask, selected device and applicable compiler versions. Review the actual
per-test outcomes; a selected two-test smoke proves only those two checks.

PTX/SASS strings, PyTorch scalar-division behavior and profiler kernel names in
the GPU suite are stack-specific regression assertions, not universal compiler
guarantees. A failure on a different stack requires inspecting retained emitted
code and runtime fields. Even eight passing GPU tests do not establish full
2^32-pattern counts, experiment-scale memory fit, real ncu/NVTX compatibility,
the timing kernel's machine lowering, or performance. The profiler's configured
options and raw schema still require a retained target-host capture/import.
