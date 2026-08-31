# W1-081 native FP8 source repair

Source and CPU contracts pass; actual CUDA GEMMs, workload error calibration and
performance remain HOLD. No GPU result is inferred from CPU FP8 conversions.

The original scale-one E4M3FN cast produces NaN for the finite CPU values 500,
2000, -2000 and 32000. The audit called this saturation; the actual PyTorch 2.8
CPU encoding exhibits nonfinite overflow. The original benchmark also returned
expert-sorted weighted routes instead of complete token outputs and compared FP8
work with a BF16 peak. The latter two defects were confirmed from source, not an
actual GPU execution of the old benchmark.

The candidate computes per-expert weight scales during setup and per-tensor
activation scales inside timing. E4M3 stores x/scale and all three `_scaled_mm`
calls receive the corresponding FP32 dequantization scales. Weight operands are
column-major. Every route is unsorted, then all top-k outputs are summed in FP32.
The named B200 reference ceiling is 4500 dense FP8 TFLOP/s; it is not detection of
the current device, a measured utilization value or a promised speedup.

Full outputs are compared with an independent unquantized BF16 reference using
original CPU-resident weights and unsorted logical routing. This reference does
not reuse candidate FP8 weights, output buffers or sorting. Its transfer and
compute are outside benchmark timing. Retaining reference weights has a memory
cost; the code no longer claims total-memory compression.

Normal setup requires `AISP_NATIVE_FP8_ACCURACY_POLICY` pointing to reviewed JSON:
`schema_version` is 1; `native_fp8` contains `relative_l2`,
`normalized_max_abs`, `pairwise_rtol` and `pairwise_atol`. No default numerical
acceptance threshold is supplied. The first three limits must be finite and in
[0,1); absolute tolerance must be finite and nonnegative. A configured policy is
not evidence that the target/workload has met it.

From `code/`, collect actual CUDA errors with a fresh destination:

```bash
python -m labs.moe_optimization_journey.calibrate_native_fp8 --output /absolute/new/attempt.json
```

The CLI records its shape, seed, target, source hashes, partial completed errors
and any failure/HOLD. It refuses to overwrite an attempt and never accepts or
creates an error threshold. Three input amplitudes at one seed/routing are a
bounded diagnostic, not workload-wide calibration. Normal benchmark generation
uses the harness's initial seed via a private CPU generator.

## Verification and retained attempts

- `first-tests.txt`: 21 CPU tests passed, one actual CUDA test skipped.
- `combined-tests.txt`: 37 passed, four CUDA skips before review follow-up.
- `final-tests.txt`: 37 passed, four skips, one newly added subprocess fixture
  failed because it inherited repository-root cwd; no product math failure.
- `post-review-tests.txt`: 38 passed, four CUDA skips after explicitly selecting
  the subprocess code directory. This includes real CLI HOLD recording and
  no-overwrite behavior.
- The original broad Ruff invocation found import formatting, new test inline
  branches and inherited naming conventions; imports/branches were corrected.
  `post-review-ruff.txt` records E/F/I checks with E501/E741 excluded.
- `cuda-preflight.txt`: original CPU CLI returned 3/HOLD without a result JSON.
  Independent review required preserving failures/HOLD, so the CLI was fixed;
  `cuda-preflight-after-review.json` and `.txt` preserve the subsequent actual
  no-CUDA attempt with source hashes and exit 3.
- Full-output negative controls reject all-zero output, last-element corruption,
  cancelling errors, nonfinite values, aliased storage and truncated output.
  An exact dyadic conversion family checks encoding/scales without inventing a
  MoE error budget. The existing source-contract function was also executed via
  AST isolation because its full test module imports unavailable Triton here.
- A real CUDA production-path test with changed input amplitudes, full payload
  readback and last-element poisoning is prepared. It requires an externally
  reviewed policy and the actual CUDA backend; neither requirement is faked.

Independent review accepted scale direction, layout, complete route combination
and separate BF16 reference. Its failure-receipt follow-up was implemented. The
review did not qualify compiled CUDA or measured performance.

Primary API: [PyTorch 2.9.1 CUDA BLAS source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/Blas.cpp)
checks tensor-wise FP32 singleton scales, operand layout and K/N dimensions
multiple of 16; no M multiple-of-16 assertion was assumed. B200 peak arithmetic
is tied to [NVIDIA HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/)
and the separate P09 receipt.
