# W1-079 MoE layer forward accuracy receipt

The false quantization-based tolerance exception is removed. Full logical forward output now has independent reference checks, omission/NaN/alias rejection, complete original input provenance and a required scale-invariant accuracy policy. **GPU numerical and performance acceptance remain HOLD**. There is no guessed replacement threshold and no claim that setting a policy constitutes measured accuracy.

`original_reproductions.py` executes the actual original benchmark, forward, capture and validation methods from base `b57e4c6a9e261c09ac09208705d040c81b03d35e` with Torch CPU tensors. At N=65,H=48,I=32,E=4,top-k=2:

- BF16 maximum output magnitude is 0.003814697265625; FP16 is 0.0038299560546875.
- An all-zero output passes the original `(0.05,0.2)` pairwise budget and the audit's proposed restored `(0.02,0.02)` budget, and original `validate_result()` returns no error.
- Original payload contains only 1024 values. Adding 1 to the last output element leaves that payload unchanged.

These are actual CPU reproductions, not CUDA simulations or GPU acceptance. An earlier exploratory fixture used N=64 and found the same failure class; its maxima were 0.00390625/BF16 and 0.003925323486328125/FP16, with zero passing both budgets. No failed GPU attempt was omitted: no GPU was used.

## Implementation boundary

Only `moe_layer` in forward mode gets the new policy and full-output reference path. Quantization, grouped GEMM and forward/backward verification contracts remain unchanged and are not newly qualified by this receipt. The original pairwise `(0.02,0.02)` budget is restored as an additional harness check, after the separate policy succeeds. It cannot grant acceptance by itself.

`LayerAccuracyLimits` requires both `relative_l2` and `normalized_max_abs` in `[0,1)`. `AISP_MOE_PTX_LAYER_ACCURACY_POLICY` must point to JSON containing `schema_version: 1` and a `moe_layer_forward` object with those fields. No threshold example or production default is supplied. Missing policy fails before workload allocation. A configured policy must still be validated against the actual device/dtype/shape/input/weight/routing cell.

The reference retains independent CPU copies of original x, expert IDs, routing weights and all three projection tensors. Reference math follows logical routes and unquantized BF16/FP16 matmuls, with bounded 128-row scratch, independently of candidate packing/grouped BMM/scatter combination and baseline output-buffer code. The complete reference output is prepared outside timing. Capture verifies every row and element, rejects all-zero omitted rows and nonfinite/aliased outputs, and copies all N×H output into separate verification storage. `validate_result` calls the same checker. Complete snapshots, rather than only shapes and 32 routing rows, supply the forward verification inputs.

The CPU suite invokes real baseline and BMM paths, independent reference and actual benchmark capture for both backends. Balanced/skewed routing and BF16/FP16 fixtures agree exactly in the bounded CPU cases. Tests use a strict zero-error test policy solely for those cases; this is not a GPU threshold calibration. Zero output, a dropped last row, last-element corruption beyond the former32×32slice, canceling errors, NaN, reference aliasing and shortened output are rejected by the actual production comparator.

Two directly related input-contract defects were also addressed: the synthetic router implements exactly top-2, so other `--top-k` values now fail clearly; explicit zero size/capacity overrides reach validation instead of silently restoring defaults. The workload generator honors the harness seed so fresh-input verification can change x and weights.

## Validation and pending target work

CPU validation from `code/`:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_moe_ptx.py tests/test_moe_cuda_ptx_layer_contract.py tests/test_moe_cuda_ptx_common.py
```

The source receipt records final counts and hashes. Four new actual CUDA cases cover both dtypes and routing histograms, N129/H128/I64, two seeds, both backends, three repeats and a dropped-row negative control through real capture. They require a reviewed policy; missing policy fails on a CUDA host. The existing fifth CUDA skip is the older backward-slice test and is not upgraded or represented as full backward verification.

`calibrate_layer_accuracy.py` uses actual production baseline/BMM functions and the independent reference without generating or applying acceptance limits. After separate GPU custody, from repository root:

```sh
python docs/audits/2026-08-30/evidence/moe-ptx/calibrate_layer_accuracy.py --output /absolute/new/attempt.json
```

The default shape matches the lab; explicit flags select shape, seed, dtype and routing. It records errors at input amplitudes1/4/16. Every attempt reserves a new output file before workload setup, preserves partial records/exceptions, and refuses overwrite. The real CPU invocation retained exit3/HOLD, no numerical GPU evidence. A real invalid-config invocation is covered by a subprocess test and retains FAILED_NOT_ACCEPTED; a second invocation cannot overwrite its file.

No timing is measured by calibration. Run the full target with its reviewed policy only after its exact cell is calibrated and correctness passes, then record clocks/topology/runtime and repeated A/B traces. CUDA forward prepacking is performed in setup while the baseline still masks routes during forward; this timing boundary must remain explicit. Historical layer/other-surface speed claims are not requalified. The root agent owns README generation; `proposed-lab-readme.md` supplies the exact scoped correction.
