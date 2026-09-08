# Ozaki arithmetic requirements

## Decision fixed before target runs

The reviewed policy is [`accuracy_policy.json`](accuracy_policy.json). Production-size
candidate arrays are compared element by element with a separately executed native-FP64
cuBLAS result. Small edge cohorts additionally use `reference_gemm_long_double()`, a
CPU row-major GEMM with wider-than-FP64 accumulation. That path refuses to run when the
host aliases `long double` to FP64 or when a request exceeds 50 million FMAs.

The source ceilings are:

| Variant | Relative L2 | Maximum error / maximum reference |
| --- | ---: | ---: |
| Dynamic, max 16 bits, offset `-56` | `0.03125` (`2^-5`) | `0.0625` (`2^-4`) |
| Fixed 12-bit | `0.000244140625` (`2^-12`) | `0.00048828125` (`2^-11`) |

NVIDIA documents that dynamic mantissa control targets native-FP64 accuracy at the
default precision, but a mantissa-bit offset explicitly trades accuracy for performance;
fixed control likewise does not guarantee native-FP64 accuracy. See the
[cuBLAS fixed-point emulation documentation](https://docs.nvidia.com/cuda/cublas/index.html#fixed-point).
Because this lab deliberately uses `dynamic_offset=-56`, the dynamic thresholds are
an explicit five-bit aggregate and four-bit worst-output engineering floor. Fixed-12
must remain within one `2^-12` grid step in aggregate and two steps at the worst output.
These are acceptance decisions for the declared lab and cohorts, not forward-error
theorems for arbitrary or ill-conditioned matrices.

The checksum tolerances are secondary harness bounds. They follow from the full-array
relative-L2 ceiling, Cauchy-Schwarz, and the declared uniform input bound:
`atol = relative_l2 * m * n * k * input_scale^2`; `rtol` is zero. Candidate results
cannot pass on the checksum alone because the executable first gates the complete array.

These limits were fixed before new candidate execution. Prior measurement-only logs
remain diagnostics and were not used to widen a bound. `load_accuracy_policy()` rejects
any JSON value above the source-defined ceilings.

## Required qualification matrix

Both dynamic and fixed variants must pass the 4096-cubed nominal seed, two unseen
4096-cubed holdouts, and two small rectangular edges. The alternating and dynamic-range
edges use the independent CPU long-double reference. The executable prints the input,
algorithm, and reference identities so `qualify_accuracy` can reject substitutions.

From `code/labs/ozaki_scheme/` on the requested B200, build once:

```bash
make ARCH=sm_100 all
accuracy_out=/tmp/ai-perf-followthrough-20260908-private/accuracy
mkdir -p "$accuracy_out"
```

The measurement-only binary intentionally exits `2`. This helper preserves the log and
accepts only that explicit disposition:

```bash
measure_only() {
  output=$1
  shift
  set +e
  "$@" >"$output" 2>&1
  status=$?
  set -e
  if [ "$status" -ne 2 ]; then
    return "$status"
  fi
}
```

Run each candidate and cohort serially:

```bash
for variant in dynamic fixed; do
  if [ "$variant" = dynamic ]; then
    binary=./optimized_ozaki_scheme_dynamic_sm100
    variant_args=(--dynamic-max-bits 16 --dynamic-offset -56)
  else
    binary=./optimized_ozaki_scheme_fixed_sm100
    variant_args=(--fixed-bits 12)
  fi

  for seed in 2026 2027 2029; do
    cohort=holdout
    if [ "$seed" -eq 2026 ]; then cohort=nominal; fi
    measure_only "$accuracy_out/ozaki-$variant-$cohort-$seed.log" "$binary" \
      --m 4096 --n 4096 --k 4096 --warmup 3 --iters 10 --seed "$seed" \
      --input-scale 0.001 --input-pattern uniform --reference-mode native_fp64 \
      --emulation-strategy eager "${variant_args[@]}" --accuracy-measure-only
  done

  measure_only "$accuracy_out/ozaki-$variant-alternating_edge-2039.log" "$binary" \
    --m 17 --n 19 --k 23 --warmup 1 --iters 1 --seed 2039 --input-scale 1 \
    --input-pattern alternating --reference-mode cpu_long_double \
    --emulation-strategy eager "${variant_args[@]}" --accuracy-measure-only

  measure_only "$accuracy_out/ozaki-$variant-dynamic_range_edge-2053.log" "$binary" \
    --m 31 --n 29 --k 37 --warmup 1 --iters 1 --seed 2053 --input-scale 1 \
    --input-pattern dynamic_range --reference-mode cpu_long_double \
    --emulation-strategy eager "${variant_args[@]}" --accuracy-measure-only
done
```

Qualify the complete set from `code/`:

```bash
python -m labs.ozaki_scheme.qualify_accuracy \
  --policy labs/ozaki_scheme/accuracy_policy.json \
  --output /tmp/ai-perf-followthrough-20260908-private/accuracy/ozaki-qualification.json \
  /tmp/ai-perf-followthrough-20260908-private/accuracy/ozaki-*.log
```

Only after the summary says `qualified_arithmetic_gate`, run the ordinary pair with
the same checked-in policy:

```bash
AISP_OZAKI_ACCURACY_POLICY="$PWD/labs/ozaki_scheme/accuracy_policy.json" \
  python -m cli.aisp bench run --targets labs/ozaki_scheme --profile minimal
```

Passing establishes the declared arithmetic GEMM requirement for this stack and these
cohorts. It does not establish solver, model, task, or application quality.
