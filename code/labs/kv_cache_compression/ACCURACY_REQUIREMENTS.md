# KV projection arithmetic requirements

## Decision fixed before target runs

The reviewed policy is [`accuracy_policy.json`](accuracy_policy.json). It compares every
BF16 K/V cache element with the existing unquantized BF16 PyTorch projection path,
which bypasses Transformer Engine quantization and packing. Shape, dtype, finite-value,
storage-alias, relative-L2, and maximum-error checks all remain mandatory.

The source ceilings are:

| Variant | Full-cache relative L2 | Maximum error / maximum reference |
| --- | ---: | ---: |
| Delayed-scaling FP8 E4M3 | `0.0625` (`2^-4`) | `0.0625` (`2^-4`) |
| NVFP4 E2M1 | `0.25` (`2^-2`) | `0.25` (`2^-2`) |

E4M3 stores three fraction bits, making half the spacing within a normal binade
`2^-4`; E2M1 stores one fraction bit, making the analogous quantity `2^-2`.
Those representation-scale quantities define the independent full-cache engineering
ceilings. NVFP4 also uses a per-16-element E4M3 block scale and a global FP32 scale,
as described in the [Transformer Engine NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html).
The pairwise allowance is secondary: each arm must first pass its own full-cache
reference check. It is a shared-reference envelope rather than a raw
`torch.allclose(rtol=0.25, atol=0.0625)` call. For the common BF16 reference `R`,
the FP8 cache `B`, the NVFP4 cache `O`, and `M = max(abs(R))`, the two unchanged
maximum-error requirements and the triangle inequality give:

```text
max(abs(B - O))
  <= max(abs(B - R)) + max(abs(O - R))
  <= (0.0625 + 0.25) * M
```

The ordinary pair therefore transports the complete raw K/V arrays and uses an
exact-keyed output policy with `rtol=0` and
`atol=(0.0625 + 0.25) * max(abs(R))`. Both arms must derive an identical policy
from their independent reference pass or the harness rejects the comparison.
The coefficient `0.3125` is fixed by the two declared format ceilings; it is not
calibrated or fitted to an observed pairwise difference. A zero reference yields
zero absolute tolerance and admits only exact zero outputs.

These limits were fixed from the declared formats before the new candidate runs.
Prior calibration errors remain diagnostics and were not used to widen a bound.
Changing the JSON above a source ceiling is rejected by `load_accuracy_policy()`;
a deliberate ceiling change therefore requires a reviewed source change.

## Required qualification matrix

Both variants must pass one nominal seed, two unseen holdout seeds, an alternating-sign
edge distribution, and a sparse two-channel outlier distribution. All cohorts preserve
the production batch, hidden dimension, sequence lengths, cache shape, and stored dtype.
`calibrate_accuracy` writes measurement-only receipts, and `qualify_accuracy` rejects a
missing, duplicate, mismatched, non-finite, or over-budget receipt while retaining every
reason in its summary.

Run these commands serially on the requested B200 from `code/`:

```bash
accuracy_out=/tmp/ai-perf-followthrough-20260908-private/accuracy
mkdir -p "$accuracy_out"

for variant in fp8 nvfp4; do
  python -m labs.kv_cache_compression.calibrate_accuracy --variant "$variant" --cohort nominal --seed 2026 --output "$accuracy_out/kv-receipt-$variant-nominal-2026.json"
  python -m labs.kv_cache_compression.calibrate_accuracy --variant "$variant" --cohort holdout --seed 2027 --output "$accuracy_out/kv-receipt-$variant-holdout-2027.json"
  python -m labs.kv_cache_compression.calibrate_accuracy --variant "$variant" --cohort holdout --seed 2029 --output "$accuracy_out/kv-receipt-$variant-holdout-2029.json"
  python -m labs.kv_cache_compression.calibrate_accuracy --variant "$variant" --cohort alternating --seed 2039 --output "$accuracy_out/kv-receipt-$variant-alternating-2039.json"
  python -m labs.kv_cache_compression.calibrate_accuracy --variant "$variant" --cohort sparse_outlier --seed 2053 --output "$accuracy_out/kv-receipt-$variant-sparse_outlier-2053.json"
done

python -m labs.kv_cache_compression.qualify_accuracy \
  --policy labs/kv_cache_compression/accuracy_policy.json \
  --output "$accuracy_out/kv-qualification.json" \
  "$accuracy_out"/kv-receipt-*.json
```

Only after `kv-qualification.json` says `qualified_arithmetic_gate`, run the ordinary
pair with the same checked-in policy:

```bash
AISP_KV_CACHE_ACCURACY_POLICY="$PWD/labs/kv_cache_compression/accuracy_policy.json" \
  python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
```

Passing establishes this lab's arithmetic full-cache requirement on the recorded
hardware and software stack. It does not establish attention quality, model quality,
task quality, or application quality, and it does not mean the BF16 cache is compressed.
