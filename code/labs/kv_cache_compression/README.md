# Lab - Quantized Projection Compute with BF16 KV Cache

## Summary
This lab compares per-tensor delayed-scaling FP8 projection GEMMs with NVFP4 projection GEMMs. Both paths store K and V as BF16. The directory name is retained for compatibility; neither path compresses the KV cache.

## Storage and workload
Both variants use batch 8, hidden dimension 16384, 64 heads, 4096 prefill tokens, and 128 decode steps of 128 tokens. The two cache tensors contain 5,368,709,120 elements and occupy 10,737,418,240 bytes at BF16. `kv_cache.storage_bytes`, `storage_bits_per_element`, and `compression_ratio` are calculated from the allocated tensors. The compression ratio relative to BF16 is 1.0, and the optimization goal is compute speed.

The FP8 recipe is `DelayedScaling`; it is not MXFP8 block scaling. The NVFP4 recipe uses supported `NVFP4BlockScaling()` defaults. Both retain identical unquantized BF16 parameter representations while Transformer Engine autocast chooses the low-precision GEMMs.

## Accuracy gate: target calibration pending
Every token, head, and channel in both K and V is checked against an independent PyTorch BF16 projection reference using the original weights and inputs. The reference bypasses Transformer Engine's GEMMs and packing. Checks reject shape mismatches, non-finite values and aliased reference storage; relative L2 and maximum error normalized by reference magnitude avoid signed-checksum cancellation. Verification then snapshots the full cache for the harness pair comparison.

No workload accuracy bound has been calibrated. An accepted benchmark run requires `AISP_KV_CACHE_ACCURACY_POLICY` pointing to a JSON file with `schema_version: 1`, separate `fp8` and `nvfp4` objects, and the fields `relative_l2`, `normalized_max_abs`, `pairwise_rtol`, `pairwise_atol`. The first three must be finite and in `[0,1)`; the last must be finite and nonnegative. Bounds are deliberately not supplied here. Configuring bounds is not evidence that they are appropriate or that this workload passes them.

Collect measurements on the actual CUDA/Transformer Engine host before reviewing a policy:

```bash
python -m labs.kv_cache_compression.calibrate_accuracy --variant fp8 --seed 42 --output /tmp/kv-fp8-seed42.json
python -m labs.kv_cache_compression.calibrate_accuracy --variant nvfp4 --seed 42 --output /tmp/kv-nvfp4-seed42.json
```

Repeat with other fixed seeds and preserve the hardware, software and workload metadata. These commands collect error metrics only; they do not accept output or claim a speedup. After independent accuracy review, run:

```bash
AISP_KV_CACHE_ACCURACY_POLICY=/absolute/path/reviewed-policy.json python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
```

The historical 6066.040/5897.083 ms measurements used the old permissive verifier and are not evidence for the revised accuracy contract or cache compression. Fresh GPU accuracy, memory and performance measurements remain pending.

## Learning Goals
- Compare FP8 and NVFP4 projection GEMMs with the same BF16 KV cache storage.
- Measure full-cache numerical error before reviewing any accuracy policy.
- Keep allocated storage bytes separate from compute precision and latency.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_kv_cache.py`, `optimized_kv_cache_nvfp4.py` | FP8/NVFP4 compute benchmark pair with BF16 cache storage. |
| `kv_cache_common.py` | Shared attention workload and cache allocation. |
| `accuracy.py`, `calibrate_accuracy.py` | Independent full-cache reference, explicit policy, and measurement-only driver. |

## Collecting Accuracy Measurements
Run on the actual CUDA/Transformer Engine host, preserving target and workload metadata.
```bash
python -m labs.kv_cache_compression.calibrate_accuracy --variant fp8 --seed 42 --output /tmp/kv-fp8-seed42.json
python -m labs.kv_cache_compression.calibrate_accuracy --variant nvfp4 --seed 42 --output /tmp/kv-nvfp4-seed42.json
```
- These collect error metrics without accepting an accuracy threshold. Accepted benchmark runs require the separately reviewed policy described above.

## Validation Checklist
- Require an independently reviewed accuracy policy and full-output comparisons before accepting timing.
- Reject zeros, corruption, non-finite values, aliasing, and shape mismatches using the independent reference.
- Verify allocated cache storage bytes and the BF16-relative compression ratio of 1.0.

## Notes
- CPU source checks do not qualify CUDA accuracy, Transformer Engine kernels, memory measurements, or performance.
