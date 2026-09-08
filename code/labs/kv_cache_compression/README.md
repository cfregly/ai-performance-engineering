# Lab - Quantized Projection Compute with BF16 KV Cache

## Summary
This lab compares per-tensor delayed-scaling FP8 projection GEMMs with NVFP4 projection GEMMs. Both paths store K and V as BF16. The directory name is retained for compatibility; neither path compresses the KV cache.

## Storage and workload
Both variants use batch 8, hidden dimension 16384, 64 heads, 4096 prefill tokens, and 128 decode steps of 128 tokens. The two cache tensors contain 5,368,709,120 elements and occupy 10,737,418,240 bytes at BF16. `kv_cache.storage_bytes`, `storage_bits_per_element`, and `compression_ratio` are calculated from the allocated tensors. The compression ratio relative to BF16 is 1.0, and the optimization goal is compute speed.

The FP8 recipe is `DelayedScaling`; it is not MXFP8 block scaling. The NVFP4 recipe uses supported `NVFP4BlockScaling()` defaults. Both retain identical unquantized BF16 parameter representations while Transformer Engine autocast chooses the low-precision GEMMs. Both refresh packed projection weights on the first token group of each complete iteration and reuse them for its remaining 129 groups; all forward operations and full-cache writes remain timed.

## Accuracy gate and measured B200 scope
Every token, head, and channel in both K and V is checked against an independent PyTorch BF16 projection reference using the original weights and inputs. The reference bypasses Transformer Engine's GEMMs and packing. Checks reject shape mismatches, non-finite values and aliased reference storage; relative L2 and maximum error normalized by reference magnitude avoid signed-checksum cancellation. Verification then snapshots the full cache for the harness pair comparison.

[`ACCURACY_REQUIREMENTS.md`](ACCURACY_REQUIREMENTS.md) records the reviewed, predeclared arithmetic policy and exact B200 qualification matrix. Full-cache relative-L2 and normalized-maximum ceilings are `2^-4` for E4M3 FP8 and `2^-2` for E2M1 NVFP4. These are format-scale engineering limits, not attention, model, task, or application-quality guarantees. `accuracy.py` rejects any configured limit above the checked-in source ceiling.

The policy requires nominal, unseen holdout, alternating-sign, and sparse-outlier receipts for both variants. Collect each on the actual CUDA/Transformer Engine host without accepting a benchmark result:

```bash
python -m labs.kv_cache_compression.calibrate_accuracy --variant fp8 --cohort nominal --seed 2026 --output /tmp/kv-fp8-nominal-2026.json
python -m labs.kv_cache_compression.calibrate_accuracy --variant nvfp4 --cohort nominal --seed 2026 --output /tmp/kv-nvfp4-nominal-2026.json
```

`qualify_accuracy.py` requires the complete matrix and retains every failure reason. These commands collect error metrics only; they do not accept output or claim a speedup. After the matrix passes, select the same policy for the ordinary pair:

```bash
AISP_KV_CACHE_ACCURACY_POLICY="$PWD/labs/kv_cache_compression/accuracy_policy.json" python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
```

The [September 8 B200 report](../../../docs/reviews/2026-09-08-b200-remaining-followthrough.md#further-optimization-work) records all ten arithmetic cases passing at source `32030e6`, with unchanged limits, and one complete profiled pair at 1.106x on Torch 2.9.1+cu130 / Transformer Engine 2.9.0+70f5366. An unprofiled old/new A/B/B/A screen on that same runtime gives cached FP8/NVFP4 ratios of 1.1062x and 1.1064x; it is one seed and two mirrored comparisons. These are measurements for the recorded source and runtime, not qualification of another installation or application quality. Historical receipts were not used to widen the ceilings.

Minimal Nsight Compute profiling defaults to `app-range` for both arms: it collects the five minimal metrics over the complete selected NVTX range. The result is aggregate range counters, not per-kernel counters or an ordinary latency sample. Kernel replay timed out for both arms on this workload; an explicit `--ncu-replay-mode kernel` still overrides the lab preference. For other metric sets, select a compatible replay mode explicitly.

## Learning Goals
- Compare FP8 and NVFP4 projection GEMMs with the same BF16 KV cache storage.
- Qualify predeclared format-scale arithmetic ceilings across nominal, holdout, and edge cohorts.
- Keep allocated storage bytes separate from compute precision and latency.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_kv_cache.py`, `optimized_kv_cache_nvfp4.py` | FP8/NVFP4 compute benchmark pair with BF16 cache storage. |
| `kv_cache_common.py` | Shared attention workload and cache allocation. |
| `accuracy.py`, `accuracy_policy.json`, `calibrate_accuracy.py`, `qualify_accuracy.py` | Independent full-cache reference, source-bounded policy, measurement-only driver, and retained-receipt qualifier. |
| `ACCURACY_REQUIREMENTS.md` | Threshold rationale, claim boundary, and exact serial B200 qualification plan. |

## Collecting Accuracy Measurements
Run on the actual CUDA/Transformer Engine host, preserving target and workload metadata.
```bash
python -m labs.kv_cache_compression.calibrate_accuracy --variant fp8 --cohort nominal --seed 2026 --output /tmp/kv-fp8-nominal-2026.json
python -m labs.kv_cache_compression.calibrate_accuracy --variant nvfp4 --cohort nominal --seed 2026 --output /tmp/kv-nvfp4-nominal-2026.json
```
- These collect measurement-only errors. Run the complete matrix in `ACCURACY_REQUIREMENTS.md` and pass `qualify_accuracy.py` before selecting the checked-in policy for an accepted benchmark.

## Validation Checklist
- Require the checked-in source-bounded policy, complete nominal/holdout/edge receipt matrix, and full-output comparisons before accepting timing.
- Reject zeros, corruption, non-finite values, aliasing, and shape mismatches using the independent reference.
- Verify allocated cache storage bytes and the BF16-relative compression ratio of 1.0.

## Notes
- CPU source checks do not qualify CUDA accuracy, Transformer Engine kernels, memory measurements, or performance.
