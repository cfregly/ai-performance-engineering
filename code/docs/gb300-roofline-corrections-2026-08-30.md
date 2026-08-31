# Corrected peak definitions and historical roofline interpretation

This is an arithmetic/source correction dated 2026-08-30, not a new GPU run.
Historical timings, profiler counters, success claims, and A/B ratios in the linked
reports remain historical evidence. Their old FP8 and FP16/BF16 theoretical
denominators must not be reused. Rebuilding the modified kernels and collecting
new correctness, clock, profiler, and repeated timing evidence remain required.

## Published peaks, with SKU and scope

| Product and scope | Dense FP8 TFLOP/s per GPU | Dense FP16/BF16 TFLOP/s per GPU | Derivation |
|---|---:|---:|---|
| HGX B200, eight B200 GPUs | 4,500 | 2,250 | Published sparse system figures: 72/36 PFLOP/s; divide by 8 GPUs and by 2 for dense |
| GB300 NVL72, 72 GPUs | 5,000 | 2,500 | Published sparse system figures: 720/360 PFLOP/s; divide by 72 GPUs and by 2 for dense |
| H100 SXM | 1,979 | approximately 989.5 | Published sparse per-GPU figures: 3,958/1,979 TFLOP/s; divide once by 2 |

Sources: [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/),
[NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/),
and [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/).
The sparsity footnotes apply to Tensor Core figures, not FP32 CUDA-core figures.
B200 FP32 is 600/8 = 75 TFLOP/s per GPU. H100 constants already represented dense
rates; halving them again, as suggested in one audit passage, would be incorrect.
HGX B300, GB200 NVL72, and GB300 NVL72 are distinct products: do not transfer one
product's peak to another because their compute capabilities match.

GB300 NVL72's published dense NVFP4 total is 1,080 PFLOP/s: 15 PFLOP/s per GPU.
Its 576 TB/s aggregate GPU memory bandwidth is 8 TB/s per GPU. These two old
denominators therefore do not receive the FP8/FP16 correction. The old FP8
7.5 PFLOP/s and FP16/BF16 3.75 PFLOP/s denominators overstated the corresponding
GB300 NVL72 dense peaks by 1.5. External `perf-tune-report/configs/sol-ceilings.yaml`
was referenced by the reports but is not present in this checkout; it has not
been changed or verified here.

## Reinterpreting the retained reports

Only a percentage computed as measured TFLOP/s divided by the old 7,500 or 3,750
denominator changes by a factor of 1.5. This rule does **not** apply to Nsight's
native tensor-utilization counters, memory percentages, speedup ratios, NVFP4,
or comparisons against a separately measured 1.9 PFLOP/s reference.

| Retained measured numerator | Old denominator and percentage | Correct denominator and percentage |
|---|---|---|
| FP8 2,481 TFLOP/s | 7,500; 33.08% | 5,000; 49.62% |
| FP8 3,432 TFLOP/s | 7,500; 45.76% | 5,000; 68.64% |
| FP16 1,171 TFLOP/s | 3,750; 31.23% | 2,500; 46.84% |
| FP16 dual-CTA 1,311.5 TFLOP/s | 3,750; 34.97% | 2,500; 52.46% |
| FP16 cuBLAS 1,818.7 TFLOP/s | 3,750; 48.50% | 2,500; 72.75% |

The numerators come from the retained [runbook](gb300-runbook.md) and
[occupancy report](gb300-gemm-occupancy-rewrite.md); this table recalculates their
percentages without recertifying the measurements. Headroom predictions based
on the inflated denominators need a new evaluation, not a relabeled win.

## Byte-accounting and architecture corrections

Local copy bandwidth counts both bytes read and bytes written. One-way peer
bandwidth counts payload once. Rates are decimal GB/s even when allocations use
MiB/GiB. This follows [CUDA's effective-bandwidth definition](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effective-bandwidth-calculation).
`benchmark_peak` now emits versioned byte/timing provenance; the target loader
rejects old or inconsistent HBM artifacts instead of silently doubling them.
Original artifact files remain unchanged. Default overall HBM policy values
3/3.5/4 TB/s become 6/7/8 TB/s in read+write units; chapter targets with different
or unknown provenance are not mechanically changed. This is policy arithmetic,
not evidence that any GPU achieved these values. A cache-sized copy alone also
does not prove cache residency, and peer access alone does not prove NVLink.

The [CUDA GPU table](https://developer.nvidia.com/cuda/gpus) distinguishes
10.0 B200/GB200, 10.3 B300/GB300, 12.0 RTX 50/RTX PRO, and 12.1 GB10.
Compute capability does not establish SKU-specific SM count, memory bandwidth,
or CPU/interconnect topology. [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
has GB10 with LPDDR5x at 273 GB/s; the old generic major-12 HBM3e/Grace/C2C/500-GB/s
description was not valid. The architecture helper preserves actual capability
and does not rewrite 12.1 to 12.0 or rewrite generated PTX to hide unsupported
instructions. SM120/121 do not acquire tcgen05 support by changing a label.

## Execution gates still required

CPU tests validate arithmetic, artifact acceptance/rejection, and metadata only.
Run the GPU cases in `tests/test_audit_wave1_peak_metrics.py` on the actual target
and pinned stack. The FP8 path uses real `torch._scaled_mm`; NVFP4 uses the actual
`NVFP4BlockScaling()` recipe documented by [Transformer Engine 2.10](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.10/user-guide/examples/fp8_primer.html).
Missing recipes/backends fail explicitly; no FP16 result is labeled FP8 or FP4.
Recipe construction and API presence are not kernel, output-accuracy, or peak
performance qualification. Full FP4 numerical accuracy calibration, low-precision
kernel inspection, L2 residency profiling, actual-link topology/counters, and
repeatable bandwidth/performance validation remain HOLD without target receipts.
