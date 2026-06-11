<!-- DRAFT — internal review pending. Do NOT post externally without review. -->
<!-- Target: github.com/pytorch/pytorch issues (performance bug) -->

# Issue title

`torch._int_mm dispatches Ampere-era cutlass_80 kernels on sm_103 (GB300): 6.4-13.4x SLOWER than the fp32 tf32 matmul of the same shape`

# Body

### 🐛 Describe the bug

On NVIDIA GB300 (sm_103, Blackwell Ultra), `torch._int_mm` — the INT8
GEMM that backs dynamic-quantization recipes — is dramatically slower
than the *fp32* matmul of the same shape, inverting the entire point of
INT8 quantization. Measured at M=8192, K=N=4096 (torch 2.12.0a0,
CUDA 13.2):

| op | time/call | dispatched kernel (torch profiler) |
|---|---:|---|
| fp32 matmul (tf32 enabled) | 272.3 us | `cutlass3x_sm100_tensorop_s256x256x8tf32gemm_f32_..._2sm_bias_f32_relu` |
| fp16 matmul | 148.7 us | `nvjet_sm103_hsh_128x256_64x6_2x1_2cta_v_bz_NNT` |
| bf16 matmul | 139.1 us | |
| `_int_mm`, rhs column-major | **1751.1 us** | `cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16>` |
| `_int_mm`, rhs row-major | **3645.9 us** | `cutlass::Kernel2<cutlass_80_wmma_tensorop_i161616gemm_s8_forwardCompat_128x128_32x2_nn_align4>` |

Ratios:

```
_int_mm (col-major rhs) vs tf32 :    6.4x slower
_int_mm (row-major rhs) vs tf32 :   13.4x slower
_int_mm (col-major rhs) vs fp16 :   11.8x slower
_int_mm (row-major rhs) vs fp16 :   24.5x slower
```

The kernel names are the diagnosis: while the floating-point paths get
modern Blackwell kernels (`cutlass3x_sm100_*_2sm`, `nvjet_sm103_*`), the
INT8 path falls back to **sm_80 (Ampere) CUTLASS 2.x kernels** — and for
a row-major rhs to an `i161616` **WMMA `forwardCompat`** kernel, i.e. the
slowest compatibility tier. No Blackwell (sm_100/sm_103) INT8 GEMM is
selected.

Practical impact: any INT8 dynamic-quant linear built on `_int_mm`
(e.g. the common `quantize -> _int_mm -> rescale` pattern) is a large
*regression* on GB300 — we measured an INT8-quantized MLP at 0.17x the
speed of its fp32 baseline, dominated by the `_int_mm` calls.

Related but distinct: #165230 reports the row-major-rhs layout penalty
(reproduced here as the 2.08x col-vs-row gap). This issue is about the
sm_103 dispatch itself: **both** layouts land on sm_80-era kernels and
both are several times slower than tf32.

### Repro

Standalone script attached (`repro.py`, no external deps). It times
fp32-tf32 / fp16 / bf16 matmuls and `_int_mm` in both rhs layouts at the
same shape, and prints the dispatched CUDA kernel for each via the torch
profiler. Asserts `_int_mm` (best layout) > 2x slower than tf32 so the
repro fails loudly once a proper kernel is dispatched.

### Expected behavior

INT8 `_int_mm` should be at least competitive with the fp16 matmul on a
part with INT8 tensor-core support, or — if cuBLAS/CUTLASS INT8 coverage
for sm_103 is intentionally absent — fail or warn rather than silently
dispatching a forward-compatibility Ampere kernel that is 13-25x off.

### Versions

* torch 2.12.0a0 (NGC 26.05 container build), CUDA 13.2
* GPU: NVIDIA GB300 (sm_103, Blackwell Ultra)
* Shape: M=8192, K=4096, N=4096 (int8 x int8 -> int32)
