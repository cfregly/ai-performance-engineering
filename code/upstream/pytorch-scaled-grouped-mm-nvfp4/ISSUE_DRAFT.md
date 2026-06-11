<!-- DRAFT — internal review pending. Do NOT post externally without review. -->
<!-- Target: github.com/pytorch/pytorch issues (feature request, "🚀 Feature" template) -->

# Issue title

`torch._scaled_grouped_mm: no NVFP4 (float4_e2m1fn_x2) support — grouped NVFP4 workloads are forced to N separate _scaled_mm launches although sm_100/sm_103 hardware and CUTLASS support grouped NVFP4`

# Body

### 🚀 The feature, motivation and pitch

`torch._scaled_mm` supports NVFP4 (`torch.float4_e2m1fn_x2` operands with
`float8_e4m3fn` block scales) and on Blackwell Ultra it dispatches a real
cuBLASLt FP4 tensor-core kernel — verified on a GB300 (sm_103), the
dispatched kernel is:

```
nvjet_sm103_oohsh_128x128_256x7_2x2_2cta_v_bz_Avec16UE4M3_Bvec16UE4M3_TNT
```

`torch._scaled_grouped_mm`, however, accepts FP8 (tensorwise/rowwise) and
MXFP8 only. The same operands in NVFP4 are refused at validation:

* with the NVFP4-style `float8_e4m3fn` block scales:
  ```
  ValueError: For FP8 tensorwise and rowwise, both scales must both be float32 tensors. For MXFP8, scales must both be float8_e8m0fnu tensors.
  ```
* with fp32 scales (to get past the scale check):
  ```
  ValueError: Expected mat_a to be Float8_e4m3 matrix got Float4_e2m1fn_x2
  ```

**Why it matters (MoE expert GEMMs):** for a grouped NVFP4 workload the
only library path is one `_scaled_mm` launch per (input, group). Measured
on GB300 (sm_103, torch 2.12.0a0, CUDA 13.2) at an MoE-expert shape —
g=2 groups of M=(192, 320), N=3072, K=4096, batched over 15 inputs =
30 GEMMs per call:

```
30 separate _scaled_mm launches: 1068 us/call GPU span (35.6 us/GEMM), 181 TFLOP/s achieved
```

Each GEMM is tiny (M=192/320), so the loop is launch-bound and the GPU is
work-starved (a single GEMM of this size cannot fill the machine). In our
testing on the same workload, a hand-written fused tcgen05
grouped kernel (one launch for all 30 sub-problems) measured 0.224 ms vs
0.701 ms for the 30-launch `_scaled_mm` loop — **3.1x wall-clock** — purely
from batching, with the same instruction class the library already uses.
A grouped NVFP4 `_scaled_grouped_mm` would give that batching win without
a custom kernel.

**Hardware/library capability evidence:**

* cuBLASLt already runs single-GEMM NVFP4 on sm_103 (the `nvjet_sm103 ...
  Avec16UE4M3_Bvec16UE4M3` kernel `_scaled_mm` dispatches above).
* CUTLASS ships grouped block-scaled NVFP4 GEMM for Blackwell, including
  an sm_103-specific example: `examples/90_sm103_fp4_ultra_grouped_gemm`
  (operands `cutlass::float_e2m1_t`, scale factors `cutlass::float_ue4m3_t`),
  plus `examples/75_blackwell_grouped_gemm_block_scaled` for sm_100.
* Precedent inside PyTorch: MXFP8 grouped support was added to
  `_scaled_grouped_mm` in #162209; #156238 asks for the SM100+ grouped
  upgrade; and #174699 (open) adds a CuTeDSL-based block-scaled grouped MM
  (its tests/benchmarks include an `nvfp4` format) registered into
  `torch.nn.functional.scaled_grouped_mm` — so NVFP4 grouped appears to be
  on the roadmap already. This issue asks to make it land for
  `_scaled_grouped_mm`/`F.scaled_grouped_mm` on sm_100a **and sm_103a**,
  and documents the measured cost of not having it.

### Additional context: FP8 grouped is also broken on sm_103 in this build

While testing the supported dtype as a control, we found that on sm_103
the FP8 rowwise grouped call passes validation but the dispatched CUTLASS
kernel traps at runtime:

```
ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.
...
AcceleratorError: CUDA error: unspecified launch failure
```

(and the CUDA context is poisoned afterwards). So in torch 2.12.0a0 there
is currently **no working `_scaled_grouped_mm` path at all on sm_103** —
the grouped kernel appears to be compiled with arch-conditional
instructions for a different `sm_1xxa` target and is not guarded against
running on sm_103. Happy to split this into a separate bug report if that
is preferred.

### Repro

Standalone script attached (`repro.py`, no external deps). It (1) runs the
per-group NVFP4 `_scaled_mm` and prints the dispatched kernel, (2) captures
both NVFP4 refusal messages from `_scaled_grouped_mm`, (3) times the
30-launch fallback, and (4) demonstrates the sm_103 FP8 grouped trap in a
child process (the trap poisons the CUDA context). Output on GB300:

```
[1] per-group NVFP4 _scaled_mm: OK (M=(192, 320), N=3072, K=4096, out fp16)
    dispatched kernel: nvjet_sm103_oohsh_128x128_256x7_2x2_2cta_v_bz_Avec16UE4M3_Bvec16UE4M3_TNT
[2] torch._scaled_grouped_mm with NVFP4 operands:
    e4m3 block scales (the NVFP4 recipe):
        REFUSED -> ValueError: For FP8 tensorwise and rowwise, both scales must both be float32 tensors. For MXFP8, scales must both be float8_e8m0fnu tensors.
    fp32 rowwise-style scales:
        REFUSED -> ValueError: Expected mat_a to be Float8_e4m3 matrix got Float4_e2m1fn_x2
[3] forced fallback, 30 separate _scaled_mm launches per batched call:
    GPU span :   1068.1 us/call ( 35.60 us/GEMM)
    host wall:   1069.6 us/call
    achieved :    180.9 TFLOP/s (launch-bound; each GEMM is a tiny M=(192, 320) problem)
[4] FP8 rowwise _scaled_grouped_mm (child process):
    FAILED (rc=1, arch-conditional trap=yes)
    ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.
```

### Alternatives

* Keep launching per-group `_scaled_mm` (measured above: launch-bound,
  ~3x away from a fused grouped launch at MoE shapes).
* Hand-written CUDA/CUTLASS grouped kernel (what we did internally; works,
  but every MoE-on-NVFP4 user should not need to).

### Versions

* torch 2.12.0a0 (NGC 26.05 container build), CUDA 13.2
* GPU: NVIDIA GB300 (sm_103, Blackwell Ultra)
* Shapes: MoE expert grouped GEMM class (g=2, M=192/320, N=3072, K=4096,
  15 inputs per batched call)
