<!-- DRAFT — internal review pending. Do NOT post externally without review. -->
<!-- Target: github.com/NVIDIA/cutlass (docs/examples PR) -->
<!-- Plus optional ptxas question via the NVIDIA bug tracker (text at bottom) -->

# PR title

`[docs/examples] Blackwell cute tutorials: narrow TMEM_LOAD atoms (32dp32b1x) carry a large per-load lowering cost — prefer wider atoms (32dp32b32x) in t2r epilogues`

# PR body

### What

All five Blackwell CuTe tutorials demonstrate the TMEM->register epilogue
with the narrowest TMEM_LOAD copy atom:

```c++
// examples/cute/tutorial/blackwell/01_mma_sm100.cu (and 02..05 likewise)
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
```

On sm_103 (CUDA 13.2 ptxas; also observed on the 13.x line generally) every
`tcgen05.ld.sync.aligned.32x32b.x1.b32` this atom emits is lowered to a
**per-load convergence-helper subroutine call** in SASS:

```
LEPC R20, 0x460 ;
CALL.ABS.NOINC R2 ;
WARPSYNC.ALL ;
...
LDTM R2, tmem[UR4] ;
```

For a 128x256 fp32 accumulator that is 256 loads per thread = 256 helper
calls per warp (~60 cycles each), which can dominate a small kernel's fixed
cost. In a warp-specialized 2048^3 fp16 GEMM kernel built from these
tutorials we measured the t2r phase at **7.65-7.81 us of a 23.8-24.2 us
kernel** (in-kernel `%globaltimer` stamps, per-CTA medians, 128 CTAs).
Switching the one line to `SM100_TMEM_LOAD_32dp32b32x` (8 loads/thread)
cut t2r to **0.42 us** and the whole kernel from **23.8 -> 16.0 us (1.49x)**,
bit-identically (`torch.equal` on the outputs — the atom only changes how
many columns one instruction moves; each thread keeps the same
(row, all-columns) fragment, so per-element conversion and store mapping
are untouched). Width sweep at the same shape: `x16` ties `x32` (0.51 vs
0.42 us); `x128` REGRESSES (0.99 us t2r + slower writeback — the 128-output
asm serializes register writeback), so widest is not best; x32 was the
sweet spot measured.

A tutorial that demonstrates the 1x atom without comment teaches a ~1.5x
performance bug as the canonical epilogue.

### Proposed change

Minimal, docs-only (no behavior change to any library kernel):

1. In tutorials 01-05, either switch the epilogue atom to
   `SM100_TMEM_LOAD_32dp32b32x` (and refresh the affected print
   annotations), or keep `32dp32b1x` for pedagogical simplicity and add a
   short comment block, e.g.:

   ```c++
   // PERF NOTE: 32dp32b1x is the simplest TMEM_LOAD atom but issues one
   // tcgen05.ld per accumulator column; ptxas (CUDA 13.x) lowers each load
   // through a per-load convergence-helper call, so the t2r phase can
   // dominate small/medium kernels. Prefer a wider atom (e.g.
   // SM100_TMEM_LOAD_32dp32b32x: 32 columns per instruction) in real
   // epilogues; on sm_103 we measured 1.49x on the whole kernel from this
   // one line. Very wide atoms (x128) can regress on register-writeback
   // serialization — sweep the width for your tile shape.
   ```

2. Optionally, a sentence in
   `media/docs/cpp/cute/0y_tmem_tensor.md` (or the blackwell functionality
   doc) noting atom width as a first-class performance knob.

### Standalone evidence

A ~120-line reproducer (attached: `tmem_load_atom_repro.cu`) containing
only the TMEM alloc + `make_tmem_copy` t2r + store (no mainloop, no MMA),
built with `nvcc -std=c++20 -arch=sm_103a` against current CUTLASS headers:

| | `32dp32b1x` | `32dp32b32x` |
|---|---|---|
| `LDTM` in SASS | 256 | 8 |
| `CALL.ABS.NOINC` / `LEPC` | 256 / 256 | 8 / 8 |
| kernel time (GB300, 1 CTA, 2000 launches) | 10.91 us | 5.86 us |

`cuobjdump -sass` one-liners to see it:

```
cuobjdump -sass repro_1x  | grep -c "CALL.ABS.NOINC"   # 256
cuobjdump -sass repro_32x | grep -c "CALL.ABS.NOINC"   # 8
cuobjdump -sass repro_1x  | grep -B7 "LDTM" | head     # LEPC/CALL/WARPSYNC wrap
```

### Environment

* NVIDIA GB300 (sm_103), driver 580.159.03
* CUDA 13.2 (nvcc V13.2.78)
* CUTLASS: reproduced against both 4.2.0 headers and current main
* Kernel-level numbers: fp16 2048^3, 128x256 tile, 4-stage warp-specialized
  pipeline; CUDA-graph interleaved A/B, 7 reps/arm, zero distribution
  overlap; ncu cross-check (`sm__pipe_tensor_cycles_active` 20.4% -> 37.9%
  of elapsed)

---

# Optional companion filing: NVIDIA bug tracker (ptxas question)

> **Title:** ptxas (CUDA 13.2, sm_103) lowers each narrow tcgen05.ld
> (32x32b.x1) through a per-load LEPC + CALL.ABS.NOINC convergence-helper —
> is the per-load call intended?
>
> **Body:** Compiling a minimal CUTLASS CuTe kernel that drains a 128x256
> fp32 TMEM accumulator with `SM100_TMEM_LOAD_32dp32b1x` (256
> `tcgen05.ld.sync.aligned.32x32b.x1.b32` per thread), ptxas emits a
> LEPC + CALL.ABS.NOINC + WARPSYNC helper-call sequence around EVERY load
> (256 call sites in the unrolled SASS), ~60 cycles each, which dominates
> the kernel: measured 7.7 us of a 24 us GEMM kernel on GB300. The x32 form
> of the same instruction family amortizes the helper 32x and removes the
> cost. Reproducer + SASS attached (same files as the CUTLASS PR above).
> Question: is the per-load helper call required for convergence
> correctness around tcgen05.ld, or can ptxas hoist/inline it for
> back-to-back loads in the same convergence region? If it is required,
> consider documenting the cost in the PTX ISA notes for tcgen05.ld so
> kernel authors pick wider shapes.
