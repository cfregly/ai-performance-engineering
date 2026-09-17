# NVFP4 GEMM on GB300

This directory contains the complete optimization ladder for an NVFP4 GEMM
with FP16 output. The kernels are plain CUDA C++ plus inline PTX and target
NVIDIA GB300/B300 (`sm_103a`). cuBLASLt is used only as the reference and
benchmark baseline.

## Build and run

From the repository root, build the final kernel:

```bash
make nvfp4
./out/nvfp4 8192 8192 8192 --rounds 5 --cooldown-sec 8
```

Every invocation checks correctness and determinism before benchmarking. The
final result includes our PFLOP/s, cuBLASLt PFLOP/s, their ratio, and paired
rounds won.

Build every optimization rung:

```bash
make nvfp4-ladder
```

Run one rung, or run the whole ladder:

```bash
./out/nvfp4-r0 8192 8192 8192 --rounds 5 --cooldown-sec 8

for rung in {0..9}; do
  ./out/nvfp4-r${rung} 8192 8192 8192 \
    --rounds 5 --cooldown-sec 8
done
```

Use `--check-only` to stop after the correctness gates:

```bash
./out/nvfp4-r9 8192 8192 8192 --check-only
```

For a quick kernel-only timing signal, keep the correctness gates but skip the
timed cuBLASLt leg:

```bash
./out/nvfp4-r9 8192 8192 8192 \
  --ours-only --rounds 5 --cooldown-sec 8
```

## Optimization ladder

Each header is a complete, directly editable implementation snapshot.

| Rung | Change |
| --- | --- |
| `r0` | Simple Blackwell baseline: K=192 feed and `threadIdx.x % 32` lane checks |
| `r1` | Read `%laneid` directly for single-lane work |
| `r2` | Fill K=256 A/B windows and pipeline their releases |
| `r3` | Overlap two TMEM accumulator buffers |
| `r4` | Use 256-bit output stores |
| `r5` | Add output cache hints |
| `r6` | Promise the exact 224-thread block shape |
| `r7` | Skip dead tail work |
| `r8` | Fold compatible tails with K64 MMAs |
| `r9` | Add L2-side ownership and per-shape tile ordering |

To modify a rung, edit its header and rebuild only that target:

```bash
$EDITOR gb300/nvfp4/gemm3.cuh
make nvfp4-r3
```

## Benchmark modes

The final kernel supports all four measurement modes used for the published
comparison:

```bash
gb300/nvfp4/run_bench_modes.sh current
gb300/nvfp4/run_bench_modes.sh clock-pinned
gb300/nvfp4/run_bench_modes.sh sustained
gb300/nvfp4/run_bench_modes.sh triton
```

Run all four with:

```bash
gb300/nvfp4/run_bench_modes.sh all
```

The shape defaults to `8192 8192 8192`; pass another shape after the mode.
`clock-pinned` requires passwordless `sudo nvidia-smi` and always resets the
clock through an exit trap. `sustained` takes several minutes.

The modes are:

- `current`: rotating inputs, alternating ours/cuBLASLt order, five paired rounds.
- `clock-pinned`: the same protocol with the SM clock fixed at 1305 MHz.
- `sustained`: alternating 60-second blocks and post-warmup tail latency.
- `triton`: hot-loop timing with a 256 MiB L2 clear before each measured launch.

## Schedulers

`r9` exposes the scheduling controls used to isolate the final optimization:

```bash
for schedule in \
  raster pocket-8x4 pocket-8x8 hilbert \
  owned-plain owned-pocket-8x4 owned-pocket-8x8 hilbert-in-owned auto
do
  ./out/nvfp4-r9 8192 8192 8192 \
    --schedule "$schedule" --ours-only --rounds 5 --cooldown-sec 8
done
```

Unprefixed schedules change only tile order. `owned-*` schedules also assign A
reuse according to the runtime L2-side census. `auto` selects the final
per-shape policy; for the 8192 cube it uses owned 8x8 pockets.

## Benchmark contract

- Every rung and cuBLASLt receives byte-identical packed E2M1 A/B inputs and
  byte-identical standard `VEC16_UE4M3` scale buffers.
- Both implementations produce row-major FP16 `[M, N]` output.
- Timed launches rotate enough operand sets to avoid self-warming L2.
- Ours and the first cuBLASLt recommendation alternate first/second order
  across paired rounds.
- Throughput uses logical `2MNK` work, including for early rungs that execute
  zero-filled tail MMAs.
- `--cooldown-sec 8` waits for the SM clock to return to its reference range;
  it aborts rather than recording a result if the clock does not recover.

## Requirements

- NVIDIA GB300/B300
- CUDA 13.1 or a compatible toolkit with the required `tcgen05` forms
- cuBLASLt and the CUDA Driver library
- GNU Make

The build deliberately uses
`-gencode arch=compute_103a,code=sm_103a`. Plain `-arch=sm_103a` can emit
family-generic PTX that rejects architecture-specific `tcgen05` instructions.
