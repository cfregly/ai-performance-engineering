<!-- DRAFT — internal review pending. Do NOT post externally without review. -->
<!-- Target: github.com/NVIDIA/cutlass (docs/examples PR) -->

# PR title

`[docs/examples] Blackwell tutorial 04 (2SM MMA): fix stale B-operand print annotations (the 2x1SM atom splits B N/2-per-CTA) and document the expect-tx pitfalls that deadlock`

# PR body

### What

Two documentation problems in
`examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`, both of which
cost us real debugging time on GB300 (one of them as a 25-minute
GPU-pegged mbarrier deadlock, hit independently by two separate kernels):

#### 1. Stale print annotations claim a full-N per-CTA B tile

The `SM100_MMA_F16BF16_2x1SM_SS` atom **splits the B operand N/2-per-CTA
across the CTA pair**, but the inline `// printed:` annotations in the
tutorial still show full-N shapes. With the tutorial's own TiledMMA
(2x1SM 256x256 atom, MmaTiler 256x256x64), what `partition_shape_B` /
`tile_to_mma_shape` actually return today (host-only check, CUTLASS main
and 4.2.0 alike — `snippet.cu` attached):

| annotation in the file | actually printed |
|---|---|
| `mma_shape_B: ((_256,_16),_1,_4)` | `((_128,_16),_1,_4)` |
| `sB_layout: Sw<3,4,3> o ... ((_256,_16),_1,_4):((_64,_1),_0,_16)` | `... ((_128,_16),_1,_4):((_64,_1),_0,_16)` |
| `tCsB / tCgB: ((_256,_16),...)` | `((_128,_16),...)` |
| `tBsB: ... o ((_16384,_1)):((_1,_0))` | `((_8192,_1))` |

(Also: the shape comments on `tCsB` / `tCrB` read
`(MmaB, NumMma_M, NumMma_K, Tiles_K)` — the second mode should be
`NumMma_N`.)

The hardware behavior matches the corrected prints, not the annotations.
On GB300 (sm_103) we confirmed with ncu on a kernel built from this
tutorial: per-CTA smem footprint is A-half 16 KiB + B-half 16 KiB per
stage at a 256-wide N tile (e.g. 64 KiB/CTA at 2 stages; a full-N B would
be 96 KiB and would change the reported Block Limit Shared Mem from 3 to
2 at the 128-col config), and the mbarrier transaction-byte balance only
closes with the halved B slice. Readers who trust the current annotations
over-budget smem 1.5x and mis-derive the expect-tx count (see below).

#### 2. The expect-tx formula's two silent-deadlock pitfalls deserve a warning comment

The tutorial's formula is correct:

```c++
int tma_transaction_bytes = size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA))
                          + size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));
```

but nothing warns about the two natural-looking modifications that each
produce a barrier that NEVER fires — i.e. a deterministic kernel hang with
no error, the worst failure mode to debug:

* **No participant multiplier.** `tma_partition`'s multicast result is an
  *offset view* into the stage buffer, not a shrunken tensor:
  `sizeof(make_tensor_like(slice))` is already the full byte count
  delivered into this CTA for that operand. Scaling it by the number of
  multicast participants (e.g. `kClusterN` when A is multicast across the
  cluster N-mode) over-expects the barrier. We hit exactly this twice
  independently: once on a single-SM cluster-2 B-multicast kernel
  (expect-tx armed 80 KiB vs 48 KiB delivered -> hang, localized via
  cuda-gdb attach) and once on a (2,2,1) A-multicast variant of this very
  tutorial pattern (`tma_bytes = 2 * (kClusterN * sizeof(tAsA_0) + ...)`
  -> 25-minute GPU-pegged hang; deleting the `kClusterN` factor fixed it,
  `rel_err 0.0` at every size after).
* **B is the N/2 slice.** Deriving the B term from the full-N tile (which
  the stale annotations of problem 1 suggest) over-expects by the same
  mechanism.

Two adjacent facts worth stating in the same comment (both verified on
sm_103): with `SM100_TMA_2SM_LOAD` both CTAs of the pair issue loads
against their own barrier handle and the hardware redirects every arrival
to the even (leader) CTA's barrier, so only the leader calls
`set_barrier_transaction_bytes` — and a peer-CTA TMA completing *before*
the leader's expect-tx is legal (the mbarrier transaction count may go
transiently negative).

### Proposed change

1. Refresh the stale print annotations in `04_mma_tma_2sm_sm100.cu`
   (table above) and fix the `NumMma_M`/`NumMma_N` comment typos.
2. Add a short warning comment above the `tma_transaction_bytes`
   computation:

   ```c++
   // NOTE: each tma_partition slice is an OFFSET VIEW into this CTA's
   // stage buffer; sizeof(make_tensor_like(slice)) is already the full
   // bytes delivered into this CTA. Do NOT multiply by the number of
   // multicast participants, and do NOT size B from the full-N tile (the
   // 2x1SM atom gives each CTA an N/2 slice) — both over-expect the
   // barrier, which then never fires: a silent deadlock, not an error.
   // The only multiplier is size<0>(cluster_layout_vmnk) (== 2): the
   // leader's barrier counts the PAIR's bytes for 2SM TMA loads.
   ```
3. Optionally mirror one sentence on 2x1SM operand residency
   (A: M-half per CTA, B: N/2 per CTA, C/TMEM: own 128 rows x N) into the
   Blackwell functionality docs.

No code behavior changes — annotations and comments only. Happy to split
1 and 2 into separate commits if preferred.

### Evidence / environment

* Host-only print check (`snippet.cu`, attached): builds with
  `nvcc -std=c++20 -arch=sm_103a -I$CUTLASS_DIR/include`, no GPU needed;
  output above reproduced against both CUTLASS 4.2.0 and current main
  headers (2026-06).
* Hardware confirmation: NVIDIA GB300 (sm_103), CUDA 13.2 (V13.2.78),
  driver 580.159.03 — ncu smem-footprint/occupancy readings and mbarrier
  tx-byte balance on a warp-specialized 2SM GEMM built from this tutorial
  pattern; both deadlock incidents reproduced and fixed by the one-line
  formula corrections described above.
