# GB300 block_scaling deepen: the tile_k structural lever (Front H2)

**Date:** 2026-06-11 | **Pod:** <gb300-pod> (GPU 3) | **Base:** 2f7e30f9 + uncommitted GB300 fixes
**Verdict: HONEST NEGATIVE — refuted with implementation-grade evidence.**

The prescribed lever (make `tile_k` configurable; test 384 and 256) was taken to
implementation. The ground truth discovered in the kernel splits the lever into
an impossible half and an implemented-and-measured half:

1. **`tile_k` is not a free parameter.** It cannot be made configurable to 384
   or 256 by moving coupled constants.
2. **The implementable equivalent of the lever's intent** (eliminate the
   zero-padded K work that `tile_k=768` causes at K=1024) **was implemented,
   verified bit-correct, and measured: 0.9994x — exactly parity.** The 31% of
   MMA work the lever would save is not on the kernel's critical path.

## 1. Why tile_k=384/256 are structurally impossible as configuration

The predecessor's static pass modeled `tile_k=768` (line ~269,
`self.mma_tiler = (M, N, 768)`) as a tunable with coupled constants
(TMA slice 384, `sf_buffers_per_tile_k=4`, smem-layout `8`). Reading the full
mainloop shows those constants are not couplings of a parameter — they are the
arithmetic of a fixed instruction:

- **MMA-K is ISA-fixed at 96 FP4 elements (48B).** `sm103_make_blockscaled_
  trivial_tiled_mma` constructs `SM103MmaMXF4NVF4Op((*mma_tiler_mn, 96), ...)`;
  its docstring says "K fixed to 96" (SM103 FP4 Ultra).
- **The AB smem buffer is a K_SW128 swizzle atom: 128B = 256 elements per
  buffer** (`((CTA_M,16),1,8,stages)` = 8 x 16B chunks per row; the `8` is
  128B/16B, a property of the swizzle, not of tile_k).
- **768 = LCM(96, 256)**: the smallest K span where 96-element MMA windows
  realign with 128B buffer boundaries. The mainloop (lines ~1330-1620) is a
  HAND-UNROLLED 8-MMA / 3-buffer period in which MMA2 and MMA5 read across two
  smem buffers via cur/next descriptor pairs
  (`make_desc_and_call_mma(..., sA[cur], sA[next], ...)`), e.g. MMA2 covers
  buffer0 k-blocks 6,7 + buffer1 k-block 0.
- The TMA "384" at lines ~853/859 is **bytes** (FP4 packs 2/byte): one full
  768-element k-tile per `local_tile` step, copied as `buffers_per_k_tile = 3`
  per-buffer TMA copies of 128B-of-K each.
- Consequences:
  - **tile_k=256 is impossible at any swizzle**: 96 does not divide 256.
  - **tile_k=384 requires K_SW64** (LCM(96 elem, 128 elem)=384) — a different
    swizzle atom, retuned TMA boxes (64B rows), and a rewritten 4-MMA unrolled
    mainloop with different cross-buffer wrap coordinates. Not a constant move;
    a mainloop rewrite.

## 2. The stage-starvation rationale is factually wrong

Predecessor model: "at tile_k=768, A+B ~96KB/stage, `_compute_stages` fits only
~2 AB stages; pipeline short AND starved."

Measured (trace-time `AISP_BLOCK_SCALING_DEBUG_STAGES=1`, new in this pass):

```
acc=2 ab=8 sf=6 c=2  mma_tiler=(256,128,768)  mma_sf_tiler=(128,128,192)
sf_buffers_per_tile_k=4  smem_capacity=232448
```

An AB "stage" is one 128B-of-K buffer (A 16KB + B 8KB = 24KB), and **8 stages
fit** — 2.67 k-tiles of lookahead. At K=1024 the whole problem is 6 buffers
deep: the ring can hold the ENTIRE K extent plus a third of the next tile's.
Stage count was never the constraint, so "smaller tile_k => more stages" buys
nothing even where it is implementable.

## 3. What the lever is actually worth: implemented and measured

At K=1024, `k_tile_cnt = ceil(1024/768) = 2` covers 1536 elements — 512 of them
TMA OOB zero-fill. 5.33 of 16 MMAs per output tile multiply pure zeros (33% of
tensor-pipe work). This padding waste is the only real inefficiency tile_k
restructuring could remove (the predecessor's own floor math: removing it
cannot beat the 18us HBM floor anyway).

**Implemented: `AISP_BLOCK_SCALING_TILE_K=256` — a short-tail k-tile path** in
`labs/block_scaling/vendor/sm103_dense_blockscaled_gemm_persistent.py`. When
the dynamic K remainder is 256 elements (K % 768 == 256, the lab's case), the
last k-tile is peeled out of all three warp loops (TMA-AB, TMA-SF, MMA) and
runs a truncated schedule: **2 of 3 AB buffers, 2 of 4 SF stages, 3 of 8 MMAs**
(MMA0-2 cover the 256 valid elements; MMA2's wrap target buffer 1 is loaded as
zero-fill). 16 -> 11 MMAs per output tile (-31%). Steady-state loop bodies are
byte-identical to the incumbent (peeling, not in-loop branching); with the env
unset the codegen is bit-for-bit the incumbent (`cutlass.const_expr` gating).
Ring/mbarrier accounting stays consistent because `PipelineState.reset()` only
resets a commit count — index/phase carry across tiles, so a 5-buffer tile is
legal.

An earlier variant with in-loop dynamic guards (no peeling) was 0.888x — the
guards alone cost 11%. The peeled version costs ~0, isolating the lever's true
value.

### Correctness (before timing)

The lab's software reference is exact for these integer-scaled inputs, so a
correct kernel must match exactly:

```
[768] max_abs_err=0.000000 mean=0.00000000 tol=0.1
[256] max_abs_err=0.000000 mean=0.00000000 tol=0.1
```

(Bit-exactness is expected: the 5 removed MMAs contributed exactly zero and the
accumulation order of the remaining MMAs is unchanged.)

### Kernel A/B (interleaved, 6 rounds x 200 iters, GPU 3, same process)

| round | 768 (us) | 256 short-tail (us) |
|---|---|---|
| 0 | 43.75 | 43.69 |
| 1 | 43.67 | 43.71 |
| 2 | 43.59 | 43.66 |
| 3 | 43.60 | 43.64 |
| 4 | 43.59 | 43.64 |
| 5 | 43.61 | 43.63 |
| **median** | **43.62** | **43.65** |

**Speedup: 0.9994x — parity to within 0.3% on every round.**
%SoL (vs 9.2us FP4-compute floor): 21.1% for both.

### ncu (--set full, fresh A/B same session; incumbent vs short-tail)

| metric | 768 incumbent | 256 short-tail | delta |
|---|---|---|---|
| gpu__time_duration | 44.51 us | 44.93 us | +0.9% (noise) |
| sm__pipe_tensor_cycles_active (% of active) | 34.50% | 24.02% | **-30.4%** |
| inst_executed tensor (hmma subpipe) | 0.54% | 0.38% | -30% |
| lts__throughput (% peak elapsed) | 48.70% | 44.46% | -8.7% |
| stalled_long_scoreboard / issue-active | 6.43 | 5.72 | -11% |
| sm__warps_active | 10.06% | 10.00% | ~0 |

The tensor-pipe activity drops by exactly the removed MMA fraction
(34.50 x 11/16 = 23.7 ~ 24.0) **and kernel time does not move**. This is direct
proof that the zero-pad MMAs — and therefore the entire k-tile compute
structure tile_k could reshape — are fully hidden behind whatever actually
binds the kernel (L2/feed latency + epilogue C traffic; lts ~48%, DRAM floor
18us, scoreboard-dominated stalls on a 10%-occupancy warp set).

### Harness (gate: >=1.05x over ~0.0811 ms + verification passed)

A/B/A sequence, all "verification passed":

| arm (run order) | optimized ms | run id |
|---|---|---|
| pre-edit BEFORE | 0.08115 | 20260611_061715 |
| post-edit, env unset (regression gate) | 0.08178 | 20260611_064517 |
| post-edit, TILE_K=256 | 0.07877 | 20260611_064643 |
| post-edit, env unset (closure) | 0.08101 | 20260611_064829 |

No regression of the committed B38 cudagraph arm (0.08178/0.08101 vs 0.08115
BEFORE, ~1% band). The TILE_K=256 harness point reads 2.8% under the closing
default — within/near the session drift band and contradicted by the
interleaved same-process kernel A/B (0.9994x), so it is treated as drift, and
in any case 0.08101/0.07877 = 1.028x < the 1.05x gate. The lever cannot clear
the gate: its entire mechanism (MMA-work removal) is measured as off the
critical path.

## 4. Why this closes the tile_k/K_SW64-384 branch too

The only mechanisms a tile_k=384 (K_SW64) rewrite could exploit are (a) less
zero-pad MMA work — 12 MMAs/tile vs the short-tail's 11, i.e. strictly worse
than what was measured at parity here; (b) more AB stages — refuted, 8 already
fit and only 6 are ever in flight; (c) it would additionally halve TMA box rows
to 64B, degrading load efficiency. With (a) measured at 0.9994x and (b)
factually false, the deep rewrite has no remaining mechanism. **Do not spend a
front on it.**

## 5. Exact diff and artifacts

Changed file (only one):
- `labs/block_scaling/vendor/sm103_dense_blockscaled_gemm_persistent.py`
  - md5 (pod == local Mac mirror): `707ec61b83d0bc722ea22ddfb1b1f8cf`
  - incumbent backup: `/tmp/frontH2/sm103_dense_blockscaled_gemm_persistent.py.bak`
    (md5 `8378991cc85349fb6f81bf1c8d2861be`)
  - Edits: (1) `import os`; (2) `__init__`: `enable_short_tail` /
    `debug_stages` env flags + structural documentation; (3) trace-time stage
    print at end of `_setup_attributes`; (4) in-kernel `short_tail_active`
    predicate after `k_tile_cnt`; (5-7) last-k-tile peel of the TMA-AB, TMA-SF
    and MMA warp loops with truncated tail bodies, all under
    `cutlass.const_expr(self.enable_short_tail)` so the default path is
    bit-identical.
- Probe/A-B scripts: `/tmp/frontH2/ab_tile_k.py`, `/tmp/frontH2/probe_stages.py`
- ncu reports: `/tmp/frontH2/incumbent768.ncu-rep`, `/tmp/frontH2/tail256.ncu-rep`
- Harness runs: `artifacts/runs/20260611_061715__*` (BEFORE),
  `20260611_064517__*` (default regression), `20260611_064643__*` (TILE_K=256),
  `20260611_064829__*` (default closure)

## 6. Named next lever

The kernel sits at 44.5us vs an 18us HBM floor with tensor pipe now provably
slack. The remaining time is feed/epilogue bound: lts ~48% with A/B re-read
traffic of ~786MB/GEMM (4096 CTA-tiles x 192KB) against 8MB of unique operand
bytes. The highest-EV lever is **L2-traffic reduction via larger output tiles**:
`mma_tiler_mn=(256,256)` (+ cluster (2,2) / SFB multicast) halves B re-reads
per output element and is already env-exposed
(`AISP_BLOCK_SCALING_MMA_TILER_MN`) — `can_implement` accepts N=256 per the
kernel's documented constraints. Second: epilogue/TMA-store stage tuning
(c=2 stages, 128MB C write = 16us of the 18us floor). Both attack the actual
binding resource; no k-dimension lever does.
