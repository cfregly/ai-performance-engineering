# GB300: In-file vectorized-router override for `labs/moe_optimization_journey:moe_pad_quant`

**Verdict: WIN — B34 "out of scope" was a mis-bank.** The OPTIMIZED arm builds its own
model instance, so it can legitimately replace its level-4 router dispatch from within
`optimized_moe_pad_quant.py` without touching `moe_model.py`, the ch19 helpers, or the
baseline arm. Doing so removes every `torch.nonzero`, every per-expert DtoH sync, and the
entire ~1 ms un-graphable host gap — the whole model now executes as a single
cudagraph-captured CompiledFxGraph.

- Date: 2026-06-11 | Pod: `<gb300-pod>` (<namespace>), GPU 2, GB300 (sm_103)
- Toolchain: torch 2.12.0a0+5aff3928d8.nv26.05, CUDA 13.2, Triton 3.7 (max-autotune
  tcgen05 guard intact: `get_optimal_compile_mode` -> "default" -> promoted to
  "reduce-overhead", unchanged)
- Base: 2f7e30f9 + uncommitted GB300 fixes; only file changed:
  `labs/moe_optimization_journey/optimized_moe_pad_quant.py`

## Before / after (harness: `python -m cli.aisp bench run --targets labs/moe_optimization_journey:moe_pad_quant --profile none --single-gpu`, GPU 2)

| run | baseline_ms | optimized_ms | speedup | verify |
|---|---|---|---|---|
| BEFORE (20260611_052913) | 10.818 | 4.2220 | 2.56x | PASS (max_diff 0.105) |
| AFTER rep1 (053228) | 10.152 | 0.3112 | 32.62x | PASS (max_diff 0.1025) |
| AFTER rep2 (053345) | ~10.9 | 0.3091 | 23.16x* | PASS (max_diff 0.1025) |
| AFTER rep3 (053402) | 10.888 | 0.3843 | 28.33x | PASS (max_diff 0.1025) |
| AFTER rep4 (053403) | ~10.9 | 0.3319 | 25.88x | PASS (max_diff 0.1025) |
| AFTER rep5 (053430) | 11.433 | 0.3237 | 35.33x | PASS (max_diff 0.1025) |

*reps 3+4 overlapped on GPU 2 (an orphaned duplicate launch from a dropped kubectl
websocket); their times are contended and still pass the gate.

**Gate: new optimized arm vs old optimized arm = 4.222 / 0.324 (median) = ~13.0x**
(required >= 1.05x). Verification passed on all 5 AFTER runs. The B34-estimated ceiling
(~1.4-1.5x) was beaten by ~9x because removing the router graph breaks did not just kill
the 0.88 ms router GPU work + 1.0 ms host gap — it let torch.compile fuse the FULL model
into one cudagraph, which also collapsed the previously fragmented compiled subgraphs.

## New wall split (torch.profiler, standalone compiled loop, 20 iters)

- Steady-state wall: **0.159 ms/iter** (harness measures 0.31-0.33 ms/iter — its per-iter
  Python/nvtx overhead, same in BEFORE so comparison is fair)
- GPU busy: ~0.158 ms/iter (single `CompiledFxGraph` cudagraph replay)
- **Host gap: ~0** (was ~1.0 ms) | **`aten::nonzero`: NONE** (was 32/iter) | **DtoH copies: NONE** (was 32/iter)
- Per-iter kernel split: expert bmm GEMMs (nvjet 256x96 bz, 3 launches) ~57 us; lm_head
  GEMM ~24 us; fused SiLU*mul ~12 us; router/dispatch (one-hot cumsum + zeros + scatter +
  topk gather + bitonic sort) ~25 us; attention/layernorm/embed ~15 us; fake-quant +
  finalize + misc ~25 us.

## Design that won (Design B: sortless fixed-capacity dense dispatch)

Bound per-instance via `types.MethodType` onto each `MoEExperts` module in `setup()`,
BEFORE `torch.compile` wraps the model (`_install_vectorized_router`). The level-4
dispatcher resolves `forward_grouped` through the instance, so it transparently picks up
the override.

1. `rank-within-expert = one_hot(flat_ids).cumsum(0).gather(...) - 1` — no argsort, no
   nonzero, and the inverse permutation problem disappears (gather back by the same slot
   indices restores original token order for free).
2. Scatter routed tokens into a `[num_experts, capacity, hidden]` slot tensor
   (`index_copy_`, slot indices unique by construction).
3. Three batched GEMMs against `w1/w3/w2_stacked`; zero rows are mathematically inert.
4. Gather by slots, scale by routing weights, `view(batch_seq, top_k, H).sum(1)`.

Capacity is calibrated once eagerly in `setup()` (the scout's per-expert `.item()` bug
fixed: the ONLY host sync happens pre-compile): 2x observed max tokens/expert, rounded to
a multiple of 64, clamped to the routed-token count -> **192** here (observed counts
43..86, mean 64, 2048 routed tokens / 32 experts). The compiled trace then sees a plain
int constant — fully static shapes, cudagraph-safe under reduce-overhead.

Numeric cross-check before any harness run: model-level output identical to the original
`forward_grouped` path (max_abs_diff = 0.0 at bf16) for both candidate designs.

## Designs considered

- **Design B (shipped)**: sortless capacity-bucketed dense dispatch — eager model fwd
  0.775 ms (vs 5.68 ms original eager), compiles/cudagraphs cleanly.
- **Design C**: `argsort` + `torch._grouped_mm(offs=int32 cumsum)` (available in torch
  2.12, works on sm_103, numerically exact) — eager 0.824 ms; slightly slower than B at
  these tiny shapes and riskier under inductor/cudagraphs, so not shipped. It is the
  right pattern at production shapes where 2x capacity padding would waste real FLOPs.
- Fixed-32-iteration loop with static max-size slices and `_forward_bmm_fused_graphable`
  -style full-dense one-hot masking (32x FLOPs): both dominated by B; not measured further.

## Exact diff

See `labs/moe_optimization_journey/optimized_moe_pad_quant.py` (only file changed):

- Module docstring: documents the override.
- New imports: `types`, `torch.nn.functional as F`, `MoEExperts`.
- New module function `_vectorized_forward_grouped(self, x, expert_indices, expert_weights)`.
- New method `_install_vectorized_router()`; called in `setup()` after model build +
  `.eval()`, followed by one eager calibration pass, all before `torch.compile`.
- `self.inputs` creation moved before the override install (RNG order unchanged:
  model init consumes CPU RNG, `randint` consumes CUDA RNG, compile consumes none —
  verified by inputs/verification PASS against the untouched baseline arm).
- tcgen05 max-autotune guard, verification payload, teardown, iterations: unchanged.

## Next lever

The optimized arm is now ~0.16 ms GPU-bound wall (single cudagraph). Remaining headroom
is GEMM-shape efficiency: the three expert bmms (~57 us) run at m=192/expert with 2x
padding waste, and lm_head (~24 us) is the next largest single kernel. Candidates:
(1) `torch._grouped_mm` (Design C) at exact counts to reclaim the 2x padding FLOPs once
inductor support is proven on sm_103; (2) pin Triton 3.5.0 to re-enable max-autotune
(tcgen05 wall) for tuned GEMM selection. Both are <=1.5x levers on a 0.16 ms wall —
far below the harness noise floor at 6 iterations, so bank this front as harvested.
