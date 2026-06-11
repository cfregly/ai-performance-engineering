# GB300 moe_pad_quant: Speed-of-Light roofline + lever verdict

## Verdict: deep-deferred (no in-scope surgical win)

The shipped `moe_pad_quant` cudagraph win is a host-overhead win that is already
captured. The residual after it is routing-bound and host-bound, not expert-GEMM-bound.
The per-expert GEMMs are BF16, so the FP8 `torch._scaled_grouped_mm` surgical lever does
not apply. The only real lever (kill the 32-iteration per-expert `nonzero` router loop, or
batch the experts into one grouped GEMM) lives in `moe_model.py` / `ch19.mxfp8_moe_common.py`,
outside the single file this task may edit (`optimized_moe_pad_quant.py`), and is a
non-surgical rewrite. No verify-passing in-file speedup was found or claimed.
`optimized_moe_pad_quant.py` is unchanged.

Workload descriptor for every number below: MoEPadQuantFinalize, level=4 grouped MoE,
hidden=512, intermediate=2048, num_experts=32, top_k=2, batch=8 x seq=128 = 1024 tokens
(2048 routed), BF16, GB300 (sm_103, cc 10.3, 4 GPU/node), torch 2.12.0a0+nv26.05,
CUDA 13.2 ncu 2026.1.1, single GPU (CUDA_VISIBLE_DEVICES=0). GB300 peaks: BF16 3.75 PFLOPS,
FP8 7.5 PFLOPS, NVFP4 15.0 PFLOPS, HBM 8.0 TB/s.

## 1. Shipped baseline (ground truth, clock-locked harness)

`python -m cli.aisp bench run labs/moe_optimization_journey:moe_pad_quant`, validity=strict,
20 iters / 5 warmup:

| arm | time/iter | verify |
| --- | --- | --- |
| baseline (eager level-4 grouped) | 7.05 ms | n/a |
| optimized (torch.compile reduce-overhead) | 4.18 ms | PASS (0 verification failures) |
| speedup | 1.686x | succeeded |

The compile mode is `reduce-overhead` (cudagraph), not `max-autotune`: on sm_103 + Triton 3.7
max-autotune emits an unloadable `tcgen05.wait.st` kernel, so the file falls back to
reduce-overhead. The 1.686x here vs the 1.927x cited earlier is run-to-run variance on the
same path; both are real cudagraph wins with verify PASS.

## 2. Host-vs-GPU split (the central question)

Profiler probe (wall measured clean, CUDA busy = sum of device self-time, both per iter):

| arm | wall/it | CUDA busy/it | host gap/it | GPU-busy % | device launches/it |
| --- | --- | --- | --- | --- | --- |
| optimized (reduce-overhead) | 3.04 ms | 2.03 ms | 1.01 ms | 66.9% | 540 |
| baseline (eager) | 7.80 ms | 2.43 ms | 5.38 ms | 31.1% | 890 |

The cudagraph win is a host-overhead win: it cut the host gap 5.38 ms to 1.01 ms while CUDA
busy barely moved (2.43 to 2.03 ms). After the win the call is 67% GPU-busy with a 1.0 ms
host-gap residual that the data-dependent router loop prevents cudagraph from capturing.
(Probe is not clock-locked, so absolute ms differ from the harness; the split ratio is the
clock-robust signal.)

## 3. Where the GPU time goes (optimized, ~2.03 ms CUDA busy)

Top device kernels, ms/iter:

| group | kernels | ms/it | share |
| --- | --- | --- | --- |
| routing / bucketing | `aten::nonzero` x32 (0.255), `aten::gather` x65 (0.133), cub select/reduce/scan x32 each, `scatter_gather` x32, `vectorized_gather` x33, `Memcpy DtoH` x32 (0.055), `aten::eq` x32 | ~0.88 | ~43% |
| compiled fused subgraphs | embed / attn / layernorm / gate / finalize / lm_head | ~0.50 | ~25% |
| per-expert BF16 GEMMs | `nvjet_sm103_tst_*` x~91 | ~0.38 | ~19% |

The expert GEMMs are not the dominant GPU cost. Routing/bucketing is the single largest GPU
consumer and is also the source of the host gap.

## 4. Expert dtype: BF16

`moe_pad_quant_common.build_moe_pad_quant_model(level=4)` then `model.to(dtype=torch.bfloat16)`.
At level 4 the experts run `MoEExperts.forward_grouped` (`moe_model.py`):
`gate = F.silu(tokens_e @ self.w1_stacked[expert_id])`, with `tokens_e` and `w*_stacked` both
BF16. The "quant" in the lab name is the INT8 fake-quant in the pad+finalize tail
(`fake_quant_int8`, quantize/dequantize in fp32 then cast back to BF16), not the expert GEMM.
So the experts are BF16, not FP8/MXFP8 and not NVFP4.

## 5. ncu --set full, dominant expert GEMM (`nvjet_sm103_tst_64x8_64x16_2x4_h_bz_NNT`)

| metric | value |
| --- | --- |
| Duration | ~7.0 us/launch |
| Compute (SM) throughput | 16.3% (re-measured 17.3%) |
| Memory throughput | 10.1% |
| DRAM throughput (HBM SoL) | 4.3% |
| Theoretical occupancy | 12.5% (register-limited: 255 reg/thread, block-limit-registers=1) |
| Achieved occupancy | 9.1 to 9.9% |
| Waves per SM | 0.63 (does not fill one wave across the GPU) |
| Issue slots busy / IPC | 3.5% / 0.40 |
| Tensor pipe % | n/a on sm_103 in this ncu; sibling roofline cited ~7% |

Bound classification: latency / launch / occupancy-bound. Compute(SM) 16%, DRAM 4%, and
0.63 waves/SM are all far below their ceilings; the kernel is too small (M~64 per expert) and
register-bound to fill the GPU. It is nowhere near the BF16 3.75 PFLOPS FLOP roofline or the
8.0 TB/s HBM roofline.

## 6. Root cause of the launch/occupancy-bound shape

The router `ch19.mxfp8_moe_common.bucket_by_expert` runs a Python
`for expert in range(num_experts)` loop with `torch.nonzero(assignments == expert)` per expert.
That is the 32x `nonzero` + 32x `Memcpy DtoH` + the cub select/reduce/scan kernels: 32
data-dependent host syncs. `forward_grouped` then adds `expert_order.tolist()` and
variable-count slicing. This pattern (a) is the largest GPU + host cost and (b) is
data-dependent, so cudagraph cannot capture it, which is exactly the 1.0 ms host-gap residual.

## 7. Lever analysis

Surgical FP8 `torch._scaled_grouped_mm` (the requested lever): NOT APPLICABLE. It requires
FP8 operands; the experts are BF16.

BF16 `torch._grouped_mm` (the BF16 analog, present in torch 2.12): end-to-end-capped and
out of scope. The expert GEMMs are ~0.38 ms of a 3.04 ms wall (~12%); batching the ~91 tiny
launches into one grouped GEMM caps the win near ~1.1x while the routing (~0.88 ms) and the
host gap (~1.0 ms), the actual bottleneck, remain. `torch._grouped_mm` also still needs the
tokens bucketed by expert with offsets, so it does not remove the 32x `nonzero` router loop.
The loop lives in `moe_model.py` / `ch19.mxfp8_moe_common.py`, which this task may not edit
(modify ONLY `optimized_moe_pad_quant.py`). This is the same shape as the nvfp4_gemv case2
finding: a kernel win on a host/routing-bound call is end-to-end-capped.

Deep deferred lever (real headroom, out of scope here): replace the per-expert `nonzero`
router with a vectorized/grouped router (sort + offsets, or the dense one-hot graphable path)
so the whole expert path becomes host-sync-free and a single grouped/batched BF16 GEMM, which
both kills the 32 host syncs (the residual host gap) and raises the GEMM occupancy. That is a
`moe_model.py` / `ch19` rewrite, not a surgical edit to `optimized_moe_pad_quant.py`.

## 8. What was changed

Nothing. No surgical in-file lever applies (BF16 experts, routing-bound residual, router out
of scope), so the prototype step (explicitly conditional on an FP8 grouped path) was not run.
`optimized_moe_pad_quant.py` is byte-for-byte unchanged; the shipped 1.686x with verify PASS
stands.

## Method / provenance

GB300 node `<gb300-pod>`, ns `<namespace>`, repo `/work/ai-performance-engineering/code`
at `2f7e30f9` with the working-tree reduce-overhead fallback in `optimized_moe_pad_quant.py`
and `moe_benchmark.py` (pre-existing, not introduced here). Run on free GPU0. Read-only roofline
(harness bench + profiler probe + ncu --set full); no benchmark file was edited.
