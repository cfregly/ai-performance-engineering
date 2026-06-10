# GB300 Speed-of-Light roofline grounding of 7 shipped wins

Ratchet-the-champion pass: each shipped win is re-profiled with `ncu --set full` to decide
whether its dominant GPU kernel is at its Speed-of-Light ceiling or still has headroom with an
addressable bottleneck. Read-only diagnosis. No committed source was modified.

## Environment and method

- Hardware: NVIDIA GB300 (Blackwell Ultra, sm_103, cc 10.3), 4 GPU/node, 284 GB HBM3e/GPU. Work on `CUDA_VISIBLE_DEVICES=3` (1-3 free; 0 in use).
- Profiler: ncu 2026.1.1 (`/usr/local/bin/ncu`), CUDA 13.2 toolkit aligned with the 13.2 driver, so the CUPTI `INVALID_DEVICE` skew that zeroes ncu on the stock infr image does not apply. Captures used `--set full`, profile-from-start with `--launch-skip`/`--launch-count` and `--kernel-name regex:` scoping (no `cudaProfilerStart`).
- GB300 per-GPU roofline peaks (from `perf-tune-report/configs/sol-ceilings.yaml` `gb300_nvl72`): NVFP4 dense 15.0 PFLOPS, FP8 dense 7.5 PFLOPS, BF16/FP16 dense 3.75 PFLOPS, HBM3e 8.0 TB/s, NVLink5 1.8 TB/s.
- Caveat on numbers: GPU clocks were not locked (read-only diagnosis), so absolute microseconds and `%SoL` are directional, not canonical-bench grade; A/B ratios are same-process back-to-back on one GPU. Per the ncu skill, at-roof vs headroom is judged from `--set full` achieved tensor/DRAM throughput, never from `--set basic` SM-busy alone.
- Discovery used `torch.profiler` to rank kernels; `--set full` ncu then measured the dominant kernel.

## Per-win roofline

| # | Win | Dominant kernel | Achieved SoL (`--set full`) | Bound | Verdict |
|---|---|---|---|---|---|
| 3 | labs/nvfp4_gemv (case2 m7168 k2048 l4) | `triton_red_fused_...index_mul_mv_...` fused fp4-unpack + matvec, 79% of GPU time, ~30.6 us | DRAM 3.8% of 8 TB/s (~304 GB/s); tensor-pipe 0%, FMA-pipe 18%; Compute(SM) 70% | memory-bound op stuck SM-ALU/LSU-bound (H1, no tensor cores) | HEADROOM ~26x to HBM floor |
| 4 | labs/moe_pad_quant (1024 tok, h512/i2048, 32 experts) | spray of ~350 tiny kernels/call; expert GEMM `nvjet_sm103_tst_64x8_..._NNT`, ~6.4 us | tensor-pipe 7%, DRAM 4.8%, Compute(SM) 17%, theoretical occupancy 12.5%, achieved 9.9% | launch/latency-bound + occupancy-starved (M about 64/expert) | launch-bound; cudagraph win correct; per-kernel SoL low but mostly moot at toy size |
| 5 | ch13/regional_compile (MLP fc1 2048->16384, fc2, BF16) | BF16 `nvjet_sm103_tst_128x256_..._bias_TNT` (69.6 us) and a 256-tile (543 us) | tensor-pipe 94-99%, Compute(SM) 86-98%, DRAM 21% | tensor-core compute-bound | AT-CEILING (cuBLAS/nvjet BF16 GEMM at peak) |
| 1 | ch15/inference_monolithic (batch=1 decode) | batch=1 GEMV chain, cudagraph-replayed | not profiled (latency-moot) | launch-overhead, then latency-scale at 1 token | AT-CEILING, latency-moot |
| 2 | ch15/speculative_decoding (batch=1) | compiled draft+target forwards + eager accept loop | not profiled (latency/host-moot) | acceptance rate + host-side accept (`.item()` syncs) | AT-CEILING, latency/host-moot |
| 6 | labs/persistent_decode/paged_kv_offload_prefetch | NVLink-C2C KV page copies (cudaMemcpyAsync) | not profiled (transfer/host) | host prefetch-thread overhead, then coherent-fabric BW | AT-CEILING, host/transfer |
| 7 | ch02/grace_coherent_memory (zero-copy 256 MB) | one HBM-bound `mul_/add_` elementwise | transfer eliminated by construction | transfer elimination (zero-copy) | TRANSFER-AT-CEILING |

### Detail: #3 nvfp4_gemv (the top-headroom win)

The shipped GB300 fast path is a `torch.compile`-fused dequant GEMV (`_run_dequant_gemv`,
`AISP_NVFP4_GEMV_USE_DEQUANT_GEMV=1`). The fp32-materialization thesis is REFUTED: inductor fused the
LUT-unpack, per-16 scale, and the matvec reduction into one `triton_red_fused_...index_mul_mv_...`
kernel, and the measured DRAM traffic (~9 MB at 3.8% of 8 TB/s over 30.6 us) matches reading the
packed fp4 matrix (7.34 MB) plus fp8 scales (0.92 MB) once. There is no 234 MB fp32 write-back and no
extern SGEMV.

The headroom is kernel inefficiency, not materialization. The GEMV is intrinsically memory-bound
(arithmetic intensity about 1.8 FLOP/byte, far below the NVFP4 roofline knee of about 1875), so its
SoL is HBM bandwidth: floor about 8.26 MB/l-iter / 8 TB/s = about 1.0 us. The fused kernel takes about
30.6 us/l-iter = 3.8% of the HBM roofline. ncu shows why: tensor-pipe 0% and FMA-pipe only 18%, while
Compute(SM) reads 70% because the SM is busy on int64 LUT-gather (`lut[a_idx]`), `torch.stack`/reshape
nibble interleave, and the fp32 reduction, not on useful FLOPs. It is an H1 (no tensor core) kernel on
a K3 op.

### Detail: #4 moe_pad_quant

The forward is a spray of about 350 tiny kernels per call: about 31 per-expert nvjet GEMMs at 64-row
tiles plus 32x each of cub select/scan, gather/scatter, elementwise, and DtoH host syncs (32 experts,
top-2, 1024 tokens, about 64 tokens/expert). The dominant expert GEMM uses tensor cores but at only 7%
tensor-pipe util with theoretical occupancy capped at 12.5% (registers/smem limit it to 1 block/SM)
because M is about 64. This is occupancy-starved and launch-bound, both inherent to a 32-expert /
1024-token / hidden-512 MoE. The shipped reduce-overhead cudagraph win (1.93x) correctly attacks the
dominant cost, which is the launch overhead of about 350 kernels, not any single GEMM.

### Detail: #5 regional_compile

The compiled MLP GEMMs (fc1 2048->16384, fc2 16384->2048, BF16, batch 16, seq up to 1536) run on the
Blackwell cuBLAS/nvjet path at 94-99% tensor-pipe and 86-98% Compute(SM): at the BF16 tensor-core
ceiling. The eager pytorch flash-attention (17% of time) and the `triton_poi_fused_gelu` (12%) are the
only non-GEMM kernels, both already efficient. The 0.09x->1.10x win was a compile-config fix
(`mode=default` drops the cudagraph that re-recorded every call on the eager-attention output and the
cycling sequence length), not a kernel change. No kernel headroom.

## Ranking by remaining headroom

1. #3 nvfp4_gemv: 3.8% of HBM SoL, about 26x to the HBM floor, real and addressable.
2. #4 moe_pad_quant: 7% tensor / 12.5% occupancy; a grouped-expert GEMM lever exists but EV is modest at this toy size (launch overhead is the real cost and the cudagraph win already took it).
3. #5 regional_compile: 94-99% tensor, at-ceiling. Negligible kernel headroom.
4-7. #1, #2, #6, #7: latency-moot or transfer-at-ceiling by construction. The shipped wins removed launch overhead (#1, #2) or transfer/host overhead (#6, #7); the residual per-kernel work is launch-latency-scale at batch=1 or coherent-fabric/HBM-bound, so per-kernel SoL is moot.

## Top deeper lever: #3 fused NVFP4 GEMV, HBM-bound not ALU-bound

The GEMV is memory-bound, so the SoL is HBM bandwidth (about 1 us/l-iter at case2). The shipped fused
kernel sits at 3.8% of that because it spends the SM on dequant ALU/LSU work (int64 gather, nibble
interleave, fp32 accumulate) at 0% tensor and 18% FMA. Two tiers of fix, both targeting an HBM-bound
kernel:

- Quick FMA fix (R2/H1, low risk): in the compiled `_nvfp4_dequant_gemv_one`, replace the int64
  `lut[a_idx]` gather plus `torch.stack`/reshape interleave with an in-register e2m1 bit-decode
  (shift/and to sign/exp/mant arithmetic), vectorized uint8 loads, and fp16/bf16 accumulation. Expected
  unlock: 3.8% -> about 25-50% of HBM SoL (about 6-13x on the kernel). Gate: ncu DRAM% on the
  re-emitted triton kernel plus the repo paired `--verify` A/B.
- Tensor-core fp4 GEMV (R1/H4, higher effort): a CUTLASS NVFP4 MMA GEMV reading packed fp4 + blocked
  scales once, batched over l, with no per-l host repack and no N=1->128 pad. Expected unlock: HBM-bound
  about 1-4 us/l-iter (about 8-26x on the kernel). The existing `scaled_mm` and `gemm_v3b` paths do not
  capture this because they pad N=1 to 128 and repack scale factors in python per l-iter.

### Prototype measured in-pod (directional, bit-exact, for review)

Measured the three existing kernel paths back-to-back on GPU3 (clocks not locked, so directional),
total `custom_kernel` latency, all bit-exact vs the scaled_mm reference (maxdiff = 0):

| shape | dequant (shipped) | scaled_mm (tensor-core) | gemm_v3b | best |
|---|---|---|---|---|
| case2 m7168 k2048 l4 | 436.6 us | 484.4 us | 701.7 us | dequant |
| case0 m7168 k16384 l1 | 345.9 us | 275.6 us | 330.7 us | scaled_mm (1.26x) |
| case1 m4096 k7168 l8 | 789.1 us | 987.4 us | 1245.6 us | dequant |

Two readings. (1) End-to-end, dequant is the right default for case1/case2 (its fusion avoids
scaled_mm's per-l python `to_blocked` repack and N=128 padded output), so the shipped default is sound
there. (2) For case0 (l=1, k=16384) the existing tensor-core `scaled_mm` path is 1.26x faster and
bit-exact, so the shipped dequant default leaves 1.26x on the table at that shape. Routing case0 to
scaled_mm is a concrete, low-risk, verify-passing (maxdiff = 0) directional win to confirm with the
repo's paired `--verify` A/B before changing any default. Neither result is claimed as a shipped win
here; committed source was not modified.

The kernel-level headroom (3.8% of HBM SoL, 0% tensor) is larger than either existing path captures and
is the real prize: a purpose-built HBM-bound fp4 GEMV. The case0 routing is the cheap, already-measured
first step.

## Artifacts

ncu reports (in-pod): `/tmp/sol/gemv_case2.ncu-rep`, `/tmp/sol/moe_gemm.ncu-rep`,
`/tmp/sol/regional_gemm.ncu-rep`. Harnesses: `/tmp/sol/gemv_*.py`, `/tmp/sol/moe_ncu.py`,
`/tmp/sol/regional_ncu.py` (scratch, outside the repo).
