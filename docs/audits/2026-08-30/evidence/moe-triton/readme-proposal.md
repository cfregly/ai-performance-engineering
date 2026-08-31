The incomplete standalone Triton MoE/FFN and raw grouped-GEMM experiments were
withdrawn; legacy `triton_fused_moe` calls fail explicitly and emit no performance
result. The active SiLU×up helper uses differentiable PyTorch whenever either
input needs gradients, and also uses explicit PyTorch on CPU or without Triton.
Only eligible CUDA inference launches the elementwise Triton kernel, including
any required contiguous copies in the call. This is activation fusion, not a
fused full expert FFN. Actual CUDA numeric, device, stream, and memory-sanitizer
acceptance remains HOLD.

The compatibility name `level4_triton.GroupedMoEExperts` refers to the separately
identified sorted PyTorch expert path. It is not a replacement fused Triton FFN.
