#!/usr/bin/env python3
"""Triton Fused MoE Kernel - All experts in ONE kernel launch!

This is the holy grail of MoE optimization:
- Single kernel launch for ALL experts
- Fused gate + SiLU + up + down operations
- Maximizes GPU utilization by avoiding Python loops
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _flat_topk_token_ids(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.int64)
    if top_k > 1:
        token_ids.div_(top_k, rounding_mode="floor")
    return token_ids


@triton.jit
def fused_moe_expert_kernel(
    # Pointers
    X_ptr, Out_ptr,
    W_gate_ptr, W_up_ptr, W_down_ptr,
    Sorted_weights_ptr,
    Expert_offsets_ptr,
    # Dimensions
    H: tl.constexpr, I: tl.constexpr,
    # Strides for X: [total_tokens, H]
    stride_x_n, stride_x_h,
    # Strides for weights: [E, H, I] or [E, I, H]
    stride_wg_e, stride_wg_h, stride_wg_i,
    stride_wu_e, stride_wu_h, stride_wu_i,
    stride_wd_e, stride_wd_i, stride_wd_h,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Process one expert's tokens in parallel."""
    # Program ID: which expert and which token block
    pid_e = tl.program_id(0)  # Expert ID
    pid_m = tl.program_id(1)  # Token block ID
    
    # Get this expert's token range
    start_offset = tl.load(Expert_offsets_ptr + pid_e)
    end_offset = tl.load(Expert_offsets_ptr + pid_e + 1)
    num_tokens = end_offset - start_offset
    
    # This block's token range
    block_start = pid_m * BLOCK_M
    if block_start >= num_tokens:
        return
    
    # Token indices for this block
    offs_m = block_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_tokens
    global_offs = start_offset + offs_m
    
    # Load expert routing weights
    routing_weights = tl.load(Sorted_weights_ptr + global_offs, mask=mask_m, other=0.0)
    
    # Accumulator for hidden states
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matmul: X @ W_gate and X @ W_up
    for k in range(0, H, BLOCK_K):
        # Load X block: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (global_offs[:, None] * stride_x_n + (k + offs_k)[None, :] * stride_x_h)
        x_mask = mask_m[:, None] & ((k + offs_k)[None, :] < H)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W_gate block: [BLOCK_K, BLOCK_N]
        wg_ptrs = W_gate_ptr + (pid_e * stride_wg_e + (k + offs_k)[:, None] * stride_wg_h + offs_n[None, :] * stride_wg_i)
        wg_mask = ((k + offs_k)[:, None] < H) & (offs_n[None, :] < I)
        wg_block = tl.load(wg_ptrs, mask=wg_mask, other=0.0)
        
        # Load W_up block
        wu_ptrs = W_up_ptr + (pid_e * stride_wu_e + (k + offs_k)[:, None] * stride_wu_h + offs_n[None, :] * stride_wu_i)
        wu_block = tl.load(wu_ptrs, mask=wg_mask, other=0.0)
        
        # Accumulate
        gate_acc += tl.dot(x_block, wg_block)
        up_acc += tl.dot(x_block, wu_block)
    
    # Apply SiLU to gate and multiply with up
    gate_silu = gate_acc * tl.sigmoid(gate_acc.to(tl.float32))
    hidden = gate_silu * up_acc
    
    # Second matmul: hidden @ W_down
    out_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, I, BLOCK_N):
        # Load hidden block
        h_block = hidden[:, n:n+BLOCK_N]
        
        # Load W_down block: [BLOCK_N, BLOCK_K]  
        wd_ptrs = W_down_ptr + (pid_e * stride_wd_e + (n + offs_n)[:, None] * stride_wd_i + offs_k[None, :] * stride_wd_h)
        wd_mask = ((n + offs_n)[:, None] < I) & (offs_k[None, :] < H)
        wd_block = tl.load(wd_ptrs, mask=wd_mask, other=0.0)
        
        out_acc += tl.dot(h_block[:, :min(BLOCK_N, I-n)], wd_block[:min(BLOCK_N, I-n), :])
    
    # Apply routing weights and store
    out_acc = out_acc * routing_weights[:, None]
    
    out_ptrs = Out_ptr + (global_offs[:, None] * stride_x_n + offs_k[None, :] * stride_x_h)
    out_mask = mask_m[:, None] & (offs_k[None, :] < H)
    tl.store(out_ptrs, out_acc.to(tl.bfloat16), mask=out_mask)


def triton_fused_moe(
    x: torch.Tensor,          # [total_tokens, H]
    w_gate: torch.Tensor,     # [E, H, I]
    w_up: torch.Tensor,       # [E, H, I]
    w_down: torch.Tensor,     # [E, I, H]
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,  # [E+1] cumulative offsets
    E: int, H: int, I: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Launch Triton fused MoE kernel."""
    total_tokens = x.shape[0]
    output = torch.empty_like(x)
    
    # Grid: (num_experts, max_tokens_per_expert / BLOCK_M). Callers should pass
    # the precomputed max for a tight launch; the fallback avoids a CUDA scalar
    # readback by using the total sorted assignments as a conservative bound.
    if max_tokens is None:
        max_tokens = total_tokens
    else:
        max_tokens = int(max_tokens)
    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_N = 64
    
    grid = (E, triton.cdiv(max_tokens, BLOCK_M))
    
    fused_moe_expert_kernel[grid](
        x, output,
        w_gate, w_up, w_down,
        sorted_weights,
        expert_offsets,
        H, I,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        BLOCK_M, BLOCK_K, BLOCK_N,
    )
    
    return output


def benchmark_triton_moe():
    """Benchmark the Triton fused MoE kernel."""
    device = 'cuda'
    torch.manual_seed(42)
    
    H, I, E, K = 4096, 11008, 8, 2
    batch_seq = 8192
    
    print(f"Benchmarking Triton Fused MoE: {batch_seq} tokens, H={H}, I={I}, E={E}")
    print()
    
    x = torch.randn(batch_seq, H, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(E, H, I, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(E, H, I, device=device, dtype=torch.bfloat16)
    w_down = torch.randn(E, I, H, device=device, dtype=torch.bfloat16)
    
    # Generate routing and precompute the launch bound on CPU to avoid a CUDA
    # scalar readback before the timed kernel loop.
    expert_indices_cpu = torch.randint(0, E, (batch_seq, K), dtype=torch.int64)
    counts_cpu = torch.bincount(expert_indices_cpu.reshape(-1), minlength=E)
    max_tokens = int(counts_cpu.max())
    expert_indices = expert_indices_cpu.to(device=device, non_blocking=True)
    expert_weights = F.softmax(torch.randn(batch_seq, K, device=device), dim=-1).to(torch.bfloat16)
    
    # Sort by expert
    flat_idx = expert_indices.view(-1)
    sorted_order = torch.argsort(flat_idx, stable=True)
    flat_token_ids = _flat_topk_token_ids(batch_seq, K, x.device)
    sorted_token_ids = flat_token_ids.index_select(0, sorted_order)
    sorted_tokens = x.index_select(0, sorted_token_ids)
    sorted_weights = expert_weights.view(-1).index_select(0, sorted_order)
    sorted_expert_ids = flat_idx.index_select(0, sorted_order)
    
    # Compute expert offsets
    counts = torch.bincount(sorted_expert_ids, minlength=E)
    expert_offsets = torch.empty(E + 1, device=device, dtype=torch.long)
    expert_offsets[0] = 0
    expert_offsets[1:].copy_(counts.cumsum(0))
    
    # Test kernel
    try:
        output = triton_fused_moe(
            sorted_tokens, w_gate, w_up, w_down,
            sorted_weights, expert_offsets,
            E, H, I, max_tokens=max_tokens
        )
        print(f"✅ Triton kernel executed! Output shape: {output.shape}")
        
        # Benchmark
        for _ in range(5):
            _ = triton_fused_moe(sorted_tokens, w_gate, w_up, w_down,
                                sorted_weights, expert_offsets, E, H, I, max_tokens=max_tokens)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        for _ in range(10):
            _ = triton_fused_moe(sorted_tokens, w_gate, w_up, w_down,
                                sorted_weights, expert_offsets, E, H, I, max_tokens=max_tokens)
        end.record(current_stream)
        end.synchronize()
        ms = start.elapsed_time(end) / 10
        
        flops = batch_seq * K * 3 * 2 * H * I
        tflops = flops / (ms / 1000) / 1e12
        
        print(f"Triton Fused MoE: {ms:.2f} ms = {tflops:.0f} TFLOPS ({tflops/2250*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Triton kernel failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark_triton_moe()
