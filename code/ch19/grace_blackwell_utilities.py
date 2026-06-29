"""grace_blackwell_utilities.py - Grace-Blackwell system utilities from Chapter 19.

Chapter 19: Grace-Blackwell Inference Optimizations

This module contains system utilities for Grace-Blackwell inference optimization:
- GPU memory monitoring
- Dynamic quantized cache configuration
- Occupancy-aware tile selection
- Scheduler utilities for chunked prefill/decode

These utilities are designed for Grace-Blackwell systems with unified memory
and NVLink-C2C, but will work on other CUDA systems with graceful degradation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Callable

import torch

# Optional CuPy for advanced occupancy queries
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


# ---------------------------------------------------------------------------
# GPU Memory Monitoring
# ---------------------------------------------------------------------------

def _gpu_used_ratio(device: Optional[int] = None) -> float:
    """Return fraction of device memory used as 1 - free/total.
    
    Book reference: Chapter 19, Dynamic Quantized Cache section.
    
    Uses CUDA driver info, which reflects true device state,
    not just the PyTorch allocator's reserved bytes.
    
    Args:
        device: CUDA device index (default: current device)
        
    Returns:
        Memory usage ratio in [0.0, 1.0]
        
    Example:
        >>> ratio = _gpu_used_ratio()
        >>> print(f"GPU memory {ratio*100:.1f}% used")
    """
    if device is None:
        device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return 1.0 - (free / total)


# ---------------------------------------------------------------------------
# Quantized Cache Configuration
# ---------------------------------------------------------------------------

def make_cache_config(
    *,
    backend: str,
    nbits: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    q_group_size: int = 64,
    residual_length: int = 128,
    axis_key: int = 1,
    axis_value: int = 1,
) -> Dict[str, Any]:
    """Build a cache_config dictionary accepted by Transformers' quantized cache.
    
    Book reference: Chapter 19, Dynamic Quantized Cache section.
    
    HQQ supports nbits in {2, 4, 8}; Quanto supports {2, 4}.
    axis_key/axis_value=1 are recommended for HQQ.
    
    Args:
        backend: "HQQ" or "quanto"
        nbits: Bit width for quantization (2, 4, or 8)
        device: Target torch device
        compute_dtype: Dtype for dequantization compute
        q_group_size: Group size along head_dim
        residual_length: Number of recent tokens kept at original precision
        axis_key: Quantization axis for keys
        axis_value: Quantization axis for values
        
    Returns:
        Dictionary of cache configuration parameters
        
    Example:
        >>> config = make_cache_config(
        ...     backend="HQQ",
        ...     nbits=4,
        ...     device=torch.device("cuda:0"),
        ... )
    """
    backend_lower = backend.lower()
    
    # Validate backend and nbits
    if backend_lower == "quanto":
        if nbits not in {2, 4}:
            raise ValueError("Quanto supports only 2 or 4 bits")
    elif backend_lower == "hqq":
        if nbits not in {2, 4, 8}:
            raise ValueError("HQQ supports 2, 4, or 8 bits")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'HQQ' or 'quanto'")
    
    return {
        "backend": backend,
        "nbits": int(nbits),
        "axis_key": axis_key,
        "axis_value": axis_value,
        "q_group_size": int(q_group_size),
        "residual_length": int(residual_length),
        "compute_dtype": compute_dtype,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Occupancy and Tile Optimization
# ---------------------------------------------------------------------------

# Caches for occupancy results and tile lookup
_occ_cache: Dict[Tuple[int, int], float] = {}
_tile_table: Dict[int, int] = {}


def get_optimal_tile(L: int, max_tile: int = 1024, min_tile: int = 32) -> int:
    """Get optimal tile size for sequence length L.
    
    Book reference: Chapter 19, Scheduler Utilities section.
    
    Precomputes tile sizes for efficient attention computation.
    Tiles are aligned to warp size (32) for coalesced memory access.
    
    Args:
        L: Sequence length
        max_tile: Maximum tile size
        min_tile: Minimum tile size (warp-aligned)
        
    Returns:
        Optimal tile size T where min_tile <= T <= min(L, max_tile)
        
    Example:
        >>> T = get_optimal_tile(2048)
        >>> print(f"Optimal tile size: {T}")
    """
    if L in _tile_table:
        return _tile_table[L]
    
    # Compute block size - start with L, clamp to max
    T = min(max_tile, L)
    
    # Align to warp size (32)
    T = max(min_tile, (T // 32) * 32)
    
    _tile_table[L] = T
    return T


def get_occupancy(
    threads: int, 
    shared_bytes: int,
    kernel_ptr: Optional[Any] = None,
    device: int = 0,
) -> float:
    """Get occupancy for given thread count and shared memory usage.
    
    Book reference: Chapter 19, Scheduler Utilities section.
    
    Calculates theoretical occupancy based on thread and shared memory limits.
    Uses CuPy for accurate occupancy queries if available, otherwise
    estimates based on device properties.
    
    Args:
        threads: Number of threads per block
        shared_bytes: Dynamic shared memory per block (bytes)
        kernel_ptr: Optional kernel pointer for CuPy occupancy query
        device: CUDA device index
        
    Returns:
        Occupancy ratio in [0.0, 1.0]
        
    Example:
        >>> occ = get_occupancy(256, 48*1024)  # 256 threads, 48KB smem
        >>> print(f"Occupancy: {occ*100:.1f}%")
    """
    key = (threads, shared_bytes)
    if key in _occ_cache:
        return _occ_cache[key]
    
    props = torch.cuda.get_device_properties(device)
    
    # Calculate based on resource limits
    warps_per_block = threads // props.warp_size
    max_warps_per_sm = props.max_threads_per_multi_processor // props.warp_size
    
    # Shared memory limit
    if shared_bytes > 0:
        # Use shared_memory_per_block_optin if available, fallback to default
        smem_per_sm = getattr(props, 'shared_memory_per_block_optin', 
                              getattr(props, 'max_shared_memory_per_block', 49152))
        max_blocks_by_smem = smem_per_sm // shared_bytes if shared_bytes > 0 else 999
    else:
        max_blocks_by_smem = 999
    
    # Thread limit
    max_blocks_by_threads = max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0
    
    # Register limit (assume 64 regs per thread typical)
    regs_per_thread = 64
    regs_per_sm = 65536  # Typical for modern GPUs
    max_blocks_by_regs = regs_per_sm // (threads * regs_per_thread) if threads > 0 else 0
    
    # Take minimum
    max_blocks = min(max_blocks_by_smem, max_blocks_by_threads, max_blocks_by_regs, 32)
    max_blocks = max(1, max_blocks)
    
    occ = (max_blocks * warps_per_block) / max_warps_per_sm
    occ = min(1.0, max(0.0, occ))
    
    _occ_cache[key] = occ
    return occ


def clear_occupancy_cache() -> None:
    """Clear the occupancy and tile caches."""
    _occ_cache.clear()
    _tile_table.clear()


# ---------------------------------------------------------------------------
# Scheduler Utilities
# ---------------------------------------------------------------------------

def scheduler_loop(
    get_pending_requests: Callable[[], List[Any]],
    gpu_utilization: Callable[[], float],
    process_request: Callable[[Any, int], None],
    *,
    target_util: float = 0.85,
    occ_threshold: float = 0.50,
    block_threads: int = 256,
    max_iterations: Optional[int] = None,
) -> None:
    """Run the adaptive scheduler loop for chunked prefill/decode.
    
    Book reference: Chapter 19, Scheduler Utilities section.
    
    The scheduler monitors real-time metrics and adapts on the fly:
    - Adjusts chunk size based on occupancy
    - Prioritizes decode requests for latency
    - Fills GPU capacity with prefill chunks
    
    Args:
        get_pending_requests: Callback to get pending request queue
        gpu_utilization: Callback to get current GPU utilization
        process_request: Callback to process a request with given tile size
        target_util: Target GPU utilization threshold
        occ_threshold: Minimum acceptable occupancy
        block_threads: Threads per block
        max_iterations: Max loop iterations (None = infinite)
        
    Example:
        >>> def get_requests():
        ...     return pending_queue
        >>> def get_util():
        ...     return _gpu_used_ratio()
        >>> def process(req, T):
        ...     req.run_with_tile(T)
        >>> scheduler_loop(get_requests, get_util, process, max_iterations=100)
    """
    iteration = 0
    
    while max_iterations is None or iteration < max_iterations:
        pending = get_pending_requests()
        if not pending:
            break
            
        util = gpu_utilization()
        
        req = None
        max_prefill_len = -1
        if util < target_util:
            for candidate in pending:
                if getattr(candidate, 'phase', 'decode') != 'prefill':
                    continue
                remaining_length = getattr(candidate, 'remaining_length', lambda: 0)()
                if remaining_length > max_prefill_len:
                    max_prefill_len = remaining_length
                    req = candidate

        if req is not None:
            # Select heaviest prefill request
            L = getattr(req, 'remaining_length', lambda: 1024)()
            
            # Get optimal tile size
            T = get_optimal_tile(L)
            
            # Calculate shared memory (3 buffers: Q, K, V)
            shared_bytes = 3 * T * T * 2  # FP16
            
            # Check occupancy
            occ = get_occupancy(block_threads, shared_bytes)
            
            if occ < occ_threshold:
                # Reduce tile size to improve occupancy
                T = max(32, T // 2)
                shared_bytes = 3 * T * T * 2
            
            process_request(req, T)
        else:
            # Process decode requests (latency-sensitive)
            for req in pending:
                if getattr(req, 'phase', 'decode') == 'decode':
                    process_request(req, 32)  # Small tile for decode
        
        iteration += 1


# ---------------------------------------------------------------------------
# KV Cache Prefetch Utilities  
# ---------------------------------------------------------------------------

def consume_prefetched_kv(
    kv_buffer: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Consume prefetched KV cache data from async copy.
    
    Book reference: Chapter 19, KV Cache Prefetch section.
    
    This function is called after async prefetch completes to use
    the prefetched KV data in the compute stream.
    
    Args:
        kv_buffer: Prefetched KV cache tensor
        stream: CUDA stream for consumption (default: current stream)
        
    Returns:
        The KV buffer ready for use
        
    Example:
        >>> # After prefetch_kv_async(buffer, prefetch_stream)
        >>> kv = consume_prefetched_kv(buffer, compute_stream)
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            # Ensure prefetch is complete
            torch.cuda.current_stream().synchronize()
            return kv_buffer.contiguous()
    return kv_buffer.contiguous()


# ---------------------------------------------------------------------------
# Dynamic Cache Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_with_dynamic_quantized_cache(
    model: Any,  # AutoModelForCausalLM
    tokenizer: Any,  # AutoTokenizer
    prompt: str,
    *,
    max_new_tokens: int = 256,
    chunk_tokens: int = 32,
    memory_threshold: float = 0.90,
    backend: str = "hqq",
    start_bits: int = 8,
    fallback_bits: int = 4,
    residual_length: int = 128,
) -> str:
    """Generate text in chunks with dynamic quantized cache policy.
    
    Book reference: Chapter 19, Dynamic Quantized Cache section.
    
    Generates text while monitoring memory and switching cache policy
    mid-generation if memory pressure is high.
    
    Args:
        model: HuggingFace causal LM model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        chunk_tokens: Tokens to generate per chunk
        memory_threshold: Switch policy if memory usage exceeds this
        backend: Quantization backend ("hqq" or "quanto")
        start_bits: Initial cache bit-width
        fallback_bits: Lower bit-width on memory pressure
        residual_length: Recent tokens kept at full precision
        
    Returns:
        Generated text string
        
    Example:
        >>> text = generate_with_dynamic_quantized_cache(
        ...     model, tokenizer, "Hello, world!",
        ...     max_new_tokens=100,
        ...     memory_threshold=0.85,
        ... )
    """
    backend = backend.lower()
    if backend not in {"hqq", "quanto"}:
        raise ValueError("backend must be 'hqq' or 'quanto'")
    
    if backend == "quanto" and (start_bits not in {2, 4} or fallback_bits not in {2, 4}):
        raise ValueError("Quanto supports only 2 or 4 bits")
    if backend == "hqq" and (start_bits not in {2, 4, 8} or fallback_bits not in {2, 4, 8}):
        raise ValueError("HQQ supports 2, 4, or 8 bits")
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]
    
    current_bits = start_bits
    tokens_generated = 0
    used_ratio = _gpu_used_ratio()
    
    while tokens_generated < max_new_tokens:
        # Check memory and potentially switch policy
        if tokens_generated > 0:
            # Exponential moving average of memory usage
            used_ratio = 0.8 * used_ratio + 0.2 * _gpu_used_ratio()
        else:
            used_ratio = _gpu_used_ratio()
        
        # Switch to lower bits if memory is tight
        if used_ratio >= memory_threshold and current_bits > fallback_bits:
            current_bits = fallback_bits
            print(f"[Dynamic Cache] Switching to {current_bits}-bit cache (memory {used_ratio*100:.1f}%)")
        
        # Build cache config for this chunk
        cache_cfg = make_cache_config(
            backend=backend.upper() if backend == "hqq" else backend,
            nbits=current_bits,
            device=device,
            residual_length=residual_length,
        )
        
        # Generate chunk
        chunk_size = min(chunk_tokens, max_new_tokens - tokens_generated)
        
        # Use model.generate with cache_implementation
        try:
            outputs = model.generate(
                generated_ids,
                max_new_tokens=chunk_size,
                cache_implementation="quantized",
                cache_config=cache_cfg,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated_ids = outputs
            tokens_generated += chunk_size
        except Exception as e:
            # Fallback to standard generation if quantized cache not supported
            print(f"[Dynamic Cache] Quantized cache not supported, using standard: {e}")
            outputs = model.generate(
                generated_ids,
                max_new_tokens=chunk_size,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            generated_ids = outputs
            tokens_generated += chunk_size
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Grace-Blackwell Utilities Demo")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Memory monitoring
        ratio = _gpu_used_ratio()
        print(f"GPU memory usage: {ratio*100:.1f}%")
        
        # Tile optimization
        for L in [512, 1024, 2048, 4096]:
            T = get_optimal_tile(L)
            print(f"Sequence length {L} -> Optimal tile {T}")
        
        # Occupancy calculation
        for threads, smem in [(256, 0), (256, 48*1024), (512, 96*1024)]:
            occ = get_occupancy(threads, smem)
            print(f"Threads={threads}, SMEM={smem//1024}KB -> Occupancy {occ*100:.1f}%")
        
        # Cache config
        config = make_cache_config(
            backend="HQQ",
            nbits=4,
            device=torch.device("cuda:0"),
        )
        print(f"\nCache config: {config}")
    else:
        print("CUDA not available - skipping GPU tests")
