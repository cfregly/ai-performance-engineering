"""
Native FP4 Quantization for Blackwell B200/B300
================================================

Demonstrates Blackwell's FP4 (4-bit floating point) quantization using
PyTorch's torch.float4_e2m1fn_x2 dtype. This format provides optimal
memory efficiency for inference workloads.

FP4 Format (E2M1):
- 1 sign bit
- 2 exponent bits (bias 1, range: 2^-1 to 2^2)
- 1 mantissa bit
- Dynamic range: ~0.5 to 6
- Precision: ~25% relative error
- 16 possible values
- Packed: 2 values per byte (float4_e2m1fn_x2)

Status in PyTorch 2.9.1:
- torch.float4_e2m1fn_x2 dtype: Available
- Tensor creation: Supported
- Copy/conversion: Not fully implemented
- Solution: Packed uint8 format with cached dequantization

Performance on B200:
- Memory bandwidth: 75% reduction vs FP16
- Model capacity: 4x larger models in same memory
- Best for inference where accuracy is less critical

Requirements:
- PyTorch 2.9+ with CUDA 13.0
- Blackwell GPU (B200/B300)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time
import math


def is_blackwell() -> bool:
    """Check if running on Blackwell GPU (B200/B300)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10


def has_native_fp4() -> bool:
    """Check if native FP4 dtype is available."""
    return hasattr(torch, 'float4_e2m1fn_x2')


def has_scaled_mm() -> bool:
    """Check if scaled matmul is available."""
    return hasattr(torch, '_scaled_mm')


def _benchmark_forward(module: nn.Module, x: torch.Tensor, warmup_iters: int, bench_iters: int) -> Tuple[torch.Tensor, float]:
    output = module(x)
    for _ in range(warmup_iters):
        output = module(x)

    count = max(bench_iters, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        for _ in range(count):
            output = module(x)
        end.record(current_stream)
        end.synchronize()
        return output, start.elapsed_time(end) / (count * 1000.0)

    start_time = time.perf_counter()
    for _ in range(count):
        output = module(x)
    return output, (time.perf_counter() - start_time) / count


# FP4 E2M1 representable values (positive)
# Sign bit gives us 16 total values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_SIGNED_VALUES = torch.cat((FP4_VALUES, -FP4_VALUES))
FP4_MAX = 6.0
_FP4_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}
_FP4_SIGNED_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_values_for(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return FP4_VALUES
    cached = _FP4_VALUES_CACHE.get(device)
    if cached is None:
        cached = FP4_VALUES.to(device=device)
        _FP4_VALUES_CACHE[device] = cached
    return cached


def _fp4_signed_values_for(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return FP4_SIGNED_VALUES
    cached = _FP4_SIGNED_VALUES_CACHE.get(device)
    if cached is None:
        cached = FP4_SIGNED_VALUES.to(device=device)
        _FP4_SIGNED_VALUES_CACHE[device] = cached
    return cached


def _unpack_fp4_codes(packed_data: torch.Tensor) -> torch.Tensor:
    unpacked = torch.empty(packed_data.numel() * 2, device=packed_data.device, dtype=torch.long)
    torch.bitwise_right_shift(packed_data, 4, out=unpacked[0::2])
    torch.bitwise_and(packed_data, 0x0F, out=unpacked[1::2])
    return unpacked


# ============================================================================
# FP4 Quantization Functions
# ============================================================================

def quantize_to_fp4_packed(
    tensor: torch.Tensor,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16/FP32 tensor to FP4 E2M1 format with per-block scaling.
    
    Returns packed uint8 tensor (2 FP4 values per byte) and scales.
    """
    device = tensor.device
    dtype = tensor.dtype
    
    # Flatten and pad to block size
    flat = tensor.flatten().float()
    n_elements = flat.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    
    if n_elements < padded_size:
        flat = F.pad(flat, (0, padded_size - n_elements))
    
    # Reshape to blocks
    blocks = flat.reshape(n_blocks, block_size)
    
    # Compute per-block scales
    block_absmax = blocks.abs().max(dim=1, keepdim=True).values
    scales = block_absmax / FP4_MAX
    scales = scales.clamp(min=1e-8)
    
    # Normalize and clamp to FP4 range
    normalized = blocks / scales
    normalized = normalized.clamp(-FP4_MAX, FP4_MAX)
    
    # Quantize to FP4 values
    fp4_vals = _fp4_values_for(device)
    abs_normalized = normalized.abs()
    
    # Find nearest FP4 value (vectorized)
    distances = (abs_normalized.unsqueeze(-1) - fp4_vals).abs()
    indices = distances.argmin(dim=-1).byte()  # 3-bit magnitude index
    signs = (normalized < 0).byte()  # 1-bit sign
    
    # Pack: 4-bit code = sign (1 bit) + magnitude index (3 bits)
    fp4_codes = (signs << 3) | indices
    
    # Pack pairs of 4-bit values into bytes
    flat_codes = fp4_codes.flatten()
    if flat_codes.numel() % 2 != 0:
        flat_codes = F.pad(flat_codes, (0, 1))
    
    pairs = flat_codes.reshape(-1, 2)
    packed = (pairs[:, 0] << 4) | pairs[:, 1]
    
    return packed.to(torch.uint8), scales.squeeze(-1).to(dtype)


def dequantize_from_fp4_packed(
    packed_data: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
    block_size: int = 128,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize FP4 packed data back to original dtype.
    """
    device = packed_data.device
    signed_fp4_vals = _fp4_signed_values_for(device)
    
    # Unpack bytes to pairs of 4-bit codes
    unpacked = _unpack_fp4_codes(packed_data)
    
    # Decode FP4 directly from the packed sign+magnitude code.
    values = signed_fp4_vals[unpacked]
    
    # Reshape to blocks and apply scales
    n_blocks = len(scales)
    n_elements = n_blocks * block_size
    blocks = values[:n_elements].reshape(n_blocks, block_size)
    blocks.mul_(scales.unsqueeze(-1))
    
    # Reshape to original
    n_orig = math.prod(original_shape)
    flat = blocks.flatten()[:n_orig]
    return flat.reshape(original_shape).to(dtype)


# ============================================================================
# FP4 Linear Layer with Cached Dequantization
# ============================================================================

class FP4Linear(nn.Module):
    """
    Linear layer with FP4 quantized weights and cached dequantization.
    
    Memory savings come from storing weights in FP4 format.
    Performance is maintained by caching dequantized weights after first use.
    
    Modes:
    - 'storage': Keep FP4 packed, dequantize on each forward (max memory savings)
    - 'cached': Dequantize once, cache for fast inference (balanced)
    - 'fp8': Use FP8 tensor cores via _scaled_mm (best throughput on Blackwell)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        block_size: int = 128,
        mode: str = 'cached',  # 'storage', 'cached', or 'fp8'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.block_size = block_size
        self.mode = mode
        
        # Initialize FP16 weights
        weight = torch.empty(out_features, in_features, dtype=dtype)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        self.register_buffer('_weight_fp16', weight)
        self.register_buffer('weight_packed', None)
        self.register_buffer('weight_scales', None)
        self.register_buffer('_weight_cache', None)  # Cached dequantized weights
        self.register_buffer('_weight_fp8_cache', None)
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        self.register_buffer('_input_fp8_buffer', torch.empty(0, dtype=fp8_dtype), persistent=False)
        self.register_buffer('_fp8_scale_a', torch.ones(1, dtype=torch.float32), persistent=False)
        self.register_buffer('_fp8_scale_b', torch.ones(1, dtype=torch.float32), persistent=False)
        self._quantized = False
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize(self) -> None:
        """Convert weights to FP4 format."""
        if self._weight_fp16 is not None:
            packed, scales = quantize_to_fp4_packed(
                self._weight_fp16,
                block_size=self.block_size,
            )
            self.weight_packed = packed
            self.weight_scales = scales
            self._weight_fp16 = None
            self._weight_cache = None
            self._weight_fp8_cache = None
            self._quantized = True
    
    def _get_weight(self) -> torch.Tensor:
        """Get weights with caching for performance."""
        if not self._quantized:
            return self._weight_fp16
        
        # Check cache first
        if self._weight_cache is not None:
            return self._weight_cache
        
        # Dequantize
        weight = dequantize_from_fp4_packed(
            self.weight_packed,
            self.weight_scales,
            torch.Size([self.out_features, self.in_features]),
            self.block_size,
            self.dtype
        )
        
        # Cache if in cached mode
        if self.mode == 'cached':
            self._weight_cache = weight
        
        return weight
    
    def clear_cache(self) -> None:
        """Clear weight cache to free memory."""
        self._weight_cache = None
        self._weight_fp8_cache = None

    def _get_weight_fp8(self) -> torch.Tensor:
        if self._weight_fp8_cache is not None:
            return self._weight_fp8_cache
        weight = self._get_weight()
        weight_fp8 = weight.to(torch.float8_e4m3fn).contiguous()
        self._weight_fp8_cache = weight_fp8
        return weight_fp8

    def _activation_fp8_buffer(self, x_2d: torch.Tensor) -> torch.Tensor:
        input_fp8 = self._input_fp8_buffer
        if (
            input_fp8.dim() != x_2d.dim()
            or input_fp8.size(0) < x_2d.size(0)
            or tuple(input_fp8.shape[1:]) != tuple(x_2d.shape[1:])
            or input_fp8.device != x_2d.device
            or input_fp8.dtype != torch.float8_e4m3fn
        ):
            input_fp8 = torch.empty(
                x_2d.shape,
                device=x_2d.device,
                dtype=torch.float8_e4m3fn,
            )
            self._input_fp8_buffer = input_fp8
        return input_fp8[: x_2d.size(0)]

    def _fp8_scale_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_scale_a.device != device or self._fp8_scale_a.dtype != torch.float32:
            self._fp8_scale_a = torch.ones(1, device=device, dtype=torch.float32)
        if self._fp8_scale_b.device != device or self._fp8_scale_b.dtype != torch.float32:
            self._fp8_scale_b = torch.ones(1, device=device, dtype=torch.float32)
        return self._fp8_scale_a, self._fp8_scale_b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FP4 weights."""
        if self.mode == 'fp8' and self._quantized and has_scaled_mm() and is_blackwell():
            return self._forward_fp8(x)
        
        weight = self._get_weight()
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        return F.linear(x, weight, self.bias)
    
    def _forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using FP8 tensor cores for acceleration."""
        weight_fp8 = self._get_weight_fp8()
        
        # Reshape for matmul
        batch_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp8 = self._activation_fp8_buffer(x_2d)
        x_fp8.copy_(x_2d)
        
        # Scales for _scaled_mm
        scale_a, scale_b = self._fp8_scale_buffers(x.device)
        
        # _scaled_mm: (M, K) @ (N, K).T -> (M, N)
        result = torch._scaled_mm(
            x_fp8, weight_fp8.T,
            scale_a, scale_b,
            out_dtype=self.dtype
        )
        
        output = result.reshape(*batch_shape, -1)
        if self.bias is not None:
            output.add_(self.bias)
        return output
    
    @property
    def compression_ratio(self) -> float:
        """Return compression ratio vs FP16."""
        fp16_bytes = self.out_features * self.in_features * 2
        if self._quantized:
            fp4_bytes = (self.weight_packed.numel() + 
                        self.weight_scales.numel() * self.weight_scales.element_size())
            return fp16_bytes / fp4_bytes
        return 1.0
    
    @property
    def memory_bytes(self) -> int:
        """Return memory usage in bytes (packed format only)."""
        if self._quantized:
            return (self.weight_packed.numel() + 
                   self.weight_scales.numel() * self.weight_scales.element_size())
        return self._weight_fp16.numel() * self._weight_fp16.element_size()


class FP4MLP(nn.Module):
    """MLP with FP4 quantized weights."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
        block_size: int = 128,
        mode: str = 'cached',
    ):
        super().__init__()
        self.fc1 = FP4Linear(d_model, d_ff, dtype=dtype, block_size=block_size, mode=mode)
        self.fc2 = FP4Linear(d_ff, d_model, dtype=dtype, block_size=block_size, mode=mode)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()
    
    def quantize(self) -> None:
        """Quantize all layers to FP4."""
        self.fc1.quantize()
        self.fc2.quantize()
    
    def clear_cache(self) -> None:
        """Clear weight caches."""
        self.fc1.clear_cache()
        self.fc2.clear_cache()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_fp4():
    """Benchmark FP4 quantization on Blackwell."""
    print("=" * 80)
    print("FP4 Quantization Benchmark for Blackwell B200/B300")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    # Configuration
    if torch.cuda.is_available():
        batch_size = 64
        seq_len = 2048
        d_model = 4096
        d_ff = 16384
        warmup_iters = 10
        bench_iters = 50
    else:
        batch_size = 8
        seq_len = 256
        d_model = 1024
        d_ff = 4096
        warmup_iters = 2
        bench_iters = 5
    
    print(f"\nSystem Info:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: {device}")
    print(f"  Blackwell: {is_blackwell()}")
    print(f"  Native FP4 dtype: {has_native_fp4()}")
    print(f"  Scaled MM (FP8): {has_scaled_mm()}")
    
    print(f"\nConfiguration:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, Model: {d_model}, FF: {d_ff}")
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    
    results = {}
    
    # ---- FP16 Baseline ----
    print("\n" + "-" * 40)
    print("FP16 Baseline")
    print("-" * 40)
    
    mlp_fp16 = nn.Sequential(
        nn.Linear(d_model, d_ff, dtype=dtype),
        nn.GELU(),
        nn.Linear(d_ff, d_model, dtype=dtype),
    ).to(device)
    
    out_fp16, time_fp16 = _benchmark_forward(mlp_fp16, x, warmup_iters, bench_iters)
    
    mem_fp16 = sum(p.numel() * p.element_size() for p in mlp_fp16.parameters()) / 1024**2
    
    print(f"  Latency: {time_fp16 * 1000:.2f} ms")
    print(f"  Memory: {mem_fp16:.2f} MB")
    results['fp16'] = {'time': time_fp16, 'mem': mem_fp16, 'output': out_fp16}
    
    # ---- FP4 Storage Mode (max compression, dequant each forward) ----
    print("\n" + "-" * 40)
    print("FP4 Storage Mode (dequant each forward)")
    print("-" * 40)
    
    mlp_fp4_storage = FP4MLP(d_model, d_ff, dtype=dtype, mode='storage').to(device)
    mlp_fp4_storage.quantize()
    
    print(f"  Compression ratio: {mlp_fp4_storage.fc1.compression_ratio:.2f}x")
    
    out_storage, time_storage = _benchmark_forward(mlp_fp4_storage, x, warmup_iters, bench_iters)
    
    mem_storage = (mlp_fp4_storage.fc1.memory_bytes + mlp_fp4_storage.fc2.memory_bytes) / 1024**2
    
    print(f"  Latency: {time_storage * 1000:.2f} ms")
    print(f"  Memory: {mem_storage:.2f} MB")
    print(f"  Speedup: {time_fp16 / time_storage:.2f}x")
    results['fp4_storage'] = {'time': time_storage, 'mem': mem_storage}
    
    # ---- FP4 Cached Mode (dequant once, reuse) ----
    print("\n" + "-" * 40)
    print("FP4 Cached Mode (dequant once, fast inference)")
    print("-" * 40)
    
    mlp_fp4_cached = FP4MLP(d_model, d_ff, dtype=dtype, mode='cached').to(device)
    mlp_fp4_cached.quantize()
    
    out_cached, time_cached = _benchmark_forward(mlp_fp4_cached, x, warmup_iters, bench_iters)
    
    print(f"  Latency: {time_cached * 1000:.2f} ms")
    print(f"  Memory (packed): {mem_storage:.2f} MB")
    print(f"  Speedup: {time_fp16 / time_cached:.2f}x")
    results['fp4_cached'] = {'time': time_cached, 'output': out_cached}
    
    # ---- FP4 + FP8 Bridge (tensor core acceleration) ----
    if has_scaled_mm() and is_blackwell():
        print("\n" + "-" * 40)
        print("FP4 Storage + FP8 Tensor Cores")
        print("-" * 40)
        
        mlp_fp4_fp8 = FP4MLP(d_model, d_ff, dtype=dtype, mode='fp8').to(device)
        mlp_fp4_fp8.quantize()
        
        out_fp8, time_fp8 = _benchmark_forward(mlp_fp4_fp8, x, warmup_iters, bench_iters)
        
        print(f"  Latency: {time_fp8 * 1000:.2f} ms")
        print(f"  Memory (packed): {mem_storage:.2f} MB")
        print(f"  Speedup: {time_fp16 / time_fp8:.2f}x")
        results['fp4_fp8'] = {'time': time_fp8, 'output': out_fp8}
    
    # ---- Summary ----
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\n{'Mode':<30} {'Latency':<15} {'Memory':<15} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'FP16 Baseline':<30} {time_fp16*1000:>10.2f} ms  {mem_fp16:>10.2f} MB  {'1.00x':>10}")
    print(f"{'FP4 Storage':<30} {time_storage*1000:>10.2f} ms  {mem_storage:>10.2f} MB  {time_fp16/time_storage:>10.2f}x")
    print(f"{'FP4 Cached':<30} {time_cached*1000:>10.2f} ms  {mem_storage:>10.2f} MB  {time_fp16/time_cached:>10.2f}x")
    if 'fp4_fp8' in results:
        print(f"{'FP4 + FP8 Tensor Cores':<30} {results['fp4_fp8']['time']*1000:>10.2f} ms  {mem_storage:>10.2f} MB  {time_fp16/results['fp4_fp8']['time']:>10.2f}x")
    
    # Accuracy
    print(f"\nAccuracy (vs FP16):")
    with torch.inference_mode():
        for name, data in [('Cached', out_cached)]:
            error = (out_fp16 - data).abs()
            error_stats = torch.empty(2, device=error.device, dtype=torch.float32)
            error_stats[0].copy_(error.mean())
            error_stats[1].copy_(out_fp16.abs().mean())
            error_stats_host = error_stats.detach().cpu()
            mean_err = float(error_stats_host[0])
            fp16_abs_mean = float(error_stats_host[1])
            rel_err = mean_err / fp16_abs_mean * 100
            print(f"  {name}: Mean abs error = {mean_err:.6f}, Relative = {rel_err:.2f}%")
    
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("  • Use 'cached' mode for inference (best latency)")
    print("  • Use 'storage' mode for multi-model serving (min memory)")
    print("  • Use 'fp8' mode on Blackwell for tensor core acceleration")
    print("=" * 80)


def demonstrate_fp4_capacity():
    """Show FP4's model capacity benefits."""
    print("\n" + "=" * 80)
    print("FP4 Model Capacity Comparison")
    print("=" * 80)
    
    configs = [
        ("7B params", 4096, 11008, 32),
        ("13B params", 5120, 13824, 40),
        ("70B params", 8192, 28672, 80),
        ("405B params", 16384, 53248, 126),
    ]
    
    print(f"\n{'Model':<15} {'FP16':<12} {'FP8':<12} {'FP4':<12} {'Capacity Gain'}")
    print("-" * 65)
    
    for name, d_model, d_ff, n_layers in configs:
        params = (d_model * d_ff + d_ff * d_model) * n_layers
        
        mem_fp16 = params * 2 / 1024**3
        mem_fp8 = params * 1 / 1024**3
        mem_fp4 = params * 0.5 / 1024**3
        
        gain = mem_fp16 / mem_fp4
        
        print(f"{name:<15} {mem_fp16:>10.1f}GB {mem_fp8:>10.1f}GB {mem_fp4:>10.1f}GB {gain:>10.1f}x")
    
    print("\n" + "=" * 80)
    print("FP4 Use Cases:")
    print("  • Draft models for speculative decoding")
    print("  • Cost-optimized inference deployment")
    print("  • Multi-model serving on single GPU")
    print("  • Edge devices with memory constraints")
    print("")
    print("FP4 E2M1 Value Table:")
    print("  Positive: 0, 0.5, 1, 1.5, 2, 3, 4, 6")
    print("  Negative: -0, -0.5, -1, -1.5, -2, -3, -4, -6")
    print("  16 total representable values")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("FP4 Quantization for Blackwell B200/B300")
    print("Packed uint8 format with per-block scaling")
    print("=" * 80)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
    
    benchmark_fp4()
    demonstrate_fp4_capacity()
