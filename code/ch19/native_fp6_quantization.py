"""Packed symmetric INT6 weight storage with an FP16/BF16 compute proxy.

The historical filename is retained for book links. The implementation is
uniform signed integer quantization, not FP6 E3M2 and not a native FP6 kernel.
It stores four zero-preserving six-bit codes in three bytes, then unpacks and
dequantizes weights before ``F.linear``. Reported latency therefore includes
unpacking and dequantization and describes only this proxy implementation.
"""
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.inference_mode()
def _benchmark_forward(
    module: nn.Module,
    x: torch.Tensor,
    warmup_iters: int,
    benchmark_iters: int,
) -> Tuple[torch.Tensor, float]:
    output = module(x)
    for _ in range(warmup_iters):
        output = module(x)

    count = max(benchmark_iters, 1)
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


# ============================================================================
# Packed INT6 quantization utilities
# ============================================================================

class PackedInt6Tensor:
    """Symmetric per-tensor INT6 storage with an exact zero code."""
    
    def __init__(self, data: torch.Tensor, dtype: torch.dtype = torch.float16):
        """
        Args:
            data: Floating-point tensor to quantize to signed INT6
            dtype: Output dtype for dequantization
        """
        self.shape = data.shape
        self.dtype = dtype
        self.device = data.device
        
        self.quantized_data, self.scales = self._quantize_int6(data)
        
    def _quantize_int6(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to symmetric values in [-31, 31] and pack their codes."""
        abs_max = data.abs().max()

        scale = abs_max / 31.0
        scale.masked_fill_(abs_max == 0, 1.0)

        codes = self._round_to_int6(data / scale)
        packed = self._pack_int6(codes)
        
        return packed, scale
    
    def _round_to_int6(self, data: torch.Tensor) -> torch.Tensor:
        """Return packed-code values for signed integers in [-31, 31]."""
        signed = torch.round(data).clamp(-31, 31).to(torch.int16)
        return (signed + 32).to(torch.uint8)
    
    def _pack_int6(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Pack 4 6-bit codes into 3 bytes.

        Layout: AAAAAA BBBBBB CCCCCC DDDDDD -> 3 bytes
        Byte 0: AAAAAABB
        Byte 1: BBBBCCCC
        Byte 2: CCDDDDDD

        The last group is zero-padded when the element count is not a
        multiple of 4. Returns a flat uint8 tensor of 3 bytes per group.
        """
        if quantized.dtype != torch.uint8:
            raise TypeError(f"_pack_int6 expects uint8 6-bit codes, got {quantized.dtype}")
        flat = quantized.flatten()

        # Pad to multiple of 4
        remainder = flat.numel() % 4
        if remainder != 0:
            padding = 4 - remainder
            flat = F.pad(flat, (0, padding))

        a, b, c, d = flat.reshape(-1, 4).unbind(dim=1)
        packed = torch.stack(
            (
                (a << 2) | (b >> 4),
                ((b & 0x0F) << 4) | (c >> 2),
                ((c & 0x03) << 6) | d,
            ),
            dim=1,
        )
        return packed.flatten()

    def dequantize(self) -> torch.Tensor:
        """Dequantize packed INT6 codes to the requested floating-point dtype."""
        unpacked = self._unpack_int6(self.quantized_data, self.shape.numel())
        data = (unpacked.to(torch.int16) - 32).to(torch.float32) * self.scales

        # Reshape to original shape
        data = data.reshape(self.shape)

        return data.to(self.dtype)

    def _unpack_int6(self, packed: torch.Tensor, numel: int) -> torch.Tensor:
        """Unpack each 3-byte group into 4 6-bit codes and drop the padding."""
        b0, b1, b2 = packed.reshape(-1, 3).unbind(dim=1)
        codes = torch.stack(
            (
                b0 >> 2,
                ((b0 & 0x03) << 4) | (b1 >> 4),
                ((b1 & 0x0F) << 2) | (b2 >> 6),
                b2 & 0x3F,
            ),
            dim=1,
        )
        return codes.flatten()[:numel]

    def memory_usage(self) -> int:
        """Return the bytes held by the packed codes and the scale tensor."""
        packed_bytes = self.quantized_data.numel() * self.quantized_data.element_size()
        return packed_bytes + self.scales.numel() * self.scales.element_size()


# ============================================================================
# Packed INT6 neural-network layers
# ============================================================================

class PackedInt6Linear(nn.Module):
    """Linear layer with packed symmetric INT6 weight storage.

    Features:
    - Six code bits per weight, 62.5% fewer than FP16 before scale metadata
    - FP16/BF16 activation
    - Dynamic dequantization during forward pass
    - FP16/BF16 ``F.linear`` compute
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Initialize with FP16 weights
        weight = torch.randn(out_features, in_features, dtype=dtype)
        self.weight_int6 = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        # Store FP16 weight temporarily for initialization
        self.register_buffer('_weight_fp16', weight)
    
    def quantize(self):
        """Convert weights to packed symmetric INT6 storage."""
        if self._weight_fp16 is not None:
            self.weight_int6 = PackedInt6Tensor(self._weight_fp16, dtype=self.dtype)
            # Clear FP16 weight to save memory
            self._weight_fp16 = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with packed INT6 weights.

        Unpacks and dequantizes the weights, then runs F.linear in the
        activation dtype. This path does not issue FP6 instructions.
        """
        if self.weight_int6 is None:
            raise RuntimeError("Call quantize() before forward pass")
        
        # Dequantize weights on-the-fly
        weight = self.weight_int6.dequantize()
        
        return F.linear(x, weight, self.bias)
    
    def memory_usage(self) -> dict:
        """Return memory usage statistics."""
        if self.weight_int6 is None:
            weight_mem = self._weight_fp16.numel() * self._weight_fp16.element_size()
        else:
            weight_mem = self.weight_int6.memory_usage()
        
        bias_mem = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        
        return {
            'weight_int6_bytes': weight_mem,
            'bias_bytes': bias_mem,
            'total_bytes': weight_mem + bias_mem,
            'weight_shape': (self.out_features, self.in_features)
        }


class PackedInt6MLP(nn.Module):
    """MLP with packed symmetric INT6 weights and floating-point compute.
    
    Suitable for transformer FFN layers. Each packed code uses six bits. The
    layer also stores one floating-point scale and any bias.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float16,
        *,
        use_int6: bool = True,
    ):
        super().__init__()
        
        self.use_int6 = use_int6
        if use_int6:
            self.fc1 = PackedInt6Linear(d_model, d_ff, dtype=dtype)
            self.fc2 = PackedInt6Linear(d_ff, d_model, dtype=dtype)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=True, dtype=dtype)
            self.fc2 = nn.Linear(d_ff, d_model, bias=True, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def quantize(self):
        """Quantize all eligible layers to packed INT6 storage."""
        for layer in (self.fc1, self.fc2):
            if hasattr(layer, "quantize"):
                layer.quantize()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with floating-point compute over dequantized INT6 weights."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _layer_total_bytes(layer: nn.Module) -> Tuple[int, dict]:
    if hasattr(layer, "memory_usage"):
        stats = layer.memory_usage()
        return int(stats["total_bytes"]), stats
    weight = layer.weight
    bias = layer.bias
    total = weight.numel() * weight.element_size()
    if bias is not None:
        total += bias.numel() * bias.element_size()
    return total, {"total_bytes": total}


def _mlp_total_bytes(mlp: PackedInt6MLP) -> int:
    return sum(_layer_total_bytes(layer)[0] for layer in (mlp.fc1, mlp.fc2))


# ============================================================================
# Benchmarking and Validation
# ============================================================================

def _build_matched_mlp_pair(
    d_model: int,
    d_ff: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[PackedInt6MLP, PackedInt6MLP]:
    """Build deterministic floating-point and INT6 proxies from identical weights."""
    baseline = PackedInt6MLP(
        d_model, d_ff, dropout=0.0, dtype=dtype, use_int6=False
    ).to(device).eval()
    quantized = PackedInt6MLP(
        d_model, d_ff, dropout=0.0, dtype=dtype, use_int6=True
    ).to(device).eval()
    with torch.no_grad():
        for baseline_layer, quantized_layer in (
            (baseline.fc1, quantized.fc1),
            (baseline.fc2, quantized.fc2),
        ):
            quantized_layer._weight_fp16.copy_(baseline_layer.weight)
            quantized_layer.bias.copy_(baseline_layer.bias)
    quantized.quantize()
    return baseline, quantized


def benchmark_int6_vs_fp16(
    *,
    batch_size: int = 128,
    seq_len: int = 512,
    d_model: int = 2048,
    d_ff: int = 8192,
):
    """
    Benchmark the packed INT6 proxy against the same weights in FP16.

    Metrics:
    - Weight and bias storage of each MLP
    - Inference latency
    - Output error from this INT6 quantizer on this seeded workload

    Without CUDA, each size is capped at the CPU limits below.
    """
    print("=" * 80)
    print("Packed INT6 Weight-Storage Proxy")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    warmup_iters = 10 if torch.cuda.is_available() else 2
    benchmark_iters = 100 if torch.cuda.is_available() else 5

    if not torch.cuda.is_available():
        batch_size = min(batch_size, 16)
        seq_len = min(seq_len, 256)
        d_model = min(d_model, 1024)
        d_ff = min(d_ff, 4096)
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {d_model}")
    print(f"  FFN dim: {d_ff}")
    print(f"  Device: {device}")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    
    # FP16 baseline
    print("\n" + "=" * 80)
    print("FP16 Baseline")
    print("=" * 80)
    mlp_fp16, mlp_int6 = _build_matched_mlp_pair(
        d_model,
        d_ff,
        device=device,
        dtype=dtype,
    )
    
    output_fp16, time_fp16 = _benchmark_forward(mlp_fp16, x, warmup_iters, benchmark_iters)

    mem_fp16 = _mlp_total_bytes(mlp_fp16) / 1024**2

    print(f"  Latency: {time_fp16 * 1000:.2f} ms")
    print(f"  Weight+bias memory: {mem_fp16:.2f} MiB")
    print(f"  Throughput: {batch_size * seq_len / time_fp16 / 1e6:.2f} M tokens/sec")
    
    # Packed INT6 proxy
    print("\n" + "=" * 80)
    print("Packed INT6 Weights with FP16 Compute")
    print("=" * 80)
    output_int6, time_int6 = _benchmark_forward(mlp_int6, x, warmup_iters, benchmark_iters)

    mem_int6 = _mlp_total_bytes(mlp_int6) / 1024**2

    print(f"  Latency: {time_int6 * 1000:.2f} ms")
    print(f"  Weight+bias memory: {mem_int6:.2f} MiB")
    print(f"  Throughput: {batch_size * seq_len / time_int6 / 1e6:.2f} M tokens/sec")
    
    # Numerical accuracy
    with torch.inference_mode():
        error = (output_fp16 - output_int6).abs().mean()
        denominator = output_fp16.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
        rel_error = error / denominator
    
    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print(f"  Measured FP16/INT6-proxy latency ratio: {time_fp16 / time_int6:.2f}x")
    print(f"  Weight+bias memory savings: {(1 - mem_int6 / mem_fp16) * 100:.1f}%")
    print(f"  Absolute error: {error:.6f}")
    print(f"  Relative error: {rel_error * 100:.2f}%")
    
    # Memory breakdown
    fp16_bytes, _ = _layer_total_bytes(mlp_fp16.fc1)
    int6_bytes, _ = _layer_total_bytes(mlp_int6.fc1)
    
    print("\n" + "=" * 80)
    print("Memory Breakdown (FC1 layer)")
    print("=" * 80)
    print(f"  FP16 weights: {fp16_bytes / 1024**2:.2f} MiB")
    print(f"  INT6 weights: {int6_bytes / 1024**2:.2f} MiB")
    print(f"  Compression: {fp16_bytes / int6_bytes:.2f}x")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("Packed INT6 weight storage:")
    print("  Six-bit codes use 62.5% fewer bits per weight than FP16 before scale metadata")
    print("  Six-bit codes use 25% fewer bits per weight than FP8 before scale metadata")
    print("  Codes use symmetric uniform INT6 with range [-31, 31] and exact zero")
    print("  Every forward unpacks and dequantizes weights before FP16 F.linear")
    print("  This proxy does not implement E3M2 or issue native FP6 instructions")


def report_int6_reconstruction_error() -> None:
    """Report reconstruction error for representative INT6 inputs."""
    print("\n" + "=" * 80)
    print("Packed INT6 Reconstruction Error")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test various ranges
    test_cases = [
        ("Small values", torch.randn(100, 100) * 0.1),
        ("Medium values", torch.randn(100, 100)),
        ("Large values", torch.randn(100, 100) * 10),
        ("Mixed range", torch.randn(100, 100) * torch.logspace(-2, 2, 100).unsqueeze(1)),
    ]
    
    for name, data in test_cases:
        data = data.to(device)
        
        # Quantize and dequantize
        int6_tensor = PackedInt6Tensor(data)
        reconstructed = int6_tensor.dequantize()
        
        # Compute error
        abs_error = (data - reconstructed).abs().mean()
        rel_error = abs_error / data.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
        max_error = (data - reconstructed).abs().max()
        
        print(f"\n{name}:")
        print(f"  Mean absolute error: {abs_error:.6f}")
        print(f"  Mean relative error: {rel_error * 100:.2f}%")
        print(f"  Max absolute error: {max_error:.6f}")
        print(f"  Memory usage: {int6_tensor.memory_usage()} bytes (vs {data.numel() * 2} bytes FP16)")
        print(f"  Compression ratio: {data.numel() * 2 / int6_tensor.memory_usage():.2f}x")


def print_key_takeaways() -> None:
    print("\nKey Takeaways:")
    print("  1. INT6 codes use 62.5% fewer bits per weight than FP16 before scale metadata")
    print("  2. Symmetric codes in [-31, 31] preserve zero exactly")
    print("  3. The measured proxy unpacks and dequantizes before FP16 compute")
    print("  4. The file does not implement FP6 E3M2 or a native FP6 kernel")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Packed Symmetric INT6 Weight-Storage Proxy")
    print("=" * 80)
    
    # Check device
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
    else:
        print("\nCUDA not available. Running the same INT6 pack, unpack, and FP16 compute proxy on CPU.")
    
    # Run tests
    report_int6_reconstruction_error()
    
    print("\n")
    benchmark_int6_vs_fp16()
    
    print("\n" + "=" * 80)
    print("Packed INT6 Proxy Complete")
    print("=" * 80)
    print_key_takeaways()
