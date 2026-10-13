"""L2 Cache Utilities for Benchmark Fairness.

This module provides utilities for L2 cache management during benchmarking,
following Triton's best practices adapted to work across different NVIDIA
GPU architectures (Ampere, Hopper, Blackwell, etc.).

Key Features:
- Dynamic L2 cache size detection from hardware
- Architecture-specific fallback defaults
- Cache clearing for fair memory-bound comparisons

Background:
- L1 cache (~128KB per SM): Auto-evicted during kernel execution, not shared
- L2 cache (40-96MB shared): Persists across kernels, can artificially speed up iterations
- Triton's approach: Write buffer > L2 size to force full eviction

Reference:
    https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# Architecture-specific L2 cache sizes (in MB)
# These are fallback defaults if dynamic detection fails
# Source: NVIDIA documentation and empirical measurements
L2_CACHE_DEFAULTS_MB = {
    # Blackwell (SM 10.0+)
    "blackwell": 96.0,      # B100/B200: 96MB L2
    
    # Hopper (SM 9.0)
    "hopper": 50.0,         # H100/H200: 50MB L2 (H200 may have more)
    
    # Ampere (SM 8.0)
    "ampere": 40.0,         # A100: 40MB L2
    "ampere_consumer": 6.0, # RTX 30xx: 6MB L2
    
    # Turing (SM 7.5)
    "turing": 6.0,          # RTX 20xx: 6MB L2
    
    # Volta (SM 7.0)
    "volta": 6.0,           # V100: 6MB L2
    
    # Pascal (SM 6.x)
    "pascal": 4.0,          # P100: 4MB L2
    
    # Safe default for unknown architectures
    "unknown": 50.0,        # Conservative default
}


@dataclass
class L2CacheInfo:
    """Information about L2 cache size and source."""
    size_bytes: int
    size_mb: float
    source: str  # "hardware", "architecture", "default"
    architecture: str
    compute_capability: str
    

def _get_architecture_from_compute_capability(major: int, minor: int) -> str:
    """Map compute capability to architecture name.

    SM 12.x labels match the capability probe and core.harness.arch_config.
    Capabilities newer than SM 12.x are not assigned a family.
    """
    if (major, minor) == (12, 1):
        return "grace_blackwell"  # GB10
    elif major == 12:
        return "blackwell_consumer"  # GeForce RTX 50 and RTX PRO
    elif major > 12:
        return "unknown"
    elif major >= 10:
        return "blackwell"
    elif major >= 9:
        return "hopper"
    elif major >= 8:
        if minor >= 6:  # SM 8.6+ are consumer Ampere (RTX 30xx)
            return "ampere_consumer"
        return "ampere"
    elif major >= 7:
        if minor >= 5:
            return "turing"
        return "volta"
    elif major >= 6:
        return "pascal"
    else:
        return "unknown"


def _get_fallback_l2_size_mb(architecture: str) -> float:
    """Get fallback L2 cache size based on architecture."""
    return L2_CACHE_DEFAULTS_MB.get(architecture, L2_CACHE_DEFAULTS_MB["unknown"])


@functools.lru_cache(maxsize=8)
def detect_l2_cache_size(device_index: int = 0) -> L2CacheInfo:
    """Detect L2 cache size for the specified GPU.
    
    Detection methods (in order of preference):
    1. PyTorch device properties (l2_cache_size attribute)
    2. Hardware capabilities cache (artifacts/hardware_capabilities.json)
    3. Architecture-based defaults
    
    Args:
        device_index: CUDA device index (default: 0)
        
    Returns:
        L2CacheInfo with size and detection source
    """
    if torch is None or not torch.cuda.is_available():
        return L2CacheInfo(
            size_bytes=int(50 * 1024 * 1024),  # 50MB default
            size_mb=50.0,
            source="default",
            architecture="unknown",
            compute_capability="unknown",
        )
    
    try:
        props = torch.cuda.get_device_properties(device_index)
        major, minor = props.major, props.minor
        architecture = _get_architecture_from_compute_capability(major, minor)
        compute_cap = f"{major}.{minor}"
        
        # Method 1: Try to get from PyTorch device properties
        # Note: l2_cache_size may be 0 or missing on some PyTorch versions
        l2_cache_bytes = getattr(props, 'l2_cache_size', 0) or getattr(props, 'L2_cache_size', 0)
        
        if l2_cache_bytes > 0:
            return L2CacheInfo(
                size_bytes=l2_cache_bytes,
                size_mb=l2_cache_bytes / (1024 * 1024),
                source="hardware",
                architecture=architecture,
                compute_capability=compute_cap,
            )
        
        # Method 2: Try hardware_capabilities cache
        try:
            from core.harness.hardware_capabilities import detect_capabilities
            caps = detect_capabilities(device_index)
            if caps and caps.l2_cache_kb and caps.l2_cache_kb > 0:
                size_bytes = int(caps.l2_cache_kb * 1024)
                return L2CacheInfo(
                    size_bytes=size_bytes,
                    size_mb=size_bytes / (1024 * 1024),
                    source="hardware",
                    architecture=architecture,
                    compute_capability=compute_cap,
                )
        except Exception:
            pass  # Fall through to defaults
        
        # Method 3: Architecture-based defaults
        fallback_mb = _get_fallback_l2_size_mb(architecture)
        return L2CacheInfo(
            size_bytes=int(fallback_mb * 1024 * 1024),
            size_mb=fallback_mb,
            source="architecture",
            architecture=architecture,
            compute_capability=compute_cap,
        )
        
    except Exception as e:
        # Ultimate fallback
        return L2CacheInfo(
            size_bytes=int(50 * 1024 * 1024),
            size_mb=50.0,
            source="default",
            architecture="unknown",
            compute_capability="unknown",
        )


def get_l2_flush_buffer_size(device_index: int = 0, margin: float = 1.1) -> int:
    """Get the buffer size needed to flush L2 cache.
    
    Returns a buffer size slightly larger than L2 cache to ensure
    complete eviction of cached data.
    
    Args:
        device_index: CUDA device index (default: 0)
        margin: Multiplier for safety margin (default: 1.1 = 10% extra)
        
    Returns:
        Buffer size in bytes
    """
    info = detect_l2_cache_size(device_index)
    # Add margin to ensure complete eviction
    return int(info.size_bytes * margin)


def create_l2_flush_buffer(device: Optional["torch.device"] = None) -> "torch.Tensor":
    """Create a buffer suitable for flushing L2 cache.
    
    The buffer is sized to be slightly larger than the detected L2 cache
    to ensure complete eviction when written to.
    
    Args:
        device: CUDA device (default: current device)
        
    Returns:
        torch.Tensor buffer for L2 flushing
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    if device is None:
        device = torch.device("cuda")
    
    device_index = device.index if device.index is not None else 0
    buffer_size_bytes = get_l2_flush_buffer_size(device_index)
    
    # Create float32 tensor (4 bytes per element)
    num_elements = buffer_size_bytes // 4
    return torch.empty(num_elements, dtype=torch.float32, device=device)


def flush_l2_cache(device: Optional["torch.device"] = None, buffer: Optional["torch.Tensor"] = None) -> None:
    """Flush L2 cache by writing to a large buffer.
    
    This implements Triton's approach to L2 cache clearing:
    - Write zeros to a buffer larger than L2 cache size
    - Synchronize to ensure write completes
    - Forces eviction of all cached data
    
    Args:
        device: CUDA device (default: current device)
        buffer: Optional pre-allocated buffer (for repeated use without allocation)
        
    Example:
        # Simple usage - allocates buffer each time
        flush_l2_cache()
        
        # Efficient usage - reuse buffer
        flush_buffer = create_l2_flush_buffer()
        for _ in range(iterations):
            flush_l2_cache(buffer=flush_buffer)
    """
    if torch is None or not torch.cuda.is_available():
        return
    
    if device is None:
        device = torch.device("cuda")
    
    if buffer is not None:
        # Use provided buffer
        buffer.zero_()
    else:
        # Create temporary buffer
        temp_buffer = create_l2_flush_buffer(device)
        temp_buffer.zero_()
        del temp_buffer
    
    torch.cuda.synchronize(device)


def format_l2_cache_report(device_index: int = 0) -> str:
    """Format a human-readable report of L2 cache detection.
    
    Args:
        device_index: CUDA device index
        
    Returns:
        Multi-line string with L2 cache information
    """
    info = detect_l2_cache_size(device_index)
    
    lines = [
        "L2 Cache Information",
        "=" * 40,
        f"Size: {info.size_mb:.1f} MB ({info.size_bytes:,} bytes)",
        f"Detection: {info.source}",
        f"Architecture: {info.architecture}",
        f"Compute Capability: {info.compute_capability}",
        "",
        "Flush Buffer Size:",
        f"  Recommended: {get_l2_flush_buffer_size(device_index) / (1024*1024):.1f} MB",
        "",
        "Architecture Defaults:",
    ]
    
    for arch, size_mb in sorted(L2_CACHE_DEFAULTS_MB.items()):
        marker = " <-- current" if arch == info.architecture else ""
        lines.append(f"  {arch}: {size_mb:.1f} MB{marker}")
    
    return "\n".join(lines)


# Export for easy access
__all__ = [
    "L2CacheInfo",
    "L2_CACHE_DEFAULTS_MB",
    "detect_l2_cache_size",
    "get_l2_flush_buffer_size",
    "create_l2_flush_buffer",
    "flush_l2_cache",
    "format_l2_cache_report",
]






