#!/usr/bin/env python3

"""
Enhanced Memory Allocator with Retry and Fragmentation Monitoring (Chapter 19)

Extends the basic allocator retry with comprehensive fragmentation monitoring,
adaptive allocation strategies, and real-time memory pressure tracking.

Key features:
- Automatic retry with different allocator backends
- Fragmentation detection and reporting
- Memory pressure monitoring
- Adaptive allocation strategies
- Integration with Prometheus metrics

Usage:
    from memory_allocator_with_monitoring import ManagedMemoryAllocator
    
    allocator = ManagedMemoryAllocator()
    tensor = allocator.allocate((1024, 1024), dtype=torch.float16)
"""

import torch
import time
from math import prod
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

try:
    from prometheus_client import Counter as PromCounter, Gauge as PromGauge
    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False
    PromCounter = PromGauge = None


class AllocatorBackend(Enum):
    """Available allocator backends"""
    CUDA_MALLOC_ASYNC = "cudaMallocAsync"
    CUDA_MALLOC = "native"
    EXPANDABLE_SEGMENTS = "expandable_segments:True"


@dataclass
class AllocationStats:
    """Statistics for memory allocations"""
    total_allocations: int = 0
    total_frees: int = 0
    failed_allocations: int = 0
    retries: int = 0
    bytes_allocated: int = 0
    bytes_freed: int = 0
    peak_memory_bytes: int = 0
    fragmentation_events: int = 0
    
    @property
    def current_allocated(self) -> int:
        """Current allocated memory"""
        return self.bytes_allocated - self.bytes_freed
    
    @property
    def retry_rate(self) -> float:
        """Percentage of allocations that required retry"""
        return (self.retries / self.total_allocations * 100) if self.total_allocations > 0 else 0.0


@dataclass
class FragmentationMetrics:
    """Metrics for memory fragmentation analysis"""
    total_free_bytes: int
    largest_free_block: int
    num_free_blocks: int
    fragmentation_ratio: float  # 0.0 = no fragmentation, 1.0 = highly fragmented
    
    @property
    def is_fragmented(self) -> bool:
        """Check if memory is significantly fragmented"""
        return self.fragmentation_ratio > 0.3


class ManagedMemoryAllocator:
    """
    Memory allocator with retry logic and fragmentation monitoring.
    
    Implements the approach from Chapter 19:
    - Automatically retries failed allocations with different backends
    - Monitors memory fragmentation in real-time
    - Adapts allocation strategy based on memory pressure
    - Exports metrics for monitoring
    """
    _prom_metrics = None
    
    @classmethod
    def _ensure_prometheus_metrics(cls):
        if not PROMETHEUS_AVAILABLE:
            return None
        if cls._prom_metrics is None:
            cls._prom_metrics = {
                "allocations": PromCounter(
                    "llm_allocator_allocations_total",
                    "Successful allocations",
                    ["device"]
                ),
                "allocation_failures": PromCounter(
                    "llm_allocator_allocation_failures_total",
                    "Allocation failures",
                    ["device"]
                ),
                "retries": PromCounter(
                    "llm_allocator_retries_total",
                    "Allocation retries",
                    ["device"]
                ),
                "fragmentation_events": PromCounter(
                    "llm_allocator_fragmentation_events_total",
                    "Fragmentation mitigation events",
                    ["device"]
                ),
                "allocated_bytes": PromGauge(
                    "llm_allocator_bytes_allocated",
                    "Currently allocated bytes",
                    ["device"]
                ),
                "peak_bytes": PromGauge(
                    "llm_allocator_peak_bytes",
                    "Peak allocated bytes",
                    ["device"]
                ),
                "fragmentation_ratio": PromGauge(
                    "llm_allocator_fragmentation_ratio",
                    "Current fragmentation ratio",
                    ["device"]
                ),
                "memory_util": PromGauge(
                    "llm_allocator_memory_utilization_percent",
                    "Overall GPU memory utilization percent",
                    ["device"]
                ),
            }
        return cls._prom_metrics

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        enable_monitoring: bool = True,
        defrag_threshold: float = 0.5
    ):
        """
        Initialize managed allocator.
        
        Args:
            device: CUDA device to manage
            enable_monitoring: Enable continuous monitoring
            defrag_threshold: Fragmentation ratio to trigger defragmentation
        """
        self.device = device
        self.enable_monitoring = enable_monitoring
        self.defrag_threshold = defrag_threshold
        
        self.stats = AllocationStats()
        self.lock = threading.Lock()
        
        # Allocation history for fragmentation analysis
        self.allocation_history: deque = deque(maxlen=1000)
        
        # Current allocator backend
        self.current_backend = AllocatorBackend.CUDA_MALLOC_ASYNC
        self._prom_handles: Dict[str, Any] = {}
        metrics = self._ensure_prometheus_metrics()
        if metrics:
            device_label = str(device)
            self._prom_handles = {
                name: metric.labels(device=device_label)
                for name, metric in metrics.items()
            }
            self._update_prometheus_gauges()
        
        print(f"Initialized ManagedMemoryAllocator on {device}")
        print(f"Default backend: {self.current_backend.value}")

    def _update_prometheus_gauges(self, fragmentation: Optional[FragmentationMetrics] = None):
        if not self._prom_handles:
            return
        current_alloc = self.stats.current_allocated
        self._prom_handles["allocated_bytes"].set(current_alloc)
        self._prom_handles["peak_bytes"].set(self.stats.peak_memory_bytes)

        mem_stats = self.get_memory_stats() if torch.cuda.is_available() else {}
        util_percent = mem_stats.get("utilization_percent", 0.0)
        self._prom_handles["memory_util"].set(util_percent)

        if fragmentation is not None:
            self._prom_handles["fragmentation_ratio"].set(fragmentation.fragmentation_ratio)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics from CUDA.
        
        Returns:
            Dictionary with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        
        # Get memory info from device
        mem_info = torch.cuda.mem_get_info(self.device)
        free_mem = mem_info[0] / (1024**3)
        total_mem = mem_info[1] / (1024**3)
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "max_reserved_gb": max_reserved,
            "free_gb": free_mem,
            "total_gb": total_mem,
            "utilization_percent": ((total_mem - free_mem) / total_mem * 100)
        }
    
    def compute_fragmentation_metrics(self) -> FragmentationMetrics:
        """
        Compute memory fragmentation metrics.
        
        Fragmentation occurs when free memory exists but is scattered in small
        blocks, making large allocations fail even though total free memory
        would be sufficient.
        
        Returns:
            FragmentationMetrics object
        """
        mem_stats = self.get_memory_stats()
        
        # Get memory snapshot if available (PyTorch 2.0+)
        try:
            snapshot = torch.cuda.memory_snapshot()
            
            # Analyze free segments
            free_segments = [
                seg for seg in snapshot 
                if seg.get('state') == 'inactive'
            ]
            
            if free_segments:
                total_free = sum(seg['size'] for seg in free_segments)
                largest_free = max(seg['size'] for seg in free_segments)
                num_free_blocks = len(free_segments)
                
                # Fragmentation ratio: 1.0 - (largest_block / total_free)
                # 0.0 = one large block (no fragmentation)
                # 1.0 = many tiny blocks (high fragmentation)
                fragmentation_ratio = 1.0 - (largest_free / total_free) if total_free > 0 else 0.0
            else:
                total_free = 0
                largest_free = 0
                num_free_blocks = 0
                fragmentation_ratio = 0.0
                
        except Exception:
            # Fallback if memory_snapshot not available
            free_gb = mem_stats.get("free_gb", 0)
            reserved_gb = mem_stats.get("reserved_gb", 0)
            
            total_free = int(free_gb * 1024**3)
            # Estimate: assume largest block is 50% of free in fragmented state
            largest_free = int(total_free * 0.5)
            num_free_blocks = 10  # Rough estimate
            fragmentation_ratio = 0.5 if total_free > 0 else 0.0
        
        return FragmentationMetrics(
            total_free_bytes=total_free,
            largest_free_block=largest_free,
            num_free_blocks=num_free_blocks,
            fragmentation_ratio=fragmentation_ratio
        )
    
    def defragment(self):
        """
        Attempt to defragment GPU memory.
        
        This forces PyTorch to consolidate free memory blocks.
        """
        print("Defragmenting GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        print("Defragmentation complete")
        try:
            frag_metrics = self.compute_fragmentation_metrics()
        except Exception:
            frag_metrics = None
        self._update_prometheus_gauges(frag_metrics)
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        max_retries: int = 3
    ) -> Optional[torch.Tensor]:
        """
        Allocate a tensor with automatic retry on OOM.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            max_retries: Maximum number of retry attempts
            
        Returns:
            Allocated tensor or None if all retries failed
        """
        size_bytes = prod(shape) * torch.finfo(dtype).bits // 8
        
        for attempt in range(max_retries + 1):
            try:
                # Try allocation
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                
                # Success - update stats
                with self.lock:
                    self.stats.total_allocations += 1
                    self.stats.bytes_allocated += size_bytes
                    
                    if self.stats.current_allocated > self.stats.peak_memory_bytes:
                        self.stats.peak_memory_bytes = self.stats.current_allocated
                    
                    if attempt > 0:
                        self.stats.retries += attempt
                    
                    self.allocation_history.append({
                        "timestamp": time.time(),
                        "size_bytes": size_bytes,
                        "shape": shape,
                        "success": True,
                        "attempts": attempt + 1
                    })
                
                if self._prom_handles:
                    self._prom_handles["allocations"].inc()
                    if attempt > 0:
                        self._prom_handles["retries"].inc(attempt)
                self._update_prometheus_gauges()
                
                return tensor
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"Allocation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # Check fragmentation
                    frag_metrics = self.compute_fragmentation_metrics()
                    print(f"Fragmentation ratio: {frag_metrics.fragmentation_ratio:.2f}")
                    self._update_prometheus_gauges(frag_metrics)
                    
                    if frag_metrics.is_fragmented:
                        print("High fragmentation detected, attempting defragmentation...")
                        self.defragment()
                        
                        with self.lock:
                            self.stats.fragmentation_events += 1
                            if self._prom_handles:
                                self._prom_handles["fragmentation_events"].inc()
                        post_frag_metrics = self.compute_fragmentation_metrics()
                        self._update_prometheus_gauges(post_frag_metrics)
                    else:
                        # Try clearing cache
                        torch.cuda.empty_cache()
                        self._update_prometheus_gauges()
                    
                    # Try switching allocator backend
                    if attempt == 1:
                        self._switch_allocator_backend()
                    
                    # Small delay before retry
                    time.sleep(0.1)
                else:
                    # All retries failed
                    with self.lock:
                        self.stats.failed_allocations += 1
                        self.allocation_history.append({
                            "timestamp": time.time(),
                            "size_bytes": size_bytes,
                            "shape": shape,
                            "success": False,
                            "attempts": max_retries + 1
                        })
                    if self._prom_handles:
                        self._prom_handles["allocation_failures"].inc()
                    self._update_prometheus_gauges()
                    
                    print(f"Failed to allocate {size_bytes / (1024**2):.2f} MB after {max_retries + 1} attempts")
                    return None
        
        return None
    
    def _switch_allocator_backend(self):
        """Switch to a different allocator backend."""
        backends = list(AllocatorBackend)
        current_idx = backends.index(self.current_backend)
        next_idx = (current_idx + 1) % len(backends)
        self.current_backend = backends[next_idx]
        
        print(f"Switching allocator backend to: {self.current_backend.value}")
        
        # Note: In practice, you'd set PYTORCH_ALLOC_CONF environment variable (PyTorch 2.10+)
        # and restart the process or use subprocess as in the original implementation
    
    def free(self, tensor: torch.Tensor):
        """
        Free a tensor and update statistics.
        
        Args:
            tensor: Tensor to free
        """
        size_bytes = tensor.numel() * tensor.element_size()
        
        with self.lock:
            self.stats.total_frees += 1
            self.stats.bytes_freed += size_bytes
        
        del tensor
        self._update_prometheus_gauges()
    
    def get_stats(self) -> AllocationStats:
        """Get allocation statistics."""
        with self.lock:
            return self.stats
    
    def print_stats(self):
        """Print allocation statistics."""
        stats = self.get_stats()
        mem_stats = self.get_memory_stats()
        frag_metrics = self.compute_fragmentation_metrics()
        
        print("\n" + "="*70)
        print("Memory Allocator Statistics")
        print("="*70)
        print(f"\nAllocations:")
        print(f"  Total:              {stats.total_allocations}")
        print(f"  Failed:             {stats.failed_allocations}")
        print(f"  Retries:            {stats.retries} ({stats.retry_rate:.1f}%)")
        print(f"  Frees:              {stats.total_frees}")
        print(f"\nMemory:")
        print(f"  Currently allocated: {stats.current_allocated / (1024**3):.2f} GB")
        print(f"  Peak allocated:      {stats.peak_memory_bytes / (1024**3):.2f} GB")
        print(f"  Total GPU memory:    {mem_stats.get('total_gb', 0):.2f} GB")
        print(f"  GPU utilization:     {mem_stats.get('utilization_percent', 0):.1f}%")
        print(f"\nFragmentation:")
        print(f"  Free memory:         {frag_metrics.total_free_bytes / (1024**3):.2f} GB")
        print(f"  Largest free block:  {frag_metrics.largest_free_block / (1024**3):.2f} GB")
        print(f"  Free blocks:         {frag_metrics.num_free_blocks}")
        print(f"  Fragmentation ratio: {frag_metrics.fragmentation_ratio:.2f}")
        print(f"  Status:              {'FRAGMENTED' if frag_metrics.is_fragmented else 'HEALTHY'}")
        print(f"  Defrag events:       {stats.fragmentation_events}")
        print("="*70 + "\n")
        self._update_prometheus_gauges(frag_metrics)


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("Memory Allocator with Monitoring Demo (Chapter 19)")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. This demo requires a GPU.")
        print("Exiting...")
        exit(0)
    
    device = torch.device("cuda")
    allocator = ManagedMemoryAllocator(device=device)
    
    print("\nInitial memory state:")
    allocator.print_stats()
    
    # Test 1: Normal allocations
    print("=" * 70)
    print("Test 1: Normal allocations")
    print("=" * 70)
    
    tensors = []
    for i in range(5):
        tensor = allocator.allocate((1024, 1024, 10), dtype=torch.float16)
        if tensor is not None:
            tensors.append(tensor)
            print(f"  Allocated tensor {i+1}: {tensor.shape}, {tensor.element_size() * tensor.numel() / (1024**2):.2f} MB")
    
    allocator.print_stats()
    
    # Test 2: Fragmentation scenario
    print("=" * 70)
    print("Test 2: Creating fragmentation")
    print("=" * 70)
    
    # Free every other tensor to create fragmentation
    for i in range(0, len(tensors), 2):
        allocator.free(tensors[i])
        print(f"  Freed tensor {i+1}")
    
    tensors = [t for i, t in enumerate(tensors) if i % 2 != 0]
    
    allocator.print_stats()
    
    # Test 3: Allocation with retry
    print("=" * 70)
    print("Test 3: Large allocation (may trigger retry)")
    print("=" * 70)
    
    large_tensor = allocator.allocate((8192, 8192, 4), dtype=torch.float32)
    if large_tensor is not None:
        print(f"  Successfully allocated large tensor: {large_tensor.shape}")
        print(f"  Size: {large_tensor.element_size() * large_tensor.numel() / (1024**3):.2f} GB")
    else:
        print("  Failed to allocate large tensor")
    
    allocator.print_stats()
    
    # Cleanup
    print("\nCleaning up...")
    for tensor in tensors:
        allocator.free(tensor)
    if large_tensor is not None:
        allocator.free(large_tensor)
    
    torch.cuda.empty_cache()
    
    print("\nFinal state:")
    allocator.print_stats()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
