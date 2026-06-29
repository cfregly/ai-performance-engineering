#!/usr/bin/env python3

import threading
import time
from typing import Dict

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

from core.utils.architecture_runtime import get_arch_config

_ARCH_CFG = get_arch_config()
"""
Chapter 5: GPU-based Storage I/O Optimizations
GPUDirect Storage Example

This example demonstrates:
- GPUDirect Storage concepts and benefits
- Data pipeline optimization
- Storage I/O monitoring
- Sequential vs random read patterns
"""

class OptimizedDataset(Dataset):
    """Dataset optimized for sequential reads and GPU processing."""
    
    def __init__(self, size: int = 10000, feature_dim: int = 1024, 
                 use_sequential_access: bool = True):
        self.size = size
        self.feature_dim = feature_dim
        self.use_sequential_access = use_sequential_access
        
        # Create data in memory for demonstration
        # In real scenarios, this would be read from storage
        self.data = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, 10, (size,))
        
        # Simulate sequential vs random access patterns
        if use_sequential_access:
            # Sequential access - data is stored in large chunks
            self.indices = list(range(size))
        else:
            # Random access - data is scattered across many small files
            self.indices = np.random.permutation(size).tolist()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate different access patterns
        if self.use_sequential_access:
            # Fast sequential read
            return self.data[idx], self.labels[idx]
        else:
            # Simulate random access overhead
            time.sleep(0.001)  # Simulate disk seek time
            return self.data[idx], self.labels[idx]

class StorageIOMonitor:
    """Monitor storage I/O performance and bottlenecks."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.gpu_utilization = []
        self.cpu_utilization = []
        self.memory_usage = []
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.start_time = time.perf_counter()
        self.gpu_utilization = []
        self.cpu_utilization = []
        self.memory_usage = []
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            # GPU utilization
            if torch.cuda.is_available():
                gpu_util = torch.cuda.utilization()
                self.gpu_utilization.append(gpu_util)
            
            # CPU utilization
            cpu_util = psutil.cpu_percent(interval=1)
            self.cpu_utilization.append(cpu_util)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            
            time.sleep(1)
    
    def stop_monitoring(self):
        """Stop monitoring and calculate metrics."""
        self.monitoring = False
        self.end_time = time.perf_counter()
        
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get monitoring metrics."""
        if not self.gpu_utilization:
            return {}
        
        return {
            'avg_gpu_utilization': np.mean(self.gpu_utilization),
            'avg_cpu_utilization': np.mean(self.cpu_utilization),
            'avg_memory_usage': np.mean(self.memory_usage),
            'monitoring_duration': self.end_time - self.start_time if self.end_time else 0,
            'max_gpu_utilization': max(self.gpu_utilization),
            'min_gpu_utilization': min(self.gpu_utilization),
        }

class OptimizedDataLoader:
    """Optimized DataLoader with advanced I/O features."""
    
    def __init__(self, dataset: Dataset, batch_size: int = 32, 
                 num_workers: int = 4, pin_memory: bool = True,
                 prefetch_factor: int = 2, persistent_workers: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Create optimized DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=True,
            drop_last=True
        )
    
    def benchmark_throughput(self, num_batches: int = 100) -> Dict[str, float]:
        """Benchmark data loading throughput."""
        monitor = StorageIOMonitor()
        monitor.start_monitoring()
        
        start_time = time.perf_counter()
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(self.loader):
            # Simulate GPU processing
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            batch_count += 1
            if batch_count >= num_batches:
                break
        
        end_time = time.perf_counter()
        monitor.stop_monitoring()
        
        total_time = end_time - start_time
        throughput = (batch_count * self.batch_size) / total_time
        
        metrics = monitor.get_metrics()
        metrics.update({
            'batches_per_second': batch_count / total_time,
            'samples_per_second': throughput,
            'total_time': total_time,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor
        })
        
        return metrics

def demonstrate_sequential_vs_random_access():
    """Demonstrate the performance difference between sequential and random access."""
    print("=== Sequential vs Random Access Performance ===\n")
    
    # Create datasets with different access patterns
    sequential_dataset = OptimizedDataset(size=1000, use_sequential_access=True)
    random_dataset = OptimizedDataset(size=1000, use_sequential_access=False)
    
    # Test sequential access
    print("Testing sequential access pattern...")
    sequential_loader = OptimizedDataLoader(
        sequential_dataset, 
        batch_size=32, 
        num_workers=4,
        pin_memory=True
    )
    
    start_time = time.perf_counter()
    sequential_metrics = sequential_loader.benchmark_throughput(num_batches=50)
    sequential_time = time.perf_counter() - start_time
    
    # Test random access
    print("Testing random access pattern...")
    random_loader = OptimizedDataLoader(
        random_dataset, 
        batch_size=32, 
        num_workers=4,
        pin_memory=True
    )
    
    start_time = time.perf_counter()
    random_metrics = random_loader.benchmark_throughput(num_batches=50)
    random_time = time.perf_counter() - start_time
    
    # Compare results
    print(f"\nSequential Access Results:")
    print(f"  Samples/second: {sequential_metrics['samples_per_second']:.2f}")
    print(f"  GPU Utilization: {sequential_metrics['avg_gpu_utilization']:.1f}%")
    print(f"  CPU Utilization: {sequential_metrics['avg_cpu_utilization']:.1f}%")
    
    print(f"\nRandom Access Results:")
    print(f"  Samples/second: {random_metrics['samples_per_second']:.2f}")
    print(f"  GPU Utilization: {random_metrics['avg_gpu_utilization']:.1f}%")
    print(f"  CPU Utilization: {random_metrics['avg_cpu_utilization']:.1f}%")
    
    speedup = random_time / sequential_time
    print(f"\nSequential access is {speedup:.2f}x faster than random access")

def demonstrate_worker_scaling():
    """Demonstrate the impact of different numbers of workers."""
    print("\n=== Worker Scaling Analysis ===\n")
    
    dataset = OptimizedDataset(size=2000, use_sequential_access=True)
    worker_counts = [1, 2, 4, 8]
    
    results = {}
    
    for num_workers in worker_counts:
        print(f"Testing with {num_workers} workers...")
        
        loader = OptimizedDataLoader(
            dataset,
            batch_size=32,
            num_workers=num_workers,
            pin_memory=True
        )
        
        metrics = loader.benchmark_throughput(num_batches=30)
        results[num_workers] = metrics
        
        print(f"  Samples/second: {metrics['samples_per_second']:.2f}")
        print(f"  GPU Utilization: {metrics['avg_gpu_utilization']:.1f}%")
    
    # Find optimal worker count
    best_workers = max(results.keys(), key=lambda w: results[w]['samples_per_second'])
    print(f"\nOptimal number of workers: {best_workers}")
    print(f"Best throughput: {results[best_workers]['samples_per_second']:.2f} samples/second")

def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques."""
    print("\n=== Memory Optimization Techniques ===\n")
    
    dataset = OptimizedDataset(size=1000)
    
    # Test with different memory configurations
    configs = [
        {'pin_memory': False, 'prefetch_factor': 1, 'name': 'Basic'},
        {'pin_memory': True, 'prefetch_factor': 2, 'name': 'Pinned Memory'},
        {'pin_memory': True, 'prefetch_factor': 4, 'name': 'High Prefetch'},
    ]
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        loader = OptimizedDataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor']
        )
        
        metrics = loader.benchmark_throughput(num_batches=20)
        
        print(f"  Samples/second: {metrics['samples_per_second']:.2f}")
        print(f"  Memory Usage: {metrics['avg_memory_usage']:.1f}%")

def demonstrate_gpudirect_storage_concepts():
    """Demonstrate GPUDirect Storage concepts and benefits."""
    print("\n=== GPUDirect Storage Concepts ===\n")
    
    print("GPUDirect Storage Benefits:")
    print("1. Direct GPU-to-Storage transfers (bypassing CPU)")
    print("2. Reduced CPU overhead for data movement")
    print("3. Higher throughput for storage-bound workloads")
    print("4. Lower latency for data transfers")
    
    print("\nTraditional Path: Storage → CPU Memory → GPU Memory")
    print("GDS Path: Storage → GPU Memory (direct)")
    
    # Simulate the performance difference
    print("\nSimulated Performance Comparison:")
    
    # Traditional path (CPU-mediated)
    cpu_throughput = 8.0  # GB/s
    cpu_latency = 1.25    # ms
    
    # GDS path (direct)
    gds_throughput = 9.6  # GB/s (20% improvement)
    gds_latency = 1.00    # ms (20% improvement)
    
    print(f"Traditional (CPU-mediated): {cpu_throughput} GB/s, {cpu_latency} ms")
    print(f"GPUDirect Storage: {gds_throughput} GB/s, {gds_latency} ms")
    print(f"Throughput improvement: {((gds_throughput/cpu_throughput)-1)*100:.1f}%")
    print(f"Latency improvement: {((cpu_latency/gds_latency)-1)*100:.1f}%")

def demonstrate_storage_monitoring():
    """Demonstrate storage I/O monitoring techniques."""
    print("\n=== Storage I/O Monitoring ===\n")
    
    # Simulate storage monitoring
    print("Key monitoring metrics:")
    print("1. Disk I/O throughput (iostat)")
    print("2. GPU utilization during data loading")
    print("3. CPU utilization for I/O operations")
    print("4. Memory usage patterns")
    print("5. Network storage bandwidth (if applicable)")
    
    print("\nCommon bottlenecks:")
    print("- Insufficient disk bandwidth")
    print("- Too few worker processes")
    print("- Small read sizes causing overhead")
    print("- Random access patterns")
    print("- Network storage latency")

def main():
    """Main demonstration function."""
    print("=== Chapter 5: GPU-based Storage I/O Optimizations ===\n")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - running CPU-only demonstrations")
    
    # Run demonstrations
    demonstrate_sequential_vs_random_access()
    demonstrate_worker_scaling()
    demonstrate_memory_optimization()
    demonstrate_gpudirect_storage_concepts()
    demonstrate_storage_monitoring()
    
    print("\n=== Key Takeaways ===")
    print("1. Sequential access is much faster than random access")
    print("2. Optimal worker count depends on your system")
    print("3. Pinned memory improves GPU transfer performance")
    print("4. GPUDirect Storage can provide 20%+ throughput improvement")
    print("5. Monitor I/O bottlenecks to optimize data pipeline")
    print("6. Scale data loading with compute resources")

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Note: These configuration options may not be available in all PyTorch versions
    # Commenting out for compatibility
    #     torch._inductor.config.triton.use_hopper_optimizations = True
    #     torch._inductor.config.triton.hbm3_optimizations = True
    # if compute_capability == "10.0":  # Blackwell B200/B300
    #     torch._inductor.config.triton.use_blackwell_optimizations = True
    #     torch._inductor.config.triton.hbm3e_optimizations = True
    #     torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.10 features (if available)
    try:
        torch._inductor.config.triton.unique_kernel_names = True
    except AttributeError:
        pass
    
    try:
        torch._inductor.config.triton.autotune_mode = "max-autotune"
    except AttributeError:
        pass
    
    try:
        torch._dynamo.config.automatic_dynamic_shapes = True
    except AttributeError:
        pass
