#!/usr/bin/env python3

"""
Chapter 10: Real cuFile (GPUDirect Storage) Example
====================================================

This example demonstrates REAL GPUDirect Storage using cuda.bindings.cufile
from CUDA 13.0+. It shows:
- Direct storage-to-GPU transfers bypassing CPU
- Performance comparison: Traditional (CPU-mediated) vs GDS (direct)
- Real cuFile API usage on B200/B300 systems with GDS support

Requirements:
- CUDA 13.0+ with cuda-bindings package
- GPUDirect Storage (GDS) enabled system
- NVMe storage on supported hardware
- Python 3.8+

Fallback: If GDS is not available, demonstrates traditional path for comparison.

Hardware: NVIDIA B200/B300 (SM 10.0) with NVMe SSD and GDS support
"""

import os
import time
import tempfile
import numpy as np
from typing import Tuple, Optional
import warnings

# Try to import cuFile (cuda.bindings.cufile from cuda-bindings 13.0)
try:
    from cuda import cuda as cuda_driver
    from cuda.bindings import cufile
    CUFILE_AVAILABLE = True
    print("cuFile bindings available (cuda-bindings 13.0+)")
except ImportError as e:
    CUFILE_AVAILABLE = False
    cuda_driver = None
    cufile = None
    warnings.warn(f"cuFile not available: {e}. Running in simulation mode.")

# Try to import PyTorch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    warnings.warn("PyTorch not available. Limited functionality.")


class CuFileError(Exception):
    """Custom exception for cuFile operations."""
    pass


class GPUDirectStorageDemo:
    """
    Demonstrates GPUDirect Storage (GDS) using real cuFile APIs.
    
    GDS enables direct data transfer between NVMe storage and GPU memory,
    bypassing the CPU and system memory, reducing latency and CPU overhead.
    """
    
    def __init__(self, use_gds: bool = True):
        """
        Initialize GDS demo.
        
        Args:
            use_gds: If True, attempt to use GPUDirect Storage. 
                     If False or GDS unavailable, use traditional path.
        """
        self.use_gds = use_gds and CUFILE_AVAILABLE
        self.driver_initialized = False
        self.file_handle = None
        
        if self.use_gds:
            self._initialize_cufile()
    
    def _initialize_cufile(self):
        """Initialize cuFile driver."""
        if not CUFILE_AVAILABLE:
            raise CuFileError("cuFile bindings not available")
        
        try:
            # Initialize cuFile driver
            status = cufile.cuFileDriverOpen()
            if status != cufile.CU_FILE_SUCCESS:
                raise CuFileError(f"cuFileDriverOpen failed: {status}")
            
            self.driver_initialized = True
            print("cuFile driver initialized")
            
        except Exception as e:
            raise CuFileError(f"Failed to initialize cuFile driver: {e}")
    
    def _check_gds_support(self) -> Tuple[bool, str]:
        """
        Check if GPUDirect Storage is supported on this system.
        
        Returns:
            Tuple of (is_supported, message)
        """
        if not CUFILE_AVAILABLE:
            return False, "cuFile bindings not available"
        
        if not TORCH_AVAILABLE:
            return False, "PyTorch CUDA not available"
        
        # Check CUDA compute capability (GDS requires 6.0+)
        props = torch.cuda.get_device_properties(0)
        compute_cap = f"{props.major}.{props.minor}"
        
        if props.major < 6:
            return False, f"GDS requires compute capability 6.0+, got {compute_cap}"
        
        # Check if GDS is enabled in the system
        gds_config = "/etc/cufile.json"
        if not os.path.exists(gds_config):
            return False, "GDS not configured (/etc/cufile.json missing)"
        
        return True, f"GDS supported on {torch.cuda.get_device_name(0)} (SM {compute_cap})"
    
    def create_test_file(self, size_mb: int = 256) -> str:
        """
        Create a test file with random data.
        
        Args:
            size_mb: Size of test file in MB
            
        Returns:
            Path to created file
        """
        size_bytes = size_mb * 1024 * 1024
        
        # Create temporary file
        fd, filepath = tempfile.mkstemp(suffix='.bin', prefix='cufile_test_')
        
        # Generate random data
        data = np.random.randn(size_bytes // 4).astype(np.float32)
        
        # Write to file
        with os.fdopen(fd, 'wb') as f:
            f.write(data.tobytes())
        
        print(f"Created test file: {filepath} ({size_mb} MB)")
        return filepath
    
    def read_traditional(self, filepath: str, device: int = 0) -> Tuple[float, float]:
        """
        Traditional read path: Storage → CPU Memory → GPU Memory
        
        Args:
            filepath: Path to file to read
            device: CUDA device ID
            
        Returns:
            Tuple of (throughput_gbps, latency_ms)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch CUDA required for traditional read")
        
        file_size = os.path.getsize(filepath)
        
        # Read file to CPU memory
        start_time = time.perf_counter()
        with open(filepath, 'rb') as f:
            cpu_data = f.read()
        cpu_read_time = time.perf_counter() - start_time
        
        # Convert to numpy array
        np_data = np.frombuffer(cpu_data, dtype=np.float32)
        
        # Transfer to GPU
        start_time = time.perf_counter()
        gpu_tensor = torch.from_numpy(np_data).cuda(device, non_blocking=False)
        torch.cuda.synchronize()
        gpu_transfer_time = time.perf_counter() - start_time
        
        total_time = cpu_read_time + gpu_transfer_time
        throughput_gbps = (file_size / total_time) / 1e9
        latency_ms = total_time * 1000
        
        return throughput_gbps, latency_ms
    
    def read_gds(self, filepath: str, device: int = 0) -> Tuple[float, float]:
        """
        GDS read path: Storage → GPU Memory (direct)
        
        Uses cuFile API to perform direct GPU reads from storage.
        
        Args:
            filepath: Path to file to read
            device: CUDA device ID
            
        Returns:
            Tuple of (throughput_gbps, latency_ms)
        """
        if not self.use_gds:
            raise CuFileError("GDS not available or not enabled")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch CUDA required for GDS read")
        
        file_size = os.path.getsize(filepath)
        num_elements = file_size // 4  # float32
        
        # Allocate GPU memory
        gpu_tensor = torch.empty(num_elements, dtype=torch.float32, device='cuda')
        
        try:
            # Open file with cuFile
            fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
            
            # Register file descriptor with cuFile
            cf_descr = cufile.CUfileDescr_t()
            cf_descr.type = cufile.CU_FILE_HANDLE_TYPE_OPAQUE_FD
            cf_descr.handle.fd = fd
            
            cf_handle = cufile.CUfileHandle_t()
            status = cufile.cuFileHandleRegister(cf_handle, cf_descr)
            
            if status != cufile.CU_FILE_SUCCESS:
                raise CuFileError(f"cuFileHandleRegister failed: {status}")
            
            # Get GPU pointer
            gpu_ptr = gpu_tensor.data_ptr()
            
            # Perform direct GPU read
            start_time = time.perf_counter()
            
            bytes_read = cufile.cuFileRead(
                cf_handle,      # File handle
                gpu_ptr,        # GPU buffer pointer
                file_size,      # Size to read
                0,              # File offset
                0               # GPU offset
            )
            
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            
            if bytes_read < 0:
                raise CuFileError(f"cuFileRead failed with status: {bytes_read}")
            
            # Cleanup
            cufile.cuFileHandleDeregister(cf_handle)
            os.close(fd)
            
            throughput_gbps = (bytes_read / elapsed_time) / 1e9
            latency_ms = elapsed_time * 1000
            
            return throughput_gbps, latency_ms
            
        except AttributeError:
            # Fallback: cuFile API structure might differ in actual implementation
            # This is a best-effort implementation based on CUDA 13.0 documentation
            raise CuFileError(
                "cuFile API mismatch. This example is compatible with cuda-bindings 13.0+. "
                "Please ensure you have the correct version installed."
            )
    
    def write_gds(self, filepath: str, data: 'torch.Tensor') -> Tuple[float, float]:
        """
        GDS write path: GPU Memory → Storage (direct)
        
        Args:
            filepath: Path to file to write
            data: GPU tensor to write
            
        Returns:
            Tuple of (throughput_gbps, latency_ms)
        """
        if not self.use_gds:
            raise CuFileError("GDS not available or not enabled")
        
        if not data.is_cuda:
            raise ValueError("Data must be on GPU")
        
        data_size = data.numel() * data.element_size()
        
        try:
            # Open file with cuFile
            fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
            
            # Register file descriptor
            cf_descr = cufile.CUfileDescr_t()
            cf_descr.type = cufile.CU_FILE_HANDLE_TYPE_OPAQUE_FD
            cf_descr.handle.fd = fd
            
            cf_handle = cufile.CUfileHandle_t()
            status = cufile.cuFileHandleRegister(cf_handle, cf_descr)
            
            if status != cufile.CU_FILE_SUCCESS:
                raise CuFileError(f"cuFileHandleRegister failed: {status}")
            
            # Get GPU pointer
            gpu_ptr = data.data_ptr()
            
            # Perform direct GPU write
            start_time = time.perf_counter()
            
            bytes_written = cufile.cuFileWrite(
                cf_handle,      # File handle
                gpu_ptr,        # GPU buffer pointer
                data_size,      # Size to write
                0,              # File offset
                0               # GPU offset
            )
            
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            
            if bytes_written < 0:
                raise CuFileError(f"cuFileWrite failed with status: {bytes_written}")
            
            # Cleanup
            cufile.cuFileHandleDeregister(cf_handle)
            os.close(fd)
            
            throughput_gbps = (bytes_written / elapsed_time) / 1e9
            latency_ms = elapsed_time * 1000
            
            return throughput_gbps, latency_ms
            
        except AttributeError:
            raise CuFileError(
                "cuFile API mismatch. This example is compatible with cuda-bindings 13.0+."
            )
    
    def benchmark_comparison(self, size_mb: int = 256) -> dict:
        """
        Compare traditional vs GDS read performance.
        
        Args:
            size_mb: Size of test file in MB
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking Storage I/O Performance ({size_mb} MB)")
        print(f"{'='*60}\n")
        
        # Create test file
        filepath = self.create_test_file(size_mb)
        
        results = {}
        
        try:
            # Test traditional path
            print("\n[1/2] Testing Traditional Path (Storage → CPU → GPU)...")
            trad_throughput, trad_latency = self.read_traditional(filepath)
            
            results['traditional'] = {
                'throughput_gbps': trad_throughput,
                'latency_ms': trad_latency,
                'path': 'Storage → CPU Memory → GPU Memory'
            }
            
            print(f"  Throughput: {trad_throughput:.2f} GB/s")
            print(f"  Latency: {trad_latency:.2f} ms")
            
            # Test GDS path (if available)
            if self.use_gds:
                print("\n[2/2] Testing GDS Path (Storage → GPU direct)...")
                try:
                    gds_throughput, gds_latency = self.read_gds(filepath)
                    
                    results['gds'] = {
                        'throughput_gbps': gds_throughput,
                        'latency_ms': gds_latency,
                        'path': 'Storage → GPU Memory (direct)'
                    }
                    
                    print(f"  Throughput: {gds_throughput:.2f} GB/s")
                    print(f"  Latency: {gds_latency:.2f} ms")
                    
                    # Calculate improvements
                    throughput_improvement = ((gds_throughput / trad_throughput) - 1) * 100
                    latency_improvement = ((trad_latency / gds_latency) - 1) * 100
                    
                    results['improvement'] = {
                        'throughput_pct': throughput_improvement,
                        'latency_pct': latency_improvement
                    }
                    
                except CuFileError as e:
                    print(f"  ✗ GDS test failed: {e}")
                    results['gds'] = None
            else:
                print("\n[2/2] GDS Path: Not available (using fallback)")
                results['gds'] = None
            
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.unlink(filepath)
        
        return results
    
    def print_results(self, results: dict):
        """Pretty print benchmark results."""
        print(f"\n{'='*60}")
        print("Performance Comparison Results")
        print(f"{'='*60}\n")
        
        trad = results.get('traditional')
        gds = results.get('gds')
        improvement = results.get('improvement')
        
        if trad:
            print("Traditional Path (Storage → CPU → GPU):")
            print(f"  Throughput: {trad['throughput_gbps']:.2f} GB/s")
            print(f"  Latency: {trad['latency_ms']:.2f} ms")
        
        if gds:
            print("\nGPUDirect Storage (Storage → GPU):")
            print(f"  Throughput: {gds['throughput_gbps']:.2f} GB/s")
            print(f"  Latency: {gds['latency_ms']:.2f} ms")
            
            if improvement:
                print("\nImprovement with GDS:")
                print(f"  Throughput: +{improvement['throughput_pct']:.1f}%")
                print(f"  Latency: -{improvement['latency_pct']:.1f}%")
        else:
            print("\nGPUDirect Storage: Not available on this system")
            print("  - Requires GDS-enabled hardware and software")
            print("  - Typically 20-30% throughput improvement over traditional path")
        
        print(f"\n{'='*60}\n")
    
    def cleanup(self):
        """Cleanup cuFile resources."""
        if self.driver_initialized:
                cufile.cuFileDriverClose()
                print("cuFile driver closed")


def simulate_gds_performance():
    """
    Simulate GDS performance characteristics when real GDS is not available.
    
    This matches the teaching simulator in the CUDA code.
    """
    print(f"\n{'='*60}")
    print("GPUDirect Storage - Simulated Performance")
    print(f"{'='*60}\n")
    
    print("NOTE: Running in simulation mode (GDS not available)")
    print("This demonstrates expected performance characteristics.\n")
    
    # Typical performance numbers based on real GDS systems
    traditional_throughput = 8.5   # GB/s (PCIe Gen4 x16)
    traditional_latency = 1.2      # ms
    
    gds_throughput = 10.8          # GB/s (20-30% improvement typical)
    gds_latency = 0.95             # ms (20-25% improvement typical)
    
    print("Traditional Path (Storage → CPU → GPU):")
    print(f"  Throughput: {traditional_throughput:.2f} GB/s")
    print(f"  Latency: {traditional_latency:.2f} ms")
    print("  CPU overhead: High (data copied twice)")
    
    print("\nGPUDirect Storage (Storage → GPU):")
    print(f"  Throughput: {gds_throughput:.2f} GB/s")
    print(f"  Latency: {gds_latency:.2f} ms")
    print("  CPU overhead: Low (bypassed)")
    
    throughput_improvement = ((gds_throughput / traditional_throughput) - 1) * 100
    latency_improvement = ((traditional_latency / gds_latency) - 1) * 100
    
    print("\nImprovement with GDS:")
    print(f"  Throughput: +{throughput_improvement:.1f}%")
    print(f"  Latency: -{latency_improvement:.1f}%")
    print(f"  CPU cycles saved: ~30-40%")
    
    print("\nKey Benefits:")
    print("  Eliminates CPU memory copy")
    print("  Reduces system memory bandwidth pressure")
    print("  Frees CPU for other work")
    print("  Lower latency for storage-bound workloads")
    
    print(f"\n{'='*60}\n")


def main():
    """Main demonstration function."""
    print("="*60)
    print("Chapter 10: GPUDirect Storage (cuFile) Example")
    print("="*60)
    
    # Check system capabilities
    print("\nSystem Check:")
    
    if not CUFILE_AVAILABLE:
        print("✗ cuFile bindings not available")
        print("  Install the isolated CUDA 13 runtime from docs/environment.md")
        print("  Note: Requires cuda-bindings package, not just cuda-python")
    
    if not TORCH_AVAILABLE:
        print("✗ PyTorch with CUDA not available")
        print("  Use torch==2.14.1+cu130 from requirements_torch2141_linux_x86_64.txt")
    
    # Run demonstration
    if CUFILE_AVAILABLE and TORCH_AVAILABLE:
        print("\nAll dependencies available")
        
        demo = GPUDirectStorageDemo(use_gds=True)
        
        # Check GDS support
        is_supported, message = demo._check_gds_support()
        print(f"\nGDS Support: {message}")
        
        if is_supported:
            # Run real benchmark
            try:
                results = demo.benchmark_comparison(size_mb=256)
                demo.print_results(results)
            except CuFileError as e:
                print(f"\n✗ GDS benchmark failed: {e}")
                print("  Falling back to simulation mode...")
                simulate_gds_performance()
            finally:
                demo.cleanup()
        else:
            print("\nGDS not available on this system.")
            print("Running simulation to demonstrate expected performance...\n")
            simulate_gds_performance()
    else:
        print("\nDependencies missing. Running simulation...\n")
        simulate_gds_performance()
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. GDS eliminates CPU memory copy for storage I/O")
    print("2. Typically provides 20-30% throughput improvement")
    print("3. Reduces CPU overhead and system memory bandwidth")
    print("4. Requires GDS-enabled hardware (B200/B300, NVMe)")
    print("5. Best for storage-bound workloads with large datasets")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
