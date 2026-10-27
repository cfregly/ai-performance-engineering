#!/usr/bin/env python3

"""
Chapter 10: Real cuFile (GPUDirect Storage) Example
====================================================

This example demonstrates REAL GPUDirect Storage using cuda.bindings.cufile
from CUDA Python 13.1.1+. It shows:
- Direct storage-to-GPU transfers bypassing CPU
- Performance comparison: Traditional (CPU-mediated) vs GDS (direct)
- Real cuFile API usage on B200/B300 systems with GDS support

Requirements:
- CUDA Python 13.1.1+ with the cuda-bindings package
- GPUDirect Storage (GDS) enabled system
- NVMe storage on supported hardware
- Python 3.10+

The example stops with a concrete diagnostic when any GDS requirement is missing.
It does not report simulated throughput or latency.

Hardware: NVIDIA B200/B300 (SM 10.0) with NVMe SSD and GDS support
"""

import os
import sys
import tempfile
import time

import numpy as np

# Import cuFile without hiding a missing runtime behind simulated results.
CUFILE_IMPORT_ERROR = None
try:
    from cuda.bindings import cufile

    CUFILE_AVAILABLE = True
except (ImportError, OSError) as exc:
    CUFILE_AVAILABLE = False
    cufile = None
    CUFILE_IMPORT_ERROR = exc

# Import PyTorch without treating a CPU-only environment as a runnable GDS path.
TORCH_IMPORT_ERROR = None
try:
    import torch

    try:
        TORCH_AVAILABLE = torch.cuda.is_available()
    except (OSError, RuntimeError) as exc:
        TORCH_AVAILABLE = False
        TORCH_IMPORT_ERROR = exc
except ImportError as exc:
    TORCH_AVAILABLE = False
    torch = None
    TORCH_IMPORT_ERROR = exc


class CuFileError(Exception):
    """Custom exception for cuFile operations."""

    pass


class CuFileUnavailableError(CuFileError):
    """Raised when the installed cuFile runtime cannot provide GDS."""


def _require_gds_support(device: int = 0) -> str:
    """Return the selected GPU name or raise a concrete GDS diagnostic."""
    if not CUFILE_AVAILABLE:
        detail = f" Import failed: {CUFILE_IMPORT_ERROR}" if CUFILE_IMPORT_ERROR else ""
        raise CuFileUnavailableError(
            "SKIPPED: cuda.bindings.cufile is unavailable. Install the pinned CUDA 13 "
            "packages from requirements_cufile.txt in the GPU environment."
            f"{detail}"
        )

    if torch is None:
        detail = f" Import failed: {TORCH_IMPORT_ERROR}" if TORCH_IMPORT_ERROR else ""
        raise CuFileUnavailableError(
            "SKIPPED: PyTorch is unavailable. Install torch==2.14.1+cu130 from the "
            "pinned CUDA 13 requirements."
            f"{detail}"
        )

    if not TORCH_AVAILABLE:
        detail = f" CUDA probe failed: {TORCH_IMPORT_ERROR}" if TORCH_IMPORT_ERROR else ""
        raise CuFileUnavailableError(
            "SKIPPED: PyTorch cannot see a CUDA GPU. Run this example on a host with a "
            "supported NVIDIA GPU and driver."
            f"{detail}"
        )

    try:
        props = torch.cuda.get_device_properties(device)
        device_name = torch.cuda.get_device_name(device)
    except (OSError, RuntimeError) as exc:
        raise CuFileUnavailableError(
            f"SKIPPED: CUDA device {device} cannot be queried for GDS support: {exc}"
        ) from exc

    compute_capability = f"{props.major}.{props.minor}"
    if props.major < 6:
        raise CuFileUnavailableError(
            "SKIPPED: GDS requires compute capability 6.0 or newer. "
            f"Device {device} reports SM {compute_capability}."
        )

    config_path = "/etc/cufile.json"
    if not os.path.isfile(config_path):
        raise CuFileUnavailableError(
            f"SKIPPED: GDS is not configured because {config_path} is missing. "
            "Install and configure NVIDIA GPUDirect Storage on this host."
        )

    return f"{device_name} (SM {compute_capability})"


def _require_complete_transfer(operation: str, result: object, requested_bytes: int) -> None:
    """Require a cuFile operation to report an exact byte count."""
    if isinstance(result, bool) or not isinstance(result, int):
        raise CuFileError(
            f"cuFile {operation} returned a non-integer byte count {result!r}. "
            f"Expected {requested_bytes}"
        )
    if result < 0:
        raise CuFileError(
            f"cuFile {operation} returned a negative byte count {result}. "
            f"Expected {requested_bytes}"
        )
    if result != requested_bytes:
        raise CuFileError(
            f"cuFile {operation} transferred {result} of {requested_bytes} requested bytes"
        )


class GPUDirectStorageDemo:
    """
    Demonstrates GPUDirect Storage (GDS) using real cuFile APIs.

    GDS enables direct data transfer between NVMe storage and GPU memory,
    bypassing the CPU and system memory, reducing latency and CPU overhead.
    """

    def __init__(self, device: int = 0):
        """Require GDS support and initialize the cuFile driver."""
        self.device = device
        self.driver_initialized = False
        self.file_handle = None
        self.device_description = _require_gds_support(device)
        self._initialize_cufile()

    def _initialize_cufile(self):
        """Initialize cuFile driver."""
        if not CUFILE_AVAILABLE:
            raise CuFileUnavailableError("cuFile bindings not available")

        try:
            # cuda-bindings 13 reports failures by raising. Successful calls
            # return None, so there is no integer status to compare.
            cufile.driver_open()
        except (cufile.cuFileError, OSError, RuntimeError) as exc:
            raise CuFileUnavailableError(
                "SKIPPED: the cuFile driver could not initialize. Confirm NVIDIA GDS is "
                f"installed and {os.path.realpath('/etc/cufile.json')} matches this host: {exc}"
            ) from exc

        self.driver_initialized = True
        print("cuFile driver initialized")

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
        fd, filepath = tempfile.mkstemp(suffix=".bin", prefix="cufile_test_")

        # Generate random data
        data = np.random.randn(size_bytes // 4).astype(np.float32)

        # Write to file
        with os.fdopen(fd, "wb") as f:
            f.write(data.tobytes())

        print(f"Created test file: {filepath} ({size_mb} MB)")
        return filepath

    def read_traditional(self, filepath: str, device: int = 0) -> tuple[float, float]:
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
        with open(filepath, "rb") as f:
            cpu_data = f.read()
        cpu_read_time = time.perf_counter() - start_time

        # Convert to numpy array
        np_data = np.frombuffer(cpu_data, dtype=np.float32)

        # Transfer to GPU
        start_time = time.perf_counter()
        _gpu_tensor = torch.from_numpy(np_data).cuda(device, non_blocking=False)
        torch.cuda.synchronize(device)
        gpu_transfer_time = time.perf_counter() - start_time

        total_time = cpu_read_time + gpu_transfer_time
        throughput_gbps = (file_size / total_time) / 1e9
        latency_ms = total_time * 1000

        return throughput_gbps, latency_ms

    def read_gds(self, filepath: str, device: int = 0) -> tuple[float, float]:
        """
        GDS read path: Storage → GPU Memory (direct)

        Uses cuFile API to perform direct GPU reads from storage.

        Args:
            filepath: Path to file to read
            device: CUDA device ID

        Returns:
            Tuple of (throughput_gbps, latency_ms)
        """
        if not TORCH_AVAILABLE:
            raise CuFileUnavailableError("SKIPPED: PyTorch CUDA is required for a GDS read")

        file_size = os.path.getsize(filepath)
        cuda_device = f"cuda:{device}"

        with torch.cuda.device(device):
            # Use one byte per element so the allocation exactly matches the file size.
            gpu_tensor = torch.empty(file_size, dtype=torch.uint8, device=cuda_device)
            torch.cuda.synchronize(device)

            # CUDA Python 13.1.1 returns the underlying ssize_t transfer count.
            try:
                fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)
            except OSError as exc:
                raise CuFileUnavailableError(
                    "SKIPPED: the input path does not support the direct I/O required by GDS. "
                    f"Path: {filepath}. Error: {exc}"
                ) from exc
            cf_handle = None
            try:
                cf_descr = cufile.Descr()
                cf_descr.type = cufile.FileHandleType.OPAQUE_FD
                cf_descr.handle.fd = fd
                try:
                    cf_handle = cufile.handle_register(int(cf_descr))
                except (cufile.cuFileError, OSError, RuntimeError) as exc:
                    raise CuFileUnavailableError(
                        "SKIPPED: cuFile could not register the input path. Use a "
                        f"GDS-supported local or NVMe filesystem. Path: {filepath}. Error: {exc}"
                    ) from exc

                gpu_ptr = gpu_tensor.data_ptr()
                start_time = time.perf_counter()
                try:
                    transferred = cufile.read(cf_handle, gpu_ptr, file_size, 0, 0)
                except (cufile.cuFileError, OSError, RuntimeError) as exc:
                    raise CuFileError(f"cuFile read failed: {exc}") from exc
                _require_complete_transfer("read", transferred, file_size)

                torch.cuda.synchronize(device)
                elapsed_time = time.perf_counter() - start_time
            finally:
                primary_exception_active = sys.exc_info()[0] is not None
                cleanup_error = None
                try:
                    if cf_handle is not None:
                        try:
                            cufile.handle_deregister(cf_handle)
                        except (cufile.cuFileError, OSError, RuntimeError) as exc:
                            cleanup_error = CuFileError(
                                f"cuFile handle deregistration failed: {exc}"
                            )
                finally:
                    try:
                        os.close(fd)
                    except OSError as exc:
                        if cleanup_error is None:
                            cleanup_error = CuFileError(f"Failed to close input file: {exc}")
                if not primary_exception_active and cleanup_error is not None:
                    raise cleanup_error

        throughput_gbps = (file_size / elapsed_time) / 1e9
        latency_ms = elapsed_time * 1000
        return throughput_gbps, latency_ms

    def write_gds(self, filepath: str, data: "torch.Tensor") -> tuple[float, float]:
        """
        GDS write path: GPU Memory → Storage (direct)

        Args:
            filepath: Path to file to write
            data: GPU tensor to write

        Returns:
            Tuple of (throughput_gbps, latency_ms)
        """
        if not data.is_cuda:
            raise ValueError("Data must be on GPU")

        with torch.cuda.device(data.device):
            contiguous_data = data.contiguous()
            data_size = contiguous_data.numel() * contiguous_data.element_size()
            torch.cuda.synchronize(data.device)

            open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT
            try:
                fd = os.open(filepath, open_flags, 0o644)
            except OSError as exc:
                raise CuFileUnavailableError(
                    "SKIPPED: the output path does not support the direct I/O required by GDS. "
                    f"Path: {filepath}. Error: {exc}"
                ) from exc
            cf_handle = None
            try:
                cf_descr = cufile.Descr()
                cf_descr.type = cufile.FileHandleType.OPAQUE_FD
                cf_descr.handle.fd = fd
                try:
                    cf_handle = cufile.handle_register(int(cf_descr))
                except (cufile.cuFileError, OSError, RuntimeError) as exc:
                    raise CuFileUnavailableError(
                        "SKIPPED: cuFile could not register the output path. Use a "
                        f"GDS-supported local or NVMe filesystem. Path: {filepath}. Error: {exc}"
                    ) from exc

                gpu_ptr = contiguous_data.data_ptr()
                start_time = time.perf_counter()
                try:
                    transferred = cufile.write(cf_handle, gpu_ptr, data_size, 0, 0)
                except (cufile.cuFileError, OSError, RuntimeError) as exc:
                    raise CuFileError(f"cuFile write failed: {exc}") from exc
                _require_complete_transfer("write", transferred, data_size)

                torch.cuda.synchronize(data.device)
                elapsed_time = time.perf_counter() - start_time
            finally:
                primary_exception_active = sys.exc_info()[0] is not None
                cleanup_error = None
                try:
                    if cf_handle is not None:
                        try:
                            cufile.handle_deregister(cf_handle)
                        except (cufile.cuFileError, OSError, RuntimeError) as exc:
                            cleanup_error = CuFileError(
                                f"cuFile handle deregistration failed: {exc}"
                            )
                finally:
                    try:
                        os.close(fd)
                    except OSError as exc:
                        if cleanup_error is None:
                            cleanup_error = CuFileError(f"Failed to close output file: {exc}")
                if not primary_exception_active and cleanup_error is not None:
                    raise cleanup_error

        throughput_gbps = (data_size / elapsed_time) / 1e9
        latency_ms = elapsed_time * 1000
        return throughput_gbps, latency_ms

    def benchmark_comparison(self, size_mb: int = 256) -> dict:
        """
        Compare traditional vs GDS read performance.

        Args:
            size_mb: Size of test file in MB

        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'=' * 60}")
        print(f"Benchmarking Storage I/O Performance ({size_mb} MB)")
        print(f"{'=' * 60}\n")

        # Create test file
        filepath = self.create_test_file(size_mb)

        results = {}

        try:
            # Exercise the required GDS path first. If the selected storage
            # path is unsupported, stop before reporting comparison metrics.
            print("\n[1/2] Testing GDS Path (Storage → GPU direct)...")
            gds_throughput, gds_latency = self.read_gds(filepath, device=self.device)

            print("\n[2/2] Testing Traditional Path (Storage → CPU → GPU)...")
            trad_throughput, trad_latency = self.read_traditional(filepath, device=self.device)

            results["traditional"] = {
                "throughput_gbps": trad_throughput,
                "latency_ms": trad_latency,
                "path": "Storage → CPU Memory → GPU Memory",
            }
            results["gds"] = {
                "throughput_gbps": gds_throughput,
                "latency_ms": gds_latency,
                "path": "Storage → GPU Memory (direct)",
            }
            results["improvement"] = {
                "throughput_pct": ((gds_throughput / trad_throughput) - 1) * 100,
                "latency_pct": ((trad_latency / gds_latency) - 1) * 100,
            }

        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.unlink(filepath)

        return results

    def print_results(self, results: dict):
        """Pretty print benchmark results."""
        print(f"\n{'=' * 60}")
        print("Performance Comparison Results")
        print(f"{'=' * 60}\n")

        traditional = results["traditional"]
        gds = results["gds"]
        improvement = results["improvement"]

        print("Traditional Path (Storage → CPU → GPU):")
        print(f"  Throughput: {traditional['throughput_gbps']:.2f} GB/s")
        print(f"  Latency: {traditional['latency_ms']:.2f} ms")

        print("\nGPUDirect Storage (Storage → GPU):")
        print(f"  Throughput: {gds['throughput_gbps']:.2f} GB/s")
        print(f"  Latency: {gds['latency_ms']:.2f} ms")

        print("\nImprovement with GDS:")
        print(f"  Throughput: {improvement['throughput_pct']:+.1f}%")
        print(f"  Latency: {improvement['latency_pct']:+.1f}%")

        print(f"\n{'=' * 60}\n")

    def cleanup(self):
        """Cleanup cuFile resources."""
        if self.driver_initialized:
            try:
                cufile.driver_close()
            except (cufile.cuFileError, OSError, RuntimeError) as exc:
                raise CuFileError(f"Failed to close cuFile driver: {exc}") from exc
            self.driver_initialized = False
            print("cuFile driver closed")


def main() -> int:
    """Main demonstration function."""
    print("=" * 60)
    print("Chapter 10: GPUDirect Storage (cuFile) Example")
    print("=" * 60)

    demo = None
    exit_code = 0
    try:
        demo = GPUDirectStorageDemo()
        print(f"\nGDS support: {demo.device_description}")
        results = demo.benchmark_comparison(size_mb=256)
        demo.print_results(results)
    except CuFileUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        exit_code = 3
    except CuFileError as exc:
        print(f"GDS execution failed: {exc}", file=sys.stderr)
        exit_code = 1

    if demo is not None:
        try:
            demo.cleanup()
        except CuFileError as exc:
            print(f"GDS cleanup failed: {exc}", file=sys.stderr)
            if exit_code == 0:
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
