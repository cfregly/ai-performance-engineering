#!/usr/bin/env python3
"""Core-owned kernel roofline analysis shared by chapters and offline reports."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.benchmark.metrics import (
    BLACKWELL_B200,
    HOPPER_H100,
    HardwareSpecs,
    hardware_specs_for_device,
)

VALIDATED_HARDWARE_PROFILES = ("b200", "h100-sxm")

__all__ = [
    "ArchitectureSpecs",
    "RooflineAnalyzer",
    "VALIDATED_HARDWARE_PROFILES",
    "get_architecture_specs",
    "get_architecture_specs_for_profile",
    "profile_for_compute_capability",
]


@dataclass(frozen=True)
class ArchitectureSpecs:
    """Dense per-GPU ceilings for one explicitly identified SKU."""

    name: str
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_fp8_tflops: float
    peak_tf32_tflops: float | None
    memory_bandwidth_gbs: float
    cpu_gpu_bandwidth_gbs: float | None = None
    profile_source: str = "explicit_caller"
    peak_tensor_fp16_tflops: float | None = None


def _from_benchmark_specs(specs: HardwareSpecs, *, profile: str) -> ArchitectureSpecs:
    """Adapt the repository's reviewed dense static profile without copying constants."""
    return ArchitectureSpecs(
        name=specs.name,
        peak_fp32_tflops=specs.fp32_tflops,
        peak_fp16_tflops=specs.fp16_tflops,
        peak_fp8_tflops=specs.fp8_tflops,
        # The reviewed shared profile does not publish a TF32 ceiling. Do not
        # derive one from another precision and present it as a sourced value.
        peak_tf32_tflops=None,
        memory_bandwidth_gbs=specs.hbm_bandwidth_gbps,
        peak_tensor_fp16_tflops=specs.tensor_tflops,
        profile_source=f"core.benchmark.metrics:{profile}:{specs.profile_source}:dense",
    )


def get_architecture_specs_for_profile(profile: str) -> ArchitectureSpecs:
    """Return a reviewed static profile for an explicitly declared exact SKU."""
    normalized = profile.strip().lower().replace("_", "-")
    if normalized == "b200":
        return _from_benchmark_specs(BLACKWELL_B200, profile="b200")
    if normalized == "h100-sxm":
        return _from_benchmark_specs(HOPPER_H100, profile="h100-sxm")
    choices = ", ".join(VALIDATED_HARDWARE_PROFILES)
    raise ValueError(
        f"Unknown or unvalidated hardware profile: {profile}. "
        f"Choose an exact reviewed profile: {choices}"
    )


def profile_for_compute_capability(major: int, minor: int) -> str | None:
    """Refuse to infer a SKU profile from compute capability alone."""
    del major, minor
    return None


def get_architecture_specs() -> ArchitectureSpecs:
    """Select a reviewed local profile using both device name and capability."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available for exact-SKU roofline selection; pass an "
            "explicit reviewed ArchitectureSpecs instance for offline analysis"
        )

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    reviewed = hardware_specs_for_device(props.name, (props.major, props.minor))
    profile = "b200" if reviewed is BLACKWELL_B200 else "h100-sxm"
    return _from_benchmark_specs(reviewed, profile=profile)


class RooflineAnalyzer:
    """Analyze kernel performance using roofline model."""

    def __init__(self, specs: ArchitectureSpecs | None = None) -> None:
        self.specs = specs if specs is not None else get_architecture_specs()

    def analyze_kernel(
        self,
        kernel_time_ms: float,
        flops: float,
        bytes_transferred: float,
        precision: str = "fp32",
    ) -> dict[str, float]:
        """Analyze kernel performance.

        Args:
            kernel_time_ms: Kernel execution time in milliseconds
            flops: Total floating point operations
            bytes_transferred: Total bytes read + written
            precision: "fp32", "fp16", "fp8", or "tf32"

        Returns:
            Dictionary with analysis results
        """
        if not math.isfinite(kernel_time_ms) or kernel_time_ms <= 0:
            raise ValueError("kernel_time_ms must be positive and finite")
        if not math.isfinite(flops) or flops < 0:
            raise ValueError("flops must be non-negative and finite")
        if not math.isfinite(bytes_transferred) or bytes_transferred <= 0:
            raise ValueError("bytes_transferred must be positive and finite")

        # Compute achieved metrics
        achieved_tflops = (flops / 1e12) / (kernel_time_ms / 1000.0)
        achieved_bandwidth_gbs = (bytes_transferred / 1e9) / (kernel_time_ms / 1000.0)
        arithmetic_intensity = flops / bytes_transferred

        # Get peak performance for precision
        peaks = {
            "fp32": self.specs.peak_fp32_tflops,
            "fp16": self.specs.peak_fp16_tflops,
            "tensor_fp16": self.specs.peak_tensor_fp16_tflops,
            "fp8": self.specs.peak_fp8_tflops,
            "tf32": self.specs.peak_tf32_tflops,
        }
        if precision not in peaks:
            choices = ", ".join(peaks)
            raise ValueError(f"Unknown precision {precision!r}; choose one of: {choices}")
        peak_tflops = peaks[precision]
        if peak_tflops is None or not math.isfinite(peak_tflops) or peak_tflops <= 0:
            raise ValueError(
                f"{precision} has no reviewed finite positive peak for {self.specs.name}"
            )
        if (
            not math.isfinite(self.specs.memory_bandwidth_gbs)
            or self.specs.memory_bandwidth_gbs <= 0
        ):
            raise ValueError("memory_bandwidth_gbs must be positive and finite")

        # The memory roof uses the architecture's peak bandwidth. Using the
        # achieved bandwidth here would make this value algebraically identical
        # to achieved TFLOPS and would classify every finite kernel as compute-bound.
        memory_bound_tflops = (
            self.specs.memory_bandwidth_gbs / 1000.0
        ) * arithmetic_intensity

        # Compute utilization
        compute_utilization = (achieved_tflops / peak_tflops) * 100.0
        memory_utilization = (achieved_bandwidth_gbs / self.specs.memory_bandwidth_gbs) * 100.0

        # Ridge point (where compute and memory rooflines meet)
        ridge_point = (peak_tflops * 1000.0) / self.specs.memory_bandwidth_gbs
        is_memory_bound = arithmetic_intensity < ridge_point
        is_compute_bound = not is_memory_bound

        return {
            "achieved_tflops": achieved_tflops,
            "achieved_bandwidth_gbs": achieved_bandwidth_gbs,
            "arithmetic_intensity": arithmetic_intensity,
            "peak_tflops": peak_tflops,
            "peak_bandwidth_gbs": self.specs.memory_bandwidth_gbs,
            "compute_utilization_pct": compute_utilization,
            "memory_utilization_pct": memory_utilization,
            "is_memory_bound": is_memory_bound,
            "is_compute_bound": is_compute_bound,
            "memory_bound_tflops": memory_bound_tflops,
            "ridge_point": ridge_point,
        }

    def print_analysis(self, results: dict[str, float], kernel_name: str = "Kernel") -> None:
        """Pretty print analysis results."""
        print("=" * 80)
        print(f"Roofline Analysis: {kernel_name}")
        print("=" * 80)
        print(f"Architecture: {self.specs.name}")
        print(f"Profile source: {self.specs.profile_source}")
        print()
        print("Achieved Performance:")
        print(
            f"  Compute: {results['achieved_tflops']:8.2f} TFLOPS "
            f"({results['compute_utilization_pct']:5.1f}% of "
            f"{results['peak_tflops']:.0f} TFLOPS peak)"
        )
        print(
            f"  Bandwidth: {results['achieved_bandwidth_gbs']:8.2f} GB/s "
            f"({results['memory_utilization_pct']:5.1f}% of "
            f"{results['peak_bandwidth_gbs']:.0f} GB/s peak)"
        )
        print(f"  Arithmetic Intensity: {results['arithmetic_intensity']:.2f} FLOPs/byte")
        print()

        # Bottleneck analysis
        print("Bottleneck Analysis:")
        if results["is_memory_bound"]:
            print("  Status: ❗ MEMORY-BOUND")
            print(f"  Memory roofline limit: {results['memory_bound_tflops']:.2f} TFLOPS")
            print()
            print("  Recommendations:")
            print("    1. Increase arithmetic intensity (more compute per byte)")
            print("    2. Reduce memory traffic: use shared memory / cache blocking")
            print("    3. Use bulk-transfer features supported by the selected GPU SKU")
        else:
            print("  Status: ✓ COMPUTE-BOUND")
            print(f"  Compute utilization: {results['compute_utilization_pct']:.1f}%")
            print()
            print("  Recommendations:")
            print("    1. Better occupancy (more warps)")
            print("    2. Use tensor cores for matmul")
            print("    3. Consider lower precision (FP16/FP8)")

        print("=" * 80)
