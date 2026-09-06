"""Pydantic models for benchmark data structures.

Provides type-safe, validated data models for benchmark results, metrics, and artifacts.
All models include schemaVersion for forward compatibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.benchmark.run_manifest import RunManifest


class MemoryStats(BaseModel):
    """Memory statistics from benchmark execution."""
    
    peak_mb: Optional[float] = Field(None, description="Peak memory allocated in MB")
    allocated_mb: Optional[float] = Field(None, description="Current memory allocated in MB")
    reserved_mb: Optional[float] = Field(None, description="Memory reserved in MB")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        frozen=False,  # Allow mutation for context manager pattern
        json_schema_extra={
            "example": {
                "peak_mb": 1024.5,
                "allocated_mb": 1024.0,
                "reserved_mb": 2048.0,
                "schemaVersion": "1.0"
            }
        }
    )


class TimingStats(BaseModel):
    """Timing statistics from benchmark execution."""
    
    mean_ms: float = Field(..., description="Mean execution time in milliseconds")
    median_ms: float = Field(..., description="Median execution time in milliseconds")
    std_ms: float = Field(..., description="Standard deviation in milliseconds")
    min_ms: float = Field(..., description="Minimum execution time in milliseconds")
    max_ms: float = Field(..., description="Maximum execution time in milliseconds")
    p50_ms: Optional[float] = Field(None, description="50th percentile (median) in milliseconds")
    p90_ms: Optional[float] = Field(None, description="90th percentile in milliseconds")
    p95_ms: Optional[float] = Field(None, description="95th percentile in milliseconds")
    p99_ms: Optional[float] = Field(None, description="99th percentile in milliseconds")
    percentiles: Dict[float, float] = Field(default_factory=dict, description="Percentile values as dict")
    iterations: int = Field(..., description="Number of iterations executed")
    warmup_iterations: int = Field(..., description="Number of warmup iterations")
    raw_times_ms: Optional[List[float]] = Field(default=None, description="Raw timing measurements")
    sample_scope: Optional[str] = Field(
        default=None,
        description="What each raw sample measures, e.g. block_mean or process_wall; None means unspecified",
    )
    iterations_per_sample: Optional[int] = Field(
        default=None, ge=1,
        description="Operations averaged into a block_mean sample, or process launches in a process_wall sample",
    )
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mean_ms": 1.23,
                "median_ms": 1.20,
                "std_ms": 0.05,
                "min_ms": 1.15,
                "max_ms": 1.35,
                "p50_ms": 1.20,
                "p90_ms": 1.30,
                "p95_ms": 1.32,
                "p99_ms": 1.34,
                "percentiles": {"25.0": 1.18, "50.0": 1.20, "75.0": 1.28, "99.0": 1.34},
                "iterations": 100,
                "warmup_iterations": 10,
                "raw_times_ms": [1.15, 1.18, 1.20, 1.28, 1.34],
                "schemaVersion": "1.0"
            }
        }
    )
    
    @field_validator('raw_times_ms', mode='before')
    @classmethod
    def validate_raw_times_ms(cls, v):
        """Validate and clean raw_times_ms field."""
        if v is None:
            return None
        
        # Ensure raw_times_ms is a list of floats, not nested lists or tuples
        cleaned_times = []
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (int, float)):
                    cleaned_times.append(float(item))
                elif isinstance(item, (list, tuple)):
                    # If we somehow got nested lists, flatten them
                    for subitem in item:
                        if isinstance(subitem, (int, float)):
                            cleaned_times.append(float(subitem))
                # Skip None values
        else:
            # If it's a single value, wrap it in a list
            if isinstance(v, (int, float)):
                cleaned_times.append(float(v))
        
        return cleaned_times if cleaned_times else None


class InferenceTimingStats(BaseModel):
    """Inference-specific timing statistics for TTFT (Time to First Token) and TPOT (Time Per Output Token)."""
    
    # TTFT (Time to First Token) statistics
    ttft_mean_ms: float = Field(..., description="Mean Time to First Token in milliseconds")
    ttft_p50_ms: Optional[float] = Field(None, description="50th percentile TTFT in milliseconds")
    ttft_p90_ms: Optional[float] = Field(None, description="90th percentile TTFT in milliseconds")
    ttft_p95_ms: Optional[float] = Field(None, description="95th percentile TTFT in milliseconds")
    ttft_p99_ms: Optional[float] = Field(None, description="99th percentile TTFT in milliseconds")
    ttft_percentiles: Dict[float, float] = Field(default_factory=dict, description="TTFT percentile values as dict")
    
    # TPOT (Time Per Output Token) statistics
    tpot_mean_ms: float = Field(..., description="Mean Time Per Output Token in milliseconds")
    tpot_p50_ms: Optional[float] = Field(None, description="50th percentile TPOT in milliseconds")
    tpot_p90_ms: Optional[float] = Field(None, description="90th percentile TPOT in milliseconds")
    tpot_p95_ms: Optional[float] = Field(None, description="95th percentile TPOT in milliseconds")
    tpot_p99_ms: Optional[float] = Field(None, description="99th percentile TPOT in milliseconds")
    tpot_percentiles: Dict[float, float] = Field(default_factory=dict, description="TPOT percentile values as dict")
    
    # Request and token counts
    num_requests: int = Field(..., description="Number of inference requests measured")
    total_tokens_generated: int = Field(..., description="Total tokens generated across all requests")
    
    # Raw timing data (optional)
    raw_ttft_times_ms: Optional[List[float]] = Field(None, description="Raw TTFT measurements in milliseconds")
    raw_tpot_times_ms: Optional[List[float]] = Field(None, description="Raw TPOT measurements in milliseconds (per-token times)")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ttft_mean_ms": 150.5,
                "ttft_p50_ms": 145.0,
                "ttft_p90_ms": 180.0,
                "ttft_p95_ms": 200.0,
                "ttft_p99_ms": 250.0,
                "ttft_percentiles": {"50.0": 145.0, "90.0": 180.0, "95.0": 200.0, "99.0": 250.0},
                "tpot_mean_ms": 35.2,
                "tpot_p50_ms": 34.0,
                "tpot_p90_ms": 38.0,
                "tpot_p95_ms": 39.5,
                "tpot_p99_ms": 42.0,
                "tpot_percentiles": {"50.0": 34.0, "90.0": 38.0, "95.0": 39.5, "99.0": 42.0},
                "num_requests": 100,
                "total_tokens_generated": 5000,
                "schemaVersion": "1.0"
            }
        }
    )


class ProfilerArtifacts(BaseModel):
    """Paths to profiling artifacts."""
    
    nsys_rep: Optional[str] = Field(None, description="Path to nsys report file")
    ncu_rep: Optional[str] = Field(None, description="Path to ncu report file")
    torch_trace_json: Optional[str] = Field(None, description="Path to PyTorch profiler trace JSON")
    proton_report: Optional[str] = Field(None, description="Path to Proton kernel summary JSON")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nsys_rep": "artifacts/20240101_120000/profiles/benchmark.nsys-rep",
                "ncu_rep": "artifacts/20240101_120000/profiles/benchmark.ncu-rep",
                "torch_trace_json": "artifacts/20240101_120000/profiles/torch_trace.json",
                "proton_report": "artifacts/20240101_120000/profiles/benchmark.proton.json",
                "schemaVersion": "1.0"
            }
        }
    )


class NsysMetrics(BaseModel):
    """Extracted metrics from nsys profiling."""
    
    total_gpu_time_ms: Optional[float] = Field(None, description="Total GPU time in milliseconds")
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Additional raw metrics")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        result = {}
        if self.total_gpu_time_ms is not None:
            result["nsys_total_gpu_time_ms"] = self.total_gpu_time_ms
        result.update({f"nsys_{k}": v for k, v in self.raw_metrics.items()})
        return result


class NcuMetrics(BaseModel):
    """Extracted metrics from ncu profiling."""
    
    kernel_time_ms: Optional[float] = Field(None, description="Kernel execution time in milliseconds")
    range_time_ms: Optional[float] = Field(
        None,
        description="NCU aggregate application-range duration in milliseconds",
    )
    sm_throughput_pct: Optional[float] = Field(None, description="SM compute throughput as % of peak")
    dram_throughput_pct: Optional[float] = Field(None, description="DRAM throughput as % of peak")
    l2_throughput_pct: Optional[float] = Field(None, description="L2 cache throughput as % of peak")
    occupancy_pct: Optional[float] = Field(None, description="GPU occupancy as percentage")
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Additional raw metrics")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        result = {}
        if self.kernel_time_ms is not None:
            result["ncu_kernel_time_ms"] = self.kernel_time_ms
        if self.range_time_ms is not None:
            result["ncu_range_time_ms"] = self.range_time_ms
        if self.sm_throughput_pct is not None:
            result["ncu_sm_throughput_pct"] = self.sm_throughput_pct
        if self.dram_throughput_pct is not None:
            result["ncu_dram_throughput_pct"] = self.dram_throughput_pct
        if self.l2_throughput_pct is not None:
            result["ncu_l2_throughput_pct"] = self.l2_throughput_pct
        if self.occupancy_pct is not None:
            result["ncu_occupancy_pct"] = self.occupancy_pct
        result.update({f"ncu_{k}": v for k, v in self.raw_metrics.items()})
        return result


class ProtonMetrics(BaseModel):
    """Extracted metrics from Proton profiling."""
    
    kernel_count: Optional[int] = Field(None, description="Number of kernels reported by Proton")
    occupancy_limited_kernels: List[str] = Field(default_factory=list, description="Kernels flagged as occupancy-limited")
    summary_stats: Dict[str, float] = Field(default_factory=dict, description="Aggregate Proton metrics (e.g., max regs/shared memory)")
    kernel_summaries: List[Dict[str, Any]] = Field(default_factory=list, description="Raw per-kernel summaries from Proton")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {}
        if self.kernel_count is not None:
            result["proton_kernel_count"] = self.kernel_count
        if self.occupancy_limited_kernels:
            result["proton_occupancy_limited_kernels"] = ",".join(self.occupancy_limited_kernels)
            result["proton_occupancy_limited_kernels_list"] = list(self.occupancy_limited_kernels)
        for key, value in self.summary_stats.items():
            result[f"proton_{key}"] = value
        if self.summary_stats:
            result["proton_summary_stats"] = self.summary_stats
        if self.kernel_summaries:
            names = [
                k.get("name") for k in self.kernel_summaries[:3]
                if isinstance(k, dict) and k.get("name")
            ]
            if names:
                result["proton_top_kernels"] = ",".join(names)
            result["proton_kernel_summaries"] = self.kernel_summaries
        return result


class TorchMetrics(BaseModel):
    """Extracted metrics from PyTorch profiler."""
    
    total_time_ms: Optional[float] = Field(None, description="Total execution time in milliseconds")
    cuda_time_ms: Optional[float] = Field(None, description="CUDA operations time in milliseconds")
    cpu_time_ms: Optional[float] = Field(None, description="CPU time in milliseconds")
    memory_allocated_mb: Optional[float] = Field(None, description="Memory allocated in MB")
    raw_metrics: Dict[str, float] = Field(default_factory=dict, description="Additional raw metrics")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        result = {}
        if self.total_time_ms is not None:
            result["torch_total_time_ms"] = self.total_time_ms
        if self.cuda_time_ms is not None:
            result["torch_cuda_time_ms"] = self.cuda_time_ms
        if self.cpu_time_ms is not None:
            result["torch_cpu_time_ms"] = self.cpu_time_ms
        if self.memory_allocated_mb is not None:
            result["torch_memory_allocated_mb"] = self.memory_allocated_mb
        result.update({f"torch_{k}": v for k, v in self.raw_metrics.items()})
        return result


class ProfilerMetrics(BaseModel):
    """All profiler metrics combined."""
    
    nsys: Optional[NsysMetrics] = Field(None, description="Nsys profiling metrics")
    ncu: Optional[NcuMetrics] = Field(None, description="NCU profiling metrics")
    proton: Optional[ProtonMetrics] = Field(None, description="Proton profiling metrics")
    torch: Optional[TorchMetrics] = Field(None, description="PyTorch profiler metrics")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")


class ThroughputStats(BaseModel):
    """Measured throughput derived from benchmark timing and workload metadata."""
    
    requests_per_s: Optional[float] = Field(None, description="Requests processed per second")
    tokens_per_s: Optional[float] = Field(None, description="Tokens processed per second")
    samples_per_s: Optional[float] = Field(None, description="Samples/items processed per second")
    bytes_per_s: Optional[float] = Field(None, description="Bytes processed per second")
    custom_unit_per_s: Optional[float] = Field(None, description="Custom workload unit throughput per second")
    custom_unit_name: Optional[str] = Field(None, description="Name of the custom workload unit")
    latency_ms: Optional[float] = Field(None, description="Mean latency per iteration in milliseconds")
    goodput: Optional[float] = Field(None, description="Goodput ratio reported by benchmark (0-1)")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests_per_s": 800.0,
                "tokens_per_s": 6400.0,
                "latency_ms": 1.25,
                "schemaVersion": "1.0"
            }
        }
    )


class BenchmarkResult(BaseModel):
    """Comprehensive benchmark result with timing, memory, and profiling data."""
    
    # Timing statistics
    timing: TimingStats = Field(..., description="Timing statistics")
    
    # Inference timing statistics (for inference benchmarks)
    inference_timing: Optional[InferenceTimingStats] = Field(None, description="Inference-specific timing statistics (TTFT/TPOT)")
    
    # Memory statistics
    memory: Optional[MemoryStats] = Field(None, description="Memory statistics")
    
    # Throughput statistics
    throughput: Optional[ThroughputStats] = Field(None, description="Throughput metrics derived from workload metadata")
    
    # Profiling artifacts
    artifacts: Optional[ProfilerArtifacts] = Field(None, description="Profiling artifact paths")
    
    # Profiling metrics
    profiler_metrics: Optional[ProfilerMetrics] = Field(None, description="Extracted profiler metrics")
    
    # Errors and validation
    errors: List[str] = Field(default_factory=list, description="Errors encountered during execution")
    validation_status: Optional[str] = Field(None, description="Validation status (e.g., 'passed', 'failed', 'warning')")
    validation_message: Optional[str] = Field(None, description="Validation message if validation failed")
    
    # Timeout information
    timeout_stage: Optional[str] = Field(None, description="Stage that timed out (e.g., 'setup', 'warmup', 'measurement', 'profiling')")
    timeout_duration_seconds: Optional[float] = Field(None, description="Duration before timeout occurred (in seconds)")
    timeout_limit_seconds: Optional[int] = Field(None, description="Timeout limit that was exceeded (in seconds)")
    
    # Metadata
    benchmark_name: Optional[str] = Field(None, description="Name of the benchmark")
    device: Optional[str] = Field(None, description="Device used (e.g., 'cuda:0', 'cpu')")
    mode: Optional[str] = Field(None, description="Benchmark mode (e.g., 'triton', 'pytorch', 'custom')")
    launch_via: Optional[str] = Field(None, description="Launcher used for execution (python or torchrun)")
    world_size: Optional[int] = Field(None, description="World size used when launching distributed benchmarks")
    multi_gpu: Optional[bool] = Field(None, description="Whether multiple GPUs were used")
    multi_gpu_required: Optional[bool] = Field(None, description="Benchmark declared multi-GPU requirement")
    seeds: Optional[Dict[str, Any]] = Field(None, description="Seeds applied for reproducibility (random, numpy, torch, cuda)")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Benchmark-specific custom metrics")
    story_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Non-numeric benchmark story metadata (comparison pair role, chapter alignment, exemplar status)",
    )
    runtime_env: Dict[str, str] = Field(
        default_factory=dict,
        description="Runtime environment overrides applied during benchmark execution",
    )
    gpu_metrics: Optional[Dict[str, Optional[float | str]]] = Field(
        None,
        description="Snapshot of GPU telemetry (temperature, power, utilization, timestamp)",
    )
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timing": {
                    "mean_ms": 1.23,
                    "median_ms": 1.20,
                    "std_ms": 0.05,
                    "min_ms": 1.15,
                    "max_ms": 1.35,
                    "iterations": 100,
                    "warmup_iterations": 10,
                    "schemaVersion": "1.0"
                },
                "memory": {
                    "peak_mb": 1024.5,
                    "allocated_mb": 1024.0,
                    "schemaVersion": "1.0"
                },
                "errors": [],
                "throughput": {
                    "requests_per_s": 800.0,
                    "tokens_per_s": 6400.0,
                    "latency_ms": 1.25,
                    "schemaVersion": "1.0"
                },
                "custom_metrics": {
                    "scenario_total_phase_ms": 123.4
                },
                "story_metadata": {
                    "pair_role": "comparison",
                    "chapter_alignment": "supplementary",
                    "chapter_native_exemplar": False
                },
                "launch_via": "python",
                "world_size": 1,
                "schemaVersion": "1.0"
            }
        }
    )


class BenchmarkRun(BaseModel):
    """Complete benchmark run with manifest and results.
    
    This is the top-level envelope for a benchmark run, including
    the environment manifest and benchmark results.
    """
    
    # Manifest capturing environment state
    manifest: Optional[RunManifest] = Field(None, description="Run manifest (environment state)")
    
    # Benchmark result
    result: BenchmarkResult = Field(..., description="Benchmark result")
    
    # Metadata
    run_id: Optional[str] = Field(None, description="Unique run identifier")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of run")
    
    schemaVersion: str = Field("1.0", description="Schema version for forward compatibility")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "manifest": {
                    "hardware": {"gpu_model": "A100", "cuda_version": "12.1"},
                    "software": {"pytorch_version": "2.1.0"},
                    "git": {"commit": "abc123", "branch": "main"},
                    "start_time": "2024-01-01T00:00:00",
                    "schemaVersion": "1.0"
                },
                "result": {
                    "timing": {"mean_ms": 1.23},
                    "schemaVersion": "1.0"
                },
                "run_id": "run_20240101_000000",
                "timestamp": "2024-01-01T00:00:00",
                "schemaVersion": "1.0"
            }
        }
    )
