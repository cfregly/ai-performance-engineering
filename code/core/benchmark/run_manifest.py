"""Run manifest schema for capturing complete environment state.

Captures hardware, software, environment, and git state for reproducibility and debugging.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import torch
from core.benchmark.evaluation_provenance import (
    EvaluationContract,
    EvaluationFailure,
    EvaluationProvenance,
    finalize_evaluation as finalize_evaluation_provenance,
    start_evaluation,
)
from core.profiling.gpu_telemetry import query_gpu_telemetry

try:
    from core.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - fallback for minimal environments
    import logging

    logger = logging.getLogger(__name__)

try:
    import triton
    TRITON_VERSION = triton.__version__
except ImportError:
    TRITON_VERSION = None

from pydantic import BaseModel, Field, ConfigDict, field_serializer

PROJECT_ROOT = Path(__file__).parents[2]
SCHEMA_VERSION = "1.0"


class GitStatusDict(TypedDict):
    commit: Optional[str]
    branch: Optional[str]
    dirty: bool


class CudaInfoDict(TypedDict):
    version: Optional[str]
    driver_version: Optional[str]


class GpuInfoDict(TypedDict):
    model: Optional[str]
    compute_capability: Optional[str]


class GpuStateDict(TypedDict):
    gpu_clock_mhz: Optional[int]
    memory_clock_mhz: Optional[int]
    gpu_app_clock_mhz: Optional[int]
    memory_app_clock_mhz: Optional[int]
    persistence_mode: Optional[bool]
    power_limit_w: Optional[float]
    power_draw_w: Optional[float]
    temperature_gpu_c: Optional[float]
    temperature_memory_c: Optional[float]
    fan_speed_pct: Optional[float]
    utilization_gpu_pct: Optional[float]
    utilization_memory_pct: Optional[float]


def get_git_info() -> GitStatusDict:
    """Get git commit hash, branch, and dirty flag.
    
    Returns:
        Dictionary with 'commit', 'branch', and 'dirty' keys.
    """
    git_info: GitStatusDict = {
        "commit": None,
        "branch": None,
        "dirty": False,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Check if working directory is dirty. Porcelain status catches
        # staged changes and untracked files, not just unstaged diffs.
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            git_info["dirty"] = bool(result.stdout.strip())
        
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Git not available or not a git repo
        pass
    
    return git_info


def get_cuda_info() -> CudaInfoDict:
    """Get CUDA version and driver version.
    
    Returns:
        Dictionary with 'version' and 'driver_version' keys.
    """
    cuda_info: CudaInfoDict = {
        "version": None,
        "driver_version": None,
    }
    
    try:
        # Get CUDA version from PyTorch
        if torch.cuda.is_available():
            cuda_info["version"] = torch.version.cuda
        
        # Get driver version
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            driver_versions = result.stdout.strip().split("\n")
            if driver_versions:
                cuda_info["driver_version"] = driver_versions[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return cuda_info


def get_gpu_info() -> GpuInfoDict:
    """Get GPU model and compute capability.
    
    Returns:
        Dictionary with 'model' and 'compute_capability' keys.
    """
    gpu_info: GpuInfoDict = {
        "model": None,
        "compute_capability": None,
    }
    
    try:
        if torch.cuda.is_available():
            # Get GPU model
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                models = result.stdout.strip().split("\n")
                if models:
                    gpu_info["model"] = models[0].strip()
            
            # Get compute capability
            if torch.cuda.device_count() > 0:
                device = torch.cuda.current_device()
                major, minor = torch.cuda.get_device_capability(device)
                gpu_info["compute_capability"] = f"{major}.{minor}"
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return gpu_info


def get_gpu_state(*, allow_telemetry_failures: bool = False) -> GpuStateDict:
    """Get GPU state information (clocks, app clocks, persistence mode, power limit).
    
    Returns:
        Dictionary with clock and power metadata for the current GPU.
    """
    gpu_state: GpuStateDict = {
        "gpu_clock_mhz": None,
        "memory_clock_mhz": None,
        "gpu_app_clock_mhz": None,
        "memory_app_clock_mhz": None,
        "persistence_mode": None,
        "power_limit_w": None,
        "power_draw_w": None,
        "temperature_gpu_c": None,
        "temperature_memory_c": None,
        "fan_speed_pct": None,
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
    }
    
    try:
        if torch.cuda.is_available():
            # Force refresh so app_clock is accurate when benchmarks lock clocks
            # and we capture state immediately after the lock is applied.
            try:
                telemetry = query_gpu_telemetry(force_refresh=True)
            except Exception:
                if not allow_telemetry_failures:
                    raise
                telemetry = None
                logger.warning(
                    "Portable validity mode: GPU telemetry fields unavailable; capturing partial GPU state.",
                    exc_info=True,
                )
            if telemetry:
                if telemetry.get("graphics_clock_mhz") is not None:
                    try:
                        gpu_state["gpu_clock_mhz"] = int(telemetry["graphics_clock_mhz"])
                    except (TypeError, ValueError):
                        pass
                if telemetry.get("memory_clock_mhz") is not None:
                    try:
                        gpu_state["memory_clock_mhz"] = int(telemetry["memory_clock_mhz"])
                    except (TypeError, ValueError):
                        pass
                if telemetry.get("applications_clock_sm_mhz") is not None:
                    try:
                        gpu_state["gpu_app_clock_mhz"] = int(telemetry["applications_clock_sm_mhz"])
                    except (TypeError, ValueError):
                        pass
                if telemetry.get("applications_clock_memory_mhz") is not None:
                    try:
                        gpu_state["memory_app_clock_mhz"] = int(telemetry["applications_clock_memory_mhz"])
                    except (TypeError, ValueError):
                        pass
                gpu_state["power_draw_w"] = telemetry.get("power_draw_w")
                gpu_state["temperature_gpu_c"] = telemetry.get("temperature_gpu_c")
                gpu_state["temperature_memory_c"] = telemetry.get("temperature_memory_c")
                gpu_state["fan_speed_pct"] = telemetry.get("fan_speed_pct")
                gpu_state["utilization_gpu_pct"] = telemetry.get("utilization_gpu_pct")
                gpu_state["utilization_memory_pct"] = telemetry.get("utilization_memory_pct")
            
            # Get persistence mode
            result = subprocess.run(
                ["nvidia-smi", "-q", "-d", "PERSISTENCE_MODE"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse "Persistence Mode : Enabled" or "Persistence Mode : Disabled"
                if "Persistence Mode" in result.stdout:
                    gpu_state["persistence_mode"] = "Enabled" in result.stdout
            
            # Get power limit
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                limits = result.stdout.strip().split("\n")
                if limits and limits[0]:
                    # Parse "250.00 W" format
                    limit_str = limits[0].strip().replace(" W", "")
                    try:
                        gpu_state["power_limit_w"] = float(limit_str)
                    except ValueError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return gpu_state


def reset_gpu_state() -> None:
    """Reset GPU state for cold start (clears cache, resets memory stats).
    
    This function performs a "cold start" by:
    - Clearing CUDA cache
    - Resetting peak memory statistics
    - Synchronizing CUDA operations
    
    Note: This does not reset GPU clocks or power limits (requires root/admin).
    For full GPU reset, use: sudo nvidia-smi --gpu-reset
    """
    if not torch.cuda.is_available():
        return
    
    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Reset peak memory statistics
        torch.cuda.reset_peak_memory_stats()
        
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
    except Exception as exc:
        logger.warning("Failed to reset GPU state (non-fatal): %s", exc)


def _append_collection_warning(collection_warnings: list[str], message: str) -> None:
    if message not in collection_warnings:
        collection_warnings.append(message)


def _collect_runtime_capability_limitations(
    collection_warnings: list[str],
) -> list["RuntimeCapabilityLimitation"]:
    """Load structured runtime capability limitations from the validity registry."""
    try:
        from core.harness.validity_checks import get_runtime_capability_limitations
    except Exception as exc:
        _append_collection_warning(
            collection_warnings,
            "Failed to import runtime capability limitation registry from "
            f"core.harness.validity_checks: {exc}",
        )
        return []

    limitations: list[RuntimeCapabilityLimitation] = []
    try:
        raw_limitations = get_runtime_capability_limitations()
    except Exception as exc:
        _append_collection_warning(
            collection_warnings,
            "Failed to read runtime capability limitations from the validity registry: "
            f"{exc}",
        )
        return []

    for raw in raw_limitations:
        try:
            limitations.append(RuntimeCapabilityLimitation(**raw))
        except Exception as exc:
            key = raw.get("key", "unknown") if isinstance(raw, dict) else "unknown"
            _append_collection_warning(
                collection_warnings,
                "Failed to serialize runtime capability limitation "
                f"{key!r} into the manifest: {exc}",
            )
    return limitations


class HardwareInfo(BaseModel):
    """Hardware information."""
    
    gpu_model: Optional[str] = Field(None, description="GPU model name")
    cuda_version: Optional[str] = Field(None, description="CUDA toolkit version")
    driver_version: Optional[str] = Field(None, description="NVIDIA driver version")
    compute_capability: Optional[str] = Field(None, description="GPU compute capability (e.g., '9.0')")
    
    # GPU state for reproducibility
    gpu_clock_mhz: Optional[int] = Field(None, description="GPU clock frequency in MHz")
    memory_clock_mhz: Optional[int] = Field(None, description="Memory clock frequency in MHz")
    gpu_app_clock_mhz: Optional[int] = Field(None, description="GPU application clock in MHz")
    memory_app_clock_mhz: Optional[int] = Field(None, description="Memory application clock in MHz")
    persistence_mode: Optional[bool] = Field(None, description="GPU persistence mode enabled")
    power_limit_w: Optional[float] = Field(None, description="GPU power limit in watts")
    power_draw_w: Optional[float] = Field(None, description="Current GPU power draw in watts")
    temperature_gpu_c: Optional[float] = Field(None, description="GPU temperature in Celsius")
    temperature_memory_c: Optional[float] = Field(None, description="HBM temperature in Celsius")
    fan_speed_pct: Optional[float] = Field(None, description="GPU fan speed percentage")
    utilization_gpu_pct: Optional[float] = Field(None, description="GPU SM utilization percentage")
    utilization_memory_pct: Optional[float] = Field(None, description="GPU memory controller utilization percentage")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class SoftwareInfo(BaseModel):
    """Software version information."""
    
    pytorch_version: str = Field(..., description="PyTorch version")
    triton_version: Optional[str] = Field(None, description="Triton version")
    python_version: str = Field(..., description="Python version")
    os: str = Field(..., description="Operating system")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class EnvironmentInfo(BaseModel):
    """Environment variable information."""
    
    cuda_visible_devices: Optional[str] = Field(None, description="CUDA_VISIBLE_DEVICES value")
    relevant_env_vars: Dict[str, str] = Field(default_factory=dict, description="Other relevant environment variables")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class GitInfo(BaseModel):
    """Git repository information."""
    
    commit: Optional[str] = Field(None, description="Git commit hash")
    branch: Optional[str] = Field(None, description="Git branch name")
    dirty: bool = Field(False, description="Whether working directory has uncommitted changes")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class SeedInfo(BaseModel):
    """Random seed information for reproducibility."""
    
    random_seed: Optional[int] = Field(None, description="Python random.seed() value")
    numpy_seed: Optional[int] = Field(None, description="numpy.random.seed() value")
    torch_seed: Optional[int] = Field(None, description="torch.manual_seed() value")
    cuda_seed: Optional[int] = Field(None, description="torch.cuda.manual_seed_all() value")
    deterministic_mode: bool = Field(False, description="Whether deterministic algorithms were enabled")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class RuntimeCapabilityLimitation(BaseModel):
    """Structured runtime capability limitation captured during a benchmark run."""

    key: str = Field(..., description="Stable identifier for the limitation")
    category: str = Field(..., description="Broad limitation category")
    component: str = Field(..., description="Runtime component that observed the limitation")
    summary: str = Field(..., description="Human-readable summary of the limitation")
    detail: Optional[str] = Field(None, description="Underlying runtime error or detail string")
    first_observed_at: Optional[str] = Field(None, description="UTC timestamp when the limitation was first observed")

    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class ComparisonResult(BaseModel):
    """Details of output comparison between baseline and optimized."""
    
    passed: bool = Field(..., description="Whether comparison passed")
    max_diff: Optional[float] = Field(None, description="Maximum difference found")
    location: Optional[list] = Field(None, description="Index location of max difference")
    expected_sample: Optional[float] = Field(None, description="Expected value at max diff location")
    actual_sample: Optional[float] = Field(None, description="Actual value at max diff location")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class ToleranceUsed(BaseModel):
    """Tolerance specification used for comparison."""
    
    rtol: float = Field(..., description="Relative tolerance")
    atol: float = Field(..., description="Absolute tolerance")
    justification: Optional[str] = Field(None, description="Justification if looser than defaults")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class WorkloadMetrics(BaseModel):
    """Workload metrics for verification."""
    
    bytes_per_iteration: Optional[float] = Field(None, description="Bytes processed per iteration")
    tokens_per_iteration: Optional[float] = Field(None, description="Tokens processed per iteration")
    ops_per_iteration: Optional[float] = Field(None, description="Operations per iteration")
    samples_per_iteration: Optional[float] = Field(None, description="Samples processed per iteration")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class VerifyManifestEntry(BaseModel):
    """Verify results stored in run manifest - complete field enumeration for CI parsing.
    
    This captures all verification-related information for a benchmark pair,
    including comparison results, checksums, workload metrics, and any
    exemptions or overrides.
    """
    
    # Core verify status
    verify_status: str = Field(
        ..., 
        description="Verification status: passed, failed, skipped, or quarantined"
    )
    
    # Checksums
    baseline_checksum: Optional[str] = Field(None, description="Checksum/hash of baseline output")
    optimized_checksum: Optional[str] = Field(None, description="Checksum/hash of optimized output")
    
    # Comparison details
    comparison_result: Optional[ComparisonResult] = Field(None, description="Detailed comparison results")
    
    # Timing and identification
    timestamp: datetime = Field(..., description="When verification was performed")
    signature_hash: str = Field(..., description="Hash of input signature for cache keying")
    
    # Workload tracking
    workload_metrics: Optional[WorkloadMetrics] = Field(None, description="Baseline workload metrics")
    workload_delta: Optional[Dict[str, float]] = Field(
        None, 
        description="Relative differences in workload metrics between baseline/optimized"
    )
    workload_ratio_justification: Optional[str] = Field(
        None,
        description="Justification for expected workload ratio difference"
    )
    
    # Quarantine info
    quarantine_reason: Optional[str] = Field(None, description="Reason for quarantine if quarantined")
    
    # Tolerance tracking
    tolerance_used: Optional[ToleranceUsed] = Field(None, description="Tolerance used for comparison")
    tolerance_override_justification: Optional[str] = Field(
        None,
        description="Justification when using looser tolerances than defaults"
    )
    
    # Exemption declarations
    jitter_exemption_reason: Optional[str] = Field(
        None,
        description="Reason jitter check was skipped"
    )
    
    # Seed tracking
    seed_info: Optional[Dict[str, int]] = Field(None, description="Seeds used in verify mode")
    
    # CUDA-specific
    cuda_verify_mode: Optional[bool] = Field(None, description="Whether CUDA verify path was used")
    cuda_binary_clean: Optional[bool] = Field(
        None,
        description="Whether perf binary has no VERIFY symbols"
    )
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")


class RunManifest(BaseModel):
    """Complete run manifest capturing environment state.
    
    This manifest is generated at the start of each benchmark run and included
    in all result files for reproducibility and debugging.
    """
    
    # Hardware information
    hardware: HardwareInfo = Field(..., description="Hardware configuration")
    
    # Software information
    software: SoftwareInfo = Field(..., description="Software versions")
    
    # Environment information
    environment: EnvironmentInfo = Field(..., description="Environment variables")
    
    # Git information
    git: GitInfo = Field(..., description="Git repository state")
    
    # Seed information for reproducibility
    seeds: Optional[SeedInfo] = Field(None, description="Random seed values used for reproducibility")
    
    # Verification results (if verify mode was run)
    verify: Optional[VerifyManifestEntry] = Field(None, description="Verification results for this run")

    # Explicit opt-in provenance for workloads that make dataset evaluation claims.
    # Kernel and systems microbenchmarks leave this unset.
    evaluation: Optional[EvaluationProvenance] = Field(
        None,
        description="Evaluation contract receipt, when the workload explicitly opts in",
    )

    # Non-fatal provenance / collection warnings
    collection_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal collection issues that degraded provenance or artifact completeness",
    )

    runtime_capability_limitations: list[RuntimeCapabilityLimitation] = Field(
        default_factory=list,
        description="Structured runtime capability gaps observed during this process that may partially degrade benchmark cleanup or provenance fidelity",
    )
    
    # Timestamps
    start_time: datetime = Field(..., description="Run start timestamp")
    end_time: Optional[datetime] = Field(None, description="Run end timestamp")
    duration_seconds: Optional[float] = Field(None, description="Run duration in seconds")
    
    # Configuration (serialized BenchmarkConfig)
    config: Optional[Dict] = Field(None, description="Serialized BenchmarkConfig used for this run")
    
    schemaVersion: str = Field(SCHEMA_VERSION, description="Schema version for forward compatibility")
    
    @classmethod
    def create(cls, config: Optional[Dict] = None, start_time: Optional[datetime] = None) -> RunManifest:
        """Create a RunManifest with current environment state.
        
        Args:
            config: Optional serialized BenchmarkConfig dictionary
            start_time: Optional start time (defaults to now)
        
        Returns:
            RunManifest instance with current environment captured
        """
        if start_time is None:
            start_time = datetime.now()
        
        collection_warnings: list[str] = []
        runtime_capability_limitations = _collect_runtime_capability_limitations(collection_warnings)

        # Get hardware info
        cuda_info = get_cuda_info()
        gpu_info = get_gpu_info()
        validity_profile = str((config or {}).get("validity_profile", "strict")).strip().lower()
        gpu_state = get_gpu_state(allow_telemetry_failures=validity_profile == "portable")
        hardware = HardwareInfo(
            gpu_model=gpu_info.get("model"),
            cuda_version=cuda_info.get("version"),
            driver_version=cuda_info.get("driver_version"),
            compute_capability=gpu_info.get("compute_capability"),
            gpu_clock_mhz=gpu_state.get("gpu_clock_mhz"),
            memory_clock_mhz=gpu_state.get("memory_clock_mhz"),
            gpu_app_clock_mhz=gpu_state.get("gpu_app_clock_mhz"),
            memory_app_clock_mhz=gpu_state.get("memory_app_clock_mhz"),
            persistence_mode=gpu_state.get("persistence_mode"),
            power_limit_w=gpu_state.get("power_limit_w"),
            power_draw_w=gpu_state.get("power_draw_w"),
            temperature_gpu_c=gpu_state.get("temperature_gpu_c"),
            temperature_memory_c=gpu_state.get("temperature_memory_c"),
            fan_speed_pct=gpu_state.get("fan_speed_pct"),
            utilization_gpu_pct=gpu_state.get("utilization_gpu_pct"),
            utilization_memory_pct=gpu_state.get("utilization_memory_pct"),
            schemaVersion=SCHEMA_VERSION,
        )
        
        # Get software info
        software = SoftwareInfo(
            pytorch_version=torch.__version__,
            triton_version=TRITON_VERSION,
            python_version=sys.version.split()[0],
            os=sys.platform,
            schemaVersion=SCHEMA_VERSION,
        )
        
        # Get environment info
        env_vars: Dict[str, str] = {}
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        # Capture other relevant environment variables
        relevant_vars = [
            "CUDA_DEVICE_ORDER",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TORCH_COMPILE_DEBUG",
            "TRITON_CACHE_DIR",
            "TRITON_OVERRIDE_DIR",
            "TRITON_DUMP_DIR",
        ]
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                env_vars[var] = value
        
        environment = EnvironmentInfo(
            cuda_visible_devices=cuda_visible_devices,
            relevant_env_vars=env_vars,
            schemaVersion=SCHEMA_VERSION,
        )
        
        # Get git info
        git_info_dict = get_git_info()
        git = GitInfo(
            commit=git_info_dict.get("commit"),
            branch=git_info_dict.get("branch"),
            dirty=bool(git_info_dict.get("dirty", False)),
            schemaVersion=SCHEMA_VERSION,
        )
        missing_git_fields = [
            field_name
            for field_name in ("commit", "branch")
            if not git_info_dict.get(field_name)
        ]
        if missing_git_fields:
            collection_warnings.append(
                "Git metadata unavailable for fields: "
                f"{', '.join(missing_git_fields)}; manifest provenance is incomplete."
            )
        
        # Extract seed info from config if present
        seeds = None
        if config:
            seed = config.get("seed")
            deterministic = config.get("deterministic", False)
            if seed is not None or deterministic:
                seeds = SeedInfo(
                    random_seed=seed,
                    numpy_seed=seed,
                    torch_seed=seed,
                    cuda_seed=seed,
                    deterministic_mode=deterministic,
                    schemaVersion=SCHEMA_VERSION,
                )
        
        return cls(
            hardware=hardware,
            software=software,
            environment=environment,
            git=git,
            seeds=seeds,
            collection_warnings=collection_warnings,
            runtime_capability_limitations=runtime_capability_limitations,
            start_time=start_time,
            end_time=None,
            duration_seconds=None,
            config=config,
            schemaVersion=SCHEMA_VERSION,
        )
    
    def begin_evaluation(self, contract: EvaluationContract) -> EvaluationProvenance:
        """Validate and snapshot an opted-in evaluation before execution."""

        if self.evaluation is not None:
            self.evaluation.failures.append(
                EvaluationFailure(
                    code="evaluation_already_started",
                    phase="start",
                    message="evaluation provenance may be started only once",
                )
            )
            self.evaluation.status = "FAIL"
            return self.evaluation
        self.evaluation = start_evaluation(contract)
        return self.evaluation

    def finalize_evaluation(
        self,
        current_contract: Optional[EvaluationContract],
    ) -> Optional[EvaluationProvenance]:
        """Finalize an opted-in evaluation and retain structured failures."""

        if self.evaluation is None:
            if current_contract is None:
                return None
            self.evaluation = start_evaluation(current_contract)
            self.evaluation.failures.append(
                EvaluationFailure(
                    code="evaluation_not_started",
                    phase="finalize",
                    message="evaluation provenance was not captured before execution",
                )
            )
            self.evaluation.status = "FAIL"
        self.evaluation = finalize_evaluation_provenance(
            self.evaluation,
            current_contract,
        )
        return self.evaluation

    def finalize(self, end_time: Optional[datetime] = None) -> None:
        """Finalize the manifest with end time and duration.
        
        Args:
            end_time: Optional end time (defaults to now)
        """
        self.refresh_runtime_capability_limitations()
        if self.evaluation is not None and self.evaluation.finalized_at is None:
            # A caller that does not supply the post-run declaration cannot silently
            # turn an unfinished evaluation receipt into a successful manifest.
            self.finalize_evaluation(None)
        if end_time is None:
            end_time = datetime.now()
        
        self.end_time = end_time
        if self.start_time:
            delta = end_time - self.start_time
            self.duration_seconds = delta.total_seconds()

    def refresh_runtime_capability_limitations(self) -> None:
        """Refresh structured runtime capability limitations from the active process registry."""
        latest = _collect_runtime_capability_limitations(self.collection_warnings)
        merged = {
            limitation.key: limitation
            for limitation in self.runtime_capability_limitations
        }
        for limitation in latest:
            merged[limitation.key] = limitation
        self.runtime_capability_limitations = [merged[key] for key in sorted(merged)]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hardware": {
                    "gpu_model": "NVIDIA B200-SXM5-192GB",
                    "cuda_version": "13.0",
                    "driver_version": "580.126.09",
                    "compute_capability": "10.0",
                    "schemaVersion": "1.0"
                },
                "software": {
                    "pytorch_version": "2.9.1+cu130",
                    "triton_version": "3.5.0",
                    "python_version": "3.12.0",
                    "os": "linux",
                    "schemaVersion": "1.0"
                },
                "environment": {
                    "cuda_visible_devices": "0",
                    "relevant_env_vars": {},
                    "schemaVersion": "1.0"
                },
                "git": {
                    "commit": "abc123def456",
                    "branch": "main",
                    "dirty": False,
                    "schemaVersion": "1.0"
                },
                "collection_warnings": [],
                "runtime_capability_limitations": [],
                "start_time": "2024-01-01T12:00:00",
                "schemaVersion": "1.0"
            }
        }
    )

    @field_serializer("start_time", "end_time", when_used="json", mode="plain")
    def _serialize_datetime(self, dt: Optional[datetime], info=None) -> Optional[str]:  # type: ignore[override]
        """Serialize datetimes defensively to handle odd call signatures."""
        try:
            return dt.isoformat() if dt else None
        except AttributeError:
            # Some pydantic versions pass SerializationInfo unexpectedly
            return None
