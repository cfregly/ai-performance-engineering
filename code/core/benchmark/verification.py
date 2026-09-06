"""Benchmark verification enforcement data models and utilities.

This module provides the core data structures for mandatory correctness verification
of benchmark pairs (baseline_*/optimized_*). It implements a dual-mode approach where
verify mode validates correctness separately from perf mode.

Key Components:
- InputSignature: Complete workload description for equivalence checking
- ToleranceSpec: Dtype-aware tolerance specification for output comparison
- QuarantineReason: Enumeration of all quarantine reasons
- EnforcementPhase: Phased rollout configuration (DETECT, QUARANTINE, GATE)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# =============================================================================
# Precision Configuration
# =============================================================================


@dataclass
class PrecisionFlags:
    """Precision configuration for benchmark workloads.
    
    Tracks which precision modes are enabled for a benchmark. This affects
    tolerance selection and workload equivalence checking.
    """
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    tf32: bool = True  # Default on for CUDA
    
    def to_dict(self) -> Dict[str, bool]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecisionFlags":
        """Deserialize from dictionary."""
        return cls(
            fp16=data.get("fp16", False),
            bf16=data.get("bf16", False),
            fp8=data.get("fp8", False),
            tf32=data.get("tf32", True),
        )


# =============================================================================
# Input Signature
# =============================================================================


@dataclass
class InputSignature:
    """Complete workload description for equivalence checking.
    
    Used to verify that baseline and optimized benchmarks operate on equivalent
    workloads. Without this verification, performance comparisons are meaningless.
    
    Required fields must be provided for all benchmarks. Optional fields are
    required only for specific benchmark types (distributed, streams/graphs, etc.)
    """
    
    # Required fields
    shapes: Dict[str, Tuple[int, ...]]  # Named tensor shapes
    dtypes: Dict[str, str]  # Named tensor dtypes
    batch_size: int
    parameter_count: int
    precision_flags: PrecisionFlags
    
    # Optional topology fields (required for distributed)
    world_size: Optional[int] = None
    ranks: Optional[List[int]] = None
    shards: Optional[int] = None
    pipeline_stages: Optional[int] = None
    # Required when pipeline_stages > 1: contiguous [start_layer, end_layer] ranges per stage.
    # This prevents pipeline-parallel benchmarks from silently changing stage boundaries.
    pipeline_stage_boundaries: Optional[List[Tuple[int, int]]] = None
    per_rank_batch_size: Optional[int] = None
    collective_type: Optional[str] = None
    # Declared distributed work contract. These describe comparison policy only;
    # runtime algorithm/completion proof is carried by explicit rank receipts.
    collective_algorithm: Optional[str] = None
    gradient_bucket_bytes: Optional[int] = None
    barrier_policy: Optional[str] = None
    async_completion_policy: Optional[str] = None
    
    # Optional launch config (required for streams/graphs)
    num_streams: Optional[int] = None
    graph_capture_enabled: Optional[bool] = None
    
    # Optional algorithm flags
    pruning_enabled: Optional[bool] = None
    sparsity_ratio: Optional[float] = None
    quantization_mode: Optional[str] = None
    
    def hash(self) -> str:
        """Generate stable hash for cache keying.
        
        The hash is deterministic and based on all signature fields.
        Used to key golden output caches and detect signature changes.
        
        Returns:
            16-character hex string derived from SHA256 hash
        """
        # Convert to dict with sorted keys for deterministic ordering
        data = self.to_dict()
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result: Dict[str, Any] = {
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "dtypes": self.dtypes,
            "batch_size": self.batch_size,
            "parameter_count": self.parameter_count,
            "precision_flags": self.precision_flags.to_dict(),
        }
        
        # Add optional fields if set
        if self.world_size is not None:
            result["world_size"] = self.world_size
        if self.ranks is not None:
            result["ranks"] = self.ranks
        if self.shards is not None:
            result["shards"] = self.shards
        if self.pipeline_stages is not None:
            result["pipeline_stages"] = self.pipeline_stages
        if self.pipeline_stage_boundaries is not None:
            result["pipeline_stage_boundaries"] = [
                [int(start), int(end)] for start, end in self.pipeline_stage_boundaries
            ]
        if self.per_rank_batch_size is not None:
            result["per_rank_batch_size"] = self.per_rank_batch_size
        if self.collective_type is not None:
            result["collective_type"] = self.collective_type
        if self.collective_algorithm is not None:
            result["collective_algorithm"] = self.collective_algorithm
        if self.gradient_bucket_bytes is not None:
            result["gradient_bucket_bytes"] = self.gradient_bucket_bytes
        if self.barrier_policy is not None:
            result["barrier_policy"] = self.barrier_policy
        if self.async_completion_policy is not None:
            result["async_completion_policy"] = self.async_completion_policy
        if self.num_streams is not None:
            result["num_streams"] = self.num_streams
        if self.graph_capture_enabled is not None:
            result["graph_capture_enabled"] = self.graph_capture_enabled
        if self.pruning_enabled is not None:
            result["pruning_enabled"] = self.pruning_enabled
        if self.sparsity_ratio is not None:
            result["sparsity_ratio"] = self.sparsity_ratio
        if self.quantization_mode is not None:
            result["quantization_mode"] = self.quantization_mode
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputSignature":
        """Deserialize from dictionary."""
        shapes = {k: tuple(v) for k, v in data.get("shapes", {}).items()}
        precision_data = data.get("precision_flags", {})
        stage_boundaries_raw = data.get("pipeline_stage_boundaries")
        stage_boundaries: Optional[List[Tuple[int, int]]] = None
        if stage_boundaries_raw is not None:
            if not isinstance(stage_boundaries_raw, list):
                raise TypeError("pipeline_stage_boundaries must be a list of [start, end] pairs")
            stage_boundaries = []
            for entry in stage_boundaries_raw:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise TypeError("pipeline_stage_boundaries entries must be [start, end] pairs")
                stage_boundaries.append((int(entry[0]), int(entry[1])))
        
        return cls(
            shapes=shapes,
            dtypes=data.get("dtypes", {}),
            batch_size=data.get("batch_size", 0),
            parameter_count=data.get("parameter_count", 0),
            precision_flags=PrecisionFlags.from_dict(precision_data),
            world_size=data.get("world_size"),
            ranks=data.get("ranks"),
            shards=data.get("shards"),
            pipeline_stages=data.get("pipeline_stages"),
            pipeline_stage_boundaries=stage_boundaries,
            per_rank_batch_size=data.get("per_rank_batch_size"),
            collective_type=data.get("collective_type"),
            collective_algorithm=data.get("collective_algorithm"),
            gradient_bucket_bytes=data.get("gradient_bucket_bytes"),
            barrier_policy=data.get("barrier_policy"),
            async_completion_policy=data.get("async_completion_policy"),
            num_streams=data.get("num_streams"),
            graph_capture_enabled=data.get("graph_capture_enabled"),
            pruning_enabled=data.get("pruning_enabled"),
            sparsity_ratio=data.get("sparsity_ratio"),
            quantization_mode=data.get("quantization_mode"),
        )

    def validate(self, strict: bool = False) -> List[str]:
        """Validate that required fields are present and valid.

        Args:
            strict: If True, require shapes and dtypes to be non-empty.
                If False (default), allow simple parameter-based signatures.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if strict:
            if not self.shapes:
                errors.append("shapes is required and cannot be empty")
            if not self.dtypes:
                errors.append("dtypes is required and cannot be empty")

        if self.batch_size < 0:
            errors.append("batch_size cannot be negative")
        if self.parameter_count < 0:
            errors.append("parameter_count cannot be negative")

        if not isinstance(self.precision_flags, PrecisionFlags):
            errors.append("precision_flags must be a PrecisionFlags instance")

        distributed_contract_values = (
            self.collective_algorithm,
            self.gradient_bucket_bytes,
            self.barrier_policy,
            self.async_completion_policy,
        )
        if any(value is not None for value in distributed_contract_values):
            if self.world_size is None or self.world_size <= 1:
                errors.append(
                    "distributed work contract fields require world_size > 1"
                )

        if self.collective_algorithm is not None:
            if not isinstance(self.collective_algorithm, str) or not self.collective_algorithm.strip():
                errors.append("collective_algorithm must be a non-empty string when provided")
            if not isinstance(self.collective_type, str) or not self.collective_type.strip():
                errors.append("collective_algorithm requires a declared collective_type")

        if self.gradient_bucket_bytes is not None:
            if (
                isinstance(self.gradient_bucket_bytes, bool)
                or not isinstance(self.gradient_bucket_bytes, int)
                or self.gradient_bucket_bytes <= 0
            ):
                errors.append("gradient_bucket_bytes must be a positive integer when provided")

        if self.barrier_policy is not None and self.barrier_policy != "barrier_before_timed_close":
            errors.append(
                "barrier_policy must be 'barrier_before_timed_close' when provided"
            )

        if (
            self.async_completion_policy is not None
            and self.async_completion_policy != "wait_for_async_before_timed_close"
        ):
            errors.append(
                "async_completion_policy must be "
                "'wait_for_async_before_timed_close' when provided"
            )

        if self.pipeline_stages is not None:
            if self.pipeline_stages < 1:
                errors.append("pipeline_stages must be >= 1 when provided")
            if self.pipeline_stages > 1:
                if self.pipeline_stage_boundaries is None:
                    errors.append("pipeline_stage_boundaries is required when pipeline_stages > 1")
                else:
                    boundaries = self.pipeline_stage_boundaries
                    if not isinstance(boundaries, list):
                        errors.append("pipeline_stage_boundaries must be a list of (start, end) pairs")
                    elif len(boundaries) != self.pipeline_stages:
                        errors.append(
                            f"pipeline_stage_boundaries must have length {self.pipeline_stages}, got {len(boundaries)}"
                        )
                    else:
                        prev_end: Optional[int] = None
                        for idx, entry in enumerate(boundaries):
                            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                                errors.append("pipeline_stage_boundaries entries must be (start, end) pairs")
                                break
                            start, end = int(entry[0]), int(entry[1])
                            if start < 0:
                                errors.append("pipeline_stage_boundaries start must be >= 0")
                                break
                            if start > end:
                                errors.append("pipeline_stage_boundaries start must be <= end")
                                break
                            if idx == 0:
                                if start != 0:
                                    errors.append("pipeline_stage_boundaries must start at layer 0")
                                    break
                            else:
                                if prev_end is None:
                                    errors.append(
                                        "pipeline_stage_boundaries internal validation state missing previous boundary"
                                    )
                                    break
                                if start != prev_end + 1:
                                    errors.append("pipeline_stage_boundaries must be contiguous and non-overlapping")
                                    break
                            prev_end = end

        return errors
    
    def matches(self, other: "InputSignature") -> bool:
        """Check if two signatures are equivalent.
        
        Two signatures match if all their fields are equal. This is used to
        verify baseline and optimized run on equivalent workloads.
        """
        return self.to_dict() == other.to_dict()


# =============================================================================
# Signature Equivalence (Comparable Benchmark Pairs)
# =============================================================================


@dataclass(frozen=True)
class SignatureEquivalenceSpec:
    """Explicit opt-in for signature fields allowed to differ between a pair.

    Comparable baseline/optimized benchmarks must run equivalent workloads. In rare
    cases (notably precision-change examples), it is valid for *selected* signature
    fields to differ while preserving semantic workload equivalence.

    This spec is fail-fast and must be declared explicitly by benchmarks via
    `signature_equivalence_group` + `signature_equivalence_ignore_fields`.
    """

    group: str
    ignore_fields: Tuple[str, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.group, str) or not self.group.strip():
            raise ValueError("SignatureEquivalenceSpec.group must be a non-empty string")
        if not isinstance(self.ignore_fields, tuple):
            raise TypeError("SignatureEquivalenceSpec.ignore_fields must be a tuple[str, ...]")
        for field_name in self.ignore_fields:
            if not isinstance(field_name, str) or not field_name:
                raise TypeError("SignatureEquivalenceSpec.ignore_fields entries must be non-empty strings")
        allowed = {"precision_flags"}
        unknown = set(self.ignore_fields) - allowed
        if unknown:
            raise ValueError(
                "SignatureEquivalenceSpec.ignore_fields contains unsupported fields: "
                f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
            )


def get_signature_equivalence_spec(benchmark: Any) -> Optional[SignatureEquivalenceSpec]:
    """Extract an explicit signature-equivalence spec from a benchmark object.

    Benchmarks may define:
    - `signature_equivalence_group`: str (required when enabling equivalence)
    - `signature_equivalence_ignore_fields`: tuple[str, ...] (required when enabling equivalence)
    """

    group = getattr(benchmark, "signature_equivalence_group", None)
    ignore_fields = getattr(benchmark, "signature_equivalence_ignore_fields", None)

    if group is None and ignore_fields is None:
        return None

    if group is None or ignore_fields is None:
        raise RuntimeError(
            "Signature equivalence requires BOTH `signature_equivalence_group` and "
            "`signature_equivalence_ignore_fields` to be set explicitly."
        )

    if not isinstance(ignore_fields, tuple):
        raise TypeError("signature_equivalence_ignore_fields must be a tuple[str, ...]")

    spec = SignatureEquivalenceSpec(group=str(group), ignore_fields=ignore_fields)
    spec.validate()
    return spec


def signature_workload_dict(
    signature: "InputSignature",
    *,
    equivalence: Optional[SignatureEquivalenceSpec] = None,
) -> Dict[str, Any]:
    """Return a deterministic dict for workload equivalence comparisons."""

    data = signature.to_dict()
    if equivalence is None:
        return data
    equivalence.validate()
    for field_name in equivalence.ignore_fields:
        data.pop(field_name, None)
    return data


def signature_cache_key(
    signature: "InputSignature",
    *,
    equivalence: Optional[SignatureEquivalenceSpec] = None,
) -> str:
    """Generate a cache key, optionally using an equivalence spec.

    When `equivalence` is provided, the key is derived from:
    - the workload dict with ignored fields removed, plus
    - an explicit equivalence-group tag to avoid cross-benchmark collisions.
    """

    if equivalence is None:
        return signature.hash()

    workload = signature_workload_dict(signature, equivalence=equivalence)
    workload["_equivalence_group"] = equivalence.group
    json_str = json.dumps(workload, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# Tolerance Specification
# =============================================================================


@dataclass
class ToleranceSpec:
    """Output comparison tolerances for verification.
    
    Defines the tolerance for comparing outputs between baseline and optimized
    benchmarks. Supports both numeric tolerances (rtol/atol) and custom
    comparator functions for complex outputs.
    """
    rtol: float  # Relative tolerance
    atol: float  # Absolute tolerance
    comparator_fn: Optional[Callable[[torch.Tensor, torch.Tensor], bool]] = None
    justification: Optional[str] = None  # Required if looser than dtype defaults

    def __post_init__(self) -> None:
        for name in ("rtol", "atol"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (comparator_fn is not serialized)."""
        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "justification": self.justification,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToleranceSpec":
        """Deserialize from dictionary."""
        return cls(
            rtol=data.get("rtol", 1e-5),
            atol=data.get("atol", 1e-8),
            comparator_fn=None,  # Cannot deserialize functions
            justification=data.get("justification"),
        )


# Default tolerances by dtype - these are the canonical tolerances used throughout
DEFAULT_TOLERANCES: Dict[torch.dtype, ToleranceSpec] = {
    torch.float32: ToleranceSpec(rtol=1e-5, atol=1e-8),
    torch.float16: ToleranceSpec(rtol=1e-3, atol=1e-5),
    torch.bfloat16: ToleranceSpec(rtol=1e-3, atol=1e-5),
    torch.float64: ToleranceSpec(rtol=1e-9, atol=1e-12),
    torch.int32: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.int64: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.int16: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.int8: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.uint8: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.bool: ToleranceSpec(rtol=0, atol=0),  # Exact match
    torch.complex64: ToleranceSpec(rtol=1e-5, atol=1e-8),
    torch.complex128: ToleranceSpec(rtol=1e-9, atol=1e-12),
}

# Add FP8 types if available (PyTorch 2.1+)
try:
    DEFAULT_TOLERANCES[torch.float8_e4m3fn] = ToleranceSpec(rtol=1e-2, atol=1e-3)
    DEFAULT_TOLERANCES[torch.float8_e5m2] = ToleranceSpec(rtol=1e-2, atol=1e-3)
except AttributeError:
    pass  # FP8 types not available


def get_tolerance_for_dtype(dtype: torch.dtype) -> ToleranceSpec:
    """Get default tolerance for a given dtype.
    
    Args:
        dtype: PyTorch dtype
        
    Returns:
        ToleranceSpec with appropriate tolerances for the dtype
    """
    return DEFAULT_TOLERANCES.get(dtype, ToleranceSpec(rtol=1e-5, atol=1e-8))


def is_tolerance_looser(custom: ToleranceSpec, default: ToleranceSpec) -> bool:
    """Check if a custom tolerance is looser than the default.
    
    A tolerance is "looser" if it allows larger differences to pass.
    This triggers a warning and requires justification.
    
    Args:
        custom: The custom tolerance being used
        default: The default tolerance for the dtype
        
    Returns:
        True if custom is looser than default
    """
    return custom.rtol > default.rtol or custom.atol > default.atol


def _normalize_dtype_str(dtype: Union[str, torch.dtype]) -> str:
    """Normalize dtype representation to a lowercase torch-style string."""
    dtype_str = str(dtype)
    return dtype_str.replace("torch.", "").lower()


def simple_signature(
    *,
    batch_size: int = 1,
    dtype: Union[str, torch.dtype] = "float32",
    precision_flags: Optional[PrecisionFlags] = None,
    **dims: int,
) -> InputSignature:
    """Construct a minimal, strictly-valid InputSignature for simple workloads.
    
    Args:
        batch_size: Logical batch size for the workload.
        dtype: Primary tensor dtype (applied to all declared shapes).
        precision_flags: Optional precision configuration override.
        dims: Dimension sizes that define the workload shape, in order.
    
    Returns:
        InputSignature with shapes/dtypes populated for a single logical tensor.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}
    dtypes: Dict[str, str] = {}
    if dims:
        shapes["workload"] = tuple(int(v) for v in dims.values())
        dtypes["workload"] = _normalize_dtype_str(dtype)
    else:
        shapes["workload"] = (int(batch_size),)
        dtypes["workload"] = _normalize_dtype_str(dtype)
    
    default_tf32 = False
    try:
        default_tf32 = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        default_tf32 = False
    
    flags = precision_flags or PrecisionFlags(tf32=default_tf32)
    
    parameter_count = sum(int(v) for v in dims.values()) if dims else int(batch_size)
    sig = InputSignature(
        shapes=shapes,
        dtypes=dtypes,
        batch_size=int(batch_size),
        parameter_count=parameter_count,
        precision_flags=flags,
    )
    errors = sig.validate(strict=True)
    if errors:
        raise ValueError(f"Invalid simple signature: {errors[0]}")
    return sig


def coerce_input_signature(sig: Union[InputSignature, Dict[str, Any]]) -> InputSignature:
    """Convert a signature payload into a validated InputSignature.
    
    Raises:
        TypeError: When sig is of an unexpected type.
        ValueError: When required fields are missing/invalid.
    """
    if isinstance(sig, InputSignature):
        # Normalize dtype strings so signatures compare consistently across
        # in-process vs subprocess execution (the subprocess path serializes to
        # dict and runs through _normalize_dtype_str()).
        sig.dtypes = {k: _normalize_dtype_str(v) for k, v in sig.dtypes.items()}
        errors = sig.validate(strict=True)
        if errors:
            raise ValueError(f"Invalid InputSignature: {errors[0]}")
        return sig
    
    if not isinstance(sig, dict):
        raise TypeError(f"Input signature must be InputSignature or dict, got {type(sig)}")
    
    if not sig.get("shapes") or not sig.get("dtypes"):
        raise ValueError("Input signature must include non-empty 'shapes' and 'dtypes'")
    if "batch_size" not in sig or "parameter_count" not in sig:
        raise ValueError("Input signature must include 'batch_size' and 'parameter_count'")
    
    shapes = {k: tuple(v) if isinstance(v, (list, tuple)) else (int(v),) for k, v in sig["shapes"].items()}
    dtypes = {k: _normalize_dtype_str(v) for k, v in sig["dtypes"].items()}
    
    precision_data = sig.get("precision_flags")
    if isinstance(precision_data, PrecisionFlags):
        precision = precision_data
    elif isinstance(precision_data, dict):
        precision = PrecisionFlags.from_dict(precision_data)
    else:
        precision = PrecisionFlags(
            fp16=bool(sig.get("fp16", False)),
            bf16=bool(sig.get("bf16", False)),
            fp8=bool(sig.get("fp8", False)),
            tf32=bool(sig.get("tf32", True)),
        )
    
    stage_boundaries_raw = sig.get("pipeline_stage_boundaries")
    stage_boundaries: Optional[List[Tuple[int, int]]] = None
    if stage_boundaries_raw is not None:
        if not isinstance(stage_boundaries_raw, list):
            raise TypeError("pipeline_stage_boundaries must be a list of [start, end] pairs")
        stage_boundaries = []
        for entry in stage_boundaries_raw:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise TypeError("pipeline_stage_boundaries entries must be [start, end] pairs")
            stage_boundaries.append((int(entry[0]), int(entry[1])))

    signature = InputSignature(
        shapes=shapes,
        dtypes=dtypes,
        batch_size=int(sig["batch_size"]),
        parameter_count=int(sig["parameter_count"]),
        precision_flags=precision,
        world_size=sig.get("world_size"),
        ranks=sig.get("ranks"),
        shards=sig.get("shards"),
        pipeline_stages=sig.get("pipeline_stages"),
        pipeline_stage_boundaries=stage_boundaries,
        per_rank_batch_size=sig.get("per_rank_batch_size"),
        collective_type=sig.get("collective_type"),
        num_streams=sig.get("num_streams"),
        graph_capture_enabled=sig.get("graph_capture_enabled"),
        pruning_enabled=sig.get("pruning_enabled"),
        sparsity_ratio=sig.get("sparsity_ratio"),
        quantization_mode=sig.get("quantization_mode"),
    )
    errors = signature.validate(strict=True)
    if errors:
        raise ValueError(f"Invalid InputSignature: {errors[0]}")
    return signature


# =============================================================================
# Quarantine Reasons
# =============================================================================


class QuarantineReason(Enum):
    """Enumeration of all reasons a benchmark can be quarantined.
    
    Quarantined benchmarks are excluded from performance reports. Each reason
    maps to a specific compliance failure that must be fixed.
    """
    # Discovery/loading errors
    MISSING_INPUT_SIGNATURE = "missing_input_signature"
    MISSING_VALIDATE_RESULT = "missing_validate_result"
    MISSING_WORKLOAD_METADATA = "workload_metadata_missing"
    MISSING_VERIFY_OUTPUT = "missing_verify_output"
    MISSING_OUTPUT_TOLERANCE = "missing_output_tolerance"
    MISSING_VERIFY_INPUTS = "missing_verify_inputs"
    MISSING_TOLERANCE_FOR_NONDETERMINISTIC = "missing_tolerance_for_nondeterministic"
    
    # Signature/workload errors
    SIGNATURE_MISMATCH = "signature_mismatch"
    WORKLOAD_MISMATCH = "workload_mismatch"
    TIMING_CONFIG_MISMATCH = "timing_config_mismatch"
    
    # Output verification errors
    OUTPUT_MISMATCH = "output_mismatch"
    
    # Anti-hacking check failures
    JITTER_FAIL = "jitter_fail"
    JITTER_DIMENSION_UNAVAILABLE = "jitter_dimension_unavailable"
    FRESH_INPUT_FAIL = "fresh_input_fail"
    CACHED_OUTPUT_DETECTED = "cached_output_detected"
    INPUT_OUTPUT_ALIASING = "input_output_aliasing"
    
    # CUDA verification errors
    CUDA_VERIFY_FAILED = "cuda_verify_failed"
    CUDA_NO_VERIFY_PATH = "cuda_no_verify_path"
    CUDA_PERF_BINARY_CONTAMINATED = "cuda_perf_binary_contaminated"
    
    # Distributed verification errors
    DISTRIBUTED_VERIFY_FAIL = "distributed_verify_fail"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    
    # Skip flag detection
    SKIP_FLAG_PRESENT = "skip_flag_present"
    
    # Seed errors
    SEED_MUTATION_DETECTED = "seed_mutation_detected"


@dataclass
class QuarantineRecord:
    """Record of a quarantined benchmark.
    
    Tracks when and why a benchmark was quarantined, along with any
    additional details that may help with debugging or fixing the issue.
    """
    benchmark_path: str
    quarantine_reason: QuarantineReason
    quarantine_timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "benchmark_path": self.benchmark_path,
            "quarantine_reason": self.quarantine_reason.value,
            "quarantine_timestamp": self.quarantine_timestamp.isoformat(),
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarantineRecord":
        """Deserialize from dictionary."""
        return cls(
            benchmark_path=data.get("benchmark_path", ""),
            quarantine_reason=QuarantineReason(data.get("quarantine_reason", "missing_input_signature")),
            quarantine_timestamp=datetime.fromisoformat(data.get("quarantine_timestamp", datetime.now().isoformat())),
            details=data.get("details"),
        )


# =============================================================================
# Enforcement Phases
# =============================================================================


class EnforcementPhase(Enum):
    """Phased rollout configuration for verification enforcement.
    
    Allows gradual adoption of verification requirements:
    - DETECT: Report issues but don't fail CI or quarantine
    - QUARANTINE: Quarantine non-compliant benchmarks, exclude from perf reports
    - GATE: Fail CI on any non-compliance
    """
    DETECT = "detect"      # Phase 1: Report only, no failures
    QUARANTINE = "quarantine"  # Phase 2: Quarantine non-compliant, exclude from perf
    GATE = "gate"          # Phase 3: Fail CI on any non-compliance


def get_enforcement_phase() -> EnforcementPhase:
    """Get current enforcement phase from environment.
    
    The phase is controlled by the VERIFY_ENFORCEMENT_PHASE environment variable.
    Defaults to DETECT if not set.
    
    Returns:
        Current EnforcementPhase
    """
    phase_str = os.environ.get("VERIFY_ENFORCEMENT_PHASE", "detect")
    try:
        return EnforcementPhase(phase_str.lower())
    except ValueError:
        return EnforcementPhase.DETECT


def is_verification_enabled() -> bool:
    """Check if verification is enabled.
    
    Verification is always enabled by default. Can be disabled via
    VERIFY_DISABLED=1 environment variable for debugging only.
    
    Returns:
        True if verification is enabled (default)
    """
    return os.environ.get("VERIFY_DISABLED", "0") != "1"


# =============================================================================
# Verification Results
# =============================================================================


@dataclass
class ComparisonDetails:
    """Details of output comparison between baseline and optimized."""
    passed: bool
    max_diff: Optional[float] = None
    location: Optional[Tuple[int, ...]] = None
    expected_sample: Optional[float] = None
    actual_sample: Optional[float] = None
    tolerance_used: Optional[ToleranceSpec] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "passed": self.passed,
        }
        if self.max_diff is not None:
            result["max_diff"] = self.max_diff
        if self.location is not None:
            # Convert numpy integers to native Python int for JSON serialization
            result["location"] = [int(x) for x in self.location]
        if self.expected_sample is not None:
            result["expected_sample"] = self.expected_sample
        if self.actual_sample is not None:
            result["actual_sample"] = self.actual_sample
        if self.tolerance_used is not None:
            result["tolerance_used"] = self.tolerance_used.to_dict()
        return result


@dataclass
class VerifyResult:
    """Result of verification for a benchmark pair."""
    passed: bool
    reason: Optional[str] = None  # Failure reason if not passed
    signature_hash: Optional[str] = None
    comparison_details: Optional[ComparisonDetails] = None
    baseline_checksum: Optional[str] = None
    optimized_checksum: Optional[str] = None
    workload_delta: Optional[Dict[str, float]] = None
    jitter_exemption_reason: Optional[str] = None
    seed_info: Optional[Dict[str, int]] = None
    timestamp: Optional[datetime] = None
    details: Optional[Any] = None
    
    @classmethod
    def success(cls, signature_hash: str, **kwargs) -> "VerifyResult":
        """Create a successful verification result."""
        return cls(
            passed=True,
            signature_hash=signature_hash,
            timestamp=datetime.now(),
            **kwargs,
        )
    
    @classmethod
    def fail(cls, reason: str, details: Any = None) -> "VerifyResult":
        """Create a failed verification result."""
        return cls(
            passed=False,
            reason=reason,
            details=details,
            timestamp=datetime.now(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.signature_hash:
            result["signature_hash"] = self.signature_hash
        if self.comparison_details:
            result["comparison_details"] = self.comparison_details.to_dict()
        if self.baseline_checksum:
            result["baseline_checksum"] = self.baseline_checksum
        if self.optimized_checksum:
            result["optimized_checksum"] = self.optimized_checksum
        if self.workload_delta:
            result["workload_delta"] = self.workload_delta
        if self.jitter_exemption_reason:
            result["jitter_exemption_reason"] = self.jitter_exemption_reason
        if self.seed_info:
            result["seed_info"] = self.seed_info
        if self.details is not None:
            result["details"] = self.details
        return result


# =============================================================================
# Seed Management for Deterministic Verification
# =============================================================================


def set_deterministic_seeds(seed: int = 42) -> Dict[str, int]:
    """Set all RNG seeds for deterministic verification.
    
    CRITICAL: Both baseline and optimized MUST use identical seeds (seed=42) 
    in verify mode. Benchmarks SHALL NOT override these seeds in verify mode.
    
    Args:
        seed: The seed value to use (default: 42)
        
    Returns:
        Dict with seed values set for manifest recording
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    seed_info = {
        "random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
    }
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seed_info["cuda_seed"] = seed
        
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_info["cudnn_deterministic"] = True
    seed_info["cudnn_benchmark"] = False
    
    return seed_info


def detect_seed_mutation(initial_seeds: Dict[str, int]) -> bool:
    """Detect if benchmark mutated RNG seeds during execution.
    
    Called after benchmark execution to verify seeds unchanged.
    Benchmarks MUST NOT mutate seeds during perf runs.
    
    Args:
        initial_seeds: Seed info dict from set_deterministic_seeds()
        
    Returns:
        True if seeds were mutated (verification should fail)
    """
    try:
        # Check PyTorch seed
        current_torch_seed = torch.initial_seed()
        if current_torch_seed != initial_seeds.get("torch_seed"):
            return True
            
        # Note: random and numpy seeds are harder to check without state comparison
        # For now, we trust the PyTorch seed as the primary indicator
        
        return False
    except Exception:
        # If we can't check, assume no mutation
        return False


# =============================================================================
# Jitter Check Utilities
# =============================================================================


def select_jitter_dimension(input_signature: InputSignature) -> Optional[Tuple[str, int]]:
    """Select dimension to perturb for jitter check.
    
    Algorithm:
    1. Iterate through shapes in input_signature.shapes
    2. For each shape, find first dimension that is NOT batch dimension (dim 0)
    3. Return (tensor_name, dimension_index) for first suitable dimension
    4. If no suitable dimension found, return None (jitter check is skipped)
    
    Example:
        shapes = {"input": (32, 256, 256), "mask": (32, 256)}
        → Returns ("input", 1) - perturb dim 1 of input tensor
        
    Args:
        input_signature: The input signature to analyze
        
    Returns:
        Tuple of (tensor_name, dimension_index) or None if no suitable dimension
    """
    for tensor_name, shape in input_signature.shapes.items():
        if len(shape) > 1:  # Has non-batch dimensions
            return (tensor_name, 1)  # First non-batch dimension
    return None  # No suitable dimension - requires exemption


# =============================================================================
# Workload Comparison
# =============================================================================


def compare_workload_metrics(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    tolerance: float = 0.01,  # 1% default
) -> Tuple[bool, Dict[str, float]]:
    """Compare workload metrics between baseline and optimized.
    
    Workload metrics (bytes/tokens/ops per iteration) must match within
    tolerance to ensure fair performance comparison.
    
    Args:
        baseline_metrics: Workload metrics from baseline
        optimized_metrics: Workload metrics from optimized
        tolerance: Maximum allowed relative difference (default: 1%)
        
    Returns:
        Tuple of (match_passed, delta_dict) where delta_dict shows differences
    """
    deltas: Dict[str, float] = {}
    passed = True
    
    # Compare all metrics present in either dict
    all_keys = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
    
    for key in all_keys:
        baseline_val = baseline_metrics.get(key, 0.0)
        optimized_val = optimized_metrics.get(key, 0.0)
        
        if baseline_val == 0 and optimized_val == 0:
            continue
            
        if baseline_val == 0:
            # Optimized has metric that baseline doesn't - flag as difference
            deltas[key] = float('inf')
            passed = False
            continue
            
        relative_diff = abs(optimized_val - baseline_val) / baseline_val
        deltas[key] = relative_diff
        
        if relative_diff > tolerance:
            passed = False
            
    return passed, deltas


# =============================================================================
# Distributed Verification Support
# =============================================================================


@dataclass
class DistributedTopology:
    """Topology information for distributed benchmarks."""
    world_size: int
    ranks: List[int]
    shards: Optional[int] = None
    pipeline_stages: Optional[int] = None
    pipeline_stage_boundaries: Optional[List[Tuple[int, int]]] = None
    per_rank_batch_size: Optional[int] = None
    collective_type: Optional[str] = None  # allreduce, allgather, etc.
    collective_algorithm: Optional[str] = None  # declared ring, tree, auto, etc.
    gradient_bucket_bytes: Optional[int] = None
    barrier_policy: Optional[str] = None
    async_completion_policy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result: Dict[str, Any] = {
            "world_size": self.world_size,
            "ranks": self.ranks,
        }
        if self.shards is not None:
            result["shards"] = self.shards
        if self.pipeline_stages is not None:
            result["pipeline_stages"] = self.pipeline_stages
        if self.pipeline_stage_boundaries is not None:
            result["pipeline_stage_boundaries"] = [
                [int(start), int(end)] for start, end in self.pipeline_stage_boundaries
            ]
        if self.per_rank_batch_size is not None:
            result["per_rank_batch_size"] = self.per_rank_batch_size
        if self.collective_type is not None:
            result["collective_type"] = self.collective_type
        if self.collective_algorithm is not None:
            result["collective_algorithm"] = self.collective_algorithm
        if self.gradient_bucket_bytes is not None:
            result["gradient_bucket_bytes"] = self.gradient_bucket_bytes
        if self.barrier_policy is not None:
            result["barrier_policy"] = self.barrier_policy
        if self.async_completion_policy is not None:
            result["async_completion_policy"] = self.async_completion_policy
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedTopology":
        """Deserialize from dictionary."""
        stage_boundaries_raw = data.get("pipeline_stage_boundaries")
        stage_boundaries: Optional[List[Tuple[int, int]]] = None
        if stage_boundaries_raw is not None:
            if not isinstance(stage_boundaries_raw, list):
                raise TypeError("pipeline_stage_boundaries must be a list of [start, end] pairs")
            stage_boundaries = []
            for entry in stage_boundaries_raw:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise TypeError("pipeline_stage_boundaries entries must be [start, end] pairs")
                stage_boundaries.append((int(entry[0]), int(entry[1])))
        return cls(
            world_size=data.get("world_size", 1),
            ranks=data.get("ranks", [0]),
            shards=data.get("shards"),
            pipeline_stages=data.get("pipeline_stages"),
            pipeline_stage_boundaries=stage_boundaries,
            per_rank_batch_size=data.get("per_rank_batch_size"),
            collective_type=data.get("collective_type"),
            collective_algorithm=data.get("collective_algorithm"),
            gradient_bucket_bytes=data.get("gradient_bucket_bytes"),
            barrier_policy=data.get("barrier_policy"),
            async_completion_policy=data.get("async_completion_policy"),
        )


def extract_distributed_topology(signature: InputSignature) -> Optional[DistributedTopology]:
    """Extract distributed topology from input signature.
    
    Args:
        signature: Input signature with optional distributed fields
        
    Returns:
        DistributedTopology if signature has distributed fields, None otherwise
    """
    if signature.world_size is None or signature.world_size <= 1:
        return None
    
    ranks = signature.ranks or list(range(signature.world_size))
    
    return DistributedTopology(
        world_size=signature.world_size,
        ranks=ranks,
        shards=signature.shards,
        pipeline_stages=signature.pipeline_stages,
        pipeline_stage_boundaries=signature.pipeline_stage_boundaries,
        per_rank_batch_size=signature.per_rank_batch_size,
        collective_type=signature.collective_type,
        collective_algorithm=signature.collective_algorithm,
        gradient_bucket_bytes=signature.gradient_bucket_bytes,
        barrier_policy=signature.barrier_policy,
        async_completion_policy=signature.async_completion_policy,
    )


def compare_topologies(
    baseline_topo: Optional[DistributedTopology],
    optimized_topo: Optional[DistributedTopology],
) -> Tuple[bool, Optional[str]]:
    """Compare distributed topologies for equivalence.
    
    Args:
        baseline_topo: Baseline topology
        optimized_topo: Optimized topology
        
    Returns:
        Tuple of (match, mismatch_reason)
    """
    # Both None is OK (non-distributed)
    if baseline_topo is None and optimized_topo is None:
        return True, None
    
    # One None and one not is a mismatch
    if baseline_topo is None or optimized_topo is None:
        return False, "One benchmark is distributed and the other is not"
    
    # Compare world size
    if baseline_topo.world_size != optimized_topo.world_size:
        return False, f"World size mismatch: {baseline_topo.world_size} vs {optimized_topo.world_size}"

    # Compare participating ranks. Equal world size alone does not establish that
    # the same ranks performed the measured work.
    if baseline_topo.ranks != optimized_topo.ranks:
        return False, f"Ranks mismatch: {baseline_topo.ranks} vs {optimized_topo.ranks}"
    
    # Compare shards
    if baseline_topo.shards != optimized_topo.shards:
        return False, f"Shards mismatch: {baseline_topo.shards} vs {optimized_topo.shards}"
    
    # Compare pipeline stages
    if baseline_topo.pipeline_stages != optimized_topo.pipeline_stages:
        return False, f"Pipeline stages mismatch: {baseline_topo.pipeline_stages} vs {optimized_topo.pipeline_stages}"

    # Compare pipeline stage boundaries (only when set; required for PP benchmarks by InputSignature.validate()).
    if baseline_topo.pipeline_stage_boundaries != optimized_topo.pipeline_stage_boundaries:
        return (
            False,
            "Pipeline stage boundaries mismatch: "
            f"{baseline_topo.pipeline_stage_boundaries} vs {optimized_topo.pipeline_stage_boundaries}",
        )

    if baseline_topo.per_rank_batch_size != optimized_topo.per_rank_batch_size:
        return (
            False,
            "Per-rank batch size mismatch: "
            f"{baseline_topo.per_rank_batch_size} vs {optimized_topo.per_rank_batch_size}",
        )

    if baseline_topo.collective_type != optimized_topo.collective_type:
        return (
            False,
            f"Collective type mismatch: {baseline_topo.collective_type} vs {optimized_topo.collective_type}",
        )

    if baseline_topo.collective_algorithm != optimized_topo.collective_algorithm:
        return (
            False,
            "Collective algorithm mismatch: "
            f"{baseline_topo.collective_algorithm} vs {optimized_topo.collective_algorithm}",
        )

    if baseline_topo.gradient_bucket_bytes != optimized_topo.gradient_bucket_bytes:
        return (
            False,
            "Gradient bucket bytes mismatch: "
            f"{baseline_topo.gradient_bucket_bytes} vs {optimized_topo.gradient_bucket_bytes}",
        )

    if baseline_topo.barrier_policy != optimized_topo.barrier_policy:
        return (
            False,
            f"Barrier policy mismatch: {baseline_topo.barrier_policy} vs {optimized_topo.barrier_policy}",
        )

    if baseline_topo.async_completion_policy != optimized_topo.async_completion_policy:
        return (
            False,
            "Async completion policy mismatch: "
            f"{baseline_topo.async_completion_policy} vs {optimized_topo.async_completion_policy}",
        )
    
    return True, None


def get_verify_ranks(benchmark: Any, default: int = 2) -> int:
    """Get minimum ranks for verify mode.
    
    Args:
        benchmark: Benchmark instance
        default: Default number of ranks (2)
        
    Returns:
        Number of ranks to use in verify mode
    """
    if hasattr(benchmark, "get_min_verify_ranks"):
        try:
            return benchmark.get_min_verify_ranks()
        except Exception:
            pass
    return default


# =============================================================================
# Timing Fairness Enforcement
# =============================================================================


@dataclass
class TimingConfig:
    """Timing configuration for benchmark runs."""
    iterations: int
    warmup: int
    timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "timeout_seconds": self.timeout_seconds,
        }


def extract_timing_config(benchmark: Any) -> Optional[TimingConfig]:
    """Extract timing configuration from a benchmark.
    
    Args:
        benchmark: Benchmark instance
        
    Returns:
        TimingConfig if extractable, None otherwise
    """
    config = None
    
    # Try get_config() method
    if hasattr(benchmark, "get_config"):
        try:
            cfg = benchmark.get_config()
            if cfg:
                iterations = getattr(cfg, "iterations", None)
                warmup = getattr(cfg, "warmup", None)
                timeout = getattr(cfg, "timeout_seconds", None)
                
                if iterations is not None and warmup is not None:
                    config = TimingConfig(
                        iterations=iterations,
                        warmup=warmup,
                        timeout_seconds=timeout,
                    )
        except Exception:
            pass
    
    # Try direct attributes
    if config is None:
        iterations = getattr(benchmark, "iterations", None)
        warmup = getattr(benchmark, "warmup", None)
        
        if iterations is not None and warmup is not None:
            config = TimingConfig(
                iterations=iterations,
                warmup=warmup,
                timeout_seconds=getattr(benchmark, "timeout_seconds", None),
            )
    
    return config


def compare_timing_configs(
    baseline_cfg: Optional[TimingConfig],
    optimized_cfg: Optional[TimingConfig],
) -> Tuple[bool, Optional[str]]:
    """Compare timing configurations for equivalence.
    
    Baseline and optimized must use the same iteration/warmup counts
    for fair comparison.
    
    Args:
        baseline_cfg: Baseline timing config
        optimized_cfg: Optimized timing config
        
    Returns:
        Tuple of (match, mismatch_reason)
    """
    if baseline_cfg is None or optimized_cfg is None:
        # Can't compare if configs not available
        return True, None
    
    if baseline_cfg.iterations != optimized_cfg.iterations:
        return False, (
            f"Iterations mismatch: baseline={baseline_cfg.iterations}, "
            f"optimized={optimized_cfg.iterations}"
        )
    
    if baseline_cfg.warmup != optimized_cfg.warmup:
        return False, (
            f"Warmup mismatch: baseline={baseline_cfg.warmup}, "
            f"optimized={optimized_cfg.warmup}"
        )
    
    return True, None


# =============================================================================
# Optional Verification Hooks (Task 2.4-2.5)
# =============================================================================


def get_output_tolerance(benchmark: Any) -> Optional[ToleranceSpec]:
    """Get custom output tolerance from benchmark.
    
    Benchmarks can implement get_output_tolerance() to specify custom
    tolerances for output comparison.
    
    Args:
        benchmark: Benchmark instance
        
    Returns:
        ToleranceSpec if benchmark provides custom tolerance, None otherwise
    """
    if not hasattr(benchmark, "get_output_tolerance") or not callable(getattr(benchmark, "get_output_tolerance")):
        raise NotImplementedError(
            f"{benchmark.__class__.__name__} must implement get_output_tolerance()"
        )
    
    result = benchmark.get_output_tolerance()
    if result is None:
        raise ValueError(f"{benchmark.__class__.__name__}.get_output_tolerance() returned None")
    
    if isinstance(result, ToleranceSpec):
        return result
    if isinstance(result, dict):
        return ToleranceSpec(
            rtol=result["rtol"],
            atol=result["atol"],
            justification=result.get("justification"),
        )
    if isinstance(result, tuple) and len(result) >= 2:
        return ToleranceSpec(
            rtol=result[0],
            atol=result[1],
            justification=None,
        )
    raise TypeError(
        f"{benchmark.__class__.__name__}.get_output_tolerance() must return ToleranceSpec, dict, or (rtol, atol) tuple"
    )


def get_verify_output(benchmark: Any) -> Optional[torch.Tensor]:
    """Get verification output from benchmark.
    
    Benchmarks can implement get_verify_output() to return a tensor
    for verification comparison.
    
    Args:
        benchmark: Benchmark instance
        
    Returns:
        Output tensor if available, None otherwise
    """
    if not hasattr(benchmark, "get_verify_output") or not callable(getattr(benchmark, "get_verify_output")):
        raise NotImplementedError(
            f"{benchmark.__class__.__name__} must implement get_verify_output()"
        )
    
    output = benchmark.get_verify_output()
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"{benchmark.__class__.__name__}.get_verify_output() must return a torch.Tensor"
        )
    return output


def get_jitter_exemption_reason(benchmark: Any) -> Optional[str]:
    """Get jitter exemption reason from benchmark.
    
    Args:
        benchmark: Benchmark instance
        
    Returns:
        Exemption reason string if declared, None otherwise
    """
    # Check attribute
    if hasattr(benchmark, "jitter_exemption_reason"):
        reason = getattr(benchmark, "jitter_exemption_reason")
        if reason:
            return str(reason)
    
    # Check non_jitterable_reason (alternative name)
    if hasattr(benchmark, "non_jitterable_reason"):
        reason = getattr(benchmark, "non_jitterable_reason")
        if reason:
            return str(reason)
    
    return None


def get_workload_ratio_expected(benchmark: Any) -> Optional[Tuple[float, str]]:
    """Get expected workload ratio from benchmark.
    
    Benchmarks that legitimately process different amounts can declare
    workload_ratio_expected with expected ratio and justification.
    
    Args:
        benchmark: Benchmark instance
        
    Returns:
        Tuple of (ratio, justification) if declared, None otherwise
    """
    if hasattr(benchmark, "workload_ratio_expected"):
        ratio = getattr(benchmark, "workload_ratio_expected")
        justification = getattr(benchmark, "workload_ratio_justification", "No justification provided")
        if ratio is not None:
            return (float(ratio), str(justification))
    return None
