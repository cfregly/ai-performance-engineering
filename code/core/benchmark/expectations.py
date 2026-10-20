"""Per-chapter expectation tracking for benchmark results."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# Schema version - increment for breaking changes
SCHEMA_VERSION = 2

# Floating-point tolerance for speedup ratio comparisons
# Used consistently across validation, storage, and property tests
SPEEDUP_TOLERANCE = 1e-6

EXPECTATION_FILENAME_TEMPLATE = "expectations_{hardware_key}.json"

# Tolerances to avoid flagging noise as regressions.
# GPU benchmarks typically have 10-30% variance between runs due to thermal
# throttling, memory state, kernel JIT, and first-call overhead.
RELATIVE_TOLERANCE = 0.25  # 25%
ABSOLUTE_TOLERANCE = 1e-5

# Speedup is derived from mean times. A mean more than this ratio above its
# p75, or above its median when the run recorded no p75, means a few slow
# outlier iterations dominate it. The derived speedup then does not describe
# the typical iteration.
#
# Why 1 + RELATIVE_TOLERANCE: a baseline mean inflated by a factor r inflates
# the stored speedup by r. A clean rerun then scores 1/r of the stored bar, and
# the regression check fails it once 1/r falls below 1 - RELATIVE_TOLERANCE,
# which happens above r = 1.33. At r <= 1.25 a clean rerun still passes.
#
# No flag overrides this limit, including --update-expectations. The harness
# stores and reports the mean, so a skewed mean is unusable evidence, not a
# slower result. When the slow iterations are a real periodic cost of the
# workload, such as a dataloader refill, change the benchmark so each timed
# call covers a whole period. Then every sample carries the same share of that
# cost, and the row can be refreshed.
MAX_MEAN_SKEW_RATIO = 1.0 + RELATIVE_TOLERANCE

# One wording for --update-expectations in the CLI, the MCP tools and the
# harness runner, so every surface states the same refusal.
UPDATE_EXPECTATIONS_HELP_TEXT = (
    "Write observed metrics into expectation files even when they regress or provenance differs. "
    "Timing whose mean exceeds its p75 (or its median when no p75 is recorded) by more than "
    f"{MAX_MEAN_SKEW_RATIO:g}x is still refused."
)

# Rejections caused by unusable run evidence rather than by a slower result.
# The harness marks a benchmark failed_expectation_invalid when its update is
# rejected for one of them, because the reported result comes from the same
# evidence. No flag stores an entry with a NON_OVERRIDABLE issue.
# accept_regressions does accept a dropped throughput metric
# (tracked_metric_missing) as a deliberate refresh.
NON_OVERRIDABLE_ISSUE_TYPES: FrozenSet[str] = frozenset(
    {"timing_outlier_skew", "invalid_timing_percentile", "missing_memory_metrics"}
)
INVALID_EVIDENCE_ISSUE_TYPES: FrozenSet[str] = NON_OVERRIDABLE_ISSUE_TYPES | {"tracked_metric_missing"}

# ThroughputStats fields with a fixed unit. custom_unit_per_s is compared
# separately because custom_unit_name sets its unit per benchmark.
THROUGHPUT_METRIC_DIRECTIONS: Dict[str, str] = {
    "requests_per_s": "higher",
    "tokens_per_s": "higher",
    "samples_per_s": "higher",
    "bytes_per_s": "higher",
    "goodput": "higher",
    "latency_ms": "lower",
}
THROUGHPUT_METRIC_PREFIXES: Tuple[str, ...] = ("baseline_throughput", "best_optimized_throughput")

# Metric direction hints. Extend as new metrics are tracked.
METRIC_DIRECTIONS: Dict[str, str] = {
    "best_speedup": "higher",
    "best_optimized_speedup": "higher",
    "baseline_time_ms": "lower",
    "best_optimized_time_ms": "lower",
    "baseline_memory_mb": "lower",
    "best_optimized_memory_mb": "lower",
    "best_memory_savings_ratio": "higher",
    "best_memory_savings_pct": "higher",
    **{
        f"{prefix}.{field_name}": direction
        for prefix in THROUGHPUT_METRIC_PREFIXES
        for field_name, direction in THROUGHPUT_METRIC_DIRECTIONS.items()
    },
    "baseline_median_ms": "lower",
    "baseline_p75_ms": "lower",
    "baseline_p90_ms": "lower",
    "best_optimized_median_ms": "lower",
    "best_optimized_p75_ms": "lower",
    "best_optimized_p90_ms": "lower",
    "baseline_custom.scenario_total_phase_ms": "lower",
    "best_optimized_custom.scenario_total_phase_ms": "lower",
}


# =============================================================================
# New Data Classes for Benchmark Expectations Integrity (Schema v2)
# =============================================================================


@dataclass
class RunProvenance:
    """Tracks the source of benchmark measurements for traceability."""

    git_commit: str
    hardware_key: str
    profile_name: str
    timestamp: str  # ISO format
    iterations: int
    warmup_iterations: int
    execution_environment: str = "unknown"
    validity_profile: Optional[str] = None
    dmi_product_name: Optional[str] = None
    # The --target-extra-arg values the harness passed to this target. None
    # means the run used the target's defaults. Other values time a different
    # workload or backend, so this field is compared like the fields above.
    target_override_argv: Optional[List[str]] = None

    def mismatch_fields(self, other: "RunProvenance") -> List[str]:
        """Return the provenance fields that materially differ."""
        mismatches: List[str] = []
        comparable_fields = [
            "git_commit",
            "hardware_key",
            "profile_name",
            "execution_environment",
            "validity_profile",
            "target_override_argv",
        ]
        for field_name in comparable_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                mismatches.append(field_name)
        return mismatches

    def matches(self, other: "RunProvenance") -> bool:
        """Check if two runs are from the same configuration (ignores timestamp)."""
        return not self.mismatch_fields(other)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        target_override_argv is written only when the run passed overrides, so
        rows from default runs keep their existing shape.
        """
        payload: Dict[str, Any] = {
            "git_commit": self.git_commit,
            "hardware_key": self.hardware_key,
            "profile_name": self.profile_name,
            "timestamp": self.timestamp,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "execution_environment": self.execution_environment,
            "validity_profile": self.validity_profile,
            "dmi_product_name": self.dmi_product_name,
        }
        if self.target_override_argv is not None:
            payload["target_override_argv"] = list(self.target_override_argv)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunProvenance":
        """Deserialize from dictionary."""
        target_override_argv = data.get("target_override_argv")
        if target_override_argv is not None and not (
            isinstance(target_override_argv, list)
            and all(isinstance(arg, str) for arg in target_override_argv)
        ):
            raise ValueError(
                f"provenance target_override_argv must be a list of strings, got {target_override_argv!r}"
            )
        return cls(
            git_commit=data.get("git_commit", ""),
            hardware_key=data.get("hardware_key", ""),
            profile_name=data.get("profile_name", ""),
            timestamp=data.get("timestamp", ""),
            iterations=data.get("iterations", 0),
            warmup_iterations=data.get("warmup_iterations", 0),
            execution_environment=data.get("execution_environment", "unknown"),
            validity_profile=data.get("validity_profile"),
            dmi_product_name=data.get("dmi_product_name"),
            target_override_argv=list(target_override_argv) if target_override_argv is not None else None,
        )


@dataclass
class ExpectationEntry:
    """Represents a single benchmark expectation with enforced consistency.

    Speedup is always computed from timing values - never stored independently.
    This ensures speedup can never drift from the underlying timing data.
    """

    example: str
    type: str  # "python" or "cuda"

    # Primary optimization goal ("speed" or "memory")
    optimization_goal: str

    # Timing metrics (source of truth)
    baseline_time_ms: float
    best_optimized_time_ms: float

    # Provenance
    provenance: RunProvenance

    # Optional memory metrics (for memory-goal benchmarks)
    baseline_memory_mb: Optional[float] = None
    best_optimized_memory_mb: Optional[float] = None

    # Optional extended metrics
    baseline_p75_ms: Optional[float] = None
    baseline_p90_ms: Optional[float] = None
    best_optimized_p75_ms: Optional[float] = None
    best_optimized_p90_ms: Optional[float] = None
    baseline_median_ms: Optional[float] = None
    best_optimized_median_ms: Optional[float] = None
    baseline_throughput: Optional[Dict[str, float]] = None
    best_optimized_throughput: Optional[Dict[str, float]] = None
    custom_metrics: Optional[Dict[str, Any]] = None

    # Metadata about best optimization
    best_optimization_name: Optional[str] = None
    best_optimization_file: Optional[str] = None
    best_optimization_technique: Optional[str] = None
    minimum_required_speedup: Optional[float] = None

    @property
    def best_speedup(self) -> float:
        """Compute speedup from timing values - this is derived, not stored."""
        if self.best_optimized_time_ms <= 0:
            return 1.0
        return self.baseline_time_ms / self.best_optimized_time_ms

    @property
    def best_memory_savings_ratio(self) -> Optional[float]:
        """Compute memory savings ratio (baseline / optimized). Higher is better."""
        if self.baseline_memory_mb is None or self.best_optimized_memory_mb is None:
            return None
        if self.best_optimized_memory_mb <= 0:
            return None
        return float(self.baseline_memory_mb) / float(self.best_optimized_memory_mb)

    @property
    def best_memory_savings_pct(self) -> Optional[float]:
        """Compute memory savings percentage (higher is better)."""
        ratio = self.best_memory_savings_ratio
        if ratio is None:
            return None
        if self.baseline_memory_mb is None or self.baseline_memory_mb <= 0:
            return None
        baseline_mb = float(self.baseline_memory_mb)
        optimized_mb = float(self.best_optimized_memory_mb) if self.best_optimized_memory_mb is not None else 0.0
        return ((baseline_mb - optimized_mb) / baseline_mb) * 100.0

    @property
    def primary_improvement(self) -> float:
        """Primary improvement ratio derived from the benchmark's optimization goal."""
        goal = (self.optimization_goal or "speed").strip().lower()
        if goal == "memory":
            ratio = self.best_memory_savings_ratio
            if ratio is None:
                raise RuntimeError(
                    "optimization_goal='memory' requires baseline_memory_mb and best_optimized_memory_mb"
                )
            return ratio
        return self.best_speedup

    @property
    def is_regression(self) -> bool:
        """Check if this represents a regression relative to the optimization goal."""
        return self.primary_improvement < 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Note: best_speedup is computed and stored for readability, but on load
        it should be validated against the timing ratio.
        """
        result: Dict[str, Any] = {
            "example": self.example,
            "type": self.type,
            "metrics": {
                "baseline_time_ms": self.baseline_time_ms,
                "best_optimized_time_ms": self.best_optimized_time_ms,
                # Speedup is derived from times - computed fresh on serialization
                "best_speedup": self.best_speedup,
                "best_optimized_speedup": self.best_speedup,  # Same value for compatibility
                "is_regression": self.is_regression,
            },
            "provenance": self.provenance.to_dict(),
            "metadata": {},
        }

        # Add optional memory metrics
        if self.baseline_memory_mb is not None:
            result["metrics"]["baseline_memory_mb"] = float(self.baseline_memory_mb)
        if self.best_optimized_memory_mb is not None:
            result["metrics"]["best_optimized_memory_mb"] = float(self.best_optimized_memory_mb)
        if self.best_memory_savings_ratio is not None:
            result["metrics"]["best_memory_savings_ratio"] = float(self.best_memory_savings_ratio)
        if self.best_memory_savings_pct is not None:
            result["metrics"]["best_memory_savings_pct"] = float(self.best_memory_savings_pct)

        # Add optional percentile metrics
        if self.baseline_p75_ms is not None:
            result["metrics"]["baseline_p75_ms"] = self.baseline_p75_ms
        if self.baseline_p90_ms is not None:
            result["metrics"]["baseline_p90_ms"] = self.baseline_p90_ms
        if self.best_optimized_p75_ms is not None:
            result["metrics"]["best_optimized_p75_ms"] = self.best_optimized_p75_ms
        if self.best_optimized_p90_ms is not None:
            result["metrics"]["best_optimized_p90_ms"] = self.best_optimized_p90_ms
        if self.baseline_median_ms is not None:
            result["metrics"]["baseline_median_ms"] = self.baseline_median_ms
        if self.best_optimized_median_ms is not None:
            result["metrics"]["best_optimized_median_ms"] = self.best_optimized_median_ms

        # Add throughput metrics
        if self.baseline_throughput:
            for key, value in self.baseline_throughput.items():
                result["metrics"][f"baseline_throughput.{key}"] = value
        if self.best_optimized_throughput:
            for key, value in self.best_optimized_throughput.items():
                result["metrics"][f"best_optimized_throughput.{key}"] = value

        # Add custom metrics (stored separately to not pollute timing-based speedup)
        if self.custom_metrics:
            result["custom_metrics"] = self.custom_metrics

        # Add metadata
        result["metadata"]["optimization_goal"] = self.optimization_goal
        if self.best_optimization_name:
            result["metadata"]["best_optimization"] = self.best_optimization_name
        if self.best_optimization_file:
            result["metadata"]["best_optimization_file"] = self.best_optimization_file
        if self.best_optimization_technique:
            result["metadata"]["best_optimization_technique"] = self.best_optimization_technique
        if self.minimum_required_speedup is not None:
            result["metadata"]["minimum_required_speedup"] = float(self.minimum_required_speedup)
        # Store speedup in metadata for compatibility (derived from times)
        result["metadata"]["best_optimization_speedup"] = self.best_speedup
        result["metadata"]["best_optimization_time_ms"] = self.best_optimized_time_ms
        if self.best_optimized_memory_mb is not None:
            result["metadata"]["best_optimization_memory_mb"] = float(self.best_optimized_memory_mb)
        if self.best_memory_savings_ratio is not None:
            result["metadata"]["best_memory_savings_ratio"] = float(self.best_memory_savings_ratio)
        if self.best_memory_savings_pct is not None:
            result["metadata"]["best_memory_savings_pct"] = float(self.best_memory_savings_pct)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectationEntry":
        """Deserialize from dictionary."""
        metrics = data.get("metrics", {})
        metadata = data.get("metadata", {})
        provenance_data = data.get("provenance", {})

        # Extract throughput metrics
        baseline_throughput: Dict[str, float] = {}
        best_optimized_throughput: Dict[str, float] = {}
        for key, value in metrics.items():
            if key.startswith("baseline_throughput."):
                baseline_throughput[key.replace("baseline_throughput.", "")] = value
            elif key.startswith("best_optimized_throughput."):
                best_optimized_throughput[key.replace("best_optimized_throughput.", "")] = value

        return cls(
            example=data.get("example", ""),
            type=data.get("type", "python"),
            optimization_goal=metadata.get("optimization_goal", "speed"),
            baseline_time_ms=metrics.get("baseline_time_ms", 0.0),
            best_optimized_time_ms=metrics.get("best_optimized_time_ms", 0.0),
            baseline_memory_mb=metrics.get("baseline_memory_mb"),
            best_optimized_memory_mb=metrics.get("best_optimized_memory_mb"),
            provenance=RunProvenance.from_dict(provenance_data),
            baseline_p75_ms=metrics.get("baseline_p75_ms"),
            baseline_p90_ms=metrics.get("baseline_p90_ms"),
            best_optimized_p75_ms=metrics.get("best_optimized_p75_ms"),
            best_optimized_p90_ms=metrics.get("best_optimized_p90_ms"),
            baseline_median_ms=metrics.get("baseline_median_ms"),
            best_optimized_median_ms=metrics.get("best_optimized_median_ms"),
            baseline_throughput=baseline_throughput if baseline_throughput else None,
            best_optimized_throughput=best_optimized_throughput if best_optimized_throughput else None,
            custom_metrics=data.get("custom_metrics"),
            best_optimization_name=metadata.get("best_optimization"),
            best_optimization_file=metadata.get("best_optimization_file"),
            best_optimization_technique=metadata.get("best_optimization_technique"),
            minimum_required_speedup=metadata.get("minimum_required_speedup"),
        )


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in an expectation entry."""

    example_key: str
    issue_type: str  # 'speedup_mismatch', 'metadata_drift', 'masked_regression', 'missing_provenance'
    message: str
    stored_value: Any
    expected_value: Any
    delta_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "example_key": self.example_key,
            "issue_type": self.issue_type,
            "message": self.message,
            "stored_value": self.stored_value,
            "expected_value": self.expected_value,
            "delta_pct": self.delta_pct,
        }


@dataclass
class ValidationReport:
    """Summary of validation results across expectation entries."""

    issues: List[ValidationIssue] = field(default_factory=list)
    total_entries: int = 0
    valid_entries: int = 0

    @property
    def has_issues(self) -> bool:
        """Check if any validation issues were found."""
        return len(self.issues) > 0

    @property
    def invalid_entries(self) -> int:
        """Number of entries with issues."""
        return self.total_entries - self.valid_entries

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_entries": self.total_entries,
            "valid_entries": self.valid_entries,
            "invalid_entries": self.invalid_entries,
            "has_issues": self.has_issues,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class UpdateResult:
    """Result of updating an expectation entry."""

    status: str  # 'updated', 'improved', 'regressed', 'rejected', 'unchanged'
    message: str
    entry: Optional[ExpectationEntry] = None
    validation_issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status,
            "message": self.message,
            "entry": self.entry.to_dict() if self.entry else None,
            "validation_issues": [issue.to_dict() for issue in self.validation_issues],
        }


# =============================================================================
# Helper Functions
# =============================================================================


def select_best_optimization(
    optimizations: Optional[List[Dict[str, Any]]],
    *,
    goal: str = "speed",
) -> Optional[Dict[str, Any]]:
    """Select the best optimization from a list based on the optimization goal.

    This is the SINGLE SOURCE OF TRUTH for selecting the best optimization.
    Used by both metrics collection and metadata building to ensure consistency.

    Args:
        optimizations: List of optimization result dicts. Speed selections use 'speedup'.
        goal: Primary optimization goal ('speed' or 'memory').

    Returns:
        The optimization dict with the highest goal-aligned score, or None if
        no successful optimizations remain after goal filtering.
    """
    goal_norm = (goal or "speed").strip().lower()

    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")

    for opt in optimizations or []:
        # Only consider successful optimizations
        if opt.get("status") != "succeeded":
            continue

        opt_goal = str(opt.get("optimization_goal") or "").strip().lower()
        if goal_norm != "comparison" and opt_goal == "comparison":
            continue

        if goal_norm == "memory":
            memory_mb = opt.get("memory_mb")
            if memory_mb is None:
                continue
            try:
                score = -float(memory_mb)  # lower memory is better
            except (TypeError, ValueError):
                continue
        else:
            # Default: speed
            try:
                score = float(opt.get("speedup") or 0.0)
            except (TypeError, ValueError):
                score = 0.0

        if score > best_score:
            best = opt
            best_score = score

    return best


def compute_speedup(baseline_time_ms: float, optimized_time_ms: float) -> float:
    """Compute speedup ratio from timing values.

    This is the canonical speedup computation used throughout the system.
    Speedup = baseline_time / optimized_time (higher is better).

    Args:
        baseline_time_ms: Baseline execution time in milliseconds.
        optimized_time_ms: Optimized execution time in milliseconds.

    Returns:
        Speedup ratio. Returns 1.0 if optimized_time_ms is <= 0.
    """
    if optimized_time_ms <= 0:
        return 1.0
    return baseline_time_ms / optimized_time_ms


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def detect_expectation_key() -> str:
    """Return a hardware/environment key for selecting expectation files."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name
            gpu_slug = _slugify(re.sub(r"^nvidia_", "", _slugify(name)))
            count = torch.cuda.device_count()
            prefix = f"{count}x_" if count > 1 else ""
            return _slugify(f"{prefix}{gpu_slug}")
    except RuntimeError:
        pass  # CUDA not initialized

    return "unknown"


def _tolerance(expected: float) -> float:
    return max(abs(expected) * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def _format_delta(observed: Optional[float], expected: Optional[float]) -> Dict[str, Optional[float]]:
    if observed is None or expected is None:
        return {"delta": None, "delta_pct": None}
    delta = observed - expected
    if expected == 0:
        delta_pct = math.inf if delta > 0 else -math.inf if delta < 0 else 0.0
    else:
        delta_pct = (delta / expected) * 100.0
    return {"delta": delta, "delta_pct": delta_pct}


def _compare_metric(metric: str, direction: Optional[str], observed: Optional[float], expected: Optional[float]) -> Dict[str, Any]:
    comparison = {
        "metric": metric,
        "direction": direction,
        "expected": expected,
        "observed": observed,
        "status": "not_tracked",
    }
    comparison.update(_format_delta(observed, expected))

    if observed is None:
        comparison["status"] = "missing"
        return comparison
    if direction is None:
        comparison["status"] = "recorded"
        return comparison
    if expected is None:
        comparison["status"] = "new"
        return comparison

    tol = _tolerance(expected)
    if direction == "higher":
        if observed > expected + tol:
            comparison["status"] = "improved"
        elif observed < expected - tol:
            comparison["status"] = "regressed"
        else:
            comparison["status"] = "met"
    else:  # lower is better
        if observed < expected - tol:
            comparison["status"] = "improved"
        elif observed > expected + tol:
            comparison["status"] = "regressed"
        else:
            comparison["status"] = "met"
    return comparison


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def _timing_outlier_issues(example_key: str, metrics: Dict[str, Any]) -> List[ValidationIssue]:
    """Return issues for mean times dominated by slow outlier iterations.

    ``metrics`` uses the stored metric names. Each side's mean
    (``baseline_time_ms`` or ``best_optimized_time_ms``) is compared with its
    ``*_p75_ms``, or with its ``*_median_ms`` when the run recorded no p75.
    That happens when a benchmark's configured percentiles omit 75. A recorded
    p75 or median that is not a positive time is reported, because the harness
    must not invent one.

    Limits: a side with neither value cannot be checked. That covers rows
    written before medians were recorded. Torchrun timing is a single sample
    (one rank-0 iteration mean or one process wall time), so its mean equals
    its median and outliers inside that sample cannot be seen.
    """
    issues: List[ValidationIssue] = []
    for side in ("baseline", "best_optimized"):
        mean_key = f"{side}_time_ms"
        references: List[Tuple[str, float]] = []
        side_invalid = False
        for reference_key in (f"{side}_p75_ms", f"{side}_median_ms"):
            value = metrics.get(reference_key)
            if value is None:
                continue
            value_ms = _finite_number(value)
            if value_ms is None or value_ms <= 0:
                side_invalid = True
                issues.append(
                    ValidationIssue(
                        example_key=example_key,
                        issue_type="invalid_timing_percentile",
                        message=f"{reference_key} must be a positive finite time in ms, got {value!r}",
                        stored_value=value,
                        expected_value="positive finite milliseconds",
                    )
                )
                continue
            references.append((reference_key, value_ms))
        if side_invalid or not references:
            continue
        mean_ms = _finite_number(metrics.get(mean_key))
        if mean_ms is None or mean_ms <= 0:
            continue
        reference_key, reference_ms = references[0]
        ratio = mean_ms / reference_ms
        if ratio > MAX_MEAN_SKEW_RATIO:
            issues.append(
                ValidationIssue(
                    example_key=example_key,
                    issue_type="timing_outlier_skew",
                    message=(
                        f"{mean_key} {mean_ms:.6g} ms is {ratio:.2f}x {reference_key} {reference_ms:.6g} ms "
                        f"(limit {MAX_MEAN_SKEW_RATIO:.2f}x). Slow outlier iterations dominate the "
                        "mean, so the speedup derived from it is not representative. Remove the "
                        "outlier source and rerun. If the slow iterations are a periodic cost of the "
                        "workload, make each timed call cover a whole period."
                    ),
                    stored_value=mean_ms,
                    expected_value=reference_ms * MAX_MEAN_SKEW_RATIO,
                    delta_pct=(ratio - 1.0) * 100.0,
                )
            )
    return issues


def _entry_timing_metrics(entry: "ExpectationEntry") -> Dict[str, Any]:
    return {
        "baseline_time_ms": entry.baseline_time_ms,
        "baseline_p75_ms": entry.baseline_p75_ms,
        "baseline_median_ms": entry.baseline_median_ms,
        "best_optimized_time_ms": entry.best_optimized_time_ms,
        "best_optimized_p75_ms": entry.best_optimized_p75_ms,
        "best_optimized_median_ms": entry.best_optimized_median_ms,
    }


def _throughput_measurement(field_name: str, value: Any) -> Optional[float]:
    """Return ``value`` when it can be a measured throughput field, else None.

    A completed run cannot have a zero rate or a zero latency, so a value at or
    below zero comes from a workload that declared no units of that kind.
    goodput is a 0-1 ratio, so a goodput of zero is a real measurement.
    """
    number = _finite_number(value)
    if number is None:
        return None
    if field_name == "goodput":
        return number if number >= 0 else None
    return number if number > 0 else None


def _throughput_field_issues(
    example_key: str,
    prefix: str,
    field_name: str,
    direction: str,
    stored: Any,
    observed: Any,
) -> List[ValidationIssue]:
    metric = f"{prefix}.{field_name}"
    stored_value = _throughput_measurement(field_name, stored)
    if stored_value is None:
        return []
    observed_value = _throughput_measurement(field_name, observed)
    if observed_value is None:
        return [
            ValidationIssue(
                example_key=example_key,
                issue_type="tracked_metric_missing",
                message=f"{metric} is tracked at {stored_value:.6g} but this run reported {observed!r}",
                stored_value=stored_value,
                expected_value=observed,
            )
        ]
    comparison = _compare_metric(metric, direction, observed_value, stored_value)
    if comparison["status"] != "regressed":
        return []
    return [
        ValidationIssue(
            example_key=example_key,
            issue_type="regression",
            message=(
                f"{metric} new {observed_value:.6g} vs existing {stored_value:.6g} "
                f"({direction} is better)"
            ),
            stored_value=stored_value,
            expected_value=observed_value,
            delta_pct=comparison["delta_pct"],
        )
    ]


def _throughput_issues(
    example_key: str,
    existing: "ExpectationEntry",
    entry: "ExpectationEntry",
) -> List[ValidationIssue]:
    """Compare every tracked throughput field of ``entry`` against ``existing``.

    The primary score is a ratio, so it hides a slowdown shared by baseline and
    optimized and rises when only the baseline slows down. These absolute
    checks catch both cases.
    """
    issues: List[ValidationIssue] = []
    sides = (
        ("baseline_throughput", existing.baseline_throughput, entry.baseline_throughput),
        ("best_optimized_throughput", existing.best_optimized_throughput, entry.best_optimized_throughput),
    )
    for prefix, stored_payload, new_payload in sides:
        stored_payload = stored_payload or {}
        new_payload = new_payload or {}
        for field_name, direction in THROUGHPUT_METRIC_DIRECTIONS.items():
            issues.extend(
                _throughput_field_issues(
                    example_key,
                    prefix,
                    field_name,
                    direction,
                    stored_payload.get(field_name),
                    new_payload.get(field_name),
                )
            )
        custom_metric = f"{prefix}.custom_unit_per_s"
        stored_custom = _throughput_measurement("custom_unit_per_s", stored_payload.get("custom_unit_per_s"))
        stored_unit = stored_payload.get("custom_unit_name")
        new_unit = new_payload.get("custom_unit_name")
        if stored_custom is not None and new_unit != stored_unit:
            issues.append(
                ValidationIssue(
                    example_key=example_key,
                    issue_type="tracked_metric_missing",
                    message=(
                        f"{custom_metric} is tracked in {stored_unit!r} units but this run "
                        f"reported {new_unit!r} units"
                    ),
                    stored_value=stored_unit,
                    expected_value=new_unit,
                )
            )
            continue
        issues.extend(
            _throughput_field_issues(
                example_key,
                prefix,
                "custom_unit_per_s",
                "higher",
                stored_payload.get("custom_unit_per_s"),
                new_payload.get("custom_unit_per_s"),
            )
        )
    return issues


@dataclass
class ExpectationEvaluation:
    example_key: str
    hardware_key: str
    expectation_exists: bool
    regressed: bool
    comparisons: List[Dict[str, Any]]
    regressions: List[Dict[str, Any]]
    improvements: List[Dict[str, Any]]
    updated_metrics: List[str] = field(default_factory=list)
    expectation_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_key": self.example_key,
            "hardware_key": self.hardware_key,
            "expectation_exists": self.expectation_exists,
            "regressed": self.regressed,
            "comparisons": self.comparisons,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "updated_metrics": self.updated_metrics,
            "expectation_path": str(self.expectation_path) if self.expectation_path else None,
        }


class ExpectationsStore:
    """Maintain expectation files per chapter and hardware target.

    Schema v2 features:
    - Atomic updates: all metrics updated together, not per-metric
    - Derived speedups: speedup always computed from timing values
    - Provenance tracking: git commit, hardware, profile tracked per entry
    - Validation on write: consistency checks before saving
    """

    def __init__(
        self,
        chapter_dir: Path,
        hardware_key: str,
        *,
        accept_regressions: bool = False,
        allow_mixed_provenance: bool = False,
        validate_on_load: bool = False,
    ) -> None:
        self.chapter_dir = chapter_dir
        self.hardware_key = hardware_key
        self.path = chapter_dir / EXPECTATION_FILENAME_TEMPLATE.format(hardware_key=hardware_key)
        self._data = self._load()
        self._changed = False
        self._accept_regressions = accept_regressions
        self._allow_mixed_provenance = allow_mixed_provenance

        # Optionally validate on load to catch drift early
        if validate_on_load and self.path.exists():
            report = self.validate_all()
            if report.has_issues:
                import warnings
                warnings.warn(
                    f"Expectation file {self.path} has {len(report.issues)} validation issues. "
                    f"Run 'python -m core.benchmark.validate_expectations {self.path} --verbose' to list them. "
                    "--fix recomputes derived ratios and provenance only. Timing outlier issues need a fresh run.",
                    UserWarning,
                    stacklevel=2,
                )

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("hardware_key", self.hardware_key)
        data.setdefault("examples", {})
        return data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def evaluate(self, example_key: str, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None) -> Optional[ExpectationEvaluation]:
        """Compare metrics against expectations and update stored bests if improved.

        INTEGRITY SAFEGUARD: Speedup values are re-derived from timing values
        before storing to ensure consistency. This prevents stale speedups from
        being written even if the caller passes incorrect values.
        """
        if not metrics:
            return None

        # SAFEGUARD: Re-derive speedup from timing values to ensure consistency
        # This is the "belt-and-suspenders" approach - even if the caller computes
        # speedup correctly, we re-derive it here to guarantee data integrity.
        metrics = dict(metrics)  # Don't mutate caller's dict
        baseline_time = metrics.get("baseline_time_ms")
        optimized_time = metrics.get("best_optimized_time_ms")
        if baseline_time and optimized_time and optimized_time > 0:
            derived_speedup = compute_speedup(baseline_time, optimized_time)
            metrics["best_speedup"] = derived_speedup
            metrics["best_optimized_speedup"] = derived_speedup

        metadata = metadata or {}
        examples = self._data.setdefault("examples", {})
        entry = examples.get(example_key)
        expectation_exists = bool(entry and entry.get("metrics"))

        if entry is None:
            entry = {
                "example": metadata.get("example", example_key),
                "type": metadata.get("type", "python"),
                "metrics": {},
                "metadata": {},
            }
            examples[example_key] = entry

        stored_metrics: Dict[str, float] = entry.setdefault("metrics", {})
        comparisons: List[Dict[str, Any]] = []
        regressions: List[Dict[str, Any]] = []
        improvements: List[Dict[str, Any]] = []
        updated_metrics: List[str] = []

        for metric_name in sorted(metrics.keys()):
            if metric_name not in METRIC_DIRECTIONS:
                continue
            observed = metrics[metric_name]
            direction = METRIC_DIRECTIONS.get(metric_name)
            expected = stored_metrics.get(metric_name)
            comp = _compare_metric(metric_name, direction, observed, expected)
            comparisons.append(comp)
            if comp["status"] == "regressed":
                if self._accept_regressions:
                    stored_metrics[metric_name] = observed
                    comp["status"] = "updated"
                    improvements.append(comp)
                    updated_metrics.append(metric_name)
                else:
                    regressions.append(comp)
            elif comp["status"] in {"improved", "new"}:
                stored_metrics[metric_name] = observed
                improvements.append(comp)
                updated_metrics.append(metric_name)

        # Metrics tracked previously but not emitted in this run
        for metric_name in sorted(stored_metrics.keys()):
            if metric_name not in metrics:
                direction = METRIC_DIRECTIONS.get(metric_name)
                comp = {
                    "metric": metric_name,
                    "direction": direction,
                    "expected": stored_metrics.get(metric_name),
                    "observed": None,
                    "status": "not_reported",
                    "delta": None,
                    "delta_pct": None,
                }
                comparisons.append(comp)

        regressed = bool(regressions)

        if not regressed and updated_metrics:
            self._update_metadata(entry, metadata)

        if not expectation_exists and not regressed and not updated_metrics:
            # First run with metrics equal to defaults: still treat as update.
            self._update_metadata(entry, metadata)

        if updated_metrics:
            self._changed = True

        return ExpectationEvaluation(
            example_key=example_key,
            hardware_key=self.hardware_key,
            expectation_exists=expectation_exists,
            regressed=regressed,
            comparisons=sorted(comparisons, key=lambda c: c["metric"]),
            regressions=regressions,
            improvements=improvements,
            updated_metrics=updated_metrics,
            expectation_path=self.path if self.path else None,
        )

    def update_entry(self, example_key: str, entry: ExpectationEntry) -> UpdateResult:
        """Atomically update an entry with enforced consistency.

        This is the new schema v2 update method that ensures:
        - All metrics are updated together (atomic)
        - Speedup is always derived from timing values
        - Timing is never stored when slow outliers dominate its mean
        - Provenance is tracked and validated
        - Mixed-provenance updates are rejected unless explicitly allowed
        - Speedup and throughput regressions and dropped throughput metrics
          are rejected unless regressions are accepted

        A rejection lists every cause found, so a regression is still reported
        when the same run also has outlier timing.

        Args:
            example_key: Key identifying the benchmark example (e.g., "matmul_cuda")
            entry: The ExpectationEntry to store

        Returns:
            UpdateResult with status and any validation issues
        """
        examples = self._data.setdefault("examples", {})
        existing = examples.get(example_key)

        goal = (entry.optimization_goal or "speed").strip().lower()

        # accept_regressions does not override this check. Outlier timing is
        # invalid evidence, not a slower result.
        timing_issues = _timing_outlier_issues(example_key, _entry_timing_metrics(entry))

        def reject(reasons: List[str], issues: List[ValidationIssue]) -> UpdateResult:
            if timing_issues:
                reasons = reasons + [
                    "Timing rejected (accept_regressions does not override this): "
                    + " ".join(issue.message for issue in timing_issues)
                ]
            return UpdateResult(
                status="rejected",
                message=" ".join(reasons),
                entry=entry,
                validation_issues=issues + timing_issues,
            )

        # Check provenance consistency if entry already exists
        if existing and not self._allow_mixed_provenance:
            existing_provenance = existing.get("provenance")
            if existing_provenance:
                existing_prov = RunProvenance.from_dict(existing_provenance)
                mismatch_fields = entry.provenance.mismatch_fields(existing_prov)
                if mismatch_fields:
                    mismatches: List[str] = []
                    for field_name in mismatch_fields:
                        mismatches.append(
                            f"{field_name}: new='{getattr(entry.provenance, field_name)}' "
                            f"stored='{getattr(existing_prov, field_name)}'"
                        )
                    mismatch_summary = ", ".join(mismatches) if mismatches else "unknown mismatch"
                    return reject(
                        [
                            f"Provenance mismatch: {mismatch_summary}. "
                            f"Use allow_mixed_provenance=True (or --allow-mixed-provenance / --update-expectations) to override."
                        ],
                        [
                            ValidationIssue(
                                example_key=example_key,
                                issue_type="provenance_mismatch",
                                message="Mixed provenance update rejected",
                                stored_value=existing_prov.to_dict(),
                                expected_value=entry.provenance.to_dict(),
                            )
                        ],
                    )

        # Determine the primary score based on optimization goal
        if goal == "memory":
            metric_label = "best_memory_savings_ratio"
            new_score = entry.best_memory_savings_ratio
            if new_score is None:
                return reject(
                    ["Memory-goal benchmark missing memory metrics (baseline_memory_mb/best_optimized_memory_mb)."],
                    [
                        ValidationIssue(
                            example_key=example_key,
                            issue_type="missing_memory_metrics",
                            message="Missing memory metrics for memory-goal benchmark",
                            stored_value=None,
                            expected_value="baseline_memory_mb and best_optimized_memory_mb",
                        )
                    ],
                )
        else:
            metric_label = "best_speedup"
            new_score = entry.best_speedup

        # Determine if this is an improvement, regression, or unchanged
        status = "updated"
        regression_reasons: List[str] = []
        regression_issues: List[ValidationIssue] = []
        throughput_problems: List[ValidationIssue] = []
        if existing:
            existing_entry = ExpectationEntry.from_dict(existing)
            if goal == "memory":
                old_score = existing_entry.best_memory_savings_ratio
                if old_score is None:
                    old_score = 0.0
            else:
                old_score = existing_entry.best_speedup

            if new_score > old_score * (1 + RELATIVE_TOLERANCE):
                status = "improved"
            elif new_score < old_score * (1 - RELATIVE_TOLERANCE):
                status = "regressed"
                regression_reasons.append(
                    f"Regression detected ({metric_label}): new {new_score:.3f} < existing {old_score:.3f}."
                )
                regression_issues.append(
                    ValidationIssue(
                        example_key=example_key,
                        issue_type="regression",
                        message="Performance regression rejected",
                        stored_value=old_score,
                        expected_value=new_score,
                        delta_pct=((new_score - old_score) / old_score) * 100 if old_score else None,
                    )
                )
            else:
                # Within tolerance - check if timing values changed significantly
                existing_metrics = existing.get("metrics", {})
                old_baseline = existing_metrics.get("baseline_time_ms", 0.0)
                old_optimized = existing_metrics.get("best_optimized_time_ms", 0.0)
                old_baseline_mem = existing_metrics.get("baseline_memory_mb")
                old_opt_mem = existing_metrics.get("best_optimized_memory_mb")
                if (
                    abs(entry.baseline_time_ms - old_baseline) < SPEEDUP_TOLERANCE
                    and abs(entry.best_optimized_time_ms - old_optimized) < SPEEDUP_TOLERANCE
                    and (
                        goal != "memory"
                        or (
                            entry.baseline_memory_mb is not None
                            and entry.best_optimized_memory_mb is not None
                            and old_baseline_mem is not None
                            and old_opt_mem is not None
                            and abs(float(entry.baseline_memory_mb) - float(old_baseline_mem)) < SPEEDUP_TOLERANCE
                            and abs(float(entry.best_optimized_memory_mb) - float(old_opt_mem)) < SPEEDUP_TOLERANCE
                        )
                    )
                ):
                    status = "unchanged"

            throughput_problems = _throughput_issues(example_key, existing_entry, entry)
            if throughput_problems:
                details = ", ".join(issue.message for issue in throughput_problems)
                throughput_regressed = any(issue.issue_type == "regression" for issue in throughput_problems)
                heading = "Regression detected (throughput)" if throughput_regressed else "Tracked throughput missing"
                regression_reasons.append(f"{heading}: {details}.")
                regression_issues.extend(throughput_problems)
                if throughput_regressed:
                    status = "regressed"

        if regression_issues and not self._accept_regressions:
            return reject(
                regression_reasons + ["Use accept_regressions=True to override."],
                regression_issues,
            )
        if timing_issues:
            return reject([], [])

        # Serialize entry to dict format (speedup is computed fresh here)
        entry_dict = entry.to_dict()

        # Store the entry atomically
        examples[example_key] = entry_dict
        self._changed = True

        message = f"Entry {example_key} {status} ({goal}) score {float(new_score):.3f}"
        if throughput_problems:
            message += ". Accepted throughput changes: " + ", ".join(
                issue.message for issue in throughput_problems
            )
        return UpdateResult(
            status=status,
            message=message,
            entry=entry,
            validation_issues=throughput_problems,
        )

    def get_entry(self, example_key: str) -> Optional[ExpectationEntry]:
        """Retrieve an entry as an ExpectationEntry object.

        Args:
            example_key: Key identifying the benchmark example

        Returns:
            ExpectationEntry if found, None otherwise
        """
        examples = self._data.get("examples", {})
        entry_dict = examples.get(example_key)
        if entry_dict is None:
            return None
        return ExpectationEntry.from_dict(entry_dict)

    def validate_entry(self, example_key: str, entry_dict: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single entry for consistency issues.

        Checks for:
        - Speedup mismatch: stored speedup differs from computed ratio
        - Masked regression: best_speedup=1.0 but actual ratio < 1.0
        - Metadata drift: metadata speedup differs from metrics speedup
        - Missing provenance: required provenance fields are missing
        - Timing outliers: a stored mean exceeds its p75 (or its median when
          no p75 is stored) by more than MAX_MEAN_SKEW_RATIO, or a stored p75
          or median is not a positive time

        Args:
            example_key: Key identifying the entry
            entry_dict: The entry dictionary to validate

        Returns:
            List of ValidationIssue objects for any problems found
        """
        issues: List[ValidationIssue] = []
        metrics = entry_dict.get("metrics", {})
        metadata = entry_dict.get("metadata", {})
        provenance = entry_dict.get("provenance", {})

        # Get timing values
        baseline_time = metrics.get("baseline_time_ms", 0.0)
        optimized_time = metrics.get("best_optimized_time_ms", 0.0)
        stored_speedup = metrics.get("best_speedup")
        stored_optimized_speedup = metrics.get("best_optimized_speedup")

        # Check 1: Speedup-timing consistency
        if baseline_time > 0 and optimized_time > 0:
            computed_speedup = compute_speedup(baseline_time, optimized_time)

            if stored_speedup is not None:
                delta = abs(stored_speedup - computed_speedup)
                if delta > SPEEDUP_TOLERANCE:
                    delta_pct = (delta / computed_speedup) * 100 if computed_speedup else 0
                    issues.append(
                        ValidationIssue(
                            example_key=example_key,
                            issue_type="speedup_mismatch",
                            message=(
                                f"Stored best_speedup {stored_speedup:.6f} differs from "
                                f"computed ratio {computed_speedup:.6f}"
                            ),
                            stored_value=stored_speedup,
                            expected_value=computed_speedup,
                            delta_pct=delta_pct,
                        )
                    )

            # Check for masked regression (best_speedup=1.0 but ratio<1.0)
            if stored_speedup is not None and abs(stored_speedup - 1.0) < SPEEDUP_TOLERANCE:
                if computed_speedup < 1.0 - SPEEDUP_TOLERANCE:
                    issues.append(
                        ValidationIssue(
                            example_key=example_key,
                            issue_type="masked_regression",
                            message=(
                                f"Stored best_speedup=1.0 but computed ratio is "
                                f"{computed_speedup:.6f} (regression hidden)"
                            ),
                            stored_value=1.0,
                            expected_value=computed_speedup,
                            delta_pct=((1.0 - computed_speedup) / computed_speedup) * 100
                            if computed_speedup
                            else None,
                        )
                    )

        # Check 2: best_optimized_speedup consistency (if different from best_speedup)
        if (
            stored_speedup is not None
            and stored_optimized_speedup is not None
            and abs(stored_speedup - stored_optimized_speedup) > SPEEDUP_TOLERANCE
        ):
            issues.append(
                ValidationIssue(
                    example_key=example_key,
                    issue_type="speedup_inconsistency",
                    message=(
                        f"best_speedup ({stored_speedup:.6f}) differs from "
                        f"best_optimized_speedup ({stored_optimized_speedup:.6f})"
                    ),
                    stored_value=stored_speedup,
                    expected_value=stored_optimized_speedup,
                )
            )

        # Check 3: Metadata speedup consistency
        metadata_speedup = metadata.get("best_optimization_speedup")
        if stored_speedup is not None and metadata_speedup is not None:
            if abs(stored_speedup - metadata_speedup) > SPEEDUP_TOLERANCE:
                issues.append(
                    ValidationIssue(
                        example_key=example_key,
                        issue_type="metadata_drift",
                        message=(
                            f"Metrics best_speedup ({stored_speedup:.6f}) differs from "
                            f"metadata best_optimization_speedup ({metadata_speedup:.6f})"
                        ),
                        stored_value=metadata_speedup,
                        expected_value=stored_speedup,
                    )
                )

        # Check 3b: Memory-savings consistency (when memory metrics are present)
        baseline_mem = metrics.get("baseline_memory_mb")
        optimized_mem = metrics.get("best_optimized_memory_mb")
        stored_mem_ratio = metrics.get("best_memory_savings_ratio")
        stored_mem_pct = metrics.get("best_memory_savings_pct")
        if isinstance(baseline_mem, (int, float)) and isinstance(optimized_mem, (int, float)):
            if baseline_mem > 0 and optimized_mem > 0:
                computed_ratio = float(baseline_mem) / float(optimized_mem)
                computed_pct = ((float(baseline_mem) - float(optimized_mem)) / float(baseline_mem)) * 100.0

                if stored_mem_ratio is not None and abs(float(stored_mem_ratio) - computed_ratio) > SPEEDUP_TOLERANCE:
                    stored_mem_ratio_f = float(stored_mem_ratio)
                    issues.append(
                        ValidationIssue(
                            example_key=example_key,
                            issue_type="memory_savings_mismatch",
                            message=(
                                f"Stored best_memory_savings_ratio {stored_mem_ratio_f:.6f} differs from "
                                f"computed ratio {computed_ratio:.6f}"
                            ),
                            stored_value=stored_mem_ratio,
                            expected_value=computed_ratio,
                        )
                    )

                if stored_mem_pct is not None and abs(float(stored_mem_pct) - computed_pct) > SPEEDUP_TOLERANCE:
                    stored_mem_pct_f = float(stored_mem_pct)
                    issues.append(
                        ValidationIssue(
                            example_key=example_key,
                            issue_type="memory_savings_pct_mismatch",
                            message=(
                                f"Stored best_memory_savings_pct {stored_mem_pct_f:.6f} differs from "
                                f"computed pct {computed_pct:.6f}"
                            ),
                            stored_value=stored_mem_pct,
                            expected_value=computed_pct,
                        )
                    )

                metadata_ratio = metadata.get("best_memory_savings_ratio")
                if stored_mem_ratio is not None and metadata_ratio is not None:
                    stored_mem_ratio_f = float(stored_mem_ratio)
                    metadata_ratio_f = float(metadata_ratio)
                    if abs(stored_mem_ratio_f - metadata_ratio_f) > SPEEDUP_TOLERANCE:
                        issues.append(
                            ValidationIssue(
                                example_key=example_key,
                                issue_type="metadata_drift_memory",
                                message=(
                                    f"Metrics best_memory_savings_ratio ({stored_mem_ratio_f:.6f}) differs from "
                                    f"metadata best_memory_savings_ratio ({metadata_ratio_f:.6f})"
                                ),
                                stored_value=metadata_ratio,
                                expected_value=stored_mem_ratio,
                            )
                        )

        # Check 4: Provenance completeness (for schema v2 entries)
        schema_version = self._data.get("schema_version", 1)
        if schema_version >= 2:
            required_prov_fields = ["git_commit", "hardware_key", "profile_name", "timestamp"]
            missing = [f for f in required_prov_fields if not provenance.get(f)]
            if missing:
                issues.append(
                    ValidationIssue(
                        example_key=example_key,
                        issue_type="missing_provenance",
                        message=f"Missing provenance fields: {', '.join(missing)}",
                        stored_value=list(provenance.keys()),
                        expected_value=required_prov_fields,
                    )
                )

        # Check 5: Stored timing dominated by slow outliers
        issues.extend(_timing_outlier_issues(example_key, metrics))

        return issues

    def validate_all(self) -> ValidationReport:
        """Validate all entries in the store for consistency issues.

        Returns:
            ValidationReport summarizing all issues found
        """
        all_issues: List[ValidationIssue] = []
        examples = self._data.get("examples", {})
        total = len(examples)
        valid = 0

        for example_key, entry_dict in examples.items():
            entry_issues = self.validate_entry(example_key, entry_dict)
            if entry_issues:
                all_issues.extend(entry_issues)
            else:
                valid += 1

        return ValidationReport(
            issues=all_issues,
            total_entries=total,
            valid_entries=valid,
        )

    def _update_metadata(self, entry: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Update entry metadata, ensuring speedup consistency.

        INTEGRITY SAFEGUARD: Metadata speedup is derived from stored metrics
        timing values, not from the metadata dict, to ensure consistency.
        """
        entry["example"] = metadata.get("example", entry.get("example"))
        entry["type"] = metadata.get("type", entry.get("type", "python"))
        meta = entry.setdefault("metadata", {})
        if metadata.get("best_optimization"):
            meta["best_optimization"] = metadata["best_optimization"]
        if metadata.get("best_optimization_file"):
            meta["best_optimization_file"] = metadata["best_optimization_file"]
        if metadata.get("best_optimization_time_ms") is not None:
            meta["best_optimization_time_ms"] = metadata["best_optimization_time_ms"]
        if metadata.get("minimum_required_speedup") is not None:
            meta["minimum_required_speedup"] = metadata["minimum_required_speedup"]
        if metadata.get("git_commit"):
            meta["git_commit"] = metadata["git_commit"]

        # SAFEGUARD: Derive metadata speedup from entry metrics (not from metadata dict)
        # This ensures metadata speedup always matches metrics speedup
        stored_metrics = entry.get("metrics", {})
        baseline_time = stored_metrics.get("baseline_time_ms")
        optimized_time = stored_metrics.get("best_optimized_time_ms")
        if baseline_time and optimized_time and optimized_time > 0:
            meta["best_optimization_speedup"] = compute_speedup(baseline_time, optimized_time)
        elif metadata.get("best_optimization_speedup") is not None:
            # Fallback to provided value if timing not available
            meta["best_optimization_speedup"] = metadata["best_optimization_speedup"]

        meta["updated_at"] = datetime.now().isoformat()

    def save(self, force: bool = False) -> None:
        """Save the expectations file.

        Args:
            force: If True, always write even if no changes detected.
        """
        if not self._changed and not force:
            return

        # Ensure schema version is set
        self._data["schema_version"] = SCHEMA_VERSION

        serialized = json.dumps(self._data, indent=2, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(serialized + "\n")
        self._changed = False

    def rewrite_all(self) -> None:
        """Rewrite the entire expectations file from scratch.

        This method serializes all entries fresh, ensuring:
        - All speedups are recomputed from timing values
        - Schema version is updated
        - File is written atomically

        Use this after making changes to ensure full consistency.
        """
        self._data["schema_version"] = SCHEMA_VERSION
        self._changed = True
        self.save(force=True)
