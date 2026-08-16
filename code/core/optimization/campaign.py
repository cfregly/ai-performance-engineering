"""Evidence ledger and promotion gates for long-running optimization campaigns.

The benchmark harness remains responsible for execution, correctness, timing, and
profiling. This module records those results, preserves failed experiments, and
applies campaign-level promotion rules without modifying the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, TypedDict

file_lock_module: Any | None
try:
    import fcntl as _fcntl_module
except ImportError:
    file_lock_module = None
else:
    file_lock_module = _fcntl_module

CONFIG_FILE = "campaign.json"
CONFIG_HASH_FILE = "campaign.sha256"
LEDGER_FILE = "experiments.jsonl"
REPORT_FILE = "REPORT.md"
PRIORS_FILE = "PRIORS.md"
TEMPLATE_FILE = "experiment-template.json"

VALID_DIRECTIONS = {"lower", "higher"}
VALID_AGGREGATES = {"geometric_mean", "arithmetic_mean", "median"}
VALID_STATUSES = {
    "planned",
    "running",
    "completed",
    "inconclusive",
    "crashed",
    "parked",
    "rejected",
    "promoted",
}
VALID_CORRECTNESS = {"passed", "failed", "unknown"}
TERMINAL_STATUSES = {"completed", "inconclusive", "crashed", "parked", "rejected", "promoted"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
RECORD_BINDING_ARTIFACT_KEY = "record_binding_artifact"
RECORD_BINDING_SHA256_KEY = "record_binding_sha256"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_identifier(value: str, field_name: str) -> str:
    normalized = _require_nonempty(value, field_name)
    if not IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must use 1 to 128 letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


def _is_sha256(value: str) -> bool:
    return bool(SHA256_PATTERN.fullmatch(str(value).strip().lower()))


def _normalize_cases(values: Iterable[Any], field_name: str) -> list[str]:
    normalized = [_require_nonempty(str(value), field_name) for value in values]
    return list(dict.fromkeys(normalized))


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, allow_nan=False, indent=indent, sort_keys=True)


@dataclass
class CampaignConfig:
    """Frozen objective and evidence policy for one optimization campaign."""

    objective: str
    primary_metric: str
    initial_control_commit: str
    direction: str = "lower"
    aggregate: str = "geometric_mean"
    primary_cases: list[str] = field(default_factory=list)
    frozen_cases: list[str] = field(default_factory=list)
    beam_width: int = 4
    min_trials: int = 3
    min_improvement_pct: float = 1.0
    max_case_regression_pct: float = 0.0
    max_cv_pct: float | None = 5.0
    require_confidence_bounds: bool = False
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    min_confidence_pairs: int = 3
    require_correctness: bool = True
    require_interleaved_measurements: bool = True
    require_balanced_trials: bool = True
    require_raw_artifact: bool = True
    require_artifact_hashes: bool = True
    require_profile_artifact: bool = False
    require_code_audit: bool = True
    require_mechanism: bool = True
    require_git_provenance: bool = True
    require_candidate_diff: bool = True
    require_derived_evidence: bool = False
    workload_spec: str = ""
    workload_sha256: str = ""
    environment_spec: str = ""
    environment_sha256: str = ""
    max_experiments: int | None = None
    max_total_duration_s: float | None = None
    max_total_cost_usd: float | None = None
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.objective = _require_nonempty(self.objective, "objective")
        self.primary_metric = _require_nonempty(self.primary_metric, "primary_metric")
        self.initial_control_commit = str(self.initial_control_commit).strip().lower()
        if not GIT_COMMIT_PATTERN.fullmatch(self.initial_control_commit):
            raise ValueError("initial_control_commit must be a full lowercase Git commit ID")
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(VALID_DIRECTIONS)}")
        if self.aggregate not in VALID_AGGREGATES:
            raise ValueError(f"aggregate must be one of {sorted(VALID_AGGREGATES)}")
        parsed_beam_width = _optional_positive_int(self.beam_width, "beam_width")
        parsed_min_trials = _optional_positive_int(self.min_trials, "min_trials")
        assert parsed_beam_width is not None
        assert parsed_min_trials is not None
        self.beam_width = parsed_beam_width
        self.min_trials = parsed_min_trials
        self.min_improvement_pct = float(self.min_improvement_pct)
        self.max_case_regression_pct = float(self.max_case_regression_pct)
        if self.max_cv_pct is not None:
            self.max_cv_pct = float(self.max_cv_pct)
        self.confidence_level = float(self.confidence_level)
        parsed_bootstrap_resamples = _optional_positive_int(
            self.bootstrap_resamples, "bootstrap_resamples"
        )
        parsed_min_confidence_pairs = _optional_positive_int(
            self.min_confidence_pairs, "min_confidence_pairs"
        )
        assert parsed_bootstrap_resamples is not None
        assert parsed_min_confidence_pairs is not None
        self.bootstrap_resamples = parsed_bootstrap_resamples
        self.min_confidence_pairs = parsed_min_confidence_pairs
        if not math.isfinite(self.min_improvement_pct) or self.min_improvement_pct < 0:
            raise ValueError("min_improvement_pct must not be negative")
        if not math.isfinite(self.max_case_regression_pct) or self.max_case_regression_pct < 0:
            raise ValueError("max_case_regression_pct must not be negative")
        if self.max_cv_pct is not None and (
            not math.isfinite(self.max_cv_pct) or self.max_cv_pct <= 0
        ):
            raise ValueError("max_cv_pct must be positive and finite")
        if not math.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if self.min_confidence_pairs < 2:
            raise ValueError("min_confidence_pairs must be at least 2")
        self.max_experiments = _optional_positive_int(self.max_experiments, "max_experiments")
        self.max_total_duration_s = _optional_positive_float(
            self.max_total_duration_s, "max_total_duration_s"
        )
        self.max_total_cost_usd = _optional_positive_float(
            self.max_total_cost_usd, "max_total_cost_usd"
        )
        self.workload_sha256 = str(self.workload_sha256).strip().lower()
        self.environment_sha256 = str(self.environment_sha256).strip().lower()
        if self.workload_sha256 and not _is_sha256(self.workload_sha256):
            raise ValueError("workload_sha256 must be a lowercase SHA-256 digest")
        if self.environment_sha256 and not _is_sha256(self.environment_sha256):
            raise ValueError("environment_sha256 must be a lowercase SHA-256 digest")
        self.primary_cases = _normalize_cases(self.primary_cases, "primary case")
        self.frozen_cases = _normalize_cases(self.frozen_cases, "frozen case")
        if not self.primary_cases:
            raise ValueError("primary_cases must contain at least one case ID")
        if not self.frozen_cases:
            raise ValueError("frozen_cases must contain at least one guard case ID")
        self.workload_spec = _require_nonempty(self.workload_spec, "workload_spec")
        self.environment_spec = _require_nonempty(self.environment_spec, "environment_spec")
        if not self.workload_sha256:
            raise ValueError("workload_sha256 is required")
        if not self.environment_sha256:
            raise ValueError("environment_sha256 is required")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CampaignConfig:
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseMeasurement:
    """Repeated control and candidate samples for one workload case."""

    control: list[float]
    candidate: list[float]

    def __post_init__(self) -> None:
        self.control = [float(value) for value in self.control]
        self.candidate = [float(value) for value in self.candidate]
        for label, samples in (("control", self.control), ("candidate", self.candidate)):
            if not samples:
                raise ValueError(f"{label} samples must not be empty")
            if any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError(f"{label} samples must contain positive finite values")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaseMeasurement:
        return cls(
            control=list(data.get("control", [])),
            candidate=list(data.get("candidate", [])),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"control": self.control, "candidate": self.candidate}


@dataclass
class ExperimentRecord:
    """One immutable ledger event for an experiment or a later status update."""

    experiment_id: str
    beam: str
    hypothesis: str
    status: str = "planned"
    parent_id: str | None = None
    changed_surface: list[str] = field(default_factory=list)
    primary_case: str = ""
    measurements: dict[str, CaseMeasurement] = field(default_factory=dict)
    correctness: str = "unknown"
    measurement_protocol: str = ""
    mechanism: str = ""
    code_audit: str = ""
    outcome: str = ""
    next_step: str = ""
    raw_artifacts: list[str] = field(default_factory=list)
    profile_artifacts: list[str] = field(default_factory=list)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    cost_usd: float = 0.0
    workload_sha256: str = ""
    environment_sha256: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    recorded_at: str = field(default_factory=_utc_now)
    revision: int = 0

    def __post_init__(self) -> None:
        self.experiment_id = _require_identifier(self.experiment_id, "experiment_id")
        self.beam = _require_nonempty(self.beam, "beam")
        self.hypothesis = _require_nonempty(self.hypothesis, "hypothesis")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        if self.correctness not in VALID_CORRECTNESS:
            raise ValueError(f"correctness must be one of {sorted(VALID_CORRECTNESS)}")
        self.changed_surface = [str(item) for item in self.changed_surface]
        self.raw_artifacts = [str(item) for item in self.raw_artifacts]
        self.profile_artifacts = [str(item) for item in self.profile_artifacts]
        self.artifact_sha256 = {
            str(artifact): str(digest).strip().lower()
            for artifact, digest in self.artifact_sha256.items()
        }
        self.duration_s = float(self.duration_s)
        self.cost_usd = float(self.cost_usd)
        self.workload_sha256 = str(self.workload_sha256).strip().lower()
        self.environment_sha256 = str(self.environment_sha256).strip().lower()
        if (
            not math.isfinite(self.duration_s)
            or not math.isfinite(self.cost_usd)
            or self.duration_s < 0
            or self.cost_usd < 0
        ):
            raise ValueError("duration_s and cost_usd must be finite and not negative")
        normalized: dict[str, CaseMeasurement] = {}
        for case_id, measurement in self.measurements.items():
            normalized_case = _require_nonempty(str(case_id), "measurement case")
            normalized[normalized_case] = (
                measurement
                if isinstance(measurement, CaseMeasurement)
                else CaseMeasurement.from_dict(measurement)
            )
        self.measurements = normalized

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentRecord:
        payload = dict(data)
        payload["measurements"] = {
            str(case_id): CaseMeasurement.from_dict(measurement)
            for case_id, measurement in dict(payload.get("measurements", {})).items()
        }
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["measurements"] = {
            case_id: measurement.to_dict() for case_id, measurement in self.measurements.items()
        }
        return payload


def capture_local_artifact_hashes(
    record: ExperimentRecord,
    base_dir: Path,
) -> dict[str, str]:
    """Hash local artifact files without treating remote references as local paths."""

    base_dir = Path(base_dir).resolve()
    for artifact in [*record.raw_artifacts, *record.profile_artifacts]:
        if "://" in artifact:
            raise ValueError(f"remote artifact requires a trusted manifest integration: {artifact}")
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = (base_dir / artifact_path).resolve()
            if not artifact_path.is_relative_to(base_dir):
                raise ValueError(f"relative artifact escapes its evidence directory: {artifact}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact file not found: {artifact_path}")
        digest = sha256_file(artifact_path)
        recorded_digest = record.artifact_sha256.get(artifact)
        if recorded_digest and not hmac.compare_digest(recorded_digest, digest):
            raise ValueError(f"artifact hash does not match local file: {artifact}")
        record.artifact_sha256[artifact] = digest
    return dict(record.artifact_sha256)


def resolve_workspace_artifact(workspace_root: Path, artifact: str) -> Path:
    """Resolve one campaign-owned artifact without allowing workspace escape."""

    root = Path(workspace_root).expanduser().resolve()
    artifact_root = (root / "artifacts").resolve()
    if not artifact or "://" in artifact:
        raise ValueError("campaign artifact must be a local workspace reference")
    raw_path = Path(artifact).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError("campaign artifact is outside workspace-owned artifact storage")
    if not resolved.is_file():
        raise FileNotFoundError(f"campaign artifact not found: {artifact}")
    return resolved


def _record_binding_bytes(record: ExperimentRecord) -> bytes:
    payload = record.to_dict()
    payload.pop("recorded_at", None)
    payload.pop("revision", None)
    provenance = dict(payload.get("provenance") or {})
    provenance.pop(RECORD_BINDING_ARTIFACT_KEY, None)
    provenance.pop(RECORD_BINDING_SHA256_KEY, None)
    payload["provenance"] = provenance
    return (_json_dumps(payload) + "\n").encode("utf-8")


def record_artifact_integrity_errors(
    record: ExperimentRecord,
    workspace_root: Path,
    *,
    require_diff_artifact: bool = False,
    require_record_binding: bool = False,
) -> list[str]:
    """Return current-byte integrity errors for all evidence bound to a record."""

    errors: list[str] = []
    artifacts = [*record.raw_artifacts, *record.profile_artifacts]
    if len(artifacts) != len(set(artifacts)):
        errors.append("raw and profile artifact references must be unique")
    for artifact in artifacts:
        expected_digest = record.artifact_sha256.get(artifact, "")
        if not _is_sha256(expected_digest):
            errors.append(f"artifact SHA-256 is missing or invalid for: {artifact}")
            continue
        try:
            artifact_path = resolve_workspace_artifact(workspace_root, artifact)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        actual_digest = sha256_file(artifact_path)
        if not hmac.compare_digest(actual_digest, expected_digest):
            errors.append(f"artifact hash does not match current bytes: {artifact}")

    diff_artifact = str(record.provenance.get("diff_artifact") or "")
    diff_digest = str(record.provenance.get("diff_sha256") or "").lower()
    if require_diff_artifact and not diff_artifact:
        errors.append("candidate diff artifact is missing")
    elif diff_artifact:
        if not _is_sha256(diff_digest):
            errors.append("candidate diff SHA-256 is missing or invalid")
        else:
            try:
                diff_path = resolve_workspace_artifact(workspace_root, diff_artifact)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
            else:
                if not hmac.compare_digest(sha256_file(diff_path), diff_digest):
                    errors.append("candidate diff SHA-256 does not match current bytes")

    binding_artifact = str(record.provenance.get(RECORD_BINDING_ARTIFACT_KEY) or "")
    binding_digest = str(record.provenance.get(RECORD_BINDING_SHA256_KEY) or "").lower()
    if require_record_binding and not binding_artifact:
        errors.append("canonical record binding artifact is missing")
    elif binding_artifact:
        if not _is_sha256(binding_digest):
            errors.append("canonical record binding SHA-256 is missing or invalid")
        else:
            try:
                binding_path = resolve_workspace_artifact(workspace_root, binding_artifact)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(str(exc))
            else:
                expected_parent = (
                    Path(workspace_root).expanduser().resolve()
                    / "artifacts"
                    / record.experiment_id
                    / "evidence"
                ).resolve()
                expected_name = f"record-{binding_digest}.json"
                if binding_path.parent != expected_parent or binding_path.name != expected_name:
                    errors.append("canonical record binding path is invalid")
                actual_digest = sha256_file(binding_path)
                if not hmac.compare_digest(actual_digest, binding_digest):
                    errors.append("canonical record binding SHA-256 does not match current bytes")
                current_digest = _sha256_bytes(_record_binding_bytes(record))
                if not hmac.compare_digest(current_digest, binding_digest):
                    errors.append("canonical record binding does not match current gate inputs")
    return errors


@dataclass
class GateDecision:
    """Promotion result with aggregate and per-case evidence."""

    decision: str
    reasons: list[str]
    aggregate_control: float | None = None
    aggregate_candidate: float | None = None
    improvement_pct: float | None = None
    case_improvements_pct: dict[str, float] = field(default_factory=dict)
    improvement_ci_pct: list[float] | None = None
    case_improvement_ci_pct: dict[str, list[float]] = field(default_factory=dict)
    confidence_method: str | None = None
    confidence_level: float | None = None
    bootstrap_resamples: int = 0
    confidence_required: bool = False
    minimum_trials: int = 0

    @property
    def promotable(self) -> bool:
        return self.decision == "promote"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ConfidenceContext(TypedDict):
    confidence_method: str
    confidence_level: float
    bootstrap_resamples: int
    confidence_required: bool


def _aggregate(values: Iterable[float], method: str) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty sequence")
    if method == "geometric_mean":
        return math.exp(fmean(math.log(value) for value in materialized))
    if method == "arithmetic_mean":
        return fmean(materialized)
    if method == "median":
        return float(median(materialized))
    raise ValueError(f"unsupported aggregate method: {method}")


def _improvement_pct(control: float, candidate: float, direction: str) -> float:
    if direction == "lower":
        return (control - candidate) / control * 100.0
    return (candidate - control) / control * 100.0


def _cv_pct(samples: Sequence[float]) -> float:
    if len(samples) < 2:
        return 0.0
    mean = fmean(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)
    return math.sqrt(variance) / mean * 100.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile from an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight)


def _bootstrap_seed(case_id: str, measurement: CaseMeasurement) -> int:
    payload = _json_dumps(
        {
            "case_id": case_id,
            "control": measurement.control,
            "candidate": measurement.candidate,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _paired_bootstrap_case_samples(
    case_id: str,
    measurement: CaseMeasurement,
    *,
    direction: str,
    resamples: int,
) -> tuple[list[float], list[float], list[float]]:
    if len(measurement.control) != len(measurement.candidate):
        raise ValueError("paired bootstrap requires balanced control and candidate samples")
    pair_count = len(measurement.control)
    generator = random.Random(_bootstrap_seed(case_id, measurement))
    control_medians: list[float] = []
    candidate_medians: list[float] = []
    improvements: list[float] = []
    for _ in range(resamples):
        indices = [generator.randrange(pair_count) for _ in range(pair_count)]
        control_median = float(median(measurement.control[index] for index in indices))
        candidate_median = float(median(measurement.candidate[index] for index in indices))
        control_medians.append(control_median)
        candidate_medians.append(candidate_median)
        improvements.append(_improvement_pct(control_median, candidate_median, direction))
    return control_medians, candidate_medians, improvements


def _confidence_interval(samples: Sequence[float], confidence_level: float) -> list[float]:
    ordered = sorted(samples)
    tail_probability = (1.0 - confidence_level) / 2.0
    return [
        _percentile(ordered, tail_probability),
        _percentile(ordered, 1.0 - tail_probability),
    ]


def evaluate_experiment(config: CampaignConfig, record: ExperimentRecord) -> GateDecision:
    """Evaluate one experiment against the frozen campaign contract."""

    confidence_required = bool(config.require_confidence_bounds or config.require_derived_evidence)
    confidence_context: _ConfidenceContext = {
        "confidence_method": "paired_bootstrap_percentile",
        "confidence_level": config.confidence_level,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_required": confidence_required,
    }

    if record.status in {"planned", "running", "inconclusive"}:
        return GateDecision(
            decision="inconclusive",
            reasons=[f"experiment status is {record.status}"],
            **confidence_context,
        )
    if record.status == "crashed":
        return GateDecision(decision="reject", reasons=["experiment crashed"], **confidence_context)
    if record.status == "rejected":
        return GateDecision(
            decision="reject",
            reasons=["experiment was explicitly rejected"],
            **confidence_context,
        )
    if record.status == "parked":
        return GateDecision(
            decision="park",
            reasons=["experiment was explicitly parked"],
            **confidence_context,
        )
    if record.correctness == "failed":
        return GateDecision(decision="reject", reasons=["correctness failed"], **confidence_context)
    if not record.measurements:
        return GateDecision(
            decision="inconclusive",
            reasons=["no measurements recorded"],
            **confidence_context,
        )

    measured_cases = set(record.measurements)
    missing_primary = sorted(set(config.primary_cases) - measured_cases)
    missing_frozen = sorted(set(config.frozen_cases) - measured_cases)
    if missing_primary or missing_frozen:
        missing_reasons = []
        if missing_primary:
            missing_reasons.append(f"missing primary cases: {', '.join(missing_primary)}")
        if missing_frozen:
            missing_reasons.append(f"missing frozen cases: {', '.join(missing_frozen)}")
        return GateDecision(decision="inconclusive", reasons=missing_reasons, **confidence_context)

    case_improvements: dict[str, float] = {}
    control_medians: dict[str, float] = {}
    candidate_medians: dict[str, float] = {}
    required_cases = list(dict.fromkeys([*config.primary_cases, *config.frozen_cases])) or sorted(
        record.measurements
    )
    trial_counts: list[int] = []
    unstable_cases: list[str] = []
    unbalanced_cases: list[str] = []
    for case_id, measurement in sorted(record.measurements.items()):
        control_median = float(median(measurement.control))
        candidate_median = float(median(measurement.candidate))
        control_medians[case_id] = control_median
        candidate_medians[case_id] = candidate_median
        case_improvements[case_id] = _improvement_pct(
            control_median, candidate_median, config.direction
        )
        if case_id not in required_cases:
            continue
        trial_counts.extend((len(measurement.control), len(measurement.candidate)))
        if len(measurement.control) != len(measurement.candidate):
            unbalanced_cases.append(case_id)
        if config.max_cv_pct is not None and (
            max(_cv_pct(measurement.control), _cv_pct(measurement.candidate)) > config.max_cv_pct
        ):
            unstable_cases.append(case_id)

    minimum_trials = min(trial_counts)
    if config.require_balanced_trials and unbalanced_cases:
        return GateDecision(
            decision="inconclusive",
            reasons=[
                "control and candidate trial counts differ for: " + ", ".join(unbalanced_cases)
            ],
            case_improvements_pct=case_improvements,
            minimum_trials=minimum_trials,
            **confidence_context,
        )
    if minimum_trials < config.min_trials:
        return GateDecision(
            decision="inconclusive",
            reasons=[f"minimum trial count {minimum_trials} is below required {config.min_trials}"],
            case_improvements_pct=case_improvements,
            minimum_trials=minimum_trials,
            **confidence_context,
        )
    if unstable_cases:
        return GateDecision(
            decision="inconclusive",
            reasons=[f"sample variance exceeds the CV gate for: {', '.join(unstable_cases)}"],
            case_improvements_pct=case_improvements,
            minimum_trials=minimum_trials,
            **confidence_context,
        )

    confidence_pair_counts = {
        case_id: min(
            len(record.measurements[case_id].control),
            len(record.measurements[case_id].candidate),
        )
        for case_id in required_cases
    }
    insufficient_confidence_cases = [
        case_id
        for case_id, pair_count in confidence_pair_counts.items()
        if pair_count < config.min_confidence_pairs
    ]
    confidence_available = not unbalanced_cases and not insufficient_confidence_cases
    if confidence_required and not confidence_available:
        confidence_reasons: list[str] = []
        if unbalanced_cases:
            confidence_reasons.append(
                "paired confidence bounds require balanced trials for: "
                + ", ".join(unbalanced_cases)
            )
        if insufficient_confidence_cases:
            confidence_reasons.append(
                f"paired confidence bounds require at least {config.min_confidence_pairs} "
                "pairs for: " + ", ".join(insufficient_confidence_cases)
            )
        return GateDecision(
            decision="inconclusive",
            reasons=confidence_reasons,
            case_improvements_pct=case_improvements,
            minimum_trials=minimum_trials,
            **confidence_context,
        )

    aggregate_cases = config.primary_cases or sorted(record.measurements)
    aggregate_control = _aggregate(
        (control_medians[case_id] for case_id in aggregate_cases), config.aggregate
    )
    aggregate_candidate = _aggregate(
        (candidate_medians[case_id] for case_id in aggregate_cases), config.aggregate
    )
    aggregate_improvement = _improvement_pct(
        aggregate_control, aggregate_candidate, config.direction
    )

    case_improvement_ci: dict[str, list[float]] = {}
    aggregate_improvement_ci: list[float] | None = None
    if confidence_available:
        bootstrap_by_case: dict[str, tuple[list[float], list[float], list[float]]] = {}
        for case_id in required_cases:
            bootstrap_by_case[case_id] = _paired_bootstrap_case_samples(
                case_id,
                record.measurements[case_id],
                direction=config.direction,
                resamples=config.bootstrap_resamples,
            )
            case_improvement_ci[case_id] = _confidence_interval(
                bootstrap_by_case[case_id][2], config.confidence_level
            )
        aggregate_bootstrap: list[float] = []
        for bootstrap_index in range(config.bootstrap_resamples):
            bootstrap_control = _aggregate(
                (bootstrap_by_case[case_id][0][bootstrap_index] for case_id in aggregate_cases),
                config.aggregate,
            )
            bootstrap_candidate = _aggregate(
                (bootstrap_by_case[case_id][1][bootstrap_index] for case_id in aggregate_cases),
                config.aggregate,
            )
            aggregate_bootstrap.append(
                _improvement_pct(bootstrap_control, bootstrap_candidate, config.direction)
            )
        aggregate_improvement_ci = _confidence_interval(
            aggregate_bootstrap, config.confidence_level
        )

    reasons: list[str] = []
    if config.require_correctness and record.correctness != "passed":
        reasons.append("correctness is not recorded as passed")
    if config.require_interleaved_measurements and record.measurement_protocol != "interleaved":
        reasons.append("measurement protocol is not interleaved")
    if config.require_raw_artifact and not record.raw_artifacts:
        reasons.append("raw benchmark artifact is missing")
    if config.require_profile_artifact and not record.profile_artifacts:
        reasons.append("required profile artifact is missing")
    if config.require_code_audit and not record.code_audit.strip():
        reasons.append("code audit is missing")
    if config.require_mechanism and not record.mechanism.strip():
        reasons.append("mechanism explanation is missing")
    if config.require_artifact_hashes:
        required_artifacts = list(record.raw_artifacts)
        if config.require_profile_artifact:
            required_artifacts.extend(record.profile_artifacts)
        unhashed_artifacts = [
            artifact
            for artifact in required_artifacts
            if not _is_sha256(record.artifact_sha256.get(artifact, ""))
        ]
        if unhashed_artifacts:
            reasons.append(
                "artifact SHA-256 is missing or invalid for: " + ", ".join(unhashed_artifacts)
            )
    if config.require_git_provenance:
        required_provenance = ("repo_root", "git_commit", "diff_sha256")
        missing_provenance = [
            field_name
            for field_name in required_provenance
            if not record.provenance.get(field_name)
        ]
        if missing_provenance:
            reasons.append("git provenance is missing: " + ", ".join(missing_provenance))
        elif not _is_sha256(str(record.provenance["diff_sha256"])):
            reasons.append("candidate diff SHA-256 is invalid")
    if config.require_candidate_diff and not record.provenance.get("diff_bytes"):
        reasons.append("candidate diff is empty or was not captured")
    if config.require_derived_evidence:
        evidence_adapter = record.provenance.get("evidence_adapter")
        if evidence_adapter != "aisp.campaign-benchmark-evidence/v1":
            reasons.append("measurements and correctness were not derived from benchmark evidence")
        if not _is_sha256(str(record.provenance.get("evidence_manifest_sha256") or "")):
            reasons.append("derived evidence manifest SHA-256 is missing or invalid")
        if record.provenance.get("derived_measurements") is not True:
            reasons.append("measurements are not marked as artifact-derived")
        if record.provenance.get("derived_correctness") is not True:
            reasons.append("correctness is not marked as artifact-derived")
    if config.workload_sha256 and record.workload_sha256 != config.workload_sha256:
        reasons.append("workload hash does not match the frozen campaign workload")
    if config.environment_sha256 and record.environment_sha256 != config.environment_sha256:
        reasons.append("environment hash does not match the campaign baseline")

    guarded_cases = config.frozen_cases or sorted(record.measurements)
    regressed_cases = [
        case_id
        for case_id in guarded_cases
        if case_improvements[case_id] < -config.max_case_regression_pct
    ]
    if regressed_cases:
        reasons.append(
            "frozen-case regression exceeds the limit for: " + ", ".join(regressed_cases)
        )
    if confidence_required:
        confidence_regressed_cases = [
            case_id
            for case_id in guarded_cases
            if case_improvement_ci[case_id][0] < -config.max_case_regression_pct
        ]
        if confidence_regressed_cases:
            reasons.append(
                "frozen-case confidence lower bound exceeds the regression limit for: "
                + ", ".join(confidence_regressed_cases)
            )
    if aggregate_improvement < config.min_improvement_pct:
        reasons.append(
            f"aggregate improvement {aggregate_improvement:.3f}% is below required "
            f"{config.min_improvement_pct:.3f}%"
        )
    if (
        confidence_required
        and aggregate_improvement_ci is not None
        and aggregate_improvement_ci[0] < config.min_improvement_pct
    ):
        reasons.append(
            f"aggregate improvement confidence lower bound {aggregate_improvement_ci[0]:.3f}% "
            f"is below required {config.min_improvement_pct:.3f}%"
        )

    return GateDecision(
        decision="park" if reasons else "promote",
        reasons=reasons,
        aggregate_control=aggregate_control,
        aggregate_candidate=aggregate_candidate,
        improvement_pct=aggregate_improvement,
        case_improvements_pct=case_improvements,
        improvement_ci_pct=aggregate_improvement_ci,
        case_improvement_ci_pct=case_improvement_ci,
        minimum_trials=minimum_trials,
        **confidence_context,
    )


def current_incumbent(
    config: CampaignConfig,
    records: Iterable[ExperimentRecord],
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Return the verified control revision for the next measured experiment."""

    latest_by_id: dict[str, ExperimentRecord] = {}
    for record in records:
        if record.experiment_id in latest_by_id:
            del latest_by_id[record.experiment_id]
        latest_by_id[record.experiment_id] = record
    promoted = [record for record in latest_by_id.values() if record.status == "promoted"]
    if not promoted:
        return {
            "commit": config.initial_control_commit,
            "source": "initial_control",
            "experiment_id": None,
            "recorded_at": config.created_at,
        }
    if workspace_root is None:
        raise ValueError("workspace_root is required to verify a promoted campaign incumbent")

    promoted_by_control: dict[str, tuple[ExperimentRecord, str]] = {}
    candidate_commits: set[str] = set()
    for promoted_record in promoted:
        integrity_errors = record_artifact_integrity_errors(
            promoted_record,
            workspace_root,
            require_diff_artifact=(
                config.require_git_provenance or config.require_candidate_diff
            ),
            require_record_binding=True,
        )
        if integrity_errors:
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} fails evidence integrity: "
                + ", ".join(integrity_errors)
            )
        decision = evaluate_experiment(config, promoted_record)
        if not decision.promotable:
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} fails its mechanical gate: "
                + ", ".join(decision.reasons)
            )
        recorded_control = str(promoted_record.provenance.get("control_commit") or "").lower()
        candidate_commit = str(
            promoted_record.provenance.get("candidate_commit")
            or promoted_record.provenance.get("git_commit")
            or ""
        ).lower()
        git_commit = str(promoted_record.provenance.get("git_commit") or "").lower()
        if promoted_record.parent_id != recorded_control:
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} has conflicting control lineage"
            )
        if not GIT_COMMIT_PATTERN.fullmatch(candidate_commit):
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} has no full candidate commit"
            )
        if git_commit and git_commit != candidate_commit:
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} has conflicting candidate commits"
            )
        if candidate_commit == recorded_control:
            raise ValueError(
                f"promoted experiment {promoted_record.experiment_id} reuses its control commit"
            )
        if recorded_control in promoted_by_control:
            raise ValueError(f"promoted experiments branch from control {recorded_control}")
        if candidate_commit in candidate_commits:
            raise ValueError(f"promoted experiments reuse candidate commit {candidate_commit}")
        promoted_by_control[recorded_control] = (promoted_record, candidate_commit)
        candidate_commits.add(candidate_commit)

    expected_control = config.initial_control_commit
    incumbent: ExperimentRecord | None = None
    candidate_commit = ""
    visited_controls: set[str] = set()
    for _ in promoted:
        if expected_control in visited_controls:
            raise ValueError("promoted experiment lineage contains a cycle")
        visited_controls.add(expected_control)
        next_promotion = promoted_by_control.get(expected_control)
        if next_promotion is None:
            raise ValueError(
                "promoted experiment lineage does not form one continuous chain from "
                f"{config.initial_control_commit}"
            )
        incumbent, candidate_commit = next_promotion
        expected_control = candidate_commit

    assert incumbent is not None
    return {
        "commit": candidate_commit,
        "source": "promoted_experiment",
        "experiment_id": incumbent.experiment_id,
        "recorded_at": incumbent.recorded_at,
    }


def _validate_record_lineage(
    config: CampaignConfig,
    existing: Sequence[ExperimentRecord],
    record: ExperimentRecord,
    workspace_root: Path | None = None,
) -> None:
    if not record.measurements:
        return
    incumbent = current_incumbent(config, existing, workspace_root=workspace_root)
    expected_control = str(incumbent["commit"])
    recorded_control = str(record.provenance.get("control_commit") or "").lower()
    candidate_commit = str(
        record.provenance.get("candidate_commit") or record.provenance.get("git_commit") or ""
    ).lower()
    git_commit = str(record.provenance.get("git_commit") or "").lower()
    if record.parent_id != expected_control:
        raise ValueError(f"experiment parent_id must match current incumbent {expected_control}")
    if recorded_control != expected_control:
        raise ValueError(
            f"experiment control_commit must match current incumbent {expected_control}"
        )
    if not GIT_COMMIT_PATTERN.fullmatch(candidate_commit):
        raise ValueError("measured experiment candidate_commit must be a full Git commit ID")
    if git_commit and git_commit != candidate_commit:
        raise ValueError("git_commit and candidate_commit identify different revisions")
    if candidate_commit == expected_control:
        raise ValueError("measured candidate revision must differ from the current incumbent")


class ExperimentLedger:
    """Append-only JSONL event ledger with latest-revision views."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @staticmethod
    def _read_handle(handle: Any) -> list[ExperimentRecord]:
        handle.seek(0)
        records: list[ExperimentRecord] = []
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(ExperimentRecord.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid ledger entry at line {line_number}: {error}") from error
        return records

    def events(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        if file_lock_module is None:
            raise RuntimeError("campaign ledgers require POSIX file locking")
        with self.path.open("r", encoding="utf-8") as handle:
            file_lock_module.flock(handle.fileno(), file_lock_module.LOCK_SH)
            try:
                return self._read_handle(handle)
            finally:
                file_lock_module.flock(handle.fileno(), file_lock_module.LOCK_UN)

    def latest(self) -> list[ExperimentRecord]:
        latest_by_id: dict[str, ExperimentRecord] = {}
        for record in self.events():
            if record.experiment_id in latest_by_id:
                del latest_by_id[record.experiment_id]
            latest_by_id[record.experiment_id] = record
        return list(latest_by_id.values())

    def get(self, experiment_id: str) -> ExperimentRecord:
        for record in reversed(self.events()):
            if record.experiment_id == experiment_id:
                return record
        raise KeyError(f"experiment not found: {experiment_id}")

    def append(
        self,
        record: ExperimentRecord,
        config: CampaignConfig | None = None,
        workspace_root: Path | None = None,
    ) -> ExperimentRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if file_lock_module is None:
            raise RuntimeError("campaign ledgers require POSIX file locking")
        with self.path.open("a+", encoding="utf-8") as handle:
            file_lock_module.flock(handle.fileno(), file_lock_module.LOCK_EX)
            try:
                existing = self._read_handle(handle)
                latest_by_id: dict[str, ExperimentRecord] = {}
                for item in existing:
                    latest_by_id[item.experiment_id] = item
                previous = latest_by_id.get(record.experiment_id)
                if config is not None:
                    _validate_record_lineage(
                        config,
                        existing,
                        record,
                        workspace_root=workspace_root,
                    )
                if previous is not None and previous.status in TERMINAL_STATUSES:
                    previous_evidence = previous.to_dict()
                    candidate_evidence = record.to_dict()
                    for evidence in (previous_evidence, candidate_evidence):
                        provenance = dict(evidence.get("provenance") or {})
                        provenance.pop(RECORD_BINDING_ARTIFACT_KEY, None)
                        provenance.pop(RECORD_BINDING_SHA256_KEY, None)
                        evidence["provenance"] = provenance
                    for mutable_field in (
                        "status",
                        "outcome",
                        "next_step",
                        "notes",
                        "recorded_at",
                        "revision",
                    ):
                        previous_evidence.pop(mutable_field, None)
                        candidate_evidence.pop(mutable_field, None)
                    if previous_evidence != candidate_evidence:
                        raise ValueError(
                            "terminal experiment evidence is immutable across revisions"
                        )
                if config is not None and previous is None:
                    budget = budget_status(config, latest_by_id.values())
                    if not budget["can_schedule"]:
                        raise RuntimeError(
                            "campaign budget is exhausted: " + ", ".join(budget["exhausted"])
                        )
                record.revision = 1 + max(
                    (
                        item.revision
                        for item in existing
                        if item.experiment_id == record.experiment_id
                    ),
                    default=0,
                )
                record.recorded_at = _utc_now()
                handle.seek(0, 2)
                handle.write(_json_dumps(record.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                file_lock_module.flock(handle.fileno(), file_lock_module.LOCK_UN)
        return record


def _record_rank(config: CampaignConfig, record: ExperimentRecord) -> tuple[int, float, str]:
    decision = evaluate_experiment(config, record)
    decision_rank = {"promote": 3, "park": 2, "inconclusive": 1, "reject": 0}
    return (
        decision_rank.get(decision.decision, 0),
        decision.improvement_pct if decision.improvement_pct is not None else -math.inf,
        record.recorded_at,
    )


def active_beam(
    config: CampaignConfig, records: Iterable[ExperimentRecord]
) -> list[ExperimentRecord]:
    """Return at most one live candidate per idea family."""

    latest_by_beam: dict[str, ExperimentRecord] = {}
    for record in records:
        if record.status in {"rejected", "crashed", "parked", "promoted"}:
            continue
        incumbent = latest_by_beam.get(record.beam)
        if incumbent is None or _record_rank(config, record) > _record_rank(config, incumbent):
            latest_by_beam[record.beam] = record
    return sorted(
        latest_by_beam.values(), key=lambda record: _record_rank(config, record), reverse=True
    )[: config.beam_width]


def promotion_frontier(
    config: CampaignConfig, records: Iterable[ExperimentRecord]
) -> list[tuple[ExperimentRecord, GateDecision]]:
    """Return gate-passing experiments that improved the best candidate score."""

    frontier: list[tuple[ExperimentRecord, GateDecision]] = []
    best_score: float | None = None
    for record in records:
        if record.status == "parked":
            continue
        decision = evaluate_experiment(config, record)
        if not decision.promotable or decision.aggregate_candidate is None:
            continue
        score = decision.aggregate_candidate
        better = best_score is None or (
            score < best_score if config.direction == "lower" else score > best_score
        )
        if better:
            frontier.append((record, decision))
            best_score = score
    return frontier


def budget_status(config: CampaignConfig, records: Iterable[ExperimentRecord]) -> dict[str, Any]:
    """Summarize experiment, duration, and cost budgets."""

    latest = list(records)
    completed = [record for record in latest if record.status not in {"planned", "running"}]
    duration_s = sum(record.duration_s for record in completed)
    cost_usd = sum(record.cost_usd for record in completed)
    exhausted: list[str] = []
    if config.max_experiments is not None and len(completed) >= config.max_experiments:
        exhausted.append("experiment count")
    if config.max_total_duration_s is not None and duration_s >= config.max_total_duration_s:
        exhausted.append("total duration")
    if config.max_total_cost_usd is not None and cost_usd >= config.max_total_cost_usd:
        exhausted.append("total cost")
    return {
        "completed_experiments": len(completed),
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "exhausted": exhausted,
        "can_schedule": not exhausted,
    }


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_report(
    config: CampaignConfig,
    records: Sequence[ExperimentRecord],
    *,
    workspace_root: Path | None = None,
) -> str:
    """Render the compact campaign view used for steering and promotion review."""

    measured = [record for record in records if record.measurements]
    if measured and workspace_root is None:
        raise ValueError("workspace_root is required to verify measured campaign evidence")
    if workspace_root is not None:
        for record in measured:
            integrity_errors = record_artifact_integrity_errors(
                record,
                workspace_root,
                require_diff_artifact=(
                    config.require_git_provenance or config.require_candidate_diff
                ),
                require_record_binding=True,
            )
            if integrity_errors:
                raise ValueError(
                    f"experiment {record.experiment_id} fails evidence integrity: "
                    + ", ".join(integrity_errors)
                )
    status_counts = Counter(record.status for record in records)
    budget = budget_status(config, records)
    frontier = promotion_frontier(config, records)
    beam = active_beam(config, records)
    incumbent = current_incumbent(config, records, workspace_root=workspace_root)
    lines = [
        "# Optimization Campaign",
        "",
        f"Objective: {_escape_table(config.objective)}",
        "",
        f"Primary metric: `{config.primary_metric}`. Direction: `{config.direction}`. "
        f"Aggregate: `{config.aggregate}`.",
        "",
        f"Confidence gate: `{config.confidence_level:.1%}` paired bootstrap with "
        f"`{config.bootstrap_resamples}` resamples. Required: "
        f"`{config.require_confidence_bounds or config.require_derived_evidence}`.",
        "",
        f"Current incumbent: `{incumbent['commit']}` from `{incumbent['source']}`.",
        "",
        "## Status",
        "",
        f"Recorded experiments: {len(records)}",
        "",
        f"Completed budget count: {budget['completed_experiments']}",
        "",
        f"Recorded duration: {budget['duration_s']:.1f} seconds",
        "",
        f"Recorded cost: ${budget['cost_usd']:.2f}",
        "",
        "Budget state: "
        + ("exhausted: " + ", ".join(budget["exhausted"]) if budget["exhausted"] else "open"),
        "",
        "Status counts: "
        + (", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())) or "none"),
        "",
        "## Promotion Frontier",
        "",
        "| Experiment | Beam | Candidate | Improvement | Confidence interval |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if frontier:
        for record, decision in frontier:
            interval_text = (
                f"{decision.improvement_ci_pct[0]:.3f}% to {decision.improvement_ci_pct[1]:.3f}%"
                if decision.improvement_ci_pct is not None
                else "unavailable"
            )
            lines.append(
                f"| {_escape_table(record.experiment_id)} | {_escape_table(record.beam)} | "
                f"{decision.aggregate_candidate:.6g} | {decision.improvement_pct:.3f}% | "
                f"{interval_text} |"
            )
    else:
        lines.append("| none |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Active Beam",
            "",
            "| Beam | Experiment | Decision | Improvement | Next step |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    if beam:
        for record in beam:
            decision = evaluate_experiment(config, record)
            improvement = (
                f"{decision.improvement_pct:.3f}%" if decision.improvement_pct is not None else ""
            )
            lines.append(
                f"| {_escape_table(record.beam)} | {_escape_table(record.experiment_id)} | "
                f"{decision.decision} | {improvement} | {_escape_table(record.next_step)} |"
            )
    else:
        lines.append("| none |  |  |  |  |")

    lines.extend(["", "## Latest Promotion Glance", ""])
    if not measured:
        lines.append("No measured experiment has been recorded.")
        return "\n".join(lines) + "\n"

    latest = measured[-1]
    decision = evaluate_experiment(config, latest)
    lines.extend(
        [
            f"Experiment: `{latest.experiment_id}`",
            "",
            f"Decision: `{decision.decision}`",
            "",
            "Reasons: " + ("none" if not decision.reasons else ", ".join(decision.reasons)),
            "",
            "| Case | Control median | Candidate median | Improvement | Confidence interval | Trials |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case_id, measurement in sorted(latest.measurements.items()):
        case_improvement = decision.case_improvements_pct.get(case_id)
        if case_improvement is None:
            case_improvement = _improvement_pct(
                float(median(measurement.control)),
                float(median(measurement.candidate)),
                config.direction,
            )
        case_interval = decision.case_improvement_ci_pct.get(case_id)
        interval_text = (
            f"{case_interval[0]:.3f}% to {case_interval[1]:.3f}%"
            if case_interval is not None
            else "unavailable"
        )
        lines.append(
            f"| {_escape_table(case_id)} | {median(measurement.control):.6g} | "
            f"{median(measurement.candidate):.6g} | "
            f"{case_improvement:.3f}% | {interval_text} | "
            f"{min(len(measurement.control), len(measurement.candidate))} |"
        )
    return "\n".join(lines) + "\n"


def render_priors(config: CampaignConfig, records: Sequence[ExperimentRecord]) -> str:
    """Render stable conclusions from both successful and unsuccessful work."""

    lines = [
        "# Campaign Priors",
        "",
        "This file is generated from the experiment ledger. Update the ledger instead of this file.",
        "",
        "| Experiment | Beam | Disposition | Hypothesis | Finding | Mechanism | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    prior_records = [
        record
        for record in records
        if record.status in {"parked", "rejected", "promoted", "crashed"}
    ]
    if not prior_records:
        lines.append("| none |  |  |  |  |  |  |")
    for record in prior_records:
        evidence = [*record.raw_artifacts, *record.profile_artifacts]
        lines.append(
            f"| {_escape_table(record.experiment_id)} | {_escape_table(record.beam)} | "
            f"{record.status} | {_escape_table(record.hypothesis)} | "
            f"{_escape_table(record.outcome or record.notes)} | "
            f"{_escape_table(record.mechanism)} | {_escape_table(', '.join(evidence))} |"
        )
    return "\n".join(lines) + "\n"


def capture_git_provenance(
    repo: Path,
    changed_surface: Sequence[str],
    artifact_dir: Path,
    experiment_id: str,
    control_revision: str | None = None,
) -> dict[str, Any]:
    """Capture git identity and a concrete candidate diff artifact."""

    repo = Path(repo).resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    repo = root
    control_commit = git(
        "rev-parse",
        "--verify",
        f"{control_revision or 'HEAD'}^{{commit}}",
    )
    candidate_commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    status = git("status", "--short")
    path_filters: list[str] = []
    for raw_surface in changed_surface:
        surface = re.sub(r":\d+(?::\d+)?$", "", str(raw_surface).strip())
        if not surface:
            continue
        surface_path = Path(surface)
        if not surface_path.is_absolute():
            surface_path = root / surface_path
        resolved_surface = surface_path.resolve()
        if not resolved_surface.is_relative_to(root):
            raise ValueError(f"changed surface escapes the repository: {raw_surface}")
        path_filters.append(str(resolved_surface.relative_to(root)))
    path_filters = list(dict.fromkeys(path_filters))
    if not path_filters:
        raise ValueError("changed_surface must name at least one repository path")

    changed_paths = set(git("diff", "--name-only", control_commit, "--").splitlines())
    changed_paths.update(git("ls-files", "--others", "--exclude-standard").splitlines())

    def is_declared(path: str) -> bool:
        return any(
            path == surface or path.startswith(surface.rstrip("/") + "/")
            for surface in path_filters
        )

    undeclared_paths = sorted(path for path in changed_paths if path and not is_declared(path))
    if undeclared_paths:
        raise ValueError(
            "candidate has changed paths outside changed_surface: " + ", ".join(undeclared_paths)
        )

    diff_args = ["diff", "--binary", control_commit, "--"]
    diff_args.extend(path_filters)
    diff_parts = [git(*diff_args)]

    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *path_filters,
    ).splitlines()
    for relative_path in untracked:
        candidate_path = (root / relative_path).resolve()
        if not candidate_path.is_relative_to(root):
            raise ValueError(f"changed surface escapes the repository: {relative_path}")
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", os.devnull, relative_path],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        diff_parts.append(result.stdout.strip())

    diff = "\n".join(part for part in diff_parts if part)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    diff_path = artifact_dir / f"{experiment_id}.diff"
    diff_payload = (diff + ("\n" if diff else "")).encode("utf-8")
    diff_path.write_bytes(diff_payload)
    return {
        "repo_root": str(root),
        "git_commit": candidate_commit,
        "control_commit": control_commit,
        "candidate_commit": candidate_commit,
        "git_branch": branch,
        "git_dirty": bool(status),
        "git_status": status,
        "diff_artifact": str(diff_path),
        "diff_sha256": _sha256_bytes(diff_payload),
        "diff_bytes": len(diff_payload),
    }


def experiment_template(config: CampaignConfig) -> dict[str, Any]:
    return {
        "experiment_id": "exp-001",
        "parent_id": config.initial_control_commit,
        "beam": "structural",
        "hypothesis": "Name one cost and the mechanism expected to reduce it.",
        "status": "planned",
        "changed_surface": [],
        "primary_case": config.primary_cases[0],
        "correctness": "unknown",
        "measurement_protocol": "",
        "measurements": {},
        "mechanism": "",
        "code_audit": "",
        "outcome": "",
        "next_step": "",
        "raw_artifacts": [],
        "profile_artifacts": [],
        "artifact_sha256": {},
        "duration_s": 0.0,
        "cost_usd": 0.0,
        "workload_sha256": config.workload_sha256,
        "environment_sha256": config.environment_sha256,
        "provenance": {
            "agent_model": "",
            "prompt_sha256": "",
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


class CampaignWorkspace:
    """Filesystem contract for a campaign config, ledger, reports, and artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.config_path = self.root / CONFIG_FILE
        self.ledger = ExperimentLedger(self.root / LEDGER_FILE)

    def _artifact_source_path(self, artifact: str, base_dir: Path) -> Path:
        if not artifact or "://" in artifact:
            raise ValueError(f"remote artifact requires a trusted manifest integration: {artifact}")
        raw_path = Path(artifact).expanduser()
        if raw_path.is_absolute():
            source = raw_path.resolve()
        else:
            owned_candidate = (self.root / raw_path).resolve()
            if owned_candidate.is_file() and owned_candidate.is_relative_to(
                (self.root / "artifacts").resolve()
            ):
                source = owned_candidate
            else:
                resolved_base = Path(base_dir).expanduser().resolve()
                source = (resolved_base / raw_path).resolve()
                if not source.is_relative_to(resolved_base):
                    raise ValueError(f"relative artifact escapes its evidence directory: {artifact}")
        if not source.is_file():
            raise FileNotFoundError(f"artifact file not found: {source}")
        return source

    @staticmethod
    def _artifact_storage_name(role: str, index: int, source: Path, digest: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", source.name).strip("-.") or "artifact"
        return f"{role}-{index:03d}-{digest[:16]}-{safe_name[:80]}"

    def _copy_immutable_artifact(self, source: Path, destination: Path, digest: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file() or not hmac.compare_digest(
                sha256_file(destination), digest
            ):
                raise ValueError(f"workspace artifact collision: {destination}")
            destination.chmod(0o444)
            return
        try:
            with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or not hmac.compare_digest(
                sha256_file(destination), digest
            ):
                raise ValueError(f"workspace artifact collision: {destination}") from None
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if not hmac.compare_digest(sha256_file(destination), digest):
            destination.unlink(missing_ok=True)
            raise ValueError(f"copied artifact hash does not match source: {source}")
        destination.chmod(0o444)

    def _materialize_record_artifacts(
        self,
        record: ExperimentRecord,
        artifact_base_dir: Path,
    ) -> None:
        original_hashes = dict(record.artifact_sha256)
        original_references = [*record.raw_artifacts, *record.profile_artifacts]
        retained_hashes = {
            artifact: digest
            for artifact, digest in original_hashes.items()
            if artifact not in original_references
        }

        def materialize(role: str, references: list[str]) -> list[str]:
            owned_references: list[str] = []
            canonical_root = (self.root / "artifacts" / record.experiment_id / "evidence").resolve()
            for index, artifact in enumerate(references, start=1):
                source = self._artifact_source_path(artifact, artifact_base_dir)
                digest = sha256_file(source)
                expected_digest = original_hashes.get(artifact)
                if expected_digest and not hmac.compare_digest(expected_digest, digest):
                    raise ValueError(f"artifact hash does not match local file: {artifact}")
                canonical_prefix = f"{role}-{index:03d}-{digest[:16]}-"
                if source.parent == canonical_root and source.name.startswith(canonical_prefix):
                    destination = source
                else:
                    destination = canonical_root / self._artifact_storage_name(
                        role,
                        index,
                        source,
                        digest,
                    )
                    self._copy_immutable_artifact(source, destination, digest)
                destination.chmod(0o444)
                reference = str(destination.relative_to(self.root))
                owned_references.append(reference)
                retained_hashes[reference] = digest
            return owned_references

        record.raw_artifacts = materialize("raw", record.raw_artifacts)
        record.profile_artifacts = materialize("profile", record.profile_artifacts)
        record.artifact_sha256 = retained_hashes

        diff_artifact = str(record.provenance.get("diff_artifact") or "")
        if diff_artifact:
            source = self._artifact_source_path(diff_artifact, artifact_base_dir)
            digest = sha256_file(source)
            expected_digest = str(record.provenance.get("diff_sha256") or "").lower()
            if expected_digest and not hmac.compare_digest(expected_digest, digest):
                raise ValueError("candidate diff SHA-256 does not match local file")
            canonical_root = (
                self.root / "artifacts" / record.experiment_id / "evidence"
            ).resolve()
            canonical_prefix = f"diff-001-{digest[:16]}-"
            if source.parent == canonical_root and source.name.startswith(canonical_prefix):
                destination = source
            else:
                destination = canonical_root / self._artifact_storage_name(
                    "diff",
                    1,
                    source,
                    digest,
                )
                self._copy_immutable_artifact(source, destination, digest)
            destination.chmod(0o444)
            record.provenance["diff_artifact"] = str(destination.relative_to(self.root))
            record.provenance["diff_sha256"] = digest
            record.provenance["diff_bytes"] = destination.stat().st_size

    def _materialize_record_binding(self, record: ExperimentRecord) -> None:
        payload = _record_binding_bytes(record)
        digest = _sha256_bytes(payload)
        destination = (
            self.root
            / "artifacts"
            / record.experiment_id
            / "evidence"
            / f"record-{digest}.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or not destination.is_file() or not hmac.compare_digest(
                sha256_file(destination), digest
            ):
                raise ValueError(f"workspace record binding collision: {destination}")
        else:
            try:
                with destination.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                if (
                    destination.is_symlink()
                    or not destination.is_file()
                    or not hmac.compare_digest(sha256_file(destination), digest)
                ):
                    raise ValueError(
                        f"workspace record binding collision: {destination}"
                    ) from None
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        if not hmac.compare_digest(sha256_file(destination), digest):
            destination.unlink(missing_ok=True)
            raise ValueError("canonical record binding changed while it was written")
        destination.chmod(0o444)
        record.provenance[RECORD_BINDING_ARTIFACT_KEY] = str(
            destination.relative_to(self.root)
        )
        record.provenance[RECORD_BINDING_SHA256_KEY] = digest

    @classmethod
    def initialize(cls, root: Path, config: CampaignConfig) -> CampaignWorkspace:
        workspace = cls(root)
        if workspace.config_path.exists():
            raise FileExistsError(f"campaign already exists: {workspace.config_path}")
        workspace.root.mkdir(parents=True, exist_ok=True)
        (workspace.root / "artifacts").mkdir(parents=True, exist_ok=True)
        config_payload = (_json_dumps(config.to_dict(), indent=2) + "\n").encode("utf-8")
        workspace.config_path.write_bytes(config_payload)
        (workspace.root / CONFIG_HASH_FILE).write_text(
            _sha256_bytes(config_payload) + "\n",
            encoding="utf-8",
        )
        (workspace.root / LEDGER_FILE).touch()
        (workspace.root / TEMPLATE_FILE).write_text(
            _json_dumps(experiment_template(config), indent=2) + "\n",
            encoding="utf-8",
        )
        workspace.write_reports()
        return workspace

    def load_config(self) -> CampaignConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"campaign config not found: {self.config_path}")
        config_hash_path = self.root / CONFIG_HASH_FILE
        if not config_hash_path.exists():
            raise FileNotFoundError(f"campaign config hash not found: {config_hash_path}")
        config_payload = self.config_path.read_bytes()
        expected_hash = config_hash_path.read_text(encoding="utf-8").strip().lower()
        actual_hash = _sha256_bytes(config_payload)
        if not _is_sha256(expected_hash) or not hmac.compare_digest(expected_hash, actual_hash):
            raise ValueError("campaign config changed after initialization")
        return CampaignConfig.from_dict(json.loads(config_payload))

    def record(
        self,
        record: ExperimentRecord,
        artifact_base_dir: Path | None = None,
    ) -> ExperimentRecord:
        config = self.load_config()
        if (
            record.raw_artifacts
            or record.profile_artifacts
            or record.provenance.get("diff_artifact")
        ):
            if artifact_base_dir is None:
                raise ValueError("artifact_base_dir is required to verify artifact files")
            self._materialize_record_artifacts(record, artifact_base_dir)
        if record.measurements:
            record.provenance.pop(RECORD_BINDING_ARTIFACT_KEY, None)
            record.provenance.pop(RECORD_BINDING_SHA256_KEY, None)
            integrity_errors = record_artifact_integrity_errors(
                record,
                self.root,
                require_diff_artifact=(
                    config.require_git_provenance or config.require_candidate_diff
                ),
            )
            if integrity_errors:
                raise ValueError("record evidence integrity failed: " + ", ".join(integrity_errors))
            self._materialize_record_binding(record)
            binding_errors = record_artifact_integrity_errors(
                record,
                self.root,
                require_diff_artifact=(
                    config.require_git_provenance or config.require_candidate_diff
                ),
                require_record_binding=True,
            )
            if binding_errors:
                raise ValueError("record evidence integrity failed: " + ", ".join(binding_errors))
        if record.status == "promoted":
            decision = evaluate_experiment(config, record)
            if not decision.promotable:
                raise ValueError(
                    "promoted status requires a passing mechanical gate: "
                    + ", ".join(decision.reasons)
                )
        appended = self.ledger.append(record, config=config, workspace_root=self.root)
        self.write_reports()
        return appended

    def write_reports(self) -> tuple[Path, Path]:
        config = self.load_config()
        records = self.ledger.latest()
        report_path = self.root / REPORT_FILE
        priors_path = self.root / PRIORS_FILE
        report_path.write_text(
            render_report(config, records, workspace_root=self.root),
            encoding="utf-8",
        )
        priors_path.write_text(render_priors(config, records), encoding="utf-8")
        return report_path, priors_path


def _config_from_args(args: argparse.Namespace) -> CampaignConfig:
    workload_spec = Path(args.workload_spec).resolve() if args.workload_spec else None
    environment_spec = Path(args.environment_spec).resolve() if args.environment_spec else None
    return CampaignConfig(
        objective=args.objective,
        primary_metric=args.metric,
        initial_control_commit=args.initial_control_commit,
        direction=args.direction,
        aggregate=args.aggregate,
        primary_cases=args.primary_case or [],
        frozen_cases=args.frozen_case or [],
        beam_width=args.beam_width,
        min_trials=args.min_trials,
        min_improvement_pct=args.min_improvement_pct,
        max_case_regression_pct=args.max_case_regression_pct,
        max_cv_pct=args.max_cv_pct,
        require_confidence_bounds=args.require_confidence_bounds,
        confidence_level=args.confidence_level,
        bootstrap_resamples=args.bootstrap_resamples,
        min_confidence_pairs=args.min_confidence_pairs,
        require_profile_artifact=args.require_profile_artifact,
        require_derived_evidence=args.require_derived_evidence,
        workload_spec=str(workload_spec) if workload_spec else "",
        workload_sha256=sha256_file(workload_spec) if workload_spec else "",
        environment_spec=str(environment_spec) if environment_spec else "",
        environment_sha256=sha256_file(environment_spec) if environment_spec else "",
        max_experiments=args.max_experiments,
        max_total_duration_s=args.max_total_duration_s,
        max_total_cost_usd=args.max_total_cost_usd,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage an evidence-first optimization campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a campaign workspace")
    init_parser.add_argument("workspace", type=Path)
    init_parser.add_argument("--objective", required=True)
    init_parser.add_argument("--metric", required=True)
    init_parser.add_argument("--initial-control-commit", required=True)
    init_parser.add_argument("--direction", choices=sorted(VALID_DIRECTIONS), default="lower")
    init_parser.add_argument(
        "--aggregate", choices=sorted(VALID_AGGREGATES), default="geometric_mean"
    )
    init_parser.add_argument("--primary-case", action="append", required=True)
    init_parser.add_argument("--frozen-case", action="append", required=True)
    init_parser.add_argument("--beam-width", type=int, default=4)
    init_parser.add_argument("--min-trials", type=int, default=3)
    init_parser.add_argument("--min-improvement-pct", type=float, default=1.0)
    init_parser.add_argument("--max-case-regression-pct", type=float, default=0.0)
    init_parser.add_argument("--max-cv-pct", type=float, default=5.0)
    init_parser.add_argument("--require-confidence-bounds", action="store_true")
    init_parser.add_argument("--confidence-level", type=float, default=0.95)
    init_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    init_parser.add_argument("--min-confidence-pairs", type=int, default=3)
    init_parser.add_argument("--require-profile-artifact", action="store_true")
    init_parser.add_argument("--require-derived-evidence", action="store_true")
    init_parser.add_argument("--workload-spec", required=True)
    init_parser.add_argument("--environment-spec", required=True)
    init_parser.add_argument("--max-experiments", type=int)
    init_parser.add_argument("--max-total-duration-s", type=float)
    init_parser.add_argument("--max-total-cost-usd", type=float)

    record_parser = subparsers.add_parser("record", help="Append an experiment event")
    record_parser.add_argument("workspace", type=Path)
    record_parser.add_argument("--experiment", type=Path, required=True)
    record_parser.add_argument("--repo", type=Path)
    record_parser.add_argument("--control-revision")

    evidence_parser = subparsers.add_parser(
        "record-evidence",
        help="Derive and append an experiment from paired benchmark artifacts",
    )
    evidence_parser.add_argument("workspace", type=Path)
    evidence_parser.add_argument("--experiment", type=Path, required=True)
    evidence_parser.add_argument("--evidence", type=Path, required=True)
    evidence_parser.add_argument("--repo", type=Path, required=True)
    evidence_parser.add_argument("--control-revision", required=True)

    report_parser = subparsers.add_parser("report", help="Regenerate campaign reports")
    report_parser.add_argument("workspace", type=Path)

    incumbent_parser = subparsers.add_parser(
        "incumbent", help="Show the control revision required for the next experiment"
    )
    incumbent_parser.add_argument("workspace", type=Path)
    incumbent_parser.add_argument("--json", action="store_true")

    gate_parser = subparsers.add_parser("gate", help="Evaluate one experiment for promotion")
    gate_parser.add_argument("workspace", type=Path)
    gate_parser.add_argument("experiment_id")
    gate_parser.add_argument("--json", action="store_true")

    beam_parser = subparsers.add_parser("beam", help="Show the active diverse beam")
    beam_parser.add_argument("workspace", type=Path)
    beam_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        workspace = CampaignWorkspace.initialize(args.workspace, _config_from_args(args))
        print(workspace.root)
        return 0

    workspace = CampaignWorkspace(args.workspace)
    if args.command == "record-evidence":
        from core.optimization.campaign_evidence import (
            derive_record_from_evidence_bundle,
            verify_evidence_git_binding,
        )

        draft = ExperimentRecord.from_dict(json.loads(args.experiment.read_text(encoding="utf-8")))
        record = derive_record_from_evidence_bundle(
            draft,
            workspace.load_config(),
            args.evidence,
        )
        record.provenance.update(
            capture_git_provenance(
                args.repo,
                record.changed_surface,
                workspace.root / "artifacts" / record.experiment_id,
                record.experiment_id,
                args.control_revision,
            )
        )
        verify_evidence_git_binding(record)
        appended = workspace.record(record, artifact_base_dir=args.evidence.parent)
        print(_json_dumps(appended.to_dict(), indent=2))
        return 0
    if args.command == "record":
        record = ExperimentRecord.from_dict(json.loads(args.experiment.read_text(encoding="utf-8")))
        existing_ids = {item.experiment_id for item in workspace.ledger.latest()}
        if (
            args.repo is None
            and record.experiment_id not in existing_ids
            and record.status not in {"planned", "running", "crashed"}
        ):
            raise ValueError("--repo is required for a new measured experiment")
        if args.repo:
            control_revision = args.control_revision or str(
                current_incumbent(
                    workspace.load_config(),
                    workspace.ledger.events(),
                    workspace_root=workspace.root,
                )["commit"]
            )
            record.provenance.update(
                capture_git_provenance(
                    args.repo,
                    record.changed_surface,
                    workspace.root / "artifacts" / record.experiment_id,
                    record.experiment_id,
                    control_revision,
                )
            )
        appended = workspace.record(record, artifact_base_dir=args.experiment.parent)
        print(_json_dumps(appended.to_dict(), indent=2))
        return 0
    if args.command == "report":
        report_path, priors_path = workspace.write_reports()
        print(report_path)
        print(priors_path)
        return 0

    config = workspace.load_config()
    if args.command == "incumbent":
        incumbent = current_incumbent(
            config,
            workspace.ledger.events(),
            workspace_root=workspace.root,
        )
        if args.json:
            print(_json_dumps(incumbent, indent=2))
        else:
            print(
                f"{incumbent['commit']}\t{incumbent['source']}\t{incumbent['experiment_id'] or ''}"
            )
        return 0
    if args.command == "gate":
        record = workspace.ledger.get(args.experiment_id)
        integrity_errors = record_artifact_integrity_errors(
            record,
            workspace.root,
            require_diff_artifact=bool(record.measurements)
            and (config.require_git_provenance or config.require_candidate_diff),
            require_record_binding=bool(record.measurements),
        )
        if integrity_errors:
            raise ValueError(
                f"experiment {record.experiment_id} fails evidence integrity: "
                + ", ".join(integrity_errors)
            )
        decision = evaluate_experiment(config, record)
        if args.json:
            print(_json_dumps(decision.to_dict(), indent=2))
        else:
            print(decision.decision)
            for reason in decision.reasons:
                print(f"- {reason}")
        return 0 if decision.promotable else 2
    if args.command == "beam":
        beam = active_beam(config, workspace.ledger.latest())
        if args.json:
            print(_json_dumps([record.to_dict() for record in beam], indent=2))
        else:
            for record in beam:
                print(f"{record.beam}\t{record.experiment_id}\t{record.status}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
