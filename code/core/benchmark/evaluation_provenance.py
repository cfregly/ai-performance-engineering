"""Explicit provenance contract for workloads that make evaluation claims.

Kernel and systems microbenchmarks do not opt in to this contract. Evaluation
workloads provide auditable sample identifiers, split policy, evaluator settings,
feature lineage, and protected source files through :class:`EvaluationContract`.
The serialized provenance retains digests and counts rather than raw sample IDs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0"
_SHA256_HEX_LENGTH = 64
_SPLIT_ORDER = ("train", "test", "holdout")


class DatasetDeclaration(BaseModel):
    """Identity declared by an evaluation workload for its dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    content_sha256: str


class TemporalSplitBoundary(BaseModel):
    """Half-open time interval assigned to one named data split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str
    start: datetime
    end: datetime


class EvaluatorDeclaration(BaseModel):
    """Evaluator identity and thresholds fixed before evaluation starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    thresholds: Mapping[str, float]


class FeatureLineage(BaseModel):
    """Workload-declared source fields and transforms for one feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    source_fields: tuple[str, ...]
    transforms: tuple[str, ...] = ()


class EvaluationContract(BaseModel):
    """Opt-in declaration supplied only by workloads making evaluation claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["evaluation"] = "evaluation"
    dataset: DatasetDeclaration
    sample_ids_by_split: Mapping[str, tuple[str, ...]]
    required_holdout: bool = True
    split_strategy: Literal["explicit_ids", "temporal"] = "explicit_ids"
    temporal_boundaries: tuple[TemporalSplitBoundary, ...] = ()
    evaluator: EvaluatorDeclaration
    feature_lineage: tuple[FeatureLineage, ...]
    label_fields: tuple[str, ...] = ()
    protected_sources: tuple[Path, ...]


class EvaluationFailure(BaseModel):
    """Machine-readable rejection or limitation discovered by the contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    phase: Literal["start", "finalize"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SplitIdentity(BaseModel):
    """Privacy-preserving identity for a declared sample split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str
    sample_count: int
    unique_sample_count: int
    sample_ids_sha256: str


class ProtectedSourceIdentity(BaseModel):
    """Content identity for a regular, non-symlink protected source file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    bytes: int
    sha256: str


class EvaluationProvenanceError(RuntimeError):
    """Raised when an evaluation provenance receipt contains failures."""

    def __init__(self, failures: tuple[EvaluationFailure, ...]):
        self.failures = failures
        codes = ", ".join(failure.code for failure in failures)
        super().__init__(f"evaluation provenance rejected: {codes}")


class EvaluationProvenance(BaseModel):
    """Serializable start/finalize receipt for an evaluation contract."""

    model_config = ConfigDict(extra="forbid")

    declared: Literal[True] = True
    status: Literal["PENDING", "PASS", "FAIL"]
    dataset: DatasetDeclaration
    splits: tuple[SplitIdentity, ...]
    required_holdout: bool
    split_strategy: Literal["explicit_ids", "temporal"]
    temporal_boundaries: tuple[TemporalSplitBoundary, ...]
    evaluator: EvaluatorDeclaration
    evaluator_thresholds_sha256: str
    feature_lineage: tuple[FeatureLineage, ...]
    feature_lineage_sha256: str
    label_fields: tuple[str, ...]
    protected_source_paths: tuple[str, ...]
    protected_sources_before: tuple[ProtectedSourceIdentity, ...]
    protected_sources_after: tuple[ProtectedSourceIdentity, ...] = ()
    contract_sha256: str
    started_at: datetime
    finalized_at: Optional[datetime] = None
    failures: list[EvaluationFailure] = Field(default_factory=list)
    limitations: tuple[str, ...] = (
        "feature_lineage_is_workload_declared_not_independently_inferred",
        "content_hashes_do_not_detect_semantic_dataset_contamination",
        "before_after_source_hashes_do_not_detect_transient_restored_mutation",
    )
    schemaVersion: str = SCHEMA_VERSION

    def raise_for_failures(self) -> None:
        """Fail the caller without discarding the structured receipt."""

        if self.failures:
            raise EvaluationProvenanceError(tuple(self.failures))


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_digest(contract: EvaluationContract) -> str:
    return _canonical_digest(contract.model_dump(mode="json"))


def _sample_id_digest(sample_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _failure(
    code: str,
    phase: Literal["start", "finalize"],
    message: str,
    **details: Any,
) -> EvaluationFailure:
    return EvaluationFailure(
        code=code,
        phase=phase,
        message=message,
        details=details,
    )


def _validate_declaration(
    contract: EvaluationContract,
    *,
    phase: Literal["start", "finalize"],
) -> list[EvaluationFailure]:
    failures: list[EvaluationFailure] = []
    dataset = contract.dataset
    if not dataset.name.strip() or not dataset.version.strip():
        failures.append(
            _failure(
                "dataset_identity_missing",
                phase,
                "dataset name and version must be non-empty",
            )
        )
    if not _is_sha256(dataset.content_sha256):
        failures.append(
            _failure(
                "dataset_content_digest_invalid",
                phase,
                "dataset content_sha256 must be lowercase SHA-256",
            )
        )

    splits = {
        str(name): tuple(sample_ids) for name, sample_ids in contract.sample_ids_by_split.items()
    }
    required_splits = ["train", "test"]
    if contract.required_holdout:
        required_splits.append("holdout")
    for name in required_splits:
        if not splits.get(name):
            code = "required_holdout_missing" if name == "holdout" else "required_split_missing"
            failures.append(
                _failure(
                    code,
                    phase,
                    f"required evaluation split is missing or empty: {name}",
                    split=name,
                )
            )

    for name, sample_ids in sorted(splits.items()):
        invalid_count = sum(
            not isinstance(sample_id, str) or not sample_id.strip() for sample_id in sample_ids
        )
        if invalid_count:
            failures.append(
                _failure(
                    "sample_id_invalid",
                    phase,
                    "sample IDs must be non-empty strings",
                    split=name,
                    invalid_count=invalid_count,
                )
            )
        duplicate_count = len(sample_ids) - len(set(sample_ids))
        if duplicate_count:
            failures.append(
                _failure(
                    "duplicate_sample_id",
                    phase,
                    "sample IDs must be unique within each split",
                    split=name,
                    duplicate_count=duplicate_count,
                )
            )

    split_names = sorted(splits)
    for index, left_name in enumerate(split_names):
        left_ids = set(splits[left_name])
        for right_name in split_names[index + 1 :]:
            overlap = left_ids.intersection(splits[right_name])
            if overlap:
                failures.append(
                    _failure(
                        "sample_id_overlap",
                        phase,
                        "sample IDs overlap across evaluation splits",
                        left_split=left_name,
                        right_split=right_name,
                        overlap_count=len(overlap),
                        overlap_ids_sha256=_sample_id_digest(tuple(overlap)),
                    )
                )

    boundaries: dict[str, TemporalSplitBoundary] = {}
    for boundary in contract.temporal_boundaries:
        if boundary.split in boundaries:
            failures.append(
                _failure(
                    "temporal_boundary_duplicate",
                    phase,
                    "each split may declare only one temporal boundary",
                    split=boundary.split,
                )
            )
            continue
        boundaries[boundary.split] = boundary
        if boundary.start.tzinfo is None or boundary.end.tzinfo is None:
            failures.append(
                _failure(
                    "temporal_boundary_timezone_missing",
                    phase,
                    "temporal boundaries must be timezone-aware",
                    split=boundary.split,
                )
            )
        elif boundary.start >= boundary.end:
            failures.append(
                _failure(
                    "temporal_boundary_invalid",
                    phase,
                    "temporal split start must precede its end",
                    split=boundary.split,
                )
            )
    if contract.split_strategy == "temporal":
        for name in required_splits:
            if name not in boundaries:
                failures.append(
                    _failure(
                        "temporal_boundary_missing",
                        phase,
                        "temporal strategy requires a boundary for every required split",
                        split=name,
                    )
                )
        available = [name for name in _SPLIT_ORDER if name in boundaries]
        for left_name, right_name in zip(available, available[1:]):
            left = boundaries[left_name]
            right = boundaries[right_name]
            if (
                left.end.tzinfo is not None
                and right.start.tzinfo is not None
                and left.end > right.start
            ):
                failures.append(
                    _failure(
                        "temporal_split_overlap",
                        phase,
                        "temporal split boundaries overlap",
                        left_split=left_name,
                        right_split=right_name,
                    )
                )

    thresholds = contract.evaluator.thresholds
    if not contract.evaluator.name.strip() or not contract.evaluator.version.strip():
        failures.append(
            _failure(
                "evaluator_identity_missing",
                phase,
                "evaluator name and version must be non-empty",
            )
        )
    if not thresholds:
        failures.append(
            _failure(
                "evaluator_thresholds_missing",
                phase,
                "evaluation workloads must fix evaluator thresholds before timing",
            )
        )
    for name, value in thresholds.items():
        if not str(name).strip() or isinstance(value, bool) or not math.isfinite(float(value)):
            failures.append(
                _failure(
                    "evaluator_threshold_invalid",
                    phase,
                    "evaluator threshold names and numeric values must be finite",
                    threshold=str(name),
                )
            )

    if not contract.feature_lineage:
        failures.append(
            _failure(
                "feature_lineage_missing",
                phase,
                "evaluation workloads must declare feature lineage",
            )
        )
    feature_names: set[str] = set()
    label_fields = set(contract.label_fields)
    for lineage in contract.feature_lineage:
        if not lineage.feature.strip() or not lineage.source_fields:
            failures.append(
                _failure(
                    "feature_lineage_invalid",
                    phase,
                    "each feature needs a name and at least one declared source field",
                    feature=lineage.feature,
                )
            )
        if lineage.feature in feature_names:
            failures.append(
                _failure(
                    "feature_lineage_duplicate",
                    phase,
                    "feature lineage names must be unique",
                    feature=lineage.feature,
                )
            )
        feature_names.add(lineage.feature)
        label_overlap = sorted(set(lineage.source_fields).intersection(label_fields))
        if label_overlap:
            failures.append(
                _failure(
                    "feature_label_lineage_overlap",
                    phase,
                    "declared feature sources include a declared label field",
                    feature=lineage.feature,
                    label_fields=label_overlap,
                )
            )

    if not contract.protected_sources:
        failures.append(
            _failure(
                "protected_sources_missing",
                phase,
                "evaluation workloads must declare protected test sources",
            )
        )
    return failures


def _capture_source_identity(path: Path) -> ProtectedSourceIdentity:
    declared = path.expanduser()
    if not declared.is_absolute():
        raise ValueError("protected source path must be absolute")
    before = declared.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("protected source must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(declared, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        before_key = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_key = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if before_key != opened_key:
            raise ValueError("protected source changed while opening")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_key = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if opened_key != after_key:
            raise ValueError("protected source changed while hashing")
    finally:
        os.close(descriptor)
    return ProtectedSourceIdentity(
        path=str(declared),
        bytes=before.st_size,
        sha256=digest.hexdigest(),
    )


def _capture_sources(
    paths: tuple[Path, ...],
    *,
    phase: Literal["start", "finalize"],
) -> tuple[tuple[ProtectedSourceIdentity, ...], list[EvaluationFailure]]:
    identities: list[ProtectedSourceIdentity] = []
    failures: list[EvaluationFailure] = []
    seen: set[str] = set()
    for path in paths:
        declared = str(path.expanduser())
        if declared in seen:
            failures.append(
                _failure(
                    "protected_source_duplicate",
                    phase,
                    "protected source paths must be unique",
                    path=declared,
                )
            )
            continue
        seen.add(declared)
        try:
            identities.append(_capture_source_identity(path))
        except (OSError, ValueError) as exc:
            failures.append(
                _failure(
                    "protected_source_unavailable",
                    phase,
                    "protected source identity could not be captured",
                    path=declared,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(identities), failures


def start_evaluation(contract: EvaluationContract) -> EvaluationProvenance:
    """Validate and snapshot an explicit evaluation contract before setup/timing."""

    failures = _validate_declaration(contract, phase="start")
    sources_before, source_failures = _capture_sources(
        tuple(contract.protected_sources),
        phase="start",
    )
    failures.extend(source_failures)
    splits = tuple(
        SplitIdentity(
            split=name,
            sample_count=len(sample_ids),
            unique_sample_count=len(set(sample_ids)),
            sample_ids_sha256=_sample_id_digest(tuple(sample_ids)),
        )
        for name, sample_ids in sorted(contract.sample_ids_by_split.items())
    )
    evaluator = contract.evaluator.model_copy(deep=True)
    feature_lineage = tuple(item.model_copy(deep=True) for item in contract.feature_lineage)
    return EvaluationProvenance(
        status="FAIL" if failures else "PENDING",
        dataset=contract.dataset.model_copy(deep=True),
        splits=splits,
        required_holdout=contract.required_holdout,
        split_strategy=contract.split_strategy,
        temporal_boundaries=tuple(
            boundary.model_copy(deep=True) for boundary in contract.temporal_boundaries
        ),
        evaluator=evaluator,
        evaluator_thresholds_sha256=_canonical_digest(dict(evaluator.thresholds)),
        feature_lineage=feature_lineage,
        feature_lineage_sha256=_canonical_digest(
            [item.model_dump(mode="json") for item in feature_lineage]
        ),
        label_fields=tuple(contract.label_fields),
        protected_source_paths=tuple(str(path.expanduser()) for path in contract.protected_sources),
        protected_sources_before=sources_before,
        contract_sha256=_contract_digest(contract),
        started_at=datetime.now(timezone.utc),
        failures=failures,
    )


def finalize_evaluation(
    provenance: EvaluationProvenance,
    current_contract: Optional[EvaluationContract],
) -> EvaluationProvenance:
    """Capture post-run identity and reject declaration or threshold drift."""

    result = provenance.model_copy(deep=True)
    if result.finalized_at is not None:
        result.failures.append(
            _failure(
                "evaluation_already_finalized",
                "finalize",
                "evaluation provenance may be finalized only once",
            )
        )
        result.status = "FAIL"
        return result

    if current_contract is None:
        result.failures.append(
            _failure(
                "evaluation_contract_missing_at_finalize",
                "finalize",
                "the opt-in evaluation contract was unavailable after execution",
            )
        )
    else:
        result.failures.extend(_validate_declaration(current_contract, phase="finalize"))
        current_thresholds = dict(current_contract.evaluator.thresholds)
        if current_thresholds != dict(result.evaluator.thresholds):
            result.failures.append(
                _failure(
                    "evaluator_threshold_drift",
                    "finalize",
                    "evaluator thresholds changed after the pre-run snapshot",
                    before_sha256=result.evaluator_thresholds_sha256,
                    after_sha256=_canonical_digest(current_thresholds),
                )
            )
        current_digest = _contract_digest(current_contract)
        if current_digest != result.contract_sha256:
            result.failures.append(
                _failure(
                    "evaluation_contract_drift",
                    "finalize",
                    "the evaluation declaration changed during execution",
                    before_sha256=result.contract_sha256,
                    after_sha256=current_digest,
                )
            )

    source_paths = tuple(Path(path) for path in result.protected_source_paths)
    sources_after, source_failures = _capture_sources(
        source_paths,
        phase="finalize",
    )
    result.protected_sources_after = sources_after
    result.failures.extend(source_failures)
    before_by_path = {identity.path: identity for identity in result.protected_sources_before}
    after_by_path = {identity.path: identity for identity in result.protected_sources_after}
    for path, before in before_by_path.items():
        after = after_by_path.get(path)
        if after is not None and (before.sha256 != after.sha256 or before.bytes != after.bytes):
            result.failures.append(
                _failure(
                    "protected_source_mutated",
                    "finalize",
                    "protected evaluation source changed during execution",
                    path=path,
                    before_sha256=before.sha256,
                    after_sha256=after.sha256,
                    before_bytes=before.bytes,
                    after_bytes=after.bytes,
                )
            )
    result.finalized_at = datetime.now(timezone.utc)
    result.status = "FAIL" if result.failures else "PASS"
    return result


__all__ = [
    "DatasetDeclaration",
    "EvaluationContract",
    "EvaluationFailure",
    "EvaluationProvenance",
    "EvaluationProvenanceError",
    "EvaluatorDeclaration",
    "FeatureLineage",
    "ProtectedSourceIdentity",
    "SplitIdentity",
    "TemporalSplitBoundary",
    "finalize_evaluation",
    "start_evaluation",
]
