from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from core.benchmark.evaluation_provenance import (
    DatasetDeclaration,
    EvaluationContract,
    EvaluationProvenanceError,
    EvaluatorDeclaration,
    FeatureLineage,
    TemporalSplitBoundary,
    finalize_evaluation,
    start_evaluation,
)
from core.benchmark.run_manifest import (
    EnvironmentInfo,
    GitInfo,
    HardwareInfo,
    RunManifest,
    SoftwareInfo,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contract(
    protected_source: Path,
    *,
    sample_ids_by_split: dict[str, tuple[str, ...]] | None = None,
    thresholds: dict[str, float] | None = None,
    split_strategy: str = "temporal",
    temporal_boundaries: tuple[TemporalSplitBoundary, ...] | None = None,
) -> EvaluationContract:
    boundaries = temporal_boundaries
    if boundaries is None:
        boundaries = (
            TemporalSplitBoundary(
                split="train",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            TemporalSplitBoundary(
                split="test",
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            TemporalSplitBoundary(
                split="holdout",
                start=datetime(2024, 3, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc),
            ),
        )
    return EvaluationContract(
        dataset=DatasetDeclaration(
            name="fraud-events",
            version="2024-04",
            content_sha256=_sha256(b"declared dataset fixture"),
        ),
        sample_ids_by_split=sample_ids_by_split
        or {
            "train": ("train-001", "train-002"),
            "test": ("test-001",),
            "holdout": ("holdout-001",),
        },
        required_holdout=True,
        split_strategy=split_strategy,
        temporal_boundaries=boundaries,
        evaluator=EvaluatorDeclaration(
            name="binary-classification",
            version="2",
            thresholds=thresholds or {"minimum_accuracy": 0.90, "maximum_fpr": 0.05},
        ),
        feature_lineage=(
            FeatureLineage(
                feature="account_age_days",
                source_fields=("account_created_at", "event_at"),
                transforms=("elapsed_days",),
            ),
        ),
        label_fields=("is_fraud",),
        protected_sources=(protected_source,),
    )


def _manifest() -> RunManifest:
    return RunManifest(
        hardware=HardwareInfo(),
        software=SoftwareInfo(
            pytorch_version="test-runtime",
            python_version="test-runtime",
            os="test-runtime",
        ),
        environment=EnvironmentInfo(cuda_visible_devices=""),
        git=GitInfo(commit="a" * 40, branch="test", dirty=False),
        start_time=datetime.now(),
    )


def _failure_codes(receipt) -> set[str]:
    return {failure.code for failure in receipt.failures}


def test_clean_evaluation_serializes_counts_and_digests_without_sample_ids(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text('{"sample": 1}\n', encoding="utf-8")
    contract = _contract(protected)

    receipt = start_evaluation(contract)
    assert receipt.status == "PENDING"
    receipt.raise_for_failures()
    receipt = finalize_evaluation(receipt, contract)
    receipt.raise_for_failures()

    assert receipt.status == "PASS"
    assert receipt.protected_sources_before == receipt.protected_sources_after
    assert {split.split: split.sample_count for split in receipt.splits} == {
        "holdout": 1,
        "test": 1,
        "train": 2,
    }
    payload = json.dumps(receipt.model_dump(mode="json"), sort_keys=True)
    for raw_sample_id in ("train-001", "train-002", "test-001", "holdout-001"):
        assert raw_sample_id not in payload
    assert "content_hashes_do_not_detect_semantic_dataset_contamination" in payload


def test_one_sample_overlap_is_a_structured_start_failure(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    contract = _contract(
        protected,
        sample_ids_by_split={
            "train": ("train-001", "shared-secret-id"),
            "test": ("shared-secret-id",),
            "holdout": ("holdout-001",),
        },
    )

    receipt = start_evaluation(contract)

    assert receipt.status == "FAIL"
    assert "sample_id_overlap" in _failure_codes(receipt)
    assert "shared-secret-id" not in json.dumps(receipt.model_dump(mode="json"))
    with pytest.raises(EvaluationProvenanceError, match="sample_id_overlap"):
        receipt.raise_for_failures()


def test_required_holdout_and_temporal_boundary_are_enforced(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    contract = _contract(
        protected,
        sample_ids_by_split={"train": ("a",), "test": ("b",)},
        temporal_boundaries=(
            TemporalSplitBoundary(
                split="train",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            TemporalSplitBoundary(
                split="test",
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
        ),
    )

    receipt = start_evaluation(contract)

    assert {"required_holdout_missing", "temporal_boundary_missing"}.issubset(
        _failure_codes(receipt)
    )


def test_overlapping_temporal_boundaries_are_rejected(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    contract = _contract(
        protected,
        temporal_boundaries=(
            TemporalSplitBoundary(
                split="train",
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            TemporalSplitBoundary(
                split="test",
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc),
            ),
            TemporalSplitBoundary(
                split="holdout",
                start=datetime(2024, 4, 1, tzinfo=timezone.utc),
                end=datetime(2024, 5, 1, tzinfo=timezone.utc),
            ),
        ),
    )

    receipt = start_evaluation(contract)

    assert "temporal_split_overlap" in _failure_codes(receipt)


def test_evaluator_threshold_drift_is_rejected_at_finalize(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    original = _contract(protected)
    changed = _contract(
        protected,
        thresholds={"minimum_accuracy": 0.80, "maximum_fpr": 0.05},
    )

    receipt = finalize_evaluation(start_evaluation(original), changed)

    assert receipt.status == "FAIL"
    assert {"evaluator_threshold_drift", "evaluation_contract_drift"}.issubset(
        _failure_codes(receipt)
    )


def test_declared_feature_label_overlap_is_rejected_without_claiming_inference(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    clean = _contract(protected)
    contract = clean.model_copy(
        update={
            "feature_lineage": (
                FeatureLineage(feature="leaked_label", source_fields=("is_fraud",)),
            )
        }
    )

    receipt = start_evaluation(contract)

    assert "feature_label_lineage_overlap" in _failure_codes(receipt)
    assert (
        "feature_lineage_is_workload_declared_not_independently_inferred"
        in receipt.limitations
    )


def test_protected_source_mutation_is_rejected_at_finalize(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("before\n", encoding="utf-8")
    contract = _contract(protected)
    receipt = start_evaluation(contract)

    protected.write_text("after with different content\n", encoding="utf-8")
    receipt = finalize_evaluation(receipt, contract)

    assert receipt.status == "FAIL"
    assert "protected_source_mutated" in _failure_codes(receipt)
    assert receipt.protected_sources_before[0].sha256 != receipt.protected_sources_after[0].sha256


def test_removed_contract_is_rejected_and_manifest_serializes_receipt(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    manifest = _manifest()
    manifest.begin_evaluation(_contract(protected)).raise_for_failures()

    receipt = manifest.finalize_evaluation(None)

    assert receipt is not None
    assert receipt.status == "FAIL"
    assert "evaluation_contract_missing_at_finalize" in _failure_codes(receipt)
    serialized = manifest.model_dump(mode="json")
    assert serialized["evaluation"]["status"] == "FAIL"


def test_manifest_finalize_fail_closes_unfinished_evaluation(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    manifest = _manifest()
    manifest.begin_evaluation(_contract(protected)).raise_for_failures()

    manifest.finalize()

    assert manifest.evaluation is not None
    assert "evaluation_contract_missing_at_finalize" in _failure_codes(manifest.evaluation)


def test_kernel_manifest_has_no_invented_evaluation_semantics() -> None:
    manifest = _manifest()

    manifest.finalize()

    assert manifest.evaluation is None
    assert manifest.model_dump(mode="json")["evaluation"] is None


def test_protected_source_symlink_is_rejected(tmp_path: Path) -> None:
    protected = tmp_path / "holdout.jsonl"
    protected.write_text("immutable\n", encoding="utf-8")
    alias = tmp_path / "holdout-link.jsonl"
    alias.symlink_to(protected)

    receipt = start_evaluation(_contract(alias))

    assert receipt.status == "FAIL"
    assert "protected_source_unavailable" in _failure_codes(receipt)
