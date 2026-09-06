"""Exercise evaluation provenance through actual CPU verification work."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest
import torch

from core.benchmark.evaluation_provenance import (
    DatasetDeclaration,
    EvaluationContract,
    EvaluatorDeclaration,
    FeatureLineage,
)
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verify_runner import VerifyRunner


@dataclass
class Workload:
    requests_per_iteration: float = 1.0
    samples_per_iteration: float = 4.0
    bytes_per_iteration: float = 16.0
    tokens_per_iteration: float | None = None
    custom_units_per_iteration: float | None = None


class EvaluationBenchmark:
    _is_deterministic = True

    def __init__(self, source: Path, mutation: str | None = None):
        self.source = source
        self.mutation = mutation
        self.ran = False
        self.torn_down = False
        self.contract = EvaluationContract(
            dataset=DatasetDeclaration(
                name="tensor-evaluation", version="1",
                content_sha256=hashlib.sha256(b"tensor-evaluation-v1").hexdigest(),
            ),
            sample_ids_by_split={"train": ("a",), "test": ("b",), "holdout": ("c",)},
            split_strategy="explicit_ids",
            evaluator=EvaluatorDeclaration(name="exact", version="1", thresholds={"accuracy": 1.0}),
            feature_lineage=(FeatureLineage(feature="input", source_fields=("input",)),),
            label_fields=("expected",),
            protected_sources=(source,),
        )

    def get_evaluation_contract(self):
        if self.ran and self.mutation == "getter":
            raise RuntimeError("evaluation getter failed after execution")
        return self.contract

    def setup(self):
        self.inputs = torch.arange(4, dtype=torch.float32)

    def benchmark_fn(self):
        self.output = self.inputs * 2
        self.ran = True
        if self.mutation == "threshold":
            self.contract.evaluator.thresholds["accuracy"] = 0.1
        elif self.mutation == "source":
            self.source.write_text("changed during execution\n")
        elif self.mutation == "removed":
            self.contract = None

    def teardown(self):
        self.torn_down = True
        if self.mutation == "teardown":
            self.source.write_text("changed during teardown\n")

    def get_input_signature(self):
        return InputSignature(
            shapes={"input": (4,)}, dtypes={"input": "float32"},
            batch_size=4, parameter_count=0, precision_flags=PrecisionFlags(tf32=False),
        )

    def get_verify_inputs(self):
        if not hasattr(self, "inputs"):
            raise RuntimeError("Run setup() before extracting evaluation inputs")
        return {"input": self.inputs}

    def get_verify_output(self):
        if not self.ran:
            raise RuntimeError("Run benchmark_fn() before extracting evaluation output")
        return self.output

    def get_output_tolerance(self):
        return 0.0, 0.0

    def get_workload_metadata(self):
        return Workload()

    def validate_result(self):
        assert torch.equal(self.output, self.inputs * 2)


def _case(tmp_path, mutation=None):
    source = tmp_path / "protected.py"
    source.write_text("THRESHOLD = 1.0\n")
    benchmark = EvaluationBenchmark(source, mutation)
    runner = VerifyRunner(
        cache_dir=tmp_path / "cache",
        quarantine_manager=QuarantineManager(tmp_path / "quarantine.json"),
    )
    return runner, benchmark


def test_real_verification_records_finalized_provenance(tmp_path):
    runner, benchmark = _case(tmp_path)
    result = runner.verify_baseline(benchmark)
    assert result.passed, result.reason
    assert benchmark.ran and benchmark.torn_down
    receipt = result.details["evaluation_provenance"]
    assert receipt["status"] == "PASS"
    assert receipt["finalized_at"] is not None


@pytest.mark.parametrize("mutation", ["threshold", "source", "teardown", "removed", "getter"])
def test_real_verification_rejects_evaluation_drift(tmp_path, mutation):
    runner, benchmark = _case(tmp_path, mutation)
    result = runner.verify_baseline(benchmark)
    assert not result.passed
    assert "evaluation" in result.reason.lower()
    assert benchmark.ran and benchmark.torn_down
    assert result.details["evaluation_provenance"]["status"] == "FAIL"


def test_real_verification_rejects_overlap_before_execution(tmp_path):
    runner, benchmark = _case(tmp_path)
    benchmark.contract.sample_ids_by_split["test"] = ("a",)
    result = runner.verify_baseline(benchmark)
    assert not result.passed
    assert "overlap" in result.reason.lower()
    assert not benchmark.ran


def test_ordinary_verification_does_not_require_dataset_contract(tmp_path):
    runner, benchmark = _case(tmp_path)
    benchmark.contract = None
    result = runner.verify_baseline(benchmark)
    assert result.passed, result.reason
    assert result.details is None
