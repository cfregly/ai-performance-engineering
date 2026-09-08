"""Exercise declared evaluation policies through real harness execution."""

from datetime import datetime, timezone

import torch

from core.benchmark.evaluation_provenance import FeatureLineage, TemporalSplitBoundary
from tests.test_evaluation_harness_integration import EvaluationLifecycleBenchmark, _harness


def assert_evaluation_contract_controls(tmp_path, fault):
    """Accept a clean workload and reject one explicit contract violation.

    These are declared sample-ID/lineage policies, not semantic contamination
    inference. Source identity compares the actual file before and after work.
    """
    expected_codes = {
        "train_test_overlap": "sample_id_overlap",
        "holdout_overlap": "sample_id_overlap",
        "feature_label": "feature_label_lineage_overlap",
        "missing_holdout": "required_holdout_missing",
        "temporal_overlap": "temporal_split_overlap",
        "source": "protected_source_mutated",
        "threshold": "evaluator_threshold_drift",
    }

    class DeclaredEvaluation(EvaluationLifecycleBenchmark):
        violation = None

        def get_evaluation_contract(self):
            contract = super().get_evaluation_contract()
            assert contract is not None
            splits = dict(contract.sample_ids_by_split)
            if self.violation == "train_test_overlap":
                splits["test"] = splits["train"]
            elif self.violation == "holdout_overlap":
                splits["holdout"] = splits["test"]
            elif self.violation == "missing_holdout":
                splits.pop("holdout")
            contract = contract.model_copy(update={"sample_ids_by_split": splits})
            if self.violation == "feature_label":
                return contract.model_copy(update={"feature_lineage": (
                    FeatureLineage(feature="value", source_fields=("label",)),
                )})
            if self.violation == "temporal_overlap":
                return contract.model_copy(update={
                    "split_strategy": "temporal",
                    "temporal_boundaries": tuple(
                        TemporalSplitBoundary(
                            split=split,
                            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                            end=datetime(2026, 2, 1, tzinfo=timezone.utc),
                        ) for split in splits
                    ),
                })
            return contract

    for violation in (None, fault):
        source = tmp_path / f"protected-{violation}.py"
        source.write_text("THRESHOLD = 0.5\n")
        work = DeclaredEvaluation()
        work.protected_path = str(source)
        work.violation = violation
        if violation in ("source", "threshold"):
            work.mutation_phase = f"timing_{violation}"
        result = _harness().benchmark(work)
        assert result.evaluation is not None
        if violation is None:
            assert result.evaluation.status == "PASS", result.errors
            assert result.timing.iterations > 0
            torch.testing.assert_close(work.output, torch.tensor([2.0]), rtol=0, atol=0)
        else:
            assert result.evaluation.status == "FAIL"
            assert result.timing.iterations == 0
            assert expected_codes[violation] in {f.code for f in result.evaluation.failures}
