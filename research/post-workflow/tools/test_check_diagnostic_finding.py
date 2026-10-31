from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.check_diagnostic_finding import (
    DIAGNOSTIC_FINDING_PATH,
    LAYERS,
    validate_declared_diagnostic_finding,
    validate_diagnostic_finding,
)


class DiagnosticFindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        (self.package / "evidence").mkdir()
        self.refs: dict[str, dict[str, str]] = {}
        for name in (
            "phase",
            "correctness",
            "readback",
            "export",
            "cleanup",
            "qualification",
        ):
            path = self.package / "evidence" / f"{name}.json"
            path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
            self.refs[name] = {
                "path": f"evidence/{name}.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        envelope_path = self.package / "evidence/envelope.json"
        envelope_path.write_text(
            json.dumps(
                {
                    "attempts": [
                        {"id": "attempt-1", "run_id": "run-1"},
                        {"id": "attempt-2", "run_id": "run-1"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.refs["envelope"] = {
            "path": "evidence/envelope.json",
            "sha256": hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
        }
        layers = {
            name: {
                "state": "NOT_APPLICABLE",
                "observations": [],
                "evidence_refs": [],
                "gaps": [f"{name} was not exercised by this exact example"],
            }
            for name in LAYERS
        }
        for name in ("storage_and_input", "application_or_job"):
            layers[name] = {
                "state": "OBSERVED",
                "observations": [f"Observed {name} interval"],
                "evidence_refs": [self.refs["phase"]],
                "gaps": [],
            }
        self.finding = {
            "schema_version": 1,
            "kind": "piq_diagnostic_finding",
            "example_id": "storage-to-application",
            "problem": {
                "user_visible_symptom": "The application response is slow.",
                "reader_decision": "Decide whether to remove repeated storage preparation from requests.",
                "application_or_job_boundary": "First required read through completed response.",
            },
            "piq_detection": {
                "detectable_issue": "Repeated input preparation remains on every request path.",
                "observed_signal": "Storage preparation intervals recur inside the application request boundary.",
                "affected_layers": ["storage_and_input", "application_or_job"],
                "decision_enabled": "Move immutable preparation to once-per-process initialization.",
                "evidence_refs": [self.refs["phase"]],
            },
            "predicted_symptom": {
                "hypothesis": "Repeated preparation is on the request path.",
                "expected_observations": [
                    "Input intervals overlap each application request."
                ],
                "evidence_refs": [self.refs["phase"]],
            },
            "competing_explanation": {
                "hypothesis": "The inference runtime itself is slow.",
                "distinguishing_observations": [
                    "Native runtime timing remains stable."
                ],
                "evidence_refs": [self.refs["phase"]],
            },
            "fixed_change": {
                "reference_identity": "load each request",
                "candidate_identity": "load once per process",
                "sole_intended_change": "model input lifecycle",
                "held_inputs_and_settings": ["same input and runtime"],
                "unavoidable_differences": [],
            },
            "complete_outcome": {
                "start_event": "first required storage access",
                "stop_event": "completed response",
                "setup_accounting": "reported separately and in finite total",
                "cleanup_accounting": "included in finite total",
                "failed_work_accounting": "all attempts retained",
                "excluded_costs": [],
            },
            "correctness": {
                "independent_oracle": "expected answer and input digest",
                "acceptance_rule": "exact answer and digest equality",
                "covered_work": "every admitted request",
                "result": "PASS",
                "evidence": self.refs["correctness"],
            },
            "layer_evidence": layers,
            "piq_custody": {
                "source_runtime_input_identities": "source, runtime and input digests retained",
                "run_ids": ["run-1"],
                "attempt_ids": ["attempt-1", "attempt-2"],
                "artifact_refs": [self.refs["envelope"]],
                "readback": self.refs["readback"],
                "export": self.refs["export"],
                "cleanup": self.refs["cleanup"],
                "qualification_receipt": self.refs["qualification"],
                "report": None,
            },
            "disposition": {
                "value": "SUPPORTED",
                "supported_observations": [
                    "Repeated preparation was retained in the request interval."
                ],
                "remaining_unknowns": ["Physical disk traffic was unavailable."],
                "evidence_refs": [self.refs["phase"]],
            },
            "tradeoff": {
                "measured_benefit": "Request-path preparation was removed in this exact comparison.",
                "added_cost_or_responsibility": ["resident memory and invalidation"],
                "applicability": "Repeated requests for one immutable revision.",
                "simpler_option": "Load per invocation for rare isolated work.",
                "when_to_use_or_skip": "Use when requests amortize setup. Skip when lifecycle simplicity dominates.",
                "next_measurement_or_action": "Measure the representative application load.",
            },
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_finding(self) -> None:
        path = self.package / DIAGNOSTIC_FINDING_PATH
        path.write_text(json.dumps(self.finding, indent=2) + "\n", encoding="utf-8")

    def declared_manifest(self) -> dict:
        return {
            "content_state": "measured",
            "measurement_receipt": "evidence/qualification.json",
            "piq_experiment_evidence": "evidence/envelope.json",
            "diagnostic_finding_required": True,
            "diagnostic_finding": DIAGNOSTIC_FINDING_PATH,
        }

    def assert_problem(self, phrase: str) -> None:
        errors = validate_diagnostic_finding(self.package, self.finding)
        self.assertTrue(any(phrase in error for error in errors), errors)

    def test_valid_finding_passes(self) -> None:
        self.assertEqual(validate_diagnostic_finding(self.package, self.finding), [])

    def test_competing_explanation_needs_distinguishing_evidence(self) -> None:
        self.finding["competing_explanation"]["evidence_refs"] = []
        self.assert_problem("competing_explanation.evidence_refs")

    def test_piq_detection_is_a_required_cross_layer_invariant(self) -> None:
        self.finding.pop("piq_detection")
        self.assert_problem("piq_detection")

        self.finding["piq_detection"] = {
            "detectable_issue": "Repeated input preparation remains on every request path.",
            "observed_signal": "Storage preparation recurs inside the request boundary.",
            "affected_layers": ["storage_and_input"],
            "decision_enabled": "Move immutable preparation to initialization.",
            "evidence_refs": [self.refs["phase"]],
        }
        self.assert_problem("application_or_job and at least one supporting layer")

        self.finding["piq_detection"]["affected_layers"] = [
            "accelerator",
            "application_or_job",
        ]
        self.assert_problem("accelerator must have OBSERVED layer evidence")

    def test_malformed_affected_layers_fails_without_crashing(self) -> None:
        self.finding["piq_detection"]["affected_layers"] = None
        self.assert_problem("affected_layers must be a nonempty list")

    def test_application_and_supporting_layer_must_be_observed(self) -> None:
        self.finding["layer_evidence"]["application_or_job"] = {
            "state": "UNAVAILABLE",
            "observations": [],
            "evidence_refs": [],
            "gaps": ["collector failed"],
        }
        self.assert_problem("application_or_job layer must be OBSERVED")
        self.assert_problem("at least one supporting layer")

    def test_unavailable_layer_requires_reason(self) -> None:
        self.finding["layer_evidence"]["accelerator"]["gaps"] = []
        self.assert_problem(
            "UNAVAILABLE"
            if self.finding["layer_evidence"]["accelerator"]["state"] == "UNAVAILABLE"
            else "NOT_APPLICABLE"
        )

    def test_correctness_must_pass(self) -> None:
        self.finding["correctness"]["result"] = "FAIL"
        self.assert_problem("correctness.result")

    def test_inconclusive_cannot_pass_package_ready_contract(self) -> None:
        self.finding["disposition"]["value"] = "INCONCLUSIVE"
        self.assert_problem("disposition.value")

    def test_refs_are_hash_checked(self) -> None:
        self.finding["piq_custody"]["export"]["sha256"] = "0" * 64
        self.assert_problem("piq_custody.export.sha256 does not match")

    def test_ref_symlink_is_rejected(self) -> None:
        target = self.package / self.refs["export"]["path"]
        replacement = self.package / "evidence/export-replacement.json"
        replacement.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(replacement.name)
        self.assert_problem("piq_custody.export.path must not use symlinks")

    def test_historical_manifest_without_declaration_remains_accepted(self) -> None:
        self.assertEqual(validate_declared_diagnostic_finding(self.package, {}), [])

    def test_piq_evidence_requires_diagnostic_declaration(self) -> None:
        errors = validate_declared_diagnostic_finding(
            self.package,
            {"piq_experiment_evidence": "evidence/envelope.json"},
        )
        self.assertTrue(
            any("diagnostic_finding_required" in error for error in errors), errors
        )
        self.assertTrue(
            any("manifest.diagnostic_finding" in error for error in errors), errors
        )

    def test_declared_contract_requires_exact_path_and_file(self) -> None:
        manifest = self.declared_manifest()
        self.assertTrue(
            any(
                "missing diagnostic finding" in error
                for error in validate_declared_diagnostic_finding(
                    self.package, manifest
                )
            )
        )
        self.write_finding()
        self.assertEqual(
            validate_declared_diagnostic_finding(self.package, manifest), []
        )

    def test_declared_contract_binds_qualification_receipt_path(self) -> None:
        self.finding["piq_custody"]["qualification_receipt"] = self.refs["readback"]
        self.write_finding()
        errors = validate_declared_diagnostic_finding(
            self.package, self.declared_manifest()
        )
        self.assertTrue(
            any("must equal manifest.measurement_receipt" in error for error in errors),
            errors,
        )

    def test_declared_contract_binds_exact_envelope_and_census(self) -> None:
        self.finding["piq_custody"]["artifact_refs"] = [self.refs["readback"]]
        self.finding["piq_custody"]["attempt_ids"] = ["attempt-2", "attempt-1"]
        self.finding["piq_custody"]["run_ids"] = ["other-run"]
        self.write_finding()
        errors = validate_declared_diagnostic_finding(
            self.package, self.declared_manifest()
        )
        self.assertTrue(
            any("must include the exact" in error for error in errors), errors
        )

        self.finding["piq_custody"]["artifact_refs"] = [self.refs["envelope"]]
        self.write_finding()
        errors = validate_declared_diagnostic_finding(
            self.package, self.declared_manifest()
        )
        self.assertTrue(
            any("attempt_ids must exactly match" in error for error in errors), errors
        )
        self.assertTrue(
            any("run_ids must exactly match" in error for error in errors), errors
        )

    def test_declared_contract_requires_measured_package_and_piq_envelope(self) -> None:
        self.write_finding()
        manifest = self.declared_manifest()
        manifest["content_state"] = "hypothetical"
        manifest.pop("piq_experiment_evidence")
        errors = validate_declared_diagnostic_finding(self.package, manifest)
        self.assertTrue(
            any("content_state 'measured'" in error for error in errors), errors
        )
        self.assertTrue(
            any(
                "requires manifest.piq_experiment_evidence" in error for error in errors
            ),
            errors,
        )

    def test_path_without_required_flag_fails(self) -> None:
        self.write_finding()
        errors = validate_declared_diagnostic_finding(
            self.package, {"diagnostic_finding": DIAGNOSTIC_FINDING_PATH}
        )
        self.assertTrue(
            any("diagnostic_finding_required" in error for error in errors), errors
        )


if __name__ == "__main__":
    unittest.main()
