"""CPU-only tests for the optimization campaign evidence contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from core.optimization.campaign import (
    CampaignConfig,
    CampaignWorkspace,
    CaseMeasurement,
    ExperimentLedger,
    ExperimentRecord,
    active_beam,
    budget_status,
    capture_git_provenance,
    capture_local_artifact_hashes,
    current_incumbent,
    evaluate_experiment,
    promotion_frontier,
    sha256_file,
)
from core.optimization.campaign_dashboard import (
    build_campaign_dashboard,
    resolve_campaign_artifact,
)

WORKLOAD_HASH = "a" * 64
ENVIRONMENT_HASH = "b" * 64
ARTIFACT_CONTENT = b"campaign test artifact\n"
ARTIFACT_HASH = hashlib.sha256(ARTIFACT_CONTENT).hexdigest()
DIFF_CONTENT = b"diff --git a/kernel.py b/kernel.py\n"
DIFF_HASH = hashlib.sha256(DIFF_CONTENT).hexdigest()
INITIAL_CONTROL_COMMIT = "d" * 40
CANDIDATE_COMMIT = "e" * 40


def _config(**overrides: object) -> CampaignConfig:
    values: dict[str, object] = {
        "objective": "Reduce representative latency without regressing guard cases.",
        "primary_metric": "latency_ms",
        "initial_control_commit": INITIAL_CONTROL_COMMIT,
        "direction": "lower",
        "primary_cases": ["common"],
        "frozen_cases": ["common", "edge"],
        "min_trials": 3,
        "min_improvement_pct": 2.0,
        "max_case_regression_pct": 0.5,
        "workload_spec": "/frozen/workload.yaml",
        "workload_sha256": WORKLOAD_HASH,
        "environment_spec": "/frozen/run-manifest.json",
        "environment_sha256": ENVIRONMENT_HASH,
    }
    values.update(overrides)
    return CampaignConfig(**values)


def _provenance(diff_bytes: int = 42) -> dict[str, object]:
    return {
        "repo_root": "/repo",
        "git_commit": CANDIDATE_COMMIT,
        "control_commit": INITIAL_CONTROL_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "git_branch": "candidate-worktree",
        "diff_artifact": "artifacts/candidate.diff",
        "diff_sha256": DIFF_HASH,
        "diff_bytes": diff_bytes,
    }


def _record(
    experiment_id: str = "exp-001",
    beam: str = "structural",
    status: str = "completed",
    common_candidate: float = 90.0,
    edge_candidate: float = 100.0,
    **overrides: object,
) -> ExperimentRecord:
    values: dict[str, object] = {
        "experiment_id": experiment_id,
        "parent_id": INITIAL_CONTROL_COMMIT,
        "beam": beam,
        "hypothesis": "Remove one repeated launch from the hot path.",
        "status": status,
        "measurements": {
            "common": CaseMeasurement(
                control=[100.0, 100.0, 100.0],
                candidate=[common_candidate] * 3,
            ),
            "edge": CaseMeasurement(
                control=[100.0, 100.0, 100.0],
                candidate=[edge_candidate] * 3,
            ),
        },
        "correctness": "passed",
        "measurement_protocol": "interleaved",
        "mechanism": "The profile shows one fewer launch per invocation.",
        "code_audit": "Only the launch path changed. Output checks are unchanged.",
        "raw_artifacts": ["artifacts/raw.json"],
        "artifact_sha256": {"artifacts/raw.json": ARTIFACT_HASH},
        "workload_sha256": WORKLOAD_HASH,
        "environment_sha256": ENVIRONMENT_HASH,
        "provenance": _provenance(),
    }
    values.update(overrides)
    return ExperimentRecord(**values)


def _workspace_record(
    workspace: CampaignWorkspace,
    record: ExperimentRecord,
) -> ExperimentRecord:
    for artifact in [*record.raw_artifacts, *record.profile_artifacts]:
        artifact_path = Path(artifact)
        if artifact_path.is_absolute() or "://" in artifact:
            continue
        resolved = workspace.root / artifact_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(ARTIFACT_CONTENT)
    diff_artifact = str(record.provenance.get("diff_artifact") or "")
    if diff_artifact and "://" not in diff_artifact:
        diff_path = Path(diff_artifact)
        if not diff_path.is_absolute():
            diff_path = workspace.root / diff_path
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_bytes(DIFF_CONTENT)
    return workspace.record(record, artifact_base_dir=workspace.root)


class PromotionGateTests(unittest.TestCase):
    def test_valid_candidate_promotes(self) -> None:
        decision = evaluate_experiment(_config(), _record())

        self.assertEqual(decision.decision, "promote")
        self.assertAlmostEqual(decision.improvement_pct or 0.0, 10.0)
        self.assertEqual(decision.minimum_trials, 3)

    def test_required_paired_bootstrap_bounds_are_reported_deterministically(self) -> None:
        config = _config(
            require_confidence_bounds=True,
            bootstrap_resamples=500,
        )

        first = evaluate_experiment(config, _record())
        second = evaluate_experiment(config, _record())

        self.assertEqual(first.decision, "promote")
        self.assertAlmostEqual((first.improvement_ci_pct or [0.0])[0], 10.0)
        self.assertAlmostEqual((first.improvement_ci_pct or [0.0, 0.0])[1], 10.0)
        self.assertEqual(first.case_improvement_ci_pct["edge"], [0.0, 0.0])
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.confidence_method, "paired_bootstrap_percentile")
        self.assertTrue(first.confidence_required)

    def test_primary_confidence_lower_bound_blocks_an_uncertain_point_win(self) -> None:
        record = _record()
        record.measurements["common"] = CaseMeasurement(
            control=[100.0] * 5,
            candidate=[90.0, 90.0, 95.0, 101.0, 101.0],
        )
        record.measurements["edge"] = CaseMeasurement(
            control=[100.0] * 5,
            candidate=[100.0] * 5,
        )

        decision = evaluate_experiment(
            _config(
                min_trials=5,
                max_cv_pct=None,
                require_confidence_bounds=True,
                bootstrap_resamples=2_000,
            ),
            record,
        )

        self.assertAlmostEqual(decision.improvement_pct or 0.0, 5.0)
        self.assertLess((decision.improvement_ci_pct or [0.0])[0], 2.0)
        self.assertEqual(decision.decision, "park")
        self.assertTrue(any("confidence lower bound" in reason for reason in decision.reasons))

    def test_frozen_case_confidence_lower_bound_blocks_uncertain_guard(self) -> None:
        record = _record()
        record.measurements["common"] = CaseMeasurement(
            control=[100.0] * 5,
            candidate=[90.0] * 5,
        )
        record.measurements["edge"] = CaseMeasurement(
            control=[100.0] * 5,
            candidate=[99.0, 99.0, 100.0, 102.0, 102.0],
        )

        decision = evaluate_experiment(
            _config(
                min_trials=5,
                max_cv_pct=None,
                require_confidence_bounds=True,
                bootstrap_resamples=2_000,
            ),
            record,
        )

        self.assertEqual(decision.case_improvements_pct["edge"], 0.0)
        self.assertLess(decision.case_improvement_ci_pct["edge"][0], -0.5)
        self.assertEqual(decision.decision, "park")
        self.assertTrue(any("edge" in reason for reason in decision.reasons))

    def test_unattended_gate_requires_enough_paired_samples(self) -> None:
        record = _record()
        for case_id in ("common", "edge"):
            record.measurements[case_id] = CaseMeasurement(
                control=[100.0, 100.0],
                candidate=[90.0, 90.0],
            )

        decision = evaluate_experiment(
            _config(
                min_trials=2,
                require_derived_evidence=True,
                min_confidence_pairs=3,
                bootstrap_resamples=500,
            ),
            record,
        )

        self.assertEqual(decision.decision, "inconclusive")
        self.assertTrue(decision.confidence_required)
        self.assertTrue(any("at least 3 pairs" in reason for reason in decision.reasons))

    def test_frozen_case_regression_blocks_an_aggregate_win(self) -> None:
        decision = evaluate_experiment(
            _config(),
            _record(common_candidate=80.0, edge_candidate=101.0),
        )

        self.assertEqual(decision.decision, "park")
        self.assertAlmostEqual(decision.improvement_pct or 0.0, 20.0)
        self.assertTrue(any("edge" in reason for reason in decision.reasons))

    def test_guard_cases_do_not_change_the_primary_aggregate(self) -> None:
        decision = evaluate_experiment(
            _config(max_case_regression_pct=100.0),
            _record(common_candidate=80.0, edge_candidate=150.0),
        )

        self.assertAlmostEqual(decision.improvement_pct or 0.0, 20.0)

    def test_correctness_failure_rejects_before_timing(self) -> None:
        decision = evaluate_experiment(
            _config(),
            _record(correctness="failed"),
        )

        self.assertEqual(decision.decision, "reject")
        self.assertEqual(decision.reasons, ["correctness failed"])

    def test_missing_trials_is_inconclusive(self) -> None:
        record = _record()
        record.measurements["edge"] = CaseMeasurement(
            control=[100.0, 100.0],
            candidate=[99.0, 99.0],
        )

        decision = evaluate_experiment(_config(), record)

        self.assertEqual(decision.decision, "inconclusive")
        self.assertTrue(any("trial count" in reason for reason in decision.reasons))

    def test_missing_hash_is_not_filled_in(self) -> None:
        record = _record(workload_sha256="")

        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(workspace, record)
            stored = workspace.ledger.get(record.experiment_id)

        self.assertEqual(stored.workload_sha256, "")
        decision = evaluate_experiment(_config(), stored)
        self.assertTrue(any("workload hash" in reason for reason in decision.reasons))

    def test_missing_or_empty_diff_blocks_promotion(self) -> None:
        decision = evaluate_experiment(
            _config(),
            _record(provenance=_provenance(diff_bytes=0)),
        )

        self.assertEqual(decision.decision, "park")
        self.assertTrue(any("candidate diff" in reason for reason in decision.reasons))

    def test_unbalanced_interleaved_trials_are_inconclusive(self) -> None:
        record = _record()
        record.measurements["edge"] = CaseMeasurement(
            control=[100.0, 100.0, 100.0],
            candidate=[99.0, 99.0, 99.0, 99.0],
        )

        decision = evaluate_experiment(_config(), record)

        self.assertEqual(decision.decision, "inconclusive")
        self.assertTrue(any("trial counts differ" in reason for reason in decision.reasons))

    def test_missing_artifact_hash_blocks_promotion(self) -> None:
        decision = evaluate_experiment(
            _config(),
            _record(artifact_sha256={}),
        )

        self.assertEqual(decision.decision, "park")
        self.assertTrue(any("artifact SHA-256" in reason for reason in decision.reasons))

    def test_unsafe_experiment_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "experiment_id"):
            _record(experiment_id="../../outside")


class LedgerAndCampaignTests(unittest.TestCase):
    def test_first_measured_candidate_uses_the_frozen_initial_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())

            before = current_incumbent(
                workspace.load_config(), workspace.ledger.events(), workspace_root=workspace.root
            )
            stored = _workspace_record(workspace, _record("first"))
            after = current_incumbent(
                workspace.load_config(), workspace.ledger.events(), workspace_root=workspace.root
            )

        self.assertEqual(before["commit"], INITIAL_CONTROL_COMMIT)
        self.assertEqual(stored.parent_id, INITIAL_CONTROL_COMMIT)
        self.assertEqual(stored.provenance["control_commit"], INITIAL_CONTROL_COMMIT)
        self.assertEqual(after["commit"], INITIAL_CONTROL_COMMIT)

    def test_manual_promotion_advances_the_incumbent_for_the_next_candidate(self) -> None:
        next_candidate_commit = "f" * 40
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            completed = _workspace_record(workspace, _record("first", status="completed"))
            promotion = ExperimentRecord.from_dict(completed.to_dict())
            promotion.status = "promoted"
            workspace.record(promotion, artifact_base_dir=workspace.root)

            incumbent = current_incumbent(
                workspace.load_config(), workspace.ledger.events(), workspace_root=workspace.root
            )
            next_record = _record(
                "second",
                parent_id=CANDIDATE_COMMIT,
                provenance={
                    **_provenance(),
                    "git_commit": next_candidate_commit,
                    "control_commit": CANDIDATE_COMMIT,
                    "candidate_commit": next_candidate_commit,
                },
            )
            stored = _workspace_record(workspace, next_record)

        self.assertEqual(incumbent["commit"], CANDIDATE_COMMIT)
        self.assertEqual(incumbent["experiment_id"], "first")
        self.assertEqual(stored.provenance["control_commit"], CANDIDATE_COMMIT)

    def test_stale_parent_candidate_is_rejected_after_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(workspace, _record("first", status="completed"))
            _workspace_record(workspace, _record("first", status="promoted"))

            with self.assertRaisesRegex(ValueError, "current incumbent"):
                _workspace_record(workspace, _record("stale"))

    def test_parked_and_failed_attempts_do_not_advance_the_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(workspace, _record("parked", status="parked"))
            _workspace_record(workspace, _record("failed", status="rejected"))

            incumbent = current_incumbent(
                workspace.load_config(), workspace.ledger.events(), workspace_root=workspace.root
            )

        self.assertEqual(incumbent["commit"], INITIAL_CONTROL_COMMIT)
        self.assertEqual(incumbent["source"], "initial_control")

    def test_ledger_is_append_only_and_latest_returns_last_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger = ExperimentLedger(Path(temporary_directory) / "experiments.jsonl")
            first = ledger.append(_record(status="running"))
            second = ledger.append(_record(status="parked", outcome="No measurable change."))

            self.assertEqual(first.revision, 1)
            self.assertEqual(second.revision, 2)
            self.assertEqual(len(ledger.events()), 2)
            self.assertEqual(ledger.latest()[0].status, "parked")

    def test_beam_keeps_one_live_candidate_per_family(self) -> None:
        records = [
            _record("structural-old", beam="structural", common_candidate=95.0),
            _record("structural-best", beam="structural", common_candidate=90.0),
            _record("layout", beam="layout", common_candidate=92.0),
            _record("parked", beam="precision", status="parked"),
        ]

        beam = active_beam(_config(beam_width=2), records)

        self.assertEqual(
            [(record.beam, record.experiment_id) for record in beam],
            [("structural", "structural-best"), ("layout", "layout")],
        )

    def test_frontier_contains_only_successive_gate_passing_improvements(self) -> None:
        records = [
            _record("first", common_candidate=95.0),
            _record("worse", common_candidate=97.0),
            _record("better", common_candidate=90.0),
            _record("human-parked", status="parked", common_candidate=80.0),
        ]

        frontier = promotion_frontier(_config(), records)

        self.assertEqual([record.experiment_id for record, _ in frontier], ["first", "better"])

    def test_budget_blocks_only_new_experiment_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(
                Path(temporary_directory), _config(max_experiments=1)
            )
            _workspace_record(workspace, _record("first", status="parked"))
            status = budget_status(workspace.load_config(), workspace.ledger.latest())

            self.assertFalse(status["can_schedule"])
            _workspace_record(workspace, _record("first", status="rejected"))
            with self.assertRaisesRegex(RuntimeError, "budget is exhausted"):
                _workspace_record(workspace, _record("second"))

    def test_workspace_writes_report_and_failed_priors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(
                workspace,
                _record(
                    status="parked",
                    outcome="The launch count changed but latency did not.",
                ),
            )

            report = (workspace.root / "REPORT.md").read_text(encoding="utf-8")
            priors = (workspace.root / "PRIORS.md").read_text(encoding="utf-8")
            ledger_line = (workspace.root / "experiments.jsonl").read_text(encoding="utf-8")

        self.assertIn("Latest Promotion Glance", report)
        self.assertIn("exp-001", priors)
        self.assertIn("latency did not", priors)
        self.assertEqual(json.loads(ledger_line)["revision"], 1)

    def test_campaign_config_is_frozen_after_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            config_payload = json.loads(workspace.config_path.read_text(encoding="utf-8"))
            config_payload["min_improvement_pct"] = 0.0
            workspace.config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "changed after initialization"):
                workspace.load_config()

    def test_terminal_evidence_cannot_be_rewritten_by_a_later_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(workspace, _record(status="completed"))

            with self.assertRaisesRegex(ValueError, "evidence is immutable"):
                _workspace_record(
                    workspace,
                    _record(status="parked", common_candidate=80.0),
                )

    def test_promoted_status_requires_a_passing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())

            with self.assertRaisesRegex(ValueError, "passing mechanical gate"):
                _workspace_record(
                    workspace,
                    _record(
                        status="promoted",
                        common_candidate=80.0,
                        edge_candidate=101.0,
                    ),
                )

    def test_local_artifacts_receive_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "raw.json"
            artifact.write_text('{"latency_ms": 1.0}\n', encoding="utf-8")
            record = _record(
                raw_artifacts=["raw.json"],
                artifact_sha256={},
            )

            hashes = capture_local_artifact_hashes(record, root)

            self.assertEqual(len(hashes["raw.json"]), 64)
            self.assertEqual(record.artifact_sha256, hashes)

    def test_missing_remote_or_escaping_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                capture_local_artifact_hashes(
                    _record(raw_artifacts=["missing.json"], artifact_sha256={}),
                    root,
                )
            with self.assertRaisesRegex(ValueError, "trusted manifest"):
                capture_local_artifact_hashes(
                    _record(
                        raw_artifacts=["s3://bucket/raw.json"],
                        artifact_sha256={"s3://bucket/raw.json": ARTIFACT_HASH},
                    ),
                    root,
                )
            with self.assertRaisesRegex(ValueError, "escapes"):
                capture_local_artifact_hashes(
                    _record(raw_artifacts=["../outside.json"], artifact_sha256={}),
                    root,
                )

    def test_template_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            template = json.loads(
                (workspace.root / "experiment-template.json").read_text(encoding="utf-8")
            )
            record = ExperimentRecord.from_dict(template)

            decision = evaluate_experiment(workspace.load_config(), record)

            self.assertEqual(record.status, "planned")
            self.assertEqual(record.correctness, "unknown")
            self.assertFalse(record.measurements)
            self.assertEqual(decision.decision, "inconclusive")

    def test_nonfinite_policy_and_cost_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_improvement_pct"):
            _config(min_improvement_pct=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            _record(duration_s=float("inf"))

    def test_campaign_requires_frozen_cases_and_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_cases"):
            _config(primary_cases=[])
        with self.assertRaisesRegex(ValueError, "workload_spec"):
            _config(workload_spec="")

    def test_latest_view_orders_experiments_by_their_last_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger = ExperimentLedger(Path(temporary_directory) / "experiments.jsonl")
            ledger.append(_record("first", status="completed"))
            ledger.append(_record("second", status="completed"))
            ledger.append(_record("first", status="parked"))

            latest_ids = [record.experiment_id for record in ledger.latest()]

            self.assertEqual(latest_ids, ["second", "first"])


class GitProvenanceTests(unittest.TestCase):
    def test_capture_git_provenance_writes_a_concrete_diff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "campaign-test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Campaign Test"],
                cwd=repo,
                check=True,
            )
            source = repo / "kernel.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            source.write_text("VALUE = 2\n", encoding="utf-8")

            provenance = capture_git_provenance(
                repo,
                ["kernel.py"],
                repo / "artifacts",
                "exp-001",
            )

            diff_path = Path(str(provenance["diff_artifact"]))
            self.assertTrue(diff_path.exists())
            self.assertGreater(provenance["diff_bytes"], 0)
            self.assertIn("VALUE = 2", diff_path.read_text(encoding="utf-8"))
            self.assertEqual(sha256_file(diff_path), provenance["diff_sha256"])

    def test_capture_git_provenance_includes_untracked_candidate_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "campaign-test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Campaign Test"],
                cwd=repo,
                check=True,
            )
            baseline = repo / "baseline.py"
            baseline.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "baseline.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            candidate = repo / "candidate.py"
            candidate.write_text("VALUE = 2\n", encoding="utf-8")

            provenance = capture_git_provenance(
                repo,
                ["candidate.py"],
                repo / "artifacts",
                "exp-untracked",
            )

            diff_path = Path(str(provenance["diff_artifact"]))
            self.assertGreater(provenance["diff_bytes"], 0)
            self.assertIn("candidate.py", diff_path.read_text(encoding="utf-8"))

    def test_capture_git_provenance_compares_committed_candidate_to_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "campaign-test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Campaign Test"],
                cwd=repo,
                check=True,
            )
            source = repo / "kernel.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            control_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source.write_text("VALUE = 3\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-qam", "candidate"], cwd=repo, check=True)

            provenance = capture_git_provenance(
                repo,
                ["kernel.py"],
                repo / "artifacts",
                "exp-committed",
                control_revision=control_commit,
            )

            self.assertEqual(provenance["control_commit"], control_commit)
            self.assertNotEqual(provenance["candidate_commit"], control_commit)
            self.assertGreater(provenance["diff_bytes"], 0)

    def test_capture_git_provenance_rejects_an_empty_or_escaping_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "campaign-test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Campaign Test"],
                cwd=repo,
                check=True,
            )
            source = repo / "kernel.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

            with self.assertRaisesRegex(ValueError, "changed_surface"):
                capture_git_provenance(repo, [], repo / "artifacts", "empty")
            with self.assertRaisesRegex(ValueError, "escapes the repository"):
                capture_git_provenance(
                    repo,
                    ["../outside.py"],
                    repo / "artifacts",
                    "escaping",
                )

    def test_capture_git_provenance_rejects_undeclared_dirty_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "campaign-test@example.com"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Campaign Test"],
                cwd=repo,
                check=True,
            )
            kernel = repo / "kernel.py"
            helper = repo / "helper.py"
            kernel.write_text("VALUE = 1\n", encoding="utf-8")
            helper.write_text("HELPER = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py", "helper.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            kernel.write_text("VALUE = 2\n", encoding="utf-8")
            helper.write_text("HELPER = 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside changed_surface"):
                capture_git_provenance(
                    repo,
                    ["kernel.py"],
                    repo / "artifacts",
                    "undeclared",
                )


class CampaignDashboardTests(unittest.TestCase):
    def test_external_evidence_is_copied_into_workspace_owned_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = CampaignWorkspace.initialize(root / "campaign", _config())
            evidence_dir = root / "external-evidence"
            evidence_dir.mkdir()
            external_artifact = evidence_dir / "raw.json"
            external_artifact.write_bytes(ARTIFACT_CONTENT)
            external_diff = evidence_dir / "candidate.diff"
            external_diff.write_bytes(DIFF_CONTENT)
            record = _record(
                raw_artifacts=[external_artifact.name],
                artifact_sha256={external_artifact.name: ARTIFACT_HASH},
                provenance={
                    **_provenance(),
                    "diff_artifact": external_diff.name,
                },
            )

            stored = workspace.record(record, artifact_base_dir=evidence_dir)
            owned_reference = stored.raw_artifacts[0]
            owned_path = workspace.root / owned_reference
            owned_diff_reference = str(stored.provenance["diff_artifact"])
            owned_diff_path = workspace.root / owned_diff_reference
            binding_reference = str(stored.provenance["record_binding_artifact"])
            binding_path = workspace.root / binding_reference
            external_artifact.unlink()
            external_diff.unlink()

            dashboard = build_campaign_dashboard(workspace.root)
            owned_bytes = owned_path.read_bytes()
            owned_mode = owned_path.stat().st_mode
            owned_diff_bytes = owned_diff_path.read_bytes()
            owned_diff_mode = owned_diff_path.stat().st_mode
            binding_mode = binding_path.stat().st_mode

        self.assertTrue(owned_reference.startswith("artifacts/exp-001/evidence/raw-"))
        self.assertEqual(owned_bytes, ARTIFACT_CONTENT)
        self.assertEqual(owned_mode & 0o222, 0)
        self.assertTrue(owned_diff_reference.startswith("artifacts/exp-001/evidence/diff-"))
        self.assertEqual(owned_diff_bytes, DIFF_CONTENT)
        self.assertEqual(owned_diff_mode & 0o222, 0)
        self.assertEqual(binding_mode & 0o222, 0)
        self.assertTrue(dashboard["experiments"][0]["evidence_integrity"]["valid"])
        self.assertTrue(dashboard["experiments"][0]["artifacts"][0]["downloadable"])

    def test_initial_control_is_the_incumbent_before_any_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())

            dashboard = build_campaign_dashboard(workspace.root)

        self.assertEqual(dashboard["incumbent"]["commit"], INITIAL_CONTROL_COMMIT)
        self.assertEqual(dashboard["incumbent"]["source"], "initial_control")
        self.assertIsNone(dashboard["incumbent"]["experiment_id"])
        self.assertIsNone(dashboard["incumbent"]["experiment"])
        self.assertEqual(dashboard["counts"]["experiments"], 0)

    def test_promoted_record_drives_incumbent_and_per_case_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            _workspace_record(workspace, _record(status="promoted"))

            dashboard = build_campaign_dashboard(workspace.root)

        incumbent = dashboard["incumbent"]
        self.assertEqual(incumbent["commit"], CANDIDATE_COMMIT)
        self.assertEqual(incumbent["experiment_id"], "exp-001")
        self.assertTrue(incumbent["experiment"]["is_incumbent"])
        self.assertEqual(incumbent["experiment"]["gate"]["decision"], "promote")
        self.assertEqual(dashboard["latest_measured"]["cases"][0]["case_id"], "common")
        self.assertEqual(dashboard["frontier"][0]["experiment_id"], "exp-001")

    def test_artifact_download_requires_ledger_hash_and_unchanged_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            record = _workspace_record(workspace, _record(status="promoted"))
            artifact = record.raw_artifacts[0]

            resolved = resolve_campaign_artifact(workspace.root, artifact)
            self.assertEqual(resolved.read_bytes(), ARTIFACT_CONTENT)
            with self.assertRaisesRegex(ValueError, "hashed reference"):
                resolve_campaign_artifact(workspace.root, "campaign.json")

            resolved.chmod(0o644)
            resolved.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                resolve_campaign_artifact(workspace.root, artifact)

    def test_tampered_promoted_evidence_blocks_review_and_incumbent_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            record = _workspace_record(workspace, _record(status="promoted"))
            artifact = workspace.root / record.raw_artifacts[0]
            artifact.chmod(0o644)
            artifact.write_bytes(b"tampered\n")

            with self.assertRaisesRegex(ValueError, "fails evidence integrity"):
                current_incumbent(
                    workspace.load_config(),
                    workspace.ledger.events(),
                    workspace_root=workspace.root,
                )
            with self.assertRaisesRegex(ValueError, "fails evidence integrity"):
                build_campaign_dashboard(workspace.root)

    def test_tampered_candidate_diff_blocks_review_and_incumbent_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            record = _workspace_record(workspace, _record(status="promoted"))
            diff_path = workspace.root / str(record.provenance["diff_artifact"])
            diff_path.chmod(0o644)
            diff_path.write_bytes(b"tampered diff\n")

            with self.assertRaisesRegex(ValueError, "candidate diff SHA-256"):
                current_incumbent(
                    workspace.load_config(),
                    workspace.ledger.events(),
                    workspace_root=workspace.root,
                )
            with self.assertRaisesRegex(ValueError, "candidate diff SHA-256"):
                build_campaign_dashboard(workspace.root)

    def test_forged_promoted_ledger_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            forged = _record(status="promoted", correctness="failed")
            forged.revision = 1
            workspace.ledger.path.write_text(
                json.dumps(forged.to_dict()) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "fails evidence integrity"):
                current_incumbent(
                    workspace.load_config(),
                    workspace.ledger.events(),
                    workspace_root=workspace.root,
                )
            with self.assertRaisesRegex(ValueError, "fails evidence integrity"):
                build_campaign_dashboard(workspace.root)

    def test_forged_winning_measurements_fail_the_canonical_record_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = CampaignWorkspace.initialize(Path(temporary_directory), _config())
            stored = _workspace_record(
                workspace,
                _record(status="completed", common_candidate=110.0),
            )
            forged = stored.to_dict()
            forged["status"] = "promoted"
            forged["measurements"]["common"]["candidate"] = [90.0, 90.0, 90.0]
            workspace.ledger.path.write_text(
                json.dumps(forged) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "canonical record binding"):
                current_incumbent(
                    workspace.load_config(),
                    workspace.ledger.events(),
                    workspace_root=workspace.root,
                )
            with self.assertRaisesRegex(ValueError, "canonical record binding"):
                build_campaign_dashboard(workspace.root)


if __name__ == "__main__":
    unittest.main()
