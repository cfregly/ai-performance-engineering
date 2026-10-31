from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import piq_gpu_training_evidence as evidence_tool
from tools.piq_gpu_training_evidence import import_bundle, validate_bundle
from tools.test_qualified_gpu_training import build_gpu_fixture, _hash_bytes


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: object) -> tuple[str, int]:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    return _hash_bytes(payload), len(payload)


def _rewrite_artifact(bundle: Path, relative: str, value: object,
                      *, update_attempt_ref: bool = False) -> None:
    envelope_path = bundle / "envelope.json"
    envelope = _load(envelope_path)
    digest, size = _write(bundle / relative, value)
    item = next(item for item in envelope["artifacts"] if item["path"] == relative)
    item["sha256"], item["size_bytes"] = digest, size
    if update_attempt_ref:
        attempt = next(item for item in envelope["attempts"] if item["raw_result"]["path"] == relative)
        attempt["raw_result"]["sha256"] = digest
    _write(envelope_path, envelope)


def _add_artifact(bundle: Path, artifacts: list[dict], relative: str,
                  value: object, role: str) -> dict[str, str]:
    path = bundle / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
        digest, size = _hash_bytes(value), len(value)
    else:
        digest, size = _write(path, value)
    artifacts.append({
        "path": relative, "sha256": digest, "size_bytes": size, "role": role,
    })
    return {"path": relative, "sha256": digest}


def _rewrite_all_refs(bundle: Path, relative: str, value: object) -> None:
    envelope_path = bundle / "envelope.json"
    envelope = _load(envelope_path)
    digest, size = _write(bundle / relative, value)
    next(item for item in envelope["artifacts"] if item["path"] == relative).update(
        {"sha256": digest, "size_bytes": size}
    )

    def update(current: object) -> None:
        if isinstance(current, dict):
            if current.get("path") == relative and "sha256" in current:
                current["sha256"] = digest
            for child in current.values():
                update(child)
        elif isinstance(current, list):
            for child in current:
                update(child)

    update(envelope)
    _write(envelope_path, envelope)


def _retention_document(kind: str, role: str, study_id: str, run_id: str,
                        attempt_ids: list[str], *, two_host: bool = False) -> dict:
    document = {
        "schema": f"piq.{kind}-{role}.v2", "study_id": study_id,
        "run_id": run_id, "attempt_ids": attempt_ids,
        "complete": True,
        "completed_at" if role == "cleanup" else "verified_at":
            "2026-10-31T10:11:00Z",
    }
    if role != "cleanup":
        document["immutable"] = True
    if role == "cleanup" and two_host:
        document.update(
            pids=[41001, 42001], workers_terminated=True, transport_closed=True,
        )
    return document


def build_stage0_fixture(root: Path) -> Path:
    bundle = root / "stage0"
    bundle.mkdir(parents=True)
    artifacts: list[dict] = []
    study_id, run_id = "synthetic-stage0-study", "synthetic-stage0-run"
    attempts = ["stage0-baseline-attempt", "stage0-candidate-attempt"]
    design_ref = _add_artifact(
        bundle, artifacts, "design.json",
        {"schema": "piq.checkpoint-write-batching-stage0-design.v2",
         "study_id": study_id, "arms": ["baseline", "candidate"]},
        "design",
    )
    input_sha = "1" * 64
    output_sha = "2" * 64
    input_ref = _add_artifact(
        bundle, artifacts, "input.json",
        {"schema": "piq.stage0-input.v2", "sha256": input_sha}, "input",
    )
    source_ref = _add_artifact(
        bundle, artifacts, "source/runner.py",
        b"# synthetic Stage 0 fixture. Never executed\n", "source",
    )
    raw_refs: list[dict[str, str]] = []
    result_refs: list[dict[str, str]] = []
    for role, attempt_id in zip(("baseline", "candidate"), attempts, strict=True):
        raw_refs.append(_add_artifact(
            bundle, artifacts, f"attempts/{role}.json",
            {
                "schema": "piq.checkpoint-write-batching-stage0-attempt.v2",
                "study_id": study_id, "run_id": run_id,
                "attempt_id": attempt_id, "role": role,
                "status": "succeeded", "accepted": True,
            },
            "attempt",
        ))
        result_refs.append(_add_artifact(
            bundle, artifacts, f"correctness/{role}.json",
            {
                "schema": "piq.checkpoint-write-batching-stage0-correctness.v2",
                "study_id": study_id, "run_id": run_id,
                "attempt_id": attempt_id, "role": role,
                "input_sha256": input_sha, "output_sha256": output_sha,
                "accepted": True,
            },
            "correctness",
        ))
    oracle_ref = _add_artifact(
        bundle, artifacts, "correctness/oracle.json",
        {
            "schema": "piq.checkpoint-write-batching-stage0-oracle.v2",
            "study_id": study_id, "input_sha256": input_sha,
            "expected_output_sha256": output_sha,
        },
        "correctness_oracle",
    )
    retention: dict[str, dict[str, str]] = {}
    for role in ("cleanup", "custody", "readback", "export"):
        retention[role] = _add_artifact(
            bundle, artifacts, f"retention/{role}.json",
            _retention_document(
                "checkpoint_write_batching_stage0", role, study_id, run_id,
                attempts,
            ),
            role,
        )
    envelope = {
        "schema_version": 2, "kind": "piq_experiment_evidence",
        "scope": "diagnostic", "scientific_disposition": "DIAGNOSTIC_ONLY",
        "experiment_kind": "checkpoint_write_batching_stage0",
        "workload_family": "checkpoint_write_batching", "study_id": study_id,
        "experiment": {
            "id": study_id, "question": "Does batching preserve exact output?",
            "hypothesis": "Both arms retain the same exact oracle output.",
            "workload_scope": "Synthetic Stage 0 contract fixture only.",
            "design": design_ref, "source_artifacts": [source_ref],
            "input": input_ref,
        },
        "stack": {
            "piq_revision": "1" * 40, "runtime_identifier": "synthetic-runtime",
            "hardware_identifier": "synthetic-stage0-host",
            "layers": [
                {"name": name, "state": "observed"}
                for name in ("storage", "orchestration", "runtime", "workload")
            ],
        },
        "objective": {
            "metric": "complete_stage0_elapsed", "unit": "nanoseconds",
            "direction": "minimize", "started_at": "2026-10-31T10:00:00Z",
            "ended_at": "2026-10-31T10:10:00Z",
        },
        "executions": [{
            "run_id": run_id, "order": ["baseline", "candidate"],
            "attempt_ids": attempts,
        }],
        "attempts": [
            {
                "id": attempt_id, "run_id": run_id, "role": role,
                "status": "succeeded", "accepted": True,
                "failure_reason": None, "raw_result": raw_ref,
            }
            for role, attempt_id, raw_ref in zip(
                ("baseline", "candidate"), attempts, raw_refs, strict=True,
            )
        ],
        "comparison": {
            "baseline_attempt_id": attempts[0],
            "candidate_attempt_id": attempts[1],
            "scientific_disposition": "DIAGNOSTIC_ONLY",
            "reason": "Stage 0 checks exact correctness and custody only.",
        },
        "correctness": {
            "accepted": True, "oracle": oracle_ref,
            "baseline": result_refs[0], "candidate": result_refs[1],
        },
        "workload_evidence": {"correctness": oracle_ref, **retention},
        "detached_objects": [], "artifacts": artifacts,
    }
    _write(bundle / "envelope.json", envelope)
    return bundle


def build_two_host_fixture(root: Path) -> Path:
    bundle = root / "two-host"
    bundle.mkdir(parents=True)
    artifacts: list[dict] = []
    study_id, run_id, attempt_id = (
        "synthetic-two-host-study", "synthetic-two-host-run",
        "synthetic-two-host-attempt",
    )
    design_ref = _add_artifact(
        bundle, artifacts, "design.json",
        {"schema": "piq.cpu-two-host-design.v2", "study_id": study_id},
        "design",
    )
    input_ref = _add_artifact(
        bundle, artifacts, "input.json",
        {"schema": "piq.cpu-two-host-input.v2", "records": 100}, "input",
    )
    source_ref = _add_artifact(
        bundle, artifacts, "source/runner.py",
        b"# synthetic two-host contract fixture. Never executed\n", "source",
    )
    raw_ref = _add_artifact(
        bundle, artifacts, "attempt.json",
        {
            "schema": "piq.cpu-two-host-attempt.v2", "study_id": study_id,
            "run_id": run_id, "attempt_id": attempt_id,
            "status": "succeeded", "accepted": True,
        },
        "execution",
    )
    hosts, boots, pids = ["a" * 64, "b" * 64], ["c" * 64, "d" * 64], [41001, 42001]
    ranks = []
    for rank in (0, 1):
        output_ref = _add_artifact(
            bundle, artifacts, f"ranks/rank-{rank}.json",
            {
                "schema": "piq.cpu-two-host-rank-output.v2",
                "study_id": study_id, "run_id": run_id,
                "attempt_id": attempt_id, "rank": rank, "pid": pids[rank],
                "physical_host_sha256": hosts[rank],
                "boot_id_sha256": boots[rank],
            },
            "rank_output",
        )
        ranks.append({
            "rank": rank, "local_rank": 0, "pid": pids[rank],
            "physical_host_sha256": hosts[rank],
            "boot_id_sha256": boots[rank], "output_ref": output_ref,
        })
    partition_values = {
        "input_records": 100, "partition_records": [48, 52],
        "rehashed_records": 100, "output_records": 100,
    }
    partition_ref = _add_artifact(
        bundle, artifacts, "partition-rehash.json",
        {
            "schema": "piq.cpu-two-host-partition-rehash.v2",
            "study_id": study_id, "run_id": run_id,
            "attempt_id": attempt_id, **partition_values,
        },
        "partition_rehash",
    )
    transport_values = {
        "family": "tcp", "source_rank": 0, "target_rank": 1,
        "source_pid": pids[0], "target_pid": pids[1], "bytes_sent": 4096,
    }
    transport_ref = _add_artifact(
        bundle, artifacts, "transport.json",
        {
            "schema": "piq.cpu-two-host-worker-transport.v2",
            "study_id": study_id, "run_id": run_id,
            "attempt_id": attempt_id, **transport_values,
            "source_host_sha256": hosts[0], "target_host_sha256": hosts[1],
        },
        "transport",
    )
    correctness_ref = _add_artifact(
        bundle, artifacts, "correctness.json",
        {
            "schema": "piq.cpu-two-host-correctness.v2",
            "study_id": study_id, "run_id": run_id,
            "attempt_id": attempt_id, "input_records": 100,
            "output_records": 100, "accepted": True,
        },
        "correctness",
    )
    owner_ref = _add_artifact(
        bundle, artifacts, "owner-attestation.json",
        {
            "schema": "piq.cpu-two-host-owner-attestation.v2",
            "study_id": study_id, "run_id": run_id, "attempt_id": attempt_id,
            "owner": "synthetic-owner", "authority": "synthetic-authority",
            "attested_at": "2026-10-31T09:59:00Z",
            "hosts": [
                {
                    "rank": rank, "physical_host_sha256": hosts[rank],
                    "boot_id_sha256": boots[rank], "pid": pids[rank],
                }
                for rank in (0, 1)
            ],
            "accepted": True,
        },
        "owner_attestation",
    )
    admission_ref = _add_artifact(
        bundle, artifacts, "current-admission.json",
        {
            "schema": "piq.cpu-two-host-current-admission.v2",
            "study_id": study_id, "run_id": run_id, "attempt_id": attempt_id,
            "authority": "synthetic-authority",
            "owner_attestation_sha256": owner_ref["sha256"],
            "admitted_at": "2026-10-31T09:58:00Z",
            "expires_at": "2026-10-31T10:30:00Z",
            "status": "ADMITTED", "revoked": False,
        },
        "current_admission",
    )
    retention: dict[str, dict[str, str]] = {}
    for role in ("cleanup", "custody", "readback", "export"):
        retention[role] = _add_artifact(
            bundle, artifacts, f"retention/{role}.json",
            _retention_document(
                "cpu_two_host_functional", role, study_id, run_id,
                [attempt_id], two_host=True,
            ),
            role,
        )
    envelope = {
        "schema_version": 2, "kind": "piq_experiment_evidence",
        "scope": "diagnostic", "scientific_disposition": "DIAGNOSTIC_ONLY",
        "experiment_kind": "cpu_two_host_functional",
        "workload_family": "cpu_two_host_functional", "study_id": study_id,
        "experiment": {
            "id": study_id, "question": "Did two distinct hosts complete one exact attempt?",
            "hypothesis": "Two owner-admitted hosts exchange partition data.",
            "workload_scope": "Synthetic two-host contract fixture only.",
            "design": design_ref, "source_artifacts": [source_ref],
            "input": input_ref,
        },
        "stack": {
            "piq_revision": "2" * 40, "runtime_identifier": "synthetic-runtime",
            "hardware_identifier": "two-distinct-host-hashes",
            "layers": [
                {"name": name, "state": "observed"}
                for name in ("host", "fabric", "storage", "orchestration", "runtime", "workload")
            ],
        },
        "objective": {
            "metric": "functional_two_host_elapsed", "unit": "nanoseconds",
            "direction": "minimize", "started_at": "2026-10-31T10:00:00Z",
            "ended_at": "2026-10-31T10:10:00Z",
        },
        "executions": [{
            "run_id": run_id, "attempt_ids": [attempt_id],
            "owner_attestation": owner_ref, "current_admission": admission_ref,
        }],
        "attempts": [{
            "id": attempt_id, "run_id": run_id, "status": "succeeded",
            "accepted": True, "failure_reason": None, "raw_result": raw_ref,
            "ranks": ranks,
            "partition_rehash": {**partition_values, "artifact": partition_ref},
            "transport": {**transport_values, "artifact": transport_ref},
        }],
        "comparison": {
            "scientific_disposition": "DIAGNOSTIC_ONLY",
            "reason": "Functional evidence does not establish scientific qualification.",
        },
        "correctness": {"accepted": True, "artifact": correctness_ref},
        "workload_evidence": {
            "execution": raw_ref, "correctness": correctness_ref,
            "partition_rehash": partition_ref, "transport": transport_ref,
            **retention, "owner_attestation": owner_ref,
            "current_admission": admission_ref,
        },
        "detached_objects": [], "artifacts": artifacts,
    }
    _write(bundle / "envelope.json", envelope)
    return bundle


@contextmanager
def fixed_two_host_admission(bundle: Path):
    """Install a test-only path for an exact source-reviewed current-v2 tuple."""
    envelope_path = bundle / "envelope.json"
    envelope = _load(envelope_path)
    if envelope.get("experiment_kind") != "cpu_two_host_functional":
        yield
        return
    execution = envelope["executions"][0]
    owner_ref = execution["owner_attestation"]
    owner = _load(bundle / owner_ref["path"])
    admission_ref = execution["current_admission"]
    document = {
        "schema_version": 2,
        "kind": "piq_v2_two_host_admissions",
        "entries": [{
            "admission_id": "synthetic-current-v2-admission",
            "reviewed_at": "2026-10-31T10:12:00Z",
            "reviewer_id": "synthetic-independent-reviewer",
            "source": {
                "id": "synthetic-owner-handback",
                "sha256": "9" * 64,
            },
            "study_id": envelope["study_id"],
            "owner": owner["owner"],
            "authority": owner["authority"],
            "envelope_sha256": _hash_bytes(envelope_path.read_bytes()),
            "owner_attestation_sha256": owner_ref["sha256"],
            "current_admission_sha256": admission_ref["sha256"],
            "run_id": execution["run_id"],
            "attempt_ids": execution["attempt_ids"],
        }],
    }
    with tempfile.TemporaryDirectory() as directory:
        anchor = Path(directory) / "piq-v2-two-host-admissions.json"
        _write(anchor, document)
        with patch.object(
            evidence_tool, "PIQ_V2_TWO_HOST_ADMISSIONS_PATH", anchor
        ):
            yield


class PiqGpuTrainingEvidenceTests(unittest.TestCase):
    def test_stage0_variant_passes_and_import_remains_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = build_stage0_fixture(root)
            self.assertEqual([], validate_bundle(bundle))
            destination = root / "stage0-imported"
            import_bundle(bundle, destination)
            receipt = _load(destination / "import-receipt.json")
            self.assertEqual("checkpoint_write_batching_stage0", receipt["experiment_kind"])
            self.assertEqual("not_established", receipt["qualification"])

    def test_stage0_requires_exact_two_arm_correctness_and_retention(self) -> None:
        for mutation, expected in (
            ("candidate-output", "candidate correctness differs"),
            ("oracle-identity", "oracle schema/study identity drifted"),
            ("cleanup", "cleanup document"),
            ("readback", "readback document must be immutable"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                bundle = build_stage0_fixture(Path(directory))
                envelope = _load(bundle / "envelope.json")
                if mutation in {"candidate-output", "oracle-identity"}:
                    role = "candidate" if mutation == "candidate-output" else "oracle"
                    relative = envelope["correctness"][role]["path"]
                    document = _load(bundle / relative)
                    if mutation == "candidate-output":
                        document["output_sha256"] = "f" * 64
                    else:
                        document["schema"] = "attacker-controlled-oracle.v999"
                        document["study_id"] = "different-study"
                else:
                    relative = envelope["workload_evidence"][mutation]["path"]
                    document = _load(bundle / relative)
                    if mutation == "cleanup":
                        document["attempt_ids"] = document["attempt_ids"][:1]
                    else:
                        document["immutable"] = False
                _rewrite_all_refs(bundle, relative, document)
                errors = validate_bundle(bundle)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_two_host_variant_passes_with_current_owner_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_two_host_fixture(Path(directory))
            with fixed_two_host_admission(bundle):
                self.assertEqual([], validate_bundle(bundle))

    def test_two_host_variant_cannot_self_admit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = build_two_host_fixture(Path(directory))
            errors = validate_bundle(bundle)
            self.assertTrue(
                any("fixed source-reviewed trust anchor" in error for error in errors),
                errors,
            )

    def test_two_host_variant_fails_closed_on_identity_transport_and_custody(self) -> None:
        cases = (
            ("same-host", "two distinct physical hosts"),
            ("same-boot", "two distinct physical hosts"),
            ("partition", "partition/rehash census"),
            ("transport", "worker PID to worker PID"),
            ("cleanup", "close both workers and transport"),
            ("admission", "admission is not current"),
            ("owner", "owner attestation is not exact"),
        )
        for mutation, expected in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                bundle = build_two_host_fixture(Path(directory))
                envelope_path = bundle / "envelope.json"
                envelope = _load(envelope_path)
                attempt = envelope["attempts"][0]
                if mutation == "same-host":
                    attempt["ranks"][1]["physical_host_sha256"] = attempt["ranks"][0]["physical_host_sha256"]
                    _write(envelope_path, envelope)
                elif mutation == "same-boot":
                    attempt["ranks"][1]["boot_id_sha256"] = attempt["ranks"][0]["boot_id_sha256"]
                    _write(envelope_path, envelope)
                elif mutation == "partition":
                    attempt["partition_rehash"]["partition_records"] = [49, 52]
                    _write(envelope_path, envelope)
                elif mutation == "transport":
                    attempt["transport"]["source_pid"] = 99999
                    _write(envelope_path, envelope)
                else:
                    role = {
                        "cleanup": "cleanup", "admission": "current_admission",
                        "owner": "owner_attestation",
                    }[mutation]
                    relative = envelope["workload_evidence"][role]["path"]
                    document = _load(bundle / relative)
                    if mutation == "cleanup":
                        document["workers_terminated"] = False
                    elif mutation == "admission":
                        document["expires_at"] = "2026-10-31T10:05:00Z"
                    else:
                        document["hosts"][1]["physical_host_sha256"] = "e" * 64
                    _rewrite_all_refs(bundle, relative, document)
                with fixed_two_host_admission(bundle):
                    errors = validate_bundle(bundle)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_complete_bundle_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            self.assertEqual([], validate_bundle(bundle))

    def test_piq_v1_envelope_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            envelope = _load(bundle / "envelope.json")
            envelope["schema_version"] = 1
            _write(bundle / "envelope.json", envelope)
            errors = validate_bundle(bundle)
            self.assertTrue(
                any(
                    "schema_version 1 is unsupported. Only version 2 is active"
                    in error
                    for error in errors
                ),
                errors,
            )

    def test_one_rank_and_duplicate_device_fail(self) -> None:
        for mutation in ("one-rank", "duplicate-device"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                receipt = build_gpu_fixture(package)
                bundle = (package / receipt["piq_envelope"]["path"]).parent
                envelope = _load(bundle / "envelope.json")
                relative = envelope["attempts"][0]["raw_result"]["path"]
                raw = _load(bundle / relative)
                if mutation == "one-rank":
                    raw["ranks"].pop()
                else:
                    raw["ranks"][1]["device_uuid"] = raw["ranks"][0]["device_uuid"]
                _rewrite_artifact(bundle, relative, raw, update_attempt_ref=True)
                errors = validate_bundle(bundle)
                self.assertTrue(any("rank" in error.lower() or "device" in error.lower() for error in errors), errors)

    def test_assignment_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            envelope = _load(bundle / "envelope.json")
            relative = envelope["executions"][0]["assignment_evidence"]["path"]
            assignment = _load(bundle / relative)
            assignment["device_uuids"].reverse()
            digest, size = _write(bundle / relative, assignment)
            for item in envelope["artifacts"]:
                if item["path"] == relative:
                    item["sha256"], item["size_bytes"] = digest, size
            envelope["executions"][0]["assignment_evidence"]["sha256"] = digest
            for attempt in envelope["attempts"][:2]:
                attempt["assignment_evidence"]["sha256"] = digest
            _write(bundle / "envelope.json", envelope)
            self.assertTrue(any("device UUID" in error for error in validate_bundle(bundle)))

    def test_assignment_plan_and_window_are_bound_to_design_and_objective(self) -> None:
        for mutation in ("plan", "window", "attempt-window"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                receipt = build_gpu_fixture(package)
                bundle = (package / receipt["piq_envelope"]["path"]).parent
                envelope = _load(bundle / "envelope.json")
                relative = envelope["executions"][0]["assignment_evidence"]["path"]
                assignment = _load(bundle / relative)
                if mutation == "plan":
                    assignment["plan_sha256"] = "f" * 64
                elif mutation == "window":
                    assignment["window"]["started_at"] = "2026-10-31T09:59:59Z"
                else:
                    assignment["window"]["started_at"] = "2026-10-31T11:20:01Z"
                    assignment["window"]["ended_at"] = "2026-10-31T11:31:47Z"
                digest, size = _write(bundle / relative, assignment)
                next(item for item in envelope["artifacts"]
                     if item["path"] == relative).update(
                         {"sha256": digest, "size_bytes": size})
                envelope["executions"][0]["assignment_evidence"]["sha256"] = digest
                for attempt in envelope["attempts"][:2]:
                    attempt["assignment_evidence"]["sha256"] = digest
                _write(bundle / "envelope.json", envelope)
                errors = validate_bundle(bundle)
                expected = (
                    "plan identity" if mutation == "plan"
                    else "raw attempt interval" if mutation == "attempt-window"
                    else "window"
                )
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_raw_arm_and_observed_host_must_match_immutable_evidence(self) -> None:
        for mutation in ("arm", "host"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                receipt = build_gpu_fixture(package)
                bundle = (package / receipt["piq_envelope"]["path"]).parent
                envelope = _load(bundle / "envelope.json")
                relative = envelope["attempts"][0]["raw_result"]["path"]
                raw = _load(bundle / relative)
                if mutation == "arm":
                    raw["arm_configuration_sha256"] = "f" * 64
                    for rank in raw["ranks"]:
                        rank["arm_configuration_sha256"] = "f" * 64
                else:
                    for rank in raw["ranks"]:
                        rank["host"] = "other-host"
                _rewrite_artifact(bundle, relative, raw, update_attempt_ref=True)
                errors = validate_bundle(bundle)
                expected = "immutable design" if mutation == "arm" else "observed host"
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_detached_object_requires_all_readbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            envelope = _load(bundle / "envelope.json")
            relative = envelope["detached_objects"][0]["local_readback"]["path"]
            readback = _load(bundle / relative)
            readback["objects"].pop(0)
            digest, size = _write(bundle / relative, readback)
            for item in envelope["artifacts"]:
                if item["path"] == relative:
                    item["sha256"], item["size_bytes"] = digest, size
            for item in envelope["detached_objects"]:
                item["local_readback"]["sha256"] = digest
            _write(bundle / "envelope.json", envelope)
            self.assertTrue(any("local_readback" in error for error in validate_bundle(bundle)))

    def test_duplicate_keys_and_nonfinite_are_rejected_in_nested_json(self) -> None:
        for raw in (b'{"a": 1, "a": 1}\n', b'{"a": NaN}\n'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                package = Path(directory)
                receipt = build_gpu_fixture(package)
                bundle = (package / receipt["piq_envelope"]["path"]).parent
                envelope = _load(bundle / "envelope.json")
                relative = "telemetry/summary-index.json"
                path = bundle / relative
                path.write_bytes(raw)
                item = next(item for item in envelope["artifacts"] if item["path"] == relative)
                item["sha256"], item["size_bytes"] = _hash_bytes(raw), len(raw)
                _write(bundle / "envelope.json", envelope)
                errors = validate_bundle(bundle)
                self.assertTrue(any("duplicate JSON key" in error or "nonfinite" in error for error in errors), errors)

    def test_case_collisions_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            envelope = _load(bundle / "envelope.json")
            original = bundle / "source/runner.py"
            collision = bundle / "source/Runner.py"
            collision.write_bytes(original.read_bytes())
            envelope["artifacts"].append({
                "path": "source/Runner.py", "sha256": _hash_bytes(collision.read_bytes()),
                "size_bytes": collision.stat().st_size, "role": "case-collision",
            })
            _write(bundle / "envelope.json", envelope)
            errors = validate_bundle(bundle)
            self.assertTrue(any("collide by case" in error for error in errors), errors)

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            (bundle / "source-link.py").symlink_to(bundle / "source/runner.py")
            errors = validate_bundle(bundle)
            self.assertTrue(any("symlink" in error for error in errors), errors)

    def test_import_is_diagnostic_and_preserves_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "source"
            receipt = build_gpu_fixture(package)
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            envelope = _load(bundle / "envelope.json")
            attempt = envelope["attempts"][0]
            relative = attempt["raw_result"]["path"]
            raw = _load(bundle / relative)
            raw["status"], raw["accepted"], raw["failure_reason"] = "failed", False, "synthetic failure"
            digest, size = _write(bundle / relative, raw)
            next(item for item in envelope["artifacts"] if item["path"] == relative).update(
                {"sha256": digest, "size_bytes": size})
            attempt["status"], attempt["accepted"], attempt["failure_reason"] = "failed", False, "synthetic failure"
            attempt["raw_result"]["sha256"] = digest
            envelope["scientific_disposition"] = "INCONCLUSIVE"
            envelope["comparison"]["scientific_disposition"] = "INCONCLUSIVE"
            envelope["correctness"]["accepted"] = False
            _write(bundle / "envelope.json", envelope)
            self.assertEqual([], validate_bundle(bundle))
            destination = root / "imported"
            import_bundle(bundle, destination)
            imported = _load(destination / "envelope.json")
            self.assertEqual("failed", imported["attempts"][0]["status"])
            receipt_data = _load(destination / "import-receipt.json")
            self.assertFalse(receipt_data["publishable"])
            self.assertEqual("not_established", receipt_data["qualification"])


if __name__ == "__main__":
    unittest.main()
