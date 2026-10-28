from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.piq_experiment_evidence import import_bundle, ingestion_metadata_sha256, validate_bundle

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/piq-diagnostic"


class PiqEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "source"
        shutil.copytree(EXAMPLE, self.bundle)
        self.data = json.loads((self.bundle / "envelope.json").read_text())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self) -> None:
        (self.bundle / "envelope.json").write_text(json.dumps(self.data, indent=2) + "\n")

    def problems(self, phrase: str) -> None:
        self.save()
        errors = validate_bundle(self.bundle)
        self.assertTrue(any(phrase in error for error in errors), errors)

    def rewrite_artifact(self, relative: str, value: dict) -> None:
        path = self.bundle / relative
        path.write_text(json.dumps(value, indent=2) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        def refresh(value):
            if isinstance(value, dict):
                if value.get("path") == relative:
                    value["sha256"] = digest
                for child in value.values():
                    refresh(child)
            elif isinstance(value, list):
                for child in value:
                    refresh(child)
        refresh(self.data)

    def test_synthetic_fixture_is_valid_internal_diagnostic(self) -> None:
        self.assertEqual(validate_bundle(self.bundle), [])
        self.assertIn("Synthetic", self.data["experiment"]["workload_scope"])

    def test_cpu_does_not_require_or_invent_gpu_evidence(self) -> None:
        self.assertNotIn("power_limit_w", self.data)
        self.assertNotIn("profile_evidence", self.data)
        self.assertNotIn("accelerator", [layer["name"] for layer in self.data["stack"]["layers"]])
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_common_contract_supports_new_family(self) -> None:
        self.data["workload_family"] = "cpu_embedding_preprocessing"
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["workload_family"] = self.data["workload_family"]
        self.rewrite_artifact(design_path, design)
        for attempt in self.data["attempts"]:
            raw_path = attempt["raw_result"]["path"]
            raw = json.loads((self.bundle / raw_path).read_text())
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
            self.rewrite_artifact(raw_path, raw)
        self.save()
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_unknown_version_and_boolean_version_fail(self) -> None:
        for version in (2, True):
            with self.subTest(version=version):
                self.data["schema_version"] = version
                self.problems("schema_version must equal 1")

    def test_qualified_scope_cannot_promote_smoke(self) -> None:
        self.data["scope"] = "qualified"
        self.data["comparison"]["scientific_disposition"] = "QUALIFIED"
        self.problems("diagnostic scope only")
        self.problems("QUALIFIED is unsupported")

    def test_hash_tampering_fails(self) -> None:
        (self.bundle / "evidence/input.csv").write_text("changed\n")
        self.problems("sha256 does not match")

    def test_design_binds_input_source_and_roles(self) -> None:
        path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / path).read_text())
        design["input"]["sha256"] = "f" * 64
        design["source_artifacts"] = []
        design["required_evidence_roles"] = ["input", "output"]
        self.rewrite_artifact(path, design)
        self.problems("experiment design input does not match")
        self.problems("experiment design source_artifacts does not match")
        self.problems("experiment design required_evidence_roles does not match")

    def test_raw_result_binds_exact_run_and_experiment_ids(self) -> None:
        path = self.data["attempts"][0]["raw_result"]["path"]
        raw = json.loads((self.bundle / path).read_text())
        raw["run_id"] = "unrelated-run"
        raw["experiment_id"] = "unrelated-experiment"
        raw["design_sha256"] = "f" * 64
        self.rewrite_artifact(path, raw)
        self.problems("raw_result run_id does not match")
        self.problems("raw_result experiment_id does not match")
        self.problems("raw_result design_sha256 does not match")

    def test_output_binds_exact_succeeded_run_ids(self) -> None:
        path = self.data["workload_evidence"]["output"]["path"]
        output = json.loads((self.bundle / path).read_text())
        output["run_ids"] = ["unrelated-run"]
        self.rewrite_artifact(path, output)
        self.problems("run_ids must exactly match deduplicated succeeded run IDs")

    def test_cpu_query_plan_binds_experiment(self) -> None:
        path = self.data["workload_evidence"]["query_plan"]["path"]
        self.rewrite_artifact(path, {"experiment_id": "wrong"})
        self.problems("query_plan experiment_id does not match")

    def test_required_cpu_evidence_and_layer_are_enforced(self) -> None:
        del self.data["workload_evidence"]["query_plan"]
        self.data["stack"]["layers"] = [item for item in self.data["stack"]["layers"] if item["name"] != "storage"]
        self.problems("missing required roles: query_plan")
        self.problems("missing required layer identities: storage")

    def test_attempt_ids_are_distinct(self) -> None:
        self.data["attempts"][1]["id"] = self.data["attempts"][0]["id"]
        self.problems("attempt IDs must be distinct")

    def test_multiple_native_attempts_can_share_one_real_run_id(self) -> None:
        run_id = self.data["attempts"][0]["run_id"]
        attempt = self.data["attempts"][1]
        attempt["run_id"] = run_id
        path = attempt["raw_result"]["path"]
        raw = json.loads((self.bundle / path).read_text())
        raw["run_id"] = run_id
        self.rewrite_artifact(path, raw)
        output_path = self.data["workload_evidence"]["output"]["path"]
        output = json.loads((self.bundle / output_path).read_text())
        output["run_ids"] = [run_id]
        self.rewrite_artifact(output_path, output)
        self.save()
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_comparison_cannot_hide_or_misclassify_attempt(self) -> None:
        self.data["comparison"]["candidate_attempt_ids"] = []
        self.problems("must exactly match all candidate attempts")

    def test_failed_attempt_cannot_be_accepted(self) -> None:
        self.data["attempts"][0].update(status="failed", failure_reason="interrupted")
        self.problems("failed attempts cannot be accepted")

    def test_correctness_failure_cannot_accept_attempts(self) -> None:
        self.data["correctness"]["accepted"] = False
        self.problems("accepted attempts require accepted correctness")
        self.problems("correctness accepted does not match retained evidence")

    def test_index_is_complete_and_not_duplicated(self) -> None:
        self.data["artifacts"].pop()
        self.problems("artifacts must include every referenced artifact")
        self.data["artifacts"].append(self.data["artifacts"][0])
        self.problems("artifacts paths must be distinct")

    def declare_extra(self, relative: str, contents: str) -> dict[str, str]:
        path = self.bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        artifact = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.data["artifacts"].append(artifact)
        return artifact

    def additional_input_fixture(self, count: int = 2, *, manifest: bool = True) -> list[dict]:
        """Small opaque input fixtures; no workload/index parsing or execution."""
        inputs = [self.declare_extra(f"evidence/inputs/part-{i}.bin", f"opaque part {i}\n") for i in range(count)]
        self.input_workload = {"scope": "synthetic opaque inputs", "input_artifacts": inputs}
        self.stack_input_ref = self.declare_extra("evidence/stack-identity.json", json.dumps({"workload": self.input_workload}) + "\n")
        self.set_input_design(inputs)
        if manifest:
            self.input_manifest = {"schemaVersion": "piq.finite-experiment-executable.v1",
                "files": [*self.data["experiment"]["source_artifacts"], self.data["workload_evidence"]["input"],
                          *inputs, self.data["experiment"]["design"], self.stack_input_ref],
                "finiteExperiment": {"stack": [{"layer": "workload", "identitySha256": ingestion_metadata_sha256(self.input_workload)}]}}
            self.declare_extra("executable-manifest.json", json.dumps(self.input_manifest) + "\n")
        self.save()
        return inputs

    def set_input_design(self, inputs: object) -> None:
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["input_artifacts"] = inputs
        self.rewrite_artifact(design_path, design)
        for attempt in self.data["attempts"]:
            raw_path = attempt["raw_result"]["path"]
            raw = json.loads((self.bundle / raw_path).read_text())
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
            self.rewrite_artifact(raw_path, raw)

    def test_additional_inputs_bind_and_import_opaque_bytes(self) -> None:
        inputs = self.additional_input_fixture()
        self.assertEqual(validate_bundle(self.bundle), [])
        destination = self.root / "imported-inputs"
        import_bundle(self.bundle, destination)
        self.assertEqual(validate_bundle(destination), [])
        for item in inputs:
            self.assertEqual((self.bundle / item["path"]).read_bytes(), (destination / item["path"]).read_bytes())

    def test_additional_inputs_allow_absent_original_manifest(self) -> None:
        self.additional_input_fixture(manifest=False)
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_additional_inputs_require_complete_index_and_unchanged_bytes(self) -> None:
        inputs = self.additional_input_fixture()
        self.data["artifacts"] = [item for item in self.data["artifacts"] if item["path"] != inputs[0]["path"]]
        self.problems("artifacts must include every referenced artifact")
        self.data["artifacts"].append(inputs[0])
        (self.bundle / inputs[0]["path"]).write_text("changed opaque bytes")
        self.problems("sha256 does not match evidence/inputs/part-0.bin")

    def test_additional_inputs_require_indexed_stack_and_original_manifest(self) -> None:
        self.additional_input_fixture()
        for relative, phrase in (("evidence/stack-identity.json", "indexed stack-identity workload"),
                                 ("executable-manifest.json", "exactly one indexed executable-manifest")):
            with self.subTest(relative=relative):
                original = self.data["artifacts"]
                self.data["artifacts"] = [item for item in original if item["path"] != relative]
                self.problems(phrase)
                self.data["artifacts"] = original

    def test_additional_inputs_require_matching_stack_and_manifest_parts(self) -> None:
        inputs = self.additional_input_fixture()
        self.rewrite_artifact("evidence/stack-identity.json", {"workload": {**self.input_workload, "input_artifacts": inputs[:1]}})
        self.problems("stack-identity workload input_artifacts do not match")
        self.rewrite_artifact("evidence/stack-identity.json", {"workload": self.input_workload})
        self.input_manifest["files"] = [item for item in self.input_manifest["files"] if item["path"] != inputs[1]["path"]]
        self.rewrite_artifact("executable-manifest.json", self.input_manifest)
        self.problems("original executable manifest input artifacts do not match")

    def test_additional_inputs_bind_original_workload_identity_and_design(self) -> None:
        self.additional_input_fixture()
        self.input_manifest["finiteExperiment"]["stack"][0]["identitySha256"] = "0" * 64
        self.input_manifest["files"] = [item for item in self.input_manifest["files"] if item["path"] != "evidence/design.json"]
        self.rewrite_artifact("executable-manifest.json", self.input_manifest)
        self.problems("workload identity does not bind additional")
        self.problems("files must bind design")

    def test_omitted_additional_inputs_cannot_hide_retained_manifest_parts(self) -> None:
        inputs = self.additional_input_fixture()
        self.set_input_design(inputs[:1])
        self.rewrite_artifact("evidence/stack-identity.json", {"workload": {**self.input_workload, "input_artifacts": inputs[:1]}})
        self.problems("original executable manifest input artifacts do not match")

    def test_missing_design_field_cannot_hide_additional_inputs(self) -> None:
        self.additional_input_fixture()
        relative = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / relative).read_text())
        del design["input_artifacts"]
        self.rewrite_artifact(relative, design)
        self.problems("input_artifacts must be a list of 1 to 16 references")

    def test_additional_inputs_require_bounded_nonempty_list(self) -> None:
        self.additional_input_fixture(manifest=False)
        for value in ([], None, {}, [self.data["workload_evidence"]["input"]] * 17):
            with self.subTest(value=value):
                self.set_input_design(value)
                self.problems("must be a list of 1 to 16 references")

    def test_additional_inputs_reject_duplicate_case_and_reserved_paths(self) -> None:
        inputs = self.additional_input_fixture(manifest=False)
        for additional in ([inputs[0], inputs[0]], [inputs[0], {**inputs[0], "path": "evidence/inputs/PART-0.bin"}],
                           [self.data["workload_evidence"]["input"]]):
            with self.subTest(additional=additional):
                self.set_input_design(additional)
                self.problems("must be distinct and must not collide")
        self.set_input_design(inputs)
        self.declare_extra("evidence/inputs/PART-0.bin", "separate indexed bytes\n")
        self.problems("must be distinct and must not collide")

    def test_additional_inputs_require_portable_names_and_safe_references(self) -> None:
        inputs = self.additional_input_fixture(manifest=False)
        for relative in ("../part.bin", "evidence/inputs/nested/part.bin", "evidence/inputs/CON.bin", "evidence/inputs/part."):
            with self.subTest(relative=relative):
                self.set_input_design([{**inputs[0], "path": relative}])
                self.problems("portable evidence/inputs filename")
        self.set_input_design([{**inputs[0], "extra": True}])
        self.problems("must contain exactly path and sha256")

    def test_additional_input_byte_caps_use_actual_file_metadata(self) -> None:
        inputs = self.additional_input_fixture(count=8, manifest=False)
        real_stat = Path.stat
        sizes: dict[Path, int] = {}

        def bounded_stat(path: Path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if path in sizes:
                values = list(observed)
                values[6] = sizes[path]
                return os.stat_result(values)
            return observed

        with patch.object(Path, "stat", bounded_stat):
            sizes[self.bundle / inputs[0]["path"]] = 8 * 1024 * 1024 + 1
            self.problems("exceeds the 8 MiB input file limit")
            sizes.clear()
            sizes.update({self.bundle / item["path"]: 8 * 1024 * 1024 for item in inputs})
            self.problems("exceed the 64 MiB combined limit")
            main_bytes = real_stat(self.bundle / self.data["workload_evidence"]["input"]["path"]).st_size
            sizes[self.bundle / inputs[-1]["path"]] -= main_bytes
            self.save()
            self.assertEqual(validate_bundle(self.bundle), [])
            sizes.clear()
            sizes[self.bundle / self.data["workload_evidence"]["input"]["path"]] = 8 * 1024 * 1024 + 1
            self.problems("main input exceeds the 8 MiB")

    def formal_fixture(self, configuration: dict | None = None, *, gpu: bool = False) -> None:
        """Synthetic producer metadata; deliberately does not run an oracle."""
        configuration = configuration or {"oracle": "json-numeric-tolerance", "atol": "0.01", "rtol": "0"}
        if configuration["oracle"] == "json-keyed-record-set-equality":
            operand = json.dumps([{configuration["recordKey"]: "synthetic", "value": 1}]) + "\n"
            candidate_operand = operand
        else:
            operand = '{"sum":1}\n'
            candidate_operand = '{"sum":1.001}\n' if configuration["oracle"] == "json-numeric-tolerance" else operand
        reference = self.declare_extra("oracle/reference.json", operand)
        candidate = self.declare_extra("oracle/candidate.json", candidate_operand)
        report = self.declare_extra("oracle/report.json", json.dumps({
            "schemaVersion": "piq.finite-correctness-report.v1", "oracle": configuration["oracle"],
            "status": "pass", "reasonCode": "synthetic-pass", "comparedValues": 1,
            "configurationSha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "referenceSha256": reference["sha256"], "candidateSha256": candidate["sha256"],
        }) + "\n")
        self.data["workload_evidence"].update(oracle_reference=reference, oracle_candidate=candidate, correctness_report=report)
        self.rewrite_artifact(self.data["correctness"]["path"], {
            "experiment_id": self.data["experiment"]["id"], "accepted": True,
            "check": {"configuration": configuration, "reference": reference, "candidate": candidate,
                      "report": report, "verification": "producer-asserted"},
        })
        resources = {"executor": "local-process", "cpuCount": 2, "memoryLimitMiB": 512,
                     "timeoutSeconds": 10, "acceleratorCount": 0, "nodeCount": 1, "processCount": 2}
        if gpu:
            resources.update(executor="assigned-gpu", acceleratorCount=1, processCount=1, targetId="synthetic-target")
            self.data["stack"]["layers"].append({"name": "accelerator", "identifier": "synthetic-GPU"})
        process_count = resources["processCount"]
        self.native_results = []
        execution_attempts, arms = [], []
        for i, attempt in enumerate(self.data["attempts"]):
            arm_id = f"arm-{i}"
            arm_configuration = {"processCount": process_count, "label": "synthetic π"}
            arms.append({"arm_id": arm_id, "configuration": arm_configuration,
                         "configuration_sha256": hashlib.sha256(json.dumps(arm_configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
            census = []
            for rank in range(process_count):
                artifact = self.declare_extra(f"attempts/{attempt['id']}/rank-{rank}.json", '{"sum":1}\n')
                census.append({"rank": rank, "localRank": rank, "host": "synthetic-host",
                               "processId": 100 + rank, "deviceType": "cuda" if gpu else "cpu",
                               "deviceUuid": f"GPU-synthetic-{rank}" if gpu else None,
                               "output": {"artifactName": artifact["path"], "sha256": artifact["sha256"]}})
            execution_attempts.append({"attempt_id": attempt["id"], "arm_id": arm_id,
                                       "expected_process_count": process_count, "rank_census": census})
            raw_path = attempt["raw_result"]["path"]
            raw = json.loads((self.bundle / raw_path).read_text())
            raw.update(native_attempt={"armId": arm_id, "state": "completed", "rankCensus": census},
                       arm_configuration=arm_configuration)
            self.native_results.append((raw_path, raw))
        self.execution = {"experiment_id": self.data["experiment"]["id"], "run_id": "synthetic-run",
                          "resources": resources, "attempts": execution_attempts}
        self.data["workload_evidence"]["execution"] = self.declare_extra("evidence/execution.json", json.dumps(self.execution) + "\n")
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design.update(correctness=configuration, resources=resources, arms=arms,
                      required_evidence_roles=sorted(self.data["workload_evidence"]))
        self.rewrite_artifact(design_path, design)
        for path, raw in self.native_results:
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
            self.rewrite_artifact(path, raw)
        if gpu:
            source = self.data["experiment"]["source_artifacts"][0]
            input_ref = self.data["workload_evidence"]["input"]
            plan = {"planSha256": "a" * 64, "experimentId": self.data["experiment"]["id"], "resources": dict(resources),
                    "source": {"artifactName": source["path"], "sha256": source["sha256"]},
                    "input": {"artifactName": input_ref["path"], "sha256": input_ref["sha256"]},
                    "stack": [{"layer": "runtime", "identitySha256": "b" * 64}]}
            manifest = {"schemaVersion": "piq.finite-experiment-executable.v1", "runId": "synthetic-run",
                        "finiteExperiment": plan, "executable": {"path": "/synthetic/python", "sha256": "c" * 64}}
            original = self.declare_extra("executable-manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
            hashes = {"planSha256": plan["planSha256"], "sourceSha256": source["sha256"],
                      "inputSha256": input_ref["sha256"], "runtimeIdentitySha256": "b" * 64,
                      "runtimeExecutableSha256": "c" * 64}
            binding = {"schemaVersion": "piq.finite-slurm-job-binding.v1", "manifestSha256": original["sha256"],
                       **hashes, "ownerEntrypoint": None, "ownerUid": 1000, "targetId": "synthetic-target",
                       "targetNode": "synthetic-host", "partition": "synthetic", "cpuCount": 2,
                       "memoryLimitMiB": 512, "wallTimeSeconds": 30, "acceleratorCount": 1,
                       "processCount": 1, "nodeCount": 1, "outputDirectory": "/synthetic/output"}
            binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            observation = {"schemaVersion": "piq.finite-slurm-observation.v1", **hashes,
                           "jobId": "123", "state": "RUNNING", "ownerUid": 1000,
                           "targetId": "synthetic-target", "nodeName": "synthetic-host",
                           "observedAt": "2026-10-27T12:00:01Z", "startedAt": "2026-10-27T12:00:00Z",
                           "expiresAt": "2026-10-27T12:00:30Z", "cpuCount": 2, "memoryLimitMiB": 512,
                           "wallTimeSeconds": 30, "acceleratorCount": 1, "nodeCount": 1, "processCount": 1,
                           "deviceUuid": "GPU-synthetic-0", "deviceIndex": 0, "processId": 100,
                           "cgroupPath": "/slurm/uid_1000/job_123/step_0", "cgroupMemoryMaxBytes": 512 * 1024 * 1024,
                           "jobBinding": binding, "jobBindingSha256": binding_sha, "jobComment": "piq-finite:" + binding_sha}
            assignment = self.declare_extra("evidence/assignment.json", json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n")
            resources["assignmentEvidence"] = {"artifactName": assignment["path"], "sha256": assignment["sha256"], "observation": observation}
            self.save_execution()
        self.save()

    def single_cpu_fixture(self) -> None:
        """Synthetic shape from the retained compression capture; no execution."""
        self.formal_fixture({"oracle": "output-sha256-equality"})
        del self.data["workload_evidence"]["execution"]
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["resources"]["processCount"] = 1
        design["required_evidence_roles"] = sorted(self.data["workload_evidence"])
        for arm in design["arms"]:
            arm["configuration"]["processCount"] = 1
            arm["configuration_sha256"] = hashlib.sha256(json.dumps(arm["configuration"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.rewrite_artifact(design_path, design)
        for i, ((path, raw), attempt, arm) in enumerate(zip(self.native_results, self.data["attempts"], design["arms"])):
            output = self.data["workload_evidence"]["oracle_reference" if i == 0 else "oracle_candidate"]
            raw["native_attempt"] = {
                "attemptId": attempt["id"], "armId": arm["arm_id"], "state": "completed",
                "childPid": 1000 + i, "exitCode": 0, "elapsedSeconds": "0.1",
                "startedAt": "2026-10-27T12:00:00.1Z", "completedAt": "2026-10-27T12:00:00.2Z",
                "output": {"artifactName": output["path"], "sha256": output["sha256"]},
            }
            raw["arm_configuration"] = arm["configuration"]
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
            self.rewrite_artifact(path, raw)
        self.save()

    def test_single_cpu_capture_without_rank_records_is_valid(self) -> None:
        self.single_cpu_fixture()
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_single_cpu_requires_actual_positive_pid(self) -> None:
        self.single_cpu_fixture()
        path, raw = self.native_results[0]
        raw["native_attempt"]["childPid"] = True
        self.rewrite_artifact(path, raw)
        self.problems("positive childPid")

    def test_single_cpu_requires_matching_output_bytes(self) -> None:
        self.single_cpu_fixture()
        path, raw = self.native_results[0]
        raw["native_attempt"]["output"]["sha256"] = "0" * 64
        self.rewrite_artifact(path, raw)
        self.problems("single CPU output")

    def test_single_cpu_requires_window_inside_objective(self) -> None:
        self.single_cpu_fixture()
        path, raw = self.native_results[0]
        raw["native_attempt"]["completedAt"] = "2026-10-27T12:00:02Z"
        self.rewrite_artifact(path, raw)
        self.problems("ordered UTC window")

    def test_single_cpu_cannot_hide_distributed_rank_records(self) -> None:
        self.single_cpu_fixture()
        path, raw = self.native_results[0]
        raw["native_attempt"]["rankCensus"] = self.execution["attempts"][0]["rank_census"]
        self.rewrite_artifact(path, raw)
        self.problems("execution is required")

    def test_single_cpu_cannot_hide_multinode_or_gpu_resources(self) -> None:
        self.single_cpu_fixture()
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["resources"].update(nodeCount=2, acceleratorCount=1)
        self.rewrite_artifact(design_path, design)
        self.problems("execution is required")

    def test_single_cpu_resource_boolean_cannot_bypass_execution(self) -> None:
        self.single_cpu_fixture()
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["resources"]["processCount"] = True
        self.rewrite_artifact(design_path, design)
        self.problems("execution is required")

    def test_single_cpu_still_binds_arm_configuration(self) -> None:
        self.single_cpu_fixture()
        path, raw = self.native_results[0]
        raw["arm_configuration"] = {"unexpected": "changed"}
        self.rewrite_artifact(path, raw)
        self.problems("arm identity/configuration")

    def save_execution(self, *, synchronize_native: bool = True) -> None:
        self.rewrite_artifact("evidence/execution.json", self.execution)
        if synchronize_native:
            for (path, raw), attempt in zip(self.native_results, self.execution["attempts"]):
                raw["native_attempt"]["rankCensus"] = attempt["rank_census"]
                self.rewrite_artifact(path, raw)
        self.save()

    def save_gpu_observation(self) -> None:
        assignment = self.execution["resources"]["assignmentEvidence"]
        relative = assignment["artifactName"]
        path = self.bundle / relative
        path.write_text(json.dumps(assignment["observation"], sort_keys=True, separators=(",", ":")) + "\n")
        assignment["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for artifact in self.data["artifacts"]:
            if artifact["path"] == relative:
                artifact["sha256"] = assignment["sha256"]
        self.save_execution()

    def test_formal_distributed_fixture_imports_as_unqualified(self) -> None:
        self.formal_fixture()
        self.assertEqual(validate_bundle(self.bundle), [])
        destination = self.root / "distributed"
        import_bundle(self.bundle, destination)
        self.assertEqual(validate_bundle(destination), [])
        receipt = json.loads((destination / "import-receipt.json").read_text())
        self.assertEqual(receipt["qualification"], "not_established")
        self.assertIs(receipt["publishable"], False)

    def test_formal_output_hash_configuration(self) -> None:
        self.formal_fixture({"oracle": "output-sha256-equality"})
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_formal_keyed_configuration_uses_utf8_canonical_hash(self) -> None:
        self.formal_fixture({"oracle": "json-keyed-record-set-equality", "recordKey": "κλειδί"})
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_formal_correctness_requires_exact_design_configuration(self) -> None:
        self.formal_fixture()
        path = self.data["correctness"]["path"]
        payload = json.loads((self.bundle / path).read_text())
        del payload["check"]["configuration"]
        self.rewrite_artifact(path, payload)
        self.problems("configuration does not match experiment design correctness")

    def test_missing_formal_design_configuration_is_rejected(self) -> None:
        self.formal_fixture()
        path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / path).read_text())
        del design["correctness"]
        self.rewrite_artifact(path, design)
        self.problems("configuration must be an object")

    def test_nonfinite_and_noncanonical_tolerances_are_rejected(self) -> None:
        from tools.piq_experiment_evidence import correctness_configuration
        for value in ("NaN", "Infinity", "-0.1", "1e-3", "00.1", 0.1, True, "0" * 65):
            with self.subTest(value=value):
                errors = []
                self.assertIsNone(correctness_configuration({"oracle": "json-numeric-tolerance", "atol": value, "rtol": "0"}, errors))
                self.assertTrue(any("finite nonnegative decimal string" in error for error in errors), errors)

    def test_forged_independent_correctness_verification_is_rejected(self) -> None:
        self.formal_fixture()
        path = self.data["correctness"]["path"]
        payload = json.loads((self.bundle / path).read_text())
        payload["check"]["verification"] = "server-verified"
        self.rewrite_artifact(path, payload)
        self.problems("verification must equal producer-asserted")

    def test_formal_report_configuration_operand_and_acceptance_bindings(self) -> None:
        self.formal_fixture()
        path = self.data["workload_evidence"]["correctness_report"]["path"]
        report = json.loads((self.bundle / path).read_text())
        report.update(configurationSha256="f" * 64, referenceSha256="e" * 64,
                      candidateSha256="d" * 64, status="fail", comparedValues=True)
        self.rewrite_artifact(path, report)
        self.problems("configurationSha256 does not match retained check")
        self.problems("referenceSha256 does not match retained check")
        self.problems("candidateSha256 does not match retained check")
        self.problems("status does not match accepted")
        self.problems("comparedValues must be a nonnegative integer")

    def test_formal_roles_cannot_substitute_or_omit_operands(self) -> None:
        self.formal_fixture()
        del self.data["workload_evidence"]["oracle_candidate"]
        self.data["workload_evidence"]["oracle_reference"] = self.data["workload_evidence"]["input"]
        self.problems("oracle_candidate must exactly match")
        self.problems("oracle_reference must exactly match")

    def test_completed_execution_requires_every_rank_and_output(self) -> None:
        self.formal_fixture()
        self.execution["attempts"][0]["rank_census"].pop()
        del self.execution["attempts"][1]["rank_census"][0]["output"]
        self.save_execution()
        self.problems("requires an exact complete rank census")
        self.problems("rank output.path must be a safe bundle-relative artifact path")

    def test_execution_rank_output_hash_and_index_are_required(self) -> None:
        self.formal_fixture()
        output = self.execution["attempts"][0]["rank_census"][0]["output"]
        self.data["artifacts"] = [artifact for artifact in self.data["artifacts"] if artifact["path"] != output["artifactName"]]
        self.problems("artifacts must include every referenced artifact")
        output["sha256"] = "f" * 64
        self.save_execution()
        self.problems("sha256 does not match attempts/baseline-01/rank-0.json")

    def test_rank_identity_and_declared_capacity_are_enforced(self) -> None:
        self.formal_fixture()
        first, second = self.execution["attempts"][0]["rank_census"]
        second.update(processId=first["processId"], localRank=first["localRank"])
        self.execution["attempts"][1]["rank_census"][1]["host"] = "undeclared-second-host"
        self.save_execution()
        self.problems("require distinct process IDs per host")
        self.problems("localRank must be distinct per host")
        self.problems("rank hosts exceed declared nodeCount")

    def test_execution_cannot_change_arm_count_or_native_census(self) -> None:
        self.formal_fixture()
        self.execution["attempts"][0]["expected_process_count"] = 1
        self.execution["attempts"][1]["arm_id"] = "unrelated-arm"
        self.execution["attempts"][1]["rank_census"][0]["processId"] = 999
        # Values came from the same in-memory fixture; sever that alias first.
        path, raw = self.native_results[1]
        raw = json.loads((self.bundle / path).read_text())
        self.native_results[1] = (path, raw)
        self.save_execution(synchronize_native=False)
        self.problems("expected_process_count does not match")
        self.problems("arm identity/configuration does not match")
        self.problems("rank_census does not match retained native attempt")

    def test_resource_arm_design_cannot_omit_execution_evidence(self) -> None:
        self.formal_fixture()
        del self.data["workload_evidence"]["execution"]
        self.problems("execution is required for the retained resource/arm design")

    def test_native_rank_metadata_cannot_substitute_boolean_for_integer(self) -> None:
        self.formal_fixture()
        path, raw = self.native_results[0]
        raw = json.loads((self.bundle / path).read_text())
        raw["native_attempt"]["rankCensus"][1]["rank"] = True
        self.rewrite_artifact(path, raw)
        self.problems("rank_census does not match retained native attempt")

    def test_malformed_execution_metadata_reports_errors(self) -> None:
        self.formal_fixture()
        self.execution["resources"] = []
        self.execution["attempts"][0].update(arm_id=[], expected_process_count=True,
            rank_census=[{"rank": [], "localRank": {}, "host": [], "processId": True,
                          "deviceType": {}, "deviceUuid": [], "output": []}])
        self.save_execution(synchronize_native=False)
        self.problems("execution.resources must be an object")
        self.problems("expected_process_count does not match")

    def test_forged_report_verification_field_is_rejected(self) -> None:
        self.formal_fixture()
        path = self.data["workload_evidence"]["correctness_report"]["path"]
        report = json.loads((self.bundle / path).read_text())
        report["verification"] = "server-verified"
        self.rewrite_artifact(path, report)
        self.problems("report fields do not match")

    def test_gpu_assignment_reference_is_retained_as_consistency_only(self) -> None:
        self.formal_fixture(gpu=True)
        self.execution["resources"]["assignmentEvidence"]["observation"]["targetId"] = "unrelated-target"
        self.save_execution()
        self.problems("GPU assignment evidence must match the retained Slurm observation")

    def test_gpu_slurm_observation_binds_source_job_comment_and_native_pid(self) -> None:
        self.formal_fixture(gpu=True)
        observation = self.execution["resources"]["assignmentEvidence"]["observation"]
        observation.update(sourceSha256="f" * 64, jobComment="unbound-job", processId=999)
        self.save_gpu_observation()
        self.problems("sourceSha256 does not match manifest/resources")
        self.problems("jobComment does not match retained job binding")
        self.problems("processId must match an observed native rank")

    def test_gpu_slurm_job_projection_and_observed_device_are_bound(self) -> None:
        self.formal_fixture(gpu=True)
        observation = self.execution["resources"]["assignmentEvidence"]["observation"]
        observation["jobBinding"]["partition"] = "forged-partition"
        observation["deviceUuid"] = "GPU-unrelated"
        self.save_gpu_observation()
        self.problems("jobBindingSha256 does not match retained jobBinding")
        self.problems("GPU rank host/device does not match retained Slurm observation")

    def test_gpu_preflight_failure_keeps_unobserved_assignment_absent(self) -> None:
        self.formal_fixture(gpu=True)
        del self.execution["resources"]["assignmentEvidence"]
        for attempt, execution_attempt, (_, raw) in zip(self.data["attempts"], self.execution["attempts"], self.native_results):
            attempt.update(status="failed", accepted=False, failure_reason="synthetic preflight refusal")
            raw["native_attempt"].update(state="not-started", rankCensus=[])
            execution_attempt["rank_census"] = []
        self.data["correctness"]["accepted"] = False
        report_path = self.data["workload_evidence"]["correctness_report"]["path"]
        report = json.loads((self.bundle / report_path).read_text())
        report["status"] = "fail"
        self.rewrite_artifact(report_path, report)
        correctness_path = self.data["correctness"]["path"]
        correctness = json.loads((self.bundle / correctness_path).read_text())
        correctness["accepted"] = False
        correctness["check"]["report"] = dict(self.data["workload_evidence"]["correctness_report"])
        self.rewrite_artifact(correctness_path, correctness)
        output_path = self.data["workload_evidence"]["output"]["path"]
        output = json.loads((self.bundle / output_path).read_text())
        output["run_ids"] = []
        self.rewrite_artifact(output_path, output)
        self.save_execution()
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_gpu_serving_lease_cannot_replace_slurm_observation(self) -> None:
        self.formal_fixture(gpu=True)
        assignment = self.execution["resources"]["assignmentEvidence"]
        assignment["leaseContext"] = {"leaseIdentity": {"targetId": "synthetic-target"}}
        del assignment["observation"]
        self.save_execution()
        self.problems("serving leaseContext is not finite execution evidence")

    def test_failed_formal_attempt_preserves_missing_oracle_pair(self) -> None:
        self.formal_fixture()
        for role in ("oracle_reference", "oracle_candidate", "correctness_report"):
            del self.data["workload_evidence"][role]
        self.data["correctness"]["accepted"] = False
        self.rewrite_artifact(self.data["correctness"]["path"], {"experiment_id": self.data["experiment"]["id"], "accepted": False})
        self.data["attempts"][0].update(status="failed", accepted=False, failure_reason="synthetic interruption")
        self.data["attempts"][1]["accepted"] = False
        self.native_results[0][1]["native_attempt"]["state"] = "failed"
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["required_evidence_roles"] = sorted(self.data["workload_evidence"])
        self.rewrite_artifact(design_path, design)
        for path, raw in self.native_results:
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
        self.save_execution()
        self.assertEqual(validate_bundle(self.bundle), [])

    def test_gpu_metadata_requires_assignment_layer_and_device_identity(self) -> None:
        self.formal_fixture(gpu=True)
        self.assertEqual(validate_bundle(self.bundle), [])
        self.data["stack"]["layers"] = [layer for layer in self.data["stack"]["layers"] if layer["name"] != "accelerator"]
        del self.execution["resources"]["assignmentEvidence"]
        self.execution["attempts"][0]["rank_census"][0].update(deviceType="cpu", deviceUuid=None)
        self.execution["attempts"][1]["rank_census"][0]["deviceUuid"] = None
        self.save_execution()
        self.problems("GPU execution requires an accelerator stack layer")
        self.problems("observed GPU execution requires retained Slurm assignmentEvidence")
        self.problems("GPU execution cannot substitute a CPU rank")
        self.problems("CUDA rank requires GPU resources and a device UUID")

    def test_cpu_cannot_claim_cuda_without_gpu_resources(self) -> None:
        self.formal_fixture()
        self.execution["attempts"][0]["rank_census"][0].update(deviceType="cuda", deviceUuid="GPU-fake")
        self.save_execution()
        self.problems("CUDA rank requires GPU resources and a device UUID")

    def test_additional_native_and_api_custody_evidence_is_retained(self) -> None:
        native = self.declare_extra("native/reference-output.csv", "category,total\na,1\n")
        custody = self.declare_extra("api/submission-confirmation.json", '{"accepted":true,"runId":"synthetic-run"}\n')
        self.save()
        self.assertEqual(validate_bundle(self.bundle), [])
        destination = self.root / "with-native-custody"
        import_bundle(self.bundle, destination)
        for extra in (native, custody):
            self.assertEqual((self.bundle / extra["path"]).read_bytes(), (destination / extra["path"]).read_bytes())

    def test_conflicting_extra_index_hash_is_rejected(self) -> None:
        extra = self.declare_extra("api/custody.json", '{"accepted":true}\n')
        self.data["artifacts"].append({**extra, "sha256": "f" * 64})
        self.problems("conflicts with another hash")
        self.problems("artifacts paths must be distinct")

    def test_mutated_extra_evidence_is_rejected(self) -> None:
        extra = self.declare_extra("api/custody.json", '{"accepted":true}\n')
        (self.bundle / extra["path"]).write_text('{"accepted":false}\n')
        self.problems("sha256 does not match api/custody.json")

    def test_extra_index_symlink_is_rejected(self) -> None:
        extra = self.declare_extra("api/custody.json", '{"accepted":true}\n')
        path = self.bundle / extra["path"]
        target = self.bundle / "api/real-custody.json"
        path.rename(target)
        path.symlink_to(target.name)
        self.problems("must not use a symlink")

    def test_unsafe_path_is_rejected(self) -> None:
        for path in ("../outside.csv", "/etc/passwd", "evidence/../input.csv", "evidence//input.csv", "evidence\\input.csv"):
            with self.subTest(path=path):
                self.data["workload_evidence"]["input"]["path"] = path
                self.problems("safe bundle-relative artifact path")

    def test_symlink_even_inside_bundle_is_rejected(self) -> None:
        path = self.bundle / self.data["workload_evidence"]["input"]["path"]
        moved = self.bundle / "evidence/input-real.csv"
        path.rename(moved)
        path.symlink_to(moved.name)
        self.problems("must not use a symlink")

    def test_reversed_window_fails(self) -> None:
        self.data["objective"]["window"]["end"] = self.data["objective"]["window"]["start"]
        self.problems("start before end")

    def test_malformed_enums_return_errors(self) -> None:
        self.data["objective"]["direction"] = {}
        self.data["attempts"][0]["status"] = []
        self.data["attempts"][0]["role"] = {}
        self.data["comparison"]["scientific_disposition"] = {}
        self.problems("objective.direction")

    def test_success_requires_explicit_null_failure_reason(self) -> None:
        del self.data["attempts"][0]["failure_reason"]
        self.problems("missing fields: failure_reason")

    def test_import_preserves_bytes_and_declares_unqualified(self) -> None:
        (self.bundle / "unindexed-secret.txt").write_text("must not be imported")
        destination = self.root / "imported"
        import_bundle(self.bundle, destination)
        self.assertEqual(validate_bundle(destination), [])
        self.assertEqual((self.bundle / "envelope.json").read_bytes(), (destination / "envelope.json").read_bytes())
        for artifact in self.data["artifacts"]:
            self.assertEqual((self.bundle / artifact["path"]).read_bytes(), (destination / artifact["path"]).read_bytes())
        self.assertFalse((destination / "unindexed-secret.txt").exists())
        receipt = json.loads((destination / "import-receipt.json").read_text())
        self.assertIs(receipt["publishable"], False)
        self.assertEqual(receipt["qualification"], "not_established")

    def test_import_refuses_existing_destination(self) -> None:
        destination = self.root / "owner-work"
        destination.mkdir()
        (destination / "owner.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "already exists"):
            import_bundle(self.bundle, destination)
        self.assertEqual((destination / "owner.txt").read_text(), "preserve")

    def test_invalid_bundle_does_not_create_destination(self) -> None:
        self.data["scope"] = "qualified"
        self.save()
        destination = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "diagnostic scope only"):
            import_bundle(self.bundle, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
