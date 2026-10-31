from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.assemble_piq_two_host_owner_receipt import (
    _atomic_create_json,
    assemble_owner_receipt_data,
)
from tools.build_piq_two_host_owner_handback import build_owner_handback
from tools.check_piq_two_host_functional import (
    READINESS_BLOCK,
    _load_json,
    _safe_relative,
    _validated_ref,
    main as two_host_functional_main,
    validate_declared_two_host_functional,
)
from tools.check_piq_two_host_owner_authority import (
    OWNER_RECEIPT_PATH,
    owner_admission_tuple,
    validate_two_host_owner_authority_receipt,
)
from tools.piq_two_host_owner_admissions import (
    ADMISSIONS_PATH,
    validate_admissions,
)
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

    def scoped_input_fixture(self, family: str = "cpu_artifact_codec", *, count: int = 4,
                             design_family: str | None = None) -> list[dict]:
        """Synthetic small files. Declared metadata sizes never run a workload."""
        inputs = self.additional_input_fixture(count=count, manifest=False)
        self.data["workload_family"] = family
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["workload_family"] = family if design_family is None else design_family
        self.rewrite_artifact(design_path, design)
        self.set_input_design(inputs)
        self.save()
        return inputs

    def test_codec_input_caps_match_exact_file_and_combined_boundaries(self) -> None:
        inputs = self.scoped_input_fixture()
        main = self.bundle / self.data["workload_evidence"]["input"]["path"]
        real_stat = Path.stat
        sizes = {self.bundle / item["path"]: 32 * 1024 * 1024 for item in inputs}
        sizes[self.bundle / inputs[-1]["path"]] -= real_stat(main).st_size

        def sized(path: Path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if path in sizes:
                values = list(observed)
                values[6] = sizes[path]
                return os.stat_result(values)
            return observed

        with patch.object(Path, "stat", sized):
            self.assertEqual(validate_bundle(self.bundle), [])
            sizes[self.bundle / inputs[-1]["path"]] += 1
            self.problems("exceed the 128 MiB combined limit")
            sizes[self.bundle / inputs[-1]["path"]] -= 1
            sizes[self.bundle / inputs[0]["path"]] += 1
            self.problems("exceeds the 32 MiB input file limit")
            sizes.clear()
            sizes[main] = 32 * 1024 * 1024
            self.save()
            self.assertEqual(validate_bundle(self.bundle), [])
            sizes[main] += 1
            self.problems("main input exceeds the 32 MiB input file limit")

    def test_codec_aliases_and_unknown_families_keep_generic_input_limits(self) -> None:
        real_stat = Path.stat
        sizes: dict[Path, int] = {}

        def sized(path: Path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if path in sizes:
                values = list(observed)
                values[6] = sizes[path]
                return os.stat_result(values)
            return observed

        for family in ("cpu_artifact_compression", "cpu_two_host_reduction", "CPU_ARTIFACT_CODEC",
                       "cpu_artifact_codec_extra", "future_workload"):
            with self.subTest(family=family):
                # A fresh fixture avoids duplicate indexed paths across families.
                shutil.rmtree(self.bundle)
                shutil.copytree(EXAMPLE, self.bundle)
                self.data = json.loads((self.bundle / "envelope.json").read_text())
                inputs = self.scoped_input_fixture(family, count=1)
                sizes.clear()
                sizes[self.bundle / inputs[0]["path"]] = 8 * 1024 * 1024 + 1
                with patch.object(Path, "stat", sized):
                    self.problems("exceeds the 8 MiB input file limit")

    def test_codec_design_mismatch_cannot_raise_input_limits(self) -> None:
        inputs = self.scoped_input_fixture(design_family="cpu_artifact_compression", count=1)
        real_stat = Path.stat
        selected = self.bundle / inputs[0]["path"]

        def sized(path: Path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if path == selected:
                values = list(observed)
                values[6] = 8 * 1024 * 1024 + 1
                return os.stat_result(values)
            return observed

        with patch.object(Path, "stat", sized):
            self.problems("experiment design workload_family does not match envelope")
            self.problems("exceeds the 8 MiB input file limit")

    def test_codec_import_rechecks_bytes_and_stays_diagnostic(self) -> None:
        inputs = self.scoped_input_fixture(count=1)
        destination = self.root / "imported-codec"
        real_stat = Path.stat
        selected_name = Path(inputs[0]["path"]).name

        def sized(path: Path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if path.name == selected_name:
                values = list(observed)
                values[6] = 32 * 1024 * 1024
                return os.stat_result(values)
            return observed

        with patch.object(Path, "stat", sized):
            self.assertEqual(validate_bundle(self.bundle), [])
            import_bundle(self.bundle, destination)
            self.assertEqual(validate_bundle(destination), [])
        imported = json.loads((destination / "envelope.json").read_text())
        self.assertEqual(imported["scope"], "diagnostic")
        self.assertEqual(imported["comparison"]["scientific_disposition"], "DIAGNOSTIC_ONLY")
        self.assertEqual((self.bundle / inputs[0]["path"]).read_bytes(),
                         (destination / inputs[0]["path"]).read_bytes())
        (destination / inputs[0]["path"]).write_bytes(b"changed synthetic retained input")
        self.assertTrue(any("sha256 does not match" in error for error in validate_bundle(destination)))

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
            operand_ref = reference if i == 0 else candidate
            native_started = "2026-10-27T12:00:00.100Z" if i == 0 else "2026-10-27T12:00:00.500Z"
            native_completed = "2026-10-27T12:00:00.400Z" if i == 0 else "2026-10-27T12:00:00.800Z"
            raw.update(native_attempt={"armId": arm_id, "state": "completed", "rankCensus": census,
                                       "startedAt": native_started, "completedAt": native_completed,
                                       "output": {"artifactName": operand_ref["path"],
                                                  "sha256": operand_ref["sha256"]}},
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

    def assigned_cpu_hosts_fixture(self) -> None:
        """Synthetic retained metadata for the bounded two-host CPU shape."""
        self.formal_fixture()
        hosts = [
            {
                "targetId": f"synthetic-host-{index}",
                "hostIdentitySha256": str(index + 1) * 64,
                "pythonExecutable": "/usr/bin/python3",
                "pythonExecutableSha256": str(index + 3) * 64,
                "cpuCount": 2,
                "memoryLimitMiB": 512,
            }
            for index in range(2)
        ]
        resources = {
            "executor": "assigned-cpu-hosts",
            "cpuCount": 4,
            "memoryLimitMiB": 1024,
            "timeoutSeconds": 20,
            "acceleratorCount": 0,
            "nodeCount": 2,
            "processCount": 2,
            "cpuHosts": hosts,
        }
        self.execution["resources"] = resources
        for attempt in self.execution["attempts"]:
            for rank in attempt["rank_census"]:
                rank["host"] = hosts[rank["rank"]]["targetId"]
                rank["localRank"] = 0
        design_path = self.data["experiment"]["design"]["path"]
        design = json.loads((self.bundle / design_path).read_text())
        design["resources"] = resources
        self.rewrite_artifact(design_path, design)
        for _, raw in self.native_results:
            raw["design_sha256"] = self.data["experiment"]["design"]["sha256"]
        self.save_execution()

    def install_two_host_functional_receipt(
        self, attempt_ids: list[str] | None = None
    ) -> tuple[Path, dict, dict]:
        """Import the synthetic bundle under evidence/ and bind an opt-in receipt."""
        package = self.root / "package"
        bundle = package / "evidence/functional-run"
        import_bundle(self.bundle, bundle)
        self.bundle = bundle
        envelope_path = bundle / "envelope.json"
        envelope = json.loads(envelope_path.read_text())

        def reference(path: Path) -> dict[str, str]:
            return {
                "path": path.relative_to(package).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        attempt_ids = attempt_ids or [envelope["attempts"][1]["id"]]
        run_id = envelope["attempts"][0]["run_id"]
        envelope_ref = reference(envelope_path)
        execution = envelope["workload_evidence"]["execution"]
        correctness = envelope["correctness"]
        execution_data = json.loads((bundle / execution["path"]).read_text())
        correctness_data = json.loads((bundle / correctness["path"]).read_text())
        aggregation_refs = []
        for attempt_id in attempt_ids:
            attempt = next(item for item in envelope["attempts"] if item["id"] == attempt_id)
            observed = next(
                item for item in execution_data["attempts"] if item["attempt_id"] == attempt_id
            )
            raw = json.loads((bundle / attempt["raw_result"]["path"]).read_text())
            native_output = raw["native_attempt"]["output"]
            operand = correctness_data["check"][
                "reference" if attempt["role"] == "baseline" else "candidate"
            ]
            aggregation = {
                "schema_version": 1,
                "kind": "piq_rank_aggregation_evidence",
                "authority": "retained_producer_assertion",
                "status": "pass",
                "run_id": run_id,
                "attempt_id": attempt_id,
                "rank_outputs": [
                    reference(bundle / rank["output"]["artifactName"])
                    for rank in sorted(observed["rank_census"], key=lambda item: item["rank"])
                ],
                "aggregate_output": reference(bundle / native_output["artifactName"]),
                "correctness_operand": reference(bundle / operand["path"]),
            }
            aggregation_path = package / f"evidence/aggregation-{attempt_id}.json"
            aggregation_path.write_text(json.dumps(aggregation, indent=2) + "\n")
            aggregation_refs.append(reference(aggregation_path))

        sidecars = {}
        for name in ("readback", "export", "cleanup"):
            path = package / f"evidence/{name}-receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": f"piq_functional_{name}_receipt",
                        "authority": "retained_producer_assertion",
                        "state": "COMPLETE",
                        "run_id": run_id,
                        "attempt_ids": attempt_ids,
                        "envelope_sha256": envelope_ref["sha256"],
                    },
                    indent=2,
                )
                + "\n"
            )
            sidecars[name] = reference(path)
        receipt = {
            "schema_version": 1,
            "kind": "piq_two_host_functional_evidence",
            "authority": "retained_producer_assertion",
            "qualification": "not_established",
            "envelope": envelope_ref,
            "run_id": run_id,
            "attempt_ids": attempt_ids,
            "aggregations": aggregation_refs,
            "execution": reference(bundle / execution["path"]),
            "correctness": reference(bundle / correctness["path"]),
            "custody": {
                "readback": sidecars["readback"],
                "export": sidecars["export"],
                "import_receipt": reference(bundle / "import-receipt.json"),
                "cleanup": sidecars["cleanup"],
            },
        }
        receipt_path = package / "evidence/two-host-functional-receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        manifest = {
            "schema_version": 2,
            "piq_two_host_functional_receipt": "evidence/two-host-functional-receipt.json",
        }
        return package, manifest, receipt

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

    def test_assigned_two_host_cpu_fixture_is_valid_diagnostic(self) -> None:
        self.assigned_cpu_hosts_fixture()
        self.assertEqual(validate_bundle(self.bundle), [])
        destination = self.root / "assigned-cpu-import"
        import_bundle(self.bundle, destination)
        self.assertEqual(validate_bundle(destination), [])

    def test_assigned_cpu_hosts_require_exact_distinct_host_contract(self) -> None:
        self.assigned_cpu_hosts_fixture()
        hosts = self.execution["resources"]["cpuHosts"]
        hosts[1]["targetId"] = hosts[0]["targetId"]
        hosts[1]["hostIdentitySha256"] = hosts[0]["hostIdentitySha256"]
        self.save_execution()
        self.problems("targetId values must be distinct")
        self.problems("identity hashes must be distinct")

    def test_assigned_cpu_rank_host_must_match_declared_target(self) -> None:
        self.assigned_cpu_hosts_fixture()
        self.execution["attempts"][0]["rank_census"][1]["host"] = "unassigned-host"
        self.save_execution()
        self.problems("rank host does not match an assigned CPU target")

    def test_two_host_functional_receipt_accepts_one_complete_attempt(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, _ = self.install_two_host_functional_receipt()
        self.assertEqual(
            validate_declared_two_host_functional(package, manifest),
            [READINESS_BLOCK],
        )

    def test_two_host_functional_receipt_rejects_declared_capacity_only(self) -> None:
        self.assigned_cpu_hosts_fixture()
        selected = self.execution["attempts"][1]["rank_census"]
        selected[1].update(host=selected[0]["host"], localRank=1)
        self.save_execution()
        package, manifest, _ = self.install_two_host_functional_receipt()
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("two distinct hosts" in error for error in errors), errors)

    def test_two_host_functional_receipt_never_unions_hosts_across_attempts(self) -> None:
        self.assigned_cpu_hosts_fixture()
        for index, attempt in enumerate(self.execution["attempts"]):
            census = attempt["rank_census"]
            census[0].update(host=f"synthetic-host-{index}", localRank=0)
            census[1].update(host=f"synthetic-host-{index}", localRank=1)
        self.save_execution()
        selected = [attempt["id"] for attempt in self.data["attempts"]]
        package, manifest, _ = self.install_two_host_functional_receipt(selected)
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertEqual(sum("two distinct hosts" in error for error in errors), 2, errors)

    def test_two_host_functional_malformed_attempt_lists_fail_without_crashing(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        receipt_path = package / "evidence/two-host-functional-receipt.json"
        envelope_path = package / receipt["envelope"]["path"]
        envelope = json.loads(envelope_path.read_text())

        envelope["attempts"] = None
        envelope_path.write_text(json.dumps(envelope, indent=2) + "\n")
        receipt["envelope"]["sha256"] = hashlib.sha256(
            envelope_path.read_bytes()
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("envelope attempts must be a list" in error for error in errors), errors)

        envelope = json.loads((self.bundle / "envelope.json").read_text())
        execution_ref = envelope["workload_evidence"]["execution"]
        execution_path = self.bundle / execution_ref["path"]
        execution = json.loads(execution_path.read_text())
        execution["attempts"] = None
        execution_path.write_text(json.dumps(execution, indent=2) + "\n")
        execution_sha = hashlib.sha256(execution_path.read_bytes()).hexdigest()
        execution_ref["sha256"] = execution_sha
        for artifact in envelope["artifacts"]:
            if artifact["path"] == execution_ref["path"]:
                artifact["sha256"] = execution_sha
        (self.bundle / "envelope.json").write_text(json.dumps(envelope, indent=2) + "\n")
        receipt["execution"]["sha256"] = execution_sha
        receipt["envelope"]["sha256"] = hashlib.sha256(
            (self.bundle / "envelope.json").read_bytes()
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("execution attempts must be a list" in error for error in errors), errors)

    def test_two_host_functional_receipt_rejects_unaccepted_attempt(self) -> None:
        self.assigned_cpu_hosts_fixture()
        self.data["attempts"][1]["accepted"] = False
        self.save()
        package, manifest, _ = self.install_two_host_functional_receipt()
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("succeeded and accepted" in error for error in errors), errors)

    def test_two_host_functional_receipt_binds_exact_run_and_artifacts(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        receipt["run_id"] = "unrelated-run"
        receipt["execution"]["sha256"] = "0" * 64
        (package / "evidence/two-host-functional-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("execution run_id must match" in error for error in errors), errors)
        self.assertTrue(any("execution.sha256 does not match" in error for error in errors), errors)

    def test_two_host_functional_receipt_binds_rank_aggregation_to_correctness(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        aggregation_ref = receipt["aggregations"][0]
        aggregation_path = package / aggregation_ref["path"]
        aggregation = json.loads(aggregation_path.read_text())
        aggregation["rank_outputs"][0] = aggregation["rank_outputs"][1]
        aggregation_path.write_text(json.dumps(aggregation, indent=2) + "\n")
        aggregation_ref["sha256"] = hashlib.sha256(aggregation_path.read_bytes()).hexdigest()
        (package / "evidence/two-host-functional-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("rank_outputs does not match" in error for error in errors), errors)

    def test_two_host_functional_custody_is_distinct_and_run_bound(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        receipt["custody"]["export"] = receipt["custody"]["readback"]
        readback_path = package / receipt["custody"]["readback"]["path"]
        readback = json.loads(readback_path.read_text())
        readback["run_id"] = "unrelated-run"
        readback_path.write_text(json.dumps(readback, indent=2) + "\n")
        digest = hashlib.sha256(readback_path.read_bytes()).hexdigest()
        receipt["custody"]["readback"]["sha256"] = digest
        receipt["custody"]["export"]["sha256"] = digest
        (package / "evidence/two-host-functional-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("roles must reference distinct" in error for error in errors), errors)
        self.assertTrue(any("readback does not match" in error for error in errors), errors)
        self.assertTrue(any("export does not match" in error for error in errors), errors)

    def test_two_host_functional_requires_exact_canonical_import_receipt(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        canonical = package / receipt["custody"]["import_receipt"]["path"]
        imported = json.loads(canonical.read_text())

        imported["unexpected"] = True
        canonical.write_text(json.dumps(imported, indent=2) + "\n")
        receipt["custody"]["import_receipt"]["sha256"] = hashlib.sha256(
            canonical.read_bytes()
        ).hexdigest()
        receipt_path = package / "evidence/two-host-functional-receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("exact canonical" in error for error in errors), errors)

        del imported["unexpected"]
        imported["imported_at"] = "not-a-UTC-timestamp"
        canonical.write_text(json.dumps(imported, indent=2) + "\n")
        receipt["custody"]["import_receipt"]["sha256"] = hashlib.sha256(
            canonical.read_bytes()
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("exact canonical" in error for error in errors), errors)

        forged = package / "evidence/forged-import-receipt.json"
        imported["imported_at"] = "2026-10-31T12:00:00Z"
        forged.write_text(json.dumps(imported, indent=2) + "\n")
        receipt["custody"]["import_receipt"] = {
            "path": forged.relative_to(package).as_posix(),
            "sha256": hashlib.sha256(forged.read_bytes()).hexdigest(),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("exact canonical" in error for error in errors), errors)

    def test_two_host_functional_receipt_rejects_symlinked_custody(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        readback = package / receipt["custody"]["readback"]["path"]
        replacement = package / "evidence/run-readback-copy.json"
        replacement.write_bytes(readback.read_bytes())
        readback.unlink()
        readback.symlink_to(replacement.name)
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("must not use symlinks" in error for error in errors), errors)

    def test_two_host_functional_manifest_opt_in_is_schema_two_only(self) -> None:
        package = self.root / "empty-package"
        package.mkdir()
        self.assertEqual(validate_declared_two_host_functional(package, {}), [])
        manifest = {
            "schema_version": 1,
            "piq_two_host_functional_receipt": "evidence/two-host-functional-receipt.json",
        }
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("schema 2" in error for error in errors), errors)
        self.assertTrue(any("manifest must be a JSON object" in error for error in validate_declared_two_host_functional(package, [])))

    def test_two_host_functional_cli_distinguishes_absent_declaration(self) -> None:
        package = self.root / "undeclared-package"
        package.mkdir()
        (package / "manifest.json").write_text('{"schema_version":2}\n')
        self.assertEqual(two_host_functional_main([str(package)]), 1)

    def test_two_host_functional_cli_help_is_read_only(self) -> None:
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    patch("tools.check_piq_two_host_functional._load_json") as load,
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    code = two_host_functional_main([flag])
                self.assertEqual(code, 0)
                load.assert_not_called()
                self.assertIn("usage:", stdout.getvalue())
                self.assertIn("retained two-host assertions", stdout.getvalue())
                self.assertIn("Does not independently establish", stdout.getvalue())
                self.assertEqual(stderr.getvalue(), "")

    def test_two_host_functional_cli_rejects_float_manifest_schema(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, _, _ = self.install_two_host_functional_receipt()
        (package / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2.0,
                    "piq_two_host_functional_receipt": (
                        "evidence/two-host-functional-receipt.json"
                    ),
                }
            )
            + "\n"
        )
        self.assertEqual(two_host_functional_main([str(package)]), 1)

    def test_two_host_functional_rejects_null_and_orphan_receipts(self) -> None:
        package = self.root / "partial-adoption-package"
        (package / "evidence").mkdir(parents=True)
        receipt = package / "evidence/two-host-functional-receipt.json"
        receipt.write_text("{}\n")

        null_manifest = {
            "schema_version": 2,
            "piq_two_host_functional_receipt": None,
        }
        errors = validate_declared_two_host_functional(package, null_manifest)
        self.assertTrue(any("must equal" in error for error in errors), errors)

        errors = validate_declared_two_host_functional(package, {"schema_version": 2})
        self.assertTrue(any("undeclared" in error for error in errors), errors)
        (package / "manifest.json").write_text('{"schema_version":2}\n')
        self.assertEqual(two_host_functional_main([str(package)]), 1)

    def test_two_host_functional_schema_rejects_dot_segments(self) -> None:
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas/piq-two-host-functional-evidence-v1.schema.json"
            ).read_text()
        )
        pattern = re.compile(
            schema["$defs"]["artifact"]["properties"]["path"]["pattern"]
        )
        for value in (
            "evidence/inside.json",
            "evidence/nested/inside.json",
            "evidence/.hidden.json",
            "evidence/../outside.json",
            "evidence/./inside.json",
            "evidence//inside.json",
            "evidence/nested/",
            "evidence/",
            "evidence\\inside.json",
        ):
            with self.subTest(value=value):
                runtime_accepts = _safe_relative(value) and value.startswith(
                    "evidence/"
                )
                self.assertEqual(pattern.fullmatch(value) is not None, runtime_accepts)

    def test_two_host_functional_refs_require_exact_on_disk_case(self) -> None:
        package = self.root / "case-package"
        evidence = package / "evidence"
        evidence.mkdir(parents=True)
        artifact = evidence / "rank-0.json"
        artifact.write_text("{}\n")
        errors: list[str] = []
        result = _validated_ref(
            package,
            {
                "path": "evidence/RANK-0.JSON",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            "rank output",
            errors,
        )
        self.assertIsNone(result)
        self.assertTrue(any("exact on-disk case" in error for error in errors), errors)

    def test_two_host_functional_unreadable_ref_fails_without_crashing(self) -> None:
        package = self.root / "unreadable-package"
        evidence = package / "evidence"
        evidence.mkdir(parents=True)
        artifact = evidence / "receipt.json"
        artifact.write_text("{}\n")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        errors: list[str] = []
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            result = _validated_ref(
                package,
                {
                    "path": "evidence/receipt.json",
                    "sha256": digest,
                },
                "receipt",
                errors,
            )
        self.assertIsNone(result)
        self.assertTrue(any("cannot be inspected" in error for error in errors), errors)

    def test_two_host_functional_runtime_rejects_raw_dot_segments(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, receipt = self.install_two_host_functional_receipt()
        receipt["envelope"]["path"] = receipt["envelope"]["path"].replace(
            "evidence/", "evidence/./", 1
        )
        (package / "evidence/two-host-functional-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(any("safe package-relative" in error for error in errors), errors)

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

    def install_two_host_owner_authority_receipt(
        self, package: Path, manifest: dict, producer: dict
    ) -> tuple[dict, dict, Path]:
        """Install synthetic owner records. They never represent a real execution."""

        def reference(path: Path) -> dict[str, str]:
            return {
                "path": path.relative_to(package).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        execution_path = package / producer["execution"]["path"]
        execution = json.loads(execution_path.read_text())
        execution_by_attempt = {
            item["attempt_id"]: item for item in execution["attempts"]
        }
        envelope_path = package / producer["envelope"]["path"]
        envelope = json.loads(envelope_path.read_text())
        envelope_by_attempt = {item["id"]: item for item in envelope["attempts"]}

        def parse_time(value: str) -> datetime:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))

        def format_time(value: datetime) -> str:
            return (
                value.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )

        native_by_attempt = {}
        for attempt_id in producer["attempt_ids"]:
            raw_ref = envelope_by_attempt[attempt_id]["raw_result"]["path"]
            raw = json.loads((envelope_path.parent / raw_ref).read_text())
            native_by_attempt[attempt_id] = raw["native_attempt"]
        latest_native_end = max(
            parse_time(native["completedAt"]) for native in native_by_attempt.values()
        )
        authority = {
            "owner_id": "owner:synthetic-runtime-3025",
            "authority_id": "authority:synthetic-command-3025",
            "attested_at": format_time(latest_native_end + timedelta(milliseconds=30)),
        }
        owner_root = package / "evidence/owner"
        owner_root.mkdir(parents=True, exist_ok=True)

        def observation(
            relative: str, kind: str, attempt_id: str, payload: dict
        ) -> dict[str, str]:
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            document = {
                "schema_version": 1,
                "kind": kind,
                "owner_id": authority["owner_id"],
                "authority_id": authority["authority_id"],
                "run_id": producer["run_id"],
                "attempt_id": attempt_id,
                **payload,
            }
            path.write_text(json.dumps(document, indent=2) + "\n")
            return reference(path)

        attempts = []
        for attempt_id in producer["attempt_ids"]:
            observed = execution_by_attempt[attempt_id]
            native = native_by_attempt[attempt_id]
            native_start = parse_time(native["startedAt"])
            native_end = parse_time(native["completedAt"])
            launch_payload = {
                "launch_id": f"launch:{attempt_id}:synthetic",
                "controller_id": "controller:synthetic-shared-3025",
                "execution_session_id": f"session:{attempt_id}:synthetic",
                "started_at": native["startedAt"],
                "ended_at": native["completedAt"],
            }
            launch = {
                **launch_payload,
                "evidence": observation(
                    f"evidence/owner/{attempt_id}/launch.json",
                    "piq_two_host_owner_launch_observation",
                    attempt_id,
                    launch_payload,
                ),
            }
            ranks = []
            rank_starts = [
                format_time(native_start + timedelta(milliseconds=10)),
                format_time(native_start + timedelta(milliseconds=20)),
            ]
            rank_ends = [
                format_time(native_end - timedelta(milliseconds=20)),
                format_time(native_end - timedelta(milliseconds=10)),
            ]
            host_identity_by_target = {
                host["targetId"]: host["hostIdentitySha256"]
                for host in execution["resources"]["cpuHosts"]
            }
            for rank_number, producer_rank in enumerate(observed["rank_census"]):
                rank_payload = {
                    "rank": rank_number,
                    "producer_host": producer_rank["host"],
                    "physical_host_id": host_identity_by_target[producer_rank["host"]],
                    "boot_id": f"00000000-0000-4000-8000-a0000000000{rank_number}",
                    "execution_session_id": launch["execution_session_id"],
                    "pid": producer_rank["processId"],
                    "started_at": rank_starts[rank_number],
                    "ended_at": rank_ends[rank_number],
                    "launch_id": launch["launch_id"],
                    "controller_id": launch["controller_id"],
                }
                ranks.append(
                    {
                        **rank_payload,
                        "evidence": observation(
                            f"evidence/owner/{attempt_id}/rank-{rank_number}.json",
                            "piq_two_host_owner_rank_observation",
                            attempt_id,
                            rank_payload,
                        ),
                    }
                )
            payload_sha256 = hashlib.sha256(
                f"{attempt_id}:synthetic-owner-transport".encode()
            ).hexdigest()
            transport_payload = {
                "transport_id": f"transport:{attempt_id}:synthetic",
                "protocol": "tcp",
                "source_rank": 0,
                "destination_rank": 1,
                "source_pid": ranks[0]["pid"],
                "destination_pid": ranks[1]["pid"],
                "source_physical_host_id": ranks[0]["physical_host_id"],
                "destination_physical_host_id": ranks[1]["physical_host_id"],
                "execution_session_id": launch["execution_session_id"],
                "sent_payload_sha256": payload_sha256,
                "received_payload_sha256": payload_sha256,
                "payload_size_bytes": 32,
                "sent_at": format_time(native_start + timedelta(milliseconds=50)),
                "received_at": format_time(native_start + timedelta(milliseconds=60)),
            }
            transport = {
                **transport_payload,
                "evidence": observation(
                    f"evidence/owner/{attempt_id}/transport.json",
                    "piq_two_host_owner_transport_observation",
                    attempt_id,
                    transport_payload,
                ),
            }
            cleanup = []
            for rank_number, rank in enumerate(ranks):
                cleanup_payload = {
                    "rank": rank_number,
                    "physical_host_id": rank["physical_host_id"],
                    "boot_id": rank["boot_id"],
                    "execution_session_id": launch["execution_session_id"],
                    "pid": rank["pid"],
                    "launch_id": launch["launch_id"],
                    "controller_id": launch["controller_id"],
                    "transport_id": transport["transport_id"],
                    "observed_at": format_time(
                        native_end + timedelta(milliseconds=10 + 10 * rank_number)
                    ),
                    "pid_absent": True,
                    "transport_closed": True,
                    "session_terminal": True,
                }
                cleanup.append(
                    {
                        **cleanup_payload,
                        "evidence": observation(
                            f"evidence/owner/{attempt_id}/cleanup-{rank_number}.json",
                            "piq_two_host_owner_cleanup_observation",
                            attempt_id,
                            cleanup_payload,
                        ),
                    }
                )
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "launch": launch,
                    "ranks": ranks,
                    "transport": transport,
                    "cleanup": cleanup,
                }
            )

        producer_path = package / "evidence/two-host-functional-receipt.json"
        handback = {
            "schema_version": 1,
            "kind": "piq_two_host_execution_owner_handback",
            "scope": "bounded_two_host_functional_execution_and_custody",
            "qualification": "not_established",
            "producer_receipt_sha256": hashlib.sha256(
                producer_path.read_bytes()
            ).hexdigest(),
            "envelope_sha256": producer["envelope"]["sha256"],
            "run_id": producer["run_id"],
            "attempt_ids": producer["attempt_ids"],
            "authority": authority,
            "attempts": attempts,
        }
        handback_path = owner_root / "execution-owner-handback.json"
        handback_path.write_text(json.dumps(handback, indent=2) + "\n")
        receipt = assemble_owner_receipt_data(
            package,
            "evidence/two-host-functional-receipt.json",
            "evidence/owner/execution-owner-handback.json",
        )
        (package / OWNER_RECEIPT_PATH).write_text(json.dumps(receipt, indent=2) + "\n")
        manifest["piq_two_host_owner_authority_receipt"] = OWNER_RECEIPT_PATH
        return receipt, handback, handback_path

    def admitted_owner_tuple(self, owner: dict) -> dict:
        identity = owner_admission_tuple(owner)
        attested = datetime.fromisoformat(
            owner["authority"]["attested_at"].replace("Z", "+00:00")
        )
        return {
            "schema_version": 1,
            "kind": "piq_two_host_owner_admissions",
            "entries": [
                {
                    "admission_id": "admission:synthetic-review-3025",
                    "admitted_at": (attested + timedelta(milliseconds=10))
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    "reviewer_id": "reviewer:synthetic-independent-3025",
                    "source": {
                        "id": "source:synthetic-owner-callback-3025",
                        "sha256": "a" * 64,
                    },
                    **identity,
                }
            ],
        }

    def owner_facts(self, owner: dict) -> dict:
        attempts = json.loads(json.dumps(owner["attempts"]))

        def remove_evidence(value):
            if isinstance(value, dict):
                value.pop("evidence", None)
                for child in value.values():
                    remove_evidence(child)
            elif isinstance(value, list):
                for child in value:
                    remove_evidence(child)

        remove_evidence(attempts)
        return {
            "schema_version": 1,
            "kind": "piq_two_host_owner_facts",
            "scope": owner["scope"],
            "qualification": "not_established",
            "producer_receipt_sha256": owner["producer_receipt"]["sha256"],
            "envelope_sha256": owner["envelope"]["sha256"],
            "run_id": owner["run_id"],
            "attempt_ids": owner["attempt_ids"],
            "authority": owner["authority"],
            "attempts": attempts,
        }

    def test_owner_authority_promotes_only_the_functional_readiness_block(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        admissions = self.admitted_owner_tuple(owner)
        self.assertEqual(
            validate_declared_two_host_functional(
                package, manifest, admissions=admissions
            ),
            [],
        )
        self.assertEqual(
            validate_two_host_owner_authority_receipt(package, owner, producer), []
        )
        self.assertEqual(producer["authority"], "retained_producer_assertion")
        self.assertEqual(producer["qualification"], "not_established")
        self.assertEqual(owner["qualification"], "not_established")

    def test_owner_authority_requires_fixed_source_reviewed_admission(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        production = json.loads(ADMISSIONS_PATH.read_text())
        self.assertEqual(production["entries"], [])
        self.assertEqual(validate_admissions(production), [])

        errors = validate_declared_two_host_functional(package, manifest)
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(any("not admitted" in error for error in errors), errors)

        admission = self.admitted_owner_tuple(owner)
        admission["entries"][0]["owner_handback_sha256"] = "f" * 64
        errors = validate_declared_two_host_functional(
            package, manifest, admissions=admission
        )
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(any("not admitted" in error for error in errors), errors)

    def test_owner_admission_and_authority_identifiers_are_portable_and_casefold_distinct(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        for identifier in ("owner:bad\ncontrol", "éxecution-owner", "x" * 256):
            with self.subTest(identifier=identifier):
                candidate = json.loads(json.dumps(owner))
                candidate["authority"]["owner_id"] = identifier
                errors = validate_two_host_owner_authority_receipt(
                    package, candidate, producer
                )
                self.assertTrue(
                    any("specific non-generic identifier" in error for error in errors),
                    errors,
                )

        candidate = json.loads(json.dumps(owner))
        candidate["authority"]["owner_id"] = "Identity:Owner-3025"
        candidate["authority"]["authority_id"] = "identity:owner-3025"
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(any("casefold identities" in error for error in errors), errors)

        admission = self.admitted_owner_tuple(owner)
        entry = admission["entries"][0]
        entry["reviewer_id"] = entry["owner_id"].upper()
        errors = validate_admissions(admission)
        self.assertTrue(any("independent" in error for error in errors), errors)

        admission = self.admitted_owner_tuple(owner)
        duplicate = json.loads(json.dumps(admission["entries"][0]))
        duplicate["admission_id"] = duplicate["admission_id"].upper()
        duplicate["owner_handback_sha256"] = "b" * 64
        admission["entries"].append(duplicate)
        errors = validate_admissions(admission)
        self.assertTrue(
            any("admission IDs must be distinct" in error for error in errors), errors
        )

        malformed = self.admitted_owner_tuple(owner)
        malformed["entries"][0]["owner_id"] = []
        errors = validate_admissions(malformed)
        self.assertTrue(
            any("specific non-generic" in error for error in errors), errors
        )

    def test_owner_authority_never_suppresses_an_invalid_producer(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        self.install_two_host_owner_authority_receipt(package, manifest, producer)
        producer["qualification"] = "qualified"
        (package / "evidence/two-host-functional-receipt.json").write_text(
            json.dumps(producer, indent=2) + "\n"
        )
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(
            any("cannot claim qualification" in error for error in errors), errors
        )

    def test_owner_authority_binds_exact_producer_envelope_run_and_attempts(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        mutations = (
            (
                "producer",
                lambda value: value["producer_receipt"].update(sha256="0" * 64),
                "sha256 does not match",
            ),
            (
                "envelope",
                lambda value: value["envelope"].update(sha256="0" * 64),
                "exactly match the producer",
            ),
            (
                "run",
                lambda value: value.update(run_id="unrelated-run"),
                "run_id must exactly match",
            ),
            (
                "attempts",
                lambda value: value.update(attempt_ids=["unrelated-attempt"]),
                "attempt_ids must exactly match",
            ),
        )
        for name, mutate, phrase in mutations:
            with self.subTest(name=name):
                candidate = json.loads(json.dumps(owner))
                mutate(candidate)
                errors = validate_two_host_owner_authority_receipt(
                    package, candidate, producer
                )
                self.assertTrue(any(phrase in error for error in errors), errors)

    def test_owner_authority_rejects_generic_authority_and_host_labels(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        candidate["authority"]["owner_id"] = "owner"
        candidate["authority"]["authority_id"] = "authority"
        candidate["attempts"][0]["ranks"][0]["physical_host_id"] = "host"
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertGreaterEqual(
            sum("specific non-generic identifier" in error for error in errors),
            2,
            errors,
        )
        self.assertTrue(
            any(
                "canonical lowercase host identity SHA-256" in error for error in errors
            ),
            errors,
        )

    def test_owner_authority_requires_exact_rank_identity_and_ordered_windows(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        ranks = candidate["attempts"][0]["ranks"]
        ranks[0]["pid"] = 999
        ranks[0]["ended_at"] = ranks[0]["started_at"]
        ranks[1]["physical_host_id"] = ranks[0]["physical_host_id"]
        ranks[1]["boot_id"] = ranks[0]["boot_id"]
        ranks[1]["launch_id"] = "launch:unrelated"
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        for phrase in (
            "pid must match the producer census",
            "inside the objective, native attempt, and launch",
            "two distinct owner-observed physical hosts",
            "distinct boot identities",
            "launch/controller identity must match",
        ):
            self.assertTrue(any(phrase in error for error in errors), (phrase, errors))

    def test_owner_rank_identity_uses_declared_physical_hash_and_canonical_boot_uuid(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        rank = candidate["attempts"][0]["ranks"][0]
        rank["physical_host_id"] = rank["producer_host"]
        rank["boot_id"] = rank["boot_id"].upper()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any(
                "canonical lowercase host identity SHA-256" in error for error in errors
            ),
            errors,
        )
        self.assertTrue(
            any("must match execution.resources.cpuHosts" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("canonical lowercase UUID" in error for error in errors), errors
        )

    def test_owner_times_bind_exact_native_and_objective_windows(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        attempt = candidate["attempts"][0]
        attempt["launch"]["started_at"] = "2026-10-27T11:59:59Z"
        attempt["ranks"][0]["started_at"] = "2026-10-27T12:00:00.010Z"
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("exactly match the producer native" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("inside the envelope objective" in error for error in errors), errors
        )
        self.assertTrue(
            any(
                "inside the objective, native attempt, and launch" in error
                for error in errors
            ),
            errors,
        )

    def test_owner_transport_is_same_attempt_inter_host_evidence(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        transport = candidate["attempts"][0]["transport"]
        transport.update(
            protocol="local",
            destination_rank=0,
            destination_pid=True,
            destination_physical_host_id="physical:unrelated-host",
            received_payload_sha256="f" * 64,
            payload_size_bytes=True,
            received_at="2026-10-31T12:00:20Z",
        )
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        for phrase in (
            "transport.protocol must be one of",
            "connect type-strict ranks 0 and 1",
            "destination_pid must be a positive type-strict integer",
            "same-attempt ranks",
            "matching sent/received payload SHA-256",
            "ordered inter-host send/receive inside both rank windows",
        ):
            self.assertTrue(any(phrase in error for error in errors), (phrase, errors))

    def test_owner_cleanup_is_terminal_on_both_same_attempt_hosts(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        cleanup = candidate["attempts"][0]["cleanup"]
        cleanup[0]["observed_at"] = candidate["attempts"][0]["launch"]["ended_at"]
        cleanup[0]["pid_absent"] = False
        cleanup[1]["transport_closed"] = False
        cleanup[1]["session_terminal"] = False
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        for phrase in (
            "after launch, both ranks, and transport completion",
            "pid_absent must be true",
            "transport_closed must be true",
            "session_terminal must be true",
        ):
            self.assertTrue(any(phrase in error for error in errors), (phrase, errors))

        candidate = json.loads(json.dumps(owner))
        candidate["attempts"][0]["cleanup"].pop()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("exactly ranks 0 and 1" in error for error in errors), errors
        )

    def test_owner_attempts_never_union_rank_transport_or_cleanup_evidence(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        selected = [attempt["id"] for attempt in self.data["attempts"]]
        package, manifest, producer = self.install_two_host_functional_receipt(selected)
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        self.assertEqual(
            validate_declared_two_host_functional(
                package, manifest, admissions=self.admitted_owner_tuple(owner)
            ),
            [],
        )
        candidate = json.loads(json.dumps(owner))
        first, second = candidate["attempts"]
        first["ranks"][1] = second["ranks"][1]
        first["cleanup"][1] = second["cleanup"][1]
        first["transport"] = second["transport"]
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("execution_session_id must match" in error for error in errors), errors
        )
        self.assertTrue(any("same-attempt" in error for error in errors), errors)

    def test_owner_attempts_reject_reused_launch_session_transport_and_event(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        selected = [attempt["id"] for attempt in self.data["attempts"]]
        package, manifest, producer = self.install_two_host_functional_receipt(selected)
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        first, second = candidate["attempts"]
        second["launch"]["launch_id"] = first["launch"]["launch_id"].upper()
        second["launch"]["execution_session_id"] = first["launch"][
            "execution_session_id"
        ].upper()
        second["transport"] = json.loads(json.dumps(first["transport"]))
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        for phrase in (
            "must not reuse launch_id",
            "must not reuse execution_session_id",
            "must not reuse transport_id",
            "must not reuse transport event identity/timing/payload fingerprint",
        ):
            self.assertTrue(any(phrase in error for error in errors), (phrase, errors))

    def test_owner_attempts_reject_same_transport_event_under_new_labels(self) -> None:
        self.assigned_cpu_hosts_fixture()
        selected = [attempt["id"] for attempt in self.data["attempts"]]
        package, manifest, producer = self.install_two_host_functional_receipt(selected)
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        first, second = candidate["attempts"]
        relabeled = second["transport"]
        for field in (
            "protocol",
            "source_rank",
            "destination_rank",
            "source_pid",
            "destination_pid",
            "source_physical_host_id",
            "destination_physical_host_id",
            "sent_payload_sha256",
            "received_payload_sha256",
            "payload_size_bytes",
            "sent_at",
            "received_at",
        ):
            relabeled[field] = first["transport"][field]
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any(
                "must not reuse transport event identity/timing/payload fingerprint"
                in error
                for error in errors
            ),
            errors,
        )

    def test_owner_attempts_normalize_timestamp_aliases_in_event_identity(self) -> None:
        self.assigned_cpu_hosts_fixture()
        selected = [attempt["id"] for attempt in self.data["attempts"]]
        package, manifest, producer = self.install_two_host_functional_receipt(selected)
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        first, second = candidate["attempts"]
        relabeled = second["transport"]
        for field in (
            "protocol",
            "source_rank",
            "destination_rank",
            "source_pid",
            "destination_pid",
            "source_physical_host_id",
            "destination_physical_host_id",
            "sent_payload_sha256",
            "received_payload_sha256",
            "payload_size_bytes",
        ):
            relabeled[field] = first["transport"][field]
        def shorter_fraction(value: str) -> str:
            whole, fraction = value.removesuffix("Z").rsplit(".", 1)
            self.assertTrue(fraction.endswith("0"))
            return f"{whole}.{fraction.rstrip('0')}Z"

        relabeled["sent_at"] = shorter_fraction(first["transport"]["sent_at"])
        relabeled["received_at"] = shorter_fraction(
            first["transport"]["received_at"]
        )
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any(
                "must not reuse transport event identity/timing/payload fingerprint"
                in error
                for error in errors
            ),
            errors,
        )

    def test_owner_receipt_malformed_types_fail_without_crashing(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidates = []
        for mutation in (
            lambda value: value.update(schema_version=True),
            lambda value: value.update(authority=[]),
            lambda value: value.update(attempt_ids={}),
            lambda value: value.update(attempts={}),
            lambda value: value["attempts"][0].update(launch=[]),
            lambda value: value["attempts"][0].update(ranks={}),
            lambda value: value["attempts"][0].update(transport=[]),
            lambda value: value["attempts"][0].update(cleanup={}),
            lambda value: value["attempts"][0]["ranks"][0].update(rank=True, pid=True),
            lambda value: value["attempts"][0]["ranks"][0].update(
                physical_host_id={}, boot_id=[]
            ),
            lambda value: value["attempts"][0]["ranks"][0].update(
                producer_host={}
            ),
            lambda value: value["attempts"][0]["ranks"][1].update(
                producer_host=[]
            ),
            lambda value: value["attempts"][0]["transport"].update(
                source_rank=[], destination_rank={}, payload_size_bytes=True
            ),
            lambda value: value["attempts"][0]["transport"].update(protocol={}),
            lambda value: value["attempts"][0]["transport"].update(protocol=[]),
        ):
            candidate = json.loads(json.dumps(owner))
            mutation(candidate)
            candidates.append(candidate)
        for index, candidate in enumerate(candidates):
            with self.subTest(index=index):
                errors = validate_two_host_owner_authority_receipt(
                    package, candidate, producer
                )
                self.assertTrue(errors)

    def test_owner_receipt_requires_exact_fields_and_exact_observation_projection(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        candidate["unexpected"] = True
        candidate["attempts"][0]["launch"]["unexpected"] = True
        candidate["attempts"][0]["transport"]["unexpected"] = True
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertGreaterEqual(
            sum("fields do not match schema version 1" in error for error in errors),
            3,
            errors,
        )

        candidate = json.loads(json.dumps(owner))
        evidence_ref = candidate["attempts"][0]["transport"]["evidence"]
        evidence_path = package / evidence_ref["path"]
        observation = json.loads(evidence_path.read_text())
        observation["unexpected"] = True
        evidence_path.write_text(json.dumps(observation, indent=2) + "\n")
        evidence_ref["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("exact owner projection" in error for error in errors), errors
        )

        candidate = json.loads(json.dumps(owner))
        handback_ref = candidate["owner_handback"]
        handback_path = package / handback_ref["path"]
        handback = json.loads(handback_path.read_text())
        handback["unexpected"] = True
        handback_path.write_text(json.dumps(handback, indent=2) + "\n")
        handback_ref["sha256"] = hashlib.sha256(handback_path.read_bytes()).hexdigest()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any(
                "exact producer/envelope/run/attempt projection" in error
                for error in errors
            ),
            errors,
        )

    def test_owner_receipt_and_observations_reject_duplicate_json_keys(self) -> None:
        nested = self.root / "duplicate-nested.json"
        nested.write_text('{"outer":{"id":"one","id":"two"}}\n')
        load_errors: list[str] = []
        self.assertIsNone(_load_json(nested, "nested", load_errors))
        self.assertTrue(any("duplicate JSON key: id" in error for error in load_errors))

        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        fixed = package / OWNER_RECEIPT_PATH
        fixed.write_text(json.dumps(owner)[:-1] + ',"kind":"duplicate"}\n')
        errors = validate_declared_two_host_functional(
            package, manifest, admissions=self.admitted_owner_tuple(owner)
        )
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(
            any("duplicate JSON key: kind" in error for error in errors), errors
        )

        fixed.write_text(json.dumps(owner, indent=2) + "\n")
        candidate = json.loads(json.dumps(owner))
        evidence_ref = candidate["attempts"][0]["launch"]["evidence"]
        evidence_path = package / evidence_ref["path"]
        document = json.loads(evidence_path.read_text())
        evidence_path.write_text(
            json.dumps(document)[:-1]
            + f',"attempt_id":{json.dumps(document["attempt_id"])}}}\n'
        )
        evidence_ref["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("duplicate JSON key: attempt_id" in error for error in errors), errors
        )

    def test_owner_references_reject_raw_dot_wrong_case_and_symlink_paths(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        candidate["owner_handback"]["path"] = candidate["owner_handback"][
            "path"
        ].replace("evidence/", "evidence/./", 1)
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("safe package-relative" in error for error in errors), errors
        )

        candidate = json.loads(json.dumps(owner))
        rank_ref = candidate["attempts"][0]["ranks"][0]["evidence"]
        rank_ref["path"] = rank_ref["path"].replace("rank-0.json", "RANK-0.JSON")
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(any("exact on-disk case" in error for error in errors), errors)

        candidate = json.loads(json.dumps(owner))
        cleanup_ref = candidate["attempts"][0]["cleanup"][0]["evidence"]
        cleanup_path = package / cleanup_ref["path"]
        replacement = cleanup_path.with_name("cleanup-real.json")
        cleanup_path.rename(replacement)
        cleanup_path.symlink_to(replacement.name)
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("must not use symlinks" in error for error in errors), errors
        )

    def test_owner_observations_must_be_distinct_physical_files(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        first_ref = candidate["attempts"][0]["ranks"][0]["evidence"]
        second_ref = candidate["attempts"][0]["ranks"][1]["evidence"]
        first_path = package / first_ref["path"]
        second_path = package / second_ref["path"]
        second_path.unlink()
        os.link(first_path, second_path)
        second_ref["sha256"] = hashlib.sha256(second_path.read_bytes()).hexdigest()
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("exactly one filesystem link" in error for error in errors), errors
        )
        self.assertTrue(any("physically distinct" in error for error in errors), errors)

    def test_fixed_owner_receipt_rejects_symlinks_and_noncanonical_time(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        candidate = json.loads(json.dumps(owner))
        candidate["authority"]["attested_at"] = "2026-10-31T12:00:13+00:00"
        errors = validate_two_host_owner_authority_receipt(package, candidate, producer)
        self.assertTrue(
            any("canonical UTC timestamp" in error for error in errors), errors
        )

        fixed = package / OWNER_RECEIPT_PATH
        real = fixed.with_name("owner-authority-real.json")
        fixed.rename(real)
        fixed.symlink_to(real.name)
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(
            any("must not use symlinks" in error for error in errors), errors
        )

    def test_owner_receipt_manifest_is_fixed_optional_and_fail_closed(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        self.install_two_host_owner_authority_receipt(package, manifest, producer)
        manifest.pop("piq_two_host_owner_authority_receipt")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(
            any("undeclared PIQ two-host owner receipt" in error for error in errors),
            errors,
        )

        manifest["piq_two_host_owner_authority_receipt"] = None
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertIn(READINESS_BLOCK, errors)
        self.assertTrue(any("must equal" in error for error in errors), errors)

        manifest["piq_two_host_owner_authority_receipt"] = OWNER_RECEIPT_PATH
        manifest.pop("piq_two_host_functional_receipt")
        errors = validate_declared_two_host_functional(package, manifest)
        self.assertTrue(
            any("requires the exact declared producer" in error for error in errors),
            errors,
        )

    def test_owner_facts_builder_writes_six_observations_and_handback_atomically(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        facts = self.owner_facts(owner)
        facts_path = package / "evidence/owner-facts.json"
        facts_path.write_text(json.dumps(facts, indent=2) + "\n")
        handback = build_owner_handback(
            package,
            producer_relative="evidence/two-host-functional-receipt.json",
            facts_relative="evidence/owner-facts.json",
            output_directory="evidence/generated-owner",
            handback_relative="evidence/generated-owner/execution-owner-handback.json",
        )
        files = sorted(
            path.relative_to(package / "evidence/generated-owner").as_posix()
            for path in (package / "evidence/generated-owner").rglob("*")
            if path.is_file()
        )
        self.assertEqual(
            files,
            [
                "attempt-000/cleanup-0.json",
                "attempt-000/cleanup-1.json",
                "attempt-000/launch.json",
                "attempt-000/rank-0.json",
                "attempt-000/rank-1.json",
                "attempt-000/transport.json",
                "execution-owner-handback.json",
            ],
        )
        assembled = assemble_owner_receipt_data(
            package,
            "evidence/two-host-functional-receipt.json",
            handback.relative_to(package.resolve()).as_posix(),
        )
        self.assertEqual(
            validate_two_host_owner_authority_receipt(package, assembled, producer), []
        )

        marker = package / "evidence/generated-owner/owner-marker.txt"
        marker.write_text("preserve\n")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            build_owner_handback(
                package,
                producer_relative="evidence/two-host-functional-receipt.json",
                facts_relative="evidence/owner-facts.json",
                output_directory="evidence/generated-owner",
                handback_relative="evidence/generated-owner/execution-owner-handback.json",
            )
        self.assertEqual(marker.read_text(), "preserve\n")

    def test_owner_facts_builder_cleans_partial_stage_on_failure(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        facts_path = package / "evidence/owner-facts.json"
        facts_path.write_text(json.dumps(self.owner_facts(owner), indent=2) + "\n")
        from tools import build_piq_two_host_owner_handback as builder

        real_create = builder._atomic_create_json
        calls = 0

        def fail_third(path, value):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("synthetic staged write failure")
            return real_create(path, value)

        with patch.object(builder, "_atomic_create_json", side_effect=fail_third):
            with self.assertRaisesRegex(OSError, "synthetic staged write failure"):
                build_owner_handback(
                    package,
                    producer_relative="evidence/two-host-functional-receipt.json",
                    facts_relative="evidence/owner-facts.json",
                    output_directory="evidence/generated-owner-failure",
                    handback_relative="evidence/generated-owner-failure/execution-owner-handback.json",
                )
        self.assertFalse((package / "evidence/generated-owner-failure").exists())
        self.assertEqual(
            list((package / "evidence").glob(".two-host-owner-stage-*")), []
        )

    def test_owner_facts_builder_does_not_replace_concurrent_owner_directory(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        facts_path = package / "evidence/owner-facts.json"
        facts_path.write_text(json.dumps(self.owner_facts(owner), indent=2) + "\n")
        from tools import build_piq_two_host_owner_handback as builder

        real_rename = builder._rename_noreplace
        raced_inode = None

        def race(source, destination):
            nonlocal raced_inode
            destination.mkdir()
            raced_inode = destination.stat().st_ino
            real_rename(source, destination)

        with patch.object(builder, "_rename_noreplace", side_effect=race):
            with self.assertRaises(FileExistsError):
                build_owner_handback(
                    package,
                    producer_relative="evidence/two-host-functional-receipt.json",
                    facts_relative="evidence/owner-facts.json",
                    output_directory="evidence/concurrent-owner",
                    handback_relative="evidence/concurrent-owner/execution-owner-handback.json",
                )
        destination = package / "evidence/concurrent-owner"
        self.assertEqual(destination.stat().st_ino, raced_inode)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(
            list((package / "evidence").glob(".two-host-owner-stage-*")), []
        )

    def test_owner_facts_builder_preserves_published_output_on_late_failure(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        facts_path = package / "evidence/owner-facts.json"
        facts_path.write_text(json.dumps(self.owner_facts(owner), indent=2) + "\n")
        from tools import build_piq_two_host_owner_handback as builder

        output = package / "evidence/published-owner-late-failure"
        marker = output / "concurrent-owner-marker.txt"

        def fail_after_publish(*_args, **_kwargs):
            marker.write_text("preserve concurrent owner work\n")
            raise OSError("synthetic post-publish verification failure")

        with patch.object(
            builder, "assemble_owner_receipt_data", side_effect=fail_after_publish
        ):
            with self.assertRaisesRegex(
                RuntimeError, "published.*preserved for explicit inspection"
            ):
                build_owner_handback(
                    package,
                    producer_relative="evidence/two-host-functional-receipt.json",
                    facts_relative="evidence/owner-facts.json",
                    output_directory="evidence/published-owner-late-failure",
                    handback_relative=(
                        "evidence/published-owner-late-failure/"
                        "execution-owner-handback.json"
                    ),
                )
        self.assertTrue(output.is_dir())
        self.assertEqual(marker.read_text(), "preserve concurrent owner work\n")
        self.assertTrue((output / "execution-owner-handback.json").is_file())
        self.assertEqual(
            list((package / "evidence").glob(".two-host-owner-stage-*")), []
        )

    def test_owner_facts_schema_template_and_example_are_valid_json(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "schemas/piq-two-host-owner-admissions-v1.schema.json",
            "schemas/piq-two-host-owner-authority-v1.schema.json",
            "schemas/piq-two-host-owner-facts-v1.schema.json",
            "templates/two-host-execution-owner-handback.json",
            "templates/two-host-owner-authority-receipt.json",
            "templates/two-host-owner-facts.json",
            "examples/piq-two-host-owner-facts/owner-facts.json",
            "trust/piq-two-host-owner-admissions.json",
        ):
            with self.subTest(relative=relative):
                self.assertIsInstance(json.loads((root / relative).read_text()), dict)

    def test_owner_identifier_schema_patterns_are_bounded_ascii_and_control_free(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "schemas/piq-two-host-owner-authority-v1.schema.json",
            "schemas/piq-two-host-owner-admissions-v1.schema.json",
            "schemas/piq-two-host-owner-facts-v1.schema.json",
        ):
            schema = json.loads((root / relative).read_text())
            specific = schema["$defs"]["identifier"]
            portable = schema["$defs"]["portableIdentifier"]
            self.assertEqual(specific["minLength"], 8)
            self.assertEqual(specific["maxLength"], 255)
            self.assertEqual(portable["minLength"], 1)
            self.assertEqual(portable["maxLength"], 255)
            specific_pattern = re.compile(specific["pattern"])
            portable_pattern = re.compile(portable["pattern"])
            self.assertIsNotNone(specific_pattern.fullmatch("owner:runtime-3025"))
            self.assertIsNotNone(portable_pattern.fullmatch("run-1"))
            for invalid in (" bad-id", "bad id", "bad\nid", "éxecution", "x" * 256):
                self.assertIsNone(specific_pattern.fullmatch(invalid))
                self.assertIsNone(portable_pattern.fullmatch(invalid))

    def test_owner_receipt_assembler_is_explicit_atomic_and_no_overwrite(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, handback_path = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        output = package / OWNER_RECEIPT_PATH
        output.unlink()
        assembled = assemble_owner_receipt_data(
            package,
            "evidence/two-host-functional-receipt.json",
            handback_path.relative_to(package).as_posix(),
        )
        self.assertEqual(assembled, owner)
        _atomic_create_json(output, assembled)
        original = output.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            _atomic_create_json(output, {"changed": True})
        self.assertEqual(output.read_bytes(), original)

        with self.assertRaisesRegex(ValueError, "fixed path"):
            assemble_owner_receipt_data(
                package,
                "evidence/./two-host-functional-receipt.json",
                handback_path.relative_to(package).as_posix(),
            )

        output.unlink()
        handback = json.loads(handback_path.read_text())
        handback["run_id"] = "unrelated-run"
        handback_path.write_text(json.dumps(handback, indent=2) + "\n")
        with self.assertRaisesRegex(ValueError, "invalid owner receipt inputs"):
            assemble_owner_receipt_data(
                package,
                "evidence/two-host-functional-receipt.json",
                handback_path.relative_to(package).as_posix(),
            )
        self.assertFalse(output.exists())

    def test_owner_receipt_schema_rejects_raw_dot_segments(self) -> None:
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas/piq-two-host-owner-authority-v1.schema.json"
            ).read_text()
        )
        pattern = re.compile(
            schema["$defs"]["artifact"]["properties"]["path"]["pattern"]
        )
        for value in (
            "evidence/inside.json",
            "evidence/nested/inside.json",
            "evidence/.hidden.json",
            "evidence/../outside.json",
            "evidence/./inside.json",
            "evidence//inside.json",
            "evidence/nested/",
            "evidence/",
            "evidence\\inside.json",
        ):
            with self.subTest(value=value):
                runtime_accepts = _safe_relative(value) and value.startswith(
                    "evidence/"
                )
                self.assertEqual(pattern.fullmatch(value) is not None, runtime_accepts)

    def test_two_host_functional_cli_keeps_producer_consistency_exit_zero_but_rejects_unadmitted_owner(
        self,
    ) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        (package / "manifest.json").write_text(json.dumps(manifest) + "\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = two_host_functional_main([str(package)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("consistency only", stdout.getvalue())
        self.assertIn("readiness remains blocked", stdout.getvalue())

        self.install_two_host_owner_authority_receipt(package, manifest, producer)
        (package / "manifest.json").write_text(json.dumps(manifest) + "\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = two_host_functional_main([str(package)])
        self.assertEqual(code, 1)
        self.assertNotIn("admitted owner-attested", stdout.getvalue())
        self.assertIn("not admitted", stderr.getvalue())

    def test_admitted_owner_cli_reports_only_the_retained_assertion_boundary(self) -> None:
        self.assigned_cpu_hosts_fixture()
        package, manifest, producer = self.install_two_host_functional_receipt()
        owner, _, _ = self.install_two_host_owner_authority_receipt(
            package, manifest, producer
        )
        admissions = self.admitted_owner_tuple(owner)
        (package / "manifest.json").write_text(json.dumps(manifest) + "\n")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "tools.check_piq_two_host_owner_authority.load_fixed_admissions",
                return_value=(admissions, []),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = two_host_functional_main([str(package)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn(
            "retained owner-attested assertions and reviewed admission tuple only",
            stdout.getvalue(),
        )
        self.assertIn(
            "Does not independently establish execution, physical hosts or traffic",
            stdout.getvalue(),
        )
        self.assertIn("Scientific qualification is not established", stdout.getvalue())
        self.assertNotIn("bounded functional execution/custody only", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
