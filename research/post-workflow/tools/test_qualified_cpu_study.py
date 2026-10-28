"""Small synthetic contract fixtures. None of these records describe execution."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .qualified_cpu_study import (
    CGROUP_CONTEXT, CPU_COVERAGE, CPU_PROCESS_METRICS, HOST_CONTEXT, CRITERIA, TIMED_STEPS,
    cpu_receipt_binds_envelope, evaluate_cpu_study, validate_cpu_telemetry,
    file_sha256, validate_cpu_receipt,
)


def write_json(package: Path, relative: str, data: dict) -> dict:
    path = package / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": relative, "sha256": file_sha256(path)}


def write_bytes(package: Path, relative: str, data: bytes) -> dict:
    path = package / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": file_sha256(path)}


def synthetic_telemetry(bundle: Path, attempt_id: str, parent: int, children: list[int], bundle_sha: str, timestamp: str) -> tuple[dict, list[dict]]:
    """Invented TEST-ONLY records; no host, PID, cgroup or counter was observed."""
    raw = bytearray()
    entries, samples, identities, terminals = [], [], [], []
    clock = 1
    def tick() -> int:
        nonlocal clock
        clock += 1
        return clock
    def native_sha(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()
    def retain(source_id: str, payload: bytes, **extra) -> dict:
        entry = {"sequence": len(entries), "source_id": source_id, "observed_at": timestamp, "monotonic_ns": tick(),
                 "offset": len(raw), "size_bytes": len(payload), "sha256": native_sha(payload),
                 "source_size_bytes": len(payload), "source_sha256": native_sha(payload), "truncated": False,
                 "synthetic_fixture": True, **extra}
        raw.extend(payload)
        entries.append(entry)
        return entry
    def stat(pid: int) -> bytes:
        fields = ["0"] * 40
        fields[0], fields[1], fields[17], fields[19] = "S", str(parent if pid != parent else 1), "1", str(100000 + pid)
        return (f"{pid} (SYNTHETIC UNEXECUTED FIXTURE) " + " ".join(fields) + "\n").encode()
    boot = b"SYNTHETIC-BOOT-ID-NO-EXECUTION\n"
    mount = "1 0 1:2 / /synthetic rw - tmpfs synthetic rw"
    retain("host.boot_id", boot)
    retain("host.mountinfo.binding", (mount + "\n").encode())
    membership = ("\n".join(str(pid) for pid in [parent, *children]) + "\n").encode()
    def register(pid: int) -> dict:
        identity = {"pid": pid, "starttime_ticks": 100000 + pid, "cgroup_path": "/synthetic-fixture",
                    "cgroup_directory_identity": [1, 2], "kind": "adapter" if pid == parent else "codec-child",
                    "host_boot_id_sha256": native_sha(boot.strip())}
        identities.append(identity)
        retain("process.identity.cgroup-membership", membership, pid=pid)
        retain("process.identity.stat", stat(pid), pid=pid)
        retain("process.identity.cgroup", b"0::/synthetic-fixture\n", pid=pid)
        return identity
    def sample(identity: dict, boundary: str) -> dict:
        pid = identity["pid"]
        moment = tick()
        pressure = b"some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        files = {"process.stat": stat(pid), "process.status": b"Name:\tSYNTHETIC\nVmRSS:\t1 kB\nvoluntary_ctxt_switches:\t0\nnonvoluntary_ctxt_switches:\t0\n",
                 "process.io": b"read_bytes: 0\nwrite_bytes: 0\ncancelled_write_bytes: 0\n", "process.schedstat": b"1 0 1\n",
                 "cgroup.cpu.pressure": pressure, "cgroup.memory.pressure": pressure, "cgroup.io.pressure": pressure}
        frame = bytearray(b"PIQ-NATIVE-FILES-V1\n")
        for name, data in files.items():
            metadata = json.dumps({"sourceId": name, "sizeBytes": len(data), "sha256": native_sha(data)}, sort_keys=True, separators=(",", ":")).encode()
            frame.extend(len(metadata).to_bytes(4, "big") + metadata + len(data).to_bytes(8, "big") + data)
        ref = retain("host.process-and-pressure", bytes(frame), pid=pid, media_type="application/vnd.piq.procfs-file-bundle.v1")
        metrics = [{"name": "host.process." + name, "value": 1 if name in ("thread_count", "memory.rss_kibibytes", "scheduling.runtime_nanoseconds", "scheduling.timeslices") else 0, "unit": unit, "cumulative": name not in ("thread_count", "cpu.processor", "memory.rss_kibibytes"),
                    "labels": {"process_id": str(pid), "cgroup_id": "cg-synthetic"}} for name, unit in CPU_PROCESS_METRICS.items()]
        row = {"sequence": len(samples), "boundary": boundary, "observed_at": timestamp, "monotonic_ns": moment,
               "process_identity": identity, "status": "sample", "raw": ref, "measurements": metrics}
        samples.append(row)
        return row
    def context(boundary: str) -> None:
        for name in sorted(HOST_CONTEXT):
            retain(name, b"SYNTHETIC UNEXECUTED HOST CONTEXT\n", boundary=boundary)
        for name in sorted(CGROUP_CONTEXT):
            # Empty io.max is real-format valid; all records here remain invented.
            retain(name, b"" if name == "cgroup.io.max" else b"SYNTHETIC UNEXECUTED CGROUP CONTEXT\n", boundary=boundary, pid=parent)
    adapter = register(parent)
    sample(adapter, "start")
    context("start")
    for pid in children:
        identity = register(pid)
        captured = sample(identity, "child-observed-before-input")
        moment = tick()
        terminals.append({"sequence": len(terminals), "pid": pid, "process_identity": identity,
                          "observed_at": timestamp, "monotonic_ns": moment, "exit_code": 0, "source": "owning-Popen-returncode",
                          "final_capture_expected": False, "last_successful_sample_monotonic_ns": captured["monotonic_ns"],
                          "unobserved_terminal_interval_nanoseconds": [captured["monotonic_ns"], moment]})
    sample(adapter, "stop")
    context("stop")
    summary = {"schema": "piq.finite-cpu-telemetry.v1", "synthetic_fixture": True,
               "source": {"kind": "host-procfs", "producerId": "finite-cpu-observer", "producerSha256": "sha256:" + bundle_sha, "rawSha256": None, "rawSizeBytes": 0},
               "boundaries": [{"boundary": "start", "observed_at": timestamp, "monotonic_ns": 1}, {"boundary": "stop", "observed_at": timestamp, "monotonic_ns": tick()}],
               "interval_seconds": 0.1, "maximum_samples": 128, "maximum_raw_bytes": 4 * 1024 * 1024,
               "registered_processes": identities, "samples": samples, "raw_sources": entries, "child_terminal_observations": terminals,
               "raw_file": {"path": "finite-cpu-telemetry.raw", "size_bytes": len(raw), "sha256": native_sha(bytes(raw))},
               "errors": [], "gaps": [], "dropped_error_count": 0, "dropped_gap_count": 0,
               "observer_wall_nanoseconds": 1, "observer_thread_cpu_nanoseconds": 1,
               "overhead_scope": "SYNTHETIC UNEXECUTED TEST DATA; invented collection overhead only, no actual observer ran.",
               "terminal_coverage": {"state": "complete", "raw_overflow": False, "sample_overflow": False, "resolved_terminal_races": 0},
               "coverage": dict(CPU_COVERAGE), "filesystem_backing_mount": {"native_record": mount},
               "limits": ["Entire summary, raw bytes, PIDs, counters and availability are SYNTHETIC UNEXECUTED TEST DATA, not an actual collection."]}
    raw_ref = write_bytes(bundle, f"attempts/{attempt_id}/finite-cpu-telemetry.raw", bytes(raw))
    summary_ref = write_json(bundle, f"attempts/{attempt_id}/finite-cpu-telemetry.json", summary)
    return summary, [raw_ref, summary_ref]


def build_cpu_fixture(package: Path) -> dict:
    """Write only receipt/evidence, never a package manifest or reader copy.

    All identities, reviews, timestamps, bytes and timings are SYNTHETIC TEST
    DATA. This reusable fixture exercises the contract, not scientific validity.
    """
    root = "evidence/qualified-cpu"
    source = write_bytes(package, f"{root}/source.py", b"# SYNTHETIC TEST FIXTURE; NO EXECUTION\n")
    payload = b"SYNTHETIC TEST FIXTURE; NOT A REAL CORPUS\n"
    input_ref = write_bytes(package, f"{root}/input.bin", payload)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(zipfile.ZipInfo("SYNTHETIC-NO-EXECUTION.txt", date_time=(2026, 1, 1, 0, 0, 0)), "Synthetic source-bundle hash fixture; contains no actual observer implementation.\n")
    telemetry_bytes = stream.getvalue()
    telemetry_ref = write_bytes(package, f"{root}/inputs/finite-cpu-telemetry.zip", telemetry_bytes)
    telemetry_bundle = {"artifactName": "inputs/finite-cpu-telemetry.zip", "sha256": telemetry_ref["sha256"], "sizeBytes": len(telemetry_bytes)}
    telemetry_json = json.dumps(telemetry_bundle, sort_keys=True, separators=(",", ":"))
    groups = []
    for number in range(1, 5):
        groups.extend([
            {"id": f"pair-{number:02}", "kind": "pair", "order": ["baseline", "candidate"] if number % 2 else ["candidate", "baseline"]},
            {"id": f"control-{number:02}", "kind": "control", "order": ["baseline", "candidate"]},
        ])
    toolchain = {"kind": "piq.codec-toolchain.v1", "threads": 2, "synthetic_fixture": True}
    toolchain_json = json.dumps(toolchain, sort_keys=True, separators=(",", ":"))
    design = {"schema_version": 1, "kind": "qualified_cpu_codec_design",
              "study_id": "SYNTHETIC-CPU-CONTRACT-FIXTURE", "synthetic_fixture": True,
              "created_at": "2026-10-28T00:00:00Z", "execution_owner": "Synthetic fixture owner",
              "execution_class": "cpu", "accelerator_count": 0,
              "workload_scope": "Synthetic lossless codec contract fixture, no execution",
              "runtime_identifier": "synthetic-runtime/no-execution",
              "hardware_identifier": "synthetic-cpu/no-hardware",
              "host_observations": {"hostname": "synthetic-host", "cpu_models": ["synthetic-cpu"], "cpu_affinity": [0, 1]},
              "python_runtime": "synthetic Python; no process executed",
              "source_artifacts": [source], "input": input_ref, "input_artifacts": [input_ref],
              "input_bytes": len(payload), "format_policy": "flexible_lossless",
              "correctness_oracle": "exact_input_sha256_roundtrip",
              "timed_boundary": "complete_fresh_process", "timed_steps": TIMED_STEPS,
              "criteria": copy.deepcopy(CRITERIA), "groups": groups,
              "arms": {"baseline": {"id": "synthetic-standard", "implementation": "synthetic baseline", "configuration": "synthetic fixed configuration", "native_configuration": {"codec": "pigz-gzip", "toolchain": toolchain_json, "telemetryBundle": telemetry_json}},
                       "candidate": {"id": "synthetic-treatment", "implementation": "synthetic candidate", "configuration": "synthetic fixed configuration", "native_configuration": {"codec": "zstd", "toolchain": toolchain_json, "telemetryBundle": telemetry_json}}}}
    design_ref = write_json(package, f"{root}/design.json", design)
    shared = {"study_id": design["study_id"], "design_sha256": design_ref["sha256"], "synthetic_fixture": True}
    registration = write_json(package, f"{root}/registration.json", {**shared, "registered_at": "2026-10-28T00:00:02Z", "authority": "Synthetic registration, not an actual authority", "record_id": "SYNTHETIC-REGISTRATION"})
    baseline_review = write_json(package, f"{root}/baseline-review.json", {**shared, "reviewer": "Synthetic independent baseline reviewer", "independent": True, "verdict": "PASS", "reviewed_at": "2026-10-28T00:00:01Z", "baseline_id": design["arms"]["baseline"]["id"], "optimized_standard": True, "primary_sources": ["https://example.invalid/synthetic-primary-source"], "reasoning": "Synthetic fixture only; no baseline assessment occurred."})
    envelope_refs, census, samples, group_design_hashes = [], [], [], {}
    moment = datetime(2026, 10, 28, 0, 0, 3, tzinfo=timezone.utc)
    def utc(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")
    for group in groups:
        bundle_relative = f"{root}/bundles/{group['id']}"
        bundle = package / bundle_relative
        experiment_id, run_id = f"synthetic-experiment-{group['id']}", f"synthetic-run-{group['id']}"
        local_source = write_bytes(bundle, "source/fixture.py", (package / source["path"]).read_bytes())
        local_input = write_bytes(bundle, "evidence/input.bin", payload)
        local_telemetry = write_bytes(bundle, "evidence/inputs/finite-cpu-telemetry.zip", telemetry_bytes)
        stack_identity = write_json(bundle, "evidence/stack-identity.json", {"workload": {"input_artifacts": [local_telemetry]}, "synthetic_fixture": True})
        objective = {"metric": "complete_process_elapsed", "unit": "ns", "direction": "minimize"}
        native_design = {"experiment_id": experiment_id, "workload_family": "cpu_artifact_codec",
                         "source_artifacts": [local_source], "input": local_input, "objective": objective,
                         "required_evidence_roles": ["input", "output"],
                         "input_artifacts": [local_telemetry],
                         "qualification_study": {"study_id": design["study_id"], "design_sha256": design_ref["sha256"], "group_id": group["id"], "group": group},
                         "synthetic_fixture": True}
        native_design["resources"] = {"executor": "local-process", "acceleratorCount": 0, "nodeCount": 1, "processCount": 1, "cpuCount": 2, "memoryLimitMiB": 512, "timeoutSeconds": 15}
        native_design["arms"] = []
        for role in ("baseline", "candidate"):
            arm = design["arms"]["baseline" if group["kind"] == "control" else role]
            configuration = arm["native_configuration"]
            native_design["arms"].append({"arm_id": role, "configuration": configuration,
                                           "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()})
        native_design_ref = write_json(bundle, "evidence/design.json", native_design)
        group_design_hashes[group["id"]] = native_design_ref["sha256"]
        attempts, native_refs, group_rows = [], [], []
        group_start = moment
        for role in group["order"]:
            attempt_id = f"{group['id']}-{role}"
            candidate = role == "candidate" and group["kind"] == "pair"
            sample = {**shared, "study_design_sha256": design_ref["sha256"], "group_id": group["id"], "role": role,
                      "run_id": run_id, "attempt_id": attempt_id, "configuration_id": design["arms"]["candidate" if candidate else "baseline"]["id"],
                      "input_sha256": input_ref["sha256"], "source_sha256s": [source["sha256"]],
                      "runtime_identifier": design["runtime_identifier"], "hardware_identifier": design["hardware_identifier"],
                      "device_type": "cpu", "accelerator_count": 0, "device_uuids": [], "fresh_process": True,
                      "process_id": 1000 + len(samples), "timed_steps": TIMED_STEPS,
                      "started_at": utc(moment), "ended_at": utc(moment + timedelta(seconds=1)),
                      "elapsed_ns": 800_000_000 if candidate else 1_000_000_000,
                      "status": "succeeded", "accepted": True, "failure_reason": None,
                      "parts": [{"input_sha256": input_ref["sha256"], "roundtrip_sha256": input_ref["sha256"], "archive_bytes": 80 if candidate else 100}]}
            raw = {"experiment_id": experiment_id, "attempt_id": attempt_id, "run_id": run_id, "role": role,
                   "design_sha256": native_design_ref["sha256"], "input_sha256": local_input["sha256"],
                   "study_sample": sample, "synthetic_fixture": True}
            arm = design["arms"]["candidate" if candidate else "baseline"]
            archive = write_bytes(bundle, f"attempts/{attempt_id}/archive.bin", b"S" * (80 if candidate else 100))
            archive_local = {"artifactName": "archive.bin", "sha256": archive["sha256"], "sizeBytes": 80 if candidate else 100}
            observations = {"runtime": {"toolchain": toolchain, "python": design["python_runtime"]},
                            "host": {**design["host_observations"], "process_id": sample["process_id"]},
                            "input_manifest_sha256": input_ref["sha256"], "input_bytes": len(payload), "timed_steps": TIMED_STEPS,
                            "parts": [{"input": {"sha256": input_ref["sha256"], "sizeBytes": len(payload)}, "archive": archive_local,
                                       "roundtrip_sha256": input_ref["sha256"], "roundtrip_bytes": len(payload)}]}
            child_pids = [100000 + len(samples) * 2, 100001 + len(samples) * 2]
            observations["parts"][0].update(compression={"pid": child_pids[0], "exit_code": 0, "synthetic_fixture": True},
                                              decompression={"pid": child_pids[1], "exit_code": 0, "synthetic_fixture": True})
            telemetry, telemetry_refs = synthetic_telemetry(bundle, attempt_id, sample["process_id"], child_pids, telemetry_ref["sha256"], sample["started_at"])
            observations["runtime"]["telemetry_bundle"] = telemetry_bundle
            observations["telemetry"] = telemetry
            observation_ref = write_json(bundle, f"attempts/{attempt_id}/native-evidence.json", observations)
            native_output = write_json(bundle, f"attempts/{attempt_id}/result.json", {"synthetic_fixture": True, "input_sha256": input_ref["sha256"]})
            raw["native_attempt"] = {"armId": role, "attemptId": attempt_id, "childPid": sample["process_id"],
                                     "startedAt": sample["started_at"], "completedAt": sample["ended_at"],
                                     "elapsedSeconds": "0.800000000" if candidate else "1.000000000", "state": "completed", "exitCode": 0, "failureReason": None,
                                     "output": {"artifactName": native_output["path"], "sha256": native_output["sha256"]}}
            raw["arm_configuration"] = arm["native_configuration"]
            raw["adapter_result"] = {"native_evidence": {"observations": {"artifactName": "native-evidence.json", "sha256": observation_ref["sha256"]},
                                                        "cpu_telemetry_raw": {"artifactName": "finite-cpu-telemetry.raw", "sha256": telemetry_refs[0]["sha256"]},
                                                        "cpu_telemetry_summary": {"artifactName": "finite-cpu-telemetry.json", "sha256": telemetry_refs[1]["sha256"]}}}
            raw_ref = write_json(bundle, f"evidence/{role}.json", raw)
            native_refs.extend([raw_ref, archive, observation_ref, native_output, *telemetry_refs])
            attempts.append({"id": attempt_id, "run_id": run_id, "role": role, "status": "succeeded", "accepted": True, "raw_result": raw_ref, "failure_reason": None})
            group_rows.append({"group_id": group["id"], "role": role, "run_id": run_id, "attempt_id": attempt_id,
                               "raw_result": {"path": f"{bundle_relative}/{raw_ref['path']}", "sha256": raw_ref["sha256"]}})
            samples.append(sample)
            moment += timedelta(seconds=3)
        output = write_json(bundle, "evidence/output.json", {"experiment_id": experiment_id, "run_ids": [run_id], "synthetic_fixture": True})
        correctness = write_json(bundle, "evidence/correctness.json", {"experiment_id": experiment_id, "accepted": True, "synthetic_fixture": True})
        envelope = {"schema_version": 1, "kind": "piq_experiment_evidence", "scope": "diagnostic", "workload_family": "cpu_artifact_codec",
                    "experiment": {"id": experiment_id, "question": "Synthetic contract fixture?", "hypothesis": "Synthetic file bindings agree.", "workload_scope": design["workload_scope"], "design": native_design_ref, "source_artifacts": [local_source]},
                    "stack": {"piq_revision": "0" * 40, "runtime_identifier": design["runtime_identifier"], "hardware_identifier": design["hardware_identifier"], "layers": [{"name": name, "identifier": f"synthetic/{name}"} for name in ("host", "runtime", "workload")]},
                    "objective": {**objective, "window": {"start": utc(group_start), "end": utc(moment)}},
                    "attempts": attempts, "comparison": {"baseline_attempt_ids": [a["id"] for a in attempts if a["role"] == "baseline"], "candidate_attempt_ids": [a["id"] for a in attempts if a["role"] == "candidate"], "scientific_disposition": "DIAGNOSTIC_ONLY", "reason": "Synthetic fixture only; no PIQ execution."},
                    "correctness": {"accepted": True, **correctness}, "workload_evidence": {"input": local_input, "output": output},
                    "artifacts": [local_source, local_input, local_telemetry, stack_identity, native_design_ref, *native_refs, output, correctness]}
        envelope_ref = write_json(package, f"{bundle_relative}/envelope.json", envelope)
        envelope_refs.append(envelope_ref)
        census.extend({**row, "envelope": envelope_ref} for row in group_rows)
    registration = write_json(package, f"{root}/registration.json", {**shared, "registered_at": "2026-10-28T00:00:02Z", "authority": "Synthetic registration, not an actual authority", "record_id": "SYNTHETIC-REGISTRATION", "group_design_sha256s": group_design_hashes})
    raw_census = write_json(package, f"{root}/raw-census.json", {**shared, "attempts": census})
    correctness_ref = write_json(package, f"{root}/correctness.json", {**shared, "accepted": True, "attempts": [{"run_id": s["run_id"], "attempt_id": s["attempt_id"], "parts": s["parts"]} for s in samples]})
    report_ref = write_json(package, f"{root}/qualification-report.json", evaluate_cpu_study(design, samples, design_sha256=design_ref["sha256"]))
    scientific_review = write_json(package, f"{root}/scientific-review.json", {**shared, "reviewer": "Synthetic independent scientific reviewer", "independent": True, "verdict": "PASS", "reviewed_at": "2026-10-28T00:02:00Z", "reasoning": "Synthetic fixture only; no scientific review occurred.", "qualification_report_sha256": report_ref["sha256"]})
    receipt = {"schema_version": 2, "state": "measured", "execution_class": "cpu", "workload_scope": design["workload_scope"],
               "runtime_identifier": design["runtime_identifier"], "hardware_identifier": design["hardware_identifier"],
               "design": design_ref, "registration": registration, "raw_census": raw_census, "correctness": correctness_ref,
               "qualification_report": report_ref, "baseline_review": baseline_review, "scientific_review": scientific_review,
               "source_artifacts": [source], "piq_envelopes": envelope_refs,
               "accepted_repeats": [{"id": group["id"], "accepted": True, "raw_result": envelope_refs[index]} for index, group in enumerate(groups) if group["kind"] == "pair"],
               "failed_attempts": [], "profile_evidence": [],
               "claims": [{"id": "synthetic-result-only", "source": report_ref, "accepted_repeat_ids": [g["id"] for g in groups if g["kind"] == "pair"], "profile_claim": False}]}
    write_json(package, "evidence/measurement-receipt.json", receipt)
    return receipt


class QualifiedCpuStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        self.receipt = build_cpu_fixture(self.package)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def document(self, name: str) -> dict:
        return json.loads((self.package / self.receipt[name]["path"]).read_text())

    def samples(self) -> list[dict]:
        return [json.loads((self.package / row["raw_result"]["path"]).read_text())["study_sample"] for row in self.document("raw_census")["attempts"]]

    def mutate_first_raw(self, mutation) -> None:
        census = self.document("raw_census")
        row = census["attempts"][0]
        raw = json.loads((self.package / row["raw_result"]["path"]).read_text())
        mutation(raw)
        raw_ref = write_json(self.package, row["raw_result"]["path"], raw)
        envelope_ref = row["envelope"]
        envelope = json.loads((self.package / envelope_ref["path"]).read_text())
        bundle_root = Path(envelope_ref["path"]).parent
        local_raw = {"path": Path(raw_ref["path"]).relative_to(bundle_root).as_posix(), "sha256": raw_ref["sha256"]}
        for attempt in envelope["attempts"]:
            if attempt["id"] == row["attempt_id"]:
                attempt["raw_result"] = local_raw
        envelope["artifacts"] = [local_raw if ref["path"] == local_raw["path"] else ref for ref in envelope["artifacts"]]
        new_envelope_ref = write_json(self.package, envelope_ref["path"], envelope)
        self.receipt["piq_envelopes"] = [new_envelope_ref if ref["path"] == envelope_ref["path"] else ref for ref in self.receipt["piq_envelopes"]]
        for repeat in self.receipt["accepted_repeats"]:
            if repeat["raw_result"]["path"] == envelope_ref["path"]:
                repeat["raw_result"] = new_envelope_ref
        for entry in census["attempts"]:
            if entry["envelope"]["path"] == envelope_ref["path"]:
                entry["envelope"] = new_envelope_ref
            if entry["raw_result"]["path"] == raw_ref["path"]:
                entry["raw_result"] = raw_ref
        self.receipt["raw_census"] = write_json(self.package, self.receipt["raw_census"]["path"], census)

    def first_telemetry(self) -> tuple[Path, dict, dict, dict]:
        envelope_path = self.package / self.receipt["piq_envelopes"][0]["path"]
        bundle = envelope_path.parent
        envelope = json.loads(envelope_path.read_text())
        raw = json.loads((bundle / envelope["attempts"][0]["raw_result"]["path"]).read_text())
        directory = Path(raw["native_attempt"]["output"]["artifactName"]).parent
        observations = json.loads((bundle / directory / "native-evidence.json").read_text())
        return bundle, envelope, raw, observations

    def check_mutated_telemetry(self, mutation) -> list[str]:
        bundle, envelope, raw, observations = self.first_telemetry()
        directory = Path(raw["native_attempt"]["output"]["artifactName"]).parent
        summary_path = bundle / directory / "finite-cpu-telemetry.json"
        original = summary_path.read_bytes()
        try:
            mutation(observations["telemetry"], observations, raw)
            ref = write_json(bundle, (directory / "finite-cpu-telemetry.json").as_posix(), observations["telemetry"])
            envelope["artifacts"] = [ref if item["path"] == ref["path"] else item for item in envelope["artifacts"]]
            if "cpu_telemetry_summary" in raw["adapter_result"]["native_evidence"]:
                raw["adapter_result"]["native_evidence"]["cpu_telemetry_summary"]["sha256"] = ref["sha256"]
            return validate_cpu_telemetry(bundle, envelope, raw, observations)
        finally:
            summary_path.write_bytes(original)

    def test_valid_cpu_v2_without_gpu_placeholders_or_profiles(self) -> None:
        self.assertEqual(validate_cpu_receipt(self.package, self.receipt), [])
        self.assertTrue(cpu_receipt_binds_envelope(self.package, self.receipt, Path(self.receipt["piq_envelopes"][0]["path"])))

    def test_fixed_arithmetic_and_scope_failures(self) -> None:
        design, samples = self.document("design"), self.samples()
        cases = {
            "missing treatment": lambda d, s: s.pop(0),
            "missing control": lambda d, s: s.pop(2),
            "gain": lambda d, s: s[1].update(elapsed_ns=950_000_000),
            "observed max latency": lambda d, s: s[1].update(elapsed_ns=1_100_000_000),
            "neutral drift": lambda d, s: s[3].update(elapsed_ns=1_100_000_000),
            "correctness": lambda d, s: s[1]["parts"][0].update(roundtrip_sha256="0" * 64),
            "archive tradeoff": lambda d, s: s[1]["parts"][0].update(archive_bytes=101),
            "identity": lambda d, s: s[1].update(hardware_identifier="another CPU"),
            "GPU": lambda d, s: s[1].update(device_type="cuda", accelerator_count=1),
            "missing I/O": lambda d, s: s[1].update(timed_steps=["compression"]),
            "failure": lambda d, s: s[1].update(status="failed", accepted=False, failure_reason="synthetic failure"),
            "changed criteria": lambda d, s: d["criteria"].update(minimum_each_paired_ratio="1.01"),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                candidate_design, candidate_samples = copy.deepcopy(design), copy.deepcopy(samples)
                mutation(candidate_design, candidate_samples)
                report = evaluate_cpu_study(candidate_design, candidate_samples, design_sha256=self.receipt["design"]["sha256"])
                self.assertEqual(report["verdict"], "INCONCLUSIVE")
                self.assertTrue(report["errors"])

    def test_receipt_file_and_binding_failures(self) -> None:
        for field, value, phrase in (
            ("power_limit_w", 1000, "power/device"),
            ("execution_class", "cuda", "cpu execution"),
            ("failed_attempts", [{"attempt_id": "omitted-failure"}], "failed_attempts"),
            ("accepted_repeats", self.receipt["accepted_repeats"][:2], "four distinct"),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt[field] = value
                self.assertTrue(any(phrase in error for error in validate_cpu_receipt(self.package, receipt)))
        receipt = copy.deepcopy(self.receipt)
        receipt["claims"][0]["accepted_repeat_ids"].pop()
        self.assertTrue(any("every distinct accepted" in error for error in validate_cpu_receipt(self.package, receipt)))
        receipt = copy.deepcopy(self.receipt)
        receipt["claims"][0]["profile_claim"] = True
        self.assertTrue(any("actual CPU captures" in error for error in validate_cpu_receipt(self.package, receipt)))

    def test_omitted_native_attempt_and_forged_qualification_report_fail(self) -> None:
        census = self.document("raw_census")
        census["attempts"].pop()
        self.receipt["raw_census"] = write_json(self.package, self.receipt["raw_census"]["path"], census)
        self.assertTrue(any("conserve every PIQ" in error for error in validate_cpu_receipt(self.package, self.receipt)))
        report = self.document("qualification_report")
        report["pairs"][0]["baseline_over_candidate_ratio"] = "100"
        self.receipt["qualification_report"] = write_json(self.package, self.receipt["qualification_report"]["path"], report)
        self.assertTrue(any("deterministic recomputation" in error for error in validate_cpu_receipt(self.package, self.receipt)))

    def test_registration_and_review_must_bind_advance_design(self) -> None:
        registration = self.document("registration")
        registration["registered_at"] = "2026-10-28T01:00:00Z"
        self.receipt["registration"] = write_json(self.package, self.receipt["registration"]["path"], registration)
        self.assertTrue(any("precede every execution" in error for error in validate_cpu_receipt(self.package, self.receipt)))
        review = self.document("baseline_review")
        review["reasoning"] = ""
        self.receipt["baseline_review"] = write_json(self.package, self.receipt["baseline_review"]["path"], review)
        self.assertTrue(any("named independent review" in error for error in validate_cpu_receipt(self.package, self.receipt)))

    def test_canonical_parent_time_is_used_and_projection_time_is_ignored(self) -> None:
        self.mutate_first_raw(lambda raw: raw["study_sample"].update(elapsed_ns=999_999_999_999))
        self.assertEqual(validate_cpu_receipt(self.package, self.receipt), [])
        self.mutate_first_raw(lambda raw: raw["native_attempt"].update(elapsedSeconds="0.100000000"))
        errors = validate_cpu_receipt(self.package, self.receipt)
        self.assertTrue(any("fixed gain" in error for error in errors))
        self.assertTrue(any("deterministic recomputation" in error for error in errors))

    def test_malformed_containers_fail_closed(self) -> None:
        for field, value in (("design", "wrong type"), ("piq_envelopes", [None] * 8), ("claims", [{"id": []}])):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt[field] = value
                self.assertTrue(validate_cpu_receipt(self.package, receipt))
        report = evaluate_cpu_study({"criteria": []}, [None], design_sha256="0" * 64)
        self.assertEqual(report["verdict"], "INCONCLUSIVE")

    def test_cpu_telemetry_requires_exact_indexed_raw_bytes_and_summary(self) -> None:
        bundle, envelope, raw, observations = self.first_telemetry()
        self.assertEqual(validate_cpu_telemetry(bundle, envelope, raw, observations), [])
        absent = copy.deepcopy(raw)
        absent["adapter_result"]["native_evidence"].pop("cpu_telemetry_raw")
        self.assertTrue(any("must name" in error for error in validate_cpu_telemetry(bundle, envelope, absent, observations)))
        summary_drift = copy.deepcopy(observations)
        summary_drift["telemetry"]["observer_wall_nanoseconds"] += 1
        self.assertTrue(any("differs from observations" in error for error in validate_cpu_telemetry(bundle, envelope, raw, summary_drift)))
        raw_path = bundle / Path(raw["native_attempt"]["output"]["artifactName"]).parent / "finite-cpu-telemetry.raw"
        raw_path.write_bytes(raw_path.read_bytes() + b"UNINDEXED SYNTHETIC DRIFT")
        errors = validate_cpu_telemetry(bundle, envelope, raw, observations)
        self.assertTrue(any("digest/size" in error or "omits retained bytes" in error for error in errors))

    def test_cpu_telemetry_rejects_partial_omitted_and_false_complete_records(self) -> None:
        cases = {
            "partial": (lambda s, o, r: s["terminal_coverage"].update(state="partial"), "complete terminal"),
            "overflow": (lambda s, o, r: s["terminal_coverage"].update(raw_overflow=True), "complete terminal"),
            "dropped": (lambda s, o, r: s.update(dropped_gap_count=1), "dropped records"),
            "omitted child": (lambda s, o, r: s["registered_processes"].pop(1), "registration omits"),
            "omitted terminal": (lambda s, o, r: s["child_terminal_observations"].pop(), "terminal census"),
            "missing required context despite complete": (lambda s, o, r: next(e for e in s["raw_sources"] if e["source_id"] == "cgroup.io.max" and e.get("boundary") == "start").update(source_id="cgroup.omitted.io.max"), "required start"),
            "truncated source despite complete": (lambda s, o, r: s["raw_sources"][0].update(truncated=True), "truncation mismatch"),
            "overlapping raw range": (lambda s, o, r: s["raw_sources"][1].update(offset=0), "sequence/offset/size"),
            "missing raw sequence": (lambda s, o, r: s["raw_sources"][1].update(sequence=999), "sequence/offset/size"),
            "forged span digest": (lambda s, o, r: s["raw_sources"][1].update(source_sha256="sha256:" + "0" * 64), "digest/truncation"),
            "fake source producer": (lambda s, o, r: s["source"].update(producerSha256="sha256:" + "0" * 64), "source identity"),
            "wrong normalized PID": (lambda s, o, r: s["samples"][0]["measurements"][0]["labels"].update(process_id="999999"), "process identity"),
            "fake accelerator": (lambda s, o, r: s["coverage"].update(accelerator="actual-capture"), "unexercised"),
        }
        for name, (mutation, phrase) in cases.items():
            with self.subTest(name=name):
                errors = self.check_mutated_telemetry(mutation)
                self.assertTrue(any(phrase in error for error in errors), errors)

    def test_cpu_telemetry_cannot_resolve_missing_cgroup_context_as_child_exit(self) -> None:
        def mutation(summary, observations, raw):
            term = summary["child_terminal_observations"][0]
            summary["errors"].append({"source_id": "cgroup.io.max", "pid": term["pid"], "error_class": "FileNotFoundError",
                                      "monotonic_ns": term["monotonic_ns"], "terminal_race_eligible": True,
                                      "resolution": "owning-Popen-terminal-observation", "terminal_sequence": 0})
        errors = self.check_mutated_telemetry(mutation)
        self.assertTrue(any("falsely resolves context" in error for error in errors), errors)

    def test_cpu_telemetry_raw_child_must_have_actual_adapter_parent(self) -> None:
        bundle, envelope, raw, observations = self.first_telemetry()
        summary = observations["telemetry"]
        directory = Path(raw["native_attempt"]["output"]["artifactName"]).parent
        raw_path = bundle / directory / "finite-cpu-telemetry.raw"
        content = raw_path.read_bytes()
        child = summary["registered_processes"][1]["pid"]
        entry = next(item for item in summary["raw_sources"] if item["source_id"] == "process.identity.stat" and item["pid"] == child)
        offset, size = entry["offset"], entry["size_bytes"]
        replacement = content[offset:offset + size].replace(b") S 1000 ", b") S 9999 ", 1)
        self.assertNotEqual(replacement, content[offset:offset + size])
        changed = content[:offset] + replacement + content[offset + size:]
        entry["sha256"] = entry["source_sha256"] = "sha256:" + hashlib.sha256(replacement).hexdigest()
        summary["raw_file"]["sha256"] = "sha256:" + hashlib.sha256(changed).hexdigest()
        raw_ref = write_bytes(bundle, (directory / "finite-cpu-telemetry.raw").as_posix(), changed)
        summary_ref = write_json(bundle, (directory / "finite-cpu-telemetry.json").as_posix(), summary)
        for name, ref in (("cpu_telemetry_raw", raw_ref), ("cpu_telemetry_summary", summary_ref)):
            raw["adapter_result"]["native_evidence"][name]["sha256"] = ref["sha256"]
            envelope["artifacts"] = [ref if item["path"] == ref["path"] else item for item in envelope["artifacts"]]
        errors = validate_cpu_telemetry(bundle, envelope, raw, observations)
        self.assertTrue(any("raw parent/child PID" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
