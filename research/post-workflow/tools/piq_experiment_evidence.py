#!/usr/bin/env python3
"""Validate and privately import versioned PIQ diagnostic evidence (stdlib only).

A valid bundle establishes retained diagnostic provenance, never qualification.
The existing measured package receipt and scientific/publication gates are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any

ENVELOPE = "envelope.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
# New families share the diagnostic contract; register only extra role requirements.
FAMILY_REQUIREMENTS = {"cpu_data_query": {"query_plan"}}
COMMON_ROLES = {"input", "output"}
COMMON_LAYERS = {"host", "runtime", "workload"}
CORRECTNESS_ROLES = {"oracle_reference", "oracle_candidate", "correctness_report"}
INPUT_ARTIFACT_PATH = re.compile(r"evidence/inputs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
RESERVED_INPUT_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)}
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024


def correctness_configuration(value: Any, errors: list[str]) -> str | None:
    """Validate the declared oracle configuration, never reimplement its math."""
    if not isinstance(value, dict):
        errors.append("correctness check configuration must be an object")
        return None
    oracle = value.get("oracle")
    fields = {
        "output-sha256-equality": {"oracle"},
        "json-numeric-tolerance": {"oracle", "atol", "rtol"},
        "json-keyed-record-set-equality": {"oracle", "recordKey"},
    }
    if not isinstance(oracle, str) or oracle not in fields:
        errors.append("correctness check configuration has an unsupported oracle")
        return None
    valid = True
    if set(value) != fields[oracle]:
        errors.append("correctness check configuration fields do not match its oracle")
        valid = False
    if oracle == "json-numeric-tolerance":
        for field in ("atol", "rtol"):
            raw = value.get(field)
            try:
                number = Decimal(raw) if isinstance(raw, str) and len(raw) <= 64 and DECIMAL.fullmatch(raw) else None
                if number is None or not number.is_finite() or number < 0:
                    raise ValueError("expected a finite nonnegative decimal string")
            except (InvalidOperation, ValueError):
                errors.append(f"correctness check configuration {field} must be a finite nonnegative decimal string")
                valid = False
    if oracle == "json-keyed-record-set-equality":
        key = value.get("recordKey")
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            errors.append("correctness check configuration recordKey must be a nonempty string")
            valid = False
    if not valid:
        return None
    # Same JSON projection as piq_verify.canonical.canonical_sha256. Configs
    # contain strings only; no numerical comparison or oracle decision occurs.
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (UnicodeError, ValueError):
        errors.append("correctness check configuration must contain canonical JSON strings")
        return None
    return hashlib.sha256(encoded).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and "\\" not in value
            and not PurePosixPath(value).is_absolute()
            and all(part not in {"", ".", ".."} for part in value.split("/")))


def read_json(path: Path, label: str, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError, UnicodeError) as exc:
        errors.append(f"{label}: cannot read JSON: {exc}")
        return None


def same_json(left: Any, right: Any) -> bool:
    """Compare metadata without Python's boolean/integer equality coercion."""
    try:
        options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def ingestion_metadata_sha256(value: Any, *, newline: bool = False) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw + (b"\n" if newline else b"")).hexdigest()


def validate_slurm_observation(
    observation: dict, manifest: dict, manifest_sha256: str, resources: dict,
    design: dict, source_refs: list, input_ref: Any, errors: list[str],
) -> None:
    """Join retained Slurm metadata; never inspect or authorize a live job."""
    if observation.get("schemaVersion") != "piq.finite-slurm-observation.v1" or observation.get("state") != "RUNNING":
        errors.append("GPU assignment requires an original finite Slurm RUNNING observation")
    job_id = observation.get("jobId")
    if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]{0,17}", job_id) is None:
        errors.append("GPU Slurm observation jobId must be an exact numeric job ID")
    plan = manifest.get("finiteExperiment")
    plan = plan if isinstance(plan, dict) else {}
    if manifest.get("schemaVersion") != "piq.finite-experiment-executable.v1" or plan.get("experimentId") != design.get("experiment_id"):
        errors.append("GPU Slurm original executable manifest does not bind this experiment")
    executable = manifest.get("executable")
    executable = executable if isinstance(executable, dict) else {}
    runtime = [item for item in plan.get("stack", []) if isinstance(item, dict) and item.get("layer") == "runtime"] if isinstance(plan.get("stack"), list) else []
    source, input_artifact = plan.get("source"), plan.get("input")
    source = source if isinstance(source, dict) else {}
    input_artifact = input_artifact if isinstance(input_artifact, dict) else {}
    requested = {key: value for key, value in resources.items() if key != "assignmentEvidence"}
    if not same_json(plan.get("resources"), requested) or not same_json(design.get("resources"), requested):
        errors.append("GPU Slurm manifest resources do not match immutable requested resources")
    if not any(same_json({"path": source.get("artifactName"), "sha256": source.get("sha256")}, item) for item in source_refs):
        errors.append("GPU Slurm manifest source does not match retained source artifacts")
    if not same_json({"path": input_artifact.get("artifactName"), "sha256": input_artifact.get("sha256")}, input_ref):
        errors.append("GPU Slurm manifest input does not match retained input")
    expected = {
        "planSha256": plan.get("planSha256"), "sourceSha256": source.get("sha256"),
        "inputSha256": input_artifact.get("sha256"),
        "runtimeIdentitySha256": runtime[0].get("identitySha256") if len(runtime) == 1 else None,
        "runtimeExecutableSha256": executable.get("sha256"),
        "targetId": requested.get("targetId"), "cpuCount": requested.get("cpuCount"),
        "memoryLimitMiB": requested.get("memoryLimitMiB"),
        "acceleratorCount": 1, "nodeCount": 1, "processCount": 1,
    }
    binding = observation.get("jobBinding")
    binding = binding if isinstance(binding, dict) else {}
    observation_fields = set(expected) | {
        "schemaVersion", "jobId", "state", "ownerUid", "nodeName", "observedAt", "startedAt", "expiresAt",
        "wallTimeSeconds", "deviceUuid", "deviceIndex", "jobComment", "jobBinding", "jobBindingSha256",
        "processId", "cgroupPath", "cgroupMemoryMaxBytes",
    }
    binding_fields = set(expected) | {
        "schemaVersion", "manifestSha256", "ownerEntrypoint", "ownerUid", "targetNode", "partition",
        "wallTimeSeconds", "outputDirectory",
    }
    if set(observation) != observation_fields or set(binding) != binding_fields:
        errors.append("GPU Slurm observation/job binding fields do not match their v1 metadata schemas")
    for field in ("targetNode", "partition"):
        value = binding.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
            errors.append(f"GPU Slurm job binding {field} must be a portable identifier")
    output = binding.get("outputDirectory")
    if not isinstance(output, str) or not output.startswith("/") or any(c in output for c in "\x00\n\r") or ".." in output.split("/"):
        errors.append("GPU Slurm job binding outputDirectory must retain an absolute single-line path")
    entrypoint = binding.get("ownerEntrypoint")
    if entrypoint is not None and (
        not isinstance(entrypoint, dict) or set(entrypoint) != {"path", "sha256"}
        or not isinstance(entrypoint.get("path"), str) or not entrypoint["path"].startswith("/")
        or not isinstance(entrypoint.get("sha256"), str) or SHA256.fullmatch(entrypoint["sha256"]) is None
    ):
        errors.append("GPU Slurm ownerEntrypoint must be null or retain its absolute path and hash")
    for field, value in expected.items():
        if value is None or not same_json(observation.get(field), value) or not same_json(binding.get(field), value):
            errors.append(f"GPU Slurm observation/job binding {field} does not match manifest/resources")
    for field in ("planSha256", "sourceSha256", "inputSha256", "runtimeIdentitySha256", "runtimeExecutableSha256", "jobBindingSha256"):
        value = observation.get(field)
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            errors.append(f"GPU Slurm observation {field} must be a SHA-256 digest")
    try:
        binding_hash = ingestion_metadata_sha256(binding)
    except (TypeError, ValueError, UnicodeError):
        binding_hash = None
    if binding.get("schemaVersion") != "piq.finite-slurm-job-binding.v1" or binding_hash != observation.get("jobBindingSha256"):
        errors.append("GPU Slurm jobBindingSha256 does not match retained jobBinding")
    if binding.get("manifestSha256") != manifest_sha256:
        errors.append("GPU Slurm job binding manifestSha256 does not match original executable manifest")
    if observation.get("jobComment") != "piq-finite:" + str(binding_hash):
        errors.append("GPU Slurm jobComment does not match retained job binding")
    for field in ("ownerUid", "processId", "cgroupMemoryMaxBytes"):
        value = observation.get(field)
        if type(value) is not int or value < 1:
            errors.append(f"GPU Slurm observation {field} must be a positive integer")
    if not same_json(binding.get("ownerUid"), observation.get("ownerUid")) or binding.get("targetNode") != observation.get("nodeName"):
        errors.append("GPU Slurm job owner/node does not match retained observation")
    for field in ("nodeName", "deviceUuid", "cgroupPath"):
        value = observation.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"GPU Slurm observation {field} must be a nonempty string")
    uuid, cgroup = observation.get("deviceUuid"), observation.get("cgroupPath")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        errors.append("GPU Slurm observation requires an actual GPU UUID")
    if not isinstance(cgroup, str) or not isinstance(job_id, str) or "slurm" not in cgroup.lower() or re.search(r"(?:^|/)job_" + re.escape(job_id) + r"(?:/|$)", cgroup) is None:
        errors.append("GPU Slurm cgroup metadata must name the retained job")
    memory = observation.get("cgroupMemoryMaxBytes")
    cap = requested.get("memoryLimitMiB")
    if type(cap) is not int or type(memory) is not int or not 0 < memory <= cap * 1024 * 1024:
        errors.append("GPU Slurm cgroup memory exceeds requested resource cap")
    index = observation.get("deviceIndex")
    if type(index) is not int or not 0 <= index <= 65535:
        errors.append("GPU Slurm observation deviceIndex must be a bounded nonnegative integer")
    wall = observation.get("wallTimeSeconds")
    if type(wall) is not int or not 10 <= wall <= 60 or not same_json(binding.get("wallTimeSeconds"), wall):
        errors.append("GPU Slurm observation wallTimeSeconds must match its bounded job binding")
    times = []
    for field in ("startedAt", "observedAt", "expiresAt"):
        try:
            value = observation.get(field)
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else None
            if parsed is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise ValueError("expected UTC timestamp")
            times.append(parsed)
        except ValueError:
            errors.append(f"GPU Slurm observation {field} must be a UTC timestamp")
    if len(times) == 3 and (not times[0] <= times[1] <= times[2] or type(wall) is not int or (times[2] - times[0]).total_seconds() > wall):
        errors.append("GPU Slurm historical observation window does not match bounded job wall time")


def implicit_single_cpu(design: Any, raw_results: dict) -> bool:
    """The finite CPU runner records its sole process directly, without ranks."""
    if not isinstance(design, dict):
        return False
    resources, arms = design.get("resources"), design.get("arms")
    if not isinstance(resources, dict) or not isinstance(arms, list) or len(arms) != 2:
        return False
    allowed = {"executor", "cpuCount", "memoryLimitMiB", "timeoutSeconds",
               "acceleratorCount", "nodeCount", "processCount"}
    if set(resources) - allowed or resources.get("executor") != "local-process":
        return False
    for field, expected in (("acceleratorCount", 0), ("nodeCount", 1), ("processCount", 1)):
        value = resources.get(field, expected if field != "acceleratorCount" else None)
        if type(value) is not int or value != expected:
            return False
    for field, low, high in (("cpuCount", 1, 64), ("memoryLimitMiB", 16, 262144),
                             ("timeoutSeconds", 1, 900)):
        value = resources.get(field)
        if type(value) is not int or not low <= value <= high:
            return False
    for arm in arms:
        config = arm.get("configuration") if isinstance(arm, dict) else None
        count = config.get("processCount", 1) if isinstance(config, dict) else None
        if type(count) is not int or count != 1:
            return False
    return len(raw_results) == 2 and all(isinstance(raw, dict) and isinstance(raw.get("native_attempt"), dict)
               and "rankCensus" not in raw["native_attempt"] for raw in raw_results.values())


def validate_bundle(bundle: Path) -> list[str]:
    """Check provenance and workload roles. No scope can bypass qualification."""
    errors: list[str] = []
    bundle = Path(bundle).absolute()
    envelope_path = bundle / ENVELOPE
    if not bundle.is_dir() or envelope_path.is_symlink():
        return ["PIQ bundle must be a directory with a regular envelope.json"]
    data = read_json(envelope_path, "PIQ envelope", errors)
    if not isinstance(data, dict):
        return errors or ["PIQ envelope must be an object"]

    def obj(value: Any, label: str) -> dict:
        if not isinstance(value, dict):
            errors.append(f"{label} must be an object")
            return {}
        return value

    def string(value: Any, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} must be a nonempty string")

    def strings(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
        if (not isinstance(value, list) or (nonempty and not value)
                or any(not isinstance(item, str) or not item.strip() for item in value)):
            errors.append(f"{label} must be a {'nonempty ' if nonempty else ''}list of nonempty strings")
            return []
        if len(set(value)) != len(value):
            errors.append(f"{label} must contain distinct values")
        return value

    refs: dict[str, str] = {}

    def ref(value: Any, label: str) -> Path | None:
        item = obj(value, label)
        relative, expected = item.get("path"), item.get("sha256")
        if not safe_path(relative) or relative in {ENVELOPE, "import-receipt.json"}:
            errors.append(f"{label}.path must be a safe bundle-relative artifact path")
            return None
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            errors.append(f"{label}.sha256 must be a lowercase SHA-256 digest")
            return None
        if relative in refs and refs[relative] != expected:
            errors.append(f"{label} conflicts with another hash for {relative}")
        refs[relative] = expected
        target = bundle / relative
        if any((bundle / Path(*Path(relative).parts[:i])).is_symlink() for i in range(1, len(Path(relative).parts) + 1)):
            errors.append(f"{label}.path must not use a symlink")
            return None
        try:
            target.resolve().relative_to(bundle.resolve())
            if not target.is_file() or target.stat().st_size == 0:
                errors.append(f"{label} must reference a nonempty regular file: {relative}")
                return None
            if digest(target) != expected:
                errors.append(f"{label}.sha256 does not match {relative}")
                return None
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: unsafe or unreadable path: {exc}")
            return None
        return target

    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        errors.append("PIQ envelope schema_version must equal 1")
    if data.get("kind") != "piq_experiment_evidence":
        errors.append("PIQ envelope kind must equal piq_experiment_evidence")
    if data.get("scope") != "diagnostic":
        errors.append("PIQ evidence v1 accepts diagnostic scope only; qualification requires its separate fixed scientific claim gate")
    family = data.get("workload_family")
    string(family, "workload_family")
    experiment = obj(data.get("experiment"), "experiment")
    for field in ("id", "question", "hypothesis", "workload_scope"):
        string(experiment.get(field), f"experiment.{field}")
    experiment_id = experiment.get("id")
    design_path = ref(experiment.get("design"), "experiment.design")
    source_refs = experiment.get("source_artifacts")
    if not isinstance(source_refs, list) or not source_refs:
        errors.append("experiment.source_artifacts must be a nonempty list")
        source_refs = []
    for i, value in enumerate(source_refs):
        ref(value, f"experiment.source_artifacts[{i}]")
    if len({item.get("path") for item in source_refs if isinstance(item, dict)}) != len(source_refs):
        errors.append("experiment.source_artifacts paths must be distinct")

    stack = obj(data.get("stack"), "stack")
    if not isinstance(stack.get("piq_revision"), str) or not REVISION.fullmatch(stack["piq_revision"]):
        errors.append("stack.piq_revision must be a full immutable Git revision")
    for field in ("runtime_identifier", "hardware_identifier"):
        string(stack.get(field), f"stack.{field}")
    layers = stack.get("layers")
    if not isinstance(layers, list):
        errors.append("stack.layers must be a list")
        layers = []
    names: list[str] = []
    for i, layer in enumerate(layers):
        layer = obj(layer, f"stack.layers[{i}]")
        for field in ("name", "identifier"):
            string(layer.get(field), f"stack.layers[{i}].{field}")
        if isinstance(layer.get("name"), str):
            names.append(layer["name"])
    required_layers = COMMON_LAYERS | ({"storage"} if family == "cpu_data_query" else set())
    if not required_layers <= set(names):
        errors.append("stack.layers missing required layer identities: " + ", ".join(sorted(required_layers - set(names))))
    if len(set(names)) != len(names):
        errors.append("stack.layers names must be distinct")

    objective = obj(data.get("objective"), "objective")
    for field in ("metric", "unit"):
        string(objective.get(field), f"objective.{field}")
    if objective.get("direction") not in ("minimize", "maximize"):
        errors.append("objective.direction must be minimize or maximize")
    window = obj(objective.get("window"), "objective.window")
    times = []
    for field in ("start", "end"):
        value = window.get(field)
        try:
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError("expected UTC timestamp ending Z")
            times.append(datetime.fromisoformat(value[:-1] + "+00:00"))
        except ValueError:
            errors.append(f"objective.window.{field} must be an ISO UTC timestamp ending Z")
    if len(times) == 2 and times[0] >= times[1]:
        errors.append("objective.window must have start before end")

    evidence = obj(data.get("workload_evidence"), "workload_evidence")
    required_roles = COMMON_ROLES | FAMILY_REQUIREMENTS.get(family if isinstance(family, str) else "", set())
    if not required_roles <= set(evidence):
        errors.append("workload_evidence missing required roles: " + ", ".join(sorted(required_roles - set(evidence))))
    evidence_paths = {role: ref(value, f"workload_evidence.{role}") for role, value in evidence.items()}
    design = read_json(design_path, "experiment design", errors) if design_path else None
    if isinstance(design, dict):
        bindings = {"experiment_id": experiment_id, "workload_family": family,
                    "source_artifacts": source_refs, "input": evidence.get("input"),
                    "objective": {key: objective.get(key) for key in ("metric", "unit", "direction")},
                    "required_evidence_roles": sorted(evidence)}
        for field, expected in bindings.items():
            if design.get(field) != expected:
                errors.append(f"experiment design {field} does not match envelope")
    elif design_path:
        errors.append("experiment design must be a JSON object")

    # Additional files stay opaque: bind their retained bytes and immutable
    # identities without interpreting an adapter's main input/index format.
    artifact_index = data.get("artifacts") if isinstance(data.get("artifacts"), list) else []

    def retained_metadata(relative: str) -> tuple[dict | None, dict | None]:
        matches = [item for item in artifact_index if isinstance(item, dict) and item.get("path") == relative]
        if len(matches) != 1:
            return None, None
        path = ref(matches[0], relative)
        # Historical bundles need no new metadata contract unless they declare
        # additional inputs. The ordinary artifact pass still checks bytes.
        value = read_json(path, relative, []) if path else None
        return matches[0], value if isinstance(value, dict) else None

    stack_ref, stack_identity = retained_metadata("evidence/stack-identity.json")
    manifest_ref, original_manifest = retained_metadata("executable-manifest.json")
    workload_identity = stack_identity.get("workload") if isinstance(stack_identity, dict) else None
    manifest_files = original_manifest.get("files") if isinstance(original_manifest, dict) else None
    manifest_inputs = [item for item in manifest_files if isinstance(item, dict)
                       and isinstance(item.get("path"), str) and item["path"].startswith("evidence/inputs/")] if isinstance(manifest_files, list) else []
    additional_declared = (isinstance(design, dict) and "input_artifacts" in design
                           or isinstance(workload_identity, dict) and "input_artifacts" in workload_identity
                           or bool(manifest_inputs))
    if additional_declared:
        additional = design.get("input_artifacts") if isinstance(design, dict) else None
        if not isinstance(additional, list) or not 1 <= len(additional) <= 16:
            errors.append("experiment design input_artifacts must be a list of 1 to 16 references")
            additional = []
        reserved = [experiment.get("design"), *source_refs, *evidence.values(), data.get("correctness")]
        if isinstance(data.get("attempts"), list):
            reserved += [item.get("raw_result") for item in data["attempts"] if isinstance(item, dict)]
        reserved_paths = {item["path"].casefold() for item in reserved if isinstance(item, dict) and isinstance(item.get("path"), str)}
        input_paths: set[str] = set()
        total_bytes = 0
        main_input = evidence_paths.get("input")
        if main_input:
            main_bytes = main_input.stat().st_size
            total_bytes += main_bytes
            if main_bytes > MAX_INPUT_BYTES:
                errors.append("main input exceeds the 8 MiB input file limit")
        for i, item in enumerate(additional):
            label = f"experiment design input_artifacts[{i}]"
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                errors.append(f"{label} must contain exactly path and sha256")
            path = ref(item, label)
            relative = item.get("path") if isinstance(item, dict) else None
            if isinstance(relative, str):
                filename = relative.rsplit("/", 1)[-1]
                if (INPUT_ARTIFACT_PATH.fullmatch(relative) is None or filename.endswith(".")
                        or filename.split(".", 1)[0].upper() in RESERVED_INPUT_NAMES):
                    errors.append(f"{label}.path must be a portable evidence/inputs filename")
                folded = relative.casefold()
                indexed_collision = any(isinstance(indexed, dict) and isinstance(indexed.get("path"), str)
                                        and indexed["path"].casefold() == folded and indexed["path"] != relative
                                        for indexed in artifact_index)
                if folded in input_paths or folded in reserved_paths or indexed_collision:
                    errors.append("additional input artifact paths must be distinct and must not collide with reserved evidence")
                input_paths.add(folded)
            if path:
                size = path.stat().st_size
                total_bytes += size
                if size > MAX_INPUT_BYTES:
                    errors.append(f"{label} exceeds the 8 MiB input file limit")
        if total_bytes > MAX_TOTAL_INPUT_BYTES:
            errors.append("main and additional input artifacts exceed the 64 MiB combined limit")
        if stack_ref is None or not isinstance(workload_identity, dict):
            errors.append("additional input artifacts require an indexed stack-identity workload object")
        elif not same_json(workload_identity.get("input_artifacts"), additional):
            errors.append("stack-identity workload input_artifacts do not match experiment design")
        if manifest_ref is not None or (bundle / "executable-manifest.json").exists():
            if manifest_ref is None:
                errors.append("additional input artifacts require exactly one indexed executable-manifest.json")
            if (not isinstance(original_manifest, dict)
                    or original_manifest.get("schemaVersion") != "piq.finite-experiment-executable.v1"
                    or not isinstance(manifest_files, list)):
                errors.append("additional input artifacts require canonical original executable manifest files")
            else:
                if not same_json(manifest_inputs, additional):
                    errors.append("original executable manifest input artifacts do not match experiment design")
                for expected in (experiment.get("design"), evidence.get("input"), stack_ref, *additional):
                    if expected is not None and sum(same_json(item, expected) for item in manifest_files) != 1:
                        errors.append("original executable manifest files must bind design, main input, stack identity and every additional input exactly once")
                plan = original_manifest.get("finiteExperiment")
                plan_stack = plan.get("stack") if isinstance(plan, dict) else None
                workload_layers = [item for item in plan_stack if isinstance(item, dict) and item.get("layer") == "workload"] if isinstance(plan_stack, list) else []
                try:
                    workload_hash = ingestion_metadata_sha256(workload_identity) if isinstance(workload_identity, dict) else None
                except (TypeError, ValueError, UnicodeError):
                    workload_hash = None
                if (len(workload_layers) != 1 or not isinstance(workload_identity, dict)
                        or workload_layers[0].get("identitySha256") != workload_hash):
                    errors.append("original executable manifest workload identity does not bind additional input artifacts")

    attempts = data.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append("attempts must be a nonempty list")
        attempts = []
    ids: dict[str, dict] = {}
    raw_results: dict[str, dict] = {}
    raw_paths: set[str] = set()
    succeeded_run_ids = []
    for i, item in enumerate(attempts):
        item = obj(item, f"attempts[{i}]")
        required = {"id", "run_id", "role", "status", "accepted", "raw_result", "failure_reason"}
        if required - set(item):
            errors.append(f"attempts[{i}] missing fields: " + ", ".join(sorted(required - set(item))))
        for field in ("id", "run_id"):
            string(item.get(field), f"attempts[{i}].{field}")
        attempt_id, run_id = item.get("id"), item.get("run_id")
        if isinstance(attempt_id, str):
            if attempt_id in ids:
                errors.append("attempt IDs must be distinct")
            ids[attempt_id] = item
        if item.get("role") not in ("baseline", "candidate"):
            errors.append(f"attempts[{i}].role must be baseline or candidate")
        if item.get("status") not in ("succeeded", "failed"):
            errors.append(f"attempts[{i}].status must be succeeded or failed")
        if type(item.get("accepted")) is not bool:
            errors.append(f"attempts[{i}].accepted must be a boolean")
        if item.get("status") == "failed":
            string(item.get("failure_reason"), f"attempts[{i}].failure_reason")
            if item.get("accepted") is not False:
                errors.append(f"attempts[{i}] failed attempts cannot be accepted")
        elif item.get("status") == "succeeded":
            if run_id not in succeeded_run_ids:
                succeeded_run_ids.append(run_id)
            if item.get("failure_reason") is not None:
                errors.append(f"attempts[{i}] succeeded attempt failure_reason must be null")
        raw = ref(item.get("raw_result"), f"attempts[{i}].raw_result")
        if raw:
            if str(raw) in raw_paths:
                errors.append("attempt raw-result paths must be distinct")
            raw_paths.add(str(raw))
            raw_data = read_json(raw, f"attempts[{i}].raw_result", errors)
            bindings = {"experiment_id": experiment_id, "attempt_id": attempt_id,
                        "run_id": run_id, "role": item.get("role"),
                        "design_sha256": obj(experiment.get("design"), "experiment.design").get("sha256"),
                        "input_sha256": obj(evidence.get("input"), "workload_evidence.input").get("sha256")}
            if isinstance(raw_data, dict):
                if isinstance(attempt_id, str):
                    raw_results[attempt_id] = raw_data
                for field, expected in bindings.items():
                    if raw_data.get(field) != expected:
                        errors.append(f"attempts[{i}].raw_result {field} does not match envelope")
            elif raw_data is not None:
                errors.append(f"attempts[{i}].raw_result must be a JSON object")

    comparison = obj(data.get("comparison"), "comparison")
    disposition = comparison.get("scientific_disposition")
    if disposition not in ("DIAGNOSTIC_ONLY", "INCONCLUSIVE", "REJECTED"):
        errors.append("comparison.scientific_disposition must be DIAGNOSTIC_ONLY, INCONCLUSIVE, or REJECTED; QUALIFIED is unsupported by v1")
    string(comparison.get("reason"), "comparison.reason")
    for role in ("baseline", "candidate"):
        linked = strings(comparison.get(f"{role}_attempt_ids"), f"comparison.{role}_attempt_ids")
        actual = [item.get("id") for item in attempts if isinstance(item, dict) and item.get("role") == role]
        if linked != actual:
            errors.append(f"comparison.{role}_attempt_ids must exactly match all {role} attempts in envelope order")

    correctness = obj(data.get("correctness"), "correctness")
    if type(correctness.get("accepted")) is not bool:
        errors.append("correctness.accepted must be a boolean")
    correctness_path = ref(correctness, "correctness")
    if correctness.get("accepted") is False and any(isinstance(item, dict) and item.get("accepted") is True for item in attempts):
        errors.append("accepted attempts require accepted correctness evidence")
    # Common outputs bind experiment/run identity; CPU plans bind the experiment too.
    json_bindings = [(correctness_path, "correctness"), (evidence_paths.get("output"), "workload_evidence.output")]
    if family == "cpu_data_query":
        json_bindings.append((evidence_paths.get("query_plan"), "workload_evidence.query_plan"))
    correctness_payload = None
    for path, label in json_bindings:
        payload = read_json(path, label, errors) if path else None
        if isinstance(payload, dict):
            if payload.get("experiment_id") != experiment_id:
                errors.append(f"{label} experiment_id does not match envelope")
            if label == "workload_evidence.output" and payload.get("run_ids") != succeeded_run_ids:
                errors.append(f"{label} run_ids must exactly match deduplicated succeeded run IDs in first-occurrence order")
            if label == "correctness" and payload.get("accepted") is not correctness.get("accepted"):
                errors.append("correctness accepted does not match retained evidence")
            if label == "correctness":
                correctness_payload = payload
        elif path and payload is not None:
            errors.append(f"{label} must be a JSON object")

    # New finite-workload exports retain the producer's formal report. Legacy
    # diagnostic v1 evidence remains readable; neither shape qualifies a claim.
    declared_config = design.get("correctness") if isinstance(design, dict) else None
    check_present = isinstance(correctness_payload, dict) and "check" in correctness_payload
    formal_roles = CORRECTNESS_ROLES & set(evidence)
    if declared_config is not None or check_present or formal_roles:
        config_hash = correctness_configuration(declared_config, errors)
        check = correctness_payload.get("check") if isinstance(correctness_payload, dict) else None
        if check is None and not formal_roles and correctness.get("accepted") is False:
            # A failed attempt may leave no pair of oracle operands or report.
            # Keep that retained failure; it cannot accept any attempt above.
            if not any(isinstance(item, dict) and item.get("status") == "failed" for item in attempts):
                errors.append("formal correctness without a check requires a retained failed attempt")
        else:
            check = obj(check, "correctness.check")
            if set(check) != {"configuration", "reference", "candidate", "report", "verification"}:
                errors.append("correctness.check must contain exactly configuration, reference, candidate, report and verification")
            if check.get("configuration") != declared_config:
                errors.append("correctness.check.configuration does not match experiment design correctness")
            if check.get("verification") != "producer-asserted":
                errors.append("correctness.check.verification must equal producer-asserted; independent verification is not established")
            for key, role in (("reference", "oracle_reference"), ("candidate", "oracle_candidate"), ("report", "correctness_report")):
                ref(check.get(key), f"correctness.check.{key}")
                if role not in evidence or evidence[role] != check.get(key):
                    errors.append(f"workload_evidence.{role} must exactly match correctness.check.{key}")
            report_path = evidence_paths.get("correctness_report")
            report = read_json(report_path, "correctness report", errors) if report_path else None
            if isinstance(report, dict):
                if set(report) != {"schemaVersion", "oracle", "status", "reasonCode", "comparedValues", "configurationSha256", "referenceSha256", "candidateSha256"}:
                    errors.append("correctness report fields do not match piq.finite-correctness-report.v1")
                bindings = {
                    "schemaVersion": "piq.finite-correctness-report.v1",
                    "oracle": declared_config.get("oracle") if isinstance(declared_config, dict) else None,
                    "configurationSha256": config_hash,
                    "referenceSha256": check.get("reference", {}).get("sha256") if isinstance(check.get("reference"), dict) else None,
                    "candidateSha256": check.get("candidate", {}).get("sha256") if isinstance(check.get("candidate"), dict) else None,
                }
                for field, expected in bindings.items():
                    if expected is None or report.get(field) != expected:
                        errors.append(f"correctness report {field} does not match retained check")
                if report.get("status") not in ("pass", "fail"):
                    errors.append("correctness report status must be pass or fail")
                if (report.get("status") == "pass") is not correctness.get("accepted"):
                    errors.append("correctness report status does not match accepted")
                reason = report.get("reasonCode")
                if not isinstance(reason, str) or re.fullmatch(r"[a-z][a-z-]{0,127}", reason) is None:
                    errors.append("correctness report reasonCode must be a bounded lowercase reason code")
                if type(report.get("comparedValues")) is not int or report["comparedValues"] < 0:
                    errors.append("correctness report comparedValues must be a nonnegative integer")
            elif report_path:
                errors.append("correctness report must be a JSON object")

    if "execution" in evidence or (isinstance(design, dict) and ("resources" in design or "arms" in design)):
        implicit_cpu = "execution" not in evidence and implicit_single_cpu(design, raw_results)
        if "execution" not in evidence and not implicit_cpu:
            errors.append("workload_evidence.execution is required for the retained resource/arm design")
        execution_path = evidence_paths.get("execution")
        if implicit_cpu:
            # A validation view of already retained bytes, never a new rank record.
            execution = {"experiment_id": experiment_id, "run_id": attempts[0].get("run_id") if attempts else None,
                         "resources": design["resources"], "attempts": [
                             {"attempt_id": key, "arm_id": raw_results.get(key, {}).get("native_attempt", {}).get("armId"),
                              "expected_process_count": 1, "rank_census": []} for key in ids]}
        else:
            execution = read_json(execution_path, "execution evidence", errors) if execution_path else None
        if not isinstance(execution, dict):
            errors.append("execution evidence must be a JSON object")
            execution = {}
        if execution.get("experiment_id") != experiment_id:
            errors.append("execution experiment_id does not match envelope")
        if any(isinstance(item, dict) and item.get("run_id") != execution.get("run_id") for item in attempts):
            errors.append("execution run_id must match every retained attempt")
        resources = obj(execution.get("resources"), "execution.resources")
        requested_resources = {key: value for key, value in resources.items() if key != "assignmentEvidence"}
        if not isinstance(design, dict) or not same_json(requested_resources, design.get("resources")):
            errors.append("execution.resources does not match experiment design requested resources")
        for field, default, minimum in (("processCount", 1, 1), ("nodeCount", 1, 1), ("acceleratorCount", 0, 0)):
            number = resources.get(field, default)
            if type(number) is not int or number < minimum:
                errors.append(f"execution.resources.{field} must be an integer >= {minimum}")
        node_count = resources.get("nodeCount", 1)
        if resources.get("executor") not in ("local-process", "assigned-gpu"):
            errors.append("execution.resources.executor must be local-process or assigned-gpu")
        gpu_requested = resources.get("executor") == "assigned-gpu" or (
            type(resources.get("acceleratorCount")) is int and resources["acceleratorCount"] > 0)
        gpu_observation = None
        gpu_ranks_observed = any(
            item.get("status") == "succeeded" or bool(raw_results.get(key, {}).get("native_attempt", {}).get("rankCensus"))
            for key, item in ids.items() if isinstance(raw_results.get(key, {}).get("native_attempt", {}), dict)
        )
        if gpu_requested:
            if (resources.get("executor"), resources.get("acceleratorCount"), node_count, resources.get("processCount", 1)) != ("assigned-gpu", 1, 1, 1):
                errors.append("GPU execution supports one assigned GPU, one node and one process")
            for field, lower, upper in (("cpuCount", 1, 64), ("memoryLimitMiB", 16, 262144), ("timeoutSeconds", 1, 30)):
                value = resources.get(field)
                if type(value) is not int or not lower <= value <= upper:
                    errors.append(f"GPU execution.resources.{field} exceeds its finite bound")
            if "accelerator" not in names:
                errors.append("GPU execution requires an accelerator stack layer")
            assignment = resources.get("assignmentEvidence")
            if assignment is None and gpu_ranks_observed:
                errors.append("observed GPU execution requires retained Slurm assignmentEvidence")
            if assignment is not None:
                assignment = obj(assignment, "execution.resources.assignmentEvidence")
                assignment_path = ref({"path": assignment.get("artifactName"), "sha256": assignment.get("sha256")}, "execution.resources.assignmentEvidence")
                observation = assignment.get("observation")
                if not isinstance(observation, dict) or "leaseContext" in assignment:
                    errors.append("GPU assignment requires a Slurm observation; serving leaseContext is not finite execution evidence")
                    observation = {}
                gpu_observation = observation
                retained = read_json(assignment_path, "GPU assignment observation", errors) if assignment_path else None
                if not same_json(retained, observation):
                    errors.append("GPU assignment evidence must match the retained Slurm observation")
                try:
                    if ingestion_metadata_sha256(observation, newline=True) != assignment.get("sha256"):
                        errors.append("GPU assignment bytes must retain ingestion identity JSON plus newline")
                except (ValueError, UnicodeError, TypeError):
                    errors.append("GPU assignment observation must contain finite JSON metadata")
                manifest_refs = [item for item in data.get("artifacts", []) if isinstance(item, dict) and item.get("path") == "executable-manifest.json"] if isinstance(data.get("artifacts"), list) else []
                manifest_path = ref(manifest_refs[0], "GPU original executable manifest") if len(manifest_refs) == 1 else None
                manifest = read_json(manifest_path, "GPU original executable manifest", errors) if manifest_path else None
                if not isinstance(manifest, dict) or manifest.get("runId") != execution.get("run_id"):
                    errors.append("GPU Slurm observation requires the indexed original executable manifest for this run")
                else:
                    try:
                        manifest_identity = ingestion_metadata_sha256(manifest, newline=True)
                    except (TypeError, ValueError, UnicodeError):
                        manifest_identity = None
                    if manifest_identity != digest(manifest_path):
                        errors.append("GPU original executable manifest must retain canonical ingestion identity JSON bytes")
                    validate_slurm_observation(observation, manifest, digest(manifest_path), resources,
                                               design if isinstance(design, dict) else {}, source_refs, evidence.get("input"), errors)
            string(resources.get("targetId"), "execution.resources.targetId")
        elif resources.get("assignmentEvidence") is not None or resources.get("targetId") is not None:
            errors.append("CPU execution cannot declare a GPU assignment or target")

        arms = design.get("arms") if isinstance(design, dict) else None
        arm_configurations: dict[str, dict] = {}
        if not isinstance(arms, list) or not arms:
            errors.append("execution requires experiment design arms")
            arms = []
        for arm in arms:
            arm = obj(arm, "experiment design arm")
            arm_id = arm.get("arm_id")
            configuration = obj(arm.get("configuration"), "experiment design arm configuration")
            string(arm_id, "experiment design arm arm_id")
            if isinstance(arm_id, str):
                if arm_id in arm_configurations:
                    errors.append("experiment design arm IDs must be distinct")
                arm_configurations[arm_id] = configuration
            scalar_configuration = bool(configuration) and not any(
                value is not None and type(value) not in (str, int, bool)
                for value in configuration.values()
            )
            if not scalar_configuration:
                errors.append("experiment design arm configuration must contain scalar JSON string, integer, boolean or null values")
            # Arm hashes use PIQ's frozen ingestion identity domain, which
            # escapes non-ASCII strings; correctness configuration hashes do not.
            expected = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest() if scalar_configuration else None
            if arm.get("configuration_sha256") != expected:
                errors.append("experiment design arm configuration_sha256 does not match configuration")
        execution_attempts = execution.get("attempts")
        if not isinstance(execution_attempts, list):
            errors.append("execution.attempts must be a list")
            execution_attempts = []
        if [item.get("attempt_id") if isinstance(item, dict) else None for item in execution_attempts] != list(ids):
            errors.append("execution.attempts must exactly match envelope attempt IDs in order")
        gpu_observation_pid_matched = False
        for item in execution_attempts:
            item = obj(item, "execution attempt")
            attempt_id, arm_id = item.get("attempt_id"), item.get("arm_id")
            label = f"execution attempt {attempt_id}"
            attempt = ids.get(attempt_id, {}) if isinstance(attempt_id, str) else {}
            raw = raw_results.get(attempt_id, {}) if isinstance(attempt_id, str) else {}
            native = obj(raw.get("native_attempt"), f"{label} native_attempt")
            if attempt.get("status") == "succeeded" and native.get("state") != "completed":
                errors.append(f"{label} succeeded status requires a completed native attempt")
            if attempt.get("status") == "failed" and native.get("state") not in ("failed", "timed-out", "not-started"):
                errors.append(f"{label} failed status does not match native attempt state")
            if implicit_cpu and attempt.get("status") == "succeeded":
                if native.get("attemptId") != attempt_id:
                    errors.append(f"{label} single CPU attemptId differs from the retained attempt")
                if type(native.get("childPid")) is not int or native["childPid"] <= 0:
                    errors.append(f"{label} single CPU attempt requires a positive childPid")
                if type(native.get("exitCode")) is not int or native["exitCode"] != 0:
                    errors.append(f"{label} single CPU completed attempt requires exitCode 0")
                output = obj(native.get("output"), f"{label} single CPU output")
                ref({"path": output.get("artifactName"), "sha256": output.get("sha256")}, f"{label} single CPU output")
                elapsed = native.get("elapsedSeconds")
                if not isinstance(elapsed, str) or len(elapsed) > 64 or DECIMAL.fullmatch(elapsed) is None:
                    errors.append(f"{label} single CPU elapsedSeconds must be a nonnegative decimal string")
                try:
                    start, end = (datetime.fromisoformat(native[field].replace("Z", "+00:00")) for field in ("startedAt", "completedAt"))
                    if start.utcoffset() != timezone.utc.utcoffset(start) or end.utcoffset() != timezone.utc.utcoffset(end) or end < start:
                        raise ValueError("invalid native window")
                    outer_start, outer_end = (datetime.fromisoformat(objective["window"][field].replace("Z", "+00:00")) for field in ("start", "end"))
                    if not outer_start <= start <= end <= outer_end:
                        raise ValueError("native window escapes objective window")
                except (KeyError, TypeError, AttributeError, ValueError):
                    errors.append(f"{label} single CPU attempt requires an ordered UTC window")
            configuration = arm_configurations.get(arm_id) if isinstance(arm_id, str) else None
            if configuration is None or native.get("armId") != arm_id or not same_json(raw.get("arm_configuration"), configuration):
                errors.append(f"{label} arm identity/configuration does not match retained design and raw result")
            expected_count = (configuration or {}).get("processCount", resources.get("processCount", 1))
            count = item.get("expected_process_count")
            if gpu_requested and expected_count != 1:
                errors.append(f"{label} GPU arm requires exactly one process")
            if type(count) is not int or not 1 <= count <= 64 or type(expected_count) is not int or count != expected_count:
                errors.append(f"{label} expected_process_count does not match declared arm/resources")
            census = item.get("rank_census")
            if not isinstance(census, list):
                errors.append(f"{label} rank_census must be a list")
                census = []
            if not same_json(census, native.get("rankCensus", [])):
                errors.append(f"{label} rank_census does not match retained native attempt")
            ranks, hosts, pids, local_ranks, rank_outputs, devices = [], set(), set(), set(), set(), set()
            for rank in census:
                rank = obj(rank, f"{label} rank")
                rank_number, local_rank = rank.get("rank"), rank.get("localRank")
                host, pid = rank.get("host"), rank.get("processId")
                if type(rank_number) is not int or not 0 <= rank_number <= 63:
                    errors.append(f"{label} rank must be a nonnegative integer")
                else:
                    ranks.append(rank_number)
                if type(local_rank) is not int or not 0 <= local_rank <= 63:
                    errors.append(f"{label} localRank must be a nonnegative integer")
                string(host, f"{label} host")
                if type(pid) is not int or pid <= 0:
                    errors.append(f"{label} processId must be a positive integer")
                elif isinstance(host, str):
                    if (host, pid) in pids:
                        errors.append(f"{label} simultaneous ranks require distinct process IDs per host")
                    pids.add((host, pid))
                if isinstance(host, str):
                    hosts.add(host)
                    if type(local_rank) is int:
                        if (host, local_rank) in local_ranks:
                            errors.append(f"{label} localRank must be distinct per host")
                        local_ranks.add((host, local_rank))
                device, uuid = rank.get("deviceType"), rank.get("deviceUuid")
                if device not in ("cpu", "cuda"):
                    errors.append(f"{label} deviceType must be cpu or cuda")
                if gpu_requested and device != "cuda":
                    errors.append(f"{label} GPU execution cannot substitute a CPU rank")
                if device == "cuda" and (not gpu_requested or not isinstance(uuid, str) or not uuid.strip() or uuid.lower().startswith("cpu")):
                    errors.append(f"{label} CUDA rank requires GPU resources and a device UUID")
                if device == "cuda" and isinstance(uuid, str):
                    devices.add(uuid)
                    if gpu_observation is not None:
                        if host != gpu_observation.get("nodeName") or uuid != gpu_observation.get("deviceUuid"):
                            errors.append(f"{label} GPU rank host/device does not match retained Slurm observation")
                        if type(pid) is int and pid == gpu_observation.get("processId"):
                            gpu_observation_pid_matched = True
                if device == "cpu" and uuid is not None:
                    errors.append(f"{label} CPU rank deviceUuid must be null")
                output = obj(rank.get("output"), f"{label} rank output")
                ref({"path": output.get("artifactName"), "sha256": output.get("sha256")}, f"{label} rank output")
                output_name = output.get("artifactName")
                if isinstance(output_name, str):
                    if output_name in rank_outputs:
                        errors.append(f"{label} rank output artifacts must be distinct")
                    rank_outputs.add(output_name)
            if not implicit_cpu and attempt.get("status") == "succeeded" and (type(count) is not int or not 1 <= count <= 64 or sorted(ranks) != list(range(count))):
                errors.append(f"{label} completed attempt requires an exact complete rank census")
            if type(node_count) is int and len(hosts) > node_count:
                errors.append(f"{label} rank hosts exceed declared nodeCount")
            accelerator_count = resources.get("acceleratorCount", 0)
            if type(accelerator_count) is int and len(devices) > accelerator_count:
                errors.append(f"{label} rank devices exceed declared acceleratorCount")

        if gpu_observation is not None and gpu_ranks_observed and not gpu_observation_pid_matched:
            errors.append("GPU Slurm observation processId must match an observed native rank")

    referenced = dict(refs)
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a nonempty complete artifact index")
        artifacts = []
    indexed: dict[str, str] = {}
    for i, artifact in enumerate(artifacts):
        ref(artifact, f"artifacts[{i}]")
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
            path = artifact["path"]
            if path in indexed:
                errors.append("artifacts paths must be distinct")
            indexed[path] = artifact.get("sha256")
    if any(indexed.get(path) != expected for path, expected in referenced.items()):
        errors.append("artifacts must include every referenced artifact with matching hashes")
    return errors


def import_bundle(source: Path, destination: Path) -> Path:
    """Validate, copy only indexed evidence, revalidate, then atomically retain it."""
    source = Path(source)
    envelope_bytes = (source / ENVELOPE).read_bytes()
    errors = validate_bundle(source)
    if errors:
        raise ValueError("\n".join(errors))
    if (source / ENVELOPE).read_bytes() != envelope_bytes:
        raise ValueError("source envelope changed during validation; retain a stable export")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists; preserving it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".piq-import-", dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        (staged / ENVELOPE).write_bytes(envelope_bytes)
        data = json.loads(envelope_bytes)
        for artifact in data["artifacts"]:
            target = staged / artifact["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / artifact["path"], target)
        errors = validate_bundle(staged)
        if errors:
            raise ValueError("copied bundle failed validation:\n" + "\n".join(errors))
        receipt = {"schema_version": 1, "state": "diagnostic_imported", "publishable": False,
                   "experiment_id": data["experiment"]["id"], "envelope_sha256": digest(staged / ENVELOPE),
                   "imported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   "qualification": "not_established"}
        (staged / "import-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Recheck before the atomic rename; never merge an existing owner directory.
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"destination already exists; preserving it: {destination}")
        staged.rename(destination)
    return destination



# Indexed originals are a separate bounded data-transport contract. They never
# enter finite scientific validation or grant execution/qualification authority.
ORIGINALS_TOTAL_BYTES = 128 * 1024 * 1024
ORIGINALS_FILE_BYTES = 64 * 1024 * 1024
ORIGINALS_INDEX_BYTES = 8 * 1024 * 1024
ORIGINALS_FILES = 4096
ORIGINALS_RECEIPT = "originals-import-receipt.json"


def _originals_name(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("bounded portable relative originals path required")
    parts = value.split("/")
    reserved = {"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", p)
           or p.endswith(".") or p.split(".")[0].casefold() in reserved for p in parts):
        raise ValueError("portable relative originals path required")
    return parts


def _originals_read(root, name, maximum):
    import os
    import stat
    root = Path(root).absolute()
    parts = _originals_name(name)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in [*root.parts[1:], *parts[:-1]]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
                raise ValueError("bounded regular originals file required")
            chunks, count = [], 0
            while True:
                block = os.read(fd, min(1024 * 1024, maximum + 1 - count))
                if not block:
                    break
                chunks.append(block)
                count += len(block)
                if count > maximum:
                    raise ValueError("originals file exceeds byte limit")
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or count != before.st_size:
                raise ValueError("originals file changed during read")
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(descriptor)


def _originals_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate originals JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_originals(source, index_name):
    """Verify indexed retained bytes only, never native identity or science."""
    import os
    import stat
    source = Path(source).absolute()
    _originals_name(index_name)
    if index_name == ORIGINALS_RECEIPT:
        raise ValueError("index collides with reserved import receipt")
    raw_index = _originals_read(source, index_name, ORIGINALS_INDEX_BYTES)
    index = _originals_json(raw_index)
    if not isinstance(index, dict) or not isinstance(index.get("kind"), str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", index["kind"]):
        raise ValueError("explicit retained-originals source kind required")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= ORIGINALS_FILES:
        raise ValueError("indexed originals count must be 1..4096")
    names, folded, total = {}, set(), 0
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("originals artifact object required")
        name = item.get("path")
        _originals_name(name)
        if name in (index_name, ORIGINALS_RECEIPT) or name.casefold() in folded:
            raise ValueError("duplicate/colliding originals artifact path")
        size = item.get("sizeBytes")
        if type(size) is not int or not 0 <= size <= ORIGINALS_FILE_BYTES or not isinstance(
                item.get("sha256"), str) or not SHA256.fullmatch(item["sha256"]):
            raise ValueError("explicit bounded originals size and sha256 required")
        total += size
        if total > ORIGINALS_TOTAL_BYTES:
            raise ValueError("originals indexed bytes exceed128MiB")
        names[name] = item
        folded.add(name.casefold())
    observed, directory_count = set(), 0
    for base, directories, files in os.walk(source, followlinks=False):
        directory_count += 1
        if directory_count > ORIGINALS_FILES + 1:
            raise ValueError("originals directory count exceeds bound")
        for name in directories:
            mode = (Path(base) / name).lstat().st_mode
            if not stat.S_ISDIR(mode):
                raise ValueError("originals symlink/special directory refused")
        for name in files:
            path = Path(base) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("originals symlink/special file refused")
            observed.add(path.relative_to(source).as_posix())
            if len(observed) > ORIGINALS_FILES + 2:
                raise ValueError("originals file census exceeds bound")
    allowed = {index_name, *names}
    if ORIGINALS_RECEIPT in observed:
        receipt = _originals_json(_originals_read(source, ORIGINALS_RECEIPT, 65536))
        expected = {"state": "indexed_originals_imported", "source_kind": index["kind"],
                    "index_sha256": hashlib.sha256(raw_index).hexdigest(),
                    "scientifically_verified": False, "qualification": "not_established",
                    "publishable": False, "indexed_files": len(names), "indexed_bytes": total}
        if not isinstance(receipt, dict) or any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v
                                                for k, v in expected.items()):
            raise ValueError("originals import receipt does not match data-only scope")
        allowed.add(ORIGINALS_RECEIPT)
    if observed != allowed:
        raise ValueError("originals exact file set differs from declared index")
    for name, item in names.items():
        data = _originals_read(source, name, ORIGINALS_FILE_BYTES)
        if len(data) != item["sizeBytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("originals indexed hash/size drift: " + name)
    return {"index_bytes": raw_index, "source_kind": index["kind"],
            "index_sha256": hashlib.sha256(raw_index).hexdigest(),
            "artifacts": names, "indexed_files": len(names), "indexed_bytes": total}


def import_originals(source, destination, index_name):
    """Copy validated raw transport data atomically; keep partial staging on failure."""
    source, destination = Path(source).absolute(), Path(destination).absolute()
    before = validate_originals(source, index_name)
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; preserving it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".piq-originals-", dir=destination.parent))
    bundle = stage / "bundle"
    bundle.mkdir()
    try:
        for name in [index_name, *before["artifacts"]]:
            data = _originals_read(source, name, ORIGINALS_INDEX_BYTES if name == index_name else ORIGINALS_FILE_BYTES)
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
                stream.flush()
                import os
                os.fsync(stream.fileno())
        copied = validate_originals(bundle, index_name)
        after = validate_originals(source, index_name)
        if copied != before or after != before:
            raise ValueError("originals changed during import")
        receipt = {"state": "indexed_originals_imported", "source_kind": before["source_kind"],
                   "index_sha256": before["index_sha256"], "scientifically_verified": False,
                   "qualification": "not_established", "publishable": False,
                   "indexed_files": before["indexed_files"], "indexed_bytes": before["indexed_bytes"],
                   "imported_at": datetime.now(timezone.utc).isoformat()}
        (bundle / ORIGINALS_RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        validate_originals(bundle, index_name)
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination appeared during import; preserving it")
        bundle.rename(destination)
        stage.rmdir()
        return destination
    except BaseException as exc:
        (stage / "PARTIAL-IMPORT.json").write_text(json.dumps({
            "state": "PARTIAL", "source": str(source), "destination": str(destination),
            "index_sha256": before["index_sha256"], "error": type(exc).__name__ + ": " + str(exc),
            "scientifically_verified": False, "qualification": "not_established",
            "publishable": False}, indent=2) + "\n")
        raise ValueError(f"originals import failed; partial evidence retained at {stage}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("bundle", type=Path)
    ingest = commands.add_parser("import")
    ingest.add_argument("bundle", type=Path)
    ingest.add_argument("destination", type=Path)
    originals_check = commands.add_parser("check-originals")
    originals_check.add_argument("bundle", type=Path)
    originals_check.add_argument("--index", required=True)
    originals_import = commands.add_parser("import-originals")
    originals_import.add_argument("bundle", type=Path)
    originals_import.add_argument("destination", type=Path)
    originals_import.add_argument("--index", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            errors = validate_bundle(args.bundle)
            if errors:
                raise ValueError("\n".join(errors))
        elif args.command == "import":
            import_bundle(args.bundle, args.destination)
        elif args.command == "check-originals":
            validate_originals(args.bundle, args.index)
        else:
            import_originals(args.bundle, args.destination, args.index)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.command in ("check-originals", "import-originals"):
        print("PASS: indexed retained original bytes only; scientifically_verified=false; qualification not established; publishable=false")
    else:
        print("PASS: retained PIQ diagnostic evidence; qualification and publication remain unestablished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
