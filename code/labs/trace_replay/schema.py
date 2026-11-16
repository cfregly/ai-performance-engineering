"""Strict schemas for captured operator values and their source records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from labs.trace_replay.storage import read_tensor, safe_file, validate_descriptor

PLAN_SCHEMA = "native-operator-capture-plan.v1"
BUNDLE_SCHEMA = "native-operator-trace.v1"
OPERATIONS = {"prefix_scan", "online_softmax", "nvfp4.quantize", "nvfp4.add_rmsnorm", "nvfp4.silu_mul"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def identifier(value, location="event_id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{location} must contain letters, digits, underscores or hyphens")
    return value


def exact_fields(value, required, location, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"{location} fields do not match the schema")


def validate_parameters(operation, parameters):
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported operator: {operation}")
    keys = {"numel"} if operation == "prefix_scan" else {"rows", "cols"}
    if operation.startswith("nvfp4."):
        keys.add("batch")
    exact_fields(parameters, keys, "parameters")
    if any(type(value) is not int or value <= 0 for value in parameters.values()):
        raise ValueError("operator dimensions must be positive integers")
    if operation.startswith("nvfp4.") and (parameters["cols"] % 16 or parameters["cols"] > 16384):
        raise ValueError("NVFP4 columns must be divisible by 16 and at most 16384")


def validate_source(source):
    exact_fields(source, {"kind", "description"}, "source", {"caller", "event_id", "artifact_sha256"})
    if source["kind"] not in {"generated_by_recipe", "observed_call"}:
        raise ValueError("source.kind must be generated_by_recipe or observed_call")
    if any(not isinstance(value, str) or not value.strip() for value in source.values()):
        raise ValueError("source values must be nonempty strings")
    if "artifact_sha256" in source and not re.fullmatch(r"[0-9a-f]{64}", source["artifact_sha256"]):
        raise ValueError("source artifact_sha256 must be a SHA-256 digest")


def _tensor_contract(event):
    operation, parameters = event["operation"], event["parameters"]
    if operation == "prefix_scan":
        inputs = {"x": ([parameters["numel"]], "int32")}
        outputs = {"output": ([parameters["numel"]], "int32")}
    elif operation == "online_softmax":
        shape = [parameters["rows"], parameters["cols"]]
        inputs, outputs = {"x": (shape, "float32")}, {"output": (shape, "float32")}
    else:
        batch, rows, cols = (parameters[key] for key in ("batch", "rows", "cols"))
        width = cols * (2 if operation == "nvfp4.silu_mul" else 1)
        inputs = {
            "x": ([batch, rows, width], "bfloat16"),
            "global_scale": ([batch], "float32"),
            "valid_rows": ([batch], "int32"),
        }
        scales = [batch, rows, cols // 16]
        if operation == "nvfp4.silu_mul":
            scales = [batch, (rows + 127) // 128 * 128, (cols // 16 + 3) // 4 * 4]
        outputs = {"packed": ([batch, rows, cols // 2], "uint8"), "scales": (scales, "uint8")}
        if operation == "nvfp4.add_rmsnorm":
            inputs.update(residual=([batch, rows, cols], "bfloat16"), weight=([cols], "bfloat16"))
            outputs["residual"] = ([batch, rows, cols], "bfloat16")
    for group, contracts in (("inputs", inputs), ("outputs", outputs)):
        if set(event[group]) != set(contracts):
            raise ValueError(f"event {group} names do not match the operator contract")
        for name, (shape, dtype) in contracts.items():
            descriptor = event[group][name]
            if descriptor["shape"] != shape or descriptor["dtype"] != dtype:
                raise ValueError(f"event {group}.{name} shape or dtype differs from the operator contract")


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    exact_fields(plan, {"schema_version", "events"}, "capture plan")
    if plan["schema_version"] != PLAN_SCHEMA:
        raise ValueError(f"capture plan schema_version must be {PLAN_SCHEMA}")
    if not isinstance(plan["events"], list) or not plan["events"]:
        raise ValueError("capture plan must contain events")
    seen = set()
    for event in plan["events"]:
        exact_fields(event, {"event_id", "operation", "parameters", "seed", "source"}, "plan event")
        event_id = identifier(event["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        validate_parameters(event["operation"], event["parameters"])
        validate_source(event["source"])
        if event["source"]["kind"] != "generated_by_recipe":
            raise ValueError("capture plans generate inputs and must declare generated_by_recipe")
        if type(event["seed"]) is not int or not 0 <= event["seed"] < 2**63:
            raise ValueError("capture seed must be an integer in [0, 2**63)")
    return plan


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict
    sha256: str

    @property
    def events(self):
        return tuple(self.manifest["events"])

    def event(self, event_id):
        for event in self.events:
            if event["event_id"] == event_id:
                return event
        raise ValueError(f"bundle has no event: {event_id}")


def load_bundle(path: Path) -> Bundle:
    root = Path(path).absolute()
    manifest_path = safe_file(root, "manifest.json")
    checksum_path = safe_file(root, "manifest.sha256")
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if checksum_path.read_text(encoding="ascii").strip() != digest:
        raise ValueError("bundle manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    exact_fields(
        manifest,
        {"schema_version", "created_at", "capture_device", "capture_backend", "producer", "git_commit", "plan_sha256", "source_files", "runtime", "events"},
        "bundle manifest",
    )
    if manifest["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError(f"bundle schema_version must be {BUNDLE_SCHEMA}")
    if manifest["producer"] != "ai-performance-engineering.native-operator-capture":
        raise ValueError("unsupported capture producer")
    if not isinstance(manifest["created_at"], str) or datetime.fromisoformat(manifest["created_at"]).tzinfo is None:
        raise ValueError("capture created_at must include a timezone")
    for name in ("git_commit", "plan_sha256"):
        value = manifest[name]
        width = 40 if name == "git_commit" else 64
        if value is not None and (not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{width}}}", value)):
            raise ValueError(f"capture {name} must be a hex digest or null")
    if manifest["capture_backend"] not in {"cpu_reference", "cuda_native"}:
        raise ValueError("unsupported capture backend")
    if manifest["capture_device"] not in {"cpu", "cuda"}:
        raise ValueError("unsupported capture device")
    if (manifest["capture_device"] == "cpu") != (manifest["capture_backend"] == "cpu_reference"):
        raise ValueError("capture device and backend disagree")
    exact_fields(manifest["runtime"], {"python", "torch", "cuda"}, "capture runtime")
    if any(not isinstance(manifest["runtime"][key], str) for key in ("python", "torch")):
        raise ValueError("capture runtime must declare Python and torch versions")
    if manifest["runtime"]["cuda"] is not None and not isinstance(manifest["runtime"]["cuda"], str):
        raise ValueError("capture CUDA version must be text or null")
    if not isinstance(manifest["source_files"], list) or not manifest["source_files"]:
        raise ValueError("bundle must declare implementation source hashes")
    for source in manifest["source_files"]:
        exact_fields(source, {"path", "sha256"}, "implementation source")
        if (
            not isinstance(source["path"], str)
            or Path(source["path"]).is_absolute()
            or ".." in Path(source["path"]).parts
            or not isinstance(source["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        ):
            raise ValueError("invalid implementation source path or digest")
    events = manifest["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("bundle must contain events")
    seen = set()
    for event in events:
        exact_fields(event, {"event_id", "operation", "parameters", "seed", "source", "status", "inputs", "outputs", "error", "capture_duration_s"}, "captured event")
        event_id = identifier(event["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate event_id: {event_id}")
        seen.add(event_id)
        validate_parameters(event["operation"], event["parameters"])
        validate_source(event["source"])
        if event["source"]["kind"] == "generated_by_recipe":
            if type(event["seed"]) is not int or not 0 <= event["seed"] < 2**63:
                raise ValueError("generated event seed must be an integer in [0, 2**63)")
        elif event["seed"] is not None:
            raise ValueError("observed calls must retain values without declaring a generated seed")
        if event["status"] not in {"completed", "failed", "cancelled", "skipped", "unsupported"}:
            raise ValueError("unsupported capture event status")
        if event["status"] == "completed" and event["error"] is not None:
            raise ValueError("completed events must not contain an error")
        if event["status"] != "completed" and (not isinstance(event["error"], str) or not event["error"]):
            raise ValueError("failed and cancelled events must retain an error")
        duration = event["capture_duration_s"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("capture duration must be finite and nonnegative")
        for field in ("inputs", "outputs"):
            tensors = event[field]
            if not isinstance(tensors, dict) or (event["status"] == "completed" and not tensors):
                raise ValueError(f"completed events need named {field}")
            for name, descriptor in tensors.items():
                identifier(name, "tensor name")
                validate_descriptor(descriptor)
                read_tensor(root, descriptor)
        if event["status"] == "completed":
            _tensor_contract(event)
    return Bundle(root, manifest, digest)
