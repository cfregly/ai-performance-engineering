#!/usr/bin/env python3
"""Retained-evidence qualification for one prospective CPU codec study.

This is deliberately separate from GPU receipt v1 and PIQ diagnostic v1. It
checks bytes, identities and fixed arithmetic; it cannot authenticate a producer
or independently prove execution, registration time, or reviewer independence.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

if __package__:
    from .piq_experiment_evidence import validate_bundle
else:
    from piq_experiment_evidence import validate_bundle


CRITERIA = {
    "paired_repeats": 4,
    "neutral_controls": 4,
    "minimum_each_paired_ratio": "1.10",
    "candidate_observed_max_latency_at_most_baseline": True,
    "neutral_ratio_minimum": "0.95",
    "neutral_ratio_maximum": "1.05",
    "candidate_archive_bytes_at_most_baseline_each_pair": True,
    "all_parts_exact_roundtrip": True,
    "no_failed_or_missing_attempts": True,
}
TIMED_STEPS = ["input_read", "input_hash", "compression", "archive_write",
               "archive_fsync", "decompression", "roundtrip_verification"]
DESIGN_KIND = "qualified_cpu_codec_design"
IDENTITY_FIELDS = ("input_sha256", "source_sha256s", "runtime_identifier", "hardware_identifier")
FORBIDDEN_GPU_FIELDS = {"power_limit_w", "expected_power_limit_w", "gpu_uuid", "cuda_device"}
CPU_COVERAGE = {
    "host_process_memory": "actual-capture", "cgroup": "actual-capture",
    "filesystem_block": "context-only-shared-device-counters",
    "runtime": "source-and-process-identity-metadata",
    "network": "host-context-only-workload-unexercised",
    "fabric": "unexercised", "accelerator": "unexercised",
}
HOST_CONTEXT = {"host." + name for name in ("stat", "meminfo", "vmstat", "pressure.cpu", "pressure.memory", "pressure.io", "diskstats", "net.dev", "mountinfo")} | {"filesystem.statvfs"}
CGROUP_CONTEXT = {"cgroup." + name for name in ("cpu.stat", "cpu.max", "cpuset.cpus.effective", "memory.current", "memory.peak", "memory.max", "memory.events", "memory.stat", "io.stat", "io.max", "pids.current", "pids.max", "cgroup.procs")}
CPU_PROCESS_METRICS = {
    "cpu.user_ticks": "ticks", "cpu.system_ticks": "ticks", "cpu.processor": "cpu", "thread_count": "count",
    "memory.rss_kibibytes": "kibibytes", "context_switches.voluntary": "count", "context_switches.involuntary": "count",
    "io.read_bytes": "bytes", "io.write_bytes": "bytes", "io.cancelled_write_bytes": "bytes",
    "scheduling.runtime_nanoseconds": "nanoseconds", "scheduling.runqueue_nanoseconds": "nanoseconds", "scheduling.timeslices": "count",
}


def _native_sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _proc_stat(data: bytes) -> tuple[int, int, int]:
    close = data.rfind(b")")
    fields = data[close + 2:].split()
    if close < 2 or len(fields) < 20:
        raise ValueError("telemetry process stat is incomplete")
    return int(data.split(b" ", 1)[0]), int(fields[1]), int(fields[19])


def _native_files(data: bytes) -> dict[str, bytes]:
    prefix = b"PIQ-NATIVE-FILES-V1\n"
    if not data.startswith(prefix):
        raise ValueError("telemetry normalized sample lacks canonical raw framing")
    position, files = len(prefix), {}
    while position < len(data):
        if position + 4 > len(data):
            raise ValueError("telemetry native metadata length is truncated")
        size = int.from_bytes(data[position:position + 4], "big")
        position += 4
        if not 0 < size <= 65536 or position + size + 8 > len(data):
            raise ValueError("telemetry native metadata is truncated")
        metadata = json.loads(data[position:position + size])
        position += size
        payload_size = int.from_bytes(data[position:position + 8], "big")
        position += 8
        if payload_size != metadata["sizeBytes"] or position + payload_size > len(data):
            raise ValueError("telemetry native payload is truncated")
        payload = data[position:position + payload_size]
        position += payload_size
        name = metadata["sourceId"]
        if name in files or metadata["sha256"] != _native_sha(payload):
            raise ValueError("telemetry native payload digest/identity mismatch")
        files[name] = payload
    expected = {"process.stat", "process.status", "process.io", "process.schedstat", "cgroup.cpu.pressure", "cgroup.memory.pressure", "cgroup.io.pressure"}
    if set(files) != expected:
        raise ValueError("telemetry normalized sample omits a required process/pressure source")
    if any(not payload for payload in files.values()):
        raise ValueError("telemetry normalized process/pressure source is empty")
    return files


def _telemetry_metadata(raw: dict, observations: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    summary = observations["telemetry"]
    bundle = json.loads(raw["arm_configuration"]["telemetryBundle"])
    if (set(bundle) != {"artifactName", "sha256", "sizeBytes"}
            or bundle["artifactName"] != "inputs/finite-cpu-telemetry.zip"
            or not _positive_int(bundle["sizeBytes"]) or len(bundle["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in bundle["sha256"])
            or observations["runtime"].get("telemetry_bundle") != bundle):
        errors.append("CPU telemetry runtime/configuration bundle identity mismatch")
    if (summary.get("schema") != "piq.finite-cpu-telemetry.v1" or summary.get("source", {}).get("kind") != "host-procfs"
            or summary["source"].get("producerId") != "finite-cpu-observer"
            or summary["source"].get("producerSha256") != "sha256:" + bundle["sha256"]
            or summary["source"].get("rawSha256") is not None or type(summary["source"].get("rawSizeBytes")) is not int or summary["source"].get("rawSizeBytes") != 0):
        errors.append("CPU telemetry schema/canonical source identity mismatch")
    terminal = summary.get("terminal_coverage", {})
    if (terminal.get("state") != "complete" or terminal.get("raw_overflow") is not False or terminal.get("sample_overflow") is not False
            or type(summary.get("dropped_error_count")) is not int or summary["dropped_error_count"] != 0
            or type(summary.get("dropped_gap_count")) is not int or summary["dropped_gap_count"] != 0):
        errors.append("CPU telemetry requires complete terminal coverage without overflow or dropped records")
    if (not isinstance(summary.get("coverage"), dict) or any(summary["coverage"].get(key) != value for key, value in CPU_COVERAGE.items())
            or not isinstance(summary.get("limits"), list) or not summary["limits"]):
        errors.append("CPU telemetry must retain scoped context/unexercised coverage and limitations")
    if any(type(summary.get(key)) is not int or summary[key] < 0 for key in ("observer_wall_nanoseconds", "observer_thread_cpu_nanoseconds")):
        errors.append("CPU telemetry must retain actual collection overhead with native nanosecond units")
    if (not isinstance(summary.get("errors"), list) or not isinstance(summary.get("gaps"), list)
            or not _text(summary.get("overhead_scope")) or isinstance(summary.get("interval_seconds"), bool)
            or not isinstance(summary.get("interval_seconds"), (int, float)) or not math.isfinite(summary["interval_seconds"])
            or not 0.01 <= summary["interval_seconds"] <= 5):
        errors.append("CPU telemetry must retain explicit error/gap census, overhead scope and finite sampling interval")
    parent = raw["native_attempt"]["childPid"]
    if observations["host"]["process_id"] != parent:
        errors.append("CPU telemetry parent differs from actual runner child PID")
    children = [part[phase] for part in observations["parts"] for phase in ("compression", "decompression")]
    child_pids = [child["pid"] for child in children]
    if any(not _positive_int(pid) for pid in [parent, *child_pids]) or any(type(child.get("exit_code")) is not int or child["exit_code"] != 0 for child in children):
        errors.append("CPU telemetry owning Popen child identities/returncodes are invalid")
    identities = summary.get("registered_processes", [])
    if [identity.get("pid") for identity in identities] != [parent, *child_pids]:
        errors.append("CPU telemetry registration omits or invents an owning Popen process")
    if len({(identity.get("pid"), identity.get("starttime_ticks")) for identity in identities}) != len(identities):
        errors.append("CPU telemetry duplicates a registered PID/starttime identity")
    identity_fields = {"pid", "starttime_ticks", "cgroup_path", "cgroup_directory_identity", "kind", "host_boot_id_sha256"}
    for index, identity in enumerate(identities):
        if (set(identity) != identity_fields or not _positive_int(identity.get("starttime_ticks"))
                or identity.get("kind") != ("adapter" if index == 0 else "codec-child")
                or not _text(identity.get("cgroup_path")) or not identity["cgroup_path"].startswith("/")
                or not isinstance(identity.get("cgroup_directory_identity"), list) or len(identity["cgroup_directory_identity"]) != 2
                or any(type(value) is not int or value < 0 for value in identity["cgroup_directory_identity"])):
            errors.append("CPU telemetry process identity is malformed or introduces device identities")
    samples = summary.get("samples", [])
    if not samples or not _positive_int(summary.get("maximum_samples")) or not len(samples) <= summary["maximum_samples"] <= 256:
        errors.append("CPU telemetry normalized sample census is missing or exceeds its bound")
    for index, sample in enumerate(samples):
        identity = sample.get("process_identity")
        if type(sample.get("sequence")) is not int or sample["sequence"] != index or identity not in identities or not _positive_int(sample.get("monotonic_ns")) or _timestamp(sample.get("observed_at")) is None:
            errors.append("CPU telemetry normalized sample identity/sequence mismatch")
        if sample.get("status") not in ("sample", "missing"):
            errors.append("CPU telemetry contains partial/unknown normalized sample status")
        if sample.get("status") == "sample":
            metrics = sample.get("measurements")
            if not isinstance(metrics, list) or not metrics:
                errors.append("CPU telemetry normalized sample omits actual measurements")
            else:
                names = [metric.get("name") for metric in metrics if isinstance(metric, dict)]
                if len(names) != len(set(names)) or not {"host.process." + name for name in CPU_PROCESS_METRICS} <= set(names):
                    errors.append("CPU telemetry normalized sample omits canonical process measurements")
                for metric in metrics:
                    if (not isinstance(metric, dict) or not _text(metric.get("name")) or not metric["name"].startswith(("host.process.", "host.cgroup."))
                            or isinstance(metric.get("value"), bool) or not isinstance(metric.get("value"), (int, float)) or not math.isfinite(metric["value"])
                            or set(metric.get("labels", {})) != {"process_id", "cgroup_id"} or not _text(metric["labels"].get("cgroup_id"))
                            or metric["labels"].get("process_id") != str(identity["pid"])):
                        errors.append("CPU telemetry normalized measurements do not conserve CPU process identity")
                    elif metric["name"].startswith("host.process.") and CPU_PROCESS_METRICS.get(metric["name"][len("host.process."):]) != metric.get("unit"):
                        errors.append("CPU telemetry normalized process metric uses an unsupported name/unit")
    terms = summary.get("child_terminal_observations", [])
    if [item.get("pid") for item in terms] != child_pids:
        errors.append("CPU telemetry terminal census omits or invents owning Popen children")
    for index, item in enumerate(terms):
        matching = [sample for sample in samples if sample.get("process_identity") == item.get("process_identity") and sample.get("status") == "sample"]
        last = matching[-1]["monotonic_ns"] if matching else None
        if (type(item.get("sequence")) is not int or item["sequence"] != index or item.get("process_identity") not in identities
                or item.get("process_identity", {}).get("pid") != item.get("pid") or item.get("source") != "owning-Popen-returncode"
                or type(item.get("exit_code")) is not int or item["exit_code"] != 0 or item.get("final_capture_expected") is not False
                or last is None or item.get("last_successful_sample_monotonic_ns") != last
                or not _positive_int(item.get("monotonic_ns")) or item["monotonic_ns"] < last
                or item.get("unobserved_terminal_interval_nanoseconds") != [last, item["monotonic_ns"]]):
            errors.append("CPU telemetry terminal observation lacks actual child identity, prior sample or owning Popen returncode")
    for record in summary.get("errors", []) + summary.get("gaps", []):
        sequence = record.get("terminal_sequence")
        eligible_source = (record.get("source_id") == "process.identity-or-capture" and record.get("error_class") == "FileNotFoundError") or record.get("reason") == "lost-or-changed-process"
        if (record.get("resolution") != "owning-Popen-terminal-observation" or record.get("terminal_race_eligible") is not True or not eligible_source
                or type(sequence) is not int or not 0 <= sequence < len(terms)):
            errors.append("CPU telemetry has unresolved errors/gaps or falsely resolves context failure as child exit")
        else:
            term = terms[sequence]
            interval = term.get("unobserved_terminal_interval_nanoseconds", [])
            if record.get("pid") != term.get("pid") or len(interval) != 2 or not interval[0] <= record.get("monotonic_ns", -1) <= interval[1]:
                errors.append("CPU telemetry resolved race is outside its real child terminal interval")
    if terminal.get("resolved_terminal_races") != sum(bool(record.get("resolution")) for record in summary.get("gaps", [])):
        errors.append("CPU telemetry resolved-race count does not conserve retained gaps")
    for identity in identities:
        good = [sample for sample in samples if sample.get("process_identity") == identity and sample.get("status") == "sample"]
        if not good:
            errors.append("CPU telemetry registered process lacks an actual normalized sample")
    adapter_good = [sample for sample in samples if sample.get("process_identity", {}).get("pid") == parent and sample.get("status") == "sample"]
    if not {"start", "stop"} <= {sample.get("boundary") for sample in adapter_good}:
        errors.append("CPU telemetry requires actual parent start and stop captures")
    for sample in samples:
        if sample.get("status") == "missing":
            valid = [term for term in terms if term.get("process_identity") == sample.get("process_identity") and term.get("last_successful_sample_monotonic_ns", 0) <= sample.get("monotonic_ns", -1) <= term.get("monotonic_ns", -1)]
            resolved = [row for row in summary.get("errors", []) + summary.get("gaps", []) if row.get("pid") == sample.get("process_identity", {}).get("pid") and row.get("resolution") == "owning-Popen-terminal-observation" and row.get("monotonic_ns", -1) >= sample.get("monotonic_ns", -1)]
            if not valid or not resolved:
                errors.append("CPU telemetry missing normalized sample is not bounded by a real terminal observation")
    return summary, errors


def _validate_cpu_telemetry(bundle: Path, envelope: dict, raw: dict, observations: dict) -> list[str]:
    errors: list[str] = []
    summary, metadata_errors = _telemetry_metadata(raw, observations)
    errors.extend(metadata_errors)
    native_refs = raw["adapter_result"]["native_evidence"]
    output_dir = PurePosixPath(raw["native_attempt"]["output"]["artifactName"]).parent

    def indexed(relative: str, sha256: str, label: str) -> Path | None:
        ref = {"path": relative, "sha256": sha256}
        if sum(item == ref for item in envelope["artifacts"]) != 1:
            errors.append(f"CPU telemetry {label} must be an exact distinct indexed native artifact")
        return _ref(bundle, ref, "CPU telemetry " + label, errors)

    paths = {}
    for key, filename in (("cpu_telemetry_summary", "finite-cpu-telemetry.json"), ("cpu_telemetry_raw", "finite-cpu-telemetry.raw")):
        ref = native_refs.get(key, {})
        if ref.get("artifactName") != filename:
            errors.append(f"CPU telemetry native_evidence.{key} must name {filename}")
            continue
        paths[key] = indexed((output_dir / filename).as_posix(), ref.get("sha256"), key)
    summary_path, raw_path = paths.get("cpu_telemetry_summary"), paths.get("cpu_telemetry_raw")
    if summary_path is None or raw_path is None:
        return errors
    if summary_path.stat().st_size > 8 * 1024 * 1024 or raw_path.stat().st_size > 4 * 1024 * 1024:
        return [*errors, "CPU telemetry retained artifacts exceed their finite byte limits"]
    retained_summary = _json(summary_path, "CPU telemetry summary", errors)
    if not _same_json(retained_summary, summary):
        errors.append("CPU telemetry retained summary differs from observations.telemetry")
    content = raw_path.read_bytes()
    raw_file = summary.get("raw_file", {})
    if (raw_file.get("path") != "finite-cpu-telemetry.raw" or type(raw_file.get("size_bytes")) is not int
            or raw_file["size_bytes"] != len(content) or raw_file.get("sha256") != _native_sha(content)
            or not _positive_int(summary.get("maximum_raw_bytes")) or not len(content) <= summary["maximum_raw_bytes"] <= 4 * 1024 * 1024):
        errors.append("CPU telemetry raw_file digest/size/path or byte limit mismatch")
    sealed_bundle = json.loads(raw["arm_configuration"]["telemetryBundle"])
    input_ref = envelope["workload_evidence"]["input"]
    source_relative = (PurePosixPath(input_ref["path"]).parent / sealed_bundle["artifactName"]).as_posix()
    source_path = indexed(source_relative, sealed_bundle["sha256"], "canonical source bundle")
    if source_path is None or source_path.stat().st_size != sealed_bundle["sizeBytes"]:
        errors.append("CPU telemetry canonical source bundle size mismatch")
    native_design = _json(bundle / envelope["experiment"]["design"]["path"], "CPU native design", errors)
    if {"path": source_relative, "sha256": sealed_bundle["sha256"]} not in native_design.get("input_artifacts", []):
        errors.append("CPU telemetry canonical source bundle is absent from immutable native inputs")
    boundaries = summary.get("boundaries", [])
    if ([row.get("boundary") for row in boundaries] != ["start", "stop"]
            or any(not _positive_int(row.get("monotonic_ns")) or _timestamp(row.get("observed_at")) is None for row in boundaries)
            or boundaries[0]["monotonic_ns"] >= boundaries[1]["monotonic_ns"]):
        errors.append("CPU telemetry requires ordered actual start/stop boundaries")
        return errors
    first, last = boundaries[0]["monotonic_ns"], boundaries[1]["monotonic_ns"]
    native_start, native_end = _timestamp(raw["native_attempt"]["startedAt"]), _timestamp(raw["native_attempt"]["completedAt"])
    if native_start is None or native_end is None or any(not native_start <= _timestamp(row["observed_at"]) <= native_end for row in boundaries):
        errors.append("CPU telemetry boundary UTC window escapes the actual parent process")
    entries = summary.get("raw_sources")
    if not isinstance(entries, list) or not entries:
        return [*errors, "CPU telemetry raw source census is missing"]
    offset, raw_parts, previous_ns = 0, [], first
    for sequence, entry in enumerate(entries):
        size = entry.get("size_bytes")
        if (type(size) is not int or size < 0 or type(entry.get("sequence")) is not int or entry["sequence"] != sequence
                or type(entry.get("offset")) is not int or entry["offset"] != offset or offset + size > len(content)):
            errors.append("CPU telemetry raw source sequence/offset/size is missing, overlapping or out of bounds")
            continue
        part = content[offset:offset + size]
        raw_parts.append((entry, part))
        offset += size
        if (entry.get("sha256") != _native_sha(part) or entry.get("source_sha256") != _native_sha(part)
                or type(entry.get("source_size_bytes")) is not int or entry["source_size_bytes"] != size
                or entry.get("truncated") is not False or ("native_read_truncated" in entry and entry["native_read_truncated"] is not False)):
            errors.append("CPU telemetry raw source digest/truncation mismatch")
        if (not _positive_int(entry.get("monotonic_ns")) or not previous_ns <= entry["monotonic_ns"] <= last
                or _timestamp(entry.get("observed_at")) is None or not _text(entry.get("source_id"))):
            errors.append("CPU telemetry raw source time/identity is malformed")
        previous_ns = entry.get("monotonic_ns", previous_ns)
    if offset != len(content):
        errors.append("CPU telemetry raw source census omits retained bytes")
    parent = raw["native_attempt"]["childPid"]
    identities = summary["registered_processes"]
    registrations = [(entry, data) for entry, data in raw_parts if entry["source_id"] == "process.identity.stat"]
    groups = [(entry, data) for entry, data in raw_parts if entry["source_id"] == "process.identity.cgroup"]
    members = [(entry, data) for entry, data in raw_parts if entry["source_id"] == "process.identity.cgroup-membership"]
    boot = [data for entry, data in raw_parts if entry["source_id"] == "host.boot_id"]
    if not boot or any(identity.get("host_boot_id_sha256") != _native_sha(boot[0].strip()) for identity in identities):
        errors.append("CPU telemetry process identities do not bind retained host boot identity")
    if any(len(records) != len(identities) for records in (registrations, groups, members)):
        errors.append("CPU telemetry raw registration census does not conserve every owning Popen PID")
    else:
        for identity, (stat_entry, stat), (group_entry, group), (member_entry, membership) in zip(identities, registrations, groups, members):
            pid, ppid, start = _proc_stat(stat)
            group_lines = [line[3:].decode() for line in group.splitlines() if line.startswith(b"0::")]
            if (pid != identity["pid"] or start != identity["starttime_ticks"]
                    or any(entry.get("pid") != pid for entry in (stat_entry, group_entry, member_entry))
                    or (identity["kind"] == "codec-child" and ppid != parent)
                    or group_lines != [identity["cgroup_path"]] or str(pid).encode() not in membership.split()):
                errors.append("CPU telemetry raw parent/child PID, starttime or cgroup registration mismatch")
    for boundary in ("start", "stop"):
        present = {entry["source_id"] for entry, _ in raw_parts if entry.get("boundary") == boundary}
        cgroup_present = {entry["source_id"] for entry, _ in raw_parts if entry.get("boundary") == boundary and entry.get("pid") == parent}
        if not HOST_CONTEXT <= present or not CGROUP_CONTEXT <= cgroup_present:
            errors.append(f"CPU telemetry false complete: required {boundary} host/cgroup context is missing")
    mounts = [data for entry, data in raw_parts if entry["source_id"] == "host.mountinfo.binding"]
    backing = summary.get("filesystem_backing_mount", {})
    if not mounts or not isinstance(backing, dict) or not _text(backing.get("native_record")) or backing["native_record"] not in mounts[0].decode().splitlines():
        errors.append("CPU telemetry filesystem context must bind a retained actual mount record")
    samples = summary["samples"]
    for sample in samples:
        if not first <= sample.get("monotonic_ns", -1) <= last:
            errors.append("CPU telemetry normalized sample escapes the observation window")
        if sample.get("status") != "sample":
            continue
        ref = sample.get("raw", {})
        sequence = ref.get("sequence")
        if (type(sequence) is not int or not 0 <= sequence < len(entries) or entries[sequence] != ref
                or ref.get("source_id") != "host.process-and-pressure" or ref.get("media_type") != "application/vnd.piq.procfs-file-bundle.v1"):
            errors.append("CPU telemetry normalized sample lacks its exact indexed raw source record")
            continue
        files = _native_files(content[ref["offset"]:ref["offset"] + ref["size_bytes"]])
        identity = sample["process_identity"]
        pid, ppid, start = _proc_stat(files["process.stat"])
        if pid != identity["pid"] or ref.get("pid") != pid or start != identity["starttime_ticks"] or (identity["kind"] == "codec-child" and ppid != parent):
            errors.append("CPU telemetry normalized raw sample PID/starttime differs from owning Popen identity")
    for term in summary["child_terminal_observations"]:
        if not first <= term.get("monotonic_ns", -1) <= last or _timestamp(term.get("observed_at")) is None or not native_start <= _timestamp(term["observed_at"]) <= native_end:
            errors.append("CPU telemetry child terminal observation escapes the actual process/collection window")
    return errors


def validate_cpu_telemetry(bundle: Path, envelope: dict, raw: dict, observations: dict) -> list[str]:
    """Validate exact indexed finite-observer bytes and actual process coverage."""
    try:
        return _validate_cpu_telemetry(bundle.resolve(), envelope, raw, observations)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return [f"malformed or missing CPU telemetry evidence ({type(exc).__name__}): {str(exc)[:160]}"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def _ref(package: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        errors.append(f"{label} must contain exactly path and sha256")
        return None
    relative, expected = value.get("path"), value.get("sha256")
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in (".", "..") for part in relative.split("/"))):
        errors.append(f"{label} must use a safe package-relative path")
        return None
    path = package / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != package):
        errors.append(f"{label} must not use symlinks")
        return None
    try:
        path.resolve().relative_to(package.resolve())
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing or empty")
    except (OSError, ValueError):
        errors.append(f"{label} file is missing, empty, or outside package")
        return None
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or file_sha256(path) != expected):
        errors.append(f"{label} sha256 does not match retained bytes")
    return path


def _json(path: Path | None, label: str, errors: list[str]) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{label} is not a JSON object: {exc}")
        return {}


def _ratio(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 40
        return str(Decimal(numerator) / Decimal(denominator))


def _validate_cpu_design(design: dict) -> list[str]:
    """Validate prospective protocol metadata, without reading any measurements."""
    errors: list[str] = []
    if type(design.get("schema_version")) is not int or design.get("schema_version") != 1 or design.get("kind") != DESIGN_KIND:
        errors.append("CPU design must declare qualified_cpu_codec_design v1")
    for field in ("study_id", "workload_scope", "execution_owner", "runtime_identifier", "hardware_identifier"):
        if not _text(design.get(field)):
            errors.append(f"CPU design {field} must be nonempty")
    if design.get("execution_class") != "cpu" or type(design.get("accelerator_count")) is not int or design.get("accelerator_count") != 0:
        errors.append("CPU design must declare CPU execution with zero accelerators")
    if not _same_json(design.get("criteria"), CRITERIA):
        errors.append("CPU design criteria differ from the fixed codec qualification protocol")
    if design.get("timed_boundary") != "complete_fresh_process" or design.get("timed_steps") != TIMED_STEPS:
        errors.append("CPU design must time the complete fresh process and all required I/O/correctness steps")
    if design.get("format_policy") != "flexible_lossless" or design.get("correctness_oracle") != "exact_input_sha256_roundtrip":
        errors.append("CPU codec design requires flexible lossless format and exact roundtrip oracle")
    if _timestamp(design.get("created_at")) is None:
        errors.append("CPU design created_at must be a UTC timestamp")
    groups = design.get("groups")
    expected = [{"id": f"pair-{number:02}", "kind": "pair", "order": order}
                for number, order in enumerate((["baseline", "candidate"], ["candidate", "baseline"]) * 2, 1)]
    controls = [{"id": f"control-{number:02}", "kind": "control", "order": ["baseline", "candidate"]}
                for number in range(1, 5)]
    # Controls follow their corresponding treatment pair; both control roles use baseline configuration.
    schedule = [item for pair in zip(expected, controls) for item in pair]
    if groups != schedule:
        errors.append("CPU design groups must be four alternating AB/BA pairs interleaved with four declared A/A controls")
    arms = design.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"baseline", "candidate"}:
        errors.append("CPU design arms must contain baseline and candidate")
    else:
        for role, arm in arms.items():
            if not isinstance(arm, dict) or not all(_text(arm.get(key)) for key in ("id", "implementation", "configuration")) or not isinstance(arm.get("native_configuration"), dict):
                errors.append(f"CPU design {role} requires id, implementation and exact configuration")
        if isinstance(arms.get("baseline"), dict) and isinstance(arms.get("candidate"), dict) and arms["baseline"].get("id") == arms["candidate"].get("id"):
            errors.append("CPU design treatment arms must be distinct")
        baseline_config, candidate_config = arms["baseline"].get("native_configuration", {}), arms["candidate"].get("native_configuration", {})
        if baseline_config.get("codec") != "pigz-gzip" or candidate_config.get("codec") != "zstd" or baseline_config.get("toolchain") != candidate_config.get("toolchain"):
            errors.append("CPU design requires pinned pigz-gzip and zstd arms with the same toolchain identity")
        toolchain = json.loads(baseline_config.get("toolchain", "null"))
        if not isinstance(toolchain, dict) or toolchain.get("kind") != "piq.codec-toolchain.v1" or type(toolchain.get("threads")) is not int or toolchain.get("threads") != 2:
            errors.append("CPU design requires a pinned two-worker codec toolchain")
        telemetry_bundle = json.loads(baseline_config.get("telemetryBundle", "null"))
        if (not isinstance(telemetry_bundle, dict) or set(telemetry_bundle) != {"artifactName", "sha256", "sizeBytes"}
                or telemetry_bundle["artifactName"] != "inputs/finite-cpu-telemetry.zip" or not _positive_int(telemetry_bundle["sizeBytes"])
                or not isinstance(telemetry_bundle["sha256"], str) or len(telemetry_bundle["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in telemetry_bundle["sha256"])
                or candidate_config.get("telemetryBundle") != baseline_config.get("telemetryBundle")):
            errors.append("CPU design requires the same sealed canonical telemetry bundle in both native configurations")
    if not isinstance(design.get("source_artifacts"), list) or not design["source_artifacts"]:
        errors.append("CPU design source_artifacts must be nonempty")
    if not isinstance(design.get("input_artifacts"), list) or not design["input_artifacts"]:
        errors.append("CPU design input_artifacts must be nonempty")
    if not _positive_int(design.get("input_bytes")):
        errors.append("CPU design input_bytes must be a positive integer")
    if not isinstance(design.get("host_observations"), dict) or not all(key in design["host_observations"] for key in ("hostname", "cpu_models", "cpu_affinity")):
        errors.append("CPU design must bind advance host model, hostname and CPU affinity observations")
    if not design.get("python_runtime"):
        errors.append("CPU design must bind the exact advance Python runtime observation")
    return errors


def validate_cpu_design(design: dict) -> list[str]:
    """Pre-execution metadata check, with malformed inputs rejected cleanly."""
    try:
        return _validate_cpu_design(design)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return [f"malformed CPU advance design ({type(exc).__name__})"]


def normalize_cpu_sample(design: dict, group: dict, envelope: dict, raw: dict,
                         observations: dict, *, design_sha256: str) -> dict:
    """Project canonical parent-runner and indexed adapter observations.

    The parent runner owns complete-process timing and child identity. The
    adapter owns retained archive and roundtrip observations. No analysis-added
    elapsed value or self-reported run/attempt ID is used.
    """
    native = raw["native_attempt"]
    role = raw["role"]
    arm = design["arms"]["baseline" if group["kind"] == "control" else role]
    if not _same_json(raw.get("arm_configuration"), arm["native_configuration"]):
        raise ValueError("native arm configuration differs from fixed design")
    _, telemetry_errors = _telemetry_metadata(raw, observations)
    if telemetry_errors:
        raise ValueError("; ".join(telemetry_errors))
    toolchain = json.loads(arm["native_configuration"]["toolchain"])
    if not _same_json(observations["runtime"]["toolchain"], toolchain) or not _same_json(observations["runtime"]["python"], design["python_runtime"]):
        raise ValueError("native runtime/toolchain identity drift")
    if any(not _same_json(observations["host"].get(key), value) for key, value in design["host_observations"].items()):
        raise ValueError("native CPU host/model/affinity identity drift")
    if observations["host"]["process_id"] != native["childPid"]:
        raise ValueError("adapter PID differs from actual parent-runner child PID")
    if native.get("state") != "completed" or native.get("exitCode") != 0 or native.get("failureReason") is not None:
        raise ValueError("canonical native attempt failed or is incomplete")
    elapsed = native["elapsedSeconds"]
    if not isinstance(elapsed, str) or not elapsed or any(character not in "0123456789." for character in elapsed) or elapsed.count(".") > 1:
        raise ValueError("native elapsedSeconds must be a plain decimal string")
    ns = Decimal(elapsed) * 1_000_000_000
    if not ns.is_finite() or ns <= 0 or ns != ns.to_integral_value():
        raise ValueError("native elapsedSeconds must represent exact positive nanoseconds")
    if observations["input_manifest_sha256"] != design["input"]["sha256"] or observations["input_bytes"] != design["input_bytes"]:
        raise ValueError("native input corpus identity or byte size drift")
    parts = [{"input_sha256": part["input"]["sha256"], "roundtrip_sha256": part["roundtrip_sha256"],
              "archive_bytes": part["archive"]["sizeBytes"]} for part in observations["parts"]]
    return {"study_id": design["study_id"], "study_design_sha256": design_sha256,
            "group_id": group["id"], "role": role, "run_id": raw["run_id"], "attempt_id": raw["attempt_id"],
            "configuration_id": arm["id"], "input_sha256": raw["input_sha256"],
            "source_sha256s": [ref["sha256"] for ref in design["source_artifacts"]],
            "runtime_identifier": envelope["stack"]["runtime_identifier"],
            "hardware_identifier": envelope["stack"]["hardware_identifier"],
            "device_type": "cpu", "accelerator_count": 0, "device_uuids": [],
            "fresh_process": True, "process_id": native["childPid"],
            "timed_steps": observations["timed_steps"], "started_at": native["startedAt"],
            "ended_at": native["completedAt"], "elapsed_ns": int(ns),
            "status": "succeeded", "accepted": True, "failure_reason": None, "parts": parts}


def _evaluate_cpu_study(design: dict, samples: list[dict], *, design_sha256: str) -> dict:
    """Return deterministic arithmetic; samples are native raw-result study_sample objects.

    This function does not load files. validate_cpu_receipt supplies verified,
    complete, identity-bound samples and checks this report byte-for-byte as JSON.
    """
    errors = validate_cpu_design(design)
    groups = design.get("groups") if isinstance(design.get("groups"), list) else []
    source_hashes = [item.get("sha256") for item in design.get("source_artifacts", []) if isinstance(item, dict)]
    inputs = [item.get("sha256") for item in design.get("input_artifacts", []) if isinstance(item, dict)]
    expected_identity = {"input_sha256": (design.get("input") or {}).get("sha256"),
                         "source_sha256s": source_hashes,
                         "runtime_identifier": design.get("runtime_identifier"),
                         "hardware_identifier": design.get("hardware_identifier")}
    by_group: dict[str, dict[str, dict]] = {}
    seen: set[tuple[str, str]] = set()
    ordered = []
    for sample in samples:
        if not isinstance(sample, dict):
            errors.append("CPU samples must be objects")
            continue
        key = (sample.get("run_id"), sample.get("attempt_id"))
        if not all(_text(value) for value in key) or key in seen:
            errors.append("CPU sample run/attempt identities must be nonempty and distinct")
        seen.add(key)
        group_id, role = sample.get("group_id"), sample.get("role")
        if role not in ("baseline", "candidate") or not _text(group_id):
            errors.append("CPU sample group/role is invalid")
            continue
        bucket = by_group.setdefault(group_id, {})
        if role in bucket:
            errors.append(f"CPU group {group_id} duplicates role {role}")
        bucket[role] = sample
        ordered.append((group_id, role))
        if sample.get("study_id") != design.get("study_id") or sample.get("study_design_sha256") != design_sha256:
            errors.append("CPU sample does not bind the prospective study design")
        if any(sample.get(field) != value for field, value in expected_identity.items()):
            errors.append("CPU sample input/source/runtime/hardware identity drift")
        if (sample.get("device_type") != "cpu" or type(sample.get("accelerator_count")) is not int or sample.get("accelerator_count") != 0
                or sample.get("device_uuids") != [] or FORBIDDEN_GPU_FIELDS.intersection(sample)):
            errors.append("CPU receipt cannot cover CUDA/GPU execution or power placeholders")
        if (sample.get("fresh_process") is not True or not _positive_int(sample.get("process_id"))
                or sample.get("timed_steps") != TIMED_STEPS):
            errors.append("CPU sample must retain fresh process identity and complete timed steps")
        if sample.get("status") != "succeeded" or sample.get("accepted") is not True or sample.get("failure_reason") is not None:
            errors.append("CPU qualification rejects failed or unaccepted attempts")
        start, end = _timestamp(sample.get("started_at")), _timestamp(sample.get("ended_at"))
        if start is None or end is None or end <= start or not _positive_int(sample.get("elapsed_ns")):
            errors.append("CPU sample requires positive elapsed_ns and ordered UTC window")
        parts = sample.get("parts")
        if (not isinstance(parts, list) or [part.get("input_sha256") for part in parts if isinstance(part, dict)] != inputs
                or any(not isinstance(part, dict) or part.get("roundtrip_sha256") != part.get("input_sha256")
                       or not _positive_int(part.get("archive_bytes")) for part in parts)):
            errors.append("CPU sample all-parts exact roundtrip or archive-byte evidence failed")
    expected_order = [(group["id"], role) for group in groups if isinstance(group, dict)
                      and isinstance(group.get("order"), list) for role in group["order"]]
    if ordered != expected_order:
        errors.append("CPU attempt census does not match the complete prospective group/arm order")
    if len({sample.get("process_id") for sample in samples if isinstance(sample, dict)}) != len(samples):
        errors.append("CPU study requires a distinct fresh process for every attempt")
    pairs, controls, baseline_times, candidate_times = [], [], [], []
    for group in groups:
        if not isinstance(group, dict):
            continue
        arms = by_group.get(group.get("id"), {})
        if set(arms) != {"baseline", "candidate"}:
            errors.append(f"CPU group {group.get('id')} is missing an arm")
            continue
        baseline, candidate = arms["baseline"], arms["candidate"]
        if baseline.get("run_id") != candidate.get("run_id"):
            errors.append(f"CPU group {group.get('id')} must conserve one actual paired PIQ run")
        baseline_id = (design.get("arms", {}).get("baseline") or {}).get("id")
        candidate_id = baseline_id if group.get("kind") == "control" else (design.get("arms", {}).get("candidate") or {}).get("id")
        if baseline.get("configuration_id") != baseline_id or candidate.get("configuration_id") != candidate_id:
            errors.append(f"CPU group {group.get('id')} configuration mismatch; A/A controls must use the baseline twice")
        a, b = baseline.get("elapsed_ns"), candidate.get("elapsed_ns")
        if not _positive_int(a) or not _positive_int(b):
            continue
        parts_a, parts_b = baseline.get("parts"), candidate.get("parts")
        if not isinstance(parts_a, list) or not isinstance(parts_b, list) or any(not isinstance(p, dict) or not _positive_int(p.get("archive_bytes")) for p in parts_a + parts_b):
            continue
        bytes_a, bytes_b = sum(p["archive_bytes"] for p in parts_a), sum(p["archive_bytes"] for p in parts_b)
        entry = {"id": group["id"], "baseline_attempt": [baseline.get("run_id"), baseline.get("attempt_id")],
                 "candidate_attempt": [candidate.get("run_id"), candidate.get("attempt_id")],
                 "baseline_elapsed_ns": a, "candidate_elapsed_ns": b,
                 "baseline_archive_bytes": bytes_a, "candidate_archive_bytes": bytes_b,
                 "baseline_over_candidate_ratio": _ratio(a, b)}
        if group.get("kind") == "control":
            entry["passed"] = 95 * b <= 100 * a <= 105 * b
            controls.append(entry)
            if not entry["passed"]:
                errors.append(f"CPU neutral control {group['id']} is outside [0.95,1.05]")
        else:
            entry["passed"] = 10 * a >= 11 * b and bytes_b <= bytes_a
            pairs.append(entry)
            baseline_times.append(a)
            candidate_times.append(b)
            if not entry["passed"]:
                errors.append(f"CPU treatment {group['id']} fails fixed gain or archive-size criteria")
    if baseline_times and candidate_times and max(candidate_times) > max(baseline_times):
        errors.append("CPU candidate observed maximum latency exceeds baseline observed maximum")
    if len(pairs) != 4 or len(controls) != 4:
        errors.append("CPU qualification requires all four treatment pairs and all four neutral controls")
    if len({s.get("run_id") for s in samples if isinstance(s, dict)}) != 8:
        errors.append("CPU qualification requires eight distinct actual PIQ group run IDs")
    return {"schema_version": 1, "kind": "qualified_cpu_study_report", "study_id": design.get("study_id"),
            "design_sha256": design_sha256, "criteria": dict(CRITERIA), "pairs": pairs, "controls": controls,
            "attempt_census": [[sample.get("run_id"), sample.get("attempt_id")] for sample in samples if isinstance(sample, dict)],
            "verdict": "QUALIFIED" if not errors else "INCONCLUSIVE", "errors": errors}


def evaluate_cpu_study(design: dict, samples: list[dict], *, design_sha256: str) -> dict:
    """Deterministic evaluator; malformed data fails closed as INCONCLUSIVE."""
    try:
        return _evaluate_cpu_study(design, samples, design_sha256=design_sha256)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return {"schema_version": 1, "kind": "qualified_cpu_study_report",
                "study_id": design.get("study_id") if isinstance(design, dict) else None,
                "design_sha256": design_sha256, "criteria": dict(CRITERIA), "pairs": [], "controls": [],
                "attempt_census": [], "verdict": "INCONCLUSIVE",
                "errors": [f"malformed CPU study evidence ({type(exc).__name__})"]}


def cpu_receipt_binds_envelope(package: Path, receipt: dict, envelope: Path) -> bool:
    """Exact membership only. Call only after validate_cpu_receipt returns []."""
    try:
        path = envelope if envelope.is_absolute() else package / envelope
        relative = path.resolve().relative_to(package.resolve()).as_posix()
        digest = file_sha256(path)
    except (OSError, ValueError):
        return False
    return {"path": relative, "sha256": digest} in receipt.get("piq_envelopes", [])


def _validate_cpu_receipt(package: Path, receipt: dict) -> list[str]:
    """Validate a v2 CPU receipt, all retained bundles, and fixed qualification."""
    package = package.resolve()
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["CPU receipt must be an object"]
    if (receipt.get("schema_version"), receipt.get("state"), receipt.get("execution_class")) != (2, "measured", "cpu"):
        errors.append("CPU receipt must declare schema_version 2, measured state and cpu execution")
    if FORBIDDEN_GPU_FIELDS.intersection(receipt):
        errors.append("CPU receipt forbids GPU power/device placeholders")
    refs = ("design", "registration", "raw_census", "correctness", "qualification_report", "baseline_review", "scientific_review")
    allowed = {"schema_version", "state", "execution_class", "workload_scope", "runtime_identifier", "hardware_identifier",
               "source_artifacts", "piq_envelopes", "accepted_repeats", "failed_attempts", "profile_evidence", "claims", *refs}
    if set(receipt) - allowed:
        errors.append("CPU receipt contains unsupported fields")
    documents = {name: _json(_ref(package, receipt.get(name), f"CPU receipt {name}", errors), name, errors) for name in refs}
    design = documents["design"]
    errors.extend(validate_cpu_design(design))
    design_hash = (receipt.get("design") or {}).get("sha256")
    for field in ("workload_scope", "runtime_identifier", "hardware_identifier"):
        if not _text(receipt.get(field)) or receipt.get(field) != design.get(field):
            errors.append(f"CPU receipt {field} must match the prospective design")
    if receipt.get("source_artifacts") != design.get("source_artifacts"):
        errors.append("CPU receipt source_artifacts must match the prospective design")
    for index, ref in enumerate(design.get("source_artifacts", []) if isinstance(design.get("source_artifacts"), list) else []):
        _ref(package, ref, f"CPU source {index}", errors)
    _ref(package, design.get("input"), "CPU input", errors)
    input_paths = [_ref(package, ref, f"CPU input part {index}", errors) for index, ref in enumerate(design.get("input_artifacts", []) if isinstance(design.get("input_artifacts"), list) else [])]
    if sum(path.stat().st_size for path in input_paths if path is not None) != design.get("input_bytes"):
        errors.append("CPU design input_bytes does not equal retained input part bytes")
    for name in ("registration", "raw_census", "correctness", "baseline_review", "scientific_review"):
        document = documents[name]
        if document.get("study_id") != design.get("study_id") or document.get("design_sha256") != design_hash:
            errors.append(f"CPU {name} does not bind the exact prospective study design")
    registration = documents["registration"]
    registered = _timestamp(registration.get("registered_at"))
    if registered is None or not _text(registration.get("authority")) or not _text(registration.get("record_id")):
        errors.append("CPU registration requires UTC time and authoritative record identity")
    created = _timestamp(design.get("created_at"))
    if registered is not None and created is not None and created > registered:
        errors.append("CPU design creation must precede registration")
    for name in ("baseline_review", "scientific_review"):
        review = documents[name]
        if (review.get("independent") is not True or review.get("verdict") != "PASS"
                or not _text(review.get("reviewer")) or review.get("reviewer") == design.get("execution_owner")
                or not _text(review.get("reasoning")) or _timestamp(review.get("reviewed_at")) is None):
            errors.append(f"CPU {name} requires a named independent review with reasoning and UTC time")
    baseline_review = documents["baseline_review"]
    sources = baseline_review.get("primary_sources")
    if (baseline_review.get("baseline_id") != (design.get("arms", {}).get("baseline") or {}).get("id")
            or baseline_review.get("optimized_standard") is not True or not isinstance(sources, list) or not sources
            or any(not isinstance(url, str) or urlsplit(url).scheme != "https" or not urlsplit(url).hostname for url in sources)):
        errors.append("CPU baseline review must explain the exact optimized standard baseline and retain primary sources")
    envelopes = receipt.get("piq_envelopes")
    if not isinstance(envelopes, list) or len(envelopes) != 8:
        errors.append("CPU receipt must bind eight distinct PIQ diagnostic group envelopes")
        envelopes = []
    inventory, group_envelopes, native_design_hashes = [], {}, {}
    seen_paths: set[str] = set()
    for envelope_ref in envelopes:
        path = _ref(package, envelope_ref, "CPU PIQ envelope", errors)
        if path is None:
            continue
        if path.as_posix() in seen_paths:
            errors.append("CPU PIQ envelope paths must be distinct")
        seen_paths.add(path.as_posix())
        errors.extend(f"CPU PIQ bundle: {error}" for error in validate_bundle(path.parent))
        envelope = _json(path, "PIQ envelope", errors)
        if envelope.get("kind") != "piq_experiment_evidence" or envelope.get("scope") != "diagnostic":
            errors.append("CPU retained PIQ envelopes must preserve diagnostic v1 scope")
        stack = envelope.get("stack", {})
        if (stack.get("runtime_identifier") != design.get("runtime_identifier")
                or stack.get("hardware_identifier") != design.get("hardware_identifier")
                or any(layer.get("name") == "accelerator" for layer in stack.get("layers", []) if isinstance(layer, dict))):
            errors.append("CPU PIQ stack identity drift or accelerator coverage")
        native_design_ref = envelope.get("experiment", {}).get("design", {})
        native_design = _json(path.parent / native_design_ref.get("path", "missing"), "native design", errors)
        resources = native_design.get("resources", {})
        if (resources.get("executor"), resources.get("acceleratorCount"), resources.get("nodeCount"), resources.get("processCount")) != ("local-process", 0, 1, 1):
            errors.append("CPU qualified codec groups require canonical one-process local CPU resources with zero accelerators")
        binding = native_design.get("qualification_study", {})
        group_id = binding.get("group_id")
        expected_group = next((g for g in design.get("groups", []) if isinstance(g, dict) and g.get("id") == group_id), None)
        if binding != {"study_id": design.get("study_id"), "design_sha256": design_hash, "group_id": group_id, "group": expected_group} or expected_group is None or group_id in group_envelopes:
            errors.append("CPU PIQ native plan does not bind a distinct prospective study group")
        group_envelopes[group_id] = envelope_ref
        native_design_hashes[group_id] = native_design_ref.get("sha256")
        native_sources = envelope.get("experiment", {}).get("source_artifacts", [])
        if any(sum(ref.get("sha256") == fixed.get("sha256") for ref in native_sources) != 1 for fixed in design.get("source_artifacts", [])):
            errors.append("CPU PIQ source identity drift")
        if envelope.get("workload_evidence", {}).get("input", {}).get("sha256") != (design.get("input") or {}).get("sha256"):
            errors.append("CPU PIQ input identity drift")
        attempts = envelope.get("attempts", [])
        for attempt in attempts:
            raw_ref = attempt.get("raw_result", {})
            raw_path = path.parent / raw_ref.get("path", "missing")
            raw = _json(raw_path, "PIQ raw attempt", errors)
            sample = None
            observation_ref = raw.get("adapter_result", {}).get("native_evidence", {}).get("observations", {})
            output_name = raw.get("native_attempt", {}).get("output", {}).get("artifactName")
            if isinstance(output_name, str) and isinstance(observation_ref.get("artifactName"), str):
                observation_path = path.parent / PurePosixPath(output_name).parent / observation_ref["artifactName"]
                observation_relative = observation_path.relative_to(path.parent).as_posix()
                indexed_observation = {"path": observation_relative, "sha256": observation_ref.get("sha256")}
                if indexed_observation not in envelope.get("artifacts", []):
                    errors.append("CPU native observations must be indexed in the PIQ bundle")
                observed = _json(_ref(package, {"path": observation_path.relative_to(package).as_posix(), "sha256": observation_ref.get("sha256")}, "CPU native observations", errors), "CPU native observations", errors)
                errors.extend(validate_cpu_telemetry(path.parent, envelope, raw, observed))
                for part in observed.get("parts", []):
                    archive = part.get("archive", {})
                    archive_path = observation_path.parent / archive.get("artifactName", "missing")
                    archive_ref = {"path": archive_path.relative_to(path.parent).as_posix(), "sha256": archive.get("sha256")}
                    if archive_ref not in envelope.get("artifacts", []):
                        errors.append("CPU archive bytes must be indexed in the PIQ bundle")
                    retained = _ref(package, {"path": archive_path.relative_to(package).as_posix(), "sha256": archive.get("sha256")}, "CPU retained archive", errors)
                    if retained is None or retained.stat().st_size != archive.get("sizeBytes") or part.get("roundtrip_bytes") != part.get("input", {}).get("sizeBytes"):
                        errors.append("CPU retained archive size or exact roundtrip byte count mismatch")
                try:
                    sample = normalize_cpu_sample(design, expected_group, envelope, raw, observed, design_sha256=design_hash)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    errors.append(f"CPU canonical native sample binding failed: {exc}")
            else:
                errors.append("CPU attempt requires canonical parent-runner output and indexed adapter observations")
            row = {"group_id": group_id, "role": attempt.get("role"), "run_id": attempt.get("run_id"),
                   "attempt_id": attempt.get("id"), "envelope": envelope_ref,
                   "raw_result": {"path": raw_path.relative_to(package).as_posix(), "sha256": raw_ref.get("sha256")}}
            inventory.append((row, sample))
            if not isinstance(sample, dict) or any(sample.get(field) != row[field] for field in ("group_id", "role", "run_id", "attempt_id")):
                errors.append("CPU raw sample does not match its native PIQ run/attempt/group/role")
            elif sample.get("status") != attempt.get("status") or sample.get("accepted") != attempt.get("accepted") or sample.get("failure_reason") != attempt.get("failure_reason"):
                errors.append("CPU raw sample native attempt disposition mismatch")
    order = [(group["id"], role) for group in design.get("groups", []) if isinstance(group, dict) and isinstance(group.get("order"), list) for role in group["order"]]
    inventory.sort(key=lambda item: order.index((item[0]["group_id"], item[0]["role"])) if (item[0]["group_id"], item[0]["role"]) in order else len(order))
    if registration.get("group_design_sha256s") != native_design_hashes or len(native_design_hashes) != 8:
        errors.append("CPU registration must bind every exact advance native group design hash")
    if documents["raw_census"].get("attempts") != [row for row, _ in inventory]:
        errors.append("CPU raw_census must conserve every PIQ run/attempt/raw-result identity in declared order")
    samples = [sample for _, sample in inventory if isinstance(sample, dict)]
    computed = evaluate_cpu_study(design, samples, design_sha256=design_hash)
    if documents["qualification_report"] != computed:
        errors.append("CPU qualification_report does not match deterministic recomputation")
    if computed["verdict"] != "QUALIFIED":
        errors.extend(computed["errors"])
    expected_correctness = [{"run_id": sample.get("run_id"), "attempt_id": sample.get("attempt_id"), "parts": sample.get("parts")} for sample in samples]
    if documents["correctness"].get("attempts") != expected_correctness or documents["correctness"].get("accepted") is not True:
        errors.append("CPU correctness receipt must retain the exact all-attempt/all-part roundtrip census")
    starts = [_timestamp(s.get("started_at")) for s in samples]
    ends = [_timestamp(s.get("ended_at")) for s in samples]
    baseline_reviewed = _timestamp(baseline_review.get("reviewed_at"))
    scientific_reviewed = _timestamp(documents["scientific_review"].get("reviewed_at"))
    if any(start is None or registered is None or baseline_reviewed is None or start <= registered or start <= baseline_reviewed for start in starts):
        errors.append("CPU registration and independent baseline review must precede every execution")
    if any(end is None or scientific_reviewed is None or scientific_reviewed < end for end in ends):
        errors.append("CPU scientific review must follow every retained execution")
    if any(previous is not None and following is not None and following < previous for previous, following in zip(ends, starts[1:])):
        errors.append("CPU attempt UTC windows must follow the declared sequential execution order")
    if documents["scientific_review"].get("qualification_report_sha256") != (receipt.get("qualification_report") or {}).get("sha256"):
        errors.append("CPU scientific review must bind the exact recomputed qualification report")
    repeats = receipt.get("accepted_repeats")
    expected_repeats = [{"id": group["id"], "accepted": True, "raw_result": group_envelopes.get(group["id"])} for group in design.get("groups", []) if isinstance(group, dict) and group.get("kind") == "pair"]
    if repeats != expected_repeats or len(expected_repeats) != 4:
        errors.append("CPU accepted_repeats must conserve all four distinct treatment pairs and bound envelopes")
    failed = [{"run_id": s.get("run_id"), "attempt_id": s.get("attempt_id"), "failure_reason": s.get("failure_reason")} for s in samples if s.get("status") != "succeeded" or s.get("accepted") is not True]
    if receipt.get("failed_attempts") != failed or failed:
        errors.append("CPU failed_attempts must retain every failure; any failure blocks qualification")
    profiles = receipt.get("profile_evidence")
    if not isinstance(profiles, list):
        errors.append("CPU profile_evidence must be a list; empty is permitted for unprofiled claims")
        profiles = []
    for profile in profiles:
        if not isinstance(profile, dict) or profile.get("kind") != "cpu_profile" or profile.get("capture_performed") is not True:
            errors.append("CPU profile evidence must identify an actual CPU capture")
        elif profile.get("study_id") != design.get("study_id") or profile.get("design_sha256") != design_hash:
            errors.append("CPU profile evidence must bind the study design")
        else:
            _ref(package, profile.get("artifact"), "CPU profile capture", errors)
    claims = receipt.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("CPU claims must be nonempty")
        claims = []
    ids = []
    repeat_ids = [repeat["id"] for repeat in expected_repeats]
    for claim in claims:
        if not isinstance(claim, dict):
            errors.append("CPU claim must be an object")
            continue
        ids.append(claim.get("id"))
        if not _text(claim.get("id")) or claim.get("accepted_repeat_ids") != repeat_ids:
            errors.append("CPU claim must bind every distinct accepted treatment repeat")
        _ref(package, claim.get("source"), "CPU claim source", errors)
        if type(claim.get("profile_claim")) is not bool or (claim.get("profile_claim") and not profiles):
            errors.append("CPU profiler claims require retained actual CPU captures; unprofiled claims must say false")
    if len(set(ids)) != len(ids):
        errors.append("CPU claim IDs must be distinct")
    return errors


def validate_cpu_receipt(package: Path, receipt: dict) -> list[str]:
    """Public fail-closed gate; malformed containers never make a package ready."""
    try:
        return _validate_cpu_receipt(package, receipt)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return [f"malformed CPU receipt evidence ({type(exc).__name__})"]
