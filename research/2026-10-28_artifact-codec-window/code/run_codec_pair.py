#!/usr/bin/env python3
"""Run two bounded standalone codec observations, or pin their native toolchain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time

FILE_CAP = 32 * 1024 * 1024
ADDRESS_CAP = 512 * 1024 * 1024
METADATA_CAP = 64 * 1024
CODECS = {"A": "pigz-gzip", "B": "zstd"}
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "TZ": "UTC"}
PYCACHE_PATH = "/var/empty/piq-python-cache"
PYTHON_FLAGS = ("-I", "-B", "-X", "pycache_prefix=" + PYCACHE_PATH)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def document(path: Path, cap: int = FILE_CAP) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("expected a bounded regular JSON file: " + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def retain(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("paths must be absolute: " + str(path))
    return path


def new_output(path: Path) -> Path:
    absolute(path)
    path.mkdir(mode=0o755)
    return path.resolve(strict=True)


def regular(path: Path) -> Path:
    absolute(path)
    resolved = path.resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("pin must resolve to a regular file")
    return resolved


def validate_toolchain(value: dict, verify_files: bool = True) -> None:
    if set(value) != {"kind", "threads", "tools", "libraries"}:
        raise ValueError("toolchain has unknown or missing keys")
    if value["kind"] != "piq.codec-toolchain.v1" or type(value["threads"]) is not int or value["threads"] != 2:
        raise ValueError("toolchain kind/threads differ")
    tools = value["tools"]
    libraries = value["libraries"]
    if not isinstance(tools, dict) or set(tools) != {"pigz", "zstd"}:
        raise ValueError("both exact codec tool pins are required")
    if not isinstance(libraries, list) or not 1 <= len(libraries) <= 16:
        raise ValueError("one to sixteen loader/library pins are required")
    paths = set()
    for item, keys in [(x, {"path", "sha256", "version"}) for x in tools.values()] + [(x, {"path", "sha256"}) for x in libraries]:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("malformed toolchain pin")
        if not isinstance(item["path"], str) or not isinstance(item["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
            raise ValueError("invalid toolchain path or digest")
        path = absolute(Path(item["path"]))
        if str(path) in paths:
            raise ValueError("duplicate toolchain paths")
        paths.add(str(path))
        if "version" in item and (not isinstance(item["version"], str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", item["version"])):
            raise ValueError("invalid tool version")
        if verify_files and (regular(path) != path or digest(path) != item["sha256"]):
            raise ValueError("toolchain file pin changed: " + str(path))


def _limits(cpus: tuple[int, int] | None, file_cap: int) -> None:
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_CAP, ADDRESS_CAP))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_cap, file_cap))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    if cpus is not None:
        os.sched_setaffinity(0, cpus)


def fresh_process(argv: list[str], directory: Path, cpus: tuple[int, int] | None,
                  timeout: float, file_cap: int = FILE_CAP) -> dict:
    """Keep the session leader unreaped until owned group cleanup signals finish."""
    started = time.monotonic_ns()
    record = {"argv": argv, "state": "FAILED", "pid": None, "exit_code": None,
              "timed_out": False, "complete_process_wall_ns": None,
              "process_exit_observed_ns": None,
              "cleanup": {"group_signal_before_reap": False, "leader_reaped": False,
                          "independent_descendant_census": False}}
    child = None
    try:
        with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
            child = subprocess.Popen(argv, cwd=directory, env=ENVIRONMENT,
                                     stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     start_new_session=True, close_fds=True,
                                     preexec_fn=lambda: _limits(cpus, file_cap))
            record["pid"] = child.pid
            total_deadline = started + int(timeout * 1e9)
            deadline = total_deadline - int(min(0.25, timeout / 5) * 1e9)
            while True:
                # WNOWAIT preserves the PID reservation, even after leader exit.
                exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                now = time.monotonic_ns()
                if exited is not None:
                    record["process_exit_observed_ns"] = now
                    break
                if now >= deadline:
                    record["timed_out"] = True
                    break
                time.sleep(min(0.005, max(0, (deadline - now) / 1e9)))
    except BaseException as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        if child is not None:
            # Never poll/reap before killpg. A retained leader PID prevents PGID reuse.
            try:
                os.killpg(child.pid, signal.SIGKILL)
                record["cleanup"]["group_signal_before_reap"] = True
            except ProcessLookupError:
                record["cleanup"]["group_already_absent"] = True
            except OSError as exc:
                record["cleanup"]["error"] = str(exc)[:1024]
            try:
                remaining = max(0.001, (started + int(timeout * 1e9) - time.monotonic_ns()) / 1e9)
                record["exit_code"] = child.wait(timeout=remaining)
                record["cleanup"]["leader_reaped"] = True
            except BaseException as exc:
                record["cleanup"]["reap_error"] = type(exc).__name__
            ended = time.monotonic_ns()
            record["complete_process_wall_ns"] = ended - started
            record["finished_monotonic_ns"] = ended
        record["started_monotonic_ns"] = started
        if child is None:
            record["finished_monotonic_ns"] = time.monotonic_ns()
    if (record["exit_code"] == 0 and not record["timed_out"] and "error" not in record
            and record["cleanup"]["leader_reaped"] and "error" not in record["cleanup"]):
        record["state"] = "COMPLETE"
    return record


def metadata(argv: list[str], output: Path, name: str) -> bytes:
    directory = output / name
    directory.mkdir()
    record = fresh_process(argv, directory, None, 3, METADATA_CAP)
    retain(directory / "process.json", record)
    raw = (directory / "stdout.txt").read_bytes() + (directory / "stderr.txt").read_bytes()
    if record["state"] != "COMPLETE" or len(raw) > METADATA_CAP:
        raise ValueError("native metadata command failed or exceeded its bound: " + name)
    return raw


def parse_ldd(raw: bytes) -> set[Path]:
    paths = set()
    for line in raw.decode("utf-8", errors="strict").splitlines():
        text = line.strip()
        if not text:
            continue
        if re.fullmatch(r"linux-vdso\.so\.\d+\s+\(0x[0-9a-fA-F]+\)", text):
            continue
        matched = re.fullmatch(r"(?:\S+\s+=>\s+)?(/\S+)\s+\(0x[0-9a-fA-F]+\)", text)
        if matched is None:
            raise ValueError("unknown or missing ldd closure entry: " + text)
        paths.add(regular(Path(matched.group(1))))
    if not paths or not any("ld-linux" in path.name or path.name.startswith("ld-musl-") for path in paths):
        raise ValueError("loader/library closure is absent or unknown")
    return paths


def pin(pigz: Path, zstd: Path, output: Path) -> dict:
    root = new_output(output)
    handback = {"state": "PARTIAL", "operation": "pin", "qualification": "not_assessed"}
    retain(root / "handback.json", handback)
    try:
        require_linux()
        libraries, tools = set(), {}
        ldd = regular(Path("/usr/bin/ldd"))
        for name, requested in (("pigz", pigz), ("zstd", zstd)):
            path = regular(requested)
            before = digest(path)
            raw = metadata([str(path), "--version"], root, name + "-version")
            pattern = rb"\bpigz\s+(\d+(?:\.\d+){1,3})\b" if name == "pigz" else rb"\bv(\d+(?:\.\d+){1,3})\b"
            found = re.search(pattern, raw)
            if found is None:
                raise ValueError("unrecognized codec version")
            libraries.update(parse_ldd(metadata([str(ldd), str(path)], root, name + "-ldd")))
            if before != digest(path):
                raise ValueError("codec changed during metadata capture")
            tools[name] = {"path": str(path), "sha256": before, "version": found.group(1).decode()}
        toolchain = {"kind": "piq.codec-toolchain.v1", "threads": 2, "tools": tools,
                     "libraries": [{"path": str(path), "sha256": digest(path)} for path in sorted(libraries)]}
        validate_toolchain(toolchain)
        retain(root / "toolchain.json", toolchain)
        handback.update(state="COMPLETE", toolchain_sha256=digest(root / "toolchain.json"),
                        metadata_tool={"path": str(ldd), "sha256": digest(ldd)})
    except BaseException as exc:
        handback["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        retain(root / "handback.json", handback)
    return handback


def require_linux() -> None:
    if sys.platform != "linux" or not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise ValueError("Linux with waitid WNOWAIT is required")
    if os.path.lexists(PYCACHE_PATH):
        raise ValueError("the configured pycache prefix must be absent")


def parse_cpus(value: str) -> tuple[int, int]:
    if re.fullmatch(r"\d+,\d+", value) is None:
        raise ValueError("--cpus must contain exactly two distinct CPU IDs")
    values = tuple(int(x) for x in value.split(","))
    if len(set(values)) != 2 or not set(values) <= os.sched_getaffinity(0):
        raise ValueError("both CPU IDs must be distinct and currently allowed")
    return values


def verify_reference(directory: Path, reference: dict) -> None:
    name = reference.get("artifactName")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("unsafe workload artifact reference")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(directory) is False:
        raise ValueError("workload artifact is missing or escapes its output")
    if path.stat().st_size > FILE_CAP or digest(path) != reference.get("sha256"):
        raise ValueError("workload artifact digest or size cap differs")
    if "sizeBytes" in reference and path.stat().st_size != reference["sizeBytes"]:
        raise ValueError("workload artifact byte count differs")


def validate_outputs(directory: Path, input_index: Path | None = None) -> list[dict]:
    values, outputs = {}, []
    for name in ("result.json", "native-evidence.json", "workload-result.json"):
        path = directory / name
        values[name] = document(path)
        outputs.append({"path": name, "sha256": digest(path), "size_bytes": path.stat().st_size})
    result = values["result.json"]
    if (not isinstance(result.get("input_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", result["input_sha256"]) is None
            or result.get("roundtrip_sha256") != result["input_sha256"]):
        raise ValueError("workload exact roundtrip correctness is absent")
    if input_index is not None:
        index = document(input_index)
        expected = {"input_manifest_sha256": digest(input_index), "input_bytes": index["corpusBytes"],
                    "input_sha256": index["corpusSha256"], "chunk_count": len(index["chunks"]),
                    "source_file_count": index["sourceFileCount"],
                    "corpus_ledger_sha256": index["ledger"]["sha256"]}
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError("workload comparison differs from the pinned corpus index")
    for name in ("native-evidence.json", "workload-result.json"):
        if values[name].get("observation_mode") != "standalone_workload" or values[name].get("qualification") != "not_assessed":
            raise ValueError("workload output observation labels differ: " + name)
    workload = values["workload-result.json"]
    if workload.get("timing_scope") != "adapter_local_workload_window":
        raise ValueError("child-local timing scope differs")
    for key, name in (("result", "result.json"), ("observations", "native-evidence.json")):
        ref = workload.get(key)
        if not isinstance(ref, dict) or ref.get("artifactName") != name:
            raise ValueError("workload lacks original result/observation references")
        verify_reference(directory, ref)
    def references(value):
        if isinstance(value, dict):
            if "artifactName" in value:
                verify_reference(directory, value)
            else:
                for member in value.values():
                    references(member)
        elif isinstance(value, list):
            for member in value:
                references(member)
    references(workload.get("native_evidence", {}))
    return outputs


def run(input_index: Path, toolchain_path: Path, output: Path, cpus: str, order: str = "AB",
        *, workload: Path | None = None) -> dict:
    root = new_output(output)
    arms = [{"label": label, "codec": CODECS[label], "state": "NOT_STARTED"} for label in order] if order in ("AB", "BA") else []
    handback = {"state": "PARTIAL", "observation_mode": "standalone_workload",
                "qualification": "not_assessed", "order": order, "arms": arms,
                "instrumentation": "standalone process observations without PIQ telemetry or admission"}
    retain(root / "handback.json", handback)
    try:
        require_linux()
        if order not in ("AB", "BA"):
            raise ValueError("order must be AB or BA")
        selected_cpus = parse_cpus(cpus)
        index = absolute(input_index).resolve(strict=True)
        document(index)
        toolchain = document(absolute(toolchain_path))
        validate_toolchain(toolchain)
        executable = Path(sys.executable).resolve(strict=True)
        source = regular(workload or Path(__file__).resolve().with_name("codec_workload.py"))
        handback.update(cpus=list(selected_cpus), limits={"address_space_bytes_per_process": ADDRESS_CAP,
                        "file_bytes": FILE_CAP, "wall_seconds_per_arm": 15},
                        input_index={"path": str(index), "sha256": digest(index)},
                        toolchain={"path": str(toolchain_path), "sha256": digest(toolchain_path)},
                        workload={"path": str(source), "sha256": digest(source)},
                        python={"path": str(executable), "sha256": digest(executable)})
        frozen_toolchain = canonical(toolchain).decode()
        for arm in arms:
            directory = root / arm["label"]
            directory.mkdir()
            request = {"input_index": str(index), "output_directory": str(directory),
                       "codec": arm["codec"], "toolchain_json": frozen_toolchain, "timeout_seconds": 15}
            request_path = directory / "request.json"
            retain(request_path, request)
            if (digest(index) != handback["input_index"]["sha256"]
                    or digest(source) != handback["workload"]["sha256"]
                    or digest(toolchain_path) != handback["toolchain"]["sha256"]
                    or digest(executable) != handback["python"]["sha256"]):
                raise ValueError("frozen input, source, toolchain or runtime drift")
            validate_toolchain(toolchain)
            arm["state"] = "RUNNING"
            retain(root / "handback.json", handback)
            process = fresh_process([sys.executable, *PYTHON_FLAGS, str(source), "--request", str(request_path)],
                                    directory, selected_cpus, 15)
            arm.update(process, codec=arm["codec"], label=arm["label"])
            retain(directory / "process.json", process)
            if process["state"] != "COMPLETE":
                raise ValueError("arm failed, later arm remains not started")
            arm["outputs"] = validate_outputs(directory, index)
            retain(root / "handback.json", handback)
        handback["state"] = "COMPLETE"
    except BaseException as exc:
        for arm in arms:
            if arm["state"] == "RUNNING":
                arm["state"] = "FAILED"
            if arm["state"] == "COMPLETE" and "outputs" not in arm:
                arm["state"] = "FAILED_OUTPUT_VALIDATION"
        handback["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        retain(root / "handback.json", handback)
    return handback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pins = commands.add_parser("pin", help="retain bounded native version/loader metadata and exact file pins")
    pins.add_argument("--pigz", type=Path, required=True)
    pins.add_argument("--zstd", type=Path, required=True)
    pins.add_argument("--output", type=Path, required=True)
    pair = commands.add_parser("run", help="run one diagnostic pair without scientific qualification")
    pair.add_argument("--input-index", type=Path, required=True)
    pair.add_argument("--toolchain", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--cpus", required=True)
    pair.add_argument("--order", choices=("AB", "BA"), default="AB")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt("received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        outcome = pin(args.pigz, args.zstd, args.output) if args.command == "pin" else run(args.input_index, args.toolchain, args.output, args.cpus, args.order)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
