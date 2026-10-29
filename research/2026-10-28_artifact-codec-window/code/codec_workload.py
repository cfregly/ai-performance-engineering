"""Standalone extraction of a bounded artifact codec workload.

The preserved core reports an adapter-local window. A parent process measures
complete-process wall time. This entry point does not assess qualification.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import resource
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


MAX_PART_BYTES = 7 * 1024 * 1024


MAX_OUTPUT_BYTES = 8 * 1024 * 1024


MAX_CORPUS_BYTES = 64 * 1024 * 1024


MAX_INDEX_BYTES = 64 * 1024


MAX_LEDGER_BYTES = 512 * 1024


MAX_PARTS = 16


TIMED_STEPS = [
    "input_read",
    "input_hash",
    "compression",
    "archive_write",
    "archive_fsync",
    "decompression",
    "roundtrip_verification",
]


WINDOW = (
    "One sequential pass over the sealed chunk corpus, including index and ledger "
    "read/hash, input read/hash, native compression, archive write and file fsync, "
    "archive readback/hash, native decompression and exact roundtrip verification, "
    "and comparison-result write/fsync. Excludes imports, request/toolchain validation "
    "and final observation serialization. Runner complete fresh-process wall time is primary."
)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _plain(value: str, *, directory: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError("paths must be canonical and absolute")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlinks are not permitted")
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("expected an existing regular file or directory")
    return path


def _read(path: Path, cap: int) -> bytes:
    _plain(str(path))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("input must be a regular file")
        data = stream.read(cap + 1)
    if not data or len(data) > cap:
        raise ValueError("file is empty or exceeds its byte cap")
    return data


def _ref(path: Path, data: bytes) -> dict:
    return {"artifactName": path.name, "sha256": _sha(data)}


def _write(path: Path, data: bytes) -> dict:
    if not data or len(data) > MAX_OUTPUT_BYTES:
        raise ValueError("output exceeds its byte cap")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return _ref(path, data)


def _chunk_ref(value: object, cap: int) -> dict:
    if not isinstance(value, dict) or set(value) != {"artifactName", "sha256", "sizeBytes"}:
        raise ValueError("input references require exactly artifactName, sha256 and sizeBytes")
    name = value["artifactName"]
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        raise ValueError("input references must be canonical relative paths")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "inputs"
        or ".." in path.parts
        or path.as_posix() != name
    ):
        raise ValueError("input references must be directly under inputs")
    if (
        not _digest(value["sha256"])
        or type(value["sizeBytes"]) is not int
        or not 0 < value["sizeBytes"] <= cap
    ):
        raise ValueError("input reference hash or size is invalid")
    return value


def _referenced(parent: Path, ref: dict, cap: int) -> bytes:
    data = _read(parent / ref["artifactName"], cap)
    if len(data) != ref["sizeBytes"] or _sha(data) != ref["sha256"]:
        raise ValueError("input reference differs from retained bytes")
    return data


def _index(data: bytes) -> dict:
    value = json.loads(data)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schemaVersion", "corpusSha256", "corpusBytes", "sourceFileCount", "ledger", "chunks"}
        or value["schemaVersion"] != "piq.artifact-compression-chunks.v1"
    ):
        raise ValueError("unsupported sealed chunk index")
    if (
        not _digest(value["corpusSha256"])
        or type(value["corpusBytes"]) is not int
        or not 0 < value["corpusBytes"] <= MAX_CORPUS_BYTES
        or type(value["sourceFileCount"]) is not int
        or value["sourceFileCount"] <= 0
    ):
        raise ValueError("invalid corpus identity or size")
    ledger = _chunk_ref(value["ledger"], MAX_LEDGER_BYTES)
    if ledger["artifactName"] != "inputs/corpus-ledger.json":
        raise ValueError("source ledger must be inputs/corpus-ledger.json")
    parts = value["chunks"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError("part count exceeds its cap")
    for part in parts:
        _chunk_ref(part, MAX_PART_BYTES)
    if (
        len({p["artifactName"] for p in parts}) != len(parts)
        or len({p["sha256"] for p in parts}) != len(parts)
        or sum(p["sizeBytes"] for p in parts) != value["corpusBytes"]
    ):
        raise ValueError("parts must be distinct and conserve corpus bytes")
    return value


def _file_hash(path: Path) -> str:
    _plain(str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _toolchain(text: object) -> dict:
    if not isinstance(text, str) or len(text.encode()) > MAX_INDEX_BYTES:
        raise ValueError("toolchain must be bounded JSON text")
    value = json.loads(text)
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "threads", "tools", "libraries"}
        or value["kind"] != "piq.codec-toolchain.v1"
        or type(value["threads"]) is not int
        or value["threads"] != 2
        or not isinstance(value["tools"], dict)
        or set(value["tools"]) != {"pigz", "zstd"}
    ):
        raise ValueError("toolchain must pin both native codecs with two threads")
    libraries = value["libraries"]
    if not isinstance(libraries, list) or not 1 <= len(libraries) <= 16:
        raise ValueError("toolchain must pin a bounded loader/library closure")
    seen = set()
    for name, identity in [*value["tools"].items(), *(("library", item) for item in libraries)]:
        fields = {"path", "sha256", "version"} if name != "library" else {"path", "sha256"}
        if (
            not isinstance(identity, dict)
            or set(identity) != fields
            or not isinstance(identity.get("path"), str)
        ):
            raise ValueError("invalid toolchain identity")
        path = _plain(identity["path"])
        if (
            path in seen
            or not _digest(identity["sha256"])
            or _file_hash(path) != identity["sha256"]
        ):
            raise ValueError("toolchain identity drift or duplicate path")
        seen.add(path)
        if name != "library" and (
            not os.access(path, os.X_OK)
            or not isinstance(identity["version"], str)
            or not identity["version"]
        ):
            raise ValueError("codec identity requires executable and version")
    return value


def _command(toolchain: dict, codec: str, *, decompress: bool) -> list[str]:
    if codec == "pigz-gzip":
        return [
            toolchain["tools"]["pigz"]["path"],
            *(["-d", "-c"] if decompress else ["-6", "-p", "2", "-n", "-c"]),
        ]
    return [
        toolchain["tools"]["zstd"]["path"],
        *(
            ["-d", "--no-progress", "-q", "-c"]
            if decompress
            else ["-3", "-T2", "--no-progress", "-q", "-c"]
        ),
    ]


def _codec(command: list[str], data: bytes, path: Path, timeout: int, observer=None) -> dict:
    """Native child shares the runner process group, affinity and resource limits."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    started = time.perf_counter_ns()
    with os.fdopen(descriptor, "wb") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stream,
            stderr=subprocess.PIPE,
            close_fds=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        stderr = b""
        try:
            if observer is not None:
                observer.observe_child(process.pid)
            _, stderr = process.communicate(input=data, timeout=timeout)
        except BaseException:
            if process.poll() is None:
                process.kill()
            _, stderr = process.communicate()
            raise
        finally:
            stderr_path = path.with_name(path.name + ".stderr.log")
            descriptor = os.open(
                stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(descriptor, "wb") as diagnostic:
                diagnostic.write(stderr[:MAX_OUTPUT_BYTES])
                diagnostic.flush()
                os.fsync(diagnostic.fileno())
            stderr_record = _write(
                path.with_name(path.name + ".stderr.json"),
                _bytes(
                    {
                        "encoding": "base64",
                        "original_size_bytes": len(stderr),
                        "original_sha256": _sha(stderr),
                        "accepted_size_bytes": min(len(stderr), 4096),
                        "truncated": len(stderr) > 4096,
                        "raw_file": None
                        if not stderr
                        else {
                            "artifactName": stderr_path.name,
                            "sha256": _sha(stderr[:MAX_OUTPUT_BYTES]),
                            "sizeBytes": min(len(stderr), MAX_OUTPUT_BYTES),
                            "truncated": len(stderr) > MAX_OUTPUT_BYTES,
                        },
                        "data": base64.b64encode(stderr[:4096]).decode("ascii"),
                    }
                ),
            )
            if observer is not None:
                observer.unobserve_child(process.pid, exit_code=process.returncode)
        if process.returncode != 0:
            raise ValueError(
                f"native codec exited {process.returncode}: {stderr[:512].decode(errors='replace')}"
            )
        if stderr:
            raise ValueError(
                "native codec emitted diagnostics; retained stderr identifies any byte-cap truncation and qualification is refused"
            )
        stream.flush()
        os.fsync(stream.fileno())
    size = path.stat().st_size
    if not 0 < size <= MAX_OUTPUT_BYTES:
        raise ValueError("native output exceeds its byte cap")
    return {
        "argv": command,
        "pid": process.pid,
        "exit_code": process.returncode,
        "wall_ns_with_write_and_file_fsync": time.perf_counter_ns() - started,
        "size_bytes": size,
        "stderr_sha256": _sha(stderr),
        "stderr_bytes": len(stderr),
        "stderr_artifact": stderr_record,
    }


def _usage(who: int) -> dict:
    usage = resource.getrusage(who)
    return {
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "block_input_operations": usage.ru_inblock,
        "block_output_operations": usage.ru_oublock,
        "high_water_rss_kib_linux": usage.ru_maxrss,
    }


def _run_corpus(input_path, output, config, declared, toolchain, host, observer):
    before = {"self": _usage(resource.RUSAGE_SELF), "children": _usage(resource.RUSAGE_CHILDREN)}
    started = time.perf_counter_ns()
    index_bytes = _read(input_path, MAX_INDEX_BYTES)
    index = _index(index_bytes)
    ledger = _referenced(input_path.parent, index["ledger"], MAX_LEDGER_BYTES)
    if not isinstance(json.loads(ledger), dict):
        raise ValueError("source ledger must be a JSON object")
    ledger_ref = _write(output / "corpus-ledger.json", ledger)
    input_hash, roundtrip_hash = hashlib.sha256(), hashlib.sha256()
    parts, archives = [], {}
    for number, source in enumerate(index["chunks"]):
        phase = time.perf_counter_ns()
        original = _referenced(input_path.parent, source, MAX_PART_BYTES)
        input_hash.update(original)
        read_ns = time.perf_counter_ns() - phase
        suffix = "gz" if config["codec"] == "pigz-gzip" else "zst"
        archive_path = output / f"archive-{number:03d}.{suffix}"
        compressed = _codec(
            _command(toolchain, config["codec"], decompress=False),
            original,
            archive_path,
            declared["timeoutSeconds"],
            observer,
        )
        archive = _read(archive_path, MAX_OUTPUT_BYTES)
        archive_ref = _ref(archive_path, archive)
        roundtrip_path = output / f"roundtrip-{number:03d}.bin"
        decompressed = _codec(
            _command(toolchain, config["codec"], decompress=True),
            archive,
            roundtrip_path,
            declared["timeoutSeconds"],
            observer,
        )
        phase = time.perf_counter_ns()
        restored = _read(roundtrip_path, MAX_PART_BYTES)
        restored_sha = _sha(restored)
        if restored != original or restored_sha != source["sha256"]:
            raise ValueError("exact roundtrip verification failed")
        roundtrip_hash.update(restored)
        roundtrip_path.unlink()
        parts.append(
            {
                "ordinal": number,
                "input": source,
                "archive": {**archive_ref, "sizeBytes": len(archive)},
                "roundtrip_sha256": restored_sha,
                "roundtrip_bytes": len(restored),
                "input_read_hash_ns": read_ns,
                "compression": compressed,
                "decompression": decompressed,
                "roundtrip_read_verify_cleanup_ns": time.perf_counter_ns() - phase,
            }
        )
        archives[f"archive_{number:03d}"] = archive_ref
        for operation, capture in (("compression", compressed), ("decompression", decompressed)):
            if "stderr_artifact" in capture:
                archives[f"{operation}_stderr_{number:03d}"] = capture["stderr_artifact"]
        del original, archive, restored
    if (
        input_hash.hexdigest() != index["corpusSha256"]
        or roundtrip_hash.hexdigest() != index["corpusSha256"]
    ):
        raise ValueError("aggregate corpus hash mismatch")
    comparison = {
        "input_manifest_sha256": _sha(index_bytes),
        "input_bytes": index["corpusBytes"],
        "input_sha256": input_hash.hexdigest(),
        "roundtrip_sha256": roundtrip_hash.hexdigest(),
        "chunk_count": len(parts),
        "source_file_count": index["sourceFileCount"],
        "corpus_ledger_sha256": ledger_ref["sha256"],
    }
    result = _write(output / "result.json", _bytes(comparison))
    elapsed_ns = time.perf_counter_ns() - started
    observations = {
        "scope": "CPU lossless artifact codec; flexible archive format",
        **comparison,
        "parts": parts,
        "codec": config["codec"],
        "compressed_bytes": sum(p["archive"]["sizeBytes"] for p in parts),
        "timed_steps": TIMED_STEPS,
        "workload_elapsed_ns": elapsed_ns,
        "measurement_window": WINDOW,
        "host": host,
        "runtime": {
            "toolchain": toolchain,
            "python": sys.version,
            "python_executable": sys.executable,
            "rlimits": {
                "address_space": list(resource.getrlimit(resource.RLIMIT_AS)),
                "file_size": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
                "cpu_seconds": list(resource.getrlimit(resource.RLIMIT_CPU)),
            },
        },
        "rusage": {
            "before": before,
            "after": {
                "self": _usage(resource.RUSAGE_SELF),
                "children": _usage(resource.RUSAGE_CHILDREN),
            },
            "limits": "Self and reaped-child snapshots; RSS is a lifetime high-water mark, not aggregate host memory.",
        },
        "io_limits": "Warm/cache-uncontrolled buffered file I/O with per-file fsync; no physical-device bandwidth or crash-durability claim.",
    }
    archives["corpus_ledger"] = ledger_ref
    return observations, result, archives

def _bounded_process_limits() -> None:
    """Apply the Linux CLI contract without increasing inherited limits."""
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise ValueError("the standalone workload CLI requires Linux")
    if len(os.sched_getaffinity(0)) != 2:
        raise ValueError("the workload CLI requires exactly two logical CPUs in its affinity mask")
    for kind, maximum in ((resource.RLIMIT_AS, 512 * 1024 * 1024),
                          (resource.RLIMIT_FSIZE, MAX_OUTPUT_BYTES)):
        soft, hard = resource.getrlimit(kind)
        bounded_hard = maximum if hard == resource.RLIM_INFINITY else min(hard, maximum)
        bounded_soft = bounded_hard if soft == resource.RLIM_INFINITY else min(soft, bounded_hard)
        if bounded_soft <= 0 or bounded_hard <= 0:
            raise ValueError("the inherited process limit cannot support workload output")
        resource.setrlimit(kind, (bounded_soft, bounded_hard))
        actual_soft, actual_hard = resource.getrlimit(kind)
        if not 0 < actual_soft <= actual_hard <= maximum:
            raise ValueError("the required finite process limit was not applied")


def _public_ref(output: Path, reference: dict) -> dict:
    path = _plain(str(output / reference["artifactName"]))
    return {**reference, "sizeBytes": path.stat().st_size}


def _linux_host_context() -> dict:
    """Read actual Linux metadata before the core can label Linux RSS units."""
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise ValueError("the standalone workload requires Linux")
    cpu_models = sorted({line.split(":", 1)[1].strip()
                         for line in _read(Path("/proc/cpuinfo"), 2 * 1024 * 1024).decode().splitlines()
                         if line.startswith(("model name\t", "Hardware\t")) and ":" in line})
    return {"hostname": platform.node(), "process_id": os.getpid(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_models": cpu_models,
            "uname": list(platform.uname()), "logical_cpu_count": os.cpu_count()}


def run_workload(input_index: Path, output: Path, codec: str, toolchain_json: str,
                 timeout_seconds: int = 15, observer=None) -> dict:
    """Run the Linux core once. The CLI also supplies process resource admission."""
    host = _linux_host_context()
    if codec not in ("pigz-gzip", "zstd"):
        raise ValueError("codec must be pigz-gzip or zstd")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 15:
        raise ValueError("timeout_seconds must be an integer within [1,15]")
    input_index = _plain(str(input_index))
    output = _plain(str(output), directory=True)
    names = {"result.json", "corpus-ledger.json", "native-evidence.json", "workload-result.json"}
    for number in range(MAX_PARTS):
        for filename in (f"archive-{number:03d}.gz", f"archive-{number:03d}.zst",
                         f"roundtrip-{number:03d}.bin"):
            names.update((filename, filename + ".stderr.log", filename + ".stderr.json"))
    if any((output / name).exists() or (output / name).is_symlink() for name in names):
        raise ValueError("a workload output path already exists")
    toolchain = _toolchain(toolchain_json)
    observations, result, archives = _run_corpus(
        input_index, output, {"codec": codec}, {"timeoutSeconds": timeout_seconds},
        toolchain, host, observer)
    labels = {"observation_mode": "standalone_workload", "qualification": "not_assessed",
              "timing_scope": "adapter_local_workload_window"}
    observations.update(labels)
    evidence = _write(output / "native-evidence.json", _bytes(observations))
    evidence = _public_ref(output, evidence)
    payload = {**labels, "metric_name": "adapter_local_workload_elapsed",
               "metric_value": f"{observations['workload_elapsed_ns'] / 1_000_000:.9f}",
               "measurement_window": WINDOW, "result": _public_ref(output, result),
               "observations": evidence,
               "native_evidence": {**{name: _public_ref(output, ref) for name, ref in archives.items()},
                                   "observations": evidence}}
    _write(output / "workload-result.json", _bytes(payload))
    return payload


def _load_request(path: Path) -> dict:
    request = json.loads(_read(_plain(str(path)), MAX_INDEX_BYTES))
    fields = {"input_index", "output_directory", "codec", "toolchain_json", "timeout_seconds"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError("request must contain exactly the five workload fields")
    if any(not isinstance(request[key], str) for key in ("input_index", "output_directory", "codec", "toolchain_json")):
        raise ValueError("request paths, codec and toolchain_json must be strings")
    if request["codec"] not in ("pigz-gzip", "zstd"):
        raise ValueError("codec must be pigz-gzip or zstd")
    if type(request["timeout_seconds"]) is not int or not 1 <= request["timeout_seconds"] <= 15:
        raise ValueError("timeout_seconds must be an integer within [1,15]")
    return request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    _bounded_process_limits()
    request = _load_request(args.request)
    payload = run_workload(Path(request["input_index"]), Path(request["output_directory"]),
                           request["codec"], request["toolchain_json"], request["timeout_seconds"])
    print(_bytes(payload).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
