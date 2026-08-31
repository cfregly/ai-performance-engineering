#!/usr/bin/env python3
"""Verify the focused 90-specification Linux dependency-install gate.

This verifier deliberately separates package installation and CPU-safe import
provenance from CUDA driver, native-build, GPU, and performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ModuleNotFoundError:  # setup-python guarantees pip, not standalone packaging
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

EXPECTED_DIRECT_COUNT = 90
EXPECTED_LOCK_COUNT = 327
EXPECTED_GPUTIL_SHA256 = (
    "099e52c65e512cdfa8c8763fca67f5a5c2afb63469602d5dcb4d296b3661efb9"
)
ALLOWED_ORIGIN_HOSTS = {
    "download-r2.pytorch.org",
    "download.pytorch.org",
    "files.pythonhosted.org",
}
EXPECTED_INDEX_COUNTS = {"download.pytorch.org": 30, "pypi.org": 297}
INDEX_ARTIFACT_HOSTS = {
    "download.pytorch.org": {"download-r2.pytorch.org", "download.pytorch.org"},
    "pypi.org": {"files.pythonhosted.org"},
}
ALLOWED_EXTRA_DISTRIBUTIONS = {"pip"}

EXPECTED_VERSIONS = {
    "cuda-python": "13.0.3",
    "flashinfer-python": "0.6.3",
    "nvidia-cutlass-dsl": "4.5.2",
    "torch": "2.9.1+cu130",
    "torchao": "0.15.0+cu130",
    "torchaudio": "2.9.1+cu130",
    "torchvision": "0.24.1+cu130",
    "triton": "3.5.1",
    "vllm": "0.16.0",
}

SAFE_IMPORTS = {
    "GPUtil": "GPUtil",
    "anthropic": "anthropic",
    "click": "click",
    "coverage": "coverage",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "hypothesis": "hypothesis",
    "numpy": "numpy",
    "openai": "openai",
    "packaging": "packaging",
    "prometheus_client": "prometheus_client",
    "psutil": "psutil",
    "pydantic": "pydantic",
    "pynvml": "pynvml",
    "pytest": "pytest",
    "pytest_cov": "pytest_cov",
    "requests": "requests",
    "rich": "rich",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchao": "torchao",
    "triton": "triton",
    "typer": "typer",
    "uvicorn": "uvicorn",
    "yaml": "yaml",
}

DIAGNOSTIC_IMPORTS = {
    "cuda.bindings": "cuda.bindings",
    "cupy": "cupy",
    "cutlass": "cutlass",
    "flashinfer": "flashinfer",
    "kvikio": "kvikio",
    "torchaudio": "torchaudio",
    "torchtitan": "torchtitan",
    "torchvision": "torchvision",
    "vllm": "vllm",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def requirement_lines(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        stripped = raw_line.strip().split(" #", 1)[0].rstrip()
        if stripped.startswith("-"):
            continue
        requirement = Requirement(stripped.rstrip("\\").strip())
        name = canonicalize_name(requirement.name)
        if name in requirements:
            raise ValueError(f"duplicate requirement {name} in {path}")
        requirements[name] = requirement
    return requirements


def locked_requirements(
    path: Path,
) -> tuple[dict[str, Requirement], dict[str, set[str]]]:
    requirements: dict[str, Requirement] = {}
    hashes: dict[str, set[str]] = {}
    current_name: str | None = None
    hash_pattern = re.compile(r"--hash=sha256:([0-9a-f]{64})$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line[0].isspace():
            match = hash_pattern.fullmatch(stripped.rstrip("\\").strip())
            if match and current_name is not None:
                hashes[current_name].add(match.group(1))
            continue
        if stripped.startswith("-"):
            current_name = None
            continue
        requirement = Requirement(stripped.rstrip("\\").strip())
        current_name = canonicalize_name(requirement.name)
        if current_name in requirements:
            raise ValueError(f"duplicate locked requirement {current_name} in {path}")
        requirements[current_name] = requirement
        hashes[current_name] = set()
    return requirements, hashes


def locked_indexes(path: Path) -> dict[str, str]:
    indexes: dict[str, str] = {}
    current_name: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if raw_line and not raw_line[0].isspace() and not stripped.startswith(
            ("#", "--")
        ):
            current_name = canonicalize_name(
                Requirement(stripped.rstrip("\\").strip()).name
            )
            continue
        if current_name is not None and stripped.startswith("# from https://"):
            source_url = stripped.removeprefix("# from ")
            hostname = urlparse(source_url).hostname
            if hostname is None:
                raise ValueError(f"invalid lock source annotation: {source_url}")
            indexes[current_name] = hostname
    return indexes


def normalized_requirement(requirement: Requirement) -> dict[str, object]:
    return {
        "extras": sorted(requirement.extras),
        "marker": str(requirement.marker) if requirement.marker else None,
        "name": canonicalize_name(requirement.name),
        "specifier": str(requirement.specifier),
        "url": requirement.url,
    }


def prepare_install_locks(source: Path, wheel_destination: Path, gputil_destination: Path) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    only_binary = "--only-binary :all:"
    no_binary = "--no-binary gputil"
    if lines.count(only_binary) != 1 or lines.count(no_binary) != 1:
        raise ValueError(
            "reviewed lock must contain exactly one global only-binary line and "
            "one GPUtil no-binary line"
        )
    gputil_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lower().startswith("gputil==")
        ),
        None,
    )
    if gputil_start is None:
        raise ValueError("reviewed lock has no GPUtil requirement")
    gputil_end = len(lines)
    for index in range(gputil_start + 1, len(lines)):
        line = lines[index]
        if line and not line[0].isspace() and not line.startswith(("#", "--")):
            gputil_end = index
            break

    wheel_lines = [
        line
        for index, line in enumerate(lines)
        if line != no_binary and not (gputil_start <= index < gputil_end)
    ]
    wheel_destination.parent.mkdir(parents=True, exist_ok=True)
    wheel_destination.write_text("\n".join(wheel_lines) + "\n", encoding="utf-8")

    _, hashes = locked_requirements(source)
    gputil_hashes = sorted(hashes.get("gputil", set()))
    if gputil_hashes != [EXPECTED_GPUTIL_SHA256]:
        raise ValueError(f"unexpected GPUtil hashes: {gputil_hashes}")
    gputil_lines = [
        "--index-url https://pypi.org/simple",
        only_binary,
        no_binary,
        "",
        "gputil==1.4.0 \\",
        f"    --hash=sha256:{EXPECTED_GPUTIL_SHA256}",
    ]
    gputil_destination.parent.mkdir(parents=True, exist_ok=True)
    gputil_destination.write_text("\n".join(gputil_lines) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock).resolve()
    direct_path = Path(args.direct).resolve()
    current_path = Path(args.current_requirements).resolve()
    output_dir = Path(args.output_dir).resolve()
    wheel_lock_path = Path(args.wheel_lock_output).resolve()
    gputil_lock_path = Path(args.gputil_lock_output).resolve()

    locked, hashes = locked_requirements(lock_path)
    indexes = locked_indexes(lock_path)
    direct = requirement_lines(direct_path)
    current = requirement_lines(current_path)
    errors: list[str] = []

    if len(locked) != EXPECTED_LOCK_COUNT:
        errors.append(
            f"expected {EXPECTED_LOCK_COUNT} locked distributions, found {len(locked)}"
        )
    if len(direct) != EXPECTED_DIRECT_COUNT:
        errors.append(
            f"expected {EXPECTED_DIRECT_COUNT} direct requirements, found {len(direct)}"
        )
    if set(direct) - set(locked):
        errors.append(
            f"direct requirements absent from lock: {sorted(set(direct) - set(locked))}"
        )
    if set(current) != set(direct):
        errors.append(
            "current/direct requirement names differ: "
            f"current_only={sorted(set(current) - set(direct))}, "
            f"snapshot_only={sorted(set(direct) - set(current))}"
        )
    for name in sorted(set(current) & set(direct)):
        if normalized_requirement(current[name]) != normalized_requirement(direct[name]):
            errors.append(f"current/direct requirement differs for {name}")
    missing_hashes = sorted(name for name, values in hashes.items() if not values)
    if missing_hashes:
        errors.append(f"locked requirements without SHA-256 values: {missing_hashes}")
    if EXPECTED_GPUTIL_SHA256 not in hashes.get("gputil", set()):
        errors.append("reviewed GPUtil source SHA-256 is absent from the lock")
    index_counts = {
        hostname: sum(value == hostname for value in indexes.values())
        for hostname in sorted(set(indexes.values()))
    }
    if len(indexes) != EXPECTED_LOCK_COUNT or index_counts != EXPECTED_INDEX_COUNTS:
        errors.append(
            f"unexpected logical index annotations: count={len(indexes)}, "
            f"hosts={index_counts}"
        )
    try:
        prepare_install_locks(lock_path, wheel_lock_path, gputil_lock_path)
        wheel_locked, wheel_hashes = locked_requirements(wheel_lock_path)
        gputil_locked, gputil_hashes = locked_requirements(gputil_lock_path)
        combined_locked = {**wheel_locked, **gputil_locked}
        combined_hashes = {**wheel_hashes, **gputil_hashes}
        if set(wheel_locked) & set(gputil_locked):
            errors.append("derived wheel and GPUtil install locks overlap")
        if len(wheel_locked) != EXPECTED_LOCK_COUNT - 1 or set(gputil_locked) != {
            "gputil"
        }:
            errors.append(
                "derived install lock counts differ from the reviewed 326+1 split"
            )
        if {
            name: normalized_requirement(requirement)
            for name, requirement in combined_locked.items()
        } != {
            name: normalized_requirement(requirement)
            for name, requirement in locked.items()
        } or combined_hashes != hashes:
            errors.append("derived install locks changed requirements or hashes")
    except (OSError, ValueError) as error:
        errors.append(f"unable to derive pip-compatible install lock: {error}")

    summary = {
        "current_requirements": str(current_path),
        "current_requirements_sha256": sha256(current_path),
        "direct_count": len(direct),
        "direct_path": str(direct_path),
        "direct_sha256": sha256(direct_path),
        "errors": errors,
        "gputil_hashes": sorted(hashes.get("gputil", set())),
        "gputil_lock_path": str(gputil_lock_path),
        "gputil_lock_sha256": (
            sha256(gputil_lock_path) if gputil_lock_path.is_file() else None
        ),
        "lock_count": len(locked),
        "logical_index_counts": index_counts,
        "lock_path": str(lock_path),
        "lock_sha256": sha256(lock_path),
        "result": "PASS" if not errors else "FAIL",
        "wheel_lock_path": str(wheel_lock_path),
        "wheel_lock_sha256": (
            sha256(wheel_lock_path) if wheel_lock_path.is_file() else None
        ),
    }
    write_json(output_dir / "lock-preflight.json", summary)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def selected_sha256(item: dict[str, object]) -> str | None:
    download = item.get("download_info")
    if not isinstance(download, dict):
        return None
    archive = download.get("archive_info")
    if not isinstance(archive, dict):
        return None
    archive_hashes = archive.get("hashes")
    if isinstance(archive_hashes, dict):
        value = archive_hashes.get("sha256")
        if isinstance(value, str):
            return value
    legacy_hash = archive.get("hash")
    if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
        return legacy_hash.split("=", 1)[1]
    return None


def run_import_probe(
    label: str, module: str, output_dir: Path, *, timeout: int = 30
) -> dict[str, object]:
    code = """
import importlib
import json
import pathlib
import sys

module = importlib.import_module(sys.argv[1])
module_file = getattr(module, "__file__", None)
if module_file:
    module_path = pathlib.Path(module_file).resolve()
    prefix = pathlib.Path(sys.prefix).resolve()
    if not module_path.is_relative_to(prefix):
        raise SystemExit(f"module loaded outside environment: {module_path}")
print(json.dumps({"module": sys.argv[1], "module_file": module_file}))
"""
    log_path = output_dir / "import-logs" / f"{label.replace('.', '_')}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code, module],
            check=False,
            cwd="/",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = completed.stdout
        result: dict[str, object] = {
            "exit_code": completed.returncode,
            "label": label,
            "module": module,
            "result": "PASS" if completed.returncode == 0 else "FAIL",
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        def timeout_text(value: str | bytes | None) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value or ""

        output = timeout_text(error.stdout) + timeout_text(error.stderr)
        result = {
            "exit_code": None,
            "label": label,
            "module": module,
            "result": "TIMEOUT",
            "timed_out": True,
        }
    log_path.write_text(output, encoding="utf-8")
    result["log"] = str(log_path)
    return result


def verify_torch_cpu(output_dir: Path) -> dict[str, object]:
    code = """
import json
import torch

result = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "tensor_sum": torch.tensor([1.0, 2.0, 3.0]).sum().item(),
    "torch_version": torch.__version__,
}
print(json.dumps(result, sort_keys=True))
if result != {
    "cuda_available": False,
    "cuda_version": "13.0",
    "tensor_sum": 6.0,
    "torch_version": "2.9.1+cu130",
}:
    raise SystemExit(f"unexpected CPU-runner Torch result: {result}")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        cwd="/",
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    log_path = output_dir / "torch-cpu-validation.txt"
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "exit_code": completed.returncode,
        "log": str(log_path),
        "result": "PASS" if completed.returncode == 0 else "FAIL",
    }


def installed(args: argparse.Namespace) -> int:
    lock_path = Path(args.lock).resolve()
    report_paths = [Path(path).resolve() for path in args.pip_report]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    locked, allowed_hashes = locked_requirements(lock_path)
    expected_indexes = locked_indexes(lock_path)
    distributions = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    installed_names = set(distributions)
    expected_names = set(locked)
    errors: list[str] = []
    missing = sorted(expected_names - installed_names)
    unexpected = sorted(
        installed_names - expected_names - ALLOWED_EXTRA_DISTRIBUTIONS
    )
    if missing or unexpected:
        errors.append(
            f"installed distribution mismatch: missing={missing}, unexpected={unexpected}"
        )

    versions: dict[str, str] = {}
    for name in sorted(expected_names & installed_names):
        version = distributions[name].version
        versions[name] = version
        requirement = locked[name]
        if requirement.specifier and not requirement.specifier.contains(
            version, prereleases=True
        ):
            errors.append(
                f"version mismatch for {name}: {version} not in {requirement.specifier}"
            )
    for name, expected_version in EXPECTED_VERSIONS.items():
        if versions.get(name) != expected_version:
            errors.append(
                f"unexpected {name} version: {versions.get(name)!r} != {expected_version!r}"
            )

    selected: list[dict[str, object]] = []
    report_names: set[str] = set()
    phase_names: list[set[str]] = []
    if len(report_paths) != 2:
        errors.append(f"expected two pip reports, found {len(report_paths)}")
    for phase_index, report_path in enumerate(report_paths):
        phase = "wheel" if phase_index == 0 else "gputil"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        current_phase: set[str] = set()
        for item in report.get("install", []):
            if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
                errors.append(f"malformed {phase} pip report item")
                continue
            metadata = item["metadata"]
            name = canonicalize_name(str(metadata.get("name", "")))
            if name in report_names:
                errors.append(f"distribution appears in multiple pip reports: {name}")
            report_names.add(name)
            current_phase.add(name)
            selected_hash = selected_sha256(item)
            download = item.get("download_info")
            origin_url = download.get("url") if isinstance(download, dict) else None
            parsed = urlparse(origin_url) if isinstance(origin_url, str) else None
            if selected_hash is None:
                errors.append(f"missing selected SHA-256 for {name}")
            elif selected_hash not in allowed_hashes.get(name, set()):
                errors.append(f"selected SHA-256 for {name} is absent from the lock")
            if (
                parsed is None
                or parsed.scheme != "https"
                or parsed.hostname not in ALLOWED_ORIGIN_HOSTS
            ):
                errors.append(f"unapproved selected origin for {name}: {origin_url!r}")
            logical_index = expected_indexes.get(name)
            if (
                parsed is None
                or logical_index not in INDEX_ARTIFACT_HOSTS
                or parsed.hostname not in INDEX_ARTIFACT_HOSTS[logical_index]
            ):
                errors.append(
                    f"selected origin for {name} does not match its locked logical "
                    f"index {logical_index!r}: {origin_url!r}"
                )
            selected.append(
                {
                    "logical_index": logical_index,
                    "name": name,
                    "phase": phase,
                    "requested": bool(item.get("requested")),
                    "sha256": selected_hash,
                    "url": origin_url,
                    "version": metadata.get("version"),
                }
            )
        phase_names.append(current_phase)
    if len(phase_names) == 2 and (
        len(phase_names[0]) != EXPECTED_LOCK_COUNT - 1
        or phase_names[1] != {"gputil"}
    ):
        errors.append(
            "pip report phases differ from the reviewed 326-wheel plus GPUtil split"
        )
    if report_names != expected_names:
        errors.append(
            "pip report distribution mismatch: "
            f"missing={sorted(expected_names - report_names)}, "
            f"unexpected={sorted(report_names - expected_names)}"
        )
    gputil = next((item for item in selected if item["name"] == "gputil"), None)
    if gputil is None or gputil["sha256"] != EXPECTED_GPUTIL_SHA256:
        errors.append(f"unexpected GPUtil selection: {gputil}")

    safe_imports = [
        run_import_probe(label, module, output_dir)
        for label, module in SAFE_IMPORTS.items()
    ]
    safe_failures = [item for item in safe_imports if item["result"] != "PASS"]
    if safe_failures:
        errors.append(
            "required CPU-safe imports failed: "
            + ", ".join(str(item["label"]) for item in safe_failures)
        )
    diagnostics = [
        run_import_probe(label, module, output_dir)
        for label, module in DIAGNOSTIC_IMPORTS.items()
    ]
    torch_cpu = verify_torch_cpu(output_dir)
    if torch_cpu["result"] != "PASS":
        errors.append("CUDA Torch identity/CPU-operation check failed")

    write_json(output_dir / "selected-artifacts.json", selected)
    write_json(output_dir / "safe-imports.json", safe_imports)
    write_json(output_dir / "diagnostic-imports.json", diagnostics)
    summary = {
        "diagnostic_import_failures_or_holds": [
            item["label"] for item in diagnostics if item["result"] != "PASS"
        ],
        "errors": errors,
        "expected_distribution_count": len(expected_names),
        "installed_distribution_count": len(expected_names & installed_names),
        "missing_distributions": missing,
        "pip_report_distribution_count": len(report_names),
        "result": "PASS" if not errors else "FAIL",
        "safe_import_count": len(safe_imports),
        "selected_artifact_count": len(selected),
        "torch_cpu": torch_cpu,
        "unexpected_distributions": unexpected,
        "versions": {name: versions.get(name) for name in EXPECTED_VERSIONS},
    }
    write_json(output_dir / "installed-validation.json", summary)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--lock", required=True)
    preflight_parser.add_argument("--direct", required=True)
    preflight_parser.add_argument("--current-requirements", required=True)
    preflight_parser.add_argument("--wheel-lock-output", required=True)
    preflight_parser.add_argument("--gputil-lock-output", required=True)
    preflight_parser.add_argument("--output-dir", required=True)

    installed_parser = subparsers.add_parser("installed")
    installed_parser.add_argument("--lock", required=True)
    installed_parser.add_argument("--pip-report", action="append", required=True)
    installed_parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return preflight(args)
    if args.command == "installed":
        return installed(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
