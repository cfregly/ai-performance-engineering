#!/usr/bin/env python3
"""Build the pinned vLLM CUDA wheel against an existing PyTorch install.

The build runs only in a dedicated virtual environment. It verifies the
upstream commit and selected source files, applies vLLM's own
``tools/use_existing_torch.py --prefix`` transformation, and builds native
CUDA and Rust extensions without build isolation. The script does not install
the resulting wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet

DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "vllm_source_build.json"
RECEIPT_NAME = "vllm-source-wheel.manifest.json"
SOURCE_SUBDIR = Path("source") / "vllm"
WHEEL_SUBDIR = Path("wheel")

TORCH_REQUIREMENT_PREFIXES = (
    "torch=",
    "torchvision=",
    "torchaudio=",
    "torchcodec=",
    '"torch =',
    '"torchvision =',
    '"torchaudio =',
    '"torchcodec =',
)


class BuildContractError(RuntimeError):
    """Raised when the source or build environment violates the contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildContractError(f"Cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BuildContractError(f"Build contract must be a JSON object: {path}")
    return payload


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise BuildContractError(
            f"Command failed with exit code {exc.returncode}: {rendered}"
        ) from exc


def _capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout.strip() if exc.stdout else "no output"
        raise BuildContractError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}\n{output}"
        ) from exc
    return completed.stdout.strip()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise BuildContractError(f"Required distribution is not installed: {name}") from exc


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != "aisp.vllm-source-build.v1":
        raise BuildContractError("Unsupported vLLM source-build contract schema")
    for section in (
        "source",
        "package",
        "runtime",
        "build",
        "source_dependencies",
        "upstream_files",
        "expected_metadata",
    ):
        if not isinstance(contract.get(section), dict):
            raise BuildContractError(f"Build contract is missing object section: {section}")

    expected_header_probe = {
        "compiler": "g++",
        "language": "c++",
        "standard": "c++20",
        "headers": ["elfutils/libdwfl.h"],
        "debian_packages": ["libdw-dev", "libelf-dev"],
    }
    if contract["build"].get("required_cxx_header_probe") != expected_header_probe:
        raise BuildContractError(
            "The vLLM build must preprocess the reviewed elfutils C++ header probe."
        )


def _validate_required_cxx_headers(
    contract: Mapping[str, Any],
    command_paths: Mapping[str, str],
    *,
    env: Mapping[str, str],
) -> dict[str, Any]:
    probe = contract["build"]["required_cxx_header_probe"]
    compiler_name = str(probe["compiler"])
    compiler = command_paths.get(compiler_name)
    if compiler is None:
        raise BuildContractError(f"Required header-probe compiler is unavailable: {compiler_name}")
    headers = [str(header) for header in probe["headers"]]
    source = "".join(f"#include <{header}>\n" for header in headers)
    command = [
        compiler,
        f"-std={probe['standard']}",
        "-x",
        str(probe["language"]),
        "-E",
        "-",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            env=env,
            input=source,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        packages = ", ".join(str(value) for value in probe["debian_packages"])
        detail = (exc.stderr or "no compiler diagnostic").strip()[-4000:]
        raise BuildContractError(
            "Required C++ header preprocessing failed for "
            f"{', '.join(headers)}. Install {packages} or provide their verified "
            f"include roots. Compiler diagnostic:\n{detail}"
        ) from exc
    return {
        "command": command,
        "headers": headers,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "debian_packages": [str(value) for value in probe["debian_packages"]],
    }


def _validate_environment(
    contract: Mapping[str, Any],
    *,
    command_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    runtime = contract["runtime"]
    process_environment = dict(command_environment or os.environ)
    expected_system = str(runtime["platform_system"])
    expected_machine = str(runtime["platform_machine"])
    expected_python = str(runtime["python"])

    for name, expected_value in contract["build"]["required_host_environment"].items():
        actual_value = process_environment.get(str(name))
        if actual_value != expected_value:
            raise BuildContractError(f"Expected {name}={expected_value!r}, found {actual_value!r}.")

    actual_system = platform.system()
    actual_machine = platform.machine()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_system != expected_system:
        raise BuildContractError(
            f"Expected {expected_system}, found {actual_system}. Build on the target host."
        )
    if actual_machine != expected_machine:
        raise BuildContractError(
            f"Expected {expected_machine}, found {actual_machine}. Build on the target host."
        )
    if actual_python != expected_python:
        raise BuildContractError(f"Expected Python {expected_python}, found {actual_python}.")
    if sys.prefix == sys.base_prefix:
        raise BuildContractError(
            "The vLLM source build must run in a dedicated virtual environment."
        )

    try:
        import torch
    except Exception as exc:
        raise BuildContractError(f"Cannot import the required PyTorch build: {exc}") from exc

    expected_torch = str(runtime["torch"])
    if torch.__version__ != expected_torch:
        raise BuildContractError(
            f"Expected torch=={expected_torch}, found torch=={torch.__version__}."
        )
    expected_cuda = str(runtime["torch_cuda"])
    actual_cuda = str(torch.version.cuda or "")
    if actual_cuda != expected_cuda:
        raise BuildContractError(
            f"Expected torch.version.cuda {expected_cuda}, found {actual_cuda or 'none'}."
        )

    checked_distributions = {
        "triton": str(runtime["triton"]),
        "torchvision": str(runtime["torchvision"]),
        "torchaudio": str(runtime["torchaudio"]),
        "torchao": str(runtime["torchao"]),
        "flashinfer-python": str(runtime["flashinfer_python"]),
    }
    actual_distributions: dict[str, str] = {}
    for name, expected in checked_distributions.items():
        actual = _distribution_version(name)
        if actual != expected:
            raise BuildContractError(f"Expected {name}=={expected}, found {name}=={actual}.")
        actual_distributions[name] = actual

    resolved_requirement_groups = (
        contract["build"]["resolved_python_requirements"],
        contract["build"]["resolved_python_transitive_requirements"],
    )
    for requirement_group in resolved_requirement_groups:
        for name, expected_value in requirement_group.items():
            expected = str(expected_value)
            actual = _distribution_version(str(name))
            if actual != expected:
                raise BuildContractError(f"Expected {name}=={expected}, found {name}=={actual}.")
            actual_distributions[str(name)] = actual

    command_paths: dict[str, str] = {}
    for command in contract["build"]["required_commands"]:
        path = shutil.which(str(command), path=process_environment.get("PATH"))
        if path is None:
            raise BuildContractError(f"Required build command is unavailable: {command}")
        command_paths[str(command)] = path

    required_cxx_headers = _validate_required_cxx_headers(
        contract, command_paths, env=process_environment
    )

    gcc_version = _capture(
        [command_paths["gcc"], "-dumpfullversion", "-dumpversion"],
        env=process_environment,
    )
    gcc_match = re.match(r"(\d+)", gcc_version)
    if gcc_match is None:
        raise BuildContractError(f"Could not parse GCC version: {gcc_version!r}")
    minimum_gcc = int(contract["build"]["minimum_gcc_major"])
    if int(gcc_match.group(1)) < minimum_gcc:
        raise BuildContractError(f"vLLM requires GCC {minimum_gcc} or newer, found {gcc_version}.")

    nvcc_version = _capture([command_paths["nvcc"], "--version"], env=process_environment)
    nvcc_release = re.search(r"release\s+(\d+\.\d+)", nvcc_version)
    if nvcc_release is None or nvcc_release.group(1) != expected_cuda:
        found = nvcc_release.group(1) if nvcc_release else "unknown"
        raise BuildContractError(f"Expected nvcc release {expected_cuda}, found {found}.")

    rustc_version = _capture([command_paths["rustc"], "--version"], env=process_environment)
    cargo_version = _capture([command_paths["cargo"], "--version"], env=process_environment)
    expected_rust = str(contract["build"]["rust_toolchain"])
    if not rustc_version.startswith(f"rustc {expected_rust} "):
        raise BuildContractError(f"Expected rustc {expected_rust}, found {rustc_version}.")
    if not cargo_version.startswith(f"cargo {expected_rust} "):
        raise BuildContractError(f"Expected cargo {expected_rust}, found {cargo_version}.")

    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": actual_system,
        "platform_machine": actual_machine,
        "required_host_environment": {
            str(name): process_environment[str(name)]
            for name in contract["build"]["required_host_environment"]
        },
        "torch": torch.__version__,
        "torch_cuda": actual_cuda,
        "distributions": actual_distributions,
        "commands": command_paths,
        "required_cxx_header_probe": required_cxx_headers,
        "gcc_version": gcc_version,
        "nvcc_version": nvcc_version,
        "cmake_version": _capture(
            [command_paths["cmake"], "--version"], env=process_environment
        ).splitlines()[0],
        "ninja_version": _capture([command_paths["ninja"], "--version"], env=process_environment),
        "rustc_version": rustc_version,
        "cargo_version": cargo_version,
    }


def _checkout_repository(
    source: Mapping[str, Any], source_dir: Path, *, label: str
) -> dict[str, Any]:
    repository = str(source["repository"])
    commit = str(source["commit"])
    expected_tree = str(source["tree"])

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(source_dir),
        ]
    )
    _run(["git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", commit])
    _run(
        [
            "git",
            "-C",
            str(source_dir),
            "-c",
            "advice.detachedHead=false",
            "switch",
            "--detach",
            commit,
        ]
    )

    actual_commit = _capture(["git", "rev-parse", "HEAD"], cwd=source_dir)
    actual_tree = _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir)
    if actual_commit != commit:
        raise BuildContractError(
            f"Checked-out {label} commit mismatch: expected {commit}, found {actual_commit}."
        )
    if actual_tree != expected_tree:
        raise BuildContractError(
            f"Checked-out {label} tree mismatch: expected {expected_tree}, found {actual_tree}."
        )

    verified_files: dict[str, str] = {}
    for relative, expected_sha256 in source.get("verified_files", {}).items():
        path = source_dir / relative
        if not path.is_file():
            raise BuildContractError(f"Pinned upstream file is missing: {relative}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise BuildContractError(
                f"Pinned upstream file hash mismatch for {relative}: "
                f"expected {expected_sha256}, found {actual_sha256}."
            )
        verified_files[str(relative)] = actual_sha256

    return {
        "repository": repository,
        "commit": actual_commit,
        "tree": actual_tree,
        "verified_files": verified_files,
    }


def _checkout_source(contract: Mapping[str, Any], source_dir: Path) -> dict[str, Any]:
    source = dict(contract["source"])
    source["verified_files"] = contract["upstream_files"]
    return _checkout_repository(source, source_dir, label="vLLM")


def _apply_existing_torch_transform(contract: Mapping[str, Any], source_dir: Path) -> list[str]:
    relative_script = Path(str(contract["build"]["use_existing_torch_script"]))
    script = source_dir / relative_script
    arguments = [str(value) for value in contract["build"]["use_existing_torch_arguments"]]
    _run([sys.executable, str(script), *arguments], cwd=source_dir)

    candidates = [source_dir / "pyproject.toml"]
    candidates.extend((source_dir / "requirements").rglob("*.txt"))
    candidates.extend((source_dir / "requirements").rglob("*.in"))
    remaining: list[str] = []
    for path in candidates:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = raw_line.lower().strip()
            if stripped.startswith(TORCH_REQUIREMENT_PREFIXES):
                remaining.append(f"{path.relative_to(source_dir)}:{line_number}:{raw_line}")
    if remaining:
        raise BuildContractError(
            "vLLM's use_existing_torch transform left torch requirements behind:\n"
            + "\n".join(remaining)
        )

    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=source_dir, text=True, capture_output=True
    )
    if diff_check.returncode != 0:
        raise BuildContractError(
            "The use_existing_torch transform produced an invalid diff:\n"
            + (diff_check.stdout or diff_check.stderr)
        )
    modified = _capture(["git", "diff", "--name-only"], cwd=source_dir).splitlines()
    unexpected = [
        path
        for path in modified
        if path != "pyproject.toml" and not path.startswith("requirements/")
    ]
    if unexpected:
        raise BuildContractError(
            "The use_existing_torch transform changed unexpected paths: " + ", ".join(unexpected)
        )
    return modified


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _canonical_requirement(requirement: str) -> tuple[str, ...]:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement as exc:
        raise BuildContractError(f"Cannot parse wheel requirement: {requirement!r}") from exc
    return (
        _normalise_distribution_name(parsed.name),
        ",".join(sorted(_normalise_distribution_name(extra) for extra in parsed.extras)),
        str(parsed.specifier),
        parsed.url or "",
        str(parsed.marker) if parsed.marker is not None else "",
    )


def _inspect_wheel(contract: Mapping[str, Any], wheel_path: Path) -> dict[str, Any]:
    with ZipFile(wheel_path) as archive:
        members = archive.namelist()
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise BuildContractError(
                f"Expected one METADATA file in {wheel_path.name}, found {metadata_members}."
            )
        metadata_text = archive.read(metadata_members[0]).decode("utf-8")

        expected = contract["expected_metadata"]
        required_native_members: list[str] = []
        for prefix in expected["required_native_member_prefixes"]:
            matches = [
                name for name in members if name.startswith(f"{prefix}.") and name.endswith(".so")
            ]
            if len(matches) != 1:
                raise BuildContractError(
                    f"Wheel must contain exactly one native member with prefix {prefix!r}, "
                    f"found {matches}."
                )
            required_native_members.extend(matches)
        for member in expected["required_archive_members"]:
            if member not in members:
                raise BuildContractError(f"Wheel is missing a required archive member {member!r}.")
            required_native_members.append(str(member))

        minimum_native_size = int(expected["minimum_native_member_size"])
        native_member_details: dict[str, dict[str, Any]] = {}
        for member in required_native_members:
            info = archive.getinfo(member)
            if info.file_size < minimum_native_size:
                raise BuildContractError(
                    f"Native member {member!r} is only {info.file_size} bytes. "
                    f"Expected at least {minimum_native_size} bytes."
                )
            member_digest = hashlib.sha256()
            with archive.open(info) as handle:
                elf_header = handle.read(20)
                member_digest.update(elf_header)
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    member_digest.update(chunk)
            if (
                len(elf_header) < 20
                or elf_header[:4] != b"\x7fELF"
                or elf_header[4] != 2
                or elf_header[5] != 1
                or int.from_bytes(elf_header[18:20], "little") != 62
            ):
                raise BuildContractError(
                    f"Native member {member!r} is not an ELF64 little-endian x86_64 binary."
                )
            native_member_details[member] = {
                "size": info.file_size,
                "sha256": member_digest.hexdigest(),
                "elf_class": "ELF64",
                "byte_order": "little-endian",
                "machine": "x86_64",
                "unix_mode": oct(info.external_attr >> 16),
            }
        executable_member = str(expected["required_executable_member"])
        executable_mode = archive.getinfo(executable_member).external_attr >> 16
        if executable_mode & 0o111 == 0:
            raise BuildContractError(
                f"Required native member {executable_member!r} is not executable."
            )

    metadata = Parser().parsestr(metadata_text)
    name = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", ""))
    requires_python = str(metadata.get("Requires-Python", ""))
    requires_dist = metadata.get_all("Requires-Dist", [])
    if _normalise_distribution_name(name) != _normalise_distribution_name(str(expected["name"])):
        raise BuildContractError(f"Wheel name mismatch: expected vllm, found {name!r}.")
    if version != expected["version"]:
        raise BuildContractError(
            f"Wheel version mismatch: expected {expected['version']}, found {version}."
        )
    try:
        actual_python_specifier = SpecifierSet(requires_python)
        expected_python_specifier = SpecifierSet(str(expected["requires_python"]))
    except InvalidSpecifier as exc:
        raise BuildContractError(f"Invalid wheel Python requirement: {exc}") from exc
    if actual_python_specifier != expected_python_specifier:
        raise BuildContractError(
            "Wheel Requires-Python mismatch: "
            f"expected {expected['requires_python']}, found {requires_python!r}."
        )

    actual_requirements = Counter(_canonical_requirement(value) for value in requires_dist)
    expected_requirements = Counter(
        _canonical_requirement(value) for value in expected["requires_dist"]
    )
    dependency_names = {value[0] for value in actual_requirements}
    forbidden = {
        _normalise_distribution_name(value) for value in expected["forbidden_dependency_names"]
    }
    retained_forbidden = sorted(dependency_names & forbidden)
    if retained_forbidden:
        raise BuildContractError(
            "Wheel metadata retained torch-family dependencies after the upstream transform: "
            + ", ".join(retained_forbidden)
        )
    if actual_requirements != expected_requirements:
        missing = list((expected_requirements - actual_requirements).elements())
        unexpected = list((actual_requirements - expected_requirements).elements())
        raise BuildContractError(
            "Wheel Requires-Dist metadata differs from the source-build contract. "
            f"Missing canonical requirements: {missing}. "
            f"Unexpected canonical requirements: {unexpected}."
        )

    native_members = sorted(required_native_members)

    if "x86_64" not in wheel_path.name:
        raise BuildContractError(f"Expected an x86_64 wheel filename, found {wheel_path.name}.")
    return {
        "path": str(wheel_path.resolve()),
        "filename": wheel_path.name,
        "size": wheel_path.stat().st_size,
        "sha256": _sha256_file(wheel_path),
        "metadata": {
            "name": name,
            "version": version,
            "requires_python": requires_python,
            "requires_dist": requires_dist,
            "metadata_member": metadata_members[0],
        },
        "native_members": native_members,
        "native_member_details": native_member_details,
    }


def _create_build_environment(
    contract: Mapping[str, Any],
    source_dependency_environment: Mapping[str, str],
    *,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int | None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    build_environment = dict(os.environ if base_environment is None else base_environment)
    build_environment.update(
        {str(key): str(value) for key, value in contract["build"]["environment"].items()}
    )
    build_environment.update(
        {str(key): str(value) for key, value in source_dependency_environment.items()}
    )
    build_environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "TORCH_CUDA_ARCH_LIST": cuda_arch_list,
            "MAX_JOBS": str(max_jobs),
            "CARGO_BUILD_JOBS": str(max_jobs),
            "VLLM_VERSION_OVERRIDE": str(contract["package"]["version"]),
            "VLLM_SKIP_VERSION_SUFFIX": "1",
        }
    )
    if nvcc_threads is not None:
        build_environment["NVCC_THREADS"] = str(nvcc_threads)
    else:
        build_environment.pop("NVCC_THREADS", None)
    return build_environment


def _create_build_command(wheel_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheel_dir),
        ".",
    ]


def _build(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    output_dir: Path,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int | None,
) -> Path:
    if not re.fullmatch(r"\d+\.\d+(;\d+\.\d+)*", cuda_arch_list):
        raise BuildContractError("--cuda-arch-list must contain values such as 10.0 or 10.0;10.3.")
    receipt_path = output_dir / RECEIPT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output_dir.iterdir())
    if existing:
        raise BuildContractError(
            f"Output directory is not empty and has no completed receipt: {output_dir}. "
            "Choose a fresh output directory."
        )

    environment = _validate_environment(contract)
    source_dir = output_dir / SOURCE_SUBDIR
    source_receipt = _checkout_source(contract, source_dir)
    modified_paths = _apply_existing_torch_transform(contract, source_dir)

    source_dependency_receipts: dict[str, Any] = {}
    source_dependency_environment: dict[str, str] = {}
    dependency_root = output_dir / "source-dependencies"
    for name, dependency in contract["source_dependencies"].items():
        dependency_dir = dependency_root / str(name)
        dependency_receipt = _checkout_repository(dependency, dependency_dir, label=str(name))
        source_subdirectory = str(dependency.get("source_subdirectory", ""))
        environment_path = dependency_dir / source_subdirectory
        if not environment_path.is_dir():
            raise BuildContractError(
                f"Pinned {name} source directory is missing: {environment_path}"
            )
        environment_variable = str(dependency["environment_variable"])
        source_dependency_environment[environment_variable] = str(environment_path.resolve())
        source_dependency_receipts[str(name)] = {
            **dependency_receipt,
            "path": str(dependency_dir.resolve()),
            "environment_variable": environment_variable,
            "environment_path": str(environment_path.resolve()),
        }

    wheel_dir = output_dir / WHEEL_SUBDIR
    wheel_dir.mkdir(parents=True)
    build_environment = _create_build_environment(
        contract,
        source_dependency_environment,
        cuda_arch_list=cuda_arch_list,
        max_jobs=max_jobs,
        nvcc_threads=nvcc_threads,
    )
    build_command = _create_build_command(wheel_dir)
    _run(build_command, cwd=source_dir, env=build_environment)

    wheels = sorted(wheel_dir.glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise BuildContractError(
            f"Expected exactly one built vLLM wheel in {wheel_dir}, found {wheels}."
        )
    wheel = _inspect_wheel(contract, wheels[0])
    receipt = {
        "schema": "aisp.vllm-source-wheel-receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256_file(contract_path),
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "source": {
            **source_receipt,
            "path": str(source_dir.resolve()),
            "modified_paths": modified_paths,
            "use_existing_torch": {
                "script": contract["build"]["use_existing_torch_script"],
                "arguments": contract["build"]["use_existing_torch_arguments"],
            },
        },
        "source_dependencies": source_dependency_receipts,
        "runtime": environment,
        "build": {
            "environment": {
                key: build_environment[key]
                for key in (
                    "CMAKE_BUILD_TYPE",
                    "MAX_JOBS",
                    "NVCC_THREADS",
                    "TORCH_CUDA_ARCH_LIST",
                    "VLLM_REQUIRE_RUST_FRONTEND",
                    "VLLM_SKIP_VERSION_SUFFIX",
                    "VLLM_TARGET_DEVICE",
                    "VLLM_USE_PRECOMPILED",
                    "VLLM_USE_PRECOMPILED_RUST",
                    "VLLM_VERSION_OVERRIDE",
                    "VLLM_CUTLASS_SRC_DIR",
                    "TRITON_KERNELS_SRC_DIR",
                )
                if key in build_environment
            },
            "command": build_command,
        },
        "wheel": wheel,
    }
    temporary_receipt = receipt_path.with_suffix(".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_receipt.replace(receipt_path)
    print(f"Built wheel: {wheel['path']}")
    print(f"Wheel SHA-256: {wheel['sha256']}")
    print(f"Receipt: {receipt_path}")
    return Path(wheel["path"])


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pinned vLLM CUDA wheel against the exact PyTorch version "
            "declared in code/vllm_source_build.json."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh directory for the source checkout, wheel, and build receipt.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
        help=f"Source-build contract. Default: {DEFAULT_CONTRACT}",
    )
    parser.add_argument(
        "--cuda-arch-list",
        default=None,
        help="CUDA architectures to compile, such as 10.0 or 10.0;10.3.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=4,
        help="Maximum parallel build jobs. Default: 4.",
    )
    parser.add_argument(
        "--nvcc-threads",
        type=int,
        default=None,
        help="Optional NVCC_THREADS value. Omit to use vLLM's default.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.max_jobs < 1:
        raise BuildContractError("--max-jobs must be at least 1.")
    if args.nvcc_threads is not None and args.nvcc_threads < 1:
        raise BuildContractError("--nvcc-threads must be at least 1.")

    contract_path = args.contract.resolve()
    contract = _load_json(contract_path)
    _validate_contract(contract)
    default_arch = str(contract["build"]["environment"]["TORCH_CUDA_ARCH_LIST"])
    _build(
        contract=contract,
        contract_path=contract_path,
        output_dir=args.output_dir.resolve(),
        cuda_arch_list=args.cuda_arch_list or default_arch,
        max_jobs=args.max_jobs,
        nvcc_threads=args.nvcc_threads,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
