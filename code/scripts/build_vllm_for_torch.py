#!/usr/bin/env python3
"""Build the pinned vLLM CUDA wheel against an existing PyTorch install.

The build runs only in a dedicated virtual environment. It verifies the
upstream commit and selected source files, applies vLLM's own
``tools/use_existing_torch.py --prefix`` transformation, removes any remaining
Torch-family requirements by canonical distribution name, and binds the
DeepGEMM setup child to its verified source root while preserving Python safe
path mode. It builds native CUDA and Rust extensions without build isolation.
The script does not install the resulting wheel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
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

MAXIMUM_BUILD_JOBS = 8
MAXIMUM_NVCC_THREADS = 2
REQUIRED_SOURCE_ENVIRONMENT_VARIABLES = frozenset(
    {
        "DEEPGEMM_SRC_DIR",
        "DEEPSELECT_SRC_DIR",
        "FLASH_KDA_SRC_DIR",
        "FLASH_MLA_SRC_DIR",
        "FMHA_SM100_SRC_DIR",
        "QUTLASS_SRC_DIR",
        "TML_FA4_SRC_DIR",
        "TRITON_KERNELS_SRC_DIR",
        "VLLM_CUTLASS_SRC_DIR",
        "VLLM_FLASH_ATTN_SRC_DIR",
    }
)
SCOPED_DIRECTORY_ENVIRONMENT = {
    "CARGO_HOME": "cargo-home",
    "CARGO_TARGET_DIR": "cargo-target",
    "CUDA_CACHE_PATH": "cuda-cache",
    "HF_HOME": "huggingface-cache",
    "MPLCONFIGDIR": "matplotlib-cache",
    "NUMBA_CACHE_DIR": "numba-cache",
    "PIP_CACHE_DIR": "pip-cache",
    "TMPDIR": "tmp",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor-cache",
    "TRITON_CACHE_DIR": "triton-cache",
    "XDG_CACHE_HOME": "xdg-cache",
}
CARGO_HOME_MUTABLE_CONTROL_PATHS = frozenset(
    {
        Path(".global-cache"),
        Path(".package-cache"),
        Path(".package-cache-mutate"),
    }
)
INHERITED_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "RUSTUP_HOME",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TZ",
    }
)
INHERITED_PATH_LIST_KEYS = frozenset(
    {
        "PATH",
    }
)
CONFLICTING_SELECTOR_KEYS = frozenset(
    {
        "ARCHFLAGS",
        "BUILD_TARGET",
        "CARGO",
        "CARGO_BUILD_JOBS",
        "CC",
        "CFLAGS",
        "CMAKE_ARGS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_INCLUDE_PATH",
        "CMAKE_LIBRARY_PATH",
        "CMAKE_PREFIX_PATH",
        "CPPFLAGS",
        "CUDACXX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDAHOSTCXX",
        "CXX",
        "CXXFLAGS",
        "LDFLAGS",
        "MAX_JOBS",
        "NVCC_APPEND_FLAGS",
        "NVCC_PREPEND_FLAGS",
        "NVCC_THREADS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTFLAGS",
        "RUSTUP_TOOLCHAIN",
        "TORCH_CUDA_HOME",
        "TORCH_CUDA_ARCH_LIST",
    }
)
TORCH_REQUIREMENT_NAMES = frozenset(
    {
        "torch",
        "torchaudio",
        "torchcodec",
        "torchvision",
    }
)
TOML_REQUIREMENT_LITERAL = re.compile(
    r"""^(?P<literal>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*,?(?:\s+#.*)?$"""
)
DEEPGEMM_BUILD_HELPER_PATH = Path("tools/build_deepgemm_C.py")
DEEPGEMM_BUILD_HELPER_SHA256 = "c6423b96e6980fcc2726feb9e3b459386c7334a13bf6bb44d026c5158bd8ee64"
DEEPGEMM_BUILD_HELPER_ENVIRONMENT = (
    b'env = os.environ.copy()\nenv.pop("DG_SKIP_CUDA_BUILD", None)\n'
)
DEEPGEMM_BUILD_HELPER_ENVIRONMENT_WITH_IMPORT = (
    DEEPGEMM_BUILD_HELPER_ENVIRONMENT
    + b"# PYTHONSAFEPATH removes the working directory from sys.path. Bind only this\n"
    + b"# setup.py subprocess to the verified DeepGEMM source root.\n"
    + b'env["PYTHONPATH"] = str(src)\n'
)
DEEPGEMM_BUILD_HELPER_COMMAND = b"""subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
    cwd=src,
    env=env,
)
"""
DEEPGEMM_BUILD_HELPER_COMMAND_OUT_OF_TREE = b"""build_root = out.parent / f"{out.name}.setuptools-build"
try:
    build_root.relative_to(src)
except ValueError:
    pass
else:
    sys.exit(f"DeepGEMM build root must be outside the pinned source tree: {build_root}")
if build_root.exists() or build_root.is_symlink():
    sys.exit(f"DeepGEMM build root must be fresh: {build_root}")
build_temp = build_root / "temp"
build_lib = build_root / "lib"
build_lib.mkdir(parents=True)
staged_package = build_lib / "deep_gemm"
shutil.copytree(_pkg, staged_package, symlinks=True)
env["PYTHONPATH"] = os.pathsep.join((str(build_lib), str(src)))
subprocess.check_call(
    [
        sys.executable,
        "setup.py",
        "build_ext",
        "--build-temp",
        str(build_temp),
        "--build-lib",
        str(build_lib),
        "--force",
    ],
    cwd=src,
    env=env,
)
"""
DEEPGEMM_BUILD_HELPER_EXTENSION = b'so = _pkg / "_C_extension.abi3.so"\n'
DEEPGEMM_BUILD_HELPER_EXTENSION_OUT_OF_TREE = b'so = staged_package / "_C_extension.abi3.so"\n'


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
    except (OSError, subprocess.CalledProcessError) as exc:
        returncode = getattr(exc, "returncode", "unavailable")
        raise BuildContractError(f"Command failed with exit code {returncode}: {rendered}") from exc


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
    except (OSError, subprocess.CalledProcessError) as exc:
        stdout = getattr(exc, "stdout", None)
        output = stdout.rstrip("\n") if stdout else "no output"
        returncode = getattr(exc, "returncode", "unavailable")
        raise BuildContractError(
            f"Command failed with exit code {returncode}: {' '.join(command)}\n{output}"
        ) from exc
    return completed.stdout.rstrip("\n")


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise BuildContractError(f"Required distribution is not installed: {name}") from exc


def _safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BuildContractError(f"{label} must be a safe relative path: {value!r}")
    return path


def _safe_dependency_name(value: Any) -> str:
    path = _safe_relative_path(value, label="Source dependency name")
    if len(path.parts) != 1:
        raise BuildContractError(
            f"Source dependency name must be one safe path component: {value!r}"
        )
    return path.name


def _validate_verified_file_keys(files: Any, *, label: str) -> None:
    if not isinstance(files, dict):
        raise BuildContractError(f"{label} must be an object of paths and SHA-256 values.")
    for relative_path in files:
        _safe_relative_path(relative_path, label=f"{label} path")


def _normalise_path_list(value: str, *, name: str) -> str:
    paths = value.split(os.pathsep)
    if any(not item for item in paths):
        raise BuildContractError(f"{name} must not contain an empty path component.")
    resolved: list[str] = []
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            raise BuildContractError(f"{name} must contain only absolute paths: {item!r}")
        if not path.is_dir():
            raise BuildContractError(f"{name} directory does not exist: {item}")
        resolved.append(str(path.absolute()))
    return os.pathsep.join(resolved)


def _resolved_command_paths(
    contract: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, str]:
    required = [str(value) for value in contract["build"]["required_commands"]]
    isolation_prefix = [
        str(value) for value in contract["build"]["network_isolation"]["command_prefix"]
    ]
    if not isolation_prefix:
        raise BuildContractError("Network isolation must declare a command prefix.")
    required.append(isolation_prefix[0])
    command_paths: dict[str, str] = {}
    for command in dict.fromkeys(required):
        path = shutil.which(command, path=environment.get("PATH"))
        if path is None:
            raise BuildContractError(f"Required build command is unavailable: {command}")
        command_path = Path(path).absolute()
        if command == "nvcc":
            try:
                command_path = command_path.resolve(strict=True)
            except OSError as exc:
                raise BuildContractError(f"Required nvcc target disappeared: {path}") from exc
            if not command_path.is_file():
                raise BuildContractError(f"Required nvcc target is not a file: {command_path}")
        command_paths[command] = str(command_path)
    return command_paths


def _command_identities(command_paths: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for name, lexical_path in sorted(command_paths.items()):
        path = Path(lexical_path)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise BuildContractError(f"Required command disappeared: {lexical_path}") from exc
        if not resolved.is_file():
            raise BuildContractError(f"Required command is not a regular file: {resolved}")
        identities[name] = {
            "path": lexical_path,
            "resolved_path": str(resolved),
            "sha256": _sha256_file(resolved),
            "size": resolved.stat().st_size,
        }
    return identities


def _planned_source_dependency_environment(
    contract: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    dependency_root = output_dir / "source-dependencies"
    planned: dict[str, str] = {}
    for name, dependency in contract["source_dependencies"].items():
        dependency_name = _safe_dependency_name(name)
        raw_subdirectory = str(dependency.get("source_subdirectory", ""))
        source_subdirectory = (
            _safe_relative_path(
                raw_subdirectory,
                label=f"source_dependencies.{name}.source_subdirectory",
            )
            if raw_subdirectory
            else Path()
        )
        environment_variable = str(dependency["environment_variable"])
        planned[environment_variable] = str(
            (dependency_root / dependency_name / source_subdirectory).resolve()
        )
    return planned


def _cmake_arguments(
    contract: Mapping[str, Any], source_environment: Mapping[str, str]
) -> list[str]:
    controls = contract["build"]["cmake_source_controls"]
    arguments = [str(value) for value in controls["arguments"]]
    by_environment = {
        str(dependency["environment_variable"]): str(dependency["fetchcontent_name"])
        for dependency in contract["source_dependencies"].values()
    }
    for environment_variable, source_path in sorted(source_environment.items()):
        if any(character.isspace() or character in "'\";\\" for character in source_path):
            raise BuildContractError(
                "vLLM's CMAKE_ARGS parser cannot safely represent a source path with "
                f"shell or CMake separators: {source_path}"
            )
        fetchcontent_name = re.sub(r"[^A-Za-z0-9_]", "_", by_environment[environment_variable])
        arguments.append(
            f"-DFETCHCONTENT_SOURCE_DIR_{fetchcontent_name.upper()}:PATH={source_path}"
        )
    return arguments


def _validate_inherited_selector(
    name: str,
    inherited_value: str,
    expected_value: str,
    *,
    path: str | None = None,
) -> None:
    if path is not None and not Path(inherited_value).is_absolute():
        resolved = shutil.which(inherited_value, path=path)
        inherited_value = resolved or inherited_value
    if (
        name.endswith("HOME")
        or name.endswith("PATH")
        or name
        in {
            "CC",
            "CXX",
            "CUDACXX",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CMAKE_CUDA_COMPILER",
            "RUSTC",
            "CARGO",
        }
    ):
        inherited_path = Path(inherited_value)
        expected_path = Path(expected_value)
        if inherited_path.exists() and expected_path.exists():
            inherited_value = str(inherited_path.resolve())
            expected_value = str(expected_path.resolve())
    if inherited_value != expected_value:
        raise BuildContractError(
            f"Inherited {name}={inherited_value!r} conflicts with the validated "
            f"build value {expected_value!r}."
        )


def _inspect_toolchain_overlay(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    overlay = contract["build"]["toolchain_overlay"]
    if not root.is_absolute():
        raise BuildContractError("--toolchain-overlay-root must be an absolute path.")
    if root.is_symlink() or not root.is_dir():
        raise BuildContractError(
            f"Toolchain overlay root must be an existing non-symlink directory: {root}"
        )
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise BuildContractError(f"Toolchain overlay root cannot be resolved: {root}") from exc
    if resolved_root != root:
        raise BuildContractError(
            f"Toolchain overlay root must not traverse symlinks: {root} -> {resolved_root}"
        )

    resolved_directories: dict[str, Path] = {}
    for name in ("include_directory", "library_directory", "pkg_config_directory"):
        relative = _safe_relative_path(overlay[name], label=f"toolchain_overlay.{name}")
        path = root / relative
        if path.is_symlink() or not path.is_dir():
            raise BuildContractError(f"Toolchain overlay {name} is missing or unsafe: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise BuildContractError(f"Toolchain overlay {name} escapes its root: {path}") from exc
        resolved_directories[name] = resolved

    required_files: list[dict[str, Any]] = []
    for value in overlay["required_files"]:
        relative = _safe_relative_path(value, label="toolchain_overlay.required_files")
        path = root / relative
        if not path.is_file():
            raise BuildContractError(f"Toolchain overlay required file is missing: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise BuildContractError(
                f"Toolchain overlay required file escapes its root: {path}"
            ) from exc
        required_files.append(
            {
                "path": relative.as_posix(),
                "resolved_path": str(resolved),
                "sha256": _sha256_file(resolved),
                "size": resolved.stat().st_size,
            }
        )

    include = str(resolved_directories["include_directory"])
    library = str(resolved_directories["library_directory"])
    pkg_config = str(resolved_directories["pkg_config_directory"])
    return {
        "root": str(root),
        "environment": {
            "CPATH": include,
            "CPLUS_INCLUDE_PATH": include,
            "LIBRARY_PATH": library,
            "PKG_CONFIG_PATH": pkg_config,
        },
        "required_files": required_files,
        "tree_manifest": _directory_tree_manifest(root, exclude_git_metadata=False),
    }


def _create_build_environment(
    contract: Mapping[str, Any],
    output_dir: Path,
    *,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int,
    toolchain_overlay_root: Path | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    inherited = dict(os.environ if base_environment is None else base_environment)
    inherited_path = inherited.get("PATH")
    if not inherited_path:
        raise BuildContractError("PATH must be nonempty for the pinned source build.")

    seed_environment = {"PATH": _normalise_path_list(inherited_path, name="PATH")}
    command_paths = _resolved_command_paths(contract, seed_environment)
    resolved_nvcc = Path(command_paths["nvcc"])
    cuda_root = resolved_nvcc.parent.parent
    canonical_nvcc = cuda_root / "bin" / "nvcc"
    try:
        canonical_nvcc_target = canonical_nvcc.resolve(strict=True)
    except OSError as exc:
        raise BuildContractError(
            f"Validated nvcc is not under a CUDA toolkit root: {resolved_nvcc}"
        ) from exc
    if canonical_nvcc_target != resolved_nvcc:
        raise BuildContractError(f"Validated nvcc is not under a CUDA toolkit root: {cuda_root}")

    expected_selectors = {
        "CARGO": command_paths["cargo"],
        "CC": command_paths["gcc"],
        "CMAKE_CUDA_COMPILER": command_paths["nvcc"],
        "CUDACXX": command_paths["nvcc"],
        "CUDA_HOME": str(cuda_root),
        "CUDA_PATH": str(cuda_root),
        "CUDA_ROOT": str(cuda_root),
        "CUDA_TOOLKIT_ROOT_DIR": str(cuda_root),
        "CXX": command_paths["g++"],
        "CARGO_BUILD_JOBS": str(max_jobs),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(max_jobs),
        "MAX_JOBS": str(max_jobs),
        "NVCC_THREADS": str(nvcc_threads),
        "RUSTC": command_paths["rustc"],
        "RUSTUP_TOOLCHAIN": str(contract["build"]["rust_toolchain"]),
        "TORCH_CUDA_HOME": str(cuda_root),
        "TORCH_CUDA_ARCH_LIST": cuda_arch_list,
    }
    for name in sorted(CONFLICTING_SELECTOR_KEYS):
        value = inherited.get(name)
        if not value:
            continue
        expected = expected_selectors.get(name)
        if expected is None:
            raise BuildContractError(
                f"Inherited build selector {name} is not allowed. Remove it before building."
            )
        _validate_inherited_selector(
            name,
            value,
            expected,
            path=seed_environment["PATH"],
        )

    for name, expected_value in contract["build"]["required_host_environment"].items():
        if inherited.get(str(name)) != str(expected_value):
            raise BuildContractError(
                f"Expected inherited {name}={expected_value!r}, found {inherited.get(str(name))!r}."
            )

    child: dict[str, str] = {"PATH": seed_environment["PATH"]}
    for name in sorted(INHERITED_ENVIRONMENT_ALLOWLIST - {"PATH"}):
        value = inherited.get(name)
        if not value:
            continue
        if name in INHERITED_PATH_LIST_KEYS:
            value = _normalise_path_list(value, name=name)
        elif name in {"HOME", "RUSTUP_HOME", "SSL_CERT_DIR", "SSL_CERT_FILE"}:
            path = Path(value)
            if not path.is_absolute() or not path.exists():
                raise BuildContractError(f"{name} must name an existing absolute path: {value}")
            value = str(path.resolve())
        child[name] = value

    home = child.get("HOME")
    if home is None:
        raise BuildContractError("HOME must name an existing absolute build-owner directory.")
    rustup_home = child.get("RUSTUP_HOME") or str((Path(home) / ".rustup").resolve())
    if not Path(rustup_home).is_dir() or Path(rustup_home).is_symlink():
        raise BuildContractError(
            f"RUSTUP_HOME must name an existing non-symlink directory: {rustup_home}"
        )
    child["RUSTUP_HOME"] = rustup_home

    child.update(
        {
            str(name): str(value)
            for name, value in contract["build"]["required_host_environment"].items()
        }
    )
    child.update(
        {str(name): str(value) for name, value in contract["build"]["environment"].items()}
    )
    child.update(
        {str(name): str(value) for name, value in contract["build"]["offline_environment"].items()}
    )
    child.update(expected_selectors)
    child.update(_planned_source_dependency_environment(contract, output_dir))
    if toolchain_overlay_root is not None:
        overlay = _inspect_toolchain_overlay(contract, toolchain_overlay_root)
        child.update(overlay["environment"])
    scoped_root = output_dir / "build-state"
    for name, relative in SCOPED_DIRECTORY_ENVIRONMENT.items():
        if name == "CARGO_HOME":
            cargo_relative = _safe_relative_path(
                contract["build"]["cargo_materialization"]["cargo_home_relative_to_output"],
                label="Cargo home relative to output",
            )
            child[name] = str((output_dir / cargo_relative).resolve())
        else:
            child[name] = str((scoped_root / relative).resolve())

    controls = contract["build"]["cmake_source_controls"]
    child[str(controls["fetchcontent_base_environment_variable"])] = str(
        (output_dir / str(controls["fetchcontent_base_relative_to_output"])).resolve()
    )
    cmake_arguments = _cmake_arguments(
        contract,
        {name: child[name] for name in REQUIRED_SOURCE_ENVIRONMENT_VARIABLES},
    )
    child[str(controls["arguments_environment_variable"])] = " ".join(cmake_arguments)
    child.update(
        {
            "CARGO_BUILD_JOBS": str(max_jobs),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(max_jobs),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "MAX_JOBS": str(max_jobs),
            "NVCC_THREADS": str(nvcc_threads),
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            "VLLM_SKIP_VERSION_SUFFIX": "1",
            "VLLM_VERSION_OVERRIDE": str(contract["package"]["version"]),
        }
    )

    removed = sorted(
        name for name in inherited if name not in child or inherited[name] != child[name]
    )
    return child, removed, command_paths


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

    build = contract["build"]
    expected_host_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    if build.get("required_host_environment") != expected_host_environment:
        raise BuildContractError(
            "The vLLM build must require the exact reviewed CPU-only host environment."
        )
    if int(build.get("maximum_jobs", 0)) != MAXIMUM_BUILD_JOBS:
        raise BuildContractError(
            f"The vLLM build contract must cap parallel jobs at {MAXIMUM_BUILD_JOBS}."
        )
    if int(build.get("maximum_nvcc_threads", 0)) != MAXIMUM_NVCC_THREADS:
        raise BuildContractError(
            f"The vLLM build contract must cap NVCC threads at {MAXIMUM_NVCC_THREADS}."
        )
    expected_offline = {
        "CARGO_NET_OFFLINE": "true",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
    }
    if build.get("offline_environment") != expected_offline:
        raise BuildContractError("The compile environment must use the reviewed offline settings.")
    expected_isolation = {
        "required_platform_system": "Linux",
        "required": True,
        "command_prefix": ["unshare", "--net", "--"],
    }
    if build.get("network_isolation") != expected_isolation:
        raise BuildContractError("The vLLM compile must require a Linux unshare network namespace.")
    controls = build.get("cmake_source_controls")
    expected_controls = {
        "fetchcontent_base_environment_variable": "FETCHCONTENT_BASE_DIR",
        "fetchcontent_base_relative_to_output": "fetchcontent",
        "arguments_environment_variable": "CMAKE_ARGS",
        "arguments": [
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
        ],
    }
    if controls != expected_controls:
        raise BuildContractError("The vLLM build must enable disconnected FetchContent mode.")
    expected_overlay = {
        "optional": True,
        "include_directory": "usr/include",
        "library_directory": "usr/lib/x86_64-linux-gnu",
        "pkg_config_directory": "usr/lib/x86_64-linux-gnu/pkgconfig",
        "required_files": [
            "usr/include/elfutils/libdwfl.h",
            "usr/lib/x86_64-linux-gnu/libdw.so",
            "usr/lib/x86_64-linux-gnu/libelf.so",
        ],
        "record_tree_manifest": True,
    }
    if build.get("toolchain_overlay") != expected_overlay:
        raise BuildContractError(
            "The optional toolchain overlay must use the reviewed elfutils sysroot layout."
        )
    required_build_environment = {
        "VLLM_TARGET_DEVICE": "cuda",
        "VLLM_USE_PRECOMPILED": "0",
        "VLLM_USE_PRECOMPILED_RUST": "0",
        "VLLM_REQUIRE_RUST_FRONTEND": "1",
    }
    for name, expected in required_build_environment.items():
        if str(build["environment"].get(name)) != expected:
            raise BuildContractError(f"The build contract must set {name}={expected}.")

    dependencies = contract["source_dependencies"]
    if len(dependencies) != len(REQUIRED_SOURCE_ENVIRONMENT_VARIABLES):
        raise BuildContractError("The contract must declare exactly ten native source roots.")
    actual_environment_variables = {
        str(value.get("environment_variable")) for value in dependencies.values()
    }
    if actual_environment_variables != REQUIRED_SOURCE_ENVIRONMENT_VARIABLES:
        missing = sorted(REQUIRED_SOURCE_ENVIRONMENT_VARIABLES - actual_environment_variables)
        unexpected = sorted(actual_environment_variables - REQUIRED_SOURCE_ENVIRONMENT_VARIABLES)
        raise BuildContractError(
            f"Native source environment variables differ. Missing: {missing}. "
            f"Unexpected: {unexpected}."
        )
    fetchcontent_names: set[str] = set()
    for name, dependency in dependencies.items():
        _safe_dependency_name(name)
        for required in (
            "repository",
            "commit",
            "tree",
            "environment_variable",
            "fetchcontent_name",
            "source_subdirectory",
            "submodules",
            "verified_files",
        ):
            if required not in dependency:
                raise BuildContractError(f"Source dependency {name} is missing {required}.")
        fetchcontent_name = str(dependency["fetchcontent_name"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", fetchcontent_name):
            raise BuildContractError(f"Source dependency {name} has an invalid FetchContent name.")
        if fetchcontent_name in fetchcontent_names:
            raise BuildContractError(f"Duplicate FetchContent name: {fetchcontent_name}")
        fetchcontent_names.add(fetchcontent_name)
        raw_source_subdirectory = str(dependency["source_subdirectory"])
        if raw_source_subdirectory:
            _safe_relative_path(
                raw_source_subdirectory,
                label=f"source_dependencies.{name}.source_subdirectory",
            )
        _validate_verified_file_keys(
            dependency["verified_files"],
            label=f"source_dependencies.{name}.verified_files",
        )
        submodule_paths: set[Path] = set()
        for submodule in dependency["submodules"]:
            path = _safe_relative_path(
                submodule.get("path"), label=f"source_dependencies.{name}.submodules.path"
            )
            if path in submodule_paths:
                raise BuildContractError(f"Duplicate {name} submodule path: {path}")
            submodule_paths.add(path)
            for required in ("repository", "commit", "tree"):
                if not submodule.get(required):
                    raise BuildContractError(
                        f"Source dependency {name} submodule {path} is missing {required}."
                    )

    _validate_verified_file_keys(contract["upstream_files"], label="upstream_files")

    cargo = build.get("cargo_materialization")
    required_cargo_values = {
        "manifest_path",
        "manifest_sha256",
        "lock_path",
        "lock_sha256",
        "lock_size",
        "cargo_home_relative_to_output",
        "prefetch_arguments",
        "locked_git_sources",
        "materialize_before_network_isolation",
        "verify_lock_unchanged_after_build",
        "record_tree_manifest",
    }
    if not isinstance(cargo, dict) or not required_cargo_values <= cargo.keys():
        raise BuildContractError("Cargo materialization is incomplete in the contract.")
    if cargo["prefetch_arguments"] != [
        "fetch",
        "--locked",
        "--manifest-path",
        str(cargo["manifest_path"]),
    ]:
        raise BuildContractError("Cargo prefetch arguments must use the exact locked manifest.")
    if not all(
        cargo[key] is True
        for key in (
            "materialize_before_network_isolation",
            "verify_lock_unchanged_after_build",
            "record_tree_manifest",
        )
    ):
        raise BuildContractError("Cargo materialization must be recorded and reverified.")
    _safe_relative_path(
        cargo["cargo_home_relative_to_output"],
        label="Cargo home relative to output",
    )
    if cargo["cargo_home_relative_to_output"] != "cargo-home":
        raise BuildContractError("Cargo home must be scoped to output/cargo-home.")
    locked_git_sources = [str(value) for value in cargo["locked_git_sources"]]
    if len(locked_git_sources) != 3 or len(set(locked_git_sources)) != 3:
        raise BuildContractError("Cargo materialization must pin exactly three Git sources.")

    receipt = build.get("receipt_requirements")
    expected_receipt_flags = {
        "record_active_lock_sha256",
        "record_installed_distributions",
        "record_effective_build_environment",
        "record_removed_inherited_environment",
        "record_commands",
        "record_network_isolation",
        "record_cmake_cache_bindings",
        "record_source_dependency_tree_manifests",
        "record_submodule_identities",
        "record_cargo_materialization",
    }
    if not isinstance(receipt, dict) or not receipt.get("active_lock_path"):
        raise BuildContractError("Receipt requirements must name the active Python lock.")
    if any(receipt.get(name) is not True for name in expected_receipt_flags):
        raise BuildContractError("Every source-build receipt requirement must remain enabled.")


def _validate_build_parameters(
    contract: Mapping[str, Any],
    *,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int,
) -> None:
    if not re.fullmatch(r"\d+\.\d+(;\d+\.\d+)*", cuda_arch_list):
        raise BuildContractError("--cuda-arch-list must contain values such as 10.0 or 10.0;10.3.")
    maximum_jobs = int(contract["build"]["maximum_jobs"])
    if not 1 <= max_jobs <= maximum_jobs:
        raise BuildContractError(f"--max-jobs must be between 1 and {maximum_jobs}.")
    maximum_nvcc_threads = int(contract["build"]["maximum_nvcc_threads"])
    if not 1 <= nvcc_threads <= maximum_nvcc_threads:
        raise BuildContractError(f"--nvcc-threads must be between 1 and {maximum_nvcc_threads}.")


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


def _require_cuda_uninitialized(torch_module: Any, *, checkpoint: str) -> bool:
    initialized = bool(torch_module.cuda.is_initialized())
    if initialized:
        raise BuildContractError(f"Torch initialized CUDA {checkpoint}.")
    return initialized


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
    cuda_initialized_after_import = _require_cuda_uninitialized(
        torch, checkpoint="during the masked preflight import"
    )

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

    command_paths = _resolved_command_paths(contract, process_environment)
    expected_selectors = {
        "CARGO": command_paths["cargo"],
        "CC": command_paths["gcc"],
        "CMAKE_CUDA_COMPILER": command_paths["nvcc"],
        "CUDACXX": command_paths["nvcc"],
        "CXX": command_paths["g++"],
        "RUSTC": command_paths["rustc"],
        "RUSTUP_TOOLCHAIN": str(contract["build"]["rust_toolchain"]),
    }
    for name, expected_value in expected_selectors.items():
        if process_environment.get(name) != expected_value:
            raise BuildContractError(
                f"Validated child environment has {name}={process_environment.get(name)!r}, "
                f"expected {expected_value!r}."
            )

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

    isolation_prefix = [
        command_paths.get(str(value), str(value))
        for value in contract["build"]["network_isolation"]["command_prefix"]
    ]
    try:
        subprocess.run(
            [*isolation_prefix, "true"],
            env=process_environment,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise BuildContractError(
            "Linux network namespace preflight failed. The native compile cannot "
            f"run offline: {detail.strip()[-2000:]}"
        ) from exc

    cuda_initialized_after_validation = _require_cuda_uninitialized(
        torch, checkpoint="during masked build validation"
    )

    python_executable = Path(sys.executable).resolve()
    return {
        "python": platform.python_version(),
        "python_executable": str(python_executable),
        "python_executable_sha256": _sha256_file(python_executable),
        "python_executable_size": python_executable.stat().st_size,
        "platform_system": actual_system,
        "platform_machine": actual_machine,
        "required_host_environment": {
            str(name): process_environment[str(name)]
            for name in contract["build"]["required_host_environment"]
        },
        "torch": torch.__version__,
        "torch_cuda": actual_cuda,
        "torch_cuda_initialization": {
            "after_import": cuda_initialized_after_import,
            "after_validation": cuda_initialized_after_validation,
        },
        "distributions": actual_distributions,
        "commands": command_paths,
        "command_identities": _command_identities(command_paths),
        "required_cxx_header_probe": required_cxx_headers,
        "gcc_version": gcc_version,
        "nvcc_version": nvcc_version,
        "cmake_version": _capture(
            [command_paths["cmake"], "--version"], env=process_environment
        ).splitlines()[0],
        "ninja_version": _capture([command_paths["ninja"], "--version"], env=process_environment),
        "rustc_version": rustc_version,
        "cargo_version": cargo_version,
        "network_isolation_preflight": {
            "command": [*isolation_prefix, "true"],
            "result": "passed",
        },
    }


def _gitlink_commit(
    repository_dir: Path,
    relative_path: Path,
    *,
    env: Mapping[str, str],
) -> str:
    output = _capture(
        ["git", "ls-tree", "HEAD", "--", relative_path.as_posix()],
        cwd=repository_dir,
        env=env,
    )
    match = re.fullmatch(r"160000 commit ([0-9a-f]{40})\t.+", output)
    if match is None:
        raise BuildContractError(
            f"Expected a pinned Git submodule at {repository_dir / relative_path}."
        )
    return match.group(1)


def _clone_exact_repository(
    source: Mapping[str, Any],
    source_dir: Path,
    *,
    label: str,
    env: Mapping[str, str],
) -> dict[str, str]:
    repository = str(source["repository"])
    commit = str(source["commit"])
    expected_tree = str(source["tree"])

    if source_dir.is_symlink() or (
        source_dir.exists() and (not source_dir.is_dir() or any(source_dir.iterdir()))
    ):
        raise BuildContractError(f"Fresh {label} source path already exists: {source_dir}")
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(source_dir),
        ],
        env=env,
    )
    _run(
        ["git", "-C", str(source_dir), "fetch", "--depth", "1", "origin", commit],
        env=env,
    )
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
        ],
        env=env,
    )

    actual_commit = _capture(["git", "rev-parse", "HEAD"], cwd=source_dir, env=env)
    actual_tree = _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, env=env)
    if actual_commit != commit:
        raise BuildContractError(
            f"Checked-out {label} commit mismatch: expected {commit}, found {actual_commit}."
        )
    if actual_tree != expected_tree:
        raise BuildContractError(
            f"Checked-out {label} tree mismatch: expected {expected_tree}, found {actual_tree}."
        )

    return {
        "repository": repository,
        "commit": actual_commit,
        "tree": actual_tree,
    }


def _declared_submodule_parent(
    path: Path,
    materialized: Mapping[Path, Path],
    source_dir: Path,
) -> tuple[Path, Path]:
    parents = [candidate for candidate in materialized if candidate in path.parents]
    if not parents:
        return source_dir, path
    logical_parent = max(parents, key=lambda candidate: len(candidate.parts))
    return materialized[logical_parent], path.relative_to(logical_parent)


def _materialize_submodules(
    source: Mapping[str, Any],
    source_dir: Path,
    *,
    label: str,
    env: Mapping[str, str],
) -> list[dict[str, str]]:
    declared = sorted(
        source.get("submodules", []),
        key=lambda value: len(Path(str(value["path"])).parts),
    )
    materialized: dict[Path, Path] = {}
    receipts: list[dict[str, str]] = []
    for submodule in declared:
        logical_path = _safe_relative_path(submodule["path"], label=f"{label} submodule path")
        parent_dir, parent_relative_path = _declared_submodule_parent(
            logical_path, materialized, source_dir
        )
        expected_commit = str(submodule["commit"])
        gitlink_commit = _gitlink_commit(parent_dir, parent_relative_path, env=env)
        if gitlink_commit != expected_commit:
            raise BuildContractError(
                f"{label} submodule {logical_path} gitlink mismatch: expected "
                f"{expected_commit}, found {gitlink_commit}."
            )
        destination = source_dir / logical_path
        identity = _clone_exact_repository(
            submodule,
            destination,
            label=f"{label} submodule {logical_path}",
            env=env,
        )
        materialized[logical_path] = destination
        receipts.append(
            {
                **identity,
                "path": logical_path.as_posix(),
                "source_path": str(destination.resolve()),
                "parent_gitlink_commit": gitlink_commit,
            }
        )
    return receipts


def _verify_declared_gitlinks(
    source: Mapping[str, Any],
    source_dir: Path,
    *,
    label: str,
    env: Mapping[str, str],
) -> None:
    declared = {
        _safe_relative_path(value["path"], label=f"{label} submodule path")
        for value in source.get("submodules", [])
    }
    repository_roots = {Path("."): source_dir}
    repository_roots.update({path: source_dir / path for path in declared})
    expected_by_parent: dict[Path, set[Path]] = {logical: set() for logical in repository_roots}
    for path in declared:
        ancestors = [candidate for candidate in declared if candidate in path.parents]
        parent = max(ancestors, key=lambda value: len(value.parts)) if ancestors else Path(".")
        expected_by_parent[parent].add(path.relative_to(parent))

    for logical_root, repository_dir in repository_roots.items():
        output = _capture(["git", "ls-tree", "-r", "HEAD"], cwd=repository_dir, env=env)
        actual: set[Path] = set()
        for line in output.splitlines():
            match = re.fullmatch(r"160000 commit [0-9a-f]{40}\t(.+)", line)
            if match is not None:
                actual.add(Path(match.group(1)))
        expected = expected_by_parent[logical_root]
        if actual != expected:
            root_name = label if logical_root == Path(".") else f"{label}/{logical_root}"
            raise BuildContractError(
                f"{root_name} Git submodule declarations differ. "
                f"Expected {sorted(map(str, expected))}, found {sorted(map(str, actual))}."
            )


def _verify_clean_repository(
    source: Mapping[str, Any],
    source_dir: Path,
    *,
    label: str,
    env: Mapping[str, str],
    verify_declared_gitlinks: bool = True,
) -> dict[str, Any]:
    actual_commit = _capture(["git", "rev-parse", "HEAD"], cwd=source_dir, env=env)
    actual_tree = _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, env=env)
    expected_commit = str(source["commit"])
    expected_tree = str(source["tree"])
    if actual_commit != expected_commit or actual_tree != expected_tree:
        raise BuildContractError(
            f"{label} identity changed. Expected {expected_commit}/{expected_tree}, "
            f"found {actual_commit}/{actual_tree}."
        )
    if verify_declared_gitlinks:
        _verify_declared_gitlinks(source, source_dir, label=label, env=env)
    status = _capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_dir,
        env=env,
    )
    if status:
        raise BuildContractError(f"{label} source tree is dirty:\n{status}")

    submodules: list[dict[str, str]] = []
    materialized: dict[Path, Path] = {}
    for submodule in sorted(
        source.get("submodules", []),
        key=lambda value: len(Path(str(value["path"])).parts),
    ):
        logical_path = _safe_relative_path(submodule["path"], label=f"{label} submodule path")
        parent_dir, parent_relative_path = _declared_submodule_parent(
            logical_path, materialized, source_dir
        )
        gitlink_commit = _gitlink_commit(parent_dir, parent_relative_path, env=env)
        destination = source_dir / logical_path
        identity = _verify_clean_repository(
            {**submodule, "submodules": []},
            destination,
            label=f"{label} submodule {logical_path}",
            env=env,
            verify_declared_gitlinks=False,
        )
        if gitlink_commit != identity["commit"]:
            raise BuildContractError(
                f"{label} submodule {logical_path} no longer matches its parent gitlink."
            )
        materialized[logical_path] = destination
        submodules.append(
            {
                "path": logical_path.as_posix(),
                "source_path": str(destination.resolve()),
                "parent_gitlink_commit": gitlink_commit,
                "repository": str(submodule["repository"]),
                "commit": identity["commit"],
                "tree": identity["tree"],
            }
        )
    return {
        "repository": str(source["repository"]),
        "commit": actual_commit,
        "tree": actual_tree,
        "submodules": submodules,
    }


def _verify_repository_file(
    source_dir: Path,
    relative_path: Any,
    expected_sha256: Any,
) -> tuple[str, str]:
    relative = _safe_relative_path(relative_path, label="verified repository file path")
    path = source_dir / relative
    if path.is_symlink() or not path.is_file():
        raise BuildContractError(f"Pinned upstream file is missing or unsafe: {relative}")
    try:
        source_root = source_dir.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise BuildContractError(
            f"Pinned upstream file escapes its source root: {relative}"
        ) from exc
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != str(expected_sha256):
        raise BuildContractError(
            f"Pinned upstream file hash mismatch for {relative}: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    return relative.as_posix(), actual_sha256


def _checkout_repository(
    source: Mapping[str, Any],
    source_dir: Path,
    *,
    label: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    identity = _clone_exact_repository(source, source_dir, label=label, env=env)
    submodules = _materialize_submodules(source, source_dir, label=label, env=env)

    verified_files: dict[str, str] = {}
    for relative, expected_sha256 in source.get("verified_files", {}).items():
        verified_path, actual_sha256 = _verify_repository_file(
            source_dir,
            relative,
            expected_sha256,
        )
        verified_files[verified_path] = actual_sha256

    clean_identity = _verify_clean_repository(source, source_dir, label=label, env=env)
    if submodules != clean_identity["submodules"]:
        raise BuildContractError(f"{label} submodule identities changed during materialization.")

    return {
        **identity,
        "submodules": submodules,
        "verified_files": verified_files,
    }


def _checkout_source(
    contract: Mapping[str, Any], source_dir: Path, *, env: Mapping[str, str]
) -> dict[str, Any]:
    source = dict(contract["source"])
    source["verified_files"] = contract["upstream_files"]
    source["submodules"] = []
    return _checkout_repository(source, source_dir, label="vLLM", env=env)


def _verify_source_dependency_unchanged(
    dependency: Mapping[str, Any],
    dependency_dir: Path,
    *,
    label: str,
    before_receipt: dict[str, Any],
    env: Mapping[str, str],
) -> None:
    post_identity = _verify_clean_repository(
        dependency,
        dependency_dir,
        label=label,
        env=env,
    )
    for identity_key in ("repository", "commit", "tree", "submodules"):
        if post_identity[identity_key] != before_receipt[identity_key]:
            raise BuildContractError(
                f"Pinned {label} {identity_key} changed during the native build."
            )
    post_manifest = _directory_tree_manifest(dependency_dir)
    if post_manifest != before_receipt["tree_manifest_before_build"]:
        raise BuildContractError(f"Pinned {label} source bytes changed during the native build.")
    before_receipt["tree_manifest_after_build"] = post_manifest
    before_receipt["unchanged_after_build"] = True


def _directory_tree_manifest(
    root: Path,
    *,
    excluded_paths: frozenset[Path] = frozenset(),
    exclude_git_metadata: bool = True,
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise BuildContractError(f"Tree-manifest root is missing or unsafe: {root}")
    for excluded in excluded_paths:
        _safe_relative_path(excluded, label="tree-manifest excluded path")
    digest = hashlib.sha256()
    entry_count = 0
    total_file_size = 0
    paths: list[Path] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for directory_name in sorted(directory_names):
            if exclude_git_metadata and directory_name == ".git":
                continue
            path = current_path / directory_name
            if path.is_symlink():
                paths.append(path)
            else:
                retained_directories.append(directory_name)
        directory_names[:] = retained_directories
        paths.extend(
            current_path / file_name
            for file_name in sorted(file_names)
            if (not exclude_git_metadata or file_name != ".git")
            and (current_path / file_name).relative_to(root) not in excluded_paths
        )

    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            payload_sha256 = hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
            size = len(os.readlink(path).encode("utf-8"))
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            payload_sha256 = _sha256_file(path)
            size = metadata.st_size
            total_file_size += size
        else:
            raise BuildContractError(f"Unsupported source-tree entry: {path}")
        record = {
            "kind": kind,
            "mode": stat.S_IMODE(metadata.st_mode),
            "path": relative.as_posix(),
            "sha256": payload_sha256,
            "size": size,
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        entry_count += 1
    return {
        "root": str(root.resolve()),
        "sha256": digest.hexdigest(),
        "entry_count": entry_count,
        "total_file_size": total_file_size,
        "excluded_git_metadata": exclude_git_metadata,
    }


def _cargo_home_dependency_manifest(root: Path) -> dict[str, Any]:
    manifest = _directory_tree_manifest(
        root,
        excluded_paths=CARGO_HOME_MUTABLE_CONTROL_PATHS,
    )
    return {
        **manifest,
        "excluded_mutable_control_paths": sorted(
            path.as_posix() for path in CARGO_HOME_MUTABLE_CONTROL_PATHS
        ),
    }


def _cargo_home_control_files(root: Path) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for relative in sorted(CARGO_HOME_MUTABLE_CONTROL_PATHS, key=lambda value: value.as_posix()):
        path = root / relative
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise BuildContractError(f"Cargo control path is not a regular file: {path}")
        controls.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return controls


def _verify_file_contract(
    root: Path,
    relative_path: Any,
    expected_sha256: Any,
    *,
    expected_size: int | None = None,
    label: str,
) -> dict[str, Any]:
    relative = _safe_relative_path(relative_path, label=f"{label} path")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise BuildContractError(f"{label} is missing or unsafe: {path}")
    actual_sha256 = _sha256_file(path)
    actual_size = path.stat().st_size
    if actual_sha256 != str(expected_sha256):
        raise BuildContractError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}."
        )
    if expected_size is not None and actual_size != expected_size:
        raise BuildContractError(
            f"{label} size mismatch: expected {expected_size}, found {actual_size}."
        )
    return {
        "path": str(path.resolve()),
        "relative_path": relative.as_posix(),
        "sha256": actual_sha256,
        "size": actual_size,
    }


def _materialize_cargo_dependencies(
    contract: Mapping[str, Any],
    source_dir: Path,
    environment: Mapping[str, str],
    command_paths: Mapping[str, str],
) -> dict[str, Any]:
    cargo = contract["build"]["cargo_materialization"]
    manifest = _verify_file_contract(
        source_dir,
        cargo["manifest_path"],
        cargo["manifest_sha256"],
        label="Cargo workspace manifest",
    )
    lock = _verify_file_contract(
        source_dir,
        cargo["lock_path"],
        cargo["lock_sha256"],
        expected_size=int(cargo["lock_size"]),
        label="Cargo lock",
    )
    lock_text = Path(lock["path"]).read_text(encoding="utf-8")
    actual_git_sources = sorted(
        set(re.findall(r'^source = "(git\+[^\"]+)"$', lock_text, flags=re.MULTILINE))
    )
    expected_git_sources = sorted({str(value) for value in cargo["locked_git_sources"]})
    if actual_git_sources != expected_git_sources:
        raise BuildContractError(
            "Cargo.lock Git sources differ from the contract. "
            f"Expected {expected_git_sources}, found {actual_git_sources}."
        )

    prefetch_environment = dict(environment)
    prefetch_environment["CARGO_NET_OFFLINE"] = "false"
    command = [
        command_paths["cargo"],
        *[str(value) for value in cargo["prefetch_arguments"]],
    ]
    _run(command, cwd=source_dir, env=prefetch_environment)
    cargo_home = Path(environment["CARGO_HOME"])
    manifest_after_fetch = _cargo_home_dependency_manifest(cargo_home)
    control_files_after_fetch = _cargo_home_control_files(cargo_home)
    lock_after_fetch = _verify_file_contract(
        source_dir,
        cargo["lock_path"],
        cargo["lock_sha256"],
        expected_size=int(cargo["lock_size"]),
        label="Cargo lock after dependency materialization",
    )
    return {
        "command": command,
        "environment_override": {"CARGO_NET_OFFLINE": "false"},
        "compile_environment": {"CARGO_NET_OFFLINE": environment["CARGO_NET_OFFLINE"]},
        "manifest": manifest,
        "lock": lock_after_fetch,
        "locked_git_sources": actual_git_sources,
        "cargo_home_tree_manifest": manifest_after_fetch,
        "cargo_home_control_files_after_fetch": control_files_after_fetch,
    }


def _verify_cargo_materialization_unchanged(
    contract: Mapping[str, Any],
    source_dir: Path,
    environment: Mapping[str, str],
    before: Mapping[str, Any],
) -> dict[str, Any]:
    cargo = contract["build"]["cargo_materialization"]
    lock = _verify_file_contract(
        source_dir,
        cargo["lock_path"],
        cargo["lock_sha256"],
        expected_size=int(cargo["lock_size"]),
        label="Cargo lock after native build",
    )
    cargo_home = Path(environment["CARGO_HOME"])
    cargo_home_manifest = _cargo_home_dependency_manifest(cargo_home)
    if cargo_home_manifest != before["cargo_home_tree_manifest"]:
        raise BuildContractError("Cargo dependency materialization changed during offline compile.")
    return {
        **before,
        "lock_after_build": lock,
        "cargo_home_tree_manifest_after_build": cargo_home_manifest,
        "cargo_home_control_files_after_build": _cargo_home_control_files(cargo_home),
        "unchanged_after_build": True,
    }


def _installed_distribution_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name or not distribution.version:
            raise BuildContractError("Installed distribution metadata is incomplete.")
        inventory.append({"name": str(name), "version": str(distribution.version)})
    return sorted(
        inventory,
        key=lambda value: (_normalise_distribution_name(value["name"]), value["version"]),
    )


def _inspect_cmake_cache_bindings(
    contract: Mapping[str, Any], source_dir: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    cache_paths = sorted(source_dir.rglob("CMakeCache.txt"))
    if not cache_paths:
        raise BuildContractError("The vLLM build produced no CMakeCache.txt provenance.")
    expected: dict[str, str] = {}
    for dependency in contract["source_dependencies"].values():
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(dependency["fetchcontent_name"])).upper()
        variable = f"FETCHCONTENT_SOURCE_DIR_{name}"
        expected[variable] = environment[str(dependency["environment_variable"])]
    expected.update(
        {
            "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
            "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
        }
    )
    found: dict[str, dict[str, str]] = {}
    cache_receipts: list[dict[str, Any]] = []
    for path in cache_paths:
        relative = path.relative_to(source_dir).as_posix()
        cache_receipts.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            match = re.fullmatch(r"([^#/][^:]*)[^=]*=(.*)", line)
            if match is None or match.group(1) not in expected:
                continue
            variable = match.group(1)
            value = match.group(2)
            if value == expected[variable]:
                found[variable] = {"cache": relative, "value": value}
    missing = sorted(set(expected) - set(found))
    if missing:
        raise BuildContractError(
            "CMake cache did not bind the reviewed offline source controls: " + ", ".join(missing)
        )
    return {"caches": cache_receipts, "bindings": found}


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_from_line(raw_line: str, *, toml: bool) -> Requirement | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if toml:
        match = TOML_REQUIREMENT_LITERAL.fullmatch(stripped)
        if match is None:
            return None
        try:
            candidate = ast.literal_eval(match.group("literal"))
        except (SyntaxError, ValueError):
            return None
        if not isinstance(candidate, str):
            return None
        attempts = [candidate]
    else:
        attempts = [stripped]
        without_comment = re.split(r"\s+#", stripped, maxsplit=1)[0].rstrip()
        if without_comment != stripped:
            attempts.append(without_comment)

    for candidate in attempts:
        try:
            return Requirement(candidate)
        except InvalidRequirement:
            continue
    return None


def _requirement_candidates(source_dir: Path) -> list[Path]:
    candidates = [source_dir / "pyproject.toml"]
    candidates.extend(sorted((source_dir / "requirements").rglob("*.txt")))
    candidates.extend(sorted((source_dir / "requirements").rglob("*.in")))
    return candidates


def _torch_requirement_locations(source_dir: Path, candidates: Sequence[Path]) -> list[str]:
    remaining: list[str] = []
    for path in candidates:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            requirement = _requirement_from_line(raw_line, toml=path.name == "pyproject.toml")
            if (
                requirement is not None
                and _normalise_distribution_name(requirement.name) in TORCH_REQUIREMENT_NAMES
            ):
                remaining.append(f"{path.relative_to(source_dir)}:{line_number}:{raw_line}")
    return remaining


def _strip_torch_requirements(source_dir: Path, candidates: Sequence[Path]) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    for path in candidates:
        original_lines = path.read_bytes().splitlines(keepends=True)
        retained_lines: list[bytes] = []
        for line_number, raw_bytes in enumerate(original_lines, start=1):
            raw_line = raw_bytes.decode("utf-8")
            requirement = _requirement_from_line(raw_line, toml=path.name == "pyproject.toml")
            if (
                requirement is not None
                and _normalise_distribution_name(requirement.name) in TORCH_REQUIREMENT_NAMES
            ):
                removed.append(
                    {
                        "path": str(path.relative_to(source_dir)),
                        "line": line_number,
                        "requirement": raw_line.rstrip("\r\n"),
                    }
                )
            else:
                retained_lines.append(raw_bytes)
        if len(retained_lines) != len(original_lines):
            path.write_bytes(b"".join(retained_lines))
    return removed


def _apply_deepgemm_source_import_binding(source_dir: Path) -> dict[str, Any]:
    relative_path, upstream_sha256 = _verify_repository_file(
        source_dir,
        DEEPGEMM_BUILD_HELPER_PATH,
        DEEPGEMM_BUILD_HELPER_SHA256,
    )
    helper = source_dir / DEEPGEMM_BUILD_HELPER_PATH
    original = helper.read_bytes()
    replacements = (
        (
            "environment",
            DEEPGEMM_BUILD_HELPER_ENVIRONMENT,
            DEEPGEMM_BUILD_HELPER_ENVIRONMENT_WITH_IMPORT,
        ),
        (
            "build command",
            DEEPGEMM_BUILD_HELPER_COMMAND,
            DEEPGEMM_BUILD_HELPER_COMMAND_OUT_OF_TREE,
        ),
        (
            "extension source",
            DEEPGEMM_BUILD_HELPER_EXTENSION,
            DEEPGEMM_BUILD_HELPER_EXTENSION_OUT_OF_TREE,
        ),
    )
    transformed = original
    for label, expected, replacement in replacements:
        if transformed.count(expected) != 1:
            raise BuildContractError(
                f"Pinned DeepGEMM build helper does not contain exactly one reviewed {label} block."
            )
        transformed = transformed.replace(expected, replacement, 1)
    helper.write_bytes(transformed)
    return {
        "path": relative_path,
        "upstream_sha256": upstream_sha256,
        "transformed_sha256": _sha256_file(helper),
        "environment_variable": "PYTHONPATH",
        "value": "complete external package stage, then resolved DeepGEMM source root",
        "scope": "DeepGEMM setup.py subprocess",
        "python_safe_path_preserved": True,
        "build_mode": "setuptools-build-ext-out-of-tree",
        "build_root": "sibling of CMake output",
        "staged_package": "complete pinned deep_gemm package under external build-lib",
        "source_write_policy": "pinned DeepGEMM tree remains byte-identical",
        "reason": (
            "Pinned DeepGEMM setup.py requires a verified import root and its default "
            "in-place extension build writes ignored artifacts into the pinned source tree."
        ),
        "retirement_condition": (
            "The pinned vLLM helper binds the source root without an ambient import "
            "fallback and builds the extension outside the DeepGEMM source tree."
        ),
    }


def _apply_source_transforms(
    contract: Mapping[str, Any], source_dir: Path, *, env: Mapping[str, str]
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    relative_script = Path(str(contract["build"]["use_existing_torch_script"]))
    script = source_dir / relative_script
    arguments = [str(value) for value in contract["build"]["use_existing_torch_arguments"]]
    _run([sys.executable, str(script), *arguments], cwd=source_dir, env=env)

    candidates = _requirement_candidates(source_dir)
    supplemental_removals = _strip_torch_requirements(source_dir, candidates)
    remaining = _torch_requirement_locations(source_dir, candidates)
    if remaining:
        raise BuildContractError(
            "Torch requirement transforms left forbidden requirements behind:\n"
            + "\n".join(remaining)
        )
    deepgemm_import_binding = _apply_deepgemm_source_import_binding(source_dir)

    diff_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=source_dir,
        env=env,
        text=True,
        capture_output=True,
    )
    if diff_check.returncode != 0:
        raise BuildContractError(
            "The reviewed source transforms produced an invalid diff:\n"
            + (diff_check.stdout or diff_check.stderr)
        )
    modified = _capture(["git", "diff", "--name-only"], cwd=source_dir, env=env).splitlines()
    unexpected = [
        path
        for path in modified
        if path not in {"pyproject.toml", DEEPGEMM_BUILD_HELPER_PATH.as_posix()}
        and not path.startswith("requirements/")
    ]
    if unexpected:
        raise BuildContractError(
            "The reviewed source transforms changed unexpected paths: " + ", ".join(unexpected)
        )
    return modified, supplemental_removals, deepgemm_import_binding


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


def _create_build_command(
    contract: Mapping[str, Any],
    wheel_dir: Path,
    command_paths: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    frontend = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheel_dir),
        ".",
    ]
    prefix = [
        command_paths.get(str(value), str(value))
        for value in contract["build"]["network_isolation"]["command_prefix"]
    ]
    return [*prefix, *frontend], frontend


def _build(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    output_dir: Path,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int,
    toolchain_overlay_root: Path | None = None,
) -> Path:
    _validate_build_parameters(
        contract,
        cuda_arch_list=cuda_arch_list,
        max_jobs=max_jobs,
        nvcc_threads=nvcc_threads,
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise BuildContractError(
            "Output directory is not empty or already exists under the fresh-output "
            f"contract: {output_dir}"
        )

    builder_path = Path(__file__).resolve()
    builder_sha256 = _sha256_file(builder_path)
    contract_sha256 = _sha256_file(contract_path)
    toolchain_overlay_before = (
        _inspect_toolchain_overlay(contract, toolchain_overlay_root)
        if toolchain_overlay_root is not None
        else None
    )

    build_environment, removed_environment, planned_command_paths = _create_build_environment(
        contract,
        output_dir,
        cuda_arch_list=cuda_arch_list,
        max_jobs=max_jobs,
        nvcc_threads=nvcc_threads,
        toolchain_overlay_root=toolchain_overlay_root,
    )
    runtime = _validate_environment(contract, command_environment=build_environment)
    if runtime["commands"] != planned_command_paths:
        raise BuildContractError(
            "Command identities changed while validating the child environment."
        )

    receipt_requirements = contract["build"]["receipt_requirements"]
    lock_relative = _safe_relative_path(
        receipt_requirements["active_lock_path"], label="active lock path"
    )
    active_lock_path = contract_path.parent / lock_relative
    if not active_lock_path.is_file() or active_lock_path.is_symlink():
        raise BuildContractError(f"Active Python lock is missing or unsafe: {active_lock_path}")
    active_lock = {
        "path": str(active_lock_path.resolve()),
        "sha256": _sha256_file(active_lock_path),
        "size": active_lock_path.stat().st_size,
    }
    installed_distributions = _installed_distribution_inventory()

    output_dir.mkdir(parents=True, exist_ok=False)
    for name in SCOPED_DIRECTORY_ENVIRONMENT:
        Path(build_environment[name]).mkdir(parents=True, exist_ok=False)
    fetchcontent_root = Path(
        build_environment[
            str(
                contract["build"]["cmake_source_controls"]["fetchcontent_base_environment_variable"]
            )
        ]
    )
    fetchcontent_root.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / RECEIPT_NAME
    source_dir = output_dir / SOURCE_SUBDIR
    source_receipt = _checkout_source(contract, source_dir, env=build_environment)
    modified_paths, supplemental_removals, deepgemm_import_binding = _apply_source_transforms(
        contract, source_dir, env=build_environment
    )
    transform_diff_sha256 = hashlib.sha256(
        _capture(["git", "diff", "--binary"], cwd=source_dir, env=build_environment).encode()
    ).hexdigest()

    source_dependency_receipts: dict[str, Any] = {}
    dependency_root = output_dir / "source-dependencies"
    for name, dependency in contract["source_dependencies"].items():
        dependency_name = _safe_dependency_name(name)
        dependency_dir = dependency_root / dependency_name
        dependency_receipt = _checkout_repository(
            dependency,
            dependency_dir,
            label=str(name),
            env=build_environment,
        )
        raw_source_subdirectory = str(dependency.get("source_subdirectory", ""))
        source_subdirectory = (
            _safe_relative_path(
                raw_source_subdirectory,
                label=f"source_dependencies.{name}.source_subdirectory",
            )
            if raw_source_subdirectory
            else Path()
        )
        environment_path = dependency_dir / source_subdirectory
        if not environment_path.is_dir():
            raise BuildContractError(
                f"Pinned {name} source directory is missing: {environment_path}"
            )
        environment_variable = str(dependency["environment_variable"])
        expected_environment_path = build_environment[environment_variable]
        if str(environment_path.resolve()) != expected_environment_path:
            raise BuildContractError(
                f"Pinned {name} source root differs from the planned child environment."
            )
        tree_manifest = _directory_tree_manifest(dependency_dir)
        source_dependency_receipts[dependency_name] = {
            **dependency_receipt,
            "path": str(dependency_dir.resolve()),
            "environment_variable": environment_variable,
            "environment_path": expected_environment_path,
            "fetchcontent_name": str(dependency["fetchcontent_name"]),
            "tree_manifest_before_build": tree_manifest,
        }

    cargo_materialization = _materialize_cargo_dependencies(
        contract,
        source_dir,
        build_environment,
        runtime["commands"],
    )
    wheel_dir = output_dir / WHEEL_SUBDIR
    wheel_dir.mkdir(parents=True)
    build_command, frontend_command = _create_build_command(
        contract, wheel_dir, runtime["commands"]
    )
    _run(build_command, cwd=source_dir, env=build_environment)

    if toolchain_overlay_root is not None:
        toolchain_overlay_after = _inspect_toolchain_overlay(contract, toolchain_overlay_root)
        if toolchain_overlay_after != toolchain_overlay_before:
            raise BuildContractError("The verified toolchain overlay changed during build.")
        toolchain_overlay_receipt: dict[str, Any] = {
            "provided": True,
            "root": toolchain_overlay_before["root"],
            "environment": toolchain_overlay_before["environment"],
            "required_files_before_build": toolchain_overlay_before["required_files"],
            "required_files_after_build": toolchain_overlay_after["required_files"],
            "tree_manifest_before_build": toolchain_overlay_before["tree_manifest"],
            "tree_manifest_after_build": toolchain_overlay_after["tree_manifest"],
            "unchanged_after_build": True,
        }
    else:
        toolchain_overlay_receipt = {
            "provided": False,
            "mode": "system-default-search-paths",
        }

    modified_paths_after_build = _capture(
        ["git", "diff", "--name-only"], cwd=source_dir, env=build_environment
    ).splitlines()
    transform_diff_after_build = hashlib.sha256(
        _capture(["git", "diff", "--binary"], cwd=source_dir, env=build_environment).encode()
    ).hexdigest()
    if (
        modified_paths_after_build != modified_paths
        or transform_diff_after_build != transform_diff_sha256
    ):
        raise BuildContractError("The reviewed vLLM source transform changed during build.")
    if (
        _capture(["git", "rev-parse", "HEAD"], cwd=source_dir, env=build_environment)
        != source_receipt["commit"]
        or _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir, env=build_environment)
        != source_receipt["tree"]
    ):
        raise BuildContractError("The pinned vLLM source identity changed during build.")

    for name, dependency in contract["source_dependencies"].items():
        dependency_name = _safe_dependency_name(name)
        dependency_dir = dependency_root / dependency_name
        _verify_source_dependency_unchanged(
            dependency,
            dependency_dir,
            label=str(name),
            before_receipt=source_dependency_receipts[dependency_name],
            env=build_environment,
        )

    cargo_materialization = _verify_cargo_materialization_unchanged(
        contract,
        source_dir,
        build_environment,
        cargo_materialization,
    )
    cmake_cache = _inspect_cmake_cache_bindings(contract, source_dir, build_environment)

    wheels = sorted(wheel_dir.glob("vllm-*.whl"))
    if len(wheels) != 1:
        raise BuildContractError(
            f"Expected exactly one built vLLM wheel in {wheel_dir}, found {wheels}."
        )
    wheel = _inspect_wheel(contract, wheels[0])
    if _sha256_file(builder_path) != builder_sha256:
        raise BuildContractError("The vLLM builder changed during the source build.")
    if _sha256_file(contract_path) != contract_sha256:
        raise BuildContractError("The vLLM source-build contract changed during the build.")
    if _sha256_file(active_lock_path) != active_lock["sha256"]:
        raise BuildContractError("The active Python lock changed during the source build.")
    if _installed_distribution_inventory() != installed_distributions:
        raise BuildContractError(
            "The installed Python distribution inventory changed during build."
        )
    receipt = {
        "schema": "aisp.vllm-source-wheel-receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": contract_sha256,
        },
        "builder": {
            "path": str(builder_path),
            "sha256": builder_sha256,
        },
        "source": {
            **source_receipt,
            "path": str(source_dir.resolve()),
            "modified_paths": modified_paths,
            "transform_diff_sha256": transform_diff_sha256,
            "use_existing_torch": {
                "script": contract["build"]["use_existing_torch_script"],
                "arguments": contract["build"]["use_existing_torch_arguments"],
            },
            "supplemental_requirement_transform": {
                "forbidden_distribution_names": sorted(TORCH_REQUIREMENT_NAMES),
                "removed_requirements": supplemental_removals,
            },
            "deepgemm_source_import_binding": deepgemm_import_binding,
        },
        "source_dependencies": source_dependency_receipts,
        "runtime": runtime,
        "active_python_lock": active_lock,
        "installed_distributions": installed_distributions,
        "build": {
            "effective_child_environment": dict(sorted(build_environment.items())),
            "removed_inherited_environment_keys": removed_environment,
            "command": build_command,
            "frontend_command": frontend_command,
            "network_isolation": {
                "required": True,
                "mechanism": "linux-unshare-network-namespace",
                "command_prefix": build_command[:3],
                "preflight": runtime["network_isolation_preflight"],
            },
            "cmake_cache": cmake_cache,
            "cargo_materialization": cargo_materialization,
            "toolchain_overlay": toolchain_overlay_receipt,
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
        default=2,
        help="NVCC thread count, capped by the source-build contract. Default: 2.",
    )
    parser.add_argument(
        "--toolchain-overlay-root",
        type=Path,
        default=None,
        help=(
            "Optional absolute elfutils sysroot. Its declared headers, libraries, "
            "and complete tree are verified before and after the build."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
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
        toolchain_overlay_root=args.toolchain_overlay_root,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
