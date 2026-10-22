#!/usr/bin/env python3
"""Build pinned Transformer Engine and FlashAttention wheels from verified sdists.

The builder is intentionally CPU-scoped: it requires accelerators to be masked
before importing Torch, never initializes CUDA, and only compiles native extensions for SM100.
It does not install either wheel. A successful run writes a receipt containing
the source, patch, metadata, native ELF, and wheel hashes needed by the caller.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.parser import BytesParser
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "native_torch_extensions.json"
RECEIPT_NAME = "native-torch-extensions.manifest.json"
CHUNK_SIZE = 1024 * 1024

_CUDA_ROOT_ENVIRONMENT_KEYS = ("CUDA_HOME", "CUDA_PATH", "TORCH_CUDA_HOME")
_CUDA_COMPILER_ENVIRONMENT_KEYS = ("CUDACXX", "CMAKE_CUDA_COMPILER")
_PREBUILT_SOURCE_SUFFIXES = (
    ".a",
    ".cubin",
    ".dll",
    ".dylib",
    ".fatbin",
    ".lib",
    ".o",
    ".obj",
    ".pyd",
    ".whl",
)
_NATIVE_FILE_MAGICS = {
    b"\x7fELF": "ELF",
    b"MZ": "PE",
    b"\xfe\xed\xfa\xce": "Mach-O",
    b"\xce\xfa\xed\xfe": "Mach-O",
    b"\xfe\xed\xfa\xcf": "Mach-O",
    b"\xcf\xfa\xed\xfe": "Mach-O",
    b"\xca\xfe\xba\xbe": "Mach-O universal",
    b"\xbe\xba\xfe\xca": "Mach-O universal",
    b"\xca\xfe\xba\xbf": "Mach-O universal",
    b"\xbf\xba\xfe\xca": "Mach-O universal",
}
_NATIVE_SDK_SEARCH_ENVIRONMENT_KEYS = {
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
}


class BuildContractError(RuntimeError):
    """Raised when the pinned source-build contract is not satisfied."""


_SANITIZED_ENVIRONMENT_KEYS = {
    "ALL_PROXY",
    "ARCHFLAGS",
    "BUILD_TARGET",
    "CC",
    "CFLAGS",
    "CMAKE_ARGS",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_CUDA_COMPILER",
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CPPFLAGS",
    "CUDACXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "CUDA_VISIBLE_DEVICES",
    "CUDAHOSTCXX",
    "C_INCLUDE_PATH",
    "CXX",
    "CXXFLAGS",
    "FLASH_ATTENTION_FORCE_BUILD",
    "FLASH_ATTENTION_FORCE_CXX11_ABI",
    "FLASH_ATTENTION_SKIP_CK_BUILD",
    "FLASH_ATTENTION_SKIP_CUDA_BUILD",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
    "FLASH_ATTN_CUDA_ARCHS",
    "FLASH_ATTN_LOCAL_VERSION",
    "GPU_ARCHS",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LDFLAGS",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "MAX_JOBS",
    "MPI_HOME",
    "NO_PROXY",
    "NVCC_APPEND_FLAGS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_THREADS",
    "NVSHMEM_HOME",
    "NVIDIA_PRODUCT_NAME",
    "NVIDIA_PYTORCH_VERSION",
    "NVTE_BUILD_DEBUG",
    "NVTE_BUILD_MAX_JOBS",
    "NVTE_BUILD_USE_NVIDIA_WHEELS",
    "NVTE_CCACHE_BIN",
    "NVTE_CMAKE_BUILD_DIR",
    "NVTE_CUDA_ARCHS",
    "NVTE_ENABLE_NVSHMEM",
    "NVTE_FRAMEWORK",
    "NVTE_NO_LOCAL_VERSION",
    "NVTE_PYTORCH_FORCE_BUILD",
    "NVTE_PYTORCH_FORCE_CXX11_ABI",
    "NVTE_PYTORCH_SKIP_CUDA_BUILD",
    "NVTE_RELEASE_BUILD",
    "NVTE_UB_WITH_MPI",
    "NVTE_USE_CCACHE",
    "NVTE_USE_PYTORCH_TRITON",
    "NVTE_WITH_CUBLASMP",
    "NVTE_WITH_NCCL_EP",
    "OPT_DIM",
    "PIP_CONFIG_FILE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_NO_INDEX",
    "PKG_CONFIG_PATH",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "TORCH_CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
    "UV_OFFLINE",
    "XLA_HOME",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _installed_distribution_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            raise BuildContractError(
                "Installed distribution metadata is missing a name or version."
            )
        inventory.append(
            {
                "name": canonicalize_name(name),
                "version": version,
            }
        )
    return sorted(inventory, key=lambda item: (item["name"], item["version"]))


def _resolve_cuda_toolkit(nvcc: str, release: str) -> dict[str, str]:
    try:
        resolved_nvcc = Path(nvcc).resolve(strict=True)
        cuda_root = resolved_nvcc.parent.parent.resolve(strict=True)
        rooted_nvcc = (cuda_root / "bin" / "nvcc").resolve(strict=True)
    except OSError as exc:
        raise BuildContractError(f"Could not resolve the CUDA toolkit from nvcc {nvcc!r}.") from exc
    if rooted_nvcc != resolved_nvcc:
        raise BuildContractError(
            f"Validated nvcc {resolved_nvcc} is not the compiler under CUDA root {cuda_root}."
        )

    for key in _CUDA_ROOT_ENVIRONMENT_KEYS:
        inherited = os.environ.get(key)
        if inherited and Path(inherited).expanduser().resolve() != cuda_root:
            raise BuildContractError(
                f"Inherited {key}={inherited!r} conflicts with validated CUDA root {cuda_root}."
            )
    for key in _CUDA_COMPILER_ENVIRONMENT_KEYS:
        inherited = os.environ.get(key)
        if not inherited:
            continue
        candidate = shutil.which(inherited) or inherited
        if Path(candidate).expanduser().resolve() != resolved_nvcc:
            raise BuildContractError(
                f"Inherited {key}={inherited!r} conflicts with validated nvcc {resolved_nvcc}."
            )

    return {
        "root": str(cuda_root),
        "nvcc": str(resolved_nvcc),
        "nvcc_sha256": _sha256_file(resolved_nvcc),
        "release": release,
    }


def _source_tree_digest(source_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(source_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise BuildContractError(f"Source tree contains an unsafe entry: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(source_root).as_posix()
        size = path.stat().st_size
        file_sha256 = _sha256_file(path)
        manifest_entry = json.dumps(
            {"path": relative, "sha256": file_sha256, "size": size},
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(manifest_entry.encode("utf-8") + b"\n")
        file_count += 1
        total_bytes += size
    return {
        "algorithm": "sha256-file-manifest-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _prebuilt_source_reason(name: str, header: bytes = b"") -> str | None:
    basename = PurePosixPath(name).name.lower()
    if re.search(r"\.so(?:\..+)?$", basename):
        return "shared object suffix"
    if basename.endswith(_PREBUILT_SOURCE_SUFFIXES):
        return "prebuilt artifact suffix"
    for magic, label in _NATIVE_FILE_MAGICS.items():
        if header.startswith(magic):
            return f"{label} file magic"
    return None


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildContractError(f"Could not read build contract {path}: {exc}") from exc
    if not isinstance(contract, dict):
        raise BuildContractError("The native-extension contract must be a JSON object.")
    _validate_contract(contract)
    return contract


def _require_hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise BuildContractError(f"{label} must be a {length}-character lowercase hex value.")
    return value


def _require_safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise BuildContractError(f"{label} must be a non-empty relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise BuildContractError(f"{label} must be a safe relative POSIX path.")
    return path


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != "aisp.native-torch-extensions.v1":
        raise BuildContractError("Unsupported native-extension contract schema.")

    runtime = contract.get("runtime")
    build = contract.get("build")
    packages = contract.get("packages")
    if (
        not isinstance(runtime, dict)
        or not isinstance(build, dict)
        or not isinstance(packages, list)
    ):
        raise BuildContractError("Contract runtime, build, and packages fields are required.")

    if build.get("frontend") != ["python", "-m", "build", "--wheel", "--no-isolation"]:
        raise BuildContractError("Build frontend must be pinned to python -m build --no-isolation.")
    if Path(str(build.get("active_lock", ""))).name != build.get("active_lock"):
        raise BuildContractError("The active lock must be a repository-local basename.")
    if "unshare" not in build.get("required_commands", []):
        raise BuildContractError("The native build requires Linux network namespace isolation.")
    if build.get("cuda_arch_list") != "10.0" or build.get("cuda_sm") != "100":
        raise BuildContractError("This contract must target B200 explicitly as 10.0 / SM100.")
    if build.get("maximum_jobs") != 8 or build.get("maximum_nvcc_threads") != 2:
        raise BuildContractError(
            "The native build limits must cap MAX_JOBS at 8 and NVCC_THREADS at 2."
        )

    native_wheel_sdks = build.get("native_wheel_sdks")
    expected_sdk_names = {
        "nvidia-cudnn-cu13",
        "nvidia-nccl-cu13",
        "nvidia-nvshmem-cu13",
    }
    if not isinstance(native_wheel_sdks, dict) or set(native_wheel_sdks) != expected_sdk_names:
        raise BuildContractError(
            "The native build must declare the locked cuDNN, NCCL, and NVSHMEM wheel SDKs."
        )
    for distribution_name, sdk in native_wheel_sdks.items():
        if not isinstance(sdk, dict) or not isinstance(sdk.get("version"), str):
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} must declare an exact version."
            )
        try:
            Version(sdk["version"])
        except ValueError as exc:
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} has an invalid version."
            ) from exc
        root = _require_safe_relative_path(sdk.get("root"), f"{distribution_name} root")
        include_directory = _require_safe_relative_path(
            sdk.get("include_directory"), f"{distribution_name} include directory"
        )
        library_directory = _require_safe_relative_path(
            sdk.get("library_directory"), f"{distribution_name} library directory"
        )
        if not include_directory.is_relative_to(root) or not library_directory.is_relative_to(root):
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} search paths must be under its root."
            )
        assets = sdk.get("assets")
        if not isinstance(assets, dict) or set(assets) != {"headers", "libraries"}:
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} must declare headers and libraries."
            )
        for kind, directory in (("headers", include_directory), ("libraries", library_directory)):
            paths = assets[kind]
            if (
                not isinstance(paths, list)
                or not paths
                or not all(isinstance(path, str) for path in paths)
                or len(paths) != len(set(paths))
            ):
                raise BuildContractError(
                    f"Native wheel SDK {distribution_name!r} must declare unique {kind}."
                )
            for index, asset in enumerate(paths):
                asset_path = _require_safe_relative_path(
                    asset, f"{distribution_name} {kind}[{index}]"
                )
                if not asset_path.is_relative_to(directory):
                    raise BuildContractError(
                        f"Native wheel SDK {distribution_name!r} {kind} must be under {directory}."
                    )

    by_id: dict[str, Mapping[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise BuildContractError("Every package entry must have a string id.")
        package_id = package["id"]
        if package_id in by_id:
            raise BuildContractError(f"Duplicate package id {package_id!r}.")
        by_id[package_id] = package

        source = package.get("source")
        sdist = package.get("sdist")
        metadata = package.get("expected_metadata")
        if (
            not isinstance(source, dict)
            or not isinstance(sdist, dict)
            or not isinstance(metadata, dict)
        ):
            raise BuildContractError(
                f"Package {package_id!r} is missing source, sdist, or metadata."
            )
        if not isinstance(package.get("preinstalled_distributions"), dict) or not isinstance(
            package.get("forbidden_distributions"), list
        ):
            raise BuildContractError(
                f"Package {package_id!r} must declare its preinstalled and forbidden distributions."
            )
        package_environment = package.get("build_environment")
        if not isinstance(package_environment, dict):
            raise BuildContractError(
                f"Package {package_id!r} must declare a build environment object."
            )
        overridden_sdk_paths = sorted(
            set(package_environment) & _NATIVE_SDK_SEARCH_ENVIRONMENT_KEYS
        )
        if overridden_sdk_paths:
            raise BuildContractError(
                f"Package {package_id!r} may not override verified native SDK search paths: "
                f"{', '.join(overridden_sdk_paths)}."
            )
        _require_hex(
            source.get("declared_upstream_commit"),
            40,
            f"{package_id} declared upstream commit",
        )
        _require_hex(
            source.get("declared_upstream_tree"),
            40,
            f"{package_id} declared upstream tree",
        )
        if "commit" in source or "tree" in source:
            raise BuildContractError(
                f"Package {package_id!r} must label unverified Git identity as declared upstream metadata."
            )
        vendored_sources = source.get("vendored_sources", {})
        if not isinstance(vendored_sources, dict):
            raise BuildContractError(f"Package {package_id!r} vendored sources must be an object.")
        for vendored_name, vendored in vendored_sources.items():
            if not isinstance(vendored, dict):
                raise BuildContractError(
                    f"Package {package_id!r} vendored source {vendored_name!r} must be an object."
                )
            _require_hex(
                vendored.get("declared_upstream_commit"),
                40,
                f"{package_id} {vendored_name} declared upstream commit",
            )
            _require_hex(
                vendored.get("declared_upstream_tree"),
                40,
                f"{package_id} {vendored_name} declared upstream tree",
            )
            if "commit" in vendored or "tree" in vendored:
                raise BuildContractError(
                    f"Vendored source {vendored_name!r} must use declared upstream Git labels."
                )
        _require_hex(sdist.get("sha256"), 64, f"{package_id} sdist sha256")
        parsed = urlparse(str(sdist.get("url", "")))
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise BuildContractError(
                f"{package_id} sdist must use the pinned files.pythonhosted.org URL."
            )
        if Path(str(sdist.get("filename", ""))).name != sdist.get("filename"):
            raise BuildContractError(f"{package_id} sdist filename must be a basename.")

    if set(by_id) != {"transformer-engine-torch", "flash-attn"}:
        raise BuildContractError(
            "Contract must contain exactly the TE Torch and FlashAttention builds."
        )
    if by_id["transformer-engine-torch"]["preinstalled_distributions"] != {
        "transformer-engine": "2.20.2",
        "transformer-engine-cu13": "2.20.2",
    } or by_id["transformer-engine-torch"]["forbidden_distributions"] != [
        "transformer-engine-torch"
    ]:
        raise BuildContractError(
            "Transformer Engine package prerequisites do not match the contract."
        )
    if by_id["flash-attn"]["preinstalled_distributions"] or by_id["flash-attn"][
        "forbidden_distributions"
    ] != ["flash-attn", "flash-attn-4"]:
        raise BuildContractError(
            "FlashAttention must build in an environment without an installed FA2 or FA4 distribution."
        )
    te_env = by_id["transformer-engine-torch"].get("build_environment", {})
    fa_env = by_id["flash-attn"].get("build_environment", {})
    if (
        te_env.get("NVTE_BUILD_USE_NVIDIA_WHEELS") != "0"
        or te_env.get("NVTE_PYTORCH_FORCE_BUILD") != "TRUE"
        or te_env.get("NVTE_PYTORCH_SKIP_CUDA_BUILD") != "FALSE"
    ):
        raise BuildContractError("Transformer Engine must force a source CUDA binding build.")
    if (
        fa_env.get("FLASH_ATTENTION_FORCE_BUILD") != "TRUE"
        or fa_env.get("FLASH_ATTENTION_SKIP_CUDA_BUILD") != "FALSE"
    ):
        raise BuildContractError("FlashAttention must force a source CUDA build.")
    patches = by_id["flash-attn"].get("patches")
    if not isinstance(patches, list) or [patch.get("id") for patch in patches] != [
        "torch-2.13-plus-cxx20"
    ]:
        raise BuildContractError(
            "FlashAttention must carry the hash-bound C++20 compatibility patch."
        )


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise BuildContractError(f"Required distribution {name!r} is not installed.") from exc


def _locked_versions(lock_path: Path) -> dict[str, str]:
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BuildContractError(f"Could not read active lock {lock_path}: {exc}") from exc
    versions: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)\s*\\?$")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            continue
        name = canonicalize_name(match.group(1))
        version = match.group(2)
        previous = versions.setdefault(name, version)
        if previous != version:
            raise BuildContractError(
                f"Active lock contains conflicting versions for {name}: {previous} and {version}."
            )
    return versions


def _validate_native_sdk_lock(
    lock_path: Path, native_wheel_sdks: Mapping[str, Any]
) -> dict[str, str]:
    locked_versions = _locked_versions(lock_path)
    validated: dict[str, str] = {}
    for distribution_name, sdk in sorted(native_wheel_sdks.items()):
        canonical_name = canonicalize_name(distribution_name)
        expected = str(sdk["version"])
        actual = locked_versions.get(canonical_name)
        if actual != expected:
            raise BuildContractError(
                f"Active lock must contain {distribution_name}=={expected}, got {actual!r}."
            )
        validated[canonical_name] = actual
    return validated


def _record_sha256(entry: Any, label: str) -> str:
    record_hash = getattr(entry, "hash", None)
    if record_hash is None or record_hash.mode != "sha256":
        raise BuildContractError(f"{label} must have a SHA-256 hash in the wheel RECORD.")
    value = record_hash.value
    try:
        encoded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (TypeError, ValueError) as exc:
        raise BuildContractError(f"{label} has an invalid wheel RECORD hash.") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise BuildContractError(f"{label} has an invalid wheel RECORD SHA-256 length.")
    return decoded.hex()


def _resolve_required_path(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise BuildContractError(f"Required {label} is missing or inaccessible: {path}.") from exc


def _validate_native_wheel_sdks(
    native_wheel_sdks: Mapping[str, Any],
) -> dict[str, Any]:
    venv_root = _resolve_required_path(Path(sys.prefix), "active virtual environment")
    validated: dict[str, Any] = {}
    for distribution_name, sdk in sorted(native_wheel_sdks.items()):
        try:
            distribution = importlib_metadata.distribution(distribution_name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise BuildContractError(
                f"Required native wheel SDK {distribution_name!r} is not installed."
            ) from exc
        actual_name = distribution.metadata.get("Name")
        if canonicalize_name(actual_name or "") != canonicalize_name(distribution_name):
            raise BuildContractError(
                f"Native wheel SDK metadata name {actual_name!r} does not match {distribution_name!r}."
            )
        expected_version = str(sdk["version"])
        if distribution.version != expected_version:
            raise BuildContractError(
                f"{distribution_name} must be {expected_version}, got {distribution.version}."
            )

        distribution_root = _resolve_required_path(
            Path(distribution.locate_file("")), f"{distribution_name} distribution root"
        )
        if not distribution_root.is_relative_to(venv_root):
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} is outside the active virtual environment."
            )
        files = distribution.files
        if files is None:
            raise BuildContractError(
                f"Native wheel SDK {distribution_name!r} has no installed wheel RECORD."
            )
        record_entries: dict[PurePosixPath, Any] = {}
        for entry in files:
            relative_entry = PurePosixPath(str(entry))
            if relative_entry in record_entries:
                raise BuildContractError(
                    f"Native wheel SDK {distribution_name!r} has duplicate RECORD path "
                    f"{relative_entry}."
                )
            record_entries[relative_entry] = entry

        sdk_root = _resolve_required_path(
            distribution_root / str(sdk["root"]), f"{distribution_name} SDK root"
        )
        include_directory = _resolve_required_path(
            distribution_root / str(sdk["include_directory"]),
            f"{distribution_name} include directory",
        )
        library_directory = _resolve_required_path(
            distribution_root / str(sdk["library_directory"]),
            f"{distribution_name} library directory",
        )
        for label, path in (
            ("root", sdk_root),
            ("include directory", include_directory),
            ("library directory", library_directory),
        ):
            if not path.is_dir() or not path.is_relative_to(venv_root):
                raise BuildContractError(
                    f"Native wheel SDK {distribution_name!r} {label} is missing or outside the venv."
                )

        asset_receipts: list[dict[str, Any]] = []
        for kind in ("headers", "libraries"):
            for relative in sdk["assets"][kind]:
                relative_path = PurePosixPath(relative)
                entry = record_entries.get(relative_path)
                if entry is None:
                    raise BuildContractError(
                        f"{distribution_name} does not own required {kind[:-1]} {relative} in RECORD."
                    )
                path = _resolve_required_path(
                    Path(distribution.locate_file(entry)),
                    f"{distribution_name} asset {relative}",
                )
                expected_directory = include_directory if kind == "headers" else library_directory
                if not path.is_file() or not path.is_relative_to(expected_directory):
                    raise BuildContractError(
                        f"Required {distribution_name} asset {relative} is outside its verified {kind} directory."
                    )
                size = path.stat().st_size
                record_size = getattr(entry, "size", None)
                if record_size is None or int(record_size) != size:
                    raise BuildContractError(
                        f"Required {distribution_name} asset {relative} does not match its RECORD size."
                    )
                sha256 = _sha256_file(path)
                record_sha256 = _record_sha256(entry, f"{distribution_name} asset {relative}")
                if sha256 != record_sha256:
                    raise BuildContractError(
                        f"Required {distribution_name} asset {relative} does not match its RECORD hash."
                    )
                asset_receipts.append(
                    {
                        "kind": kind[:-1],
                        "relative_path": relative,
                        "path": str(path),
                        "size": size,
                        "sha256": sha256,
                        "record_sha256": record_sha256,
                    }
                )
        validated[canonicalize_name(distribution_name)] = {
            "name": actual_name,
            "version": distribution.version,
            "distribution_root": str(distribution_root),
            "sdk_root": str(sdk_root),
            "include_directory": str(include_directory),
            "library_directory": str(library_directory),
            "assets": asset_receipts,
        }
    return validated


def _assert_native_wheel_sdks_unchanged(native_wheel_sdks: Mapping[str, Any]) -> None:
    for distribution_name, sdk in native_wheel_sdks.items():
        for asset in sdk["assets"]:
            path = Path(asset["path"])
            try:
                changed = (
                    not path.is_file()
                    or path.stat().st_size != int(asset["size"])
                    or _sha256_file(path) != asset["sha256"]
                )
            except OSError:
                changed = True
            if changed:
                raise BuildContractError(
                    f"Verified native wheel SDK asset changed during the build: "
                    f"{distribution_name} {asset['relative_path']}."
                )


def _capture(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildContractError(
            f"Command failed during build preflight: {' '.join(command)}"
        ) from exc
    return completed.stdout.strip()


def _validate_environment(
    contract: Mapping[str, Any],
    selected_packages: Sequence[Mapping[str, Any]],
    *,
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int,
) -> dict[str, Any]:
    runtime = contract["runtime"]
    build = contract["build"]

    if sys.prefix == sys.base_prefix:
        raise BuildContractError(
            "Run the native-extension builder inside a dedicated virtual environment."
        )
    for key, expected in build["required_host_environment"].items():
        actual = os.environ.get(key)
        if actual != expected:
            raise BuildContractError(
                f"Host environment {key} must be {expected!r}, got {actual!r}."
            )
    if (
        platform.system() != runtime["platform_system"]
        or platform.machine() != runtime["platform_machine"]
    ):
        raise BuildContractError(
            f"Build host must be {runtime['platform_system']} {runtime['platform_machine']}."
        )
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_version != runtime["python"]:
        raise BuildContractError(f"Python must be {runtime['python']}, got {python_version}.")
    if cuda_arch_list != build["cuda_arch_list"]:
        raise BuildContractError(
            f"CUDA architecture must be {build['cuda_arch_list']}, got {cuda_arch_list}."
        )
    if not 1 <= max_jobs <= int(build["maximum_jobs"]):
        raise BuildContractError(
            f"MAX_JOBS must be between 1 and {build['maximum_jobs']}, got {max_jobs}."
        )
    if not 1 <= nvcc_threads <= int(build["maximum_nvcc_threads"]):
        raise BuildContractError(
            "NVCC_THREADS must be between 1 and "
            f"{build['maximum_nvcc_threads']}, got {nvcc_threads}."
        )

    # Importing torch after the host environment check is intentional. Do not call
    # torch.cuda APIs here: the build contract is CPU-only even on a CUDA build host.
    import torch

    if torch.__version__ != runtime["torch"]:
        raise BuildContractError(f"Torch must be {runtime['torch']}, got {torch.__version__}.")
    if torch.version.cuda != runtime["torch_cuda"]:
        raise BuildContractError(
            f"Torch CUDA metadata must be {runtime['torch_cuda']}, got {torch.version.cuda}."
        )

    versions: dict[str, str] = {}
    required_distributions = dict(build["resolved_python_requirements"])
    for package in selected_packages:
        for name, expected in package["preinstalled_distributions"].items():
            previous = required_distributions.setdefault(name, expected)
            if previous != expected:
                raise BuildContractError(
                    f"Selected packages require conflicting versions of {name}: "
                    f"{previous} and {expected}."
                )
    for name, expected in required_distributions.items():
        actual = _distribution_version(name)
        if Version(actual) != Version(expected):
            raise BuildContractError(f"{name} must be {expected}, got {actual}.")
        versions[name] = actual
    forbidden_distributions = {
        name for package in selected_packages for name in package["forbidden_distributions"]
    }
    for name in sorted(forbidden_distributions):
        try:
            actual = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        raise BuildContractError(
            f"{name}=={actual} is already installed; build in a clean environment without native bindings."
        )
    native_wheel_sdks = _validate_native_wheel_sdks(build["native_wheel_sdks"])
    for name, sdk in native_wheel_sdks.items():
        versions[name] = sdk["version"]

    commands: dict[str, str] = {}
    for name in build["required_commands"]:
        path = shutil.which(name)
        if path is None:
            raise BuildContractError(f"Required build command {name!r} was not found on PATH.")
        commands[name] = str(Path(path).resolve())

    gcc_output = _capture([commands["gcc"], "-dumpfullversion"])
    match = re.match(r"(\d+)", gcc_output)
    if match is None or int(match.group(1)) < int(build["minimum_gcc_major"]):
        raise BuildContractError(
            f"GCC {build['minimum_gcc_major']} or newer is required, got {gcc_output!r}."
        )
    nvcc_output = _capture([commands["nvcc"], "--version"])
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc_output)
    if match is None or match.group(1) != build["nvcc_release"]:
        raise BuildContractError(
            f"NVCC release must be {build['nvcc_release']}, got output {nvcc_output!r}."
        )
    cuda_toolkit = _resolve_cuda_toolkit(commands["nvcc"], match.group(1))
    network_preflight = [
        commands["unshare"],
        "--net",
        "--",
        sys.executable,
        "-c",
        "pass",
    ]
    _capture(network_preflight)

    return {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "distributions": dict(sorted(versions.items())),
        "installed_distributions": _installed_distribution_inventory(),
        "commands": commands,
        "command_sha256": {
            name: _sha256_file(Path(path)) for name, path in sorted(commands.items())
        },
        "cuda_toolkit": cuda_toolkit,
        "native_wheel_sdks": native_wheel_sdks,
        "network_isolation": {
            "mechanism": "linux-unshare-network-namespace",
            "preflight_command": network_preflight,
            "preflight": "passed",
        },
        "sanitized_inherited_environment_keys": sorted(
            key for key in _SANITIZED_ENVIRONMENT_KEYS if key in os.environ
        ),
        "gcc": gcc_output,
        "nvcc": nvcc_output,
    }


def _download_sdist(sdist: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        sdist["url"],
        headers={"User-Agent": "ai-performance-native-extension-builder/1"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open("xb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > int(sdist["size"]):
                    raise BuildContractError(
                        f"Downloaded {sdist['filename']} exceeds its pinned size."
                    )
                digest.update(chunk)
                output.write(chunk)
    except BuildContractError:
        temporary.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise BuildContractError(f"Could not download pinned sdist {sdist['url']}: {exc}") from exc
    actual_sha256 = digest.hexdigest()
    if size != int(sdist["size"]) or actual_sha256 != sdist["sha256"]:
        temporary.unlink(missing_ok=True)
        raise BuildContractError(
            f"Downloaded {sdist['filename']} does not match its pinned size and SHA-256."
        )
    temporary.replace(destination)
    return {
        "url": sdist["url"],
        "filename": sdist["filename"],
        "path": str(destination.resolve()),
        "size": size,
        "sha256": actual_sha256,
    }


def _validate_archive_name(name: str, expected_root: str) -> tuple[str, ...]:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise BuildContractError(f"Unsafe tar member name {name!r}.")
    parts = tuple(name.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise BuildContractError(f"Unsafe tar member name {name!r}.")
    if parts[0] != expected_root:
        raise BuildContractError(
            f"Tar member {name!r} is outside the expected root {expected_root!r}."
        )
    if PurePosixPath(*parts).is_absolute():
        raise BuildContractError(f"Unsafe absolute tar member {name!r}.")
    return parts


def _safe_extract_sdist(archive_path: Path, destination: Path, sdist: Mapping[str, Any]) -> Path:
    if destination.exists() or destination.is_symlink():
        raise BuildContractError(f"Extraction destination already exists: {destination}")
    expected_root = str(sdist["root"])
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        files = [member for member in members if member.isfile()]
        directories = [member for member in members if member.isdir()]
        if len(members) != int(sdist["member_count"]):
            raise BuildContractError("Sdist tar member count does not match the contract.")
        if len(files) != int(sdist["file_count"]) or len(directories) != int(
            sdist["directory_count"]
        ):
            raise BuildContractError("Sdist file or directory count does not match the contract.")
        if sum(member.size for member in files) != int(sdist["expanded_file_bytes"]):
            raise BuildContractError("Sdist expanded byte count does not match the contract.")

        seen: set[tuple[str, ...]] = set()
        file_paths: set[tuple[str, ...]] = set()
        validated: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
        for member in members:
            if not member.isfile() and not member.isdir():
                raise BuildContractError(
                    f"Sdist member {member.name!r} is not a regular file or directory."
                )
            parts = _validate_archive_name(member.name, expected_root)
            if parts in seen:
                raise BuildContractError(f"Duplicate sdist member {member.name!r}.")
            if any(parts[:index] in file_paths for index in range(1, len(parts))):
                raise BuildContractError(f"Sdist member has a file parent: {member.name!r}.")
            if member.isfile() and (reason := _prebuilt_source_reason(member.name)):
                raise BuildContractError(
                    f"Sdist contains a prebuilt artifact ({reason}): {member.name!r}."
                )
            seen.add(parts)
            if member.isfile():
                file_paths.add(parts)
            validated.append((member, parts))
        for file_path in file_paths:
            if any(
                len(other) > len(file_path) and other[: len(file_path)] == file_path
                for other in seen
            ):
                raise BuildContractError(
                    f"Sdist file is also the parent of another member: {'/'.join(file_path)!r}."
                )

        destination.mkdir(parents=True, mode=0o755)
        for member, parts in validated:
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            source = archive.extractfile(member)
            if source is None:
                raise BuildContractError(f"Could not read sdist member {member.name!r}.")
            copied = 0
            with source, target.open("xb") as output:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    if copied == 0 and (reason := _prebuilt_source_reason(member.name, chunk[:4])):
                        raise BuildContractError(
                            f"Sdist contains a prebuilt artifact ({reason}): {member.name!r}."
                        )
                    copied += len(chunk)
                    if copied > member.size:
                        raise BuildContractError(
                            f"Sdist member grew during extraction: {member.name!r}."
                        )
                    output.write(chunk)
            if copied != member.size:
                raise BuildContractError(f"Sdist member size changed: {member.name!r}.")
            mode = member.mode & 0o777
            target.chmod(mode or 0o644)

    source_root = destination / expected_root
    if not source_root.is_dir():
        raise BuildContractError(f"Expected extracted source root is missing: {source_root}")
    return source_root


def _verify_source_files(
    source_root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for relative, details in expected.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise BuildContractError(f"Required source file is missing or unsafe: {relative}")
        size = path.stat().st_size
        sha256 = _sha256_file(path)
        if size != int(details["size"]) or sha256 != details["sha256"]:
            raise BuildContractError(f"Source file does not match contract: {relative}")
        receipt[relative] = {"size": size, "sha256": sha256}
    return receipt


def _apply_patches(source_root: Path, patches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    receipt: list[dict[str, Any]] = []
    for patch in patches:
        relative = str(patch["path"])
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise BuildContractError(f"Patch target is missing or unsafe: {relative}")
        before_sha256 = _sha256_file(path)
        if before_sha256 != patch["before_sha256"]:
            raise BuildContractError(f"Patch input hash does not match for {relative}.")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise BuildContractError(f"Patch target is not UTF-8: {relative}") from exc
        for operation in patch["replacements"]:
            before = operation["before"]
            expected_count = int(operation["count"])
            actual_count = text.count(before)
            if actual_count != expected_count:
                raise BuildContractError(
                    f"Patch {patch['id']} expected {expected_count} matches, got {actual_count}."
                )
            text = text.replace(before, operation["after"], expected_count)
        if path.suffix == ".py":
            compile(text, str(path), "exec")
        path.write_bytes(text.encode("utf-8"))
        after_sha256 = _sha256_file(path)
        if after_sha256 != patch["after_sha256"]:
            raise BuildContractError(f"Patch output hash does not match for {relative}.")
        receipt.append(
            {
                "id": patch["id"],
                "path": relative,
                "reason": patch["reason"],
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "operation_count": len(patch["replacements"]),
                "upstream_issue": patch.get("upstream_issue"),
                "upstream_fix_commit": patch.get("upstream_fix_commit"),
                "pytorch_release_matrix": patch.get("pytorch_release_matrix"),
            }
        )
    return receipt


def _normalized_requirement(value: str) -> tuple[str, tuple[str, ...], str, str, str]:
    requirement = Requirement(value)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url or "",
        str(requirement.marker) if requirement.marker is not None else "",
    )


def _inspect_native_member(
    archive: zipfile.ZipFile, member: str, minimum_bytes: int
) -> dict[str, Any]:
    info = archive.getinfo(member)
    if info.file_size < minimum_bytes:
        raise BuildContractError(
            f"Native wheel member {member!r} is too small ({info.file_size} bytes)."
        )
    digest = hashlib.sha256()
    read_bytes = 0
    with archive.open(info, "r") as source:
        header = source.read(64)
        digest.update(header)
        read_bytes += len(header)
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            read_bytes += len(chunk)
    if read_bytes != info.file_size:
        raise BuildContractError(f"Native wheel member size changed while reading {member!r}.")
    if len(header) < 64 or header[:7] != b"\x7fELF\x02\x01\x01":
        raise BuildContractError(
            f"Native wheel member {member!r} is not an ELF64 little-endian x86_64 binary."
        )
    (
        _ident,
        elf_type,
        elf_machine,
        elf_version,
        _entry,
        program_header_offset,
        _section_header_offset,
        _flags,
        elf_header_size,
        program_header_entry_size,
        program_header_count,
        _section_header_entry_size,
        _section_header_count,
        _section_name_index,
    ) = struct.unpack("<16sHHIQQQIHHHHHH", header)
    program_header_table_size = program_header_entry_size * program_header_count
    program_header_end = program_header_offset + program_header_table_size
    if (
        elf_type != 3
        or elf_machine != 62
        or elf_version != 1
        or not 64 <= elf_header_size <= info.file_size
        or program_header_offset < elf_header_size
        or not 56 <= program_header_entry_size <= 4096
        or not 1 <= program_header_count <= 4096
        or program_header_table_size > 16 * CHUNK_SIZE
        or program_header_end > info.file_size
    ):
        raise BuildContractError(
            f"Native wheel member {member!r} is not a sane x86_64 ET_DYN ELF image."
        )

    with archive.open(info, "r") as source:
        source.seek(program_header_offset)
        program_headers = source.read(program_header_table_size)
    if len(program_headers) != program_header_table_size:
        raise BuildContractError(f"Native wheel member {member!r} has a truncated program table.")
    executable_load_segments = 0
    for index in range(program_header_count):
        offset = index * program_header_entry_size
        (
            segment_type,
            segment_flags,
            segment_offset,
            _virtual_address,
            _physical_address,
            segment_file_size,
            segment_memory_size,
            _alignment,
        ) = struct.unpack_from("<IIQQQQQQ", program_headers, offset)
        if segment_type != 1 or not segment_flags & 0x1:
            continue
        if (
            segment_file_size <= 0
            or segment_memory_size < segment_file_size
            or segment_offset > info.file_size
            or segment_file_size > info.file_size - segment_offset
        ):
            raise BuildContractError(
                f"Native wheel member {member!r} has an invalid executable PT_LOAD segment."
            )
        executable_load_segments += 1
    if executable_load_segments == 0:
        raise BuildContractError(
            f"Native wheel member {member!r} has no executable PT_LOAD segment."
        )
    return {
        "path": member,
        "size": info.file_size,
        "sha256": digest.hexdigest(),
        "elf_class": "ELF64",
        "elf_data": "little-endian",
        "elf_type": "ET_DYN",
        "elf_machine": "x86_64",
        "program_header_count": program_header_count,
        "executable_load_segments": executable_load_segments,
    }


def _inspect_wheel(package: Mapping[str, Any], wheel_path: Path) -> dict[str, Any]:
    expected = package["expected_metadata"]
    expected_tag = expected["wheel_tag"]
    if not wheel_path.name.endswith(f"-{expected_tag}.whl"):
        raise BuildContractError(
            f"Wheel {wheel_path.name!r} does not carry required tag {expected_tag!r}."
        )

    with zipfile.ZipFile(wheel_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise BuildContractError(f"Wheel {wheel_path.name!r} has duplicate members.")
        for name in names:
            if name.startswith("/") or "\\" in name or ".." in name.split("/"):
                raise BuildContractError(f"Wheel {wheel_path.name!r} has unsafe member {name!r}.")

        metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_members = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_members) != 1 or len(wheel_members) != 1:
            raise BuildContractError("Wheel must contain exactly one METADATA and one WHEEL file.")
        metadata_info = archive.getinfo(metadata_members[0])
        wheel_info = archive.getinfo(wheel_members[0])
        if metadata_info.file_size > CHUNK_SIZE or wheel_info.file_size > CHUNK_SIZE:
            raise BuildContractError("Wheel metadata is unexpectedly large.")
        metadata = BytesParser().parsebytes(archive.read(metadata_info))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_info))

        actual_name = metadata.get("Name")
        actual_version = metadata.get("Version")
        actual_requires_python = metadata.get("Requires-Python")
        if canonicalize_name(actual_name or "") != canonicalize_name(expected["name"]):
            raise BuildContractError(f"Unexpected wheel package name {actual_name!r}.")
        if actual_version is None or Version(actual_version) != Version(expected["version"]):
            raise BuildContractError(f"Unexpected wheel package version {actual_version!r}.")
        if actual_requires_python is None or SpecifierSet(actual_requires_python) != SpecifierSet(
            expected["requires_python"]
        ):
            raise BuildContractError(
                f"Unexpected Requires-Python metadata {actual_requires_python!r}."
            )
        actual_requires_dist = metadata.get_all("Requires-Dist") or []
        actual_requirements = Counter(
            _normalized_requirement(item) for item in actual_requires_dist
        )
        expected_requirements = Counter(
            _normalized_requirement(item) for item in expected["requires_dist"]
        )
        if actual_requirements != expected_requirements:
            raise BuildContractError("Wheel Requires-Dist metadata does not match the contract.")
        tags = wheel_metadata.get_all("Tag") or []
        if tags != [expected_tag]:
            raise BuildContractError(f"Wheel tags do not match the contract: {tags!r}.")
        if (wheel_metadata.get("Root-Is-Purelib") or "").lower() != "false":
            raise BuildContractError("Native extension wheel was marked as pure Python.")

        native_candidates = sorted(name for name in names if name.endswith(".so"))
        if len(native_candidates) != int(expected["native_member_count"]):
            raise BuildContractError(
                f"Wheel has {len(native_candidates)} native members; "
                f"expected {expected['native_member_count']}."
            )
        required_native_members: list[str] = []
        for prefix in expected["required_native_member_prefixes"]:
            matches = [
                name
                for name in native_candidates
                if name.startswith(f"{prefix}.") and name.endswith(".so")
            ]
            if len(matches) != 1:
                raise BuildContractError(
                    f"Wheel requires exactly one native member for prefix {prefix!r}."
                )
            required_native_members.extend(matches)
        if sorted(required_native_members) != native_candidates:
            raise BuildContractError("Wheel contains native members outside the required prefixes.")
        native_details = [
            _inspect_native_member(archive, member, int(expected["minimum_native_member_bytes"]))
            for member in native_candidates
        ]

    return {
        "path": str(wheel_path.resolve()),
        "filename": wheel_path.name,
        "size": wheel_path.stat().st_size,
        "sha256": _sha256_file(wheel_path),
        "metadata": {
            "name": actual_name,
            "version": actual_version,
            "requires_python": actual_requires_python,
            "requires_dist": actual_requires_dist,
            "tag": expected_tag,
        },
        "native_members": native_details,
    }


def _explicit_build_environment(
    contract: Mapping[str, Any],
    package: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    max_jobs: int,
    nvcc_threads: int,
) -> dict[str, str]:
    build = contract["build"]
    cuda_toolkit = runtime["cuda_toolkit"]
    native_wheel_sdks = runtime["native_wheel_sdks"]
    include_directories = list(
        dict.fromkeys(sdk["include_directory"] for _, sdk in sorted(native_wheel_sdks.items()))
    )
    library_directories = list(
        dict.fromkeys(sdk["library_directory"] for _, sdk in sorted(native_wheel_sdks.items()))
    )
    include_search_path = os.pathsep.join(include_directories)
    library_search_path = os.pathsep.join(library_directories)
    explicit = {
        "CC": runtime["commands"]["gcc"],
        "CMAKE_INCLUDE_PATH": include_search_path,
        "CMAKE_LIBRARY_PATH": library_search_path,
        "CMAKE_CUDA_COMPILER": cuda_toolkit["nvcc"],
        "CPATH": include_search_path,
        "CPLUS_INCLUDE_PATH": include_search_path,
        "CUDACXX": cuda_toolkit["nvcc"],
        "CUDA_HOME": cuda_toolkit["root"],
        "CUDA_PATH": cuda_toolkit["root"],
        "CXX": runtime["commands"]["g++"],
        "CUDA_VISIBLE_DEVICES": build["required_host_environment"]["CUDA_VISIBLE_DEVICES"],
        "MAX_JOBS": str(max_jobs),
        "LIBRARY_PATH": library_search_path,
        "NVCC_THREADS": str(nvcc_threads),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "TORCH_CUDA_HOME": cuda_toolkit["root"],
        "TORCH_CUDA_ARCH_LIST": build["cuda_arch_list"],
        "TORCH_DEVICE_BACKEND_AUTOLOAD": build["required_host_environment"][
            "TORCH_DEVICE_BACKEND_AUTOLOAD"
        ],
        "UV_OFFLINE": "1",
    }
    explicit.update({key: str(value) for key, value in package["build_environment"].items()})
    return explicit


def _build_package(
    contract: Mapping[str, Any],
    package: Mapping[str, Any],
    source_root: Path,
    wheel_root: Path,
    runtime: Mapping[str, Any],
    *,
    max_jobs: int,
    nvcc_threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_id = package["id"]
    destination = wheel_root / package_id
    destination.mkdir(parents=True, exist_ok=False)
    explicit = _explicit_build_environment(
        contract,
        package,
        runtime,
        max_jobs=max_jobs,
        nvcc_threads=nvcc_threads,
    )
    environment = os.environ.copy()
    sanitized_inherited_keys = sorted(
        key for key in _SANITIZED_ENVIRONMENT_KEYS if key in environment
    )
    for key in _SANITIZED_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment.update(explicit)
    frontend_command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(destination.resolve()),
        ".",
    ]
    command = [runtime["commands"]["unshare"], "--net", "--", *frontend_command]
    started_at = _utc_now()
    try:
        subprocess.run(command, cwd=source_root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildContractError(f"Source build failed for {package_id}.") from exc
    wheels = sorted(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildContractError(f"Expected one wheel for {package_id}, found {len(wheels)}.")
    finished_at = _utc_now()
    return _inspect_wheel(package, wheels[0]), {
        "command": command,
        "frontend_command": frontend_command,
        "environment": explicit,
        "sanitized_inherited_environment_keys": sanitized_inherited_keys,
        "network_isolation": "linux-unshare-network-namespace",
        "dependency_installation": "disabled by python -m build --no-isolation",
        "started_at": started_at,
        "finished_at": finished_at,
    }


def build_extensions(
    contract_path: Path,
    output_dir: Path,
    *,
    package_ids: Sequence[str],
    cuda_arch_list: str,
    max_jobs: int,
    nvcc_threads: int,
) -> Path:
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise BuildContractError(f"Output directory must not already exist: {output_dir}")
    builder_path = Path(__file__).resolve()
    contract_sha256 = _sha256_file(contract_path)
    builder_sha256 = _sha256_file(builder_path)
    contract = _load_contract(contract_path)
    packages_by_id = {package["id"]: package for package in contract["packages"]}
    if not package_ids:
        raise BuildContractError("Select at least one package to build.")
    if len(package_ids) != len(set(package_ids)):
        raise BuildContractError("Each package may be selected only once.")
    unknown = sorted(set(package_ids) - set(packages_by_id))
    if unknown:
        raise BuildContractError(f"Unknown package selection: {', '.join(unknown)}")
    selected_packages = [packages_by_id[package_id] for package_id in package_ids]
    unresolved_lock_path = contract_path.parent / contract["build"]["active_lock"]
    if not unresolved_lock_path.is_file() or unresolved_lock_path.is_symlink():
        raise BuildContractError(f"Active lock is missing or unsafe: {unresolved_lock_path}")
    lock_path = unresolved_lock_path.resolve()
    lock_sha256 = _sha256_file(lock_path)
    native_sdk_lock_versions = _validate_native_sdk_lock(
        lock_path, contract["build"]["native_wheel_sdks"]
    )
    runtime = _validate_environment(
        contract,
        selected_packages,
        cuda_arch_list=cuda_arch_list,
        max_jobs=max_jobs,
        nvcc_threads=nvcc_threads,
    )
    runtime["active_lock"] = {
        "path": str(lock_path),
        "sha256": lock_sha256,
        "native_wheel_sdk_versions": native_sdk_lock_versions,
    }
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o755)
    sdist_root = output_dir / "sdists"
    source_root = output_dir / "source"
    wheel_root = output_dir / "wheel"
    sdist_root.mkdir()
    source_root.mkdir()
    wheel_root.mkdir()

    build_started_at = _utc_now()
    package_receipts: dict[str, Any] = {}
    for package in selected_packages:
        package_id = package["id"]
        sdist_path = sdist_root / package["sdist"]["filename"]
        downloaded = _download_sdist(package["sdist"], sdist_path)
        extracted = _safe_extract_sdist(sdist_path, source_root / package_id, package["sdist"])
        verified_files = _verify_source_files(extracted, package["verified_files"])
        patches = _apply_patches(extracted, package["patches"])
        post_patch_tree = _source_tree_digest(extracted)
        _assert_native_wheel_sdks_unchanged(runtime["native_wheel_sdks"])
        wheel, build_receipt = _build_package(
            contract,
            package,
            extracted,
            wheel_root,
            runtime,
            max_jobs=max_jobs,
            nvcc_threads=nvcc_threads,
        )
        package_receipts[package_id] = {
            "source": {
                "declared_upstream": package["source"],
                "directory": str(extracted.resolve()),
                "verified_sdist": {
                    **downloaded,
                    "archive": {
                        key: package["sdist"][key]
                        for key in (
                            "root",
                            "member_count",
                            "file_count",
                            "directory_count",
                            "expanded_file_bytes",
                        )
                    },
                },
                "verified_files": verified_files,
                "patches": patches,
                "post_patch_pre_build_tree": post_patch_tree,
            },
            "build": build_receipt,
            "wheel": wheel,
        }

    build_finished_at = _utc_now()
    _assert_native_wheel_sdks_unchanged(runtime["native_wheel_sdks"])
    if _sha256_file(contract_path) != contract_sha256:
        raise BuildContractError("Build contract changed while native wheels were being built.")
    if _sha256_file(builder_path) != builder_sha256:
        raise BuildContractError("Native builder changed while native wheels were being built.")
    if _sha256_file(lock_path) != lock_sha256:
        raise BuildContractError("Active lock changed while native wheels were being built.")
    receipt_path = output_dir / RECEIPT_NAME
    receipt = {
        "schema": "aisp.native-torch-extensions-receipt.v1",
        "contract_sha256": contract_sha256,
        "builder_sha256": builder_sha256,
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha256,
        },
        "builder": {
            "path": str(builder_path),
            "sha256": builder_sha256,
        },
        "runtime": runtime,
        "build": {
            "cuda_arch_list": cuda_arch_list,
            "max_jobs": max_jobs,
            "nvcc_threads": nvcc_threads,
            "started_at": build_started_at,
            "finished_at": build_finished_at,
        },
        "selected_packages": list(package_ids),
        "packages": package_receipts,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=CONTRACT_PATH,
        help=f"Build contract (default: {CONTRACT_PATH})",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package",
        dest="packages",
        action="append",
        choices=("transformer-engine-torch", "flash-attn"),
        required=True,
        help="Package to build. Repeat to build both into the same fresh output root.",
    )
    parser.add_argument("--cuda-arch-list", required=True)
    parser.add_argument("--max-jobs", type=int, required=True)
    parser.add_argument("--nvcc-threads", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = build_extensions(
            args.contract,
            args.output_dir,
            package_ids=args.packages,
            cuda_arch_list=args.cuda_arch_list,
            max_jobs=args.max_jobs,
            nvcc_threads=args.nvcc_threads,
        )
    except BuildContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
