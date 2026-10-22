#!/usr/bin/env python3
"""Build the Torch 2.14-only torch-c-dlpack-ext compatibility wheel.

The builder verifies the upstream sdist, applies two reviewed source patches,
and builds one pure-Python wheel in an offline Linux network namespace. It
never installs the wheel and never calls a CUDA availability API.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "torch_dlpack_source_build.json"
RECEIPT_NAME = "torch-dlpack-source-wheel.manifest.json"
CHUNK_SIZE = 1024 * 1024
PACKAGE_NAME = "torch-c-dlpack-ext"
UPSTREAM_VERSION = "0.1.5"
OUTPUT_VERSION = "0.1.5+torch2141.nobackend1"
TORCH_BUILD_VERSION = "2.14.1+cu130"
TORCH_RUNTIME_REQUIREMENT = "torch==2.14.1"

_PYPROJECT_REPLACEMENTS = [
    {
        "count": 1,
        "before": 'version = "0.1.5"',
        "after": 'version = "0.1.5+torch2141.nobackend1"',
    },
    {
        "count": 1,
        "before": 'dependencies = ["torch"]',
        "after": 'dependencies = ["torch==2.14.1"]',
    },
    {
        "count": 1,
        "before": 'py-modules = ["build_backend"]',
        "after": "py-modules = []",
    },
]

_CORE_BEFORE = """    if hasattr(torch.Tensor, "__dlpack_c_exchange_api__") or hasattr(
        torch.Tensor, "__c_dlpack_exchange_api__"
    ):
        return None
"""
_CORE_AFTER = """    if hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        return None
    raise RuntimeError(
        "torch-c-dlpack-ext 0.1.5+torch2141.nobackend1 requires "
        "torch.Tensor.__dlpack_c_exchange_api__. Legacy native fallbacks "
        "are intentionally absent."
    )
"""
_CORE_REPLACEMENTS = [{"count": 1, "before": _CORE_BEFORE, "after": _CORE_AFTER}]

_SANITIZED_ENVIRONMENT_KEYS = {
    "ALL_PROXY",
    "CFLAGS",
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CPPFLAGS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "C_INCLUDE_PATH",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LDFLAGS",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NO_PROXY",
    "PIP_CONFIG_FILE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_NO_INDEX",
    "PIP_TRUSTED_HOST",
    "PYTHONHOME",
    "PYTHONPATH",
    "SETUPTOOLS_SCM_PRETEND_VERSION",
    "SOURCE_DATE_EPOCH",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
    "UV_OFFLINE",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}

_NATIVE_SUFFIXES = (
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
_NATIVE_MAGICS = {
    b"\x7fELF": "ELF",
    b"MZ": "PE",
    b"\xfe\xed\xfa\xce": "Mach-O",
    b"\xce\xfa\xed\xfe": "Mach-O",
    b"\xfe\xed\xfa\xcf": "Mach-O",
    b"\xcf\xfa\xed\xfe": "Mach-O",
}


class BuildContractError(RuntimeError):
    """Raised when the source-build contract is not satisfied."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise BuildContractError(f"{label} must be {length} lowercase hexadecimal characters.")
    return value


def _load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildContractError(f"Could not read build contract {path}: {exc}") from exc
    if not isinstance(contract, dict):
        raise BuildContractError("The DLPack build contract must be a JSON object.")
    _validate_contract(contract)
    return contract


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != "aisp.torch-dlpack-source-build.v1":
        raise BuildContractError("Unsupported DLPack build contract schema.")

    source = contract.get("source")
    sdist = contract.get("sdist")
    runtime = contract.get("runtime")
    build = contract.get("build")
    patches = contract.get("patches")
    wheel = contract.get("expected_wheel")
    if not all(isinstance(value, dict) for value in (source, sdist, runtime, build, wheel)):
        raise BuildContractError("Contract source, sdist, runtime, build, and wheel fields are required.")
    if not isinstance(patches, list):
        raise BuildContractError("Contract patches must be a list.")

    if source.get("repository") != "https://github.com/apache/tvm-ffi":
        raise BuildContractError("The source repository must be the Apache TVM FFI repository.")
    _require_hex(source.get("declared_upstream_commit"), 40, "declared upstream commit")
    if source.get("upstream_version") != UPSTREAM_VERSION:
        raise BuildContractError(f"Upstream version must be {UPSTREAM_VERSION}.")

    parsed_url = urlparse(str(sdist.get("url", "")))
    if parsed_url.scheme != "https" or parsed_url.hostname != "files.pythonhosted.org":
        raise BuildContractError("The sdist must use its pinned files.pythonhosted.org URL.")
    if Path(str(sdist.get("filename", ""))).name != sdist.get("filename"):
        raise BuildContractError("The sdist filename must be a basename.")
    _require_hex(sdist.get("sha256"), 64, "sdist sha256")
    if sdist.get("root") != "torch_c_dlpack_ext-0.1.5":
        raise BuildContractError("The sdist root does not match the pinned release.")
    for key in ("size", "member_count", "file_count", "directory_count", "expanded_file_bytes"):
        if not isinstance(sdist.get(key), int) or int(sdist[key]) <= 0:
            raise BuildContractError(f"The sdist {key} must be a positive integer.")

    verified_files = contract.get("verified_files")
    if not isinstance(verified_files, dict) or not verified_files:
        raise BuildContractError("The contract must hash every regular sdist file.")
    if len(verified_files) != sdist["file_count"]:
        raise BuildContractError("The verified file map must cover every sdist file.")
    for relative, identity in verified_files.items():
        if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise BuildContractError(f"Unsafe verified source path {relative!r}.")
        if not isinstance(identity, dict) or not isinstance(identity.get("size"), int):
            raise BuildContractError(f"Verified source identity for {relative!r} is incomplete.")
        _require_hex(identity.get("sha256"), 64, f"verified source {relative} sha256")

    expected_patches = [
        ("torch2141-pure-wheel-metadata", "pyproject.toml", _PYPROJECT_REPLACEMENTS),
        ("torch2141-no-legacy-fallback", "torch_c_dlpack_ext/core.py", _CORE_REPLACEMENTS),
    ]
    if len(patches) != len(expected_patches):
        raise BuildContractError("The contract must contain the two reviewed DLPack patches.")
    for patch, (patch_id, path, replacements) in zip(patches, expected_patches, strict=True):
        if patch.get("id") != patch_id or patch.get("path") != path:
            raise BuildContractError(f"Unexpected patch identity for {patch_id!r}.")
        if patch.get("replacements") != replacements:
            raise BuildContractError(f"Patch {patch_id!r} does not match the reviewed replacements.")
        _require_hex(patch.get("before_sha256"), 64, f"{patch_id} before sha256")
        _require_hex(patch.get("after_sha256"), 64, f"{patch_id} after sha256")
        if patch["before_sha256"] != verified_files[path]["sha256"]:
            raise BuildContractError(f"Patch {patch_id!r} is not bound to the verified source file.")

    expected_tree = contract.get("expected_post_patch_tree")
    if not isinstance(expected_tree, dict):
        raise BuildContractError("The expected post-patch source tree is required.")
    _require_hex(expected_tree.get("sha256"), 64, "post-patch tree sha256")
    if expected_tree.get("file_count") != sdist["file_count"]:
        raise BuildContractError("The post-patch tree file count must match the sdist.")
    if not isinstance(expected_tree.get("total_bytes"), int) or expected_tree["total_bytes"] <= 0:
        raise BuildContractError("The post-patch tree byte count is invalid.")

    if runtime != {
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "python": "3.12",
        "torch": TORCH_BUILD_VERSION,
        "required_tensor_attribute": "__dlpack_c_exchange_api__",
    }:
        raise BuildContractError("The runtime must bind Linux CPython 3.12 and Torch 2.14.1 CUDA 13.")
    if build.get("frontend") != ["python", "-m", "build", "--wheel", "--no-isolation"]:
        raise BuildContractError("The build frontend must be python -m build without isolation.")
    if build.get("source_date_epoch") != 1768217108:
        raise BuildContractError("SOURCE_DATE_EPOCH must match the verified sdist publication second.")
    if build.get("resolved_python_requirements") != {
        "apache-tvm-ffi": "0.1.11",
        "build": "1.6.1",
        "packaging": "26.2",
        "setuptools": "80.10.2",
        "wheel": "0.48.0",
    }:
        raise BuildContractError("The exact build frontend requirements do not match the contract.")
    if build.get("required_commands") != ["unshare"]:
        raise BuildContractError("Linux unshare is the only external build command.")
    if build.get("required_host_environment") != {
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }:
        raise BuildContractError("The build must mask CUDA and backend autoloading.")

    if wheel.get("name") != PACKAGE_NAME or wheel.get("version") != OUTPUT_VERSION:
        raise BuildContractError("The derived wheel name or version is incorrect.")
    if wheel.get("filename") != (
        "torch_c_dlpack_ext-0.1.5+torch2141.nobackend1-py3-none-any.whl"
    ):
        raise BuildContractError("The derived wheel filename is incorrect.")
    if wheel.get("tag") != "py3-none-any" or wheel.get("root_is_purelib") is not True:
        raise BuildContractError("The derived wheel must be pure Python and platform independent.")
    if wheel.get("requires_python") != ">=3.9" or wheel.get("requires_dist") != [
        TORCH_RUNTIME_REQUIREMENT
    ]:
        raise BuildContractError("The derived wheel must require the public Torch 2.14.1 version.")
    if Version(TORCH_BUILD_VERSION) not in Requirement(TORCH_RUNTIME_REQUIREMENT).specifier:
        raise BuildContractError("The public Torch requirement must accept the CUDA local version.")
    package_files = wheel.get("package_files")
    license_files = wheel.get("license_files")
    for label, files in (("package", package_files), ("license", license_files)):
        if not isinstance(files, dict) or not files:
            raise BuildContractError(f"Expected {label} wheel files are required.")
        for relative, identity in files.items():
            if not isinstance(identity, dict) or not isinstance(identity.get("size"), int):
                raise BuildContractError(f"Expected wheel identity for {relative!r} is incomplete.")
            _require_hex(identity.get("sha256"), 64, f"expected wheel {relative} sha256")
    if wheel.get("top_level") != "torch_c_dlpack_ext\n":
        raise BuildContractError("The wheel top-level declaration must contain only the package.")
    if wheel.get("forbidden_top_level_modules") != ["build_backend.py", "build_utils.py"]:
        raise BuildContractError("The forbidden generic top-level modules are incomplete.")


def _native_reason(name: str, header: bytes = b"") -> str | None:
    basename = PurePosixPath(name).name.lower()
    if re.search(r"\.so(?:\..+)?$", basename):
        return "shared object suffix"
    if basename.endswith(_NATIVE_SUFFIXES):
        return "native artifact suffix"
    for magic, label in _NATIVE_MAGICS.items():
        if header.startswith(magic):
            return f"{label} file magic"
    return None


def _download_sdist(sdist: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(sdist["url"], headers={"User-Agent": "ai-performance-dlpack-builder/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open("xb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > sdist["size"]:
                    raise BuildContractError("The downloaded sdist exceeds its pinned size.")
                digest.update(chunk)
                output.write(chunk)
            final_url = response.geturl()
    except BuildContractError:
        temporary.unlink(missing_ok=True)
        raise
    except (HTTPError, URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise BuildContractError(f"Could not download the pinned sdist: {exc}") from exc
    actual_sha256 = digest.hexdigest()
    if size != sdist["size"] or actual_sha256 != sdist["sha256"]:
        temporary.unlink(missing_ok=True)
        raise BuildContractError("The downloaded sdist does not match its pinned size and hash.")
    if final_url != sdist["url"]:
        temporary.unlink(missing_ok=True)
        raise BuildContractError(f"The sdist URL redirected unexpectedly to {final_url!r}.")
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "url": final_url,
        "filename": destination.name,
        "size": size,
        "sha256": actual_sha256,
    }


def _archive_parts(name: str, expected_root: str) -> tuple[str, ...]:
    if not name or "\\" in name or name.startswith("/"):
        raise BuildContractError(f"Unsafe sdist member path {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BuildContractError(f"Unsafe sdist member path {name!r}.")
    if not path.parts or path.parts[0] != expected_root:
        raise BuildContractError(f"Sdist member {name!r} is outside the expected root.")
    return path.parts


def _safe_extract_sdist(
    archive_path: Path,
    destination: Path,
    sdist: Mapping[str, Any],
) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    file_count = 0
    directory_count = 0
    expanded_bytes = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != sdist["member_count"]:
                raise BuildContractError("The sdist member count does not match the contract.")
            for member in members:
                parts = _archive_parts(member.name, sdist["root"])
                normalized = PurePosixPath(*parts).as_posix()
                if normalized in seen:
                    raise BuildContractError(f"Duplicate sdist member {normalized!r}.")
                seen.add(normalized)
                target = destination.joinpath(*parts)
                if member.isdir():
                    directory_count += 1
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise BuildContractError(f"Sdist member {normalized!r} is not a regular file.")
                reason = _native_reason(normalized)
                if reason:
                    raise BuildContractError(f"Sdist member {normalized!r} has a forbidden {reason}.")
                file_count += 1
                expanded_bytes += member.size
                if expanded_bytes > sdist["expanded_file_bytes"]:
                    raise BuildContractError("The sdist expands beyond its pinned byte count.")
                source = archive.extractfile(member)
                if source is None:
                    raise BuildContractError(f"Could not read sdist member {normalized!r}.")
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = source.read()
                if len(payload) != member.size:
                    raise BuildContractError(f"Sdist member {normalized!r} was truncated.")
                reason = _native_reason(normalized, payload[:8])
                if reason:
                    raise BuildContractError(f"Sdist member {normalized!r} has forbidden {reason}.")
                with target.open("xb") as output:
                    output.write(payload)
                target.chmod(0o644)
    except (tarfile.TarError, OSError) as exc:
        raise BuildContractError(f"Could not safely extract the sdist: {exc}") from exc
    if (
        file_count != sdist["file_count"]
        or directory_count != sdist["directory_count"]
        or expanded_bytes != sdist["expanded_file_bytes"]
    ):
        raise BuildContractError("The extracted sdist counts do not match the contract.")
    root = destination / sdist["root"]
    if not root.is_dir() or root.is_symlink():
        raise BuildContractError("The expected extracted source root is missing or unsafe.")
    return root


def _verify_source_files(
    source_root: Path,
    expected: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    actual_paths: set[str] = set()
    identities: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise BuildContractError(f"The source tree contains an unsafe entry: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        actual_paths.add(relative)
        identity = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        if expected.get(relative) != identity:
            raise BuildContractError(f"Source file {relative!r} does not match the contract.")
        identities[relative] = identity
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise BuildContractError(f"Source file set mismatch. Missing={missing}, extra={extra}")
    return identities


def _apply_patches(
    source_root: Path,
    patches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for patch in patches:
        path = source_root / patch["path"]
        before_sha256 = _sha256_file(path)
        if before_sha256 != patch["before_sha256"]:
            raise BuildContractError(f"Patch {patch['id']!r} source hash does not match.")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise BuildContractError(f"Could not read patch target {patch['path']!r}.") from exc
        applied: list[dict[str, Any]] = []
        for replacement in patch["replacements"]:
            count = text.count(replacement["before"])
            if count != replacement["count"]:
                raise BuildContractError(
                    f"Patch {patch['id']!r} expected {replacement['count']} matches, found {count}."
                )
            text = text.replace(
                replacement["before"],
                replacement["after"],
                replacement["count"],
            )
            applied.append(
                {
                    "count": replacement["count"],
                    "before_sha256": _sha256_bytes(replacement["before"].encode("utf-8")),
                    "after_sha256": _sha256_bytes(replacement["after"].encode("utf-8")),
                }
            )
        path.write_bytes(text.encode("utf-8"))
        after_sha256 = _sha256_file(path)
        if after_sha256 != patch["after_sha256"]:
            raise BuildContractError(f"Patch {patch['id']!r} produced an unexpected source hash.")
        receipts.append(
            {
                "id": patch["id"],
                "path": patch["path"],
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "replacements": applied,
            }
        )
    return receipts


def _source_tree_digest(source_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise BuildContractError(f"The patched source tree contains an unsafe entry: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        identity = {
            "path": relative,
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        digest.update(
            json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        )
        file_count += 1
        total_bytes += identity["size"]
    return {
        "algorithm": "sha256-file-manifest-v1",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _distribution_identity(name: str, expected_version: str) -> dict[str, str]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise BuildContractError(f"Required build distribution {name!r} is not installed.") from exc
    if distribution.version != expected_version:
        raise BuildContractError(
            f"Build distribution {name!r} must be {expected_version}, found {distribution.version}."
        )
    metadata_file = next(
        (
            item
            for item in distribution.files or []
            if PurePosixPath(str(item)).parts[-1:] == ("METADATA",)
            and any(part.endswith(".dist-info") for part in PurePosixPath(str(item)).parts)
        ),
        None,
    )
    if metadata_file is None:
        raise BuildContractError(f"Could not locate installed METADATA for {name!r}.")
    metadata_path = Path(distribution.locate_file(metadata_file)).resolve(strict=True)
    return {
        "name": canonicalize_name(name),
        "version": distribution.version,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
    }


def _validate_environment(contract: Mapping[str, Any]) -> dict[str, Any]:
    runtime = contract["runtime"]
    build = contract["build"]
    if sys.prefix == sys.base_prefix:
        raise BuildContractError("Build the derived wheel inside a dedicated virtual environment.")
    if platform.system() != runtime["platform_system"] or platform.machine() != runtime[
        "platform_machine"
    ]:
        raise BuildContractError("The builder requires Linux x86_64.")
    if (sys.version_info.major, sys.version_info.minor) != (3, 12):
        raise BuildContractError("The builder requires CPython 3.12.")
    for key, expected in build["required_host_environment"].items():
        if os.environ.get(key) != expected:
            raise BuildContractError(f"Set {key}={expected!r} before starting the builder.")

    torch = importlib.import_module("torch")
    torch_version = str(getattr(torch, "__version__", ""))
    if torch_version != runtime["torch"]:
        raise BuildContractError(f"Torch must be {runtime['torch']}, found {torch_version!r}.")
    tensor_type = getattr(torch, "Tensor", None)
    attribute = runtime["required_tensor_attribute"]
    if tensor_type is None or not hasattr(tensor_type, attribute):
        raise BuildContractError(
            f"Torch {torch_version} must provide torch.Tensor.{attribute} before backend execution."
        )

    distributions = {
        name: _distribution_identity(name, version)
        for name, version in build["resolved_python_requirements"].items()
    }
    distributions["torch"] = _distribution_identity("torch", TORCH_BUILD_VERSION)

    unshare = shutil.which("unshare")
    if not unshare:
        raise BuildContractError("Required command 'unshare' was not found.")
    resolved_unshare = Path(unshare).resolve(strict=True)
    try:
        version_result = subprocess.run(
            [str(resolved_unshare), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            [str(resolved_unshare), "--net", "--", "true"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildContractError("Linux unshare network isolation preflight failed.") from exc

    python = Path(sys.executable).resolve(strict=True)
    return {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": str(python),
        "python_sha256": _sha256_file(python),
        "virtual_environment": str(Path(sys.prefix).resolve()),
        "torch": torch_version,
        "required_tensor_attribute": attribute,
        "required_tensor_attribute_present": True,
        "installed_distributions": distributions,
        "commands": {
            "unshare": {
                "path": str(resolved_unshare),
                "sha256": _sha256_file(resolved_unshare),
                "version": version_result.stdout.strip(),
            }
        },
    }


def _wheel_member_set(contract: Mapping[str, Any]) -> tuple[str, set[str]]:
    wheel = contract["expected_wheel"]
    distribution = "torch_c_dlpack_ext"
    dist_info = f"{distribution}-{wheel['version']}.dist-info"
    members = set(wheel["package_files"])
    members.update(
        {
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/top_level.txt",
            f"{dist_info}/RECORD",
        }
    )
    members.update(f"{dist_info}/licenses/{name}" for name in wheel["license_files"])
    return dist_info, members


def _validate_zip_path(name: str) -> None:
    if not name or "\\" in name or name.startswith("/"):
        raise BuildContractError(f"Unsafe wheel member path {name!r}.")
    path = PurePosixPath(name.rstrip("/"))
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BuildContractError(f"Unsafe wheel member path {name!r}.")


def _inspect_wheel(contract: Mapping[str, Any], wheel_path: Path) -> dict[str, Any]:
    expected = contract["expected_wheel"]
    if wheel_path.name != expected["filename"]:
        raise BuildContractError(f"Unexpected wheel filename {wheel_path.name!r}.")
    try:
        name, version, build_tag, tags = parse_wheel_filename(wheel_path.name)
    except ValueError as exc:
        raise BuildContractError(f"Could not parse wheel filename {wheel_path.name!r}.") from exc
    if (
        canonicalize_name(name) != PACKAGE_NAME
        or str(version) != expected["version"]
        or build_tag != ()
        or {str(tag) for tag in tags} != {expected["tag"]}
    ):
        raise BuildContractError("The wheel filename identity does not match the contract.")

    dist_info, expected_files = _wheel_member_set(contract)
    expected_timestamp = datetime.fromtimestamp(
        contract["build"]["source_date_epoch"], timezone.utc
    ).timetuple()[:6]
    payloads: dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for info in archive.infolist():
                _validate_zip_path(info.filename)
                if info.filename in seen:
                    raise BuildContractError(f"Duplicate wheel member {info.filename!r}.")
                seen.add(info.filename)
                if info.flag_bits & 0x1:
                    raise BuildContractError(f"Encrypted wheel member {info.filename!r} is forbidden.")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise BuildContractError(f"Wheel symlink {info.filename!r} is forbidden.")
                if info.date_time != expected_timestamp:
                    raise BuildContractError(
                        f"Wheel member {info.filename!r} does not use SOURCE_DATE_EPOCH."
                    )
                if info.is_dir():
                    prefix = info.filename.rstrip("/") + "/"
                    if not any(path.startswith(prefix) for path in expected_files):
                        raise BuildContractError(f"Unexpected wheel directory {info.filename!r}.")
                    continue
                payload = archive.read(info)
                if info.filename in expected["forbidden_top_level_modules"]:
                    raise BuildContractError(
                        f"Generic top-level module {info.filename!r} is forbidden."
                    )
                reason = _native_reason(info.filename, payload[:8])
                if reason:
                    raise BuildContractError(
                        f"Wheel member {info.filename!r} has forbidden {reason}."
                    )
                payloads[info.filename] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise BuildContractError(f"Could not inspect wheel {wheel_path}: {exc}") from exc
    if set(payloads) != expected_files:
        missing = sorted(expected_files - set(payloads))
        extra = sorted(set(payloads) - expected_files)
        raise BuildContractError(f"Wheel payload mismatch. Missing={missing}, extra={extra}")

    for relative, identity in expected["package_files"].items():
        payload = payloads[relative]
        if len(payload) != identity["size"] or _sha256_bytes(payload) != identity["sha256"]:
            raise BuildContractError(f"Wheel package file {relative!r} does not match the contract.")
    for relative, identity in expected["license_files"].items():
        payload = payloads[f"{dist_info}/licenses/{relative}"]
        if len(payload) != identity["size"] or _sha256_bytes(payload) != identity["sha256"]:
            raise BuildContractError(f"Wheel license file {relative!r} does not match the contract.")
    top_level_path = f"{dist_info}/top_level.txt"
    if payloads[top_level_path].decode("utf-8") != expected["top_level"]:
        raise BuildContractError("The wheel exposes an unexpected top-level import namespace.")

    metadata_path = f"{dist_info}/METADATA"
    metadata = BytesParser(policy=email_policy).parsebytes(payloads[metadata_path])
    requires_dist = [str(Requirement(value)) for value in metadata.get_all("Requires-Dist", [])]
    if (
        canonicalize_name(str(metadata.get("Name", ""))) != PACKAGE_NAME
        or metadata.get("Version") != expected["version"]
        or metadata.get("Requires-Python") != expected["requires_python"]
        or requires_dist != expected["requires_dist"]
    ):
        raise BuildContractError("Wheel METADATA does not match the runtime contract.")

    wheel_metadata_path = f"{dist_info}/WHEEL"
    wheel_metadata = BytesParser(policy=email_policy).parsebytes(payloads[wheel_metadata_path])
    if (
        wheel_metadata.get("Wheel-Version") != "1.0"
        or str(wheel_metadata.get("Root-Is-Purelib", "")).lower() != "true"
        or wheel_metadata.get_all("Tag", []) != [expected["tag"]]
    ):
        raise BuildContractError("WHEEL metadata does not describe the required pure wheel.")

    record_path = f"{dist_info}/RECORD"
    rows: dict[str, tuple[str, str]] = {}
    try:
        for row in csv.reader(io.StringIO(payloads[record_path].decode("utf-8"), newline="")):
            if len(row) != 3 or row[0] in rows:
                raise BuildContractError("The wheel RECORD contains malformed or duplicate rows.")
            rows[row[0]] = (row[1], row[2])
    except UnicodeDecodeError as exc:
        raise BuildContractError("The wheel RECORD is not UTF-8.") from exc
    if set(rows) != expected_files:
        raise BuildContractError("The wheel RECORD does not cover the exact payload.")
    for path, payload in payloads.items():
        hash_field, size_field = rows[path]
        if path == record_path:
            if hash_field or size_field:
                raise BuildContractError("The RECORD self-entry must have empty hash and size.")
            continue
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        if hash_field != f"sha256={encoded}" or size_field != str(len(payload)):
            raise BuildContractError(f"RECORD identity mismatch for wheel member {path!r}.")

    members = [
        {"path": path, "size": len(payload), "sha256": _sha256_bytes(payload)}
        for path, payload in sorted(payloads.items())
    ]
    return {
        "path": str(wheel_path.resolve()),
        "filename": wheel_path.name,
        "size": wheel_path.stat().st_size,
        "sha256": _sha256_file(wheel_path),
        "metadata": {
            "name": canonicalize_name(str(metadata["Name"])),
            "version": metadata["Version"],
            "requires_python": metadata["Requires-Python"],
            "requires_dist": requires_dist,
            "tag": expected["tag"],
            "root_is_purelib": True,
        },
        "members": members,
        "native_members": [],
        "record_validated": True,
        "source_date_epoch": contract["build"]["source_date_epoch"],
    }


def _build_environment(contract: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    sanitized = sorted(key for key in _SANITIZED_ENVIRONMENT_KEYS if key in environment)
    for key in _SANITIZED_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    explicit = {
        "CUDA_VISIBLE_DEVICES": "",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "SOURCE_DATE_EPOCH": str(contract["build"]["source_date_epoch"]),
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "UV_OFFLINE": "1",
    }
    environment.update(explicit)
    return environment, sanitized


def _build_wheel(
    contract: Mapping[str, Any],
    source_root: Path,
    wheel_root: Path,
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    wheel_root.mkdir(parents=True, exist_ok=False)
    environment, sanitized = _build_environment(contract)
    frontend = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheel_root.resolve()),
        ".",
    ]
    command = [runtime["commands"]["unshare"]["path"], "--net", "--", *frontend]
    started_at = _utc_now()
    try:
        subprocess.run(command, cwd=source_root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BuildContractError("The offline DLPack source build failed.") from exc
    finished_at = _utc_now()
    wheels = sorted(wheel_root.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildContractError(f"Expected one derived wheel, found {len(wheels)}.")
    explicit_environment = {
        key: environment[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "PIP_CONFIG_FILE",
            "PIP_DISABLE_PIP_VERSION_CHECK",
            "PIP_NO_INDEX",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "PYTHONNOUSERSITE",
            "PYTHONSAFEPATH",
            "SOURCE_DATE_EPOCH",
            "TORCH_DEVICE_BACKEND_AUTOLOAD",
            "UV_OFFLINE",
        )
    }
    return _inspect_wheel(contract, wheels[0]), {
        "command": command,
        "frontend_command": frontend,
        "environment": explicit_environment,
        "sanitized_inherited_environment_keys": sanitized,
        "network_isolation": "linux-unshare-network-namespace",
        "dependency_installation": "disabled by python -m build --no-isolation",
        "started_at": started_at,
        "finished_at": finished_at,
    }


def build_dlpack_wheel(contract_path: Path, output_dir: Path) -> Path:
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise BuildContractError(f"Output directory must not already exist: {output_dir}")
    builder_path = Path(__file__).resolve()
    contract_sha256 = _sha256_file(contract_path)
    builder_sha256 = _sha256_file(builder_path)
    contract = _load_contract(contract_path)
    runtime = _validate_environment(contract)

    output_dir.mkdir(parents=True, exist_ok=False, mode=0o755)
    sdist_root = output_dir / "sdist"
    source_parent = output_dir / "source"
    sdist_root.mkdir()
    sdist_path = sdist_root / contract["sdist"]["filename"]
    downloaded = _download_sdist(contract["sdist"], sdist_path)
    source_root = _safe_extract_sdist(sdist_path, source_parent, contract["sdist"])
    verified_files = _verify_source_files(source_root, contract["verified_files"])
    patches = _apply_patches(source_root, contract["patches"])
    post_patch_tree = _source_tree_digest(source_root)
    if post_patch_tree != contract["expected_post_patch_tree"]:
        raise BuildContractError("The patched source tree does not match the reviewed contract.")

    build_started_at = _utc_now()
    wheel, build_receipt = _build_wheel(contract, source_root, output_dir / "wheel", runtime)
    build_finished_at = _utc_now()
    if _sha256_file(contract_path) != contract_sha256:
        raise BuildContractError("The DLPack build contract changed during the build.")
    if _sha256_file(builder_path) != builder_sha256:
        raise BuildContractError("The DLPack builder changed during the build.")

    receipt_path = output_dir / RECEIPT_NAME
    receipt = {
        "schema": "aisp.torch-dlpack-source-wheel-receipt.v1",
        "contract_sha256": contract_sha256,
        "builder_sha256": builder_sha256,
        "contract": {"path": str(contract_path), "sha256": contract_sha256},
        "builder": {"path": str(builder_path), "sha256": builder_sha256},
        "runtime": runtime,
        "source": {
            "declared_upstream": contract["source"],
            "directory": str(source_root.resolve()),
            "verified_sdist": {
                **downloaded,
                "archive": {
                    key: contract["sdist"][key]
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
        "build": {
            **build_receipt,
            "started_at_total": build_started_at,
            "finished_at_total": build_finished_at,
        },
        "wheel": wheel,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = build_dlpack_wheel(args.contract, args.output_dir)
    except BuildContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
