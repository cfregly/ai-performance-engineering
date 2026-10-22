"""Offline controls for the pinned native Torch extension source builds."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import struct
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts import build_torch_extensions as native_builder

CODE_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, Any]:
    return json.loads((CODE_ROOT / "native_torch_extensions.json").read_text(encoding="utf-8"))


def _packages_by_id() -> dict[str, dict[str, Any]]:
    return {package["id"]: package for package in _contract()["packages"]}


def _exact_pins(path: Path, *, locked: bool = False) -> dict[str, str]:
    pins: dict[str, str] = {}
    suffix = r"\s*\\?" if locked else ""
    pattern = re.compile(rf"(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^\s\\]+){suffix}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line if locked else raw_line.split("#", 1)[0].strip()
        match = pattern.fullmatch(line)
        if match is None:
            continue
        name = match.group("name").lower().replace("_", "-")
        pins[name] = match.group("version")
    return pins


def _write_fake_native_sdk_distributions(
    venv_root: Path, native_wheel_sdks: dict[str, Any]
) -> dict[str, Any]:
    site_packages = venv_root / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    distributions: dict[str, Any] = {}
    for distribution_name, sdk in native_wheel_sdks.items():
        records: list[str] = []
        for kind in ("headers", "libraries"):
            for relative in sdk["assets"][kind]:
                path = site_packages / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"{distribution_name}:{relative}\n".encode()
                path.write_bytes(payload)
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
                records.append(f"{relative},sha256={digest.decode()},{len(payload)}")
        dist_info_name = distribution_name.replace("-", "_") + f"-{sdk['version']}.dist-info"
        dist_info = site_packages / dist_info_name
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\n" f"Name: {distribution_name}\n" f"Version: {sdk['version']}\n",
            encoding="utf-8",
        )
        records.append(f"{dist_info_name}/METADATA,,")
        records.append(f"{dist_info_name}/RECORD,,")
        (dist_info / "RECORD").write_text("\n".join(records) + "\n", encoding="utf-8")
        distributions[distribution_name] = native_builder.importlib_metadata.Distribution.at(
            dist_info
        )
    return distributions


def _write_wheel(
    directory: Path,
    package: dict[str, Any],
    *,
    native_prefix: str | None = None,
    version: str | None = None,
    valid_elf: bool = True,
) -> Path:
    expected = package["expected_metadata"]
    distribution = expected["name"].replace("-", "_")
    wheel_path = directory / (f"{distribution}-{expected['version']}-{expected['wheel_tag']}.whl")
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {expected['name']}",
        f"Version: {version or expected['version']}",
        f"Requires-Python: {expected['requires_python']}",
        *(f"Requires-Dist: {requirement}" for requirement in expected["requires_dist"]),
        "",
        "",
    ]
    wheel = [
        "Wheel-Version: 1.0",
        "Generator: native-builder-test",
        "Root-Is-Purelib: false",
        f"Tag: {expected['wheel_tag']}",
        "",
        "",
    ]
    payload = bytearray(expected["minimum_native_member_bytes"])
    if valid_elf:
        ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
        struct.pack_into(
            "<16sHHIQQQIHHHHHH",
            payload,
            0,
            ident,
            3,
            62,
            1,
            0,
            64,
            0,
            0,
            64,
            56,
            1,
            0,
            0,
            0,
        )
        struct.pack_into(
            "<IIQQQQQQ",
            payload,
            64,
            1,
            0x5,
            120,
            0,
            0,
            len(payload) - 120,
            len(payload) - 120,
            0x1000,
        )
    else:
        payload[:7] = b"\x7fELF\x02\x01\x01"
        payload[18:20] = (62).to_bytes(2, "little")
    prefix = native_prefix or expected["required_native_member_prefixes"][0]
    native_member = f"{prefix}.cpython-312-x86_64-linux-gnu.so"
    dist_info = f"{distribution}-{expected['version']}.dist-info"
    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        archive.writestr(f"{dist_info}/WHEEL", "\n".join(wheel))
        archive.writestr(native_member, payload)
    return wheel_path


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for info, payload in members:
            archive.addfile(info, io.BytesIO(payload) if info.isfile() else None)


def test_contract_pins_sources_patch_and_disjoint_fa_namespaces() -> None:
    contract = _contract()
    native_builder._validate_contract(contract)
    packages = {package["id"]: package for package in contract["packages"]}
    te = packages["transformer-engine-torch"]
    fa = packages["flash-attn"]

    assert contract["build"]["frontend"] == [
        "python",
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
    ]
    assert contract["build"]["cuda_arch_list"] == "10.0"
    assert contract["build"]["cuda_sm"] == "100"
    assert contract["build"]["maximum_jobs"] == 8
    assert contract["build"]["maximum_nvcc_threads"] == 2
    assert contract["build"]["active_lock"] == "requirements_torch2141_linux_x86_64.lock"
    assert "unshare" in contract["build"]["required_commands"]
    assert contract["build"]["native_wheel_sdks"] == {
        "nvidia-cudnn-cu13": {
            "version": "9.24.0.43",
            "root": "nvidia/cudnn",
            "include_directory": "nvidia/cudnn/include",
            "library_directory": "nvidia/cudnn/lib",
            "assets": {
                "headers": ["nvidia/cudnn/include/cudnn.h"],
                "libraries": ["nvidia/cudnn/lib/libcudnn.so.9"],
            },
        },
        "nvidia-nccl-cu13": {
            "version": "2.30.7",
            "root": "nvidia/nccl",
            "include_directory": "nvidia/nccl/include",
            "library_directory": "nvidia/nccl/lib",
            "assets": {
                "headers": ["nvidia/nccl/include/nccl.h"],
                "libraries": ["nvidia/nccl/lib/libnccl.so.2"],
            },
        },
        "nvidia-nvshmem-cu13": {
            "version": "3.4.5",
            "root": "nvidia/nvshmem",
            "include_directory": "nvidia/nvshmem/include",
            "library_directory": "nvidia/nvshmem/lib",
            "assets": {
                "headers": ["nvidia/nvshmem/include/nvshmem.h"],
                "libraries": ["nvidia/nvshmem/lib/libnvshmem_host.so.3"],
            },
        },
    }
    assert te["source"]["declared_upstream_commit"] == ("6ea2a74a9e98c99e6d7b164a33775cc457520027")
    assert te["sdist"]["sha256"] == (
        "4c40ced8a6261649e689076f198d4feaf02277b1ba18931a66eccdaceb33c43e"
    )
    assert te["preinstalled_distributions"] == {
        "transformer-engine": "2.20.2",
        "transformer-engine-cu13": "2.20.2",
    }
    assert te["forbidden_distributions"] == ["transformer-engine-torch"]
    assert te["build_environment"]["NVTE_BUILD_USE_NVIDIA_WHEELS"] == "0"
    assert fa["source"]["declared_upstream_commit"] == ("49774b31c8d1172a5b28f72e9aed0f322bf4ccba")
    assert fa["sdist"]["sha256"] == (
        "d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b"
    )
    assert fa["forbidden_distributions"] == ["flash-attn", "flash-attn-4"]
    assert fa["build_environment"]["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"
    assert fa["build_environment"]["FLASH_ATTN_CUDA_ARCHS"] == "100"
    patch = fa["patches"][0]
    assert patch["id"] == "torch-2.13-plus-cxx20"
    assert patch["before_sha256"] == fa["verified_files"]["setup.py"]["sha256"]
    assert patch["upstream_fix_commit"].endswith("edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d")
    assert fa["source"]["vendored_sources"]["cutlass"]["declared_upstream_commit"] == (
        "dc4817921edda44a549197ff3a9dcf5df0636e7b"
    )


def test_package_contract_cannot_override_verified_native_sdk_search_paths() -> None:
    contract = json.loads(json.dumps(_contract()))
    contract["packages"][0]["build_environment"]["CPATH"] = "/unverified/include"

    with pytest.raises(native_builder.BuildContractError, match="may not override"):
        native_builder._validate_contract(contract)


def test_native_builder_requirements_match_active_source_and_linux_lock() -> None:
    contract = _contract()
    source = _exact_pins(CODE_ROOT / "requirements_latest.txt")
    lock = _exact_pins(CODE_ROOT / "requirements_torch2141_linux_x86_64.lock", locked=True)
    te = _packages_by_id()["transformer-engine-torch"]
    required = {
        **contract["build"]["resolved_python_requirements"],
        **te["preinstalled_distributions"],
    }

    assert {name: source.get(name) for name in required} == required
    assert {name: lock.get(name) for name in required} == required
    assert source["flash-attn-4"] == "4.0.0b36"
    assert lock["flash-attn-4"] == "4.0.0b36"
    assert "flash-attn" not in lock
    assert "transformer-engine-torch" not in lock

    native_sdk_versions = {
        name: sdk["version"] for name, sdk in contract["build"]["native_wheel_sdks"].items()
    }
    assert {name: lock.get(name) for name in native_sdk_versions} == native_sdk_versions
    assert (
        native_builder._validate_native_sdk_lock(
            CODE_ROOT / contract["build"]["active_lock"],
            contract["build"]["native_wheel_sdks"],
        )
        == native_sdk_versions
    )


def test_fa2_selection_rejects_installed_fa4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    fa = _packages_by_id()["flash-attn"]
    monkeypatch.setattr(sys, "prefix", "/tmp/native-venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(native_builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(native_builder.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "version_info", SimpleNamespace(major=3, minor=12))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.14.1+cu130",
            version=SimpleNamespace(cuda="13.0"),
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setattr(
        native_builder,
        "_distribution_version",
        lambda name: contract["build"]["resolved_python_requirements"][name],
    )

    def fake_installed_version(name: str) -> str:
        if name == "flash-attn-4":
            return "4.0.0b36"
        raise native_builder.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(native_builder.importlib_metadata, "version", fake_installed_version)

    with pytest.raises(native_builder.BuildContractError, match="flash-attn-4==4.0.0b36"):
        native_builder._validate_environment(
            contract,
            [fa],
            cuda_arch_list="10.0",
            max_jobs=8,
            nvcc_threads=2,
        )


def test_safe_extract_accepts_regular_members_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    root = tarfile.TarInfo("source-1.0")
    root.type = tarfile.DIRTYPE
    nested = tarfile.TarInfo("source-1.0/src")
    nested.type = tarfile.DIRTYPE
    source = tarfile.TarInfo("source-1.0/src/kernel.cpp")
    payload = b"int kernel() { return 1; }\n"
    source.size = len(payload)
    archive = tmp_path / "source.tar.gz"
    _write_tar(archive, [(root, b""), (nested, b""), (source, payload)])
    sdist = {
        "root": "source-1.0",
        "member_count": 3,
        "file_count": 1,
        "directory_count": 2,
        "expanded_file_bytes": len(payload),
    }

    extracted = native_builder._safe_extract_sdist(archive, tmp_path / "extracted", sdist)

    assert (extracted / "src/kernel.cpp").read_bytes() == payload
    traversal = tarfile.TarInfo("source-1.0/../escape")
    traversal_payload = b"escape"
    traversal.size = len(traversal_payload)
    bad_archive = tmp_path / "bad.tar.gz"
    _write_tar(bad_archive, [(traversal, traversal_payload)])
    bad_sdist = {
        "root": "source-1.0",
        "member_count": 1,
        "file_count": 1,
        "directory_count": 0,
        "expanded_file_bytes": len(traversal_payload),
    }
    with pytest.raises(native_builder.BuildContractError, match="Unsafe tar member"):
        native_builder._safe_extract_sdist(bad_archive, tmp_path / "bad-extracted", bad_sdist)
    assert not (tmp_path / "escape").exists()


def test_safe_extract_rejects_links(tmp_path: Path) -> None:
    link = tarfile.TarInfo("source-1.0/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/tmp/escape"
    archive = tmp_path / "link.tar.gz"
    _write_tar(archive, [(link, b"")])
    sdist = {
        "root": "source-1.0",
        "member_count": 1,
        "file_count": 0,
        "directory_count": 0,
        "expanded_file_bytes": 0,
    }

    with pytest.raises(native_builder.BuildContractError, match="regular file or directory"):
        native_builder._safe_extract_sdist(archive, tmp_path / "out", sdist)


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("source-1.0/prebuilt.so.1", b"not-even-elf"),
        ("source-1.0/renamed.dat", b"\x7fELF" + b"\0" * 60),
    ],
)
def test_safe_extract_rejects_prebuilt_or_renamed_native_artifacts(
    tmp_path: Path, filename: str, payload: bytes
) -> None:
    member = tarfile.TarInfo(filename)
    member.size = len(payload)
    archive = tmp_path / "native.tar.gz"
    _write_tar(archive, [(member, payload)])
    sdist = {
        "root": "source-1.0",
        "member_count": 1,
        "file_count": 1,
        "directory_count": 0,
        "expanded_file_bytes": len(payload),
    }

    with pytest.raises(native_builder.BuildContractError, match="prebuilt artifact"):
        native_builder._safe_extract_sdist(archive, tmp_path / "native-out", sdist)


def test_cuda_toolkit_is_derived_from_nvcc_and_rejects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvcc = tmp_path / "cuda-13.0" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_bytes(b"pinned-nvcc")
    for name in native_builder._CUDA_ROOT_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, str(nvcc.parents[1]))
    for name in native_builder._CUDA_COMPILER_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, str(nvcc))

    toolkit = native_builder._resolve_cuda_toolkit(str(nvcc), "13.0")

    assert toolkit["root"] == str(nvcc.parents[1].resolve())
    assert toolkit["nvcc"] == str(nvcc.resolve())
    assert toolkit["nvcc_sha256"] == hashlib.sha256(b"pinned-nvcc").hexdigest()
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "other-cuda"))
    with pytest.raises(native_builder.BuildContractError, match="conflicts"):
        native_builder._resolve_cuda_toolkit(str(nvcc), "13.0")


def test_native_wheel_sdks_are_record_verified_inside_active_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_wheel_sdks = _contract()["build"]["native_wheel_sdks"]
    venv_root = tmp_path / "venv"
    distributions = _write_fake_native_sdk_distributions(venv_root, native_wheel_sdks)
    monkeypatch.setattr(sys, "prefix", str(venv_root))
    monkeypatch.setattr(
        native_builder.importlib_metadata,
        "distribution",
        lambda name: distributions[name],
    )

    validated = native_builder._validate_native_wheel_sdks(native_wheel_sdks)

    assert set(validated) == set(native_wheel_sdks)
    for distribution_name, sdk in validated.items():
        assert sdk["version"] == native_wheel_sdks[distribution_name]["version"]
        assert Path(sdk["distribution_root"]).is_relative_to(venv_root)
        assert Path(sdk["include_directory"]).is_relative_to(venv_root)
        assert Path(sdk["library_directory"]).is_relative_to(venv_root)
        for asset in sdk["assets"]:
            assert asset["sha256"] == asset["record_sha256"]
            assert Path(asset["path"]).is_relative_to(venv_root)

    other_venv = tmp_path / "other-venv"
    other_venv.mkdir()
    monkeypatch.setattr(sys, "prefix", str(other_venv))
    with pytest.raises(native_builder.BuildContractError, match="outside the active virtual"):
        native_builder._validate_native_wheel_sdks(native_wheel_sdks)

    monkeypatch.setattr(sys, "prefix", str(venv_root))
    nccl_header = venv_root / "lib/python3.12/site-packages/nvidia/nccl/include/nccl.h"
    nccl_header.write_bytes(b"tampered\n")
    with pytest.raises(native_builder.BuildContractError, match="RECORD (size|hash)"):
        native_builder._validate_native_wheel_sdks(native_wheel_sdks)


def test_native_wheel_sdk_mutation_is_detected_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_wheel_sdks = _contract()["build"]["native_wheel_sdks"]
    venv_root = tmp_path / "venv"
    distributions = _write_fake_native_sdk_distributions(venv_root, native_wheel_sdks)
    monkeypatch.setattr(sys, "prefix", str(venv_root))
    monkeypatch.setattr(
        native_builder.importlib_metadata,
        "distribution",
        lambda name: distributions[name],
    )
    validated = native_builder._validate_native_wheel_sdks(native_wheel_sdks)
    asset = validated["nvidia-nccl-cu13"]["assets"][0]
    Path(asset["path"]).write_bytes(b"changed after preflight\n")

    with pytest.raises(native_builder.BuildContractError, match="changed during the build"):
        native_builder._assert_native_wheel_sdks_unchanged(validated)


def test_native_wheel_sdk_lock_requires_exact_versions(tmp_path: Path) -> None:
    native_wheel_sdks = _contract()["build"]["native_wheel_sdks"]
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "nvidia-cudnn-cu13==9.24.0.43 \\\n"
        "nvidia-nccl-cu13==2.30.6 \\\n"
        "nvidia-nvshmem-cu13==3.4.5 \\\n",
        encoding="utf-8",
    )

    with pytest.raises(native_builder.BuildContractError, match="nvidia-nccl-cu13==2.30.7"):
        native_builder._validate_native_sdk_lock(lock, native_wheel_sdks)


def test_post_patch_source_tree_digest_changes_with_content(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "kernel.cpp"
    source.write_text("int kernel() { return 1; }\n", encoding="utf-8")

    before = native_builder._source_tree_digest(source_root)
    source.write_text("int kernel() { return 2; }\n", encoding="utf-8")
    after = native_builder._source_tree_digest(source_root)

    assert before["file_count"] == after["file_count"] == 1
    assert before["sha256"] != after["sha256"]


def test_patch_application_is_hash_bound(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    target = source_root / "flags.txt"
    before = b"standard=c++17\n"
    after = b"standard=c++20\n"
    target.write_bytes(before)
    patch = {
        "id": "cxx20",
        "path": "flags.txt",
        "reason": "test",
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "replacements": [{"count": 1, "before": "standard=c++17", "after": "standard=c++20"}],
    }

    receipt = native_builder._apply_patches(source_root, [patch])

    assert target.read_bytes() == after
    assert receipt[0]["after_sha256"] == patch["after_sha256"]
    target.write_bytes(before + b"# drift\n")
    with pytest.raises(native_builder.BuildContractError, match="input hash"):
        native_builder._apply_patches(source_root, [patch])


@pytest.mark.parametrize("package_id", ["transformer-engine-torch", "flash-attn"])
def test_wheel_inspection_requires_exact_metadata_and_native_elf(
    tmp_path: Path, package_id: str
) -> None:
    package = _packages_by_id()[package_id]
    wheel = _write_wheel(tmp_path, package)

    inspected = native_builder._inspect_wheel(package, wheel)

    assert inspected["metadata"]["version"] == package["expected_metadata"]["version"]
    assert inspected["native_members"][0]["elf_machine"] == "x86_64"
    assert inspected["native_members"][0]["elf_type"] == "ET_DYN"
    assert inspected["native_members"][0]["executable_load_segments"] == 1
    assert inspected["native_members"][0]["sha256"]


def test_wheel_inspection_rejects_near_prefix_native_member(tmp_path: Path) -> None:
    package = _packages_by_id()["flash-attn"]
    wheel = _write_wheel(tmp_path, package, native_prefix="flash_attn_2_cuda_evil")

    with pytest.raises(native_builder.BuildContractError, match="exactly one native member"):
        native_builder._inspect_wheel(package, wheel)


def test_wheel_inspection_rejects_magic_only_zero_blob(tmp_path: Path) -> None:
    package = _packages_by_id()["flash-attn"]
    wheel = _write_wheel(tmp_path, package, valid_elf=False)

    with pytest.raises(native_builder.BuildContractError, match="ET_DYN ELF"):
        native_builder._inspect_wheel(package, wheel)


def test_build_command_uses_pinned_frontend_and_force_source_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    package = _packages_by_id()["transformer-engine-torch"]
    source_root = tmp_path / "source"
    source_root.mkdir()
    runtime = {
        "commands": {
            "gcc": "/usr/bin/gcc",
            "g++": "/usr/bin/g++",
            "unshare": "/usr/bin/unshare",
        },
        "cuda_toolkit": {
            "root": "/opt/cuda-13.0",
            "nvcc": "/opt/cuda-13.0/bin/nvcc",
        },
        "native_wheel_sdks": {
            "nvidia-cudnn-cu13": {
                "include_directory": "/venv/nvidia/cudnn/include",
                "library_directory": "/venv/nvidia/cudnn/lib",
            },
            "nvidia-nccl-cu13": {
                "include_directory": "/venv/nvidia/nccl/include",
                "library_directory": "/venv/nvidia/nccl/lib",
            },
            "nvidia-nvshmem-cu13": {
                "include_directory": "/venv/nvidia/nvshmem/include",
                "library_directory": "/venv/nvidia/nvshmem/lib",
            },
        },
    }
    observed: dict[str, Any] = {}
    monkeypatch.setenv("CPATH", "/untrusted/include")
    monkeypatch.setenv("CPLUS_INCLUDE_PATH", "/untrusted/cxx-include")
    monkeypatch.setenv("CMAKE_INCLUDE_PATH", "/untrusted/cmake-include")
    monkeypatch.setenv("CMAKE_LIBRARY_PATH", "/untrusted/cmake-lib")
    monkeypatch.setenv("LIBRARY_PATH", "/untrusted/lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted/lib")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")

    def fake_run(command: list[str], **kwargs: Any) -> None:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        destination = Path(command[command.index("--outdir") + 1])
        (destination / "built.whl").write_bytes(b"wheel")

    monkeypatch.setattr(native_builder.subprocess, "run", fake_run)
    monkeypatch.setattr(
        native_builder,
        "_inspect_wheel",
        lambda _package, wheel: {"path": str(wheel)},
    )

    native_builder._build_package(
        contract,
        package,
        source_root,
        tmp_path / "wheel",
        runtime,
        max_jobs=4,
        nvcc_threads=1,
    )

    assert observed["command"][:4] == [
        "/usr/bin/unshare",
        "--net",
        "--",
        sys.executable,
    ]
    assert observed["command"][4:9] == [
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
    ]
    assert "pip" not in observed["command"]
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["environment"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert observed["environment"]["MAX_JOBS"] == "4"
    assert observed["environment"]["NVCC_THREADS"] == "1"
    assert observed["environment"]["CUDA_HOME"] == "/opt/cuda-13.0"
    assert observed["environment"]["CUDA_PATH"] == "/opt/cuda-13.0"
    assert observed["environment"]["TORCH_CUDA_HOME"] == "/opt/cuda-13.0"
    assert observed["environment"]["CUDACXX"] == "/opt/cuda-13.0/bin/nvcc"
    assert observed["environment"]["PIP_NO_INDEX"] == "1"
    include_search_path = ":".join(
        [
            "/venv/nvidia/cudnn/include",
            "/venv/nvidia/nccl/include",
            "/venv/nvidia/nvshmem/include",
        ]
    )
    library_search_path = ":".join(
        [
            "/venv/nvidia/cudnn/lib",
            "/venv/nvidia/nccl/lib",
            "/venv/nvidia/nvshmem/lib",
        ]
    )
    assert observed["environment"]["CPATH"] == include_search_path
    assert observed["environment"]["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert observed["environment"]["CPLUS_INCLUDE_PATH"] == include_search_path
    assert observed["environment"]["CMAKE_INCLUDE_PATH"] == include_search_path
    assert observed["environment"]["LIBRARY_PATH"] == library_search_path
    assert observed["environment"]["CMAKE_LIBRARY_PATH"] == library_search_path
    assert "/untrusted/" not in observed["environment"]["CPATH"]
    assert "/untrusted/" not in observed["environment"]["LIBRARY_PATH"]
    assert "LD_LIBRARY_PATH" not in observed["environment"]
    assert "HTTPS_PROXY" not in observed["environment"]
    assert observed["environment"]["NVTE_PYTORCH_FORCE_BUILD"] == "TRUE"
    assert observed["environment"]["NVTE_BUILD_USE_NVIDIA_WHEELS"] == "0"
    assert observed["environment"]["NVTE_CUDA_ARCHS"] == "100"


def test_existing_output_directory_fails_before_contract_or_environment_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        native_builder,
        "_load_contract",
        lambda _path: pytest.fail("contract loading must not run"),
    )

    with pytest.raises(native_builder.BuildContractError, match="must not already exist"):
        native_builder.build_extensions(
            CODE_ROOT / "native_torch_extensions.json",
            output,
            package_ids=["transformer-engine-torch"],
            cuda_arch_list="10.0",
            max_jobs=8,
            nvcc_threads=2,
        )


def test_cli_requires_explicit_package_selection() -> None:
    args = native_builder._parse_args(
        [
            "--output-dir",
            "/tmp/native-build",
            "--package",
            "transformer-engine-torch",
            "--cuda-arch-list",
            "10.0",
            "--max-jobs",
            "8",
            "--nvcc-threads",
            "2",
        ]
    )

    assert args.packages == ["transformer-engine-torch"]
