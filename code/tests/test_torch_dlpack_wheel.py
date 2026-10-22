"""Contract tests for the Torch 2.14-only DLPack compatibility wheel."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import stat
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from scripts import build_torch_dlpack_wheel as dlpack_builder

CODE_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, Any]:
    return json.loads((CODE_ROOT / "torch_dlpack_source_build.json").read_text(encoding="utf-8"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _test_wheel_contract() -> tuple[dict[str, Any], dict[str, bytes]]:
    contract = copy.deepcopy(_contract())
    package_payloads = {
        "torch_c_dlpack_ext/__init__.py": b'"""test package"""\nfrom . import core\n',
        "torch_c_dlpack_ext/core.py": b'"""test core"""\nVALUE = 1\n',
    }
    license_payloads = {"LICENSE": b"test license\n", "NOTICE": b"test notice\n"}
    contract["expected_wheel"]["package_files"] = {
        path: {"size": len(payload), "sha256": _sha256(payload)}
        for path, payload in package_payloads.items()
    }
    contract["expected_wheel"]["license_files"] = {
        path: {"size": len(payload), "sha256": _sha256(payload)}
        for path, payload in license_payloads.items()
    }
    return contract, {**package_payloads, **license_payloads}


def _write_wheel(
    directory: Path,
    *,
    extra_payloads: dict[str, bytes] | None = None,
    timestamp_offset: int = 0,
) -> tuple[dict[str, Any], Path]:
    contract, source_payloads = _test_wheel_contract()
    wheel = contract["expected_wheel"]
    dist_info, _ = dlpack_builder._wheel_member_set(contract)
    payloads = {
        **{path: source_payloads[path] for path in wheel["package_files"]},
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.4\n"
            "Name: torch_c_dlpack_ext\n"
            f"Version: {wheel['version']}\n"
            f"Requires-Python: {wheel['requires_python']}\n"
            f"Requires-Dist: {wheel['requires_dist'][0]}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: dlpack-builder-test\n"
            "Root-Is-Purelib: true\n"
            f"Tag: {wheel['tag']}\n\n"
        ).encode(),
        f"{dist_info}/top_level.txt": wheel["top_level"].encode(),
        **{
            f"{dist_info}/licenses/{path}": source_payloads[path]
            for path in wheel["license_files"]
        },
    }
    payloads.update(extra_payloads or {})
    record_path = f"{dist_info}/RECORD"
    rows = [
        (path, _record_hash(payload), str(len(payload)))
        for path, payload in sorted(payloads.items())
    ]
    rows.append((record_path, "", ""))
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    writer.writerows(rows)
    payloads[record_path] = record_buffer.getvalue().encode()

    path = directory / wheel["filename"]
    epoch = contract["build"]["source_date_epoch"] + timestamp_offset
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return contract, path


def test_contract_binds_source_patch_runtime_and_pure_wheel() -> None:
    contract = _contract()
    dlpack_builder._validate_contract(contract)

    assert contract["sdist"]["sha256"] == (
        "d06f0357d575d22a168cc77acb9020fc4bae30968ceb6718a055dcbe92bacabe"
    )
    assert contract["source"]["declared_upstream_commit"] == (
        "dcd07cfe27465287ee5b203b742e85dcfb99606a"
    )
    assert [patch["id"] for patch in contract["patches"]] == [
        "torch2141-pure-wheel-metadata",
        "torch2141-no-legacy-fallback",
    ]
    assert contract["patches"][0]["after_sha256"] == (
        "b6d9e3aef40f0fd4ede62e9cb9d4b77fd510ac39141c9f0591c1cdafd5b212b8"
    )
    assert contract["patches"][1]["after_sha256"] == (
        "1d65fe82da1f6254400a6d9897cbb066828113b641833500afb292777ca5c9d4"
    )
    assert contract["expected_wheel"]["version"] == (
        "0.1.5+torch2141.nobackend1"
    )
    assert contract["expected_wheel"]["requires_dist"] == ["torch==2.14.1"]
    assert contract["build"]["resolved_python_requirements"] == {
        "apache-tvm-ffi": "0.1.11",
        "build": "1.6.1",
        "packaging": "26.2",
        "setuptools": "80.10.2",
        "wheel": "0.48.0",
    }
    assert Version("2.14.1+cu130") in Requirement("torch==2.14.1").specifier
    assert contract["expected_wheel"]["tag"] == "py3-none-any"
    assert contract["expected_wheel"]["forbidden_top_level_modules"] == [
        "build_backend.py",
        "build_utils.py",
    ]
    core_patch = contract["patches"][1]["replacements"][0]["after"]
    assert "hasattr(torch.Tensor, \"__dlpack_c_exchange_api__\")" in core_patch
    assert "__c_dlpack_exchange_api__" not in core_patch
    assert "Legacy native fallbacks" in core_patch


def test_patch_application_is_hash_bound_and_rejects_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "pyproject.toml"
    original = b'name = "upstream"\npy-modules = ["build_backend"]\n'
    expected = b'name = "derived"\npy-modules = []\n'
    target.write_bytes(original)
    patch = {
        "id": "test-patch",
        "path": "pyproject.toml",
        "before_sha256": _sha256(original),
        "after_sha256": _sha256(expected),
        "replacements": [
            {"count": 1, "before": 'name = "upstream"', "after": 'name = "derived"'},
            {
                "count": 1,
                "before": 'py-modules = ["build_backend"]',
                "after": "py-modules = []",
            },
        ],
    }

    receipt = dlpack_builder._apply_patches(source, [patch])
    assert target.read_bytes() == expected
    assert receipt[0]["before_sha256"] == _sha256(original)
    assert receipt[0]["after_sha256"] == _sha256(expected)

    target.write_bytes(original + b"# changed\n")
    with pytest.raises(dlpack_builder.BuildContractError, match="source hash"):
        dlpack_builder._apply_patches(source, [patch])


def test_safe_extract_rejects_native_members_and_links(tmp_path: Path) -> None:
    contract = {
        "root": "source-1.0",
        "member_count": 2,
        "file_count": 1,
        "directory_count": 1,
        "expanded_file_bytes": 8,
    }
    archive_path = tmp_path / "native.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo("source-1.0")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        native = tarfile.TarInfo("source-1.0/renamed.bin")
        native.size = 8
        archive.addfile(native, io.BytesIO(b"\x7fELFdata"))
    with pytest.raises(dlpack_builder.BuildContractError, match="ELF file magic"):
        dlpack_builder._safe_extract_sdist(archive_path, tmp_path / "native-out", contract)

    link_archive = tmp_path / "link.tar.gz"
    with tarfile.open(link_archive, "w:gz") as archive:
        root = tarfile.TarInfo("source-1.0")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        link = tarfile.TarInfo("source-1.0/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        archive.addfile(link)
    link_contract = {**contract, "file_count": 0, "expanded_file_bytes": 1}
    with pytest.raises(dlpack_builder.BuildContractError, match="not a regular file"):
        dlpack_builder._safe_extract_sdist(
            link_archive,
            tmp_path / "link-out",
            link_contract,
        )


def test_wheel_inspection_accepts_exact_deterministic_payload(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_contract, first = _write_wheel(first_dir)
    second_contract, second = _write_wheel(second_dir)

    first_receipt = dlpack_builder._inspect_wheel(first_contract, first)
    second_receipt = dlpack_builder._inspect_wheel(second_contract, second)
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["sha256"] == second_receipt["sha256"]
    assert first_receipt["members"] == second_receipt["members"]
    assert first_receipt["metadata"]["requires_dist"] == ["torch==2.14.1"]
    assert first_receipt["native_members"] == []
    assert first_receipt["record_validated"] is True


@pytest.mark.parametrize(
    ("extra_payloads", "message"),
    [
        ({"build_backend.py": b"development backend\n"}, "Generic top-level module"),
        ({"build_utils.py": b"development helper\n"}, "Generic top-level module"),
        ({"torch_c_dlpack_ext/native.so": b"\x7fELFpayload"}, "shared object suffix"),
        ({"torch_c_dlpack_ext/renamed.bin": b"\x7fELFpayload"}, "ELF file magic"),
    ],
)
def test_wheel_inspection_rejects_backend_and_native_payloads(
    tmp_path: Path,
    extra_payloads: dict[str, bytes],
    message: str,
) -> None:
    contract, wheel = _write_wheel(tmp_path, extra_payloads=extra_payloads)
    with pytest.raises(dlpack_builder.BuildContractError, match=message):
        dlpack_builder._inspect_wheel(contract, wheel)


def test_wheel_inspection_rejects_nonreproducible_timestamp(tmp_path: Path) -> None:
    contract, wheel = _write_wheel(tmp_path, timestamp_offset=2)
    with pytest.raises(dlpack_builder.BuildContractError, match="SOURCE_DATE_EPOCH"):
        dlpack_builder._inspect_wheel(contract, wheel)


def test_environment_requires_builtin_api_before_any_build_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()

    class Tensor:
        pass

    torch = SimpleNamespace(__version__="2.14.1+cu130", Tensor=Tensor)
    monkeypatch.setattr(sys, "prefix", "/tmp/dlpack-venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(dlpack_builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dlpack_builder.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(dlpack_builder.importlib, "import_module", lambda name: torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    with pytest.raises(dlpack_builder.BuildContractError, match="before backend execution"):
        dlpack_builder._validate_environment(contract)


def test_environment_preflight_never_reads_torch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()

    class Tensor:
        __dlpack_c_exchange_api__ = object()

    class TorchModule:
        __version__ = "2.14.1+cu130"

        def __getattr__(self, name: str) -> Any:
            if name == "cuda":
                raise AssertionError("The builder must not access torch.cuda")
            raise AttributeError(name)

    TorchModule.Tensor = Tensor

    monkeypatch.setattr(sys, "prefix", "/tmp/dlpack-venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(dlpack_builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dlpack_builder.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(dlpack_builder.importlib, "import_module", lambda name: TorchModule())
    monkeypatch.setattr(
        dlpack_builder,
        "_distribution_identity",
        lambda name, version: {"name": name, "version": version},
    )
    monkeypatch.setattr(dlpack_builder.shutil, "which", lambda command: sys.executable)
    monkeypatch.setattr(dlpack_builder, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(stdout="unshare from util-linux 2.41\n")

    monkeypatch.setattr(dlpack_builder.subprocess, "run", fake_run)
    runtime = dlpack_builder._validate_environment(contract)

    assert runtime["required_tensor_attribute_present"] is True
    assert commands[1][-3:] == ["--net", "--", "true"]
    assert all("cuda" not in token.lower() for command in commands for token in command)


def test_build_environment_is_offline_reproducible_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.invalid/simple")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("PYTHONPATH", "/tmp/inherited")
    environment, sanitized = dlpack_builder._build_environment(_contract())

    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert environment["SOURCE_DATE_EPOCH"] == "1768217108"
    assert "PIP_INDEX_URL" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "PYTHONPATH" not in environment
    assert {"PIP_INDEX_URL", "HTTPS_PROXY", "PYTHONPATH"} <= set(sanitized)


def test_existing_output_fails_before_contract_or_environment_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        dlpack_builder,
        "_load_contract",
        lambda path: pytest.fail("contract must not be loaded for an existing output"),
    )
    with pytest.raises(dlpack_builder.BuildContractError, match="must not already exist"):
        dlpack_builder.build_dlpack_wheel(CODE_ROOT / "torch_dlpack_source_build.json", output)
