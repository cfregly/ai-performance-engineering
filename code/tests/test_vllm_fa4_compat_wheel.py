"""Offline controls for the guarded vLLM and FlashAttention 4 wheel backport."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from scripts import build_vllm_fa4_compat_wheel as backport

DIST_INFO = "vllm-0.16.0+cu130.dist-info"


def _record_digest(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 1, 2, 3, 4, 6))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_test_wheel(
    directory: Path,
    *,
    common: bytes,
    version: str = "0.16.0+cu130",
) -> Path:
    directory.mkdir(parents=True)
    wheel = directory / backport.CONTRACT.wheel_filename
    members = {
        backport.CONTRACT.common_path: common,
        "vllm/_C.abi3.so": b"test-bundled-vllm-extension",
        f"{DIST_INFO}/METADATA": (
            f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n\n"
        ).encode(),
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
    }
    rows = [[name, _record_digest(payload), str(len(payload))] for name, payload in members.items()]
    rows.append([f"{DIST_INFO}/RECORD", "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    members[f"{DIST_INFO}/RECORD"] = record.getvalue().encode()
    with ZipFile(wheel, "w") as archive:
        for name, payload in members.items():
            archive.writestr(_zip_info(name), payload)
    return wheel


def _test_contract(wheel: Path, common: bytes):
    patched = common.replace(backport.OLD_IMPORT, backport.NEW_IMPORT).replace(
        backport.OLD_INIT, backport.NEW_INIT
    )
    return replace(
        backport.CONTRACT,
        wheel_sha256=backport.sha256_file(wheel),
        common_before_sha256=backport.sha256_bytes(common),
        common_after_sha256=backport.sha256_bytes(patched),
    )


@pytest.fixture
def common_source() -> bytes:
    return (
        b"# SPDX-License-Identifier: Apache-2.0\n"
        + backport.OLD_IMPORT
        + b"\nclass ApplyRotaryEmb:\n    def __init__(self):\n"
        + backport.OLD_INIT
    )


def test_backport_builds_deterministic_installable_wheel_and_manifest(
    tmp_path: Path,
    common_source: bytes,
) -> None:
    input_wheel = _write_test_wheel(tmp_path / "input", common=common_source)
    contract = _test_contract(input_wheel, common_source)

    first_wheel, first_manifest = backport.build_backport(
        input_wheel, tmp_path / "first", contract=contract
    )
    second_wheel, _ = backport.build_backport(input_wheel, tmp_path / "second", contract=contract)

    assert backport.sha256_file(first_wheel) == backport.sha256_file(second_wheel)
    assert first_manifest["input"]["sha256"] == contract.wheel_sha256
    assert first_manifest["output"]["sha256"] == backport.sha256_file(first_wheel)
    assert first_manifest["pin"]["requirement"] == "vllm==0.16.0+cu130"
    assert first_manifest["requirements"]["selected_versions"] == {
        "flashinfer-python": "0.6.3",
        "torch": "2.9.1+cu130",
    }
    assert first_manifest["upstream"]["merge_commit"] == backport.UPSTREAM["merge_commit"]
    assert "vllm_no_deps.pin" in first_manifest["retirement"]
    assert first_manifest["changed_members"] == [
        f"{DIST_INFO}/RECORD",
        contract.common_path,
    ]

    with ZipFile(input_wheel) as original, ZipFile(first_wheel) as patched:
        patched_common = patched.read(contract.common_path)
        assert backport.OLD_IMPORT not in patched_common
        assert backport.OLD_INIT not in patched_common
        assert backport.NEW_IMPORT in patched_common
        assert backport.NEW_INIT in patched_common
        assert patched.read("vllm/_C.abi3.so") == original.read("vllm/_C.abi3.so")
        backport._validate_record(
            patched.read(f"{DIST_INFO}/RECORD"), contract.common_path, patched_common
        )


def test_backport_rejects_wheel_source_and_metadata_drift(
    tmp_path: Path,
    common_source: bytes,
) -> None:
    input_wheel = _write_test_wheel(tmp_path / "input", common=common_source)
    contract = _test_contract(input_wheel, common_source)
    wrong_pin = tmp_path / "vllm.pin"
    wrong_pin.write_text("vllm==0.16.1+cu130\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contain only vllm==0.16.0\\+cu130"):
        backport.build_backport(
            input_wheel,
            tmp_path / "pin-output",
            contract=contract,
            pin_file=wrong_pin,
        )

    corrupt = tmp_path / "corrupt" / input_wheel.name
    corrupt.parent.mkdir()
    shutil.copyfile(input_wheel, corrupt)
    with corrupt.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(RuntimeError, match="wheel hash"):
        backport.build_backport(corrupt, tmp_path / "hash-output", contract=contract)

    changed_source = _write_test_wheel(
        tmp_path / "source-drift", common=common_source + b"# drift\n"
    )
    changed_source_contract = replace(contract, wheel_sha256=backport.sha256_file(changed_source))
    with pytest.raises(RuntimeError, match="upstream rotary source hash"):
        backport.build_backport(
            changed_source, tmp_path / "source-output", contract=changed_source_contract
        )

    changed_metadata = _write_test_wheel(
        tmp_path / "metadata-drift", common=common_source, version="0.16.1+cu130"
    )
    changed_metadata_contract = replace(
        contract, wheel_sha256=backport.sha256_file(changed_metadata)
    )
    with pytest.raises(RuntimeError, match="version"):
        backport.build_backport(
            changed_metadata,
            tmp_path / "metadata-output",
            contract=changed_metadata_contract,
        )


def test_install_is_explicit_no_deps_and_probes_both_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    wheel = tmp_path / backport.CONTRACT.wheel_filename
    wheel.write_bytes(b"wheel")
    probes: list[bool] = []

    def fake_probe(_python: Path, *, patched: bool):
        probes.append(patched)
        return {"receipt": {"patched": patched}, "stdout": "probe\n"}

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="pip install passed\n")

    monkeypatch.setattr(backport, "_runtime_probe", fake_probe)
    monkeypatch.setattr(backport.subprocess, "run", fake_run)

    receipt = backport.install_backport(python.resolve(), wheel)

    assert probes == [False, True]
    assert commands == [
        [
            str(python.resolve()),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ]
    ]
    assert receipt["before"]["receipt"] == {"patched": False}
    assert receipt["after"]["receipt"] == {"patched": True}
