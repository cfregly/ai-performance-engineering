from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

from core.benchmark.run_manifest import (
    CPU_RUNTIME_PARITY_FIELDS,
    CUDA_RUNTIME_PARITY_FIELDS,
    EnvironmentInfo,
    GitInfo,
    HardwareInfo,
    RunManifest,
    RuntimeProvenance,
    RuntimeProvenanceParityError,
    SoftwareInfo,
    capture_runtime_provenance,
    compare_runtime_provenance,
    require_runtime_provenance_parity,
)


def _runtime(**updates: object) -> RuntimeProvenance:
    values: dict[str, object] = {
        "cuda_available": True,
        "driver_version": "580.126.09",
        "torch_version": "2.9.1+cu130",
        "cuda_version": "13.0",
        "cudnn_version": "91300",
        "python_version": "3.12.0",
        "os": "linux",
        "python_executable": "/usr/bin/python3",
        "process_id": 1234,
        "captured_at": "2026-09-07T00:00:00+00:00",
        "library_versions": {
            "nvidia-cublas-cu13": "13.0.0",
            "numpy": "2.1.2",
            "torch": "2.9.1+cu130",
            "triton": "3.5.1",
        },
        "library_versions_complete": True,
        "collection_warnings": [],
    }
    values.update(updates)
    return RuntimeProvenance(**values)


def _manifest(runtime: RuntimeProvenance | None) -> RunManifest:
    return RunManifest(
        hardware=HardwareInfo(
            gpu_model="NVIDIA B200",
            cuda_version="13.0",
            driver_version="580.126.09",
            compute_capability="10.0",
        ),
        software=SoftwareInfo(
            pytorch_version="2.9.1+cu130",
            triton_version="3.5.1",
            python_version="3.12.0",
            os="linux",
        ),
        runtime_provenance=runtime,
        environment=EnvironmentInfo(),
        git=GitInfo(commit="deadbeef", branch="main", dirty=False),
        start_time=datetime(2026, 9, 7),
    )


def test_capture_runtime_provenance_records_current_process_and_libraries() -> None:
    runtime = capture_runtime_provenance()

    assert runtime.process_id == os.getpid()
    assert Path(runtime.python_executable).resolve() == Path(sys.executable).resolve()
    assert runtime.torch_version == str(torch.__version__)
    assert runtime.python_version == sys.version.split()[0]
    assert runtime.os == sys.platform
    assert runtime.library_versions_complete is True
    assert runtime.library_versions["torch"]


def test_capture_runtime_provenance_runs_inside_real_child_process() -> None:
    code_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from core.benchmark.run_manifest import capture_runtime_provenance; "
                "print(capture_runtime_provenance().model_dump_json())"
            ),
        ],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)

    assert payload["process_id"] != os.getpid()
    assert Path(payload["python_executable"]).resolve() == Path(sys.executable).resolve()
    assert payload["torch_version"] == str(torch.__version__)
    assert payload["library_versions_complete"] is True


def test_library_capture_uses_sys_path_precedence_and_records_shadowed_versions(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    donor = tmp_path / "donor"
    for root, version in ((selected, "1.0"), (donor, "0.9")):
        metadata_dir = root / f"vllm-{version}.dist-info"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n",
            encoding="utf-8",
        )

    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(selected), str(donor), str(code_root)))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from core.benchmark.run_manifest import capture_runtime_provenance; "
                "print(capture_runtime_provenance().model_dump_json())"
            ),
        ],
        cwd=code_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)

    assert payload["library_versions"]["vllm"] == "1.0"
    assert "0.9" in payload["shadowed_library_versions"]["vllm"]
    assert payload["library_versions_complete"] is True


def test_library_capture_marks_conflicting_versions_at_same_precedence_unknown(
    tmp_path: Path,
) -> None:
    ambiguous = tmp_path / "ambiguous"
    for version in ("1.0", "2.0"):
        metadata_dir = ambiguous / f"vllm-{version}.dist-info"
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n",
            encoding="utf-8",
        )

    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ambiguous), str(code_root)))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from core.benchmark.run_manifest import capture_runtime_provenance; "
                "print(capture_runtime_provenance().model_dump_json())"
            ),
        ],
        cwd=code_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)

    assert payload["library_versions"]["vllm"] == "1.0 | 2.0"
    assert payload["library_versions_complete"] is False
    assert any(
        "conflicting active distribution versions" in warning
        for warning in payload["collection_warnings"]
    )


def test_canonical_cpu_and_cuda_parity_fields_are_explicit() -> None:
    assert CPU_RUNTIME_PARITY_FIELDS == (
        "torch_version",
        "python_version",
        "os",
        "library_versions",
    )
    assert CUDA_RUNTIME_PARITY_FIELDS == (
        "cuda_available",
        "driver_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "python_version",
        "os",
        "library_versions",
    )


def test_matching_executor_snapshots_pass_for_the_declared_target() -> None:
    reference = _manifest(_runtime(process_id=101))
    candidate = _manifest(
        _runtime(
            process_id=202,
            python_executable="/venv/bin/python",
            shadowed_library_versions={"vllm": ["0.15.0"]},
        )
    )

    cpu = compare_runtime_provenance(reference, candidate, target="cpu")
    cuda = require_runtime_provenance_parity(reference, candidate, target="cuda")

    assert cpu.matches is True
    assert cuda.matches is True
    assert cpu.mismatched_fields == []
    assert cuda.unknown_fields == []
    assert "process_id" not in cuda.required_fields
    assert "python_executable" not in cuda.required_fields


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("driver_version", "580.173.02"),
        ("torch_version", "2.10.0+cu130"),
        ("cuda_version", "13.1"),
        ("cudnn_version", "91400"),
        ("library_versions", {"numpy": "2.2.0", "torch": "2.9.1+cu130"}),
    ],
)
def test_cuda_parity_reports_each_runtime_or_library_mismatch(
    field_name: str,
    different_value: object,
) -> None:
    reference = _manifest(_runtime())
    candidate = _manifest(_runtime(**{field_name: different_value}))

    comparison = compare_runtime_provenance(reference, candidate, target="cuda")

    assert comparison.matches is False
    assert comparison.mismatched_fields == [field_name]
    assert comparison.unknown_fields == []
    assert comparison.fields[field_name].status == "mismatch"


def test_incomplete_library_snapshot_is_unknown_and_gate_raises() -> None:
    reference = _manifest(_runtime())
    candidate = _manifest(_runtime(library_versions_complete=False))

    comparison = compare_runtime_provenance(reference, candidate, target="cpu")

    assert comparison.matches is False
    assert comparison.mismatched_fields == []
    assert comparison.unknown_fields == ["library_versions"]
    with pytest.raises(RuntimeProvenanceParityError) as exc_info:
        require_runtime_provenance_parity(reference, candidate, target="cpu")
    assert exc_info.value.comparison == comparison


def test_cuda_profile_rejects_cpu_snapshots_even_when_they_match() -> None:
    cpu_runtime = _runtime(
        cuda_available=False,
        driver_version=None,
        cuda_version=None,
        cudnn_version=None,
    )
    reference = _manifest(cpu_runtime)
    candidate = _manifest(cpu_runtime.model_copy(deep=True))

    comparison = compare_runtime_provenance(reference, candidate, target="cuda")

    assert comparison.matches is False
    assert comparison.fields["cuda_available"].status == "mismatch"
    assert comparison.unknown_fields == [
        "driver_version",
        "cuda_version",
        "cudnn_version",
    ]


def test_historical_manifest_without_runtime_provenance_loads_but_is_unknown() -> None:
    legacy_payload = _manifest(None).model_dump(mode="json")
    legacy_payload.pop("runtime_provenance")

    historical = RunManifest.model_validate(legacy_payload)
    comparison = compare_runtime_provenance(historical, historical, target="cpu")

    assert historical.runtime_provenance is None
    assert comparison.matches is False
    assert comparison.unknown_fields == list(CPU_RUNTIME_PARITY_FIELDS)


def test_subprocess_coordinator_does_not_claim_executor_runtime() -> None:
    manifest = RunManifest.create(config={"execution_mode": "subprocess"})

    assert manifest.runtime_provenance is None
    manifest.finalize()
    comparison = compare_runtime_provenance(manifest, manifest, target="cpu")
    assert comparison.matches is False
    assert comparison.unknown_fields == list(CPU_RUNTIME_PARITY_FIELDS)


def test_runtime_parity_requires_an_explicit_supported_target() -> None:
    manifest = _manifest(_runtime())

    with pytest.raises(ValueError, match="Unsupported runtime parity target"):
        compare_runtime_provenance(manifest, manifest, target="gpu")  # type: ignore[arg-type]
