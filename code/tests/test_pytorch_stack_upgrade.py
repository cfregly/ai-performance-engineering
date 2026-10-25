"""Contracts for the repository's supported PyTorch 2.14 CUDA 13 stack."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from scripts import build_vllm_for_torch as vllm_builder

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent

EXPECTED_STACK = {
    "torch": "2.14.1+cu130",
    "torch-c-dlpack-ext": "0.1.5+torch2141.nobackend1",
    "torchaudio": "2.11.0+cu130",
    "torchvision": "0.29.1+cu130",
    "torchao": "0.18.0+cu130",
    "triton": "3.8.0",
    "torchtitan": "0.3.0",
    "cuda-python": "13.0.3",
    "flashinfer-python": "0.7.2",
    "transformer-engine": "2.20.2",
}


def _vllm_source_contract() -> dict[str, Any]:
    return json.loads(
        (CODE_ROOT / "vllm_source_build.json").read_text(encoding="utf-8")
    )


def _cluster_setup_function(name: str) -> str:
    setup = (CODE_ROOT / "cluster/setup.sh").read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n",
        setup,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"cluster/setup.sh must define {name}()"
    return match.group(0)


def _b200_smoke_expected_versions() -> dict[str, str]:
    module = ast.parse(
        (CODE_ROOT / "core/scripts/b200_torch2141_smoke.py").read_text(encoding="utf-8")
    )
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_VERSIONS"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _write_contract_wheel(
    tmp_path: Path,
    contract: dict[str, Any],
    *,
    extra_requirements: tuple[str, ...] = (),
    include_rust_executable: bool = True,
    filename_suffix: str = "",
) -> Path:
    expected = contract["expected_metadata"]
    metadata_lines = [
        "Metadata-Version: 2.4",
        f"Name: {expected['name']}",
        f"Version: {expected['version']}",
        f"Requires-Python: {expected['requires_python']}",
    ]
    requirements = [
        *expected["requires_dist"],
        *extra_requirements,
    ]
    metadata_lines.extend(f"Requires-Dist: {name}" for name in requirements)
    wheel_path = tmp_path / (
        "vllm-0.33.0.dev0+cu130.torch2141-cp312-cp312-linux_x86_64"
        f"{filename_suffix}.whl"
    )
    native_payload = bytearray(expected["minimum_native_member_size"])
    native_payload[:6] = b"\x7fELF\x02\x01"
    native_payload[18:20] = (62).to_bytes(2, "little")
    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "vllm-0.33.0.dev0+cu130.torch2141.dist-info/METADATA",
            "\n".join(metadata_lines) + "\n",
        )
        archive.writestr(
            "vllm/_C_stable_libtorch.cpython-312-x86_64-linux-gnu.so",
            native_payload,
        )
        archive.writestr(
            "vllm/_moe_C_stable_libtorch.cpython-312-x86_64-linux-gnu.so",
            native_payload,
        )
        if include_rust_executable:
            executable = ZipInfo("vllm/vllm-rs")
            executable.external_attr = 0o100755 << 16
            archive.writestr(executable, native_payload)
    return wheel_path


def _active_exact_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(
            r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^\s]+)",
            line,
        )
        if match is None:
            continue
        name = match.group("name").lower().replace("_", "-")
        assert name not in pins, f"duplicate exact pin for {name} in {path}"
        pins[name] = match.group("version")
    return pins


def _locked_exact_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(
            r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?=="
            r"(?P<version>[^\s\\]+)\s*\\?",
            raw_line,
        )
        if match is None:
            continue
        name = match.group("name").lower().replace("_", "-")
        assert name not in pins, f"duplicate exact pin for {name} in {path}"
        pins[name] = match.group("version")
    return pins


def test_requirements_select_the_supported_pytorch_214_stack() -> None:
    pins = _active_exact_pins(CODE_ROOT / "requirements_latest.txt")

    assert {name: pins.get(name) for name in EXPECTED_STACK} == EXPECTED_STACK


def test_linux_x86_64_lock_resolves_the_supported_pytorch_214_stack() -> None:
    pins = _locked_exact_pins(
        CODE_ROOT / "requirements_torch2141_linux_x86_64.txt"
    )

    assert {name: pins.get(name) for name in EXPECTED_STACK} == EXPECTED_STACK
    assert pins["datasets"] == "4.7.0"
    assert pins["fastapi"] == "0.136.1"
    assert pins["nvidia-cutlass-dsl"] == "4.8.0"
    assert "vllm" not in pins


def test_vllm_builder_requirements_are_present_in_the_active_lock() -> None:
    contract = _vllm_source_contract()
    requirements = _active_exact_pins(CODE_ROOT / "requirements_latest.txt")
    lock = _locked_exact_pins(
        CODE_ROOT / "requirements_torch2141_linux_x86_64.txt"
    )
    build_requirements = contract["build"]["resolved_python_requirements"]

    assert contract["build"]["rust_toolchain"] == "1.95.0"
    assert {name: requirements.get(name) for name in build_requirements} == (
        build_requirements
    )
    assert {name: lock.get(name) for name in build_requirements} == build_requirements


def test_setup_requires_core_versions_from_the_supported_stack() -> None:
    setup = (CODE_ROOT / "setup.sh").read_text(encoding="utf-8")
    fallback_names = {
        "torch": "PYTORCH_TORCH_VERSION",
        "torchaudio": "PYTORCH_TORCHAUDIO_VERSION",
        "torchvision": "PYTORCH_TORCHVISION_VERSION",
        "torchao": "PYTORCH_TORCHAO_VERSION",
        "triton": "PYTORCH_TRITON_VERSION",
    }

    for package, variable in fallback_names.items():
        assert f'{variable}="$(requirements_pin {package})"' in setup

    lock = _locked_exact_pins(CODE_ROOT / "requirements_torch2141_linux_x86_64.txt")
    assert lock["torchtitan"] == EXPECTED_STACK["torchtitan"]


def test_cpu_ci_exercises_the_same_torch_and_triton_release() -> None:
    workflow = (
        REPOSITORY_ROOT / ".github" / "workflows" / "benchmark-validation.yml"
    ).read_text(encoding="utf-8")

    assert f"triton=={EXPECTED_STACK['triton']}" in workflow
    assert "python -m pip install torch==2.14.1 " in workflow
    assert "--index-url https://download.pytorch.org/whl/cpu" in workflow


def test_full_graph_audit_is_bound_to_the_current_requirements() -> None:
    requirements = CODE_ROOT / "requirements_latest.txt"
    workflow = (
        REPOSITORY_ROOT / ".github" / "workflows" / "audit-linux-full-graph.yml"
    ).read_text(encoding="utf-8")
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()

    assert f'"{digest}"' in workflow


def test_run_manifest_schema_example_uses_the_supported_stack() -> None:
    from core.benchmark.run_manifest import RunManifest

    example = RunManifest.model_config["json_schema_extra"]["example"]

    assert example["software"]["pytorch_version"] == EXPECTED_STACK["torch"]
    assert example["software"]["triton_version"] == EXPECTED_STACK["triton"]
    assert example["runtime_provenance"]["torch_version"] == EXPECTED_STACK["torch"]
    assert example["runtime_provenance"]["library_versions"]["torch"] == (
        EXPECTED_STACK["torch"]
    )
    assert example["runtime_provenance"]["library_versions"]["triton"] == (
        EXPECTED_STACK["triton"]
    )


def test_setup_reads_every_abi_sensitive_pin_from_requirements() -> None:
    setup = (CODE_ROOT / "setup.sh").read_text(encoding="utf-8")
    parsed_packages = (
        "torch",
        "torchaudio",
        "torchvision",
        "torchao",
        "triton",
        "flashinfer-python",
    )

    for package in parsed_packages:
        assert f"requirements_pin {package}" in setup


def test_vllm_source_builder_accepts_only_the_declared_native_wheel_contract(
    tmp_path: Path,
) -> None:
    contract = _vllm_source_contract()
    wheel_path = _write_contract_wheel(tmp_path, contract)

    inspected = vllm_builder._inspect_wheel(contract, wheel_path)

    assert inspected["metadata"]["version"] == contract["package"]["version"]
    assert {
        member.split(".", 1)[0] for member in inspected["native_members"]
    } == {
        "vllm/_C_stable_libtorch",
        "vllm/_moe_C_stable_libtorch",
        "vllm/vllm-rs",
    }

    wheel_with_torch = _write_contract_wheel(
        tmp_path,
        contract,
        extra_requirements=("torch==2.14.1",),
        filename_suffix="-with-torch",
    )
    with pytest.raises(
        vllm_builder.BuildContractError,
        match="retained torch-family dependencies",
    ):
        vllm_builder._inspect_wheel(contract, wheel_with_torch)

    wheel_without_rust = _write_contract_wheel(
        tmp_path,
        contract,
        include_rust_executable=False,
        filename_suffix="-without-rust",
    )
    with pytest.raises(
        vllm_builder.BuildContractError,
        match="required archive member 'vllm/vllm-rs'",
    ):
        vllm_builder._inspect_wheel(contract, wheel_without_rust)


def test_vllm_source_builder_refuses_a_nonempty_output_directory_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path = CODE_ROOT / "vllm_source_build.json"
    contract = _vllm_source_contract()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / vllm_builder.RECEIPT_NAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        vllm_builder,
        "_validate_environment",
        lambda _contract: pytest.fail("environment validation must not run"),
    )

    with pytest.raises(
        vllm_builder.BuildContractError,
        match="Output directory is not empty",
    ):
        vllm_builder._build(
            contract=contract,
            contract_path=contract_path,
            output_dir=output_dir,
            cuda_arch_list="10.0",
            max_jobs=4,
            nvcc_threads=2,
        )



def test_shared_lock_has_one_harmony_module_provider() -> None:
    requirements = _active_exact_pins(CODE_ROOT / "requirements_latest.txt")
    lock = _locked_exact_pins(CODE_ROOT / "requirements_torch2141_linux_x86_64.txt")
    # Both distributions own openai_harmony and its native binary. The pinned
    # vLLM source requires oss-harmony, so retaining the old provider corrupts it.
    assert requirements["oss-harmony"] == "0.0.11"
    assert lock["oss-harmony"] == "0.0.11"
    assert "openai-harmony" not in requirements
    assert "openai-harmony" not in lock


def test_cluster_stack_dry_run_selects_only_the_hash_bound_graph(tmp_path: Path) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    date = binary_dir / "date"
    date.write_text("#!/bin/sh\nprintf '2026-10-22T00:00:00Z\\n'\n")
    date.chmod(0o755)
    venv = tmp_path / "new runtime"
    environment = {
        **os.environ,
        "PATH": f"{binary_dir}:/usr/bin:/bin",
        "RUN_ID": "2026-10-22_upgrade_test",
        "LABEL": "compatibility",
        "USER": "testworker",
        "VENV_DIR": str(venv),
        "CLUSTER_RESULTS_RAW_DIR": str(tmp_path / "raw"),
        "CLUSTER_RESULTS_STRUCTURED_DIR": str(tmp_path / "structured"),
    }
    result = subprocess.run(
        ["/bin/bash", str(CODE_ROOT / "cluster/scripts/enable_researcher_stack.sh"),
         "--no-enable-services", "--no-docker-group", "--install-python-stack"],
        env=environment, text=True, capture_output=True, check=True,
    )
    commands = [shlex.split(line.removeprefix("[dry-run] "))
                for line in result.stdout.splitlines() if line.startswith("[dry-run] ")]
    installs = [command for command in commands if "install" in command]
    assert installs == [[str(venv / "bin/python"), "-m", "pip", "install",
                         "--require-hashes", "--no-deps", "-r",
                         str(CODE_ROOT / "requirements_torch2141_linux_x86_64.txt")]]
    assert not venv.exists()
    assert "Build and validate those source wheels" in result.stdout


def test_cluster_setup_uses_the_locked_stack_and_pinned_vllm_builder() -> None:
    setup = (CODE_ROOT / "cluster/setup.sh").read_text(encoding="utf-8")

    assert "requirements_torch2141_linux_x86_64.txt" in setup
    assert "--require-hashes --no-deps" in setup
    assert "scripts/build_vllm_for_torch.py" in setup
    assert "vllm_source_build.json" in setup
    assert "torch.compiled_with_cxx11_abi()" in setup
    assert "2.10.0a0" not in setup
    assert "pypi.ngc.nvidia.com" not in setup
    assert 'pip install vllm' not in setup
    assert "--force-reinstall" not in setup


def test_cluster_setup_bootstraps_vllm_rust_and_native_smoke_dependency() -> None:
    setup = (CODE_ROOT / "cluster/setup.sh").read_text(encoding="utf-8")
    rust_toolchain = _vllm_source_contract()["build"]["rust_toolchain"]

    assert rust_toolchain == "1.95.0"
    assert f'VLLM_RUST_TOOLCHAIN="{rust_toolchain}"' in setup
    rust_install = setup.index('rustup toolchain install "$expected" --profile minimal')
    rust_validation = setup.index('require_exact_rust_toolchain "$expected"')
    vllm_build = setup.index('"${CODE_ROOT}/scripts/build_vllm_for_torch.py"')
    assert rust_install < rust_validation < vllm_build
    assert '[[ "$rustc_version" != "rustc ${expected} "* ]]' in setup
    assert '[[ "$cargo_version" != "cargo ${expected} "* ]]' in setup

    enable_vllm = setup.index("--install-vllm)")
    enable_vllm_end = setup.index(";;", enable_vllm)
    assert "INSTALL_TRANSFORMER_ENGINE=1" in setup[enable_vllm:enable_vllm_end]
    native_build = setup.index('"${CODE_ROOT}/scripts/build_torch_extensions.py"')
    assert native_build < vllm_build
    assert '--contract "$NATIVE_EXTENSION_CONTRACT"' in setup
    assert "--package transformer-engine-torch" in setup
    assert '"packages"]["transformer-engine-torch"]["wheel"]' in setup
    assert "Transformer Engine build receipt wheel hash does not match" in setup
    assert '"$VENV_PY" -m pip install --no-deps "$TE_WHEEL"' in setup
    assert 'package["expected_metadata"]["version"]' in setup
    assert 'pip install transformer-engine-torch' not in setup
    assert "libdw-dev" in setup
    assert "libelf-dev" in setup


def test_cluster_setup_covers_every_smoke_distribution_outside_the_lock() -> None:
    lock = _locked_exact_pins(
        CODE_ROOT / "requirements_torch2141_linux_x86_64.txt"
    )
    missing_from_lock = set(_b200_smoke_expected_versions()) - set(lock)
    setup = (CODE_ROOT / "cluster/setup.sh").read_text(encoding="utf-8")

    assert missing_from_lock == {"transformer-engine-torch", "vllm"}
    assert "--package transformer-engine-torch" in setup
    assert "scripts/build_vllm_for_torch.py" in setup


def test_cluster_rust_validator_requires_exact_rustc_and_cargo(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rustc = bin_dir / "rustc"
    cargo = bin_dir / "cargo"
    rustc.write_text("#!/bin/sh\nprintf 'rustc 1.95.0 (test 2026-10-01)\\n'\n")
    cargo.write_text("#!/bin/sh\nprintf 'cargo 1.95.0 (test 2026-10-01)\\n'\n")
    rustc.chmod(0o755)
    cargo.chmod(0o755)
    script = (
        "set -e\n"
        f"{_cluster_setup_function('require_exact_rust_toolchain')}"
        "require_exact_rust_toolchain 1.95.0\n"
    )
    environment = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}

    accepted = subprocess.run(
        ["/bin/bash", "-s"],
        input=script,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "Using rustc 1.95.0" in accepted.stdout
    assert "Using cargo 1.95.0" in accepted.stdout

    cargo.write_text("#!/bin/sh\nprintf 'cargo 1.94.0 (test 2026-09-01)\\n'\n")
    rejected = subprocess.run(
        ["/bin/bash", "-s"],
        input=script,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "Expected cargo 1.95.0, found cargo 1.94.0" in rejected.stderr


def test_cluster_setup_rejects_skipping_python_with_source_builds() -> None:
    setup = CODE_ROOT / "cluster/setup.sh"

    for install_option in ("--install-transformer-engine", "--install-vllm"):
        result = subprocess.run(
            ["/bin/bash", str(setup), "--skip-python", install_option],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert (
            "--skip-python cannot be combined with source-build install options."
            in result.stderr
        )


def test_cufile_sidecar_matches_the_supported_cuda_stack() -> None:
    pins = _active_exact_pins(CODE_ROOT / "ch10/requirements_cufile.txt")
    lock = _locked_exact_pins(
        CODE_ROOT / "requirements_torch2141_linux_x86_64.txt"
    )
    packages = (
        "cuda-python",
        "cuda-bindings",
        "nvidia-cufile",
        "torch",
        "numpy",
    )

    assert {package: pins.get(package) for package in packages} == {
        package: lock[package] for package in packages
    }


def test_torchtitan_demo_binds_current_compile_config_to_the_mesh() -> None:
    source = (CODE_ROOT / "ch04/torchtitan_async_tp_multigpu_demo.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "enable_async_tp")

    @dataclass(kw_only=True, slots=True)
    class CompileConfig:
        enable: bool
        enable_async_tensor_parallel: bool
        components: list[str]
        backend: str

    calls = []
    namespace = {
        "CompileConfig": CompileConfig,
        "_maybe_enable_async_tp": lambda config, mesh: calls.append((config, mesh)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "async_tp", "exec"), namespace)
    mesh = object()
    config = namespace["enable_async_tp"](mesh)
    assert isinstance(config, CompileConfig)
    assert config.enable and config.enable_async_tensor_parallel
    assert config.components == ["model"]
    assert config.backend == "inductor"
    assert calls == [(config, mesh)]
    assert "torchtitan.config.job_config" not in source
