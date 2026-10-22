"""Tests for the privileged setup repository bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent.resolve()
SETUP_SCRIPT = CODE_ROOT / "setup.sh"

EXPECTED_TORCH = "2.14.1+cu130"
EXPECTED_TORCHAUDIO = "2.11.0+cu130"
EXPECTED_TORCHVISION = "0.29.1+cu130"
EXPECTED_TORCHAO = "0.18.0+cu130"
EXPECTED_TRITON = "3.8.0"
EXPECTED_VLLM = "0.33.0.dev0+cu130.torch2141"


def _fake_git(tmp_path: Path, *, fail_on: str = "") -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "git.log"
    git_path = bin_dir / "git"
    git_path.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$AISP_TEST_GIT_LOG"
if [ -n "$AISP_TEST_GIT_FAIL_ON" ]; then
    case "$*" in
        *"$AISP_TEST_GIT_FAIL_ON"*) exit 42 ;;
    esac
fi
exit 0
""",
        encoding="utf-8",
    )
    git_path.chmod(0o755)
    return bin_dir, log_path


def _run_bootstrap(tmp_path: Path, *, fail_on: str = "") -> subprocess.CompletedProcess[str]:
    bin_dir, log_path = _fake_git(tmp_path, fail_on=fail_on)
    setup_link = tmp_path / "setup-link.sh"
    setup_link.symlink_to(SETUP_SCRIPT)
    env = os.environ.copy()
    env.update(
        {
            "AISP_SETUP_BOOTSTRAP_ONLY": "1",
            "AISP_TEST_GIT_FAIL_ON": fail_on,
            "AISP_TEST_GIT_LOG": str(log_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
        }
    )
    return subprocess.run(
        ["/bin/bash", str(setup_link)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_rejected_host(
    tmp_path: Path,
    *,
    architecture: str,
    distributor: str,
    version: str,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    mutation_log = tmp_path / "mutations.log"
    _write_executable(
        bin_dir / "uname",
        f"#!/bin/sh\nprintf '%s\\n' '{architecture}'\n",
    )
    _write_executable(
        bin_dir / "lsb_release",
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  -is) printf '%s\\n' '{distributor}' ;;\n"
        f"  -rs) printf '%s\\n' '{version}' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
    )
    mutation_stub = (
        "#!/bin/sh\n" 'printf \'%s %s\\n\' "$0" "$*" >> "$AISP_TEST_MUTATION_LOG"\n' "exit 97\n"
    )
    for command in (
        "apt",
        "apt-get",
        "chmod",
        "chown",
        "git",
        "install",
        "ln",
        "mkdir",
        "pip3",
        "python3",
        "rm",
        "sed",
        "sudo",
        "tee",
        "touch",
    ):
        _write_executable(bin_dir / command, mutation_stub)

    env = os.environ.copy()
    env.pop("AISP_SETUP_BOOTSTRAP_ONLY", None)
    env.update(
        {
            "AISP_TEST_MUTATION_LOG": str(mutation_log),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "SETUP_LOG_FILE": str(tmp_path / "setup.log"),
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(SETUP_SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    return result, mutation_log


def test_bootstrap_resolves_physical_root_and_marks_it_safe_before_git_reads(
    tmp_path: Path,
) -> None:
    result = _run_bootstrap(tmp_path)
    calls = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(f"Repository bootstrap complete: {REPOSITORY_ROOT}\n")
    assert calls[0] == f"config --global --add safe.directory {REPOSITORY_ROOT}"
    assert calls[1:] == [
        f"-C {REPOSITORY_ROOT} submodule sync --recursive",
        f"-C {REPOSITORY_ROOT} submodule update --init --recursive",
    ]
    assert all("rev-parse" not in call for call in calls)


def test_bootstrap_propagates_injected_submodule_git_failure(tmp_path: Path) -> None:
    result = _run_bootstrap(tmp_path, fail_on="submodule update --init --recursive")
    calls = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()

    assert result.returncode == 42
    assert calls[-1] == f"-C {REPOSITORY_ROOT} submodule update --init --recursive"
    assert "Repository bootstrap complete" not in result.stdout


def test_unsupported_host_rejection_precedes_every_setup_mutation(tmp_path: Path) -> None:
    cases = (
        ("x86_64", "Ubuntu", "20.04", "Ubuntu 20.04 is not supported"),
        ("aarch64", "Ubuntu", "22.04", "Architecture aarch64 is not supported"),
    )
    for index, (architecture, distributor, version, message) in enumerate(cases):
        result, mutation_log = _run_rejected_host(
            tmp_path / str(index),
            architecture=architecture,
            distributor=distributor,
            version=version,
        )

        assert result.returncode != 0
        assert message in result.stderr
        assert not mutation_log.exists(), mutation_log.read_text(encoding="utf-8")
        assert not (tmp_path / str(index) / "setup.log").exists()


def _pytorch_source_script() -> str:
    """Execute the real source-selection blocks without privileged setup actions."""
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    def shell_function(name: str) -> str:
        match = re.search(
            rf"^{re.escape(name)}\(\) \{{\n.*?^\}}$",
            setup,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, name
        return match.group(0)

    pin_assignments = "\n".join(
        match.group(0)
        for name in (
            "PYTORCH_TORCH_VERSION",
            "PYTORCH_TORCHAUDIO_VERSION",
            "PYTORCH_TORCHVISION_VERSION",
            "PYTORCH_TORCHAO_VERSION",
            "PYTORCH_TRITON_VERSION",
            "FLASHINFER_EXPECTED_VERSION",
            "REQ_CMAKE_VERSION",
            "REQ_NINJA_VERSION",
        )
        if (
            match := re.search(
                rf"^{name}=\"\$\(requirements_pin [^)]+\)\"$",
                setup,
                re.MULTILINE,
            )
        )
    )
    pins = setup.split("# Single-source serving stack pins", 1)[1].split(
        "# This lock targets Linux x86_64", 1
    )[0]
    return (
        "set -e\n"
        'REQUIREMENTS_FILE="$1"\n'
        'VLLM_PIN_FILE="$2"\n'
        'VLLM_SOURCE_BUILD_MANIFEST="$4"\n'
        'VLLM_PIP_SPEC=""\n'
        'FLASHINFER_EXPECTED_VERSION=""\n'
        'HOST_ARCH="${3:-x86_64}"\n'
        'PYTORCH_STABLE_INDEX="https://download.pytorch.org/whl"\n'
        'PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly"\n'
        f"{shell_function('requirements_pin')}\n"
        f"{shell_function('pytorch_package_index')}\n"
        f"{shell_function('require_torchao_cuda_wheel_support')}\n"
        'if [ ! -f "${REQUIREMENTS_FILE}" ]; then\n'
        '    echo "ERROR: Required pin file is missing: ${REQUIREMENTS_FILE}" >&2\n'
        "    exit 1\n"
        "fi\n"
        f"{pin_assignments}\n"
        f"# Single-source serving stack pins{pins}\n"
    )


def _resolve_pytorch_sources(
    requirements: Path,
    *,
    find_links_override: str | None = None,
    architecture: str = "x86_64",
) -> dict[str, str]:
    source_vars = (
        "PYTORCH_TORCH_VERSION",
        "PYTORCH_CU130_INDEX",
        "PYTORCH_TORCH_FIND_LINKS",
        "PYTORCH_TORCHAUDIO_INDEX",
        "PYTORCH_TORCHAUDIO_VERSION",
        "PYTORCH_TORCHVISION_VERSION",
        "PYTORCH_TORCHAO_INDEX",
        "PYTORCH_TORCHAO_VERSION",
        "PYTORCH_TRITON_INDEX",
        "PYTORCH_TRITON_VERSION",
        "FLASHINFER_EXPECTED_VERSION",
        "VLLM_PIP_SPEC",
    )
    script = _pytorch_source_script()
    for name in source_vars:
        script += f'printf "SOURCE:{name}=%s\\n" "${{{name}:-}}"\n'
    env = os.environ.copy()
    env.pop("PYTORCH_TORCH_FIND_LINKS", None)
    if find_links_override is not None:
        env["PYTORCH_TORCH_FIND_LINKS"] = find_links_override
    result = subprocess.run(
        [
            "/bin/bash",
            "-s",
            "--",
            str(requirements),
            str(CODE_ROOT / "vllm_no_deps.pin"),
            architecture,
            str(CODE_ROOT / "vllm_source_build.json"),
        ],
        input=script,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return dict(
        line.removeprefix("SOURCE:").split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("SOURCE:")
    )


def test_pinned_stable_stack_uses_compatible_published_stable_sources() -> None:
    sources = _resolve_pytorch_sources(CODE_ROOT / "requirements_latest.txt")

    assert sources["PYTORCH_TORCH_VERSION"] == EXPECTED_TORCH
    assert sources["PYTORCH_CU130_INDEX"] == "https://download.pytorch.org/whl/cu130"
    assert sources["PYTORCH_TORCH_FIND_LINKS"] == ("https://download.pytorch.org/whl/cu130/torch/")
    assert sources["PYTORCH_TORCHAUDIO_INDEX"] == sources["PYTORCH_CU130_INDEX"]
    assert sources["PYTORCH_TORCHAUDIO_VERSION"] == EXPECTED_TORCHAUDIO
    assert sources["PYTORCH_TORCHVISION_VERSION"] == EXPECTED_TORCHVISION
    assert sources["PYTORCH_TORCHAO_VERSION"] == EXPECTED_TORCHAO
    assert sources["PYTORCH_TORCHAO_INDEX"] == sources["PYTORCH_CU130_INDEX"]
    assert sources["PYTORCH_TRITON_INDEX"] == "https://download.pytorch.org/whl"
    # The official torch 2.14.1+cu130 wheel requires Triton 3.8.0 on Linux.
    assert sources["PYTORCH_TRITON_VERSION"] == EXPECTED_TRITON


def test_nightly_requirements_select_nightly_sources(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "torch==2.10.0.dev20251213+cu130\n"
        "torchaudio==2.10.0.dev20251213+cu130\n"
        "torchvision==0.25.0.dev20251213+cu130\n"
        "triton==3.6.0+git8fedd49b\n"
        "torchao==0.17.0.dev20260301+cu130\n"
        "flashinfer-python==0.6.3\n"
        "cmake==4.4.4\n"
        "ninja==1.13.2\n",
        encoding="utf-8",
    )
    sources = _resolve_pytorch_sources(requirements)

    assert sources["PYTORCH_CU130_INDEX"] == ("https://download.pytorch.org/whl/nightly/cu130")
    assert sources["PYTORCH_TORCH_FIND_LINKS"] == (
        "https://download.pytorch.org/whl/nightly/cu130/torch/"
    )
    assert sources["PYTORCH_TORCHAUDIO_INDEX"] == sources["PYTORCH_CU130_INDEX"]
    assert sources["PYTORCH_TORCHAO_VERSION"] == "0.17.0.dev20260301+cu130"
    assert sources["PYTORCH_TORCHAO_INDEX"] == sources["PYTORCH_CU130_INDEX"]
    assert sources["PYTORCH_TRITON_INDEX"] == "https://download.pytorch.org/whl/nightly"


def test_missing_requirements_fails_before_source_selection(tmp_path: Path) -> None:
    missing = tmp_path / "missing-requirements.txt"
    result = subprocess.run(
        [
            "/bin/bash",
            "-s",
            "--",
            str(missing),
            str(CODE_ROOT / "vllm_no_deps.pin"),
            "x86_64",
            str(CODE_ROOT / "vllm_source_build.json"),
        ],
        input=_pytorch_source_script(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert f"Required pin file is missing: {missing}" in result.stderr


def test_aarch64_keeps_the_published_cuda_suffix_for_torchaudio_and_torch() -> None:
    sources = _resolve_pytorch_sources(
        CODE_ROOT / "requirements_latest.txt", architecture="aarch64"
    )
    assert sources["PYTORCH_TORCHAUDIO_VERSION"] == EXPECTED_TORCHAUDIO
    assert sources["PYTORCH_TORCH_VERSION"] == EXPECTED_TORCH
    assert sources["PYTORCH_TORCHAUDIO_INDEX"] == "https://download.pytorch.org/whl/cu130"


def test_aarch64_missing_torchao_cuda_wheel_fails_with_source_build_guidance() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    preflight = 'require_torchao_cuda_wheel_support "${HOST_ARCH}"'
    assert preflight in setup
    assert setup.index(preflight) < setup.index("pip_install --require-hashes")
    result = subprocess.run(
        [
            "/bin/bash",
            "-s",
            "--",
            str(CODE_ROOT / "requirements_latest.txt"),
            str(CODE_ROOT / "vllm_no_deps.pin"),
            "aarch64",
            str(CODE_ROOT / "vllm_source_build.json"),
        ],
        input=_pytorch_source_script() + f"\n{preflight}\n",
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "No official aarch64 CUDA wheel" in result.stderr
    assert f"torchao=={EXPECTED_TORCHAO}" in result.stderr
    assert "v0.18.0" in result.stderr
    assert "source build" in result.stderr


def test_hash_locked_environment_includes_the_exact_torch_ecosystem() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    lock = (CODE_ROOT / "requirements_torch2141_linux_x86_64.lock").read_text(encoding="utf-8")
    assert f"torchao=={EXPECTED_TORCHAO}" in requirements.splitlines()
    assert re.search(rf"^torch=={re.escape(EXPECTED_TORCH)} \\$", lock, re.MULTILINE)
    assert re.search(rf"^torchao=={re.escape(EXPECTED_TORCHAO)} \\$", lock, re.MULTILINE)
    assert "REQUIREMENTS_EXCLUDE_REGEX" not in setup
    assert "TEMP_REQUIREMENTS" not in setup
    locked_install = setup.split("# Install the preflight-validated Linux/Python 3.12 graph", 1)[
        1
    ].split("# The graph includes the exact CUDA 13 torch family", 1)[0]
    assert "--require-hashes --no-deps" in locked_install
    assert '--find-links "${PROJECT_ROOT}/vendor/python-wheels"' in locked_install
    assert '-r "${PYTORCH_STACK_LOCK}"' in locked_install
    assert setup.index('export VIRTUAL_ENV="${AISP_VENV_DIR}"') < setup.index(
        "pip_install --require-hashes"
    )
    assert setup.index('export PATH="${AISP_VENV_DIR}/bin:${PATH}"') < setup.index(
        "pip_install --require-hashes"
    )
    assert "pip_uninstall -y torch" not in setup
    for package in (
        "torch",
        "torchaudio",
        "torchvision",
        "torchao",
        "triton",
        "flashinfer-python",
        "cmake",
        "ninja",
    ):
        assert f"requirements_pin {package}" in setup
    assert setup.index("Core requirements pins match the hash lock") < setup.index(
        "\nlock_gpu_clocks_if_supported\n"
    )


def test_explicit_torch_find_links_is_preserved() -> None:
    sources = _resolve_pytorch_sources(
        CODE_ROOT / "requirements_latest.txt",
        find_links_override="/opt/aisp/pytorch-wheels",
    )
    assert sources["PYTORCH_TORCH_FIND_LINKS"] == "/opt/aisp/pytorch-wheels"
    assert sources["PYTORCH_CU130_INDEX"] == "https://download.pytorch.org/whl/cu130"


def test_setup_installs_one_hash_bound_graph_without_package_reinstalls() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    installs = re.findall(r"^.*pip_install .*$", setup, re.MULTILINE)
    assert len(installs) == 3
    assert "--require-hashes --no-deps" in installs[0]
    assert '--no-deps "${TE_WHEEL_PATH}"' in installs[1]
    assert '--no-deps "${VLLM_WHEEL_PATH}"' in installs[2]
    assert "--ignore-installed" not in setup
    assert "pip_uninstall -y torch" not in setup


def test_locked_install_failure_stops_before_native_builds(tmp_path: Path) -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    section = setup.split("# Install the preflight-validated Linux/Python 3.12 graph", 1)[1]
    section = section.split("# The graph includes the exact CUDA 13 torch family", 1)[0]
    section = section.split("\n", 1)[1]
    (tmp_path / "requirements_torch2141_linux_x86_64.lock").write_text("fixture")
    log = tmp_path / "pip-args.json"
    script = (
        """set -eu
pip_install() {
    printf '%s\n' "$@" > "$AISP_TEST_PIP_ARGS"
    return 42
}
"""
        + section
        + "\necho UNEXPECTED_CONTINUATION\n"
    )
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={
            **os.environ,
            "PROJECT_ROOT": str(tmp_path),
            "PYTORCH_STACK_LOCK": str(tmp_path / "requirements_torch2141_linux_x86_64.lock"),
            "AISP_VENV_DIR": str(tmp_path / "venv"),
            "AISP_TEST_PIP_ARGS": str(log),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "locked PyTorch runtime installation failed" in result.stdout
    assert "UNEXPECTED_CONTINUATION" not in result.stdout
    args = log.read_text().splitlines()
    assert args[:2] == ["--require-hashes", "--no-deps"]
    assert args[2:4] == [
        "--find-links",
        str(tmp_path / "vendor/python-wheels"),
    ]
    assert args[-2:] == ["-r", str(tmp_path / "requirements_torch2141_linux_x86_64.lock")]


def test_locked_dlpack_wheel_is_bound_to_the_vendored_artifact() -> None:
    lock = (CODE_ROOT / "requirements_torch2141_linux_x86_64.lock").read_text(
        encoding="utf-8"
    )
    wheel = next(
        (CODE_ROOT / "vendor" / "python-wheels").glob(
            "torch_c_dlpack_ext-0.1.5+torch2141.nobackend1-*.whl"
        )
    )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    entry = re.search(
        r"^torch-c-dlpack-ext==0\.1\.5\+torch2141\.nobackend1 \\\n"
        r"    --hash=sha256:([0-9a-f]{64})$",
        lock,
        re.MULTILINE,
    )

    assert entry is not None
    assert entry.group(1) == digest


def test_vllm_native_build_headers_are_explicit_system_prerequisites() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    section = setup.split("Installing system performance tools...", 1)[1].split(
        "Installing the isolated Rust 1.95.0 toolchain...", 1
    )[0]

    assert re.search(r"^    libdw-dev \\\s*$", section, re.MULTILINE)
    assert re.search(r"^    libelf-dev \\\s*$", section, re.MULTILINE)


def test_requirements_quickstart_exposes_the_cu130_index() -> None:
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    active_lines = {
        line.strip() for line in requirements.splitlines() if not line.lstrip().startswith("#")
    }
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in active_lines


def test_vllm_uses_the_fixed_source_build_and_installs_its_wheel_without_deps() -> None:
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    vllm_pin = (CODE_ROOT / "vllm_no_deps.pin").read_text(encoding="utf-8")
    source_contract = json.loads((CODE_ROOT / "vllm_source_build.json").read_text(encoding="utf-8"))
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    sources = _resolve_pytorch_sources(CODE_ROOT / "requirements_latest.txt")

    assert not re.search(r"^vllm==", requirements, re.MULTILINE)
    assert re.findall(r"^vllm==[^\s#]+", vllm_pin, re.MULTILINE) == [f"vllm=={EXPECTED_VLLM}"]
    assert source_contract["package"] == {
        "name": "vllm",
        "version": EXPECTED_VLLM,
    }
    assert sources["VLLM_PIP_SPEC"] == f"vllm=={EXPECTED_VLLM}"

    install = setup.split("Installing pinned vLLM serving stack", 1)[1].split(
        "Installing pinned vLLM runtime dependencies", 1
    )[0]
    assert "vllm_source_build.json" in setup
    assert "scripts/build_vllm_for_torch.py" in install
    assert "vllm-source-wheel.manifest.json" in install
    assert "--output-dir" in install
    assert "--cuda-arch-list" in install
    assert "--max-jobs" in install
    assert "--no-deps" in install
    assert "wheels.vllm.ai" not in install
    assert "--only-binary=:all:" not in install
    assert "vllm._C_stable_libtorch" in setup
    assert "vllm._moe_C_stable_libtorch" in setup
    assert "vllm-rs" in setup
    assert "os.access(rust_frontend, os.X_OK)" in setup
    assert 'importlib.import_module("vllm._C")' not in setup


def test_vllm_dependencies_are_resolved_by_the_shared_lock() -> None:
    from packaging.requirements import Requirement
    from packaging.version import Version

    lock = (CODE_ROOT / "requirements_torch2141_linux_x86_64.lock").read_text()
    setup = SETUP_SCRIPT.read_text()
    torch_line = next(line for line in lock.splitlines() if line.startswith("torch=="))
    requirement = Requirement(torch_line.rstrip("\\ "))
    assert Version(EXPECTED_TORCH) in requirement.specifier
    for other_build in ("2.14.1", "2.14.1+cpu", "2.14.1+cu128", "2.15.0+cu130"):
        assert Version(other_build) not in requirement.specifier
    for package in ("anthropic", "openai", "mcp", "torchtitan", "accelerate"):
        assert re.search(rf"^{package}==", lock, re.MULTILINE)
    assert "VLLM_RUNTIME_DEPS=" not in setup
    assert "VLLM_DEPS_CONSTRAINTS=" not in setup


def test_setup_uses_hash_locked_cli_pins_without_fallback_installs() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    requirements = (CODE_ROOT / "requirements_latest.txt").read_text(encoding="utf-8")
    lock = (CODE_ROOT / "requirements_torch2141_linux_x86_64.lock").read_text(encoding="utf-8")
    expected = {
        "typer": "0.27.3",
        "click": "8.5.0",
        "rich": "13.9.4",
    }

    for package, version in expected.items():
        assert re.search(rf"^{package}=={re.escape(version)}(?:\s|$)", requirements, re.MULTILINE)
        assert re.search(rf"^{package}=={re.escape(version)} \\$", lock, re.MULTILINE)
        assert f"{package}==" not in setup
    assert not re.search(r"^typer-slim(?:==|\s|$)", requirements, re.MULTILINE)
    assert "--require-hashes --no-deps" in setup


def test_setup_uses_a_fresh_venv_without_mutating_system_python() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    guard = 'if [ -e "${AISP_VENV_DIR}" ] || [ -L "${AISP_VENV_DIR}" ]; then'
    bootstrap_call = "\nbootstrap_repository_checkout\n"

    assert setup.index(guard) < setup.index(bootstrap_call)
    assert '"${PYTHON_TARGET_BIN}" -m venv "${AISP_VENV_DIR}"' in setup
    assert setup.index(guard) < setup.index('"${PYTHON_TARGET_BIN}" -m venv "${AISP_VENV_DIR}"')
    assert "get-pip.py" not in setup
    assert "--break-system-packages" not in setup
    assert "PIP_BREAK_SYSTEM_PACKAGES" not in setup
    assert not re.search(r"update-alternatives .*?/usr/bin/python", setup)
    assert "pip install --user" not in setup
    assert "site.getsitepackages" not in setup
    assert "python3-apt" not in setup
    assert "python3-libnvinfer" not in setup
    assert "CODEX_INSTALL_CMD" not in setup
    assert "sudo -H -u" not in setup


def test_setup_switches_the_global_cli_only_after_success_checks() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    wrapper_call = setup.rindex("\ninstall_aisp_cli_wrapper\n")

    assert wrapper_call > setup.index("Every locked runtime version matches")
    assert wrapper_call > setup.index("Running comprehensive Blackwell feature verification")
    assert wrapper_call < setup.index("Final cleanup (APT/pip caches)")


def test_cuda_toolkit_and_tools_are_exact_without_generic_fallbacks() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'apt install -y "cuda-toolkit-13-0=13.0.2-1"' in setup
    assert 'apt install -y "cuda-command-line-tools-13-0=13.0.2-1"' in setup
    assert "INSTALLED_CUDA_TOOLKIT_VERSION" in setup
    assert "INSTALLED_CUDA_TOOLS_VERSION" in setup
    assert "dpkg-query -W -f='${Version}' cuda-toolkit-13-0" in setup
    assert "dpkg-query -W -f='${Version}' cuda-command-line-tools-13-0" in setup
    assert "nvidia-cuda-toolkit" not in setup
    assert not re.search(r"apt install -y cuda-command-line-tools(?:\s|$)", setup)
    assert not re.search(r"apt install -y cuda-sanitizer(?:\s|$)", setup)
    assert "Refusing to build with a different toolkit" in setup
    assert "compute-sanitizer is missing" in setup


def test_runtime_uses_locked_wheel_sdk_assets_without_apt_replacement() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    runtime = setup.split("# Select wheel runtime libraries", 1)[1].split(
        "if ! PYTORCH_EXPECTED_VERSION", 1
    )[0]
    summary = setup.split("# Final summary\n", 1)[1]

    expected_assets = (
        '("nvidia-nccl-cu13", "nccl", "nccl.h", "libnccl.so.2")',
        '("nvidia-cudnn-cu13", "cudnn", "cudnn.h", "libcudnn.so.9")',
        '("nvidia-nvshmem-cu13", "nvshmem", "nvshmem.h", ' '"libnvshmem_host.so.3")',
    )
    for asset in expected_assets:
        assert asset in runtime
    assert 'if [ "${#CUDA_WHEEL_SDK_PATHS[@]}" -ne 12 ]; then' in runtime
    assert "metadata.distribution(distribution_name)" in runtime
    assert "if not header.is_file() or not library.is_file():" in runtime
    assert 'export LD_LIBRARY_PATH="${RUNTIME_WHEEL_LIB_DIRS}' in runtime
    assert 'export NCCL_HOME="${NCCL_WHEEL_ROOT}"' in runtime
    assert 'export CUDNN_HOME="${CUDNN_WHEEL_ROOT}"' in runtime
    assert 'export NVSHMEM_HOME="${NVSHMEM_WHEEL_ROOT}"' in runtime
    assert "NCCL_WHEEL_VERSION" in summary
    assert "CUDNN_WHEEL_VERSION" in summary
    assert "NVSHMEM_WHEEL_VERSION" in summary
    assert "report_installed_cudnn" not in setup
    assert not re.search(
        r"apt (?:install|remove|purge).*?(?:nccl|cudnn|nvshmem)",
        setup,
        re.IGNORECASE,
    )


def test_cuda_compile_and_blackwell_verifications_fail_closed() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    checks = setup.split("# Comprehensive setup verification", 1)[1].split(
        "# Verify the locked CUTLASS DSL backend", 1
    )[0]

    assert "sys.exit(0)" not in setup
    assert "--allow-unavailable" not in setup
    assert "Skipping performance test" not in setup
    assert "duplicate template name" not in checks
    assert "Skipping GPU performance test" not in checks
    assert "CUDA runtime is not available" in checks
    assert "CUDA runtime is unavailable for torch.compile verification" in checks
    assert "ERROR: torch.compile verification failed." in checks
    assert "ERROR: Transformer Engine capability verification failed." in setup
    assert "ERROR: Failed to regenerate arch_detection.cuh" in setup
    assert "ERROR: Blackwell feature verification failed." in setup
    assert re.search(
        r"ERROR: Blackwell feature verification failed\..*?exit 1",
        setup,
        re.DOTALL,
    )


def test_cutlass_installer_and_setup_require_immutable_provenance() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    installer = (CODE_ROOT / "core/scripts/install_cutlass.sh").read_text(encoding="utf-8")
    commit = "db1c288993354c88e551c40c19a8fb93a774a241"

    assert f'CUTLASS_REF="${{CUTLASS_REF:-{commit}}}"' in setup
    assert f'DEFAULT_COMMIT="{commit}"' in installer
    assert "^[0-9a-f]{40}$" in installer
    assert 'switch --detach "${REF}"' in installer
    assert " checkout " not in installer
    assert "rev-parse 'HEAD^{tree}'" in installer
    assert '"requested_commit"' in installer
    assert '"resolved_commit"' in installer
    assert '"tree"' in installer
    assert '"source_version"' in installer
    assert 'rm -rf "${DEST_DIR}/.git"' not in installer
    assert 'rm -rf "${backup_source}"' not in installer
    assert "rev-parse --show-toplevel" in installer
    assert "Previous CUTLASS source preserved at" in installer
    assert "status --porcelain --untracked-files=all" in installer
    assert "receipt != expected" in setup
    assert 'origin != os.environ["CUTLASS_REPO_URL"]' in setup
    assert "CUTLASS_RESOLVED_TREE" in setup


def _create_cutlass_fixture(source: Path) -> tuple[str, str]:
    version = source / "include/cutlass/version.h"
    version.parent.mkdir(parents=True)
    version.write_text(
        "#define CUTLASS_MAJOR 4\n" "#define CUTLASS_MINOR 5\n" "#define CUTLASS_PATCH 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=AISP Test",
            "-c",
            "user.email=aisp-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tree


def test_cutlass_installer_records_the_exact_local_git_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "installed"
    commit, tree = _create_cutlass_fixture(source)

    result = subprocess.run(
        [str(CODE_ROOT / "core/scripts/install_cutlass.sh")],
        env={
            **os.environ,
            "CUTLASS_REPO": f"file://{source}",
            "CUTLASS_REF": commit,
            "CUTLASS_SRC_DIR": str(destination),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((destination / ".git/aisp-source.json").read_text(encoding="utf-8"))
    assert receipt == {
        "release_tag": "v4.5.2",
        "repository": f"file://{source}",
        "requested_commit": commit,
        "resolved_commit": commit,
        "schema_version": 1,
        "source_version": "4.5.2",
        "tree": tree,
    }
    status = subprocess.run(
        ["git", "-C", str(destination), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_cutlass_installer_preserves_replaced_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "installed"
    commit, _ = _create_cutlass_fixture(source)
    destination.mkdir()
    marker = destination / "owner-data.txt"
    marker.write_text("keep me\n", encoding="utf-8")

    result = subprocess.run(
        [str(CODE_ROOT / "core/scripts/install_cutlass.sh")],
        env={
            **os.environ,
            "CUTLASS_REPO": f"file://{source}",
            "CUTLASS_REF": commit,
            "CUTLASS_SRC_DIR": str(destination),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    backups = list(tmp_path.glob("installed.backup.*"))
    assert len(backups) == 1
    assert (backups[0] / marker.name).read_text(encoding="utf-8") == "keep me\n"
    assert f"Previous CUTLASS source preserved at {backups[0]}" in result.stdout


def test_cutlass_installer_rejects_dirty_linked_worktree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "installed"
    commit, _ = _create_cutlass_fixture(source)
    subprocess.run(
        ["git", "-C", str(source), "worktree", "add", "--detach", str(destination), commit],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (destination / ".git").is_file()
    owner_file = destination / "owner-change.txt"
    owner_file.write_text("do not discard\n", encoding="utf-8")

    result = subprocess.run(
        [str(CODE_ROOT / "core/scripts/install_cutlass.sh")],
        env={
            **os.environ,
            "CUTLASS_REPO": f"file://{source}",
            "CUTLASS_REF": commit,
            "CUTLASS_SRC_DIR": str(destination),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "has local changes" in result.stderr
    assert owner_file.read_text(encoding="utf-8") == "do not discard\n"
    assert list(tmp_path.glob("installed.backup.*")) == []


def test_cutlass_installer_rejects_mutable_refs_before_clone(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(CODE_ROOT / "core/scripts/install_cutlass.sh")],
        env={
            **os.environ,
            "CUTLASS_REPO": str(tmp_path / "missing"),
            "CUTLASS_REF": "v4.5.2",
            "CUTLASS_SRC_DIR": str(tmp_path / "destination"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "full lowercase 40-character commit" in result.stderr
    assert not (tmp_path / "destination").exists()
