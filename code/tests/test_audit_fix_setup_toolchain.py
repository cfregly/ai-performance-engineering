"""Regression tests for the build and profiling toolchain installed by setup.sh.

Ubuntu's apt CMake (3.22 on 22.04, 3.28 on 24.04) cannot configure the
first-party CUTLASS lab, which requires CMake 3.31.8 or newer. The CUDA 13.0
toolkit also installs Nsight 2025.3.x, while ncu_torchrun_capture.py accepts only
its qualified NCU version. These tests read the real setup.sh,
requirements_latest.txt, workflows, docs and tracked CMakeLists.txt files. The
shell checks run the real setup.sh functions with only the cmake, nsys and ncu
binaries replaced, so no privileged setup step runs.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
SETUP_SCRIPT = CODE_ROOT / "setup.sh"
REQUIREMENTS = CODE_ROOT / "requirements_latest.txt"
PYTORCH_STACK_LOCK = CODE_ROOT / "requirements_torch2141_linux_x86_64.lock"
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
ENVIRONMENT_DOC = REPOSITORY_ROOT / "docs" / "environment.md"
NCU_CAPTURE = CODE_ROOT / "core" / "profiling" / "ncu_torchrun_capture.py"
CUTLASS_LAB_CMAKE = "code/labs/custom_vs_cublas/cutlass_gemm/CMakeLists.txt"


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _first_party_cmake_floor() -> tuple[str, str]:
    listing = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "-z", "--", ":(glob)**/CMakeLists.txt"],
        capture_output=True,
        text=True,
        check=True,
    )
    declaration = re.compile(
        r"^\s*cmake_minimum_required\s*\(\s*VERSION\s+([0-9]+(?:\.[0-9]+)*)",
        re.IGNORECASE | re.MULTILINE,
    )
    declarations = [
        (match.group(1), relative)
        for relative in filter(None, listing.stdout.split("\0"))
        if "third_party" not in PurePosixPath(relative).parts
        for match in declaration.finditer(
            (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        )
    ]
    assert CUTLASS_LAB_CMAKE in {relative for _, relative in declarations}
    return max(declarations, key=lambda item: _version_key(item[0]))


def _requirements_pin(package: str) -> str:
    pins = re.findall(
        rf"^{re.escape(package)}==([^\s#]+)",
        REQUIREMENTS.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert len(pins) == 1, f"requirements_latest.txt must pin {package} exactly once"
    return pins[0]


def _lock_pin(package: str) -> str:
    pins = re.findall(
        rf"^{re.escape(package)}==([^\s\\]+)",
        PYTORCH_STACK_LOCK.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert len(pins) == 1, f"PyTorch stack lock must pin {package} exactly once"
    return pins[0]


def _setup_text() -> str:
    return SETUP_SCRIPT.read_text(encoding="utf-8")


def _setup_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", _setup_text(), re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"setup.sh must define {name}()"
    return match.group(0)


def _run_cmake_check(tmp_path: Path, reported_version: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cmake = bin_dir / "cmake"
    cmake.write_text(
        "#!/bin/sh\n"
        f"printf 'cmake version %s\\n\\nCMake suite maintained and supported by Kitware.\\n' "
        f"'{reported_version}'\n",
        encoding="utf-8",
    )
    cmake.chmod(0o755)
    script = (
        "set -e\n"
        f'REPOSITORY_ROOT="{REPOSITORY_ROOT}"\n'
        f"{_setup_function('require_cmake_toolchain')}"
        f'require_cmake_toolchain "{_requirements_pin("cmake")}"\n'
    )
    return subprocess.run(
        ["/bin/bash", "-s"],
        input=script,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_requirements_pin_cmake_at_or_above_first_party_floor() -> None:
    floor, source = _first_party_cmake_floor()
    cmake_pin = _requirements_pin("cmake")
    assert _version_key(cmake_pin)[:3] >= _version_key(floor), (
        f"cmake=={cmake_pin} is older than cmake_minimum_required {floor} in {source}"
    )
    _requirements_pin("ninja")


def test_setup_reads_cmake_and_ninja_from_requirements_pins(tmp_path: Path) -> None:
    script = (
        "set -e\n"
        f'REQUIREMENTS_FILE="{REQUIREMENTS}"\n'
        f"{_setup_function('requirements_pin')}"
        "requirements_pin cmake\n"
        "requirements_pin ninja\n"
    )
    result = subprocess.run(
        ["/bin/bash", "-s"], input=script, capture_output=True, text=True, check=True
    )
    assert result.stdout.splitlines() == [_requirements_pin("cmake"), _requirements_pin("ninja")]

    missing = tmp_path / "requirements.txt"
    missing.write_text("ninja==1.13.2\n", encoding="utf-8")
    result = subprocess.run(
        ["/bin/bash", "-s"],
        input=script.replace(str(REQUIREMENTS), str(missing)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must pin cmake with an exact == version" in result.stderr

    setup = _setup_text()
    assert 'REQ_CMAKE_VERSION="$(requirements_pin cmake)"' in setup
    assert 'REQ_NINJA_VERSION="$(requirements_pin ninja)"' in setup
    assert _lock_pin("cmake") == _requirements_pin("cmake")
    assert _lock_pin("ninja") == _requirements_pin("ninja")

    lock_check_start = setup.index('python3 - "${PYTORCH_STACK_LOCK}"')
    lock_check_end = setup.index("\nPYLOCKPINS", lock_check_start)
    lock_check = setup[lock_check_start:lock_check_end]
    assert '"cmake=${REQ_CMAKE_VERSION}"' in lock_check
    assert '"ninja=${REQ_NINJA_VERSION}"' in lock_check

    install_start = setup.index("# Install the preflight-validated Linux/Python 3.12 graph.")
    install_end = setup.index("# The graph includes the exact CUDA 13 torch family.")
    locked_install = setup[install_start:install_end]
    assert "pip_install --require-hashes --no-deps" in locked_install
    assert '-r "${PYTORCH_STACK_LOCK}"' in locked_install
    assert not re.search(r"\b(?:cmake|ninja)==[0-9]", setup), "setup.sh must not hardcode these pins"


def test_setup_does_not_install_distribution_cmake() -> None:
    apt_installs = re.findall(
        r"\bapt(?:-get)? install\b.*?(?<!\\)$", _setup_text(), re.MULTILINE | re.DOTALL
    )
    assert apt_installs
    for command in apt_installs:
        assert not re.search(r"(?<![\w-])cmake(?![\w-])", command), command


def test_pip_cmake_is_first_on_path_and_checked_before_cmake_builds() -> None:
    setup = _setup_text()
    install = setup.index("pip_install --require-hashes --no-deps")
    path_export = setup.index('export PATH="${CMAKE_PIP_BIN_DIR}:${PATH}"')
    check = setup.index('require_cmake_toolchain "${REQ_CMAKE_VERSION}"')
    assert '-r "${PYTORCH_STACK_LOCK}"' in setup[install:path_export]
    assert "import cmake; print(cmake.CMAKE_BIN_DIR)" in setup[install:path_export]
    assert install < path_export < check

    cmake_consumers = (
        '"${PROJECT_ROOT}/core/scripts/install_cutlass.sh"',
        'python3 "${PROJECT_ROOT}/scripts/build_torch_extensions.py"',
    )
    for consumer in cmake_consumers:
        assert check < setup.index(consumer, check), consumer


@pytest.mark.parametrize("distribution_version", ["3.22.1", "3.28.3"])
def test_cmake_check_rejects_distribution_cmake(tmp_path: Path, distribution_version: str) -> None:
    floor, source = _first_party_cmake_floor()
    result = _run_cmake_check(tmp_path, distribution_version)
    assert result.returncode != 0
    assert f"is CMake {distribution_version}" in result.stderr
    assert f"{source} requires cmake_minimum_required {floor}" in result.stderr


def test_cmake_check_rejects_cmake_other_than_the_pin(tmp_path: Path) -> None:
    floor, _ = _first_party_cmake_floor()
    pin = _version_key(_requirements_pin("cmake"))
    other = ".".join(str(part) for part in (*pin[:2], pin[2] + 1))
    assert _version_key(other) >= _version_key(floor)
    result = _run_cmake_check(tmp_path, other)
    assert result.returncode != 0
    assert f"not the cmake=={_requirements_pin('cmake')} pin" in result.stderr


def test_cmake_check_accepts_the_pinned_cmake(tmp_path: Path) -> None:
    floor, source = _first_party_cmake_floor()
    pin = _requirements_pin("cmake")
    result = _run_cmake_check(tmp_path, pin)
    assert result.returncode == 0, result.stderr
    assert f"CMake {pin}" in result.stdout
    assert f"cmake_minimum_required {floor} ({source})" in result.stdout


def test_ci_cmake_and_ninja_pins_match_requirements() -> None:
    ci_pins: dict[str, list[tuple[str, str]]] = {"cmake": [], "ninja": []}
    for workflow in sorted(WORKFLOW_ROOT.glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        for package in ci_pins:
            for version in re.findall(rf"(?<![\w-]){package}==([^\s\\\"']+)", text):
                ci_pins[package].append((workflow.name, version))

    assert ("benchmark-validation.yml", _requirements_pin("cmake")) in ci_pins["cmake"]
    for package, pins in ci_pins.items():
        for workflow_name, version in pins:
            assert version == _requirements_pin(package), (
                f"{workflow_name} pins {package}=={version}, but requirements_latest.txt "
                f"pins {package}=={_requirements_pin(package)}"
            )


def _setup_assignment(name: str) -> str:
    values = re.findall(rf'^{name}="([^"]+)"$', _setup_text(), re.MULTILINE)
    assert len(values) == 1, f"setup.sh must assign {name} exactly once"
    return values[0]


def test_setup_installs_the_ncu_version_the_capture_tool_accepts() -> None:
    supported = re.search(
        r'^SUPPORTED_NCU_VERSION = "([^"]+)"$',
        NCU_CAPTURE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert supported is not None
    assert _setup_assignment("NCU_VERSION") == supported.group(1)


def test_setup_and_docs_report_the_installed_nsight_pins() -> None:
    setup = _setup_text()
    doc = ENVIRONMENT_DOC.read_text(encoding="utf-8")
    for tool, variable in (("Nsight Systems", "NSYS_VERSION"), ("Nsight Compute", "NCU_VERSION")):
        version = _setup_assignment(variable)
        assert re.search(rf"^#.*NVIDIA {tool} {re.escape(version)}\b", setup, re.MULTILINE)
        assert f'echo "  • NVIDIA {tool} {version}"' in setup
        assert re.search(rf"^\| {tool} \| {re.escape(version)} \|", doc, re.MULTILINE)
    assert re.search(r"^\| CMake \| " + re.escape(_requirements_pin("cmake")) + r" \|", doc, re.MULTILINE)
    assert re.search(r"^\| Ninja \| " + re.escape(_requirements_pin("ninja")) + r" \|", doc, re.MULTILINE)


def test_setup_installs_the_exact_signed_nsight_systems_cli_package() -> None:
    setup = _setup_text()
    assert _setup_assignment("NSYS_VERSION") == "2026.5.1"
    assert _setup_assignment("NSYS_PACKAGE_VERSION") == "2026.5.1.161-265138896106v0"
    assert 'NSYS_PACKAGE="nsight-systems-cli-${NSYS_VERSION}"' in setup
    assert 'apt install -y --upgrade "${NSYS_PACKAGE}=${NSYS_PACKAGE_VERSION}"' in setup
    assert 'apt install -y --upgrade "nsight-systems-${NSYS_VERSION}"' not in setup
    assert (
        'NSYS_BIN="/opt/nvidia/nsight-systems-cli/${NSYS_VERSION}/target-linux-x64/nsys"' in setup
    )
    source_install = setup.index('install -m 0644 "${TEMP_DIR}/nvidia-devtools.list"')
    metadata_refresh = setup.index("apt update -qq", source_install)
    exact_install = setup.index(
        'apt install -y --upgrade "${NSYS_PACKAGE}=${NSYS_PACKAGE_VERSION}"',
        metadata_refresh,
    )
    installed_version_check = setup.index('dpkg-query -W -f=\'${Version}\' "${NSYS_PACKAGE}"', exact_install)
    path_gate = setup.index('if [[ ! -x "${NSYS_BIN}" ]]', installed_version_check)
    assert source_install < metadata_refresh < exact_install < installed_version_check < path_gate
    assert "nsight-systems-cli-2026.5.1=2026.5.1.161-265138896106v0" in ENVIRONMENT_DOC.read_text(
        encoding="utf-8"
    )


def test_setup_scopes_and_verifies_the_devtools_repository_key() -> None:
    setup = _setup_text()
    assert (
        _setup_assignment("DEVTOOLS_KEY_SHA256")
        == "47217c49dcb9e47a8728b354450f694c9898cd4a126173044a69b1e9ac0fba96"
    )
    assert (
        _setup_assignment("DEVTOOLS_KEYRING_SHA256")
        == "0d4c5cbe167a020fc2978ea28eaea0e797f89fdee696488e93bf446dd868486f"
    )
    assert (
        _setup_assignment("DEVTOOLS_KEY_FINGERPRINT") == "AE09FE4BBD223A84B2CCFCE3F60F4B3D7FA2AF80"
    )
    assert (
        'DEVTOOLS_REPO_URL="https://developer.download.nvidia.com/devtools/repos/'
        '${CUDA_REPO_DISTRO}/amd64"' in setup
    )
    assert "signed-by=%s" in setup
    assert 'DEVTOOLS_KEYRING="/usr/share/keyrings/nvidia-devtools-keyring.gpg"' in setup
    assert "gpg --no-options --homedir" in setup
    assert "sha256sum --check --status" in setup


def test_pinned_nsight_directories_precede_cuda_bin_on_runtime_paths() -> None:
    setup = _setup_text()
    assert 'NSIGHT_BIN_PATH="$(dirname "${NSYS_BIN}"):$(dirname "${NCU_BIN}")"' in setup
    cuda_path_exports = re.findall(
        r"^export PATH=.*CUDA_HOME(?:_DIR)?\}?/bin.*$", setup, re.MULTILINE
    )
    assert cuda_path_exports == [
        'export PATH="${AISP_VENV_DIR}/bin:${NSIGHT_BIN_PATH}:${CUDA_HOME_DIR}/bin:\\$PATH"',
        'export PATH="${NSIGHT_BIN_PATH}:$CUDA_HOME/bin:$PATH"',
    ]
    for line in cuda_path_exports:
        nsight = line.find("${NSIGHT_BIN_PATH}:")
        assert 0 <= nsight < line.find("CUDA_HOME"), line

    profile_export = setup.index(cuda_path_exports[0])
    script_export = setup.index('export PATH="${NSIGHT_BIN_PATH}:$CUDA_HOME/bin:$PATH"')
    check = setup.index("\nrequire_pinned_nsight_tools\n")
    assert setup.index('NSIGHT_BIN_PATH="') < profile_export < script_export < check
    assert "/etc/environment" not in setup
    assert '"/opt/nvidia/nsight-compute/"*/bin' not in setup
    assert '"/opt/nvidia/nsight-systems/"*/bin' not in setup


def _run_nsight_check(
    tmp_path: Path,
    nsys_version: str,
    ncu_version: str,
    *,
    expected_nsys: Path | None = None,
    expected_ncu: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nsys").write_text(
        f"#!/bin/sh\nprintf 'NVIDIA Nsight Systems version {nsys_version}.161-265138896106v0\\n'\n",
        encoding="utf-8",
    )
    (bin_dir / "ncu").write_text(
        "#!/bin/sh\n"
        "printf 'NVIDIA (R) Nsight Compute Command Line Profiler\\n'\n"
        "printf 'Copyright (c) 2018-2026 NVIDIA Corporation\\n'\n"
        f"printf 'Version {ncu_version}.0 (build 37000000) (public-release)\\n'\n",
        encoding="utf-8",
    )
    for tool in ("nsys", "ncu"):
        (bin_dir / tool).chmod(0o755)
    script = (
        "set -e\n"
        f'NSYS_VERSION="{_setup_assignment("NSYS_VERSION")}"\n'
        f'NCU_VERSION="{_setup_assignment("NCU_VERSION")}"\n'
        f'NSYS_BIN="{expected_nsys or (bin_dir / "nsys")}"\n'
        f'NCU_BIN="{expected_ncu or (bin_dir / "ncu")}"\n'
        f"{_setup_function('require_pinned_nsight_tools')}"
        "require_pinned_nsight_tools\n"
    )
    return subprocess.run(
        ["/bin/bash", "-s"],
        input=script,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_nsight_check_accepts_the_pinned_tools(tmp_path: Path) -> None:
    nsys, ncu = _setup_assignment("NSYS_VERSION"), _setup_assignment("NCU_VERSION")
    result = _run_nsight_check(tmp_path, nsys, ncu)
    assert result.returncode == 0, result.stderr
    assert f"Nsight Systems {nsys}" in result.stdout
    assert f"Nsight Compute {ncu}" in result.stdout


def test_nsight_check_rejects_a_different_binary_path(tmp_path: Path) -> None:
    result = _run_nsight_check(
        tmp_path,
        _setup_assignment("NSYS_VERSION"),
        _setup_assignment("NCU_VERSION"),
        expected_nsys=tmp_path / "different" / "nsys",
    )
    assert result.returncode != 0
    assert "does not resolve to the pinned Nsight Systems binary" in result.stderr


@pytest.mark.parametrize(
    ("nsys_version", "ncu_version", "message"),
    (
        ("2025.3.2", None, "not the pinned Nsight Systems"),
        (None, "2025.3.1", "not the pinned Nsight Compute"),
    ),
)
def test_nsight_check_rejects_the_cuda_toolkit_copies(
    tmp_path: Path, nsys_version: str | None, ncu_version: str | None, message: str
) -> None:
    result = _run_nsight_check(
        tmp_path,
        nsys_version or _setup_assignment("NSYS_VERSION"),
        ncu_version or _setup_assignment("NCU_VERSION"),
    )
    assert result.returncode != 0
    assert message in result.stderr
