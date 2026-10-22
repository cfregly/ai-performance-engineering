"""Focused contracts for the pinned vLLM native source build."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import build_vllm_for_torch as builder

CODE_ROOT = Path(__file__).resolve().parents[1]
PINNED_DEEPGEMM_BUILD_HELPER = b'''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's TORCH_LIBRARY extension and copy vendored artifacts.

DeepGEMM now registers ops via TORCH_LIBRARY into ``deep_gemm._C_extension``
(abi3) and exposes the legacy API through ``deep_gemm/_C.py``. This driver
delegates to DeepGEMM's ``setup.py build_ext --inplace`` and copies the shim
plus extension into the cmake output directory.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR>
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
_pkg = src / "deep_gemm"
out.mkdir(parents=True, exist_ok=True)

if not (_pkg / "_C.py").is_file():
    sys.exit(
        f"DeepGEMM source at {src} is missing deep_gemm/_C.py; "
        "expected TORCH_LIBRARY migration layout"
    )

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.pop("DG_SKIP_CUDA_BUILD", None)

print(f"[build_deepgemm_C] building in {src} with {sys.executable}", flush=True)
# CMake decides when to rebuild; setuptools would only check python_api.cpp.
subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
    cwd=src,
    env=env,
)

shim = _pkg / "_C.py"
shutil.copy2(shim, out / shim.name)

so = _pkg / "_C_extension.abi3.so"
if not so.is_file():
    sys.exit(f"DeepGEMM build did not produce {so}")
for stale in out.glob("_C_extension*.so"):
    stale.unlink()
shutil.copy2(so, out / so.name)
print(f"[build_deepgemm_C] installed {so.name} -> {out}", flush=True)
'''


def _contract() -> dict[str, Any]:
    return json.loads((CODE_ROOT / "vllm_source_build.json").read_text(encoding="utf-8"))


def _write_compiler(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _fake_toolchain_environment(tmp_path: Path) -> dict[str, str]:
    toolchain_bin = tmp_path / "toolchain" / "bin"
    toolchain_bin.mkdir(parents=True)
    for name in _contract()["build"]["required_commands"]:
        _write_compiler(toolchain_bin / name, "exit 0\n")
    home = tmp_path / "home"
    rustup_home = tmp_path / "rustup"
    include = tmp_path / "sdk" / "include"
    library = tmp_path / "sdk" / "lib"
    for path in (home, rustup_home, include, library):
        path.mkdir(parents=True)
    return {
        "PATH": str(toolchain_bin),
        "HOME": str(home),
        "RUSTUP_HOME": str(rustup_home),
        "CPATH": str(include),
        "CPLUS_INCLUDE_PATH": str(include),
        "LIBRARY_PATH": str(library),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.rstrip("\n")


def _create_git_repository(path: Path, filename: str) -> dict[str, str]:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(path)],
        check=True,
        capture_output=True,
    )
    (path / filename).write_text(f"{path.name}\n", encoding="utf-8")
    _git(path, "add", filename)
    _git(
        path,
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    return {
        "repository": path.as_uri(),
        "commit": _git(path, "rev-parse", "HEAD"),
        "tree": _git(path, "rev-parse", "HEAD^{tree}"),
    }


def test_contract_requires_the_reviewed_elfutils_header_probe() -> None:
    contract = _contract()

    builder._validate_contract(contract)

    assert contract["build"]["required_cxx_header_probe"] == {
        "compiler": "g++",
        "language": "c++",
        "standard": "c++20",
        "headers": ["elfutils/libdwfl.h"],
        "debian_packages": ["libdw-dev", "libelf-dev"],
    }
    assert len(contract["source_dependencies"]) == 10
    assert sum(len(value["submodules"]) for value in contract["source_dependencies"].values()) == 11
    assert contract["build"]["maximum_jobs"] == 8
    assert contract["build"]["maximum_nvcc_threads"] == 2


@pytest.mark.parametrize(
    "required_environment",
    [
        {
            "CUDA_VISIBLE_DEVICES": "",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_NVML_BASED_CUDA_CHECK": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
    ],
)
def test_contract_rejects_changed_cpu_only_host_environment(
    required_environment: dict[str, str],
) -> None:
    contract = _contract()
    contract["build"]["required_host_environment"] = required_environment

    with pytest.raises(builder.BuildContractError, match="exact reviewed CPU-only"):
        builder._validate_contract(contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dependency-parent", "Source dependency name"),
        ("dependency-absolute", "Source dependency name"),
        ("root-file-parent", "upstream_files path"),
        ("dependency-file-absolute", "verified_files path"),
    ],
)
def test_contract_rejects_source_paths_before_output_creation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    contract = _contract()
    if mutation.startswith("dependency-") and "file" not in mutation:
        name, dependency = next(iter(contract["source_dependencies"].items()))
        del contract["source_dependencies"][name]
        replacement = "../escape" if mutation.endswith("parent") else "/absolute/escape"
        contract["source_dependencies"][replacement] = dependency
    elif mutation == "root-file-parent":
        contract["upstream_files"]["../escape"] = "0" * 64
    else:
        dependency = next(iter(contract["source_dependencies"].values()))
        dependency["verified_files"]["/absolute/escape"] = "0" * 64

    contract_path = tmp_path / "unsafe-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(builder.BuildContractError, match=message):
        builder.main(
            [
                "--contract",
                str(contract_path),
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_verified_repository_file_rejects_symlinks_and_resolved_escape(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    outside_file = outside / "payload"
    outside_file.write_text("outside\n", encoding="utf-8")
    digest = hashlib.sha256(outside_file.read_bytes()).hexdigest()

    leaf_symlink = source / "leaf"
    leaf_symlink.symlink_to(outside_file)
    with pytest.raises(builder.BuildContractError, match="missing or unsafe"):
        builder._verify_repository_file(source, "leaf", digest)

    directory_symlink = source / "linked"
    directory_symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(builder.BuildContractError, match="escapes its source root"):
        builder._verify_repository_file(source, "linked/payload", digest)


def test_header_probe_preprocesses_the_exact_source_with_the_supplied_environment(
    tmp_path: Path,
) -> None:
    compiler = tmp_path / "g++"
    arguments = tmp_path / "arguments.txt"
    source = tmp_path / "source.cc"
    include_path = tmp_path / "include"
    include_path.mkdir()
    _write_compiler(
        compiler,
        'printf "%s\\n" "$@" > "$AISP_TEST_ARGUMENTS"\n'
        'cat > "$AISP_TEST_SOURCE"\n'
        'test "$CPATH" = "$AISP_TEST_EXPECTED_CPATH"\n',
    )
    environment = {
        **os.environ,
        "AISP_TEST_ARGUMENTS": str(arguments),
        "AISP_TEST_SOURCE": str(source),
        "AISP_TEST_EXPECTED_CPATH": str(include_path),
        "CPATH": str(include_path),
    }

    receipt = builder._validate_required_cxx_headers(
        _contract(), {"g++": str(compiler)}, env=environment
    )

    expected_source = "#include <elfutils/libdwfl.h>\n"
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "-std=c++20",
        "-x",
        "c++",
        "-E",
        "-",
    ]
    assert source.read_text(encoding="utf-8") == expected_source
    assert receipt == {
        "command": [str(compiler), "-std=c++20", "-x", "c++", "-E", "-"],
        "headers": ["elfutils/libdwfl.h"],
        "source_sha256": hashlib.sha256(expected_source.encode()).hexdigest(),
        "debian_packages": ["libdw-dev", "libelf-dev"],
    }


def test_header_probe_failure_names_both_required_debian_packages(
    tmp_path: Path,
) -> None:
    compiler = tmp_path / "g++"
    _write_compiler(
        compiler,
        "cat >/dev/null\n"
        'printf "%s\\n" "fatal error: elfutils/libdwfl.h: No such file" >&2\n'
        "exit 42\n",
    )

    with pytest.raises(
        builder.BuildContractError,
        match=r"elfutils/libdwfl\.h\. Install libdw-dev, libelf-dev",
    ):
        builder._validate_required_cxx_headers(_contract(), {"g++": str(compiler)}, env=os.environ)


def test_build_environment_is_explicit_offline_and_scoped(tmp_path: Path) -> None:
    contract = _contract()
    base_environment = _fake_toolchain_environment(tmp_path)
    base_environment.update(
        {
            "CARGO_HOME": str(tmp_path / "shared-cargo"),
            "PIP_INDEX_URL": "https://example.invalid/simple",
            "PYTHONPATH": str(tmp_path / "ambient-python"),
            "UNRELATED_SECRET": "must-not-cross-the-boundary",
        }
    )
    output_dir = tmp_path / "output"

    environment, removed, command_paths = builder._create_build_environment(
        contract,
        output_dir,
        cuda_arch_list="10.0",
        max_jobs=8,
        nvcc_threads=2,
        base_environment=base_environment,
    )

    assert "CPATH" not in environment
    assert "CPLUS_INCLUDE_PATH" not in environment
    assert "LIBRARY_PATH" not in environment
    assert environment["MAX_JOBS"] == "8"
    assert environment["CARGO_BUILD_JOBS"] == "8"
    assert environment["CMAKE_BUILD_PARALLEL_LEVEL"] == "8"
    assert environment["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert environment["NVCC_THREADS"] == "2"
    assert environment["VLLM_USE_PRECOMPILED"] == "0"
    assert environment["VLLM_USE_PRECOMPILED_RUST"] == "0"
    assert environment["VLLM_VERSION_OVERRIDE"] == contract["package"]["version"]
    assert environment["VLLM_CUTLASS_SRC_DIR"].endswith("/output/source-dependencies/cutlass")
    assert environment["TRITON_KERNELS_SRC_DIR"].endswith("/python/triton_kernels/triton_kernels")
    assert environment["CARGO_HOME"] == str(output_dir / "cargo-home")
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["CARGO_NET_OFFLINE"] == "true"
    assert environment["CC"] == command_paths["gcc"]
    assert environment["CXX"] == command_paths["g++"]
    assert environment["CUDACXX"] == command_paths["nvcc"]
    assert "PIP_INDEX_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert "UNRELATED_SECRET" not in environment
    assert {
        "CARGO_HOME",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "PIP_INDEX_URL",
        "PYTHONPATH",
        "UNRELATED_SECRET",
    } <= set(removed)
    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in environment["CMAKE_ARGS"]
    assert environment["CMAKE_ARGS"].count("-DFETCHCONTENT_SOURCE_DIR_") == 10
    alternate_environment, _, _ = builder._create_build_environment(
        contract,
        tmp_path / "alternate-output",
        cuda_arch_list="10.3",
        max_jobs=4,
        nvcc_threads=1,
        base_environment=_fake_toolchain_environment(tmp_path / "alternate"),
    )
    assert alternate_environment["TORCH_CUDA_ARCH_LIST"] == "10.3"
    assert alternate_environment["MAX_JOBS"] == "4"
    assert alternate_environment["NVCC_THREADS"] == "1"


def test_command_resolution_preserves_rustup_proxy_names(tmp_path: Path) -> None:
    environment = _fake_toolchain_environment(tmp_path)
    cargo = Path(environment["PATH"]) / "cargo"
    rustup_proxy = Path(environment["PATH"]) / "rustup-proxy"
    cargo.unlink()
    _write_compiler(rustup_proxy, "exit 0\n")
    cargo.symlink_to(rustup_proxy.name)

    commands = builder._resolved_command_paths(_contract(), environment)

    assert commands["cargo"] == str(cargo.absolute())
    assert Path(commands["cargo"]).resolve() == rustup_proxy.resolve()


def test_nvcc_symlink_is_resolved_before_deriving_cuda_root(tmp_path: Path) -> None:
    environment = _fake_toolchain_environment(tmp_path)
    lexical_nvcc = Path(environment["PATH"]) / "nvcc"
    lexical_nvcc.unlink()
    resolved_nvcc = tmp_path / "cuda-13.0" / "bin" / "nvcc"
    resolved_nvcc.parent.mkdir(parents=True)
    _write_compiler(resolved_nvcc, "exit 0\n")
    lexical_nvcc.symlink_to(resolved_nvcc)

    child, _, commands = builder._create_build_environment(
        _contract(),
        tmp_path / "output",
        cuda_arch_list="10.0",
        max_jobs=8,
        nvcc_threads=2,
        base_environment=environment,
    )

    assert commands["nvcc"] == str(resolved_nvcc)
    assert child["CUDACXX"] == str(resolved_nvcc)
    assert child["CMAKE_CUDA_COMPILER"] == str(resolved_nvcc)
    assert child["CUDA_HOME"] == str(resolved_nvcc.parent.parent)


def test_toolchain_overlay_is_the_only_bound_include_and_library_root(
    tmp_path: Path,
) -> None:
    overlay_root = tmp_path / "elfutils-sysroot"
    include = overlay_root / "usr/include"
    library = overlay_root / "usr/lib/x86_64-linux-gnu"
    pkg_config = library / "pkgconfig"
    (include / "elfutils").mkdir(parents=True)
    pkg_config.mkdir(parents=True)
    (include / "elfutils/libdwfl.h").write_text("/* pinned */\n", encoding="utf-8")
    (library / "libdw.so").write_bytes(b"pinned libdw")
    (library / "libelf.so").write_bytes(b"pinned libelf")
    (pkg_config / "libdw.pc").write_text("Version: 0.190\n", encoding="utf-8")

    before = builder._inspect_toolchain_overlay(_contract(), overlay_root)
    child, removed, _ = builder._create_build_environment(
        _contract(),
        tmp_path / "output",
        cuda_arch_list="10.0",
        max_jobs=8,
        nvcc_threads=2,
        toolchain_overlay_root=overlay_root,
        base_environment=_fake_toolchain_environment(tmp_path / "environment"),
    )

    assert child["CPATH"] == str(include)
    assert child["CPLUS_INCLUDE_PATH"] == str(include)
    assert child["LIBRARY_PATH"] == str(library)
    assert child["PKG_CONFIG_PATH"] == str(pkg_config)
    assert "LD_LIBRARY_PATH" not in child
    assert {"CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"} <= set(removed)
    assert len(before["required_files"]) == 3
    (pkg_config / "libdw.pc").write_text("Version: changed\n", encoding="utf-8")
    after = builder._inspect_toolchain_overlay(_contract(), overlay_root)
    assert before["tree_manifest"] != after["tree_manifest"]


def test_cuda_initialization_check_is_read_only_and_fail_closed() -> None:
    calls = 0

    def is_initialized() -> bool:
        nonlocal calls
        calls += 1
        return False

    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_initialized=is_initialized))
    assert (
        builder._require_cuda_uninitialized(torch_module, checkpoint="at the fixture checkpoint")
        is False
    )
    assert calls == 1

    initialized = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    with pytest.raises(builder.BuildContractError, match="initialized CUDA at the checkpoint"):
        builder._require_cuda_uninitialized(initialized, checkpoint="at the checkpoint")


def test_build_command_wraps_the_pinned_frontend_in_network_isolation(
    tmp_path: Path,
) -> None:
    command, frontend = builder._create_build_command(
        _contract(), tmp_path / "wheel", {"unshare": "/usr/bin/unshare"}
    )

    assert command == ["/usr/bin/unshare", "--net", "--", *frontend]
    assert frontend == [
        builder.sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(tmp_path / "wheel"),
        ".",
    ]


@pytest.mark.parametrize(
    ("relative_path", "requirement"),
    [
        ("requirements/cuda.txt", "torch >= 2.14 ; python_version >= '3.12'"),
        ("requirements/cuda.txt", "TorchVision[image] != 0"),
        (
            "requirements/cuda.txt",
            "torchaudio @ https://example.invalid/torchaudio.whl#sha256=abc123",
        ),
        (
            "pyproject.toml",
            "\"TorchCodec[video] ~= 0.17 ; platform_system == 'Linux'\", # retained form",
        ),
    ],
)
def test_post_transform_check_detects_canonical_torch_requirement_forms(
    tmp_path: Path, relative_path: str, requirement: str
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(requirement + "\n", encoding="utf-8")

    assert builder._torch_requirement_locations(tmp_path, [path]) == [
        f"{relative_path}:1:{requirement}"
    ]


def test_supplemental_transform_removes_only_canonical_torch_requirements(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements" / "cuda.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_bytes(
        b"# Common dependencies\r\n"
        b"torchcodec >= 0.14\r\n"
        b"TorchVision[image] != 0  # target with extras\r\n"
        b"torchdata==0.11.0  # unrelated Torch project\r\n"
        b"mytorch==1.0\r\n"
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(
        b"requires = [\n"
        b"    \"torch >= 2.14 ; python_version >= '3.12'\",\n"
        b"    'torchaudio ~= 2.11 ; platform_system == \"Linux\"', # target\n"
        b'    "packaging>=24.2", # unrelated\n'
        b"]\n"
    )
    candidates = builder._requirement_candidates(tmp_path)

    before = builder._torch_requirement_locations(tmp_path, candidates)
    removed = builder._strip_torch_requirements(tmp_path, candidates)

    assert len(before) == 4
    assert {entry["requirement"].strip() for entry in removed} == {
        "torchcodec >= 0.14",
        "TorchVision[image] != 0  # target with extras",
        "\"torch >= 2.14 ; python_version >= '3.12'\",",
        "'torchaudio ~= 2.11 ; platform_system == \"Linux\"', # target",
    }
    assert builder._torch_requirement_locations(tmp_path, candidates) == []
    assert requirements.read_bytes() == (
        b"# Common dependencies\r\ntorchdata==0.11.0  # unrelated Torch project\r\nmytorch==1.0\r\n"
    )
    assert pyproject.read_bytes() == (b'requires = [\n    "packaging>=24.2", # unrelated\n]\n')


def test_deepgemm_source_import_binding_runs_local_setup_under_safe_path(
    tmp_path: Path,
) -> None:
    vllm_source = tmp_path / "vllm"
    helper = vllm_source / builder.DEEPGEMM_BUILD_HELPER_PATH
    helper.parent.mkdir(parents=True)
    helper.write_bytes(PINNED_DEEPGEMM_BUILD_HELPER)
    upstream_sha256 = hashlib.sha256(helper.read_bytes()).hexdigest()
    assert upstream_sha256 == builder.DEEPGEMM_BUILD_HELPER_SHA256

    deepgemm_source = tmp_path / "deepgemm"
    package = deepgemm_source / "deep_gemm"
    scripts = deepgemm_source / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "_C.py").write_text("# fixture shim\n", encoding="utf-8")
    (scripts / "generate_pyi.py").write_text(
        "def generate_pyi_file():\n    return 'verified-local-source'\n",
        encoding="utf-8",
    )
    (deepgemm_source / "setup.py").write_text(
        """\
import os
from pathlib import Path
from scripts.generate_pyi import generate_pyi_file

source_root = Path(__file__).resolve().parent
assert os.environ["PYTHONSAFEPATH"] == "1"
assert os.environ["PYTHONPATH"] == str(source_root)
(source_root / "setup-import.txt").write_text(generate_pyi_file(), encoding="utf-8")
(source_root / "deep_gemm" / "_C_extension.abi3.so").write_bytes(b"fixture extension")
""",
        encoding="utf-8",
    )
    baseline_environment = os.environ.copy()
    baseline_environment.pop("PYTHONPATH", None)
    baseline_environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    baseline = subprocess.run(
        [
            sys.executable,
            str(helper),
            str(deepgemm_source),
            str(tmp_path / "baseline-output"),
        ],
        cwd=tmp_path,
        env=baseline_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert baseline.returncode != 0
    assert "ModuleNotFoundError: No module named 'scripts" in baseline.stderr

    binding = builder._apply_deepgemm_source_import_binding(vllm_source)
    ambient = tmp_path / "ambient"
    (ambient / "scripts").mkdir(parents=True)
    (ambient / "scripts" / "generate_pyi.py").write_text(
        "def generate_pyi_file():\n    return 'ambient-fallback'\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(ambient),
        }
    )
    output = tmp_path / "deepgemm-output"

    subprocess.run(
        [sys.executable, str(helper), str(deepgemm_source), str(output)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (deepgemm_source / "setup-import.txt").read_text(encoding="utf-8") == (
        "verified-local-source"
    )
    assert (output / "_C.py").read_text(encoding="utf-8") == "# fixture shim\n"
    assert (output / "_C_extension.abi3.so").read_bytes() == b"fixture extension"
    assert binding == {
        "path": "tools/build_deepgemm_C.py",
        "upstream_sha256": upstream_sha256,
        "transformed_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "environment_variable": "PYTHONPATH",
        "value": "resolved DeepGEMM source root",
        "scope": "DeepGEMM setup.py subprocess",
        "python_safe_path_preserved": True,
        "reason": (
            "Pinned DeepGEMM setup.py imports scripts.generate_pyi while "
            "PYTHONSAFEPATH omits its working directory from sys.path."
        ),
        "retirement_condition": (
            "The pinned vLLM helper or DeepGEMM setup binds the source root without "
            "an ambient import fallback."
        ),
    }


def test_deepgemm_source_import_binding_rejects_unreviewed_helper(tmp_path: Path) -> None:
    helper = tmp_path / builder.DEEPGEMM_BUILD_HELPER_PATH
    helper.parent.mkdir(parents=True)
    helper.write_bytes(builder.DEEPGEMM_BUILD_HELPER_ENVIRONMENT + b"# changed upstream\n")
    original = helper.read_bytes()

    with pytest.raises(builder.BuildContractError, match="hash mismatch"):
        builder._apply_deepgemm_source_import_binding(tmp_path)

    assert helper.read_bytes() == original


@pytest.mark.parametrize(
    "name",
    [
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CMAKE_ARGS",
        "CMAKE_PREFIX_PATH",
        "CMAKE_CUDA_COMPILER",
        "CUDACXX",
        "CUDA_HOME",
        "RUSTC",
        "RUSTFLAGS",
        "RUSTUP_TOOLCHAIN",
    ],
)
def test_build_environment_rejects_conflicting_toolchain_selectors(
    tmp_path: Path, name: str
) -> None:
    base_environment = _fake_toolchain_environment(tmp_path)
    base_environment[name] = "/hostile-selector"

    with pytest.raises(builder.BuildContractError, match=name):
        builder._create_build_environment(
            _contract(),
            tmp_path / "output",
            cuda_arch_list="10.0",
            max_jobs=8,
            nvcc_threads=2,
            base_environment=base_environment,
        )


def test_build_environment_removes_unvalidated_search_roots(tmp_path: Path) -> None:
    base_environment = _fake_toolchain_environment(tmp_path)
    base_environment["LD_LIBRARY_PATH"] = "relative-or-missing"

    child, removed, _ = builder._create_build_environment(
        _contract(),
        tmp_path / "output",
        cuda_arch_list="10.0",
        max_jobs=8,
        nvcc_threads=2,
        base_environment=base_environment,
    )

    assert "LD_LIBRARY_PATH" not in child
    assert "LD_LIBRARY_PATH" in removed


@pytest.mark.parametrize(
    ("max_jobs", "nvcc_threads", "message"),
    [
        (0, 1, "--max-jobs must be between 1 and 8"),
        (9, 1, "--max-jobs must be between 1 and 8"),
        (1, 0, "--nvcc-threads must be between 1 and 2"),
        (1, 3, "--nvcc-threads must be between 1 and 2"),
    ],
)
def test_resource_caps_fail_before_output_mutation(
    tmp_path: Path, max_jobs: int, nvcc_threads: int, message: str
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(builder.BuildContractError, match=message):
        builder._build(
            contract=_contract(),
            contract_path=CODE_ROOT / "vllm_source_build.json",
            output_dir=output,
            cuda_arch_list="10.0",
            max_jobs=max_jobs,
            nvcc_threads=nvcc_threads,
        )

    assert not output.exists()


def test_source_materialization_verifies_nested_parent_gitlinks_and_cleanliness(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested_identity = _create_git_repository(nested, "nested.txt")
    parent = tmp_path / "parent"
    _create_git_repository(parent, "parent.txt")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        nested_identity["repository"],
        "vendor/nested",
    )
    _git(parent, "add", ".gitmodules", "vendor/nested")
    _git(
        parent,
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "add nested",
    )
    parent_identity = {
        "repository": parent.as_uri(),
        "commit": _git(parent, "rev-parse", "HEAD"),
        "tree": _git(parent, "rev-parse", "HEAD^{tree}"),
    }
    root = tmp_path / "root"
    _create_git_repository(root, "root.txt")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        parent_identity["repository"],
        "deps/parent",
    )
    _git(root, "add", ".gitmodules", "deps/parent")
    _git(
        root,
        "-c",
        "user.name=Contract Test",
        "-c",
        "user.email=contract@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "add parent",
    )
    source = {
        "repository": root.as_uri(),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "verified_files": {},
        "submodules": [
            {"path": "deps/parent", **parent_identity},
            {"path": "deps/parent/vendor/nested", **nested_identity},
        ],
    }
    environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    destination = tmp_path / "materialized"

    receipt = builder._checkout_repository(
        source,
        destination,
        label="fixture",
        env=environment,
    )

    assert [value["path"] for value in receipt["submodules"]] == [
        "deps/parent",
        "deps/parent/vendor/nested",
    ]
    assert receipt["submodules"][1]["parent_gitlink_commit"] == nested_identity["commit"]
    with pytest.raises(builder.BuildContractError, match="Git submodule declarations differ"):
        builder._verify_clean_repository(
            {**source, "submodules": source["submodules"][:1]},
            destination,
            label="fixture",
            env=environment,
        )
    (destination / "deps/parent/vendor/nested/nested.txt").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(builder.BuildContractError, match="source tree is dirty"):
        builder._verify_clean_repository(
            source,
            destination,
            label="fixture",
            env=environment,
        )


def test_cargo_dependencies_are_prefetched_then_sealed_for_offline_build(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    rust = source / "rust"
    rust.mkdir(parents=True)
    manifest = rust / "Cargo.toml"
    manifest.write_text('[workspace]\nresolver = "2"\n', encoding="utf-8")
    locked_sources = [
        "git+https://example.invalid/a?rev=1#1111111111111111111111111111111111111111",
        "git+https://example.invalid/b?rev=2#2222222222222222222222222222222222222222",
        "git+https://example.invalid/c?rev=3#3333333333333333333333333333333333333333",
    ]
    lock = rust / "Cargo.lock"
    lock.write_text(
        "".join(
            f'[[package]]\nname = "fixture{index}"\nversion = "0.0.0"\nsource = "{value}"\n\n'
            for index, value in enumerate(locked_sources)
        ),
        encoding="utf-8",
    )
    cargo = tmp_path / "cargo"
    log = tmp_path / "cargo.log"
    _write_compiler(
        cargo,
        'printf "%s|%s\\n" "$CARGO_NET_OFFLINE" "$*" > "$AISP_CARGO_LOG"\n'
        'mkdir -p "$CARGO_HOME/registry/cache"\n'
        'printf "%s\\n" pinned > "$CARGO_HOME/registry/cache/materialized"\n'
        'printf "%s\\n" tracking > "$CARGO_HOME/.global-cache"\n'
        ': > "$CARGO_HOME/.package-cache"\n'
        ': > "$CARGO_HOME/.package-cache-mutate"\n',
    )
    contract = _contract()
    contract["build"]["cargo_materialization"] = {
        "manifest_path": "rust/Cargo.toml",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "lock_path": "rust/Cargo.lock",
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "lock_size": lock.stat().st_size,
        "cargo_home_relative_to_output": "cargo-home",
        "prefetch_arguments": [
            "fetch",
            "--locked",
            "--manifest-path",
            "rust/Cargo.toml",
        ],
        "locked_git_sources": locked_sources,
        "materialize_before_network_isolation": True,
        "verify_lock_unchanged_after_build": True,
        "record_tree_manifest": True,
    }
    cargo_home = tmp_path / "cargo-home"
    cargo_home.mkdir()
    environment = {
        "PATH": os.environ["PATH"],
        "CARGO_HOME": str(cargo_home),
        "CARGO_NET_OFFLINE": "true",
        "AISP_CARGO_LOG": str(log),
    }

    receipt = builder._materialize_cargo_dependencies(
        contract,
        source,
        environment,
        {"cargo": str(cargo)},
    )
    sealed = builder._verify_cargo_materialization_unchanged(
        contract,
        source,
        environment,
        receipt,
    )

    assert log.read_text(encoding="utf-8").startswith("false|fetch --locked")
    assert sealed["unchanged_after_build"] is True
    assert sealed["locked_git_sources"] == sorted(locked_sources)
    assert sealed["cargo_home_tree_manifest"]["excluded_mutable_control_paths"] == [
        ".global-cache",
        ".package-cache",
        ".package-cache-mutate",
    ]
    assert {value["path"] for value in sealed["cargo_home_control_files_after_fetch"]} == {
        ".global-cache",
        ".package-cache",
        ".package-cache-mutate",
    }

    (cargo_home / ".global-cache").write_text("updated last-use timestamps\n", encoding="utf-8")
    control_update = builder._verify_cargo_materialization_unchanged(
        contract,
        source,
        environment,
        receipt,
    )
    assert control_update["unchanged_after_build"] is True
    assert (
        control_update["cargo_home_control_files_after_fetch"]
        != control_update["cargo_home_control_files_after_build"]
    )

    (cargo_home / "registry/cache/materialized").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        builder.BuildContractError,
        match="Cargo dependency materialization changed",
    ):
        builder._verify_cargo_materialization_unchanged(
            contract,
            source,
            environment,
            receipt,
        )


def test_cmake_cache_must_bind_every_local_source_and_disconnected_control(
    tmp_path: Path,
) -> None:
    contract = _contract()
    source = tmp_path / "source"
    cache = source / "build" / "CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    environment: dict[str, str] = {}
    lines = [
        "FETCHCONTENT_FULLY_DISCONNECTED:UNINITIALIZED=ON",
        "FETCHCONTENT_UPDATES_DISCONNECTED:UNINITIALIZED=ON",
    ]
    for name, dependency in contract["source_dependencies"].items():
        environment_variable = dependency["environment_variable"]
        source_path = str(tmp_path / "dependencies" / name)
        environment[environment_variable] = source_path
        fetchcontent_name = dependency["fetchcontent_name"].upper().replace("-", "_")
        lines.append(f"FETCHCONTENT_SOURCE_DIR_{fetchcontent_name}:PATH={source_path}")
    cache.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = builder._inspect_cmake_cache_bindings(contract, source, environment)

    assert len(receipt["bindings"]) == 12
    assert receipt["caches"][0]["path"] == "build/CMakeCache.txt"
    cache.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(builder.BuildContractError, match="did not bind"):
        builder._inspect_cmake_cache_bindings(contract, source, environment)
