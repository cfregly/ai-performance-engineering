from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core.harness import serving_stack

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_get_serving_stack_pins_reads_the_cuda13_vllm_manifest() -> None:
    pins = serving_stack.get_serving_stack_pins()

    assert pins.torch_version == "2.14.1+cu130"
    assert pins.vllm_version == "0.33.0.dev0+cu130.torch2141"
    assert pins.flashinfer_version == "0.7.2"
    assert pins.vllm_pip_spec == "vllm==0.33.0.dev0+cu130.torch2141"


def test_get_serving_stack_pins_keeps_vllm_separate_from_base_requirements(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements_latest.txt"
    requirements.write_text(
        "torch==2.14.1+cu130\n"
        "vllm==0.15.0\n"
        "flashinfer-python==0.7.2\n",
        encoding="utf-8",
    )
    vllm_pin = tmp_path / "vllm_no_deps.pin"
    vllm_pin.write_text(
        "vllm==0.33.0.dev0+cu130.torch2141\n", encoding="utf-8"
    )

    pins = serving_stack.get_serving_stack_pins(
        requirements_path=requirements,
        vllm_pin_path=vllm_pin,
    )

    assert pins.vllm_version == "0.33.0.dev0+cu130.torch2141"


def test_get_serving_stack_pins_requires_the_separate_vllm_manifest(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements_latest.txt"
    requirements.write_text(
        "torch==2.14.1+cu130\nflashinfer-python==0.7.2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ABI-bound vLLM pin manifest is missing"):
        serving_stack.get_serving_stack_pins(
            requirements_path=requirements,
            vllm_pin_path=tmp_path / "missing-vllm.pin",
        )


def test_site_package_roots_warn_on_discovery_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr(serving_stack.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(
        serving_stack.site,
        "getusersitepackages",
        lambda: (_ for _ in ()).throw(RuntimeError("user boom")),
    )
    monkeypatch.setattr(
        serving_stack.site,
        "getsitepackages",
        lambda: (_ for _ in ()).throw(RuntimeError("system boom")),
    )

    with pytest.warns(RuntimeWarning) as record:
        roots = serving_stack._site_package_roots()

    assert roots == []
    messages = [str(w.message) for w in record]
    assert any("user site-packages" in message for message in messages)
    assert any("system site-packages" in message for message in messages)


@pytest.mark.parametrize(
    ("enable_user_site", "python_no_user_site"),
    [(False, None), (True, "1")],
)
def test_site_package_roots_exclude_disabled_user_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enable_user_site: bool,
    python_no_user_site: str | None,
) -> None:
    active_prefix = tmp_path / "venv"
    active_site = active_prefix / "lib" / "python3.12" / "site-packages"
    active_lib = active_site / "nvidia" / "cu13" / "lib"
    active_lib.mkdir(parents=True)
    (active_lib / "libcublas.so.13").touch()

    monkeypatch.setattr(serving_stack.sys, "prefix", str(active_prefix))
    monkeypatch.setattr(
        serving_stack.site, "getsitepackages", lambda: [str(active_site)]
    )
    monkeypatch.setattr(serving_stack.site, "ENABLE_USER_SITE", enable_user_site)
    if python_no_user_site is None:
        monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    else:
        monkeypatch.setenv("PYTHONNOUSERSITE", python_no_user_site)
    monkeypatch.setattr(
        serving_stack.site,
        "getusersitepackages",
        lambda: pytest.fail("disabled user site must not be queried"),
    )

    assert serving_stack._site_package_roots() == [active_site]
    assert serving_stack._discover_cuda_wheel_lib_dirs() == [str(active_lib)]


def test_site_package_roots_prefer_active_environment_over_user_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_prefix = tmp_path / "venv"
    active_site = active_prefix / "lib" / "python3.12" / "site-packages"
    base_site = tmp_path / "base" / "lib" / "python3.12" / "site-packages"
    user_site = tmp_path / "home" / ".local" / "lib" / "python3.12" / "site-packages"
    for root in (active_site, base_site, user_site):
        lib_dir = root / "nvidia" / "cudnn" / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libcudnn.so.9").touch()

    monkeypatch.setattr(serving_stack.sys, "prefix", str(active_prefix))
    monkeypatch.setattr(
        serving_stack.site,
        "getsitepackages",
        lambda: [str(base_site), str(active_site)],
    )
    monkeypatch.setattr(serving_stack.site, "ENABLE_USER_SITE", True)
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr(
        serving_stack.site, "getusersitepackages", lambda: str(user_site)
    )

    assert serving_stack._site_package_roots() == [active_site, base_site, user_site]
    assert serving_stack._discover_cuda_wheel_lib_dirs() == [
        str(active_site / "nvidia" / "cudnn" / "lib"),
        str(base_site / "nvidia" / "cudnn" / "lib"),
        str(user_site / "nvidia" / "cudnn" / "lib"),
    ]


def test_cuda_wheel_discovery_finds_the_consolidated_cu13_library_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    cu13_lib = site_packages / "nvidia" / "cu13" / "lib"
    cu13_lib.mkdir(parents=True)
    (cu13_lib / "libcublas.so.13").touch()
    existing_lib = tmp_path / "existing-libs"
    existing_lib.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(existing_lib))
    monkeypatch.setattr(serving_stack, "_site_package_roots", lambda: [])

    discovered = serving_stack.configure_serving_stack_runtime_env(
        extra_roots=[site_packages]
    )

    assert discovered == [str(cu13_lib)]
    assert serving_stack.os.environ["LD_LIBRARY_PATH"].split(serving_stack.os.pathsep) == [
        str(cu13_lib),
        str(existing_lib),
    ]
    assert "libcublas.so.13" in serving_stack._DEFAULT_PRELOAD_LIBS
    assert "libcublas.so.12" not in serving_stack._DEFAULT_PRELOAD_LIBS


def test_vllm_native_loader_imports_both_stable_libtorch_modules(tmp_path: Path) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for index, module_name in enumerate(serving_stack.VLLM_NATIVE_EXTENSION_MODULES):
        module_path = package / f"{module_name.rsplit('.', 1)[-1]}.py"
        module_path.write_text(f"LOAD_ORDER = {index}\n", encoding="utf-8")

    probe = textwrap.dedent(
        """
        import json
        from core.harness.serving_stack import load_vllm_native_extensions

        modules = load_vllm_native_extensions()
        print(json.dumps([[module.__name__, module.LOAD_ORDER] for module in modules]))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(CODE_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=CODE_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        '[["vllm._C_stable_libtorch", 0], '
        '["vllm._moe_C_stable_libtorch", 1]]'
    )


def test_vllm_native_loader_fails_when_moe_extension_is_missing(tmp_path: Path) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_C_stable_libtorch.py").write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(CODE_ROOT), env.get("PYTHONPATH", "")]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from core.harness.serving_stack import load_vllm_native_extensions; "
                "load_vllm_native_extensions()"
            ),
        ],
        cwd=CODE_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "vllm._moe_C_stable_libtorch" in result.stderr


@pytest.mark.parametrize(
    "extension_name",
    serving_stack.VLLM_NATIVE_EXTENSION_MODULES,
)
def test_vllm_abi_error_detection_covers_current_native_modules(
    extension_name: str,
) -> None:
    error = OSError(f"{extension_name}.so: undefined symbol: torch_abi_symbol")

    assert serving_stack.is_vllm_native_extension_abi_error(error)


def test_get_serving_stack_pins_fails_fast_on_unreadable_requirements(tmp_path: Path) -> None:
    requirements_dir = tmp_path / "requirements_latest.txt"
    requirements_dir.mkdir()

    with pytest.raises(RuntimeError, match="Unable to read serving stack requirements file"):
        serving_stack.get_serving_stack_pins(requirements_path=requirements_dir)
