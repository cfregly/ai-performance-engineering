"""NanoChat's lab-local uv environment must match the current stable Torch stack."""

from __future__ import annotations

from pathlib import Path
import tomllib

from packaging.version import Version


CODE = Path(__file__).resolve().parents[1]
NANOCHAT = CODE / "labs" / "nanochat_fullstack"


def test_nanochat_project_selects_exact_stable_torch_extras() -> None:
    pyproject_text = (NANOCHAT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)

    project = pyproject["project"]
    assert project["requires-python"] == ">=3.10,<3.15"
    assert "torch==2.14.1" in project["dependencies"]
    assert project["optional-dependencies"] == {
        "cpu": ["torch==2.14.1"],
        "gpu": ["torch==2.14.1"],
    }

    uv = pyproject["tool"]["uv"]
    assert "prerelease-mode" not in uv
    assert uv["sources"]["torch"] == [
        {
            "index": "pytorch-cpu",
            "extra": "cpu",
            "marker": "sys_platform == 'linux' and platform_machine == 'x86_64'",
        },
        {
            "index": "pytorch-cu130",
            "extra": "gpu",
            "marker": "sys_platform == 'linux' and platform_machine == 'x86_64'",
        },
    ]
    indexes = {entry["name"]: entry for entry in uv["index"]}
    assert indexes == {
        "pytorch-cpu": {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        },
        "pytorch-cu130": {
            "name": "pytorch-cu130",
            "url": "https://download.pytorch.org/whl/cu130",
            "explicit": True,
        },
    }
    assert "nightly" not in pyproject_text
    assert "third_party/wheels" not in pyproject_text


def test_nanochat_lock_contains_only_stable_torch_2141_sources() -> None:
    lock_text = (NANOCHAT / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)

    assert lock["requires-python"] == ">=3.10, <3.15"
    resolved_versions = [Version(package["version"]) for package in lock["package"]]
    assert not any(
        version.is_prerelease or version.is_devrelease for version in resolved_versions
    )
    torch_packages = [package for package in lock["package"] if package["name"] == "torch"]
    assert torch_packages
    assert {Version(package["version"]).public for package in torch_packages} == {"2.14.1"}
    registries = {package["source"]["registry"] for package in torch_packages}
    assert registries <= {
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu130",
        "https://pypi.org/simple",
    }
    assert "https://download.pytorch.org/whl/cpu" in registries
    assert "https://download.pytorch.org/whl/cu130" in registries

    triton_packages = [package for package in lock["package"] if package["name"] == "triton"]
    assert {package["version"] for package in triton_packages} == {"3.8.0"}
    assert "nightly" not in lock_text
    assert "third_party/wheels" not in lock_text


def test_nanochat_readme_describes_the_stable_lab_local_activation() -> None:
    readme = (NANOCHAT / "README.md").read_text(encoding="utf-8")
    assert "`uv sync --extra gpu`" in readme
    assert "exact stable Torch 2.14.1 CUDA 13.0 wheel" in readme
    assert "lab-local environment" in readme
