from pathlib import Path

from core.harness.arch_config import _inductor_cutlass_root


def source_tree(root: Path) -> Path:
    for relative in (
        "include/cutlass/cutlass.h",
        "python/cutlass_library/generator.py",
        "python/cutlass_library/library.py",
        "python/cutlass_library/manifest.py",
    ):
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    return root


def test_keeps_configured_cutlass_source_when_dsl_is_installed(tmp_path, monkeypatch):
    configured = source_tree(tmp_path / "configured")
    source_tree(tmp_path / "repo/third_party/cutlass")
    dsl = tmp_path / "nvidia_cutlass_dsl"
    (dsl / "python_packages/cutlass").mkdir(parents=True)
    monkeypatch.setenv("CUTLASS_PATH", str(dsl))

    assert _inductor_cutlass_root(str(configured), repo_root=tmp_path / "repo") == str(configured)


def test_selects_complete_checkout_instead_of_dsl_or_headers_only(tmp_path, monkeypatch):
    dsl = tmp_path / "nvidia_cutlass_dsl"
    (dsl / "python_packages/cutlass").mkdir(parents=True)
    headers = tmp_path / "headers/include/cutlass/cutlass.h"
    headers.parent.mkdir(parents=True)
    headers.touch()
    monkeypatch.setenv("CUTLASS_PATH", str(tmp_path / "headers"))
    checkout = source_tree(tmp_path / "repo/third_party/cutlass")

    assert _inductor_cutlass_root(str(dsl), repo_root=tmp_path / "repo") == str(checkout)


def test_does_not_claim_uninitialized_submodule_is_usable(tmp_path, monkeypatch):
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    (tmp_path / "third_party/cutlass").mkdir(parents=True)

    assert _inductor_cutlass_root(None, repo_root=tmp_path) is None
