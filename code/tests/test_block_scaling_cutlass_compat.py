from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SUBJECT_PATH = (
    Path(__file__).resolve().parents[1] / "labs" / "block_scaling" / "block_scaling_common.py"
)


def _load_subject(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    torch_stub = ModuleType("torch")
    for name in ("bfloat16", "float16", "float32", "int64"):
        setattr(torch_stub, name, object())
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    spec = importlib.util.spec_from_file_location(
        "test_block_scaling_common_cutlass_compat",
        SUBJECT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    subject = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, subject)
    spec.loader.exec_module(subject)
    return subject


def _write_example(path: Path, cute_definition: str) -> None:
    path.write_text(
        "from types import SimpleNamespace\n"
        f"{cute_definition}\n"
        "cutlass_torch = object()\n"
        "def invoke():\n"
        "    return cute.make_fragment('shape', 'dtype')\n"
    )


def test_loader_aliases_removed_make_fragment_to_pinned_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subject = _load_subject(monkeypatch)
    example_path = tmp_path / "blockscaled_example.py"
    _write_example(
        example_path,
        "cute = SimpleNamespace(make_rmem_tensor=lambda *args: args)",
    )
    monkeypatch.setattr(subject, "_resolve_cutlass_example_path", lambda: example_path)

    module = subject.load_cutlass_example_module()

    assert module.invoke() == ("shape", "dtype")
    assert module.cute.make_fragment is module.cute.make_rmem_tensor


def test_loader_preserves_runtime_with_legacy_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subject = _load_subject(monkeypatch)
    example_path = tmp_path / "blockscaled_example.py"
    _write_example(
        example_path,
        "cute = SimpleNamespace("
        "make_fragment=lambda *args: ('legacy', *args), "
        "make_rmem_tensor=lambda *args: ('replacement', *args))",
    )
    monkeypatch.setattr(subject, "_resolve_cutlass_example_path", lambda: example_path)

    module = subject.load_cutlass_example_module()

    assert module.invoke() == ("legacy", "shape", "dtype")


def test_loader_fails_closed_when_cute_exposes_neither_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subject = _load_subject(monkeypatch)
    example_path = tmp_path / "blockscaled_example.py"
    _write_example(example_path, "cute = SimpleNamespace()")
    monkeypatch.setattr(subject, "_resolve_cutlass_example_path", lambda: example_path)

    with pytest.raises(RuntimeError, match="neither make_fragment nor make_rmem_tensor"):
        subject.load_cutlass_example_module()
