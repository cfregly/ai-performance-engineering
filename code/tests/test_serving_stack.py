from __future__ import annotations

from pathlib import Path

import pytest

from core.harness import serving_stack


def test_get_serving_stack_pins_reads_the_cuda13_vllm_manifest() -> None:
    pins = serving_stack.get_serving_stack_pins()

    assert pins.torch_version == "2.9.1+cu130"
    assert pins.vllm_version == "0.16.0+cu130"
    assert pins.flashinfer_version == "0.6.3"
    assert pins.vllm_pip_spec == "vllm==0.16.0+cu130"


def test_get_serving_stack_pins_keeps_vllm_separate_from_base_requirements(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements_latest.txt"
    requirements.write_text(
        "torch==2.9.1+cu130\n"
        "vllm==0.15.0\n"
        "flashinfer-python==0.6.3\n",
        encoding="utf-8",
    )
    vllm_pin = tmp_path / "vllm_no_deps.pin"
    vllm_pin.write_text("vllm==0.16.0+cu130\n", encoding="utf-8")

    pins = serving_stack.get_serving_stack_pins(
        requirements_path=requirements,
        vllm_pin_path=vllm_pin,
    )

    assert pins.vllm_version == "0.16.0+cu130"


def test_get_serving_stack_pins_requires_the_separate_vllm_manifest(
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements_latest.txt"
    requirements.write_text(
        "torch==2.9.1+cu130\nflashinfer-python==0.6.3\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ABI-bound vLLM pin manifest is missing"):
        serving_stack.get_serving_stack_pins(
            requirements_path=requirements,
            vllm_pin_path=tmp_path / "missing-vllm.pin",
        )


def test_site_package_roots_warn_on_discovery_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serving_stack.site, "getusersitepackages", lambda: (_ for _ in ()).throw(RuntimeError("user boom")))
    monkeypatch.setattr(serving_stack.site, "getsitepackages", lambda: (_ for _ in ()).throw(RuntimeError("system boom")))

    with pytest.warns(RuntimeWarning) as record:
        roots = serving_stack._site_package_roots()

    assert roots == []
    messages = [str(w.message) for w in record]
    assert any("user site-packages" in message for message in messages)
    assert any("system site-packages" in message for message in messages)


def test_get_serving_stack_pins_fails_fast_on_unreadable_requirements(tmp_path: Path) -> None:
    requirements_dir = tmp_path / "requirements_latest.txt"
    requirements_dir.mkdir()

    with pytest.raises(RuntimeError, match="Unable to read serving stack requirements file"):
        serving_stack.get_serving_stack_pins(requirements_path=requirements_dir)
