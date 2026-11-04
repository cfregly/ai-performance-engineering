"""Test configuration for nanochat_fullstack tests."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
importlib.import_module("labs.nanochat_fullstack")


def _ensure_rustbpe_extension() -> None:
    """Build and install the rustbpe Python extension when it is unavailable.

    The repository contains a Rust crate at `labs/nanochat_fullstack/rustbpe/`.
    When `labs/nanochat_fullstack/` is on `sys.path`, that directory can be
    imported as a *namespace package* named `rustbpe`, which shadows the real
    extension module and lacks `Tokenizer`.

    Maturin builds the PyO3 extension for the active Python interpreter and
    applies the platform-specific linker and extension-module settings.
    """
    try:
        module = importlib.import_module("rustbpe")
    except Exception as exc:
        initial_import_problem = f"{type(exc).__name__}: {exc}"
    else:
        if hasattr(module, "Tokenizer"):
            return
        module_path = getattr(module, "__file__", None) or "<namespace package>"
        initial_import_problem = f"rustbpe resolved to {module_path} without a Tokenizer attribute"

    crate_dir = PROJECT_ROOT / "rustbpe"
    command = [
        sys.executable,
        "-m",
        "maturin",
        "develop",
        "--release",
        "--manifest-path",
        str(crate_dir / "Cargo.toml"),
    ]
    build_env = os.environ.copy()
    build_env["VIRTUAL_ENV"] = sys.prefix
    build_env.pop("CONDA_PREFIX", None)
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build and install the rustbpe Python extension via maturin.\n"
            f"initial import: {initial_import_problem}\n"
            f"python: {sys.executable}\n"
            f"cwd: {PROJECT_ROOT}\n"
            f"command: {command!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )

    sys.modules.pop("rustbpe", None)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("rustbpe")
    except Exception as exc:
        raise RuntimeError(
            "Maturin built rustbpe successfully, but Python could not import it.\n"
            f"initial import: {initial_import_problem}\n"
            f"python: {sys.executable}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        ) from exc
    if not hasattr(module, "Tokenizer"):
        module_path = getattr(module, "__file__", None) or "<namespace package>"
        raise RuntimeError(
            "Maturin built rustbpe successfully, but the imported module is invalid.\n"
            f"module: {module_path}\n"
            "missing attribute: Tokenizer"
        )


_ensure_rustbpe_extension()
