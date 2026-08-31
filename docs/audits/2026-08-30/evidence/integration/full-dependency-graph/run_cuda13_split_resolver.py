#!/usr/bin/env python3
"""Resolve the collision-free CUDA 13 base and ABI-bound vLLM wheel locks."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
UV = shutil.which("uv")
if UV is None:
    raise SystemExit("uv is required")

BASE_NAME = "cuda13-base-final-source-attempt-1"
VLLM_NAME = "vllm-cu130-no-deps-attempt-1"
BASE_PLATFORM = "x86_64-manylinux_2_31"
VLLM_PLATFORM = "x86_64-manylinux_2_35"
VLLM_INDEX = "https://wheels.vllm.ai/0.16.0/cu130"
VLLM_WHEEL_URL = (
    "https://wheels.vllm.ai/89a77b10846fd96273cce78d86d2556ea582d26e/"
    "vllm-0.16.0%2Bcu130-cp38-abi3-manylinux_2_35_x86_64.whl"
)
VLLM_WHEEL_SHA256 = "bda6ff19ead743fb30c6271cdeb7daf62d5bd5f7a53cb6c2e7d987d53ea3d49f"
VLLM_WHEEL_SIZE = 280_962_334


def run_case(
    *,
    name: str,
    content: str,
    platform: str,
    default_index: str,
    flags: list[str],
    python: Path,
    work: Path,
) -> dict[str, object]:
    case = OUT / name
    case.mkdir(exist_ok=False)
    requirements = case / "requirements.in"
    requirements.write_text(content, encoding="utf-8")
    output = case / "resolved-requirements.txt"
    argv = [
        UV,
        "pip",
        "compile",
        "--verbose",
        "--python",
        str(python),
        "--python-version",
        "3.12",
        "--python-platform",
        platform,
        "--only-binary",
        ":all:",
        *flags,
        "--build-constraints",
        str(OUT / "build-constraints.txt"),
        "--no-python-downloads",
        "--no-config",
        "--keyring-provider",
        "disabled",
        "--default-index",
        default_index,
        "--index-strategy",
        "unsafe-best-match",
        "--cache-dir",
        str(work / f"{name}-cache"),
        "--no-progress",
        "--color",
        "never",
        "--generate-hashes",
        "--emit-index-annotation",
        "--emit-index-url",
        "--emit-build-options",
        "--output-file",
        str(output),
        str(requirements),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PIP_", "UV_"))
    }
    environment["UV_HTTP_TIMEOUT"] = "30"
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    result = subprocess.run(
        argv,
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    (case / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (case / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    command = {
        "argv": argv,
        "build_policy": (
            "Only GPUtil is allowed from reviewed source in the base lock; the "
            "ABI-bound vLLM wheel is resolved without dependencies. No target install."
        ),
        "cwd": str(work),
        "elapsed_seconds": time.monotonic() - start,
        "environment_policy": (
            "Inherited UV_/PIP_ settings removed; explicit public indexes; "
            "keyring disabled."
        ),
        "exit_code": result.returncode,
        "started_utc": started,
        "timeout_seconds": 240,
        "timed_out": False,
    }
    (case / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "case": name,
        "exit_code": result.returncode,
        "stderr_tail": result.stderr[-4000:],
    }


def finalize_vllm_lock() -> None:
    """Bind the hashless upstream Simple entry to the reviewed x86-64 wheel."""

    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(VLLM_WHEEL_URL, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    observed_sha256 = digest.hexdigest()
    if (size, observed_sha256) != (VLLM_WHEEL_SIZE, VLLM_WHEEL_SHA256):
        raise ValueError(
            "ABI-bound vLLM wheel changed: "
            f"size={size}, sha256={observed_sha256}"
        )

    case = OUT / VLLM_NAME
    (case / "resolved-requirements.txt").write_text(
        "# Exact no-deps CUDA 13 wheel selected from the versioned vLLM release index.\n"
        f"--index-url {VLLM_INDEX}\n"
        "--only-binary :all:\n\n"
        f"vllm @ {VLLM_WHEEL_URL} \\\n"
        f"    --hash=sha256:{VLLM_WHEEL_SHA256}\n"
        f"    # from {VLLM_INDEX}\n",
        encoding="utf-8",
    )
    (case / "selected-wheel.json").write_text(
        json.dumps(
            {
                "index": VLLM_INDEX,
                "name": "vllm",
                "no_deps": True,
                "platform": VLLM_PLATFORM,
                "sha256": VLLM_WHEEL_SHA256,
                "size_bytes": VLLM_WHEEL_SIZE,
                "url": VLLM_WHEEL_URL,
                "version": "0.16.0+cu130",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    source = (ROOT / "code" / "requirements_latest.txt").read_text(encoding="utf-8")
    if source.count("-f ./third_party/wheels") != 1:
        raise ValueError("expected one local find-links directive")
    base = source.replace(
        "-f ./third_party/wheels",
        "# Resolution evidence omits absent local find-links; all base specs remain.",
    )
    active = {
        line.split("==", 1)[0].split("[", 1)[0].strip().lower()
        for line in base.splitlines()
        if line and not line[0].isspace() and not line.startswith(("#", "-"))
    }
    if "vllm" in active or "typer-slim" in active:
        raise ValueError("base requirements must exclude vLLM and typer-slim")
    if "cupy-cuda13x" not in active:
        raise ValueError("base requirements must explicitly select CUDA 13 CuPy")

    vllm_pin = (ROOT / "code" / "vllm_no_deps.pin").read_text(encoding="utf-8")
    if [
        line.strip()
        for line in vllm_pin.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] != ["vllm==0.16.0+cu130"]:
        raise ValueError("unexpected ABI-bound vLLM pin")

    work = Path(tempfile.mkdtemp(prefix="aisp-cuda13-split-resolver-"))
    venv = work / "venv"
    venv_result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        text=True,
        capture_output=True,
        check=False,
    )
    (OUT / "cuda13-split-resolver-context.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "task_root": str(work),
                "venv_argv": [sys.executable, "-m", "venv", str(venv)],
                "venv_exit_code": venv_result.returncode,
                "uv": UV,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if venv_result.returncode != 0:
        raise SystemExit(venv_result.stderr)
    python = venv / "bin" / "python"

    results = [
        run_case(
            name=BASE_NAME,
            content=base,
            platform=BASE_PLATFORM,
            default_index="https://pypi.org/simple",
            flags=["--no-binary", "gputil"],
            python=python,
            work=work,
        ),
        run_case(
            name=VLLM_NAME,
            content=vllm_pin,
            platform=VLLM_PLATFORM,
            default_index=VLLM_INDEX,
            flags=["--no-deps"],
            python=python,
            work=work,
        ),
    ]
    if int(results[1]["exit_code"]) == 0:
        finalize_vllm_lock()
    for result in results:
        print(json.dumps(result), flush=True)
    return max(int(result["exit_code"]) for result in results)


if __name__ == "__main__":
    raise SystemExit(main())
