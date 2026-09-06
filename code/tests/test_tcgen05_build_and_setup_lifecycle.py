from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn import functional

from ch10 import baseline_matmul_tcgen05_epilogue as epilogue
from ch10 import optimized_matmul_tcgen05_vs_cublas as vs_cublas
from core.common import tcgen05
from core.harness.validity_checks import check_setup_precomputation

CODE_ROOT = Path(__file__).resolve().parents[1]


def _wait_for_path(path: Path, processes: tuple[subprocess.Popen[str], ...]) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        exited = [process for process in processes if process.poll() is not None]
        if exited:
            details = "\n".join(
                f"rc={process.returncode}\n{process.communicate()[1]}" for process in exited
            )
            pytest.fail(f"lock test subprocess exited before {path.name}:\n{details}")
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {path}")


def test_extension_build_lock_serializes_processes_outside_deleted_build_dir(
    tmp_path: Path,
) -> None:
    if tcgen05.fcntl is None:
        pytest.skip("POSIX advisory locking is unavailable")

    cache_root = tmp_path / "extensions"
    owner_acquired = tmp_path / "owner-acquired"
    waiter_acquired = tmp_path / "waiter-acquired"
    release_owner = tmp_path / "release-owner"
    child = r"""
import os
from pathlib import Path
import shutil
import sys
import time

from core.common import tcgen05

cache_root, role, acquired_path, release_path = sys.argv[1:]
os.environ["TORCH_EXTENSIONS_DIR"] = cache_root
with tcgen05._extension_build_lock("shared_ext"):
    build_dir = tcgen05._get_extension_build_dir("shared_ext")
    if role == "owner":
        build_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(build_dir)
    Path(acquired_path).write_text(role, encoding="utf-8")
    if role == "owner":
        deadline = time.monotonic() + 20.0
        while not Path(release_path).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for owner release")
            time.sleep(0.01)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(CODE_ROOT), env.get("PYTHONPATH", ""))))
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child,
            str(cache_root),
            "owner",
            str(owner_acquired),
            str(release_owner),
        ],
        cwd=CODE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    waiter: subprocess.Popen[str] | None = None
    try:
        _wait_for_path(owner_acquired, (owner,))
        waiter = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child,
                str(cache_root),
                "waiter",
                str(waiter_acquired),
                str(release_owner),
            ],
            cwd=CODE_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        assert not waiter_acquired.exists(), "second process entered the protected build cache"
        release_owner.write_text("release", encoding="utf-8")
        _wait_for_path(waiter_acquired, (waiter,))
        for process in (owner, waiter):
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"
    finally:
        release_owner.touch()
        for process in (owner, waiter):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()

    assert (cache_root / ".shared_ext.build.lock").is_file()
    assert not (cache_root / "shared_ext" / "lock").exists()


def test_load_extension_holds_outer_lock_through_fingerprint_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = {"locked": False}
    events: list[str] = []
    module = object()
    build_dir = tmp_path / "test_ext"

    @contextmanager
    def recording_lock(name: str) -> Iterator[None]:
        assert name == "test_ext"
        state["locked"] = True
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")
            state["locked"] = False

    def fake_load(**_kwargs: Any) -> object:
        assert state["locked"]
        events.append("load")
        return module

    def fake_write(*_args: Any) -> None:
        assert state["locked"]
        events.append("fingerprint")

    monkeypatch.setattr(tcgen05, "_extension_build_lock", recording_lock)
    monkeypatch.setattr(tcgen05, "_tcgen05_cuda_flags", lambda: [])
    monkeypatch.setattr(tcgen05, "_check_and_invalidate_cache", lambda *_args: "digest")
    monkeypatch.setattr(tcgen05, "_clean_stale_build", lambda _name: None)
    monkeypatch.setattr(tcgen05, "_get_extension_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(tcgen05, "_write_build_fingerprint", fake_write)
    monkeypatch.setattr(tcgen05, "load", fake_load)

    assert tcgen05._load_extension("test_ext", [tmp_path / "kernel.cu"]) is module
    assert events == ["lock", "load", "fingerprint", "unlock"]


def test_vs_cublas_setup_keeps_public_output_empty_until_real_cpu_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vs_cublas, "check_tcgen05_support", lambda **_kwargs: (True, None))
    monkeypatch.setattr(vs_cublas, "require_cuda_device", lambda _reason: torch.device("cpu"))
    benchmark = vs_cublas.OptimizedMatmulTCGen05Benchmark()
    benchmark.size = benchmark.n = 5

    valid, error = check_setup_precomputation(lambda: {"output": benchmark.output}, benchmark.setup)

    assert valid, error
    assert benchmark.output is None
    assert benchmark._output_buffer is not None
    benchmark.benchmark_fn()
    assert benchmark.output is benchmark._output_buffer
    torch.testing.assert_close(benchmark.output, benchmark.A @ benchmark.B.T)


class _CpuMatmulModule:
    @staticmethod
    def matmul_tcgen05(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left @ right.T


def test_epilogue_setup_keeps_public_output_empty_until_real_cpu_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(epilogue, "check_tcgen05_support", lambda **_kwargs: (True, None))
    benchmark = epilogue.BaselineMatmulTCGen05EpilogueBenchmark()
    benchmark.device = torch.device("cpu")
    benchmark.M, benchmark.N, benchmark.K = 5, 7, 3
    benchmark.module = _CpuMatmulModule()

    valid, error = check_setup_precomputation(lambda: {"output": benchmark.output}, benchmark.setup)

    assert valid, error
    assert benchmark.output is None
    assert benchmark._output_buffer is not None
    expected = functional.silu((benchmark.A @ benchmark.B.T).float() + benchmark.bias).half()
    benchmark.benchmark_fn()
    assert benchmark.output is benchmark._output_buffer
    torch.testing.assert_close(benchmark.output, expected)
