"""Regression tests for CUDA graph state in short-lived Python workers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from core.utils import compile_utils  # noqa: F401 - installs the guarded get_obj


def test_missing_cudagraph_buckets_stay_python_thread_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._inductor.cudagraph_trees as trees

    local = threading.local()
    native_stashes: list[tuple[str, object]] = []
    monkeypatch.setattr(torch._C, "_is_key_in_tls", lambda _name: False)
    monkeypatch.setattr(
        torch._C,
        "_stash_obj_in_tls",
        lambda name, value: native_stashes.append((name, value)),
    )

    def load_buckets() -> tuple[str, str, bool, bool]:
        containers = trees.get_obj(local, "tree_manager_containers")
        locks = trees.get_obj(local, "tree_manager_locks")
        return (
            type(containers).__name__,
            type(locks).__name__,
            containers is local.tree_manager_containers,
            locks is local.tree_manager_locks,
        )

    for _ in range(16):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark-thread")
        assert executor.submit(load_buckets).result(timeout=10) == (
            "dict",
            "defaultdict",
            True,
            True,
        )
        executor.shutdown(wait=False, cancel_futures=True)

    assert native_stashes == []


def test_existing_native_cudagraph_bucket_remains_authoritative() -> None:
    import torch._inductor.cudagraph_trees as trees

    key = "tree_manager_containers"
    assert torch._C._is_key_in_tls(key)
    expected = torch._C._get_obj_in_tls(key)
    assert trees.get_obj(threading.local(), key) is expected


def test_unknown_cudagraph_tls_attribute_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._inductor.cudagraph_trees as trees

    monkeypatch.setattr(torch._C, "_is_key_in_tls", lambda _name: False)
    with pytest.raises(AssertionError, match="Missing TLS object for unknown_bucket"):
        trees.get_obj(threading.local(), "unknown_bucket")


def test_short_lived_worker_tls_survives_parent_logging(tmp_path: Path) -> None:
    """Exercise the real private-TLS boundary in a disposable debug-allocator process."""
    script = textwrap.dedent(
        """
        import logging
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import torch
        import torch._inductor.cudagraph_trees as trees

        from core.harness.run_benchmarks import reset_cuda_state
        from core.utils.logger import get_logger, setup_logging

        assert not torch.cuda.is_available()
        setup_logging(
            level="INFO",
            log_file=Path(sys.argv[1]),
            log_format="json",
            use_rich=False,
        )
        logger = get_logger("tls-lifetime-regression")

        def load_buckets():
            return (
                type(trees.get_obj(trees.local, "tree_manager_containers")).__name__,
                type(trees.get_obj(trees.local, "tree_manager_locks")).__name__,
            )

        for index in range(64):
            reset_cuda_state(allow_cuda_context=False)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark-thread")
            assert executor.submit(load_buckets).result(timeout=30) == ("dict", "defaultdict")
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("completed short-lived worker %d", index)

        logging.shutdown()
        print("completed=64")
        """
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONMALLOC"] = "debug"
    env["PYTHONFAULTHANDLER"] = "1"
    code_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (code_root, env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "benchmark.log")],
        cwd=code_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed=64" in result.stdout
