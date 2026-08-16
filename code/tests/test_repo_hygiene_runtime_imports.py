from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import core.env as core_env


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_ENV_KEYS = (
    "PYTHONFAULTHANDLER",
    "TORCH_SHOW_CPP_STACKTRACES",
    "TORCH_CUDNN_V8_API_ENABLED",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTORCH_ALLOC_CONF",
    "TORCH_COMPILE_DEBUG",
    "AISP_CUDNN_RUNTIME_POLICY",
    "CUDA_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)


def test_env_defaults_do_not_preload_system_nccl_over_wheel_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    wheel_lib_dir = tmp_path / "nvidia" / "nccl" / "lib"
    wheel_lib_dir.mkdir(parents=True)
    (wheel_lib_dir / "libnccl.so.2").touch()
    system_nccl = tmp_path / "system" / "libnccl.so.2"
    system_nccl.parent.mkdir()
    system_nccl.touch()

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(core_env, "NCCL_LIBRARY_PATH", str(system_nccl))
    monkeypatch.setattr(
        core_env,
        "_discover_nvidia_wheel_lib_dirs",
        lambda: [str(wheel_lib_dir)],
    )
    monkeypatch.setenv("AISP_CUDNN_RUNTIME_POLICY", "auto")
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    core_env._ensure_ld_preload()

    assert os.environ.get("LD_PRELOAD") is None


def test_env_defaults_preload_system_nccl_for_explicit_system_policy(
    tmp_path: Path, monkeypatch
) -> None:
    wheel_lib_dir = tmp_path / "nvidia" / "nccl" / "lib"
    wheel_lib_dir.mkdir(parents=True)
    (wheel_lib_dir / "libnccl.so.2").touch()
    system_nccl = tmp_path / "system" / "libnccl.so.2"
    system_nccl.parent.mkdir()
    system_nccl.touch()

    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(core_env, "NCCL_LIBRARY_PATH", str(system_nccl))
    monkeypatch.setattr(
        core_env,
        "_discover_nvidia_wheel_lib_dirs",
        lambda: [str(wheel_lib_dir)],
    )
    monkeypatch.setenv("AISP_CUDNN_RUNTIME_POLICY", "system")
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    core_env._ensure_ld_preload()

    assert os.environ["LD_PRELOAD"] == str(system_nccl)


def test_ch04_and_ch15_import_verification_payload_mixin_from_core() -> None:
    legacy_imports = (
        "from ch04.verification_payload_mixin import VerificationPayloadMixin",
        "from ch15.verification_payload_mixin import VerificationPayloadMixin",
    )
    offenders: list[str] = []
    for root_name in ("ch04", "ch15"):
        for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if any(legacy in text for legacy in legacy_imports):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"legacy verification mixin shim imports remain: {offenders}"


def test_transformer_engine_benchmarks_do_not_mutate_env_on_import() -> None:
    script = textwrap.dedent(
        f"""
        import json
        import os

        tracked_keys = {TRACKED_ENV_KEYS!r}
        before = {{key: os.environ.get(key) for key in tracked_keys}}

        import ch13.baseline_precisionfp8_te  # noqa: F401
        import ch13.optimized_precisionfp8_te  # noqa: F401
        import ch13.optimized_fp4_perchannel  # noqa: F401

        after = {{key: os.environ.get(key) for key in tracked_keys}}
        print(json.dumps({{"before": before, "after": after}}, sort_keys=True))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["before"] == payload["after"]


def test_target_chapters_do_not_define_local_resolve_device_helpers() -> None:
    offenders: list[str] = []
    roots = (
        REPO_ROOT / "ch11",
        REPO_ROOT / "ch16",
        REPO_ROOT / "ch19",
        REPO_ROOT / "ch20",
        REPO_ROOT / "labs" / "persistent_decode",
    )
    pattern = re.compile(r"^def resolve_device\b", re.MULTILINE)

    for root in roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], f"local resolve_device helpers remain: {offenders}"
