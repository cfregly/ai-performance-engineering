"""Utility to build a stub libnvToolsExt archive when CUDA 13+ ships NVTX as header-only."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
from pathlib import Path

_NVTX_STUB_LOCK = threading.Lock()


def _default_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    stub_dir = repo_root / "core" / "common" / "nvtx_stub"
    src_path = stub_dir / "nvtx_stub.cpp"
    lib_path = stub_dir / "libnvToolsExt.a"
    return stub_dir, src_path, lib_path


def _ensure_nvtx_stub_unlocked(output_path: Path | None = None) -> Path:
    """Build or reuse the archive while the process-wide lock is held.

    Args:
        output_path: Optional custom path for the generated archive.

    Returns:
        Path to the stub archive.
    """
    stub_dir, src_path, default_lib_path = _default_paths()
    lib_path = Path(output_path).resolve() if output_path else default_lib_path
    obj_path = lib_path.with_suffix(".o")

    if not src_path.exists():
        raise FileNotFoundError(f"Missing NVTX stub source: {src_path}")

    # Rebuild when the archive is missing or older than the source.
    needs_build = (
        not lib_path.exists()
        or lib_path.stat().st_mtime < src_path.stat().st_mtime
    )
    if not needs_build:
        return lib_path

    lib_path.parent.mkdir(parents=True, exist_ok=True)

    cxx = os.environ.get("NVTX_STUB_CXX") or os.environ.get("CXX") or "g++"
    ar = os.environ.get("NVTX_STUB_AR") or os.environ.get("AR") or "ar"
    compile_cmd = [
        cxx,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-c",
        str(src_path),
        "-o",
        str(obj_path),
    ]
    archive_cmd = [
        ar,
        "rcs",
        str(lib_path),
        str(obj_path),
    ]

    subprocess.run(compile_cmd, check=True)
    subprocess.run(archive_cmd, check=True)
    obj_path.unlink(missing_ok=True)
    return lib_path


def ensure_nvtx_stub(output_path: Path | None = None) -> Path:
    """Ensure the stub exists without racing parallel extension loaders."""

    with _NVTX_STUB_LOCK:
        return _ensure_nvtx_stub_unlocked(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NVTX stub archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit path for the generated libnvToolsExt.a",
    )
    args = parser.parse_args()
    lib_path = ensure_nvtx_stub(args.output)
    print(str(lib_path))


if __name__ == "__main__":
    main()
