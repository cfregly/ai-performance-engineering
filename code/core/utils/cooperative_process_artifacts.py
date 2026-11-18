"""Retain an owned command's streams and execution record on success or failure."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

from core.utils.cooperative_subprocess import run_cooperative_process


def require_owned_uncapped_native_config(config: Any, *, launcher_name: str) -> None:
    """Require the harness path whose outer worker owns cooperative cleanup."""
    value = getattr(config, "execution_timeout_seconds", None)
    if type(value) is not int or value != 0:
        raise RuntimeError(
            f"SKIPPED: {launcher_name} requires explicit execution_timeout_seconds=0. "
            "Its MPI launcher starts a separate owned process group that a finite outer "
            "worker timeout cannot safely clean up. Use the documented uncapped bench "
            "command or run the standalone native launcher directly."
        )


def run_recorded_cooperative_process(command: Sequence[str], *, output_prefix: Path) -> dict[str, Any]:
    """Create fresh artifact files before launch and retain diagnostics after interruption."""
    paths = {name: output_prefix.with_name(output_prefix.name + suffix) for name, suffix in (
        ("stdout", ".stdout.log"), ("stderr", ".stderr.log"), ("execution", ".execution.json"),
    )}
    with ExitStack() as stack:
        streams = {}
        created_paths: list[Path] = []
        try:
            for path in paths.values():
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"Refusing to reuse an existing command artifact: {path}")
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                streams[name] = stack.enter_context(path.open("x", encoding="utf-8"))
                created_paths.append(path)
        except BaseException as error:
            stack.close()
            for path in reversed(created_paths):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    error.add_note(f"Failed to remove partial command artifact {path}: {cleanup_error}")
            raise

        def retain(result: dict[str, Any]) -> None:
            result["artifact_paths"] = {name: str(path) for name, path in paths.items()}
            streams["stdout"].write(result["stdout"])
            streams["stderr"].write(result["stderr"])
            json.dump(result, streams["execution"], indent=2)
            streams["execution"].write("\n")
            for stream in streams.values():
                stream.flush()

        try:
            result = run_cooperative_process(command)
        except BaseException as error:
            result = getattr(error, "result", None) or getattr(error, "cooperative_process_result", None)
            if result is not None:
                try:
                    retain(result)
                except Exception as retention_error:
                    error.add_note(f"Failed to retain command diagnostics: {retention_error}")
            else:
                try:
                    json.dump({
                        "command": list(command), "receipt_available": False,
                        "error_type": type(error).__name__, "error": str(error),
                    }, streams["execution"], indent=2)
                    streams["execution"].write("\n")
                    streams["execution"].flush()
                except Exception as retention_error:
                    error.add_note(f"Failed to retain launch failure: {retention_error}")
            raise
        retain(result)
        return result
