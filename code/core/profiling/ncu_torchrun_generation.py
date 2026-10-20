"""Run one rank-local worker generation for NCU Application Replay."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

GENERATION_CONFIG_SCHEMA = "aisp.ncu-torchrun-generation.v1"
GENERATION_MARKER_PREFIX = "__AISP_NCU_TORCHRUN_APPLICATION_GENERATION_V1__="
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "worker_argv",
        "output_dir",
        "world_size",
        "expected_passes",
        "master_addr",
        "master_port_base",
    }
)


def _require_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(value)


def load_generation_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate one generation configuration."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Generation config must be a real file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Generation config must contain a JSON object")
    keys = frozenset(payload)
    if keys != _CONFIG_KEYS:
        raise ValueError(
            "Generation config fields differ from schema. "
            f"missing={sorted(_CONFIG_KEYS - keys)}, unexpected={sorted(keys - _CONFIG_KEYS)}"
        )
    if payload["schema"] != GENERATION_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported generation config schema: {payload['schema']!r}")
    if type(payload["world_size"]) is not int or payload["world_size"] != 2:
        raise ValueError("Generation config world_size must be exactly 2")
    if type(payload["expected_passes"]) is not int or payload["expected_passes"] <= 0:
        raise ValueError("Generation config expected_passes must be a positive integer")
    if payload["master_addr"] != "127.0.0.1":
        raise ValueError("Generation config master_addr must be '127.0.0.1'")
    base = payload["master_port_base"]
    if (
        type(base) is not int
        or base < 1024
        or base + payload["expected_passes"] - 1 > 65535
    ):
        raise ValueError("Generation config master port range must stay within 1024..65535")
    payload["worker_argv"] = _require_string_list(payload["worker_argv"], name="worker_argv")
    output_dir = payload["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Generation config output_dir must be a non-empty string")
    resolved_output = Path(output_dir)
    if resolved_output.is_symlink() or not resolved_output.is_dir():
        raise ValueError(f"Generation output must be a real directory: {resolved_output}")
    payload["output_dir"] = resolved_output
    return payload


def _read_rank_environment(environment: Mapping[str, str]) -> int:
    expected = {"WORLD_SIZE": 2, "LOCAL_WORLD_SIZE": 2, "GROUP_RANK": 0}
    for name, wanted in expected.items():
        raw = environment.get(name)
        try:
            observed = int(raw) if raw is not None else None
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if observed != wanted:
            raise ValueError(f"{name} must be {wanted}, got {raw!r}")
    raw_local_rank = environment.get("LOCAL_RANK")
    raw_rank = environment.get("RANK")
    try:
        local_rank = int(raw_local_rank) if raw_local_rank is not None else None
        rank = int(raw_rank) if raw_rank is not None else None
    except ValueError as exc:
        raise ValueError(
            f"RANK and LOCAL_RANK must be integers, got {raw_rank!r} and {raw_local_rank!r}"
        ) from exc
    if local_rank not in (0, 1) or rank != local_rank:
        raise ValueError(
            "Single-node generation mapping requires RANK == LOCAL_RANK in {0, 1}. "
            f"got RANK={raw_rank!r}, LOCAL_RANK={raw_local_rank!r}"
        )
    return local_rank


def _allocate_generation(
    output_dir: Path, *, local_rank: int, expected_passes: int
) -> tuple[int, Path]:
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Rank application generation locking requires fcntl") from exc

    counter_path = output_dir / f"rank-{local_rank}-generation.counter"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(counter_path, flags, 0o600)
    with os.fdopen(descriptor, "r+", encoding="ascii") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Generation counter must be a regular file: {counter_path}")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        raw = stream.read().strip()
        try:
            generation = int(raw) if raw else 0
        except ValueError as exc:
            raise ValueError(f"Generation counter is not an integer: {counter_path}") from exc
        if not 0 <= generation < expected_passes:
            raise ValueError(
                f"Rank {local_rank} generation must be in 0..{expected_passes - 1}, "
                f"got {generation}"
            )
        stream.seek(0)
        stream.truncate()
        stream.write(f"{generation + 1}\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return generation, counter_path


def _emit_marker(payload: Mapping[str, Any]) -> None:
    print(
        GENERATION_MARKER_PREFIX
        + json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def run_generation(config: Mapping[str, Any], environment: Mapping[str, str]) -> int:
    """Run the unchanged worker argv against a fresh rendezvous store."""

    local_rank = _read_rank_environment(environment)
    expected_passes = int(config["expected_passes"])
    generation, _counter_path = _allocate_generation(
        Path(config["output_dir"]),
        local_rank=local_rank,
        expected_passes=expected_passes,
    )
    master_port = int(config["master_port_base"]) + generation
    event = {
        "event": "begin",
        "generation": generation,
        "master_port": master_port,
        "pid": os.getpid(),
        "rank": local_rank,
    }
    _emit_marker(event)
    child_environment = dict(environment)
    child_environment["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    child_environment["MASTER_ADDR"] = str(config["master_addr"])
    child_environment["MASTER_PORT"] = str(master_port)
    child_environment.pop("TORCHELASTIC_RESTART_COUNT", None)
    completed = subprocess.run(
        list(config["worker_argv"]),
        env=child_environment,
        check=False,
    )
    _emit_marker({**event, "event": "end", "returncode": completed.returncode})
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one rank-local NCU Application Replay worker generation."
    )
    parser.add_argument("generation_config", type=Path)
    args = parser.parse_args(argv)
    return run_generation(load_generation_config(args.generation_config), os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
