"""Launch one rank-local Nsight Compute process under ``torchrun``.

This module is an implementation detail of :mod:`core.profiling.ncu_torchrun_capture`.
It intentionally supports exactly one local, two-rank launch.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RANK_CONFIG_SCHEMA = "aisp.ncu-torchrun-rank.v1"
_CONFIG_KEYS = frozenset({"schema", "ncu_prefix", "worker_argv", "output_dir", "world_size"})


def _require_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(value)


def load_rank_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate a rank-worker configuration file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Rank config must be a real file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Rank config must contain a JSON object")
    keys = frozenset(payload)
    if keys != _CONFIG_KEYS:
        raise ValueError(
            "Rank config fields differ from schema; "
            f"missing={sorted(_CONFIG_KEYS - keys)}, unexpected={sorted(keys - _CONFIG_KEYS)}"
        )
    if payload["schema"] != RANK_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported rank config schema: {payload['schema']!r}")
    if type(payload["world_size"]) is not int or payload["world_size"] != 2:
        raise ValueError("Rank config world_size must be exactly 2")
    payload["ncu_prefix"] = _require_string_list(payload["ncu_prefix"], name="ncu_prefix")
    payload["worker_argv"] = _require_string_list(payload["worker_argv"], name="worker_argv")
    output_dir = payload["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Rank config output_dir must be a non-empty string")
    resolved_output = Path(output_dir)
    if resolved_output.is_symlink() or not resolved_output.is_dir():
        raise ValueError(f"Rank output must be a real directory: {resolved_output}")
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
    raw_rank = environment.get("LOCAL_RANK")
    try:
        local_rank = int(raw_rank) if raw_rank is not None else None
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer, got {raw_rank!r}") from exc
    if local_rank not in (0, 1):
        raise ValueError(f"LOCAL_RANK must be 0 or 1, got {raw_rank!r}")
    raw_global_rank = environment.get("RANK")
    try:
        global_rank = int(raw_global_rank) if raw_global_rank is not None else None
    except ValueError as exc:
        raise ValueError(f"RANK must be an integer, got {raw_global_rank!r}") from exc
    if global_rank != local_rank:
        raise ValueError(
            f"Single-node rank mapping requires RANK == LOCAL_RANK, got {raw_global_rank!r}"
        )
    return local_rank


def build_rank_command(
    config: Mapping[str, Any], environment: Mapping[str, str]
) -> tuple[list[str], Path, Path]:
    """Return the exact NCU argv and rank-specific artifact paths."""

    local_rank = _read_rank_environment(environment)
    output_dir = Path(config["output_dir"])
    report_path = output_dir / f"rank-{local_rank}.ncu-rep"
    argv_path = output_dir / f"rank-{local_rank}-argv.json"
    for path in (report_path, argv_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite rank artifact: {path}")
    command = [
        *config["ncu_prefix"],
        "--devices",
        str(local_rank),
        "--export",
        str(report_path),
        *config["worker_argv"],
    ]
    return command, report_path, argv_path


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one rank-local NCU command from a validated capture plan."
    )
    parser.add_argument("rank_config", type=Path)
    args = parser.parse_args(argv)
    config = load_rank_config(args.rank_config)
    command, _report_path, argv_path = build_rank_command(config, os.environ)
    _write_json(argv_path, command)
    os.execvpe(command[0], command, dict(os.environ))
    raise AssertionError("os.execvpe returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
