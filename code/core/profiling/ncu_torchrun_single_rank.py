"""Run one rank-local NCU process beside an unprofiled external peer.

The rank dispatcher is launched by one two-rank ``torchrun --no-python``
command. The selected rank runs NCU Application Replay over the generation
entrypoint in this module. The other rank invokes that same entrypoint once per
replay pass without placing itself or its workers below NCU.

The generation entrypoint runs the unchanged stock generation helper, then
uses durable per-rank files to keep the next generation from starting until
both ranks have completed the current one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.profiling.ncu_torchrun_generation import (
    _read_rank_environment,
    load_generation_config,
)
from core.profiling.profiler_config import MINIMAL_METRICS


SINGLE_RANK_APPLICATION_CONFIG_SCHEMA = (
    "aisp.ncu-torchrun-single-rank-application.v1"
)
SINGLE_RANK_PEER_DISPATCH_SCHEMA = (
    "aisp.ncu-torchrun-single-rank-peer-dispatch.v1"
)
SINGLE_RANK_GENERATION_GATE_SCHEMA = (
    "aisp.ncu-torchrun-single-rank-generation-gate.v1"
)
EXPECTED_APPLICATION_PASSES = 9
GENERATION_GATE_TIMEOUT_SECONDS = 60.0

_SINGLE_RANK_MODULE = "core.profiling.ncu_torchrun_single_rank"
_GENERATION_MODULE = "core.profiling.ncu_torchrun_generation"
_CAPTURE_OWNER_ENV = "AISP_NCU_TORCHRUN_CAPTURE_ID"
_POLL_SECONDS = 0.05
_OWNER_ID = re.compile(r"^[0-9a-f]{32}$")
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "ncu_prefix",
        "generation_worker_argv",
        "generation_config_path",
        "output_dir",
        "world_size",
        "target_local_rank",
        "expected_application_passes",
    }
)
_GATE_KEYS = frozenset(
    {
        "schema",
        "capture_owner_id",
        "generation_config_path",
        "generation_config_sha256",
        "expected_passes",
        "generation",
        "rank",
        "counter_before",
        "counter_after",
        "stock_generation_argv",
        "stock_generation_returncode",
        "wrapper_pid",
        "completed_time_ns",
    }
)
_PEER_DISPATCH_KEYS = frozenset(
    {
        "schema",
        "state",
        "capture_owner_id",
        "pid",
        "local_rank",
        "target_local_rank",
        "role",
        "rank_config_path",
        "rank_config_sha256",
        "generation_config_path",
        "generation_config_sha256",
        "generation_worker_argv",
        "expected_passes",
        "started_time_ns",
        "completed_time_ns",
        "invocations",
        "errors",
    }
)


def _require_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Artifact write made no progress")
        remaining = remaining[written:]


def _write_json(path: Path, value: Any) -> None:
    """Write one replaceable receipt without exposing partial JSON."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Receipt path must be a real file or absent: {path}")
    data = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing stale receipt temporary file: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _require_absent(path: Path, *, description: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing stale {description}: {path}")


def _single_option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, item in enumerate(argv) if item == name]
    if len(positions) != 1:
        raise ValueError(f"{name} must occur exactly once, got {len(positions)}")
    position = positions[0]
    if position + 1 == len(argv):
        raise ValueError(f"{name} requires a value")
    return argv[position + 1]


def _validate_ncu_prefix(prefix: Sequence[str]) -> None:
    expected_options = {
        "--clock-control": "none",
        "--target-processes": "all",
        "--replay-mode": "application",
        "--app-replay-mode": "strict",
        "--app-replay-match": "grid",
        "--filter-mode": "per-gpu",
        "--kill": "no",
        "--check-exit-code": "yes",
        "--metrics": ",".join(MINIMAL_METRICS),
    }
    if len(MINIMAL_METRICS) != 5 or len(set(MINIMAL_METRICS)) != 5:
        raise RuntimeError("Repository minimal NCU contract must contain exactly five metrics")
    for name, expected in expected_options.items():
        observed = _single_option(prefix, name)
        if observed != expected:
            raise ValueError(f"{name} must be {expected!r}, got {observed!r}")
    for forbidden in ("--devices", "--export", "--log-file"):
        if forbidden in prefix:
            raise ValueError(f"Single-rank NCU prefix must not contain {forbidden}")
    forbidden_coupling = [
        item
        for item in prefix
        if item.startswith("--communicator") or item == "--lockstep-kernel-launch"
    ]
    if forbidden_coupling:
        raise ValueError(
            "Single-rank NCU prefix contains rank-coupling options: "
            f"{forbidden_coupling}"
        )


def load_single_rank_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate one single-profiled-rank configuration."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Single-rank config must be a real file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Single-rank config must contain a JSON object")
    keys = frozenset(payload)
    if keys != _CONFIG_KEYS:
        raise ValueError(
            "Single-rank config fields differ from schema. "
            f"missing={sorted(_CONFIG_KEYS - keys)}, "
            f"unexpected={sorted(keys - _CONFIG_KEYS)}"
        )
    if payload["schema"] != SINGLE_RANK_APPLICATION_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported single-rank config schema: {payload['schema']!r}")
    if type(payload["world_size"]) is not int or payload["world_size"] != 2:
        raise ValueError("Single-rank config world_size must be exactly 2")
    target_local_rank = payload["target_local_rank"]
    if type(target_local_rank) is not int or target_local_rank not in (0, 1):
        raise ValueError("Single-rank config target_local_rank must be 0 or 1")
    if (
        type(payload["expected_application_passes"]) is not int
        or payload["expected_application_passes"] != EXPECTED_APPLICATION_PASSES
    ):
        raise ValueError(
            "Single-rank config expected_application_passes must be exactly "
            f"{EXPECTED_APPLICATION_PASSES}"
        )

    payload["ncu_prefix"] = _require_string_list(
        payload["ncu_prefix"], name="ncu_prefix"
    )
    _validate_ncu_prefix(payload["ncu_prefix"])
    payload["generation_worker_argv"] = _require_string_list(
        payload["generation_worker_argv"], name="generation_worker_argv"
    )

    output_dir = payload["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Single-rank config output_dir must be a non-empty string")
    resolved_output = Path(output_dir)
    if not resolved_output.is_absolute():
        raise ValueError("Single-rank config output_dir must be absolute")
    if resolved_output.is_symlink() or not resolved_output.is_dir():
        raise ValueError(f"Single-rank output must be a real directory: {resolved_output}")
    payload["output_dir"] = resolved_output

    raw_generation_path = payload["generation_config_path"]
    if not isinstance(raw_generation_path, str) or not raw_generation_path:
        raise ValueError("generation_config_path must be a non-empty string")
    generation_path = Path(raw_generation_path)
    if not generation_path.is_absolute():
        raise ValueError("generation_config_path must be absolute")
    if generation_path.is_symlink() or not generation_path.is_file():
        raise ValueError(f"Generation config must be a real file: {generation_path}")
    generation_sha256 = _sha256(generation_path)
    generation_config = load_generation_config(generation_path)
    if _sha256(generation_path) != generation_sha256:
        raise ValueError("Generation config changed while it was being loaded")
    if Path(generation_config["output_dir"]).resolve() != resolved_output.resolve():
        raise ValueError("Single-rank and generation configs name different output directories")
    if generation_config["expected_passes"] != EXPECTED_APPLICATION_PASSES:
        raise ValueError(
            "Single-rank generation config must require exactly "
            f"{EXPECTED_APPLICATION_PASSES} passes"
        )
    expected_generation_argv = (
        generation_config["worker_argv"][0],
        "-m",
        _SINGLE_RANK_MODULE,
        "--generation",
        str(generation_path),
    )
    if payload["generation_worker_argv"] != expected_generation_argv:
        raise ValueError(
            "Single-rank generation_worker_argv differs from the exact gated entrypoint"
        )
    payload["generation_config_path"] = generation_path
    return payload


def _counter_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"rank-{rank}-generation.counter"


def _read_counter(path: Path, *, allow_missing: bool) -> int:
    if path.is_symlink():
        raise ValueError(f"Generation counter must not be a symlink: {path}")
    if not path.exists():
        if allow_missing:
            return 0
        raise ValueError(f"Generation counter is missing: {path}")
    if not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"Generation counter must be a regular file: {path}")
    raw = path.read_bytes()
    if re.fullmatch(rb"(?:0|[1-9][0-9]*)\n", raw) is None:
        raise ValueError(f"Generation counter has invalid contents: {path}")
    return int(raw[:-1])


def _gate_path(output_dir: Path, generation: int, rank: int) -> Path:
    return output_dir / f"application-generation-gate-{generation}-rank-{rank}.json"


def _stock_generation_argv(
    generation_config: Mapping[str, Any], config_path: Path
) -> tuple[str, ...]:
    return (
        generation_config["worker_argv"][0],
        "-m",
        _GENERATION_MODULE,
        str(config_path),
    )


def _publish_gate(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete gate JSON with no overwrite window."""

    _require_absent(path, description="generation gate marker")
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    _require_absent(temporary, description="generation gate temporary file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    linked = False
    try:
        os.link(temporary, path, follow_symlinks=False)
        linked = True
    finally:
        try:
            temporary.unlink()
        finally:
            _fsync_directory(path.parent)
    if not linked:
        raise RuntimeError(f"Generation gate marker was not published: {path}")


def load_generation_gate(
    path: Path,
    *,
    capture_owner_id: str,
    config_path: Path,
    config_sha256: str,
    stock_argv: Sequence[str],
    generation: int,
    rank: int,
) -> dict[str, Any]:
    """Load one gate marker and verify its complete provenance."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Generation gate marker must be a real file: {path}")
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"Generation gate marker must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Generation gate marker must contain a JSON object: {path}")
    keys = frozenset(payload)
    if keys != _GATE_KEYS:
        raise ValueError(
            "Generation gate marker fields differ from schema. "
            f"path={path}, missing={sorted(_GATE_KEYS - keys)}, "
            f"unexpected={sorted(keys - _GATE_KEYS)}"
        )
    integer_fields = (
        "expected_passes",
        "generation",
        "rank",
        "counter_before",
        "counter_after",
        "stock_generation_returncode",
    )
    invalid_integer_fields = [
        name for name in integer_fields if type(payload[name]) is not int
    ]
    if invalid_integer_fields:
        raise ValueError(
            f"Generation gate marker integer fields have invalid types for {path}: "
            f"{invalid_integer_fields}"
        )
    expected_values: dict[str, Any] = {
        "schema": SINGLE_RANK_GENERATION_GATE_SCHEMA,
        "capture_owner_id": capture_owner_id,
        "generation_config_path": str(config_path.resolve()),
        "generation_config_sha256": config_sha256,
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "generation": generation,
        "rank": rank,
        "counter_before": generation,
        "counter_after": generation + 1,
        "stock_generation_argv": list(stock_argv),
        "stock_generation_returncode": 0,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in expected_values.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Generation gate marker provenance differs for {path}: {mismatches}"
        )
    if type(payload["wrapper_pid"]) is not int or payload["wrapper_pid"] <= 1:
        raise ValueError(f"Generation gate marker has an invalid wrapper PID: {path}")
    if (
        type(payload["completed_time_ns"]) is not int
        or payload["completed_time_ns"] <= 0
    ):
        raise ValueError(f"Generation gate marker has an invalid completion time: {path}")
    return payload


def _expected_gate_paths(output_dir: Path) -> set[Path]:
    return {
        _gate_path(output_dir, generation, rank)
        for generation in range(EXPECTED_APPLICATION_PASSES)
        for rank in (0, 1)
    }


def _validate_gate_history(
    *,
    output_dir: Path,
    generation: int,
    rank: int,
    capture_owner_id: str,
    config_path: Path,
    config_sha256: str,
    stock_argv: Sequence[str],
) -> None:
    expected_paths = _expected_gate_paths(output_dir)
    observed_paths = set(
        output_dir.glob("application-generation-gate-*-rank-*.json")
    )
    unexpected = sorted(str(path) for path in observed_paths - expected_paths)
    if unexpected:
        raise ValueError(f"Unexpected generation gate marker paths: {unexpected}")
    for prior_generation in range(generation):
        for prior_rank in (0, 1):
            load_generation_gate(
                _gate_path(output_dir, prior_generation, prior_rank),
                capture_owner_id=capture_owner_id,
                config_path=config_path,
                config_sha256=config_sha256,
                stock_argv=stock_argv,
                generation=prior_generation,
                rank=prior_rank,
            )
    own_path = _gate_path(output_dir, generation, rank)
    _require_absent(own_path, description="current rank generation gate marker")
    peer_rank = 1 - rank
    peer_path = _gate_path(output_dir, generation, peer_rank)
    if peer_path.exists() or peer_path.is_symlink():
        load_generation_gate(
            peer_path,
            capture_owner_id=capture_owner_id,
            config_path=config_path,
            config_sha256=config_sha256,
            stock_argv=stock_argv,
            generation=generation,
            rank=peer_rank,
        )
    for future_generation in range(generation + 1, EXPECTED_APPLICATION_PASSES):
        for future_rank in (0, 1):
            _require_absent(
                _gate_path(output_dir, future_generation, future_rank),
                description="future generation gate marker",
            )


def _wait_for_peer_gate(
    *,
    output_dir: Path,
    generation: int,
    rank: int,
    capture_owner_id: str,
    config_path: Path,
    config_sha256: str,
    stock_argv: Sequence[str],
    timeout_seconds: float = GENERATION_GATE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("Generation gate timeout must be positive and finite")
    peer_rank = 1 - rank
    peer_path = _gate_path(output_dir, generation, peer_rank)
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        if peer_path.exists() or peer_path.is_symlink():
            return load_generation_gate(
                peer_path,
                capture_owner_id=capture_owner_id,
                config_path=config_path,
                config_sha256=config_sha256,
                stock_argv=stock_argv,
                generation=generation,
                rank=peer_rank,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {float(timeout_seconds):g} seconds waiting "
                f"for rank {peer_rank} generation {generation} completion"
            )
        time.sleep(_POLL_SECONDS)


def validate_generation_gate_artifacts(
    *,
    output_dir: Path,
    capture_owner_id: str,
    generation_config_path: Path,
) -> list[dict[str, Any]]:
    """Validate the complete paired gate set and return hashed artifacts."""

    if _OWNER_ID.fullmatch(capture_owner_id) is None:
        raise ValueError("Capture owner ID must contain 32 lowercase hexadecimal characters")
    config_sha256 = _sha256(generation_config_path)
    generation_config = load_generation_config(generation_config_path)
    if _sha256(generation_config_path) != config_sha256:
        raise ValueError("Generation config changed while gate artifacts were validated")
    if generation_config["expected_passes"] != EXPECTED_APPLICATION_PASSES:
        raise ValueError(
            f"Gate validation requires exactly {EXPECTED_APPLICATION_PASSES} passes"
        )
    if Path(generation_config["output_dir"]).resolve() != output_dir.resolve():
        raise ValueError("Generation gate output directory differs from its config")
    stock_argv = _stock_generation_argv(generation_config, generation_config_path)
    expected_paths = _expected_gate_paths(output_dir)
    observed_paths = set(
        output_dir.glob("application-generation-gate-*-rank-*.json")
    )
    if observed_paths != expected_paths:
        raise ValueError(
            "Generation gate marker set differs. "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )

    artifacts: list[dict[str, Any]] = []
    wrapper_pids: set[int] = set()
    for generation in range(EXPECTED_APPLICATION_PASSES):
        for rank in (0, 1):
            path = _gate_path(output_dir, generation, rank)
            payload = load_generation_gate(
                path,
                capture_owner_id=capture_owner_id,
                config_path=generation_config_path,
                config_sha256=config_sha256,
                stock_argv=stock_argv,
                generation=generation,
                rank=rank,
            )
            wrapper_pid = payload["wrapper_pid"]
            if wrapper_pid in wrapper_pids:
                raise ValueError(
                    f"Generation gate reused wrapper process ID {wrapper_pid}"
                )
            wrapper_pids.add(wrapper_pid)
            raw = path.read_bytes()
            path_stat = path.stat()
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "mtime_ns": path_stat.st_mtime_ns,
                    "payload": payload,
                }
            )
    if _sha256(generation_config_path) != config_sha256:
        raise ValueError("Generation config changed while gate artifacts were validated")
    return artifacts


def _normalized_failure_returncode(returncode: int) -> int:
    return returncode if 1 <= returncode <= 255 else 1


def run_gated_generation(
    config_path: Path, environment: Mapping[str, str]
) -> int:
    """Run one stock generation and wait for its paired rank completion."""

    if not config_path.is_absolute():
        raise ValueError("Generation config path must be absolute")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(f"Generation config must be a real file: {config_path}")
    config_sha256 = _sha256(config_path)
    config = load_generation_config(config_path)
    if _sha256(config_path) != config_sha256:
        raise ValueError("Generation config changed while it was being loaded")
    if config["expected_passes"] != EXPECTED_APPLICATION_PASSES:
        raise ValueError(
            f"Gated generation requires exactly {EXPECTED_APPLICATION_PASSES} passes"
        )
    rank = _read_rank_environment(environment)
    capture_owner_id = environment.get(_CAPTURE_OWNER_ENV, "")
    if _OWNER_ID.fullmatch(capture_owner_id) is None:
        raise ValueError(
            f"{_CAPTURE_OWNER_ENV} must be the active 32-character lowercase capture ID"
        )

    output_dir = Path(config["output_dir"])
    counter_path = _counter_path(output_dir, rank)
    counter_existed = counter_path.exists() or counter_path.is_symlink()
    generation = _read_counter(counter_path, allow_missing=True)
    if counter_existed and generation == 0:
        raise ValueError(f"Refusing a stale zero generation counter: {counter_path}")
    if not 0 <= generation < EXPECTED_APPLICATION_PASSES:
        raise ValueError(
            f"Rank {rank} generation must be in 0..{EXPECTED_APPLICATION_PASSES - 1}, "
            f"got {generation}"
        )
    stock_argv = _stock_generation_argv(config, config_path)
    _validate_gate_history(
        output_dir=output_dir,
        generation=generation,
        rank=rank,
        capture_owner_id=capture_owner_id,
        config_path=config_path,
        config_sha256=config_sha256,
        stock_argv=stock_argv,
    )

    completed = subprocess.run(list(stock_argv), env=dict(environment), check=False)
    if completed.returncode != 0:
        print(
            f"single-rank stock generation helper returned {completed.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return _normalized_failure_returncode(completed.returncode)
    if _sha256(config_path) != config_sha256:
        raise ValueError("Generation config changed while the stock helper was running")
    observed_counter = _read_counter(counter_path, allow_missing=False)
    if observed_counter != generation + 1:
        raise ValueError(
            f"Rank {rank} generation counter must advance from {generation} to "
            f"{generation + 1}, got {observed_counter}"
        )

    marker_path = _gate_path(output_dir, generation, rank)
    marker = {
        "schema": SINGLE_RANK_GENERATION_GATE_SCHEMA,
        "capture_owner_id": capture_owner_id,
        "generation_config_path": str(config_path.resolve()),
        "generation_config_sha256": config_sha256,
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "generation": generation,
        "rank": rank,
        "counter_before": generation,
        "counter_after": observed_counter,
        "stock_generation_argv": list(stock_argv),
        "stock_generation_returncode": completed.returncode,
        "wrapper_pid": os.getpid(),
        "completed_time_ns": time.time_ns(),
    }
    _publish_gate(marker_path, marker)
    load_generation_gate(
        marker_path,
        capture_owner_id=capture_owner_id,
        config_path=config_path,
        config_sha256=config_sha256,
        stock_argv=stock_argv,
        generation=generation,
        rank=rank,
    )
    _wait_for_peer_gate(
        output_dir=output_dir,
        generation=generation,
        rank=rank,
        capture_owner_id=capture_owner_id,
        config_path=config_path,
        config_sha256=config_sha256,
        stock_argv=stock_argv,
    )
    if _sha256(config_path) != config_sha256:
        raise ValueError("Generation config changed while waiting at the rank gate")
    if _read_counter(counter_path, allow_missing=False) != generation + 1:
        raise ValueError("Generation counter changed before this rank left the gate")
    return 0


def _peer_dispatch_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"rank-{rank}-peer-dispatch.json"


def validate_peer_dispatch_artifact(
    path: Path,
    *,
    rank_config_path: Path,
    capture_owner_id: str,
) -> dict[str, Any]:
    """Validate the completed unprofiled-peer dispatch receipt."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Peer dispatch receipt must be a real file: {path}")
    receipt_sha256 = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Peer dispatch receipt must contain a JSON object")
    keys = frozenset(payload)
    if keys != _PEER_DISPATCH_KEYS:
        raise ValueError(
            "Peer dispatch fields differ from schema. "
            f"missing={sorted(_PEER_DISPATCH_KEYS - keys)}, "
            f"unexpected={sorted(keys - _PEER_DISPATCH_KEYS)}"
        )
    rank_config_sha256 = _sha256(rank_config_path)
    config = load_single_rank_config(rank_config_path)
    if _sha256(rank_config_path) != rank_config_sha256:
        raise ValueError("Single-rank config changed while peer dispatch was validated")
    generation_config_path = Path(config["generation_config_path"])
    generation_config_sha256 = _sha256(generation_config_path)
    target_local_rank = config["target_local_rank"]
    peer_rank = 1 - target_local_rank
    expected_values: dict[str, Any] = {
        "schema": SINGLE_RANK_PEER_DISPATCH_SCHEMA,
        "state": "complete",
        "capture_owner_id": capture_owner_id,
        "local_rank": peer_rank,
        "target_local_rank": target_local_rank,
        "role": "unprofiled_peer",
        "rank_config_path": str(rank_config_path.resolve()),
        "rank_config_sha256": rank_config_sha256,
        "generation_config_path": str(generation_config_path.resolve()),
        "generation_config_sha256": generation_config_sha256,
        "generation_worker_argv": list(config["generation_worker_argv"]),
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "invocations": [
            {
                "invocation": invocation,
                "returncode": 0,
                "observed_generation_counter": invocation + 1,
            }
            for invocation in range(EXPECTED_APPLICATION_PASSES)
        ],
        "errors": [],
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in expected_values.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Peer dispatch provenance differs: {mismatches}")
    for name in ("pid", "started_time_ns", "completed_time_ns"):
        if type(payload[name]) is not int or payload[name] <= 0:
            raise ValueError(f"Peer dispatch {name} must be a positive integer")
    if payload["completed_time_ns"] < payload["started_time_ns"]:
        raise ValueError("Peer dispatch completed before it started")
    if _sha256(path) != receipt_sha256:
        raise ValueError("Peer dispatch receipt changed while it was validated")
    if _sha256(rank_config_path) != rank_config_sha256:
        raise ValueError("Single-rank config changed while peer dispatch was validated")
    if _sha256(generation_config_path) != generation_config_sha256:
        raise ValueError("Generation config changed while peer dispatch was validated")
    return payload


def _target_command(config: Mapping[str, Any], target_local_rank: int) -> list[str]:
    output_dir = Path(config["output_dir"])
    return [
        *config["ncu_prefix"],
        "--log-file",
        str(output_dir / f"rank-{target_local_rank}.ncu.log"),
        "--devices",
        str(target_local_rank),
        "--export",
        str(output_dir / f"rank-{target_local_rank}.ncu-rep"),
        *config["generation_worker_argv"],
    ]


def expected_target_command(config: Mapping[str, Any]) -> list[str]:
    """Return the exact target NCU command for a validated config."""

    return _target_command(config, int(config["target_local_rank"]))


def run_single_rank_dispatch(
    config_path: Path, environment: Mapping[str, str]
) -> int:
    """Run the target NCU process or the independent nine-generation peer."""

    if not config_path.is_absolute():
        raise ValueError("Single-rank config path must be absolute")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(f"Single-rank config must be a real file: {config_path}")
    config_sha256 = _sha256(config_path)
    config = load_single_rank_config(config_path)
    if _sha256(config_path) != config_sha256:
        raise ValueError("Single-rank config changed while it was being loaded")
    local_rank = _read_rank_environment(environment)
    target_local_rank = int(config["target_local_rank"])
    output_dir = Path(config["output_dir"])
    capture_owner_id = environment.get(_CAPTURE_OWNER_ENV, "")
    if _OWNER_ID.fullmatch(capture_owner_id) is None:
        raise ValueError(
            f"{_CAPTURE_OWNER_ENV} must be the active 32-character lowercase capture ID"
        )

    own_counter = _counter_path(output_dir, local_rank)
    _require_absent(own_counter, description="generation counter")
    report_path = output_dir / f"rank-{local_rank}.ncu-rep"
    log_path = output_dir / f"rank-{local_rank}.ncu.log"
    argv_path = output_dir / f"rank-{local_rank}-argv.json"
    _require_absent(report_path, description="rank NCU report")
    _require_absent(log_path, description="rank NCU log")
    _require_absent(argv_path, description="rank NCU argv receipt")

    if local_rank == target_local_rank:
        command = _target_command(config, target_local_rank)
        _write_json(argv_path, command)
        if _sha256(config_path) != config_sha256:
            raise ValueError("Single-rank config changed before the target NCU launch")
        os.execvpe(command[0], command, dict(environment))
        raise AssertionError("os.execvpe returned unexpectedly")

    peer_receipt_path = _peer_dispatch_path(output_dir, local_rank)
    _require_absent(peer_receipt_path, description="peer dispatch receipt")
    generation_config_path = Path(config["generation_config_path"])
    generation_config_sha256 = _sha256(generation_config_path)
    started_time_ns = time.time_ns()
    invocations: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "schema": SINGLE_RANK_PEER_DISPATCH_SCHEMA,
        "state": "running",
        "capture_owner_id": capture_owner_id,
        "pid": os.getpid(),
        "local_rank": local_rank,
        "target_local_rank": target_local_rank,
        "role": "unprofiled_peer",
        "rank_config_path": str(config_path.resolve()),
        "rank_config_sha256": config_sha256,
        "generation_config_path": str(generation_config_path.resolve()),
        "generation_config_sha256": generation_config_sha256,
        "generation_worker_argv": list(config["generation_worker_argv"]),
        "expected_passes": EXPECTED_APPLICATION_PASSES,
        "started_time_ns": started_time_ns,
        "completed_time_ns": 0,
        "invocations": invocations,
        "errors": [],
    }
    _write_json(peer_receipt_path, receipt)

    for invocation in range(EXPECTED_APPLICATION_PASSES):
        completed = subprocess.run(
            list(config["generation_worker_argv"]),
            env=dict(environment),
            check=False,
        )
        try:
            observed_counter = _read_counter(own_counter, allow_missing=False)
        except ValueError as exc:
            observed_counter = None
            counter_error = str(exc)
        else:
            counter_error = None
        event: dict[str, Any] = {
            "invocation": invocation,
            "returncode": completed.returncode,
            "observed_generation_counter": observed_counter,
        }
        invocations.append(event)
        expected_counter = invocation + 1
        errors: list[str] = []
        if completed.returncode != 0:
            errors.append(
                f"generation invocation {invocation} returned {completed.returncode}"
            )
        if counter_error is not None:
            errors.append(counter_error)
        elif observed_counter != expected_counter:
            errors.append(
                f"rank {local_rank} counter after invocation {invocation} must be "
                f"{expected_counter}, got {observed_counter!r}"
            )
        if errors:
            receipt.update(
                state="failed",
                completed_time_ns=time.time_ns(),
                errors=errors,
            )
            _write_json(peer_receipt_path, receipt)
            return _normalized_failure_returncode(completed.returncode)
        _write_json(peer_receipt_path, receipt)

    if _sha256(config_path) != config_sha256:
        raise ValueError("Single-rank config changed while the peer was running")
    if _sha256(generation_config_path) != generation_config_sha256:
        raise ValueError("Generation config changed while the peer was running")
    if report_path.exists() or report_path.is_symlink():
        raise ValueError("Unprofiled peer unexpectedly produced an NCU report")
    if log_path.exists() or log_path.is_symlink():
        raise ValueError("Unprofiled peer unexpectedly produced an NCU log")
    if argv_path.exists() or argv_path.is_symlink():
        raise ValueError("Unprofiled peer unexpectedly produced an NCU argv receipt")
    receipt.update(
        state="complete",
        completed_time_ns=time.time_ns(),
        errors=[],
    )
    _write_json(peer_receipt_path, receipt)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one rank-local NCU Application Replay process beside an unprofiled "
            "peer, or run one gated stock generation."
        )
    )
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.generation:
            return run_gated_generation(args.config, os.environ)
        return run_single_rank_dispatch(args.config, os.environ)
    except (
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ncu-torchrun single-rank error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
