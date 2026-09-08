"""Fail-closed runtime-provenance transport for torchrun workers.

Each opted-in worker captures its own runtime after the target succeeds. Large
logical records are split into bounded frames, and each frame is written with
one ``os.write`` call no larger than ``PIPE_BUF`` so concurrent ranks cannot
interleave bytes. Coordinators retain the raw stream and use ``clean_stdout``
for ordinary timing and diagnostic parsing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import select
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from core.benchmark.run_manifest import (
    RunManifest,
    RuntimeParityTarget,
    RuntimeProvenance,
    RuntimeProvenanceParityError,
    capture_runtime_provenance,
    require_runtime_provenance_parity,
)

TORCHRUN_RUNTIME_PROVENANCE_SCHEMA = "aisp.torchrun-runtime-provenance.v1"
TORCHRUN_RUNTIME_PROVENANCE_PREFIX = "__AISP_TORCHRUN_RUNTIME_PROVENANCE_V1__="

_FRAME_KEYS = frozenset({"s", "r", "lr", "ws", "lws", "p", "i", "n", "h", "d"})
_SHA256_HEX_LENGTH = 64


class TorchrunRuntimeProvenanceError(RuntimeError):
    """Raised when a torchrun runtime receipt is absent, ambiguous, or invalid."""


@dataclass(frozen=True)
class TorchrunRuntimeProvenanceParseResult:
    """Validated worker snapshots plus raw and protocol-free stdout."""

    raw_stdout: str
    clean_stdout: str
    snapshots_by_local_rank: dict[int, RuntimeProvenance]
    execution_process_ids: dict[int, int]

    @property
    def primary_snapshot(self) -> RuntimeProvenance:
        """Return local rank zero's snapshot after strict rank validation."""

        try:
            return self.snapshots_by_local_rank[0]
        except KeyError as exc:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance has no local-rank-zero snapshot"
            ) from exc


@dataclass(frozen=True)
class _FrameMetadata:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    process_id: int
    frame_count: int
    digest: str


def _require_plain_int(value: Any, *, field_name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise TorchrunRuntimeProvenanceError(
            f"Torchrun runtime provenance {field_name} must be an integer >= {minimum}, "
            f"got {value!r}"
        )
    return value


def _environment_int(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise TorchrunRuntimeProvenanceError(
            f"Invalid integer for torchrun runtime provenance {name}: {raw!r}"
        ) from exc
    return _require_plain_int(value, field_name=name, minimum=minimum)


def _pipe_buf(fd: int) -> int:
    try:
        value = int(os.fpathconf(fd, "PC_PIPE_BUF"))
    except (OSError, ValueError):
        value = int(getattr(select, "PIPE_BUF", 512))
    if value <= 0:
        raise TorchrunRuntimeProvenanceError(
            f"Invalid PIPE_BUF={value} for torchrun runtime provenance stdout"
        )
    return value


def _validate_rank_metadata(
    *, rank: int, local_rank: int, world_size: int, local_world_size: int
) -> None:
    if rank >= world_size or local_rank >= local_world_size:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance rank metadata is outside its declared world size"
        )
    if local_world_size > world_size:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance local_world_size cannot exceed world_size"
        )


def _serialize_frame(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    process_id: int,
    frame_index: int,
    frame_count: int,
    digest: str,
    data: str,
) -> bytes:
    payload = {
        "s": TORCHRUN_RUNTIME_PROVENANCE_SCHEMA,
        "r": rank,
        "lr": local_rank,
        "ws": world_size,
        "lws": local_world_size,
        "p": process_id,
        "i": frame_index,
        "n": frame_count,
        "h": digest,
        "d": data,
    }
    return (
        TORCHRUN_RUNTIME_PROVENANCE_PREFIX
        + json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def format_torchrun_runtime_provenance_frames(
    snapshot: RuntimeProvenance,
    *,
    local_rank: int,
    rank: int,
    world_size: int,
    local_world_size: int,
    pipe_buf: int,
) -> tuple[bytes, ...]:
    """Encode one complete worker snapshot as independently atomic frames."""

    if not isinstance(snapshot, RuntimeProvenance):
        raise TypeError(f"snapshot must be RuntimeProvenance, got {type(snapshot).__name__}")
    local_rank = _require_plain_int(local_rank, field_name="local_rank", minimum=0)
    rank = _require_plain_int(rank, field_name="rank", minimum=0)
    world_size = _require_plain_int(world_size, field_name="world_size", minimum=1)
    local_world_size = _require_plain_int(
        local_world_size, field_name="local_world_size", minimum=1
    )
    pipe_buf = _require_plain_int(pipe_buf, field_name="pipe_buf", minimum=1)
    process_id = _require_plain_int(snapshot.process_id, field_name="process_id", minimum=1)
    _validate_rank_metadata(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
    )

    runtime_json = json.dumps(
        snapshot.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(runtime_json).hexdigest()
    encoded_runtime = base64.b64encode(runtime_json).decode("ascii")

    frame_count = 1
    while True:
        empty_frame = _serialize_frame(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            process_id=process_id,
            frame_index=frame_count - 1,
            frame_count=frame_count,
            digest=digest,
            data="",
        )
        chunk_capacity = pipe_buf - len(empty_frame)
        if chunk_capacity <= 0:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance PIPE_BUF cannot hold frame metadata: "
                f"{pipe_buf} bytes available, {len(empty_frame)} required"
            )
        required_frames = max(1, (len(encoded_runtime) + chunk_capacity - 1) // chunk_capacity)
        if required_frames <= frame_count:
            break
        frame_count = required_frames

    base_chunk_size, larger_chunk_count = divmod(len(encoded_runtime), frame_count)
    frames: list[bytes] = []
    offset = 0
    for frame_index in range(frame_count):
        chunk_size = base_chunk_size + (frame_index < larger_chunk_count)
        data = encoded_runtime[offset : offset + chunk_size]
        offset += chunk_size
        frame = _serialize_frame(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            process_id=process_id,
            frame_index=frame_index,
            frame_count=frame_count,
            digest=digest,
            data=data,
        )
        if len(frame) > pipe_buf:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance frame exceeds atomic stdout PIPE_BUF: "
                f"frame {frame_index + 1}/{frame_count} is {len(frame)} > {pipe_buf} bytes"
            )
        frames.append(frame)
    if offset != len(encoded_runtime):
        raise AssertionError("Torchrun runtime provenance frame partition was incomplete")
    return tuple(frames)


def emit_torchrun_runtime_provenance(*, local_rank: int) -> RuntimeProvenance:
    """Capture and atomically emit this worker's complete runtime provenance."""

    local_rank = _require_plain_int(local_rank, field_name="local_rank", minimum=0)
    rank = _environment_int("RANK", default=local_rank, minimum=0)
    world_size = _environment_int("WORLD_SIZE", default=1, minimum=1)
    local_world_size = _environment_int("LOCAL_WORLD_SIZE", default=1, minimum=1)

    snapshot = capture_runtime_provenance()
    actual_process_id = os.getpid()
    if snapshot.process_id != actual_process_id:
        raise TorchrunRuntimeProvenanceError(
            "Captured torchrun runtime provenance process ID does not identify the "
            f"emitting worker: captured {snapshot.process_id}, actual {actual_process_id}"
        )
    stdout_fd = sys.stdout.fileno()
    frames = format_torchrun_runtime_provenance_frames(
        snapshot,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        pipe_buf=_pipe_buf(stdout_fd),
    )
    sys.stdout.flush()
    for frame_index, frame in enumerate(frames):
        written = os.write(stdout_fd, frame)
        if written != len(frame):
            raise TorchrunRuntimeProvenanceError(
                "Incomplete torchrun runtime provenance stdout write: "
                f"frame {frame_index + 1}/{len(frames)} wrote {written} of {len(frame)} bytes"
            )
    return snapshot


def _runtime_only_manifest(snapshot: RuntimeProvenance) -> RunManifest:
    """Build the narrow carrier accepted by the canonical parity comparator."""

    return RunManifest.model_construct(runtime_provenance=snapshot)


def _require_rank_runtime_consistency(
    snapshots_by_local_rank: Mapping[int, RuntimeProvenance],
    *,
    target: RuntimeParityTarget,
) -> None:
    ordered_ranks = sorted(snapshots_by_local_rank)
    if not ordered_ranks:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance contains no worker snapshots"
        )

    reference_rank = ordered_ranks[0]
    reference = _runtime_only_manifest(snapshots_by_local_rank[reference_rank])
    try:
        # A self-comparison also rejects incomplete required provenance for a
        # single-rank launch instead of postponing the failure to pair gating.
        require_runtime_provenance_parity(reference, reference, target=target)
        for local_rank in ordered_ranks[1:]:
            candidate = _runtime_only_manifest(snapshots_by_local_rank[local_rank])
            require_runtime_provenance_parity(reference, candidate, target=target)
    except RuntimeProvenanceParityError as exc:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun worker runtime provenance is incomplete or inconsistent "
            f"relative to local rank {reference_rank}: {exc}"
        ) from exc


def _parse_frame_payload(encoded_payload: str, *, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(encoded_payload)
    except json.JSONDecodeError as exc:
        raise TorchrunRuntimeProvenanceError(
            f"Malformed torchrun runtime provenance JSON on stdout line {line_number}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise TorchrunRuntimeProvenanceError(
            f"Torchrun runtime provenance line {line_number} must contain a JSON object"
        )
    payload_keys = frozenset(payload)
    if payload_keys != _FRAME_KEYS:
        missing = sorted(_FRAME_KEYS - payload_keys)
        unexpected = sorted(payload_keys - _FRAME_KEYS)
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance frame fields differ from schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if payload["s"] != TORCHRUN_RUNTIME_PROVENANCE_SCHEMA:
        raise TorchrunRuntimeProvenanceError(
            f"Unsupported torchrun runtime provenance schema {payload['s']!r}"
        )
    return payload


def _validate_digest(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TorchrunRuntimeProvenanceError(
            f"Torchrun runtime provenance digest must be lowercase SHA-256 hex, got {value!r}"
        )
    return value


def _decode_snapshot(*, metadata: _FrameMetadata, encoded_chunks: list[str]) -> RuntimeProvenance:
    encoded_runtime = "".join(encoded_chunks)
    try:
        runtime_json = base64.b64decode(encoded_runtime.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise TorchrunRuntimeProvenanceError(
            f"Invalid base64 runtime provenance payload for local rank {metadata.local_rank}"
        ) from exc
    actual_digest = hashlib.sha256(runtime_json).hexdigest()
    if not hmac.compare_digest(actual_digest, metadata.digest):
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance digest mismatch for local rank " f"{metadata.local_rank}"
        )
    try:
        runtime_payload = json.loads(runtime_json)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TorchrunRuntimeProvenanceError(
            f"Malformed runtime provenance payload for local rank {metadata.local_rank}: {exc}"
        ) from exc
    try:
        snapshot = RuntimeProvenance.model_validate(runtime_payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise TorchrunRuntimeProvenanceError(
            f"Invalid runtime provenance snapshot for local rank {metadata.local_rank}: {exc}"
        ) from exc
    if snapshot.process_id <= 0:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance requires an actual process_id > 0; "
            f"local rank {metadata.local_rank} reported {snapshot.process_id}"
        )
    if snapshot.process_id != metadata.process_id:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance process ID differs between frame metadata and "
            f"snapshot for local rank {metadata.local_rank}"
        )
    return snapshot


def parse_torchrun_runtime_provenance_stdout(
    raw_stdout: str,
    *,
    expected_local_ranks: Collection[int],
    target: RuntimeParityTarget,
) -> TorchrunRuntimeProvenanceParseResult:
    """Reassemble and validate opted-in worker receipts from captured stdout.

    Every recognized protocol frame is removed from ``clean_stdout``. Missing,
    duplicate, partial, malformed, reordered, unexpected, or cross-rank-
    inconsistent records fail closed.
    """

    if not isinstance(raw_stdout, str):
        raise TypeError(f"raw_stdout must be str, got {type(raw_stdout).__name__}")
    expected_sequence = tuple(expected_local_ranks)
    if not expected_sequence:
        raise ValueError("expected_local_ranks must contain at least local rank zero")
    if any(type(rank) is not int or rank < 0 for rank in expected_sequence):
        raise ValueError("expected_local_ranks must contain nonnegative plain integers")
    if len(set(expected_sequence)) != len(expected_sequence):
        raise ValueError("expected_local_ranks contains duplicates")
    expected = set(expected_sequence)

    clean_parts: list[str] = []
    frame_groups: dict[int, tuple[_FrameMetadata, list[str]]] = {}
    for line_number, line in enumerate(raw_stdout.splitlines(keepends=True), start=1):
        content = line.rstrip("\r\n")
        if TORCHRUN_RUNTIME_PROVENANCE_PREFIX not in content:
            clean_parts.append(line)
            continue
        if not line.endswith("\n"):
            raise TorchrunRuntimeProvenanceError(
                f"Incomplete torchrun runtime provenance frame on stdout line {line_number}"
            )
        if not content.startswith(TORCHRUN_RUNTIME_PROVENANCE_PREFIX):
            raise TorchrunRuntimeProvenanceError(
                "Malformed torchrun runtime provenance prefix placement on stdout "
                f"line {line_number}"
            )
        payload = _parse_frame_payload(
            content[len(TORCHRUN_RUNTIME_PROVENANCE_PREFIX) :],
            line_number=line_number,
        )

        local_rank = _require_plain_int(payload["lr"], field_name="local_rank", minimum=0)
        rank = _require_plain_int(payload["r"], field_name="rank", minimum=0)
        world_size = _require_plain_int(payload["ws"], field_name="world_size", minimum=1)
        local_world_size = _require_plain_int(
            payload["lws"], field_name="local_world_size", minimum=1
        )
        process_id = _require_plain_int(payload["p"], field_name="process_id", minimum=1)
        frame_index = _require_plain_int(payload["i"], field_name="frame_index", minimum=0)
        frame_count = _require_plain_int(payload["n"], field_name="frame_count", minimum=1)
        digest = _validate_digest(payload["h"])
        data = payload["d"]
        if not isinstance(data, str):
            raise TorchrunRuntimeProvenanceError(
                f"Torchrun runtime provenance frame data must be a string, got {type(data).__name__}"
            )
        _validate_rank_metadata(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        if frame_index >= frame_count:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance frame index is outside its declared frame count"
            )

        metadata = _FrameMetadata(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            process_id=process_id,
            frame_count=frame_count,
            digest=digest,
        )
        existing = frame_groups.get(local_rank)
        if existing is None:
            if frame_index != 0:
                raise TorchrunRuntimeProvenanceError(
                    "Torchrun runtime provenance frame sequence must start at zero for "
                    f"local rank {local_rank}, got {frame_index}"
                )
            frame_groups[local_rank] = (metadata, [data])
            continue

        existing_metadata, encoded_chunks = existing
        if metadata != existing_metadata:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance frame metadata changed within local rank "
                f"{local_rank}"
            )
        expected_frame_index = len(encoded_chunks)
        if frame_index < expected_frame_index:
            raise TorchrunRuntimeProvenanceError(
                "Duplicate torchrun runtime provenance frame "
                f"{frame_index} for local rank {local_rank}"
            )
        if frame_index != expected_frame_index:
            raise TorchrunRuntimeProvenanceError(
                "Torchrun runtime provenance frame sequence is missing or reordered for "
                f"local rank {local_rank}: expected {expected_frame_index}, got {frame_index}"
            )
        encoded_chunks.append(data)

    observed = set(frame_groups)
    if observed != expected:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance rank set mismatch; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )

    records: dict[int, tuple[_FrameMetadata, RuntimeProvenance]] = {}
    for local_rank, (metadata, encoded_chunks) in frame_groups.items():
        if len(encoded_chunks) != metadata.frame_count:
            raise TorchrunRuntimeProvenanceError(
                "Incomplete torchrun runtime provenance frame sequence for local rank "
                f"{local_rank}: expected {metadata.frame_count}, observed {len(encoded_chunks)}"
            )
        records[local_rank] = (
            metadata,
            _decode_snapshot(metadata=metadata, encoded_chunks=encoded_chunks),
        )

    expected_count = len(expected)
    declared_world_sizes = {record[0].world_size for record in records.values()}
    declared_local_world_sizes = {record[0].local_world_size for record in records.values()}
    if len(declared_world_sizes) != 1 or len(declared_local_world_sizes) != 1:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance ranks disagree on world-size metadata"
        )
    declared_local_world_size = next(iter(declared_local_world_sizes))
    if declared_local_world_size != expected_count:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance record count differs from local_world_size: "
            f"expected {declared_local_world_size}, observed {expected_count}"
        )
    global_ranks = [record[0].rank for record in records.values()]
    if len(set(global_ranks)) != expected_count:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance contains duplicate global ranks"
        )
    process_ids = [record[0].process_id for record in records.values()]
    if len(set(process_ids)) != expected_count:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance contains duplicate worker process IDs"
        )

    snapshots = {local_rank: records[local_rank][1] for local_rank in sorted(records)}
    _require_rank_runtime_consistency(snapshots, target=target)
    return TorchrunRuntimeProvenanceParseResult(
        raw_stdout=raw_stdout,
        clean_stdout="".join(clean_parts),
        snapshots_by_local_rank=snapshots,
        execution_process_ids={
            local_rank: records[local_rank][0].process_id for local_rank in sorted(records)
        },
    )


__all__ = [
    "TORCHRUN_RUNTIME_PROVENANCE_PREFIX",
    "TORCHRUN_RUNTIME_PROVENANCE_SCHEMA",
    "TorchrunRuntimeProvenanceError",
    "TorchrunRuntimeProvenanceParseResult",
    "emit_torchrun_runtime_provenance",
    "format_torchrun_runtime_provenance_frames",
    "parse_torchrun_runtime_provenance_stdout",
]
