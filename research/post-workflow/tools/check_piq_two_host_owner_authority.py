#!/usr/bin/env python3
"""Check retained owner-attested assertions and the reviewed admission tuple.

A matching receipt can remove only the producer receipt's readiness block.
This code does not independently establish execution, physical hosts or
traffic. Scientific qualification and performance claims remain unestablished.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

if __package__:
    from .check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        SHA256_RE,
        _bundle_ref,
        _load_json,
        _nonempty,
        _same_json,
        _validated_ref,
    )
    from .piq_two_host_owner_admissions import (
        load_fixed_admissions,
        require_admitted_tuple,
    )
else:
    from check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        SHA256_RE,
        _bundle_ref,
        _load_json,
        _nonempty,
        _same_json,
        _validated_ref,
    )
    from piq_two_host_owner_admissions import (
        load_fixed_admissions,
        require_admitted_tuple,
    )


OWNER_RECEIPT_PATH = "evidence/two-host-owner-authority-receipt.json"
OWNER_SCOPE = "bounded_two_host_functional_execution_and_custody"
OWNER_KIND = "piq_two_host_owner_authority_receipt"
HAND_BACK_KIND = "piq_two_host_execution_owner_handback"
UTC_RE = re.compile(
    r"^(?P<whole>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,9}))?Z$"
)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PORTABLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}$")
SPECIFIC_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{7,254}$")
TRANSPORT_PROTOCOLS = {"tcp", "tcp4", "tcp6"}
GENERIC_IDENTIFIERS = {
    "authority",
    "controller",
    "custodian",
    "execution-owner",
    "execution_owner",
    "host",
    "launcher",
    "node",
    "owner",
    "runtime-owner",
    "runtime_owner",
    "system",
    "unknown",
}


def _utc_nanoseconds(value: Any) -> int | None:
    """Return a strict Z timestamp as comparable nanoseconds, or None."""
    if not isinstance(value, str):
        return None
    match = UTC_RE.fullmatch(value)
    if match is None:
        return None
    try:
        whole = datetime.strptime(match.group("whole"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return int(whole.timestamp()) * 1_000_000_000 + int(fraction or "0")


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _specific_identifier(value: Any, label: str, errors: list[str]) -> str | None:
    if (
        not isinstance(value, str)
        or SPECIFIC_IDENTIFIER_RE.fullmatch(value) is None
        or value.casefold() in GENERIC_IDENTIFIERS
    ):
        errors.append(f"{label} must be a specific non-generic identifier")
        return None
    return value


def _portable_identifier(value: Any, label: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or PORTABLE_IDENTIFIER_RE.fullmatch(value) is None:
        errors.append(f"{label} must be a 1-255 character portable ASCII identifier")
        return None
    return value


def _exact_object(
    value: Any, fields: set[str], label: str, errors: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    if set(value) != fields:
        errors.append(f"{label} fields do not match schema version 1")
    return value


def _fixed_receipt_is_regular(
    package: Path, relative: str, errors: list[str]
) -> Path | None:
    current = package
    for part in Path(relative).parts:
        try:
            if part not in {entry.name for entry in current.iterdir()}:
                errors.append(
                    f"missing nonsymlink PIQ two-host owner receipt: {relative}"
                )
                return None
        except OSError as exc:
            errors.append(f"PIQ two-host owner receipt path cannot be inspected: {exc}")
            return None
        current /= part
        if current.is_symlink():
            errors.append(
                f"PIQ two-host owner receipt must not use symlinks: {relative}"
            )
            return None
    try:
        if not current.is_file():
            errors.append(f"missing nonsymlink PIQ two-host owner receipt: {relative}")
            return None
    except OSError as exc:
        errors.append(f"PIQ two-host owner receipt cannot be inspected: {exc}")
        return None
    return current


def _validate_projection(
    package: Path,
    value: Any,
    label: str,
    expected: dict[str, Any],
    errors: list[str],
    seen_files: dict[tuple[int, int], str],
) -> None:
    result = _validated_ref(package, value, label, errors)
    if result is None:
        return
    path, _ = result
    try:
        stat = path.stat()
    except OSError as exc:
        errors.append(f"{label} file cannot be inspected: {exc}")
        return
    if stat.st_nlink != 1:
        errors.append(f"{label} must have exactly one filesystem link")
    identity = (stat.st_dev, stat.st_ino)
    if identity in seen_files:
        errors.append(
            f"PIQ two-host owner observation files must be physically distinct: "
            f"{label} reuses {seen_files[identity]}"
        )
    else:
        seen_files[identity] = label
    document = _load_json(path, label, errors)
    if not _same_json(document, expected):
        errors.append(f"{label} does not match its exact owner projection")


def _owner_projection(
    kind: str,
    authority: dict[str, Any],
    run_id: Any,
    attempt_id: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "owner_id": authority.get("owner_id"),
        "authority_id": authority.get("authority_id"),
        "run_id": run_id,
        "attempt_id": attempt_id,
        **payload,
    }


def _producer_execution(
    package: Path, producer: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    result = _validated_ref(
        package,
        producer.get("execution"),
        "PIQ two-host owner producer execution",
        errors,
    )
    if result is None:
        return {}
    value = _load_json(result[0], "PIQ two-host owner producer execution", errors)
    if not isinstance(value, dict):
        errors.append("PIQ two-host owner producer execution must be a JSON object")
        return {}
    return value


def _producer_attempt_windows(
    package: Path,
    envelope: dict[str, Any],
    envelope_relative: str,
    attempt_ids: list[str],
    objective_window: tuple[int, int] | None,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    attempts = envelope.get("attempts")
    by_id = (
        {
            item.get("id"): item
            for item in attempts
            if isinstance(attempts, list)
            and isinstance(item, dict)
            and isinstance(item.get("id"), str)
        }
        if isinstance(attempts, list)
        else {}
    )
    windows: dict[str, dict[str, Any]] = {}
    for attempt_id in attempt_ids:
        attempt = by_id.get(attempt_id)
        if not isinstance(attempt, dict):
            errors.append(
                f"PIQ two-host owner envelope attempt is missing: {attempt_id}"
            )
            continue
        raw_ref = _bundle_ref(
            envelope_relative,
            attempt.get("raw_result"),
            f"owner selected attempt {attempt_id} raw result",
            errors,
        )
        result = _validated_ref(
            package,
            raw_ref,
            f"PIQ two-host owner selected attempt {attempt_id} raw result",
            errors,
        )
        if result is None:
            continue
        raw = _load_json(
            result[0],
            f"PIQ two-host owner selected attempt {attempt_id} raw result",
            errors,
        )
        native = raw.get("native_attempt") if isinstance(raw, dict) else None
        if not isinstance(native, dict):
            errors.append(
                f"PIQ two-host owner selected attempt {attempt_id} native attempt must be a JSON object"
            )
            continue
        started_at = native.get("startedAt")
        completed_at = native.get("completedAt")
        start = _utc_nanoseconds(started_at)
        end = _utc_nanoseconds(completed_at)
        if start is None or end is None or start >= end:
            errors.append(
                f"PIQ two-host owner selected attempt {attempt_id} must retain an ordered native UTC window"
            )
            continue
        if (
            objective_window is None
            or not objective_window[0] <= start < end <= objective_window[1]
        ):
            errors.append(
                f"PIQ two-host owner selected attempt {attempt_id} native window must be inside the envelope objective"
            )
        windows[attempt_id] = {
            "started_at": started_at,
            "completed_at": completed_at,
            "start": start,
            "end": end,
        }
    return windows


def _event_fingerprint(transport: dict[str, Any]) -> str | None:
    fields = (
        "protocol",
        "source_rank",
        "destination_rank",
        "source_pid",
        "destination_pid",
        "source_physical_host_id",
        "destination_physical_host_id",
        "sent_payload_sha256",
        "received_payload_sha256",
        "payload_size_bytes",
        "sent_at",
        "received_at",
    )
    try:
        normalized = {field: transport.get(field) for field in fields}
        normalized["sent_at"] = _utc_nanoseconds(transport.get("sent_at"))
        normalized["received_at"] = _utc_nanoseconds(transport.get("received_at"))
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def _validate_attempt(
    package: Path,
    value: Any,
    expected_attempt_id: str,
    run_id: Any,
    authority: dict[str, Any],
    producer_attempt: dict[str, Any],
    host_identities: dict[str, str],
    objective_window: tuple[int, int] | None,
    native_window: dict[str, Any],
    errors: list[str],
    seen_files: dict[tuple[int, int], str],
) -> dict[str, Any]:
    label = f"PIQ two-host owner attempt {expected_attempt_id}"
    attempt = _exact_object(
        value,
        {"attempt_id", "launch", "ranks", "transport", "cleanup"},
        label,
        errors,
    )
    if attempt.get("attempt_id") != expected_attempt_id:
        errors.append(f"{label} attempt_id must match the selected producer attempt")

    launch = _exact_object(
        attempt.get("launch"),
        {
            "launch_id",
            "controller_id",
            "execution_session_id",
            "started_at",
            "ended_at",
            "evidence",
        },
        f"{label} launch",
        errors,
    )
    launch_id = _specific_identifier(
        launch.get("launch_id"), f"{label} launch_id", errors
    )
    controller_id = _specific_identifier(
        launch.get("controller_id"), f"{label} controller_id", errors
    )
    session_id = _specific_identifier(
        launch.get("execution_session_id"), f"{label} execution_session_id", errors
    )
    launch_start = _utc_nanoseconds(launch.get("started_at"))
    launch_end = _utc_nanoseconds(launch.get("ended_at"))
    if launch_start is None or launch_end is None or launch_start >= launch_end:
        errors.append(f"{label} launch must have a strict ordered UTC start/end window")
    if launch.get("started_at") != native_window.get("started_at") or launch.get(
        "ended_at"
    ) != native_window.get("completed_at"):
        errors.append(
            f"{label} launch must exactly match the producer native attempt window"
        )
    if (
        launch_start is None
        or launch_end is None
        or objective_window is None
        or not objective_window[0] <= launch_start < launch_end <= objective_window[1]
    ):
        errors.append(f"{label} launch must be inside the envelope objective window")
    _validate_projection(
        package,
        launch.get("evidence"),
        f"{label} launch evidence",
        _owner_projection(
            "piq_two_host_owner_launch_observation",
            authority,
            run_id,
            expected_attempt_id,
            {
                key: launch.get(key)
                for key in (
                    "launch_id",
                    "controller_id",
                    "execution_session_id",
                    "started_at",
                    "ended_at",
                )
            },
        ),
        errors,
        seen_files,
    )

    producer_census = producer_attempt.get("rank_census")
    producer_by_rank = (
        {
            item.get("rank"): item
            for item in producer_census
            if isinstance(producer_census, list)
            and isinstance(item, dict)
            and type(item.get("rank")) is int
        }
        if isinstance(producer_census, list)
        else {}
    )
    ranks_value = attempt.get("ranks")
    if not isinstance(ranks_value, list) or len(ranks_value) != 2:
        errors.append(f"{label} ranks must contain exactly ranks 0 and 1")
        ranks_value = []
    ranks: dict[int, dict[str, Any]] = {}
    rank_times: dict[int, tuple[int, int]] = {}
    for index, raw_rank in enumerate(ranks_value):
        rank_label = f"{label} ranks[{index}]"
        rank = _exact_object(
            raw_rank,
            {
                "rank",
                "producer_host",
                "physical_host_id",
                "boot_id",
                "execution_session_id",
                "pid",
                "started_at",
                "ended_at",
                "launch_id",
                "controller_id",
                "evidence",
            },
            rank_label,
            errors,
        )
        rank_number = rank.get("rank")
        if type(rank_number) is not int or rank_number not in (0, 1):
            errors.append(f"{rank_label}.rank must be the type-strict integer 0 or 1")
            continue
        if rank_number != index:
            errors.append(f"{label} ranks must be ordered exactly as [0, 1]")
        if rank_number in ranks:
            errors.append(f"{label} ranks must be distinct")
            continue
        ranks[rank_number] = rank
        producer_rank = producer_by_rank.get(rank_number, {})
        producer_host = rank.get("producer_host")
        _portable_identifier(producer_host, f"{rank_label}.producer_host", errors)
        if producer_host != producer_rank.get("host"):
            errors.append(f"{rank_label}.producer_host must match the producer census")
        if type(rank.get("pid")) is not int or rank.get("pid") <= 0:
            errors.append(f"{rank_label}.pid must be a positive type-strict integer")
        elif rank.get("pid") != producer_rank.get("processId"):
            errors.append(f"{rank_label}.pid must match the producer census")
        physical_host_id = rank.get("physical_host_id")
        if (
            not isinstance(physical_host_id, str)
            or SHA256_RE.fullmatch(physical_host_id) is None
        ):
            errors.append(
                f"{rank_label}.physical_host_id must be a canonical lowercase host identity SHA-256"
            )
        expected_host_identity = (
            host_identities.get(producer_host)
            if isinstance(producer_host, str)
            else None
        )
        if physical_host_id != expected_host_identity:
            errors.append(
                f"{rank_label}.physical_host_id must match execution.resources.cpuHosts"
            )
        boot_id = rank.get("boot_id")
        if not isinstance(boot_id, str) or UUID_RE.fullmatch(boot_id) is None:
            errors.append(f"{rank_label}.boot_id must be a canonical lowercase UUID")
        if rank.get("execution_session_id") != session_id:
            errors.append(f"{rank_label}.execution_session_id must match the launch")
        if (
            rank.get("launch_id") != launch_id
            or rank.get("controller_id") != controller_id
        ):
            errors.append(
                f"{rank_label} launch/controller identity must match the attempt"
            )
        started = _utc_nanoseconds(rank.get("started_at"))
        ended = _utc_nanoseconds(rank.get("ended_at"))
        if (
            started is None
            or ended is None
            or started >= ended
            or launch_start is None
            or launch_end is None
            or not launch_start <= started < ended <= launch_end
            or objective_window is None
            or not objective_window[0] <= started < ended <= objective_window[1]
            or native_window.get("start") is None
            or native_window.get("end") is None
            or not native_window["start"] <= started < ended <= native_window["end"]
        ):
            errors.append(
                f"{rank_label} must have an ordered UTC window inside the objective, native attempt, and launch"
            )
        else:
            rank_times[rank_number] = (started, ended)
        _validate_projection(
            package,
            rank.get("evidence"),
            f"{rank_label} evidence",
            _owner_projection(
                "piq_two_host_owner_rank_observation",
                authority,
                run_id,
                expected_attempt_id,
                {
                    key: rank.get(key)
                    for key in (
                        "rank",
                        "producer_host",
                        "physical_host_id",
                        "boot_id",
                        "execution_session_id",
                        "pid",
                        "started_at",
                        "ended_at",
                        "launch_id",
                        "controller_id",
                    )
                },
            ),
            errors,
            seen_files,
        )
    if sorted(ranks) != [0, 1]:
        errors.append(f"{label} ranks must contain one complete [0, 1] census")
    physical_hosts = [rank.get("physical_host_id") for rank in ranks.values()]
    if (
        len(physical_hosts) == 2
        and all(isinstance(value, str) for value in physical_hosts)
        and physical_hosts[0].casefold() == physical_hosts[1].casefold()
    ):
        errors.append(f"{label} must bind two distinct owner-observed physical hosts")
    boot_identities = [rank.get("boot_id") for rank in ranks.values()]
    if (
        len(boot_identities) == 2
        and all(isinstance(value, str) for value in boot_identities)
        and boot_identities[0].casefold() == boot_identities[1].casefold()
    ):
        errors.append(
            f"{label} must bind distinct boot identities for its two physical hosts"
        )

    transport = _exact_object(
        attempt.get("transport"),
        {
            "transport_id",
            "protocol",
            "source_rank",
            "destination_rank",
            "source_pid",
            "destination_pid",
            "source_physical_host_id",
            "destination_physical_host_id",
            "execution_session_id",
            "sent_payload_sha256",
            "received_payload_sha256",
            "payload_size_bytes",
            "sent_at",
            "received_at",
            "evidence",
        },
        f"{label} transport",
        errors,
    )
    _specific_identifier(transport.get("transport_id"), f"{label} transport_id", errors)
    protocol = transport.get("protocol")
    if not isinstance(protocol, str) or protocol not in TRANSPORT_PROTOCOLS:
        errors.append(
            f"{label} transport.protocol must be one of {sorted(TRANSPORT_PROTOCOLS)!r}"
        )
    source_rank = transport.get("source_rank")
    destination_rank = transport.get("destination_rank")
    if (
        type(source_rank) is not int
        or type(destination_rank) is not int
        or {source_rank, destination_rank} != {0, 1}
    ):
        errors.append(f"{label} transport must connect type-strict ranks 0 and 1")
        source_rank = destination_rank = None
    source = ranks.get(source_rank, {})
    destination = ranks.get(destination_rank, {})
    transport_bindings = {
        "source_pid": source.get("pid"),
        "destination_pid": destination.get("pid"),
        "source_physical_host_id": source.get("physical_host_id"),
        "destination_physical_host_id": destination.get("physical_host_id"),
        "execution_session_id": session_id,
    }
    for field, expected in transport_bindings.items():
        if transport.get(field) != expected:
            errors.append(
                f"{label} transport.{field} must match its same-attempt ranks"
            )
    for field in ("source_pid", "destination_pid", "payload_size_bytes"):
        if not _positive_int(transport.get(field)):
            errors.append(
                f"{label} transport.{field} must be a positive type-strict integer"
            )
    sent_digest = transport.get("sent_payload_sha256")
    received_digest = transport.get("received_payload_sha256")
    if (
        not isinstance(sent_digest, str)
        or SHA256_RE.fullmatch(sent_digest) is None
        or received_digest != sent_digest
    ):
        errors.append(
            f"{label} transport must retain matching sent/received payload SHA-256"
        )
    sent_at = _utc_nanoseconds(transport.get("sent_at"))
    received_at = _utc_nanoseconds(transport.get("received_at"))
    source_times = rank_times.get(source_rank) if source_rank is not None else None
    destination_times = (
        rank_times.get(destination_rank) if destination_rank is not None else None
    )
    if (
        sent_at is None
        or received_at is None
        or sent_at >= received_at
        or source_times is None
        or destination_times is None
        or not max(source_times[0], destination_times[0]) <= sent_at
        or not received_at <= min(source_times[1], destination_times[1])
    ):
        errors.append(
            f"{label} transport must retain an ordered inter-host send/receive inside both rank windows"
        )
    _validate_projection(
        package,
        transport.get("evidence"),
        f"{label} transport evidence",
        _owner_projection(
            "piq_two_host_owner_transport_observation",
            authority,
            run_id,
            expected_attempt_id,
            {
                key: transport.get(key)
                for key in (
                    "transport_id",
                    "protocol",
                    "source_rank",
                    "destination_rank",
                    "source_pid",
                    "destination_pid",
                    "source_physical_host_id",
                    "destination_physical_host_id",
                    "execution_session_id",
                    "sent_payload_sha256",
                    "received_payload_sha256",
                    "payload_size_bytes",
                    "sent_at",
                    "received_at",
                )
            },
        ),
        errors,
        seen_files,
    )

    cleanup_value = attempt.get("cleanup")
    if not isinstance(cleanup_value, list) or len(cleanup_value) != 2:
        errors.append(f"{label} cleanup must contain exactly ranks 0 and 1")
        cleanup_value = []
    cleanup_ranks: list[int] = []
    cleanup_times: list[int] = []
    terminal_boundary = max(
        [
            time
            for time in [
                launch_end,
                received_at,
                *(end for _, end in rank_times.values()),
            ]
            if time is not None
        ],
        default=None,
    )
    for index, raw_cleanup in enumerate(cleanup_value):
        cleanup_label = f"{label} cleanup[{index}]"
        cleanup = _exact_object(
            raw_cleanup,
            {
                "rank",
                "physical_host_id",
                "boot_id",
                "execution_session_id",
                "pid",
                "launch_id",
                "controller_id",
                "transport_id",
                "observed_at",
                "pid_absent",
                "transport_closed",
                "session_terminal",
                "evidence",
            },
            cleanup_label,
            errors,
        )
        rank_number = cleanup.get("rank")
        if type(rank_number) is not int or rank_number not in (0, 1):
            errors.append(
                f"{cleanup_label}.rank must be the type-strict integer 0 or 1"
            )
            continue
        if rank_number != index:
            errors.append(f"{label} cleanup must be ordered exactly as [0, 1]")
        cleanup_ranks.append(rank_number)
        rank = ranks.get(rank_number, {})
        expected_bindings = {
            "physical_host_id": rank.get("physical_host_id"),
            "boot_id": rank.get("boot_id"),
            "execution_session_id": session_id,
            "pid": rank.get("pid"),
            "launch_id": launch_id,
            "controller_id": controller_id,
            "transport_id": transport.get("transport_id"),
        }
        for field, expected in expected_bindings.items():
            if cleanup.get(field) != expected:
                errors.append(
                    f"{cleanup_label}.{field} must match its same-attempt rank"
                )
        if not _positive_int(cleanup.get("pid")):
            errors.append(f"{cleanup_label}.pid must be a positive type-strict integer")
        for field in ("pid_absent", "transport_closed", "session_terminal"):
            if cleanup.get(field) is not True:
                errors.append(f"{cleanup_label}.{field} must be true")
        observed_at = _utc_nanoseconds(cleanup.get("observed_at"))
        if (
            observed_at is None
            or terminal_boundary is None
            or observed_at <= terminal_boundary
        ):
            errors.append(
                f"{cleanup_label} must be an ordered UTC observation after launch, both ranks, and transport completion"
            )
        else:
            cleanup_times.append(observed_at)
        _validate_projection(
            package,
            cleanup.get("evidence"),
            f"{cleanup_label} evidence",
            _owner_projection(
                "piq_two_host_owner_cleanup_observation",
                authority,
                run_id,
                expected_attempt_id,
                {
                    key: cleanup.get(key)
                    for key in (
                        "rank",
                        "physical_host_id",
                        "boot_id",
                        "execution_session_id",
                        "pid",
                        "launch_id",
                        "controller_id",
                        "transport_id",
                        "observed_at",
                        "pid_absent",
                        "transport_closed",
                        "session_terminal",
                    )
                },
            ),
            errors,
            seen_files,
        )
    if cleanup_ranks != [0, 1]:
        errors.append(
            f"{label} cleanup must retain one terminal observation for each rank"
        )
    return {
        "cleanup_time": max(cleanup_times, default=None),
        "launch_id": launch.get("launch_id"),
        "execution_session_id": launch.get("execution_session_id"),
        "transport_id": transport.get("transport_id"),
        "event_fingerprint": _event_fingerprint(transport),
    }


def validate_two_host_owner_authority_receipt(
    package: Path, data: Any, producer_data: Any | None = None
) -> list[str]:
    """Validate a fixed owner receipt against one already validated producer receipt."""
    package = package.resolve()
    errors: list[str] = []
    receipt = _exact_object(
        data,
        {
            "schema_version",
            "kind",
            "scope",
            "qualification",
            "producer_receipt",
            "envelope",
            "run_id",
            "attempt_ids",
            "owner_handback",
            "authority",
            "attempts",
        },
        "PIQ two-host owner receipt",
        errors,
    )
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
    ):
        errors.append("PIQ two-host owner receipt schema_version must equal 1")
    if receipt.get("kind") != OWNER_KIND:
        errors.append("PIQ two-host owner receipt kind is unsupported")
    if receipt.get("scope") != OWNER_SCOPE:
        errors.append("PIQ two-host owner receipt scope is unsupported")
    if receipt.get("qualification") != "not_established":
        errors.append(
            "PIQ two-host owner receipt cannot claim scientific qualification"
        )

    producer_result = _validated_ref(
        package,
        receipt.get("producer_receipt"),
        "PIQ two-host owner producer receipt",
        errors,
    )
    producer: dict[str, Any] = {}
    producer_ref: dict[str, str] = {}
    if producer_result is not None:
        producer_ref = producer_result[1]
        if producer_ref["path"] != PRODUCER_RECEIPT_PATH:
            errors.append(
                f"PIQ two-host owner producer receipt must equal {PRODUCER_RECEIPT_PATH}"
            )
        loaded = _load_json(
            producer_result[0], "PIQ two-host owner producer receipt", errors
        )
        if isinstance(loaded, dict):
            producer = loaded
        else:
            errors.append("PIQ two-host owner producer receipt must be a JSON object")
    if producer_data is not None and not _same_json(producer_data, producer):
        errors.append(
            "PIQ two-host owner receipt does not bind the validated producer bytes"
        )

    envelope_result = _validated_ref(
        package, receipt.get("envelope"), "PIQ two-host owner envelope", errors
    )
    envelope_ref = envelope_result[1] if envelope_result is not None else {}
    envelope: dict[str, Any] = {}
    if envelope_result is not None:
        loaded_envelope = _load_json(
            envelope_result[0], "PIQ two-host owner envelope", errors
        )
        if isinstance(loaded_envelope, dict):
            envelope = loaded_envelope
        else:
            errors.append("PIQ two-host owner envelope must be a JSON object")
    if not _same_json(receipt.get("envelope"), producer.get("envelope")):
        errors.append(
            "PIQ two-host owner envelope must exactly match the producer receipt"
        )
    objective = envelope.get("objective")
    objective_value = objective.get("window") if isinstance(objective, dict) else None
    objective_start = (
        _utc_nanoseconds(objective_value.get("start"))
        if isinstance(objective_value, dict)
        else None
    )
    objective_end = (
        _utc_nanoseconds(objective_value.get("end"))
        if isinstance(objective_value, dict)
        else None
    )
    objective_window = (
        (objective_start, objective_end)
        if objective_start is not None
        and objective_end is not None
        and objective_start < objective_end
        else None
    )
    if objective_window is None:
        errors.append(
            "PIQ two-host owner envelope must retain an ordered UTC objective window"
        )

    run_id = receipt.get("run_id")
    _portable_identifier(run_id, "PIQ two-host owner run_id", errors)
    if not _nonempty(run_id) or run_id != producer.get("run_id"):
        errors.append(
            "PIQ two-host owner run_id must exactly match the producer receipt"
        )
    attempt_ids = receipt.get("attempt_ids")
    if (
        not isinstance(attempt_ids, list)
        or not attempt_ids
        or any(
            not isinstance(value, str)
            or PORTABLE_IDENTIFIER_RE.fullmatch(value) is None
            for value in attempt_ids
        )
        or len(attempt_ids)
        != len({value.casefold() for value in attempt_ids if isinstance(value, str)})
    ):
        errors.append(
            "PIQ two-host owner attempt_ids must be a distinct nonempty string list"
        )
        attempt_ids = []
    if not _same_json(attempt_ids, producer.get("attempt_ids")):
        errors.append(
            "PIQ two-host owner attempt_ids must exactly match the producer receipt"
        )
    native_windows = _producer_attempt_windows(
        package,
        envelope,
        envelope_ref.get("path", ""),
        attempt_ids,
        objective_window,
        errors,
    )

    authority = _exact_object(
        receipt.get("authority"),
        {"owner_id", "authority_id", "attested_at"},
        "PIQ two-host owner authority",
        errors,
    )
    owner_id = _specific_identifier(
        authority.get("owner_id"), "PIQ two-host owner authority.owner_id", errors
    )
    authority_id = _specific_identifier(
        authority.get("authority_id"),
        "PIQ two-host owner authority.authority_id",
        errors,
    )
    if (
        owner_id is not None
        and authority_id is not None
        and owner_id.casefold() == authority_id.casefold()
    ):
        errors.append(
            "PIQ two-host owner authority IDs must be distinct casefold identities"
        )
    attested_at = _utc_nanoseconds(authority.get("attested_at"))
    if attested_at is None:
        errors.append(
            "PIQ two-host owner authority.attested_at must be a canonical UTC timestamp"
        )

    execution = _producer_execution(package, producer, errors)
    resources = execution.get("resources")
    cpu_hosts = resources.get("cpuHosts") if isinstance(resources, dict) else None
    host_identities: dict[str, str] = {}
    if not isinstance(cpu_hosts, list) or len(cpu_hosts) != 2:
        errors.append(
            "PIQ two-host owner execution.resources.cpuHosts must contain exactly two hosts"
        )
        cpu_hosts = []
    target_casefolds: set[str] = set()
    identity_casefolds: set[str] = set()
    for index, host in enumerate(cpu_hosts):
        label = f"PIQ two-host owner execution.resources.cpuHosts[{index}]"
        if not isinstance(host, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        target_id = host.get("targetId")
        identity = host.get("hostIdentitySha256")
        if _portable_identifier(target_id, f"{label}.targetId", errors) is None:
            continue
        if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
            errors.append(
                f"{label}.hostIdentitySha256 must be a canonical lowercase SHA-256"
            )
            continue
        if target_id.casefold() in target_casefolds:
            errors.append("PIQ two-host owner CPU target identities must be distinct")
        if identity.casefold() in identity_casefolds:
            errors.append(
                "PIQ two-host owner physical host identities must be distinct"
            )
        target_casefolds.add(target_id.casefold())
        identity_casefolds.add(identity.casefold())
        host_identities[target_id] = identity
    execution_attempts = execution.get("attempts")
    producer_attempts = (
        {
            item.get("attempt_id"): item
            for item in execution_attempts
            if isinstance(execution_attempts, list)
            and isinstance(item, dict)
            and isinstance(item.get("attempt_id"), str)
        }
        if isinstance(execution_attempts, list)
        else {}
    )
    attempts_value = receipt.get("attempts")
    if not isinstance(attempts_value, list) or len(attempts_value) != len(attempt_ids):
        errors.append(
            "PIQ two-host owner attempts must contain one entry per selected attempt"
        )
        attempts_value = []
    seen_files: dict[tuple[int, int], str] = {}
    cleanup_times: list[int] = []
    observed_ids: list[Any] = []
    attempt_projections: list[dict[str, Any]] = []
    for index, attempt_id in enumerate(attempt_ids):
        if index >= len(attempts_value):
            break
        raw_attempt = attempts_value[index]
        observed_ids.append(
            raw_attempt.get("attempt_id") if isinstance(raw_attempt, dict) else None
        )
        producer_attempt = producer_attempts.get(attempt_id)
        if not isinstance(producer_attempt, dict):
            errors.append(
                f"PIQ two-host owner selected producer attempt is missing: {attempt_id}"
            )
            producer_attempt = {}
        projection = _validate_attempt(
            package,
            raw_attempt,
            attempt_id,
            run_id,
            authority,
            producer_attempt,
            host_identities,
            objective_window,
            native_windows.get(attempt_id, {}),
            errors,
            seen_files,
        )
        attempt_projections.append(projection)
        if projection.get("cleanup_time") is not None:
            cleanup_times.append(projection["cleanup_time"])
    if observed_ids != attempt_ids:
        errors.append(
            "PIQ two-host owner attempts must exactly match attempt_ids in order"
        )
    for field, label in (
        ("launch_id", "launch_id"),
        ("execution_session_id", "execution_session_id"),
        ("transport_id", "transport_id"),
        ("event_fingerprint", "transport event identity/timing/payload fingerprint"),
    ):
        seen: set[str] = set()
        for projection in attempt_projections:
            value = projection.get(field)
            if not isinstance(value, str):
                continue
            identity = value.casefold() if field != "event_fingerprint" else value
            if identity in seen:
                errors.append(
                    f"PIQ two-host owner selected attempts must not reuse {label}"
                )
            seen.add(identity)
    if cleanup_times and attested_at is not None and attested_at <= max(cleanup_times):
        errors.append(
            "PIQ two-host owner attestation must be after all terminal cleanup observations"
        )

    handback_result = _validated_ref(
        package, receipt.get("owner_handback"), "PIQ two-host owner handback", errors
    )
    if handback_result is not None:
        handback = _load_json(handback_result[0], "PIQ two-host owner handback", errors)
        expected_handback = {
            "schema_version": 1,
            "kind": HAND_BACK_KIND,
            "scope": OWNER_SCOPE,
            "qualification": "not_established",
            "producer_receipt_sha256": producer_ref.get("sha256"),
            "envelope_sha256": envelope_ref.get("sha256"),
            "run_id": run_id,
            "attempt_ids": attempt_ids,
            "authority": authority,
            "attempts": attempts_value,
        }
        if not _same_json(handback, expected_handback):
            errors.append(
                "PIQ two-host owner handback does not match its exact producer/envelope/run/attempt projection"
            )
    return errors


def owner_admission_tuple(receipt: Any) -> dict[str, Any]:
    """Project only the exact tuple admitted by the repository trust anchor."""
    if not isinstance(receipt, dict):
        return {}
    authority = receipt.get("authority")
    producer = receipt.get("producer_receipt")
    envelope = receipt.get("envelope")
    handback = receipt.get("owner_handback")
    return {
        "owner_id": authority.get("owner_id") if isinstance(authority, dict) else None,
        "authority_id": (
            authority.get("authority_id") if isinstance(authority, dict) else None
        ),
        "producer_receipt_sha256": (
            producer.get("sha256") if isinstance(producer, dict) else None
        ),
        "envelope_sha256": (
            envelope.get("sha256") if isinstance(envelope, dict) else None
        ),
        "owner_handback_sha256": (
            handback.get("sha256") if isinstance(handback, dict) else None
        ),
        "run_id": receipt.get("run_id"),
        "attempt_ids": receipt.get("attempt_ids"),
    }


def validate_declared_owner_authority(
    package: Path,
    manifest: Any,
    producer_data: Any,
    *,
    admissions: Any | None = None,
) -> tuple[list[str], bool]:
    """Return owner-receipt errors and whether it can remove the readiness block."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return errors, False
    declared = "piq_two_host_owner_authority_receipt" in manifest
    receipt_path = package / OWNER_RECEIPT_PATH
    if not declared:
        if receipt_path.exists() or receipt_path.is_symlink():
            errors.append(
                f"undeclared PIQ two-host owner receipt must be removed or declared: {OWNER_RECEIPT_PATH}"
            )
        return errors, False
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
    ):
        errors.append(
            "manifest.piq_two_host_owner_authority_receipt is supported only by schema 2"
        )
    if manifest.get("piq_two_host_owner_authority_receipt") != OWNER_RECEIPT_PATH:
        errors.append(
            f"manifest.piq_two_host_owner_authority_receipt must equal {OWNER_RECEIPT_PATH!r}"
        )
        return errors, False
    if manifest.get("piq_two_host_functional_receipt") != PRODUCER_RECEIPT_PATH:
        errors.append(
            "PIQ two-host owner receipt requires the exact declared producer functional receipt"
        )
    fixed = _fixed_receipt_is_regular(package.resolve(), OWNER_RECEIPT_PATH, errors)
    if fixed is None:
        return errors, False
    data = _load_json(fixed, "PIQ two-host owner receipt", errors)
    errors.extend(
        validate_two_host_owner_authority_receipt(package, data, producer_data)
    )
    if not errors and isinstance(data, dict):
        if admissions is None:
            admissions, admission_errors = load_fixed_admissions()
            errors.extend(admission_errors)
        if not errors:
            authority = data.get("authority")
            errors.extend(
                require_admitted_tuple(
                    admissions,
                    owner_admission_tuple(data),
                    attested_at=(
                        authority.get("attested_at")
                        if isinstance(authority, dict)
                        else None
                    ),
                )
            )
    return errors, not errors
