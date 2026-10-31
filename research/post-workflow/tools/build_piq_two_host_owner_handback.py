#!/usr/bin/env python3
"""Stage owner observations and a handback from one explicit local facts file.

The builder performs no runtime calls, infers no authority, and creates no
admission. It stages six observation documents per attempt plus one handback,
validates them, then installs the directory without overwriting owner work.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

if __package__:
    from .assemble_piq_two_host_owner_receipt import (
        _atomic_create_json,
        _explicit_file,
        _fsync_directory,
        assemble_owner_receipt_data,
    )
    from .check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        _load_json,
        _safe_relative,
        validate_two_host_functional_receipt,
    )
    from .check_piq_two_host_owner_authority import (
        HAND_BACK_KIND,
        OWNER_KIND,
        OWNER_SCOPE,
        validate_two_host_owner_authority_receipt,
    )
else:
    from assemble_piq_two_host_owner_receipt import (
        _atomic_create_json,
        _explicit_file,
        _fsync_directory,
        assemble_owner_receipt_data,
    )
    from check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        _load_json,
        _safe_relative,
        validate_two_host_functional_receipt,
    )
    from check_piq_two_host_owner_authority import (
        HAND_BACK_KIND,
        OWNER_KIND,
        OWNER_SCOPE,
        validate_two_host_owner_authority_receipt,
    )


FACTS_KIND = "piq_two_host_owner_facts"
FACTS_FIELDS = {
    "schema_version",
    "kind",
    "scope",
    "qualification",
    "producer_receipt_sha256",
    "envelope_sha256",
    "run_id",
    "attempt_ids",
    "authority",
    "attempts",
}
ATTEMPT_FIELDS = {"attempt_id", "launch", "ranks", "transport", "cleanup"}
LAUNCH_FIELDS = {
    "launch_id",
    "controller_id",
    "execution_session_id",
    "started_at",
    "ended_at",
}
RANK_FIELDS = {
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
}
TRANSPORT_FIELDS = {
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
}
CLEANUP_FIELDS = {
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
}


def _exact(
    value: Any, fields: set[str], label: str, errors: list[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    if set(value) != fields:
        errors.append(f"{label} fields do not match owner facts schema version 1")
    return value


def _explicit_output_directory(package: Path, relative: str) -> Path:
    if (
        not _safe_relative(relative)
        or not relative.startswith("evidence/")
        or len(Path(relative).parts) < 2
    ):
        raise ValueError(
            "output directory must be a raw-dot-free package-relative path below evidence/"
        )
    output = package / relative
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )
    current = package
    for part in Path(relative).parent.parts:
        try:
            if part not in {entry.name for entry in current.iterdir()}:
                raise ValueError(
                    f"output directory parent must preserve exact on-disk case: {relative}"
                )
        except OSError as exc:
            raise ValueError(
                f"output directory parent cannot be inspected: {exc}"
            ) from exc
        current /= part
        if current.is_symlink():
            raise ValueError("output directory parent must not use symlinks")
    if not current.is_dir():
        raise ValueError(f"output directory parent must exist: {current}")
    return output


def _ref(package: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(package).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _owner_observation(
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


def _replace_stage_prefix(value: Any, old: str, new: str) -> None:
    if isinstance(value, dict):
        evidence = value.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("path"), str):
            prefix = old + "/"
            if evidence["path"].startswith(prefix):
                evidence["path"] = new + "/" + evidence["path"][len(prefix) :]
        for child in value.values():
            _replace_stage_prefix(child, old, new)
    elif isinstance(value, list):
        for child in value:
            _replace_stage_prefix(child, old, new)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing concurrent owner work."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise OSError("atomic no-replace directory publication is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise OSError("atomic no-replace directory publication is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise OSError("atomic no-replace directory publication is unavailable")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def build_owner_handback(
    package: Path,
    *,
    producer_relative: str,
    facts_relative: str,
    output_directory: str,
    handback_relative: str,
) -> Path:
    """Validate, atomically publish, and never overwrite an owner handback."""
    package = package.resolve()
    if producer_relative != PRODUCER_RECEIPT_PATH:
        raise ValueError(
            f"producer receipt must equal the fixed path {PRODUCER_RECEIPT_PATH}"
        )
    output = _explicit_output_directory(package, output_directory)
    if not _safe_relative(handback_relative) or not handback_relative.startswith(
        output_directory + "/"
    ):
        raise ValueError(
            "handback must be an explicit file directly inside output directory"
        )
    if Path(handback_relative).parent.as_posix() != output_directory:
        raise ValueError("handback must be directly inside output directory")
    facts_path, _ = _explicit_file(package, facts_relative, "owner facts")
    producer_path, producer_ref = _explicit_file(
        package, producer_relative, "producer receipt"
    )
    errors: list[str] = []
    facts = _load_json(facts_path, "owner facts", errors)
    producer = _load_json(producer_path, "producer receipt", errors)
    errors.extend(validate_two_host_functional_receipt(package, producer))
    facts = _exact(facts, FACTS_FIELDS, "owner facts", errors)
    if type(facts.get("schema_version")) is not int or facts.get("schema_version") != 1:
        errors.append("owner facts schema_version must equal 1")
    if facts.get("kind") != FACTS_KIND:
        errors.append("owner facts kind is unsupported")
    if (
        facts.get("scope") != OWNER_SCOPE
        or facts.get("qualification") != "not_established"
    ):
        errors.append("owner facts scope/qualification is unsupported")
    if not isinstance(producer, dict):
        errors.append("producer receipt must be a JSON object")
        producer = {}
    if facts.get("producer_receipt_sha256") != producer_ref["sha256"]:
        errors.append(
            "owner facts producer_receipt_sha256 does not match producer bytes"
        )
    envelope = producer.get("envelope")
    envelope_sha = envelope.get("sha256") if isinstance(envelope, dict) else None
    if facts.get("envelope_sha256") != envelope_sha:
        errors.append("owner facts envelope_sha256 does not match producer receipt")
    if facts.get("run_id") != producer.get("run_id"):
        errors.append("owner facts run_id does not match producer receipt")
    if facts.get("attempt_ids") != producer.get("attempt_ids"):
        errors.append("owner facts attempt_ids do not match producer receipt in order")
    authority = _exact(
        facts.get("authority"),
        {"owner_id", "authority_id", "attested_at"},
        "owner facts authority",
        errors,
    )
    attempt_ids = (
        facts.get("attempt_ids") if isinstance(facts.get("attempt_ids"), list) else []
    )
    attempts = facts.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(attempt_ids):
        errors.append("owner facts attempts must contain one entry per attempt_id")
        attempts = []
    normalized: list[dict[str, Any]] = []
    for index, raw_attempt in enumerate(attempts):
        attempt = _exact(
            raw_attempt, ATTEMPT_FIELDS, f"owner facts attempts[{index}]", errors
        )
        launch = _exact(
            attempt.get("launch"),
            LAUNCH_FIELDS,
            f"owner facts attempts[{index}].launch",
            errors,
        )
        ranks = attempt.get("ranks")
        if not isinstance(ranks, list) or len(ranks) != 2:
            errors.append(
                f"owner facts attempts[{index}].ranks must contain exactly two entries"
            )
            ranks = []
        rank_values = [
            _exact(
                rank,
                RANK_FIELDS,
                f"owner facts attempts[{index}].ranks[{rank_index}]",
                errors,
            )
            for rank_index, rank in enumerate(ranks)
        ]
        transport = _exact(
            attempt.get("transport"),
            TRANSPORT_FIELDS,
            f"owner facts attempts[{index}].transport",
            errors,
        )
        cleanup = attempt.get("cleanup")
        if not isinstance(cleanup, list) or len(cleanup) != 2:
            errors.append(
                f"owner facts attempts[{index}].cleanup must contain exactly two entries"
            )
            cleanup = []
        cleanup_values = [
            _exact(
                item,
                CLEANUP_FIELDS,
                f"owner facts attempts[{index}].cleanup[{cleanup_index}]",
                errors,
            )
            for cleanup_index, item in enumerate(cleanup)
        ]
        normalized.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "launch": launch,
                "ranks": rank_values,
                "transport": transport,
                "cleanup": cleanup_values,
            }
        )
    if [item.get("attempt_id") for item in normalized] != attempt_ids:
        errors.append("owner facts attempts must exactly match attempt_ids in order")
    if errors:
        raise ValueError("invalid owner facts inputs:\n- " + "\n- ".join(errors))

    stage = Path(tempfile.mkdtemp(prefix=".two-host-owner-stage-", dir=output.parent))
    installed = False
    try:
        staged_attempts: list[dict[str, Any]] = []
        for index, attempt in enumerate(normalized):
            attempt_id = attempt["attempt_id"]
            attempt_dir = stage / f"attempt-{index:03d}"
            attempt_dir.mkdir()

            def write_observation(
                name: str, kind: str, payload: dict[str, Any]
            ) -> dict[str, str]:
                path = attempt_dir / name
                _atomic_create_json(
                    path,
                    _owner_observation(
                        kind, authority, facts["run_id"], attempt_id, payload
                    ),
                )
                return _ref(package, path)

            launch = dict(attempt["launch"])
            launch["evidence"] = write_observation(
                "launch.json",
                "piq_two_host_owner_launch_observation",
                attempt["launch"],
            )
            ranks = []
            for rank_index, rank in enumerate(attempt["ranks"]):
                projected = dict(rank)
                projected["evidence"] = write_observation(
                    f"rank-{rank_index}.json",
                    "piq_two_host_owner_rank_observation",
                    rank,
                )
                ranks.append(projected)
            transport = dict(attempt["transport"])
            transport["evidence"] = write_observation(
                "transport.json",
                "piq_two_host_owner_transport_observation",
                attempt["transport"],
            )
            cleanup = []
            for cleanup_index, item in enumerate(attempt["cleanup"]):
                projected = dict(item)
                projected["evidence"] = write_observation(
                    f"cleanup-{cleanup_index}.json",
                    "piq_two_host_owner_cleanup_observation",
                    item,
                )
                cleanup.append(projected)
            staged_attempts.append(
                {
                    "attempt_id": attempt_id,
                    "launch": launch,
                    "ranks": ranks,
                    "transport": transport,
                    "cleanup": cleanup,
                }
            )

        validation_handback = stage / ".validation-handback.json"
        handback = {
            "schema_version": 1,
            "kind": HAND_BACK_KIND,
            "scope": OWNER_SCOPE,
            "qualification": "not_established",
            "producer_receipt_sha256": producer_ref["sha256"],
            "envelope_sha256": envelope_sha,
            "run_id": facts["run_id"],
            "attempt_ids": facts["attempt_ids"],
            "authority": authority,
            "attempts": staged_attempts,
        }
        _atomic_create_json(validation_handback, handback)
        candidate = {
            "schema_version": 1,
            "kind": OWNER_KIND,
            "scope": OWNER_SCOPE,
            "qualification": "not_established",
            "producer_receipt": producer_ref,
            "envelope": envelope,
            "run_id": facts["run_id"],
            "attempt_ids": facts["attempt_ids"],
            "owner_handback": _ref(package, validation_handback),
            "authority": authority,
            "attempts": staged_attempts,
        }
        validation_errors = validate_two_host_owner_authority_receipt(
            package, candidate, producer
        )
        if validation_errors:
            raise ValueError(
                "invalid staged owner handback:\n- " + "\n- ".join(validation_errors)
            )

        final_attempts = json.loads(json.dumps(staged_attempts))
        _replace_stage_prefix(
            final_attempts,
            stage.relative_to(package).as_posix(),
            output_directory,
        )
        validation_handback.unlink()
        final_handback = {
            **handback,
            "attempts": final_attempts,
        }
        staged_handback_path = stage / Path(handback_relative).name
        _atomic_create_json(staged_handback_path, final_handback)
        for directory in sorted(
            (path for path in stage.rglob("*") if path.is_dir()), reverse=True
        ):
            _fsync_directory(directory)
        _fsync_directory(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output}"
            )
        _rename_noreplace(stage, output)
        installed = True
        _fsync_directory(output)
        _fsync_directory(output.parent)
        final_path = package / handback_relative
        assemble_owner_receipt_data(package, producer_relative, handback_relative)
        return final_path
    except Exception as exc:
        if not installed and stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
        if installed:
            raise RuntimeError(
                "owner handback directory was published at "
                f"{output}, but post-publish durability/verification failed. "
                "The published directory was preserved for explicit inspection"
            ) from exc
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--producer-receipt", required=True)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--handback", required=True)
    args = parser.parse_args(argv)
    try:
        path = build_owner_handback(
            args.package,
            producer_relative=args.producer_receipt,
            facts_relative=args.facts,
            output_directory=args.output_directory,
            handback_relative=args.handback,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(f"CREATED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
