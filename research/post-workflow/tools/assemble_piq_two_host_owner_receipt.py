#!/usr/bin/env python3
"""Assemble a fixed PIQ two-host owner receipt from explicit retained files.

The command is pure and local: it performs no runtime calls and infers no
authority.  It validates the producer receipt and the exact owner handback,
then atomically creates (and never overwrites) the fixed receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

if __package__:
    from .check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        _load_json,
        _safe_relative,
        validate_two_host_functional_receipt,
    )
    from .check_piq_two_host_owner_authority import (
        OWNER_KIND,
        OWNER_RECEIPT_PATH,
        OWNER_SCOPE,
        validate_two_host_owner_authority_receipt,
    )
else:
    from check_piq_two_host_functional import (
        RECEIPT_PATH as PRODUCER_RECEIPT_PATH,
        _load_json,
        _safe_relative,
        validate_two_host_functional_receipt,
    )
    from check_piq_two_host_owner_authority import (
        OWNER_KIND,
        OWNER_RECEIPT_PATH,
        OWNER_SCOPE,
        validate_two_host_owner_authority_receipt,
    )


def _explicit_file(
    package: Path, relative: str, label: str
) -> tuple[Path, dict[str, str]]:
    if not _safe_relative(relative) or not relative.startswith("evidence/"):
        raise ValueError(
            f"{label} must be a raw-dot-free package-relative evidence/ path"
        )
    current = package
    for part in Path(relative).parts:
        try:
            if part not in {entry.name for entry in current.iterdir()}:
                raise ValueError(
                    f"{label} must preserve exact on-disk case: {relative}"
                )
        except OSError as exc:
            raise ValueError(f"{label} path cannot be inspected: {exc}") from exc
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlinks: {relative}")
    try:
        current.resolve().relative_to(package.resolve())
        if not current.is_file() or current.stat().st_size == 0:
            raise ValueError(f"{label} must be a nonempty regular file: {relative}")
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected: {exc}") from exc
    except ValueError as exc:
        if str(exc).startswith(label):
            raise
        raise ValueError(f"{label} escapes the package: {relative}") from exc
    return current, {"path": relative, "sha256": digest}


def assemble_owner_receipt_data(
    package: Path, producer_relative: str, handback_relative: str
) -> dict[str, Any]:
    """Build and fully validate receipt data without writing it."""
    package = package.resolve()
    if producer_relative != PRODUCER_RECEIPT_PATH:
        raise ValueError(
            f"producer receipt must equal the fixed path {PRODUCER_RECEIPT_PATH}"
        )
    if handback_relative in {PRODUCER_RECEIPT_PATH, OWNER_RECEIPT_PATH}:
        raise ValueError("owner handback must be a distinct retained evidence file")
    errors: list[str] = []
    producer_path, producer_ref = _explicit_file(
        package, producer_relative, "producer receipt"
    )
    producer = _load_json(producer_path, "producer receipt", errors)
    errors.extend(validate_two_host_functional_receipt(package, producer))
    handback_path, handback_ref = _explicit_file(
        package, handback_relative, "owner handback"
    )
    handback = _load_json(handback_path, "owner handback", errors)
    if not isinstance(producer, dict):
        errors.append("producer receipt must be a JSON object")
        producer = {}
    if not isinstance(handback, dict):
        errors.append("owner handback must be a JSON object")
        handback = {}
    receipt = {
        "schema_version": 1,
        "kind": OWNER_KIND,
        "scope": OWNER_SCOPE,
        "qualification": "not_established",
        "producer_receipt": producer_ref,
        "envelope": producer.get("envelope"),
        "run_id": producer.get("run_id"),
        "attempt_ids": producer.get("attempt_ids"),
        "owner_handback": handback_ref,
        "authority": handback.get("authority"),
        "attempts": handback.get("attempts"),
    }
    if not errors:
        errors.extend(
            validate_two_host_owner_authority_receipt(package, receipt, producer)
        )
    if errors:
        raise ValueError("invalid owner receipt inputs:\n- " + "\n- ".join(errors))
    return receipt


def _atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    """Install complete bytes atomically while refusing every existing target."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            f"output parent must be an existing nonsymlink directory: {parent}"
        )
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".owner-receipt-", dir=parent)
    temporary = Path(temporary_name)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        installed = True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if installed:
        _fsync_directory(parent)


def _fsync_directory(path: Path) -> None:
    """Durably commit directory entries after an atomic link or rename."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--producer-receipt", required=True)
    parser.add_argument("--owner-handback", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    package = args.package.resolve()
    if args.output != OWNER_RECEIPT_PATH:
        parser.error(f"--output must equal {OWNER_RECEIPT_PATH}")
    try:
        receipt = assemble_owner_receipt_data(
            package, args.producer_receipt, args.owner_handback
        )
        _atomic_create_json(package / args.output, receipt)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(f"CREATED {package / args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
