#!/usr/bin/env python3
"""Project one deterministic public-safe summary from a valid private v3 receipt."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__:
    from .qualified_gpu_training import (
        EvidenceError, build_gpu_public_summary, canonical_json, strict_load_json,
    )
else:
    from qualified_gpu_training import (
        EvidenceError, build_gpu_public_summary, canonical_json, strict_load_json,
    )


DEFAULT_RECEIPT = Path("evidence/measurement-receipt.json")


def project(package: Path, output: Path,
            receipt_relative: Path = DEFAULT_RECEIPT) -> Path:
    """Write no bytes unless the receipt is valid, qualified, and output is absent."""
    package = package.resolve(strict=True)
    try:
        output.parent.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"public-summary output parent is missing: {exc}") from exc
    if output.exists() or output.is_symlink():
        raise EvidenceError("public-summary output already exists. Refusing overwrite")
    receipt_path = package / receipt_relative
    try:
        receipt_path.resolve(strict=True).relative_to(package)
    except (OSError, ValueError) as exc:
        raise EvidenceError("measurement receipt is missing or escapes the package") from exc
    if receipt_path.is_symlink():
        raise EvidenceError("measurement receipt must not be a symlink")
    receipt = strict_load_json(receipt_path, "GPU measurement receipt")
    if not isinstance(receipt, dict):
        raise EvidenceError("GPU measurement receipt must be an object")
    summary = build_gpu_public_summary(package, receipt)
    payload = canonical_json(summary)
    try:
        with output.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise EvidenceError("public-summary output appeared. Refusing overwrite") from exc
    if output.read_bytes() != payload:
        output.unlink(missing_ok=True)
        raise EvidenceError("public-summary bytes changed during write/readback")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        project(args.package, args.output, args.receipt)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: wrote deterministic public-safe summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
