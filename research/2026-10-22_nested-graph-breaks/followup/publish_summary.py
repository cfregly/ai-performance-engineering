#!/usr/bin/env python3
"""Copy B200-calculated follow-up data while redacting operational metadata.

This performs no benchmark analysis, unit conversion, or ratio calculation.
The source must match the retained compact answer-data hash.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


INPUT_SHA256 = "d7518190c33b30383171cb30649512ac32da63432ba448ba1084290cebf2cbec"
PENDING_LIMIT = "Profile geometry reader correction remains pending. No additive causal percentage."
METADATA_FIELDS = {"device", "inode", "mode", "mtime_ns"}


def public_projection(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preserve all scientific rows and report each metadata-only replacement."""
    result = copy.deepcopy(data)
    changes: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, list):
            return [walk(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            location = f"{path}.{key}" if path else key
            replacement = item
            if key == "path" and isinstance(item, str) and item.startswith("/"):
                replacement = Path(item).name
            elif key in METADATA_FIELDS:
                replacement = f"<operational-{key}>"
            elif key in {"pid", "tid"} and path.endswith(".cpu_marker"):
                replacement = f"<cpu-{key}>"
            if replacement != item:
                changes.append({"path": location, "category": "operational metadata", "replacement": replacement})
                out[key] = replacement
            else:
                out[key] = walk(item, location)
        return out

    result = walk(result, "")
    # The B200 reader completed before this projection. Keep the stale note
    # visible as provenance rather than presenting it as the current status.
    if PENDING_LIMIT in result.get("limits", []):
        result["limits"].remove(PENDING_LIMIT)
        result["superseded_limit"] = {
            "recorded_text": PENDING_LIMIT,
            "resolution": "Corrected reader completed on all four retained traces. Boundary exclusions remain explicit. No causal percentage follows.",
        }
        changes.append({"path": "limits", "category": "superseded status note", "replacement": "superseded_limit"})
    result["publication"] = {
        "schema": "nested-graph-followup-public-summary/v1",
        "source_compact_sha256": INPUT_SHA256,
        "operation": "Copy scientific values. Redact operational metadata. No statistical analysis or unit conversion.",
    }
    return result, changes


def project_file(source: Path, output: Path, manifest: Path) -> None:
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != INPUT_SHA256:
        raise ValueError(f"source hash mismatch: {digest}")
    if output.resolve() == manifest.resolve() or any(path.exists() for path in (output, manifest)):
        raise ValueError("output and manifest must be distinct unused paths")
    result, changes = public_projection(json.loads(raw))
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    receipt = {
        "schema": "nested-graph-followup-public-projection/v1",
        "source_sha256": digest,
        "source_bytes": len(raw),
        "public_sha256": hashlib.sha256(encoded).hexdigest(),
        "public_bytes": len(encoded),
        "changes": changes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(encoded)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    project_file(args.source, args.out, args.manifest)


if __name__ == "__main__":
    main()
