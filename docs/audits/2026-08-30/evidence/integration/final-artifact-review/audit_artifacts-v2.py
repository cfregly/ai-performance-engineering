"""Read-only receipt audit derived from artifact-review/audit_artifacts.py.

Only this new final-artifact-review directory may be written.
Invocation: python audit_artifacts.py snapshot-1
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / ".git").exists())
AUDIT = REPO / "docs/audits/2026-08-30"
EVIDENCE = AUDIT / "evidence"
HEX = re.compile(r"^[0-9a-fA-F]{64}$")
EXTRA_EVIDENCE = REPO / "code/labs/nanochat_fullstack/audit_wave1"

def is_evidence_path(path):
    return path.startswith(("docs/audits/", "code/labs/nanochat_fullstack/audit_wave1/"))


SUFFIXES = {".mk", ".json", ".jsonl", ".txt", ".log", ".xml", ".md", ".py", ".cu", ".cuh", ".hpp", ".h", ".sh", ".toml", ".html", ".yaml", ".yml", ".cpp", ".cmake", ".csv", ".patch"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def resolve(raw, receipt):
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith(("code/", "docs/")) or raw in {"AGENTS.md", "AUDIT_REMEDIATION_PLAN.md"}:
        return REPO / path
    # The stage runner deliberately records names relative to its lab directory.
    if receipt.name in {"cuda-stage-no-device-receipt.json", "cuda-stage-lifetime-receipt.json"} and (
        raw.startswith("experimental/") or raw.startswith("tcgen05_")
    ):
        return REPO / "code/labs/custom_vs_cublas" / path
    # Commands in the receipts run from code/. Prefer an existing receipt-local
    # artifact first, then known repository and command-relative paths.
    choices = [receipt.parent / path, REPO / path, REPO / "code" / path, AUDIT / path, EVIDENCE / path]
    return next((candidate for candidate in choices if candidate.is_file()), choices[0])


def pathlike(value):
    if not isinstance(value, str) or len(value) > 500 or any(c.isspace() for c in value):
        return False
    if value.startswith(("http:", "https:", "file:")) or "*" in value:
        return False
    return Path(value).suffix in SUFFIXES or value.endswith(("Makefile", "CMakeLists.txt"))


def references(value, receipt, pointer="", path_hint=None):
    """Extract explicit file/hash pairs, keeping historical context for review."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from references(child, receipt, f"{pointer}/{index}")
    elif isinstance(value, dict):
        archive = (value.get("preserved_as") or value.get("preserved_original") or
                   value.get("archival_path") or value.get("original_receipt") or value.get("original"))
        named = next((value[k] for k in ("path", "file", "filename", "source_path", "source_file")
                      if k in value and pathlike(value[k])), None)
        local_path = archive or named or path_hint
        expected = value.get("sha256") or value.get("after_sha256")
        if archive and expected is None:
            expected = value.get("original_sha256")
        if local_path and isinstance(expected, str) and HEX.fullmatch(expected):
            yield {"receipt": display(receipt), "pointer": pointer, "raw_path": local_path,
                   "expected_sha256": expected.lower(), "historical_archive": bool(archive)}
        if value.get("state") == "deleted" and path_hint:
            yield {"receipt": display(receipt), "pointer": pointer, "raw_path": path_hint,
                   "expected_state": "deleted", "historical_archive": False}
        for key, child in value.items():
            if isinstance(child, str) and HEX.fullmatch(child) and pathlike(key):
                yield {"receipt": display(receipt), "pointer": f"{pointer}/{key}", "raw_path": key,
                       "expected_sha256": child.lower(), "historical_archive": False}
            elif isinstance(child, (dict, list)):
                yield from references(child, receipt, f"{pointer}/{key}", key if pathlike(key) else None)


def strings(value, pointer=""):
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from strings(child, f"{pointer}/{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings(child, f"{pointer}/{key}")
    elif pathlike(value):
        yield pointer, value


def main():
    destination = HERE / (sys.argv[1] if len(sys.argv) == 2 else "snapshot-1")
    if destination.parent != HERE or destination.exists():
        raise ValueError("Choose a new immediate child snapshot directory; no overwrites")
    destination.mkdir()
    snapshot = datetime.now(timezone.utc).isoformat()
    files = [path for scope in (EVIDENCE, EXTRA_EVIDENCE) for path in scope.rglob("*")
             if path.is_file() and HERE not in path.parents]
    file_hashes = {display(path): sha(path) for path in files}
    hash_locations = defaultdict(list)
    for path, digest in file_hashes.items():
        hash_locations[digest].append(path)
    ignore = subprocess.run(["git", "check-ignore", "-v", "--stdin"], cwd=REPO,
                            input="\n".join(file_hashes) + "\n", capture_output=True, text=True)
    ignored = {}
    unignored_log_rules = []
    for line in ignore.stdout.splitlines():
        rule, path = line.split("\t", 1)
        if rule.rsplit(":", 1)[-1].startswith("!"):
            unignored_log_rules.append({"path": path, "rule": rule})
            continue
        ignored[path] = rule
    ignored_files = []
    for path, rule in ignored.items():
        aliases = [item for item in hash_locations[file_hashes[path]] if item != path and item not in ignored]
        ignored_files.append({"path": path, "rule": rule, "sha256": file_hashes[path],
                              "byte_identical_unignored_copies": aliases,
                              "disposable": "__pycache__" in path or path.endswith(".pyc")})

    extra_records = {"report.json", "original-sources.json", "source-changes.json", "log-migration.json", "original-production-sources.json"}
    manifests = [path for path in files if path.suffix == ".json" and
                 (any(part in path.name.lower() for part in ("receipt", "manifest", "source-files", "source-before", "sha256")) or
                  path.name in extra_records)]
    refs, parse_errors, missing_strings, manifest_hashes = [], [], [], {}
    for receipt in manifests:
        manifest_hashes[display(receipt)] = sha(receipt)
        try:
            value = json.loads(receipt.read_text())
        except (ValueError, OSError) as exc:
            parse_errors.append({"path": display(receipt), "error": str(exc)})
            continue
        for ref in references(value, receipt):
            path = resolve(ref["raw_path"], receipt)
            ref["resolved_path"] = display(path)
            ref["exists"] = path.is_file()
            if ref["exists"]:
                ref["current_sha256"] = sha(path)
            if "expected_sha256" in ref:
                ref["hash_matches"] = ref.get("current_sha256") == ref["expected_sha256"]
                if not ref["hash_matches"]:
                    ref["preserved_bytes_elsewhere"] = hash_locations.get(ref["expected_sha256"], [])
            refs.append(ref)
        for pointer, raw in strings(value):
            resolved = resolve(raw, receipt)
            if not resolved.is_file():
                missing_strings.append({"receipt": display(receipt), "pointer": pointer,
                                        "raw_path": raw, "resolved_path": display(resolved)})

    # A source may have several valid historical epochs. A newer matching receipt
    # covers the current bytes; do not rewrite the earlier receipt to pretend so.
    by_source = defaultdict(list)
    for ref in refs:
        if (not is_evidence_path(ref["resolved_path"]) and not ref["resolved_path"].startswith("/") and
                not ref.get("historical_archive")):
            by_source[ref["resolved_path"]].append(ref)
    source_drift = []
    for path, observations in sorted(by_source.items()):
        mismatched = [item for item in observations if item.get("hash_matches") is False or
                      (item.get("expected_state") == "deleted" and item["exists"])]
        if mismatched:
            matches = [item for item in observations if item.get("hash_matches") is True]
            source_drift.append({"path": path, "current_sha256": sha(REPO / path) if (REPO / path).is_file() else None,
                                 "current_covered_by": sorted(set(item["receipt"] for item in matches)),
                                 "historical_mismatches": mismatched})
    artifact_mismatches = [ref for ref in refs if is_evidence_path(ref["resolved_path"]) and
                           ref.get("hash_matches") is False]
    missing_hash_refs = [ref for ref in refs if not ref["exists"] and "expected_sha256" in ref]
    markdown_links = []
    for document in files:
        if document.suffix != ".md":
            continue
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            raw = match.group(1).strip().strip("<>").split("#", 1)[0]
            if not raw or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw):
                continue
            # Markdown filesystem links resolve against their document, except
            # the repo-root paths deliberately used by the audit's text links.
            target = resolve(raw, document)
            markdown_links.append({"document": display(document), "raw_path": raw,
                                   "resolved_path": display(target), "exists": target.exists()})

    source_path, ledger_path = AUDIT / "wave-1-source.json", AUDIT / "remediation-ledger.json"
    source_bytes, ledger_bytes = source_path.read_bytes(), ledger_path.read_bytes()
    source, ledger = json.loads(source_bytes), json.loads(ledger_bytes)
    expected_ids = {f"W1-{index:03d}" for index in range(1, 129)}
    source_ids = [row["id"] for row in source["findings"]]
    ledger_ids = [row["id"] for row in ledger["findings"]]
    package_ids = [identifier for package in ledger["packages"] for identifier in package["finding_ids"]]
    source_map = {row["id"]: row for row in source["findings"]}
    changed_fields = []
    for row in ledger["findings"]:
        original = source_map.get(row["id"], {})
        for ledger_key, source_key in (("original_title", "title"), ("original_location", "location"),
                                       ("original_severity", "severity")):
            if row.get(ledger_key) != original.get(source_key):
                changed_fields.append({"id": row["id"], "field": ledger_key, "ledger": row.get(ledger_key),
                                       "source": original.get(source_key)})
    ledger_check = {
        "ledger_snapshot_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "ledger_changed_while_reading": ledger_bytes != ledger_path.read_bytes(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "recorded_source_sha256": ledger["source_manifest"]["wave_1"]["sha256"],
        "source_hash_matches": hashlib.sha256(source_bytes).hexdigest() == ledger["source_manifest"]["wave_1"]["sha256"],
        "source_count": len(source_ids), "ledger_count": len(ledger_ids),
        "source_ids_exact": len(source_ids) == 128 and set(source_ids) == expected_ids,
        "ledger_ids_exact": len(ledger_ids) == 128 and set(ledger_ids) == expected_ids,
        "package_assignment_count": len(package_ids),
        "package_ids_exact_once": len(package_ids) == 128 and set(package_ids) == expected_ids,
        "source_severities": dict(Counter(row["severity"] for row in source["findings"])),
        "original_field_drift": changed_fields,
        "wave2_required": ledger["source_manifest"]["wave_2"].get("required_for_goal_completion"),
        "wave2_status": ledger["source_manifest"]["wave_2"].get("status"),
    }
    result = {"snapshot_utc": snapshot, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "evidence_files_scanned": len(files), "manifests_scanned": len(manifests),
              "explicit_hash_references": len(refs), "parse_errors": parse_errors,
              "manifest_snapshot_sha256": manifest_hashes, "ignored_files": ignored_files,
              "scoped_log_exceptions": unignored_log_rules,
              "artifact_hash_mismatches": artifact_mismatches, "missing_hash_references": missing_hash_refs,
              "source_hash_drift": source_drift, "unresolved_path_strings_for_manual_triage": missing_strings,
              "markdown_file_links": markdown_links,
              "manifest_changed_during_scan": [path for path, digest in manifest_hashes.items()
                                               if sha(REPO / path) != digest],
              "ledger": ledger_check}
    (destination / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "all-hash-references.json").write_text(json.dumps(refs, indent=2) + "\n")
    print(json.dumps({"manifests": len(manifests), "references": len(refs), "ignored": len(ignored_files),
                      "artifact_hash_mismatches": len(artifact_mismatches), "missing_hash_references": len(missing_hash_refs),
                      "source_paths_with_drift": len(source_drift), "unresolved_path_strings": len(missing_strings),
                      "ledger": ledger_check}, indent=2))


if __name__ == "__main__":
    main()
