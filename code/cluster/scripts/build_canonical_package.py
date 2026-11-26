#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class RunSelection:
    run_id: str
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a clean, non-destructive canonical cluster package."
    )
    parser.add_argument("--canonical-run-id", required=True, help="Primary canonical run id to package.")
    parser.add_argument(
        "--comparison-run-id",
        action="append",
        default=[],
        help="Additional comparison/baseline run id to retain in the package.",
    )
    parser.add_argument(
        "--historical-run-id",
        action="append",
        default=[],
        help="Historical run id to preserve as an explicit reference or include when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="cluster/canonical_package",
        help="Package output directory (default: cluster/canonical_package).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root for testing or custom packaging.",
    )
    return parser.parse_args()


def belongs_to_run(path_name: str, run_id: str) -> bool:
    return path_name == run_id or path_name.startswith(f"{run_id}_") or path_name.startswith(f"{run_id}.")


def copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def gather_run_artifacts(root: Path, run_id: str) -> Dict[str, List[Path]]:
    results = {
        "structured": [],
        "raw": [],
        "figures": [],
    }
    run_dir = root / "cluster" / "runs" / run_id
    if run_dir.exists():
        for category in results:
            base = run_dir / category
            if not base.exists():
                continue
            results[category].extend(sorted(path for path in base.rglob("*") if path.is_file()))
        return results

    locations = {
        "structured": root / "cluster" / "results" / "structured",
        "raw": root / "cluster" / "results" / "raw",
        "figures": root / "cluster" / "docs" / "figures",
    }
    for category, base in locations.items():
        if not base.exists():
            continue
        for path in sorted(base.iterdir()):
            if belongs_to_run(path.name, run_id):
                results[category].append(path)
    return results


def find_markdown_references(root: Path, run_id: str) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    cluster_root = root / "cluster"
    if not cluster_root.exists():
        return refs
    for path in sorted(cluster_root.rglob("*.md")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        hits = []
        for idx, line in enumerate(lines, start=1):
            if run_id in line:
                hits.append({"line": idx, "text": line.strip()[:240]})
        if hits:
            refs.append({"path": str(path.relative_to(root)), "hits": hits})
    return refs


def _check_published_snapshot(snapshot: Path, run_id: str) -> bool:
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        return False
    files = manifest.get("files")
    summary = manifest.get("summary")
    if not isinstance(files, list) or not files or not isinstance(summary, dict):
        raise ValueError("Published manifest has no complete artifact inventory.")
    hashes = summary.get("sha256")
    actual = {}
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Published snapshot contains a symlink: {path.relative_to(snapshot)}")
        if path.is_file() and path != manifest_path:
            actual[str(path.relative_to(snapshot))] = path
    if (
        not all(isinstance(name, str) for name in files)
        or set(files) != set(actual)
        or len(files) != len(actual)
        or summary.get("file_count") != len(actual)
        or not isinstance(hashes, dict)
        or set(hashes) != set(actual)
    ):
        raise ValueError("Published manifest does not match the artifact inventory.")
    for name, path in actual.items():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != hashes[name]:
            raise ValueError(f"Published snapshot hash mismatch: {name}")
    return True


def copy_run_family(
    root: Path, out_dir: Path, run_id: str, *, published_snapshot: Path | None = None
) -> Dict[str, object]:
    live_run_dir = published_snapshot or root / "cluster" / "runs" / run_id
    if live_run_dir.exists():
        dst_run_dir = out_dir / "runs" / run_id
        copy_path(live_run_dir, dst_run_dir)
        if published_snapshot and not _check_published_snapshot(dst_run_dir, run_id):
            raise ValueError("Published snapshot changed during packaging.")
        counts = {
            category: sum(1 for path in (dst_run_dir / category).rglob("*") if path.is_file()) if (dst_run_dir / category).exists() else 0
            for category in ("structured", "raw", "figures", "reports")
        }
        return {
            "run_id": run_id,
            "present": True,
            "layout": "published_snapshot" if published_snapshot else "run_dir",
            "artifacts": {
                "run_dir": str(dst_run_dir.relative_to(out_dir)),
                "structured": [str(path.relative_to(out_dir)) for path in sorted((dst_run_dir / "structured").rglob("*")) if path.is_file()] if (dst_run_dir / "structured").exists() else [],
                "raw": [str(path.relative_to(out_dir)) for path in sorted((dst_run_dir / "raw").rglob("*")) if path.is_file()] if (dst_run_dir / "raw").exists() else [],
                "figures": [str(path.relative_to(out_dir)) for path in sorted((dst_run_dir / "figures").rglob("*")) if path.is_file()] if (dst_run_dir / "figures").exists() else [],
                "reports": [str(path.relative_to(out_dir)) for path in sorted((dst_run_dir / "reports").rglob("*")) if path.is_file()] if (dst_run_dir / "reports").exists() else [],
            },
            "counts": counts,
        }

    artifacts = gather_run_artifacts(root, run_id)
    copied: Dict[str, List[str]] = {"structured": [], "raw": [], "figures": []}
    for category, paths in artifacts.items():
        if not paths:
            continue
        dest_base = out_dir / "runs" / run_id / category
        for src in paths:
            dst = dest_base / src.name
            copy_path(src, dst)
            copied[category].append(str(dst.relative_to(out_dir)))
    present = any(copied.values())
    return {
        "run_id": run_id,
        "present": present,
        "layout": "legacy_materialized",
        "artifacts": copied,
        "counts": {key: len(value) for key, value in copied.items()},
    }


def copy_if_present(root: Path, out_dir: Path, rel_path: str) -> bool:
    src = root / rel_path
    if not src.exists():
        return False
    dst = out_dir / Path(rel_path).name
    copy_path(src, dst)
    return True


def write_package_readme(
    out_dir: Path,
    canonical_run_id: str,
    comparison_run_ids: List[str],
    historical_run_ids: List[str],
    copied_runs: List[Dict[str, object]],
    historical_refs: Dict[str, List[Dict[str, object]]],
) -> None:
    copied_map = {entry["run_id"]: entry for entry in copied_runs}
    lines = [
        "# Canonical Cluster Package",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Recommendation",
        "",
        "Keep exactly three run classes visible in the clean package:",
        "",
        f"- current canonical localhost run: `{canonical_run_id}`",
        f"- recent localhost comparison baseline(s): `{', '.join(comparison_run_ids) if comparison_run_ids else '<none>'}`",
        f"- historical multi-node reference run(s): `{', '.join(historical_run_ids) if historical_run_ids else '<none>'}`",
        "",
        "Do not delete the noisy source directories until the multi-node historical artifacts are either restored or intentionally retired.",
        "",
        "## Included Live Artifacts",
        "",
        "| Run id | Structured | Raw | Figures | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for run_id in [canonical_run_id, *comparison_run_ids, *historical_run_ids]:
        entry = copied_map.get(run_id)
        if entry is None:
            continue
        counts = entry["counts"]
        status = (
            "published snapshot" if entry.get("layout") == "published_snapshot"
            else "materialized" if entry["present"] else "reference-only"
        )
        lines.append(
            f"| `{run_id}` | {counts['structured']} | {counts['raw']} | {counts['figures']} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Package Contents",
            "",
            "- `runs/<run_id>/structured/`: selected structured outputs for packaged live runs.",
            "- `runs/<run_id>/raw/`: selected raw logs for packaged live runs.",
            "- `runs/<run_id>/figures/`: selected figures for packaged live runs.",
            "- `runs/<run_id>/reports/`: per-run localhost report package when present.",
            "- `field-report-localhost.md` + `field-report-localhost-notes.md`: current localhost canonical report package.",
            "- `historical-multinode-reference.md`: references to important multi-node history that is not live in this workspace.",
            "- `cleanup_keep_run_ids.txt`: keep-list to use with cleanup dry-runs or future pruning.",
            "",
            "## Cleanup Recommendation",
            "",
            "Preview cleanup with the keep-list from this package.",
            "",
            "```bash",
            "cluster/scripts/cleanup_run_artifacts.sh \\",
            f"  --canonical-run-id {canonical_run_id} \\",
            *[f"  --allow-run-id {run_id} \\" for run_id in [*comparison_run_ids, *historical_run_ids]],
            "  --plan-file ./cluster-cleanup-plan.json",
            "```",
            "",
            "Review the listed paths and verify retained history is available. Rerun the same command with `--apply` to execute the saved plan.",
            "",
        ]
    )

    missing_hist = [run_id for run_id in historical_run_ids if not copied_map.get(run_id, {}).get("present")]
    if missing_hist:
        lines.extend(
            [
                "## Missing Historical Runs",
                "",
                "These run ids are still important, but their underlying artifacts are not present in this workspace:",
                "",
            ]
        )
        for run_id in missing_hist:
            ref_count = len(historical_refs.get(run_id, []))
            lines.append(f"- `{run_id}`: reference-only ({ref_count} markdown file(s) mention it).")
        lines.append("")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_historical_reference_doc(
    out_dir: Path,
    historical_run_ids: List[str],
    historical_refs: Dict[str, List[Dict[str, object]]],
    copied_runs: List[Dict[str, object]],
) -> None:
    copied_map = {entry["run_id"]: entry for entry in copied_runs}
    lines = [
        "# Historical Multi-Node Reference",
        "",
        "This package preserves historical multi-node run ids as explicit references when the underlying artifacts are not present in the current workspace.",
        "",
    ]
    for run_id in historical_run_ids:
        entry = copied_map.get(run_id)
        if entry and entry["present"]:
            lines.extend(
                [
                    f"## `{run_id}`",
                    "",
                    "Artifacts for this run are present and were materialized into this package.",
                    "",
                ]
            )
            continue
        refs = historical_refs.get(run_id, [])
        lines.extend(
            [
                f"## `{run_id}`",
                "",
                "Artifacts are not present in this workspace. The run is retained here as a documented cleanup allowlist entry and historical reference.",
                "",
            ]
        )
        if not refs:
            lines.append("No markdown references were found in the current repo snapshot.")
            lines.append("")
            continue
        lines.append("| Source doc | Example lines |")
        lines.append("| --- | --- |")
        for ref in refs:
            hit_preview = "; ".join(f"{item['line']}: {item['text']}" for item in ref["hits"][:2])
            lines.append(f"| `{ref['path']}` | `{hit_preview}` |")
        lines.append("")

    (out_dir / "historical-multinode-reference.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_keep_list(out_dir: Path, canonical_run_id: str, comparison_run_ids: List[str], historical_run_ids: List[str]) -> None:
    keep_lines = [canonical_run_id, *comparison_run_ids, *historical_run_ids]
    (out_dir / "cleanup_keep_run_ids.txt").write_text("\n".join(keep_lines) + "\n", encoding="utf-8")


def write_manifest(
    out_dir: Path,
    canonical_run_id: str,
    comparison_run_ids: List[str],
    historical_run_ids: List[str],
    copied_runs: List[Dict[str, object]],
    copied_reports: List[str],
    historical_refs: Dict[str, List[Dict[str, object]]],
) -> None:
    payload = {
        "package_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_run_id": canonical_run_id,
        "comparison_run_ids": comparison_run_ids,
        "historical_run_ids": historical_run_ids,
        "copied_reports": copied_reports,
        "runs": copied_runs,
        "historical_markdown_references": {
            run_id: {"file_count": len(refs), "files": refs}
            for run_id, refs in historical_refs.items()
        },
    }
    (out_dir / "package_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.output_dir).resolve()

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory already exists and is not empty: {out_dir}")

    canonical_artifacts = gather_run_artifacts(repo_root, args.canonical_run_id)
    flat_manifest_name = f"{args.canonical_run_id}_manifest.json"
    has_run_artifacts = any(
        path.name != flat_manifest_name
        and (path.is_file() or any(child.is_file() for child in path.rglob("*")))
        for paths in canonical_artifacts.values()
        for path in paths
    )
    published_snapshot = None
    if not has_run_artifacts:
        candidate = repo_root / "cluster" / "published" / "current"
        try:
            if not _check_published_snapshot(candidate, args.canonical_run_id):
                raise ValueError(f"No canonical artifacts for run {args.canonical_run_id}.")
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        published_snapshot = candidate

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    selections: List[RunSelection] = [RunSelection(args.canonical_run_id, "canonical")]
    selections.extend(RunSelection(run_id, "comparison") for run_id in args.comparison_run_id)
    selections.extend(RunSelection(run_id, "historical") for run_id in args.historical_run_id)

    copied_runs: List[Dict[str, object]] = []
    for selection in selections:
        entry = copy_run_family(
            repo_root, out_dir, selection.run_id,
            published_snapshot=published_snapshot if selection.category == "canonical" else None,
        )
        entry["category"] = selection.category
        copied_runs.append(entry)

    copied_reports: List[str] = []
    for rel_path in (
        "cluster/field-report-localhost.md",
        "cluster/field-report-localhost-notes.md",
        "cluster/docs/advanced-runbook.md",
    ):
        if copy_if_present(repo_root, out_dir, rel_path):
            copied_reports.append(Path(rel_path).name)

    historical_refs = {
        run_id: find_markdown_references(repo_root, run_id)
        for run_id in args.historical_run_id
    }

    write_keep_list(out_dir, args.canonical_run_id, args.comparison_run_id, args.historical_run_id)
    write_historical_reference_doc(out_dir, args.historical_run_id, historical_refs, copied_runs)
    write_package_readme(
        out_dir,
        args.canonical_run_id,
        args.comparison_run_id,
        args.historical_run_id,
        copied_runs,
        historical_refs,
    )
    write_manifest(
        out_dir,
        args.canonical_run_id,
        args.comparison_run_id,
        args.historical_run_id,
        copied_runs,
        copied_reports,
        historical_refs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
