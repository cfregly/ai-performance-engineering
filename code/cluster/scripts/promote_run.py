#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit


ARTIFACT_DIRS = ("structured", "raw", "figures", "reports")
LOCAL_LINK = re.compile(r'(?<=\]\()[^\s)]+|(?<=href=")[^"]+|(?<=src=")[^"]+')
STAGE_MARKER = ".owned-stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote one run-local cluster result tree into the published cluster package."
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--run-id", help="Run id under cluster/runs/<run_id>.")
    operation.add_argument("--recover", action="store_true", help="Recover an interrupted publication.")
    parser.add_argument("--label", default="localhost", help="Host label for localhost report rendering.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root for testing or custom promotion targets.",
    )
    parser.add_argument(
        "--publish-report-path",
        default="cluster/field-report-localhost.md",
        help="Published localhost report path (default: cluster/field-report-localhost.md).",
    )
    parser.add_argument(
        "--publish-notes-path",
        default="cluster/field-report-localhost-notes.md",
        help="Published localhost notes path (default: cluster/field-report-localhost-notes.md).",
    )
    parser.add_argument(
        "--skip-render-localhost-report",
        action="store_true",
        help="Skip rendering run-local + published localhost report markdown.",
    )
    parser.add_argument(
        "--skip-validate-localhost-report",
        action="store_true",
        help="Skip the localhost report check before publication.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup_run_artifacts.sh after promotion using this run_id as canonical.",
    )
    parser.add_argument(
        "--allow-run-id",
        action="append",
        default=[],
        help="Additional run ids to retain during cleanup/validation hygiene checks.",
    )
    return parser.parse_args()


def _repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def _script_root() -> Path:
    return Path(__file__).resolve().parent


def _copy_overwrite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir() and not src.is_symlink():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def _sync_children(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_dir.iterdir()):
        _copy_overwrite(child, dst_dir / child.name)
        copied.append(child.name)
    return copied


def _run_subprocess(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }


class RollbackError(RuntimeError):
    """Publishing failed and a backup must be retained for recovery."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_projection(
    manifest_path: Path,
    manifest: dict,
    files: list[str],
    hashes: dict | None,
    **changes: Any,
) -> None:
    projection = manifest.setdefault("publication_projection", {})
    if not isinstance(projection, dict):
        raise ValueError("Malformed publication projection in candidate manifest.")
    projection.setdefault("source_manifest_sha256", _sha256(manifest_path))
    projection.update(changes)
    summary = manifest.get("summary") or {}
    summary["file_count"] = len(files)
    summary["artifact_counts"] = dict(Counter(Path(name).suffix.lstrip(".") or "no_ext" for name in files))
    if hashes is not None:
        summary["sha256"] = hashes
    manifest.update(files=files, summary=summary)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _validate_candidate(candidate: Path, run_id: str) -> None:
    for name in (*ARTIFACT_DIRS, "manifest.json"):
        artifact = candidate / name
        if artifact.is_symlink() or any(path.is_symlink() for path in artifact.rglob("*")):
            raise ValueError(f"Candidate must contain actual artifact files under {name}.")
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise ValueError(f"Candidate manifest does not identify run {run_id}.")
    if manifest.get("manifest_mode", "run_dir") != "run_dir":
        raise ValueError("Promotion requires a run-local manifest.")
    if manifest.get("success") is False or manifest.get("finalized") is False:
        raise ValueError(f"Candidate manifest records an unsuccessful or unfinished run {run_id}.")
    git_provenance = manifest.get("git")
    if isinstance(git_provenance, dict) and git_provenance.get("probe_ok") is False:
        raise ValueError(f"Candidate manifest records a failed Git provenance probe for {run_id}.")

    steps_path = f"structured/{run_id}_suite_steps.json"
    steps = json.loads((candidate / steps_path).read_text(encoding="utf-8"))
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(step, dict) and isinstance(step.get("name"), str) and step["name"] for step in steps)
    ):
        raise ValueError(f"Candidate has no valid suite step records for {run_id}.")
    latest_steps = {step["name"]: step for step in steps}
    if any(str(step.get("exit_code")) != "0" for step in latest_steps.values()):
        raise ValueError(f"Candidate suite has unfinished or failed steps for {run_id}.")

    modern = manifest.get("manifest_version") == 2
    files = manifest.get("files")
    if files is None:
        if modern:
            raise ValueError("Version 2 candidate manifest needs an artifact inventory.")
        return
    if (
        not isinstance(files, list)
        or not all(isinstance(name, str) for name in files)
        or steps_path not in files
        or len(set(files)) != len(files)
    ):
        raise ValueError("Candidate manifest must inventory its suite steps and artifact paths.")
    summary = manifest.get("summary") or {}
    if not isinstance(summary, dict) or summary.get("file_count", len(files)) != len(files):
        raise ValueError("Candidate manifest file count does not match its inventory.")
    hashes = summary.get("sha256")
    if modern and hashes is None:
        raise ValueError("Version 2 candidate manifest needs artifact hashes.")
    if hashes is not None and (not isinstance(hashes, dict) or set(hashes) != set(files)):
        raise ValueError("Candidate manifest hashes do not match its inventory.")
    legacy_receipts = {
        f"raw/{run_id}_postrun_promote_watch_status.json",
        f"raw/{run_id}_postrun_promote_watch.launch.log",
        f"raw/{run_id}_promote.log",
        f"raw/activity/{run_id}_activity.log",
        f"raw/activity/{run_id}_shell.log",
    }
    excluded = set()
    for name in files:
        relative = Path(name)
        path = candidate / relative
        if (
            not relative.parts
            or relative.parts[0] not in ARTIFACT_DIRS
            or ".." in relative.parts
            or not path.resolve().is_relative_to(candidate)
        ):
            raise ValueError(f"Candidate manifest artifact unavailable: {name}")
        if not path.is_file():
            if name in legacy_receipts:
                excluded.add(name)
                continue
            raise ValueError(f"Candidate manifest artifact unavailable: {name}")
        if hashes is not None and _sha256(path) != hashes[name]:
            if name not in legacy_receipts:
                raise ValueError(f"Candidate manifest hash mismatch: {name}")
            excluded.add(name)

    if modern:
        copied_files = {
            str(path.relative_to(candidate))
            for name in ARTIFACT_DIRS
            for path in (candidate / name).rglob("*")
            if path.is_file()
        }
        unlisted = copied_files.difference(files)
        if unlisted.difference(legacy_receipts):
            raise ValueError(f"Candidate has unlisted artifacts: {', '.join(sorted(unlisted - legacy_receipts))}")
        excluded.update(unlisted)

    if excluded:
        files = [name for name in files if name not in excluded]
        roles = manifest.get("artifact_roles")
        if isinstance(roles, dict):
            manifest["artifact_roles"] = {
                role: [name for name in members if name not in excluded] if isinstance(members, list) else members
                for role, members in roles.items()
            }
        for name in excluded:
            path = candidate / name
            if path.is_file():
                path.unlink()
        _write_projection(
            manifest_path, manifest, files,
            {name: hashes[name] for name in files} if hashes is not None else None,
            excluded_mutable_operational_files=sorted(excluded),
        )


def _record_rendered_reports(candidate: Path, reports: list[str]) -> None:
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if files is None:
        return
    hashes = (manifest.get("summary") or {}).get("sha256")
    if hashes is not None:
        hashes.update({name: _sha256(candidate / name) for name in reports})
    _write_projection(
        manifest_path, manifest, sorted([*files, *reports]), hashes, rendered_reports=sorted(reports)
    )


def _replace_report(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".promotion-report-", dir=destination.parent) as scratch:
        incoming = Path(scratch) / "report"
        shutil.copy2(source, incoming, follow_symlinks=False)
        incoming.replace(destination)


def _write_report_variant(
    source: Path, staged: Path, final_parent: Path, cluster_root: Path
) -> None:
    staged.parent.mkdir(parents=True, exist_ok=True)
    if final_parent == cluster_root:
        shutil.copy2(source, staged)
        return

    def rebase(match: re.Match[str]) -> str:
        link = match.group()
        target = urlsplit(link)
        if target.scheme or target.netloc or not target.path or target.path.startswith("/") or "<" in link:
            return link
        relative = os.path.relpath(cluster_root / unquote(target.path), final_parent)
        return urlunsplit(("", "", quote(relative, safe="/"), target.query, target.fragment))

    content = LOCAL_LINK.sub(rebase, source.read_text(encoding="utf-8"))
    staged.write_text(content, encoding="utf-8")


def _restore_publication(stage: Path, published: Path) -> None:
    candidate = stage / "candidate"
    rollback = stage / "rollback"
    previous = rollback / "package"
    journal = rollback / "pending.json"
    try:
        transaction = json.loads(journal.read_text(encoding="utf-8"))
        for index, (name, had_report) in enumerate(transaction["reports"]):
            destination = Path(name)
            if had_report:
                _replace_report(rollback / f"report-{index}", destination)
            elif destination.exists() or destination.is_symlink():
                destination.unlink()
        if previous.exists() or previous.is_symlink():
            if published.exists() or published.is_symlink():
                published.replace(candidate)
            previous.replace(published)
        elif not candidate.exists():
            if transaction["had_package"]:
                raise ValueError("The previous package backup is missing.")
            published.replace(candidate)
        journal.unlink()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RollbackError(f"Publication recovery required in {rollback}: {exc}") from exc


def _recover_publications(published: Path) -> list[str]:
    recovered = []
    for stage in sorted(published.parent.glob(".promotion-*")):
        if stage.is_symlink():
            continue
        journal = stage / "rollback" / "pending.json"
        if journal.is_file() and not journal.is_symlink():
            _restore_publication(stage, published)
            shutil.rmtree(stage)
            recovered.append(stage.name)
        elif not journal.exists() and (stage / STAGE_MARKER).is_file():
            shutil.rmtree(stage)
            recovered.append(stage.name)
    return recovered


def _install_publication(candidate: Path, published: Path, reports: dict[Path, Path]) -> None:
    stage = candidate.parent
    rollback = stage / "rollback"
    rollback.mkdir()
    had_package = published.exists() or published.is_symlink()
    report_backups = []
    for index, destination in enumerate(reports):
        had_report = destination.exists() or destination.is_symlink()
        if had_report:
            shutil.copy2(destination, rollback / f"report-{index}", follow_symlinks=False)
        report_backups.append((str(destination), had_report))

    journal = rollback / "pending.json"
    draft = rollback / "journal.json"
    draft.write_text(json.dumps({"had_package": had_package, "reports": report_backups}), encoding="utf-8")
    draft.replace(journal)
    try:
        if had_package:
            published.replace(rollback / "package")
        candidate.replace(published)
        for destination, source in reports.items():
            _replace_report(source, destination)
        journal.unlink()
    except BaseException:
        if journal.exists():
            _restore_publication(stage, published)
        raise


def _protect_report_links(plan: Path, repo_root: Path, reports: set[Path]) -> None:
    targets = [repo_root / item["path"] for item in json.loads(plan.read_text())["targets"]]
    for report in sorted(reports):
        if not report.is_file():
            continue
        for match in LOCAL_LINK.finditer(report.read_text(encoding="utf-8")):
            link = urlsplit(match.group())
            if link.scheme or link.netloc or not link.path:
                continue
            asset = (report.parent / unquote(link.path)).resolve()
            if any(asset.is_relative_to(target) for target in targets):
                raise ValueError(f"Cleanup would remove evidence linked by {report.name}: {asset}")


def _promote(
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    published: Path,
    report: Path,
    notes: Path,
    stage: Path,
    summary: dict[str, Any],
) -> None:
    cluster_root = repo_root / "cluster"
    candidate = stage / "candidate"
    candidate.mkdir()
    managed = {*ARTIFACT_DIRS, "manifest.json"}
    if published.is_dir():
        for child in published.iterdir():
            if child.name not in managed:
                _copy_overwrite(child, candidate / child.name)
    if any((run_dir / name).is_symlink() for name in managed):
        raise ValueError("Candidate artifact roots must be contained in the run directory.")
    copied = {name: _sync_children(run_dir / name, candidate / name) for name in ARTIFACT_DIRS}
    _copy_overwrite(run_dir / "manifest.json", candidate / "manifest.json")
    _validate_candidate(candidate, args.run_id)

    steps = summary["steps"]
    outputs: dict[Path, Path] = {}
    staged_report, staged_notes = report, notes
    if not args.skip_render_localhost_report:
        rendered_report = stage / "rendered-report.md"
        rendered_notes = stage / "rendered-notes.md"
        render_cmd = [
            sys.executable,
            str(_script_root() / "render_localhost_field_report_package.py"),
            "--run-id", args.run_id,
            "--label", args.label,
            "--root", str(cluster_root),
            "--run-dir", str(candidate),
            "--report", str(stage / "run-report.md"),
            "--notes", str(stage / "run-notes.md"),
            "--publish-report", str(rendered_report),
            "--publish-notes", str(rendered_notes),
        ]
        steps["render_localhost_report"] = _run_subprocess(render_cmd, cwd=repo_root)
        if not steps["render_localhost_report"]["success"]:
            raise ValueError("Candidate report rendering failed.")

        targets = [(report, rendered_report), (notes, rendered_notes)]
        if args.label == "localhost":
            targets.extend([
                (cluster_root / "field-report.md", rendered_report),
                (cluster_root / "field-report-notes.md", rendered_notes),
            ])
        sources: dict[Path, Path] = {}
        archive_names: dict[str, Path] = {}
        for destination, source in targets:
            if (
                destination.is_relative_to(published)
                or sources.get(destination, source) != source
                or archive_names.get(destination.name, source) != source
            ):
                raise ValueError(f"Conflicting publication output path: {destination}")
            sources[destination] = source
            archive_names[destination.name] = source
        for index, (destination, source) in enumerate(sources.items()):
            staged = stage / f"report-{index}" / destination.name
            _write_report_variant(source, staged, destination.parent, cluster_root)
            outputs[destination] = staged
        staged_report, staged_notes = outputs[report], outputs[notes]
        rendered_reports = []
        for name, source in archive_names.items():
            destination = candidate / "reports" / name
            if destination.exists():
                destination = candidate / "reports" / "publication" / name
                if destination.exists():
                    raise ValueError(f"Candidate already contains the generated publication report {name}.")
            final_parent = published / destination.parent.relative_to(candidate)
            _write_report_variant(source, destination, final_parent, cluster_root)
            rendered_reports.append(str(destination.relative_to(candidate)))
        _record_rendered_reports(candidate, rendered_reports)
    if not args.skip_validate_localhost_report:
        validate_cmd = [
            "bash", str(_script_root() / "validate_field_report_requirements.sh"),
            "--repo-root", str(repo_root),
            "--report", str(staged_report),
            "--notes", str(staged_notes),
            "--published-root", str(candidate),
            "--report-link-base", str(report.parent),
            "--notes-link-base", str(notes.parent),
            "--canonical-run-id", args.run_id,
            "--package-only",
        ]
        steps["validate_localhost_report"] = _run_subprocess(validate_cmd, cwd=repo_root)
        if not steps["validate_localhost_report"]["success"]:
            raise ValueError("Candidate report validation failed.")

    _install_publication(candidate, published, outputs)
    steps["publish_package_sync"] = {"success": True, **copied, "manifest": True}
    summary["published"] = True
    if outputs and args.label == "localhost":
        steps["sync_top_level_reports"] = {
            "success": True,
            "field_report_path": str(cluster_root / "field-report.md"),
            "field_report_notes_path": str(cluster_root / "field-report-notes.md"),
        }

    if args.cleanup:
        plan = stage / "cleanup-plan.json"
        cleanup_cmd = [
            "bash", str(_script_root() / "cleanup_run_artifacts.sh"),
            "--repo-root", str(repo_root),
            "--canonical-run-id", args.run_id,
            "--plan-file", str(plan),
        ]
        for run_id in args.allow_run_id:
            if run_id:
                cleanup_cmd.extend(["--allow-run-id", run_id])
        steps["cleanup_plan"] = _run_subprocess(cleanup_cmd, cwd=repo_root)
        if not steps["cleanup_plan"]["success"]:
            raise ValueError("Publication succeeded. Cleanup planning failed.")
        _protect_report_links(
            plan, repo_root, {report, notes, cluster_root / "field-report.md", cluster_root / "field-report-notes.md"}
        )
        steps["cleanup"] = _run_subprocess([*cleanup_cmd, "--apply"], cwd=repo_root)
        if not steps["cleanup"]["success"]:
            raise ValueError("Publication succeeded. Cleanup did not complete.")


def main() -> int:
    args = parse_args()
    repo_root = _repo_root(args)
    cluster_root = repo_root / "cluster"
    run_dir = cluster_root / "runs" / (args.run_id or "")
    published = cluster_root / "published" / "current"
    report = (repo_root / args.publish_report_path).resolve()
    notes = (repo_root / args.publish_notes_path).resolve()
    summary: dict[str, Any] = {
        "success": False,
        "published": False,
        "run_id": args.run_id,
        "label": args.label,
        "run_dir": str(run_dir) if args.run_id else None,
        "published_root": str(published),
        **{f"published_{name}_dir": str(published / name) for name in ARTIFACT_DIRS},
        "published_manifest_path": str(published / "manifest.json"),
        "published_localhost_report_path": str(report),
        "published_localhost_notes_path": str(notes),
        "steps": {},
    }
    try:
        if not args.recover:
            if args.run_id in ("", ".", "..") or Path(args.run_id).name != args.run_id:
                raise ValueError("Run ID must name one directory under cluster/runs.")
            if not run_dir.is_dir() or not run_dir.resolve().is_relative_to((cluster_root / "runs").resolve()):
                raise ValueError(f"Missing or invalid run directory: {run_dir}")
        published.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(published.parent, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            recovered = _recover_publications(published)
            if recovered or args.recover:
                summary["steps"]["recover_publication"] = {"success": True, "stages": recovered}
            if args.recover:
                summary["success"] = True
            else:
                stage = Path(tempfile.mkdtemp(prefix=".promotion-", dir=published.parent))
                retain_stage = False
                try:
                    (stage / STAGE_MARKER).touch()
                    _promote(args, repo_root, run_dir, published, report, notes, stage, summary)
                    summary["success"] = True
                except RollbackError:
                    retain_stage = True
                    raise
                finally:
                    if not retain_stage:
                        shutil.rmtree(stage)
        finally:
            os.close(descriptor)
    except (OSError, ValueError, RollbackError) as exc:
        summary["error"] = str(exc)
        summary["success"] = False

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
