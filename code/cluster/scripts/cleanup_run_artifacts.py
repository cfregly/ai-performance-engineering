#!/usr/bin/env python3
"""Preview and apply cleanup of superseded cluster run artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from core.benchmark.artifact_manager import validate_run_id  # noqa: E402

RUNS = Path("cluster/runs")
STRUCTURED = Path("cluster/results/structured")
RAW = Path("cluster/results/raw")
FIGURES = Path("cluster/docs/figures")
BASES = (RUNS, STRUCTURED, RAW, FIGURES)
PUBLISHED = Path("cluster/published/current")
MANIFEST_RUN = re.compile(r"(20\d{2}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9._-]*)_manifest\.json")
DATED = re.compile(r"20\d{2}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9._-]*")
ORPHAN = re.compile(r"_[A-Za-z0-9]")
SMOKE = re.compile(r"mcp_cluster_smoke_test(?:_[A-Za-z0-9._-]+)?")


def _directory(root: Path, relative: Path) -> Path | None:
    path = root
    for component in relative.parts:
        path /= component
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Expected a real directory at {path}")
    return path


def _identity(info: os.stat_result) -> list[int]:
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def _read_regular(path: Path) -> tuple[bytes, list[int]]:
    fd = os.open(path, os.O_NOFOLLOW | os.O_NONBLOCK | os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Expected a regular file at {path}")
        content = stream.read()
        identity = _identity(info)
        if identity != _identity(os.fstat(fd)):
            raise ValueError(f"File changed while reading {path}")
    return content, identity


def _manifest(root: Path, relative: Path) -> dict[str, Any] | None:
    parent = _directory(root, relative.parent)
    if parent is None:
        return None
    try:
        content, identity = _read_regular(parent / relative.name)
    except FileNotFoundError:
        return None
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
        raise ValueError(f"Missing run_id in {relative}")
    return {
        "path": str(relative),
        "run_id": validate_run_id(payload["run_id"]),
        "identity": identity,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _canonical_context(
    root: Path, canonical: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    selected = []
    for relative in (RUNS / canonical / "manifest.json", STRUCTURED / f"{canonical}_manifest.json"):
        evidence = _manifest(root, relative)
        if evidence is not None:
            if evidence["run_id"] != canonical:
                raise ValueError(f"Canonical run ID {canonical} does not match {relative}")
            selected.append(evidence)
    published = _manifest(root, PUBLISHED / "manifest.json")
    if published is not None and published["run_id"] == canonical:
        selected.append(published)
    if not selected:
        raise ValueError(f"Canonical run ID {canonical} has no matching run or published manifest")
    return selected, published


def _belongs_to_run(name: str, run_id: str) -> bool:
    return name == run_id or name.startswith((f"{run_id}_", f"{run_id}."))


def _tree_digest(path: Path, device: int) -> str:
    digest = hashlib.sha256()

    def visit(entry: Path) -> None:
        info = entry.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)) or info.st_dev != device:
            raise ValueError(
                f"Cleanup target contains a symlink, mount, or special file at {entry}"
            )
        digest.update(json.dumps([str(entry.relative_to(path)), _identity(info)]).encode())
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(entry.iterdir()):
                visit(child)

    visit(path)
    return digest.hexdigest()


def _plan(root: Path, canonical: str, allow: set[str]) -> dict[str, Any]:
    evidence, published = _canonical_context(root, canonical)
    keep = allow | {canonical}
    if published is not None:
        keep.add(published["run_id"])

    directories = {relative: _directory(root, relative) for relative in BASES}
    stale_ids: set[str] = set()
    structured = directories[STRUCTURED]
    if structured is not None:
        for path in structured.iterdir():
            match = MANIFEST_RUN.fullmatch(path.name)
            if match is not None:
                stale_ids.add(match[1])
    runs = directories[RUNS]
    if runs is not None:
        stale_ids.update(
            path.name for path in runs.iterdir() if DATED.fullmatch(path.name) and path.is_dir()
        )
    stale_ids.difference_update(keep)

    targets = []
    base_identities = {}
    for relative, base in directories.items():
        if base is None:
            continue
        base_info = base.lstat()
        base_identities[str(relative)] = [base_info.st_dev, base_info.st_ino]
        for path in sorted(base.iterdir()):
            info = path.lstat()
            if relative == RUNS:
                stale = (
                    path.name not in keep
                    and (DATED.fullmatch(path.name) or SMOKE.fullmatch(path.name))
                    and (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode))
                )
            else:
                stale = not any(_belongs_to_run(path.name, run_id) for run_id in keep) and (
                    DATED.match(path.name)
                    or ORPHAN.match(path.name)
                    or SMOKE.fullmatch(path.name)
                    or any(_belongs_to_run(path.name, run_id) for run_id in stale_ids)
                )
            if stale:
                targets.append(
                    {
                        "path": str(path.relative_to(root)),
                        "identity": _identity(info),
                        "tree_sha256": _tree_digest(path, base_info.st_dev),
                    }
                )
    root_info = root.stat()
    return {
        "version": 1,
        "repo_root": str(root),
        "root_identity": [root_info.st_dev, root_info.st_ino],
        "canonical_run_id": canonical,
        "allow_run_ids": sorted(allow),
        "queue_lock_path": str(_queue_lock_path(root)),
        "canonical_manifests": evidence,
        "published_manifest": published,
        "stale_run_ids": sorted(stale_ids),
        "base_identities": base_identities,
        "targets": sorted(targets, key=lambda target: target["path"]),
    }


def _queue_lock_path(root: Path) -> Path:
    configured = os.environ.get("AISP_CLUSTER_SUITE_QUEUE_LOCK_PATH")
    lock_path = (
        Path(configured) if configured else root / "artifacts/parallel_runs/queue.runner.lock"
    )
    if not lock_path.is_absolute():
        lock_path = root / "cluster" / lock_path
    return Path(os.path.abspath(lock_path))


@contextmanager
def _queue_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"Expected a regular queue lock file at {lock_path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"Queue lock is held at {lock_path}. Retry after the active run exits"
            ) from exc
        yield
    finally:
        os.close(fd)


def _require_same_plan(reviewed: Any, current: dict[str, Any]) -> None:
    if reviewed == current:
        return
    if isinstance(reviewed, dict) and isinstance(reviewed.get("targets"), list):
        before = {
            target["path"]: target
            for target in reviewed["targets"]
            if isinstance(target, dict) and isinstance(target.get("path"), str)
        }
        after = {target["path"]: target for target in current["targets"]}
        for path in sorted(before.keys() | after.keys()):
            if before.get(path) != after.get(path):
                print(f"PLAN_DRIFT {path}", file=sys.stderr)
    raise ValueError(
        "Cleanup plan differs from the current selection or filesystem. Preview a new plan"
    )


def _apply(root: Path, plan: dict[str, Any], verbose: bool) -> None:
    if plan["targets"] and not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("Cleanup apply requires directory-relative filesystem operations")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        # Keep deletion tied to the directory identities recorded in the plan.
        root_fd = os.open(root, flags)
        stack.callback(os.close, root_fd)
        root_info = os.fstat(root_fd)
        if [root_info.st_dev, root_info.st_ino] != plan["root_identity"]:
            raise ValueError("Repository root changed during cleanup")

        base_fds = {}
        for target in plan["targets"]:
            path = Path(target["path"])
            base_key = str(path.parent)
            if base_key not in base_fds:
                fd = root_fd
                for component in path.parent.parts:
                    fd = os.open(component, flags, dir_fd=fd)
                    stack.callback(os.close, fd)
                info = os.fstat(fd)
                if [info.st_dev, info.st_ino] != plan["base_identities"][base_key]:
                    raise ValueError(f"Cleanup directory changed at {path.parent}")
                base_fds[base_key] = fd

        for target in plan["targets"]:
            path = Path(target["path"])
            fd = base_fds[str(path.parent)]
            info = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
            if _identity(info) != target["identity"]:
                raise ValueError(f"Cleanup target changed at {path}")
            if stat.S_ISDIR(info.st_mode):
                shutil.rmtree(path.name, dir_fd=fd)
            else:
                os.unlink(path.name, dir_fd=fd)
            if verbose:
                print(f"DELETE {root / path}")


def _print_plan(root: Path, plan: dict[str, Any]) -> None:
    print(f"canonical_run_id={plan['canonical_run_id']}")
    print(f"allow_run_ids={','.join(plan['allow_run_ids']) or '<none>'}")
    print(f"stale_run_ids={len(plan['stale_run_ids'])}")
    for run_id in plan["stale_run_ids"]:
        print(f"STALE_RUN {run_id}")
    print(f"stale_paths={len(plan['targets'])}")
    print(f"stale_files={len(plan['targets'])}")
    for target in plan["targets"]:
        print(f"STALE_PATH {root / target['path']}")
    if not plan["targets"]:
        print("No stale artifacts found.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="cleanup_run_artifacts.sh", description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=CODE_ROOT, help="Override the checkout root."
    )
    parser.add_argument("--canonical-run-id", required=True, help="Run ID to retain.")
    parser.add_argument(
        "--allow-run-id", action="append", default=[], help="Additional run ID to retain."
    )
    parser.add_argument("--plan-file", type=Path, help="Export a new plan or read it with --apply.")
    parser.add_argument(
        "--apply", action="store_true", help="Remove only paths in the accepted plan."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print each deletion as it finishes."
    )
    args = parser.parse_args()
    if args.apply and args.plan_file is None:
        parser.error("--apply requires --plan-file from a prior cleanup preview")
    try:
        root = args.repo_root.resolve(strict=True)
        canonical = validate_run_id(args.canonical_run_id)
        allow = {validate_run_id(run_id) for run_id in args.allow_run_id}
        if args.apply:
            reviewed = json.loads(_read_regular(args.plan_file)[0])
            with _queue_lock(_queue_lock_path(root)):
                current = _plan(root, canonical, allow)
                _require_same_plan(reviewed, current)
                _print_plan(root, current)
                _apply(root, current, args.verbose)
            print(f"Deleted {len(current['targets'])} stale artifact paths.")
        else:
            current = _plan(root, canonical, allow)
            _print_plan(root, current)
            if args.plan_file is not None:
                with args.plan_file.open("x", encoding="utf-8") as stream:
                    json.dump(current, stream, indent=2)
                    stream.write("\n")
                print(f"Plan saved to {args.plan_file}")
            print("Dry run. Use --apply with a saved --plan-file after checking the listed paths.")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
