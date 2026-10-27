#!/usr/bin/env python3
"""Validate portable pipeline settings and print their normalized values.

Why: identity, repositories, local schedule, theme and a short GPU smoke limit
belong in an owner profile rather than copies of the pipeline's source code.
This tool reads JSON only. It never schedules, publishes, connects or installs.

Usage:
    python3 tools/pipeline_config.py pipeline.example.json
    python3 tools/pipeline_config.py pipeline.json --schedule
    python3 tools/pipeline_config.py pipeline.json --public-root /checkout/public
    python3 tools/pipeline_config.py pipeline.json --drafts-root /checkout/drafts
    python3 tools/pipeline_config.py pipeline.json --get repositories.public
    python3 -m unittest tools.test_pipeline_config

Outputs: normalized JSON by default, schedule JSON with --schedule, a scalar
with --get dotted.key, or a resolved subdirectory with a root option. Paths are
relative to their repository checkout. The theme is an optional name string.
Exit 0 for valid settings. Exit 1 with an error on stderr for invalid settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
REPOSITORY_NAME = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
REPOSITORY_PATH = re.compile(r"[A-Za-z0-9._/-]+\Z")
SSH_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z")
THEME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SECRET_FIELD = re.compile(r"token|password|passwd|credential|secret|api.?key|private.?key|access.?key", re.I)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def _validate(value):
    errors = []

    def object_fields(item, label, required, optional=()):
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object")
            return {}
        for key in sorted(set(required) - set(item)):
            errors.append(f"{label}.{key} is required")
        for key in sorted(set(item) - set(required) - set(optional)):
            if SECRET_FIELD.search(key):
                errors.append(f"{label}.{key}: secret and credential fields are forbidden")
            else:
                errors.append(f"{label}.{key}: unknown field")
        return item

    def text(item, label):
        if not isinstance(item, str) or not item.strip() or any(ord(character) < 32 or ord(character) == 127 for character in item):
            errors.append(f"{label} must be a nonempty single-line string")
            return ""
        return item.strip()

    def owner(item, label):
        item = text(item, label)
        if not OWNER.fullmatch(item) or "--" in item:
            errors.append(f"{label} must be a valid GitHub owner name")
        return item

    def repository(item, label):
        item = text(item, label)
        parts = item.split("/")
        if len(parts) != 2:
            errors.append(f"{label} must use owner/repository syntax")
            return item
        owner(parts[0], f"{label} owner")
        if not REPOSITORY_NAME.fullmatch(parts[1]) or parts[1] in (".", ".."):
            errors.append(f"{label} must have a valid repository name")
        return item

    def relative_path(item, label):
        original = item
        item = text(item, label)
        path = PurePosixPath(item)
        if not isinstance(original, str) or not REPOSITORY_PATH.fullmatch(original) or path.is_absolute() or ".." in item.split("/") or any(part.lower() == ".git" for part in path.parts):
            errors.append(f"{label} must be a safe repository-relative path, or '.'")
            return item
        return path.as_posix()

    root = object_fields(value, "config", ("schema_version", "identity", "repositories", "schedule", "gpu", "publication"), ("theme",))
    if type(root.get("schema_version")) is not int or root["schema_version"] != 1:
        errors.append("schema_version must be 1")
    identity = object_fields(root.get("identity"), "identity", ("name", "github_owner"))
    normalized_identity = {"name": text(identity.get("name"), "identity.name"), "github_owner": owner(identity.get("github_owner"), "identity.github_owner")}
    repositories = object_fields(root.get("repositories"), "repositories", ("drafts", "public", "public_path"), ("drafts_path",))
    normalized_repositories = {
        "drafts": repository(repositories.get("drafts"), "repositories.drafts"),
        "public": repository(repositories.get("public"), "repositories.public"),
        "public_path": relative_path(repositories.get("public_path"), "repositories.public_path"),
        "drafts_path": relative_path(repositories.get("drafts_path", "posts"), "repositories.drafts_path"),
    }
    if normalized_repositories["drafts"].split("/", 1)[0].lower() != normalized_identity["github_owner"].lower():
        errors.append("repositories.drafts owner must match identity.github_owner")
    if normalized_repositories["drafts"].lower() == normalized_repositories["public"].lower():
        errors.append("repositories.drafts and repositories.public must differ")
    schedule = object_fields(root.get("schedule"), "schedule", ("timezone", "hours", "minute"))
    timezone = text(schedule.get("timezone"), "schedule.timezone")
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError):
        errors.append("schedule.timezone must be an available IANA timezone")
    hours = schedule.get("hours")
    if not isinstance(hours, list) or not hours or any(type(hour) is not int or not 0 <= hour <= 23 for hour in hours):
        errors.append("schedule.hours must be a nonempty list of integer hours from 0 through 23")
        hours = []
    elif len(hours) != len(set(hours)):
        errors.append("schedule.hours must be unique")
    minute = schedule.get("minute")
    if type(minute) is not int or not 0 <= minute <= 59:
        errors.append("schedule.minute must be an integer from 0 through 59")
    gpu = object_fields(root.get("gpu"), "gpu", ("ssh_host", "max_smoke_seconds"))
    ssh_host = text(gpu.get("ssh_host"), "gpu.ssh_host")
    if not SSH_HOST.fullmatch(ssh_host):
        errors.append("gpu.ssh_host must be an SSH host alias or hostname, without commands or credentials")
    smoke_seconds = gpu.get("max_smoke_seconds")
    if type(smoke_seconds) is not int or not 1 <= smoke_seconds <= 300:
        errors.append("gpu.max_smoke_seconds must be an integer from 1 through 300")
    publication = object_fields(root.get("publication"), "publication", ("mode",))
    if publication.get("mode") != "draft_only":
        errors.append("publication.mode must be draft_only")
    normalized = {"schema_version": 1, "identity": normalized_identity, "repositories": normalized_repositories, "schedule": {"timezone": timezone, "hours": sorted(hours), "minute": minute}, "gpu": {"ssh_host": ssh_host, "max_smoke_seconds": smoke_seconds}, "publication": {"mode": "draft_only"}}
    if "theme" in root:
        theme = text(root["theme"], "theme")
        if not THEME.fullmatch(theme):
            errors.append("theme must be a name using letters, numbers, dots, underscores or hyphens")
        normalized["theme"] = theme
    if errors:
        raise ValueError("\n".join(errors))
    return normalized


def load_config(path: str | Path) -> dict:
    """Read and normalize a strict JSON owner profile, or raise ValueError."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read pipeline config {path}: {exc}") from exc
    return _validate(value)


def _repository_root(config, checkout, kind):
    checkout = Path(checkout).resolve()
    result = (checkout / config["repositories"][f"{kind}_path"]).resolve()
    if not result.is_relative_to(checkout):
        raise ValueError(f"repositories.{kind}_path resolves outside the checkout")
    return result


def _get_scalar(config, key):
    value = config
    for part in key.split("."):
        if not part or not isinstance(value, dict) or part not in value:
            raise ValueError(f"configuration key not found: {key}")
        value = value[part]
    if isinstance(value, (dict, list)):
        raise ValueError(f"configuration key must identify a scalar: {key}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--schedule", action="store_true", help="Print timezone, local hours and minute as JSON")
    output.add_argument("--public-root", type=Path, metavar="CHECKOUT", help="Resolve public_path within this checkout")
    output.add_argument("--drafts-root", type=Path, metavar="CHECKOUT", help="Resolve drafts_path within this checkout")
    output.add_argument("--get", dest="get_key", metavar="DOTTED.KEY", help="Print one scalar configuration value")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.get_key is not None:
            value = _get_scalar(config, args.get_key)
            print(value if isinstance(value, str) else json.dumps(value))
        elif args.public_root is not None:
            print(_repository_root(config, args.public_root, "public"))
        elif args.drafts_root is not None:
            print(_repository_root(config, args.drafts_root, "drafts"))
        else:
            print(json.dumps(config["schedule"] if args.schedule else config, indent=2, sort_keys=True))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
