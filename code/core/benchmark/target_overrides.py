"""Parsing helpers for per-target benchmark arguments."""

from __future__ import annotations

import shlex
from collections.abc import Sequence


def parse_target_extra_args(entries: Sequence[str] | None) -> dict[str, list[str]]:
    """Parse repeatable ``target=arguments`` values and reject ambiguous input."""
    parsed: dict[str, list[str]] = {}
    normalized_targets: dict[str, str] = {}
    for entry in entries or []:
        target, separator, payload = str(entry).partition("=")
        target = target.strip()
        if not separator or not target or not payload.strip():
            raise ValueError(
                "Invalid --target-extra-arg value "
                f'{entry!r}. Expected target="--flag value" with a non-empty target and arguments.'
            )
        try:
            argv = shlex.split(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid --target-extra-arg value {entry!r}: {exc}") from exc
        if not argv:
            raise ValueError(f"Invalid --target-extra-arg value {entry!r}: arguments are empty.")
        normalized_target = normalize_target_label(target)
        if normalized_target in normalized_targets:
            previous = normalized_targets[normalized_target]
            raise ValueError(
                f"Duplicate --target-extra-arg target {target!r}. It resolves to the same "
                f"benchmark as {previous!r}."
            )
        parsed[target] = argv
        normalized_targets[normalized_target] = target
    return parsed


def normalize_target_label(label: str) -> str:
    """Normalize chapter separators while preserving the example name."""
    chapter, separator, example = str(label).partition(":")
    normalized_chapter = chapter.replace("\\", "/").replace("/", "_")
    return f"{normalized_chapter}:{example}" if separator else normalized_chapter
