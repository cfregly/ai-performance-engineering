"""Compatibility shim for the retired automatic optimizer."""

from __future__ import annotations

import sys

from . import MIGRATION_MESSAGE


class AutoOptimizer:
    """Reject construction of the retired measured optimization engine."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(MIGRATION_MESSAGE)


def main() -> int:
    """Explain the supported campaign and benchmark harness path."""
    print(f"ERROR: {MIGRATION_MESSAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
