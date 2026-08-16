"""Fail-closed entry point for the retired automatic optimizer."""

import sys

from . import MIGRATION_MESSAGE


def main() -> int:
    """Explain the supported measured optimization path."""
    print(f"ERROR: {MIGRATION_MESSAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
