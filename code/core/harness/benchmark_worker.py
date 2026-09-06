"""Explicit torchrun adapter for calling a named module entrypoint."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Import a benchmark module and call one explicit entrypoint."
    )
    parser.add_argument("--module", required=True)
    parser.add_argument("--callable", dest="callable_name", required=True)
    if "--" not in values:
        parser.error("an explicit '--' delimiter is required before target arguments")
    delimiter_index = values.index("--")
    return parser.parse_args(values[:delimiter_index]), values[delimiter_index + 1 :]


def run_named_entrypoint(
    module_name: str,
    callable_name: str,
    entrypoint_args: Sequence[str],
) -> int:
    """Call exactly the requested top-level function with its CLI arguments."""
    if not module_name or not callable_name.isidentifier():
        raise ValueError("--module and an identifier-valued --callable are required")

    module = importlib.import_module(module_name)
    entrypoint: Any = getattr(module, callable_name, None)
    if not isinstance(entrypoint, Callable):
        raise TypeError(
            f"Requested entrypoint {module_name}.{callable_name} is not callable"
        )

    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *entrypoint_args]
        result = entrypoint()
    finally:
        sys.argv = previous_argv

    if result is None:
        return 0
    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError(
            f"Entrypoint {module_name}.{callable_name} returned unsupported "
            f"status {result!r}; expected None or int"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args, entrypoint_args = _parse_args(argv)
    return run_named_entrypoint(args.module, args.callable_name, entrypoint_args)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
