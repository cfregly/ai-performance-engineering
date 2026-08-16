"""Input adapters retained from the retired automatic optimizer prototype."""

from .input_adapters import (
    BenchmarkAdapter,
    CodeSource,
    FileAdapter,
    RepoAdapter,
    detect_input_type,
)

MIGRATION_MESSAGE = (
    "The AutoOptimizer execution path has been retired because it does not satisfy the "
    "benchmark evidence contract. Use `python -m core.optimization.campaign --help` "
    "with measurements from the trusted benchmark harness."
)

__all__ = [
    "BenchmarkAdapter",
    "CodeSource",
    "FileAdapter",
    "MIGRATION_MESSAGE",
    "RepoAdapter",
    "detect_input_type",
]
