"""Right-sized decode benchmark target with flag-based tier and quantization controls.

The target times the persistent Triton decode kernel. Pass the knobs with
`--target-extra-arg labs/persistent_decode:right_sized_decode="--tier large --num-programs 1"`.
The harness hands them to apply_target_overrides. Any other argument raises
ValueError, which fails the target. The CUDA graphs backend is a separate
target, labs/persistent_decode:persistent_decode_graphs, because the harness
builds the benchmark before it applies overrides, so an override cannot change
which benchmark class runs.
"""

from __future__ import annotations

import argparse
from typing import List

from core.harness.benchmark_harness import BaseBenchmark
from labs.persistent_decode.optimized_persistent_decode_triton import OptimizedPersistentDecodeTritonBenchmark
from labs.persistent_decode.persistent_decode_common import (
    DecodeOptions,
    decode_options_scope,
    parse_right_sized_overrides,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="right_sized_decode",
        description="Right-sized decode benchmark flags",
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False,
    )
    p.add_argument("--tier", choices=["small", "medium", "large"], default="medium", help="Decode tier sizing preset")
    p.add_argument(
        "--quantization",
        choices=["fp32", "fp16", "int4"],
        default="fp32",
        help="Decode quantization mode",
    )
    p.add_argument("--block-k", type=int, default=None, help="Override BLOCK_K (defaults depend on tier)")
    p.add_argument("--num-programs", type=int, default=None, help="Override number of Triton programs")
    return p


class RightSizedTritonDecodeBenchmark(OptimizedPersistentDecodeTritonBenchmark):
    """Persistent Triton decode at the tier, quantization and launch shape chosen by target overrides."""

    def __init__(self) -> None:
        self._configure(DecodeOptions())

    def _configure(self, options: DecodeOptions) -> None:
        """Rebuild every field the parent constructor derives from the decode options."""
        self.decode_options = options
        with decode_options_scope(options):
            OptimizedPersistentDecodeTritonBenchmark.__init__(self)

    def apply_target_overrides(self, argv: List[str]) -> None:
        self._configure(parse_right_sized_overrides(_build_parser(), argv, target="right_sized_decode optimized"))

    def setup(self) -> None:
        # build_inputs reads the shapes and quantization from the installed options.
        with decode_options_scope(self.decode_options):
            super().setup()


def get_benchmark() -> BaseBenchmark:
    return RightSizedTritonDecodeBenchmark()
