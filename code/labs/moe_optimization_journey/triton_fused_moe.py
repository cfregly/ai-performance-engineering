#!/usr/bin/env python3
"""Retired, incomplete Triton MoE experiment (legacy import/CLI compatibility).

The removed kernel omitted intermediate/output tiles and never provided a
complete top-k MoE result. It is not an implementation or benchmark. For a
complete sorted PyTorch expert path, see ``level4_triton.GroupedMoEExperts``;
its legacy filename does not imply Triton execution. No backend is substituted
by the compatibility entry points below.
"""
from __future__ import annotations

import sys


RETIREMENT_REASON = (
    "Retired incomplete Triton MoE experiment: full intermediate and output "
    "tiles and top-k combine were not implemented. No kernel or benchmark "
    "result is produced. Use level4_triton.GroupedMoEExperts for the separately "
    "named sorted PyTorch expert path; it is not a fused Triton FFN."
)


class RetiredMoEKernelError(NotImplementedError):
    """A removed experiment must not be mistaken for a usable implementation."""


def triton_fused_moe(
    x, w_gate, w_up, w_down, sorted_weights, expert_offsets,
    E: int, H: int, I: int, max_tokens: int | None = None,
):
    """Reject legacy calls explicitly; never return partially computed output."""
    raise RetiredMoEKernelError(RETIREMENT_REASON)


def benchmark_triton_moe():
    """No timing or throughput may be published for the removed experiment."""
    raise RetiredMoEKernelError(RETIREMENT_REASON)


def main() -> int:
    try:
        benchmark_triton_moe()
    except RetiredMoEKernelError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
