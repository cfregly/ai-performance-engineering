"""Experiment 3a: PyTorch's hardsigmoid CUDA kernel against the formula its documentation gives.

The docs define hardsigmoid as x/6 + 1/2 clamped to [0, 1] (torch-hardsig-docs). The CUDA kernel computes
min(max(x + 3, 0), 6) * one_sixth with one_sixth = 1.0f / 6.0f (torch-hardsig-const). Over all 2^32 float32
patterns, in chunks, this compares F.hardsigmoid(x) on the GPU with four torch expressions:

  docs, as typed      clamp(x / 6 + 0.5, 0, 1)        the documented formula typed into torch (x / 6 with a Python scalar)
  docs, IEEE divide   clamp(x / six + 0.5, 0, 1)      the same with a tensor divisor, so the divide is an IEEE divide
  source              clamp(x + 3, 0, 6) * fl(1/6)    the kernel's own expression, retyped: should match bit for bit
  source, IEEE divide clamp(x + 3, 0, 6) / six        the kernel's expression with an IEEE divide for the multiply

The last pair isolates the reciprocal multiply inside the shipped kernel: on the inputs where the clamp is
inactive (strictly between -3 and 3) the residue argument applies to the clamped operand y = x + 3 where y is in the census.
It does not predict the share over input x: rounding x + 3 can map many x values to the same y. The docs pairs carry two more roundings (x/6 then + 0.5), so the
post makes no prediction for them beyond "two functions"; the count says how far apart they are.
"""

from __future__ import annotations

from . import constants as C
from .ulp import Tally, patterns, tally

CHUNK = 1 << 27
PAIRS = ("docs, as typed: clamp(x / 6 + 0.5, 0, 1)", "docs, IEEE divide: clamp(x / six + 0.5, 0, 1)",
         "source: clamp(x + 3, 0, 6) * fl(1/6)", "source, IEEE divide: clamp(x + 3, 0, 6) / six")


def references(x):
    import torch
    one_sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)   # fl(1/6), as the kernel stores it
    six = torch.full((1,), 6.0, dtype=torch.float32, device=x.device)           # a tensor divisor: the IEEE path
    return (torch.clamp(x / 6 + 0.5, 0, 1),
            torch.clamp(x / six + 0.5, 0, 1),
            torch.clamp(x + 3, 0, 6) * one_sixth,
            torch.clamp(x + 3, 0, 6) / six)


def run(device, quick: bool = False, progress=None) -> dict:
    import torch
    import torch.nn.functional as F
    inside = {p: Tally() for p in PAIRS}      # inputs strictly between -3 and 3, where the clamp is inactive
    moved = {p: Tally() for p in PAIRS}       # the inside inputs large enough that x + 3 is not exactly 3 (most patterns are tiny)
    outside = {p: Tally() for p in PAIRS}     # every other finite input: both sides should be 0 or 1
    ranges = [(0x3F800000, 0x3F800000 + CHUNK), (0xBF800000, 0xBF800000 + CHUNK)] if quick \
        else [(lo, lo + CHUNK) for lo in range(0, C.PATTERNS, CHUNK)]
    completed_ranges = []

    def snapshot(complete=False):
        return {"reference": "p=torch.nn.functional.hardsigmoid(x); q=each row expression; direction compares p against q",
                "chunks": len(completed_ranges), "planned_chunks": len(ranges),
                "completed_ranges": list(completed_ranges), "complete": complete,
                "coverage": ("all 2^32 patterns" if not quick else "2 of 32 chunks (quick)") if complete
                            else f"{len(completed_ranges)} of {len(ranges)} requested chunks complete",
                "inside": {p: inside[p].as_dict() for p in PAIRS},
                "moved": {p: moved[p].as_dict() for p in PAIRS},
                "outside": {p: outside[p].as_dict() for p in PAIRS}}

    for lo, hi in ranges:
        x = patterns(lo, hi, device)
        hs = F.hardsigmoid(x)
        in_mask = (x > -3) & (x < 3)
        moved_mask = in_mask & ((x + 3) != 3)
        out_mask = ~in_mask & torch.isfinite(x)
        for name, ref in zip(PAIRS, references(x)):
            inside[name].add(tally(hs, ref, x, in_mask))
            moved[name].add(tally(hs, ref, x, moved_mask))
            outside[name].add(tally(hs, ref, x, out_mask))
        del x, hs
        completed_ranges.append([lo, hi])
        if progress is not None:
            progress(snapshot())
    return snapshot(complete=True)
