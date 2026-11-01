"""Experiment 2: Triton's divisions, the reciprocal literal, and PyTorch's own scalar division, against the IEEE
quotient, over every float32 pattern.

The comparison reference is torch.div(x, six), with six a one-element float32 tensor on the GPU.
The tl.div_rn row supplies a separate round-to-nearest GPU path. Agreement on covered inputs corroborates
these implementations; it does not independently prove every IEEE result. Optional GPU tests also compare
the tensor-divisor reference with NumPy on sampled bit patterns.

Rows, each tallied against that reference in chunks that together cover all 2^32 bit patterns, with
torch.isclose at torch.allclose's defaults over the finite inputs:

  x / 6.0                                 Triton's default division (div.full.f32 on NVIDIA)
  tl.fdiv(x, six, ieee_rounding=True)     what the post names as the pinned divide
  tl.div_rn(x, six)                       Triton's precise divide
  x * 0.16666666666666666                 the literal the KernelBench candidate wrote (narrowed to fl(1/6))
  libdevice fast_dividef(x, six)          the approximate divide (div.approx)
  torch.div(x, 6.0)                       PyTorch eager, a Python scalar divisor

The emitted PTX of each Triton kernel is read for its divide instruction. PTX describes an intermediate
program; it does not certify the final machine instructions or their execution. The residue prediction applies to finite |x| >= 2^-123, not every bit pattern.
The experiment checks emitted PTX and numerical agreement separately; a source-level ieee_rounding
argument is not proof of compiler behavior on every Triton version.
"""

from __future__ import annotations

import re

from . import constants as C
from .ulp import Tally, patterns, tally

CHUNK = 1 << 27                    # 512 MiB of input per chunk, 32 chunks for the whole space
BLOCK = 1024
MODES = {0: "x / 6.0", 1: "tl.fdiv(x, six, ieee_rounding=True)", 2: "tl.div_rn(x, six)",
         3: "x * 0.16666666666666666", 4: "libdevice fast_dividef(x, six)"}
TORCH_SCALAR = "torch.div(x, 6.0), Python scalar divisor"
PTX_OPS = re.compile(r"\b(div\.(?:rn|full|approx)(?:\.ftz)?\.f32|mul(?:\.ftz)?\.f32|rcp\.approx(?:\.ftz)?\.f32)\b")


def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda.libdevice import fast_dividef

    @triton.jit
    def div6_kernel(x_ptr, y_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        six = tl.full([BLOCK], 6.0, tl.float32)
        if MODE == 0:
            y = x / 6.0
        elif MODE == 1:
            y = tl.fdiv(x, six, ieee_rounding=True)
        elif MODE == 2:
            y = tl.div_rn(x, six)
        elif MODE == 3:
            y = x * 0.16666666666666666
        else:
            y = fast_dividef(x, six)
        tl.store(y_ptr + offs, y, mask=m)

    return div6_kernel


_KERNEL = None


def kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _kernel()
    return _KERNEL


def div6(x, mode: int, out=None):
    """Run one Triton variant; returns (out, compiled kernel) so the caller can read the PTX."""
    import torch
    out = torch.empty_like(x) if out is None else out
    n = x.numel()
    grid = ((n + BLOCK - 1) // BLOCK,)
    compiled = kernel()[grid](x, out, n, MODE=mode, BLOCK=BLOCK, num_warps=4)
    return out, compiled


def ieee_div6(x):
    """The reference: PyTorch's tensor-by-tensor division (an IEEE divide in the eager kernel)."""
    import torch
    six = torch.full((1,), 6.0, dtype=torch.float32, device=x.device)
    return torch.div(x, six)


def ptx_ops(compiled) -> str:
    ops = PTX_OPS.findall(compiled.asm["ptx"])
    return ", ".join(dict.fromkeys(ops)) or "(none found)"


def run(device, quick: bool = False, progress=None) -> dict:
    """Tally every row against the IEEE reference over the whole space (or two chunks under --quick)."""
    import torch
    names = list(MODES.values()) + [TORCH_SCALAR]
    totals = {n: Tally() for n in names}
    notes = {TORCH_SCALAR: "eager CUDA kernel"}
    # Under --quick: one chunk of ordinary positive magnitudes and the chunk holding the negative specials.
    ranges = [(0x3F800000, 0x3F800000 + CHUNK), (C.PATTERNS - CHUNK, C.PATTERNS)] if quick \
        else [(lo, lo + CHUNK) for lo in range(0, C.PATTERNS, CHUNK)]
    out = torch.empty(CHUNK, dtype=torch.float32, device=device)
    # Pairs whose outputs are compared bit for bit with each other (NaN inputs aside), to tell "the same counts"
    # from "the same function": the default `/` and the literal, PyTorch's scalar path and the literal.
    pairs = {f"{MODES[0]} against {MODES[3]}": 0, f"{TORCH_SCALAR} against {MODES[3]}": 0, f"{MODES[1]} against {MODES[0]}": 0}
    completed_ranges = []

    def snapshot(complete=False):
        for n in names:
            totals[n].note = notes.get(n, "not compiled yet")
        return {"reference": "torch.div(x, six) with six a float32 tensor on the GPU",
                "chunks": len(completed_ranges), "planned_chunks": len(ranges), "chunk": CHUNK,
                "completed_ranges": list(completed_ranges), "complete": complete,
                "coverage": ("all 2^32 patterns" if not quick else "2 of 32 chunks (quick)") if complete
                            else f"{len(completed_ranges)} of {len(ranges)} requested chunks complete",
                "modes": {n: totals[n].as_dict() for n in names}, "pairs_differ": dict(pairs)}

    for lo, hi in ranges:
        x = patterns(lo, hi, device)
        q = ieee_div6(x)
        keep = {}
        for m, name in MODES.items():
            p, compiled = div6(x, m, out)
            notes.setdefault(name, ptx_ops(compiled))
            totals[name].add(tally(p, q, x))
            if m in (0, 1, 3):
                keep[m] = p.clone()
        scalar = torch.div(x, 6.0)
        totals[TORCH_SCALAR].add(tally(scalar, q, x))
        notnan = ~torch.isnan(x)
        bits = lambda t: t.view(torch.int32)
        pairs[f"{MODES[0]} against {MODES[3]}"] += int(((bits(keep[0]) != bits(keep[3])) & notnan).sum())
        pairs[f"{TORCH_SCALAR} against {MODES[3]}"] += int(((bits(scalar) != bits(keep[3])) & notnan).sum())
        pairs[f"{MODES[1]} against {MODES[0]}"] += int(((bits(keep[1]) != bits(keep[0])) & notnan).sum())
        del x, q, keep, scalar
        completed_ranges.append([lo, hi])
        if progress is not None:
            progress(snapshot())
    return snapshot(complete=True)
