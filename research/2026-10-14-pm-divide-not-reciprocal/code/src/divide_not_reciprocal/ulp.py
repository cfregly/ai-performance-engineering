"""Bit-pattern inputs and ulp tallies on the GPU, shared by the Triton and hardsigmoid experiments.

A chunk of consecutive bit patterns becomes a float32 tensor; two results are compared as ordered integers
(neighbouring floats of one sign differ by exactly 1, +0 and -0 both map to 0), so a distance of 1 is one
ulp (unit in the last place). The tally also runs torch.isclose at torch.allclose's defaults over the finite
inputs and keeps the largest gap in units of that tolerance, which says how much wider the tolerance is than
anything the variant did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as C


def patterns(lo: int, hi: int, device) -> "torch.Tensor":
    """float32 values whose bit patterns are lo, lo+1, ..., hi-1 (hi up to 2^32)."""
    if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= C.PATTERNS:
        raise ValueError("patterns requires 0 <= lo <= hi <= 2^32")
    import torch
    u = torch.arange(lo, hi, dtype=torch.int64, device=device)
    u = torch.where(u >= (1 << 31), u - (1 << 32), u).to(torch.int32)
    return u.view(torch.float32)


def ordered(f: "torch.Tensor") -> "torch.Tensor":
    import torch
    i = f.view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -(i & 0x7FFFFFFF), i)


def ordered_py(bits: int) -> int:
    """The same order on one bit pattern, for tests."""
    m = bits & 0x7FFFFFFF
    return -m if bits & 0x80000000 else m


@dataclass
class Tally:
    inputs: int = 0
    same: int = 0
    away1: int = 0
    toward1: int = 0
    far: int = 0
    nan_both: int = 0
    nan_one: int = 0
    max_ulp: int = 0
    isclose_fail: int = 0          # finite inputs where torch.isclose at the defaults says False
    max_gap_tol: float | None = 0.0       # max over finite inputs of |p - q| / (atol + rtol |q|); under 1 means allclose passes
    chunks: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def differ(self) -> int:
        return self.away1 + self.toward1 + self.far

    def add(self, other: "Tally") -> "Tally":
        for k in ("inputs", "same", "away1", "toward1", "far", "nan_both", "nan_one", "isclose_fail", "chunks"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.max_ulp = max(self.max_ulp, other.max_ulp)
        self.max_gap_tol = (None if self.max_gap_tol is None or other.max_gap_tol is None
                            else max(self.max_gap_tol, other.max_gap_tol))
        return self

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("inputs", "same", "away1", "toward1", "far", "nan_both", "nan_one",
                                            "max_ulp", "isclose_fail", "max_gap_tol", "chunks", "note")}
        d["differ"] = self.differ
        d.update(self.extra)
        return d


def tally(p: "torch.Tensor", q: "torch.Tensor", x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> Tally:
    """Compare p against the reference q over the inputs x (optionally only where mask is True)."""
    import torch
    if p.shape != q.shape or p.shape != x.shape:
        raise ValueError("p, q and x must have identical shapes; broadcasting is not a tally")
    if p.dtype != torch.float32 or q.dtype != torch.float32 or x.dtype != torch.float32:
        raise ValueError("tally requires float32 p, q and x")
    if p.device != q.device or p.device != x.device:
        raise ValueError("p, q and x must share a device")
    if mask is not None and (mask.shape != x.shape or mask.dtype != torch.bool or mask.device != x.device):
        raise ValueError("mask must be boolean with the input shape and device")
    if mask is not None:
        p, q, x = p[mask], q[mask], x[mask]
    t = Tally(inputs=int(x.numel()), chunks=1)
    qn, pn = torch.isnan(q), torch.isnan(p)
    anynan = qn | pn
    t.nan_both = int((qn & pn).sum())
    t.nan_one = int((anynan & ~(qn & pn)).sum())
    d = ordered(p) - ordered(q)
    d = torch.where(anynan, torch.zeros_like(d), d)
    ad = d.abs()
    # Unsigned magnitudes classify signed zero and subnormals without a floating comparison.
    pm = p.view(torch.int32).to(torch.int64) & 0x7FFFFFFF
    qm = q.view(torch.int32).to(torch.int64) & 0x7FFFFFFF
    away = pm > qm
    t.same = int(((d == 0) & ~anynan).sum())
    t.away1 = int(((ad == 1) & away).sum())
    t.toward1 = int(((ad == 1) & ~away).sum())
    t.far = int((ad >= 2).sum())
    t.max_ulp = int(ad.max()) if ad.numel() else 0
    finite = torch.isfinite(x)
    close = torch.isclose(p, q, rtol=C.RTOL, atol=C.ATOL)
    t.isclose_fail = int((~close & finite).sum())
    gap = (p.double() - q.double()).abs() / (C.ATOL + C.RTOL * q.double().abs())
    gap = torch.where(finite, gap, torch.zeros_like(gap))
    # Do not turn an undefined finite-input gap (for example NaN output) into a small maximum.
    t.max_gap_tol = None if bool((finite & ~torch.isfinite(gap)).any()) else (float(gap.max()) if gap.numel() else 0.0)
    return t
