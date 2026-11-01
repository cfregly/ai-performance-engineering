"""Experiment 1: the exhaustive count. Build count.cu twice (nvcc defaults, then --use_fast_math), walk all
2^32 float32 bit patterns on the GPU, and tally each division variant against the IEEE quotient per input binade.

What a row means: over finite inputs with |x| >= 2^-123 (exponent fields 4..254, both signs), how
many patterns the variant moved, by one ulp away from zero, one ulp toward zero, or two or more, and the
largest distance. The post predicts, for x * fl(1/6): one in three, all one ulp away from zero, in every one
of those binades. The bottom three normal-input binades (some quotients are normal), exponent-field-0 inputs (signed zeros and subnormals), and
exponent-field-255 inputs (infinities and NaNs) are tallied separately: the derivation does not cover them.

Also reads the PTX nvcc emits for each variant function (which divide instruction it is) and the SASS ptxas
made of it (whether the hardware path is a multiply, a reciprocal plus refinement, or a call to a slow path).
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C

CU = Path(__file__).with_name("count.cu")
VARIANTS = {0: "x * (1.0f / 6.0f)", 1: "x / 6.0f", 2: "__fdividef(x, 6.0f)", 3: "__fdiv_rn(x, 6.0f)"}
BUILDS = {"default": [], "fast_math": ["--use_fast_math"]}
NBIN, NCLASS = 512, 6
SAME, AWAY1, TOWARD1, FAR, NAN_BOTH, NAN_ONE = range(NCLASS)
PTX_OPS = re.compile(r"\b(div\.(?:rn|full|approx)(?:\.ftz)?\.f32|mul(?:\.ftz)?\.f32|rcp\.approx(?:\.ftz)?\.f32|fma\.rn(?:\.ftz)?\.f32)\b")
SASS_OPS = re.compile(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_.]*)", re.M)


@dataclass
class Build:
    name: str
    flags: list[str]
    lib: Path
    ptx: str
    sass: str


@dataclass
class Tally:
    build: str
    variant: int
    expression: str
    ptx_ops: str
    sass_ops: str
    hist: list[list[int]]          # [bin][class]; bin = sign << 8 | exponent
    max_ulp: list[int]             # [bin]: the largest distance seen among that bin's inputs
    ms: float
    per_binade_flips: dict = field(default_factory=dict)

    def bins(self, exps, signs=(0, 1)):
        return [(s << 8) | e for s in signs for e in exps]

    def sums(self, exps):
        out = [0] * NCLASS
        for b in self.bins(exps):
            for k in range(NCLASS):
                out[k] += self.hist[b][k]
        return out

    def region_max(self, exps):
        return max(self.max_ulp[b] for b in self.bins(exps))


def nvcc() -> str:
    exe = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(exe).is_file():
        raise RuntimeError("nvcc not found on PATH or at /usr/local/cuda/bin/nvcc")
    return exe


def cuobjdump() -> str:
    exe = shutil.which("cuobjdump") or str(Path(nvcc()).with_name("cuobjdump"))
    if not Path(exe).is_file():
        raise RuntimeError("cuobjdump not found on PATH or beside nvcc")
    return exe


def host_compiler() -> str:
    import os
    requested = os.environ.get("CXX")
    exe = shutil.which(requested) if requested else (shutil.which("g++") or shutil.which("clang++") or shutil.which("c++"))
    if not exe:
        raise RuntimeError("A host C++ compiler is required; CXX may name one executable")
    return exe


def checked_command(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed ({exc.returncode}): {command!r}\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}") from exc


def arch() -> str:
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def build(workdir: Path, name: str, flags: list[str]) -> Build:
    """Compile count.cu into a shared library, and emit its PTX and SASS for the instruction read."""
    lib = workdir / f"libcount_{name}.so"
    ptx = workdir / f"count_{name}.ptx"
    base = [nvcc(), "-O3", "-ccbin", host_compiler(), f"-arch={arch()}", *flags, str(CU)]
    checked_command([*base, "-shared", "-Xcompiler", "-fPIC", "-o", str(lib)])
    checked_command([*base, "-ptx", "-o", str(ptx)])
    sass = checked_command([cuobjdump(), "-sass", str(lib)])
    return Build(name, flags, lib, ptx.read_text(), sass)


def _body(text: str, header_re: str) -> str:
    """The text of one PTX .func or SASS function, from its header to the closing brace or the next header."""
    m = re.search(header_re, text)
    if not m:
        return ""
    rest = text[m.end():]
    end = re.search(r"\n\}|\n\s*\.\.\.\.\.\.\.\.\.\.|\n\s+Function :", rest)
    return rest[: end.start()] if end else rest


def variant_ops(b: Build, v: int) -> tuple[str, str]:
    """The divide-related PTX instructions in variant<V>'s body, and the SASS opcode mix of probe_kernel<V>
    (load, variant, store). Whether the call is retained or inlined depends on the toolchain.
    The opcode mix alone is not proof of identical machine code or outputs."""
    sym = f"_Z7variantILi{v}EEff"
    ptx_body = _body(b.ptx, rf"\.func\s+\(\.param \.b32 func_retval0\)\s+{re.escape(sym)}\(")
    ops = PTX_OPS.findall(ptx_body)
    ptx_s = ", ".join(f"{o} x{ops.count(o)}" if ops.count(o) > 1 else o for o in dict.fromkeys(ops)) or "(none found)"
    probe = f"_Z12probe_kernelILi{v}EEvPKfPf"
    sass_body = _body(b.sass, rf"Function : {re.escape(probe)}\n")
    # This filtered opcode summary omits address, memory and control operations.
    # Inspect the saved full SASS for calls and any instructions outside the probe body.
    skip = ("EXIT", "NOP", "MOV", "S2R", "S2UR", "IADD3", "LEA", "SHF.L.U32", "USHF", "UIADD3", "UMOV", "CALL.REL.NOINC", "RET.REL.NODEC", "HFMA2.MMA", "HFMA2", "BRA")
    sops = [o for o in SASS_OPS.findall(sass_body) if o not in skip and not o.startswith(("LDG", "STG", "LDC", "IMAD", "ULDC"))]
    counts = {}
    for o in sops:
        counts[o] = counts.get(o, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])
    sass_s = (f"{len(sops)} instructions: " + ", ".join(f"{o} x{n}" if n > 1 else o for o, n in top[:5])
              + (", ..." if len(top) > 5 else "")) if sops else "(none found)"
    return ptx_s, sass_s


def run(b: Build, v: int, lo: int = 0, hi: int = C.PATTERNS) -> Tally:
    if v not in VARIANTS or type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= C.PATTERNS:
        raise ValueError("run requires a known variant and 0 <= lo < hi <= 2^32")
    lib = ctypes.CDLL(str(b.lib))
    fn = lib.run_count
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong,
                   ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_float)]
    hist = (ctypes.c_uint * (NBIN * NCLASS))()
    mx = (ctypes.c_uint * NBIN)()
    ms = ctypes.c_float(0)
    rc = fn(v, lo, hi, hist, mx, ctypes.byref(ms))
    if rc != 0:
        raise RuntimeError(f"run_count({v}) failed with CUDA error {rc}")
    h = [[int(hist[i * NCLASS + k]) for k in range(NCLASS)] for i in range(NBIN)]
    if sum(map(sum, h)) != hi - lo:
        raise RuntimeError(f"Histogram coverage mismatch: counted {sum(map(sum, h))}, expected {hi - lo}")
    ptx_s, sass_s = variant_ops(b, v)
    t = Tally(b.name, v, VARIANTS[v], ptx_s, sass_s, h, [int(mx[i]) for i in range(NBIN)], float(ms.value))
    for s in (0, 1):
        for e in range(C.CENSUS_EXP_MIN, 255):
            r = h[(s << 8) | e]
            t.per_binade_flips[(s, e)] = r[AWAY1] + r[TOWARD1] + r[FAR]
    return t


def run_all(builds: dict[str, Build]) -> list[Tally]:
    return [run(b, v) for b in builds.values() for v in VARIANTS]


def make_builds(workdir: Path | None = None) -> dict[str, Build]:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="divide-not-reciprocal-"))
    return {name: build(workdir, name, flags) for name, flags in BUILDS.items()}


def summarize(t: Tally) -> dict:
    """The numbers the report prints for one tally."""
    NQ, BOTTOM, SUBN, SPECIAL = range(C.CENSUS_EXP_MIN, 255), range(1, C.CENSUS_EXP_MIN), [0], [255]
    normal_q, bottom, subn, special = (t.sums(r) for r in (NQ, BOTTOM, SUBN, SPECIAL))
    n = sum(normal_q)
    flips = normal_q[AWAY1] + normal_q[TOWARD1] + normal_q[FAR]
    pb = list(t.per_binade_flips.values())
    return {
        "build": t.build, "variant": t.variant, "expression": t.expression, "ptx": t.ptx_ops, "sass": t.sass_ops,
        "census": {"definition": "finite |x| >= 2^-123, exponent fields 4..254","inputs": n, "differ": flips, "share": flips / n if n else 0.0,
                            "away1": normal_q[AWAY1], "toward1": normal_q[TOWARD1], "far": normal_q[FAR],
                            "max_ulp": t.region_max(NQ), "per_binade_min": min(pb), "per_binade_max": max(pb)},
        "bottom_binades": {"inputs": sum(bottom), "differ": bottom[AWAY1] + bottom[TOWARD1] + bottom[FAR],
                           "away1": bottom[AWAY1], "toward1": bottom[TOWARD1], "far": bottom[FAR], "max_ulp": t.region_max(BOTTOM)},
        "subnormal_inputs": {"inputs": sum(subn), "differ": subn[AWAY1] + subn[TOWARD1] + subn[FAR],
                             "away1": subn[AWAY1], "toward1": subn[TOWARD1], "far": subn[FAR], "max_ulp": t.region_max(SUBN)},
        "specials": {"inputs": sum(special), "nan_both": special[NAN_BOTH], "nan_one": special[NAN_ONE],
                     "differ": special[AWAY1] + special[TOWARD1] + special[FAR]},
        "ms": t.ms,
        "total": sum(sum(r) for r in t.hist),
        "all_differ": sum(r[AWAY1] + r[TOWARD1] + r[FAR] for r in t.hist),
        "normal_inputs_differ": flips + bottom[AWAY1] + bottom[TOWARD1] + bottom[FAR],
    }
