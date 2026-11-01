"""Atomic per-run JSON checkpoints and readable Markdown summaries."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import constants as C


def fmt(n: float, digits: int = 3) -> str:
    return f"{n:,.{digits}f}"


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.3f}%" if b else "n/a"


def share(a: int, b: int) -> str:
    return f"1 in {b / a:.3f}" if a else "0"


def smi(query: str, index: str | None) -> str:
    from .exhaustive import checked_command
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader"]
    if index is not None:
        cmd += ["-i", index]
    out = checked_command(cmd).strip()
    if query.startswith("compute-apps"):
        out = ", ".join(l for l in out.splitlines() if not l.startswith(f"{os.getpid()},"))
    return out


def conditions(gpu_index: str | None, clocks: dict, procs: dict, extra: str, preflight: dict) -> str:
    driver = smi("gpu=driver_version", gpu_index)
    version = preflight["tools"].get("nvcc_version", "")
    match = re.search(r"release\s+([^,\s]+)", version)
    nvcc_version = match.group(1) if match else ("not requested" if not version else version)
    return (f"Conditions: {preflight['gpu']} (visible device {gpu_index or 'default'}), driver {driver}, "
            f"CUDA {preflight['torch_cuda']} (nvcc {nvcc_version}), torch {preflight['torch']}, triton {preflight['triton']}; "
            f"SM clock {clocks.get('start', 'unavailable')} at start, {clocks.get('end', 'pending')} at end; "
            f"{extra}; other GPU processes at start: {procs.get('start') or 'none'}, "
            f"at end: {procs.get('end', 'pending') or 'none'}.")


def render(name: str, R: dict) -> str:
    L = [f"# {name}: division-by-six lab, {R['date']}", "", f"State: **{R['state']}**.", "",
         "The exact residue prediction covers finite |x| >= 2^-123 (exponent fields 4..254, both signs). "
         "It is not a label for every input with a normal quotient. Some field-3 inputs also have normal quotients. "
         "The predicted count is 2,796,202 of 8,388,608 significands per binade (approximately one third), "
         "each one representable step away from zero. Other regions are counted separately.", ""]
    if R.get("stages"):
        L += ["Stages: " + "; ".join(f"{k}: {v}" for k, v in R["stages"].items()), ""]
    for key, table in (("exhaustive", table1), ("triton", table2), ("hardsigmoid", table3), ("timing", table4)):
        if R.get(key):
            L.extend(table(R[key]))
    if R.get("ncu"):
        L.extend(table5(R["ncu"]))
    L += [R.get("conditions", "Conditions are pending; see preflight and errors in the JSON."), ""]
    if R.get("errors"):
        L += ["## Failures", ""]
        for error in R["errors"]:
            L += [f"- {error['stage']}: {error['type']}: {error['message']}"]
        L += ["", "The JSON preserves completed results, partial timing samples when available, and full error details.", ""]
    return "\n".join(L)


def table1(ex: list) -> list[str]:
    L = []
    L.append("## Table 1: every float32 bit pattern, CUDA (nvcc), each variant against div.rn.f32")
    L.append("")
    L.append(f"Whole-binade census, finite |x| >= 2^-123: exponent field {C.CENSUS_EXP_MIN}..254, both signs, {2 * C.CENSUS_BINADES_PER_SIGN} binades, {fmt(ex[0]['census']['inputs'], 0)} inputs. Prediction for the reciprocal multiply: {fmt(C.PREDICTED_FLIPS_PER_BINADE, 0)} per binade (the M mod 3 count; approximately one third), {fmt(C.PREDICTED_FLIPS_CENSUS, 0)} in all, every one of them 1 ulp away from zero.")
    L.append("")
    L.append("| build | expression | PTX in the variant | SASS of the variant | differ | share | 1 ulp away | 1 ulp toward | 2+ ulp | max ulp | per binade (min..max) | kernel ms |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ex:
        nq = s["census"]
        L.append(f"| {s['build']} | `{s['expression']}` | {s['ptx']} | {s['sass']} | {fmt(nq['differ'], 0)} | {share(nq['differ'], nq['inputs'])} ({pct(nq['differ'], nq['inputs'])}) | {fmt(nq['away1'], 0)} | {fmt(nq['toward1'], 0)} | {fmt(nq['far'], 0)} | {nq['max_ulp']} | {fmt(nq['per_binade_min'], 0)}..{fmt(nq['per_binade_max'], 0)} | {s['ms']:.1f} |")
    L.append("")
    L.append("The rest of the input space, same variants; the derivation does not cover these. --use_fast_math requests -ftz=true. Check the retained generated code and regional counts before attributing observed differences to flushing subnormals.")
    L.append("")
    L.append("| build | expression | 3 bottom normal binades: inputs / differ (away, toward, 2+) / max ulp | zeros and subnormal inputs (field 0): inputs / differ (away, toward, 2+) / max ulp | infinities and NaNs (field 255): inputs / NaN both / NaN one side / differ | all normal inputs: differ of 4,261,412,864 | all patterns: differ of total |")
    L.append("|---|---|---|---|---|---|---|")
    for s in ex:
        b, sn, sp = s["bottom_binades"], s["subnormal_inputs"], s["specials"]
        L.append(f"| {s['build']} | `{s['expression']}` | {fmt(b['inputs'], 0)} / {fmt(b['differ'], 0)} ({fmt(b['away1'], 0)}, {fmt(b['toward1'], 0)}, {fmt(b['far'], 0)}) / {b['max_ulp']} | {fmt(sn['inputs'], 0)} / {fmt(sn['differ'], 0)} ({fmt(sn['away1'], 0)}, {fmt(sn['toward1'], 0)}, {fmt(sn['far'], 0)}) / {sn['max_ulp']} | {fmt(sp['inputs'], 0)} / {fmt(sp['nan_both'], 0)} / {fmt(sp['nan_one'], 0)} / {fmt(sp['differ'], 0)} | {fmt(s['normal_inputs_differ'], 0)} ({pct(s['normal_inputs_differ'], C.NORMAL_INPUTS)}) | {fmt(s['all_differ'], 0)} of {fmt(s['total'], 0)} |")
    L.append("")
    return L


def table2(tr: dict) -> list[str]:
    L = []
    L.append(f"## Table 2: Triton kernels and PyTorch's scalar division against tensor division, {tr['coverage']}")
    L.append("")
    L.append(f"Reference: {tr['reference']}; the tl.div_rn row requests round-to-nearest division and provides a separate comparison path. Agreement corroborates these implementations; it does not independently prove every IEEE result. Input and difference shares use all supplied patterns, with NaN counts separate. torch.isclose at torch.allclose's defaults (rtol {C.RTOL:g}, atol {C.ATOL:g}) over the finite inputs; \"gap / tolerance\" is the largest |p - q| / (atol + rtol |q|) seen, a descriptive gap ratio. Use the explicit finite-input isclose failure count for pass/fail. Prediction: the relative tolerance alone is at least {C.TOLERANCE_OVER_ULP:.1f}x a 1-ulp normal-range gap, so the reciprocal multiply's largest gap is about {1 / C.TOLERANCE_OVER_ULP:.4f} of it.")
    L.append("")
    L.append("| expression | PTX (Triton rows) | inputs | differ | share | 1 ulp away | 1 ulp toward | 2+ ulp | max ulp | isclose fails (finite) | gap / tolerance | NaN both / one side |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, t in tr["modes"].items():
        L.append(f"| `{k}` | {t['note']} | {fmt(t['inputs'], 0)} | {fmt(t['differ'], 0)} | {share(t['differ'], t['inputs'])} ({pct(t['differ'], t['inputs'])}) | {fmt(t['away1'], 0)} | {fmt(t['toward1'], 0)} | {fmt(t['far'], 0)} | {t['max_ulp']} | {fmt(t['isclose_fail'], 0)} | {gap_text(t['max_gap_tol'])} | {fmt(t['nan_both'], 0)} / {fmt(t['nan_one'], 0)} |")
    L.append("")
    L.append("Outputs compared with each other bit for bit (NaN inputs aside), to tell the same counts from the same function:")
    L.append("")
    for k, v in tr.get("pairs_differ", {}).items():
        L.append(f"- `{k}`: {fmt(v, 0)} inputs differ")
    L.append("")
    return L


def table3(hs: dict) -> list[str]:
    L = []
    L.append(f"## Table 3: torch.nn.functional.hardsigmoid against four expressions, {hs['coverage']}")
    L.append("")
    L.append("Comparison roles: p is torch.nn.functional.hardsigmoid(x), q is the row expression. Away/toward describe p relative to q in magnitude, and isclose uses q in its relative-tolerance denominator. Inside: inputs strictly between -3 and 3, where the clamp is inactive. When x + 3 rounds to 3, the two source-form expressions round to 0.5; the docs-form expressions divide before adding 0.5 and can still differ. The moved group keeps only inside inputs where x + 3 changed from 3. The residue rule concerns the source-form division comparison on y = x + 3 within the census; it does not predict the share over x or the docs-form rows. Outside: every other finite input, where both sides should be exactly 0 or 1. Inside plus outside counts finite inputs only; the moved group is a subset of inside. NaN and infinity input patterns are excluded from these groups. `six` is a float32 tensor on the GPU (the IEEE divide path); `/ 6` with a Python scalar is PyTorch's scalar path.")
    L.append("")
    L.append("| expression | inside: inputs | differ | share | max ulp | isclose fails | x + 3 moved: inputs | differ | share | 1 ulp away | 1 ulp toward | 2+ ulp | max ulp | outside: inputs / differ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in hs["inside"]:
        i, m, o = hs["inside"][p], hs["moved"][p], hs["outside"][p]
        L.append(f"| `{p}` | {fmt(i['inputs'], 0)} | {fmt(i['differ'], 0)} | {share(i['differ'], i['inputs'])} ({pct(i['differ'], i['inputs'])}) | {i['max_ulp']} | {fmt(i['isclose_fail'], 0)} | {fmt(m['inputs'], 0)} | {fmt(m['differ'], 0)} | {share(m['differ'], m['inputs'])} ({pct(m['differ'], m['inputs'])}) | {fmt(m['away1'], 0)} | {fmt(m['toward1'], 0)} | {fmt(m['far'], 0)} | {m['max_ulp']} | {fmt(o['inputs'], 0)} / {fmt(o['differ'], 0)} |")
    L.append("")
    return L


def table4(tm: dict) -> list[str]:
    L = [f"## Table 4: timing, {tm['n']:,} float32 elements, {tm['bytes_moved']:,} nominal bytes per call", "",
         f"CUDA events; {tm['warmup']} warm-up and {tm['iters']} timed rotations. "
         "Each Triton case uses its lowest-median launch shape from the saved sweep. "
         "The eager scalar/tensor division and tensor multiplication rows omit the clamp. "
         "These are component timings, not an end-to-end serving benchmark.", "",
         "| case | launch | median ms | p10 ms | p90 ms | nominal TB/s |",
         "|---|---|---|---|---|---|"]
    for name, r in tm["cases"].items():
        L.append(f"| {name} | {r['launch']} | {r['median_ms']:.4f} | {r['p10_ms']:.4f} | {r['p90_ms']:.4f} | {r['tb_per_s']:.2f} |")
    return L + [""]


def table5(ncu: dict) -> list[str]:
    L = [f"## Table 5: Nsight Compute, {ncu['n']:,} elements per reported large launch", "",
         "One large launch is reported per case in its own process; the paired tiny launch is retained for identity. --clock-control none. "
         "Metric sm__inst_executed.sum counts warp instructions. Normalization below is "
         "warp instructions per 32 elements (count x 32 / elements), not scalar instruction count. "
         "Raw CSV, stderr and native reports are stored in this run's ncu directory with hashes in JSON.", "",
         "| case | kernel | grid | block | warp instructions | warp instructions / 32 elements | DRAM bytes | GPU seconds |",
         "|---|---|---|---|---|---|---|---|"]
    for row in ncu["launches"]:
        L.append(f"| {row['case']} | `{row['kernel']}` | {tuple(row['grid'])} | {tuple(row['block'])} | {row['inst_executed']:,.0f} | {row['warp_instructions_per_32_elements']:.3f} | {row['dram_bytes']:,.0f} | {row['time_s']:.9g} |")
    if ncu["errors"]:
        L += ["", "Capture failures (run remains PARTIAL):", *[f"- {e['case']}: {e['message']}" for e in ncu["errors"]]]
    return L + [""]


def gap_text(value: float | None) -> str:
    return "undefined (nonfinite output)" if value is None else f"{value:.4f}"


def default_results_root() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "divide-not-reciprocal" / "runs"


def reserve_run(root: Path, name: str | None) -> tuple[Path, str]:
    name = name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("run name must start with a letter or digit and contain only letters, digits, _, . or -")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / name
    run_dir.mkdir(exist_ok=False)  # Atomic reservation; an existing run is never overwritten.
    return run_dir, name


def atomic_text(path: Path, text: str) -> None:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False) as f:
            tmp = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # On Linux this also persists the directory entry through a power loss.
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def write(results_dir: Path, name: str, R: dict) -> None:
    # JSON first: rendering or Markdown errors cannot discard completed counts.
    atomic_text(results_dir / f"{name}.json", json.dumps(R, indent=2, default=str, allow_nan=False) + "\n")
    atomic_text(results_dir / f"{name}.md", render(name, R))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
