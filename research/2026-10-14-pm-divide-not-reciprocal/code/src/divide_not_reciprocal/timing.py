"""Optional component timing and profiling, using the original fixed launch protocol.

Seven cases, a seven-shape Triton sweep, and a 2^28-element tensor. These are
not end-to-end serving measurements. No historical timing or profiler result
is asserted by this maintained implementation. Completed samples survive
handled failures; no disk writes are inserted into timed rotations.
"""

from __future__ import annotations

import csv
import io
import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

N = 1 << 28                        # 2^28 float32: 1 GiB in, 1 GiB out
CASES = ("triton clamp / 6.0", "triton clamp * 0.16666666666666666", "triton tl.div_rn(clamp, six)",
         "torch F.hardsigmoid (eager)", "torch x / 6.0 (eager, scalar divisor)", "torch x / six (eager, tensor divisor)",
         "torch x * sixth (eager, tensor multiplier: the control for the tensor-divisor row)")
N_TRITON = 3
WARMUP, ITERS = 20, 100
NCU_N = 1 << 26
# Launch shapes tried for each Triton case before the timed comparison (elements per program, warps per
# program): the IEEE divide is a long dependent chain of FFMAs, so it may want more warps in flight than a
# multiply does, and a comparison at one fixed shape would be a comparison of shapes. Each case is timed at
# its own best shape.
CONFIGS = ((1024, 4), (2048, 4), (2048, 8), (4096, 8), (8192, 8), (8192, 16), (16384, 16))
SWEEP_WARMUP, SWEEP_ITERS = 3, 10


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def hardsigmoid_kernel(x_ptr, y_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        t = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        if MODE == 0:
            y = t / 6.0
        elif MODE == 1:
            y = t * 0.16666666666666666
        else:
            six = tl.full([BLOCK], 6.0, tl.float32)
            y = tl.div_rn(t, six)
        tl.store(y_ptr + offs, y, mask=m)

    return hardsigmoid_kernel


_KERNEL = None


def kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _kernel()
    return _KERNEL


def triton_case(x, out, mode, block, warps):
    n = x.numel()
    grid = ((n + block - 1) // block,)
    return lambda: kernel()[grid](x, out, n, MODE=mode, BLOCK=block, num_warps=warps)


def make_cases(x, out, configs):
    """The seven callables: Triton modes 0..2 at the launch shapes in `configs`, eager hardsigmoid, and PyTorch's
    two eager divisions by 6 and one tensor multiplication. Compiler behavior must be checked
    in the generated code on the current software stack."""
    import torch
    import torch.nn.functional as F
    six = torch.full((1,), 6.0, dtype=torch.float32, device=x.device)
    sixth = torch.full((1,), 1.0 / 6.0, dtype=torch.float32, device=x.device)
    return ([triton_case(x, out, m, *configs[m]) for m in range(N_TRITON)]
            + [lambda: F.hardsigmoid(x), lambda: torch.div(x, 6.0), lambda: torch.div(x, six), lambda: torch.mul(x, sixth)])


def _time(cases, warmup, iters):
    """CUDA-event samples in the original rotation; retain samples on handled failure."""
    import torch
    samples = [[] for _ in cases]
    warmup_completed = 0
    try:
        for _ in range(warmup):
            for c in cases:
                c()
            warmup_completed += 1
        torch.cuda.synchronize()
        for _ in range(iters):
            for i, c in enumerate(cases):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                c()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end))
        return samples
    except BaseException as exc:
        exc.partial_results = {"phase": "timed_rotation", "samples_ms": samples,
                               "warmup_completed": warmup_completed, "warmup_requested": warmup,
                               "iterations_requested": iters}
        raise


def sweep(x, out) -> tuple[dict, dict]:
    """Best launch shape per Triton mode, and all completed grid medians."""
    grid = {}
    try:
        for block, warps in CONFIGS:
            cases = [triton_case(x, out, m, block, warps) for m in range(N_TRITON)]
            for m, ms in enumerate(_time(cases, SWEEP_WARMUP, SWEEP_ITERS)):
                grid[(m, block, warps)] = statistics.median(ms)
        best = {m: min(((b, w) for (mm, b, w) in grid if mm == m), key=lambda bw: grid[(m, *bw)]) for m in range(N_TRITON)}
        return best, {f"mode{m} block{b} warps{w}": round(v, 5) for (m, b, w), v in grid.items()}
    except BaseException as exc:
        exc.partial_results = {"phase": "launch_sweep", "completed_grid_ms": {str(k): v for k, v in grid.items()},
                               "current_shape": [block, warps], "current_samples": getattr(exc, "partial_results", None)}
        raise


def time_cases(device, warmup=WARMUP, iters=ITERS) -> dict:
    import torch
    best, grid, samples = None, None, None
    try:
        x = torch.empty(N, dtype=torch.float32, device=device).uniform_(-8, 8)
        out = torch.empty_like(x)
        best, grid = sweep(x, out)
        cases = make_cases(x, out, best)
        samples = _time(cases, warmup, iters)
        rows = {}
        for i, (name, ms) in enumerate(zip(CASES, samples)):
            srt = sorted(ms)
            med = statistics.median(srt)
            rows[name] = {"median_ms": med, "p10_ms": srt[int(0.1 * len(srt))], "p90_ms": srt[int(0.9 * len(srt)) - 1],
                          "tb_per_s": (2 * N * 4) / (med * 1e-3) / 1e12, "samples": ms,
                          "launch": f"{best[i][0]} elements x {best[i][1]} warps" if i < N_TRITON else "ATen elementwise kernel"}
        return {"n": N, "bytes_moved": 2 * N * 4, "warmup": warmup, "iters": iters, "cases": rows, "sweep": grid,
                "best": best, "configs_tried": [f"{b} x {w}" for b, w in CONFIGS]}
    except BaseException as exc:
        exc.partial_results = {"n": N, "best": best, "sweep": grid, "completed_samples_ms": samples,
                               "incomplete": getattr(exc, "partial_results", None)}
        raise


def _profile_configs(configs: dict | None) -> dict:
    """Require the actual sweep selections; never silently choose a profile shape."""
    if not isinstance(configs, dict):
        raise ValueError("profiling requires all three swept Triton configurations")
    normalized = {}
    for key, value in configs.items():
        if str(key) not in {"0", "1", "2"} or isinstance(key, bool):
            raise ValueError("invalid Triton profile mode")
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int for v in value):
            raise ValueError("invalid Triton profile shape")
        if tuple(value) not in CONFIGS:
            raise ValueError("profile shape was not in the fixed launch sweep")
        mode = int(key)
        if mode in normalized:
            raise ValueError("duplicate Triton profile mode")
        normalized[mode] = tuple(value)
    if set(normalized) != set(range(N_TRITON)):
        raise ValueError("profiling requires all three swept Triton configurations")
    return normalized


NCU_WARM_N = 16384
# These identify supported implementation families, not verified PTX lowering.
CASE_KERNEL_PATTERNS = (r"^hardsigmoid_kernel\b",) * N_TRITON + (
    r"\bat::native::.*hardsigmoid", r"\bat::native::.*MulFunctor",
    r"\bat::native::.*DivFunctor", r"\bat::native::.*MulFunctor")
METRICS = ["gpu__time_duration.sum", "sm__inst_executed.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"]
BASE_UNITS = ("nsecond", "inst", "byte", "byte")
LAUNCH_COLUMNS = ("ID", "Process ID", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC")
_NUMBER = re.compile(r"(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _profile_range(case: int) -> str:
    return f"DNR_case_{case}_n_{NCU_N}"


def ncu_probe(case: int):
    """Profile the existing tiny/large calls inside one named NVTX range.

    Both calls are retained as an identity pair. No launch skip/count can hide
    a warm-only capture or an extra kernel. This is separate from CUDA timing.
    """
    import json
    import torch
    from .report import atomic_text
    if type(case) is not int or not 0 <= case < len(CASES):
        raise ValueError("invalid profiler case")
    configs = _profile_configs(json.loads(os.environ["DNR_CONFIGS"]))
    path = Path(os.environ["DNR_PROBE_PATH"])
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("probe manifest needs an existing absolute destination")
    # Prepare allocations and scalar operands outside the profiled range.
    x = torch.empty(NCU_N, dtype=torch.float32, device="cuda").uniform_(-8, 8)
    tiny = torch.empty(NCU_WARM_N, dtype=torch.float32, device="cuda").uniform_(-8, 8)
    warm_call = make_cases(tiny, torch.empty_like(tiny), configs)[case]
    large_call = make_cases(x, torch.empty_like(x), configs)[case]
    torch.cuda.synchronize()
    record = {"schema_version": 1, "state": "PREPARED", "pid": os.getpid(),
              "case_index": case, "case_label": CASES[case], "n": int(x.numel()),
              "warmup_n": int(tiny.numel()), "dtype": str(x.dtype),
              "configs": {str(m): list(bw) for m, bw in configs.items()},
              "nvtx_range": _profile_range(case), "calls_completed": 0,
              "lowering_verification": "NOT_PERFORMED"}
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    entered = False
    try:
        torch.cuda.nvtx.range_push(record["nvtx_range"])
        entered = True
        warm_call()
        torch.cuda.synchronize()
        record.update(state="WARM_COMPLETE", calls_completed=1)
        atomic_text(path, json.dumps(record, indent=2) + "\n")
        large_call()
        torch.cuda.synchronize()
        record.update(state="COMPLETE", calls_completed=2)
        atomic_text(path, json.dumps(record, indent=2) + "\n")
    except BaseException as exc:
        record.update(state="PARTIAL", error={"type": type(exc).__name__, "message": str(exc)})
        atomic_text(path, json.dumps(record, indent=2) + "\n")
        raise
    finally:
        if entered:
            torch.cuda.nvtx.range_pop()


def _dimension(text: str) -> tuple[int, int, int]:
    """A raw CSV launch shape: one-dimensional size or explicit (x, y, z)."""
    if not isinstance(text, str):
        raise ValueError("missing launch dimensions")
    if re.fullmatch(r"[1-9][0-9]*", text):
        return int(text), 1, 1
    match = re.fullmatch(r"\(\s*([1-9][0-9]*)\s*,\s*([1-9][0-9]*)\s*,\s*([1-9][0-9]*)\s*\)", text)
    if not match:
        raise ValueError(f"invalid launch dimensions {text!r}")
    return tuple(int(v) for v in match.groups())


def _metric_value(text: str, metric: str) -> float:
    # Validate grouping before removing it: 1.234,5 is not 1234.5.
    if not isinstance(text, str) or not _NUMBER.fullmatch(text):
        raise ValueError(f"invalid base-unit value for {metric}: {text!r}")
    value = float(text.replace(",", ""))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"nonfinite or negative value for {metric}")
    return value


def _validate_probe(probe: dict, case: int, configs: dict, n: int) -> None:
    expected = {"schema_version": 1, "state": "COMPLETE", "case_index": case,
                "case_label": CASES[case], "n": n, "warmup_n": NCU_WARM_N,
                "dtype": "torch.float32", "configs": {str(m): list(bw) for m, bw in configs.items()},
                "nvtx_range": _profile_range(case), "calls_completed": 2,
                "lowering_verification": "NOT_PERFORMED"}
    if not isinstance(probe, dict) or any(type(probe.get(k)) is not type(v) or probe.get(k) != v for k, v in expected.items()):
        raise ValueError("probe manifest does not bind the completed tiny/NCU_N case pair")
    if type(probe.get("pid")) is not int or probe["pid"] <= 0:
        raise ValueError("probe manifest has no process identity")


def parse_ncu_csv(stdout: str, n: int, *, case: int, configs: dict, probe: dict) -> dict:
    """Parse raw-page wide CSV plus its units row and bind the two launches.

    Only the large launch is normalized. A synthetic fixture verifies this
    contract, not the behavior of a particular installed ncu/CUDA stack.
    """
    if type(n) is not int or n != NCU_N or type(case) is not int or not 0 <= case < len(CASES):
        raise ValueError("profile must name one fixed NCU_N case")
    configs = _profile_configs(configs)
    _validate_probe(probe, case, configs, n)
    # Only documented profiler status lines and blanks are skipped. An unquoted
    # extra launch or unexpected application output must not disappear.
    lines = [line for line in stdout.splitlines() if line.strip() and not line.startswith("==PROF==")]
    try:
        rows = list(csv.reader(io.StringIO("\n".join(lines)), strict=True))
    except csv.Error as exc:
        raise ValueError("malformed ncu raw CSV") from exc
    if not rows or len(set(rows[0])) != len(rows[0]):
        raise ValueError("ncu raw CSV has no unique header")
    header = rows[0]
    if not set(LAUNCH_COLUMNS + tuple(METRICS)) <= set(header):
        raise ValueError("ncu raw CSV is missing required launch/metric columns")
    if len(rows) != 4 or any(len(r) != len(header) for r in rows[1:]):
        raise ValueError("ncu raw CSV requires a units row and exactly two launch rows")
    units = dict(zip(header, rows[1]))
    if any(units[m] != unit for m, unit in zip(METRICS, BASE_UNITS)):
        raise ValueError("ncu raw CSV does not contain the requested exact base units")
    if any(units[k] for k in ("ID", "Process ID", "Kernel Name", "Context", "Stream")):
        raise ValueError("ncu raw CSV is missing its units row")
    parsed = []
    for values in rows[2:]:
        row = dict(zip(header, values))
        if any(not row[k] for k in LAUNCH_COLUMNS):
            raise ValueError("ncu raw CSV has an empty launch identity")
        if not all(re.fullmatch(r"[0-9]+", row[k]) for k in ("ID", "Process ID", "Context", "Stream")):
            raise ValueError("ncu raw CSV has an invalid launch identity")
        if int(row["Process ID"]) != probe["pid"]:
            raise ValueError("ncu launch belongs to another probe process")
        if not re.search(CASE_KERNEL_PATTERNS[case], row["Kernel Name"]):
            raise ValueError("ncu kernel does not match the configured case")
        grid, block = _dimension(row["Grid Size"]), _dimension(row["Block Size"])
        if grid[1:] != (1, 1) or block[1:] != (1, 1) or block[0] > 1024:
            raise ValueError("unsupported non-1D or oversized launch shape")
        metrics = {m: _metric_value(row[m], m) for m in METRICS}
        if metrics[METRICS[0]] <= 0 or metrics[METRICS[1]] <= 0:
            raise ValueError("profile duration and warp instruction count must be positive")
        metrics[METRICS[0]] *= 1e-9
        identity = {key: row[key] for key in LAUNCH_COLUMNS}
        parsed.append({"identity": identity, "grid": grid, "block": block, "metrics_si": metrics})
    warm, large = parsed
    if int(warm["identity"]["ID"]) >= int(large["identity"]["ID"]):
        raise ValueError("profile launches are duplicated or not in warm/large order")
    if any(warm["identity"][k] != large["identity"][k] for k in ("Process ID", "Kernel Name", "Context", "Stream", "Device", "CC")):
        raise ValueError("warm/large launch identities disagree")
    if warm["block"] != large["block"] or large["grid"][0] * NCU_WARM_N != warm["grid"][0] * n:
        raise ValueError("launch geometry does not bind tiny and NCU_N element counts")
    if case < N_TRITON:
        block, warps = configs[case]
        if (warm["grid"] != ((NCU_WARM_N + block - 1) // block, 1, 1)
                or large["grid"] != ((n + block - 1) // block, 1, 1)
                or large["block"] != (warps * 32, 1, 1)):
            raise ValueError("Triton launch differs from its swept BLOCK/warps configuration")
    metrics = large["metrics_si"]
    inst = metrics[METRICS[1]]
    return {"kernel": large["identity"]["Kernel Name"], "launch_identity": large["identity"],
            "grid": list(large["grid"]), "block": list(large["block"]),
            "warm_launch": {"identity": warm["identity"], "grid": list(warm["grid"]),
                            "block": list(warm["block"]), "metrics_si": warm["metrics_si"]},
            "profiled_elements": n, "metrics_si": metrics, "inst_executed": inst,
            "warp_instructions_per_32_elements": inst * 32 / n,
            "dram_bytes": metrics[METRICS[2]] + metrics[METRICS[3]], "time_s": metrics[METRICS[0]],
            "lowering_verification": "NOT_PERFORMED"}


def prepare_profile_dir(output_dir: Path) -> None:
    """Reserve and prove the profile destination before experiment work."""
    from .report import atomic_text
    output_dir.mkdir(exist_ok=False)
    atomic_text(output_dir / "preflight.txt", "Reserved for new native ncu reports, raw CSV, stderr and probe manifests.\n")


def ncu_counts(gpu_env: dict, configs: dict | None, output_dir: Path, progress=None) -> dict:
    """Capture all kernels in each tiny/large NVTX range and retain tool bytes.

    Capture, import, manifest or parse failures are explicit errors. The caller
    must mark a requested failure PARTIAL and exit nonzero. No timeout is added.
    """
    import json
    ncu = shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu was requested but is unavailable")
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        prepare_profile_dir(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError("ncu destination is not a directory")
    suffixes = (".ncu-rep", ".csv", ".stderr.txt", ".import.csv", ".import.stderr.txt", ".probe.json")
    for i in range(len(CASES)):
        if any((output_dir / f"case-{i}{s}").exists() for s in suffixes):
            raise FileExistsError("Refusing to replace an existing ncu artifact")
    configs = _profile_configs(configs)
    src = str(Path(__file__).resolve().parents[1])
    out = {"n": NCU_N, "warmup_n": NCU_WARM_N, "launches": [], "errors": [], "artifacts": [],
           "metric_units": {METRICS[0]: "seconds", METRICS[1]: "warp instructions", METRICS[2]: "bytes", METRICS[3]: "bytes"},
           "clock_control": "none", "cache_control": "all", "config_file": "off", "rename_kernels": "off", "configs": {str(m): list(bw) for m, bw in configs.items()},
           "capture_scope": "all kernels within the exact per-case NVTX tiny/large range; exactly two required",
           "case_kernel_patterns": list(CASE_KERNEL_PATTERNS), "lowering_verification": "NOT_PERFORMED"}
    console = ["--config-file", "off", "--rename-kernels", "off", "--csv", "--page", "raw", "--print-metric-name", "name", "--print-units", "base", "--print-kernel-base", "demangled"]
    for i, name in enumerate(CASES):
        stem = output_dir / f"case-{i}"
        native, csv_path, err_path, import_csv, import_err, probe_path = [stem.with_suffix(s) for s in suffixes]
        env = {**os.environ, **gpu_env, "DNR_CONFIGS": json.dumps(out["configs"]), "DNR_PROBE_PATH": str(probe_path)}
        cmd = [ncu, *console, "--clock-control", "none", "--cache-control", "all", "--metrics", ",".join(METRICS), "-o", str(native),
               "--nvtx", "--nvtx-include", _profile_range(i) + "/", sys.executable, "-P", "-c",
               f"import sys; sys.path.insert(0, {src!r}); from divide_not_reciprocal.timing import ncu_probe; ncu_probe({i})"]
        record = {"case": name, "case_index": i, "command": cmd, "nvtx_range": _profile_range(i)}
        try:
            r = subprocess.run(cmd, capture_output=True, env=env, cwd=output_dir)
            record["returncode"] = r.returncode
            with csv_path.open("xb") as stream:
                stream.write(r.stdout)
            with err_path.open("xb") as stream:
                stream.write(r.stderr)
            if r.returncode:
                raise RuntimeError(f"ncu exited {r.returncode}; see retained stdout/stderr")
            if not native.is_file() or native.stat().st_size == 0:
                raise RuntimeError("ncu did not create a nonempty native .ncu-rep report")
            import_cmd = [ncu, "--import", str(native), *console]
            record["import_command"] = import_cmd
            decoded = subprocess.run(import_cmd, capture_output=True, env=env, cwd=output_dir)
            record["import_returncode"] = decoded.returncode
            with import_csv.open("xb") as stream:
                stream.write(decoded.stdout)
            with import_err.open("xb") as stream:
                stream.write(decoded.stderr)
            if decoded.returncode:
                raise RuntimeError(f"ncu native report import exited {decoded.returncode}; see retained import output")
            probe = json.loads(probe_path.read_text())
            parsed = parse_ncu_csv(r.stdout.decode("utf-8"), NCU_N, case=i, configs=configs, probe=probe)
            imported = parse_ncu_csv(decoded.stdout.decode("utf-8"), NCU_N, case=i, configs=configs, probe=probe)
            if parsed != imported:
                raise ValueError("native report import and capture CSV disagree on launch identities or metrics_si")
            out["launches"].append({"case": name, "probe": probe, **parsed})
        except Exception as exc:
            out["errors"].append({"case": name, "type": type(exc).__name__, "message": str(exc)})
        finally:
            record["files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                               for p in (csv_path, err_path, native, import_csv, import_err, probe_path) if p.is_file()]
            out["artifacts"].append(record)
            if progress is not None:
                progress(out)
    return out
