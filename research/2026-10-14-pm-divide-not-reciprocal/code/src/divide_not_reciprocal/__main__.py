"""Run the maintained lab in a new external directory; handled failures retain PARTIAL output."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

STAGES = ("exhaustive", "triton", "hardsigmoid", "timing", "ncu")


def preflight(skip: set[str], quick: bool) -> dict:
    """Resolve tools and metadata before compiling or allocating experiment tensors."""
    from . import exhaustive
    torch = importlib.import_module("torch")
    triton_needed = "triton" not in skip or "timing" not in skip
    triton = importlib.import_module("triton") if triton_needed else None
    tools = {}
    if "exhaustive" not in skip:
        tools = {"nvcc": exhaustive.nvcc(), "cuobjdump": exhaustive.cuobjdump(),
                 "host_compiler": exhaustive.host_compiler()}
        for key in ("nvcc", "cuobjdump", "host_compiler"):
            tools[f"{key}_version"] = exhaustive.checked_command([tools[key], "--version"]).strip()
    if "timing" not in skip and "ncu" not in skip and not quick:
        tools["ncu"] = shutil.which("ncu")
        if not tools["ncu"]:
            raise RuntimeError("ncu was requested but is unavailable; use --skip ncu to omit profiling")
        tools["ncu_version"] = exhaustive.checked_command([tools["ncu"], "--version"]).strip()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device is available; no GPU experiments were started")
    free, total = torch.cuda.mem_get_info()
    return {"tools": tools, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "triton": triton.__version__ if triton is not None else "not requested",
            "gpu": torch.cuda.get_device_name(), "gpu_free_bytes_before": free, "gpu_total_bytes": total}


def main(argv=None) -> int:
    from . import report
    ap = argparse.ArgumentParser(prog="divide_not_reciprocal")
    ap.add_argument("--name", help="new run identifier; default is UTC timestamp plus random suffix")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, set before torch loads")
    ap.add_argument("--quick", action="store_true", help="fewer count chunks and timing iterations; tensor sizes and timing sweep stay unchanged")
    ap.add_argument("--skip", default="", help="comma-separated: " + ",".join(STAGES))
    ap.add_argument("--results", type=Path, default=report.default_results_root(), help="external parent directory for new runs")
    a = ap.parse_args(argv)
    skip = {s for s in a.skip.split(",") if s}
    if skip - set(STAGES):
        ap.error("unknown --skip stage(s): " + ", ".join(sorted(skip - set(STAGES))))
    try:
        run_dir, name = report.reserve_run(a.results, a.name)
    except (OSError, ValueError) as exc:
        print(f"Cannot reserve a new output directory: {exc}", file=sys.stderr)
        return 2
    gpu_env = {}
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
        gpu_env["CUDA_VISIBLE_DEVICES"] = a.gpu
    requested = [s for s in STAGES if s not in skip and not (s == "ncu" and (a.quick or "timing" in skip))]
    source = Path(__file__).resolve().parent
    R = {"date": report.today(), "started_utc": report.now(), "state": "PARTIAL", "quick": a.quick,
         "name": name, "run_directory": str(run_dir), "requested": requested,
         "stages": {s: "PENDING" for s in requested}, "errors": []}
    stage, code = "preflight", 0

    def checkpoint():
        R["updated_utc"] = report.now()
        report.write(run_dir, name, R)

    def progress(key):
        def save(value):
            R[key] = value
            checkpoint()
        return save

    try:
        checkpoint()  # Prove result writes before GPU work.
        R["implementation_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(source.iterdir()) if p.suffix in (".py", ".cu")}
        R["preflight"] = preflight(skip, a.quick)
        import torch
        from . import exhaustive, hardsigmoid, timing, triton_div
        if "ncu" in requested:
            timing.prepare_profile_dir(run_dir / "ncu")
        device = torch.device("cuda")
        smi_index = a.gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0] or None
        procs = {"start": report.smi("compute-apps=pid,process_name,used_memory", smi_index)}
        clocks = {"start": report.smi("gpu=clocks.sm", smi_index)}
        R["metadata_start"] = {"processes": procs["start"], "sm_clock": clocks["start"]}
        R["conditions"] = report.conditions(smi_index, clocks, procs, "Preflight metadata; end metadata is pending.", R["preflight"])
        checkpoint()
        if "exhaustive" in requested:
            stage = "exhaustive"
            R["stages"][stage] = "RUNNING"
            R[stage], R["builds"] = [], {}
            build_dir = run_dir / "build"
            build_dir.mkdir()
            for build_name, flags in exhaustive.BUILDS.items():
                b = exhaustive.build(build_dir, build_name, flags)
                R["builds"][build_name] = {"flags": flags, "arch": exhaustive.arch(), "library": str(b.lib)}
                (build_dir / f"count_{build_name}.sass").write_text(b.sass)
                checkpoint()
                for v in exhaustive.VARIANTS:
                    R[stage].append(exhaustive.summarize(exhaustive.run(b, v)))
                    checkpoint()
            R["stages"][stage] = "COMPLETE"
            checkpoint()
        for key, module in (("triton", triton_div), ("hardsigmoid", hardsigmoid)):
            if key in requested:
                stage = key
                R["stages"][key] = "RUNNING"
                checkpoint()
                R[key] = module.run(device, quick=a.quick, progress=progress(key))
                R["stages"][key] = "COMPLETE"
                checkpoint()
        if "timing" in requested:
            stage = "timing"
            R["stages"][stage] = "RUNNING"
            checkpoint()
            R[stage] = timing.time_cases(device, warmup=5 if a.quick else timing.WARMUP,
                                         iters=20 if a.quick else timing.ITERS)
            R["stages"][stage] = "COMPLETE"
            checkpoint()
            if "ncu" in requested:
                stage = "ncu"
                R["stages"][stage] = "RUNNING"
                checkpoint()
                R[stage] = timing.ncu_counts(gpu_env, R["timing"]["best"], run_dir / "ncu", progress("ncu"))
                if R[stage]["errors"]:
                    raise RuntimeError("Requested ncu capture failed; see ncu errors and artifacts")
                R["stages"][stage] = "COMPLETE"
                checkpoint()
        stage = "end_metadata"
        clocks["end"] = report.smi("gpu=clocks.sm", smi_index)
        procs["end"] = report.smi("compute-apps=pid,process_name,used_memory", smi_index)
        time_note = (f"Timing protocol: {R['timing']['warmup']} warm-up and {R['timing']['iters']} timed rotations, {len(timing.CASES)} cases. "
                     if "timing" in R else "Timing/profiling not requested. ")
        R["conditions"] = report.conditions(smi_index, clocks, procs,
            "Counts are not timings. " + time_note +
            "Table 1 milliseconds describe one census-kernel launch per variant, not an operation benchmark.", R["preflight"])
        R["state"] = "COMPLETE"
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        R["state"] = "PARTIAL"
        if stage in R["stages"]:
            R["stages"][stage] = "FAILED"
        if hasattr(exc, "partial_results"):
            R[f"{stage}_partial"] = exc.partial_results
        R["errors"].append({"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        print(f"PARTIAL ({stage}): {exc}", file=sys.stderr)
    finally:
        R["finished_utc"] = report.now()
        try:
            checkpoint()
        except BaseException as exc:
            code = code or 1
            R["state"] = "PARTIAL"
            R["errors"].append({"stage": "final_checkpoint", "type": type(exc).__name__, "message": str(exc)})
            # Markdown rendering may fail after JSON succeeds. Preserve the failure in JSON when the disk permits it.
            try:
                report.atomic_text(run_dir / (name + ".json"), json.dumps(R, indent=2, default=str, allow_nan=False) + "\n")
            except BaseException as fallback:
                print(f"JSON failure checkpoint could not be saved: {fallback}", file=sys.stderr)
            print(f"Result checkpoint failed: {exc}\nLatest state: {R!r}", file=sys.stderr)
        print(f"{R['state']}: {run_dir / (name + '.json')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
