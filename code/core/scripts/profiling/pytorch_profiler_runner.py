"""Utility to execute a target script under the PyTorch profiler."""
from __future__ import annotations

import argparse
import importlib.util
import contextlib
import runpy
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional

import torch
from core.profiling.nvtx_helper import standardize_nvtx_label
from core.profiling.torch_profiler_compat import prepare_torch_cuda_profiler
from arch_config import ArchitectureConfig
from torch.profiler import ProfilerActivity, profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a script under torch.profiler")
    parser.add_argument("script", type=Path, help="Path to the Python script to execute")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where profiler artifacts will be written",
    )
    parser.add_argument(
        "--profile-mode",
        default="full",
        choices=["full", "memory", "flops", "modules", "blackwell"],
        help="Profiler preset to use",
    )
    parser.add_argument(
        "--nvtx-label",
        default="aisp_torch_profile",
        help="NVTX/record_function range label to wrap the run (helps correlate with nsys/NVTX traces)",
    )
    parser.add_argument(
        "--nvtx/--no-nvtx",
        dest="use_nvtx",
        default=True,
        help="Emit a host-side NVTX range around the profiled section",
    )
    parser.add_argument(
        "--force-lineinfo/--no-force-lineinfo",
        dest="force_lineinfo",
        default=True,
        help="Force -lineinfo into NVCC/TORCH_NVCC_FLAGS for better source mapping (default: on)",
    )
    parser.add_argument(
        "--script-args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script",
    )
    return parser.parse_args()


def _set_env(output_dir: Path, force_lineinfo: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
    # Use new PYTORCH_ALLOC_CONF (preferred), fallback to legacy PYTORCH_CUDA_ALLOC_CONF
    alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if alloc_conf is None:
        alloc_conf = legacy_conf or "max_split_size_mb:256,expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    if force_lineinfo:
        def _append_flag(key: str, flag: str) -> None:
            current = os.environ.get(key, "").strip()
            if flag not in current.split():
                os.environ[key] = f"{flag} {current}".strip()
        _append_flag("NVCC_PREPEND_FLAGS", "-lineinfo")
        _append_flag("TORCH_NVCC_FLAGS", "-lineinfo")


def _load_module(script: Path) -> ModuleType:
    module_name = script.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _resolve_entrypoint(module: ModuleType, script: Path) -> Callable[[], None]:
    if hasattr(module, "main") and callable(module.main):  # type: ignore[attr-defined]
        return module.main  # type: ignore[return-value]
    if hasattr(module, "run") and callable(module.run):  # type: ignore[attr-defined]
        return module.run  # type: ignore[return-value]

    def _fallback() -> None:
        runpy.run_path(str(script))

    return _fallback


def _configure_profiler(mode: str) -> dict:
    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    config = {
        "activities": activities,
        "record_shapes": mode in {"full", "flops", "blackwell"},
        "with_stack": mode in {"full", "blackwell"},
        "with_flops": mode in {"full", "flops", "blackwell"},
        "with_modules": mode in {"full", "modules", "blackwell"},
        "profile_memory": mode in {"full", "memory", "blackwell"},
    }

    # Use immediate collection to support short-running scripts
    config["schedule"] = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
    return config


def _apply_blackwell_tuning() -> None:
    if not torch.cuda.is_available():
        return

    arch_cfg = ArchitectureConfig()

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if arch_cfg.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
        for attr in (
            "use_blackwell_optimizations",
            "hbm3e_optimizations",
            "tma_support",
            "stream_ordered_memory",
            "nvlink_c2c",
        ):
            if hasattr(triton_cfg, attr):
                setattr(triton_cfg, attr, True)
        if hasattr(triton_cfg, "autotune_mode"):
            triton_cfg.autotune_mode = "max-autotune"

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_cfg is not None and hasattr(dynamo_cfg, "automatic_dynamic_shapes"):
        dynamo_cfg.automatic_dynamic_shapes = True


@contextlib.contextmanager
def _nvtx_range(label: str, enabled: bool):
    """Emit both NVTX and record_function ranges for cross-profiler linking."""
    range_label = standardize_nvtx_label(label) if enabled else label
    rf_cm = torch.autograd.profiler.record_function(range_label) if enabled else contextlib.nullcontext()
    nvtx_push = getattr(torch.cuda.nvtx, "range_push", None)
    nvtx_pop = getattr(torch.cuda.nvtx, "range_pop", None)
    has_nvtx = enabled and torch.cuda.is_available() and nvtx_push and nvtx_pop

    try:
        if has_nvtx:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            nvtx_push(range_label)
        with rf_cm:
            yield
    finally:
        if has_nvtx:
            try:
                nvtx_pop()
            except Exception:
                pass


def _write_summary(prof: torch.profiler.profile, output_dir: Path, mode: str) -> Optional[Path]:
    """Persist a lightweight summary for API/UI consumption."""
    try:
        averages = prof.key_averages()
        rows: List[Dict[str, object]] = []
        total_cuda = 0.0
        total_cpu = 0.0
        for evt in averages:
            cuda_time = float(getattr(evt, "cuda_time_total", 0.0) or 0.0)
            cpu_time = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
            total_cuda += cuda_time
            total_cpu += cpu_time
            rows.append(
                {
                    "name": getattr(evt, "key", ""),
                    "count": getattr(evt, "count", 0),
                    "cuda_time_total_us": cuda_time,
                    "cpu_time_total_us": cpu_time,
                    "self_cuda_time_total_us": float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0),
                    "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0),
                    "cpu_memory_usage": float(getattr(evt, "cpu_memory_usage", 0.0) or 0.0),
                    "cuda_memory_usage": float(getattr(evt, "cuda_memory_usage", 0.0) or 0.0),
                }
            )
        summary = {
            "mode": mode,
            "total_cuda_time_us": total_cuda,
            "total_cpu_time_us": total_cpu,
            "top_ops": sorted(rows, key=lambda r: r.get("cuda_time_total_us", 0.0), reverse=True)[:50],
        }
        summary_path = output_dir / "torch_profile_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path
    except Exception:
        return None


def run_profiler(
    script: Path,
    output_dir: Path,
    mode: str,
    script_args: Optional[Iterable[str]],
    nvtx_label: str = "aisp_torch_profile",
    use_nvtx: bool = True,
    force_lineinfo: bool = True,
) -> None:
    _set_env(output_dir, force_lineinfo=force_lineinfo)
    if mode in {"full", "blackwell"}:
        _apply_blackwell_tuning()

    sys.argv = [str(script)] + list(script_args or [])
    module = _load_module(script)
    entry = _resolve_entrypoint(module, script)

    profiler_kwargs = _configure_profiler(mode)
    prepare_torch_cuda_profiler(
        cuda_profile_requested=ProfilerActivity.CUDA
        in profiler_kwargs["activities"]
    )
    start_ts = time.time()
    error: Optional[str] = None

    try:
        with profile(**profiler_kwargs) as prof:
            with _nvtx_range(nvtx_label, enabled=use_nvtx):
                entry()
            prof.step()
    except Exception as exc:  # pragma: no cover - surfaced in metadata
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = time.time() - start_ts
        metadata = {
            "script": str(script),
            "args": list(script_args or []),
            "mode": mode,
            "start_time": start_ts,
            "duration_seconds": duration,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "error": error,
            "nvtx_label": nvtx_label,
            "force_lineinfo": bool(force_lineinfo),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        if 'prof' in locals() and getattr(prof, "profiler", None) is not None:
            try:
                trace_path = output_dir / f"chrome_trace_{mode}.json"
                prof.export_chrome_trace(str(trace_path))
                # Standardize name for loaders
                alias_path = output_dir / "trace.json"
                if not alias_path.exists():
                    try:
                        alias_path.write_bytes(trace_path.read_bytes())
                    except Exception:
                        pass

                summary_path = output_dir / f"summary_{mode}.txt"
                parts: List[str] = []
                parts.append("Top operators by CUDA time\n")
                try:
                    parts.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                except Exception:
                    parts.append("CUDA metrics unavailable\n")
                parts.append("\nTop operators by CPU time\n")
                parts.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
                if profiler_kwargs.get("profile_memory"):
                    parts.append("\nTop operators by CPU memory usage\n")
                    parts.append(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=25))
                if profiler_kwargs.get("with_flops"):
                    parts.append("\nTop operators by FLOPs\n")
                    try:
                        parts.append(prof.key_averages().table(sort_by="flops", row_limit=25))
                    except Exception:
                        parts.append("FLOP metrics unavailable\n")
                summary_path.write_text("\n".join(parts))

                try:
                    averages = prof.key_averages()
                    rows: List[Dict[str, object]] = []
                    for evt in averages:
                        rows.append(
                            {
                                "name": getattr(evt, "key", ""),
                                "count": getattr(evt, "count", 0),
                                "cpu_time_total_us": getattr(evt, "cpu_time_total", 0.0),
                                "cuda_time_total_us": getattr(evt, "cuda_time_total", 0.0),
                                "self_cpu_time_total_us": getattr(evt, "self_cpu_time_total", 0.0),
                                "self_cuda_time_total_us": getattr(evt, "self_cuda_time_total", 0.0),
                                "cpu_memory_usage": getattr(evt, "cpu_memory_usage", 0.0),
                                "cuda_memory_usage": getattr(evt, "cuda_memory_usage", 0.0),
                            }
                        )
                    averages_path = output_dir / f"key_averages_{mode}.json"
                    averages_path.write_text(json.dumps(rows, indent=2))
                    _write_summary(prof, output_dir, mode)
                except AttributeError:
                    pass
            except Exception:
                pass


def main() -> None:
    args = _parse_args()
    run_profiler(
        args.script,
        args.output_dir,
        args.profile_mode,
        args.script_args,
        nvtx_label=args.nvtx_label,
        use_nvtx=bool(args.use_nvtx),
        force_lineinfo=bool(args.force_lineinfo),
    )


if __name__ == "__main__":
    main()
