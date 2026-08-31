"""
Blackwell Profiling Guide
=========================

Provides utilities and quick references for profiling NVIDIA B200 systems with
Nsight Systems, Nsight Compute, and the PyTorch profiler. Static ceilings in
this module are the reviewed B200 profile; they do not describe B300.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
import torch.profiler as profiler

from core.analysis.deep_profiling_report import parse_ncu_csv
from core.benchmark.metrics import BLACKWELL_B200, hardware_specs_for_device
from core.profiling.nsight_systems import NsightSystemsReportParser
from core.profiling.nvtx_helper import standardize_nvtx_label


class NsightSystemsProfiler(NsightSystemsReportParser):
    """Chapter capture context plus core-owned offline report parsing."""

    def __init__(
        self,
        output_name: str,
        *,
        trace_nvtx: bool = True,
    ) -> None:
        self.output_name = output_name
        self.trace_nvtx = trace_nvtx
        self._nvtx_pushed = False

    def __enter__(self) -> NsightSystemsProfiler:
        """Push an NVTX range when CUDA is available."""
        if self.trace_nvtx and torch.cuda.is_available():
            try:
                label = standardize_nvtx_label(f"step:profile_{self.output_name}")
                torch.cuda.nvtx.range_push(label)
                self._nvtx_pushed = True
            except Exception:
                self._nvtx_pushed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        """Pop NVTX range and synchronize if needed."""
        if self._nvtx_pushed:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
                self._nvtx_pushed = False

    @staticmethod
    def profile_command(
        script_path: str,
        output_name: str,
        duration: int = 30,
        *,
        ib_switch_guids: list[str] | None = None,
        use_hardware_trace: bool = True,
        include_sqlite_export: bool = True,
    ) -> str:
        """Return a CLI command that records an Nsight Systems capture."""
        trace_domain = (
            "cuda-hw,osrt,nvtx,ucx,gds" if use_hardware_trace else "cuda,osrt,nvtx,ucx,gds"
        )
        cmd = [
            "nsys",
            "profile",
            "-o",
            output_name,
            f"--trace={trace_domain}",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=graph",
            "--cuda-event-trace=true",
            "--sample=cpu",
            "--gpu-metrics-devices=all",
            "--nic-metrics=true",
            "--storage-metrics",
            "--storage-devices=all",
            "--gds-metrics=driver",
            f"--duration={duration}",
            "--force-overwrite=true",
        ]
        if ib_switch_guids:
            cmd.append(f"--ib-switch-metrics-device={','.join(ib_switch_guids)}")
        if include_sqlite_export:
            cmd.append("--export=sqlite")
        cmd.extend(["python", script_path])
        return " ".join(cmd)

    @staticmethod
    def analyze_blackwell_metrics(report_path: str) -> dict[str, Any]:
        """Parse an Nsight Systems artifact and return its structured status."""
        summary = NsightSystemsProfiler.summarize_report(
            report_path,
            print_summary=True,
        )
        print(f"=== Nsight Systems Analysis: {report_path} ===")
        print("Key Metrics:")
        print(" 1. GPU Utilization: inspect active SMs for the artifact's exact GPU SKU")
        print(" 2. Memory Bandwidth: compare with a reviewed profile for that exact SKU")
        print(" 3. Tensor Core Utilization: target >70%")
        print(" 4. Kernel Launch Overhead: <100 µs when using CUDA Graphs")
        print(
            " 5. NVLink bandwidth (multi-GPU): "
            f"~{BLACKWELL_B200.nvlink_bandwidth_gbps:g} GB/s aggregate one-way per GPU"
        )
        print(f"\nOpen the trace with: nsys-ui {report_path}")
        return {
            "status": "parsed",
            "tool": "nsight_systems",
            "report_path": summary["report"],
            "summary": summary,
        }


class NsightComputeProfiler:
    """Helpers for Nsight Compute profiling on Blackwell."""

    @staticmethod
    def profile_kernel_command(
        script_path: str, output_name: str, kernel_filter: str | None = None
    ) -> str:
        metrics = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__time_duration.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
        ]
        output_path = Path(output_name)
        if output_path.suffix.lower() in {".csv", ".ncu-rep"}:
            output_path = output_path.with_suffix("")
        csv_path = Path(f"{output_path}.csv")
        binary_report_path = Path(f"{output_path}.ncu-rep")
        cmd = [
            "ncu",
            "--metrics",
            ",".join(metrics),
            "--csv",
            "--page",
            "raw",
            "--log-file",
            str(csv_path),
            "--kernel-name-base",
            "demangled",
            "--export",
            str(binary_report_path),
            "--force-overwrite",
        ]
        if kernel_filter:
            cmd.extend(["--kernel-name", kernel_filter])
        cmd.extend(["python", script_path])
        return " ".join(cmd)

    @staticmethod
    def analyze_blackwell_kernel(report_path: str) -> dict[str, Any]:
        report = Path(report_path).expanduser()
        if not report.exists():
            raise FileNotFoundError(
                f"Nsight Compute artifact not found: {report}; expected a raw .csv export"
            )
        if not report.is_file():
            raise ValueError(f"Nsight Compute artifact is not a file: {report}")
        if report.suffix.lower() == ".ncu-rep":
            if report.stat().st_size == 0:
                raise ValueError(
                    "Nsight Compute .ncu-rep artifact is empty; capture or export a "
                    "non-empty raw CSV artifact"
                )
            raise ValueError(
                "Binary .ncu-rep artifacts are not parsed by this offline guide; "
                "export one with 'ncu --import REPORT --csv --page raw' and pass "
                "the resulting .csv file"
            )
        if report.suffix.lower() != ".csv":
            raise ValueError(
                f"Unsupported Nsight Compute artifact {report}; expected a raw .csv export"
            )
        if report.stat().st_size == 0:
            raise ValueError(f"Nsight Compute CSV is empty: {report}")
        with report.open("rb") as artifact:
            prefix = artifact.read(4096)
        if b"\x00" in prefix:
            raise ValueError(f"Nsight Compute CSV must be text, not binary data: {report}")
        try:
            prefix.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Nsight Compute CSV must be UTF-8 text, not binary data: {report}"
            ) from exc

        parsed = parse_ncu_csv(report)
        if not parsed:
            raise ValueError(
                "Nsight Compute CSV is empty or unrecognized; expected raw CSV with "
                "an ID, kernel name, metric name, metric value, and metric unit"
            )
        kernels = [
            {
                "name": kernel.name,
                "metric_count": len(kernel.metrics),
                "metrics": {
                    name: {
                        "value": metric.value,
                        "unit": metric.unit,
                        "section": metric.section,
                    }
                    for name, metric in sorted(kernel.metrics.items())
                },
            }
            for kernel in parsed.values()
        ]
        print(f"=== Nsight Compute Analysis: {report_path} ===")
        print(f"Parsed {len(kernels)} kernel launch(es) from {report.resolve()}")
        print(" 1. Record the exact GPU SKU, precision, and dense/sparse convention")
        print(" 2. Compare throughput with a reviewed profile for that exact SKU")
        print(" 3. Inspect HBM/L2/TMEM rates without assigning another SKU's ceiling")
        print(" 4. Inspect warp efficiency, divergence, and achieved occupancy")
        print(" 5. Confirm cluster, DSM, TMA, and tcgen05 use from emitted metrics/code")
        print("\nOpen the companion .ncu-rep artifact with ncu-ui when available")
        return {
            "status": "parsed",
            "tool": "nsight_compute",
            "artifact_type": "ncu_raw_csv",
            "report_path": str(report.resolve()),
            "kernel_count": len(kernels),
            "kernels": kernels,
        }


def profile_with_pytorch_profiler(
    fn: Callable[[], None],
    output_dir: str = "./profiling_results",
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
) -> None:
    """Run the PyTorch profiler on the provided callable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    before_files = {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_path.rglob("*")
        if path.is_file()
    }
    activities = [profiler.ProfilerActivity.CPU]
    include_cuda = False
    if torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            activities.append(profiler.ProfilerActivity.CUDA)
            include_cuda = True
        except Exception:
            include_cuda = False

    wait_steps = 1
    warmup_steps = 5
    active_steps = 3
    repeat = 1
    trace_ready_calls = 0
    trace_handler = profiler.tensorboard_trace_handler(str(output_path))

    def _write_trace(profile) -> None:
        nonlocal trace_ready_calls
        trace_handler(profile)
        trace_ready_calls += 1

    with profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=_write_trace,
        schedule=profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=repeat,
        ),
    ) as prof:
        schedule_steps = (wait_steps + warmup_steps + active_steps) * repeat
        for _ in range(schedule_steps):
            fn()
            prof.step()

    changed_trace_files = []
    for path in output_path.rglob("*"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        current = (path.stat().st_mtime_ns, path.stat().st_size)
        if before_files.get(path.resolve()) != current:
            changed_trace_files.append(path)
    if trace_ready_calls == 0 or not changed_trace_files:
        raise RuntimeError("PyTorch profiler completed without producing a fresh TensorBoard trace")

    sort_key = "cuda_time_total" if include_cuda else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    print(f"\nTensorBoard trace saved to {output_path}")
    print(f"Launch TensorBoard with: tensorboard --logdir={output_path}")


def require_reviewed_b200_device() -> torch.device:
    """Return the active CUDA device only when it matches the reviewed B200 profile."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SKIPPED: the B200 profiling workflow requires an available NVIDIA B200; "
            "CPU profiling is a separate workflow"
        )
    try:
        device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device_index)
    except Exception as exc:
        raise RuntimeError(
            "SKIPPED: unable to inspect the active CUDA device for the B200 profiling workflow"
        ) from exc

    capability = (int(properties.major), int(properties.minor))
    try:
        profile = hardware_specs_for_device(properties.name, capability)
    except ValueError as exc:
        raise RuntimeError(
            "SKIPPED: active CUDA device "
            f"{properties.name!r} at compute capability "
            f"{capability[0]}.{capability[1]} does not match the reviewed "
            "NVIDIA B200 profile"
        ) from exc
    if profile is not BLACKWELL_B200:
        raise RuntimeError(
            "SKIPPED: the B200 profiling workflow requires the reviewed NVIDIA B200 profile"
        )

    device = torch.device("cuda", device_index)
    try:
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"SKIPPED: unable to acquire reviewed NVIDIA B200 device cuda:{device_index}"
        ) from exc
    return device


def _require_workload_on_device(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> None:
    if input_tensor.device != device:
        raise RuntimeError(
            "B200 profiling workflow requires input_tensor on "
            f"{device}, but received {input_tensor.device}"
        )
    for kind, tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in tensors:
            if tensor.device != device:
                raise RuntimeError(
                    f"B200 profiling workflow requires model {kind} {name!r} on "
                    f"{device}, but received {tensor.device}"
                )


def complete_profiling_workflow(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_dir: str = "./profiling_blackwell",
) -> None:
    """Demonstrate a profiling workflow on an exact reviewed B200 device."""
    device = require_reviewed_b200_device()
    _require_workload_on_device(model, input_tensor, device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Complete Profiling Workflow for Blackwell")
    print("=" * 80)

    def run_model() -> None:
        with torch.inference_mode():
            model(input_tensor)

    print("\nStep 1: PyTorch Profiler")
    profile_with_pytorch_profiler(run_model, output_dir=f"{output_dir}/pytorch_profiler")

    print("\nStep 2: Nsight Systems")
    print(NsightSystemsProfiler.profile_command("your_script.py", f"{output_dir}/nsys_trace"))

    print("\nStep 3: Nsight Compute")
    print(
        NsightComputeProfiler.profile_kernel_command("your_script.py", f"{output_dir}/ncu_report")
    )


def print_quick_reference() -> None:
    """Print quick reference commands for profiling."""
    print("=" * 80)
    print("Blackwell Profiling Quick Reference")
    print("=" * 80)
    print("\nNsight Systems:")
    print(
        "nsys profile -o trace --trace=cuda-hw,osrt,nvtx,ucx,gds "
        "--trace-fork-before-exec=true --cuda-graph-trace=graph "
        "--cuda-event-trace=true --sample=cpu --gpu-metrics-devices=all "
        "--nic-metrics=true --storage-metrics --storage-devices=all "
        "--gds-metrics=driver python script.py"
    )
    print("\nNsight Compute:")
    print(NsightComputeProfiler.profile_kernel_command("script.py", "report"))
    print("\nPyTorch Profiler: use profile_with_pytorch_profiler(fn)")
    print("=" * 80)


class BlackwellMetricsGuide:
    """Helpful Nsight Compute metrics for Blackwell SM 10.0."""

    @staticmethod
    def get_essential_blackwell_metrics() -> list[str]:
        return [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
            "gpu__time_duration.sum",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram_read_throughput",
            "dram_write_throughput",
            "sm__inst_executed_pipe_tensor_op.sum",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "l2_cache_hit_rate",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "achieved_occupancy",
            "sm__ctas_launched.sum",
        ]

    @staticmethod
    def print_metric_guide() -> None:
        print("=" * 80)
        print("Blackwell SM 10.0 Metrics Guide")
        print("=" * 80)
        peak_tb_s = BLACKWELL_B200.hbm_bandwidth_gbps / 1000.0
        print(
            "HBM3e memory bandwidth (dram__throughput): "
            f"target >90% of reviewed B200 peak ({peak_tb_s:g} TB/s)"
        )
        print("Tensor core utilization (sm__pipe_tensor_cycles_active): target >70%")
        print("L2 cache hit rate: >80% when data is reused")
        print("Warp efficiency (smsp__warps_launched / active): >80%")
        print("Occupancy (achieved_occupancy): >0.5 for most kernels")


class HBMMemoryAnalyzer:
    """Integrate B200 HBM throughput and report L2 sector traffic separately.

    Aggregate DRAM bytes and L2 sector counts describe different hierarchy
    levels.  They cannot establish transaction or coalescing efficiency without
    requested-byte counters, so this analyzer does not invent such a percentage.
    """

    HBM3E_PEAK_BW = BLACKWELL_B200.hbm_bandwidth_gbps  # Decimal GB/s
    L2_SECTOR_SIZE_BYTES = 32

    @staticmethod
    def analyze_memory_pattern(
        dram_read_throughput: float,
        dram_write_throughput: float,
        l2_read_sectors: int,
        l2_write_sectors: int,
        kernel_duration_ns: float,
    ) -> dict[str, Any]:
        if (
            isinstance(dram_read_throughput, bool)
            or not isinstance(dram_read_throughput, Real)
            or isinstance(dram_write_throughput, bool)
            or not isinstance(dram_write_throughput, Real)
            or not math.isfinite(dram_read_throughput)
            or not math.isfinite(dram_write_throughput)
            or dram_read_throughput < 0
            or dram_write_throughput < 0
        ):
            raise ValueError("DRAM throughput must be finite and non-negative")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in (l2_read_sectors, l2_write_sectors)
        ):
            raise ValueError("L2 sector counts must be non-bool, non-negative integers")
        if (
            isinstance(kernel_duration_ns, bool)
            or not isinstance(kernel_duration_ns, Real)
            or not math.isfinite(kernel_duration_ns)
            or kernel_duration_ns < 0
        ):
            raise ValueError("kernel_duration_ns must be finite and non-negative")
        if kernel_duration_ns == 0 and any(
            value > 0
            for value in (
                dram_read_throughput,
                dram_write_throughput,
                l2_read_sectors,
                l2_write_sectors,
            )
        ):
            raise ValueError("kernel_duration_ns must be positive for nonzero observations")

        total_throughput = dram_read_throughput + dram_write_throughput
        bw_utilization = (total_throughput / HBMMemoryAnalyzer.HBM3E_PEAK_BW) * 100
        seconds = kernel_duration_ns / 1e9
        # Throughput inputs are decimal GB/s; convert the integrated traffic to bytes.
        dram_read_bytes = dram_read_throughput * 1e9 * seconds
        dram_write_bytes = dram_write_throughput * 1e9 * seconds
        return {
            "bandwidth_utilization_pct": bw_utilization,
            "total_throughput_gbps": total_throughput,
            "dram_read_bytes": dram_read_bytes,
            "dram_write_bytes": dram_write_bytes,
            "l2_read_sector_bytes": (l2_read_sectors * HBMMemoryAnalyzer.L2_SECTOR_SIZE_BYTES),
            "l2_write_sector_bytes": (l2_write_sectors * HBMMemoryAnalyzer.L2_SECTOR_SIZE_BYTES),
            "coalescing_efficiency_pct": None,
            "coalescing_status": (
                "not_computable_without_requested_bytes_and_matching_l2_sector_counts"
            ),
        }

    @staticmethod
    def print_hbm3e_best_practices() -> None:
        print("=" * 80)
        print("HBM3e Best Practices")
        print("=" * 80)
        print(" - Align data to 128-byte cache lines.")
        print(" - Measure requested bytes against 32-byte L2 sectors before claiming coalescing.")
        print(" - Use float4/int4 vectorization.")
        print(" - Employ cache streaming modifiers for write-only traffic.")


def run_complete_blackwell_analysis(
    nsys_report: str,
    ncu_report: str,
) -> dict[str, Any]:
    """Parse both profiler artifacts before returning combined success."""
    print("=" * 80)
    print("Complete Blackwell Analysis")
    print("=" * 80)
    nsys_status = NsightSystemsProfiler.analyze_blackwell_metrics(nsys_report)
    if nsys_status.get("status") != "parsed":
        raise RuntimeError("Nsight Systems analysis did not return parsed status")
    ncu_status = NsightComputeProfiler.analyze_blackwell_kernel(ncu_report)
    if ncu_status.get("status") != "parsed":
        raise RuntimeError("Nsight Compute analysis did not return parsed status")
    BlackwellMetricsGuide.print_metric_guide()
    HBMMemoryAnalyzer.print_hbm3e_best_practices()
    return {
        "status": "success",
        "hardware_profile": "b200",
        "nsys": nsys_status,
        "ncu": ncu_status,
    }


if __name__ == "__main__":
    print("=== Blackwell Profiling Guide ===")
    exec_device = require_reviewed_b200_device()
    print(f"GPU: {torch.cuda.get_device_properties(exec_device.index).name}")
    print_quick_reference()
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).to(exec_device)
    x = torch.randn(32, 1024, device=exec_device)

    def run_model() -> None:
        with torch.inference_mode():
            model(x)

    profile_with_pytorch_profiler(run_model, output_dir="./example_profiling")
