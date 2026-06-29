"""
Blackwell Profiling Guide
=========================

Provides utilities and quick references for profiling Blackwell B200/B300
systems with Nsight Systems, Nsight Compute, and the PyTorch profiler.
"""

from __future__ import annotations

import csv
import heapq
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.profiler as profiler
from core.profiling.nvtx_helper import standardize_nvtx_label


class NsightSystemsProfiler:
    """Context manager and helpers for Nsight Systems profiling."""

    def __init__(
        self,
        output_name: str,
        trace_cuda: bool = True,
        trace_nvtx: bool = True,
        trace_cudnn: bool = True,
        trace_cublas: bool = True,
    ) -> None:
        self.output_name = output_name
        self.trace_cuda = trace_cuda
        self.trace_nvtx = trace_nvtx
        self.trace_cudnn = trace_cudnn
        self.trace_cublas = trace_cublas
        self._nvtx_pushed = False

    def __enter__(self) -> "NsightSystemsProfiler":
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
        ib_switch_guids: Optional[List[str]] = None,
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
    def analyze_blackwell_metrics(report_path: str) -> None:
        """Print key metrics to review inside Nsight Systems."""
        print(f"=== Nsight Systems Analysis: {report_path} ===")
        print("Key Metrics:")
        print(" 1. GPU Utilization: target >80% on 148 SMs")
        print(" 2. Memory Bandwidth: target >7 TB/s (HBM3e)")
        print(" 3. Tensor Core Utilization: target >70%")
        print(" 4. Kernel Launch Overhead: <100 µs when using CUDA Graphs")
        print(" 5. NVLink bandwidth (multi-GPU): ~900 GB/s per link")
        print(f"\nOpen the trace with: nsys-ui {report_path}")
        try:
            NsightSystemsProfiler.summarize_report(report_path, print_summary=True)
        except Exception as exc:
            print(f"[warning] Failed to summarize Nsight Systems report: {exc}")

    @staticmethod
    def summarize_report(
        report_path: str,
        *,
        kernel_regex: Optional[str] = None,
        top_k: int = 5,
        print_summary: bool = True,
    ) -> Dict[str, Any]:
        """Parse an Nsight Systems report via `nsys stats`."""
        report = Path(report_path)
        if not report.exists():
            raise FileNotFoundError(f"Nsight Systems report not found: {report_path}")

        summary_rows = NsightSystemsProfiler._run_nsys_stats(report, "summary")
        kernel_rows = NsightSystemsProfiler._run_nsys_stats(report, "cuda_gpu_kern_sum")
        kernels = NsightSystemsProfiler._filter_and_rank_kernels(
            kernel_rows, kernel_regex, top_k
        )

        summary: Dict[str, Any] = {
            "report": str(report.resolve()),
            "summary": summary_rows,
            "kernels": kernels,
        }
        if print_summary:
            NsightSystemsProfiler._print_nsys_summary(summary, kernel_regex)
        return summary

    @staticmethod
    def _run_nsys_stats(report: Path, section: str) -> List[Dict[str, str]]:
        cmd = [
            "nsys",
            "stats",
            "--format",
            "csv",
            "--report",
            section,
            "--force-export",
            "true",
            str(report),
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI
            raise RuntimeError(f"Failed to run {' '.join(cmd)}\n{exc.stderr.strip()}") from exc

        rows: List[Dict[str, str]] = []
        headers: Optional[List[str]] = None
        reader = csv.reader(proc.stdout.splitlines())
        for raw_row in reader:
            if not raw_row:
                headers = None
                continue
            if raw_row[0].startswith("#") or raw_row[0].startswith("NOTICE"):
                headers = None
                continue
            if headers is None:
                headers = raw_row
                continue
            padded = (raw_row + [""] * len(headers))[: len(headers)]
            rows.append({headers[i]: padded[i] for i in range(len(headers))})
        return rows

    @staticmethod
    def _filter_and_rank_kernels(
        kernel_rows: List[Dict[str, str]],
        kernel_regex: Optional[str],
        top_k: int,
    ) -> List[Dict[str, str]]:
        rows: Iterable[Dict[str, str]] = kernel_rows
        if kernel_regex:
            pattern = re.compile(kernel_regex)
            rows = (row for row in kernel_rows if pattern.search(row.get("Name", "")))

        def parse_pct(row: Dict[str, str]) -> float:
            value = row.get("Time (%)") or row.get("Time (%) [sum]", "0")
            try:
                return float(value.replace("%", "").replace('"', ""))
            except ValueError:
                return 0.0

        return heapq.nlargest(top_k, rows, key=parse_pct)

    @staticmethod
    def _print_nsys_summary(summary: Dict[str, Any], kernel_regex: Optional[str]) -> None:
        print(f"\n=== Nsight Systems Summary: {summary['report']} ===")
        kernels = summary["kernels"]
        if kernel_regex:
            print(f"Filter: {kernel_regex}")
        if not kernels:
            print("No kernel entries found.")
            return
        for idx, row in enumerate(kernels, 1):
            name = row.get("Name", "Unknown")
            pct = row.get("Time (%)") or row.get("Time (%) [sum]", "0")
            ns = (
                row.get("Total Time (ns)")
                or row.get("Time (ns)")
                or row.get("Total Time (ns) [sum]")
                or "0"
            )
            try:
                ms = float(ns.replace('"', "")) / 1e6
            except ValueError:
                ms = 0.0
            print(f"{idx:>2}. {name}  Time: {ms:.3f} ms  Share: {pct}")


class NsightComputeProfiler:
    """Helpers for Nsight Compute profiling on Blackwell."""

    @staticmethod
    def profile_kernel_command(
        script_path: str, output_name: str, kernel_filter: Optional[str] = None
    ) -> str:
        metrics = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__time_duration.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum",
        ]
        cmd = [
            "ncu",
            "--metrics",
            ",".join(metrics),
            "--kernel-name-base",
            "demangled",
            f"--export={output_name}",
            "--force-overwrite",
        ]
        if kernel_filter:
            cmd.extend(["--kernel-name", kernel_filter])
        cmd.extend(["python", script_path])
        return " ".join(cmd)

    @staticmethod
    def analyze_blackwell_kernel(report_path: str) -> None:
        print(f"=== Nsight Compute Analysis: {report_path} ===")
        print(" 1. Compute throughput targets:")
        print("    - FP8 Tensor Cores: >1000 TFLOPS")
        print("    - FP16 Tensor Cores: >600 TFLOPS")
        print(" 2. HBM3e bandwidth: >7 TB/s (aim for >90% of peak 7.8 TB/s)")
        print(" 3. Warp efficiency: >80% active, <10% divergence")
        print(" 4. Occupancy: >75% of theoretical for compute-bound kernels")
        print(" 5. Thread block clusters / DSM usage: confirm when relevant")
        print(f"\nOpen the profile with: ncu-ui {report_path}")


def profile_with_pytorch_profiler(
    fn: Callable[[], None],
    output_dir: str = "./profiling_results",
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
) -> None:
    """Run the PyTorch profiler on the provided callable."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    with profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        on_trace_ready=profiler.tensorboard_trace_handler(output_dir),
        schedule=profiler.schedule(wait=1, warmup=5, active=3, repeat=1),
    ) as prof:
        for _ in range(5):
            fn()
            prof.step()

    sort_key = "cuda_time_total" if include_cuda else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    print(f"\nTensorBoard trace saved to {output_dir}")
    print(f"Launch TensorBoard with: tensorboard --logdir={output_dir}")


def complete_profiling_workflow(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_dir: str = "./profiling_blackwell",
) -> None:
    """Demonstrate a full profiling workflow for Blackwell."""
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
    print(NsightComputeProfiler.profile_kernel_command("your_script.py", f"{output_dir}/ncu_report"))


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
    print(
        "ncu --metrics "
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed "
        "--export=report python script.py"
    )
    print("\nPyTorch Profiler: use profile_with_pytorch_profiler(fn)")
    print("=" * 80)


class BlackwellMetricsGuide:
    """Helpful Nsight Compute metrics for Blackwell SM 10.0."""

    @staticmethod
    def get_essential_blackwell_metrics() -> List[str]:
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
        print("HBM3e memory bandwidth (dram__throughput): target >90% of 7.8 TB/s")
        print("Tensor core utilization (sm__pipe_tensor_cycles_active): target >70%")
        print("L2 cache hit rate: >80% when data is reused")
        print("Warp efficiency (smsp__warps_launched / active): >80%")
        print("Occupancy (achieved_occupancy): >0.5 for most kernels")


class HBMMemoryAnalyzer:
    """Utility routines for evaluating HBM3e access patterns."""

    HBM3E_PEAK_BW = 7800  # GB/s
    SECTOR_SIZE = 128  # bytes
    OPTIMAL_BURST = 256  # bytes

    @staticmethod
    def analyze_memory_pattern(
        dram_read_throughput: float,
        dram_write_throughput: float,
        l2_read_sectors: int,
        l2_write_sectors: int,
        kernel_duration_ns: float,
    ) -> Dict[str, Any]:
        total_throughput = dram_read_throughput + dram_write_throughput
        bw_utilization = (total_throughput / HBMMemoryAnalyzer.HBM3E_PEAK_BW) * 100
        seconds = kernel_duration_ns / 1e9
        dram_read_bytes = dram_read_throughput * seconds
        dram_write_bytes = dram_write_throughput * seconds
        avg_bytes_per_read_sector = (
            dram_read_bytes / l2_read_sectors if l2_read_sectors else 0.0
        )
        avg_bytes_per_write_sector = (
            dram_write_bytes / l2_write_sectors if l2_write_sectors else 0.0
        )
        read_eff = min(avg_bytes_per_read_sector / HBMMemoryAnalyzer.OPTIMAL_BURST, 1.0) * 100
        write_eff = (
            min(avg_bytes_per_write_sector / HBMMemoryAnalyzer.OPTIMAL_BURST, 1.0) * 100
        )
        return {
            "bandwidth_utilization_pct": bw_utilization,
            "total_throughput_gbps": total_throughput,
            "avg_bytes_per_read_sector": avg_bytes_per_read_sector,
            "avg_bytes_per_write_sector": avg_bytes_per_write_sector,
            "read_burst_efficiency_pct": read_eff,
            "write_burst_efficiency_pct": write_eff,
        }

    @staticmethod
    def print_hbm3e_best_practices() -> None:
        print("=" * 80)
        print("HBM3e Best Practices")
        print("=" * 80)
        print(" - Align data to 128-byte cache lines.")
        print(" - Aim for 256-byte bursts (8 x float4).")
        print(" - Use float4/int4 vectorization.")
        print(" - Employ cache streaming modifiers for write-only traffic.")


def run_complete_blackwell_analysis(nsys_report: str, ncu_report: str) -> None:
    """Print a combined Nsight Systems + Nsight Compute analysis."""
    print("=" * 80)
    print("Complete Blackwell Analysis")
    print("=" * 80)
    NsightSystemsProfiler.analyze_blackwell_metrics(nsys_report)
    NsightComputeProfiler.analyze_blackwell_kernel(ncu_report)
    BlackwellMetricsGuide.print_metric_guide()
    HBMMemoryAnalyzer.print_hbm3e_best_practices()


if __name__ == "__main__":
    print("=== Blackwell Profiling Guide ===")

    def _select_execution_device() -> torch.device:
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available - using CPU")
            return torch.device("cpu")
        try:
            torch.cuda.current_device()
            torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as exc:
            print(f"WARNING: Unable to acquire CUDA device ({exc}); falling back to CPU")
            return torch.device("cpu")

    exec_device = _select_execution_device()
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
