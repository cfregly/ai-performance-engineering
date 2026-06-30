"""Rack-prep optimized: locality-aware staging with copy/compute overlap."""

from __future__ import annotations

import os
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)

from ch03.grace_blackwell_topology import (
    NICInfo,
    cpulist_to_mask,
    discover_nics,
    recommended_cpuset,
    render_affinity_block,
)


def _compute_topology(reserve: int = 2, nic_names: Optional[List[str]] = None) -> tuple[List[NICInfo], Optional[NICInfo], List[int], List[str]]:
    """Return NIC discovery + primary NIC + CPU set + rendered snippet."""
    nic_plan = discover_nics(nic_names)
    primary = nic_plan[0] if nic_plan else None
    target_cpus = recommended_cpuset(primary.local_cpus if primary else [], reserve=reserve)
    snippet = render_affinity_block(primary, target_cpus) if primary else []
    return nic_plan, primary, target_cpus, snippet


class OptimizedRackPrepBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Adds affinity planning plus pinned double-buffered staging."""

    def __init__(self):
        super().__init__()
        self.seq_len = 4096
        self.hidden_size = 4096
        self.reserve_cores = 2
        self.apply_affinity = False
        self.preferred_nics: List[str] = []
        self.host_buffers: List[torch.Tensor] = []
        self.device_buffers: List[torch.Tensor] = []
        self.norm: Optional[nn.Module] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.copy_events: List[torch.cuda.Event] = []
        self.cur_slot = 0
        self.next_slot = 1
        self.nic_plan: List[NICInfo] = []
        self.bound_cpus: List[int] = []
        self.affinity_snippet: List[str] = []
        self.apply_actions: List[str] = []
        self.verify_report: Optional[dict] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._last_slot: int = 0
        self._payload_parameter_count = 0
        bytes_per_iter = self.seq_len * self.hidden_size * 4  # float32 bytes (matches baseline)
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def _bind_local_cpus(self, cpus: List[int]) -> None:
        if not cpus:
            return
        try:
            os.sched_setaffinity(0, cpus)
        except (AttributeError, PermissionError, OSError):
            # No-op when affinities cannot be set in this environment.
            pass

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.nic_plan, primary_nic, target_cpus, snippet = _compute_topology(
            reserve=self.reserve_cores,
            nic_names=self.preferred_nics,
        )
        self.bound_cpus = target_cpus
        self.affinity_snippet = snippet
        self._bind_local_cpus(target_cpus)
        self.apply_actions = []
        self.verify_report = None

        if self.apply_affinity and primary_nic and target_cpus:
            if os.geteuid() != 0:
                self.apply_actions.append("SKIP apply: requires root")
            else:
                self.apply_actions.extend(_apply_affinity(primary_nic, target_cpus))
                self.verify_report = _verify_affinity(primary_nic, target_cpus)

        # Use pinned memory for efficient async H2D (the optimization)
        # Same dtype as baseline (float32) for fair verification comparison
        host_template = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32)
        self.host_buffers = [
            host_template.pin_memory(),
            host_template.clone().pin_memory(),
        ]
        self.device_buffers = [
            torch.empty_like(self.host_buffers[0], device=self.device),
            torch.empty_like(self.host_buffers[0], device=self.device),
        ]
        self._verify_output_buffer = torch.empty_like(self.device_buffers[0])
        self.norm = nn.LayerNorm(self.hidden_size, device=self.device, dtype=torch.float32)
        self._payload_parameter_count = sum(p.numel() for p in self.norm.parameters())
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_events = [torch.cuda.Event() for _ in self.device_buffers]
        self.cur_slot = 0
        self.next_slot = 1
        current_stream = torch.cuda.current_stream()
        self._start_copy(self.cur_slot, wait_stream=current_stream)
        current_stream.wait_event(self.copy_events[self.cur_slot])
        self._start_copy(self.next_slot, wait_stream=current_stream)
        

    def _start_copy(
        self,
        slot: int,
        wait_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if self.copy_stream is None:
            raise RuntimeError("Copy stream not initialized")
        if not self.copy_events:
            raise RuntimeError("Copy events not initialized")
        producer_stream = wait_stream or torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer_stream)
            self.device_buffers[slot].copy_(self.host_buffers[slot], non_blocking=True)
            self.copy_events[slot].record(self.copy_stream)

    def benchmark_fn(self) -> None:
        assert self.norm is not None
        if self.copy_stream is None:
            raise RuntimeError("Copy stream not initialized")
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(self.copy_events[self.cur_slot])
        with torch.inference_mode(), self._nvtx_range("optimized_rack_prep"):
            self.output = self.norm(self.device_buffers[self.cur_slot])
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._last_slot = self.cur_slot
        self._start_copy(self.cur_slot, wait_stream=current_stream)
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "host_batch": self.host_buffers[self._last_slot],
                "device_batch": self.device_buffers[self._last_slot],
            },
            output=self._verify_output_buffer,
            batch_size=self.host_buffers[self._last_slot].shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.host_buffers = []
        self.device_buffers = []
        self.norm = None
        self.output = None
        self._verify_output_buffer = None
        self.copy_stream = None
        self.copy_events = []
        self._last_slot = 0
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=10,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        return [self.copy_stream] if self.copy_stream is not None else []

    def validate_result(self) -> Optional[str]:
        if not self.host_buffers or self.norm is None:
            return "Buffers or model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def apply_target_overrides(self, argv: List[str]) -> None:
        """Handle aisp bench --target-extra-arg overrides."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--apply", action="store_true", help="Apply IRQ/RPS/XPS affinity on setup (root required).")
        parser.add_argument("--reserve", type=int, default=self.reserve_cores, help="Number of local CPUs to reserve for system tasks.")
        parser.add_argument("--nic", action="append", default=None, help="NIC(s) to prefer (first found will be primary). Repeatable.")
        try:
            opts, _ = parser.parse_known_args(argv)
        except SystemExit:
            return
        self.apply_affinity = bool(opts.apply)
        self.reserve_cores = max(0, int(opts.reserve))
        self.preferred_nics = [n for n in (opts.nic or []) if n]


def get_benchmark() -> BaseBenchmark:
    return OptimizedRackPrepBenchmark()


def _mask_no_prefix(cpus: List[int]) -> Optional[str]:
    mask = cpulist_to_mask(cpus)
    if not mask:
        return None
    return mask[2:] if mask.startswith("0x") else mask


def _apply_affinity(nic: NICInfo, cpus: List[int]) -> List[str]:
    """Write IRQ/RPS/XPS affinities for a NIC; requires root."""
    actions: List[str] = []
    mask = _mask_no_prefix(cpus)
    if not mask:
        return actions

    irq_base = Path("/proc/irq")
    for irq in nic.irq_ids:
        target = irq_base / str(irq) / "smp_affinity"
        try:
            target.write_text(f"{mask}\n")
            actions.append(f"IRQ {irq} -> {mask}")
        except OSError as exc:
            actions.append(f"IRQ {irq} -> FAILED ({exc})")

    queues_dir = Path(f"/sys/class/net/{nic.name}/queues")
    for rx in sorted(queues_dir.glob("rx-*")):
        try:
            (rx / "rps_cpus").write_text(f"{mask}\n")
            (rx / "rps_flow_cnt").write_text("32768\n")
            actions.append(f"{rx.name}/rps_cpus -> {mask}")
        except OSError as exc:
            actions.append(f"{rx.name}/rps_cpus -> FAILED ({exc})")
    for tx in sorted(queues_dir.glob("tx-*")):
        try:
            (tx / "xps_cpus").write_text(f"{mask}\n")
            actions.append(f"{tx.name}/xps_cpus -> {mask}")
        except OSError as exc:
            actions.append(f"{tx.name}/xps_cpus -> FAILED ({exc})")
    return actions


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _verify_affinity(nic: NICInfo, cpus: List[int]) -> dict:
    """Return a validation report for IRQs, queues, and PID affinity."""
    mask = _mask_no_prefix(cpus) or ""
    irq_base = Path("/proc/irq")
    irq_state = {
        irq: _read_text(irq_base / str(irq) / "smp_affinity_list")
        for irq in nic.irq_ids
    }
    queues_dir = Path(f"/sys/class/net/{nic.name}/queues")
    rx = {
        q.name: _read_text(q / "rps_cpus")
        for q in sorted(queues_dir.glob("rx-*"))
    }
    tx = {
        q.name: _read_text(q / "xps_cpus")
        for q in sorted(queues_dir.glob("tx-*"))
    }
    pid_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []

    numactl_out = ""
    if shutil.which("numactl"):
        try:
            out = subprocess.run(
                ["numactl", "-s", "-p", str(os.getpid())],
                check=False,
                capture_output=True,
                text=True,
            )
            numactl_out = out.stdout.strip() or out.stderr.strip()
        except OSError:
            pass

    return {
        "expected_mask": mask,
        "irq_affinity": irq_state,
        "rx_rps": rx,
        "tx_xps": tx,
        "pid_affinity": ",".join(map(str, pid_affinity)) if pid_affinity else "",
        "numactl": numactl_out,
    }
