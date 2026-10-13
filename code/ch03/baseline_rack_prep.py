"""Rack-prep baseline: topology-unaware pageable staging.

Both rack_prep variants accept the same --target-extra-arg flags: --apply,
--reserve and --nic. The baseline is the topology-unaware control, so it never
binds CPUs or writes IRQ/RPS/XPS affinity. It records the requested values and
uses --nic only to choose which NICs its snapshot describes.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence
import os

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

from ch03.grace_blackwell_topology import NICInfo, discover_nics, format_cpulist

DEFAULT_RESERVE_CORES = 2


def build_rack_prep_override_parser() -> argparse.ArgumentParser:
    """Return the parser for the --target-extra-arg flags both rack_prep variants accept."""
    parser = argparse.ArgumentParser(prog="rack_prep", add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--apply", action="store_true", help="Apply IRQ/RPS/XPS affinity on setup (root required).")
    parser.add_argument(
        "--reserve",
        type=int,
        default=DEFAULT_RESERVE_CORES,
        help="Number of local CPUs to reserve for system tasks.",
    )
    parser.add_argument("--nic", action="append", default=None, help="NIC(s) to prefer (first found will be primary). Repeatable.")
    return parser


def parse_rack_prep_overrides(argv: Sequence[str]) -> argparse.Namespace:
    """Parse rack_prep --target-extra-arg values.

    An unknown argument, an invalid value, a negative --reserve or an empty
    --nic raises ValueError, so the harness fails the target.
    """
    parser = build_rack_prep_override_parser()
    try:
        args, unknown = parser.parse_known_args(list(argv))
    except argparse.ArgumentError as exc:
        raise ValueError(f"Invalid rack_prep override {list(argv)}: {exc}") from exc
    if unknown:
        raise ValueError(f"Unsupported rack_prep override {unknown}. Accepted: {parser.format_usage().strip()}")
    if args.reserve < 0:
        raise ValueError(f"--reserve must be zero or more CPUs, got {args.reserve}")
    args.nic = list(args.nic or [])
    if any(not name.strip() for name in args.nic):
        raise ValueError("--nic needs a NIC name")
    return args


def discover_requested_nics(names: Sequence[str]) -> List[NICInfo]:
    """Return NIC metadata. Every NIC named with --nic must exist."""
    nics = discover_nics(list(names) or None)
    missing = sorted(set(names) - {nic.name for nic in nics})
    if missing:
        raise RuntimeError(f"NIC(s) requested with --nic were not found under /sys/class/net: {', '.join(missing)}")
    return nics


class BaselineRackPrepBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Stages a NIC-style batch without locality hints or overlap."""

    def __init__(self):
        super().__init__()
        self.seq_len = 4096
        self.hidden_size = 4096
        self.host_batch: Optional[torch.Tensor] = None
        self.device_batch: Optional[torch.Tensor] = None
        self.norm: Optional[nn.Module] = None
        self.nic_snapshot: List[NICInfo] = []
        # Requested through --target-extra-arg. The baseline applies none of them.
        self.apply_affinity = False
        self.reserve_cores = DEFAULT_RESERVE_CORES
        self.preferred_nics: List[str] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        bytes_per_iter = self.seq_len * self.hidden_size * 4  # float32 bytes
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        self.nic_snapshot = discover_requested_nics(self.preferred_nics)
        self.host_batch = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32)
        self.device_batch = torch.empty_like(self.host_batch, device=self.device)
        self.norm = nn.LayerNorm(self.hidden_size, device=self.device)
        self._verify_output_buffer = torch.empty_like(self.device_batch)
        self._payload_parameter_count = sum(p.numel() for p in self.norm.parameters())
        
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.host_batch is not None and self.device_batch is not None and self.norm is not None
        with torch.inference_mode(), self._nvtx_range("baseline_rack_prep"):
            self.device_batch.copy_(self.host_batch, non_blocking=False)
            self.output = self.norm(self.device_batch)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"host_batch": self.host_batch, "device_batch": self.device_batch},
            output=self._verify_output_buffer,
            batch_size=self.host_batch.shape[0],
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
        self.host_batch = None
        self.device_batch = None
        self.norm = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.host_batch is None or self.norm is None:
            return "Host batch or model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        if not self.nic_snapshot:
            return None
        summaries = [f"{n.name}:numa={n.numa_node},cpus={format_cpulist(n.local_cpus)}" for n in self.nic_snapshot]
        return {"nic_layouts": len(self.nic_snapshot), "nic_summary": ";".join(summaries)}


    def apply_target_overrides(self, argv: List[str]) -> None:
        """Accept the rack_prep flags. Unknown or invalid flags raise ValueError."""
        args = parse_rack_prep_overrides(argv)
        self.apply_affinity = args.apply
        self.reserve_cores = args.reserve
        self.preferred_nics = args.nic


def get_benchmark() -> BaseBenchmark:
    return BaselineRackPrepBenchmark()
